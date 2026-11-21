"""Verified inputs and actual inference workloads for the WB development study."""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

import ta_hafid_engine as e
from sbfe import iter_pcap
from wb_behavior import PriorEndpointRing, PRIOR_DESTINATION, PRIOR_SYMMETRIC
from wb_research import WBPredictor

VIEWS = ['legacy29', 'legacy29_destination', 'legacy29_symmetric', 'legacy25_symmetric']
POLICIES = {f'{view}__{mode}': dict(view=view, mode=mode)
            for view in VIEWS for mode in ['shared', 'certified']}
POLICIES.update({f'{view}__independent': dict(view=view, mode='independent')
                 for view in ['legacy29', 'legacy29_symmetric']})
POLICIES.update({f'legacy29_symmetric__{name}': dict(view='legacy29_symmetric', mode=name)
                 for name in ['logistic_regression', 'random_forest']})


def checked_file(root, record):
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or path.stat().st_size != record['bytes'] or e.sha256(path) != record['sha256']):
        raise ValueError('File provenance/checksum mismatch: ' + str(record.get('path')))
    return path


def file_record(root, path):
    return dict(path=str(Path(path).relative_to(root)).replace('\\', '/'),
                bytes=Path(path).stat().st_size, sha256=e.sha256(path))


def quality_manifest(run):
    done = json.loads((run / 'quality_complete.json').read_text(encoding='utf-8'))
    if done['status'] != 'WB_QUALITY_COMPLETE':
        raise ValueError('Completed WB quality phase required before profiling')
    path = run / 'models_manifest.json'
    if e.sha256(path) != done['models_manifest_sha256']:
        raise ValueError('WB model manifest checksum mismatch')
    return json.loads(path.read_text(encoding='utf-8'))


def owned_bundle(run, view):
    matches = [r for r in quality_manifest(run) if r['stage'] == 'temporal_seed42'
               and r['view'] == view and r['case'] == 'balanced']
    if len(matches) != 1:
        raise ValueError('Exactly one matching run-owned balanced bundle required')
    record = matches[0]
    bundle = joblib.load(checked_file(run, record))
    if bundle['view'] != view or bundle['case'] != 'balanced':
        raise ValueError('Bundle identity differs from manifest')
    if set(bundle['models']) != set(e.NAMES):
        raise ValueError('Expected the existing five base-model families')
    prep = bundle['models']['decision_tree']['preprocessor']
    if any(bundle['models'][name]['preprocessor'] is not prep
           for name in e.NAMES if name != 'logistic_regression'):
        raise ValueError('Shared preprocessing requires the same fitted transformer object')
    return bundle, record


def read_frame(path, schema):
    frame = pd.read_csv(path, dtype={c: str for c in schema['categorical']},
                        keep_default_na=False, float_precision='round_trip', index_col='row_id')
    for column in schema['numeric']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    if list(frame.columns) != schema['columns'] or not frame.index.is_unique:
        raise ValueError('Profiling feature schema or row uniqueness mismatch')
    return frame


def load_inputs(run, policy):
    if policy not in POLICIES:
        raise ValueError('Unknown profiling policy')
    manifest = json.loads((run / 'profiling/inputs_manifest.json').read_text(encoding='utf-8'))
    rows = {r['path']: r for r in manifest['files']}
    spec = POLICIES[policy]
    view = spec['view']
    schema = manifest['schemas'][view]
    files = {}
    for batch in ['batch1', 'batch1000']:
        rel = f'profiling/inputs/{view}_{batch}.csv'
        files[batch] = read_frame(checked_file(run, rows[rel]), schema)
    base = 'legacy25' if view.startswith('legacy25') else 'legacy29'
    rel = f'profiling/inputs/{base}_replay.csv'
    files['replay_base'] = read_frame(checked_file(run, rows[rel]), manifest['schemas'][base])
    rel = 'profiling/inputs/expected.npz'
    with np.load(checked_file(run, rows[rel]), allow_pickle=False) as z:
        expected = {key: z[key].copy() for key in z.files if key.startswith(policy + '__')}
        expected['replay_row_ids'] = z['replay_row_ids'].copy()
        expected['batch1_row_ids'] = z['batch1_row_ids'].copy()
        expected['batch1000_row_ids'] = z['batch1000_row_ids'].copy()
        behavior = {key: z[key].copy() for key in ['prior_destination', 'prior_symmetric']}
    rel = 'profiling/workload.json'
    evidence = json.loads(checked_file(run, rows[rel]).read_text(encoding='utf-8'))
    for item in evidence['files']:
        checked_file(run, dict(path=item['file'], bytes=item['bytes'], sha256=item['sha256']))
    files.update(expected=expected, behavior=behavior, evidence=evidence,
                 ring_capacity=manifest['ring_capacity'], windows=manifest['windows_seconds'])
    return files


def load_predictor(run, policy):
    spec = POLICIES[policy]
    if spec['mode'] in e.NAMES:
        quality_manifest(run)
        freeze = json.loads((run / 'profiling/controls_manifest.json').read_text(encoding='utf-8'))
        matches = [r for r in freeze if r['policy'] == policy]
        if len(matches) != 1:
            raise ValueError('Derived standalone control is missing')
        record = matches[0]
        # Source bundle remains tied to the authoritative completed run manifest.
        source = [r for r in quality_manifest(run) if r['sha256'] == record['source_bundle_sha256']]
        if len(source) != 1:
            raise ValueError('Derived model has unknown source bundle')
        predictor = joblib.load(checked_file(run, record))
    else:
        bundle, record = owned_bundle(run, spec['view'])
        predictor = WBPredictor(bundle['models'], bundle['weights'], bundle['order'], spec['mode'])
    return predictor, record['bytes']


class LocalSource:
    def open(self, path):
        return Path(path).open('rb')


def check_labels(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError('Runtime labels differ from frozen validation predictions')


def replay(run, policy, predictor, inputs, *, verify_features=False):
    """Actual raw-byte parsing plus prior context and model inference.

    Legacy 25/29 domain features are precomputed from canonical CSV. Their full
    extraction from packets is excluded; this is not a complete live IDS stack.
    """
    spec = POLICIES[policy]
    view = spec['view']
    kind = 'prior_destination' if view.endswith('_destination') else 'prior_symmetric'
    names = PRIOR_DESTINATION if kind == 'prior_destination' else PRIOR_SYMMETRIC
    use_context = view.endswith(('_destination', '_symmetric'))
    base = inputs['replay_base']
    expected_positions = {int(r): i for i, r in enumerate(inputs['expected']['replay_row_ids'])}
    evidence = inputs['evidence']
    start = time.perf_counter()
    ring = PriorEndpointRing(inputs['ring_capacity'], inputs['windows']) if use_context else None
    packet_count = 0
    row_ids = []
    labels = []
    update_seconds = query_seconds = model_seconds = 0.
    capacity_evictions = 0
    for item in evidence['files']:
        if ring is not None:
            ring.reset()
        for index, timestamp, _second, _nano, packet in iter_pcap(LocalSource(), run / item['file']):
            packet_count += 1
            target = item['targets'].get(str(index))
            if target is not None:
                rid = int(target['row_id'])
                frame = base.loc[[rid]].copy()
                if ring is not None:
                    tick = time.perf_counter()
                    context = ring.query_prior(timestamp, packet.src, packet.dst)[kind]
                    query_seconds += time.perf_counter() - tick
                    if verify_features:
                        expected = inputs['behavior'][kind][expected_positions[rid]]
                        if not np.allclose(context, expected, rtol=0, atol=1e-12):
                            raise ValueError('Prior-only replay context differs from frozen training representation')
                    frame = pd.concat([frame, pd.DataFrame(context[None, :], index=frame.index, columns=names)], axis=1)
                tick = time.perf_counter()
                prediction = predictor.predict(frame)
                model_seconds += time.perf_counter() - tick
                labels.append(int(prediction[0]))
                row_ids.append(rid)
            # The current packet must not enter its own prior-only statistics.
            if ring is not None:
                tick = time.perf_counter()
                ring.push(timestamp, packet)
                update_seconds += time.perf_counter() - tick
        if ring is not None:
            capacity_evictions += ring.evicted_capacity
    elapsed = time.perf_counter() - start
    if packet_count != evidence['packet_rows'] or len(row_ids) != evidence['decision_rows']:
        raise ValueError('Replay workload count mismatch')
    if not np.array_equal(row_ids, inputs['expected']['replay_row_ids']):
        raise ValueError('Replay decision row order changed')
    return dict(policy=policy, total_seconds=elapsed, update_seconds=update_seconds,
                query_seconds=query_seconds, model_seconds=model_seconds,
                other_seconds=elapsed-update_seconds-query_seconds-model_seconds,
                raw_packet_rows=packet_count, decision_rows=len(row_ids),
                ring_allocated_bytes=0 if ring is None else ring.data.nbytes,
                capacity_evictions=capacity_evictions, labels=np.asarray(labels),
                row_ids=np.asarray(row_ids), prior_features_checked=bool(verify_features))


def measure_once(run, policy, workload, predictor, inputs):
    if workload == 'replay':
        result = replay(run, policy, predictor, inputs)
        check_labels(result['labels'], inputs['expected'][policy + '__replay'])
        return result
    if workload not in ['batch1', 'batch1000']:
        raise ValueError('Unsupported workload')
    start = time.perf_counter()
    labels = predictor.predict(inputs[workload])
    elapsed = time.perf_counter() - start
    check_labels(labels, inputs['expected'][policy + '__' + workload])
    return dict(policy=policy, total_seconds=elapsed, decision_rows=len(labels), labels=labels)
