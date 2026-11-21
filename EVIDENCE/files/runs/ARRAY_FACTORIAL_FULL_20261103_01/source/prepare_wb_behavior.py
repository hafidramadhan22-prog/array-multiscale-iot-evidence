"""Replay verified PCAP sources for prior-only, symmetric endpoint features."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import ta_hafid_engine as e
from e4 import verified_historical_file
from sbfe import iter_pcap, role_for
from wb_behavior import PriorEndpointRing, feature_schema


def verify_catalog(folder):
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if e.sha256(folder / "catalog.json") != complete["catalog_sha256"]:
        raise ValueError("SBFE source catalog checksum mismatch")
    rows = json.loads((folder / "catalog.json").read_text(encoding="utf-8"))
    for row in rows:
        file = folder / row["path"]
        if file.stat().st_size != row["bytes"] or e.sha256(file) != row["sha256"]:
            raise ValueError(f"SBFE source changed: {row['path']}")
    return rows


def extract(source, source_mapping, archive_path, cfg, output, log):
    n = len(source["row_ids"])
    added = {key: np.full((n, len(names)), np.nan) for key, names in feature_schema().items()}
    hits = np.zeros(n, dtype=np.int8)
    records_audit = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        for old in source_mapping["sources"]:
            capid = old["capture_id"]
            positions = np.flatnonzero(source["capture"] == capid)
            if not len(positions):
                continue
            wanted = {int(source["record"][pos]): int(pos) for pos in positions}
            if len(wanted) != len(positions):
                raise ValueError("Two target rows refer to the same chosen physical record")
            entry = old["entry"]
            total = old["packet_count"]
            digest = hashlib.sha256()
            ring = PriorEndpointRing(cfg["ring_capacity"], cfg["windows_seconds"])
            last_role = None
            role_start = None
            evicted_capacity = evicted_time = backwards = seen = 0
            counters = collections.Counter()
            started = time.perf_counter()
            log(f"Replay capture={capid} {Path(entry).name}, {total} packets, {len(positions)} targets")
            for index, timestamp, _seconds, _nanoseconds, packet in iter_pcap(archive, infos[entry], digest):
                current_role = role_for(index, total)
                if current_role != last_role:
                    evicted_capacity += ring.evicted_capacity
                    evicted_time += ring.evicted_time
                    backwards += ring.backward
                    ring.reset()
                    last_role = current_role
                    role_start = timestamp
                pos = wanted.get(index)
                if pos is not None:
                    if int(source["role"][pos]) != current_role:
                        raise ValueError("Inherited target role changed")
                    if not np.array_equal(packet.header(), source["base"][pos]):
                        raise ValueError("Decoded target header differs from verified source")
                    # The current observation identifies endpoints but is not in history.
                    features = ring.query_prior(timestamp, packet.src, packet.dst)
                    for key, values in features.items():
                        added[key][pos] = values
                ring.push(timestamp, packet)
                seen += 1
                if pos is not None:
                    if ring.clock != source["logical_ns"][pos]:
                        raise ValueError("Inherited causal clock changed")
                    if source["valid_after_purge"][pos] and ring.clock - role_start < int(cfg["purge_seconds"] * 1e9):
                        raise ValueError("Inherited purge acceptance fails replay")
                    # This independent reproduction also checks chosen alias context.
                    if not np.allclose(ring.query(packet.dst), source["behavior"][pos], rtol=0, atol=1e-12):
                        raise ValueError("Original destination context no longer reproduces")
                    hits[pos] += 1
                    counters[current_role] += 1
            evicted_capacity += ring.evicted_capacity
            evicted_time += ring.evicted_time
            backwards += ring.backward
            if seen != total or digest.hexdigest() != old["sha256"]:
                raise ValueError("Selected source full PCAP SHA256/record-count mismatch")
            row = dict(capture_id=capid, entry=entry, packets=seen, targets=len(positions),
                       target_counts_by_role=dict(counters), sha256=digest.hexdigest(),
                       capacity_evictions=evicted_capacity, time_evictions=evicted_time,
                       backward_records=backwards, elapsed_seconds=time.perf_counter() - started)
            records_audit.append(row)
            log(f"Verified capture={capid} in {row['elapsed_seconds']:.2f}s")
    if not np.all(hits == 1) or not all(np.isfinite(value).all() for value in added.values()):
        raise ValueError("Every development target must have exactly one finite feature vector")
    result = {**source, **added}
    np.savez_compressed(output / "packet_features.npz", **result)
    return records_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/WB_BEHAVIOR_DATA_20261019_01")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Use a new output directory inside the project")
    started = time.perf_counter()
    source_folder = ROOT / "runs/SBFE_DATA_20261019_03"
    verified = verify_catalog(source_folder)
    cfg = json.loads((source_folder / "configuration_freeze.json").read_text(encoding="utf-8"))
    archive_path = Path(cfg["zip_path"])
    if e.sha256(archive_path) != cfg["zip_sha256"]:
        raise ValueError("Raw ZIP SHA256 mismatch")
    if e.sha256(ROOT / "ML-EdgeIIoT-dataset.csv") != cfg["canonical_sha256"]:
        raise ValueError("Canonical CSV SHA256 mismatch")
    with np.load(source_folder / "packet_features.npz", allow_pickle=False) as z:
        source = {key: z[key].copy() for key in z.files}
    with np.load(source_folder / "target_indices.npz", allow_pickle=False) as z:
        indices = {key: z[key].copy() for key in z.files}
    split_file = verified_historical_file(ROOT / "runs/FULL_20261013_01", "canonical_split_indices.npz")
    with np.load(split_file, allow_pickle=False) as z:
        if not np.array_equal(source["row_ids"], z["training_row_ids"]):
            raise ValueError("Source target order differs from historical development split")
        if not np.array_equal(indices["historical_holdout_ids"], z["holdout_row_ids"]):
            raise ValueError("Historical holdout identity mismatch")
    if not np.array_equal(source["row_ids"], indices["row_ids"]):
        raise ValueError("Features and labels are misaligned")
    if len(source["row_ids"]) != 52684 or np.intersect1d(source["row_ids"], indices["historical_holdout_ids"]).size:
        raise ValueError("Development-only target gate failed")
    source_mapping = json.loads((source_folder / "mapping_audit.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    log_lines = []

    def log(message):
        print(message, flush=True)
        log_lines.append(message)
        (output / "execution.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    try:
        configuration = dict(
            experiment="WB_PRIOR_SYMMETRIC_HOST_CONTEXT",
            source_data=str(source_folder.relative_to(ROOT)),
            source_catalog_sha256=e.sha256(source_folder / "catalog.json"),
            zip_path=str(archive_path), zip_sha256=cfg["zip_sha256"],
            canonical_sha256=cfg["canonical_sha256"], ring_capacity=cfg["ring_capacity"],
            windows_seconds=cfg["windows_seconds"], purge_seconds=cfg["purge_seconds"],
            current_packet_in_context=False, query_before_push=True,
            endpoint_order="Descending longest-window activity vector then full vector; never endpoint ID",
            alias_policy="Reuse previously selected SBFE source record; no reselection from new features or labels",
            target_roles_purge_identical_to_sbfe=True, historical_holdout_targets_used=False,
            raw_unlabeled_context_may_include_historical_holdout_packets=True,
            source_capture_independent_evaluation=False, new_training=False,
            novelty_status="Representation ablation, no claim of new host statistics",
        )
        e.write_json(output / "configuration_freeze.json", configuration)
        e.write_json(output / "feature_schema.json", feature_schema())
        e.write_json(output / "source_catalog_verified.json", verified)
        snapshots = output / "source"
        snapshots.mkdir()
        for rel in ["src/wb_behavior.py", "src/sbfe.py", "scripts/prepare_wb_behavior.py",
                    "tests/test_wb_behavior.py", "docs/WB_BEHAVIOR_FEATURES.md"]:
            shutil.copy2(ROOT / rel, snapshots / Path(rel).name)
        log("Input hashes, source catalog, development row order and holdout exclusion verified")
        audits = extract(source, source_mapping, archive_path, cfg, output, log)
        shutil.copy2(source_folder / "target_indices.npz", output / "target_indices.npz")
        shutil.copy2(source_folder / "role_support.json", output / "role_support.json")
        e.write_json(output / "environment.json", e.environment())
        e.write_json(output / "mapping_audit.json", dict(
            rows=len(source["row_ids"]), target_roles_purge_preserved=True,
            exact_old_header_and_destination_context_reproduced=True,
            source_record_selection_preserved=True,
            original_alias_count=source_mapping["duplicate_alias_count"],
            alternative_alias_new_context_not_recomputed=True,
            all_selected_source_hashes_verified=True, sources=audits,
            buffer_allocated_bytes=cfg["ring_capacity"] * 8 * 8,
            packets_replayed=sum(row["packets"] for row in audits),
            prior_context_only=True, metadata_not_predictors=True,
        ))
        elapsed = time.perf_counter() - started
        log(f"Completed {len(source['row_ids'])} targets, elapsed {elapsed:.2f}s; no training")
        e.write_json(output / "catalog.json", [dict(
            path=str(path.relative_to(output)).replace("\\", "/"),
            bytes=path.stat().st_size, sha256=e.sha256(path))
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "catalog.json"])
        e.write_json(output / "complete.json", dict(
            status="WB_BEHAVIOR_DATA_COMPLETE", rows=len(source["row_ids"]),
            valid_targets=int(source["valid_after_purge"].sum()),
            elapsed_seconds=elapsed, new_training=False,
            catalog_sha256=e.sha256(output / "catalog.json"),
        ))
    except Exception:
        log(traceback.format_exc())
        e.write_json(output / "failed.json", dict(status="FAILED_BEFORE_MODEL_TRAINING"))
        raise


if __name__ == "__main__":
    main()
