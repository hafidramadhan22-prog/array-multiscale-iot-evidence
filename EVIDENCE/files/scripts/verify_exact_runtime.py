"""Audit raw evidence, aggregates, provenance and unchanged runtime sources; no models loaded."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from run_time_repair import check_record


def verify(run):
    cfg=json.loads((run/'configuration_freeze.json').read_text());freeze=json.loads((run/'inputs_manifest.json').read_text())
    done=json.loads((run/'complete.json').read_text());eq=json.loads((run/'equivalence.json').read_text())
    assert done['status']=='PASS_EXACT_RUNTIME' and done['new_classifier_fits']==0
    assert done['inputs_manifest_sha256']==e.sha256(run/'inputs_manifest.json')
    assert done['source_sha256']==freeze['source_sha256']==e.sha256(ROOT/'scripts/profile_exact_runtime.py')
    for rec in cfg['source_manifest']:assert e.sha256(ROOT/rec['path'])==rec['sha256']
    for rec in freeze['files']:check_record(run,rec)
    for spec in freeze['policies'].values():check_record(ROOT/spec['origin_folder'],spec['origin_model'])
    assert eq['status']=='PASS_EXACT_RUNTIME_INPUTS' and all(r['rows']==52684 and r['expert_probability_max_abs_delta']<=1e-12 and r['full_and_certified_labels_identical'] for r in eq['checks'])
    latency=pd.read_csv(run/'latency_raw.csv');replay=pd.read_csv(run/'replay_raw.csv');mem=pd.read_csv(run/'memory_raw.csv')
    policies=set(freeze['policies']);assert set(latency.policy)==set(replay.policy)==set(mem.policy)==policies
    assert len(latency)==len(policies)*2*45==done['timings']
    assert (latency.seconds>0).all() and np.isfinite(latency.seconds).all()
    assert (latency.groupby(['policy','batch']).size()==45).all()
    summary=latency.groupby(['policy','batch']).seconds.agg(['mean','median','std','count']).reset_index()
    pd.testing.assert_frame_equal(summary,pd.read_csv(run/'latency_summary.csv'),check_dtype=False,atol=1e-12,rtol=1e-12)
    assert len(replay)==len(policies)*5==done['randomized_replays'] and (replay.groupby('policy').size()==5).all()
    assert replay.labels_match.all() and (replay.decisions==160).all() and (replay.packets==15665).all()
    assert (replay.ring_allocated_bytes==1048576).all() and (replay.total_seconds>0).all()
    assert len(mem)==len(policies)*2*3==done['cold_memory_runs']
    assert (mem.groupby(['policy','workload']).size()==3).all() and mem.labels_match.all()
    assert (mem.delta_peak_inference_mib>=0).all() and (mem.samples>=2).all()
    np.testing.assert_allclose(mem.peak_rss_mib-mem.baseline_rss_mib,mem.delta_peak_inference_mib,rtol=0,atol=1e-12)
    assert mem.query("workload == 'replay'").ring_included.all()
    memory_summary=mem.groupby(['policy','workload'])[['baseline_rss_mib','peak_rss_mib','delta_peak_inference_mib']].mean().reset_index()
    pd.testing.assert_frame_equal(memory_summary,pd.read_csv(run/'memory_summary.csv'),check_dtype=False,atol=1e-12,rtol=1e-12)
    work=json.loads((run/'replay/workload.json').read_text());quality=ROOT/cfg['quality_run']
    qc=json.loads((quality/'configuration_freeze.json').read_text());audit=json.loads((ROOT/qc['data_folder']/'mapping_audit.json').read_text())
    from sbfe import role_for
    for item in work['files']:
        total=next(r['packets'] for r in audit['sources'] if r['capture_id']==item['capture_id'])
        assert role_for(item['first_source_record'],total)==3 and role_for(item['first_source_record']-1,total)!=3
    xai=json.loads((run/'xai_equivalence.json').read_text());assert xai['status']=='PASS_COMPACT_XAI_GAMES' and xai['games']==300
    assert xai['max_probability_difference']<=1e-12 and xai['max_shap_difference']<=1e-12
    result=dict(status='PASS_RUNTIME_EVIDENCE_RECONSTRUCTION',policies=len(policies),raw_timings=len(latency),
        raw_replays=len(replay),cold_rss=len(mem),full_input_equivalence_rows=52684,coalition_games=xai['games'],
        all_sources_and_artifact_hashes_match=True,same_role_boundary_workload=True,new_model_loads=0,new_classifier_fits=0,
        zero_delta_rss_observations=int((mem.delta_peak_inference_mib==0).sum()),
        independent_capture_evaluation=False)
    e.write_json(run/'verification_independent.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run.resolve())
