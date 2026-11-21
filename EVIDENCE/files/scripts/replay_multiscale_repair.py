"""Same byte-preserving workload, actual causal history plus supplied legacy features."""
from pathlib import Path
import argparse,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from sbfe import iter_pcap,role_for
from wb_behavior import PriorEndpointRing,PRIOR_SYMMETRIC
from wb_profile import LocalSource,checked_file
from multiscale_context import MultiscaleRing,CONTEXT_NAMES,DIRECTION_COLUMNS,repair_frame
from wb_research import view_data
from run_wb_research import load_data
from run_context_repair import check_record
from profile_multiscale_repair import load_policy
from analyze_context_repair import evaluation,labels

def multiscale60_query(ring,stamp,src,dst):
    records,clock=ring._records_before(stamp)
    hosts=[ring._host_vector(records,clock,host) for host in [src,dst]]
    hosts.sort(key=ring._activity_key,reverse=True)
    # Directional tie-break cannot affect 60 features when both host vectors tie.
    return np.concatenate(hosts)

def run(run):
    run=run.resolve()
    cfg=json.loads((run/'configuration_freeze.json').read_text())
    freeze=json.loads((run/'profiling/inputs_manifest.json').read_text())
    assert json.loads((run/'profiling/complete.json').read_text())['status']=='PASS_REPAIR_FEATURE_INPUT_PROFILE'
    selected=json.loads((run/'stages/temporal_seed42/candidate_selection_before_V.json').read_text())['selected']['balanced']['wb_only']['view']
    if selected=='baseline25':raise ValueError('No changed G-selected model to replay')
    policies=[f'{view}__{mode}' for view in ['baseline25','baseline29'] for mode in ['shared','certified']]
    policies += [f'{selected}__{mode}' for mode in ['shared','certified','logistic_regression','random_forest']]
    old=ROOT/'runs/WB_RESEARCH_FULL_20261019_01';catalog={r['path']:r for r in json.loads((old/'artifacts_sha256.json').read_text())}
    rec=catalog['profiling/workload.json'];check_record(old,rec);work=json.loads((old/rec['path']).read_text())
    dest=run/'profiling/replay'
    if dest.exists():raise ValueError('Replay output already exists')
    dest.mkdir();(dest/'source').mkdir()
    shutil.copy2(Path(__file__),dest/'source'/Path(__file__).name)
    for item in work['files']:
        record=catalog[item['file']];check_record(old,record)
        assert record['sha256']==item['sha256'] and record['bytes']==item['bytes']
        shutil.copy2(old/item['file'],dest/Path(item['file']).name)
    raw,y,holdout,data=load_data(cfg)
    data_audit=json.loads((ROOT/cfg['data_folder']/'mapping_audit.json').read_text())
    for item in work['files']:
        total=next(r['packets'] for r in data_audit['sources'] if r['capture_id']==item['capture_id'])
        assert role_for(item['first_source_record'],total)==3 and role_for(item['first_source_record']-1,total)!=3
    ids=np.array([int(item['targets'][str(i)]['row_id']) for item in work['files'] for i in sorted(map(int,item['targets']))])
    assert not np.intersect1d(ids,holdout).size
    e.write_json(dest/'configuration_freeze.json',dict(policies=policies,repeats=3,decision_rows=len(ids),
        packet_rows=work['packet_rows'],workload_sha256=rec['sha256'],all_slices_start_at_role_boundary=True,
        current_packet_in_history=False,legacy_feature_extraction_included=False,source_sha256=e.sha256(Path(__file__)),
        limitation='160 early-role annotations; histories may be shorter than10s, not steady-state every-packet IDS throughput'))
    rows=[]
    with threadpool_limits(limits=1):
        for policy in policies:
            spec=freeze['policies'][policy];view=spec['view'];predictor,_,_=load_policy(run,freeze,policy)
            X,_,_=(view_data(raw,data,'legacy25_symmetric' if view=='baseline25' else 'legacy29_symmetric')
                       if view.startswith('baseline') else repair_frame(raw,data,view))
            history=list(PRIOR_SYMMETRIC) if view.startswith('baseline') else list(CONTEXT_NAMES)+(list(DIRECTION_COLUMNS) if view.endswith('directional96') else [])
            base=X.drop(columns=history);ev,sel=evaluation(run,cfg,'temporal_seed42',view,'balanced')
            lookup={int(r):i for i,r in enumerate(ev['row_ids'])};wanted=labels(ev,sel,'weighted_voting' if spec['mode'] in ['shared','certified'] else spec['mode'])[[lookup[int(r)] for r in ids]]
            for repeat in range(3):
                start=time.perf_counter();seen=0;pred=[];got_ids=[];update=query=model=0.;capacity=0
                for item in work['files']:
                    ring=PriorEndpointRing() if view.startswith('baseline') else MultiscaleRing()
                    for index,stamp,_,_,packet in iter_pcap(LocalSource(),dest/Path(item['file']).name):
                        seen+=1;target=item['targets'].get(str(index))
                        if target is not None:
                            rid=int(target['row_id']);tick=time.perf_counter()
                            if view.startswith('baseline'):context=ring.query_prior(stamp,packet.src,packet.dst)['prior_symmetric']
                            elif view.endswith('directional96'):
                                _,host,direction=ring.query_multiscale(stamp,packet.src,packet.dst);context=np.r_[host,direction]
                            else:context=multiscale60_query(ring,stamp,packet.src,packet.dst)
                            query+=time.perf_counter()-tick
                            frame=pd.concat([base.loc[[rid]],pd.DataFrame(context[None],index=[rid],columns=history)],axis=1)
                            if repeat==0:pd.testing.assert_frame_equal(frame,X.loc[[rid]],check_dtype=False,atol=1e-12,rtol=0)
                            tick=time.perf_counter();pred.append(int(predictor.predict(frame)[0]));model+=time.perf_counter()-tick;got_ids.append(rid)
                        tick=time.perf_counter();ring.push(stamp,packet);update+=time.perf_counter()-tick
                    capacity+=ring.evicted_capacity
                elapsed=time.perf_counter()-start
                np.testing.assert_array_equal(got_ids,ids);np.testing.assert_array_equal(pred,wanted);assert seen==work['packet_rows']
                rows.append(dict(policy=policy,repeat=repeat,total_seconds=elapsed,update_seconds=update,query_seconds=query,
                    model_seconds=model,other_seconds=elapsed-update-query-model,decisions=len(ids),packets=seen,labels_match=True,
                    capacity_evictions=capacity,ring_allocated_bytes=ring.data.nbytes,feature_verification_in_timing=repeat==0))
            print('REPLAY',policy,flush=True)
    table=pd.DataFrame(rows);table.to_csv(dest/'replay_raw.csv',index=False)
    # First pass includes equality checks, so do not use it as a timing observation.
    summary=table[~table.feature_verification_in_timing].groupby('policy')[['total_seconds','update_seconds','query_seconds','model_seconds']].mean().reset_index()
    summary.to_csv(dest/'replay_summary.csv',index=False)
    e.write_json(dest/'complete.json',dict(status='PASS_MULTISCALE_REPLAY',policies=len(policies),replays=len(rows),
        timed_repeats_per_policy=2,verification_pass_excluded=True,all_labels_match=True,all_contexts_match=True,
        source_sha256=e.sha256(Path(__file__)),legacy_extraction_included=False))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
