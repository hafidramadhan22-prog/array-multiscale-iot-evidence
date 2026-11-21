"""Audit a G-selected owned WB with exact donor games, without refitting."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_wb_research import load_data
from run_context_repair import check_record,read_npz
from multiscale_context import MultiscaleRing,repair_frame
from sbfe import iter_pcap,role_for
from wb_research import cube,mixture,bijective_ports
from xai_audit import semantic_groups,exact_shapley,attribution_distance,METHODS
from run_xai_audit import group_evaluate,save_group,metric_rows,model_output_indices,verify_catalog
from run_xai_background_extension import changed_evaluate,robust_rows

SOURCE=['scripts/run_multiscale_xai.py','src/multiscale_context.py','src/xai_audit.py',
 'scripts/run_xai_audit.py','scripts/run_xai_background_extension.py','docs/XAI_MULTISCALE_PROTOCOL_20261020.md']
INDEX_SOURCE='runs/XAI_BACKGROUND_FULL_20261020_01'
DATA_SOURCE='runs/MULTISCALE_DATA_20261020_01'

def groups_for(columns):
    history=[c for c in columns if c.startswith(('multiscale_','directional_'))]
    rest=[c for c in columns if c not in history]
    groups=semantic_groups(rest) if rest else {};groups['endpoint_history']=history
    assert history and len(set(c for group in groups.values() for c in group))==len(columns)
    return groups

def prepare_jitter(output):
    if output.exists() or output.parent.resolve()!=(ROOT/'runs').resolve():raise ValueError('New run required')
    raw,y,holdout,data=load_data(dict(data_folder=DATA_SOURCE))
    verify_catalog(ROOT/INDEX_SOURCE);idx=read_npz(ROOT/INDEX_SOURCE/'indices.npz')
    pos=raw.index.get_indexer(idx['row_ids']);assert (pos>=0).all() and not np.intersect1d(idx['row_ids'],holdout).size
    config=json.loads((ROOT/DATA_SOURCE/'configuration_freeze.json').read_text());assert e.sha256(Path(config['zip_path']))==config['zip_sha256']
    audit=json.loads((ROOT/DATA_SOURCE/'mapping_audit.json').read_text());wanted={}
    for i,p in enumerate(pos):wanted.setdefault(int(data['capture'][p]),{})[int(data['record'][p])]=i
    context=np.full((len(pos),60),np.nan);direct=np.full((len(pos),36),np.nan);hits=np.zeros(len(pos),dtype=int);sources=[]
    output.mkdir();(output/'source').mkdir()
    for rel in SOURCE:shutil.copy2(ROOT/rel,output/'source'/Path(rel).name)
    with zipfile.ZipFile(config['zip_path']) as archive:
        for old in audit['sources']:
            cap=old['capture_id']
            if cap not in wanted:continue
            ring=MultiscaleRing();previous=None;digest=hashlib.sha256();seen=0
            for record,stamp,_,_,packet in iter_pcap(archive,archive.getinfo(old['entry']),digest):
                role=role_for(record,old['packets'])
                if role!=previous:ring.reset();previous=role
                if record in wanted[cap]:
                    i=wanted[cap][record];p=pos[i];assert role==3
                    _,base,base_dir=ring.query_multiscale(stamp,packet.src,packet.dst)
                    np.testing.assert_array_equal(base,data['prior_multiscale'][p]);np.testing.assert_array_equal(base_dir,data['prior_directional'][p])
                    _,context[i],direct[i]=ring.query_multiscale(stamp+5_000_000,packet.src,packet.dst);hits[i]+=1
                ring.push(stamp,packet);seen+=1
            assert seen==old['packets'] and digest.hexdigest()==old['sha256']
            sources.append(old);print('JITTER VERIFIED',cap,seen,flush=True)
    assert (hits==1).all() and np.isfinite(context).all() and np.isfinite(direct).all()
    np.savez_compressed(output/'jitter.npz',row_ids=idx['row_ids'],prior_multiscale=context,prior_directional=direct)
    e.write_json(output/'audit.json',dict(source_data_catalog_sha256=e.sha256(ROOT/DATA_SOURCE/'catalog.json'),
        source_indices_sha256=e.sha256(ROOT/INDEX_SOURCE/'indices.npz'),sources=sources,rows=len(pos),query_delay_ms=5,
        old_context_reproduced_exactly=True,query_before_push=True,no_future_arrivals=True))
    e.write_json(output/'complete.json',dict(status='MULTISCALE_JITTER_VERIFIED',file_sha256=e.sha256(output/'jitter.npz'),
        audit_sha256=e.sha256(output/'audit.json')))

def run(output,quality_run,jitter_run):
    output=output.resolve();quality_run=quality_run.resolve();jitter_run=jitter_run.resolve()
    if output.exists() or output.parent.resolve()!=(ROOT/'runs').resolve():raise ValueError('New run required')
    cfg=json.loads((quality_run/'configuration_freeze.json').read_text())
    assert json.loads((quality_run/'verification_quality.json').read_text())['status']=='PASS_CONTEXT_REPAIR_QUALITY'
    selection=json.loads((quality_run/'stages/temporal_seed42/candidate_selection_before_V.json').read_text())['selected']['balanced']['wb_only']
    view=selection['view']
    if view=='baseline25':raise ValueError('G selected unchanged baseline; use old XAI with that identity explicitly')
    manifest=json.loads((quality_run/'models_manifest.json').read_text())
    matches=[r for r in manifest if (r['stage'],r['view'],r['case'])==('temporal_seed42',view,'balanced')]
    assert len(matches)==1;record=matches[0];check_record(quality_run,record)
    verify_catalog(ROOT/INDEX_SOURCE);idx=read_npz(ROOT/INDEX_SOURCE/'indices.npz')
    raw,y,holdout,data=load_data(cfg);pos=raw.index.get_indexer(idx['row_ids']);ap=raw.index.get_indexer(idx['background_A']);bp=raw.index.get_indexer(idx['background_B'])
    assert (pos>=0).all() and (ap>=0).all() and (bp>=0).all()
    split=read_npz(quality_run/'stages/temporal_seed42/split.npz')
    assert set(idx['row_ids']).issubset(split['V']) and set(np.r_[idx['background_A'],idx['background_B']]).issubset(split['A'])
    assert not set(idx['background_A'])&set(idx['background_B']) and not set(idx['row_ids'])&set(holdout)
    done=json.loads((jitter_run/'complete.json').read_text());assert done['file_sha256']==e.sha256(jitter_run/'jitter.npz')
    jitter=read_npz(jitter_run/'jitter.npz');np.testing.assert_array_equal(jitter['row_ids'],idx['row_ids'])
    ja=json.loads((jitter_run/'audit.json').read_text());assert ja['source_data_catalog_sha256']==e.sha256(ROOT/cfg['data_folder']/'catalog.json')
    output.mkdir();(output/'source').mkdir();(output/'tables').mkdir()
    for rel in SOURCE:shutil.copy2(ROOT/rel,output/'source'/Path(rel).name)
    e.write_json(output/'configuration_freeze.json',dict(quality_run=str(quality_run.relative_to(ROOT)),view=view,
        selection_role='G',owned_model=record,background_A=32,background_B=32,rows=100,new_classifier_fits=0,
        indices_sha256=e.sha256(ROOT/INDEX_SOURCE/'indices.npz'),jitter_sha256=e.sha256(jitter_run/'jitter.npz')))
    np.savez_compressed(output/'indices.npz',**idx)
    check_record(quality_run,record);bundle=joblib.load(quality_run/record['path'])
    assert bundle['view']==view and bundle['case']=='balanced'
    X,_,_=repair_frame(raw,data,view);frame=X.iloc[pos];groups=groups_for(frame.columns)
    shifted,_,_=repair_frame(bijective_ports(raw),data,view);pd.testing.assert_frame_equal(frame,shifted.iloc[pos])
    modified_data={**data,'prior_multiscale':data['prior_multiscale'].copy(),'prior_directional':data['prior_directional'].copy()}
    modified_data['prior_multiscale'][pos]=jitter['prior_multiscale'];modified_data['prior_directional'][pos]=jitter['prior_directional']
    modified,_,_=repair_frame(raw,modified_data,view);modified=modified.iloc[pos]
    all_faith=[];curves=[];results={}
    with threadpool_limits(limits=1):
        for name,positions in [('A32',ap),('B32',bp)]:
            result=group_evaluate(bundle,frame,X.iloc[positions],groups,idx['row_ids'],view+'_'+name)
            save_group(output/'groups'/name,result,idx['row_ids'],y[pos],groups,raw.index[positions].to_numpy());results[name]=result
        targets=results['A32']['values'][:,-1].argmax(2)
        for donor,rank in [('A32','A32'),('B32','B32'),('B32','A32')]:
            rows,paths=metric_rows(results[donor],idx['row_ids'],y[pos],targets,ranking_result=results[rank],donor=donor,ranking_donor=rank)
            all_faith+=rows;curves+=paths
        changed=changed_evaluate(bundle,frame,modified,X.iloc[ap],groups,idx['row_ids'],results['A32'],'query_delay_5ms')
        save_group(output/'groups/query_delay_5ms',changed,idx['row_ids'],y[pos],groups,idx['background_A'])
    robust=robust_rows(results['A32'],changed,frame,modified,idx['row_ids'],y[pos],'query_delay_5ms')
    robust+=robust_rows(results['A32'],results['A32'],frame,frame,idx['row_ids'],y[pos],'port_bijection')
    distances=[]
    for i,row_id in enumerate(idx['row_ids']):
        for method,j in zip(METHODS,model_output_indices()):
            target=int(targets[i,j]);distances.append(dict(row_id=int(row_id),method=method,
                **attribution_distance(results['A32']['phi'][i,:,j,target],results['B32']['phi'][i,:,j,target])))
    faith=pd.DataFrame(all_faith);faith.to_csv(output/'tables/faithfulness.csv',index=False)
    pd.DataFrame(robust).to_csv(output/'tables/robustness.csv',index=False);pd.DataFrame(distances).to_csv(output/'tables/background_sensitivity.csv',index=False)
    e.write_json(output/'curves.json',curves)
    # Reconstruct every attribution and metric from saved games before completion.
    rebuilt=[]
    for name in ['A32','B32','query_delay_5ms']:
        saved=read_npz(output/'groups'/name/'coalitions.npz')
        for i in range(len(pos)):np.testing.assert_allclose(exact_shapley(saved['values'][i]),saved['phi'][i],rtol=1e-9,atol=1e-11)
    for donor,rank in [('A32','A32'),('B32','B32'),('B32','A32')]:
        a=read_npz(output/'groups'/donor/'coalitions.npz');b=read_npz(output/'groups'/rank/'coalitions.npz')
        rows,_=metric_rows(a,idx['row_ids'],y[pos],targets,ranking_result=b,donor=donor,ranking_donor=rank);rebuilt+=rows
    pd.testing.assert_frame_equal(pd.DataFrame(rebuilt),faith,check_dtype=False,atol=1e-12,rtol=0)
    expected=read_npz(quality_run/'stages/temporal_seed42'/f'{view}__balanced'/'evaluation.npz')
    ix=pd.Index(expected['row_ids']).get_indexer(idx['row_ids']);assert (ix>=0).all()
    np.testing.assert_allclose(results['A32']['values'][:,-1,0],expected['p_weighted'][ix],atol=1e-12,rtol=0)
    summary=faith[(faith.donor=='B32')&(faith.ranking_donor=='A32')&(faith.ranking=='signed')].groupby('method')[['deletion_auc','insertion_auc','deletion_gain','insertion_gain']].mean().reset_index()
    summary.to_csv(output/'tables/faithfulness_summary.csv',index=False)
    r=pd.DataFrame(robust);summaries=[]
    for variant in r.variant.unique():
        for subset,mask in [('all',r.variant==variant),('changed_input',(r.variant==variant)&~r.input_unchanged)]:
            for method in METHODS:
                sub=r[mask&(r.method==method)]
                summaries.append(dict(variant=variant,subset=subset,method=method,n=len(sub),prediction_unchanged=sub.prediction_unchanged.mean(),
                    top3_jaccard=sub.top3_jaccard.mean(),normalized_l1=sub.normalized_l1.mean()))
    pd.DataFrame(summaries).to_csv(output/'tables/robustness_summary.csv',index=False)
    e.write_json(output/'verification.json',dict(status='PASS_G_SELECTED_MULTISCALE_XAI',view=view,rows=len(pos),
        background_A=32,background_B=32,exact_attributions=300,new_classifier_fits=0,metric_rows=len(faith),
        role_G_model_selection=True,full_posterior_cache_matches=True,port_inputs_unchanged=True,
        query_delay_changed_inputs=int(sum(not frame.iloc[i].equals(modified.iloc[i]) for i in range(len(frame))))))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-jitter',action='store_true')
    p.add_argument('--quality-run',type=Path);p.add_argument('--jitter-run',type=Path);args=p.parse_args()
    if args.prepare_jitter:prepare_jitter(args.output)
    else:run(args.output,args.quality_run,args.jitter_run)
