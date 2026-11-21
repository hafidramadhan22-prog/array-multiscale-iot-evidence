"""Extend a verified SHAP audit with larger backgrounds, never refit classifiers."""
from pathlib import Path
import argparse,json,os,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_wb_context_ablation import check_catalog
from run_wb_research import load_data
from wb_research import view_data,cube,mixture,bijective_ports
from run_xai_audit import group_evaluate,metric_rows,save_group,model_output_indices
from xai_audit import semantic_groups,attribution_distance,METHODS
from wb_behavior import PRIOR_SYMMETRIC

SOURCE=['scripts/run_xai_background_extension.py','configs/xai_background_extension.json',
        'docs/XAI_BACKGROUND_EXTENSION_PROTOCOL_20261020.md','scripts/run_xai_audit.py',
        'src/xai_audit.py','src/wb_research.py','src/wb_behavior.py','src/ta_hafid_engine.py',
        'src/e4.py','src/sbfe.py','scripts/run_wb_research.py','scripts/prepare_wb_behavior.py',
        'scripts/prepare_sbfe.py','requirements.lock.txt']


def robust_rows(base,other,frame,modified,ids,y,variant):
    result=[];targets=base['values'][:,-1].argmax(2)
    for i,row_id in enumerate(ids):
        same=bool(frame.iloc[i].equals(modified.iloc[i]))
        for method,j in zip(METHODS,model_output_indices()):
            target=int(targets[i,j]);op=base['values'][i,-1,j];npred=other['values'][i,-1,j]
            result.append(dict(row_id=int(row_id),true_class=int(y[i]),method=method,variant=variant,
                original_prediction=target,perturbed_prediction=int(npred.argmax()),input_unchanged=same,
                prediction_unchanged=bool(npred.argmax()==target),probability_l1=float(np.abs(op-npred).sum()),
                **attribution_distance(base['phi'][i,:,j,target],other['phi'][i,:,j,target])))
    return result


def changed_evaluate(bundle,frame,modified,background,groups,ids,base,label):
    changed=np.array([not frame.iloc[i].equals(modified.iloc[i]) for i in range(len(frame))])
    values=base['values'].copy();phi=base['phi'].copy();seconds=np.zeros(len(frame))
    checks=[];rows=0
    if changed.any():
        result=group_evaluate(bundle,modified.iloc[np.flatnonzero(changed)],background,groups,ids[changed],label)
        values[changed]=result['values'];phi[changed]=result['phi'];seconds[changed]=result['seconds']
        checks=result['official_checks'];rows=result['rows_evaluated']
    return dict(values=values,phi=phi,seconds=seconds,rows_evaluated=rows,official_checks=checks,label=label)


def run(output,smoke=False):
    output=output.resolve()
    if output.parent!=(ROOT/'runs').resolve() or output.exists():raise ValueError('New runs child required')
    cfg=json.loads((ROOT/'configs/xai_background_extension.json').read_text());cfg['smoke_test']=smoke
    old=ROOT/cfg['source_run'];xai=ROOT/cfg['source_xai_run']
    before=[check_catalog(old),check_catalog(xai)]
    wc=json.loads((old/'configuration_freeze.json').read_text());raw,y,holdout,data=load_data(wc)
    with np.load(xai/'indices.npz',allow_pickle=False) as z:indices={k:z[k] for k in z.files}
    chosen=raw.index.get_indexer(indices['row_ids']);assert (chosen>=0).all()
    if smoke:
        take=np.array([np.flatnonzero(indices['y']==c)[0] for c in range(5)])
        chosen=chosen[take];cfg.update(views=['legacy29_symmetric'],background_rows=10,validation_rows=5)
    ids=raw.index[chosen].to_numpy();labels=y[chosen]
    a8=indices['background_A'];b8=indices['background_B'];pool=np.setdiff1d(indices['A'],np.r_[a8,b8])
    rng=np.random.default_rng(cfg['seed']);extra=cfg['background_rows']-len(a8)
    selected=rng.choice(pool,extra*2,replace=False)
    a=np.r_[a8,selected[:extra]];b=np.r_[b8,selected[extra:]]
    assert not set(a)&set(b) and set(np.r_[a,b]).issubset(indices['A'])
    assert not set(np.r_[a,b])&set(ids) and not set(ids)&set(holdout)
    assert set(ids).issubset(indices['V'])
    ap=raw.index.get_indexer(a);bp=raw.index.get_indexer(b)
    with np.load(xai/'perturbed_context.npz',allow_pickle=False) as z:
        lookup=pd.Index(z['row_ids']).get_indexer(ids);assert (lookup>=0).all()
        jitter=z[cfg['jitter_condition']][lookup]
    manifest=json.loads((old/'models_manifest.json').read_text())
    models=[next(r for r in manifest if (r['stage'],r['view'],r['case'])==('temporal_seed42',v,cfg['case'])) for v in cfg['views']]
    for item in models:
        path=(old/item['path']).resolve();assert path.is_relative_to(old.resolve())
        assert path.stat().st_size==item['bytes'] and e.sha256(path)==item['sha256']
    output.mkdir();(output/'source').mkdir();(output/'tables').mkdir()
    e.write_json(output/'configuration_freeze.json',cfg);e.write_json(output/'environment.json',e.environment())
    e.write_json(output/'inputs_manifest.json',dict(old_runs=before,owned_models=models,
        canonical_sha256=e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv'),source_indices_sha256=e.sha256(xai/'indices.npz'),
        perturbed_context_sha256=e.sha256(xai/'perturbed_context.npz')))
    np.savez_compressed(output/'indices.npz',row_ids=ids,y=labels,background_A=a,background_B=b,A=indices['A'],V=indices['V'])
    frozen=[]
    for rel in SOURCE:
        dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        frozen.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    e.write_json(output/'source_manifest.json',frozen)
    shifted=bijective_ports(raw,wc['port_bijection_offset'])
    all_metrics=[];all_dist=[];all_robust=[]
    with threadpool_limits(limits=1):
        for record in models:
            view=record['view'];path=old/record['path']
            # Explicitly owned model, checked immediately before deserialization.
            assert e.sha256(path)==record['sha256']
            bundle=joblib.load(path)
            X,_,_=view_data(raw,data,view);frame=X.iloc[chosen];groups=semantic_groups(frame.columns)
            original=cube(bundle['models'],frame);base=np.concatenate([mixture(original,bundle['weights'])[:,None,:],original],axis=1)
            results={}
            for name,positions in [('A32',ap),('B32',bp)]:
                print('BACKGROUND',view,name,'rows',len(frame),'donors',len(positions),flush=True)
                result=group_evaluate(bundle,frame,X.iloc[positions],groups,ids,view+'_'+name,base=base)
                save_group(output/'groups'/view/name,result,ids,labels,groups,raw.index[positions].to_numpy())
                results[name]=result
                rows,curves=metric_rows(result,ids,labels,base.argmax(2),donor=name,ranking_donor=name)
                all_metrics += [dict(view=view,**row) for row in rows]
                e.write_json(output/'groups'/view/name/'curves.json',curves)
            cross,curves=metric_rows(results['B32'],ids,labels,base.argmax(2),ranking_result=results['A32'],donor='B32',ranking_donor='A32')
            all_metrics += [dict(view=view,**row) for row in cross]
            e.write_json(output/'groups'/view/'B32'/'cross_curves.json',curves)
            oldfile=xai/'groups'/f'{view}__balanced__seed42'/'donor_A'/'coalitions.npz'
            with np.load(oldfile,allow_pickle=False) as z:
                take=pd.Index(z['row_ids']).get_indexer(ids);assert (take>=0).all()
                old_phi=z['phi'][take];old_values=z['values'][take]
            np.testing.assert_allclose(old_values[:,-1],base,rtol=0,atol=1e-12)
            for label,first,second in [('A8_to_A32',old_phi,results['A32']['phi']),('A32_to_B32',results['A32']['phi'],results['B32']['phi'])]:
                for i,row_id in enumerate(ids):
                    for method,j in zip(METHODS,model_output_indices()):
                        target=int(base[i,j].argmax())
                        all_dist.append(dict(view=view,comparison=label,row_id=int(row_id),method=method,
                            **attribution_distance(first[i,:,j,target],second[i,:,j,target])))
            portframe,_,_=view_data(shifted,data,view)
            jitterframe=frame.copy();jitterframe.loc[:,PRIOR_SYMMETRIC]=jitter
            for variant,modified in [('port_bijection',portframe.iloc[chosen]),(cfg['jitter_condition'],jitterframe)]:
                result=changed_evaluate(bundle,frame,modified,X.iloc[ap],groups,ids,results['A32'],view+'_'+variant)
                save_group(output/'groups'/view/variant,result,ids,labels,groups,a)
                all_robust += [dict(view=view,**row) for row in robust_rows(results['A32'],result,frame,modified,ids,labels,variant)]
    for name,rows in [('faithfulness',all_metrics),('background_sensitivity',all_dist),('robustness',all_robust)]:
        pd.DataFrame(rows).to_csv(output/'tables'/f'{name}.csv',index=False,float_format='%.17g')
    after=[check_catalog(old),check_catalog(xai)];assert before==after
    for item in frozen:assert e.sha256(ROOT/item['path'])==item['sha256']
    e.write_json(output/'execution_complete.json',dict(status='XAI_BACKGROUND_SMOKE_COMPLETE' if smoke else 'XAI_BACKGROUND_EXECUTION_COMPLETE',
        bundles=len(models),validation_rows=len(ids),background_rows=len(a),new_classifier_fits=0,
        old_runs_unchanged=after,metric_rows=len(all_metrics),robustness_rows=len(all_robust)))
    print('XAI_BACKGROUND_COMPLETE',output.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();run(args.output,args.smoke)
