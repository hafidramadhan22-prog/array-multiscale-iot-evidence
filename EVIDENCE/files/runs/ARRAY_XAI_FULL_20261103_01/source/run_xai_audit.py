"""Execute a frozen quantitative XAI audit. Owned checksum-verified models only; no fit."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys, time, zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
import shap
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_wb_research import load_data
from wb_research import view_data, bijective_ports, cube, mixture
from wb_behavior import PriorEndpointRing, PRIOR_SYMMETRIC
from sbfe import iter_pcap, role_for
from xai_audit import (METHODS,semantic_groups,subset_masks,exact_shapley,HybridGame,
    grouped_curves,curve_metrics,attribution_distance,path_masks,mask_violations)

SOURCE=['scripts/run_xai_audit.py','src/xai_audit.py','configs/xai_audit.json',
    'docs/XAI_AUDIT_PROTOCOL_20261020.md','docs/XAI_LITERATURE_20261020.md','tests/test_xai_audit.py',
    'src/wb_research.py','src/wb_behavior.py','src/e4.py','src/sbfe.py','src/ta_hafid_engine.py',
    'scripts/run_wb_research.py','scripts/prepare_sbfe.py','scripts/prepare_wb_behavior.py',
    'configs/target.json','requirements.lock.txt']


def verify_catalog(folder, catalog_name='artifacts_sha256.json'):
    catalog=folder/catalog_name
    items=json.loads(catalog.read_text(encoding='utf8'))
    for item in items:
        path=(folder/item['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or path.stat().st_size!=item['bytes'] or e.sha256(path)!=item['sha256']:
            raise ValueError('Artifact identity failure: '+str(path))
    return dict(folder=folder.relative_to(ROOT).as_posix(),catalog=catalog_name,catalog_sha256=e.sha256(catalog),artifacts=len(items))


def jitter_context(cfg, data, positions, shifts):
    folder=ROOT/cfg['data_folder'];config=json.loads((folder/'configuration_freeze.json').read_text())
    if e.sha256(Path(config['zip_path']))!=config['zip_sha256']: raise ValueError('Source ZIP changed')
    audit=json.loads((folder/'mapping_audit.json').read_text());selected={}
    for i,pos in enumerate(positions): selected.setdefault(int(data['capture'][pos]),{})[int(data['record'][pos])]=i
    values={f'query_delay_{ms}ms':np.empty((len(positions),40)) for ms in shifts}
    sources=[];seen_ids=set()
    with zipfile.ZipFile(config['zip_path']) as archive:
        for source in audit['sources']:
            cap=int(source['capture_id'])
            if cap not in selected: continue
            ring=PriorEndpointRing(config['ring_capacity'],config['windows_seconds']);digest=hashlib.sha256();prev=None;count=0
            for record,stamp,_,_,packet in iter_pcap(archive,archive.getinfo(source['entry']),digest):
                role=role_for(record,source['packets'])
                if role!=prev:ring.reset();prev=role
                if record in selected[cap]:
                    if role!=3: raise ValueError('Jitter requires V only')
                    i=selected[cap][record];pos=positions[i]
                    np.testing.assert_allclose(ring.query_prior(stamp,packet.src,packet.dst)['prior_symmetric'],data['prior_symmetric'][pos],atol=1e-12,rtol=0)
                    for ms in shifts:
                        values[f'query_delay_{ms}ms'][i]=ring.query_prior(stamp+int(ms*1e6),packet.src,packet.dst)['prior_symmetric']
                    seen_ids.add(i)
                ring.push(stamp,packet);count+=1
            if digest.hexdigest()!=source['sha256'] or count!=source['packets']: raise ValueError('PCAP source mismatch')
            sources.append(dict(capture_id=cap,packets=count,sha256=digest.hexdigest()))
    if len(seen_ids)!=len(positions): raise ValueError('Incomplete jitter extraction')
    return values,sources


def model_output_indices(): return [0,1+e.NAMES.index('random_forest'),1+e.NAMES.index('logistic_regression')]


def group_evaluate(bundle, frame, background, groups, ids, label, base=None, library_check=True):
    masks=subset_masks(len(groups));values=[];attrs=[];durations=[];shap_checks=[];rows_evaluated=0
    for i,(_,sample) in enumerate(frame.iterrows()):
        start=time.perf_counter();game=HybridGame(bundle,sample,background,groups)
        val=game(masks).reshape(len(masks),6,5);phi=exact_shapley(val)
        # Algebraic linearity independently checks the full weighted probability target.
        np.testing.assert_allclose(phi[:,0],np.einsum('m,jmc->jc',bundle['weights'],phi[:,1:]),rtol=1e-8,atol=1e-10)
        if library_check and i<2:
            def lookup(z):return val[(np.asarray(z,dtype=int)@(2**np.arange(len(groups)))).astype(int)].reshape(len(z),30)
            check=shap.ExactExplainer(lookup,np.zeros((1,len(groups))))(np.ones((1,len(groups))),max_evals=len(masks),silent=True)
            error=float(np.max(np.abs(check.values[0].reshape(phi.shape)-phi)))
            if error>1e-10:raise ValueError('Official SHAP implementation disagreement')
            shap_checks.append(dict(row_id=int(ids[i]),max_difference=error))
        values.append(val);attrs.append(phi);durations.append(time.perf_counter()-start);rows_evaluated+=game.rows_evaluated
        if len(frame)>5 and (i+1)%25==0:print('GROUP_PROGRESS',label,i+1,'/',len(frame),flush=True)
    values=np.stack(values);attrs=np.stack(attrs)
    np.testing.assert_allclose(values[:,-1,0],mixture(cube(bundle['models'],frame),bundle['weights']),atol=1e-12,rtol=0)
    if base is not None:np.testing.assert_allclose(values[:,-1],base,atol=1e-12,rtol=0)
    return dict(values=values,phi=attrs,seconds=np.array(durations),rows_evaluated=rows_evaluated,official_checks=shap_checks,label=label)


def metric_rows(result, ids, y, targets=None, ranking_result=None, donor='A', ranking_donor='A'):
    values,phi=result['values'],result['phi'];rows=[];curves=[]
    if ranking_result is not None:phi=ranking_result['phi']
    for i,row_id in enumerate(ids):
        for method,j in zip(METHODS,model_output_indices()):
            target=int(values[i,-1,j].argmax()) if targets is None else int(targets[i,j])
            for ranking in ['signed','absolute','least']:
                order,d,ins,rd,ri=grouped_curves(values[i],phi[i],target,j,ranking)
                rows.append(dict(row_id=int(row_id),true_class=int(y[i]),target=target,method=method,
                    donor=donor,ranking_donor=ranking_donor,ranking=ranking,units=len(phi[i]),
                    full_probability=float(values[i,-1,j,target]),correct=bool(target==y[i]),
                    **curve_metrics(d,ins,rd,ri)))
                curves.append(dict(row_id=int(row_id),method=method,donor=donor,ranking_donor=ranking_donor,ranking=ranking,deletion=d.tolist(),insertion=ins.tolist(),random_deletion=rd.tolist(),random_insertion=ri.tolist()))
    return rows,curves


def save_group(folder,result,ids,y,groups,background_ids):
    folder.mkdir(parents=True)
    np.savez_compressed(folder/'coalitions.npz',row_ids=ids,y=y,background_ids=background_ids,
        values=result['values'],phi=result['phi'],seconds=result['seconds'])
    e.write_json(folder/'metadata.json',dict(groups=groups,official_checks=result['official_checks'],
        hybrid_rows_evaluated=result['rows_evaluated'],coalitions_per_sample=2**len(groups),
        output_order=['weighted_voting']+e.NAMES,link='identity_probability',exact_for_empirical_game=True))


def atomic_evaluate(bundle,frame,background,ids,y,config,output):
    output.mkdir(parents=True);groups={c:[c] for c in frame.columns};d=len(groups);records=[];distances=[];stored=[];violations=[]
    seeds=[config['seed'],config['seed']+1];rng=np.random.default_rng(config['seed']+100)
    random_orders=[rng.permutation(d) for _ in range(config['atomic_random_orders'])]
    for i,(_,sample) in enumerate(frame.iterrows()):
        game=HybridGame(bundle,sample,background,groups);attrs=[];times=[]
        for seed in seeds:
            t=time.perf_counter()
            explainer=shap.PermutationExplainer(game,np.zeros((1,d)),seed=seed)
            result=explainer(np.ones((1,d)),max_evals=config['permutation_cycles']*(2*d+1),batch_size=1024,silent=True)
            phi=result.values[0].reshape(d,6,5);attrs.append(phi);times.append(time.perf_counter()-t)
            ends=game(np.stack([np.zeros(d),np.ones(d)])).reshape(2,6,5)
            np.testing.assert_allclose(phi.sum(0),ends[1]-ends[0],atol=1e-9,rtol=1e-8)
        sample_curves=[]
        for method,j in zip(METHODS,model_output_indices()):
            target=int(ends[1,j].argmax());rd=[];ri=[]
            for order in random_orders:
                dm,im=path_masks(order);out=game(np.r_[dm,im]).reshape(2*(d+1),6,5)
                rd.append(out[:d+1,j,target]);ri.append(out[d+1:,j,target])
            rd=np.mean(rd,axis=0);ri=np.mean(ri,axis=0)
            for seed_index,phi in enumerate(attrs):
                score=phi[:,j,target]
                order=np.argsort(-score,kind='stable');dm,im=path_masks(order)
                out=game(np.r_[dm,im]).reshape(2*(d+1),6,5);deletion=out[:d+1,j,target];insertion=out[d+1:,j,target]
                records.append(dict(row_id=int(ids[i]),true_class=int(y[i]),target=target,method=method,seed=seeds[seed_index],units=d,
                    seconds=times[seed_index],**curve_metrics(deletion,insertion,rd,ri)))
                sample_curves.append(dict(method=method,seed=seeds[seed_index],deletion=deletion.tolist(),insertion=insertion.tolist(),random_deletion=rd.tolist(),random_insertion=ri.tolist()))
            distances.append(dict(row_id=int(ids[i]),true_class=int(y[i]),method=method,target=target,
                **attribution_distance(attrs[0][:,j,target],attrs[1][:,j,target])))
        masks=np.array(list(game.cache),dtype=bool)
        violations.append(dict(row_id=int(ids[i]),**mask_violations(sample,background,masks,groups)))
        stored.append(dict(phi=np.stack(attrs),curves=sample_curves))
        if (i+1)%5==0: print('ATOMIC',bundle['view'],i+1,'/',len(frame),flush=True)
    np.savez_compressed(output/'attributions.npz',row_ids=ids,y=y,phi=np.stack([r['phi'] for r in stored]),seeds=seeds)
    e.write_json(output/'curves.json',[dict(row_id=int(ids[i]),curves=r['curves']) for i,r in enumerate(stored)])
    e.write_json(output/'mask_violations.json',violations)
    e.write_json(output/'schema.json',dict(columns=list(frame.columns),background_ids=background.index.tolist(),random_orders=[r.tolist() for r in random_orders],cycles=config['permutation_cycles']))
    return records,distances


def run(output,smoke=False):
    output=output.resolve()
    if output.parent!=(ROOT/'runs').resolve() or output.exists():raise ValueError('New run directory required')
    config=json.loads((ROOT/'configs/xai_audit.json').read_text());config['smoke_test']=smoke
    old_run=ROOT/config['source_run'];done=json.loads((old_run/'run_complete.json').read_text())
    if done['status']!='WB_RESEARCH_COMPLETE_VERIFIED' or e.sha256(old_run/'artifacts_sha256.json')!=done['catalog_sha256']: raise ValueError('Frozen WB run incomplete')
    before=verify_catalog(old_run)
    cfg=json.loads((old_run/'configuration_freeze.json').read_text());raw,y,holdout,data=load_data(cfg)
    with np.load(old_run/'stages/temporal_seed42/split.npz',allow_pickle=False) as z:parts={k:z[k].copy() for k in z.files}
    ids=raw.index.to_numpy();pos={k:raw.index.get_indexer(v) for k,v in parts.items()}
    if any((v<0).any() for v in pos.values()) or set(ids)&set(holdout):raise ValueError('Development identity gate')
    rng=np.random.default_rng(config['seed']);per_class=1 if smoke else config['validation_per_class']
    chosen=np.column_stack([rng.choice(pos['V'][y[pos['V']]==k],per_class,replace=False) for k in range(5)]).ravel()
    bsize=2 if smoke else config['background_rows'];bpos=rng.choice(pos['A'],2*bsize,replace=False);b1,b2=np.split(bpos,2)
    selected_ids=ids[chosen];labels=y[chosen]
    if set(selected_ids)&set(ids[np.r_[b1,b2]]) or set(ids[b1])&set(ids[b2]):raise ValueError('Background overlap')
    jobs=[(42,view,case) for view in config['primary_views'] for case in config['primary_cases']]
    if smoke:jobs=[(42,'legacy29_symmetric','balanced')]
    else:jobs += [(seed,view,'balanced') for seed in config['repeat_seeds'] for view in config['focus_views']]
    manifest=json.loads((old_run/'models_manifest.json').read_text());quality=json.loads((old_run/'quality_complete.json').read_text())
    if e.sha256(old_run/'models_manifest.json')!=quality['models_manifest_sha256']:raise ValueError('Model manifest changed')
    selected_models=[]
    for seed,view,case in jobs:
        record=[r for r in manifest if (r['stage'],r['view'],r['case'])==(f'temporal_seed{seed}',view,case)]
        if len(record)!=1:raise ValueError('Owned model missing')
        record=record[0];path=(old_run/record['path']).resolve()
        if not path.is_relative_to(old_run.resolve()) or e.sha256(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('Model checksum before load')
        selected_models.append(record)
    output.mkdir();(output/'source').mkdir()
    source_manifest=[]
    for rel in SOURCE:
        path=ROOT/rel;dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        source_manifest.append(dict(path=rel,sha256=e.sha256(path)))
    e.write_json(output/'source_manifest.json',source_manifest)
    e.write_json(output/'configuration_freeze.json',config)
    e.write_json(output/'inputs_manifest.json',dict(source_run=before,models=selected_models,
        data_catalog_sha256=e.sha256(ROOT/cfg['data_folder']/'catalog.json'),
        source_models_manifest_sha256=quality['models_manifest_sha256'],canonical_sha256=e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv')))
    e.write_json(output/'environment.json',{**e.environment(),'shap_version':shap.__version__})
    np.savez_compressed(output/'indices.npz',row_ids=selected_ids,y=labels,background_A=ids[b1],background_B=ids[b2],A=parts['A'],V=parts['V'],historical_holdout_ids=holdout)
    print('FREEZE',len(jobs),'bundles;',len(chosen),'V rows;',bsize,'A donors',flush=True)
    jitter,sources=jitter_context(cfg,data,chosen,config['jitter_ms'])
    np.savez_compressed(output/'perturbed_context.npz',row_ids=selected_ids,**jitter)
    e.write_json(output/'pcap_sources.json',sources)
    shifted_raw=bijective_ports(raw,cfg['port_bijection_offset'])
    all_metrics=[];all_robust=[];all_background=[];all_atomic=[];all_seed_noise=[];all_timing=[]
    with threadpool_limits(limits=1):
        for record in selected_models:
            seed=int(record['stage'].split('seed')[1]);view=record['view'];case=record['case'];label=f'{view}__{case}__seed{seed}'
            print('GROUP',label,flush=True)
            path=old_run/record['path']
            if e.sha256(path)!=record['sha256']:raise ValueError('Owned model changed immediately before load')
            bundle=joblib.load(path)
            X,_,_=view_data(raw,data,view);frame=X.iloc[chosen];background=X.iloc[b1];groups=semantic_groups(X.columns)
            with np.load(path.parent/'evaluation.npz',allow_pickle=False) as z:
                p=raw.index.get_indexer(z['row_ids']);vi={int(v):i for i,v in enumerate(p)}
                take=np.array([vi[int(c)] for c in chosen]);reference=z['p_experts'][take]
            native=cube(bundle['models'],frame);np.testing.assert_allclose(native,reference,rtol=0,atol=1e-12)
            base=np.concatenate([mixture(native,bundle['weights'])[:,None,:],native],axis=1)
            original=group_evaluate(bundle,frame,background,groups,selected_ids,label,base=base)
            targets=base.argmax(2)
            out=output/'groups'/label/'donor_A';save_group(out,original,selected_ids,labels,groups,ids[b1])
            metrics,curves=metric_rows(original,selected_ids,labels,targets)
            e.write_json(out/'curves.json',curves)
            all_metrics.extend([dict(bundle=label,**m) for m in metrics])
            all_timing.append(dict(bundle=label,variant='donor_A',rows=len(frame),seconds=float(original['seconds'].sum()),hybrid_rows=original['rows_evaluated']))
            if seed!=42 or case!='balanced' or view not in config['focus_views']:continue
            for donor,refs,bids in [('B',X.iloc[b2],ids[b2]),('zero',background.iloc[:1].copy(),np.array([-1]))]:
                if donor=='zero':
                    for c in refs.columns:refs[c]=0. if c in bundle['numeric'] else '0'
                result=group_evaluate(bundle,frame,refs,groups,selected_ids,label+'_'+donor,base=base)
                out=output/'groups'/label/('donor_'+donor);save_group(out,result,selected_ids,labels,groups,bids)
                own,curves=metric_rows(result,selected_ids,labels,targets,donor=donor,ranking_donor=donor)
                cross,cross_curves=metric_rows(result,selected_ids,labels,targets,ranking_result=original,donor=donor,ranking_donor='A')
                e.write_json(out/'curves.json',curves+cross_curves)
                all_metrics.extend([dict(bundle=label,**m) for m in own+cross])
                for i,row_id in enumerate(selected_ids):
                    for method,j in zip(METHODS,model_output_indices()):
                        t=targets[i,j]
                        all_background.append(dict(bundle=label,donor=donor,row_id=int(row_id),true_class=int(labels[i]),method=method,
                            **attribution_distance(original['phi'][i,:,j,t],result['phi'][i,:,j,t])))
            port_frame,_,_=view_data(shifted_raw,data,view)
            variants={'port_bijection':port_frame.iloc[chosen]}
            for variant,array in jitter.items():
                mod=frame.copy();mod.loc[:,PRIOR_SYMMETRIC]=array;variants[variant]=mod
            for variant,mod in variants.items():
                result=group_evaluate(bundle,mod,background,groups,selected_ids,label+'_'+variant)
                save_group(output/'groups'/label/variant,result,selected_ids,labels,groups,ids[b1])
                for i,row_id in enumerate(selected_ids):
                    same_input=bool(frame.iloc[i].equals(mod.iloc[i]))
                    for method,j in zip(METHODS,model_output_indices()):
                        t=targets[i,j];newp=result['values'][i,-1,j];oldp=base[i,j]
                        all_robust.append(dict(bundle=label,variant=variant,row_id=int(row_id),true_class=int(labels[i]),method=method,
                            original_prediction=int(t),perturbed_prediction=int(newp.argmax()),prediction_unchanged=bool(newp.argmax()==t),
                            input_unchanged=same_input,probability_l1=float(np.abs(newp-oldp).sum()),target_probability_change=float(newp[t]-oldp[t]),
                            **attribution_distance(original['phi'][i,:,j,t],result['phi'][i,:,j,t])))
            ac=1 if smoke else config['atomic_per_class'];ai=np.column_stack([np.flatnonzero(labels==k)[:ac] for k in range(5)]).ravel()
            atomic,noise=atomic_evaluate(bundle,frame.iloc[ai],background,selected_ids[ai],labels[ai],config,output/'atomic'/label)
            all_atomic.extend([dict(bundle=label,**m) for m in atomic]);all_seed_noise.extend([dict(bundle=label,**m) for m in noise])
    (output/'tables').mkdir()
    for name,rows in [('faithfulness',all_metrics),('robustness',all_robust),('background_sensitivity',all_background),('atomic_faithfulness',all_atomic),('atomic_seed_sensitivity',all_seed_noise),('group_timing',all_timing)]:
        pd.DataFrame(rows).to_csv(output/'tables'/f'{name}.csv',index=False,float_format='%.17g')
    after=verify_catalog(old_run)
    if before!=after:raise ValueError('Source run changed')
    for item in source_manifest:
        if e.sha256(ROOT/item['path'])!=item['sha256']:raise ValueError('Audit source changed while executing')
    e.write_json(output/'execution_complete.json',dict(status='XAI_SMOKE_COMPLETE' if smoke else 'XAI_AUDIT_EXECUTION_COMPLETE',
        bundles=len(jobs),validation_rows=len(chosen),metric_rows=len(all_metrics),robustness_rows=len(all_robust),atomic_metric_rows=len(all_atomic),
        new_classifier_fits=0,source_run_artifacts_unchanged=before['artifacts'],historical_holdout_used=False))
    print('XAI_EXECUTION_COMPLETE',output.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(a.output,a.smoke)
