"""Recover legacy cache adapter without refitting or changing G-selected SVMs."""
from pathlib import Path
import sys,argparse,json,time,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_svm_20261118 import read,npz,gated,verify_sources,stage_inputs,STAGES
from array_svm_20261118 import label_metrics,select_on_G,CLASSES
from wb_research import mixture

def finalize_fit_inventory(training):
    verify_sources(training);cfg=read(training/'configuration_freeze.json');assert not cfg['smoke'] and cfg['stages']==STAGES
    count=0
    for name in STAGES:
        d=training/'stages'/name;g=pd.read_csv(d/'G_candidates.csv');assert len(g)==9 and (g.fit_status==0).all()
        freeze=read(d/'selection_before_V.json');manifest=read(d/'models_manifest.json')
        assert freeze['selected']==select_on_G(g.to_dict('records')) and freeze['V_scored'] is False
        for r in manifest:
            f=training/r['path'];assert f.resolve().is_relative_to(training.resolve()) and f.stat().st_size==r['bytes'] and e.sha256(f)==r['sha256']
        count+=len(manifest)
    assert count==63
    if not (training/'complete.json').exists():
        e.write_json(training/'cache_adapter_failure.json',dict(date='2026-11-18',error="KeyError weights: legacy full-history evaluation.npz does not contain voting weights",training_complete=True,SVC_fits=63,selection_complete_before_V=True,repair='Read unchanged weights from hash-verified selection_before_V.json in a separate evaluation run',new_search_or_refit=False,private_evaluations_retained=True))
        records=[dict(path=f.relative_to(training).as_posix(),bytes=f.stat().st_size,sha256=e.sha256(f)) for f in sorted(training.rglob('*')) if f.is_file() and f.relative_to(training).as_posix() not in ['catalog.json','complete.json']]
        e.write_json(training/'catalog.json',records)
        e.write_json(training/'complete.json',dict(status='ALL63_SVM_FITS_RETAINED_LEGACY_EVALUATION_ADAPTER_FAILURE',new_SVC_fits=63,new_WV_fits=0,catalog_sha256=e.sha256(training/'catalog.json')))
    done=read(training/'complete.json');assert done['catalog_sha256']==e.sha256(training/'catalog.json')
    for r in read(training/'catalog.json'):
        f=training/r['path'];assert f.stat().st_size==r['bytes'] and e.sha256(f)==r['sha256']
    return done

def load_svm(training,name,candidate):
    assert read(training/'complete.json')['catalog_sha256']==e.sha256(training/'catalog.json')
    record=next(r for r in read(training/'stages'/name/'models_manifest.json') if r['candidate']==candidate)
    p=training/record['path'];assert p.resolve().is_relative_to(training.resolve()) and e.sha256(p)==record['sha256'] and p.stat().st_size==record['bytes']
    return joblib.load(p)

def main(training,out):
    assert not out.exists();finalize_fit_inventory(training);out.mkdir();(out/'source').mkdir();(out/'stages').mkdir()
    e.write_json(out/'configuration_freeze.json',dict(date='2026-11-18',training=training.relative_to(ROOT).as_posix(),training_catalog_sha256=e.sha256(training/'catalog.json'),new_SVC_fits=0,selected_candidates_unchanged=True,repair_only='Legacy cache weights read from recorded selection file',frozen_unix=time.time()))
    for rel in ['scripts/evaluate_array_svm_20261118.py','src/array_svm_20261118.py','docs/ARRAY_SVM_COMPARATOR_PROTOCOL_20261118.md']:shutil.copy2(ROOT/rel,out/'source'/Path(rel).name)
    rows=[]
    for name in STAGES:
        X,y,raw,holdout,parts,num,cat,local,ref=stage_inputs(name,False);a,g,c,v=parts;d=out/'stages'/name;d.mkdir()
        oldsplit=npz(training/'stages'/name/'split.npz')
        for key,pos in zip('AGCV',parts):np.testing.assert_array_equal(raw.index[pos],oldsplit[key])
        freeze=read(training/'stages'/name/'selection_before_V.json')
        predictions={};model_checks=[]
        with threadpool_limits(limits=1):
            for method,candidate in [('SVM_default',freeze['default']),('SVM_G_selected',freeze['selected'])]:
                model=load_svm(training,name,candidate)
                observedG=model.predict(X.iloc[g]);storedG=npz(training/'stages'/name/'G_predictions.npz')
                np.testing.assert_array_equal(observedG,storedG[candidate])
                predictions[method]=model.predict(X.iloc[v]);model_checks.append(dict(method=method,candidate=candidate,G_predictions_replayed_exactly=True))
        if 'weights' in ref:w=ref['weights'];weight_source='evaluation.npz'
        else:
            q=gated('reference',f'stages/{name}/legacy25_multiscale60__balanced/selection_before_V.json')
            w=read(q)['weights'];weight_source=q.relative_to(ROOT).as_posix()
        p=ref['p_experts'];np.testing.assert_array_equal(ref['row_ids'],raw.index[v]);np.testing.assert_array_equal(ref['y'],y[v])
        predictions.update(WV=mixture(p,w).argmax(1),RF=p[:,1].argmax(1),LR=p[:,4].argmax(1),Uniform=p.mean(1).argmax(1))
        protocol='packet_private' if name.startswith('private') else 'temporal' if name.startswith('temporal') else 'signature_shared'
        arrays=dict(row_ids=raw.index[v].to_numpy(),y=y[v]);localrows=[]
        for method,pred in predictions.items():
            m,cm=label_metrics(y[v],pred);row=dict(stage=name,protocol=protocol,method=method,**m);rows.append(row);localrows.append(row);arrays[method]=pred;arrays['CM_'+method]=cm
        earlier=training/'stages'/name/'evaluation.npz'
        if earlier.exists():
            z=npz(earlier)
            for k in arrays:np.testing.assert_array_equal(arrays[k],z[k])
        np.savez_compressed(d/'evaluation.npz',**arrays);pd.DataFrame(localrows).to_csv(d/'metrics.csv',index=False)
        e.write_json(d/'checks.json',dict(stage=name,models=model_checks,weight_source=weight_source,selected_before_V=freeze['selected'],no_refits=True,old_private_outputs_identical=earlier.exists(),split_matches=True,new_timestamp=time.time(),original_freeze_unix=freeze['frozen_unix']))
        print('RECOVERED_AND_VERIFIED',name,flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    records=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=e.sha256(f)) for f in sorted(out.rglob('*')) if f.is_file()]
    e.write_json(out/'catalog.json',records);e.write_json(out/'complete.json',dict(status='PASS_SVM_MATCHED_EVALUATION',stages=7,actual_evaluated_label_vectors=42,new_SVC_fits=0,prior_SVC_fits=63,RF_retained=True,weights_unchanged=True,catalog_sha256=e.sha256(out/'catalog.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,default=ROOT/'runs/ARRAY_SVM_FULL_20261118_01');p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.training.resolve(),a.output.resolve())
