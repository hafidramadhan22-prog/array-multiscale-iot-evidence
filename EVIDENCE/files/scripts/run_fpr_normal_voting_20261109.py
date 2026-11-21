"""A separate changed-predictor study with a second frozen later-packet block."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys,time,traceback,gc,threading
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from fpr_target8_20261109 import alternatives,metrics,safety_gate,NAMES,CLASSES
from rf_normal_voting_type_20261109 import combine,infer
from run_fpr_target8_20261109 import read,write,loadnp,verify_parent,verify_run_source,fresh,load_exported,profile_data,event,infer as simple_infer
FIRST=ROOT/'runs/FPR_TARGET8_20261109_01'

def setup(run):
    assert not run.exists() and run.parent==(ROOT/'runs').resolve();verify_parent();verify_run_source(FIRST)
    run.mkdir();(run/'source').mkdir();(run/'models').mkdir();(run/'posteriors').mkdir()
    cfg=read(FIRST/'configuration_freeze.json');cfg.update(experiment='RF_NORMAL_VOTING_ATTACK_TYPE',primary='rf_normal_then_voting',primary_chosen_after_historical_results=True,motivating_observation='Cached C Ransomware and seed44 SQL type loss for RF singleton; no threshold or fit changes',second_block_frozen_before_scoring=True,later_start_index=200999,later_end_exclusive=300000,csv_prefix=400000)
    cfg['sources']=[{**s,'planned_ids':list(range(200999,min(300000,s['packet_count']),1000))} for s in cfg['sources'] if s['packet_count']>200999];assert len(cfg['sources'])==7 and sum(len(s['planned_ids']) for s in cfg['sources'])==700
    write(run/'configuration_freeze.json',cfg);write(run/'environment.json',e.environment())
    rels=[r['path'] for r in read(FIRST/'source_manifest.json')]+['scripts/run_fpr_normal_voting_20261109.py','src/rf_normal_voting_type_20261109.py','docs/FPR_TARGET8_HIERARCHY_PROTOCOL_20261109.md'];manifest=[]
    for rel in rels:shutil.copy2(ROOT/rel,run/'source'/Path(rel).name);manifest.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    write(run/'source_manifest.json',manifest);rows=[];gates=[];changes=[];pooled=[]
    paths=list((FIRST/'posteriors').glob('*.npz'))
    for path in paths:
        if path.stem.endswith('_later'):continue
        z=loadnp(path);p,reason=combine(z['weighted_reference'],z['random_forest']);y=z['y'];ref=z['weighted_reference'].argmax(1);out=p.argmax(1)
        m,cm=metrics(y,out);rows.append(dict(cohort=path.stem,policy='rf_normal_then_voting',**m));gates.append(dict(cohort=path.stem,**safety_gate(y,ref,out)))
        changed=out!=ref
        for true in range(5):changes.append(dict(cohort=path.stem,true_class=CLASSES[true],support=int((y==true).sum()),changed=int((changed&(y==true)).sum()),rescued_from_Normal=int(((ref==0)&(out!=0)&(y==true)).sum()),correct_lost=int(((ref==y)&(out!=y)&(y==true)).sum()),correct_gained=int(((ref!=y)&(out==y)&(y==true)).sum())))
        np.savez_compressed(run/'posteriors'/(path.stem+'.npz'),row_ids=z['row_ids'],y=y,weighted_reference=z['weighted_reference'],random_forest=z['random_forest'],rf_normal_then_voting=p,reason=reason)
        if path.stem in ['signature_fold1','signature_fold2']:pooled.append((y,p))
    y=np.concatenate([a[0] for a in pooled]);p=np.vstack([a[1] for a in pooled]);m,cm=metrics(y,p.argmax(1));rows.append(dict(cohort='signature_pooled',policy='rf_normal_then_voting',**m))
    pd.DataFrame(rows).to_csv(run/'cached_metrics.csv',index=False);write(run/'cached_safety_gates.json',gates);pd.DataFrame(changes).to_csv(run/'activation_by_true_class.csv',index=False)
    required=['G','C','V_seed42','temporal_seed43','temporal_seed44'];assert all(next(r for r in gates if r['cohort']==s)['passed'] for s in required),gates
    write(run/'cached_complete.json',dict(status='CACHED_DIAGNOSIS_COMPLETE',first_study=FIRST.relative_to(ROOT).as_posix(),all_five_temporal_role_seed_safety_gates_passed=True,second_block_not_scored=True))
    event(run,'NEW_RULE_FROZEN; ALL_G_C_V42_V43_V44_ATTACK_RECALL_GATES_PASS')
    print(pd.DataFrame(rows)[['cohort','false_alarms','attack_to_Normal','macro_F1_percent']].to_string(index=False),flush=True)

def decorate(run):
    rows=[];decisions=[]
    for path in (run/'posteriors').glob('*_later.npz'):
        z=loadnp(path);p,reason=combine(z['weighted_reference'],z['random_forest']);m,cm=metrics(z['y'],p.argmax(1));rows.append(dict(source=path.stem[:-6],**m))
        decisions.extend(dict(row_id=str(i),policy='rf_normal_then_voting',true_class=CLASSES[int(y)],prediction=CLASSES[int(a)],reason=str(q),**{'p_'+c:float(pp[k]) for k,c in enumerate(CLASSES)}) for i,y,a,q,pp in zip(z['row_ids'],z['y'],p.argmax(1),reason,p))
    pd.DataFrame(rows).to_csv(run/'hierarchy_later_per_capture.csv',index=False);pd.DataFrame(decisions).to_csv(run/'hierarchy_later_decisions.csv',index=False)
    # Reanalysis of first study's newly scored block is labelled retrospective for this rule.
    ret=[]
    for path in (FIRST/'posteriors').glob('*_later.npz'):
        z=loadnp(path);p,_=combine(z['weighted_reference'],z['random_forest']);m,_=metrics(z['y'],p.argmax(1));ret.append(dict(source=path.stem[:-6],scope='retrospective first later block for this new rule',**m))
    pd.DataFrame(ret).to_csv(run/'hierarchy_first_block_reanalysis.csv',index=False)
    manifest=read(run/'models_manifest.json');r=next(r for r in manifest if r['key']=='weighted_reference');manifest.append({**r,'key':'rf_normal_then_voting','creator':'Original owned fitted experts with new frozen Python decision rule; zero fits'});write(run/'models_manifest.json',manifest)
    expected=loadnp(run/'profile_expected.npz')
    for cohort in ['V1000','later']:expected[cohort+'_hier']=combine(expected[cohort+'_WV'],expected[cohort+'_RF'])[0]
    np.savez_compressed(run/'profile_expected.npz',**expected);event(run,'SECOND_LATER_BLOCK_AND_NEW_RULE_COMPLETE')

def probability(model,policy,X):return infer(model,X,NAMES) if policy=='rf_normal_then_voting' else simple_infer(model,policy,X)
def memory(run,policy,cohort,rep):
    model=load_exported(run,policy);X=profile_data(run)[cohort];key='_hier' if policy=='rf_normal_then_voting' else '_RF' if policy=='random_forest' else '_WV';expected=loadnp(run/'profile_expected.npz')[cohort+key]
    gc.collect();proc=psutil.Process();baseline=proc.memory_info().rss;peak=[baseline];stop=threading.Event()
    def watch():
        while not stop.wait(.001):peak.append(proc.memory_info().rss)
    t=threading.Thread(target=watch,daemon=True);t.start();p=probability(model,policy,X);peak.append(proc.memory_info().rss);stop.set();t.join();np.testing.assert_allclose(p,expected,rtol=0,atol=1e-12)
    print(json.dumps(dict(policy=policy,cohort=cohort,repeat=rep,n=len(X),baseline_RSS_MiB=baseline/2**20,delta_peak_RSS_MiB=(max(peak)-baseline)/2**20,expected_probabilities_match=True)))
def profile(run):
    data=profile_data(run);names=['random_forest','weighted_reference','rf_normal_then_voting'];models={n:load_exported(run,n) for n in names};expected=loadnp(run/'profile_expected.npz');times=[];rng=np.random.default_rng(20261109)
    for cohort,X in data.items():
        for name in names:
            for _ in range(3):probability(models[name],name,X)
        for rep in range(30):
            for name in rng.permutation(names):
                start=time.perf_counter();p=probability(models[name],name,X);ms=1000*(time.perf_counter()-start);k='_hier' if name=='rf_normal_then_voting' else '_RF' if name=='random_forest' else '_WV';np.testing.assert_allclose(p,expected[cohort+k],rtol=0,atol=1e-12);times.append(dict(cohort=cohort,policy=name,repeat=rep,n=len(X),milliseconds=ms))
        event(run,'LATENCY_COMPLETE '+cohort)
    pd.DataFrame(times).to_csv(run/'latency_raw.csv',index=False);rss=[]
    for cohort in data:
        for name in names:
            for rep in range(3):
                p=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--phase','memory','--run',str(run),'--policy',name,'--cohort',cohort,'--repeat',str(rep)],cwd=ROOT,text=True,capture_output=True);assert p.returncode==0,p.stdout+p.stderr;rss.append(json.loads(p.stdout.strip().splitlines()[-1]))
            event(run,'RSS_COMPLETE '+cohort+' '+name)
    pd.DataFrame(rss).to_csv(run/'memory_raw.csv',index=False)
    a=pd.DataFrame(times).groupby(['cohort','policy']).milliseconds.agg(['count','mean','median','std']).reset_index();b=pd.DataFrame(rss).groupby(['cohort','policy']).delta_peak_RSS_MiB.agg(['mean','std']).rename(columns={'mean':'RSS_delta_mean_MiB','std':'RSS_delta_SD_MiB'}).reset_index();a.merge(b,on=['cohort','policy']).to_csv(run/'efficiency_summary.csv',index=False)
    write(run/'artifacts_sha256.json',[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ['complete.json','artifacts_sha256.json','failure.json']])
    write(run/'complete.json',dict(status='FPR_NORMAL_VOTING_COMPLETE',catalog_sha256=e.sha256(run/'artifacts_sha256.json'),new_classifier_fits=0,primary='rf_normal_then_voting'));print('HIERARCHY_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['setup','fresh','profile','memory'],required=True);ap.add_argument('--run',type=Path,default=ROOT/'runs/FPR_NORMAL_VOTING_20261109_01');ap.add_argument('--policy');ap.add_argument('--cohort');ap.add_argument('--repeat',type=int,default=0);a=ap.parse_args();r=a.run.resolve()
    try:
        with threadpool_limits(limits=1):
            if a.phase=='setup':setup(r)
            elif a.phase=='fresh':fresh(r);decorate(r)
            elif a.phase=='profile':profile(r)
            else:memory(r,a.policy,a.cohort,a.repeat)
    except Exception:
        if r.exists() and a.phase!='memory':write(r/'failure.json',dict(phase=a.phase,traceback=traceback.format_exc()))
        raise
