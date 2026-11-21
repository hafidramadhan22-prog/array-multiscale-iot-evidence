"""Third frozen later block, targeted SQL guard plus attack rescue, no refit."""
from pathlib import Path
import argparse,gc,json,shutil,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from fpr_target8_20261109 import NAMES,CLASSES,metrics,safety_gate
from sql_guard_rf_rescue_20261109 import combine,infer
from rf_normal_voting_type_v2_20261109 import full_voting
from run_fpr_target8_20261109 import read,write,loadnp,verify_parent,verify_run_source,fresh,profile_data,load_exported,event
FIRST=ROOT/'runs/FPR_TARGET8_20261109_01';SECOND=ROOT/'runs/FPR_NORMAL_VOTING_20261109_01';RUN=ROOT/'runs/FPR_SQL_GUARD_20261109_01'
KEY='sql_guard_rf_rescue'

def setup(run):
    assert not run.exists() and run.parent==(ROOT/'runs').resolve();verify_parent();verify_run_source(FIRST)
    run.mkdir();(run/'source').mkdir();(run/'models').mkdir();(run/'posteriors').mkdir()
    cfg=read(FIRST/'configuration_freeze.json');cfg.update(experiment='SQL_GUARD_RF_RESCUE',primary=KEY,chosen_after_two_later_blocks=True,third_block_frozen_before_scoring=True,later_start_index=300999,later_end_exclusive=400000,csv_prefix=500000)
    cfg['sources']=[{**s,'planned_ids':list(range(300999,min(400000,s['packet_count']),1000))} for s in cfg['sources'] if s['packet_count']>=400000]
    assert len(cfg['sources'])==7 and sum(len(s['planned_ids']) for s in cfg['sources'])==700
    write(run/'configuration_freeze.json',cfg);write(run/'environment.json',e.environment())
    rels=[r['path'] for r in read(FIRST/'source_manifest.json')]+['src/rf_normal_voting_type_20261109.py','src/rf_normal_voting_type_v2_20261109.py','src/sql_guard_rf_rescue_20261109.py','scripts/run_fpr_sql_guard_20261109.py','docs/FPR_TARGET8_SQL_GUARD_PROTOCOL_20261109.md']
    manifest=[]
    for rel in rels:shutil.copy2(ROOT/rel,run/'source'/Path(rel).name);manifest.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    write(run/'source_manifest.json',manifest);rows=[];gates=[];activations=[];pool=[]
    for path in (FIRST/'posteriors').glob('*.npz'):
        if path.stem.endswith('_later'):continue
        z=loadnp(path);v=z['weighted_reference'];r=z['random_forest'];p,reason=combine(v,r);y=z['y'];a=p.argmax(1);b=v.argmax(1)
        m,_=metrics(y,a);rows.append(dict(cohort=path.stem,policy=KEY,**m));gates.append(dict(cohort=path.stem,**safety_gate(y,b,a)))
        for k,c in enumerate(CLASSES):activations.append(dict(cohort=path.stem,true_class=c,support=int((y==k).sum()),changed=int(((a!=b)&(y==k)).sum()),correct_lost=int(((b==y)&(a!=y)&(y==k)).sum()),correct_gained=int(((b!=y)&(a==y)&(y==k)).sum()),rescued_from_Normal=int(((b==0)&(a!=0)&(y==k)).sum())))
        np.savez_compressed(run/'posteriors'/path.name,row_ids=z['row_ids'],y=y,weighted_reference=v,random_forest=r,sql_guard_rf_rescue=p,reason=reason)
        if path.stem.startswith('signature'):pool.append((y,p))
    y=np.concatenate([a[0] for a in pool]);p=np.vstack([a[1] for a in pool]);m,_=metrics(y,p.argmax(1));rows.append(dict(cohort='signature_pooled',policy=KEY,**m))
    pd.DataFrame(rows).to_csv(run/'cached_metrics.csv',index=False);write(run/'cached_safety_gates.json',gates);pd.DataFrame(activations).to_csv(run/'activation_by_true_class.csv',index=False)
    assert all(g['passed'] for g in gates if g['evaluable']),gates
    write(run/'cached_complete.json',dict(status='CACHED_DIAGNOSIS_COMPLETE',all_seven_cached_attack_recall_gates_passed=True,third_block_not_scored=True))
    event(run,'TARGETED_SQL_GUARD_FROZEN_AFTER_TWO_BLOCKS; ALL_SEVEN_CACHED_GATES_PASS')

def decorate(run):
    results=[];decisions=[]
    for path in (run/'posteriors').glob('*_later.npz'):
        z=loadnp(path);p,reason=combine(z['weighted_reference'],z['random_forest']);m,_=metrics(z['y'],p.argmax(1));results.append(dict(source=path.stem[:-6],**m))
        decisions.extend(dict(row_id=str(i),true_class=CLASSES[int(y)],prediction=CLASSES[int(a)],reason=str(q),**{'p_'+c:float(pp[k]) for k,c in enumerate(CLASSES)}) for i,y,a,q,pp in zip(z['row_ids'],z['y'],p.argmax(1),reason,p))
    pd.DataFrame(results).to_csv(run/'candidate_later_per_capture.csv',index=False);pd.DataFrame(decisions).to_csv(run/'candidate_later_decisions.csv',index=False)
    old=[]
    for folder in [FIRST,SECOND]:
        for path in (folder/'posteriors').glob('*_later.npz'):
            z=loadnp(path);p,_=combine(z['weighted_reference'],z['random_forest']);m,_=metrics(z['y'],p.argmax(1));old.append(dict(run=folder.name,source=path.stem[:-6],scope='retrospective for final targeted rule',**m))
    pd.DataFrame(old).to_csv(run/'earlier_block_reanalysis.csv',index=False)
    manifest=read(run/'models_manifest.json');r=next(r for r in manifest if r['key']=='weighted_reference');manifest.append({**r,'key':KEY,'creator':'Owned fitted experts with frozen targeted SQL guard + RF rescue; zero fits'});write(run/'models_manifest.json',manifest)
    expected=loadnp(run/'profile_expected.npz')
    for cohort in ['V1000','later']:expected[cohort+'_candidate']=combine(expected[cohort+'_WV'],expected[cohort+'_RF'])[0]
    np.savez_compressed(run/'profile_expected.npz',**expected);event(run,'THIRD_BLOCK_FINAL_CANDIDATE_COMPLETE')

def probability(model,policy,X):
    if policy=='random_forest':return model.predict_proba(X)
    return infer(model,X,NAMES) if policy==KEY else full_voting(model,X,NAMES)
def memory(run,policy,cohort,rep):
    verify_run_source(run);model=load_exported(run,policy);X=profile_data(run)[cohort];key='_candidate' if policy==KEY else '_RF' if policy=='random_forest' else '_WV';gold=loadnp(run/'profile_expected.npz')[cohort+key]
    gc.collect();proc=psutil.Process();base=proc.memory_info().rss;peak=[base];stop=threading.Event()
    def sample():
        while not stop.wait(.001):peak.append(proc.memory_info().rss)
    worker=threading.Thread(target=sample,daemon=True);worker.start();p=probability(model,policy,X);peak.append(proc.memory_info().rss);stop.set();worker.join();np.testing.assert_allclose(p,gold,rtol=0,atol=1e-12)
    print(json.dumps(dict(policy=policy,cohort=cohort,repeat=rep,n=len(X),baseline_RSS_MiB=base/2**20,delta_peak_RSS_MiB=(max(peak)-base)/2**20,expected_probabilities_match=True)))
def profile(run):
    assert not (run/'complete.json').exists();verify_run_source(run);data=profile_data(run);names=['random_forest','weighted_reference',KEY];models={n:load_exported(run,n) for n in names};expected=loadnp(run/'profile_expected.npz');times=[];rng=np.random.default_rng(20261109);maxdiff={}
    for cohort,X in data.items():
        for n in names:
            for _ in range(3):probability(models[n],n,X)
        for rep in range(30):
            for n in rng.permutation(names):
                start=time.perf_counter();p=probability(models[n],n,X);ms=1000*(time.perf_counter()-start);key='_candidate' if n==KEY else '_RF' if n=='random_forest' else '_WV';gold=expected[cohort+key];np.testing.assert_allclose(p,gold,rtol=0,atol=1e-12);maxdiff[cohort+'__'+n]=max(maxdiff.get(cohort+'__'+n,0.),float(abs(p-gold).max()));times.append(dict(cohort=cohort,policy=n,repeat=rep,n=len(X),milliseconds=ms))
        event(run,'LATENCY_COMPLETE '+cohort)
    pd.DataFrame(times).to_csv(run/'latency_raw.csv',index=False);write(run/'profile_probability_maxdiff.json',maxdiff);rss=[]
    for cohort in data:
        for n in names:
            for rep in range(3):
                p=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--phase','memory','--run',str(run),'--policy',n,'--cohort',cohort,'--repeat',str(rep)],cwd=ROOT,text=True,capture_output=True);assert p.returncode==0,p.stdout+p.stderr;rss.append(json.loads(p.stdout.strip().splitlines()[-1]))
            event(run,'RSS_COMPLETE '+cohort+' '+n)
    pd.DataFrame(rss).to_csv(run/'memory_raw.csv',index=False)
    a=pd.DataFrame(times).groupby(['cohort','policy']).milliseconds.agg(['count','mean','median','std']).reset_index();b=pd.DataFrame(rss).groupby(['cohort','policy']).delta_peak_RSS_MiB.agg(['mean','std']).rename(columns={'mean':'RSS_delta_mean_MiB','std':'RSS_delta_SD_MiB'}).reset_index();a.merge(b,on=['cohort','policy']).to_csv(run/'efficiency_summary.csv',index=False)
    write(run/'artifacts_sha256.json',[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ['artifacts_sha256.json','complete.json']]);write(run/'complete.json',dict(status='FPR_SQL_GUARD_COMPLETE',catalog_sha256=e.sha256(run/'artifacts_sha256.json'),new_classifier_fits=0,primary=KEY,all_seven_cached_gates_passed=True,third_block_frozen_before_scoring=True));print('FPR_SQL_GUARD_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['setup','fresh','profile','memory'],required=True);ap.add_argument('--run',type=Path,default=RUN);ap.add_argument('--policy');ap.add_argument('--cohort');ap.add_argument('--repeat',type=int,default=0);a=ap.parse_args();r=a.run.resolve()
    with threadpool_limits(limits=1):
        if a.phase=='setup':setup(r)
        elif a.phase=='fresh':fresh(r);decorate(r)
        elif a.phase=='profile':profile(r)
        else:memory(r,a.policy,a.cohort,a.repeat)
