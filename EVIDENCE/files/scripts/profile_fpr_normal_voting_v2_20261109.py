"""Versioned execution repair; no new fits, targets, thresholds or tolerance."""
from pathlib import Path
import argparse, gc, json, shutil, subprocess, sys, threading, time, traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from fpr_target8_20261109 import NAMES
from rf_normal_voting_type_v2_20261109 import infer,full_voting
from run_fpr_target8_20261109 import read,write,loadnp,load_exported,profile_data,event,verify_run_source
RUN=ROOT/'runs/FPR_NORMAL_VOTING_20261109_01'
EXTRA=['src/rf_normal_voting_type_v2_20261109.py','scripts/profile_fpr_normal_voting_v2_20261109.py']

def probability(model,policy,X):
    if policy=='random_forest':return model.predict_proba(X)
    return infer(model,X,NAMES) if policy=='rf_normal_then_voting' else full_voting(model,X,NAMES)

def check(run):
    verify_run_source(run)
    for r in read(run/'execution_source_manifest.json'):
        assert e.sha256(ROOT/r['path'])==r['sha256'],r['path']

def memory(run,policy,cohort,repeat):
    check(run);model=load_exported(run,policy);X=profile_data(run)[cohort]
    key='_hier' if policy=='rf_normal_then_voting' else '_RF' if policy=='random_forest' else '_WV'
    gold=loadnp(run/'profile_expected.npz')[cohort+key]
    gc.collect();process=psutil.Process();baseline=process.memory_info().rss;peak=[baseline];stop=threading.Event()
    def sample():
        while not stop.wait(.001):peak.append(process.memory_info().rss)
    worker=threading.Thread(target=sample,daemon=True);worker.start()
    p=probability(model,policy,X);peak.append(process.memory_info().rss);stop.set();worker.join()
    np.testing.assert_allclose(p,gold,rtol=0,atol=1e-12)
    print(json.dumps(dict(policy=policy,cohort=cohort,repeat=repeat,n=len(X),baseline_RSS_MiB=baseline/2**20,delta_peak_RSS_MiB=(max(peak)-baseline)/2**20,expected_probabilities_match=True)))

def profile(run):
    assert not (run/'complete.json').exists()
    assert read(run/'fresh_complete.json')['scored']==699
    rows=[]
    for rel in EXTRA:
        shutil.copy2(ROOT/rel,run/'source'/Path(rel).name)
        rows.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    write(run/'execution_source_manifest.json',rows)
    shutil.copy2(run/'failure.json',run/'profile_initial_failure.json')
    write(run/'execution_amendment.json',dict(reason='Initial full-probability gate exposed float32 weighted-product rounding, not a label or feature change',max_observed_initial_difference=1.00248714e-8,repair='Cast expert probabilities to float64 before weighted multiplication',probability_tolerance_unchanged=1e-12,threshold_changes=0,new_fits=0,new_targets=0,original_source_preserved=True))
    check(run);data=profile_data(run);names=['random_forest','weighted_reference','rf_normal_then_voting'];models={n:load_exported(run,n) for n in names};expected=loadnp(run/'profile_expected.npz');rng=np.random.default_rng(20261109);times=[];maxdiff={}
    for cohort,X in data.items():
        for name in names:
            for _ in range(3):probability(models[name],name,X)
        for rep in range(30):
            for name in rng.permutation(names):
                start=time.perf_counter();p=probability(models[name],name,X);ms=1000*(time.perf_counter()-start)
                key='_hier' if name=='rf_normal_then_voting' else '_RF' if name=='random_forest' else '_WV';gold=expected[cohort+key]
                np.testing.assert_allclose(p,gold,rtol=0,atol=1e-12)
                maxdiff[cohort+'__'+name]=max(maxdiff.get(cohort+'__'+name,0.),float(np.max(abs(p-gold))))
                times.append(dict(cohort=cohort,policy=name,repeat=rep,n=len(X),milliseconds=ms))
        event(run,'V2_LATENCY_COMPLETE '+cohort)
    pd.DataFrame(times).to_csv(run/'latency_raw.csv',index=False);write(run/'profile_probability_maxdiff.json',maxdiff);rss=[]
    for cohort in data:
        for name in names:
            for rep in range(3):
                result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--phase','memory','--run',str(run),'--policy',name,'--cohort',cohort,'--repeat',str(rep)],cwd=ROOT,text=True,capture_output=True)
                assert result.returncode==0,result.stdout+result.stderr
                rss.append(json.loads(result.stdout.strip().splitlines()[-1]))
            event(run,'V2_RSS_COMPLETE '+cohort+' '+name)
    pd.DataFrame(rss).to_csv(run/'memory_raw.csv',index=False)
    a=pd.DataFrame(times).groupby(['cohort','policy']).milliseconds.agg(['count','mean','median','std']).reset_index()
    b=pd.DataFrame(rss).groupby(['cohort','policy']).delta_peak_RSS_MiB.agg(['mean','std']).rename(columns={'mean':'RSS_delta_mean_MiB','std':'RSS_delta_SD_MiB'}).reset_index()
    a.merge(b,on=['cohort','policy']).to_csv(run/'efficiency_summary.csv',index=False)
    write(run/'artifacts_sha256.json',[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ['complete.json','artifacts_sha256.json']])
    write(run/'complete.json',dict(status='FPR_NORMAL_VOTING_COMPLETE',catalog_sha256=e.sha256(run/'artifacts_sha256.json'),new_classifier_fits=0,primary='rf_normal_then_voting',initial_profile_gate_failure_resolved_by_versioned_float64_execution=True))
    print('HIERARCHY_V2_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['profile','memory'],required=True);parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--policy');parser.add_argument('--cohort');parser.add_argument('--repeat',type=int,default=0);args=parser.parse_args();run=args.run.resolve()
    with threadpool_limits(limits=1):
        if args.phase=='profile':profile(run)
        else:memory(run,args.policy,args.cohort,args.repeat)
