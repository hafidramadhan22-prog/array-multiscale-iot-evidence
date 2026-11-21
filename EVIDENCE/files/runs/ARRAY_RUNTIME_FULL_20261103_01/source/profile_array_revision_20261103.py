"""Isolated paired repeated runtime, resource and workload-size measurements."""
from pathlib import Path
import argparse,json,sys,time,gc,threading,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
import profile_exact_runtime as old
from wb_research import WBPredictor
from wb_profile import checked_file,LocalSource
from sbfe import iter_pcap
from prepare_array_history_20261103 import AuditedRing
SOURCE=ROOT/'runs/EXACT_RUNTIME_20261020_01'
POLICIES=['native_shared','uniform_voting','random_forest','logistic_regression','native_certified','compact_certified']

def load(mode,out=None):
    freeze=json.loads((SOURCE/'inputs_manifest.json').read_text());assert freeze['source_sha256']==e.sha256(Path(old.__file__))
    p='baseline25__'+('native_shared' if mode=='uniform_voting' else mode)
    predictor,frames,expected=old.load_policy(SOURCE,freeze,p)
    if mode=='uniform_voting':
        predictor=WBPredictor(predictor.models,np.ones(5)/5,policy='shared')
        # Labels are the uniform predictor's own contract, not forced ensemble agreement.
        if out is None:expected={k:predictor.predict(v) for k,v in frames.items()}
        else:
            with np.load(out/'uniform_expected.npz',allow_pickle=False) as z:expected={k:z[k] for k in z.files}
    return freeze,p,predictor,frames,expected

def worker(out,mode,index,workload):
    with threadpool_limits(limits=1):
        freeze,p,predictor,frames,expected=load(mode,out);gc.collect();proc=psutil.Process();base=proc.memory_info().rss;peaks=[base];stop=threading.Event();ready=threading.Event()
        def sample():
            peaks.append(proc.memory_info().rss);ready.set()
            while not stop.wait(.002):peaks.append(proc.memory_info().rss)
        t=threading.Thread(target=sample,daemon=True);t.start();ready.wait()
        if workload=='replay':old.replay(SOURCE,freeze,p,predictor,frames,expected)
        else:np.testing.assert_array_equal(predictor.predict(frames['1000']),expected['1000'])
        peaks.append(proc.memory_info().rss);stop.set();t.join()
    e.write_json(out/f'rss_{mode}_{workload}_{index}.json',dict(policy=mode,workload=workload,repeat=index,baseline_mib=base/2**20,peak_mib=max(peaks)/2**20,delta_mib=(max(peaks)-base)/2**20,samples=len(peaks)))

def main(out):
    assert not out.exists()
    # Explicitly refuse profiling while new experiment jobs are incomplete.
    for run in ['ARRAY_FACTORIAL_FULL_20261103_01','ARRAY_HISTORY_FULL_20261103_01','ARRAY_XAI_FULL_20261103_01','ARRAY_BUFFER_FULL_20261103_01']:
        assert (ROOT/'runs'/run/'complete.json').exists(),run
    out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source'/Path(__file__).name)
    e.write_json(out/'configuration_freeze.json',dict(seed=20261103,repeats=30,warmups=5,bootstrap=10000,cold_rss_repeats=5,interval_seconds=.002,source_manifest_sha256=e.sha256(SOURCE/'inputs_manifest.json'),same_machine_only=True))
    e.write_json(out/'environment.json',e.environment());rng=np.random.default_rng(20261103);raw=[];replays=[];scaling=[];buffer=[]
    with threadpool_limits(limits=1):
        loaded={m:load(m) for m in POLICIES}
        np.savez_compressed(out/'uniform_expected.npz',**loaded['uniform_voting'][4])
        freeze=loaded['native_shared'][0]
        for r in freeze['files']:checked_file(SOURCE,r)
        for mode,(_,p,obj,frames,expected) in loaded.items():
            for batch in ['1','1000']:
                for _ in range(5):np.testing.assert_array_equal(obj.predict(frames[batch]),expected[batch])
        for rep in range(30):
            for mode in rng.permutation(POLICIES):
                _,p,obj,frames,expected=loaded[mode]
                for batch in ['1','1000']:
                    t=time.perf_counter();pred=obj.predict(frames[batch]);elapsed=time.perf_counter()-t
                    np.testing.assert_array_equal(pred,expected[batch]);raw.append(dict(policy=mode,batch=int(batch),repeat=rep,seconds=elapsed))
        pair=['native_certified','compact_certified']
        for mode in pair:
            f,p,obj,frames,expected=loaded[mode];old.replay(SOURCE,f,p,obj,frames,expected,verify=True)
            for _ in range(5):old.replay(SOURCE,f,p,obj,frames,expected)
        for rep in range(30):
            for position,mode in enumerate(rng.permutation(pair)):
                f,p,obj,frames,expected=loaded[mode];cpu=time.process_time();r=old.replay(SOURCE,f,p,obj,frames,expected)
                replays.append(dict(**r,mode=mode,repeat=rep,position=position,cpu_seconds=time.process_time()-cpu))
            print('PAIRED_REPLAY',rep+1,flush=True)
        # Fixed segments are repeated with fresh histories, not invented traffic.
        for copies in [1,2,4]:
            for rep in range(5):
                for mode in rng.permutation(pair):
                    f,p,obj,frames,expected=loaded[mode];t=time.perf_counter()
                    for _ in range(copies):old.replay(SOURCE,f,p,obj,frames,expected)
                    scaling.append(dict(mode=mode,copies=copies,repeat=rep,seconds=time.perf_counter()-t,packets=15665*copies,decisions=160*copies))
        work=json.loads((SOURCE/'replay/workload.json').read_text())
        # Isolate actual ring query/update costs on the same fixed replay, all capacities.
        for rep in range(10):
            for cap in rng.permutation([4096,16384,65536]):
                t=time.perf_counter();q=u=0.;queries=0
                for item in work['files']:
                    ring=AuditedRing(int(cap))
                    for index,stamp,_,_,packet in iter_pcap(LocalSource(),SOURCE/'replay'/Path(item['file']).name):
                        if str(index) in item['targets']:
                            tick=time.perf_counter();ring.context(stamp,packet.src,packet.dst);q+=time.perf_counter()-tick;queries+=1
                        tick=time.perf_counter();ring.push(stamp,packet);u+=time.perf_counter()-tick
                buffer.append(dict(capacity=int(cap),repeat=rep,total_seconds=time.perf_counter()-t,query_seconds=q,update_seconds=u,decisions=queries))
    pd.DataFrame(raw).to_csv(out/'latency_raw.csv',index=False);pd.DataFrame(replays).to_csv(out/'replay_raw.csv',index=False);pd.DataFrame(scaling).to_csv(out/'scaling_raw.csv',index=False);pd.DataFrame(buffer).to_csv(out/'buffer_latency_raw.csv',index=False)
    d=pd.DataFrame(replays).pivot(index='repeat',columns='mode',values='total_seconds');delta=(d.native_certified-d.compact_certified).to_numpy();boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
    e.write_json(out/'paired_summary.json',dict(mean_seconds=float(delta.mean()),median_seconds=float(np.median(delta)),sd_seconds=float(delta.std(ddof=1)),ci95_mean_seconds=np.quantile(boot,[.025,.975]).tolist(),compact_slower=int((delta<0).sum()),repeats=len(delta),mean_relative_reduction=float(1-d.compact_certified.mean()/d.native_certified.mean())))
    for mode in POLICIES:
        for workload in (['feature','replay'] if mode in pair else ['feature']):
            for index in range(5):
                with (out/f'rss_{mode}_{workload}_{index}.log').open('w') as log:
                    subprocess.run([sys.executable,'-B',str(Path(__file__)),'--output',str(out),'--worker',mode,'--index',str(index),'--workload',workload],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COLD_RSS',mode,flush=True)
    pd.DataFrame([json.loads(p.read_text()) for p in out.glob('rss_*.json')]).to_csv(out/'memory_raw.csv',index=False)
    e.write_json(out/'catalog.json',[dict(path=p.relative_to(out).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
    e.write_json(out/'complete.json',dict(status='PASS_RUNTIME',paired_repeats=30,all_labels_match=True,catalog_sha256=e.sha256(out/'catalog.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker');p.add_argument('--index',type=int);p.add_argument('--workload');a=p.parse_args()
    if a.worker:worker(a.output.resolve(),a.worker,a.index,a.workload)
    else:main(a.output.resolve())
