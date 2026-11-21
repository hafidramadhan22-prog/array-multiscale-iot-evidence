"""Existing predictors, explicit safety gates, and fixed later-packet benign block."""
from pathlib import Path
import argparse,csv,gc,hashlib,itertools,json,shutil,subprocess,sys,threading,time,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib,psutil
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from fpr_target8_20261109 import alternatives,metrics,safety_gate,NAMES,CLASSES
from array_revision import frame
from e4 import RAW_COLUMNS,columns
from sbfe import iter_pcap,row_identity,TIME_RE
from wb_profile import LocalSource
from prepare_array_history_20261103 import AuditedRing
from audit_array_flag_format_20261104 import semantic_flag,FIELDS

PARENT=ROOT/'runs/PAYLOAD_TARGET_REPAIR_20261105_01'
FACT=ROOT/'runs/ARRAY_FACTORIAL_FULL_20261103_01'
BEN=ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04'
FLAG=ROOT/'runs/ARRAY_FLAG_FORMAT_20261104_01'
SOURCES=['scripts/run_fpr_target8_20261109.py','src/fpr_target8_20261109.py','docs/FPR_TARGET8_PROTOCOL_20261109.md','src/array_revision.py','src/e4.py','src/sbfe.py','src/wb_behavior.py','src/multiscale_context.py','src/ta_hafid_engine.py','scripts/prepare_array_history_20261103.py','scripts/audit_array_flag_format_20261104.py']
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):e.write_json(p,x)
def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def event(run,s):
    print(s,flush=True)
    with (run/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),event=s))+'\n')
def checked(folder,rel,catalog='artifacts_sha256.json'):
    rows=read(folder/catalog);r=next(r for r in rows if r['path']==rel);p=folder/rel
    assert p.stat().st_size==r['bytes'] and e.sha256(p)==r['sha256'],str(p);return p
def verify_parent():
    assert e.sha256(PARENT/'artifacts_sha256.json')==read(PARENT/'complete.json')['catalog_sha256']
    for r in read(PARENT/'artifacts_sha256.json'):checked(PARENT,r['path'])
    for r in read(PARENT/'source_manifest.json'):assert e.sha256(ROOT/r['path'])==r['sha256'],r['path']
def verify_run_source(run):
    for r in read(run/'source_manifest.json'):assert e.sha256(ROOT/r['path'])==r['sha256'],r['path']
def reference_model():
    verify_parent();m=next(r for r in read(PARENT/'models_manifest.json') if r['key']=='reference')
    p=checked(PARENT,m['path']);assert e.sha256(p)==m['sha256'] and m['creator']=='PAYLOAD_TARGET_REPAIR_20261105'
    return joblib.load(p)

def cached(run):
    assert run.parent==(ROOT/'runs').resolve() and not run.exists();verify_parent()
    run.mkdir();(run/'source').mkdir();(run/'posteriors').mkdir();(run/'models').mkdir()
    sources=[r for r in read(BEN/'source_audit.json') if r['scored']>0];assert len(sources)==8
    times=pd.read_csv(ROOT/'reports/DATA_MENTAH_AUDIT_ZIP_20261019/pcap_times.csv');packet_counts={Path(r.entry).stem:int(r.packet_count) for r in times.itertuples()}
    cfg=dict(experiment='FPR_TARGET8',date='2026-11-09',target_FPR_percent=8.,primary='random_forest',primary_chosen_after_historical_results=True,primary_frozen_before_later_block=True,no_fit=True,no_new_captures=True,source_run=PARENT.relative_to(ROOT).as_posix(),candidate_policies=NAMES+['weighted_reference','uniform_reference','RF_XGB_50','RF_XGB_75'],mixture_weights_RF_XGB=[[.5,.5],[.75,.25]],feature_view='c60_o10',ring_capacity=16384,warmup_seconds=10,csv_prefix=300000,later_start_index=100999,later_end_exclusive=200000,later_step=1000,coverage_floor=.9,macro_F1_max_drop_pp=.1,sources=[dict(source=r['source'],pcap=r['pcap'],pcap_sha256=r['pcap_sha256'],csv=r['csv'],csv_sha256=r['csv_sha256'],packet_count=packet_counts[r['source']],planned_ids=list(range(100999,min(200000,packet_counts[r['source']]),1000))) for r in sources],new_classifier_fits=0)
    assert sum(len(r['planned_ids']) for r in cfg['sources'])==770
    write(run/'configuration_freeze.json',cfg);write(run/'environment.json',e.environment())
    manifest=[]
    for rel in SOURCES:shutil.copy2(ROOT/rel,run/'source'/Path(rel).name);manifest.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    write(run/'source_manifest.json',manifest)
    rows=[];gates=[];decisions=[];pooled={}
    def evaluate(group,z,cube,w):
        probs=alternatives(cube,w);y=z['y'];ids=z['row_ids'];reference=probs['weighted_reference'].argmax(1)
        np.savez_compressed(run/'posteriors'/(group+'.npz'),row_ids=ids,y=y,**probs)
        for name,p in probs.items():
            m,cm=metrics(y,p.argmax(1));rows.append(dict(cohort=group,policy=name,**m));gates.append(dict(cohort=group,policy=name,**safety_gate(y,reference,p.argmax(1))))
            if group in ['V_seed42','additional760']:decisions.extend(dict(cohort=group,row_id=str(i),policy=name,true_class=CLASSES[int(a)],prediction=CLASSES[int(b)],**{'p_'+c:float(q[j]) for j,c in enumerate(CLASSES)}) for i,a,b,q in zip(ids,y,p.argmax(1),p))
        return probs
    for cohort in ['G','C','V','additional']:
        z=loadnp(checked(PARENT,'posteriors/reference__'+cohort+'.npz'));group='V_seed42' if cohort=='V' else 'additional760' if cohort=='additional' else cohort
        cube=np.stack([z[n] for n in NAMES],axis=1);evaluate(group,z,cube,read(PARENT/'configuration_freeze.json')['weights'])
    for stage in ['temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']:
        z=loadnp(checked(FACT,'stages/'+stage+'/c60_o10/evaluation.npz'));evaluate(stage,z,z['p_experts'],z['weights'])
        if stage.startswith('signature'):pooled[stage]=z
    z1,z2=pooled.values();assert not set(z1['row_ids'])&set(z2['row_ids'])
    # Each fold has its own fixed weights: pool predicted probabilities, never average fold scores.
    ps=[alternatives(z['p_experts'],z['weights']) for z in [z1,z2]]
    for name in ps[0]:
        y=np.r_[z1['y'],z2['y']];p=np.vstack([ps[0][name],ps[1][name]]);m,cm=metrics(y,p.argmax(1));rows.append(dict(cohort='signature_pooled',policy=name,**m))
    pd.DataFrame(rows).to_csv(run/'cached_metrics.csv',index=False);write(run/'cached_safety_gates.json',gates);pd.DataFrame(decisions).to_csv(run/'cached_decisions.csv',index=False)
    write(run/'cached_complete.json',dict(status='CACHED_DIAGNOSIS_COMPLETE',later_block_scored=False,primary_frozen='random_forest',config_sha256=e.sha256(run/'configuration_freeze.json')))
    event(run,'CACHED_COMPARISON_COMPLETE; PRIMARY_RF_FROZEN_BEFORE_LATER_BLOCK')
    print(pd.DataFrame(rows).query("cohort in ['V_seed42','additional760']")[["cohort","policy","false_alarms","attack_to_Normal","FPR_percent","macro_F1_percent"]].to_string(index=False),flush=True)

def fresh(run):
    verify_run_source(run);assert read(run/'cached_complete.json')['status']=='CACHED_DIAGNOSIS_COMPLETE';assert not (run/'fresh_complete.json').exists()
    cfg=read(run/'configuration_freeze.json');bundle=reference_model();cfgparent=read(PARENT/'configuration_freeze.json')
    rawcols=list(dict.fromkeys(RAW_COLUMNS+['frame.time','ip.src_host','ip.dst_host','tcp.checksum','Attack_label','Attack_type','tcp.payload','mqtt.msg']))
    cols=cfgparent['base_numeric']+cfgparent['base_categorical'];allbase=[];met=[];audit=[];decisions=[];expected={};sealed_models=[]
    # Serialize the already-fitted RF pipeline only; no .fit call occurs here.
    rfpath=run/'models/rf_context85.joblib';joblib.dump(bundle['models']['random_forest'],rfpath,compress=1)
    original=next(r for r in read(PARENT/'models_manifest.json') if r['key']=='reference');refpath=run/'models/reference_five.joblib';shutil.copy2(PARENT/original['path'],refpath)
    for key,path in [('random_forest',rfpath),('weighted_reference',refpath)]:sealed_models.append(dict(key=key,path=path.relative_to(run).as_posix(),sha256=e.sha256(path),bytes=path.stat().st_size,creator='FPR_TARGET8; export of verified owned fitted pipelines; zero fits'))
    write(run/'models_manifest.json',sealed_models)
    for src in cfg['sources']:
        name=src['source'];pcap=Path(src['pcap']);csvpath=Path(src['csv'])
        assert e.sha256(pcap)==src['pcap_sha256'] and e.sha256(csvpath)==src['csv_sha256'],name
        event(run,'SOURCE_HASHES_VERIFIED '+name)
        source=next(r for r in read(BEN/'source_audit.json') if r['source']==name);oldids=set(map(int,source['packet_to_csv_row']))
        newids=set(src['planned_ids']);wanted=oldids|newids;keys={};contexts={};elapsed={};excluded=[];ring=AuditedRing(16384);clock=None;start=None
        for index,t,sec,nano,p in itertools.islice(iter_pcap(LocalSource(),pcap),min(cfg['later_end_exclusive'],src['packet_count'])):
            if start is None:start=t
            clock=max(t,clock if clock is not None else t)
            if index in wanted:
                key=p.identity(sec,nano);assert key not in keys,(name,'ambiguous candidate packet identity');keys[key]=index
                if clock-start>=10_000_000_000:
                    contexts[index]=ring.context(clock,p.src,p.dst);elapsed[index]=(clock-start)/1e9
                else:excluded.append(index)
            ring.push(clock,p)
        raw=pd.read_csv(csvpath,nrows=cfg['csv_prefix'],usecols=rawcols,low_memory=False)
        assert (raw.Attack_type=='Normal').all() and (raw.Attack_label==0).all()
        times={k[:2] for k in keys};mapped={};duplicates=0
        for j,text in enumerate(raw['frame.time']):
            match=TIME_RE.search(str(text))
            if match is None:continue
            _,_,minute,second,fraction=match.groups()
            if (int(minute)*60+int(second),int(fraction.ljust(9,'0'))) not in times:continue
            key,_,_=row_identity(raw.iloc[j])
            if key not in keys:continue
            ix=keys[key]
            if ix in mapped:
                pd.testing.assert_series_equal(raw.iloc[j][RAW_COLUMNS],raw.iloc[mapped[ix]][RAW_COLUMNS],check_names=False);duplicates+=1
            else:mapped[ix]=j
        def features(ids):
            aligned=raw.iloc[[mapped[i] for i in ids]].copy();aligned.index=ids
            X,_,_=frame(aligned,np.asarray([contexts[i] for i in ids]),'c60_o10')
            for c in FIELDS:X[c]=X[c].map(semantic_flag)
            assert set(X)==set(cols) and not set(X)&{'Attack_type','Attack_label','source','row_id','tcp.payload','mqtt.msg'}
            return X[cols],aligned
        previous=loadnp(checked(FLAG,name+'_predictions.npz','catalog.json'));oldlist=list(map(int,previous['record_index']))
        assert set(oldlist)<=mapped.keys() and set(oldlist)<=contexts.keys()
        xo,_=features(oldlist);gold=pd.read_csv(FLAG/(name+'_features.csv'),index_col='record_index',dtype={c:str for c in cfgparent['base_categorical']},float_precision='round_trip')
        for c in cfgparent['base_categorical']:np.testing.assert_array_equal(xo[c],gold.loc[oldlist,c])
        np.testing.assert_allclose(xo[cfgparent['base_numeric']],gold.loc[oldlist,cfgparent['base_numeric']],rtol=0,atol=1e-12,equal_nan=True)
        oldcube=np.stack([bundle['models'][n].predict_proba(xo) for n in NAMES],axis=1);np.testing.assert_allclose(oldcube,previous['p_after'],rtol=0,atol=1e-12)
        ids=sorted(newids & contexts.keys() & mapped.keys());assert not set(ids)&oldids
        missing=sorted(newids-set(ids));X,aligned=features(ids);rowids=[name+':'+str(i) for i in ids];X.index=rowids;X.index.name='row_id'
        X.to_csv(run/(name+'_later_features.csv'));aligned.to_csv(run/(name+'_later_raw.csv'),index_label='packet_index')
        cube=np.stack([bundle['models'][n].predict_proba(X) for n in NAMES],axis=1);probabilities=alternatives(cube,bundle['weights'])
        np.savez_compressed(run/'posteriors'/(name+'_later.npz'),row_ids=np.array(rowids),y=np.zeros(len(ids),dtype=int),**probabilities)
        for policy,prob in probabilities.items():
            m,cm=metrics(np.zeros(len(ids),int),prob.argmax(1));fp=m['false_alarms'];met.append(dict(source=name,policy=policy,planned=len(newids),scored=len(ids),unscorable=len(missing),coverage=len(ids)/len(newids),FPR_lower_percent=100*fp/len(newids),FPR_upper_percent=100*(fp+len(missing))/len(newids),**m))
            decisions.extend(dict(source=name,packet_index=i,csv_record_index=mapped[i],row_id=rid,policy=policy,true_class='Normal',prediction=CLASSES[int(pred)],**{'p_'+c:float(pp[k]) for k,c in enumerate(CLASSES)}) for i,rid,pred,pp in zip(ids,rowids,prob.argmax(1),prob))
        audit.append(dict(source=name,pcap_sha256=src['pcap_sha256'],csv_sha256=src['csv_sha256'],planned_ids=sorted(newids),scored_ids=ids,missing_or_excluded_ids=missing,excluded_before_warmup=excluded,packet_to_csv_row={str(i):mapped[i] for i in ids},old_targets_verified=len(oldlist),old_max_probability_difference=float(np.max(abs(oldcube-previous['p_after']))),duplicate_raw_rows_identical=duplicates,capacity_evictions=ring.evicted_capacity,history_query_before_current_push=True))
        allbase.append(X);event(run,'LATER_BLOCK_SCORED '+name+' n='+str(len(ids))+' RF_FP='+str(int((probabilities['random_forest'].argmax(1)!=0).sum())))
        pd.DataFrame(met).to_csv(run/'later_per_capture.csv',index=False);pd.DataFrame(decisions).to_csv(run/'later_decisions.csv',index=False);write(run/'mapping_audit.json',audit)
    both=pd.concat(allbase);both.to_csv(run/'all_later_features.csv');summary=[]
    D=pd.DataFrame(decisions)
    for policy,dd in D.groupby('policy',sort=False):
        m,cm=metrics(np.zeros(len(dd),int),np.array([CLASSES.index(x) for x in dd.prediction]));planned=sum(len(s['planned_ids']) for s in cfg['sources']);scored=len(dd)
        summary.append(dict(policy=policy,planned=planned,coverage=scored/planned,FPR_upper_percent=100*(m['false_alarms']+planned-scored)/planned,**m))
    pd.DataFrame(summary).to_csv(run/'later_summary.csv',index=False)
    # Profile using the exact same prepared feature rows for both implementations.
    inputs=joblib.load(checked(PARENT,read(PARENT/'input_manifest.json')['path']));v=inputs['parts']['V'];sample=e.sample_positions(inputs['y'][v],1000,20261109)
    prof={'V1000':inputs['base'].iloc[v[sample]][cols].copy(),'later':both}
    path=run/'profile_inputs.joblib';joblib.dump(prof,path,compress=1);write(run/'profile_input_manifest.json',dict(path=path.name,sha256=e.sha256(path),bytes=path.stat().st_size,creator='current verified FPR_TARGET8 run'))
    for cohort,base in prof.items():
        expected[cohort+'_RF']=bundle['models']['random_forest'].predict_proba(base)
        cb=np.stack([bundle['models'][n].predict_proba(base) for n in NAMES],axis=1);expected[cohort+'_WV']=alternatives(cb,bundle['weights'])['weighted_reference']
    np.savez_compressed(run/'profile_expected.npz',**expected)
    write(run/'fresh_complete.json',dict(status='FIXED_LATER_BLOCK_COMPLETE',primary='random_forest',new_fits=0,new_sources=0,planned=sum(len(s['planned_ids']) for s in cfg['sources']),scored=len(both),fresh_benign_only=True,attack_sensitivity_not_evaluable_on_later_block=True))
    event(run,'FRESH_NORMAL_BLOCK_COMPLETE')

def infer(model,policy,X):
    if policy=='random_forest':return model.predict_proba(X)
    return sum(float(w)*model['models'][n].predict_proba(X) for w,n in zip(model['weights'],NAMES))
def load_exported(run,policy):
    verify_run_source(run);r=next(r for r in read(run/'models_manifest.json') if r['key']==policy);p=run/r['path'];assert p.stat().st_size==r['bytes'] and e.sha256(p)==r['sha256'];return joblib.load(p)
def profile_data(run):
    r=read(run/'profile_input_manifest.json');p=run/r['path'];assert e.sha256(p)==r['sha256'] and p.stat().st_size==r['bytes'];return joblib.load(p)
def memory(run,policy,cohort,rep):
    model=load_exported(run,policy);data=profile_data(run)[cohort];expected=loadnp(run/'profile_expected.npz')[cohort+('_RF' if policy=='random_forest' else '_WV')]
    gc.collect();proc=psutil.Process();baseline=proc.memory_info().rss;peak=[baseline];stop=threading.Event()
    def watch():
        while not stop.wait(.001):peak.append(proc.memory_info().rss)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start();p=infer(model,policy,data);peak.append(proc.memory_info().rss);stop.set();watcher.join()
    np.testing.assert_array_equal(p.argmax(1),expected.argmax(1))
    print(json.dumps(dict(policy=policy,cohort=cohort,repeat=rep,n=len(data),baseline_RSS_MiB=baseline/2**20,delta_peak_RSS_MiB=(max(peak)-baseline)/2**20,expected_labels_match=True)))
def profile(run):
    assert read(run/'fresh_complete.json')['status']=='FIXED_LATER_BLOCK_COMPLETE';verify_run_source(run)
    models={n:load_exported(run,n) for n in ['random_forest','weighted_reference']};data=profile_data(run);expected=loadnp(run/'profile_expected.npz');times=[];rng=np.random.default_rng(20261109)
    for cohort,X in data.items():
        for name,model in models.items():
            for _ in range(3):infer(model,name,X)
        for rep in range(30):
            for name in rng.permutation(list(models)):
                start=time.perf_counter();p=infer(models[name],name,X);ms=1000*(time.perf_counter()-start);gold=expected[cohort+('_RF' if name=='random_forest' else '_WV')]
                np.testing.assert_array_equal(p.argmax(1),gold.argmax(1));times.append(dict(cohort=cohort,policy=name,repeat=rep,n=len(X),milliseconds=ms))
        event(run,'LATENCY_COMPLETE '+cohort)
    pd.DataFrame(times).to_csv(run/'latency_raw.csv',index=False);rss=[]
    for cohort in data:
        for name in models:
            for rep in range(3):
                result=subprocess.run([sys.executable,'-B',str(Path(__file__)), '--run',str(run),'--phase','memory','--policy',name,'--cohort',cohort,'--repeat',str(rep)],cwd=ROOT,text=True,capture_output=True)
                assert result.returncode==0,result.stdout+result.stderr;rss.append(json.loads(result.stdout.strip().splitlines()[-1]))
            event(run,'RSS_COMPLETE '+cohort+' '+name)
    pd.DataFrame(rss).to_csv(run/'memory_raw.csv',index=False)
    summary=pd.DataFrame(times).groupby(['cohort','policy']).milliseconds.agg(['count','mean','median','std']).reset_index()
    mm=pd.DataFrame(rss).groupby(['cohort','policy']).delta_peak_RSS_MiB.agg(['mean','std']).rename(columns={'mean':'RSS_delta_mean_MiB','std':'RSS_delta_SD_MiB'}).reset_index();summary.merge(mm,on=['cohort','policy']).to_csv(run/'efficiency_summary.csv',index=False)
    write(run/'artifacts_sha256.json',[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ['artifacts_sha256.json','complete.json','failure.json']])
    write(run/'complete.json',dict(status='FPR_TARGET8_COMPLETE',catalog_sha256=e.sha256(run/'artifacts_sha256.json'),new_classifier_fits=0,primary='random_forest',frozen_before_later_targets=True));print('FPR_TARGET8_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'runs/FPR_TARGET8_20261109_01');ap.add_argument('--phase',choices=['cached','fresh','profile','memory'],required=True);ap.add_argument('--policy');ap.add_argument('--cohort');ap.add_argument('--repeat',type=int,default=0);a=ap.parse_args();run=a.run.resolve()
    try:
        with threadpool_limits(limits=1):
            if a.phase=='cached':cached(run)
            elif a.phase=='fresh':fresh(run)
            elif a.phase=='profile':profile(run)
            else:memory(run,a.policy,a.cohort,a.repeat)
    except Exception:
        if run.exists() and a.phase!='memory':write(run/'failure.json',dict(phase=a.phase,traceback=traceback.format_exc()))
        raise
