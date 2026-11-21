"""Post-hoc saved-output diagnosis and independent feature reconstruction; zero fits."""
from pathlib import Path
import sys,json,hashlib,itertools,collections,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from run_array_factorial_20261103 import read_npz,methods,checked,OLD
from run_array_history_models_20261103 import verify_data
from run_wb_research import load_data
from array_revision import frame
from e4 import RAW_COLUMNS,columns
from sbfe import iter_pcap,row_identity,META,TIME_RE
from wb_profile import LocalSource
OUT=ROOT/'reports/ARRAY_FINAL_REVIEW_20261104';OUT.mkdir(exist_ok=True)
BEN=ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04'
EXPERTS=['DT','RF','XGB','LGB','LR']

def context(records,t,src,dst):
    """Independent deque implementation of the documented 60-field definition."""
    vectors=[]
    for host in [src,dst]:
        v=[]
        for width in [100_000_000,1_000_000_000,10_000_000_000]:
            r=[p for p in records if p[0]>=t-width and host and (p[1]==host or p[2]==host)]
            if not r:v.extend([0.]*10);continue
            a=np.asarray(r,dtype=np.int64);sent=a[:,1]==host;tcp=a[:,7]==6
            peers=np.where(sent,a[:,2],a[:,1]);local=np.where(sent,a[:,3],a[:,4])[tcp];remote=np.where(sent,a[:,4],a[:,3])[tcp]
            v.extend([np.log1p(len(a)),np.log1p(a[:,5].sum()),np.log1p(len(set(peers))),np.log1p(len(set(local))),np.log1p(len(set(remote))),a[:,5].mean(),a[:,5].std(),np.mean((a[:,6]&2)!=0),np.mean((a[:,6]&16)!=0),np.mean((a[:,6]&4)!=0)])
        vectors.append(np.array(v))
    vectors.sort(key=lambda v:tuple(v[20:])+tuple(v),reverse=True)
    return np.concatenate(vectors)

def main():
    verify_data(BEN)
    raw,y,holdout,data=load_data(dict(data_folder='runs/MULTISCALE_DATA_20261020_01'))
    X,num,cat=frame(raw,data['prior_multiscale'],'c60_o10')
    split=read_npz(checked(OLD,'stages/temporal_seed42/split.npz'));A=X.loc[split['A']];ay=pd.Series(y,index=X.index).loc[A.index].to_numpy()
    selection=json.loads(checked(OLD,'stages/temporal_seed42/legacy25_multiscale60__balanced/selection_before_V.json').read_text())
    rows=[];expertrows=[];contrib=[];allfeatures=[];alignment=[];idsall=[]
    for record in json.loads((BEN/'source_audit.json').read_text()):
        name=record['source'];pcap=Path(record['pcap']);csv=Path(record['csv'])
        assert e.sha256(pcap)==record['pcap_sha256'] and e.sha256(csv)==record['csv_sha256']
        z=read_npz(BEN/(name+'_predictions.npz'));p=z['p_experts'];w=z['weights'];ids=z['record_index'].astype(int)
        assert np.allclose(w,selection['weights']) and p.shape==(len(ids),5,5)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        if len(p):np.testing.assert_allclose(p.sum(2),1,atol=1e-12)
        probs=methods(p,w);wv=probs['weighted_voting'].argmax(1);u=probs['uniform_voting'].argmax(1);bad=wv!=0
        for method,prob in probs.items():
            pred=prob.argmax(1);rows.append(dict(source=name,method=method,n=len(ids),**{'pred_'+c:int((pred==i).sum()) for i,c in enumerate(e.CLASSES)}))
        for j,expert in enumerate(EXPERTS):
            pred=p[:,j].argmax(1);expertrows.append(dict(source=name,expert=expert,n=len(ids),false_alarms=int((pred!=0).sum()),sql_predictions=int((pred==3).sum()),mean_p_normal=float(p[:,j,0].mean()) if len(ids) else None,mean_p_sql=float(p[:,j,3].mean()) if len(ids) else None))
            if bad.any():contrib.append(dict(source=name,expert=expert,n=int(bad.sum()),weight=w[j],mean_p_sql_minus_normal=float((p[bad,j,3]-p[bad,j,0]).mean()),mean_weighted_sql_minus_normal=float((w[j]*(p[bad,j,3]-p[bad,j,0])).mean()),normal_argmax_on_wv_errors=int((pred[bad]==0).sum())))
        if len(ids):
            idsall.append(pd.DataFrame(dict(source=name,record_index=ids,wv_class=wv,uniform_class=u,lr_class=p[:,4].argmax(1),rf_class=p[:,1].argmax(1),wv_p_normal=probs['weighted_voting'][:,0],wv_p_sql=probs['weighted_voting'][:,3],margin_sql_normal=probs['weighted_voting'][:,3]-probs['weighted_voting'][:,0])))
            saved=pd.read_csv(BEN/(name+'_features.csv'),index_col='record_index',dtype={c:str for c in cat});assert list(saved.columns)==list(X.columns)
            rawcsv=pd.read_csv(csv,nrows=100000,usecols=list(dict.fromkeys(RAW_COLUMNS+META+['Attack_type','Attack_label'])),low_memory=False)
            assert (rawcsv.Attack_type=='Normal').all() and (rawcsv.Attack_label==0).all()
            mapping={int(k):int(v) for k,v in record['packet_to_csv_row'].items()};wanted=set(ids);history=collections.deque(maxlen=16384);values=[];starts=None;clock=None;offsets=[];duplicate_keys=collections.Counter();selected={};outorder=0
            # Inspect every prefix record; query before insertion, independent of production ring.
            for i,t,sec,nano,packet in itertools.islice(iter_pcap(LocalSource(),pcap),100000):
                if starts is None:starts=t
                if clock is not None and t<clock:outorder+=1
                clock=max(t,clock if clock is not None else t)
                key=packet.identity(sec,nano)
                if i in wanted:
                    row=rawcsv.iloc[mapping[i]];rkey,year,hour=row_identity(row);assert key==rkey
                    assert year==datetime.datetime.fromtimestamp(sec,datetime.timezone.utc).year
                    offsets.append((hour-(sec//3600)%24)%24)
                    assert clock-starts>=10_000_000_000
                    selected[key]=i;values.append(context(history,clock,packet.src,packet.dst))
                if key in selected:duplicate_keys[key]+=1
                while history and history[0][0]<clock-10_000_000_000:history.popleft()
                history.append((clock,packet.src,packet.dst,packet.sp,packet.dp,packet.wire,packet.flags,packet.protocol))
            assert len(set(offsets))==1 and len(values)==len(ids)
            assert max(duplicate_keys.values())==1  # No later duplicate of selected identity.
            # Also count earlier duplicates over all selected keys in an independent second prefix pass.
            counts=collections.Counter()
            for i,t,sec,nano,packet in itertools.islice(iter_pcap(LocalSource(),pcap),100000):
                key=packet.identity(sec,nano)
                if key in selected:counts[key]+=1
            assert all(counts[k]==1 for k in selected)
            aligned=rawcsv.iloc[[mapping[i] for i in ids]].copy();aligned.index=ids
            rebuilt,_,_=frame(aligned,np.asarray(values),'c60_o10')
            np.testing.assert_allclose(saved[num].to_numpy(),rebuilt[num].to_numpy(),rtol=0,atol=1e-12,equal_nan=True)
            assert saved[cat].equals(rebuilt[cat].astype(str))
            assert not saved[num].isna().any().any()
            allfeatures.append(saved.assign(source=name,wv_error=bad))
            alignment.append(dict(source=name,matched=len(ids),identity_fields=8,constant_hour_offset=offsets[0],unique_selected_identity=True,all_85_features_equal=True,numeric_max_abs_error=float(np.nanmax(np.abs(saved[num].to_numpy()-rebuilt[num].to_numpy()))),categorical_mismatches=0,backward_records=outorder,capacity_evictions=record['capacity_evictions'],csv_index_mismatch_count=int(sum(mapping[i]!=i for i in ids))))
            print('RECONSTRUCTED',name,len(ids),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'capture_predicted_classes.csv',index=False)
    pd.DataFrame(expertrows).to_csv(OUT/'expert_by_capture.csv',index=False)
    pd.DataFrame(contrib).to_csv(OUT/'weighted_sql_margin_by_capture.csv',index=False)
    pd.DataFrame(alignment).to_csv(OUT/'alignment_reconstruction.csv',index=False)
    decisions=pd.concat(idsall,ignore_index=True);decisions.to_csv(OUT/'decision_diagnosis.csv',index=False)
    B=pd.concat(allfeatures);shift=[];summaries=[]
    for c in num:
        train=A[c].to_numpy(float);lo=np.nanmin(train);hi=np.nanmax(train);q1,q3=np.nanquantile(train,[.25,.75]);scale=max(q3-q1,1e-12)
        for label,part in [('all',B),('false_alarm',B[B.wv_error]),('normal_prediction',B[~B.wv_error])]:
            v=part[c].to_numpy(float);shift.append(dict(feature=c,group=label,n=len(v),train_min=lo,train_max=hi,train_median=float(np.nanmedian(train)),target_median=float(np.nanmedian(v)),outside_training_range=int(((v<lo)|(v>hi)).sum()),outside_fraction=float(((v<lo)|(v>hi)).mean()),scaled_median_shift=float(abs(np.nanmedian(v)-np.nanmedian(train))/scale)))
    pd.DataFrame(shift).to_csv(OUT/'numeric_distribution_shift.csv',index=False)
    unknown=[]
    for c in cat:
        known=set(A[c].astype(str));unknown.append(dict(feature=c,n=len(B),unknown=int((~B[c].isin(known)).sum()),unknown_false_alarm=int((~B.loc[B.wv_error,c].isin(known)).sum()),training_categories=len(known)))
    pd.DataFrame(unknown).to_csv(OUT/'categorical_unknowns.csv',index=False)
    # Reference source/class profiles use only original A rows, never synthetic or holdout.
    selected=['multiscale_higher_10000ms_log_packets','multiscale_higher_10000ms_mean_bytes','multiscale_higher_10000ms_log_local_tcp_ports','multiscale_higher_10000ms_rst_fraction','multiscale_higher_1000ms_log_packets','mqtt.len','tcp.len']
    for c in selected:
        for cl in [0,3]:summaries.append(dict(feature=c,group='original_A_'+e.CLASSES[cl],n=int((ay==cl).sum()),median=float(A.loc[ay==cl,c].median()),q25=float(A.loc[ay==cl,c].quantile(.25)),q75=float(A.loc[ay==cl,c].quantile(.75))))
        for label,g in [('benign_false_alarm',B[B.wv_error]),('benign_normal_prediction',B[~B.wv_error])]:summaries.append(dict(feature=c,group=label,n=len(g),median=float(g[c].median()),q25=float(g[c].quantile(.25)),q75=float(g[c].quantile(.75))))
    pd.DataFrame(summaries).to_csv(OUT/'feature_profiles.csv',index=False)
    ep=pd.DataFrame(expertrows).groupby('expert')[['n','false_alarms','sql_predictions']].sum().reset_index();ep.to_csv(OUT/'expert_summary.csv',index=False)
    cr=pd.DataFrame(contrib);cr['total_weighted_margin']=cr.n*cr.mean_weighted_sql_minus_normal
    margins=cr.groupby('expert').agg(n=('n','sum'),total=('total_weighted_margin','sum'));margins['mean_weighted_sql_minus_normal']=margins.total/margins.n;margins.to_csv(OUT/'expert_error_margin_summary.csv')
    info=dict(status='PASS_SAVED_OUTPUT_AND_INDEPENDENT_FEATURE_RECONSTRUCTION',new_fits=0,new_model_loads=0,new_predictions=0,matched=760,unscorable=107,weighted_false_alarms=int((decisions.wv_class!=0).sum()),weighted_wrong_class_counts=decisions.loc[decisions.wv_class!=0,'wv_class'].value_counts().to_dict(),weighted_vs_uniform_extra_alarms=int(((decisions.wv_class!=0)&(decisions.uniform_class==0)).sum()),weighted_vs_uniform_recovered=int(((decisions.wv_class==0)&(decisions.uniform_class!=0)).sum()),weighted_errors_also_lr_errors=int(((decisions.wv_class!=0)&(decisions.lr_class!=0)).sum()),all85features_reconstructed=True,matched_sources=8,full_alignment_validated=False,all_extra_capture_capacity_evictions_zero=all(r['capacity_evictions']==0 for r in json.loads((BEN/'source_audit.json').read_text())),limitations=['Post-hoc diagnosis on previous fixed outputs; no causal feature intervention','107 eligible candidates remain unscorable','Legacy protocol fields are verified against supplied CSV, not independently extracted by a fresh dissector','No new five-class generalization test'])
    e.write_json(OUT/'diagnosis.json',info)
    e.write_json(OUT/'catalog.json',[dict(path=p.name,bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='catalog.json'])
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
