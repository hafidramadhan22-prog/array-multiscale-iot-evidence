"""Independent count/probability/label audit; no deserialization or fitting."""
from pathlib import Path
import argparse,collections,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score,recall_score,fbeta_score
import ta_hafid_engine as e
RUN=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01';OUT=ROOT/'reports/INDEPENDENT_CAPTURE_20261109';KEY='sql_guard_rf_rescue';CLASSES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware'];SOURCES=['CIC_Normal0','CIC_Normal3','CIC_SQL','CIC_PortScan','CIC_DDoS_SYN','TON_Ransom1','TON_Ransom2']
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):p.write_text(json.dumps(e.jsonable(x),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def load(p):
 with np.load(p,allow_pickle=False) as z:return dict(z)
def score(y,p):
 a=p.argmax(1);cm=confusion_matrix(y,a,labels=range(5));n=cm.sum(1);result=dict(n=len(y),normal_support=int(n[0]),attack_support=int(n[1:].sum()),false_alarms=int(cm[0,1:].sum()),attack_to_Normal=int(cm[1:,0].sum()),errors=int((y!=a).sum()),FPR_percent=100*cm[0,1:].sum()/n[0] if n[0] else None,attack_recall_percent=100*(1-cm[1:,0].sum()/n[1:].sum()) if n[1:].sum() else None,macro_F1_percent=100*f1_score(y,a,labels=range(5),average='macro',zero_division=0) if (n>0).all() else None,balanced_accuracy_percent=100*balanced_accuracy_score(y,a) if (n>0).all() else None)
 for i,c in enumerate(CLASSES):result['support_'+c]=int(n[i]);result['recall_'+c]=100*cm[i,i]/n[i] if n[i] else None
 result['DDoS_F2_percent']=100*fbeta_score(y,a,beta=2,labels=[1],average=None,zero_division=0)[0] if n[1] else None
 return result,cm
def audit():
 assert all((RUN/s/'scored.json').exists() for s in SOURCES);rows=[];decisions=[];ids=set();source_checks=[];coverage=[];all_frames=[]
 manifests=[RUN/'source_manifest.json',RUN/'execution_source_manifest_v2.json',RUN/'ton_execution_source_manifest.json',RUN/'syn_resume_source_manifest.json',RUN/'flow_preserving_execution/source_manifest.json',RUN/'TON_flow_mapping/source_manifest.json',RUN/'TON_flow_mapping/fast_source_manifest.json']
 for manifest in manifests:
  x=read(manifest);records=x if isinstance(x,list) else [x]
  for row in records:assert e.sha256(ROOT/row['path'])==row['sha256'],row['path']
 assert read(RUN/'flow_preserving_execution/CIC_Normal3/validation.json')['status']=='PASS_85_FEATURES_EXACT'
 gt=pd.read_csv(RUN/'TON_flow_mapping/matching_author_flows.csv',dtype={'start_ns':np.int64,'end_ns':np.int64});gtindex=collections.defaultdict(list)
 for i,r in enumerate(gt.itertuples()):gtindex[(r.proto,r.ip_low,r.port_low,r.ip_high,r.port_high)].append((r.start_ns,r.end_ns,r.type,i))
 for source in SOURCES:
  folder=RUN/source;prep=read(folder/'preparation.json');assert e.sha256(folder/'features.csv')==prep['feature_sha256'];X=pd.read_csv(folder/'features.csv',index_col='row_id',dtype=str,keep_default_na=False);assert X.shape[1]==85 and not set(X.index)&ids;ids.update(X.index);all_frames.append(pd.read_csv(folder/'features.csv',dtype=str,keep_default_na=False))
  z=load(folder/'probabilities.npz');np.testing.assert_array_equal(z['row_ids'],X.index);y=z['y_scenario'] if 'y_scenario' in z else z['y'];assert len(y)==len(X);v=z['weighted_reference'];rf=z['random_forest'];vl=v.argmax(1);rl=rf.argmax(1);new=v.copy();use=((vl==3)&(rl==0))|((vl==0)&(rl!=0));new[use]=rf[use];np.testing.assert_allclose(new,z[KEY],rtol=0,atol=1e-12)
  saved=pd.read_csv(folder/'metrics.csv').set_index('policy')
  scope='author_flow_derived' if source.startswith('TON') else read(folder/'scored.json')['label_scope']
  if source.startswith('TON'):
   truth=pd.read_csv(folder/'truth_and_packet_indices.csv',index_col='row_id').loc[X.index];np.testing.assert_array_equal(y,truth.label);assert set(y)=={0,4};original=read(RUN/'TON_flow_mapping/packet_index_complete.json');name='normal_runsomware1' if source.endswith('1') else 'normal_runsomware2';packets=np.load(RUN/'TON_flow_mapping'/(name+'_packets.npy'),mmap_mode='r',allow_pickle=False)
   for tr in truth.itertuples():
    p=packets[int(tr.packet_index)];assert p[1]==tr.epoch_ns;a=(int(p[3]),int(p[4]));b=(int(p[5]),int(p[6]));key=(int(p[2]),*min(a,b),*max(a,b));matches=[r for r in gtindex[key] if r[0]<=p[1]<r[1]];assert {r[2] for r in matches}=={tr.truth_type};assert {r[3] for r in matches}==set(map(int,str(tr.flow_record_indices).split(';')))
  for policy,p in z.items():
   if not isinstance(p,np.ndarray) or p.shape!=(len(y),5):continue
   assert np.isfinite(p).all() and (p>=0).all();np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-6);m,cm=score(y,p);rows.append(dict(source=source,policy=policy,label_scope=scope,**m))
   if policy in saved.index:
    for k,val in m.items():
     if val is None:assert pd.isna(saved.loc[policy,k])
     else:assert abs(float(saved.loc[policy,k])-val)<1e-10,(source,policy,k)
   if policy not in ['weighted_reference','random_forest',KEY]:continue
   cm_saved=pd.read_csv(folder/('CM_'+policy+'.csv'),index_col=0).to_numpy();np.testing.assert_array_equal(cm,cm_saved)
   for rid,yy,qq in zip(X.index,y,p):decisions.append(dict(source=source,row_id=rid,truth=CLASSES[int(yy)],label_scope=scope,policy=policy,prediction=CLASSES[int(qq.argmax())],**{'p_'+c:float(qq[k]) for k,c in enumerate(CLASSES)}))
  source_checks.append(dict(source=source,n=len(X),feature_sha256=e.sha256(folder/'features.csv'),posterior_sha256=e.sha256(folder/'probabilities.npz'),label_scope=scope))
 pd.DataFrame(rows).to_csv(OUT/'all_policy_metrics_verified.csv',index=False);D=pd.DataFrame(decisions);D.to_csv(OUT/'all_predictions_verified.csv',index=False);pd.concat(all_frames).to_csv(OUT/'all_external_features.csv',index=False)
 main=pd.DataFrame(rows).query("policy in ['weighted_reference','random_forest','sql_guard_rf_rescue']");main.to_csv(OUT/'primary_policy_metrics_verified.csv',index=False)
 pooled=[]
 for family in ['CIC','TON']:
  for policy in ['weighted_reference','random_forest',KEY]:
   a=D.query('source.str.startswith(@family) and policy==@policy',engine='python');normal=a.query("truth=='Normal'");fp=int((normal.prediction!='Normal').sum());attack=a.query("truth!='Normal'");miss=int((attack.prediction=='Normal').sum());pooled.append(dict(dataset=family,policy=policy,Normal_n=len(normal),Normal_FP=fp,FPR_percent=100*fp/len(normal),attack_or_scenario_n=len(attack),attack_or_scenario_to_Normal=miss,pooled_descriptive_not_IID=True))
 pd.DataFrame(pooled).to_csv(OUT/'dataset_summaries.csv',index=False)
 mapping=read(RUN/'TON_flow_mapping/mapping_complete.json')
 for src in mapping['sources']:
  c=src['mapping_counts'];coverage.append(dict(source=src['name'],total=src['total_packets'],normal=c.get('normal',0),ransomware=c.get('ransomware',0),warmup=c.get('warmup',0),unmapped=c.get('unmapped',0),ambiguous=c.get('ambiguous_labels',0),in_scope_labelled_percent=100*(c.get('normal',0)+c.get('ransomware',0))/src['total_packets']))
 pd.DataFrame(coverage).to_csv(OUT/'TON_label_coverage.csv',index=False)
 for truth,source in [('Normal','CIC_Normal3'),('Ransomware','TON_Ransom1')]:
  a=D.query('source==@source and truth==@truth and policy==@KEY');a=a.query("prediction!='Normal'") if truth=='Normal' else a.query("prediction=='Normal'");a.head(10).to_csv(OUT/('examples_'+source+'.csv'),index=False)
 assert len(ids)==9000 and len(D)==27000
 result=dict(status='PASS_SAVED_RESULTS_AND_LABEL_MAPPING',unique_targets=len(ids),primary_policy_predictions=len(D),sources=source_checks,all_five_category_names_covered_across_different_datasets=True,single_independent_five_class_testbed=False,independent_FPR_target8_met=False,CIC_candidate_FPR_percent=31.9,TON_labels='derived author flow labels with quantized time; unmapped excluded',model_sha256='77aa1beb6980f4eb9db9838af6505949899c5aa574f3be49d55adfdc1efcf31d',new_fits=0,all_primary_metrics_independently_recomputed=True,all_4000_TON_labels_checked_against_all_matching_author_flows=True)
 write(OUT/'independent_verification.json',result);print(json.dumps(result,indent=2))
def seal():
 assert read(OUT/'independent_verification.json')['status']=='PASS_SAVED_RESULTS_AND_LABEL_MAPPING';assert not (RUN/'complete.json').exists();models=RUN/'models';models.mkdir();source=ROOT/'runs/FPR_SQL_GUARD_20261109_01/models/reference_five.joblib';expected='77aa1beb6980f4eb9db9838af6505949899c5aa574f3be49d55adfdc1efcf31d';assert e.sha256(source)==expected;shutil.copy2(source,models/source.name);write(RUN/'model_manifest.json',dict(path='models/reference_five.joblib',sha256=expected,bytes=source.stat().st_size,creator='Verified owned PAYLOAD_TARGET_REPAIR/FPR_SQL_GUARD fitted experts; no new fit'))
 write(RUN/'artifacts_sha256.json',[dict(path=p.relative_to(RUN).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name not in ['complete.json','artifacts_sha256.json']]);write(RUN/'complete.json',dict(status='INDEPENDENT_EVALUATION_COMPLETE_GENERALIZATION_TARGET_NOT_MET',catalog_sha256=e.sha256(RUN/'artifacts_sha256.json'),targets=9000,new_fits=0,model_sha256=expected,independent_target8_met=False));print('INDEPENDENT_RUN_SEALED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','seal'],required=True);a=p.parse_args()
 if a.phase=='audit':audit()
 else:seal()
