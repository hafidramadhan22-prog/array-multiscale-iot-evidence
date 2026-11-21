"""Verify published WV/GNB evidence and unfiltered serving logs without model loading."""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,balanced_accuracy_score,fbeta_score
def metrics(y,p):
 return dict(F1=100*f1_score(y,p,labels=range(5),average='macro',zero_division=0),BA=100*balanced_accuracy_score(y,p),FPR=100*np.mean(p[y==0]!=0),F2=100*fbeta_score(y==1,p==1,beta=2,zero_division=0),errors=np.sum(y!=p))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
 root=Path(__file__).resolve().parent;base=root/'EVIDENCE/files'
 tables=json.loads((root/'manuscript/REVISION_LOG/qc_numeric_tables.json').read_text('utf8'))
 stages={};vectors=0;scores={};recall_checks=0
 for p in sorted((root/'R1/predictions').glob('*/evaluation.npz')):
  with np.load(p,allow_pickle=False) as z:
   stages[p.parent.name]={k:z[k].copy() for k in z.files}
 assert len(stages)==7
 recall=pd.read_csv(root/'EVIDENCE/current_s2_recall.csv')
 for protocol,label,prefix in [('Temporal','Temporal','temporal'),('Private history','Packet-private','private'),('Shared signature','Shared-signature','signature')]:
  group=[v for n,v in stages.items() if n.startswith(prefix)]
  assert len(group)==(3 if prefix=='temporal' else 2)
  if prefix!='temporal':
   ids=np.concatenate([v['row_ids'] for v in group]);assert len(ids)==len(np.unique(ids))==52584
  for name,display in [('WV','WV'),('GNB_default','GNB default'),('GNB_G_selected','GNB G-selected')]:
   vectors+=len(group)
   if prefix=='temporal':
    ms=[metrics(v['y'],v[name]) for v in group];m={k:np.mean([a[k] for a in ms]) for k in ms[0]}
    rec=np.mean([[np.mean(v[name][v['y']==c]==c) for c in range(5)] for v in group],axis=0)*100
   else:
    y=np.concatenate([v['y'] for v in group]);pred=np.concatenate([v[name] for v in group]);m=metrics(y,pred);rec=np.array([np.mean(pred[y==c]==c) for c in range(5)])*100
   scores[(protocol,name)]=m
   r=recall[(recall.protocol==label)&(recall.predictor==display)].iloc[0]
   assert np.allclose(rec,r[['Normal','DDoS','Scanning','SQL','Ransomware']].to_numpy(float),atol=1e-10,rtol=0)
   recall_checks+=5
 lookup={'WV (reference)':'WV','GNB (default)':'GNB_default','GNB (G-selected)':'GNB_G_selected'}
 for row in tables[2][1:]:
  protocol,method,*values=row;m=scores[(protocol,lookup[method])]
  for text,k in zip(values,['F1','BA','FPR','F2','errors']):assert abs(float(text.replace(',',''))-m[k])<=.00501,(protocol,method,k)
 fac=pd.read_csv(root/'R1/derived/factorial_summary.csv')
 for row,view in zip(tables[0][1:],['c40_o1','c40_o10','c60_o1','c60_o10']):
  t=fac[(fac.view==view)&(fac.protocol=='temporal')].iloc[0];s=fac[(fac.view==view)&(fac.protocol=='signature')].iloc[0]
  expected=[100*t.macro_f1,100*t.balanced_accuracy,t.errors,100*s.macro_f1,s.errors]
  for text,value in zip(row[1:],expected):
   text=text.replace(',','');decimals=len(text.split('.')[1]) if '.' in text else 0;assert abs(float(text)-value)<=.5*10**-decimals+1e-8
 history=pd.read_csv(root/'R1/derived/history_summary.csv')
 for row,key in zip(tables[1][1:],['shared','density','exclusive']):
  # Exact archive spellings are reported in diagnostics if this contract changes.
  selected=fac[(fac.view=='c60_o10')&(fac.protocol=='signature')] if key=='shared' else history[history.history.str.contains(key,case=False)]
  if len(selected)==0 and key=='density':selected=history[history.history.str.contains('matched',case=False)]
  assert len(selected)==1,(key,history.history.tolist())
  r=selected.iloc[0];expected=[100*r.macro_f1,100*r.balanced_accuracy,100*r.recall_Port_Scanning,100*r.recall_Ransomware,r.errors]
  for text,value in zip(row[1:],expected):
   text=text.replace(',','');decimals=len(text.split('.')[1]) if '.' in text else 0;assert abs(float(text)-value)<=.5*10**-decimals+1e-8,(key,text,value)
 serving=base/'runs/ARRAY_SERVING_ABLATION_20261117_01'
 lat=pd.read_csv(serving/'latency_raw.csv');rep=pd.read_csv(serving/'replay_raw.csv');mem=pd.read_csv(serving/'memory_raw.csv')
 assert (len(lat),len(rep),len(mem))==(480,240,80)
 assert rep.labels_match.all() and mem.labels_match.all()
 for row in tables[3][1:]:
  policy=row[0].replace(' / ','_').replace(' ','_')
  values=[1000*lat[(lat.policy==policy)&(lat.batch==n)].seconds.median() for n in [1,1000]]
  values += [mem[(mem.policy==policy)&(mem.workload=='feature')].delta_mib.mean(),rep[rep.policy==policy].total_seconds.mean(),mem[(mem.policy==policy)&(mem.workload=='replay')].delta_mib.mean()]
  for text,value in zip(row[1:],values):
   decimals=len(text.split('.')[1]) if '.' in text else 0;assert abs(float(text)-value)<=.5*10**-decimals+1e-8
 means=rep.groupby('policy').total_seconds.mean();saving=100*(1-means['WV_compact_certified']/means['WV_native_full']);assert abs(saving-60.55)<.005
 faith=pd.read_csv(root/'R1/derived/xai_faithfulness.csv').iloc[0]
 for text,key in zip(tables[4][1][1:],['deletion_auc','insertion_auc','deletion_gain','insertion_gain']):assert abs(float(text)-faith[key])<.000051
 for row in tables[5][1:]:
  n=int(row[1].replace(',',''))
  for value in row[2:4]:
   fp,percent=re.match(r'(\d+) \(([\d.]+)%\)',value).groups();assert abs(100*int(fp)/n-float(percent))<.000051
 cov=pd.read_csv(base/'reports/INDEPENDENT_CAPTURE_20261109/TON_label_coverage.csv');assert np.allclose(100*(cov.normal+cov.ransomware)/cov.total,cov.in_scope_labelled_percent,atol=1e-10)
 out={'status':'PASS_FOCUSED_WV_GNB_SAVED_RESULTS','prediction_vectors_reconstructed':vectors,'class_recalls_verified':recall_checks,'main_tables_checked':[1,2,3,4,5,6],'raw_measurements':{'feature_calls':len(lat),'replays':len(rep),'RSS_workers':len(mem)},'replay_reduction_percent':saving,'new_fits':0,'new_predictions':0,'model_deserializations':0,'scope':'Current WV/GNB reporting subset. Versioned historical comparisons remain indexed; no universal classifier superiority or new independent test is claimed.'}
 if args.output:args.output.write_text(json.dumps(out,indent=2),encoding='utf8')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
