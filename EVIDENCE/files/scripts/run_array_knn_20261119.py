from pathlib import Path
import sys,json,time,shutil,platform
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import pandas as pd
import sklearn,joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from run_array_svm_20261118 import stage_inputs,STAGES,gated,read,e
from wb_research import prepare,mixture
from array_svm_20261118 import label_metrics

def main(smoke=False):
 out=ROOT/'runs'/('ARRAY_KNN_SMOKE_20261119_01' if smoke else 'ARRAY_KNN_FULL_20261119_01');out.mkdir(exist_ok=False)
 specs=[(k,w) for k in [1,3,5,11,31] for w in ['uniform','distance']]
 e.write_json(out/'configuration_freeze.json',dict(specs=specs,smoke=smoke,time=time.time(),python=platform.python_version(),sklearn=sklearn.__version__,selection='G only .5F1+.5DDoSF2',previously_observed_evaluation=True))
 for f in ['scripts/run_array_knn_20261119.py','docs/ARRAY_KNN_PROTOCOL_20261119.md']:shutil.copy2(ROOT/f,out/Path(f).name)
 rows=[];allarrays={}
 with threadpool_limits(limits=1):
  for name in (STAGES[:1] if smoke else STAGES):
   print('START',name,flush=True)
   d=out/name;d.mkdir();X,y,raw,holdout,parts,num,cat,local,ref=stage_inputs(name,smoke);a,g,c,v=parts
   np.savez_compressed(d/'split.npz',**{k:np.asarray(raw.index[p],dtype=str) for k,p in zip('AGCV',parts)},holdout=np.asarray(holdout,dtype=str))
   prep,Z,yy,plan=prepare(X.iloc[a],y[a],'balanced','linear',num,cat,local,raw.index[np.r_[g,c,v]])
   e.write_json(d/'training_plan.json',dict(plan=plan,numeric=num,categorical=cat,feature_order=list(X.columns),train_shape=Z.shape))
   Z=Z.tocsr();G=prep.transform(X.iloc[g]).tocsr();models={};gr=[];gp={}
   for k,w in specs:
    tag=f'k{k}_{w}';m=KNeighborsClassifier(n_neighbors=k,weights=w,algorithm='brute',metric='euclidean',n_jobs=1)
    t=time.perf_counter();m.fit(Z,yy);pred=m.predict(G);elapsed=time.perf_counter()-t
    met,_=label_metrics(y[g],pred);gr.append(dict(id=tag,k=k,weights=w,score=.5*(met['macro_f1']+met['ddos_f2']),**met,fit_and_G_seconds=elapsed));gp[tag]=pred;models[tag]=m
   winner=max(gr,key=lambda r:(r['score'],r['macro_f1'],-abs(r['k']-5),r['weights']=='uniform',-r['k']))['id']
   pd.DataFrame(gr).to_csv(d/'G_candidates.csv',index=False);np.savez_compressed(d/'G_predictions.npz',y=y[g],row_ids=np.asarray(raw.index[g],dtype=str),**gp)
   e.write_json(d/'selection_before_V.json',dict(selected=winner,default='k5_uniform',time=time.time(),V_scored=False))
   V=prep.transform(X.iloc[v]).tocsr();preds={}
   for policy,tag in [('KNN_default','k5_uniform'),('KNN_G_selected',winner)]:
    preds[policy]=models[tag].predict(V)
    joblib.dump(Pipeline([('preprocessor',prep),('estimator',models[tag])]),d/(policy+'.joblib'),compress=1)
   w=ref['weights'] if 'weights' in ref else read(gated('reference',f'stages/{name}/legacy25_multiscale60__balanced/selection_before_V.json'))['weights']
   ix={int(r):i for i,r in enumerate(ref['row_ids'])};pp=ref['p_experts'][[ix[int(r)] for r in raw.index[v]]]
   preds.update(WV=mixture(pp,w).argmax(1),RF=pp[:,1].argmax(1),LR=pp[:,4].argmax(1),Uniform=pp.mean(1).argmax(1))
   arrays=dict(y=y[v],row_ids=np.asarray(raw.index[v],dtype=str),**preds);protocol='packet_private' if name.startswith('private') else 'temporal' if name.startswith('temporal') else 'signature_shared'
   for method,pred in preds.items():
    met,cm=label_metrics(y[v],pred);arrays['CM_'+method]=cm;rows.append(dict(stage=name,protocol=protocol,method=method,**met))
   np.savez_compressed(d/'evaluation.npz',**arrays);allarrays[name]=arrays
   pd.DataFrame(rows).to_csv(out/'stage_metrics.csv',index=False)
   print('DONE',name,winner,[(r['method'],round(100*r['macro_f1'],4)) for r in rows if r['stage']==name],flush=True)
 df=pd.DataFrame(rows);summary=[]
 for protocol,group in df.groupby('protocol'):
  names=group.stage.unique()
  for method in group.method.unique():
   if protocol=='temporal':
    sub=group[group.method==method];met={k:float(sub[k].mean()) for k in ['macro_f1','normal_fpr','balanced_accuracy','ddos_f2','binary_attack_recall','errors']};met['macro_f1_sd']=float(sub.macro_f1.std()) if len(sub)>1 else 0
   else:
    ids=np.concatenate([allarrays[n]['row_ids'] for n in names]);assert len(set(ids))==len(ids)
    met,cm=label_metrics(np.concatenate([allarrays[n]['y'] for n in names]),np.concatenate([allarrays[n][method] for n in names]))
   summary.append(dict(protocol=protocol,method=method,**met))
 pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
 manifest=[dict(path=f.relative_to(out).as_posix(),sha256=e.sha256(f),bytes=f.stat().st_size) for f in sorted(out.rglob('*')) if f.is_file()]
 e.write_json(out/'catalog.json',manifest);e.write_json(out/'complete.json',dict(status='COMPLETE',smoke=smoke,stages=len(allarrays),catalog_sha256=e.sha256(out/'catalog.json')))
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main('--smoke' in sys.argv)
