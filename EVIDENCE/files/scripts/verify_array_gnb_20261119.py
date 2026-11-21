from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,fbeta_score
from run_array_svm_20261118 import e,STAGES
r=ROOT/'runs/ARRAY_GNB_FULL_20261119_01'
done=json.loads((r/'complete.json').read_text());assert done['status']=='COMPLETE' and not done['smoke'];assert e.sha256(r/'catalog.json')==done['catalog_sha256']
for rec in json.loads((r/'catalog.json').read_text()):
 p=r/rec['path'];assert p.stat().st_size==rec['bytes'] and e.sha256(p)==rec['sha256']
count=0
for n in STAGES:
 d=r/n;s=np.load(d/'split.npz',allow_pickle=False)
 plan=json.loads((d/'training_plan.json').read_text())['plan']
 assert set(map(str,plan['preprocessing_fit_row_ids']))==set(s['A'])
 for cl,num in plan['synthetic'].items():assert num<=np.floor(.5*plan['before'][cl])
 assert plan['denominator']=='original_per_class_before_RUS'
 for i,k in enumerate('AGCV'):
  assert not set(s[k])&set(s['holdout'])
  for q in 'AGCV'[i+1:]:assert not set(s[k])&set(s[q])
 g=np.load(d/'G_predictions.npz',allow_pickle=False);c=pd.read_csv(d/'G_candidates.csv');assert len(c)==13
 for _,a in c.iterrows():
  p=g[a['id']];f=f1_score(g['y'],p,average='macro');f2=fbeta_score(g['y']==1,p==1,beta=2)
  assert np.isclose(f,a.macro_f1,rtol=0,atol=1e-12) and np.isclose(.5*(f+f2),a.score,rtol=0,atol=1e-12)
 win=max(c.to_dict('records'),key=lambda a:(a['score'],a['macro_f1'],-abs(a['exponent']+9),-a['exponent']))['id']
 assert win==json.loads((d/'selection_before_V.json').read_text())['selected']
 z=np.load(d/'evaluation.npz',allow_pickle=False);np.testing.assert_array_equal(s['V'],z['row_ids'])
 old=np.load(ROOT/'runs/ARRAY_SVM_EVALUATION_20261118_01/stages'/n/'evaluation.npz',allow_pickle=False)
 for m in ['WV','RF','LR','Uniform']:np.testing.assert_array_equal(z[m],old[m])
 for m in ['GNB_default','GNB_G_selected','WV','RF','LR','Uniform']:
  np.testing.assert_array_equal(confusion_matrix(z['y'],z[m],labels=range(5)),z['CM_'+m]);count+=1
out=dict(status='PASS',stages=7,G_candidates_recomputed=91,V_confusion_matrices=count,split_disjointness=True,historical_holdout_excluded=True,reference_predictions_unchanged=True,catalog_sha256=e.sha256(r/'catalog.json'))
e.write_json(ROOT/'reports/ARRAY_GNB_20261119/verification.json',out);print(json.dumps(out))
