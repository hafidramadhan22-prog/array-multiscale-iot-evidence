"""Recompute Shapley arrays and external counts from saved observations, without model loading."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
base=ROOT/'EVIDENCE/files'
results=[]
for run in ['ARRAY_XAI_FULL_20261103_01','XAI_MULTISCALE_FULL_20261020_01']:
 for file in sorted((base/'runs'/run/'groups').glob('*/coalitions.npz')):
  with np.load(file,allow_pickle=False) as z:
   values=z['values'];phi=z['phi'];n=phi.shape[1];assert values.shape[1]==2**n
   computed=np.zeros_like(phi,dtype=float)
   for j in range(n):
    for mask in range(2**n):
     if mask & (1<<j):continue
     k=mask.bit_count();weight=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
     computed[:,j]+=weight*(values[:,mask|(1<<j)]-values[:,mask])
   error=float(np.max(np.abs(computed-phi)));assert error<1e-10
   add=float(np.max(np.abs(phi.sum(axis=1)-(values[:,-1]-values[:,0]))));assert add<1e-10
   results.append({'path':str(file.relative_to(ROOT)),'targets':len(z['row_ids']),'groups':n,'donors':len(z['background_ids']),'max_shapley_reconstruction_error':error,'max_additivity_error':add})
pred=pd.read_csv(base/'reports/INDEPENDENT_CAPTURE_20261109/all_predictions_verified.csv')
rows=[]
for (source,policy),g in pred.groupby(['source','policy']):
 cm=pd.crosstab(g.truth,g.prediction)
 rows.append({'source':source,'policy':policy,'n':len(g),'confusion_counts':cm.to_dict()})
(ROOT/'R4/external_confusion_recomputed.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
out={'status':'PASS','shapley_reconstructions':results,'external_prediction_records':len(pred),'external_source_policy_groups':len(rows),'scope':'Reconstructs stored finite games and counts; does not execute classifiers or claim independent validation.'}
(ROOT/'EXTENDED_VERIFICATION.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
