from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_array_svm_20261118 import stage_inputs,STAGES,e
import numpy as np,joblib
from threadpoolctl import threadpool_limits
r=ROOT/'runs/ARRAY_GNB_FULL_20261119_01';done=json.loads((r/'complete.json').read_text());assert e.sha256(r/'catalog.json')==done['catalog_sha256'];catalog={x['path']:x for x in json.loads((r/'catalog.json').read_text())};count=0;probability_checks=[]
with threadpool_limits(limits=1):
 for name in STAGES:
  X,y,raw,h,parts,*_=stage_inputs(name,False);a,g,c,v=parts;z=np.load(r/name/'evaluation.npz',allow_pickle=False);gz=np.load(r/name/'G_predictions.npz',allow_pickle=False);fr=json.loads((r/name/'selection_before_V.json').read_text())
  for method,tag in [('GNB_default',fr['default']),('GNB_G_selected',fr['selected'])]:
   f=r/name/(method+'.joblib');assert e.sha256(f)==catalog[f.relative_to(r).as_posix()]['sha256'];m=joblib.load(f)
   pred=m.predict(X.iloc[v]);np.testing.assert_array_equal(pred,z[method]);count+=len(pred)
   np.testing.assert_array_equal(m.predict(X.iloc[g]),gz[tag]);prob=m.predict_proba(X.iloc[v]);assert np.isfinite(prob).all();deviation=float(np.max(np.abs(prob.sum(1)-1)));probability_checks.append(dict(stage=name,method=method,max_sum_deviation=deviation,within_1e_8=deviation<=1e-8))
  print('REPLAY',name,flush=True)
e.write_json(ROOT/'reports/ARRAY_GNB_20261119/model_replay.json',dict(status='PASS',hash_verified_own_models=14,matched_V_labels=count,all_G_predictions_identical=True,all_V_probabilities_finite=True,probability_normalization_diagnostics=probability_checks,probability_gate_note='Strict 1e-8 normalization diagnostic failed for some GaussianNB outputs; models and labels unchanged, label-only comparator, probabilities not certified or calibrated',scope='All G and V rows, both policies, seven stages'))
