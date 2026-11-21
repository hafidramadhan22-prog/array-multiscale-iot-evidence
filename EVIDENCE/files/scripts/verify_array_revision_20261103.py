"""Independent saved-result reconstruction and provenance checks; no deserialization."""
from pathlib import Path
import sys,json,hashlib,re
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score,fbeta_score
from run_array_factorial_20261103 import read_npz
from run_array_history_models_20261103 import verify_data
import ta_hafid_engine as e
OUT=ROOT/'reports/ARRAY_MAJOR_REVISION_20261103'

def main():
    checks=0;files=0;fits=0
    for name in ['ARRAY_HISTORY_DATA_20261103_01','ARRAY_HISTORY_FULL_20261103_01','ARRAY_BUFFER_FULL_20261103_01','ARRAY_XAI_FULL_20261103_01','ARRAY_RUNTIME_FULL_20261103_01']:
        folder=ROOT/'runs'/name;verify_data(folder);files+=len(json.loads((folder/'catalog.json').read_text()))
    folder=ROOT/'runs/ARRAY_FACTORIAL_FULL_20261103_01'
    for r in json.loads((folder/'artifacts_sha256.json').read_text()):
        p=folder/r['path'];assert p.stat().st_size==r['bytes'] and e.sha256(p)==r['sha256'];files+=1
    holdout=set(read_npz(ROOT/'runs/FULL_20261013_01/canonical_split_indices.npz')['holdout_row_ids'])
    f=pd.read_csv(folder/'metrics.csv');h=pd.read_csv(ROOT/'runs/ARRAY_HISTORY_FULL_20261103_01/metrics.csv')
    for group,metric in [('factorial',f),('history',h)]:
        for row in metric.itertuples():
            dest=folder/'stages'/row.stage/row.view if group=='factorial' else ROOT/'runs/ARRAY_HISTORY_FULL_20261103_01'/f'{row.stage}_{row.history}'
            z=read_npz(dest/'evaluation.npz');p=z['p_experts'];w=z['weights']
            assert not set(z['row_ids'])&holdout and len(set(z['row_ids']))==len(z['row_ids'])
            probs={'weighted_voting':np.einsum('nmc,m->nc',p,w),'uniform_voting':p.mean(1),'random_forest':p[:,1],'logistic_regression':p[:,4]}
            pred=probs[row.method].argmax(1);y=z['y'];assert len(y)==row.n
            np.testing.assert_allclose(f1_score(y,pred,average='macro'),row.macro_f1,rtol=0,atol=1e-12)
            np.testing.assert_allclose(recall_score(y,pred,average='macro'),row.balanced_accuracy,rtol=0,atol=1e-12)
            np.testing.assert_allclose(fbeta_score(y==1,pred==1,beta=2),row.ddos_f2,rtol=0,atol=1e-12)
            assert int((y!=pred).sum())==row.errors;checks+=4
    for base in [folder,ROOT/'runs/ARRAY_HISTORY_FULL_20261103_01']:
        for path in base.rglob('frozen_before_V.json'):
            record=json.loads(path.read_text());splitpath=path.parent/'split.npz'
            if not splitpath.exists():splitpath=path.parent.parent/'split.npz'
            split=read_npz(splitpath);a=set(split['A']);other=set(np.r_[split['G'],split['C'],split['V']]);assert not a&other and not a&holdout
            assert e.sha256(path.parent/'model.joblib')==record['sha256']
            for plan in record['plans']:
                assert set(plan['preprocessing_fit_row_ids'])==a
                assert max(plan['synthetic_per_original_class'].values())<=.5
                assert plan['synthetic_fraction_of_final_training']<=.5
                for k,v in plan['synthetic'].items():assert v<=.5*plan['before'][k]
            fits+=5;checks+=4
    assert fits==70
    a=json.loads((ROOT/'runs/MULTISCALE_REPAIR_FULL_20261020_01/configuration_freeze.json').read_text());b=json.loads((ROOT/'runs/WB_RESEARCH_FULL_20261019_01/configuration_freeze.json').read_text());assert a['params']==b['params']
    runtime=ROOT/'runs/ARRAY_RUNTIME_FULL_20261103_01';r=pd.read_csv(runtime/'replay_raw.csv');p=r.pivot(index='repeat',columns='mode',values='total_seconds');delta=p.native_certified-p.compact_certified;s=json.loads((runtime/'paired_summary.json').read_text());assert len(delta)==30 and r.labels_match.all();np.testing.assert_allclose(delta.mean(),s['mean_seconds'],atol=1e-12)
    mem=pd.read_csv(runtime/'memory_raw.csv');np.testing.assert_allclose(mem.peak_mib-mem.baseline_mib,mem.delta_mib,atol=1e-12);assert (mem.groupby(['policy','workload']).size()==5).all()
    x=ROOT/'runs/ARRAY_XAI_FULL_20261103_01'
    for name in ['A32','B32']:
        z=read_npz(x/'groups'/name/'coalitions.npz');np.testing.assert_allclose(z['phi'].sum(1),z['values'][:,-1]-z['values'][:,0],atol=1e-10);checks+=1
    # Check every article citation resolves, and every included figure exists.
    paper=ROOT/'paper/array_20261103_major_revision/elsarticle';tex=(paper/'main.tex').read_text();bib=(paper/'references.bib').read_text();keys=set(re.findall(r'@\w+\{([^,]+)',bib));cited={k for g in re.findall(r'\\cite\{([^}]+)\}',tex) for k in g.split(',')};assert cited<=keys and len(cited)==25
    for doc in ['main','supplement']:
        t=(paper/(doc+'.tex')).read_text()
        for rel in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',t):assert (paper/rel).exists()
        labels=re.findall(r'\\label\{([^}]+)\}',t);assert len(labels)==len(set(labels));assert set(re.findall(r'\\ref\{([^}]+)\}',t))<=set(labels)
    report=dict(status='PASS_REVISION_EVIDENCE',catalog_files=files,metric_and_guard_checks=checks,new_expert_fits=fits,cited_references=len(cited),runtime_pairs=30,cold_memory_repeats=len(mem),holdout_exclusion=True,training_only_resampling=True,smoke_results_used=False,independent_five_class_final_test=False)
    e.write_json(OUT/'verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
