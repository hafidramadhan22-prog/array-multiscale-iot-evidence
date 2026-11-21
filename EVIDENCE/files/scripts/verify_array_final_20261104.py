"""Verify saved final-review outputs without model loading or new inference."""
from pathlib import Path
import sys,re,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from run_array_factorial_20261103 import read_npz,methods
from run_array_history_models_20261103 import verify_data
from e4 import columns
from multiscale_context import CONTEXT_NAMES

def main():
    old=ROOT/'paper/array_20261103_major_revision/elsarticle';new=ROOT/'paper/array_20261104_final_review/elsarticle';out=ROOT/'reports/ARRAY_FINAL_REVIEW_20261104'
    a=re.findall(r'\\begin\{table\}.*?\\end\{table\}',(old/'supplement.tex').read_text(),re.S);b=re.findall(r'\\begin\{table\}.*?\\end\{table\}',(new/'supplement.tex').read_text(),re.S)
    assert len(a)==40 and len(b)==45
    for x,y in zip(a,b):assert x[x.index('\\begin{tabular}'):]==y[y.index('\\begin{tabular}'):]
    text=(new/'main.tex').read_text();bib=(new/'references.bib').read_text();keys=set(re.findall(r'@\w+\{([^,]+)',bib));cited={k for s in re.findall(r'\\cite\{([^}]+)\}',text) for k in s.split(',')};assert len(cited)==26 and cited<=keys
    refs=re.findall(r'\\ref\{([^}]+)\}',text)
    for k in ['history','ensemble','benign','buffer','runtime','faith']:assert 'tab:'+k in refs
    for name in ['main','supplement']:
        t=(new/(name+'.tex')).read_text();ll=re.findall(r'\\label\{([^}]+)\}',t);rr=re.findall(r'\\ref\{([^}]+)\}',t);assert len(ll)==len(set(ll)) and set(rr)<=set(ll)
    run=ROOT/'runs/ARRAY_FLAG_FORMAT_20261104_01';verify_data(run);verify_data(ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04');metrics=pd.read_csv(run/'metrics.csv');checks=0
    for p in run.glob('*_predictions.npz'):
        z=read_npz(p);source=p.stem.replace('_predictions','');previous=read_npz(ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04'/p.name)
        np.testing.assert_array_equal(z['record_index'],previous['record_index']);np.testing.assert_allclose(z['p_before'],previous['p_experts'],rtol=0,atol=1e-12)
        num,cat=columns('noport25');num+=list(CONTEXT_NAMES)
        before=pd.read_csv(ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04'/(source+'_features.csv'),dtype={c:str for c in cat});after=pd.read_csv(run/(source+'_features.csv'),dtype={c:str for c in cat})
        np.testing.assert_array_equal(before.record_index,after.record_index)
        np.testing.assert_allclose(before[num],after[num],rtol=0,atol=1e-12,equal_nan=True)  # CSV decimal round-trip; in-memory equality checked before inference.
        unchanged_cat=[c for c in cat if c not in ['tcp.flags','mqtt.conflags','mqtt.hdrflags']]
        assert before[unchanged_cat].equals(after[unchanged_cat])
        for fmt,key in [('historical_format','p_before'),('semantic_format','p_after')]:
            for m,prob in methods(z[key],z['weights']).items():
                pred=prob.argmax(1);row=metrics[(metrics.source==source)&(metrics['format']==fmt)&(metrics.method==m)].iloc[0]
                assert row.n==len(pred) and row.false_alarms==int((pred!=0).sum());checks+=2
                for cl in range(5):assert row['predicted_'+str(cl)]==int((pred==cl).sum());checks+=1
    s=metrics.groupby(['format','method'])[['n','false_alarms']].sum();assert s.loc[('semantic_format','weighted_voting'),'false_alarms']==321
    assert e.sha256(ROOT/'output/pdf/AGERS_HAFID_EDAS_20261028.pdf')=='7d0263f89d37de6e58d154145a3b23346a38748485c69e74c9d7f9ee6d6f61c2'
    audit=pd.read_csv(out/'alignment_reconstruction.csv');assert audit.matched.sum()==760 and audit.categorical_mismatches.sum()==0 and audit.numeric_max_abs_error.max()<1e-12
    replay=pd.read_csv(ROOT/'runs/ARRAY_RUNTIME_FULL_20261103_01/replay_raw.csv').pivot(index='repeat',columns='mode',values='total_seconds')
    table45=b[44];assert len(replay)==30
    for r in replay.itertuples():assert f'{r.native_certified:.6f}' in table45 and f'{r.compact_certified:.6f}' in table45
    report=dict(status='PASS_FINAL_REVISION_CHECKS',unchanged_supplementary_tables=40,new_supplementary_tables=5,table3_to8_explicit_refs=True,references=26,saved_metric_checks=checks,new_fits=0,flag_diagnostic_targets=760,weighted_before=324,weighted_aligned=321,unscorable=107,author_approval=False,new_runtime_repetitions=0,agers_copy_hash_verified=True)
    e.write_json(out/'verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
