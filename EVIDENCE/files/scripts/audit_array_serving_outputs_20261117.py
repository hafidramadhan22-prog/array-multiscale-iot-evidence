"""Persist actual outputs of owned frozen serving policies after profiling is complete."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
from threadpoolctl import threadpool_limits
from profile_array_serving_20261117 import load,POLICIES
from analyze_array_enhancement_20261117 import sha,RUN,OUT
def main():
    done=json.loads((RUN/'complete.json').read_text('utf8'));assert done['status']=='PASS_SERVING_ABLATION'
    assert not (OUT/'observed_serving_outputs.npz').exists()
    OUT.mkdir(parents=True,exist_ok=True);arrays={};checks=[]
    with threadpool_limits(limits=1):
        freeze,bundle,objects,frames,expected=load()
        for key,X in frames.items():
            arrays[key+'__row_ids']=X.index.to_numpy()
            for policy,obj in objects.items():
                labels=obj.predict(X);np.testing.assert_array_equal(labels,expected[policy][key]);arrays[policy+'__'+key+'__labels']=labels
                checks.append(dict(policy=policy,workload=key,rows=len(X),labels_match=True))
                if not policy.endswith('certified'):arrays[policy+'__'+key+'__probabilities']=obj.predict_proba(X)
            for family in ['WV','RF','LR']:
                ref=arrays[family+'_native_full__'+key+'__probabilities'] if family=='WV' else arrays[family+'_native__'+key+'__probabilities']
                compact=arrays[family+'_compact_full__'+key+'__probabilities'] if family=='WV' else arrays[family+'_compact__'+key+'__probabilities']
                np.testing.assert_allclose(compact,ref,atol=1e-12,rtol=0)
        # Values and sparse indices/indptr must both retain the trained semantics.
        enc=objects['WV_compact_full'].encoding
        for actual,reference in [(enc.tree(frames['1000']),bundle['models']['random_forest']['preprocessor'].transform(frames['1000'])),
            (enc.linear(enc.tree(frames['1000'])),bundle['models']['logistic_regression']['preprocessor'].transform(frames['1000']))]:
            for attr in ['data','indices','indptr']:np.testing.assert_array_equal(getattr(actual,attr),getattr(reference,attr))
    np.savez_compressed(OUT/'observed_serving_outputs.npz',**arrays)
    result=dict(status='PASS_ACTUAL_SERVING_OUTPUTS_AND_CSR_PERSISTED',date='2026-11-17',checks=checks,
        arrays_sha256=sha(OUT/'observed_serving_outputs.npz'),source_sha256=sha(Path(__file__)),
        posterior_tolerance=1e-12,CSR_values_and_structure_identical=True,new_classifier_fits=0,post_profile_verification_inference=True)
    (OUT/'observed_outputs_audit.json').write_text(json.dumps(result,indent=2)+'\n','utf8');print(result['status'])
if __name__=='__main__':main()
