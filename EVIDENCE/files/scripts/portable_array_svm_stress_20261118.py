"""Encode IDs as unicode without loading object arrays or recomputing predictions."""
from pathlib import Path
import sys,shutil,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from run_array_svm_20261118 import read

def main():
    source=ROOT/'runs/ARRAY_SVM_STRESS_20261118_01';out=ROOT/'runs/ARRAY_SVM_STRESS_PORTABLE_20261118_01';assert not out.exists()
    done=read(source/'complete.json');assert done['catalog_sha256']==e.sha256(source/'catalog.json')
    for r in read(source/'catalog.json'):
        f=source/r['path'];assert f.stat().st_size==r['bytes'] and e.sha256(f)==r['sha256']
    out.mkdir();converted=[]
    for f in source.rglob('*'):
        if not f.is_file() or f.relative_to(source).as_posix() in ['catalog.json','complete.json']:continue
        dest=out/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if f.parent.name=='predictions' and f.suffix=='.npz':
            csv=source/'inputs'/(f.stem+'.csv');ids=pd.read_csv(csv,usecols=['row_id'],dtype=str,keep_default_na=False)['row_id'].to_numpy(dtype=str)
            with np.load(f,allow_pickle=False) as z:
                numeric={k:z[k] for k in z.files if k!='row_ids'}
            assert all(v.dtype!=object for v in numeric.values()) and len(ids)==len(numeric['y'])
            np.savez_compressed(dest,row_ids=ids,**numeric)
            with np.load(dest,allow_pickle=False) as z:
                for key,value in numeric.items():np.testing.assert_array_equal(z[key],value)
                np.testing.assert_array_equal(z['row_ids'],ids)
            converted.append(dict(path=dest.relative_to(out).as_posix(),source_sha256=e.sha256(f),portable_sha256=e.sha256(dest),rows=len(ids),ids_source=csv.relative_to(ROOT).as_posix(),numeric_arrays_identical=True))
        else:shutil.copy2(f,dest)
    shutil.copy2(Path(__file__),out/'source'/Path(__file__).name)
    (out/'provenance').mkdir();shutil.copy2(source/'catalog.json',out/'provenance/original_catalog.json');shutil.copy2(source/'complete.json',out/'provenance/original_complete.json')
    e.write_json(out/'portable_encoding_correction.json',dict(finding='pandas string Index.to_numpy produced object dtype for saved IDs',action='Reconstruct unicode IDs from verified scored input CSV; copy numeric arrays without deserializing object arrays',no_pickle_loading=True,new_fits=0,new_inference=0,source_catalog_sha256=e.sha256(source/'catalog.json'),converted=converted))
    records=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=e.sha256(f)) for f in sorted(out.rglob('*')) if f.is_file()]
    e.write_json(out/'catalog.json',records);e.write_json(out/'complete.json',dict(status='PASS_SVM_POST_FREEZE_STRESS',portable_ids=True,cohorts=len(converted),new_fits=0,new_inference=0,independent_new_test=False,numeric_arrays_unchanged=True,catalog_sha256=e.sha256(out/'catalog.json')))
    print('PORTABLE_STRESS_IDS_VERIFIED',len(converted))
if __name__=='__main__':main()
