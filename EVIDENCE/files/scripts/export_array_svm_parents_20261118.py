"""Copy only hash-verified parent evidence needed for portable numeric audits."""
from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import ta_hafid_engine as e
from run_array_svm_20261118 import PARENTS,gated,STAGES

def main():
    out=ROOT/'reports/ARRAY_SVM_20261118/parent_inputs';assert not out.exists();out.mkdir()
    records=[];catalogs=[]
    def copy(parent,rel):
        p=gated(parent,rel);q=out/parent/'files'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        records.append(dict(parent=parent,original_rel=rel,path=q.relative_to(out).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size))
    for parent in ['reference','history_models','aligned','guard','external']:
        folder,name,digest=PARENTS[parent];p=ROOT/'runs'/folder/name;assert e.sha256(p)==digest
        q=out/parent/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);catalogs.append(dict(parent=parent,path=q.relative_to(out).as_posix(),sha256=digest))
    for stage in STAGES:
        if stage.startswith('private'):
            parent='history_models';base='signature_fold'+stage[-1]+'_exclusive'
            for f in ['split.npz','evaluation.npz','calibration.npz']:copy(parent,base+'/'+f)
        else:
            parent='reference';base='stages/'+stage+'/legacy25_multiscale60__balanced'
            copy(parent,'stages/'+stage+'/split.npz')
            for f in ['evaluation.npz','calibration.npz','selection_before_V.json']:copy(parent,base+'/'+f)
    for p in sorted((ROOT/'runs/ARRAY_FLAG_FORMAT_20261104_01').glob('*_predictions.npz')):copy('aligned',p.name)
    for p in sorted((ROOT/'runs/FPR_SQL_GUARD_20261109_01/posteriors').glob('*_later.npz')):copy('guard','posteriors/'+p.name)
    for name in ['CIC_Normal0','CIC_Normal3','CIC_DDoS_SYN','CIC_PortScan','CIC_SQL','TON_Ransom1','TON_Ransom2']:copy('external',name+'/probabilities.npz')
    e.write_json(out/'manifest.json',dict(catalogs=catalogs,files=records,raw_dataset_included=False,old_model_loading=False,new_fits=0))
    print('HASH_VERIFIED_PARENT_INPUTS_EXPORTED',len(records))
if __name__=='__main__':main()
