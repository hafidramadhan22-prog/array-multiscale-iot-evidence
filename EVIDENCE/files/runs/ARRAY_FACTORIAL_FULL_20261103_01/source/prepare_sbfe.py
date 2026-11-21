"""Verify canonical identity and build causal, label-free PCAP features on development targets."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from e4 import RAW_COLUMNS,verified_historical_file
from sbfe import META,extract

def load_raw():
    dataset=ROOT/"ML-EdgeIIoT-dataset.csv"
    if e.sha256(dataset)!=e.DEFAULT_CONFIG["dataset_sha256"]:raise ValueError("Canonical SHA256 mismatch")
    split_file=verified_historical_file(ROOT/"runs/FULL_20261013_01","canonical_split_indices.npz")
    with np.load(split_file,allow_pickle=False) as z:
        ids=z["training_row_ids"];holdout=z["holdout_row_ids"]
    wanted=set(ids.tolist())
    raw=pd.read_csv(dataset,low_memory=False,usecols=list(dict.fromkeys(RAW_COLUMNS+META+[e.TARGET])),
                    skiprows=lambda line:line>0 and line-1 not in wanted)
    raw.index=sorted(wanted);raw=raw.loc[ids]
    y=raw.pop(e.TARGET).map({c:i for i,c in enumerate(e.CLASSES)}).to_numpy(dtype=int)
    if len(raw)!=52684 or set(ids)&set(holdout):raise ValueError("Development/holdout gate")
    return raw,y,holdout

def main():
    output=ROOT/"runs/SBFE_DATA_20261019_03"
    if output.exists():raise ValueError("Use a new data preparation directory")
    cfg=json.loads((ROOT/"configs/sbfe.json").read_text())
    archive=Path(cfg["zip_path"])
    if e.sha256(archive)!=cfg["zip_sha256"]:raise ValueError("ZIP SHA256 mismatch")
    raw,y,holdout=load_raw()
    output.mkdir()
    e.write_json(output/"configuration_freeze.json",cfg)
    for src in ["src/sbfe.py","scripts/prepare_sbfe.py","configs/sbfe.json","docs/SBFE_PROTOKOL_20261019.md"]:
        shutil.copy2(ROOT/src,output/Path(src).name)
    audit=json.loads((ROOT/"reports/DATA_MENTAH_AUDIT_ZIP_20261019/zip_audit.json").read_text(encoding="utf-8"))
    result=extract(raw,archive,audit,cfg,output)
    np.savez_compressed(output/"target_indices.npz",row_ids=raw.index.to_numpy(),y=y,historical_holdout_ids=holdout)
    e.write_json(output/"environment.json",e.environment())
    support=[]
    for role in range(4):
        part=np.flatnonzero((result["role"]==role)&result["valid_after_purge"])
        counts=np.bincount(y[part],minlength=5)
        if (counts==0).any():raise ValueError("Missing class in temporal role")
        support.append(dict(role=["A_fit","G_gate","C_threshold","V_evaluation"][role],rows=len(part),
                            counts=dict(zip(e.CLASSES,counts.tolist()))))
    e.write_json(output/"role_support.json",support)
    e.write_json(output/"catalog.json",[dict(path=p.name,bytes=p.stat().st_size,sha256=e.sha256(p))
                    for p in output.iterdir() if p.is_file() and p.name!="catalog.json"])
    e.write_json(output/"complete.json",dict(status="DATA_PREPARATION_COMPLETE",rows=len(raw),source_zip_sha256=cfg["zip_sha256"],
                    catalog_sha256=e.sha256(output/"catalog.json"),new_training=False))
    print(json.dumps(support,indent=2),flush=True)
if __name__=="__main__":main()

