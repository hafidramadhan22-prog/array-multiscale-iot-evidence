"""Runnable research prototype for already-prepared C60/O10 feature rows.

The reference and binary models must be the project's checksum-verified repair
run. No arbitrary joblib path or label input is accepted for model selection.
"""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from threadpoolctl import threadpool_limits
from run_sql_disagreement_repair_20261105 import load_models,expert_probabilities,PARENT,read,owned
from payload_tfidf_pilot import augment
from payload_target_repair import predict
from sql_disagreement_guard import eligible_rows,apply_guard

def main(basepath,payloadpath,output):
    if output.exists():raise ValueError('Output already exists; choose a new output file')
    done=read(PARENT/'complete.json');assert e.sha256(PARENT/'artifacts_sha256.json')==done['catalog_sha256']
    catalog=read(PARENT/'artifacts_sha256.json')
    for rel in ('models_manifest.json','configuration_freeze.json','source_manifest.json'):
        owned(PARENT,next(r for r in catalog if r['path']==rel))
    for r in read(PARENT/'source_manifest.json'):assert e.sha256(ROOT/r['path'])==r['sha256'],r['path']
    cfg=read(PARENT/'configuration_freeze.json');cats=cfg['base_categorical'];cols=cfg['base_numeric']+cats
    base=pd.read_csv(basepath,index_col='row_id',dtype={'row_id':str,**{c:str for c in cats}},keep_default_na=False,float_precision='round_trip')
    raw=pd.read_csv(payloadpath,index_col='row_id',dtype={'row_id':str,'tcp.payload':str,'mqtt.msg':str},keep_default_na=False)
    if not base.index.is_unique or not base.index.equals(raw.index):raise ValueError('Base/payload row_id mismatch or duplicate')
    if not set(cols).issubset(base) or not {'tcp.payload','mqtt.msg'}.issubset(raw):raise ValueError('Missing required feature columns')
    base=base[cols].copy()
    for c in cfg['base_numeric']:base[c]=pd.to_numeric(base[c],errors='raise') if not base[c].eq('').any() else pd.to_numeric(base[c],errors='coerce')
    models=load_models();w,rf=expert_probabilities(models['reference'],base);eligible=eligible_rows(w,rf,.5);has=np.zeros(len(base),dtype=bool);bp=np.full((len(base),2),np.nan);ix=np.flatnonzero(eligible)
    if len(ix):
        x=augment(base.iloc[ix],raw.iloc[ix]);nonempty=x.payload_text.str.len().gt(0).to_numpy();use=ix[nonempty];has[use]=True
        if len(use):bp[use]=predict(models['binary_syntax_tfidf'],x.iloc[np.flatnonzero(nonempty)],'weighted_voting')
    p,changed,reason=apply_guard(w,rf,has,bp,.5,True);names=np.array(['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware'])
    out=pd.DataFrame({'row_id':base.index,'original_prediction':names[w.argmax(1)],'repaired_prediction':names[p.argmax(1)],'changed':changed,'reason':reason,'RF_p_Normal':rf[:,0]})
    for i,name in enumerate(names):out['p_'+name]=p[:,i]
    output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(output,index=False);print('SAVED',output,'rows',len(out),'changed',int(changed.sum()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.base,a.payload,a.output)
