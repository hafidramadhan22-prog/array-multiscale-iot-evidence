"""Compare newly extracted features against owned saved context85 references."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from independent_capture_20261109 import extract_protocol,contexts_for,context85
from e4 import columns
import ta_hafid_engine as e
OUT=ROOT/'reports/INDEPENDENT_CAPTURE_20261109';FLAG=ROOT/'runs/ARRAY_FLAG_FORMAT_20261104_01'
def main():
    source='Flame_Sensor';rec=next(x for x in json.loads((ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04/source_audit.json').read_text()) if x['source']==source)
    pcap=Path(rec['pcap']);assert e.sha256(pcap)==rec['pcap_sha256']
    cat=columns('noport25')[1]
    for rel in [source+'_features.csv',source+'_predictions.npz']:
        row=next(r for r in json.loads((FLAG/'catalog.json').read_text()) if r['path']==rel);assert e.sha256(FLAG/rel)==row['sha256']
    with np.load(FLAG/(source+'_predictions.npz'),allow_pickle=False) as z:ids=z['record_index'].astype(int)
    fields,meta=extract_protocol(pcap,ids,OUT/'extractor_check_tshark.csv');values,history=contexts_for(pcap,ids);X,nums,cats=context85(fields,values)
    gold=pd.read_csv(FLAG/(source+'_features.csv'),index_col='record_index',dtype={c:str for c in cats},float_precision='round_trip').loc[ids]
    diffs=[]
    for c in cats:
        mask=X[c].to_numpy()!=gold[c].to_numpy()
        if mask.any():diffs.append(dict(feature=c,n=int(mask.sum()),examples=[dict(packet_index=int(ids[i]),new=str(X[c].iloc[i]),cached=str(gold[c].iloc[i])) for i in np.flatnonzero(mask)[:8]]))
    for c in nums:
        delta=abs(X[c].to_numpy(float)-gold[c].to_numpy(float));mask=delta>1e-12
        if mask.any():diffs.append(dict(feature=c,n=int(mask.sum()),max_absolute_difference=float(delta.max()),examples=[dict(packet_index=int(ids[i]),new=float(X[c].iloc[i]),cached=float(gold[c].iloc[i])) for i in np.flatnonzero(mask)[:8]]))
    result=dict(status='PASS' if not diffs else 'FAIL',source=source,source_sha256=rec['pcap_sha256'],targets=len(ids),numeric_features=len(nums),categorical_features=len(cats),differences=diffs,external_model_scoring_performed=False,metadata=meta,history=history)
    (OUT/'extractor_alignment_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ['metadata','history']},indent=2));assert not diffs,'Extractor must match before external scoring'
if __name__=='__main__':main()
