"""Verify displayed class recalls and retain the historical audit separately.
No training, new prediction, or estimator deserialization.
"""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent/'files');parser.add_argument('--output',type=Path)
    args=parser.parse_args();base=args.data_root.resolve();here=Path(__file__).resolve().parent
    old=subprocess.run([sys.executable,str(here/'verify_historical_saved_results.py'),'--data-root',str(base)],capture_output=True,text=True)
    assert old.returncode==0,old.stdout+old.stderr
    historical=json.loads(old.stdout)
    historical['historical_ten_predictor_summary_checked']=historical.pop('compact_comparator_table_checked')
    target=pd.read_csv(here/'current_s2_recall.csv');checks=0
    for protocol,prefix in [('Temporal','temporal'),('Packet-private','private'),('Shared-signature','signature')]:
        paths=sorted((base/'runs/ARRAY_GNB_FULL_20261119_01').glob(prefix+'*/evaluation.npz'))
        assert len(paths)==(3 if prefix=='temporal' else 2)
        for method,label in [('WV','WV'),('GNB_default','GNB default'),('GNB_G_selected','GNB G-selected')]:
            a=[]
            for path in paths:
                with np.load(path,allow_pickle=False) as z:a.append((z['y'].copy(),z[method].copy(),z['row_ids'].copy()))
            if prefix=='temporal':v=np.mean([[np.mean(p[y==c]==c) for c in range(5)] for y,p,_ in a],axis=0)
            else:
                y=np.concatenate([x[0] for x in a]);p=np.concatenate([x[1] for x in a]);ids=np.concatenate([x[2] for x in a]);assert len(ids)==len(np.unique(ids))==52584
                v=np.array([np.mean(p[y==c]==c) for c in range(5)])
            row=target[(target.protocol==protocol)&(target.predictor==label)].iloc[0]
            assert np.allclose(100*v,row[['Normal','DDoS','Scanning','SQL','Ransomware']].to_numpy(float),atol=1e-10,rtol=0)
            checks+=5
    outcome={'status':'PASS_GNB_FOCUS_SAVED_RESULTS','current_S2_class_recall_values_verified':checks,'displayed_comparator':'GNB default and G-selected','historical_evidence_preserved_and_checked':historical,'new_training':0,'new_predictions':0,'model_deserialization':0,'scope':'Editorial reporting subset. Historical alternatives remain in the supplied archive; no independent classifier ranking is established.'}
    if args.output:args.output.write_text(json.dumps(outcome,indent=2),'utf-8')
    print(json.dumps(outcome,indent=2))
if __name__=='__main__':main()
