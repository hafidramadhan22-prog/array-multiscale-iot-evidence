"""Descriptive post-warmup subset, no selection, fitting or new test claim."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from e4 import metrics
from analyze_context_repair import evaluation,labels
from run_context_repair import read_npz

def run(run):
    cfg=json.loads((run/'configuration_freeze.json').read_text())
    data=read_npz(ROOT/cfg['data_folder']/'packet_features.npz');lookup={int(r):i for i,r in enumerate(data['row_ids'])}
    rows=[]
    for stage in json.loads((run/'planned_stages.json').read_text()):
        for case in cfg['cases']:
            for view in ['baseline25','baseline29']+cfg['views']:
                ev,sel=evaluation(run,cfg,stage,view,case)
                ix=np.array([lookup[int(r)] for r in ev['row_ids']]);mask=~data['partial_10s_history'][ix]
                # Same IDs for both compared methods; no deleting error rows.
                assert set(ev['y'][mask])==set(range(5))
                for method in ['weighted_voting','logistic_regression','random_forest']:
                    pred=labels(ev,sel,method)
                    rows.append(dict(stage=stage,case=case,view=view,method=method,subset='elapsed_since_reset_at_least_10s',
                        **metrics(ev['y'][mask],pred[mask])))
    table=pd.DataFrame(rows);table.to_csv(run/'tables/post_warmup_metrics.csv',index=False)
    summary=table[table.stage.str.startswith('temporal')].groupby(['case','view','method'])[
        ['macro_f1','ddos_f2','recall_Port_Scanning','recall_Ransomware','n']].mean().reset_index()
    summary.to_csv(run/'tables/post_warmup_temporal_summary.csv',index=False)
    print(summary[(summary['case']=='balanced')&(summary.method=='weighted_voting')].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
