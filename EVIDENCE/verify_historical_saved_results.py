"""Reconstruct selected manuscript evidence from saved arrays and raw logs only.
Portable copy: python verify_saved_results.py --data-root files
Requires numpy, pandas, scikit-learn. Never loads estimators or performs fitting.
"""
from pathlib import Path
import argparse,json,re,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,balanced_accuracy_score,fbeta_score

def metric(y,p):
    return dict(macro_f1=f1_score(y,p,labels=range(5),average='macro',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,p),normal_fpr=np.mean(p[y==0]!=0),
        ddos_f2=fbeta_score(y==1,p==1,beta=2,zero_division=0),errors=float(np.sum(y!=p)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent/'files');parser.add_argument('--output',type=Path)
    args=parser.parse_args();base=args.data_root.resolve()
    root_report=base/'reports/ARRAY_CLEAN_SUPERVISOR_20261121_01'
    items=json.loads((root_report/'docx_structure.json').read_text('utf-8'))
    tables=[x['rows'] for x in items if x['kind']=='table']
    stages={};vectors=0
    stage_names=['temporal_seed42','temporal_seed43','temporal_seed44','private_fold1','private_fold2','signature_fold1','signature_fold2']
    for stage in stage_names:
        sm=base/'runs/ARRAY_SVM_EVALUATION_20261118_01/stages'/stage/'evaluation.npz'
        gn=base/'runs/ARRAY_GNB_FULL_20261119_01'/stage/'evaluation.npz'
        kn=base/'reports/ARRAY_KNN_20261119'/f'{stage}_predictions.csv.gz'
        with np.load(sm,allow_pickle=False) as z:
            data={k:z[k].copy() for k in ['y','row_ids','WV','RF','LR','Uniform','SVM_default','SVM_G_selected']}
        with np.load(gn,allow_pickle=False) as z:
            # GNB persisted decimal IDs as Unicode; other runs used int64 IDs.
            # Compare their exact canonical decimal strings, preserving row order.
            assert all(re.fullmatch(r'0|[1-9][0-9]*',str(x)) for x in z['row_ids'])
            assert np.array_equal(z['y'],data['y']) and np.array_equal(z['row_ids'].astype(str),data['row_ids'].astype(str))
            for k in ['GNB_default','GNB_G_selected']:data[k]=z[k].copy()
        kdf=pd.read_csv(kn)
        assert np.array_equal(kdf.y,data['y']) and np.array_equal(kdf.row_ids,data['row_ids'])
        for k in ['KNN_default','KNN_G_selected']:data[k]=kdf[k].to_numpy()
        stages[stage]=data;vectors+=10
    rows=[]
    methods=['WV','RF','LR','Uniform','SVM_default','SVM_G_selected','KNN_default','KNN_G_selected','GNB_default','GNB_G_selected']
    for protocol,prefix in [('Temporal','temporal'),('Private history','private'),('Shared signature','signature')]:
        group=[v for n,v in stages.items() if n.startswith(prefix)]
        for method in methods:
            if prefix=='temporal':
                vals=[metric(z['y'],z[method]) for z in group]
                row={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
            else:
                ids=np.concatenate([v['row_ids'] for v in group]);assert len(ids)==len(np.unique(ids))==52584
                row=metric(np.concatenate([v['y'] for v in group]),np.concatenate([v[method] for v in group]))
            rows.append(dict(protocol=protocol,method=method,**row))
    scores=pd.DataFrame(rows)
    lookup={'WV (reference)':'WV','GNB (default)':'GNB_default','GNB (G-selected)':'GNB_G_selected'}
    for values in tables[2][1:]:
        protocol,name,*numbers=values;r=scores[(scores.protocol==protocol)&(scores.method==lookup[name])].iloc[0]
        expected=[100*r.macro_f1,100*r.balanced_accuracy,100*r.normal_fpr,100*r.ddos_f2,r.errors]
        for value,exact in zip(numbers,expected):assert abs(float(value.replace(',',''))-exact)<=.00501,(protocol,name,value,exact)
    # Compact Table S2 is checked against the ten retained predictor families.
    s2=pd.read_csv(root_report/'compact_s2_values.csv')
    for row in s2.to_dict('records'):
        for protocol,column in [('Temporal','Temporal'),('Private history','Packet-private'),('Shared signature','Shared-signature')]:
            r=scores[(scores.protocol==protocol)&(scores.method==row['method'])].iloc[0]
            assert abs(row[column]-100*r.macro_f1)<.00501,(row['method'],protocol)
    serving=base/'runs/ARRAY_SERVING_ABLATION_20261117_01'
    latency=pd.read_csv(serving/'latency_raw.csv');replay=pd.read_csv(serving/'replay_raw.csv');memory=pd.read_csv(serving/'memory_raw.csv')
    assert len(latency)==480 and len(replay)==240 and len(memory)==80
    assert replay.labels_match.all() and memory.labels_match.all()
    for cells in tables[3][1:]:
        policy=cells[0].replace(' / ','_').replace(' ','_')
        vals=[1000*latency[(latency.policy==policy)&(latency.batch==n)].seconds.median() for n in [1,1000]]
        vals += [memory[(memory.policy==policy)&(memory.workload=='feature')].delta_mib.mean(),replay[replay.policy==policy].total_seconds.mean(),memory[(memory.policy==policy)&(memory.workload=='replay')].delta_mib.mean()]
        for value,exact in zip(cells[1:],vals):
            decimals=len(value.split('.')[1]) if '.' in value else 0
            assert abs(float(value)-exact)<=.5*10**(-decimals)+1e-8,(policy,value,exact)
    means=replay.groupby('policy').total_seconds.mean()
    gains={
        'combined_vs_native_full_percent':100*(1-means['WV_compact_certified']/means['WV_native_full']),
        'encoding_vs_native_certified_percent':100*(1-means['WV_compact_certified']/means['WV_native_certified'])}
    assert abs(gains['combined_vs_native_full_percent']-60.55)<.0051
    assert abs(gains['encoding_vs_native_certified_percent']-37.40)<.0051
    coverage=pd.read_csv(base/'reports/INDEPENDENT_CAPTURE_20261109/TON_label_coverage.csv')
    computed=100*(coverage.normal+coverage.ransomware)/coverage.total
    assert np.allclose(computed,coverage.in_scope_labelled_percent,atol=1e-10)
    assert np.allclose(np.round(computed,4),[62.9036,6.6602])
    factorial=pd.read_csv(base/'reports/ARRAY_MAJOR_REVISION_20261103/factorial_summary.csv')
    factorial=factorial[factorial.method=='weighted_voting']
    for cells,view in zip(tables[0][1:],['c40_o1','c40_o10','c60_o1','c60_o10']):
        t=factorial[(factorial.view==view)&(factorial.protocol=='temporal')].iloc[0]
        s=factorial[(factorial.view==view)&(factorial.protocol=='signature')].iloc[0]
        vals=[100*t.macro_f1,100*t.balanced_accuracy,t.errors,100*s.macro_f1,s.errors]
        for value,exact in zip(cells[1:],vals):
            v=value.replace(',','');decimals=len(v.split('.')[1]) if '.' in v else 0
            assert abs(float(v)-exact)<=.5*10**(-decimals)+1e-8,(view,value,exact)
    faith=pd.read_csv(base/'reports/ARRAY_MAJOR_REVISION_20261103/xai_faithfulness.csv').set_index('method').loc['weighted_voting']
    for value,key in zip(tables[4][1][1:],['deletion_auc','insertion_auc','deletion_gain','insertion_gain']):assert abs(float(value)-faith[key])<.000051
    # FPR denominators in the manuscript remain Normal support, never all rows.
    for cells in tables[5][1:]:
        n=int(cells[1].replace(',',''))
        for value in cells[2:4]:
            fp,percent=re.match(r'(\d+) \(([\d.]+)%\)',value).groups()
            assert abs(100*int(fp)/n-float(percent))<.000051
    outcome=dict(status='PASS_SELECTED_SAVED_RESULT_RECONSTRUCTION',prediction_vectors_recomputed=vectors,
        ten_predictors_three_protocols=True,main_tables_checked=[1,3,4,5,6],
        compact_comparator_table_checked=True,coverage_denominator='all raw packets including warm-up',
        runtime_raw_counts={'feature_calls':len(latency),'replays':len(replay),'RSS_workers':len(memory)},
        runtime_gains=gains,new_model_fits=0,new_predictions=0,model_deserializations=0,
        row_id_storage='GNB Unicode decimal IDs equal int64-run IDs after exact canonical decimal representation; order and labels unchanged',
        scope='Selected saved-result reconstruction; not a new validation run, raw-PCAP extraction, full training reproduction or re-enumeration of every Shapley coalition.')
    if args.output:args.output.write_text(json.dumps(outcome,indent=2),'utf-8')
    print(json.dumps(outcome,indent=2))
if __name__=='__main__':main()
