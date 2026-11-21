"""Frozen-model evaluation on uniquely author-flow-labelled TON packet targets."""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from decimal import Decimal
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from independent_capture_20261109 import extract_protocol,contexts_for,context85
from predict_fpr_sql_guard_rf_rescue_20261109 import verify_run
from fpr_target8_20261109 import NAMES,CLASSES,alternatives,metrics
from sql_guard_rf_rescue_20261109 import combine
from run_independent_capture_v2_20261109 import freeze as inherited_freeze
RUN=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01';MAP=RUN/'TON_flow_mapping'
SOURCES={'TON_Ransom1':'normal_runsomware1','TON_Ransom2':'normal_runsomware2'}
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):p.write_text(json.dumps(e.jsonable(x),indent=2)+'\n',encoding='utf8')
def freeze():
    inherited_freeze();assert read(MAP/'mapping_complete.json')['status']=='LABELS_AND_TARGETS_FROZEN_BEFORE_TON_SCORING'
    for row in read(MAP/'source_manifest.json'):assert e.sha256(ROOT/row['path'])==row['sha256']
    fast=read(MAP/'fast_source_manifest.json');assert e.sha256(ROOT/fast['path'])==fast['sha256']
    p=RUN/'ton_execution_source_manifest.json';paths=[Path(__file__),ROOT/'docs/INDEPENDENT_TON_FLOW_LABEL_ADDENDUM_20261109.md']
    if p.exists():
        for row in read(p):assert e.sha256(ROOT/row['path'])==row['sha256']
    else:
        assert not list(RUN.glob('TON_Ransom*/scored.json'));rows=[]
        for path in paths:shutil.copy2(path,RUN/'source'/path.name);rows.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=e.sha256(path)))
        write(p,rows)
def prepare(source):
    freeze();name=SOURCES[source];selection_path=MAP/(name+'_selected_targets.csv');selected=pd.read_csv(selection_path);ids=selected.packet_index.to_numpy(np.int64);assert len(ids)>0 and selected.packet_index.is_unique
    info=next(m for m in read(MAP/'packet_index_complete.json')['sources'] if m['name']==name);pcap=ROOT/info['pcap_path'];assert e.sha256(pcap)==info['pcap_sha256'];folder=RUN/source;assert not folder.exists();folder.mkdir();write(folder/'selection_frozen.json',dict(path=str(selection_path.relative_to(ROOT)),sha256=e.sha256(selection_path),n=len(ids),counts=selected.label.value_counts().to_dict(),label_scope='author-derived flow labels, one-second quantization, unique bidirectional5tuple interval'))
    print('TON_TARGETS_FROZEN '+source+' '+str(len(ids)),flush=True);fields,command=extract_protocol(pcap,ids,folder/'protocol_fields.csv');values,history=contexts_for(pcap,ids);X,nums,cats=context85(fields,values);schema=read(ROOT/'configs/fpr_normal_voting_schema_20261109.json');assert nums==schema['numeric'] and cats==schema['categorical'];X=X[nums+cats];X.index=[source+':'+str(i) for i in ids];X.index.name='row_id'
    times=np.array([int(Decimal(t)*Decimal(10**9)) for t in fields['frame.time_epoch']],dtype=np.int64);np.testing.assert_array_equal(times,selected.epoch_ns.to_numpy(np.int64));X.to_csv(folder/'features.csv');selected['row_id']=X.index;selected.to_csv(folder/'truth_and_packet_indices.csv',index=False)
    write(folder/'preparation.json',dict(status='PREPARED_NOT_SCORED',pcap_path=info['pcap_path'],pcap_sha256=info['pcap_sha256'],feature_sha256=e.sha256(folder/'features.csv'),truth_sha256=e.sha256(folder/'truth_and_packet_indices.csv'),mapping_complete_sha256=e.sha256(MAP/'mapping_complete.json'),history=history,tshark=command));print('TON_FEATURES_PREPARED '+source,flush=True)
def score(source):
    freeze();folder=RUN/source;prep=read(folder/'preparation.json');assert not (folder/'scored.json').exists();assert e.sha256(folder/'features.csv')==prep['feature_sha256'] and e.sha256(folder/'truth_and_packet_indices.csv')==prep['truth_sha256'];schema=read(ROOT/'configs/fpr_normal_voting_schema_20261109.json');X=pd.read_csv(folder/'features.csv',index_col='row_id',dtype={c:str for c in schema['categorical']},keep_default_na=False,float_precision='round_trip');truth=pd.read_csv(folder/'truth_and_packet_indices.csv',index_col='row_id').loc[X.index];y=truth.label.to_numpy(np.int64);assert np.isin(y,[0,4]).all()
    model_path=verify_run(ROOT/'runs/FPR_SQL_GUARD_20261109_01');assert e.sha256(model_path)==read(RUN/'configuration_freeze.json')['model_sha256'];import joblib
    bundle=joblib.load(model_path)
    with threadpool_limits(limits=1):cube=np.stack([np.asarray(bundle['models'][n].predict_proba(X),dtype=float) for n in NAMES],axis=1)
    probs=alternatives(cube,bundle['weights']);p,reason=combine(probs['weighted_reference'],probs['random_forest']);probs['sql_guard_rf_rescue']=p;np.savez_compressed(folder/'probabilities.npz',row_ids=np.asarray(X.index,dtype=str),y=y,**probs,reason=reason);rows=[];decisions=[]
    for policy in ['weighted_reference','random_forest','sql_guard_rf_rescue']:
        p=probs[policy];m,cm=metrics(y,p.argmax(1));rows.append(dict(source=source,policy=policy,label_scope='author_flow_derived',**m));pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(folder/('CM_'+policy+'.csv'))
        decisions.extend(dict(row_id=rid,source=source,policy=policy,truth=CLASSES[int(yy)],label_scope='author_flow_derived',prediction=CLASSES[int(pred)],**{'p_'+c:float(qq[k]) for k,c in enumerate(CLASSES)}) for rid,yy,pred,qq in zip(X.index,y,p.argmax(1),p))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(decisions).to_csv(folder/'predictions.csv',index=False);write(folder/'scored.json',dict(status='SCORED',source=source,n=len(X),normal=int((y==0).sum()),ransomware=int((y==4).sum()),new_fits=0,model_sha256=e.sha256(model_path),label_scope='author_flow_derived'))
    print(pd.DataFrame(rows)[['source','policy','normal_support','attack_support','FPR_percent','attack_to_Normal','recall_Ransomware']].to_string(index=False),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','score'],required=True);ap.add_argument('--source',choices=list(SOURCES),required=True);a=ap.parse_args()
    if a.phase=='prepare':prepare(a.source)
    else:score(a.source)
