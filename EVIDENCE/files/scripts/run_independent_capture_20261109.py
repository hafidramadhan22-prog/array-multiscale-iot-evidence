"""Evaluate fixed project-owned predictors on new PCAP captures, without fit."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from independent_capture_20261109 import extract_protocol,contexts_for,context85,scan_targets,TSHARK
from predict_fpr_sql_guard_rf_rescue_20261109 import verify_run
from sql_guard_rf_rescue_20261109 import combine
from fpr_target8_20261109 import NAMES,CLASSES,alternatives,metrics
RUN=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01';REPORT=ROOT/'reports/INDEPENDENT_CAPTURE_20261109';DATA=ROOT/'data/external/CICIoT2023_20261109'
SOURCES={
 'CIC_Normal3':('PCAP/Benign_Final/BenignTraffic3.pcap',0,'known_benign_capture'),
 'CIC_Normal0':('PCAP/Benign_Final/BenignTraffic.pcap',0,'known_benign_capture'),
 'CIC_SQL':('PCAP/SqlInjection/SqlInjection.pcap',3,'capture_scenario_not_verified_packet_attack'),
 'CIC_PortScan':('PCAP/Recon-PortScan/Recon-PortScan.pcap',2,'capture_scenario_not_verified_packet_attack'),
 'CIC_DDoS_SYN':('PCAP/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap',1,'capture_scenario_not_verified_packet_attack'),
}
FILES=['scripts/run_independent_capture_20261109.py','src/independent_capture_20261109.py','scripts/audit_independent_extractor_20261109.py','src/sql_guard_rf_rescue_20261109.py','src/rf_normal_voting_type_20261109.py','src/rf_normal_voting_type_v2_20261109.py','src/array_revision.py','src/e4.py','src/sbfe.py','src/wb_behavior.py','src/multiscale_context.py','src/ta_hafid_engine.py','scripts/prepare_array_history_20261103.py','configs/fpr_normal_voting_schema_20261109.json']
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):p.write_text(json.dumps(e.jsonable(x),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def event(text):
 with (RUN/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),event=text))+'\n')
 print(text,flush=True)
def freeze():
 assert read(REPORT/'extractor_alignment_audit.json')['status']=='PASS'
 out=RUN/'source_manifest.json'
 if out.exists():
  for row in read(out):assert e.sha256(ROOT/row['path'])==row['sha256'],row['path']
  return
 rows=[]
 for rel in FILES:shutil.copy2(ROOT/rel,RUN/'source'/Path(rel).name);rows.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
 write(out,rows);write(RUN/'environment.json',e.environment());event('INFERENCE_SOURCE_FROZEN_BEFORE_FIRST_EXTERNAL_SCORING')
def owned_download(path):
 manifests=[]
 for p in [REPORT/'cic_download_manifest.json',REPORT/'lfs_download_manifest.json']:
  if p.exists():manifests.extend(read(p))
 rows=[r for r in manifests if (Path(r.get('destination','')).resolve()==path.resolve() if r.get('destination') else DATA/r['path']==path)]
 assert rows,'No verified download provenance'
 assert all(path.stat().st_size==r['bytes'] and e.sha256(path)==r['sha256'] for r in rows)
 return rows
def capmeta(path):
 cmd=[str(TSHARK.parent/'capinfos.exe'),'-c','-a','-e','-E','-T','-t','-S',str(path)];p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8');assert p.returncode==0,p.stderr
 lines=p.stdout.splitlines();a=dict(zip(lines[0].split('\t'),lines[1].split('\t')));return a
def prepare(name):
 freeze();rel,label,scope=SOURCES[name];raw=DATA/rel;assert raw.exists(),'PCAP not downloaded yet';provenance=owned_download(raw)
 old=read(ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04/source_audit.json');assert e.sha256(raw) not in {r['pcap_sha256'] for r in old}
 folder=RUN/name;assert not folder.exists(),str(folder);folder.mkdir();before=capmeta(raw);pcap=raw
 if before['File type']=='pcapng':
  pcap=raw.with_name(raw.stem+'_ethernet_nsec.pcap')
  if not pcap.exists():
   p=subprocess.run([str(TSHARK.parent/'editcap.exe'),'-F','nsecpcap',str(raw),str(pcap)],capture_output=True,text=True);assert p.returncode==0,p.stderr
  after=capmeta(pcap)
  for k in ['File encapsulation','Number of packets','Start time','End time']:assert before[k]==after[k],(k,before[k],after[k])
 else:after=before
 assert before['File encapsulation']=='ether','Inherited decoder requires Ethernet'
 ids,selection=scan_targets(pcap,1000);assert len(ids)>0,'No targets after unchanged10-second warm-up'
 np.save(folder/'packet_indices.npy',ids,allow_pickle=False);write(folder/'selection_frozen.json',selection);event('TARGETS_FROZEN '+name+' n='+str(len(ids)))
 protocol,meta=extract_protocol(pcap,ids,folder/'protocol_fields.csv');values,history=contexts_for(pcap,ids);X,nums,cats=context85(protocol,values)
 schema=read(ROOT/'configs/fpr_normal_voting_schema_20261109.json');assert nums==schema['numeric'] and cats==schema['categorical']
 X=X[nums+cats];X.index=[name+':'+str(i) for i in ids];X.index.name='row_id';X.to_csv(folder/'features.csv')
 metadata=pd.DataFrame(history['targets']);metadata['row_id']=X.index;metadata['source']=name;metadata['scenario_label']=CLASSES[label];metadata['label_scope']=scope
 for c in ['ip.src','ip.dst','eth.src','eth.dst','http.request.uri']:metadata[c]=protocol[c].to_numpy()
 metadata.to_csv(folder/'target_metadata.csv',index=False)
 write(folder/'preparation.json',dict(status='PREPARED_NOT_SCORED',source=name,label=label,label_scope=scope,raw_path=str(raw.relative_to(ROOT)),raw_sha256=e.sha256(raw),converted_path=str(pcap.relative_to(ROOT)),converted_sha256=e.sha256(pcap),raw_metadata=before,converted_metadata=after,provenance=provenance,selection=selection,history=history,tshark=meta,feature_sha256=e.sha256(folder/'features.csv')));event('FEATURES_PREPARED '+name)
def score(name):
 freeze();folder=RUN/name;prep=read(folder/'preparation.json');assert prep['status']=='PREPARED_NOT_SCORED';assert not (folder/'scored.json').exists();assert e.sha256(folder/'features.csv')==prep['feature_sha256']
 model_path=verify_run(ROOT/'runs/FPR_SQL_GUARD_20261109_01');assert e.sha256(model_path)==read(RUN/'configuration_freeze.json')['model_sha256']
 import joblib
 bundle=joblib.load(model_path);schema=read(ROOT/'configs/fpr_normal_voting_schema_20261109.json');X=pd.read_csv(folder/'features.csv',index_col='row_id',dtype={c:str for c in schema['categorical']},keep_default_na=False,float_precision='round_trip');X=X[schema['numeric']+schema['categorical']]
 for c in schema['numeric']:X[c]=pd.to_numeric(X[c],errors='raise')
 with threadpool_limits(limits=1):cube=np.stack([np.asarray(bundle['models'][n].predict_proba(X),dtype=float) for n in NAMES],axis=1)
 probs=alternatives(cube,bundle['weights']);p,reason=combine(probs['weighted_reference'],probs['random_forest']);probs['sql_guard_rf_rescue']=p;y=np.full(len(X),prep['label'],int);rows=[];decisions=[]
 np.savez_compressed(folder/'probabilities.npz',row_ids=np.asarray(X.index,dtype=str),y_scenario=y,**probs,reason=reason)
 for name_policy in ['weighted_reference','random_forest','sql_guard_rf_rescue']:
  pp=probs[name_policy];m,cm=metrics(y,pp.argmax(1));rows.append(dict(source=name,policy=name_policy,label_scope=prep['label_scope'],**m));pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(folder/('CM_'+name_policy+'.csv'))
  decisions.extend(dict(row_id=rid,source=name,policy=name_policy,scenario_truth=CLASSES[int(yy)],label_scope=prep['label_scope'],prediction=CLASSES[int(pred)],**{'p_'+c:float(qq[k]) for k,c in enumerate(CLASSES)}) for rid,yy,pred,qq in zip(X.index,y,pp.argmax(1),pp))
 pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(decisions).to_csv(folder/'predictions.csv',index=False)
 write(folder/'scored.json',dict(status='SCORED',source=name,n=len(X),label_scope=prep['label_scope'],new_fits=0,model_sha256=e.sha256(model_path)));event('SCORED '+name)
 print(pd.DataFrame(rows)[['policy','n','FPR_percent','attack_to_Normal','errors']].to_string(index=False),flush=True)
def summarize():
 rows=[]
 for folder in RUN.iterdir():
  if folder.is_dir() and (folder/'metrics.csv').exists():rows.append(pd.read_csv(folder/'metrics.csv'))
 if rows:pd.concat(rows).to_csv(RUN/'metrics_available.csv',index=False)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','score','summarize'],required=True);ap.add_argument('--source',choices=list(SOURCES));args=ap.parse_args()
 if args.phase=='prepare':prepare(args.source)
 elif args.phase=='score':score(args.source)
 else:summarize()
