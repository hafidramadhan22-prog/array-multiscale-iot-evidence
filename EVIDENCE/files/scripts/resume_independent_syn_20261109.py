"""Resume interrupted SYN extraction on exactly the previously frozen targets."""
from pathlib import Path
import json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
import run_independent_capture_20261109 as r
from run_independent_capture_v2_20261109 import freeze,canonical_capmeta
from independent_capture_20261109 import extract_protocol,contexts_for,context85
def main():
    freeze();name='CIC_DDoS_SYN';folder=r.RUN/name;assert not (folder/'preparation.json').exists() and not (folder/'scored.json').exists()
    ids=np.load(folder/'packet_indices.npy',allow_pickle=False);selection=r.read(folder/'selection_frozen.json');assert len(ids)==1000 and len(set(ids))==1000 and ids[0]==selection['first_eligible_index'] and ids[-1]==selection['total_packets']-1
    extra=r.RUN/'syn_resume_source_manifest.json';source=Path(__file__)
    if extra.exists():assert r.read(extra)['sha256']==e.sha256(source)
    else:shutil.copy2(source,r.RUN/'source'/source.name);r.write(extra,dict(path=source.relative_to(ROOT).as_posix(),sha256=e.sha256(source),packet_indices_sha256=e.sha256(folder/'packet_indices.npy'),reason='Initial concurrent extraction interrupted when available RAM was0.17GiB; same fields/model/targets, sequential retry'))
    raw=r.DATA/r.SOURCES[name][0];provenance=r.owned_download(raw);pcap=raw.with_name(raw.stem+'_ethernet_nsec.pcap');before=canonical_capmeta(raw);after=canonical_capmeta(pcap)
    for k in ['File encapsulation','Number of packets','Start time','End time']:assert before[k]==after[k]
    assert int(after['Number of packets'])==selection['total_packets']
    protocol_file=folder/'protocol_fields.csv'
    if protocol_file.exists():
        history=folder/'protocol_fields_interrupted.csv';assert not history.exists();protocol_file.rename(history)
    protocol,meta=extract_protocol(pcap,ids,protocol_file);values,history=contexts_for(pcap,ids);X,nums,cats=context85(protocol,values);schema=r.read(ROOT/'configs/fpr_normal_voting_schema_20261109.json');assert nums==schema['numeric'] and cats==schema['categorical'];X=X[nums+cats];X.index=[name+':'+str(i) for i in ids];X.index.name='row_id';X.to_csv(folder/'features.csv')
    metadata=pd.DataFrame(history['targets']);metadata['row_id']=X.index;metadata['source']=name;metadata['scenario_label']='DDoS_TCP';metadata['label_scope']=r.SOURCES[name][2]
    for c in ['ip.src','ip.dst','eth.src','eth.dst','http.request.uri']:metadata[c]=protocol[c].to_numpy()
    metadata.to_csv(folder/'target_metadata.csv',index=False)
    r.write(folder/'preparation.json',dict(status='PREPARED_NOT_SCORED',source=name,label=1,label_scope=r.SOURCES[name][2],raw_path=str(raw.relative_to(ROOT)),raw_sha256=e.sha256(raw),converted_path=str(pcap.relative_to(ROOT)),converted_sha256=e.sha256(pcap),raw_metadata=before,converted_metadata=after,provenance=provenance,selection=selection,history=history,tshark=meta,feature_sha256=e.sha256(folder/'features.csv'),resume_used_exact_original_targets=True));r.event('FEATURES_PREPARED '+name+' SAME_FROZEN_TARGETS')
if __name__=='__main__':main()
