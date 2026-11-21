"""Label-blind PCAP -> inherited context85 bridge for independent evaluation."""
from pathlib import Path
import csv,hashlib,json,subprocess
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from e4 import DomainFeatures
from array_revision import frame
from prepare_array_history_20261103 import AuditedRing
from sbfe import iter_pcap
from wb_profile import LocalSource

TSHARK=Path('C:/Program Files/Wireshark/tshark.exe')
NUMERIC_CATEGORIES=set(e.CATEGORICAL_KEEP)-{'http.request.method','http.request.version','mqtt.protoname'}
FIELDS=list(dict.fromkeys(e.NUMERIC_KEEP+e.CATEGORICAL_KEEP+['dns.qry.name.len','frame.number','frame.time_epoch','ip.src','ip.dst','tcp.checksum','tcp.srcport','tcp.dstport','udp.port','http.request.uri','eth.src','eth.dst']))

def numeric(value):
    s=str(value).strip()
    if s in ['','0','0.0']:return 0.
    if s.lower() in ['true','false']:return float(s.lower()=='true')
    if s.lower().startswith('0x'):return float(int(s,16))
    x=float(s)
    if not np.isfinite(x):raise ValueError('Nonfinite extracted protocol value')
    return x

def raw_from_fields(tshark):
    if not set(FIELDS)<=set(tshark):raise ValueError('Missing extractor fields, not protocol absence')
    out={}
    for c in e.NUMERIC_KEEP+['dns.qry.name.len']:out[c]=tshark[c].map(numeric)
    for c in e.CATEGORICAL_KEEP:
        if c in NUMERIC_CATEGORIES:out[c]=tshark[c].map(lambda v:e.normalize_absent(str(float(numeric(v)))))
        else:out[c]=tshark[c].map(e.normalize_absent)
    return pd.DataFrame(out,index=tshark.index)

def extract_protocol(pcap,ids,destination,limit=None):
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)==0:raise ValueError('No target packets')
    target_numbers=(ids+1).tolist();last=int(max(target_numbers));assert limit is None or last<=limit
    expr='frame.number in {'+','.join(map(str,target_numbers))+'}'
    command=[str(TSHARK),'-n','-r',str(pcap),'-c',str(last),'-Y',expr,'-T','fields','-E','header=y','-E','separator=,','-E','quote=d','-E','occurrence=f']
    for name in FIELDS:command.extend(['-e',name])
    with Path(destination).open('w',encoding='utf8',newline='') as output:
        p=subprocess.run(command,stdout=output,stderr=subprocess.PIPE,text=True,encoding='utf8')
    if p.returncode:raise RuntimeError(p.stderr[-3000:])
    fields=pd.read_csv(destination,dtype=str,keep_default_na=False)
    fields.index=pd.to_numeric(fields['frame.number']).to_numpy(np.int64)-1
    if fields.index.duplicated().any() or set(fields.index)!=set(ids):raise ValueError('Packet extraction alignment failure')
    return fields.loc[ids],dict(command=command,stderr=p.stderr)

def contexts_for(pcap,ids):
    wanted=set(map(int,ids));ring=AuditedRing(16384);values={};clock=None;start=None;time_meta={}
    for index,t,sec,nano,packet in iter_pcap(LocalSource(),Path(pcap)):
        if start is None:start=t
        clock=max(t,clock if clock is not None else t)
        if index in wanted:
            values[index]=ring.context(clock,packet.src,packet.dst)
            time_meta[index]=dict(epoch_ns=t,logical_elapsed_seconds=(clock-start)/1e9,wire_bytes=packet.wire,protocol=packet.protocol)
        ring.push(clock,packet)
        if index>=max(wanted):break
    if set(values)!=wanted:raise ValueError('Target packet missing from PCAP')
    return np.asarray([values[int(i)] for i in ids]),dict(capacity_evictions=ring.evicted_capacity,backward_steps=ring.backward,targets=[dict(packet_index=int(i),**time_meta[int(i)]) for i in ids],query_before_push=True)

def context85(protocol,values):
    raw=raw_from_fields(protocol);X,nums,cats=frame(raw,values,'c60_o10')
    if not np.isfinite(X[nums].to_numpy(float)).all():raise ValueError('Nonfinite final features')
    return X,nums,cats

def scan_targets(pcap,maximum=1000):
    start=None;clock=None;first=None;count=0;backward=0;last_t=None
    for index,t,sec,nano,packet in iter_pcap(LocalSource(),Path(pcap)):
        if start is None:start=t
        if clock is not None and t<clock:backward+=1
        clock=max(t,clock if clock is not None else t);last_t=t;count=index+1
        if first is None and clock-start>=10_000_000_000:first=index
    if first is None:return np.array([],dtype=np.int64),dict(total_packets=count,eligible_packets=0,duration_seconds=(clock-start)/1e9,backward_steps=backward)
    ids=np.unique(np.linspace(first,count-1,min(maximum,count-first),dtype=np.int64))
    return ids,dict(total_packets=count,first_eligible_index=first,eligible_packets=count-first,selected=len(ids),duration_seconds=(clock-start)/1e9,backward_steps=backward)
