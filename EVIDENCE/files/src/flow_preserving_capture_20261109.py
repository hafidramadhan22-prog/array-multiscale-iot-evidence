"""Keep complete selected TCP flows for dissection; history uses EVERY packet.

All non-TCP packets, IP fragments, and TCP DNS packets remain in the dissection
subsequence. Packet bytes/order/timestamps are preserved. Labels are not read.
"""
from pathlib import Path
import struct,json
import numpy as np
import pandas as pd
from independent_capture_20261109 import contexts_for,extract_protocol,context85
from prepare_array_history_20261103 import AuditedRing
from sbfe import iter_pcap
from wb_profile import LocalSource

def key(packet):
    a=(packet.src,packet.sp);b=(packet.dst,packet.dp);return (*min(a,b),*max(a,b))
def keep_packet(packet,selected_keys):
    return packet.protocol!=6 or bool(packet.fragment) or packet.sp==53 or packet.dp==53 or key(packet) in selected_keys
def prepare(pcap,ids,destination,target_packets):
    ids=np.asarray(ids,dtype=np.int64);wanted=set(map(int,ids));keys={key(p) for i,t,s,n,p in iter_pcap(LocalSource(),Path(target_packets)) if p.protocol==6};ring=AuditedRing(16384);clock=None;start=None;values={};times={};mapping={};kept=0;seen=0;last=max(wanted)
    pcap=Path(pcap);destination=Path(destination);assert not destination.exists()
    with pcap.open('rb') as f:global_header=f.read(24)
    mag=global_header[:4].hex();endian='<' if mag in ['4d3cb2a1','d4c3b2a1'] else '>';nano=mag in ['4d3cb2a1','a1b23c4d'];pack=struct.Struct(endian+'IIII').pack
    with destination.open('xb') as output:
        output.write(global_header)
        for i,t,sec,ns,p in iter_pcap(LocalSource(),pcap):
            if start is None:start=t
            clock=max(t,clock if clock is not None else t)
            keep=keep_packet(p,keys)
            if i in wanted:
                assert keep;mapping[i]=kept;values[i]=ring.context(clock,p.src,p.dst);times[i]=t
            if keep:
                output.write(pack(sec,ns if nano else ns//1000,len(p.raw),p.wire));output.write(p.raw);kept+=1
            ring.push(clock,p);seen=i+1
            if i==last:break
    assert set(values)==wanted
    return np.asarray([mapping[int(i)] for i in ids]),np.asarray([values[int(i)] for i in ids]),dict(original_prefix_packets=seen,retained_packets=kept,selected_TCP_flow_keys=len(keys),target_epoch_ns=[times[int(i)] for i in ids],query_before_push=True,history_all_original_packets=True,capacity_evictions=ring.evicted_capacity)

def selected_copy(pcap,ids,dest):
    # Fast packet-identity copy; never fabricate data or timestamps.
    import struct
    wanted=set(map(int,ids));last=max(wanted);count=0
    with Path(pcap).open('rb') as source,Path(dest).open('xb') as output:
        h=source.read(24);output.write(h);endian='<' if h[:4].hex() in ['d4c3b2a1','4d3cb2a1'] else '>';unpack=struct.Struct(endian+'IIII').unpack;i=0
        while i<=last:
            header=source.read(16);assert len(header)==16;_,_,cap,_=unpack(header)
            if i in wanted:output.write(header);b=source.read(cap);assert len(b)==cap;output.write(b);count+=1
            else:source.seek(cap,1)
            i+=1
    assert count==len(wanted)
