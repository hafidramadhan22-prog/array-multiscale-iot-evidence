"""SBFE: causal packet buffer, strict provenance matching and development-only models."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import collections
import csv
import hashlib
import io
import ipaddress
import json
import re
import struct
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from e4 import DomainFeatures, columns, metrics
import ta_hafid_engine as e

HEADER = ["wire_bytes","ip_bytes","protocol","fragment","tcp_payload","udp_payload",
          "syn","ack","rst","fin","psh","urg"]
WINDOW_NAMES=["log_packets","log_bytes","log_sources","log_dstports","mean_bytes","std_bytes",
              "syn_fraction","ack_fraction","rst_fraction","capacity_fraction"]
BEHAVIOR=[f"behavior_{int(w*1000)}ms_{name}" for w in [.1,1.] for name in WINDOW_NAMES]
META=["frame.time","ip.src_host","ip.dst_host","tcp.checksum"]
TIME_RE=re.compile(r"(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\.(\d+)")
ACTIVE={"DDoS TCP SYN Flood Attacks.pcap","Port Scanning attack.pcap",
        "SQL injection attack.pcap","Ransomware attack.pcap"}

def entity(text):
    text=str(text).strip()
    if text in {"0","0.0","","nan"}: return 0
    try:
        address=ipaddress.ip_address(text)
    except ValueError:
        raise ValueError("Unresolved endpoint name cannot be matched silently")
    if address.version==4:return int(address)
    return int.from_bytes(hashlib.blake2b(address.packed,digest_size=8).digest(),"big")%(2**62)+2**62

def number(x):
    text=str(x).strip()
    if text in {"","nan","None"}:return 0
    return int(text,16) if text.lower().startswith("0x") else int(float(text))

def row_identity(row):
    m=TIME_RE.search(str(row["frame.time"]))
    if m is None:raise ValueError("Unsupported canonical clock representation")
    year,hour,minute,second,fraction=m.groups()
    key=(int(minute)*60+int(second),int(fraction.ljust(9,"0")),
         entity(row["ip.src_host"]),entity(row["ip.dst_host"]),number(row["tcp.checksum"]),
         number(row["tcp.srcport"]),number(row["tcp.dstport"]),number(row["tcp.len"]))
    return key,int(year),int(hour)

@dataclass(slots=True)
class Packet:
    src:int=0;dst:int=0;sp:int=0;dp:int=0;checksum:int=0
    wire:int=0;iplen:int=0;protocol:int=0;fragment:int=0
    tcp_payload:int=0;udp_payload:int=0;flags:int=0;raw:bytes=b""
    def header(self):
        return [self.wire,self.iplen,self.protocol,self.fragment,self.tcp_payload,self.udp_payload,
                *[int(bool(self.flags&v)) for v in [2,16,4,1,8,32]]]
    def identity(self,seconds,nanos):
        # CSV ip.* fields are IPv4-only; IPv6 endpoints remain available to state keys.
        src,dst=(0,0) if self.src>=2**62 else (self.src,self.dst)
        return (seconds%3600,nanos,src,dst,self.checksum,self.sp,self.dp,self.tcp_payload)

def decode_packet(data,wire):
    p=Packet(wire=wire,raw=data)
    if len(data)<14:return p
    eth=int.from_bytes(data[12:14],"big");offset=14
    while eth in [0x8100,0x88a8]:
        if len(data)<offset+4:raise ValueError("Truncated VLAN header")
        eth=int.from_bytes(data[offset+2:offset+4],"big");offset+=4
    if eth==0x0800:
        if len(data)<offset+20:raise ValueError("Truncated IPv4 header")
        ihl=(data[offset]&15)*4
        p.iplen=int.from_bytes(data[offset+2:offset+4],"big")
        p.protocol=data[offset+9]
        fragment=int.from_bytes(data[offset+6:offset+8],"big")
        p.fragment=int(bool(fragment&0x3fff))
        p.src=int.from_bytes(data[offset+12:offset+16],"big")
        p.dst=int.from_bytes(data[offset+16:offset+20],"big")
        if fragment&0x1fff:return p
        transport=offset+ihl;remaining=p.iplen-ihl
    elif eth==0x86dd:
        if len(data)<offset+40:raise ValueError("Truncated IPv6 header")
        p.iplen=40+int.from_bytes(data[offset+4:offset+6],"big")
        p.protocol=data[offset+6]
        p.src=entity(str(ipaddress.IPv6Address(data[offset+8:offset+24])))
        p.dst=entity(str(ipaddress.IPv6Address(data[offset+24:offset+40])))
        transport=offset+40;remaining=p.iplen-40
        while p.protocol in [0,43,60,44]:
            if len(data)<transport+8:raise ValueError("Truncated IPv6 extension")
            next_protocol=data[transport]
            if p.protocol==44:
                fragment=int.from_bytes(data[transport+2:transport+4],"big")
                p.fragment=int(bool(fragment&0xfff9));length=8
                if fragment&0xfff8:return p
            else:length=(data[transport+1]+1)*8
            transport+=length;remaining-=length;p.protocol=next_protocol
    else:return p
    if p.protocol==6:
        if len(data)<transport+20:return p
        p.sp,p.dp=struct.unpack_from("!HH",data,transport)
        p.checksum=int.from_bytes(data[transport+16:transport+18],"big")
        p.tcp_payload=remaining-(data[transport+12]>>4)*4
        if p.tcp_payload<0:raise ValueError("Invalid TCP payload length")
        p.flags=data[transport+13]
    elif p.protocol==17:
        if len(data)<transport+8:return p
        p.udp_payload=max(0,int.from_bytes(data[transport+4:transport+6],"big")-8)
        # Canonical tcp fields are zero for UDP. UDP destination is used only as a state key.
        p.dp=0;p.sp=0
    return p

class CausalRing:
    """Fixed allocation; all observed packets pay storage, query statistics are selective."""
    def __init__(self,capacity=16384,windows=(.1,1.)):
        self.capacity=int(capacity);self.windows=tuple(windows)
        self.data=np.zeros((self.capacity,8),dtype=np.int64)
        self.reset()
    def reset(self):
        self.start=0;self.count=0;self.clock=None
        self.evicted_capacity=0;self.evicted_time=0;self.backward=0;self.queries=0
    def push(self,timestamp,p):
        if self.clock is not None and timestamp<self.clock:self.backward+=1
        self.clock=max(timestamp,self.clock if self.clock is not None else timestamp)
        cutoff=self.clock-int(max(self.windows)*1e9)
        while self.count and self.data[self.start,0]<cutoff:
            self.start=(self.start+1)%self.capacity;self.count-=1;self.evicted_time+=1
        if self.count==self.capacity:
            self.start=(self.start+1)%self.capacity;self.count-=1;self.evicted_capacity+=1
        at=(self.start+self.count)%self.capacity
        self.data[at]=[self.clock,p.src,p.dst,p.sp,p.dp,p.wire,p.flags,p.protocol]
        self.count+=1
    def query(self,dst):
        self.queries+=1
        if self.clock is None:raise ValueError("Cannot query unseen history")
        if self.start+self.count<=self.capacity:records=self.data[self.start:self.start+self.count]
        else:records=np.concatenate([self.data[self.start:],self.data[:(self.start+self.count)%self.capacity]])
        out=[]
        for window in self.windows:
            r=records[(records[:,0]>=self.clock-int(window*1e9))&(records[:,2]==dst)]
            n=len(r)
            if n:
                flags=r[:,6]
                out.extend([np.log1p(n),np.log1p(r[:,5].sum()),np.log1p(len(np.unique(r[:,1]))),
                            np.log1p(len(np.unique(r[:,4]))),r[:,5].mean(),r[:,5].std(),
                            np.mean((flags&2)!=0),np.mean((flags&16)!=0),np.mean((flags&4)!=0),
                            self.count/self.capacity])
            else:out.extend([0.]*10)
        return np.asarray(out,dtype=float)

def iter_pcap(archive,info,digest=None):
    with archive.open(info) as zipped,io.BufferedReader(zipped,buffer_size=1024*1024) as stream:
        def read(size):
            data=stream.read(size)
            if digest is not None:digest.update(data)
            return data
        h=read(24)
        magics={bytes.fromhex("d4c3b2a1"):("<",1000),bytes.fromhex("a1b2c3d4"):(">",1000),
                bytes.fromhex("4d3cb2a1"):("<",1),bytes.fromhex("a1b23c4d"):(">",1)}
        if len(h)!=24 or h[:4] not in magics:raise ValueError("Unsupported PCAP")
        endian,mult=magics[h[:4]]
        snap,link=struct.unpack(endian+"II",h[16:])
        if link!=1:raise ValueError("This decoder requires Ethernet PCAP")
        unpack=struct.Struct(endian+"IIII").unpack
        index=0
        while True:
            hdr=read(16)
            if not hdr:break
            if len(hdr)!=16:raise ValueError("Truncated PCAP record")
            sec,frac,cap,wire=unpack(hdr)
            if cap>snap or cap>16*1024*1024:raise ValueError("Invalid record length")
            data=read(cap)
            if len(data)!=cap:raise ValueError("Truncated packet")
            yield index,sec*1_000_000_000+frac*mult,sec,frac*mult,decode_packet(data,wire)
            index+=1

def role_for(index,total):
    return 0 if index<.6*total else 1 if index<.7*total else 2 if index<.8*total else 3

def extract(raw,zip_path,audit,cfg,folder,smoke=False):
    from datetime import datetime,timezone
    lookup=collections.defaultdict(list)
    for position,(_,row) in enumerate(raw.iterrows()):
        key,year,hour=row_identity(row)
        lookup[key].append((position,year,hour))
    if any(len(v)>1 for v in lookup.values()):raise ValueError("Canonical packet identity collision")
    n=len(raw);hits=np.zeros(n,dtype=int);base=np.full((n,len(HEADER)),np.nan)
    behavior=np.full((n,len(BEHAVIOR)),np.nan);capture=np.full(n,-1,dtype=int)
    record=np.full(n,-1,dtype=np.int64);role=np.full(n,-1,dtype=int)
    logical=np.full(n,-1,dtype=np.int64);valid=np.zeros(n,dtype=bool)
    source=[];offset_audit=[];events={};aliases=[]
    with zipfile.ZipFile(zip_path) as archive:
        infos={i.filename:i for i in archive.infolist()}
        for capid,old in enumerate(audit["pcap_audits"]):
            entry=old["entry"];info=infos[entry];total=old["packet_count"]
            if info.file_size!=old["bytes"]:raise ValueError("PCAP size mismatch")
            print(f"  extract {capid+1}/14 {Path(entry).name}",flush=True)
            digest=hashlib.sha256();ring=CausalRing(cfg["ring_capacity"],cfg["windows_seconds"])
            last_role=None;role_start=None;counts=collections.Counter();matched=0;seen=0;evictions=0;backwards=0
            started=time.perf_counter()
            for index,ts,sec,nano,p in iter_pcap(archive,info,digest):
                current=role_for(index,total)
                if current!=last_role:
                    evictions+=ring.evicted_capacity;backwards+=ring.backward
                    ring.reset();last_role=current;role_start=ts
                ring.push(ts,p);seen+=1
                key=p.identity(sec,nano)
                candidates=lookup.get(key)
                if candidates is None:continue
                for pos,year,hour in candidates:
                    actual_year=datetime.fromtimestamp(sec,timezone.utc).year
                    if year!=actual_year:continue
                    event=(ts,hashlib.sha256(p.raw).hexdigest())
                    h=np.asarray(p.header(),dtype=float);b=ring.query(p.dst)
                    if hits[pos]:
                        if events[pos]!=event or not np.array_equal(base[pos],h) or not np.allclose(behavior[pos],b,rtol=0,atol=1e-12):
                            e.write_json(folder/"mapping_conflict.json",dict(row_id=int(raw.index[pos]),
                              previous_capture=int(capture[pos]),new_capture=capid,previous_event=events[pos],new_event=event,
                              header_equal=bool(np.array_equal(base[pos],h)),behavior_equal=bool(np.allclose(behavior[pos],b,rtol=0,atol=1e-12))))
                            raise ValueError(f"Conflicting event or temporal context for row {raw.index[pos]}")
                        aliases.append(dict(row_id=int(raw.index[pos]),previous_capture=int(capture[pos]),new_capture=capid,
                                            previous_role=int(role[pos]),new_role=current,identical_event_and_context=True))
                        valid[pos]=bool(valid[pos] and ring.clock-role_start>=int(cfg["purge_seconds"]*1e9))
                        if current>role[pos]:
                            capture[pos]=capid;record[pos]=index;role[pos]=current;logical[pos]=ring.clock
                    else:
                        events[pos]=event
                        base[pos]=h;behavior[pos]=b
                        capture[pos]=capid;record[pos]=index;role[pos]=current;logical[pos]=ring.clock
                        valid[pos]=ring.clock-role_start>=int(cfg["purge_seconds"]*1e9)
                    hits[pos]+=1
                    counts[(hour-(sec//3600)%24)%24]+=1;matched+=1
            evictions+=ring.evicted_capacity;backwards+=ring.backward
            if seen!=total or digest.hexdigest()!=old["sha256"]:raise ValueError("Full PCAP checksum/record count gate failed")
            if len(counts)>1:raise ValueError("Nonconstant hour offset in capture mapping")
            source.append(dict(capture_id=capid,entry=entry,packet_count=seen,matched_development_rows=matched,
                               inferred_hour_offsets=dict(counts),capacity_evictions=evictions,
                               backward_records=backwards,seconds=time.perf_counter()-started,sha256=digest.hexdigest()))
            offset_audit.append(dict(capture_id=capid,offset_hours=dict(counts)))
    if not np.all(hits>=1):
        missing=raw.index.to_numpy()[hits<1]
        e.write_json(folder/"mapping_failure.json",dict(missing_count=len(missing),missing_development_ids=missing))
        raise ValueError(f"Packet identity gate: {len(missing)} rows unmatched")
    result=dict(row_ids=raw.index.to_numpy(),base=base,behavior=behavior,capture=capture,record=record,
                role=role,logical_ns=logical,valid_after_purge=valid)
    np.savez_compressed(folder/"packet_features.npz",**result)
    e.write_json(folder/"mapping_audit.json",dict(rows=n,unique_physical_event_and_context=True,
                 sources=source,metadata_not_predictors=True,all_source_hashes_verified=True,
                 duplicate_capture_aliases=aliases,duplicate_alias_count=len(aliases),
                 discarded_first_second_rows=int((~valid).sum()),buffer_allocated_bytes=cfg["ring_capacity"]*8*8))
    return result

def view_data(raw,data,view):
    if view.startswith("legacy"):
        mode="noport25" if "25" in view else "dstport29"
        x=DomainFeatures(mode).transform(raw)
        numeric,categorical=columns(mode)
    else:
        x=pd.DataFrame(data["base"],index=raw.index,columns=HEADER)
        numeric=list(HEADER);categorical=[]
    if view.endswith("_behavior"):
        b=pd.DataFrame(data["behavior"],index=raw.index,columns=BEHAVIOR)
        x=pd.concat([x,b],axis=1);numeric=list(numeric)+BEHAVIOR
    return x,list(numeric),list(categorical)

def make_model(name,numeric,categorical,cfg,seed):
    steps=[("imputer",SimpleImputer(strategy="median",keep_empty_features=True))]
    if name=="lr":steps.append(("scaler",StandardScaler()))
    branches=[("numeric",Pipeline(steps),numeric)]
    if categorical:
        branches.append(("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=True),categorical))
    prep=ColumnTransformer(branches,remainder="drop",sparse_threshold=1.)
    if name=="lr":est=LogisticRegression(**cfg["lr"],random_state=seed)
    elif name=="rf":est=RandomForestClassifier(**cfg["rf"],random_state=seed)
    else:raise ValueError("Unreviewed model family")
    return Pipeline([("preprocessor",prep),("model",est)])

def gate_features(base,probabilities):
    p=np.asarray(probabilities);ordered=np.sort(p,axis=1)
    return np.column_stack([base,p,ordered[:,-1]-ordered[:,-2],-np.sum(p*np.log(np.maximum(p,1e-15)),axis=1)])

def fit_utility(base,p0,p1,y,cfg,seed,risk=True):
    gain=(p1.argmax(1)==y).astype(float)-(p0.argmax(1)==y).astype(float)
    if risk:gain*=np.where(np.asarray(y)==1,cfg["ddos_miss_weight"],1.)
    model=DecisionTreeRegressor(**cfg["utility_tree"],random_state=seed)
    model.fit(gate_features(base,p0),gain)
    return model,gain

def choose_threshold(y,p0,p1,scores,thresholds,tolerance):
    rich=metrics(y,p1.argmax(1));simple=metrics(y,p0.argmax(1));rows=[]
    for threshold in thresholds:
        mask=scores>threshold
        p=np.where(mask[:,None],p1,p0);m=metrics(y,p.argmax(1))
        good=(m["macro_f1"]>=rich["macro_f1"]-tolerance and m["ddos_f2"]>=rich["ddos_f2"]-tolerance
              and m["recall_DDoS_TCP"]>=simple["recall_DDoS_TCP"])
        rows.append(dict(threshold=float(threshold),escalation=float(mask.mean()),constraint_pass=bool(good),
                         selection_score=.5*(m["macro_f1"]+m["ddos_f2"]),**m))
    passed=[r for r in rows if r["constraint_pass"]]
    if passed:best=min(passed,key=lambda r:(r["escalation"],-r["selection_score"],-r["threshold"]))
    else:best=max(rows,key=lambda r:(r["selection_score"],-r["escalation"],r["threshold"]))
    return best,rows

def guard_roles(ids,parts,holdout):
    unions=set()
    for part in parts:
        values=set(np.asarray(ids)[part].tolist())
        if len(values)!=len(part) or unions&values or values&set(holdout):raise ValueError("Role overlap or holdout leakage")
        unions|=values
    return True

