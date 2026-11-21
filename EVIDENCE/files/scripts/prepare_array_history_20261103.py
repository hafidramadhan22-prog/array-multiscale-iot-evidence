"""Reconstruct bounded histories with disjoint packet ownership and fixed buffer controls."""
from pathlib import Path
import argparse,hashlib,json,sys,time,zipfile,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from run_wb_research import load_data,parts_for
from wb_behavior import PriorEndpointRing
from sbfe import iter_pcap,role_for

class AuditedRing(PriorEndpointRing):
    def __init__(self,capacity):super().__init__(capacity,(.1,1.,10.))
    def reset(self):super().reset();self.last_capacity_timestamp=-10**30
    def push(self,t,p):
        clock=max(t,self.clock if self.clock is not None else t)
        if self.count==self.capacity and self.data[self.start,0]>=clock-10_000_000_000:
            self.last_capacity_timestamp=int(self.data[self.start,0])
        super().push(t,p)
    def context(self,t,src,dst):
        r,clock=self._records_before(t)
        h=[self._host_vector(r,clock,src),self._host_vector(r,clock,dst)]
        h.sort(key=self._activity_key,reverse=True)
        return np.concatenate(h)

def main(output):
    assert not output.exists()
    raw,y,holdout,data=load_data(dict(data_folder='runs/MULTISCALE_DATA_20261020_01'))
    old=ROOT/'runs/MULTISCALE_DATA_20261020_01'
    config=json.loads((old/'configuration_freeze.json').read_text());archive=Path(config['zip_path'])
    if not archive.exists():archive=ROOT/'arsip/AKUN_20261022/external_data/Edge-IIoTset dataset.zip'
    assert e.sha256(archive)==config['zip_sha256']
    n=len(raw);roles={};fractions={};arrays={};audit=[]
    for fold in [1,2]:
        parts,_=parts_for('signature_fold'+str(fold),raw,y,holdout,data,False)
        r=np.full(n,-1,dtype=int)
        for k,pos in enumerate(parts):r[pos]=k
        roles[fold]=r;fractions[fold]=np.array([len(p) for p in parts])/sum(map(len,parts))
        for mode in ['exclusive','shared_density']:arrays[f'fold{fold}_{mode}']=np.full((n,60),np.nan)
    for cap in [4096,65536]:arrays[f'buffer{cap}']=np.full((n,60),np.nan)
    truncated={str(cap):np.zeros(n,dtype=bool) for cap in [4096,16384,65536]}
    warm=np.zeros(n,dtype=bool);hits=np.zeros(n,dtype=int)
    output.mkdir();(output/'source').mkdir()
    for rel in ['scripts/prepare_array_history_20261103.py','src/wb_behavior.py','src/sbfe.py','docs/ARRAY_MAJOR_REVISION_PROTOCOL_20261103.md']:
        shutil.copy2(ROOT/rel,output/'source'/Path(rel).name)
    e.write_json(output/'configuration_freeze.json',dict(source=old.relative_to(ROOT).as_posix(),source_catalog_sha256=e.sha256(old/'catalog.json'),
        zip_path=str(archive),zip_sha256=config['zip_sha256'],fractions=fractions,seed=20261103,capacities=[4096,16384,65536],
        history_modes=['exclusive','shared_density'],target_roles='fixed signature A/G/C/V; mapped target packets private in both modes',
        clock='full-stream monotonic clock',reset='unchanged temporal role boundaries',historical_holdout_targets_used=False))
    with zipfile.ZipFile(archive) as z:
        for source in json.loads((old/'mapping_audit.json').read_text())['sources']:
            cap=source['capture_id'];total=source['packets'];positions=np.flatnonzero(data['capture']==cap)
            wanted={int(data['record'][p]):int(p) for p in positions};masks={}
            for fold in [1,2]:
                u=np.random.default_rng(20261103+fold*100+cap).random(total);f=fractions[fold]
                exclusive=(1<<np.searchsorted(np.cumsum(f),u,side='right')).astype(np.uint8)
                shared=np.sum((u[:,None]<f)*np.array([1,2,4,8]),axis=1).astype(np.uint8)
                assigned=positions[roles[fold][positions]>=0];rec=data['record'][assigned]
                exclusive[rec]=1<<roles[fold][assigned];shared[rec]=exclusive[rec]
                assert np.isin(exclusive,[1,2,4,8]).all()
                for mode,mask in [('exclusive',exclusive),('shared_density',shared)]:
                    masks[(fold,mode)]=mask
                    audit.append(dict(capture=cap,fold=fold,mode=mode,packets=total,packets_shared_by_roles=int(np.sum((mask&(mask-1))!=0)),
                        unallocated_packets=int(np.sum(mask==0)),**{f'allocated_role_{k}':int(np.sum((mask&(1<<k))!=0)) for k in range(4)}))
            buffers={k:AuditedRing(k) for k in [4096,16384,65536]}
            rings={(fold,mode,k):AuditedRing(16384) for fold in [1,2] for mode in ['exclusive','shared_density'] for k in range(4)}
            last=None;clock=None;start=None;digest=hashlib.sha256();seen=0;evictions={str(k):0 for k in buffers};begin=time.perf_counter()
            for record,t,_,_,packet in iter_pcap(z,z.getinfo(source['entry']),digest):
                temporal=role_for(record,total)
                if temporal!=last:
                    for k,r in buffers.items():evictions[str(k)]+=r.evicted_capacity;r.reset()
                    for r in rings.values():r.reset()
                    last=temporal;clock=t;start=t
                clock=max(t,clock);pos=wanted.get(record)
                if pos is not None:
                    assert data['role'][pos]==temporal and np.array_equal(packet.header(),data['base'][pos])
                    for k,r in buffers.items():
                        value=r.context(clock,packet.src,packet.dst)
                        if k==16384:np.testing.assert_array_equal(value,data['prior_multiscale'][pos])
                        else:arrays[f'buffer{k}'][pos]=value
                        truncated[str(k)][pos]=r.last_capacity_timestamp>=clock-10_000_000_000
                    warm[pos]=clock-start>=10_000_000_000
                    for fold in [1,2]:
                        k=roles[fold][pos]
                        if k>=0:
                            for mode in ['exclusive','shared_density']:
                                arrays[f'fold{fold}_{mode}'][pos]=rings[fold,mode,k].context(clock,packet.src,packet.dst)
                    hits[pos]+=1
                for r in buffers.values():r.push(clock,packet)
                for (fold,mode),mask in masks.items():
                    bits=int(mask[record])
                    for k in range(4):
                        if bits&(1<<k):rings[fold,mode,k].push(clock,packet)
                seen+=1
            assert digest.hexdigest()==source['sha256'] and seen==total
            for k,r in buffers.items():evictions[str(k)]+=r.evicted_capacity
            audit.append(dict(capture=cap,kind='buffer',capacity_evictions=evictions,elapsed_seconds=time.perf_counter()-begin,sha256=digest.hexdigest()))
            print('HISTORY_RECONSTRUCTED',cap,total,'seconds',round(time.perf_counter()-begin,1),flush=True)
    assert (hits==1).all()
    for name,arr in arrays.items():
        valid=data['valid_after_purge'] if name.startswith('fold') else np.ones(n,dtype=bool)
        assert np.isfinite(arr[valid]).all(),name
    np.savez_compressed(output/'histories.npz',row_ids=raw.index.to_numpy(),warm10=warm,
        **arrays,**{'truncated_'+k:v for k,v in truncated.items()},**{f'fold{k}_roles':r for k,r in roles.items()})
    e.write_json(output/'audit.json',dict(records=audit,original_context60_reproduced_exactly=True,prior_only=True,
        no_shared_packet_in_exclusive_roles=True,valid1=int(data['valid_after_purge'].sum()),valid10=int((data['valid_after_purge']&warm).sum()),
        independent_capture_validation=False,limitations='Thinning changes density; marginal shared control and realized allocation counts retained. Source/class confounding remains.'))
    # Context summaries are diagnostics, never used for feature/model selection.
    f=pd.DataFrame(data['prior_multiscale']);f['capture']=data['capture'];f['role']=data['role']
    f.groupby(['capture','role']).agg(['mean','std']).to_csv(output/'context_by_capture_and_role.csv')
    e.write_json(output/'catalog.json',[dict(path=p.relative_to(output).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()])
    e.write_json(output/'complete.json',dict(status='PASS_HISTORY_DATA',catalog_sha256=e.sha256(output/'catalog.json'),source_data_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
