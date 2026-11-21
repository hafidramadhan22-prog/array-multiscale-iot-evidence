"""Verified replay of existing PCAP sources; add no dataset, labels or captures."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import ta_hafid_engine as e
from prepare_wb_behavior import verify_catalog
from sbfe import iter_pcap,role_for
from multiscale_context import MultiscaleRing,CONTEXT_NAMES,DIRECTION_COLUMNS

def main(output):
    output=output.resolve()
    if output.parent!=(ROOT/'runs').resolve() or output.exists():raise ValueError('New run directory required')
    source_folder=ROOT/'runs/WB_BEHAVIOR_DATA_20261019_01';verify_catalog(source_folder)
    cfg=json.loads((source_folder/'configuration_freeze.json').read_text());archive_path=Path(cfg['zip_path'])
    assert e.sha256(archive_path)==cfg['zip_sha256'] and e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv')==cfg['canonical_sha256']
    with np.load(source_folder/'packet_features.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    mapping=json.loads((source_folder/'mapping_audit.json').read_text());n=len(data['row_ids'])
    context=np.full((n,60),np.nan);direction=np.full((n,36),np.nan);hits=np.zeros(n,dtype=int);warm=np.zeros(n,dtype=bool)
    output.mkdir();(output/'source').mkdir()
    e.write_json(output/'configuration_freeze.json',{**cfg,'extension':'MULTISCALE_DIRECTIONAL','new_windows':[.1,1.,10.],
        'source_catalog_sha256':e.sha256(source_folder/'catalog.json'),'purge_unchanged_seconds':1,'capacity_records':16384})
    for rel in ['scripts/prepare_multiscale_context.py','src/multiscale_context.py','tests/test_multiscale_context.py','docs/MULTISCALE_REPAIR_PROTOCOL_20261020.md']:
        shutil.copy2(ROOT/rel,output/'source'/Path(rel).name)
    audits=[]
    with zipfile.ZipFile(archive_path) as archive:
        infos={z.filename:z for z in archive.infolist()}
        for old in mapping['sources']:
            cap=old['capture_id'];positions=np.flatnonzero(data['capture']==cap)
            wanted={int(data['record'][p]):int(p) for p in positions};assert len(wanted)==len(positions)
            ring=MultiscaleRing();last_role=None;start=None;digest=hashlib.sha256();seen=0;capacity=time_evictions=backwards=0
            begin=time.perf_counter()
            for index,timestamp,_,_,packet in iter_pcap(archive,infos[old['entry']],digest):
                role=role_for(index,old['packets'])
                if role!=last_role:
                    capacity+=ring.evicted_capacity;time_evictions+=ring.evicted_time;backwards+=ring.backward
                    ring.reset();last_role=role;start=timestamp
                pos=wanted.get(index)
                if pos is not None:
                    assert data['role'][pos]==role and np.array_equal(packet.header(),data['base'][pos])
                    source,multi,direct=ring.query_multiscale(timestamp,packet.src,packet.dst)
                    np.testing.assert_array_equal(source[:20],data['prior_source'][pos])
                    context[pos]=multi;direction[pos]=direct;hits[pos]+=1
                    clock=max(timestamp,ring.clock if ring.clock is not None else timestamp);warm[pos]=clock-start<int(10e9)
                ring.push(timestamp,packet);seen+=1
                if pos is not None:assert ring.clock==data['logical_ns'][pos]
            assert seen==old['packets'] and digest.hexdigest()==old['sha256']
            capacity+=ring.evicted_capacity;time_evictions+=ring.evicted_time;backwards+=ring.backward
            row=dict(capture_id=cap,entry=old['entry'],packets=seen,targets=len(positions),sha256=digest.hexdigest(),
                     capacity_evictions=capacity,time_evictions=time_evictions,backwards=backwards,elapsed_seconds=time.perf_counter()-begin)
            audits.append(row);print(json.dumps(row),flush=True)
    assert np.all(hits==1) and np.isfinite(context).all() and np.isfinite(direction).all()
    np.savez_compressed(output/'packet_features.npz',**data,prior_multiscale=context,prior_directional=direction,partial_10s_history=warm)
    for name in ['target_indices.npz','role_support.json']:shutil.copy2(source_folder/name,output/name)
    e.write_json(output/'feature_schema.json',dict(prior_multiscale=CONTEXT_NAMES,prior_directional=DIRECTION_COLUMNS))
    e.write_json(output/'mapping_audit.json',dict(sources=audits,rows=n,old_source20_reproduced_exactly=True,
        target_rows_with_partial_10s_history=int(warm.sum()),eligible_partial_10s=int((warm&data['valid_after_purge']).sum()),
        packet_ring_allocated_bytes=16384*8*8,prior_only=True,old_roles_purge_preserved=True))
    e.write_json(output/'environment.json',e.environment())
    e.write_json(output/'catalog.json',[dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='catalog.json'])
    e.write_json(output/'complete.json',dict(status='WB_BEHAVIOR_DATA_COMPLETE',extension='MULTISCALE_DIRECTIONAL',rows=n,
        valid_targets=int(data['valid_after_purge'].sum()),catalog_sha256=e.sha256(output/'catalog.json'),new_training=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
