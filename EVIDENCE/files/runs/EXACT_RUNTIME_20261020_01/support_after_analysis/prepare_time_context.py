"""Replay the same verified sources, preserving all old predictor values exactly."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import ta_hafid_engine as e
from prepare_wb_behavior import verify_catalog
from run_context_repair import check_record,read_npz
from sbfe import iter_pcap,role_for
from time_context import HostTimeRing,TIME_COLUMNS

def main(output):
    output=output.resolve()
    if output.parent!=(ROOT/'runs').resolve() or output.exists():raise ValueError('New run directory required')
    folder=ROOT/'runs/MULTISCALE_DATA_20261020_01';verify_catalog(folder)
    cfg=json.loads((folder/'configuration_freeze.json').read_text());archive_path=Path(cfg['zip_path'])
    if e.sha256(archive_path)!=cfg['zip_sha256'] or e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv')!=cfg['canonical_sha256']:raise ValueError('Raw input hash gate')
    data=read_npz(folder/'packet_features.npz');n=len(data['row_ids']);extra=np.full((n,24),np.nan);hits=np.zeros(n,dtype=int)
    xi=ROOT/'runs/XAI_MULTISCALE_FULL_20261020_01';rec=next(r for r in json.loads((xi/'artifacts_sha256.json').read_text()) if r['path']=='indices.npz');check_record(xi,rec)
    ids=read_npz(xi/'indices.npz')['row_ids'];lookup={int(row):i for i,row in enumerate(ids)}
    jitter_context=np.full((len(ids),60),np.nan);jitter_time=np.full((len(ids),24),np.nan)
    output.mkdir();(output/'source').mkdir()
    e.write_json(output/'configuration_freeze.json',{**cfg,'extension':'RETAINED_HOST_TIME',
        'source_catalog_sha256':e.sha256(folder/'catalog.json'),'windows':[.1,1.,10.],
        'jitters_for_frozen_old_100V_only_ms':5,'source_indices_sha256':e.sha256(xi/'indices.npz')})
    for rel in ['src/time_context.py','scripts/prepare_time_context.py','tests/test_time_context.py',
                'docs/TIME_RUNTIME_PROTOCOL_20261020.md','docs/TIME_RUNTIME_LITERATURE_20261020.md']:
        shutil.copy2(ROOT/rel,output/'source'/Path(rel).name)
    audits=[];mapping=json.loads((folder/'mapping_audit.json').read_text())
    with zipfile.ZipFile(archive_path) as archive:
        for old in mapping['sources']:
            cap=old['capture_id'];positions=np.flatnonzero(data['capture']==cap)
            wanted={int(data['record'][p]):int(p) for p in positions};assert len(wanted)==len(positions)
            ring=HostTimeRing();last=None;digest=hashlib.sha256();seen=0;capacity=evicted=backward=0;started=time.perf_counter()
            for record,stamp,_,_,packet in iter_pcap(archive,archive.getinfo(old['entry']),digest):
                role=role_for(record,old['packets'])
                if role!=last:
                    capacity+=ring.evicted_capacity;evicted+=ring.evicted_time;backward+=ring.backward;ring.reset();last=role
                pos=wanted.get(record)
                if pos is not None:
                    assert role==data['role'][pos] and np.array_equal(packet.header(),data['base'][pos])
                    host,direction,timing=ring.query_time(stamp,packet.src,packet.dst)
                    np.testing.assert_array_equal(host,data['prior_multiscale'][pos]);np.testing.assert_array_equal(direction,data['prior_directional'][pos])
                    extra[pos]=timing;hits[pos]+=1
                    if int(data['row_ids'][pos]) in lookup:
                        j=lookup[int(data['row_ids'][pos])];jitter_context[j],_,jitter_time[j]=ring.query_time(stamp+5_000_000,packet.src,packet.dst)
                ring.push(stamp,packet);seen+=1
                if pos is not None:assert ring.clock==data['logical_ns'][pos]
            assert seen==old['packets'] and digest.hexdigest()==old['sha256']
            capacity+=ring.evicted_capacity;evicted+=ring.evicted_time;backward+=ring.backward
            row=dict(capture_id=cap,entry=old['entry'],packets=seen,targets=len(positions),sha256=digest.hexdigest(),
                capacity_evictions=capacity,time_evictions=evicted,backwards=backward,elapsed_seconds=time.perf_counter()-started)
            audits.append(row);print(json.dumps(row),flush=True)
    assert (hits==1).all() and np.isfinite(extra).all() and np.isfinite(jitter_time).all() and np.isfinite(jitter_context).all()
    np.savez_compressed(output/'packet_features.npz',**{**data,'prior_hosttime':extra})
    np.savez_compressed(output/'jitter.npz',row_ids=ids,prior_multiscale=jitter_context,prior_hosttime=jitter_time)
    for name in ['target_indices.npz','role_support.json']:shutil.copy2(folder/name,output/name)
    e.write_json(output/'feature_schema.json',dict(prior_hosttime=TIME_COLUMNS))
    e.write_json(output/'mapping_audit.json',dict(sources=audits,rows=n,old60_and_direction36_exact=True,
        query_before_push=True,new_absolute_time_or_identity_predictors=False,ring_bytes=16384*8*8))
    e.write_json(output/'environment.json',e.environment())
    e.write_json(output/'catalog.json',[dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='catalog.json'])
    e.write_json(output/'complete.json',dict(status='WB_BEHAVIOR_DATA_COMPLETE',extension='RETAINED_HOST_TIME',rows=n,
        valid_targets=int(data['valid_after_purge'].sum()),catalog_sha256=e.sha256(output/'catalog.json'),new_training=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
