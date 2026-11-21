"""Predeclared benign-only cross-capture stress test; never tune on these outputs."""
from pathlib import Path
import sys,json,hashlib,zlib,shutil,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_factorial_20261103 import OLD,checked,methods
from array_revision import frame
from e4 import RAW_COLUMNS
from sbfe import iter_pcap,row_identity,META,TIME_RE
from wb_profile import LocalSource
from prepare_array_history_20261103 import AuditedRing
from wb_research import cube

def main():
    out=ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04';assert not out.exists()
    base=Path('C:/TA HAFID');audit=json.loads((ROOT/'reports/DATA_MENTAH_AUDIT_ZIP_20261019/zip_audit.json').read_text());catalog={r['entry']:r for r in audit['entries']}
    pcaps=pd.read_csv(ROOT/'reports/DATA_MENTAH_AUDIT_ZIP_20261019/pcap_times.csv');sources=pcaps[(pcaps.scope=='Normal')&~pcaps.entry.str.contains('Temperature_and_Humidity')]
    assert len(sources)==9
    model=checked(OLD,'stages/temporal_seed42/legacy25_multiscale60__balanced/model.joblib')
    out.mkdir();(out/'source').mkdir()
    for rel in ['scripts/audit_array_benign_20261103.py','docs/ARRAY_BENIGN_CAPTURE_ADDENDUM_20261103.md','src/array_revision.py','scripts/prepare_array_history_20261103.py']:shutil.copy2(ROOT/rel,out/'source'/Path(rel).name)
    e.write_json(out/'configuration_freeze.json',dict(model_sha256=e.sha256(model),sources=sources.entry.tolist(),prefix_packets=100000,candidates=list(range(999,100000,1000)),warmup_seconds=10,capacity=16384,fits=0,selection=False,evaluation='benign-only distinct-capture same-corpus stress; not independent five-class final test'))
    bundle=joblib.load(model);scores=[];records=[];predrows=[]
    with threadpool_limits(limits=1):
        for row in sources.itertuples():
            pcap=base/row.entry;csv=pcap.with_suffix('.csv');assert e.sha256(pcap)==row.sha256
            rel=csv.relative_to(base).as_posix();rec=catalog[rel];crc=0;sha=hashlib.sha256()
            with csv.open('rb') as stream:
                for block in iter(lambda:stream.read(2**20),b''):crc=zlib.crc32(block,crc);sha.update(block)
            assert csv.stat().st_size==rec['bytes'] and f'{crc:08x}'==rec['crc32']
            raw=pd.read_csv(csv,nrows=100000,low_memory=False,usecols=list(dict.fromkeys(RAW_COLUMNS+META+['Attack_label','Attack_type'])))
            assert len(raw)==100000 and (raw.Attack_type=='Normal').all() and (raw.Attack_label==0).all()
            candidates=set(range(999,100000,1000));ids=[];contexts=[];elapsed=[];excluded=[];ring=AuditedRing(16384);clock=None;start=None;hits=0;keys={}
            for index,t,sec,nano,p in itertools.islice(iter_pcap(LocalSource(),pcap),100000):
                if start is None:start=t
                clock=max(t,clock if clock is not None else t)
                if index in candidates:
                    key=p.identity(sec,nano);assert key not in keys;keys[key]=index
                    hits+=1
                    if clock-start>=10_000_000_000:
                        ids.append(index);contexts.append(ring.context(clock,p.src,p.dst));elapsed.append((clock-start)/1e9)
                    else:excluded.append(index)
                ring.push(clock,p)
            assert hits==100 and ids
            times={k[:2] for k in keys};mapped={};duplicates=0;invalid_clocks=0
            for i,text in enumerate(raw['frame.time']):
                m=TIME_RE.search(str(text))
                if m is None:invalid_clocks+=1;continue
                year,hour,minute,second,fraction=m.groups()
                if (int(minute)*60+int(second),int(fraction.ljust(9,'0'))) not in times:continue
                key,_,_=row_identity(raw.iloc[i])
                if key not in keys:continue
                packet_index=keys[key]
                if packet_index in mapped:
                    pd.testing.assert_series_equal(raw.iloc[i][RAW_COLUMNS],raw.iloc[mapped[packet_index]][RAW_COLUMNS],check_names=False);duplicates+=1
                else:mapped[packet_index]=i
            eligible=list(ids);missing=sorted(set(ids)-set(mapped));keep=[j for j,i in enumerate(ids) if i in mapped]
            ids=[ids[j] for j in keep];contexts=[contexts[j] for j in keep];elapsed=[elapsed[j] for j in keep]
            aligned=raw.iloc[[mapped[i] for i in ids]].copy();aligned.index=ids
            name=pcap.stem
            if ids:
                X,_,_=frame(aligned,np.asarray(contexts),'c60_o10');ps=cube(bundle['models'],X);X.to_csv(out/(name+'_features.csv'),index_label='record_index')
            else:ps=np.empty((0,5,5),dtype=float)
            np.savez_compressed(out/(name+'_predictions.npz'),record_index=ids,p_experts=ps,weights=bundle['weights'],elapsed_seconds=elapsed)
            for method,prob in methods(ps,bundle['weights']).items():
                pred=prob.argmax(1);fp=int((pred!=0).sum());scores.append(dict(source=name,method=method,n=len(ids),eligible=len(eligible),unscorable=len(missing),false_alarms=fp,normal_recall=float((pred==0).mean()) if len(ids) else np.nan,false_positive_rate=float((pred!=0).mean()) if len(ids) else np.nan,fpr_lower=fp/len(eligible),fpr_upper=(fp+len(missing))/len(eligible),**{f'predicted_{cl}':int((pred==cl).sum()) for cl in range(5)}))
                for i,lab in enumerate(pred):predrows.append(dict(source=name,record_index=ids[i],elapsed_seconds=elapsed[i],method=method,predicted_class=int(lab),true_class=0))
            records.append(dict(source=name,pcap=str(pcap),pcap_sha256=row.sha256,csv=str(csv),csv_sha256=sha.hexdigest(),csv_crc32=f'{crc:08x}',csv_prefix_rows=100000,pcap_prefix_records=100000,candidate_identities_match=len(mapped),invalid_csv_clock_rows=invalid_clocks,unmatched_candidates=sorted(candidates-set(mapped)),unscorable_eligible=missing,identity_preserving_csv_duplicates=duplicates,packet_to_csv_row=mapped,excluded_before_10s=excluded,eligible=len(eligible),scored=len(ids),capacity_evictions=ring.evicted_capacity))
            pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False);e.write_json(out/'source_audit.json',records);print('BENIGN_CHECKED',name,len(ids),flush=True)
    pd.DataFrame(predrows).to_csv(out/'predictions.csv',index=False)
    e.write_json(out/'catalog.json',[dict(path=p.relative_to(out).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
    e.write_json(out/'complete.json',dict(status='COMPLETED_PARTIAL_COVERAGE_AUDIT',complete_mapping=False,new_fits=0,sources=9,eligible=sum(r['eligible'] for r in records),targets=sum(r['scored'] for r in records),five_class_evaluation=False,catalog_sha256=e.sha256(out/'catalog.json')))
if __name__=='__main__':main()
