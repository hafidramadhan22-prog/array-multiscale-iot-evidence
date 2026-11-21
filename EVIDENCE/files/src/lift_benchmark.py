"""Same cold RSS/warm latency definitions as the historical engine, new isolated worker."""
from pathlib import Path
import json
import os
import sys
import threading
import queue
import subprocess
import time
import numpy as np
import psutil
import ta_hafid_engine as e


def launch(model,batch,directory,name,mode,cfg):
    directory=Path(directory)
    conf=dict(model=str(Path(model).resolve()),batch=str(Path(batch).resolve()),mode=mode,
              latency_repeats=cfg['latency_repeats'],model_sha256=e.sha256(model),batch_sha256=e.sha256(batch))
    result=directory/f'{name}.json';inp=directory/f'{name}_input.json'
    if result.exists():
        if json.loads(inp.read_text())!=conf:raise ValueError('Profiling resume input mismatch')
        return json.loads(result.read_text())
    e.write_json(inp,conf)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    worker=Path(__file__).resolve().parents[1]/'scripts/lift_profile_worker.py'
    with (directory/f'{name}.log').open('w',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-B',str(worker),'--worker',str(inp)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env=env,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),bufsize=1)
        messages=queue.Queue()
        def read_lines():
            for line in p.stdout:
                try:value=json.loads(line)
                except json.JSONDecodeError:continue
                if 'event' in value:messages.put(value)
            messages.put(None)
        threading.Thread(target=read_lines,daemon=True).start()
        def message():
            try:value=messages.get(timeout=120)
            except queue.Empty:raise RuntimeError('Profiler timeout; inspect worker log')
            if value is None:raise RuntimeError(f'Profiler exited; inspect {name}.log')
            return value
        try:
            if mode=='latency':out=message()
            else:
                out={}
                for phase in ['cold','warm']:
                    ready=message()
                    if ready['event']!='ready' or ready['phase']!=phase:raise ValueError('Invalid profiler handshake')
                    out['load']=ready['load'];samples=[(0.,ready['baseline'])];stop=threading.Event();start=time.perf_counter()
                    def monitor():
                        while not stop.wait(cfg['memory_interval_seconds']):
                            try:rss=psutil.Process(p.pid).memory_info().rss;samples.append((time.perf_counter()-start,rss))
                            except psutil.NoSuchProcess:break
                    thread=threading.Thread(target=monitor,daemon=True);thread.start()
                    p.stdin.write('START\n');p.stdin.flush()
                    try:done=message()
                    finally:stop.set();thread.join()
                    if done['event']!='done' or done['phase']!=phase:raise ValueError('Invalid end handshake')
                    samples.append((time.perf_counter()-start,done['end']))
                    peak=max(rss for _,rss in samples);gaps=np.diff([t for t,_ in samples])
                    out[phase]=dict(baseline_rss_mib=ready['baseline']/2**20,sampled_peak_rss_mib=peak/2**20,
                        peak_minus_baseline_mib=(peak-ready['baseline'])/2**20,ending_rss_mib=done['end']/2**20,
                        samples=len(samples),calls=ready['calls'],requested_interval_ms=cfg['memory_interval_seconds']*1000,
                        max_actual_gap_ms=float(gaps.max()*1000),trace_seconds=[t for t,_ in samples],trace_rss_mib=[v/2**20 for _,v in samples])
                    p.stdin.write('ACK\n');p.stdin.flush()
            if p.wait(timeout=30)!=0:raise RuntimeError('Profiler nonzero exit')
        finally:
            if p.poll() is None:p.kill();p.wait()
    e.write_json(result,out)
    return out
