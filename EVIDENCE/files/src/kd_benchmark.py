"""Cold inference RSS in a fresh, verified worker; latency measured separately."""
from pathlib import Path
import json
import os
import queue
import subprocess
import sys
import threading
import time
import numpy as np
import psutil
import ta_hafid_engine as e


def profile_memory(model,batch,cfg,directory,name):
    model,batch,directory=Path(model).resolve(),Path(batch).resolve(),Path(directory)
    conf=dict(model=str(model),batch=str(batch),model_sha256=e.sha256(model),batch_sha256=e.sha256(batch))
    inp=directory/f'memory_{name}_input.json';e.write_json(inp,conf)
    worker=Path(__file__).resolve().parents[1]/'scripts/kd_profile_worker.py'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (directory/f'memory_{name}.log').open('w',encoding='utf8') as error:
        process=subprocess.Popen([sys.executable,'-B',str(worker),'--input',str(inp)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=error,text=True,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),bufsize=1)
        messages=queue.Queue()
        def read():
            for line in process.stdout:
                try:messages.put(json.loads(line))
                except json.JSONDecodeError:continue
            messages.put(None)
        threading.Thread(target=read,daemon=True).start()
        def message():
            item=messages.get(timeout=90)
            if item is None:raise RuntimeError('Memory worker exited; inspect log')
            return item
        try:
            ready=message()
            if ready['event']!='ready':raise ValueError('Invalid memory worker handshake')
            samples=[(0.,ready['baseline'])];stop=threading.Event();start=time.perf_counter()
            def monitor():
                while not stop.wait(cfg['memory_interval_seconds']):
                    try:samples.append((time.perf_counter()-start,psutil.Process(process.pid).memory_info().rss))
                    except psutil.NoSuchProcess:break
            sampler=threading.Thread(target=monitor,daemon=True);sampler.start()
            process.stdin.write('START\n');process.stdin.flush()
            try:done=message()
            finally:stop.set();sampler.join()
            if done['event']!='done':raise ValueError('Missing predict completion')
            samples.append((time.perf_counter()-start,done['ending']))
            process.stdin.write('ACK\n');process.stdin.flush()
            if process.wait(timeout=20)!=0:raise RuntimeError('Memory worker failed')
        finally:
            if process.poll() is None:process.kill();process.wait()
    peak=max(v for _,v in samples);gaps=np.diff([t for t,_ in samples])
    out=dict(baseline_rss_mib=ready['baseline']/2**20,peak_rss_mib=peak/2**20,
             delta_peak_inference_mib=(peak-ready['baseline'])/2**20,ending_rss_mib=done['ending']/2**20,
             samples=len(samples),batch_rows=ready['rows'],requested_interval_ms=cfg['memory_interval_seconds']*1000,
             max_actual_gap_ms=float(gaps.max()*1000),trace_seconds=[t for t,_ in samples],trace_rss_mib=[v/2**20 for _,v in samples])
    e.write_json(directory/f'memory_{name}.json',out)
    return {k:v for k,v in out.items() if not k.startswith('trace_')}
