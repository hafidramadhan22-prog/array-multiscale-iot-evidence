"""Execute every cell in a fresh kernel; retain actual display outputs and audit."""
from pathlib import Path
import json, hashlib, os, sys, time, argparse, queue
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'notebooks/independent_20261109/UJI_INDEPENDEN_HASIL_DAN_PREDIKSI_20261109.ipynb'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(path,cwd,report):
    nb=json.loads(path.read_text('utf8'));assert nb['metadata']['ta_hafid']['no_model_deserialization']
    catalogs=[ROOT/'runs'/r/'artifacts_sha256.json' for r in ['INDEPENDENT_CAPTURE_20261109_01']]
    before={str(p):sha(p) for p in catalogs};km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
    ipdir=ROOT/'tmp/independent_review_ipython';ipdir.mkdir(parents=True,exist_ok=True)
    kernel_env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',IPYTHONDIR=str(ipdir))
    kernel_env.pop('ARRAY_NOTEBOOK_PATH',None)
    km.start_kernel(cwd=str(cwd),env=kernel_env)
    client=km.client();client.start_channels();client.wait_for_ready(timeout=60)
    def run(src,cell=None):
        mid=client.execute(src,store_history=True,allow_stdin=False,stop_on_error=True);out=[];deadline=time.monotonic()+300
        while True:
            if time.monotonic()>deadline:raise TimeoutError('Notebook cell exceeded 300s')
            try:msg=client.get_iopub_msg(timeout=5)
            except queue.Empty:continue
            if msg.get('parent_header',{}).get('msg_id')!=mid:continue
            kind=msg['msg_type'];data=msg['content']
            if kind=='status' and data['execution_state']=='idle':break
            if kind=='execute_input' and cell is not None:cell['execution_count']=data['execution_count']
            if kind in ['stream','display_data','execute_result','error']:
                entry=dict(output_type=kind,**data);entry.pop('transient',None);out.append(entry)
                if kind=='error':raise RuntimeError('\n'.join(data['traceback']))
        if cell is not None:cell['outputs']=out
    try:
        run("import joblib,pickle\ndef _blocked(*a,**k):raise RuntimeError('Review-only notebook: fitting/model deserialization forbidden')\njoblib.load=_blocked\npickle.load=_blocked\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfor _cls in [DecisionTreeClassifier,RandomForestClassifier,LogisticRegression]:_cls.fit=_blocked")
        count=0
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':
                count+=1
                if count%10==0 or count<4:print('Executing',count,'cell',i,flush=True)
                run(c['source'],c)
        assert before=={str(p):sha(p) for p in catalogs}
        nb['metadata']['ta_hafid']['executed_code_cells']=count
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
        outputs=[o for c in nb['cells'] if c['cell_type']=='code' for o in c['outputs']]
        result=dict(status='PASS_ALL_CELLS_EXECUTED',notebook=str(path),sha256=sha(path),code_cells=count,images=sum('image/png' in o.get('data',{}) for o in outputs),paper_tables=sum('paper_table' in c['metadata'] for c in nb['cells']),paper_figures=sum('paper_figure' in c['metadata'] for c in nb['cells']),errors=sum(o['output_type']=='error' for o in outputs),guarded_model_loading=True,guarded_fitting=True,new_training=0,new_inference=0,frozen_catalogs_unchanged=True,execution_cwd=str(cwd))
        report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2),flush=True)
    finally:client.stop_channels();km.shutdown_kernel(now=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--notebook',type=Path,default=DEFAULT);a.add_argument('--cwd',type=Path,default=ROOT);a.add_argument('--report',type=Path,default=ROOT/'reports/INDEPENDENT_CAPTURE_20261109/notebook_execution.json');v=a.parse_args();execute(v.notebook.resolve(),v.cwd.resolve(),v.report.resolve())
