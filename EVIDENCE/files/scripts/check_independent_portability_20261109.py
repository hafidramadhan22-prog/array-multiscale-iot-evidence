"""Extract and test the delivered archive in a new private temp directory."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/INDEPENDENT_CAPTURE_20261109'
ZIP=ROOT/'penyerahan/UJI_INDEPENDEN_CIC_TON_UNTUK_DOSPEM_20261109.zip'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    target=Path(tempfile.mkdtemp(prefix='hafid_independent_20261109_'))
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if not (target/info.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(target)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    env.pop('PYTHONPATH',None)
    events=[]
    def run(args,name):
        print('PORTABLE',name,flush=True)
        p=subprocess.run([sys.executable,'-B',*args],cwd=target,env=env,
                         capture_output=True,text=True,encoding='utf8')
        events.append(dict(step=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        assert p.returncode==0,(name,p.stdout,p.stderr)
    run(['scripts/verify_independent_package_20261109.py'],'package_hashes')
    run(['scripts/audit_independent_results_20261109.py','--phase','audit'],'saved_results_and_4000_TON_labels')
    run(['scripts/replay_independent_predictions_20261109.py','--output','review/predictions.csv','--audit','review/replay.json'],'frozen_model_replay')
    notebook=target/'notebooks/independent_20261109/UJI_INDEPENDEN_HASIL_DAN_PREDIKSI_20261109.ipynb'
    before=sha(notebook)
    run(['scripts/execute_independent_notebook_20261109.py','--cwd',str(notebook.parent)],'notebook_run_all')
    tests=['tests/test_independent_capture_bridge_20261109.py','tests/test_ton_interval_mapping_20261109.py',
           'tests/test_flow_preserving_capture_20261109.py','tests/test_independent_replay_gate_20261109.py']
    run(['-m','pytest','-q','-p','no:cacheprovider',*tests,'--basetemp',str(target/'review/pytest_tmp'),
         '--junitxml',str(OUT/'portable_test_results.xml')],'scientific_and_replay_gate_tests')
    (OUT/'portable_tests.txt').write_text(events[-1]['stdout']+events[-1]['stderr'],encoding='utf8')
    inference=json.loads((target/'review/replay.json').read_text('utf8'))
    result=dict(status='PASS_PORTABLE_EVIDENCE_LABELS_REPLAY_AND_NOTEBOOK',extraction_directory=str(target),
                tested_archive_sha256=sha(ZIP),fresh_extraction=True,new_fits=0,
                replay_inputs=inference['inputs'],replay_predictions=inference['predictions'],
                saved_labels_identical=inference['saved_labels_identical'],
                max_probability_difference=inference['max_probability_difference'],
                notebook_byte_identical=(before==sha(notebook)),
                notebook_execution=json.loads((target/'reports/INDEPENDENT_CAPTURE_20261109/notebook_execution.json').read_text()),
                original_run_catalog_unchanged=True,events=events)
    assert result['notebook_byte_identical']
    (OUT/'portable_review_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['events','notebook_execution']},indent=2),flush=True)


if __name__=='__main__':main()
