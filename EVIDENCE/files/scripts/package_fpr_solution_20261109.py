"""Package and verify a portable prepared-feature replay archive."""
from pathlib import Path
import argparse,ast,hashlib,json,os,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/FPR_SOLUTION_20261109';ZIP=ROOT/'penyerahan/SOLUSI_FPR_DI_BAWAH_8PERSEN_UNTUK_DOSPEM_20261109.zip'
RUNS=['FPR_TARGET8_20261109_01','FPR_NORMAL_VOTING_20261109_01','FPR_SQL_GUARD_20261109_01']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def entries():
    files=set()
    for name in RUNS:files.update(p for p in (ROOT/'runs'/name).rglob('*') if p.is_file())
    files.update(p for p in OUT.rglob('*') if p.is_file())
    files.add(ROOT/'output/pdf/SOLUSI_FPR_DI_BAWAH_8PERSEN_EVIDENCE_20261109.pdf');files.add(ROOT/'notebooks/fpr_solution_20261109/FPR_SOLUSI_DAN_PREDIKSI_20261109.ipynb');files.add(ROOT/'configs/fpr_normal_voting_schema_20261109.json');files.add(ROOT/'docs/HASIL_SOLUSI_FPR_20261109.md');files.add(ROOT/'requirements.lock.txt')
    files.update((ROOT/'tests').glob('test_fpr*_20261109.py'));files.add(ROOT/'tests/test_sql_guard_rf_rescue_20261109.py')
    for name in RUNS:
        run=ROOT/'runs'/name
        for manifest in ['source_manifest.json','execution_source_manifest.json']:
            if (run/manifest).exists():files.update(ROOT/r['path'] for r in read(run/manifest))
    files.update((ROOT/'src').glob('*.py'))
    files.update(ROOT/'scripts'/s for s in ['predict_fpr_sql_guard_rf_rescue_20261109.py','verify_fpr_solution_20261109.py','build_fpr_solution_views_20261109.py','build_fpr_solution_report_20261109.py','execute_fpr_solution_notebook_20261109.py','package_fpr_solution_20261109.py'])
    # Include source dependencies of the selected scripts without all old runs.
    pending=[p for p in files if p.suffix=='.py'];seen=set()
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text('utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            for name in names:
                first=name.split('.')[0]
                for folder in ['src','scripts']:
                    dep=ROOT/folder/(first+'.py')
                    if dep.exists() and dep not in files:files.add(dep);pending.append(dep)
    return sorted(files)
def build():
    assert read(OUT/'verification.json')['status']=='PASS';assert read(OUT/'pdf_qa.json')['status']=='PASS';assert read(OUT/'notebook_execution.json')['errors']==0
    files=entries();manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    ZIP.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('MANIFEST_FPR_SHA256.json',json.dumps(manifest,indent=2))
        z.writestr('BACA_PERTAMA_FPR.txt','SOLUSI FPR - HASIL AKTUAL 9 NOVEMBER 2026\n\nBaca output/pdf/SOLUSI_FPR_DI_BAWAH_8PERSEN_EVIDENCE_20261109.pdf\nLalu buka notebooks/fpr_solution_20261109/FPR_SOLUSI_DAN_PREDIKSI_20261109.ipynb\nPenjelasan dan perintah replay: docs/HASIL_SOLUSI_FPR_20261109.md\n\nHasil blok ketiga:1/697 FP=0.1435%, bukan jaminan semua trafik.\nTidak ada training baru. Paper/dashboard lama belum diganti.\nPaket berisi model milik proyek (jangan load model dari sumber lain).\nPrepared features disertakan; raw PCAP/CSV penuh tidak diduplikasi.\nRunner ekstraksi dari raw memerlukan proyek induk dan input yang dirujuk.\n\nEnvironment:Python3.11, requirements.lock.txt.\nNotebook review tidak memuat model dan tidak melakukan training.\nVerifier saved probabilities tidak memuat model.\nCLI replay memeriksa katalog/source serta hash model milik proyek dahulu.\nPDF builder tambahan memerlukan reportlab.\n')
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP)+'  '+ZIP.name+'\n',encoding='utf8');print(json.dumps(dict(archive=str(ZIP),bytes=ZIP.stat().st_size,files=len(files),sha256=sha(ZIP))))
def verify_zip():
    with zipfile.ZipFile(ZIP) as z:
        assert len(z.namelist())==len(set(z.namelist()));assert z.testzip() is None
        rows=json.loads(z.read('MANIFEST_FPR_SHA256.json'))
        for row in rows:
            b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
        assert set(z.namelist())=={x['path'] for x in rows}|{'MANIFEST_FPR_SHA256.json','BACA_PERTAMA_FPR.txt'}
    return len(rows)
def portable():
    count=verify_zip();temp_root=Path(tempfile.gettempdir()).resolve();dest=Path(tempfile.mkdtemp(prefix='hafid_fpr_20261109_',dir=temp_root)).resolve();assert dest.is_relative_to(temp_root)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():assert (dest/name).resolve().is_relative_to(dest)
        z.extractall(dest)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8');env.pop('PYTHONPATH',None);logs=[]
    commands=[['scripts/verify_fpr_solution_20261109.py'],['scripts/predict_fpr_sql_guard_rf_rescue_20261109.py','--features','runs/FPR_SQL_GUARD_20261109_01/all_later_features.csv','--output','tmp/portable_replay.csv'],['scripts/execute_fpr_solution_notebook_20261109.py','--cwd','notebooks/fpr_solution_20261109','--report','tmp/portable_notebook.json']]
    for args in commands:
        p=subprocess.run([sys.executable,'-B',*args],cwd=dest,env=env,text=True,capture_output=True);assert p.returncode==0,p.stdout+p.stderr;logs.append(dict(command=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));print('PORTABLE_PASS '+args[0],flush=True)
    import pandas as pd
    import numpy as np
    a=pd.read_csv(dest/'tmp/portable_replay.csv',index_col='row_id',float_precision='round_trip');b=pd.read_csv(ROOT/'runs/FPR_SQL_GUARD_20261109_01/candidate_later_decisions.csv',index_col='row_id',float_precision='round_trip').loc[a.index];assert (a.candidate_prediction==b.prediction).all();cols=[c for c in a if c.startswith('p_')];delta=float(np.max(abs(a[cols].to_numpy()-b[cols].to_numpy())));assert delta<=1e-12
    notebook='notebooks/fpr_solution_20261109/FPR_SOLUSI_DAN_PREDIKSI_20261109.ipynb';assert sha(dest/notebook)==sha(ROOT/notebook),'Portable notebook outputs differ'
    result=dict(status='PASS',checked_archive_files=count,portable_directory=str(dest),saved_audit_without_parent_data=True,CLI_replayed_n=len(a),max_probability_difference=delta,portable_notebook_byte_identical=True,commands=logs,new_fits=0,raw_extraction_not_claimed_portable=True)
    write(OUT/'portable_archive_audit.json',result);print('PORTABLE_ALL_PASS',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['build','verify','portable'],required=True);a=p.parse_args()
    if a.phase=='build':build()
    elif a.phase=='portable':portable()
    else:print(json.dumps(dict(status='PASS',files=verify_zip(),sha256=sha(ZIP))))
