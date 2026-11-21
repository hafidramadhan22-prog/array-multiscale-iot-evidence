"""Package verified experimental evidence; test fresh extraction, never retrain."""
from pathlib import Path
import sys,json,hashlib,zipfile,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/ARRAY_SVM_20261118'
ZIP=ROOT/'penyerahan/UJI_SVM_ARRAY_UNTUK_DOSPEM_20261118.zip'
README='''# Uji SVM comparator - hasil aktual, 18 November 2026

Buka 01_LAPORAN_HASIL_UJI_SVM.pdf (6 halaman) terlebih dahulu.
Notebook interaktif: 02_HASIL_DAN_PREDIKSI_SVM.ipynb, atau salinan dalam
notebooks/array_svm_20261118/. Keenam sel kode sudah dieksekusi; Run All hanya
membaca hasil tersimpan dan memeriksa evidence, tanpa memuat model/training.

## Kesimpulan utama

Tujuan membuat WV konsisten lebih tinggi dari SVM BELUM tercapai.

Protokol                       WV         RF       SVM-D      SVM-G
Temporal, mean 3 seeds        97.8954    97.8648    97.8793    97.9175
Packet-private, pooled OOF    93.7694    96.9966    94.8780    65.7460
Signature/shared, pooled OOF  97.2955    97.1965    68.4596    68.4596

Angka adalah macro-F1 (%). SVM-D = RBF C1/gamma scale. SVM-G = pilihan G dari
sembilan kandidat sebelum V. Default SVM yang lebih tinggi dan RF tetap
dilaporkan. Skor SVM yang lebih rendah tidak dipilih sendirian untuk membuat
klaim superioritas. WV, bobotnya dan paper jurnal 17 November tidak diubah.

SVM-G Normal FPR: additional aligned 63/760 (8.2895%); block 3 27/697 (3.8737%);
CIC 29.05%; TON 48.30%. Pada CIC target skenario SYN, typed recall SVM-G 0%.
FPR WV tidak berubah. Semua stress sources sudah pernah diamati sebelumnya.
CIC scenario labels dan TON flow-derived labels memiliki scope berbeda dan
tidak digabung menjadi satu skor generalisasi multiclass lintas dataset.

## Memeriksa bukti

Python 3.11 dan requirements.lock.txt. Dari root folder hasil ekstraksi:

    python -B scripts/verify_array_svm_20261118.py

Audit default tidak membuka joblib, menjalankan inference atau training.
Ia menghitung ulang skor dari label/CM, memeriksa split/cap/G selection,
48 parent artifacts dengan hash asli, dan raw latency/RSS.

Replay model opsional dan eksplisit (hanya model milik eksperimen ini yang
telah diverifikasi checksum, tidak ada fit):

    python -B scripts/replay_array_svm_20261118.py

22.916 label stress/prepared diperiksa terhadap actual saved outputs.
Untuk membangun kembali ringkasan/plot dari saved results:

    python -B scripts/analyze_array_svm_20261118.py

Data rinci ada di reports/ARRAY_SVM_20261118/. CSV protocol_summary,
stress_summary, stress_per_capture, serving_summary, semua confusion matrices,
dan parent_inputs tersedia. Grafik punya PNG/PDF/SVG serta kode plotting.
Model kandidat dan G predictions: runs/ARRAY_SVM_FULL_20261118_01.
Matched evaluation: runs/ARRAY_SVM_EVALUATION_20261118_01.
Stress dengan ID unicode: runs/ARRAY_SVM_STRESS_PORTABLE_20261118_01.
Raw latency/RSS: runs/ARRAY_SVM_SERVING_20261118_01.

## Provenance dan batas

63 scientific SVC fits, 1 smoke fit terpisah, 0 fit ulang setelah melihat V.
Runner awal mengalami cache adapter error karena bobot WV berada pada berkas
seleksi terpisah. Snapshot/error logs dan koreksinya dipertahankan. Evaluasi
recovery memakai model yang sama dan kandidat G yang sudah dibekukan.
Dua tahap yang belum dimulai diselesaikan dengan grid beku yang sama.
Koreksi ID object ke unicode tidak mengubah array numerik atau membuka pickle
object arrays. Parent archive dari stress sebelum perbaikan ID juga disimpan.

Runner V2 disediakan untuk reproduksi training mendatang pada parent workspace
lengkap, bukan hasil full run baru. Jangan menjalankannya hanya untuk membaca
evidence. Canonical CSV, PCAP, dan parent feature arrays besar tidak disalin;
training ulang memerlukan data/parent project yang dirujuk protokol.
requirements.lock.txt mencatat environment; tidak ada kredensial/raw malware.

Cost adalah prepared-feature label latency pada satu laptop. SVM memakai
pipeline standar; compact SVM belum dibuat. Tidak ada klaim semua hardware,
penghematan energi atau setiap native allocation event. Source/class confounding,
shared histories dan adaptasi tahap penelitian sebelumnya tetap membatasi hasil.
Tidak ada revisi otomatis paper, submission atau persetujuan final penulis.

Semua member kecuali manifest.json sendiri mempunyai SHA256 dan ukuran dalam
manifest.json. Berkas .zip.sha256 serta .zip.verification.json mendampingi ZIP.
'''
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def command(args,cwd):
    r=subprocess.run([str(a) for a in args],cwd=cwd,text=True,capture_output=True,timeout=300)
    if r.returncode:raise RuntimeError(r.stdout+'\n'+r.stderr)
    return dict(returncode=r.returncode,stdout=r.stdout[-10000:],stderr=r.stderr[-2000:])
def main():
    q=json.loads((R/'pdf_quality.json').read_text());assert q['visual_review'].startswith('PASS')
    pdf=ROOT/'output/pdf/ARRAY_HAFID_UJI_SVM_EVIDENCE_20261118.pdf';assert sha(pdf)==q['sha256']
    assert json.loads((R/'verification.json').read_text())['status']=='PASS_INDEPENDENT_SVM_SAVED_EVIDENCE_AUDIT'
    assert json.loads((R/'model_replay_verification.json').read_text())['label_checks']==22916
    notebook=ROOT/'notebooks/array_svm_20261118/UJI_SVM_HASIL_DAN_PREDIKSI_20261118.ipynb'
    nb=json.loads(notebook.read_text(encoding='utf-8'));code=[c for c in nb['cells'] if c['cell_type']=='code'];assert len(code)==6 and all(c['execution_count'] and not any(o['output_type']=='error' for o in c['outputs']) for c in code)
    entries={'00_BACA_DULU.md':README.encode('utf-8'),'01_LAPORAN_HASIL_UJI_SVM.pdf':pdf,'02_HASIL_DAN_PREDIKSI_SVM.ipynb':notebook}
    for folder in [ROOT/'runs'/n for n in ['ARRAY_SVM_FULL_20261118_01','ARRAY_SVM_EVALUATION_20261118_01','ARRAY_SVM_STRESS_20261118_01','ARRAY_SVM_STRESS_PORTABLE_20261118_01','ARRAY_SVM_SERVING_20261118_01','ARRAY_SVM_SMOKE_20261118_01']]+[R,ROOT/'notebooks/array_svm_20261118']:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ['package_verification.json','delivery_final.json']:
                entries[p.relative_to(ROOT).as_posix()]=p
    for p in ROOT.glob('scripts/*array*svm*20261118.py'):entries[p.relative_to(ROOT).as_posix()]=p
    for p in (ROOT/'src').glob('*.py'):entries[p.relative_to(ROOT).as_posix()]=p
    for name in ['run_wb_research.py','prepare_wb_behavior.py','prepare_sbfe.py','profile_array_serving_20261117.py','profile_exact_runtime.py']:
        entries['scripts/'+name]=ROOT/'scripts'/name
    for name in ['ARRAY_SVM_COMPARATOR_PROTOCOL_20261118.md','ARRAY_SVM_CACHE_ADAPTER_ADDENDUM_20261118.md','HASIL_UJI_SVM_ARRAY_20261118.md']:entries['docs/'+name]=ROOT/'docs'/name
    entries['tests/test_array_svm_20261118.py']=ROOT/'tests/test_array_svm_20261118.py'
    entries['requirements.lock.txt']=ROOT/'requirements.lock.txt';entries['configs/target.json']=ROOT/'configs/target.json'
    entries[pdf.relative_to(ROOT).as_posix()]=pdf
    records=[];partial=ZIP.with_suffix('.zip.tmp')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,p in sorted(entries.items()):
            assert not name.startswith('/') and '..' not in Path(name).parts
            b=p if isinstance(p,bytes) else p.read_bytes();z.writestr(name,b);records.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        z.writestr('manifest.json',json.dumps(dict(files=records,scientific_SVC_fits=63,smoke_fits_separate=1,RF_retained=True,WV_unchanged=True,full_raw_dataset_included=False),indent=2)+'\n')
    partial.replace(ZIP)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for r in records:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
        target=Path(tempfile.mkdtemp(prefix='svm_evidence_'))
        for name in z.namelist():
            p=target/name;assert p.resolve().is_relative_to(target.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    result=dict(status='PASS_SVM_EXPERIMENT_PACKAGE_AND_FRESH_EXTRACTION',path=ZIP.relative_to(ROOT).as_posix(),bytes=ZIP.stat().st_size,sha256=sha(ZIP),files=len(records)+1,scientific_SVC_fits=63,RF_retained=True,WV_unchanged=True,pdf_pages=6,notebook_code_cells=6,extraction=str(target),checks={})
    result['checks']['saved_audit']=command([sys.executable,'-B','scripts/verify_array_svm_20261118.py'],target)
    result['checks']['explicit_owned_model_replay']=command([sys.executable,'-B','scripts/replay_array_svm_20261118.py'],target)
    result['checks']['notebook_Run_All']=command([sys.executable,'-B','scripts/build_array_svm_notebook_20261118.py'],target)
    # Same code, inputs and environment produce byte-identical executed notebook.
    assert sha(target/notebook.relative_to(ROOT))==sha(notebook)
    result['notebook_portable_outputs_byte_identical']=True
    write(R/'package_verification.json',result);write(ZIP.with_suffix('.zip.verification.json'),result)
    ZIP.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+ZIP.name+'\n',encoding='ascii')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
