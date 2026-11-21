"""Verify and deliver cached-data WV weight experiment without changing the paper."""
from pathlib import Path
import json,sys,hashlib,zipfile,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/ARRAY_WEIGHT_RETUNE_20261118';ZIP=ROOT/'penyerahan/UJI_BOBOT_G_ARRAY_UNTUK_DOSPEM_20261118.zip'
README='''# Uji bobot WV pada G - 18 November 2026

Buka 01_LAPORAN_UJI_BOBOT_G.pdf (4 halaman) dan 02_HASIL_DAN_PREDIKSI.ipynb.
Notebook berisi 4 sel kode yang telah dieksekusi dan tidak memuat model.

## Hasil

Kebijakan utama dipilih pada G dengan syarat FPR/recall kelas G tidak memburuk.
Syarat empiris G tidak menjamin V. Tidak ada pencarian ulang menggunakan V.

Macro-F1 (%)                    WV lama     WV-G utama      RF      SVM default
Temporal, mean 3 seeds          97.8954      97.8189      97.8648     97.8793
Packet-private, pooled OOF      93.7694      94.1734      96.9966     94.8780
Signature/shared, pooled OOF    97.2955      97.2955      97.1965     68.4596

Private FPR 0.5041% -> 0.4578%, errors 3071 -> 2952. Namun SQL recall turun,
attack-to-Normal 1816 -> 1960, dan binary attack recall juga turun. Temporal
FPR naik 0.9240% -> 1.1054%. Belum ada perbaikan yang konsisten atau dasar
klaim WV melebihi RF/default SVM pada private history.

Stress menggunakan predictor temporal seed-42 dan bobotnya sendiri:
- Additional aligned: 321/760 -> 0/760 false alarms.
- Block 3: 396/697 -> 1/697 false alarms.
- CIC Normal FPR 30.15% -> 36.10%; typed SYN scenario recall -> 0%.
- TON Normal FPR 24.35% -> 8.15%, tetapi Ransomware-to-Normal 1630 -> 1769/2000.
Semua sumber sudah pernah diamati; tidak ada independent final test baru.
CIC scenario dan TON flow-derived labels dilaporkan dengan scope terpisah.

## Apa yang dihitung

Expert, fitur, split, A training, RUS+SMOTE dan floor tetap. Hanya bobot yang
dipilih ulang memakai saved expert probabilities. 3881 kandidat per tahap,
7 tahap = 27167 penilaian G; 0 refit expert dan 0 joblib loading. Semua tiga
kebijakan baru, RF, SVM default/selected, LR, DT, XGB, LGB dan uniform disimpan.
Tidak ada pemilihan konfigurasi yang lebih lemah untuk membuat WV menang.

## Audit dan reproduksi angka tanpa model

Python 3.11, numpy/pandas; requirements.lock.txt mencatat environment lengkap.
Dari root folder hasil ekstraksi:

    python -B scripts/verify_array_weight_retune_20261118.py

Audit membentuk ulang seluruh G search dan V/stress outputs dari probabilitas
yang disertakan, tanpa classifier training, model loading atau unduhan data.
Ia memeriksa 50 parent artifacts, ID/split, constraints, candidate selection,
84 label vectors tahap, 28 probability vectors WV dan 220 stress label vectors.
Angka diringkas di reports/ARRAY_WEIGHT_RETUNE_20261118/; seluruh G candidates,
labels, weights, freeze records, source snapshots dan original input catalogs
ada di runs/ARRAY_WEIGHT_RETUNE_FULL_20261118_01/. Smoke terpisah tidak dipakai
di hasil. Raw canonical CSV/PCAP dan parent model files tidak disalin.

Untuk membangun ulang tabel/plot, tanpa model:

    python -B scripts/analyze_array_weight_retune_20261118.py

Notebook Run All membaca saved results dan menjalankan audit di atas.
Runner eksperimental asli mengacu pada parent project/canonical CSV yang
diidentifikasi protokol. Untuk mengaudit hasil, gunakan perintah verifier;
data cached lengkap untuk merekonstruksi search sudah disertakan di paket.

## Batas

Perubahan bobot mengubah fungsi prediksi. Saving runtime dan audit SHAP WV
lama tidak otomatis berlaku. Tidak ada profil runtime/XAI, capture baru,
perubahan paper, submission atau approval penulis. Data V telah diamati di
riset sebelumnya; semua hasil tetap exploratory follow-up. Tidak ada grid
kedua, perubahan floor/toleransi atau pemilihan dari hasil V setelah pengujian.
Manuskrip 17 November dan seluruh hasil sebelumnya dipertahankan.

Manifest mencatat size/SHA256 setiap member kecuali manifest sendiri. Sidecar
.zip.sha256 dan .zip.verification.json melengkapi paket.
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
    return dict(returncode=0,stdout=r.stdout[-12000:],stderr=r.stderr[-2000:])
def main():
    assert json.loads((R/'verification.json').read_text())['status']=='PASS_INDEPENDENT_WEIGHT_SEARCH_AND_PREDICTIONS'
    quality=json.loads((R/'pdf_quality.json').read_text());assert quality['visual_review'].startswith('PASS')
    pdf=ROOT/'output/pdf/ARRAY_HAFID_UJI_BOBOT_G_EVIDENCE_20261118.pdf';assert sha(pdf)==quality['sha256']
    notebook=ROOT/'notebooks/array_weight_retune_20261118/UJI_BOBOT_G_HASIL_DAN_PREDIKSI_20261118.ipynb'
    nb=json.loads(notebook.read_text(encoding='utf-8'));cells=[c for c in nb['cells'] if c['cell_type']=='code'];assert len(cells)==4 and all(c['execution_count'] for c in cells)
    entries={'00_BACA_DULU.md':README.encode('utf-8'),'01_LAPORAN_UJI_BOBOT_G.pdf':pdf,'02_HASIL_DAN_PREDIKSI.ipynb':notebook}
    for folder in [ROOT/'runs/ARRAY_WEIGHT_RETUNE_FULL_20261118_01',ROOT/'runs/ARRAY_WEIGHT_RETUNE_SMOKE_20261118_01',R,ROOT/'notebooks/array_weight_retune_20261118']:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ['package_verification.json','delivery_final.json']:entries[p.relative_to(ROOT).as_posix()]=p
    for p in ROOT.glob('scripts/*array*weight*20261118.py'):entries[p.relative_to(ROOT).as_posix()]=p
    for p in (ROOT/'src').glob('*.py'):entries[p.relative_to(ROOT).as_posix()]=p
    for name in ['verify_array_svm_20261118.py','run_array_svm_20261118.py','run_wb_research.py','prepare_wb_behavior.py','prepare_sbfe.py']:entries['scripts/'+name]=ROOT/'scripts'/name
    for name in ['ARRAY_WEIGHT_RETUNE_PROTOCOL_20261118.md','HASIL_UJI_BOBOT_G_ARRAY_20261118.md']:entries['docs/'+name]=ROOT/'docs'/name
    entries['requirements.lock.txt']=ROOT/'requirements.lock.txt';entries['configs/target.json']=ROOT/'configs/target.json';entries['tests/test_array_weight_retune_20261118.py']=ROOT/'tests/test_array_weight_retune_20261118.py';entries[pdf.relative_to(ROOT).as_posix()]=pdf
    assert not any(n.endswith(('.joblib','.pcap','.pcapng','.cookies')) for n in entries)
    records=[];partial=ZIP.with_suffix('.zip.tmp')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,item in sorted(entries.items()):
            assert not name.startswith('/') and '..' not in Path(name).parts
            b=item if isinstance(item,bytes) else item.read_bytes();z.writestr(name,b);records.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        z.writestr('manifest.json',json.dumps(dict(files=records,expert_refits=0,models_loaded=False,G_candidate_evaluations=27167,RF_and_SVMs_retained=True),indent=2)+'\n')
    partial.replace(ZIP)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for r in records:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
        target=Path(tempfile.mkdtemp(prefix='weight_evidence_'))
        for name in z.namelist():
            p=target/name;assert p.resolve().is_relative_to(target.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    result=dict(status='PASS_WEIGHT_EXPERIMENT_ARCHIVE_AND_PORTABLE_AUDIT',path=ZIP.relative_to(ROOT).as_posix(),bytes=ZIP.stat().st_size,sha256=sha(ZIP),files=len(records)+1,pdf_pages=4,notebook_code_cells=4,new_expert_fits=0,G_candidate_evaluations=27167,RF_and_SVMs_retained=True,old_journal_unchanged=True,extraction=str(target),checks={})
    result['checks']['fresh_saved_audit']=command([sys.executable,'-B','scripts/verify_array_weight_retune_20261118.py'],target)
    result['checks']['fresh_notebook_Run_All']=command([sys.executable,'-B','scripts/build_array_weight_retune_notebook_20261118.py'],target)
    assert sha(target/notebook.relative_to(ROOT))==sha(notebook);result['portable_notebook_byte_identical']=True
    write(R/'package_verification.json',result);write(ZIP.with_suffix('.zip.verification.json'),result);ZIP.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+ZIP.name+'\n',encoding='ascii')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
