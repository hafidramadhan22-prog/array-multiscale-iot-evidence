"""Package sealed FPR evidence and verify it from a separate extracted root."""
from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/SQL_FPR_REPAIR_20261105'
PDF=ROOT/'output/pdf/HASIL_PERBAIKAN_FPR_NORMAL_SQL_20261105.pdf'
ZIP=ROOT/'penyerahan/PERBAIKAN_FPR_NORMAL_SQL_UNTUK_DOSPEM_20261105.zip'
RUNS=['runs/PAYLOAD_TARGET_REPAIR_20261105_01','runs/SQL_DISAGREEMENT_REPAIR_20261105_01']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    assert read(R/'verification.json')['status']=='PASS'
    assert read(R/'pdf_qa.json')['status']=='PASS' and PDF.exists()
    assert not ZIP.exists(),'Archive already exists; preserve it and choose a new version.'
    guide='''# Mulai di sini - audit dan prototipe penurunan false alarms

Untuk M. Hafid Ramadhan dan dosen pembimbing, 5 November 2026.

## Baca terlebih dahulu

1. `output/pdf/HASIL_PERBAIKAN_FPR_NORMAL_SQL_20261105.pdf`: enam halaman penyebab, eksperimen, confusion matrix, biaya dan batas kesimpulan.
2. `docs/HASIL_PERBAIKAN_FPR_NORMAL_SQL_20261105.md`: uraian serta perintah lengkap.
3. `reports/SQL_FPR_REPAIR_20261105/`: seluruh tabel ringkas, contoh input/output dan hasil verifikasi.
4. Kedua folder `runs/`: bukti terperinci, model/manifest, input beku, indeks, OOF, hasil semua metode, timing dan RSS.

## Arti hasil

- Model utama lama: 321/760 alarm setelah perbaikan format, sebelumnya324/760. Semuanya Normal dianggapSQL.
- Refit langsung +syntax310, +TF-IDF312, +gabungan331; payload saja belum menyelesaikan masalah.
- Prototipe konflik RF/WV dengan pengaman payload:0/760; seluruh10.358 labelV identik, termasuk dua missed attacks.
- RF-only guard juga0/760 dan lebih ringan. Manfaat unik TF-IDF belum terbukti.
- G/C/V tidak memiliki satu pun konflik eligible; belum ada serangan yang menguji aktivasi guard. Ini retrospektif/within-trace, bukan jaminan keamanan/independent generalization.
- Naskah, dashboard, dan model utama proyek tidak diganti. Jalur guard adalah kandidat baru; klaim SHAP/preservation/runtime artikel lama tidak otomatis berpindah.

## Menjalankan verifikasi tanpa training classifier ulang

Gunakan Python3.11 dan environment terisolasi dengan versi paket sama seperti `requirements.lock.txt` serta `environment.json` di run. Pada komputer asal, `.venv` yang sesuai sudah tersedia. Untuk komputer lain:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.lock.txt
.\\.venv\\Scripts\\python.exe -B scripts\\verify_sql_fpr_repair_20261105.py
```

Verifier memeriksa checksum sebelum membuka model/input milik eksperimen ini; jangan mengganti dengan joblib dari sumber lain. Verifier memfit ulang vocabulary training-only untuk audit, bukan classifier. Raw CSV/PCAP dan run lama besar tidak diperlukan untuk menghitung ulang metrik dan memeriksa model yang dibekukan.

## Menjalankan prediksi contoh

```powershell
.\\.venv\\Scripts\\python.exe -B scripts\\predict_sql_guard_20261105.py --base reports\\SQL_FPR_REPAIR_20261105\\example_base760.csv --payload reports\\SQL_FPR_REPAIR_20261105\\example_payload760.csv --output reports\\SQL_FPR_REPAIR_20261105\\prediksi_uji_baru.csv
```

Output harus path baru. Input dua CSV memakai `row_id` unik dan urutan identik. Base adalah85 fitur hasil extractor C60/O10 yang sama; raw payload berupa hex `tcp.payload` dan `mqtt.msg`. Ini bukan loader raw CSV63/PCAP langsung. Label tidak masuk keputusan. Contoh hasil yang sudah diverifikasi tersedia di `example_predictions760.csv`.

## Batas archive

Archive berisi input fitur beku yang cukup untuk inference dan audit hasil kedua eksperimen. Untuk mengulang pipeline dari raw PCAP/CSV dan training utama dari awal, gunakan project induk dengan raw data dan run dependencies yang dicatat manifest. Jangan otomatis menjalankan driver training; seluruh68 fit telah selesai. Source disertakan untuk pemeriksaan. Data turunan/content mengikuti ketentuan sumber dataset; paket ini bukan klaim repository publik atau lisensi baru.

`PACKAGE_MANIFEST_SHA256.json` mencatat setiap file archive selain dirinya sendiri. Artefak run juga mempunyai catalog tersegel masing-masing. Portable verification tidak melatih classifier ulang atau mengubah data asli.
'''
    (R/'README_MULAI_DI_SINI.md').write_text(guide,encoding='utf8')
    files=set()
    for rel in RUNS:
        folder=ROOT/rel
        files.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        files.update(ROOT/r['path'] for r in read(folder/'source_manifest.json'))
    files.update((ROOT/'src').glob('*.py'))
    for rel in ['scripts/predict_sql_guard_20261105.py','scripts/verify_sql_fpr_repair_20261105.py','scripts/build_sql_fpr_repair_report_20261105.py','scripts/package_sql_fpr_repair_20261105.py','docs/HASIL_PERBAIKAN_FPR_NORMAL_SQL_20261105.md','docs/PAYLOAD_TFIDF_PILOT_PROTOCOL_20261105.md','tests/test_payload_pilot_20261105.py','tests/test_payload_tfidf_pilot_20261105.py','tests/test_payload_target_repair_20261105.py','tests/test_sql_disagreement_guard_20261105.py']:
        files.add(ROOT/rel)
    for rel in ['tests/conftest.py','pytest.ini','pyproject.toml']:
        if (ROOT/rel).exists():files.add(ROOT/rel)
    files.add(PDF)
    files.update(p for p in R.iterdir() if p.is_file() and p.name not in ['package_verification.json','archive_manifest.json'])
    assert all(p.is_file() and ROOT in p.resolve().parents for p in files)
    catalog=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    extra=dict(path='README_MULAI_DI_SINI.md',bytes=len(guide.encode('utf8')),sha256=hashlib.sha256(guide.encode('utf8')).hexdigest());catalog.append(extra)
    ZIP.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('README_MULAI_DI_SINI.md',guide)
        z.writestr('PACKAGE_MANIFEST_SHA256.json',json.dumps(catalog,indent=2,ensure_ascii=False)+'\n')
    print('PACKAGED',len(catalog),'files',ZIP.stat().st_size,'bytes',flush=True)
    with tempfile.TemporaryDirectory(prefix='sql_fpr_repair_',dir=ROOT/'tmp') as td:
        dest=Path(td).resolve();assert (ROOT/'tmp').resolve() in dest.parents
        with zipfile.ZipFile(ZIP) as z:
            assert z.testzip() is None
            for name in z.namelist():assert dest in (dest/name).resolve().parents
            z.extractall(dest)
        for r in read(dest/'PACKAGE_MANIFEST_SHA256.json'):
            p=dest/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
        env=str((ROOT/'.venv/Scripts/python.exe').resolve())
        check=subprocess.run([env,'-B',str(dest/'scripts/verify_sql_fpr_repair_20261105.py')],cwd=dest,text=True,capture_output=True)
        if check.returncode:raise RuntimeError(check.stdout+check.stderr)
        c=dest/'reports/SQL_FPR_REPAIR_20261105'
        pred=subprocess.run([env,'-B',str(dest/'scripts/predict_sql_guard_20261105.py'),'--base',str(c/'example_base760.csv'),'--payload',str(c/'example_payload760.csv'),'--output',str(c/'portable_new_predictions.csv')],cwd=dest,text=True,capture_output=True)
        if pred.returncode:raise RuntimeError(pred.stdout+pred.stderr)
        import pandas as pd
        a=pd.read_csv(c/'example_predictions760.csv');b=pd.read_csv(c/'portable_new_predictions.csv')
        pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-12,rtol=0)
        assert read(c/'verification.json')['status']=='PASS'
    result=dict(status='PASS',archive=ZIP.relative_to(ROOT).as_posix(),bytes=ZIP.stat().st_size,sha256=sha(ZIP),files=len(catalog),zip_CRC='PASS',every_manifest_digest='PASS',portable_independent_verifier='PASS',portable_CSV_inference='PASS',classifier_fits_during_packaging=0)
    write(R/'package_verification.json',result);write(R/'archive_manifest.json',catalog)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
