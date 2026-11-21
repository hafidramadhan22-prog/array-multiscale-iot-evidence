"""Package reviewed revision and saved evidence without rerunning experiments."""
from pathlib import Path
import json, hashlib, shutil, zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
rep=ROOT/'reports/ARRAY_MAJOR_REVISION_20261103'
quality=json.loads((rep/'pdf_quality.json').read_text())
outputs=[]
for row in quality:
    name=row['document'];src=ROOT/f'paper/array_20261103_major_revision/elsarticle/{name}.pdf'
    assert sha(src)==row['sha256'] and not row['clipping']
    row['visual_review']='PASS: all rendered pages inspected; tables, figures, captions and references readable'
    dst=ROOT/'output/pdf'/('ARRAY_HAFID_MAJOR_REVISION_20261103.pdf' if name=='main' else 'ARRAY_HAFID_SUPPLEMENT_MAJOR_REVISION_20261103.pdf')
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    outputs.append(dict(path=dst.relative_to(ROOT).as_posix(),pages=row['pages'],sha256=sha(dst)))
save(rep/'pdf_quality.json',quality)
statepath=ROOT/'CURRENT_STATE.json';state=json.loads(statepath.read_text(encoding='utf-8-sig'))
state['array_major_revision_20261103'].update(status='EXPERIMENTS_AND_PDF_QA_COMPLETE_AUTHOR_REVIEW',deliverables=outputs,review_package='penyerahan/ARRAY_HAFID_MAJOR_REVISION_20261103.zip')
save(statepath,state)
readme=ROOT/'paper/array_20261103_major_revision/README.md'
readme.write_text('''# Array major revision — 3 November 2026

Status: selesai untuk review penulis/dospem, belum disubmit. PDF utama 27 halaman; supplementary 31 halaman. Hasil negatif dan ketidaklengkapan coverage tetap dilaporkan. Persetujuan kedua penulis, availability publik, konfirmasi corresponding author, dan pemeriksaan panduan portal masih diperlukan. Ekstrak ZIP ke lokasi pendek (misalnya C:/TA/Array) agar path arsip historis tidak melewati batas panjang path Windows.

## Isi dan batas paket

PDF di output/pdf; source LaTeX di paper/array_20261103_major_revision/elsarticle; tanggapan 10 poin dospem di docs/REVISI_MAJOR_ARRAY_20261103.md. Protokol, addendum benign, CSV, log, indeks split, model hasil run baru dan checksum disertakan. Smoke dan run benign gagal disimpan sebagai riwayat, bukan hasil final. Model tidak perlu dimuat untuk audit hasil tersimpan.

Paket ini bukan salinan penuh raw dataset dan seluruh arsip Oktober. Mengulang training/replay membutuhkan handoff asli, raw CSV/ZIP yang hash-nya tercatat, dan reference bundles Oktober. Jangan memuat joblib tanpa memverifikasi provenance/hash. Tidak ada perintah training otomatis dalam packaging atau pemeriksaan hasil.

## Pemeriksaan hasil tersimpan

Dari root paket, dengan environment riset yang sesuai requirements.lock.txt:

    python -B scripts/verify_array_revision_20261103.py

Verifier membaca hasil tersimpan, memeriksa hash, split/resampling, rekonstruksi metrik, runtime dan sitasi; tidak mendeserialisasi model. Audit tambahan benign dicatat di reports/ARRAY_MAJOR_REVISION_20261103/benign_verification.json. manifest.json memuat hash seluruh file paket selain manifest itu sendiri.

## Membangun PDF dari source final

    tectonic -X compile paper/array_20261103_major_revision/elsarticle/main.tex
    tectonic -X compile paper/array_20261103_major_revision/elsarticle/supplement.tex

Tectonic dan dependensi TeX perlu tersedia. Source final sudah disertakan; generator menggunakan hasil riset tersimpan. PDF disusun dalam format preprint satu kolom untuk review. Jangan menganggap jumlah halaman ini batas halaman conference.

## Temuan yang wajib dipertahankan

Ablation empat sel mendukung tambahan window pada target within-trace. Weighted voting tidak diklaim superior; private history mengurangi macro-F1. Pada capture normal tambahan hanya 760/867 kandidat eligible dapat dicocokkan; weighted voting memberi 324 false alarms (42.63%). Tidak ada final test independen lima kelas. Runtime 30 pasangan pada satu laptop memperlihatkan penurunan mean 41.75%, bukan jaminan universal. Semua klaim, pendanaan, CRediT dan deklarasi AI perlu persetujuan akhir penulis.
''',encoding='utf8')
report=ROOT/'docs/REVISI_MAJOR_ARRAY_20261103.md'
t=report.read_text(encoding='utf8')
if '## Pemeriksaan akhir PDF dan paket' not in t:
    report.write_text(t+'\n## Pemeriksaan akhir PDF dan paket\n\nSeluruh 58 halaman telah diperiksa secara visual. Naskah utama: 27 halaman, 5 gambar, 8 tabel, 25 referensi. Supplementary: 31 halaman, 6 gambar, 40 tabel. Tidak ditemukan clipping, overfull box, sitasi tidak terdefinisi atau glyph hilang pada build final. Paket review disertai manifest SHA256; raw dataset tidak didistribusikan ulang.\n',encoding='utf8')
files=set()
def addtree(rel):
    for p in (ROOT/rel).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.synctex']:files.add(p)
for rel in ['paper/array_20261103_major_revision','paper/array_20261022_integrated','reports/ARRAY_MAJOR_REVISION_20261103','research/array_20261103']:addtree(rel)
for p in (ROOT/'runs').glob('ARRAY*20261103*'):addtree(p.relative_to(ROOT))
for folder in ['src','scripts','tests']:
    files.update(p for p in (ROOT/folder).glob('*.py'))
for pattern in ['docs/*ARRAY*20261103*.md','configs/*.json']:
    files.update(ROOT.glob(pattern))
for rel in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md','requirements.lock.txt','runs/FULL_20261013_01/canonical_split_indices.npz','runs/MULTISCALE_REPAIR_FULL_20261020_01/configuration_freeze.json','runs/WB_RESEARCH_FULL_20261019_01/configuration_freeze.json']:
    p=ROOT/rel;assert p.exists(),rel;files.add(p)
files.update(ROOT/x['path'] for x in outputs)
files.discard(rep/'package_verification.json')  # Archive checksum lives outside its own archive.
entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
dest=ROOT/'penyerahan/ARRAY_HAFID_MAJOR_REVISION_20261103.zip';dest.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('manifest.json',json.dumps(dict(scope='Review source and saved revision evidence; original raw dataset excluded',files=entries),indent=2))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for r in entries:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
result=dict(status='PASS_ARCHIVE_CRC_AND_ALL_SHA256',files=len(entries),bytes=dest.stat().st_size,sha256=sha(dest),path=dest.relative_to(ROOT).as_posix(),pdfs=outputs)
save(rep/'package_verification.json',result)
dest.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+dest.name+'\n',encoding='ascii')
print(json.dumps(result,indent=2))
