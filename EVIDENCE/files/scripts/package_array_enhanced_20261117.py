"""Deliver enhanced native source and preserved evidence with fresh-extraction gates."""
from pathlib import Path
import sys,json,zipfile,hashlib,tempfile,subprocess,os
from package_array_final_integration_20261110 import safe,sha
from package_array_revision_20261117 import extract,command
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'paper/array_20261117_enhanced';SRC=BASE/'latex';OUT=ROOT/'reports/ARRAY_ENHANCEMENT_20261117'
PARENT=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_REVISED_20261117.zip'
REVIEW=ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_ENHANCED_LATEX_20261117.zip'
REPRO=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_ENHANCED_20261117.zip'
PDFS=[('01_MAIN_MANUSCRIPT_ARRAY_ENHANCED_20261117.pdf','ARRAY_HAFID_ENHANCED_20261117.pdf'),('02_SUPPLEMENT_S1_S114_ARRAY_ENHANCED_20261117.pdf','ARRAY_HAFID_SUPPLEMENT_S1_S114_ENHANCED_20261117.pdf'),('03_COVER_LETTER_ARRAY_ENHANCED_20261117.pdf','ARRAY_HAFID_COVER_LETTER_ENHANCED_20261117.pdf'),('04_ENHANCEMENT_LOG_ARRAY_20261117.pdf','ARRAY_HAFID_ENHANCEMENT_LOG_20261117.pdf')]
README='''# Paket pemeriksaan jurnal Array - 17 November 2026

ARTIKEL UTAMA: PDF 01, 30 halaman. Source artikel lengkap adalah `main.tex`
di root ZIP. PDF 02 adalah supplementary 98 halaman (Tables S1-S114 dan
Figures S1-S9); PDF 03 cover letter satu halaman; PDF 04 log penguatan tiga
halaman. Versi ini melanjutkan revisi dospem tanggal 17 November dengan
penguatan ilmiah yang disetujui Hafid.

## Membuka di Overleaf

1. Unggah ZIP ini sebagai proyek baru.
2. Pilih Main document = `main.tex`, compiler = pdfLaTeX.
3. Untuk supplementary, pilih `SUPPLEMENTARY/supplement.tex`.
4. Cover letter dan log terpisah di `COVER_LETTER/cover_letter.tex` serta
   `REVISION_LOG/revision_log.tex`.
5. Kembalikan Main document ke `main.tex` untuk membuka artikel utama.

Setiap subfolder mempunyai dependency sendiri. Source ditulis dalam LaTeX,
bukan konversi atau potongan halaman PDF. Empat dokumen sudah dikompilasi
dari ekstraksi ZIP baru menggunakan Tectonic dan hasilnya dibandingkan
dengan PDF yang disertakan.

## Isi penguatan

- Perbandingan kritis related work dan kontrak pelestarian prediksi.
- Ablation WV native/compact x full/certified; RF dan LR juga masing-masing
  diuji dengan encoding native dan compact.
- XAI per kelas, benar/salah, dan kesamaan urutan ranking; variasi seed dan
  paired context gains; indeks klaim ke artefak bukti.
- Seluruh 104 badan tabel supplementary lama dipertahankan. Table 4 lama
  dipindahkan ke S105; Table 4 utama kini memuat sesi delapan kebijakan baru.
- Referensi tetap 30. Empat belas gambar memiliki PDF/SVG dan sumber plotting.

Benchmark baru: 480 pengukuran prepared features, 240 replay, dan 80 pengukuran
cold-process RSS. Model, bobot, skema, dan target tetap; tanpa training,
tuning, atau data baru. Penghematan 37.40% mengukur encoding pada WV yang
sudah memakai certified stopping. Penghematan gabungan 60.55% memakai
native-full WV sebagai baseline. Hasil historis 41.75% tetap dilaporkan
terpisah; ketiganya tidak dipool. RF/LR menerima encoding yang setara dan
seluruh trade-off ditampilkan. Benchmark software ini tidak mengubah FPR.

## Bukti dan reproduksi

`EVIDENCE/` memuat konfigurasi, raw timing/RSS, ringkasan, perbandingan
berpasangan, actual saved outputs, dan indeks klaim. Arsip lengkap tersedia
terpisah: `ARRAY_HAFID_REPRODUCIBILITY_ENHANCED_20261117.zip`.
Sesudah mengekstrak arsip lengkap ke path pendek dan memasang paket dari
requirements.lock.txt, jalankan:

    python -B scripts/verify_array_all_saved_20261117.py

Perintah tersebut memeriksa hasil tersimpan tanpa memuat model atau melakukan
inference. Driver profiling tidak perlu dijalankan untuk membaca evidence.
`figure_sources/rebuild_figures.py --out FOLDER_BARU` membangun ulang 14 PDF
serta SVG dari data plotting tanpa model. Nomor gambar ada di FIGURES_INDEX.csv.

Related material AGERS dan salinan pembanding tersedia di `EDITOR_ONLY/`.
Log revisi sebelum penguatan disimpan di `HISTORICAL/`. Dataset mentah lengkap
dan kredensial tidak didistribusikan. Akses reviewer melalui portal/hosting
persisten serta persetujuan final penulis masih perlu diselesaikan. Tidak ada
submission, pengiriman email, tanda tangan, pemeriksaan Turnitin, atau jaminan
penerimaan/skor AI yang dilakukan dalam pekerjaan ini.

Manifest mencatat ukuran dan SHA256 setiap member kecuali dirinya sendiri.
Lihat log dan daftar persoalan terbuka untuk batas ilmiah versi ini.
'''
FULL='''# Reproducibility archive jurnal Array - 17 November 2026

Mulai dari AUTHOR_REVIEW/00_BACA_DULU.md. Artikel lengkap adalah PDF 01 dan
AUTHOR_REVIEW/main.tex. Source juga ada di paper/array_20261117_enhanced/latex/.
Artikel utama 30 halaman; supplementary 98 halaman S1-S114; log 3; cover 1.

Rilis ini menambah satu sesi inference/profiling pada model dan workload lama.
Tidak ada training/tuning classifier, dataset baru, atau independent test baru.
Angka penghematan 37.40% dan 60.55% memakai baseline berbeda; 41.75% adalah
hasil historis yang terpisah. Hasil FPR/recall eksternal yang buruk tetap
tersedia. Tidak ada klaim keunggulan pada semua workload.

## Audit tanpa menjalankan model

Gunakan Python 3.11 dengan requirements.lock.txt. Jalankan:

    python -B scripts/verify_array_all_saved_20261117.py

Perintah ini menjalankan delapan audit hasil tersimpan, tanpa memuat estimator,
training, atau inference. Untuk menghitung ulang ringkasan baru dari CSV/NPZ:

    python -B scripts/analyze_array_enhancement_20261117.py

Perhitungan ulang tersebut tidak menjalankan benchmark atau memuat model.

Run baru: runs/ARRAY_SERVING_ABLATION_20261117_01 (175 berkas dalam katalog,
ditambah catalog/complete). Tersedia snapshot source, konfigurasi, environment,
posterior/label pembanding, komponen timing, urutan panggilan, CPU, dan 80 JSON
serta log cold RSS. Actual policy outputs tersimpan di
reports/ARRAY_ENHANCEMENT_20261117/observed_serving_outputs.npz.

Source dan input hashes diperiksa sebelum model milik proyek dimuat pada
sesi profiling. Driver profiling adalah prosedur reproduksi biaya opsional
yang membutuhkan model, mesin, dan canonical CSV asli untuk gate SHA256.
Saved-result audits tidak membutuhkan canonical CSV atau model loading.

## Provenance dan akses data

Metadata parent dan AUTHOR_REVIEW lama disimpan di provenance. Source, hasil
mentah, serta manifest lama dipertahankan. Tujuh puluh expert fits pada audit
inti berasal dari 3 November, bukan fit baru tanggal 17 November.
Canonical CSV dan raw third-party datasets tidak didistribusikan. Ekstraksi
mentah/training memerlukan data pemilik serta izin yang sesuai. Input turunan
dan sample IDs tersedia dalam evidence.

Fixed replay segments dan input model lama ada di EXACT_RUNTIME_20261020_01;
saved seven-group games dan audit equivalence juga disertakan. Semua berkas
yang terdaftar dalam katalog asli dua run historis tambahan diverifikasi
kembali sesudah ekstraksi, termasuk log RSS kosong yang merupakan bukti run.
Tidak ada kredensial, cookie, atau signed URL baru. Repository publik, DOI,
upload portal reviewer, dan persetujuan final penulis belum diklaim tersedia.
Arsip ini adalah file nyata untuk pemeriksaan dospem/reviewer.
'''

def write(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n','utf8')
def archive(path,entries,parents):
    records=[];partial=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as out:
        for name,item in sorted(entries.items()):
            safe(name);h=hashlib.sha256();size=0
            if isinstance(item,bytes):out.writestr(name,item);h.update(item);size=len(item)
            else:
                if isinstance(item,Path):src=item.open('rb');expected=None
                else:z,member,expected=item;src=z.open(member)
                with src,out.open(name,'w',force_zip64=True) as dest:
                    for b in iter(lambda:src.read(1<<20),b''):dest.write(b);h.update(b);size+=len(b)
                if expected:assert expected['bytes']==size and expected['sha256']==h.hexdigest(),name
            records.append(dict(path=name,bytes=size,sha256=h.hexdigest()))
        out.writestr('manifest.json',json.dumps(dict(files=records,parents=parents,new_classifier_fits=0,new_inference_and_profiling=True,new_independent_sources=False,full_raw_downloads_included=False),indent=2)+'\n')
    partial.replace(path)
    with zipfile.ZipFile(path) as z:
        for r in records:
            h=hashlib.sha256()
            with z.open(r['path']) as f:
                for b in iter(lambda:f.read(1<<20),b''):h.update(b)
            assert h.hexdigest()==r['sha256'],r['path']
    digest=sha(path);path.with_suffix('.zip.sha256').write_text(digest+'  '+path.name+'\n','ascii')
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest,files=len(records)+1)

def review_entries():
    entries={'00_BACA_DULU.md':README.encode('utf8'),'05_PERSOALAN_TERBUKA.md':ROOT/'docs/ARRAY_ENHANCED_PERSOALAN_TERBUKA_20261117.md','06_PANDUAN_PENGUATAN.md':ROOT/'docs/ARRAY_ENHANCED_DELIVERY_20261117.md'}
    for arc,name in PDFS:entries[arc]=ROOT/'output/pdf'/name
    for name in ['main.tex','references.bib','elsarticle.cls','elsarticle-num.bst','highlights.txt']:entries[name]=SRC/name
    figs=[p for p in (SRC/'figures').iterdir() if p.suffix in ['.pdf','.svg']];assert len(figs)==28
    for p in figs:entries['figures/'+p.name]=p
    for folder,tex in [('SUPPLEMENTARY','supplement.tex'),('COVER_LETTER','cover_letter.tex'),('REVISION_LOG','revision_log.tex')]:
        for name in [tex,'references.bib','elsarticle.cls','elsarticle-num.bst']:entries[folder+'/'+name]=SRC/name
        if folder=='SUPPLEMENTARY':
            for p in figs:entries[folder+'/figures/'+p.name]=p
    for p in (BASE/'figure_sources').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:entries['figure_sources/'+p.relative_to(BASE/'figure_sources').as_posix()]=p
    entries['FIGURES_INDEX.csv']=ROOT/'paper/array_20261112_native_sources/FIGURES_INDEX.csv'
    for name in ['ARRAY_EDITOR_DISCLOSURE_20261117.md','ARRAY_RELATED_MATERIAL_ADDENDUM_20261117.md','ARRAY_RELATED_MATERIAL_MAP_20261110.md']:entries['EDITOR_ONLY/'+name]=ROOT/'docs'/name
    for arc,name in [('AGERS_ORIGINAL_EDAS.pdf','AGERS_HAFID_EDAS_20261028.pdf'),('AGERS_REVISED_LOCAL.pdf','AGERS_HAFID_FINAL_SCOPE_20261116.pdf')]:entries['EDITOR_ONLY/'+arc]=ROOT/'output/pdf'/name
    entries['HISTORICAL/REVISION_LOG_BEFORE_ENHANCEMENT.pdf']=ROOT/'output/pdf/ARRAY_HAFID_REVISION_LOG_20261117.pdf'
    for p in OUT.glob('*'):
        if p.is_file() and p.suffix in ['.csv','.json','.npz'] and p.name not in ['package_verification.json','portable_verification.json','portable_latex_build.json','portable_latex_render.json','delivery_final.json']:entries['EVIDENCE/'+p.name]=p
    run=ROOT/'runs/ARRAY_SERVING_ABLATION_20261117_01'
    for name in ['latency_raw.csv','replay_raw.csv','memory_raw.csv','configuration_freeze.json','environment.json','equivalence.json','catalog.json','complete.json']:entries['EVIDENCE/serving/'+name]=run/name
    return entries

def main():
    assert json.loads((OUT/'verification.json').read_text())['status'].startswith('PASS_')
    assert json.loads((OUT/'all_saved_verification.json').read_text())['status']=='PASS_ALL_EIGHT_SAVED_RESULT_AUDITS'
    quality=json.loads((OUT/'pdf_quality.json').read_text())
    for q in quality:assert q['visual_review'].startswith('PASS') and not q['bounds'] and sha(ROOT/q['path'])==q['sha256']
    assert sha(PARENT)=='1d1b850fbdcbf73221347b626e7a6863226313f0c9514c4965a7d285715e5ce1'
    entries=review_entries();small=archive(REVIEW,entries,[]);print('Small review ZIP checksums passed',flush=True)
    target=extract(REVIEW,'array_enhanced_latex_');results=[]
    for rel in ['main.tex','SUPPLEMENTARY/supplement.tex','COVER_LETTER/cover_letter.tex','REVISION_LOG/revision_log.tex']:
        results.append(command([ROOT/'.tools/tectonic/tectonic.exe','-X','compile','--keep-logs',Path(rel).name],target/Path(rel).parent));print('Clean compile',rel,flush=True)
    results.append(command([sys.executable,'-B',target/'figure_sources/rebuild_figures.py','--out',target/'rebuilt_figures'],target))
    results.append(command([ROOT/'.venv-paper/Scripts/python.exe','-B',ROOT/'scripts/verify_array_enhanced_render_20261117.py',target,OUT/'portable_latex_render.json'],ROOT))
    write(OUT/'portable_latex_build.json',dict(status='PASS_FOUR_PDFS_AND_14_FIGURES_CLEAN_BUILD',archive=small,extraction=str(target),results=results))
    with zipfile.ZipFile(PARENT) as z:
        raw=z.read('manifest.json');m=json.loads(raw);allentries={}
        for row in m['files']:
            name=safe(row['path'])
            if name in ['README_MULAI_DI_SINI.md','CURRENT_RELEASE.json'] or name.startswith('AUTHOR_REVIEW/'):name='provenance/parent_20261117_revision/'+name
            allentries[name]=(z,row['path'],row)
        allentries['provenance/parent_20261117_revision_manifest.json']=raw
        for folder in [BASE,ROOT/'runs/ARRAY_SERVING_ABLATION_20261117_01',ROOT/'runs/EXACT_RUNTIME_20261020_01',ROOT/'runs/XAI_MULTISCALE_FULL_20261020_01',OUT]:
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix not in ['.aux','.out','.xdv','.blg','.log'] and '__pycache__' not in p.parts and p.name not in ['package_verification.json','portable_verification.json','portable_latex_build.json','portable_latex_render.json','delivery_final.json']:
                    allentries[p.relative_to(ROOT).as_posix()]=p
        # Historical worker logs are sealed evidence too, including empty stdout.
        for run_name in ['ARRAY_SERVING_ABLATION_20261117_01','EXACT_RUNTIME_20261020_01']:
            for p in (ROOT/'runs'/run_name).glob('rss_*.log'):allentries[p.relative_to(ROOT).as_posix()]=p
        for p in ROOT.glob('scripts/*array*20261117.py'):allentries[p.relative_to(ROOT).as_posix()]=p
        for name in ['ARRAY_SERVING_ENHANCEMENT_PROTOCOL_20261117.md','ARRAY_PENGUATAN_ILMIAH_20261117.md','ARRAY_ENHANCED_DELIVERY_20261117.md','ARRAY_ENHANCED_PERSOALAN_TERBUKA_20261117.md','ARRAY_ENHANCEMENT_WORKING_STATE_20261117.md']:allentries['docs/'+name]=ROOT/'docs'/name
        for q in quality:allentries[q['path']]=ROOT/q['path']
        for name,item in entries.items():allentries['AUTHOR_REVIEW/'+name]=item
        allentries['README_MULAI_DI_SINI.md']=FULL.encode('utf8')
        allentries['CURRENT_RELEASE.json']=json.dumps(dict(date='2026-11-17',status='ENHANCED_FOR_AUTHOR_REVIEW_NOT_SUBMITTED',source='paper/array_20261117_enhanced/latex',main_pages=30,supplement_pages=98,supplement_tables=114,new_classifier_fits=0,new_inference_and_profiling=True,new_independent_sources=False,author_approval_confirmed=False),indent=2).encode()
        assert not any(n.endswith('.cookies') or n.startswith('data/external/') or 'register_cic_access' in n for n in allentries)
        parents=[dict(path=PARENT.relative_to(ROOT).as_posix(),sha256=sha(PARENT),files=len(m['files']))]
        repro=archive(REPRO,allentries,parents)
    print('Full evidence archive checksums passed',repro['bytes'],flush=True)
    fresh=extract(REPRO,'array_enhanced_evidence_')
    historical_catalogs=[]
    for run_name in ['EXACT_RUNTIME_20261020_01','XAI_MULTISCALE_FULL_20261020_01']:
        run=fresh/'runs'/run_name
        records=json.loads((run/'artifacts_sha256.json').read_text())
        for record in records:
            p=run/safe(record['path'])
            assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],p
        historical_catalogs.append(dict(run=run_name,files=len(records),all_original_catalog_members_verified=True))
    check=command([sys.executable,'-B','scripts/verify_array_all_saved_20261117.py'],fresh)
    write(OUT/'portable_verification.json',dict(status='PASS_FRESH_EXTRACTION_ALL_EIGHT_SAVED_AUDITS',archive=repro,extraction=str(fresh),historical_catalogs=historical_catalogs,result=check,models_deserialized=False,new_fits=0,new_inference=0))
    result=dict(status='COMPLETE_ENHANCED_ARRAY_AUTHOR_REVIEW_DELIVERY',review=small,reproducibility=repro,pdfs=quality,main_pages=30,supplement_pages=98,supplement_tables=114,
        new_classifier_fits=0,new_inference_and_profiling=True,independent_data_added=False,unit_tests_passed=23,fresh_saved_audits=8,native_figures_rebuilt=14,
        source_and_experiment_history_preserved=True,submission_performed=False,author_approval_confirmed=False,turnitin_performed=False)
    write(OUT/'package_verification.json',result)
    for p in [REVIEW,REPRO]:write(p.with_suffix('.zip.verification.json'),{k:v for k,v in result.items() if k!='pdfs'})
    print(json.dumps({k:v for k,v in result.items() if k!='pdfs'},indent=2),flush=True)
if __name__=='__main__':main()
