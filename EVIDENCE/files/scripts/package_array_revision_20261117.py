"""Package verified text/results; fresh extraction checks do not load estimators."""
from pathlib import Path
import json,sys,zipfile,tempfile,subprocess,os,hashlib,csv
from package_array_final_integration_20261110 import archive,safe,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'paper/array_20261117_revision';SRC=BASE/'latex';OUT=ROOT/'reports/ARRAY_REVISION_20261117'
REVIEW=ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_REVISED_LATEX_20261117.zip'
REPRO=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_REVISED_20261117.zip'
PARENT=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_CONCISE_20261110.zip'
PDFS=[('01_MAIN_MANUSCRIPT_ARRAY_20261117.pdf','ARRAY_HAFID_REVISED_20261117.pdf'),
      ('02_SUPPLEMENT_S1_S104_ARRAY_20261117.pdf','ARRAY_HAFID_SUPPLEMENT_S1_S104_REVISED_20261117.pdf'),
      ('03_COVER_LETTER_ARRAY_20261117.pdf','ARRAY_HAFID_COVER_LETTER_20261117.pdf'),
      ('04_REVISION_LOG_ARRAY_20261117.pdf','ARRAY_HAFID_REVISION_LOG_20261117.pdf')]
README='''# PAKET ARRAY REVISI 17 NOVEMBER 2026 — MULAI DI SINI

ARTIKEL UTAMA ADA: `01_MAIN_MANUSCRIPT_ARRAY_20261117.pdf` (26 halaman).
Source lengkapnya adalah `main.tex` di root ZIP: Abstract sampai References.
Ini source LaTeX native, bukan hasil konversi atau potongan halaman PDF.

Untuk dospem, periksa PDF 01 (main), 02 (supplement S1–S104, 93 halaman),
04 (revision log 4 halaman) dan 05 (persoalan terbuka). PDF 03 adalah cover
letter terpisah satu halaman; tidak menggantikan main text. Satria Mandala
tercantum sebagai corresponding author. Persetujuan final belum diklaim.

OVERLEAF / LATEX:
1. Upload ZIP ini sebagai proyek baru dan pilih compiler pdfLaTeX.
2. Buka Menu / Settings → Main document → `main.tex`, lalu Recompile.
3. Untuk supplement, pilih `SUPPLEMENTARY/supplement.tex` sebagai Main document.
4. Cover terpisah: `COVER_LETTER/cover_letter.tex`.
5. Revision log: `REVISION_LOG/revision_log.tex`.
   Masing-masing subfolder mempunyai dependency sendiri. Sesudah itu kembalikan
   Main document ke `main.tex` untuk membaca artikel utama.
Build lokal yang sudah diuji menggunakan Tectonic:
    tectonic -X compile main.tex
    tectonic -X compile SUPPLEMENTARY/supplement.tex

Folder `figures/` berisi seluruh 14 PDF dan SVG editable; `figure_sources/`
berisi kode dan data plotting, requirements dan provenance. Perintah
`python figure_sources/rebuild_figures.py --out OUTPUT_BARU` membangun ulang
14 gambar dari data turunan tanpa membaca PDF naskah atau memuat model.
`FIGURES_INDEX.csv` memetakan gambar ke nomor pada main/supplement.

INTI REVISI:
- S90: denominator seluruh paket, warm-up tetap masuk denominator.
  Angka 62.9036%/6.6602% dipertahankan karena sesuai formula hasil tersimpan.
- RF private-history 96.9966% > WV 93.7694%; WV dipertahankan sebagai fungsi
  referensi eksekusi, bukan klaim classifier terbaik.
- WV/f berbeda dari GV/g. Saving 41.75% dan audit Shapley WV tidak diwariskan GV.
- Local block3 GV 1/697; reused-V FPR1.0104%; external31.90%/18.85% dan
  TON correct-type recall1.55% tetap dilaporkan. Belum deployment-ready.
- Seluruh 111 badan tabel numerik dipertahankan; narasi/definisi diperbaiki.
- Tidak ada training, tuning, inference, profiling atau dataset baru.

Arsip evidence penuh disediakan terpisah:
`ARRAY_HAFID_REPRODUCIBILITY_REVISED_20261117.zip` (~320 MB).
Arsip tersebut berisi source, split/target IDs, config/seed, cached predictions,
confusion matrices, finite-game arrays, raw runtime/RSS dan verifikasi SHA256.
Raw dataset lengkap milik pihak ketiga dan kredensial tidak didistribusikan.
Sediakan archive tersebut sebagai supplementary reviewer file saat submission.
Belum ada upload portal atau persistent repository/DOI yang diklaim.

`EDITOR_ONLY/` untuk pemeriksaan penulis/editor atas materi AGERS terkait,
termasuk original EDAS dan revisi lokal 16 November. Peta historis 10 November
dibaca bersama addendum17November; revisi conference juga memakai sebagian
external sources yang sama. Dokumen ini bukan persetujuan editor atas overlap.

File ini untuk review penulis/dospem. Tidak ada submission, email, tanda tangan,
final author approval atau Turnitin baru. Tidak ada jaminan skor AI/acceptance.
Manifest JSON memuat SHA256/size setiap member kecuali dirinya sendiri;
checksum ZIP dan log clean-build/fresh-extraction disimpan sebagai sidecar.
'''

FULL_README='''# REPRODUCIBILITY ARRAY — 17 November 2026

Rilis ini untuk pemeriksaan penulis/reviewer, belum disubmit. Mulai dari
`AUTHOR_REVIEW/00_BACA_DULU.md` dan PDF bernomor01 untuk artikel utama.
`AUTHOR_REVIEW/main.tex` adalah source lengkap untuk compile artikel utama.
Source penelitian juga ada di `paper/array_20261117_revision/latex/`.

Rilis ini mempertahankan seluruh evidence parent10November. Folder provenance
menyimpan manifest/readme/current-release induk; readme dan PDF lama yang tetap
ada adalah sejarah, bukan hasil baru17November. 70 expert fits pada core audit
adalah hitungan eksperimen3November, tidak diulang oleh verifikasi ini.

Ekstrak ke path pendek, misalnya C:/TA/Array17. Gunakan Python3.11 dengan
`requirements.lock.txt` pada environment terisolasi. Perintah saved-result:

    python -B scripts/verify_array_revision_20261117.py
    python -B scripts/audit_array_revision_20261117.py core
    python -B scripts/audit_array_revision_20261117.py integration
    python -B scripts/audit_array_revision_20261117.py concise
    python -B scripts/audit_array_revision_20261117.py external
    python -B scripts/audit_array_revision_20261117.py guard
    python -B scripts/audit_array_revision_20261117.py payload

Perintah tersebut membaca saved arrays/CSV/JSON dan checksum. Tidak melakukan
training, tuning, profiling, deserialisasi model atau inference baru. Beberapa
verifier historis memeriksa source/PDF historis mereka; verifier17November
memeriksa main/supplement terbaru dan menghitung ulang finite-game Shapley.
S90 ditelusuri pada `reports/ARRAY_REVISION_20261117/S90_denominator_audit.csv`
dan formula historis audit_independent_results_20261109.py. Canonical CSV asli
tidak disalin ke archive; hash/verifikasi di workspace dicatat. Raw extraction
dan training memerlukan data asli/izin pemilik, bukan data buatan.

Lihat `reports/ARRAY_REVISION_20261117/verification.json`, `pdf_quality.json`,
`finite_game_recomputation.json` dan `saved_evidence_checks/`.
Semua hasil eksternal/GV tetap bersifat diagnostic dengan batas label/source.
Model tersimpan hanya artefak proyek; jangan memuat estimator tidak dikenal.
Full raw third-party downloads, cookies dan signed access URLs tidak disalin.
Tidak ada repository publik/DOI/upload reviewer atau approval baru yang diklaim.
Manifest root berisi SHA256 dan size seluruh member kecuali manifest itu sendiri.
'''

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n','utf8')
def extract(p,prefix):
    target=Path(tempfile.mkdtemp(prefix=prefix,dir=ROOT/'tmp')).resolve()
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():safe(name);assert (target/name).resolve().is_relative_to(target)
        z.extractall('\\\\?\\'+str(target) if os.name=='nt' else target)
    return target
def command(args,cwd):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8');env.pop('PYTHONPATH',None)
    p=subprocess.run([str(x) for x in args],cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8')
    result=dict(command=[str(x) for x in args],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    assert p.returncode==0,result
    return result
def review_entries():
    entries={'00_BACA_DULU.md':README.encode('utf8'),'05_PERSOALAN_TERBUKA.md':ROOT/'docs/ARRAY_PERSOALAN_TERBUKA_20261117.md'}
    for arc,name in PDFS:entries[arc]=ROOT/'output/pdf'/name
    deps=['references.bib','elsarticle.cls','elsarticle-num.bst']
    for name in ['main.tex','highlights.txt',*deps]:entries[name]=SRC/name
    with (ROOT/'paper/array_20261112_native_sources/FIGURES_INDEX.csv').open(encoding='utf-8-sig') as f:
        names={Path(r['PDF']).stem for r in csv.DictReader(f)}
    assert len(names)==14
    figs=[p for p in (SRC/'figures').iterdir() if p.suffix in ['.pdf','.svg'] and p.stem in names]
    assert len(figs)==28
    for p in figs:entries['figures/'+p.name]=p
    for folder,tex in [('SUPPLEMENTARY','supplement.tex'),('COVER_LETTER','cover_letter.tex'),('REVISION_LOG','revision_log.tex')]:
        entries[folder+'/'+tex]=SRC/tex
        for name in deps:entries[folder+'/'+name]=SRC/name
        if folder=='SUPPLEMENTARY':
            for p in figs:entries[folder+'/figures/'+p.name]=p
    for p in (BASE/'figure_sources').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:entries['figure_sources/'+p.relative_to(BASE/'figure_sources').as_posix()]=p
    entries['FIGURES_INDEX.csv']=ROOT/'paper/array_20261112_native_sources/FIGURES_INDEX.csv'
    for name in ['ARRAY_EDITOR_DISCLOSURE_20261117.md','ARRAY_RELATED_MATERIAL_ADDENDUM_20261117.md','ARRAY_RELATED_MATERIAL_MAP_20261110.md']:
        entries['EDITOR_ONLY/'+name]=ROOT/'docs'/name
    entries['EDITOR_ONLY/AGERS_ORIGINAL_EDAS_1571374892.pdf']=ROOT/'output/pdf/AGERS_HAFID_EDAS_20261028.pdf'
    entries['EDITOR_ONLY/AGERS_REVISED_LOCAL_20261116.pdf']=ROOT/'output/pdf/AGERS_HAFID_FINAL_SCOPE_20261116.pdf'
    for name in ['verification.json','pdf_quality.json','denominator_and_comparison_audit.json','S90_denominator_audit.csv','verified_comparison_numbers.json','finite_game_recomputation.json','revision_steps.json','figure_revision.json']:
        entries['AUDIT/'+name]=OUT/name
    return entries

def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS_REVISED_TEXT_AND_RECOMPUTED_SAVED_EVIDENCE'
    q=json.loads((OUT/'pdf_quality.json').read_text())
    for row in q:assert row['visual_review'].startswith('PASS') and not row['bounds'] and sha(ROOT/row['path'])==row['sha256']
    assert sha(PARENT)=='5d826e16439fc14ddf42cd501d0d5c3a7bb4dba9c008797816fd644dca017912'
    review=review_entries();small=archive(REVIEW,review,[],[]);print('Review ZIP built and all member SHA256 verified',flush=True)
    target=extract(REVIEW,'array17_latex_');checks=[]
    for rel in ['main.tex','SUPPLEMENTARY/supplement.tex','COVER_LETTER/cover_letter.tex','REVISION_LOG/revision_log.tex']:
        print('Clean compile',rel,flush=True);checks.append(command([ROOT/'.tools/tectonic/tectonic.exe','-X','compile','--keep-logs',Path(rel).name],target/Path(rel).parent))
    checks.append(command([sys.executable,'-B',target/'figure_sources/rebuild_figures.py','--out',target/'rebuilt_figures'],target))
    checks.append(command([ROOT/'.venv-paper/Scripts/python.exe','-B',ROOT/'scripts/verify_array_portable_render_20261117.py',target,OUT/'portable_latex_render.json'],ROOT))
    write(OUT/'portable_latex_build.json',dict(status='PASS_CLEAN_EXTRACTED_LATEX_AND_FIGURE_BUILD',archive=small,extraction=str(target),results=checks))
    print('Four PDFs and fourteen rebuilt figures match packaged text/pixels',flush=True)
    # Preserve all parent files; relocate only root handoff metadata as provenance.
    with zipfile.ZipFile(PARENT) as z:
        raw=z.read('manifest.json');m=json.loads(raw);entries={};overrides=[]
        for row in m['files']:
            name=safe(row['path'])
            if name in ['README_MULAI_DI_SINI.md','CURRENT_RELEASE.json']:name='provenance/parent_20261110_'+name
            entries[name]=(z,row['path'],row)
        entries['provenance/parent_20261110_manifest.json']=raw
        for p in BASE.rglob('*'):
            if p.is_file() and p.suffix not in ['.xdv','.aux','.out','.log','.blg'] and '__pycache__' not in p.parts:entries[p.relative_to(ROOT).as_posix()]=p
        for p in OUT.rglob('*'):
            if p.is_file() and p.name not in ['package_verification.json','portable_verification.json','delivery_final.json']:
                entries[p.relative_to(ROOT).as_posix()]=p
        for p in list(ROOT.glob('scripts/*array*20261117.py'))+list(ROOT.glob('docs/ARRAY*20261117.md')):
            entries[p.relative_to(ROOT).as_posix()]=p
        for row in q:entries[row['path']]=ROOT/row['path']
        for name,item in review.items():entries['AUTHOR_REVIEW/'+name]=item
        entries['README_MULAI_DI_SINI.md']=FULL_README.encode('utf8')
        entries['CURRENT_RELEASE.json']=json.dumps(dict(date='2026-11-17',status='REVISED_FOR_AUTHOR_REVIEW_NOT_SUBMITTED',source='paper/array_20261117_revision/latex',main_pages=26,supplement_pages=93,supplement_tables=104,new_fits=0,new_inference=0,author_approval_confirmed=False,reviewer_upload_completed=False),indent=2).encode()
        parents=[dict(path=PARENT.relative_to(ROOT).as_posix(),sha256=sha(PARENT),files=len(m['files']))]
        assert not any('register_cic_access' in name or name.endswith('.cookies') or name.startswith('data/external/') for name in entries)
        repro=archive(REPRO,entries,parents,overrides)
    print('Repro ZIP built and all member SHA256 verified',repro['bytes'],flush=True)
    fresh=extract(REPRO,'array17_evidence_');scientific=[]
    for phase in ['core','integration','concise','external','guard','payload']:
        print('Fresh evidence audit',phase,flush=True)
        scientific.append(command([sys.executable,'-B','scripts/audit_array_revision_20261117.py',phase],fresh))
    scientific.append(command([sys.executable,'-B','scripts/verify_array_revision_20261117.py'],fresh))
    portable=dict(status='PASS_FRESH_EXTRACTION_ALL_SEVEN_SAVED_EVIDENCE_AUDITS',archive=repro,extraction=str(fresh),results=scientific,new_fits=0,new_inference=0,models_deserialized=False)
    write(OUT/'portable_verification.json',portable)
    result=dict(status='COMPLETE_REVISED_ARRAY_AUTHOR_REVIEW_DELIVERY',review=small,reproducibility=repro,pdfs=q,
       clean_build='reports/ARRAY_REVISION_20261117/portable_latex_build.json',portable_saved_evidence='reports/ARRAY_REVISION_20261117/portable_verification.json',
       reviewer_access='Actual archive prepared locally; upload/persistent hosting remains an author submission step',
       steps_1_to_9_checked=True,step10='No new experiment for restricted three contributions',new_fits=0,new_inference=0,models_deserialized=False,submission_performed=False,author_approval_confirmed=False,turnitin_performed=False)
    write(OUT/'package_verification.json',result)
    for p in [REVIEW,REPRO]:
        p.with_suffix('.zip.verification.json').write_text(json.dumps(dict(status=result['status'],review=small,reproducibility=repro,clean_build_status='PASS',fresh_saved_evidence_status='PASS',new_fits=0,new_inference=0),indent=2)+'\n','utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='pdfs'},indent=2),flush=True)
if __name__=='__main__':main()
