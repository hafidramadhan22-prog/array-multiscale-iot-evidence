"""Package native LaTeX and complete retained experiment evidence, then verify clean builds."""
from pathlib import Path,PurePosixPath
from contextlib import ExitStack
import sys,json,zipfile,hashlib,shutil,subprocess,time,tempfile
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'paper/array_20261119_gnb/latex'
BASE=SRC.parent;OUT=ROOT/'reports/ARRAY_GNB_REVISION_20261119'
REVIEW=ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_GNB_LATEX_20261119.zip'
REPRO=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_GNB_REVISION_20261119.zip'
PDFS=[('01_MAIN_MANUSCRIPT_ARRAY_GNB_20261119.pdf','ARRAY_HAFID_GNB_REVISED_20261119.pdf'),
 ('02_SUPPLEMENT_S1_S128_ARRAY_GNB_20261119.pdf','ARRAY_HAFID_SUPPLEMENT_S1_S128_GNB_20261119.pdf'),
 ('03_COVER_LETTER_ARRAY_GNB_20261119.pdf','ARRAY_HAFID_COVER_LETTER_GNB_20261119.pdf'),
 ('04_REVISION_LOG_ARRAY_GNB_20261119.pdf','ARRAY_HAFID_GNB_REVISION_LOG_20261119.pdf')]
PARENTS=[
 ('enhanced','ARRAY_HAFID_REPRODUCIBILITY_ENHANCED_20261117.zip','95fbdeaee0b528a4d5aa9ca955e2407156dc7d30672cb93b0a9d706d431d7efc'),
 ('svm','UJI_SVM_ARRAY_UNTUK_DOSPEM_20261118.zip','71ade414f337109d794f814719706ed29f6672635dd15e32cef360481c670545'),
 ('weight','UJI_BOBOT_G_ARRAY_UNTUK_DOSPEM_20261118.zip','0aca1d661891aa706efba4f9a7f61b087ddfcbe0d1022636a4958f64734bdb0b'),
 ('knn','UJI_KNN_ARRAY_20261119.zip','22c0a9ad5cfea1455fa2c186e7d4a2021c9a95e7cab7126acea46a7f0b4e0c9f'),
 ('gnb','UJI_GNB_ARRAY_20261119.zip','38102985d24360fb05f0653a1c3ba01a7146f9e02a2ddc4144aef7afd898fa00')]
README='''# Paket review jurnal Array — 19 November 2026

ARTIKEL UTAMA adalah 01_MAIN_MANUSCRIPT_ARRAY_GNB_20261119.pdf (31 halaman).
Source artikel lengkap ada di main.tex pada ROOT ZIP, bukan di cover letter.
Supplementary 02:107 halaman,S1–S128,FigureS1–S10; cover03:1 halaman; log04:2.

## Membuka LaTeX

Upload ZIP sebagai proyek baru di Overleaf, pilih Main document=main.tex,
compiler=pdfLaTeX. Untuk dokumen terpisah pilih SUPPLEMENTARY/supplement.tex,
COVER_LETTER/cover_letter.tex atau REVISION_LOG/revision_log.tex. Masing-masing
folder memiliki dependency sendiri. Kembalikan ke main.tex untuk artikel.
Seluruh source ditulis dalam LaTeX native; tidak mengambil halaman dari PDF.
Gambar memiliki PDF/SVG dan kode/data plot; figure_sources/rebuild_figures.py
membangun14gambar lama, rebuild_gnb_figure.py membangun1gambar baru.

## Isi revisi

GNB menjadi pembanding klasifikasi utama; default dan hasil pilihan G sama-sama
dilaporkan. WV, bobot, model, runtime dan SHAP tetap. Hasil pembanding terdahulu
tetap tersedia di supplementary karena memengaruhi kesimpulan. RF tetap lebih
baik pada private history; SVM terpilih sedikit lebih tinggi pada temporal.
Pemilihan pembanding utama setelah melihat hasil lain dinyatakan eksploratif.
Tidak ada klaim WV paling baik secara umum atau generalisasi independen.

## Bukti

EVIDENCE memuat ringkasan dan pemeriksaan integrasi. EXPERIMENT_EVIDENCE memuat
paket GNB tersendiri. Arsip lengkap untuk seluruh hasil tersedia terpisah:
ARRAY_HAFID_REPRODUCIBILITY_GNB_REVISION_20261119.zip. Setelah ekstraksi ke path
pendek dan pemasangan dependencies, gunakan perintah audit saved results:

    python -B scripts/verify_array_all_saved_20261117.py
    python -B scripts/verify_array_svm_20261118.py
    python -B scripts/verify_array_weight_retune_20261118.py
    python -B scripts/verify_array_knn_20261119.py
    python -B scripts/verify_array_gnb_20261119.py
    python -B scripts/verify_array_gnb_revision_20261119.py

Perintah audit tidak melakukan fit atau memuat estimator. Replay/training
memerlukan input asli sesuai hash dan scope masing-masing. Dataset mentah
pihak ketiga tidak dibundel; filejoblib jangan dimuat dari sumber tak dikenal.

EDITOR_ONLY memuat pengungkapan related material AGERS untuk editor.
05_PERSOALAN_TERBUKA menjelaskan batas dan langkah administrasi yang tersisa.
Tidak ada submission, tanda tangan, approval penulis atau Turnitin baru.
'''

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def safe(n):
    p=PurePosixPath(n.replace('\\','/'));assert not p.is_absolute() and '..' not in p.parts
    return p.as_posix()
def contents(item):
    if isinstance(item,bytes):return item
    if isinstance(item,Path):return item.read_bytes()
    return item[0].read(item[1])
def archive(dest,entries,parents):
    rows=[];temp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,item in sorted(entries.items()):
            safe(name);b=contents(item);z.writestr(name,b);rows.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        z.writestr('manifest.json',json.dumps(dict(files=rows,parents=parents,date='2026-11-19',new_fits_in_manuscript_integration=0,completed_GNB_scientific_fits=91,reference_WV_changed=False,prior_comparisons_retained=True),indent=2)+'\n')
    temp.replace(dest)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for r in rows:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    result=dict(path=dest.relative_to(ROOT).as_posix(),bytes=dest.stat().st_size,files=len(rows)+1,sha256=sha(dest))
    dest.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+dest.name+'\n',encoding='ascii')
    return result
def review_entries():
    en={'00_BACA_DULU.md':README.encode('utf8'),'05_PERSOALAN_TERBUKA.md':ROOT/'docs/ARRAY_GNB_PERSOALAN_TERBUKA_20261119.md','06_PANDUAN_REVISI.md':ROOT/'docs/ARRAY_GNB_REVISION_DELIVERY_20261119.md'}
    for arc,name in PDFS:en[arc]=ROOT/'output/pdf'/name
    for name in ['main.tex','references.bib','elsarticle.cls','elsarticle-num.bst','highlights.txt']:en[name]=SRC/name
    figures=[p for p in (SRC/'figures').iterdir() if p.suffix in ['.pdf','.svg']];assert len(figures)==30
    for p in figures:en['figures/'+p.name]=p
    for folder,tex in [('SUPPLEMENTARY','supplement.tex'),('COVER_LETTER','cover_letter.tex'),('REVISION_LOG','revision_log.tex')]:
        for name in [tex,'references.bib','elsarticle.cls','elsarticle-num.bst']:en[folder+'/'+name]=SRC/name
        if folder=='SUPPLEMENTARY':
            for p in figures:en[folder+'/figures/'+p.name]=p
    for p in (BASE/'figure_sources').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:en['figure_sources/'+p.relative_to(BASE/'figure_sources').as_posix()]=p
    for name in ['ARRAY_EDITOR_DISCLOSURE_20261117.md','ARRAY_RELATED_MATERIAL_ADDENDUM_20261117.md','ARRAY_RELATED_MATERIAL_MAP_20261110.md']:en['EDITOR_ONLY/'+name]=ROOT/'docs'/name
    for arc,name in [('AGERS_ORIGINAL_EDAS.pdf','AGERS_HAFID_EDAS_20261028.pdf'),('AGERS_REVISED_LOCAL.pdf','AGERS_HAFID_FINAL_SCOPE_20261116.pdf')]:en['EDITOR_ONLY/'+arc]=ROOT/'output/pdf'/name
    for name in ['gnb_metrics_recomputed.csv','source_revision.json','verification.json','pdf_quality.json']:
        en['EVIDENCE/'+name]=OUT/name
    en['EXPERIMENT_EVIDENCE/UJI_GNB_ARRAY_20261119.zip']=ROOT/'penyerahan/UJI_GNB_ARRAY_20261119.zip'
    return en
def extract(zippath,dest):
    dest.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(zippath) as z:
        for n in z.namelist():safe(n)
        z.extractall(dest)
    return dest
def command(args,cwd):
    start=time.perf_counter();p=subprocess.run([str(a) for a in args],cwd=cwd,capture_output=True,text=True,encoding='utf8',errors='replace')
    assert p.returncode==0,(args,p.stdout[-1800:],p.stderr[-2500:])
    return dict(args=[str(a) for a in args],seconds=time.perf_counter()-start,stdout=p.stdout[-1500:],stderr=p.stderr[-1200:])

def main():
    quality=json.loads((OUT/'pdf_quality.json').read_text());assert all(q['visual_review'].startswith('PASS') for q in quality)
    assert json.loads((OUT/'verification.json').read_text())['status'].startswith('PASS_')
    for q in quality:assert sha(ROOT/q['path'])==q['sha256']
    review=review_entries();small=archive(REVIEW,review,[]);print('REVIEW_ARCHIVE',small,flush=True)
    work=Path(tempfile.mkdtemp(prefix='ag19_tex_'))/'r';fresh=extract(REVIEW,work);builds=[]
    for rel in ['main.tex','SUPPLEMENTARY/supplement.tex','COVER_LETTER/cover_letter.tex','REVISION_LOG/revision_log.tex']:
        builds.append(command([ROOT/'.tools/tectonic/tectonic.exe','-X','compile','--keep-logs',Path(rel).name],fresh/Path(rel).parent));print('CLEAN_BUILD',rel,flush=True)
    builds.append(command([ROOT/'.venv/Scripts/python.exe',fresh/'figure_sources/rebuild_gnb_figure.py','--out',fresh/'rebuilt_gnb'],fresh))
    # PDF text/pixel comparison uses the PDF environment, not a model-loading environment.
    args=[ROOT/'.venv-paper/Scripts/python.exe',ROOT/'scripts/verify_array_gnb_portable_pdf_20261119.py',fresh,OUT/'portable_render.json']
    builds.append(command(args,ROOT));write(OUT/'portable_builds.json',builds)
    with ExitStack() as stack:
        allentries={};parents=[]
        for label,name,digest in PARENTS:
            p=ROOT/'penyerahan'/name;assert sha(p)==digest,(name,sha(p));z=stack.enter_context(zipfile.ZipFile(p));assert z.testzip() is None
            parents.append(dict(label=label,path=p.relative_to(ROOT).as_posix(),sha256=digest))
            for info in z.infolist():
                if info.is_dir():continue
                original=safe(info.filename);new=original
                if original=='manifest.json':new='provenance/'+label+'/manifest.json'
                if label=='enhanced' and (original.startswith('AUTHOR_REVIEW/') or original in ['README_MULAI_DI_SINI.md','CURRENT_RELEASE.json']):new='provenance/parent_enhanced/'+original
                item=(z,original)
                if new in allentries:
                    if contents(allentries[new])==contents(item):continue
                    new='provenance/comparator_collisions/'+label+'/'+new
                allentries[new]=item
        for folder in [BASE,OUT]:
            for p in folder.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.aux','.blg','.log','.out','.xdv'] and p.name not in ['package_verification.json','delivery_final.json']:
                    allentries[p.relative_to(ROOT).as_posix()]=p
        # Compilation logs are intentional evidence for the saved source audit.
        for name in ['main','supplement','cover_letter','revision_log']:
            p=SRC/(name+'.log');allentries[p.relative_to(ROOT).as_posix()]=p
        for p in ROOT.glob('scripts/*gnb*20261119.py'):allentries[p.relative_to(ROOT).as_posix()]=p
        for name in ['ARRAY_GNB_REVISION_DELIVERY_20261119.md','ARRAY_GNB_PERSOALAN_TERBUKA_20261119.md']:
            allentries['docs/'+name]=ROOT/'docs'/name
        for q in quality:allentries[q['path']]=ROOT/q['path']
        for name,item in review.items():allentries['AUTHOR_REVIEW/'+name]=item
        allentries['README_MULAI_DI_SINI.md']=README.encode('utf8')
        allentries['CURRENT_RELEASE.json']=json.dumps(dict(date='2026-11-19',status='GNB_MAIN_COMPARATOR_AUTHOR_REVIEW',main='AUTHOR_REVIEW/main.tex',source='paper/array_20261119_gnb/latex',main_pages=31,supplement_pages=107,supplement_tables=128,author_approval_confirmed=False,submitted=False),indent=2).encode()
        state=json.loads((ROOT/'CURRENT_STATE.json').read_text(encoding='utf-8-sig'))
        state.update(updated_date='2026-11-19',active_work='ARRAY_GNB_MANUSCRIPT_REVISION_20261119',
          active_journal_manuscript='output/pdf/ARRAY_HAFID_GNB_REVISED_20261119.pdf',
          active_journal_supplement='output/pdf/ARRAY_HAFID_SUPPLEMENT_S1_S128_GNB_20261119.pdf',
          active_journal_source='paper/array_20261119_gnb/latex')
        if 'CURRENT_STATE.json' in allentries:allentries['provenance/parent_enhanced/CURRENT_STATE.json']=allentries['CURRENT_STATE.json']
        allentries['CURRENT_STATE.json']=json.dumps(state,indent=2,ensure_ascii=False).encode('utf8')
        for name in ['START_HERE.md','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
            if name in allentries:allentries['provenance/parent_enhanced/'+name]=allentries[name]
            allentries[name]=ROOT/name
        repro=archive(REPRO,allentries,parents);print('FULL_ARCHIVE',repro,flush=True)
    fresh=extract(REPRO,Path(tempfile.mkdtemp(prefix='ag19_repro_'))/'r');checks=[]
    for name in ['verify_array_all_saved_20261117.py','verify_array_svm_20261118.py','verify_array_weight_retune_20261118.py','verify_array_knn_20261119.py','verify_array_gnb_20261119.py','verify_array_gnb_revision_20261119.py']:
        checks.append(command([ROOT/'.venv/Scripts/python.exe','-B',fresh/'scripts'/name],fresh));print('FRESH_SAVED_AUDIT',name,flush=True)
    result=dict(status='PASS_GNB_REVIEW_AND_FULL_ARCHIVE',review=small,reproducibility=repro,clean_builds=4,fresh_saved_audit_commands=checks,all_eight_historical_saved_audits=True,prior_comparisons_retained=True,new_fits_during_integration=0,reference_WV_changed=False,submitted=False,author_approval_confirmed=False)
    write(OUT/'package_verification.json',result)
    print('COMPLETE_GNB_REVISION_PACKAGE',flush=True)

if __name__=='__main__':main()
