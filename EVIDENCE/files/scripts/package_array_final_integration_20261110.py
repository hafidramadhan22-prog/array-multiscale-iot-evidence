"""Stream-merge verified evidence archives; no model deserialization or fitting."""
from pathlib import Path,PurePosixPath
import hashlib,json,os,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ARRAY_FINAL_INTEGRATION_20261110';BASE=ROOT/'paper/array_20261110_integrated'
REVIEW=ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_REVISI_LENGKAP_20261110.zip'
REPRO=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_GUARD_EXTERNAL_20261110.zip'

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def safe(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive member: '+name)
    return name
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

README='''# Paket revisi jurnal Array - 10 November 2026

Kirim tiga PDF kepada dospem:
1. Main manuscript45 halaman,15 tabel,8 gambar,30 referensi.
2. Supplementary76 halaman,S1-S95 lengkap,6 gambar.
3. Audit lima halaman: status komentar dan perbedaan dari9 November.

Naskah dan supplementary telah diperiksa visual; angka direkonstruksi dari
hasil tersimpan. Tidak ada training/inference/tuning atau profiling baru.
Main/supplement menggunakan predictor acuan f untuk kontrak execution/XAI,
sedangkan guarded voting g adalah predictor berbeda yang diperiksa terpisah.

FPR acuan pada Normal V=36/3860 (0,9326%); guard=39/3860 (1,0104%).
Later block3:1/697 (0,1435%), aturan beku sebelum scoring namun source lama.
External guard FPR:31,90% CIC dan18,85% TON Normal terpetakan.
TON Ransomware:1657/2000 diprediksi Normal. Target generalisasi belum tercapai.
Jangan menyebut semua FPR utama di bawah1% atau problem sudah selesai umum.

`docs/ARRAY_FINAL_INTEGRATION_20261110.md` menjelaskan hasil, sumber dan scope.
`paper/array_20261110_integrated/elsarticle/` berisi source editable/highlights.
26 referensi lama tetap;3 paper dataset dan1 resource resmi ditambahkan.

Arsip reproduksi terpisah menggabungkan evidence versi jurnal9 November,
studi FPR dan external test, setelah checksum semua member diverifikasi.
Termasuk tiga notebook studi yang berbeda, bukan satu test gabungan.
Notebook jurnal lama adalah snapshot S1-S65/payload audit; dua notebook lain
melengkapi guard dan eksternal. Main PDF/S1-S95 adalah susunan publikasi terbaru.
Full raw dataset downloads tidak disalin; beberapa subset packet/flow turunan,
target features/indices dan derived labels tetap ada agar katalog run utuh.
Cookie, kredensial dan signed download URLs tidak disertakan.

Sesudah ekstraksi ke pathpendek (mis.C:/TA/ArrayFinal), gunakan environment
terisolasi Python3.11 dengan requirements.lock.txt untuk pemeriksaan:

    python -B scripts/verify_array_final_integration_20261110.py
    python -B scripts/verify_fpr_solution_20261109.py
    python -B scripts/audit_independent_results_20261109.py --phase audit

Perintah tersebut membaca bukti tersimpan, tanpa memuat model atau fit.
Jangan menjalankan trainingdrivers atau phase seal ulang. Prepared-feature
replay tersedia terpisah dan hanya memuat model proyek setelah gate SHA:

    python -B scripts/replay_independent_predictions_20261109.py --output review/predictions.csv --audit review/replay.json

Untuk build PDF, pasang Tectonic dan dependency paper opsional yang dicatat
di requirements.paper-extra.txt. Audit PDF memakai font Arial Windows.
Build sumber final:

    tectonic -X compile --keep-logs paper/array_20261110_integrated/elsarticle/main.tex
    tectonic -X compile --keep-logs paper/array_20261110_integrated/elsarticle/supplement.tex

Draft related-submission disclosure, map overlap dan salinan conference
disediakan terpisah untuk penulis/editor. Tidak ada pesan atau submission.
Final approval semua penulis belum dikonfirmasi. Funding/COI/CRediT mengikuti
jawaban pengguna. Tidak ada repository publik kode/DOI yang dikarang.
Turnitin belum dijalankan; tidak ada klaim AI score<20%. Tulisan dirapikan
untuk kejelasan, dengan bantuanAI diungkap secara faktual.

manifest.json berisi size/SHA seluruh member kecuali dirinya, dan checksum ZIP
ada pada sidecar .sha256. Parent manifests disimpan di provenance/.
'''


def archive(path,entries,parents,overrides):
    partial=path.with_suffix('.zip.tmp');records=[]
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as out:
        for name,item in sorted(entries.items()):
            safe(name);h=hashlib.sha256();size=0
            if isinstance(item,bytes):
                out.writestr(name,item);h.update(item);size=len(item)
            elif isinstance(item,Path):
                with item.open('rb') as src,out.open(name,'w',force_zip64=True) as dest:
                    for b in iter(lambda:src.read(1<<20),b''):dest.write(b);h.update(b);size+=len(b)
            else:
                z,member,expected=item
                with z.open(member) as src,out.open(name,'w',force_zip64=True) as dest:
                    for b in iter(lambda:src.read(1<<20),b''):dest.write(b);h.update(b);size+=len(b)
                assert size==expected['bytes'] and h.hexdigest()==expected['sha256'],name
            records.append(dict(path=name,bytes=size,sha256=h.hexdigest()))
        out.writestr('manifest.json',json.dumps(dict(files=records,parents=parents,overrides=overrides,new_fits=0,new_inference=0,full_raw_downloads_included=False),indent=2,ensure_ascii=False)+'\n')
    partial.replace(path);digest=sha(path);path.with_suffix('.zip.sha256').write_text(digest+'  '+path.name+'\n',encoding='ascii')
    # Verify every emitted member with a streaming read, not only ZIP CRC.
    with zipfile.ZipFile(path) as z:
        for row in records:
            h=hashlib.sha256();size=0
            with z.open(row['path']) as f:
                for b in iter(lambda:f.read(1<<20),b''):h.update(b);size+=len(b)
            assert size==row['bytes'] and h.hexdigest()==row['sha256'],row['path']
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest,files=len(records)+1)


def main():
    assert read(OUT/'verification.json')['status']=='PASS_TABLES_COUNTS_REFERENCES_AND_RECOMPUTED_EVIDENCE'
    quality=read(OUT/'pdf_quality.json')
    for q in quality:assert q['visual_review'].startswith('PASS') and not q['bounds'] and sha(ROOT/q['path'])==q['sha256']
    state=read(ROOT/'CURRENT_STATE.json');p0=state['array_payload_audit_revision_20261109']['reproducibility']
    parents=[(p0['path'],p0['sha256'],'manifest.json'),
             ('penyerahan/SOLUSI_FPR_DI_BAWAH_8PERSEN_UNTUK_DOSPEM_20261109.zip','ad04ace42352c2a2899031b4032222b9109981e2388928b089ae17b4823e0a56','MANIFEST_FPR_SHA256.json'),
             ('penyerahan/UJI_INDEPENDEN_CIC_TON_UNTUK_DOSPEM_20261109.zip','bf49aa2ad5cf682082d9537e9d8d837cf68ed4f0db16f141c779640d93c91127','PACKAGE_MANIFEST.json')]
    handles=[];entries={};previous={};provenance=[];overrides=[]
    try:
        for i,(rel,digest,manifestname) in enumerate(parents,1):
            p=ROOT/rel;assert sha(p)==digest;z=zipfile.ZipFile(p);handles.append(z);mbytes=z.read(manifestname);manifest=json.loads(mbytes);records=manifest if isinstance(manifest,list) else manifest['files']
            entries[f'provenance/parent_{i}_manifest.json']=mbytes;provenance.append(dict(path=rel,sha256=digest,files=len(records)))
            for row in records:
                name=safe(row['path'])
                # Root handoff files belonged to an earlier release; retain as provenance.
                if name in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
                    name=f'provenance/parent_{i}_'+Path(name).name
                if name in previous and previous[name]!=row['sha256']:
                    assert not name.startswith('runs/'),('sealed run conflict',name)
                    overrides.append(dict(path=name,older_sha=previous[name],newer_sha=row['sha256'],source_parent=i))
                entries[name]=(z,row['path'],row);previous[name]=row['sha256']
            print('Parent archive verified',i,p.name,flush=True)
        local=set(p for p in BASE.rglob('*') if p.is_file() and p.suffix not in ['.xdv'])
        local.update(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['package_verification.json','portable_verification.json','delivery_final.json'])
        local.update(ROOT.glob('scripts/*array*20261110.py'))
        local.update(ROOT.glob('docs/ARRAY*20261110.md'))
        local.update([ROOT/'notebooks/jurnal_array_20261109/JURNAL_ARRAY_HASIL_LENGKAP_20261109.ipynb',ROOT/'requirements.lock.txt'])
        for q in quality:local.add(ROOT/q['path'])
        for p in local:
            name=p.relative_to(ROOT).as_posix();assert p.resolve().is_relative_to(ROOT.resolve()) and p.is_file();entries[name]=p
        entries['README_MULAI_DI_SINI.md']=README.encode('utf8')
        entries['requirements.paper-extra.txt']=b'reportlab==5.0.1\nPyMuPDF==2.0\npypdf==6.21.1\nPillow==13.0.0\n'
        entries['CURRENT_RELEASE.json']=json.dumps(dict(date='2026-11-10',status='REVISED_FOR_AUTHOR_REVIEW_NOT_SUBMITTED',source='paper/array_20261110_integrated/elsarticle',main_pages=45,supplement_pages=76,supplement_tables=95,new_fits=0,new_inference=0,external_target8_met=False,author_approval_confirmed=False),indent=2).encode()
        assert not any('register_cic_access' in name or name.endswith('.cookies') or name.startswith('data/external/') for name in entries)
        # Publication authorship emails remain; registration forms and access credentials do not.
        write(OUT/'parent_archives.json',provenance)
        entries['reports/ARRAY_FINAL_INTEGRATION_20261110/parent_archives.json']=OUT/'parent_archives.json'
        repro=archive(REPRO,entries,provenance,overrides);print('Repro archive complete',repro['bytes'],flush=True)
        review={'00_BACA_DULU.md':README.encode('utf8'),'04_CATATAN_REVISI.md':(ROOT/'docs/ARRAY_FINAL_INTEGRATION_20261110.md').read_bytes(),'05_HIGHLIGHTS.txt':(BASE/'elsarticle/highlights.txt').read_bytes()}
        for name,q in zip(['01_NASKAH_UTAMA_ARRAY.pdf','02_SUPPLEMENT_ARRAY_S1_S95.pdf','03_AUDIT_KOMENTAR_DOSPEM.pdf'],quality):review[name]=ROOT/q['path']
        for p in (BASE/'elsarticle').rglob('*'):
            if p.is_file() and (p.suffix in ['.tex','.bib','.cls','.bst','.txt'] or (p.parent.name=='figures' and p.suffix=='.pdf')):review['source/elsarticle/'+p.relative_to(BASE/'elsarticle').as_posix()]=p
        for name in ['ARRAY_EDITOR_DISCLOSURE_20261110.md','ARRAY_RELATED_MATERIAL_MAP_20261110.md']:review['editor_review/'+name]=ROOT/'docs'/name
        review['editor_review/RELATED_MANUSCRIPT_AGERS_EDAS_1571374892.pdf']=ROOT/'output/pdf/AGERS_HAFID_EDAS_20261028.pdf'
        small=archive(REVIEW,review,[],[]);print('Review archive complete',small['bytes'],flush=True)
        # Fresh extraction proves the merged source/result dependencies are present.
        target=Path(tempfile.mkdtemp(prefix='array_final_20261110_',dir=ROOT/'tmp')).resolve();assert target.is_relative_to((ROOT/'tmp').resolve())
        with zipfile.ZipFile(REPRO) as z:
            for n in z.namelist():assert (target/n).resolve().is_relative_to(target)
            z.extractall('\\\\?\\'+str(target) if os.name=='nt' else target)
        results=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8');env.pop('PYTHONPATH',None)
        for args in [['scripts/verify_array_final_integration_20261110.py'],['scripts/verify_fpr_solution_20261109.py'],['scripts/audit_independent_results_20261109.py','--phase','audit']]:
            print('Portable audit',args[0],flush=True)
            p=subprocess.run([sys.executable,'-B',*args],cwd=target,env=env,capture_output=True,text=True,encoding='utf8');results.append(dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,results[-1]
        portable=dict(status='PASS_FRESH_EXTRACTED_REPRODUCIBILITY_AUDITS',extraction=str(target),archive_sha256=repro['sha256'],results=results,new_fits=0,new_inference=0,models_deserialized=False)
        write(OUT/'portable_verification.json',portable)
        result=dict(status='COMPLETE_AUTHOR_REVIEW_PACKAGE',review=small,reproducibility=repro,portable_audit='reports/ARRAY_FINAL_INTEGRATION_20261110/portable_verification.json',pdfs=quality,new_fits=0,new_inference=0,turnitin_performed=False,submission_performed=False,author_approval_confirmed=False)
        write(OUT/'package_verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='pdfs'},indent=2),flush=True)
    finally:
        for z in handles:z.close()


if __name__=='__main__':main()
