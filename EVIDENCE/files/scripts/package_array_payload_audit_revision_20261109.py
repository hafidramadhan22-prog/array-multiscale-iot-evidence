"""Merge verified predecessor evidence and deliver three reviewed PDFs."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'reports/ARRAY_PAYLOAD_AUDIT_REVISION_20261109'
BASE=ROOT/'paper/array_20261109_payload_audit'
REVIEW=ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_REVISI_20261109.zip'
REPRO=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_PAYLOAD_AUDIT_20261109.zip'
def read(p):return json.loads(p.read_text('utf8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def archive(p,members):
    assert not p.exists(),'Preserve existing release: '+str(p)
    records=[dict(path=n,bytes=len(b),sha256=sha(b)) for n,b in sorted(members.items())]
    manifest=json.dumps(dict(files=records,new_training=0,new_inference=0,full_raw_dataset_copies=False,original_record_excerpts=True),indent=2).encode('utf8')
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(members.items()):z.writestr(n,b)
        z.writestr('manifest.json',manifest)
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        for r in records:
            b=z.read(r['path']);assert len(b)==r['bytes'] and sha(b)==r['sha256']
    result=dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.read_bytes()),files=len(records)+1)
    p.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+p.name+'\n',encoding='ascii');return result

def main():
    assert read(REP/'verification.json')['status']=='PASS_SOURCES_AND_RECORDED_EVIDENCE'
    quality=read(REP/'pdf_quality.json');state=read(ROOT/'CURRENT_STATE.json')
    for q in quality:
        assert q['visual_review'].startswith('PASS') and not q['bounds'];assert sha((ROOT/q['path']).read_bytes())==q['sha256']
    note='''# Paket revisi jurnal Array - 9 November 2026

Kirimkan tiga PDF bernomor kepada dospem:
1. Naskah utama38halaman,12tabel,7gambar,26referensi.
2. Supplementary58halaman,S1-S73lengkap,6gambar.
3. Penjelasan perbedaan5halaman terhadapmain35/S1-S65versi5 November.

Revisi mengintegrasikan audit6 November. Tidak ada training atau hasil deteksi baru. Angka0/760 sudahada pada versi5 November dan tidak membuktikan keselamatan SQL: RF-only sama, tidak ada evaluated attack-conflict activation, serta hybrid/TF-IDF gagal pada ilustrasi SQL dospem. Semua hasil negatif disertakan. Source/model/bobot/dashboard lama tetap.

`04_CATATAN_REVISI.md` menjelaskan perubahan dan verifikasi; `05_HIGHLIGHTS.txt` sesuai isi baru. Tidak ada hasil/skor Turnitin, klaim penerimaan atau approval baru. Bahasa diedit untuk keterbacaan dan ketepatan, dengan bantuanAI diungkap sesuai penggunaan nyata.

Arsip reproduksi lengkap `ARRAY_HAFID_REPRODUCIBILITY_PAYLOAD_AUDIT_20261109.zip` menggabungkan source/evidence jurnal5 November dan raw-payload/replay6 November setelah verifikasi setiap hash. Termasuk sourcefinal, input/config/split/OOF manifests, owned models lama, prediksi, rawtimings/RSS, partialsource-row exports1.972trace, seluruh48manualprobe outputs dan verifier. Fullraw datasets/PCAP tetap di sumbernya. Notebook sebelumnya tidak ditimpa; source/angka/arraynya tetap, gambar tabel publikasi terbaru ada padaPDF/S1-S73.

Ekstrak ke pathpendek, misalnyaC:/TA/ArrayAudit. Audit tanpa model loading, inference atau training:

    python -B scripts/verify_array_payload_audit_revision_20261109.py
    python -B scripts/verify_array_fpr_update_20261105.py

Build sourcefinal denganTectonic:

    tectonic -X compile --keep-logs paper/array_20261109_payload_audit/elsarticle/main.tex
    tectonic -X compile --keep-logs paper/array_20261109_payload_audit/elsarticle/supplement.tex

`manifest.json` mencatat semua hashmember kecualidirinya; sidecarSHA256 untukZIP. Tidak perlu menjalankan trainingdrivers/joblib untuk membacaPDFatau memeriksa perhitungan. Disclosureeditor/provenance publikasi terkait berada terpisah dan belum dikirim. Identitas author, funding, competing interests danCRediT mengikuti informasi terkonfirmasi; persetujuan submission tidak diasumsikan dari revisi bahasa.
'''
    (BASE/'README.md').write_text(note,encoding='utf8')
    parent_paths=[('penyerahan/ARRAY_HAFID_REPRODUCIBILITY_FPR_UPDATE_20261105.zip',state['array_fpr_update_20261105']['reproducibility_sha256']),('penyerahan/EVIDENCE_PAYLOAD_RAW_DAN_REPLAY_UNTUK_DOSPEM_20261106.zip',state['payload_evidence_20261106']['package_sha256'])]
    members={};parents=[]
    for i,(rel,want) in enumerate(parent_paths):
        p=ROOT/rel;assert sha(p.read_bytes())==want
        with zipfile.ZipFile(p) as z:
            manifest=json.loads(z.read('manifest.json'))
            for r in manifest['files']:
                b=z.read(r['path']);assert len(b)==r['bytes'] and sha(b)==r['sha256']
                if r['path'].startswith('runs/') and r['path'] in members:assert members[r['path']]==b,('sealed run conflict',r['path'])
                members[r['path']]=b
            members[f'provenance/parent_manifest_20261109_{i+1}.json']=z.read('manifest.json')
        parents.append(dict(path=rel,sha256=want));print('VERIFIED_PARENT',p.name,flush=True)
    write(REP/'parent_archives.json',parents)
    disclosure=ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_PAYLOAD_AUDIT_20261109.md'
    disclosure.write_text((ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_FPR_UPDATE_20261105.md').read_text('utf8')+'\n\nThe 9 November manuscript revision incorporates the raw-payload audit completed on 6 November. It adds no classifier fits or independent test set. The failed manual SQL probes and protocol-confounded binary data now limit the interpretation of the retrospective disagreement rule in the main text and Tables S66-S73. Existing related-manuscript provenance/disclosure remains applicable. This disclosure draft has not been sent.\n',encoding='utf8')
    added=set()
    for folder in [BASE,REP]:added.update(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ['.xdv'] and p.name not in ['package_verification.json','portable_verification.json'])
    for pattern in ['scripts/*array*20261109.py','docs/ARRAY*20261109.md']:added.update(ROOT.glob(pattern))
    for p in added:members[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    for q in quality:members[q['path']]=(ROOT/q['path']).read_bytes()
    for n in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:members[n]=(ROOT/n).read_bytes()
    members['README_MULAI_DI_SINI.md']=note.encode('utf8');members['README_CURRENT_PAYLOAD_AUDIT_20261109.md']=note.encode('utf8')
    repro=archive(REPRO,members);print('REPRODUCIBILITY_ARCHIVE_VERIFIED',repro['bytes'],flush=True)
    review_members={'00_BACA_DULU.md':note.encode('utf8'),'04_CATATAN_REVISI.md':(ROOT/'docs/ARRAY_PAYLOAD_AUDIT_REVISION_20261109.md').read_bytes(),'05_HIGHLIGHTS.txt':(BASE/'elsarticle/highlights.txt').read_bytes()}
    for n,q in zip(['01_NASKAH_UTAMA_ARRAY.pdf','02_SUPPLEMENT_ARRAY_S1_S73.pdf','03_PERBANDINGAN_DENGAN_VERSI_SEBELUMNYA.pdf'],quality):review_members[n]=(ROOT/q['path']).read_bytes()
    review=archive(REVIEW,review_members)
    dest=Path(tempfile.mkdtemp(prefix='apa9_',dir=ROOT/'tmp')).resolve();assert (ROOT/'tmp').resolve() in dest.parents
    native='\\\\?\\'+str(dest) if sys.platform=='win32' else str(dest)
    try:
        with zipfile.ZipFile(REPRO) as z:
            assert all(dest in (dest/n).resolve().parents for n in z.namelist())
            z.extractall(native)
        logs=[]
        for filename in ['verify_array_fpr_update_20261105.py','verify_array_payload_audit_revision_20261109.py']:
            run=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-B',str(dest/'scripts'/filename)],cwd=dest,text=True,capture_output=True)
            assert run.returncode==0,run.stdout+run.stderr
            logs.append(dict(script=filename,status='PASS'))
        portable=dict(status='PASS',checks=logs,new_training=0,new_model_loads=0,new_inference=0)
        write(REP/'portable_verification.json',portable)
    finally:
        assert (ROOT/'tmp').resolve() in dest.parents;shutil.rmtree(native)
    result=dict(status='PASS_THREE_PDFS_BOTH_ARCHIVES_AND_PORTABLE_SAVED_EVIDENCE',pdfs=[{k:q[k] for k in ['path','pages','sha256']} for q in quality],review=review,reproducibility=repro,parents=parents,portable_audit='PASS',new_fits=0,new_inference=0,turnitin_performed=False,submission_performed=False,all_authors_approval_confirmed=False)
    write(REP/'package_verification.json',result)
    history=ROOT/'history/BEFORE_ARRAY_PAYLOAD_AUDIT_MANUSCRIPT_20261109';history.mkdir(exist_ok=False)
    for n in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:shutil.copy2(ROOT/n,history/n)
    state=read(ROOT/'CURRENT_STATE.json');state['array_payload_audit_revision_20261109']=result;state['array_payload_audit_revision_20261109'].update(source='paper/array_20261109_payload_audit/elsarticle',main_tables=12,main_figures=7,supplementary_tables=73,supplementary_figures=6,references=26,prior_tables_preserved=True,negative_manual_SQL_probe_integrated=True)
    state['updated_date']='2026-11-09';write(ROOT/'CURRENT_STATE.json',state)
    note_head='**Update jurnal 9 November 2026 - audit payload terintegrasi:** [main38 halaman](output/pdf/ARRAY_HAFID_REVISED_PAYLOAD_AUDIT_20261109.pdf), [supplement58 halaman S1-S73](output/pdf/ARRAY_HAFID_SUPPLEMENT_S1_S73_20261109.pdf), [perbandingan5 halaman](output/pdf/PERBANDINGAN_REVISI_JURNAL_ARRAY_20261109.pdf), [paket dospem](penyerahan/UNTUK_DOSPEM_ARRAY_REVISI_20261109.zip), [reproducibility lengkap](penyerahan/ARRAY_HAFID_REPRODUCIBILITY_PAYLOAD_AUDIT_20261109.zip). Audit6 November kini masuk Abstract/metode/Results/Discussion/Conclusion, Table12/Figure7/S66-S73. Badan tabel main1-11 dan S1-S65 tetap identik. Semua101halaman dirender; angka/references/cross-refs dan portable saved-evidence audits lulus. Tidak ada training, inference, perubahan predictor, Turnitin atau submission baru. Nol760 tetap retrospektif; hybrid/TF-IDF gagal pada ilustrasiSQL dan belum ada actual attack conflict activation. DisclosureAI tetap jujur; tidak mengklaim approval penulis baru. Notebook sebelumnya tetap snapshot hasil+audit; PDFpublikasi terbaru memakaiS1-S73.\n\n'
    for n in ['00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
        p=ROOT/n;p.write_text(note_head+p.read_text('utf8'),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
