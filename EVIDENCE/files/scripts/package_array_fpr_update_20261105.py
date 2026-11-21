"""Deliver new manuscript/supplement and merge both verified evidence releases."""
from pathlib import Path
import sys,json,hashlib,zipfile,shutil,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from package_array_final_20261104 import archive
REP=ROOT/'reports/ARRAY_FPR_UPDATE_20261105';BASE=ROOT/'paper/array_20261105_fpr_update'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    quality=read(REP/'pdf_quality.json');assert read(REP/'verification.json')['status']=='PASS_SAVED_EVIDENCE_AND_MANUSCRIPT_AUDIT'
    outputs=[]
    for row in quality:
        assert row['visual_review'].startswith('PASS') and not row['clipping'];p=ROOT/row['output'];assert sha(p)==row['sha256'];outputs.append(dict(path=row['output'],pages=row['pages'],sha256=row['sha256']))
    repro=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_FPR_UPDATE_20261105.zip';review=ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_FPR_UPDATE_20261105.zip'
    assert not repro.exists() and not review.exists(),'Preserve existing releases; choose a new filename.'
    hist=ROOT/'history/ARRAY_BEFORE_FPR_MANUSCRIPT_UPDATE_20261105'
    if hist.exists():
        assert read(ROOT/'CURRENT_STATE.json')['status']=='ARRAY_FPR_UPDATE_PDFS_COMPLETE_PACKAGING'
        assert all((hist/r).exists() for r in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md'])
    else:
        hist.mkdir(parents=True,exist_ok=False)
        for rel in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:shutil.copy2(ROOT/rel,hist/rel)
    state=read(ROOT/'CURRENT_STATE.json');state['updated_date']='2026-11-05';state['status']='ARRAY_FPR_UPDATE_PDFS_COMPLETE_PACKAGING'
    info=dict(status='PDFS_AND_SAVED_EVIDENCE_COMPLETE_PACKAGING',pdfs=outputs,source='paper/array_20261105_fpr_update/elsarticle',review_package=review.relative_to(ROOT).as_posix(),reproducibility=repro.relative_to(ROOT).as_posix(),revision_notes='docs/ARRAY_FPR_UPDATE_20261105.md',new_scientific_experiments_this_integration=0,new_classifier_fits_this_integration=0,completed_fits_integrated=68,main_tables=11,main_figures=6,supplementary_tables=65,supplementary_figures=6,references=26,historical_numeric_tables_preserved=True,guard_is_changed_predictor=True,original_model_and_dashboard_replaced=False,independent_attack_test=False,attack_guard_activations=0,authors_final_approval_confirmed=False,submission_performed=False,turnitin_completed=False,verification='reports/ARRAY_FPR_UPDATE_20261105/verification.json')
    state['array_fpr_update_20261105']=info;state['active_payload_target_repair']['integrated_journal_revision']='array_fpr_update_20261105'
    state['next']=['Send the two new FPR-update PDFs and the evidence archive to the supervisor for review.','Retain zero attack-conflict support and retrospective scope; do not present guard zero760 as independent safety.','Do not rerun completed training automatically; main reference model and dashboard remain unchanged.','Author approval and any official Turnitin report must be recorded from real evidence before submission.']
    write(ROOT/'CURRENT_STATE.json',state)
    intro='**Update naskah FPR 5 November 2026 (versi jurnal terbaru):** [main35 halaman](output/pdf/ARRAY_HAFID_UPDATED_FPR_20261105.pdf), [supplement51 halaman S1-S65](output/pdf/ARRAY_HAFID_SUPPLEMENT_S1_S65_20261105.pdf), [paket dospem](penyerahan/UNTUK_DOSPEM_ARRAY_FPR_UPDATE_20261105.zip), [reproducibility archive](penyerahan/ARRAY_HAFID_REPRODUCIBILITY_FPR_UPDATE_20261105.zip), [catatan revisi](docs/ARRAY_FPR_UPDATE_20261105.md). Seluruh68 fit payload/binary dan rule tanpa fit sudah diintegrasikan sebagai hasil eksploratif, termasuk hasil negatif. Main11 tabel/6gambar/26referensi; S1-S45 values tetap, S46-S65 baru. Guard0/760 sama dengan RF-only, seluruhV identik, tetapi belum ada serangan yang memicu guard. Klaim compact/runtime/XAI tetap hanya untuk predictor acuan. S8 N diperjelas Scored N; Training tetap perlakuan. Model/dashboard tidak diganti; tidak ada training atau Turnitin baru, tidak ada submission. Versi di bawah adalah sejarah.\n\n'
    for rel in ['00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
        p=ROOT/rel;t=p.read_text(encoding='utf8')
        if not t.startswith(intro):p.write_text(intro+t,encoding='utf8')
    disclosure=ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_FPR_UPDATE_20261105.md'
    disclosure.write_text((ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_20261105.md').read_text(encoding='utf8')+'\n\nThe later 5 November FPR update adds matched payload interventions, a content-disjoint but protocol-confounded binary pilot, and a retrospective RF-disagreement rule. These analyses are not reported in the AGERS manuscript. The rule changes the predictor; its zero observed benign alarms are not an independent final test or evidence of safety on attack conflicts. The original execution/XAI claims retain their original predictor scope. The updated main and S1-S65 supplement preserve the historical provenance and disclosure. This draft has not been sent.\n',encoding='utf8')
    guide='''# Versi jurnal terbaru untuk dospem - FPR update, 5 November 2026

1. `01_NASKAH_UTAMA_ARRAY.pdf`:35 halaman,11 tabel,6 gambar,26 referensi.
2. `02_SUPPLEMENT_ARRAY_S1_S65.pdf`:51 halaman; S1-S45 values tetap; S46-S65 hasil payload/guard lengkap.
3. `03_CATATAN_REVISI.md`: perubahan, hasil negatif, cakupan dan batas bukti.
4. `04_HIGHLIGHTS.txt`: highlights yang sesuai dengan hasil terbaru.

File root pada reproducibility archive memakai nama `output/pdf/ARRAY_HAFID_UPDATED_FPR_20261105.pdf` dan `output/pdf/ARRAY_HAFID_SUPPLEMENT_S1_S65_20261105.pdf`. Source final berada di `paper/array_20261105_fpr_update/elsarticle`. Versi editorial/major sebelumnya adalah sejarah.

## Arti hasil FPR

Pada760 additional Normal yang sama: reference321false alarms(42,24%); direct full-A refit syntax310, TF-IDF312, hybrid331. Rule konflik RF/WV dengan payload check0/760, sama dengan RF-only. V10.358 labels identik, termasuk dua missed attacks. G/C/V tidak memiliki eligible conflict: belum ada serangan yang menguji aktivasi rule. Ini retrospektif/within-trace, bukan jaminan recall atau independent final test. Binary200 content-disjoint scores tinggi tetapi protocol/source-confounded; model binary melewatkan banyak SQL saat content kosong dan tidak dipakai sebagai general detector.

Rule baru mengubah predictor. Klaim preservation/SHAP dan saving replay41,75% tetap milik predictor acuan. Native hybrid-rule profile justru lebih lambat pada batch tambahan. Tidak ada training, inference atau benchmark baru yang diulang untuk mengintegrasikan naskah ini.

## Arsip reproduksi lengkap

`ARRAY_HAFID_REPRODUCIBILITY_FPR_UPDATE_20261105.zip` menggabungkan evidence jurnal sebelumnya dan seluruh evidence payload/guard. Source, configurations, hashes, split/OOF IDs, owned fitted models, posterior/label arrays, semua256parent+48correction+116guard metric rows, raw timings/RSS, contoh content dan prototype tersedia. Raw third-party CSV/PCAP tidak diduplikasi dan tetap mengikuti sumber dataset. Tidak mengklaim repository kode publik.

Ekstrak ke path pendek, misalnya `C:/TA/ArrayFPR`. Dengan Python3.11 dan environment sesuai requirements.lock.txt, audit artikel tanpa loading model/training/inference:

    python -B scripts/verify_array_fpr_update_20261105.py

Audit eksperimen asli yang memverifikasi fitted artifacts dimiliki proyek tersedia di `scripts/verify_sql_fpr_repair_20261105.py`; baca dokumentasi sebelum memuat model. Jangan mengganti joblib dengan sumber lain. Untuk review saja, jangan menjalankan driver training.

Build dari source final:

    tectonic -X compile --keep-logs paper/array_20261105_fpr_update/elsarticle/main.tex
    tectonic -X compile --keep-logs paper/array_20261105_fpr_update/elsarticle/supplement.tex

`manifest.json` berisi hash setiap member selain dirinya; sidecar `.zip.sha256` memeriksa ZIP. Evidence run mempunyai catalog sendiri. Arsip induk dan patch payload yang dipakai sudah diverifikasi sebelum penggabungan. Snapshot CURRENT_STATE mencatat tahap pengemasan; laporan package_verification.json pada proyek asal mencatat hasil akhir pemeriksaan ZIP/portable audit.

Naskah untuk review dospem, belum disubmit. Tidak ada klaim persetujuan baru seluruh penulis atau hasil Turnitin. Editor disclosure/provenance terkait ditempatkan terpisah. Model reference dan dashboard lama tidak diganti oleh revisi tulisan.
'''
    (BASE/'README.md').write_text(guide,encoding='utf8')
    # Verify every predecessor member before reusing it. Merge derived evidence,
    # not nested duplicate ZIP files or raw dataset files.
    members={};parents=[]
    prev=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_20261105.zip'
    with zipfile.ZipFile(prev) as z:
        rec=read(ROOT/'reports/ARRAY_EDITORIAL_20261105/package_verification.json')['reproducibility'];assert sha(prev)==rec['sha256']
        for r in json.loads(z.read('manifest.json'))['files']:
            b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];members[r['path']]=b
    parents.append(dict(path=prev.relative_to(ROOT).as_posix(),sha256=sha(prev)))
    patch=ROOT/'penyerahan/PERBAIKAN_FPR_NORMAL_SQL_UNTUK_DOSPEM_20261105.zip';wanted=read(ROOT/'reports/SQL_FPR_REPAIR_20261105/package_verification.json');assert sha(patch)==wanted['sha256']
    with zipfile.ZipFile(patch) as z:
        manifest=json.loads(z.read('PACKAGE_MANIFEST_SHA256.json'))
        for r in manifest:
            b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];members[r['path']]=b
        members['provenance/PAYLOAD_GUARD_PACKAGE_MANIFEST_20261105.json']=z.read('PACKAGE_MANIFEST_SHA256.json')
    parents.append(dict(path=patch.relative_to(ROOT).as_posix(),sha256=sha(patch)));write(REP/'parent_archives.json',parents)
    added=set()
    for rel in ['paper/array_20261105_fpr_update','reports/ARRAY_FPR_UPDATE_20261105']:
        added.update(p for p in (ROOT/rel).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['package_verification.json','portable_verification.json'])
    for pattern in ['scripts/*array_fpr*20261105.py','docs/ARRAY*FPR*20261105.md']:
        added.update(ROOT.glob(pattern))
    added.update(ROOT/r for r in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md'])
    added.update(ROOT/r['path'] for r in outputs)
    for p in added:members[p.relative_to(ROOT).as_posix()]=p
    members['README_CURRENT_FPR_UPDATE_20261105.md']=BASE/'README.md';members['README_MULAI_DI_SINI.md']=BASE/'README.md'
    # Source files used by each sealed run remain byte-identical to its manifest.
    repro_info=archive(repro,members);print('REPRO_ZIP_VERIFIED',repro_info['bytes'],'bytes',flush=True)
    review_info=archive(review,{'00_BACA_DULU.md':BASE/'README.md','01_NASKAH_UTAMA_ARRAY.pdf':ROOT/outputs[0]['path'],'02_SUPPLEMENT_ARRAY_S1_S65.pdf':ROOT/outputs[1]['path'],'03_CATATAN_REVISI.md':ROOT/'docs/ARRAY_FPR_UPDATE_20261105.md','04_HIGHLIGHTS.txt':BASE/'elsarticle/highlights.txt','EDITOR_DISCLOSURE/related_manuscript_disclosure.md':disclosure,'EDITOR_DISCLOSURE/AGERS_EDAS_1571374892.pdf':ROOT/'output/pdf/AGERS_HAFID_EDAS_20261028.pdf'})
    dest=Path(tempfile.mkdtemp(prefix='afpr_',dir=ROOT/'tmp')).resolve()
    assert (ROOT/'tmp').resolve() in dest.parents
    # Windows extended absolute paths avoid MAX_PATH for retained historical
    # evidence. The resolved deletion target was checked against ROOT/tmp.
    native_dest='\\\\?\\'+str(dest) if sys.platform=='win32' else str(dest)
    try:
        with zipfile.ZipFile(repro) as z:
            for name in z.namelist():assert dest in (dest/name).resolve().parents
            z.extractall(native_dest)
        env=str((ROOT/'.venv/Scripts/python.exe').resolve())
        run=subprocess.run([env,'-B',str(dest/'scripts/verify_array_fpr_update_20261105.py')],cwd=dest,capture_output=True,text=True)
        if run.returncode:raise RuntimeError(run.stdout+run.stderr)
        portable=read(dest/'reports/ARRAY_FPR_UPDATE_20261105/verification.json');assert portable['status']=='PASS_SAVED_EVIDENCE_AND_MANUSCRIPT_AUDIT'
        write(REP/'portable_verification.json',dict(status='PASS',checks=portable['checks'],new_classifier_fits=0,new_inference=0,new_model_loads=0))
    finally:
        assert (ROOT/'tmp').resolve() in dest.parents
        shutil.rmtree(native_dest)
    result=dict(status='PASS_BOTH_ZIPS_AND_PORTABLE_ARTICLE_AUDIT',pdfs=outputs,review=review_info,reproducibility=repro_info,parents=parents,portable_article_audit='PASS',raw_dataset_redistributed=False,new_classifier_fits_this_revision=0,submission_performed=False)
    write(REP/'package_verification.json',result)
    state=read(ROOT/'CURRENT_STATE.json');state['status']='ARRAY_FPR_UPDATE_COMPLETE_FOR_SUPERVISOR_REVIEW';info=state['array_fpr_update_20261105'];info.update(status='COMPLETE_VERIFIED_NOT_SUBMITTED',review_package_sha256=review_info['sha256'],reproducibility_sha256=repro_info['sha256'],portable_saved_evidence_audit='PASS',main_pages=35,supplement_pages=51);write(ROOT/'CURRENT_STATE.json',state)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
