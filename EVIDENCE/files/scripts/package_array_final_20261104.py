"""Deliver supervisor review and evidence archives; preserve earlier releases."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];REP=ROOT/'reports/ARRAY_FINAL_REVIEW_20261104'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def archive(path,members):
    entries=[]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,value in sorted(members.items()):
            data=value.read_bytes() if isinstance(value,Path) else value
            z.writestr(name,data);entries.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('manifest.json',json.dumps(dict(files=entries,raw_csv_pcap_redistributed=False),indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for r in entries:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    info=dict(path=path.relative_to(ROOT).as_posix(),files=len(entries),bytes=path.stat().st_size,sha256=sha(path),status='PASS_ALL_MEMBER_SHA256_AND_CRC')
    path.with_suffix('.zip.sha256').write_text(info['sha256']+'  '+path.name+'\n',encoding='ascii')
    return info

def main():
    quality=json.loads((REP/'pdf_quality.json').read_text());base=ROOT/'paper/array_20261104_final_review'
    names={'main':'ARRAY_HAFID_FINAL_REVIEW_20261104.pdf','supplement':'ARRAY_HAFID_SUPPLEMENT_FINAL_REVIEW_20261104.pdf','response':'ARRAY_HAFID_RESPONSE_FINAL_REVIEW_20261104.pdf'};outputs=[]
    for r in quality:
        rel=('response/response' if r['document']=='response' else 'elsarticle/'+r['document'])+'.pdf';src=base/rel;assert sha(src)==r['sha256'] and not r['clipping']
        r['visual_review']='PASS: every rendered page reviewed; tables, figures, references and captions readable'
        dst=ROOT/'output/pdf'/names[r['document']];shutil.copy2(src,dst);outputs.append(dict(path=dst.relative_to(ROOT).as_posix(),sha256=sha(dst),pages=r['pages']))
    write(REP/'pdf_quality.json',quality)
    backup=ROOT/'history/ARRAY_BEFORE_FINAL_REVIEW_20261104';backup.mkdir(parents=True,exist_ok=True)
    for rel in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
        if not (backup/rel).exists():shutil.copy2(ROOT/rel,backup/rel)
    s=json.loads((ROOT/'CURRENT_STATE.json').read_text(encoding='utf-8-sig'));s['updated_date']='2026-11-04';s['status']='ARRAY_FINAL_REVIEW_COMPLETE_AUTHOR_APPROVAL_PENDING'
    s['array_final_review_20261104']=dict(status='COMPLETE_FOR_SUPERVISOR_FINAL_REVIEW_NOT_SUBMITTED',deliverables=outputs,review_package='penyerahan/UNTUK_DOSPEM_ARRAY_20261104.zip',reproducibility='penyerahan/ARRAY_HAFID_REPRODUCIBILITY_20261104.zip',source='paper/array_20261104_final_review',response='docs/ARRAY_FINAL_REVIEW_20261104.md',related_agers='output/pdf/AGERS_HAFID_EDAS_20261028.pdf',new_fits=0,additional_diagnostic='Frozen-model semantic flag alignment on the same 760 targets; historical predictions retained',weighted_false_alarms_before=324,weighted_false_alarms_aligned=321,unscorable=107,independent_five_class_test=False,supplementary_tables=45,references=26,authors_final_approval=False,public_code_repository_found=False,availability='Code and derived evidence in accompanying archive; original dataset provider linked',editor_disclosure='Draft prepared, not sent',submission_performed=False)
    s['next']=['User sends 4-Nov review package to supervisor','Record actual approval by both authors of the exact final PDF hashes','Confirm AGERS status and submit related-manuscript disclosure for editorial assessment','Attach reproducibility archive and verify current Array portal requirements before submission','Do not rerun completed training or benchmarks automatically']
    write(ROOT/'CURRENT_STATE.json',s)
    intro='**Update 4 November 2026:** [paket pemeriksaan terakhir dospem](penyerahan/UNTUK_DOSPEM_ARRAY_20261104.zip), [tanggapan enam poin](docs/ARRAY_FINAL_REVIEW_20261104.md). Main 29 halaman, supplement 35 halaman S1–S45, response 3 halaman. Ada mismatch hex/desimal pada tiga flag additional benign; koreksi pada frozen model menurunkan weighted false alarms 324/760 menjadi 321/760, seluruh error SQL injection. Tidak ada training/runtime/XAI yang diulang; hanya targeted format diagnostic. AGERS EDAS PDF disertakan dan overlap historis diungkap. Repository kode publik milik Hafid tidak ditemukan; code/evidence disediakan sebagai archive pendamping. Final author approval belum diberikan. Versi sebelumnya tetap sejarah.\n\n'
    for rel in ['00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
        p=ROOT/rel;t=p.read_text(encoding='utf8')
        if not t.startswith('**Update 4 November 2026:**'):p.write_text(intro+t,encoding='utf8')
    readme=base/'README.md'
    readme.write_text('''# Paket pemeriksaan terakhir dospem — Array, 4 November 2026

File yang dikirim ke dospem ada dalam UNTUK_DOSPEM_ARRAY_20261104.zip:
1. 01_NASKAH_UTAMA_ARRAY.pdf — 29 halaman.
2. 02_SUPPLEMENT_ARRAY_S1_S45.pdf — 35 halaman; S1–S40 dipertahankan, S41–S45 diagnosis baru.
3. 03_TANGGAPAN_ENAM_REVISI.pdf — 3 halaman.
4. 04_ARTIKEL_AGERS_EDAS_1571374892.pdf — file enam halaman lokal EDAS 28 Oktober, tidak diubah.
5. Draft disclosure editor, form approval dan catatan penggunaan.

Naskah ini untuk pemeriksaan akhir, belum disubmit. Author approval belum diberikan. Overlap historis dengan AGERS masih ada dan ditandai; penilaian editor belum diperoleh. Ada 324/760 false alarm sebelum dan 321/760 setelah normalisasi format flag; hasil tidak mendukung independent generalization.

## Reproducibility archive pendamping

ARRAY_HAFID_REPRODUCIBILITY_20261104.zip berisi source, configurations, split IDs, saved predictions, hasil turunan, checksums, source PDF, serta artefak November dan bukti Oktober terpilih yang sudah ada pada paket 3 November. Raw CSV dan PCAP pihak ketiga tidak disertakan. Dataset publik dirujuk ke sumber aslinya; file lokal yang dipakai memiliki hash dalam konfigurasi. Tidak ada repository kode publik milik Hafid yang ditemukan dalam audit folder/metadata.

Ekstrak ke path pendek di Windows, misalnya C:/TA/Array. Jalankan audit hasil tersimpan dengan environment yang sesuai requirements.lock.txt:

    python -B scripts/verify_array_final_20261104.py

Perintah ini tidak memuat model atau melakukan inferensi/training. Hasil utama 3 November dapat diaudit terpisah dengan scripts/verify_array_revision_20261103.py. Rekonstruksi dari raw PCAP memerlukan raw source asli; package ini tidak diklaim sebagai full raw-data mirror. scripts/audit_array_flag_format_20261104.py melakukan model inference dan sengaja menolak overwrite run yang sudah ada; jangan menjalankannya untuk sekadar membaca hasil.

Build dari source final:

    tectonic -X compile paper/array_20261104_final_review/elsarticle/main.tex
    tectonic -X compile paper/array_20261104_final_review/elsarticle/supplement.tex
    tectonic -X compile paper/array_20261104_final_review/response/response.tex

Manifest memeriksa setiap member ZIP; sidecar .sha256 memeriksa ZIP. Archive tidak memberi izin redistribusi baru terhadap data pihak ketiga atau menyatakan approval penulis.
''',encoding='utf8')
    approval=ROOT/'docs/ARRAY_AUTHOR_APPROVAL_20261104.md';t=approval.read_text(encoding='utf8');marker='\n## Hash PDF yang perlu disetujui\n'
    if marker in t:t=t[:t.index(marker)]
    approval.write_text(t+marker+'\n'+'\n'.join('- '+r['path']+' — SHA256 '+r['sha256'] for r in outputs[:2])+'\n',encoding='utf8')
    # Extend the previously verified evidence release; do not overwrite its ZIP.
    previous=ROOT/'penyerahan/ARRAY_HAFID_MAJOR_REVISION_20261103.zip';members={}
    with zipfile.ZipFile(previous) as z:
        manifest=json.loads(z.read('manifest.json'))
        for r in manifest['files']:
            data=z.read(r['path']);assert hashlib.sha256(data).hexdigest()==r['sha256'];members[r['path']]=data
    additions=set()
    for rel in ['paper/array_20261104_final_review','reports/ARRAY_FINAL_REVIEW_20261104','runs/ARRAY_FLAG_FORMAT_20261104_01']:
        additions.update(p for p in (ROOT/rel).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    additions.update(ROOT.glob('scripts/*20261104.py'));additions.update(ROOT.glob('docs/*20261104.md'))
    additions.update(ROOT/r['path'] for r in outputs)
    for rel in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md','output/pdf/AGERS_HAFID_EDAS_20261028.pdf','runs/MULTISCALE_REPAIR_FULL_20261020_01/artifacts_sha256.json','runs/MULTISCALE_REPAIR_FULL_20261020_01/stages/temporal_seed42/split.npz','runs/MULTISCALE_REPAIR_FULL_20261020_01/stages/temporal_seed42/legacy25_multiscale60__balanced/model.joblib','runs/MULTISCALE_REPAIR_FULL_20261020_01/stages/temporal_seed42/legacy25_multiscale60__balanced/selection_before_V.json']:
        additions.add(ROOT/rel)
    additions.discard(REP/'package_verification.json')
    for p in additions:members[p.relative_to(ROOT).as_posix()]=p
    members['README_FINAL_20261104.md']=readme
    repro=archive(ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_20261104.zip',members)
    review={'01_NASKAH_UTAMA_ARRAY.pdf':ROOT/outputs[0]['path'],'02_SUPPLEMENT_ARRAY_S1_S45.pdf':ROOT/outputs[1]['path'],'03_TANGGAPAN_ENAM_REVISI.pdf':ROOT/outputs[2]['path'],'04_ARTIKEL_AGERS_EDAS_1571374892.pdf':ROOT/'output/pdf/AGERS_HAFID_EDAS_20261028.pdf','00_BACA_DULU.md':readme,'05_DRAFT_DISCLOSURE_EDITOR.md':ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_20261104.md','06_FORM_PERSETUJUAN_PENULIS.md':approval}
    review_info=archive(ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_20261104.zip',review)
    result=dict(status='PASS_BOTH_ARCHIVES_ALL_HASHES_AND_CRC',pdfs=outputs,reproducibility=repro,supervisor_review=review_info,visually_reviewed_revised_pages=sum(r['pages'] for r in quality),no_submission=True)
    write(REP/'package_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
