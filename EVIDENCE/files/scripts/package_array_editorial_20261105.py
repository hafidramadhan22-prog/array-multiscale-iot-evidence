"""Package the editorial PDFs and complete saved reproducibility evidence."""
from pathlib import Path
import json,hashlib,shutil,zipfile
from package_array_final_20261104 import archive
ROOT=Path(__file__).resolve().parents[1];REP=ROOT/'reports/ARRAY_EDITORIAL_20261105'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    quality=json.loads((REP/'pdf_quality.json').read_text());outputs=[]
    names={'main':'ARRAY_HAFID_REVISED_20261105.pdf','supplement':'ARRAY_HAFID_SUPPLEMENT_S1_S45_20261105.pdf'}
    for r in quality:
        p=ROOT/'paper/array_20261105_editorial/elsarticle'/(r['document']+'.pdf');assert sha(p)==r['sha256'] and not r['clipping']
        r['visual_review']='PASS: all main pages and changed supplementary pages visually reviewed; supplementary pages 2-34 pixel-identical to previously reviewed version'
        out=ROOT/'output/pdf'/names[r['document']];shutil.copy2(p,out);outputs.append(dict(path=out.relative_to(ROOT).as_posix(),pages=r['pages'],sha256=sha(out)))
    write(REP/'pdf_quality.json',quality)
    backup=ROOT/'history/ARRAY_BEFORE_EDITORIAL_20261105';backup.mkdir(parents=True,exist_ok=True)
    for rel in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
        if not (backup/rel).exists():shutil.copy2(ROOT/rel,backup/rel)
    state=json.loads((ROOT/'CURRENT_STATE.json').read_text(encoding='utf-8-sig'));state['updated_date']='2026-11-05';state['status']='ARRAY_EDITORIAL_REVISION_COMPLETE_TURNITIN_NOT_RUN'
    state['array_editorial_20261105']=dict(status='COMPLETE_LOCAL_FILES_NOT_SUBMITTED',pdfs=outputs,source='paper/array_20261105_editorial/elsarticle',review_package='penyerahan/UNTUK_DOSPEM_ARRAY_20261105.zip',reproducibility='penyerahan/ARRAY_HAFID_REPRODUCIBILITY_20261105.zip',main_agers_narrative_moved_to_editor_disclosure=True,editor_disclosure='docs/ARRAY_EDITOR_DISCLOSURE_20261105.md',overlap_provenance_retained=True,pending_wording_removed_by_user_request=True,approval_claim_added=False,all_authors_approval_confirmed_in_session=False,supplementary_tables=45,references=25,new_training=False,new_inference=False,turnitin=dict(status='NOT_RUN',reason='Browser access host unavailable; native Chrome interaction stopped because current URL could not be verified',document_uploaded=False,report_id=None,similarity_percent=None,ai_percent=None,target_below_20_confirmed=False),submission=False)
    state['next']=['User or campus staff checks latest 5-Nov main PDF using an authorized Turnitin account','Read authentic Similarity and AI Writing reports separately; no score is available yet','Confirm final author agreement and retain related-manuscript editor disclosure before submission','Do not rerun completed experiments automatically']
    write(ROOT/'CURRENT_STATE.json',state)
    intro='**Update 5 November 2026:** [paket dospem terbaru](penyerahan/UNTUK_DOSPEM_ARRAY_20261105.zip), [arsip reproduksi lengkap](penyerahan/ARRAY_HAFID_REPRODUCIBILITY_20261105.zip), [catatan editorial](docs/ARRAY_EDITORIAL_20261105.md). Frasa final author approval pending dihapus dari main atas permintaan pengguna, tanpa klaim approval baru. Narasi AGERS dipindahkan ke disclosure editor terpisah; provenance overlap historis tetap ada. Semua tabel/angka/gambar tetap. Supplement lengkap S1–S44 ditambah S45 runtime. Turnitin BELUM dijalankan: akses browser dihentikan karena URL tidak terverifikasi; tidak ada skor AI/similarity, upload atau bukti <20%. Tidak mengulang eksperimen. Versi di bawah adalah sejarah.\n\n'
    for rel in ['00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md']:
        p=ROOT/rel;t=p.read_text(encoding='utf8')
        if not t.startswith('**Update 5 November 2026:**'):p.write_text(intro+t,encoding='utf8')
    readme=ROOT/'paper/array_20261105_editorial/README.md'
    readme.write_text('''# Paket Array — revisi editorial 5 November 2026

Untuk dospem: 01_NASKAH_UTAMA_ARRAY.pdf (29 halaman), 02_SUPPLEMENT_ARRAY_S1_S45.pdf (35 halaman). S1–S44 lengkap; S45 menyimpan 30 pasangan runtime. Frasa approval pending dihapus, tanpa membuat klaim approval sudah diberikan. Tabel, persamaan, hasil dan gambar tetap. Deklarasi bantuan AI tetap tersedia.

Narasi AGERS tidak berada di main manuscript. Disclosure editor dan artikel terkait disimpan dalam folder EDITOR_DISCLOSURE; provenance hasil historis tetap ditandai dalam supplementary. Penghapusan narasi utama tidak menyatakan bahwa overlap hilang.

Turnitin belum dijalankan: browser access host tidak tersedia dan kontrol Chrome dihentikan saat URL tidak dapat diverifikasi. Tidak ada upload, submission ID, skor similarity atau skor AI. Gunakan akun kampus untuk pemeriksaan resmi; panduan ringkas disertakan. Tidak ada jaminan AI <20% dan tidak ada perubahan tulisan untuk mengakali detektor.

Reproducibility archive menyediakan source/config/environment/split IDs/saved predictions/derived evidence beserta checksum dari versi sebelumnya dan tambahan editorial ini. Raw CSV/PCAP tidak disertakan; sumber dan hash tetap dicatat. Arsip bukan mirror lengkap raw dataset. Tidak ada eksperimen baru.

Ekstrak ke path pendek di Windows, misalnya C:/TA/Array. Audit editorial tanpa model/inference:

    python -B scripts/verify_array_editorial_20261105.py

Build source dengan Tectonic:

    tectonic -X compile paper/array_20261105_editorial/elsarticle/main.tex
    tectonic -X compile paper/array_20261105_editorial/elsarticle/supplement.tex

manifest.json memuat SHA256 seluruh member selain manifest itu sendiri; sidecar .sha256 memeriksa ZIP. PDF/source sebelumnya tetap disimpan sebagai sejarah, bukan versi aktif.
''',encoding='utf8')
    members={};prev=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_20261104.zip'
    with zipfile.ZipFile(prev) as z:
        records=json.loads(z.read('manifest.json'))['files']
        for r in records:
            b=z.read(r['path']);assert hashlib.sha256(b).hexdigest()==r['sha256'];members[r['path']]=b
    added=set()
    for rel in ['paper/array_20261105_editorial','reports/ARRAY_EDITORIAL_20261105']:
        added.update(p for p in (ROOT/rel).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    added.update(ROOT.glob('scripts/*20261105.py'));added.update(ROOT.glob('docs/*20261105.md'))
    added.update(ROOT/r['path'] for r in outputs)
    added.update(ROOT/r for r in ['CURRENT_STATE.json','00_BACA_PERTAMA_AKUN_BARU.md','MEMORI_PENELITIAN_AKUN_BARU.md'])
    added.discard(REP/'package_verification.json')
    for p in added:members[p.relative_to(ROOT).as_posix()]=p
    members['README_CURRENT_20261105.md']=readme
    repro=archive(ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_20261105.zip',members)
    review={'01_NASKAH_UTAMA_ARRAY.pdf':ROOT/outputs[0]['path'],'02_SUPPLEMENT_ARRAY_S1_S45.pdf':ROOT/outputs[1]['path'],'00_BACA_DULU.md':readme,'03_CATATAN_REVISI.md':ROOT/'docs/ARRAY_EDITORIAL_20261105.md','04_PEMERIKSAAN_TURNITIN.md':ROOT/'docs/TURNITIN_CHECK_20261105.md','EDITOR_DISCLOSURE/related_manuscript_disclosure.md':ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_20261105.md','EDITOR_DISCLOSURE/AGERS_EDAS_1571374892.pdf':ROOT/'output/pdf/AGERS_HAFID_EDAS_20261028.pdf'}
    review_info=archive(ROOT/'penyerahan/UNTUK_DOSPEM_ARRAY_20261105.zip',review)
    result=dict(status='PASS_ALL_MEMBER_SHA256_AND_CRC_BOTH_ARCHIVES',pdfs=outputs,review=review_info,reproducibility=repro,turnitin_completed=False,turnitin_ai_score=None,scientific_results_unchanged=True)
    write(REP/'package_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
