"""Register delivery pointers, preserving the preceding state as history."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    result=json.loads((ROOT/'penyerahan/ARRAY_HAFID.verification.json').read_text())
    assert result['status']=='ALL_DELIVERY_CHECKS_PASSED'
    state_path=ROOT/'CURRENT_STATE.json';start=ROOT/'START_HERE.md'
    history=ROOT/'history/array_delivery_20261021';history.mkdir(parents=True,exist_ok=True)
    for src in [state_path,start]:
        dst=history/src.name
        if not dst.exists():shutil.copy2(src,dst)
    state=json.loads(state_path.read_text(encoding='utf8'))
    old=state.get('latest_manuscript')
    if old and old.get('draft_date')!='2026-10-21':state['previous_manuscript_20261020']=old
    state['latest_manuscript']=dict(status='COMPLETE_REVIEW_DRAFT_NOT_SUBMITTED',draft_date='2026-10-21',journal='Array',
      authors=['M Hafid Ramadhan','Satria Mandala'],corresponding_author='Satria Mandala',correspondence_email='satriamandala@telkomuniversity.ac.id',
      shared_affiliation='Humic Engineering (Humic Eng), School of Computing, Telkom University',
      address='Jl. Telekomunikasi no. 1, Bandung 40257, West Java, Indonesia',pdf=result['pdf'],pdf_sha256=result['pdf_sha256'],
      pages=39,figures=4,tables_including_appendix=15,references=25,abstract_words=176,
      source='paper/array_20261021/elsarticle/main.tex',archives=result['archives'],
      visual_review_all_pdf_pages=True,array_specific_guide_access='HTTP403',submission_performed=False,new_training=False,
      new_independent_scientific_validation=False,missing_author_information=['funding','competing_interests','CRediT','final_all_author_approval','persistent_public_repository'],
      dashboard=dict(path='dashboard/app.py',verification='dashboard/verification.json',model_deserialization='Known project-owned model only after SHA256/provenance verification',
        unit_tests=10,http_checks=16,virtual_dom_checks=20,standalone_isolated_python_check='PASS',graphical_review='STOPPED_BY_USER_ESCAPE'),
      delivery_notes='docs/PENYERAHAN_ARRAY_20261021.md',previous_deliverables_preserved=True)
    state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    tag='**Naskah Array dan dashboard, 21 Oktober 2026:**'
    content=start.read_text(encoding='utf8')
    paragraph=tag+' [Paket lengkap](penyerahan/ARRAY_HAFID_LENGKAP.zip), [ZIP elsarticle](penyerahan/ARRAY_HAFID_ELSARTICLE.zip), [ZIP dashboard](penyerahan/DASHBOARD_HAFID.zip), [PDF 39 halaman](output/pdf/ARRAY_M_HAFID_RAMADHAN_SATRIA_MANDALA.pdf), dan [catatan penyerahan](docs/PENYERAHAN_ARRAY_20261021.md). Penulis M Hafid Ramadhan dan Satria Mandala; korespondensi Satria Mandala, afiliasi bersama Telkom University sesuai konfirmasi. Dashboard model nyata terverifikasi: 10 tes unit/integrasi, 16 HTTP, 20 DOM virtual, dan cek paket terisolasi lulus. Review grafis Chrome dihentikan lewat Escape. Semua halaman PDF serta CRC/SHA256 ZIP diperiksa; hasil/paket lama tetap identik. Empat artikel Array dan satu preseden kompresi/fidelity dibahas. Guide khusus Array masih 403; funding/konflik/CRediT/persetujuan akhir dan repository publik belum lengkap. Draft belum disubmit, tanpa training atau validasi capture baru.\n\n'
    if tag not in content:start.write_text(paragraph+content,encoding='utf8')
    print('Delivery registered; prior state backed up.')
if __name__=='__main__':main()
