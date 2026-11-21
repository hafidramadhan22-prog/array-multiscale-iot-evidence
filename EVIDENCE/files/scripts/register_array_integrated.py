"""Register the verified review package while retaining previous state history."""
from pathlib import Path
import copy
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    result=json.loads((ROOT/'penyerahan/ARRAY_HAFID_ALUR_LENGKAP_20261022.verification.json').read_text())
    assert result['status']=='ALL_DELIVERY_CHECKS_PASSED'
    for r in result['pdfs']+[result['archive']]:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    backup=ROOT/'history/array_integrated_20261022';backup.mkdir(parents=True,exist_ok=True)
    for name in ['CURRENT_STATE.json','START_HERE.md']:
        target=backup/name;assert not target.exists();shutil.copy2(ROOT/name,target)
    state_path=ROOT/'CURRENT_STATE.json'
    state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    assert state['latest_manuscript']['pages']==21
    old=copy.deepcopy(state['latest_manuscript'])
    state['previous_manuscript_20261022_complete']=old
    current=copy.deepcopy(old)
    a,b=result['pdfs']
    current.update(pdf=a['path'],pdf_sha256=a['sha256'],pages=a['pages'],figures=a['figures'],
        tables_main=a['tables'],source='paper/array_20261022_integrated/elsarticle/main.tex',
        archives=[result['archive']],delivery_notes='docs/REVISI_ALUR_LENGKAP_ARRAY_20261022.md',
        revision='INTEGRATED_INITIAL_AND_EXTENSION_STUDY',numeric_checks=result['numeric_checks'],
        standalone_figure_reproduction=result['standalone_figure_reproduction'],
        page_policy='User allows a few pages above the 20-page reference; 23 main pages including references, original 12pt/margins. No official Array numeric limit inferred.',
        results_coverage='Initial 28/29 feature comparison, 5x2 nested CV, baseline/expanded tuning, all-six historical holdout ranking, five-class recalls and initial SHAP integrated with later context, runtime, grouped XAI and negative pathways. Detailed raw evidence remains in supplementary/package and earlier full archive.',
        new_analyses='Historical result integration and saved-data visualizations only',
        supplementary=dict(pdf=b['path'],pdf_sha256=b['sha256'],pages=b['pages'],tables=b['tables'],figures=b['figures'],
                           source='paper/array_20261022_integrated/elsarticle/supplement.tex',separate_from_main=True))
    state['latest_manuscript']=current
    state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    start=ROOT/'START_HERE.md'
    intro='''**Naskah Array: alur awal dan tambahan disatukan, 22 Oktober 2026.** [PDF utama 23 halaman termasuk referensi](output/pdf/ARRAY_HAFID_ALUR_LENGKAP_20261022.pdf), [supplementary terpisah 22 halaman](output/pdf/ARRAY_HAFID_SUPPLEMENT_ALUR_LENGKAP_20261022.pdf), [ZIP sumber/bukti 146 entri](penyerahan/ARRAY_HAFID_ALUR_LENGKAP_20261022.zip), [peta alur dan cakupan](paper/array_20261022_integrated/ALUR_DAN_CAKUPAN.md), dan [perubahan](docs/REVISI_ALUR_LENGKAP_ARRAY_20261022.md). Main 8 tabel/4 gambar/25 referensi; supplement 24 tabel/5 gambar. Memasukkan 28/29 fitur, nested CV, tuning awal dan semua enam ensemble ke alur utama, disambungkan dengan multiscale, runtime dan SHAP kuantitatif. Seluruh 45 halaman ditinjau, 691 pemeriksaan numerik dan reproduksi mandiri lima grafik lulus; CRC/SHA256 isi ZIP cocok. Font 12 pt/margin template tetap. Sepuluh PDF/ZIP lama diperiksa identik. Hasil historis dan temporal tetap berbeda protokol; tanpa training/inferensi/model loading/game baru. Draft belum disubmit, deklarasi/persetujuan final penulis masih diperlukan.

'''
    start.write_text(intro+start.read_text(encoding='utf-8-sig'),encoding='utf-8')
    note=ROOT/'docs/REVISI_ALUR_LENGKAP_ARRAY_20261022.md'
    changes=(ROOT/'paper/array_20261022_integrated/CATATAN_PERUBAHAN.md').read_text(encoding='utf-8')
    note.write_text(changes+'''

## Keluaran terverifikasi

- PDF utama: `output/pdf/ARRAY_HAFID_ALUR_LENGKAP_20261022.pdf`.
- PDF supplementary: `output/pdf/ARRAY_HAFID_SUPPLEMENT_ALUR_LENGKAP_20261022.pdf`.
- ZIP penulisan: `penyerahan/ARRAY_HAFID_ALUR_LENGKAP_20261022.zip` (4.084.769 byte, 146 entri).
- Pemeriksaan: `penyerahan/ARRAY_HAFID_ALUR_LENGKAP_20261022.verification.json`.
- Source: `paper/array_20261022_integrated/`.
- Peta lengkap: `paper/array_20261022_integrated/ALUR_DAN_CAKUPAN.md`.

ZIP berisi LaTeX/template, dua PDF, grafik, bibliografi, data pendukung terpilih, snapshot engine awal, manifest dan audit. Arsip penuh semua model/notebook dan dashboard sebelumnya tetap terpisah. Snapshot CURRENT_STATE/START_HERE sebelum revisi disimpan di history/array_integrated_20261022/; semua versi lama tetap tersedia.
''',encoding='utf-8')
    assert json.loads(state_path.read_text(encoding='utf-8'))['latest_manuscript']['pdf_sha256']==a['sha256']
    print(json.dumps(dict(status='REGISTERED',main_pages=a['pages'],supplement_pages=b['pages'],
                         archive=result['archive']['path'],previous_state_backed_up=True),indent=2))


if __name__=='__main__':main()
