"""Append the new artifact-only XAI audit to the same combined review notebook."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from execute_review_notebook import execute as execute_guarded
from export_review_html import export


def build(run,execute=False):
    run=run.resolve();assert json.loads((run/'verification.json').read_text())['status']=='PASS_XAI_RECONSTRUCTION_AND_PROVENANCE'
    path=ROOT/'notebooks/TA_HAFID_TINJAUAN_HASIL.ipynb';backup=ROOT/'history/20261020_before_XAI_review';backup.mkdir(exist_ok=True)
    for rel in ['notebooks/TA_HAFID_TINJAUAN_HASIL.ipynb','CURRENT_STATE.json','START_HERE.md','docs/PROTOCOL_TARGET.md']:
        dest=backup/Path(rel).name
        if not dest.exists():shutil.copy2(ROOT/rel,dest)
    nb=json.loads(path.read_text(encoding='utf8'));nb['cells']=[c for c in nb['cells'] if not c['metadata'].get('xai_audit_20261020')]
    def cell(kind,text):
        value=dict(cell_type=kind,metadata={'xai_audit_20261020':True},source=text.splitlines(keepends=True),id=hashlib.sha256(text.encode()).hexdigest()[:12])
        if kind=='code':value.update(execution_count=None,outputs=[])
        nb['cells'].append(value)
    cell('markdown','## 64. Audit kuantitatif faithfulness dan robustness SHAP\n\nTambahan20 Oktober2026 mengevaluasi model Weighted Voting yang sudah dibekukan. Tidak ada pelatihan classifier/bobot baru.100V berkuota20 perkelas;8 donor trainingA dan8 donor alternatif yang disjoint.10 bundle fit lama dan kontrol expertLR/RF. Hasil eksploratif pada reused-development, bukan testcapture independen. SHAP menjelaskan probabilitas ensemble lengkap; label-only scheduler tidak diperlakukan sebagai posterior lengkap.')
    cell('code',f"from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nxai_root=Path.cwd()\nif not (xai_root/'CURRENT_STATE.json').exists():xai_root=xai_root.parent\nxai_run=xai_root/{run.relative_to(ROOT).as_posix()!r}\nxai_summary=json.loads((xai_run/'report/summary.json').read_text(encoding='utf8'))\ndisplay(pd.DataFrame([xai_summary['complete']]))")
    cell('markdown','## 65. Deletion dan insertion terhadap baseline acak\n\nAudit utama memakai exact group Shapley pada kelompok protokol/port/riwayat. Semakin kecil AUC probabilitas tersisa pada deletion, semakin baik; insertion sebaliknya. Gain dihitung terhadap random pada game yang sama, dengan target kelas prediksi awal tetap. Additivity tidak menggantikan pengukuran faithfulness dan skor ini bukan bukti kausal.')
    cell('code',"display(pd.read_csv(xai_run/'tables/group_summary.csv'))\ndisplay(pd.read_csv(xai_run/'tables/group_per_class.csv'))\ndisplay(pd.read_csv(xai_run/'tables/group_attribution_summary.csv'))\ndisplay(Image(filename=str(xai_run/'report/faithfulness_curves.png')))\ndisplay(pd.read_csv(xai_run/'tables/group_win_rates.csv'))")
    cell('markdown','## 66. Background dan baseline masking\n\nDonorA danB dipilih dari training asli yang saling terpisah. RankingA juga dievaluasi pada gameB untuk menguji ketergantungan referensi. Zero baseline hanya diagnosis; hybrid antargrup tidak dijamin sah sebagai paket jaringan. Kelompok40 fitur riwayat tetap utuh, sehingga efek pergrup tidak setara efek perkolom.')
    cell('code',"display(pd.read_csv(xai_run/'tables/masker_sensitivity_summary.csv'))\ndisplay(pd.read_csv(xai_run/'tables/background_summary.csv'))\ndisplay(pd.read_csv(xai_run/'tables/ranking_summary.csv'))")
    cell('markdown','## 67. Ketahanan penjelasan saat prediksi tetap\n\nPort bijection mengubah nilai port eksplisit. Query delay1/5ms menghitung ulang konteks dari ring sebelum paket target masuk, tanpa paket masa depan. Ini bukan adaptive adversarial attack. Prediction retention dan attribution drift dilaporkan terpisah; tabel inputberubah+labeltetap mencegah input identik menggelembungkan klaim robustness.')
    cell('code',"display(pd.read_csv(xai_run/'tables/robustness_summary.csv'))\ndisplay(pd.read_csv(xai_run/'tables/robustness_counts.csv'))\ndisplay(pd.read_csv(xai_run/'tables/robustness_unchanged_prediction.csv'))\ndisplay(pd.read_csv(xai_run/'tables/robustness_changed_input_same_prediction.csv'))\ndisplay(Image(filename=str(xai_run/'report/robustness.png')))")
    cell('markdown','## 68. Individual permutation SHAP dan ketidakpastian aproksimasi\n\n25V pada dua richBalanced seed42;4 siklus antithetic dan2seed. Individual SHAP merupakan objek yang berbeda dari exact group Shapley. Masking individual dapat memecah konsistensi port/band dan40 statistik riwayat. AUC antarresolusi tidak dibandingkan langsung sebagai lomba. Dua seed explainer mengukur noise aproksimasi, bukan dua capture.')
    cell('code',"display(pd.read_csv(xai_run/'tables/atomic_summary.csv'))\ndisplay(pd.read_csv(xai_run/'tables/atomic_seed_summary.csv'))\nxai_mask=[]\nfor xai_folder in sorted((xai_run/'atomic').iterdir()):\n    for xai_row in json.loads((xai_folder/'mask_violations.json').read_text()):\n        xai_mask.append(dict(bundle=xai_folder.name,**xai_row))\ndisplay(pd.DataFrame(xai_mask))")
    cell('markdown','## 69. Research gap dan penilaian novelty\n\nE-XAI2024, evaluasiNIDS2025, feature-groupingIoT2025 dan FL-XAI2026 sudah mengevaluasi penjelasan secara kuantitatif. Klaim95% paper hanya plot tidak memiliki denominator systematic review. Kontribusi yang diuji adalah audit pada ensemble heterogen berkonteks streaming dan sensitivitas terhadap pilihan masker/background, bukan penemuan deletion/insertion. Justifikasi lengkap: docs/XAI_LITERATURE_20261020.md.')
    cell('code',"display(Markdown(chr(10).join('- '+s for s in xai_summary['conclusions'])))\ndisplay(Markdown(chr(10).join('- '+s for s in xai_summary['limitations'])))")
    cell('markdown','## 70. Verifikasi dan lokasi keluaran audit SHAP\n\nLaporan: runs/XAI_FULL_20261020_01/report/LAPORAN_XAI_SHAP.html. Seluruhkoalisi/atribusi/kurva/indeks/background/probabilitas/source tersimpan. Tidak mengubah pemenang rekomendasi historis atau mengulang250fit. Jumlah baris bukan unitcapture independen dan hasil audit tidak menjamin penerimaan jurnalQ1.')
    cell('code',"xai_check=json.loads((xai_run/'verification.json').read_text())\ndisplay(pd.DataFrame([{k:v for k,v in xai_check.items() if isinstance(v,(str,int,float,bool))}]))\ndisplay(pd.DataFrame(xai_check['old_runs']))")
    nb['metadata']['ta_hafid'].update(xai_audit_extension=True,xai_run=run.name,sections=70)
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'xai_review_cell_{i}','exec')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if execute:
        os.environ['IPYTHONDIR']=str(ROOT/'audit/ipython_xai_review');execute_guarded(path,ROOT/'audit/xai_combined_notebook_execution_20261020.json')
    export(path,ROOT/'LAPORAN_GABUNGAN_XAI_20261020.html')
    print('XAI_REVIEW_READY',len(nb['cells']),'cells')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();build(a.run,a.execute)
