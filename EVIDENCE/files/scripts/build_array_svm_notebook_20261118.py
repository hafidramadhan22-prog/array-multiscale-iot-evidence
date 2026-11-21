"""Execute a saved-evidence notebook with real IPython outputs; no training."""
from pathlib import Path
import json,time
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks/array_svm_20261118';OUT.mkdir(parents=True,exist_ok=True)
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))
def code(s):return dict(cell_type='code',metadata={'tags':['saved-evidence-only','no-training']},execution_count=None,outputs=[],source=s.splitlines(keepends=True))
cells=[md('''# Uji SVM comparator - jurnal Array
18 November 2026. Hasil aktual: WV tidak konsisten mengungguli SVM.

WV tetap berisi DT, RF, XGBoost, LightGBM, dan LR. SVM adalah comparator
standalone. Notebook ini hanya membaca hasil tersimpan dan menjalankan audit;
**tidak melakukan training atau memuat model**. SVM-D = C1/gamma scale;
SVM-G = kandidat dipilih pada G dari grid 9 kombinasi sebelum V dinilai.
RF dan semua hasil negatif tetap disertakan. Run All dari paket yang diekstrak.
'''),code('''from pathlib import Path
import json, sys, subprocess
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'reports/ARRAY_SVM_20261118/analysis.json').exists())
R = ROOT/'reports/ARRAY_SVM_20261118'
analysis = json.loads((R/'analysis.json').read_text(encoding='utf-8'))
print(analysis['conclusion'])
summary = pd.read_csv(R/'protocol_summary.csv')
display(summary.assign(macro_F1_percent=100*summary.macro_f1).pivot(index='protocol', columns='method', values='macro_F1_percent').round(4))
'''),md('''## Tujuan dan perbandingan berpasangan
Nilai positif WV_minus_comparator_pp berarti WV lebih tinggi pada protokol itu.
Satu hasil lebih tinggi tidak membuktikan superioritas pada semua kondisi.
'''),code('''comparisons = pd.read_csv(R/'goal_comparisons.csv')
display(comparisons.round(4))
display(pd.read_csv(R/'selection_and_training.csv'))
'''),md('''## Confusion matrix packet-private
Teks adalah jumlah prediksi aktual; warna menunjukkan proporsi baris.
Target pooled OOF adalah 52.584, dengan true classes dan urutan yang sama.
'''),code('''display(Image(filename=str(R/'figures/packet_private_confusions.png')))
'''),md('''## False alarms dan recall eksternal
FPR = false alarms / jumlah Normal. CIC memakai label skenario file;
TON memakai label turunan author flows. Tidak ada macro-F1 lima kelas
yang menggabungkan kedua dataset. Semua merupakan stress check retrospektif.
'''),code('''stress = pd.read_csv(R/'stress_summary.csv')
cols = ['family','method','normal_support','false_alarms','normal_fpr','attack_support','attack_to_normal','binary_attack_recall','recall_DDoS_TCP','recall_Ransomware']
display(stress[cols].round(4))
print('Per-capture details:')
display(pd.read_csv(R/'stress_per_capture.csv')[['cohort','method','n','false_alarms','normal_fpr','errors']].round(4))
'''),md('''## Serving cost
Satu laptop, satu numerical thread, prepared features, label-only requests.
30 repetitions setelah 5 warm-up; batch latency dalam ms; RSS dari 5 cold
processes per kebijakan. Model loading tidak masuk inference delta RSS.
SVM memakai pipeline standar; tidak ada compact SVM khusus yang diuji.
'''),code('''display(pd.read_csv(R/'serving_summary.csv').round(4))
'''),md('''## Audit hasil tersimpan
Perintah berikut memeriksa SHA256, split, cap, G selection, parent labels,
confusion matrices dan raw measurements. Tidak menjalankan estimator.
'''),code('''check_output = ROOT/'tmp/SVM_NOTEBOOK_SAVED_AUDIT.json'
result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/verify_array_svm_20261118.py'), '--root', str(ROOT), '--output', str(check_output)], cwd=ROOT, text=True, capture_output=True)
print(result.stdout)
if result.returncode:
    raise RuntimeError(result.stderr)
assert json.loads(check_output.read_text())['status'] == 'PASS_INDEPENDENT_SVM_SAVED_EVIDENCE_AUDIT'
print('PASS - no model loading, inference or training in this notebook')
'''),md('''## Kesimpulan
SVM default pada packet-private history melebihi WV; SVM-G pada temporal juga
sedikit melebihi WV. WV jauh lebih tinggi pada signature/shared-history
diagnostic dan dibanding SVM-G pada private history. Semua hasil disajikan
bersama. FPR WV tidak berubah karena classifier WV tidak dimodifikasi.
Kegagalan transfer, source/class confounding, dan bias tahap penelitian yang
telah diamati tetap membatasi klaim. Artikel utama belum diganti otomatis.
''')]
shell=InteractiveShell.instance();count=0
for cell in cells:
    if cell['cell_type']!='code':continue
    count+=1;cell['execution_count']=count
    with capture_output() as captured:
        result=shell.run_cell(''.join(cell['source']),store_history=False)
    if result.error_before_exec or result.error_in_exec:raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
    outputs=[]
    if captured.stdout:outputs.append(dict(output_type='stream',name='stdout',text=captured.stdout.splitlines(keepends=True)))
    if captured.stderr:outputs.append(dict(output_type='stream',name='stderr',text=captured.stderr.splitlines(keepends=True)))
    for output in captured.outputs:
        outputs.append(dict(output_type='display_data',data=output.data,metadata=output.metadata or {}))
    cell['outputs']=outputs
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3 (saved evidence)',language='python',name='python3'),language_info=dict(name='python',version='3.11'),experiment='standalone SVM comparator; saved evidence, no fitting',all_code_cells_executed=True),cells=cells)
for i,c in enumerate(cells):c['id']=f'svm-20261118-{i:02d}'
path=OUT/'UJI_SVM_HASIL_DAN_PREDIKSI_20261118.ipynb';path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('NOTEBOOK_EXECUTED',count,'code cells;',len(cells),'total cells;',path)
