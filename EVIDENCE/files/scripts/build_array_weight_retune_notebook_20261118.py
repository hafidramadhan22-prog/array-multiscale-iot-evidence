"""Execute a saved-results notebook for the weight experiment, no model loading."""
from pathlib import Path
import json
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'notebooks/array_weight_retune_20261118';OUT.mkdir(parents=True,exist_ok=True)
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))
def code(s):return dict(cell_type='code',metadata={'tags':['saved-results','no-classifier-refit']},execution_count=None,outputs=[],source=s.splitlines(keepends=True))
cells=[md('''# Uji penentuan ulang bobot WV pada G
18 November 2026. Expert, fitur, A fits dan split tetap. Bobot dipilih pada G
sebelum penilaian V dalam eksperimen ini; V sudah diamati pada tahap penelitian
sebelumnya, sehingga hasil tetap eksploratif. Tidak ada model loading atau
training expert dalam notebook ini. RF dan kedua SVM tetap ditampilkan.
'''),code('''from pathlib import Path
import json, sys, subprocess
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'reports/ARRAY_WEIGHT_RETUNE_20261118/analysis.json').exists())
R = ROOT/'reports/ARRAY_WEIGHT_RETUNE_20261118'
analysis = json.loads((R/'analysis.json').read_text())
print(analysis['conclusion'])
summary = pd.read_csv(R/'protocol_summary.csv')
display(summary.assign(F1_percent=100*summary.macro_f1).pivot(index='protocol', columns='method', values='F1_percent').round(4))
'''),md('''## Tujuan, recall dan bobot
WV_G_constrained adalah kebijakan utama: syarat empiris pada G tidak otomatis
berlaku pada V. WV_G_unconstrained dan WV_G_heuristic adalah kontrol yang
ditetapkan sebelum penilaian. Tidak ada pemilihan kebijakan dari hasil V.
'''),code('''display(pd.read_csv(R/'goal_assessment.csv').round(4))
display(pd.read_csv(R/'selected_weights.csv').round(4))
private = summary[summary.protocol=='packet_private']
display(private[['method','macro_f1','normal_fpr','binary_attack_recall','recall_SQL_injection','recall_Ransomware','errors']].round(6))
with np.load(R/'packet_private_pooled_predictions.npz', allow_pickle=False) as z:
    sample = pd.DataFrame({k:z[k] for k in ['row_ids','y','WV_reference','WV_G_constrained','RF','SVM_default']})
print('Contoh aktual target dengan label berubah:')
display(sample[sample.WV_reference != sample.WV_G_constrained].head(25))
'''),md('''## Stress checks dan confusion matrix
Stress memakai bobot temporal seed-42, bukan bobot rata-rata private folds.
CIC memakai scenario labels; TON memakai label turunan author flows. Jangan
menggabungkan keduanya sebagai satu independent multiclass score.
'''),code('''stress = pd.read_csv(R/'stress_summary.csv')
display(stress[['family','method','normal_support','false_alarms','normal_fpr','attack_support','attack_to_normal','binary_attack_recall','recall_DDoS_TCP','recall_Ransomware']].round(5))
display(Image(filename=str(R/'figures/private_confusions.png')))
'''),md('''## Audit portabel
Audit membentuk ulang semua kandidat G dan probabilitas/label baru dari input
expert probabilities. Tidak ada classifier training, model loading atau source
data download. Pencarian ulang memakai G yang telah dibekukan, bukan V.
'''),code('''output = ROOT/'tmp/WEIGHT_NOTEBOOK_AUDIT.json'
check = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/verify_array_weight_retune_20261118.py'), '--root', str(ROOT), '--output', str(output)], cwd=ROOT, text=True, capture_output=True)
print(check.stdout)
if check.returncode:
    raise RuntimeError(check.stderr)
assert json.loads(output.read_text())['status'] == 'PASS_INDEPENDENT_WEIGHT_SEARCH_AND_PREDICTIONS'
print('PASS - saved evidence only, no model deserialization')
'''),md('''## Keputusan
Private macro-F1 membaik, tetapi SQL recall dan binary attack recall menurun.
Temporal memburuk; shared signature tidak berubah. Penurunan false alarms pada
benign lama tidak membuktikan generalisasi; trade-off eksternal tetap besar.
Tidak ada dasar untuk mengganti predictor jurnal secara otomatis. Bobot baru
mengubah fungsi, sehingga runtime/SHAP lama tidak otomatis diwariskan.
''')]
shell=InteractiveShell.instance();count=0
for cell in cells:
    if cell['cell_type']!='code':continue
    count+=1;cell['execution_count']=count
    with capture_output() as capture:result=shell.run_cell(''.join(cell['source']),store_history=False)
    if result.error_before_exec or result.error_in_exec:raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
    outputs=[]
    if capture.stdout:outputs.append(dict(output_type='stream',name='stdout',text=capture.stdout.splitlines(keepends=True)))
    if capture.stderr:outputs.append(dict(output_type='stream',name='stderr',text=capture.stderr.splitlines(keepends=True)))
    for value in capture.outputs:outputs.append(dict(output_type='display_data',data=value.data,metadata=value.metadata or {}))
    cell['outputs']=outputs
for i,c in enumerate(cells):c['id']=f'weight-20261118-{i:02d}'
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3 (saved results)',language='python',name='python3'),language_info=dict(name='python',version='3.11'),all_code_cells_executed=True),cells=cells)
p=OUT/'UJI_BOBOT_G_HASIL_DAN_PREDIKSI_20261118.ipynb';p.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8');print('EXECUTED_WEIGHT_NOTEBOOK',count,'code cells;',len(cells),'total cells')
