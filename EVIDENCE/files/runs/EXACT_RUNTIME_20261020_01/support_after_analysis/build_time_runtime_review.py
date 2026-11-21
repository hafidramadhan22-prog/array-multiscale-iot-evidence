"""Extend the combined review notebook with verified timing and runtime evidence."""
from pathlib import Path
import argparse,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from execute_review_notebook import execute as execute_guarded
from export_review_html import export


def build(execute=False):
    path=ROOT/'notebooks/TA_HAFID_TINJAUAN_HASIL.ipynb';nb=json.loads(path.read_text(encoding='utf8'))
    assert nb['metadata']['ta_hafid']['sections'] in (95,102)
    nb['cells']=[c for c in nb['cells'] if not c['metadata'].get('time_runtime_20261020')]
    def add(kind,text):
        cell=dict(cell_type=kind,metadata={'time_runtime_20261020':True},source=text.splitlines(keepends=True),
                  id=hashlib.sha256(text.encode()).hexdigest()[:12])
        if kind=='code':cell.update(execution_count=None,outputs=[])
        nb['cells'].append(cell)
    add('markdown','## 96. Lanjutan setelah penelusuran paper primer\n\nHipotesis tambahan informasi waktu dan optimasi runtime disusun setelah menelaah ICISSP2017, Kitsune, Hummingbird dan CATO. Metode mempunyai preseden; konstanta lokal bukan persetujuan numerik dospem. Protokol dibekukan sebelum fitting. V lama tetap development yang digunakan ulang.')
    add('code',"from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\ntr_root=Path.cwd()\nif not (tr_root/'CURRENT_STATE.json').exists():tr_root=tr_root.parent\ntr_q=tr_root/'runs/IAT_REPAIR_FULL_20261020_01'\ntr_r=tr_root/'runs/EXACT_RUNTIME_20261020_01'\ndisplay(Markdown((tr_root/'docs/TIME_RUNTIME_LITERATURE_20261020.md').read_text(encoding='utf8')))\ndisplay(Markdown((tr_root/'docs/TIME_RUNTIME_PROTOCOL_20261020.md').read_text(encoding='utf8')))")
    add('markdown','## 97. Data waktu: query sebelum push dan reproduksi pembanding\n\n24 statistik waktu relatif untuk dua host dan tiga jendela; kontrol age memakai enam kolom dari slot host yang sama. Seluruh old60/arah36 harus persis. Tidak menambah capture, tidak memasukkan ID/waktu absolut sebagai predictor, dan tidak mengatasi clipping kapasitas ring secara otomatis.')
    add('code',"display(json.loads((tr_root/'runs/IAT_CONTEXT_DATA_20261020_01/complete.json').read_text()))\ndisplay(json.loads((tr_root/'runs/IAT_CONTEXT_DATA_20261020_01/mapping_audit.json').read_text()))\ndisplay(json.loads((tr_q/'verification_quality.json').read_text()))")
    add('markdown','## 98. Kualitas seluruh variasi dan treatment\n\n150 fit penelitian dan30 bundle; smoke30fit terpisah. Baseline utama adalah WB25+multiscale60, bukan baseline40 yang lebih lemah. Ringkas Balanced di bawah, diikuti semua hasil Original/Balanced dan model tunggal. Temporal mean tiga seed; signature pooled dua fold, bukan dua capture baru.')
    add('code',"display(pd.read_csv(tr_q/'tables/time_runtime_quality_summary.csv'))\ndisplay(pd.read_csv(tr_q/'tables/temporal_summary.csv'))\ndisplay(pd.read_csv(tr_q/'tables/signature_pooled.csv'))\ndisplay(Image(filename=str(tr_q/'report/time_runtime_summary.png')))")
    add('markdown','## 99. Kompromi kelas, pemilihan G dan sensitivitas blok\n\nRecall scan signature naik, Ransomware dan temporal dapat turun. G Balancedseed42 tetap memilih baseline. Tidak mengganti kandidat berdasarkanV terbaik. Kuantil paired-block1/5/10detik bersyarat pada trace; bukan p-value atau confidence interval populasi.')
    add('code',"display(pd.read_csv(tr_q/'tables/G_selected_V_results.csv'))\ndisplay(pd.read_csv(tr_q/'tables/signature_confusions.csv').query(\"method == 'weighted_voting' and count > 0\"))\ndisplay(pd.read_csv(tr_q/'tables/paired_block_sensitivity.csv'))")
    add('markdown','## 100. Runtime ekuivalen: latency dan delta peak RSS\n\nMedians/categories/scale dan lima expert lama dipakai tanpa fit. CSR/zero/missing, bobot dan urutan C tetap. Setiap batch45timing; replay5ulang dengan urutan policy diacak; coldRSS3proses per policy/workload. Replay mencakup ring/decode/konteks/model tetapi fiturlegacy tetap dariCSV. Ukuranfile bukanRAM; semua angka satu sesi laptop baru.')
    add('code',"display(json.loads((tr_r/'equivalence.json').read_text()))\ndisplay(json.loads((tr_r/'verification_independent.json').read_text()))\ndisplay(pd.read_csv(tr_r/'readable_summary.csv'))\ndisplay(pd.read_csv(tr_r/'latency_summary.csv'))\ndisplay(pd.read_csv(tr_r/'memory_summary.csv'))\ndisplay(pd.read_csv(tr_r/'replay_raw.csv'))")
    add('markdown','## 101. SHAP tetap terhubung dengan fungsi model utama\n\n300game A32/B32/querydelay5ms lama diperiksa ulang menggunakan runtimekompak:38.400koalisi dan1.228.800input donor. Ini verifikasi ekuivalensi fungsi, bukan classifier baru atau klaim kausal. Audit faithfulness/robustness utama tetap hasil model multiscale60 yang dipilihG sebelumnya.')
    add('code',"display(json.loads((tr_r/'xai_equivalence.json').read_text()))\ndisplay(pd.read_csv(tr_r/'xai_equivalence_raw.csv').groupby('game')[['max_probability_difference','max_shap_difference']].max())\ndisplay(pd.read_csv(tr_root/'runs/XAI_MULTISCALE_FULL_20261020_01/tables/faithfulness_summary.csv'))\ndisplay(pd.read_csv(tr_root/'runs/XAI_MULTISCALE_FULL_20261020_01/tables/robustness_summary.csv'))")
    add('markdown','## 102. Keputusan setelah pengujian lanjutan\n\nModel utama dipertahankan, tambahan waktu menjadi ablation dengan kompromi, dan runtime lebih ringan dinilai dari benchmark aktual. Bukti dalam capture lama tidak menjamin generalisasi, novelty algoritmik atau jurnalQ1. Ring/capture/fiturlegacy/hardware tetap batas klaim.')
    add('code',"display(Markdown((tr_root/'docs/HASIL_TIME_RUNTIME_20261020.md').read_text(encoding='utf8')))")
    nb['metadata']['ta_hafid'].update(time_runtime_extension=True,time_quality_run='IAT_REPAIR_FULL_20261020_01',
        exact_runtime_run='EXACT_RUNTIME_20261020_01',sections=102)
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'time_runtime_review_{i}','exec')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if execute:
        os.environ['IPYTHONDIR']=str(ROOT/'audit/ipython_time_runtime_review')
        execute_guarded(path,ROOT/'audit/time_runtime_notebook_execution_20261020.json')
    export(path,ROOT/'LAPORAN_GABUNGAN_TIME_RUNTIME_20261020.html')
    print('TIME_RUNTIME_REVIEW_READY',len(nb['cells']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');build(p.parse_args().execute)
