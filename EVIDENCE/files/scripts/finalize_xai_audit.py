"""Finish XAI provenance, navigation and delivery report without altering old runs."""
from pathlib import Path
from datetime import datetime
import json,shutil,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import ta_hafid_engine as e


def finalize():
    run=ROOT/'runs/XAI_FULL_20261020_01'
    if (run/'run_complete.json').exists():raise ValueError('Do not alter a completed XAI run')
    result=json.loads((run/'verification.json').read_text());assert result['status']=='PASS_XAI_RECONSTRUCTION_AND_PROVENANCE'
    nb=json.loads((ROOT/'audit/xai_combined_notebook_execution_20261020.json').read_text());assert nb['status']=='PASS_COMBINED_NOTEBOOK_EXECUTION'
    assert e.sha256(ROOT/'notebooks/TA_HAFID_TINJAUAN_HASIL.ipynb')==nb['sha256']
    tests=ET.parse(ROOT/'audit/xai_tests_20261020_final.xml').getroot();suites=list(tests.iter('testsuite'))
    assert all(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0))==0 for s in suites)
    count=sum(int(s.attrib['tests']) for s in suites)
    summary=(run/'report/HASIL_DAN_KESIMPULAN.md').read_text(encoding='utf8')
    changes='''\n## Perubahan dan status lima ide awal\n\nTambahan baru: src/xai_audit.py, scripts/run_xai_audit.py, verifier/report/review/finalizer XAI; konfigurasi, protokol, referensi dan tes XAI. Model, hyperparameter, bobot, split, serta hasil run lama tidak diubah. Notebook yang sama ditambah bagian64–70; backup di history/20261020_before_XAI_review/. Tidak melakukan training baru. ZIP18 Oktober tidak otomatis diperbarui.\n\nLima ide awal **tidak dinyatakan pasti tidak layak**. MPED prototipe sudah diuji dan belum melampaui supervised control; bukan bukti semua distillation gagal. Structural Weighted-Ensemble Compilation belum dikembangkan penuh, jadi tidak ada hasil negatif yang sah untuk keseluruhan ide2. Shared-Computation Certified Scheduling memberi penghematan waktu terukur, tetapi prinsip earlyexit/sharedcost sudah punya priorart. Port-Robust Adaptive Fusion baru didukung ablation/intervensi port; learned adaptivefusion lengkap belum diuji. Selective Behavioral Feature Escalation memberi manfaat fitur konteks; learnedgate sebelumnya belum jelas mengalahkan confidence sederhana, dan sertifikatinterval baru0/100. Hasil ini membatasi prototipe yang dicoba, bukan membuktikan seluruh kemungkinan algoritma mustahil.\n\nKandidat kontribusi terkuat dari hasil tersedia adalah studi empiris yang menyatukan kualitas deteksi, ketergantungan port, biaya inferensi dan keandalan penjelasan pada pipeline yang sama. Ini boleh menjadi fokus artikel empiris, tetapi klaim algoritma baru memerlukan mekanisme yang benar-benar berbeda dari paper terdekat. Menambahkan SHAP faithfulness saja tidak memenuhi syarat tersebut. Validasi capture independen dan populasiNormal yang lebih luas masih menjadi batas utama; mengganti seed atau menambah plot tidak menggantikannya.\n'''
    (ROOT/'docs/XAI_HASIL_DAN_PERUBAHAN_20261020.md').write_text(summary+changes,encoding='utf8')
    state=json.loads((ROOT/'CURRENT_STATE.json').read_text());state.update(updated_date='2026-10-20',updated_jakarta=datetime.now().astimezone().isoformat(),
        status='EXPERIMENT_COMPLETE_XAI_SHAP_VERIFIED',tests_passed=count)
    state['xai_audit']=dict(run=run.relative_to(ROOT).as_posix(),status='COMPLETE_VERIFIED',bundles=10,validation_rows_per_bundle=100,
        new_classifier_fits=0,notebook_sections='64-70',novelty_established=False,independent_capture_validation=False)
    state['next']=['Review XAI notebook64-70 and quantitative report with background/masking limitations',
        'Do not repeat completed WB fits or infer algorithmic novelty from adding existing XAI metrics',
        'Independent capture confirmation and faithful baseline reproduction remain requirements for stronger claims']
    e.write_json(ROOT/'CURRENT_STATE.json',state)
    lead='''**Terbaru20 Oktober2026: audit faithfulness dan robustness SHAP selesai.** [Laporan kuantitatif](runs/XAI_FULL_20261020_01/report/LAPORAN_XAI_SHAP.html), [hasil dan perubahan](docs/XAI_HASIL_DAN_PERUBAHAN_20261020.md), [justifikasi literatur](docs/XAI_LITERATURE_20261020.md), [HTML gabungan](LAPORAN_GABUNGAN_XAI_20261020.html). Notebook yang sama bagian64–70:10 bundle WB tersimpan,100V perbundle dengan kuota kelas sama, exact group Shapley, kontrol LR/RF/random/least, background alternatif, port/querydelay dan individual permutationSHAP25 baris pada2 bundle. Tidak melatih ulang model dan tidak menggunakan holdout historis. Evaluasi masih eksploratif; quantitativeXAI padaIDS/IoT sudah punya priorart dan novelty/Q1 tidak otomatis ditetapkan.\n\n'''
    start=ROOT/'START_HERE.md';start.write_text(lead+start.read_text(encoding='utf8'),encoding='utf8')
    protocol=ROOT/'docs/PROTOCOL_TARGET.md';protocol.write_text('''## Audit XAI20 Oktober2026\n\nPengguna secara eksplisit meminta faithfulness/robustnessSHAP serta justifikasi paper. Protokol `XAI_AUDIT_PROTOCOL_20261020.md`, sumber `XAI_LITERATURE_20261020.md`, run `runs/XAI_FULL_20261020_01`. ModelWB yang dibuat proyek dan cocokchecksum dimuat hanya untuk probabilitas lengkap; tidak ada fit/retuning. Abackground, Vevaluasi, holdout tidak dipakai. Default8donor/100V/25individual/4cycles merupakan desain audit eksploratif. Deletion/insertion adalah tes perturbasi dengan batas validitas, bukan bukti sebab-akibat.\n\n'''+protocol.read_text(encoding='utf8'),encoding='utf8')
    post=run/'source_postexecution';post.mkdir()
    for rel in ['scripts/build_xai_audit_report.py','scripts/verify_xai_audit.py','scripts/build_xai_review.py','scripts/finalize_xai_audit.py','tests/test_xai_interventions.py','docs/XAI_HASIL_DAN_PERUBAHAN_20261020.md','docs/KEPUTUSAN_JALUR_NOVELTY_20261020.md']:
        target=post/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    for rel in ['audit/xai_tests_20261020_final.xml','audit/xai_combined_notebook_execution_20261020.json','audit/xai_full_20261020_01.log']:
        shutil.copy2(ROOT/rel,run/Path(rel).name)
    records=[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*')) if p.is_file()]
    e.write_json(run/'artifacts_sha256.json',records)
    e.write_json(run/'run_complete.json',dict(status='XAI_AUDIT_COMPLETE_VERIFIED',catalog_sha256=e.sha256(run/'artifacts_sha256.json'),
        artifacts=len(records),tests_passed=count,new_classifier_fits=0,metric_rows=result['metric_rows_reconstructed'],novelty_established=False))
    e.write_json(ROOT/'audit/xai_completion_20261020.json',dict(status='PASS_XAI_AUDIT',run_complete_sha256=e.sha256(run/'run_complete.json'),
        tests_passed=count,notebook=nb,old_artifacts_unchanged=result['old_artifacts_unchanged'],
        outputs={rel:e.sha256(ROOT/rel) for rel in ['CURRENT_STATE.json','START_HERE.md','docs/PROTOCOL_TARGET.md','docs/XAI_HASIL_DAN_PERUBAHAN_20261020.md','LAPORAN_GABUNGAN_XAI_20261020.html']}))
    print(json.dumps(dict(status='XAI_AUDIT_COMPLETE_VERIFIED',tests_passed=count,artifacts=len(records)),indent=2))


if __name__=='__main__':finalize()
