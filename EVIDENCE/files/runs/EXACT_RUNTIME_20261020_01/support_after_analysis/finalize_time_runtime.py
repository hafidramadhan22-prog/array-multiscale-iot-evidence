"""Seal this extension and update pointers; prior artifacts stay byte-identical."""
from pathlib import Path
import json,shutil,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import ta_hafid_engine as e
from run_time_repair import check_record
from prepare_wb_behavior import verify_catalog

NEW=['IAT_CONTEXT_DATA_20261020_01','IAT_REPAIR_SMOKE_20261020_01','IAT_REPAIR_FULL_20261020_01','EXACT_RUNTIME_20261020_01']
SUPPORT=['scripts/analyze_time_repair.py','scripts/profile_exact_runtime.py','scripts/verify_exact_runtime.py',
    'scripts/verify_exact_xai.py','scripts/predict_exact_runtime.py','scripts/build_time_runtime_report.py',
    'scripts/build_time_runtime_review.py','scripts/finalize_time_runtime.py','scripts/prepare_time_context.py',
    'scripts/create_time_runners.py','src/exact_runtime.py','tests/test_exact_runtime.py','tests/test_time_context.py',
    'docs/TIME_RUNTIME_LITERATURE_20261020.md','docs/TIME_RUNTIME_PROTOCOL_20261020.md',
    'docs/TIME_RUNTIME_REPORT_TEMPLATE.md','docs/HASIL_TIME_RUNTIME_20261020.md']


if __name__=='__main__':
    test=ET.parse(ROOT/'audit/time_runtime_full_tests_20261020_02.xml').getroot();suites=list(test.iter('testsuite'))
    assert sum(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0)) for s in suites)==0
    assert sum(int(s.attrib['tests']) for s in suites)==170
    q=ROOT/'runs/IAT_REPAIR_FULL_20261020_01';r=ROOT/'runs/EXACT_RUNTIME_20261020_01'
    assert json.loads((q/'verification_quality.json').read_text())['status']=='PASS_CONTEXT_REPAIR_QUALITY'
    assert json.loads((r/'verification_independent.json').read_text())['status']=='PASS_RUNTIME_EVIDENCE_RECONSTRUCTION'
    summary=json.loads((q/'report/time_runtime_summary.json').read_text())
    assert summary['main_classifier_unchanged'] and summary['G_selected_balanced_seed42']=='baseline25'
    notebook=ROOT/'notebooks/TA_HAFID_TINJAUAN_HASIL.ipynb';nb=json.loads(notebook.read_text(encoding='utf8'))
    assert nb['metadata']['ta_hafid']['sections']==102
    assert all(c.get('execution_count') is not None and not any(o.get('output_type')=='error' for o in c.get('outputs',[]))
        for c in nb['cells'] if c['cell_type']=='code')
    execution=json.loads((ROOT/'audit/time_runtime_notebook_execution_20261020.json').read_text())
    assert execution['status']=='PASS_COMBINED_NOTEBOOK_EXECUTION' and execution['sha256']==e.sha256(notebook)
    old=[]
    for folder in sorted((ROOT/'runs').iterdir()):
        cat=folder/'artifacts_sha256.json'
        if folder.name in NEW or not cat.is_file():continue
        records=json.loads(cat.read_text())
        for rec in records:check_record(folder,rec)
        old.append(dict(run=folder.name,files=len(records),catalog_sha256=e.sha256(cat)))
    verify_catalog(ROOT/'runs/IAT_CONTEXT_DATA_20261020_01')
    for run in [q,r]:
        support=run/'support_after_analysis';support.mkdir()
        for rel in SUPPORT:shutil.copy2(ROOT/rel,support/Path(rel).name)
        e.write_json(support/'scope.json',dict(files=SUPPORT,role='Reporting and verification support copied after fitting; prefit snapshot remains separate'))
    for name in ['time_runtime_full_tests_20261020_02.xml','compact_cli_verification_20261020.json']:
        shutil.copy2(ROOT/'audit'/name,r/'support_after_analysis'/name)
    shutil.copy2(ROOT/'LAPORAN_TIME_RUNTIME_20261020.html',q/'report/LAPORAN_TIME_RUNTIME_20261020.html')
    completed=[]
    for name in NEW:
        run=ROOT/'runs'/name;cat=run/'artifacts_sha256.json'
        if cat.exists() or (run/'run_complete.json').exists():raise ValueError('Refuse to overwrite sealed run '+name)
        records=[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*'))
            if p.is_file() and p.name not in ['artifacts_sha256.json','run_complete.json']]
        e.write_json(cat,records)
        status='SMOKE_TEST_ONLY' if 'SMOKE' in name else 'TIME_RUNTIME_COMPLETE_VERIFIED'
        e.write_json(run/'run_complete.json',dict(status=status,artifacts=len(records),catalog_sha256=e.sha256(cat),
            historical_holdout_used=False,independent_capture_validation=False,old_runs_preserved=True))
        for rec in records:check_record(run,rec)
        completed.append(dict(run=name,status=status,artifacts=len(records),catalog_sha256=e.sha256(cat)))
    entry=dict(date='2026-10-20',status='TIME_RUNTIME_COMPLETE_VERIFIED',quality_run=q.name,runtime_run=r.name,
        full_expert_fits=150,smoke_expert_fits=30,tests_passed=170,notebook_sections=102,notebook_sha256=e.sha256(notebook),
        main_classifier_unchanged=True,G_selected_balanced_seed42='baseline25',
        exact_runtime_full_input_rows=52684,xai_equivalence_games=300,report='LAPORAN_TIME_RUNTIME_20261020.html',
        combined_report='LAPORAN_GABUNGAN_TIME_RUNTIME_20261020.html',results='docs/HASIL_TIME_RUNTIME_20261020.md',
        replay_time_reduction_vs_native_certified_percent=summary['replay_time_reduction_vs_native_certified_percent'],
        replay_time_reduction_vs_native_full_percent=summary['replay_time_reduction_vs_native_full_percent'],
        completed_runs=completed,old_runs_verified=old,old_artifacts_unchanged=sum(z['files'] for z in old),
        independent_capture_evaluation=False,latest_delivery_contains_this_extension=False)
    state=json.loads((ROOT/'CURRENT_STATE.json').read_text(encoding='utf8'))
    state.update(status='EXPERIMENT_COMPLETE_TIME_RUNTIME_VERIFIED',updated_date='2026-10-20',research_time_runtime_20261020=entry,
                 latest_delivery_contains_time_runtime=False)
    e.write_json(ROOT/'CURRENT_STATE.json',state);e.write_json(ROOT/'audit/time_runtime_completion_20261020.json',entry)
    start=ROOT/'START_HERE.md';prefix=('**Terbaru: lanjutan fitur waktu dan runtime selesai terverifikasi, 20 Oktober 2026.** '
        '[Hasil dan batas klaim](docs/HASIL_TIME_RUNTIME_20261020.md), [laporan](LAPORAN_TIME_RUNTIME_20261020.html), '
        '[notebook 102 bagian](LAPORAN_GABUNGAN_TIME_RUNTIME_20261020.html). 150 fit penelitian tambahan, 170 tes lulus. '
        'Model WB25+multiscale60 Balanced tetap; fitur waktu memiliki kompromi; runtime kompak menjaga probabilitas/label dan SHAP identik. '
        'ZIP penyerahan lama belum memuat perluasan ini.\n\n')
    start.write_text(prefix+start.read_text(encoding='utf8'),encoding='utf8')
    print(json.dumps(entry,indent=2),flush=True)
