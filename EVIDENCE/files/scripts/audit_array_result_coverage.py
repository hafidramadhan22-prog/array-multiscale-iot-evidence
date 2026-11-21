"""Read-only evidence/PDF audit. Writes a separate coverage report, not a paper."""
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'paper/array_20261022_20pages'
OUT = ROOT / 'audit/array_coverage_20261022'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(name):
    with (BASE / 'evidence' / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = json.loads((BASE / 'tables.json').read_text())
    evidence = json.loads((BASE / 'evidence/evidence.json').read_text())
    for item in evidence['source_records']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    checks = []

    def compare(actual, expected, context, digits=4):
        assert f'{float(actual):.{digits}f}' == f'{float(expected):.{digits}f}', (context, actual, expected)
        checks.append(context)

    quality = rows('quality_multiscale_all_treatments.csv')
    views = ['baseline25', 'baseline29', 'multiscale60', 'legacy25_multiscale60', 'legacy25_multiscale_directional96']
    for row, view in zip(tables['main_quality']['rows'], views):
        source = next(r for r in quality if r['view'] == view and r['case'] == 'balanced')
        for i, key in enumerate(['macro_f1_mean', 'macro_f1', 'recall_Port_Scanning', 'recall_Ransomware'], 1):
            compare(row[i], 100 * float(source[key]), 'main_quality/' + view + '/' + key)
    methods = ['decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'soft_voting', 'weighted_voting']
    for row, method in zip(tables['expert_controls']['rows'], methods):
        source = next(r for r in rows('same_feature_controls.csv') if r['case'] == 'balanced' and r['method'] == method)
        for i, key in enumerate(['macro_f1_mean', 'ddos_f2_mean', 'macro_f1', 'recall_Port_Scanning'], 1):
            compare(row[i], 100 * float(source[key]), 'expert_controls/' + method + '/' + key)
    for row, method in zip(tables['faithfulness']['rows'], ['logistic_regression', 'random_forest', 'weighted_voting']):
        source = next(r for r in rows('faithfulness.csv') if r['method'] == method)
        for i, key in enumerate(['deletion_auc', 'insertion_auc', 'deletion_gain', 'insertion_gain'], 1):
            compare(row[i], source[key], 'faithfulness/' + method + '/' + key)
    policies = ['native_shared', 'native_certified', 'compact_shared', 'compact_certified', 'logistic_regression', 'random_forest']
    for row, policy in zip(tables['runtime_short']['rows'], policies):
        source = next(r for r in rows('runtime_summary.csv') if r['Policy'] == policy)
        for i, key in enumerate(['Batch1 ms', 'Batch1000 ms', 'Replay mean s', 'Delta RSS feature MiB', 'Delta RSS replay MiB'], 1):
            compare(row[i], source[key], 'runtime/' + policy + '/' + key)
    time_rows = rows('quality_time_extension.csv')
    for table_row, source in zip(tables['timing_quality']['rows'], [time_rows[i] for i in [0, 2, 3, 4]]):
        for i, key in enumerate(list(source)[1:], 1):
            compare(table_row[i], source[key], 'timing/' + source['Model'] + '/' + key)
    with (ROOT / 'reports/REKOMENDASI_20261015_01/ranking_all_six.csv').open(encoding='utf-8-sig', newline='') as stream:
        historical = list(csv.DictReader(stream))
    for table_row, source in zip(tables['historical_six']['rows'], historical):
        assert table_row[1] == source['config']
        for i, key, multiplier, digits in [(2, 'macro_f1', 100, 4), (3, 'latency_batch_ms', 1, 4),
                                            (4, 'delta_peak_mib', 1, 4), (5, 'tradeoff_score', 1, 6)]:
            compare(table_row[i], multiplier * float(source[key]), 'historical/' + source['config'] + '/' + key, digits)
    pdf_records = []
    page_texts = {}
    deliver = json.loads((ROOT / 'penyerahan/ARRAY_HAFID_REVISI_MAKS20_20261022.verification.json').read_text())
    for record, stem in zip(deliver['pdfs'], ['main', 'supplement']):
        path = ROOT / record['path']
        assert sha(path) == record['sha256'] == sha(BASE / 'elsarticle' / (stem + '.pdf'))
        doc = pymupdf.open(path)
        assert len(doc) == record['pages']
        captions = []
        page_texts[stem] = '\n'.join(p.get_text() for p in doc)
        for i, page in enumerate(doc):
            for line in page.get_text().splitlines():
                if re.match(r'^(Table|Figure) (?:S)?\d+:', line):
                    captions.append(dict(page=i + 1, caption=line))
        pdf_records.append(dict(path=record['path'], sha256=record['sha256'], pages=len(doc), captions=captions))
    figures = []
    for path in sorted((BASE / 'elsarticle/figures').glob('*.pdf')):
        assert sha(path) == sha(ROOT / 'paper/array_20261021/elsarticle/figures' / path.name)
        figures.append(dict(name=path.name, sha256=sha(path), unchanged_since_39_page_version=True))
    # Verify every ZIP member; no model loading or experiment execution.
    with zipfile.ZipFile(ROOT / deliver['archive']['path']) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST_SHA256.json'))
        for item in manifest:
            assert hashlib.sha256(archive.read(item['path'])).hexdigest() == item['sha256']
        csv_paths = [name for name in archive.namelist() if name.startswith('evidence/') and name.endswith('.csv')]
    expected_csv = {'evidence/' + path.name for path in (BASE / 'evidence').glob('*.csv')}
    assert set(csv_paths) == expected_csv
    # Presence in the full historical archive is checked from its central directory only.
    full_archive = ROOT / 'penyerahan/TA_HAFID_FINAL.zip'
    with zipfile.ZipFile(full_archive) as archive:
        names = archive.namelist()
        archive_history = {key: any(key in name for name in names) for key in [
            'LIFT_ID_FULL_20261018_01', 'E4_FULL_20261019_01', 'MPED_FULL_20261019_04',
            'SBFE_FULL_20261019_01', 'PORT_FUSION_FULL_20261020_01']}
    assert all(archive_history.values())
    report = dict(date='2026-10-22', status='AUDIT_PASSED_WITH_PRESENTATION_GAPS',
                  all_historical_experiments_in_current_pdfs=False,
                  main_results_present=True, numeric_cells_matched=len(checks), numeric_mismatches=[],
                  source_records_sha256_verified=len(evidence['source_records']), pdfs=pdf_records,
                  figures=figures, source_csv_files_in_revision_zip=len(csv_paths),
                  zip_all_member_sha256_verified=True, full_archive_history_presence=archive_history,
                  full_archive_crc_rechecked=False, no_paper_changes=True, no_training=True)
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['pdfs', 'figures']}, indent=2))


if __name__ == '__main__':
    main()
