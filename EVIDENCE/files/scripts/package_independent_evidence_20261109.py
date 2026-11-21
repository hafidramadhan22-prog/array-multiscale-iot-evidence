"""Whitelist the scientific evidence, excluding access credentials and raw downloads."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/INDEPENDENT_CAPTURE_20261109'
RUN=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01'
ZIP=ROOT/'penyerahan/UJI_INDEPENDEN_CIC_TON_UNTUK_DOSPEM_20261109.zip'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    files=set(p for p in RUN.rglob('*') if p.is_file())
    for manifest in [RUN/'source_manifest.json', RUN/'execution_source_manifest_v2.json',
                     RUN/'ton_execution_source_manifest.json',RUN/'syn_resume_source_manifest.json',
                     RUN/'flow_preserving_execution/source_manifest.json',
                     RUN/'TON_flow_mapping/source_manifest.json',RUN/'TON_flow_mapping/fast_source_manifest.json']:
        x=json.loads(manifest.read_text('utf8'))
        for row in x if isinstance(x,list) else [x]:
            p=ROOT/row['path'];assert sha(p)==row['sha256'];files.add(p)
    report_files='''all_external_features.csv all_policy_metrics_verified.csv
all_predictions_verified.csv primary_policy_metrics_verified.csv dataset_summaries.csv
examples_CIC_Normal3.csv examples_TON_Ransom1.csv false_alarm_predicted_classes.csv
extractor_alignment_audit.json extractor_check_tshark.csv old_flame_tshark_1100.csv
independent_verification.json label_scope_and_execution_notes.json
cic_download_manifest.json lfs_download_manifest.json lfs_plan_all.json
mirror_metadata.json mirror_file_tree.json mirror_DDoS-SYN_Flood_files.json mirror_Recon-PortScan_files.json
notebook_execution.json model_replay_audit.json pdf_qa.json
TON_label_coverage.csv rule_changes.csv CIC_scenario_view.csv expert_CIC_Normal_view.csv
Normal_FPR_view.csv Ransomware_detection_view.csv rule_changes_view.csv TON_coverage_view.csv
trace_examples_CIC_Normal3.csv trace_examples_TON_Ransom1.csv tshark_field_gate.json
README_extracted.txt README_PCAP_extracted.txt README_Victims_List_extracted.txt
CIC_main_README_extracted.txt SqlInjection_capinfos.txt'''.split()
    for name in report_files:files.add(OUT/name)
    files.update(OUT.glob('figures/*.png'))
    # These records are added after a test from a separate extraction, when available.
    for name in ['portable_review_audit.json','portable_test_results.xml','portable_tests.txt']:
        if (OUT/name).exists():files.add(OUT/name)
    files.update(ROOT/rel for rel in [
        'docs/HASIL_UJI_INDEPENDEN_20261109.md','docs/INDEPENDENT_CAPTURE_PROTOCOL_20261109.md',
        'docs/INDEPENDENT_TON_FLOW_LABEL_ADDENDUM_20261109.md',
        'notebooks/independent_20261109/UJI_INDEPENDEN_HASIL_DAN_PREDIKSI_20261109.ipynb',
        'output/pdf/UJI_INDEPENDEN_CIC_TON_HASIL_DAN_EVIDENCE_20261109.pdf','requirements.lock.txt',
        'scripts/audit_independent_results_20261109.py','scripts/replay_independent_predictions_20261109.py',
        'scripts/build_independent_views_20261109.py','scripts/execute_independent_notebook_20261109.py',
        'scripts/build_independent_report_20261109.py','scripts/package_independent_evidence_20261109.py',
        'scripts/verify_independent_package_20261109.py','scripts/check_independent_portability_20261109.py',
        'tests/test_independent_capture_bridge_20261109.py','tests/test_ton_interval_mapping_20261109.py',
        'tests/test_flow_preserving_capture_20261109.py','tests/test_independent_replay_gate_20261109.py'])
    # Include the local Python dependency closure, never execute acquisition scripts.
    queue=[p for p in files if p.suffix=='.py' and 'source' not in p.parts]
    seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text('utf-8-sig'))):
            names=([x.name for x in node.names] if isinstance(node,ast.Import)
                   else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                for directory in ['src','scripts']:
                    dependency=ROOT/directory/(name.split('.')[0]+'.py')
                    if dependency.exists() and dependency not in files:
                        files.add(dependency);queue.append(dependency)
    assert not any('register_cic_access' in p.name for p in files)
    for p in files:
        assert p.is_file(),str(p)
        assert p.resolve().is_relative_to(ROOT.resolve())
        assert not set(p.relative_to(ROOT).parts)&{'tmp','data','.git','.venv'}
        if p.suffix in ['.json','.jsonl','.md','.txt','.py','.html']:
            content=p.read_text('utf-8-sig').lower()
            for secret in ['x-amz'+'-signature','x-goog'+'-signature','set'+'-cookie:','bearer'+' hf_']:
                assert secret not in content,(p,secret)
    readme='''# Paket uji independen untuk dospem - 9 November 2026

Mulai dari output/pdf/UJI_INDEPENDEN_CIC_TON_HASIL_DAN_EVIDENCE_20261109.pdf.
Hasil: target FPR <8% BELUM TERCAPAI. Kandidat 638/2000 (31,90%) CIC Normal,
377/2000 (18,85%) TON Normal terpetakan. Pada TON, 1657/2000 Ransomware
terpetakan diprediksi Normal. Tidak ada training/tuning baru.

Notebook notebooks/independent_20261109/UJI_INDEPENDEN_HASIL_DAN_PREDIKSI_20261109.ipynb
sudah berisi output nyata (18 sel kode, 16 gambar). CSV di reports/ memuat
27000 keputusan, probabilitas, 9000 input, label scope dan distribusi kesalahan.
Penjelasan metode, caveat label dan perintah ada di docs/HASIL_UJI_INDEPENDEN_20261109.md.

Pemeriksaan offline dari root hasil ekstraksi (Python 3.11):
  python scripts/verify_independent_package_20261109.py

Untuk inferensi ulang, pasang requirements.lock.txt di environment terisolasi:
  python -B scripts/replay_independent_predictions_20261109.py --output review/predictions.csv --audit review/replay.json
  python -B scripts/audit_independent_results_20261109.py --phase audit
  python -B scripts/execute_independent_notebook_20261109.py
Jangan menjalankan phase seal ulang; run sudah tersegel.

Prepared-feature replay dan audit label berjalan mandiri tanpa raw download.
Raw PCAP/23CSV lengkap tidak disalin karena ukuran; provenance/ukuran/hash
disertakan. Beberapa PCAP turunan dan packet-index arrays tetap disertakan
agar katalog run utuh. Raw pipeline source adalah catatan eksekusi historis;
pengulangan dari raw memerlukan dataset terpisah, Wireshark4.4.9, serta gate
asal model dari arsip penelitian historis. Cookie, formulir registrasi dan
URL unduhan bertanda tangan tidak disertakan. Tidak ada upload/submission.

CIC attack file memakai label SKENARIO, bukan ground truth serangan tiap paket.
TON memakai label flow turunan pada subset terpetakan, bukan label paket asli.
Jangan menghitung satu macro-F1 gabungan lintas CIC dan TON. Jangan mengartikan
0,14% pada trace lama sebagai FPR independen. Guard mengubah predictor lama;
jaminan prediction preservation/runtime/XAI lama tidak otomatis berlaku.

PACKAGE_MANIFEST.json mencatat SHA/ukuran seluruh member kecuali dirinya.
Checksum ZIP disediakan terpisah. Paket berisi bukti untuk diperiksa, bukan
persetujuan akhir penulis atau naskah jurnal yang sudah diubah.
'''
    records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    records.append(dict(path='BACA_PERTAMA_UJI_INDEPENDEN.md',bytes=len(readme.encode()),sha256=hashlib.sha256(readme.encode()).hexdigest()))
    manifest=dict(format=1,scope='independent frozen-model evidence, prepared-feature replay and label audit',
                  no_training=True,private_access_data_excluded=True,files=records)
    ZIP.parent.mkdir(exist_ok=True);partial=ZIP.with_suffix('.zip.tmp')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('BACA_PERTAMA_UJI_INDEPENDEN.md',readme)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    partial.replace(ZIP)
    digest=sha(ZIP);ZIP.with_suffix('.zip.sha256').write_text(digest+'  '+ZIP.name+'\n',encoding='ascii')
    result=dict(path=ZIP.relative_to(ROOT).as_posix(),bytes=ZIP.stat().st_size,sha256=digest,
                files=len(records)+1,raw_external_downloads_included=False,full_sealed_run_included=True)
    (OUT/'delivery_archive.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
