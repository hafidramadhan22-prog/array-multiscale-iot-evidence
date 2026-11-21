from pathlib import Path
import json,hashlib,zipfile,shutil,re
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'paper/array_20261121_quality_control_github';R=W/'repository';L=W/'latex'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 # Keep unnecessary packet-extraction diagnostics local instead of redistributing them.
 for name in ['old_flame_tshark_1100.csv','extractor_check_tshark.csv','trace_examples_CIC_Normal3.csv','trace_examples_TON_Ransom1.csv']:
  p=R/'EVIDENCE/files/reports/INDEPENDENT_CAPTURE_20261109'/name
  if p.exists():
   dest=W/'excluded_publication'/name;dest.parent.mkdir(exist_ok=True);shutil.move(str(p),dest)
 files=[{'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted((R/'EVIDENCE').rglob('*')) if p.is_file()]
 (R/'SHA256_MANIFEST.json').write_text(json.dumps(files,indent=2),encoding='utf8')
 valid={x['path'] for x in files}
 for group in ['R1','R2','R3','R4']:
  p=R/group/'index.json';rows=json.loads(p.read_text());p.write_text(json.dumps([x for x in rows if x['path'] in valid],indent=2),encoding='utf8')
 audit=json.loads((R/'PUBLICATION_AUDIT.json').read_text());audit.update(files=len(files),bytes=sum(x['bytes'] for x in files));(R/'PUBLICATION_AUDIT.json').write_text(json.dumps(audit,indent=2))
 # Keep publication files separate from source supervisor DOCX, which is unchanged.
 for folder in ['output/pdf','penyerahan']: (ROOT/folder).mkdir(exist_ok=True)
 outputs={L/'main.pdf':ROOT/'output/pdf/ARRAY_HAFID_QC_GITHUB_20261121.pdf',L/'SUPPLEMENTARY/supplement.pdf':ROOT/'output/pdf/ARRAY_HAFID_SUPPLEMENT_QC_GITHUB_20261121.pdf'}
 for src,dest in outputs.items():shutil.copy2(src,dest)
 pack=W/'package';pack.mkdir(exist_ok=True)
 for p in L.rglob('*'):
  if p.is_file() and p.suffix not in {'.log','.blg','.aux','.bbl','.xdv'} and '__pycache__' not in p.parts:
   dest=pack/p.relative_to(L);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 (pack/'REVISION_LOG').mkdir(exist_ok=True)
 for n in ['editorial_changes.json'] :shutil.copy2(W/n,pack/'REVISION_LOG'/n)
 for n in ['VERIFICATION_RESULTS.json','EXTENDED_VERIFICATION.json']:shutil.copy2(R/n,pack/'REVISION_LOG'/n)
 (pack/'README.md').write_text('''# Array main manuscript and supplementary

Main document: `main.tex` (19-page PDF preview `main.pdf`). Companion: `SUPPLEMENTARY/supplement.tex` (4-page PDF preview).
The latest supervisor QUALITY_CONTROL_FINAL DOCX files dated 21 November are the prose authority. All seven main tables, five figures, five supplementary tables and four native equations are included. Availability paragraphs now link to the public R1–R4 repository and release qc-20261121. Numerical results were not changed.

Repository: https://github.com/hafidramadhan22-prog/array-multiscale-iot-evidence
Release: https://github.com/hafidramadhan22-prog/array-multiscale-iot-evidence/releases/tag/qc-20261121

Compile from this folder: `tectonic --untrusted main.tex`. Then enter SUPPLEMENTARY and run `tectonic --untrusted supplement.tex`. On Overleaf select main.tex as the main document; compile the supplementary separately. These are editable native LaTeX sources, not embedded manuscript PDF pages. Figure PDFs and their Python source/input tables are supplied.

The supervisor's latest supplementary prose is retained in full and occupies four pages at readable 10pt. It exceeds the earlier three-page target; no scientific evidence was silently removed to meet that older length. Confirm this updated length with the supervisor before submission.

The evidence repository excludes raw provider datasets and fitted estimator binaries. Its saved-results verifiers do not train, deserialize estimators or produce new classifier predictions. R1 preserves historical comparisons, including RF/SVM/KNN. Local raw and historical archives remain unchanged.

Packaging and verification scripts were prepared with ChatGPT/Codex assistance. This package is for author review, not a record of submission or final author approval.
''',encoding='utf8')
 # Public manuscript sources, figures and compile previews.
 shutil.copytree(pack,R/'manuscript',dirs_exist_ok=True)
 for filename in ['build_array_qc_github_20261121.py','prepare_array_public_evidence_20261121.py','finalize_array_qc_github_20261121.py']:
  d=pack/'PACKAGING_SOURCE';d.mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts'/filename,d/filename)
 manifest=[{'path':p.relative_to(pack).as_posix(),'sha256':digest(p)} for p in pack.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json']
 (pack/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 out=ROOT/'penyerahan/ARRAY_HAFID_LATEX_QC_GITHUB_20261121.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for p in pack.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(pack).as_posix())
 with zipfile.ZipFile(out) as z:
  assert z.testzip() is None
  assert all(hashlib.sha256(z.read(x['path'])).hexdigest()==x['sha256'] for x in manifest)
 evidencezip=ROOT/'penyerahan/ARRAY_R1_R4_PUBLIC_EVIDENCE_20261121.zip'
 with zipfile.ZipFile(evidencezip,'w',zipfile.ZIP_DEFLATED) as z:
  for p in R.rglob('*'):
   if p.is_file() and '.git' not in p.relative_to(R).parts:z.write(p,p.relative_to(R).as_posix())
 (W/'delivery.json').write_text(json.dumps({'repository':'https://github.com/hafidramadhan22-prog/array-multiscale-iot-evidence','latex_zip':str(out.relative_to(ROOT)),'latex_sha256':digest(out),'evidence_zip':str(evidencezip.relative_to(ROOT)),'evidence_sha256':digest(evidencezip),'main_pages':19,'supplement_pages':4,'evidence_files':len(files),'new_model_training':0},indent=2))
 print((W/'delivery.json').read_text())
if __name__=='__main__':main()
