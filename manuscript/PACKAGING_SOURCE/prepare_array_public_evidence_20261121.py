from pathlib import Path
import zipfile,json,hashlib,shutil,re

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'paper/array_20261121_quality_control_github'
DEST=WORK/'repository'
ARCHIVE=ROOT/'penyerahan/ARRAY_HAFID_REPRODUCIBILITY_GNB_REVISION_20261119.zip'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 evidence=DEST/'EVIDENCE'
 shutil.copytree(ROOT/'paper/array_20261121_gnb_focus/latex/EVIDENCE',evidence,dirs_exist_ok=True)
 runs=['ARRAY_GNB_FULL_20261119_01','ARRAY_SVM_EVALUATION_20261118_01','ARRAY_KNN_FULL_20261119_01','ARRAY_FACTORIAL_FULL_20261103_01','ARRAY_HISTORY_FULL_20261103_01','ARRAY_BUFFER_FULL_20261103_01','ARRAY_SERVING_ABLATION_20261117_01','ARRAY_RUNTIME_FULL_20261103_01','EXACT_RUNTIME_20261020_01','ARRAY_XAI_FULL_20261103_01','XAI_MULTISCALE_FULL_20261020_01']
 allowed={'.json','.csv','.npz','.py','.txt','.gz'}
 with zipfile.ZipFile(ARCHIVE) as z:
  for i in z.infolist():
   n=i.filename;p=Path(n)
   take=(any(n.startswith('runs/'+r+'/') for r in runs) and p.suffix in allowed and '/support_after_fit/' not in n)
   take|=(n.startswith(('src/','research/','configs/')) and p.suffix in {'.py','.json','.txt'})
   take|=(n.startswith('scripts/') and p.suffix=='.py' and any(s in p.name for s in ['array','multiscale','xai','exact','independent','fpr','guard']))
   take|=n in ['requirements.lock.txt','requirements.paper-extra.txt']
   take|=n.startswith('reports/INDEPENDENT_CAPTURE_20261109/') and p.suffix in {'.csv','.json'} and 'features' not in n
   take|=n.startswith('reports/FPR_SOLUTION_20261109/') and p.suffix in {'.csv','.json'}
   take|=n.startswith('runs/INDEPENDENT_CAPTURE_20261109_01/TON_flow_mapping/') and (n.endswith('_selected_targets.csv') or p.suffix=='.json')
   if take:
    assert not p.is_absolute() and '..' not in p.parts
    out=evidence/'files'/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(z.read(n))
 # No raw packets, extracted payloads, executable programs, serialized estimators or private correspondence.
 files=[];findings=[]
 for p in sorted(evidence.rglob('*')):
  if not p.is_file():continue
  b=p.read_bytes();rel=p.relative_to(DEST).as_posix()
  if p.suffix in {'.py','.json','.csv','.txt','.md'}:
   text=b.decode('utf-8',errors='replace')
   for pattern in [r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',r'AKIA[0-9A-Z]{16}']:
    if re.search(pattern,text):findings.append(rel)
  assert p.suffix.lower() not in {'.pcap','.pcapng','.joblib','.pkl','.exe','.dll'}
  files.append({'path':rel,'bytes':len(b),'sha256':sha(b)})
 assert not findings,findings
 def group(n):
  if any(x in n.lower() for x in ['independent','fpr_','guard','ton_','benign','flag_format']):return 'R4'
  if any(x in n.lower() for x in ['xai','coalition','shap','faithfulness','importance','background_sensitivity']):return 'R3'
  if any(x in n.lower() for x in ['runtime','serving','replay','latency','memory','equivalence','rss_']):return 'R2'
  return 'R1'
 for r,title in [('R1','Predictive evidence'),('R2','Serving evidence'),('R3','Explanation evidence'),('R4','Diagnostic evidence')]:
  d=DEST/r;d.mkdir(exist_ok=True);rows=[x for x in files if group(x['path'])==r]
  (d/'index.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
  (d/'README.md').write_text('# '+title+'\n\n`index.json` lists relative repository paths and SHA256 checksums. Historical paths are preserved under EVIDENCE/files so the saved-results verifier can resolve them. These are archived experiments, not newly run validations.\n',encoding='utf-8')
 (DEST/'requirements-verification.txt').write_text('numpy==2.2.6\npandas==2.2.3\nscikit-learn==1.6.1\n',encoding='utf-8')
 (DEST/'PUBLICATION_AUDIT.json').write_text(json.dumps({'source_archive_sha256':sha(ARCHIVE.read_bytes()),'files':len(files),'bytes':sum(x['bytes'] for x in files),'credential_pattern_matches':findings,'excluded':'Raw PCAPs, packet arrays, full third-party datasets, serialized estimators, private correspondence and unrelated working history. Original local archive unchanged.','scope':'Saved-result audit, not full end-to-end raw-data reconstruction.'},indent=2),encoding='utf-8')
 (DEST/'SHA256_MANIFEST.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
 print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files),'largest':sorted(files,key=lambda x:x['bytes'],reverse=True)[:5]},indent=2))
if __name__=='__main__':main()
