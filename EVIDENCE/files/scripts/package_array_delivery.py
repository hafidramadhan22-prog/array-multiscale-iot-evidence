"""Package the reviewed source/demo without retraining or changing old archives."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'paper/array_20261021'
DEST=ROOT/'penyerahan';STAGE=DEST/'ARRAY_HAFID_20261021'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()

def copy_tree(source,dest,omit=()):
    for p in source.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in omit:continue
        q=dest/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)

def archive(name,files):
    assert not (DEST/name).exists(),'Refusing to overwrite an existing delivery ZIP'
    records=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(files.items())]
    path=DEST/name
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,p in sorted(files.items()):z.write(p,n)
        z.writestr('MANIFEST_SHA256.json',json.dumps(records,indent=2)+'\n')
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(files)|{'MANIFEST_SHA256.json'}
        for rec in records:
            body=z.read(rec['path']);assert len(body)==rec['bytes'] and hashlib.sha256(body).hexdigest()==rec['sha256']
    return dict(path=str(path.relative_to(ROOT)).replace('\\','/'),bytes=path.stat().st_size,entries=len(records)+1,
        sha256=sha(path),crc_passed=True,all_member_sha256_passed=True)

def main():
    quality=json.loads((BASE/'pdf_quality_check.json').read_text())
    assert quality['status']=='PDF_VERIFIED_ALL_PAGES_REVIEWED' and quality['visual_review_all_pages']
    assert sha(BASE/'elsarticle/main.pdf')==quality['pdf_sha256']
    gate=json.loads((ROOT/'dashboard/verification.json').read_text());assert gate['status']=='DASHBOARD_FUNCTIONALLY_VERIFIED'
    DEST.mkdir(exist_ok=True);STAGE.mkdir(exist_ok=True)
    copy_tree(BASE/'elsarticle',STAGE/'elsarticle',('.aux','.out','.log','.blg'))
    copy_tree(BASE/'evidence',STAGE/'evidence')
    copy_tree(ROOT/'research/array_20261021',STAGE/'literature')
    copy_tree(ROOT/'dashboard',STAGE/'dashboard')
    shutil.copy2(BASE/'README_MULAI_DI_SINI.md',STAGE/'README_MULAI_DI_SINI.md')
    audit=STAGE/'audit';audit.mkdir(exist_ok=True)
    for src,name in [(BASE/'pdf_quality_check.json','pdf_quality_check.json'),(BASE/'template_inventory.json','template_inventory.json'),
       (BASE/'elsarticle/main.log','latex_build.log'),(BASE/'elsarticle/main.blg','bibtex_build.log')]:shutil.copy2(src,audit/name)
    helper=audit/'scripts';helper.mkdir(exist_ok=True)
    for name in ['build_array_manuscript.py','verify_array_pdf.py','prepare_array_dashboard.py','check_dashboard_http.py','check_dashboard_dom.cjs','package_array_delivery.py']:
        shutil.copy2(ROOT/'scripts'/name,helper/name)
    # Check the staged standalone package in isolated Python from its own directory.
    # Imported app/core/assets must resolve within the package, not the original project.
    proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-I','-B','check_installation.py'],cwd=STAGE/'dashboard',capture_output=True,text=True,timeout=60)
    if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
    check=json.loads(proc.stdout);assert check['status']=='PASS' and check['isolated_python']==1
    (audit/'standalone_dashboard_check.json').write_text(json.dumps(check,indent=2)+'\n')
    build=dict(compiler='Tectonic 0.17.0 / XeTeX / BibTeX',class_version='elsarticle 3.4 (2024-04-04)',
        class_origin='Supplied dtx single class guard; original dtx/ins and licence retained',
        template_sha256=json.loads((BASE/'template_inventory.json').read_text())['sha256'],
        author_order=['M Hafid Ramadhan','Satria Mandala'],corresponding_author='Satria Mandala',
        corresponding_email='satriamandala@telkomuniversity.ac.id',journal='Array',
        manuscript_status='COMPLETE_REVIEW_DRAFT_NOT_SUBMITTED',pages=quality['pages'],abstract_words=176,
        scientific_results_changed=False,new_training=False,public_repository_assigned=False,
        array_guide_access='HTTP 403; journal-specific requirements need checking',dashboard_gate=gate)
    (audit/'build_notes.json').write_text(json.dumps(build,indent=2)+'\n')
    # Keep old deliverables byte-identical; verify the exact prior hashes.
    protected={
      'output/pdf/MANUSCRIPT_M_HAFID_RAMADHAN.pdf':'5b0e54cc3cb8f74aa5afe243753e30ef1dae0a752036fe8d93dce3e88d777c25',
      'penyerahan/MANUSKRIP_JURNAL_HAFID.zip':'6b92183d73d7065b7b45465741da56fd1719fc6f2b145ce1d754cb58aa8851ff',
      'penyerahan/TA_HAFID_FINAL.zip':'bb59f51be15f6821db03d437f8c1c963d0f4926e5add9621f90dab271c8fad57'}
    for name,expected in protected.items():assert sha(ROOT/name)==expected,name
    (audit/'preserved_deliverables.json').write_text(json.dumps(protected,indent=2)+'\n')
    final=ROOT/'output/pdf/ARRAY_M_HAFID_RAMADHAN_SATRIA_MANDALA.pdf'
    assert not final.exists(),'Refusing to overwrite previous final PDF'
    shutil.copy2(BASE/'elsarticle/main.pdf',final)
    allfiles={p.relative_to(STAGE).as_posix():p for p in STAGE.rglob('*') if p.is_file()}
    # The conventional source ZIP has the same elsarticle root as the user template.
    source={n:p for n,p in allfiles.items() if not n.startswith('dashboard/')}
    demo={n:p for n,p in allfiles.items() if n.startswith('dashboard/')}
    delivered=[archive('ARRAY_HAFID_ELSARTICLE.zip',source),archive('DASHBOARD_HAFID.zip',demo),archive('ARRAY_HAFID_LENGKAP.zip',allfiles)]
    result=dict(status='ALL_DELIVERY_CHECKS_PASSED',pdf=str(final.relative_to(ROOT)).replace('\\','/'),pdf_sha256=sha(final),
       pdf_pages=quality['pages'],archives=delivered,standalone_dashboard=check,protected_files_unchanged=len(protected),
       no_training=True,no_submission=True,dashboard_graphical_review='Stopped by user Escape; function/API/virtual DOM verified')
    (DEST/'ARRAY_HAFID.verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
