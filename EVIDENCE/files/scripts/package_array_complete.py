"""Deliver the visually reviewed manuscript revision; never overwrite older ZIPs."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paper/array_20261022_complete'
DEST=ROOT/'penyerahan'
STAGE=DEST/'ARRAY_HAFID_REVISI_LENGKAP_20261022'
ARCHIVE=DEST/'ARRAY_HAFID_REVISI_LENGKAP_20261022.zip'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def copy_tree(source,target,omit=()):
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in omit:
            out=target/path.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)


def main():
    assert not ARCHIVE.exists() and not STAGE.exists(),'Refusing to overwrite delivery'
    quality=json.loads((BASE/'quality_check.json').read_text())
    assert quality['status']=='PDFS_VERIFIED_ALL_PAGES_REVIEWED' and quality['visual_review_all_pdf_pages']
    assert [x['pages'] for x in quality['pdfs']]==[21,16]
    for item in quality['pdfs']:assert sha(BASE/'elsarticle'/(item['name']+'.pdf'))==item['sha256']
    protected={
        'output/pdf/ARRAY_M_HAFID_RAMADHAN_SATRIA_MANDALA.pdf':'62544ac3065e1d33c3605cb6253b1ccda8af3be3ca1d43b2de8751dcad1e316b',
        'penyerahan/ARRAY_HAFID_ELSARTICLE.zip':'a6b63cb1c629b5b194f85c03fc244721c96aebd65b6821b481b4434317902f83',
        'penyerahan/DASHBOARD_HAFID.zip':'3d50f6ed47e18279dab3da52cbe4a7aa5811cad0ceb14442eab6371b4ba90903',
        'penyerahan/ARRAY_HAFID_LENGKAP.zip':'1ff5deea93614f27b15e08db751503035f813a67f7861fd58ae90a070b3b8d96',
    }
    prior=json.loads((BASE/'editorial_build.json').read_text())['previous_deliverables']
    for item in prior['pdfs']:protected[item['path']]=item['sha256']
    protected[prior['archive']['path']]=prior['archive']['sha256']
    for name,expected in protected.items():assert sha(ROOT/name)==expected,name
    STAGE.mkdir(parents=True)
    copy_tree(BASE/'elsarticle',STAGE/'elsarticle',('.aux','.out','.log','.blg'))
    copy_tree(BASE/'evidence',STAGE/'evidence')
    for name in ['README_MULAI_DI_SINI.md','CATATAN_PERUBAHAN.md','reproduce_added_figures.py']:
        shutil.copy2(BASE/name,STAGE/name)
    editorial=STAGE/'editorial';editorial.mkdir()
    for name in ['main_body.md','supplement_body.md','tables.json','references.json']:
        shutil.copy2(BASE/name,editorial/name)
    audit=STAGE/'audit';audit.mkdir()
    for name in ['quality_check.json','editorial_build.json','template_inventory.json']:
        shutil.copy2(BASE/name,audit/name)
    for name in ['main.log','main.blg','supplement.log']:
        shutil.copy2(BASE/'elsarticle'/name,audit/name)
    helper=audit/'project_script_snapshot';helper.mkdir()
    for name in ['build_array_complete.py','verify_array_complete.py','build_array_short.py','build_array_manuscript.py','package_array_complete.py']:
        shutil.copy2(ROOT/'scripts'/name,helper/name)
    # Test the standalone figure script from the extracted package and isolated Python.
    out=ROOT/'tmp/array_complete_standalone_figures'
    run=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-I','-B','reproduce_added_figures.py','--output-dir',str(out)],
                       cwd=STAGE,capture_output=True,text=True,timeout=50)
    assert run.returncode==0,run.stdout+'\n'+run.stderr
    for name in ['figure5_confusion.png','figure6_group_importance.png']:
        assert sha(out/name)==sha(STAGE/'elsarticle/figures'/name),name
    figure_check=dict(status='PASS',isolated_python=True,package_only_data=True,training=False,model_load=False,
                      exact_png_matches=2,script='reproduce_added_figures.py')
    (audit/'standalone_figure_reproduction.json').write_text(json.dumps(figure_check,indent=2)+'\n')
    (audit/'preserved_deliverables.json').write_text(json.dumps(protected,indent=2)+'\n')
    records=[dict(path=p.relative_to(STAGE).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(STAGE.rglob('*')) if p.is_file()]
    manifest=json.dumps(records,indent=2)+'\n'
    (STAGE/'MANIFEST_SHA256.json').write_text(manifest)
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for rec in records:z.write(STAGE/rec['path'],rec['path'])
        z.writestr('MANIFEST_SHA256.json',manifest)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={r['path'] for r in records}|{'MANIFEST_SHA256.json'}
        for rec in records:
            content=z.read(rec['path']);assert len(content)==rec['bytes'] and hashlib.sha256(content).hexdigest()==rec['sha256']
    finals=[]
    for stem,name in [('main','ARRAY_HAFID_REVISI_LENGKAP_20261022.pdf'),('supplement','ARRAY_HAFID_SUPPLEMENT_LENGKAP_20261022.pdf')]:
        target=ROOT/'output/pdf'/name;assert not target.exists()
        shutil.copy2(BASE/'elsarticle'/(stem+'.pdf'),target)
        item=next(r for r in quality['pdfs'] if r['name']==stem)
        finals.append(dict(path=target.relative_to(ROOT).as_posix(),pages=item['pages'],sha256=sha(target),
                           tables=item['tables'],figures=item['figures']))
    archive=dict(path=ARCHIVE.relative_to(ROOT).as_posix(),bytes=ARCHIVE.stat().st_size,entries=len(records)+1,
                 sha256=sha(ARCHIVE),crc_passed=True,all_member_sha256_passed=True)
    result=dict(status='ALL_DELIVERY_CHECKS_PASSED',date='2026-10-22',pdfs=finals,archive=archive,
                numeric_checks=quality['numeric_checks'],visual_review_all_pdf_pages=True,
                standalone_figure_reproduction=figure_check,previous_files_verified_unchanged=len(protected),
                original_font_pt=12,original_margins=True,new_training=False,new_model_inference=False,
                new_game_execution=False,submission_performed=False)
    (DEST/'ARRAY_HAFID_REVISI_LENGKAP_20261022.verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
