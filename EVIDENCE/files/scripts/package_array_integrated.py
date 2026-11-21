"""Package the integrated review draft, source template and selected evidence."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paper/array_20261022_integrated'
DEST=ROOT/'penyerahan'
NAME='ARRAY_HAFID_ALUR_LENGKAP_20261022'
STAGE=DEST/NAME
ARCHIVE=DEST/(NAME+'.zip')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def copy_tree(source,target,omit=()):
    for p in source.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in omit:
            out=target/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)


def main():
    assert not ARCHIVE.exists() and not STAGE.exists(),'Never overwrite a delivered revision'
    q=json.loads((BASE/'quality_check.json').read_text())
    assert q['status']=='PDFS_VERIFIED_ALL_PAGES_REVIEWED' and q['visual_review_all_pdf_pages']
    assert [x['pages'] for x in q['pdfs']]==[23,22] and q['numeric_checks']==691
    build=json.loads((BASE/'editorial_build.json').read_text())
    protected={}
    for file in ['ARRAY_HAFID_REVISI_LENGKAP_20261022','ARRAY_HAFID_REVISI_MAKS20_20261022']:
        record=json.loads((DEST/(file+'.verification.json')).read_text())
        for p in record['pdfs']:protected[p['path']]=p['sha256']
        protected[record['archive']['path']]=record['archive']['sha256']
    protected.update({
        'output/pdf/ARRAY_M_HAFID_RAMADHAN_SATRIA_MANDALA.pdf':'62544ac3065e1d33c3605cb6253b1ccda8af3be3ca1d43b2de8751dcad1e316b',
        'penyerahan/ARRAY_HAFID_ELSARTICLE.zip':'a6b63cb1c629b5b194f85c03fc244721c96aebd65b6821b481b4434317902f83',
        'penyerahan/DASHBOARD_HAFID.zip':'3d50f6ed47e18279dab3da52cbe4a7aa5811cad0ceb14442eab6371b4ba90903',
        'penyerahan/ARRAY_HAFID_LENGKAP.zip':'1ff5deea93614f27b15e08db751503035f813a67f7861fd58ae90a070b3b8d96'})
    for name,digest in protected.items():assert sha(ROOT/name)==digest,name
    for r in build['source_records']:
        assert sha(ROOT/r['path'])==r['sha256'] and sha(ROOT/r['catalog'])==r['catalog_sha256']
    for r in q['pdfs']:assert sha(BASE/'elsarticle'/(r['name']+'.pdf'))==r['sha256']
    for r in json.loads((BASE/'template_inventory.json').read_text())['files']:
        assert sha(BASE/r['path'])==r['sha256']
    # Every figure and bibliography reference resolves relative to elsarticle.
    for stem in ['main','supplement']:
        text=(BASE/'elsarticle'/(stem+'.tex')).read_text(encoding='utf-8')
        for p in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',text):assert (BASE/'elsarticle'/p).is_file(),p
    STAGE.mkdir(parents=True)
    copy_tree(BASE/'elsarticle',STAGE/'elsarticle',('.aux','.out','.log','.blg'))
    copy_tree(BASE/'evidence',STAGE/'evidence')
    for name in ['README_MULAI_DI_SINI.md','CATATAN_PERUBAHAN.md','ALUR_DAN_CAKUPAN.md',
                 'reproduce_added_figures.py','reproduce_initial_figures.py']:
        shutil.copy2(BASE/name,STAGE/name)
    editorial=STAGE/'editorial';editorial.mkdir()
    for name in ['main_body.md','supplement_body.md','tables.json','references.json']:shutil.copy2(BASE/name,editorial/name)
    audit=STAGE/'audit';audit.mkdir()
    for name in ['quality_check.json','editorial_build.json','template_inventory.json']:shutil.copy2(BASE/name,audit/name)
    for name in ['main.log','main.blg','supplement.log']:shutil.copy2(BASE/'elsarticle'/name,audit/name)
    helpers=audit/'project_script_snapshot';helpers.mkdir()
    for name in ['build_array_integrated.py','verify_array_integrated.py','finalize_array_integrated.py',
                 'package_array_integrated.py','build_array_complete.py','verify_array_complete.py',
                 'build_array_short.py','build_array_manuscript.py']:
        shutil.copy2(ROOT/'scripts'/name,helpers/name)
    reproduced=[]
    for script,sub,paths in [
        ('reproduce_added_figures.py','added',['figure5_confusion.png','figure6_group_importance.png']),
        ('reproduce_initial_figures.py','initial',['figures/figure0_study_flow.png','figures/figure8_historical_features.png','figures/figure9_historical_shap.png'])]:
        out=ROOT/'tmp/array_integrated_standalone_figures'/sub
        run=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-I','-B',script,'--output-dir',str(out)],
                           cwd=STAGE,capture_output=True,text=True,timeout=55)
        assert run.returncode==0,run.stdout+'\n'+run.stderr
        for name in paths:
            assert sha(out/name)==sha(STAGE/'elsarticle/figures'/Path(name).name),name
            reproduced.append(Path(name).name)
    reproduction=dict(status='PASS',isolated_python=True,package_only_data=True,training=False,model_load=False,
                      exact_png_matches=len(reproduced),figures=reproduced,
                      scripts=['reproduce_added_figures.py','reproduce_initial_figures.py'])
    (audit/'standalone_figure_reproduction.json').write_text(json.dumps(reproduction,indent=2)+'\n')
    (audit/'preserved_deliverables.json').write_text(json.dumps(protected,indent=2)+'\n')
    records=[dict(path=p.relative_to(STAGE).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(STAGE.rglob('*')) if p.is_file()]
    manifest=json.dumps(records,indent=2)+'\n'
    (STAGE/'MANIFEST_SHA256.json').write_text(manifest)
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for r in records:z.write(STAGE/r['path'],r['path'])
        z.writestr('MANIFEST_SHA256.json',manifest)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={r['path'] for r in records}|{'MANIFEST_SHA256.json'}
        for r in records:
            b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    finals=[]
    for stem,name in [('main','ARRAY_HAFID_ALUR_LENGKAP_20261022.pdf'),('supplement','ARRAY_HAFID_SUPPLEMENT_ALUR_LENGKAP_20261022.pdf')]:
        target=ROOT/'output/pdf'/name;assert not target.exists()
        shutil.copy2(BASE/'elsarticle'/(stem+'.pdf'),target)
        r=next(x for x in q['pdfs'] if x['name']==stem)
        finals.append(dict(path=target.relative_to(ROOT).as_posix(),pages=r['pages'],sha256=sha(target),tables=r['tables'],figures=r['figures']))
    archive=dict(path=ARCHIVE.relative_to(ROOT).as_posix(),bytes=ARCHIVE.stat().st_size,entries=len(records)+1,
                 sha256=sha(ARCHIVE),crc_passed=True,all_member_sha256_passed=True)
    result=dict(status='ALL_DELIVERY_CHECKS_PASSED',date='2026-10-22',revision='INTEGRATED_INITIAL_AND_EXTENSION_STUDY',
                pdfs=finals,archive=archive,numeric_checks=q['numeric_checks'],visual_review_all_pdf_pages=True,
                standalone_figure_reproduction=reproduction,previous_files_verified_unchanged=len(protected),
                original_font_pt=12,original_margins=True,new_training=False,new_model_inference=False,
                new_game_execution=False,submission_performed=False)
    (DEST/(NAME+'.verification.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
