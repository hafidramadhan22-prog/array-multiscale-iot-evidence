"""Package the reviewed 19-page revision; retain all previous deliveries."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'paper/array_20261022_20pages'
DEST = ROOT / 'penyerahan'
STAGE = DEST / 'ARRAY_HAFID_REVISI_MAKS20_20261022'
ZIP = DEST / 'ARRAY_HAFID_REVISI_MAKS20_20261022.zip'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_tree(src, dst, omit=()):
    for path in src.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in omit:
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def main():
    assert not ZIP.exists() and not STAGE.exists(), 'Refusing to overwrite delivery'
    quality = json.loads((BASE / 'quality_check.json').read_text())
    assert quality['status'] == 'PDFS_VERIFIED_ALL_PAGES_REVIEWED'
    assert quality['visual_review_all_pdf_pages']
    assert [x['pages'] for x in quality['pdfs']] == [19, 9]
    for item in quality['pdfs']:
        assert sha(BASE / 'elsarticle' / (item['name'] + '.pdf')) == item['sha256']
    protected = {
        'output/pdf/ARRAY_M_HAFID_RAMADHAN_SATRIA_MANDALA.pdf': '62544ac3065e1d33c3605cb6253b1ccda8af3be3ca1d43b2de8751dcad1e316b',
        'penyerahan/ARRAY_HAFID_ELSARTICLE.zip': 'a6b63cb1c629b5b194f85c03fc244721c96aebd65b6821b481b4434317902f83',
        'penyerahan/DASHBOARD_HAFID.zip': '3d50f6ed47e18279dab3da52cbe4a7aa5811cad0ceb14442eab6371b4ba90903',
        'penyerahan/ARRAY_HAFID_LENGKAP.zip': '1ff5deea93614f27b15e08db751503035f813a67f7861fd58ae90a070b3b8d96',
    }
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, name
    STAGE.mkdir(parents=True)
    copy_tree(BASE / 'elsarticle', STAGE / 'elsarticle', ('.aux', '.out', '.log', '.blg'))
    copy_tree(BASE / 'evidence', STAGE / 'evidence')
    for name in ['README_MULAI_DI_SINI.md', 'CATATAN_REVISI_DAN_BATAS_HALAMAN.md']:
        shutil.copy2(BASE / name, STAGE / name)
    audit = STAGE / 'audit'
    audit.mkdir()
    for name in ['quality_check.json', 'editorial_build.json', 'template_inventory.json']:
        shutil.copy2(BASE / name, audit / name)
    for name in ['main.log', 'main.blg', 'supplement.log']:
        shutil.copy2(BASE / 'elsarticle' / name, audit / name)
    (audit / 'preserved_deliverables.json').write_text(json.dumps(protected, indent=2) + '\n')
    editorial = STAGE / 'editorial'
    editorial.mkdir()
    for name in ['main_body.md', 'supplement_body.md', 'tables.json']:
        shutil.copy2(BASE / name, editorial / name)
    records = [dict(path=p.relative_to(STAGE).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(STAGE.rglob('*')) if p.is_file()]
    manifest = json.dumps(records, indent=2) + '\n'
    (STAGE / 'MANIFEST_SHA256.json').write_text(manifest)
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in records:
            archive.write(STAGE / item['path'], item['path'])
        archive.writestr('MANIFEST_SHA256.json', manifest)
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {x['path'] for x in records} | {'MANIFEST_SHA256.json'}
        for item in records:
            content = archive.read(item['path'])
            assert len(content) == item['bytes']
            assert hashlib.sha256(content).hexdigest() == item['sha256']
    finals = []
    for name, source, pages in [
        ('ARRAY_HAFID_REVISI_MAKS20_20261022.pdf', 'main.pdf', 19),
        ('ARRAY_HAFID_SUPPLEMENT_20261022.pdf', 'supplement.pdf', 9),
    ]:
        target = ROOT / 'output/pdf' / name
        assert not target.exists(), 'Refusing to overwrite PDF'
        shutil.copy2(BASE / 'elsarticle' / source, target)
        finals.append(dict(path=target.relative_to(ROOT).as_posix(), pages=pages, sha256=sha(target)))
    archive_record = dict(path=ZIP.relative_to(ROOT).as_posix(), bytes=ZIP.stat().st_size,
                          entries=len(records) + 1, sha256=sha(ZIP), crc_passed=True,
                          all_member_sha256_passed=True)
    result = dict(status='ALL_DELIVERY_CHECKS_PASSED', date='2026-10-22', pdfs=finals,
                  archive=archive_record, protected_files_unchanged=len(protected),
                  visual_review_all_pdf_pages=True, main_font_pt=12, original_margins=True,
                  scientific_results_changed=False, new_training=False, submission_performed=False)
    (DEST / 'ARRAY_HAFID_REVISI_MAKS20_20261022.verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
