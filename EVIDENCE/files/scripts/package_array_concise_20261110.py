"""Package the concise review and its unchanged scientific evidence."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,zipfile
from package_array_final_integration_20261110 import archive,sha,safe

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/ARRAY_CONCISE_20261110"
BASE=ROOT/"paper/array_20261110_concise"
PARENT=ROOT/"penyerahan/ARRAY_HAFID_REPRODUCIBILITY_GUARD_EXTERNAL_20261110.zip"
REVIEW=ROOT/"penyerahan/UNTUK_DOSPEM_ARRAY_RINGKAS_20261110.zip"
REPRO=ROOT/"penyerahan/ARRAY_HAFID_REPRODUCIBILITY_CONCISE_20261110.zip"
README="""# Paket review jurnal Array - versi ringkas 10 November 2026

Kirim tiga PDF:
1. Main manuscript25 halaman,7 tabel,5 gambar,30 referensi.
2. Supplementary93 halaman,S1-S104 dan9 gambar.
3. Catatan pemadatan dua halaman, termasuk peta pemindahan tabel/gambar.

Format main tetap satu kolom preprint12pt. Versi45 halaman tetap sejarah.
Tidak ada training, inference, tuning atau profiling baru.
Seluruh15 badan tabel main lama dan95 badan tabel supplementary lama tetap
tersedia identik. Tabel eksternal ringkas dihitung dari27.000 keputusan tersimpan.

Kontribusi utama:
- Konteks multiscale: gain temporal1,1425-1,1475 poin persentase pada ordering tetap.
- Eksekusi predictor acuan yang sama: mean replay saving41,75% pada satu laptop.
- Penjelasan: audit full-probability consistency dan donor-game sensitivity.

Guard g berbeda dari predictor acuan f dan tidak mewarisi kontrak execution/XAI.
FPR guard Normal V39/3860=1,0104%; blok ketiga1/697=0,1435% tetap within-trace.
External guard FPR31,90% CIC dan18,85% TON terpetakan; TON Ransomware1657/2000
diprediksi Normal. Tidak ada klaim generalisasi deteksi atau seluruhFPR<1%.
CIC attack memakai label skenario; TON label flow turunan dengan coverage terbatas.

Source editable: paper/array_20261110_concise/elsarticle/
Catatan: docs/ARRAY_CONCISE_REVIEW_20261110.md
Audit: reports/ARRAY_CONCISE_20261110/

Arsip reproduksi lengkap mempertahankan seluruh member evidence archive
sebelumnya, ditambah source/PDF/verifier versi ringkas. Prediksi, konfigurasi,
manifests, timing/RSS dan tiga notebook studi tetap tersedia. Fullraw dataset,
cookie/kredensial dan signed accessURLs tidak ditambahkan.
Root CURRENT_RELEASE.json menjelaskan versi aktif; metadata lama ada di provenance/.

Untuk memeriksa bukti tersimpan tanpa model loading, dari root archive dengan
environment Python3.11 terisolasi sesuai requirements.lock.txt:

    python -B scripts/verify_array_concise_20261110.py

Jangan menjalankan training driver atau phase seal untuk membaca hasil.
PDF dapat dibaca langsung. BuildTectonic:

    tectonic -X compile --keep-logs paper/array_20261110_concise/elsarticle/main.tex
    tectonic -X compile --keep-logs paper/array_20261110_concise/elsarticle/supplement.tex

Audit PDF memakai requirements.paper-extra.txt dan font Arial Windows.
Sumber builder/editorial juga disertakan, dengan parent source dan gateSHA.

Status: bahan review penulis/dospem, belum submission. Persetujuan final semua
penulis belum dikonfirmasi, tanpa klaim approval palsu. Funding/COI/CRediT
mengikuti data terkonfirmasi. Draft disclosure terkait dan salinan conference
tetap terpisah untuk penulis/editor, belum dikirim.
Tidak ada Turnitin atau jaminan skor AI; bantuanAI tetap diungkap secara faktual.
Tidak ada URL repository/DOI kode yang dikarang.
25 halaman bukan batas resmi atau jaminan diterimaArray.

manifest.json memuat size/SHA seluruh member selain dirinya; checksumZIP
tersedia di sidecar .sha256. Gunakan path ekstraksi singkat bila memungkinkan.
"""

def write(p,data):
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf8")

def main():
    verified=json.loads((OUT/"verification.json").read_text("utf8"))
    quality=json.loads((OUT/"pdf_quality.json").read_text("utf8"))
    assert verified["status"]=="PASS_CONDENSED_MANUSCRIPT_AND_SAVED_EVIDENCE"
    assert [(q["document"],q["pages"]) for q in quality]==[("main",25),("supplement",93),("changes",2)]
    for q in quality:
        assert q["visual_review"].startswith("PASS") and not q["bounds"]
        assert sha(ROOT/q["path"])==q["sha256"]
    assert sha(PARENT)=="90cda6d2d1e4731a26d6f883c5990de805bc1664eb854c47cab53e20df253a67"
    parent_info=dict(path=PARENT.relative_to(ROOT).as_posix(),sha256=sha(PARENT))
    local=set(p for p in BASE.rglob("*") if p.is_file() and p.suffix not in [".xdv"])
    local.update(ROOT.glob("scripts/*array_concise*20261110.py"))
    local.add(ROOT/"docs/ARRAY_CONCISE_REVIEW_20261110.md")
    local.update(p for p in OUT.rglob("*") if p.is_file() and p.name not in
                 ["package_verification.json","portable_verification.json","delivery_final.json"])
    for q in quality:local.add(ROOT/q["path"])
    with zipfile.ZipFile(PARENT) as z:
        raw=z.read("manifest.json");old=json.loads(raw)
        entries={}
        for row in old["files"]:
            name=safe(row["path"])
            if name in ["README_MULAI_DI_SINI.md","CURRENT_RELEASE.json"]:
                name="provenance/pre_concise_"+Path(name).name
            entries[name]=(z,row["path"],row)
        entries["provenance/pre_concise_manifest.json"]=raw
        entries["README_MULAI_DI_SINI.md"]=README.encode("utf8")
        release=dict(date="2026-11-10",edition="CONCISE",status="REVISED_FOR_AUTHOR_REVIEW_NOT_SUBMITTED",
                     source=BASE.relative_to(ROOT).as_posix()+"/elsarticle",
                     main_pages=25,supplement_pages=93,supplement_tables=104,
                     main_tables=7,main_figures=5,supplement_figures=9,
                     new_fits=0,new_inference=0,author_approval_confirmed=False,
                     external_detection_reliability_established=False)
        entries["CURRENT_RELEASE.json"]=json.dumps(release,indent=2).encode("utf8")
        for p in local:
            assert p.resolve().is_relative_to(ROOT.resolve())
            entries[p.relative_to(ROOT).as_posix()]=p
        repro=archive(REPRO,entries,[parent_info],[])
    print("Reproducibility archive written and every member checked",flush=True)
    review={"00_BACA_DULU.md":README.encode("utf8"),
            "04_CATATAN_REVISI.md":ROOT/"docs/ARRAY_CONCISE_REVIEW_20261110.md",
            "05_HIGHLIGHTS.txt":BASE/"elsarticle/highlights.txt"}
    for name,q in zip(["01_NASKAH_UTAMA_ARRAY_25_HALAMAN.pdf",
                       "02_SUPPLEMENT_ARRAY_S1_S104.pdf","03_CATATAN_PEMADATAN.pdf"],quality):
        review[name]=ROOT/q["path"]
    for p in (BASE/"elsarticle").rglob("*"):
        if p.is_file() and (p.suffix in [".tex",".bib",".cls",".bst",".txt"] or
                           (p.parent.name=="figures" and p.suffix==".pdf")):
            review["source/elsarticle/"+p.relative_to(BASE/"elsarticle").as_posix()]=p
    for fn in ["ARRAY_EDITOR_DISCLOSURE_20261110.md","ARRAY_RELATED_MATERIAL_MAP_20261110.md"]:
        review["editor_review/"+fn]=ROOT/"docs"/fn
    review["editor_review/RELATED_MANUSCRIPT_AGERS_EDAS_1571374892.pdf"]=ROOT/"output/pdf/AGERS_HAFID_EDAS_20261028.pdf"
    small=archive(REVIEW,review,[],[])
    print("Review archive written and checked",flush=True)
    target=Path(tempfile.mkdtemp(prefix="array_concise_",dir=ROOT/"tmp")).resolve()
    assert target.is_relative_to((ROOT/"tmp").resolve())
    with zipfile.ZipFile(REPRO) as z:
        for n in z.namelist():assert (target/n).resolve().is_relative_to(target)
        z.extractall("\\\\?\\"+str(target) if os.name=="nt" else target)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH",None)
    p=subprocess.run([sys.executable,"-B","scripts/verify_array_concise_20261110.py"],cwd=target,
                     env=env,capture_output=True,text=True,encoding="utf8")
    portable=dict(status="PASS" if p.returncode==0 else "FAIL",archive_sha256=repro["sha256"],
                  extraction=str(target),command="python -B scripts/verify_array_concise_20261110.py",
                  returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
                  new_fits=0,new_inference=0,models_deserialized=False)
    write(OUT/"portable_verification.json",portable)
    assert p.returncode==0,portable
    result=dict(status="COMPLETE_CONCISE_AUTHOR_REVIEW_PACKAGE",review=small,reproducibility=repro,
                pdfs=quality,parent=parent_info,portable_audit="reports/ARRAY_CONCISE_20261110/portable_verification.json",
                new_fits=0,new_inference=0,submission_performed=False,turnitin_performed=False,
                author_approval_confirmed=False)
    write(OUT/"package_verification.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ["pdfs"]},indent=2),flush=True)

if __name__=="__main__":
    main()
