"""Record visual review and update handoff metadata after the verified delivery."""
from pathlib import Path
import argparse,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/ARRAY_CONCISE_20261110"

def read(p):return json.loads(p.read_text("utf8"))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def visual():
    q=read(OUT/"pdf_quality.json")
    assert [r["pages"] for r in q]==[25,93,2]
    for row in q:
        assert not row["bounds"] and sha(ROOT/row["path"])==row["sha256"]
        row["visual_review"]="PASS_ALL_PAGES_CONTACT_SHEETS_AND_CHANGED_PAGE_RECHECKS"
        row["reviewed_pages"]=list(range(1,row["pages"]+1))
    for name in ["main","supplement"]:
        log=(ROOT/"paper/array_20261110_concise/elsarticle"/(name+".log")).read_text("utf8")
        assert "Overfull" not in log and "undefined" not in log.lower()
    write(OUT/"pdf_quality.json",q)
    write(OUT/"visual_review.json",dict(status="PASS",total_pages=120,main_pages=25,
          supplement_pages=93,notes_pages=2,findings="No clipping, overlap, unresolved references or illegible new labels.",
          format="Unchanged one-column elsarticle preprint 12pt",
          numerical_table_bodies_preserved=110,reviewed_current_output_hashes={r["document"]:r["sha256"] for r in q}))
    print("Visual review recorded: 120 pages.")

def release():
    delivery=read(OUT/"package_verification.json")
    assert delivery["status"]=="COMPLETE_CONCISE_AUTHOR_REVIEW_PACKAGE"
    assert read(OUT/"portable_verification.json")["status"]=="PASS"
    for q in delivery["pdfs"]:
        assert sha(ROOT/q["path"])==q["sha256"]
    for key in ["review","reproducibility"]:
        assert sha(ROOT/delivery[key]["path"])==delivery[key]["sha256"]
    hist=ROOT/"history/BEFORE_ARRAY_CONCISE_20261110"
    hist.mkdir(parents=True,exist_ok=True)
    for fn in ["CURRENT_STATE.json","00_BACA_PERTAMA_AKUN_BARU.md","MEMORI_PENELITIAN_AKUN_BARU.md"]:
        dst=hist/fn
        if not dst.exists():shutil.copy2(ROOT/fn,dst)
    state=read(ROOT/"CURRENT_STATE.json")
    state["status"]="ARRAY_JOURNAL_CONCISE_REVIEW_WITH_TRANSFER_LIMITATIONS"
    state["active_journal_manuscript"]=delivery["pdfs"][0]["path"]
    state["active_journal_supplement"]=delivery["pdfs"][1]["path"]
    state["active_journal_source"]="paper/array_20261110_concise/elsarticle"
    state["array_concise_revision_20261110"]=delivery
    write(ROOT/"CURRENT_STATE.json",state)
    note=("**Versi jurnal ringkas terbaru - 10 November 2026:** "
          "[main25 halaman](output/pdf/ARRAY_HAFID_CONCISE_REVIEW_20261110.pdf), "
          "[supplement93 halaman S1-S104](output/pdf/ARRAY_HAFID_SUPPLEMENT_S1_S104_CONCISE_20261110.pdf), "
          "[catatan pemadatan2 halaman](output/pdf/ARRAY_HAFID_CATATAN_PEMADATAN_20261110.pdf), "
          "[ZIP dospem](penyerahan/UNTUK_DOSPEM_ARRAY_RINGKAS_20261110.zip), "
          "[archive reproduksi](penyerahan/ARRAY_HAFID_REPRODUCIBILITY_CONCISE_20261110.zip), "
          "[rincian](docs/ARRAY_CONCISE_REVIEW_20261110.md). "
          "Main7tabel/5gambar/30referensi; supplement104tabel/9gambar. Format tetap preprint12pt satu kolom. "
          "Seluruh15+95 badan tabel sebelumnya tetap identik pada lokasi masing-masing;9tabel/3gambar dipindahkan. "
          "27.000 keputusan eksternal tersimpan diperiksa ulang;120halaman dirender dan ditinjau. "
          "Tidak ada fit/inference/tuning/profiling baru. Guard NormalV1,0104%; blok3 0,1435% within-trace; "
          "external31,90% CIC/18,85% TON, Ransomware1657/2000 menjadiNormal. "
          "Saving41,75% dan kontrak execution/XAI tetap milik predictor acuan f, bukan guard g. "
          "Verifikasi dari fresh extraction lulus tanpa model loading. "
          "Versi45/76halaman tetap arsip. Tidak ada Turnitin, submission atau approval penulis baru. "
          "Catatan berikut mempertahankan sejarah.\n\n")
    for fn in ["00_BACA_PERTAMA_AKUN_BARU.md","MEMORI_PENELITIAN_AKUN_BARU.md"]:
        p=ROOT/fn;s=p.read_text("utf8")
        if not s.startswith("**Versi jurnal ringkas terbaru"):
            p.write_text(note+s,encoding="utf8")
    write(OUT/"delivery_final.json",dict(status="COMPLETE",pdfs=delivery["pdfs"],
          review=delivery["review"],reproducibility=delivery["reproducibility"],
          previous_release_preserved=True,total_visual_pages=120,
          independent_archive_audit="PASS",new_fits=0,new_inference=0))
    print("Handoff updated; concise review delivery complete.")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["visual","release"],required=True)
    args=ap.parse_args()
    visual() if args.phase=="visual" else release()
