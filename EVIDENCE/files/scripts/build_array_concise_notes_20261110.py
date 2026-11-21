"""Build a short Indonesian account of the editorial condensation."""
from pathlib import Path
import json,html
import pymupdf as fitz
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/ARRAY_CONCISE_20261110"
FONT=Path("C:/Windows/Fonts")
for name,fn in [("ArialC","arial.ttf"),("ArialCB","arialbd.ttf")]:
    pdfmetrics.registerFont(TTFont(name,str(FONT/fn)))
pdfmetrics.registerFontFamily("ArialC",normal="ArialC",bold="ArialCB",italic="ArialC",boldItalic="ArialCB")
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyC",fontName="ArialC",fontSize=9.4,leading=13.1,spaceAfter=7,textColor=colors.HexColor("#243a4b")))
styles.add(ParagraphStyle(name="SmallC",parent=styles["BodyC"],fontSize=8.2,leading=11.2,spaceAfter=4))
styles.add(ParagraphStyle(name="TitleC",parent=styles["BodyC"],fontName="ArialCB",fontSize=20,leading=25,spaceAfter=12))
styles.add(ParagraphStyle(name="HeadC",parent=styles["BodyC"],fontName="ArialCB",fontSize=11,leading=15,spaceBefore=7))

def main():
    v=json.loads((OUT/"verification.json").read_text("utf8"))
    assert v["status"]=="PASS_CONDENSED_MANUSCRIPT_AND_SAVED_EVIDENCE"
    m=len(fitz.open(ROOT/"output/pdf/ARRAY_HAFID_CONCISE_REVIEW_20261110.pdf"))
    s=len(fitz.open(ROOT/"output/pdf/ARRAY_HAFID_SUPPLEMENT_S1_S104_CONCISE_20261110.pdf"))
    story=[]
    def p(t,style="BodyC"):story.append(Paragraph(t,styles[style]))
    def table(head,rows,widths):
        cells=[[Paragraph(html.escape(str(x)),styles["SmallC"]) for x in head]]
        cells += [[Paragraph(html.escape(str(x)),styles["SmallC"]) for x in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7eef3")),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#758d9f")),
                               ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#d4dfe7")),
                               ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
                               ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,9))
    p("REVISI EDITORIAL | ARRAY | 10 NOVEMBER 2026","SmallC")
    p("Naskah lebih ringkas,<br/>bukti tetap lengkap","TitleC")
    p(f"Artikel utama dipadatkan dari <b>45 menjadi {m} halaman</b>. Format tetap preprint satu kolom, 12 pt. "
      f"Artikel memiliki 7 tabel, 5 gambar, dan 30 referensi. Supplementary menjadi {s} halaman dengan "
      "Tables S1-S104 dan Figures S1-S9.")
    p("Perubahan pada susunan","HeadC")
    table(["Bagian","Perubahan"],[
        ["Alur kontribusi","RQ1: konteks; RQ2: preservation dan biaya eksekusi; RQ3: consistency dan sensitivity penjelasan."],
        ["Methods","Inti metode tetap di main. Rincian aturan lama, payload, ekstraksi eksternal dan prototipe tersedia di supplementary."],
        ["Results","Empat pembahasan payload/perbaikan alarm dirangkum menjadi satu bagian terintegrasi."],
        ["Tabel/gambar","Enam badan tabel main lama tetap; sembilan pindah. Satu tabel eksternal ringkas baru dihitung dari hasil tersimpan."],
        ["Discussion/Conclusion","Jawaban RQ dipisahkan dari diagnosis dan batas transfer. Tidak mengulang seluruh riwayat eksperimen."]],[115,384])
    p("Hasil penting yang tetap terlihat","HeadC")
    p("Gain konteks <b>1,1425-1,1475 poin persentase</b> pada ordering tetap; replay acuan "
      "<b>1,5791 menjadi 0,9198 detik (41,75%)</b>; audit penjelasan mempertahankan full-probability consistency.")
    p("FPR guard pada Normal V <b>39/3.860 = 1,0104%</b>, pada blok ketiga <b>1/697 = 0,1435%</b>. "
      "External guard FPR <b>31,90% CIC</b> dan <b>18,85% TON</b>; <b>1.657/2.000</b> target Ransomware terpetakan menjadi Normal. "
      "Tidak ada klaim seluruh masalah FPR atau generalisasi telah selesai.")
    p("<b>Tidak ada training, inference, tuning atau profiling baru.</b> Perubahan ini menyunting penyajian eksperimen yang sudah selesai. "
      "Versi 45 halaman dan seluruh arsip sebelumnya tetap disimpan.","SmallC")
    story.append(PageBreak())
    p("PETA PERUBAHAN DAN VERIFIKASI","SmallC")
    p("Lokasi bukti setelah pemadatan","TitleC")
    table(["Main lama","Lokasi sekarang"],[
        ["Table 1: related work","Supplementary Table S96"],
        ["Tables 2/3/4: konteks/history/model","Main Tables 1/2/3"],
        ["Table 5: initial benign; Table 6: buffer","Supplementary Tables S97/S98"],
        ["Tables 7/8: runtime/faithfulness","Main Tables 4/5"],
        ["Tables 9/10/11/12: payload/refit/rule/cost/probe","Supplementary Tables S99/S100/S101/S102"],
        ["Table 13: final guard FPR","Main Table 6"],
        ["Tables 14/15: external Normal/attacks","Supplementary Tables S103/S104; ringkasan Main Table 7"],
        ["Figures 1/2/4/5/8","Main Figures 1/2/3/4/5"],
        ["Figures 3/6/7","Supplementary Figures S7/S8/S9"]],[244,255])
    p("Pemeriksaan yang dilakukan","HeadC")
    p("Seluruh <b>15 badan tabel main lama</b> masih tersedia identik, dan seluruh <b>95 badan tabel supplementary lama</b> "
      "dipertahankan. Rujukan main dalam catatan runtime S45 diperbarui ke Table 4/Figure 3. "
      "Tabel eksternal baru diperiksa terhadap <b>27.000 keputusan tersimpan</b>, tanpa memuat model.")
    p("Label dan rujukan internal serta 30 entri bibliography diperiksa. Semua grafik menggunakan byte sumber yang sama "
      "dengan versi terverifikasi. Inventaris PDF, posisi konten dan tampilan halaman diperiksa sebelum paket dikirim.")
    p("Batas ilmiah dan status review","HeadC")
    p("Hasil context/history tetap within-trace dan exploratory. CIC memakai label skenario file; TON memakai label flow turunan "
      "dengan coverage terbatas. Predictor guard berbeda dari predictor acuan, sehingga tidak mewarisi kontrak eksekusi/XAI "
      "atau penghematan replay acuan.")
    p("Funding, konflik kepentingan dan CRediT mengikuti jawaban penulis yang terkonfirmasi. Persetujuan final seluruh penulis "
      "belum dikonfirmasi; tidak dibuat klaim approval baru. Companion archive tersedia, tanpa URL repository/DOI yang dikarang. "
      "Tidak ada submission atau pemeriksaan Turnitin. Bantuan AI tetap diungkap secara faktual.")
    p(f"<b>Kirim kepada dospem:</b> main {m} halaman, supplementary {s} halaman, dan catatan ini. "
      "Paket ZIP review memuat source editable. Reproducibility archive terpisah memuat bukti lengkap.","SmallC")
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor("#d4dfe7"));c.line(48,39,547,39)
        c.setFont("ArialC",8);c.setFillColor(colors.HexColor("#556b7e"))
        c.drawString(48,26,"TA Hafid | Pemadatan naskah Array | 10 November 2026")
        c.drawRightString(547,26,str(doc.page))
    path=ROOT/"output/pdf/ARRAY_HAFID_CATATAN_PEMADATAN_20261110.pdf"
    SimpleDocTemplate(str(path),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=52,
                      title="Catatan pemadatan naskah Array",author="M Hafid Ramadhan").build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)

if __name__=="__main__":
    main()
