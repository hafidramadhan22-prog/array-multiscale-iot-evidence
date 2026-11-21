"""Five-page Indonesian change report comparing the actual previous/new PDFs."""
from pathlib import Path
import csv,hashlib,json,shutil,html
import pymupdf as fitz
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'paper/array_20261109_payload_audit/elsarticle'
OUT=ROOT/'output/pdf';REP=ROOT/'reports/ARRAY_PAYLOAD_AUDIT_REVISION_20261109'
FILES=[('main.pdf','ARRAY_HAFID_REVISED_PAYLOAD_AUDIT_20261109.pdf'),('supplement.pdf','ARRAY_HAFID_SUPPLEMENT_S1_S73_20261109.pdf')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf8'))

def main():
    assert read(REP/'verification.json')['status']=='PASS_SOURCES_AND_RECORDED_EVIDENCE'
    OUT.mkdir(exist_ok=True)
    for src,dest in FILES:shutil.copy2(B/src,OUT/dest)
    fonts=Path('C:/Windows/Fonts')
    for n,f in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Consolas','consola.ttf')]:pdfmetrics.registerFont(TTFont(n,str(fonts/f)))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle',fontName='Arial-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#163d54'),spaceAfter=13))
    styles.add(ParagraphStyle(name='ReportSub',fontName='Arial-Bold',fontSize=14,leading=18,textColor=colors.HexColor('#163d54'),spaceAfter=9,spaceBefore=8))
    styles.add(ParagraphStyle(name='ReportBody',fontName='Arial',fontSize=10,leading=14.1,spaceAfter=8))
    styles.add(ParagraphStyle(name='Cell',fontName='Arial',fontSize=9,leading=12,spaceAfter=0))
    styles.add(ParagraphStyle(name='SmallText',fontName='Arial',fontSize=8.1,leading=11,spaceAfter=6,textColor=colors.HexColor('#4b6272')))
    styles.add(ParagraphStyle(name='CodeText',fontName='Consolas',fontSize=9.5,leading=13,spaceAfter=9,backColor=colors.HexColor('#edf3f6'),borderPadding=7))
    def P(t,sty='ReportBody'):return Paragraph(t,styles[sty])
    story=[]
    def h(t):story.append(P(t,'ReportSub'))
    def p(t):story.append(P(t))
    def table(head,rows,widths):
        data=[[P('<b>'+html.escape(str(v))+'</b>','Cell') for v in head]]+[[P(html.escape(str(v)),'Cell') for v in r] for r in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8aa5b7')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d7e2e9'))]))
        story.extend([t,Spacer(1,10)])
    mainpages=len(fitz.open(B/'main.pdf'));suppages=len(fitz.open(B/'supplement.pdf'))
    story.append(P('Apa yang berubah<br/>pada revisi jurnal Array?','ReportTitle'))
    p('<b>M Hafid Ramadhan - Satria Mandala</b><br/>Laporan perbandingan, 9 November 2026')
    p('Pembandingnya adalah naskah jurnal <b>5 November 2026</b> (35 halaman) dan supplementary S1-S65 (51 halaman). Versi baru memasukkan audit payload yang selesai pada 6 November dan bukti yang ditampilkan dalam notebook hasil 9 November. Dokumen lama dipertahankan sebagai arsip.')
    h('Perubahan paling penting')
    p('<b>Revisi ini tidak melaporkan penurunan FPR baru.</b> Angka 0/760 sudah ada pada versi sebelumnya. Tambahan utama adalah bukti bahwa angka tersebut dapat direproduksi, sekaligus kegagalan model TF-IDF/hybrid pada ilustrasi SQL. Karena itu, nol alarm tidak dipakai sebagai bukti bahwa kemampuan deteksi SQL telah aman.')
    table(['Dokumen','Sebelumnya','Revisi ini'],[
        ['Main manuscript','35 halaman; 11 tabel; 6 gambar',f'{mainpages} halaman; 12 tabel; 7 gambar'],
        ['Supplementary','51 halaman; S1-S65; 6 gambar',f'{suppages} halaman; S1-S73; 6 gambar'],
        ['Referensi','26 entri','26 entri yang sama; semua sitasi terpakai dan terhubung'],
        ['Model dan eksperimen','Model/fitting terdokumentasi pada run sebelumnya','Tidak ada fit, perubahan bobot atau inference baru untuk integrasi dokumen ini']
    ],[113,196,196])
    p('Bahasa pada Introduction, bagian audit, Discussion dan Conclusion disusun ulang agar hubungan antara hasil, alasan dan batas klaim lebih jelas. Perbaikan bahasa tidak mengubah label, probabilitas, sampel atau hasil eksperimen.')
    story.append(PageBreak())
    story.append(P('1. Bukti isi payload asli','ReportTitle'))
    p('Versi sebelumnya sudah membandingkan syntax, TF-IDF dan gabungan, tetapi belum menyajikan audit lengkap yang menelusuri record asli sampai ke fitur. Revisi ini menambahkan Figure 7 dan Tables S66-S68 agar pembaca dapat memeriksa apa yang sebenarnya masuk ke classifier.')
    table(['Record aktual','Isi yang diekstrak','Maknanya'],[
        ['Normal 105044','24.68 76.42','Isi pesan MQTT Temperature/Humidity; bukan request HTTP Normal.'],
        ['SQL 23740','2 Submitid=53 ORDER BY 1-- MEbL','Nilai query yang dipulihkan dari request GET DVWA. Header seperti User-Agent tidak masuk ke representasi isi.']
    ],[105,193,207])
    table(['Kelas canonical','Jumlah baris','TCP hex valid','HTTP request / GET'],[
        ['Normal','24.301','7.975','0 / 0'],['SQL injection','10.311','1.707','822 / 822']
    ],[145,103,115,142])
    p('Pencarian dibatasi pada awal current payload, tanpa reassembly atau dekripsi. Karena itu, nol request HTTP Normal yang dikenali tidak berarti tidak pernah ada HTTP pada seluruh sesi/capture. Scan seluruh canonical hanya menjelaskan ketersediaan isi; baris tersebut tidak ditambahkan ke fitting atau dijadikan final test baru.')
    h('Mengapa skor binary yang tinggi perlu dibatasi?')
    p('Seratus content Normal pada training terdiri dari 50 isi MQTT dan 50 TCP lain. Seratus content SQL semuanya berupa nilai query HTTP. Protokol/jenis isi masih berimpit dengan label. Empat indikator - OR, numeric tautology, UNION SELECT dan isdecimal - tidak pernah bernilai positif pada 200 contoh training. Implementasi extractor yang benar belum membuktikan model telah mempelajari pola positif tersebut.')
    p('Delapan CSV sumber tambahan diperiksa ukuran/hash-nya. Nilai tcp.payload dan mqtt.msg pada 760 target cocok dengan input beku. Sebanyak 1.972 trace aktual tersedia sebagai ekspor record, hex, decode, fitur dan provenance; data tersebut tidak diganti dengan contoh buatan.')
    story.append(PageBreak())
    story.append(P('2. Hasil negatif yang kini masuk paper','ReportTitle'))
    p('Dua isi berikut berasal dari ilustrasi dospem, bukan record dataset. Keduanya dilewatkan ke parser yang sama sebagai query GET dengan URL encoding. Tidak ada request jaringan atau SQL yang dieksekusi.')
    story.append(P('Normal ilustratif: 10<br/>Pola SQL ilustratif: 10\' OR \'1\'=\'1','CodeText'))
    table(['Representasi / WV','p(SQL) pada pola SQL','Prediksi'],[
        ['Syntax-only','0,935234','SQL injection'],['Character TF-IDF','0,111796','Normal'],['Syntax + TF-IDF','0,157671','Normal']
    ],[194,160,151])
    p('Enam nilai fitur pertama memang sesuai contoh: <b>[0,0,0,0,0,1]</b> untuk 10 dan <b>[4,1,1,1,0,0]</b> untuk pola SQL. Kegagalan TF-IDF/hybrid terjadi sesudah ekstraksi, pada keputusan model biner. Table 12 merangkum weighted voting; Tables S69-S71 menampilkan seluruh 48 keluaran, yaitu dua ilustrasi x tiga representasi x delapan predictor.')
    h('Batas kesimpulan yang diperbaiki')
    p('Main predictor 85 fitur/history tidak diuji menggunakan konteks buatan untuk dua string ini. Karena itu, hasilnya tidak dihitung sebagai FNR sistem utama. Namun kegagalan modul content cukup untuk menolak penafsiran bahwa hybrid telah terbukti menjaga sensitivitas SQL. Jika paket nyata masuk wilayah konflik yang eligible dan content check memberi keputusan Normal seperti probe ini, modul tersebut tidak mempertahankan alarm SQL awal.')
    p('Keberhasilan syntax-only pada satu ilustrasi juga tidak dipakai untuk memilih pemenang baru. Tidak ada refit, perubahan threshold atau penambahan data setelah probe. Semua hasil negatif tetap disertakan.')
    h('Reproduksi berbeda dari generalisasi')
    p('Replay model beku memberikan selisih probabilitas maksimum <b>0</b> terhadap cache. Angka 321/760 menjadi 0/760 dan label V yang tetap sama dapat direproduksi. Itu memverifikasi perhitungannya, bukan membuktikan bahwa model aman terhadap serangan atau capture independen.')
    story.append(PageBreak())
    story.append(P('3. Peta perubahan naskah','ReportTitle'))
    table(['Bagian','Revisi konkret'],[
        ['Abstract dan Introduction','Menempatkan tiga kontribusi pada bukti yang sesuai; memasukkan audit dan kegagalan probe tanpa mengklaim metode deteksi baru yang unggul.'],
        ['Methods, bagian 6.5','Menjelaskan sumber raw payload, identitas record, checksum, scope replay dan pemisahan ilustrasi dari data penelitian.'],
        ['Results, bagian 7.9','Menghubungkan isi aktual, coverage fitur training, prediksi probe dan batas keselamatan rule.'],
        ['Table 12 / Figure 7','Menambah prediksi weighted pada dua ilustrasi serta diagram dua trace aktual dari raw payload ke content.'],
        ['Discussion','Menjelaskan mengapa source dependence, RF-only ablation dan kegagalan hybrid membatasi klaim repair.'],
        ['Conclusion','Menjawab RQ1-RQ3 dengan angka utama; menyatakan nol alarm belum membuktikan SQL safety.'],
        ['Availability / AI declaration','Menjelaskan cuplikan 1.972 record dan notebook hasil, mempertahankan pernyataan pendanaan/konflik/CRediT serta penggunaan bantuan AI secara transparan.'],
        ['Supplement S66-S73','Menambah availability, trace asli, 12-feature coverage, semua 48 probe, raw/replay checks dan decision ledger.']
    ],[130,375])
    h('Angka lama yang tetap dipertahankan')
    p('Gain temporal +1,1425 pp pada ordering 1 detik; 97,8954% rerata tiga seed versus 97,8800% seed-42 yang berbeda ringkasannya; FPR historical 42,63% dan aligned 42,24%; rule 0/760; V masih mempunyai 36 Normal false alarms serta dua attack-to-Normal errors. Mean compact replay tetap 1,5791 menjadi 0,9198 s (41,75%); latency native hybrid pada additional 760 tetap lebih besar, 61,109 versus 41,675 ms.')
    p('Semua isi numerik tabel utama 1-11 dan supplementary S1-S65 diperiksa tetap identik. Perubahan interpretasi tidak ditampilkan sebagai peningkatan skor baru.')
    story.append(PageBreak())
    story.append(P('4. Berkas dan pemeriksaan akhir','ReportTitle'))
    table(['Berkas baru','Fungsi'],[
        [FILES[0][1],f'Naskah utama {mainpages} halaman untuk pemeriksaan dospem.'],
        [FILES[1][1],f'Supplementary {suppages} halaman, Table S1-S73 lengkap.'],
        ['PDF perbandingan ini','Menjelaskan perubahan terhadap versi 5 November, bukan laporan training baru.'],
        ['Companion reproducibility archive','Source LaTeX, konfigurasi, data turunan, model/manifest lama, trace/prediksi/probe dan hasil verifikasi. Data mentah lengkap pihak ketiga tetap mengikuti sumber dataset.']
    ],[275,230])
    p('Pemeriksaan angka menghitung ulang 11.118 keputusan reference/guard, mencocokkan sembilan tabel baru dengan bukti, memeriksa seluruh label/cross-reference dan 26 entri bibliografi. Hasil training lama tidak ditimpa. Ketiga PDF dirender untuk memeriksa tabel, gambar, pemenggalan teks dan keterbacaan; hasil pemeriksaan visual dicatat bersama arsip.')
    h('Penulisan dan Turnitin')
    p('Tulisan diperbaiki untuk ketepatan, alur argumen dan keterbacaan. Tidak ada pemeriksaan Turnitin baru atau angka similarity/AI yang dihasilkan pada tahap ini, sehingga tidak ada jaminan skor tertentu. Deklarasi bantuan AI dipertahankan sesuai penggunaan yang sebenarnya; kebijakan Elsevier mewajibkan keterbukaan atas bantuan substantif dan tanggung jawab penulis atas isi.')
    story.append(P('<link href="https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals" color="#17617e">Sumber: Elsevier, Generative AI policies for journals</link> (diperiksa 9 November 2026).','SmallText'))
    h('Identitas dokumen untuk pengecekan')
    for src,name in FILES:
        story.append(P(html.escape(name)+'<br/>SHA256: '+sha(OUT/name),'SmallText'))
    p('Versi ini adalah hasil revisi untuk ditinjau. Paket tidak menyatakan penerimaan jurnal, generalisasi independen atau persetujuan penulis yang belum dikonfirmasi.')
    def footer(c,d):
        c.saveState();w,h=A4;c.setStrokeColor(colors.HexColor('#d7e3ea'));c.line(45,42,w-45,42)
        c.setFont('Arial',8);c.setFillColor(colors.HexColor('#537083'));c.drawString(45,29,'ARRAY | Perbandingan revisi 5 - 9 November 2026');c.drawRightString(w-45,29,str(d.page));c.restoreState()
    path=OUT/'PERBANDINGAN_REVISI_JURNAL_ARRAY_20261109.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=44,bottomMargin=56,title='Perbandingan revisi jurnal Array - Hafid',author='M Hafid Ramadhan; Satria Mandala')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    d=fitz.open(path);assert len(d)==5,('comparison page overflow',len(d))
    info=dict(status='THREE_PDFS_WRITTEN',pdfs=[dict(path=(OUT/n).relative_to(ROOT).as_posix(),pages=len(fitz.open(OUT/n)),sha256=sha(OUT/n)) for n in [FILES[0][1],FILES[1][1],path.name]])
    (REP/'pdf_outputs.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
