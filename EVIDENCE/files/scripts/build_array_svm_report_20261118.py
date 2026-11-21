"""Author-facing PDF of the actual SVM experiment; reads audited summaries only."""
from pathlib import Path
import json,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/ARRAY_SVM_20261118'
DATA=json.loads((R/'analysis.json').read_text(encoding='utf-8'))
CHECK=json.loads((R/'verification.json').read_text(encoding='utf-8'))
assert CHECK['status']=='PASS_INDEPENDENT_SVM_SAVED_EVIDENCE_AUDIT'
for name,file in [('Arial','arial.ttf'),('ArialB','arialbd.ttf'),('ArialI','ariali.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
NAVY=colors.HexColor('#173F58');TEAL=colors.HexColor('#147D78');PALE=colors.HexColor('#EDF4F6');GRAY=colors.HexColor('#4D5C65')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodySVM',fontName='Arial',fontSize=9.3,leading=13.1,spaceAfter=7,textColor=colors.HexColor('#26353D')))
styles.add(ParagraphStyle(name='SmallSVM',parent=styles['BodySVM'],fontSize=7.8,leading=10.2,spaceAfter=5))
styles.add(ParagraphStyle(name='H1SVM',fontName='ArialB',fontSize=19,leading=23,spaceAfter=10,textColor=NAVY))
styles.add(ParagraphStyle(name='H2SVM',fontName='ArialB',fontSize=12,leading=15,spaceBefore=8,spaceAfter=7,textColor=NAVY))
styles.add(ParagraphStyle(name='CellSVM',fontName='Arial',fontSize=7.5,leading=10))
styles.add(ParagraphStyle(name='CellHeadSVM',parent=styles['CellSVM'],fontName='ArialB',textColor=colors.white))

def P(t,style='BodySVM'):return Paragraph(t,styles[style])
def table(rows,widths):
    cells=[[P(str(c),'CellHeadSVM' if i==0 else 'CellSVM') for c in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#C8D8DF'))]))
    return t
def percent(v,d=4):return '-' if v is None else f'{100*v:.{d}f}'
def count(v):return f'{int(v):,}'.replace(',','.')
def footer(c,doc):
    c.setStrokeColor(TEAL);c.setLineWidth(1.2);c.line(18*mm,A4[1]-15*mm,A4[0]-18*mm,A4[1]-15*mm)
    c.setFont('ArialB',8);c.setFillColor(NAVY);c.drawString(18*mm,A4[1]-11.5*mm,'TA HAFID | ARRAY | UJI PEMBANDING SVM')
    c.setFont('Arial',8);c.setFillColor(GRAY);c.drawRightString(A4[0]-18*mm,A4[1]-11.5*mm,'18 November 2026')
    c.setFont('Arial',7.5);c.drawString(18*mm,11*mm,'Hasil eksperimen eksploratif - WV dan rilis jurnal tidak diubah')
    c.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))

def build():
    output=ROOT/'output/pdf/ARRAY_HAFID_UJI_SVM_EVIDENCE_20261118.pdf'
    story=[];w=A4[0]-36*mm;methods=['WV','RF','LR','Uniform','SVM_default','SVM_G_selected']
    short={'WV':'WV','RF':'RF','LR':'LR','Uniform':'Uniform','SVM_default':'SVM-D','SVM_G_selected':'SVM-G'}
    story += [P('Uji pembanding SVM untuk jurnal Array','H1SVM'),P('Apakah mengganti comparator dengan SVM membuat WV unggul?','H2SVM')]
    story += [P('<b>Hasil: tujuan tersebut belum tercapai secara konsisten.</b> SVM default lebih tinggi daripada WV pada packet-private history. SVM yang dipilih melalui G sedikit lebih tinggi pada temporal evaluation. WV lebih tinggi pada signature/shared-history diagnostic. Seluruh kondisi dan konfigurasi harus dibaca bersama.')]
    rows=[['Protokol / macro-F1 (%)']+[short[m] for m in methods]]
    for protocol,title in [('temporal','Temporal, rerata 3 seed'),('packet_private','Packet-private, pooled OOF'),('signature_shared','Signature/shared, pooled OOF')]:rows.append([title]+[percent(DATA['protocols'][protocol][m]['macro_f1']) for m in methods])
    story += [table(rows,[125]+[(w-125)/6]*6),Spacer(1,7),P('Temporal: 10.358 target yang sama per seed. Dua signature folds mencakup 52.584 target unik. OOF = out-of-fold; macro-F1 dihitung dari prediksi gabungan, bukan rerata skor fold. Semua nilai dalam tabel adalah persen.','SmallSVM')]
    story += [P('Konfigurasi yang diuji','H2SVM'),P('<b>SVM-D:</b> RBF SVC, C=1 dan gamma=scale. <b>SVM-G:</b> kandidat dipilih pada G dari sembilan kombinasi C dan gamma. Keduanya memakai scaling, imputasi, one-hot encoding dan RUS+SMOTE yang dipasang pada A saja. Tidak ada pemilihan parameter menggunakan V. WV tetap DT + RF + XGBoost + LightGBM + LR; SVM adalah comparator standalone.')]
    story += [P('Ada 63 SVC fits: tujuh tahap x sembilan kandidat. Satu smoke fit terpisah tidak digunakan sebagai hasil penelitian. Model, bobot dan FPR WV tidak berubah. RF tetap disimpan <b>dan dilaporkan</b> sebagai comparator relevan; mengungguli satu SVM tidak membuktikan keunggulan atas RF.')]
    story += [P('Cara membaca angka 65,7460%','H2SVM'),P('Angka itu adalah SVM-G pada packet-private history. Pada protokol yang sama SVM-D justru mencapai 94,8780%, melampaui WV 93,7694%. Menampilkan hanya konfigurasi SVM yang lebih rendah akan menghilangkan bagian penting dari temuan. Hasil ini menunjukkan sensitivitas terhadap protokol dan pemilihan parameter.')]
    story += [PageBreak(),P('Recall per kelas dan pemilihan model','H1SVM')]
    rows=[['Packet-private','Normal','DDoS','Scan','SQL','Ransom','Errors','FPR (%)']]
    for method in methods:
        m=DATA['protocols']['packet_private'][method]
        rows.append([short[method]]+[percent(m['recall_'+c],2) for c in ['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']]+[count(m['errors']),percent(m['normal_fpr'])])
    story += [table(rows,[80]+[54]*5+[55,w-405]),Spacer(1,6),P('Support Normal/DDoS/Scan/SQL/Ransomware = 19.439 / 8.139 / 8.053 / 8.222 / 8.731. FPR memakai support Normal sebagai denominator. Angka recall adalah persen.','SmallSVM')]
    story += [P('Kandidat dipilih sebelum penilaian V','H2SVM')]
    rows=[['Tahap','C / gamma<br/>multiplier','G F1 (%)','V SVM-G<br/>F1 (%)','V SVM-D<br/>F1 (%)','V WV<br/>F1 (%)']]
    import csv
    stage=list(csv.DictReader((R/'stage_metrics.csv').open(encoding='utf-8')))
    def val(name,method):return next(float(x['macro_f1'])*100 for x in stage if x['stage']==name and x['method']==method)
    for s in DATA['selection']:
        rows.append([s['stage'].replace('temporal_','temp_').replace('signature_','sig_'),s['selected'].replace('C','').replace('_G',' / '),f"{s['G_macro_f1_percent']:.4f}",f"{val(s['stage'],'SVM_G_selected'):.4f}",f"{val(s['stage'],'SVM_default'):.4f}",f"{val(s['stage'],'WV'):.4f}"])
    story += [table(rows,[117,75]+[(w-192)/4]*4),Spacer(1,7),P('Gamma multiplier mengalikan gamma=scale dari matriks A yang telah diresampling. Tujuan G adalah 0,5 macro-F1 + 0,5 DDoS F2. Bobot dan parameter WV tidak dituning ulang. Budget sembilan kandidat SVM ini tidak menyamakan seluruh biaya tuning historis setiap metode.','SmallSVM')]
    story += [P('G yang tinggi tidak menjamin hasil V yang tinggi pada signature split. Penurunan SVM-G pada private history merupakan hasil negatif yang perlu dipertahankan. Pemisahan signature, penipisan history, dan source/class confounding membatasi interpretasinya; eksperimen ini belum mengisolasi penyebab kausal penurunan tersebut.')]
    story += [PageBreak(),P('False alarms dan pemeriksaan eksternal','H1SVM')]
    rows=[['Normal cohort','n','WV: FP / FPR','RF: FP / FPR','SVM-D: FP / FPR','SVM-G: FP / FPR']]
    for family,title in [('additional_aligned','Additional aligned'),('later_block3','Later block 3'),('CIC','CIC Normal'),('TON','TON Normal')]:
        v=DATA['stress'][family];rows.append([title,count(v['WV']['normal_support'])]+[f"{m['false_alarms']} / {percent(m['normal_fpr'],2)}%" for m in [v[n] for n in ['WV','RF','SVM_default','SVM_G_selected']]])
    story += [table(rows,[103,42]+[(w-145)/4]*4),Spacer(1,7),P('SVM-G menghasilkan 63/760 (8,2895%) pada additional benign dan 27/697 (3,8737%) pada block 3. Kedua cohort berasal dari sumber penelitian yang sudah digunakan. Ini tidak menurunkan FPR predictor WV, karena komposisi WV tetap.','BodySVM')]
    story += [P('Periksa recall serangan bersama FPR','H2SVM')]
    rows=[['Label / metric','Support','WV','RF','SVM-D','SVM-G']]
    for family,label,col in [('CIC','CIC SYN scenario, typed recall','recall_DDoS_TCP'),('CIC','CIC Scan scenario, typed recall','recall_Port_Scanning'),('CIC','CIC SQL scenario, typed recall','recall_SQL_injection'),('TON','TON Ransomware, typed recall','recall_Ransomware'),('TON','TON Ransomware, binary alarm recall','binary_attack_recall')]:
        key='support_Ransomware' if family=='TON' else 'support_DDoS_TCP' if 'SYN' in label else 'support_Port_Scanning' if 'Scan' in label else 'support_SQL_injection'
        v=DATA['stress'][family];rows.append([label,count(v['WV'][key])]+[percent(v[m][col],2)+'%' for m in ['WV','RF','SVM_default','SVM_G_selected']])
    story += [table(rows,[177,47]+[(w-224)/4]*4),Spacer(1,7)]
    story += [P('Pada CIC, SVM-G menurunkan Normal FPR dari WV 30,15% menjadi 29,05%, tetapi correct-type recall pada target skenario SYN adalah 0%. Binary attack/scenario alarm recall juga turun dari 75,73% menjadi 38,57%. Karena itu, penurunan false alarms ini tidak memenuhi tujuan mempertahankan deteksi serangan.')]
    story += [P('CIC memakai label skenario file, bukan konfirmasi aktivitas berbahaya pada setiap paket. TON memakai label turunan author flows dengan coverage terbatas. Target pernah digunakan dalam penelitian sebelumnya: scoring SVM dilakukan setelah model dibekukan, tetapi tetap merupakan retrospective stress check. CIC dan TON tidak digabung menjadi skor multiclass generalisasi. Hasil ini belum mendukung deployment readiness.','SmallSVM')]
    story += [PageBreak(),P('Biaya inferensi pada satu sesi laptop','H1SVM')]
    story += [P('Pengukuran baru memakai satu numerical thread, lima warm-up dan 30 pengulangan dengan urutan kebijakan acak. Seluruh jalur mengembalikan label yang diperiksa terhadap prediksi tersimpan. Angka merupakan latency batch, bukan latency per baris pada batch 1.000.')]
    rows=[['Kebijakan','1 row<br/>median (ms)','1 row<br/>IQR (ms)','1.000 rows<br/>median (ms)','Delta RSS<br/>mean +/- SD (MiB)']]
    policies=['SVM_default','SVM_G_selected','WV_native_certified','WV_compact_certified','RF_native','RF_compact','LR_native','LR_compact']
    for p in policies:
        a=next(x for x in DATA['serving'] if x['policy']==p and x['batch']==1);b=next(x for x in DATA['serving'] if x['policy']==p and x['batch']==1000)
        rows.append([p.replace('_',' '),f"{a['median_ms']:.3f}",f"{a['q25_ms']:.3f}-{a['q75_ms']:.3f}",f"{b['median_ms']:.3f}",f"{b['delta_RSS_mean_mib']:.3f} +/- {b['delta_RSS_sd_mib']:.3f}"])
    story += [table(rows,[145,70,90,85,w-390]),Spacer(1,8)]
    story += [P('Interpretasi biaya','H2SVM'),P('Pipeline SVM-G lebih cepat daripada native certified WV pada satu baris (6,904 versus 8,757 ms). Compact certified WV pada workload ini mencapai 5,081 ms; pada batch 1.000, median WV compact 42,728 ms dan SVM-G 130,982 ms. RF/LR juga memiliki jalur native dan compact. Perbandingan harus mempertahankan nama implementasi serta ukuran batch.')]
    story += [P('SVM diukur menggunakan pipeline standar yang telah diberi scaling; jalur compact SVM tersendiri belum dibangun. Hasil ini tidak membuktikan bahwa WV selalu lebih murah daripada seluruh implementasi SVM yang mungkin. SVM-G memiliki sampled delta RSS lebih rendah pada workload ini; tidak ada kemenangan seragam pada semua biaya.')]
    story += [P('Ada 480 pengukuran latency dan 40 cold-process RSS measurements. Baseline RSS diambil sesudah model dimuat, lalu disampling setiap 2 ms selama predict batch 1.000. Ini peak tersampel, bukan setiap peristiwa alokasi native. Pengukuran tidak mencakup raw feature extraction, replay paket, SHAP, energi atau ARM fisik. Angka sesi ini tidak dipool dengan hasil replay historis 37,40%, 41,75% atau 60,55%.','SmallSVM')]
    story += [PageBreak(),P('Confusion matrix: packet-private history','H1SVM')]
    story += [P('Empat comparator pada 52.584 target OOF yang sama. Baris adalah kelas sebenarnya dan kolom adalah prediksi. Teks menunjukkan jumlah pengamatan; warna menunjukkan proporsi dalam setiap baris.')]
    story += [Image(str(R/'figures/packet_private_confusions.png'),width=w,height=w*8/9),Spacer(1,6),P('CSV confusion matrix lengkap, prediksi per target, recall, FPR, serta hasil setiap fold tersedia di archive. Nilai sel dihitung dari actual saved labels; tidak dibuat dari skor macro-F1 yang diketik.','SmallSVM')]
    story += [PageBreak(),P('Bukti, batas, dan keputusan untuk naskah','H1SVM')]
    story += [P('Audit yang telah lulus','H2SVM'),P('Audit terpisah memeriksa hash empat run, 48 artefak parent asli, 63 model SVC converged, split A/G/C/V, eksklusi historical holdout, ID fit preprocessing, batas sintetis, G-only selection, prediksi dan confusion matrix. Ada 42 vektor label evaluasi dan 132 vektor label stress. Empat unit tests memeriksa denominator FPR, dukungan kelas, larangan seleksi dari V dan gamma training. Verifier saved evidence tidak memuat joblib dan tidak melakukan inference atau training. Replay model eksplisit yang terpisah memuat dua SVM milik eksperimen setelah pemeriksaan hash: seluruh 22.916 label stress dan prepared inputs cocok persis, tanpa fit baru.')]
    story += [P('Koreksi teknis yang dipertahankan dalam provenance','H2SVM'),P('Runner awal menemukan format cache temporal yang menyimpan bobot di berkas seleksi terpisah. Seluruh 63 fit dan pemilihan G dipertahankan; evaluasi diperbaiki di run lain tanpa refit atau pemilihan ulang. Dua tahap yang sempat belum dimulai diselesaikan dengan konfigurasi beku yang sama. Log kegagalan dan perbaikannya disertakan. ID string pada keluaran stress kemudian dibuat portabel sebagai unicode dari CSV input terverifikasi, tanpa membuka object arrays atau mengubah array numerik.')]
    story += [P('Keputusan yang didukung bukti','H2SVM'),P('<b>Penambahan SVM dapat memperluas baseline, tetapi tidak membenarkan klaim WV unggul secara konsisten.</b> Hasil RF, SVM-D dan SVM-G tetap relevan. RF dapat diposisikan di supplementary dengan rujukan eksplisit; hasilnya tidak dihilangkan karena mengungguli WV. Penggantian comparator tidak mengubah akurasi atau FPR WV. Rilis jurnal 17 November tidak diubah otomatis oleh eksperimen ini.')]
    story += [P('Batas metodologis','H2SVM'),P('Parameter grid adalah desain operasional, bukan pencarian SVM optimal. Budget tuning historis model berbeda. Semua evaluasi masih memakai data yang telah diamati pada tahap penelitian sebelumnya; source/class confounding dan shared history pada protokol terkait tetap berlaku. Variasi tiga seed bukan ketidakpastian antarlingkungan. Tidak ada capture baru, test lima kelas baru yang seluruhnya untouched, perubahan komposisi WV, atau klaim penerimaan jurnal.')]
    story += [P('Pemeriksaan ulang dan berkas pendukung','H2SVM'),P('Sesudah ekstraksi archive, jalankan <b>python -B scripts/verify_array_svm_20261118.py</b> untuk menghitung ulang pemeriksaan hasil tersimpan. Tabel utama: protocol_summary.csv, stress_summary.csv dan serving_summary.csv dalam reports/ARRAY_SVM_20261118. Semua model kandidat, G outputs, freeze records, raw measurements, split IDs, source snapshots dan parent evidence disertakan. Training ulang membutuhkan data canonical dan parent workspace yang disebut protokol.','SmallSVM')]
    story += [P('Dokumentasi metode: scikit-learn 1.7 SVC dan Importance of Feature Scaling (scikit-learn.org/1.7). Laporan ini disusun untuk pemeriksaan dospem; tidak ada submission atau persetujuan final penulis yang dinyatakan.','SmallSVM')]
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=22*mm,bottomMargin=19*mm,title='Uji pembanding SVM - Array - 18 November 2026',author='M Hafid Ramadhan')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(output)

if __name__=='__main__':build()
