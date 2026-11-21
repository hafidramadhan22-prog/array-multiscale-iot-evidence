"""Six-page supervisor report from verified, sealed FPR experiments."""
from pathlib import Path
import csv,json,re
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'runs/PAYLOAD_TARGET_REPAIR_20261105_01';G=ROOT/'runs/SQL_DISAGREEMENT_REPAIR_20261105_01';R=ROOT/'reports/SQL_FPR_REPAIR_20261105'
PDF=ROOT/'output/pdf/HASIL_PERBAIKAN_FPR_NORMAL_SQL_20261105.pdf'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
M=rows(P/'metrics.csv');GM=rows(G/'metrics.csv');EFF=rows(G/'efficiency_summary.csv');V=json.loads((R/'verification.json').read_text(encoding='utf8'));assert V['status']=='PASS'
def m(key,cohort,method='weighted_voting'):return next(r for r in M if r['key']==key and r['cohort']==cohort and r['method']==method)
def gm(key,cohort):return next(r for r in GM if r['key']==key and r['cohort']==cohort)
def f(v,d=2):return f'{float(v):.{d}f}'.replace('.',',')
def p(v,d=2):return f(v,d)+'%'
def n(v):return f'{int(float(v)):,}'.replace(',','.')
for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
ss=getSampleStyleSheet()
ss.add(ParagraphStyle(name='BodyP',fontName='Arial',fontSize=9.6,leading=13.3,spaceAfter=7,textColor=colors.HexColor('#25394c')))
ss.add(ParagraphStyle(name='TitleP',parent=ss['BodyP'],fontName='Arial-Bold',fontSize=19,leading=23,spaceAfter=10,textColor=colors.HexColor('#143f60')))
ss.add(ParagraphStyle(name='HeadP',parent=ss['BodyP'],fontName='Arial-Bold',fontSize=11.5,leading=15,spaceBefore=7,spaceAfter=5,textColor=colors.HexColor('#143f60')))
ss.add(ParagraphStyle(name='CellP',parent=ss['BodyP'],fontSize=8.4,leading=10.6,spaceAfter=0))
ss.add(ParagraphStyle(name='SmallP',parent=ss['BodyP'],fontSize=8.3,leading=11.2,spaceAfter=6))
story=[];W=178*mm
def para(s,kind='BodyP'):
    # Keep filenames executable while spacing prose and numeric quantities.
    s=s.replace('warm-up3,10','warm-up 3, 10').replace('Normal,64','Normal, 64')
    pieces=re.split(r'(<[^>]+>|[^\s<>]+\.(?:csv|py|json))',s)
    for i,part in enumerate(pieces):
        if part.startswith('<') or re.search(r'\.(csv|py|json)$',part):continue
        part=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',part)
        part=re.sub(r'(?<=[a-z])(?=(?:SQL|Normal|WV|RF|FPR)\b)',' ',part)
        pieces[i]=part.replace('1 e-12','1e-12').replace('L 2','L2').replace('F 1','F1').replace('normalisasiL2','normalisasi L2').replace('C 60/O 10','C60/O10')
    return Paragraph(''.join(pieces),ss[kind])
def add(s,kind='BodyP'):story.append(para(s,kind))
def table(data,widths=None,pad=4):
    t=Table([[para(str(c),'CellP') for c in row] for row in data],colWidths=widths or [W/len(data[0])]*len(data[0]),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#a1bacd')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e4ed'))]));story.extend([t,Spacer(1,5)])
def page():story.append(PageBreak())

add('Mengapa false alarms tinggi,<br/>dan apa yang berhasil diuji?','TitleP')
add('M. Hafid Ramadhan | 5 November 2026 | laporan untuk dosen pembimbing','SmallP')
add('<b>Hasil pada data yang tersedia:</b> prototipe pemeriksaan konflik menurunkan false alarms dari <b>321/760 (42,24%) menjadi 0/760</b>. Seluruh label pada V 10.358 sampel tetap identik. Ini perbaikan eksploratif pada data yang sudah diamati; keselamatan pada konflik serangan baru belum terbukti.')
add('1. Sumber angka 42%','HeadP')
add('Angka historis 324/760 (42,63%) mencakup tiga kesalahan format flag hex/desimal. Setelah format diselaraskan, masih ada 321 error: <b>seluruhnya Normal diprediksi SQL injection</b>. Model asli dan susunan input direproduksi; posterior V dan additional cocok dengan cache acuan sampai toleransi absolut 1e-12. Tidak ditemukan salah alignment yang menjelaskan sisa error.')
add('2. Konflik probabilitas antar-model','HeadP')
data=[['Expert','FP / 760','Rerata kontribusi berbobot<br/>p(SQL) - p(Normal), pada321 error']]
for r in rows(R/'expert_margin_diagnosis.csv'):data.append([{'decision_tree':'Decision Tree','random_forest':'Random Forest','xgboost':'XGBoost','lightgbm':'LightGBM','logistic_regression':'Logistic Regression'}[r['model']],r['additional_false_alarms'],f(r['weighted_SQL_minus_Normal_on_321'],6)])
table(data,[53*mm,30*mm,95*mm])
add('DT dan LR memberi dukungan SQL yang lebih besar daripada dukungan RF dan XGB untuk Normal; jumlah rerata margin berbobot +0,099759. RF sendiri tidak membuat false alarm pada760 target tersebut. Ini menjelaskan keputusan voting pada kasus teramati, tanpa membuktikan penyebab tunggal pergeseran domain.','SmallP')
add('3. Payload dan context membatasi solusi','HeadP')
add('Hanya <b>64 dari321 error</b> memiliki current content yang dapat diperiksa. Sisa257 merupakan paket tcp.len=0: ACK140, RST61, SYN-ACK24, SYN16, FIN-ACK16. Fitur isi tidak dapat membaca payload yang tidak tersedia. Distribusi multiscale context juga bergeser dari Normal training; perturbasi context merusak deteksi sehingga tidak digunakan sebagai solusi.')
add('FPR = Normal yang dianggap serangan / seluruh Normal. Serangan yang salah diklasifikasikan menjadi jenis serangan lain dihitung sebagai error kelas, bukan serangan yang lolos. Additional760 hanya berisi Normal; attack recall tidak tersedia di sana.','SmallP')

page();add('Training payload langsung:<br/>penurunannya belum memadai','TitleP')
add('Model acuan yang diuji adalah model utama besar yang menghasilkan321 alarm, bukan pilot500 baris. Refit menggunakan <b>A31.676 baris</b>, lima kelas, parameter/seed42/bobot voting yang sama. Preprocessing dan RUS+SMOTE hanya pada training. V10.358 serta additional760 tidak digunakan untuk fit, vocabulary, atau memilih bobot.')
data=[['Representasi WV','FP tambahan /760','FPR tambahan','Macro-F1 V','Serangan V<br/>menjadi Normal']]
labels=[('reference','Acuan85 (frozen)'),('matched_base_availability','+ metadata ketersediaan'),('matched_base_syntax','+ syntax'),('matched_base_tfidf','+ TF-IDF'),('matched_base_hybrid','+ syntax + TF-IDF')]
for key,label in labels:
    a=m(key,'additional');b=m(key,'V');data.append([label,a['false_alarms'],p(a['fpr_percent']),p(b['macro_f1_percent'],4),b['false_negatives']])
table(data,[52*mm,29*mm,28*mm,32*mm,37*mm])
add('<b>Interpretasi:</b> syntax menekan11 alarm (1,45 percentage points), TF-IDF9 alarm, sedangkan gabungan justru menambah10 alarm. False negatives V meningkat menjadi3 atau4 pada penambahan isi. Karena itu, refit langsung ini tidak memenuhi tujuan menurunkan FPR tanpa kehilangan deteksi.')
add('Koreksi hanya ketika ada isi','HeadP')
add('Binary classifier juga dicoba hanya pada alarm SQL yang mempunyai usable content. Threshold berasal dari G, dikonfirmasi pada C, lalu dibekukan sebelum penerapan pada V/additional. Beberapa kandidat mengoreksi64 kasus dan mempertahankan seluruh label V; <b>257/760 false alarms (33,82%) tetap tersisa</b>. Semua kandidat dan hasil negatif tetap disimpan.')
add('Apa yang dijalankan','HeadP')
table([['Eksperimen','Training dan predictor','Fit classifier baru'],['Binary Normal vs SQL','100 content nyata unik per kelas; syntax, TF-IDF, gabungan; masing-masing5 singleton +3 ensemble','48, termasuk OOF dan meta'],['Matched main lima kelas','A31.676; availability, syntax, TF-IDF, gabungan;5 base learner tiap representasi','20'],['Pemeriksaan konflik','Menggunakan model yang sudah diverifikasi; tidak melatih classifier lagi','0']],[38*mm,106*mm,34*mm])
add('Total68 fit baru selesai; OOF stacking memakai dua fold group-aware. Cap sintetis <=50% dari original per kelas sebelum RUS, hanya training. Full main stacking tidak direfit pada eksperimen matched; stacking diuji pada ketiga pendekatan binary. Naskah, dashboard, dan model utama lama tidak diganti.','SmallP')

page();add('Uji binary Normal vs SQL:<br/>tiga pendekatan, delapan predictor','TitleP')
add('Training200 contoh nyata nonempty dari A:100 Normal dan100 SQL. Test utama adalah <b>1.012 content nonempty yang berbeda dari training</b>, terdiri dari895 Normal dan117 SQL pada V. Hash content dinormalisasi lowercase; fold OOF dan test utama tidak berbagi content training identik. Nilai vocabulary/IDF/scaler dipelajari dari training saja.')
data=[['Predictor','Syntax saja<br/>FP / FN; Macro-F1','Character TF-IDF<br/>FP / FN; Macro-F1','Gabungan<br/>FP / FN; Macro-F1']]
methods=['decision_tree','random_forest','xgboost','lightgbm','logistic_regression','soft_voting','weighted_voting','stacking'];short=['DT','RF','XGBoost','LightGBM','LR','Soft voting','Weighted voting','Stacking']
for method,label in zip(methods,short):
    rr=[m('binary_'+view,'binary_content_disjoint',method) for view in ('syntax_only','tfidf_only','syntax_tfidf')]
    data.append([label]+[r['false_alarms']+' / '+r['false_negatives']+'; '+p(r['macro_f1_percent'],4) for r in rr])
table(data,[34*mm,48*mm,48*mm,48*mm],pad=5)
add('FP = Normal menjadi SQL; FN = SQL menjadi Normal. Penyebut tetap895 Normal dan117 SQL untuk semua sel. Weighted voting syntax: FPR0,2235%, recallSQL100%; TF-IDF: FPR0%, recall99,1453%; gabungan: FPR0%, recall100%.','SmallP')
add('Isi fitur yang dipelajari','HeadP')
add('<b>Syntax12:</b> enam fitur contoh dospem (kutip tunggal, OR, tanda =, numeric tautology, UNION SELECT, isdecimal), ditambah kutip ganda, AND, komentar baris/blok, titik koma dan SQL keyword. <b>TF-IDF:</b> character2-5 gram, maksimum2.048 fitur, lowercase, sublinear TF, normalisasiL2.')
add('HTTP request memakai nilai query/form yang didekode sekali; header identitas, path, Host, Cookie, Date, IP, dan label tidak menjadi fitur isi. MQTT memakai current body. Content dipotong maksimum4.096 byte. Binary label1 adalah SQL, sehingga loss weighting DDoS tidak dipakai.')
add('Batas interpretasi yang penting','HeadP')
add('Normal training berisi50 MQTT dan50 other TCP, sedangkan SQL berisi100 nilai query HTTP. <b>Perbedaan protokol dan capture masih dapat menjadi shortcut.</b> Hash-disjoint tidak sama dengan capture-independent. Nilai100% gabungan belum membuktikan benign HTTP dan malicious HTTP dapat dipisahkan pada capture baru. Test kosong dipisahkan dalam artefak dan tidak digunakan untuk klaim keunggulan isi.')

page();add('Prototipe pemeriksaan konflik:<br/>aturan dan confusion matrix','TitleP')
add('Ini predictor baru yang dirancang sesudah hasil sebelumnya diamati. Acuan WV tetap disimpan. Threshold RF0,5 ditetapkan sebagai default operasional; tidak dipilih melalui760 target tambahan.')
table([['Langkah','Aturan'],['1. Tentukan konflik','WV memilihSQL; RF memilihNormal; pRF(Normal)>=0,5. Selain itu, gunakanWV semula.'],['2. Periksa isi bila tersedia','Binary syntax+TF-IDF weighted voting memberi buktiSQL atau tie0,5/0,5: pertahankan alarmSQL. Jika tidak ada usable content, tetap diperlukan buktiRF pada langkah1.'],['3. Putuskan','Jika konflik lolos pemeriksaan, gunakan seluruh posteriorRF agar label dan probabilitas konsisten. Kelas WV selainSQL tidak diubah.']],[31*mm,147*mm],pad=5)
add('RF sudah merupakan expert di dalam voting; posterior semua expert dihitung sekali. Parsing dan model binary hanya dijalankan pada konflik dengan isi. Payload kosong sendiri tidak menyebabkan prediksiNormal.','SmallP')
add('Confusion matrix V: acuan dan prototipe identik','HeadP')
cm=rows(R/'reference_and_guard_V_confusion.csv');names=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware'];shortnames=['Normal','DDoS','Scan','SQL','Ransom']
table([['Aktual / prediksi']+shortnames]+[[label]+[r[c] for c in names] for label,r in zip(shortnames,cm)],[43*mm]+[27*mm]*5)
add('Baris = kelas sebenarnya; kolom = prediksi. Seluruh10.358 label V tetap sama, termasuk36 Normal yang salah dianggap serangan dan2 Port Scanning yang dianggapNormal. Macro-F1 tetap97,8800%; balanced accuracy dan recall tiap kelas juga tetap.','SmallP')
data=[['Recall V','Acuan WV','Prototipe']]
base=gm('reference','V');guard=gm('hybrid_WV','V')
for key,label in [('DDoS_TCP','DDoS TCP'),('Port_Scanning','Port Scanning'),('SQL_injection','SQL injection'),('Ransomware','Ransomware')]:data.append([label,p(base['recall_'+key],4),p(guard['recall_'+key],4)])
table(data,[78*mm,50*mm,50*mm])
add('<b>Ini preservasi empiris pada V ini.</b> Klaim jaminan prediction-preserving execution, kesetaraan SHAP dan replay speed model lama tidak otomatis berlaku pada guard baru. Posterior additional yang dikoreksi memang berubah.','SmallP')

page();add('Hasil per capture dan<br/>cakupan pengujian keselamatan','TitleP')
data=[['Additional Normal capture','Target matched','FP acuan','FP prototipe']]
for r in rows(G/'primary_per_capture.csv'):data.append([escape(r['source'].replace('_',' ')),r['n'],r['original_FP'],r['new_FP']])
data.append(['Total','760','321 (42,24%)','0 (0%)']);table(data,[76*mm,34*mm,34*mm,34*mm])
add('Pada sampel yang sama, guard RF-only tanpa payload juga menghasilkan0/760. Maka penurunan FPR ini didukung oleh resolusi konflik dengan RF; <b>keuntungan unik syntax/TF-IDF terhadap RF-only belum terbukti</b>. Binary content berperan sebagai pengaman untuk mempertahankan buktiSQL, bila bukti itu tersedia.')
add('Berapa serangan yang benar-benar mengaktifkan guard?','HeadP')
data=[['Partisi','Jumlah','Konflik eligible','Di antaranya serangan','Konflik berisi']]
for r in rows(G/'activation_coverage.csv'):data.append([r['cohort'],n(r['n']),r['eligible'],r['eligible_attack'],r['eligible_nonempty']])
table(data,[30*mm,30*mm,37*mm,46*mm,35*mm])
add('<b>Nol pada kolom serangan adalah keterbatasan, bukan keberhasilan deteksi guard.</b> Tidak ada konflik RF-Normal/WV-SQL pada G,C,V. Semua321 aktivasi berasal dari additional Normal,64 di antaranya berisi. Karena itu, label V yang identik belum menguji risiko guard membatalkan alarm serangan baru yang benar-benar berada dalam wilayah konflik.')
add('Batas klaim dan tindak lanjut ilmiah','HeadP')
add('Hasil tetap retrospektif/within-trace pada data yang sudah diamati. Target tambahan hanya760 Normal yang matched;107 kandidat yang sebelumnya tidak dapat disejajarkan tidak masuk penyebut. Hasil0/760 tidak berarti FPR populasi pasti nol atau sistem aman di semua capture/perangkat. Penilaian final keselamatan memerlukan serangan dan benign relevan yang memicu konflik, dengan aturan dibekukan sebelum evaluasi tersebut.')

page();add('Biaya komputasi dan<br/>paket yang dapat dijalankan','TitleP')
add('Benchmark native pipeline memakai base85/context yang sudah disiapkan: probabilitas lima expert, parsing isi kondisional, dan binary inference bila diperlukan. Single thread, warm-up3,10 pengulangan, urutan kandidat diacak tiap pengulangan. Ini bukan pengukuran full PCAP replay atau jalur compiled artikel.')
data=[['Batch','Predictor','Median ms','SD ms','Mean delta peak<br/>RSS MiB*']]
for cohort in ('V','additional'):
    for key,label in [('reference','Acuan WV'),('RF_only','RF-only guard'),('hybrid_WV','Hybrid guard')]:
        r=next(x for x in EFF if x['cohort']==cohort and x['key']==key);data.append([cohort+' '+r['n'],label,f(r['median_ms'],3),f(r['sd_ms'],3),f(r['delta_peak_rss_mean_mib'],3)])
table(data,[32*mm,44*mm,30*mm,29*mm,43*mm])
add('*Memori: dua proses cold terpisah, baseline sesudah load; nilai adalah delta peak RSS saat inferensi, bukan ukuran model atau RSS saat load. Seluruh raw latency100 pengukuran dan cold memory12 pengukuran tersedia.','SmallP')
add('Pada batch additional760, hybrid membutuhkan median61,109 ms vs acuan41,675 ms, sekitar46,63% lebih lambat; RF-only42,664 ms, sekitar2,37% lebih lambat. Dengan FPR teramati sama, RF-only adalah pembanding efisiensi yang lebih kuat. Keuntungan keselamatan tambahan payload perlu dibuktikan pada konflik serangan nyata.')
add('Verifikasi dan isi paket dospem','HeadP')
add('Independent verifier lulus:272 file run tersegel,26 fit plans,256 baris metrik training/evaluasi dan116 baris metrik guard; split/OOF/cap, vocabulary training-only, posterior konsisten dan identitas760 keluaran CLI diperiksa. Verifikasi tidak melatih classifier ulang. Model/naskah utama lama tetap sama.')
add('ZIP pendamping memuat dua run lengkap, model dengan checksum, input fitur yang dibekukan, indeks, konfigurasi, contoh200 content training, hasil semua predictor, posterior, confusion matrix, raw timing/RSS, environment, source, tests, dan verifier. Data mentah besar tidak diduplikasi; archive memungkinkan verifikasi hasil dan inferensi contoh tanpa training ulang.','SmallP')
add('Menjalankan prototype dari root proyek','HeadP')
add('Gunakan environment terisolasi yang cocok dengan environment.json. Jalankan <b>scripts/predict_sql_guard_20261105.py</b> dengan <b>--base</b> example_base760.csv, <b>--payload</b> example_payload760.csv, dan <b>--output</b> path CSV baru. Contoh perintah lengkap ada dalam README paket. CLI menerima85 fitur C60/O10 yang sudah diekstrak dan raw payload hex, bukan raw CSV63/PCAP langsung. Label tidak menjadi input keputusan.','SmallP')
add('<b>Kesimpulan untuk dospem:</b> penambahan payload langsung belum cukup. Pemeriksaan konflik dengan RF menghilangkan321 alarm teramati pada760 Normal tanpa perubahan label V. RF-only sama efektif dan lebih ringan pada data ini. Guard tetap prototipe eksploratif; belum ada bukti keselamatan pada serangan yang mengaktifkan konfliknya.','SmallP')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#c4d5e3'));c.line(16*mm,15*mm,194*mm,15*mm)
    c.setFont('Arial',7.5);c.setFillColor(colors.HexColor('#536b7d'));c.drawString(16*mm,10.5*mm,'TA Hafid | Audit false alarms dan prototipe | 5 November 2026');c.drawRightString(194*mm,10.5*mm,str(doc.page))
PDF.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(PDF),pagesize=(210*mm,297*mm),leftMargin=16*mm,rightMargin=16*mm,topMargin=16*mm,bottomMargin=21*mm,title='Audit dan Perbaikan False Alarms Normal-SQL',author='M. Hafid Ramadhan').build(story,onFirstPage=footer,onLaterPages=footer)
print(PDF)
