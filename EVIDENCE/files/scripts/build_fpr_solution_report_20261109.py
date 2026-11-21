"""Create a six-page supervisor report from independently audited results."""
from pathlib import Path
import json,html
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/FPR_SOLUTION_20261109';RUN=ROOT/'runs/FPR_SQL_GUARD_20261109_01';K='sql_guard_rf_rescue'
FONT=Path('C:/Windows/Fonts');pdfmetrics.registerFont(TTFont('ArialH',str(FONT/'arial.ttf')));pdfmetrics.registerFont(TTFont('ArialHB',str(FONT/'arialbd.ttf')));pdfmetrics.registerFontFamily('ArialH',normal='ArialH',bold='ArialHB',italic='ArialH',boldItalic='ArialHB')
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyH',fontName='ArialH',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#273849')));styles.add(ParagraphStyle(name='SmallH',parent=styles['BodyH'],fontSize=8.2,leading=11,spaceAfter=5));styles.add(ParagraphStyle(name='TitleH',parent=styles['BodyH'],fontName='ArialHB',fontSize=21,leading=26,spaceAfter=12));styles.add(ParagraphStyle(name='SubH',parent=styles['BodyH'],fontName='ArialHB',fontSize=13,leading=17,spaceBefore=8,spaceAfter=8));styles.add(ParagraphStyle(name='CellH',parent=styles['BodyH'],fontSize=8.4,leading=11,spaceAfter=0));styles.add(ParagraphStyle(name='HeadH',parent=styles['CellH'],fontName='ArialHB',textColor=colors.white))
W=499;story=[]
def p(s,style='BodyH'):return Paragraph(s,styles[style])
def add(s,style='BodyH'):story.append(p(s,style))
def title(n,s):add('LAPORAN EKSPERIMEN | 9 NOVEMBER 2026','SmallH');add(s,'TitleH')
def table(headers,rows,widths):
    data=[[p(html.escape(str(v)),'HeadH') for v in headers]]+[[p(html.escape(str(v)),'CellH') for v in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173c5a')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#173c5a')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f8'),colors.white]),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dbe3e9'))]));story.extend([t,Spacer(1,9)])
def img(name,width=W):
    from PIL import Image as PI
    path=R/'figures'/name
    with PI.open(path) as im:h=width*im.height/im.width
    story.append(Image(str(path),width=width,height=h));story.append(Spacer(1,7))
def page():story.append(PageBreak())
def main():
    assert json.loads((R/'verification.json').read_text())['status']=='PASS'
    m=pd.read_csv(R/'metrics_all_cached.csv');b=pd.read_csv(R/'metrics_later_blocks.csv');per=pd.read_csv(R/'metrics_per_capture.csv');g=pd.read_csv(R/'independent_safety_gates.csv');eff=pd.read_csv(RUN/'efficiency_summary.csv');paired=pd.read_csv(R/'paired_latency_comparison.csv');d=pd.read_csv(R/'predictions_all.csv',dtype={'row_id':str},low_memory=False);cause=pd.read_csv(R/'expert_disagreement_cause.csv')
    title(1,'Solusi FPR di bawah 8%')
    add('<b>Target tercapai pada sampel yang diuji.</b> Aturan SQL guard + RF rescue menghasilkan 1/697 false alarm (0,1435%) pada blok Normal ketiga. Weighted voting lama menghasilkan 396/697 (56,8149%) pada target yang sama. Aturan sudah dibekukan sebelum blok ketiga dinilai.')
    rows=[]
    old=m.query("cohort=='additional760'").set_index('policy');rows.append(['760 target lama','321/760 (42,24%)','0/760 (0%)','Retrospektif'])
    for bi in [1,2,3]:
        x=b.query('block==@bi').set_index('policy');r=x.loc['weighted_reference'];n=x.loc[K];rows.append([f'Blok {bi}: {int(r.n)} target',f'{int(r.false_alarms)}/{int(r.n)} ({r.FPR_percent:.2f}%)',f'{int(n.false_alarms)}/{int(n.n)} ({n.FPR_percent:.2f}%)','Dibekukan sebelum blok' if bi==3 else 'Retrospektif'])
    table(['Evaluasi Normal','Voting lama','Aturan baru','Status untuk aturan baru'],rows,[115,128,118,138])
    img('FPR_comparison.png')
    add('Blok ketiga merencanakan 700 target dari tujuh capture yang sudah ada. Sebanyak 697 dapat dipetakan (99,57%); tiga tidak diganti. Jika ketiganya juga dianggap false alarm, batas worst-case menjadi 4/700 = <b>0,5714%</b>, tetap di bawah 8%. Ini batas coverage, bukan confidence interval populasi.','SmallH')
    add('Angka 42,63% pada revisi lama adalah 324/760 sebelum koreksi format flag. Baseline yang dipakai di sini telah dikoreksi: 321/760 = 42,24%. Semua perbandingan menggunakan target yang sama pada masing-masing cohort.','SmallH')
    page();title(2,'Apa yang diubah dan mengapa')
    add('Seluruh 321 false alarm pada 760 Normal lama diprediksi sebagai <b>SQL injection</b> oleh weighted voting. RF tidak menghasilkan false alarm pada 760 target itu. Ini menunjukkan masalah praktis pada keputusan agregasi untuk trafik Normal tersebut; belum membuktikan penyebab kausal atau generalisasi lintas capture.')
    table(['Expert','Bobot tetap','FP / 760','Prediksi SQL','Margin tertimbang SQL-Normal*'],[[r.model,f'{r.weight:.6f}',int(r.false_alarms),int(r.SQL_predictions),f'{r.weighted_mean_SQL_minus_Normal_on_321_WV_false_alarms:+.6f}'] for r in cause.itertuples()],[115,72,75,83,154])
    add('*Rata-rata pada 321 target yang salah oleh voting. Margin DT +0,149300 dan LR +0,182493 mengarah ke SQL; RF -0,097531 dan XGB -0,136580 mengarah ke Normal. Margin adalah diagnostic keluaran model, bukan atribusi kausal fitur. Bobot dan model tidak dituning ulang.','SmallH')
    add('Aturan final yang diuji','SubH')
    table(['Kondisi','Posterior yang dipakai'],[['Voting = SQL injection; RF = Normal','Gunakan lima probabilitas RF.'],['Voting = Normal; RF = serangan','Gunakan lima probabilitas RF untuk memulihkan alarm.'],['Kondisi lainnya','Pertahankan lima probabilitas weighted voting.']],[260,239])
    add('Aturan ini mempertahankan prediksi non-SQL dari voting ketika RF berkata Normal. Pemeriksaan ini penting: alternatif awal yang menyerahkan semua keputusan Normal kepada RF melewatkan satu Port Scanning pada signature fold 1. Alternatif itu disimpan sebagai hasil negatif dan tidak dipilih sebagai solusi akhir.')
    add('Tidak ada training baru, model dasar baru, threshold baru, atau capture baru. Payload syntax/TF-IDF tidak dipakai oleh solusi ini. Fitur tetap context85 (c60_o10); label, nama capture, IP/row ID, dan payload mentah bukan input predictor ini. Konteks endpoint menggunakan alamat untuk membangun history sesuai extractor lama, bukan sebagai kolom identifier model.','SmallH')
    add('<b>Ini predictor yang diubah.</b> Jaminan prediction preservation, SHAP, dan penghematan replay dari voting lama tidak otomatis berlaku untuk aturan ini.','SmallH')
    page();title(3,'Deteksi serangan dan trade-off')
    v=m.query("cohort=='V_seed42'").set_index('policy');rows=[]
    for n,label in [('weighted_reference','Voting lama'),('random_forest','RF saja'),(K,'Aturan baru')]:
        r=v.loc[n];rows.append([label,f'{int(r.false_alarms)}/3860 ({r.FPR_percent:.4f}%)',f'{int(r.attack_to_Normal)}/6498',f'{r.macro_F1_percent:.6f}%',f'{r.balanced_accuracy_percent:.6f}%'])
    table(['Predictor','FP pada Normal','Serangan ke Normal','Macro-F1','Balanced accuracy'],rows,[95,128,97,90,89])
    img('CM_V_seed42.png')
    rows=[]
    for c,label in [('DDoS_TCP','DDoS TCP'),('Port_Scanning','Port Scanning'),('SQL_injection','SQL injection'),('Ransomware','Ransomware')]:rows.append([label,int(v.loc[K,'support_'+c]),f"{v.loc['weighted_reference','recall_'+c]:.6f}%",f"{v.loc[K,'recall_'+c]:.6f}%"])
    table(['Kelas sebenarnya','n','Recall voting','Recall aturan baru'],rows,[150,55,147,147])
    add('Recall keempat kelas serangan pada V seed42 sama dengan voting. Namun, FP Normal bertambah tiga dan macro-F1 turun 0,029769 percentage points. Dua true Port Scanning yang sebelumnya dianggap Normal menjadi Ransomware: kini memicu alarm, tetapi <b>jenis serangannya tetap salah</b>. Jadi binary detection 100% pada V bukan klasifikasi lima kelas 100%. DDoS F2 tetap 100%.','SmallH')
    page();title(4,'Kontrol evaluasi dan sumber data')
    add('Seluruh tujuh cohort yang memuat empat kelas serangan lolos pemeriksaan empiris: recall tiap kelas tidak turun, attack-to-Normal tidak bertambah, dan penurunan macro-F1 maksimal 0,1 pp. Batas 0,1 pp adalah keputusan operasional studi, bukan angka yang diklaim disetujui dospem.')
    table(['Cohort historis','Delta macro-F1 (pp)','Delta serangan ke Normal','Recall empat kelas'],[[r.cohort,f'{r.macro_F1_delta_pp:+.6f}',f'{r.attack_to_Normal_delta:+d}','Tidak turun'] for r in g.itertuples()],[140,119,130,110])
    add('Blok ketiga: hasil per capture','SubH')
    q=per.query('block==3 and policy==@K');table(['Capture Normal','Rencana / dinilai','FP / dinilai','FPR'],[[r.source,f'{int(r.planned)} / {int(r.n)}',f'{int(r.false_alarms)} / {int(r.n)}',f'{r.FPR_percent:.4f}%'] for r in q.itertuples()],[170,120,115,94])
    add('Target blok 1 adalah indeks paket 100999-199999, blok 2 adalah 200999-299999, dan blok 3 adalah 300999-399999; langkah 1000. Target antarblok tidak beririsan. HeartRate hanya cukup panjang untuk blok 1; tujuh source lain dipakai pada blok 2/3. CSV dan PCAP diverifikasi full SHA-256 sebelum diproses. Mapping memakai identitas paket, bukan menganggap indeks baris CSV sama dengan paket.','SmallH')
    add('History prior-only tetap berjalan sejak awal capture (.1/1/10 detik; ring16384 record). Target lama direplay dengan perbedaan probabilitas maksimal 0. Format flag semantik tetap. Source, history, dan kedekatan temporal tetap berkorelasi. Dua blok pertama dan data serangan lama sudah diketahui saat aturan final dipilih; hanya blok ketiga dibekukan sebelum scoring untuk aturan final, dan isinya Normal saja.','SmallH')
    page();title(5,'Efisiensi dan bukti per prediksi')
    img('latency_comparison.png')
    rows=[]
    for r in eff.itertuples():rows.append([r.cohort,{'random_forest':'RF','weighted_reference':'Voting',K:'Aturan baru'}[r.policy],f'{r.median:.3f}',f'{r.mean:.3f} +/- {r.std:.3f}',f'{r.RSS_delta_mean_MiB:.3f} +/- {r.RSS_delta_SD_MiB:.3f}'])
    table(['Batch','Predictor','Median ms','Mean +/- SD ms','Delta peak RSS MiB'],rows,[61,90,75,132,141])
    add('30 pengulangan berpasangan, urutan acak, tiga warmup, satu thread; RSS diukur pada tiga cold process dengan sampling 1 ms, baseline setelah model dan input dimuat. Batch V1000 adalah sampel tetap dari V; batch Normal adalah 697 target blok ketiga. Ini biaya inferensi prepared features, bukan end-to-end PCAP/replay.','SmallH')
    add('CI bootstrap 95% perbedaan mean latency (aturan baru minus voting): '+ '; '.join(f'{r.cohort}: [{r.bootstrap_95_low_ms:.3f}, {r.bootstrap_95_high_ms:.3f}] ms' for r in paired.query('policy==@K').itertuples())+'. Keduanya mencakup nol; <b>tidak ada klaim percepatan yang meyakinkan</b>. RF singleton jauh lebih cepat, tetapi pada cache tertentu mengubah jenis serangan yang semula benar.','SmallH')
    add('Satu false alarm yang tersisa','SubH')
    bad=d.query("cohort=='later_block3' and candidate_prediction!='Normal'").iloc[0]
    add(f'Target {html.escape(str(bad.row_id))}: true Normal; voting lama {bad.weighted_prediction}; RF {bad.RF_prediction}; aturan baru {bad.candidate_prediction}. False alarm ini tetap ada pada kedua predictor. Cabang SQL guard tidak mengubahnya karena voting memprediksi Ransomware, bukan SQL. Kasus ini tetap dihitung dalam FPR.')
    add('CLI memuat hanya model milik proyek dengan SHA-256 yang dikenal. Replay 697 feature rows cocok dengan saved candidate labels dan probabilitas, selisih maksimal 0. Dua belas tes menguji aturan disagreement, probabilitas float64, pemakaian RF satu kali, dan risiko salah menekan SQL baru. Semua lulus.','SmallH')
    page();title(6,'Berkas pemeriksaan dan batas klaim')
    table(['Berkas / folder dalam paket','Isi yang dapat diperiksa'],[['FPR_SOLUSI_DAN_PREDIKSI_20261109.ipynb','14 sel kode telah dieksekusi, 12 output gambar; tabel, confusion matrix, dan filter prediksi aktual. Data review tertanam.'],['reports/FPR_SOLUTION_20261109/','Prediksi per baris, metric CSV, confusion matrix, paired latency, expert diagnostic, verification dan replay audit.'],['runs/FPR_SQL_GUARD_20261109_01/','Konfigurasi beku, hash/source snapshot, raw target terpilih, context85 features, posterior, log, model dan benchmark.'],['runs/FPR_TARGET8_... dan FPR_NORMAL_VOTING_...','Hasil alternatif terdahulu, termasuk kegagalan signature dan profile gate awal yang diperbaiki dengan akumulasi float64.'],['scripts/predict_fpr_sql_guard_rf_rescue_20261109.py','CLI inferensi terpisah pada feature CSV context85, dengan verifikasi model/source sebelum deserialisasi.'],['scripts/verify_fpr_solution_20261109.py','Audit independen saved probabilities tanpa model loading/fitting.'],['docs/FPR_TARGET8_SQL_GUARD_PROTOCOL_20261109.md','Definisi keputusan, chronology model selection, target, coverage dan batas evaluasi.']],[248,251])
    add('Cara memeriksa ulang','SubH')
    add('1. Buka notebook untuk melihat output yang sudah tersimpan; tidak perlu memuat model.<br/>2. Jalankan verifier dari root paket dengan environment terisolasi sesuai requirements.<br/>3. Untuk replay, jalankan script predict dengan --features runs/FPR_SQL_GUARD_20261109_01/all_later_features.csv dan --output tmp/replay_baru.csv. Output harus memakai path baru.<br/>4. Arsip menyertakan prepared target features dan model. Full CSV/PCAP tidak diduplikasi; hash dan path sumber dicatat. Ekstraksi ulang dari raw memerlukan arsip proyek induk dan raw asli.','SmallH')
    add('Keputusan ilmiah yang dapat dipertahankan','SubH')
    add('Kandidat ini memenuhi target FPR kurang dari 8% pada sampel Normal yang dinilai dan tidak menurunkan recall kelas serangan pada tujuh cohort historis. Ini cukup untuk dibawa ke dospem sebagai hasil eksperimen baru dengan evidence yang dapat ditelusuri. <b>Belum ada bukti independen bahwa FPR akan selalu di bawah 8% atau serangan baru tidak akan terlewat.</b> SQL asli dapat ditekan menjadi Normal jika RF keliru; pengujian serangan independen masih diperlukan.')
    add('Paper dan supplementary publikasi sebelumnya tetap arsip. Eksperimen ini belum mengubah dashboard atau naskah jurnal, belum menjalankan ulang SHAP/replay preservation untuk predictor baru, dan belum disubmit. Gunakan angka sesuai cohort, bukan menggabungkan protokol. Tidak ada klaim Turnitin, persetujuan penulis baru, atau acceptance jurnal.','SmallH')
    dest=ROOT/'output/pdf/SOLUSI_FPR_DI_BAWAH_8PERSEN_EVIDENCE_20261109.pdf';dest.parent.mkdir(parents=True,exist_ok=True)
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#d6e0e7'));c.line(48,39,547,39);c.setFont('ArialH',8);c.setFillColor(colors.HexColor('#556b7e'));c.drawString(48,26,'TA Hafid | Eksperimen perubahan predictor | 9 November 2026');c.drawRightString(547,26,f'{doc.page}')
    SimpleDocTemplate(str(dest),pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=52,title='Solusi FPR di bawah 8 persen - evidence Hafid',author='M. Hafid Ramadhan - laporan hasil eksperimen').build(story,onFirstPage=footer,onLaterPages=footer)
    print(dest)
if __name__=='__main__':main()
