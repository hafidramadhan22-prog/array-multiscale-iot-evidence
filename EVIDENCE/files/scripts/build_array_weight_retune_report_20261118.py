"""Four-page author report for the audited G-only WV reweighting experiment."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/ARRAY_WEIGHT_RETUNE_20261118';D=json.loads((R/'analysis.json').read_text())
assert json.loads((R/'verification.json').read_text())['status']=='PASS_INDEPENDENT_WEIGHT_SEARCH_AND_PREDICTIONS'
for n,f in [('Arial','arial.ttf'),('ArialB','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(n,'C:/Windows/Fonts/'+f))
NAVY=colors.HexColor('#173F58');TEAL=colors.HexColor('#147D78');PALE=colors.HexColor('#EDF4F6')
style=getSampleStyleSheet();style.add(ParagraphStyle(name='B',fontName='Arial',fontSize=9.2,leading=12.8,spaceAfter=7))
style.add(ParagraphStyle(name='S',parent=style['B'],fontSize=7.8,leading=10.4,spaceAfter=6))
style.add(ParagraphStyle(name='H',fontName='ArialB',fontSize=18,leading=22,spaceAfter=11,textColor=NAVY))
style.add(ParagraphStyle(name='H2',fontName='ArialB',fontSize=11.5,leading=15,spaceBefore=7,spaceAfter=6,textColor=NAVY))
style.add(ParagraphStyle(name='C',fontName='Arial',fontSize=7.4,leading=9.9))
style.add(ParagraphStyle(name='CH',parent=style['C'],fontName='ArialB',textColor=colors.white))
W=A4[0]-36*mm
def P(t,s='B'):return Paragraph(t,style[s])
def table(rows,widths):
    t=Table([[P(str(c),'CH' if i==0 else 'C') for c in row] for i,row in enumerate(rows)],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5.5),('BOTTOMPADDING',(0,0),(-1,-1),5.5),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#C8D8DF'))]));return t
def pct(v,d=4):return '-' if v is None else f'{100*v:.{d}f}'
def footer(c,doc):
    c.setStrokeColor(TEAL);c.setLineWidth(1.1);c.line(18*mm,A4[1]-15*mm,A4[0]-18*mm,A4[1]-15*mm)
    c.setFont('ArialB',8);c.setFillColor(NAVY);c.drawString(18*mm,A4[1]-11.5*mm,'TA HAFID | ARRAY | G-ONLY WV REWEIGHTING')
    c.setFont('Arial',8);c.drawRightString(A4[0]-18*mm,A4[1]-11.5*mm,'18 November 2026')
    c.setFont('Arial',7.4);c.setFillColor(colors.HexColor('#52636A'));c.drawString(18*mm,11*mm,'Eksperimen eksploratif - expert dan naskah jurnal tetap');c.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))

def build():
    story=[P('Uji penentuan ulang bobot WV','H'),P('Hasil utama: ada perbaikan lokal, belum konsisten','H2')]
    story += [P('Pada packet-private history, kebijakan utama menaikkan macro-F1 dari <b>93,7694% ke 94,1734%</b> dan menurunkan Normal FPR dari <b>0,5041% ke 0,4578%</b>. Namun SQL recall turun. Pada temporal evaluation, macro-F1 justru turun dan FPR naik. RF serta SVM default tetap lebih tinggi pada private history.')]
    methods=['WV_reference','WV_G_constrained','WV_G_unconstrained','WV_G_heuristic','RF','SVM_default','SVM_G_selected'];short=['WV lama','WV-G utama','WV-G bebas','WV-G heur','RF','SVM-D','SVM-G']
    rows=[['Macro-F1 (%)']+short]
    for protocol,title in [('temporal','Temporal, mean 3 seed'),('packet_private','Packet-private, pooled OOF'),('signature_shared','Signature/shared, pooled OOF')]:rows.append([title]+[pct(D['protocols'][protocol][m]['macro_f1']) for m in methods])
    story += [table(rows,[120]+[(W-120)/7]*7),Spacer(1,6),P('Temporal: 10.358 target yang sama per seed. Signature/private: 52.584 target unik pooled OOF. SVM-D adalah baseline default; SVM-G dipilih pada G dalam eksperimen sebelumnya. Angka bukan hasil gabungan antarprotokol. DT, XGB, LGB, LR dan uniform juga tersedia dalam CSV lengkap.','S')]
    story += [P('Apa yang diuji','H2'),P('Lima expert, preprocessing, fitur C60/O10, resampling dan split tidak berubah. Hanya bobot voting dipilih ulang dari probabilitas expert yang telah tersimpan dan terverifikasi. Ada <b>3.881 kandidat bobot per tahap</b>, tujuh tahap, total 27.167 penilaian G; ini bukan jumlah training classifier. Tidak ada refit expert atau loading joblib.')]
    story += [P('<b>WV-G utama:</b> pencarian bobot dengan syarat pada G: Normal false alarms tidak bertambah dan correct-type recall setiap kelas serangan tidak berkurang. <b>WV-G bebas:</b> tujuan yang sama tanpa syarat tersebut. <b>WV-G heur:</b> kualitas G dihitung ulang memakai heuristik lama. Semua memaksimalkan rerata macro-F1 dan DDoS F2 di G; pilihan dibekukan sebelum V dinilai.')]
    story += [P('Grid memakai floor 0,05 dan langkah 0,05; bobot referensi tetap eligible. Data V telah diamati pada penelitian sebelumnya, sehingga proses ini tetap exploratory follow-up. Syarat empiris pada G tidak menjamin hasil V. Tidak ada perubahan grid atau pemilihan kebijakan berdasarkan hasil V.','S')]
    story += [PageBreak(),P('Per kelas, perubahan error, dan bobot','H')]
    rows=[['Private recall (%)','Normal','DDoS','Scan','SQL','Ransom','Errors']]
    for m,title in [('WV_reference','WV lama'),('WV_G_constrained','WV-G utama'),('WV_G_unconstrained','WV-G bebas'),('WV_G_heuristic','WV-G heur'),('RF','RF'),('SVM_default','SVM default')]:
        v=D['protocols']['packet_private'][m];rows.append([title]+[pct(v['recall_'+c]) for c in ['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']]+[str(v['errors'])])
    story += [table(rows,[105]+[(W-105)/6]*6),Spacer(1,6),P('Kebijakan utama mengoreksi 318 error lama, tetapi mengubah 199 prediksi yang tadinya benar menjadi salah: net 119 error lebih sedikit (3.071 menjadi 2.952). SQL recall turun dari 99,6473% ke 99,4892%; karena itu syarat tidak mengorbankan seluruh recall serangan belum terpenuhi pada V.')]
    old=D['protocols']['packet_private']['WV_reference'];new=D['protocols']['packet_private']['WV_G_constrained']
    countold=f"{old['attack_to_normal']:,}".replace(',','.');countnew=f"{new['attack_to_normal']:,}".replace(',','.')
    recold=pct(old['binary_attack_recall']).replace('.',',');recnew=pct(new['binary_attack_recall']).replace('.',',')
    story += [P(f"Serangan yang diprediksi Normal juga bertambah dari {countold} menjadi {countnew}. Binary attack recall turun dari {recold}% menjadi {recnew}%. Perbaikan macro-F1 tidak berarti seluruh kemampuan mendeteksi alarm meningkat.")]
    story += [P('Temporal juga perlu dipertahankan dalam pembahasan','H2'),P('Rerata macro-F1 temporal turun 97,8954% menjadi 97,8189%; Normal FPR naik 0,9240% menjadi 1,1054%. Rerata error naik 184,67 menjadi 193,33. Pada signature/shared, seluruh kebijakan menghasilkan label evaluasi yang sama seperti WV lama. Heuristik yang dihitung ulang memberi kenaikan private sangat kecil (93,7913%) dan tidak mengubah hasil temporal/shared.')]
    story += [P('Bobot kebijakan utama per tahap','H2')]
    rows=[['Tahap','DT','RF','XGB','LGB','LR']]
    for s in D['weights']:
        if s['policy']=='WV_G_constrained':rows.append([s['stage'].replace('temporal_','temp_').replace('signature_','sig_')]+[f"{s[n]:.2f}" for n in ['DT','RF','XGB','LGB','LR']])
    story += [table(rows,[145]+[(W-145)/5]*5),Spacer(1,5),P('Setiap fold/seed memiliki expert dan bobot sendiri. Tidak ada satu bobot pooled yang dilatih dari label gabungan V. Bobot private berbeda dari bobot temporal yang dipakai untuk stress scoring. RF tidak dinaikkan bobotnya berdasarkan keunggulan RF yang baru diketahui dari V. Bobot tabel dibulatkan dua desimal; nilai presisi penuh dengan jumlah satu tersedia dalam CSV/JSON.','S')]
    story += [PageBreak(),P('False alarms dan transfer lintas sumber','H')]
    rows=[['Normal cohort','n','WV lama: FP / FPR','WV-G utama: FP / FPR','RF: FP / FPR']]
    for fam,title in [('additional_aligned','Additional aligned'),('later_block3','Later block 3'),('CIC','CIC Normal'),('TON','TON Normal')]:
        v=D['stress'][fam];rows.append([title,str(v['WV_reference']['normal_support'])]+[f"{v[m]['false_alarms']} / {pct(v[m]['normal_fpr'],2)}%" for m in ['WV_reference','WV_G_constrained','RF']])
    story += [table(rows,[104,45]+[(W-149)/3]*3),Spacer(1,7),P('<b>Stress memakai predictor temporal seed-42 dan bobotnya sendiri.</b> Tidak memakai rerata bobot private folds. Pada 760 target benign aligned, false alarms turun menjadi 0; pada block 3 menjadi 1/697. Ini hasil lokal pada sumber yang telah digunakan, tanpa tuning pada cohort tersebut dalam eksperimen ini.')]
    rows=[['Recall / agreement (%)','Support','WV lama','WV-G utama','RF']]
    for family,title,key,n in [('CIC','CIC SYN, correct-type recall','recall_DDoS_TCP',1000),('CIC','CIC attack-scenario binary alarm recall','binary_attack_recall',3000),('TON','TON Ransomware, correct-type recall','recall_Ransomware',2000),('TON','TON Ransomware, binary alarm recall','binary_attack_recall',2000)]:
        v=D['stress'][family];rows.append([title,str(n)]+[pct(v[m][key],2) for m in ['WV_reference','WV_G_constrained','RF']])
    story += [table(rows,[215,50]+[(W-265)/3]*3),Spacer(1,7)]
    story += [P('Pada TON, Normal FPR turun dari 24,35% menjadi 8,15%, tetapi Ransomware yang diprediksi Normal bertambah dari 1.630 menjadi 1.769 dari 2.000 target. Binary alarm recall turun dari 18,50% menjadi 11,55%. Correct-type recall naik sedikit, namun tidak menghapus kenaikan missed alarms.')]
    story += [P('Pada CIC, Normal FPR justru naik dari 30,15% menjadi 36,10%. Correct-type recall pada target skenario SYN turun menjadi 0%, walau binary alarm recall lintas tiga skenario meningkat. Menandai serangan dengan jenis yang salah berbeda dari deteksi jenis yang benar.')]
    story += [P('CIC memakai label skenario file; TON memakai label turunan author flows dengan coverage terbatas. Keduanya tidak digabung menjadi skor multiclass lintas dataset. Sumber sudah pernah diamati dalam penelitian sebelumnya; hasil ini tidak membuktikan generalisasi atau deployment readiness.','S')]
    story += [PageBreak(),P('Confusion matrix dan audit bukti','H')]
    story += [Image(str(R/'figures/private_confusions.png'),width=W,height=W*4.8/9),Spacer(1,5),P('Packet-private pooled OOF, 52.584 target. Teks = actual counts; warna = proporsi baris. RF, SVM dan seluruh kebijakan lain tetap disajikan dalam tabel/CSV serta confusion matrix terpisah.','S')]
    story += [P('Pemeriksaan yang telah lulus','H2'),P('Verifier terpisah menghitung ulang seluruh 27.167 kandidat G, memeriksa bobot/argmax tie, formula heuristik, batas bobot, syarat G, pemilihan, identitas split dan parent hashes, lalu membentuk ulang probabilitas serta label V/stress. Ada 84 vektor label tahap evaluasi, 28 vektor probabilitas WV, dan 220 vektor label stress. Empat unit tests lulus. Tidak ada deserialisasi model atau refit expert.')]
    story += [P('Keputusan untuk saat ini','H2'),P('<b>Belum ada dasar untuk mengganti model acuan jurnal dengan bobot baru.</b> Ada gain private sebesar 0,4040 percentage points dan penurunan false alarms lokal, tetapi terdapat penurunan temporal, SQL recall private dan trade-off eksternal. RF/default SVM tetap lebih tinggi pada private history. Semua hasil, termasuk negatif, dipertahankan.')]
    story += [P('Mengubah bobot mengubah fungsi prediksi. Hasil runtime 37,40%/41,75%/60,55% dan audit SHAP predictor lama tidak otomatis berlaku pada kebijakan baru. Tidak dilakukan profil runtime, audit XAI baru, capture baru, perubahan naskah, submission, atau klaim persetujuan penulis.','S')]
    story += [P('Audit portabel: <b>python -B scripts/verify_array_weight_retune_20261118.py</b>. Tabel, prediksi, G candidates, source snapshot, manifest dan notebook tersedia dalam paket evidence. Refleksi utamanya: model selection di G pada sumber yang sama tetap tidak menjamin kinerja V atau capture eksternal.','S')]
    pdf=ROOT/'output/pdf/ARRAY_HAFID_UJI_BOBOT_G_EVIDENCE_20261118.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=22*mm,bottomMargin=19*mm,title='Uji G-only weight retuning - Array',author='M Hafid Ramadhan')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(pdf)
if __name__=='__main__':build()
