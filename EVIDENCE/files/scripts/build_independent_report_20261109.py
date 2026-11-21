"""Six-page supervisor report; every number comes from verified saved results."""
from pathlib import Path
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib import colors
import build_fpr_solution_report_20261109 as L
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/INDEPENDENT_CAPTURE_20261109';RUN=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01';K='sql_guard_rf_rescue';P=['weighted_reference','random_forest',K];N={'weighted_reference':'Voting lama','random_forest':'RF',K:'Aturan koreksi'}
def main():
    assert json.loads((OUT/'independent_verification.json').read_text())['unique_targets']==9000
    m=pd.read_csv(OUT/'primary_policy_metrics_verified.csv');pool=pd.read_csv(OUT/'dataset_summaries.csv');cover=pd.read_csv(OUT/'TON_label_coverage.csv');changes=pd.read_csv(OUT/'rule_changes.csv');L.R=OUT;L.story=[]
    L.title(1,'Uji independen: target FPR belum tercapai')
    L.add('<b>Hasil 0,14% pada capture lama belum menggeneralisasi.</b> Pada data eksternal, aturan SQL guard + RF rescue menghasilkan FPR 31,90% pada 2.000 target Normal CICIoT2023 dan 18,85% pada 2.000 target Normal TON_IoT yang berhasil dipetakan ke label flow. Kedua nilai melampaui target 8%.')
    L.table(['Data Normal eksternal','Voting lama','RF','Aturan koreksi'],[[family+' (n=2000)']+[f"{int(pool.loc[(pool.dataset==family)&(pool.policy==p)].iloc[0].Normal_FP)}/2000 ({pool.loc[(pool.dataset==family)&(pool.policy==p)].iloc[0].FPR_percent:.2f}%)" for p in P] for family in ['CIC','TON']],[139,120,120,120])
    L.img('FPR_independent.png')
    L.add('TON pada gambar memakai Normal yang terpetakan dari dua file berisi Normal/Ransomware; bukan seluruh paket dalam file. Pooling hanya ringkasan deskriptif. Dua file tidak otomatis mewakili dua sesi independen.','SmallH')
    L.add('9.000 target unik dinilai dengan tiga predictor utama: 27.000 keputusan dan seluruh probabilitas lima kelas tersimpan. Model, bobot, aturan koreksi, dan preprocessing dibekukan; tidak ada training atau tuning ulang sesudah hasil eksternal diketahui.')
    L.add('Kelima nama kategori terwakili melalui CIC dan TON dengan kekuatan label yang berbeda. Hasil ini tidak diperlakukan sebagai satu testbed independen lima kelas atau satu macro-F1 gabungan.','SmallH')
    L.page();L.title(2,'Normal: perbaikan tidak konsisten')
    rows=[]
    for source in ['CIC_Normal0','CIC_Normal3','TON_Ransom1','TON_Ransom2']:
        a=m.query('source==@source').set_index('policy');rows.append([source]+[f"{int(a.loc[p,'false_alarms'])}/1000 ({a.loc[p,'FPR_percent']:.1f}%)" for p in P])
    L.table(['Source','Voting lama','RF','Aturan koreksi'],rows,[139,120,120,120])
    L.add('Pada dua file Normal CIC, kandidat menambah 35 false alarm secara total: 603 menjadi 638. Pada Normal TON yang terpetakan, jumlahnya turun 487 menjadi 377, tetapi penurunan tidak konsisten antarfile. File TON kedua justru naik dari 17,2% menjadi 22,3%.')
    L.add('Mengapa aturan koreksi dapat menambah false alarm','SubH')
    L.add('Cabang pertama hanya menekan prediksi SQL saat RF memprediksi Normal. Cabang kedua mengangkat prediksi voting Normal menjadi serangan bila RF memprediksi serangan. Pada trafik Normal baru, cabang kedua juga dapat mengangkat alarm yang keliru. Alarm non-SQL dari voting tetap dapat lolos.')
    L.table(['Source','FP Normal baru','FP Normal dikoreksi','Selisih FP'],[[r.source,int(r.Normal_FP_added),int(r.Normal_FP_corrected),f'{int(r.Normal_FP_added-r.Normal_FP_corrected):+d}'] for r in changes.query("source in ['CIC_Normal0','CIC_Normal3','TON_Ransom1','TON_Ransom2']").itertuples()],[160,113,113,113])
    L.add('Angka perubahan di atas berasal dari pasangan prediksi pada target yang sama. Distribusi kelas prediksi salah dan contoh row ID tersedia di notebook/CSV. Ini menjelaskan mekanisme keputusan; belum membuktikan fitur atau source tertentu sebagai penyebab kausal.')
    L.add('Ekstraksi baru dibandingkan dengan 94 target Edge lama: seluruh 85 fitur cocok. Kontrol tambahan pada 1.000 target CIC Normal3 juga menghasilkan fitur identik antara ekstraksi native dan alur diseksi terpilih. Hasil buruk eksternal tetap dipertahankan, tidak diperbaiki dengan mengubah bobot atau threshold.','SmallH')
    L.page();L.title(3,'Ransomware: deteksi masih lemah')
    rows=[]
    for source in ['TON_Ransom1','TON_Ransom2']:
        for policy in P:
            r=m.query('source==@source and policy==@policy').iloc[0];rows.append([source,N[policy],int(r.attack_to_Normal),f'{r.attack_recall_percent:.2f}%',f'{r.recall_Ransomware:.2f}%'])
    L.table(['Source','Predictor','Ransomware ke Normal /1000','Recall biner','Recall jenis Ransomware'],rows,[98,109,116,78,98])
    L.add('Pada 2.000 target berlabel Ransomware dari pemetaan flow, kandidat memprediksi <b>1.657 sebagai Normal (82,85%)</b>. Sebanyak 343 memicu salah satu alarm serangan (recall biner 17,15%), dan hanya 31 diprediksi sebagai jenis Ransomware yang benar (recall jenis 1,55%).')
    for source in ['TON_Ransom1','TON_Ransom2']:
        cm=pd.read_csv(RUN/source/('CM_'+K+'.csv'),index_col=0);L.add(source+' - confusion matrix kandidat','SubH');L.table(['Label flow sebenarnya','Normal','DDoS','Scan','SQL','Ransom'],[[truth]+[int(v) for v in cm.loc[truth].values] for truth in ['Normal','Ransomware']],[149,70,70,70,70,70])
    L.add('Label TON di sini diturunkan dari CSV flow penulis melalui IP/port/protokol bidirectional dan interval waktu. Angka ini berlaku pada target yang terpetakan secara unik; bukan anotasi serangan per paket asli. Ransomware yang memicu alarm dengan jenis lain tetap dianggap salah pada recall jenis.','SmallH')
    L.add('FPR yang lebih rendah pada sebagian Normal TON belum memenuhi syarat keselamatan deteksi. Dibanding voting, kandidat menambah 27 target Ransomware yang menjadi Normal.','SmallH')
    L.page();L.title(4,'CIC: pisahkan label skenario dari paket')
    rows=[]
    for source in ['CIC_SQL','CIC_PortScan','CIC_DDoS_SYN']:
        for policy in P:
            r=m.query('source==@source and policy==@policy').iloc[0];rows.append([source,N[policy],f'{int(r.n-r.errors)}/1000',int(r.attack_to_Normal)])
    L.table(['Skenario file','Predictor','Prediksi sesuai kelas skenario','Prediksi Normal /1000'],rows,[128,114,133,124])
    L.add('PCAP CIC memuat trafik pada skenario serangan. Penamaan folder/file belum membuktikan setiap paket merupakan aksi serangan. Karena itu, tabel menunjukkan kesesuaian dengan label skenario dan jumlah prediksi Normal, bukan recall serangan paket yang terkonfirmasi.')
    L.add('Pada Port Scanning, jumlah prediksi Normal naik 465 menjadi 555. Pada SYN, jumlahnya naik 2 menjadi 12. Beberapa alarm yang semula salah jenis dapat ditekan menjadi Normal oleh guard. Temuan ini membatasi klaim bahwa koreksi FPR tidak mengorbankan kemampuan mendeteksi serangan.')
    L.add('Sumber, pemilihan target dan input model','SubH')
    L.add('CIC menggunakan BenignTraffic, BenignTraffic3, SqlInjection, Recon-PortScan, dan DDoS-SYN_Flood. Masing-masing 1.000 target dipilih merata menurut indeks paket setelah warm-up 10 detik, tanpa memilih berdasarkan keluaran model. History memakai seluruh paket sebelumnya dalam file, jendela 0,1/1/10 detik dan ring 16.384 record.')
    L.add('CSV flow CIC tidak dimasukkan langsung ke model. Field protokol diekstrak ulang dari PCAP, lalu digabung dengan multiscale context menjadi 85 fitur yang sama. Konversi PCAPNG ke nanosecond PCAP mempertahankan jumlah paket serta waktu awal/akhir secara numerik tepat.','SmallH')
    L.add('Source resmi CIC: <link href="https://www.unb.ca/cic/datasets/iotdataset-2023.html">UNB CICIoT2023</link>. Registrasi dilakukan setelah persetujuan pengguna. Beberapa berkas diperoleh dari mirror publik dengan checksum LFS yang dicatat; file SQL dan dokumentasi berasal dari portal resmi. Tidak ada klaim bahwa checksum mirror merupakan checksum resmi penulis.','SmallH')
    L.page();L.title(5,'Cakupan label dan pemeriksaan teknis')
    L.table(['TON file','Total paket','Normal terpetakan','Ransom terpetakan','Paket in-scope (%)'],[[r.source,int(r.total),int(r.normal),int(r.ransomware),f'{r.in_scope_labelled_percent:.2f}'] for r in cover.itertuples()],[130,95,105,105,64])
    L.add('Unmapped: file1 128.482 paket; file2 1.777.764 paket. Paket tanpa label yang pasti tidak diubah menjadi Normal. Sampel 1.000 per kelas dipilih hanya dari paket yang terpetakan dan melewati warm-up; populasi evaluasi TON terbatas pada bagian tersebut.')
    L.add('Pemetaan memakai 23 CSV Network_dataset yang cocok hash, pasangan IP/port/protokol bidirectional, dan interval [ts, ts + duration + 1 detik). Tambahan satu detik mengakomodasi timestamp sumber yang terkuantisasi. Label yang saling bertentangan ditandai ambigu. Seluruh 4.000 label target TON diperiksa ulang terhadap semua flow yang berpotensi cocok.')
    L.add('Optimasi interval menghasilkan label dan target yang sama persis dengan implementasi linear pada seluruh paket terpetakan file pertama, termasuk target yang dipilih. Tiga tes batas interval/overlap lulus.','SmallH')
    L.add('Penggunaan RAM dan kesetaraan fitur','SubH')
    L.add('Diseksi native SYN dihentikan saat RAM hampir penuh. Alur pengganti mempertahankan seluruh paket dari flow TCP target, semua non-TCP, fragmen IP, dan TCP DNS untuk diseksi protokol. History tetap menghitung setiap paket asli. Target tidak diganti. Kontrol native CIC Normal3 (1.000 target) membuktikan seluruh 85 fitur identik sebelum alur dipakai pada SYN.')
    L.add('11 tes baru memeriksa formatting/field yang hilang, interval label dan pemilihan flow. Audit terpisah menghitung ulang confusion matrix, FPR, recall serta probabilitas; semua cocok dengan artefak tersimpan. Model memiliki SHA-256 tetap 77aa1beb...efcf31d. Tidak ada fit baru.','SmallH')
    L.add('Source TON: <link href="https://research.unsw.edu.au/projects/toniot-datasets">UNSW TON_IoT</link>. Tautan asli membutuhkan akses login; PCAP dan CSV berlabel diperoleh melalui mirror publik. Ground-truth yang dipakai adalah label flow turunan, bukan tabel SecurityEvents asli. Mirror, commit, ukuran dan hash setiap file dicatat.','SmallH')
    L.page();L.title(6,'Bukti dan implikasi untuk jurnal')
    L.table(['Berkas','Isi'],[['Notebook UJI_INDEPENDEN_HASIL_DAN_PREDIKSI','18 sel kode telah dieksekusi, 16 gambar, tabel, confusion matrix, filter seluruh prediksi.'],['all_predictions_verified.csv','27.000 keputusan utama dengan row ID, label/scope dan lima probabilitas.'],['all_policy_metrics_verified.csv','Hasil expert dan konfigurasi tersimpan sebagai diagnostic; tidak memilih ulang model.'],['runs/INDEPENDENT_CAPTURE_20261109_01','9.000 input, indeks paket, label/flow mapping, source beku, model dan posterior; katalog SHA-256.'],['independent_verification.json','Audit independen saved probabilities, jumlah sampel, CM dan 4.000 label TON.'],['CSV sumber dan metadata download','Asal PCAP, commit mirror, checksum, konversi serta cakupan label. Cookie dan data registrasi pribadi tidak dimasukkan ke paket.']],[228,271])
    L.add('Perubahan klaim yang diperlukan','SubH')
    L.add('Hasil 0,14% tetap boleh dilaporkan sebagai hasil pada blok lanjutan dalam trace Edge yang sudah digunakan. Hasil itu tidak mendukung klaim FPR di bawah 8% secara umum. Evaluasi independen ini memperlihatkan keterbatasan transfer lintas lingkungan dan perlunya kontrol source/capture yang lebih kuat.')
    L.add('Penggunaan guard harus tetap dinyatakan sebagai perubahan predictor eksploratif. Keselamatan deteksi tidak dijamin: pada TON terdapat Ransomware yang ditekan menjadi Normal; pada CIC terdapat alarm skenario yang juga hilang. Hasil negatif harus dibahas bersama perbaikan dalam trace sebelumnya.')
    L.add('Jika model atau aturan diperbaiki setelah membaca hasil ini, dataset evaluasi yang sudah diamati menjadi bagian pengembangan. Penilaian final berikutnya memerlukan capture atau sesi lain yang belum dipakai untuk memilih perubahan tersebut. Tidak ada training lanjutan atau perubahan dashboard/paper yang dilakukan dalam uji ini.')
    L.add('Untuk dospem','SubH')
    L.add('Pak, uji independen sudah selesai. Target FPR belum tercapai: kandidat 31,9% pada Normal CIC dan 18,85% pada Normal TON yang terpetakan. Deteksi Ransomware juga masih rendah. Saya lampirkan prediksi per paket, confusion matrix, konfigurasi/model beku dan audit label agar keterbatasan generalisasi dapat diperiksa sebelum revisi naskah berikutnya.','SmallH')
    destination=ROOT/'output/pdf/UJI_INDEPENDEN_CIC_TON_HASIL_DAN_EVIDENCE_20261109.pdf'
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#d6e0e7'));c.line(48,39,547,39);c.setFont('ArialH',8);c.setFillColor(colors.HexColor('#556b7e'));c.drawString(48,26,'TA Hafid | Uji independen model beku | Protokol 9 November 2026');c.drawRightString(547,26,str(doc.page))
    SimpleDocTemplate(str(destination),pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=52,title='Uji independen CICIoT2023 dan TON_IoT - hasil dan evidence',author='M. Hafid Ramadhan - laporan eksperimen').build(L.story,onFirstPage=footer,onLaterPages=footer);print(destination)
if __name__=='__main__':main()
