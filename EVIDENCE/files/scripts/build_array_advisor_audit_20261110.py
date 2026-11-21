"""Supervisor-facing audit and revision account, based on retained evidence."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib import colors
import build_fpr_solution_report_20261109 as L

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ARRAY_FINAL_INTEGRATION_20261110'


def main():
    assert json.loads((OUT/'verification.json').read_text())['new_tables_verified']==25
    L.story=[]
    def title(t):L.add('AUDIT REVISI JURNAL | 10 NOVEMBER 2026','SmallH');L.add(t,'TitleH')
    title('Revisi selesai, cakupan hasil harus tepat')
    L.add('Paper dan supplementary diperbarui menggunakan eksperimen yang sudah selesai. Tidak ada training, tuning, model baru, atau pengukuran runtime baru dalam integrasi naskah ini. Angka dihitung ulang dari prediksi tersimpan dan seluruh nilai tabel sebelumnya dipertahankan.')
    L.add('<b>FPR tidak berada di bawah 1% pada semua populasi evaluasi.</b> Weighted voting acuan memang menghasilkan 36/3.860 = 0,9326% pada Normal V. Aturan koreksi baru memberi hasil sangat rendah pada blok Normal lanjutan, tetapi FPR V justru menjadi 39/3.860 = 1,0104%. Kedua angka tidak boleh dipertukarkan.')
    L.table(['Populasi','Voting acuan','Guarded voting / GV','Makna'],[
        ['Normal V, n=3.860','36 FP; 0,9326%','39 FP; 1,0104%','Validasi internal yang telah digunakan'],
        ['Normal Edge blok 3, n=697','396 FP; 56,8149%','1 FP; 0,1435%','Aturan beku sebelum blok; source lama'],
        ['CIC Normal, n=2.000','603 FP; 30,15%','638 FP; 31,90%','Dataset eksternal'],
        ['TON Normal terpetakan, n=2.000','487 FP; 24,35%','377 FP; 18,85%','Dataset eksternal; label flow turunan']],[130,104,123,142])
    L.add('Pada TON, GV menganggap 1.657/2.000 target Ransomware terpetakan sebagai Normal (82,85%). Recall alarm biner 17,15%; recall jenis Ransomware 1,55%. Jumlah target Ransomware ke Normal bertambah 27 dibanding voting acuan.')
    L.add('Keberhasilan utama tetap: penambahan skala 10 detik memberi sekitar 1,14 percentage points macro-F1 dalam trace; compact execution menghemat mean replay 41,75% untuk predictor acuan; audit Shapley mengukur konsistensi permainan penjelasan yang didefinisikan. Generalisasi deteksi dan FPR <1% secara menyeluruh belum terbukti.','SmallH')
    L.page();title('Audit prioritas ilmiah dospem')
    L.table(['Permintaan','Bukti yang sudah tersedia','Status / batas'],[
        ['1. Validitas sumber dan history','Kontrol shared-density/private; pemetaan label; uji eksternal 9.000 target. Main Table 3 dan 14-15; S29/S39/S86-S94.','Dijawab dengan batas. Source/class Edge masih confounded; transfer eksternal buruk.'],
        ['2. Ablation context/ordering','C40/O1, C40/O10, C60/O1, C60/O10 pada split/model sama. Main Table 2, Figure 2; S25-S26/S38.','Selesai. Gain 1,1425-1,1475 pp tetap muncul dengan ordering ditahan.'],
        ['3. Nilai tambah weighted ensemble','RF/LR/uniform/WV, per-class recall, disagreement dua arah, biaya. Main Tables 4/7; S27-S30.','Selesai sebagai evaluasi objektif. Tidak ada klaim superioritas umum weighted voting.'],
        ['4. Selection bias / final evaluation','Riwayat holdout/V, pemilihan guard sesudah dua blok, freeze sebelum blok3/eksternal. Main study design; S34/S64/S93.','Transparan, belum konfirmasi deteksi umum. Label eksternal tidak seragam sebagai lima kelas.']],[126,216,157])
    L.add('Tidak ada perbaikan angka dengan mengganti predictor sesudah membaca hasil eksternal. CIC attack dilaporkan sebagai agreement skenario file. TON memakai pencocokan flow unik dan mengecualikan unmapped/ambigu. Cakupan in-scope TON hanya 62,90% dan 6,66%; hasil tidak diekstrapolasi ke seluruh paket.','SmallH')
    L.add('Permintaan menjaga deteksi saat menurunkan FPR','SubH')
    L.add('Tujuh cohort serangan historis memenuhi constraint empiris recall/FN/macro-F1, tetapi semuanya telah diamati saat aturan dipilih. Uji TON menambah missed Ransomware. Karena itu, penurunan FPR dalam trace tidak dipromosikan sebagai perbaikan universal atau jaminan serangan tidak terlewat.')
    L.page();title('Audit eksperimen pendukung dan narasi')
    L.table(['Permintaan','Perbaikan yang dipertahankan / ditambahkan'],[
        ['5. Runtime lebih kuat','30 pasangan replay, median/SD/bootstrap interval, kasus lambat dan workload sensitivity; main Table 7, Figure 4, S10/S33/S45. GV mempunyai 30 pasangan dan RSS terpisah di S81-S85; interval latency GV mencakup nol.'],
        ['6. Buffer / history window','0,25/1/4 MiB, capacity eviction, cohort purge dan query/update timing; main Table 6, S31-S32. Window nominal 10 detik tidak dijanjikan lengkap pada trafik padat.'],
        ['7. Explainability','History dipecah menjadi tiga kelompok 20 fitur; background sensitivity dan faithfulness dibanding RF/LR; main Table 8/Figure 5, S35-S36. Permainan 7/9 kelompok dibedakan.'],
        ['8. Novelty','Introduction/Related Work memosisikan endpoint statistics, voting, early exit dan SHAP sebagai metode mapan. Kontribusi berpusat pada representation, kontrak execution dan audit explanation.'],
        ['9. Results / Discussion','Hasil negatif payload, penurunan recall, perubahan FP V dan kegagalan transfer disajikan. Conclusion menjawab RQ1-RQ3 dengan angka dan cakupan yang tepat.'],
        ['10. Tabel, gambar dan sitasi','Rujukan eksplisit Table 3-8 dipertahankan; diagram alur diperbarui, satu Figure 8 dan tiga tabel main baru ditambahkan. S1-S95 lengkap, 30 referensi, metadata tiga paper dataset baru diperiksa melalui Crossref.']],[126,373])
    L.add('Batas kontrak predictor','SubH')
    L.add('f adalah weighted voting acuan yang memiliki bukti compact-execution dan penjelasan. g adalah GV yang mengganti posterior pada kondisi konflik tertentu. g memakai lima expert lama tanpa classifier payload. Jaminan prediction preservation, SHAP dan penghematan replay f tidak otomatis berlaku untuk g. Penamaan dan ruang lingkup ini dibuat konsisten di seluruh naskah.')
    L.page();title('Komentar terakhir dan deklarasi submission')
    L.table(['Butir dospem','Hasil audit / tindakan'],[
        ['False alarms dan confusion matrix','Distribusi source, kelas prediksi, flag alignment dan kontribusi expert tetap ada; Tables 13-15 serta S74-S94 menambah guard, seluruh kelas output, later blocks dan eksternal.'],
        ['Overlap artikel conference','Tidak ditambahkan sebagai narasi paper. Bukti historis tetap ditandai sebagai provenance; salinan AGERS, peta overlap dan draft disclosure editor dipertahankan terpisah. Status publikasi conference perlu dikonfirmasi saat submission.'],
        ['Table cross-references / Conclusion','Rujukan eksplisit dipertahankan dan semua tabel baru dirujuk. Conclusion membedakan gain konteks, FPR tiap cohort, saving predictor acuan dan kegagalan transfer.'],
        ['Supplement / reproducibility','S1-S95 lengkap; source LaTeX, ledger baris tabel, notebook, prediksi, konfigurasi/model manifests, timing/RSS dan checksum dikemas untuk review. Dataset raw penuh tidak disalin.'],
        ['Pendanaan / konflik / kontribusi','Tanpa hibah khusus; tidak ada konflik yang diketahui; CRediT mengikuti pembagian yang dikonfirmasi pengguna. Email Hafid tetap tercantum.'],
        ['Data/software availability','Menggunakan companion archive yang benar-benar disediakan. Tidak mengarang URL repository publik atau DOI yang belum ada.'],
        ['Final author approval','Frasa pending dihapus dari publikasi; tidak diganti dengan klaim seluruh penulis sudah menyetujui. Persetujuan versi final tetap harus benar-benar diperoleh sebelum submit.'],
        ['Turnitin / bantuan AI','Turnitin belum dijalankan dan tidak ada skor yang diklaim. Bahasa diedit untuk kejelasan dan ketepatan, dengan penggunaan bantuan AI diungkap secara faktual. Tidak menjanjikan AI score <20%.']],[126,373])
    L.add('Panduan Array tidak berhasil diakses langsung karena respons 403. Source memakai elsarticle; kebijakan umum Elsevier untuk highlights dan deklarasi AI diperiksa. Persyaratan portal/jurnal pada saat upload tetap perlu dicek oleh corresponding author. Tidak ada submission atau pesan kepada editor yang dilakukan.','SmallH')
    L.page();title('Apa yang berubah dari versi 9 November')
    L.table(['Bagian','Perubahan konkret'],[
        ['Abstract dan Introduction','Mengintegrasikan guard terakhir dan external test; memisahkan keberhasilan internal, perubahan predictor dan batas transfer.'],
        ['Methods','Definisi GV tanpa payload, kronologi freeze, target selection, ekstraksi PCAP, kontrol RAM, scope label CIC, pemetaan flow TON dan coverage.'],
        ['Results','Tiga tabel baru: Edge guard, external Normal FPR, serta attack-scenario/Ransomware. Figure 8 membandingkan FPR antarcohort; Figure 1 memperbarui keseluruhan alur.'],
        ['Discussion / Conclusion','Menegaskan V Normal FPR 1,0104%, later block 0,1435%, external 31,90%/18,85%, Ransomware miss82,85%, dan batas klaim efisiensi/XAI acuan.'],
        ['Supplement','Tabel S74-S95 menambah bukti lengkap guard, attack checks, confusion matrix, raw30-pair timings, RSS, eksternal, label coverage dan chronology. S1-S73 tetap bernilai sama.'],
        ['References / availability','26 referensi lama dipertahankan, tiga paper dataset dan satu resource resmi ditambahkan. Pernyataan akses data/software sesuai berkas yang benar-benar tersedia.']],[126,373])
    L.add('Berkas yang dikirim untuk review','SubH')
    L.add('1. Main manuscript: 45 halaman, 15 tabel, 8 gambar, 30 referensi.<br/>2. Supplementary: 76 halaman, S1-S95 lengkap, 6 gambar.<br/>3. Laporan audit ini: status komentar dospem dan perbedaan versi.<br/>4. ZIP review berisi PDF/source/highlights/catatan; archive reproducibility terpisah berisi evidence lengkap.')
    L.add('Integrasi ini mempertahankan 12 badan tabel main dan 73 badan tabel supplement versi sebelumnya secara identik. Verifier menghitung ulang 24 hasil policy/cohort historis, tiga blok Normal dan tujuh sumber eksternal. Tidak ada hasil training yang diciptakan atau metrik yang diubah agar tampak memenuhi target.','SmallH')
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#d6e0e7'));c.line(48,39,547,39);c.setFont('ArialH',8);c.setFillColor(colors.HexColor('#556b7e'));c.drawString(48,26,'TA Hafid | Audit revisi Array | 10 November 2026');c.drawRightString(547,26,str(doc.page))
    path=ROOT/'output/pdf/ARRAY_HAFID_AUDIT_REVISI_DOSPEM_20261110.pdf'
    SimpleDocTemplate(str(path),pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=52,title='Audit revisi jurnal dan supplementary',author='M Hafid Ramadhan').build(L.story,onFirstPage=footer,onLaterPages=footer)
    print(path)


if __name__=='__main__':main()
