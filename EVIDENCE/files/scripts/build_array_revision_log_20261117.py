"""Native TeX revision log with traceable findings, actions and unresolved matters."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/ARRAY_REVISION_20261117';D=ROOT/'paper/array_20261117_revision/latex'

def main():
    records=[
      ('1. Denominator Table S90','S90; Sections 5.4/6.4; S86-S95',
       'Keterangan lama menyebut pengeluaran warm-up dari denominator, tetapi formula kode memakai seluruh paket.',
       'Definisi dikoreksi menjadi 100 x (Normal + Ransomware terpetakan setelah warm-up) / seluruh paket. Warm-up tetap dalam denominator. Angka 62.9036% dan 6.6602% dipertahankan. Kategori mapping dijumlahkan kembali terhadap seluruh paket; formula dan hash source dicatat dalam audit.'),
      ('2. Posisi WV terhadap RF dan uniform voting','Sections 3.2/5.2/6.1-6.2; Table 3; S39',
       'RF private-history 96.9966% melebihi WV 93.7694%; temporal uniform voting 97.8971% sedikit melebihi WV 97.8954%.',
       'RF unggul sekitar 3.23 percentage points pada kontrol private history, dihitung dari skor tidak dibulatkan. WV dipertahankan sebagai fungsi referensi yang dioptimasi eksekusinya, bukan hasil pemilihan ulang classifier terbaik. Bobot referensi tidak dituning ulang untuk kontrol history; keterbatasan komparasinya dijelaskan.'),
      ('3. Independensi data dan history dependence','Sections 3.1/5.1/6.1/6.5; Table 2; S39',
       'Satu capture per kelas mengaitkan source dan label. Signature split memisahkan pola supervised tetapi tidak seluruh riwayat paket.',
       'Peran temporal V, signature diagnostic dan holdout historis diperjelas. Penurunan WV 3.5262 pp pada private history dihitung dari nilai tidak dibulatkan. Kontrol shared-density mencegah atribusi kausal seluruh perubahan kepada sharing. Tidak ada klaim independensi source dari signature split.'),
      ('4. Pemisahan reference WV dan guarded GV','Sections 4.5/5.4/6.2-6.4; Tables 6-7; Figures 1/5',
       'GV dipilih setelah error beberapa cohort diketahui; kontrak eksekusi/XAI WV tidak otomatis berlaku untuk fungsi yang diubah.',
       'WV adalah f referensi dan GV adalah g final. Cohort desain (initial, blocks 1-2, tujuh cohort serangan teramati) dibedakan dari block 3 setelah freeze. Block 3 tetap dari source yang telah dipakai. Istilah diagram diperjelas; 41.75% dan audit Shapley tetap hanya untuk WV.'),
      ('5. False alarms dan hasil eksternal','Section 5.4/6.4; Tables 6-7; Figure 5; S90/S97/S103-S104',
       '324/760 sebelum alignment berbeda dari 321/760 sesudah alignment. Keberhasilan block 3 tidak mewakili hasil V atau source eksternal.',
       'Hasil 1/697 GV ditafsirkan sebagai perbaikan lokal. V Normal FPR 1.0104%, external CIC 31.90%, TON 18.85% dan TON type recall 1.55% tetap ditonjolkan. Efek cabang rescue/removal dan error RF dibahas tanpa mengklaim sebab tunggal. CIC scenario agreement tidak digabung dengan TON flow-derived recall sebagai multiclass generalization.'),
      ('6. Batas runtime dan explainability','Sections 5.2-5.3/6.2-6.3; Tables 4-5; S35-S36',
       'Saving 41.75% milik replay WV satu laptop. Exact finite-game Shapley tidak membuktikan kausalitas atau explainability unggul.',
       'Workload 15,665 paket/160 keputusan, 30 pasangan dan prepared-feature RF/LR yang lebih cepat dipertahankan. Tidak ada klaim energi atau embedded efficiency. Implementation consistency, donor sensitivity dan faithfulness dipisahkan; audit seven-group/nine-group tidak dipool. Caption biaya menggunakan istilah recorded, bukan seolah eksperimen baru.'),
      ('7. Integrasi narasi ilmiah','Abstract; Sections 1/6/7',
       'Narasi perlu menghubungkan kontribusi, bukti dan batas secara konsisten.',
       'Novelty dijelaskan melalui bounded representation dengan ablation ordering tetap, kontrak eksekusi fungsi yang sama, dan audit finite-game explanation. Discussion disusun RQ1-RQ3, lalu false alarms dan validity. Conclusion menjawab RQ dengan hasil Section 5. Abstract diperbarui terakhir, termasuk kegagalan eksternal dan pemisahan WV/GV.'),
      ('8. Reproducibility dan supplementary','Data/software availability; S1-S104; Figures S1-S9',
       'Reviewer memerlukan data turunan, ID, konfigurasi, seed dan bukti hasil yang dapat diperiksa tanpa mengandalkan angka naskah.',
       'Enam audit hasil tersimpan dijalankan ulang di folder baru. Split/resampling gates, saved predictions/confusion matrices, mapping 4,000 label TON, 30 pasangan runtime/RSS dan finite-game arrays diperiksa. Arsip penuh mempertahankan seluruh artefak induk, ditambah sumber revisi dan verifikasinya; tidak memuat dataset mentah lengkap atau kredensial.'),
      ('9. Konsistensi akhir dan kemasan','Seluruh artikel, supplement, cover letter dan ZIP',
       'Main text dilaporkan sulit ditemukan dalam ZIP sebelumnya; cross-references dan istilah harus cocok setelah revisi.',
       'Main.tex berisi naskah lengkap dan diletakkan pada root ZIP baru, dengan PDF bernama 01_MAIN_MANUSCRIPT. Supplement dan cover letter terpisah. Semua 7+104 baris data tabel dibandingkan terhadap versi 10 November; hanya header/definisi/catatan yang diubah. S1-S104/Figure S1-S9, sitasi, label dan rendering diperiksa; clean compile dan pemeriksaan arsip dicatat sebelum rilis.'),
      ('10. Keputusan eksperimen tambahan','Sections 6.5/7; batas metodologis',
       'Eksperimen tambahan hanya diperlukan bila kontribusi diperluas melampaui bukti yang tersedia.',
       'Tidak ada eksperimen baru pada revisi ini. Cakupan tetap pada gain representasi dalam evaluasi tersedia, prediction-preserving execution untuk WV yang sama, dan quantitative explanation audit pada fungsi terdefinisi. Generalisasi lebih kuat, reliabilitas deteksi eksternal dan efektivitas luas GV tetap memerlukan source independen baru dengan aturan/model dibekukan sebelum pengujian.')]
    (R/'revision_steps.json').write_text(json.dumps([dict(step=a,location=b,finding=c,action=d) for a,b,c,d in records],indent=2,ensure_ascii=False)+'\n','utf8')
    unresolved=[
      'Source/class confounding dan shared history tidak hilang melalui perbaikan tulisan. Klaim tetap within-trace dan sesuai label eksternal yang tersedia.',
      'GV belum memiliki bukti prediction-preserving compact execution atau explanation-preservation tersendiri. Saving 41.75% dan audit WV tidak dialihkan.',
      'False alarms dan recall eksternal masih buruk. Tidak ada klaim semua FPR di bawah 1% atau deployment readiness.',
      'SecurityEvents_GroundTruth asli TON tidak tersedia dari tautan terlindungi; label tetap turunan processed flows dan coverage terbatas. Dataset mentah tetap mengikuti izin pemilik.',
      'Arsip reviewer disiapkan sebagai file nyata dan diverifikasi lokal. Belum diunggah ke portal jurnal atau hosting persisten; tidak ada DOI/repository publik yang dibuat-buat. Upload/akses reviewer aktual harus dilengkapi saat submission.',
      'Persetujuan final seluruh penulis dan pengiriman belum dikonfirmasi. Declarations funding/COI/CRediT mengikuti jawaban yang telah dikonfirmasi pengguna; bantuan AI tetap diungkapkan.',
      'Hubungan dengan artikel conference tetap tersedia pada catatan khusus penulis/editor beserta salinan pembanding. Tidak dimasukkan sebagai klaim kontribusi baru jurnal atau disembunyikan dari persiapan editorial.'
    ]
    md='# Persoalan terbuka setelah revisi Array 17 November 2026\n\n'+'\n\n'.join(str(i+1)+'. '+v for i,v in enumerate(unresolved))+'\n\nPersoalan ilmiah di atas merupakan batas bukti yang dipertahankan, bukan alasan untuk mengarang hasil atau mengulang eksperimen tanpa protokol. Step 10 tidak menuntut eksperimen baru selama tiga kontribusi tetap dibatasi. Submission belum dilakukan.\n'
    (ROOT/'docs/ARRAY_PERSOALAN_TERBUKA_20261117.md').write_text(md,'utf8')
    def esc(s):
        for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_')]:s=s.replace(a,b)
        return s
    t=r'''\documentclass[12pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{amsmath,booktabs,array,xurl}
\usepackage[hidelinks]{hyperref}
\setlength{\textwidth}{162mm}\setlength{\textheight}{245mm}
\setlength{\oddsidemargin}{-1.4mm}\setlength{\evensidemargin}{-1.4mm}
\setlength{\topmargin}{-4mm}\setlength{\headheight}{0pt}\setlength{\headsep}{0pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\emergencystretch}{2em}
\begin{document}\fontsize{10.5}{12.3}\selectfont
\begin{center}{\Large\bfseries Revision log and evidence audit}\par\medskip
M Hafid Ramadhan and Satria Mandala\par
17 November 2026\end{center}
\textbf{Article:} Multiscale endpoint context and prediction-preserving execution for explainable IoT intrusion detection.\par
\textbf{Acuan:} main dan supplementary ringkas 10 November 2026. Revisi ini merupakan pemeriksaan ulang bukti dan perubahan dokumen, tanpa training, pemuatan model, inference atau profiling baru. Angka hasil tidak diubah untuk menyesuaikan narasi.\par
\textbf{Koreksi utama S90:} denominator adalah seluruh paket. Warm-up dikeluarkan dari hitungan label yang eligible pada numerator, tetap dimasukkan ke denominator.
\begin{center}\begin{tabular}{lrrrr}\toprule
TON file & Seluruh paket & Label eligible & Warm-up & Coverage (\%)\\\midrule
1 & 357,819 & 225,081 & 4,256 & 62.9036\\
2 & 1,905,124 & 126,885 & 475 & 6.6602\\\bottomrule
\end{tabular}\end{center}
Denominator alternatif setelah warm-up akan menghasilkan 63.6608\% dan 6.6619\%; angka itu dicatat untuk audit, bukan mengganti hasil asli. Rumus kode semula benar untuk definisi all-packet; keterangan tabel semula yang tidak konsisten.\par
\textbf{Bukti perhitungan:} \nolinkurl{reports/ARRAY_REVISION_20261117/S90_denominator_audit.csv} dan laporan audit yang disertakan.
'''
    for a,b,c,d in records:
        t+='\n\\subsection*{'+esc(a)+'}\n\\textbf{Lokasi:} '+esc(b)+'\\par\n\\textbf{Temuan:} '+esc(c)+'\\par\n\\textbf{Tindakan:} '+esc(d)+'\n'
    t+='\n\\section*{Persoalan yang tidak dapat diselesaikan dari bukti tersedia}\n'
    for i,v in enumerate(unresolved,1):t+='\\textbf{'+str(i)+'.} '+esc(v)+'\\par\n'
    t+=r'''
\section*{Berkas dan pemeriksaan sebelum penyerahan}
Paket review memuat main manuscript, supplementary S1--S104, log ini, cover letter terpisah, serta source LaTeX dan gambar. Arsip reproduksi penuh disediakan terpisah dengan manifest SHA256 dan perintah audit saved outputs. Verifikasi menghindari fitting/deserialisasi dan mempertahankan tanggal run historis.\par
Main document untuk Overleaf adalah \texttt{main.tex} pada root ZIP, bukan \texttt{cover\_letter.tex}. PDF utama diberi awalan \texttt{01\_MAIN\_MANUSCRIPT}. Cover dan supplement berada dalam subfolder terpisah yang mempunyai dependency sendiri.\par
Source artikel/supplement/cover menggunakan LaTeX native, bukan konversi teks PDF. Empat belas gambar dapat dibuat ulang dari kode plotting dan data turunan yang disertakan. Tidak ada pengiriman jurnal, email, Turnitin atau klaim approval penulis baru pada workflow ini.
\end{document}
'''
    (D/'revision_log.tex').write_text(t,'utf8');print('Revision log source and unresolved list prepared')

if __name__=='__main__':main()
