"""Write the author-review change log from the completed experiment."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ARRAY_ENHANCEMENT_20261117';DEST=ROOT/'paper/array_20261117_enhanced/latex'
def esc(s):
    for a,b in [('&',r'\&'),('_',r'\_'),('%',r'\%'),('#',r'\#')]:s=s.replace(a,b)
    return s
def main():
    j=json.loads((OUT/'verification.json').read_text('utf8'));assert j['status'].startswith('PASS_')
    contrasts=pd.read_csv(OUT/'paired_contrasts.csv').set_index('contrast')
    changes=[
      ('Kebaruan dan Related Work','Sections 1--3; S96/S114','Related Work sebelumnya sekitar190 kata untuk banyak sitasi.','Diperluas menjadi tiga bagian kritis: representasi/evaluasi, serving/preservation dan explanation reliability. QWYC, Hummingbird, CATO, FedSHAP dan attribution-fidelity diposisikan sebagai pendahulu. RQ dan batas evidence dipetakan, tanpa klaim algoritma dasar baru.'),
      ('Kontrak eksekusi','Section 4.3; S106','Sertifikat dan batas software tersebar dalam paragraf.','Ditambahkan langkah input/schema, full-posterior path, unresolved-row stopping dan canonical fallback. Bound1MiB diperjelas hanya untuk record array, bukan totalRSS atau model.'),
      ('Benchmark 2x2 dan baseline setara','Sections 4.4/5.2/6.2/7; Table4; S105--S109','Sesi lama tidak memisahkan seluruh kombinasi full/certified dan native/compact, serta singleton belum memakai encoding setara.','Delapan policy diukur pada sesi baru: empat jalur WV, RF native/compact dan LR native/compact. 480 panggilan feature,240 replay dan80 cold RSS; satu laptop, model/target tetap, tanpa refit. Semua hasil ditampilkan. Tabel biaya lama dipertahankan sebagaiS105; 41.75% tetap hasil sesi historis dan tidak dipool.'),
      ('Interpretasi biaya','Section 5.2; Table4; S107--S109','Keunggulan versus native singleton dapat dipengaruhi tingkat optimasi encoding.','Analisis menunjukkan perbandingan berubah menurut batch dan implementasi. LR compact lebih cepat daripada compactWV pada workload yang dicatat. WV certified tetap lebih cepat daripada compactRF untuk1row, tetapi singleton lebih cepat untuk1000row. Tidak membuat skor composite atau memilih ulang classifier.'),
      ('Explainability rinci','Sections 5.3/6.3/7; S110--S112','Rata-rata100target dan top3set Jaccard belum menunjukkan semua variasi.','Ditambahkan perkelas20target, correct/incorrect dan ordered-rank/Kendalltau-b. WV97benar/3salah; ketiga prediksi salah tetap memiliki positive faithfulness. Orderedtop3sama99/100 meski setJaccard1. Tidak mengubah grouping atau melakukan SHAP inference baru.'),
      ('Variasi dan gain representasi','Table1 note; S38/S113','Mean dan SD antar-seed berada di lokasi berbeda.','SD empat sel ditampilkan pada noteTable1; gain dan pengurangan error dihitung ulang berpasangan perseed dengan ordering tetap. Tiga seed bukan tiga capture; tidak dibuat CI generalisasi berdasarkan row yang berkorelasi.'),
      ('Presisi dan bahasa','Sections 5.4/Abstract/Discussion/Conclusion','RF apostrophe ter-escape sebagai aksen dan ada persentase enam desimal dalam prosa.','Apostrophe diperbaiki,97.880025/97.850256% ditampilkan97.8800/97.8503%; presisi penuh tetapCSV. Abstract/Conclusion menggunakan37.40% dari sesi baru untuk perbandingan native-certified versus compact-certified; 41.75% historis tetap dijelaskan padaResults/Discussion.'),
      ('Reproducibility','Data/software availability; S114; claim index; ZIP','Bukti lengkap tetapi memerlukan banyak entry points.','Satu script menjalankan delapan saved-result audits tanpa memuat estimator. Claim index mencatat exact paths/SHA/command/scope. Actual outputs tiga workload disimpan dan diperiksa terhadap fungsi native. Paket LaTeX tetap menempatkan main.tex lengkap pada root; cover/supplement/log terpisah.')]
    (OUT/'enhancement_changes.json').write_text(json.dumps([dict(topic=a,location=b,before=c,after=d) for a,b,c,d in changes],indent=2,ensure_ascii=False)+'\n','utf8')
    intro=r'''\documentclass[12pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{geometry,booktabs,array,xurl}
\usepackage[hidelinks]{hyperref}\geometry{margin=20mm}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\emergencystretch}{2em}
\begin{document}\fontsize{10.5}{12.6}\selectfont
\begin{center}{\Large\bfseries Penguatan artikel Array: log perubahan dan bukti}\par
M Hafid Ramadhan dan Satria Mandala\par17 November 2026\end{center}
\textbf{Acuan:} versi revisi dospem17November (main26halaman/supplement93halaman,S1--S104). Pengguna kemudian mengizinkan penguatan dari crosscheck. Versi sebelumnya tetap arsip; tidak ditimpa.\par
\textbf{Perubahan eksperimen:} satu sesi profiling/inference baru pada model dan workload lama, tanpa training/tuning/data baru. Ini bukan final test independen. Smoke terpisah tidak digunakan sebagai hasil artikel.\par
\textbf{Bukti baru:} delapan policy,480feature calls,240replay calls dan80cold-process RSS measurements. Seluruh label cocok dengan fungsi native masing-masing; full posteriors dan sparse values/structure diperiksa. Actual outputs tersedia dalam NPZ, bukan hanya angka naskah.\par
\section*{Perbedaan baseline yang harus dibaca dengan tepat}
'''
    intro+=r'\begin{tabular}{p{.42\linewidth}rr}\toprule Perbandingan replay & Mean saving s & Reduction \%\\\midrule'+'\n'
    for key,label in [('encoding_certified','Native-certified minus compact-certified WV'),('combined_WV','Native-full minus compact-certified WV'),('RF_encoding','RF native minus RF compact'),('LR_encoding','LR native minus LR compact')]:
        c=contrasts.loc[key];intro+=esc(label)+f' & {c.mean_saving_s:.4f} & {c.relative_reduction_percent:.2f}'+r'\\'+'\n'
    intro+=r'''\bottomrule\end{tabular}\par
Reduction adalah1 minus ratio of means. Nilai37.40\% dan60.55\% memakai baseline berbeda, sehingga tidak boleh dianggap dua estimasi dari perbandingan yang sama. Penghematan41.75\% adalah sesi3November, dilaporkan terpisah. Seluruh hasil tetap hanya satu laptop dan workload15,665packet/160decision, tanpa klaim energi/ARM/deployment.\par
'''
    for a,b,c,d in changes:intro+=r'\subsection*{'+esc(a)+r'}\textbf{Lokasi:} '+esc(b)+r'\par\textbf{Sebelum:} '+esc(c)+r'\par\textbf{Sesudah:} '+esc(d)+'\n\n'
    intro+=r'''\section*{Pelestarian hasil dan pemeriksaan}
Enam badan tabel main lama tetap identik. Badan Table4 lama dipindahkan utuh keS105. Seluruh104badan tabel supplementary lama identik; S106--S114 adalah tambahan. Tiga puluh referensi tetap, tanpa menambah sitasi sekadar memenuhi jumlah. Lima gambar main dan sembilan gambar supplementary tetap dapat dibangun dari kode/data plotting native.\par
Tes risiko encoding/CSR, class-label certificate, canonical ties, callback unresolved dan multiscale runtime:23tes lulus. Peringatan pytest hanya menyangkut cache yang tidak dapat ditulis. Semua delapan saved-result audit juga lulus; angka70expert fits pada audit historis merujuk eksperimen3November, bukan fit baru.\par
PDF, cross-reference, caption, font embedding, layout, ZIPmanifest, clean compile dan fresh extraction diperiksa sebelum rilis. Perintah audit:
\begin{quote}\small\nolinkurl{python -B scripts/verify_array_all_saved_20261117.py}\end{quote}
Source dan gambar tetap native, bukan konversi atau crop halaman PDF. Main document adalah\texttt{main.tex} pada root ZIP. Kebutuhan data pihak ketiga/raw extraction tetap mengikuti sumber dan izin pemilik.\par
\section*{Batas yang belum diselesaikan}
Source/class confounding dan keterpaparan data historis tetap ada. Signature split/history control bukan bukti independensi capture. GV tetap predictor berbeda; kontrak runtime/SHAP WV tidak dialihkan. Local FPR0.1435\% tidak menghapus reused-V1.0104\%, external31.90\%/18.85\% atau TON type recall1.55\%. Label TON turunan flow dan coverage terbatas; CIC attack memakai scenario-label agreement.\par
Profiling baru tidak mengukur ARM, energi, setiap native allocation event atau seluruh SHAP execution. LR/RF lebih sederhana tetap pembanding relevan; optimasi WV bukan bukti classifier superior. Penambahan tulisan dan profiling tidak membuktikan generalisasi baru atau menjamin penerimaanQ1.\par
Arsip reviewer tersedia sebagai file lokal yang diverifikasi; upload portal/hosting persisten belum dilakukan. Persetujuan final seluruh penulis belum dikonfirmasi. Catatan related-material AGERS tetap editor-only, termasuk shared external sources dengan revisi conference16November. Tidak ada submission, email, signature atau Turnitin baru.\par
\end{document}
'''
    # Spaces around common quantities improve readability of the Indonesian review log.
    for a,b in [('dospem17','dospem 17'),('main26','main 26'),('supplement93','supplement 93'),('halaman,S','halaman, S'),('17November','17 November'),('3November','3 November'),('16November','16 November'),('480feature','480 feature'),('240replay','240 replay'),('80cold','80 cold'),('37.40','37.40'),('41.75','41.75'),('15,665packet/160decision','15,665 packet/160 decision'),('104badan','104 badan'),('70expert','70 expert'),('23tes','23 tes'),('penerimaanQ1','penerimaan Q1'),('FPR0.1435','FPR 0.1435'),('reused-V1.0104','reused-V 1.0104'),('external31.90','external 31.90')]:intro=intro.replace(a,b)
    for a,b in [('Rata-rata100target','Agregat 100 target'),('top3set','top-three set'),('perkelas20target','per kelas (20 target)'),('Kendalltau-b','Kendall tau-b'),('WV97benar/3salah','WV: 97 benar dan 3 salah'),('Orderedtop3sama99/100','Urutan top-three sama pada 99/100'),('setJaccard1','set Jaccard 1'),('compactWV','compact WV'),('compactRF','compact RF'),('untuk1row','untuk 1 row'),('untuk1000row','untuk 1,000 row'),('Bound1MiB','Bound 1 MiB'),('totalRSS','total RSS'),('Table4','Table 4'),('Table1','Table 1'),('noteTable 1','note Table 1'),('sebagaiS105','sebagai S105'),('presisi penuh tetapCSV','presisi penuh tetap di CSV'),('padaResults/Discussion','pada Results/Discussion'),('ditampilkan97.8800','ditampilkan 97.8800'),('menggunakan37.40','menggunakan 37.40'),('adalah1','adalah 1'),('Nilai37.40','Nilai 37.40'),('dan60.55','dan 60.55'),('Penghematan41.75','Penghematan 41.75'),('sesi3 November','sesi 3 November'),('workload15,665','workload 15,665'),('dan80','dan 80'),('26halaman','26 halaman'),('93halaman','93 halaman')]:intro=intro.replace(a,b)
    (DEST/'revision_log.tex').write_text(intro,'utf8')
if __name__=='__main__':main()
