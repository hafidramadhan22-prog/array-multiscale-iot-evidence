# Aturan SQL guard + RF rescue, 9 November 2026

Pengguna meminta solusi FPR di bawah 8%, dengan konteks mempertahankan deteksi
serangan. Ini perubahan predictor eksploratif menggunakan model milik proyek
yang sudah fitted. Tidak ada dataset, capture, algoritma dasar atau fit baru.

Studi RF singleton dan RF-Normal/voting-type disimpan sebagai hasil terdahulu.
Yang kedua gagal menjaga satu prediksi Port Scanning pada signature fold 1.
Keputusan baru ini dibuat SETELAH seluruh hasil cache dan dua blok Normal
lanjutan diketahui. Dua blok tersebut hanya reanalysis retrospektif untuk
aturan ini. Tidak disebut sebagai test independen atau praregistrasi.

Aturan final untuk studi ini:

1. Jika weighted voting memprediksi SQL injection dan RF memprediksi Normal,
   gunakan seluruh posterior RF.
2. Jika weighted voting memprediksi Normal dan RF memprediksi serangan,
   gunakan seluruh posterior RF.
3. Selain itu gunakan seluruh posterior weighted voting.

Aturan bukan analisis payload, bukan jaminan prediction preservation terhadap
voting lama, dan bukan bukti bahwa RF selalu benar. Pemeriksaan cached harus
menjaga recall keempat kelas serangan pada G, C, V seed42, temporal seed43/44,
signature fold1/2; tidak menambah attack-to-Normal; macro-F1 drop <=0.1 pp.
Ambang 0.1 pp adalah batas operasional studi, bukan persetujuan numerik dospem.

Sebelum inferensi target berikutnya: bekukan konfigurasi dan hash source.
Gunakan tujuh capture Normal yang sudah ada dan memiliki >400000 paket.
Target paket 300999,301999,...399999,100 per source; total700 direncanakan.
CSV dibaca maksimal500000 baris; mapping berdasarkan identitas paket, bukan
menyamakan baris CSV dengan indeks paket. History prior-only .1/1/10 detik,
buffer16384 record tetap dijalankan sejak awal capture. Tidak reset di batas
blok. Delapan source/dataset lain tidak ditambahkan; HeartRate tidak cukup
panjang dan dikecualikan berdasarkan metadata sebelum scoring.

Full hash CSV/PCAP dan replay760 target lama harus cocok. Target unscorable
tetap dihitung dalam coverage serta batas worst-case, tidak diganti data.
Blok ketiga menguji hanya Normal dari capture lama; serangan baru/capture
independen tetap belum tersedia. Tidak ada klaim non-inferiority populasi.

Benchmark dilakukan sesudah ekstraksi selesai, satu thread, tiga warmup,
30 pengulangan berpasangan dan tiga cold-process delta peak RSS/batch.
Tolerance probabilitas1e-12 tetap, expert dikonversi float64 sebelum perkalian.
Biaya ini adalah inferensi prepared features, bukan end-to-end PCAP/replay.
