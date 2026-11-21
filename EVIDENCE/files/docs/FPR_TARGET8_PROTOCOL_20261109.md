# Eksperimen FPR di bawah8%, 9 November2026

Permintaan pengguna: mencari solusi agar false-positive rate tambahan yang sekitar42% turun setidaknya di bawah8%. Persyaratan dospem tetap: periksa akibatnya terhadap deteksi serangan. Penelitian boleh memilih classifier yang lebih sederhana; tidak memaksakan weighted voting menang. Ini eksperimen setelah hasil sebelumnya diamati, bukan praregistrasi sebelum hasil historis diketahui.

## Diagnosis dan kandidat dibekukan

Model/context85 dan fitted experts milik runPAYLOAD_TARGET_REPAIR_20261105_01 tetap. Semua checksum catalog, model, source dan input diverifikasi sebelum deserialisasi. Tidak ada capture/dataset/learner baru dan tidak ada fit baru. Perubahan predictor tetap dibedakan dari optimasi prediction-preserving model lama.

Primary candidate untuk pemeriksaan baru adalah **RF singleton** yang sama, bukan RF-veto/payload guard. RF dipilih setelah mengetahui hasil historis:0/760 Normal tambahan dan0attack-to-Normal padaV, sedangkan weighted reference321/760 dan2attack-to-Normal. Pemilihan ini eksploratif dan tidak diklaim berasal dari G/C yang belum dilihat. Kandidat pembanding: lima singleton, original weighted/soft voting, soft RF/XGB50:50 dan75:25. Dua campuran adalah kontrol fixed-weight, bukan hasil grid search sampai FPR kelihatan baik. Semua hasil ditampilkan.

Tidak menaikkan threshold SQL sampai alarm hilang, tidak memakai payload kosong sebagaiNormal, tidak mengubah label kebenaran atau jumlah penyebut, dan tidak memakai ilustrasiSQL sebagai data training. Tidak ada parameter dipilih ulang setelah blok lanjutan dibuka. Bila RF gagal pada pemeriksaan baru, laporkan gagal tanpa menyembunyikan hasil atau mempersempit sumber.

## Pengukuran lama dan safety gate

Hitung ulang seluruhG/C/V dan760additional dengan prediksi tersimpan. Tambahkan tiga temporal estimator seeds dan dua signature folds; signature/history-private tetapdiagnostic berbeda, bukan capture independen atau satu test yang dapat dijumlahkan. Gate primary deploymentseed42: FPR<8% pada760additional dan NormalV; attack-to-Normal tidak bertambah dari2; tidak ada recall keempat kelas serangan yang lebih rendah daripada reference padaV; macro-F1 boleh turun paling banyak0,1percentage point. Margin0,1pp adalah default operasional baru yang dinyatakan terbuka, bukan klaim non-inferiority statistik. Penurunan metric di cohort/seed lain dilaporkan.

Blok benign lanjutan tidak memiliki serangan: gate sensitivitas harus bernilai tidak-terukur di sana, bukan lulus karenaFN=0 pada himpunan tanpa serangan. Keamanan terhadap capture serangan independen tetap belum dapat disimpulkan.

## Blok paket lanjutan yang belum dipakai dalam audit sebelumnya

Delapan captureNormal yang sudah tersedia dan mempunyai mappingvalid padaaudit lama dipakai lagi; Modbus tetap tidak bisa diskor karena jamCSV rusak dan tidak diam-diam direkonstruksi. Bukan penambahan capture/session. PCAPexisting diproses dari awal dengan ring1MiB, prior-only0,1/1/10s, ordering10s, logicalclockmonotonic dan warmup10s yang sama. Query sebelumpush. RawCSV/PCAPfull hashes harus cocok audit asli.

Kandidat baru adalah indeks paket100999,101999,...,199999, dengan batas akhir min(200000,packet_count) dari metadata audit sebelumnya. HeartRate yang lebih pendek memberi70kandidat; tujuh sumber lain100, total770planned sebelum mapping/warm-up. CSV dibaca sampai300000record untuk menyediakan mapping tanpa mengasumsikan indeksCSV=PCAP. Cocokkan delapan field packetidentity yang sama. Kandidat ambigu/tidakcocok tidak memperoleh label prediksi buatan; jumlah dan bounds dilaporkan.

First-prefix760 lama direkonstruksi sebagai gate alignment: semua85field numerik/categorical dan probabilities harus cocok cache setelah flagformat correction. Ini menguji pipeline feature extraction sebelum hasil lanjutan digunakan. Context tidak direset tepat pada100000: konteks awal blok baru tetap berasal dari paket yang benar-benar mendahuluinya pada source yang sama.

Primary target pada blokbaru: FPR<8% pada scoredtargets dengan coverage>=90%; laporkan per-source, overall, dan coverage bounds. Blok ini baru dinilai oleh run ini tetapi berasal dari capture/corpus yang sudah diketahui; bukan independent-capture final validation.

## Efisiensi, prototype dan artefak

Bandingkan RF dan limaexpertreference pada batchV1000 yang sama dan seluruh blok baru: warmup3,30randomized paired repeats, satu numericalthread. Tiga coldprocess perpolicy mengukur delta peakRSS setelahload,1ms polling. Seluruhinput sudahdisiapkan; tidak dinamai fullPCAPreplay. Model RF diekspor dari fittedpipeline yang sama tanpafit, disertai checksum dan CLI yang hanya menerima schema85field. Dashboard/paper/reference lama tidak ditimpa otomatis.

Simpan config freeze, source snapshots, sourcehashes, candidateinputs/probabilities/per-record decisions, confusion, per-class recall, seed/fold summaries, mapping, rawtimings/RSS, environment, verification dan checksum. Laporan akhir harus menyebut apakah target8% terpenuhi pada setiapcohort serta trade-offnya.

Dasar teknis primer: [scikit-learn1.7 voting](https://scikit-learn.org/1.7/modules/ensemble.html#voting-classifier), [pemilihan threshold dan pemisahan calibrationdata](https://scikit-learn.org/1.7/modules/classification_threshold.html). Dokumentasi tidak menjamin FPR untuk data Hafid; angka hanya berasal dari eksperimen lokal.
