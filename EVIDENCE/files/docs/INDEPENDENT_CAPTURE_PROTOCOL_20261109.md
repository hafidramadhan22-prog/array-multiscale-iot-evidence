# Uji independen model beku - 9 November 2026

Pengguna meminta eksekusi CICIoT2023, menyetujui TON_IoT Normal + Ransomware,
dan secara terpisah mengizinkan registrasi CIC memakai identitas yang disebut
dalam percakapan. Tidak ada persetujuan untuk mengganti model atau melatih ulang.

Predictor: weighted voting acuan, RF singleton, dan SQL guard + RF rescue.
Bundle acuan milik proyek memiliki SHA-256
`77aa1beb6980f4eb9db9838af6505949899c5aa574f3be49d55adfdc1efcf31d`.
Aturan kandidat, bobot expert, parameter, preprocessing dan feature view
`c60_o10` dibekukan seperti studi FPR sebelumnya. Semua hasil buruk dilaporkan;
hasil eksternal tidak dipakai memilih ulang classifier atau threshold.

## CICIoT2023

Sumber utama: https://www.unb.ca/cic/datasets/iotdataset-2023.html dan portal
unduhan resmi. Registrasi berhasil; mirror hanya diperiksa sebagai cadangan.
File dipilih berdasarkan kesesuaian kelas dan nama, sebelum skor model dibuka:

- `PCAP/Benign_Final/BenignTraffic.pcap`: Normal.
- `PCAP/Benign_Final/BenignTraffic3.pcap`: Normal, kontrol capture kedua.
- `PCAP/SqlInjection/SqlInjection.pcap`: skenario SQL injection.
- `PCAP/Recon-PortScan/Recon-PortScan.pcap`: skenario Port Scanning.
- `PCAP/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap`: skenario DDoS TCP SYN.

Normal, SQL dan PortScan maksimum512MiB per file; SYN maksimum3GiB.
Jika batas ini tidak mencukupi, jangan menyebut unduhan parsial sebagai file
lengkap. Seluruh file yang dinilai harus selesai EOF, hash tercatat dan PCAP
valid secara struktural. Raw tidak dimasukkan ke paket evidence ringkas.

Per file, pilih hingga1000 target yang terdistribusi merata berdasarkan indeks
paket setelah warm-up10detik. History prior-only .1/1/10detik memakai ring16384
record, diproses sejak awal capture dan direset antarfile; tidak berbagi history
lintas dataset. Tidak melakukan resampling atau training.

Satu paket adalah unit fitur model. Label file skenario TIDAK otomatis menjadi
ground-truth setiap paket: dokumentasi, flow labels, alamat endpoint dan waktu
harus diperiksa. Bila hanya label skenario tersedia, laporkan agreement/detection
terhadap label skenario secara eksplisit, bukan recall serangan paket yang
terkonfirmasi. Background yang diketahui tidak boleh diberi label serangan.

## Ekstraksi dan validasi

CSV CIC hasil flow extraction tidak cocok langsung dengan context85. Ekstrak
field protokol dari PCAP menggunakan TShark4.4.9 dan history menggunakan source
proyek yang diverifikasi. Cocokkan numeric/boolean/category formatting terhadap
cached features lama dahulu, tanpa melihat prediksi data eksternal. Field yang
benar-benar tidak ada mendapat representasi absent sumber; fitur yang gagal
diekstrak tidak boleh disamarkan menjadi nol. Tulis perbedaan versi/field yang
tidak dapat diselaraskan. Gate gagal menghentikan inferensi ilmiah.

Rekonstruksi old-target check boleh memperbaiki serializer atau decoder dengan
evidence label-blind; setiap perubahan baru dicatat sebelum scoring eksternal.
Tidak mengubah preprocessing model agar cocok dengan label test baru.

## TON_IoT

Sumber: https://research.unsw.edu.au/projects/toniot-datasets.
Hanya raw network PCAP Normal dan Ransomware beserta
`SecurityEvents_GroundTruth_datasets/SecurityEvents_Network_datasets` yang
relevan. Memerlukan pemetaan timestamp dan IP, bukan menamai semua paket
sebagai Ransomware karena berada pada hari serangan. CSV Zeek/NetFlow tidak
diisi nol agar menyerupai85fitur. Tautan resmi saat ini mengarah ke login;
catat status akses jika PCAP/ground-truth belum tersedia.

TON dan CIC dilaporkan per dataset/capture. Menggabungkan empat kelas dari CIC
dan Ransomware dari TON tidak menghasilkan satu testbed independen lima kelas.
Ransomware yang tidak tersedia diberi status belum diuji, bukan recall0/100%.

## Pelaporan

Simpan konfigurasi, source/hash, paket versi, izin sumber, checksum raw, mapping
label, packet indices, fitur, probabilitas, label, confusion matrix, FPR per
capture, recall per kelas yang benar-benar berlabel, jumlah false negatives,
latency dan coverage. Report five-class probabilities tetap mengizinkan model
memilih Ransomware pada CIC meskipun kelas itu tidak ada sebagai label benar.
Jangan menghitung five-class macro-F1 seolah semua lima kelas terwakili.

Target FPR<8% diperiksa pada Normal independen; mempertahankan recall serangan
dibanding predictor lama diperiksa pada target eksternal yang sama. Perbandingan
hasil dengan cohort historis adalah domain shift, bukan estimasi effect causal.
Jika source, label, atau extractor gate gagal, simpan bukti dan nyatakan tahap
belum dapat dinilai. Tidak ada prediksi atau metrik buatan.
