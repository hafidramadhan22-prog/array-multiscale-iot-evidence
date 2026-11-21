# TON_IoT: pemetaan label flow sebelum scoring TON

Tambahan ini ditulis setelah hasil awal CIC tersedia, tetapi sebelum model
menilai TON_IoT. Model/aturan/fitur tetap; tidak ada pemilihan predictor baru.

Tautan penulis ke SharePoint belum dapat diakses tanpa login. Mirror publik
`mtasfi/ton-iot-with-pcap-payload`, commit
`69799b22499220ae2546655b7d8359f8d7169b63`, menyediakan raw PCAP dan 23 CSV
`Network_dataset_*.csv` berlabel. Raw yang dipilih berdasarkan nama/ketersediaan
adalah `normal_runsomware1.pcap` dan `normal_runsomware2.pcap`; seluruh checksum
LFS harus cocok. Dataset ini tetap TON_IoT, bukan sumber keenam atau data buatan.

Karena tabel SecurityEvents asli belum dapat diunduh, gunakan label flow pada
CSV processed sebagai **ground-truth turunan dari anotasi flow penulis**, bukan
klaim label per-paket asli. Seluruh23CSV harus lengkap dan cocok hash sebelum
menganggap tidak ada flow yang berkonflik.

Paket TCP/UDP dicocokkan melalui pasangan IP/port bidirectional dan protokol.
Timestamp CSV pada contoh terdistribusi beresolusi satu detik. Interval yang
dipakai adalah [ts, ts+duration+1detik), dengan duration kosong/negatif ditolak.
Satu detik mengakomodasi kuantisasi timestamp sumber, bukan tolerance yang
diubah mengikuti hasil model. Mapping dengan lebih dari satu jenis label
ditandai ambigu dan tidak diberi label buatan. Paket tanpa pasangan flow atau
dengan kelas lain dicatat terpisah, bukan otomatis Normal.

Ambil maksimal1000 paket Normal dan1000 paket Ransomware per file dari seluruh
paket yang terpetakan unik dan sudah melewati warm-up10detik, terdistribusi
merata menurut indeks paket. Jika kelas kurang dari1000, gunakan seluruh yang
tersedia dan laporkan jumlahnya; tidak mengisi dengan salinan atau sintetis.
History tetap seluruh paket terdahulu, tanpa menggunakan label flow sebagai
fitur. Statistik history tidak di-reset saat label berubah.

PCAPNG dikonversi lossless ke nanosecond Ethernet PCAP; jumlah paket dan waktu
awal/akhir harus sama persis dalam angka Decimal, bukan perbandingan padding
string. Raw asli dipertahankan. Target ID, flow record asal, coverage, ambiguous
dan unmatched counts disimpan sebelum scoring.

Laporan tetap per file/dataset. Dua file tidak otomatis berarti dua sesi yang
independen. Menggabungkan kelas lintas CIC dan TON tidak menghasilkan satu
testbed baru dengan lima kelas yang tervalidasi seragam. Hasil berbasis label
flow dibedakan dari skenario file CIC dan dari label paket yang dikonfirmasi.
