# Membuat sendiri lima gambar artikel Array

Paket ini berisi kode Python dan enam CSV asli yang dipakai menggambar lima figure pada versi clean 21 November. Bukan gambar dari image generator dan bukan ekstraksi piksel dari PDF artikel. Kode disiapkan dengan bantuan AI; menjalankan kode sendiri tidak menjadikan bantuan tersebut tidak pernah ada. Baca dan verifikasi gambar sebelum dipakai.

## Cara menjalankan di Windows

1. Ekstrak seluruh ZIP ke folder biasa. Jangan menjalankan langsung dari tampilan ZIP.
2. Buka folder hasil ekstraksi, klik address bar File Explorer, ketik `powershell`, tekan Enter.
3. Buat environment khusus dan pasang tiga library. Tidak perlu aktivasi PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe buat_gambar_array.py
```

Jika `py` tidak ditemukan tetapi Python sudah dipasang, ganti perintah pertama menjadi `python -m venv .venv`. Uji paket dilakukan dengan Python 3.11.15 dan versi library pada requirements.txt.

4. Buka `hasil_gambar`. Terdapat lima PDF vektor, lima SVG editable, lima PNG 300 dpi dan log hash/versi.
5. Gunakan PDF untuk LaTeX, PNG untuk Word, atau buka SVG di editor vektor bila ingin mengatur posisi label secara manual. Angka dan makna label harus tetap sesuai hasil.

Hanya satu gambar atau simpan versi baru:

```powershell
.\.venv\Scripts\python.exe buat_gambar_array.py --figure 2 --output hasil_edit_saya
```

Menjalankan ulang pada folder output yang sama mengganti nama keluaran gambar tersebut. CSV tidak diubah. Skrip memeriksa hash keenam CSV sebelum menggambar. Tidak ada pembukaan estimator, training, inferensi, atau pengiriman data jaringan.

## Bagian kode yang dapat dibaca dan diedit

- `gambar_1`: diagram alur; teks, posisi kotak dan panah tertulis eksplisit.
- `gambar_2`: ablasi context/ordering. O1/O10 memakai huruf O (ordering), bukan angka nol. Error bar temporal adalah SD tiga fitting seed. Sumbu Y dimulai pada 95%/65% dan ditampilkan eksplisit; bar ini tidak boleh dipakai untuk menafsirkan rasio besar peningkatan.
- `gambar_3`: 30 pasangan replay sesi 3 November, bukan sesi delapan kebijakan 17 November. Grafik ini mendukung hasil historis 41.75%, bukan 60.55%/37.40% dari sesi lain.
- `gambar_4`: share atribusi dari finite donor game A32; bukan bukti kausal atau akurasi deteksi.
- `gambar_5`: WV/GV per cohort dan per file Normal eksternal. Nilai per file TON berbeda dari pooled FPR tabel utama; label bertanda * adalah flow-mapped targets. Garis 8% adalah target penelitian, bukan standar universal.

Parameter `figsize`, `fontsize`, `color`, `set_ylim`, `box` dan `arrow` mengendalikan tampilan. Mengganti font, warna atau posisi label diperbolehkan sebagai tata letak; jangan mengubah nilai hasil demi tampilan lebih baik. `save()` menulis tiga format dengan nama yang sama seperti berkas figure pada LaTeX artikel.

## Penggunaan dalam naskah

Setelah menjalankan dan memeriksa hasil, salin lima `revision_*.pdf` dari `hasil_gambar` ke folder `figures` pada salinan proyek LaTeX Anda, lalu kompilasi `main.tex`. Simpan cadangan versi artikel sebelumnya. Untuk Word, gunakan PNG dan pertahankan caption serta rasio aspek.

Lima plot tetap berasal dari hasil eksperimen tersimpan. Eksekusi skrip adalah reproduksi visualisasi, bukan eksperimen model baru. Log mencatat waktu, versi library, hash skrip, hash data dan hash keluaran agar asal gambar dapat ditelusuri.
