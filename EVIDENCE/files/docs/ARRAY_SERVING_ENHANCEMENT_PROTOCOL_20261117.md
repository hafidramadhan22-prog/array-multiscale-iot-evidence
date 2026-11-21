# Protokol penguatan serving Array - 17 November 2026

Pengguna mengizinkan eksekusi penguatan artikel setelah crosscheck. Ini studi
eksekusi baru pada model dan workload lama, bukan test generalisasi baru.
Tidak ada training, tuning, perubahan bobot, threshold, guard, fitur atau data.
Hasil lengkap dan hasil negatif dipertahankan. Default di bawah adalah desain
peneliti untuk studi ini, bukan angka yang diatribusikan kepada dospem.

## Fungsi dan input tetap

- Sumber milik proyek: `runs/EXACT_RUNTIME_20261020_01/`.
- Gate induk: status PASS_EXACT_RUNTIME; inputs_manifest SHA256
  `b935207d33fb17b3fb74dbb977683f9c1e3444997c07b7fea955ff8aa7ce03f2`.
- Periksa source snapshot, model, prepared input, expected IDs/labels dan PCAP
  replay terhadap manifest sebelum deserialisasi model yang diketahui asalnya.
- Reference baseline25 adalah 25 packet fields +60 multiscale fields (85 total),
  tidak sama dengan predictor conference packet-fields-only.
- Target tetap: batch1 dan1000 dari temporal V, replay15,665packet/160decision
  yang sudah diamati. Historical holdout tidak ditambahkan atau dipakai.
- Canonical CSV workspace dicek SHA25653101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a.

## Delapan kebijakan

WV native full, WV native certified, WV compact full, WV compact certified;
RF native/compact; LR native/compact. Native WV memakai preprocessing bersama
dalam family yang sudah ada. Compact memakai FrozenEncoding yang sama untuk
WV dan singleton. RF membutuhkan tree encoding, LR tree encoding+linear scale.
Fitted estimators tidak diubah. Kesetaraan probabilitas dan label diuji melawan
fungsi native milik masing-masing model; singleton tidak dipaksa mengikuti WV.

## Pengukuran

Satu sesi laptop, satu thread numerical libraries, lima warmup perpolicy/batch
dan perpolicy/replay,30repetisi. Dalam setiap repetisi, delapan policy diacak
berdasarkan seed20261117. Feature timing mencakup encoding+prediction; replay
menambahkan fixed-PCAP decoding, history update/query dan framing. Legacy
packet fields tersedia, bukan rekonstruksi semua field dari rawwire atau SHAP.
Full mode menghasilkan posterior lengkap, certified hanya label. Tidak ada
latency/efisiensi/XAI GV yang disimpulkan dari eksperimen ini.

Lima cold process perpolicy/workload untuk feature1000 dan replay. Memori adalah
delta peak RSS setelah load, sampled2ms; bukan ukuran array/model, bukan seluruh
native allocation events dan bukan hasil ARM. Simpan baseline/peak/delta/samples.

Laporkan median/IQR feature costs; mean/median/SD replay; paired differences
untuk compact-native/full-certified dan paired2x2interaction. Seeded10000paired
bootstrap hanya mendeskripsikan variasi pengulangan pada workload/mesin tetap;
bukan confidence interval generalisasi capture. Tidak membuat skor composite
atau memilih predictor berdasarkan sesi ini. Semua delapan hasil dilaporkan.

## Gate dan arsip

Smoke terpisah2repetisi/1warmup, tanpa cold RSS; bukan hasil artikel. Full output
baru `runs/ARRAY_SERVING_ABLATION_20261117_01`. Simpan source snapshot/config,
environment/software inventory, seed/order, input/source hashes, label/ posterior
equivalence, rawtiming/CPU/RSS, expected labels, catalog/complete dan executionlog.
Perubahan input/source/hash atau probabilitas/label menyebabkan run gagal.
Tidak melonggarkan gate. Sealed parent dan rilis17November sebelumnya tetap utuh.

## Analisis ulang data tersimpan

Tambahkan paired context gains/seed, XAI perclass/correctness dan ordered ranks
dari saved arrays tanpa explanation inference baru. Hasil tersebut eksploratif
post-hoc dan tidak mengubah estimator/sample IDs. Pertahankan external failures,
history/sourceconfounding dan keterbatasan label. Kontribusi diperjelas dengan
critical related work, matriks RQ/bukti/batas dan indeks reproduksi.
