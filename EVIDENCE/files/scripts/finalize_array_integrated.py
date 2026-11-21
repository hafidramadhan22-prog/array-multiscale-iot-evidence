"""Record the completed visual review and provide a clear coverage/readme map."""
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_array_complete import verified, RECORDS, sha
BASE=ROOT/'paper/array_20261022_integrated'


def main():
    assert not (ROOT/'penyerahan/ARRAY_HAFID_ALUR_LENGKAP_20261022.zip').exists()
    q=json.loads((BASE/'quality_check.json').read_text())
    assert q['numeric_checks']==691 and q['status']=='AUTOMATED_CHECKS_PASSED_VISUAL_PENDING'
    before=json.loads((ROOT/'tmp/pdfs/array_integrated_20261022/reviewed_before_final_layout.json').read_text())
    changed={}
    for record in q['pdfs']:
        old=next(x for x in before['pdfs'] if x['name']==record['name'])
        changed[record['name']]=[k for k,v in record['page_image_sha256'].items() if v!=old['page_image_sha256'].get(k)]
        assert sha(BASE/'elsarticle'/(record['name']+'.pdf'))==record['sha256']
        for name,digest in record['page_image_sha256'].items():
            assert sha(ROOT/'tmp/pdfs/array_integrated_20261022'/name)==digest
        record['visual_review_all_pages']=True
        record['reviewed_pages']=list(range(1,record['pages']+1))
    assert changed=={'main':[],'supplement':['supplement_09.png','supplement_17.png','supplement_21.png']}
    # All 45 pages were viewed on contact sheets; these three changed pages
    # were viewed again after the final layout correction.
    q.update(status='PDFS_VERIFIED_ALL_PAGES_REVIEWED',visual_review_all_pdf_pages=True,
             visual_review_method='All page PNGs inspected in contact sheets; final three changed supplement pages reinspected; 42 unchanged page PNGs verified byte-identical',
             visually_reinspected_after_final_edit=changed,visual_defects_remaining=[])
    build=json.loads((BASE/'editorial_build.json').read_text())
    for name in ['E2_cv_summary.csv','E2_effects_vs_DT.csv','E2_policy_choices.csv',
                 'E2_leaf_validation_changes.csv','E3_benchmark.csv']:
        p=verified('reports/LIFT_ID_HASIL_20261018/'+name)
        dest=BASE/'evidence/history/reports'/name;shutil.copy2(p,dest)
    records={r['path']:r for r in build['source_records']}
    records.update({r['path']:r for r in RECORDS.values()})
    build['source_records']=list(records.values())
    for r in build['source_records']:
        assert sha(ROOT/r['path'])==r['sha256'] and sha(ROOT/r['catalog'])==r['catalog_sha256']
    for catalog in ['runs/FULL_20261013_01/artifacts_sha256.json',
                    'reports/REKOMENDASI_20261015_01/artifacts_sha256.json',
                    'reports/LIFT_ID_HASIL_20261018/artifacts_sha256.json']:
        src=ROOT/catalog;dst=BASE/'evidence/source_catalogs'/catalog
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    q['additional_source_records_verified']=len(build['source_records'])
    (BASE/'editorial_build.json').write_text(json.dumps(build,indent=2)+'\n')
    (BASE/'quality_check.json').write_text(json.dumps(q,indent=2)+'\n')
    bat=BASE/'elsarticle/BUILD.bat'
    original=bat.read_text()
    assert not any(x['path']=='elsarticle/BUILD.bat' for x in json.loads((BASE/'template_inventory.json').read_text())['files'])
    bat.write_text(original.replace('has 21 pages','has 23 pages'))
    (BASE/'README_MULAI_DI_SINI.md').write_text('''# Naskah Array: alur awal sampai pengembangan akhir

Revisi 22 Oktober 2026 untuk tinjauan dospem. Penulis M Hafid Ramadhan dan Satria Mandala, afiliasi bersama Humic Engineering (Humic Eng), School of Computing, Telkom University. Satria Mandala adalah penulis korespondensi.

- `elsarticle/main.pdf`: 23 halaman termasuk referensi; 8 tabel, 4 gambar dan 25 referensi.
- `elsarticle/supplement.pdf`: 22 halaman terpisah; 24 tabel dan 5 gambar.
- `ALUR_DAN_CAKUPAN.md`: urutan penelitian dan lokasi hasil awal/tambahan.
- `CATATAN_PERUBAHAN.md`: perubahan dari versi utama 21 halaman.
- `evidence/`: CSV/manifes/array terpilih yang mendasari tulisan; tanpa joblib.
- `editorial/`: Markdown naskah, tabel dan metadata referensi pada ZIP penyerahan.
- `audit/`: pemeriksaan angka, halaman, sumber, template, kompilasi dan checksum.

Ukuran utama elsarticle tetap 12 pt dan margin bawaan. Jumlah 23 halaman adalah hasil penyuntingan mengikuti izin pengguna menambah sedikit dari acuan 20 halaman, bukan klaim batas resmi Array atau angka yang diberikan dospem. Supplementary terpisah dari naskah utama.

## Menyunting dan reproduksi

Sunting `elsarticle/main.tex`, `supplement.tex` dan `references.bib`. Jalankan `BUILD.bat` bila pdflatex/BibTeX tersedia, atau gunakan Tectonic. PDF disertakan dibangun dengan Tectonic 0.17.0; pergantian mesin TeX/isi dapat mengubah halaman.

`python reproduce_added_figures.py --output-dir folder_baru` membuat ulang confusion matrix dan grouped-importance plot dari CSV paket. `python reproduce_initial_figures.py --output-dir folder_baru` membuat ulang bagan alur, grafik 28/29 fitur dan grafik SHAP awal; tiga gambar ini muncul di subfolder figures. Memerlukan NumPy/Matplotlib. Lima PNG telah dibandingkan dengan gambar naskah menggunakan Python terisolasi; tidak ada training, model loading atau game SHAP baru.

Snapshot builder/verifier di audit memerlukan struktur workspace/run asal. Sumber LaTeX dan dua skrip gambar mandiri dapat dipakai dari ZIP ini. MANIFEST_SHA256.json menyimpan checksum semua anggota; katalog sumber asli juga disertakan sebagai jejak audit, bukan janji semua file run aslinya ada di paket penulisan.

Paket ini adalah paket naskah dan bukti terpilih. Semua model, notebook dan PCAP penelitian tetap pada arsip eksperimen penuh yang telah diserahkan sebelumnya. Dashboard yang sudah dibuat tetap pada paket terpisah; revisi penulisan ini tidak mengubahnya. Seluruh PDF/ZIP versi sebelumnya dipertahankan.

Funding, competing interests, CRediT, persetujuan final seluruh penulis dan repository publik masih harus ditetapkan penulis. Naskah belum disubmit. Perbaikan penyajian tidak menambahkan validasi capture independen dan tidak menjamin penerimaan jurnal.
''',encoding='utf-8')
    (BASE/'CATATAN_PERUBAHAN.md').write_text('''# Perubahan revisi alur lengkap

Dari versi utama 21 halaman/supplementary 16 halaman menjadi 23/22 halaman, dengan font 12 pt dan margin template tetap.

1. Bagian 3 kini menjelaskan penelitian awal: split 80:20, 28/29 fitur, nested CV 5 outer/2 inner, dua grid tuning, Original/Balanced, dan Soft/Weighted/Stacking.
2. Main Table 2 menampilkan pemilihan fitur pada development. Table 3 menampilkan seluruh enam ranking holdout historis: F1, F2 DDoS, latency, inference delta RSS dan composite score. Tabel utama berikutnya bernomor 4-8.
3. Figure 1 adalah alur lengkap penelitian. Diagram arsitektur rinci tetap ada sebagai Figure S5.
4. S12 kini menampilkan lima recall dari semua enam ensemble awal; S16 mempertahankan accuracy, macro recall, AUC, performance/efficiency score. Informasi historis yang sebelumnya di dua tabel supplement kini terhubung dengan Table 3 utama tanpa mengulang ranking penuh.
5. S22-S24 melengkapi baseline vs DDoS-tuned dan semua 24 konfigurasi CV (dua fitur x dua search x dua treatment x tiga ensemble), termasuk standar deviasi outer folds.
6. Figure S3 memperlihatkan efek 28/29 dan perluasan tuning; Figure S4 mengembalikan SHAP awal berdasarkan CSV tersimpan, dibedakan dari audit grouped SHAP akhir. CSV global enam konfigurasi, local predictions, manifest dan engine awal ikut di paket.
7. Bukti ringkasan jalur LIFT-ID kini juga menyertakan E2 CV, pilihan kebijakan/perubahan leaf, efek terhadap DT dan benchmark E3. Jalur yang tidak dipakai tetap diringkas di S21.

WB Balanced adalah pemenang skor gabungan 0,974691 pada kebijakan all-six tanggal 15 Oktober; bukan pemenang mutlak setiap metrik. Macro-F1 historisnya 95,5179% sama dengan tiga konfigurasi lain; Balanced stacking memiliki F2 DDoS sedikit lebih tinggi. Perubahan kebijakan sesudah observasi holdout diungkapkan. Seluruh enam konfigurasi tetap eligible.

Pada 29 fitur, expanded tuning hanya mengubah rerata CV Macro-F1 +0,0116 pp dan F2 DDoS -0,0035 pp. Peningkatan port pada split acak tidak dijadikan bukti ketahanan distribusi. Hasil historis 95,5179% tidak dikurangkan dari skor temporal baru 97,8954%, karena target dan protokol berbeda. Kenaikan yang sah untuk perbandingan konteks berpasangan tetap +1,1562 pp.

Seluruh angka dan hasil negatif sebelumnya dipertahankan: kompromi recall ransomware, singleton/uniform voting kompetitif, satu repeat compact lebih lambat, RAM tidak turun seragam, sensitivitas background, shared histories dan reused development. Novelty diletakkan pada desain dan bukti gabungan yang terbatas, tanpa mengklaim algoritma voting/SHAP baru.

Ada 691 pemeriksaan numerik yang lulus, 45 halaman diperiksa, tanpa overfull/missing glyph/undefined references. Tidak ada fit, inferensi model, retuning, model deserialization atau penghitungan game SHAP baru. Ini integrasi penulisan dari hasil yang telah disimpan, bukan peningkatan hasil eksperimen baru.
''',encoding='utf-8')
    (BASE/'ALUR_DAN_CAKUPAN.md').write_text('''# Alur penelitian dan peta hasil

**Perbandingan awal -> WB Balanced sebagai acuan -> diagnosis -> multiscale -> efisiensi implementasi -> audit SHAP.**

| Tahap | Pertanyaan dan hasil | Lokasi tulisan |
| --- | --- | --- |
| Data dan pembagian awal | 65.855 target lima kelas; development 52.684, holdout historis 13.171; preprocessing/resampling hanya training | Main 3.1; supplementary 10; CSV/manifes awal |
| Representasi port | 28 fitur vs 29 fitur pada development dengan nested CV; 29 dipilih pada protokol acak awal | Main Table 2; S22-S24; Figure S3 |
| Tuning DDoS | Dua kandidat awal vs delapan kandidat expanded; pengaruh agregat kecil pada 29 fitur | Main 3.1; S22-S24; seluruh outer-fold CSV |
| Enam ensemble | Original/Balanced x Soft/Weighted/Stacking; WB Balanced teratas menurut skor gabungan | Main Table 3; S12 recall lima kelas; S16 subskor lain |
| Penjelasan awal | KernelSHAP individual, 50 target training/20 background dan 11 ilustrasi holdout; bersifat deskriptif | Supplementary 11, Figure S4; CSV enam konfigurasi |
| Diagnosis dan jalur alternatif | LIFT-ID E1-E3, E4 port, MPED, SBFE, adaptive fusion; tidak semua meningkatkan kualitas | S21; evidence/history; benchmark/ablation LIFT-ID tambahan |
| Model utama yang dikembangkan | Lima expert weighted voting tetap; paket25 + prior-only symmetric multiscale60, bukan jenis classifier baru | Main 4; Figure 1; diagram rinci Figure S5 |
| Uji representasi dan generalisasi terbatas | Temporal: 96,7392% -> 97,8954% pada target sepadan; signature scanning 0% -> 94,8342%; ada biaya recall ransomware | Main Table 4, Figure 2; Table 5, Figure 3; S6-S9/S13-S15 |
| Efisiensi prediktor tetap | Compact preprocessing + certified early exit; mean replay turun 34,46% dibanding native certified; satu repeat lebih lambat | Main Table 6, Figure 4; S10/S17-S18 |
| Faithfulness dan robustness | Exact grouped Shapley; deletion/insertion disjoint donors, kontrol random/RF/LR, changed-input stability, background sensitivity | Main Tables 7-8; S19-S20; Figures S1-S2 |
| Dashboard | Uji coba model reference, inferensi tersimpan/CSV siap fitur dan replay; penjelasan tersimpan hanya target yang sesuai | Main 8; paket dashboard sebelumnya |

## Cara membaca kesimpulan

Angka historis 95,5179% dan skor temporal 97,8954% berasal dari evaluasi berbeda. Perbandingan temporal konteks pendek versus multiscale pada target yang sama mendukung kenaikan +1,1562 poin persentase; ini tetap bukti within-trace, bukan validasi antar-capture independen. Pemilihan WB Balanced dari tahap awal memberi dasar proyek, tetapi hasil kontrol akhir tetap boleh menunjukkan RF/LR atau soft voting kompetitif.

Kontribusi tulisan adalah keterhubungan representasi history yang tersedia sebelum keputusan, eksekusi lebih cepat dengan fungsi prediksi yang dipertahankan, dan audit penjelasan kuantitatif dari full probabilities yang sama. Kombinasi nama IDS, ensemble, lightweight dan SHAP saja bukan bukti novelty algoritmik.

Eksperimen terdahulu yang punya artefak lokal terwakili secara proporsional. Semua grafik lama tidak dicetak ulang; data rinci dan model ada di arsip eksperimen penuh sebelumnya. Hasil versi lama tanpa artefak lokal tidak dikarang. Fokus naskah utama tetap satu pertanyaan penelitian yang tersambung; supplementary menampung rincian dan jalur yang ditinggalkan.
''',encoding='utf-8')
    print(json.dumps(dict(status=q['status'],pages=[x['pages'] for x in q['pdfs']],
                         numeric_checks=q['numeric_checks'],source_records=len(build['source_records'])),indent=2))


if __name__=='__main__':main()
