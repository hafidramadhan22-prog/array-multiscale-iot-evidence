from pathlib import Path
import sys,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import ta_hafid_engine as e
import pandas as pd,numpy as np
r=ROOT/'runs/ARRAY_KNN_FULL_20261119_01';dest=ROOT/'reports/ARRAY_KNN_20261119';dest.mkdir(exist_ok=True,parents=True)
for d in sorted(r.iterdir()):
 if d.is_dir() and (d/'evaluation.npz').exists():
  with np.load(d/'evaluation.npz',allow_pickle=False) as z:
   methods=['KNN_default','KNN_G_selected','WV','RF','LR','Uniform']
   pd.DataFrame({k:z[k] for k in ['row_ids','y']+methods}).to_csv(dest/(d.name+'_predictions.csv.gz'),index=False)
   for m in methods:pd.DataFrame(z['CM_'+m],index=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware'],columns=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']).to_csv(dest/(d.name+'_CM_'+m+'.csv'))
df=pd.read_csv(r/'summary.csv');lines=['# Hasil uji KNN - 19 November 2026','','Eksperimen tambahan; WV lama dan naskah jurnal tidak diubah. Semua hasil berasal dari development yang sudah pernah diamati, bukan independent final test.','','## Macro-F1 (%)','','| Evaluasi | WV | KNN default | KNN pilihan G | RF |','|---|---:|---:|---:|---:|']
for p in ['temporal','packet_private','signature_shared']:
 sub=df[df.protocol==p].set_index('method');lines.append('| '+p+' | '+' | '.join(f'{sub.loc[m,"macro_f1"]*100:.4f}' for m in ['WV','KNN_default','KNN_G_selected','RF'])+' |')
lines+=['','## Normal FPR (%)','','| Evaluasi | WV | KNN default | KNN pilihan G |','|---|---:|---:|---:|']
for p in ['temporal','packet_private','signature_shared']:
 sub=df[df.protocol==p].set_index('method');lines.append('| '+p+' | '+' | '.join(f'{sub.loc[m,"normal_fpr"]*100:.4f}' for m in ['WV','KNN_default','KNN_G_selected'])+' |')
lines+=['','## Metode dan batas interpretasi','','KNN Euclidean, fitur numerik distandardisasi pada A dan kategorikal one-hot. RUS+SMOTE hanya A, cap lama. Grid k=1,3,5,11,31 x uniform/distance; default k5/uniform. G memilih berdasarkan 0.5 macro-F1 + 0.5 DDoS F2. Seluruh 70 kandidat G tersimpan. Pemilihan dibekukan sebelum prediksi V KNN; V tetap telah diamati dalam riset sebelumnya.','','Temporal: mean tiga seed pada target yang sama, bukan tiga sumber independen. Dua fold packet-private dan signature/shared dipool berdasarkan ID OOF yang terpisah. History shared bukan bukti independensi. Tidak ada pengujian eksternal KNN atau benchmark latency/memori pada sesi ini. Durasi fit+prediksi G merupakan log operasional, bukan benchmark inferensi. KNN tidak dimasukkan sebagai anggota WV.','','Pembanding yang lebih rendah pada salah satu protokol tidak membuktikan WV unggul secara umum. Hasil RF dan kedua pilihan KNN tetap dilaporkan. Budget pencarian baru tidak menyamakan total tuning historis semua model.','','## Bukti','','`runs/ARRAY_KNN_FULL_20261119_01/` memuat model, split, training plan, prediksi G/V, confusion matrices, konfigurasi dan katalog SHA256. `scripts/verify_array_knn_20261119.py` memeriksa katalog, disjointness split/holdout, seluruh skor/pilihan G, confusion matrices dan kesamaan prediksi WV/RF/LR/Uniform dengan eksperimen sebelumnya.','','Dokumentasi KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html']
text='\n'.join(lines)+'\n';(dest/'HASIL_KNN_20261119.md').write_text(text,encoding='utf-8');(ROOT/'docs/HASIL_UJI_KNN_ARRAY_20261119.md').write_text(text,encoding='utf-8')
files=list(r.rglob('*'))+list(dest.rglob('*'))+[ROOT/'scripts/run_array_knn_20261119.py',ROOT/'scripts/verify_array_knn_20261119.py',ROOT/'scripts/package_array_knn_20261119.py',ROOT/'docs/ARRAY_KNN_PROTOCOL_20261119.md',ROOT/'docs/HASIL_UJI_KNN_ARRAY_20261119.md']
(dest/'README.txt').write_text('Evidence KNN 19 November 2026. Read HASIL_KNN_20261119.md first. Archived model files were produced by this run; do not load untrusted pickles. Raw dataset and parent context arrays are not bundled: rerunning training requires the original verified project inputs referenced by the scripts. This archive contains predictions, fitted models, configuration and source, not a self-contained raw-data training distribution. No new independent test, no paper edits, no hardware benchmark. WV remains the old fixed reference.\n',encoding='utf-8')
files.append(dest/'README.txt')
archive=ROOT/'penyerahan/UJI_KNN_ARRAY_20261119.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:
  if f.is_file():z.write(f,f.relative_to(ROOT))
 for rec in json.loads((dest/'source_snapshot/manifest.json').read_text()):z.write(ROOT/rec['path'],rec['path'])
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(text);print('ZIP SHA256',e.sha256(archive))
