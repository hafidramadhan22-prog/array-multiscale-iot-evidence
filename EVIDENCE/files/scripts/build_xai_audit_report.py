"""Summarize fixed-model XAI audit artifacts; no training or deserialization."""
from pathlib import Path
import argparse,base64,html,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ta_hafid_engine as e


def table(frame):return frame.to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)
def md_table(frame):
    names=list(frame.columns);rows=['| '+' | '.join(names)+' |','| '+' | '.join(['---']*len(names))+' |']
    for _,r in frame.iterrows():rows.append('| '+' | '.join(f'{r[c]:.6f}' if isinstance(r[c],float) else str(r[c]) for c in names)+' |')
    return '\n'.join(rows)


def build(run):
    run=run.resolve();complete=json.loads((run/'execution_complete.json').read_text())
    if complete['status']!='XAI_AUDIT_EXECUTION_COMPLETE':raise ValueError('Full audit required')
    out=run/'report';out.mkdir(exist_ok=True);tabs=run/'tables'
    faithful=pd.read_csv(tabs/'faithfulness.csv');rob=pd.read_csv(tabs/'robustness.csv');bg=pd.read_csv(tabs/'background_sensitivity.csv')
    atomic=pd.read_csv(tabs/'atomic_faithfulness.csv');noise=pd.read_csv(tabs/'atomic_seed_sensitivity.csv')
    cols=['deletion_auc','random_deletion_auc','deletion_gain','insertion_auc','random_insertion_auc','insertion_gain']
    primary=faithful[(faithful.donor=='A')&(faithful.ranking_donor=='A')&(faithful.ranking=='signed')]
    summary=primary.groupby(['bundle','method'],sort=False)[cols].mean().reset_index()
    byclass=primary.groupby(['bundle','method','true_class'],sort=False)[cols].mean().reset_index()
    wins=primary.assign(deletion_win=primary.deletion_gain>1e-12,deletion_tie=primary.deletion_gain.abs()<=1e-12,insertion_win=primary.insertion_gain>1e-12).groupby(['bundle','method'])[['deletion_win','deletion_tie','insertion_win']].mean().reset_index()
    sensitivity=faithful[(faithful.ranking=='signed')&(faithful.bundle.str.contains('balanced__seed42'))].groupby(['bundle','method','donor','ranking_donor'],sort=False)[cols].mean().reset_index()
    rankings=faithful[(faithful.donor=='A')&(faithful.ranking_donor=='A')].groupby(['bundle','method','ranking'],sort=False)[cols].mean().reset_index()
    rcols=['prediction_unchanged','input_unchanged','probability_l1','normalized_l1','top3_jaccard','cosine','attribution_degenerate']
    robust=rob.groupby(['bundle','method','variant'],sort=False)[rcols].mean().reset_index()
    unchanged=rob[rob.prediction_unchanged].groupby(['bundle','method','variant'],sort=False)[rcols].mean().reset_index()
    changed=rob[rob.prediction_unchanged&~rob.input_unchanged].groupby(['bundle','method','variant'],sort=False)[rcols].mean().reset_index()
    counts=rob.groupby(['bundle','method','variant']).agg(rows=('row_id','size'),same_prediction=('prediction_unchanged','sum'),same_input=('input_unchanged','sum')).reset_index()
    robust_per_class=rob.groupby(['bundle','method','variant','true_class'],sort=False)[rcols].mean().reset_index()
    ranking_blind=rob.assign(changed_prediction_high_top3=(~rob.prediction_unchanged)&(rob.top3_jaccard>=.9)).groupby(['bundle','method','variant'],sort=False).agg(rows=('row_id','size'),changed_prediction_high_top3=('changed_prediction_high_top3','sum')).reset_index()
    bg_summary=bg.groupby(['bundle','method','donor'],sort=False)[['normalized_l1','top3_jaccard','cosine']].mean().reset_index()
    atom_summary=atomic.groupby(['bundle','method','seed'],sort=False)[cols+['seconds']].mean().reset_index()
    noise_summary=noise.groupby(['bundle','method'],sort=False)[['normalized_l1','top3_jaccard','cosine']].mean().reset_index()
    importance=[];resolution=[]
    for folder in sorted((run/'groups').iterdir()):
        group_names=list(json.loads((folder/'donor_A/metadata.json').read_text())['groups'])
        with np.load(folder/'donor_A/coalitions.npz',allow_pickle=False) as z:
            targets=z['values'][:,-1,0].argmax(1)
            attributions=np.stack([z['phi'][i,:,0,int(t)] for i,t in enumerate(targets)])
            for i,rid in enumerate(z['row_ids']):
                magnitudes=np.sort(np.abs(attributions[i]))[::-1]
                resolution.append(dict(bundle=folder.name,row_id=int(rid),true_class=int(z['y'][i]),active_groups=int(np.sum(magnitudes>1e-10)),top3_boundary_tie=bool(abs(magnitudes[2]-magnitudes[3])<1e-12)))
            for k in range(5):
                take=z['y']==k;absolute=np.abs(attributions[take]);mass=absolute/(absolute.sum(1,keepdims=True)+1e-15)
                for j,name in enumerate(group_names):importance.append(dict(bundle=folder.name,true_class=k,class_name=e.CLASSES[k],group=name,mean_signed=float(attributions[take,j].mean()),mean_absolute=float(absolute[:,j].mean()),mean_absolute_mass=float(mass[:,j].mean())))
    importance=pd.DataFrame(importance)
    importance.to_csv(tabs/'group_attribution_summary.csv',index=False,float_format='%.17g')
    resolution=pd.DataFrame(resolution);resolution.to_csv(tabs/'group_resolution_diagnostics.csv',index=False)
    robust_per_class.to_csv(tabs/'robustness_per_class.csv',index=False,float_format='%.17g')
    ranking_blind.to_csv(tabs/'ranking_prediction_mismatch.csv',index=False)
    for name,frame in [('group_summary',summary),('group_per_class',byclass),('group_win_rates',wins),('masker_sensitivity_summary',sensitivity),('ranking_summary',rankings),('robustness_summary',robust),('robustness_unchanged_prediction',unchanged),('robustness_changed_input_same_prediction',changed),('robustness_counts',counts),('background_summary',bg_summary),('atomic_summary',atom_summary),('atomic_seed_summary',noise_summary)]:
        frame.to_csv(tabs/(name+'.csv'),index=False,float_format='%.17g')
    key_views=['legacy29__balanced__seed42','legacy29_symmetric__balanced__seed42','legacy25_symmetric__balanced__seed42']
    main=summary[(summary.method=='weighted_voting')&summary.bundle.isin(key_views)].copy()
    focus=robust[robust.method=='weighted_voting'];focus_bg=bg_summary[bg_summary.method=='weighted_voting']
    zero=sensitivity[(sensitivity.method=='weighted_voting')&(sensitivity.donor=='zero')]
    statements=[
      'Audit selesai tanpa fitting classifier baru. Model/bobot tidak diubah; peningkatan hasil klasifikasi bukan keluaran eksperimen ini.',
      'SHAP diuji sebagai ranking bukti model melalui deletion/insertion, bukan hanya plot. Nilai gain positif berarti lebih baik dari urutan acak pada game masking yang disebutkan.',
      'Exact group Shapley dan individual permutation SHAP merupakan objek berbeda. AUC kedua resolusi tidak dibandingkan langsung sebagai bukti satu metode lebih baik.',
      'Additivity dan kecocokan library adalah verifikasi perhitungan; tidak membuktikan sebab-akibat jaringan atau kejujuran penjelasan secara absolut.',
      'Evaluasi kuantitatif SHAP/robustness pada IDS dan IoT sudah mempunyai preseden. Klaim95% paper hanya plot dan novelty sangat tinggi tidak didukung systematic review.',
      'Hasil menguatkan studi empiris bila temuan bertahan pada donor alternatif dan gangguan yang jelas; tidak otomatis menghasilkan model/algoritma utama baru atau menjamin jurnalQ1.'
    ]
    for _,row in main.iterrows():
        statements.append(f"{row['bundle']}: deletionAUC {row['deletion_auc']:.6f} vs random {row['random_deletion_auc']:.6f}; insertionAUC {row['insertion_auc']:.6f} vs random {row['random_insertion_auc']:.6f}. AngkaAUC bukan persentase kejujuran penjelasan.")
    cross=sensitivity[(sensitivity.method=='weighted_voting')&(sensitivity.donor=='B')&(sensitivity.ranking_donor=='A')]
    for _,row in cross.iterrows():statements.append(f"RankingA dievaluasi dengan donorB pada {row['bundle']}: gain deletion {row['deletion_gain']:.6f}, insertion {row['insertion_gain']:.6f}; hasil positif bertahan pada donor evaluasi yang berbeda dalam audit ini.")
    blind=ranking_blind[(ranking_blind.method=='weighted_voting')&(ranking_blind.bundle=='legacy29_symmetric__balanced__seed42')&(ranking_blind.variant=='port_bijection')].iloc[0]
    statements.append(f"Pada WB29+symmetric, {int(blind['changed_prediction_high_top3'])}/100 sampel berubah label setelah bijeksiport walaupun top3Jaccard>=0.9. Stabilitas nama kelompok fitur teratas tidak menjamin stabilitas prediksi atau besaran atribusi.")
    statements.append('Querydelay1/5ms mempertahankan seluruh100 label pada keduaWB rich. Namun input tetap identik pada81/67 sampel; robustness pada perubahan nyata hanya didukung19/33 sampel. Portbijection versi25 tidak mengubah inputmodel sehingga invariance tersebut merupakan sifat representasi, bukan uji pada fitur yang berubah.')
    statements.append('Background alternatif pada WB25+symmetric memberi top3Jaccard0.545 tetapi cosine0.998589. Perubahan ranking dapat didorong kelompok dengan atribusi kecil; rendahnyaJaccard saja tidak membuktikan kontribusi utama berubah drastis.')
    limits=[
      '100V berkuota20 perkelas; hasil bukan rata-rata dengan prevalensi alami. Individual SHAP hanya25 baris (5 perkelas),4 siklus/2seed; aproksimasi dapat bervariasi.',
      'Background utama dan alternatif masing-masing8 baris asli A. Background kecil dapat memberi estimasi marginal yang sensitif; dua background tidak menjamin representasi populasi.',
      'Development dan hyperparameter telah digunakan sebelumnya. DDoS satu capture, Normal satu sumber; tiga seed model bukan tiga capture independen.',
      'Penggantian donor per kelompok menjaga relasi dalam kelompok, tetapi tidak menjamin seluruh hybrid valid sebagai paket/alur jaringan; ini marginal intervention, bukan conditional/causal SHAP.',
      'Kelompok riwayat memuat40 kolom dan kelompok lain lebih kecil. Dampak penghapusan grup tidak mengukur nilai perkolom atau biaya komputasi yang setara.',
      'Zero baseline adalah diagnosis OOD. AUC rendah pada zero tidak otomatis lebih faithful. Pemetaan port bukan serangan jaringan; query delay bukan adversarial search.',
      'Robustness dilaporkan untuk seluruh input dan subset prediksi tetap; input identik/tie/null atribusi dibedakan. Tidak ada jaminan stabil terhadap seluruh perturbasi.',
      'Tidak melakukan LIME/IntegratedGradients, ROAR/retrain atau reproduksi E-XAI/FusionSHAP lengkap. Kontrol yang dijalankan adalah random/least/absolute, LR/RF, background/masker, dua resolusi SHAP dan fit seed.',
      'Scheduler sebelumnya menjaga label, bukan posterior lengkap. SHAP di sini dihitung dari ensemble penuh; biaya penjelasan terpisah dari latency deteksi.',
      'Durasi atribusi bersifat diagnostik: cache koalisi dipakai ulang antarseed explainer dan tes perangkat lunak sempat berjalan bersamaan. Tidak dipakai sebagai benchmark efisiensi classifier atau untuk menilai kecepatan dua seed SHAP.',
      'Tidak ada ambang universal untuk menyebut AUC tertentu cukup publish. Tidak melakukan uji signifikansi yang menganggap baris dalam capture sebagai unit capture independen.'
    ]
    # Curves for the primary empirical game, identical units within each panel.
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for row,view in enumerate(['legacy29_symmetric','legacy25_symmetric']):
        curve=json.loads((run/'groups'/f'{view}__balanced__seed42'/'donor_A/curves.json').read_text())
        for col,(metric,random_metric) in enumerate([('deletion','random_deletion'),('insertion','random_insertion')]):
            ax=axes[row,col]
            for rank,name in [('signed','SHAP signed'),('absolute','SHAP absolute'),('least','Least important')]:
                values=np.array([r[metric] for r in curve if r['method']=='weighted_voting' and r['ranking']==rank]);x=np.linspace(0,1,values.shape[1]);ax.plot(x,values.mean(0),marker='o',label=name)
            random=np.array([r[random_metric] for r in curve if r['method']=='weighted_voting' and r['ranking']=='signed'])
            ax.plot(x,random.mean(0),'--',color='black',label='Exact random expectation')
            ax.set(title=f'{view}: {metric}',xlabel='Fraction of groups',ylabel='Original predicted-class probability',ylim=(-.02,1.02));ax.grid(alpha=.2)
    handles,labels_=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels_,loc='outside lower center',ncol=4)
    for suffix in ['png','svg']:fig.savefig(out/f'faithfulness_curves.{suffix}',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for index,view in enumerate(['legacy29_symmetric','legacy25_symmetric']):
        subset=focus[focus.bundle==f'{view}__balanced__seed42'];x=np.arange(len(subset));ax=axes[index]
        ax.bar(x-.18,subset.prediction_unchanged,width=.36,label='Prediction unchanged');ax.bar(x+.18,1-subset.normalized_l1,width=.36,label='1 - attribution L1 drift')
        labels_=['port bijection'+('\ninput identik' if view=='legacy25_symmetric' else ''),'delay 1ms\n19 input berubah','delay 5ms\n33 input berubah']
        ax.set(xticks=x,xticklabels=labels_,title=view,ylim=(0,1.05));ax.tick_params(axis='x',rotation=0);ax.grid(axis='y',alpha=.2)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2)
    for suffix in ['png','svg']:fig.savefig(out/f'robustness.{suffix}',dpi=150)
    plt.close(fig)
    lines=['# Hasil audit SHAP Weighted Voting','',*['- '+s for s in statements],'','## Hasil utama kelompok, WB Balanced seed42','',md_table(main),'',
      '## Sensitivitas prediksi dan atribusi,100V','',md_table(focus),'',
      '## Sensitivitas terhadap background','',md_table(focus_bg),'','## Zero baseline sebagai diagnosis','',md_table(zero),'',
      '## Batas interpretasi','',*['- '+s for s in limits],'',
      'Justifikasi: docs/XAI_LITERATURE_20261020.md. Protokol: docs/XAI_AUDIT_PROTOCOL_20261020.md. Semua metrik perbaris, koalisi, atribusi, kurva, indeks, source dan environment tersedia pada run ini.']
    (out/'HASIL_DAN_KESIMPULAN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    e.write_json(out/'summary.json',dict(status='XAI_REPORT_READY',complete=complete,main_weighted_voting=main.to_dict('records'),
        robustness_weighted_voting=focus.to_dict('records'),background_weighted_voting=focus_bg.to_dict('records'),conclusions=statements,limitations=limits,
        algorithmic_novelty_established=False,publication_guaranteed=False,independent_capture_validation=False))
    def img(name):return '<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">'
    sections=[('Temuan utama','<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in statements)+'</ul>'),
      ('Faithfulness WB Balanced seed42',table(main)+img('faithfulness_curves.png')),
      ('Semua model, treatment dan fit seed',table(summary)),('Perkelas',table(byclass)),
      ('Atribusi kelompok WB perkelas',table(importance)),
      ('Baseline signed/absolute/least',table(rankings)),('Fraksi gain terhadap random',table(wins)),
      ('Background/masker dan ranking dari donor berbeda',table(sensitivity)+table(bg_summary)),
      ('Robustness seluruh sampel',table(robust)+table(counts)+img('robustness.png')),
      ('Robustness perkelas dan label berubah walau ranking tetap',table(robust_per_class)+table(ranking_blind)),
      ('Resolusi ranking dan ties',table(resolution.groupby('bundle')[['active_groups','top3_boundary_tie']].mean().reset_index())),
      ('Robustness saat label prediksi tetap',table(unchanged)),('Robustness saat input berubah dan label prediksi tetap',table(changed)),
      ('Individual permutation SHAP dan noise estimator',table(atom_summary)+table(noise_summary)),
      ('Batas klaim','<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in limits)+'</ul>'),
      ('Justifikasi literatur','<pre>'+html.escape((ROOT/'docs/XAI_LITERATURE_20261020.md').read_text(encoding='utf8'))+'</pre>')]
    body=''.join('<section><h2>'+title+'</h2>'+content+'</section>' for title,content in sections)
    (out/'LAPORAN_XAI_SHAP.html').write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>Audit SHAP WB</title><style>body{font:16px/1.55 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#182635}h1,h2{color:#133e58}section{overflow:auto;margin:36px 0}table{border-collapse:collapse;font-size:13px}td,th{padding:6px 9px;border:1px solid #ced6dd}th{background:#e6f1f5}pre{white-space:pre-wrap}.banner{padding:16px;background:#fff3d6}</style><h1>Evaluasi kuantitatif SHAP Weighted Voting</h1><p>20 Oktober2026. Model tetap; evaluasi reused-development.</p><div class="banner">Tidak membenarkan klaim95% paper hanya plot. Hasil audit penjelasan bukan jaminan novelty algoritmik atau penerimaanQ1.</div>'+body+'</html>',encoding='utf8')
    print(json.dumps(dict(main=main.to_dict('records'),robustness=focus.to_dict('records')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run)
