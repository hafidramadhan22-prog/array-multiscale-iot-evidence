"""Finite-matrix reporting and paired trace sensitivity, never fitting a model."""
from pathlib import Path
import argparse,base64,html,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ta_hafid_engine as e
from e4 import metrics
from wb_research import mixture
from wb_context_ablation import confusion_by_blocks,paired_block_bootstrap
from run_context_repair import read_npz,source_file

LABELS={'baseline25':'25 + konteks40 lama','baseline29':'29 + konteks40 lama',
        'context40':'Konteks40 saja','context64':'Konteks40 + rasio24','legacy25_context64':'25 + konteks40 + rasio24'}

def evaluation(run,cfg,stage,view,case):
    if view.startswith('baseline'):
        old='legacy25_symmetric' if view=='baseline25' else 'legacy29_symmetric'
        ev=read_npz(source_file(cfg,f'stages/{stage}/{old}__{case}/evaluation.npz'))
        sel=json.loads(source_file(cfg,f'stages/{stage}/{old}__{case}/selection_before_V.json').read_text())
    else:
        folder=run/'stages'/stage/f'{view}__{case}'
        ev=read_npz(folder/'evaluation.npz');sel=json.loads((folder/'selection_before_V.json').read_text())
    return ev,sel

def labels(ev,sel,method,reference=None):
    if method in e.NAMES:return ev['p_experts'][:,e.NAMES.index(method)].argmax(1)
    if method=='soft_voting':return ev['p_experts'].mean(1).argmax(1)
    if method=='weighted_reference':return mixture(ev['p_experts'],reference).argmax(1)
    if method=='weighted_voting':return mixture(ev['p_experts'],sel['weights']).argmax(1)
    raise ValueError('Unknown method')

def analyze(run,enrich_only=False):
    cfg=json.loads((run/'configuration_freeze.json').read_text());assert not cfg['smoke_test']
    verification=json.loads((run/'verification_quality.json').read_text());assert verification['status']=='PASS_CONTEXT_REPAIR_QUALITY'
    table=pd.read_csv(run/'tables/metrics.csv');out=run/'tables';report=run/'report';report.mkdir(exist_ok=True)
    keys=['view','case','condition','method'];scores=['macro_f1','ddos_f2']+[f'recall_{c}' for c in e.CLASSES]
    temporal=table[table.protocol=='temporal'].groupby(keys)[scores].agg(['mean','std']).reset_index()
    temporal.columns=['_'.join(z).rstrip('_') if isinstance(z,tuple) else z for z in temporal.columns]
    temporal.to_csv(out/'temporal_summary.csv',index=False)
    pooled=[];confusions=[];selection_rows=[];predictions={}
    stages=json.loads((run/'planned_stages.json').read_text());all_views=['baseline25','baseline29']+cfg['views']
    for case in cfg['cases']:
        for view in all_views:
            for method in e.NAMES+['soft_voting','weighted_voting','weighted_reference']:
                targets=[];guesses=[]
                for stage in ['signature_fold1','signature_fold2']:
                    ev,sel=evaluation(run,cfg,stage,view,case)
                    _,base=evaluation(run,cfg,stage,'baseline25',case)
                    targets.append(ev['y']);guesses.append(labels(ev,sel,method,base['weights']))
                yy=np.concatenate(targets);pred=np.concatenate(guesses)
                pooled.append(dict(view=view,case=case,method=method,**metrics(yy,pred)))
                cm=np.bincount(5*yy+pred,minlength=25).reshape(5,5)
                for a in range(5):
                    for b in range(5):confusions.append(dict(view=view,case=case,method=method,true=e.CLASSES[a],predicted=e.CLASSES[b],count=int(cm[a,b])))
    pooled=pd.DataFrame(pooled);pooled.to_csv(out/'signature_pooled.csv',index=False)
    pd.DataFrame(confusions).to_csv(out/'signature_confusions.csv',index=False)
    for stage in stages:
        chosen=json.loads((run/'stages'/stage/'candidate_selection_before_V.json').read_text())['selected']
        for case in cfg['cases']:
            for policy in ['wb_only','all_models']:
                item=chosen[case][policy];ev,sel=evaluation(run,cfg,stage,item['view'],case)
                pred=labels(ev,sel,item['method'])
                selection_rows.append(dict(stage=stage,case=case,policy=policy,view=item['view'],method=item['method'],
                    G_selection_score=item['score'],**metrics(ev['y'],pred)))
    pd.DataFrame(selection_rows).to_csv(out/'G_selected_V_results.csv',index=False)
    if not enrich_only:
        # Shared time-block construction and paired resampling; three seeds are not three datasets.
        sensitivity=[];draws={};support=[]
        for case in cfg['cases']:
            cached={}
            for view in all_views:
                cached[view]=[evaluation(run,cfg,f'temporal_seed{s}',view,case) for s in cfg['seeds']]
            first=cached['baseline25'][0][0]
            # Logical timestamps are from the data file, aligned by row identity.
            with np.load(ROOT/cfg['data_folder']/'packet_features.npz',allow_pickle=False) as z:
                mapping={int(row):i for i,row in enumerate(z['row_ids'])}
                ix=np.array([mapping[int(row)] for row in first['row_ids']]);times=z['logical_ns'][ix]
            for seconds in cfg['bootstrap_block_seconds']:
                block_cache={}
                for view in all_views:
                    for method in ['weighted_voting','weighted_reference','logistic_regression','random_forest']:
                        preds=[]
                        for j,(ev,sel) in enumerate(cached[view]):
                            np.testing.assert_array_equal(ev['row_ids'],first['row_ids'])
                            reference=cached['baseline25'][j][1]['weights']
                            preds.append(labels(ev,sel,method,reference))
                        blocks,strata,sup=confusion_by_blocks(first['y'],np.stack(preds),first['capture'],times,seconds)
                        block_cache[(view,method)]=blocks
                if case==cfg['cases'][0]:support += [dict(block_seconds=seconds,**z) for z in sup]
                for view in cfg['views']:
                    pairs=[('weighted_voting','baseline25','weighted_voting'),
                           ('weighted_reference','baseline25','weighted_voting'),
                           ('weighted_voting',view,'logistic_regression'),
                           ('weighted_voting',view,'random_forest')]
                    for method,other,comparator in pairs:
                        result,values=paired_block_bootstrap(block_cache[(view,method)],block_cache[(other,comparator)],
                            strata,cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                        key=f'{case}__{view}__{method}__vs__{other}__{comparator}__{seconds}s'
                        draws[key]=values;sensitivity.append(dict(case=case,view=view,method=method,comparison_view=other,
                            comparison_method=comparator,block_seconds=seconds,blocks=len(strata),captures=len(np.unique(strata)),**result))
                        print(key,result['point_delta'],flush=True)
        pd.DataFrame(sensitivity).to_csv(out/'paired_block_sensitivity.csv',index=False)
        pd.DataFrame(support).to_csv(out/'block_support.csv',index=False)
        np.savez_compressed(out/'block_bootstrap_draws.npz',**draws)
    main_t=temporal[(temporal['case']=='balanced')&(temporal.condition=='original')&(temporal.method=='weighted_voting')]
    main_s=pooled[(pooled['case']=='balanced')&(pooled.method=='weighted_voting')]
    main_t.to_csv(out/'main_balanced_temporal.csv',index=False);main_s.to_csv(out/'main_balanced_signature.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5));order=all_views;x=np.arange(len(order));width=.35
    for case,offset,color in [('original',-width/2,'#4b769d'),('balanced',width/2,'#d07b40')]:
        st=pooled[(pooled['case']==case)&(pooled.method=='weighted_voting')].set_index('view').loc[order]
        axes[0].bar(x+offset,100*st.macro_f1,width,label=case,color=color)
        axes[1].bar(x+offset,100*st.recall_Port_Scanning,width,label=case,color=color)
    for ax,title in zip(axes,['Macro-F1 signature diagnostic','Recall Port Scanning signature diagnostic']):
        ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,105);ax.set_xticks(x)
        ax.set_xticklabels([LABELS[v] for v in order],rotation=25,ha='right',fontsize=8);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Dua fold diagnostik dalam capture yang sama; bukan evaluasi capture baru')
    fig.tight_layout();fig.savefig(report/'signature_repair.png',dpi=170);fig.savefig(report/'signature_repair.svg');plt.close(fig)
    plot=base64.b64encode((report/'signature_repair.png').read_bytes()).decode()
    def as_html(df):return df.to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)
    contents=f'''<!doctype html><html lang="id"><meta charset="utf-8"><title>Perbaikan konteks WB</title>
    <style>body{{font:16px/1.55 system-ui;margin:35px auto;max-width:1250px;color:#203347;padding:0 20px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #ccd5dd;text-align:left}}th{{background:#eaf0f5}}section{{overflow:auto;margin:28px 0}}img{{width:100%}}.note{{background:#fff4dc;padding:15px;border-left:5px solid #b27a19}}</style>
    <h1>Perbaikan fitur untuk kegagalan Port Scanning</h1><p>20 Oktober 2026 ·150 expert fit ·30 bundle ·parameter/split tetap ·G memilih sebelum V.</p>
    <p class="note">Pengembangan eksploratif setelah observasi; V lama dipakai ulang. Tidak ada capture baru, p-value, klaim generalisasi independen atau jaminan novelty/Q1. Semua kandidat/treatment/model tunggal tetap ditampilkan. Audit SHAP lama tidak menjadi audit model baru.</p>
    <p>Context40 menghapus semua fitur paket lama. Context64 menambah24 rasio deterministik tanpa state/informasi mentah baru. Legacy25_context64 mempertahankan25 fitur paket untuk mengisolasi efek penghapusan fitur tersebut. Baseline25/29 memakai cache terverifikasi tanpa refit.</p>
    <img alt="Macro-F1 dan recall Port Scanning signature" src="data:image/png;base64,{plot}">
    <section><h2>WB Balanced, temporal: mean tiga seed pada V yang sama</h2>{as_html(main_t)}</section>
    <section><h2>WB Balanced, signature: pooled dua fold</h2>{as_html(main_s)}</section>
    <section><h2>Keputusan G dan hasilnya pada V</h2>{as_html(pd.DataFrame(selection_rows))}</section>
    <section><h2>Semua model pada signature</h2>{as_html(pooled)}</section>
    <p>Kuantil paired-block dalam tables/paired_block_sensitivity.csv adalah sensitivitas1/5/10detik dalam trace, bukan confidence interval populasi antar-capture. Kegagalan/pengurangan skor tidak disembunyikan.</p>
    <p><a href="../verification_quality.json">Verifikasi kualitas</a> · <a href="../tables/metrics.csv">Semua metrik per stage</a> · <a href="../tables/signature_confusions.csv">Confusion counts</a> · <a href="../tables/paired_block_sensitivity.csv">Sensitivitas blok</a></p></html>'''
    (report/'LAPORAN_CONTEXT_REPAIR.html').write_text(contents,encoding='utf8')
    e.write_json(report/'summary.json',dict(quality=verification,balanced_temporal=main_t.to_dict('records'),
        balanced_signature=main_s.to_dict('records'),new_experiments_not_independent=True))
    print('Report complete:',report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--enrich-only',action='store_true')
    args=p.parse_args();analyze(args.run,args.enrich_only)
