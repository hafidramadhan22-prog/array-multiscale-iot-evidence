"""Readable outcome and scientific plots from verified evidence only."""
from pathlib import Path
import base64,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ta_hafid_engine as e

NAMES={'baseline25':'WB25 + multiscale60', 'baseline29':'WB29 + context40',
       'time_context84':'Context60 + waktu24','legacy25_time_context84':'WB25 + context60 + waktu24',
       'legacy25_age_context66':'WB25 + context60 + age6'}


def md_table(df):
    cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def build():
    q=ROOT/'runs/IAT_REPAIR_FULL_20261020_01';r=ROOT/'runs/EXACT_RUNTIME_20261020_01'
    assert json.loads((q/'verification_quality.json').read_text())['status']=='PASS_CONTEXT_REPAIR_QUALITY'
    assert json.loads((r/'verification_independent.json').read_text())['status']=='PASS_RUNTIME_EVIDENCE_RECONSTRUCTION'
    out=q/'report';t=pd.read_csv(q/'tables/temporal_summary.csv');s=pd.read_csv(q/'tables/signature_pooled.csv')
    t=t.query("case == 'balanced' and condition == 'original' and method == 'weighted_voting'").set_index('view')
    s=s.query("case == 'balanced' and method == 'weighted_voting'").set_index('view');views=list(NAMES)
    quality=pd.DataFrame([{'Model':NAMES[v],'Macro-F1 temporal (%)':100*t.loc[v,'macro_f1_mean'],
        'Macro-F1 signature (%)':100*s.loc[v,'macro_f1'],'Recall scan signature (%)':100*s.loc[v,'recall_Port_Scanning'],
        'Recall ransom signature (%)':100*s.loc[v,'recall_Ransomware']} for v in views])
    quality.to_csv(q/'tables/time_runtime_quality_summary.csv',index=False)
    lat=pd.read_csv(r/'latency_summary.csv');mem=pd.read_csv(r/'memory_summary.csv');raw_replay=pd.read_csv(r/'replay_raw.csv')
    rp=raw_replay.groupby('policy').total_seconds.agg(['mean','std']);policies=list(rp.index)
    runtime=[]
    for policy in policies:
        row={'Policy':policy.replace('baseline25__',''),'Batch1 ms':1000*lat.query('policy == @policy and batch == 1')['median'].iloc[0],
             'Batch1000 ms':1000*lat.query('policy == @policy and batch == 1000')['median'].iloc[0],
             'Replay mean s':rp.loc[policy,'mean'],'Replay std s':rp.loc[policy,'std']}
        for workload,label in [('feature_input','Delta RSS feature MiB'),('replay','Delta RSS replay MiB')]:
            row[label]=mem.query('policy == @policy and workload == @workload').delta_peak_inference_mib.iloc[0]
        runtime.append(row)
    runtime=pd.DataFrame(runtime);runtime.to_csv(r/'readable_summary.csv',index=False)
    idx=runtime.set_index('Policy');native=float(idx.loc['native_certified','Replay mean s']);compact=float(idx.loc['compact_certified','Replay mean s'])
    full=float(idx.loc['native_shared','Replay mean s']);lr=float(idx.loc['logistic_regression','Replay mean s']);rf=float(idx.loc['random_forest','Replay mean s'])
    saved_native=100*(1-compact/native);saved_full=100*(1-compact/full)
    pairing=raw_replay.pivot(index='repeat',columns='policy',values='total_seconds')
    paired=100*(1-pairing['baseline25__compact_certified']/pairing['baseline25__native_certified'])
    eq=json.loads((r/'equivalence.json').read_text());xai=json.loads((r/'xai_equivalence.json').read_text())
    block=pd.read_csv(q/'tables/paired_block_sensitivity.csv')
    subset=block.query("case == 'balanced' and method == 'weighted_voting' and comparison_view == 'baseline25'")
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    x=np.arange(len(views));axes[0].bar(x-.18,[100*t.loc[v,'macro_f1_mean'] for v in views],.36,label='Temporal')
    axes[0].bar(x+.18,[100*s.loc[v,'macro_f1'] for v in views],.36,label='Signature');axes[0].set_ylim(70,100);axes[0].legend();axes[0].set_ylabel('%')
    axes[0].set_xticks(x, [NAMES[v] for v in views],rotation=45,ha='right',fontsize=8);axes[0].set_title('WB Balanced: kualitas pada development')
    names=list(idx.index);xx=np.arange(len(names));axes[1].bar(xx,idx['Replay mean s'],yerr=idx['Replay std s'],capsize=3,color='#3c8792')
    axes[1].set_xticks(xx,names,rotation=45,ha='right',fontsize=8);axes[1].set_ylabel('detik / 160 keputusan, 15.665 paket');axes[1].set_title('Replay: mean / SD lima pengulangan')
    axes[2].bar(xx,idx['Delta RSS replay MiB'],color='#bd8248');axes[2].set_xticks(xx,names,rotation=45,ha='right',fontsize=8)
    axes[2].set_ylabel('MiB: mean 3 proses cold');axes[2].set_title('Delta peak RSS replay termasuk ring')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Satu laptop dan capture lama; bukan estimasi generalisasi atau perangkat IoT')
    fig.tight_layout();fig.savefig(out/'time_runtime_summary.png',dpi=165);fig.savefig(out/'time_runtime_summary.svg');plt.close(fig)
    note=(ROOT/'docs/TIME_RUNTIME_REPORT_TEMPLATE.md').read_text(encoding='utf8').format(
        prob_delta=f"{eq['checks'][0]['expert_probability_max_abs_delta']:.3g}",
        xai_prob_delta=f"{xai['max_probability_difference']:.3g}",shap_delta=f"{xai['max_shap_difference']:.3g}",
        quality_table=md_table(quality),runtime_table=md_table(runtime),compact=compact,native=native,
        saved_native=saved_native,saved_full=saved_full,full=full,paired_min=paired.min(),paired_max=paired.max(),lr=lr,rf=rf)
    (ROOT/'docs/HASIL_TIME_RUNTIME_20261020.md').write_text(note,encoding='utf8')
    # Standalone report: embedded scientific figure and evidence tables.
    img=base64.b64encode((out/'time_runtime_summary.png').read_bytes()).decode()
    from export_review_html import markdown
    contents=f'''<!doctype html><html lang="id"><meta charset="utf-8"><title>Hasil waktu dan runtime</title>
    <style>body{{font:16px/1.6 system-ui;max-width:1350px;margin:35px auto;padding:0 24px;color:#203347}}img{{width:100%}}table{{font-size:12px;border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ccd5dd;text-align:left}}pre{{white-space:pre-wrap;font:15px/1.6 system-ui}}section{{overflow:auto}}</style>
    <h1>Perbaikan waktu dan runtime: hasil terverifikasi</h1><p>150fit tambahan; runtime classifier sama; 170tes lulus. Development lama, bukan capture independen.</p>
    <img src="data:image/png;base64,{img}" alt="Kualitas, runtime dan delta RSS">
    <section>{quality.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</section>
    <h2>Benchmark sesi baru</h2><section>{runtime.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</section>
    <h2>Penjelasan lengkap dan batas klaim</h2>{markdown(note)}</html>'''
    (ROOT/'LAPORAN_TIME_RUNTIME_20261020.html').write_text(contents,encoding='utf8')
    summary=dict(status='TIME_RUNTIME_ANALYSIS_COMPLETE',quality=quality.to_dict('records'),runtime=runtime.to_dict('records'),
        replay_time_reduction_vs_native_certified_percent=saved_native,replay_time_reduction_vs_native_full_percent=saved_full,
        paired_reduction_percent_min=float(paired.min()),paired_reduction_percent_max=float(paired.max()),
        independent_capture_evaluation=False,main_classifier_unchanged=True,G_selected_balanced_seed42='baseline25',
        model_change_recommended=False,report='LAPORAN_TIME_RUNTIME_20261020.html')
    e.write_json(out/'time_runtime_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':build()
