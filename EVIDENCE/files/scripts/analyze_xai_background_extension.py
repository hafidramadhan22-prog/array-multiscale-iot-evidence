"""Reconstruct SHAP extension scores from saved games, then render the audit."""
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
from run_wb_context_ablation import check_catalog
from run_xai_audit import metric_rows,model_output_indices
from xai_audit import exact_shapley,attribution_distance,METHODS
from run_wb_research import load_data
from wb_research import view_data,bijective_ports
from wb_behavior import PRIOR_SYMMETRIC
from review_formatting import markdown_table


def analyze(run):
    cfg=json.loads((run/'configuration_freeze.json').read_text());done=json.loads((run/'execution_complete.json').read_text())
    before=json.loads((run/'inputs_manifest.json').read_text())
    assert done['new_classifier_fits']==0
    assert [check_catalog(ROOT/z) for z in [cfg['source_run'],cfg['source_xai_run']]]==before['old_runs']
    for item in json.loads((run/'source_manifest.json').read_text()):assert e.sha256(run/'source'/item['path'])==item['sha256']
    for item in before['owned_models']:
        path=(ROOT/cfg['source_run']/item['path']).resolve()
        assert path.is_relative_to((ROOT/cfg['source_run']).resolve()) and e.sha256(path)==item['sha256'] and path.stat().st_size==item['bytes']
    with np.load(run/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
    with np.load(ROOT/cfg['source_xai_run']/'indices.npz',allow_pickle=False) as z:oldidx={k:z[k] for k in z.files}
    assert set(idx['background_A']).issubset(idx['A']) and set(idx['background_B']).issubset(idx['A'])
    assert not set(idx['background_A'])&set(idx['background_B'])
    assert set(oldidx['background_A']).issubset(idx['background_A']) and set(oldidx['background_B']).issubset(idx['background_B'])
    assert set(idx['row_ids']).issubset(idx['V']) and len(idx['row_ids'])==cfg['validation_rows']
    assert len(idx['background_A'])==len(idx['background_B'])==cfg['background_rows']
    wc=json.loads((ROOT/cfg['source_run']/'configuration_freeze.json').read_text())
    raw,y,holdout,data=load_data(wc)
    assert e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv')==before['canonical_sha256']
    assert e.sha256(ROOT/cfg['source_xai_run']/'indices.npz')==before['source_indices_sha256']
    assert e.sha256(ROOT/cfg['source_xai_run']/'perturbed_context.npz')==before['perturbed_context_sha256']
    positions=raw.index.get_indexer(idx['row_ids']);assert (positions>=0).all()
    np.testing.assert_array_equal(y[positions],idx['y'])
    assert not set(idx['row_ids'])&set(holdout)
    shifted=bijective_ports(raw,wc['port_bijection_offset'])
    with np.load(ROOT/cfg['source_xai_run']/'perturbed_context.npz',allow_pickle=False) as z:
        take=pd.Index(z['row_ids']).get_indexer(idx['row_ids']);assert (take>=0).all()
        jitter=z[cfg['jitter_condition']][take]
    faith=pd.read_csv(run/'tables/faithfulness.csv');dist=pd.read_csv(run/'tables/background_sensitivity.csv');robust=pd.read_csv(run/'tables/robustness.csv')
    checked=0;expected_dist=[];expected_robust=[];faith_rows=[]
    for view in cfg['views']:
        full,_,_=view_data(raw,data,view);frame=full.iloc[positions]
        shiftframe,_,_=view_data(shifted,data,view)
        jitterframe=frame.copy();jitterframe.loc[:,PRIOR_SYMMETRIC]=jitter
        modified={'port_bijection':shiftframe.iloc[positions],cfg['jitter_condition']:jitterframe}
        cache={}
        for folder in sorted((run/'groups'/view).iterdir()):
            with np.load(folder/'coalitions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
            np.testing.assert_array_equal(d['row_ids'],idx['row_ids']);np.testing.assert_array_equal(d['y'],idx['y'])
            meta=json.loads((folder/'metadata.json').read_text());groups=meta['groups']
            assert len(groups['endpoint_history'])==40
            for i in range(len(idx['row_ids'])):np.testing.assert_allclose(exact_shapley(d['values'][i]),d['phi'][i],rtol=1e-9,atol=1e-11)
            assert all(c['max_difference']<=1e-10 for c in meta['official_checks'])
            donors=idx['background_B'] if folder.name=='B32' else idx['background_A']
            np.testing.assert_array_equal(d['background_ids'],donors)
            cache[folder.name]=d
        base=cache['A32'];targets=base['values'][:,-1].argmax(2)
        for key in ['A32','B32']:
            np.testing.assert_allclose(cache[key]['values'][:,-1],base['values'][:,-1],rtol=0,atol=1e-12)
        for donor,rank in [('A32','A32'),('B32','B32'),('B32','A32')]:
            kwargs={} if donor==rank else dict(ranking_result=cache[rank])
            rows,_=metric_rows(cache[donor],idx['row_ids'],idx['y'],targets,donor=donor,ranking_donor=rank,**kwargs)
            faith_rows += [dict(view=view,**row) for row in rows]
        oldpath=ROOT/cfg['source_xai_run']/'groups'/f'{view}__balanced__seed42'/'donor_A'/'coalitions.npz'
        with np.load(oldpath,allow_pickle=False) as z:
            take=pd.Index(z['row_ids']).get_indexer(idx['row_ids']);assert (take>=0).all()
            old_phi=z['phi'][take]
            np.testing.assert_allclose(z['values'][take,-1],base['values'][:,-1],rtol=0,atol=1e-12)
        for label,first,second in [('A8_to_A32',old_phi,base['phi']),('A32_to_B32',base['phi'],cache['B32']['phi'])]:
            for i,row_id in enumerate(idx['row_ids']):
                for method,j in zip(METHODS,model_output_indices()):
                    t=int(targets[i,j]);expected_dist.append(dict(view=view,comparison=label,row_id=int(row_id),method=method,
                        **attribution_distance(first[i,:,j,t],second[i,:,j,t])))
        for variant in ['port_bijection',cfg['jitter_condition']]:
            result=cache[variant]
            for i,row_id in enumerate(idx['row_ids']):
                for method,j in zip(METHODS,model_output_indices()):
                    t=int(targets[i,j]);oldp=base['values'][i,-1,j];newp=result['values'][i,-1,j]
                    expected_robust.append(dict(view=view,variant=variant,row_id=int(row_id),method=method,
                        original_prediction=t,perturbed_prediction=int(newp.argmax()),prediction_unchanged=bool(newp.argmax()==t),
                        input_unchanged=bool(frame.iloc[i].equals(modified[variant].iloc[i])),
                        probability_l1=float(np.abs(oldp-newp).sum()),**attribution_distance(base['phi'][i,:,j,t],result['phi'][i,:,j,t])))
            subset=robust[(robust.view==view)&(robust.variant==variant)]
            identical=subset[subset.input_unchanged].row_id.unique();pos=pd.Index(idx['row_ids']).get_indexer(identical)
            np.testing.assert_array_equal(base['values'][pos],result['values'][pos]);np.testing.assert_array_equal(base['phi'][pos],result['phi'][pos])
    def compare(actual,rows,keys):
        nonlocal checked
        expected=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
        actual=actual.sort_values(keys).reset_index(drop=True)
        assert len(actual)==len(expected)
        for col in expected:
            if expected[col].dtype.kind in 'if':np.testing.assert_allclose(actual[col],expected[col],rtol=1e-11,atol=1e-12)
            else:np.testing.assert_array_equal(actual[col],expected[col])
        checked+=len(actual)
    compare(faith,faith_rows,['view','row_id','method','donor','ranking_donor','ranking'])
    compare(dist,expected_dist,['view','row_id','method','comparison'])
    compare(robust,expected_robust,['view','row_id','method','variant'])
    result=dict(status='PASS_XAI_BACKGROUND_SMOKE' if cfg['smoke_test'] else 'PASS_XAI_BACKGROUND_GAMES_AND_METRICS',
        metric_rows_reconstructed=checked,validation_rows=cfg['validation_rows'],background_rows=cfg['background_rows'],
        bundles=len(cfg['views']),new_classifier_fits=0,old_runs_unchanged=done['old_runs_unchanged'],
        independent_validation=False,previous_V_reused=True)
    e.write_json(run/'verification.json',result)
    if cfg['smoke_test']:
        print(json.dumps(result,indent=2));return
    signed=faith[faith.ranking=='signed'].groupby(['view','method','donor','ranking_donor'])[[c for c in faith if c.endswith('_auc') or c.endswith('_gain')]].mean().reset_index()
    signed.to_csv(run/'tables/faithfulness_summary.csv',index=False)
    distances=dist.groupby(['view','method','comparison'])[['normalized_l1','top3_jaccard','cosine']].mean().reset_index()
    distances.to_csv(run/'tables/background_summary.csv',index=False)
    rows=[]
    for (view,method,variant),frame in robust.groupby(['view','method','variant']):
        for name,part in [('all',frame),('changed_input',frame[~frame.input_unchanged]),
                          ('prediction_unchanged',frame[frame.prediction_unchanged]),
                          ('changed_input_prediction_unchanged',frame[(~frame.input_unchanged)&frame.prediction_unchanged])]:
            row=dict(view=view,method=method,variant=variant,subset=name,rows=len(part))
            row.update({c:float(part[c].mean()) if len(part) else None for c in ['prediction_unchanged','input_unchanged','probability_l1','normalized_l1','top3_jaccard','cosine']})
            rows.append(row)
    rs=pd.DataFrame(rows);rs.to_csv(run/'tables/robustness_summary.csv',index=False)
    report=run/'report';report.mkdir(exist_ok=True)
    primary=signed[signed.method=='weighted_voting']
    conclusions=[
        'BackgroundA8 diperbesar menjadiA32 dengan donor lama tetap di dalamnya; B32 disjoint.100V yang sama digunakan, tidak ada classifierfit baru atau konfirmasi independen.',
        'Pengelompokan endpoint_history40 tetap utuh untuk menjaga relasi internal. Audit ini menguji sensitivitas background dan perturbasi, bukan menciptakan metrik atau classifierSHAP baru.',
    ]
    for view in cfg['views']:
        cross=primary[(primary.view==view)&(primary.donor=='B32')&(primary.ranking_donor=='A32')].iloc[0]
        conclusions.append(f"{view}: rankingA32 dievaluasi padaB32 memberi deletion_gain={cross.deletion_gain:.6f}, insertion_gain={cross.insertion_gain:.6f}; tanda positif berarti lebih baik dari random dalam game ini.")
    limitations=[
        '32donor belum menjamin representasi distribusi training; A32/B32 berbeda game empiris. NilaiAUC bukan persentase kejujuran atau bukti kausal.',
        '100V berkuota kelas sama dan telah diamati sebelumnya. Konteks/model/port/querydelay sama dengan audit lama; hasil tidak dianggap replikasi pada capture baru.',
        'Hybrid marginal perkelompok belum dijamin valid sebagai paket/flow lengkap. Kelompok context40 tidak setara biaya/dimensi dengan kelompok lain.',
        'Baris input identik menggunakan ulang koalisi baseline; pisahkan subset input berubah. Invarianceport25 bukan robustness pada perubahan fitur nyata.',
        'Waktu komputasi selama run lain berjalan dan cache yang dipakai ulang adalah diagnosis, bukan benchmark deteksi/explainer.',
        'DDoS satu capture dan Normal canonical satu sumber; novelty kuat atau kelayakan jurnalQ1 belum dibuktikan oleh audit ini.',
    ]
    e.write_json(report/'summary.json',dict(verification=result,conclusions=conclusions,limitations=limitations))
    text='# Penguatan audit background SHAP\n\n'+'\n'.join('- '+s for s in conclusions)
    text+='\n\n## Faithfulness WB\n\n'+markdown_table(primary)
    text+='\n\n## Sensitivitas background\n\n'+markdown_table(distances[distances.method=='weighted_voting'])
    text+='\n\n## Robustness WB menurut subset\n\n'+markdown_table(rs[rs.method=='weighted_voting'])
    text+='\n\n## Batas interpretasi\n\n'+'\n'.join('- '+s for s in limitations)+'\n'
    (report/'HASIL_DAN_KESIMPULAN.md').write_text(text,encoding='utf8')
    fig,ax=plt.subplots(figsize=(8,4))
    p=primary[(primary.donor=='B32')&(primary.ranking_donor=='A32')]
    positions=np.arange(len(p));ax.bar(positions-.18,p.deletion_gain,.36,label='Deletion gain');ax.bar(positions+.18,p.insertion_gain,.36,label='Insertion gain')
    ax.set_xticks(positions,p.view);ax.set_ylabel('Gain over random subset expectation');ax.set_title('Ranking A32 evaluated with disjoint donors B32');ax.axhline(0,color='black',lw=.8)
    ax.set_ylim(top=1.30*max(p.deletion_gain.max(),p.insertion_gain.max()))
    ax.legend();fig.tight_layout()
    fig.savefig(report/'cross_background.png',dpi=180);fig.savefig(report/'cross_background.svg');plt.close(fig)
    def bullets(s):return '<ul>'+''.join('<li>'+html.escape(z)+'</li>' for z in s)+'</ul>'
    encoded=base64.b64encode((report/'cross_background.png').read_bytes()).decode()
    body='<h1>Penguatan faithfulness dan robustness SHAP</h1>'+bullets(conclusions)+f'<img src="data:image/png;base64,{encoded}" alt="Cross background gains">'
    for name,frame in [('Faithfulness, WB',primary),('Background sensitivity',distances),('Robustness by actual input/prediction changes',rs)]:
        body+='<h2>'+name+'</h2><div class="table">'+frame.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'</div>'
    body+='<h2>Batas</h2>'+bullets(limitations)
    (report/'LAPORAN_XAI_BACKGROUND.html').write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>SHAP background audit</title><style>body{font:16px system-ui;margin:32px;color:#1c303b}li{margin:10px 0}img{max-width:100%}.table{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccc}th{background:#e7f3f1}</style>'+body+'</html>',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();analyze(args.run.resolve())
