"""Summaries from sealed serving measurements and saved prediction/XAI arrays."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
RUN=ROOT/'runs/ARRAY_SERVING_ABLATION_20261117_01';OUT=ROOT/'reports/ARRAY_ENHANCEMENT_20261117'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
def gate(folder,catalog,expected_sha):
    assert sha(folder/catalog)==expected_sha
    rec=read(folder/catalog)
    for r in rec:
        p=folder/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    return len(rec)
def main():
    done=read(RUN/'complete.json');assert done['status']=='PASS_SERVING_ABLATION' and done['policies']==8 and done['all_labels_match']
    count=gate(RUN,'catalog.json',done['catalog_sha256']);OUT.mkdir(parents=True,exist_ok=True)
    cfg=read(RUN/'configuration_freeze.json');assert not cfg['smoke'] and cfg['repeats']==30 and cfg['cold_rss_repeats']==5
    raw=pd.read_csv(RUN/'latency_raw.csv');replay=pd.read_csv(RUN/'replay_raw.csv');mem=pd.read_csv(RUN/'memory_raw.csv')
    assert len(raw)==480 and len(replay)==240 and len(mem)==80
    assert (raw.groupby(['policy','batch']).size()==30).all() and (replay.groupby('policy').size()==30).all()
    assert (mem.groupby(['policy','workload']).size()==5).all()
    assert replay.labels_match.all() and (replay.decisions==160).all() and (replay.packets==15665).all()
    np.testing.assert_allclose(mem.peak_mib-mem.baseline_mib,mem.delta_mib,atol=1e-12,rtol=0)
    feature=raw.groupby(['policy','batch']).seconds.agg(n='size',median='median',mean='mean',sd='std',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
    for c in ['median','mean','sd','q25','q75']:feature[c+'_ms']=1000*feature.pop(c)
    rs=replay.groupby('policy').total_seconds.agg(n='size',mean='mean',median='median',sd='std',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
    ms=mem.groupby(['policy','workload']).delta_mib.agg(n='size',mean='mean',sd='std',minimum='min',maximum='max').reset_index()
    feature.to_csv(OUT/'feature_summary.csv',index=False);rs.to_csv(OUT/'replay_summary.csv',index=False);ms.to_csv(OUT/'memory_summary.csv',index=False)
    p=replay.pivot(index='repeat',columns='policy',values='total_seconds');rng=np.random.default_rng(20261117)
    index=rng.integers(0,30,size=(10000,30));contrasts=[]
    pairs=[('encoding_full','WV_native_full','WV_compact_full'),('encoding_certified','WV_native_certified','WV_compact_certified'),
      ('stopping_native','WV_native_full','WV_native_certified'),('stopping_compact','WV_compact_full','WV_compact_certified'),
      ('combined_WV','WV_native_full','WV_compact_certified'),('RF_encoding','RF_native','RF_compact'),('LR_encoding','LR_native','LR_compact')]
    for key,a,b in pairs:
        delta=(p[a]-p[b]).to_numpy();boot=delta[index].mean(1)
        contrasts.append(dict(contrast=key,reference=a,optimized=b,mean_saving_s=float(delta.mean()),median_saving_s=float(np.median(delta)),sd_saving_s=float(delta.std(ddof=1)),
           ci95_low_s=float(np.quantile(boot,.025)),ci95_high_s=float(np.quantile(boot,.975)),optimized_slower=int((delta<0).sum()),relative_reduction_percent=float(100*(1-p[b].mean()/p[a].mean()))))
    interaction=((p.WV_native_full-p.WV_native_certified)-(p.WV_compact_full-p.WV_compact_certified)).to_numpy();boot=interaction[index].mean(1)
    contrasts.append(dict(contrast='factorial_interaction',reference='native stopping saving',optimized='compact stopping saving',mean_saving_s=float(interaction.mean()),median_saving_s=float(np.median(interaction)),sd_saving_s=float(interaction.std(ddof=1)),ci95_low_s=float(np.quantile(boot,.025)),ci95_high_s=float(np.quantile(boot,.975)),optimized_slower=None,relative_reduction_percent=None))
    pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)
    # Match estimator seeds within a fixed ordering and target set; no new fit.
    fact=ROOT/'runs/ARRAY_FACTORIAL_FULL_20261103_01';fc=read(fact/'complete.json')
    assert fc['status']=='PASS_FACTORIAL_FULL'
    catalog=read(fact/'artifacts_sha256.json');lookup={r['path'].replace('\\','/'):r for r in catalog}
    weights=[]
    from sklearn.metrics import f1_score
    for seed in [42,43,44]:
        data={}
        for view in ['c40_o1','c40_o10','c60_o1','c60_o10']:
            rel=f'stages/temporal_seed{seed}/{view}/evaluation.npz';r=lookup[rel];path=fact/rel;assert sha(path)==r['sha256']
            with np.load(path,allow_pickle=False) as z:data[view]=dict(z)
        weights_path=f'stages/temporal_seed{seed}/fixed_weights.json';assert sha(fact/weights_path)==lookup[weights_path]['sha256']
        w=read(fact/weights_path)['weights']
        for order in ['o1','o10']:
            a=data['c40_'+order];b=data['c60_'+order];np.testing.assert_array_equal(a['row_ids'],b['row_ids']);np.testing.assert_array_equal(a['y'],b['y'])
            pred=lambda z:sum(float(w[j])*np.asarray(z['p_experts'][:,j],dtype=np.float64) for j in range(5)).argmax(1)
            pa=pred(a);pb=pred(b);y=a['y'];fa=f1_score(y,pa,average='macro');fb=f1_score(y,pb,average='macro')
            weights.append(dict(seed=seed,ordering=order,n=len(y),C40_F1_percent=100*fa,C60_F1_percent=100*fb,gain_pp=100*(fb-fa),C40_errors=int((pa!=y).sum()),C60_errors=int((pb!=y).sum()),errors_removed=int((pa!=y).sum()-(pb!=y).sum())))
    pd.DataFrame(weights).to_csv(OUT/'paired_context_seed_gains.csv',index=False)
    # More detailed summaries of the same fixed finite-donor games.
    xai=ROOT/'runs/ARRAY_XAI_FULL_20261103_01';xd=read(xai/'complete.json');gate(xai,'catalog.json',xd['catalog_sha256'])
    faith=pd.read_csv(xai/'faithfulness.csv');q=faith[(faith.donor=='B32')&(faith.ranking_donor=='A32')&(faith.ranking=='signed')]
    q.groupby(['method','true_class']).agg(n=('row_id','size'),correct=('correct','sum'),deletion_gain=('deletion_gain','mean'),deletion_gain_sd=('deletion_gain','std'),insertion_gain=('insertion_gain','mean'),insertion_gain_sd=('insertion_gain','std'),nonpositive_deletion=('deletion_gain',lambda a:int((a<=0).sum()))).reset_index().to_csv(OUT/'XAI_per_class.csv',index=False)
    q.groupby(['method','correct']).agg(n=('row_id','size'),deletion_gain=('deletion_gain','mean'),insertion_gain=('insertion_gain','mean')).reset_index().to_csv(OUT/'XAI_correctness.csv',index=False)
    games={}
    for b in ['A32','B32']:
        with np.load(xai/'groups'/b/'coalitions.npz',allow_pickle=False) as z:games[b]=dict(z)
    rank=[]
    for name,j in [('weighted_voting',0),('random_forest',2),('logistic_regression',5)]:
        for i,rid in enumerate(games['A32']['row_ids']):
            c=int(games['A32']['values'][i,-1,j].argmax());a=np.abs(games['A32']['phi'][i,:,j,c]);b=np.abs(games['B32']['phi'][i,:,j,c]);ia=np.argsort(-a,kind='stable');ib=np.argsort(-b,kind='stable')
            rank.append(dict(method=name,row_id=int(rid),top1_same=bool(ia[0]==ib[0]),ordered_top3_same=bool(np.array_equal(ia[:3],ib[:3])),kendall_tau=float(kendalltau(a,b).statistic)))
    ranks=pd.DataFrame(rank);ranks.to_csv(OUT/'XAI_ordered_rank_raw.csv',index=False)
    ranks.groupby('method').agg(n=('row_id','size'),top1_same=('top1_same','sum'),ordered_top3_same=('ordered_top3_same','sum'),kendall_tau_mean=('kendall_tau','mean'),kendall_tau_min=('kendall_tau','min')).reset_index().to_csv(OUT/'XAI_ordered_rank.csv',index=False)
    evidence=[
      ('RQ1_context','Table1; S25-S26/S38/S113','runs/ARRAY_FACTORIAL_FULL_20261103_01/metrics.csv','scripts/verify_array_revision_20261117.py','within-trace; fixed target seeds'),
      ('RQ1_history','Table2; S39','reports/ARRAY_MAJOR_REVISION_20261103/history_summary.csv','scripts/audit_array_revision_20261117.py core','shared/private histories; source confounding'),
      ('RQ2_serving','Table4; S106-S109','runs/ARRAY_SERVING_ABLATION_20261117_01/replay_raw.csv','scripts/verify_array_enhancement_20261117.py','same frozen models; laptop/workload'),
      ('RQ2_historical','Figure3; S45/S105','runs/ARRAY_RUNTIME_FULL_20261103_01/replay_raw.csv','scripts/audit_array_revision_20261117.py core','3 November session; not pooled'),
      ('RQ3_game','Table5; S110-S112','runs/ARRAY_XAI_FULL_20261103_01/faithfulness.csv','scripts/verify_array_revision_20261117.py','finite game; 100 reused V targets'),
      ('RQ3_execution','S35-S36','runs/EXACT_RUNTIME_20261020_01/xai_equivalence.json','scripts/audit_array_revision_20261117.py core','seven-group contract; separate from nine-group'),
      ('GV_local','Table6; S74-S85','runs/FPR_SQL_GUARD_20261109_01/complete.json','scripts/audit_array_revision_20261117.py guard','post-freeze block3 uses existing sources'),
      ('External','Table7; S86-S95/S103-S104','runs/INDEPENDENT_CAPTURE_20261109_01/TON_flow_mapping/mapping_complete.json','scripts/audit_array_revision_20261117.py external','CIC scenario; TON flow labels; not pooled')]
    rows=[]
    for key,tables,path,command,limit in evidence:
        pth=ROOT/path;assert pth.is_file(),path
        rows.append(dict(claim_id=key,manuscript_location=tables,evidence_path=path,sha256=sha(pth),saved_result_verifier=command,scope=limit))
    pd.DataFrame(rows).to_csv(OUT/'claim_evidence_index.csv',index=False)
    result=dict(status='PASS_ENHANCEMENT_SUMMARIES_FROM_SEALED_EVIDENCE',serving_catalog_files=count,feature_calls=480,replay_calls=240,cold_RSS_calls=80,
       new_classifier_fits=0,new_sources=0,new_profiling_session=True,models_and_weights_unchanged=True,smoke_results_used=False,
       reference_41_75_percent_is_historical=True,new_session_contrasts=contrasts,source_catalog_sha256=done['catalog_sha256'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n','utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
