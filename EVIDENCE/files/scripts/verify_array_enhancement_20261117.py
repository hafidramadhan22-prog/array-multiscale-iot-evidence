"""Saved-results audit for the enhanced paper; never deserialize or infer."""
from pathlib import Path
import sys,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import numpy as np
import pandas as pd
RUN=ROOT/'runs/ARRAY_SERVING_ABLATION_20261117_01';OUT=ROOT/'reports/ARRAY_ENHANCEMENT_20261117';BASE=ROOT/'paper/array_20261117_enhanced/latex';OLD=ROOT/'paper/array_20261117_revision/latex'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
def tables(s):return re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S)
def body(s):return s.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
def main():
    complete=read(RUN/'complete.json');assert complete['status']=='PASS_SERVING_ABLATION' and complete['new_classifier_fits']==0
    assert sha(RUN/'catalog.json')==complete['catalog_sha256']
    for row in read(RUN/'catalog.json'):
        p=RUN/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    assert read(RUN/'configuration_freeze.json')['smoke'] is False
    eq=read(RUN/'equivalence.json');assert len(eq)==24 and all(r['labels_match'] for r in eq)
    assert max(r['max_posterior_difference'] or 0 for r in eq)<=1e-12
    observed=read(OUT/'observed_outputs_audit.json');assert observed['status']=='PASS_ACTUAL_SERVING_OUTPUTS_AND_CSR_PERSISTED'
    assert sha(OUT/'observed_serving_outputs.npz')==observed['arrays_sha256']
    with np.load(OUT/'observed_serving_outputs.npz',allow_pickle=False) as z:actual=dict(z)
    with np.load(RUN/'expected_and_native_posteriors.npz',allow_pickle=False) as z:expected=dict(z)
    for key,value in actual.items():
        if key.endswith('__labels') or key.endswith('__row_ids'):np.testing.assert_array_equal(value,expected[key])
        else:
            family=key.split('_')[0];workload=key.split('__')[1]
            np.testing.assert_allclose(value,expected[family+'__'+workload+'__probabilities'],atol=1e-12,rtol=0)
    raw=pd.read_csv(RUN/'latency_raw.csv');replay=pd.read_csv(RUN/'replay_raw.csv');mem=pd.read_csv(RUN/'memory_raw.csv')
    assert len(raw)==480 and len(replay)==240 and len(mem)==80
    assert (raw.groupby(['policy','batch']).size()==30).all() and (replay.groupby('policy').size()==30).all()
    assert (mem.groupby(['policy','workload']).size()==5).all()
    np.testing.assert_allclose(mem.delta_mib,mem.peak_mib-mem.baseline_mib,atol=1e-12,rtol=0)
    f=raw.groupby(['policy','batch']).seconds.median()*1000;r=replay.groupby('policy').total_seconds.mean();m=mem.groupby(['policy','workload']).delta_mib.mean()
    for row in pd.read_csv(OUT/'feature_summary.csv').itertuples():np.testing.assert_allclose(row.median_ms,f.loc[(row.policy,row.batch)],rtol=0,atol=1e-12)
    for row in pd.read_csv(OUT/'replay_summary.csv').itertuples():np.testing.assert_allclose(row.mean,r.loc[row.policy],rtol=0,atol=1e-12)
    p=replay.pivot(index='repeat',columns='policy',values='total_seconds');indices=np.random.default_rng(20261117).integers(0,30,size=(10000,30))
    for c in pd.read_csv(OUT/'paired_contrasts.csv').itertuples():
        if c.contrast=='factorial_interaction':d=(p.WV_native_full-p.WV_native_certified)-(p.WV_compact_full-p.WV_compact_certified)
        else:d=p[c.reference]-p[c.optimized]
        boot=d.to_numpy()[indices].mean(1);np.testing.assert_allclose([c.mean_saving_s,c.ci95_low_s,c.ci95_high_s],[d.mean(),*np.quantile(boot,[.025,.975])],rtol=0,atol=1e-12)
    main=(BASE/'main.tex').read_text('utf8');sup=(BASE/'supplement.tex').read_text('utf8');before=(OLD/'main.tex').read_text('utf8');s0=(OLD/'supplement.tex').read_text('utf8')
    assert len(tables(main))==7 and len(tables(sup))==114
    for i,(a,b) in enumerate(zip(tables(before),tables(main))):
        if i!=3:assert body(a)==body(b),('main',i)
    for i,(a,b) in enumerate(zip(tables(s0),tables(sup)[:104])):assert body(a)==body(b),('supplement',i)
    assert body(tables(before)[3])==body(tables(sup)[104])
    names={'WV_native_full':'WV native/full','WV_native_certified':'WV native/cert.','WV_compact_full':'WV compact/full','WV_compact_certified':'WV compact/cert.','RF_native':'RF native','RF_compact':'RF compact','LR_native':'LR native','LR_compact':'LR compact'}
    for policy,name in names.items():
        row=[name,f'{f.loc[(policy,1)]:.3f}',f'{f.loc[(policy,1000)]:.3f}',f'{m.loc[(policy,"feature")]:.3f}',f'{r.loc[policy]:.4f}',f'{m.loc[(policy,"replay")]:.3f}']
        assert ' & '.join(row)+r'\\' in tables(main)[3],policy
    for name,s in [('main',main),('supplement',sup)]:
        labels=re.findall(r'\\label\{([^}]+)',s);refs=re.findall(r'\\ref\*?\{([^}]+)',s)
        assert len(labels)==len(set(labels)) and set(refs)<=set(labels),(name,set(refs)-set(labels))
        assert 'author approval pending' not in s.lower() and 'AGERS' not in s and "RF\\'s" not in s
        for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)',s):assert (BASE/image).is_file()
    assert sha(BASE/'references.bib')==sha(OLD/'references.bib')
    keys=set(re.findall(r'@\w+\{([^,]+),',(BASE/'references.bib').read_text('utf8')))
    used={k for x in re.findall(r'\\cite\{([^}]+)',main) for k in x.split(',')};assert keys==used and len(keys)==30
    for value in ['41.75','1.1425','1.1475','93.7694','96.9966','1.0104','31.90','18.85','1.55','1,657/2,000']:assert value in main,value
    for claim in pd.read_csv(OUT/'claim_evidence_index.csv').itertuples():assert sha(ROOT/claim.evidence_path)==claim.sha256
    xai=ROOT/'runs/ARRAY_XAI_FULL_20261103_01';q=pd.read_csv(xai/'faithfulness.csv');q=q[(q.donor=='B32')&(q.ranking_donor=='A32')&(q.ranking=='signed')]
    for row in pd.read_csv(OUT/'XAI_correctness.csv').itertuples():
        z=q[(q.method==row.method)&(q.correct==row.correct)];assert len(z)==row.n
        np.testing.assert_allclose([z.deletion_gain.mean(),z.insertion_gain.mean()],[row.deletion_gain,row.insertion_gain],atol=1e-12,rtol=0)
    j=read(OUT/'source_revision.json');assert j['main_source_sha256']==sha(BASE/'main.tex') and j['supplement_source_sha256']==sha(BASE/'supplement.tex')
    result=dict(status='PASS_ENHANCED_ARTICLE_AND_SEALED_SERVING_RESULTS',main_tables=7,supplement_tables=114,main_figures=5,supplement_figures=9,references=30,
      historical_supplement_tables_preserved=104,historical_main_tables_preserved=6,historical_cost_preserved_as='S105',feature_calls=480,replay_calls=240,cold_RSS_calls=80,
      serving_catalog_sha256=complete['catalog_sha256'],main_source_sha256=sha(BASE/'main.tex'),supplement_source_sha256=sha(BASE/'supplement.tex'),new_classifier_fits=0,
      new_profiling_session=True,old_41_75_percent_session_not_pooled=True,new_independent_sources=False,models_deserialized_by_verifier=False,submission_performed=False,author_approval_confirmed=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n','utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
