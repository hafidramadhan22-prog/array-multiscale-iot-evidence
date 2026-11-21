"""Recompute saved finite games and audit revised native text; no model loading."""
from pathlib import Path
import hashlib, json, re, math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paper/array_20261117_revision/latex'
OLD=ROOT/'paper/array_20261110_concise/elsarticle'
OUT=ROOT/'reports/ARRAY_REVISION_20261117'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def tables(s):return re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S)
def body(s):return s.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]

def xai_check():
    folder=ROOT/'runs/ARRAY_XAI_FULL_20261103_01'
    games={b:load(folder/'groups'/b/'coalitions.npz') for b in ['A32','B32']}
    results=[];maximum=0.
    G=9;states=np.arange(1<<G);sizes=np.array([int(s).bit_count() for s in states])
    for b,z in games.items():
        reconstructed=np.zeros_like(z['phi'])
        for i in range(G):
            s=states[(states&(1<<i))==0]
            weights=np.array([math.factorial(int(k))*math.factorial(G-int(k)-1)/math.factorial(G) for k in sizes[s]])
            reconstructed[:,i]=np.einsum('s,nsmc->nmc',weights,z['values'][:,s+(1<<i)]-z['values'][:,s])
        maximum=max(maximum,float(np.abs(reconstructed-z['phi']).max()))
        np.testing.assert_allclose(reconstructed,z['phi'],rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(reconstructed.sum(1),z['values'][:,-1]-z['values'][:,0],rtol=0,atol=1e-11)
    saved=pd.read_csv(folder/'faithfulness.csv').set_index(['row_id','method','donor','ranking_donor','ranking'])
    methods={'weighted_voting':0,'random_forest':2,'logistic_regression':5}
    gains={}
    for donor,rank in [('A32','A32'),('B32','B32'),('B32','A32')]:
        for method,m in methods.items():
            rows=[]
            for n,rid in enumerate(games[donor]['row_ids']):
                target=int(games['A32']['values'][n,-1,m].argmax())
                val=games[donor]['values'][n,:,m,target]
                ph=games[rank]['phi'][n,:,m,target]
                for ranking in ['signed','absolute','least']:
                    order=np.argsort(-ph if ranking=='signed' else (-np.abs(ph) if ranking=='absolute' else ph),kind='stable')
                    ins=np.r_[0,np.cumsum(1<<order)];dele=(1<<G)-1-ins
                    rand=np.array([val[sizes==k].mean() for k in range(G+1)])
                    auc=lambda x:float(np.trapezoid(x,dx=1/G))
                    calc=dict(deletion_auc=auc(val[dele]),insertion_auc=auc(val[ins]),random_deletion_auc=auc(rand[::-1]),random_insertion_auc=auc(rand),
                        deletion_gain=auc(rand[::-1])-auc(val[dele]),insertion_gain=auc(val[ins])-auc(rand),
                        comprehensiveness_one=float(val[dele[0]]-val[dele[1]]),sufficiency_gap_one=float(val[ins[-1]]-val[ins[1]]))
                    previous=saved.loc[(int(rid),method,donor,rank,ranking)]
                    for key,v in calc.items():np.testing.assert_allclose(v,previous[key],atol=1e-12,rtol=0)
                    if ranking=='signed':rows.append([calc['deletion_gain'],calc['insertion_gain']])
            gains[f'{donor}_rank_{rank}_{method}']=np.mean(rows,axis=0).tolist()
    for m,expected in [('weighted_voting',[.3203,.3270]),('random_forest',[.3016,.3246]),('logistic_regression',[.3043,.3155])]:
        assert [round(v,4) for v in gains['B32_rank_A32_'+m]]==expected
    return dict(status='PASS_FINITE_GAME_ATTRIBUTIONS_AND_2700_FAITHFULNESS_ROWS',rows=len(saved),games=200,coalitions_per_game=512,
        maximum_shapley_reconstruction_difference=maximum,cross_background_gains=gains,game_files=[dict(path=(folder/'groups'/b/'coalitions.npz').relative_to(ROOT).as_posix(),sha256=sha(folder/'groups'/b/'coalitions.npz')) for b in games])

def main():
    main=(BASE/'main.tex').read_text('utf8');sup=(BASE/'supplement.tex').read_text('utf8')
    for name,s,expected in [('main',main,7),('supplement',sup,104)]:
        old=(OLD/(name+'.tex')).read_text('utf8')
        assert len(tables(s))==len(tables(old))==expected
        for i,(a,b) in enumerate(zip(tables(s),tables(old)),1):assert body(a)==body(b),(name,i)
        labels=re.findall(r'\\label\{([^}]+)',s);refs=re.findall(r'\\ref\*?\{([^}]+)',s)
        assert len(labels)==len(set(labels)) and set(refs)<=set(labels),(name,set(refs)-set(labels))
        assert 'author approval pending' not in s.lower() and 'AGERS' not in s
        assert not any(t in s for t in ['%%TABLE_','%%FIGURE_','unused_cross_document_placeholder'])
    keys=set(re.findall(r'@\w+\{([^,]+),',(BASE/'references.bib').read_text('utf8')))
    used=lambda s:set(k for group in re.findall(r'\\cite\{([^}]+)',s) for k in group.split(','))
    assert len(keys)==30 and used(main)==keys and used(sup)<=keys
    assert sha(BASE/'references.bib')==sha(OLD/'references.bib')
    main_table_labels=[re.search(r'\\label\{([^}]+)',t)[1] for t in tables(main)]
    assert all(r'\ref{'+label+'}' in main for label in main_table_labels)
    images=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)',main+sup)
    assert len(images)==len(set(images))==14
    assert len(re.findall(r'\\begin\{figure\}',main))==5 and len(re.findall(r'\\begin\{figure\}',sup))==9
    assert all((BASE/i).is_file() for i in images)
    assert all(t in main for t in ['93.7694','96.9966','97.8971','97.8954','3.23','3.5262','41.75','1.0104','31.90','18.85','1.55','324/760','321/760'])
    assert 'All-packet in-scope' in tables(sup)[89] and 'denominator' in tables(sup)[89]
    assert 'after warm-up' not in tables(sup)[89].split(r'\midrule')[0].lower()
    coverage=read(OUT/'denominator_and_comparison_audit.json');assert coverage['status'].startswith('PASS_')
    phases={}
    for phase in ['core','integration','concise','external','guard','payload']:
        filename='independent_verification.json' if phase=='external' else 'verification.json'
        record=read(OUT/'saved_evidence_checks'/phase/filename);assert record['status'].startswith('PASS')
        phases[phase]=dict(status=record['status'],path=f'reports/ARRAY_REVISION_20261117/saved_evidence_checks/{phase}/{filename}')
    # Historical core report's new_expert_fits denotes 3-November fits, not this audit.
    r=ROOT/'runs/ARRAY_RUNTIME_FULL_20261103_01';raw=pd.read_csv(r/'replay_raw.csv')
    p=raw.pivot(index='repeat',columns='mode',values='total_seconds');assert len(p)==30 and raw.labels_match.all()
    reduction=100*(1-p.compact_certified.mean()/p.native_certified.mean())
    assert round(reduction,2)==41.75
    memory=pd.read_csv(r/'memory_raw.csv');np.testing.assert_allclose(memory.peak_mib-memory.baseline_mib,memory.delta_mib,rtol=0,atol=1e-12)
    xai=xai_check();(OUT/'finite_game_recomputation.json').write_text(json.dumps(xai,indent=2)+'\n','utf8')
    fig=read(OUT/'figure_revision.json');assert fig['status'].startswith('PASS')
    result=dict(status='PASS_REVISED_TEXT_AND_RECOMPUTED_SAVED_EVIDENCE',date='2026-11-17',
       main_tables=7,supplement_tables=104,main_figures=5,supplement_figures=9,references=30,all_111_numeric_table_bodies_unchanged=True,
       main_source_sha256=sha(BASE/'main.tex'),supplement_source_sha256=sha(BASE/'supplement.tex'),
       phases=phases,S90_definition='all packets including warm-up in denominator; post-warm-up mapped Normal+Ransomware in numerator',
       replay_reduction_percent_ratio_of_means=reduction,replay_pairs=30,memory_repeats=len(memory),finite_game_check=xai['status'],
       old_fits_not_repeated=True,models_deserialized=False,new_fits=0,new_inference=0,
       new_experiments_required_for_restricted_three_contributions=False,independent_source_generalization_established=False,GV_runtime_and_XAI_contract_established=False,
       author_approval_confirmed=False,submission_performed=False,reviewer_upload_completed=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n','utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
