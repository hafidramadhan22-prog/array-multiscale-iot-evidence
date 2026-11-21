"""Run the retained numerical/visual gates plus the initial-study checks."""
from pathlib import Path
import inspect
import json
import statistics
import sys
import unicodedata
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import verify_array_complete as previous
BASE=ROOT/'paper/array_20261022_integrated'


def main():
    # Reuse the complete 427 numerical checks unchanged. Only the new revision
    # path and expected document lengths/table/figure counts differ.
    namespace=dict(previous.__dict__)
    namespace.update(BASE=BASE,OUT=ROOT/'tmp/pdfs/array_integrated_20261022')
    code=inspect.getsource(previous.main).replace("('main',22,6,4),('supplement',30,21,2)",
                                               "('main',23,8,4),('supplement',32,24,5)")
    assert code!=inspect.getsource(previous.main)
    exec(compile(code,__file__,'exec'),namespace)
    namespace['main']()
    t=json.loads((BASE/'tables.json').read_text())
    initial=BASE/'evidence/historical_initial'
    full=initial/'runs/FULL_20261013_01'
    rank=previous.read(initial/'reports/REKOMENDASI_20261015_01/ranking_all_six.csv')
    selection=previous.read(full/'feature_mode_selection.csv')
    cv=previous.read(full/'cv_feature_comparison.csv')
    checks=[]
    def check(a,b,name,digits=4):
        assert f'{float(a):.{digits}f}'==f'{float(b):.{digits}f}',(name,a,b)
        checks.append(name)
    for r,s in zip(t['initial_features']['rows'],selection):
        assert r[0]==s['feature_mode']
        for i,k in enumerate(['macro_f1_mean_across_six','ddos_f2_mean_across_six','ddos_recall_mean_across_six'],1):
            check(r[i],100*float(s[k]),'feature/'+k+'/'+s['feature_mode'])
        check(r[4],s['score'],'selection_score/'+s['feature_mode'],6)
        assert abs(float(s['score'])-.5*(float(s['macro_f1_mean_across_six'])+float(s['ddos_f2_mean_across_six'])))<1e-12
    for r,s in zip(t['initial_six']['rows'],rank):
        for i,k in enumerate(['macro_f1','ddos_f2','latency_batch_ms','delta_peak_mib','tradeoff_score'],1):
            check(r[i],float(s[k])*(100 if i<3 else 1),'ranking/'+s['config']+'/'+k,6 if i==5 else 4)
        perf=statistics.mean(float(s[k]) for k in ['macro_f1','recall_macro','auc_ovr_macro'])
        eff=statistics.mean(min(float(x[k]) for x in rank)/float(s[k]) for k in ['latency_batch_ms','delta_peak_mib'])
        assert abs(.6*perf+.4*eff-float(s['tradeoff_score']))<1e-12
        checks.append('composite/'+s['config'])
    for r,s in zip(t['historical_six']['rows'],rank):
        rows=previous.read(full/f"dstport29/tables/holdout_per_class_{s['config']}.csv")
        for j,(c,support) in enumerate(zip(previous.CLASSES,[4860,2050,2014,2062,2185]),1):
            x=next(x for x in rows if x['']==c)
            assert int(float(x['support']))==support
            check(r[j],100*float(x['recall']),'historical_recall/'+s['config']+'/'+c)
    for i,(mode,variant) in enumerate((m,v) for m in ['bands28','dstport29'] for v in ['baseline_search','ddos_tuned']):
        rows=[r for r in cv if r['feature_mode']==mode and r['variant']==variant];assert len(rows)==6
        for j,k in enumerate(['macro_f1_mean','ddos_f2_mean','ddos_recall_mean','port_scanning_recall_mean'],1):
            check(t['initial_tuning']['rows'][i][j],100*statistics.mean(float(r[k]) for r in rows),'tuning/'+mode+'/'+variant+'/'+k)
    for mode in ['bands28','dstport29']:
        rows=[r for r in cv if r['feature_mode']==mode]
        assert len(rows)==len(t['initial_cv_'+mode]['rows'])==12
        for i,(r,s) in enumerate(zip(t['initial_cv_'+mode]['rows'],rows)):
            for j,k in enumerate(['macro_f1','ddos_f2','ddos_recall'],1):
                mean,sd=r[j].split(' +/- ')
                check(mean,100*float(s[k+'_mean']),f'cv/{mode}/{i}/{k}/mean')
                check(sd,100*float(s[k+'_std']),f'cv/{mode}/{i}/{k}/sd')
    for r,s in zip(t['historical_full']['rows'],rank):
        for i,k in enumerate(['accuracy','recall_macro','auc_ovr_macro','performance_score','efficiency_score'],1):
            check(r[i],float(s[k])*(100 if i<3 else 1),'historical_full/'+s['config']+'/'+k,4 if i<3 else 6)
    assert float(rank[2]['ddos_f2'])>float(rank[0]['ddos_f2'])
    assert sum(abs(float(r['macro_f1'])-float(rank[0]['macro_f1']))<1e-12 for r in rank)==4
    freeze=json.loads((full/'feature_mode_freeze.json').read_text())
    assert freeze['selection_used_holdout'] is False
    policy=json.loads((initial/'reports/REKOMENDASI_20261015_01/recommendation.json').read_text())
    assert policy['revision_after_observation'] and not policy['main_rule_frozen_before_holdout']
    assert policy['unchanged_scores'] and not policy['new_training'] and not policy['model_deserialization']
    q=json.loads((BASE/'quality_check.json').read_text())
    for stem,terms in [('main',['28 and 29','two-inner','0.974691','95.5179','+0.0116','-0.0035','Table 8:',
                               'Initial feature, tuning and ensemble comparison','Figure 1: Complete study flow']),
                       ('supplement',['Table S24:','Figure S4:','Figure S5:','KernelSHAP','50 class-stratified training'])]:
        doc=pymupdf.open(BASE/'elsarticle'/f'{stem}.pdf')
        text=unicodedata.normalize('NFKC','\n'.join(p.get_text() for p in doc))
        for term in terms:assert term in text,term
    q['numeric_checks']+=len(checks)
    q['additional_initial_study_checks']=len(checks)
    q['initial_study_checks_passed']=checks
    q['historical_policy_revision_disclosed']=True
    q['cross_protocol_score_subtraction_rejected']=True
    (BASE/'quality_check.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(dict(total_numeric_checks=q['numeric_checks'],initial_study_checks=len(checks),
                         pages={r['name']:r['pages'] for r in q['pdfs']}),indent=2))


if __name__=='__main__':main()
