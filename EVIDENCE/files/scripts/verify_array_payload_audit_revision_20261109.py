"""Verify the new sources against sealed raw evidence and preserved prior tables."""
from pathlib import Path
import csv,hashlib,json,re,difflib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'paper/array_20261109_payload_audit/elsarticle'
O=ROOT/'paper/array_20261105_fpr_update/elsarticle'
R=ROOT/'reports/ARRAY_PAYLOAD_AUDIT_REVISION_20261109'
E=ROOT/'reports/PAYLOAD_EVIDENCE_20261105_01'
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blocks(s):return re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S)
def numbers(s):return re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',s)

def main():
    checks={};main=(B/'main.tex').read_text('utf8');supp=(B/'supplement.tex').read_text('utf8')
    oldm=(O/'main.tex').read_text('utf8');olds=(O/'supplement.tex').read_text('utf8')
    assert len(blocks(main))==12 and len(blocks(supp))==73
    for new,old,n in [(main,oldm,11),(supp,olds,65)]:
        for i,(a,b) in enumerate(zip(blocks(new)[:n],blocks(old))):
            # Row body retains every previous numeric value and label, independently of notes.
            aa=a[a.index(r'\begin{tabular}'):a.index(r'\end{tabular}')]
            bb=b[b.index(r'\begin{tabular}'):b.index(r'\end{tabular}')]
            assert aa==bb,('previous table changed',i+1)
    checks['previous_table_bodies_identical']=dict(main=11,supplement=65)
    assert main.count(r'\begin{figure}')==7 and supp.count(r'\begin{figure}')==6
    assert sha(B/'references.bib')==sha(O/'references.bib')
    keys=re.findall(r'@\w+\{([^,]+),',(B/'references.bib').read_text('utf8'));assert len(keys)==len(set(keys))==26
    cited=set(k.strip() for g in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',main) for k in g.split(','))
    assert cited==set(keys),(cited,set(keys))
    for name,s,n in [('main',main,12),('supplement',supp,73)]:
        labels=re.findall(r'\\label\{([^}]+)\}',s);assert len(labels)==len(set(labels))
        refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',s);assert set(refs)<=set(labels),(name,set(refs)-set(labels))
        log=(B/(name+'.log')).read_text('utf8')
        assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character|invalid character',log,re.I),name
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in s)
    assert 'author approval pending' not in main.lower() and 'all authors have approved' not in main.lower()
    assert 'hafid1050677@gmail.com' in main and 'OpenAI Codex' in main
    assert 'Turnitin' not in main and 'AI-detection' not in main
    checks['citations']=dict(entries=26,unchanged_bibliography=True,all_used=True,labels_resolved=True)
    source=read(E/'verification.json');assert source['max_probability_difference_from_saved']==0
    with (E/'advisor_demo_predictions.csv').open(encoding='utf-8-sig',newline='') as f:demo=list(csv.DictReader(f))
    assert len(demo)==48
    traces=[json.loads(l) for l in (E/'actual_payload_traces.jsonl').read_text('utf8').splitlines()]
    train=[r for r in traces if r['role']=='training200'];assert len(train)==200
    for field in ['syntax_or','syntax_tautology','syntax_union_select','syntax_isdecimal']:
        assert all(r['syntax'][field]==0 for r in train)
    assert len([r for r in train if r['actual_class']=='Normal'])==100
    ledger=read(R/'table_ledger.json');assert len(ledger)==9
    for t in ledger:
        s=main if t['scope']=='main' else supp;assert t['label'] in s
        block=next(b for b in blocks(s) if '\\label{'+t['label']+'}' in b)
        for row in t['rows']:
            cells=[]
            for v in row:
                ss=str(v).replace('&',r'\&').replace('_',r'\_\allowbreak{}').replace('%',r'\%').replace('#',r'\#');cells.append(ss)
            assert ' & '.join(cells)+r'\\' in block,(t['label'],row)
    assert source['canonical_availability']['Normal']['HTTP_requests']==0 and source['canonical_availability']['SQL_injection']['GET']==822
    for field,value in [('syntax_only','0.935234'),('tfidf_only','0.111796'),('syntax_tfidf','0.157671')]:
        r=next(x for x in demo if x['case']=='DEMO_SQL' and x['view']==field and x['method']=='weighted_voting')
        assert f"{float(r['p_SQL']):.6f}"==value
        assert (r['predicted']=='SQL_injection')==(field=='syntax_only')
    # Recompute the displayed reference/guard FP and FN from saved per-record decisions.
    with (E/'all_11118_main_decisions_recomputed.csv').open(encoding='utf-8-sig',newline='') as f:decisions=list(csv.DictReader(f))
    v=[r for r in decisions if r['cohort']=='V'];extra=[r for r in decisions if r['cohort']=='additional']
    assert len(v)==10358 and len(extra)==760
    assert sum(r['reference_prediction']!='Normal' for r in extra)==321 and all(r['guard_prediction']=='Normal' for r in extra)
    assert sum(r['true_class']=='Normal' and r['reference_prediction']!='Normal' for r in v)==36
    assert sum(r['true_class']!='Normal' and r['reference_prediction']=='Normal' for r in v)==2
    assert all(r['reference_prediction']==r['guard_prediction'] for r in v)
    checks.update(new_tables_checked_against_rows=9,actual_trace_records=1972,manual_probe_outputs=48,reference_and_guard_decisions_recounted=11118,negative_probe_preserved=True,old_PDFs_preserved=True,new_classifier_fits=0,new_inference_this_integration=0)
    state=read(ROOT/'CURRENT_STATE.json')
    for r in state['array_fpr_update_20261105']['pdfs']:assert sha(ROOT/r['path'])==r['sha256']
    for name in ['main','supplement']:
        (R/(name+'_changes.diff')).write_text(''.join(difflib.unified_diff((O/(name+'.tex')).read_text('utf8').splitlines(True),(B/(name+'.tex')).read_text('utf8').splitlines(True),fromfile='5 November/'+name,tofile='9 November/'+name)),encoding='utf8')
    result=dict(status='PASS_SOURCES_AND_RECORDED_EVIDENCE',checks=checks,main_source_sha256=sha(B/'main.tex'),supplement_source_sha256=sha(B/'supplement.tex'),model_or_weight_changes=False,independent_final_test=False,turnitin_performed=False)
    (R/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
