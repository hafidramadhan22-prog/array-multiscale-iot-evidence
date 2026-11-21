"""Saved evidence/source audit for the GNB manuscript integration; no model loading."""
from pathlib import Path
import sys,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,confusion_matrix,fbeta_score
from build_array_enhanced_20261117 import tabs,body

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return p.read_text(encoding='utf8')

def main():
    src=ROOT/'paper/array_20261119_gnb/latex';old=ROOT/'paper/array_20261117_enhanced/latex'
    m=read(src/'main.tex');s=read(src/'supplement.tex');om=read(old/'main.tex');os=read(old/'supplement.tex')
    assert sha(old/'main.tex')=='cd4ecf0e994ac6752a58168c5cb9eebd49539755b8239500be77fbff07bbc2b4'
    assert sha(old/'supplement.tex')=='52dec5b885f0f574a2e2b1557dd5a371e9977f9580512269e87d6045f6c88795'
    assert len(tabs(m))==7 and len(tabs(s))==128
    assert [body(t) for t in tabs(s)[:114]]==[body(t) for t in tabs(os)]
    assert [body(t) for t in tabs(m)[:2]]==[body(t) for t in tabs(om)[:2]]
    for i,j in [(2,122),(3,123),(4,124),(5,125),(6,126)]:assert body(tabs(om)[i])==body(tabs(s)[j])
    for n in ['main','supplement']:
        text=read(src/(n+'.tex'));labels=re.findall(r'\\label\{([^}]+)\}',text)
        assert len(labels)==len(set(labels)),n
        refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text);assert set(refs)<=set(labels),(n,set(refs)-set(labels))
        bib=set(re.findall(r'@\w+\{([^,]+),',read(src/'references.bib')))
        for x in re.findall(r'\\cite\{([^}]+)\}',text):assert set(x.split(','))<=bib
        for figure in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):assert (src/figure).is_file()
        log=read(src/(n+'.log'));assert not re.search('undefined|Overfull|Float too large',log,re.I),n
    for figure in (old/'figures').iterdir():
        if figure.is_file():assert sha(figure)==sha(src/'figures'/figure.name)
    run=ROOT/'runs/ARRAY_GNB_FULL_20261119_01';done=json.loads(read(run/'complete.json'))
    assert sha(run/'catalog.json')==done['catalog_sha256']
    for rec in json.loads(read(run/'catalog.json')):assert sha(run/rec['path'])==rec['sha256']
    summary=pd.read_csv(ROOT/'reports/ARRAY_GNB_REVISION_20261119/gnb_metrics_recomputed.csv')
    for _,row in summary.iterrows():
        prefix={'temporal':'temporal_seed','packet_private':'private_fold','signature_shared':'signature_fold'}[row.protocol]
        arrays=[dict(np.load(p/'evaluation.npz',allow_pickle=False)) for p in sorted(run.glob(prefix+'*'))]
        if row.protocol=='temporal':
            observed=np.mean([f1_score(z['y'],z[row.method],average='macro') for z in arrays])
        else:observed=f1_score(np.concatenate([z['y'] for z in arrays]),np.concatenate([z[row.method] for z in arrays]),average='macro')
        assert abs(observed-row.macro_f1)<1e-12
        for field in ['macro_f1','balanced_accuracy','normal_fpr','ddos_f2']:assert f'{100*row[field]:.2f}' in tabs(m)[2]
    data=json.loads(read(ROOT/'paper/array_20261119_gnb/figure_sources/gnb_confusion_data.json'))
    ar=[dict(np.load(run/f'private_fold{i}/evaluation.npz',allow_pickle=False)) for i in [1,2]]
    for tag,display in [('WV','WV'),('GNB_default','GNB default'),('GNB_G_selected','GNB selected')]:
        cm=confusion_matrix(np.concatenate([z['y'] for z in ar]),np.concatenate([z[tag] for z in ar]),labels=range(5))
        np.testing.assert_array_equal(cm,data['matrices'][display])
    q=json.loads(read(ROOT/'reports/ARRAY_GNB_REVISION_20261119/pdf_quality.json'))
    for x in q:
        assert sha(ROOT/x['path'])==x['sha256'] and not x['bounds']
    result=dict(status='PASS_GNB_MANUSCRIPT_SAVED_EVIDENCE_AUDIT',main_tables=7,supplement_tables=128,old_supplement_table_bodies_unchanged=114,
      original_main_comparison_tables_retained=5,GNB_aggregate_macro_f1_recomputed=9,all_figure_inputs_exist=True,original_figures_unchanged=True,
      reference_WV_changed=False,reference_models_loaded=False,new_training=False,independent_test_added=False,
      main_sha256=sha(src/'main.tex'),supplement_sha256=sha(src/'supplement.tex'))
    out=ROOT/'reports/ARRAY_GNB_REVISION_20261119';(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
