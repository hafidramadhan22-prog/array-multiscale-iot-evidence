"""Independently check saved-result tables, PDF contents, traceability and layout."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics
import unicodedata
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paper/array_20261022_complete'
OUT=ROOT/'tmp/pdfs/array_complete_20261022'
CLASSES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']
METHODS=['decision_tree','random_forest','xgboost','lightgbm','logistic_regression','soft_voting','weighted_voting']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    t=json.loads((BASE/'tables.json').read_text())
    build=json.loads((BASE/'editorial_build.json').read_text())
    for r in build['source_records']:assert sha(ROOT/r['path'])==r['sha256'] and sha(ROOT/r['catalog'])==r['catalog_sha256']
    old=json.loads((BASE/'evidence/evidence.json').read_text())
    for r in old['source_records']:assert sha(ROOT/r['path'])==r['sha256']
    template=json.loads((BASE/'template_inventory.json').read_text())
    for r in template['files']:assert sha(BASE/r['path'])==r['sha256']
    previous=build['previous_deliverables']
    for r in previous['pdfs']:assert sha(ROOT/r['path'])==r['sha256']
    assert sha(ROOT/previous['archive']['path'])==previous['archive']['sha256']
    checks=[]

    def check(a,b,label,digits=4):
        assert f'{float(a):.{digits}f}'==f'{float(b):.{digits}f}',(label,a,b)
        checks.append(label)

    quality=read(BASE/'evidence/quality_multiscale_all_treatments.csv')
    for row,view in zip(t['main_quality']['rows'],['baseline25','baseline29','multiscale60','legacy25_multiscale60','legacy25_multiscale_directional96']):
        src=next(r for r in quality if r['view']==view and r['case']=='balanced')
        for i,k in enumerate(['macro_f1_mean','macro_f1','recall_Port_Scanning','recall_Ransomware'],1):check(row[i],100*float(src[k]),'quality/'+view+'/'+k)
    same=read(BASE/'evidence/same_feature_controls.csv')
    for row,m in zip(t['expert_controls']['rows'],METHODS):
        src=next(r for r in same if r['case']=='balanced' and r['method']==m)
        for i,k in enumerate(['macro_f1_mean','ddos_f2_mean','macro_f1','recall_Port_Scanning'],1):check(row[i],100*float(src[k]),'controls/'+m+'/'+k)
    for temporal,key in [(True,'recall_temporal_all'),(False,'recall_signature_all')]:
        assert len(t[key]['rows'])==14
        for row,(case,m) in zip(t[key]['rows'],[(c,m) for c in ['original','balanced'] for m in METHODS]):
            src=next(r for r in same if r['case']==case and r['method']==m)
            fields=['macro_f1','ddos_f2']+['recall_'+c for c in CLASSES]
            for i,k in enumerate(fields,1):check(row[i],100*float(src[k+('_mean' if temporal else '')]),key+'/'+case+'/'+m+'/'+k,2)
    source=read(BASE/'evidence/signature_confusions.csv')
    matrices={}
    for view in ['baseline25','legacy25_multiscale60']:
        cm=np.zeros((5,5),dtype=np.int64)
        for r in source:
            if r['view']==view and r['case']=='balanced' and r['method']=='weighted_voting':cm[CLASSES.index(r['true']),CLASSES.index(r['predicted'])]=int(r['count'])
        assert cm.sum()==52584
        matrices[view]=cm
    # Validate exact heatmap counts and class-recall rows independently of the derived CSV.
    for offset,view in [(0,'baseline25'),(5,'legacy25_multiscale60')]:
        for i in range(5):
            for j in range(5):check(t['confusion_counts']['rows'][offset+i][j+1].replace(',',''),matrices[view][i,j],'cm/'+view+f'/{i}/{j}',0)
    with np.load(ROOT/'runs/MULTISCALE_REPAIR_FULL_20261020_01/stages/temporal_seed42/legacy25_multiscale60__balanced/evaluation.npz',allow_pickle=False) as a:
        cm=np.zeros((5,5),dtype=np.int64);np.add.at(cm,(a['y'],a['p_weighted'].argmax(1)),1)
        row_ids=a['row_ids'].copy();prob=a['p_weighted'].copy();expert=a['p_experts'].copy()
    for row,cm in zip(t['class_recall']['rows'],[cm,matrices['legacy25_multiscale60']]):
        assert int(row[1].replace(',',''))==cm.sum()
        for i in range(5):check(row[i+2],100*cm[i,i]/cm[i].sum(),'main_recall/'+row[0]+f'/{i}')
    faith=read(BASE/'evidence/faithfulness.csv')
    for row,m in zip(t['faithfulness']['rows'],['logistic_regression','random_forest','weighted_voting']):
        src=next(r for r in faith if r['method']==m)
        for i,k in enumerate(['deletion_auc','insertion_auc','deletion_gain','insertion_gain'],1):check(row[i],src[k],'faith/'+m+'/'+k)
    runtime=read(BASE/'evidence/runtime_summary.csv')
    for row,policy in zip(t['runtime_short']['rows'],['native_shared','native_certified','compact_shared','compact_certified','logistic_regression','random_forest']):
        src=next(r for r in runtime if r['Policy']==policy)
        for i,k in enumerate(['Batch1 ms','Batch1000 ms','Replay mean s','Delta RSS feature MiB','Delta RSS replay MiB'],1):check(row[i],src[k],'runtime/'+policy+'/'+k)
    raw=read(BASE/'evidence/latency_raw.csv')
    for row in t['throughput']['rows']:
        policy=row[0].replace(' ','_')
        for i,batch in enumerate([1,1000],1):
            values=[float(r['rows_per_second']) for r in raw if r['policy']=='baseline25__'+policy and int(r['batch'])==batch]
            assert len(values)==45;check(row[i],statistics.median(values),'throughput/'+policy+'/'+str(batch),2)
    robust=read(BASE/'evidence/robustness.csv')
    for row,m in zip(t['robustness_changed']['rows'],['weighted_voting','random_forest','logistic_regression']):
        src=next(r for r in robust if r['variant']=='query_delay_5ms' and r['subset']=='changed_input' and r['method']==m)
        assert int(src['n'])==33
        check(row[1],100*float(src['prediction_unchanged']),'robust/'+m+'/label',1)
        check(row[2],src['top3_jaccard'],'robust/'+m+'/jaccard')
        check(row[3],src['normalized_l1'],'robust/'+m+'/l1',6)
    # Check selected-class SHAP aggregation and additivity directly from the frozen game.
    importance=read(BASE/'evidence/grouped_shap_importance.csv')
    xr=ROOT/'runs/XAI_MULTISCALE_FULL_20261020_01/groups/A32'
    meta=json.loads((xr/'metadata.json').read_text());lookup={int(r):i for i,r in enumerate(row_ids)}
    with np.load(xr/'coalitions.npz',allow_pickle=False) as a:
        ix=[lookup[int(r)] for r in a['row_ids']]
        for m,j,expected in [('weighted_voting',0,prob[ix]),('random_forest',2,expert[ix,1,:]),('logistic_regression',5,expert[ix,4,:])]:
            assert np.allclose(a['values'][:,-1,j,:],expected,rtol=0,atol=1e-12)
            cls=a['values'][:,-1,j,:].argmax(1)
            phi=np.stack([a['phi'][i,:,j,int(c)] for i,c in enumerate(cls)])
            delta=np.array([a['values'][i,-1,j,c]-a['values'][i,0,j,c] for i,c in enumerate(cls)])
            assert np.max(np.abs(phi.sum(1)-delta))<1e-10
            mean=np.abs(phi).mean(0)
            for g,v in zip(meta['groups'],mean):
                src=next(r for r in importance if r['method']==m and r['group']==g)
                assert abs(float(src['mean_abs_phi'])-v)<1e-12
                assert abs(float(src['share_of_mean_abs'])-v/mean.sum())<1e-12
                checks.extend(['importance/'+m+'/'+g,'importance_share/'+m+'/'+g])
    # Validate the exploratory chronology without equating its protocols.
    history=BASE/'evidence/history'
    lift=read(history/'reports/E1_summary.csv')
    assert [round(100*float(r['dt_error_rescue_fraction']),2) for r in lift]==[5.04,1.37]
    mped=next(r for r in read(history/'MPED_FULL_20261019_04/mped_comparisons.csv') if r['reference']=='student_ce')
    check(-.1101,100*float(mped['mped_macro_f1_delta']),'history/mped_f1')
    check(-.4551,100*float(mped['mped_ddos_f2_delta']),'history/mped_f2')
    sbfe=read(history/'SBFE_FULL_20261019_01/temporal_summary.csv')
    for m,val in [('learned_utility',98.2140),('confidence',98.2606)]:
        check(val,100*float(next(r for r in sbfe if r['configuration']==m)['macro_f1_mean']),'history/sbfe/'+m)
    sig=read(history/'SBFE_FULL_20261019_01/signature_pooled.csv')
    assert all(float(r['ddos_f2'])==0 for r in sig if r['configuration'] in ['learned_utility','confidence'])
    port=json.loads((history/'PORT_FUSION_FULL_20261020_01/summary.json').read_text())
    selected=[r for r in port['selections'] if r['method']=='learned_robust']
    assert len(selected)==10 and all(r['C_mean_alpha']==0 for r in selected)
    bg=read(BASE/'evidence/background_sensitivity.csv')
    for row,m in zip(t['background_full']['rows'],['weighted_voting','random_forest','logistic_regression']):
        subset=[r for r in bg if r['method']==m];assert len(subset)==100
        for i,k in enumerate(['top3_jaccard','normalized_l1','cosine'],2):
            check(row[i],statistics.mean(float(r[k]) for r in subset),'background/'+m+'/'+k,6)
    selection=json.loads((BASE/'evidence/reference_selection_summary.json').read_text())
    assert sha(ROOT/selection['source'])==selection['source_sha256']
    for row,v in zip(t['serialized_sizes']['rows'][:5],selection['costs']['serialized_pipeline_bytes']):
        check(row[1].replace(',',''),v,'serialized/'+row[0],0)
    # Inspect all final PDF text and rendered pages; visual review follows separately.
    records=[]
    for stem,pages_max,tables,figures in [('main',22,6,4),('supplement',30,21,2)]:
        tex=(BASE/'elsarticle'/f'{stem}.tex').read_text(encoding='utf-8')
        log=(BASE/'elsarticle'/f'{stem}.log').read_text(encoding='utf-8')
        assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log,re.I)
        assert 'preprint,12pt' in tex
        assert not any(x in tex for x in ['geometry','newgeometry','textheight','textwidth=','\\tiny','\\scriptsize'])
        assert len(re.findall(r'\\begin\{table\}',tex))==tables
        assert len(re.findall(r'\\begin\{figure\}',tex))==figures
        doc=pymupdf.open(BASE/'elsarticle'/f'{stem}.pdf');assert len(doc)<=pages_max
        text=unicodedata.normalize('NFKC','\n'.join(p.get_text() for p in doc))
        if stem=='main':
            assert (BASE/'elsarticle/main.bbl').read_text().count(r'\bibitem')==25
            for term in ['M Hafid Ramadhan','Satria Mandala','97.8954','97.8800','0.9697','635','34.46','99.14','Confusion']:
                assert term.lower() in text.lower(),term
        else:
            for term in ['Table S21:','Table S13:','Table S20:','Figure S2:','LIFT-ID','MPED','18.3492','98.2140']:
                assert term in text,term
        pngs=[];captions=[]
        for i,page in enumerate(doc):
            assert len(page.get_text().strip())>65
            for block in page.get_text('blocks'):
                assert block[0]>=-.5 and block[1]>=-.5 and block[2]<=page.rect.width+.5 and block[3]<=page.rect.height+.5
            for line in page.get_text().splitlines():
                if re.match(r'^(Table|Figure) (?:S)?\d+:',line):captions.append(dict(page=i+1,caption=line))
            out=OUT/f'{stem}_{i+1:02d}.png';page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4),alpha=False).save(out);pngs.append(out)
        sheets=[]
        for start in range(0,len(pngs),4):
            ims=[Image.open(p).convert('RGB') for p in pngs[start:start+4]];w=max(x.width for x in ims);h=max(x.height for x in ims)
            sheet=Image.new('RGB',(2*w+24,2*(h+30)+24),'#dce3eb');draw=ImageDraw.Draw(sheet)
            for j,im in enumerate(ims):
                x=12+j%2*w;y=12+j//2*(h+30);draw.text((x+12,y+5),f'{stem.upper()} PAGE {start+j+1}',fill='black');sheet.paste(im,(x,y+30))
            p=OUT/f'{stem}_sheet_{start//4+1:02d}.png';sheet.save(p);sheets.append(p.relative_to(ROOT).as_posix())
        (OUT/f'{stem}_text.txt').write_text(text,encoding='utf-8')
        records.append(dict(name=stem,pages=len(doc),sha256=sha(BASE/'elsarticle'/f'{stem}.pdf'),tables=tables,figures=figures,
                            captions=captions,contact_sheets=sheets,page_image_sha256={p.name:sha(p) for p in pngs},visual_review_all_pages=False))
    result=dict(status='AUTOMATED_CHECKS_PASSED_VISUAL_PENDING',date='2026-10-22',numeric_checks=len(checks),
                numeric_mismatches=[],original_source_records_verified=len(old['source_records']),
                additional_source_records_verified=len(build['source_records']),template_files_verified=len(template['files']),
                prior_pdf_and_zip_preserved=True,main_font_pt=12,original_margins=True,
                new_training=False,model_deserialization=False,new_attribution_game_execution=False,pdfs=records)
    (BASE/'quality_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pdfs'},indent=2))
    print(json.dumps([dict(name=r['name'],pages=r['pages'],tables=r['tables'],figures=r['figures']) for r in records]))


if __name__=='__main__':main()
