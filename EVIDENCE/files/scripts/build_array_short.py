"""Build a <=20-page editorial revision and a separate technical supplement.

No changes to experiments, fitting, prediction, dashboard or prior deliveries.
"""
from pathlib import Path
import hashlib,json,re,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_array_manuscript import inline
OLD=ROOT/'paper/array_20261021';BASE=ROOT/'paper/array_20261022_20pages';SRC=BASE/'elsarticle'
EQS={
 'mixture':r'f_c(x)=\sum_{j=1}^{5}w_jp_{jc}(x),\qquad \widehat y(x)=\operatorname*{arg\,max}_{c}f_c(x).',
 'certificate':r'\min_{b\ne a}(s_a-s_b)>R+\tau+\eta,\qquad R=\sum_{j\notin E}w_j.',
 'game':r'v^{B}_{x,c}(S)=\frac{1}{|B|}\sum_{b\in B}f_c(x_S,b_{\bar S}).',
 'gains':r'\Delta_D=A_D^{\mathrm{random}}-A_D^{\mathrm{rank}},\qquad \Delta_I=A_I^{\mathrm{rank}}-A_I^{\mathrm{random}}.'}
PREAMBLE=r'''\documentclass[preprint,12pt,nopreprintline]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,xurl}
\usepackage[hidelinks]{hyperref}
\usepackage[section]{placeins}
\journal{Array}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\emergencystretch}{2em}
\setlength{\belowcaptionskip}{5pt}
\raggedbottom
\begin{document}
'''
AUTHORS=r'''\author[humic]{M Hafid Ramadhan}
\author[humic]{Satria Mandala\corref{cor1}}
\cortext[cor1]{Corresponding author.}
\ead{satriamandala@telkomuniversity.ac.id}
\affiliation[humic]{organization={Humic Engineering (Humic Eng), School of Computing, Telkom University},addressline={Jl. Telekomunikasi no. 1},city={Bandung},postcode={40257},state={West Java},country={Indonesia}}
'''

def table(key,t):
    n=len(t['headers']);width=t.get('widths') or [1]*n;total=sum(width)
    cols=''.join('L{'+f'{w/total:.6f}'+r'\dimexpr\textwidth-'+str(2*n)+r'\tabcolsep\relax}' for w in width)
    cap=re.sub(r'^Table (?:\d+|A\d+)\.\s*','',t['caption'])
    font=r'\footnotesize' if key in ['stage_results','evidence_map','historical_six'] else r'\small'
    lines=[r'\begin{table}[!htbp]\centering'+font+r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}',
        r'\caption{'+inline(cap)+r'}\label{tab:'+key+r'}',r'\begin{tabular}{'+cols+r'}\toprule']
    head=lambda x:inline(x).replace('Ransomware',r'Ransom-\allowbreak{}ware')
    lines.append(' & '.join(r'\textbf{'+head(x)+'}' for x in t['headers'])+r'\\\midrule')
    for row in t['rows']:lines.append(' & '.join(inline(x) for x in row)+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    if t.get('note'):lines.append(r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize '+inline(t['note'])+r'\end{minipage}')
    lines.append(r'\end{table}');return '\n'.join(lines)

def body(md,tables):
    out=[]
    for para in re.split(r'\n\s*\n',md.strip()):
        if para.startswith('### '):out.append(r'\subsection{'+inline(para[4:])+'}')
        elif para.startswith('## '):
            title=para[3:];star='*' if title in ['Data and software availability','Declarations'] else ''
            out.append(r'\section'+star+'{'+inline(title)+'}')
        elif para.startswith('{{TABLE:'):out.append(table(para[8:-2],tables[para[8:-2]]))
        elif para.startswith('{{EQUATION:'):
            k=para[11:-2];out.append(r'\begin{equation}'+EQS[k]+r'\label{eq:'+k+r'}\end{equation}')
        elif para.startswith('{{FIGURE:'):
            k,cap=para[9:-2].split('|',1)
            out.append(r'\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/'+k+r'.pdf}\caption{'+inline(cap)+r'}\label{fig:'+k+r'}\end{figure}')
        else:out.append(inline(para.replace('\n',' ')))
    return '\n\n'.join(out)

def main():
    e=json.loads((OLD/'evidence/evidence.json').read_text())
    for rec in e['source_records']:assert hashlib.sha256((ROOT/rec['path']).read_bytes()).hexdigest()==rec['sha256'],rec['path']
    oldpdf=ROOT/'output/pdf/ARRAY_M_HAFID_RAMADHAN_SATRIA_MANDALA.pdf'
    assert hashlib.sha256(oldpdf.read_bytes()).hexdigest()=='62544ac3065e1d33c3605cb6253b1ccda8af3be3ca1d43b2de8751dcad1e316b'
    SRC.mkdir(parents=True,exist_ok=True)
    inv=json.loads((OLD/'template_inventory.json').read_text())
    for rec in inv['files']:
        src=OLD/rec['path'];assert hashlib.sha256(src.read_bytes()).hexdigest()==rec['sha256']
        dest=BASE/rec['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for name in ['elsarticle.cls','references.bib','highlights.txt']:
        shutil.copy2(OLD/'elsarticle'/name,SRC/name)
    shutil.copytree(OLD/'elsarticle/figures',SRC/'figures',dirs_exist_ok=True)
    shutil.copytree(OLD/'evidence',BASE/'evidence',dirs_exist_ok=True)
    shutil.copy2(OLD/'template_inventory.json',BASE/'template_inventory.json')
    t=json.loads((OLD/'tables.json').read_text())
    t['related_short']=dict(caption='Closest precedents and the empirical distinction.',headers=['Precedent','Established scope','Present distinction'],widths=[110,175,202],rows=[
        ['Kitsune [5]','Communication statistics for IDS','Bounded symmetric context under matched multiclass protocols'],
        ['QWYC [6]; CATO [8]','Early exit and pipeline cost','Same-predictor certificate and compact-encoding equivalence'],
        ['E-XAI [10]; grouped IoT/WSN [11,24]','Quantitative/stable explanations','Cross-background finite-donor game and changed-input audit'],
        ['Compressed IDS [25]','Attribution fidelity under compression','No quantization/distillation; full-probability game preservation']],
        note='Conceptual comparison; no external system was rerun on these targets.')
    rt=t['runtime'];keep=[0,1,2,3,5,6,7]
    assert len(rt['headers'])==7 or len(rt['headers'])==8
    # Use the saved summary values, dropping only the replay SD column if present.
    sd=[i for i,h in enumerate(rt['headers']) if 'SD' in h]
    keep=[i for i in range(len(rt['headers'])) if i not in sd]
    t['runtime_short']=dict(caption='Same-session latency, replay time and inference delta peak RSS.',
        headers=[rt['headers'][i] for i in keep],rows=[[r[i] for i in keep] for r in rt['rows']],
        widths=[104,65,72,65,85,85],note='Batch columns are medians (45 calls per policy/batch); replay is the mean of five repeats. RSS means summarize three fresh processes per policy/workload; baseline is post-load. LR/RF use native pipelines.')
    hist=json.loads((ROOT/'dashboard/assets/research.json').read_text())['historical']['rows']
    t['historical_six']=dict(caption='Historical holdout comparison of all six configurations.',headers=['Rank','Configuration','Macro-F1 (%)','Batch (ms)','Delta RSS (MiB)','Score'],widths=[42,167,72,68,76,62],rows=[
        [r['rank_all_six'],r['config'],f"{100*float(r['macro_f1']):.4f}",f"{float(r['latency_batch_ms']):.4f}",f"{float(r['delta_peak_mib']):.4f}",f"{float(r['tradeoff_score']):.6f}"] for r in hist],note='Historically observed holdout; exploratory ranking. Balanced names the training treatment, not a balanced test set.')
    (BASE/'tables.json').write_text(json.dumps(t,indent=2)+'\n')
    oldmd=(OLD/'manuscript.md').read_text(encoding='utf8');abstract=oldmd.split('## Abstract\n\n',1)[1].split('\n\nKeywords:',1)[0]
    md=(BASE/'main_body.md').read_text(encoding='utf8')
    supp=(BASE/'supplement_body.md').read_text(encoding='utf8')
    maintex=PREAMBLE+r'\begin{frontmatter}\title{Multiscale endpoint context and prediction preserving execution for explainable IoT intrusion detection}'+'\n'+AUTHORS
    maintex+=r'\begin{abstract}'+inline(abstract)+r'\end{abstract}'+'\n'+r'\begin{keyword}IoT intrusion detection \sep endpoint context \sep weighted voting \sep early exit \sep SHAP \sep reproducibility\end{keyword}\end{frontmatter}'+'\n'
    maintex+=r'\noindent\textit{Research article draft for author review, 22 October 2026. Technical details are provided separately as supplementary material.}'+'\n\n'
    maintex+=body(md,t)+r'\FloatBarrier\bibliographystyle{elsarticle-num}\bibliography{references}\end{document}'
    (SRC/'main.tex').write_text(maintex,encoding='utf8')
    stex=PREAMBLE+r'\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\theHtable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}'
    stex+=r'\begin{center}{\Large\bfseries Supplementary material}\par\medskip M Hafid Ramadhan and Satria Mandala\par\smallskip 22 October 2026\end{center}'+'\n\n'+body(supp,t)+r'\FloatBarrier\end{document}'
    (SRC/'supplement.tex').write_text(stex,encoding='utf8')
    (BASE/'editorial_build.json').write_text(json.dumps(dict(date='2026-10-22',page_limit_main_including_references=20,
        format='elsarticle preprint 12pt, original margins',font_size_reduced=False,margin_reduced=False,
        main_body_words=len(re.sub(r'\{\{.*?\}\}','',md).split()),abstract_words=len(abstract.split()),
        tables_main=4,figures_main=3,tables_supplement=12,figures_supplement=1,
        original_pdf_sha256=hashlib.sha256(oldpdf.read_bytes()).hexdigest(),source_records_verified=len(e['source_records']),
        method_changed=False,results_changed=False,training_performed=False,models_loaded=False),indent=2)+'\n')
    print((BASE/'editorial_build.json').read_text())
if __name__=='__main__':main()
