"""Create a real elsarticle source from frozen, already reviewed research evidence.

No fitting or model deserialization. The demo must pass its functional gate first.
"""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'paper/20261020';OUT=ROOT/'paper/array_20261021';SRC=OUT/'elsarticle'
TEMPLATE=Path(r'C:\Users\PREDATOR NEO 16\Downloads\elsarticle.zip')

def esc(s):
    trans={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(trans.get(c,c) for c in str(s))

def inline(s):
    # Numeric references from the evidence draft become real natbib citations.
    pieces=re.split(r'(\[(?:\d+)(?:(?:-|,)\d+)*\])',s)
    out=[]
    for part in pieces:
        if re.fullmatch(r'\[\d+(?:(?:-|,)\d+)*\]',part):
            nums=[]
            for v in part[1:-1].split(','):
                if '-' in v:
                    a,b=map(int,v.split('-'));nums.extend(range(a,b+1))
                else:nums.append(int(v))
            if all(1<=n<=25 for n in nums):out.append(r'\cite{'+','.join('r'+str(n) for n in nums)+'}')
            else:out.append(esc(part))
        else:
            # Inline source identifiers may contain underscores; keep them breakable.
            bits=re.split(r'(`[^`]+`)',part)
            out.extend((r'\texttt{'+esc(b[1:-1])+'}') if b.startswith('`') else esc(b) for b in bits)
    result=''.join(out).replace(r'\_',r'\_\allowbreak{}')
    for plain,math in [('p_j(x)',r'p_j(x)'),('w_j',r'w_j'),('s_c',r's_c'),('s_a',r's_a'),('s_b',r's_b'),('B_A',r'B_A'),('B_B',r'B_B')]:
        result=result.replace(esc(plain).replace(r'\_',r'\_\allowbreak{}'),'$'+math+'$')
    return re.sub(r'(\d+)\\textasciicircum\{\}(-?\d+)',lambda m:'$'+m[1]+'^{'+m[2]+'}$',result)

def main():
    gate=json.loads((ROOT/'dashboard/verification.json').read_text())
    assert gate['status']=='DASHBOARD_FUNCTIONALLY_VERIFIED',gate
    assert gate['unit_integration_tests_passed']==10 and gate['live_http_checks_passed']==16 and gate['virtual_dom_checks_passed']==20
    OUT.mkdir(parents=True,exist_ok=True);SRC.mkdir(exist_ok=True)
    # Safely preserve all files of the supplied template, including its licence.
    template_records=[]
    with zipfile.ZipFile(TEMPLATE) as z:
        for item in z.infolist():
            if item.is_dir():continue
            rel=Path(item.filename)
            if rel.parts[0]!='elsarticle' or '..' in rel.parts:raise ValueError('Unexpected template path')
            dest=(OUT/rel).resolve()
            if not dest.is_relative_to(OUT.resolve()):raise ValueError('Template traversal')
            dest.parent.mkdir(parents=True,exist_ok=True);data=z.read(item)
            # Do not overwrite customized manuscript files; template paths are standard.
            dest.write_bytes(data)
            template_records.append(dict(path=str(rel).replace('\\','/'),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (OUT/'template_inventory.json').write_text(json.dumps(dict(archive=str(TEMPLATE),sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),files=template_records),indent=2)+'\n')
    # Tectonic's virtual filesystem discards non-standard docstrip outputs.
    # This dtx has exactly one simple class guard, with no conditional subguards.
    # Extract the same guarded code locally, preserving comments/licence and source.
    dtx=(SRC/'elsarticle.dtx').read_text(encoding='utf8')
    assert dtx.count('%<*class>')==1 and dtx.count('%</class>')==1
    code=dtx.split('%<*class>\n',1)[1].split('%</class>',1)[0]
    assert not re.search(r'^%[<>]',code,re.M)
    cls='%% Generated from the supplied elsarticle.dtx, class guard.\n'
    cls+='%% Copyright 2007-2024 Elsevier Ltd. LPPL 1.3 or later.\n'
    cls+='%% Original dtx/ins, README and manifest accompany this file.\n'
    cls+='\n'.join(s for s in code.splitlines() if not s.startswith('%') or s.startswith('%%'))+'\n\\endinput\n'
    (SRC/'elsarticle.cls').write_text(cls,encoding='utf8')
    for p in (OLD/'figures').glob('figure*.pdf'):
        (SRC/'figures').mkdir(exist_ok=True);shutil.copy2(p,SRC/'figures'/p.name)
    ev=OUT/'evidence';ev.mkdir(exist_ok=True)
    for p in OLD.glob('*.csv'):shutil.copy2(p,ev/p.name)
    for n in ['evidence.json','tables.json','references.json']:
        shutil.copy2(OLD/n,ev/n)
    shutil.copy2(ROOT/'dashboard/verification.json',ev/'dashboard_verification.json')
    tables=json.loads((OLD/'tables.json').read_text())
    tables['prior_work']['rows'] += [
      ['Hossain and Islam [21]; Alharthi et al. [22]','Ensemble IDS and ML/DL comparisons in Array','Matched context ablations, fixed-function serving and grouped donor audit'],
      ['Ahmed et al. [23]','Threat-model-led lightweight ADS-B detection','Five-class IoT trace study; no ADS-B or embedded energy claim'],
      ['Sharmila and Deepanramkumar [24]','SHAP rank stability for federated WSN intrusion detection','Centralized full-voting donor game and execution equivalence'],
      ['Tulla et al. [25]','Attribution fidelity risks in compressed IDS','Same estimators and encoding semantics, without quantization or distillation']]
    (OUT/'tables.json').write_text(json.dumps(tables,indent=2)+'\n')
    refs=json.loads((OLD/'references.json').read_text())
    refs += [
      dict(id=21,authors='Md. Alamgir Hossain and Md. Saiful Islam',year=2023,title='Ensuring network security with a robust intrusion detection system using ensemble-based machine learning',venue='Array 19, 100306',doi='10.1016/j.array.2023.100306',url='https://doi.org/10.1016/j.array.2023.100306',verification='Crossref publisher deposit; author-uploaded full text read on ResearchGate.'),
      dict(id=22,authors='Ayesha Alharthi, Meera Alaryani and Sanaa Kaddoura',year=2025,title='A comparative study of machine learning and deep learning models in binary and multiclass classification for intrusion detection systems',venue='Array 26, 100406',doi='10.1016/j.array.2025.100406',url='https://doi.org/10.1016/j.array.2025.100406',verification='Primary Zayed institutional record and author-uploaded full text read on ResearchGate.'),
      dict(id=23,authors='Waqas Ahmed, Ammar Masood and Jawad Manzoor',year=2026,title='Toward realistic and lightweight intrusion detection for ADS-B: A comprehensive threat model and multi-attack machine learning evaluation',venue='Array 31, 101012',doi='10.1016/j.array.2026.101012',url='https://doi.org/10.1016/j.array.2026.101012',verification='Crossref publisher metadata and publisher-indexed abstract; full paper not retrieved.'),
      dict(id=24,authors='A. Helen Sharmila and P. Deepanramkumar',year=2026,title='Explanation stability assessment for federated explainable intrusion detection in wireless sensor networks',venue='Array 31, 101114',doi='10.1016/j.array.2026.101114',url='https://doi.org/10.1016/j.array.2026.101114',verification='Crossref publisher metadata, publisher-indexed title/DOI and primary author repository README. Repository still labels under review; no acceptance date inferred.'),
      dict(id=25,authors='Md. Hamid Borkot Tulla, Lin Yuan, Jiawei Zhang, Xiao Pu and Gwanggil Jeon',year=2026,title='Silent corruption: Logic collapse and attribution fidelity failure in compressed intrusion detection systems',venue='Information Sciences 757, 123908',doi='10.1016/j.ins.2026.123908',url='https://doi.org/10.1016/j.ins.2026.123908',verification='Crossref publisher metadata and publisher-indexed abstract/highlights. December issue date is future as of this draft; no online date inferred.')]
    (OUT/'references.json').write_text(json.dumps(refs,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    bib=[]
    for r in refs:
        authors=re.split(r', | and ',r['authors'])
        authors=[{'F. De Turck':'{De Turck}, F.','G. Draper Gil':'{Draper Gil}, G.'}.get(a,a) for a in authors]
        fields=dict(author=' and '.join(authors),title='{'+r['title']+'}',year=str(r['year']),howpublished=r['venue'])
        if r.get('doi'):fields['doi']=r['doi']
        else:fields['url']=r['url']
        # Generic bib type preserves the recorded venue exactly rather than inventing volume fields.
        bib.append('@misc{r'+str(r['id'])+',\n'+',\n'.join('  '+k+' = {'+(v if k in ['doi','url'] else esc(v).replace(r'\{','{').replace(r'\}','}'))+'}' for k,v in fields.items())+'\n}')
    (SRC/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf8')
    md=(OLD/'manuscript_template.md').read_text(encoding='utf8')
    abstract_new='Efficient IoT intrusion detection requires informative traffic context and consistent predictions after runtime optimization. We study a bounded, prior-only multiscale endpoint representation, a five-expert weighted classifier, and prediction-preserving execution on five Edge-IIoTset captures. Context for 52,684 development targets is reconstructed from 3,735,488 packets; boundary purging leaves 52,584 targets. Temporal development macro-F1 increases from 96.739% with a short-context reference to 97.895% across three estimator seeds. A separate packet-signature diagnostic increases from 71.843% to 97.296%, although shared unlabelled histories limit generalization claims. Compact execution reduces mean replay time from 1.011 to 0.663 s relative to the same native certified predictor. One of five paired repetitions is slower, with no uniform memory reduction. Encodings and expert probabilities match on all development inputs. Exact seven-group Shapley audits using disjoint training backgrounds yield deletion and insertion gains of 0.3297 and 0.3310 over random ordering. Re-executing 300 games preserves probabilities and attributions. Singleton, uniform-voting and timing controls expose competing explanations and class trade-offs. A local dashboard supports fixed-model inspection. Results support an auditable representation-and-execution study within observed traces; independent-capture validation remains necessary.'
    oldabstract=md.split('## Abstract\n\n',1)[1].split('\n\nKeywords:',1)[0]
    md=md.replace(oldabstract,abstract_new,1)
    md=md.replace('M Hafid Ramadhan\n\nResearch article manuscript | Draft for author and supervisor review | 20 October 2026',
      'M Hafid Ramadhan and Satria Mandala\n\nHumic Engineering (Humic Eng), School of Computing, Telkom University, Bandung, Indonesia\n\nCorrespondence: satriamandala@telkomuniversity.ac.id\n\nResearch article | Array review draft | 21 October 2026')
    md=md.replace('This manuscript was assembled without deserializing models or rerunning training.','The article is assembled from verified saved results without rerunning training. The companion dashboard loads only a fixed project-owned model after verifying its provenance and SHA256 checksum.')
    # Specific literature, under 200 derived words from any one newly consulted paper.
    newrelated='''### 2.4 Closest journal studies and the remaining empirical question

Array already includes ensemble intrusion detection: Hossain and Islam compare ensemble methods with feature selection on several public datasets [21]. Alharthi et al. compare classical and deep models on binary SDN and multiclass industrial-control data, reporting predictive and training-resource results [22]. These articles establish relevant journal scope; they do not provide matched scores for the present traces. Our distinction is an auditable sequence from prior-only context to fixed-function execution and quantitative explanation checks, rather than another claim that voting alone is novel.

Ahmed et al. frame lightweight ADS-B detection around a domain-specific threat model [23]. We likewise define the monitored classes and measured workloads, but our laptop trace replay cannot establish embedded feasibility. Sharmila and Deepanramkumar study SHAP explanation stability in federated wireless-sensor intrusion detection [24]. Their work further precludes treating quantitative SHAP stability as an unexplored IDS topic. The present audit instead uses a centralized five-expert predictor, exact seven-group donor games, separate ranking/evaluation backgrounds and implementation-equivalence checks.

Tulla et al. examine attribution-fidelity failures under IDS compression and propose fidelity-oriented compression methods [25]. This is particularly close to the question of whether efficiency preserves explanation. Our optimization retains the fitted estimators and their encoding semantics; it neither quantizes nor distils a new decision function. Its contribution is evidence that a shared compact implementation preserves the enumerated full-probability games, alongside explicit separation of early-exit labels from full-probability explanations. No new attribution metric or universally superior runtime is claimed.

'''
    md=md.replace('{{TABLE:prior_work}}',newrelated+'{{TABLE:prior_work}}',1)
    demo='''## 8 Local prototype and functional verification

A loopback-only dashboard operationalizes the reference Balanced model for reproducible supervisor review. It was implemented and functionally verified before this Array version was written. Its three views support model testing, inspection of stored research comparisons, and inspection of grouped SHAP audit results. It serves local files without third-party scripts or a cloud service. A manifest anchored in the application source checks the model and copied inference modules before any model deserialization. The fixed model has five experts and the 85-field domain schema described above; the interface neither trains a replacement nor accepts uploaded model objects.

The testing view predicts one target or all 100 saved audit targets and reports which experts actually executed. In comparison mode, certified labels are checked against full weighted voting. Certified-only mode returns labels and expert counts without presenting partial scores as posterior probabilities. Full mode evaluates all five experts before showing probabilities. CSV input accepts up to 1,000 already prepared rows, at most 8 MiB, with exactly the expected feature schema and an optional unique row identifier. It rejects missing history fields, label columns, duplicate headers and invalid context values. Uploaded rows are processed in memory and are not retained on disk. Unknown categories follow the fixed encoder contract.

The research view keeps the historically observed six-configuration comparison separate from temporal and signature diagnostics. Historical selection favored Balanced weighted voting according to the project's composite trade-off score; that historical ranking is not an independent post-selection evaluation or a numerical baseline for claiming the new temporal gain. A fixed replay reconstructs context from 15,665 saved packets and checks 160 decisions. It still obtains the 25 legacy fields from the supplied table. Request timings are labelled interactive measurements and do not replace the repeated benchmark in Section 6.

The explanation view displays only the saved A32 exact grouped attributions for the 100 audit targets, including the explained class, baseline and full probability. It does not attach those explanations to a newly uploaded CSV row. Background, grouping, donor-mask and robustness limitations remain visible. Recomputing new grouped explanations online was outside the prototype's scope and is not implied by its charts.

Ten unit/integration tests passed, covering fixed-prediction equivalence, manifest tampering before model loading, schema boundaries, label exclusion, full-versus-partial outputs, attribution additivity and context replay. Sixteen live HTTP checks passed, and twenty virtual-DOM interaction checks exercised navigation, sample filtering, predictions, CSV input and errors, research tables, seven attribution bars and replay results. The application binds only to localhost, restricts Host/Origin values, and does not expose model assets through its HTTP routes. Graphical browser inspection was interrupted; virtual-DOM checks do not establish pixel-level layout quality, accessibility compliance, operator usability or production security. These are software-verification results, not additional detection-performance experiments.

'''
    md=md.replace('## 8 Discussion',demo+'## 9 Discussion')
    md=md.replace('### 8.1','### 9.1').replace('### 8.2','### 9.2').replace('### 8.3','### 9.3')
    md=md.replace('## 9 Threats','## 10 Threats').replace('## 10 Conclusion','## 11 Conclusion')
    md=md.replace('The research software has 170 passing tests,','The preceding research snapshot records 170 passing tests,')
    md=md.replace('The named author is responsible','The named authors are responsible')
    start=md.index('## Author information and declarations')
    end=md.index('## Appendix A',start)
    md=md[:start]+'''## Author declarations awaiting confirmation

The supplied author order is M Hafid Ramadhan followed by Satria Mandala, with Satria Mandala as corresponding author. Both authors share the affiliation stated on the title page. Funding, competing interests and individual CRediT contributions have not yet been confirmed. Their absence must not be interpreted as a declaration of no funding or no conflict. The authors must complete and approve these statements before submission.

'''+md[end:]
    (OUT/'manuscript.md').write_text(md,encoding='utf8')
    abstract=md.split('## Abstract\n\n',1)[1].split('\n\nKeywords:',1)[0]
    eqs={
      'mixture':r'f_c(x)=\sum_{j=1}^{5}w_jp_{jc}(x),\qquad \widehat y(x)=\operatorname*{arg\,max}_{c}f_c(x).',
      'certificate':r'\min_{b\ne a}(s_a-s_b)>R+\tau+\eta,\qquad R=\sum_{j\notin E}w_j.',
      'game':r'v^{B}_{x,c}(S)=\frac{1}{|B|}\sum_{b\in B} f_c(x_S,b_{\bar S}).',
      'gains':r'\Delta_D=A_D^{\mathrm{random}}-A_D^{\mathrm{rank}},\qquad\Delta_I=A_I^{\mathrm{rank}}-A_I^{\mathrm{random}}.'}
    tex=[r'''\documentclass[preprint,12pt,nopreprintline]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,array,longtable,tabularx}
\usepackage{graphicx,xurl}
\usepackage[hidelinks]{hyperref}
\usepackage[section]{placeins}
\journal{Array}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\emergencystretch}{2em}
\begin{document}
\begin{frontmatter}
\title{Multiscale endpoint context and prediction preserving execution for explainable IoT intrusion detection}
\author[humic]{M Hafid Ramadhan}
\author[humic]{Satria Mandala\corref{cor1}}
\cortext[cor1]{Corresponding author.}
\ead{satriamandala@telkomuniversity.ac.id}
\affiliation[humic]{organization={Humic Engineering (Humic Eng), School of Computing, Telkom University},
 addressline={Jl. Telekomunikasi no. 1},city={Bandung},postcode={40257},state={West Java},country={Indonesia}}
\begin{abstract}''',inline(abstract),r'''\end{abstract}
\begin{keyword}
IoT intrusion detection \sep endpoint context \sep weighted voting \sep early exit \sep SHAP \sep reproducibility
\end{keyword}
\end{frontmatter}
\noindent\textit{Research article draft for author and supervisor review, 21 October 2026.}
''']
    body=md.split('## 1 Introduction\n',1)[1]
    body='## 1 Introduction\n'+body
    body=body.split('## References',1)[0]
    for para in re.split(r'\n\s*\n',body.strip()):
        if para.startswith('## Appendix A'):
            tex.append(r'\clearpage\appendix\renewcommand{\thetable}{A\arabic{table}}\renewcommand{\theHtable}{A\arabic{table}}\setcounter{table}{0}\section{Reproducibility details}')
        elif para.startswith('### '):
            title=re.sub(r'^### (?:\d+\.\d+|A\.\d+)\s+','',para)
            tex.append(r'\subsection{'+inline(title)+'}')
        elif para.startswith('## '):
            title=re.sub(r'^## (?:\d+\s+)?','',para)
            star='' if re.match(r'^## \d',para) else '*'
            tex.append(r'\section'+star+'{'+inline(title)+'}')
        elif para.startswith('{{TABLE:'):
            key=para[8:-2];t=tables[key]
            caption=re.sub(r'^Table (?:\d+|A\d+)\.\s*','',t['caption'])
            n=len(t['headers']);width=t.get('widths') or [487/n]*n
            # Widths are fractions of line width minus intercolumn padding.
            total=sum(width);fractions=[x/total for x in width]
            col=''.join('L{'+f'{f:.6f}'+r'\dimexpr\textwidth-'+str(2*n)+r'\tabcolsep\relax}' for f in fractions)
            # All tables fit one page at this size; keep captions, rows and notes together.
            size=r'\footnotesize' if key in ['prior_work','stage_results'] else r'\small'
            tex.append(r'\begin{table}[!htbp]\centering'+size+r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}')
            tex.append(r'\caption{'+inline(caption)+r'}\label{tab:'+key+r'}')
            tex.append(r'\begin{tabular}{'+col+r'}\toprule')
            def head(x):
                return inline(x).replace('Ransomware',r'Ransom-\allowbreak{}ware').replace('Training',r'Train-\allowbreak{}ing')
            hdr=' & '.join(r'\textbf{'+head(x)+'}' for x in t['headers'])+r'\\\midrule'
            tex.append(hdr)
            for row in t['rows']:tex.append(' & '.join(inline(x) for x in row)+r'\\')
            tex.append(r'\bottomrule\end{tabular}')
            if t.get('note'):tex.append(r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize '+inline(t['note'])+r'\end{minipage}')
            tex.append(r'\end{table}')
        elif para.startswith('{{FIGURE:'):
            key,caption=para[9:-2].split('|',1)
            caption=re.sub(r'^Figure \d+\.\s*','',caption)
            tex.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/'+key+r'.pdf}\caption{'+inline(caption)+r'}\label{fig:'+key+r'}\end{figure}')
        elif para.startswith('{{EQUATION:'):
            key=para[11:-2];tex.append(r'\begin{equation}'+eqs[key]+r'\label{eq:'+key+r'}\end{equation}')
        else:tex.append(inline(para.replace('\n',' ')))
    tex += [r'\FloatBarrier',r'\bibliographystyle{elsarticle-num}',r'\bibliography{references}',r'\end{document}']
    (SRC/'main.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf8')
    highlights=[
      'Prior-only multiscale context improves within-trace IoT intrusion detection.',
      'Compact execution preserves tested expert probabilities and grouped SHAP games.',
      'Certified early exit returns full-voting labels without partial posteriors.',
      'Cross-background explanation audits separate faithfulness from invariance.',
      'Controls expose class trade-offs, memory costs and capture-validity limits.']
    assert all(len(s)<=85 for s in highlights)
    (SRC/'highlights.txt').write_text('\n'.join(highlights)+'\n')
    (SRC/'BUILD.bat').write_text('@echo off\ncd /d "%~dp0"\nwhere latexmk >nul 2>nul\nif errorlevel 1 (\n  pdflatex main.tex\n  bibtex main\n  pdflatex main.tex\n  pdflatex main.tex\n) else (\n  latexmk -pdf main.tex\n)\npause\n')
    print(json.dumps(dict(source=str(SRC),abstract_words=len(abstract.split()),references=len(refs),tables=len(tables),dashboard_gate=gate['status'],template_files=len(template_records)),indent=2))

if __name__=='__main__':main()
