"""Evidence-derived major revision; old manuscript and experiment runs are immutable."""
from pathlib import Path
import sys,json,re,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_array_factorial_20261103 import read_npz,methods
OLD=ROOT/'paper/array_20261022_integrated/elsarticle'
OUT=ROOT/'paper/array_20261103_major_revision/elsarticle';OUT.mkdir(parents=True,exist_ok=True)
REP=ROOT/'reports/ARRAY_MAJOR_REVISION_20261103';RUN=ROOT/'runs/ARRAY_RUNTIME_FULL_20261103_01'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
for name in ['elsarticle.cls','elsarticle-num.bst']:shutil.copy2(OLD/name,OUT/name)
for p in (OLD/'figures').glob('*'):shutil.copy2(p,FIG/p.name)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def num(x,d=4):return f'{float(x):.{d}f}'
def pct(x,d=4):return num(100*float(x),d)
def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(caption,label,headers,rows,note,widths=None):
    note=re.sub(r'(?<!\\)_',r'\\_',note)
    rows=[[re.sub(r'(?<!\\)_',r'\\_',str(x)) for x in row] for row in rows]
    headers=[h.replace('Ransom','Rans.') for h in headers]
    n=len(headers);widths=widths or [1/n]*n
    cols=''.join('L{'+str(round(w,6))+r'\dimexpr\textwidth-'+str(2*n)+r'\tabcolsep\relax}' for w in widths)
    return '\n'+r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}'+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}\n'+r'\begin{tabular}{'+cols+r'}\toprule'+'\n'+' & '.join(r'\textbf{'+h+'}' for h in headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\par\smallskip\begin{minipage}{\linewidth}\footnotesize '+note+r'\end{minipage}\end{table}'+'\n'
def figure(file,caption,label):return '\n'+r'\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/'+file+'}'+r'\caption{'+caption+'}'+r'\label{'+label+r'}\end{figure}'+'\n'
def replace_between(text,a,b,new):return text[:text.index(a)]+new+text[text.index(b):]

def bibliography():
    src=(OLD/'references.bib').read_text();audit={r['key']:r for r in json.loads((ROOT/'research/array_20261103/audit.json').read_text(encoding='utf8'))};blocks=[];report=[]
    for block in src.split('@misc{')[1:]:
        key=block.split(',')[0];r=audit[key];m=r.get('metadata');report.append(dict(key=key,status=r['status'],url=r['url'],doi=r['doi'],claim_scope='Conceptual precedent; no external published score treated as matched baseline'))
        if m:
            kind='article' if m['type']=='journal-article' else 'inproceedings' if m['type']=='proceedings-article' else 'incollection'
            title=m['title'][0]+(': '+m['subtitle'][0].strip() if m.get('subtitle') and m['subtitle'][0].strip() else '')
            authors=' and '.join(a.get('family','')+', '+a.get('given','') for a in m.get('author',[]))
            year=m.get('published',m.get('issued'))['date-parts'][0][0]
            fields=dict(author=authors,title='{'+title+'}',year=str(year),doi=m['DOI'])
            fields['journal' if kind=='article' else 'booktitle']=m.get('container-title',[''])[0]
            if key=='r2':fields['booktitle']='Detection of Intrusions and Malware, and Vulnerability Assessment';fields['series']='Lecture Notes in Computer Science';fields['volume']='13358'
            if key=='r24':fields['author']='{Helen Sharmila A.} and {Deepanramkumar P.}'
            if kind!='article':fields['publisher']=m.get('publisher','')
            for k,outkey in [('volume','volume'),('issue','number'),('page','pages'),('article-number','eid')]:
                if m.get(k):fields[outkey]=str(m[k]).replace('-','--') if k=='page' else str(m[k])
            blocks.append('@'+kind+'{'+key+',\n'+',\n'.join('  '+k+' = {'+v.replace('&',r'\&')+'}' for k,v in fields.items())+'\n}\n')
        else:
            # Verified primary HTML; retain supplied bibliographic wording for non-DOI items.
            blocks.append('@misc{'+block)
    (OUT/'references.bib').write_text('\n'.join(blocks).replace('\u2019',"'").replace('\u00a0',' '),encoding='utf8');pd.DataFrame(report).to_csv(REP/'reference_audit.csv',index=False)

def figures(f,history,rt,xai):
    # The flow explicitly separates exploratory predictive evidence from functional checks.
    fig,ax=plt.subplots(figsize=(8.8,5));ax.axis('off')
    boxes=[(.5,.90,'Five existing sources / one source per class\nReused development targets; historical holdout excluded'),(.25,.65,'Temporal A/G/C/V\nHistory resets at boundaries'),(.75,.65,'Signature diagnostic\nFull / shared-density / private history'),(.5,.39,'Matched 2 × 2 context / ordering controls\nFixed parameters, weights and target sets'),(.25,.12,'Fixed predictor execution\n30 paired replays; label/probability checks'),(.75,.12,'Full-probability explanation\n7-group archive + 9-group window audit')]
    for x,y,t in boxes:ax.text(x,y,t,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.65',fc='#eaf2f8',ec='#24516e'),transform=ax.transAxes)
    for start,end in [((.5,.82),(.25,.73)),((.5,.82),(.75,.73)),((.25,.56),(.5,.47)),((.75,.56),(.5,.47)),((.5,.30),(.25,.21)),((.5,.30),(.75,.21))]:ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#24516e'))
    fig.savefig(FIG/'revision_flow.pdf',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    labels=['C40/O1','C40/O10','C60/O1','C60/O10']
    for ax,protocol in zip(axs,['temporal','signature']):
        g=f[(f.protocol==protocol)&(f.method=='weighted_voting')].set_index('view').loc[['c40_o1','c40_o10','c60_o1','c60_o10']]
        ax.bar(np.arange(4),g.macro_f1*100,color=['#8798a4','#aebac2','#236b8e','#50a6a0']);ax.set_ylim(65 if protocol=='signature' else 95,100);ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_ylabel('Macro-F1 (%)');ax.set_title('Temporal: three-seed mean' if protocol=='temporal' else 'Signature: pooled diagnostic')
        if protocol=='temporal':ax.errorbar(range(4),g.macro_f1*100,yerr=g.macro_f1_sd*100,fmt='none',ecolor='black',capsize=3)
        for i,v in enumerate(g.macro_f1*100):ax.text(i,v+.25,f'{v:.3f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(FIG/'revision_factorial.pdf',bbox_inches='tight');plt.close(fig)
    full=f.query("protocol=='signature' and view=='c60_o10' and method=='weighted_voting'").iloc[0];h=history[history.method=='weighted_voting'].set_index('history')
    fig,ax=plt.subplots(figsize=(8.8,3.8));xx=np.arange(5)
    cols=['recall_Normal','recall_DDoS_TCP','recall_Port_Scanning','recall_SQL_injection','recall_Ransomware']
    for i,(name,row,color) in enumerate([('Full shared history',full,'#245b80'),('Shared-density control',h.loc['shared_density'],'#cf8d49'),('Packet-private control',h.loc['exclusive'],'#51a29b')]):ax.bar(xx+(i-1)*.25,row[cols].astype(float)*100,.25,label=name,color=color)
    ax.set_xticks(xx,['Normal','DDoS','Scanning','SQL injection','Ransomware']);ax.set_ylabel('Recall (%)');ax.set_ylim(0,107);ax.legend(loc='lower left',ncol=1,fontsize=9);fig.tight_layout();fig.savefig(FIG/'revision_history.pdf',bbox_inches='tight');plt.close(fig)
    piv=rt.pivot(index='repeat',columns='mode',values='total_seconds');fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    for mode,label,color in [('native_certified','Native certified','#a46539'),('compact_certified','Compact certified','#216f94')]:axs[0].plot(piv.index+1,piv[mode],'.-',lw=.8,label=label,color=color)
    axs[0].set(xlabel='Paired repetition',ylabel='Replay time (s)');axs[0].legend(fontsize=9)
    delta=piv.native_certified-piv.compact_certified;axs[1].bar(piv.index+1,delta,color=np.where(delta>0,'#216f94','#b35f4a'));axs[1].axhline(0,color='black',lw=.8);axs[1].set(xlabel='Paired repetition',ylabel='Native − compact (s)');fig.tight_layout();fig.savefig(FIG/'revision_runtime.pdf',bbox_inches='tight');plt.close(fig)
    g=xai.query("background=='A32' and method=='weighted_voting'").set_index('group');order=['history_100ms','history_1000ms','history_10000ms'];fig,ax=plt.subplots(figsize=(8,3.3));values=[100*g.loc[k,'share'] for k in order]+[100*g.loc[~g.index.str.startswith('history'),'share'].sum()]
    ax.barh(['0.1 s history (20)','1 s history (20)','10 s history (20)','Six protocol groups'],values,color=['#78a8b8','#3e8a9f','#245b80','#a6acaf']);ax.invert_yaxis();ax.set_xlabel('Share of mean absolute attribution (%)');ax.set_xlim(0,max(values)*1.18)
    for i,v in enumerate(values):ax.text(v+.4,i,f'{v:.2f}%',va='center')
    fig.tight_layout();fig.savefig(FIG/'revision_xai.pdf',bbox_inches='tight');plt.close(fig)

def build():
    bibliography()
    f=pd.read_csv(REP/'factorial_summary.csv');h=pd.read_csv(REP/'history_summary.csv');b=pd.read_csv(REP/'buffer_summary.csv');d=pd.read_csv(REP/'correctness_disagreements.csv')
    rt=pd.read_csv(RUN/'replay_raw.csv');lat=pd.read_csv(RUN/'latency_raw.csv');mem=pd.read_csv(RUN/'memory_raw.csv');pair=json.loads((RUN/'paired_summary.json').read_text())
    xa=pd.read_csv(REP/'xai_faithfulness.csv');xs=pd.read_csv(REP/'xai_sensitivity.csv');xi=pd.read_csv(REP/'xai_shares.csv')
    benign=pd.read_csv(ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04/metrics.csv');bt=benign.groupby('method')[['n','eligible','unscorable','false_alarms']].sum();bw=bt.loc['weighted_voting']
    figures(f,h,rt,xi)
    tw=f.query("protocol=='temporal' and method=='weighted_voting'").set_index('view');sw=f.query("protocol=='signature' and method=='weighted_voting'").set_index('view');hw=h[h.method=='weighted_voting'].set_index('history')
    rr=rt.groupby('mode').total_seconds.agg(['mean','median','std']);speed=100*pair['mean_relative_reduction'];lo,hi=pair['ci95_mean_seconds']
    methodname={'weighted_voting':'Weighted voting','uniform_voting':'Uniform voting','random_forest':'Random forest','logistic_regression':'Logistic regression'}
    factorial=table('Matched context/ordering ablation with fixed per-stage reference weights.','tab:factorial',['Context / order','Temporal F1','Temporal BA','Mean errors','Signature F1','Signature errors'],[[v.replace('_',' / ').upper(),pct(tw.loc[v,'macro_f1']),pct(tw.loc[v,'balanced_accuracy']),num(tw.loc[v,'errors'],2),pct(sw.loc[v,'macro_f1']),int(sw.loc[v,'errors'])] for v in tw.index],r'F1 and balanced accuracy (BA) are percentages. Temporal: three estimator seeds on the same 10,358 targets. Signature: pooled two-fold diagnostic, 52,584 targets. C40 contains 0.1/1 s statistics; C60 adds 10 s. O1/O10 select the ordering window. Full per-seed metrics and all recalls appear in Supplementary Tables S25--S26.',[.20,.16,.16,.15,.17,.16])
    hist=table('Signature history controls for the unchanged modelling budget and reference weights.','tab:history',['History','F1','BA','Scan recall','Ransom recall','Errors'],[[name,pct(row.macro_f1),pct(row.balanced_accuracy),pct(row.recall_Port_Scanning),pct(row.recall_Ransomware),int(row.errors)] for name,row in [('Full shared',sw.loc['c60_o10']),('Shared-density',hw.loc['shared_density']),('Packet-private',hw.loc['exclusive'])]],r'All percentages use the same 52,584 targets. Each control refits the five experts, without retuning. The two thinning controls have matched marginal allocation probabilities, not identical realized packet histories. No control separates capture identity from class.',[.25,.14,.14,.16,.17,.14])
    ens=f.query("protocol=='temporal' and view=='c60_o10'").set_index('method')
    ensemble=table('Aggregation and singleton controls on identical temporal targets.','tab:ensemble',['Predictor','F1','BA','DDoS F2','Errors','Ransom recall'],[[methodname[m],pct(row.macro_f1),pct(row.balanced_accuracy),pct(row.ddos_f2),num(row.errors,2),pct(row.recall_Ransomware)] for m,row in ens.iterrows()],r'Three-seed means; all use the same feature representation, split and fitted experts. Mean error counts can be fractional. Full per-class results and signature controls are in Supplementary Tables S27--S28.',[.25,.15,.15,.15,.14,.16])
    buffers=[]
    for cap in [4096,16384,65536]:
        for protocol in ['temporal','signature']:
            row=b.query('capacity==@cap and protocol==@protocol and cohort=="warm1" and method=="weighted_voting"').iloc[0]
            buffers.append([num(cap/16384,2),protocol.title(),pct(row.macro_f1),pct(row.recall_Port_Scanning),pct(row.recall_Ransomware),num(row.errors,2)])
    buffertable=table('Frozen-predictor sensitivity to ring capacity on the original eligible targets.','tab:buffer',['Array MiB','Evaluation','F1','Scan recall','Ransom recall','Errors'],buffers,r'Capacities 4,096/16,384/65,536 records; 8 int64 fields per record. No new classifier is fitted for this analysis. Percentages and temporal three-seed means. Ring-array size is distinct from process delta RSS.',[.14,.20,.16,.17,.18,.15])
    latency=lat.groupby(['policy','batch']).seconds.median();memory=mem.groupby(['policy','workload']).delta_mib.mean();runtime=[]
    names={'native_shared':'Native weighted full','uniform_voting':'Native uniform full','random_forest':'Native RF','logistic_regression':'Native LR','native_certified':'Native certified','compact_certified':'Compact certified'}
    for mode,name in names.items():runtime.append([name,num(latency[mode,1]*1000,3),num(latency[mode,1000]*1000,3),num(memory[mode,'feature'],3),num(rr.loc[mode,'mean'],4) if mode in rr.index else '--',num(memory[mode,'replay'],3) if (mode,'replay') in memory.index else '--'])
    runtable=table('Fresh single-session serving costs for the fixed seed-42 predictor and controls.','tab:runtime',['Policy','1 row ms','1,000 rows ms','Feature delta MiB','Replay s','Replay delta MiB'],runtime,r'Batch latency is the median of 30 calls after five warm-ups. Replay is the mean of 30 paired calls. RSS is the mean of five cold processes per policy/workload, sampled every 2 ms after load. Dashes mean not measured in this new replay session.',[.25,.13,.16,.17,.13,.16])
    faith=table('Nine-group, cross-background explanation ranking on 100 targets.','tab:faith',['Predictor','Deletion AUC','Insertion AUC','Deletion gain','Insertion gain'],[[methodname[row.method],num(row.deletion_auc),num(row.insertion_auc),num(row.deletion_gain),num(row.insertion_gain)] for row in xa.itertuples()],r'Signed rankings from A32; donor interventions from disjoint B32. Gains are probability-area differences relative to the exact random-order expectation. Positive gains are not causal accuracy or a claim of superior explainability.',[.28,.18,.18,.18,.18])
    main=(OLD/'main.tex').read_text()
    main=main.replace(r'\author[humic]{M Hafid Ramadhan}',r'\author[humic]{M Hafid Ramadhan}'+'\n'+r'\ead{hafid1050677@gmail.com}')
    abstract=f'''\\begin{{abstract}}Traffic context can improve an IoT intrusion detector while also exposing it to capture-specific shortcuts. We study bounded prior-only endpoint statistics, execution of a fixed weighted ensemble, and auditable full-probability explanations on five Edge-IIoTset sources. A matched four-cell ablation separates adding a 10-second window from changing endpoint ordering. With one-second ordering held fixed, temporal macro-F1 increases from {pct(tw.loc['c40_o1','macro_f1'],3)}\\% to {pct(tw.loc['c60_o1','macro_f1'],3)}\\%; the ordering-only change is much smaller. Signature macro-F1 falls from {pct(sw.loc['c60_o10','macro_f1'],3)}\\% to {pct(hw.loc['exclusive','macro_f1'],3)}\\% with packet-private history, while a shared-history control of similar marginal density yields {pct(hw.loc['shared_density','macro_f1'],3)}\\%. Singletons and uniform voting remain competitive. Compact execution preserves tested probabilities and labels, reducing mean replay time by {speed:.2f}\\% across 30 paired repetitions on one laptop. Exact grouped Shapley analysis separates the three history windows and measures donor-background sensitivity and removal faithfulness. These results support a reproducible representation-and-execution study within observed traces. One source per class and adaptive development reuse preclude claims of independent-traffic generalization.\\end{{abstract}}

'''
    main=replace_between(main,r'\begin{abstract}',r'\begin{keyword}',abstract)
    main=main.replace('Singletons and uniform voting remain competitive.',f"Singletons and uniform voting remain competitive. A coverage-limited audit of additional benign captures yields {int(bw.false_alarms)}/{int(bw.n)} false alarms for weighted voting, exposing a serious source-transfer limitation.")
    main=main.replace('Research article draft for author review, 22 October 2026. Initial and extension experiments retain distinct evaluation scopes. Technical details are provided as separate supplementary material.','Major-revision research draft for author review, 3 November 2026. All predictive evaluations are exploratory within-trace analyses; no independent final test is claimed.')
    main=re.sub(r'\\noindent\\textit\{Major-revision research draft[^}]+\}',r'\\noindent{\\footnotesize Author-review draft, 3 November 2026.}',main)
    main=main.replace('whether a bounded history of both communicating endpoints can repair weaknesses of a shorter-context representation','how much a bounded history of both communicating endpoints contributes after controlling endpoint ordering')
    main=main.replace('Quantitative IDS explanation evaluation also precedes this work.',r'Time-based traffic descriptors also have established uses in traffic classification~\cite{r20}; our archived timing controls are not presented as the invention of temporal features. Quantitative IDS explanation evaluation also precedes this work.')
    main=main.replace('grouped IoT SHAP and federated IDS studies examine faithfulness and robustness \\cite{r11,r12}',r'grouped IoT SHAP examines grouping and robustness \cite{r11}, while federated IDS work measures faithfulness and robustness \cite{r12}')
    main=main.replace('Representation, aggregation and runtime controls separate these questions.','The representation contribution is tested by a matched context/ordering factorial and history-ownership controls; the execution contribution by same-predictor equivalence and paired timing; the explanation contribution by full-probability consistency and explicit donor-game audits. Endpoint statistics, weighted voting, early exit and SHAP are established techniques, not newly invented algorithms. Representation, aggregation and runtime controls separate these questions.')
    initial=r'''\subsection{Historical selection and subsequent reuse}

The original study used nested development folds to compare 28/29 port representations and six Original/Balanced ensembles. Balanced denotes training-only RUS followed by SMOTE, with a synthetic cap defined against each pre-RUS class count. The selected 29-field configurations were evaluated on a historical holdout of 13,171 observations. A performance--efficiency ranking subsequently removed an Original-priority rule on 15 October, after the holdout had been observed. Balanced weighted voting ranked first under that revised composite, while four configurations produced identical holdout labels. This history motivates the fixed predictor studied here; it does not establish its classification superiority. Full historical tables and operational weights remain in Supplementary Tables S22--S24 and the provenance package.

These initial results and port-dependence diagnostics overlap a related AGERS conference manuscript, \textit{Port Dependence and Cost Trade-offs of Ensemble Intrusion Detection for Industrial IoT}, submitted by the authors through EDAS (paper 1571374892). The present work adds the packet-history representation controls, fixed-predictor execution and quantitative explanation investigations. The related submission must be disclosed to the editor; overlapping initial results are not presented as a new independent experiment.

The October context, ordering and timing investigations followed inspection of earlier development outcomes. The 3 November protocol froze the additional controls before their execution, but it could not make the reused development data unseen. Thus every predictive score in this article is exploratory. No data in this revision qualify as an untouched final test, and the new controls are not used to choose a replacement winner. Supplementary Table S34 records post-observation decisions and their scope.

'''
    main=replace_between(main,r'\subsection{Initial feature, tuning and ensemble comparison}',r'\subsection{Development protocols for the extensions}',initial)
    oldfig=re.search(r'\\begin\{figure\}\[H\].*?\\end\{figure\}',main,re.S).group()
    main=main.replace(oldfig,figure('revision_flow.pdf',r'Evaluation flow. Predictive controls reuse development targets; functional equivalence is a separate software property. Only full probabilities feed the explanation branch. OpenAI Codex (OpenAI; exact application build not recorded) assisted the programmatic diagram code.','fig:flow'))
    main=main.replace(r'\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/revision_flow.pdf}',r'\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/revision_flow.pdf}')
    main=main.replace('Figure 1 connects',r'Figure~\ref{fig:flow} connects')
    main=main.replace('Thus the comparison changes both dimension and ordering, rather than isolating the ten-second window alone.',r'The original comparison changed dimension and ordering together. We now cross context40/context60 with O1/O10, holding targets, seeds, expert settings, treatment and the recorded C60/O10 weights fixed within each stage. O1 uses the ten 1-second statistics, then the two-window vector, as the lexicographic key; O10 uses the ten 10-second statistics, then the full vector. C40/O10 intentionally uses 10-second information only to order the two slots. This factorial separates adding the emitted 10-second statistics from changing the ordering rule. Cached corners are reused only after exact feature and split identity checks; the two crossed conditions require 50 new expert fits across three temporal seeds and two signature folds.')
    main=main.replace('For any rival b, the full margin is at least',r'Here $\tau=10^{-12}$ is a fixed, nonnegative stopping tolerance, and $\eta=64\epsilon_{64}\max(1,M)\max(1,\sum_jw_j)$ is the implementation guard, with $M=5$ and $\epsilon_{64}=2^{-52}$. For normalized weights, $\eta\simeq7.11\times10^{-14}$. These are arithmetic safeguards, not estimated confidence levels. A tie or unresolved margin triggers canonical full summation with the same class-order tie rule. For any rival b, the full margin is at least')
    main=main.replace('all sixty history fields form one atomic group.','the archived audit treats all sixty history fields as one atomic group. The revised audit instead uses three twenty-field groups for 0.1, 1 and 10 s, giving nine groups in total. It retains exactly the same 100 targets, predictor and two backgrounds. The grouping changes the cooperative game, so seven- and nine-group shares and AUCs are different estimands.')
    main=main.replace('All 128 coalitions are enumerated, then exact Shapley values are computed.','All 512 coalitions are enumerated in each revised game (128 in the archived seven-group audit), then exact Shapley values are computed.')
    main=main.replace('Finally, all 300 saved background/delay games are re-executed through compact full prediction:', 'In the archived seven-group equivalence experiment, all 300 background/delay games were re-executed through compact full prediction:')
    measurement=r'''\section{Experimental measurements}

Predictive outcomes are macro-F1, balanced accuracy (mean class recall), DDoS-positive F2, all five recalls and total/per-class errors. Temporal means and sample SDs describe three estimator seeds on the same V targets. Signature metrics pool the two validation sides, with per-fold results retained. Estimator variation is not uncertainty over independent traffic sources. Paired correctness tables retain examples corrected by the ensemble and examples lost relative to RF, LR or uniform voting; no hypothesis of ensemble superiority is imposed.

\subsection{History ownership and capture dependence}

Because each class has one source, capture identity and class are perfectly confounded. Endpoint activity, peer/port cardinalities and temporal density can encode the recording environment. Excluding raw identifiers does not prove that these statistics are attack-specific. A classifier trained to predict source would not resolve this confounding. We instead preserve context summaries by source and temporal role and test sensitivity to history ownership. No matched independent capture with the same task and provenance has yet been established.

For each fixed signature fold, mapped eligible target packets belong exclusively to their A/G/C/V role. Other packet records receive deterministic assignments. Packet-private history assigns each record to exactly one ring, with role probabilities equal to its target fraction. A shared-density control uses the same uniform random draw and includes the packet in each role whose fraction exceeds that draw. This allows overlaps while matching the marginal inclusion probabilities. Both restrict mapped targets to their own role, retain the full-stream monotonic clock and original temporal reset boundaries, and query before insertion. Every role has a separate ring. The private control verifies zero shared history records across supervised roles. These two controls require 20 new expert fits with the fixed hyperparameters and stage weights.

Thinning changes density, warm history and the observed feature distribution; it is not a new real session. Its shared-density counterpart helps interpret the change but does not identify a pure causal leakage effect. Source/class confounding persists even when every history record has one owner. Historical holdout labels and targets remain excluded from supervised fitting/evaluation, although unmapped history packets can include traffic associated with that earlier holdout.

\subsection{Ring capacity, warm-up and serving measurements}

We reconstruct 4,096-, 16,384- and 65,536-record rings (0.25, 1 and 4 MiB arrays), then score the same targets with each already fitted reference predictor. This is input sensitivity without refitting or model selection. We record source-level capacity evictions and whether a target has lost a record that still belongs to its nominal 10-second window. A 10-second warm-up is compared on the common eligible targets; it removes decisions rather than changing their retained input. Therefore common-target predictions are identical for 1/10-second purge policies at a fixed capacity. Scores on all versus retained targets describe cohort changes, not a causal warm-up gain. Queries scan retained records; constant-time context acquisition is not claimed.

Fresh runtime measurements use one Windows laptop and one numerical-library thread after other experiment jobs finish. Each policy receives five warm-ups. Feature-input batches of 1/1,000 use 30 randomized calls. Native and compact certified execution use 30 randomized paired replays of the same 15,665 packets and 160 annotated decisions. We retain wall time, process CPU time and component times. A fixed-seed, 10,000-resample paired bootstrap gives a percentile interval for the mean native-minus-compact difference. This interval describes repetition variation on one workload/machine. Repeating the recorded segments 1/2/4 times tests workload length with fresh segment histories; it does not create independent traffic or independently vary active endpoints.

Five cold processes per policy/workload sample RSS every 2 ms. Inference delta peak RSS subtracts the post-load baseline; replay includes the ring. Full weighted voting, uniform voting, RF and LR use the same feature-input rows. The optimized ensemble and native singletons do not have equal implementation optimization. Replay includes decoding, updates, queries, frame assembly and prediction, but uses canonical legacy fields instead of reconstructing all 25 fields online. It is not an every-packet or embedded-hardware throughput benchmark. Buffer query/update timing uses ten repeats of the same replay at each capacity and excludes model calls.

The environment, source hashes, full split identifiers, seeds, resampling counts, pre-evaluation model hashes, saved predictions, coalition values and raw timing samples are retained. New revision experiments comprise 70 expert fits; separate smoke fits are not scientific results. Figures are programmatic transformations of retained results, with OpenAI Codex assisting their code and manuscript preparation. The historical seven-group implementation-equivalence experiment remains distinct from the new nine-group attribution analysis.

'''
    main=replace_between(main,r'\section{Experimental measurements}',r'\section{Results and interpretation}',measurement)
    benign_method=r'''\subsection{Additional benign captures: coverage-limited stress test}

A user-requested local inventory found nine Normal captures beyond the Temperature/Humidity source, but no second matched source for each of the four target attack classes. A separate addendum fixes the existing temporal seed-42 model, all nine sources, the first 100,000 packets per source, and candidate packet indices 999, 1999, \ldots, 99999. A 10-second warm-up excludes 33 candidates, leaving 867. Each source starts with an empty 1 MiB ring; only prior packets from that source supply context. CSV size/CRC32 and PCAP SHA256 are checked against the archived dataset audit.

Preliminary alignment gates reveal that CSV row indices do not reliably equal packet indices. Exact matching uses time modulo the hour, nanoseconds, both IP addresses, TCP checksum, both TCP ports and payload length; unmatched candidates receive no features or predictions. The Modbus CSV clock field contains IP addresses, making all its candidates unscorable. Overall, 760 candidates from eight sources match exactly; 107 eligible candidates remain unscorable. Earlier aborted attempts and the post-observation amendments are retained. This is a coverage-limited, same-corpus benign stress test, not an untouched five-class final evaluation. No source, model or parameter is selected from its outcomes. We report conditional false-positive rates on matched rows and bounds over all eligible rows, assigning missing outcomes neither benign nor attack predictions.

'''
    main=main.replace(r'\section{Results and interpretation}',benign_method+r'\section{Results and interpretation}')
    delta1=100*(tw.loc['c60_o1','macro_f1']-tw.loc['c40_o1','macro_f1']);delta10=100*(tw.loc['c60_o10','macro_f1']-tw.loc['c40_o10','macro_f1']);order40=100*(tw.loc['c40_o10','macro_f1']-tw.loc['c40_o1','macro_f1']);order60=100*(tw.loc['c60_o10','macro_f1']-tw.loc['c60_o1','macro_f1'])
    results=r'\section{Results and interpretation}'+'\n\n'+r'\subsection{Separating context length from endpoint ordering}'+'\n\n'
    results+=f'''Table~\\ref{{tab:factorial}} and Figure~\\ref{{fig:factorial}} isolate the representation contrast. Adding the 10-second statistics improves temporal macro-F1 by {delta1:.4f} percentage points under O1 and {delta10:.4f} points under O10. Changing only ordering gives {order40:+.4f} points for C40 and {order60:+.4f} for C60. Thus the gain persists under either fixed ordering rule; it cannot be attributed only to the ordering change. This is conditional evidence within these traces, not proof that the added window generalizes to a new capture. The revised C40/O1 value differs slightly from the earlier 96.7392\\% because the factorial holds the C60/O10 weights fixed instead of reusing each representation's own weights.

Under O1, mean scanning errors fall from 83 to 61 and ransomware errors from 147 to 82 when C40 becomes C60. Under O10, the respective changes are 83 to 61.33 and 145.67 to 82. These are three-seed means on the same target set, not new independent attack trials.

'''+factorial+figure('revision_factorial.pdf',r'Matched four-cell ablation. Error bars are temporal sample SDs across three seeds on identical targets; signature bars pool two diagnostic folds. The panels have different lower axis limits to retain readable differences.','fig:factorial')
    results+=f'''For signature grouping, C60/O1 reaches {pct(sw.loc['c60_o1','macro_f1'])}\\%, compared with {pct(sw.loc['c60_o10','macro_f1'])}\\% for C60/O10. The ten-second ordering is therefore not universally better. The original C60/O10 predictor remains fixed for the execution and explanation study; the higher diagnostic score is not used to choose a replacement. Per-class results show why aggregate scores need care: the large improvement in scanning coexists with lower ransomware recall in the historical short-to-long signature comparison. Earlier short-context ransomware recall was 93.0936\\%, versus 92.7042\\% for the retained long-context reference. Additional timing/directional candidates and their negative results remain in Supplementary Tables S6--S9.

\\subsection{{What history controls do and do not establish}}

'''+hist+figure('revision_history.pdf',r'Per-class recall on the same signature diagnostic targets. Both thinned controls change the available traffic history; neither supplies an independent capture. Ransomware and scanning account for most of the deterioration.','fig:history')
    results+=f'''Packet-private history lowers weighted-voting macro-F1 by {100*(sw.loc['c60_o10','macro_f1']-hw.loc['exclusive','macro_f1']):.4f} percentage points relative to full history. Scanning recall changes from {pct(sw.loc['c60_o10','recall_Port_Scanning'])}\\% to {pct(hw.loc['exclusive','recall_Port_Scanning'])}\\%, and ransomware recall from {pct(sw.loc['c60_o10','recall_Ransomware'])}\\% to {pct(hw.loc['exclusive','recall_Ransomware'])}\\%. Figure~\\ref{{fig:history}} shows all five classes. The shared-density control is slightly worse than the packet-private control, by {100*(hw.loc['exclusive','macro_f1']-hw.loc['shared_density','macro_f1']):.4f} F1 points. This prevents attributing the entire full-to-private drop specifically to cross-partition sharing. History density and membership matter, and the old 97.2955\\% signature score is insufficient evidence of independent-traffic generalization. Fold-level differences and all singleton controls are retained in Supplementary Tables S28--S29.

\\subsection{{Does weighted voting add classification value?}}

'''+ensemble
    ed=d[(d.stage=='temporal_seed42')&(d.true_class==-1)].set_index('comparator')
    results+=f'''Uniform voting has slightly higher mean F1 than weighted voting, and RF/LR remain close. On the seed-42 targets, weighted voting corrects {int(ed.loc['random_forest','ensemble_only_correct'])} RF errors but loses {int(ed.loc['random_forest','comparator_only_correct'])} targets that RF classifies correctly. Against LR, the corresponding counts are {int(ed.loc['logistic_regression','ensemble_only_correct'])} and {int(ed.loc['logistic_regression','comparator_only_correct'])}. These paired observations distinguish different errors from a simple score tie. The full target identifiers, true/predicted classes and both directions of disagreement are saved; Supplementary Table S30 summarizes them by class. Repeated temporal seeds measure fitting variation on the same targets, not robustness across new sessions.

The evidence does not justify a general weighted-ensemble classification advantage. We retain weighted voting as the fixed predictor whose execution is optimized, rather than as a claimed superior detector. In particular, native singleton batch costs remain lower, and the private-history diagnostic exposes additional differences between weighted voting and RF. No new evaluation result is used to retune voting weights or promote a different winner.

\\subsection{{Capacity and warm-up sensitivity}}

'''+buffertable
    results+=r'''All three capacities retain the same temporal weighted-voting labels in this evaluation. Signature scores change only slightly, but that does not mean every nominal window is complete. The 1 MiB DDoS ring records 1,954,616 capacity evictions. Once more eligible records arrive than the array can retain, nominal 10-second context becomes a censored recent subset. The smaller array increases censoring; the larger array reduces it without proving that capacity can never bind.

A 10-second warm-up retains 50,964 of 52,584 eligible development targets and 10,078 of 10,358 temporal V targets. At fixed capacity, the common-target predictions are identical under either purge policy. The slightly changed score after exclusion is therefore a cohort effect. Supplementary Tables S31--S32 report class supports, censored-target proportions, full versus retained scores and query/update time. The replay is small and cannot establish sensitivity to arbitrary active-endpoint counts.

\subsection{Same-predictor execution and measured cost}

The existing equivalence audit checks all 52,684 development inputs: encoded values and CSR structure match, maximum expert-probability difference is zero, and full/certified labels agree. Early exit uses an average of 3.0715 experts for the seed-42 reference. Compact full prediction preserves the posterior used for explanations; certified-only partial scores are not substituted for it.

'''+runtable+figure('revision_runtime.pdf',r'Thirty paired replay measurements, with the order of native/compact execution randomized within each pair. Positive differences favor compact execution. These are repeated measurements of one fixed workload on one laptop.','fig:runtime')
    results+=f'''Figure~\\ref{{fig:runtime}} reports every pair. Native and compact certified mean times are {rr.loc['native_certified','mean']:.4f} and {rr.loc['compact_certified','mean']:.4f} s; medians are {rr.loc['native_certified','median']:.4f} and {rr.loc['compact_certified','median']:.4f} s, and sample SDs are {rr.loc['native_certified','std']:.4f} and {rr.loc['compact_certified','std']:.4f} s. The mean reduction is {speed:.2f}\\%. The paired native-minus-compact mean is {pair['mean_seconds']:.4f} s, with a descriptive bootstrap 95\\% interval [{lo:.4f}, {hi:.4f}] s; the paired median is {pair['median_seconds']:.4f} s. Compact execution is slower in {pair['compact_slower']} of the 30 pairs. Raw component and CPU times support inspection of variation, but no claim of universally faster execution follows.

The earlier five-repeat experiment included one slower compact repetition. Its record does not contain sufficient operating-system telemetry to identify a cause, so attributing it to scheduling, thermal throttling or another process would be speculation. The new repeated session supplements, rather than retroactively explains, that result. Workload-length measurements and all RSS replicates are supplementary. Resource comparisons concern these implementations; they do not establish architectural superiority over equally optimized singletons or uniform memory savings.

\\subsection{{Window-level explanation and background sensitivity}}

'''+faith+figure('revision_xai.pdf',r'Nine-group weighted-voting attribution shares for the A32 donor game. Each history window contains twenty fields; the six protocol groups are summed only for display. The bars describe mean absolute attribution on the 100-target audit, not individual-feature or causal importance.','fig:xai')
    wx=xs.set_index('method').loc['weighted_voting']
    results+=f'''Figure~\\ref{{fig:xai}} replaces the interpretation of one sixty-field history block with three window groups. It allows a comparison between equally sized history groups while leaving dependencies among windows explicit. Even a large window share does not imply that all twenty fields are individually useful. Cross-background top-three Jaccard for weighted voting is {wx.top3_jaccard:.4f}, with normalized L1 distance {wx.normalized_l1:.6f}. The archived seven-group Jaccard was 0.5520 despite cosine similarity 0.999948: dominant attribution magnitude could remain similar while secondary group rankings changed. Background choice therefore matters, and a stable label or high cosine similarity is not a background-invariant explanation.

Deletion/insertion gains remain evaluations of the specified empirical donor game. Grouping changes the intervention and the AUC horizontal axis, so seven- and nine-group gains are not directly interchangeable performance improvements. RF and LR also have positive gains; weighted voting is not claimed to be more explainable. The archived 300-game compact/native audit preserves full-probability coalition values and attributions. The new nine-group audit separately checks additivity, weighted-expert linearity and agreement with the exact library explainer. These checks support auditable explanation consistency, not a new SHAP algorithm or causal validity.

'''
    benign_results=r'''\subsection{False alarms on additional benign sources}

'''+table('Additional-capture benign diagnostic: all methods use the same 760 matched candidates.','tab:benign',['Predictor','False alarms','Matched FPR (percent)','All-eligible FPR bounds'],[[methodname[m],int(row.false_alarms),pct(row.false_alarms/row.n,2),'['+pct(row.false_alarms/row.eligible,2)+', '+pct((row.false_alarms+row.unscorable)/row.eligible,2)+']'] for m,row in bt.iterrows()],r'867 candidates remain after warm-up; 107 lack an exact CSV match. Bounds treat every missing outcome as either non-alarm or alarm and are not confidence intervals. All source-specific counts appear in Supplementary Table S37. No five-class F1 is defined here.',[.29,.18,.24,.29])+r'''
The transfer failure is substantial: weighted voting raises 324 false alarms among 760 exactly matched benign candidates (42.63\%). Rates differ sharply by source, from zero on several matched subsets to 88.04\% on Sound Sensor and 82.80\% on Distance. RF raises no false alarms on these matched rows, whereas uniform voting and LR raise 299 and 417. This negative control is inconsistent with a claim of reliable source-independent weighted-ensemble detection. It also shows that small differences on the original V set can hide much larger behavior changes across benign sources.

The zero RF count is not proof of a generally superior IDS: this diagnostic has no new attack capture, incomplete CSV coverage, correlated targets and only a limited source prefix. The comparison is not used to replace the fixed reference model. Instead, it strengthens the case for separating the execution contribution from classifier generalization and for testing a completely frozen pipeline on properly aligned independent sessions.

'''
    results=results.replace(r'\subsection{Capacity and warm-up sensitivity}',benign_results+r'\subsection{Capacity and warm-up sensitivity}')
    results=results.replace('Background choice therefore matters, and a stable label or high cosine similarity is not a background-invariant explanation.','All three history windows occupy the top-three set for weighted voting in both revised backgrounds; Jaccard 1.0 therefore concerns set membership, not invariant ordering or magnitude. Background choice still matters, and a stable label or high cosine similarity is not a background-invariant explanation.')
    main=replace_between(main,r'\section{Results and interpretation}',r'\section{Prototype and reproducibility}',results)
    discussion=r'''\section{Discussion and limitations}

RQ1 is supported by a narrower result than the original combined comparison: adding the 10-second statistics helps under either fixed ordering rule on the observed temporal targets. The signature controls show sensitivity to history density and ownership, while one source per class leaves attack behavior inseparable from recording conditions. Neither signature separation nor packet-private context establishes independent-capture generalization. A new source/session with a frozen feature pipeline, model and decision rule is still needed for that claim.

RQ2 is answered at implementation level. Repeated paired measurements show the cost of executing the same fitted predictor, with all observed repetitions and resource variation retained. The certificate has a clear full-prediction fallback, and compact encoding retains sparse semantics. A faster implementation of this ensemble need not beat a simpler, equally optimized classifier. The replay has only 160 annotated decisions, obtains legacy fields from a table and tests one laptop; packet-rate saturation, energy consumption and embedded deployment remain unmeasured.

RQ3 is answered for a fully specified donor game. Splitting history by window makes the explanation more interpretable without turning attribution into a causal analysis. Donor sensitivity, correlated windows, nonphysical masking combinations and unequal protocol-group sizes limit interpretation. The old query-delay audit changed 33/100 inputs with labels retained; it is a specific perturbation test, not evidence of broad adversarial robustness. Full-probability implementation equivalence and explanation quality are separate properties.

The additional protocol was frozen before new scoring, but all predictive evidence remains adaptive reuse of development data. The historical holdout also influenced the earlier recommendation policy. These limitations cannot be removed by renaming a split or by quoting a narrow repeat interval. Encoded-space SMOTE and synthetic packet/history hybrids introduce further modelling assumptions. The revised study therefore offers an auditable within-trace representation-and-execution analysis, with adverse outcomes and competitive singleton alternatives retained. It does not establish independent operational detection performance.

\section{Conclusion}

The matched ablation separates the benefit of emitted multiscale statistics from endpoint ordering on the available traces. History-ownership controls materially reduce the signature score and qualify its interpretation, while uniform voting and singletons remain competitive. Repeated serving measurements quantify an implementation-specific gain for the fixed predictor, with tested label/probability preservation. Window-level grouped Shapley analysis adds a reproducible account of attribution and donor sensitivity. The three contributions are representation controls, prediction-preserving execution and auditable explanation consistency; independent-capture evaluation remains necessary before broader deployment claims.

\section*{Data and software availability}

Edge-IIoTset is described in Ref.~\cite{r1}. The local review package contains source, configuration, split identifiers, derived results and checksums. Raw data availability follows the original dataset terms; redistribution permissions require verification. No public repository URL or permanent artifact DOI has yet been designated. A public availability statement must be completed before submission, rather than implying that the local package is already publicly accessible.

\section*{Funding}

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

\section*{Declaration of competing interests}

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{CRediT authorship contribution statement}

\textbf{M Hafid Ramadhan:} Software, Investigation, Formal analysis, Visualization, Writing -- original draft. \textbf{Satria Mandala:} Supervision, Methodology, Validation, Writing -- review \& editing. These roles and the funding/interests declarations were confirmed by the first author on 3 November 2026; all authors must approve the final version before submission.

\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}

OpenAI Codex (OpenAI; exact application build not recorded) assisted code implementation, experimental checks, literature organization, computational figure code and manuscript drafting. Research results derive from the recorded data and executable analyses, not generated observations. The revision includes additional model fits and donor-game evaluations, recorded separately from earlier results. The authors are responsible for verifying the methods, outputs, references and text; final approval of this revision by all authors is pending.

\FloatBarrier\bibliographystyle{elsarticle-num}\bibliography{references}\end{document}
'''
    main=main[:main.index(r'\section{Discussion and limitations}')]+discussion
    main=main.replace('A new source/session with a frozen feature pipeline, model and decision rule is still needed for that claim.','The added benign-source diagnostic reveals a large weighted-voting false-alarm burden, with incomplete source-CSV coverage. A fully aligned independent test with both normal and attack sources and a frozen pipeline is still needed for a general detection claim.')
    main=main.replace('The matched ablation separates the benefit of emitted multiscale statistics from endpoint ordering on the available traces.','The matched ablation separates the benefit of emitted multiscale statistics from endpoint ordering on the available traces. The high false-alarm rate on additional benign sources prevents interpreting those gains as reliable source transfer.')
    main=main.replace('prediction preserving execution','prediction-preserving execution')
    main=main.replace('The strict inequality preserves the winner; the implementation adds tolerance $10^{-12}$ and a conservative float64 guard. Unresolved rows evaluate all positive-weight experts and use canonical full summation.','The strict inequality preserves the winner; unresolved rows evaluate all positive-weight experts.')
    main=main.replace(r'\FloatBarrier\bibliographystyle{elsarticle-num}\bibliography{references}',r'\FloatBarrier\begingroup\small\setlength{\bibsep}{4pt}\bibliographystyle{elsarticle-num}\bibliography{references}\endgroup')
    main=main.replace('Full per-seed metrics and all recalls appear in Supplementary Tables S25--S26.','All recalls appear in Supplementary Tables S25--S26 and per-seed metrics in Table S38.')
    main=main.replace('Full historical tables and operational weights remain in Supplementary Tables S22--S24 and the provenance package.','Historical class recalls, component scores and tuning tables remain in Supplementary Tables S12, S16 and S22--S24; Table S40 retains the complete historical ranking.')
    main=main.replace('Supplementary Tables S28--S29.','Supplementary Tables S28--S29 and S39.')
    main=main.replace('A user-requested local inventory found','An audit of the available source files found')
    main=main.replace('This is a packet-pattern diagnostic, not a temporal forecast or capture-independent test.','This diagnostic does not establish temporal forecasting or independent-capture generalization.')
    main=main.replace('Supplementary Tables S31--S32 report class supports, censored-target proportions, full versus retained scores and query/update time.','Supplementary Tables S31--S32 report cohort counts, censoring, retained scores and query/update time; all class supports are in the companion CSV.')
    main=main.replace('These roles and the funding/interests declarations were confirmed by the first author on 3 November 2026; all authors must approve the final version before submission.','')
    (OUT/'main.tex').write_text(main,encoding='utf8')
    # Preserve original detailed supplement; append explicitly identified new controls.
    supp=(OLD/'supplement.tex').read_text().replace('22 October 2026','3 November 2026')
    supp=supp.replace('prediction preserving execution','prediction-preserving execution')
    supp=supp.replace('Complements main Table 3 (F1, F2, latency, inference delta RSS and composite score) and Table S12 (all five class recalls).','Complements the archived October holdout ranking and Table S12 (all five class recalls).')
    supp=supp.replace('Main Table 3 gives','Table S40 gives').replace('Main Table 3 supplies','Table S40 supplies')
    supp=supp.replace('used in the main heatmaps','retained in the archived confusion analysis (Figure S6)')
    supp=supp.replace('It contains no new fits, newly acquired captures, independent evaluation or changed scientific results.','Tables S1--S24 and Figures S1--S5 retain the archived October evidence. Tables S25 onward add the November major-revision experiments, including 70 expert fits. There is no new independent capture evaluation. Historical values are not silently replaced by the fixed-weight factorial values.')
    supp=supp.replace('There is no new independent capture evaluation.','There is no independent five-class final evaluation; a coverage-limited benign-source diagnostic is reported separately.')
    at=supp.index(r'\end{document}');supp=supp[:at]
    supp+='\n'+r'\clearpage\section{November matched factorial and history controls}'+'\n'
    # S25, S26: exactly eight rows each, all metrics/recalls in companion CSV.
    for protocol in ['temporal','signature']:
        rows=[]
        for row in f[(f.protocol==protocol)&(f.method=='weighted_voting')].itertuples():rows.append([row.view.replace('_','/'),pct(row.macro_f1),pct(row.balanced_accuracy),pct(row.recall_Normal),pct(row.recall_DDoS_TCP),pct(row.recall_Port_Scanning),pct(row.recall_SQL_injection),pct(row.recall_Ransomware)])
        supp+=table(protocol.title()+' fixed-weight factorial, all class recalls (percent).','sup:factorial_'+protocol,['Cell','F1','BA','Normal','DDoS','Scan','SQL','Ransom'],rows,'Temporal means use the same three seeds and targets. Signature values pool both folds. See factorial_summary.csv and full per-stage metrics for errors and variation.',[.16,.12,.12,.12,.12,.12,.12,.12])
    for protocol in ['temporal','signature']:
        rows=[]
        for row in f[(f.protocol==protocol)&(f.view=='c60_o10')].itertuples():rows.append([methodname[row.method],pct(row.macro_f1),pct(row.balanced_accuracy),pct(row.recall_Normal),pct(row.recall_DDoS_TCP),pct(row.recall_Port_Scanning),pct(row.recall_SQL_injection),pct(row.recall_Ransomware)])
        supp+=table(protocol.title()+' singleton/ensemble class recalls (percent).','sup:ensemble_'+protocol,['Model','F1','BA','Normal','DDoS','Scan','SQL','Ransom'],rows,'DDoS F2, errors, seed SDs and supports remain in factorial_summary.csv. All models use the same fitted reference experts and targets.',[.20,.115,.115,.114,.114,.114,.114,.114])
    supp+=table('History controls by fold: weighted-voting F1 and class recalls (percent).','sup:history',['Fold / history','F1','Normal','DDoS','Scan','SQL','Ransom','Errors'],[[row.stage[-1]+' / '+row.history.replace('_','-'),pct(row.macro_f1),pct(row.recall_Normal),pct(row.recall_DDoS_TCP),pct(row.recall_Port_Scanning),pct(row.recall_SQL_injection),pct(row.recall_Ransomware),row.errors] for row in pd.read_csv(ROOT/'runs/ARRAY_HISTORY_FULL_20261103_01/metrics.csv').query("method=='weighted_voting'").itertuples()],r'20 new expert fits. G/C/V remain original. Allocation counts, zero-overlap checks, source hashes and reconstructed contexts are in ARRAY\_HISTORY\_DATA\_20261103\_01.',[.22,.112,.112,.112,.112,.112,.112,.108])
    rows=[]
    for row in d[(d.stage=='temporal_seed42')&(d.true_class>=0)&d.comparator.isin(['random_forest','logistic_regression'])].itertuples():rows.append([methodname[row.comparator],['Normal','DDoS','Scanning','SQL','Ransomware'][row.true_class],row.ensemble_only_correct,row.comparator_only_correct,row.both_wrong])
    supp+=table('Paired correctness disagreements, temporal seed 42.','sup:disagreements',['Comparator','True class','Only weighted correct','Only comparator correct','Both wrong'],rows,'Every discordant target ID, true label and both predictions is retained in disagreement_examples.csv; all seeds and signature folds are in correctness_disagreements.csv.',[.24,.19,.19,.19,.19])
    rows=[]
    for row in b.query("method=='weighted_voting' and cohort!='early_excluded'").itertuples():rows.append([row.protocol[:4],num(row.capacity/16384,2),row.cohort,pct(row.macro_f1),int(row.n),num(row.errors,2)])
    supp+=table('Capacity and warm-up cohorts for the fixed predictor.','sup:buffer',['Protocol','MiB','Cohort','F1 (percent)','N','Errors'],rows,'warm1 is the original eligible set; common10 retains decisions at least ten seconds after reset. Common rows receive identical predictions under either purge policy. Scores across these cohorts are not matched gains.',[.16,.12,.20,.18,.16,.18])
    audit=json.loads((ROOT/'runs/ARRAY_HISTORY_DATA_20261103_01/audit.json').read_text());bl=pd.read_csv(RUN/'buffer_latency_raw.csv').groupby('capacity')[['query_seconds','update_seconds']].mean();rows=[]
    for cap in [4096,16384,65536]:
        z=read_npz(ROOT/'runs/ARRAY_HISTORY_DATA_20261103_01/histories.npz');tr=z['truncated_'+str(cap)];ev=sum(r['capacity_evictions'][str(cap)] for r in audit['records'] if r.get('kind')=='buffer');rows.append([cap,num(cap/16384,2),ev,int(tr.sum()),num(bl.loc[cap,'query_seconds'],5),num(bl.loc[cap,'update_seconds'],5)])
    supp+=table('Capacity evictions and separately timed ring operations.','sup:buffer_cost',['Records','Array MiB','Evictions','Truncated targets','Query s','Update s'],rows,'Evictions/truncation cover all reconstructed development targets, including purged targets; query/update means use ten fixed-replay repeats and exclude model calls. A truncated target has lost a packet still inside its nominal ten-second window.',[.16,.14,.18,.20,.16,.16])
    rows=[]
    scale=pd.read_csv(RUN/'scaling_raw.csv')
    for (copies,mode),g in scale.groupby(['copies','mode']):rows.append([copies,mode.replace('_',' '),15665*copies,160*copies,num(g.seconds.mean()),num(g.seconds.median()),num(g.seconds.std())])
    supp+=table('Repeated-segment workload sensitivity.','sup:scaling',['Copies','Policy','Packets','Decisions','Mean s','Median s','SD s'],rows,'Five repetitions per length/policy. Each original segment resets its ring. This varies workload length; active-endpoint count and packet rate are not independently manipulated.',[.10,.25,.14,.14,.13,.12,.12])
    ledger=[['13 Oct','Historical holdout','Used in model recommendation; original-only validation/holdout retained.'],['15 Oct','Holdout already observed','Original priority removed; all six eligible under the composite.'],['19--20 Oct','Development repeatedly inspected','Context, ordering, timing and runtime candidates developed; no untouched final set.'],['22 Oct','Prior results available','Integrated draft and archived seven-group evidence organized.'],['3 Nov','All prior results available','Four-cell/history/runtime/window-SHAP protocol frozen; fixed reference retained; no new V-based winner.']]
    supp+=table('Post-observation decision ledger and interpretation.','sup:ledger',['Date 2026','Observed evidence','Decision / consequence'],ledger,'The frozen November controls reduce additional adaptation but do not erase previous development reuse. A future independent final test requires a separate fixed protocol and new appropriate captures.',[.16,.28,.56])
    rows=[]
    for row in xs.itertuples():rows.append([methodname[row.method],num(row.top3_jaccard),num(row.normalized_l1,6),num(row.cosine,6)])
    supp+=table('Nine-group background sensitivity, A32 versus B32.','sup:xai_background',['Model','Top-3 Jaccard','Normalized L1','Cosine'],rows,'Same 100 targets, fixed predicted class, two disjoint 32-row training backgrounds. These rankings are not required to be invariant to donor selection.',[.34,.22,.22,.22])
    rows=[]
    for bg in ['A32','B32']:
        for method in methodname:
            if method=='uniform_voting':continue
            q=xi[(xi.background==bg)&(xi.method==method)].set_index('group');rows.append([bg,methodname[method],pct(q.loc['history_100ms','share']),pct(q.loc['history_1000ms','share']),pct(q.loc['history_10000ms','share']),pct(q.loc[~q.index.str.startswith('history'),'share'].sum())])
    supp+=table('Nine-group mean absolute attribution shares (percent).','sup:xai_shares',['Donors','Model','0.1 s','1 s','10 s','Protocols'],rows,'Window groups contain twenty fields each. Protocols sums six unequal-sized groups only for display. These are finite-game group magnitudes, not causal or individual-feature importance.',[.13,.29,.145,.145,.145,.145])
    supp+=table('Additional benign sources: matched coverage and weighted-voting false alarms.','sup:benign',['Source','Eligible','Matched','Missing','False alarms','Matched FPR'],[[row.source.replace('_',' '),row.eligible,row.n,row.unscorable,row.false_alarms,pct(row.false_positive_rate,2) if row.n else '--'] for row in benign.query("method=='weighted_voting'").itertuples()],r'All outcomes are Normal by source provenance. Missing candidates are never assigned fabricated predictions. Modbus has no valid matched clock. Additional data are a limited prefix of existing corpus sources, not an independent five-class dataset. The companion CSV gives all four methods and coverage bounds.',[.25,.14,.14,.14,.14,.19])
    stages=pd.read_csv(ROOT/'runs/ARRAY_FACTORIAL_FULL_20261103_01/metrics.csv')
    supp+=table('Temporal factorial outcomes for every estimator seed.','sup:seed_factorial',['Seed','Cell','F1','BA','Errors','Scan errors','Rans. errors'],[[row.stage[-2:],row.view.replace('_','/'),pct(row.macro_f1),pct(row.balanced_accuracy),row.errors,row.errors_class_2,row.errors_class_4] for row in stages[(stages.method=='weighted_voting')&stages.stage.str.startswith('temporal')].itertuples()],r'F1/BA in percent; identical targets and stage weights across each four-cell comparison. Seed-specific predictions are correlated evaluations of the same 10,358 observations.',[.08,.18,.15,.15,.14,.15,.15])
    supp+=table('Pooled history-control comparison across all four predictors.','sup:history_all',['History / model','F1','BA','DDoS F2','Scan recall','Rans. recall','Errors'],[[row.history.replace('_','-')+' / '+methodname[row.method],pct(row.macro_f1),pct(row.balanced_accuracy),pct(row.ddos_f2),pct(row.recall_Port_Scanning),pct(row.recall_Ransomware),row.errors] for row in h.itertuples()],r'Percentages on the same 52,584 signature targets. Role-specific history controls refit each expert with fixed settings; voting weights remain the original stage reference weights. Full five-class recalls are retained in history_summary.csv.',[.28,.12,.12,.12,.12,.12,.12])
    oldrank=next(t for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}',(OLD/'main.tex').read_text(),re.S) if r'\label{tab:initial_six}' in t)
    supp+=oldrank.replace(r'\label{tab:initial_six}',r'\label{sup:historical_rank}')
    supp+=figure('figure5_confusion.pdf',r'Archived October short-context and multiscale signature confusion matrices, corresponding to Table S15. The short-context model uses its original weights; this figure is not the November fixed-weight factorial. Source/class confounding and shared unlabelled histories remain.','sup:old_confusions')
    supp+=r'''\section{Artifact map and submission checks}

The new evidence directories are ARRAY\_FACTORIAL\_FULL\_20261103\_01 (50 fits), ARRAY\_HISTORY\_DATA\_20261103\_01 (raw-history reconstruction), ARRAY\_HISTORY\_FULL\_20261103\_01 (20 fits), ARRAY\_BUFFER\_FULL\_20261103\_01 (fixed-predictor sensitivity), ARRAY\_XAI\_FULL\_20261103\_01 (200 nine-group games), and ARRAY\_RUNTIME\_FULL\_20261103\_01 (fresh runtime/RSS). Smoke outputs are separate and are never used in the tables. SHA256 catalogs, configurations and derivation code accompany the local source package. All percentages are obtained from unrounded saved predictions before display rounding.

The later ARRAY\_BENIGN\_CAPTURE\_20261103\_04 run records the coverage-limited additional-source audit. Earlier attempts 01--03 retain failed alignment/completeness gates. Their failure is not hidden by relabelling the final audit as a full mapping success. The addendum records index-to-identity matching and explicit unscorable candidates; no new model is fitted or selected.

Funding, interests and contribution roles were confirmed by the first author. Public repository/redistribution arrangements, corresponding-author contact details, disclosure of the related conference submission and final approval by both authors must be completed or confirmed before submission. No journal acceptance, quartile, AI-detector score or independent validation is inferred from this revision.
\end{document}
'''
    supp=supp.replace('No journal acceptance, quartile, AI-detector score or independent validation is inferred from this revision.','')
    supp=re.sub(r'ARRAY(?:\\_[A-Z0-9]+)+',lambda m:r'\path{'+m[0].replace(r'\_','_')+'}',supp)
    (OUT/'supplement.tex').write_text(supp,encoding='utf8')
    (OUT/'highlights.txt').write_text('Matched controls separate multiscale context from endpoint ordering.\nPacket-private histories expose limits of signature-based diagnostics.\nRepeated paired replays measure same-predictor execution savings.\nWindow-level donor games audit attribution and background sensitivity.\nAll predictive claims remain exploratory and within-trace.\n')
    print('BUILT',OUT)
if __name__=='__main__':build()
