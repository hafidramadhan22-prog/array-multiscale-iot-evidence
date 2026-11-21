"""Integrate verified guard and external-test evidence in a new manuscript copy.

No fitted artifact is loaded. No inference, fitting, tuning or profiling occurs.
All previous numerical table bodies remain unchanged; additions have a row ledger.
"""
from pathlib import Path
import csv, hashlib, json, re, shutil, difflib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'paper/array_20261109_payload_audit/elsarticle'
NEW=ROOT/'paper/array_20261110_integrated/elsarticle'
OUT=ROOT/'reports/ARRAY_FINAL_INTEGRATION_20261110'
F=ROOT/'reports/FPR_SOLUTION_20261109'
E=ROOT/'reports/INDEPENDENT_CAPTURE_20261109'
R=ROOT/'runs/FPR_SQL_GUARD_20261109_01'
EXT=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01'
K='sql_guard_rf_rescue';POLICIES=['weighted_reference','random_forest',K]
PN={'weighted_reference':'WV','random_forest':'RF',K:'GV'}
CLASSES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']
LEDGER=[]

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def esc(v):
    if str(v).startswith('10.') and '/' in str(v):return r'\url{https://doi.org/'+str(v)+'}'
    return str(v).replace('&',r'\&').replace('_',r'\_\allowbreak{}').replace('%',r'\%').replace('#',r'\#')
def num(v,d=4):return '--' if pd.isna(v) else f'{float(v):.{d}f}'
def typography(text):
    text=re.sub(r'(?<=[a-z])(?=\d)', ' ', str(text))
    text=re.sub(r'(?<=\d)(?=[a-z]{2,})', ' ', text)
    text=re.sub(r'(?<=\d)(?=[A-Z][a-z]{2,})', ' ', text)
    return text.replace('float 64','float64').replace('Access 8,165130','Access 8, 165130')
def table(label,caption,header,rows,widths,note,scope='supplement'):
    assert len(widths)==len(header) and abs(sum(widths)-1)<1e-6
    assert all(len(r)==len(header) for r in rows)
    header=[{'Policy':'ID','Normal n':'n','Normal FP':'FP','Targets':'n','Ransom':'Rans.','Unmapped':'Unmap.','Ambiguous':'Amb.','Repeat':'Rep.','Attack/scenario alarm suppressed':'Alarm suppressed','Attack/scenario alarm recovered':'Alarm recovered'}.get(h,h) for h in header]
    for old,new in [
        ('Initial760 has867 eligible after warm-up and107 unmatched. Later block1 has770 planned/769 scored; block2 700/699; block3 700/697.', 'The initial cohort has 867 eligible targets after warm-up and 107 unmatched. Later block 1 has 770 planned / 769 scored; block 2 has 700 / 699; block 3 has 700 / 697.'),
        ('at most0.1pp','at most 0.1 pp'),('Flame Sensor306999','Flame Sensor target 306999'),
        ('Normal support3860, attack support6498, total10358','Normal support 3,860, attack support 6,498, total 10,358'),
        ('V1000 uses1000 original V feature rows; later uses697 third-block rows. Each policy has30 calls,3 warm-ups,one numerical thread and3 cold RSS processes.', 'V1000 uses 1,000 original V feature rows; later uses 697 third-block rows. Each policy has 30 calls, three warm-ups, one numerical thread and three cold RSS processes.'),
        ('every2ms','every 2 ms'),('CIC hasfive files with1000targets each; each TON file has1000Normal and1000Ransomware targets.', 'CIC has five files with 1,000 targets each; each TON file has 1,000 Normal and 1,000 Ransomware targets.'),
        ('the finaltwo columns','the final two columns'),('TON suppression102minus recovery75 adds27Ransomware-to-Normal decisions.', 'On TON, 102 suppressed alarms minus 75 recovered alarms add 27 Ransomware-to-Normal decisions.'),
        ('with1000targets of each','with 1,000 targets of each'),('has1000targets of each','has 1,000 targets of each'),
        ('coverage is62.90%/6.66%','coverage is 62.90% / 6.66%'),('Coverage is62.90%/6.66%','Coverage is 62.90% / 6.66%'),
        ('Each policy/source has1000targets','Each policy/source has 1,000 targets'),('allfive','all five'),('41.75%replay','41.75% replay'),('replay30-pair','replay 30-pair'),
        ('the existingtargets','the existing targets')]:note=note.replace(old,new)
    note=typography(note)
    rows=[[typography(v) if isinstance(v,str) and ' ' in v else v for v in row] for row in rows]
    header=[h.replace('Bootstrap95','Bootstrap 95') for h in header]
    note=re.sub(r'(?<!\\)%',r'\\%',note)
    LEDGER.append(dict(scope=scope,label=label,caption=caption,header=header,rows=rows,note=note))
    n=len(header);spec=''.join(r'L{'+f'{w:.8f}'+r'\dimexpr\textwidth-'+str(2*n)+r'\tabcolsep\relax}' for w in widths)
    return (r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+spec+r'}\toprule'+'\n'+' & '.join(r'\textbf{'+esc(v)+'}' for v in header)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(esc(v) for v in row)+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\par\smallskip\begin{minipage}{\linewidth}\footnotesize '+note+r'\end{minipage}\end{table}'+'\n')
def paragraph(s,start,new):
    lines=s.split('\n');hits=[i for i,l in enumerate(lines) if l.startswith(start)]
    assert len(hits)==1,(start,hits);lines[hits[0]]=new;return '\n'.join(lines)


def figures(cached,later,external):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11.4,4.0),gridspec_kw={'width_ratios':[1.18,1]})
    names=['Initial\n760','Block 1\n769','Block 2\n699','Block 3\n697','V Normal\n3,860']
    v=[cached.query("cohort=='additional760' and policy=='weighted_reference'").iloc[0].FPR_percent]
    g=[0.]
    for b in [1,2,3]:
        v.append(later.loc[(later.block==b)&(later.policy=='weighted_reference'),'FPR_percent'].iloc[0]);g.append(later.loc[(later.block==b)&(later.policy==K),'FPR_percent'].iloc[0])
    v.append(100*36/3860);g.append(100*39/3860)
    source_names=['CIC_Normal0','CIC_Normal3','TON_Ransom1','TON_Ransom2']
    vv=[external.loc[(external.source==s)&(external.policy=='weighted_reference'),'FPR_percent'].iloc[0] for s in source_names]
    gg=[external.loc[(external.source==s)&(external.policy==K),'FPR_percent'].iloc[0] for s in source_names]
    for ax,a,b,ticks,title in [(axs[0],v,g,names,'(a) Edge within-trace Normal cohorts'),(axs[1],vv,gg,['CIC N0','CIC N3','TON 1*','TON 2*'],'(b) External Normal targets')]:
        xx=np.arange(len(a));ax.bar(xx-.18,a,.34,color='#b66441',label='Reference WV');ax.bar(xx+.18,b,.34,color='#147d82',label='Guarded voting (GV)')
        for offset,values in [(-.18,a),(.18,b)]:
            for x,val in zip(xx+offset,values):ax.text(x,val+.7,f'{val:.2f}' if val<2 else f'{val:.1f}',ha='center',fontsize=8)
        ax.set_xticks(xx,ticks,fontsize=9);ax.set_ylabel('False-positive rate (%)');ax.set_ylim(0,66 if ax is axs[0] else 44);ax.set_title(title,loc='left',fontsize=10)
    axs[1].axhline(8,color='#a4293c',linestyle='--',linewidth=1,label='External target: 8%')
    handles,labels=axs[1].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.03));fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(NEW/'figures/revision_fpr_transfer.pdf',bbox_inches='tight');fig.savefig(NEW/'figures/revision_fpr_transfer.png',bbox_inches='tight',dpi=180);plt.close(fig)
    # The revised flow replaces an outdated diagram, in this new source copy only.
    fig,ax=plt.subplots(figsize=(9.5,6.1));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    def box(x,y,w,h,text,color='#eaf1f6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#31546e',linewidth=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#31546e','lw':1.2})
    box(.35,5.85,9.3,.8,'Edge-IIoTset development: prior research observations retained\nA fits experts; G weights; C execution order; V reused evaluation')
    for x,t in [(.35,'Representation (RQ1)\nC40/C60 x O1/O10\nPrivate/shared-history controls'),(3.6,'Execution (RQ2)\nFixed weighted predictor f\nLabel certificate + compact encoding'),(6.85,'Explanation (RQ3)\nFull probabilities of f\nWindow groups + donor sensitivity')]:
        box(x,3.98,2.8,1.1,t);arrow((x+1.4,5.85),(x+1.4,5.1))
    arrow((6.42,4.52),(6.78,4.52))
    box(.35,2.0,4.35,1.08,'Observed false alarms -> payload diagnostics\nPost-observation guarded predictor g\nOld f execution/XAI guarantees do not transfer',color='#fff1e7')
    arrow((1.75,3.95),(1.75,3.15))
    box(5.2,2.0,4.45,1.08,'Frozen external evaluation: f, RF and g\nCIC capture-scenario labels\nTON uniquely mapped Normal/Ransomware flows',color='#e6f2ef')
    arrow((4.8,2.55),(5.1,2.55));arrow((5,3.95),(7.1,3.16))
    box(.35,.22,9.3,.9,'Separate conclusions: within-trace representation gains; measured same-function savings;\nauditable reference explanations; weak cross-dataset detection transfer.\nNo external fitting, threshold changes or pooled five-class score.')
    arrow((2.4,1.9),(2.4,1.22));arrow((7.5,1.9),(7.5,1.22))
    fig.tight_layout();fig.savefig(NEW/'figures/revision_flow.pdf',bbox_inches='tight');plt.close(fig)


def main():
    provenance=read(OUT/'input_provenance.json');assert provenance['previous_sources_verified']
    NEW.mkdir(parents=True,exist_ok=True)
    for p in OLD.iterdir():
        if p.is_file() and p.suffix in ['.cls','.bst']:shutil.copy2(p,NEW/p.name)
    shutil.copytree(OLD/'figures',NEW/'figures',dirs_exist_ok=True)
    cached=pd.read_csv(F/'metrics_all_cached.csv');later=pd.read_csv(F/'metrics_later_blocks.csv');external=pd.read_csv(E/'primary_policy_metrics_verified.csv');pool=pd.read_csv(E/'dataset_summaries.csv');coverage=pd.read_csv(E/'TON_label_coverage.csv')
    figures(cached,later,external)
    from polish_array_final_figures_20261110 import polish
    polish(NEW)
    s=(OLD/'main.tex').read_text('utf8')
    abstract=r'''A detector can achieve high within-trace accuracy while raising alarms on benign traffic from another environment. We examine this distinction alongside bounded endpoint context, efficient execution and quantitative explanation checks for an IoT intrusion detector. On Edge-IIoTset, a matched context/ordering ablation attributes a 1.1425--1.1475 percentage-point temporal macro-F1 gain to adding ten-second statistics. Packet-private history reduces signature macro-F1 from 97.2955\% to 93.7694\%, exposing evaluation sensitivity. Compact execution of the original weighted ensemble preserves tested labels and probabilities and reduces mean replay time by 41.75\% over thirty paired repetitions on one laptop. Window-level Shapley audits measure faithfulness and background dependence. False-alarm diagnosis identifies 321/760 aligned benign errors; payload refits and manual SQL probes do not establish a reliable content-based repair. A later guarded-voting rule, fixed before scoring a third benign block, reduces its alarms from 396/697 to 1/697 (0.1435\%). On the reused validation set, however, Normal false alarms increase from 36 to 39/3,860 (1.0104\%). With all models and rules frozen, external Normal false-positive rates are 31.90\% on CICIoT2023 and 18.85\% on uniquely mapped TON_IoT targets. The guard predicts 1,657/2,000 mapped Ransomware targets as Normal. The evidence supports within-trace representation gains, measured same-function execution savings and inspectable explanation consistency; it does not establish robust cross-dataset detection.'''
    s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+abstract+r'\end{abstract}',s,count=1,flags=re.S)
    intro=r'''\section{Introduction}

At an IoT gateway, a packet often reveals little about the exchange to which it belongs. Recent endpoint activity can supply useful context, but the amount of history is limited by memory and processing time. An intrusion detector must therefore connect three separate claims: that its representation is informative, that its implementation meets a measured resource budget, and that its explanations describe the prediction function actually used. High classification accuracy alone answers none of these questions. Edge-IIoTset provides packet and tabular records with which to examine them \cite{r1}; earlier studies also show how network metadata can inflate apparent detection performance \cite{r2}.

We study a five-expert weighted ensemble retained from an earlier performance--efficiency comparison. That historical selection fixes the object of the execution and explanation experiments; it does not establish an architectural advantage over RF, LR or uniform voting. RQ1 asks whether multiscale endpoint context contributes beyond a change in endpoint ordering. RQ2 asks whether a faster execution path preserves the outputs of the same fitted predictor. RQ3 asks how its grouped Shapley explanations behave under changes of donor background and implementation.

The representation study crosses two context sizes with two ordering rules and controls ownership of unlabelled history. The execution study combines a conservative label certificate with shared fitted preprocessing, retaining encoding semantics and full probabilities where required. The explanation study evaluates window-level attribution, deletion/insertion behavior and full-probability consistency. Endpoint statistics, weighted voting, early exit and SHAP are established methods. Our contribution is the bounded representation and the explicit evidence connecting a fixed predictor, its execution contract and its explanation game.

False alarms provide a further test of the representation's limits. Additional benign Edge sources trigger many SQL alarms, prompting a payload investigation and, later, an RF-based guarded-voting rule. The latter reduces alarms to 1/697 on a later Normal block, yet produces 39/3,860 false alarms on the reused validation Normal rows. A frozen external evaluation on CICIoT2023 and TON_IoT then shows substantial false alarms and weak Ransomware detection. We distinguish these results from the execution savings: preserving a function efficiently does not make that function a reliable detector in a new environment.

The Edge detection analyses reuse development data observed during earlier research stages, and source identity is confounded with class. Their predictive gains are therefore within-trace, exploratory findings \cite{r3,r4}. The external files were not used to fit the evaluated models or choose their weights or rules, but their label scope also requires care: CIC attack results use file-scenario labels, whereas TON labels are derived from uniquely matched flows. These datasets are reported separately, without a combined five-class generalization score.

'''
    a=s.index(r'\section{Introduction}');b=s.index(r'\section{Related work}');s=s[:a]+intro+s[b:]
    s=paragraph(s,'Array includes ensemble IDS',r'''Array includes ensemble IDS and comparative ML/DL studies \cite{r21,r22}, lightweight ADS-B detection \cite{r23}, and federated WSN explanation stability \cite{r24}. Attribution-fidelity failure under IDS compression is a close precedent \cite{r25}. Here the estimators and encoding semantics are retained rather than quantized or distilled. Table~\ref{tab:related_short} states this distinction. CICIoT2023 provides an additional IoT testbed \cite{r27}, while TON_IoT supplies heterogeneous network traffic \cite{r28}; its feature-standardization concerns are directly relevant to transfer testing \cite{r29}. Published scores from these studies are not treated as matched baselines.''')
    s=paragraph(s,'The October context, ordering',r'''The context, ordering and timing investigations followed inspection of earlier development outcomes. The 3 November protocol froze additional controls before execution, but could not make the reused development data unseen. The 5 November payload candidates and 9 November guard responded to observed benign errors. Tables S34, S64 and S93 document the sequence. The guard was fixed before the third later Edge Normal block and before external scoring; no external result changed the model, rule or threshold. Stage-local freezing supports that particular prospective comparison, while the surrounding model-development program remains adaptive. The external diagnostic is distinguished from the exploratory Edge results rather than described as a uniformly labelled five-class final test.''')
    s=paragraph(s,'Because each class has one source',r'''Each canonical target class has one source, so capture identity and class are perfectly confounded. Endpoint activity, peer/port cardinalities and temporal density can encode the recording environment. Excluding raw identifiers does not prove these statistics are attack-specific, and predicting source would not resolve the confounding. We retain context summaries by source/role, compare history ownership, and separately evaluate the frozen predictors on external files in Section~\ref{sec:external_methods}.''')
    s=s.replace('The later repair in Section~\\ref{sec:payload_results} is reported as a separate post-observation candidate, not promoted as a universally superior replacement.','The payload-assisted reconciliation in Section~\\ref{sec:payload_results} and the later guard in Section~\\ref{sec:guard_results} are separate post-observation predictors with their own evidence.')
    s=paragraph(s,'The zero RF count does not establish',r'''RF's zero count on these matched benign rows does not establish a generally superior IDS. The diagnostic has no new attack source, incomplete CSV coverage, correlated targets and a limited source prefix. Sections~\ref{sec:payload_results} and~\ref{sec:guard_results} examine successive responses to the known errors. Section~\ref{sec:external_results} reports the later frozen external evaluation, including outcomes that contradict a broad repair claim.''')

    methods=r'''\subsection{Guarded voting and a later within-trace block}\label{sec:guard_methods}

The later rule uses the existing weighted predictor $f$ and RF probabilities, without a payload classifier, refit, weight change or new threshold. Let $\widehat y_f$ and $\widehat y_{\rm RF}$ be their five-class argmax labels. Guarded voting (GV) returns the full RF posterior when (i) $\widehat y_f$ is SQL injection and $\widehat y_{\rm RF}$ is Normal, or (ii) $\widehat y_f$ is Normal and $\widehat y_{\rm RF}$ is any attack. Otherwise it returns $f$. The first branch checks SQL alarms; the second recovers alarms suppressed by voting. A guard can also remove a true SQL alarm or introduce a false alarm when RF is wrong. It is a changed predictor $g$, not a prediction-preserving implementation of $f$.

The initial 760 benign targets and two later blocks informed this rule, together with seven already observed attack-containing cohorts. An alternative that reconciled every RF-Normal disagreement lost a Port Scanning alarm in one signature fold; that failure is retained. The more restricted rule was fixed before a third block of 700 planned targets from seven existing benign sources. Exactly matched, nonoverlapping target IDs yield 697 scorable rows. No missing target is replaced. The sources and their histories remain shared with earlier blocks, so this is a later within-trace check, not independent capture validation. Tables S74--S80 provide the rule, selection ledger, full class outcomes and confusion matrices.

Historical attack checks require no decrease in any attack-class recall, no increase in attack-to-Normal decisions, and at most 0.1 percentage-point loss of macro-F1. This is an explicit empirical design threshold, not statistical noninferiority or a supervisor-prescribed constant. The seven reused cohorts cannot establish independent safety. Separate native batch profiling uses thirty paired calls, three warm-ups and three cold RSS processes on the same laptop with one thread. Full five-expert posteriors are accumulated in float64. These measurements and their probability checks apply to GV, separately from the original compact replay (Tables S81--S85).

\subsection{Frozen external evaluation and label scope}\label{sec:external_methods}

CICIoT2023 \cite{r27} supplies two declared-benign files and SQL injection, Port Scanning and TCP SYN flood scenarios. We select 1,000 packet targets per file, evenly spaced by packet index after a ten-second warm-up. TON_IoT \cite{r28,r29,r30} supplies two files containing Normal and Ransomware traffic. From each, 1,000 Normal and 1,000 Ransomware targets are selected after uniquely mapping flow labels. The seven files yield 9,000 unique feature rows and 27,000 decisions across WV, RF and GV. File choice follows class relevance and availability, not model output. This is one frozen transfer attempt, with no external fitting, resampling, weight selection or threshold change. Two files do not by themselves establish independent recording sessions.

All 85 domain fields are reconstructed from PCAP rather than substituting a foreign flow-feature CSV. TShark 4.4.9 supplies protocol fields; prior-only context processes every original packet with a 16,384-record ring, windows of 0.1/1/10 s, and reset at each file boundary. Training-compatible numeric/category formats are enforced. A missing parser column is an error; true protocol absence uses the documented absent representation. Conversion from PCAPNG preserves packet count and endpoint timestamps numerically exactly. A 94-target Edge control reproduces all 85 features.

Native dissection of the 26.6-million-packet SYN file exhausts available RAM. The replacement dissects complete bidirectional TCP flows containing the same fixed target packets, together with all non-TCP packets, IP fragments and TCP DNS. It preserves their original bytes, order and timestamps. Context still uses every original packet. Before applying it to SYN, all 85 features for 1,000 CIC Normal3 targets match native extraction exactly. This control and whole-flow retention support the execution substitution; direct native equality on the complete SYN file was not measured. Failure logs and unchanged target IDs are retained.

CIC attack-file names identify scenarios, not independently confirmed malicious actions in every packet. We therefore report agreement with the scenario class and the count predicted Normal, not confirmed packet-attack recall. For TON, the original SecurityEvents table was unavailable through the login-protected link. All 23 processed, author-labelled network CSVs are checked by size/SHA before matching TCP/UDP packets by bidirectional IP/port/protocol and interval $[t_s,t_s+d+1\,\mathrm{s})$, for valid nonnegative duration $d$. The one-second extension handles source timestamp quantization and was specified before TON scoring. Unmatched, conflicting or out-of-scope labels are excluded and counted, never assigned Normal. All 4,000 selected labels are independently checked against every potentially matching flow. These are derived flow labels, not original packet annotations. Cross-dataset metrics retain this distinction; no pooled five-class macro-F1 is computed. Source provenance, coverage and all per-file results appear in Tables S86--S95.

'''
    anchor=r'\section{Results and interpretation}';s=s.replace(anchor,methods+anchor,1)
    # New result blocks follow the prior payload investigation, in chronological order.
    guard=r'''\subsection{A later guard reduces within-trace alarms, with trade-offs}\label{sec:guard_results}

Table~\ref{tab:guard_fpr} separates the low-alarm later block from the original validation population. On block 3, GV lowers alarms from 396/697 to 1/697 (0.1435\%). RF has exactly the same count. If all three unscorable planned targets were also false alarms, the coverage bound would be 4/700 (0.5714\%); it is not a confidence interval. The remaining observed alarm is a Flame Sensor target predicted Ransomware by WV, RF and GV, so the SQL-specific guard leaves it intact. The earlier zero-alarm blocks are retrospective for this rule.

On seed-42 V, GV raises Normal false alarms from 36/3,860 (0.9326\%) to 39/3,860 (1.0104\%). Macro-F1 falls from 97.880025\% to 97.850256\%, a loss of 0.029769 percentage points. Two Port Scanning targets previously predicted Normal become Ransomware alarms; their attack status is recovered but their type remains wrong. All four typed attack recalls are unchanged on that cohort. The seven historical cohorts satisfy the empirical constraints, but were already observed during design. It would therefore be incorrect to describe all main-evaluation FPRs as below 1\% or the detection problem as solved in general.

'''
    fprrows=[]
    oldrows=cached.query("cohort=='additional760'").set_index('policy')
    fprrows.append(['Initial benign',760,'321 (42.2368%)','0 (0%)','0 (0%)','Observed before rule'])
    for b in [1,2,3]:
        a=later.loc[later.block==b].set_index('policy')
        fprrows.append([f'Later block {b}',int(a.loc[K,'n'])]+[f"{int(a.loc[p,'false_alarms'])} ({a.loc[p,'FPR_percent']:.4f}%)" for p in POLICIES]+['Fixed before scoring' if b==3 else 'Observed before rule'])
    a=cached.query("cohort=='V_seed42'").set_index('policy')
    fprrows.append(['V Normal',int(a.loc[K,'normal_support'])]+[f"{int(a.loc[p,'false_alarms'])} ({a.loc[p,'FPR_percent']:.4f}%)" for p in POLICIES]+['Reused validation'])
    guard+=table('tab:guard_fpr','Normal false alarms for the final guard, with evaluation roles separated.',['Cohort','Normal n','WV: FP (FPR)','RF: FP (FPR)','GV: FP (FPR)','Role'],fprrows,[.17,.09,.18,.16,.18,.22],r'FPR uses Normal support, not all-class sample count. GV denotes SQL guard + RF rescue; it is distinct from the earlier payload-assisted rule. Later blocks are different target IDs from existing sources, not independent sessions. Tables S74--S80 give full outcomes.','main')
    guard+=r'''
The guard also adds no convincing speed advantage to its native reference. On the 1,000-row V batch, median times are 85.373 ms for WV, 81.686 ms for GV and 19.627 ms for RF. The mean GV-minus-WV difference is $-0.551$ ms, with paired bootstrap interval $[-3.753,3.049]$ ms; the later 697-row batch gives $-2.270$ ms, $[-7.073,2.784]$ ms. Both intervals include zero. RSS deltas and every timing repetition are retained in Tables S81--S85. These measurements do not inherit the 41.75\% compact replay saving for $f$.

\subsection{External transfer remains weak}\label{sec:external_results}

Table~\ref{tab:external_normal} reports the fixed-model external Normal results. GV produces 638/2,000 false alarms (31.90\%) on CIC and 377/2,000 (18.85\%) on mapped TON Normal targets. Reference WV produces 603 (30.15\%) and 487 (24.35\%), respectively. The guard improves the pooled TON count while worsening CIC; TON file 2 also worsens. Every individual Normal-file FPR remains above 8\%. Figure~\ref{fig:fpr_transfer} keeps these results separate from the later Edge block.

'''
    extrows=[]
    for source in ['CIC_Normal0','CIC_Normal3','TON_Ransom1','TON_Ransom2']:
        a=external.loc[external.source==source].set_index('policy')
        extrows.append([source,1000]+[f"{int(a.loc[p,'false_alarms'])} ({a.loc[p,'FPR_percent']:.2f}%)" for p in POLICIES])
    for family in ['CIC','TON']:
        a=pool.loc[pool.dataset==family].set_index('policy')
        extrows.append([family+' pooled Normal',2000]+[f"{int(a.loc[p,'Normal_FP'])} ({a.loc[p,'FPR_percent']:.2f}%)" for p in POLICIES])
    guard+=table('tab:external_normal','External Normal false alarms on the same targets, without model adaptation.',['Source','Normal n','WV: FP (FPR)','RF: FP (FPR)','GV: FP (FPR)'],extrows,[.27,.10,.21,.21,.21],r'TON rows contain only uniquely flow-mapped Normal targets. Pooling is descriptive, not an IID population estimate. Two files need not be independent sessions. Label coverage is reported separately in Table S90.','main')
    guard+=r'''
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/revision_fpr_transfer.pdf}\caption{Reference and guarded-voting FPRs on distinct Normal cohorts. The initial Edge targets and blocks 1--2 informed the guard; block 3 was scored after freezing it. V was reused. All external predictions used frozen models and rules. Asterisks denote TON targets with uniquely derived flow labels. Numeric annotations retain low rates that are difficult to see as bars.}\label{fig:fpr_transfer}\end{figure}

The decision changes are directly countable. On CIC Normal, the rescue branch adds 62 alarms and the SQL check removes 27, giving 35 extra false alarms. On TON Normal, it adds 95 and removes 205, a net reduction of 110. The rescue branch can therefore amplify false alarms when RF transfers poorly. Table S91 records these transitions. They explain the rule's behavior on the same inputs, without identifying a causal feature mechanism.

Table~\ref{tab:external_attack} shows why reduced benign alarms cannot be considered alone. On 2,000 mapped TON Ransomware targets, GV predicts 1,657 as Normal (82.85\%), versus 1,630 for WV. Its binary alarm recall is 17.15\%, while correct Ransomware-type recall is 31/2,000 (1.55\%). For CIC Port Scanning, predictions of Normal increase from 465 to 555; for SYN, from 2 to 12. These CIC counts concern scenario-labelled packets, so they are not confirmed attack false negatives. Nevertheless, they cannot support a claim that detection is preserved on external traffic.

'''
    attackrows=[]
    for source in ['CIC_SQL','CIC_PortScan','CIC_DDoS_SYN','TON_Ransom1','TON_Ransom2']:
        a=external.loc[external.source==source].set_index('policy');n=int(a.loc[K,'attack_support'])
        typed=[f"{int(a.loc[p,'n']-a.loc[p,'errors'])}/{n}" if source.startswith('CIC') else f"{round(a.loc[p,'recall_Ransomware']*n/100)}/{n}" for p in POLICIES]
        attackrows.append([source]+typed+[int(a.loc['weighted_reference','attack_to_Normal']),int(a.loc[K,'attack_to_Normal'])])
    guard+=table('tab:external_attack','External attack-scenario agreement and flow-derived Ransomware outcomes.',['Source','WV: type matches','RF: type matches','GV: type matches','WV to Normal','GV to Normal'],attackrows,[.22,.16,.16,.16,.15,.15],r'CIC: 1,000 targets per file; matches refer to the capture scenario, not verified malicious action in each packet. TON: 1,000 Ransomware targets per file; labels are derived from uniquely matching author-labelled flows. TON Normal rows are excluded from this table. Tables S88--S92 retain complete per-policy outcomes and confusion matrices.','main')
    guard+=r'''
Mapping coverage limits the TON result further. Uniquely mapped Normal/Ransomware packets cover 62.90\% of file 1 and 6.66\% of file 2; 128,482 and 1,777,764 packets remain unmapped. The selected targets characterize only the mapped subset after warm-up. Source/class confounding in the original training, changed traffic distributions and label-granularity differences remain plausible contributors. Alignment and feature checks reduce several implementation explanations, but do not establish a single cause or make the external labels more precise than their source. The per-file scope is therefore retained throughout the supplement.

'''
    s=s.replace(r'\section{Prototype and reproducibility}',guard+r'\section{Prototype and reproducibility}',1)
    s=paragraph(s,'A separate command-line prototype evaluates',r'''Separate command-line tools reproduce the earlier payload-assisted reconciliation, the later GV predictor and the frozen external predictions from prepared 85-field inputs. They check row/schema identity and model/catalog SHA before loading known artifacts, refuse to overwrite existing evidence and perform no fit. The GV CLI does not require payload input. External replay reproduces all 27,000 labels and five-class probabilities with maximum difference zero; its review notebook runs all eighteen code cells and emits sixteen saved-result images without loading models. Fourteen extraction, label/flow and replay-gate checks pass. These tools leave the reference dashboard and its saved explanations unchanged; they do not make those explanations valid for $g$.''')
    discussion=r'''\section{Discussion and limitations}

The matched factorial answers RQ1 within the observed Edge traces: adding ten-second statistics helps under either ordering rule, whereas changing ordering alone has a small effect. The representation is bounded and prior-only, but that does not make it attack-specific. Packet-private history lowers signature performance, and the additional benign sources expose endpoint-activity profiles that the original Normal source did not represent. Capture identity remains confounded with class. Neither exact-signature separation nor history thinning supplies new recording environments.

The false-alarm investigation makes the distinction between a local repair and a transferable detector concrete. Three flag-format corrections account for only three net alarms. Syntax/TF-IDF refits leave substantial alarms, and the manual SQL probe reveals missing training support for several indicators. The earlier payload-assisted rule has no evaluated attack activation in its decision-changing region. The later GV rule does activate on historical attacks and satisfies the recorded empirical recall constraints. Its 0.1435\% later-block FPR is a useful within-trace result, but V Normal FPR rises to 1.0104\%. These different denominators and evaluation roles must not be collapsed into a single claim of FPR below 1\%.

External testing strengthens the diagnosis of limited transfer. GV reduces false alarms on some TON Normal traffic but worsens CIC Normal and leaves most mapped Ransomware targets unalarmed. The results do not show that RF is universally preferable either: its external false alarms and missed attacks also remain substantial. The two datasets differ from the training environment and from one another, while CIC scenario labels and TON flow-derived labels have different evidential strength. The low TON mapping coverage, especially in file 2, prevents extrapolation to the complete captures. The frozen experiment is informative as a failed transfer attempt, not a uniform five-class deployment benchmark.

For RQ2, compact encoding and the conservative stopping condition preserve the original function under the tested contracts. The 41.75\% mean replay reduction concerns $f$, one laptop and 160 annotated decisions in a 15,665-packet workload. Table-derived legacy fields remain part of that replay; energy use, saturation and embedded-device operation were not measured. RF is faster in native feature-input comparisons, and no equivalently optimized singleton comparison was performed. The later guard changes the function and needs separate execution and explanation evidence; its native latency intervals do not establish an additional speed gain.

For RQ3, the grouped audit describes an explicit finite-donor game. Dividing history into equal-size window groups improves interpretability of the grouping, but correlated windows and physically implausible donor hybrids remain limitations. The seven-group top-three Jaccard of 0.5520 and nine-group value of 1.0000 refer to different games. An unchanged top-three set does not imply unchanged magnitudes or order. Full-probability game preservation verifies implementation consistency; the deletion/insertion results do not establish causal explanations or an explainability advantage over RF/LR.

Finally, selection history constrains the interpretation. Earlier V and holdout observations informed the research program; the first two later benign blocks informed the final guard. The third block and external targets were scored after that rule was fixed, but only the external files change the recording environment, with the label limitations just described. Future changes motivated by these results would make the external test part of development and require another untouched evaluation. Encoded-space SMOTE, source dependence and incomplete payload reconstruction remain explicit modelling assumptions. The contribution is a reproducible separation of representation, execution, explanation and transfer evidence, rather than a general claim of deployment readiness.

\section{Conclusion}

RQ1 is supported within the Edge temporal traces: adding ten-second statistics raises macro-F1 by 1.1425 percentage points under fixed one-second ordering and 1.1475 points under fixed ten-second ordering. History ownership matters: packet-private signature macro-F1 is 93.7694\%, compared with 97.2955\% for full shared history. The original additional-benign FPR is 42.63\% before flag alignment and 42.24\% afterward. A later guard reduces a separately scored within-trace block to 1/697 alarms (0.1435\%), but raises reused V Normal FPR to 39/3,860 (1.0104\%). External Normal FPRs of 31.90\% on CIC and 18.85\% on mapped TON targets, together with 1,657/2,000 mapped Ransomware targets predicted Normal, limit the detection claim to the observed setting.

For RQ2, the original compact certified implementation preserves tested probabilities and labels on 52,684 inputs. Mean replay time falls from 1.5791 to 0.9198 s across thirty paired repetitions, a 41.75\% reduction. The mean native-minus-compact difference is 0.6593 s, with descriptive bootstrap interval [0.5965, 0.7313] s. This measured saving applies to the unchanged reference function and the recorded laptop workload, not automatically to the guard or another device.

For RQ3, the nine-group cross-background audit yields WV deletion/insertion gains of 0.3203/0.3270 on 100 targets. The separate 300-game implementation comparison preserves full-probability coalition values and attributions after compact execution. These checks establish auditable explanation consistency for a specified predictor and donor game. They do not establish causal faithfulness or superiority over simpler models. The combined evidence supports bounded-context representation gains and same-function computational savings while making the unresolved cross-dataset detection failure visible.

'''
    a=s.index(r'\section{Discussion and limitations}');b=s.index(r'\section*{Data and software availability}');s=s[:a]+discussion+s[b:]
    s=paragraph(s,'Edge-IIoTset is distributed',r'''Edge-IIoTset is distributed by its creators through IEEE DataPort (\url{https://ieee-dataport.org/8939}) \cite{r1}. CICIoT2023 is available through UNB's official dataset portal \cite{r27}; TON_IoT is available through UNSW \cite{r28,r30}. The companion reproducibility archive supplies the manuscript source, frozen configurations, source/model manifests, split/target IDs, saved predictions, raw timing/RSS samples, explanation arrays and derived label mappings. It also contains selected payload excerpts and clearly marked manual-probe outputs. Full third-party CSV/PCAP downloads are not redistributed in that archive; source terms continue to apply. Size/SHA and public mirror commits identify the acquired files, without presenting mirror hashes as author-issued checksums. Prepared-feature replay and saved-label audits run independently of raw download access. Raw extraction requires the corresponding source data and software recorded in the manifests. Review notebooks expose saved tables, images and decisions without fitting or loading a model. No public code-repository or archive DOI has been established; these reproducibility materials accompany the manuscript. Provider-requested TON supporting citations are listed in Supplementary Table S95, with their component scopes.''')
    s=paragraph(s,'OpenAI Codex (OpenAI;',r'''OpenAI Codex (OpenAI; the exact application build was not recorded) assisted code development, computational checks, literature organization, programmatic figure code, and revision of manuscript language and structure. The methods and companion archive identify the recorded experiments, provenance checks and their limits. Numerical observations are taken from retained data and executable experiments; the two manual payload probes are identified as illustrations. The authors are responsible for reviewing the assisted content and for the accuracy, originality and interpretation of the submitted work.''')
    s=s.replace('TON_IoT',r'TON\_IoT')
    (NEW/'main.tex').write_text(s,encoding='utf8')

    # Bibliography: existing 26 entries retained byte-for-byte; 3 metadata-checked
    # dataset papers and the official TON resource page are appended.
    bib=(OLD/'references.bib').read_text('utf8')
    for key in ['r27','r28','r29']:
        meta=read(OUT/'references'/(key+'.json'))['message']
        authors=' and '.join(a.get('family','')+', '+a.get('given','') for a in meta['author'])
        year=meta.get('published-print',meta.get('published',{}))['date-parts'][0][0]
        fields=dict(author=authors,title='{'+meta['title'][0].replace('_',r'\_')+'}',journal=meta['container-title'][0],year=year,volume=meta.get('volume'),number=meta.get('issue'),pages=meta.get('page','').replace('-', '--'),doi=meta['DOI'])
        bib+='\n@article{'+key+',\n'+''.join('  '+k+' = {'+str(v)+'},\n' for k,v in fields.items() if v)+'}\n'
    bib+=r'''
@misc{r30,
  author = {{UNSW Canberra}},
  title = {{The TON\_IoT Datasets}},
  year = {2021},
  url = {https://research.unsw.edu.au/projects/toniot-datasets},
  note = {Official dataset description and academic-use terms; accessed 10 November 2026}
}
'''
    (NEW/'references.bib').write_text(bib,encoding='utf8')
    sup=(OLD/'supplement.tex').read_text('utf8')
    sup=sup.replace('9 November 2026\\end{center}','10 November 2026\\end{center}')
    sup=paragraph(sup,'This supplement supports the article',r'''This supplement supports ``Multiscale endpoint context and prediction-preserving execution for explainable IoT intrusion detection'' by M Hafid Ramadhan and Satria Mandala. Tables S1--S73 retain the previously documented numerical evidence: October context/execution results, November major-revision controls, false-alarm diagnosis, payload experiments and raw-content audit. Historical table values are retained rather than overwritten by a later predictor. Tables S74--S85 add the subsequent no-new-fit guarded-voting study, later benign blocks, historical attack checks and separate native cost measurements. Tables S86--S95 add frozen external source/label details, complete per-file outcomes, confusion matrices, decision changes and verification scope. These additions reuse the completed 9 November runs; this manuscript integration performs no training or model inference. Edge studies remain exploratory within-trace evaluations. External CIC scenario labels and TON flow-derived labels are reported separately, without a combined five-class final score. Earlier statements that no external test had yet been performed refer to the historical stage described in those archived sections.''')
    # Remove the related-submission identifier from publication captions as requested;
    # shared-result provenance and the editor disclosure are preserved separately.
    sup=sup.replace('Shared historical evidence with AGERS EDAS 1571374892; retained for provenance, not a new contribution.','Historical shared-cohort evidence; retained for provenance, not a new contribution.')
    sup=sup.replace('with shared AGERS results identified in the historical captions.', 'with historical shared-cohort results identified as provenance. Related-submission correspondence is retained separately for editorial review.')
    sup=sup.replace(r'\end{document}','')
    sup+=r'\FloatBarrier\clearpage\section{Final guard: scope, outcomes and cost}'+'\n'
    sup+=r'''The following tables concern GV, the later SQL guard + RF rescue, not the earlier payload-assisted rule. WV is the original weighted reference, RF the existing singleton, and GV the changed predictor. Tables S74--S85 keep these results separate from earlier execution and explanation guarantees. Percentages use the support named in each table; missing classes receive no fabricated macro-F1 or recall. All seven historical attack checks were available during rule selection. Only later block 3 was scored after freezing this rule; it still uses existing source histories.

'''
    sup+=table('sup:guard_definition','The final decision rule and its provenance.',['Condition / stage','Action / evidence'],[
        ['WV = SQL, RF = Normal','Return all five RF probabilities.'],['WV = Normal, RF = attack','Return all five RF probabilities.'],['Every other disagreement/agreement','Retain all five WV probabilities.'],['Model and preprocessing','Existing seed-42 C60/O10 experts, original weights; no fit, new threshold or payload model.'],['Before the final rule','Initial benign760, later blocks1/2 and historical attack cohorts were observed.'],['Later block3','700 planned,697 scored; unchanged rule fixed before scoring.'],['Scope','Changed predictor; no automatic compact-runtime or reference-SHAP guarantee.']], [.31,.69],r'The exact implementation and float64 posterior-selection checks are retained in the sealed GV run. A rejected RF-Normal/voting-type alternative missed one scanning alarm in signature fold 1.')
    rows=[]
    for name,a,planned,status in [('Initial760',cached.query("cohort=='additional760'"),867,'Retrospective')]+[(f'Later{b}',later.loc[later.block==b],int(later.loc[later.block==b,'planned'].iloc[0]),'Fixed before scoring' if b==3 else 'Retrospective') for b in [1,2,3]]:
        for r in a.itertuples():rows.append([name,PN[r.policy],planned,int(r.n),int(r.false_alarms),num(r.FPR_percent),status])
    sup+=table('sup:guard_blocks','All policies on the initial and later Normal cohorts.',['Cohort','Policy','Planned / eligible','Scored','FP','FPR (%)','Role'],rows,[.14,.07,.15,.10,.08,.13,.33],r'Initial760 has867 eligible after warm-up and107 unmatched. Later block1 has770 planned/769 scored; block2 700/699; block3 700/697. Missing rows are not replaced. Their all-eligible bounds are coverage bounds, not confidence intervals.')
    cohorts=['G','C','V_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2','signature_pooled']
    a=cached.loc[cached.cohort.isin(cohorts)]
    rows=[[r.cohort,PN[r.policy],int(r.n),int(r.false_alarms),num(r.FPR_percent),int(r.attack_to_Normal),num(r.macro_F1_percent),num(r.balanced_accuracy_percent)] for r in a.itertuples()]
    sup+=table('sup:guard_metrics','Complete aggregate historical outcomes for WV, RF and GV.',['Cohort','Policy','n','Normal FP','FPR (%)','Attack to Normal','Macro-F1 (%)','BA (%)'],rows,[.22,.07,.09,.10,.12,.12,.14,.14],r'FPR denominators are Normal support. Macro-F1/BA refer to all five classes, with all classes present. The pooled signature result combines the two diagnostic validation sides, not independent sessions.')
    gates=pd.read_csv(F/'independent_safety_gates.csv')
    rows=[[r.cohort,num(r.macro_F1_delta_pp,6),int(r.attack_to_Normal_delta)]+[num(getattr(r,'delta_recall_'+c),6) for c in CLASSES[1:]]+[str(r.passed)] for r in gates.itertuples()]
    sup+=table('sup:guard_gates','Empirical GV-minus-WV changes on seven reused attack-containing cohorts.',['Cohort','F1 delta (pp)','Attack to Normal delta','DDoS recall delta','Scan recall delta','SQL recall delta','Ransom recall delta','Pass'],rows,[.20,.13,.13,.12,.12,.11,.12,.07],r'Recall differences are percentage points. Passing requires no typed attack-recall decline, no increased missed binary attacks and at most0.1pp macro-F1 loss. These are operational checks on observed cohorts, not independent noninferiority tests. The filename of the historical CSV does not make its cohorts independent.')
    rows=[[r.cohort,PN[r.policy]]+[num(getattr(r,'recall_'+c)) for c in CLASSES] for r in a.itertuples()]
    sup+=table('sup:guard_recalls','All historical per-class recalls for the three policies.',['Cohort','Policy','Normal','DDoS','Scan','SQL','Ransom'],rows,[.23,.08,.138,.138,.138,.138,.138],r'Recalls are percentages. Cohort counts, complete predictions and support per class are retained in CSV. V seed-42 attack recalls remain unchanged although two Scan-to-Normal errors become Scan-to-Ransomware errors.')
    capture=pd.read_csv(F/'metrics_per_capture.csv').query('block==3')
    rows=[[r.source,PN[r.policy],int(r.planned),int(r.n),int(r.false_alarms),num(r.FPR_percent)] for r in capture.itertuples()]
    sup+=table('sup:guard_block3_captures','Third later Normal block by source and policy.',['Source','Policy','Planned','Scored','FP','FPR (%)'],rows,[.27,.10,.15,.15,.12,.21],r'Target IDs do not overlap earlier blocks, but traffic sources and histories are reused. The only observed GV alarm is Flame Sensor306999, predicted Ransomware by all three policies. Three unmatched targets remain outside the conditional FPR.')
    rows=[]
    for policy in POLICIES:
        cm=pd.read_csv(F/('CM_V_seed42__'+policy+'.csv'),index_col=0)
        for truth in CLASSES:rows.append([PN[policy],truth]+[int(v) for v in cm.loc[truth].values])
    sup+=table('sup:guard_V_confusion','V seed-42 confusion matrices for the final guard and controls.',['Policy','True class','Normal','DDoS','Scan','SQL','Ransom'],rows,[.09,.25,.132,.132,.132,.132,.132],r'Rows are true classes, columns predicted classes. Normal support3860, attack support6498, total10358. Label order is fixed across models.')
    efficiency=pd.read_csv(R/'efficiency_summary.csv')
    rows=[[r.cohort,PN[r.policy],int(r.count),num(r.mean,3),num(r.median,3),num(r.std,3),num(r.RSS_delta_mean_MiB,4),num(r.RSS_delta_SD_MiB,4)] for r in efficiency.itertuples()]
    sup+=table('sup:guard_cost','Separate native GV feature-input cost and sampled inference memory.',['Batch','Policy','Calls','Mean ms','Median ms','SD ms','RSS delta mean MiB','RSS delta SD MiB'],rows,[.13,.08,.09,.13,.13,.13,.16,.15],r'V1000 uses1000 original V feature rows; later uses697 third-block rows. Each policy has30 calls,3 warm-ups,one numerical thread and3 cold RSS processes. Delta peak subtracts the post-load baseline. Array buffers, model size and process RSS are distinct quantities.')
    paired=pd.read_csv(F/'paired_latency_comparison.csv')
    rows=[[r.cohort,PN[r.policy],num(r.mean_paired_delta_ms,3),num(r.median_paired_delta_ms,3),num(r.paired_delta_SD_ms,3),f"[{r.bootstrap_95_low_ms:.3f}, {r.bootstrap_95_high_ms:.3f}]"] for r in paired.itertuples()]
    sup+=table('sup:guard_paired_cost','Paired native latency differences relative to WV.',['Batch','Policy','Mean delta ms','Median delta ms','SD delta ms','Bootstrap95% ms'],rows,[.14,.10,.18,.18,.18,.22],r'Difference = policy minus WV on the same repetition. Both GV intervals include zero. This is repetition uncertainty on one workload/machine, not hardware-general speedup. It is separate from the original compact replay30-pair experiment.')
    raw=pd.read_csv(R/'latency_raw.csv')
    for cohort,lab in [('V1000','V'),('later','later')]:
        pivot=raw.loc[raw.cohort==cohort].pivot(index='repeat',columns='policy',values='milliseconds')
        rows=[[int(i)+1]+[num(r[p],4) for p in POLICIES] for i,r in pivot.iterrows()]
        sup+=table('sup:guard_latency_'+lab,'Every paired native latency repetition: '+cohort+'.',['Pair','WV ms','RF ms','GV ms'],rows,[.13,.29,.29,.29],r'Order was randomized within the recorded measurement schedule. Same prepared inputs and thread settings; numerical outputs checked against saved posteriors before timing was accepted.')
    rawmem=pd.read_csv(R/'memory_raw.csv');rows=[[r.cohort,PN[r.policy],int(r.repeat)+1,int(r.n),num(r.baseline_RSS_MiB),num(r.delta_peak_RSS_MiB),str(r.expected_probabilities_match)] for r in rawmem.itertuples()]
    sup+=table('sup:guard_memory_raw','All cold-process inference RSS measurements for the guard study.',['Batch','Policy','Repeat','n','Baseline MiB','Delta peak MiB','Posterior check'],rows,[.13,.09,.10,.11,.17,.18,.22],r'RSS sampled every2ms. Small deltas can miss short allocation spikes. Baseline is after model/input loading. These measurements do not certify a memory ceiling or reuse the old replay timing.')
    sup+=r'\FloatBarrier\clearpage\section{External frozen-model transfer diagnostic}'+'\n'
    rows=[]
    for name in ['CIC_Normal0','CIC_Normal3','CIC_SQL','CIC_PortScan','CIC_DDoS_SYN','TON_Ransom1','TON_Ransom2']:
        prep=read(EXT/name/'preparation.json')
        raw_hash=prep.get('raw_sha256')
        if raw_hash is None:
            filename='normal_runsomware'+name[-1]+'.pcap'
            raw_hash=next(r['sha256'] for r in read(E/'lfs_download_manifest.json') if r['path']==filename)
        rows.append([name,int(external.loc[external.source==name,'n'].iloc[0]),'Declared benign' if 'Normal' in name else 'Author flow-derived Normal/Ransomware' if name.startswith('TON') else 'Capture scenario only',raw_hash[:16]])
    sup+=table('sup:external_sources','External source files and evaluation label scope.',['Source','Targets','Label scope','Raw SHA prefix'],rows,[.23,.10,.40,.27],r'Full hashes, sizes, mirror commits, conversion metadata and packet indices are in the run. CIC hasfive files with1000targets each; each TON file has1000Normal and1000Ransomware targets. All target selection precedes the corresponding scoring. No source-file pair is asserted to be two independent sessions.')
    sup+=table('sup:external_normal_full','External Normal outcomes for all primary policies.',['Source','Normal n','WV: FP (FPR)','RF: FP (FPR)','GV: FP (FPR)'],extrows,[.27,.10,.21,.21,.21],r'FPR is conditional on the selected declared-benign or uniquely mapped Normal targets. Do not combine this denominator with the Edge cohorts or with unknown-label packets.')
    rows=[]
    for r in external.loc[external.source.str.startswith('CIC_')&~external.source.str.contains('Normal')].itertuples():rows.append([r.source,PN[r.policy],int(r.n),int(r.n-r.errors),num(100*(r.n-r.errors)/r.n),int(r.attack_to_Normal)])
    sup+=table('sup:external_CIC_scenarios','All CIC attack-scenario predictions, without packet-attack relabelling.',['Scenario','Policy','n','Scenario-class matches','Agreement (%)','Predicted Normal'],rows,[.22,.09,.10,.20,.18,.21],r'These are file-scenario agreement counts. Labels do not independently establish malicious action in each packet. Macro-F1 over five classes and confirmed per-packet attack recall are not reported from these single-scenario files.')
    rows=[]
    for r in external.loc[external.source.str.startswith('TON')].itertuples():rows.append([r.source,PN[r.policy],int(r.false_alarms),num(r.FPR_percent),int(r.attack_to_Normal),num(r.attack_recall_percent),num(r.recall_Ransomware)])
    sup+=table('sup:external_TON_metrics','Complete TON metrics on uniquely mapped Normal/Ransomware targets.',['Source','Policy','Normal FP /1000','FPR (%)','Ransom to Normal /1000','Binary recall (%)','Typed Ransom recall (%)'],rows,[.19,.08,.13,.12,.18,.15,.15],r'Each policy/source has1000targets of each flow-derived label. Binary recall counts any non-Normal alarm; typed recall counts only correct Ransomware predictions. Other predicted types are not silently counted as correct typed detections.')
    rows=[[r.source,int(r.total),int(r.normal),int(r.ransomware),int(r.warmup),int(r.unmapped),int(r.ambiguous),num(r.in_scope_labelled_percent,4)] for r in coverage.itertuples()]
    sup+=table('sup:external_TON_coverage','TON packet-label mapping coverage before target sampling.',['Source','All packets','Normal','Ransom','Warm-up','Unmapped','Ambiguous','In-scope (%)'],rows,[.23,.12,.11,.11,.11,.13,.08,.11],r'In-scope=(Normal+Ransomware)/all packets after exclusion of warm-up. Coverage is62.90%/6.66%; percentages are not overall capture detection performance. Mapped Normal/Ransomware packets are sampled evenly by packet index,1,000 per class. Unmapped labels are never replaced by Normal.')
    transitions=pd.read_csv(E/'rule_changes.csv');rows=[[r.source,int(r.Normal_FP_added),int(r.Normal_FP_corrected),int(r.attack_or_scenario_suppressed),int(r.attack_or_scenario_rescued)] for r in transitions.itertuples()]
    sup+=table('sup:external_rule_changes','External decision changes introduced by GV relative to WV.',['Source','Normal FP added','Normal FP removed','Attack/scenario alarm suppressed','Attack/scenario alarm recovered'],rows,[.25,.17,.17,.205,.205],r'For CIC attacks the finaltwo columns use scenario labels, not confirmed malicious packets. For TON they use derived Ransomware flow labels. TON suppression102minus recovery75 adds27Ransomware-to-Normal decisions.')
    rows=[]
    for source in ['TON_Ransom1','TON_Ransom2']:
        for policy in POLICIES:
            cm=pd.read_csv(EXT/source/('CM_'+policy+'.csv'),index_col=0)
            for truth in ['Normal','Ransomware']:rows.append([source,PN[policy],truth]+[int(v) for v in cm.loc[truth].values])
    sup+=table('sup:external_TON_confusion','Full five-output confusion matrices for both TON files.',['Source','Policy','True flow label','Normal','DDoS','Scan','SQL','Ransom'],rows,[.19,.07,.19,.11,.11,.11,.11,.11],r'Rows represent only the uniquely mapped true classes; all five possible model outputs are retained. Ground-truth strength is the flow-label mapping described in the main methods, not original per-packet annotations.')
    ledger=[['Earlier Edge development','Observed','Hyperparameters/context candidates and historical holdout results predate the current article.'],['Additional benign760','Observed','321aligned WV alarms; motivates payload and disagreement diagnostics.'],['Manual payload probe','Observed','Hybrid/TF-IDF SQL miss informs interpretation, without retraining.'],['Later blocks1/2 + historical attack cohorts','Observed before final GV','Reject RF-Normal/voting-type after scanning miss; choose targeted SQL rule.'],['Third later Edge block','GV frozen before scoring','1/697alarm; still same sources and correlated history.'],['CIC external scoring','Models/rule frozen','No threshold/weight change after any external outcome.'],['TON mapping addendum','Before TON scoring, after initial CIC','Defines flow-label interval/exclusions; predictor unchanged.'],['External findings','No replacement selected','Retain failed FPR target and low Ransomware recall. Any future adaptation requires another untouched test.']]
    sup+=table('sup:final_decision_ledger','Chronology and model-selection boundaries for the later studies.',['Stage','Observation status','Decision / interpretation'],ledger,[.25,.25,.50],r'Freezing a rule before one block does not erase earlier adaptive selection. A diagnostic resource substitution preserves the existingtargets; it is not permission to change the model or remove adverse results.')
    verification=[['Artifacts','All four FPR/external catalogs','Every size/SHA matches before manuscript integration; no model loading.'],['Protocol formatting','94 old Edge targets','All 85 features exact; missing parser columns rejected.'],['Native/selected-flow control','1,000 CIC Normal3 targets','All 85 features exact; SYNnative RAM failure retained.'],['TON interval implementation','First file full mapping','Fast interval unions agree with the original linear implementation.'],['TON chosen targets','4,000 labels','Each label checked against all matching author-labelled flows.'],['External metrics','9,000 inputs / 27,000 decisions','Independent CM/FPR/recall reconstruction and five-class probability checks pass.'],['External model replay','9,000 inputs / 27,000 decisions','All labels and probabilities reproduced; maximum difference 0.'],['Notebook','18 code cells / 16 images','RunAll in separate extraction byte-identical; no fit or model loading.'],['Software tests','11 scientific + 3 replay-gate','14 pass; tests do not establish scientific generalization.'],['Limits','One resource-constrained laptop','No new external runtime/energy/device performance claim.']]
    sup+=table('sup:final_verification','Verification evidence and its practical limits.',['Check','Scope','Result'],verification,[.22,.26,.52],r'Verification reduces implementation/provenance errors. It is not a substitute for independent label quality, representative sampling or author review. The 41.75%replay reduction remains scoped to the original reference predictor.')
    provider=[['Network','Moustafa, A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets (2021).','10.1016/j.scs.2021.102994'],['Feature diversity','Booij et al., ToN_IoT: The Role of Heterogeneity and the Need for Standardization of Features and Attack Types in IoT Network Intrusion Data Sets.','10.1109/JIOT.2021.3085194'],['Telemetry','Alsaedi et al., TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems (2020).','IEEE Access8,165130-165150'],['Windows','Moustafa et al., Federated TON_IoT Windows Datasets for Evaluating AI-Based Security Applications (2020).','10.1109/TrustCom50675.2020.00114'],['Linux','Moustafa et al., Data Analytics-Enabled Intrusion Detection: Evaluations of ToN_IoT Linux Datasets (2020).','10.1109/TrustCom50675.2020.00100'],['Dataset overview','Moustafa, New Generations of Internet of Things Datasets for Cybersecurity Applications based Machine Learning: TON_IoT Datasets (2019).','eResearch Australasia 2019'],['Architecture review','Moustafa, A systemic IoT-Fog-Cloud architecture for big-data analytics and cyber security systems: a review of fog computing (2019).','arXiv:1906.01055'],['Botnet framework','Ashraf et al., IoTBoT-IDS: A novel statistical learning-enabled botnet detection framework for protecting networks of smart cities (2021).','10.1016/j.scs.2021.103041']]
    sup+=table('sup:TON_provider_resources','TON provider-requested supporting citations, with component scope.',['Scope','Supporting publication listed by provider','Identifier'],provider,[.16,.58,.26],r'The official UNSW page requests acknowledgment of these supporting resources. Only network PCAP and processed network labels are used here; Windows/Linux/telemetry models and the botnet framework are not experimental baselines. The main bibliography identifies CIC and the TON network, heterogeneity and official resource sources. Source: \url{https://research.unsw.edu.au/projects/toniot-datasets}.')
    sup+=r'\FloatBarrier\end{document}'+'\n'
    sup=sup.replace('TON_IoT',r'TON\_IoT')
    (NEW/'supplement.tex').write_text(sup,encoding='utf8')
    highlights=['Ten-second context improves matched within-trace macro-F1 by about 1.14 points.','Compact execution preserves tested outputs and saves 41.75% mean replay time.','Window-level Shapley audits quantify faithfulness and background sensitivity.','A later guard reaches 0.14% FPR on one within-trace benign block.','Frozen external tests expose high false alarms and weak Ransomware detection.']
    assert len(highlights)==5 and max(map(len,highlights))<=85
    (NEW/'highlights.txt').write_text('\n'.join('- '+x for x in highlights)+'\n',encoding='utf8')
    write(OUT/'new_table_ledger.json',LEDGER)
    write(OUT/'build_summary.json',dict(main_tables=len(re.findall(r'\\begin\{table\}',s)),supplement_tables=len(re.findall(r'\\begin\{table\}',sup)),main_figures=len(re.findall(r'\\begin\{figure\}',s)),supplement_figures=len(re.findall(r'\\begin\{figure\}',sup)),references=30,abstract_words=len(abstract.split()),new_fits=0,new_inference=0,old_numerical_tables_preserved=True))
    for name in ['main','supplement']:
        (OUT/(name+'_changes.diff')).write_text(''.join(difflib.unified_diff((OLD/(name+'.tex')).read_text('utf8').splitlines(True),(NEW/(name+'.tex')).read_text('utf8').splitlines(True),fromfile='9November/'+name,tofile='10November/'+name)),encoding='utf8')
    print(json.dumps(read(OUT/'build_summary.json'),indent=2))


if __name__=='__main__':main()
