"""Author native enhanced manuscript from verified, separately dated evidence."""
from pathlib import Path
import json,re,shutil,hashlib,difflib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'paper/array_20261117_revision/latex';BASE=ROOT/'paper/array_20261117_enhanced';DEST=BASE/'latex';OUT=ROOT/'reports/ARRAY_ENHANCEMENT_20261117'
def text(p):return p.read_text('utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def para(s,start,new):
    a=s.index(start);b=s.find('\n\n',a);return s[:a]+new+s[b:]
def tabs(s):return re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S)
def body(s):return s.split(r'\midrule')[1].split(r'\bottomrule')[0]
def table(caption,label,headers,rows,widths,note):
    n=len(headers);spec=''.join('L{'+str(w)+r'\dimexpr\textwidth-'+str(2*n)+r'\tabcolsep\relax}' for w in widths)
    return '\n'+r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}'+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}\n'+r'\begin{tabular}{'+spec+r'}\toprule'+'\n'+' & '.join(r'\textbf{'+h+'}' for h in headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\par\smallskip\begin{minipage}{\linewidth}\footnotesize '+note+r'\end{minipage}\end{table}'+'\n'
POL=['WV_native_full','WV_native_certified','WV_compact_full','WV_compact_certified','RF_native','RF_compact','LR_native','LR_compact']
NAMES=dict(zip(POL,['WV native/full','WV native/cert.','WV compact/full','WV compact/cert.','RF native','RF compact','LR native','LR compact']))
METHOD={'weighted_voting':'WV','random_forest':'RF','logistic_regression':'LR'}
CLASSES=['Normal','DDoS','Scanning','SQL','Ransomware']

RELATED=r'''\section{Related work}

\subsection{Traffic representation and evaluation context}
Communication statistics provide a long-established alternative to inspecting isolated packet fields. Kitsune uses efficient network statistics for anomaly detection \cite{r5}, while time-based traffic descriptors predate current IoT benchmarks \cite{r20}. Edge-IIoTset provides relevant packet and tabular observations \cite{r1}. The present representation study asks a narrower question than whether historical statistics are useful: whether adding a ten-second scale helps when the endpoint-ordering rule, target split and model settings remain fixed. The four-cell control separates emitted statistics from the rule assigning the two endpoint slots. Its bounded, query-before-insertion implementation specifies which earlier traffic is actually available to a target.

These properties do not establish that the statistics identify attacks. Metadata can affect IDS outcomes \cite{r2}, and leakage or repeated model selection can distort scientific evaluation \cite{r3,r4}. The history-ownership controls therefore examine the traffic surrounding each target, alongside signature separation and temporal boundaries. They retain the limitation that source identity and class coincide in the five canonical captures. CICIoT2023 and TON\_IoT supply additional recording and labelling conditions \cite{r27,r28,r29}; their scenario and flow labels require separate interpretations rather than a direct comparison of published accuracy values.

\subsection{Serving cost and preservation of a fitted function}
QWYC jointly optimizes ensemble order and early-stopping thresholds \cite{r6}. It establishes the relevance of selective expert evaluation; early termination itself is not a new contribution here. We instead specify a sufficient remaining-weight certificate, numerical guards and a canonical tie fallback for a fixed fitted vote. The label-only contract is distinguished from the full posterior required by probability metrics and explanations. Selecting an execution order changes which computation is requested, while the reference class decision remains the target of preservation.

Hummingbird compiles featurization operators and traditional models into tensor operations \cite{r7}. CATO jointly optimizes predictive performance and serving-pipeline costs, including feature acquisition \cite{r8}. Both motivate accounting for processing around the estimator. Our implementation retains the fitted estimators and categorical/CSR semantics, and shares their existing encoding. A matched serving ablation crosses native/compact preprocessing with full/certified evaluation, while RF and LR receive the same compact encoding. This comparison isolates implementation choices within the tested software contract; it does not benchmark against those systems or establish a general-purpose compiler.

\subsection{Explanation consistency, sensitivity and faithfulness}
SHAP attributes the output of a specified function \cite{r9}. E-XAI, grouped IoT explanation studies and federated IDS work already evaluate explanation quality beyond a single importance plot \cite{r10,r11,r12}. Array studies also address ensemble IDS, model comparisons and lightweight detection \cite{r21,r22,r23}. In particular, FedSHAP-WSN examines stability of top-feature sets under resampling \cite{r24}. We separate three questions: agreement between executions of the same function, sensitivity to donor backgrounds, and ranking faithfulness under defined deletion/insertion interventions. Unchanged top-set membership can coexist with changed ordering or attribution magnitudes.

Attribution-fidelity failure in compressed IDS models is a close precedent \cite{r25}. Here the fitted function is retained rather than replaced by a compressed approximation. The audit checks full-probability coalition outputs before comparing attributions. RISE supplies an intervention-evaluation precedent \cite{r13}; feature dependence and removal semantics constrain its interpretation \cite{r14,r15}. Enumerating a finite donor game gives exact values for that game, not causal validity. Per-class and correct/incorrect-target summaries further distinguish explaining a model from validating its decision. Supplementary Table S96 summarizes the precedents, and Table S114 maps each research question to its evidence and remaining limits.

'''

ALGORITHM=r'''
\paragraph{Execution contract.}
The following steps summarize the serving path; they are an implementation specification of Eq.~\ref{eq:certificate}, not an additional learning algorithm.
\begin{enumerate}
\item Validate the frozen schema, class alignment, finite nonnegative weights and supported fitted encoding. Preserve sparse structure and missing/unknown-category semantics.
\item If a full posterior is requested, encode once, evaluate every expert and sum in canonical order. Use this path for coalition games and probability metrics.
\item For label-only requests, visit experts in the frozen order and evaluate unresolved rows only. Update partial scores and remaining weight.
\item Stop a row only when the strict certificate passes. Otherwise evaluate the remaining experts and apply the canonical full-sum tie rule. Do not expose partial scores as full probabilities.
\end{enumerate}
The 1 MiB bound concerns the history record array. It is not a bound on fitted models, temporary feature matrices or process RSS.

'''

def main():
    analysis=json.loads(text(OUT/'analysis.json'));assert analysis['status']=='PASS_ENHANCEMENT_SUMMARIES_FROM_SEALED_EVIDENCE'
    DEST.mkdir(parents=True,exist_ok=True)
    used=set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)',text(OLD/'main.tex')+text(OLD/'supplement.tex')))
    for name in ['main.tex','supplement.tex','cover_letter.tex','references.bib','elsarticle.cls','elsarticle-num.bst','highlights.txt']:
        shutil.copy2(OLD/name,DEST/name)
    (DEST/'figures').mkdir(exist_ok=True)
    for rel in used:
        for ext in ['.pdf','.svg']:shutil.copy2((OLD/rel).with_suffix(ext),(DEST/rel).with_suffix(ext))
    fs=BASE/'figure_sources'
    if not fs.exists():shutil.copytree(ROOT/'paper/array_20261117_revision/figure_sources',fs)
    main0=text(OLD/'main.tex');sup0=text(OLD/'supplement.tex');m=main0;s=sup0
    a=m.index(r'\section{Related work}');b=m.index(r'\section{Study design}');m=m[:a]+RELATED+m[b:]
    m=m.replace('Figure~\\ref{fig:flow} connects the evaluation roles.', 'Figure~\\ref{fig:flow} connects the evaluation roles. The representation question uses matched context/ordering and history controls; the execution question compares the same frozen function under different serving paths; the explanation question tests the defined full-probability game. Table S114 and the companion claim index identify their inputs, verification commands and limits. The subsequent eight-policy serving session was specified before its measurements; it reuses the reference model and workloads and introduces no new predictive test.')
    pos=m.index(r'\subsection{Explanation game and measurements}');m=m[:pos]+ALGORITHM+m[pos:]
    m=m.replace('Raw resource measurements are supplementary.', 'Raw resource measurements are supplementary.')
    mark=r'\subsection{False-alarm and transfer diagnostics}'
    methods=r'''A separate 17 November serving session crosses native/compact encoding with full/certified WV evaluation. Native and compact RF/LR retain their own fitted estimators and native-output targets. Five warm-ups and thirty randomized eight-policy repetitions use the same one-row/1,000-row prepared inputs and fixed replay. Five cold processes per policy/workload yield eighty sampled RSS measurements. Within-repeat differences, including the additive two-factor interaction, receive a seeded 10,000-resample paired bootstrap. These intervals describe timing repetition on this machine/workload, not variability across captures. Configuration and raw results are retained in Tables S106--S109. The historical thirty-pair session is reported separately and not pooled with this one.

'''
    m=m.replace(mark,methods+mark)
    # Keep main Table1's original numerical body; expose seed SDs in its note.
    f=pd.read_csv(ROOT/'reports/ARRAY_MAJOR_REVISION_20261103/factorial_summary.csv');f=f[(f.protocol=='temporal')&(f.method=='weighted_voting')].set_index('view')
    sd=', '.join(f'{100*f.loc[v,"macro_f1_sd"]:.4f}' for v in ['c40_o1','c40_o10','c60_o1','c60_o10'])
    m=m.replace('Full recalls appear', 'Full recalls appear')
    target='All recalls appear in Supplementary Tables S25--S26 and per-seed metrics in Table S38.'
    assert target in m
    m=m.replace(target,'Temporal macro-F1 sample SDs in row order are '+sd+' percentage points across three estimator seeds on fixed targets. All recalls appear in Supplementary Tables S25--S26; Tables S38 and S113 retain per-seed results and paired gains.')
    ft=pd.read_csv(OUT/'feature_summary.csv').set_index(['policy','batch']);rp=pd.read_csv(OUT/'replay_summary.csv').set_index('policy');mm=pd.read_csv(OUT/'memory_summary.csv').set_index(['policy','workload']);ct=pd.read_csv(OUT/'paired_contrasts.csv').set_index('contrast')
    rows=[[NAMES[p],f'{ft.loc[(p,1),"median_ms"]:.3f}',f'{ft.loc[(p,1000),"median_ms"]:.3f}',f'{mm.loc[(p,"feature"),"mean"]:.3f}',f'{rp.loc[p,"mean"]:.4f}',f'{mm.loc[(p,"replay"),"mean"]:.3f}'] for p in POL]
    oldcost=tabs(m)[3]
    newcost=table('Matched eight-policy serving costs in the separate 17 November session.','tab:runtime',['Policy','1 row ms','1,000 rows ms','Feature delta MiB','Replay s','Replay delta MiB'],rows,[.25,.13,.16,.17,.13,.16],
      'Feature latency is the median of 30 calls after five warm-ups; memory refers to the 1,000-row workload. Replay is the mean of 30 repetitions over 15,665 packets/160 decisions. RSS is mean delta peak after load from five cold processes, sampled every 2 ms. Each singleton preserves its own native outputs. The historical session is retained in Table S105; its timings are not pooled here.')
    m=m.replace(oldcost,newcost)
    m=m.replace('Thirty paired replay measurements, with the order', 'Historical 3 November paired replay measurements, with the order')
    a=m.index('All 52,684 development inputs pass');b=m.index(r'\subsection{Explanation consistency and sensitivity}',a)
    combined=ct.loc['combined_WV'];encoding=ct.loc['encoding_certified'];interaction=ct.loc['factorial_interaction']
    serving=r'''The earlier implementation audit checks all 52,684 development inputs, preserving encoded values/CSR structure, full probabilities and labels. Certified WV evaluates an average of 3.0715 experts. In the historical 3 November session, native/compact certified replay means are 1.5791/0.9198 s (41.75\% reduction), medians 1.5135/0.9310 s and sample SDs 0.2243/0.1137 s. The mean paired saving is 0.6593 s with descriptive bootstrap interval [0.5965, 0.7313] s; none of those thirty compact repetitions is slower (Figure~\ref{fig:runtime}, Tables S45/S105). The separate five-pair archive contains one reversal without sufficient operating-system telemetry to identify its cause. It is not explained retrospectively by the later measurements.

'''
    serving+=f'Table~\\ref{{tab:runtime}} reports the separate eight-policy session. Native full WV takes {rp.loc["WV_native_full","mean"]:.4f} s per replay and compact certified WV {rp.loc["WV_compact_certified","mean"]:.4f} s, a {combined.relative_reduction_percent:.2f}\\% reduction in the ratio of means. Compact full WV takes {rp.loc["WV_compact_full","mean"]:.4f} s. Under certification alone, compact encoding changes mean replay time by {encoding.mean_saving_s:.4f} s relative to native encoding, with paired interval [{encoding.ci95_low_s:.4f}, {encoding.ci95_high_s:.4f}] s. All tested labels match their own native function; tested full posteriors agree within $10^{{-12}}$. These are implementation checks on fixed workloads, not a new detection evaluation.\n\n'
    serving+=f'The paired additive interaction, defined as native stopping savings minus compact stopping savings, is {interaction.mean_saving_s:.4f} s [{interaction.ci95_low_s:.4f}, {interaction.ci95_high_s:.4f}] (Table S109). Component percentage reductions should consequently not be added. Comparing all four cells distinguishes the effect of encoding from the effect of selective expert evaluation, including the full-posterior path needed by explanation workloads. This session does not measure a complete SHAP computation.\n\n'
    serving+=f'The same compact encoding is also applied to RF and LR. Their one-row medians are {ft.loc[("RF_compact",1),"median_ms"]:.3f}/{ft.loc[("LR_compact",1),"median_ms"]:.3f} ms, compared with {ft.loc[("WV_compact_certified",1),"median_ms"]:.3f} ms for compact certified WV. For 1,000 rows the corresponding values are {ft.loc[("RF_compact",1000),"median_ms"]:.3f}/{ft.loc[("LR_compact",1000),"median_ms"]:.3f}/{ft.loc[("WV_compact_certified",1000),"median_ms"]:.3f} ms. Thus model choice and preprocessing implementation must both accompany a latency comparison. Table S107 gives IQRs rather than treating individual repetitions as independent traffic samples.\n\n'
    serving+=r'''In this session, compact certified WV has lower one-row latency than native LR, but compact LR reverses that comparison. WV also has lower one-row latency than compact RF, while both singletons have lower 1,000-row latency. Selective evaluation can save work on individual decisions, whereas batching amortizes fixed costs differently. These observed reversals explain why an optimized ensemble should not be compared only with native singleton pipelines or described as universally cheaper.

'''
    serving+=r'''Resource changes are not uniform across policies; Tables S107--S108 retain every measured RSS summary. The record-array bound does not bound total inference memory. Both sessions use one laptop and legacy packet fields in the replay. No energy saving, embedded-hardware performance, or optimality over all possible singleton implementations is inferred. The 41.75\% historical saving and all WV preservation contracts concern reference WV, not GV.

'''
    m=m[:a]+serving+m[b:]
    x=pd.read_csv(OUT/'XAI_correctness.csv');w=x[x.method=='weighted_voting'].set_index('correct');rank=pd.read_csv(OUT/'XAI_ordered_rank.csv').set_index('method');rw=rank.loc['weighted_voting']
    a=m.index("WV's revised top-three Jaccard");b=m.index('The separate seven-group',a)
    xpara=r'''WV's nine-group top-three set Jaccard is 1.0000 with normalized L1 distance 0.028066. All three history windows occupy that set under both backgrounds. A finer ordered-rank check nevertheless changes one of the 100 ordered top-three lists; mean full-rank Kendall $\tau_b$ is 0.8320, with minimum 0.6789. Top-one identity remains unchanged on all 100 targets. Stable set membership therefore does not imply identical secondary ordering or magnitude (Table S112). The archived seven-group Jaccard is 0.5520 despite cosine similarity 0.999948; changing the group definition changes the game.

'''
    xpara+=f'Tables S110--S111 disaggregate the same 100-target audit, with twenty targets per true class. WV has 97 correct and three incorrect predictions. Mean deletion/insertion gains are {w.loc[True,"deletion_gain"]:.4f}/{w.loc[True,"insertion_gain"]:.4f} on correct targets and {w.loc[False,"deletion_gain"]:.4f}/{w.loc[False,"insertion_gain"]:.4f} on incorrect targets; every WV deletion gain remains positive. These interventions follow the original predicted class, so a faithful explanation can describe an incorrect classification. Three errors are insufficient for a general comparison between correctness subgroups. Per-class and RF/LR controls remain descriptive, with no claim that WV is more explainable.\n\n'
    m=m[:a]+xpara+m[b:]
    m=m.replace("RF\\'s","RF's").replace('97.880025\\%','97.8800\\%').replace('97.850256\\%','97.8503\\%')
    m=m.replace('Source/class confounding remains', 'Source/class confounding remains')
    m=m.replace('WV attribution results do not explain GV\'s changed decisions.', "WV attribution results do not explain GV's changed decisions. The correctness-disaggregated audit also shows why faithfulness cannot validate an alert: all three misclassified WV targets have positive deletion gains for their original, incorrect predicted classes. This is a property of the defined intervention, not evidence of operationally trustworthy decisions.")
    # Integrate the new measured controls into RQ2 while retaining historical scope.
    old='The 41.75\\% replay reduction in Section 5.2 concerns one laptop, 160 annotated decisions and a fixed workload using legacy packet fields.'
    m=m.replace(old,'The historical 41.75\\% replay reduction in Section 5.2 concerns one laptop, 160 annotated decisions and a fixed workload using legacy packet fields. The separate eight-policy session disentangles encoding and stopping and applies compact encoding to RF/LR, avoiding a comparison restricted to unoptimized singleton pipelines. Its paired interaction and batch-dependent costs qualify the practical value of retaining WV.')
    m=m.replace('Code and derived evidence are distributed in the companion \\nolinkurl{ARRAY_HAFID_REPRODUCIBILITY_REVISED_20261117.zip}', 'Code and derived evidence are distributed in the companion \\nolinkurl{ARRAY_HAFID_REPRODUCIBILITY_ENHANCED_20261117.zip}')
    m=m.replace('Tables S96--S104 retain relocated methods and detailed diagnostics.', 'Tables S96--S104 retain relocated methods and detailed diagnostics; Tables S105--S114 retain the historical cost table, additional serving controls and disaggregated audits. The machine-readable claim index links each main result to its saved inputs and a verification command; one entry point checks all saved results without loading estimators.')
    m=m.replace('and reduces mean replay time by 41.75\\% over thirty pairs on one laptop.', f'and reduces mean replay time by {encoding.relative_reduction_percent:.2f}\\% against the native certified path in a thirty-repeat, eight-policy laptop session. The serving ablation isolates preprocessing and stopping while giving RF/LR the same compact encoding.')
    m=m.replace('support an auditable finite-game explanation evaluation', 'support an auditable finite-game explanation evaluation, complemented by per-class, correctness and ordered-rank diagnostics')
    m=m.replace('The saving applies to the recorded laptop/workload.', 'The saving applies to the recorded laptop/workload. The additional eight-policy control separates encoding and stopping effects and reports equally encoded RF/LR costs, without selecting a new classifier.')
    m=m.replace('For RQ2, compact execution of that unchanged WV predictor reduces mean replay time from 1.5791 to 0.9198 s across thirty pairs (41.75\\%), with tested label preservation and full-posterior equivalence when requested (Section 5.2).',
       f'For RQ2, compact encoding of the unchanged certified WV predictor reduces mean replay time from {rp.loc["WV_native_certified","mean"]:.4f} to {rp.loc["WV_compact_certified","mean"]:.4f} s ({encoding.relative_reduction_percent:.2f}\\%) in the eight-policy session, with tested label preservation and full-posterior equivalence when requested (Section 5.2). The historical session records 41.75\\% under the same type of comparison; its measurements remain separate.')
    m=m.replace('Recent endpoint activity can inform packet-level IoT intrusion detection, but representation, execution and explanation need separate evidence. ', '')
    m=m.replace(r'\setlength{\bibsep}{3pt}',r'\setlength{\bibsep}{1.5pt}')
    # Preserve every historical supplement table body and append new evidence.
    s=s.replace("RF\\'s","RF's")
    s=s.replace('These are the measurements summarized in main Table 4 and Figure 3.', 'These historical measurements are summarized in Table S105 and main Figure 3; the additional eight-policy session is separate.')
    s=s.replace('the 41.75\\% replay saving in main Table 4', 'the historical 41.75\\% replay saving in Table S105')
    s=s.replace('Numerical bodies of Tables S1--S104 are unchanged.', 'Numerical bodies of Tables S1--S104 are unchanged. Tables S105--S114 add the historical cost summary and the 17 November serving and saved-result analyses. The serving session performs inference/profiling on frozen project models without refitting; its measurements are separate from the older runs.')
    # The scope note must distinguish document-only revision from this authorized profiling.
    s=s.replace('no historical result is relabelled as a new experiment by this 17 November document revision.', 'no historical result is relabelled as a new experiment. The separately dated serving ablation in Tables S106--S109 is the new profiling experiment.')
    s=s.replace('this revision performs no training or model inference.', 'this revision performs no training. Its additional inference/profiling concerns the separately frozen serving study described in Tables S106--S109.')
    addition='\n'+r'\FloatBarrier\section{Additional serving controls and disaggregated audits}'+'\n'
    hist=oldcost.replace(r'\label{tab:runtime}',r'\label{sup:historical_serving_cost}').replace('Recorded single-session serving costs for the fixed seed-42 predictor and controls.', 'Historical 3 November serving costs for the reference predictor and controls.')
    addition+=hist+'\n'
    addition+=table('Frozen design of the additional serving ablation.','sup:serving_design',['Item','Recorded design'],[
      ['Reference','Same seed-42 C60/O10 Balanced experts, weights, order and 85-field schema.'],['WV factors','Native/compact fitted encoding crossed with full/certified evaluation.'],['Singleton controls','Native/compact RF and LR; each preserves its own native posterior and labels.'],['Workloads','Fixed 1/1,000-row inputs; 15,665-packet replay with 160 annotated decisions.'],['Timing','30 randomized eight-policy repetitions; five warm-ups; one numerical-library thread.'],['Memory','Five cold processes per policy/workload, 80 total; post-load delta peak RSS, 2 ms sampling.'],['Inference checks','All tested labels agree; full posteriors compared at tolerance $10^{-12}$.'],['Scope','One laptop; observed workload; no fitting, new data, GV profiling or new predictive evaluation.']],[.22,.78],
       'Protocol and source/input hashes are frozen before measurement in ARRAY\_SERVING\_ABLATION\_20261117\_01. Smoke outputs are excluded. Raw call order, timing components, CPU and RSS replicates are distributed in the archive.')
    rows=[]
    for p in POL:
        for batch in [1,1000]:
            r=ft.loc[(p,batch)];rows.append([NAMES[p],str(batch),f'{r.median_ms:.3f}',f'[{r.q25_ms:.3f}, {r.q75_ms:.3f}]',f'{mm.loc[(p,"feature"),"mean"]:.3f}' if batch==1000 else '--'])
    addition+=table('Prepared-feature latency and cold inference RSS in the additional session.','sup:serving_feature',['Policy','Batch','Median ms','IQR ms','Delta MiB'],rows,[.28,.10,.18,.27,.17],'Thirty calls per row after five warm-ups. Delta RSS is measured only for the 1,000-row workload, averaged over five cold processes; no batch-one RSS is imputed. Data: latency\_raw.csv and memory\_raw.csv in the new serving run.')
    rows=[[NAMES[p],f'{rp.loc[p,"mean"]:.4f}',f'{rp.loc[p,"median"]:.4f}',f'{rp.loc[p,"sd"]:.4f}',f'{mm.loc[(p,"replay"),"mean"]:.3f}'] for p in POL]
    addition+=table('Replay costs for all eight policies on identical fixed segments.','sup:serving_replay',['Policy','Mean s','Median s','SD s','Delta MiB'],rows,[.32,.17,.17,.17,.17], 'Thirty repetitions per policy; 160 decisions and 15,665 packets each. Delta RSS is the mean of five cold replay processes after model load. Legacy packet fields are supplied; this is not complete raw-field extraction, every-packet classification or SHAP timing.')
    names={'encoding_full':'Encoding / full','encoding_certified':'Encoding / certified','stopping_native':'Stopping / native','stopping_compact':'Stopping / compact','combined_WV':'Combined WV','RF_encoding':'RF encoding','LR_encoding':'LR encoding','factorial_interaction':'Interaction'}
    rows=[]
    for key,r in ct.iterrows():rows.append([names[key],f'{r.mean_saving_s:.4f}',f'[{r.ci95_low_s:.4f}, {r.ci95_high_s:.4f}]','--' if pd.isna(r.relative_reduction_percent) else f'{r.relative_reduction_percent:.2f}', '--' if pd.isna(r.optimized_slower) else str(int(r.optimized_slower))])
    addition+=table('Paired replay contrasts and the two-factor interaction.','sup:serving_contrasts',['Contrast','Mean diff. s','Descriptive 95\%','Reduction \%','Slower /30'],rows,[.27,.17,.29,.15,.12],
      'Encoding contrasts are native minus compact at fixed scoring mode; stopping contrasts are full minus certified at fixed encoding. Combined is native-full minus compact-certified. Positive differences favor the second path. Interaction = native stopping saving minus compact stopping saving. Reductions use ratios of means. Seeded 10,000-pair bootstrap intervals describe repeats on one workload/machine, not independent traffic uncertainty.')
    per=pd.read_csv(OUT/'XAI_per_class.csv');rows=[[METHOD[r.method],CLASSES[int(r.true_class)],str(r.n),str(r.correct),f'{r.deletion_gain:.4f}',f'{r.insertion_gain:.4f}'] for r in per.itertuples()]
    addition+=table('Nine-group explanation gains by true class on the fixed audit targets.','sup:xai_class',['Model','True class','n','Correct','Deletion gain','Insertion gain'],rows,[.10,.24,.08,.13,.22,.23],
      'A32 ranks and B32 interventions target the original predicted class. Twenty reused targets per true class; means shown. Per-target values remain in faithfulness.csv, subgroup SDs in XAI\_per\_class.csv. All deletion gains are positive; this does not imply correct predictions or transfer.')
    rows=[[METHOD[r.method],'Correct' if r.correct else 'Incorrect',str(r.n),f'{r.deletion_gain:.4f}',f'{r.insertion_gain:.4f}'] for r in x.itertuples()]
    addition+=table('Faithfulness disaggregated by prediction correctness.','sup:xai_correctness',['Model','Prediction','n','Deletion gain','Insertion gain'],rows,[.15,.25,.10,.25,.25],
      'The same A32-ranked/B32-intervention game is used. Each model has only three incorrect predictions in this selected audit; these means are diagnostic, not a population comparison. An attribution can faithfully explain the model probability assigned to an incorrect class.')
    rows=[[METHOD[p],str(int(r['n'])),str(int(r.top1_same)),str(int(r.ordered_top3_same)),f'{r.kendall_tau_mean:.4f}',f'{r.kendall_tau_min:.4f}'] for p,r in rank.iterrows()]
    addition+=table('Ordered background sensitivity beyond top-set Jaccard.','sup:xai_rank',['Model','n','Same top 1','Same ordered top 3','Mean tau','Min. tau'],rows,[.13,.08,.17,.24,.19,.19],
      'Absolute attributions for each original predicted class are ranked with stable tie handling. Kendall tau-b compares all nine magnitudes with ties accounted for. These are paired A32/B32 diagnostics on fixed targets, not resampled-background confidence intervals. The seven-group audit is a different game.')
    gains=pd.read_csv(OUT/'paired_context_seed_gains.csv');rows=[[str(r.seed),r.ordering.upper(),f'{r.C40_F1_percent:.4f}',f'{r.C60_F1_percent:.4f}',f'{r.gain_pp:.4f}',str(r.errors_removed)] for r in gains.itertuples()]
    addition+=table('Paired temporal context gains with ordering held fixed.','sup:context_gains',['Seed','Order','C40 F1','C60 F1','Gain pp','Errors removed'],rows,[.10,.12,.21,.21,.18,.18],
       'Percent F1 is recomputed from saved expert posteriors in canonical expert order with the original stage weights. Both cells use the same 10,358 targets within a seed. Three seeds vary estimator/resampling randomness; they are not three captures. No new fit or prediction call is required for these summaries.')
    addition+=table('Claim-to-evidence navigation for the main manuscript.','sup:claim_index',['Claim','Main evidence','Saved evidence key','Scope'],[
      ['RQ1 context','Table 1; S113','Factorial posteriors / fixed weights','Same targets/seeds; within-trace.'],['RQ1 history','Table 2; S39','History-control predictions','Source/class confounding persists.'],['RQ2 serving','Table 4; S106--S109','Serving raw calls / frozen inputs','One session, laptop and workload.'],['RQ2 historical','Figure 3; S45/S105','3 November replay pairs','Separate historical session.'],['RQ3 game','Table 5; S110--S112','Finite-game arrays / faithfulness','Defined donor game; 100 V targets.'],['RQ3 execution','S35--S36','Compact XAI equivalence audit','Seven-group contract is separate.'],['GV local','Table 6; S74--S85','Guard cohort posteriors','Post-freeze block, existing sources.'],['External','Table 7; S86--S95','External posteriors / label mapping','Scenario and flow-label diagnostics.']],[.16,.23,.30,.31],
       'The companion claim\_evidence\_index.csv contains exact relative paths, SHA256 and saved-result verification commands for each key. The entry point verify\_array\_all\_saved\_20261117.py runs the audits without estimator loading. Full raw third-party downloads are not redistributed; extraction requires source data and permissions.')
    s=s.replace(r'\FloatBarrier\section{Relocated methodological detail and prototype audit}',addition+r'\FloatBarrier\section{Relocated methodological detail and prototype audit}')
    # Preserve originals rather than updating history to look like new measurements.
    assert len(tabs(m))==7 and len(tabs(s))==114
    for i,(a,b) in enumerate(zip(tabs(main0),tabs(m))):
        if i!=3:assert body(a)==body(b),i
    for a,b in zip(tabs(sup0),tabs(s)[:104]):assert body(a)==body(b)
    assert body(oldcost)==body(tabs(s)[104])
    for name,src in [('main',m),('supplement',s)]:
        (DEST/(name+'.tex')).write_text(src,'utf8');(OUT/(name+'_enhancement.diff')).write_text(''.join(difflib.unified_diff(text(OLD/(name+'.tex')).splitlines(True),src.splitlines(True),fromfile='revision/'+name,tofile='enhanced/'+name)),'utf8')
    cover=text(OLD/'cover_letter.tex');cover=cover.replace('The measured gain is conditional on this reference predictor, hardware and replay workload.', 'The measured gain is conditional on this reference predictor, hardware and replay workload. A separate eight-policy ablation isolates preprocessing and stopping and gives RF/LR the same compact encoding.')
    cover=cover.replace('Compact execution reduces mean replay time by 41.75\\% across thirty paired repetitions on one laptop.', 'Compact encoding reduces certified-WV mean replay time by 37.40\\% in the eight-policy session; the historical paired session records 41.75\\%.')
    (DEST/'cover_letter.tex').write_text(cover,'utf8')
    (DEST/'highlights.txt').write_text('- Matched context controls isolate a within-trace ten-second representation gain.\n- A serving factorial separates encoding from certified expert stopping.\n- Compact RF/LR controls qualify the practical cost of retaining weighted voting.\n- Finite-game audits distinguish faithful explanations from correct decisions.\n- Local guard improvements coexist with substantial external detection failures.\n','utf8')
    result=dict(status='PREPARED_ENHANCED_NATIVE_ARTICLE',main_source_sha256=sha(DEST/'main.tex'),supplement_source_sha256=sha(DEST/'supplement.tex'),main_tables=7,main_figures=5,supplement_tables=114,supplement_figures=9,old_main_tables_preserved=6,old_cost_table_preserved_as='S105',all_old_104_supplement_table_bodies_unchanged=True,references=30,new_classifier_fits=0,new_profiling_session=True)
    (OUT/'source_revision.json').write_text(json.dumps(result,indent=2)+'\n','utf8');print(result['status'])
if __name__=='__main__':main()
