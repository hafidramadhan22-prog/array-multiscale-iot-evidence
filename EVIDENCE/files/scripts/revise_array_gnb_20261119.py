"""Integrate the completed GNB study; retain all material prior comparisons."""
from pathlib import Path
import sys, json, re, shutil, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from build_array_enhanced_20261117 import table, tabs, body, para
from array_svm_20261118 import CLASSES, label_metrics

OLD = ROOT/'paper/array_20261117_enhanced'
BASE = ROOT/'paper/array_20261119_gnb'
DEST = BASE/'latex'
OUT = ROOT/'reports/ARRAY_GNB_REVISION_20261119'
RUN = ROOT/'runs/ARRAY_GNB_FULL_20261119_01'
PROTOCOLS = ['temporal', 'packet_private', 'signature_shared']
PN = dict(temporal='Temporal', packet_private='Private history', signature_shared='Shared signature')
MN = dict(WV='WV', GNB_default='GNB default', GNB_G_selected='GNB selected')
STAGES = ['private_fold1','private_fold2','temporal_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']

def read(p): return p.read_text(encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, indent=2)+'\n', encoding='utf-8')
def f(x): return f'{100*float(x):.2f}'
def metrics():
    arrays={}
    for n in STAGES:
        with np.load(RUN/n/'evaluation.npz', allow_pickle=False) as z: arrays[n]=dict(z)
    records=[]
    for p in PROTOCOLS:
        names=[n for n in STAGES if ('temporal' in n if p=='temporal' else n.startswith('private' if p=='packet_private' else 'signature'))]
        for method in MN:
            if p=='temporal':
                mm=[label_metrics(arrays[n]['y'], arrays[n][method])[0] for n in names]
                rec={k:float(np.mean([m[k] for m in mm])) for k in mm[0] if isinstance(mm[0][k],(float,int))}
                rec['macro_f1_sd']=float(np.std([m['macro_f1'] for m in mm],ddof=1))
            else:
                ids=np.concatenate([arrays[n]['row_ids'] for n in names]); assert len(set(ids))==len(ids)
                rec,_=label_metrics(np.concatenate([arrays[n]['y'] for n in names]),np.concatenate([arrays[n][method] for n in names]))
                rec['macro_f1_sd']=None
            records.append(dict(protocol=p,method=method,**rec))
    return pd.DataFrame(records), arrays

def main():
    assert not BASE.exists(), 'Use a new release; do not overwrite an existing revision'
    OUT.mkdir(parents=True,exist_ok=True)
    check=json.loads(read(ROOT/'reports/ARRAY_GNB_20261119/verification.json')); assert check['status']=='PASS'
    assert sha(RUN/'catalog.json')==check['catalog_sha256']
    for rec in json.loads(read(RUN/'catalog.json')):
        p=RUN/rec['path']; assert sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes']
    DEST.mkdir(parents=True)
    for name in ['main.tex','supplement.tex','cover_letter.tex','references.bib','elsarticle.cls','elsarticle-num.bst','highlights.txt']:
        shutil.copy2(OLD/'latex'/name,DEST/name)
    shutil.copytree(OLD/'latex/figures',DEST/'figures')
    shutil.copytree(OLD/'figure_sources',BASE/'figure_sources')
    summary, arrays=metrics(); summary.to_csv(OUT/'gnb_metrics_recomputed.csv',index=False)
    stage=pd.read_csv(RUN/'stage_metrics.csv')
    idx=summary.set_index(['protocol','method'])
    old=read(DEST/'main.tex'); m=old; s=read(DEST/'supplement.tex'); oldsup=s
    original_tables=tabs(old); assert len(original_tables)==7 and len(tabs(s))==114

    abstract=r'''We study bounded prior-only multiscale context, prediction-preserving execution of a fixed weighted-voting (WV) reference, and quantitative finite-game Shapley audits. A matched context/ordering ablation on Edge-IIoTset gives a 1.1425--1.1475 percentage-point temporal macro-F1 gain from ten-second statistics. A Gaussian Naive Bayes (GNB) baseline selected on G reaches 95.05\% temporal and 43.92\% packet-private macro-F1, compared with WV's 97.90\% and 93.77\%. Earlier stronger controls remain available and preclude a general classifier-superiority claim. Compact execution preserves tested labels and full posteriors when requested and reduces mean replay time by 37.40\% against the native certified path in a thirty-repeat, eight-policy laptop session. The serving ablation isolates preprocessing and stopping, with equally encoded singleton controls retained separately. Explanation tests distinguish implementation consistency, donor-background sensitivity and intervention-specific faithfulness. A separate guarded-voting (GV) predictor reaches 1/697 false alarms on a post-freeze benign block from existing sources, but raises reused-validation Normal FPR to 1.0104\%. Its external Normal FPRs are 31.90\% on CIC and 18.85\% on flow-mapped TON; TON Ransomware correct-type recall is 1.55\%. GV inherits neither WV's runtime saving nor its explanation guarantees. The evidence supports within-trace representation gains, measured same-predictor execution and auditable explanation consistency, while source/class confounding, shared histories and transfer failures limit generalization.'''
    m=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+abstract+r'\end{abstract}',m,flags=re.S)
    m=para(m,'The WV predictor selected during',r'''The WV predictor selected during an earlier performance--efficiency comparison is retained as the object of execution optimization. Retention fixes the function to be preserved; it does not select a new best classifier. The main predictive comparison uses a Gaussian Naive Bayes (GNB) baseline on matched inputs and targets. Earlier aggregation, singleton, SVM and KNN controls remain in Supplementary Table S116; an RF advantage under private history still limits the comparative conclusion. Matched context/ordering controls, history ownership, paired runtime measurements and window-level explanation audits supply separate evidence. Development data and historical holdout results had already been observed, so evaluation roles and selection history remain explicit \cite{r3,r4}.''')
    mark=r'\subsection{Prediction-preserving execution}'
    methods=r'''\paragraph{Gaussian Naive Bayes comparator.}
GNB is a standalone conditional-independence baseline implemented with scikit-learn \cite{r26}; it is not inserted into WV or GV. It receives the same eighty-five fields, A/G/C/V IDs and training-only Balanced resampling as the corresponding reference stage. A fits median imputation, numeric standardization without centering and categorical one-hot encoding. The resulting matrix is converted to dense without changing its values. Priors are estimated from resampled A. The default variance smoothing is $10^{-9}$; thirteen predeclared candidates use $10^j$, $j=-12,\ldots,0$. G selects the mean of macro-F1 and DDoS F2, then macro-F1; ties prefer the exponent nearest $-9$, then the lower exponent. Both the default and G-selected models are evaluated on V after freezing that stage's selection. Seven stages yield 91 candidate fits, with a separate smoke run excluded. The GNB study uses the same three temporal seeds and two shared/private signature folds as the earlier comparisons.

The decision to emphasize GNB followed the earlier RF/LR, SVM and KNN results. It is therefore an exploratory comparator choice, not a preregistered search establishing WV as the strongest classifier. All earlier comparator results and the subsequent weight-retuning outcomes are retained in Tables S115--S128. G-only stage selection does not remove prior exposure to V. Gaussian and conditional-independence assumptions are restrictive for one-hot protocol fields and correlated context statistics; GNB tests this simple baseline rather than the full class of probabilistic detectors. No GNB runtime, SHAP or external-validation result is inferred from the WV audits.

'''
    m=m.replace(mark,methods+mark)
    m=m.replace('with RF/LR controls and two disjoint','with the retained RF/LR controls (Table S125) and two disjoint')

    rows=[]
    for p in PROTOCOLS:
        for method in MN:
            v=idx.loc[(p,method)]
            rows.append([PN[p],MN[method],f(v.macro_f1),f(v.balanced_accuracy),f(v.normal_fpr),f(v.ddos_f2),f'{v.errors:.2f}' if p=='temporal' else str(int(v.errors))])
    note='Percentages except errors. Temporal: means of three seeds on the same 10,358 V targets; private/shared signature: pooled disjoint two-fold outputs on 52,584 targets. GNB default uses smoothing $10^{-9}$; selected means G-only choice, not V selection. Temporal macro-F1 SDs for WV/default/selected GNB are '+', '.join(f'{100*idx.loc[("temporal",v),"macro_f1_sd"]:.4f}' for v in MN)+' percentage points. Full selection and class results are in Tables S115--S122; earlier controls remain in S116/S123.'
    newtable=table('Reference WV and the Gaussian Naive Bayes classification baseline.','tab:ensemble',
        ['Evaluation','Predictor','F1','BA','Normal FPR','DDoS F2','Errors'],rows,[.19,.18,.115,.115,.13,.115,.155],note)
    m=m.replace(original_tables[2],newtable)
    a=m.index('Table~\\ref{tab:ensemble} shows temporal');b=m.index(r'\begin{table}',a)
    m=m[:a]+r'''Table~\ref{tab:ensemble} compares the unchanged WV reference with default and G-selected GNB. On temporal V, tuning variance smoothing raises GNB macro-F1 from 82.61\% to 95.05\%, while WV reaches 97.90\%. Normal FPR is 7.05\% for selected GNB and 0.92\% for WV. Thus a default-only GNB comparison would substantially overstate the gap. With private history, WV reaches 93.77\% versus 43.92\% for selected GNB; under shared signature history the values are 97.30\% and 49.53\%. These are conditional comparisons on reused data, not external detection guarantees.

GNB's private-history DDoS recall is zero, and its Normal FPR is 17.69\%. Its error distribution is not a uniformly reduced version of the temporal result: both fold-specific histories and the two available DDoS signatures constrain what this baseline can learn (Tables S118--S122; Figure S10). Correlated context fields and non-Gaussian one-hot attributes are plausible limitations of its likelihood assumptions, but this study does not isolate a single cause. WV's advantage over this baseline does not resolve history dependence or establish broad model superiority. The retained RF control is stronger under private history, and selected SVM slightly exceeds WV temporally (Table S116). WV remains fixed because RQ2 asks whether its execution can be improved while preserving its outputs.

''' +m[b:]
    m=m.replace(r'\subsection{Value and cost of the fixed ensemble}',r'\subsection{GNB comparison and cost of the fixed ensemble}')

    # Keep the WV serving ablation prominent; preserve the full measured controls in S124.
    rt=original_tables[3]; rtnew=rt
    rtnew=rtnew.replace('Matched eight-policy serving costs in the separate 17 November session.','WV serving ablation within the separate eight-policy 17 November session.')
    rtnew=re.sub(r'^(?:RF|LR) (?:native|compact) &.*?\\\\\s*\n','',rtnew,flags=re.M)
    rtnew=rtnew.replace('Each singleton preserves its own native outputs.','The complete eight-policy table, including native/compact singleton controls, is retained in Table S124. GNB has not been profiled in this session.')
    m=m.replace(rt,rtnew)
    a=m.index('The same compact encoding is also applied to RF and LR.');b=m.index('Resource changes are not uniform',a)
    m=m[:a]+r'''The same session also applies compact encoding to RF and LR; their complete native/compact measurements remain in Table S124 and Tables S107--S109. Those controls have lower prepared-batch latency than compact WV, while some one-row comparisons reverse with encoding or batching. The runtime contribution is therefore the measured reduction for the same WV function, not an efficiency advantage inferred from GNB's lower classification scores. GNB has no matched serving-cost measurement in this revision.

'''+m[b:]
    faith=original_tables[4]; faithnew=re.sub(r'^(?:Logistic regression|Random forest) &.*?\\\\\s*\n','',faith,flags=re.M)
    faithnew=faithnew.replace('Positive gains are not causal accuracy or a claim of superior explainability.','Positive gains are not causal accuracy or a claim of superior explainability. The retained RF/LR controls are in Table S125; no GNB explanation audit has been performed.')
    m=m.replace(faith,faithnew)
    m=m.replace('RF and LR also have positive gains.','The retained controls also have positive gains (Table S125).')

    # RF remains a necessary part of the unchanged GV rule, not a replacement GNB component.
    guard=original_tables[5]; lines=[]
    for line in guard.splitlines():
        if ' & ' in line and (line.startswith('\\textbf{Cohort}') or '\\\\' in line):
            bits=line.split(' & ');assert len(bits)==6;line=' & '.join(bits[:3]+bits[4:])
        lines.append(line)
    ng='\n'.join(lines)
    begin=ng.index(r'\begin{tabular}');end=ng.index(r'\toprule',begin)
    ng=ng[:begin]+r'\begin{tabular}{'+''.join('L{'+str(x)+r'\dimexpr\textwidth-10\tabcolsep\relax}' for x in [.22,.10,.23,.23,.22])+'}'+ng[end:]
    ng=ng.replace('Tables S74--S80 give full outcomes.','Tables S74--S80 and S126 retain full outcomes and the RF component control. GNB was not scored on these cohorts.')
    m=m.replace(guard,ng)
    transfer=original_tables[6];lines=[]
    for line in transfer.splitlines():
        if ' & ' in line and (line.startswith('\\textbf{Targets}') or '\\\\' in line):
            bits=line.split(' & ');assert len(bits)==6;line=' & '.join(bits[:4]+bits[5:])
        lines.append(line)
    nt='\n'.join(lines);begin=nt.index(r'\begin{tabular}');end=nt.index(r'\toprule',begin)
    nt=nt[:begin]+r'\begin{tabular}{'+''.join('L{'+str(x)+r'\dimexpr\textwidth-10\tabcolsep\relax}' for x in [.25,.11,.28,.18,.18])+'}'+nt[end:]
    nt=nt.replace('WV: reference voting; RF: random forest; GV: guarded voting.','WV: reference voting; GV: guarded voting. The RF component control is retained in Table S127. No external GNB evaluation was performed.')
    m=m.replace(transfer,nt)
    m=m.replace('all labelled SQL; RF raises none.','all labelled SQL. The retained RF component control raises none (Table S126).')
    m=para(m,'Shared unlabelled histories can provide related context',r'''Shared unlabelled histories can provide related context to targets on opposite sides of a signature split. The drop under private ownership shows sensitivity to that evaluation context, while the matched-density control prevents attributing the whole change to sharing alone. WV exceeds the tested GNB baseline under each aggregate protocol, but the large GNB history-dependent deterioration and the retained RF advantage under private history (Table S116) prevent a general classifier-superiority claim. Temporal validation, signature diagnostics and the historical holdout answer different within-trace questions; none establishes independent-source generalization.''')
    m=para(m,'WV remains the reference because',r'''WV remains the reference because execution optimization needs a defined function to preserve. The GNB study adds a simple conditional-independence baseline, and its G-selected temporal result is much stronger than its default. It does not supersede earlier stronger controls: uniform voting is slightly higher temporally, RF performs better with private history, and selected SVM slightly exceeds temporal WV (Table S116). Choosing a main comparator after these observations is exploratory. Shared fitted preprocessing and conservative stopping establish the stated same-WV contracts on tested inputs; a label certificate does not supply a replacement posterior.''')
    m=para(m,'For RQ1, adding ten-second',r'''For RQ1, adding ten-second endpoint statistics improves within-trace temporal macro-F1 by 1.1425--1.1475 percentage points with ordering fixed (Section 5.1). WV reaches 97.90\% temporal and 93.77\% packet-private macro-F1, compared with 95.05\% and 43.92\% for G-selected GNB (Section 5.2). WV's signature score still falls from 97.2955\% to 93.7694\% with private history. Its advantage over GNB is conditional on these data and settings; retained stronger controls prevent identifying WV as a generally superior classifier.''')
    m=m.replace('ARRAY_HAFID_REPRODUCIBILITY_ENHANCED_20261117.zip','ARRAY_HAFID_REPRODUCIBILITY_GNB_REVISION_20261119.zip')
    m=m.replace('The machine-readable claim index links each main result', 'Tables S115--S128 add the GNB protocol, all earlier comparator summaries, per-class outcomes, saved original comparison tables and the later weight-retuning diagnostic. The companion archive includes the SVM, KNN, GNB and weight-study evidence, with prediction IDs and source hashes. No new fit is performed by this manuscript revision. The machine-readable claim index links each main result')
    (DEST/'main.tex').write_text(m,encoding='utf-8')

    s=s.replace('17 November 2026\\end{center}','19 November 2026\\end{center}',1)
    s=s.replace('The separately dated serving ablation in Tables S106--S109 is the new profiling experiment.','The separately dated serving ablation in Tables S106--S109 is the 17 November profiling experiment. Tables S115--S128 add the completed GNB study and retain earlier controls; these predictor comparisons do not change WV or GV.')
    s=s.replace('this revision performs no training.','the manuscript integration performs no training.',1)
    s=s.replace('Its additional inference/profiling concerns','The 17 November additional inference/profiling concerns',1)
    add=r'''\FloatBarrier\section{Gaussian Naive Bayes baseline and retained comparisons}

GNB is the main-text classification comparator in this revision. Its selection and the preceding RF/LR, SVM and KNN experiments are reported together to retain the comparison history. This change of presentation does not remove a stronger observed baseline or establish a preregistered superiority test. The fixed reference WV, its weights, execution audit and finite-donor games are unchanged. GNB is neither a WV member nor the RF component of GV. This section uses completed 18--19 November runs; manuscript editing introduces no new fit or independent data.

'''
    add+=table('Design and provenance of the additional GNB comparison.','sup:gnb_design',['Item','Recorded design'],[
        ['Predictor','GaussianNB; empirical priors from resampled A; no calibration.'],
        ['Inputs','Same 85 domain fields, A-fitted scaling and one-hot encoding; dense conversion only.'],
        ['Search','Variance smoothing $10^j$, $j=-12,\\ldots,0$; default $10^{-9}$.'],
        ['Selection','G score $0.5$ macro-F1 + $0.5$ DDoS F2; ties by F1, exponent nearest $-9$, lower exponent.'],
        ['Stages','Three temporal seeds and two folds each for private/shared signature history; 91 scientific fits.'],
        ['Training','Balanced RUS+SMOTE on A only, original per-class pre-RUS cap; G/C/V original.'],
        ['Timing','Stage selection frozen before new GNB V scoring; V and sources previously observed.'],
        ['Audit','All 91 G scores/choices, 42 stored confusion matrices, splits and hashes checked.'],
        ['Prediction replay','All 272,484 saved V labels from both policies/seven stages match; all G outputs match.'],
        ['Probability caveat','Default maximum probability-sum deviation $8.0764\\times10^{-6}$; strict $10^{-8}$ diagnostic fails. Labels unchanged; no calibration or voting claim.'],
        ['Scope','No GNB external stress, SHAP, runtime or memory benchmark.'],
        ['Search history','GNB emphasis follows prior comparator observations; budgets do not equal total historical tuning.']
    ],[.22,.78],'Protocol was fixed before the GNB fits. A separate smoke run is excluded. The strict probability diagnostic is reported, not weakened to certify a posterior.')
    prior=pd.read_csv(ROOT/'reports/ARRAY_SVM_20261118/protocol_summary.csv')
    knn=pd.read_csv(ROOT/'runs/ARRAY_KNN_FULL_20261119_01/summary.csv')
    records=[]
    for method,source in [(x,prior) for x in ['WV','RF','LR','Uniform','SVM_default','SVM_G_selected']]+[(x,knn) for x in ['KNN_default','KNN_G_selected']]+[(x,summary) for x in ['GNB_default','GNB_G_selected']]:
        vals=source.set_index(['protocol','method']); rec=[method.replace('_',' ')]
        for p in PROTOCOLS:rec.append(f(vals.loc[(p,method),'macro_f1']))
        records.append(rec)
    add+=table('Complete matched macro-F1 comparison history on the three reused protocols.','sup:all_comparators',['Predictor','Temporal F1','Private F1','Shared-signature F1'],records,[.34,.22,.22,.22],
        'All values are percentages. Temporal is a three-seed mean; other columns pool disjoint OOF targets. Default and G-selected variants are both retained. SVM is RBF, KNN uses Euclidean distances. Legacy singleton/ensemble settings and candidate searches have unequal lifetime tuning budgets. RF exceeds WV privately; selected SVM exceeds WV temporally. These results prohibit a universal WV-superiority claim.')
    add+=table('Full aggregate metrics of reference WV and both GNB policies.','sup:gnb_summary',
        ['Evaluation','Predictor','F1','BA','DDoS F2','Attack alarm recall','Errors'],[
        [PN[p],MN[method],f(idx.loc[(p,method),'macro_f1']),f(idx.loc[(p,method),'balanced_accuracy']),f(idx.loc[(p,method),'ddos_f2']),f(idx.loc[(p,method),'binary_attack_recall']),f'{idx.loc[(p,method),"errors"]:.2f}'] for p in PROTOCOLS for method in MN],
        [.19,.18,.115,.115,.115,.15,.135],'Percentages except errors. Binary attack-alarm recall counts any non-Normal output on an attack target, whereas per-class recall requires the correct type. Main Table 3 separately reports Normal FPR.')
    add+=table('Stage-level GNB outcomes before pooling or seed averaging.','sup:gnb_stages',
        ['Stage','Predictor','F1','Normal FPR','DDoS F2','Errors'],[
        [n.replace('private_fold','P').replace('signature_fold','S').replace('temporal_seed','T'),MN[method],f(rr.macro_f1),f(rr.normal_fpr),f(rr.ddos_f2),int(rr.errors)]
        for n in STAGES for method in MN for _,rr in stage[(stage.stage==n)&(stage.method==method)].iterrows()],
        [.15,.23,.15,.17,.15,.15], 'P: packet-private signature fold; S: shared-history signature fold; T: temporal seed. F1/FPR/F2 are percentages; each stage has its own A/G/C/V and recorded selected smoothing.')
    for p,label in [('packet_private','private'),('signature_shared','shared'),('temporal','temporal')]:
        add+=table('Per-class '+PN[p].lower()+' recall for WV and GNB.','sup:gnb_'+label,
        ['Class','Support','WV recall','GNB default','GNB selected'],[
            [cl.replace('_',' '),int(idx.loc[(p,'WV'),'support_'+cl]),f(idx.loc[(p,'WV'),'recall_'+cl]),f(idx.loc[(p,'GNB_default'),'recall_'+cl]),f(idx.loc[(p,'GNB_G_selected'),'recall_'+cl])] for cl in CLASSES],
        [.25,.12,.21,.21,.21], 'Recall is a percentage. '+('Means across three seeds on the same V IDs; supports count targets once.' if p=='temporal' else 'Pooled disjoint two-fold OOF outputs; supports count target IDs once.'))
    selections=[]
    for n in STAGES:
        g=pd.read_csv(RUN/n/'G_candidates.csv');chosen=json.loads(read(RUN/n/'selection_before_V.json'))['selected'];x=g[g.id==chosen].iloc[0]
        selections.append([n.replace('private_fold','P').replace('signature_fold','S').replace('temporal_seed','T'),f'$10^{{{int(x.exponent)}}}$',f(x.macro_f1),f(x.ddos_f2),f(x.score)])
    add+=table('Frozen G-only smoothing choices for Gaussian Naive Bayes.','sup:gnb_selection',
        ['Stage','Selected smoothing','G macro-F1','G DDoS F2','G objective'],selections,[.15,.23,.22,.22,.18],
        'Scores are percentages. All thirteen candidate G predictions per stage are archived. Default smoothing is always $10^{-9}$ regardless of which candidate wins G. No V score selects a candidate.')
    add+=r'''The low private-history GNB score includes zero DDoS type recall in both folds. Predictions are traceable to fixed row IDs rather than generated examples. Gaussian marginal assumptions, conditional dependence among context statistics and sparse class/signature support are plausible contributors, but none is isolated causally. GNB's selected temporal score improves substantially over default; both must be considered. Numerical posterior diagnostics are separate from label accuracy: all replayed labels agree, while the default probability sum can deviate by up to $8.0764\times10^{-6}$. No calibrated-probability, runtime or explanation claim is made for GNB.

'''
    # Retain complete original numerical tables with explicit supplementary labels.
    for i,title in [(2,'Original temporal aggregation and singleton comparison'),(3,'Complete eight-policy serving comparison'),(4,'Complete nine-group explanation comparison'),(5,'Complete local false-alarm component comparison'),(6,'Complete external component comparison')]:
        tt=original_tables[i]
        tt=re.sub(r'\\caption\{[^}]+\}',lambda _:r'\caption{'+title+'; retained from the preceding manuscript.}',tt,count=1)
        tt=re.sub(r'\\label\{[^}]+\}',lambda _:r'\label{retained:table'+str(i+1)+'}',tt,count=1)
        add+=tt+'\n\n'
    wr=pd.read_csv(ROOT/'reports/ARRAY_WEIGHT_RETUNE_20261118/protocol_summary.csv').set_index(['protocol','method'])
    add+=table('Previously completed G-only WV weight search; all declared policies retained.','sup:weight_diagnostic',
        ['Evaluation','WV weights','F1','Normal FPR','Attack alarm recall'],[
        [PN[p],{'WV_reference':'Reference','WV_G_constrained':'G constrained','WV_G_unconstrained':'G unconstrained','WV_G_heuristic':'G heuristic'}[method],f(wr.loc[(p,method),'macro_f1']),f(wr.loc[(p,method),'normal_fpr']),f(wr.loc[(p,method),'binary_attack_recall'])]
        for p in PROTOCOLS for method in ['WV_reference','WV_G_constrained','WV_G_unconstrained','WV_G_heuristic']],
        [.23,.23,.18,.18,.18], 'These are changed WV functions, not GV and not the reference execution/XAI target. Each stage compared 3,881 fixed candidates on G; no expert refit. All values are percentages. G constraints do not guarantee no attack-recall loss on V.')
    add+=r'''The constrained weight search raises private macro-F1 from 93.7694\% to 94.1734\% but increases attack-to-Normal outcomes from 1,816 to 1,960. Temporal F1 falls to 97.8189\%. With temporal seed-42 weights, later-block Normal FPR is 1/697, yet CIC Normal FPR rises to 36.10\%, TON remains 8.15\%, and TON binary Ransomware alarm recall falls to 11.55\%. All of those sources had already been observed. These trade-offs explain retaining the original WV function; no runtime or explanation guarantee is transferred to changed weights. The complete saved search, stress cohorts and all declared policies remain in the companion archive.

'''
    # New, native figure driven by actual pooled confusion counts.
    plotting={'classes':CLASSES,'matrices':{}}
    y=np.concatenate([arrays[n]['y'] for n in ['private_fold1','private_fold2']])
    for method in MN:
        pred=np.concatenate([arrays[n][method] for n in ['private_fold1','private_fold2']]);_,cm=label_metrics(y,pred);plotting['matrices'][MN[method]]=cm.tolist()
    write(BASE/'figure_sources/gnb_confusion_data.json',plotting)
    add+=r'''\FloatBarrier\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/gnb_private_confusion.pdf}\caption{Actual packet-private OOF confusion counts for WV and the two GNB policies. Rows are true classes, columns predicted classes; each panel contains the same 52,584 targets. Color is row-normalized, while annotations are integer counts. GNB has zero correctly typed DDoS decisions. This is a within-trace history diagnostic.}\label{sup:gnb_confusion}\end{figure}\FloatBarrier

'''
    end=r'\FloatBarrier\begingroup\small\setlength{\bibsep}{3pt}\bibliographystyle{elsarticle-num}'
    assert end in s;s=s.replace(end,add+end)
    assert len(tabs(s))==128,len(tabs(s))
    assert [body(t) for t in tabs(s)[:114]]==[body(t) for t in tabs(oldsup)]
    (DEST/'supplement.tex').write_text(s,encoding='utf-8')
    cover=read(DEST/'cover_letter.tex').replace('November 17, 2026','November 19, 2026')
    cover=cover.replace('Supplementary material records configurations, per-class results, runtime variation and negative findings.','The main classification comparison uses default and G-selected Gaussian Naive Bayes on matched data. Earlier stronger baselines remain in the supplementary material, so this comparison does not imply general classification superiority. Supplementary material records configurations, per-class results, runtime variation and negative findings.')
    (DEST/'cover_letter.tex').write_text(cover,encoding='utf-8')
    changes={
      'main_original_sha256':sha(OLD/'latex/main.tex'), 'main_revised_sha256':sha(DEST/'main.tex'),
      'supplement_original_sha256':sha(OLD/'latex/supplement.tex'),'supplement_revised_sha256':sha(DEST/'supplement.tex'),
      'original_supplement_tables_preserved':114,'supplement_tables':128,'main_tables':len(tabs(m)),
      'main_comparator':'Gaussian Naive Bayes; default and G-selected',
      'prior_comparators_retained_in_supplement':True,'new_training_in_revision':False,'reference_WV_changed':False,
      'no_new_GNB_runtime_XAI_external_result':True,'source_latex_not_pdf_conversion':True}
    write(OUT/'source_revision.json',changes)
    for name in ['main','supplement','cover_letter']:
        import difflib
        (OUT/(name+'.diff')).write_text(''.join(difflib.unified_diff(read(OLD/'latex'/(name+'.tex')).splitlines(True),read(DEST/(name+'.tex')).splitlines(True),fromfile='17Nov/'+name,tofile='19Nov/'+name)),encoding='utf-8')
    print(json.dumps(changes,indent=2))

if __name__=='__main__': main()
