"""Integrate the traceable initial study and later extensions, without new fitting."""
from pathlib import Path
import csv
import hashlib
import inspect
import json
import re
import shutil
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_array_short import PREAMBLE, AUTHORS, body, table as render_table
from build_array_complete import verified, RECORDS, sha, table
OLD = ROOT / 'paper/array_20261022_complete'
BASE = ROOT / 'paper/array_20261022_integrated'
SRC = BASE / 'elsarticle'
FULL = Path('runs/FULL_20261013_01')
CLASSES = ['Normal', 'DDoS_TCP', 'Port_Scanning', 'SQL_injection', 'Ransomware']
METHOD = {'soft_voting': 'Soft', 'weighted_voting': 'Weighted', 'stacking': 'Stacking'}


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    p = BASE / 'evidence' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def retained(rel):
    p = verified(rel)
    target = BASE / 'evidence/historical_initial' / p.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, target)
    return p


def figure_flow():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    ax.set(xlim=(0, 10), ylim=(-.2, 7)); ax.axis('off')
    def box(x, y, w, h, title, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                   linewidth=.8, edgecolor='#4c6170', facecolor=color))
        ax.text(x+w/2, y+h-.22, title, ha='center', va='top', weight='bold', fontsize=9)
        ax.text(x+w/2, y+.18, text, ha='center', va='bottom', fontsize=8, linespacing=1.25)
    def arrow(x1,y1,x2,y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops={'arrowstyle':'->','color':'#526573','lw':1.05})
    box(.15, 5.35, 4.3, 1.35, '1. Initial comparison',
        '28/29 fields; nested CV (5 outer / 2 inner)\nOriginal / Balanced; Soft / Weighted / Stacking', '#eaf0f6')
    box(5.1, 5.35, 4.7, 1.35, '2. Historical recommendation',
        '29 fields; six eligible ensembles\nBalanced weighted voting wins composite score', '#eaf0f6')
    arrow(4.58,6.05,4.98,6.05)
    box(.15, 3.5, 9.65, 1.1, '3. Diagnose limits; reset the evaluation question',
        'Port/signature sensitivity; LIFT-ID and other alternatives retained\nHistorical holdout ranking is exploratory; scores do not transfer across splits', '#fff1dd')
    arrow(7.45,5.24,7.45,4.72)
    box(.15, 1.72, 9.65, 1.1, '4. Matched representation study',
        'Prior-only symmetric endpoint context: short40 vs multiscale60\nTemporal A/G/C/V + signature diagnostic; expert/uniform controls; class costs', '#e7f3ef')
    arrow(5,3.39,5,2.93)
    box(.15, .02, 4.45, 1.03, '5. Same fitted predictor: execution',
        'Shared compact encoding + certified labels\nLatency / replay / inference delta RSS', '#e8eef9')
    box(5.0, .02, 4.8, 1.03, '6. Full probabilities: explanation audit',
        'Grouped SHAP; deletion/insertion; stability\nDisjoint donors; runtime equivalence', '#e8eef9')
    arrow(3,1.6,2.45,1.16); arrow(7,1.6,7.4,1.16)
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)
    fig.savefig(SRC/'figures/figure0_study_flow.pdf', bbox_inches='tight')
    fig.savefig(SRC/'figures/figure0_study_flow.png', bbox_inches='tight', dpi=190)
    plt.close(fig)


def figure_initial(shap_rows, cv_rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(6.6,2.8),constrained_layout=True)
    for ax, metric, title in zip(axes,['macro_f1_mean','ddos_f2_mean'],['Macro-F1','DDoS F2']):
        for offset,variant,color in [(-.18,'baseline_search','#456681'),(.18,'ddos_tuned','#149080')]:
            vals=[100*statistics.mean(float(r[metric]) for r in cv_rows
                                      if r['feature_mode']==mode and r['variant']==variant)
                  for mode in ['bands28','dstport29']]
            ax.bar([offset,1+offset],vals,width=.34,label='Initial grid' if variant=='baseline_search' else 'Expanded DDoS grid',color=color)
        ax.set_xticks([0,1],['28 fields','29 fields']);ax.set_ylim(0,100)
        ax.set_ylabel(title+' (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[1].legend(loc='lower right',fontsize=8)
    fig.savefig(SRC/'figures/figure8_historical_features.pdf',bbox_inches='tight')
    fig.savefig(SRC/'figures/figure8_historical_features.png',bbox_inches='tight',dpi=190);plt.close(fig)
    chosen=sorted([r for r in shap_rows if r['config']=='balanced__weighted_voting'
                   and r['output_class']=='ALL_MEAN_ABS'],key=lambda r:float(r['mean_abs_shap']),reverse=True)[:10]
    assert len(chosen)==10
    fig,ax=plt.subplots(figsize=(6.3,3.1),constrained_layout=True)
    ax.barh(range(10),[float(r['mean_abs_shap']) for r in chosen],color='#456681')
    ax.set_yticks(range(10),[r['feature'] for r in chosen]);ax.invert_yaxis()
    ax.set_xlabel('Stored mean absolute KernelSHAP attribution');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(SRC/'figures/figure9_historical_shap.pdf',bbox_inches='tight')
    fig.savefig(SRC/'figures/figure9_historical_shap.png',bbox_inches='tight',dpi=190);plt.close(fig)


def main():
    assert not BASE.exists() or '--refresh' in sys.argv, 'Explicit --refresh is required for an unfinished revision'
    assert not (ROOT/'penyerahan/ARRAY_HAFID_ALUR_LENGKAP_20261022.zip').exists(), 'Published revisions are immutable'
    prior=json.loads((ROOT/'penyerahan/ARRAY_HAFID_REVISI_LENGKAP_20261022.verification.json').read_text())
    for r in prior['pdfs']: assert sha(ROOT/r['path'])==r['sha256']
    assert sha(ROOT/prior['archive']['path'])==prior['archive']['sha256']
    shutil.copytree(OLD,BASE,dirs_exist_ok=True,ignore=shutil.ignore_patterns('quality_check.json','__pycache__'))
    previous_build=json.loads((BASE/'editorial_build.json').read_text())
    t=json.loads((BASE/'tables.json').read_text())
    selection=read(retained(FULL/'feature_mode_selection.csv'))
    cv=read(retained(FULL/'cv_feature_comparison.csv'))
    for rel in ['cv_feature_folds.csv','paired_feature_effects.csv','feature_mode_freeze.json',
                'comparison_protocol.json','config_requested.json','preflight.json',
                'artifact_verification.json','runner_source/ta_hafid_engine.py',
                'dstport29/tables/holdout_all_six.csv','dstport29/profiling/latency_summary.csv',
                'dstport29/profiling/memory_summary.csv','dstport29/manifests/fit_balanced.json',
                'dstport29/manifests/fit_original.json']:
        retained(FULL/rel)
    ranking=read(retained(Path('reports/REKOMENDASI_20261015_01/ranking_all_six.csv')))
    policy=json.loads(retained(Path('reports/REKOMENDASI_20261015_01/recommendation.json')).read_text())
    assert policy['revision_after_observation'] and not policy['main_rule_frozen_before_holdout']
    assert policy['winners']==['balanced__weighted_voting']
    assert policy['unchanged_scores'] and not policy['new_training']
    t['initial_features']=table('Initial development selection of port representations: means across six DDoS-tuned ensemble configurations (%).',
        ['Domain fields','Macro-F1','DDoS F2','DDoS recall','Selection score'],
        [[r['feature_mode'],f"{100*float(r['macro_f1_mean_across_six']):.4f}",f"{100*float(r['ddos_f2_mean_across_six']):.4f}",
          f"{100*float(r['ddos_recall_mean_across_six']):.4f}",f"{float(r['score']):.6f}"] for r in selection],
        [110,90,90,100,95],
        'bands28 adds three categorical port bands to packet25; dstport29 also adds raw TCP destination port. Both share all outer/inner folds. The selection score is the equal-weight mean of macro-F1 and DDoS F2 on development CV, not an independent performance estimate.')
    label=lambda r:r['case'].title()+' '+METHOD[r['method']]
    t['initial_six']=table('Historical holdout ranking after removal of the Original-priority policy (15 October 2026).',
        ['Rank / training / ensemble','F1 (%)','F2 DDoS (%)','Batch ms','Delta MiB','Score'],
        [[r['rank_all_six']+'. '+label(r),f"{100*float(r['macro_f1']):.4f}",f"{100*float(r['ddos_f2']):.4f}",
          f"{float(r['latency_batch_ms']):.4f}",f"{float(r['delta_peak_mib']):.4f}",f"{float(r['tradeoff_score']):.6f}"] for r in ranking],
        [150,78,88,82,79,90],
        'N=13,171; original class distribution. Batch=1,000 observations; delta MiB=inference delta peak RSS. All six are eligible. The composite is 0.6 performance + 0.4 efficiency. The recommendation policy changed after holdout observation, without changing models or scores; this is an exploratory historical ranking.')
    class_rows=[]
    for r in ranking:
        rows=read(retained(FULL/f"dstport29/tables/holdout_per_class_{r['config']}.csv"))
        class_rows.append([label(r)]+[f"{100*float(next(x for x in rows if x['']==c)['recall']):.4f}" for c in CLASSES])
    t['historical_six']=table('Historical holdout recall for all five classes and six ensembles (%).',
        ['Training / ensemble','Normal','DDoS','Scan','SQL','Ransom'],class_rows,[155,80,80,80,80,80],
        'Class supports are 4,860 / 2,050 / 2,014 / 2,062 / 2,185 in column order. Main Table 3 supplies the overall ranking. These recalls belong to the earlier random holdout and are not matched with later temporal/signature results.')
    t['historical_full']['note']=t['historical_full']['note'].replace('Complements Table S12, which supplies macro-F1, latency, inference delta RSS and the final composite score.',
        'Complements main Table 3 (F1, F2, latency, inference delta RSS and composite score) and Table S12 (all five class recalls).')
    avg=[]
    for mode in ['bands28','dstport29']:
        for variant in ['baseline_search','ddos_tuned']:
            rows=[r for r in cv if r['feature_mode']==mode and r['variant']==variant];assert len(rows)==6
            avg.append(dict(feature_mode=mode,variant=variant,**{k:statistics.mean(float(r[k]) for r in rows)
                for k in ['macro_f1_mean','ddos_f2_mean','ddos_recall_mean','port_scanning_recall_mean']}))
    write_csv('historical_initial/baseline_tuned_averages.csv',avg)
    t['initial_tuning']=table('Initial versus expanded DDoS tuning: arithmetic means across all six configurations (%).',
        ['Fields / search','Macro-F1','DDoS F2','DDoS recall','Scan recall'],
        [[r['feature_mode']+' / '+r['variant'],*[f"{100*r[k]:.4f}" for k in ['macro_f1_mean','ddos_f2_mean','ddos_recall_mean','port_scanning_recall_mean']]] for r in avg],
        [165,80,80,80,80],
        'Initial grid has two candidates per expert; expanded grid has eight including the initial candidates. Candidate choice uses inner-fold results; reported ensemble scores use the five outer folds. The selected feature mode is still a development choice.')
    for mode in ['bands28','dstport29']:
        rows=[r for r in cv if r['feature_mode']==mode]
        t['initial_cv_'+mode]=table('All initial ensemble development results for '+mode+' (%; mean +/- standard deviation across five outer folds).',
            ['Training / ensemble / search','Macro-F1','DDoS F2','DDoS recall'],
            [[r['case'].title()+' '+METHOD[r['method']]+' / '+('initial' if r['variant']=='baseline_search' else 'expanded'),
              *[f"{100*float(r[k+'_mean']):.4f} +/- {100*float(r[k+'_std']):.4f}" for k in ['macro_f1','ddos_f2','ddos_recall']]] for r in rows],
            [165,110,110,110],
            'Original/Balanced training, Soft/Weighted/Stacking, and both search grids are retained. Validation folds keep the original observations. Fold dispersion is not a confidence interval over independent captures; row-level scores remain in the CSV.')
    shap_rows=read(retained(FULL/'dstport29/shap/global_and_per_output_class.csv'))
    retained(FULL/'dstport29/shap/local_predictions_additivity.csv')
    shap_meta=json.loads(retained(FULL/'dstport29/manifests/shap_complete.json').read_text())
    assert shap_meta['global_rows']==50 and shap_meta['background_rows']==20 and shap_meta['local_rows']==11
    figure_flow();figure_initial(shap_rows,cv)

    md=(OLD/'main_body.md').read_text(encoding='utf-8')
    md=re.sub(r'Table ([2-6])\b',lambda m:'Table '+str(int(m[1])+2),md)
    md=md.replace('This study asks whether a bounded history',
        'An initial study compared six ensemble configurations and selected Balanced weighted voting by a recorded performance-efficiency score. The subsequent research asks whether a bounded history',1)
    md=md.replace('## Data and evaluation protocol','## Study design and historical comparison',1)
    start=md.index('The canonical five-class subset contains')
    end=md.index('\n\nFor the temporal protocol',start)
    md=md[:start]+'''The canonical five-class subset contains 65,855 targets, with a fixed seed-42 stratified 80:20 partition: 52,684 development and 13,171 historical holdout observations. The classes are Normal, DDoS TCP SYN flood, Port Scanning, SQL injection and Ransomware. Five existing sources contain 3,735,488 packets. All canonical Normal targets come from Temperature/Humidity traffic, with one source for each attack. SHA256 manifests link data, splits, saved results and implementation; source counts appear in Supplementary Tables S1-S2.

### Initial feature, tuning and ensemble comparison

The initial study compares 28 and 29 domain fields on development data only. Both start from ten numeric and fifteen categorical packet fields. Three categorical TCP-source/TCP-destination/UDP port bands yield 28 fields; adding raw numeric TCP destination port yields 29. Port bands distinguish inactive, well-known, registered and dynamic values, with invalid/missing values separate. Labels, endpoint identities, timestamps and selected high-cardinality payload/metadata fields are excluded from the initial predictors.

Identical five-outer/two-inner stratified folds compare DT, RF, XGB, LGB and LR under Original and Balanced training. Fold training alone fits imputation, encoding, LR scaling and, for Balanced, RUS+SMOTE with the recorded 50% per-original-class synthetic cap. Validation/holdout observations remain original. Two candidates per expert form the initial grid, selected by inner macro-F1; an expanded eight-candidate grid includes those baselines and selects by 0.5 macro-F1 + 0.5 DDoS F2, subject to a macro-F1 floor 0.01 below the best baseline. Uniform soft voting, quality/cost weighted voting and LR stacking over inner out-of-fold probabilities give six ensembles per grid. Historical stacking uses C=0.1 and a 1,000-iteration limit; later context experiments use their separately specified expert settings.

Table 2 summarizes expanded-grid outer-CV means used for feature selection. The score averages macro-F1 and DDoS F2 across the six configurations; both satisfy the recorded eligibility floor, and 29 fields are selected without holdout use. With 29 fields, expanding the search changes mean macro-F1 by only +0.0116 percentage points and DDoS F2 by -0.0035 points (Supplementary Tables S22-S24). Thus the strong port-associated result under this random-fold protocol, rather than a substantial DDoS-tuning gain, motivates the later distribution diagnostics.

{{TABLE:initial_features}}

Each selected 29-field configuration is refitted on development data and evaluated on the historical holdout (Table 3). Performance averages macro-F1, macro recall and macro one-vs-rest AUC. Efficiency averages minimum/current batch latency and minimum/current inference delta peak RSS; the final score weights performance/efficiency 0.6/0.4. These are operational research choices, not universal weights. The all-six policy removes an earlier Original priority after holdout observation (15 October), retaining unchanged predictions and measurements. Balanced weighted voting ranks first at 0.974691; its 95.5179% macro-F1 ties three other ensembles, and Balanced stacking has slightly higher DDoS F2. The selected reference is therefore a composite-score choice, not a universal accuracy or speed winner.

{{TABLE:initial_six}}

### Development protocols for the extensions

The extensions return to the existing 52,684 development targets. Historical holdout identifiers are excluded from new fitting, selection, predictive scoring and profiling. Figure 1 connects the initial comparison to the diagnostic, representation, execution and explanation stages. Neither the new temporal score nor the signature score can be subtracted from the historical random-holdout score as a matched gain.

{{FIGURE:figure0_study_flow|Complete study flow, from the initial feature/tuning comparison and historical ensemble selection to the later context, execution and explanation investigations. The final two branches use the same fitted weighted predictor: certified labels for execution, full probabilities for explanation. Each evaluation retains its own split, measurement session and scope. OpenAI Codex (OpenAI; exact build not recorded) assisted the programmatic diagram code.}}'''+md[end:]
    old_fig=re.search(r'\{\{FIGURE:figure1_pipeline\|.*?\}\}',md).group(0)
    md=md.replace(old_fig+'\n\n','',1)
    md=md.replace('the architecture caption discloses diagram assistance','the workflow caption discloses diagram assistance')
    supp=(OLD/'supplement_body.md').read_text(encoding='utf-8')
    supp=supp.replace('exact counts above permit direct recovery of every recall',
                      'the exact counts in Table S15 permit direct recovery of every recall')
    supp=supp.replace('The following six-configuration ranking belongs to the previously observed historical holdout. It is preserved for project continuity and dashboard interpretation.',
        'Main Table 3 gives the six-configuration historical ranking; Table S12 below adds all five class recalls. These belong to the previously observed random holdout and preserve the initial study for project continuity.')
    supp+='''

## Initial study: protocol, tuning and descriptive SHAP

The initial full run is FULL_20261013_01; its Original-priority recommendation is historical. REKOMENDASI_20261015_01 removes that priority after observation without retraining, reprofiling or changing scores. Both manifests and the frozen initial engine are retained. These local artifacts support this chronology; absent earlier-version runs are not reconstructed from conversation.

The initial development/holdout class supports are 19,441/4,860 Normal, 8,197/2,050 DDoS, 8,057/2,014 Scan, 8,249/2,062 SQL and 8,740/2,185 Ransomware. Outer StratifiedKFold uses five shuffled folds at seed 42; inner StratifiedKFold uses two shuffled folds at seed 59. Both feature modes and training treatments share fold assignments. Numeric medians, missing-category replacement, one-hot vocabulary and LR scaling are fitted within the corresponding training fold. LR scales numeric columns without centering. Raw TCP destination port accepts valid integers 0-65,535; band boundaries are 0, 1-1,023, 1,024-49,151 and 49,152-65,535, with an invalid/missing category. The frozen engine and saved feature-freeze manifest retain exact field/transform definitions.

The initial RUS+SMOTE target scales the configured 10,000-per-class reference with the training partition size, then applies the recorded cap. Each class may add at most floor(0.5 times its pre-RUS count), and the final synthetic fraction must also be at most 0.5. The cap can prevent equal final class counts. Inner/outer validation and historical holdout never enter resampling. Initial weighted voting uses inner out-of-fold macro-F1, latency/serialized-pipeline-size proxies, coefficients 0.5/0.6/0.7, half-shrinkage toward uniform and a 0.05 weight floor. Alpha choice follows the applicable inner quality objective, with ties nearest 0.6. Stacking fits an LR meta-learner on concatenated inner out-of-fold probabilities before refitting experts. These procedures differ from the subsequent temporal A/G/C/V workflow and do not share its evaluation scores.

{{TABLE:initial_tuning}}

{{TABLE:initial_cv_bands28}}

{{TABLE:initial_cv_dstport29}}

{{FIGURE:figure8_historical_features|Initial-grid and expanded-grid development means across all six configurations. Both use the same outer folds. The near-overlapping bars show that expanded tuning contributes little relative to the 28/29 port-representation difference under this historical random-fold protocol. These are model-selection observations, not an independently held-out feature comparison.}}

## Initial descriptive explanation archive

Initial KernelSHAP is descriptive: 50 class-stratified training targets, 20 training-background rows and nsamples=256, across all six ensembles. Eleven shared illustrative holdout cases come from the frozen Original Soft Voting reference's correct/error cases and DDoS-to-Scan errors; they are not a random population estimate. The recorded maximum additivity residual is 2.22e-16. Additivity alone establishes neither faithfulness nor robustness. Figure S5 reproduces the stored global top-ten attribution values for Balanced weighted voting. Its individual-feature KernelSHAP output is distinct from the later exact seven-group donor game; their magnitudes must not be combined. The full six-configuration/output-class attribution CSV and local prediction/additivity CSV accompany this supplement.

{{FIGURE:figure9_historical_shap|Descriptive initial KernelSHAP: ten largest stored ALL_MEAN_ABS values for the historical 29-field Balanced weighted ensemble. The aggregation is across output classes and 50 training examples as defined by the saved run. This is not the later predicted-class grouped audit and is not a causal or quantitative faithfulness result.}}

'''+old_fig+'\n'
    # In insertion order the previous figures are S1/S2, then feature chart S3,
    # initial SHAP S4 and detailed architecture S5. Match the text explicitly.
    supp=supp.replace('Figure S5 reproduces the stored global','Figure S4 reproduces the stored global')
    (BASE/'main_body.md').write_text(md,encoding='utf-8')
    (BASE/'supplement_body.md').write_text(supp,encoding='utf-8')
    (BASE/'tables.json').write_text(json.dumps(t,indent=2)+'\n')
    oldtex=(OLD/'elsarticle/main.tex').read_text(encoding='utf-8')
    abstract=oldtex.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
    abstract=abstract.replace('We evaluate', 'We evaluate',1)
    # Keep the abstract focused on the final contribution; the complete origin
    # is explained in Section 3 and Figure 1 rather than crowding this summary.
    preamble=PREAMBLE.replace(r'\usepackage[hidelinks]{hyperref}',r'\usepackage{float}'+'\n'+r'\usepackage[hidelinks]{hyperref}')
    tex=preamble+r'\begin{frontmatter}\title{Multiscale endpoint context and prediction preserving execution for explainable IoT intrusion detection}'+'\n'+AUTHORS
    tex+=r'\begin{abstract}'+abstract+r'\end{abstract}'+'\n'+r'\begin{keyword}IoT intrusion detection \sep endpoint context \sep weighted voting \sep early exit \sep SHAP \sep reproducibility\end{keyword}\end{frontmatter}'+'\n'
    tex+=r'\noindent\textit{Research article draft for author review, 22 October 2026. Initial and extension experiments retain distinct evaluation scopes. Technical details are provided as separate supplementary material.}'+'\n\n'
    tex+=body(md,t)+r'\FloatBarrier\bibliographystyle{elsarticle-num}\bibliography{references}\end{document}'
    st=render_table('robustness_changed',t['robustness_changed'])
    tex=tex.replace(st,st.replace('[!htbp]','[H]',1))
    tex=tex.replace(r'\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/figure0_study_flow.pdf}',
                    r'\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/figure0_study_flow.pdf}',1)
    tex=tex.replace(r'\section{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}',
                    r'\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}')
    (SRC/'main.tex').write_text(tex,encoding='utf-8')
    stex=preamble+r'\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\theHtable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}'
    stex+=r'\begin{center}{\Large\bfseries Supplementary material}\par\medskip M Hafid Ramadhan and Satria Mandala\par\smallskip 22 October 2026\end{center}'+'\n\n'+body(supp,t)+r'\FloatBarrier\end{document}'
    (SRC/'supplement.tex').write_text(stex,encoding='utf-8')
    records={r['path']:r for r in previous_build['source_records']}
    records.update({r['path']:r for r in RECORDS.values()})
    build=dict(date='2026-10-22',operational_main_page_target=23,
        page_target_authority='Editorial interpretation of user permission for a few extra pages, not an official journal/supervisor numeric rule',
        original_font_pt=12,original_margins=True,new_training=False,model_deserialization=False,new_game_execution=False,
        new_analyses='Traceable historical result integration and saved-data figures only',
        source_records=list(records.values()),previous_deliverables=prior,
        main_tables=len(re.findall(r'\{\{TABLE:',md)),main_figures=len(re.findall(r'\{\{FIGURE:',md)),
        supplementary_tables=len(re.findall(r'\{\{TABLE:',supp)),supplementary_figures=len(re.findall(r'\{\{FIGURE:',supp)),
        previous_build_preserved=previous_build,reference_metadata_entries=25)
    (BASE/'editorial_build.json').write_text(json.dumps(build,indent=2)+'\n')
    standalone='''"""Reproduce the three integrated-study figures from packaged CSVs only."""
from pathlib import Path
import argparse, csv, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BASE=Path(__file__).resolve().parent
'''
    standalone+=inspect.getsource(figure_flow)+'\n'+inspect.getsource(figure_initial)
    standalone+='''
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();(args.output_dir/'figures').mkdir(parents=True,exist_ok=True)
    SRC=args.output_dir
    def read_local(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    data=BASE/'evidence/historical_initial/runs/FULL_20261013_01'
    figure_flow()
    figure_initial(read_local(data/'dstport29/shap/global_and_per_output_class.csv'),
                   read_local(data/'cv_feature_comparison.csv'))
    print('Reproduced workflow, initial comparison and initial SHAP figures; no fitting or model loading.')
'''
    (BASE/'reproduce_initial_figures.py').write_text(standalone,encoding='utf-8')
    print(json.dumps({k:v for k,v in build.items() if k not in ['source_records','previous_deliverables','previous_build_preserved']},indent=2))


if __name__=='__main__':main()
