"""Complete the article from checksum-verified saved results; no model loading."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import statistics
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_array_short import PREAMBLE, AUTHORS, body, table as render_table
from build_array_manuscript import inline
OLD = ROOT / 'paper/array_20261022_20pages'
BASE = ROOT / 'paper/array_20261022_complete'
SRC = BASE / 'elsarticle'
CLASSES = ['Normal', 'DDoS_TCP', 'Port_Scanning', 'SQL_injection', 'Ransomware']
LABELS = ['Normal', 'DDoS', 'Scanning', 'SQL', 'Ransomware']
METHODS = ['decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'soft_voting', 'weighted_voting']
SHORT = dict(zip(METHODS, ['DT', 'RF', 'XGB', 'LGB', 'LR', 'Soft', 'WB']))
RECORDS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    if path in RECORDS: return path
    for parent in path.parents:
        if not parent.is_relative_to(ROOT): break
        catalog = parent / 'artifacts_sha256.json'
        if not catalog.exists(): continue
        entries = json.loads(catalog.read_text())
        if isinstance(entries, dict): entries = entries.get('files', entries.get('artifacts', []))
        if not isinstance(entries, list): continue
        relative = path.relative_to(parent).as_posix()
        match = next((r for r in entries if r['path'].replace('\\', '/') == relative), None)
        if match:
            assert sha(path) == match['sha256'], str(path)
            RECORDS[path] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                                 catalog=catalog.relative_to(ROOT).as_posix(), catalog_sha256=sha(catalog))
            return path
    raise AssertionError('No matching catalog record: ' + str(path))


def read_csv(path, verify=False):
    path = verified(path) if verify else Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def csv_out(name, data):
    path = BASE / 'evidence' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)


def table(caption, headers, rows, widths, note):
    return dict(caption=caption, headers=headers, rows=rows, widths=widths, note=note)


def metrics(cm):
    support = cm.sum(1)
    recall = cm.diagonal() / support
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(5), where=cm.sum(0) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) != 0)
    tp = cm[1, 1]; fn = support[1] - tp; fp = cm[:, 1].sum() - tp
    return dict(n=int(cm.sum()), macro_f1=float(f1.mean()), ddos_f2=float(5 * tp / (5 * tp + 4 * fn + fp)),
                **{'recall_' + c: float(v) for c, v in zip(CLASSES, recall)})


def matrix_from_rows(data, view):
    cm = np.zeros((5, 5), dtype=np.int64)
    subset = [r for r in data if r['view'] == view and r['case'] == 'balanced' and r['method'] == 'weighted_voting']
    assert len(subset) == 25
    for row in subset: cm[CLASSES.index(row['true']), CLASSES.index(row['predicted'])] = int(row['count'])
    assert cm.sum() == 52584
    return cm


def save_figure(fig, name):
    fig.savefig(SRC / 'figures' / (name + '.pdf'), bbox_inches='tight', metadata={'Creator': 'Matplotlib; reproducible saved-result analysis'})
    fig.savefig(SRC / 'figures' / (name + '.png'), dpi=190, bbox_inches='tight')
    plt.close(fig)


def main():
    original = json.loads((OLD / 'evidence/evidence.json').read_text())
    for item in original['source_records']: assert sha(ROOT / item['path']) == item['sha256']
    prior = json.loads((ROOT / 'penyerahan/ARRAY_HAFID_REVISI_MAKS20_20261022.verification.json').read_text())
    for item in prior['pdfs']: assert sha(ROOT / item['path']) == item['sha256']
    assert sha(ROOT / prior['archive']['path']) == prior['archive']['sha256']
    SRC.mkdir(parents=True, exist_ok=True)
    shutil.copytree(OLD / 'elsarticle', SRC, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.aux', '*.out', '*.log', '*.blg', '*.pdf'))
    # Template documentation PDFs and the four original figures remain unmodified.
    inv = json.loads((OLD / 'template_inventory.json').read_text())
    for rec in inv['files']:
        source = OLD / rec['path']; assert sha(source) == rec['sha256']
        target = BASE / rec['path']; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    for source in (OLD / 'elsarticle/figures').glob('*.pdf'):
        shutil.copy2(source, SRC / 'figures' / source.name)
    shutil.copytree(OLD / 'evidence', BASE / 'evidence', dirs_exist_ok=True)
    shutil.copy2(OLD / 'template_inventory.json', BASE / 'template_inventory.json')
    shutil.copy2(ROOT / 'paper/array_20261021/references.json', BASE / 'references.json')
    t = json.loads((OLD / 'tables.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    data = read_csv(verified('runs/MULTISCALE_REPAIR_FULL_20261020_01/tables/signature_confusions.csv'))
    short_cm = matrix_from_rows(data, 'baseline25')
    full_cm = matrix_from_rows(data, 'legacy25_multiscale60')
    evaluated = verified('runs/MULTISCALE_REPAIR_FULL_20261020_01/stages/temporal_seed42/legacy25_multiscale60__balanced/evaluation.npz')
    with np.load(evaluated, allow_pickle=False) as arr:
        y = arr['y']; predicted = arr['p_weighted'].argmax(1)
        temporal_cm = np.zeros((5, 5), dtype=np.int64)
        np.add.at(temporal_cm, (y, predicted), 1)
        assert np.array_equal(predicted, arr['certified_labels'])
        temporal_ids = arr['row_ids'].copy(); weighted = arr['p_weighted'].copy()
        expert_p = arr['p_experts'].copy()
    result_records = [dict(protocol='temporal_seed42', **metrics(temporal_cm)), dict(protocol='signature_pooled', **metrics(full_cm))]
    stage = read_csv(BASE / 'evidence/multiscale_metrics_all.csv')
    expected = next(r for r in stage if r['stage'] == 'temporal_seed42' and r['view'] == 'legacy25_multiscale60'
                    and r['case'] == 'balanced' and r['condition'] == 'original' and r['method'] == 'weighted_voting')
    assert abs(result_records[0]['macro_f1'] - float(expected['macro_f1'])) < 1e-12
    assert abs(result_records[1]['macro_f1'] - .9729551440176138) < 1e-12
    csv_out('class_recall_reference.csv', result_records)
    t['class_recall'] = table('Five-class recall of the fixed Balanced multiscale weighted predictor (%).',
        ['Evaluation', 'N', 'Normal', 'DDoS', 'Scan', 'SQL', 'Ransom'],
        [[r['protocol'].replace('_', ' '), f"{r['n']:,}"] + [f"{100*r['recall_'+c]:.4f}" for c in CLASSES] for r in result_records],
        [111, 56, 64, 64, 64, 64, 64],
        'Temporal is the seed 42 reference (macro-F1 97.8800%), not the three-seed mean (97.8954%). Signature pools the two diagnostic folds; their shared history is not session isolation.')
    matrices = dict(short_signature=short_cm, multiscale_signature=full_cm, multiscale_temporal_seed42=temporal_cm)
    exact = []
    for name, cm in matrices.items():
        for i in range(5):
            for j in range(5): exact.append(dict(matrix=name, true=CLASSES[i], predicted=CLASSES[j], count=int(cm[i, j])))
    csv_out('confusion_matrices_selected.csv', exact)
    tick = ['Normal', 'DDoS', 'Port scan', 'SQL inj.', 'Ransom']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.05), constrained_layout=True)
    for ax, cm, title in zip(axes, [short_cm, full_cm], ['25 + symmetric40', '25 + multiscale60']):
        perc = 100 * cm / cm.sum(1, keepdims=True)
        im = ax.imshow(perc, cmap='Blues', vmin=0, vmax=100, aspect='equal')
        ax.set_xticks(range(5), tick, rotation=35, ha='right'); ax.set_yticks(range(5), tick)
        ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(title, fontsize=10)
        for i in range(5):
            for j in range(5): ax.text(j, i, f'{perc[i,j]:.1f}', ha='center', va='center', fontsize=9,
                                      color='white' if perc[i,j] > 55 else '#142438')
    fig.colorbar(im, ax=axes, label='Row-normalized count (%)', shrink=.80, pad=.02)
    save_figure(fig, 'figure5_confusion')
    # Grouped SHAP is summarized from finite games, not refitted or re-explained.
    xr = Path('runs/XAI_MULTISCALE_FULL_20261020_01')
    meta = json.loads(verified(xr / 'groups/A32/metadata.json').read_text())
    game_path = verified(xr / 'groups/A32/coalitions.npz')
    importance = []
    with np.load(game_path, allow_pickle=False) as game:
        groups = list(meta['groups'])
        assert len(groups) == 7 and len(game['row_ids']) == 100
        position = {int(r): i for i, r in enumerate(temporal_ids)}
        indices = [position[int(r)] for r in game['row_ids']]
        assert np.max(np.abs(game['values'][:, -1, 0, :] - weighted[indices])) < 1e-12
        for method, index, cached in [('weighted_voting', 0, weighted[indices]),
                                      ('random_forest', 2, expert_p[indices, 1, :]),
                                      ('logistic_regression', 5, expert_p[indices, 4, :])]:
            assert meta['output_order'][index] == method
            assert np.max(np.abs(game['values'][:, -1, index, :] - cached)) < 1e-12
            explained = game['values'][:, -1, index, :].argmax(1)
            phi = np.take_along_axis(game['phi'][:, :, index, :], explained[:, None, None], axis=2).squeeze(2)
            base = game['values'][np.arange(100), 0, index, explained]
            full = game['values'][np.arange(100), -1, index, explained]
            assert np.max(np.abs(phi.sum(1) + base - full)) < 1e-10
            means = np.abs(phi).mean(0)
            for group, value in zip(groups, means):
                importance.append(dict(method=method, group=group, mean_abs_phi=float(value),
                                       share_of_mean_abs=float(value / means.sum()), n=100, background='A32'))
    csv_out('grouped_shap_importance.csv', importance)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True, gridspec_kw={'width_ratios': [1.35, 1]})
    colors = ['#234e70', '#159895', '#d57a36']; methods = ['weighted_voting', 'random_forest', 'logistic_regression']
    group_labels = [g.upper() if g != 'endpoint_history' else 'History (60 fields)' for g in groups]
    for j, (method, color) in enumerate(zip(methods, colors)):
        vals = [r['mean_abs_phi'] for r in importance if r['method'] == method]
        axes[0].barh(np.arange(7) + (j-1)*.23, vals, height=.22, color=color, label={'weighted_voting':'WB','random_forest':'RF','logistic_regression':'LR'}[method])
        axes[1].barh(np.arange(6) + (j-1)*.23, vals[:6], height=.22, color=color)
    for ax in axes:
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True); ax.invert_yaxis()
        ax.set_xlabel('Mean absolute grouped Shapley value')
    axes[0].set_yticks(range(7), group_labels); axes[0].legend(loc='upper right'); axes[0].set_title('All seven groups')
    axes[1].set_yticks(range(6), group_labels[:6]); axes[1].set_title('Packet groups (linear-axis zoom)')
    save_figure(fig, 'figure6_group_importance')
    robust = read_csv(BASE / 'evidence/robustness.csv')
    chosen = [next(r for r in robust if r['variant']=='query_delay_5ms' and r['subset']=='changed_input' and r['method']==m) for m in methods]
    t['robustness_changed'] = table('Query-delay explanation stability on the 33 changed inputs.',
        ['Predictor', 'Label retained (%)', 'Top-3 Jaccard', 'Normalized L1'],
        [[{'weighted_voting':'Weighted voting','random_forest':'Random forest','logistic_regression':'Logistic regression'}[r['method']],
          f"{100*float(r['prediction_unchanged']):.1f}", f"{float(r['top3_jaccard']):.4f}", f"{float(r['normalized_l1']):.6f}"] for r in chosen],
        [150, 112, 112, 113], 'The same 100 audit targets are queried 5 ms later, without future packets. Only 33 inputs change. Larger overlap and smaller distance indicate stability under this specific perturbation.')
    # Full recall and confusion tables complement the compact main article.
    controls = read_csv(BASE / 'evidence/same_feature_controls.csv')
    for temporal, key in [(True, 'recall_temporal_all'), (False, 'recall_signature_all')]:
        rs=[]
        for case in ['original', 'balanced']:
            for method in METHODS:
                row=next(r for r in controls if r['case']==case and r['method']==method)
                suffix='_mean' if temporal else ''
                rs.append([case.title()+' '+SHORT[method], f"{100*float(row['macro_f1'+suffix]):.2f}", f"{100*float(row['ddos_f2'+suffix]):.2f}"]
                          + [f"{100*float(row['recall_'+c+suffix]):.2f}" for c in CLASSES])
        t[key]=table(('Temporal three-seed mean' if temporal else 'Pooled signature')+' metrics with the same multiscale25 features (%).',
                     ['Predictor / training','F1','F2 DDoS','Norm.','DDoS','Scan','SQL','Rans.'],rs,[115,58,63,58,58,58,58,58],
                     'All five recalls are shown (Norm.=Normal; Rans.=Ransomware). Values are rounded to two decimals; full precision is in the accompanying CSV. Temporal seeds reuse the same validation targets; signature folds pool counts. Different protocols must not be combined as independent replicates.')
    cm_rows=[]
    for name,cm in [('Short',short_cm),('Multiscale',full_cm)]:
        for i in range(5): cm_rows.append([name+' / '+LABELS[i]]+[f'{int(v):,}' for v in cm[i]])
    t['confusion_counts']=table('Exact pooled signature confusion counts used in the main heatmaps.',
        ['True class / view','Normal','DDoS','Scan','SQL','Ransom'],cm_rows,[170,64,64,64,64,64],
        'Columns are predicted classes. Each view has 52,584 targets with identical class support. Short denotes packet25+symmetric40; Multiscale denotes packet25+multiscale60. Both use Balanced training.')
    historical=read_csv(ROOT/'reports/REKOMENDASI_20261015_01/ranking_all_six.csv')
    csv_out('historical_all_six_full.csv',historical)
    t['historical_full']=table('Full predictive and component scores of the six historical configurations.',
        ['Configuration','Acc. (%)','Recall (%)','AUC','Perf.','Eff.'],
        [[r['config'].replace('__',' '),f"{100*float(r['accuracy']):.4f}",f"{100*float(r['recall_macro']):.4f}",f"{float(r['auc_ovr_macro']):.6f}",f"{float(r['performance_score']):.6f}",f"{float(r['efficiency_score']):.6f}"] for r in historical],
        [170,70,70,70,80,80], 'Acc.=accuracy; Recall=macro recall; AUC=macro one-vs-rest ROC-AUC; Perf.=performance score; Eff.=efficiency score. Complements Table S12, which supplies macro-F1, latency, inference delta RSS and the final composite score. These observed historical holdout results are separate from new temporal/signature evaluations.')
    timings=read_csv(BASE/'evidence/latency_raw.csv')
    throughput=[]
    for policy in ['native_shared','native_certified','compact_shared','compact_certified','logistic_regression','random_forest']:
        r={'policy':policy}
        for batch in [1,1000]:
            x=[float(row['rows_per_second']) for row in timings if row['policy']=='baseline25__'+policy and int(row['batch'])==batch]
            assert len(x)==45
            r['batch'+str(batch)+'_rows_per_second']=statistics.median(x)
        throughput.append(r)
    csv_out('throughput_summary.csv',throughput)
    t['throughput']=table('Feature-input batch throughput from the recorded timing calls.',
        ['Policy','Batch 1 (rows/s)','Batch 1000 (rows/s)'],
        [[r['policy'].replace('_',' '),f"{r['batch1_rows_per_second']:.2f}",f"{r['batch1000_rows_per_second']:.2f}"] for r in throughput],[190,145,145],
        'Median of 45 recorded calls per policy/batch. Prepared features are already in memory; these rates exclude complete feature acquisition and do not measure every-packet operational throughput.')
    selection_path=verified('runs/MULTISCALE_REPAIR_FULL_20261020_01/stages/temporal_seed42/legacy25_multiscale60__balanced/selection_before_V.json')
    selection=json.loads(selection_path.read_text())
    (BASE/'evidence/reference_selection_summary.json').write_text(json.dumps(
        dict(source=selection_path.relative_to(ROOT).as_posix(),source_sha256=sha(selection_path),
             weights=selection['weights'],alpha=selection['alpha'],order=selection['order'],costs=selection['costs']),indent=2)+'\n')
    size_rows=[[SHORT[m],f'{size:,}','Recorded serialized pipeline proxy'] for m,size in zip(METHODS[:5],selection['costs']['serialized_pipeline_bytes'])]
    size_rows.append(['Reference bundle','644,528','Compressed saved joblib bundle'])
    t['serialized_sizes']=table('Serialized-size records for the reference model and inherited weighting heuristic.',
        ['Scope','Bytes','Definition'],size_rows,[130,90,260],
        'Pipeline proxies and the compressed whole-model bundle have different serialization definitions and must not be summed or treated as directly comparable deployed sizes. They are not inference RAM. The packet ring separately allocates 1,048,576 bytes.')
    t['robustness_full']=table('Complete saved query-delay and port-bijection robustness summary.',
        ['Variant / subset','Model','N','Label (%)','Jaccard','Norm. L1'],
        [[('Delay / ' if r['variant']=='query_delay_5ms' else 'Port / ')+('all' if r['subset']=='all' else 'changed'),
          SHORT[r['method']],r['n'],f"{100*float(r['prediction_unchanged']):.1f}" if r['prediction_unchanged'] else 'N/A',
          f"{float(r['top3_jaccard']):.4f}" if r['top3_jaccard'] else 'N/A',
          f"{float(r['normalized_l1']):.6f}" if r['normalized_l1'] else 'N/A'] for r in robust],
        [165,55,40,75,75,85], 'Port relabeling leaves all audited inputs unchanged. Zero changed inputs give an undefined changed-subset metric, not evidence of perfect robustness to altered inputs.')
    background=read_csv(BASE/'evidence/background_sensitivity.csv')
    t['background_full']=table('Attribution sensitivity between the two disjoint 32-row backgrounds.',
        ['Model','N','Mean top-3 Jaccard','Mean norm. L1','Mean cosine'],
        [[SHORT[m],'100']+[f"{statistics.mean(float(r[k]) for r in background if r['method']==m):.6f}" for k in ['top3_jaccard','normalized_l1','cosine']] for m in methods],
        [70,45,130,130,105], 'All 100 saved targets per method are summarized. Background variation changes the empirical intervention, rather than providing independent test captures.')
    # Historical alternative paths are preserved as an explicitly unmatched chronology.
    history_sources=[
        'reports/LIFT_ID_HASIL_20261018/E1_summary.csv', 'reports/LIFT_ID_HASIL_20261018/E2_holdout.csv',
        'runs/E4_FULL_20261019_01/tables/metrics.csv', 'runs/MPED_FULL_20261019_04/tables/mped_comparisons.csv',
        'runs/SBFE_FULL_20261019_01/tables/temporal_summary.csv', 'runs/SBFE_FULL_20261019_01/tables/signature_pooled.csv',
        'runs/PORT_FUSION_FULL_20261020_01/report/summary.json']
    hist_files=[]
    for rel in history_sources:
        p=verified(rel); target=BASE/'evidence/history'/p.parent.parent.name/p.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);hist_files.append(target.relative_to(BASE).as_posix())
    t['pathways']=table('Earlier exploratory paths and why they were not promoted as the final method.',
        ['Path','Recorded finding','Evaluation boundary'],[
        ['LIFT-ID E1-E3','OOF rescue fraction 5.04% Original / 1.37% Balanced; final LIFT-ID labels equal DT on both holdout treatments.','Observed historical holdout; extra routing cost without label improvement.'],
        ['Port representation E4','Pooled F1: LR25 18.3492%, LR28 38.9551%, LR29 35.2687%, RF29 49.2875%; all DDoS F2=0.','Two signature folds without session identity; 52,684 targets, different from boundary-purged later protocols.'],
        ['MPED distillation','Relative to supervised student CE: macro-F1 -0.1101 pp; DDoS F2 -0.4551 pp.','Five historical random folds; the two signature folds also fail DDoS. No general quality gain established.'],
        ['Pilot SBFE','Learned temporal F1 98.2140% versus confidence 98.2606%; both DDoS F2=0 on signature diagnostics.','Different predictor/rich-feature candidates; temporal quality does not erase diagnostic failure.'],
        ['Adaptive port fusion','Robust learned selection returns to port-free alpha=0 in all 10 stage/treatment cases; temporal F1 96.7392%.','Earlier short-context WB pair; not a matched evaluation of the final multiscale representation.']
        ],[88,225,174], 'Chronology only: these rows use different models/splits and must not be ranked together or presented as a single controlled comparison. Original CSV/JSON summaries accompany this supplement.')
    # Derivation metadata and small source arrays make the new figures reproducible.
    for p in [evaluated,game_path,verified(xr/'groups/A32/metadata.json')]:
        target=BASE/'evidence/selected_saved_arrays'/p.name
        if target.exists() and sha(target)!=sha(p): target=target.with_name(p.parent.name+'_'+p.name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    md=(OLD/'main_body.md').read_text(encoding='utf-8')
    md=re.sub(r'Table ([34])\b',lambda m:'Table '+{'3':'4','4':'5'}[m[1]],md)
    anchor='With the same features, RF and LR reach temporal macro-F1'
    insertion='''The pooled confusion matrices (Figure 3) make the class trade-off explicit. Under short context, all 8,053 scanning targets are missed: 3,625 are labelled Normal and 4,428 ransomware. Multiscale context correctly labels 7,637 scanning targets, with 408 still labelled ransomware. Conversely, 635 ransomware targets are labelled scanning; signature ransomware recall decreases from 93.0936% to 92.7042%. Table 3 supplies all five recalls, separating the seed 42 runtime/XAI reference from three-seed predictive means.

{{FIGURE:figure5_confusion|Row-normalized signature confusion matrices for Balanced weighted voting on identical pooled targets (N=52,584). Cells are percentages within each true class, with one common color scale. Exact counts are retained in Supplementary Table S15. Signature grouping separates packet-field patterns, while unlabelled histories remain shared.}}

{{TABLE:class_recall}}

'''
    assert anchor in md;md=md.replace(anchor,insertion+anchor,1)
    robust_anchor='Stability under one delay and implementation equivalence do not establish adversarial or causal validity.'
    md=md.replace(robust_anchor,robust_anchor+' Table 6 compares all three models on changed inputs; complete all/changed subsets and background sensitivity are supplied separately.\n\n{{TABLE:robustness_changed}}\n\nAcross A32 and B32, WB has mean top-three Jaccard 0.5520, normalized L1 0.019849 and cosine similarity 0.999948 (Supplementary Table S20). Small magnitude changes can therefore reorder secondary groups even when the dominant attribution is stable. Query-delay stability does not imply background-invariant rankings.\n\nMean absolute grouped Shapley values are shown in Supplementary Figure S2. The atomic history group accounts for 99.14% of WB attribution magnitude under A32, compared with 99.31% for RF and 98.14% for LR. These unequal-size group shares describe the selected 100-target empirical game, not percentages of causal importance or proof that each history field is useful.',1)
    md=md.replace('Raw repeats are retained, and no cross-machine efficiency score is constructed.',
        'Raw repeats are retained, and no cross-machine efficiency score is constructed. Supplementary Tables S17-S18 report feature-input throughput and recorded serialized-size definitions separately.\n\nFigures and added tables are reproducible Matplotlib/NumPy transformations of saved predictions, confusion counts and attribution games; no classifier or explanation game is rerun. OpenAI Codex (OpenAI; exact application/model build not recorded) assisted analytical code, evidence organization and writing. Figure data and derivation scripts are included for author verification; the architecture caption discloses diagram assistance.')
    md=md.replace('Independent captures with a frozen pipeline, complete feature-acquisition profiling and equally optimized singleton controls are the priorities for stronger claims.',
        'Earlier LIFT-ID, port-only, distillation, selective-feature and adaptive-fusion paths are retained in Supplementary Table S21 with their adverse findings. They motivate the present focus but are not additional matched baselines. Independent captures with a frozen pipeline, complete feature-acquisition profiling and equally optimized singleton controls are the priorities for stronger claims.')
    md=md.replace('## Declarations\n\nOpenAI Codex assisted implementation, evidence checks, literature organization, figure code and writing. No new fitting was performed for this revision. Human review of scientific claims, citations and declarations remains the authors\' responsibility. Funding, competing interests, CRediT contributions and final approval by all authors are awaiting confirmation; no absence of funding/conflicts is inferred.',
        '## Declarations\n\nFunding, competing interests, CRediT contributions and final approval by all authors await confirmation. No absence of funding or conflicts is inferred.\n\n## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process\n\nOpenAI Codex (OpenAI) assisted research-code implementation, evidence checks, literature organization, figure code and manuscript drafting. The figures are computational visualizations of retained data, not generated research observations. No new fitting was performed for this revision. The named authors must review and edit all content and approve the scientific claims, references and declarations before submission; final human approval is still pending in this review draft.')
    md=md.replace('Legacy fields are supplied from the canonical table.}}','Legacy fields are supplied from the canonical table. OpenAI Codex (OpenAI; exact build not recorded) assisted the programmatic diagram code.}}',1)
    supp=(OLD/'supplement_body.md').read_text(encoding='utf-8')
    supp += '''

## Complete class outcomes and attribution summaries

{{TABLE:recall_temporal_all}}

{{TABLE:recall_signature_all}}

{{TABLE:confusion_counts}}

The heatmaps use rows normalized to 100%; exact counts above permit direct recovery of every recall. Temporal seed 42 raw predictions are also included in the selected saved arrays. No confusion matrices are pooled across temporal seeds, because each seed predicts the same validation targets.

{{TABLE:historical_full}}

## Throughput, serialization and robustness

{{TABLE:throughput}}

{{TABLE:serialized_sizes}}

{{TABLE:robustness_full}}

{{TABLE:background_full}}

{{FIGURE:figure6_group_importance|Mean absolute exact grouped Shapley values over 100 class-balanced targets, using the A32 donor background and each model's original predicted class. The right panel zooms only the six packet groups on a separate linear scale; the left panel includes history60. Groups have unequal numbers of fields, so magnitude shares are not individual-feature rankings or causal importance. Zero attribution is specific to this finite game and sample.}}

## Earlier exploratory alternatives

{{TABLE:pathways}}

These summaries document adaptive development, not independent confirmation. They cannot be ranked across protocols. The review ZIP includes the cited CSV/JSON summaries; full runs and the earlier archive retain remaining figures and raw results.
'''
    # Ensure all manually stated historical figures match the retained sources.
    e4=read_csv(verified(history_sources[2]))
    for config,expected_f1 in [('LR25',.183492),('LR28',.389551),('LR29',.352687),('RF29',.492875)]:
        r=next(r for r in e4 if r['config']==config and r['scenario']=='original' and r['fold']=='0')
        assert abs(float(r['macro_f1'])-expected_f1)<.0000005
        assert float(r['ddos_f2'])==0
    (BASE/'main_body.md').write_text(md,encoding='utf-8');(BASE/'supplement_body.md').write_text(supp,encoding='utf-8')
    (BASE/'tables.json').write_text(json.dumps(t,indent=2)+'\n')
    maintext=(OLD/'elsarticle/main.tex').read_text(encoding='utf-8')
    abstract=maintext.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
    preamble=PREAMBLE.replace(r'\usepackage[hidelinks]{hyperref}',r'\usepackage{float}'+'\n'+r'\usepackage[hidelinks]{hyperref}')
    main=preamble+r'\begin{frontmatter}\title{Multiscale endpoint context and prediction preserving execution for explainable IoT intrusion detection}'+'\n'+AUTHORS
    main+=r'\begin{abstract}'+abstract+r'\end{abstract}'+'\n'+r'\begin{keyword}IoT intrusion detection \sep endpoint context \sep weighted voting \sep early exit \sep SHAP \sep reproducibility\end{keyword}\end{frontmatter}'+'\n'
    main+=r'\noindent\textit{Research article draft for author review, 22 October 2026. Technical details are provided as separate supplementary material.}'+'\n\n'
    main+=body(md,t)+r'\FloatBarrier\bibliographystyle{elsarticle-num}\bibliography{references}\end{document}'
    stability_table=render_table('robustness_changed',t['robustness_changed'])
    main=main.replace(stability_table,stability_table.replace('[!htbp]','[H]',1))
    main=main.replace(r'\section{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}',
                      r'\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}')
    (SRC/'main.tex').write_text(main,encoding='utf-8')
    supplement=preamble+r'\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\theHtable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}'
    supplement+=r'\begin{center}{\Large\bfseries Supplementary material}\par\medskip M Hafid Ramadhan and Satria Mandala\par\smallskip 22 October 2026\end{center}'+'\n\n'+body(supp,t)+r'\FloatBarrier\end{document}'
    (SRC/'supplement.tex').write_text(supplement,encoding='utf-8')
    build=dict(date='2026-10-22',operational_main_page_target=22,
        page_target_authority='Editorial interpretation of user permission to add a few pages; not an official journal or supervisor numeric rule',
        original_font_pt=12,original_margins=True,new_training=False,model_deserialization=False,new_game_execution=False,
        new_analyses='Saved-result summaries and visualizations only',source_records=list(RECORDS.values()),
        previous_deliverables=prior,main_tables=len(re.findall(r'\{\{TABLE:',md)),main_figures=len(re.findall(r'\{\{FIGURE:',md)),
        supplementary_tables=len(re.findall(r'\{\{TABLE:',supp)),supplementary_figures=len(re.findall(r'\{\{FIGURE:',supp)),
        history_summary_files=hist_files,reference_metadata_entries=len(json.loads((BASE/'references.json').read_text())))
    (BASE/'editorial_build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps({k:v for k,v in build.items() if k not in ['source_records','previous_deliverables']},indent=2))


if __name__=='__main__': main()
