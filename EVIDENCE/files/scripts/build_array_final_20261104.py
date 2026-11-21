"""Six-comment revision using the reviewed 3-Nov source and retained evidence."""
from pathlib import Path
import sys,re,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import pandas as pd
from build_array_major_20261103 import table
OLD=ROOT/'paper/array_20261103_major_revision/elsarticle'
OUT=ROOT/'paper/array_20261104_final_review/elsarticle'
REP=ROOT/'reports/ARRAY_FINAL_REVIEW_20261104'

def replace(text,a,b):
    assert a in text,a[:90]
    return text.replace(a,b,1)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in OLD.glob('*'):
        if p.is_file() and p.suffix in ['.cls','.bst','.bib','.txt']:shutil.copy2(p,OUT/p.name)
    shutil.copytree(OLD/'figures',OUT/'figures',dirs_exist_ok=True)
    text=(OLD/'main.tex').read_text(encoding='utf8')
    text=text.replace('Author-review draft, 3 November 2026.','Author-review draft, 4 November 2026; final author approval pending.')
    text=replace(text,'A coverage-limited audit of additional benign captures yields 324/760 false alarms for weighted voting, exposing a serious source-transfer limitation.','A coverage-limited audit of additional benign captures yields 324/760 false alarms (42.63\\%); correcting a hexadecimal/decimal flag-format mismatch reduces this only to 321/760 (42.24\\%). All residual alarms are SQL injection predictions, exposing a serious source-transfer limitation.')
    text=replace(text,'submitted by the authors through EDAS (paper 1571374892).','submitted by the authors through EDAS (paper 1571374892) \\cite{agers2026}.')
    text=replace(text,'The related submission must be disclosed to the editor; overlapping initial results are not presented as a new independent experiment.','The historical results retained in Supplementary Tables S12, S16, S22--S24 and S40 partly reproduce or expand results in that related manuscript and are marked as shared provenance. They establish research history, not additional contributions or independent replication. The conference does not report the matched context/ordering factorial, history-ownership controls, additional-benign format diagnostic, compact same-predictor replay or full-probability donor-game audits. The complete related manuscript and an overlap map accompany the author-review package; the relationship must also be disclosed in the submission cover letter for editorial assessment.')
    text=replace(text,'Packet-private history lowers weighted-voting macro-F1','Table~\\ref{tab:history} shows that packet-private history lowers weighted-voting macro-F1')
    text=replace(text,'Uniform voting has slightly higher mean F1','Table~\\ref{tab:ensemble} shows that uniform voting has slightly higher mean F1')
    text=replace(text,'All three capacities retain the same temporal weighted-voting labels','Table~\\ref{tab:buffer} shows that all three capacities retain the same temporal weighted-voting labels')
    text=replace(text,'Figure~\\ref{fig:runtime} reports every pair.','Table~\\ref{tab:runtime} summarizes serving costs, and Figure~\\ref{fig:runtime} reports every replay pair.')
    text=replace(text,'Deletion/insertion gains remain evaluations of the specified empirical donor game.','Table~\\ref{tab:faith} reports weighted-voting deletion/insertion gains of 0.3203/0.3270 under the nine-group, cross-background game. These gains remain evaluations of the specified empirical donor game.')
    method=r'''The later false-alarm diagnosis uses saved per-expert probabilities and independently reconstructs each matched target's sixty history values with a deque implementation of the documented ring definition. It rechecks eight-field packet identity, uniqueness within the source prefix, a constant timestamp-hour offset, the mapped CSV fields and the original target IDs. This verifies agreement with the supplied legacy CSV fields, not independent extraction of every protocol field by a second packet dissector. The check also compares categorical encodings with the original unresampled A rows. A hexadecimal/decimal mismatch in three flag fields motivates one paired-format diagnostic with the same frozen predictor and all other input cells unchanged. This is a post-observation pipeline correction, not a new classifier fit, retuning or an untouched final test. Both formats and their transition counts are retained.

'''
    text=replace(text,r'\section{Results and interpretation}',method+r'\section{Results and interpretation}')
    start=text.index(r'\begin{table}',text.index(r'\subsection{False alarms on additional benign sources}'))
    end=text.index(r'\subsection{Capacity and warm-up sensitivity}',start)
    m=pd.read_csv(ROOT/'runs/ARRAY_FLAG_FORMAT_20261104_01/summary.csv').set_index(['format','method'])
    labels={'logistic_regression':'Logistic regression','random_forest':'Random forest','uniform_voting':'Uniform voting','weighted_voting':'Weighted voting'}
    rows=[]
    for key,name in labels.items():
        a=m.loc[('historical_format',key)];b=m.loc[('semantic_format',key)]
        rows.append([name,int(a.false_alarms),f'{a.fpr_percent:.2f}',int(b.false_alarms),f'{b.fpr_percent:.2f}'])
    replacement=table('Additional-benign diagnostic before and after semantic flag-format alignment.','tab:benign',['Predictor','Before alarms','Before FPR (\\%)','Aligned alarms','Aligned FPR (\\%)'],rows,r'Both formats use the same 760 exactly matched candidates; no fitting or weight selection is performed. Only tcp.flags, mqtt.conflags and mqtt.hdrflags are converted from hex text to the decimal-float text used by training. Of 867 eligible candidates, 107 remain unscorable. Aligned weighted-voting all-eligible FPR bounds are [37.02, 49.37]\%, treating all missing outcomes as either non-alarm or alarm; these are not confidence intervals. Tables S37 and S41 retain source-specific outcomes.',[.28,.18,.18,.18,.18])
    replacement+=r'''
Table~\ref{tab:benign} identifies a format-sensitive component of the false alarms. Historically, weighted voting raises 324/760 alarms (42.63\%), compared with zero for RF. All 324 errors predict SQL injection; none predicts DDoS, scanning or ransomware. The alarms are concentrated in Distance (77/93), Flame Sensor (73/94), IR Receiver (69/97), Sound Sensor (81/92) and Water Level (24/93). Heart Rate, phValue and Soil Moisture have none on their matched subsets. Modbus has no scorable target. These are highly uneven source-conditional rates, not a representative operational prevalence.

The reconstruction finds no row/context mismatch on those 760 targets: all selected eight-field packet identities are unique within their source prefixes, timestamp offsets are consistently one hour, and the 85-field vectors agree with saved inputs within $7.11\times10^{-15}$ for numeric fields and exactly for categorical fields. All 760 mapped CSV row indices differ from packet indices, which confirms why index-based alignment would be invalid. No capacity eviction occurs in these prefixes. These checks do not resolve the 107 unscorable candidates or independently validate the upstream extraction of every supplied protocol field.

A separate representation defect is present. Training encodes flag values as decimal-float strings, such as \texttt{16.0}, whereas the additional CSVs encode the same flag as \texttt{0x00000010}. Because categorical mapping retains text, all 760 \texttt{tcp.flags} values are unseen categories before correction; \texttt{mqtt.conflags} and \texttt{mqtt.hdrflags} have analogous mismatches on 50 and 215 rows. Converting only these three fields to the training format, without changing their semantic values, lowers weighted-voting alarms to 321/760 (42.24\%). Four earlier errors are corrected and one new error appears. RF remains at zero, uniform voting changes from 299 to 295 alarms, and LR from 417 to 420. Thus serialization is a real defect, but its correction produces a net change of only three weighted-voting alarms and leaves the main transfer failure intact. Table S41 reports both formats by capture; S42--S44 document expert behavior and feature checks.

The probability decomposition explains how the residual disagreement arises. On the 321 aligned weighted-voting errors, mean weighted SQL-minus-Normal contributions are +0.1493 for DT, +0.1825 for LR and +0.0021 for LightGBM, versus -0.0975 for RF and -0.1366 for XGBoost. The sum is +0.0998. Soft voting can therefore choose SQL even when RF chooses Normal: class probabilities and weights determine the sum, rather than the number of experts agreeing. Across all matched rows, aligned standalone alarm counts are DT 611, RF 0, XGBoost 2, LightGBM 509 and LR 420. RF's Normal decision is outweighed by other experts' probability contributions; this is not a label-order or aggregation mismatch.

Observed feature distributions offer an associated, rather than causal, explanation. Before format correction, median higher-slot ten-second log packet count is 8.9672 on the 324 alarm targets, compared with 5.0562 on original A Normal rows and 7.1562 on A SQL rows. Corresponding log local TCP-port cardinalities are 6.2066, 2.3979 and 4.5643. The novel benign sources expose dense endpoint activity beyond the usual Normal training profile. Packet size and reset fractions do not uniformly resemble SQL traffic, so this is not evidence that the packets are attacks or that a single statistic causes the errors. Source-confounded training, correlated history summaries and extrapolation remain plausible contributors. Categorical formatting alone cannot account for the residual alarms, and no new evaluation outcome is used to select a replacement model.

The zero RF count does not establish a generally superior IDS: the diagnostic has no new attack capture, incomplete CSV coverage, correlated targets and a limited source prefix. Both the historical 42.63\% and format-aligned 42.24\% rates contradict reliable source-independent weighted-ensemble detection. An independent, fully aligned test with normal and attack sources remains necessary.

'''
    text=text[:start]+replacement+text[end:]
    a=text.index(r'\section{Conclusion}');b=text.index(r'\section*{Data and software availability}',a)
    text=text[:a]+r'''\section{Conclusion}

For RQ1, the matched factorial isolates a temporal macro-F1 gain from adding ten-second endpoint statistics: 96.7363\% to 97.8787\% with one-second ordering fixed (+1.1425 percentage points), and +1.1475 points with ten-second ordering fixed. These gains coexist with an important limitation: additional benign sources yield 324/760 false alarms (42.63\%) before semantic flag alignment and 321/760 (42.24\%) afterward; all residual alarms predict SQL injection, while RF raises none on these rows. The history-ownership control also reduces signature macro-F1 from 97.2955\% to 93.7694\%. Context helps within the observed temporal traces, but these controls preclude interpreting that gain as reliable source-independent detection.

For RQ2, compact certified execution reduces mean replay time from 1.5791 to 0.9198 s (41.75\%) across thirty paired repetitions. The paired saving is 0.6593 s, with a descriptive bootstrap 95\% interval of [0.5965, 0.7313] s. Tested full probabilities and labels are preserved on 52,684 inputs. This is an implementation-specific result on one laptop and a replay with 160 annotated decisions, not a hardware-independent speed guarantee.

For RQ3, the nine-group audit yields cross-background weighted-voting deletion/insertion gains of 0.3203/0.3270 on 100 targets and a background top-three Jaccard of 1.0000; the latter reflects the same three history windows occupying that set, not invariant ranks or magnitudes. The separate archived 300-game audit preserves full-probability coalition values and attributions after compact execution. Together these results support auditable explanation consistency for a specified donor game, without establishing causal explanations or general explainability superiority. One source per target class, reused development data and the adverse benign-source results remain decisive limits; independent-capture final evaluation is still required.

'''+text[b:]
    a=text.index('Edge-IIoTset is described in Ref.',text.index(r'\section*{Data and software availability}'));b=text.index(r'\section*{Funding}',a)
    text=text[:a]+r'''Edge-IIoTset is publicly distributed by its creators; the source publication \cite{r1} identifies the IEEE DataPort record (\url{https://ieee-dataport.org/8939}). The canonical CSV and raw-capture archive used here are identified by SHA256 in the accompanying reproducibility archive. That archive contains the research code, frozen configurations, split identifiers, saved predictions, derived audit results and environment specification underlying the reported analyses. Raw third-party CSV/PCAP files are not redistributed in that archive and remain subject to the dataset provider's access and use terms. The supporting archive accompanies this manuscript; no separate public code-repository URL is claimed.

'''+text[b:]
    (OUT/'main.tex').write_text(text,encoding='utf8')
    bib=(OUT/'references.bib').read_text(encoding='utf8')
    bib+='\n'+r'''@misc{agers2026,
 author = {M Hafid Ramadhan and Satria Mandala},
 title = {Port Dependence and Cost Trade-offs of Ensemble Intrusion Detection for Industrial {IoT}},
 year = {2026},
 note = {Related manuscript submitted to AGERS 2026, EDAS paper 1571374892; local submitted-version file supplied for editorial review. Publication status not established}
}
'''
    (OUT/'references.bib').write_text(bib,encoding='utf8')
    # Label shared historical results by table environment, preserving S1--S40.
    supp=(OLD/'supplement.tex').read_text(encoding='utf8').replace('3 November 2026','4 November 2026')
    supp=supp.replace('Tables S25 onward add the November major-revision experiments, including 70 expert fits.','Tables S25--S40 add the November major-revision evidence, including 70 expert fits. Tables S41 onward add the 4 November fixed-model false-alarm diagnosis and raw replay measurements; no model is refitted for these additions.')
    tables=list(re.finditer(r'\\begin\{table\}.*?\\end\{table\}',supp,re.S));assert len(tables)==40
    for number in [40,24,23,22,16,12]:
        m=tables[number-1];block=m[0];block=re.sub(r'(\\caption\{[^{}]*)(\})',r'\1 Shared historical evidence with AGERS EDAS 1571374892; retained for provenance, not a new contribution.\2',block,count=1)
        supp=supp[:m.start()]+block+supp[m.end():]
    supp=supp.replace('Additional benign sources: matched coverage and weighted-voting false alarms.','Additional benign sources: matched coverage and historical-format weighted-voting false alarms.')
    at=supp.index(r'\section{Artifact map and submission checks}')
    intro=r'''\clearpage\section{Final-review false-alarm diagnosis and runtime audit}

Tables S1--S40 retain their original numbering and values, with shared AGERS results identified in the historical captions. Table S37 is the historical-format benign audit; Table S41 adds the paired semantic-format control. There is no new fit, resampling, hyperparameter or weight selection. The 760 matched targets and all 107 unscorable candidates retain their original status. Detailed target probabilities, expert margins, unknown categories and alignment checks are in the accompanying reproducibility archive. Figure S6 and Table S15 retain their original short-context weights; they must not be substituted for the fixed-weight factorial in Tables S25--S26.

'''
    trans=json.loads((REP/'semantic_transitions.json').read_text());audit=pd.read_csv(REP/'alignment_reconstruction.csv');records=json.loads((ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04/source_audit.json').read_text())
    rows=[]
    for rec in records:
        item=next((x for x in trans if x['source']==rec['source']),None);n=rec['scored'];rows.append([rec['source'].replace('_',' '),n,rec['eligible']-n,item['before'] if item else '--',item['after'] if item else '--',f"{100*item['after']/n:.2f}" if n else '--'])
    intro+=table('Capture-specific historical and format-aligned weighted-voting outcomes.','sup:benign_format',['Source','Matched','Missing','Before FP','Aligned FP','Aligned FPR (\\%)'],rows,r'Every false alarm in both formats predicts SQL injection; all other matched targets predict Normal. Modbus has no estimate. Four historical errors become correct and one new error appears after three-field format alignment. RF has zero false alarms in every matched source in both formats.',[.25,.14,.14,.14,.14,.19])
    ex=pd.read_csv(REP/'expert_summary.csv').set_index('expert');after=pd.read_csv(REP/'semantic_expert_by_capture.csv').groupby('expert')[['false_alarms','sql_predictions']].sum();margin=pd.read_csv(REP/'semantic_error_margin_summary.csv').set_index('expert')
    intro+=table('Expert alarms and weighted SQL-minus-Normal contributions after flag alignment.','sup:expert_diagnosis',['Expert','Before FP','Aligned FP','Aligned SQL','Mean weighted margin'],[[n,int(ex.loc[n,'false_alarms']),int(after.loc[n,'false_alarms']),int(after.loc[n,'sql_predictions']),f"{margin.loc[n,'mean_weighted_sql_minus_normal']:+.6f}"] for n in ['DT','RF','XGB','LGB','LR']],r'Alarm counts use all 760 matched benign rows. Margin contributions use only the 321 aligned weighted-voting errors and the fixed reference weights; their sum is +0.099759. A negative contribution favors Normal against SQL. This is exact aggregation decomposition, not causal feature attribution.',[.16,.18,.18,.18,.30])
    intro+=table('Independent reconstruction of matched inputs.','sup:alignment',['Source','Targets','CSV index differs','Numeric max error','Categorical mismatches'],[[r.source.replace('_',' '),r.matched,r.csv_index_mismatch_count,f'{r.numeric_max_abs_error:.2e}',r.categorical_mismatches] for r in audit.itertuples()],r'Every selected eight-field packet identity occurs once within its prefix. All sources have a constant one-hour CSV/PCAP clock offset. A separate deque implementation reconstructs the sixty prior-only history values, and mapped CSV fields reproduce the other twenty-five. The supplied legacy CSV is not independently re-dissected, and 107 unmatched candidates remain unscorable. There are no capacity evictions in these prefixes.',[.25,.12,.20,.23,.20])
    prof=pd.read_csv(REP/'feature_profiles.csv');features=['multiscale_higher_10000ms_log_packets','multiscale_higher_10000ms_log_local_tcp_ports','multiscale_higher_10000ms_mean_bytes','multiscale_higher_10000ms_rst_fraction'];names=['10 s log packet count','10 s log local ports','10 s mean wire bytes','10 s RST fraction'];rows=[]
    for c,n in zip(features,names):
        q=prof[prof.feature==c].set_index('group');rows.append([n]+[f"{q.loc[g,'median']:.4f}" for g in ['original_A_Normal','original_A_SQL_injection','benign_false_alarm','benign_normal_prediction']])
    intro+=table('Selected higher-slot feature medians for distribution diagnosis.','sup:feature_profiles',['Feature','A Normal','A SQL','Before alarm','Before Normal'],rows,r'A profiles use original, unresampled training rows (Normal 11,669; SQL 4,909). Benign groups use the historical-format decisions (324 alarms, 436 Normal); these numeric features are unchanged by flag alignment. Features were selected post hoc for diagnosis, not to select a model or establish causation. Three flag-string mismatches are checked separately; actual unseen MQTT topic lengths remain unchanged.',[.28,.18,.18,.18,.18])
    replay=pd.read_csv(ROOT/'runs/ARRAY_RUNTIME_FULL_20261103_01/replay_raw.csv').pivot(index='repeat',columns='mode',values='total_seconds')
    intro+=table('All thirty November paired replay measurements.','sup:runtime30',['Pair','Native s','Compact s','Difference s'],[[i+1,f'{r.native_certified:.6f}',f'{r.compact_certified:.6f}',f'{r.native_certified-r.compact_certified:.6f}'] for i,r in replay.iterrows()],r'The same 15,665 packets and 160 annotated decisions are used in every pair; policy order is randomized. These are the measurements summarized in main Table 7 and Figure 4. Table S10 retains the separate historical five-pair session and is not pooled with this session. Component/CPU timings and 40 cold-process RSS replicates remain in the companion CSV.',[.16,.28,.28,.28])
    supp=supp[:at]+intro+supp[at:]
    supp=supp.replace('Public repository/redistribution arrangements, corresponding-author contact details, disclosure of the related conference submission and final approval by both authors must be completed or confirmed before submission.','The accompanying reproducibility archive provides code and derived evidence; the original dataset remains with its provider. No existing public code repository was identified in the project. A cover-letter disclosure draft and the AGERS manuscript accompany the author-review package. Final approval by both authors and corresponding-author contact confirmation remain required before submission.')
    (OUT/'supplement.tex').write_text(supp,encoding='utf8')
    (OUT/'highlights.txt').write_text('Matched controls isolate a 1.14-point within-trace temporal F1 gain.\nBenign false alarms persist after correcting flag serialization.\nCompact same-predictor execution reduces mean replay time by 41.75%.\nWindow-level donor games quantify explanation consistency and sensitivity.\nRelated conference results and exploratory evaluation limits are disclosed.\n',encoding='utf8')
    print('BUILT',OUT)
if __name__=='__main__':main()
