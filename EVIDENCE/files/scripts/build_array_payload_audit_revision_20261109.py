"""New manuscript copy integrating the completed raw-payload audit. No model use."""
from pathlib import Path
import csv, difflib, hashlib, json, re, shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'paper/array_20261105_fpr_update/elsarticle'
NEW=ROOT/'paper/array_20261109_payload_audit/elsarticle'
REP=ROOT/'reports/ARRAY_PAYLOAD_AUDIT_REVISION_20261109'
E=ROOT/'reports/PAYLOAD_EVIDENCE_20261105_01'
LEDGER=[]
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def esc(x):
    s=''.join(c if ord(c)>=32 else '[0x%02x]'%ord(c) for c in str(x))
    return s.replace('&',r'\&').replace('_',r'\_\allowbreak{}').replace('%',r'\%').replace('#',r'\#')
def table(label,caption,header,rows,widths,note,scope='supplement'):
    assert len(widths)==len(header) and abs(sum(widths)-1)<1e-6
    assert all(len(r)==len(header) for r in rows)
    LEDGER.append(dict(scope=scope,label=label,caption=caption,header=header,rows=rows,note=note))
    n=len(header);spec=''.join(r'L{'+f'{w:.8f}'+r'\dimexpr\textwidth-'+str(n*2)+r'\tabcolsep\relax}' for w in widths)
    return (r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+spec+r'}\toprule'+'\n'+' & '.join(r'\textbf{'+esc(v)+'}' for v in header)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(esc(v) for v in row)+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\par\smallskip\begin{minipage}{\linewidth}\footnotesize '+note+r'\end{minipage}\end{table}'+'\n')
def replace_paragraph(s,starts,text):
    pattern=re.compile(r'(?m)^'+re.escape(starts)+r'[^\n]*(?:\n(?!\n)[^\n]*)?')
    # Source prose paragraphs are single lines; do not consume a following command.
    lines=s.split('\n');hits=[i for i,l in enumerate(lines) if l.startswith(starts)]
    assert len(hits)==1,(starts,hits);lines[hits[0]]=text
    return '\n'.join(lines)

def main():
    assert not NEW.exists(),'Keep previous revisions; choose a new source directory.'
    state=read(ROOT/'CURRENT_STATE.json')
    assert sha(ROOT/'penyerahan/EVIDENCE_PAYLOAD_RAW_DAN_REPLAY_UNTUK_DOSPEM_20261106.zip')==state['payload_evidence_20261106']['package_sha256']
    assert sha(OLD/'main.tex')==read(ROOT/'reports/ARRAY_FPR_UPDATE_20261105/verification.json')['main_source_sha256']
    assert sha(OLD/'supplement.tex')==read(ROOT/'reports/ARRAY_FPR_UPDATE_20261105/verification.json')['supplement_source_sha256']
    audit=read(E/'verification.json');assert audit['status']=='PASS_RAW_BYTES_FEATURES_FROZEN_MODEL_REPLAY' and audit['max_probability_difference_from_saved']==0
    assert read(E/'portable_verification.json')['status']=='PASS'
    with (E/'advisor_demo_predictions.csv').open(encoding='utf-8-sig',newline='') as f:demo=list(csv.DictReader(f))
    trace=[json.loads(l) for l in (E/'actual_payload_traces.jsonl').read_text('utf8').splitlines()]
    assert len(trace)==1972 and len(demo)==48
    train=[r for r in trace if r['role']=='training200'];assert len(train)==200
    NEW.mkdir(parents=True);REP.mkdir(parents=True,exist_ok=True)
    for p in OLD.iterdir():
        if p.is_file() and p.suffix in ['.cls','.bst','.bib']:shutil.copy2(p,NEW/p.name)
    shutil.copytree(OLD/'figures',NEW/'figures')
    s=(OLD/'main.tex').read_text('utf8')
    abstract=r'''Multiscale endpoint statistics can improve an IoT intrusion detector while leaving it sensitive to recording conditions. We examine bounded prior-only context, execution of a fixed ensemble and full-probability explanation checks on Edge-IIoTset. Adding ten-second statistics improves temporal macro-F1 by 1.14 percentage points under matched endpoint ordering. Packet-private history reduces signature macro-F1 from 97.296\% to 93.769\%, showing evaluation dependence. Additional benign sources produce 321/760 false alarms (42.24\%) after flag alignment. Matched payload refits leave 310--331 alarms. A retrospective RF-disagreement rule removes the observed alarms, but RF-only reconciliation matches it and no evaluated attack activates the rule. A raw-payload audit further finds protocol-confounded binary training and four syntax indicators with no positive training values. On a separate illustrative SQL string, TF-IDF and hybrid content classifiers predict Normal. The zero-alarm result therefore does not validate SQL detection. Compact execution of the original ensemble preserves tested probabilities and labels while reducing mean replay time by 41.75\% over thirty paired repetitions on one laptop. Grouped Shapley evaluations document window contributions, background sensitivity and faithfulness. These findings support measured execution savings and inspectable explanations for a fixed predictor; detection claims remain limited to the observed traces.'''
    s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+abstract+r'\end{abstract}',s,count=1,flags=re.S)
    start=s.index(r'\section{Introduction}');end=s.index(r'\section{Related work}')
    intro=r'''\section{Introduction}

An IoT intrusion detector must classify traffic with information available at the time of the decision. It must also operate within the gateway's resource budget and give analysts explanations of the function that actually produced each alert. These requirements need separate evidence. A useful representation can still depend on the recording environment, and a fast implementation can preserve a weak predictor exactly. Edge-IIoTset provides packet and tabular records for examining these issues \cite{r1}. Prior work also shows why network metadata, including destination ports, can give misleadingly strong classification results \cite{r2}.

Our earlier comparison selected Balanced weighted voting through a recorded performance--efficiency score. That selection does not show that weighted voting is the most accurate or cheapest classifier. Here we keep the predictor fixed to investigate three questions. RQ1 asks what multiscale endpoint history contributes when context length and endpoint ordering are varied separately. RQ2 asks whether the same fitted predictor can be executed faster without changing its probabilities or labels. RQ3 asks how its grouped Shapley explanations behave under specified donor backgrounds and implementation changes.

The study makes three corresponding contributions. First, it evaluates bounded prior-only endpoint context with a matched context/ordering ablation and history-ownership controls. Second, it combines a label-preserving stopping condition with shared fitted preprocessing, and checks both output equivalence and measured execution cost. Third, it specifies the full-probability donor game used for explanation and measures faithfulness and background sensitivity. Endpoint statistics, weighted voting, early exit and SHAP are established techniques; the contribution lies in the representation, execution contract and evidence connecting them.

The false-alarm results also require attention. On additional benign sources, the ensemble often predicts SQL injection where RF predicts Normal. We examine syntax features, character TF-IDF and a selective disagreement rule as possible responses to those observed errors. A subsequent raw-to-feature audit finds that the small binary experiment separates different protocol/content types and does not cover several of its own syntax indicators. An illustrative SQL string is then missed by the TF-IDF and hybrid content classifiers. We retain these failures alongside the zero-alarm result rather than present the repair as validated detection.

All detection results are exploratory. Development records were examined during earlier research stages, each target class has one capture source, and the original signature diagnostic shares unlabelled packet history. These conditions leave model-selection bias and source dependence unresolved \cite{r3,r4}. We therefore distinguish within-trace detection evidence from checks that an implementation preserves a specified function.

'''
    s=s[:start]+intro+s[end:]
    s=s.replace('Table 1 states the narrower distinction.','Table~\\ref{tab:related} states the narrower distinction.') if r'\label{tab:related}' in s else s
    methods=r'''\subsection{Raw-payload provenance and diagnostic probes}\label{sec:payload_audit_methods}

The payload audit follows each selected record from its stored TCP/MQTT hexadecimal fields to decoded current content, twelve syntax values and saved classifier output. Canonical row identifiers and source CSV/packet indices are kept distinct. For the 760 additional targets, TCP payload and MQTT message values are checked against eight original source CSVs after size/SHA256 verification. The audit also scans all canonical Normal and SQL rows to describe content availability; this descriptive scan supplies no fitting targets, tuning decisions or final-test claim. It reads at most the recorded current payload, without TCP reassembly or decryption.

We replay the owned fitted binary and reference models only after their model, source and input checksums match the run manifests. Replay tests reproducibility of the retained outputs. It does not provide new training data or an independent estimate of detection performance. The audit exports 200 training records, 1,012 content-disjoint test records and 760 additional-source records with their raw fields, decoded content and per-record results. Full source captures remain separate from these record excerpts.

Two manually specified contents, \texttt{10} and \texttt{10' OR '1'='1}, form a separate diagnostic probe. They are embedded in a GET query at \texttt{/sensor?id=} with URL encoding and then passed through the same parser. The example host is \texttt{example.invalid}; no network request or SQL command is executed. These strings are illustrations, not records recovered from the dataset, and neither is inserted into training or counted in an empirical FPR. All three representations and all eight retained binary predictors are inspected without refitting or choosing a new winner from the probe. The main 85-field/context predictor is not scored on these strings because they supply no real packet history. Supplementary Tables S66--S73 retain the audit scope, record examples, feature coverage, all probe outputs and replay checks.

'''
    anchor=r'\section{Results and interpretation}';assert s.count(anchor)==1;s=s.replace(anchor,methods+anchor,1)
    s=replace_paragraph(s,'The agreement with RF-only is decisive',r'''RF-only reconciliation explains the observed alarm reduction without requiring a content model. None of the 5,360 G, 5,190 C or 10,358 V targets enters the eligible conflict region. All 321 rule activations occur on additional Normal targets, and only 64 have usable current content (Table S58). The unchanged V labels therefore do not test the rule where it changes a decision. Moreover, the later probe in Section~\ref{sec:payload_audit_results} shows that the hybrid content check can miss an explicit SQL pattern. Figure~\ref{fig:repair} describes a retrospective candidate whose safety remains unvalidated.''')
    s=replace_paragraph(s,'Table~\\ref{tab:repair_cost} reports',r'''Table~\ref{tab:repair_cost} gives the cost of that separate native pipeline. On the 760 additional targets, median batch time increases from 41.675 ms to 61.109 ms with the hybrid check (+46.63\%), compared with 42.664 ms for RF-only (+2.37\%). The content model thus adds cost without a demonstrated detection benefit over RF-only in this evaluation. The 107 unscorable benign candidates and absence of labelled conflict attacks remain outside the measured zero-alarm result. These measurements neither inherit nor invalidate the compact-execution saving for the original predictor.''')
    probe_rows=[]
    for view,name in [('syntax_only','Syntax'),('tfidf_only','TF-IDF'),('syntax_tfidf','Hybrid')]:
        for case,tag in [('DEMO_NORMAL','Numeric'),('DEMO_SQL','SQL pattern')]:
            r=next(r for r in demo if r['view']==view and r['method']=='weighted_voting' and r['case']==case)
            probe_rows.append([name,tag,f"{float(r['p_SQL']):.6f}",r['predicted']])
    results=r'''\subsection{What the raw-payload audit changes}\label{sec:payload_audit_results}

The stored payloads do contain readable attack content, but the selected Normal and SQL contents are not comparable HTTP samples. Across the canonical data, 7,975 of 24,301 Normal rows and 1,707 of 10,311 SQL rows contain valid TCP hexadecimal content. At the beginning of those current payloads, the parser recognizes no Normal HTTP request and 822 SQL requests, all GET. It finds no literal \texttt{/sensor?id=10} in either class. These counts describe the supplied records; without stream reassembly or decryption they say nothing about HTTP elsewhere in a complete session (Table S66).

Figure~\ref{fig:payload_audit} follows two actual training records. Normal record 105044 contains an MQTT measurement, from which the content extractor obtains \texttt{24.68 76.42}. SQL record 23740 contains a DVWA GET request; its decoded query values are \texttt{2 Submitid=53 ORDER BY 1-- MEbL}. The request headers, including the User-Agent, do not enter this content representation. The raw hex, decoded text, content hash and field provenance are retained for both records (Table S67).

The feature audit explains why correct parsing is not enough. Of the 200 binary training contents, 50 Normal examples are MQTT bodies and 50 are other TCP contents; all 100 SQL examples are HTTP query values. OR, numeric-tautology, UNION SELECT and isdecimal indicators are zero for every training example. Quote and comment counts also occur in non-HTTP Normal bytes. Thus a classifier can obtain a high conditional test score without learning a general distinction between benign and malicious HTTP parameters. Table S68 reports positive-value coverage for all twelve indicators; a zero column does not by itself determine a fitted model's output on a new nonzero value.

The manual probe exposes this limitation directly (Table~\ref{tab:payload_probe}). All three weighted binary classifiers label \texttt{10} as Normal. For \texttt{10' OR '1'='1}, the syntax vector starts with [4,1,1,1,0,0], compared with [0,0,0,0,0,1] for \texttt{10}. Syntax-only voting predicts SQL with probability 0.935234. TF-IDF and hybrid voting predict Normal, with SQL probabilities 0.111796 and 0.157671. Tables S69--S71 retain every singleton, voting and stacking result for both illustrations. Success by syntax-only on one chosen string is not a basis for replacing the model or claiming robustness to SQL injection.

'''
    results+=table('tab:payload_probe','Frozen weighted binary predictions on two manually specified contents.',['Content model','Illustration','p(SQL)','Prediction'],probe_rows,[.22,.23,.20,.35],r'Numeric: \texttt{10}; SQL pattern: \texttt{10\textquotesingle{} OR \textquotesingle{}1\textquotesingle{}=\textquotesingle{}1}. These are illustrative inputs, not dataset test records; no FPR or full-system false-negative rate is estimated from them. The same parser and frozen classifiers are used without refitting. Tables S69--S71 retain all 48 predictor outputs.',scope='main')
    results+=r'''
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/payload_trace_audit.pdf}\caption{Two actual binary-training records followed from raw current-packet content to the extracted representation. The Normal example is MQTT telemetry; the SQL example is an HTTP query value. Their difference illustrates protocol/content confounding rather than a matched benign/malicious HTTP comparison. Full hex fields and source identities are retained in the record-level evidence.}\label{fig:payload_audit}\end{figure}

The frozen-model replay reproduces the stored reference and binary posteriors with maximum absolute difference zero, and reproduces the selective-rule labels (Table S72). The 321/760 to 0/760 benign-alarm change is therefore a reproducible calculation, not a transcription error. Reproducibility does not make the repair safe. If an eligible real packet received the hybrid probe's Normal content decision, that content check would not preserve its original SQL alarm. No real 85-field history is available for the illustrations, so this observation is not counted as an end-to-end missed attack. It is sufficient to reject the stronger interpretation that the hybrid check has already protected SQL sensitivity. Table S73 records the resulting interpretation without another fit, threshold change or method-selection step.

'''
    s=s.replace(r'\section{Prototype and reproducibility}',results+r'\section{Prototype and reproducibility}',1)
    # Avoid non-core text-symbol packages in a table footnote.
    s=s.replace(r'\textquotesingle{}',"'")
    a=s.index(r'\section{Discussion and limitations}');b=s.index(r'\section*{Data and software availability}')
    discussion=r'''\section{Discussion and limitations}

The matched factorial gives a clear answer to RQ1 within the observed temporal traces: adding ten-second endpoint statistics helps under either fixed ordering rule. It does not establish an attack-specific mechanism. Endpoint counts, peer cardinalities and traffic density can encode a source environment, and one capture per class prevents that environment from being separated from the label. The history controls reinforce the limitation: changing ownership and density materially changes signature performance. Neither exact-signature separation nor packet-private history is an independent-capture test.

The benign-source results make this limitation operationally visible. Weighted voting raises 321 aligned alarms on 760 Normal targets, whereas RF raises none. Adding syntax or character features to full-A training does not remove that burden without increasing attack-to-Normal errors. The selective rule removes the observed alarms by resolving known RF/voting conflicts, but RF-only achieves the same outcome more cheaply. Its content check has no evaluated attack activation, and the hybrid classifier misses the manual SQL illustration. The supported finding is therefore a reproducible retrospective alarm reduction, not a validated repair or evidence that payload training preserves attack detection. A matched benign/malicious HTTP evaluation would be needed before attributing a benefit to the content model.

For RQ2, output preservation and computational cost are testable properties of a fixed implementation. Compact preprocessing retains fitted encoding semantics, while the stopping condition returns to full evaluation when a winner cannot be certified. The measured replay saving applies to that unchanged function on one laptop. It does not show that this ensemble is preferable to an equally optimized singleton, and it does not transfer to the disagreement rule. The replay contains 160 annotated decisions and uses table-derived legacy fields. Energy, saturation behavior and embedded-device performance were not measured.

RQ3 is answered for the stated empirical donor game. Window-level grouping shows how attribution is distributed across three equally sized history groups, but correlated windows and nonphysical donor combinations limit interpretation. The archived seven-group top-three Jaccard of 0.5520 and the revised nine-group value of 1.0000 are different grouping-specific results. A shared top-three set does not make attribution magnitudes or their order invariant to the background. Likewise, preserving full-probability coalition values verifies implementation consistency; it does not establish causal explanation quality. Changed guard predictions require their own explanations rather than the archived reference SHAP values.

Finally, development and historical-holdout reuse constrain every detection comparison in this paper. The additional-source failures informed the repair design, and the later manual probe informed the interpretation reported here. Keeping these decisions in a ledger makes the adaptation visible; it cannot turn reused data into a final test. Encoded-space SMOTE and synthetic packet/history hybrids add further modelling assumptions. We retain competitive singleton results and failed interventions so that the representation, runtime and explanation findings can be assessed without treating them as proof of deployment readiness.

\section{Conclusion}

For RQ1, adding ten-second statistics raises temporal macro-F1 from 96.7363\% to 97.8787\% with one-second ordering held fixed, a gain of 1.1425 percentage points. The gain is 1.1475 points with ten-second ordering fixed. This within-trace improvement coexists with substantial source sensitivity: packet-private history reduces signature macro-F1 from 97.2955\% to 93.7694\%, and additional benign sources produce 324/760 false alarms (42.63\%) before flag alignment and 321/760 (42.24\%) afterward. Syntax, TF-IDF and hybrid full-A refits leave 310, 312 and 331 alarms, with three, four and four attack-to-Normal decisions on V compared with two for the reference.

A post-observation disagreement rule reduces the additional alarms to 0/760 and leaves all 10,358 V labels unchanged. RF-only reconciliation gives the same reduction, no evaluated attack activates the rule, and the hybrid content model misses the manually specified SQL pattern. Its Normal/SQL training also separates different content protocols and never activates four of its syntax indicators. The zero-alarm count is reproducible, but SQL safety and a unique payload benefit are not established.

For RQ2, compact certified execution preserves the tested probabilities and labels on 52,684 inputs. Mean replay time falls from 1.5791 to 0.9198 s across thirty paired repetitions, a 41.75\% reduction. The mean paired difference is 0.6593 s, with a descriptive bootstrap 95\% interval of [0.5965, 0.7313] s. This result concerns one workload and one laptop. The separate hybrid-rule batch time is higher than its native reference, 61.109 versus 41.675 ms on the 760 additional targets.

For RQ3, the nine-group, cross-background audit gives weighted-voting deletion/insertion gains of 0.3203/0.3270 on 100 targets. The archived 300-game comparison separately preserves coalition values and attributions after compact execution. These results provide checks of explanation consistency for a specified predictor and donor game, without establishing causal explanations or superiority over RF and LR. Independent capture-level evaluation remains necessary before making a general detection claim.

'''
    s=s[:a]+discussion+s[b:]
    s=replace_paragraph(s,'Edge-IIoTset is publicly distributed',r'''Edge-IIoTset is distributed by its creators; the source publication \cite{r1} identifies the IEEE DataPort record (\url{https://ieee-dataport.org/8939}). The companion reproducibility archive identifies the canonical CSV and source captures by checksum and contains code, frozen configurations, split identifiers, fitted-artifact manifests, saved predictions, raw runtime measurements and derived explanation/audit results. Full third-party datasets and captures are not redistributed. Selected record excerpts are included for the raw-payload audit: 200 binary-training records, 1,012 content-disjoint test records and 760 additional-source records, with hex fields, decoded content, hashes and provenance. Source-data access remains subject to the provider's terms. All 48 manual-probe outputs are labelled as illustrations. The accompanying results notebook also exposes tables, figures and saved per-record predictions without training or model loading. These materials accompany the manuscript; no public code-repository URL is claimed.''')
    s=replace_paragraph(s,'OpenAI Codex (OpenAI;',r'''OpenAI Codex (OpenAI; the exact application build was not recorded) was used for code-development assistance, computational checks, literature organization, programmatic figure code and revision of manuscript language and structure. The methods describe the numerical and provenance checks applied to the retained analyses. Reported observations come from the recorded data and executable experiments; the two manually specified payload probes are identified separately. The authors retain responsibility for checking the code, results, references and interpretation. This declaration does not imply that an AI-detection or similarity score has been measured.''')
    s=s.replace('Figures are programmatic transformations of retained results, with OpenAI Codex assisting their code and manuscript preparation.','OpenAI Codex assisted code development and manuscript preparation; its use is disclosed below. Figures are generated programmatically from retained results. Independent recomputation and raw-to-feature checks are documented with their scope, rather than treated as a substitute for scientific review.')
    assert s.count(r'\begin{table}')==12 and s.count(r'\begin{figure}')==7
    (NEW/'main.tex').write_text(s,encoding='utf8')
    # Supplement preserves every existing numerical row, then adds the audit.
    sup=(OLD/'supplement.tex').read_text('utf8').replace('5 November 2026\\end{center}','9 November 2026\\end{center}')
    sup=sup.replace('These are retrospective candidates, with zero evaluated attack activations of the rule.','These are retrospective candidates, with zero evaluated attack activations of the rule. Tables S66--S73 add the completed raw-payload and frozen-model audit, including the negative manual-probe results. They add no classifier fits or independent test set.',1)
    additions=r'''\section{Raw-payload evidence, coverage and frozen diagnostic probes}

This section integrates the audit completed on 6 November. Canonical corpus counts describe availability, not supervised fitting. The 1,972 exported actual traces comprise 200 binary-training rows, 1,012 primary binary-test rows and 760 additional-source rows. The two illustrations in Tables S69--S71 are not members of those cohorts. The audit retains the previously fitted classifiers and does not turn the observed benign sources into an independent final test.

'''
    def st(label,cap,head,data,w,note):return table('sup:'+label,cap,head,data,w,note)
    avail=audit['canonical_availability']
    rows=[[name,str(avail[name]['rows']),str(avail[name]['valid_TCP']),str(avail[name]['HTTP_requests']),str(avail[name]['GET'])] for name in ['Normal','SQL_injection']]
    additions+=st('canonical_payload','Current TCP-payload availability in the complete canonical Normal/SQL rows.',['Canonical label','Rows','TCP hex valid','HTTP requests','GET'],rows,[.25,.16,.20,.21,.18],r'Request recognition is restricted to the beginning of the recorded current payload, at most 4,096 bytes, without stream reassembly or decryption. Literal \texttt{/sensor?id=10} is absent in both classes. No recognized request has a parse failure. These descriptive counts include canonical records outside development and do not supply fitting or model-selection targets.')
    records=[next(r for r in trace if r['row_id']==i) for i in [105044,23740,-1000074,-1000091]]
    rows=[]
    for r in records:
        raw=r['tcp_payload_hex'];prefix=' '.join(raw[i:i+2] for i in range(0,min(24,len(raw)),2))+' ...' if r['tcp_payload_sha256'] else raw+' (placeholder)'
        rows.append([str(r['row_id']),r['actual_class'],prefix,r['extracted_content'] if r['extracted_content'] else '[no usable current content]'])
    additions+=st('actual_payload_rows','Actual records linking raw fields to extracted content.',['Row ID','Source label','TCP raw prefix','Extracted content'],rows,[.18,.19,.29,.34],r'Positive IDs are zero-based canonical CSV record indices; negative IDs are internal additional-target identifiers with separate source CSV and packet indices. Row 105044 is MQTT telemetry; 23740 is a DVWA GET request. The two additional rows are Normal by source provenance. Raw hex, escaped decode, full content/hash and original indices are retained in the trace files; prefixes here are abbreviated, not replacement payloads. The value 0 in a placeholder field is not decoded as a one-byte payload.')
    names=list(train[0]['syntax']);short=['Single quotes','OR','Equals','Numeric tautology','UNION SELECT','isdecimal','Double quotes','AND','Line comments','Block comments','Semicolons','SQL keywords']
    rows=[[name,str(sum(r['syntax'][key]>0 for r in train if r['actual_class']=='Normal')),str(sum(r['syntax'][key]>0 for r in train if r['actual_class']=='SQL_injection'))] for key,name in zip(names,short)]
    additions+=st('syntax_training_coverage','Positive-value coverage of all twelve syntax fields in binary training.',['Indicator','Normal / 100','SQL / 100'],rows,[.48,.26,.26],r'Entries count records with feature value greater than zero, not sums of token occurrences. Training has 100 unique nonempty contents per class: Normal comprises 50 MQTT and 50 other TCP contents; SQL comprises 100 HTTP query values. OR, numeric tautology, UNION SELECT and isdecimal are zero in all 200 training records. Counts of quotes/comments in binary bytes do not establish SQL semantics.')
    methods=['decision_tree','random_forest','xgboost','lightgbm','logistic_regression','soft_voting','weighted_voting','stacking'];mn=['DT','RF','XGB','LGB','LR','Uniform','Weighted','Stacking']
    for view,shortname in [('syntax_only','syntax'),('tfidf_only','TF-IDF'),('syntax_tfidf','hybrid')]:
        rows=[]
        for method,name in zip(methods,mn):
            for case,tag in [('DEMO_NORMAL','Numeric'),('DEMO_SQL','SQL pattern')]:
                r=next(r for r in demo if r['view']==view and r['method']==method and r['case']==case)
                rows.append([name,tag,f"{float(r['p_Normal']):.6f}",f"{float(r['p_SQL']):.6f}",r['predicted']])
        additions+=st('probe_'+view,'All '+shortname+' binary predictor outputs on the two manual illustrations.',['Predictor','Content','p(Normal)','p(SQL)','Prediction'],rows,[.18,.19,.19,.19,.25],r'Numeric content is \texttt{10}; SQL-pattern content is \texttt{10\textquotesingle{} OR \textquotesingle{}1\textquotesingle{}=\textquotesingle{}1}. Both pass through the same URL-encoded GET/query parser with the non-routable example host. No request is sent. These are frozen-model diagnostic illustrations, not observed test records, training examples, attack-conflict activations or an estimate of FPR/recall. No classifier or threshold is changed afterward. Full-precision probabilities and original illustrative inputs are in the archive.')
    additions+=st('audit_replay','Raw-source checks and replay of the owned frozen artifacts.',['Check','Scope','Retained result'],[
        ['Canonical identity','Complete canonical CSV','Expected SHA256 matched; descriptive availability scan only.'],
        ['Additional-source identity','8 original CSVs; 760 targets','Size/SHA256 matched; tcp.payload and mqtt.msg exactly match mapped frozen input.'],
        ['Training reconstruction','200 binary contents; 12 syntax fields','Original record values, extracted content and saved features match.'],
        ['Binary replay','24 predictors; 1,012 content-disjoint targets','Maximum absolute posterior difference from saved output: 0.'],
        ['Reference replay','Five experts plus weighted/uniform voting; 10,358 V and 760 additional targets','Maximum absolute posterior difference from saved output: 0.'],
        ['Selective-rule reconstruction','Same 11,118 target decisions','Labels reproduced; 321 observed additional alarms become 0; V labels unchanged.'],
        ['Manual illustrations','2 contents x 3 representations x 8 predictors','48 outputs retained separately; no fit or final-test interpretation.'],
        ['Portable replay','Source-row exports and verified owned models','Replay passes without full original CSV/PCAP files; it does not repeat the full-source hash audit.']
    ],[.25,.31,.44],r'All model/source/input manifests are checked before owned fitted-artifact loading. The fresh-source audit and portable exported-row replay have different access requirements. These are reproducibility checks, not independent scientific replication. The audit introduced zero new classifier fits and made no changes to the predictor or its weights.')
    additions+=st('audit_decisions','Post-audit interpretation and evidence boundaries.',['Finding or decision','Consequence'],[
        ['822 recognized SQL GET requests; no recognized Normal HTTP requests at the current-payload start','Do not describe the binary experiment as a matched benign/malicious HTTP test. No claim about every reassembled session follows.'],
        ['Four syntax indicators never active in 200 training contents','Distinguish an implemented pattern extractor from evidence that its positive patterns were learned.'],
        ['Hybrid and TF-IDF predict Normal for the SQL illustration','Retain the failure in the main text. Do not infer a full-system false-negative rate without real context.'],
        ['Syntax-only succeeds on one chosen SQL string','Do not promote it as a new winner or tune the frozen models to this illustration.'],
        ['0/760 hybrid/RF-only additional alarms remains reproducible','Report the retrospective count together with zero evaluated attack-conflict support and unvalidated SQL safety.'],
        ['Original execution and SHAP claims','Continue to bind these claims to the original predictor, not the changed disagreement rule.'],
        ['Reused development and source data','No independent final test is created by the audit or by the executed results notebook.'],
        ['Versioned interpretation','Preserve the numerical record and retain the failed probes alongside the previously reported alarm reduction.']
    ],[.44,.56],r'This ledger records interpretation after the 5 November model experiments and the 6 November payload audit. No new dataset, capture, learner, fitting stage or threshold selection was added to prepare this document revision.')
    additions=additions.replace(r'\textquotesingle{}',"'")
    sup=sup.replace(r'\end{document}',additions+'\n'+r'\end{document}')
    assert sup.count(r'\begin{table}')==73 and sup.count(r'\begin{figure}')==6
    (NEW/'supplement.tex').write_text(sup,encoding='utf8')
    highlights=['Ten-second context raises matched temporal macro-F1 by 1.14 points.',
                'Raw-payload checks expose protocol confounding and missed SQL evidence.',
                'Zero observed benign alarms do not validate the disagreement rule.',
                'Fixed-predictor compact execution saves 41.75% mean replay time.',
                'Full-probability donor games quantify explanation consistency.']
    assert all(len(h)<=85 for h in highlights);(NEW/'highlights.txt').write_text('\n'.join(highlights)+'\n',encoding='utf8')
    write(REP/'table_ledger.json',LEDGER)
    write(REP/'input_provenance.json',[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in [OLD/'main.tex',OLD/'supplement.tex',OLD/'references.bib',E/'verification.json',E/'advisor_demo_predictions.csv',E/'actual_payload_traces.jsonl']])
    for n in ['main','supplement']:
        (REP/(n+'_changes.diff')).write_text(''.join(difflib.unified_diff((OLD/(n+'.tex')).read_text('utf8').splitlines(True),(NEW/(n+'.tex')).read_text('utf8').splitlines(True),fromfile='5 November/'+n,tofile='9 November/'+n)),encoding='utf8')
    write(REP/'build_summary.json',dict(status='SOURCE_READY',main_tables=12,main_figures=7,supplement_tables=73,supplement_figures=6,references=26,new_fits=0,new_inference_this_integration=0,abstract_words=len(abstract.split()),old_numerical_tables_preserved='main1-11 and S1-S65'))
    print('SOURCE_READY: main12tables/7figures; supplementS1-S73/6figures;',len(abstract.split()),'abstract words',flush=True)

if __name__=='__main__':main()
