"""Editorial condensation of verified completed experiments; no fitting/inference."""
from pathlib import Path
import hashlib, json, re, shutil, difflib

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "paper/array_20261110_integrated/elsarticle"
NEW = ROOT / "paper/array_20261110_concise/elsarticle"
OUT = ROOT / "reports/ARRAY_CONCISE_20261110"
KEEP = ["factorial", "history", "ensemble", "runtime", "faith", "guard_fpr"]
MOVED = ["related_short", "benign", "buffer", "payload_refits", "repair",
         "repair_cost", "payload_probe", "external_normal", "external_attack"]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tables(s):
    return re.findall(r"\\begin\{table\}.*?\\end\{table\}", s, re.S)

def body(t):
    return re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", t, re.S)[0]

def table_map(s):
    return {re.search(r"\\label\{tab:([^}]+)\}", t)[1]: t for t in tables(s)}

def figure(s, label):
    fs = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", s, re.S)
    return next(f for f in fs if r"\label{fig:" + label + "}" in f)

def subsection(s, name):
    start = s.index(r"\subsection{" + name + "}")
    end = re.search(r"\n\\(?:subsection|section)\{", s[start + 1:])
    return s[start:start + 1 + end.start()] if end else s[start:]

def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding="utf8", newline="\n")

def main():
    assert sha(OLD/"main.tex") == "e840575c1fbc757ff8282d58234dab8f8f3ef93a39df60ec47169b23654b0665"
    assert sha(OLD/"supplement.tex") == "3c0ccac7b71cc91deda7fe2d7ec79e688c6a12ecc52fd32542c8fcb5f0afa4ed"
    before = (OLD/"main.tex").read_text("utf8")
    sup0 = (OLD/"supplement.tex").read_text("utf8")
    oldtables = table_map(before)
    assert len(oldtables) == 15 and set(oldtables) == set(KEEP + MOVED)
    NEW.mkdir(parents=True, exist_ok=True)
    for p in OLD.rglob("*"):
        if p.is_file() and (p.suffix in [".cls", ".bst", ".bib", ".txt"] or p.parent.name == "figures"):
            q = NEW/p.relative_to(OLD)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    prefix = before[:before.index(r"\section{Introduction}")]
    abstract = ("Recent endpoint activity can make packet-level IoT intrusion detection more informative, "
                "but its representation, execution cost and explanation require separate evidence. "
                "We study a fixed five-expert ensemble using bounded prior-only multiscale context, "
                "prediction-preserving execution and quantitative grouped Shapley audits. "
                "On Edge-IIoTset, a matched context/ordering ablation attributes a 1.1425--1.1475 "
                "percentage-point temporal macro-F1 gain to adding ten-second statistics. "
                "Separating history ownership lowers signature macro-F1 from 97.2955\\% to 93.7694\\%, "
                "showing sensitivity to evaluation context. Compact execution preserves tested labels "
                "and full posteriors when requested, while reducing mean replay time by 41.75\\% "
                "over thirty paired repetitions on one laptop. Window-level explanation tests distinguish "
                "implementation consistency from donor-background sensitivity and faithfulness. "
                "False-alarm diagnostics reveal substantial source dependence: a later guarded predictor "
                "reduces alarms to 1/697 on a separately scored within-trace benign block, but raises "
                "reused-validation Normal false-positive rate to 1.0104\\%. Frozen external tests retain "
                "substantial false alarms and missed mapped Ransomware. "
                "The results support within-trace representation gains, measured same-predictor efficiency "
                "and auditable explanation consistency, with cross-dataset detection reliability unresolved.")
    prefix = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
                    lambda m: r"\begin{abstract}"+abstract+r"\end{abstract}", prefix, flags=re.S)
    content = (ROOT/"paper/array_20261110_concise/main_body.tex").read_text("utf8")
    # Cross-document supplementary references are explicit numbers, never unresolved TeX refs.
    content = content.replace(r"Supplementary Section~\ref*{sec:unused_cross_document_placeholder}",
                              "Supplementary Tables S86--S95 and the relocated extraction methods")
    for key in KEEP:
        content = content.replace("%%TABLE_"+key+"%%", oldtables[key])
    figs = ["flow", "factorial", "runtime", "xai", "fpr_transfer"]
    for key in figs:
        fig = figure(before, key).replace("[H]", "[!htbp]")
        if key == "fpr_transfer":
            fig = fig.replace("Numeric annotations retain low rates that are difficult to see as bars.",
                              "Numeric annotations retain low rates that are difficult to see as bars. "
                              "The dashed 8\\% line marks a study-specific operational target, not a general IDS standard.")
        content = content.replace("%%FIGURE_"+key+"%%", fig)
    transfer = r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}
\caption{Frozen external results, retaining each label scope.}\label{tab:transfer}
\begin{tabular}{L{0.24\dimexpr\textwidth-12\tabcolsep\relax}L{0.10\dimexpr\textwidth-12\tabcolsep\relax}L{0.24\dimexpr\textwidth-12\tabcolsep\relax}L{0.14\dimexpr\textwidth-12\tabcolsep\relax}L{0.14\dimexpr\textwidth-12\tabcolsep\relax}L{0.14\dimexpr\textwidth-12\tabcolsep\relax}}\toprule
\textbf{Targets} & \textbf{n} & \textbf{Metric} & \textbf{WV} & \textbf{RF} & \textbf{GV}\\\midrule
CIC Normal & 2000 & FPR & 30.15 & 30.60 & 31.90\\
TON Normal & 2000 & FPR & 24.35 & 18.45 & 18.85\\
CIC SQL & 1000 & Scenario match & 29.80 & 33.00 & 32.10\\
CIC Scan & 1000 & Scenario match & 3.80 & 4.20 & 4.00\\
CIC SYN & 1000 & Scenario match & 62.10 & 98.10 & 62.10\\
TON Ransomware & 2000 & Type recall & 1.35 & 1.50 & 1.55\\
\bottomrule\end{tabular}\par\smallskip\begin{minipage}{\linewidth}\footnotesize All metric values are percentages. WV: reference voting; RF: random forest; GV: guarded voting. CIC attack rows use file-scenario labels, not verified malicious action in each packet. TON rows use uniquely derived flow labels; Normal and Ransomware supports are separate. Pooling within a dataset is descriptive. Per-file values and complete outcomes appear in Supplementary Tables S87--S92 and S103--S104.\end{minipage}\end{table}"""
    content = content.replace("%%TABLE_transfer%%", transfer)
    assert "%%TABLE_" not in content and "%%FIGURE_" not in content
    manuscript = prefix+content
    # The nine migrated summary tables retain their numerical bodies verbatim.
    reloc = [r"\FloatBarrier\clearpage\section{Consolidated summaries relocated from the main manuscript}",
             "The concise manuscript retains the core representation, execution and explanation results. "
             "Tables S96--S104 below preserve nine summary tables from the preceding main manuscript, "
             "with their numerical bodies unchanged. Tables S1--S95 retain the complete earlier evidence. "
             "The later guard changes the reference predictor and does not inherit its execution or explanation contract."]
    titles = ["Closest precedents", "Initial benign false alarms", "Ring capacity",
              "Matched payload additions", "Earlier selective reconciliation",
              "Separate cost of the earlier reconciliation", "Illustrative payload probes",
              "External Normal outcomes by file", "External scenario and mapped attack outcomes"]
    ledger = []
    for n, key, title in zip(range(96,105), MOVED, titles):
        t = oldtables[key]
        # tab:runtime was main Table 7/Figure4 previously; main is now Table4/Figure3.
        t = t.replace(r"Table~\ref{tab:runtime}", "main Table 4")
        t = t.replace(r"\label{tab:"+key+"}", r"\label{relocated:"+key+"}")
        assert body(t) == body(oldtables[key])
        reloc.append(t)
        ledger.append(dict(previous_label="tab:"+key, current_location=f"Supplementary Table S{n}",
                           body_sha256=hashlib.sha256(body(t).encode()).hexdigest()))
    reloc.extend([r"\FloatBarrier\section{Relocated methodological detail and prototype audit}"])
    for title in ["Bounded prior-only context", "Training and weighted aggregation"]:
        block = subsection(before, title)
        block = block.replace(r"\label{eq:mixture}", r"\label{relocated:mixture}")
        reloc.append(block)
    reloc.extend([r"\subsection{History ownership and resource measurement details}",
                  "The shared-density history control uses the same random draw as packet-private ownership and matching role fractions. "
                  "A packet can enter more than one shared-density ring, whereas private rings have no shared records. "
                  "Mapped eligible targets are restricted to their own role in both controls. "
                  "The full-stream clock and original resets are retained. Thinning changes density, membership and warm history; "
                  "it does not create independent recording environments. Twenty fixed-setting expert fits supply these controls.",
                  "For the purge comparison, ten seconds retains 50,964 of 52,584 eligible development targets and "
                  "10,078 of 10,358 temporal V targets. Common-target inputs and predictions are identical at fixed capacity. "
                  "Workload-length tests repeat recorded segments 1/2/4 times with fresh segment histories. They do not independently vary endpoint count. "
                  "Query/update capacity timing uses ten repetitions without model calls. RSS measures inference delta above post-load baseline, "
                  "not model file size. Native singleton and compact ensemble implementations are not equally optimized.",
                  r"\subsection{Additional-benign alignment and probability diagnosis}",
                  "All 760 selected packet identities are unique within their source prefixes, and every mapped CSV row index differs "
                  "from its packet index. Timestamp offsets are consistently one hour. Numeric reconstructed inputs differ from saved "
                  r"values by at most $7.11\times10^{-15}$; categorical values agree exactly. These checks validate supplied legacy CSV fields, "
                  "not a second independent dissection of every original protocol field. No capacity evictions occur in these prefixes.",
                  "Before alignment, SQL alarms concentrate in Distance (77/93), Flame Sensor (73/94), IR Receiver (69/97), "
                  "Sound Sensor (81/92) and Water Level (24/93). Heart Rate, phValue and Soil Moisture have no observed alarms. "
                  "Three text-format corrections reduce WV alarms by three net cases. On aligned errors, weighted SQL-minus-Normal "
                  "contributions are +0.1493 DT, -0.0975 RF, -0.1366 XGBoost, +0.0021 LightGBM and +0.1825 LR. "
                  "Their rounded sum is +0.0998; this is probability aggregation rather than majority voting. "
                  "Training-source context profiles, source support and all outcomes are in Tables S37 and S41--S44."])
    # Keep the complete original intervention specification and audit (not just its scores).
    for title in ["Post-observation payload and disagreement controls", "Raw-payload provenance and diagnostic probes"]:
        block = subsection(before, title)
        block = re.sub(r"\\label\{sec:[^}]+\}", "", block)
        block = block.replace(r"\label{eq:disagreement}", r"\label{relocated:disagreement}")
        block = block.replace(r"Equation~\ref{eq:disagreement}", r"Equation~\ref{relocated:disagreement}")
        block = block.replace(r"Eq.~\ref{eq:certificate}", "main Eq. (2)")
        reloc.append(block)
    reloc.extend([r"\subsection{External extraction and label checks}",
                  "External features come from PCAP, using TShark 4.4.9 and the original ring definition on every packet. "
                  "Missing parser columns fail the gate; true protocol absence uses the training-compatible absent representation. "
                  "PCAPNG conversion preserves packet count and endpoint timestamps numerically exactly. "
                  "A 94-target Edge control reproduces all eighty-five fields.",
                  "The 26.6-million-packet SYN file exceeds available native-dissection RAM. Its substitution retains complete "
                  "bidirectional TCP flows containing the fixed targets, all non-TCP packets, IP fragments and TCP DNS, "
                  "with original bytes, order and timestamps. Context still uses every original packet. "
                  "On 1,000 CIC Normal3 targets, the selected-flow and native paths match all eighty-five features. "
                  "Native equality on the complete SYN file was not measured; failures and target IDs are archived.",
                  "All twenty-three processed TON network CSVs are checked by size/SHA before bidirectional interval matching. "
                  "Duration must be nonnegative. The quantization extension precedes TON scoring. Unmatched, conflicting and "
                  "out-of-scope records receive no inferred Normal label. Selected labels are checked against every potentially matching flow. "
                  "Coverage remains 62.90\\%/6.66\\%, with 128,482/1,777,764 unmapped packets; selected-target results are not "
                  "extrapolated to complete captures.",
                  r"\subsection{Prototype and reproducibility}"])
    prototype = before.split(r"\section{Prototype and reproducibility}",1)[1].split(r"\section{Discussion",1)[0]
    reloc.append(prototype)
    reloc.extend([r"\FloatBarrier\section{Relocated diagnostic figures}",
                  "These figures retain the preceding main-manuscript evidence with supplementary numbering. "
                  "Figure S7 describes history ownership, Figure S8 the earlier payload-assisted rule, and Figure S9 actual payload provenance. "
                  "The rule in Figure S8 is distinct from final GV, which has no payload classifier."])
    for key in ["history", "repair", "payload_audit"]:
        f = figure(before,key).replace(r"\label{fig:"+key+"}", r"\label{relocated_fig:"+key+"}")
        reloc.append(f)
    # Minimal editorial cross-reference change, outside every preserved numeric body.
    supplement = sup0.replace("main Table 7 and Figure 4", "main Table 4 and Figure 3")
    supplement = supplement.replace(
        "Earlier statements that no external test had yet been performed refer to the historical stage described in those archived sections.",
        "Earlier statements that no external test had yet been performed refer to the historical stage described in those archived sections. "
        "For the concise edition, Tables S96--S104, Figures S7--S9 and relocated methodological details follow the existing evidence; "
        "the numerical bodies of Tables S1--S95 are unchanged.")
    supplement = supplement.replace(
        r"\FloatBarrier\end{document}",
        "\n\n"+"\n\n".join(reloc)+
        r"\FloatBarrier\begingroup\small\setlength{\bibsep}{3pt}\bibliographystyle{elsarticle-num}\bibliography{references}\endgroup\end{document}")
    assert len(tables(manuscript)) == 7 and len(tables(supplement)) == 104
    for a,b in zip(tables(sup0),tables(supplement)):
        assert body(a)==body(b)
    save(NEW/"main.tex", manuscript)
    save(NEW/"supplement.tex", supplement)
    summary=dict(status="EDITORIAL_CONDENSATION_NO_NEW_EXPERIMENTS",
                 parent_source=OLD.relative_to(ROOT).as_posix(),
                 source=NEW.relative_to(ROOT).as_posix(), main_tables=7, main_figures=5,
                 supplement_tables=104,supplement_figures=9,main_retained_verbatim_tables=KEEP,
                 relocated_tables=ledger,previous_supplement_table_bodies_preserved=95,
                 bibliography_unchanged=sha(OLD/"references.bib")==sha(NEW/"references.bib"),
                 abstract_words=len(abstract.split()),new_fits=0,new_inference=0)
    save(OUT/"build_summary.json",json.dumps(summary,indent=2)+"\n")
    save(OUT/"main_changes.diff","".join(difflib.unified_diff(before.splitlines(True),manuscript.splitlines(True),
               fromfile="integrated/main.tex",tofile="concise/main.tex")))
    save(OUT/"supplement_changes.diff","".join(difflib.unified_diff(sup0.splitlines(True),supplement.splitlines(True),
               fromfile="integrated/supplement.tex",tofile="concise/supplement.tex")))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
