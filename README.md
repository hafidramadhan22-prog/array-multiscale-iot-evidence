# Multiscale endpoint context and prediction-preserving execution

Research evidence accompanying the Array manuscript by M Hafid Ramadhan and Satria Mandala. This repository supports verification of saved experiments; publication of these files does not indicate journal acceptance.

## Evidence map

| Archive | Evidence | Main use |
|---|---|---|
| [R1](R1/index.json) | Split and target IDs, training plans, frozen configurations, saved predictions; WV, GNB, RF, LR, uniform voting, SVM and KNN historical results | RQ1 and Tables 1–3 |
| [R2](R2/index.json) | Encoding/posterior checks; 480 feature measurements, 240 replay measurements, 80 fresh-process RSS measurements; separate earlier sessions | RQ2 and Table 4 |
| [R3](R3/index.json) | Seven- and nine-group coalition values, Shapley arrays, donor IDs, interventions and feature rankings | RQ3 and Table 5 |
| [R4](R4/index.json) | Selected packet-to-flow record mappings, coverage, local WV/GV predictions and external per-file results | Tables 6–7 and false-alarm diagnostics |

Indexes list repository-relative paths and SHA256 checksums. Original experimental directory names remain under `EVIDENCE/files` to preserve provenance and allow the verification scripts to run. Source snapshots and run-specific environments are retained there; original historical manifests describe their original runs, not necessarily all files in this public subset. `SHA256_MANIFEST.json` is the public evidence inventory.

## Verify saved results

Use Python 3.11 in a virtual environment:

```sh
python -m pip install -r requirements-verification.txt
python EVIDENCE/verify_saved_results.py --output local_verification.json
python verify_extended.py
```

The first command reconstructs 70 prediction vectors across ten predictors and three protocols, 45 displayed class recalls, selected manuscript tables, runtime record counts, and TON coverage denominators. The second recomputes Shapley arrays from archived coalition values and external confusion counts. Neither deserializes estimators, trains a model, obtains new predictions, nor performs fresh raw-PCAP extraction.

For the 17 November crossed serving experiment, compute `100 * (1 - mean(compact_certified) / mean(native_full))` from `EVIDENCE/files/runs/ARRAY_SERVING_ABLATION_20261117_01/replay_raw.csv`. The result is **60.5497700923%**, displayed as **60.55%**. Comparison with native certified execution gives **37.3965218826%**. The earlier 3 November 41.75% session is separate and must not be pooled with this one.

## Interpretation and access scope

Historical predictive comparisons are exploratory. Development and historical holdout outcomes had been observed during model selection. Five class-associated source traces in the main data do not establish cross-source independence; history controls and signature-disjoint targets do not by themselves remove source/class dependence. GNB is the displayed comparator, while stronger historical results remain available here.

The unchanged WV function is the subject of execution and explanation preservation. The modified GV rule requires separate evaluation. Its local false-alarm improvement is not evidence of deployment readiness; adverse external results are retained. Seven-group and nine-group Shapley games are distinct estimands, and no causal interpretation is claimed. The history record array is bounded at 1 MiB, not the entire process. RSS sampling is not complete native-allocation tracing. Runtime evidence concerns one laptop and the measured workloads.

Raw Edge-IIoTset, CICIoT2023 and TON_IoT datasets, packet payloads, full extracted packet arrays, serialized fitted estimators, private correspondence and unrelated working history are excluded. Selected derived target mappings preserve packet indices and provider flow record indices; obtain raw data and original labels from their providers to reconstruct extraction. Provider terms continue to apply. The historical scripts are source snapshots, and some full-training/extraction paths require local adaptation, provider data and omitted estimator artifacts. This is a saved-result reproducibility package, not a claim that every end-to-end experiment is runnable from this repository alone.

No new license is asserted over provider datasets or third-party code. Copyright and existing third-party notices remain with their owners. Contact the corresponding author for reuse permissions not covered by existing notices.

See release `qc-20261121` for the versioned evidence archive and the complete main/supplement LaTeX package. The LaTeX main document is `main.tex`; the companion is `SUPPLEMENTARY/supplement.tex`. Compile both separately. The latest supervisor DOCX is the prose authority; source construction and verification do not constitute author submission approval.
