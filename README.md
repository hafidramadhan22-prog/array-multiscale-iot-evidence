# Array manuscript and evidence final check 22 November 2026

The current main predictive comparison is WV/GNB. R1 contains split/target IDs, frozen settings, training plans and exact saved WV/GNB predictions. A versioned index preserves access to broader historical comparisons; the current subset is not a comprehensive classifier ranking.

| Group | Location and purpose |
|---|---|
| R1 | `R1/predictions`, `R1/derived`, `R1/index.json`: RQ1 and Tables 1–3 |
| R2 | `R2/index.json`: 480 feature calls, 240 replays, 80 RSS records; Table 4 |
| R3 | `R3/index.json`: donor IDs, coalition values, Shapley arrays, interventions and rankings |
| R4 | `R4/index.json`: derived selected packet–flow mappings, WV/GV predictions and confusion counts |

## Run saved-result verification

Use Python 3.11 and a short local directory. On Windows, clone with Git long-path support enabled for this operation:

```sh
git -c core.longpaths=true -c core.autocrlf=false clone https://github.com/hafidramadhan22-prog/array-multiscale-iot-evidence.git
cd array-multiscale-iot-evidence
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements-verification.txt
python verify_saved_results.py --output local_verification.json
python verify_extended.py
```

The repository .gitattributes preserves original file bytes across checkout, so SHA256 checks are unaffected by Windows newline conversion. The saved-results environment uses numpy 2.2.6, pandas 2.2.3 and scikit-learn 1.7.2. Run-specific environment records remain under EVIDENCE/files/runs. Verification reconstructs 21 WV/GNB prediction vectors, 45 per-class recalls, selected main tables, 480/240/80 measurement counts and finite-game Shapley arrays. It does not train, deserialize estimators, obtain fresh classifier predictions or repeat raw-PCAP extraction.

From `EVIDENCE/files/runs/ARRAY_SERVING_ABLATION_20261117_01/replay_raw.csv`, `100*(1-mean(WV_compact_certified)/mean(WV_native_full))` is 60.5497700923%, reported as 60.55%. Native-certified comparison is 37.3965218826%. The earlier 3 November 41.75% session remains separate.

## Scope of this release

The predictive tables focus on GNB; standalone historical comparator result files are indexed through the prior public release rather than duplicated in this subset. No remote history was rewritten. RF remains one of WV's five experts. Original R2 logs retain all eight policies, including RF/LR controls, because removing rows would invalidate the reported 480/240/80 totals. R3 expert axes support weighted-linearity checks; R4 RF-component predictions support the GV diagnostic. These engineering records are retained intact.

Historical development/holdout reuse and adaptive selection constrain predictive interpretation. Source-associated classes and shared histories do not establish cross-source independence. GV changes the prediction function and does not inherit WV serving or explanation guarantees. Adverse external results remain available. No deployment-readiness, causal-SHAP, energy-saving or embedded-hardware claim is made. The 1 MiB history array excludes model/process memory; sampled delta RSS is not exhaustive native-allocation tracing.

Raw third-party datasets, payloads, full packet arrays, fitted estimator binaries and private correspondence are excluded. Selected derived mappings retain packet indices and provider flow-record indices. Provider data and permissions are needed for full raw extraction. Source snapshots may need local path adaptation and omitted artifacts for full retraining. Historical manifests describe their original run; root SHA256_MANIFEST.json is the inventory for this published release. No new license over provider datasets or third-party code is asserted.

## Manuscript files

`manuscript/main.tex` is the main document; `manuscript/SUPPLEMENTARY/supplement.tex` is the companion. Both have editable tables/equations, bibliography and Python-generated figures. Compile separately using Tectonic, or select the corresponding file in Overleaf. The main PDF has 19 pages and the supplement has three pages. The supervisor's 21 November QC revisions are the source; redundant supplementary prose was condensed while retaining all five tables and numerical findings. The main availability paragraph points to this release.

The release includes a complete LaTeX ZIP and a separate evidence ZIP. Packaging and verification scripts were prepared with ChatGPT/Codex assistance. Author review remains necessary; this repository does not record journal submission or final author approval.
