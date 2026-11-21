# Source code for the reported experiments

| Study | Entry point under EVIDENCE/files |
|---|---|
| RQ1 factorial | scripts/run_array_factorial_20261103.py |
| History controls | scripts/run_array_history_models_20261103.py |
| GNB | scripts/run_array_gnb_20261119.py |
| RQ2 serving ablation | scripts/profile_array_serving_20261117.py |
| Seven-group implementation check | scripts/verify_exact_xai.py |
| RQ3 nine-group audit | scripts/run_array_xai_20261103.py |
| CIC external extraction/scoring | scripts/run_independent_capture_v2_20261109.py |
| TON external mapping/scoring | scripts/run_ton_independent_20261109.py |
| GV construction/scoring | scripts/run_fpr_normal_voting_20261109.py; scripts/predict_fpr_sql_guard_rf_rescue_20261109.py |

Source snapshots also accompany individual runs, together with configurations, environments and hashes. Current saved-result verifiers are at the repository root. Figure source and numeric input tables are under manuscript/figure_sources.

The historical GNB runner imports shared stage-input/hash-gating helpers from run_array_svm_20261118.py and shared label metrics from array_svm_20261118.py. Those two original source files are included unchanged to preserve the actual dependency chain. Their filenames reflect source history; adding the dependency files does not add SVM results to the current WV/GNB predictive comparison. RF remains an expert of WV and a component of GV; source describing it is essential.

SOURCE_CODE_AUDIT.json checks local-module availability through static import closure from the listed experimental entry points. It does not certify a complete end-to-end rerun. Provider data, omitted feature caches/models, path adaptation and the recorded experimental environments remain necessary for full reconstruction. Never load an unverified serialized model.
