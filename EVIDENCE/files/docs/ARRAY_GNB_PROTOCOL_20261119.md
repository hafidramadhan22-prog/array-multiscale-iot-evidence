# Gaussian Naive Bayes comparator — 19 November 2026

User authorized standalone GaussianNB, not replacement of WV experts. Preserve
WV, KNN/SVM/RF/LR and the journal. Seven matched stages, same verified source
hashes, A/G/C/V splits, 85 domain features, training-only preprocessing and
Balanced RUS+SMOTE as the KNN/SVM experiment. No new data or holdout selection.

GaussianNB(priors=None). Keep median imputation, A-fitted standard scaling and
one-hot categories; convert the resulting matrix to dense for GaussianNB.
This Gaussian model over mixed encoded features is a simple baseline: categorical
one-hot variables need not satisfy Gaussian assumptions. No feature removal
or deliberate weakening. var_smoothing grid = 10**j, j=-12,...,0 (13 values).
Default=1e-9. Choose only on G: .5 macro-F1+.5 DDoS F2, then macro-F1,
then nearest exponent to -9, then lower exponent. Freeze before V scoring.
Report default AND G-selected. Grid fixed before fit, no post-V grid extension.

Temporal mean/sample SD across 3 seeds; pool disjoint OOF IDs for private and
shared signature. Report macro-F1, FPR, per-class recall, balanced accuracy,
DDoS F2, binary attack recall and errors; save row predictions/CM, split IDs,
training plans, selected models, G candidate predictions and checksum catalogs.
Smoke separate. Verify all G selections/CM/split/cap and unchanged WV results.
Reused development evaluation is exploratory, not independent confirmation.
No external stress or latency/memory benchmark requested in this comparison.
Older tuning budgets remain unequal; do not claim global WV superiority.
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
