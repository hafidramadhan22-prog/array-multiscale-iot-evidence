"""Reconstruct manuscript metrics directly from frozen predictions/coalition arrays.

No estimator imports, fitting, model loading or explanation inference. Formulas
here do not call the experiment's metric or Shapley implementation.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/CROSSCHECK_DOSPEM_20261022_01"
EVIDENCE = ROOT / "paper/array_20261022_integrated/evidence"
CLASSES = ["Normal", "DDoS_TCP", "Port_Scanning", "SQL_injection", "Ransomware"]
EXPERTS = ["decision_tree", "random_forest", "xgboost", "lightgbm", "logistic_regression"]
CATALOGS = {}
FILES = {}
CHECKS = []
MAX_ERROR = 0.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def checked(run, relative):
    base = ROOT / "runs" / run
    if run not in CATALOGS:
        CATALOGS[run] = {r["path"].replace("\\", "/"): r for r in json.loads((base / "artifacts_sha256.json").read_text())}
    record = CATALOGS[run][relative]
    path = base / relative
    assert sha(path) == record["sha256"] and path.stat().st_size == record["bytes"]
    FILES[path.relative_to(ROOT).as_posix()] = record["sha256"]
    return path


def close(a, b, label, tol=1e-12):
    global MAX_ERROR
    error = abs(float(a) - float(b))
    assert error <= tol, (label, a, b, error)
    MAX_ERROR = max(MAX_ERROR, error)
    CHECKS.append(label)


def confusion(y, p):
    assert np.isfinite(p).all() and p.shape == (len(y), 5)
    assert np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (y, p.argmax(1)), 1)
    return matrix


def metrics(cm):
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = cm.diagonal()
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(5), where=(support + predicted) != 0)
    return dict(macro_f1=float(f1.mean()), ddos_f2=float(5 * tp[1] / (5 * tp[1] + 4 * (support[1] - tp[1]) + predicted[1] - tp[1])),
        accuracy=float(tp.sum() / cm.sum()), **{"recall_" + c: float(tp[i] / support[i]) for i, c in enumerate(CLASSES)})


def stage_metrics(run, view, case):
    rows = []
    for stage in ["temporal_seed42", "temporal_seed43", "temporal_seed44", "signature_fold1", "signature_fold2"]:
        relative = f"stages/{stage}/{view}__{case}/evaluation.npz"
        with np.load(checked(run, relative), allow_pickle=False) as a:
            y, ids = a["y"], a["row_ids"]
            assert len(set(map(int, ids))) == len(ids)
            probs = {"weighted_voting": a["p_weighted"], "soft_voting": a["p_soft"]}
            if "p_reference" in a.files:
                probs["weighted_reference"] = a["p_reference"]
            probs.update({n: a["p_experts"][:, j, :] for j, n in enumerate(EXPERTS)})
            rows.append((stage, ids.copy(), {n: confusion(y, p) for n, p in probs.items()}))
    return rows


def summary(rows, method):
    temporal = [metrics(r[2][method]) for r in rows if r[0].startswith("temporal")]
    signature = [r for r in rows if r[0].startswith("signature")]
    pooled_ids = np.concatenate([r[1] for r in signature])
    assert len(pooled_ids) == len(np.unique(pooled_ids)) == 52584
    out = metrics(sum(r[2][method] for r in signature))
    for name in temporal[0]:
        out[name + "_mean"] = float(np.mean([r[name] for r in temporal]))
        out[name + "_std"] = float(np.std([r[name] for r in temporal], ddof=1))
    return out


def main():
    decoded = {}
    mapping = {"baseline25": ("WB_RESEARCH_FULL_20261019_01", "legacy25_symmetric"),
               "baseline29": ("WB_RESEARCH_FULL_20261019_01", "legacy29_symmetric"),
               **{v: ("MULTISCALE_REPAIR_FULL_20261020_01", v) for v in ["multiscale60", "legacy25_multiscale60", "legacy25_multiscale_directional96"]}}
    quality = read(EVIDENCE / "quality_multiscale_all_treatments.csv")
    for row in quality:
        view, case = row["view"], row["case"]
        key = (view, case)
        run, source_view = mapping[view]
        decoded[key] = stage_metrics(run, source_view, case)
        result = summary(decoded[key], "weighted_voting")
        for name, value in result.items():
            if name in row:
                close(value, row[name], f"raw_quality/{view}/{case}/{name}")
    for row in read(EVIDENCE / "same_feature_controls.csv"):
        result = summary(decoded[("legacy25_multiscale60", row["case"])], row["method"])
        for name, value in result.items():
            if name in row:
                close(value, row[name], f"raw_control/{row['case']}/{row['method']}/{name}")
    time_rows = read(EVIDENCE / "quality_time_extension.csv")
    for row, view in zip(time_rows[2:], ["time_context84", "legacy25_time_context84", "legacy25_age_context66"]):
        result = summary(stage_metrics("IAT_REPAIR_FULL_20261020_01", view, "balanced"), "weighted_voting")
        for field, name in [("Macro-F1 temporal (%)", "macro_f1_mean"), ("Macro-F1 signature (%)", "macro_f1"),
                            ("Recall scan signature (%)", "recall_Port_Scanning"), ("Recall ransom signature (%)", "recall_Ransomware")]:
            close(100 * result[name], row[field], f"raw_time/{view}/{name}", 1e-10)
    # Historical predictions and all four conference policies: independently derive label metrics.
    for run, pattern, metric_file, config_field in [
        ("FULL_20261013_01", "dstport29/tables/holdout_predictions_*.csv", "dstport29/tables/holdout_all_six.csv", "config"),
        ("CONFERENCE_COMPARISON_FULL_20261022_01", "predictions/*.csv", "holdout_metrics.csv", "config"),
    ]:
        scores = {r[config_field]: r for r in read(checked(run, metric_file))}
        for p in sorted((ROOT / "runs" / run).glob(pattern)):
            rel = p.relative_to(ROOT / "runs" / run).as_posix()
            rows = read(checked(run, rel))
            name = p.stem.removeprefix("holdout_predictions_")
            y = np.array([int(r["true_class"]) for r in rows])
            probs = np.array([[float(r["prob_" + c]) for c in CLASSES] for r in rows])
            result = metrics(confusion(y, probs))
            for key in ["macro_f1", "ddos_f2", "accuracy"]:
                close(result[key], scores[name][key], "raw_holdout/" + name + "/" + key)
    # Derive Shapley values from the coalition game, then cross-donor deletion/insertion.
    games = {}
    for name in ["A32", "B32", "query_delay_5ms"]:
        path = checked("XAI_MULTISCALE_FULL_20261020_01", f"groups/{name}/coalitions.npz")
        with np.load(path, allow_pickle=False) as a:
            games[name] = {k: a[k].copy() for k in a.files if k != "seconds"}
        values = games[name]["values"]
        phi = np.zeros_like(games[name]["phi"])
        for group in range(7):
            for mask in range(128):
                if (mask >> group) & 1:
                    continue
                size = mask.bit_count()
                weight = math.factorial(size) * math.factorial(6 - size) / math.factorial(7)
                phi[:, group] += weight * (values[:, mask | (1 << group)] - values[:, mask])
        close(np.max(np.abs(phi - games[name]["phi"])), 0, "coalition_shapley/" + name)
    assert len(np.intersect1d(games["A32"]["background_ids"], games["B32"]["background_ids"])) == 0
    np.testing.assert_array_equal(games["A32"]["row_ids"], games["B32"]["row_ids"])
    faith = {r["method"]: r for r in read(EVIDENCE / "faithfulness.csv")}
    by_method = {}
    for name, index in [("weighted_voting", 0), ("random_forest", 2), ("logistic_regression", 5)]:
        deletion, insertion, random = [], [], []
        for i in range(100):
            cls = int(games["A32"]["values"][i, -1, index].argmax())
            attribution = games["A32"]["phi"][i, :, index, cls]
            order = sorted(range(7), key=lambda g: (-attribution[g], g))
            donor = games["B32"]["values"][i, :, index, cls]
            added = 0
            dc, ic = [donor[127]], [donor[0]]
            for g in order:
                added |= 1 << g
                dc.append(donor[127 ^ added])
                ic.append(donor[added])
            expected_random = [np.mean([donor[m] for m in range(128) if m.bit_count() == k]) for k in range(8)]
            integrate = lambda x: float(np.trapezoid(x, dx=1 / 7))
            deletion.append(integrate(dc)); insertion.append(integrate(ic)); random.append(integrate(expected_random))
        result = dict(deletion_auc=np.mean(deletion), insertion_auc=np.mean(insertion),
                      deletion_gain=np.mean(random) - np.mean(deletion), insertion_gain=np.mean(insertion) - np.mean(random))
        for key, value in result.items():
            close(value, faith[name][key], "cross_donor_faithfulness/" + name + "/" + key)
        by_method[name] = {k: float(v) for k, v in result.items()}
    runtime = read(checked("EXACT_RUNTIME_20261020_01", "replay_raw.csv"))
    times = {n: [float(r["total_seconds"]) for r in runtime if r["policy"] == "baseline25__" + n] for n in ["native_shared", "native_certified", "compact_certified"]}
    assert all(len(v) == 5 for v in times.values())
    reduction = 100 * (1 - np.mean(times["compact_certified"]) / np.mean(times["native_certified"]))
    close(round(reduction, 2), 34.46, "mean_replay_reduction_rounded")
    slower = sum(c > n for c, n in zip(times["compact_certified"], times["native_certified"]))
    assert slower == 1
    result = dict(status="PASS_INDEPENDENT_RAW_PREDICTIONS_AND_COALITION_RECONSTRUCTION",
        checks=len(CHECKS), check_names=CHECKS, raw_arrays_and_tables_verified=len(FILES), source_sha256=FILES,
        largest_numeric_error=MAX_ERROR, cross_donor_faithfulness=by_method,
        replay_reduction_percent=float(reduction), compact_repeats_slower=slower,
        new_fitting=False, model_deserialization=False, new_game_inference=False,
        note="Reconstructs stored probabilities/coalition values, not a new independent scientific evaluation.")
    (OUT / "INDEPENDENT_RAW_CHECKS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["check_names", "source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
