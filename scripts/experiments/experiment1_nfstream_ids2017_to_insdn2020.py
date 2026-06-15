"""
NFStream pipeline: trains on CIC-IDS2017 and evaluates on InSDN2020, with and without window features.

Output: output/results/results_experiment1.json

"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
TFG_DATA_DIR = os.environ["TFG_DATA_DIR"]
PATH_TRAIN = os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_ids2017.csv")
PATH_TEST  = os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_insdn2020.csv")
LABEL_COL  = "attack_cat"

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "output")
RESULTS_DIR = os.path.join(OUTPUT_BASE, "results")

RANDOM_STATE   = 42
BALANCE_TARGET = 30_000
WINDOW_SIZE    = 500
VAL_SIZE       = 0.20

TAXONOMY = {
    "BENIGN": "Normal", "Normal": "Normal",
    "DoS": "DoS_DDoS", "DDoS": "DoS_DDoS",
    "DoS Hulk": "DoS_DDoS", "DoS GoldenEye": "DoS_DDoS",
    "DoS Slowhttptest": "DoS_DDoS", "DoS slowloris": "DoS_DDoS",
    "Reconnaissance": "Scan", "PortScan": "Scan", "Probe": "Scan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Web Attack – Brute Force": "BruteForce",
    "BFA": "BruteForce", "Brute Force": "BruteForce",
    "Web Attack – XSS": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web-Attack": "WebAttack", "Web Attack": "WebAttack",
    "Bot": "Botnet", "BOTNET": "Botnet",
    "Exploits": None, "Fuzzers": None, "Generic": None,
    "Shellcode": None, "Analysis": None, "Backdoor": None,
    "Backdoors": None, "Worms": None, "Infiltration": None,
    "Heartbleed": None, "U2R": None,
}

DROP_COLUMNS = {
    "src_ip", "dst_ip", "src_mac", "dst_mac", "src_oui", "dst_oui", "src_port",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "id", "expiration_id", "application_confidence",
}

RF_CONFIG = dict(
    n_estimators=500, min_samples_leaf=30,
    max_features=0.3, max_depth=20,
    random_state=RANDOM_STATE, n_jobs=-1,
)


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["unified_label"] = df[LABEL_COL].astype(str).str.strip().map(TAXONOMY)
    df = df[df["unified_label"].notna()].copy().reset_index(drop=True)
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


def window_features(df, w):
    """Per-flow statistics over the previous `w` flows to the same dst port."""
    if "dst_port" not in df.columns:
        return df, []
    gv = df["dst_port"].values
    bv = df["bidirectional_bytes"].values if "bidirectional_bytes" in df.columns else np.ones(len(df))
    pv = df["bidirectional_packets"].values if "bidirectional_packets" in df.columns else np.ones(len(df))
    fc = np.zeros(len(df), dtype=np.float32)
    bs = np.zeros(len(df), dtype=np.float32)
    pr = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        s = max(0, i - w)
        mask = gv[s:i] == gv[i]
        n = mask.sum()
        fc[i] = n
        if n > 0:
            bs[i] = bv[s:i][mask].sum()
            pr[i] = pv[s:i][mask].mean()
    cols = [f"win{w}_flow_count", f"win{w}_byte_sum", f"win{w}_pkt_rate"]
    df = df.copy()
    df[cols[0]], df[cols[1]], df[cols[2]] = fc, bs, pr
    return df, cols


def balance(X, y):
    counts = pd.Series(y).value_counts()
    under  = {c: BALANCE_TARGET for c, n in counts.items() if n > BALANCE_TARGET}
    over   = {c: BALANCE_TARGET for c, n in counts.items() if n < BALANCE_TARGET}
    steps  = []
    if under:
        steps.append(("u", RandomUnderSampler(sampling_strategy=under, random_state=RANDOM_STATE)))
    if over:
        k = min(5, counts.min() - 1)
        steps.append(("o", SMOTE(sampling_strategy=over, k_neighbors=k, random_state=RANDOM_STATE)))
    if not steps:
        return X, y
    return ImbPipeline(steps).fit_resample(X, y)


def apply_thresholds(probas, thresholds):
    y_pred = np.zeros(probas.shape[0], dtype=int)
    for i in range(probas.shape[0]):
        p = probas[i]
        above = p >= thresholds
        y_pred[i] = np.where(above)[0][np.argmax(p[above])] if above.any() else np.argmax(p)
    return y_pred


def calibrate_thresholds(probas_val, y_val, n_classes):
    grid = np.arange(0.05, 0.95, 0.05)
    best = np.full(n_classes, 0.5)
    for ci in range(n_classes):
        best_f1, best_t = -1, 0.5
        for t in grid:
            trial = best.copy(); trial[ci] = t
            f1_t = f1_score(y_val, apply_thresholds(probas_val, trial), average="macro", zero_division=0)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, t
        best[ci] = best_t
    return best


def evaluate(y_true, y_pred, le, title):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    names  = le.inverse_transform(labels)
    bal    = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   zero_division=0, output_dict=True)
    print(f"\n--- {title} ---")
    print(f"Balanced accuracy: {bal:.4f}   F1 macro: {f1_mac:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    return bal, f1_mac, report


def class_balance_data(y_tr, y_tr_bal, le):
    before = pd.Series(le.inverse_transform(y_tr)).value_counts().reindex(le.classes_, fill_value=0)
    after  = pd.Series(le.inverse_transform(y_tr_bal)).value_counts().reindex(le.classes_, fill_value=0)
    return {"classes": list(le.classes_), "before": before.tolist(), "after": after.tolist()}


def confusion_matrix_data(y_true, y_pred, le, title):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": list(le.inverse_transform(labels)), "matrix": cm.tolist(), "title": title}


def samples_per_split_data(y_tr, y_val, y_te, le):
    rows = {split: pd.Series(le.inverse_transform(y)).value_counts().reindex(le.classes_, fill_value=0)
            for split, y in [("train", y_tr), ("val", y_val), ("test", y_te)]}
    table = pd.DataFrame(rows)
    print("\nSamples per class and split (Table 4.1):")
    print(table)
    return {"classes": list(le.classes_), "train": table["train"].tolist(),
            "val": table["val"].tolist(), "test": table["test"].tolist()}


if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading datasets...")
    df_train = load(PATH_TRAIN)
    df_test  = load(PATH_TEST)

    print(f"Computing window features (w={WINDOW_SIZE})...")
    df_train, win_cols = window_features(df_train, WINDOW_SIZE)
    df_test,  _        = window_features(df_test,  WINDOW_SIZE)

    meta       = {"unified_label"}
    num_tr     = set(df_train.select_dtypes(include=[np.number]).columns) - meta
    num_te     = set(df_test.select_dtypes(include=[np.number]).columns)  - meta
    feats_full = sorted(num_tr & num_te)
    feats_base = [c for c in feats_full if c not in win_cols]
    print(f"Features: {len(feats_base)} base + {len(feats_full)-len(feats_base)} window = {len(feats_full)} total")

    le = LabelEncoder()
    le.fit(pd.concat([df_train["unified_label"], df_test["unified_label"]]).unique())
    n_cls = len(le.classes_)
    print(f"Classes: {list(le.classes_)}")

    y_all = le.transform(df_train["unified_label"])
    y_te  = le.transform(df_test["unified_label"])
    idx_tr, idx_val = train_test_split(
        np.arange(len(df_train)), test_size=VAL_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    y_tr, y_val = y_all[idx_tr], y_all[idx_val]
    print(f"\nTrain: {len(idx_tr):,}  Val: {len(idx_val):,}  Test: {len(df_test):,}")

    CONFIGURATIONS = [
        ("calibrated",             feats_base, "IDS2017 → InSDN2020, calibrated"),
        ("calibrated_with_window", feats_full, "IDS2017 → InSDN2020, calibrated + window"),
    ]

    result = {
        "experiment": "Experiment 1 (NFStream, IDS2017 → InSDN2020)",
        "description": "Training on CIC-IDS2017, evaluating on InSDN2020 (real domain shift), "
                        "comparing calibrated and calibrated+window configurations",
        "classes": list(le.classes_),
    }
    final_artifacts = {}

    for key, feats, title in CONFIGURATIONS:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        X_all = df_train[feats].values.astype(float)
        X_tr, X_val, X_te = X_all[idx_tr], X_all[idx_val], df_test[feats].values.astype(float)

        X_tr_bal, y_tr_bal = balance(X_tr, y_tr)
        print(f"Balanced train: {len(X_tr_bal):,}")

        print("Training RF...")
        rf = RandomForestClassifier(**RF_CONFIG)
        rf.fit(X_tr_bal, y_tr_bal)

        bal_05, f1_05, _ = evaluate(y_te, rf.predict(X_te), le, f"{title} | threshold 0.5")

        print("Calibrating thresholds on validation set...")
        thresholds = calibrate_thresholds(rf.predict_proba(X_val), y_val, n_cls)
        for i, cls in enumerate(le.classes_):
            print(f"  {cls:<14} {thresholds[i]:.2f}")

        y_pred_cal = apply_thresholds(rf.predict_proba(X_te), thresholds)
        bal_cal, f1_cal, report_cal = evaluate(y_te, y_pred_cal, le, f"{title} | calibrated threshold")

        importances = pd.Series(rf.feature_importances_, index=feats).nlargest(15)
        print("\nTop 15 features:")
        for i, (feat, val) in enumerate(importances.items(), 1):
            print(f"  {i:>2}. {feat:<40} {val:.4f}")

        result[key] = {
            "threshold_05": {"balanced_accuracy": bal_05, "f1_macro": f1_05},
            "calibrated":   {"balanced_accuracy": bal_cal, "f1_macro": f1_cal,
                             "classification_report": report_cal},
            "per_class_thresholds": {cls: float(t) for cls, t in zip(le.classes_, thresholds)},
            "top_features": {feat: float(val) for feat, val in importances.items()},
            "n_features": len(feats),
        }

        if key == "calibrated_with_window":
            final_artifacts = dict(y_tr=y_tr, y_tr_bal=y_tr_bal, y_pred_cal=y_pred_cal, bal_cal=bal_cal)

    result["n_train"] = len(final_artifacts["y_tr_bal"])
    result["n_test"] = len(df_test)
    result["class_balance"] = class_balance_data(final_artifacts["y_tr"], final_artifacts["y_tr_bal"], le)
    result["confusion_matrix"] = confusion_matrix_data(
        y_te, final_artifacts["y_pred_cal"], le,
        f"IDS2017 → InSDN2020, calibrated + window  |  Bal.Acc: {final_artifacts['bal_cal']:.3f}")
    result["samples_per_split"] = samples_per_split_data(y_tr, y_val, y_te, le)

    print(f"\n{'='*70}\nSUMMARY - configurations cited in TAB:COMPARATIVA\n{'='*70}")
    for key, _, title in CONFIGURATIONS:
        r = result[key]["calibrated"]
        print(f"  {title:<50} bal={r['balanced_accuracy']:.4f}  F1 macro={r['f1_macro']:.4f}")

    results_file = os.path.join(RESULTS_DIR, "results_experiment1.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n-> Results saved to: {results_file}")
