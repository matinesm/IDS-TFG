"""
Explores UNSW-NB2015 as a training base across three train/test combinations with IDS2017 and InSDN2020.

Output: output/results/results_experiment0.json

"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
TFG_DATA_DIR = os.environ["TFG_DATA_DIR"]
PATHS = {
    2015: os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_nb2015.csv"),
    2017: os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_ids2017.csv"),
    2020: os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_insdn2020.csv"),
}
LABEL_COL = "attack_cat"

OUTPUT_BASE  = os.path.join(os.path.dirname(__file__), "..", "..", "output")
RESULTS_DIR  = os.path.join(OUTPUT_BASE, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results_experiment0.json")

RANDOM_STATE   = 42
N_ESTIMATORS   = 200
BALANCE_TARGET = 30_000

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

EXPERIMENTS = [
    {"name": "UNSW-NB2015 → IDS2017",              "train": [2015],       "test": 2017},
    {"name": "UNSW-NB2015 → InSDN2020",            "train": [2015],       "test": 2020},
    {"name": "UNSW-NB2015 + IDS2017 → InSDN2020",  "train": [2015, 2017], "test": 2020},
]


def load(year):
    df = pd.read_csv(PATHS[year], low_memory=False)
    df.columns = df.columns.str.strip()
    df["unified_label"] = df[LABEL_COL].astype(str).str.strip().map(TAXONOMY)
    df["year"] = year
    return df[df["unified_label"].notna()].copy().reset_index(drop=True)


def unify_columns(dfs):
    """Keeps only numeric columns common to all three datasets."""
    meta = {"unified_label", "year"}
    common = None
    for df in dfs.values():
        num = set(df.select_dtypes(include=[np.number]).columns) - meta
        common = num if common is None else common & num
    frames = [df[[c for c in list(common) + list(meta) if c in df.columns]] for df in dfs.values()]
    return pd.concat(frames, ignore_index=True)


def balance(X, y, target=BALANCE_TARGET):
    counts = pd.Series(y).value_counts()
    under = {c: target for c, n in counts.items() if n > target}
    over  = {c: target for c, n in counts.items() if n < target}
    steps = []
    if under:
        steps.append(("u", RandomUnderSampler(sampling_strategy=under, random_state=RANDOM_STATE)))
    if over:
        k = min(5, counts.min() - 1)
        steps.append(("o", SMOTE(sampling_strategy=over, k_neighbors=k, random_state=RANDOM_STATE)))
    if not steps:
        return X, y
    return ImbPipeline(steps).fit_resample(X, y)


def run_experiment(df, feature_cols, le, exp):
    name = exp["name"]
    train_mask = df["year"].isin(exp["train"])
    test_mask  = df["year"] == exp["test"]

    train_classes = set(df.loc[train_mask, "unified_label"].unique())
    y_te_raw = df.loc[test_mask, "unified_label"]
    unseen = set(y_te_raw.unique()) - train_classes

    valid = y_te_raw.isin(train_classes)
    X_te = df.loc[test_mask, feature_cols][valid].values
    y_te = le.transform(y_te_raw[valid])

    X_tr_raw = df.loc[train_mask, feature_cols].values
    y_tr_raw = le.transform(df.loc[train_mask, "unified_label"])
    X_tr, y_tr = balance(X_tr_raw, y_tr_raw)

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    labels = np.unique(np.concatenate((y_te, y_pred)))
    names  = le.inverse_transform(labels)
    bal = balanced_accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, labels=labels, target_names=names,
                                   zero_division=0, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]

    print(f"\n{'='*60}\n  {name}")
    if unseen:
        print(f"  (classes unseen during training, excluded from test: {unseen})")
    print(f"  Balanced accuracy: {bal:.4f}  |  F1 macro: {f1_macro:.4f}\n{'='*60}")
    print(classification_report(y_te, y_pred, labels=labels, target_names=names, zero_division=0))

    return {"name": name, "balanced_accuracy": bal, "f1_macro": f1_macro, "classification_report": report,
            "train_classes": sorted(train_classes), "excluded_test_classes": sorted(unseen)}


if __name__ == "__main__":

    print("Loading datasets (NB2015, IDS2017, InSDN2020)...")
    dfs = {year: load(year) for year in [2015, 2017, 2020]}
    df = unify_columns(dfs)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    meta = {"unified_label", "year"}
    feature_cols = [c for c in df.columns if c not in meta]

    le = LabelEncoder()
    le.fit(df["unified_label"])
    print(f"Classes: {list(le.classes_)}")
    print(f"Features common to all 3 datasets: {len(feature_cols)}")

    results = [run_experiment(df, feature_cols, le, exp) for exp in EXPERIMENTS]

    print(f"\n\n{'='*60}\nSUMMARY - Balanced accuracy per experiment\n{'='*60}")
    for r in results:
        print(f"  {r['name']:<42}  {r['balanced_accuracy']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "experiment": "Experiment 0 (UNSW-NB2015 cross-domain)",
        "description": "Cross-domain combinations using UNSW-NB2015 as training base, "
                        "before settling on CIC-IDS2017 (Experiments 1-3)",
        "classes": list(le.classes_),
        "n_train": None,
        "n_test": None,
        "confusion_matrix": None,
        "class_balance": None,
        "samples_per_split": None,
        "experiments": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n-> Results saved to: {RESULTS_FILE}")
