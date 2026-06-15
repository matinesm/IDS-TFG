"""
Validates the classifier on CIC-IDS2017 alone (stratified 70/30 split, no domain shift).

Output: output/results/results_validation.json

"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
TFG_DATA_DIR = os.environ["TFG_DATA_DIR"]
PATH_DATASET = os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_ids2017.csv")
LABEL_COL    = "attack_cat"

OUTPUT_BASE  = os.path.join(os.path.dirname(__file__), "..", "..", "output")
RESULTS_DIR  = os.path.join(OUTPUT_BASE, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results_validation.json")

RANDOM_STATE   = 42
TEST_SIZE      = 0.30
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


def balance(X, y, target=BALANCE_TARGET):
    counts = pd.Series(y).value_counts()
    under  = {c: target for c, n in counts.items() if n > target}
    over   = {c: target for c, n in counts.items() if n < target}
    steps  = []
    if under:
        steps.append(("u", RandomUnderSampler(sampling_strategy=under, random_state=RANDOM_STATE)))
    if over:
        k = min(5, counts.min() - 1)
        steps.append(("o", SMOTE(sampling_strategy=over, k_neighbors=k, random_state=RANDOM_STATE)))
    if not steps:
        return X, y
    return ImbPipeline(steps).fit_resample(X, y)


def evaluate(y_true, y_pred, le, title):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    names  = le.inverse_transform(labels)
    bal    = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=names, zero_division=0, output_dict=True)
    print(f"\n--- {title} ---")
    print(f"Balanced accuracy: {bal:.4f}   F1 macro: {f1_mac:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    return bal, f1_mac, report


if __name__ == "__main__":

    print(f"Loading {PATH_DATASET}...")
    df = load(PATH_DATASET)
    print(f"Classes present: {sorted(df['unified_label'].unique())}")

    feats = sorted(set(df.select_dtypes(include=[np.number]).columns) - {"unified_label"})
    print(f"Features: {len(feats)}")

    le = LabelEncoder()
    le.fit(df["unified_label"])

    X = df[feats].values.astype(float)
    y = le.transform(df["unified_label"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_tr):,}   Test: {len(X_te):,}  ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)")

    X_tr_bal, y_tr_bal = balance(X_tr, y_tr)
    print(f"Balanced train: {len(X_tr_bal):,}")

    print("Training RF...")
    rf = RandomForestClassifier(**RF_CONFIG)
    rf.fit(X_tr_bal, y_tr_bal)

    bal, f1_mac, report = evaluate(
        y_te, rf.predict(X_te), le,
        "Intra-IDS2017 validation (stratified 70/30 split)"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "experiment": "Intra-IDS2017 validation",
        "description": "Stratified 70/30 train/test split on CIC-IDS2017 (no domain shift)",
        "balanced_accuracy": bal,
        "f1_macro": f1_mac,
        "classification_report": report,
        "n_train": len(X_tr_bal),
        "n_test": len(X_te),
        "classes": list(le.classes_),
        "confusion_matrix": None,
        "class_balance": None,
        "samples_per_split": None,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n-> Results saved to: {RESULTS_FILE}")
