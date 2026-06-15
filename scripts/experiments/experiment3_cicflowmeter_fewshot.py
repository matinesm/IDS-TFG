"""
CICFlowMeter pipeline with 20% few-shot transfer of Botnet and DoS_DDoS from CIC-IDS2018.

Output: output/results/results_experiment3.json

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
TFG_DATA_DIR   = os.environ["TFG_DATA_DIR"]
PATH_TRAIN_DIR = os.path.join(TFG_DATA_DIR, "raw_data/CIC-IDS2017/CSV")
PATH_TEST_DIR  = os.path.join(TFG_DATA_DIR, "raw_data/CIC-IDS2018/CSV")
LABEL_COL = "unified_label"

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "output")
RESULTS_DIR = os.path.join(OUTPUT_BASE, "results")

RANDOM_STATE   = 42
BALANCE_TARGET = 30_000
TRANSFER_RATIO = 0.20

DROP_COLUMNS = {
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Timestamp", "Fwd Header Length.1", "Protocol", "Label",
}

RF_CONFIG = dict(
    n_estimators=500, min_samples_leaf=30,
    max_features=0.3, max_depth=20,
    random_state=RANDOM_STATE, n_jobs=-1,
)

RENAME_2018 = {
    "Dst Port": "Destination Port", "Src IP": "Source IP", "Dst IP": "Destination IP",
    "Tot Fwd Pkts": "Total Fwd Packets", "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets", "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Flow Byts/s": "Flow Bytes/s", "Flow Pkts/s": "Flow Packets/s",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean", "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std", "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Pkt Size Avg": "Average Packet Size", "Bwd Pkts/s": "Bwd Packets/s",
    "Init Fwd Win Byts": "Init_Win_bytes_forward", "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd IAT Tot": "Fwd IAT Total", "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length", "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Pkt Len Min": "Min Packet Length", "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean", "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count", "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count", "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count", "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Subflow Fwd Pkts": "Subflow Fwd Packets", "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets", "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Fwd Act Data Pkts": "act_data_pkt_fwd", "Fwd Seg Size Min": "min_seg_size_forward",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size", "Bwd Seg Size Avg": "Avg Bwd Segment Size",
}

TAXONOMY = {
    "BENIGN": "Normal", "Benign": "Normal",
    "Bot": "Botnet", "BOTNET": "Botnet",
    "DDoS": "DoS_DDoS", "DoS GoldenEye": "DoS_DDoS", "DoS Hulk": "DoS_DDoS",
    "DoS Slowhttptest": "DoS_DDoS", "DoS slowloris": "DoS_DDoS",
    "DDOS attack-HOIC": "DoS_DDoS", "DDOS attack-LOIC-UDP": "DoS_DDoS",
    "DDoS attacks-LOIC-HTTP": "DoS_DDoS", "DoS attacks-GoldenEye": "DoS_DDoS",
    "DoS attacks-Hulk": "DoS_DDoS", "DoS attacks-SlowHTTPTest": "DoS_DDoS",
    "DoS attacks-Slowloris": "DoS_DDoS", "PortScan": "Scan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Web Attack \x96 Brute Force": "BruteForce", "Web Attack – Brute Force": "BruteForce",
    "FTP-BruteForce": "BruteForce", "SSH-Bruteforce": "BruteForce", "Brute Force -Web": "BruteForce",
    "Web Attack \x96 XSS": "WebAttack", "Web Attack – XSS": "WebAttack",
    "Web Attack \x96 Sql Injection": "WebAttack", "Web Attack – Sql Injection": "WebAttack",
    "Brute Force -XSS": "WebAttack", "SQL Injection": "WebAttack",
    "Heartbleed": None, "Infiltration": None, "Infilteration": None, "Label": None,
}


def load_folder(path, rename_cols=None):
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv"))
    frames = []
    for filepath in files:
        df = None
        for enc in ["utf-8", "latin-1"]:
            try:
                df = pd.read_csv(filepath, dtype=str, encoding=enc, on_bad_lines="skip"); break
            except UnicodeDecodeError:
                continue
        if df is None:
            continue
        df.columns = df.columns.str.strip()
        if rename_cols:
            df = df.rename(columns=rename_cols)
        df["unified_label"] = df["Label"].astype(str).str.strip().map(TAXONOMY)
        df = df[df["unified_label"].notna()].copy()
        frames.append(df)
        print(f"    {os.path.basename(filepath)}: {len(frames[-1]):,} rows")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop(columns=[c for c in DROP_COLUMNS | {"Label"} if c in combined.columns])
    for col in combined.columns:
        if col != LABEL_COL:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined = combined.dropna(axis=1, how="all")
    num_cols = combined.select_dtypes(include=[np.number]).columns
    combined[num_cols] = combined[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return combined.reset_index(drop=True)


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
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   zero_division=0, output_dict=True)
    print(f"\n--- {title} ---")
    print(f"Balanced accuracy: {bal:.4f}   F1 macro: {f1_mac:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    return bal, f1_mac, report


if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading IDS2017 (train base)...")
    df17 = load_folder(PATH_TRAIN_DIR)
    print(f"  {len(df17):,} rows")

    print("\nLoading IDS2018 (few-shot source + test)...")
    df18 = load_folder(PATH_TEST_DIR, rename_cols=RENAME_2018)
    print(f"  {len(df18):,} rows")

    meta  = {LABEL_COL}
    num17 = set(df17.select_dtypes(include=[np.number]).columns) - meta
    num18 = set(df18.select_dtypes(include=[np.number]).columns) - meta
    feats = sorted(num17 & num18)
    print(f"\nCommon features: {len(feats)}")

    le = LabelEncoder()
    le.fit(sorted(set(df17[LABEL_COL].unique()) | set(df18[LABEL_COL].unique())))
    print(f"Classes: {list(le.classes_)}")
    idx_botnet = list(le.classes_).index("Botnet")

    print(f"\n{'='*70}\nEXPERIMENT 3 - {int(TRANSFER_RATIO*100)}% few-shot"
          f" (Botnet + DoS_DDoS from IDS2018 -> train)\n{'='*70}")

    transfer_train, transfer_test = [], []
    for cls in ["Botnet", "DoS_DDoS"]:
        df_cls = df18[df18[LABEL_COL] == cls]
        tr, te = train_test_split(df_cls, test_size=1 - TRANSFER_RATIO, random_state=RANDOM_STATE)
        transfer_train.append(tr)
        transfer_test.append(te)
        print(f"  {cls}: {len(tr):,} -> train | {len(te):,} -> test")

    df18_rest = df18[~df18[LABEL_COL].isin(["Botnet", "DoS_DDoS"])]

    df_train_fs = (pd.concat([df17] + transfer_train, ignore_index=True)
                     .sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True))
    df_test_fs  = (pd.concat([df18_rest] + transfer_test, ignore_index=True)
                     .sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True))

    print("\nTrain (IDS2017 + few-shot):")
    print(df_train_fs[LABEL_COL].value_counts().to_string())
    print("\nTest (remaining IDS2018):")
    print(df_test_fs[LABEL_COL].value_counts().to_string())

    X_fs = df_train_fs[feats].values.astype(float)
    y_fs = le.transform(df_train_fs[LABEL_COL])
    X_fs_bal, y_fs_bal = balance(X_fs, y_fs)
    print(f"\nBalanced train: {len(X_fs_bal):,}")

    print("Training RF (few-shot)...")
    rf = RandomForestClassifier(**RF_CONFIG)
    rf.fit(X_fs_bal, y_fs_bal)

    X_te = df_test_fs[feats].values.astype(float)
    y_te = le.transform(df_test_fs[LABEL_COL])
    y_pred = rf.predict(X_te)

    bal3, f1_3, rep3 = evaluate(y_te, y_pred, le,
                                f"Exp.3 - {int(TRANSFER_RATIO*100)}% few-shot (test = remaining IDS2018)")

    # Generalization check: F1(Botnet) on held-out 80% vs full 100% Botnet set
    mask_full = df18[LABEL_COL] == "Botnet"
    y_pred_full = rf.predict(df18.loc[mask_full, feats].values.astype(float))
    y_true_full = le.transform(df18.loc[mask_full, LABEL_COL])
    f1_full = f1_score(y_true_full == idx_botnet, y_pred_full == idx_botnet, zero_division=0)

    mask_held = df_test_fs[LABEL_COL] == "Botnet"
    f1_held = f1_score(y_te[mask_held.values] == idx_botnet,
                       y_pred[mask_held.values] == idx_botnet, zero_division=0)
    print(f"\nGeneralization check - F1 Botnet:")
    print(f"  held-out 80% (unseen during training): {f1_held:.4f}")
    print(f"  full 100% of Botnet in IDS2018:        {f1_full:.4f}  (should be close)")

    # Confusion matrix data (Figure 4.6a)
    labels_cm = np.unique(np.concatenate((y_te, y_pred)))
    cm = confusion_matrix(y_te, y_pred, labels=labels_cm)
    confusion_matrix_result = {
        "labels": list(le.inverse_transform(labels_cm)), "matrix": cm.tolist(),
        "title": f"IDS2017+few-shot -> IDS2018, test {100-int(TRANSFER_RATIO*100)}%  |  Bal.Acc: {bal3:.3f}",
    }

    # Feature importances (Figure 4.6b)
    importances = pd.Series(rf.feature_importances_, index=feats).nlargest(15)

    print("\nTop 15 features:")
    for i, (feat, val) in enumerate(importances.items(), 1):
        print(f"  {i:>2}. {feat:<42} {val:.4f}")

    result = {
        "experiment": f"Experiment 3 (CICFlowMeter, IDS2017+{int(TRANSFER_RATIO*100)}% few-shot -> IDS2018)",
        "description": f"Adds {int(TRANSFER_RATIO*100)}% of Botnet and DoS_DDoS samples from IDS2018 "
                        "to the training set (alongside all of IDS2017), to show how much the model "
                        "improves with a few target-domain examples (few-shot)",
        "balanced_accuracy": bal3, "f1_macro": f1_3, "classification_report": rep3,
        "generalization_check_f1_botnet": {"held_out_80pct": f1_held, "full_100pct": f1_full},
        "top_features": {feat: float(val) for feat, val in importances.items()},
        "n_train": len(X_fs_bal), "n_test": len(X_te), "classes": list(le.classes_),
        "confusion_matrix": confusion_matrix_result,
        "class_balance": None,
        "samples_per_split": None,
    }
    results_file = os.path.join(RESULTS_DIR, "results_experiment3.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"-> Results saved to: {results_file}")
