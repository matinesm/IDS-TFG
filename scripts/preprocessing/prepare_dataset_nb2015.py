"""
Prepares the UNSW-NB2015 dataset for the NFStream pipeline.

Output: <TFG_DATA_DIR>/processed_data/dataset_ml_nb2015.csv

"""

import gc
import glob
import os
import socket

import numpy as np
import pandas as pd
from nfstream import NFStreamer

# ── Configuration ─────────────────────────────────────────────────────────────
TFG_DATA_DIR = os.environ["TFG_DATA_DIR"]

CSV_DIR     = os.path.join(TFG_DATA_DIR, "raw_data/UNSW-NB2015/CSV")
PCAP_DIR    = os.path.join(TFG_DATA_DIR, "raw_data/UNSW-NB2015/PCAP")
TEMP_FILE   = os.path.join(TFG_DATA_DIR, "processed_data/_tmp_nb2015_labeled.csv")
OUTPUT_FILE = os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_nb2015.csv")

LABEL_COL = "attack_cat"

# NB2015 timestamps are reliable (no external capture buffering like CICFlowMeter),
# so a much narrower tolerance is sufficient
TIME_TOLERANCE = 3.0

JOIN_COLS = ["ip_min", "ip_max", "port_min", "port_max", "protocol"]

RELABEL = {"Backdoor": "Backdoors", "Normal": "BENIGN"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_port(value):
    if pd.isna(value):
        return 0
    s = str(value).strip().lower()
    if s in ("-", ""):
        return 0
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except ValueError:
            return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_protocol(name):
    if pd.isna(name):
        return 255
    try:
        return socket.getprotobyname(str(name).strip().lower())
    except OSError:
        return 255


def bidirectional_keys(df, col_ip_a="src_ip", col_ip_b="dst_ip",
                       col_port_a="src_port", col_port_b="dst_port"):
    cond = df[col_ip_a] < df[col_ip_b]
    df["ip_min"]   = np.where(cond, df[col_ip_a], df[col_ip_b])
    df["ip_max"]   = np.where(cond, df[col_ip_b], df[col_ip_a])
    df["port_min"] = np.where(cond, df[col_port_a], df[col_port_b])
    df["port_max"] = np.where(cond, df[col_port_b], df[col_port_a])
    return df


# ── Phase 1: master label database ───────────────────────────────────────────
def build_master_labels():
    print("1. Building master label database...")
    df_features = pd.read_csv(os.path.join(CSV_DIR, "NUSW-NB15_features.csv"), encoding="latin1")
    col_names = df_features["Name"].astype(str).str.strip().tolist()

    frames = []
    for i in range(1, 5):
        name = f"UNSW-NB15_{i}.csv"
        path = os.path.join(CSV_DIR, name)
        print(f"   Loading: {name}")
        df = pd.read_csv(path, names=col_names, header=None, low_memory=False)
        frames.append(df[["srcip", "sport", "dstip", "dsport", "proto", "Stime", "Ltime", LABEL_COL]])

    df_labels = pd.concat(frames, ignore_index=True)
    df_labels["sport"]  = df_labels["sport"].apply(parse_port)
    df_labels["dsport"] = df_labels["dsport"].apply(parse_port)
    df_labels["proto"]  = df_labels["proto"].apply(parse_protocol)

    df_labels = df_labels.rename(columns={
        "srcip": "src_ip", "dstip": "dst_ip",
        "sport": "src_port", "dsport": "dst_port", "proto": "protocol",
    })
    df_labels["protocol"] = pd.to_numeric(df_labels["protocol"], errors="coerce").fillna(255).astype(int)
    df_labels["src_ip"] = df_labels["src_ip"].astype(str).str.strip()
    df_labels["dst_ip"] = df_labels["dst_ip"].astype(str).str.strip()
    df_labels["Stime"]  = pd.to_numeric(df_labels["Stime"], errors="coerce")

    df_labels = bidirectional_keys(df_labels)
    df_labels[LABEL_COL] = df_labels[LABEL_COL].replace(["nan", "NaN", ""], "Normal").fillna("Normal")

    df_reduced = df_labels[JOIN_COLS + ["Stime", "Ltime", LABEL_COL]].copy()
    del df_labels
    gc.collect()

    print(f"-> Master labels ready: {len(df_reduced):,} labeled flows\n")
    return df_reduced


# ── Phase 2: flow extraction + join with labels, PCAP by PCAP ────────────────
def process_pcaps(df_labels):
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    pcaps = (sorted(glob.glob(os.path.join(PCAP_DIR, "1", "*.pcap"))) +
             sorted(glob.glob(os.path.join(PCAP_DIR, "2", "*.pcap"))))
    print(f"2. Processing {len(pcaps)} PCAPs with NFStream...\n")

    total_extracted = 0
    total_matched   = 0

    for i, pcap in enumerate(pcaps, 1):
        name = os.path.basename(pcap)
        print(f"--- [{i}/{len(pcaps)}] {name} ---")

        streamer = NFStreamer(source=pcap, statistical_analysis=True, splt_analysis=True)
        temp_csv = "_tmp_nfstream_nb2015.csv"
        streamer.to_csv(temp_csv)
        df_nf = pd.read_csv(temp_csv, low_memory=False, on_bad_lines="skip")
        os.remove(temp_csv)

        total_extracted += len(df_nf)

        df_nf["src_port"] = pd.to_numeric(df_nf["src_port"], errors="coerce")
        df_nf["dst_port"] = pd.to_numeric(df_nf["dst_port"], errors="coerce")
        df_nf = df_nf.dropna(subset=["src_port", "dst_port"])
        for col in ["src_port", "dst_port", "protocol"]:
            df_nf[col] = pd.to_numeric(df_nf[col], errors="coerce").fillna(0).astype(int)
        for col in ["src_ip", "dst_ip"]:
            df_nf[col] = df_nf[col].astype(str).str.strip()

        df_nf["flow_start_sec"] = pd.to_numeric(df_nf["bidirectional_first_seen_ms"], errors="coerce") / 1000.0
        df_nf = bidirectional_keys(df_nf)

        df_final = pd.merge(df_nf, df_labels, on=JOIN_COLS, how="inner")
        df_final = df_final[abs(df_final["flow_start_sec"] - df_final["Stime"]) <= TIME_TOLERANCE]

        df_final = df_final.sort_values(by=LABEL_COL, ascending=False)
        if "id" in df_final.columns:
            df_final = df_final.drop_duplicates(subset=["id"])

        df_final = df_final.drop(columns=["ip_min", "ip_max", "port_min", "port_max",
                                          "Stime", "Ltime", "flow_start_sec"],
                                 errors="ignore")

        total_matched += len(df_final)
        print(f"   Flows extracted: {len(df_nf):,}  |  Valid matches (tol={TIME_TOLERANCE}s): {len(df_final):,}")

        first = (i == 1)
        df_final.to_csv(TEMP_FILE, mode="a", header=first, index=False)

        del streamer, df_nf, df_final
        gc.collect()

    print(f"\n-> Total flows extracted: {total_extracted:,}")
    print(f"-> Total valid matches:   {total_matched:,}\n")


# ── Phase 3: minor relabeling and save final dataset ─────────────────────────
def relabel_and_save():
    print("3. Applying relabeling (Backdoor->Backdoors, Normal->BENIGN)...")
    df = pd.read_csv(TEMP_FILE, low_memory=False)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().replace(RELABEL)

    print("\nFinal class distribution:")
    print(df[LABEL_COL].value_counts())

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    os.remove(TEMP_FILE)
    print(f"\n-> Final dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    labels = build_master_labels()
    process_pcaps(labels)
    relabel_and_save()
