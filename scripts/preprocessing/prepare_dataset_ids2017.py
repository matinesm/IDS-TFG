"""
Prepares the CIC-IDS2017 dataset for the NFStream pipeline.

Output: <TFG_DATA_DIR>/processed_data/dataset_ml_ids2017.csv

"""

import gc
import glob
import os

import numpy as np
import pandas as pd
from nfstream import NFStreamer

# ── Configuration ─────────────────────────────────────────────────────────────
TFG_DATA_DIR = os.environ["TFG_DATA_DIR"]

PCAP_DIR    = os.path.join(TFG_DATA_DIR, "raw_data/CIC-IDS2017/PCAP")
CSV_DIR     = os.path.join(TFG_DATA_DIR, "raw_data/CIC-IDS2017/CSV")
TEMP_FILE   = os.path.join(TFG_DATA_DIR, "processed_data/_tmp_ids2017_labeled.csv")
OUTPUT_FILE = os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_ids2017.csv")

LABEL_COL = "attack_cat"

# 120 s to absorb CICFlowMeter buffering; PCAP timestamps are 3 h ahead of CSV
TIME_TOLERANCE    = 120.0
TIME_OFFSET_HOURS = 3

JOIN_COLS = ["ip_min", "ip_max", "port_min", "port_max", "protocol"]
WEEKDAYS  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_protocol(value):
    if pd.isna(value):
        return 255
    try:
        return int(value)
    except ValueError:
        return 255


def group_labels(label):
    """Groups attack subcategories into families (Web Attack, DoS, Brute Force)."""
    if "Web Attack" in label:
        return "Web Attack"
    if "DoS" in label and label != "DDoS":
        return "DoS"
    if "Patator" in label:
        return "Brute Force"
    return label


def bidirectional_keys(df, col_ip_a="src_ip", col_ip_b="dst_ip",
                       col_port_a="src_port", col_port_b="dst_port"):
    """Builds ordered (min/max) IP/port keys so the join is direction-independent."""
    cond = df[col_ip_a] < df[col_ip_b]
    df["ip_min"]   = np.where(cond, df[col_ip_a], df[col_ip_b])
    df["ip_max"]   = np.where(cond, df[col_ip_b], df[col_ip_a])
    df["port_min"] = np.where(cond, df[col_port_a], df[col_port_b])
    df["port_max"] = np.where(cond, df[col_port_b], df[col_port_a])
    return df


# ── Phase 1: master label database ───────────────────────────────────────────
def build_master_labels():
    print(f"1. Building master label database from {CSV_DIR}...")
    required_cols = ["Source IP", "Source Port", "Destination IP",
                     "Destination Port", "Protocol", "Timestamp", "Label"]

    frames = []
    for path in sorted(glob.glob(os.path.join(CSV_DIR, "*.csv"))):
        print(f"   Loading: {os.path.basename(path)}")
        df = pd.read_csv(path, low_memory=False, encoding="cp1252")
        df.columns = df.columns.str.strip()
        if all(c in df.columns for c in required_cols):
            frames.append(df[required_cols])

    df_labels = pd.concat(frames, ignore_index=True)
    df_labels = df_labels.rename(columns={
        "Source IP": "src_ip", "Destination IP": "dst_ip",
        "Source Port": "src_port", "Destination Port": "dst_port",
        "Protocol": "protocol", "Label": LABEL_COL,
    })

    df_labels["src_port"] = pd.to_numeric(df_labels["src_port"], errors="coerce").fillna(0).astype(int)
    df_labels["dst_port"] = pd.to_numeric(df_labels["dst_port"], errors="coerce").fillna(0).astype(int)
    df_labels["protocol"] = df_labels["protocol"].apply(parse_protocol)
    df_labels["src_ip"] = df_labels["src_ip"].astype(str).str.strip()
    df_labels["dst_ip"] = df_labels["dst_ip"].astype(str).str.strip()

    print("   Parsing timestamps and correcting time zones...")
    df_labels["Timestamp"] = df_labels["Timestamp"].astype(str).str.strip()
    df_labels["Timestamp"] = pd.to_datetime(df_labels["Timestamp"], format="mixed",
                                             dayfirst=True, errors="coerce")

    # Some afternoon timestamps were stored as early morning, add 12 h to fix
    is_afternoon = df_labels["Timestamp"].dt.hour < 8
    df_labels.loc[is_afternoon, "Timestamp"] += pd.Timedelta(hours=12)

    # Day name prevents false matches across captures from different days
    df_labels["day_name"] = df_labels["Timestamp"].dt.day_name()
    df_labels = df_labels.dropna(subset=["Timestamp"])

    df_labels["Stime"] = (df_labels["Timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    df_labels["Stime"] += TIME_OFFSET_HOURS * 3600

    df_labels = bidirectional_keys(df_labels)
    df_labels[LABEL_COL] = df_labels[LABEL_COL].replace(["nan", "NaN", ""], "BENIGN").fillna("BENIGN")

    df_reduced = df_labels[JOIN_COLS + ["Stime", "day_name", LABEL_COL]].copy()
    del df_labels
    gc.collect()

    print(f"-> Master labels ready: {len(df_reduced):,} labeled flows\n")
    return df_reduced


# ── Phase 2: flow extraction + join with labels, PCAP by PCAP ────────────────
def process_pcaps(df_labels):
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    pcaps = sorted(glob.glob(os.path.join(PCAP_DIR, "*.pcap")))
    print(f"2. Processing {len(pcaps)} PCAPs with NFStream...\n")

    total_extracted = 0
    total_matched   = 0

    for i, pcap in enumerate(pcaps, 1):
        name = os.path.basename(pcap)
        print(f"--- [{i}/{len(pcaps)}] {name} ---")

        current_day = next((d for d in WEEKDAYS if d.lower() in name.lower()), "Unknown")
        df_day = df_labels[df_labels["day_name"] == current_day]
        print(f"   Labels for {current_day}: {len(df_day):,}")
        if df_day.empty:
            print("   [!] No labels for this day, skipping.")
            continue

        streamer = NFStreamer(source=pcap, statistical_analysis=True, splt_analysis=True)
        temp_csv = "_tmp_nfstream_ids2017.csv"
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

        df_final = pd.merge(df_nf, df_day, on=JOIN_COLS, how="inner")
        df_final = df_final[abs(df_final["flow_start_sec"] - df_final["Stime"]) <= TIME_TOLERANCE]

        df_final["is_attack"] = np.where(df_final[LABEL_COL] == "BENIGN", 0, 1)
        df_final = df_final.sort_values(by="is_attack", ascending=False)
        if "id" in df_final.columns:
            df_final = df_final.drop_duplicates(subset=["id"])

        df_final = df_final.drop(columns=["ip_min", "ip_max", "port_min", "port_max",
                                          "Stime", "day_name", "flow_start_sec", "is_attack"],
                                 errors="ignore")

        total_matched += len(df_final)
        print(f"   Flows extracted: {len(df_nf):,}  |  Valid matches (tol={TIME_TOLERANCE}s): {len(df_final):,}")

        first = (i == 1)
        df_final.to_csv(TEMP_FILE, mode="a", header=first, index=False)

        del streamer, df_nf, df_final, df_day
        gc.collect()

    print(f"\n-> Total flows extracted: {total_extracted:,}")
    print(f"-> Total valid matches:   {total_matched:,}\n")


# ── Phase 3: group labels and save final dataset ──────────────────────────────
def group_and_save():
    print("3. Grouping attack subcategories into families...")
    df = pd.read_csv(TEMP_FILE, low_memory=False)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().apply(group_labels)

    print("\nFinal class distribution:")
    summary = pd.DataFrame({
        "Flows": df[LABEL_COL].value_counts(),
        "Percentage (%)": (df[LABEL_COL].value_counts(normalize=True) * 100).round(4),
    })
    print(summary)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    os.remove(TEMP_FILE)
    print(f"\n-> Final dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    labels = build_master_labels()
    process_pcaps(labels)
    group_and_save()
