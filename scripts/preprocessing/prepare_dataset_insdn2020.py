"""
Prepares the InSDN2020 dataset for the NFStream pipeline.

Output: <TFG_DATA_DIR>/processed_data/dataset_ml_insdn2020.csv

"""

import gc
import glob
import os

import numpy as np
import pandas as pd
from nfstream import NFStreamer

# ── Configuration ─────────────────────────────────────────────────────────────
TFG_DATA_DIR = os.environ["TFG_DATA_DIR"]

CSV_DIR     = os.path.join(TFG_DATA_DIR, "raw_data/InSDN2020/CSV")
PCAP_DIR    = os.path.join(TFG_DATA_DIR, "raw_data/InSDN2020/PCAP")
TEMP_FILE   = os.path.join(TFG_DATA_DIR, "processed_data/_tmp_insdn2020_labeled.csv")
OUTPUT_FILE = os.path.join(TFG_DATA_DIR, "processed_data/dataset_ml_insdn2020.csv")

LABEL_COL = "attack_cat"

TIME_TOLERANCE    = 120.0
TIME_OFFSET_HOURS = 0

# Each official label CSV corresponds to one capture sub-directory
GROUPS = {
    "metasploitable-2": "Metsplotable-2_Group",
    "Normal_data":      "Normal_Group",
    "OVS":              "OVS_Group",
}

JOIN_COLS = ["ip_min", "ip_max", "port_min", "port_max", "protocol"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_protocol(value):
    if pd.isna(value):
        return 255
    try:
        return int(value)
    except ValueError:
        return 255


def standardize_label(label):
    """Normalizes InSDN2020 labels to names consistent with the other datasets:
    Normal->BENIGN, web variants->Web Attack, BFA->Brute Force,
    U2R (micro-class) grouped as Exploits."""
    lower = label.lower()
    if lower == "normal":
        return "BENIGN"
    if "web" in lower:
        return "Web Attack"
    if label == "BFA":
        return "Brute Force"
    if label == "U2R":
        return "Exploits"
    return label


def bidirectional_keys(df, col_ip_a="src_ip", col_ip_b="dst_ip",
                       col_port_a="src_port", col_port_b="dst_port"):
    cond = df[col_ip_a] < df[col_ip_b]
    df["ip_min"]   = np.where(cond, df[col_ip_a], df[col_ip_b])
    df["ip_max"]   = np.where(cond, df[col_ip_b], df[col_ip_a])
    df["port_min"] = np.where(cond, df[col_port_a], df[col_port_b])
    df["port_max"] = np.where(cond, df[col_port_b], df[col_port_a])
    return df


def load_group_labels(csv_name):
    path = os.path.join(CSV_DIR, f"{csv_name}.csv")
    df = pd.read_csv(path, low_memory=False, encoding="cp1252")
    df.columns = df.columns.str.strip()

    mapping = {}
    for col in df.columns:
        c = col.lower()
        if "src ip" in c or "source ip" in c:
            mapping[col] = "src_ip"
        elif "dst ip" in c or "destination ip" in c:
            mapping[col] = "dst_ip"
        elif "src port" in c or "source port" in c:
            mapping[col] = "src_port"
        elif "dst port" in c or "destination port" in c:
            mapping[col] = "dst_port"
        elif "protocol" in c:
            mapping[col] = "protocol"
        elif "timestamp" in c:
            mapping[col] = "Timestamp"
        elif "label" in c or "class" in c:
            mapping[col] = LABEL_COL
    df = df.rename(columns=mapping)[["src_ip", "src_port", "dst_ip", "dst_port",
                                     "protocol", "Timestamp", LABEL_COL]]

    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype(int)
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int)
    df["protocol"] = df["protocol"].apply(parse_protocol)
    df["src_ip"] = df["src_ip"].astype(str).str.strip()
    df["dst_ip"] = df["dst_ip"].astype(str).str.strip()

    df["Timestamp"] = df["Timestamp"].astype(str).str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Stime"] = (df["Timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    df["Stime"] += TIME_OFFSET_HOURS * 3600

    df = bidirectional_keys(df)
    df[LABEL_COL] = df[LABEL_COL].replace(["nan", "NaN", ""], "Normal").fillna("Normal")
    return df[JOIN_COLS + ["Stime", LABEL_COL]].copy()


# ── Join a single PCAP against the labels for its group ──────────────────────
def cross_pcap(pcap, df_labels):
    streamer = NFStreamer(source=pcap, statistical_analysis=True, splt_analysis=True)
    temp_csv = "_tmp_nfstream_insdn.csv"
    streamer.to_csv(temp_csv)
    df_nf = pd.read_csv(temp_csv, low_memory=False, on_bad_lines="skip")
    os.remove(temp_csv)

    extracted = len(df_nf)

    df_nf["src_port"] = pd.to_numeric(df_nf["src_port"], errors="coerce")
    df_nf["dst_port"] = pd.to_numeric(df_nf["dst_port"], errors="coerce")
    df_nf = df_nf.dropna(subset=["src_port", "dst_port"])
    for col in ["src_port", "dst_port", "protocol"]:
        df_nf[col] = pd.to_numeric(df_nf[col], errors="coerce").fillna(0).astype(int)
    for col in ["src_ip", "dst_ip"]:
        df_nf[col] = df_nf[col].astype(str).str.strip()

    df_nf["flow_start_sec"] = pd.to_numeric(df_nf["bidirectional_first_seen_ms"], errors="coerce") / 1000.0
    df_nf = bidirectional_keys(df_nf)

    # Join 1 (strict): valid for normal traffic and scans
    df_1 = pd.merge(df_nf, df_labels, on=JOIN_COLS, how="inner")
    df_1 = df_1[abs(df_1["flow_start_sec"] - df_1["Stime"]) <= TIME_TOLERANCE]

    # Join 2 (DDoS rescue): the official InSDN2020 CSV stores DDoS flows with
    # protocol and ports set to zero, so we replicate that quirk to match them
    if "id" in df_1.columns:
        matched = df_1["id"].unique()
        df_unmatched = df_nf[~df_nf["id"].isin(matched)].copy()
    else:
        df_unmatched = df_nf.copy()
    df_unmatched[["port_min", "port_max", "protocol"]] = 0

    df_2 = pd.merge(df_unmatched, df_labels, on=JOIN_COLS, how="inner")
    df_2 = df_2[abs(df_2["flow_start_sec"] - df_2["Stime"]) <= TIME_TOLERANCE]

    df_final = pd.concat([df_1, df_2], ignore_index=True)
    df_final["is_attack"] = np.where(df_final[LABEL_COL] == "Normal", 0, 1)
    df_final = df_final.sort_values(by="is_attack", ascending=False)
    if "id" in df_final.columns:
        df_final = df_final.drop_duplicates(subset=["id"])

    df_final = df_final.drop(columns=["ip_min", "ip_max", "port_min", "port_max",
                                      "Stime", "flow_start_sec", "is_attack"],
                             errors="ignore")

    del streamer, df_nf, df_1, df_2, df_unmatched
    gc.collect()
    return df_final, extracted


# ── Phases 1+2: for each group, load labels and process its PCAPs ─────────────
def process_groups():
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    total_extracted = 0
    total_matched   = 0
    first_block     = True

    for csv_name, folder_name in GROUPS.items():
        print(f"\n{'='*60}\nGroup: CSV '{csv_name}.csv'  <->  folder '{folder_name}'\n{'='*60}")

        csv_path = os.path.join(CSV_DIR, f"{csv_name}.csv")
        pcaps = sorted(glob.glob(os.path.join(PCAP_DIR, folder_name, "**", "*.pcap"),
                                 recursive=True))
        if not os.path.exists(csv_path) or not pcaps:
            print(f"   [!] Missing data for this group (csv={os.path.exists(csv_path)}, "
                  f"pcaps={len(pcaps)}); skipping.")
            continue

        print(f"1. Loading labels from {csv_path}...")
        df_labels = load_group_labels(csv_name)
        print(f"   {len(df_labels):,} labeled flows")

        print(f"2. Processing {len(pcaps)} PCAPs...")
        for i, pcap in enumerate(pcaps, 1):
            rel = os.path.relpath(pcap, PCAP_DIR)
            df_final, extracted = cross_pcap(pcap, df_labels)

            total_extracted += extracted
            total_matched   += len(df_final)
            print(f"   [{i}/{len(pcaps)}] {rel}: extracted={extracted:,}  valid matches={len(df_final):,}")

            df_final.to_csv(TEMP_FILE, mode="a", header=first_block, index=False)
            first_block = False
            del df_final
            gc.collect()

        del df_labels
        gc.collect()

    print(f"\n-> Total flows extracted: {total_extracted:,}")
    print(f"-> Total valid matches:   {total_matched:,}\n")


# ── Phase 3: standardize labels and save final dataset ───────────────────────
def standardize_and_save():
    print("3. Standardizing labels...")
    df = pd.read_csv(TEMP_FILE, low_memory=False)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().apply(standardize_label)

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
    process_groups()
    standardize_and_save()
