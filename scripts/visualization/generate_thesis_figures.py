"""
Final thesis figures: aggregates ALL experiment results into
a single comparison table and generates the figures that compare 
results across experiments. Also generates the class distribution 
figure from the already-prepared datasets, and the confusion matrices, 
class balance and feature importance from data saved  inside each 
experiment's results.json. The experiment scripts themselves produce 
no figures, only results.json files.

Requires having run beforehand (in any order):
  - scripts/preprocessing/prepare_dataset_{ids2017,insdn2020,nb2015}.py
  - scripts/experiments/validation_intra_ids2017.py
  - scripts/experiments/experiment0_nb2015_cross_domain.py
  - scripts/experiments/experiment1_nfstream_ids2017_to_insdn2020.py
  - scripts/experiments/experiment2_cicflowmeter_baseline.py
  - scripts/experiments/experiment3_cicflowmeter_fewshot.py

Outputs:
  output/figures/fig_3_2_distribucion_clases_dataset.png
  output/figures/fig_3_3_balanceo_clases.png             (Figure 3.3)
  output/figures/fig_4_1_matriz_confusion_exp1.png       (Figure 4.1)
  output/figures/fig_4_2_matriz_confusion_exp2.png       (Figure 4.2)
  output/figures/fig_4_3_matriz_confusion_exp3.png       (Figure 4.3)
  output/figures/fig_4_4_importancia_caracteristicas_exp3.png (Figure 4.4)
  output/figures/fig_4_5_evolucion_experimentos.png
  output/figures/fig_4_6_f1_clase_fewshot.png
  output/results/comparison_table.json
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
TFG_DATA_DIR  = os.environ["TFG_DATA_DIR"]
PROCESSED_DIR = os.path.join(TFG_DATA_DIR, "processed_data")
ORIGINALS_DIR = os.path.join(TFG_DATA_DIR, "raw_data")

OUTPUT_BASE  = os.path.join(os.path.dirname(__file__), "..", "..", "output")
RESULTS_BASE = os.path.join(OUTPUT_BASE, "results")
FIGURES_DIR  = os.path.join(OUTPUT_BASE, "figures")

TAXONOMY_NFSTREAM = {
    "BENIGN": "Normal", "Normal": "Normal",
    "DoS": "DoS_DDoS", "DDoS": "DoS_DDoS",
    "DoS Hulk": "DoS_DDoS", "DoS GoldenEye": "DoS_DDoS",
    "DoS Slowhttptest": "DoS_DDoS", "DoS slowloris": "DoS_DDoS",
    "Reconnaissance": "Scan", "PortScan": "Scan", "Probe": "Scan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Web Attack – Brute Force": "BruteForce",
    "BFA": "BruteForce", "Brute Force": "BruteForce",
    "Web Attack – XSS": "WebAttack", "Web Attack – Sql Injection": "WebAttack",
    "Web-Attack": "WebAttack", "Web Attack": "WebAttack",
    "Bot": "Botnet", "BOTNET": "Botnet",
    "Exploits": None, "Fuzzers": None, "Generic": None, "Shellcode": None,
    "Analysis": None, "Backdoor": None, "Backdoors": None, "Worms": None,
    "Infiltration": None, "Heartbleed": None, "U2R": None,
}

TAXONOMY_CICFLOWMETER = {
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


COMPARISON_ROWS = [
    ("Validación intra-IDS2017 (NFstream)", "Validación intra-IDS2017",
     "results_validation.json", []),
    ("Experimento 0: UNSW-NB2015 → IDS2017 (NFstream)", "Exp. 0: NB2015 → IDS2017",
     "results_experiment0.json", ["__nb2015__", "UNSW-NB2015 → IDS2017"]),
    ("Experimento 0: UNSW-NB2015 → InSDN2020 (NFstream)", "Exp. 0: NB2015 → InSDN2020",
     "results_experiment0.json", ["__nb2015__", "UNSW-NB2015 → InSDN2020"]),
    ("Experimento 0: UNSW-NB2015 + IDS2017 → InSDN2020 (NFstream)", "Exp. 0: NB2015+IDS2017 → InSDN2020",
     "results_experiment0.json", ["__nb2015__", "UNSW-NB2015 + IDS2017 → InSDN2020"]),
    ("Experimento 1: IDS2017 → InSDN2020, calibrado (NFstream)", "Exp. 1: calibrado",
     "results_experiment1.json", ["calibrated", "calibrated"]),
    ("Experimento 1: IDS2017 → InSDN2020, + window features (NFstream)", "Exp. 1: + window",
     "results_experiment1.json", ["calibrated_with_window", "calibrated"]),
    ("Experimento 2: IDS2017 → IDS2018, sin few-shot (CICFlowMeter)", "Exp. 2: sin few-shot",
     "results_experiment2.json", ["threshold_05"]),
    ("Experimento 3: IDS2017 → IDS2018, few-shot 20% (CICFlowMeter)", "Exp. 3: few-shot 20 %",
     "results_experiment3.json", []),
]


# ── Loading utilities ─────────────────────────────────────────────────────────
def load_json(relative_path):
    path = os.path.join(RESULTS_BASE, relative_path)
    if not os.path.exists(path):
        print(f"  [!] Missing {path} - run the script that generates it first. Row skipped.")
        return None
    with open(path) as f:
        return json.load(f)


def navigate(data, keys):
    """Traverses a nested dict following `keys`, with a special case for
    experiment0: its JSON stores the three cross-domain runs as a list under
    'experiments' rather than a dict by name."""
    if keys and keys[0] == "__nb2015__":
        target = keys[1]
        for exp in data.get("experiments", []):
            if exp.get("name") == target:
                return exp
        return None
    for key in keys:
        if data is None:
            return None
        data = data.get(key)
    return data


def build_comparison_table():
    print("Aggregating results into TAB:COMPARATIVA...")
    cache, rows = {}, []
    for label, short_label, rel_path, keys in COMPARISON_ROWS:
        if rel_path not in cache:
            cache[rel_path] = load_json(rel_path)
        node = navigate(cache[rel_path], keys)
        if node is None:
            continue
        rows.append({
            "label": label,
            "short_label": short_label,
            "balanced_accuracy": node.get("balanced_accuracy"),
            "f1_macro": node.get("f1_macro"),
        })
    return rows


def print_table(rows):
    print("\n" + "=" * 72)
    print("TAB:COMPARATIVA - summary of all experiments")
    print("=" * 72)
    for row in rows:
        f1 = f"{row['f1_macro']:.4f}" if row["f1_macro"] is not None else "N/A"
        print(f"  {row['label']:<52} bal={row['balanced_accuracy']:.4f}  F1={f1}")
    print("\n--- LaTeX rows (paste into TAB:COMPARATIVA) ---")
    for row in rows:
        f1 = f"{row['f1_macro']:.4f}" if row["f1_macro"] is not None else "---"
        print(f"  {row['label']} & {row['balanced_accuracy']:.4f} & {f1} \\\\")


# ── Figure 3.2: class distribution by dataset ─────────────────────────────────
def count_nfstream(csv_path):
    if not os.path.exists(csv_path):
        print(f"  [!] Missing {csv_path}"); return None
    raw = pd.read_csv(csv_path, usecols=["attack_cat"], low_memory=False)["attack_cat"]
    return raw.astype(str).str.strip().map(TAXONOMY_NFSTREAM).dropna().value_counts()


def count_cicflowmeter(folder):
    if not os.path.isdir(folder):
        print(f"  [!] Missing {folder}"); return None
    total = pd.Series(dtype="int64")
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(folder, name)
        for enc in ["utf-8", "latin-1"]:
            try:
                raw = pd.read_csv(path, usecols=["Label"], dtype=str, encoding=enc, on_bad_lines="skip")["Label"]
                break
            except (UnicodeDecodeError, ValueError):
                raw = None
        if raw is None:
            continue
        total = total.add(raw.astype(str).str.strip().map(TAXONOMY_CICFLOWMETER).dropna().value_counts(),
                          fill_value=0)
    return total


def plot_class_distribution_by_dataset():
    print("\nGenerating Figure 3.2 (class distribution by dataset)...")
    datasets = [
        ("CIC-IDS2017", lambda: count_nfstream(os.path.join(PROCESSED_DIR, "dataset_ml_ids2017.csv"))),
        ("CIC-IDS2018", lambda: count_cicflowmeter(os.path.join(ORIGINALS_DIR, "CIC-IDS2018/CSV"))),
        ("InSDN2020",   lambda: count_nfstream(os.path.join(PROCESSED_DIR, "dataset_ml_insdn2020.csv"))),
        ("UNSW-NB2015", lambda: count_nfstream(os.path.join(PROCESSED_DIR, "dataset_ml_nb2015.csv"))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (name, load) in zip(axes.flat, datasets):
        counts = load()
        if counts is None:
            ax.set_visible(False); continue
        classes = sorted(counts.index)
        ax.bar(classes, counts.reindex(classes).values, color="steelblue")
        ax.set_title(name, fontsize=15, fontweight="bold")
        ax.set_ylabel("Número de flujos", fontsize=14, fontweight="bold")
        ax.set_yscale("log"); ax.tick_params(axis="x", rotation=35, labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_3_2_distribucion_clases_dataset.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"-> Figure saved: {path}")


# ── Figures 3.3 and 4.1/4.2/4.3: class balance and confusion matrices ─────────
def plot_class_balance(experiment, data):
    print(f"\nGenerating Figure 3.3 (class balance, {experiment})...")
    balance = data.get("class_balance")
    if balance is None:
        print("  [!] No 'class_balance' in results.json - rerun the experiment to save it. Skipping.")
        return
    classes = balance["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, title in [(axes[0], "before", "Antes del balanceo"),
                           (axes[1], "after",  "Después del balanceo")]:
        ax.bar(classes, balance[key], color="steelblue")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylabel("Número de flujos", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=35, labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_3_3_balanceo_clases.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"-> Figure saved: {path}")


def plot_feature_importance(experiment, data, figsize=(10.5, 6)):
    print(f"\nGenerating Figure 4.4 (feature importance, {experiment})...")
    top_features = data.get("top_features")
    if top_features is None:
        print("  [!] No 'top_features' in results.json - rerun the experiment to save it. Skipping.")
        return
    feats = list(top_features.keys())[::-1]
    values = list(top_features.values())[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(feats, values, color="steelblue")
    ax.set_xlabel("Importancia", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_4_4_importancia_caracteristicas_exp3.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"-> Figure saved: {path}")


CONFUSION_MATRIX_FIGURE_NUMBERS = {"experiment1": "4.1", "experiment2": "4.2", "experiment3": "4.3"}
CONFUSION_MATRIX_FILENAMES = {
    "experiment1": "fig_4_1_matriz_confusion_exp1.png",
    "experiment2": "fig_4_2_matriz_confusion_exp2.png",
    "experiment3": "fig_4_3_matriz_confusion_exp3.png",
}


def plot_confusion_matrix(experiment, data, figsize=(7, 6)):
    fig_num = CONFUSION_MATRIX_FIGURE_NUMBERS.get(experiment, "?")
    print(f"\nGenerating Figure {fig_num} (confusion matrix, {experiment})...")
    cm_data = data.get("confusion_matrix")
    if cm_data is None:
        print("  [!] No 'confusion_matrix' in results.json - rerun the experiment to save it. Skipping.")
        return
    labels = cm_data["labels"]
    cm = np.array(cm_data["matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cm_pct, cmap="Greens", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Etiqueta predicha", fontsize=11); ax.set_ylabel("Etiqueta real", fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n({int(cm[i, j]):,})",
                    ha="center", va="center", color=color, fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, CONFUSION_MATRIX_FILENAMES[experiment])
    plt.savefig(path, dpi=150); plt.close()
    print(f"-> Figure saved: {path}")


# ── Figures 4.5 and 4.6: evolution + with/without few-shot ────────────────────
def plot_experiment_evolution(rows):
    print("\nGenerating Figure 4.5 (balanced accuracy / F1 macro evolution)...")
    rows = [r for r in rows if r["short_label"] != "Validación intra-IDS2017"]
    if not rows:
        print("  [!] No rows in comparison table; skipping."); return
    labels = [r["short_label"] for r in rows]
    bals   = [r["balanced_accuracy"] for r in rows]
    f1s    = [r["f1_macro"] if r["f1_macro"] is not None else np.nan for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.plot(x, bals, marker="o", color="steelblue", label="Balanced accuracy", linewidth=3, markersize=10)
    ax.plot(x, f1s,  marker="s", color="darkorange", label="F1 macro", linewidth=3, markersize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Referencia (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel("Valor de la métrica", fontsize=24, fontweight="bold")
    ax.legend(fontsize=20); plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_4_5_evolucion_experimentos.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"-> Figure saved: {path}")


def plot_f1_per_class_fewshot():
    print("\nGenerating Figure 4.6 (F1 per class, with and without few-shot)...")
    data2 = load_json("results_experiment2.json")
    data3 = load_json("results_experiment3.json")
    if data2 is None or data3 is None:
        print("  [!] experiment2 or experiment3 results missing; skipping."); return

    rep_base = data2["threshold_05"]["classification_report"]
    rep_fs   = data3["classification_report"]
    classes  = sorted(data2["classes"])

    f1_base = [rep_base.get(c, {}).get("f1-score", 0.0) for c in classes]
    f1_fs   = [rep_fs.get(c, {}).get("f1-score", 0.0) for c in classes]

    x, w = np.arange(len(classes)) * 1.5, 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, f1_base, w, label="Sin few-shot (Experimento 2)", color="darkorange")
    ax.bar(x + w/2, f1_fs,   w, label="Few-shot 20% (Experimento 3)", color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("F1 por clase", fontsize=12, fontweight="bold"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11); plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_4_6_f1_clase_fewshot.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"-> Figure saved: {path}")


if __name__ == "__main__":

    os.makedirs(RESULTS_BASE, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    rows = build_comparison_table()
    print_table(rows)

    with open(os.path.join(RESULTS_BASE, "comparison_table.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n-> Comparison table saved to: {RESULTS_BASE}/comparison_table.json")

    plot_class_distribution_by_dataset()
    plot_experiment_evolution(rows)
    plot_f1_per_class_fewshot()

    for experiment in ["experiment1", "experiment2", "experiment3"]:
        data = load_json(f"results_{experiment}.json")
        if data is None:
            continue
        if experiment == "experiment1":
            plot_class_balance(experiment, data)
        if experiment == "experiment3":
            plot_feature_importance(experiment, data)
        plot_confusion_matrix(experiment, data)
