 import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import linregress
from pathlib import Path
import sys


MODEL_PATH = Path(sys.argv[1])

with open("configs/labeltoname.json") as f:
    class_names = json.load(f)

LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]

# ---- load predictions (binary true_ + pred_) ----
df_preds = pd.concat([
    pd.read_csv(MODEL_PATH / f"fold_{i}" / "val_predictions.csv")
    for i in range(3)
], ignore_index=True)

# ---- load continuous ground truth ----
df_gt = pd.read_csv("data/datasets/dataset_version_1.csv")
df_gt = df_gt[["name_of_img"] + LABEL_COLS].copy()
# normalize to [0, 1]
df_gt[LABEL_COLS] = df_gt[LABEL_COLS] / 100.0

# ---- merge on image name ----
df = df_preds.merge(df_gt, left_on="image", right_on="name_of_img", how="left")

missing = df[LABEL_COLS[0]].isna().sum()
if missing > 0:
    print(f"[warn] {missing} images not found in continuous GT — they will be skipped")
    df = df.dropna(subset=LABEL_COLS)

prob_cols = [c for c in df.columns if c.startswith("pred_")]
probs     = df[prob_cols].values                  # (N, C) predicted probabilities
presence  = df[LABEL_COLS].values                 # (N, C) continuous [0, 1]
binary    = (df[[f"true_{c}" for c in LABEL_COLS]].values > 0).astype(int)  # (N, C) binary

n_classes  = len(class_names)
n_cols     = 4
n_rows     = (n_classes + n_cols - 1) // n_cols
thresholds = np.arange(1, 11) / 10

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
axes      = axes.flatten()

results = []

for c, name in enumerate(class_names):
    y_true_cont = presence[:, c]    # continuous presence
    y_true_bin  = binary[:, c]      # binary presence
    y_score     = probs[:, c]

    auc_per_thresh   = []
    valid_thresh     = []
    n_pos_per_thresh = []

    for t in thresholds:
        # positives: images where continuous presence falls in (t-0.1, t]
        # negatives: images with zero presence
        mask     = ((y_true_cont > 0) & (y_true_cont > t - 0.1) & (y_true_cont <= t)) | (y_true_cont == 0)
        _y_true  = y_true_bin[mask]    # still evaluate AUC on binary labels
        _y_score = y_score[mask]

        if len(np.unique(_y_true)) < 2:
            continue

        auc_per_thresh.append(roc_auc_score(_y_true, _y_score))
        valid_thresh.append(t)
        n_pos_per_thresh.append(_y_true.sum())

    ax = axes[c]
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="random")
    ax.set_title(name, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Presence (generalist → specialist)")
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.3)

    if not valid_thresh:
        continue

    ax.plot(valid_thresh, auc_per_thresh, marker="o", color="steelblue")

    for x, y, n in zip(valid_thresh, auc_per_thresh, n_pos_per_thresh):
        ax.text(x, y + 0.02, str(n), ha="center", fontsize=6, color="gray")

    ax.text(0.98, 0.05, f"AUC={auc_per_thresh[-1]:.3f}",
            transform=ax.transAxes, ha="right", fontsize=7, color="steelblue")

    if len(valid_thresh) >= 3:
        slope, intercept, r, p, se = linregress(valid_thresh, auc_per_thresh)
        results.append((name, slope, p))

        x_line = np.array([min(valid_thresh), max(valid_thresh)])
        ax.plot(x_line, slope * x_line + intercept,
                color="orange", linestyle="--", linewidth=1.5,
                label=f"slope={slope:+.3f}")
        ax.text(0.02, 0.95, f"p={p:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="green" if p < 0.05 else "gray")

    ax.legend(fontsize=7)

for i in range(n_classes, len(axes)):
    axes[i].set_visible(False)

plt.suptitle("AUC vs Presence threshold per habitat — fold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "auc_vs_per_hab.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n{'habitat':40s} {'slope':>8} {'p':>8}")
for name, slope, p in results:
    print(f"{name:40s} {slope:+8.3f} {p:8.3f}")





fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
axes      = axes.flatten()

results = []

for c, name in enumerate(class_names):
    y_true_cont = presence[:, c]    # continuous presence
    y_true_bin  = binary[:, c]      # binary presence
    y_score     = probs[:, c]

    auc_per_thresh   = []
    valid_thresh     = []
    n_pos_per_thresh = []

    for t in thresholds:
        # positives: images where continuous presence falls in (t-0.1, t]
        # negatives: images with zero presence
        mask     = ((y_true_cont > 0) & (y_true_cont <= t)) | (y_true_cont == 0)
        _y_true  = y_true_bin[mask]    # still evaluate AUC on binary labels
        _y_score = y_score[mask]

        if len(np.unique(_y_true)) < 2:
            continue

        auc_per_thresh.append(roc_auc_score(_y_true, _y_score))
        valid_thresh.append(t)
        n_pos_per_thresh.append(_y_true.sum())

    ax = axes[c]
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="random")
    ax.set_title(name, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Presence (generalist → specialist)")
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.3)

    if not valid_thresh:
        continue

    ax.plot(valid_thresh, auc_per_thresh, marker="o", color="steelblue")

    for x, y, n in zip(valid_thresh, auc_per_thresh, n_pos_per_thresh):
        ax.text(x, y + 0.02, str(n), ha="center", fontsize=6, color="gray")

    ax.text(0.98, 0.05, f"AUC={auc_per_thresh[-1]:.3f}",
            transform=ax.transAxes, ha="right", fontsize=7, color="steelblue")

    if len(valid_thresh) >= 3:
        slope, intercept, r, p, se = linregress(valid_thresh, auc_per_thresh)
        results.append((name, slope, p))

        x_line = np.array([min(valid_thresh), max(valid_thresh)])
        ax.plot(x_line, slope * x_line + intercept,
                color="orange", linestyle="--", linewidth=1.5,
                label=f"slope={slope:+.3f}")
        ax.text(0.02, 0.95, f"p={p:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="green" if p < 0.05 else "gray")

    ax.legend(fontsize=7)

for i in range(n_classes, len(axes)):
    axes[i].set_visible(False)

plt.suptitle("AUC vs Presence threshold per habitat — fold CUMULATIVE", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "auc_vs_per_hab_cumu.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n{'habitat':40s} {'slope':>8} {'p':>8}")
for name, slope, p in results:
    print(f"{name:40s} {slope:+8.3f} {p:8.3f}")