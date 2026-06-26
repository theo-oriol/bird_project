import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import linregress
from pathlib import Path

BENCHMARK_PATH = Path(sys.argv[1])
CROSS_PATH     = Path("data/cross/")

with open(BENCHMARK_PATH) as f:
    cfg = yaml.safe_load(f)

bench      = cfg["benchmark"]
bench_name = bench["name"]
dataset_dir = CROSS_PATH / bench["dataset_dir"]

with open(dataset_dir / "labeltoname.json") as f:
    class_names = json.load(f)

with open(dataset_dir / "labelname.json") as f:
    label_ids = json.load(f)

LABEL_COLS = [str(lid) for lid in label_ids]

df_gt = pd.read_csv("data/datasets/dataset_version_1.csv")
df_gt = df_gt[["name_of_img"] + LABEL_COLS].copy()
df_gt[LABEL_COLS] = df_gt[LABEL_COLS] / 100.0

thresholds = np.arange(1, 11) / 10
n_classes  = len(class_names)
n_cols     = 4
n_rows     = (n_classes + n_cols - 1) // n_cols


def compute_auc_curves(df_preds, mode):
    """
    Returns a dict keyed by class name:
      { name: (valid_thresh, auc_per_thresh, n_pos_per_thresh, slope, p) }
    slope/p are None when fewer than 3 thresholds have valid data.
    """
    df = df_preds.merge(df_gt, left_on="image", right_on="name_of_img", how="left")
    missing = df[LABEL_COLS[0]].isna().sum()
    if missing > 0:
        df = df.dropna(subset=LABEL_COLS)

    prob_cols = [c for c in df.columns if c.startswith("pred_")]
    probs    = df[prob_cols].values
    presence = df[LABEL_COLS].values
    binary   = (df[[f"true_{c}" for c in LABEL_COLS]].values > 0).astype(int)

    curves = {}
    for c, name in enumerate(class_names):
        y_true_cont = presence[:, c]
        y_true_bin  = binary[:, c]
        y_score     = probs[:, c]

        auc_per_thresh, valid_thresh, n_pos_per_thresh = [], [], []
        for t in thresholds:
            if mode == "slice":
                mask = ((y_true_cont > 0) & (y_true_cont > t - 0.1) & (y_true_cont <= t)) | (y_true_cont == 0)
            else:
                mask = ((y_true_cont > 0) & (y_true_cont <= t)) | (y_true_cont == 0)

            _y_true  = y_true_bin[mask]
            _y_score = y_score[mask]
            if len(np.unique(_y_true)) < 2:
                continue

            auc_per_thresh.append(roc_auc_score(_y_true, _y_score))
            valid_thresh.append(t)
            n_pos_per_thresh.append(_y_true.sum())

        slope = intercept = p = None
        if len(valid_thresh) >= 3:
            slope, intercept, _, p, _ = linregress(valid_thresh, auc_per_thresh)

        curves[name] = (valid_thresh, auc_per_thresh, n_pos_per_thresh, slope, intercept, p)
    return curves


def _draw_habitat_grid(all_curves, title, out_path, show_n=True):
    """
    all_curves: dict  { exp_name: curves_dict }
    curves_dict: { habitat_name: (valid_thresh, auc_per_thresh, n_pos, slope, intercept, p) }
    Draws one subplot per habitat with all models overlaid.
    """
    cmap    = plt.cm.tab10
    exp_names = list(all_curves.keys())
    colors  = [cmap(i % 10) for i in range(len(exp_names))]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for c, hab in enumerate(class_names):
        ax = axes[c]
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(hab, fontsize=9)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Presence (generalist → specialist)", fontsize=7)
        ax.set_ylabel("AUC", fontsize=7)
        ax.grid(alpha=0.3)

        for exp_name, color in zip(exp_names, colors):
            curves = all_curves[exp_name]
            if hab not in curves:
                continue
            valid_thresh, auc_per_thresh, n_pos, slope, intercept, p = curves[hab]
            if not valid_thresh:
                continue
            
            if p :
                ax.plot(valid_thresh, auc_per_thresh, marker="o", color=color,
                        linewidth=1.5, markersize=4, label=f"{exp_name} = {slope:.3f} : {p:.3f}")
            else : 
                ax.plot(valid_thresh, auc_per_thresh, marker="o", color=color,
                        linewidth=1.5, markersize=4, label=f"{exp_name} = {slope} : {p}")

            if slope is not None:
                x_line = np.array([min(valid_thresh), max(valid_thresh)])
                ax.plot(x_line, slope * x_line + intercept,
                        color=color, linestyle="--", linewidth=1, alpha=0.7)

        ax.legend(fontsize=5, loc="upper left")

    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def run_specialisation(df_preds, out_dir):
    missing = df_preds.merge(df_gt, left_on="image", right_on="name_of_img", how="left")[LABEL_COLS[0]].isna().sum()
    if missing > 0:
        print(f"  [warn] {missing} images not found in GT — skipping them")

    all_curves = {}
    for mode in ("slice", "cumulative"):
        curves = compute_auc_curves(df_preds, mode)
        all_curves[mode] = curves

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes = axes.flatten()
        results = []

        for c, name in enumerate(class_names):
            valid_thresh, auc_per_thresh, n_pos_per_thresh, slope, intercept, p = curves[name]

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

            if slope is not None:
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

        suffix = "" if mode == "slice" else "_cumu"
        title  = "AUC vs Presence threshold per habitat" + (" — CUMULATIVE" if mode == "cumulative" else "")
        plt.suptitle(title, y=1.01)
        plt.tight_layout()

        out_path = out_dir / f"auc_vs_per_hab{suffix}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {out_path}")

        print(f"\n  {'habitat':40s} {'slope':>8} {'p':>8}")
        for name, slope, p in results:
            print(f"  {name:40s} {slope:+8.3f} {p:8.3f}")

    return all_curves


combined_curves = {"slice": {}, "cumulative": {}}   # { mode: { exp_name: curves } }

for exp_name, exp_cfg in cfg["experiments"].items():
    print(f"\n{'='*60}\n{exp_name}\n{'='*60}")

    done_folds = {
        fold_idx: fold_cfg
        for fold_idx, fold_cfg in exp_cfg["folds"].items()
        if fold_cfg["status"] == "done"
    }

    if not done_folds:
        print("  no done folds — skipping")
        continue

    fold_dfs = []
    for fold_idx, fold_cfg in done_folds.items():
        csv_path = Path("experiments") / bench_name / exp_name / fold_cfg["run_dir"] / "val_predictions.csv"
        if not csv_path.exists():
            print(f"  [warn] {csv_path} not found — skipping fold {fold_idx}")
            continue
        fold_dfs.append(pd.read_csv(csv_path))

    if not fold_dfs:
        print("  no prediction files found — skipping")
        continue

    df_preds = pd.concat(fold_dfs, ignore_index=True)

    # save next to the experiment-level summary (one dir above fold dirs)
    first_run_dir = next(iter(done_folds.values()))["run_dir"]
    out_dir = Path("experiments") / bench_name / exp_name / Path(first_run_dir).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    all_curves = run_specialisation(df_preds, out_dir)
    for mode in ("slice", "cumulative"):
        combined_curves[mode][exp_name] = all_curves[mode]

# ---- combined benchmark-level plots ----
bench_out = Path("benchmarks") / "specialisation" / bench_name
bench_out.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}\nCombined benchmark plots\n{'='*60}")
_draw_habitat_grid(
    combined_curves["slice"],
    title=f"{bench_name} — AUC vs Presence per habitat (all models)",
    out_path=bench_out / "auc_vs_per_hab_combined.png",
)
_draw_habitat_grid(
    combined_curves["cumulative"],
    title=f"{bench_name} — AUC vs Presence per habitat — CUMULATIVE (all models)",
    out_path=bench_out / "auc_vs_per_hab_cumu_combined.png",
)
