import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, spearmanr

MODEL_PATH = Path(sys.argv[1])

LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]

with open("configs/labeltoname.json") as f:
    class_names = json.load(f)

# ------------------------------------------------------------------ #
#  Load data                                                           #
# ------------------------------------------------------------------ #

# predictions across all folds
df_preds = pd.concat([
    pd.read_csv(MODEL_PATH / f"fold_{i}" / "val_predictions.csv")
    for i in range(3)
    if (MODEL_PATH / f"fold_{i}" / "val_predictions.csv").exists()
], ignore_index=True)

# metadata + continuous labels
df_meta = pd.read_csv("data/datasets/dataset_version_1.csv")
df_meta = df_meta[["name_of_img", "sexe", "family", "genre" , "species"] + LABEL_COLS].copy()
df_meta[LABEL_COLS] = df_meta[LABEL_COLS] / 100.0

# merge
df = df_preds.merge(df_meta, left_on="image", right_on="name_of_img", how="left")
missing = df["sexe"].isna().sum()
if missing > 0:
    print(f"[warn] {missing} images not found in metadata — dropped")
    df = df.dropna(subset=["sexe", "family"])

prob_cols     = [c for c in df_preds.columns if c.startswith("pred_")]
true_bin_cols = [f"true_{c}" for c in LABEL_COLS]

probs  = df[prob_cols].values                              # (N, C)
binary = (df[true_bin_cols].values > 0).astype(int)       # (N, C)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def per_class_auc(probs, binary):
    """Returns array of AUC per class, nan if class has no positives."""
    n = probs.shape[1]
    aucs = []
    for i in range(n):
        if len(np.unique(binary[:, i])) < 2:
            aucs.append(float("nan"))
        else:
            aucs.append(roc_auc_score(binary[:, i], probs[:, i]))
    return np.array(aucs)


def mean_auc(probs, binary):
    aucs = per_class_auc(probs, binary)
    valid = aucs[~np.isnan(aucs)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


# ------------------------------------------------------------------ #
#  1. Sex analysis                                                     #
# ------------------------------------------------------------------ #

def sex_analysis(df, probs, binary, class_names, out_dir):
    sexes     = df["sexe"].values
    sex_vals  = sorted(df["sexe"].dropna().unique())

    # per-class AUC per sex
    sex_aucs = {}
    for s in sex_vals:
        mask = sexes == s
        if mask.sum() < 10:
            continue
        sex_aucs[s] = per_class_auc(probs[mask], binary[mask])

    if len(sex_aucs) < 2:
        print("[sex] not enough groups to compare")
        return

    # ---- plot: per-class AUC bar chart grouped by sex ----
    n_classes = len(class_names)
    x         = np.arange(n_classes)
    width     = 0.8 / len(sex_aucs)
    colors    = ["steelblue", "coral", "seagreen", "gold"]

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 1.2), 5))
    for i, (s, aucs) in enumerate(sex_aucs.items()):
        offset = (i - len(sex_aucs) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, aucs, width, label=s, color=colors[i % len(colors)], alpha=0.8)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("AUC per class by sex")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sex_auc_per_class.png", dpi=150)
    plt.close()

    # ---- plot: mean AUC per sex with Mann-Whitney p-value ----
    group_means = {s: np.nanmean(aucs) for s, aucs in sex_aucs.items()}

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(list(group_means.keys()), list(group_means.values()),
           color=colors[:len(group_means)], alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Mean AUC by sex")
    ax.grid(axis="y", alpha=0.3)



# ------------------------------------------------------------------ #
#  2. Family analysis                                                  #
# ------------------------------------------------------------------ #

def family_analysis(df, probs, binary, class_names, out_dir):
    families   = df["family"].values
    fam_unique = df["family"].dropna().unique()

    # per-family: mean AUC + habitat variation
    fam_rows = []
    for fam in fam_unique:
        mask = families == fam
        if mask.sum() < 5:
            continue

        # mean AUC for this family
        fam_auc = mean_auc(probs[mask], binary[mask])

        # habitat variation = mean entropy across classes
        # use the continuous labels from df_meta merged into df
        cont_labels = df.loc[mask, LABEL_COLS].values   # (n_fam, C)
        # total variation: std across images for each class, then mean
        variation   = cont_labels.std(axis=0).mean()
        # also compute entropy of mean distribution
        mean_dist   = cont_labels.mean(axis=0)
        mean_dist   = mean_dist / (mean_dist.sum() + 1e-8)
        entropy     = -np.sum(mean_dist * np.log(mean_dist + 1e-8))

        fam_rows.append({
            "family":    fam,
            "n":         mask.sum(),
            "mean_auc":  fam_auc,
            "variation": variation,
            "entropy":   entropy,
        })

    df_fam = pd.DataFrame(fam_rows).dropna()

    # ---- plot: variation vs AUC scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, x_col, x_label in zip(
        axes,
        ["variation", "entropy"],
        ["Habitat std (within-family variation)", "Habitat entropy (mean distribution)"]
    ):
        x = df_fam[x_col].values
        y = df_fam["mean_auc"].values

        ax.scatter(x, y, s=df_fam["n"] / df_fam["n"].max() * 200,
                   alpha=0.7, color="steelblue", edgecolors="white", linewidth=0.5)

        # annotate family names
        # for _, row in df_fam.iterrows():
        #     ax.annotate(row["family"],
        #                 (row[x_col], row["mean_auc"]),
        #                 fontsize=5, alpha=0.7,
        #                 xytext=(3, 3), textcoords="offset points")

        # regression line + spearman
        rho, p = spearmanr(x, y)
        m, b   = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color="orange",
                linestyle="--", linewidth=1.5,
                label=f"ρ={rho:+.3f}  p={p:.3f}")

        ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean AUC")
        ax.set_title(f"Family AUC vs {x_col}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.05, f"p={p:.3f}",
                transform=ax.transAxes, fontsize=9,
                color="green" if p < 0.05 else "gray")


    plt.suptitle("Family habitat variation vs AUC", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "family_variation_vs_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

        # ---- plot: family size vs AUC ----
    fig, ax = plt.subplots(figsize=(6, 5))

    x = df_fam["n"].values
    y = df_fam["mean_auc"].values

    ax.scatter(x, y, alpha=0.7, color="steelblue",
            edgecolors="white", linewidth=0.5)


    rho, p = spearmanr(x, y)
    m, b   = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color="orange",
            linestyle="--", linewidth=1.5,
            label=f"ρ={rho:+.3f}  p={p:.3f}")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of images in family")
    ax.set_ylabel("Mean AUC")
    ax.set_title("Family size vs AUC")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.05, f"p={p:.3f}",
            transform=ax.transAxes, fontsize=9,
            color="green" if p < 0.05 else "gray")

    plt.tight_layout()
    plt.savefig(out_dir / "family_size_vs_auc.png", dpi=150)
    plt.close()






    
# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    out_dir = MODEL_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    sex_analysis(df, probs, binary, class_names, out_dir)
    family_analysis(df, probs, binary, class_names, out_dir)

    print(f"\nPlots saved to {out_dir}")