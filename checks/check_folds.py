import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import json 
from pathlib import Path

CROSS_NAME = "cross_version_1_FAM"
N_FOLDS = 3

CROSS_DIR = Path(__file__).parent.parent / "data/cross"
CROSS_PATH = CROSS_DIR / CROSS_NAME
CONFIG_DIR = Path(__file__).parent.parent / "configs" 


path_out_json = os.path.join(CONFIG_DIR,"labelname.json")
with open(path_out_json, "r") as f:
    LABEL_COLS = json.load(f)  


TYPE_OF_SPLIT = CROSS_NAME.split("_")[-1]

split = {"FAM": "family", "GENRE": "genre", "SPE": "species"}.get(TYPE_OF_SPLIT)
if split is None:
    raise ValueError(f"Unknown split type '{TYPE_OF_SPLIT}'")

fig, axes = plt.subplots(N_FOLDS, 1, figsize=(12, N_FOLDS * 4), sharey=False)
for fold in range(N_FOLDS):
    ax = axes[fold]

    df_train = pd.read_csv(os.path.join(CROSS_PATH, f"train_fold_{fold}.csv"))
    df_val   = pd.read_csv(os.path.join(CROSS_PATH, f"valid_fold_{fold}.csv"))

    
    shared = set(df_train[split].unique()) & set(df_val[split].unique())
    print(f"shared {TYPE_OF_SPLIT}: {len(shared)} / {df_train[split].nunique()} train, {df_val[split].nunique()} val")

    train_presence = (df_train[LABEL_COLS] > 0).mean(axis=0) * 100  # % of images with label
    val_presence   = (df_val[LABEL_COLS]   > 0).mean(axis=0) * 100

    x     = np.arange(len(LABEL_COLS))
    width = 0.35

    ax.bar(x - width/2, train_presence, width, label=f"train ({len(df_train)})", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, val_presence,   width, label=f"val   ({len(df_val)})",   color="salmon",    alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_COLS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% images with label")
    ax.set_title(f"Fold {fold} — label presence distribution")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # annotate diff
    for i, (t, v) in enumerate(zip(train_presence, val_presence)):
        diff = abs(t - v)
        if diff > 5:
            ax.text(i, max(t, v) + 1, f"Δ{diff:.1f}", ha="center", fontsize=7, color="red")

plt.suptitle("Binary label distribution per fold\n(red Δ = train/val difference > 5%)", y=1.01)
plt.tight_layout()

plot_out_path = os.path.join(CROSS_PATH, "fold_label_distribution.png")
plt.savefig(plot_out_path, dpi=150, bbox_inches="tight")
