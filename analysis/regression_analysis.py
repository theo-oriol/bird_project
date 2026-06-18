import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import linregress, kendalltau
from pathlib import Path
import sys


MODEL_PATH = Path(sys.argv[1])

with open("configs/labeltoname.json") as f:
    class_names = json.load(f)
LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]

folds = [
    pd.read_csv(MODEL_PATH / f"fold_{i}" / "val_predictions.csv")
    for i in range(3)
]

prob_cols = [c for c in folds[0].columns if c.startswith("pred_")]
gt_cols   = [c for c in folds[0].columns if c.startswith("true_")]

results = []
for c, name in enumerate(class_names):
    fold_stats = []
    for df in folds:
        y_true = df[gt_cols].values[:, c]
        y_pred = df[prob_cols].values[:, c]

        slope, intercept, r, p, se = linregress(y_pred, y_true)
        tau, p_tau = kendalltau(y_pred, y_true)
        prob_concordant = (tau + 1) / 2


        y_mean       = y_true.mean()
        mae_model    = np.mean(np.abs(y_true - y_pred))          # direct model error
        mae_constant = np.mean(np.abs(y_true - y_true.mean()))   # baseline error
        mae_skill    = 1 - mae_model / mae_constant

        fold_stats.append({
            "slope": slope, "intercept": intercept,
            "r": r, "r2": r**2, "p": p, "se": se,
            "prob_concordant" : prob_concordant, "p_tau" : p_tau,
            "mae_constant": mae_constant,
            "mae_model": mae_model,
            "mae_skill": mae_skill,
        })

    avg = {k: np.mean([s[k] for s in fold_stats]) for k in fold_stats[0]}
    results.append({
        "class": name,
        **avg,
        "significant": avg["p"] < 0.05,
    })
    

df_results = pd.DataFrame(results)


print(f"\n{'='*70}")
print(f"{'class':45s} {'slope':>8} {'r':>6} {'r2':>6} {'p':>10} {'prob_concordant':>10} {'p_tau':>10} {'sig':>4}")
print(f"{'='*70}")
for _, row in df_results.iterrows():
    sig = "✓" if row["significant"] else ""
    print(
        f"{row['class']:45s} "
        f"{row['slope']:>8.4f} "
        f"{row['r']:>6.3f} "
        f"{row['r2']:>6.3f} "
        f"{row['p']:>10.2e} "
        f"{row['prob_concordant']:>6.3f} "
        f"{row['p_tau']:>10.2e} "
        f"{sig:>4}"
    )

print(f"\n  significant classes: {df_results['significant'].sum()}/{len(df_results)}")
print(f"  mean r2:             {df_results['r2'].mean():.4f}")


print(f"\n{'='*80}")
print(f"{'class':45s} {'MAE_const':>10} {'MAE_model':>10} {'skill':>8} {'sig':>4}")
print(f"{'='*80}")
for _, row in df_results.iterrows():
    sig = "✓" if row["significant"] else ""
    print(
        f"{row['class']:45s} "
        f"{row['mae_constant']:>10.4f} "
        f"{row['mae_model']:>10.4f} "
        f"{row['mae_skill']:>8.1%} "
        f"{sig:>4}"
    )
print(f"\n  mean MAE skill: {df_results['mae_skill'].mean():.1%}")