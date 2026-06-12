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

df = pd.concat([
    pd.read_csv(MODEL_PATH / f"fold_{i}" / "val_predictions.csv")
    for i in range(3)
], ignore_index=True)


prob_cols = [c for c in df.columns if c.startswith("pred_")]
gt_cols = [c for c in df.columns if c.startswith("true_")]
probs     = df[prob_cols].values                  
presence  = df[gt_cols].values                

results = []
for c, name in enumerate(class_names):
    y_true = presence[:, c]   
    y_pred = probs[:, c] 

    slope, intercept, r, p, se = linregress(y_pred, y_true)
    results.append({
        "class":     name,
        "slope":     slope,
        "intercept": intercept,
        "r":         r,
        "r2":        r**2,
        "p":         p,
        "se":        se,
        "significant": p < 0.05,
    })

df_results = pd.DataFrame(results)


print(f"\n{'='*70}")
print(f"{'class':45s} {'slope':>8} {'r':>6} {'r2':>6} {'p':>10} {'sig':>4}")
print(f"{'='*70}")
for _, row in df_results.iterrows():
    sig = "✓" if row["significant"] else ""
    print(
        f"{row['class']:45s} "
        f"{row['slope']:>8.4f} "
        f"{row['r']:>6.3f} "
        f"{row['r2']:>6.3f} "
        f"{row['p']:>10.2e} "
        f"{sig:>4}"
    )

print(f"\n  significant classes: {df_results['significant'].sum()}/{len(df_results)}")
print(f"  mean r2:             {df_results['r2'].mean():.4f}")