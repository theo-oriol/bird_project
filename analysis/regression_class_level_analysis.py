import json
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import linregress
from statsmodels.stats.outliers_influence import variance_inflation_factor

MODEL_PATH = Path(sys.argv[1])

with open("configs/labeltoname.json") as f:
    class_names = json.load(f)
LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]

# ------------------------------------------------------------------ #
#  Compute per-class r2 from val predictions                          #
# ------------------------------------------------------------------ #

folds = [
    pd.read_csv(MODEL_PATH / f"fold_{i}" / "val_predictions.csv")
    for i in range(3)
    if (MODEL_PATH / f"fold_{i}" / "val_predictions.csv").exists()
]

prob_cols = [c for c in folds[0].columns if c.startswith("pred_")]
gt_cols   = [c for c in folds[0].columns if c.startswith("true_")]

r2_per_class = []
for c, name in enumerate(class_names):
    fold_r2s = []
    for df in folds:
        y_true = df[gt_cols].values[:, c]
        y_pred = df[prob_cols].values[:, c]
        _, _, r, _, _ = linregress(y_pred, y_true)
        fold_r2s.append(r ** 2)
    r2_per_class.append({"habitat": name, "r2": np.mean(fold_r2s)})

df_r2 = pd.DataFrame(r2_per_class)

# ------------------------------------------------------------------ #
#  Load family habitat stats saved by familly_influence_habitats.py  #
# ------------------------------------------------------------------ #

df_hab = pd.read_csv("data/meta/family_habitat_stats.csv")

# ------------------------------------------------------------------ #
#  Build class-level dataframe                                        #
# ------------------------------------------------------------------ #

df_class_level = df_r2.merge(df_hab[["habitat", "n_families_for_80", "n_families_any", "global_frequency"]], on="habitat")

print("Class-level data:")
print(df_class_level.to_string(index=False))

# ------------------------------------------------------------------ #
#  OLS: r2 ~ n_families_for_80 + global_frequency                    #
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("Class-level OLS: r2 ~ n_families_for_80 + global_frequency")
print("=" * 60)

model = smf.ols("r2 ~ n_families_for_80 + global_frequency", data=df_class_level).fit()
print(model.summary())

X = df_class_level[["n_families_for_80", "global_frequency"]].copy()
vif = pd.DataFrame({
    "variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
})
print("\nVIF:")
print(vif.to_string(index=False))

# ------------------------------------------------------------------ #
#  Extended model with n_families_any                                 #
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("Extended OLS: r2 ~ n_families_for_80 + n_families_any + global_frequency")
print("=" * 60)

model_ext = smf.ols(
    "r2 ~ n_families_for_80 + n_families_any + global_frequency",
    data=df_class_level,
).fit()
print(model_ext.summary())
