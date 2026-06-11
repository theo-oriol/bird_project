import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor

MODEL_PATH = Path(sys.argv[1])

LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]

with open("configs/labeltoname.json") as f:
    class_names = json.load(f)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def per_class_auc(probs, binary):
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
#  Load data                                                           #
# ------------------------------------------------------------------ #

df_preds = pd.concat([
    pd.read_csv(MODEL_PATH / f"fold_{i}" / "val_predictions.csv")
    for i in range(3)
    if (MODEL_PATH / f"fold_{i}" / "val_predictions.csv").exists()
], ignore_index=True)

df_meta = pd.read_csv("data/datasets/dataset_version_1.csv")
df_meta = df_meta[["name_of_img", "sexe", "family", "genre", "species"] + LABEL_COLS].copy()
df_meta[LABEL_COLS] = df_meta[LABEL_COLS] / 100.0

df = df_preds.merge(df_meta, left_on="image", right_on="name_of_img", how="left")
missing = df["sexe"].isna().sum()
if missing > 0:
    print(f"[warn] {missing} images not found in metadata — dropped")
    df = df.dropna(subset=["sexe", "family"])

prob_cols     = [c for c in df_preds.columns if c.startswith("pred_")]
true_bin_cols = [f"true_{c}" for c in LABEL_COLS]

probs  = df[prob_cols].values
binary = (df[true_bin_cols].values > 0).astype(int)


# ------------------------------------------------------------------ #
#  Family-level analysis — cut families with < 15 images              #
# ------------------------------------------------------------------ #

MIN_FAMILY_SIZE = 15

families   = df["family"].values
fam_unique = df["family"].dropna().unique()

fam_rows = []
for fam in fam_unique:
    mask = families == fam
    if mask.sum() < MIN_FAMILY_SIZE:
        continue

    fam_auc = mean_auc(probs[mask], binary[mask])

    cont_labels = df.loc[mask, LABEL_COLS].values
    variation   = cont_labels.std(axis=0).mean()
    mean_dist   = cont_labels.mean(axis=0)
    mean_dist   = mean_dist / (mean_dist.sum() + 1e-8)
    entropy     = -np.sum(mean_dist * np.log(mean_dist + 1e-8))

    fam_rows.append({
        "family":   fam,
        "n":        mask.sum(),
        "mean_auc": fam_auc,
        "variation": variation,
        "entropy":  entropy,
    })

df_fam = pd.DataFrame(fam_rows).dropna()

# scale
for col in ["variation", "n"]:
    df_fam[f"{col}_scaled"] = (
        (df_fam[col] - df_fam[col].mean()) / df_fam[col].std()
    )

# add species, genre, sex ratio
df_species = df.groupby("family")["species"].nunique().reset_index()
df_species.columns = ["family", "n_species"]

df_genre = df.groupby("family")["genre"].nunique().reset_index()
df_genre.columns = ["family", "n_genre"]

df_sex_ratio = df.groupby("family").apply(
    lambda x: (x["sexe"] == "M").mean()
).reset_index()
df_sex_ratio.columns = ["family", "prop_male"]

df_fam = df_fam.merge(df_species,   on="family")
df_fam = df_fam.merge(df_genre,     on="family")
df_fam = df_fam.merge(df_sex_ratio, on="family")


for col in ["n_species", "n_genre", "prop_male"]:
    df_fam[f"{col}_scaled"] = (
        (df_fam[col] - df_fam[col].mean()) / df_fam[col].std()
    )

# no weighting — small families already excluded
model_fam = smf.ols(
    "mean_auc ~ variation_scaled + n_species_scaled + prop_male_scaled",
    data=df_fam,
).fit()
print("\n" + "="*60)
print("Family-level AUC analysis")
print("="*60)
print(model_fam.summary())

X_fam = df_fam[["variation_scaled", "n_species_scaled", "prop_male_scaled"]].dropna()
vif_fam = pd.DataFrame({
    "variable": X_fam.columns,
    "VIF": [variance_inflation_factor(X_fam.values, i) for i in range(X_fam.shape[1])]
})
print(vif_fam)


# ------------------------------------------------------------------ #
#  Per-image analysis using log loss                                   #
# ------------------------------------------------------------------ #

eps = 1e-7
probs_clipped = np.clip(df[prob_cols].values, eps, 1 - eps)

# per-image log loss averaged across classes
df["logloss"] = -np.mean(
    binary * np.log(probs_clipped) + (1 - binary) * np.log(1 - probs_clipped),
    axis=1
)

# entropy of each image's own label vector
label_vals    = df[LABEL_COLS].values
label_vals    = label_vals / (label_vals.sum(axis=1, keepdims=True) + 1e-8)
df["entropy"] = -np.sum(label_vals * np.log(label_vals + 1e-8), axis=1)

# number of habitats present
df["n_habitats"] = (df[LABEL_COLS].values > 0).sum(axis=1)

# attach family-level variation
df = df.merge(
    df_fam[["family", "variation", "n"]],
    on="family",
    how="left"
)
df = df.rename(columns={"variation": "family_variation", "n": "family_n"})

# drop images from families that were excluded
df_img = df.dropna(subset=["logloss", "family_variation", "sexe", "family_n"])

# scale
df_img = df_img.copy() 
for col in ["family_variation", "n_habitats"]:
    df_img[f"{col}_scaled"] = (
        (df_img[col] - df_img[col].mean()) / df_img[col].std()
    )

# ---- Gamma GLM ----
model_gamma = smf.glm(
    "logloss ~ C(sexe) + family_variation_scaled + n_habitats_scaled",
    data=df_img,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

print("\n" + "="*60)
print("Per-image log loss analysis (Gamma GLM)")
print("="*60)
print(model_gamma.summary())

# VIF
X_img = pd.get_dummies(
    df_img[["sexe", "family_variation_scaled", "n_habitats_scaled"]].dropna(),
    columns=["sexe"],
    drop_first=True
).astype(float)

vif_img = pd.DataFrame({
    "variable": X_img.columns,
    "VIF": [variance_inflation_factor(X_img.values, i) for i in range(X_img.shape[1])]
})
print(vif_img)

