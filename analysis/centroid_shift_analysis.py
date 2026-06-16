"""
Regression: per-family AUC ~ centroid-shift metrics (Back / Belly / Side).

Usage:
    python analysis/centroid_shift_analysis.py <model_path>

<model_path> must contain fold_0/, fold_1/, fold_2/ sub-directories with
val_predictions.csv files.
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

MODEL_PATH = Path(sys.argv[1])
META_DIR = Path("data/meta")
DATASET_CSV = Path("data/datasets/dataset_version_1.csv")

LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]
PRED_COLS = [f"pred_{c}" for c in LABEL_COLS]
TRUE_COLS = [f"true_{c}" for c in LABEL_COLS]

VIEWS = ["Back", "Belly", "Side"]
N_FOLDS = 3
MIN_FAMILY_SIZE = 10


# ------------------------------------------------------------------ #
#  Loaders                                                             #
# ------------------------------------------------------------------ #

def load_predictions() -> pd.DataFrame:
    frames = []
    for fold in range(N_FOLDS):
        path = MODEL_PATH / f"fold_{fold}" / "val_predictions.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["fold"] = fold
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No val_predictions.csv found under {MODEL_PATH}")
    return pd.concat(frames, ignore_index=True)


def load_family_metadata() -> pd.DataFrame:
    return pd.read_csv(DATASET_CSV, usecols=["name_of_img", "family"])


def load_centroid_shift() -> pd.DataFrame:
    frames = []
    for fold in range(N_FOLDS):
        for view in VIEWS:
            path = META_DIR / f"family_centroid_shift_fold{fold}_{view}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No centroid shift files found in {META_DIR}")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ #
#  Feature engineering                                                 #
# ------------------------------------------------------------------ #

def extract_view(image: str) -> str | None:
    for view in VIEWS:
        if f"_{view}_" in image:
            return view
    return None


def mean_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    aucs = []
    for i in range(probs.shape[1]):
        if len(np.unique(targets[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_family_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, view, fold), grp in df.groupby(["family", "view", "fold"]):
        if len(grp) < MIN_FAMILY_SIZE:
            continue
        probs = grp[PRED_COLS].values
        targets = (grp[TRUE_COLS].values > 0).astype(int)
        rows.append({
            "val_family": family,
            "view": view,
            "fold": int(fold),
            "mean_auc": mean_auc(probs, targets),
        })
    return pd.DataFrame(rows)


def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()
    return df


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

# Load and enrich predictions with family + view
df_preds = load_predictions()
df_meta = load_family_metadata()
df_preds = df_preds.merge(df_meta, left_on="image", right_on="name_of_img", how="left")
df_preds["view"] = df_preds["image"].apply(extract_view)

missing = df_preds["family"].isna().sum()
if missing:
    print(f"[warn] {missing} images not found in metadata — dropped")
df_preds = df_preds.dropna(subset=["family", "view"])

# Compute per-family AUC
df_auc = compute_family_auc(df_preds)

# Load centroid shift data
df_centroid = load_centroid_shift()
centroid_cols = ["val_family", "view", "fold", "min_dist_to_train", "dist_to_train_global", "label_entropy"]

# Merge
df = df_auc.merge(df_centroid[centroid_cols], on=["val_family", "view", "fold"], how="inner")
df = df.dropna(subset=["mean_auc", "min_dist_to_train", "dist_to_train_global", "label_entropy"])

print(f"\nAnalysis dataframe: {len(df)} observations across {df['val_family'].nunique()} families")
print(df.groupby("view").size().rename("n_obs").to_string())

# Standardize predictors
PRED_VARS = ["min_dist_to_train", "dist_to_train_global", "label_entropy"]
df = standardize(df, PRED_VARS)

# ------------------------------------------------------------------ #
#  Pooled regression (all views, view as covariate)                   #
# ------------------------------------------------------------------ #

model_pooled = smf.ols(
    "mean_auc ~ min_dist_to_train_z + dist_to_train_global_z + label_entropy_z + C(view)",
    data=df,
).fit()

print("\n" + "=" * 60)
print("Pooled: mean_auc ~ centroid shift + label entropy + view")
print("=" * 60)
print(model_pooled.summary())

X_pooled = pd.get_dummies(df[["view"] + [f"{c}_z" for c in PRED_VARS]], columns=["view"], drop_first=True).astype(float)
vif_pooled = pd.DataFrame({
    "variable": X_pooled.columns,
    "VIF": [variance_inflation_factor(X_pooled.values, i) for i in range(X_pooled.shape[1])],
})
print("\nVIF (pooled):")
print(vif_pooled.to_string(index=False))

model_mixed = smf.mixedlm(
    "mean_auc ~ min_dist_to_train_z + dist_to_train_global_z + label_entropy_z + C(view)",
    data=df,
    groups=df["val_family"]   # random intercept per family
).fit()
print(model_mixed.summary())


# ------------------------------------------------------------------ #
#  Per-view regressions                                                #
# ------------------------------------------------------------------ #

# for view in VIEWS:
#     df_view = df[df["view"] == view]
#     if len(df_view) < 5:
#         print(f"\n[skip] {view}: only {len(df_view)} observations")
#         continue

#     model_view = smf.ols(
#         "mean_auc ~ min_dist_to_train_z + dist_to_train_global_z + label_entropy_z",
#         data=df_view,
#     ).fit()

#     print(f"\n{'=' * 60}")
#     print(f"{view}: mean_auc ~ centroid shift + label entropy  (n={len(df_view)})")
#     print("=" * 60)
#     print(model_view.summary())
