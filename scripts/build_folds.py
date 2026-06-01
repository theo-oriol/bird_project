import pandas as pd
import numpy as np
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pathlib import Path
import json 

VERSION = 1
TYPE_OF_SPLIT = "SPE" # ["FAM","GENRE", "SPE"]
N_FOLDS  = 3

CONFIG_DIR = "configs" 
DOCS_DIR  = Path("docs")
DATASET = f"data/datasets/dataset_version_{VERSION}.csv"
FOLD_DIR =f"data/cross"
FOLD_NAME= f"cross_version_{VERSION}_{TYPE_OF_SPLIT}"

CROSS_PATH = os.path.join(FOLD_DIR,FOLD_NAME)

if os.path.exists(CROSS_PATH):
    raise FileExistsError(f"Output file '{CROSS_PATH}' already exists")
os.makedirs(CROSS_PATH)

path_out_json = os.path.join(CONFIG_DIR,"labelname.json")
with open(path_out_json, "r") as f:
    LABEL_COLS = json.load(f)  

cv = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

split = {"FAM": "family", "GENRE": "genre", "SPE": "species"}.get(TYPE_OF_SPLIT)
if split is None:
    raise ValueError(f"Unknown split type '{TYPE_OF_SPLIT}'")

df = pd.read_csv(DATASET)
split_dict = (df.drop_duplicates(subset=split, keep="first")
              .set_index(split)[LABEL_COLS]
              .to_dict("index"))

X      = np.array(list(split_dict.keys()))
y      = (np.array([list(split_dict[k].values()) for k in X])>0).astype(int)
fold_stats = []


for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    df_train = df[df[split].isin(X[train_idx])]
    df_val = df[df[split].isin(X[val_idx])]

    df_train.to_csv( os.path.join(CROSS_PATH, f"train_fold_{fold}.csv"), index=False)
    df_val.to_csv(  os.path.join(CROSS_PATH, f"valid_fold_{fold}.csv"), index=False)

    fold_stats.append({
        "fold":           fold,
        "train_images":   len(df_train),
        "val_images":     len(df_val),
        "train_species":  df_train["species"].nunique(),
        "val_species":    df_val["species"].nunique(),
        "train_families": df_train["family"].nunique(),
        "val_families":   df_val["family"].nunique(),
        "train_genres":   df_train["genre"].nunique(),
        "val_genres":     df_val["genre"].nunique(),
    })

path_out_json = os.path.join(CROSS_PATH,"labelname.json")
if not os.path.exists(path_out_json):
    with open(path_out_json, "w") as f:
        json.dump(LABEL_COLS, f, indent=2)


# ── Append cross-validation section to the dataset doc ───────────────────────
doc_path = DOCS_DIR / f"dataset_v{VERSION}.md"
with open(doc_path, "a") as f:
    f.write(f"\n---\n\n")
    f.write(f"# Cross-Validation — {FOLD_NAME}\n\n")

    f.write("## Configuration\n")
    f.write(f"| Parameter | Value |\n|---|---|\n")
    f.write(f"| Folds | {N_FOLDS} |\n")
    f.write(f"| Split strategy | {split.capitalize()} |\n")
    f.write(f"| Random state | 42 |\n\n")

    f.write("## Per-Fold Statistics\n")
    f.write("| Fold | Train images | Val images | Train species | Val species | Train families | Val families | Train genres | Val genres |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for s in fold_stats:
        f.write(f"| {s['fold']} | {s['train_images']:,} | {s['val_images']:,} | {s['train_species']} | {s['val_species']} | {s['train_families']} | {s['val_families']} | {s['train_genres']} | {s['val_genres']} |\n")

print(f"Cross-validation '{FOLD_NAME}' built. Doc updated at {doc_path}")