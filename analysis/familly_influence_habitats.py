import pandas as pd
import json

with open("configs/labeltoname.json") as f:
    class_names = json.load(f)
LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]

# Load all folds for FAM split, deduplicate
dfs = []
for fold in range(3):
    for split in ["train", "valid"]:
        df = pd.read_csv(f"data/cross/cross_version_1_FAM/{split}_fold_{fold}.csv")
        dfs.append(df)
full = pd.concat(dfs).drop_duplicates(subset="name_of_img")

# Aggregate to one row per family (mean % per class)
fam_df = full.groupby("family")[LABEL_COLS].mean()

print(f"Total unique images: {len(full)}")
print(f"Total unique families: {fam_df.shape[0]}\n")

print("Families per habitat class (family has >0% mean presence):")
print("=" * 65)
for col, name in zip(LABEL_COLS, class_names):
    n = (fam_df[col] > 0).sum()
    pct = n / fam_df.shape[0] * 100
    print(f"  {name:50s}: {n:3d} / {fam_df.shape[0]} ({pct:.0f}%)")

print()
print("Habitat classes per family (how many classes each family touches):")
print("=" * 65)
classes_per_fam = (fam_df > 0).sum(axis=1)
print(classes_per_fam.value_counts().sort_index().to_string())

fam_sum = full.groupby("family")[LABEL_COLS].sum()


print("Habitat concentration analysis")
print("Top families driving each class + cumulative share\n")

meta_rows = []

for col, name in zip(LABEL_COLS, class_names):
    total = fam_sum[col].sum()
    if total == 0:
        continue
    shares = (fam_sum[col] / total * 100).sort_values(ascending=False)
    shares = shares[shares > 0]

    cumshare = shares.cumsum()
    top1  = shares.iloc[0]
    top3  = cumshare.iloc[min(2, len(cumshare)-1)]
    top10 = cumshare.iloc[min(9, len(cumshare)-1)]
    n_fam = len(shares)

    # families needed to cover 50% and 80%
    n50 = (cumshare <= 50).sum() + 1
    n80 = (cumshare <= 80).sum() + 1
    n50 = min(n50, n_fam)
    n80 = min(n80, n_fam)

    # global frequency: fraction of images where this habitat is present
    global_freq = (full[col] > 0).mean()

    meta_rows.append({
        "habitat":          name,
        "label_col":        col,
        "n_families_any":   n_fam,
        "n_families_for_50": n50,
        "n_families_for_80": n80,
        "global_frequency": global_freq,
    })

    print(f"{'─'*65}")
    print(f"{name}  ({n_fam} families present)")
    print(f"  Top-1 family  : {top1:.1f}%  |  Top-3: {top3:.1f}%  |  Top-10: {top10:.1f}%")
    print(f"  Families needed to cover 50%: {n50}   |   80%: {n80}")
    print(f"  Global frequency: {global_freq:.3f}")
    print(f"  Top 5 families:")
    for fam, share in shares.head(5).items():
        print(f"    {fam:30s}  {share:5.1f}%")

df_meta = pd.DataFrame(meta_rows)
df_meta.to_csv("data/meta/family_habitat_stats.csv", index=False)
print(f"\nSaved habitat stats to data/meta/family_habitat_stats.csv")