import argparse
import numpy as np
import pandas as pd
from pathlib import Path


VIEWS_DEFAULT = ["Back", "Belly", "Side"]
TARGET_TEMPLATE = "V0_NEW_Segmented-Aves-{view}-224-RGBA"


def load_embeddings(meta_dir: Path, view: str):
    target = TARGET_TEMPLATE.format(view=view)
    emb = np.load(meta_dir / f"{target}_embeddings.npy")          # (N, D)
    names = pd.read_csv(meta_dir / f"{target}_names.csv")["name"].tolist()
    return emb, names


def family_centroids(embeddings, names, img_to_family):
    families = np.array([img_to_family.get(n) for n in names])
    valid = np.array([f is not None for f in families], dtype=bool)
    embeddings = embeddings[valid]
    families   = families[valid]

    centroids = {}
    counts    = {}
    for fam in np.unique(families):
        mask = families == fam
        c = embeddings[mask].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids[fam] = c
        counts[fam]    = int(mask.sum())
    return centroids, counts


def compute_shift(val_centroids, val_counts, train_centroids):
    train_fams = np.array(list(train_centroids.keys()))
    train_mat  = np.stack([train_centroids[f] for f in train_fams])  # (T, D)
    train_global = train_mat.mean(axis=0)
    train_global = train_global / (np.linalg.norm(train_global) + 1e-8)

    rows = []
    for vfam, vc in val_centroids.items():
        sims_to_train   = train_mat @ vc                  # (T,)
        dists_to_train  = 1 - sims_to_train
        nearest_idx     = int(np.argmin(dists_to_train))

        rows.append({
            "val_family":               vfam,
            "n_val":                    val_counts[vfam],
            "nearest_train_family":     train_fams[nearest_idx],
            "min_dist_to_train":        float(dists_to_train[nearest_idx]),
            "mean_dist_to_train":       float(dists_to_train.mean()),
            "dist_to_train_global":     float(1 - float(vc @ train_global)),
        })

    return pd.DataFrame(rows).sort_values("min_dist_to_train", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir",  default="data/cross/cross_version_1_FAM")
    parser.add_argument("--meta-dir",   default="data/meta")
    parser.add_argument("--views",      nargs="+", default=VIEWS_DEFAULT)
    parser.add_argument("--folds",      type=int,  default=3)
    parser.add_argument("--output-dir", default="data/meta")
    args = parser.parse_args()

    cross_dir  = Path(args.cross_dir)
    meta_dir   = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for fold in range(args.folds):
        train_csv = cross_dir / f"train_fold_{fold}.csv"
        val_csv   = cross_dir / f"valid_fold_{fold}.csv"
        if not train_csv.exists() or not val_csv.exists():
            print(f"  fold {fold}: missing CSVs, skipping")
            continue

        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)

        train_lookup = dict(zip(df_train["name_of_img"], df_train["family"]))
        val_lookup   = dict(zip(df_val["name_of_img"],   df_val["family"]))

        print(f"\nFold {fold}: {len(df_train['family'].unique())} train families, "
              f"{len(df_val['family'].unique())} val families")

        for view in args.views:
            print(f"  view {view} ...", end=" ", flush=True)
            emb, names = load_embeddings(meta_dir, view)

            train_centroids, _           = family_centroids(emb, names, train_lookup)
            val_centroids,   val_counts  = family_centroids(emb, names, val_lookup)

            df = compute_shift(val_centroids, val_counts, train_centroids)
            df["view"] = view
            df["fold"] = fold

            out = output_dir / f"family_centroid_shift_fold{fold}_{view}.csv"
            df.to_csv(out, index=False)
            print(f"saved {out.name}  ({len(df)} families)")

            all_rows.append(df)

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        agg = (
            df_all
            .groupby("val_family")[["min_dist_to_train", "mean_dist_to_train",
                                     "dist_to_train_global", "n_val"]]
            .mean()
            .reset_index()
            .sort_values("min_dist_to_train", ascending=False)
        )
        out_agg = output_dir / "family_centroid_shift_aggregated.csv"
        agg.to_csv(out_agg, index=False)
        print(f"\nAggregated (mean over views & folds): {out_agg}")


if __name__ == "__main__":
    main()
