from dotenv import load_dotenv
import os 
from os import listdir
from os.path import join, exists
from dreamsim import dreamsim
from PIL import Image
import torch
from tqdm import tqdm 
import shutil 
import numpy as np 
import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
load_dotenv()


def embed_images(image_paths, desc="embedding"):
    embeddings = []
    names      = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=desc):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        imgs        = []

        for p in batch_paths:
            img = preprocess(Image.open(p).convert("RGB")).to(device)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            imgs.append(img)
            names.append(p.name)

        batch = torch.cat(imgs, dim=0)
        with torch.no_grad():
            emb = model.embed(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()   # (N, D)
    return embeddings, names


def compute_family_distances(embeddings, names, img_to_family):
    # map each image to its family
    families = np.array([img_to_family.get(n, None) for n in names])

    valid_mask  = np.array([f is not None for f in families])
    embeddings  = embeddings[valid_mask]
    families    = families[valid_mask]
    names_valid = [n for n, v in zip(names, valid_mask) if v]

    # global centroid
    global_centroid = embeddings.mean(axis=0)
    global_centroid = global_centroid / (np.linalg.norm(global_centroid) + 1e-8)

    unique_families = sorted(set(families))
    rows = []

    for fam in tqdm(unique_families, desc="computing family distances"):
        mask      = families == fam
        fam_embs  = embeddings[mask]           # (n_fam, D)
        n         = len(fam_embs)

        if n < 2:
            continue

        # family centroid
        centroid = fam_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # intra: mean distance of each image to family centroid
        sims_intra = fam_embs @ centroid                  # (n_fam,)
        intra_dist = float(np.mean(1 - sims_intra))

        # inter: distance of family centroid to global centroid
        sim_inter  = centroid @ global_centroid
        inter_dist = float(1 - sim_inter)

        intra_std  = float(np.std(1 - sims_intra))

        rows.append({
            "family":     fam,
            "n_images":   n,
            "intra_dist": intra_dist,
            "intra_std":  intra_std,
            "inter_dist": inter_dist,
        })

    return pd.DataFrame(rows)


def add_family_metadata(df_dist, df_meta):
    n_species = df_meta.groupby("family")["species"].nunique().reset_index()
    n_species.columns = ["family", "n_species"]

    n_genre = df_meta.groupby("family")["genre"].nunique().reset_index()
    n_genre.columns = ["family", "n_genre"]

    prop_male = df_meta.groupby("family").apply(
        lambda x: (x["sexe"] == "M").mean()
    ).reset_index()
    prop_male.columns = ["family", "prop_male"]

    df_dist = df_dist.merge(n_species,  on="family", how="left")
    df_dist = df_dist.merge(n_genre,    on="family", how="left")
    df_dist = df_dist.merge(prop_male,  on="family", how="left")

    return df_dist




if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True, device=device)
    model.eval()
    
    SOURCE_IMG = Path(os.environ["SOURCE_IMG"])
    VIEWS = ["Back", "Belly", "Side"]
    BATCH_SIZE  = 128
    METADATA    = Path("data/datasets/dataset_version_1.csv")
    OUTPUT_DIR = Path("data/meta/") 
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_meta = pd.read_csv(METADATA)
    df_meta = df_meta[["name_of_img", "family", "genre", "species", "sexe"]].copy()

    # build lookup: image name → family
    img_to_family = dict(zip(df_meta["name_of_img"], df_meta["family"]))
    img_to_meta   = df_meta.set_index("name_of_img").to_dict(orient="index")

    for view in VIEWS:
        print(f"\n{'='*60}\nView: {view}\n{'='*60}")

        target     = f"V0_NEW_Segmented-Aves-{view}-224-RGBA"
        images_dir = SOURCE_IMG / target
        img_paths  = sorted([
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix == ".png"
        ])
        print(f"  {len(img_paths)} images found")

        embeddings, names = embed_images(img_paths, desc=f"embedding {view}")

        np.save(OUTPUT_DIR / f"{target}_embeddings.npy", embeddings)
        pd.Series(names).to_csv(
            OUTPUT_DIR / f"{target}_names.csv", index=False, header=["name"]
        )
        print(f"  embeddings saved: {embeddings.shape}")

        df_dist = compute_family_distances(embeddings, names, img_to_family)
        df_dist = add_family_metadata(df_dist, df_meta)
        df_dist["view"] = view

        out_path = OUTPUT_DIR / f"family_distances_{view}.csv"
        df_dist.to_csv(out_path, index=False)
        print(f"  saved: {out_path}")