import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torchvision import transforms as T
from tqdm import tqdm
import statsmodels.formula.api as smf
from dotenv import load_dotenv

load_dotenv()

VIEWS = ["Back", "Belly", "Side"]
TARGET_TEMPLATE = "V0_NEW_Segmented-Aves-{view}-224-RGBA"
DATASET_CSV = Path("data/datasets/dataset_version_1.csv")
BATCH_SIZE = 64
MIN_IMAGES = 15

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def load_backbone(device):
    backbone_cfg = "dinov3_vits16"
    local_path   = os.environ["dinov3_vits16"]
    dinov3_path = os.environ["DINOV3_PATH"]
    backbone = torch.hub.load(dinov3_path, backbone_cfg, source="local", weights = local_path )
    backbone.eval().to(device)
    return backbone


def extract_features(img_paths, backbone, device):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    feats, names = [], []
    for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="  extracting", leave=False):
        batch_paths = img_paths[i:i + BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
            names.append(p.name)
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            f = backbone(batch)
        feats.append(F.normalize(f, dim=-1).cpu().numpy())
    return np.vstack(feats), names


def pca_residuals(feats: np.ndarray, k: int) -> np.ndarray:
    pca = PCA(n_components=k)
    pca.fit(feats)
    V         = pca.components_           # (k, D)
    proj      = feats @ V.T @ V           # projection onto top-k subspace
    residuals = feats - proj
    norms     = np.linalg.norm(residuals, axis=1, keepdims=True)
    return residuals / (norms + 1e-8)

def sanity_check_features(feats, names, img_paths_by_name, n_queries=3):
    idx = np.random.choice(len(feats), n_queries, replace=False)
    fig, axes = plt.subplots(n_queries, 6, figsize=(18, n_queries * 3))
    
    for row, q_idx in enumerate(idx):
        sims  = feats @ feats[q_idx]           # cosine sim (already normalized)
        top5  = np.argsort(sims)[::-1][1:6]    # skip self
        
        axes[row, 0].imshow(Image.open(img_paths_by_name[names[q_idx]]))
        axes[row, 0].set_title("query", fontsize=8)
        axes[row, 0].axis("off")
        
        for col, nn_idx in enumerate(top5):
            axes[row, col+1].imshow(Image.open(img_paths_by_name[names[nn_idx]]))
            axes[row, col+1].set_title(f"sim={sims[nn_idx]:.3f}", fontsize=8)
            axes[row, col+1].axis("off")
    
    plt.suptitle("Nearest neighbors — raw features")
    plt.tight_layout()
    plt.savefig("sanity_raw_features.png", dpi=150)
    plt.close()

def sanity_check_residuals(feats_raw, feats_residual, names, img_paths_by_name, n_queries=3):
    idx = np.random.choice(len(feats_raw), n_queries, replace=False)
    fig, axes = plt.subplots(n_queries, 7, figsize=(21, n_queries * 3))

    for row, q_idx in enumerate(idx):
        # query image
        axes[row, 0].imshow(Image.open(img_paths_by_name[names[q_idx]]))
        axes[row, 0].set_title("query", fontsize=8)
        axes[row, 0].axis("off")

        # top 3 by raw features
        sims_raw = feats_raw @ feats_raw[q_idx]
        top3_raw = np.argsort(sims_raw)[::-1][1:4]
        for col, nn_idx in enumerate(top3_raw):
            axes[row, col+1].imshow(Image.open(img_paths_by_name[names[nn_idx]]))
            axes[row, col+1].set_title(f"raw {sims_raw[nn_idx]:.3f}", fontsize=8)
            axes[row, col+1].axis("off")

        # top 3 by residuals
        sims_res = feats_residual @ feats_residual[q_idx]
        top3_res = np.argsort(sims_res)[::-1][1:4]
        for col, nn_idx in enumerate(top3_res):
            axes[row, col+4].imshow(Image.open(img_paths_by_name[names[nn_idx]]))
            axes[row, col+4].set_title(f"residual {sims_res[nn_idx]:.3f}", fontsize=8)
            axes[row, col+4].axis("off")

    plt.suptitle("Left: raw neighbors | Right: residual neighbors")
    plt.tight_layout()
    plt.savefig("sanity_residuals_vs_raw.png", dpi=150)
    plt.close()

def visualize_pca_components(feats, names, img_paths_by_name, k=4):
    pca = PCA(n_components=k)
    coords = pca.fit_transform(feats)   # (N, k)

    fig, axes = plt.subplots(k, 10, figsize=(30, k * 3))
    for comp in range(k):
        order    = np.argsort(coords[:, comp])
        extremes = np.concatenate([order[:5], order[-5:]])   # 5 lowest + 5 highest
        for col, idx in enumerate(extremes):
            axes[comp, col].imshow(Image.open(img_paths_by_name[names[idx]]))
            axes[comp, col].set_title(f"{coords[idx, comp]:.2f}", fontsize=7)
            axes[comp, col].axis("off")
        axes[comp, 0].set_ylabel(f"PC{comp+1}", fontsize=10)

    plt.suptitle("Images with lowest (left) and highest (right) values per PCA component")
    plt.tight_layout()
    plt.savefig("pca_components_visualization.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading backbone (dinov3S) ...")
    backbone = load_backbone( device)

    df_meta      = pd.read_csv(DATASET_CSV)
    img_to_family = dict(zip(df_meta["name_of_img"], df_meta["family"]))

    source_img = Path(os.environ["SOURCE_IMG"])
    all_results = []

    for view in VIEWS:
        print(f"\n── View: {view}")
        img_dir   = source_img / TARGET_TEMPLATE.format(view=view)
        img_paths = sorted(p for p in img_dir.iterdir() if p.suffix == ".png")
        print(f"  {len(img_paths)} images")

        feats, names = extract_features(img_paths, backbone, device)
        residuals    = pca_residuals(feats, args.k)
        img_paths_by_name = {p.name: p for p in img_paths}
        sanity_check_residuals(feats, residuals, names, img_paths_by_name)
        visualize_pca_components(feats, names, img_paths_by_name, k=args.k)

    



if __name__ == "__main__":
    main()
