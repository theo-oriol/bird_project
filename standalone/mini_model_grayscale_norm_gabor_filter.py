import cv2

from dotenv import load_dotenv

import os
from time import time
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF

load_dotenv()


ORIENTATIONS = 8
FREQUENCIES  = [0.1, 0.2, 0.3, 0.4, 0.5]
KSIZE        = 31
SIGMA        = 4.0
GAMMA        = 0.5

NUM_CLASSES = 10
BATCH_SIZE  = 32*4
EPOCHS      = 30
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "runs/runs_mini_model_couleur_histogram_0"
suffixes = ["Back", "Side", "Belly"]


def _normalize_rows(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sums = a.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return (a / row_sums).astype(np.float32)


def import_data(path):
    LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]
    df = pd.read_csv(path)
    X  = df["name_of_img"].values
    y  = df[LABEL_COLS].values.astype(np.float32) / 100.0
    y  = _normalize_rows(y)
    return X, y



class Minimodel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        feat_dim = 80
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim*3, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(x)

def rgb_to_ggg(img) -> np.ndarray:
    img = np.array(img) 
    alpha = img[:, :, 3]
    mask = img[:,:, 3]>0
    g_channel = img[:, :, 1]
    arr = g_channel[mask]
    return arr, mask, g_channel, alpha


def lum_match(arrays,M,S):
    target_mean = M
    target_std  = S


    m, s = arrays.mean(), arrays.std()
    if s < 1e-8:
        results = np.full_like(arrays, target_mean)
    else:
        Z = (arrays - m) / s
        E = Z * target_std + target_mean
        results = np.clip(E, 0, 255)
    return results

def _exact_hist_match(source: np.ndarray,
                      target_hist: np.ndarray) -> np.ndarray:
    flat = source.ravel().astype(np.float64)
    N = flat.size
    if N == 0:
        return source.copy()

    noise = np.random.uniform(0, 1e-9, size=N)
    sort_idx = np.argsort(flat + noise, kind='stable')

    # Scale target_hist to sum exactly to N so every slot gets assigned
    total = target_hist.sum()
    counts = np.round(target_hist / total * N).astype(int)
    diff = N - counts.sum()
    if diff != 0:
        counts[np.argmax(counts)] += diff

    new_values = np.empty(N, dtype=np.float64)
    pos = 0
    for value, count in enumerate(counts):
        new_values[sort_idx[pos:pos + count]] = value
        pos += count

    return new_values.reshape(source.shape)

def hist_match(arrays, target_hist) -> list[np.ndarray]:
    matched = _exact_hist_match(arrays, target_hist)
    results= np.clip(matched, 0, 255)
    return results

def lum_norm(img, M, S, target_hist):
        arrays,mask,g_chanel,alpha = rgb_to_ggg(img)
        arrays = lum_match(arrays,M,S)
        arrays = hist_match(arrays, target_hist)
        g_chanel[~mask]=0
        g_chanel[mask] = arrays
        
        out_uint8 = g_chanel.astype(np.uint8)
        out_rgb = np.stack([out_uint8, out_uint8, out_uint8,alpha], axis=-1)
        return out_rgb


def build_gabor_kernels():
    kernels = []
    for freq in FREQUENCIES:
        for i in range(ORIENTATIONS):
            theta  = i * np.pi / ORIENTATIONS
            kernel = cv2.getGaborKernel(
                (KSIZE, KSIZE),
                SIGMA,
                theta,
                1.0 / freq,
                GAMMA,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernels.append(kernel)
    return kernels


GABOR_KERNELS = build_gabor_kernels()

def img_grayscale_norm_gabor(img, target_hist, M, S):
    arr = lum_norm(img, M, S, target_hist)
    mask = arr[:, :, 3] > 0
    gray2d = arr[:, :, 0].astype(np.float32) / 255.0   # (H, W) — full spatial image

    features = []
    for kernel in GABOR_KERNELS:
        response        = cv2.filter2D(gray2d, cv2.CV_32F, kernel)
        masked_response = response[mask]   # only stats on foreground pixels
        features.append(masked_response.mean())
        features.append(masked_response.std())
    return np.array(features, dtype=np.float32) 



class PathsAndLabels(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, root: Optional[str] = None):
        assert len(paths) == len(labels), "paths and labels must match"
        self.paths = list(paths) 
        self.labels = list(labels) 
        self.root = root
        self.images = [None] * len(self.paths)


        with open(f'configs/meta_data/grayscale/V0_NEW_Segmented-Aves-Back-224-RGBA/meta_data_grayscale_Back.json', 'r') as f:
            parameters = json.load(f)
        self.target_hist_back = np.load(f"configs/meta_data/grayscale/V0_NEW_Segmented-Aves-Back-224-RGBA/target_hist_grayscale_Back.npy")
        self.M_back = parameters["M"]
        self.S_back = parameters["S"]

        with open(f'configs/meta_data/grayscale/V0_NEW_Segmented-Aves-Side-224-RGBA/meta_data_grayscale_Side.json', 'r') as f:
            parameters = json.load(f)
        self.target_hist_side = np.load(f"configs/meta_data/grayscale/V0_NEW_Segmented-Aves-Side-224-RGBA/target_hist_grayscale_Side.npy")
        self.M_side = parameters["M"]
        self.S_side = parameters["S"]

        with open(f'configs/meta_data/grayscale/V0_NEW_Segmented-Aves-Belly-224-RGBA/meta_data_grayscale_Belly.json', 'r') as f:
            parameters = json.load(f)
        self.target_hist_belly = np.load(f"configs/meta_data/grayscale/V0_NEW_Segmented-Aves-Belly-224-RGBA/target_hist_grayscale_Belly.npy")
        self.M_belly = parameters["M"]
        self.S_belly = parameters["S"]




        for i in tqdm(range(len(self.paths))):
            p = self.paths[i]
            full_p = os.path.join(self.root, p)

            if "Side" in p:
                full_p = full_p.replace("Back", "Side")
            elif "Belly" in p:
                full_p = full_p.replace("Back", "Belly")

            target_hist = None
            if "Side" in p:
                target_hist = self.target_hist_side
                M = self.M_side
                S = self.S_side
            elif "Belly" in p:
                target_hist = self.target_hist_belly
                M = self.M_belly
                S = self.S_belly
            else:
                target_hist = self.target_hist_back
                M = self.M_back
                S = self.S_back

            self.images[i] = img_grayscale_norm_gabor(Image.open(full_p).convert("RGBA"), target_hist, M, S)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        label = (self.labels[idx] > 0).astype(np.float32)
        return self.images[idx], label, p 
    


def train_simple(
    train_paths, train_labels,
    valid_paths, valid_labels,
    CLASS_NAMES,
    root=None,
):
    model = Minimodel().to(DEVICE)

    ds_train = PathsAndLabels(train_paths, train_labels,  root=root)
    ds_val   = PathsAndLabels(valid_paths, valid_labels,  root=root)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    label_arr = np.array(ds_train.labels)
    N   = label_arr.shape[0]
    pos = (label_arr > 0).sum(axis=0)
    neg = N - pos
    pos_weight = torch.from_numpy(
        neg / np.clip(pos, a_min=1, a_max=None)
    ).float().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.amp.GradScaler()

    hist = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_auc_per_class": [], "val_auc_per_class": [],
        "train_ap_per_class": [], "val_ap_per_class": [],
    }

    start_epoch = 1
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        tr_loss_accum = 0.0
        all_probs, all_presence = [], []

        for imgs, presence, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            imgs     = imgs.to(DEVICE)
            presence = presence.to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                loss   = criterion(logits, presence)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            tr_loss_accum += loss.item()
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_presence.append(presence.detach().cpu())


        scheduler.step()

        tr_loss    = tr_loss_accum / max(1, len(train_loader))
        all_probs    = torch.cat(all_probs,    dim=0).numpy()
        all_presence = torch.cat(all_presence, dim=0).numpy()

        train_ap = average_precision_score(all_presence, all_probs, average=None)
        try:
            train_auc = roc_auc_score(all_presence, all_probs, average=None)
        except ValueError:
            train_auc = np.array([
                roc_auc_score(all_presence[:, i], all_probs[:, i])
                if len(np.unique(all_presence[:, i])) > 1 else float("nan")
                for i in range(NUM_CLASSES)
            ])

        

        model.eval()
        va_loss_accum = 0.0
        all_probs, all_presence, all_name = [], [], []

        with torch.no_grad():
            for imgs, presence, name in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                imgs     = imgs.to(DEVICE, non_blocking=True)
                presence = presence.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(imgs)
                    loss   = criterion(logits, presence)
                va_loss_accum += loss.item()
                all_probs.append(torch.sigmoid(logits).detach().cpu())
                all_presence.append(presence.detach().cpu())
                all_name.extend(name)

        va_loss      = va_loss_accum / max(1, len(val_loader))
        all_probs    = torch.cat(all_probs,    dim=0).numpy()
        all_presence = torch.cat(all_presence, dim=0).numpy()
        all_name     = np.array(all_name)

        val_ap = average_precision_score(all_presence, all_probs, average=None)
        try:
            val_auc = roc_auc_score(all_presence, all_probs, average=None)
        except ValueError:
            val_auc = np.array([
                roc_auc_score(all_presence[:, i], all_probs[:, i])
                if len(np.unique(all_presence[:, i])) > 1 else float("nan")
                for i in range(10)
            ])

        
        print(f"[Epoch {epoch:02d}] train loss {tr_loss:.4f} | AUC {np.nanmean(train_auc):.4f}", end =" ")
        print(f" valid loss {va_loss:.4f} | AUC {np.nanmean(val_auc):.4f} | {val_auc}")

        
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss)
        hist["train_ap_per_class"].append(train_ap)
        hist["train_auc_per_class"].append(train_auc)
        hist["val_loss"].append(va_loss)
        hist["val_ap_per_class"].append(val_ap)
        hist["val_auc_per_class"].append(val_auc)

    for auc_cls in range(hist["val_auc_per_class"][-1].shape[0]):
        print(f"Class {CLASS_NAMES[auc_cls]}: AP {hist['val_ap_per_class'][-1][auc_cls]:.4f} | AUC {hist['val_auc_per_class'][-1][auc_cls]:.4f}")
    
    return model, hist



if __name__ == "__main__":

    data_root = "data/cross/cross_version_1_FAM"
    # data_root = "data/cross/cross_version_1_GENRE"

    train_paths, train_labels = import_data(os.path.join(data_root, "train_fold_0.csv"))
    valid_paths, valid_labels = import_data(os.path.join(data_root, "valid_fold_0.csv"))
    
    with open(os.path.join(data_root, "labeltoname.json"), "r") as f:
        CLASS_NAMES = json.load(f)

    print(np.shape(train_paths), np.shape(valid_paths))

    SOURCE_IMG = os.environ["SOURCE_IMG"]

    model, hist = train_simple(
        train_paths, train_labels,
        valid_paths, valid_labels,
        CLASS_NAMES,
        root= os.path.join(SOURCE_IMG, "V0_NEW_Segmented-Aves-Back-224-RGBA"),
        
    )

