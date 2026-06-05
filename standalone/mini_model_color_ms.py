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




NAME = "mini_model_couleur_histogram"
NUM_CLASSES = 10
BATCH_SIZE  = 32*4
EPOCHS      = 30
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "runs/runs_mini_model_couleur_histogram_0"
suffixes = ["Back", "Side", "Belly"]

def startup_dir(name):
    i = 0
    while os.path.isdir(os.path.join(SAVE_DIR, f"{name}_multi_{i}")):
        i += 1
    destination_dir = os.path.join(SAVE_DIR, f"{name}_multi_{i}")
    os.makedirs(destination_dir)
    return destination_dir


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
        feat_dim = 6
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim*3, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(x)

def img_mean_std(path):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    mask = arr[:, :, 3] > 0
    pixels = arr[mask].astype(np.float32) / 255.0
    pixels = pixels[:, :3]
    mean = pixels.mean(axis=0)  
    std = pixels.std(axis=0)
    mstd = np.concatenate([mean, std])
    return mstd

class PathsAndLabels(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, root: Optional[str] = None):
        assert len(paths) == len(labels), "paths and labels must match"
        self.paths = list(paths) 
        self.labels = list(labels) 
        self.root = root
        self.images = [None] * len(self.paths)

        for i in tqdm(range(len(self.paths))):
            p = self.paths[i]
            full_p = os.path.join(self.root, p)

            if "Side" in p:
                full_p = full_p.replace("Back", "Side")
            elif "Belly" in p:
                full_p = full_p.replace("Back", "Belly")

            self.images[i] = img_mean_std(full_p)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        label = (self.labels[idx] > 0).astype(np.float32)
        return self.images[idx], label, p 
    


def train_simple(
    train_paths, train_labels,
    valid_paths, valid_labels,
    dir_path,
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
    
    save_dir = os.path.join(dir_path, "val_results.csv")
    prob_df     = pd.DataFrame(all_probs,    columns=[f"prob_{i}"     for i in range(all_probs.shape[1])])
    presence_df = pd.DataFrame(all_presence, columns=[f"presence_{i}" for i in range(all_presence.shape[1])])
    df = pd.concat([pd.Series(all_name, name="name"), prob_df, presence_df], axis=1)

    return model, hist



if __name__ == "__main__":
    dir_path    = startup_dir(NAME)
    resume_path = None


    data_root = "data/cross/cross_version_1_FAM"

    train_paths, train_labels = import_data(os.path.join(data_root, "train_fold_0.csv"))
    valid_paths, valid_labels = import_data(os.path.join(data_root, "valid_fold_0.csv"))
    
    with open(os.path.join(data_root, "labeltoname.json"), "r") as f:
        CLASS_NAMES = json.load(f)

    print(np.shape(train_paths), np.shape(valid_paths))

    SOURCE_IMG = os.environ["SOURCE_IMG"]

    model, hist = train_simple(
        train_paths, train_labels,
        valid_paths, valid_labels,
        dir_path,
        CLASS_NAMES,
        root= os.path.join(SOURCE_IMG, "V0_NEW_Segmented-Aves-Back-224-RGBA"),
        
    )

