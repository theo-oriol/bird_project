import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ #
#  Data loading                                                        #
# ------------------------------------------------------------------ #

def load_csv(path):
    df = pd.read_csv(path)
    X  = df["name_of_img"].values
    y  = df[LABEL_COLS].values.astype(np.float32) / 100.0
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    y = y / row_sums
    return X, y


# ------------------------------------------------------------------ #
#  Crop sampling: anywhere in the masked region, >=70% bird coverage   #
# ------------------------------------------------------------------ #

def compute_coverage_map(mask, crop_size):
    """
    mask: (H, W) boolean array
    Returns: (H - crop_size + 1, W - crop_size + 1) float array
             coverage[y0, x0] = fraction of mask pixels inside the
             crop_size x crop_size window with top-left corner (y0, x0)
    """
    H, W = mask.shape
    mask_f = mask.astype(np.float64)

    # integral image, padded with a row/col of zeros for easy indexing
    integral = np.zeros((H + 1, W + 1), dtype=np.float64)
    integral[1:, 1:] = mask_f.cumsum(axis=0).cumsum(axis=1)

    out_h = H - crop_size + 1
    out_w = W - crop_size + 1

    # sum inside window [y0:y0+crop_size, x0:x0+crop_size] via inclusion-exclusion
    A = integral[0:out_h,            0:out_w]
    B = integral[0:out_h,            crop_size:crop_size + out_w]
    C = integral[crop_size:crop_size + out_h, 0:out_w]
    D = integral[crop_size:crop_size + out_h, crop_size:crop_size + out_w]

    window_sum = D - B - C + A
    coverage   = window_sum / (crop_size * crop_size)
    return coverage


def sample_valid_crop(mask, crop_size, min_coverage=1):
    """
    Returns (y0, x0) top-left corner of a crop_size x crop_size window
    with at least min_coverage fraction of pixels inside the mask,
    sampled uniformly among all valid positions.
    Falls back to the single best-coverage position if none meet the
    threshold (e.g. very thin/sparse birds at this crop size).
    """
    H, W = mask.shape
    if crop_size >= H or crop_size >= W:
        return 0, 0

    coverage = compute_coverage_map(mask, crop_size)   # (out_h, out_w)
    valid_ys, valid_xs = np.where(coverage >= min_coverage)

    if len(valid_ys) > 0:
        idx = np.random.randint(len(valid_ys))
        return int(valid_ys[idx]), int(valid_xs[idx])

    # fallback: no position satisfies the threshold, take the global best
    flat_idx = np.argmax(coverage)
    y0, x0 = np.unravel_index(flat_idx, coverage.shape)
    return int(y0), int(x0) # best effort if threshold never reached


# ------------------------------------------------------------------ #
#  Dataset — returns K crops per bird (a "bag")                        #
# ------------------------------------------------------------------ #

class CropBagDataset(Dataset):
    def __init__(self, paths, labels, img_dir, crop_size=96, n_crops=8,
                 min_coverage=0.70, is_train=True, binarize=True):
        self.paths        = paths
        self.labels       = labels
        self.img_dir      = img_dir
        self.crop_size    = crop_size
        self.n_crops      = n_crops
        self.min_coverage = min_coverage
        self.is_train     = is_train
        self.binarize     = binarize

        aug = []
        if is_train:
            aug = [T.RandomVerticalFlip(), T.RandomHorizontalFlip()]
        self.transform = T.Compose([
            *aug,
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.paths[idx])
        if "Belly" in img_path:
            img_path = img_path.replace("Back","Belly")
        elif "Side" in img_path:
            img_path = img_path.replace("Back","Side")
            
        rgba     = np.array(Image.open(img_path).convert("RGBA"))
        mask     = rgba[:, :, 3] > 0
        rgb_img  = Image.fromarray(rgba[:, :, :3], mode="RGB")

        crops = []
        for _ in range(self.n_crops):
            y0, x0 = sample_valid_crop(mask, self.crop_size, self.min_coverage)
            crop = rgb_img.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))
            crops.append(self.transform(crop))
        crops = torch.stack(crops)   # (K, C, crop_size, crop_size)

        label = self.labels[idx].copy()
        if self.binarize:
            label = (label > 0).astype(np.float32)

        return crops, label, self.paths[idx]


def build_dataloaders(cfg):
    fold     = cfg["data"]["fold"]
    data_dir = cfg["data"]["dataset_dir"]
    img_dir  = cfg["data"]["img_dir"]

    train_paths, train_labels = load_csv(os.path.join(data_dir, f"train_fold_{fold}.csv"))
    valid_paths, valid_labels = load_csv(os.path.join(data_dir, f"valid_fold_{fold}.csv"))

    ds_train = CropBagDataset(
        train_paths, train_labels, img_dir,
        crop_size=cfg["model"]["crop_size"],
        n_crops=cfg["model"]["n_crops"],
        min_coverage=cfg["model"]["min_coverage"],
        is_train=True,
        binarize=cfg["data"].get("binarize", True),
    )
    ds_val = CropBagDataset(
        valid_paths, valid_labels, img_dir,
        crop_size=cfg["model"]["crop_size"],
        n_crops=cfg["model"]["n_crops"],
        min_coverage=cfg["model"]["min_coverage"],
        is_train=False,
        binarize=cfg["data"].get("binarize", True),
    )

    train_loader = DataLoader(
        ds_train, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["training"]["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader, train_labels


# ------------------------------------------------------------------ #
#  Model: DINOv3-small backbone, per-crop head, max-pool over bag       #
# ------------------------------------------------------------------ #

class MILCropModel(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )
        # learned attention score per crop, per class
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, num_classes),   # one attention logit per class
        )

    def forward(self, crops):
        B, K, C, H, W = crops.shape
        flat  = crops.view(B * K, C, H, W)
        feats = self.backbone(flat)                        # (B*K, feat_dim)

        crop_logits = self.head(feats).view(B, K, -1)        # (B, K, num_classes)
        attn_logits = self.attention(feats).view(B, K, -1)   # (B, K, num_classes)
        attn_weights = torch.softmax(attn_logits, dim=1)     # softmax over crops, per class

        bag_logits = (crop_logits * attn_weights).sum(dim=1)  # (B, num_classes)
        return bag_logits, crop_logits #, attn_weights


def build_backbone(cfg):
    local_path = cfg["model"]["backbone"]["local_path"]
    name       = cfg["model"]["backbone"]["name"]
    weights    = cfg["model"]["backbone"]["weights"]
    backbone = torch.hub.load(local_path, name, source="local", weights=weights)
    return backbone


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def safe_auc(labels, probs):
    n = labels.shape[1]
    try:
        return roc_auc_score(labels, probs, average=None)
    except ValueError:
        return np.array([
            roc_auc_score(labels[:, i], probs[:, i])
            if len(np.unique(labels[:, i])) > 1 else float("nan")
            for i in range(n)
        ])


def train(cfg, train_loader, val_loader, run_dir):
    device  = cfg["device"]
    backbone = build_backbone(cfg).to(device)
    model = MILCropModel(
        backbone,
        feat_dim=cfg["model"]["feat_dim"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                                   weight_decay=float(cfg["training"].get("weight_decay", 1e-2)))
    scheduler = CosineAnnealingLR(optimizer, T_max=float(cfg["training"]["epochs"]), eta_min=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler()

    empty_hist = {"epoch": [], "train_loss": [], "val_loss": [],
                  "train_ap": [], "val_ap": [], "train_auc": [], "val_auc": []}
    start_epoch, hist, best_val_auc = load_checkpoint(run_dir, model, optimizer, scheduler, scaler)
    if not hist:
        hist = empty_hist

    for epoch in range(start_epoch + 1, cfg["training"]["epochs"] + 1):
        model.train()
        tr_loss, all_probs, all_labels = 0.0, [], []

        for crops, labels, _ in tqdm(train_loader, desc=f"[{epoch:02d}] train", leave=False):
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                bag_logits, _ = model(crops)
                loss = criterion(bag_logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()
            all_probs.append(torch.sigmoid(bag_logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        scheduler.step()
        tr_loss /= max(1, len(train_loader))
        probs_np  = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)
        train_ap  = average_precision_score(labels_np, probs_np, average=None)
        train_auc = safe_auc(labels_np, probs_np)

        model.eval()
        va_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for crops, labels, _ in tqdm(val_loader, desc=f"[{epoch:02d}] val", leave=False):
                crops, labels = crops.to(device), labels.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    bag_logits, _ = model(crops)
                    loss = criterion(bag_logits, labels)
                va_loss += loss.item()
                all_probs.append(torch.sigmoid(bag_logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        va_loss /= max(1, len(val_loader))
        probs_np  = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)
        val_ap    = average_precision_score(labels_np, probs_np, average=None)
        val_auc   = safe_auc(labels_np, probs_np)

        cur_val_auc = float(np.nanmean(val_auc))
        is_best     = cur_val_auc > best_val_auc
        if is_best:
            best_val_auc = cur_val_auc

        print(f"[{epoch:02d}] train loss {tr_loss:.4f} mAP {np.nanmean(train_ap):.4f} AUC {np.nanmean(train_auc):.4f} | "
              f"val loss {va_loss:.4f} mAP {np.nanmean(val_ap):.4f} AUC {cur_val_auc:.4f}"
              + (" *" if is_best else ""))

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_ap"].append(train_ap.tolist())
        hist["val_ap"].append(val_ap.tolist())
        hist["train_auc"].append(train_auc.tolist())
        hist["val_auc"].append(val_auc.tolist())

        save_checkpoint(model, optimizer, scheduler, scaler, hist, cfg, run_dir,
                        epoch, best_val_auc, is_best=is_best)

    return model, hist


# ------------------------------------------------------------------ #
#  Checkpoint                                                          #
# ------------------------------------------------------------------ #

def save_checkpoint(model, optimizer, scheduler, scaler, hist, cfg, run_dir,
                    epoch, best_val_auc, is_best=False):
    state = {
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "scaler":       scaler.state_dict(),
        "hist":         hist,
        "cfg":          cfg,
        "best_val_auc": best_val_auc,
    }
    last_path = run_dir / "checkpoint_last.pth"
    torch.save(state, last_path)
    if is_best:
        best_path = run_dir / "checkpoint_best.pth"
        torch.save(state, best_path)
        print(f"[checkpoint] best saved (val_auc={best_val_auc:.4f}) → {best_path}")


def load_checkpoint(run_dir, model, optimizer, scheduler, scaler):
    path = run_dir / "checkpoint_last.pth"
    if not path.exists():
        return 0, {}, 0.0
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"[checkpoint] resumed from epoch {ckpt['epoch']} (best val_auc={ckpt['best_val_auc']:.4f})")
    return ckpt["epoch"], ckpt["hist"], ckpt["best_val_auc"]


# ------------------------------------------------------------------ #
#  Plots                                                               #
# ------------------------------------------------------------------ #

def save_plots(hist, run_dir):
    epochs = hist["epoch"]

    fig, ax = plt.subplots()
    ax.plot(epochs, hist["train_loss"], label="train")
    ax.plot(epochs, hist["val_loss"],   label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.savefig(run_dir / "loss.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, [np.nanmean(v) for v in hist["train_auc"]], label="train")
    ax.plot(epochs, [np.nanmean(v) for v in hist["val_auc"]],   label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean AUC")
    ax.legend()
    fig.savefig(run_dir / "val_auc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, [np.nanmean(v) for v in hist["train_ap"]], label="train")
    ax.plot(epochs, [np.nanmean(v) for v in hist["val_ap"]],   label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean AP")
    ax.legend()
    fig.savefig(run_dir / "val_ap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots] saved loss.png, val_auc.png, val_ap.png to {run_dir}")


# ------------------------------------------------------------------ #
#  Final inference: save bag-level + per-crop scores for inspection    #
# ------------------------------------------------------------------ #

def run_inference(model, val_loader, device, run_dir, class_names):
    model.eval()
    rows = []
    with torch.no_grad():
        for crops, labels, names in tqdm(val_loader, desc="final inference"):
            crops = crops.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                bag_logits, crop_logits = model(crops)
            bag_probs  = torch.sigmoid(bag_logits).cpu().numpy()    # (B, C)
            crop_probs = torch.sigmoid(crop_logits).cpu().numpy()   # (B, K, C)
            labels_np  = labels.numpy()

            for i, name in enumerate(names):
                row = {"image": name}
                for c, cname in enumerate(class_names):
                    row[f"true_{cname}"] = labels_np[i, c]
                    row[f"bag_pred_{cname}"] = bag_probs[i, c]
                    for k in range(crop_probs.shape[1]):
                        row[f"crop{k}_pred_{cname}"] = crop_probs[i, k, c]
                rows.append(row)

    df_out = pd.DataFrame(rows)
    out_path = run_dir / "val_predictions_mil.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved per-crop + bag predictions to {out_path}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    with open(os.path.join(cfg["data"]["dataset_dir"], "labeltoname.json")) as f:
        class_names = json.load(f)

    run_dir = Path("standalone/experiments") / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model, hist = train(cfg, train_loader, val_loader, run_dir)

    save_plots(hist, run_dir)

    with open(run_dir / "hist.json", "w") as f:
        json.dump(hist, f, indent=2)

    run_inference(model, val_loader, cfg["device"], run_dir, class_names)