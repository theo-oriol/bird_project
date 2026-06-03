import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu

from src.models.factory      import build_model
from src.datasets.dataset    import BirdDataset, load_csv
from src.datasets.transforms import make_transforms


CONFIG_DIR = Path("configs")
CROSS_PATH = Path("data/cross")
SOURCE_IMG = os.environ["SOURCE_IMG"]

# ------------------------------------------------------------------ #
#  Load model from checkpoint                                          #
# ------------------------------------------------------------------ #

def load_model(run_dir, cfg):
    checkpoint_path = os.path.join("experiments", cfg["benchmark"]["name"], run_dir, "checkpoint_last.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    omega_cfg = OmegaConf.load(
        os.path.join("experiments", cfg["benchmark"]["name"], run_dir, "config.yaml")
    )
    omega_cfg.model.backbone.lora = None  # ignore LoRA config at eval time — we load the merged checkpoint
    model = build_model(omega_cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, omega_cfg


# ------------------------------------------------------------------ #
#  Build dataloader for a specific augmentation                        #
# ------------------------------------------------------------------ #

def build_aug_loader(omega_cfg, aug_name, dataset_dir, img_dir, fold):
    aug_cfg = OmegaConf.merge(
        omega_cfg,
        OmegaConf.create({"data": {"augmentation": aug_name}})
    )

    data_dir = os.path.join(CROSS_PATH, dataset_dir)        # path to the fold CSVs
    img_dir  = os.path.join(SOURCE_IMG ,img_dir) 

    valid_paths, valid_labels = load_csv(
        os.path.join(dataset_dir, f"valid_fold_{fold}.csv")
    )

    # infer view from img_dir name
    view = None
    for v in ("Back", "Side", "Belly"):
        if v in img_dir:
            view = v
            break

    ds_val = BirdDataset(
        valid_paths, valid_labels, img_dir,
        cfg=aug_cfg,
        transform=make_transforms,
        is_train=False,
        binarize=aug_cfg.data.get("binarize", False),
    )

    return DataLoader(
        ds_val,
        batch_size=omega_cfg.training.batch_size,
        shuffle=False,
        num_workers=omega_cfg.training.num_workers,
        pin_memory=True,
    )


# ------------------------------------------------------------------ #
#  Inference                                                           #
# ------------------------------------------------------------------ #

def run_inference(model, loader, device):
    model.to(device)
    model.eval()
    all_probs, all_labels, all_names = [], [], []

    with torch.no_grad():
        for imgs, labels, names in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
            all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_names.extend(names)

    return (
        np.concatenate(all_probs,  axis=0),
        np.concatenate(all_labels, axis=0),
        all_names,
    )


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

def compute_metrics(probs, labels, class_names):
    from sklearn.metrics import average_precision_score, roc_auc_score

    ap  = average_precision_score(labels, probs, average=None)
    try:
        auc = roc_auc_score(labels, probs, average=None)
    except ValueError:
        auc = np.array([
            roc_auc_score(labels[:, i], probs[:, i])
            if len(np.unique(labels[:, i])) > 1 else float("nan")
            for i in range(labels.shape[1])
        ])

    # mann-whitney p-values
    pvalues = {}
    all_pos, all_neg = [], []
    for i, c in enumerate(class_names):
        pos = probs[labels[:, i] > 0, i]
        neg = probs[labels[:, i] == 0, i]
        all_pos.append(pos)
        all_neg.append(neg)
        if len(pos) == 0 or len(neg) == 0:
            pvalues[c] = None
            continue
        _, p = mannwhitneyu(pos, neg, alternative="greater")
        pvalues[c] = float(p)

    pos_all = np.concatenate(all_pos)
    neg_all = np.concatenate(all_neg)
    _, p_overall = mannwhitneyu(pos_all, neg_all, alternative="greater")

    return {
        "val_mAP":                      float(np.nanmean(ap)),
        "val_auc":                       float(np.nanmean(auc)),
        "val_ap_per_class":             ap.tolist(),
        "val_auc_per_class":            auc.tolist(),
        "mannwhitney_pvalue_per_class": pvalues,
        "mannwhitney_pvalue_overall":   float(p_overall),
    }


# ------------------------------------------------------------------ #
#  Save predictions CSV                                                #
# ------------------------------------------------------------------ #

def save_predictions(probs, labels, names, class_names, out_path):
    rows = {"image": names}
    for i, c in enumerate(class_names):
        rows[f"true_{c}"] = labels[:, i]
        rows[f"pred_{c}"] = probs[:, i]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  saved: {out_path}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", help="path to benchmark YAML")
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    with open(args.benchmark) as f:
        cfg = yaml.safe_load(f)

    bench         = cfg["benchmark"]
    dataset_dir   = str(CROSS_PATH / bench["dataset_dir"])
    img_dir       = bench["img_folder"]
    device        = args.device

    # load class names
    with open(os.path.join(dataset_dir, "labelname.json")) as f:
        class_names = json.load(f)

    leaderboard_rows = []

    for exp_name, exp_cfg in cfg["experiments"].items():
        for fold_idx, fold_cfg in exp_cfg["folds"].items():
            if fold_cfg["status"] != "done":
                print(f"[skip] {exp_name} fold {fold_idx} — status: {fold_cfg['status']}")
                continue

            run_dir = fold_cfg["run_dir"]
            print(f"\n{'='*60}")
            print(f"  exp  : {exp_name}  fold : {fold_idx}")
            print(f"{'='*60}")

            try:
                model, omega_cfg = load_model(run_dir, cfg)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue

            
            aug_names = exp_cfg.model.data.get("additional_transforms", None)
            if aug_names is None:
                raise ValueError(f"No additional_transforms defined for experiment '{exp_name}' in config.yaml")
            aug_names.append("no_aug") 

            aug_results = {}
            for aug_name in aug_names:
                print(f"\n  augmentation: {aug_name}")

                loader = build_aug_loader(
                    omega_cfg, aug_name, dataset_dir, img_dir, fold_idx
                )
                probs, labels, names = run_inference(model, loader, device)
                metrics              = compute_metrics(probs, labels, class_names)
                aug_results[aug_name] = metrics

                # save predictions CSV
                out_dir = Path("experiments") / bench["name"] / run_dir
                save_predictions(
                    probs, labels, names, class_names,
                    out_dir / f"val_predictions_{aug_name}.csv"
                )

                # save per-aug metrics JSON
                with open(out_dir / f"metrics_{aug_name}.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                # leaderboard row
                row = {
                    "experiment": f"{exp_name}__{aug_name}",
                    "exp_name":   exp_name,
                    "aug_name":   aug_name,
                    "fold":       fold_idx,
                    "val_mAP":    metrics["val_mAP"],
                    "val_auc":    metrics["val_auc"],
                    "pval_overall": metrics["mannwhitney_pvalue_overall"],
                }
                for i, c in enumerate(class_names):
                    row[f"AP_{c}"]   = metrics["val_ap_per_class"][i]
                    row[f"AUC_{c}"]  = metrics["val_auc_per_class"][i]
                    row[f"pval_{c}"] = metrics["mannwhitney_pvalue_per_class"].get(c)

                leaderboard_rows.append(row)
                print(f"    mAP={metrics['val_mAP']:.4f}  AUC={metrics['val_auc']:.4f}  p={metrics['mannwhitney_pvalue_overall']:.4e}")

            # update main metrics.json with all aug results
            metrics_path = Path("experiments") / bench["name"] / run_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    existing = json.load(f)
            else:
                existing = {}
            existing["inference_augmentations"] = aug_results
            with open(metrics_path, "w") as f:
                json.dump(existing, f, indent=2)

    if not leaderboard_rows:
        print("No completed experiments found.")
        return

    # ---- build leaderboard ----
    df = (
        pd.DataFrame(leaderboard_rows)
        .sort_values(["exp_name", "aug_name", "fold"])
        .reset_index(drop=True)
    )

    # summary: mean/std across folds per exp+aug
    summary_rows = []
    for (exp_name, aug_name), group in df.groupby(["exp_name", "aug_name"]):
        row = {
            "experiment":       f"{exp_name}__{aug_name}",
            "exp_name":         exp_name,
            "aug_name":         aug_name,
            "folds_done":       len(group),
            "val_mAP_mean":     group["val_mAP"].mean(),
            "val_mAP_std":      group["val_mAP"].std(),
            "val_auc_mean":     group["val_auc"].mean(),
            "val_auc_std":      group["val_auc"].std(),
            "pval_overall_mean": group["pval_overall"].mean(),
        }
        for c in class_names:
            row[f"AP_{c}_mean"]   = group[f"AP_{c}"].mean()
            row[f"AP_{c}_std"]    = group[f"AP_{c}"].std()
            row[f"AUC_{c}_mean"]  = group[f"AUC_{c}"].mean()
            row[f"AUC_{c}_std"]   = group[f"AUC_{c}"].std()
            row[f"pval_{c}_mean"] = group[f"pval_{c}"].mean()
        summary_rows.append(row)

    df_summary = (
        pd.DataFrame(summary_rows)
        .sort_values("val_mAP_mean", ascending=False)
        .reset_index(drop=True)
    )
    df_summary.insert(0, "rank", df_summary.index + 1)

    # save
    out_dir  = Path(args.benchmark).parent
    stem     = Path(args.benchmark).stem
    df_summary.to_csv(out_dir / f"{stem}_aug_leaderboard.csv",   index=False)
    df.to_csv(       out_dir / f"{stem}_aug_folds.csv",          index=False)

    # print
    print(f"\n{'='*60}")
    print(f"  Benchmark : {bench['name']}")
    print(f"{'='*60}\n")
    print("── Global metrics ──")
    print(df_summary[[
        "rank", "experiment",
        "val_mAP_mean", "val_mAP_std",
        "val_auc_mean", "val_auc_std",
        "pval_overall_mean", "folds_done",
    ]].to_string(index=False))

    print(f"\nLeaderboard saved to {out_dir / f'{stem}_aug_leaderboard.csv'}")
    print(f"Per-fold detail  saved to {out_dir / f'{stem}_aug_folds.csv'}")


if __name__ == "__main__":
    main()