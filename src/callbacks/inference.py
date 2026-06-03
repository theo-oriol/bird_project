import os
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr



class InferenceMultiBinCallback:
    """
    At the last epoch:
    - runs inference on the val set
    - saves one CSV per fold with per-image predictions and true labels
    - computes AUC p-values via Mann-Whitney U (per class + overall)
    - appends p-values to metrics.json
    """

    def __init__(self, cfg, data_dir=None, val_loader=None):
        self.cfg        = cfg
        self.device     = cfg.device
        self.val_loader = val_loader
        path_out_json = os.path.join("data","cross", data_dir,"labelname.json")
        with open(path_out_json, "r") as f:
            self.labels_cols = json.load(f)  

    def on_epoch_end(self, epoch, model, hist, run_dir):
        if epoch != self.cfg.training.epochs:
            return

        print("[inference] running final val inference...")
        all_probs, all_labels, all_names = self._collect(model)

        self._save_predictions(all_probs, all_labels, all_names, run_dir)
        pvalues_per_class, pvalue_overall = self._compute_pvalues(all_probs, all_labels)
        self._update_metrics(pvalues_per_class, pvalue_overall, run_dir)

    def _collect(self, model):
        model.eval()
        all_probs, all_labels, all_names = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, labels, names = batch
                imgs   = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(imgs)

                all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_names.extend(names)

        return (
            np.concatenate(all_probs,  axis=0),   # (N, C)
            np.concatenate(all_labels, axis=0),   # (N, C)
            all_names,                             # list of N strings
        )

    def _save_predictions(self, probs, labels, names, run_dir):
        n_classes = probs.shape[1]
        class_names = self.labels_cols[:n_classes]

        rows = {"image": names}

        for i, c in enumerate(class_names):
            rows[f"true_{c}"]  = labels[:, i]
            rows[f"pred_{c}"]  = probs[:, i]

        df = pd.DataFrame(rows)
        out_path = os.path.join(run_dir, "val_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"[inference] predictions saved to {out_path}")

    def _compute_pvalues(self, probs, labels):
        """
        Mann-Whitney U p-value per class:
            H0: scores for positives and negatives are equal
            A low p-value means the model ranks positives above negatives.

        Overall p-value: Mann-Whitney U on all classes pooled together.
        """
        n_classes = probs.shape[1]
        class_names = self.labels_cols[:n_classes]
        pvalues_per_class = {}

        all_pos_scores = []
        all_neg_scores = []

        for i, c in enumerate(class_names):
            pos_mask = labels[:, i] > 0
            neg_mask = ~pos_mask

            pos_scores = probs[pos_mask, i]
            neg_scores = probs[neg_mask, i]

            # collect for overall
            all_pos_scores.append(pos_scores)
            all_neg_scores.append(neg_scores)

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                pvalues_per_class[c] = None
                continue

            _, p = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
            pvalues_per_class[c] = float(p)

        # overall: pool all classes together
        all_pos = np.concatenate(all_pos_scores)
        all_neg = np.concatenate(all_neg_scores)

        if len(all_pos) > 0 and len(all_neg) > 0:
            _, p_overall = mannwhitneyu(all_pos, all_neg, alternative="greater")
            pvalue_overall = float(p_overall)
        else:
            pvalue_overall = None

        return pvalues_per_class, pvalue_overall

    def _update_metrics(self, pvalues_per_class, pvalue_overall, run_dir):
        metrics_path = os.path.join(run_dir, "metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}

        metrics["mannwhitney_pvalue_per_class"] = pvalues_per_class
        metrics["mannwhitney_pvalue_overall"]   = pvalue_overall

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("[inference] p-values added to metrics.json")
        print(f"  overall p-value : {pvalue_overall:.4e}" if pvalue_overall else "  overall p-value : N/A")
        for c, p in pvalues_per_class.items():
            print(f"  class {c:>8} : {p:.4e}" if p is not None else f"  class {c:>8} : N/A")




class InferenceMultiRegCallback:
    """
    At the last epoch:
    - runs inference on the val set
    - saves one CSV per fold with per-image predictions and true labels
    - computes Spearman correlation + p-value per output + overall
    - appends results to metrics.json
    """

    def __init__(self, cfg, data_dir=None, val_loader=None):
        self.cfg        = cfg
        self.device     = cfg.device
        self.val_loader = val_loader
        path_out_json   = os.path.join("data", "cross", data_dir, "labelname.json")
        with open(path_out_json, "r") as f:
            self.labels_cols = json.load(f)

    def on_epoch_end(self, epoch, model, hist, run_dir):
        if epoch != self.cfg.training.epochs:
            return

        print("[inference] running final val inference...")
        all_preds, all_labels, all_names = self._collect(model)

        self._save_predictions(all_preds, all_labels, all_names, run_dir)
        spearman_per_class, spearman_overall = self._compute_spearman(all_preds, all_labels)
        self._update_metrics(spearman_per_class, spearman_overall, run_dir)

    def _collect(self, model):
        model.eval()
        all_preds, all_labels, all_names = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, labels, names = batch
                imgs   = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds = model(imgs)   # raw outputs, no sigmoid for regression

                all_preds.append(torch.softmax(preds, dim=1).float().cpu().numpy())
                all_labels.append(labels.float().cpu().numpy())
                all_names.extend(names)

        return (
            np.concatenate(all_preds,  axis=0),   # (N, C)
            np.concatenate(all_labels, axis=0),   # (N, C)
            all_names,
        )

    def _save_predictions(self, preds, labels, names, run_dir):
        n_classes   = preds.shape[1]
        class_names = self.labels_cols[:n_classes]

        rows = {"image": names}
        for i, c in enumerate(class_names):
            rows[f"true_{c}"] = labels[:, i]
            rows[f"pred_{c}"] = preds[:, i]

        df = pd.DataFrame(rows)
        out_path = os.path.join(run_dir, "val_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"[inference] predictions saved to {out_path}")

    def _compute_spearman(self, preds, labels):
        """
        Spearman correlation + p-value per output:
            H0: no monotonic relationship between predictions and targets.
            A low p-value + high rho means the model ranks samples correctly.

        Overall: flatten all outputs and compute a single correlation.
        """
        n_classes   = preds.shape[1]
        class_names = self.labels_cols[:n_classes]
        spearman_per_class = {}

        for i, c in enumerate(class_names):
            p_vals = preds[:, i]
            t_vals = labels[:, i]

            # skip if all targets are identical (rho undefined)
            # if len(np.unique(t_vals)) < 2:
            #     spearman_per_class[c] = {"rho": None, "pvalue": None}
            #     continue
            if len(np.unique(t_vals)) < 2:
                 spearman_per_class[c] = None
                 continue
            rho, pvalue = spearmanr(p_vals, t_vals)
            # spearman_per_class[c] = {
            #     "rho":    float(rho),
            #     "pvalue": float(pvalue),
            # }
            spearman_per_class[c] = float(pvalue)

        # overall: flatten all outputs together
        rho_overall, pvalue_overall = spearmanr(
            preds.flatten(), labels.flatten()
        )
        # spearman_overall = {
        #     "rho":    float(rho_overall),
        #     "pvalue": float(pvalue_overall),
        # }
        spearman_overall = float(pvalue_overall)

        return spearman_per_class, spearman_overall

    def _update_metrics(self, spearman_per_class, spearman_overall, run_dir):
        metrics_path = os.path.join(run_dir, "metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}

        metrics["spearman_per_class"] = spearman_per_class
        metrics["spearman_overall"]   = spearman_overall

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("[inference] Spearman results added to metrics.json")
        # print(f"  overall rho={spearman_overall['rho']:.4f}  p={spearman_overall['pvalue']:.4e}")
        print(f"  overall p={spearman_overall:.4e}")
        for c, s in spearman_per_class.items():
            # if s["rho"] is not None:
            #     print(f"  class {c:>8} : rho={s['rho']:.4f}  p={s['pvalue']:.4e}")
            # else:
            #     print(f"  class {c:>8} : N/A")
            if s is not None:
                print(f"  class {c:>8} : p={s:.4e}")
            else:
                print(f"  class {c:>8} : N/A")


def Inference(cfg, data_dir=None, val_loader=None):
    if cfg.model.head.type == "multi_binary":
        return InferenceMultiBinCallback(cfg, data_dir=data_dir, val_loader=val_loader)
    elif cfg.model.head.type == "multi_regression":
        return InferenceMultiRegCallback(cfg, data_dir=data_dir, val_loader=val_loader)
    else:
        raise ValueError(f"Unknown head type '{cfg.model.head.type}'")