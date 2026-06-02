import os
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu




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

                all_probs.append(torch.sigmoid(logits).cpu().numpy())
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