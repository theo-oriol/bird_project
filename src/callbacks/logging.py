import os
import json
import numpy as np


class MetricsDumpCallback:
    """
    Writes metrics.json after every epoch.
    This is what run_benchmark.py and make_leaderboard.py read.
    Always up to date — if the job crashes you still have the last epoch.
    """

    def on_epoch_end(self, epoch, model, hist, run_dir):
        metrics = {
            "epochs_trained": epoch,
            "val_mAP":        float(np.nanmean(np.stack(hist["val_ap"])[-1])),
            "val_auc":        float(np.nanmean(np.stack(hist["val_auc"])[-1])),
            "train_mAP":      float(np.nanmean(np.stack(hist["train_ap"])[-1])),
            "train_auc":      float(np.nanmean(np.stack(hist["train_auc"])[-1])),
            # per-class so you can debug individual classes later
            "val_ap_per_class":  np.stack(hist["val_ap"])[-1].tolist(),
            "val_auc_per_class": np.nanmean(
                np.stack(hist["val_auc"]), axis=0
            ).tolist(),
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


class HistoryDumpCallback:
    """
    Writes full training history to final_hist.json at the end of training.
    Useful for detailed post-hoc analysis and plotting.
    Only writes at the last epoch to avoid repeated serialization overhead.
    """

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, model, hist, run_dir):
        if epoch != self.total_epochs:
            return
        serializable = {
            k: [v.tolist() if isinstance(v, np.ndarray) else v for v in vals]
            for k, vals in hist.items()
        }
        with open(os.path.join(run_dir, "final_hist.json"), "w") as f:
            json.dump(serializable, f, indent=2)