import os
import numpy as np
import matplotlib.pyplot as plt


class PlottingCallback:
    """
    Saves loss, mAP and AUC curves to the run directory.
    Plots every `plot_every` epochs to avoid I/O overhead on long runs.
    """

    def __init__(self, cfg, plot_every=5):
        self.plot_every = plot_every
        self.cfg = cfg

    def on_epoch_end(self, epoch, model, hist, run_dir):
        if epoch % self.plot_every != 0 and epoch != 1:
            return
        self._plot_loss(hist, run_dir)
        if self.cfg.data.binarize == True:
            self._plot_metric(hist, "train_ap",  "val_ap",  "mAP", run_dir)
            self._plot_metric(hist, "train_auc", "val_auc", "AUC", run_dir)
        elif self.cfg.data.binarize == False:
            self._plot_metric(hist, "train_mse", "val_mse", "MSE", run_dir)
            self._plot_metric(hist, "train_mae", "val_mae", "MAE", run_dir)
        else:
            raise ValueError(f"Unknown head type '{self.cfg.model.head.type}'")

    def _plot_loss(self, hist, run_dir):
        plt.figure()
        plt.plot(hist["epoch"], hist["train_loss"], label="train")
        plt.plot(hist["epoch"], hist["val_loss"],   label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss.png"), dpi=150)
        plt.close()

    def _plot_metric(self, hist, train_key, val_key, ylabel, run_dir):
        train_vals = np.stack(hist[train_key])
        val_vals   = np.stack(hist[val_key])
        plt.figure()
        plt.plot(hist["epoch"], np.nanmean(train_vals, axis=-1), label="train")
        plt.plot(hist["epoch"], np.nanmean(val_vals,   axis=-1), label="val")
        plt.xlabel("epoch")
        plt.ylabel(f"mean {ylabel}")
        plt.title(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{val_key}.png"), dpi=150)
        plt.close()