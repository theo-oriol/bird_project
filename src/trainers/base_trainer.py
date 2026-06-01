import os
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .strategies import build_strategy


class Trainer:
    def __init__(self, model, cfg, run_dir, callbacks=None):
        self.model     = model.to(cfg.device)
        self.cfg       = cfg
        self.run_dir   = run_dir
        self.callbacks = callbacks or []
        self.device    = cfg.device

        self.strategy  = build_strategy(cfg)
        self.strategy.setup(self.model, cfg)

        self.optimizer = self.strategy.build_optimizer(model, cfg)
        self.scheduler = self.strategy.build_scheduler(self.optimizer, cfg)
        self.scaler    = torch.amp.GradScaler()

        self.hist = {
            "epoch":      [],
            "train_loss": [], "val_loss":  [],
            "train_ap":   [], "val_ap":    [],
            "train_auc":  [], "val_auc":   [],
        }

    def fit(self, train_loader, val_loader, criterion):
        for epoch in range(1, self.cfg.training.epochs + 1):

            # let strategy react — for frozen_backbone this is a no-op
            new_opt, new_sched = self.strategy.on_epoch_start(
                epoch, self.model, self.cfg, self.optimizer, self.scheduler
            )
            if new_opt is not None:
                self.optimizer = new_opt
                self.scheduler = new_sched

            train_metrics = self._train_epoch(train_loader, criterion)
            val_metrics   = self._val_epoch(val_loader, criterion)
            self.scheduler.step()

            self._update_hist(epoch, train_metrics, val_metrics)
            self._log(epoch, train_metrics, val_metrics)

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, self.model, self.hist, self.run_dir)

        self._save_checkpoint(epoch)
        return self.model, self.hist

    def _train_epoch(self, loader, criterion):
        self.model.train()
        loss_sum, all_probs, all_labels = 0.0, [], []

        for batch in loader:
            imgs, labels, _ = batch
            imgs  = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(imgs)
                loss   = criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.strategy.post_step(self.model)

            loss_sum += loss.item()
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())

        return self._compute_metrics(loss_sum, all_probs, all_labels, loader)

    def _val_epoch(self, loader, criterion):
        if hasattr(self.strategy, "apply_ema"):
            self.strategy.apply_ema(self.model)

        self.model.eval()
        loss_sum, all_probs, all_labels = 0.0, [], []

        with torch.no_grad():
            for batch in loader:
                imgs, labels, _ = batch
                imgs  = imgs.to(self.device)
                labels = labels.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(imgs)
                    loss   = criterion(logits, labels)

                loss_sum += loss.item()
                all_probs.append(torch.sigmoid(logits).detach().cpu())
                all_labels.append(labels.detach().cpu())

        if hasattr(self.strategy, "restore_ema"):
            self.strategy.restore_ema(self.model)

        return self._compute_metrics(loss_sum, all_probs, all_labels, loader)

    def _compute_metrics(self, loss_sum, all_probs, all_labels, loader):
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score

        probs  = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()
        loss   = loss_sum / max(1, len(loader))

        ap = average_precision_score(labels, probs, average=None)

        try:
            auc = roc_auc_score(labels, probs, average=None)
        except ValueError:
            n   = labels.shape[1]
            auc = np.array([
                roc_auc_score(labels[:, i], probs[:, i])
                if len(np.unique(labels[:, i])) > 1 else float("nan")
                for i in range(n)
            ])

        return {"loss": loss, "ap": ap, "auc": auc}

    def _update_hist(self, epoch, tr, va):
        self.hist["epoch"].append(epoch)
        self.hist["train_loss"].append(tr["loss"])
        self.hist["val_loss"].append(va["loss"])
        self.hist["train_ap"].append(tr["ap"])
        self.hist["val_ap"].append(va["ap"])
        self.hist["train_auc"].append(tr["auc"])
        self.hist["val_auc"].append(va["auc"])

    def _log(self, epoch, tr, va):
        import numpy as np
        print(
            f"[{epoch:03d}] "
            f"train  loss {tr['loss']:.4f}  mAP {np.mean(tr['ap']):.4f}  AUC {np.nanmean(tr['auc']):.4f} | "
            f"val    loss {va['loss']:.4f}  mAP {np.mean(va['ap']):.4f}  AUC {np.nanmean(va['auc']):.4f}"
        )

    def _save_checkpoint(self, epoch):
        import copy
        m = copy.deepcopy(self.model)
        if hasattr(m.backbone, "merge_and_unload"):
            m.backbone = m.backbone.merge_and_unload()
        torch.save(
            {
                "model":  m.state_dict(),
                "epoch":  epoch,
                "hist":   self._serializable_hist(),
                "config": self.cfg,
            },
            os.path.join(self.run_dir, "checkpoint_last.pth"),
        )
        print(f"[checkpoint] saved to {self.run_dir}/checkpoint_last.pth")

    def _serializable_hist(self):
        import numpy as np
        return {
            k: [v.tolist() if isinstance(v, np.ndarray) else v for v in vals]
            for k, vals in self.hist.items()
        }