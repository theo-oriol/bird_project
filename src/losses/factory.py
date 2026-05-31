import torch
import torch.nn as nn
import numpy as np


def build_criterion(cfg, train_labels=None):
    loss_type = cfg.loss.type

    if loss_type == "bce_weighted":
        assert train_labels is not None, \
            "bce_weighted requires train_labels to compute pos_weight"
        N          = train_labels.shape[0]
        pos        = (train_labels > 0).sum(axis=0)
        neg        = N - pos
        pos_weight = torch.from_numpy(
            neg / np.clip(pos, 1, None)
        ).float()
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_type == "mse":
        return nn.MSELoss()
    
    if loss_type =="KLDiv":
        return nn.KLDivLoss(reduction="batchmean")

    raise ValueError(
        f"Unknown loss '{loss_type}'. "
        f"Available: bce_weighted | bce | mse | KLDiv"
    )