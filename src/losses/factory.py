import torch
import torch.nn as nn
import numpy as np

def AslLoss(gamma_pos=0, gamma_neg=0, clip=0.05, eps=1e-8):
    def asl_loss(inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        targets = targets.float()
        inputs_sigmoid = torch.clamp(inputs_sigmoid, clip, 1 - clip)
        loss_pos = -targets * torch.log(inputs_sigmoid + eps) * (1 - inputs_sigmoid) ** gamma_pos
        loss_neg = -(1 - targets) * torch.log(1 - inputs_sigmoid + eps) * inputs_sigmoid ** gamma_neg
        return (loss_pos + loss_neg).mean()
    return asl_loss

def bce_weighted_loss(pos_weight):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def bce_weighted_loss(inputs, targets): 
        return loss_fn(inputs, targets)
    return bce_weighted_loss

def bce_loss():
    loss_fn = nn.BCEWithLogitsLoss()
    def bce_loss(inputs, targets):
        return loss_fn(inputs, targets)
    return bce_loss

def mse_loss():
    loss_fn = nn.MSELoss()
    def mse_loss(inputs, targets):
        inputs_softmax = torch.softmax(inputs, dim=-1)
        return loss_fn(inputs_softmax, targets)
    return mse_loss

def kl_div_loss():
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    def kl_div_loss(inputs, targets):
        inputs_log_softmax = torch.log_softmax(inputs, dim=-1)
        return loss_fn(inputs_log_softmax, targets)
    return kl_div_loss

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
        return bce_weighted_loss(pos_weight.to(cfg.device))

    if loss_type == "bce":
        return bce_loss()

    if loss_type == "mse":
        return mse_loss()
    
    if loss_type =="KLDiv":
        return kl_div_loss()

    if loss_type == "asl":
        return AslLoss(
            gamma_pos=cfg.loss.get("gamma_pos", 0),
            gamma_neg=cfg.loss.get("gamma_neg", 0),
            clip=cfg.loss.get("clip", 0.05),
            eps=cfg.loss.get("eps", 1e-8)
        )

    raise ValueError(
        f"Unknown loss '{loss_type}'. "
        f"Available: bce_weighted | bce | mse | KLDiv | asl"
    )