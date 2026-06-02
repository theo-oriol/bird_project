import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class FrozenBackbone:
    needs_optimizer_refresh = False

    def setup(self, model, cfg):
        if not cfg.model.backbone.get("lora") :
            for p in model.backbone.parameters():
                p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

    def on_epoch_start(self, epoch, model, cfg, optimizer, scheduler):
        return None, None

    def post_step(self, model):
        pass

    def build_optimizer(self, model, cfg):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

    def build_scheduler(self, optimizer, cfg):
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.training.lr_min,
        )



STRATEGY_REGISTRY = {
    "frozen_backbone":      FrozenBackbone,
}


def build_strategy(cfg):
    name = cfg.strategy.type
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]()