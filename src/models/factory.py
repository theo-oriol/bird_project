import torch
import torch.nn as nn

from .backbone import build_backbone
from .heads import build_head



class Model(nn.Module):
    """
    Wraps a backbone + head into a single model.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        feats = self.backbone(x)                  
        return self.head(feats)

    def merge_lora(self):
        """Call before saving — merges LoRA weights into backbone."""
        if hasattr(self.backbone, "merge_and_unload"):
            self.backbone = self.backbone.merge_and_unload()


def build_model(cfg):
    backbone = build_backbone(cfg)
    head     = build_head(cfg)
    return Model(backbone, head)