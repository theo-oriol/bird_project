import torch
import torch.nn as nn


class MultiBinaryHead(nn.Module):
    """
    Multi-label binary classification.
    Output: (B, num_classes) logits → BCEWithLogitsLoss
    """
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MultiRegressionHead(nn.Module):
    """
    Multi-output regression.
    Output: (B, num_outputs) → KL
    """
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MultiQuantileHead(nn.Module):
    """
    Multi-output quantile regression.
    Output: (B, num_outputs, num_quantiles) → PinballLoss
    Quantiles defined in config, e.g. [0.1, 0.5, 0.9]
    """
    def __init__(self, feat_dim, num_outputs, quantiles, dropout=0.1):
        super().__init__()
        self.num_outputs  = num_outputs
        self.num_quantiles = len(quantiles)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_outputs * len(quantiles)),
        )

    def forward(self, x):
        out = self.net(x)
        # reshape so loss functions can address each quantile separately
        return out.view(x.shape[0], self.num_outputs, self.num_quantiles)


HEAD_REGISTRY = {
    "multi_binary":     MultiBinaryHead,
    "multi_regression": MultiRegressionHead,
    "multi_quantile":   MultiQuantileHead,
}


def build_head(cfg):
    head_type = cfg.model.head.type
    if head_type not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head '{head_type}'. Available: {list(HEAD_REGISTRY)}"
        )
    head_cls = HEAD_REGISTRY[head_type]
    # pass all head config fields except 'type' as kwargs
    kwargs = {k: v for k, v in cfg.model.head.items() if k != "type"}
    return head_cls(**kwargs)