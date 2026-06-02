from .plotting   import PlottingCallback
from .logging    import MetricsDumpCallback, HistoryDumpCallback
from .checkpoint import CheckpointCallback
from .inference  import Inference


CALLBACK_REGISTRY = {
    "plotting":      PlottingCallback,
    "metrics_dump":  MetricsDumpCallback,
    "history_dump":  HistoryDumpCallback,
    "checkpoint":    CheckpointCallback,
    "inference":     Inference, 
}


def build_callbacks(cfg, val_loader=None, data_dir=None):
    callbacks = []
    for name in cfg.callbacks:
        if name not in CALLBACK_REGISTRY:
            raise ValueError(
                f"Unknown callback '{name}'. "
                f"Available: {list(CALLBACK_REGISTRY)}"
            )
        cls = CALLBACK_REGISTRY[name]
        # callbacks that need cfg get it, others don't
        if cls in (CheckpointCallback,):
            callbacks.append(cls(cfg))
        elif cls is HistoryDumpCallback:
            callbacks.append(cls(total_epochs=cfg.training.epochs))
        elif cls is Inference:
            assert data_dir is not None, \
                "inference callback requires data_source argument"
            callbacks.append(cls(cfg, data_dir=data_dir, val_loader=val_loader))
        elif cls is PlottingCallback:
            callbacks.append(cls(cfg))
        elif cls is MetricsDumpCallback:
            callbacks.append(cls(cfg))
        else:            raise ValueError(f"Don't know how to initialize callback '{name}'")
    return callbacks