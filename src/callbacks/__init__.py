from .plotting   import PlottingCallback
from .logging    import MetricsDumpCallback, HistoryDumpCallback
from .checkpoint import CheckpointCallback
from .inference  import InferenceMultiBinCallback


CALLBACK_REGISTRY = {
    "plotting":      PlottingCallback,
    "metrics_dump":  MetricsDumpCallback,
    "history_dump":  HistoryDumpCallback,
    "checkpoint":    CheckpointCallback,
    "inferenceMultiBin":     InferenceMultiBinCallback, 
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
        elif cls is InferenceMultiBinCallback:
            assert data_dir is not None, \
                "inference callback requires data_source argument"
            callbacks.append(cls(cfg, data_dir=data_dir, val_loader=val_loader))
        else:
            callbacks.append(cls())
    return callbacks