from .plotting   import PlottingCallback
from .logging    import MetricsDumpCallback, HistoryDumpCallback
from .checkpoint import CheckpointCallback


CALLBACK_REGISTRY = {
    "plotting":      PlottingCallback,
    "metrics_dump":  MetricsDumpCallback,
    "history_dump":  HistoryDumpCallback,
    "checkpoint":    CheckpointCallback,
}


def build_callbacks(cfg):
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
        else:
            callbacks.append(cls())
    return callbacks