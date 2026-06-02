import os
import copy
import torch


class CheckpointCallback:
    """
    Saves checkpoint_last.pth every epoch — overwrites previous.
    LoRA weights are merged before saving so checkpoints are self-contained.
    """

    def __init__(self, cfg):
        self.cfg      = cfg

    def on_epoch_end(self, epoch, model, hist, run_dir):
        import numpy as np

        state = self._build_state(model, epoch, hist)
        torch.save(state, os.path.join(run_dir, "checkpoint_last.pth"))

    def _build_state(self, model, epoch, hist):
        import numpy as np
        m = copy.deepcopy(model)
        if hasattr(m.backbone, "merge_and_unload"):
            m.backbone = m.backbone.merge_and_unload()

        return {
            "model":  m.state_dict(),
            "epoch":  epoch,
            "config": self.cfg,
            "hist": {
                k: [v.tolist() if isinstance(v, np.ndarray) else v for v in vals]
                for k, vals in hist.items()
            },
        }