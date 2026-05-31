import os
import sys
import json
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

# make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.factory      import build_model
from src.datasets.dataset    import build_dataloaders
from src.losses.factory      import build_criterion
from src.trainers.base_trainer import Trainer
from src.callbacks           import build_callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name",  required=True,  help="key in the benchmark YAML")
    parser.add_argument("--run-dir",   required=True,  help="where to save outputs")
    parser.add_argument("--benchmark", required=True,  help="path to benchmark YAML")
    parser.add_argument("--fold",      required=True,  type=int, help="fold index")
    args = parser.parse_args()

    # ---- load and merge configs ----
    base = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    )
    benchmark = OmegaConf.load(args.benchmark)
    exp_cfg   = benchmark.experiments[args.exp_name]

    # remove benchmark-only keys before merging into training config
    train_override = OmegaConf.masked_copy(
        exp_cfg, [k for k in exp_cfg if k not in ("folds",)]
    )
    cfg = OmegaConf.merge(base, train_override)

    # set fold
    OmegaConf.update(cfg, "data.fold", args.fold)

    # ---- setup run directory ----
    os.makedirs(args.run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(args.run_dir, "config.yaml"))
    print(f"\n{'='*60}")
    print(f"  exp  : {args.exp_name}")
    print(f"  fold : {args.fold}")
    print(f"  dir  : {args.run_dir}")
    print(f"{'='*60}\n")

    # ---- build everything ----
    train_loader, val_loader, train_labels = build_dataloaders(cfg)
    model     = build_model(cfg)
    criterion = build_criterion(cfg, train_labels)
    callbacks = build_callbacks(cfg)

    # ---- train ----
    trainer = Trainer(model, cfg, args.run_dir, callbacks=callbacks)
    trainer.fit(train_loader, val_loader, criterion)


if __name__ == "__main__":
    main()