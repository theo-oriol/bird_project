# Bird Project

Deep learning experiments for multi-view bird image classification, regression, and quantile regression. Built around a YAML-driven benchmark system — no hardcoded paths, no one-file-per-experiment chaos.

---

## Project structure

```
.
├── configs/
│   ├── base.yaml                        # default values for all experiments
│   └── benchmarks/
│       └── classification_v1.yaml       # benchmark definition + experiment registry
├── data/
│   ├── raw/                             # source CSVs, never modified
│   ├── processed/                       # intermediate files
│   └── datasets/                        # versioned dataset folders
│       └── cross_version_1_FAM/
│           ├── labelname.json
│           ├── train_fold_0.csv
│           └── valid_fold_0.csv
├── experiments/                         # auto-generated, gitignored
│   └── dinov3S_Freeze_cross_v1/
│       ├── fold_0/
│       │   ├── config.yaml
│       │   ├── checkpoint_last.pth
│       │   ├── checkpoint_best.pth
│       │   ├── metrics.json
│       │   ├── final_hist.json
│       │   ├── loss.png
│       │   ├── val_ap.png
│       │   └── val_auc.png
│       ├── fold_1/
│       ├── fold_2/
│       └── summary.json                 # mean/std across folds
├── src/
│   ├── models/
│   │   ├── backbone.py                  # loads backbone, wraps with LoRA if configured
│   │   ├── heads.py                     # multi_binary, multi_regression, multi_quantile
│   │   └── factory.py                   # assembles backbone + head into MultiViewModel
│   ├── datasets/
│   │   └── dataset.py                   # BirdDataset, load_csv, build_dataloaders
│   ├── trainers/
│   │   ├── strategies.py                # frozen_backbone, progressive_unfreeze, ema
│   │   └── base_trainer.py              # training loop, delegates to strategy
│   ├── losses/
│   │   └── factory.py                   # bce_weighted, bce, mse, huber, pinball
│   └── callbacks/
│       ├── plotting.py                  # loss/mAP/AUC curves
│       ├── logging.py                   # metrics.json, final_hist.json
│       ├── checkpoint.py                # checkpoint_last.pth, checkpoint_best.pth
│       └── __init__.py                  # CALLBACK_REGISTRY, build_callbacks
└── scripts/
    ├── train.py                         # single fold entrypoint
    ├── run_benchmark.py                 # runs all pending folds, updates YAML status
    ├── make_leaderboard.py              # reads metrics.json, builds leaderboard CSV
    ├── build_dataset.py                 # dataset creation
    └── slurm_submit.sh                  # SLURM job submission
```

---

## First-time setup

**1. Clone and install dependencies**
```bash
pip install torch torchvision omegaconf python-dotenv peft tqdm scikit-learn matplotlib pandas
```

**2. Create your `.env` file**
```bash
cp .env.example .env
```

Fill in the paths for your machine:
```bash
# .env
SOURCE_IMG=/data/theo/V0_NEW_Segmented-Aves-Back-224-RGBA
EBIRD_CSV=/data/theo/PHA_Jung_lvl1_fraction.csv
MATCH_CSV=/home/lionel/theo/post_doc/papier/scripts/crosswalk_image_habitat_v2.csv
OUT_CSV=/home/lionel/theo/post_doc/papier/scripts/V2_dataset_with_labels.csv
```

The `.env` is machine-specific and gitignored — you fill it once per machine and never touch it again.

**3. Verify your paths**
```bash
ls data/datasets/cross_version_1_FAM
ls /data/theo/V0_NEW_Segmented-Aves-Back-224-RGBA | head -5
ls /data/theo/dinov3
```

---

## Running a benchmark

**Check available GPUs**
```bash
nvidia-smi
```

**Launch all pending experiments**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py configs/benchmarks/classification_v1.yaml
```

This will:
1. Find every fold with `status: pending` in the benchmark YAML
2. Train them one by one, writing `status: running` then `status: done` as each finishes
3. Save all outputs to `experiments/{exp_name}/fold_{n}/`
4. Print and save the leaderboard when all folds are done

**Run in the background (won't die if you close the terminal)**
```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_benchmark.py configs/benchmarks/classification_v1.yaml > logs/benchmark_v1.log 2>&1 &
echo $!   # prints the process ID — save this to kill it later
```

Monitor progress:
```bash
tail -f logs/benchmark_v1.log
```

**Run a single fold manually**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --exp-name dinov3_freeze \
  --fold 0 \
  --run-dir experiments/dinov3S_Freeze_cross_v1/fold_0 \
  --benchmark configs/benchmarks/classification_v1.yaml
```

**Rebuild the leaderboard without retraining**
```bash
python scripts/make_leaderboard.py configs/benchmarks/classification_v1.yaml
```

---

## Adding a new experiment to a benchmark

Open the benchmark YAML and append a new block under `experiments:`. Set all folds to `pending`:

```yaml
experiments:

  # existing experiment — already done, won't be re-run
  dinov3_freeze:
    ...
    folds:
      0: { status: done, run_dir: experiments/dinov3S_Freeze_cross_v1/fold_0 }
      1: { status: done, run_dir: experiments/dinov3S_Freeze_cross_v1/fold_1 }
      2: { status: done, run_dir: experiments/dinov3S_Freeze_cross_v1/fold_2 }

  # new experiment — will be picked up on next run
  dinov3_lora:
    model:
      backbone:
        name: dinov3_vits16
        local_path: /data/theo/dinov3
        weights: /data/theo/dinov3_vits16_pretrain.pth
        feat_dim: 384
        lora:
          r: 8
          alpha: 16
          dropout: 0.05
          target_modules: ["attn.qkv", "attn.proj"]
      head:
        type: multi_binary
        feat_dim: 384
        num_classes: 10
    strategy:
      type: frozen_backbone
    data:
      dataset_dir: /home/lionel/theo/bird_project/data/cross/cross_version_1_FAM
      img_dir: /data/theo/V0_NEW_Segmented-Aves-Back-224-RGBA
      binarize: true
    training:
      epochs: 100
      lr: 1e-4
      batch_size: 64
    loss:
      type: bce_weighted
    folds:
      0: { status: pending, run_dir: experiments/dinov3_lora_cross_v1/fold_0 }
      1: { status: pending, run_dir: experiments/dinov3_lora_cross_v1/fold_1 }
      2: { status: pending, run_dir: experiments/dinov3_lora_cross_v1/fold_2 }
```

Then rerun:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py configs/benchmarks/classification_v1.yaml
```

Only the new `pending` folds will train. Done folds are never touched.

---

## Running on multiple GPUs in parallel

Each GPU gets its own benchmark YAML — never share a YAML between two concurrent runs:

```bash
# terminal 1 — GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_benchmark.py configs/benchmarks/classification_v1.yaml > logs/bench_v1.log 2>&1 &

# terminal 2 — GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_benchmark.py configs/benchmarks/regression_v1.yaml > logs/bench_reg.log 2>&1 &
```

---

## Outputs

Every fold produces:

| File | Description |
|---|---|
| `config.yaml` | exact config used — full reproducibility |
| `checkpoint_last.pth` | weights at the last epoch |
| `checkpoint_best.pth` | weights at the best val_mAP |
| `metrics.json` | val_mAP, val_auc, train_mAP (updated every epoch) |
| `final_hist.json` | full per-epoch history |
| `loss.png` | train/val loss curve |
| `val_ap.png` | mean AP curve |
| `val_auc.png` | mean AUC curve |

Each experiment also produces a `summary.json` with mean/std across folds, and the benchmark runner produces:

| File | Description |
|---|---|
| `configs/benchmarks/{name}_leaderboard.csv` | one row per experiment, ranked by metric |
| `configs/benchmarks/{name}_folds.csv` | one row per fold for detailed analysis |

---

## Config reference

All fields and their defaults live in `configs/base.yaml`. An experiment only needs to specify what differs from the base.

**Available head types**

| type | task | loss to use |
|---|---|---|
| `multi_binary` | multi-label classification | `bce_weighted` or `bce` |
| `multi_regression` | multi-output regression | `mse` or `huber` |
| `multi_quantile` | quantile regression | `pinball` |

**Available strategies**

| type | behaviour |
|---|---|
| `frozen_backbone` | backbone frozen, head trains only |
| `progressive_unfreeze` | unfreezes one block every `unfreeze_every` epochs |
| `ema` | progressive unfreeze + exponential moving average of weights |

**Available losses**

| type | notes |
|---|---|
| `bce_weighted` | BCE with auto-computed class weights from train set |
| `bce` | unweighted BCE |
| `mse` | mean squared error |
| `huber` | Huber loss, set `delta` in config |
| `pinball` | quantile loss, set `quantiles: [0.1, 0.5, 0.9]` in config |

---

## Dataset versioning

Dataset versions live in `data/datasets/` with a self-documenting name:

```
data/datasets/
└── v2__2024-03-05__augmented/
    ├── dataset_card.yaml    ← what this dataset is, how it was built
    ├── train_fold_0.csv
    ├── valid_fold_0.csv
    └── labelname.json
```

`dataset_card.yaml` documents the version so you never lose track of what a dataset contains:

```yaml
version: 2
date: 2024-03-05
description: augmented with eBird cross-validation correction
source_csv: raw/PHA_Jung_lvl1_fraction.csv
classes: [1, 2, 3, 4, 8, 12, 14, 9_11, 5_13_15, 6_7]
num_train: 12400
num_valid: 3100
folds: 3
transforms_applied: [normalize, random_vertical_flip]
known_issues: null
changelog:
  - v1: baseline
  - v2: added eBird cross correction, rebalanced splits
```

To create a new dataset version, run:
```bash
python scripts/build_dataset.py
```
