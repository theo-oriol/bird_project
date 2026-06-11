# Bird Project

---

## Project structure

```
.
├── analysis/                                # post-hoc analysis notebooks / scripts
│   ├── discriminant_analysis.py
│   ├── dream_sim.py                         # perceptual similarity via DreamSim
│   ├── extract_info_from_eval.py
│   ├── linear_analysis.py
│   └── specialisation.py                    # per-class specialisation metrics
├── benchmarks/                              # auto-generated, gitignored
│   ├── cross/                               # cross-fold leaderboard CSVs
│   └── per_fold/                            # per-fold detailed CSVs
├── checks/
│   └── check_folds.py                       # verify folds integrity
├── configs/
│   ├── benchmarks/                          # benchmark YAMLs (gitignored contents)
│   ├── test_benchmarks/                     # lightweight benchmarks for quick checks
│   ├── base.yaml                            # default config, all fields documented
│   ├── labelname.json
│   └── labeltoname.json
├── data/
│   ├── raw/                                 # source CSVs, never modified
│   ├── cross/                               # versioned fold splits
│   ├── datasets/                            # versioned flat datasets
│   └── meta/                               # embeddings (*.npy, gitignored) + metadata CSVs
├── docs/
│   └── dataset_v1.md
├── experiments/                             # auto-generated, gitignored
│   └── {exp_name}/
│       ├── fold_{n}/
│       │   ├── config.yaml
│       │   ├── checkpoint_last.pth
│       │   ├── checkpoint_best.pth
│       │   ├── metrics.json
│       │   ├── final_hist.json
│       │   ├── val_predictions.csv
│       │   ├── loss.png
│       │   ├── val_ap.png
│       │   └── val_auc.png
│       └── summary.json                     # mean/std across folds
├── models/                                  # pretrained weights (gitignored binaries)
│   ├── ensemble_lora/                       # LoRA adapter weights
│   └── facebookresearch_dino_main/          # DINO source code
├── scripts/
│   ├── train.py                             # single fold entrypoint
│   ├── run_benchmark.py                     # runs all pending folds, updates YAML status
│   ├── run_aug_eval.py                      # augmentation robustness evaluation
│   ├── make_leaderboard.py                  # reads metrics.json, builds leaderboard CSV
│   ├── fusion_predictions.py                # merges per-fold val_predictions.csv into one
│   ├── specialisation.py                    # per-class specialisation analysis from preds
│   ├── generate_meta_data_grayscale.py      # extract green-channel embeddings from images
│   ├── build_dataset.py                     # dataset creation
│   ├── build_folds.py                       # folds creation
│   ├── slurm_parse_and_submit.py            # submit one SLURM job per pending fold
│   └── slurm_submit.sh                      # raw SLURM job template
├── src/
│   ├── models/
│   │   ├── backbone.py                      # loads backbone, wraps with LoRA if configured
│   │   ├── heads.py                         # multi_binary, multi_regression, multi_quantile
│   │   └── factory.py                       # assembles backbone + head into MultiViewModel
│   ├── datasets/
│   │   ├── dataset.py                       # BirdDataset, load_csv, build_dataloaders
│   │   └── transforms.py                    # image augmentation pipelines
│   ├── trainers/
│   │   ├── strategies.py                    # frozen_backbone, progressive_unfreeze, ema
│   │   └── base_trainer.py                  # training loop, delegates to strategy
│   ├── losses/
│   │   └── factory.py                       # bce_weighted, bce, mse, huber, pinball
│   └── callbacks/
│       ├── inference.py                     # runs inference and saves val_predictions.csv
│       ├── plotting.py                      # loss/mAP/AUC curves
│       ├── logging.py                       # metrics.json, final_hist.json
│       ├── checkpoint.py                    # checkpoint_last.pth, checkpoint_best.pth
│       └── __init__.py                      # CALLBACK_REGISTRY, build_callbacks
└── standalone/                              # self-contained mini models (no src dependency)
    ├── mini_model_color_hist.py
    ├── mini_model_color_ms.py
    ├── mini_model_grayscale_hist.py
    ├── mini_model_grayscale_norm_gabor_filter.py
    └── mini_model_grayscale_norm_hist.py
```

---

## First-time setup

**1. Clone and install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create your `.env` file**
```bash
cp .env.example .env
```

Fill in the paths for your machine:
```bash
# .env
SOURCE_IMG=/path/to/SOURCE_IMG
DINOV3_PATH=/path/to/DINOV3
dinov3_vits16=/path/to/dinov3_vits16.pth
```

The `.env` is machine-specific and gitignored — fill it once per machine.

**3. Place pretrained weights**

Put your model weights under `models/` (gitignored). Expected files:
- `models/dino_vitb16_pretrain.pth`
- `models/clip_vitb16_pretrain.pth.tar`
- `models/open_clip_vitb16_pretrain.pth.tar`

---

## Running a benchmark

**Launch all pending experiments**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py configs/benchmarks/multi_binary_classification_v1_fam.yaml
```

This will:
1. Find every fold with `status: pending` in the benchmark YAML
2. Train them one by one, writing `status: running` then `status: done` as each finishes
3. Save all outputs to `experiments/{exp_name}/fold_{n}/`
4. Print and save the leaderboard when all folds are done

**Run in the background (won't die if you close the terminal)**
```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_benchmark.py configs/benchmarks/multi_binary_classification_v1_fam.yaml > logs/benchmark.log 2>&1 &
echo $!   # prints the process ID — save this to kill it later
```

Monitor progress:
```bash
tail -f logs/benchmark.log
```

**Submit to SLURM (one job per pending fold)**
```bash
python scripts/slurm_parse_and_submit.py configs/benchmarks/multi_binary_classification_v1_fam.yaml
```

**Run a single fold manually**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --exp-name my_exp \
  --fold 0 \
  --run-dir experiments/my_exp/fold_0 \
  --benchmark configs/benchmarks/multi_binary_classification_v1_fam.yaml
```

**Rebuild the leaderboard without retraining**
```bash
python scripts/make_leaderboard.py configs/benchmarks/multi_binary_classification_v1_fam.yaml
```

---

## Post-training scripts

**Merge per-fold predictions into one file**
```bash
python scripts/fusion_predictions.py experiments/my_exp/
```
Writes `experiments/my_exp/{benchmark_name}.csv`.

**Augmentation robustness evaluation**
```bash
python scripts/run_aug_eval.py --benchmark configs/benchmarks/multi_binary_classification_v1_fam.yaml
```

**Per-class specialisation analysis**
```bash
python scripts/specialisation.py experiments/my_exp/
```

---

## Adding a new experiment to a benchmark

Open the benchmark YAML and append a new block under `experiments:`. Set all folds to `pending`:

```yaml
experiments:

  # existing experiment — already done, won't be re-run
  dinov3_freeze:
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
      dataset_dir: data/cross/cross_version_1_FAM
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
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py configs/benchmarks/multi_binary_classification_v1_fam.yaml
```

Only the new `pending` folds will train. Done folds are never touched.

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
| `val_predictions.csv` | raw predictions for each validation sample |
| `loss.png` | train/val loss curve |
| `val_ap.png` | mean AP curve |
| `val_auc.png` | mean AUC curve |

Each experiment also produces a `summary.json` with mean/std across folds, and the benchmark runner produces:

| File | Description |
|---|---|
| `benchmarks/cross/{name}_leaderboard.csv` | one row per experiment, ranked by metric |
| `benchmarks/per_fold/{name}_folds.csv` | one row per fold for detailed analysis |

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
| `KLDiv` | Kullback–Leibler divergence |

---

## Dataset versioning

Dataset versions live in `data/cross/` with a self-documenting name:

```
data/cross/
└── cross_version_1_FAM/
    ├── labelname.json
    ├── train_fold_0.csv
    ├── valid_fold_0.csv
    └── ...
docs/
└── dataset_v1.md   ← what this dataset is, how it was built
```

To create a new dataset version:
```bash
python scripts/build_dataset.py
python scripts/build_folds.py
```
