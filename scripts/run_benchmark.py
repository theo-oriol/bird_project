import sys
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import os 

BENCHMARK_PATH = Path(sys.argv[1])


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_fold(exp_name, fold_idx, fold_cfg, exp_cfg, bench_cfg):
    fold_cfg["status"] = "running"
    save(cfg, BENCHMARK_PATH)

    result = subprocess.run([
        "python", "scripts/train.py",
        "--exp-name",  exp_name,
        "--fold",      str(fold_idx),
        "--run-dir",   os.path.join("experiments", bench_cfg["name"], fold_cfg["run_dir"]),
        "--benchmark", str(BENCHMARK_PATH),
    ])

    fold_cfg["status"] = "done" if result.returncode == 0 else "failed"
    save(cfg, BENCHMARK_PATH)


cfg   = load(BENCHMARK_PATH)
bench = cfg["benchmark"]

for exp_name in list(load(BENCHMARK_PATH)["experiments"].keys()):
    print(f"\n{'='*60}\n{exp_name}\n{'='*60}")

    cfg     = load(BENCHMARK_PATH)
    exp_cfg = cfg["experiments"][exp_name]
    bench   = cfg["benchmark"]

    for fold_idx, fold_cfg in exp_cfg["folds"].items():
        cfg      = load(BENCHMARK_PATH)
        fold_cfg = cfg["experiments"][exp_name]["folds"][fold_idx]

        if fold_cfg["status"] in ("pending", "failed"):
            print(f"  launching fold {fold_idx}...")
            run_fold(exp_name, fold_idx, fold_cfg, exp_cfg, bench)
        else:
            print(f"  fold {fold_idx} [{fold_cfg['status']}] — skipped")

# rebuild leaderboard
subprocess.run(["python", "scripts/make_leaderboard.py", str(BENCHMARK_PATH)])