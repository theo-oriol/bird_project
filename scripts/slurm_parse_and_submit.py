"""
Reads a benchmark YAML and submits one SLURM job per pending fold.
Each job calls scripts/train.py directly.
"""

import sys
import yaml
import subprocess
from pathlib import Path

BENCHMARK_PATH = Path(sys.argv[1])

with open(BENCHMARK_PATH) as f:
    cfg = yaml.safe_load(f)

bench      = cfg["benchmark"]
bench_name = bench["name"]

for exp_name, exp_cfg in cfg["experiments"].items():
    for fold_idx, fold_cfg in exp_cfg["folds"].items():
        if fold_cfg["status"] not in ("pending", "failed"):
            print(f"[skip] {exp_name} fold {fold_idx} [{fold_cfg['status']}]")
            continue

        run_dir  = f"experiments/{bench_name}/{exp_name}/{fold_cfg['run_dir']}"
        job_name = f"{exp_name}_f{fold_idx}"
        log_dir  = Path(run_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=phj@a100
#SBATCH --partition=gpu_p5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output={run_dir}/slurm_%j.out
#SBATCH --error={run_dir}/slurm_%j.err
#SBATCH --constraint=a100
#SBATCH --mail-user=theo.oriol@cefe.cnrs.fr

# ---- env
export CACHE_ROOT=/lustre/fswork/projects/rech/phj/uzk68zl/.cache
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HUB_DIR="$CACHE_ROOT/torch/hub"
module purge
module load arch/a100
module load pytorch-gpu/py3/2.4.0


source $HOME/.bashrc
conda activate torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $WORK/bird_project

python scripts/train.py \\
    --exp-name {exp_name} \\
    --fold {fold_idx} \\
    --run-dir {run_dir} \\
    --benchmark {BENCHMARK_PATH}
"""

        # write temp slurm script
        tmp_script = Path(f"slurm/slurm_{job_name}.slurm")
        tmp_script.write_text(slurm_script)

        result = subprocess.run(
            ["sbatch", str(tmp_script)],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"[submitted] {exp_name} fold {fold_idx} → job {job_id}  ({run_dir})")
        else:
            print(f"[error] {exp_name} fold {fold_idx}: {result.stderr.strip()}")

        tmp_script.unlink()