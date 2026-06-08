#!/bin/bash
# Usage:
#   bash scripts/slurm_submit.sh configs/benchmarks/my_benchmark.yaml
#
# Submits one SLURM job per pending fold across all experiments.

BENCHMARK=$1

if [ -z "$BENCHMARK" ]; then
    echo "Usage: bash scripts/slurm_submit.sh <benchmark_yaml>"
    exit 1
fi

python scripts/slurm_parse_and_submit.py "$BENCHMARK"