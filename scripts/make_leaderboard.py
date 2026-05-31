import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


BENCHMARK_PATH = Path(sys.argv[1])


def load_fold_metrics(run_dir):
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def summarize_exp(exp_name, exp_cfg, metric):
    """
    Reads metrics.json from every done fold.
    Returns a summary row + per-fold rows.
    """
    fold_rows = []

    for fold_idx, fold_cfg in exp_cfg["folds"].items():
        if fold_cfg["status"] != "done":
            continue
        m = load_fold_metrics(fold_cfg["run_dir"])
        if m is None:
            continue
        fold_rows.append({
            "experiment":     exp_name,
            "fold":           fold_idx,
            "status":         "done",
            "epochs_trained": m.get("epochs_trained"),
            "val_mAP":        m.get("val_mAP"),
            "val_auc":        m.get("val_auc"),
            "train_mAP":      m.get("train_mAP"),
            "train_auc":      m.get("train_auc"),
            "run_dir":        fold_cfg["run_dir"],
        })

    if not fold_rows:
        return None, []

    # ---- summary across folds ----
    vals = [r[metric] for r in fold_rows if r[metric] is not None]
    summary = {
        "experiment":          exp_name,
        "fold":                "all",
        "status":              "done",
        "epochs_trained":      fold_rows[0]["epochs_trained"],
        f"{metric}_mean":      float(np.mean(vals)),
        f"{metric}_std":       float(np.std(vals)),
        f"{metric}_min":       float(np.min(vals)),
        f"{metric}_max":       float(np.max(vals)),
        "folds_done":          len(fold_rows),
        "val_auc_mean":        float(np.mean([r["val_auc"] for r in fold_rows])),
        "run_dir":             str(Path(fold_rows[0]["run_dir"]).parent),
    }

    # save summary.json next to the fold dirs
    summary_path = Path(summary["run_dir"]) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({**summary, "per_fold": fold_rows}, f, indent=2)

    return summary, fold_rows


def main():
    with open(BENCHMARK_PATH) as f:
        cfg = yaml.safe_load(f)

    bench  = cfg["benchmark"]
    metric = bench["metric"]   # e.g. val_mAP

    summary_rows = []
    fold_rows    = []

    for exp_name, exp_cfg in cfg["experiments"].items():
        summary, folds = summarize_exp(exp_name, exp_cfg, metric)
        if summary:
            summary_rows.append(summary)
        fold_rows.extend(folds)

    if not summary_rows:
        print("No completed experiments found.")
        return

    # ---- leaderboard: one row per experiment, ranked by mean metric ----
    df_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(f"{metric}_mean", ascending=False)
        .reset_index(drop=True)
    )
    df_summary.insert(0, "rank", df_summary.index + 1)

    # ---- per-fold detail table ----
    df_folds = (
        pd.DataFrame(fold_rows)
        .sort_values(["experiment", "fold"])
        .reset_index(drop=True)
    )

    # ---- save ----
    out_dir  = BENCHMARK_PATH.parent
    stem     = BENCHMARK_PATH.stem

    leaderboard_path = out_dir / f"{stem}_leaderboard.csv"
    folds_path       = out_dir / f"{stem}_folds.csv"

    df_summary.to_csv(leaderboard_path, index=False)
    df_folds.to_csv(folds_path, index=False)

    # ---- print ----
    print(f"\n{'='*60}")
    print(f"  Benchmark : {bench['name']}")
    print(f"  Metric    : {metric}")
    print(f"  Folds done: {len(fold_rows)}")
    print(f"{'='*60}\n")
    print(df_summary[[
        "rank", "experiment", f"{metric}_mean", f"{metric}_std",
        "val_auc_mean", "folds_done"
    ]].to_string(index=False))
    print(f"\nLeaderboard saved to {leaderboard_path}")
    print(f"Per-fold detail saved to {folds_path}")


if __name__ == "__main__":
    main()