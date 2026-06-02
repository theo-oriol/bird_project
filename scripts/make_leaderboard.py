import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import os

CROSS_PATH = Path("data/cross/")

BENCHMARK_PATH = Path(sys.argv[1])


def load_fold_metrics(run_dir):
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def summarize_exp(exp_name, exp_cfg, metric, class_names, benchmark_name=None):
    """
    Reads metrics.json from every done fold.
    Returns a summary row + per-fold rows.
    """
    fold_rows = []

    for fold_idx, fold_cfg in exp_cfg["folds"].items():
        if fold_cfg["status"] != "done":
            print(f"Fold {fold_idx} is not done yet (status: {fold_cfg['status']})")
            continue
        m = load_fold_metrics(os.path.join("experiments", benchmark_name, fold_cfg["run_dir"]))
        if m is None:
            raise ValueError(f"Metrics file not found for {exp_name} fold {fold_idx}")

        row = {
            "experiment":        exp_name,
            "fold":              fold_idx,
            "status":            "done",
            "epochs_trained":    m.get("epochs_trained"),
            "val_mAP":           m.get("val_mAP"),
            "val_auc":           m.get("val_auc"),
            "train_mAP":         m.get("train_mAP"),
            "train_auc":         m.get("train_auc"),
            "run_dir":           str(Path(fold_cfg["run_dir"]).parent),
        }

        # per-class AP and AUC
        ap_per_class  = m.get("val_ap_per_class", [])
        auc_per_class = m.get("val_auc_per_class", [])
        for i, name in enumerate(class_names):
            row[f"AP_{name}"]  = ap_per_class[i]  if i < len(ap_per_class)  else None
            row[f"AUC_{name}"] = auc_per_class[i] if i < len(auc_per_class) else None

        pval_per_class  = m.get("mannwhitney_pvalue_per_class", {})
        pval_overall    = m.get("mannwhitney_pvalue_overall")

        for name in class_names:
            row[f"pval_{name}"] = pval_per_class.get(str(name))

        row["pval_overall"] = pval_overall
        
        fold_rows.append(row)   

    if not fold_rows:
        return None, []

    # ---- summary across folds ----
    def stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return None, None, None, None
        return (
            float(np.nanmean(values)),
            float(np.nanstd(values)),
            float(np.nanmin(values)),
            float(np.nanmax(values)),
        )

    summary = {
        "benchmark":      benchmark_name,
        "experiment":     exp_name,
        "fold":           "all",
        "status":         "done",
        "epochs_trained": fold_rows[0]["epochs_trained"],
        "folds_done":     len(fold_rows),
        "run_dir":        str(Path(fold_rows[0]["run_dir"])),
    }

    # mean/std/min/max for mAP and mAUC
    for key in ("val_mAP", "val_auc", "train_mAP", "train_auc"):
        mean, std, mn, mx = stats([r[key] for r in fold_rows])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"]  = std


    # per-class AP mean/std across folds
    for name in class_names:
        ap_mean, ap_std, _, _ = stats([r[f"AP_{name}"]  for r in fold_rows])
        au_mean, au_std, _, _ = stats([r[f"AUC_{name}"] for r in fold_rows])
        summary[f"AP_{name}_mean"]  = ap_mean
        summary[f"AP_{name}_std"]   = ap_std
        summary[f"AUC_{name}_mean"] = au_mean
        summary[f"AUC_{name}_std"]  = au_std

    # per-class p-value mean/std across folds
    for name in class_names:
        pv_mean, pv_std, _, _ = stats([r[f"pval_{name}"] for r in fold_rows])
        summary[f"pval_{name}_mean"] = pv_mean
        summary[f"pval_{name}_std"]  = pv_std

        pv_mean, pv_std, _, _ = stats([r["pval_overall"] for r in fold_rows])
        summary["pval_overall_mean"] = pv_mean
        summary["pval_overall_std"]  = pv_std

    
    # save summary.json next to the fold dirs
    summary_path = Path(os.path.join("experiments", summary["benchmark"], summary["run_dir"])) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({**summary, "per_fold": fold_rows}, f, indent=2)

    return summary, fold_rows


def main():
    with open(BENCHMARK_PATH) as f:
        cfg = yaml.safe_load(f)

    bench  = cfg["benchmark"]
    metric = bench["metric"]

    # load class names from labelname.json if available, else use indices
    dataset_dir = bench["dataset_dir"]
    dataset_dir = os.path.join(CROSS_PATH,dataset_dir)

    label_path = Path(dataset_dir) / "labelname.json" if dataset_dir else None
    with open(label_path, "r") as f:
        class_names = json.load(f)  
    
    if class_names is None:
        raise ValueError("No class names found in labelname.json")

    summary_rows = []
    fold_rows    = []

    for exp_name, exp_cfg in cfg["experiments"].items():
        summary, folds = summarize_exp(exp_name, exp_cfg, metric, class_names, benchmark_name=bench["name"])
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

    # ---- per-fold detail ----
    df_folds = (
        pd.DataFrame(fold_rows)
        .sort_values(["experiment", "fold"])
        .reset_index(drop=True)
    )

    # ---- save ----
    out_dir = Path("benchmarks")
    stem    = BENCHMARK_PATH.stem

    leaderboard_path = out_dir / "cross" / f"{stem}_leaderboard.csv"
    folds_path       = out_dir / "per_fold" / f"{stem}_folds.csv"

    df_summary.to_csv(leaderboard_path, index=False)
    df_folds.to_csv(folds_path, index=False)

    # ---- print summary table ----
    print(f"\n{'='*60}")
    print(f"  Benchmark : {bench['name']}")
    print(f"  Metric    : {metric}")
    print(f"  Folds done: {len(fold_rows)}")
    print(f"{'='*60}\n")

    # global metrics
    global_cols = [
        "rank", "experiment",
        "val_mAP_mean", "val_mAP_std",
        "val_auc_mean", "val_auc_std",
        "folds_done",
    ]
    print("── Global metrics ──")
    print(df_summary[global_cols].to_string(index=False))

    # per-class AP
    ap_cols  = ["experiment"] + [f"AP_{n}_mean"  for n in class_names] + [f"AP_{n}_std"  for n in class_names]
    auc_cols = ["experiment"] + [f"AUC_{n}_mean" for n in class_names] + [f"AUC_{n}_std" for n in class_names]
    print("\n── Per-class AP (mean ± std across folds) ──")
    print(df_summary[ap_cols].to_string(index=False))
    print("\n── Per-class AUC (mean ± std across folds) ──")
    print(df_summary[auc_cols].to_string(index=False))

    # per-class p-values
    pval_cols = ["experiment"] + [f"pval_{n}_mean" for n in class_names] + [f"pval_{n}_std" for n in class_names]
    print("\n── Per-class Mann-Whitney p-value (mean ± std across folds) ──")
    print(df_summary[pval_cols].to_string(index=False))

    overall_cols = ["experiment", "pval_overall_mean", "pval_overall_std"]
    print("\n── Overall p-value ──")
    print(df_summary[overall_cols].to_string(index=False))

    print(f"\nLeaderboard saved to {leaderboard_path}")
    print(f"Per-fold detail  saved to {folds_path}")


if __name__ == "__main__":
    main()