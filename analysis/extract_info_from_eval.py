import pandas as pd
import sys
from pathlib import Path

BENCHMARK_PATH = Path(sys.argv[1])

df = pd.read_csv(BENCHMARK_PATH)
if "classification" in str(BENCHMARK_PATH) :
    auc_cols = [c for c in df.columns if c.startswith("AUC_") and c.endswith("_mean")]

    class_names = [c.replace("AUC_", "").replace("_mean", "") for c in auc_cols]

    result = df[["experiment"] + auc_cols].copy()
    result.columns = ["experiment"] + class_names
    
    # print(result.to_string(index=False))
    for _, row in df.iterrows():
        print(f"\n{'='*60}")
        print(f"  {row['experiment']}")
        print(f"{'='*60}")
        for c, auc_col in zip(class_names, auc_cols):
            print(f"  Class {c:<45}| AUC {row[auc_col]:.4f}")
elif "reg" in str(BENCHMARK_PATH) : 
        auc_cols = [c for c in df.columns if c.startswith("MAE_") and c.endswith("_mean")]

        class_names = [c.replace("MAE_", "").replace("_mean", "") for c in auc_cols]

        result = df[["experiment"] + auc_cols].copy()
        result.columns = ["experiment"] + class_names
        # print(result.to_string(index=False))
        for _, row in df.iterrows():
            print(f"\n{'='*60}")
            print(f"  {row['experiment']}")
            print(f"{'='*60}")
            for c, auc_col in zip(class_names, auc_cols):
                print(f"  Class {c:<45}| MAE {row[auc_col]:.4f}")
else :
     raise ValueError("type of benchamrk unknown")