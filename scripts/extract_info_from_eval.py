import pandas as pd
import sys
from pathlib import Path

BENCHMARK_PATH = Path(sys.argv[1])

df = pd.read_csv(BENCHMARK_PATH)

auc_cols = [c for c in df.columns if c.startswith("AUC_") and c.endswith("_mean")]

class_names = [c.replace("AUC_", "").replace("_mean", "") for c in auc_cols]

result = df[["experiment"] + auc_cols].copy()
result.columns = ["experiment"] + class_names

print(result.to_string(index=False))