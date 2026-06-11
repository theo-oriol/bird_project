import pandas as pd
from pathlib import Path
import sys
import os 

MODEL_PATH = Path(sys.argv[1])


df1 = pd.read_csv(os.path.join(MODEL_PATH,"fold_0","val_predictions.csv"))
df2 = pd.read_csv(os.path.join(MODEL_PATH,"fold_1","val_predictions.csv"))
df3 = pd.read_csv(os.path.join(MODEL_PATH,"fold_2","val_predictions.csv"))

df = pd.concat([df1, df2, df3], ignore_index=True)
name = str(MODEL_PATH.parent).split("/")[-1]
name = f"{name}.csv"
path = os.path.join(MODEL_PATH,name)

df.to_csv(path , index=False)