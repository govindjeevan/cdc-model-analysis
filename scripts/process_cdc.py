CDC_ROOT = "data-processed/"

import pandas as pd
import glob
import matplotlib.pyplot as plt 
import os
from tqdm import tqdm
import numpy as np


all_files = glob.glob(os.path.join(CDC_ROOT,"*", "*.csv"))
print(len(all_files))

data = []
prev = 0 
for filename in tqdm(all_files):
    model_name = filename.split("/")[-2]
    df = pd.read_csv(filename, index_col=None, header=0)
    df = df[(df.target.str.endswith('wk ahead inc case')) & (df.type=="point") & (df.location=="US")].reset_index(drop=True).drop(['quantile'], axis=1)
    df["model"]=model_name
    data.append(df)
    
    
us_df = pd.concat(data, axis=0, ignore_index=True)
us_df["point"] = us_df["value"]
us_df["Model"] = us_df["model"]
us_df = us_df.drop(["type", "value", "model"], axis=1)
us_df.to_csv("processed_data/cdc-inc-cases.csv")