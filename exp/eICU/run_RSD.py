import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/projects/OTTEHR/")
sys.path.append(f"/home/{user_id}/projects/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/projects/OTTEHR/competitors/")

from ast import literal_eval
from mimic_common import *
import numpy as np
import os
import ot
import pandas as pd
from RSD import *
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from common import *
import time 
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim



data = 'eICU'
data_dir = os.path.join(os.path.expanduser("~"), f"projects/OTTEHR/data/{data}")
output_dir = os.path.join(os.path.expanduser("~"), f"projects/OTTEHR/outputs/{data}")
print(f"Will save outputs to {output_dir}")

""" 
Read in the original dataframe
"""
df = pd.read_csv(os.path.join(data_dir, "admission_patient_diagnosis_ICD.csv"), index_col=None, header=0, converters={'ICD codes': literal_eval})

# Update group_name and groups to appropriate values 
trans_metric = 'RSD'
group_name = 'hospitalid'
groups = [420, 264, 243, 338, 73, 458, 167, 443, 208, 300]

source_count = 120
target_count = 100
type = 'cat'
iterations = 100

for group_1 in groups:
    for group_2 in groups:
        if group_1 == group_2:
            continue
        score_path = os.path.join(output_dir, f"{group_name}_{group_2}_to_{group_1}_{trans_metric}.csv")

        maes, rmses = multi_proc_daregram_RSD(df,  group_name, type, group_1, group_2, source_count, target_count, trans_metric, iteration=100)

        print("rmses is:", rmses)
        print("maes is:", maes)
        save_results(rmses, maes, score_path)