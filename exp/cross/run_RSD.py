import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/OTTEHR/")
sys.path.append(f"/home/{user_id}/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/OTTEHR/competitors/")

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


# Set output path
output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/cross")
print(f"Will save outputs to {output_dir}")

# Read in the original dataframe
cross_df = pd.read_csv(os.path.join(output_dir, "admission_patient_diagnosis_ICD.csv"), index_col=None, header=0, converters={'ICD codes': literal_eval})


# Update group_name and groups to appropriate values 
trans_metric = 'RSD'
group_name = 'version'
source = 'mimic_iv'
target = 'mimic_iii'

source_count = 120
target_count = 100
type = 'cat'
iterations = 100


score_path = os.path.join(output_dir, f"{group_name}_{target}_to_{source}_{trans_metric}.csv")

maes, rmses = multi_proc_daregram_RSD(cross_df,  group_name, type, source, target, source_count, target_count, trans_metric, iteration=100)

print("rmses is:", rmses)
print("maes is:", maes)
save_results(rmses, maes, score_path)