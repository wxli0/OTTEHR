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
from daregram import *
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from common import *
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time



output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/cross")
print(f"Will save outputs to {output_dir}")

""" 
Read in the original dataframe
"""
cross_df = pd.read_csv(os.path.join(output_dir, "admission_patient_diagnosis_ICD.csv"), index_col=None, header=0, converters={'ICD codes': literal_eval})

n_components = 50

suffix = None

# Update group_name and groups to appropriate values 
trans_metric = 'daregram'
group_name = 'version'
group_1 = 'mimic_iv'
group_2 = 'mimic_iii'

group_1_count = 120
group_2_count = 100
iterations = 100
type = 'cat'



score_path = os.path.join(output_dir, f"{group_name}_{group_2}_to_{group_1}_{trans_metric}.csv")


maes = []
rmses = []
for i in range(iterations):
    print("iteration:", i)
    start_time = time.time()
    selected_df = select_samples(cross_df, group_name, type, group_1, group_2, group_1_count, group_2_count)
    code_feature_name = 'ICD codes'
    label_name = 'duration'
    source_data, source_labels, target_data, target_labels = gen_code_feature_label(selected_df, group_name, type, group_1, group_2, code_feature_name, label_name)
    print("print data dimensions:", source_data.shape, source_labels.shape, target_data.shape, target_labels.shape)

    test_rmse, test_mae = run_daregram(source_data, source_labels, target_data, target_labels)
    maes.append(test_mae)
    rmses.append(test_rmse)
    print("time for one iteration is:", time.time()-start_time)

print("rmses is:", rmses)
print("maes is:", maes)
save_results(rmses, maes, score_path)