import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/OTTEHR/")


from ast import literal_eval
from common import *
from mimic_common import *
from multiprocess import Pool
import os
import ot
import ot.plot
import random
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import time

# Set output directory
output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/mimic_iii")
print(f"Will save outputs to {output_dir}")

# Read in dataframe
suffix = "with_age"
admid_diagnosis_df = pd.read_csv(os.path.join(output_dir, f"admission_patient_diagnosis_ICD_{suffix}.csv"), index_col=0, header=0, converters={'ICD codes': literal_eval})
print(admid_diagnosis_df)



def custom_train_reps(source_features, target_features, n_components, pca_explain=False):
    """ 
    Customized training algorithm for generating target representations and source representations

    :param bool pca_explain: print the explained variance of each components
    
    :returns: target representations, source representations
    """
    source_pca = PCA(n_components=n_components)
    print("source_features shape is:", source_features.shape)
    source_reps = source_pca.fit_transform(source_features)

    # Use source PCA to embed target representations (based on the assumption source and target are using the same ICD encoding system)
    target_reps = source_pca.fit_transform(target_features)

    if pca_explain:
        source_exp_var = source_pca.explained_variance_ratio_
        source_cum_sum_var = np.cumsum(source_exp_var)
        target_exp_var = source_pca.explained_variance_ratio_
        target_cum_sum_var = np.cumsum(target_exp_var)
        print("Cummulative variance explained by the source PCA is:", source_cum_sum_var[-1])
        print("Cummulative variance explained by the target PCA is:", target_cum_sum_var[-1])

    return source_reps, target_reps


# Set parameters
n_components = 50

# Update group_name and groups to appropriate values 
group_name = 'age'
# groups = [[10, 25], [25, 40], [40, 55], [55, 70], [50, 70]]

source = [50, 70]
target_groups = [[25, 45], [30, 50], [35, 55], [45, 65], [50, 70], [55, 75], [60, 80]]
source_count = 120
target_count = 100
iterations = 100

# trans_metric = 'OT' 
# trans_metric = 'TCA'
# trans_metric = 'GFK'
trans_metric = 'CA'
feature_type = 'cts'
append_features = ['age']


# Run multiple iterations using OTTEHR
# groups.reverse()

for target in target_groups:
    
    print(f"source is: {source}, target is: {target}")

    score_path = os.path.join(output_dir, f"{group_name}_{target}_to_{source}_{trans_metric}.csv")
    
    source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses, target_clf_maes, target_clf_mses, target_clf_rmses, \
        trans_target_maes, trans_target_mses, trans_target_rmses, label_div_scores, wa_dists, coupling_diffs, diameters, max_hs \
            = multi_proc(n_components, admid_diagnosis_df, custom_train_reps, group_name, feature_type, source, target, \
                source_count, target_count, trans_metric=trans_metric, model_func = linear_model.LinearRegression, \
                iteration=iterations, equity=False, append_features=append_features)

    save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, target_clf_maes, target_clf_mses, target_clf_rmses, \
        trans_target_maes, trans_target_mses, trans_target_rmses, label_div_scores, wa_dists, coupling_diffs, diameters, max_hs, score_path)
