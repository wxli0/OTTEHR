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


output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/cross")
print(f"Will save outputs to {output_dir}")

""" 
Read in the original dataframe
"""
cross_df = pd.read_csv(os.path.join(output_dir, "admission_patient_diagnosis_ICD.csv"), index_col=None, header=0, converters={'ICD codes': literal_eval})



def custom_train_reps(source_features, target_features, n_components, pca_explain=False):
    """ 
    Customized training algorithm for generating target representations and source representations

    In the cross database experiments, we run PCA separately on source and target, and use Gromov Wasserstein OT to enable transfer learning

    :param bool pca_explain: print the explained variance of each components
    
    :returns: target representations, source representations
    """
    source_pca = PCA(n_components=n_components)
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


""" 
Run multiple iterations using linear regression
"""
n_components = 50

suffix = None

# Update group_name and groups to appropriate values 
group_name = 'version'
group_1 = 'mimic_iv'
group_2 = 'mimic_iii'

group_1_count = 120
group_2_count = 100

# trans_metric = 'OT'
# trans_metric = 'GWOT'

# trans_metric = 'TCA'
# trans_metric = 'MMD'
# trans_metric = 'NN'
# trans_metric = 'GFK'
trans_metric = 'CA'

        

score_path = os.path.join(output_dir, f"{group_name}_{group_2}_to_{group_1}_{trans_metric}.csv")

source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses, target_clf_maes, target_clf_mses, target_clf_rmses, \
    trans_target_maes, trans_target_mses, trans_target_rmses, label_div_scores, wa_dists, coupling_diffs, diameters, max_hs \
        = multi_proc_cts(n_components, cross_df, custom_train_reps, group_name, group_1, group_2, \
            group_1_count, group_2_count, trans_metric=trans_metric, model_func = linear_model.LinearRegression, iteration=100, equity=False, suffix=suffix)

save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, target_clf_maes, target_clf_mses, target_clf_rmses, \
    trans_target_maes, trans_target_mses, trans_target_rmses, label_div_scores, wa_dists, coupling_diffs, diameters, max_hs, score_path)
