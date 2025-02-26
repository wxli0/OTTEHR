import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/projects/OTTEHR/")
sys.path.append(f"/home/{user_id}/projects/OTTEHR/competitors")
sys.path.append(f"/home/{user_id}/projects/unbalanced_gromov_wasserstein/")

from ast import literal_eval
from datetime import datetime
import copy
from CA import *
from common import *
from collections import Counter
from daregram import *
from GFK import *
from MMD import *
import matplotlib.pyplot as plt
from multiprocess import Pool
import numpy as np
from NN import *
import os
import random
from RSD import *
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
    f1_score, mean_absolute_error, mean_squared_error, \
    mutual_info_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.stats import entropy
from TCA import *
from transfertools.models import TCA
import torch
import time

from unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn
from unbalancedgw._vanilla_utils import ugw_cost
from unbalancedgw.utils import generate_measure

mimic_output_dir = f"/home/{user_id}/projects/OTTEHR/outputs/mimic_iii"
mimic_data_dir = f"/home/{user_id}/projects/OTTEHR/data/mimic_iii"
cross_output_dir = f"/home/{user_id}/projects/OTTEHR/outputs/cross"
cross_data_dir = f"/home/{user_id}/projects/OTTEHR/data/cross"
eICU_output_dir = f"/home/{user_id}/projects/OTTEHR/outputs/eICU"
eICU_data_dir = f"/home/{user_id}/projects/OTTEHR/data/eICU"


def find_unique_code(df, ICD_name = 'ICD codes'):
    """ 
    Find all unique codes in df, and returns code to index dictionary, \
        and the number of unique codes
    :param str ICD_col_name: name of the ICD code column
    """

    all_codes = list(df[ICD_name])
    all_codes = [item for sublist in all_codes for item in sublist]
    unique_codes = list(set(all_codes))
    unique_code_dict = {}
    for index, code in enumerate(unique_codes):
        unique_code_dict[code] = index
    return unique_code_dict, len(unique_codes) 
    

# def gen_features_labels(df, group_name, source, target, label_code, ICD_col_name = 'ICD codes'):
#     """ 
#     Generate source features, source labels, target features and target labels from dataframe df

#     """

#     unique_code_dict, num_codes = find_unique_code(df)
#     # print("number of unique code is:", num_codes)
#     # print("label code in unique_code_dict is:", unique_code_dict[label_code])

#     source_df = df.loc[df[group_name] == source]
#     target_df = df.loc[df[group_name] == target]

#     # Prepare source
#     source_features = np.empty(shape=[source_df.shape[0], num_codes])
#     feature_index = 0
#     for _, row in source_df.iterrows():
#         code_ind = np.zeros(num_codes)
#         for code in row[ICD_col_name]:
#             if code != label_code:
#                 code_ind[unique_code_dict[code]] += 1
#         source_features[feature_index] = code_ind
#         feature_index += 1
#     source_labels = np.array(list(source_df['label']))

#     # Prepare target
#     target_features = np.empty(shape=[target_df.shape[0], num_codes])
#     feature_index = 0
#     for _, row in target_df.iterrows():
#         code_ind = np.zeros(num_codes)
#         for code in row[ICD_col_name]:
#             if code != label_code:
#                 code_ind[unique_code_dict[code]] += 1
#         target_features[feature_index] = code_ind
#         feature_index += 1
#     target_labels = np.array(list(target_df['label']))

#     return source_features, source_labels, target_features, target_labels


def gen_feature(df, group_name, source, target, feature_code_name):
    """ 
    Generate source features and target features from dataframe df
    :param str feature_code_name: name of the input. For the experiments on ICD codes vs. duration, the input_name is 'ICD codes'

    """
    unique_code_dict, num_codes = find_unique_code(df, ICD_name = feature_code_name)

    source_df = df.loc[df[group_name] == source]
    target_df = df.loc[df[group_name] == target]

    # Prepare source
    source_features = np.empty(shape=[source_df.shape[0], num_codes])
    feature_index = 0
    for _, row in source_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row[feature_code_name]:
            code_ind[unique_code_dict[code]] += 1
        source_features[feature_index] = code_ind
        feature_index += 1

    # Prepare target
    target_features = np.empty(shape=[target_df.shape[0], num_codes])
    feature_index = 0
    for _, row in target_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row[feature_code_name]:
            code_ind[unique_code_dict[code]] += 1
        target_features[feature_index] = code_ind
        feature_index += 1
    return source_features, target_features


def gen_code_feature_label(df, group_name, type, source, target, feature_code_name, label_name, append_features=[]):
    """ 
    Generate source features, source durations (continuous response), \
        target features and target durations (continuous response) from dataframe df
    :param str group_name: the continuous/categorical feature name to divide the population
    :param str type: cat (categorical) or cts (continuous)        
    :param str feature_code_name: name of the input. For the experiments on ICD codes vs. duration, the input_name is 'ICD codes'
    :param str label_name: name of the label/response. For the experiments on ICD codes vs. duration, the label_name is 'duration'.
    :param array append_feature: Other features to append to the feature arrays. For example, [age]. Empty array by default.
    """

    unique_code_dict, num_codes = find_unique_code(df, ICD_name = feature_code_name)
    print("number of codes is:", num_codes)

    source_df = None
    target_df = None
    if type == 'cat':
        source_df = df.loc[df[group_name] == source]
        target_df = df.loc[df[group_name] == target]
    elif type == 'cts':
        source_df = df[(df[group_name] >= source[0]) & (df[group_name] <= source[1])]
        target_df = df[(df[group_name] >= target[0]) & (df[group_name] <= target[1])]


    # Prepare source
    source_features = np.empty(shape=[source_df.shape[0], num_codes+len(append_features)])
    feature_index = 0
    for _, row in source_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row[feature_code_name]:
            code_ind[unique_code_dict[code]] += 1
        combined_info = np.append(code_ind, row[append_features].values)
        source_features[feature_index] = combined_info
        feature_index += 1

    source_labels = np.array(list(source_df[label_name]))

    # Prepare target
    target_features = np.empty(shape=[target_df.shape[0], num_codes+len(append_features)])
    feature_index = 0
    for _, row in target_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row[feature_code_name]:
            code_ind[unique_code_dict[code]] += 1
        combined_info = np.append(code_ind, row[append_features].values)
        target_features[feature_index] = combined_info
        feature_index += 1
        
    target_labels = np.array(list(target_df[label_name]))

    return source_features, source_labels, target_features, target_labels


def select_samples(df, group_name, type, source, target, source_count, target_count):
    """ 
    Select rows in the dataframe df by source and target constraints 

    :param Dataframe df: the dataframe to select samples from.
    :param str group_name: the continuous/categorical feature name to divide the population
    :param str type: cat (categorical) or cts (continuous)
    :param str source: the source type for categorical type, or \
    :param [float, float] source: the range of the continuous feature for the source population for continuous type
    :param str target: the target type for categorical type, or \
    :param [float, float] target: the range of the continuous feature for the target population for continuous type 
    :param int source_count: the number of samples for the source
    :param int target_count: the number of samples for the target

    :returns:
        - the selected dataframe
    """

    df_copy = copy.deepcopy(df)
    
    source_indices = []
    target_indices = []
    other_indices = []
    if type == 'cat':
        for index, row in df_copy.iterrows():
            if row[group_name] == source:
                source_indices.append(index)
            elif row[group_name] == target:
                target_indices.append(index)
            else:
                other_indices.append(index)
    elif type == 'cts':
        for index, row in df_copy.iterrows():
            if (source[0] <= row[group_name] <= source[1] or target[0] <= row[group_name] <= target[1]):
                if source[0] <= row[group_name] <= source[1]:
                    source_indices.append(index)
                if target[0] <= row[group_name] <= target[1]:
                    target_indices.append(index)
            else:
                other_indices.append(index)
    
    # indices to delete from the dataframe

    print("number of target indices is:", len(target_indices), 'target_count is:', target_count)
    print("number of source indices is:", len(source_indices), 'source_count is:', source_count)

    selected_target_indices = random.sample(target_indices, target_count)
    selected_source_indices = random.sample(source_indices, source_count)
    selected_target_indices.extend(selected_source_indices)

    df_copy = df_copy.loc[selected_target_indices]

    return df_copy


def select_drug_samples(df, group_name, drug, source, target, source_count = None, target_count = None):
    """ 
    Select rows in the dataframe df for source and target samples, return the results in \
        a format that is compatible to Taskesen et al's statistical test 
    
    :param df Dataframe: the dataframe to select samples from.
    :param str drug: drug name, used for preparing the labels array
    :param str source: source group name, e.g., White
    :param str target: target group name, e.g., Black
    :param int source_count: if specified, the number of samples for the source. \
        Otherwise, delete all rows that do not belong to target or source
    :param int target_count: if specified, the number of samples for the target. \
        Otherwise, delete all rows that do not belong to target or source
    """
    df_copy = copy.deepcopy(df)
    
    source_indices = []
    target_indices = []
    other_indices = []
    for index, row in df_copy.iterrows():
        if row[group_name] == source:
            source_indices.append(index)
        elif row[group_name] == target:
            target_indices.append(index)
        else:
            other_indices.append(index)

    
    # indices to delete from the dataframe
    if source_count is not None:
        delete_source_indices = random.sample(source_indices, len(source_indices)-source_count)
        delete_target_indices = random.sample(target_indices, len(target_indices)-target_count)

        other_indices.extend(delete_source_indices)
        other_indices.extend(delete_target_indices)

    df_copy = df_copy.drop(other_indices, axis=0, inplace=False)

    # prepare labels array, indicator of label attributes and \
    # sensitive_labels array: indicator of protected attributes
    labels = []
    sensitive_labels = []
    for _, row in df_copy.iterrows():
        if drug in row['drug']:
            labels.append(1)
        else:
            labels.append(0)

        if row[group_name] == source:
            sensitive_labels.append(1)
        elif row[group_name] == target:
            sensitive_labels.append(0)

    return df_copy, labels, sensitive_labels


def select_df_binary(df, group_name, source, target, source_count, target_count, label_code, input_name):
    """ 
    Select row in the dataframe df with balanced number of labels for males and females
    Specifically, we want to reduce the number of rows with label 0 for males and females

    :param Dataframe df: the dataframe to select samples with label 0 and label 1
    :param str group_name: the criteria for dividing groups, e.g., "gender", "adm_type"
    :param str source: group 1 label for group_name
    :param str target: group 2 label for group_name
    :param int source_count: the number of samples with label 1s and label 0s for target (male). 
    :param int target_count: the number of samples with label 1s and label 0s for source (female). 
    :param str label_code: the ICD code for determining labels. This code should be removed from ICD codes.
    :param str input_name: name of the input. For the experiments on predicting the missing ICD code, the input_name is 'ICD codes'



    :returns:
        - the selected dataframe
    """
    print(f"label_code is {label_code}")
    # print(f"label_code is {label_code}, group 1 is {source}, group 2 is {target}")

    # select samples based on counts
    female_1_indices = []
    female_0_indices = []
    male_1_indices = []
    male_0_indices = []

    # generate label column based on label_code
    df_copy = copy.deepcopy(df)
    if 'label' in df_copy.columns:
        df_copy = df_copy.drop(['label'], axis=1)
    labels = []
    for index, row in df_copy.iterrows():
        if label_code in row[input_name]:
            labels.append(1)
        else:
            labels.append(0)
    df_copy['label'] = labels

    for index, row in df_copy.iterrows():
        if row['label'] == 0 and row[group_name] == source:
            female_0_indices.append(index)
        elif row['label'] == 0 and row[group_name] == target:
            male_0_indices.append(index)
        elif row['label'] == 1 and row[group_name] == source:
            female_1_indices.append(index)
        elif row['label'] == 1 and row[group_name] == target:
            male_1_indices.append(index)
    print("group 1 with label 0 length is:", len(female_0_indices))
    print("group 2 with label 0 length is:", len(male_0_indices))
    print("group 1 with label 1 length is:", len(female_1_indices))
    print("group 2 with label 1 length is:", len(male_1_indices))
    
    # indices to delete from the dataframe
    # sample the same number of label 0s and label 1s
    delete_female_0_indices = random.sample(female_0_indices, len(female_0_indices)-target_count)
    delete_male_0_indices = random.sample(male_0_indices, len(male_0_indices)-source_count)
    
    delete_female_1_indices = random.sample(female_1_indices, len(female_1_indices)-target_count)
    delete_male_1_indices = random.sample(male_1_indices, len(male_1_indices)-source_count)

    delete_female_0_indices.extend(delete_male_0_indices)
    delete_female_0_indices.extend(delete_female_1_indices)
    delete_female_0_indices.extend(delete_male_1_indices)
    
    df_copy = df_copy.drop(delete_female_0_indices, axis=0, inplace=False)
    
    return df_copy


def train_model(reps, labels, model_func):
    """ 
    Trains a model using reps and labels and returns the model
    """
    clf = model_func()
    if 'SVR' in str(model_func):
        clf = model_func(kernel='linear')
    clf.fit(reps, labels)
    return clf


def compute_label_div_score(source_reps, source_labels, target_reps, target_labels, model_func):
    """ 
    Computes the transfer score using the third term in Theorem 1: \mathbb{E}_{x\sim D_T}[|f_T(x)-f_S(x)|]
    where f_T is the labeling function for target domain, f_S is the labeling function for the source domain

    :param function model_func: the function to model the relationship between transported source reps and source labels.
        Linear regression can be the simplest, can be more complicated. 
    """

    source_model = train_model(source_reps, source_labels, model_func)
    source_pred = source_model.predict(target_reps)

    target_model = train_model(target_reps, target_labels, model_func)
    target_pred = target_model.predict(target_reps)

    diffs = np.subtract(source_pred, target_pred)
    diffs = [abs(diff) for diff in diffs]

    return np.sum(diffs)



def entire_proc_binary(n_components, group_name, source, target, label_code, full_df, custom_train_reps, trans_metric, \
                       model_func = linear_model.LogisticRegression, source_count = 120, target_count = 100, pca_explain=False):
    """
    Wrap up the entire procedure

    :param int n_components: the number of components for PCA learning
    :param str group_name: group name to divide groups
    :param str source: group 1 name, the target in transfer learning (e.g. female)
    :param str target: group 2 name, the source in transfer learning (e.g. male)
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param function model_func: the function to model the relationship between target reps and target labels
    :param str trans_metric: transporting metric, NN, TCA, MMD or OT
    :param int source_count: the number of samples with label 1s and label 0s for target (male)
    :param int target_count: the number of samples with label 1s and label 0s for source (female)
    :param bool pca_explain: print the variance explained by the PCA, if True. Default False


    :returns accuracy statistics and optimized Wasserstein distance
    """
    
    selected_df = select_df_binary(full_df, group_name, source, target, \
                source_count, target_count, label_code)

    source_features, source_labels, target_features, target_labels = gen_code_feature_label(selected_df, group_name, source, target, label_code)

    source_reps = None
    target_reps = None
    if trans_metric != 'TCA':
        source_reps, target_reps = custom_train_reps(source_features, target_features, n_components, pca_explain=pca_explain)

    wa_dist = None
    coupling = None 

    if trans_metric == 'GWOT':
         # coupling will be useful if we want to study equity
        trans_target_reps, coupling, wa_dist = trans_GWOT(target_reps, source_reps)
    elif trans_metric == 'MMD':
        trans_target_reps = trans_MMD(target_reps, source_reps)
    elif trans_metric == 'TCA':
        source_reps, target_reps, trans_target_reps = TCA(source_features, target_features, n_components=n_components)


    
    source_model = train_model(source_reps, source_labels, model_func)
    source_preds = source_model.predict(source_reps)
    target_preds = source_model.predict(target_reps)
    trans_target_preds = source_model.predict(trans_target_reps)

    # Compute accuracies
    source_accuracy = accuracy_score(source_labels, source_preds)
    source_precision = precision_score(source_labels, source_preds)
    source_recall = recall_score(source_labels, source_preds)
    source_f1 = f1_score(source_labels, source_preds)

    target_accuracy = accuracy_score(target_labels, target_preds)
    target_precision = precision_score(target_labels, target_preds)
    target_recall = recall_score(target_labels, target_preds)
    target_f1 = f1_score(target_labels, target_preds)

    trans_target_accuracy = accuracy_score(target_labels, trans_target_preds)
    trans_target_precision = precision_score(target_labels, trans_target_preds)
    trans_target_recall = recall_score(target_labels, trans_target_preds)
    trans_target_f1 = f1_score(target_labels, trans_target_preds)
        
    transfer_score = compute_label_div_score(source_reps, source_labels, target_reps, target_labels, model_func)

    return source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1,\
        transfer_score, wa_dist



def entire_proc(n_components, full_df, custom_train_reps, model_func, trans_metric, \
                    group_name, type, source, target, source_count = 120, \
                    target_count = 100, pca_explain=False, equity=False, suffix=None, \
                    input_name = 'ICD codes', output_name = 'duration', append_features = []):
    """
    Wrap up the entire procedure for transportation method being OT, GWOT, TCA, CA, GFK when the feature to divide groups are categorical 

    :param int n_components: the number of components for PCA learning
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param function model_func: the function the model the relationship between target representations and target response
    :param str trans_metric: transport metric, OT, GWOT, TCA, CA, GFK
    :param str group_name: the group name to divide populations
    :param str type: the type of the group name, cat (categorical) or cts (continuous), for example, cat for insurance, cts for age 
    :param int male_count: the number of samples with label 1s and label 0s for target (male). Default 120.
    :param int female_count: the number of samples with label 1s and label 0s for source (female). Default 100.
    :param bool pca_explain: print the variance explained by the PCA, if True. Default False.
    :param bool equity: track differences in predicted values versus ground-truth values, to prepare data to check for equity
    :param str suffix: suffix added to the equity file
    :param str input_name: the feature name of model_func (the name of the ICD code column)
    :param str output_name: the output name of the model_func
    :param np.array append_features: features other than the input_name to be appended to the input, e.g., [age]

    """
    
    selected_df = select_samples(full_df, group_name, type, source, target, source_count=source_count, target_count=target_count)

    source_features, source_labels, target_features, target_labels = \
        gen_code_feature_label(selected_df, group_name, type, source, target, input_name, output_name, append_features)


    source_reps = None
    target_reps = None
    # if trans_metric != 'TCA':
    source_reps, target_reps = custom_train_reps(source_features, target_features, n_components, pca_explain=pca_explain)

    trans_target_reps = None
    coupling = None
    wa_dist = None
    label_div_score = None
    coupling_diff = None
    max_h = None
    diameter = None 
    
    # Computing the coupling is common to both tasks: 
    # (1) OTTEHR - mitigate bias in the regressor (2) bias detection between two subpopulations
    if trans_metric == 'GWOT': # Unbalanced Gromov Wasserstein OT
        trans_target_reps, coupling, wa_dist = trans_GWOT(target_reps, source_reps)
    elif trans_metric == 'GFK': # Geodesic flow kernel minimization 
        source_reps, trans_target_reps = trans_GFK(source_reps, target_reps)
    elif trans_metric == 'OT': # Unbalanced OT
        trans_target_reps, wa_dist, coupling, diameter = trans_UOT(target_reps, source_reps)
    elif trans_metric == 'CA': # Correlation alignment
        trans_target_reps = trans_CA(target_reps, source_reps)
    elif trans_metric == 'TCA': # Transfer component analysis
        source_reps, target_reps, trans_target_reps = trans_TCA(source_features, target_features, n_components=n_components)

    
    clf = train_model(source_reps, source_labels, model_func) 
    source_preds = clf.predict(source_reps)
    target_preds = clf.predict(target_reps)
    trans_target_preds = clf.predict(trans_target_reps)
    target_clf = train_model(target_reps, target_labels, model_func)
    target_clf_preds = target_clf.predict(target_reps)

    if equity:
        # compute transported target using the model, used for studying the bias 
        # trans_target_mappings = np.matmul(coupling, source_labels)
        # trans_target_mappings = np.multiply(trans_target_mappings, target_count)
        target_equity_path = os.path.join(mimic_output_dir, f"{group_name}_{target}_to_{source}_{trans_metric}_equity.csv")
        if suffix is not None:
            target_equity_path = os.path.join(mimic_output_dir, f"{group_name}_{target}_to_{source}_{trans_metric}_{suffix}_equity.csv")
        target_equity_df = pd.read_csv(target_equity_path, header=0, index_col = None)

        # target_diffs = np.divide(target_preds - target_labels, target_labels)
        target_codes = list(selected_df.loc[selected_df[group_name] == target][input_name])
        target_new_df = pd.DataFrame([target_labels, target_preds, trans_target_preds, target_codes]).transpose()
        target_new_df.columns = target_equity_df.columns
        target_equity_df = pd.concat([target_equity_df, target_new_df], ignore_index=True)
        target_equity_df.to_csv(target_equity_path, index=False, header=True)
    
    if trans_metric == 'OT':
        # Compute the difference between the transported mass and the original mass
        coupling_target =  coupling.sum(axis=1)
        coupling_diff = np.subtract(np.full(coupling_target.shape, 1/target_count), coupling_target)
        coupling_diff = np.sum(np.absolute(coupling_diff))
        label_div_score = compute_label_div_score(source_reps, source_labels, target_reps, target_labels, model_func)

        # Compute the maximum |||h||| (triple norm of h)
        target_pred_norms = np.linalg.norm(target_preds, axis=0)
        max_h = np.max(np.divide(target_preds, target_pred_norms))

        
    source_mae = metrics.mean_absolute_error(source_labels, source_preds)
    source_mse = mean_squared_error(source_labels, source_preds)
    source_rmse = np.sqrt(metrics.mean_squared_error(source_labels, source_preds))
    target_mae = metrics.mean_absolute_error(target_labels, target_preds)
    target_mse = mean_squared_error(target_labels, target_preds)
    target_rmse = np.sqrt(metrics.mean_squared_error(target_labels, target_preds))
    target_clf_mae = metrics.mean_absolute_error(target_labels, target_clf_preds)
    target_clf_mse = mean_squared_error(target_labels, target_clf_preds)
    target_clf_rmse = np.sqrt(metrics.mean_squared_error(target_labels, target_clf_preds))
    trans_target_mae = metrics.mean_absolute_error(target_labels, trans_target_preds)
    trans_target_mse = mean_squared_error(target_labels, trans_target_preds)
    trans_target_rmse = np.sqrt(metrics.mean_squared_error(target_labels, trans_target_preds))

    return source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse, target_clf_mae, target_clf_mse, target_clf_rmse, \
        trans_target_mae, trans_target_mse, trans_target_rmse, label_div_score, wa_dist, coupling_diff, diameter, max_h


def multi_proc_binary(score_path, n_components, label_code, full_df, custom_train_reps, \
               male_count, female_count, model_func, iteration=20, max_iter = None):
    """ 
    Run the entire_proc function multiple times (iteration) for binary responses

    :param str score_path: the path to save results
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param int n_components: the number of components in PCA to learn representations
    :param int male_count: the number of samples with label 1s and label 0s for target (male)
    :param int female_count: the number of samples with label 1s and label 0s for source (female)
    :param int iteration: the number of iterations (repetitions)
    :param int max_iter: maximum number of iterations for OT
    """

    random.seed(0)
    res = np.empty(shape=[iteration, 12])
    for i in range(iteration):
        print("iteration:", i)
        cur_res = entire_proc_binary(n_components, label_code, full_df, custom_train_reps, male_count, female_count, max_iter=max_iter, model_func=model_func)
        res[i] = cur_res
    res_df = pd.DataFrame(res, \
                          columns = ['source_accuracy', 'source_precision', 'source_recall', 'source_f1', \
                                     'target_accuracy', 'target_precision', 'target_recall', 'target_f1', \
                                        'trans_target_accuracy', 'trans_target_precision', 'trans_target_recall', 'trans_target_f1'])
    res_df.to_csv(score_path, index=False, header=True)
    return res


def multi_proc(n_components, full_df, custom_train_reps, \
               group_name, type, source, target, source_count, target_count, \
                trans_metric, model_func = linear_model.LinearRegression, \
                iteration=20, equity=False, suffix=None, append_features = []):
    """ 
    Run the entire_proc function multiple times (iteration) 

    :param str score_path: the path to save results
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param int n_components: the number of components in PCA to learn representations
    :param str group_name: the name of the dividing group
    :param str type: the type of the group name, cat (categorical) or cts (continuous), for example, cat for insurance, cts for age 
    :param str source: value for group 1
    :param str target: value for group 2
    :param int source_count: the number of samples with label 1s and label 0s for target (male)
    :param int target_count: the number of samples with label 1s and label 0s for source (female)
    :param str trans_metric: transport metric, OT, MMD, TCA or NN
    :param int iteration: the number of iterations (repetitions)
    :param bool equity: prepare statistics for equity
    :param array append_feature: Other features to append to the feature arrays. For example, [age]. Empty array by default.

    """
    random.seed(0)
    source_maes = []
    source_mses = []
    source_rmses = [] 
    target_maes = []
    target_mses = [] 
    target_rmses = [] 
    target_clf_maes = [] 
    target_clf_mses = [] 
    target_clf_rmses = []
    trans_target_maes = []
    trans_target_mses = []
    trans_target_rmses = []
    label_div_scores = []
    wa_dists = []
    coupling_diffs = []
    diameters = []
    max_hs = []

    if equity:
        target_equity_df = pd.DataFrame(columns=['target_label', 'target_pred_label', 'trans_target_pred_label',  'target_codes'])
        target_equity_path = os.path.join(mimic_output_dir, f"{group_name}_{target}_to_{source}_{trans_metric}_equity.csv")
        if suffix is not None:
            target_equity_path = os.path.join(mimic_output_dir, f"{group_name}_{target}_to_{source}_{trans_metric}_{suffix}_equity.csv")
        target_equity_df.to_csv(target_equity_path, header=True, index=False)

    for i in range(iteration):
        print("iteration:", i)
        source_mae = None
        source_mse = None
        source_rmse = None 
        target_mae = None
        target_mse = None
        target_rmse = None 
        trans_target_mae = None
        trans_target_mse = None
        trans_target_rmse = None
        target_clf_mae = None 
        target_clf_mse = None
        target_clf_rmse = None

        start_time = time.time()
        source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse, target_clf_mae, target_clf_mse, target_clf_rmse, \
            trans_target_mae, trans_target_mse, trans_target_rmse, label_div_score, wa_dist, coupling_diff, diameter, max_h = \
                entire_proc(n_components, full_df, custom_train_reps, model_func, trans_metric, \
                    group_name, type, source, target, source_count, target_count, equity=equity, suffix=suffix, append_features = append_features)
        print("one iteration time is:", time.time()-start_time)
            
        source_maes.append(source_mae)
        source_mses.append(source_mse)
        source_rmses.append(source_rmse)
        target_maes.append(target_mae)
        target_mses.append(target_mse)
        target_rmses.append(target_rmse)
        target_clf_maes.append(target_clf_mae)
        target_clf_mses.append(target_clf_mse)
        target_clf_rmses.append(target_clf_rmse)
        trans_target_maes.append(trans_target_mae)
        trans_target_mses.append(trans_target_mse)
        trans_target_rmses.append(trans_target_rmse)
        wa_dists.append(wa_dist)
        coupling_diffs.append(coupling_diff)
        label_div_scores.append(label_div_score)
        diameters.append(diameter)
        max_hs.append(max_h)
        
    return source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses, target_clf_maes, target_clf_mses, target_clf_rmses, \
        trans_target_maes, trans_target_mses, trans_target_rmses, label_div_scores, \
        wa_dists, coupling_diffs, diameters, max_hs


def multi_proc_daregram_RSD(full_df,  group_name, type, source, target, source_count, target_count, trans_metric, iteration=20):
        maes = []
        rmses = []
        for i in range(iteration):
            print("iteration:", i)
            start_time = time.time()
            selected_df = select_samples(full_df, group_name, type, source, target, source_count, target_count)
            code_feature_name = 'ICD codes'
            label_name = 'duration'
            source_data, source_labels, target_data, target_labels = gen_code_feature_label(selected_df, group_name, type, source, target, code_feature_name, label_name)
            print("print data dimensions:", source_data.shape, source_labels.shape, target_data.shape, target_labels.shape)
            test_rmse = None,
            test_mae = None
            if trans_metric == 'RSD':
                test_rmse, test_mae = run_RSD(source_data, source_labels, target_data, target_labels)
            elif trans_metric == 'daregram':
                test_rmse, test_mae = run_daregram(source_data, source_labels, target_data, target_labels)
            maes.append(test_mae)
            rmses.append(test_rmse)
            print("time for one iteration is:", time.time()-start_time)
        return maes, rmses



def build_maps(admission_file, diagnosis_file, patient_file):
    """ 
    Building pid-admission mapping, pid-gender mapping, admission-date mapping, admission-code mapping and pid-sorted visits mapping
    :param str admission_file: path to ADMISSIONS.csv
    :param str diagnosis_file: path to DIAGNOSES_ICD.csv
    :param str patient_file: path to PATIENTS.csv
    """

    # Building pid-admissions mapping and pid-date mapping
    pid_adms = {}
    adm_date = {}
    infd = open(admission_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admid = int(tokens[2])
        adm_time = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        adm_date[admid] = adm_time
        if pid in pid_adms: pid_adms[pid].append(admid)
        else: pid_adms[pid] = [admid]
    infd.close()

    # Bulding admission-codes mapping
    admid_codes = {}
    infd = open(diagnosis_file, 'r')
    infd.readline()
    for line in infd: # read ADMISSIONS.CSV in order
        tokens = line.strip().split(',')
        admid = int(tokens[2])
        code = tokens[4][1:-1]

        if admid in admid_codes: 
            admid_codes[admid].append(code)
        else: 
            admid_codes[admid] = [code]
    infd.close()
    
    # Building pid-sorted visits mapping, a visit consists of (time, codes)
    pid_visits = {}
    for pid, adms in pid_adms.items():
        if len(adms) < 2: continue

        # sort by date
        sorted_visit = sorted([(adm_date[admid], admid_codes[admid]) for admid in adms])
        pid_visits[pid] = sorted_visit
    
    pid_gender = {}
    infd = open(patient_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        gender = str(tokens[2])
        pid_gender[pid] = gender[1] # remove quotes
    infd.close()
                
    return pid_adms, pid_gender, adm_date, admid_codes, pid_visits


def custom_train_reps_default(source_features, target_features, n_components, pca_explain=False):
    """ 
    Customized training algorithm for generating target representations and source representations

    :param bool pca_explain: print the explained variance of each components
    
    :returns: target representations, source representations
    """
    source_pca = PCA(n_components=n_components)
    source_reps = source_pca.fit_transform(source_features)

    target_pca = PCA(n_components=n_components)
    target_reps = target_pca.fit_transform(target_features)

    if pca_explain:
        source_exp_var = source_pca.explained_variance_ratio_
        source_cum_sum_var = np.cumsum(source_exp_var)
        target_exp_var = target_pca.explained_variance_ratio_
        target_cum_sum_var = np.cumsum(target_exp_var)
        print("Cummulative variance explained by the source PCA is:", source_cum_sum_var[-1])
        print("Cummulative variance explained by the target PCA is:", target_cum_sum_var[-1])

    return source_reps, target_reps


def decide_ICD_chapter(code):
    """ 
    Decide ICD code chapter based on 
    https://en.wikipedia.org/wiki/List_of_ICD-9_codes_E_and_V_codes:_external_causes_of_injury_and_supplemental_classification 

    Rules:
    Chapter	Block	Title
    1	001–139	Infectious and Parasitic Diseases
    2	140–239	Neoplasms
    3	240–279	Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders
    4	280–289	Diseases of the Blood and Blood-forming Organs
    5	290–319	Mental Disorders
    6	320–389	Diseases of the Nervous System and Sense Organs
    7	390–459	Diseases of the Circulatory System
    8	460–519	Diseases of the Respiratory System
    9	520–579	Diseases of the Digestive System
    10	580–629	Diseases of the Genitourinary System
    11	630–679	Complications of Pregnancy, Childbirth, and the Puerperium
    12	680–709	Diseases of the Skin and Subcutaneous Tissue
    13	710–739	Diseases of the Musculoskeletal System and Connective Tissue
    14	740–759	Congenital Anomalies
    15	760–779	Certain Conditions originating in the Perinatal Period
    16	780–799	Symptoms, Signs and Ill-defined Conditions
    17	800–999	Injury and Poisoning
    18  E800–E999   Supplementary Classification of External Causes of Injury and Poisoning
    19  V01–V82	Supplementary Classification of Factors influencing Health Status and Contact with Health Services
    20  M8000–M9970	Morphology of Neoplasms
    """

    block = code.split(".")[0]
    if block.startswith("E"):
        return 18, "Supplementary Classification of External Causes of Injury and Poisoning"
    if block.startswith("V"):
        return 19, "Supplementary Classification of Factors influencing Health Status and Contact with Health Services"
    if block.startswith("M"):
        return 20, "Morphology of Neoplasms"
    
    # otherwise, numerical code
    block_segs = [1, 140, 240, 280, 290, 320, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 1000]
    titles = ["Infectious and Parasitic Diseases", 
              "Neoplasms", 
              "Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders", \
        "Diseases of the Blood and Blood-forming Organs", 
        "Mental Disorders", 
        "Diseases of the Nervous System and Sense Organs", \
        "Diseases of the Circulatory System", 
        "Diseases of the Respiratory System", 
        "Diseases of the Digestive System", \
        "Diseases of the Genitourinary System", 
        "Complications of Pregnancy, Childbirth, and the Puerperium", 
        "Diseases of the Skin and Subcutaneous Tissue",\
        "Diseases of the Musculoskeletal System and Connective Tissue", 
        "Congenital Anomalies", 
        "Certain Conditions originating in the Perinatal Period", \
        "Symptoms, Signs and Ill-defined Conditions", 
        "Injury and Poisoning" ]
    block = int(block)
    for chapter_index in range(len(block_segs)):
        left = block_segs[chapter_index]
        right = block_segs[chapter_index+1]
        if left <= block and block < right:
            return chapter_index+1, titles[chapter_index]


def decide_all_ICD_chapters(codes):
    chapters = []
    for code in codes:
        chapter, _ = decide_ICD_chapter(code)
        chapters.append(chapter)
    return chapters


def get_label_codes():
    """ 
    Get label codes (the used label codes)
    """
    mimic_output_dir = f"/home/{user_id}/OTTEHR/outputs/mimic"
    label_codes = []
    for file in os.listdir(mimic_output_dir):
        if file.endswith("OT_score.csv") and "exp3" in file:
            label_codes.append(file.split("_")[1])
    return label_codes

def get_code_title(code):
    """ 
    Get ICD code titles for code 
    """
    diagnose_path = os.path.join(mimic_data_dir, "D_ICD_DIAGNOSES.csv")
    diagnose_df = pd.read_csv(diagnose_path, header=0, index_col=0)
    return diagnose_df.loc[code]['SHORT_TITLE']


def compute_metric_ratio(score_df, eval_metric):
    """ 
    Computes the improvement ratios for a evaluation metric (eval_metric) on dataset (score_df). \
        Used for visualizing results.
    :param Dataframe score_df: the dataframe for scores
    :param str eval_metric: the metric name for computing ratios, can be mae or rmse for regression, \
        precision, recall and f1 for classification
    """
    improve_ratios = []
    eval_metric = eval_metric.lower()
    for target_metric, trans_target_metric in zip(score_df[f'target_{eval_metric}'], score_df[f'trans_target_{eval_metric}']):
        if eval_metric == 'f1' or eval_metric == 'precision' or eval_metric == 'recall':
            improve_ratios.append(trans_target_metric-target_metric)
        elif eval_metric == 'mae' or eval_metric == 'rmse':
            improve_ratios.append((target_metric-trans_target_metric)/target_metric) # smaller is better
    return improve_ratios


def get_target_stats(score_df, eval_metric, method, log=True, filter_na=True):
    """ 
    Gets the metric statistics of eval_metric from score_df for the transported target domain
    Used for comparison between DeepJDOT and OTTEHR
    :param Dataframe score_df: the dataframe for scores
    :param str eval_metric: the metric name for computing ratios, can be mae or rmse for regression, \
        precision, recall and f1 for classification
    :param method str: OT, TCA, GFK, CA and DeepJDOT
    :param log bool: log the data or not 
    :param filter_na bool: filter out nan or not
    """
    stats = []
    eval_metric = eval_metric.lower()

    if method == 'deepJDOT' or method == 'RSD' or method == 'daregram':
        stats = list(score_df[f'target_{eval_metric}'])
    else:
        stats = list(score_df[f'trans_target_{eval_metric}'])

    if filter_na:
        stats = [x for x in stats if not pd.isna(x)]

    if log:
        stats = [np.log(x) for x in stats ]
    return stats
