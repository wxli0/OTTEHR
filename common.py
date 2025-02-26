""" 
Common functions for synthetic datasets
"""

import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/projects/OTTEHR")
sys.path.append(f"/home/{user_id}/projects/unbalanced_gromov_wasserstein/")

import numpy as np
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import ot
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from statistics import median
import torch
from unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn
from unbalancedgw._vanilla_utils import ugw_cost
from unbalancedgw.utils import generate_measure, dist_matrix


def trans_target2source(target_reps, source_reps, reg_e = 0.1, max_iter = None, ret_cost=False, ret_coupling=False):
    """ 
    Optimal transport (without entropy regularization) source representations \
        to target representations

    :param str type: balanced or unbalanced
    :param bool ret_cost: return OT cost or not
    :param bool ret_coupling: return coupling or not
    :returns: transported source representations and the optimized Wasserstein distance (if cost is True), default False

    TODO: the unbalanced case has not been implemented 
    """
    trans_target_reps = None
    # if type == "balanced":
    ot_emd = ot.da.SinkhornTransport(reg_e=reg_e, log=True)
    if max_iter is not None:
        ot_emd = ot.da.SinkhornTransport(reg_e=reg_e, max_iter=max_iter, log=True)
    ot_emd.fit(Xs=target_reps, Xt=source_reps)
    trans_target_reps = ot_emd.transform(Xs=target_reps)
    if not ret_cost and not ret_coupling:
        return trans_target_reps
    if not ret_cost:
        return trans_target_reps, ot_emd.coupling_
    
    wa_dist = ot_emd.log_['err'][-1]
    if not ret_coupling:
        return trans_target_reps, wa_dist,

    return trans_target_reps, wa_dist, ot_emd.coupling_



def trans_GWOT(target_reps, source_reps):
    """ 
    Unbalanced Gromov Wasserstein Optimal transport (without entropy regularization) source representations \
        to target representations

    :param str type: balanced or unbalanced
    :param bool ret_cost: return OT cost or not
    :param bool ret_coupling: return coupling or not
    :returns: transported source representations, the transport plan and the optimized Wasserstein distance 

    """

    # Generate measures
    target_measure = torch.from_numpy(np.array([1/target_reps.shape[0]] * target_reps.shape[0]))
    source_measure = torch.from_numpy(np.array([1/source_reps.shape[0]] * source_reps.shape[0]))

    # Generate metric
    target_metric = torch.from_numpy(cdist(target_reps, target_reps, metric='euclidean'))
    source_metric = torch.from_numpy(cdist(source_reps, source_reps, metric='euclidean'))

    # Run unbalanced Gromov-Wasserstein OT
    eps = 1.0
    rho, rho2 = 1.0, 1.0
    coupling_1, coupling_2 = exp_ugw_sinkhorn(source_measure, source_metric, target_measure, target_metric, \
                          init=None, eps=eps,
                          rho=rho, rho2=rho2,
                          nits_plan=1000, tol_plan=1e-5,
                          nits_sinkhorn=1000, tol_sinkhorn=1e-5,
                          two_outputs=True)
    wa_dist = ugw_cost(coupling_1, coupling_2, source_measure, source_metric, target_measure, target_metric, eps=eps, rho=rho, rho2=rho2)
    wa_dist = wa_dist.detach().cpu().numpy()

    coupling_1 = coupling_1.detach().cpu().numpy()
    transp = np.transpose(coupling_1)
    transp = transp/ np.sum(transp, axis=1)[:, None]
    print("target_reps shape is:", target_reps.shape)
    print("coupling_1 shape is:", coupling_1.shape)
    trans_target_reps = np.matmul(transp, source_reps)
    print("trans_target_reps is:", trans_target_reps.shape)

    return trans_target_reps, coupling_1, wa_dist


def compute_wa_dist(cost_matrix):
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    w_dist = np.sum(cost_matrix[row_indices, col_indices])
    return w_dist


def trans_EMD_OT(target_reps, source_reps):
    """ 
    Transport by balanced earth move distance optimal transport (optimal transport map by Monge)

    :returns the transported target representations
    """

    M = ot.dist(source_reps, target_reps, metric='euclidean')
    M /= M.max()
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=target_reps, Xt=source_reps)
    return ot_emd.transform(Xs=target_reps)



def trans_UOT(target_reps, source_reps, reg=0.1, reg_m=1):
    """ 
    Transport by unbalanced optimal transport
    
    :param float reg: entropy regularization param
    :param float reg_m: marginal relaxation paramter 

    :returns the transported target representations, the Wasserstein distance, \
        the tranport plan, and the maximum distance (diameter) between source and target embeddings
    """

    source_measure = np.ones((source_reps.shape[0],))/source_reps.shape[0]
    target_measure = np.ones((target_reps.shape[0],))/target_reps.shape[0]

    M = ot.dist(source_reps, target_reps, metric='euclidean')
    M /= M.max()
    wa_dist = ot.emd2(source_measure, target_measure, M)
    # print("wa_distance is:", wa_dist)

    ot_sinkhorn = ot.da.UnbalancedSinkhornTransport(reg_e=reg, reg_m=reg_m)
    ot_sinkhorn.fit(Xs=target_reps, Xt=source_reps)
    trans_target_reps = ot_sinkhorn.transform(Xs=target_reps)

    # Compute the maximum distance between source representations and target representations
    return trans_target_reps, wa_dist, np.transpose(ot_sinkhorn.coupling_), M.max()


""" 
Caculate result statistics for binary labels
"""

def cal_stats_binary(source_reps, source_labels, target_reps, target_labels, \
    trans_target_reps, source_model):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    This function is for binary labels

    :param function source_model: the model trained by source data
    
    :returns: using the target model,\
        - accuracy for target/source/transported target
        - precision for target/source/transported target
        - recall for target/source/transported target
        - f1 for target/source/transported target
            
    """

    # calculate the stats
    source_pred_labels = source_model.predict(source_reps)
    source_accuracy = accuracy_score(source_labels, source_pred_labels)
    source_precision = precision_score(source_labels, source_pred_labels)
    source_recall = recall_score(source_labels, source_pred_labels)
    source_f1 = f1_score(source_labels, source_pred_labels, average="weighted")

    target_pred_labels = source_model.predict(target_reps)
    target_accuracy = accuracy_score(target_labels, target_pred_labels)
    target_precision = precision_score(target_labels, target_pred_labels)
    target_recall = recall_score(target_labels, target_pred_labels)
    target_f1 = f1_score(target_labels, target_pred_labels, average="weighted")

    trans_target_pred_labels = source_model.predict(trans_target_reps)
    trans_target_accuracy = accuracy_score(target_labels, trans_target_pred_labels)
    trans_target_precision = precision_score(target_labels, trans_target_pred_labels)
    trans_target_recall = recall_score(target_labels, trans_target_pred_labels)
    trans_target_f1 = f1_score(target_labels, trans_target_pred_labels, average="weighted")

    return source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1


def cal_stats_cts(source_reps, source_labels, target_reps, target_labels, \
    trans_target_reps, model_func):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    This function is for continous labels

    :param function model_func: the function to model the relationship between \
        representations and reponse
    
    :returns: using the target model,\
        - mean absoluate error (MAE) for target/source/transported source
        - mean squared error (MSE) for target/source/transported source
        - residual mean squared error (RMSE) for target/source/transported source
            
    """
    # fit the model
    source_model = model_func()
    source_model.fit(source_reps, source_labels)

    # calculate the stats
    source_pred_labels = source_model.predict(source_reps)
    source_mae = metrics.mean_absolute_error(source_labels, source_pred_labels)
    source_mse = metrics.mean_squared_error(source_labels, source_pred_labels)
    source_rmse = np.sqrt(metrics.mean_squared_error(source_labels, source_pred_labels))

    target_pred_labels = source_model.predict(target_reps)
    target_mae = metrics.mean_absolute_error(target_labels, target_pred_labels)
    target_mse = metrics.mean_squared_error(target_labels, target_pred_labels)
    target_rmse = np.sqrt(metrics.mean_squared_error(target_labels, target_pred_labels))

    trans_target_pred_labels = source_model.predict(trans_target_reps)
    trans_target_mae = metrics.mean_absolute_error(target_labels, trans_target_pred_labels)
    trans_target_mse = metrics.mean_squared_error(target_labels, trans_target_pred_labels)
    trans_target_rmse =  np.sqrt(metrics.mean_squared_error(target_labels, trans_target_pred_labels))

    return source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse,\
        trans_target_mae, trans_target_mse, trans_target_rmse



def train_model(reps, labels, model_func):
    """ 
    Trains a model using reps and labels and returns the model
    """
    clf = model_func()
    clf.fit(reps, labels)
    return clf



""" 
Wrap up everything for binary labels
"""

def entire_proc_binary(sim_func, custom_train_reps, model_func, max_iter):
    """ 
    Executes the entire procedure including
        - generate target sequences, target labels, source sequences and source labels
        - generate target representations and source representations
        - transport source representations to target representations
        - train logistic regression model using target representations and target expires
        - calculate the transferability score by computing the KL divergence between the two model weights
        - calculate accuracy statistics for targets, sources and transported sources 

    :param function sim_func: simulation function
    :param function custom_train_reps: customized deep patient function for training representations
    :param function model_func: the function to model the relationship bewteen representations and response
    :param int max_iter: maximum number of iteration for Sinkhorn transport
    :param bool transfer_score: wheter to compute transferability score, default False
    :returns: the accuracy scores, and the transferability score
    """
    source_seqs, source_labels, target_seqs, target_labels = sim_func()
    source_reps, target_reps = custom_train_reps(source_seqs, target_seqs)
    trans_target_reps = trans_target2source(target_reps, source_reps, max_iter=max_iter)

    source_model = train_model(source_reps, source_labels, model_func)

    source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1 = \
        cal_stats_binary(source_reps, source_labels, target_reps, target_labels, trans_target_reps, source_model)
    
    return source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1
    

""" 
Wrap up everything for continuous labels
"""

def entire_proc_cts(sim_func, custom_train_reps, model_func, max_iter, reg_e=1e-1):
    """ 
    Executes the entire procedure including
        - generate target sequences, target labels, source sequences and source labels
        - generate target representations and source representations
        - transport source representations to target representations
        - train regression model using target representations and target expires
        - calculate accuracy statistics for targets, sources and transported sources

    :param function sim_func: simulation function
    :param function custom_train_reps: customized deep patient function for training representations
    :param function model_func: the function to model the relationship bewteen representations and response
    :param int max_iter: maximum number of iterations for Sinkhorn OT
    :param float reg_e: regularizer for sinkhorn OT (default 1e-1)
    :returns: the accuracy scores
    """
    source_seqs, source_labels, target_seqs, target_labels = sim_func()
    source_reps, target_reps = custom_train_reps(source_seqs, target_seqs)
    trans_target_reps = trans_target2source(target_reps, source_reps, max_iter=max_iter)
    
    source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse,\
        trans_target_mae, trans_target_mse, trans_target_rmse = \
        cal_stats_cts(source_reps, source_labels, target_reps, target_labels, trans_target_reps, model_func)
    return source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse,\
        trans_target_mae, trans_target_mse, trans_target_rmse


def train_model(reps, labels, model_func = linear_model.LinearRegression): 
    """ 
    Train a model using a model function and by representations reps and labels

    :param function model_func: the model function, e.g. linear_model.LinearRegression

    :returns:
        - the learned linear model
    """
    clf = model_func()
    clf.fit(reps, labels)
    return clf


""" 
Run entire procedure on multiple simulations and print accuracy statistics, \
    for binary labels
"""

def run_proc_multi(sim_func, custom_train_reps, model_func, max_iter = None, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for binary labels

    :param function model_func: the function to model the relationship between representations and responses
    :param bool filter: whether to filter out source accuracies > 0.7
    :param int max_iter: maximum number of iterations for Sinkhorn transport

    :returns: vectors of accuracy statistics of multiple rounds
    """
    
    target_accuracies = []
    target_precisions = [] 
    target_recalls = [] 
    target_f1s = []
    source_accuracies = []
    source_precisions = []
    source_recalls = [] 
    source_f1s = []
    trans_target_accuracies = []
    trans_target_precisions = []
    trans_target_recalls = []
    trans_target_f1s = []

    for i in range(n_times):
        print(f"iteration: {i}")
        # init accuracies
        target_accuracy = None
        target_precision = None
        target_recall = None
        target_f1 = None
        source_accuracy = None
        source_precision = None
        source_recall = None
        source_f1 = None
        trans_target_accuracy = None
        trans_target_precision = None
        trans_target_recall = None
        trans_target_f1 = None

        try:
            source_accuracy, source_precision, source_recall, source_f1, \
            target_accuracy, target_precision, target_recall, target_f1, \
            trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1 = \
                entire_proc_binary(sim_func, custom_train_reps, model_func, max_iter)

        except Exception: # most likely only one label is generated for the examples
            print("exception 1")
            continue

        # if domain 2 data performs better using the model trained by domain 1 data, \
        # there is no need to transport
        if source_accuracy <= target_accuracy: 
            print("exception 2")
            continue

        # denominator cannot be 0
        min_deno = 0.001
        target_accuracy = max(target_accuracy, min_deno)
        target_precision = max(target_precision, min_deno)
        target_recall = max(target_recall, min_deno)
        target_f1 = max(target_f1, min_deno)
        source_accuracy = max(source_accuracy, min_deno)
        source_precision = max(source_precision, min_deno)
        source_recall = max(source_recall, min_deno)
        source_f1 = max(source_f1, min_deno)
        trans_target_accuracy = max(trans_target_accuracy, min_deno)
        trans_target_precision = max(trans_target_precision, min_deno)
        trans_target_recall = max(trans_target_recall, min_deno)
        trans_target_f1 = max(trans_target_f1, min_deno)

        target_accuracies.append(target_accuracy)
        target_precisions.append(target_precision)
        target_recalls.append(target_recall)
        target_f1s.append(target_f1)
        source_accuracies.append(source_accuracy)
        source_precisions.append(source_precision)
        source_recalls.append(source_recall)
        source_f1s.append(source_f1)
        trans_target_accuracies.append(trans_target_accuracy)
        trans_target_precisions.append(trans_target_precision)
        trans_target_recalls.append(trans_target_recall) 
        trans_target_f1s.append(trans_target_f1)
    return source_accuracies, source_precisions, source_recalls, source_f1s, \
        target_accuracies, target_precisions, target_recalls, target_f1s, \
        trans_target_accuracies, trans_target_precisions, trans_target_recalls, trans_target_f1s


""" 
Run entire procedure on multiple simulations and print accuracy statistics, \
    for continuous labels
"""

def run_proc_multi_cts(sim_func, custom_train_reps, model_func, reg_e = 1e-1, max_iter = None, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for continuous labels

    :param function model_func: the function to model the relationship between representations and responses
    :param int max_iter: maximum number of iterations for Sinkhorn OT
    :param float reg_e: the regularizer for sinkhorn OT, default 1e-1

    :returns: vectors of accuracy statistics of multiple rounds
    """
    
    target_maes = []
    target_mses = [] 
    target_rmses = [] 
    source_maes = []
    source_mses = []
    source_rmses = [] 
    trans_target_maes = []
    trans_target_mses = []
    trans_target_rmses = []

    for i in range(n_times):
        print("iteration:", i)
        # init accuracies
        target_mae = None
        target_mse = None
        target_rmse = None 
        source_mae = None
        source_mse = None
        source_rmse = None 
        trans_target_mae = None
        trans_target_mse = None
        trans_target_rmse = None


        try:
            source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse,\
                trans_target_mae, trans_target_mse, trans_target_rmse = \
                    entire_proc_cts(sim_func, custom_train_reps, model_func, max_iter, reg_e=reg_e)
                    
        except Exception: # most likely only one label is generated for the examples
            print("exception 1")
            continue

        # if domain 2 data performs better using the model trained by domain 1 data, \
        # there is no need to transport
        if target_mae <= source_mae: 
            print("exception 2")
            continue

        target_maes.append(target_mae)
        target_mses.append(target_mse)
        target_rmses.append(target_rmse)
        source_maes.append(source_mae)
        source_mses.append(source_mse)
        source_rmses.append(source_rmse)
        trans_target_maes.append(trans_target_mae)
        trans_target_mses.append(trans_target_mse)
        trans_target_rmses.append(trans_target_rmse)
    return source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
        trans_target_maes, trans_target_mses, trans_target_rmses


""" 
Constructs a dataframe to demonstrate the accuracy statistics for binary labels
"""

def save_scores(source_accuracies, source_precisions, source_recalls, source_f1s, \
        target_accuracies, target_precisions, target_recalls, target_f1s, \
        trans_target_accuracies, trans_target_precisions, trans_target_recalls, trans_target_f1s, file_path):
    """ 
    Save accuracy statistics to file path
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['source_accuracy'] = source_accuracies
    score_df['source_precision'] = source_precisions
    score_df['source_recall'] = source_recalls
    score_df['source_f1'] = source_f1s
    score_df['target_accuracy'] = target_accuracies
    score_df['target_precision'] = target_precisions
    score_df['target_recall'] = target_recalls
    score_df['target_f1'] = target_f1s
    score_df['trans_target_accuracy'] = trans_target_accuracies
    score_df['trans_target_precision'] = trans_target_precisions
    score_df['trans_target_recall'] = trans_target_recalls
    score_df['trans_target_f1'] = trans_target_f1s
    # save
    score_df.to_csv(file_path, index=None, header=True)



""" 
Constructs a dataframe to demonstrate the accuracy statistics for continuous labels
"""

def save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, target_clf_maes, target_clf_mses, target_clf_rmses, \
        trans_target_maes, trans_target_mses, trans_target_rmses, label_div_scores, wa_dists, coupling_diffs, diameters, max_hs, file_path):
    """ 
    Save accuracy statistics to file path
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['source_mae'] = source_maes
    score_df['source_mse'] = source_mses
    score_df['source_rmse'] = source_rmses
    score_df['target_mae'] = target_maes
    score_df['target_mse'] = target_mses
    score_df['target_rmse'] = target_rmses
    score_df['target_clf_mae'] = target_clf_maes
    score_df['target_clf_mse'] = target_clf_mses
    score_df['target_clf_rmse'] = target_clf_rmses
    score_df['trans_target_mae'] = trans_target_maes
    score_df['trans_target_mse'] = trans_target_mses
    score_df['trans_target_rmse'] = trans_target_rmses
    score_df['label_div_score'] = label_div_scores
    score_df['wa_dist'] = wa_dists
    score_df['coupling_diff'] = coupling_diffs
    score_df['diameter'] = diameters
    score_df['max_h'] = max_hs

    # save
    score_df.to_csv(file_path, index=None, header=True)


""" 
Box plot of simulation result statistics
"""

def box_plot_binary_short(score_path, save_path = None):
    """ 
    Box plot of the scores in score dataframe stored in score_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of source

    :param str score_path: the path to scores.csv
    :param str save_path: the path to save plot
    """

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    target_precision = scores_df['target_precision']
    target_recall = scores_df['target_recall']
    target_f1 = scores_df['target_f1']

    trans_target_precision = scores_df['trans_target_precision']
    trans_target_recall = scores_df['trans_target_recall']
    trans_target_f1 = scores_df['trans_target_f1']



    # transported target to target precision
    trans_target_target_precision = [i / j for i, j in zip(trans_target_precision, target_precision)]
    print("average trans target to target precision is:", np.mean(trans_target_target_precision))
    print("median trans target to target precision is:", np.median(trans_target_target_precision))


    # transported target to target recall
    trans_target_target_recall = [i / j for i, j in zip(trans_target_recall, target_recall)]
    print("average trans source to source recall is:", np.mean(trans_target_target_recall))
    print("median trans source to source recall is:", np.median(trans_target_target_recall))

    # transported source to source f1
    trans_target_target_f1 = [i / j for i, j in zip(trans_target_f1, target_f1)]
    print("average trans target to target f1 is:", np.mean(trans_target_target_f1))
    print("median trans target to target f1 is:", np.median(trans_target_target_f1))

    # Set the figure size
    plt.figure()
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    data = pd.DataFrame({
        'precision': trans_target_target_precision,
        'recall': trans_target_target_recall,
        'f1': trans_target_target_f1
    })

    # Plot the dataframe
    ax = data[['precision', 'recall', 'f1']].plot(kind='box')

    # Plot the baseline
    plt.axhline(y = 1, color = 'r', linestyle = '-')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()





""" 
Shorter version of box plot of simulation result statistics
"""

def box_plot_label_binary_short(score_path, label_code):
    """ 
    Box plot of the scores in score dataframe stored in score_path for binary labels with respect to the label_code. \
        Specifically, we plot the box plots of 
        - precision/recall of transported source over precision/recall of source

    :param str score_path: the path to scores.csv
    :param str label_code: the ICD code as the response

    Returns:
        - the median of trans source to source precision
        - the median of trans source to source recall
        - the median of the transferability score
    """

    def special_div(x, y):
        """ 
        Special division operation
        """
        if y == 0:
            y = 1e-5
        return x/y

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    target_precision = scores_df['target_precision']
    target_recall = scores_df['target_recall']

    trans_target_precision = scores_df['trans_target_precision']
    trans_target_recall = scores_df['trans_target_recall']

    transfer_score = scores_df['transfer_score']

    # original_score = scores_df['original_score']

    # transported source to source precision
    trans_target2target_precision = [special_div(i, j) for i, j in zip(trans_target_precision, target_precision)]

    # transported source to source recall
    trans_target2target_recall = [special_div(i, j) for i, j in zip(trans_target_recall, target_recall)]

    # # transfer score to original score
    # transfer2original_score = [special_div(i, j) for i, j in zip(transfer_score, original_score)]

    # Set the figure size
    plt.figure()
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    data = pd.DataFrame({
        'precision': trans_target2target_precision,
        'recall': trans_target2target_recall,
    })

    # Plot the dataframe
    ax = data[['precision', 'recall']].plot(kind='box', title=f'transported target to target for {label_code}')

    # Plot the baseline
    plt.axhline(y = 1, color = 'r', linestyle = '-')

    # Display the plot
    plt.show()
    return median(trans_target2target_precision), median(trans_target2target_recall), median(transfer_score)


""" 
Shorter version of box plot of simulation result statistics, for continuous response
"""

def box_plot_cts_short(score_path, save_path=None):
    """ 
    Box plot of the scores in score dataframe stored in score_path for binary labels. \
        Specifically, we plot the box plots of 
        - mae/rmse of transported source over accuracy/precision/recall of source

    :param str score_path: the path to scores.csv
    :param str response_name: the name of response

    Returns:
        - the medians of trans source to source mae
        - the medians of trans source to source rmse
    """

    def special_div(x, y):
        """ 
        Special division operation
        """
        if y == 0:
            y = 1e-5
        return x/y

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    target_mae = scores_df['target_mae']
    target_rmse = scores_df['target_rmse']

    trans_target_mae = scores_df['trans_target_mae']
    trans_target_rmse = scores_df['trans_target_rmse']

    # transported source to source mae
    trans_target_target_mae = [special_div(i, j) for i, j in zip(trans_target_mae, target_mae)]

    # transported source to source rmse
    trans_target_target_rmse = [special_div(i, j) for i, j in zip(trans_target_rmse, target_rmse)]

    # Set the figure size
    plt.figure()
    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    data = pd.DataFrame({
        'MAE': trans_target_target_mae,
        'RMSE': trans_target_target_rmse
    })

    # Plot the dataframe
    ax = data[['MAE', 'RMSE']].plot(kind='box')

    # Plot the baseline
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')


    plt.show()
    
    return median(trans_target_target_mae), median(trans_target_target_rmse)


def box_plot_cts_tca_short(ot_score_path, tca_score_path, save_path=None):
    """ 
    Box plot of the scores in score dataframe stored in score_path for ordered. \
        Specifically, we plot the box plots of 
        - mae/rmse of OTTEHR over TCA

    :param str ot_score_path: the path to OTTEHR scores
    :param str tca_score_path: the path to TCA scores

    Returns:
        - the medians of trans source to source mae
        - the medians of trans source to source rmse
    """

    def special_div(x, y):
        """ 
        Special division operation
        """
        if y == 0:
            y = 1e-5
        return x/y

    ot_score_df = pd.read_csv(ot_score_path, index_col=None, header=0)
    tca_score_df = pd.read_csv(tca_score_path, index_col=None, header=0)

    target_mae = tca_score_df['trans_target_mae']
    target_rmse = tca_score_df['trans_target_rmse']

    trans_target_mae = ot_score_df['trans_target_mae']
    trans_target_rmse = ot_score_df['trans_target_rmse']

    # transported source to source mae
    trans_target_target_mae = [special_div(i, j) for i, j in zip(trans_target_mae, target_mae)]

    # transported source to source rmse
    trans_target_target_rmse = [special_div(i, j) for i, j in zip(trans_target_rmse, target_rmse)]

    # Set the figure size
    plt.figure()

    # Pandas dataframe
    data = pd.DataFrame({
        'MAE ratio': trans_target_target_mae,
        'RMSE ratio': trans_target_target_rmse
    })

    # Plot the dataframe
    ax = data[['MAE ratio', 'RMSE ratio']].plot(kind='box')

    # Plot the baseline
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    return median(trans_target_target_mae), median(trans_target_target_rmse)

def save_results(rmses, maes, score_path):

    # read dataframe
    score_df = pd.DataFrame()
    score_df['target_rmse'] = rmses
    score_df['target_mae'] = maes

    # save
    score_df.to_csv(score_path, index=None, header=True)


# Define a simple dataset class
class PreparedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample.float(), label.float()