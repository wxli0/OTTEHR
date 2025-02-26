import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels

def MMD(X, Y, kernel):
    K_XX = pairwise_kernels(X, X, metric=kernel)
    K_XY = pairwise_kernels(X, Y, metric=kernel)
    K_YY = pairwise_kernels(Y, Y, metric=kernel)
    
    return np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)

# # Preprocess the data (standardization)
# scaler = StandardScaler()
# source_data = scaler.fit_transform(source_data)
# target_data = scaler.transform(target_data)

def trans_MMD(target_embs, source_embs):
    # Calculate MMD between source and target data
    mmd_value = MMD(source_embs, target_embs, kernel='laplacian')
    print(f"MMD value between source and target domains: {mmd_value}")

    # Perform domain adaptation using MMD
    # Weight the target domain samples by MMD distance for training
    weights = np.exp(-mmd_value * np.arange(len(target_embs))) 
    trans_target_embs = target_embs * weights[:, np.newaxis]
    return trans_target_embs

