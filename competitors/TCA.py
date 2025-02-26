""" 
Implement Transfer Component Analysis
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer


def trans_TCA(Xs, Xt, n_components=None, scale=True):
    """
    Transfer Component Analysis (TCA) implementation.

    Parameters:
    -----------
    Xs : array-like, shape (n_samples_source, n_features)
        Source domain data.
    Xt : array-like, shape (n_samples_target, n_features)
        Target domain data.
    n_components : int or None, optional (default=None)
        Number of components to keep. If None, keeps all components.
    scale : bool, optional (default=True)
        Whether to perform feature scaling before applying TCA.

    Returns:
    --------
    Xs_tca : array, shape (n_samples_source, n_components)
        Transformed source domain data after TCA.
    Xt_pca_centered: array, shape (n_samples_target, n_components)
        Target data
    Xt_tca : array, shape (n_samples_target, n_components)
        Transformed target domain data after TCA.
    """

    if scale:
        # Standardize the data
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.transform(Xt)

    # Perform PCA on both source and target domains
    pca = PCA(n_components=n_components)
    Xs_pca = pca.fit_transform(Xs)
    Xt_pca = pca.transform(Xt)

    # Center the PCA transformed source and target data
    Xs_mean = np.mean(Xs_pca, axis=0)
    Xt_mean = np.mean(Xt_pca, axis=0)
    Xs_pca_centered = Xs_pca - Xs_mean
    Xt_pca_centered = Xt_pca - Xt_mean

    # Compute the covariance matrices of centered PCA transformed source and target data
    # print("Xs is:", Xs)
    # print("Xt is:", Xt)
    cov_source = (1 / (Xs_pca_centered.shape[0] - 1)) * np.dot(Xs_pca_centered.T, Xs_pca_centered)
    cov_target = (1 / (Xt_pca_centered.shape[0] - 1)) * np.dot(Xt_pca_centered.T, Xt_pca_centered)

    # Perform singular value decomposition (SVD) on the covariance matrices
    U, _, Vt = np.linalg.svd(cov_source.T @ cov_target)

    # Compute the optimal projection matrix
    W = np.dot(U, Vt)
    # print("W is:", W)

    # Transform the source and target data using the optimal projection matrix
    Xs_tca = np.dot(Xs_pca_centered, W.T)
    Xt_tca = np.dot(Xt_pca_centered, W.T)
    # print("does X_t change?:", Xt_tca == Xt)

    return Xs_tca, Xt_pca_centered, Xt_tca
