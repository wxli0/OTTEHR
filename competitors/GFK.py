import numpy as np

def trans_GFK(source_reps, target_reps):
    # Compute covariance matrices
    cov_source = np.cov(source_reps, rowvar=False)
    cov_target = np.cov(target_reps, rowvar=False)
    
    # Cholesky decomposition
    L_source = np.linalg.cholesky(cov_source + np.eye(source_reps.shape[1]) * 1e-6)
    L_target = np.linalg.cholesky(cov_target + np.eye(target_reps.shape[1]) * 1e-6)
    
    # Compute the inverse of the square root of covariance matrices
    S_source = np.linalg.inv(L_source)
    S_target = np.linalg.inv(L_target)
    
    # Compute the transformation matrix
    W = np.dot(S_source, S_target.T)
    trans_source_reps = np.dot(source_reps, W)
    trans_target_reps = np.dot(target_reps, W)
    
    return trans_source_reps, trans_target_reps
