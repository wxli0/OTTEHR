import numpy as np
from scipy.optimize import minimize

def correlation_alignment_loss(params, C_s, C_t, d):
    """ 
    Define the Correlation Alignment loss function
    """
    W = params.reshape(d, d)
    aligned_C_t = np.dot(np.dot(W.T, C_t), W)
    loss = np.linalg.norm(C_s - aligned_C_t, 'fro') ** 2
    return loss


def trans_CA(target_reps, source_reps):
    """ 
    Transport by correlation alignment 
    """

    # Initialize transformation matrix W
    d = source_reps.shape[1]  # Dimensionality of features
    W_init = np.random.rand(d, d).reshape(-1)

    # Minimize the Correlation Alignment loss
    C_s = np.corrcoef(source_reps.T)
    C_t = np.corrcoef(target_reps.T)
    print("C_s shape is:", C_s.shape)
    print("C_t shape is:", C_t.shape)
    result = minimize(correlation_alignment_loss, W_init, args=(C_s, C_t, d), method='BFGS')
    # print("result shape is:", result.shape)

    # Get the learned transformation matrix W
    W = result.x.reshape(d, d)

    # Apply the transformation to align the target domain data
    trans_target_reps = np.dot(target_reps, W)

    return trans_target_reps




