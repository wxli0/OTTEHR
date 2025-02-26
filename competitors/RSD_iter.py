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



# Define the linear regression model with a feature extraction layer
class LinearRegressionModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super(LinearRegressionModel, self).__init__()
        # Feature extraction layers
        self.extraction = nn.Linear(input_features, hidden_units)  # First hidden layer

        # Linear regression output layer
        self.output = nn.Linear(hidden_units, output_features)  # Output layer

    def forward(self, x):
        feature = self.extraction(x)
        x = self.output(feature)
        return feature, x

def RSD_BMP(source_feature, target_feature, RSD_coef = 0.001, BMP_coef = 0.1, eps=1e-6):
    print("source_feature shape is:", source_feature.shape)
    print("target_feature_shape is:", target_feature.shape)
    def softplus_abs(x, beta=10):
        # A smooth approximation of the absolute value function
        return (torch.nn.functional.softplus(beta * x) + torch.nn.functional.softplus(-beta * x)) / (2 * beta)


    def add_noise(feature, eps=1e-7):
        # Normalize the matrix: subtract the mean and divide by the standard deviation of each feature
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True)
        matrix_norm = (feature - mean) / (std + eps)  # Adding a small value to prevent division by zero

        # Add small Gaussian noise to the matrix to improve stability
        noise = torch.randn_like(matrix_norm) * eps
        noisy_feature = matrix_norm + noise
        return noisy_feature

    noisy_source_feature_t = add_noise(source_feature.t())
    noisy_target_feature_t = add_noise(target_feature.t())
    print("noisy_source_feature_t shape is:", noisy_source_feature_t.shape)
    print("noisy_target_feature_t shape is:", noisy_target_feature_t.shape)

    u_s, _, _ = torch.svd(noisy_source_feature_t)
    u_t, _, _ = torch.svd(noisy_target_feature_t)
    print("u_s shape is:", u_s.shape, "u_t shape is:", u_t.shape)

    noisy_product = add_noise(torch.mm(u_s.t(), u_t))
    print("noisy_product shape is:", noisy_product.shape)
    p_s, cosine, p_t = torch.svd(noisy_product)
    adjusted_cosine = torch.clamp(1 - torch.pow(cosine, 2), min=eps)
    sine = torch.sqrt(adjusted_cosine)
    print("p_s shape is:", p_s.shape, "p_t shape is:", p_t.shape)

    soft_diff = softplus_abs(p_s - p_t)

    return RSD_coef*(torch.norm(sine, 1) + BMP_coef * torch.norm(soft_diff, 2))


def run_RSD(source_data, source_labels, target_data, target_labels):
    """ 
    Return the RMSE and MAE
    """
    num_iter = 10

    # Convert the numpy arrays into torch tensors
    source_data_tensor = torch.tensor(source_data.astype(np.float32))
    print("source_data shape is:", source_data.shape)
    print("source_data_tensor shape is:", source_data_tensor.shape)

    source_labels_tensor = torch.tensor(source_labels.astype(np.float32)).view(-1, 1)  # Reshaping for a single output feature
    target_data_tensor = torch.tensor(target_data.astype(np.float32))
    target_labels_tensor = torch.tensor(target_labels.astype(np.float32)).view(-1, 1)

    # Create data loaders
    source_dataset = TensorDataset(source_data_tensor, source_labels_tensor)
    target_dataset = TensorDataset(target_data_tensor, target_labels_tensor)    
    source_loader = DataLoader(dataset=source_dataset, batch_size=50, shuffle=True, num_workers=4)
    target_loader = DataLoader(dataset=target_dataset, batch_size=50, shuffle=False, num_workers=4)  # Typically no need to shuffle target data
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    len_source = len(source_loader) - 1
    len_target = len(target_loader) - 1




    n_components = 50

    # Instantiate the model
    reg_model = LinearRegressionModel(input_features=source_data.shape[1], output_features=1, hidden_units=n_components)

    # Define the loss criterion and the optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(reg_model.parameters(), lr=0.01)

    # Training loop
    
    for iter_num in range(1, num_iter + 1):
        reg_model.train(True)

        if iter_num % len_source == 0:
            iter_source = iter(source_loader)
        if iter_num % len_target == 0:
            iter_target = iter(target_loader)
        data_source = next(iter_source)
        data_target = next(iter_target)
        inputs_source, labels_source = data_source
        inputs_target, _ = data_target



        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward
        optimizer.zero_grad()

        # Get output from the model, given the inputs
        source_feature, source_outputs = reg_model(inputs_source)
        target_feature, _ =  reg_model(inputs_target)

        # Get loss for the predicted output
        regression_loss = criterion(source_outputs, labels_source)
        rsd_bmp_loss = RSD_BMP(source_feature, target_feature)
        total_loss = regression_loss + rsd_bmp_loss
        
        # Get gradients w.r.t to parameters
        total_loss.backward()

        # Clip graidents
        clip_value = 1e-3
        torch.nn.utils.clip_grad_norm_(reg_model.parameters(), clip_value)

        # Update parameters
        optimizer.step()


        print(f'Iteration {iter_num+1}/{num_iter}, Loss: {total_loss.item()}')

    # Evaluate the model with training data
    reg_model.eval()
    with torch.no_grad():  # We don't need gradients in the testing phase
        _, predicted_train = reg_model(Variable(source_data_tensor))
        predicted_train = predicted_train.data.numpy()
        train_loss = np.mean((predicted_train - source_labels) ** 2)
        print(f'Training Mean Squared Error: {train_loss}')

        # Similarly for testing data
        _, predicted_test = reg_model(Variable(target_data_tensor))
        predicted_test = predicted_test.data.numpy()
        test_RMSE = np.sqrt(np.mean((predicted_test - target_labels) ** 2))
        test_MAE = np.mean(np.abs(predicted_test - target_labels))
    return test_RMSE, test_MAE

