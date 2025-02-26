import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/OTTEHR/")
sys.path.append(f"/home/{user_id}/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/OTTEHR/competitors/")

from common import *
from mimic_common import *
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


torch.manual_seed(0)
np.random.seed(0)



def daregram_loss(H1, H2):  

    def add_noise(feature, eps=1e-7):
        # Normalize the matrix: subtract the mean and divide by the standard deviation of each feature
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True)
        matrix_norm = (feature - mean) / (std + eps)  # Adding a small value to prevent division by zero

        # Add small Gaussian noise to the matrix to improve stability
        noise = torch.randn_like(matrix_norm) * eps
        noisy_feature = matrix_norm + noise
        return noisy_feature  
    
    b,p = H1.shape

    A = torch.cat((torch.ones(b,1), H1), 1)
    B = torch.cat((torch.ones(b,1), H2), 1)


    cov_A = (A.t()@A)
    cov_B = (B.t()@B) 
    

    _,L_A,_ = torch.svd(add_noise(cov_A))
    _,L_B,_ = torch.svd(add_noise(cov_B))
    
    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

    threshold = 0.999
    tradeoff_angle = 0.1
    tradeoff_scale = 0.001

    if(eigen_A[1]> threshold):
        T = eigen_A[1].detach()
    else:
        T = threshold
        
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

    if(eigen_B[1]> threshold):
        T = eigen_B[1].detach()
    else:
        T = threshold

    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    
    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    
    cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
    cos = torch.dist(torch.ones((p+1)),(cos_sim(A,B)),p=1)/(p+1)
    
    return tradeoff_angle*(cos) + tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


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

def run_daregram(source_data, source_labels, target_data, target_labels):
    """ 
    Return the RMSE and MAE
    """

    # set dataset
    batch_size = {"source": 10, "target": 10}

    # Convert numpy arrays to PyTorch tensors
    source_data = torch.tensor(source_data)
    source_labels = torch.tensor(source_labels)
    target_data = torch.tensor(target_data)
    target_labels = torch.tensor(target_labels)

    # Create datasets
    source_dataset = PreparedDataset(source_data, source_labels)
    target_dataset = PreparedDataset(target_data, target_labels)

    # Create data loaders
    dset_loaders = {
        "source": torch.utils.data.DataLoader(source_dataset, batch_size=batch_size["source"], shuffle=True, num_workers=4),
        "target": torch.utils.data.DataLoader(target_dataset, batch_size=batch_size["target"], shuffle=False, num_workers=4)
    }


    n_components = 50
    reg_model = LinearRegressionModel(input_features=source_data.shape[1], output_features=1, hidden_units=n_components)
    # Model_R = Model_R.to(device)

    reg_model.train(True)
    criterion = {"regressor": nn.MSELoss()}
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, reg_model.extraction.parameters()), "lr": 0.1},
                    {"params": filter(lambda p: p.requires_grad, reg_model.output.parameters()), "lr": 1}]

    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)


    len_source = len(dset_loaders["source"]) - 1
    len_target = len(dset_loaders["target"]) - 1
    param_lr = []
    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])
    lr = 0.1
    gamma = 0.0001

    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    test_interval = 5
    num_iter = 100

    train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0

    for iter_num in range(1, num_iter + 1):
        print("iter_num is:", iter_num)
        reg_model.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=lr, gamma=gamma, power=0.75,
                                    weight_decay=0.0005)
        optimizer.zero_grad()
        if iter_num % len_source == 0:
            iter_source = iter(dset_loaders["source"])
        if iter_num % len_target == 0:
            iter_target = iter(dset_loaders["target"])

        data_source = next(iter_source)
        data_target = next(iter_target)

        inputs_source, labels_source = data_source
        inputs_target, _ = data_target


        feature_s, outC_s,  = reg_model(inputs_source)
        feature_t, _ = reg_model(inputs_target)


        regression_loss = criterion["regressor"](outC_s, labels_source)
        dare_gram_loss = daregram_loss(feature_s,feature_t)

        total_loss = regression_loss + dare_gram_loss

        reg_model = reg_model
        total_loss.backward()

        # Clip graidents
        clip_value = 1
        torch.nn.utils.clip_grad_norm_(reg_model.parameters(), clip_value)

        optimizer.step()

        train_regression_loss += regression_loss.item()
        train_dare_gram_loss += dare_gram_loss.item()
        train_total_loss += total_loss.item()
        if iter_num % test_interval == 0:
            print("Iter {:05d}, Average MSE Loss: {:.4f}; Average DARE-GRAM Loss: {:.4f}; Average Training Loss: {:.4f}".format(
                iter_num, train_regression_loss / float(test_interval), train_dare_gram_loss / float(test_interval), train_total_loss / float(test_interval)))
            train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0


    # Evaluate the model with training data
    with torch.no_grad():  # We don't need gradients in the testing phase

        # Similarly for testing data
        _, target_pred_labels = reg_model(target_data.float())
        target_pred_labels = target_pred_labels.data.numpy()
        target_labels = target_labels.data.numpy()
        
        test_RMSE = np.sqrt(np.mean((target_pred_labels- target_labels) ** 2))
        test_MAE = np.mean(np.abs(target_pred_labels - target_labels))
        print("log RMSE is:", np.log(test_RMSE), "log MAE is:", np.log(test_MAE))
    return test_RMSE, test_MAE


