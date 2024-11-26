import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm
import pyro
import math
import pickle
import time
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az
import seaborn as sns


import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.GP_models as GP_models
import GP_functions.Tools as Tools
import GP_functions.FeatureE as FeatureE

X_train = pd.read_csv('Data/X_train.csv', header=None, delimiter=',').values
X_test = pd.read_csv('Data/X_test.csv', header=None, delimiter=',').values

Y_train_std = pd.read_csv('Data/Y_train_std.csv', header=None, delimiter=',').values
Y_test_std = pd.read_csv('Data/Y_test_std.csv', header=None, delimiter=',').values

train_x = torch.tensor(X_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)

train_y = torch.tensor(Y_train_std, dtype=torch.float32)
test_y = torch.tensor(Y_test_std, dtype=torch.float32)


####################################################################

Device = 'cpu'

row_idx = 0

input_point = test_y[row_idx,:]
local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k = 100)



LocalGP_models, LocalGP_likelihoods = Training.train_one_row_LocalGP_Parallel(train_x, train_y, 
                                                                                    test_y, row_idx, covar_type = 'RBF', k_num = 100,
                                                                                    lr=0.025, num_iterations=5000, 
                                                                                    patience=10, device=Device)


full_test_preds = Prediction.full_preds(LocalGP_models_48, LocalGP_likelihoods_48, test_x[row_idx,:].unsqueeze(0).to(Device)).cpu().detach().numpy()



bounds = bound.get_bounds(local_train_x)

estimated_params, func_loss = Estimation.multi_start_estimation(LocalGP_models_48, LocalGP_likelihoods_48, row_idx, test_y_48, bounds, Estimation.estimate_params_Adam, 
                                                       num_starts=5, num_iterations=2000, lr=0.01, patience=50, attraction_threshold=0.1, repulsion_strength=0.1, device=Device)


full_estimated_params = estimated_params.detach().numpy()


mcmc_result_Normal_48 = Estimation.run_mcmc_Normal(Prediction.full_preds, LocalGP_models_48, LocalGP_likelihoods_48, row_idx, test_y_48, local_train_x, num_sampling=3000, warmup_step=1000)



posterior_samples_Normal = mcmc_result_Normal_48.get_samples()
param_names = [f'param_{i}' for i in range(len(bounds))]


posterior_means_array = np.zeros(len(param_names))


for idx, param_name in enumerate(param_names):
    samples = posterior_samples_Normal[param_name]
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    mean_value = torch.mean(samples).item()
    posterior_means_array[idx] = mean_value



np.save('FixedG/LocalGP_full_preds_48.npy', full_test_preds)

np.save('FixedG/LocalGP_full_estimated_params_48.npy', full_estimated_params)

np.save('FixedG/LocalGP_posterior_means_48.npy', posterior_means_array)


for row_idx in range(1,test_y_48.shape[0]):
# for row_idx in range(1,2):
    input_point = test_y_48[row_idx,:]

    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y_48, k = 100)


    LocalGP_models_48, LocalGP_likelihoods_48 = Training.train_one_row_LocalGP_Parallel(train_x, train_y_48, 
                                                                              test_y_48, row_idx, covar_type = 'RQ', k_num = 100,
                                                                              lr=0.025, num_iterations=5000, 
                                                                              patience=10, device=Device)

    preds_tmp = Prediction.full_preds(LocalGP_models_48, LocalGP_likelihoods_48, test_x[row_idx,:].unsqueeze(0).to(Device)).cpu().detach().numpy()
    full_test_preds = np.vstack((full_test_preds, preds_tmp))
    np.save('FixedG/LocalGP_full_preds_48.npy', full_test_preds)

    bounds = bound.get_bounds(local_train_x)

    estimated_params_tmp, estimated_params_loss_tmp = Estimation.multi_start_estimation(LocalGP_models_48, LocalGP_likelihoods_48, row_idx, test_y_48, bounds, Estimation.estimate_params_Adam, 
                                                       num_starts=5, num_iterations=2000, lr=0.01, patience=50, attraction_threshold=0.1, repulsion_strength=0.1, device=Device)

    full_estimated_params = np.vstack((full_estimated_params, estimated_params_tmp.detach().numpy()))
    np.save('FixedG/LocalGP_full_estimated_params_48.npy', full_estimated_params)


    mcmc_result_Normal_48 = Estimation.run_mcmc_Normal(Prediction.full_preds, LocalGP_models_48, LocalGP_likelihoods_48, row_idx, test_y_48, local_train_x, num_sampling=3000, warmup_step=1000)

    posterior_samples_Normal = mcmc_result_Normal_48.get_samples()

    posterior_means_array_tmp = np.zeros(len(param_names))


    for idx, param_name in enumerate(param_names):
        samples = posterior_samples_Normal[param_name]
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        mean_value = torch.mean(samples).item()
        posterior_means_array_tmp[idx] = mean_value

    posterior_means_array = np.vstack((posterior_means_array, posterior_means_array_tmp))
    np.save('FixedG/LocalGP_posterior_means_48.npy', posterior_means_array)
    



# nohup python MGP_10_48.py > MulGPout.log 2>&1 &