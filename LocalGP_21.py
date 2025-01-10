import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm

from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import os

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

Y_train_21 = pd.read_csv('Data/Y_train_std_21.csv', header=None, delimiter=',').values
Y_test_21 = pd.read_csv('Data/Y_test_std_21.csv', header=None, delimiter=',').values

Y_train_std = pd.read_csv('Data/Y_train_std.csv', header=None, delimiter=',').values
Y_test_std = pd.read_csv('Data/Y_test_std.csv', header=None, delimiter=',').values


train_x = torch.tensor(X_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)

train_y_21 = torch.tensor(Y_train_21, dtype=torch.float32)
test_y_21 = torch.tensor(Y_test_21, dtype=torch.float32)

train_y = torch.tensor(Y_train_std, dtype=torch.float32)
test_y = torch.tensor(Y_test_std, dtype=torch.float32)


pca_20 = PCA(n_components = 20)

pca_20.fit(train_y[:,1:])

####################################################################

Device = 'cpu'


output_file = 'Result/LocalGP_21_result.csv'


if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,test_preds,estimated_params,posterior_means\n')

for row_idx in range(test_y_21.shape[0]):
    input_point = test_y_21[row_idx, :]

    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y_21, k=100)

    LocalGP_models, LocalGP_likelihoods = Training.train_one_row_LocalGP_Parallel(
        train_x, train_y_21, test_y_21, row_idx,
        covar_type='RBF', k_num=100, lr=0.025,
        num_iterations=5000, patience=10, device=Device
    )


    preds_tmp = Prediction.full_preds(
        LocalGP_models, LocalGP_likelihoods, test_x[row_idx, :].unsqueeze(0).to(Device)
    ).cpu().detach().numpy()


    bounds = bound.get_bounds(local_train_x)
    estimated_params_tmp, _ = Estimation.multi_start_estimation(
        LocalGP_models, LocalGP_likelihoods, row_idx, test_y_21, bounds,
        Estimation.estimate_params_Adam, num_starts=5, num_iterations=2000, lr=0.01,
        patience=50, attraction_threshold=0.1, repulsion_strength=0.1, device=Device
    )


    mcmc_result_Uniform = Estimation.run_mcmc_Uniform(
        Prediction.full_preds, LocalGP_models, LocalGP_likelihoods, 
        row_idx, test_y, bounds, 
        PCA_func=pca_20, 
        num_sampling=4000, warmup_step=1000, num_chains=1
    )
    posterior_samples_Uniform = mcmc_result_Uniform.get_samples()

    param_names = [f'param_{i}' for i in range(len(bounds))]
    posterior_means_array_tmp = np.zeros(len(param_names))

    for idx, param_name in enumerate(param_names):
        samples = posterior_samples_Uniform[param_name]
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        posterior_means_array_tmp[idx] = torch.mean(samples).item()


    with open(output_file, 'a') as f:
        f.write(f"{row_idx + 1},\"{list(preds_tmp)}\",\"{list(estimated_params_tmp.detach().numpy())}\",\"{list(posterior_means_array_tmp)}\"\n")




# nohup python LocalGP_21.py > LocalGP_21out.log 2>&1 &