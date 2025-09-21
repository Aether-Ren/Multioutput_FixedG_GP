import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm

import pickle

import pyro
import pyro.distributions as dist

from pyro.infer import MCMC, NUTS

import os


import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.NN_models as NN_models
import GP_functions.Tools as Tools
import GP_functions.FeatureE as FeatureE

X_train = pd.read_csv('Data/X_train.csv', header=None, delimiter=',').values
X_test = pd.read_csv('Data/X_test.csv', header=None, delimiter=',').values

Y_train_21 = pd.read_csv('Data/Y_train_std_21.csv', header=None, delimiter=',').values
Y_test_21 = pd.read_csv('Data/Y_test_std_21.csv', header=None, delimiter=',').values

Y_train_std = pd.read_csv('Data/Y_train_std.csv', header=None, delimiter=',').values
Y_test_std = pd.read_csv('Data/Y_test_std.csv', header=None, delimiter=',').values

Device = 'cpu'

train_x = torch.tensor(X_train, dtype=torch.float32, device=Device)
test_x = torch.tensor(X_test, dtype=torch.float32, device=Device)

train_y_21 = torch.tensor(Y_train_21, dtype=torch.float32, device=Device)
test_y_21 = torch.tensor(Y_test_21, dtype=torch.float32, device=Device)

train_y = torch.tensor(Y_train_std, dtype=torch.float32, device=Device)
test_y = torch.tensor(Y_test_std, dtype=torch.float32, device=Device)


torch.set_default_dtype(torch.float32)

####################################################################


def run_mcmc_Uniform_NN(Models, row_idx, test_y, bounds, num_sampling=2000, warmup_step=1000, num_chains=1, device='cpu'):
    test_y = test_y.to(dtype=torch.float32, device=device)
    
    bounds = [
        (
            torch.tensor(b[0], dtype=torch.float32, device=device),
            torch.tensor(b[1], dtype=torch.float32, device=device)
        ) for b in bounds
    ]
    
    def model():
        params = []
        for i, (min_val, max_val) in enumerate(bounds):
            param_i = pyro.sample(f'param_{i}', dist.Uniform(min_val, max_val))
            params.append(param_i)
        
        theta = torch.stack(params)
        sigma = pyro.sample('sigma', dist.HalfNormal(5.0))
        mu_value = Models(theta.unsqueeze(0))

        
        y_obs = test_y[row_idx, :]
        pyro.sample('obs', dist.Normal(mu_value, sigma), obs=y_obs)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_sampling, warmup_steps=warmup_step, num_chains=num_chains)
    
    mcmc.run()
    
    return mcmc





mcmc_dir = 'Result/DNN_21_mcmc_result'
if not os.path.exists(mcmc_dir):
    os.makedirs(mcmc_dir)


NN_4 = Training.train_DNN_MSE(NN_models.NN_4,
                              train_x,
                              train_y_21,
                              num_iterations= 50000,
                              device= Device,
                              show_progress = True,
                              weight_decay = 0,
                              val_x=test_x,
                              val_y=test_y_21,
                              early_stopping = True,
                              patience = 1000,
                              val_check_interval = 100)

NN_4.eval()


for row_idx in range(test_y_21.shape[0]):
    input_point = test_y_21[row_idx, :]

    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_GPU(input_point, train_x, train_y_21, k=100)

    bounds = bound.get_bounds(local_train_x)
  
    mcmc_result_Uniform = run_mcmc_Uniform_NN(
        NN_4, 
        row_idx, test_y_21, bounds, 
        num_sampling=1200, warmup_step=300, num_chains=1, device=Device
    )

    posterior_samples_Uniform = mcmc_result_Uniform.get_samples()

    mcmc_file = os.path.join(mcmc_dir, f'result_{row_idx + 1}.pkl')
    with open(mcmc_file, 'wb') as f:
        pickle.dump(posterior_samples_Uniform, f)



# nohup python DNN_21_mcmc.py > DNN_21_mcmcout.log 2>&1 &

