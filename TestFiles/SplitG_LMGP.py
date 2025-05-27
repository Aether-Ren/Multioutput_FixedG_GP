import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import os


from statsmodels.graphics.tsaplots import plot_acf

import statsmodels


from pyro.ops.stats import (
    gelman_rubin,
    split_gelman_rubin,
    autocorrelation,
    effective_sample_size,
    resample,
    quantile,
    weighed_quantile
)

import pickle

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as transforms
from pyro.infer import MCMC, NUTS
import arviz as az



import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.GP_models as GP_models
import GP_functions.Tools as Tools
import GP_functions.FeatureE as FeatureE


Device = 'cpu'

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


row_idx = 10

input_point = test_y_21[row_idx,:]
local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y_21, k=500)

MultitaskGP_models, MultitaskGP_likelihoods = Training.train_one_row_MultitaskGP(
    local_train_x, local_train_y, n_tasks = local_train_y.shape[1], 
    covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=20, device=Device,disable_progbar=False)



bounds = bound.get_bounds(local_train_x)


def run_mcmc_Uniform_group(Pre_function, Models, Likelihoods,
                             row_idx, test_y, bounds,
                             true_params, groupN,
                             num_sampling=2000, warmup_step=1000,
                             num_chains=1, device='cpu'):
    if not (1 <= groupN <= 5):
        raise ValueError("groupN should between 1-5")

    test_y = test_y.to(dtype=torch.float32, device=device)
    true_params = torch.as_tensor(true_params, dtype=torch.float32, device=device)

    bounds = [
        (
            torch.tensor(lo, dtype=torch.float32, device=device),
            torch.tensor(hi, dtype=torch.float32, device=device),
        )
        for lo, hi in bounds
    ]

    sample_idx = list(range(2 * groupN))

    def model():
        theta = true_params.clone()

        for k in sample_idx:
            lo_k, hi_k = bounds[k]
            theta_k = pyro.sample(f"param_{k}", dist.Uniform(lo_k, hi_k))
            theta[k] = theta_k

        pred_dist = Pre_function(Models, Likelihoods, theta.unsqueeze(0))
        pyro.sample("obs", pred_dist, obs=test_y[row_idx])


    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_sampling,
        warmup_steps=warmup_step,
        num_chains=num_chains,
    )
    mcmc.run()
    return mcmc




true_params = test_x[row_idx].numpy()
groupN = 1

mcmc_result_1 = run_mcmc_Uniform_group(
    Prediction.preds_distribution_fast_pred_var, MultitaskGP_models, MultitaskGP_likelihoods, 
    row_idx, test_y_21, bounds, 
    true_params, groupN, 
    num_sampling=2000, warmup_step=1000, num_chains=1, device=Device
    )

samples_1 = mcmc_result_1.get_samples()

torch.save(samples_1, "mcmc_samples_group_1.pt")


groupN = 2

mcmc_result_1 = run_mcmc_Uniform_group(
    Prediction.preds_distribution_fast_pred_var, MultitaskGP_models, MultitaskGP_likelihoods, 
    row_idx, test_y_21, bounds, 
    true_params, groupN, 
    num_sampling=2000, warmup_step=1000, num_chains=1, device=Device
    )

samples_1 = mcmc_result_1.get_samples()

torch.save(samples_1, "mcmc_samples_group_2.pt")


groupN = 3

mcmc_result_1 = run_mcmc_Uniform_group(
    Prediction.preds_distribution_fast_pred_var, MultitaskGP_models, MultitaskGP_likelihoods, 
    row_idx, test_y_21, bounds, 
    true_params, groupN, 
    num_sampling=2000, warmup_step=1000, num_chains=1, device=Device
    )

samples_1 = mcmc_result_1.get_samples()

torch.save(samples_1, "mcmc_samples_group_3.pt")


groupN = 4

mcmc_result_1 = run_mcmc_Uniform_group(
    Prediction.preds_distribution_fast_pred_var, MultitaskGP_models, MultitaskGP_likelihoods, 
    row_idx, test_y_21, bounds, 
    true_params, groupN, 
    num_sampling=2000, warmup_step=1000, num_chains=1, device=Device
    )

samples_1 = mcmc_result_1.get_samples()

torch.save(samples_1, "mcmc_samples_group_4.pt")

groupN = 5

mcmc_result_1 = run_mcmc_Uniform_group(
    Prediction.preds_distribution_fast_pred_var, MultitaskGP_models, MultitaskGP_likelihoods, 
    row_idx, test_y_21, bounds, 
    true_params, groupN, 
    num_sampling=10000, warmup_step=1000, num_chains=1, device=Device
    )

samples_1 = mcmc_result_1.get_samples()

torch.save(samples_1, "mcmc_samples_group_5.pt")