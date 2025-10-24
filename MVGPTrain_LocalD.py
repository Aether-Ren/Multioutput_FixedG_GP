import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm
from linear_operator import settings

import pyro
import math
import pickle
import time
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az
import seaborn as sns

import os

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

X_edge = pd.read_csv('Data/X_edge.csv', header=None, delimiter=',').values


Y_train_pca = pd.read_csv('LocalDisease/Y_train_std_pca.csv', header=None, delimiter=',').values
Y_test_pca = pd.read_csv('LocalDisease/Y_test_std_pca.csv', header=None, delimiter=',').values

Y_edge_std_pca = pd.read_csv('LocalDisease/Y_edge_std_pca.csv', header=None, delimiter=',').values

X_train = np.vstack([X_train, X_edge])
Y_train_pca = np.vstack([Y_train_pca, Y_edge_std_pca])


train_x = torch.tensor(X_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)


train_y_pca = torch.tensor(Y_train_pca, dtype=torch.float32)
test_y_pca = torch.tensor(Y_test_pca, dtype=torch.float32)



Device = 'cuda'

num_latents_candidates = [24, 32]
num_inducing_candidates = [300, 400, 500]
covar_type_candidates = ['RBF', 'RQ']

best_mse = float('inf')
best_params = None
best_model = None
best_likelihood = None

for num_latents in num_latents_candidates:
    for num_inducing in num_inducing_candidates:
        for covar_type in covar_type_candidates:
            MVGP_models, MVGP_likelihoods = Training.train_MultitaskVGP_minibatch(
                train_x=train_x.to(Device),
                train_y=train_y_pca.to(Device),
                covar_type=covar_type,
                num_latents=num_latents,
                num_inducing=num_inducing,
                lr_hyper=0.01,
                lr_variational=0.1,
                num_iterations=10000,
                patience=10,
                device=Device,
                batch_size=512,
                eval_every=100,
                eval_batch_size=1024
            )
            
            full_test_preds_MVGP = Prediction.preds_for_one_model(
                MVGP_models,
                MVGP_likelihoods,
                test_x.to(Device)
            ).cpu().detach().numpy()
            
            mse = np.mean((full_test_preds_MVGP.reshape(-1, 21) - test_y_pca.numpy()) ** 2)
            print(f"Done: covar_type={covar_type}, num_latents={num_latents}, "
                  f"num_inducing={num_inducing}, MSE={mse:.4f}")
            
            if mse < best_mse:
                best_mse = mse
                best_params = {
                    'covar_type': covar_type,
                    'num_latents': num_latents,
                    'num_inducing': num_inducing
                }
                best_model = MVGP_models  # 保留当下最好的模型
                best_likelihood = MVGP_likelihoods

print("=====================================")
print(f"best paramaters: {best_params}")
print(f"best MSE: {best_mse:.4f}")

# ========== 训练结束后保存最优模型 ==========
checkpoint = {
    'model_state_dict': best_model.state_dict(),
    'likelihood_state_dict': best_likelihood.state_dict(),
    'model_params': {
        'num_latents': best_params['num_latents'],
        'num_inducing': best_params['num_inducing'],
        'covar_type': best_params['covar_type'],
        'input_dim': train_x.size(1),
        'num_tasks': train_y_pca.size(1)
    }
}

torch.save(checkpoint, 'multitask_gp_checkpoint_LocalD.pth')
print("save 'multitask_gp_checkpoint_LocalD.pth'")


# nohup python MVGPTrain_LocalD.py > MVGPTrain_LocalDout.log 2>&1 &