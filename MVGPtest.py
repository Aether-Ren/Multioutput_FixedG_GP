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

# Y_train_8 = pd.read_csv('Data/Y_train_8.csv', header=None, delimiter=',').values
# Y_test_8 = pd.read_csv('Data/Y_test_8.csv', header=None, delimiter=',').values

Y_train_21 = pd.read_csv('Data/Y_train_std_21.csv', header=None, delimiter=',').values
Y_test_21 = pd.read_csv('Data/Y_test_std_21.csv', header=None, delimiter=',').values

Y_train_std = pd.read_csv('Data/Y_train_std.csv', header=None, delimiter=',').values
Y_test_std = pd.read_csv('Data/Y_test_std.csv', header=None, delimiter=',').values


train_x = torch.tensor(X_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)

# train_y_8 = torch.tensor(Y_train_8, dtype=torch.float32)
# test_y_8 = torch.tensor(Y_test_8, dtype=torch.float32)

train_y_21 = torch.tensor(Y_train_21, dtype=torch.float32)
test_y_21 = torch.tensor(Y_test_21, dtype=torch.float32)

train_y = torch.tensor(Y_train_std, dtype=torch.float32)
test_y = torch.tensor(Y_test_std, dtype=torch.float32)

Device = 'cpu'

num_latents_list = [8, 10, 12, 14]
num_inducing_list = [100, 150, 175, 200]

best_mse = float('inf')
best_params = None
results = []  # 存储所有成功组合及对应的 mse

for num_latents in num_latents_list:
    for num_inducing in num_inducing_list:
        try:
            # 训练模型
            MVGP_models, MVGP_likelihoods = Training.train_full_MultitaskVGP(
                train_x, train_y_21,
                num_latents=num_latents,
                num_inducing=num_inducing,
                lr_hyper=0.05,
                lr_variational=0.05,
                num_iterations=5000,
                patience=50,
                device=Device
            )
            # 得到预测值
            full_test_preds_MVGP = Prediction.preds_for_one_model(
                MVGP_models, MVGP_likelihoods, test_x.to(Device)
            ).cpu().detach().numpy()

            # 计算均方误差
            mse = np.mean((full_test_preds_MVGP.reshape(120, 21) - test_y_21.numpy()) ** 2)
            results.append((num_latents, num_inducing, mse))

            if mse < best_mse:
                best_mse = mse
                best_params = (num_latents, num_inducing)
            print(f"组合 num_latents={num_latents}, num_inducing={num_inducing} -> mse: {mse}")

        except Exception as e:
            # 出现异常，记录并跳过当前参数组合
            print(f"组合 num_latents={num_latents}, num_inducing={num_inducing} 训练失败: {e}")
            continue

print("最佳 MSE:", best_mse)
print("最佳参数组合 (num_latents, num_inducing):", best_params)