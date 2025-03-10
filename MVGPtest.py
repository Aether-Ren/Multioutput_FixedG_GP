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


import multiprocessing as mp
import time
def train_model(covar_type, num_latents, num_inducing, return_dict):
    try:
        # 调用训练函数（示例代码，根据实际情况修改）
        MVGP_models, MVGP_likelihoods = Training.train_full_MultitaskVGP(
            train_x, train_y_21,
            covar_type=covar_type,
            num_latents=num_latents,
            num_inducing=num_inducing,
            lr_hyper=0.05,
            lr_variational=0.05,
            num_iterations=5000,
            patience=50,
            device=Device
        )
        # 得到预测值并计算均方误差
        full_test_preds_MVGP = Prediction.preds_for_one_model(
            MVGP_models, MVGP_likelihoods, test_x.to(Device)
        ).cpu().detach().numpy()
        mse = np.mean((full_test_preds_MVGP.reshape(120, 21) - test_y_21.numpy()) ** 2)
        return_dict['mse'] = mse
    except Exception as e:
        return_dict['error'] = str(e)

if __name__ == '__main__':
    manager = mp.Manager()
    results = []  # 存储所有成功组合及对应的 mse
    best_mse = float('inf')
    best_params = None

    num_latents_list = [12, 14, 16, 18]
    num_inducing_list = [100, 150, 175, 200, 300, 400]
    covar_type_list = ['Matern5/2', 'RBF', 'RQ']

    for num_latents in num_latents_list:
        for num_inducing in num_inducing_list:
            for covar_type in covar_type_list:
                return_dict = manager.dict()
                # 将 covar_type 参数加入 args 中
                p = mp.Process(target=train_model, args=(covar_type, num_latents, num_inducing, return_dict))
                p.start()
                # 设置超时时间（单位：秒）
                p.join(timeout=120)
                if p.is_alive():
                    p.terminate()
                    print(f"组合 covar_type={covar_type}, num_latents={num_latents}, num_inducing={num_inducing} 超时并终止。")
                    continue
                if 'error' in return_dict:
                    print(f"组合 covar_type={covar_type}, num_latents={num_latents}, num_inducing={num_inducing} 训练失败: {return_dict['error']}")
                    continue

                mse = return_dict.get('mse')
                results.append((covar_type, num_latents, num_inducing, mse))
                print(f"组合 covar_type={covar_type}, num_latents={num_latents}, num_inducing={num_inducing} -> mse: {mse}")
                if mse < best_mse:
                    best_mse = mse
                    best_params = (covar_type, num_latents, num_inducing)

    print("最佳 MSE:", best_mse)
    print("最佳参数组合 (covar_type, num_latents, num_inducing):", best_params)