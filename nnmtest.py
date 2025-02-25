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








from joblib import Parallel, delayed
import numpy as np

# 修改后的函数，新增 covar_type 和 feature_extractor_class 参数
def train_and_predict_NNMGP(row_idx, train_x, train_y, test_x, test_y, 
                            K_num=100, Device='cpu', PCA_trans='None', 
                            covar_type='RBF', feature_extractor_class=FeatureE.FeatureExtractor_4):
    # 取出当前测试行对应的目标值作为输入点
    input_point = test_y[row_idx, :]
    # 找到距离 input_point 最近的 K_num 个训练样本
    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k=K_num)

    # 使用指定的 covar_type 和 feature_extractor_class 训练模型
    NNMultitaskGP_models, NNMultitaskGP_likelihoods = Training.train_one_row_NNMultitaskGP(
        local_train_x, local_train_y, n_tasks=train_y.shape[1],
        feature_extractor_class=feature_extractor_class, covar_type=covar_type, 
        lr=0.05, num_iterations=5000, patience=10, device=Device
    )

    preds = Prediction.preds_for_one_model(
        NNMultitaskGP_models, NNMultitaskGP_likelihoods, 
        test_x[row_idx, :].unsqueeze(0).to(Device)
    ).squeeze().detach().numpy()

    if PCA_trans != 'None':
        # 对部分预测值进行逆变换
        first_column = preds[0]
        remaining_columns = preds[1:]
        remaining_columns = PCA_trans.inverse_transform(remaining_columns)
        preds = np.concatenate((first_column, remaining_columns), axis=1)
    
    return preds

# 定义评估函数，对单个参数组合计算整体 MSE
def evaluate_combination(K_num, covar_type, feature_extractor_class, train_x, train_y, test_x, test_y, Device='cpu', PCA_trans='None'):
    try:
        # 对每个测试样本并行调用预测函数
        results = Parallel(n_jobs=-1)(
            delayed(train_and_predict_NNMGP)(
                row_idx, train_x, train_y, test_x, test_y,
                K_num=K_num, Device=Device, PCA_trans=PCA_trans,
                covar_type=covar_type, feature_extractor_class=feature_extractor_class
            )
            for row_idx in range(test_y.shape[0])
        )
        full_test_preds = np.vstack(results)
        mse = np.mean((full_test_preds - test_y.numpy()) ** 2)
        return mse
    except Exception as e:
        # 如果某个组合出错，打印错误信息并跳过该组合
        print(f"K_num={K_num}, covar_type={covar_type}, feature_extractor_class={feature_extractor_class.__name__}; erro, {e}")
        return None

# 定义候选的参数值
K_values = [100, 200, 300, 400, 500]
covar_types = ['RBF', 'RQ', 'Matern5/2']
feature_extractors = [
    FeatureE.FeatureExtractor_1, 
    FeatureE.FeatureExtractor_2, 
    FeatureE.FeatureExtractor_3, 
    FeatureE.FeatureExtractor_4, 
    FeatureE.FeatureExtractor_5
]

# 用于存储每个有效组合的 MSE
results_dict = {}

# 遍历所有参数组合
for K in K_values:
    for cov in covar_types:
        for fe in feature_extractors:
            mse = evaluate_combination(K, cov, fe, train_x, train_y_21, test_x, test_y_21, Device='cpu', PCA_trans='None')
            if mse is not None:
                results_dict[(K, cov, fe.__name__)] = mse
                print(f"K_num={K}, covar_type={cov}, feature_extractor_class={fe.__name__}, MSE={mse}")

# 找到 MSE 最小的组合
if results_dict:
    best_params = min(results_dict, key=results_dict.get)
    best_mse = results_dict[best_params]
    print(f"Best K_num={best_params[0]}, covar_type={best_params[1]}, feature_extractor_class={best_params[2]}, MSE={best_mse}")
else:
    print("All wrong")
