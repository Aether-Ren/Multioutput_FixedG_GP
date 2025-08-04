import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm


import os
import pickle

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


torch.set_default_dtype(torch.float32)

####################################################################

Device = 'cpu'


output_file = 'Result/L.DKMGP_21_result.csv'
mcmc_dir = 'Result/L.DKMGP_21_mcmc_result'
if not os.path.exists(mcmc_dir):
    os.makedirs(mcmc_dir)

if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,test_preds,estimated_params\n')

for row_idx in range(test_y_21.shape[0]):
    input_point = test_y_21[row_idx, :]

    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y_21, k=500)

    MultitaskGP_models, MultitaskGP_likelihoods = Training.train_one_row_NNMultitaskGP(
        local_train_x, local_train_y, n_tasks = local_train_y.shape[1], 
        feature_extractor_class = FeatureE.FeatureExtractor_4, covar_type = 'RQ', 
        lr=0.05, num_iterations=5000, patience=10, device = Device)

    preds_tmp = Prediction.preds_for_one_model(
        MultitaskGP_models, MultitaskGP_likelihoods, test_x[row_idx,:].unsqueeze(0).to(Device)
        ).cpu().detach().numpy()

    # local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y_21, k=100)
    bounds = bound.get_bounds(local_train_x)

    estimated_params_tmp, _ = Estimation.multi_start_estimation(
        MultitaskGP_models, MultitaskGP_likelihoods, row_idx, test_y_21, bounds,
        Estimation.estimate_params_for_one_model_Adam, num_starts=4, num_iterations=1000, lr=0.01,
        patience=10, attraction_threshold=0.1, repulsion_strength=0.1, device=Device
    )

    with open(output_file, 'a') as f:
        f.write(f"{row_idx + 1},\"{list(preds_tmp)}\",\"{list(estimated_params_tmp)}\"\n")

    mcmc_result_Uniform = Estimation.run_mcmc_Uniform_initial_params(
        Prediction.preds_distribution_fast_pred_var, MultitaskGP_models, MultitaskGP_likelihoods, 
        row_idx, test_y_21, bounds, 
        num_sampling=1500, warmup_step=500, num_chains=1, device=Device, initial_params=estimated_params_tmp
    )
    posterior_samples_Uniform = mcmc_result_Uniform.get_samples()


    mcmc_file = os.path.join(mcmc_dir, f'result_{row_idx + 1}.pkl')
    with open(mcmc_file, 'wb') as f:
        pickle.dump(posterior_samples_Uniform, f)




# nohup python DkMGP_21.py > DkMGP_21out.log 2>&1 &