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


X_train = pd.read_csv('RealCase/RealCase_X_train.csv', header=None, delimiter=',').values
X_test = pd.read_csv('RealCase/RealCase_X_test.csv', header=None, delimiter=',').values

# Y_train_pca = pd.read_csv('RealCase/RealCase_Y_train_pca.csv', header=None, delimiter=',').values
# Y_test_pca = pd.read_csv('RealCase/RealCase_Y_test_pca.csv', header=None, delimiter=',').values
# Realcase_data_pca = pd.read_csv('RealCase/RealCase_Y_pca.csv', header=None, delimiter=',').values

Y_train = pd.read_csv('RealCase/RealCase_Y_train_std.csv', header=None, delimiter=',').values
Y_test = pd.read_csv('RealCase/RealCase_Y_test_std.csv', header=None, delimiter=',').values
Realcase_data = pd.read_csv('RealCase/RealCase_Y.csv', header=None, delimiter=',').values



train_x = torch.tensor(X_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)

train_y = torch.tensor(Y_train, dtype=torch.float32)
test_y = torch.tensor(Y_test, dtype=torch.float32)
realcase_y = torch.tensor(Realcase_data, dtype=torch.float32)


torch.set_default_dtype(torch.float32)

Device = 'cpu'



output_file = 'RealCase/Result/MVGP_result.csv'
# mcmc_dir = 'LocalDisease/Result/MVGP_21_mcmc_result'
# if not os.path.exists(mcmc_dir):
#     os.makedirs(mcmc_dir)

if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,estimated_params\n')

checkpoint = torch.load('multitask_gp_checkpoint_Realcase.pth', map_location=Device)
model_params = checkpoint['model_params']

MVGP_models = GP_models.MultitaskVariationalGP(train_x, train_y, 
                                               num_latents=model_params['num_latents'],
                                               num_inducing=model_params['num_inducing'],  
                                               covar_type=model_params['covar_type']).to(Device)

MVGP_models.load_state_dict(checkpoint['model_state_dict'])

MVGP_likelihoods = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1]).to(Device)
MVGP_likelihoods.load_state_dict(checkpoint['likelihood_state_dict'])

MVGP_models.eval()
MVGP_likelihoods.eval()

for row_idx in range(realcase_y.shape[0]):
    input_point = realcase_y[row_idx, :]

    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k=100)

    bounds = bound.get_bounds(local_train_x)

    estimated_params_tmp, _ = Estimation.multi_start_estimation(
        MVGP_models, MVGP_likelihoods, row_idx, realcase_y, bounds,
        Estimation.estimate_params_for_one_model_Adam, num_starts=16, num_iterations=2000, lr=0.01,
        patience=10, attraction_threshold=0.1, repulsion_strength=0.1, device=Device
    )

    with open(output_file, 'a') as f:
        # f.write(f"{row_idx + 1},\"{list(preds_tmp)}\",\"{list(estimated_params_tmp.detach().numpy())}\"\n")
        f.write(f"{row_idx + 1},\"{list(estimated_params_tmp)}\"\n")

    # mcmc_result_Uniform = Estimation.run_mcmc_Uniform(
    #     Prediction.preds_distribution, MVGP_models, MVGP_likelihoods, 
    #     row_idx, realcase_y_pca, bounds, 
    #     num_sampling=1200, warmup_step=300, num_chains=1, device=Device
    # )
    # posterior_samples_Uniform = mcmc_result_Uniform.get_samples()


    # mcmc_file = os.path.join(mcmc_dir, f'result_{row_idx + 1}.pkl')
    # with open(mcmc_file, 'wb') as f:
    #     pickle.dump(posterior_samples_Uniform, f)


# nohup python RealCase_PE.py > RealCase_PEout.log 2>&1 &