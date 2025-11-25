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
X_test = pd.read_csv('LocalDisease/X_1_1.csv', header=None, delimiter=',').values

Y_train_pca = pd.read_csv('Data/Y_train_std_21.csv', header=None, delimiter=',').values
Y_test_21 = pd.read_csv('LocalDisease/Y_data_1_1_pca.csv', header=None, delimiter=',').values

# X_edge = pd.read_csv('Data/X_edge.csv', header=None, delimiter=',').values
# Y_edge_std_pca = pd.read_csv('LocalDisease/Y_edge_std_pca.csv', header=None, delimiter=',').values

# X_train = np.vstack([X_train, X_edge])
# Y_train_pca = np.vstack([Y_train_pca, Y_edge_std_pca])


train_x = torch.tensor(X_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)

train_y_21 = torch.tensor(Y_train_pca, dtype=torch.float32)
test_y_21 = torch.tensor(Y_test_21, dtype=torch.float32)



torch.set_default_dtype(torch.float32)

####################################################################

Device = 'cuda'


output_file = 'LocalDisease/Result/MVGP_X_1_1_result.csv'

if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,test_preds,estimated_params\n')

checkpoint = torch.load('multitask_gp_checkpoint_21.pth', map_location=Device)
model_params = checkpoint['model_params']

MVGP_models = GP_models.MultitaskVariationalGP(train_x, train_y_21, 
                                               num_latents=model_params['num_latents'], 
                                               num_inducing=model_params['num_inducing'], 
                                               covar_type=model_params['covar_type']).to(Device)

MVGP_models.load_state_dict(checkpoint['model_state_dict'])

MVGP_likelihoods = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_21.shape[1]).to(Device)
MVGP_likelihoods.load_state_dict(checkpoint['likelihood_state_dict'])

MVGP_models.eval()
MVGP_likelihoods.eval()


for row_idx in range(test_y_21.shape[0]):
    input_point = test_y_21[row_idx, :]

    # local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y_21, k=100)
    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_GPU(input_point, train_x, train_y_21, k=100)

    preds_tmp = Prediction.preds_for_one_model(
        MVGP_models, MVGP_likelihoods, test_x[row_idx,:].unsqueeze(0).to(Device)
        ).cpu().detach().numpy()


    bounds = bound.get_bounds(local_train_x)

    estimated_params_tmp, _ = Estimation.multi_start_estimation(
        MVGP_models, MVGP_likelihoods, row_idx, test_y_21, bounds,
        Estimation.estimate_params_for_one_model_Adam, num_starts=16, num_iterations=2000, lr=0.01,
        patience=10, attraction_threshold=0.1, repulsion_strength=0.1, device=Device
    )

    with open(output_file, 'a') as f:
        # f.write(f"{row_idx + 1},\"{list(preds_tmp)}\",\"{list(estimated_params_tmp.detach().numpy())}\"\n")
        f.write(f"{row_idx + 1},\"{list(preds_tmp)}\",\"{list(estimated_params_tmp)}\"\n")





# nohup python MVGP_LocalD_PE.py > MVGP_LocalD_PEout.log 2>&1 &