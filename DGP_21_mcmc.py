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

Device = 'cuda'



mcmc_dir = 'Result/DGP_21_mcmc_result'
if not os.path.exists(mcmc_dir):
    os.makedirs(mcmc_dir)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_x, train_y_21 = train_x.to(Device), train_y_21.to(Device)
test_x, test_y_21 = test_x.to(Device), test_y_21.to(Device)

ckpt_path = 'final_dgp_2_checkpoint_21.pth'
checkpoint = torch.load(ckpt_path, map_location=Device)

state_dict   = checkpoint['model_state_dict']
model_params = checkpoint['model_params']

dgp_model = GP_models.DeepGP2(
    train_x, train_y_21, 
    hidden_dim = model_params['num_hidden_dgp_dims'], 
    inducing_num = model_params['inducing_num'], 
    covar_types = model_params['covar_types']
).to(Device)

dgp_model.load_state_dict(state_dict, strict=False)
dgp_model.eval()
dgp_model.likelihood.eval()


for row_idx in range(test_y_21.shape[0]):
    input_point = test_y_21[row_idx, :]

    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_GPU(input_point, train_x, train_y_21, k=500)

    bounds = bound.get_bounds(local_train_x)
  
    mcmc_result_Uniform = Estimation.run_mcmc_Uniform_dgp(
        Prediction.dgp_predict_cov, dgp_model, 
        row_idx, test_y_21, bounds, 
        num_sampling=1200, warmup_step=300, num_chains=1, device=Device
    )

    posterior_samples_Uniform = mcmc_result_Uniform.get_samples()

    mcmc_file = os.path.join(mcmc_dir, f'result_{row_idx + 1}.pkl')
    with open(mcmc_file, 'wb') as f:
        pickle.dump(posterior_samples_Uniform, f)





# nohup python DGP_21_mcmc.py > DGP_21_mcmcout.log 2>&1 &