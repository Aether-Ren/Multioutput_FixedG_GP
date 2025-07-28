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


output_file = 'Result/DGP_21_point_result.csv'


if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,test_preds,estimated_params\n')

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

    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_GPU(input_point, train_x, train_y_21, k=100)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mean, var = dgp_model.predict(test_x[row_idx,:].unsqueeze(0).to(Device))

    preds_tmp = mean.cpu().detach().numpy()


    bounds = bound.get_bounds(local_train_x)
  
    estimated_params_tmp, _ = Estimation.multi_start_estimation_DModel(dgp_model, row_idx, test_y_21, bounds,
                                                                   Estimation.estimate_params_for_DGP_Adam, num_starts=8, num_iterations=2000, lr=0.01,
                                                                   patience=10, attraction_threshold=0.1, repulsion_strength=0.1, device=Device)

    with open(output_file, 'a') as f:
        f.write(f"{row_idx + 1},\"{list(preds_tmp)}\",\"{list(estimated_params_tmp)}\"\n")
        # f.write(f"{row_idx + 1},\"{list(preds_tmp)}\",\"{list(estimated_params_tmp)}\"\n")





# nohup python DGP_21_point.py > DGP_21_pointout.log 2>&1 &