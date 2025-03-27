import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm
from torch.utils.data import TensorDataset, DataLoader

import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.GP_models as GP_models
import GP_functions.Tools as Tools
import GP_functions.FeatureE as FeatureE


def evaluate_full_dataset_loss_dgp(model, x_data, y_data, mll, device='cuda', batch_size=1024):

    model.eval()
    total_loss = 0.0
    dataset = TensorDataset(x_data, y_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model.predict(x_batch)[0]
            loss = mll(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(dataset)
    model.train()
    return avg_loss


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


Device = 'cuda'

num_hidden_dgp_dims_candidates = [10, 20]
inducing_num_candidates = [100, 300, 500]

# ============================== #

best_mse = float('inf')
best_params = None
best_dgp_model = None
best_dgp_likelihood = None

# ============================== #

# for hidden_dims in num_hidden_dgp_dims_candidates:

hidden_dims = [10, 10, 10]

for inducing_num in inducing_num_candidates:
    
    dgp_model= Training.train_DGP_minibatch(train_x, train_y_21, GP_models.DeepGP_4, num_hidden_dgp_dims=hidden_dims, inducing_num=inducing_num, num_iterations=10000, patience=50, 
                                                device='cuda',batch_size=512,eval_every=100,eval_batch_size=1024,lr=0.1)
    

    mse = evaluate_full_dataset_loss_dgp(dgp_model, test_x.to(Device), test_y_21.to(Device), torch.nn.MSELoss(), device='cuda', batch_size=20)
    
    print(f"HiddenDims={hidden_dims}, InducingNum={inducing_num}, MSE={mse:.4f}")
    

    if mse < best_mse:
        best_mse = mse
        best_params = {
            'num_hidden_dgp_dims': hidden_dims,
            'inducing_num': inducing_num
        }
        best_dgp_model = dgp_model


# ============================== #

print("=====================================")
print(f"best paramater: {best_params}")
print(f"MSE: {best_mse:.4f}")

# ============================== #

checkpoint = {
    'model_state_dict': best_dgp_model.state_dict(),
    'model_params': {
        'num_hidden_dgp_dims': best_params['num_hidden_dgp_dims'],
        'inducing_num': best_params['inducing_num'],
        'input_dim': train_x.size(1),  # 输入特征维度
        'output_dim': train_y_21.size(1)  # 输出维度 (21)
    }
}


save_path = 'best_dgp_4_checkpoint_21.pth'
torch.save(checkpoint, save_path)
print(f"save {save_path}")