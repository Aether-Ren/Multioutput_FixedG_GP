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

from torch.utils.data import TensorDataset, DataLoader
import itertools

import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.NN_models as NN_models
import GP_functions.Tools as Tools
import GP_functions.FeatureE as FeatureE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train = pd.read_csv('Data/X_train.csv', header=None, delimiter=',').values
X_test = pd.read_csv('Data/X_test.csv', header=None, delimiter=',').values

Y_train_21 = pd.read_csv('Data/Y_train_std_21.csv', header=None, delimiter=',').values
Y_test_21 = pd.read_csv('Data/Y_test_std_21.csv', header=None, delimiter=',').values

Y_train = pd.read_csv('Data/Y_train_std.csv', header=None, delimiter=',').values
Y_test = pd.read_csv('Data/Y_test_std.csv', header=None, delimiter=',').values


# train_x = torch.tensor(X_train, dtype=torch.float32)
# test_x = torch.tensor(X_test, dtype=torch.float32)

# train_y_21 = torch.tensor(Y_train_21, dtype=torch.float32)
# test_y_21 = torch.tensor(Y_test_21, dtype=torch.float32)


train_x = torch.from_numpy(X_train).float().to(device)
test_x  = torch.from_numpy(X_test).float().to(device)
train_y_21 = torch.from_numpy(Y_train_21).float().to(device)
test_y_21  = torch.from_numpy(Y_test_21).float().to(device)


# train_y = torch.tensor(Y_train, dtype=torch.float32)
# test_y = torch.tensor(Y_test, dtype=torch.float32)


# torch.set_default_dtype(torch.float32)




def evaluate_mse(model, x, y):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        # model(x) returns a Distribution when y=None
        pred_dist = model(x)
        preds = pred_dist.mean
    # MSE over all outputs
    return torch.mean((preds - y) ** 2).item()

# Dictionary of your model classes
model_classes = {
    'BNN_2': NN_models.BNN_2,
    'BNN_3': NN_models.BNN_3,
    'BNN_4': NN_models.BNN_4,
    'BNN_5': NN_models.BNN_5
}

results = {}
best_mse = float('inf')
best_name = None
best_model = None
best_guide = None
best_param_state = None

for name, ModelClass in model_classes.items():
    print(f"\n--- Training {name} BNN ---")
    # This will clear pyro.param_store internally
    model, guide = Training.train_BNN_minibatch(
        NN_model=ModelClass,
        full_train_x=train_x,
        full_train_y=train_y_21,
        num_iterations=50000,       # adjust as needed
        batch_size=256,
        device=device,
        show_progress=True,
        lr=1e-2,
        early_stopping=False
    )
    model.to(device)
    guide.to(device)
    mse = evaluate_mse(model, test_x, test_y_21)
    print(f"{name} MSE on test set: {mse:.4f}")
    
    results[name] = mse
    # Snapshot if best so far
    if mse < best_mse:
        best_mse = mse
        best_name = name
        best_model = model
        best_guide = guide
        # grab the whole param store state so we can reload later
        best_param_state = pyro.get_param_store().get_state()

print(f"\n=== Best model: {best_name} (MSE={best_mse:.4f}) ===")

# Save best model & guide weights
torch.save(best_model.state_dict(), f"best_model_{best_name}.pt")
torch.save(best_guide.state_dict(), f"best_guide_{best_name}.pt")

# Save pyro parameter store (covers any learned std‑dev or ARD scales, etc.)
import pickle
with open(f"best_param_store_{best_name}.pkl", "wb") as f:
    pickle.dump(best_param_state, f)

print("Saved:")
print(f"  • best_model_{best_name}.pt")
print(f"  • best_guide_{best_name}.pt")
print(f"  • best_param_store_{best_name}.pkl")



# nohup python BNNTrain.py > BNNTrainout.log 2>&1 &