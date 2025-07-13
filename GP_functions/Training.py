"""
File: Training.py
Author: Hongjin Ren
Description: Train the Gaussian process models

"""

#############################################################################
## Package imports
#############################################################################
import torch
import gpytorch
import tqdm as tqdm
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import itertools

import GP_functions.GP_models as GP_models
import GP_functions.NN_models as NN_models
import GP_functions.Loss_function as Loss_function
import GP_functions.Tools as Tools

from joblib import Parallel, delayed



#############################################################################
## Training LocalGP
#############################################################################

def train_one_column_LocalGP(local_train_x, local_train_y, covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=10, device='cpu'):

    local_train_x = local_train_x.to(device)
    local_train_y = local_train_y.to(device)

    # local_train_y_column = local_train_y[:, column_idx]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP_models.LocalGP(local_train_x, local_train_y, likelihood, covar_type)


    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    counter = 0
    # iterator = tqdm.tqdm(range(num_iterations))

    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(local_train_x)
        loss = -mll(output, local_train_y)
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if loss.item() <= best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)
                break

    return model, likelihood



def train_one_row_LocalGP(train_x, train_y, test_y, row_idx, covar_type = 'RBF', k_num = 100, lr=0.05, num_iterations=5000, patience=10, device='cpu'):
    # Train the all columns of the output
    Models = []
    Likelihoods = []
    input_point = test_y[row_idx,:]
    for column_idx in range(train_y.shape[1]):
        local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point[column_idx:(column_idx+1)], train_x, train_y[:,column_idx:(column_idx+1)], k = k_num)
        model, likelihood = train_one_column_LocalGP(local_train_x, local_train_y.squeeze(), covar_type, lr, num_iterations, patience, device)
        Models.append(model)
        Likelihoods.append(likelihood)
    return Models, Likelihoods


def train_one_row_LocalGP_Parallel(train_x, train_y, test_y, row_idx, covar_type = 'RBF', k_num=100, lr=0.05, num_iterations=5000, patience=10, device='cpu'):
    # Helper function to train a single column
    def train_column(column_idx):
        input_point = test_y[row_idx, :]
        local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point[column_idx:(column_idx+1)], train_x, train_y[:,column_idx:(column_idx+1)], k=k_num)
        model, likelihood = train_one_column_LocalGP(local_train_x, local_train_y.squeeze(), covar_type, lr, num_iterations, patience, device)
        return model, likelihood
    
    # Parallelize the training of all columns
    results = Parallel(n_jobs=17)(delayed(train_column)(column_idx) for column_idx in range(train_y.shape[1]))
    
    # Unzip the results
    Models, Likelihoods = zip(*results)
    return list(Models), list(Likelihoods)


#############################################################################
## 
#############################################################################


def train_one_column_VGP(column_idx, full_train_x, full_train_y, inducing_points, covar_type = 'RBF', lr=0.01, num_iterations=5000, patience=10, device='cpu'):
    
    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)
    inducing_points = inducing_points.to(device)

    train_y_column = full_train_y[:, column_idx]
    
    model = GP_models.VGPModel(full_train_x, inducing_points=inducing_points, covar_type = covar_type)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     likelihood = likelihood.cuda()

    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()


    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y_column.size(0), lr=0.1)

    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)


    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y_column.size(0))

    best_loss = float('inf')
    counter = 0
    # iterator = tqdm.tqdm(range(num_iterations))


    # for i in iterator:
    for i in range(num_iterations):
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = model(full_train_x)

        loss = -mll(output, train_y_column)
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()  
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)  
                break

    return model, likelihood



def train_full_VGP(train_x, train_y, inducing_points, covar_type = 'RBF', lr=0.01, num_iterations=5000, patience=10, device='cpu'):
    # Train the all columns of the output
    Models = []
    Likelihoods = []
    for column_idx in range(train_y.shape[1]):
        model, likelihood = train_one_column_VGP(column_idx, train_x, train_y, inducing_points, covar_type, lr, num_iterations, patience, device)
        Models.append(model)
        Likelihoods.append(likelihood)
    return Models, Likelihoods


def train_full_VGP_Parallel(train_x, train_y, inducing_points, covar_type = 'RBF', lr=0.01, num_iterations=5000, patience=10, device='cpu'):
    # Helper function to train a single column
    def train_column(column_idx):
        model, likelihood = train_one_column_VGP(column_idx, train_x, train_y, inducing_points, covar_type, lr, num_iterations, patience, device)
        return model, likelihood
    
    # Parallelize the training of all columns
    results = Parallel(n_jobs=-1)(delayed(train_column)(column_idx) for column_idx in range(train_y.shape[1]))
    
    # Unzip the results
    Models, Likelihoods = zip(*results)
    return list(Models), list(Likelihoods)



#############################################################################
## 
#############################################################################

def train_full_MultitaskVGP(train_x, train_y, covar_type = 'Matern3/2', num_latents=20, num_inducing=100, lr_hyper=0.01, lr_variational=0.1, num_iterations=5000, patience=10, device='cpu'):
    """
    Training a multi-task variational Gaussian process model.
    
    Parameters.
    - train_x: Input features of the training data.
    - train_y: target value of the training data.
    - num_latents: number of latent functions.
    - num_inducing: number of induced points.
    - lr_hyper: Learning rate of the hyperparameter optimiser.
    - lr_variational: Learning rate of the variational optimiser.
    - num_iterations: number of training iterations.
    - patience: The patience value for early stopping.
    
    Returns: The trained model and the likelihood function.
    - The trained model and likelihood function.
    """
 
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    model = GP_models.MultitaskVariationalGP(train_x, train_y, num_latents=num_latents, num_inducing=num_inducing, covar_type = covar_type)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     likelihood = likelihood.cuda()

    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=lr_variational)
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=lr_hyper)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    best_loss = float('inf')
    counter = 0

    iterator = tqdm.tqdm(range(num_iterations))

    for i in iterator:
    # for i in range(num_iterations):
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)
                break

    return model, likelihood


#################
##
################


def evaluate_full_dataset_loss(model, likelihood, mll, train_x, train_y, batch_size=1024, device='cpu'):
    model.eval()
    likelihood.eval()
    total_loss = 0.0
    dataset = TensorDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = -mll(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)
    
    avg_loss = total_loss / len(dataset)
    model.train()
    likelihood.train()
    return avg_loss

def train_MultitaskVGP_minibatch(train_x, train_y, covar_type='Matern3/2', num_latents=20, num_inducing=100, 
                           lr_hyper=0.01, lr_variational=0.1, num_iterations=1000, patience=10, 
                           device='cpu', batch_size=256, eval_every=100, eval_batch_size=1024):


    train_x = train_x.to(device)
    train_y = train_y.to(device)

    model = GP_models.MultitaskVariationalGP(
        train_x, train_y, 
        num_latents=num_latents, 
        num_inducing=num_inducing, 
        covar_type=covar_type
    ).to(device)
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=train_y.shape[1]
    ).to(device)

    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(),
        num_data=train_y.size(0),
        lr=lr_variational
    )
    
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()}
    ], lr=lr_hyper)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))


    best_loss = float('inf')
    counter = 0
    best_state = model.state_dict()
    data_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True
    )
    minibatch_iter = itertools.cycle(data_loader)

    with tqdm.tqdm(total=num_iterations, desc="Training") as pbar:
        for step in range(num_iterations):

            x_batch, y_batch = next(minibatch_iter)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            variational_ngd_optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            variational_ngd_optimizer.step()

            hyperparameter_optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            hyperparameter_optimizer.step()

            if (step + 1) % eval_every == 0 or step == num_iterations - 1:
                current_loss = evaluate_full_dataset_loss(
                    model, likelihood, mll,
                    train_x, train_y,
                    batch_size=eval_batch_size,
                    device=device
                )
                
                pbar.set_postfix(full_loss=current_loss)
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        model.load_state_dict(best_state)
                        pbar.update(num_iterations - step - 1)
                        break

            pbar.update(1)

    return model, likelihood


#############################################################################
## Train Multitask GP Model
#############################################################################



def train_one_row_MultitaskGP(local_train_x, local_train_y, n_tasks, covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=10, device='cpu', disable_progbar=True):

    local_train_x = local_train_x.to(device)
    local_train_y = local_train_y.to(device)


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = GP_models.MultitaskGPModel(local_train_x, local_train_y, likelihood, n_tasks, covar_type)

    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    best_loss = float('inf')
    counter = 0
    iterator = tqdm.tqdm(range(num_iterations), disable=disable_progbar)

    for i in iterator:
    # for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(local_train_x)
        loss = -mll(output, local_train_y)
        loss.backward()
        if not disable_progbar:
            iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()  
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)  
                break

    return model, likelihood






def train_one_row_MultitaskGP_lcm(local_train_x, local_train_y, n_tasks, lr=0.05, num_iterations=5000, patience=10, device='cpu'):

    local_train_x = local_train_x.to(device)
    local_train_y = local_train_y.to(device)


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = GP_models.MultitaskGPModel_lcm(local_train_x, local_train_y, likelihood, n_tasks)

    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    best_loss = float('inf')
    counter = 0
    iterator = tqdm.tqdm(range(num_iterations))

    for i in iterator:  
        optimizer.zero_grad()
        output = model(local_train_x)
        loss = -mll(output, local_train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()  
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)  
                break

    return model, likelihood


#############################################################################
## Train NN + Multitask GP Model
#############################################################################



def train_one_row_NNMultitaskGP(local_train_x, local_train_y, n_tasks, feature_extractor_class, covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=10, device='cuda'):

    local_train_x = local_train_x.to(device)
    local_train_y = local_train_y.to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = GP_models.NNMultitaskGP(local_train_x, local_train_y, likelihood, n_tasks, feature_extractor_class, covar_type)

    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    best_loss = float('inf')
    counter = 0

    # iterator = tqdm.tqdm(range(num_iterations))
    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(local_train_x)
        loss = -mll(output, local_train_y)
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()  
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)  
                break

    return model, likelihood


#############################################################################
## Train NN + Local GP Model
#############################################################################

def train_one_column_NNLocalGP(local_train_x, local_train_y, feature_extractor_class, covar_type = 'RBF', lr=0.01, num_iterations=5000, patience=10, device='cuda'):

    local_train_x = local_train_x.to(device)
    local_train_y = local_train_y.to(device)

    # local_train_y_column = local_train_y[:, column_idx]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP_models.NNLocalGP(local_train_x, local_train_y, likelihood, feature_extractor_class, covar_type)

    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    counter = 0

    # iterator = tqdm.tqdm(range(num_iterations))

    # for i in iterator:  
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(local_train_x)
        loss = -mll(output, local_train_y)
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()  
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)  
                break

    return model, likelihood




def train_one_row_NNLocalGP(train_x, train_y, test_y, row_idx, feature_extractor_class, covar_type = 'RBF', k_num = 100, lr=0.01, num_iterations=5000, patience=10, device='cuda'):
    # Train the all columns of the output
    Models = []
    Likelihoods = []
    input_point = test_y[row_idx,:]
    for column_idx in range(train_y.shape[1]):
        local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point[column_idx:(column_idx+1)], train_x, train_y[:,column_idx:(column_idx+1)], k = k_num)
        model, likelihood = train_one_column_NNLocalGP(local_train_x, local_train_y.squeeze(), feature_extractor_class, covar_type, lr, num_iterations, patience, device)
        Models.append(model)
        Likelihoods.append(likelihood)
    return Models, Likelihoods




def train_one_row_NNLocalGP_Parallel(train_x, train_y, test_y, row_idx, feature_extractor_class, covar_type = 'RBF', k_num=100, lr=0.01, num_iterations=5000, patience=10, device='cuda'):
    # Helper function to train a single column
    def train_column(column_idx):
        input_point = test_y[row_idx, :]
        local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point[column_idx:(column_idx+1)], train_x, train_y[:,column_idx:(column_idx+1)], k=k_num)
        model, likelihood = train_one_column_NNLocalGP(local_train_x, local_train_y.squeeze(), feature_extractor_class, covar_type, lr, num_iterations, patience, device)
        return model, likelihood
    
    # Parallelize the training of all columns
    results = Parallel(n_jobs=17)(delayed(train_column)(column_idx) for column_idx in range(train_y.shape[1]))
    
    # Unzip the results
    Models, Likelihoods = zip(*results)
    return list(Models), list(Likelihoods)

#############################################################################
## Train DNN Model
#############################################################################


def train_DNN_MSE(
    NN_model,
    full_train_x,
    full_train_y,
    num_iterations= 50000,
    device= 'cuda',
    show_progress = True,
    weight_decay = 0.2,
    val_x=None,
    val_y=None,
    early_stopping = False,
    patience = 1000,
    val_check_interval = 100
):



    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)
    if early_stopping and (val_x is not None and val_y is not None):
        val_x = val_x.to(device)
        val_y = val_y.to(device)
    else:
        early_stopping = False

    model = NN_model(full_train_x, full_train_y).to(device)
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.2,
        weight_decay=weight_decay  # L2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )


    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    iterator = tqdm.tqdm(range(num_iterations), disable=not show_progress)
    for i in iterator:
        optimizer.zero_grad()
        output = model(full_train_x)
        loss = criterion(output, full_train_y)
        loss.backward()
        optimizer.step()

        scheduler.step(loss)

        if early_stopping and (i + 1) % val_check_interval == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(val_x)
                val_loss = criterion(val_out, val_y)
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if show_progress:
                    iterator.write(f"Early stopping at iter {i+1}, best val loss: {best_val_loss:.4f}")

                model.load_state_dict(best_state)
                break

        if show_progress:
            postfix = {'train_loss': loss.item()}
            if early_stopping and (i + 1) % val_check_interval == 0:
                postfix['val_loss'] = best_val_loss.item() if best_state is not None else None
            iterator.set_postfix(**postfix)

    return model



def train_DNN_Euclidean(NN_model, full_train_x, full_train_y, num_iterations = 50000, device='cuda'):
    
    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)

    model = NN_model(full_train_x, full_train_y)


    model = model.to(device)
    model.train()


    criterion = Loss_function.euclidean_distance_loss 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 100)
    iterator = tqdm.tqdm(range(num_iterations))


    for i in iterator:  
        optimizer.zero_grad()
        output = model(full_train_x)
        loss = criterion(output, full_train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        scheduler.step(loss)

    return model







#############################################################################
## Train DGP Model
#############################################################################



def evaluate_full_dataset_loss_dgp(model, x_data, y_data, mll, device='cuda', batch_size=1024):

    model.eval()
    total_loss = 0.0
    dataset = TensorDataset(x_data, y_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = -mll(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(dataset)
    model.train()
    return avg_loss


# def train_DGP_minibatch(
#     full_train_x, 
#     full_train_y, 
#     DGP_model,
#     num_hidden_dgp_dims=4, 
#     inducing_num=500, 
#     num_iterations=2000, 
#     patience=50, 
#     device='cuda',
#     batch_size=32,
#     eval_every=100,
#     eval_batch_size=1024,
#     lr=0.1
# ):
#     """
#     训练Deep GP (2层) 的完整流程，支持小批量训练、早停、全数据集评估和学习率调度。
    
#     参数说明：
#     - full_train_x, full_train_y: 训练数据
#     - num_hidden_dgp_dims: Deep GP中隐藏层维度
#     - inducing_num: 每层诱导点数量
#     - num_iterations: 总迭代次数上限
#     - patience: 早停耐心值 (评估损失连续多少次不下降就停止)
#     - device: 'cpu' 或 'cuda'
#     - batch_size: 小批量训练时的批量大小
#     - eval_every: 每隔多少次迭代进行一次全数据评估
#     - eval_batch_size: 进行全数据评估时的批量大小
#     - lr: 初始学习率
#     """

#     full_train_x = full_train_x.to(device)
#     full_train_y = full_train_y.to(device)


#     model = DGP_model(
#         full_train_x.shape, 
#         full_train_y, 
#         num_hidden_dgp_dims, 
#         inducing_num
#     ).to(device)

#     model.train()


#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     mll = gpytorch.mlls.DeepApproximateMLL(
#         gpytorch.mlls.VariationalELBO(
#             model.likelihood,
#             model,
#             num_data=full_train_y.size(0)
#         )
#     )
#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     #     optimizer, 
#     #     mode='min', 
#     #     factor=0.5, 
#     #     patience=25
#     # )


#     best_loss = float('inf')
#     best_state = model.state_dict()
#     counter = 0


#     data_loader = DataLoader(
#         TensorDataset(full_train_x, full_train_y),
#         batch_size=batch_size,
#         shuffle=True
#     )
#     minibatch_iter = itertools.cycle(data_loader)


#     with tqdm.tqdm(total=num_iterations, desc="Training DGP_2") as pbar:
#         for step in range(num_iterations):
#             x_batch, y_batch = next(minibatch_iter)
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)

#             optimizer.zero_grad()
#             output = model(x_batch)
#             loss = -mll(output, y_batch)
#             loss.backward()
#             optimizer.step()

#             if (step + 1) % eval_every == 0 or (step == num_iterations - 1):
#                 current_loss = evaluate_full_dataset_loss_dgp(
#                     model=model,
#                     x_data=full_train_x,
#                     y_data=full_train_y,
#                     mll=mll,
#                     device=device,
#                     batch_size=eval_batch_size
#                 )
#                 pbar.set_postfix(full_loss=current_loss)
                
#                 # scheduler.step(current_loss)

#                 if current_loss < best_loss:
#                     best_loss = current_loss
#                     best_state = model.state_dict()
#                     counter = 0
#                 else:
#                     counter += 1
#                     if counter >= patience:
#                         model.load_state_dict(best_state)
#                         pbar.update(num_iterations - step - 1)
#                         break

#             pbar.update(1)

#     return model







def train_dgp_minibatch(
    train_x,
    train_y,
    hidden_dim = 4,
    inducing_num = 512,
    num_iterations = 3000,
    patience = 100,
    batch_size = 256,
    eval_every = 200,
    eval_batch_size = 1024,
    lr = 0.05,
    device = "cuda"
):
    train_x, train_y = train_x.to(device), train_y.to(device)

    model = GP_models.DeepGP2(
        train_x, train_y, hidden_dim, inducing_num
    ).to(device)

    model.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(
            likelihood=model.likelihood, model=model, num_data=train_y.size(0)
        )
    )


    best_loss = float("inf")
    best_state = model.state_dict()
    no_improve = 0

    loader = itertools.cycle(
        DataLoader(TensorDataset(train_x, train_y), batch_size, shuffle=True)
    )

    # --- jitter ---
    jitter_ctx = gpytorch.settings.variational_cholesky_jitter(1e-3)

    with tqdm.tqdm(total=num_iterations, desc="Training DGP") as pbar, jitter_ctx:
        for step in range(num_iterations):
            x_batch, y_batch = next(loader)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            if (step + 1) % eval_every == 0:
                model.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    total_loss = 0.0
                    for i in range(0, train_x.size(0), eval_batch_size):
                        xb, yb = (
                            train_x[i : i + eval_batch_size],
                            train_y[i : i + eval_batch_size],
                        )
                        out = model(xb)
                        total_loss += -mll(out, yb).item() * yb.size(0)
                full_loss = total_loss / train_x.size(0)
                pbar.set_postfix(loss=f"{full_loss:.4f}")
                model.train()

                if full_loss < best_loss - 1e-4:
                    best_loss, best_state, no_improve = full_loss, model.state_dict(), 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("Early stopping")
                        break
            pbar.update(1)

    model.load_state_dict(best_state)
    model.eval()
    return model






def train_BNN_minibatch(
    NN_model,
    full_train_x,
    full_train_y,
    num_iterations=50000,
    batch_size=256,
    device='cuda',
    show_progress=True,
    weight_decay=0.2,
    val_x=None,
    val_y=None,
    early_stopping=False,
    patience=1000,
    val_check_interval=100
):

    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)
    if early_stopping and (val_x is not None and val_y is not None):
        val_x = val_x.to(device)
        val_y = val_y.to(device)
    else:
        early_stopping = False

    dataset = TensorDataset(full_train_x, full_train_y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = NN_model(full_train_x, full_train_y).to(device)
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.2,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0


    step = 0
    pbar = tqdm.tqdm(total=num_iterations, disable=not show_progress, desc="training")
    while step < num_iterations:
        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            loss = -out.log_prob(batch_y).mean()

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            step += 1
            pbar.update(1)

            if early_stopping and (step % val_check_interval == 0):
                model.eval()
                with torch.no_grad():
                    val_out  = model(val_x)
                    val_loss = -val_out.log_prob(val_y).mean()
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = model.state_dict()
                    no_improve    = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    if show_progress:
                        pbar.write(
                            f"Early stopping at step {step}, best val loss: {best_val_loss:.4f}"
                        )

                    model.load_state_dict(best_state)
                    pbar.close()
                    return model

            if step >= num_iterations:
                break

    pbar.close()
    return model