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








#############################################################################
## Train Multitask GP Model
#############################################################################



def train_one_row_MultitaskGP(local_train_x, local_train_y, n_tasks, covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=10, device='cpu'):

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


def train_DNN_MSE(NN_model, full_train_x, full_train_y, num_iterations = 50000, device='cuda'):

    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)

    model = NN_model(full_train_x, full_train_y)

    model = model.to(device)

    model.train()
 

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 100)
    

    # iterator = tqdm.tqdm(range(num_iterations))


    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(full_train_x)
        loss = criterion(output, full_train_y)
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()
        scheduler.step(loss)

        # if loss <= best_loss:
        #     best_loss = loss
        #     best_state = model.state_dict()  
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         model.load_state_dict(best_state)  
        #         break

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


def train_full_DGP_2(full_train_x, full_train_y, num_hidden_dgp_dims = 4, inducing_num = 500, num_iterations = 2000, patiences = 50, device='cuda'):

    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)

    model = GP_models.DeepGP_2(full_train_x.shape, full_train_y, num_hidden_dgp_dims, inducing_num)

    model = model.to(device)

    model.train()
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.DeepApproximateMLL(gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=full_train_y.size(0)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 25)
    
    best_loss = float('inf')
    counter = 0

    iterator = tqdm.tqdm(range(num_iterations))
    for i in iterator:
    # for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(full_train_x)
        loss = -mll(output, full_train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        scheduler.step(loss)

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()  
            counter = 0
        else:
            counter += 1
            if counter >= patiences:
                model.load_state_dict(best_state)  
                break

    return model


def train_full_DGP_3(full_train_x, full_train_y, num_hidden_dgp_dims = [4,4], inducing_num = 500, num_iterations = 2000, patiences = 50, device='cuda'):

    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)

    model = GP_models.DeepGP_3(full_train_x.shape, full_train_y, num_hidden_dgp_dims, inducing_num)

    model = model.to(device)

    model.train()
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.DeepApproximateMLL(gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=full_train_y.size(0)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 10)
    
    best_loss = float('inf')
    counter = 0


    # iterator = tqdm.tqdm(range(num_iterations))
    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(full_train_x)
        loss = -mll(output, full_train_y)
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()
        scheduler.step(loss)

        if loss <= best_loss:
            best_loss = loss
            best_state = model.state_dict()  
            counter = 0
        else:
            counter += 1
            if counter >= patiences:
                model.load_state_dict(best_state)  
                break

    return model

