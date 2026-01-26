"""
File: Prediction.py
Author: Hongjin Ren
Description: Predict the reslut from Gaussian process models

"""

#############################################################################
## Package imports
#############################################################################
import torch
import gpytorch
import pyro
from pyro.infer import Predictive

#############################################################################
## 
#############################################################################
def preds_distribution(model, likelihood, xxx):
    model.eval()
    likelihood.eval()
    preds = likelihood(model(xxx))
    return preds


def preds_distribution_fast_pred_var(model, likelihood, xxx):
    model.eval()
    likelihood.eval()
    with gpytorch.settings.fast_pred_var():
        preds = likelihood(model(xxx))
    return preds

def preds_for_one_model(model, likelihood, xxx):
    model.eval()
    likelihood.eval()
    with gpytorch.settings.fast_pred_var():
        preds = likelihood(model(xxx)).mean
    # preds = likelihood(model(xxx)).mean
    return preds.view(-1)

def full_preds(models, likelihoods, xxx):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_one_model(models[0], likelihoods[0], xxx).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_one_model(models[i], likelihoods[i],xxx).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.view(-1)


#############################################################################
## 
#############################################################################


def preds_for_VGP(model, likelihood, xxx):
    # Prediction of a column of the local data
    model.eval()
    likelihood.eval()
    # with torch.no_grad():
    preds = model(xxx).mean
    return preds.view(-1)

def full_preds_for_VGP(models, likelihoods, xxx):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_VGP(models[0], likelihoods[0], xxx).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_VGP(models[i], likelihoods[i],xxx).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.view(-1)

#############################################################################
## 
#############################################################################

def preds_for_column_var(model, likelihood, local_train_x):
    # Prediction of a column of the local data
    model.eval()
    likelihood.eval()
    preds_var = model(local_train_x).variance
    return preds_var

def full_preds_var(models, likelihoods, local_train_x):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_column_var(models[0], likelihoods[0], local_train_x).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_column_var(models[i], likelihoods[i],local_train_x).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.squeeze()

#############################################################################
## 
#############################################################################
def preds_for_DNN(model, xxx):
    # Prediction of a column of the local data
    model.eval()  
    # with torch.no_grad():  
    preds = model(xxx)
    return preds




def preds_distribution_for_BNN(model, Likelihood, xxx):
    predictive = Predictive(model, guide=guide, 
                            num_samples=1000)
    return preds




def dgp_predict_cov(dgpmodel,
               x,
               num_mc: int = 10,
               jitter: float = 1e-8,
               device="cuda"):

    x = x.to(device)

    with gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(num_mc):
        post = dgpmodel.likelihood(dgpmodel(x)).to_data_independent_dist()
        # mean : [num_mc, 1, T]
        # cov_matrix : [num_mc, 1, T, T]
    mean_mc = post.mean.squeeze(1)                     # [num_mc, T]
    cov_mc  = post.covariance_matrix.squeeze(1)        # [num_mc, T, T]

    # 2. Moment matching across MC dimension
    mu_bar = mean_mc.mean(dim=0)                       # [T]
    centered = mean_mc - mu_bar                        # [num_mc, T]
    cov_mu   = (centered.unsqueeze(2)                  # [num_mc, T, 1]
                @ centered.unsqueeze(1))               # [num_mc, 1, T] → [num_mc, T, T]

    Sigma_bar = cov_mc.mean(dim=0) + cov_mu.mean(dim=0)  # [T, T]

    Sigma_bar = Sigma_bar + jitter * torch.eye(Sigma_bar.size(0),
                                               device=Sigma_bar.device)

    return pyro.distributions.MultivariateNormal(mu_bar, covariance_matrix=Sigma_bar)