"""
File: Tools.py
Author: Hongjin Ren
Description: Some tools which can help analyze some data, I wish.

"""

#############################################################################
## Package imports
#############################################################################

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.spatial import distance
from scipy.stats import qmc, multivariate_normal
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.graphics.tsaplots import plot_acf

# Pyro diagnostics
from pyro.ops.stats import gelman_rubin, split_gelman_rubin, effective_sample_size



#############################################################################
## 
#############################################################################


def Print_percentiles(mse_array):
    """
    Prints the 1st, 2nd, and 3rd quantiles of the given data.
    """
    return {
        '25th Perc.': np.percentile(mse_array, 25),
        'Median': np.percentile(mse_array, 50),
        '75th Perc.': np.percentile(mse_array, 75)
    }

#############################################################################
## 
#############################################################################

def select_inducing_points_with_pca(train_x, train_y, num_inducing, num_latents):

    pca = PCA(n_components=num_latents)
    train_y_pca = pca.fit_transform(train_y.cpu().detach().numpy())  # shape: (n_samples, num_latents)
    

    train_y_pca_tensor = torch.tensor(train_y_pca, dtype=train_x.dtype, device=train_x.device)
    

    combined_features = torch.cat([train_x, train_y_pca_tensor], dim=-1)  # (n_samples, input_dim + num_latents)
    combined_features_np = combined_features.cpu().detach().numpy()
    
    inducing_points_list = []
    

    for latent in range(num_latents):
        kmeans = KMeans(n_clusters=num_inducing, random_state=latent).fit(combined_features_np)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=train_x.dtype, device=train_x.device)
        inducing_points = centers[:, :train_x.size(-1)]
        inducing_points_list.append(inducing_points)
    
    inducing_points_all = torch.stack(inducing_points_list, dim=0)
    return inducing_points_all

#############################################################################
## Set up two function suit for different device
#############################################################################

def find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k):
    distances = [distance.euclidean(input_point, train_pt) for train_pt in train_y]
    nearest_neighbors = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    return train_x[nearest_neighbors], train_y[nearest_neighbors]

def find_k_nearest_neighbors_GPU(input_point, train_x, train_y, k):

    input_point = input_point.view(1, -1).expand_as(train_y)
    distances = torch.norm(input_point - train_y, dim=1)
    _, nearest_neighbor_idxs = torch.topk(distances, k, largest=False, sorted=True)
    nearest_train_x = train_x[nearest_neighbor_idxs]
    nearest_train_y = train_y[nearest_neighbor_idxs]
    
    return nearest_train_x, nearest_train_y




#############################################################################
## Set up two function suit for different device
#############################################################################


def find_k_nearest_neighbors_Mahalanobis(input_point, train_x, train_y, k):

    cov_matrix = np.cov(train_y, rowvar=False)
    inv_cov_matrix = inv(cov_matrix)
    
    def mahalanobis_dist(x, y):
        return mahalanobis(x, y, inv_cov_matrix)
    
    distances = [mahalanobis_dist(input_point, train_pt) for train_pt in train_y]
    
    nearest_neighbors_idx = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    
    return train_x[nearest_neighbors_idx], train_y[nearest_neighbors_idx]

#############################################################################
## 
#############################################################################

def select_subsequence(original_points, target_num_points):

    # Calculate the step to select points to approximately get the target number of points
    total_points = len(original_points)
    step = max(1, total_points // target_num_points)
    
    # Select points by stepping through the original sequence
    selected_points = original_points[::step]
    
    # Ensure we have exactly target_num_points by adjusting the selection if necessary
    if len(selected_points) > target_num_points:
        # If we selected too many points, trim the excess
        selected_points = selected_points[:target_num_points]
    elif len(selected_points) < target_num_points:
        # If we selected too few points, this indicates a rounding issue with step; handle as needed
        # This is a simple handling method and might need refinement based on specific requirements
        additional_indices = np.random.choice(range(total_points), size=target_num_points - len(selected_points), replace=False)
        additional_points = original_points[additional_indices]
        selected_points = np.vstack((selected_points, additional_points))
    
    return selected_points 








#############################################################################
## 
#############################################################################

def train_test_split_uniform(x, y, test_ratio):
    total_samples = x.shape[0]
    test_size = int(total_samples * test_ratio)
    
    test_indices = np.linspace(0, total_samples - 1, test_size, dtype=int)
    
    x_test = x[test_indices]
    y_test = y[test_indices]
    x_train = np.delete(x, test_indices, axis=0)
    y_train = np.delete(y, test_indices, axis=0)
    
    return x_train, x_test, y_train, y_test


#############################################################################
## 
#############################################################################

def GPU_to_CPU(Models, Likelihoods):
    for column_idx in range(len(Models)):
        Models[column_idx] = Models[column_idx].cpu()
        Likelihoods[column_idx] = Likelihoods[column_idx].cpu()
    return Models, Likelihoods


#############################################################################
## Save and Load
#############################################################################

def save_models_likelihoods(Models, Likelihoods, file_path):
    state_dicts = {
        'models': [model.state_dict() for model in Models],
        'likelihoods': [likelihood.state_dict() for likelihood in Likelihoods]
    }
    torch.save(state_dicts, file_path)


def load_models_likelihoods(file_path, model_class, likelihood_class, train_x, inducing_points, covar_type='RBF', device='cpu'):
    state_dicts = torch.load(file_path)
    
    Models = []
    Likelihoods = []
    for model_state, likelihood_state in zip(state_dicts['models'], state_dicts['likelihoods']):
        model = model_class(train_x, inducing_points=inducing_points, covar_type=covar_type)
        model.load_state_dict(model_state)
        model = model.to(device)
        
        likelihood = likelihood_class()
        likelihood.load_state_dict(likelihood_state)
        likelihood = likelihood.to(device)
        
        Models.append(model)
        Likelihoods.append(likelihood)
    
    return Models, Likelihoods

#################
##
################

def get_outlier_indices_iqr(data, outbound = 1.5):
    mask = np.ones(data.shape[0], dtype=bool)
    
    for i in range(data.shape[1]):
        Q1 = np.percentile(data[:, i], 25)
        Q3 = np.percentile(data[:, i], 75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - outbound * IQR
        upper_bound = Q3 + outbound * IQR
        
        mask = mask & (data[:, i] >= lower_bound) & (data[:, i] <= upper_bound)
    
    outlier_indices = np.where(~mask)[0]  
    return outlier_indices

#################
##
################

def extract_vector_params_from_mcmc(samples, *, key="params", param_names=None):
    """
    从 Pyro MCMC 对象中提取向量参数站点 key='params'，并拆成 dict[name -> Tensor[N]]。
    """

    theta = samples[key]  # Tensor[N, D] or Tensor[N] if D=1
    if theta.ndim == 1:
        theta = theta.unsqueeze(-1)

    N, D = theta.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(D)]
    else:
        if len(param_names) != D:
            raise ValueError(f"param_names length {len(param_names)} != D {D}")

    out = {name: theta[:, i].detach() for i, name in enumerate(param_names)}
    return out



def extract_exp_params_from_mcmc(samples, *, param_names=None, key="log_params"):
    """
    从 Pyro MCMC 对象中提取 samples，并对 log_params 做 exp 变成真实参数。
    返回 dict: name -> Tensor[N]
    """

    log_theta = samples[key]  # Tensor[N, D] or Tensor[N] if D=1
    if log_theta.ndim == 1:
        log_theta = log_theta.unsqueeze(-1)  # [N,1]

    theta = torch.exp(log_theta)  # [N, D]
    N, D = theta.shape

    if param_names is None:
        param_names = [f"param_{i}" for i in range(D)]
    else:
        if len(param_names) != D:
            raise ValueError(f"param_names length {len(param_names)} != D {D}")

    out = {name: theta[:, i].detach() for i, name in enumerate(param_names)}
    return out



def split_chain(chain_tensor: torch.Tensor):
    """
    将单链样本拆成两半，形成两条“伪链”
    输入: Tensor[N]
    输出: (Tensor[N//2], Tensor[N//2])
    """
    n = chain_tensor.shape[0]
    half = n // 2
    return chain_tensor[:half], chain_tensor[half:2*half]


def visualize_posterior_1d_params(
    single_chain_samples: dict,
    *,
    bins=15,
    acf_lags=40,
    clip_percentiles=(0.5, 99.5),
    xlim=None,
):
    """
    single_chain_samples: dict[name -> Tensor[N]]
    对每个参数：split Rhat/ESS + trace + hist+KDE+quantiles + ACF
    """
    # 整理成 mcmc_samples：param -> Tensor[2, n_half]
    mcmc_samples = {}
    for param, samples in single_chain_samples.items():
        if samples.ndim != 1:
            raise ValueError(f"{param} should be 1D Tensor[N], got shape {tuple(samples.shape)}")
        chain_a, chain_b = split_chain(samples)
        mcmc_samples[param] = torch.stack([chain_a, chain_b], dim=0)  # [2, n_half]

    # 诊断和可视化
    for param, samples_chains in mcmc_samples.items():
        rhat = gelman_rubin(samples_chains, chain_dim=0, sample_dim=1)
        split_rhat = split_gelman_rubin(samples_chains, chain_dim=0, sample_dim=1)
        ess = effective_sample_size(samples_chains, chain_dim=0, sample_dim=1)
        print(f"{param}: R-hat = {rhat:.3f}, split R-hat = {split_rhat:.3f}, ESS = {ess:.1f}")

        # --- Trace + Histogram/KDE ---
        plt.figure(figsize=(12, 4))

        # Trace
        plt.subplot(1, 2, 1)
        for i in range(2):
            plt.plot(samples_chains[i].cpu().numpy(), marker='o', label=f"Chain {i+1}", alpha=0.7)
        plt.title(f"Trace Plot for {param}")
        plt.xlabel("Sample Index")
        plt.ylabel(param)
        plt.legend()

        # Histogram + KDE + Quantiles
        plt.subplot(1, 2, 2)
        all_samps = samples_chains.reshape(-1).cpu().numpy()

        p_lo, p_hi = clip_percentiles
        xmin, xmax = np.percentile(all_samps, [p_lo, p_hi])

        plt.hist(all_samps, bins=bins, density=True, alpha=0.7, color='gray')

        # KDE（样本太少/方差太小有时会报错，所以做个保护）
        if np.std(all_samps) > 0 and len(all_samps) > 5:
            kde = gaussian_kde(all_samps)
            x_grid = np.linspace(xmin, xmax, 200)
            plt.plot(x_grid, kde(x_grid), linewidth=2)
        else:
            x_grid = None

        qs = torch.quantile(torch.from_numpy(all_samps), torch.tensor([0.025, 0.5, 0.975]))
        for q in qs:
            plt.axvline(q.item(), color='red', linestyle='--', linewidth=2)

        if xlim is not None:
            plt.xlim(*xlim)

        plt.title(f"Histogram + 2.5/50/97.5% Quantiles")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

        # --- ACF（仅第一“伪链”） ---
        plt.figure(figsize=(6, 4))
        plot_acf(samples_chains[0].cpu().numpy(), lags=acf_lags)
        plt.title(f"ACF for {param} (Chain 1)")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.tight_layout()
        plt.show()
