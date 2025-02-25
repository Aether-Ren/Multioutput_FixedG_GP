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