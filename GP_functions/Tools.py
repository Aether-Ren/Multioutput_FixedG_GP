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

def Initialize_inducing_points_kmeans_x(train_x, num_inducing_pts=500):
    
    train_n = train_x.shape[0]

    inducing_points = train_x[torch.randperm(min(1000 * 100, train_n))[0:num_inducing_pts], :]
    inducing_points = inducing_points.clone().data.cpu().numpy()
    
    inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),
                                           inducing_points, minit='matrix')[0])

    if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()
    
    return inducing_points



def Initialize_inducing_points_kmeans_y(train_x, train_y, num_inducing_pts=500):

    _, label = kmeans2(train_y.data.cpu().numpy(), num_inducing_pts, minit='random')


    inducing_points = []
    for i in range(num_inducing_pts):

        indices = torch.where(torch.tensor(label) == i)[0]
        if len(indices) > 0:
            selected_index = indices[torch.randint(0, len(indices), (1,))]
            inducing_points.append(train_x[selected_index].clone().data.cpu().numpy())


    inducing_points = torch.tensor(inducing_points)

    if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()
    
    return inducing_points.squeeze()

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

def generate_MVN_inde_datasets(dimension_x, dimension_y, num_train_locations, num_test_locations, seed_train, seed_test, variance):
    def kernel(X1, X2, length_scales, variance):
        length_scales = np.asarray(length_scales)
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        sqdist = np.sum(X1_scaled**2, 1).reshape(-1, 1) + np.sum(X2_scaled**2, 1) - 2 * np.dot(X1_scaled, X2_scaled.T)
        return variance * np.exp(-0.5 * sqdist)

    # Generate Sobol sequences
    sobol_train_gen = qmc.Sobol(d=dimension_x, seed=seed_train)
    sobol_test_gen = qmc.Sobol(d=dimension_x, seed=seed_test)
    X_train = sobol_train_gen.random_base2(m=num_train_locations)
    X_test = sobol_test_gen.random_base2(m=num_test_locations)

    # Scale design locations
    l_bounds, u_bounds = [0.1] * dimension_x, [5] * dimension_x
    X_train = qmc.scale(X_train, l_bounds, u_bounds)
    X_test = qmc.scale(X_test, l_bounds, u_bounds)

    Y_train_all = np.zeros((2**num_train_locations, dimension_y))
    Y_test_all = np.zeros((2**num_test_locations, dimension_y))

    for i in range(dimension_y):
        rng = np.random.default_rng(seed=i)
        length_scales_ard = rng.random(dimension_x) * 10 + 5
        # Calculate covariance matrices
        K_train = kernel(X_train, X_train, length_scales_ard, variance)
        K_test = kernel(X_test, X_test, length_scales_ard, variance)

        # Define mean vectors and generate datasets
        random_mean = round(np.random.uniform(-10, 10), 2)
        mean_vector_train = np.zeros(2**num_train_locations) + random_mean
        mean_vector_test = np.zeros(2**num_test_locations) + random_mean

        Y_train = multivariate_normal.rvs(mean=mean_vector_train, cov=K_train)
        Y_test = multivariate_normal.rvs(mean=mean_vector_test, cov=K_test)
        Y_train_all[:, i] = Y_train
        Y_test_all[:, i] = Y_test


    tmp_mean_train = np.mean(Y_train_all, axis=0)
    tmp_std_train = np.std(Y_train_all, axis=0)
    Y_train_all_standardized = (Y_train_all - tmp_mean_train) / tmp_std_train

    tmp_mean_test = np.mean(Y_test_all, axis=0)
    tmp_std_test = np.std(Y_test_all, axis=0)
    Y_test_all_standardized = (Y_test_all - tmp_mean_test) / tmp_std_test
    
    return X_train, X_test, Y_train_all_standardized, Y_test_all_standardized



#############################################################################
## 
#############################################################################

def generate_MVN_datasets(dimension_x, dimension_y, inner_dim, num_train_locations, num_test_locations, seed_train, seed_test, variance):
    
    def kernel(X1, X2, length_scales, variance):
        length_scales = np.asarray(length_scales)
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        sqdist = np.sum(X1_scaled**2, 1).reshape(-1, 1) + np.sum(X2_scaled**2, 1) - 2 * np.dot(X1_scaled, X2_scaled.T)
        return variance * np.exp(-0.5 * sqdist)

    # Generate Sobol sequences
    sobol_train_gen = qmc.Sobol(d=dimension_x, seed=seed_train)
    sobol_test_gen = qmc.Sobol(d=dimension_x, seed=seed_test)
    X_train = sobol_train_gen.random_base2(m=num_train_locations)
    X_test = sobol_test_gen.random_base2(m=num_test_locations)


    X_all = np.concatenate((X_train, X_test), axis=0)

    # Scale design locations
    l_bounds, u_bounds = [0.1] * dimension_x, [5] * dimension_x
    X_all = qmc.scale(X_all, l_bounds, u_bounds)

    Y_inner_all = np.zeros(((2**num_train_locations + 2**num_test_locations), inner_dim))

    for i in range(inner_dim):
        rng = np.random.default_rng(seed=i)
        length_scales_ard = rng.random(dimension_x) * 10 + 5
        # Calculate covariance matrices
        K_all = kernel(X_all, X_all, length_scales_ard, variance)


        # Define mean vectors and generate datasets
        random_mean = round(np.random.uniform(-10, 10), 2)
        # random_mean = round(np.random.uniform(0, 20), 4)
        mean_vector_all = np.zeros((2**num_train_locations + 2**num_test_locations)) + random_mean

        Y_tmp = multivariate_normal.rvs(mean=mean_vector_all, cov=K_all)

        Y_inner_all[:, i] = Y_tmp


    rng_y = np.random.default_rng(seed=999)

    # matrix = rng_y.uniform(low=-10, high=10, size=(inner_dim, dimension_y))
    matrix = rng_y.uniform(low=0, high=10, size=(inner_dim, dimension_y))

    Y_all = Y_inner_all @ matrix


    tmp_mean_train = np.mean(Y_all, axis=0)
    tmp_std_train = np.std(Y_all, axis=0)
    Y_train_all_standardized = (Y_all - tmp_mean_train) / tmp_std_train


    X_train = X_all[:(2**num_train_locations), :]
    X_test = X_all[(2**num_train_locations + 5): -5, :]

    Y_train = Y_train_all_standardized[:(2**num_train_locations), :]
    Y_test = Y_train_all_standardized[(2**num_train_locations + 5): -5, :]
    
    return X_train, X_test, Y_train, Y_test

# , Y_inner_all, matrix, Y_all






# def generate_MVN_datasets(dimension_x, dimension_y, num_train_locations, seed, variance, l_bound, u_bound):
#     """
#     Generate a dataset with spatial correlation and correlation between Y dimensions.
#     """

#     def kernel(X1, X2, length_scales, variance):
#         """
#         Computes a covariance matrix based on a Gaussian kernel.
#         """
#         sqdist = cdist(X1 / length_scales, X2 / length_scales, 'sqeuclidean')
#         return variance * np.exp(-0.5 * sqdist)

#     def generate_total_covariance_matrix(X_train, length_scales_ards, variance, dimension_y, Y_correlation_matrix):
#         """
#         Construct an overall covariance matrix that takes into account both the spatial correlation of X and the correlation between Y dimensions.
#         """
#         num_points = X_train.shape[0]
#         total_covariance_matrix = np.zeros((num_points * dimension_y, num_points * dimension_y))

#         for i in range(dimension_y):
#             for j in range(dimension_y):
#                 if i == j:
#                     K = kernel(X_train, X_train, length_scales_ards[i], variance)
#                 else:
#                     # Adjust the covariance matrix using the correlation between Y dimensions
#                     avg_length_scales = (length_scales_ards[i] + length_scales_ards[j]) / 2
#                     K = kernel(X_train, X_train, avg_length_scales, variance) * Y_correlation_matrix[i, j]
#                 total_covariance_matrix[i*num_points:(i+1)*num_points, j*num_points:(j+1)*num_points] = K
        
#         return total_covariance_matrix


#     def generate_correlation_matrix(dimension_y):
#         np.random.seed(9999)
        
#         random_matrix = np.random.uniform(0, 1, size=(dimension_y, dimension_y))
#         sym_matrix = (random_matrix + random_matrix.T) / 2
        
#         np.fill_diagonal(sym_matrix, 1)
        
#         return sym_matrix


#     sobol_train_gen = qmc.Sobol(d=dimension_x, scramble=False, seed=seed)
#     X_train = sobol_train_gen.random_base2(m=num_train_locations)
#     l_bounds, u_bounds = [l_bound] * dimension_x, [u_bound] * dimension_x
#     X_train = qmc.scale(X_train, l_bounds, u_bounds)
    
#     # Generate independent length scales for each Y dimension
#     rng_1 = np.random.default_rng(1)
#     length_scales_ards = [rng_1.random(dimension_x) * 10 + 5 for _ in range(dimension_y)]
    
#     # Define the correlation matrix between Y dimensions
#     # Y_correlation_matrix = np.full((dimension_y, dimension_y), 0.5)
#     # np.fill_diagonal(Y_correlation_matrix, 1)  
#     Y_correlation_matrix = generate_correlation_matrix(dimension_y)
    
#     # Construct the overall covariance matrix
#     total_covariance_matrix = generate_total_covariance_matrix(X_train, length_scales_ards, variance, dimension_y, Y_correlation_matrix)
    
#     # Define the overall mean vector
#     rng_2 = np.random.default_rng(999)
#     total_mean_vector = np.concatenate([np.full(X_train.shape[0], rng_2.uniform(-10, 10)) for _ in range(dimension_y)])
    
#     # Sample from the overall covariance matrix
#     total_samples = multivariate_normal.rvs(mean=total_mean_vector, cov=total_covariance_matrix, size=1)
    
#     # Reshape sampled data into required format
#     Y_train_all = total_samples.reshape(-1, dimension_y)
    
#     tmp_mean_train = np.mean(Y_train_all, axis=0)
#     tmp_std_train = np.std(Y_train_all, axis=0)
#     Y_train_all_standardized = (Y_train_all - tmp_mean_train) / tmp_std_train


#     return X_train, Y_train_all_standardized




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