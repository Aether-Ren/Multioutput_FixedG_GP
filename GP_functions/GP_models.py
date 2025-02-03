"""
File: GP_models.py
Author: Hongjin Ren
Description: Various Gaussian process models based on GpyTorch

"""

#############################################################################
## Package imports
#############################################################################

import torch
import gpytorch
import GP_functions.FeatureE as FeatureE


#############################################################################
## Set up the model structure (LocalGP, SparseGP, MultitaskGP)
#############################################################################

## LocalGP

class LocalGP(gpytorch.models.ExactGP):
    # get the training data and likelihoods, and construct any objects needed for the model's forward methods.
    def __init__(self, train_x, train_y, likelihood, covar_type = 'RBF'):
        super(LocalGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if covar_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)))
        elif covar_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1)))
        elif covar_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1)))
        elif covar_type == 'RQ':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=train_x.size(-1)))
        elif covar_type == 'PiecewisePolynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=2, ard_num_dims=train_x.size(-1)))
        else:
            print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')
        

    def forward(self, x):
        # denotes the a priori mean and covariance matrix of GP
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    



# class BatchIndependentLocalGP(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(BatchIndependentLocalGP, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y.shape[1]]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1), batch_shape=torch.Size([train_y.shape[1]])),
#             batch_shape=torch.Size([train_y.shape[1]])
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
#             gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#         )





## SparseGP

# class SparseGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, inducing_points):
#         super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)))
#         self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points, 
#                                                                  likelihood=likelihood)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    






## MultitaskGP

# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, n_tasks):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ConstantMean(), num_tasks=n_tasks
#         )
#         self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))), 
#             num_tasks=n_tasks, rank=1
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks, covar_type = 'RBF'):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_tasks
        )
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)), 
        #     num_tasks=n_tasks, rank=1
        # )

        if covar_type == 'RBF':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=train_x.size(-1)), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=train_x.size(-1)), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'RQ':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RQKernel(ard_num_dims=train_x.size(-1)), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'PiecewisePolynomial':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.PiecewisePolynomialKernel(q=2, ard_num_dims=train_x.size(-1)), 
                num_tasks=n_tasks, rank=1)
        else:
            print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)






class MultitaskGPModel_lcm(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super(MultitaskGPModel_lcm, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_tasks
        )
        self.covar_module = gpytorch.kernels.LCMKernel(
            [
            #  gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3, ard_num_dims=train_x.size(-1)),
             gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=train_x.size(-1)),
             gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=train_x.size(-1)),
             gpytorch.kernels.RQKernel(ard_num_dims=train_x.size(-1))], 
            num_tasks=n_tasks, rank=1
        )

        # if covar_type == 'RBF':
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(
        #         gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)), 
        #         num_tasks=n_tasks, rank=1)
        # elif covar_type == 'Matern5/2':
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(
        #         gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=train_x.size(-1)), 
        #         num_tasks=n_tasks, rank=1)
        # elif covar_type == 'Matern3/2':
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(
        #         gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=train_x.size(-1)), 
        #         num_tasks=n_tasks, rank=1)
        # elif covar_type == 'RQ':
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(
        #         gpytorch.kernels.RQKernel(ard_num_dims=train_x.size(-1)), 
        #         num_tasks=n_tasks, rank=1)
        # elif covar_type == 'PiecewisePolynomial':
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(
        #         gpytorch.kernels.PiecewisePolynomialKernel(q=2, ard_num_dims=train_x.size(-1)), 
        #         num_tasks=n_tasks, rank=1)
        # else:
        #     print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
















#############################################################################
## Set up the model structure (NN + (LocalGP, SparseGP, MultitaskGP))
#############################################################################


## NN + SparseGP

# class NNSparseGP(gpytorch.models.ApproximateGP):

#     def __init__(self, train_x, inducing_points):
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
#         variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
#         super(NNSparseGP, self).__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#                 gpytorch.kernels.RBFKernel(ard_num_dims = 32) # the value of ard_num_dims should change with the value of feature_extractor
#                 )
#         self.feature_extractor = FeatureE.FeatureExtractor(train_x)
#         self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1,1)

#     def forward(self, x):
#         projected_x = self.feature_extractor(x)
#         projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
#         # denotes the a priori mean and covariance matrix of GP
#         mean_x = self.mean_module(projected_x)
#         covar_x = self.covar_module(projected_x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
## NN + LocalGP

# class NNLocalGP(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(NNLocalGP, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 32)) # the value of ard_num_dims should change with the value of feature_extractor

#         self.feature_extractor = FeatureE.FeatureExtractor(train_x)
#         self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1,1)

#     def forward(self, x):
#         projected_x = self.feature_extractor(x)
#         projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
#         # denotes the a priori mean and covariance matrix of GP
#         mean_x = self.mean_module(projected_x)
#         covar_x = self.covar_module(projected_x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NNLocalGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor_class, covar_type = 'RBF'):
        super(NNLocalGP, self).__init__(train_x, train_y, likelihood)

        self.feature_extractor = feature_extractor_class(train_x)
        output_dim = self.feature_extractor[-1].out_features

        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = output_dim))
        if covar_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = output_dim))
        elif covar_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims = output_dim))
        elif covar_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims = output_dim))
        elif covar_type == 'RQ':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims = output_dim))
        elif covar_type == 'PiecewisePolynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=2, ard_num_dims = output_dim))
        else:
            print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1,1)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        # denotes the a priori mean and covariance matrix of GP
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



## NN + MultitaskGP

class NNMultitaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks, feature_extractor_class, covar_type = 'RBF'):
        super(NNMultitaskGP, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor_class(train_x)
        output_dim = self.feature_extractor[-1].out_features
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_tasks
        )
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims = output_dim), # the value of ard_num_dims should change with the value of feature_extractor
        #     num_tasks=n_tasks, rank=1
        # )

        if covar_type == 'RBF':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=output_dim), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=output_dim), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=output_dim), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'RQ':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RQKernel(ard_num_dims=output_dim), 
                num_tasks=n_tasks, rank=1)
        elif covar_type == 'PiecewisePolynomial':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.PiecewisePolynomialKernel(q=2,ard_num_dims=output_dim), 
                num_tasks=n_tasks, rank=1)
        else:
            print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')
        
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1,1)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    





## Using Natural Gradient Descent with Variational Models

class VGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, inducing_points, covar_type = 'RBF'):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VGPModel, self).__init__(variational_strategy)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)))
        if covar_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)))
        elif covar_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1)))
        elif covar_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1)))
        elif covar_type == 'RQ':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=train_x.size(-1)))
        elif covar_type == 'PiecewisePolynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=2,ard_num_dims=train_x.size(-1)))
        else:
            print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class MultitaskVariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, num_latents = 8, num_inducing = 100, covar_type = 'Matern3/2'):

        # inducing_points = torch.rand(num_latents, num_inducing, train_x.shape[1]) * (5 - 0.1) + 0.1
        # inducing_points = train_x[torch.randint(0, train_x.size(0), (num_latents, num_inducing))]
        inducing_points = train_x[:num_inducing].unsqueeze(0).expand(num_latents, -1, -1)

        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=train_y.shape[1],
            num_latents=num_latents,
            latent_dim=-1
        )

        super(MultitaskVariationalGP,self).__init__(variational_strategy)

        # self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]), ard_num_dims=train_x.size(-1)),
        #     batch_shape=torch.Size([num_latents])
        # )
        if covar_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([num_latents]), ard_num_dims=train_x.size(-1)),
                batch_shape=torch.Size([num_latents]))
        elif covar_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]), ard_num_dims=train_x.size(-1)),
                batch_shape=torch.Size([num_latents]))
        elif covar_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([num_latents]), ard_num_dims=train_x.size(-1)),
                batch_shape=torch.Size([num_latents]))
        elif covar_type == 'RQ':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RQKernel(batch_shape=torch.Size([num_latents]), ard_num_dims=train_x.size(-1)),
                batch_shape=torch.Size([num_latents]))
        elif covar_type == 'PiecewisePolynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PiecewisePolynomialKernel(q=2, batch_shape=torch.Size([num_latents]), ard_num_dims=train_x.size(-1)),
                batch_shape=torch.Size([num_latents]))
        else:
            print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)








#############################################################################
## Set up the model structure (DeepGP)
#############################################################################

class DGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing = 500, linear_mean=True):
        # inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        inducing_points = torch.rand(output_dims, num_inducing, input_dims) * (5 - 0.1) + 0.1

        batch_shape = torch.Size([output_dims])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        # self.mean_module = gpytorch.means.ConstantMean() if linear_mean else gpytorch.means.LinearMean(input_dims)
        self.mean_module = gpytorch.means.ZeroMean() if linear_mean else gpytorch.means.LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

    


class DeepGP_2(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, train_x_shape, train_y, num_hidden_dgp_dims = 4, inducing_num = 500):
        num_tasks = train_y.size(-1)

        hidden_layer_1 = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            num_inducing=inducing_num, 
            linear_mean=True
        )


        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer_1.output_dims,
            output_dims = num_tasks,
            num_inducing=inducing_num, 
            linear_mean=False
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        output = self.last_layer(hidden_rep1)
        return output
    
    def predict(self, test_x):
        # with torch.no_grad():
        preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0).squeeze(), preds.variance.mean(0).squeeze()





class DeepGP_3(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, train_x_shape, train_y,num_hidden_dgp_dims = [4,4], inducing_num = 500):
        num_tasks = train_y.size(-1)

        hidden_layer_1 = DGPHiddenLayer(
            input_dims = train_x_shape[-1],
            output_dims = num_hidden_dgp_dims[0],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        hidden_layer_2 = DGPHiddenLayer(
            input_dims = hidden_layer_1.output_dims,
            output_dims = num_hidden_dgp_dims[1],
            num_inducing = inducing_num, 
            linear_mean = True
        )


        last_layer = DGPHiddenLayer(
            input_dims = hidden_layer_2.output_dims,
            output_dims = num_tasks,
            num_inducing = inducing_num, 
            linear_mean = False
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        hidden_rep2 = self.hidden_layer_2(hidden_rep1)
        output = self.last_layer(hidden_rep2)
        return output
    
    def predict(self, test_x):
        # with torch.no_grad():
        preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0).squeeze(), preds.variance.mean(0).squeeze()




class DeepGP_4(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, train_x_shape, train_y,num_hidden_dgp_dims = [4,4,4], inducing_num = 500):
        num_tasks = train_y.size(-1)

        hidden_layer_1 = DGPHiddenLayer(
            input_dims = train_x_shape[-1],
            output_dims = num_hidden_dgp_dims[0],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        hidden_layer_2 = DGPHiddenLayer(
            input_dims = hidden_layer_1.output_dims,
            output_dims = num_hidden_dgp_dims[1],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        hidden_layer_3 = DGPHiddenLayer(
            input_dims = hidden_layer_2.output_dims,
            output_dims = num_hidden_dgp_dims[2],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        last_layer = DGPHiddenLayer(
            input_dims = hidden_layer_3.output_dims,
            output_dims = num_tasks,
            num_inducing = inducing_num, 
            linear_mean = False
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        hidden_rep2 = self.hidden_layer_2(hidden_rep1)
        hidden_rep3 = self.hidden_layer_3(hidden_rep2)
        output = self.last_layer(hidden_rep3)
        return output
    
    def predict(self, test_x):
        # with torch.no_grad():
        preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0).squeeze(), preds.variance.mean(0).squeeze()




class DeepGP_5(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, train_x_shape, train_y,num_hidden_dgp_dims = [4,4,4,4], inducing_num = 500):
        num_tasks = train_y.size(-1)

        hidden_layer_1 = DGPHiddenLayer(
            input_dims = train_x_shape[-1],
            output_dims = num_hidden_dgp_dims[0],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        hidden_layer_2 = DGPHiddenLayer(
            input_dims = hidden_layer_1.output_dims,
            output_dims = num_hidden_dgp_dims[1],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        hidden_layer_3 = DGPHiddenLayer(
            input_dims = hidden_layer_2.output_dims,
            output_dims = num_hidden_dgp_dims[2],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        hidden_layer_4 = DGPHiddenLayer(
            input_dims = hidden_layer_3.output_dims,
            output_dims = num_hidden_dgp_dims[3],
            num_inducing = inducing_num, 
            linear_mean = True
        )

        last_layer = DGPHiddenLayer(
            input_dims = hidden_layer_4.output_dims,
            output_dims = num_tasks,
            num_inducing = inducing_num, 
            linear_mean = False
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.hidden_layer_4 = hidden_layer_4
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        hidden_rep2 = self.hidden_layer_2(hidden_rep1)
        hidden_rep3 = self.hidden_layer_3(hidden_rep2)
        hidden_rep4 = self.hidden_layer_4(hidden_rep3)
        output = self.last_layer(hidden_rep4)
        return output
    
    def predict(self, test_x):
        # with torch.no_grad():
        preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0).squeeze(), preds.variance.mean(0).squeeze()



#############################################################################
## Deep Sigma Point Processes ()
#############################################################################


class DSPPHiddenLayer_Matern(gpytorch.models.deep_gps.dspp.DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)

        # Let's use mean field / diagonal covariance structure.
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference.
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        super(DSPPHiddenLayer_Matern, self).__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = gpytorch.means.LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DSPPHiddenLayer_RBF(gpytorch.models.deep_gps.dspp.DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)

        # Let's use mean field / diagonal covariance structure.
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference.
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        super(DSPPHiddenLayer_Matern, self).__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = gpytorch.means.LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims), 
                                                         batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DSPP_2(gpytorch.models.deep_gps.dspp.DSPP):
    def __init__(self, train_x_shape, train_y, inducing_points, num_inducing, hidden_dim=3, Q=3):
        num_tasks = train_y.size(-1)
        hidden_layer = DSPPHiddenLayer_Matern(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            mean_type='linear',
            inducing_points=inducing_points,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer_Matern(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            mean_type='constant',
            inducing_points=None,
            num_inducing=num_inducing,
            Q=Q,
        )

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        super().__init__(Q)
        self.likelihood = likelihood
        self.last_layer = last_layer
        self.hidden_layer = hidden_layer

    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer(inputs, **kwargs)
        output = self.last_layer(hidden_rep1, **kwargs)
        return output

    def predict(self, loader):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls = [], [], []
            for x_batch, y_batch in loader:
                preds = self.likelihood(self(x_batch, mean_input=x_batch))
                mus.append(preds.mean.cpu())
                variances.append(preds.variance.cpu())

                # Step 1: Get log marginal for each Gaussian in the output mixture.
                base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))

                # Step 2: Weight each log marginal by its quadrature weight in log space.
                deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll

                # Step 3: Take logsumexp over the mixture dimension, getting test log prob for each datapoint in the batch.
                batch_log_prob = deep_batch_ll.logsumexp(dim=0)
                lls.append(batch_log_prob.cpu())

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)




## VNNGP: Variational Nearest Neighbor Gaussian Procceses
# class VNNGPModel(gpytorch.models.ApproximateGP):
#     # There are two hyperparameters: k: number of nearest neighbors used.
#     # training_batch_size: the mini-batch size of inducing points used in stochastic optimization. 
#     def __init__(self, inducing_points, likelihood, k = 256, training_batch_size = 256):

#         m, d = inducing_points.shape
#         self.m = m
#         self.k = k

#         variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)

#         # if torch.cuda.is_available():
#         #     inducing_points = inducing_points.cuda()

#         variational_strategy = gpytorch.variational.nearest_neighbor_variational_strategy.NNVariationalStrategy(self, inducing_points, variational_distribution,
#                                                                                                                  k=k, training_batch_size=training_batch_size)
#         super(VNNGPModel, self).__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))

#         self.likelihood = likelihood

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#     def __call__(self, x, prior=False, **kwargs):
#         if x is not None:
#             if x.dim() == 1:
#                 x = x.unsqueeze(-1)
#         return self.variational_strategy(x=x, prior=False, **kwargs)