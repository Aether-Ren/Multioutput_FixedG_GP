"""
File: NN_models.py
Author: Hongjin Ren
Description: Neural Network

"""

#############################################################################
## Package imports
#############################################################################

import torch
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
import pyro
import pyro.distributions as dist

#############################################################################
## NN
#############################################################################
class NN_4(torch.nn.Module):
    def __init__(self, train_x, train_y):
        super(NN_4, self).__init__()

        self.fc1 = torch.nn.Linear(train_x.size(-1), 128) 
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.fc2 = torch.nn.Linear(128, 256) 
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.fc3 = torch.nn.Linear(256, 64)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(64)

        self.fc4 = torch.nn.Linear(64, train_y.size(-1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        return x
    


class NN_5(torch.nn.Module):
    def __init__(self, train_x, train_y):
        super(NN_5, self).__init__()

        self.fc1 = torch.nn.Linear(train_x.size(-1), 128) 
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.fc2 = torch.nn.Linear(128, 256) 
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.fc3 = torch.nn.Linear(256, 256) 
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.fc4 = torch.nn.Linear(256, 64)
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm1d(64)

        self.fc5 = torch.nn.Linear(64, train_y.size(-1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.fc5(x)
        return x
    



class NN_5_1(torch.nn.Module):
    def __init__(self, train_x, train_y):
        super(NN_5_1, self).__init__()

        self.fc1 = torch.nn.Linear(train_x.size(-1), 128)
        self.relu1 = torch.nn.LeakyReLU()
        self.bn1 = torch.nn.LayerNorm(128)
        self.dropout1 = torch.nn.Dropout(p=0.2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.relu2 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.LayerNorm(256)
        self.dropout2 = torch.nn.Dropout(p=0.2)

        self.fc3 = torch.nn.Linear(256, 256)
        self.relu3 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.LayerNorm(256)
        self.dropout3 = torch.nn.Dropout(p=0.2)

        self.fc4 = torch.nn.Linear(256, 64)
        self.relu4 = torch.nn.LeakyReLU()
        self.bn4 = torch.nn.LayerNorm(64)
        self.dropout4 = torch.nn.Dropout(p=0.2)

        self.fc_mu = torch.nn.Linear(64, train_y.size(-1))


    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu1(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu2(self.fc2(x))))
        x = self.dropout3(self.bn3(self.relu3(self.fc3(x))))
        x = self.dropout4(self.bn4(self.relu4(self.fc4(x))))

        x = self.fc_mu(x)

        # x = torch.sigmoid(x)
        # x = x * (5 - 0.1) + 0.1

        return x
    

#############################################################################
## BNN
#############################################################################


class BNN_2(PyroModule):
    def __init__(self, train_x, train_y):
        super(BNN_2, self).__init__()
        in_dim  = train_x.size(-1)
        out_dim = train_y.size(-1)
        device  = train_x.device

        self.fc1 = PyroModule[nn.Linear](in_dim, 200)
        self.fc1.weight = PyroSample(
            dist.Normal(
                torch.zeros(200, in_dim, device=device),
                torch.ones(200, in_dim, device=device) * 0.1
            ).to_event(2)
        )
        self.fc1.bias   = PyroSample(
            dist.Normal(
                torch.zeros(200, device=device),
                torch.ones(200, device=device) * 0.1
            ).to_event(1)
        )
        self.relu1 = nn.ReLU()
        self.bn1   = nn.BatchNorm1d(200).to(device)

        self.fc2 = PyroModule[nn.Linear](200, 200)
        self.fc2.weight = PyroSample(
            dist.Normal(
                torch.zeros(200, 200, device=device),
                torch.ones(200, 200, device=device) * 0.1
            ).to_event(2)
        )
        self.fc2.bias   = PyroSample(
            dist.Normal(
                torch.zeros(200, device=device),
                torch.ones(200, device=device) * 0.1
            ).to_event(1)
        )
        self.relu2 = nn.ReLU()
        self.bn2   = nn.BatchNorm1d(200).to(device)

        self.fc3 = PyroModule[nn.Linear](200, out_dim)
        self.fc3.weight = PyroSample(
            dist.Normal(
                torch.zeros(out_dim, 200, device=device),
                torch.ones(out_dim, 200, device=device) * 0.1
            ).to_event(2)
        )
        self.fc3.bias = PyroSample(
            dist.Normal(
                torch.zeros(out_dim, device=device),
                torch.ones(out_dim, device=device) * 0.1
            ).to_event(1)
        )

        self.sigma = PyroSample(dist.HalfCauchy(torch.tensor(1.0, device=device)))

    def forward(self, x, y=None):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        mean = self.fc3(x)

        pred_dist = dist.Normal(mean, self.sigma).to_event(1)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", pred_dist, obs=y)

        return mean if y is not None else pred_dist
    






class BNN_Deep(PyroModule):
    def __init__(self, train_x, train_y):
        super(BNN_Deep, self).__init__()

        in_dim  = train_x.size(-1)
        out_dim = train_y.size(-1)
        device  = train_x.device

        self.fc1 = PyroModule[nn.Linear](in_dim, 200)
        self.fc1.weight = PyroSample(
            dist.Normal(
                torch.zeros(200, in_dim, device=device),
                torch.ones(200, in_dim, device=device) * 0.1
            ).to_event(2))
        self.fc1.bias = PyroSample(
            dist.Normal(
                torch.zeros(200, device=device),
                torch.ones(200, device=device) * 0.1
            ).to_event(1))
        self.bn1 = nn.BatchNorm1d(200).to(device)

        self.fc2 = PyroModule[nn.Linear](200, 100)
        self.fc2.weight = PyroSample(dist.Normal(0., 0.1).expand([100, 200]).to_event(2))
        self.fc2.bias   = PyroSample(dist.Normal(0., 0.1).expand([100]).to_event(1))
        self.bn2 = nn.BatchNorm1d(100).to(device)

        self.fc3 = PyroModule[nn.Linear](100, 50)
        self.fc3.weight = PyroSample(dist.Normal(0., 0.1).expand([50, 100]).to_event(2))
        self.fc3.bias   = PyroSample(dist.Normal(0., 0.1).expand([50]).to_event(1))
        self.bn3 = nn.BatchNorm1d(50).to(device)

        self.fc4 = PyroModule[nn.Linear](50, out_dim)
        self.fc4.weight = PyroSample(dist.Normal(0., 0.1).expand([out_dim, 50]).to_event(2))
        self.fc4.bias   = PyroSample(dist.Normal(0., 0.1).expand([out_dim]).to_event(1))

        self.sigma = PyroSample(dist.HalfCauchy(1.0))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        mean = self.fc4(x)
        dist_pred = dist.Normal(mean, self.sigma).to_event(1)
        with pyro.plate("data", x.size(0)):
            pyro.sample("obs", dist_pred, obs=y)
        return mean if y is not None else dist_pred


class BNN_WideDrop(PyroModule):
    def __init__(self, train_x, train_y):
        super(BNN_WideDrop, self).__init__()

        in_dim  = train_x.size(-1)
        out_dim = train_y.size(-1)
        device  = train_x.device
        self.fc1 = PyroModule[nn.Linear](in_dim, 400)
        self.fc1.weight = PyroSample(dist.Normal(0., 0.5).expand([400, in_dim]).to_event(2))
        self.fc1.bias   = PyroSample(dist.Normal(0., 0.5).expand([400]).to_event(1))
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = PyroModule[nn.Linear](400, 400)
        self.fc2.weight = PyroSample(dist.Normal(0., 0.5).expand([400, 400]).to_event(2))
        self.fc2.bias   = PyroSample(dist.Normal(0., 0.5).expand([400]).to_event(1))
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = PyroModule[nn.Linear](400, out_dim)
        self.fc3.weight = PyroSample(dist.Normal(0., 0.5).expand([out_dim, 400]).to_event(2))
        self.fc3.bias   = PyroSample(dist.Normal(0., 0.5).expand([out_dim]).to_event(1))
        self.sigma = PyroSample(dist.HalfCauchy(1.0))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        mean = self.fc3(x)
        dist_pred = dist.Normal(mean, self.sigma).to_event(1)
        with pyro.plate("data", x.size(0)):
            pyro.sample("obs", dist_pred, obs=y)
        return mean if y is not None else dist_pred
    

class BNN_ARD(PyroModule):
    def __init__(self, train_x, train_y):
        super(BNN_ARD, self).__init__()

        in_dim  = train_x.size(-1)
        out_dim = train_y.size(-1)
        device  = train_x.device

        self.tau1 = PyroSample(dist.HalfCauchy(1.0))
        self.tau2 = PyroSample(dist.HalfCauchy(1.0))
        self.fc1 = PyroModule[nn.Linear](in_dim, 200)

        self.fc1.weight = PyroSample(
            lambda tau1=self.tau1: dist.Normal(0., tau1).expand([200, in_dim]).to_event(2)
        )
        self.fc1.bias   = PyroSample(
            lambda tau1=self.tau1: dist.Normal(0., tau1).expand([200]).to_event(1)
        )
        self.fc2 = PyroModule[nn.Linear](200, 200)
        self.fc2.weight = PyroSample(
            lambda tau2=self.tau2: dist.Normal(0., tau2).expand([200, 200]).to_event(2)
        )
        self.fc2.bias   = PyroSample(
            lambda tau2=self.tau2: dist.Normal(0., tau2).expand([200]).to_event(1)
        )
        self.fc3 = PyroModule[nn.Linear](200, out_dim)
        self.fc3.weight = PyroSample(dist.Normal(0., 0.1).expand([out_dim, 200]).to_event(2))
        self.fc3.bias   = PyroSample(dist.Normal(0., 0.1).expand([out_dim]).to_event(1))
        self.sigma = PyroSample(dist.HalfCauchy(1.0))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.fc3(x)
        dist_pred = dist.Normal(mean, self.sigma).to_event(1)
        with pyro.plate("data", x.size(0)):
            pyro.sample("obs", dist_pred, obs=y)
        return mean if y is not None else dist_pred