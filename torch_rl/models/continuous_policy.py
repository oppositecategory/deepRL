import torch
from torch import nn
import numpy as np
from torch.distributions.normal import Normal

class GaussianPolicy(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=128,log_std=0,**kwargs):
        # Assume the network outputs both mean and log_std for each action.
        # For environment with N actions the output would be Nx2 and for batched: BxNx2.
        super().__init__()
        self.__dict__.update(kwargs)
        self.d = output_dim
        self.layer1 = nn.Linear(input_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,hidden_dim)  

        self.mean = nn.Linear(hidden_dim,output_dim)

        self.log_std = nn.Parameter(torch.ones(1,output_dim)*log_std)


    def forward(self,x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()
        return mean, log_std, std
    
    def sample_action(self, x):
        mean, log_std,std = self.forward(x)
        action = Normal(mean,std).sample()
        return action.squeeze(0).detach().numpy()

    def log_probs(self, x,actions):
        mean, log_std,std = self.forward(x)
        log_probs = Normal(mean,std).log_prob(actions)
        return log_probs.sum(1,keepdim=True)
    
    def kl_divergence(self,x):
        mean, log_std, std = self.forward(x)
        mean0, log_std0 , std0 = mean.data, log_std.data, std.data
        kl = log_std - log_std0 + (mean0-mean).pow(2)/ (2*std.pow(2))
        return kl.sum(1,keepdim=True)

