'''
Trust Region Policy Optimization

Surrogate loss function + KL Divergence constraint

Need to understand :
         How to optimize linear objective function with constraint(quadratic)
         What is conjugate gradient algorithm
         What is Fisher Information Martix(FIM)
         ...
'''
import sys,os
sys.path.append(os.pardir)
from common.function import mlp

import torch
import torch.nn as nn
from torch.distribution import Normal



#Actor : Stocastic Policy: observation ->  mean and std
class Actor(nn.Module):
    def __init__(self, obs_size, act_size, args):
        super().__init__()
        self.pi_size = [obs_size] + args.hidden_sizes + [act_size]
        self.pi_mlp = mlp(self.pi_size, nn.Tanh)

    def forward(self, obs):
        mu = self.pi_mlp(obs)
        std = torch.exp(torch.zeros_like(mu)) #std:1.0 for all action
        return mu, std

#Critic : Q Function: observation, action -> output
#target Q: r+d*gamma*Q
class Critic(nn.Moudle):
    def __init__(self, obs_size, act_size, args):
        super().__init__()
        self.q_size = [obs_size + act_size] + args.hidden_sizes + [1]
        self.q_mlp = mlp(self.q_size, nn.Tanh)
    def forward(self, obs, act):
        input = torch.cat((obs, act), dim=-1)
        return self.q_mlp(input)
