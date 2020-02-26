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
from torch.distributions import Normal

#Actor : Stocastic Policy: observation ->  mean and std
class Actor(nn.Module):
    def __init__(self, obs_size, act_size, args):
        super().__init__()
        self.pi_size = [obs_size] + args.hidden_sizes + [act_size]
        self.pi_mlp = mlp(self.pi_size, nn.Tanh).float()

    def forward(self, obs):
        mu = self.pi_mlp(obs) #[-1,1]
        std = torch.exp(torch.zeros_like(mu)) #std:1.0 for all action
        return mu, std

    def get_action(self, mu, std):
        normal_dis = Normal(mu, std)
        action = normal_dis.sample()
        action = action.cpu()
        return action.data.numpy()

    def get_log_prob(self, actions, mu, std):
        #print(actions)
        normal_dis = Normal(mu, std)
        log_prob = normal_dis.log_prob(actions)
        return log_prob

#Critic : Q Function: observation, action -> output
#target Q: r+d*gamma*Q
class Critic(nn.Module):
    def __init__(self, obs_size, act_size, args):
        super().__init__()
        self.v_size = [obs_size] + args.hidden_sizes + [1]
        self.v_mlp = mlp(self.v_size, nn.Tanh)
    def forward(self, obs):
        return self.v_mlp(obs)
