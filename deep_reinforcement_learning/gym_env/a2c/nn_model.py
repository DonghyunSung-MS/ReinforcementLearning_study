import sys, os
sys.path.append(os.pardir)
from common.function import mlp
from torch.distributions.categorical import Categorical

import torch
import torch.nn as nn

#torch tensor input tensor output

class Actor(nn.Module):
    def __init__(self, obs_size, act_size, args):
        super().__init__()
        self.pi_size = [obs_size] + args.hidden_sizes + [act_size]
        self.pi = mlp(self.pi_size, nn.Tanh)

    def forward(self, obs):
        # logits(p) -> [-inf,inf] mapping
        return self.pi(obs)

    def get_policy(self, obs):
        logits = self.forward(obs)
        policy = Categorical(logits=logits) #[0,1] mapping
        return policy

    def get_action(self, obs):
        action = self.get_policy(obs).sample().item() #[0,1] sample according to policy
        return action

class Critic(nn.Module):
    def __init__(self, obs_size, act_size, args):
        super().__init__()
        self.v_size = [obs_size] + args.hidden_sizes + [1]
        self.pi = mlp(self.v_size, nn.ReLU)
    def forward(self, obs):
        return self.pi(obs)
