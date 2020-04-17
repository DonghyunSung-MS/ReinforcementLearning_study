import torch
import torch.nn as nn

from torch.distributions import Normal

def mlp(input_size, hidden_size, output_size, layer_size, act,out_act=nn.Identity):
    layers=[]
    for i in range(layer_size):
        ac = act if i<layer_size-1 else out_act
        if i ==0:
            layers += [nn.Linear(input_size, hidden_size),ac()]

        elif i==layer_size-1:
            layers += [nn.Linear(hidden_size, output_size),ac()]
        else:
            layers += [nn.Linear(hidden_size, hidden_size),ac()]

    return nn.Sequential(*layers)

class PPOActor(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.a_net = mlp(input_dim, args.hidden_size, output_dim, args.layer_size, nn.Tanh)
        #print(self.a_net)

    def forward(self, x):
        # input -> output(mean of torque+std(constant))
        mu = self.a_net(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)

        return mu, std

    def get_action(self, mu, std):
        normal = Normal(mu, std)
        action = normal.sample()
        return action.data.numpy()

    def get_log_prob(self, actions, mu, std):
        normal = Normal(mu, std)
        log_prob = normal.log_prob(actions) #log_probability of policy
        return log_prob

class PPOCritic(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.c_net = mlp(input_dim, args.hidden_size, 1, args.layer_size, nn.Tanh)
        #print(self.c_net)


    def forward(self, x):
        return self.c_net(x)
