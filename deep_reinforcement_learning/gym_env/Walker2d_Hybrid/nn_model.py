import torch
import torch.nn as nn

from torch.distributions import Normal

class PPOActor(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, output_dim)

    def forward(self, x):
        # input -> output(mean of torque+std(constant))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        mu = self.fc3(x)
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
        self.fc1 = nn.Linear(input_dim, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
