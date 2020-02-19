'''
In ddpg, we need actor(policy), critic(action-value function) + target_net , replay buffer(s,a,r,s',d)
gradient with batch

Optimization point of view

    -- policy: expectation of Q--> maximize

    -- critic: expectation of mean squared bellman error(MSBE)

    -- update parameters with polyak averaging new <- rho*new+ (1-rho)*old

Architecture

class-Agent
    class-actor

                Sequential(
                    (0): Linear(in_features=2, out_features=256, bias=True)
                    (1): ReLU()
                    (2): Linear(in_features=256, out_features=256, bias=True)
                    (3): ReLU()
                    (4): Linear(in_features=256, out_features=1, bias=True)
                    (5): Tanh()
                )

    class-critic

                Sequential(
                    (0): Linear(in_features=3, out_features=256, bias=True)
                    (1): ReLU()
                    (2): Linear(in_features=256, out_features=256, bias=True)
                    (3): ReLU()
                    (4): Linear(in_features=256, out_features=1, bias=True)
                    (5): Tanh()
                )

class-replay buffer
    method load
    method save
'''

#Ref: pytorch docs, spinning up docs

import torch
import torch.nn as nn

import numpy as np

# Build multi layer perceptron
def mlp(layer_sizes,activation,output_activation = nn.Identity):
    layers = []
    '''
    input_dim activattion
    hidden_dim activattion
            ...
    hidden_dim activattion
    output_dim output_activation
    '''
    for i in range(len(layer_sizes)-1):
        act = activation if i<len(layer_sizes)-2 else output_activation
        layers +=[nn.Linear(layer_sizes[i],layer_sizes[i+1]),act()]
    return nn.Sequential(*layers)

# o ---> a
class Actor(nn.Module):
    def __init__(self, observation_size, action_size, hidden_sizes, activation, action_limit, device):
        super().__init__()
        self.device = device
        self.policy_sizes = [observation_size] + list(hidden_sizes) + [action_size]
        self.policy_mlp = mlp(self.policy_sizes,activation).float().to(device) #-1 to 1
        self.action_limit = action_limit
    def forward(self, observation):
        observation = observation.float().to(self.device)
        return self.action_limit*self.policy_mlp(observation) # scale to action_limit

# o, a ---> Q structure
class Critic(nn.Module):
    def __init__(self, observation_size, action_size, hidden_sizes, activation,device):
        super().__init__()
        self.device = device
        self.Q_sizes = [observation_size + action_size] + list(hidden_sizes) + [1]
        self.Q_mlp = mlp(self.Q_sizes,activation).float().to(device)
    def forward(self, observation, action):
        inputs = torch.cat([observation, action], dim = -1).float().to(self.device)
        Q_out = self.Q_mlp(inputs)
        return torch.squeeze(Q_out,-1)

# contain Actor and Critic
class AC_Agent(nn.Module):
    def __init__(self, env, device, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        self.device = device
        self.observation_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]

        self.actor = Actor(self.observation_size, self.action_size, hidden_sizes, activation, self.action_limit, self.device).float().to(device)
        self.critic = Critic(self.observation_size, self.action_size, hidden_sizes, activation, self.device).float().to(device)

    def act(self, observation):
        with torch.no_grad():
            return self.actor(observation)

class ReplayBuffer: #priority Queue based on MSBE or random access memory
    def __init__(self,env,memory_size=10000):
        self.memory_size = memory_size
        self.observation_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.observation = np.zeros((self.memory_size, self.observation_size)).astype(np.float32)
        self.action = np.zeros((self.memory_size, self.action_size)).astype(np.float32)
        self.reward = np.zeros((self.memory_size, 1)).astype(np.float32)
        self.next_observation = np.zeros((self.memory_size, self.observation_size)).astype(np.float32)
        self.done = np.zeros((self.memory_size, 1)).astype(np.float32)
        self.count = 0

    def save_sample(self, samples):
        self.count = self.count%self.memory_size
        self.observation[self.count, :] = samples[0]
        self.action[self.count, :] = samples[1]
        self.reward[self.count] = samples[2]
        self.next_observation[self.count, :] = samples[3]
        self.done[self.count] = samples[4]
        self.count+=1



    def sample_batch(self,mini_batch_size):
        indices = np.random.choice(self.memory_size, mini_batch_size)
        o = self.observation[indices, :]
        a = self.action[indices, :]
        r = self.reward[indices]
        o_ = self.next_observation[indices, :]
        d = self.done[indices]
        return (o, a, r, o_, d)
