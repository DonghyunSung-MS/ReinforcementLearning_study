import sys, os
import gym
import random
import argparse
import numpy as np
from collections import deque
sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.optim as optim

from a2c.nn_model import Actor, Critic
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1")
parser.add_argument('--act_lr',type=float, default=1e-4)
parser.add_argument('--cri_lr',type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--ent_coef', type=float, default=0.1)
parser.add_argument('--hidden_sizes', type=list, default=[64, 64])
parser.add_argument('--goal_score', type=int, default=400)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--logdir', default='./logs')
parser.add_argument('--save_path',default='./model')
args = parser.parse_args()

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def update(actor, critic, actor_optimizer, critic_optimizer, sars):
    obs, a, r, obs_, mask = sars  # tensors cpu
    criterion = torch.nn.MSELoss() # critic

    value = critic(obs.to(device)) # tensor gpu
    target = r + mask * args.gamma*critic(obs_.to(device)) # tensor gpu

    critic_loss = criterion(value, target.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    '''
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
    '''
    #print(a)
    policy = actor.get_policy(obs.to(device)) #categorical object
    log_policy = policy.log_prob(torch.Tensor([a]))
    advantage = target - value
    entropy = policy.entropy()
    #print(log_policy,advantage.item(),entropy)

    actor_loss = -log_policy*advantage.item() + args.ent_coef*entropy
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

def a2c():

    board = SummaryWriter(args.logdir)

    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    obs_size = env.observation_space.shape[0] #Box
    act_size = env.action_space.n # Discrete

    # graph in device
    actor = Actor(obs_size, act_size, args).float().to(device)
    critic = Critic(obs_size, act_size, args).float().to(device)

    # Adam optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.act_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.cri_lr)

    ep_running_return = 0

    for i in range(args.max_iter_num):

        obs = env.reset()
        obs = torch.from_numpy(obs).float() # tensor cpu
        done = False

        ep_return = 0
        ep_len = 0

        while not done:

            a = actor.get_action(obs.to(device)) #Discrete, cpu
            obs_, r, done, _ = env.step(a)


            mask = 0 if done else 1
            obs_ = torch.from_numpy(obs_).float() # tensor cpu
            #reward modifying
            r = r if not done or ep_return == 499 else -1

            ep_len += 1
            ep_return += r

            #TD policy evaluation
            a = torch.tensor(a).float()
            sars = [obs, a, r, obs_, mask] # tensors cpu

            actor.train(), critic.train()
            update(actor, critic, actor_optimizer, critic_optimizer, sars)

            obs = obs_

        ep_return = ep_return if ep_return == 500.0 else ep_return + 1
        ep_running_return = 0.99 * ep_running_return + 0.01 * ep_return
        if i%args.log_interval==0:
            print('{} episode | average_return: {:.2f} | ep_len: {}'.format(i, ep_running_return, ep_len))
            board.add_scalar('log/return', float(ep_return), i)
            board.add_scalar('log/length', float(ep_len), i)

        if ep_running_return > args.goal_score or i==args.max_iter_num-1:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)

            ckpt_path = args.save_path + 'model.pth.tar'
            torch.save(actor.state_dict(), ckpt_path)
            print('Train terminated and save_model')
            break

if __name__=='__main__':
    a2c()
