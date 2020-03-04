import sys,os

import gym
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

sys.path.append(os.pardir)
from trpo.nn_model import Actor, Critic
from common.Tensorboard2Csv import board2csv
from common.function import *

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Walker2d-v2")
parser.add_argument('--hidden_sizes', type=list, default=[64, 64])
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--goal_score', type=int, default=3500)
parser.add_argument('--max_kl', type=float, default=1e-2)
parser.add_argument('--max_iter_num', type=int, default=100000)
parser.add_argument('--total_sample_size', type=int, default=10000)
parser.add_argument('--logdir', default='./logs')
parser.add_argument('--save_path',default='./model/')
parser.add_argument('--log_interval', type=int, default=10)
args = parser.parse_args()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def update(actor, critic, critic_optimizer, trajectories, obs_dim, act_dim):
    trajectories = np.array(trajectories)
    obs_samples = np.vstack(trajectories[:, 0])
    if act_dim == 1:
        act_samples = list(trajectories[:, 1])
    else:
        act_samples = np.vstack(trajectories[:, 1])
    r_samples = list(trajectories[:, 2]) #always scalar
    mask_samples = list(trajectories[:, 3]) #always scalar

    act_samples = torch.Tensor(act_samples)
    r_samples = torch.Tensor(r_samples)
    mask_samples = torch.Tensor(mask_samples)
    return_samples = cost_to_go(r_samples, mask_samples, args.gamma)

    criterion = torch.nn.MSELoss()

    values = critic(torch.from_numpy(obs_samples).float().to(device))
    targets = return_samples.unsqueeze(1).to(device) #reinforece

    critic_loss = criterion(values, targets) #(R-V(s))*grad(V)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # compute surrogate_loss
    mu, std = actor(torch.from_numpy(obs_samples).float().to(device))
    old_policy = actor.get_log_prob(act_samples.to(device), mu, std)
    actor_loss = surrogate_loss(actor, values, targets, obs_samples, old_policy.detach(), act_samples)

    # gradient of surrogate_loss
    actor_loss_grad = torch.autograd.grad(actor_loss, actor.parameters()) #del L
    actor_loss_grad = flat_grad(actor_loss_grad)


    search_dir = conjugate_gradient(actor.cpu(), obs_samples, actor_loss_grad.data.cpu(), nsteps = 10)

    gHg = (hessian_vector_product(actor.cpu(), obs_samples, search_dir) * search_dir).sum(0, keepdim=True)
    step_size = torch.sqrt(2 * args.max_kl / gHg)[0]
    maximal_step = step_size * search_dir

    params = flat_params(actor)

    old_actor = Actor(obs_dim, act_dim, args)
    update_model(old_actor, params)

    backtracking_line_search(old_actor.cpu(), actor.cpu(), actor_loss, actor_loss_grad,
                             old_policy.cpu(), params, maximal_step, args.max_kl,
                             values, targets, obs_samples, act_samples)

def trpo(seed_number):
    env = gym.make(args.env_name)

    env.seed(seed_number) #action_space.sample()
    torch.manual_seed(seed_number) #normal

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_limit = (env.action_space.low, env.action_space.high)

    print()
    print('=====================================================================')
    print('Trust Region Policy Optimization')
    print(args.env_name)
    print('State  dimension : ', obs_dim)
    print('Action dimension : ', act_dim)
    print('Action   limit   : ', action_limit,' ', action_limit)
    print('=====================================================================')
    print()

    actor = Actor(obs_dim, act_dim, args).float().to(device)
    critic = Critic(obs_dim, act_dim, args).float().to(device)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    writer = SummaryWriter(args.logdir)

    recent_rewards = deque(maxlen=100)
    episodes = 0

    for i in range(args.max_iter_num):
        trajectories = deque()
        steps = 0

        while steps < args.total_sample_size:
            done = False
            score = 0
            episodes +=1

            obs = env.reset()

            while not done:
                #env.render()
                steps += 1
                mu, std = actor(torch.from_numpy(obs).float().to(device))
                act = actor.get_action(mu, std)
                obs_, r, done, _ = env.step(act)
                mask = 0 if done else 1
                #obs, act, r -> numpy or scalar
                trajectories.append((obs, act, r, mask)) #(s,a,r,1-int(done))

                score += r
                obs = obs_

                if done:
                    recent_rewards.append(score)

        if i%10==0:
            print('Episodes : {} | Socre_avg : {}'.format(episodes, np.mean(recent_rewards)))
        #off-line MC(cost to go)
        actor.train(), critic.train()
        update(actor, critic, critic_optimizer, trajectories, obs_dim, act_dim)
        writer.add_scalar('log/score', float(score), episodes)

        if np.mean(recent_rewards) > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)

            ckpt_path = args.save_path +str(args.env_name)+'_'+str(seed_number)+'_'+'model.pth.tar'
            torch.save(actor.state_dict(), ckpt_path)
            print('Recent rewards exceed'+str(args.goal_score)+' . So end')
            break

if __name__=='__main__':
    for i in range(5):
        trpo(i*10)
    board2csv(os.getcwd(),os.listdir(args.logdir),'trpo',args)
