import sys, os
import gym
import gym.wrappers as wrappers
sys.path.append(os.pardir)
from trpo.nn_model import Actor

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Walker2d-v2")
parser.add_argument('--hidden_sizes', type=list, default=[64, 64])
parser.add_argument('--test_iter', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--save_path',default='./model/')
args = parser.parse_args()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def test():
    env = gym.make(args.env_name)
    #env = wrappers.Monitor(env_to_wrap, '../trpo/wrapper',force=True)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    actor = Actor(state_size, action_size, args)

    tr_model_path = args.save_path+'Walker2d-v2_40_model.pth.tar'
    tr_model = torch.load(tr_model_path)
    actor.load_state_dict(tr_model)

    steps = 0

    for episode in range(args.test_iter):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            mu, std = actor(torch.from_numpy(obs).float().to(device))
            act = actor.get_action(mu, std)
            obs_, r, done, _ = env.step(act)
            env.render()

            mask = 0 if done else 1

            score += r
            obs = obs_

        if episode % args.log_interval == 0:
            print('{} episode | score: {:.2f}'.format(episode, score))
    env.close()
    #env_to_wrap.close()

if __name__=='__main__':
    test()
