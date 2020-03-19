import sys, os
import gym
import gym.wrappers as wrappers
sys.path.append(os.pardir)
from a2c.nn_model import Actor

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1")
parser.add_argument('--hidden_sizes', type=list, default=[64, 64])
parser.add_argument('--test_iter', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--save_path',default='./model/')
args = parser.parse_args()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def test():
    env_to_wrap = gym.make(args.env_name)
    env = wrappers.Monitor(env_to_wrap, '../a2c/wrapper',force=True)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = Actor(state_size,action_size,args)

    tr_model_path = args.save_path+'model.pth.tar'
    tr_model = torch.load(tr_model_path)
    actor.load_state_dict(tr_model)

    steps = 0

    for episode in range(args.test_iter):
        obs = env.reset()
        obs = torch.from_numpy(obs).float() # tensor cpu
        done = False
        score = 0
        while not done:
            a = actor.get_action(obs.to(device)) #Discrete, cpu
            obs_, r, done, _ = env.step(a)


            mask = 0 if done else 1
            obs_ = torch.from_numpy(obs_).float() # tensor cpu
            #reward modifying
            r = r if not done or score == 499 else -1

            score += r
            obs = obs_
        score = score if score == 500.0 else score + 1

        if episode % args.log_interval == 0:
            print('{} episode | score: {:.2f}'.format(episode, score))
    env.close()
    env_to_wrap.close()

if __name__=='__main__':
    test()
