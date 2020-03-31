import gym
import torch
import argparse
import numpy as np

from model import Actor

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="Pendulum-v0")
parser.add_argument("--load_model", type=str, default="./save_model/Pendulum/actor/10th_model_a.pth.tar")
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--test_iter', type=int, default=1000)
parser.add_argument('--hidden_size', type=int, default=128)
args = parser.parse_args()

def test():
    env=gym.make(args.env_name)

    env.seed(10)
    torch.manual_seed(10)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    actor = Actor(state_size, action_size, args)
    actor.load_state_dict(torch.load(args.load_model))

    for ep in range(args.test_iter):
        score = 0
        done = False
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            mu, std = actor(torch.Tensor(state))
            action = actor.get_action(mu, std)
            #random_action = env.action_space.sample()
            next_state, reward, done, info = env.step(mu.detach().numpy())
            env.render()

            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
        if ep% args.log_interval == 0:
            print(ep," ep | score ", score)
    env.close()

if __name__=='__main__':
    test()
