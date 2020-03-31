import sys, os
sys.path.append(os.pardir)

#package
import argparse
from collections import deque
import gym
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import numpy as np

#customizing class(Capital) or function(lower)
from custumized_env.walker2d_modify import Walker2dModify
from common.Tensorboard2Csv import board2csv
from nn_model import PPOActor, PPOCritic

#parameters setting-------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--target_velocity", type=float, default=0.5)
parser.add_argument("--target_pitch", type=float, default=0)
parser.add_argument("--target_height", type=float, default=1)

parser.add_argument("--num_trial", type=int, default=5)
parser.add_argument("--env_name", type=str, default="Walker")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/'+str("Pendulum")+'/', help='')#should modify in sh file.
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lamda', type=float, default=0.98)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--actor_lr', type=float, default=1e-3)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--model_update_num', type=int, default=10)
parser.add_argument('--clip_param', type=float, default=0.2)
parser.add_argument('--max_iter_num', type=int, default=2000)
parser.add_argument('--total_sample_size', type=int, default=10000)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--goal_score', type=int, default=-300)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory') #should  modify in sh file.
args = parser.parse_args()

def obs2feature(observation):
    #observation [x, z, pitch, x_dot, z_dot, pitch_dot ]
    #HyperParameter
    K_x = [0.01, 0.02]
    K_z = [0.01, 0.02]
    K_f = 0.02
    #print(observation)
    Fx = K_x[0]*(args.target_pitch - observation[2]) + K_x[1]*(-observation[5])
    Fz = K_z[0]*(args.target_height - observation[1]) + K_z[1]*(-observation[4])
    xp = observation[0] + K_f*(args.target_velocity - observation[3]) #world frame

    return np.array([Fx, Fz, xp])

def update():
    raise NotImportError

def main(seed_num, target_velocity, target_pitch, target_height):
    env = Walker2dModify(target_velocity, target_pitch, target_height)
    env.seed(seed_num*10)
    torch.manual_seed(seed_num*10)

    action_size = env.observation_space.shape[0] # joint torque
    state_size = 3 #Fx Fz xp

    print("----------------------------")
    print(i+1,"th Try")
    print("env(PPO) : ",args.env_name)
    print("state_size : ",state_size)
    print("action_size : ",action_size)
    print("----------------------------")

    actor = PPOActor(state_size, action_size, args)
    critic = PPOCritic(state_size, args)

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)

    writer = SummaryWriter(args.logdir)

    #sample data(trajectories)--------------------------------------------------
    episodes = 0
    recent_rewards = deque(maxlen=1000)
    info = None

    for iter in range(args.max_iter_num):
        trajectories = deque()
        steps = 0

        # how many sample is needed to update critic or actor
        while steps < args.total_sample_size:
            done = False
            episodes +=1
            reward_sum = 0

            observation = env.reset() # []
            state_feature = obs2feature(observation) #[1,[feature_dim]] numpy

            while not done:
                steps+=1
                print(state_feature)
                mu, std = actor(torch.Tensor(np.reshape(state_feature,[1, 3])))
                action = actor.get_action(mu, std) #torque

                next_observation, reward, done, info = env.step(action)

                next_state_feature = obs2feature(next_observation)
                mask = 0 if done else 1



if __name__=="__main__":
    for i in range(args.num_trial):
        main(args.num_trial, args.target_velocity, args.target_pitch, args.target_height)
