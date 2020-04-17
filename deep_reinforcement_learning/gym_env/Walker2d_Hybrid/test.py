import sys, os
sys.path.append(os.pardir)

#package
import argparse
from collections import deque
import gym
import torch
import numpy as np

#customizing class(Capital) or function(lower)
from custumized_env.walker2d_modify import Walker2dModify
from nn_model import PPOActor, PPOCritic
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--target_velocity", type=float, default=0.5)
parser.add_argument("--target_pitch", type=float, default=0)
parser.add_argument("--target_height", type=float, default=1)

parser.add_argument("--learning_continue",action="store_true",default=True)
parser.add_argument("--num_trial", type=int, default=10)
parser.add_argument("--env_name", type=str, default="Walker2dModify")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/'+str("Walker2dModify_0.5_0_1_reward_modification")+'/', help='')#should modify in sh file.
#parser.add_argument('--save_path', default='./save_model/'+str("Pendulum")+'/', help='')#should modify in sh file.
parser.add_argument('--render', action="store_true", default=True)

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lamda', type=float, default=0.98)

parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--layer_size', type=int, default=4)

parser.add_argument('--batch_size', type=int, default=100)

parser.add_argument('--actor_lr', type=float, default=1e-3)
parser.add_argument('--critic_lr', type=float, default=1e-4)
parser.add_argument('--clip_param', type=float, default=0.2)

parser.add_argument('--model_update_num', type=int, default=10)
parser.add_argument('--max_iter_num', type=int, default=2000)
parser.add_argument('--total_sample_size', type=int, default=50000)
parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--goal_score', type=int, default=1000)
parser.add_argument('--logdir', type=str, default='./logs/20200403_64_4',
                    help='tensorboardx logs directory') #should  modify in sh file.args = parser.parse_args()


def main(seed_num, target_velocity, target_pitch, target_height):
    env = Walker2dModify(target_velocity, target_pitch, target_height)
    #env = gym.make("Pendulum-v0")
    env.seed(seed_num*10)
    torch.manual_seed(seed_num*10)

    action_size = env.action_space.shape[0] # joint torque
    #state_size = env.observation_space.shape[0]
    state_size = 3 #Fx Fz xp
    # observation x,z,pitch, d(x,z,pitch)

    print("----------------------------")
    print(i+1,"th Try")
    print("env(PPO) : ",args.env_name)
    print("state_size : ",state_size)
    print("action_size : ",action_size)
    print("----------------------------")

    actor = PPOActor(state_size, action_size, args)
    critic = PPOCritic(state_size, args)

    if args.learning_continue:
        ckpt_path_a = args.save_path + str(1)+'th_model_a.pth.tar'
        ckpt_path_c = args.save_path + str(1)+'th_model_c.pth.tar'
        print(ckpt_path_a)
        pretrained_actor = torch.load(ckpt_path_a)
        pretrained_critic = torch.load(ckpt_path_c)
        actor.load_state_dict(pretrained_actor)
        critic.load_state_dict(pretrained_critic)

    #writer = SummaryWriter(args.logdir)

    #sample data(trajectories)--------------------------------------------------
    episodes = 0
    recent_return = deque(maxlen=100)
    observation_sample = deque(maxlen=100)
    ep_len_history = deque(maxlen=100)

    info = None

    for iter in range(args.num_trial):
        trajectories = deque()

        # how many sample is needed to update critic or actor
        done = False
        episodes +=1
        time_steps = 0
        reward_sum = 0

        observation = env.reset() # []
        #state_feature = obs2feature(observation, args) #[1,[feature_dim]] numpy
        #state_feature = observation #for full observablility
        state_feature = np.reshape(state_feature,[1, state_size])

        while not done:
            time_steps += 1
            env.render()
            #print(state_feature)
            mu, std = actor(torch.Tensor(state_feature))
            action = actor.get_action(mu, std) #torque

            next_observation, reward, done, info = env.step(mu.detach().numpy())

            next_state_feature = obs2feature(next_observation, args)
            #next_state_feature = next_observation #full observablility

            mask = 0 if done else 1
            reward_sum+=reward

            observation_sample.append(observation)

            trajectories.append((state_feature, action, reward, mask))

            observation = next_observation
            state_feature = next_state_feature
            state_feature = np.reshape(state_feature,[1, state_size])

            recent_return.append(reward_sum)
            ep_len_history.append(time_steps)
            #logging
            if episodes%args.log_interval==0:
                #writer.add_scalar('log/ep_len',time_steps,episodes)
                #writer.add_scalar('log/reward',reward_sum,episodes)
                print("{}th episodes | episode length : {} | running average return : {:.2f}".format(episodes, np.array(ep_len_history).mean(), np.array(recent_return).mean()))




if __name__=="__main__":
    for i in range(args.num_trial):
        main(args.num_trial, args.target_velocity, args.target_pitch, args.target_height)
