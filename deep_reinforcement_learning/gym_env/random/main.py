import os
import sys
sys.path.append(os.pardir)

import gym
import numpy as np

from custumized_env.walker2d_modify import Walker2dModify

#env = gym.make('CartPole-v0')
env = Walker2dModify(0.5, 0)

iterations = 1000
ep_list = []

for i in range(iterations):
    done = False
    random_reward = 0
    state = env.reset()
    while not done :
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        random_reward+=reward
        #env.render()
    ep_list.append(random_reward)
        #if done:
            #print(info)


ep_list = np.array(ep_list)
print("random agent reward(baseline)",ep_list.max(),ep_list.min(),ep_list.mean(), ep_list.std())
#Env Walker2dModify: random agent reward(baseline) 31.7736014403536 0.9993246043850222 7.579464718095872 4.501425560087963
