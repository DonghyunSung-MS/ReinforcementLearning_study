import os
import mujoco_py
import time

import numpy as np
import math
from gym.envs.mujoco import mujoco_env
from gym import utils

#Ref : Openai/gym
class Walker2dModify(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, target_vel, target_torso_pitch, target_height):
        self.target_vel = target_vel
        self.target_torso_pitch = target_torso_pitch
        self.target_height = target_height
        mujoco_env.MujocoEnv.__init__(self, "walker2d_modify.xml", 4)
        utils.EzPickle.__init__(self)


        #print("init")
    def step(self, a):
        x_= self.sim.data.qpos[0] #qpos [x z torso_pitch
                                  #right_hip right_knee right_ankle
                                  #left_hip left_knee left_ankle]
        self.do_simulation(a, self.frame_skip)
        x, z, torso_pitch = self.sim.data.qpos[0:3]

        #Reward shaping

        x_dot = (x - x_) / self.dt

        alive_bonus = 1.0 #hyper parameters
        error = [(x_dot - self.target_vel), z-self.target_height, torso_pitch - self.target_torso_pitch ]
        Q = [1, 0.1 , 0.1]

        error_cost = Q[0]*error[0]**2 + Q[1]*error[1]**2 + Q[2]*error[2]**2 
        error_cost = math.exp(-error_cost)
        ctrl_cost = math.exp(-np.square(a).sum())

        weight = [0.6, 0.3, 0.1] #hyper parameters
        reward_ele = [alive_bonus, error_cost, ctrl_cost]

        reward = np.dot(weight,reward_ele)

        #termination condition
        done = not (z > 0.8 and z < 2.0 and
                    torso_pitch > -1.0 and torso_pitch < 1.0)

        # partially observable state feature

        observation = self._get_obs()#(x, z, torso_pitch,
                                 #       x_dot, z_dot, torso_pitch_dot)

        return observation, reward, done, dict(alive_bonus = alive_bonus,
                                                 error_cost = error_cost,
                                                 ctrl_cost = ctrl_cost)


    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[0:3], qvel[0:3]]).ravel()

    def reset_model(self):
        c = 0.1
        qvel_size = self.sim.data.qvel.shape[0]
        self.set_state(
            self.init_qpos + self.np_random.uniform(
                low=c, high=-c, size=self.model.nq),
            np.zeros(qvel_size)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

'''
env = Walker2dModify(0.5, 0, 0.8)

print(env.action_space.shape[0])
print(env.observation_space.shape[0])

ep_list = []

for i in range(1000):
    env.reset()
    random_reward = 0
    step = 0
    done = False
    while not done and step<3000:
        obs, r, done, info = env.step(np.zeros(7))
        step+=1
        random_reward+=r
        #print(r)
        #env.render()
        #print(obs[3],obs[2],obs[1])
    ep_list.append(random_reward)

ep_list = np.array(ep_list)
print("random agent reward(baseline)",ep_list.max(),ep_list.min(),ep_list.mean(), ep_list.std())
'''
