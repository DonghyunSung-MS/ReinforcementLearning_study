#-*-conding : utf-8 -*-

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#print(device)
class Agent(nn.Module): # pytorch nn 상속
    def __init__(self,env,device, h_size=5):
        super().__init__() # python 3 ref:https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        self.env = env
        self.device = device
        # state ->(Affine+relu)-> hidden ->(Affine+tanh)-> action)
        print()
        print("Model Architecture : state ->(Affine+relu)-> hidden ->(Affine+tanh)-> action[-1,1]")
        print()
        self.state_size = env.observation_space.shape[0] #state 갯수
        self.hidden_size = h_size # hidden layer 갯수
        self.action_size = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.state_size, self.hidden_size)# first fully connected
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)
        #print('action size is {}'.format(self.action_size))

    def set_weights(self,weights):
        # weights 1 dim np.array : [1~s*h(첫번째 층 weights),s*h+1 ~ s*h+h(첫번째 층 bias),두번째 같은 방식~]
        fc1_end = (self.state_size*self.hidden_size) + self.hidden_size
        fc1_W = torch.from_numpy(weights[:self.state_size*self.hidden_size].reshape(self.state_size,self.hidden_size))
        fc1_b = torch.from_numpy(weights[self.state_size*self.hidden_size : fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end : fc1_end + self.hidden_size*self.action_size].reshape(self.hidden_size,self.action_size))
        fc2_b = torch.from_numpy(weights[fc1_end + self.hidden_size*self.action_size : ])

        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.state_size + 1)*self.hidden_size + (self.hidden_size + 1)*self.action_size

    def forward(self,x):
        x.cuda(self.device)
        x = F.relu(self.fc1(x)).to(self.device)
        x = F.tanh(self.fc2(x)).to(self.device) #[-1,1]
        return x.cpu().data

    def evaluate(self,weights,gamma=0.99, max_t=5000):
        # Monte Carlo 방법을 사용하여 리턴을 반환한다.
        # max_t 시뮬레이션 시간 제한
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            #self.env.render()
            state = torch.from_numpy(state).float().to(self.device) ## cuda 로 보내기
            action = self.forward(state)

            state, reward, done, _ = self.env.step(action) # 환경에서 놀
            #print(state," ",action," ",reward,done," ",_)
            episode_return +=reward*math.pow(gamma, t)
            if done:
                break
        return episode_return
