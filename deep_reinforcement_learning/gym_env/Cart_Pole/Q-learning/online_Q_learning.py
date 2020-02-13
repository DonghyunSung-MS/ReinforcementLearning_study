#-*-conding: utf-8-*-

import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
'''
state[4,] : [cart pos, cart vel,pole pos, pole vel] continuous
action[1,] : discrete (0 or 1)
linear function approximation Q(s,)[2,] = s^T W
action = epsilon greedy w.r.t Q
'''
state_dim = env.observation_space.shape[0]
action_dim = 1
Q_dim = action_dim + 1 # 액션 스페이스 크기는 1이지만 액션이 불연속적인(discrete) 2가지 뿐이라 이대하여 모두 계산하자
epsilon = 0.9
epsilon_decay = 0.8
lr = 0.0001
gamma = 0.99
actions = [0,1]
iterations = 10000

def e_greedy(Q):
    global epsilon
    epsilon *= epsilon_decay
    tmp = np.random.rand(1)
    if epsilon < tmp: #확률보다 크면 greedy
        return np.argmax(Q)
    else:
        return np.random.choice(actions)

weights = np.random.randn(state_dim,Q_dim)  #weights[4,2] 초기화
loss_history = []
#------------------------------train-------------------------------------------
for i in range(iterations):
    done = False
    state = env.reset()
    Q = np.dot(state,weights)
    while not done:
        #env.render()
        action = e_greedy(Q)
        #print(action)
        next_state, reward, done, info = env.step(action)
        TD_target = reward + gamma*(1-int(done))*np.dot(next_state,weights)
        weights = weights + lr*np.outer(state,(TD_target - Q).reshape(-1,1))
        state = next_state
        Q = np.dot(state,weights)
    loss_history.append(np.sum((TD_target-Q)**2))
        #env.render()
#------------------------------test-------------------------------------------

plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

Q = np.dot(state,weights)
env.reset()

while not done:
    action = e_greedy(Q)
    env.render()
    next_state, reward, done, info = env.step(action)
