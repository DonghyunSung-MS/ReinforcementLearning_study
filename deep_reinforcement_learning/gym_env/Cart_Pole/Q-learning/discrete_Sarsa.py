#-*-coding: utf-8-*-
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
'''
CartPole은 action space는 2개로 불연속적이지만
state space(observation space)는 연속이다. 이를 grid world화 하여
sarsa로 Q(action-value function)을 계속적으로 갱신하여한다. (policy evaluation)
그에 대하여 action은 epsilon greedy 하게 진행할 것이다. (policy improvement)
'''
def MaxAction(Q,state):
    values = np.array([Q[state,a] for a in range(2)])
    action = np.argmax(values)
    return action

pole_theta_space = np.linspace(-0.2, 0.2, 10)
pole_omega_space = np.linspace(-4, 4, 10)
cart_pos_space = np.linspace(-2.4, 2.4, 10)
cart_vel_space = np.linspace(-4, 4, 10)

def getState(observation):
    cart_x, cart_v, pole_th, pole_w = observation
    cart_x = int(np.digitize(cart_x, cart_pos_space))
    cart_v = int(np.digitize(cart_v, cart_vel_space))
    pole_th = int(np.digitize(pole_th, pole_theta_space))
    pole_w = int(np.digitize(pole_w, pole_omega_space))

    return (cart_x, cart_v, pole_th, pole_w)

if __name__ =="__main__":
    env = gym.make('CartPole-v0')
    # Hyper parameters
    GAMMA = 0.9
    ALPHA = 0.1
    EPS = 0.999
    NUM_ITERATION = 50000
    
    states = []
    # state space
    for i in range(len(cart_pos_space)+1):
        for j in range(len(cart_vel_space)+1):
            for k in range(len(pole_theta_space)+1):
                for l in range(len(pole_omega_space)+1):
                    states.append((i,j,k,l))

    Q = {} # look up table method(not recommaned when high-demsion of state and action)

    for s in states:
        for a in range(2):
            Q[s,a] = 0


    totalRewards = np.zeros(NUM_ITERATION)

    for i in range(NUM_ITERATION):
        if i%5000 == 0:
            print('im in ',i)
        done = False
        # s a r s'a'
        observation = env.reset()
        s = getState(observation) # continuous -> discrete
        rand = np.random.rand()
        a = MaxAction(Q,s) if rand<(1-EPS) else env.action_space.sample() # epsilon greedy part
        epReward = 0
        t = 0
        while not done:
            t +=1
            next_observation, reward, done, _ = env.step(a)
            next_s = getState(next_observation)
            next_a = MaxAction(Q,next_s) if rand<(1-EPS) else env.action_space.sample() # epsilon greedy part
            Q[s,a] = Q[s,a] + ALPHA * (reward + (1-int(done))*GAMMA*Q[next_s,next_a] - Q[s,a])
            epReward += GAMMA*pow(reward,t)
            s, a = next_s, next_a

        EPS -=2/NUM_ITERATION if EPS>0 else 0
        totalRewards[i] = epReward

plt.plot(totalRewards)
plt.xlabel("#iter")
plt.ylabel("discounted return")
plt.show()