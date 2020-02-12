#-*-conding : utf-8 -*-
## 출처 : https://towardsdatascience.com/reinforcement-learning-tutorial-with-open-ai-gym-9b11f4e3c204
import sys,os
sys.path.append(os.pardir)
from cross_entropy_method.Agent import *
import gym

import matplotlib.pyplot as plt
from collections import deque

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('MountainCarContinuous-v0')
env.seed(101)
np.random.seed(101)
agent = Agent(env) #class

'''
[min max]
action(force): [-1,1] continuous
observation: pos[-1.2, 0.6], vel[-0.07 0.07] continuous
print(env.observation_space.shape) (2,)
print(env.action_space.shape)      (1,)
'''

def cem(n_iterations=500, max_t=1000, gamma=0.99, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
    '''
    Cross Entropy Method 진화 알고리즘의 일종으로 점수순으로 줄을 세워 일부분을 추출하여 평균을 내는 작업을 반복한다.
    이는 결국 샘플들이 점점 잘하는 사람만 남는다는 아이디어에서 출발한다.
    자세한 사항은 Wiki 검색.
    '''
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores=[]
    # 초기에는 랜덤노말(가우시안분포) N(0,sigma^2)로 초기 weight 설정
    best_weight = sigma*np.random.randn(agent.get_weights_dim())

    for i_iteration in range(1,n_iterations+1):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weight,gamma,max_t) for weight in weights_pop ])
        elite_idxs = rewards.argsort()[-n_elite:] # 점수 높은 그룹 인덱스 추출
        elite_weights = [weights_pop[i] for i in elite_idxs] # 인덱스에 해당하는 weight 찾기
        best_weight = np.array(elite_weights).mean(axis=0) #그들을 평균낸다.
        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)

        torch.save(agent.state_dict(), 'checkpoint.pth')

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
    return scores

scores = cem()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
