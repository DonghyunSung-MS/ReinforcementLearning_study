import gym


env = gym.make('CartPole-v0')
iterations = 100
for i in range(iterations):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        #if done:
            #print(info)
