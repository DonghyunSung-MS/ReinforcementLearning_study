import numpy as np
from dqn_env import Enviroment
from dqn_policy import dqnPolicy
import matplotlib.pyplot as plt

class Mdp(Enviroment):
    def __init__(self, policy, battery_consumption=-0.02, gamma = 0.99, num_simulation=1000):
        super().__init__(battery_consumption, gamma)
        self.policy = policy
        self.num_simulation = int(num_simulation)

    def run_one_simulation(self, is_printing=True):
        self.current_state,self.done = self.reset()

        while not self.done:
            action = np.random.choice(self.actions,p=self.policy[self.current_state,:])
            prob = self.policy[self.current_state,action]
            self.reward ,next_state, self.done , self.info ,self.final_reward=self.step(action,prob)

            if is_printing:
                msg= "s: {:2}, a: {}, r: {:5.2f}, sl: {:2}, done: {:1}, info: {:4}, final reward: {} "
                print(msg.format(self.current_state, action, self.reward, next_state, self.done, self.info, self.final_reward))

            self.current_state = next_state

        if self.current_state == self.win_state:
            return 1
        elif self. current_state == self.loose_state:
            return 0
    def run_many_simulation(self):
        simulation_history=[]
        for _ in range(self.num_simulation):
            win_or_loose = self.run_one_simulation(is_printing=False)
            simulation_history.append(win_or_loose)

        x = np.array(simulation_history)
        running_success_rate = np.cumsum(x)/(np.arange(self.num_simulation)+1)

        print("Number of simulations : {}".format(self.num_simulation))
        print("Sucees rate           : {}".format(running_success_rate[-1]))

        plt.plot(running_success_rate)
        plt.title("Running Sucess Rate")
        plt.show()


class IPE_V(Enviroment):
    def __init__(self, policy, num_iteration=10, battery_consumption=-0.02,gamma=0.99):
        super().__init__(battery_consumption,gamma)
        self.policy = policy_choice
        self.num_iteration = int(num_iteration)
