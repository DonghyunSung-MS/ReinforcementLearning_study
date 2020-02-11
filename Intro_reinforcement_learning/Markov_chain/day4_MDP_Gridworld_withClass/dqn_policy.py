import numpy as np

class dqnPolicy:
    def __init__(self, n_states=11, n_actions=4, policy_choice="random"):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.policy_choice = policy_choice
        # given states which action
        self.policy = np.zeros((self.n_states,self.n_actions))
        if self.policy_choice=="random":
            self.policy = 0.25*np.ones((self.n_states,self.n_actions))
        elif self.policy_choice == 'bad':
            self.policy[0,2] = 1
            self.policy[0,2] = 1
            self.policy[1,2] = 1
            self.policy[2,2] = 1
            self.policy[3,2] = 1
            self.policy[4,3] = 1
            self.policy[5,2] = 1
            self.policy[6,2] = 1
            self.policy[7,2] = 1
            self.policy[8,2] = 1
            self.policy[9,2] = 1
            self.policy[10,1] = 1
        elif self.policy_choice =='optimal':
            self.policy[0,2] = 1
            self.policy[1,2] = 1
            self.policy[2,2] = 1
            self.policy[3,2] = 1
            self.policy[4,1] = 1
            self.policy[5,1] = 1
            self.policy[6,1] = 1
            self.policy[7,1] = 1
            self.policy[8,0] = 1
            self.policy[9,0] = 1
            self.policy[10,0] = 1
        elif self.policy_choice=="optimalNoise":
            ep = 0.1
            self.policy[0,2] = 1
            self.policy[1,2] = 1
            self.policy[2,2] = 1
            self.policy[3,2] = 1
            self.policy[4,1] = 1
            self.policy[5,1] = 1
            self.policy[6,1] = 1
            self.policy[7,1] = 1
            self.policy[8,0] = 1
            self.policy[9,0] = 1
            self.policy[10,0] = 1
            self.policy = self.policy+(ep/4)*np.ones((self.n_states,self.n_actions))
            self.policy = self.policy/np.sum(self.policy,axis=1).reshape((self.n_states,1))
