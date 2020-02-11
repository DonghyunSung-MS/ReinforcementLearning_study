import numpy as np

class Enviroment:
    def __init__(self, battery_consumption = -0.02, gamma = 0.99):
        self.num_states = 11
        self.states = [x for x in range(self.num_states)]

        self.terminal_states = [3,6]
        self.win_state = self.terminal_states[0]
        self.loose_state = self.terminal_states[1]
        self.non_terminal_states = [t for t in range(self.num_states) if t!=3 and t!=6]
        self.num_actions = 4
        self.actions = [x for x in range(self.num_actions)]

        locations = []
        index = 0
        self.nCols = 3
        self.nRows = 4
        # set map
        self.map = -np.ones((self.nCols+2,self.nRows+2))
        for i in range(self.nCols):
            for j in range(self.nRows):
                self.map[i+1,j+1] = 0
        self.map[2,2] = -1 # add wall
        for i in range(self.nCols):
            for j in range(self.nRows):
                if self.map[i+1,j+1]==0:
                    locations.append((i+1,j+1))
                    index = index + 1
        # action -> move
        self.move = [(0,-1),(-1,0),(0,1),(1,0)] # match index with actions

        self.P = np.zeros((self.num_states,self.num_actions,self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                current_location = locations[s]
                # heading collectly  ####################################################################################
                next_location = (current_location[0] + self.move[a][0],current_location[1] + self.move[a][1])

                if self.map[next_location[0],next_location[1]] == -1: # there is barrier or wall
                    next_location = current_location
                    next_s = self.states[locations.index(next_location)]
                else:
                    next_s = self.states[locations.index(next_location)]
                self.P[s,a,next_s] = self.P[s,a,next_s] + 0.8
                # left error ############################################################################################
                next_location = (current_location[0] + self.move[a-1][0],current_location[1] + self.move[a-1][1])
                if self.map[next_location[0],next_location[1]] == -1: # there is barrier or wall
                    next_location = current_location
                    next_s = self.states[locations.index(next_location)]
                else:
                    next_s = self.states[locations.index(next_location)]
                self.P[s,a,next_s] = self.P[s,a,next_s] + 0.1
                # right error ############################################################################################
                next_location = (current_location[0] + self.move[(a+1)%4][0],current_location[1] + self.move[(a+1)%4][1])

                if self.map[next_location[0],next_location[1]] == -1: # there is barrier or wall
                    next_location = current_location
                    next_s = self.states[locations.index(next_location)]
                else:
                    next_s = self.states[locations.index(next_location)]
                self.P[s,a,next_s] = self.P[s,a,next_s] + 0.1


        self.reward = float(battery_consumption)
        self.gamma = float(gamma)
        self.current_state =None
        self.done = None
    def reset(self):
        self.current_state = np.random.choice(self.non_terminal_states)
        self.done = False
        self.final_reward = None
        return self.current_state, self.done
    def step(self, action, prob):
        if self.current_state is None:
            raise ValueError("current state should be initialized by reset method")
        if self.current_state is self.terminal_states:
            raise ValueError("current state is terminal states")
        action = int(action)
        self.info = "prob : "+str(prob)
        next_state = np.random.choice(self.num_states,p=self.P[self.current_state,action,:])

        if next_state == self.win_state:
            self.final_reward = 1
            self.done = True
        elif next_state == self.loose_state:
            self.final_reward = -1
            self.done = True

        return self.reward, next_state, self.done, self.info, self.final_reward

    def random_action(self):
        return np.random.choice(self.num_actions)
