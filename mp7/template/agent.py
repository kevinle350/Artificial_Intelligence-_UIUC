import numpy as np
import utils
import math

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO: write your function here

        if dead:
            reward = -1
        elif points > self.points:
            reward = 1
        else:
            reward = -0.1

        if self._train and self.s != None:              #and self.a != None:
            self.N[self.s][self.a] += 1
            #prev = self.generate_state(self.s)
#            prev = self.s
            maxQ = -99999  
            for i in self.actions:
                if maxQ < self.Q[s_prime][i]:
                    maxQ = self.Q[s_prime][i]

            learning_rate = ((self.C)/(self.C + self.N[self.s][self.a]))

            self.Q[self.s][self.a] += learning_rate * (reward + self.gamma * maxQ - self.Q[self.s][self.a])
        
        if not dead:     

            # self.s = (food_dir_x, food_dir_y, wall_x, wall_y, body_top, body_bot, body_left, body_right)
            # self.s = environment
            self.s = s_prime
            self.points = points

            best = -99999
            action = 0
            for i in self.actions:
                nTable = self.N[self.s][i]
                if nTable < self.Ne and self._train:
                    if best <= 1:
                        best = 1
                        action = i
                        
                else:
                    qTable = self.Q[self.s][i]
                    if qTable >= best:
                        best = qTable
                        action = i

            #self.N[food_dir_x][food_dir_y][wall_x][wall_y][body_top][body_bot][body_left][body_right][action] += 1
            self.a = action
            return self.a

        else:
            self.reset()
            return 0

            # for i in self.actions:
            #     nTable = self.N[food_dir_x][food_dir_y][wall_x][wall_y][body_top][body_bot][body_left][body_right][i]
            #     if nTable < self.Ne:
            #         qTable = 1
                        
            #     else:
            #         qTable = self.Q[food_dir_x][food_dir_y][wall_x][wall_y][body_top][body_bot][body_left][body_right][i]
            #     if qTable >= best:
            #         best = qTable
            #         action = i

            # self.N[food_dir_x][food_dir_y][wall_x][wall_y][body_top][body_bot][body_left][body_right][action] += 1
            # self.a = action
            # return self.a
            
        
        
        

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        head_x = environment[0]#int(environment[0]/utils.GRID_SIZE)
        head_y = environment[1]#int(environment[1]/utils.GRID_SIZE)
        body = environment[2]
        food_x = environment[3]#int(environment[3]/utils.GRID_SIZE)
        food_y = environment[4]#int(environment[4]/utils.GRID_SIZE)

        #initialize variables
        wall_x = 0
        wall_y = 0 
        body_top = 0 
        body_bot = 0 
        body_left = 0 
        body_right = 0 
        food_dir_x = 0
        food_dir_y = 0

        #head
        if head_x == utils.GRID_SIZE:
            wall_x = 1
        elif head_x == utils.DISPLAY_SIZE - 2*utils.GRID_SIZE:  
            wall_x = 2

        if head_y == utils.GRID_SIZE:
            wall_y = 1
        elif head_y == utils.DISPLAY_SIZE - 2*utils.GRID_SIZE: 
            wall_y = 2
 
        #body
        #body = [(int(headx/utils.GRID_SIZE), int(heady/utils.GRID_SIZE)) for (headx,heady) in body]
        
        if (head_x,head_y-utils.GRID_SIZE) in body:
            body_top = 1
        if (head_x,head_y+utils.GRID_SIZE) in body:
            body_bot = 1
        if (head_x-utils.GRID_SIZE,head_y) in body:
            body_left = 1
        if (head_x+utils.GRID_SIZE,head_y) in body:
            body_right = 1

        #food
        if head_x > food_x:
            food_dir_x = 1
        elif head_x < food_x:
            food_dir_x = 2

        if head_y > food_y:
            food_dir_y = 1
        elif head_y < food_y:
            food_dir_y = 2

        # print(body1)
        # print(body2)
        # print(head_x)
        # print(head_y)
        # print(body)
        return (food_dir_x, food_dir_y, wall_x, wall_y, body_top, body_bot, body_left, body_right)