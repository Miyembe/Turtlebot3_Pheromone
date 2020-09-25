# custom gazebo gym env

import sys
import numpy as np
import math
import time
import gym
#import rospy
from gym import spaces

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class GazeboEnv(gym.Env):

    def __init__(self):
        print('Gazebo Environment is successfully initialised.')
        self.nrow = 10
        self.ncol = 10
        
        nA = 4
        nS = self.nrow * self.ncol

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.state = np.array(0).reshape(1,)
        self.done = False
        self.reward = None

    
    def step(self, action):
        
        # Decomposing the state with row and column
        row = self.state[0] // self.ncol
        col = self.state[0] % self.ncol

        print("Current state is: {}, ({}, {})".format(self.state[0], row, col))

        # Update state by the given action
        if action == LEFT:
            col = max(col-1, 0)
            print("Action taken is LEFT")
        elif action == DOWN:
            row = min(row+1, self.nrow-1)
            print("Action taken is DOWN")
        elif action == RIGHT:
            col = min(col+1, self.ncol-1)
            print("Action taken is RIGHT")
        elif action == UP:
            row = max(row-1, 0)
            print("Action taken is UP")

        # Synthesize row and column into the state again
        next_state = row*self.ncol + col
        
        print("Next state is: {}, ({}, {})".format(next_state, row, col))


        
        # Episode termination
        if next_state == 99:
            done = True
            reward = 0.0
        else:
            done = False

            
        # Reward Functions

        # Constant Negative Penalty 
        if done is False:
            reward = -1.0
        
        # Obstacle Penalty
        # if next_state == ???:
        #     self.reward = -10
        print("Reward: {}, Done: {}".format(reward, done))
       

        self.state = np.array(next_state).reshape(1,)

        return self.state, reward, done

    def reset(self):
        self.state = np.array(0).reshape(1,)
        print("The robot is placed at the starting point.")
        
        return self.state