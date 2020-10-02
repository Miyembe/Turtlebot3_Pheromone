#!/usr/bin/env python
# Node that handles pheromone layer
# Subscriber - Robot (x, y) position
# Publisher - Pheromone value at (x, y)

import numpy as np
import rospy
import gazebo_msgs.msg
from gazebo_msgs.msg import ModelStates 
from std_msgs.msg import Float32
import math
import time

class Node():

    def __init__(self, phero):
        self.pheromone = phero
        self.pub_phero = rospy.Publisher('/phero_layer', Float32, queue_size=1)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pheroCallback, self.pheromone)
    
    def pheroCallback(self, message, cargs):
        
        # Reading from arguments
        pose = message.pose[1]
        pos = pose.position
        phero = cargs

        # Read pheromone value at the robot position
        res = phero.resolution
        round_dp = int(math.log10(res))
        x = round(pos.x, round_dp) # round the position value so that they fit into the centre of the cell.
        y = round(pos.y, round_dp) # e.g. 0.13 -> 0.1
        x = int(x*res)
        y = int(y*res)
        
        # Position conversion from Robot into pheromone matrix (0, 0) -> (n+1, n+1) of 2n+1 matrix
        x_index = x + (phero.num_cell-1)/2
        y_index = y + (phero.num_cell-1)/2
        if x_index < 0 or y_index < 0 or x_index > phero.num_cell-1 or y_index > phero.num_cell-1:
            raise Exception("The pheromone matrix index is out of range.")
        
        # Assign pheromone value and publish it
        phero_val = phero.getPhero(x_index, y_index)
        self.pub_phero.publish(phero_val)

        # Pheromone injection
        phero.injection(x_index, y_index, 1, 3)


        # Update pheromone matrix in every 0.1s
        time_cur = time.clock()
        if time_cur-phero.step_timer >= 0.1: 
            phero.update()
            phero.step_timer = time_cur
        
        
        print("Position: ({}, {}), Index position: ({}, {}), Pheromone Value: {}".format(x, y, x_index, y_index, phero_val))
        print("Real position: ({}, {})".format(pos.x, pos.y))
        #print("x: {}, y: {}".format(x, y))
    


class Pheromone():

    def __init__(self):
        self.resolution = 10 # grid cell size = 1 m / resolution
        self.size = 10 # m
        self.num_cell = self.resolution * self.size + 1
        if self.num_cell % 2 == 0:
            raise Exception("Number of cell is even. It needs to be an odd number")
        self.grid = np.ones((self.num_cell, self.num_cell))
        self.evaporation = 10 # elapsed seconds for pheromone to be halved

        # Timers
        self.update_timer = time.clock()
        self.step_timer = time.clock()
        self.injection_timer = time.clock()

    def getPhero(self, x, y):
        return self.grid[x, y]

    def setPhero(self, x, y, value):
        self.grid[x, y] = value

    # Inject pheromone at the robot position and nearby cells in square. Size must be an odd number. 
    def injection(self, x, y, value, size):
        if size % 2 == 0:
            raise Exception("Pheromone injection size must be an odd number.")
        time_cur = time.clock()
        if time_cur-self.injection_timer > 0.4:
            for i in range(size):
                for j in range(size):
                    self.grid[x-(size-1)/2+i, y-(size-1)/2+j] = value
            self.injection_timer = time_cur
    
    # Update all the pheromone values depends on natural phenomena, e.g. evaporation
    def update(self):
        time_cur = time.clock()
        #print('current time: {}, Last time: {}'.format(time_cur, self.time))
        time_elapsed = time_cur - self.update_timer
        self.update_timer = time_cur

        decay = 2**(-time_elapsed/self.evaporation)
        for i in range(self.num_cell):
            for j in range(self.num_cell):
                self.grid[i, j] = decay * self.grid[i, j]
        
        

    
if __name__ == "__main__":
    rospy.init_node('pheromone')
    Phero1 = Pheromone()
    node1 = Node(Phero1)
    rospy.spin()
