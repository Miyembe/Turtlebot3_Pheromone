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
        phero_val = phero.getPhero(x,y)

        
        self.pub_phero.publish(phero_val)
        print("Position: ({}, {}), Pheromone Value: {}".format(x, y, phero_val))
        #print("x: {}, y: {}".format(x, y))
    


class Pheromone():

    def __init__(self):
        self.resolution = 10 # grid cell size = 1 m / resolution
        self.size = 10 # m
        self.num_axis = self.resolution * self.size
        self.grid = np.zeros((self.num_axis, self.num_axis))

    def getPhero(self, x, y):
        return self.grid[x, y]

    def setPhero(self, x, y, value):
        self.grid[x, y] = value

    # Inject pheromone at the robot position and nearby cells in square. Size must be an odd number. 
    def injection(self, x, y, value, size):
        for i in range(size):
            for j in range(size):
                self.grid[x-(size-1)/2+i, y-(size-1)/2+j] = value
    
    def update()

    
if __name__ == "__main__":
    rospy.init_node('pheromone')
    Phero1 = Pheromone()
    node1 = Node(Phero1)
    rospy.spin()
