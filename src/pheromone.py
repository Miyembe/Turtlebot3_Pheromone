#!/usr/bin/env python
# Node that handles pheromone layer
# Subscriber - Robot (x, y) position
# Publisher - Pheromone value at (x, y)
import sys
sys.path.append('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation')
import roslib; roslib.load_manifest('turtlebot3_waypoint_navigation')
import os
import numpy as np
import tf
import rospy
import gazebo_msgs.msg
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from math import *
import time
from turtlebot3_waypoint_navigation.srv import PheroGoal, PheroGoalResponse
from turtlebot3_waypoint_navigation.srv import PheroInj, PheroInjResponse
from turtlebot3_waypoint_navigation.srv import PheroReset, PheroResetResponse
class Node():

    def __init__(self, phero):
        self.pheromone = phero
        self.phero_max = 1.0
        self.phero_min = 0.0
        self.is_phero_inj = True

        # Publisher & Subscribers
        self.pub_phero = rospy.Publisher('/phero_value', Float32MultiArray, queue_size=10)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pheroCallback, self.pheromone)
        #self.sub_inj = rospy.Subscriber('/phero_inj', Bool, self.injCallback)
        
        # Services
        self.srv_inj = rospy.Service('phero_inj', PheroInj, self.injAssign) # Used in continuous controller 
        self.srv_goal = rospy.Service('phero_goal', PheroGoal, self.nextGoal) # Used in phero_controller (discrete action)
        self.srv_reset = rospy.Service('phero_reset', PheroReset, self.serviceReset)
        self.is_service_requested = False
        self.theta = 0 
        
        self.log_timer = time.clock()
        self.log_file = open("phero_value.txt", "a+")
        self.is_saved = False
        self.is_loaded = False
        self.is_reset = False

    def posToIndex(self, x, y):
        phero = self.pheromone
        # Read pheromone value at the robot position
        res = self.pheromone.resolution
        round_dp = int(log10(res))
        x = round(x, round_dp) # round the position value so that they fit into the centre of the cell.
        y = round(y, round_dp) # e.g. 0.13 -> 0.1
        x = int(x*res)
        y = int(y*res)
        
        # Position conversion from Robot into pheromone matrix (0, 0) -> (n+1, n+1) of 2n+1 matrix
        x_index = x + (phero.num_cell-1)/2
        y_index = y + (phero.num_cell-1)/2
        if x_index < 0 or y_index < 0 or x_index > phero.num_cell-1 or y_index > phero.num_cell-1:
            raise Exception("The pheromone matrix index is out of range.")
        return x_index, y_index

    def indexToPos(self, x_index, y_index):
        
        x = x_index - (self.pheromone.num_cell-1)/2
        y = y_index - (self.pheromone.num_cell-1)/2

        x = float(x) / self.pheromone.resolution
        y = float(y) / self.pheromone.resolution

        return x, y

    def pheroCallback(self, message, cargs):
        
        # Reading from arguments
        pose = message.pose[-1]
        twist = message.twist[-1]
        pos = pose.position
        ori = pose.orientation
        phero = cargs
        x = pos.x
        y = pos.y

        angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        if angles[2] < 0:
            self.theta = angles[2] + 2*pi
        else: self.theta = angles[2]

        # Add two wheel position
        wheel_distance = 0.2
        pos_l = np.array([x+(cos(pi/2)*cos(self.theta)*(wheel_distance/2) - sin(pi/2)*sin(self.theta)*(wheel_distance/2)), y+(sin(pi/2)*cos(self.theta)*(wheel_distance/2) + cos(pi/2)*sin(self.theta)*(wheel_distance/2))])
        pos_r = np.array([x+(cos(pi/2)*cos(self.theta)*(wheel_distance/2) + sin(pi/2)*sin(self.theta)*(wheel_distance/2)), y+(-sin(pi/2)*cos(self.theta)*(wheel_distance/2) + cos(pi/2)*sin(self.theta)*(wheel_distance/2))])
        
        x_index, y_index = self.posToIndex(pos.x, pos.y)
        x_l_index, y_l_index = self.posToIndex(pos_l[0], pos_l[1])
        x_r_index, y_r_index = self.posToIndex(pos_r[0], pos_r[1]) 
        # Assign pheromone values from two positions and publish it
        phero_val = Float32MultiArray()
        phero_val.data = [phero.getPhero(x_l_index, y_l_index), phero.getPhero(x_r_index, y_r_index)]
        self.pub_phero.publish(phero_val)
        # # Assign pheromone value and publish it
        # phero_val = phero.getPhero(x_index, y_index)
        # self.pub_phero.publish(phero_val)
        
        # Pheromone injection (uncomment it when injection is needed)
        #if self.is_phero_inj is True:
        #    phero.injection(x_index, y_index, 0.2, 3, self.phero_max)


        # Update pheromone matrix in every 0.1s
        time_cur = time.clock()
        if time_cur-phero.step_timer >= 0.1: 
            phero.update(self.phero_min, self.phero_max)
            phero.step_timer = time_cur

        #log_time_cur = time.clock()
        # Logging Pheromone grid
        # if log_time_cur - self.log_timer >= 2:
        #     self.log_file = open("phero_value.txt", "a+")
        #     np.savetxt(self.log_file, self.pheromone.grid, delimiter=',')
        #     self.log_file.close()

        # Save the pheromone when robot return home.
        # distance_to_origin = sqrt(x**2+y**2)
        # if self.is_saved is False and distance_to_origin < 0.05:
        #     #self.pheromone.save("foraging")
        #     self.is_saved = True
        #     self.is_phero_inj = False

        # Load the pheromone
        # 1. When use continuous contoller. (1) It hasn't previously loaded, (2) pheromone injection is disabled, 
        #    (3) service is requested by continuous controller script
        # if self.is_loaded is False and self.is_phero_inj is False and self.is_service_requested is True:
        #     try:
        #         self.pheromone.load("foraging")
        #         self.is_loaded = True
        #     except IOError as io:
        #         print("No pheromone to load: %s"%io)
        
        # 2. When reset is requested. 
        if self.is_reset == True:
            try:
                self.pheromone.load("foraging") # you can load any types of pheromone grid
                self.is_reset = False           # Reset the flag for next use
            except IOError as io:
                print("No pheromone to load: %s"%io)
                
    def injAssign(self, req):
        '''
        Service Function that takes whether robot injects pheromone or not. 
        '''
        self.is_phero_inj = req.is_inj
        service_ok = True
        self.is_service_requested = True
        return PheroInjResponse(service_ok)

    def serviceReset(self, req):
        '''
        When request is received, pheromone grid is reset and load the prepared pheromone grid.
        '''
        self.is_reset = req.reset
        is_reset = True
        return PheroResetResponse(is_reset)

    def nextGoal(self, req):

        # Turn off pheromone injection
        self.is_phero_inj = False
        
        # Convert the robot position to index for the pheromone matrix
        x = req.x
        y = req.y
        x_index, y_index = self.posToIndex(x, y)
        
        # read the 9 nearby values and choose the cell with maximum value
        max_phero = self.pheromone.getPhero(x_index, y_index)
        phero_index = np.array([0,0])
        phero_value = np.zeros((3,3))
        rand_index = 0

        ## Get the indices that contains maximum pheromone
        for i in range(3):
            for j in range(3):
                if self.pheromone.getPhero(x_index+i-1, y_index+j-1) > max_phero: # TODO: Randomly select the cell if the values are equal
                    phero_index = np.array([i-1, j-1])
                    max_phero = self.pheromone.getPhero(x_index+i-1, y_index+j-1)
                    print("Set new max")
                elif self.pheromone.getPhero(x_index+i-1, y_index+j-1) == max_phero:
                    phero_index = np.vstack((phero_index, [i-1, j-1]))
                phero_value[i,j] = self.pheromone.getPhero(x_index+i-1, y_index+j-1)
                print("At the point ({}, {}), the pheromone value is {}".format(i,j,self.pheromone.getPhero(x_index+i-1, y_index+j-1)))
                    #print("Append phero val")
        print("Phero_index: {}".format(phero_index))
        print("Phero_value: {}".format(phero_value))

        # Choose the index as a next goal from the array
        ## Check the front cells (highest priority)
        # if self.theta > (7/4) * pi or self.theta < (1/4) * pi:
        #     for x in phero_index:
        #         if x is np.array([1,0]) or x is np.array([1,1]) or x is np.array([1,2]):
        #             np.append(final_index, x, axis=0)
        rand_index = np.random.choice(phero_index.shape[0], 1)
        final_index = phero_index[rand_index]
        next_x_index = x_index + final_index[0,0]
        next_y_index = y_index + final_index[0,1]

        # Reconvert index values into position. 
        next_x, next_y = self.indexToPos(next_x_index, next_y_index)
        print("Pheromone value of goal: {}".format(self.pheromone.getPhero(next_x_index, next_y_index)))
        
        return PheroGoalResponse(next_x, next_y) 

    


class Pheromone():

    def __init__(self):
        self.resolution = 10 # grid cell size = 1 m / resolution
        self.size = 10 # m
        self.num_cell = self.resolution * self.size + 1
        if self.num_cell % 2 == 0:
            raise Exception("Number of cell is even. It needs to be an odd number")
        self.grid = np.zeros((self.num_cell, self.num_cell))
        self.grid_copy = np.zeros((self.num_cell, self.num_cell))
        self.evaporation = 60 # elapsed seconds for pheromone to be halved

        # Timers
        self.update_timer = time.clock()
        self.step_timer = time.clock()
        self.injection_timer = time.clock()

    def getPhero(self, x, y):
        return self.grid[x, y]

    def setPhero(self, x, y, value):
        self.grid[x, y] = value

    # Inject pheromone at the robot position and nearby cells in square. Size must be an odd number. 
    def injection(self, x, y, value, size, max):
        if size % 2 == 0:
            raise Exception("Pheromone injection size must be an odd number.")
        time_cur = time.clock()
        if time_cur-self.injection_timer > 0.1:
            for i in range(size):
                for j in range(size):
                    self.grid[x-(size-1)/2+i, y-(size-1)/2+j] += value
                    if self.grid[x-(size-1)/2+i, y-(size-1)/2+j] >= max:
                        self.grid[x-(size-1)/2+i, y-(size-1)/2+j] = max
            self.injection_timer = time_cur
    
    # Update all the pheromone values depends on natural phenomena, e.g. evaporation
    def update(self, min, max):
        time_cur = time.clock()
        time_elapsed = time_cur - self.update_timer
        self.update_timer = time_cur
    
        # Diffusion 
        for i in range(self.num_cell):
            for j in range(self.num_cell):
                self.grid_copy[i, j] += 0.9*self.grid[i, j]
                if i >= 1: self.grid_copy[i-1, j] += 0.025*self.grid[i, j]
                if j >= 1: self.grid_copy[i, j-1] += 0.025*self.grid[i, j]
                if i < self.num_cell-1: self.grid_copy[i+1, j] += 0.025*self.grid[i, j]
                if j < self.num_cell-1: self.grid_copy[i, j+1] += 0.025*self.grid[i, j]
        #self.grid_copy = np.clip(self.grid_copy, a_min = min, a_max = max) 
        self.grid = np.copy(self.grid_copy)
        self.grid_copy = np.zeros((self.num_cell, self.num_cell))
        # evaporation
        decay = 2**(-time_elapsed/self.evaporation)
        for i in range(self.num_cell):
            for j in range(self.num_cell):
                self.grid[i, j] = decay * self.grid[i, j]

    def save(self, file_name):
        # dir_name = os.path.dirname('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/tmp/{}.npy'.format(file_name))
        # if not os.path.exists(dir_name):
        #     os.makedirs(dir_name)
        with open('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/tmp/{}.npy'.format(file_name), 'wb') as f:
            np.save(f, self.grid)
        print("The pheromone matrix {} is successfully saved".format(file_name))

    def load(self, file_name):
        with open('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/tmp/{}.npy'.format(file_name), 'rb') as f:
            self.grid = np.load(f)
        #os.remove('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/tmp/{}.npy'.format(file_name))
        print("The pheromone matrix {} is successfully loaded".format(file_name))

    
if __name__ == "__main__":
    rospy.init_node('pheromone')
    Phero1 = Pheromone()
    node1 = Node(Phero1)
    rospy.spin()
