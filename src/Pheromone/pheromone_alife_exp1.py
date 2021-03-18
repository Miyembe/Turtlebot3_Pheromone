#!/usr/bin/env python
# Node that handles pheromone layer
# Subscriber - Robot (x, y) position
# Publisher - Pheromone value at (x, y)
import sys
sys.path.append('/home/sub/catkin_ws/src/Turtlebot3_Pheromone')
import roslib; roslib.load_manifest('turtlebot3_pheromone')
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
from turtlebot3_pheromone.srv import PheroGoal, PheroGoalResponse
from turtlebot3_pheromone.srv import PheroInj, PheroInjResponse
from turtlebot3_pheromone.srv import PheroReset, PheroResetResponse
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
        self.srv_reset = rospy.Service('phero_reset', PheroReset, self.serviceReset)
        self.is_service_requested = False
        self.theta = 0 
        
        self.log_timer = time.clock()
        self.log_file = open("phero_value.txt", "a+")
        self.is_saved = False
        self.is_loaded = False
        self.is_reset = True # False for reset

        self.pheromone.isDiffusion = True
        self.pheromone.isEvaporation = False
        self.startTime = time.time()

        

    def posToIndex(self, x, y):
        '''
        Convert 2D coordinates (x, y) into the pheromone matrix index (x_index, y_index) 
        '''
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
        '''
        Convert matrix indices into 2D coordinate (x, y)
        '''
        
        x = x_index - (self.pheromone.num_cell-1)/2
        y = y_index - (self.pheromone.num_cell-1)/2

        x = float(x) / self.pheromone.resolution
        y = float(y) / self.pheromone.resolution

        return x, y

    def pheroCallback(self, message, cargs):
        
        # Reading from arguments
        pose = message.pose[-2]
        twist = message.twist[-2]
        pos = pose.position
        ori = pose.orientation
        phero = cargs
        x = pos.x
        y = pos.y
        x_idx, y_idx = self.posToIndex(x, y)

        angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        if angles[2] < 0:
            self.theta = angles[2] + 2*pi
        else: self.theta = angles[2]

        # ========================================================================= #
	    #                           Pheromone Reading                               #
	    # ========================================================================= #


        '''
        Pheromone Value Reading
        '''

        ''' 2 values from the antennae'''
        # Define antennae parameters
        theta = self.theta
        a_angle = 0.5
        a_len = 0.45

        # Get pheromone values and define antennae tips positions
        phero_val = Float32MultiArray()
        antennae_pos_l = [cos(theta)*cos(a_angle)*a_len-sin(theta)*sin(a_angle)*a_len + pos.x, sin(theta)*cos(a_angle)*a_len+cos(theta)*sin(a_angle)*a_len + pos.y] # [cos(0.3)*a_len, sin(0.3)*a_len]
        antennae_pos_r = [cos(theta)*cos(-a_angle)*a_len-sin(theta)*sin(-a_angle)*a_len + pos.x, sin(theta)*cos(-a_angle)*a_len+cos(theta)*sin(-a_angle)*a_len + pos.y]
        a_l_index_x, a_l_index_y = self.posToIndex(antennae_pos_l[0], antennae_pos_l[1])
        a_r_index_x, a_r_index_y = self.posToIndex(antennae_pos_r[0], antennae_pos_r[1])

        # Get the pheromone values at the position of antennae and publish
        phero_val.data.append(self.pheromone.getPhero(a_l_index_x, a_l_index_y))
        phero_val.data.append(self.pheromone.getPhero(a_r_index_x, a_r_index_y))
        self.pub_phero.publish(phero_val)


        '''9 pheromone values'''
        # # Position of 9 cells surrounding the robot
        # x_index, y_index = self.posToIndex(x, y)
        # phero_val = Float32MultiArray()
        # #phero_arr = np.array( )
        # for i in range(3):
        #     for j in range(3):
        #         phero_val.data.append(self.pheromone.getPhero(x_index+i-1, y_index+j-1))
        # #print("phero_avg: {}".format(np.average(np.asarray(phero_val.data))))
        # self.pub_phero.publish(phero_val)
        # # # Assign pheromone value and publish it
        # # phero_val = phero.getPhero(x_index, y_index)
        # # self.pub_phero.publish(phero_val)

        # ========================================================================= #
	    #                           Pheromone Injection                             #
	    # ========================================================================= #
        
        ''' Pheromone injection (uncomment it when injection is needed) '''
        # if self.is_phero_inj is True:
        #     phero.gradInjection(x_idx, y_idx, 1, 0.6, 0.7, self.phero_max)

        # ========================================================================= #
	    #                           Pheromone Update                                #
	    # ========================================================================= #
        
        ''' Pheromone Update '''
        # time_cur = time.clock()
        # if time_cur-phero.step_timer >= 0.1: 
        #     phero.update(self.phero_min, self.phero_max)
        #     phero.step_timer = time_cur

        # ========================================================================= #
	    #                           Save Pheromone                                  #
	    # ========================================================================= #
        
        '''Saving pheromone'''
        # # Save after 20s
        # time_check = time.time()
        # if time_check - self.startTime >= 20 and self.is_saved is False:
        #     self.pheromone.save("simple_collision_diffused3")
        #     self.is_saved = True
        #print("x, y: ({}, {})".format(x, y))
        # Save the pheromone when robot return home.
        # distance_to_origin = sqrt((x-4)**2+y**2)
        # if self.is_saved is False and distance_to_origin < 0.3:
        #     self.pheromone.save("foraging_static")
        #     self.is_saved = True
        #     self.is_phero_inj = False

        # ========================================================================= #
	    #                           Load Pheromone                                  #
	    # ========================================================================= #
        
        '''Loading Pheromone'''
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
                self.pheromone.load("foraging_static_L") # you can load any types of pheromone grid
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

class Pheromone():

    # ========================================================================= #
	#                           Pheromone Class                                 #
	# ========================================================================= #
    '''
    Pheromone Class
    1. Initialise Pheromone
    2. Get Pheromone
    3. Set Pheromone
    4. Inject Pheromone at the specified position
    5. Circle 
    6. Update Pheromone (Evaporation & Diffusion)
    7. Save Pheormone Grid
    8. Load Pheromone Grid
    '''

    def __init__(self, evaporation, diffusion):
        self.resolution = 10 # grid cell size = 1 m / resolution
        self.size = 12 # m
        self.num_cell = self.resolution * self.size + 1
        if self.num_cell % 2 == 0:
            raise Exception("Number of cell is even. It needs to be an odd number")
        self.grid = np.zeros((self.num_cell, self.num_cell))
        self.grid_copy = np.zeros((self.num_cell, self.num_cell))
        self.evaporation = evaporation # elapsed seconds for pheromone to be halved
        self.diffusion = diffusion
        self.isDiffusion = True
        self.isEvaporation = True

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

    def gradInjection(self, x, y, value, min_val, rad, maxp):
        time_cur = time.clock()
        if time_cur-self.injection_timer > 0.1:
            radius = int(rad*self.resolution)
            #print("Radius: {}".format(radius))
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    if sqrt(i**2+j**2) <= radius:
                        self.grid[x+i, y+j] = value - value*(sqrt(i**2+j**2))/radius + min_val
                        if self.grid[x+i, y+j] >= maxp:
                            self.grid[x+i, y+j] = maxp
            self.injection_timer = time_cur

    def circle(self, x, y, value, radius):
        radius = int(radius*self.resolution)
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if sqrt(i**2+j**2) <= radius:
                    self.grid[x+i, y+j] = value

    
    # Update all the pheromone values depends on natural phenomena, e.g. evaporation
    def update(self, min, max):
        time_cur = time.clock()
        time_elapsed = time_cur - self.update_timer
        self.update_timer = time_cur

        if self.isDiffusion == True:
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
        if self.isEvaporation == True:
            # evaporation
            decay = 2**(-time_elapsed/self.evaporation)
            for i in range(self.num_cell):
                for j in range(self.num_cell):
                    self.grid[i, j] = decay * self.grid[i, j]

    def save(self, file_name):
        with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/tmp/{}.npy'.format(file_name), 'wb') as f: # Please change the path to your local machine's path to the catkin workspace
            np.save(f, self.grid)
        print("The pheromone matrix {} is successfully saved".format(file_name))

    def load(self, file_name):
        with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/tmp/{}.npy'.format(file_name), 'rb') as f:
            self.grid = np.load(f)
        print("The pheromone matrix {} is successfully loaded".format(file_name))

    
if __name__ == "__main__":
    rospy.init_node('pheromone')
    Phero1 = Pheromone(180,0)
    node1 = Node(Phero1)
    rospy.spin()
