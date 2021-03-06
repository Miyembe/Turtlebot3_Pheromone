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
from turtlebot3_waypoint_navigation.srv import PheroRead, PheroReadResponse
from turtlebot3_waypoint_navigation.msg import fma

class Node():

    def __init__(self, phero):
        self.num_robots = 1
        self.pheromone = [None] * self.num_robots
        for i in range(len(phero)):
            self.pheromone[i] = phero[i]
        self.phero_max = 1.0
        self.phero_min = 0.0
        self.is_phero_inj = True

        # Publisher & Subscribers
        self.pub_phero = rospy.Publisher('/phero_value', fma, queue_size=10)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pheroCallback, self.pheromone)
        #self.sub_inj = rospy.Subscriber('/phero_inj', Bool, self.injCallback)
        
        # Services
        self.srv_inj = rospy.Service('phero_inj', PheroInj, self.injAssign) # Used in continuous controller 
        self.srv_goal = rospy.Service('phero_goal', PheroGoal, self.nextGoal) # Used in phero_controller (discrete action)
        self.srv_reset = rospy.Service('phero_reset', PheroReset, self.serviceReset)
        #self.srv_read = rospy.Service('phero_read', PheroRead, self.serviceRead)
        self.is_service_requested = False
        self.theta = 0 
        
        self.log_timer = time.clock()
        self.log_file = open("phero_value.txt", "a+")
        self.is_saved = False
        self.is_loaded = False
        self.is_reset = True # False for reset

        for i in range(len(phero)):
            self.pheromone[i].isDiffusion = False
            self.pheromone[i].isEvaporation = True
        self.startTime = time.time()

        # Robot positions
        self.x = [0.0, 4.0]
        self.y = [0.0, 0.0]
        self.x_idx = [0, 0]
        self.y_idx = [0, 0]

        # PosToIndex for pheromone circle
        # x_2, y_0 = self.posToIndex(2,0)
        # x_n3, y_n3 = self.posToIndex(-3,-3)
        # x_0, y_0 = self.posToIndex(0,0)
        # Pheromone Initilaisation
        # self.pheromone.circle(x_2, y_0, 1, 1)
        # self.pheromone.circle(x_3, y_0, 1, 1)
        # self.pheromone.circle(x_n3, y_0, 1, 1)
        # self.pheromone.circle(x_0, y_3, 1, 1)
        # self.pheromone.circle(x_0,y_n3, 1, 1)

        

    def posToIndex(self, x, y):
        phero = self.pheromone
        # Read pheromone value at the robot position
        x_index = [0]*len(phero)
        y_index = [0]*len(phero)
        for i in range(len(phero)):
            res = phero[i].resolution
            round_dp = int(log10(res))
            x[i] = round(x[i], round_dp) # round the position value so that they fit into the centre of the cell.
            y[i] = round(y[i], round_dp) # e.g. 0.13 -> 0.1
            x[i] = int(x[i]*res)
            y[i] = int(y[i]*res)
        
            # Position conversion from Robot into pheromone matrix (0, 0) -> (n+1, n+1) of 2n+1 matrix
            x_index[i] = x[i] + (phero[i].num_cell-1)/2
            y_index[i] = y[i] + (phero[i].num_cell-1)/2
            if x_index[i] < 0 or y_index[i] < 0 or x_index[i] > phero[i].num_cell-1 or y_index[i] > phero[i].num_cell-1:
                raise Exception("The pheromone matrix index is out of range.")
        return x_index, y_index

    def indexToPos(self, x_index, y_index):
        phero = self.pheromone[0]
        x = x_index - (phero.num_cell-1)/2
        y = y_index - (phero.num_cell-1)/2

        x = float(x) / phero.resolution
        y = float(y) / phero.resolution

        return x, y

    def pheroCallback(self, message, cargs):
        
        # Reading from arguments
        pos = [message.pose[-1].position, message.pose[-2].position] 
        # twist = message.twist[-[]
        # ori = pose.orientation
        phero = cargs
        x = [pos[0].x, pos[1].x]
        y = [pos[0].y, pos[1].y]
        x_idx, y_idx = self.posToIndex(x, y)
        # x = pos.x
        # y = pos.y

        # angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        # if angles[2] < 0:
        #     self.theta = angles[2] + 2*pi
        # else: self.theta = angles[2]

        '''
        Pheromone Value Reading
        '''

        '''2 pheromone values'''
        # Add two wheel position
        # wheel_distance = 0.2
        # pos_l = np.array([x+(cos(pi/2)*cos(self.theta)*(wheel_distance/2) - sin(pi/2)*sin(self.theta)*(wheel_distance/2)), y+(sin(pi/2)*cos(self.theta)*(wheel_distance/2) + cos(pi/2)*sin(self.theta)*(wheel_distance/2))])
        # pos_r = np.array([x+(cos(pi/2)*cos(self.theta)*(wheel_distance/2) + sin(pi/2)*sin(self.theta)*(wheel_distance/2)), y+(-sin(pi/2)*cos(self.theta)*(wheel_distance/2) + cos(pi/2)*sin(self.theta)*(wheel_distance/2))])
        
        # x_index, y_index = self.posToIndex(pos.x, pos.y)
        # x_l_index, y_l_index = self.posToIndex(pos_l[0], pos_l[1])
        # x_r_index, y_r_index = self.posToIndex(pos_r[0], pos_r[1]) 

        # # Assign pheromone values from two positions and publish it
        # phero_val = Float32MultiArray()
        # phero_val.data = [phero.getPhero(x_l_index, y_l_index), phero.getPhero(x_r_index, y_r_index)]
        # self.pub_phero.publish(phero_val)

        '''9 pheromone values'''
        # Position of 9 cells surrounding the robot
        # x_index, y_index = self.posToIndex(x, y)
        # phero_val = Float32MultiArray()
        # #phero_arr = np.array( )
        # for i in range(3):
        #     for j in range(3):
        #         phero_val.data.append(self.pheromone[0].getPhero(x_index+i-1, y_index+j-1)) # TODO: Randomly select the cell if the values are equal
        #print("phero_avg: {}".format(np.average(np.asarray(phero_val.data))))
        # self.pub_phero.publish(phero_val)
        # # Assign pheromone value and publish it
        # phero_val = phero.getPhero(x_index, y_index)
        # self.pub_phero.publish(phero_val)
        
        '''Set of pheromone values'''
        # 9 pheromone value for two robots read from the other's pheromone grid. 
        phero_arr = [Float32MultiArray()]*self.num_robots
        phero_val = [None] * self.num_robots
        #phero_arr = np.array( )
        for n in range(self.num_robots):
            phero_val[n] = list()     
            for i in range(3):
                for j in range(3):
                    phero_val[n].append(self.pheromone[n].getPhero(x_idx[n]+i-1, y_idx[n]+j-1)) # Read the other's pheromone
            phero_arr[n].data = phero_val[n]
        self.pub_phero.publish(phero_arr)
        # Pheromone injection (uncomment it when injection is needed)
        ## Two robots inject pheromone in different grids
        # if self.is_phero_inj is True:
        #     for i in range(len(self.pheromone)):
        #         phero[i].injection(x_idx[i], y_idx[i], 1, 13, self.phero_max)


        # Update pheromone matrix in every 0.1s
        time_cur = time.clock()
        if time_cur-phero[0].step_timer >= 0.1: 
            phero[0].update(self.phero_min, self.phero_max)
            phero[0].step_timer = time_cur

        #log_time_cur = time.clock()
        # Logging Pheromone grid
        # if log_time_cur - self.log_timer >= 2:
        #     self.log_file = open("phero_value.txt", "a+")
        #     np.savetxt(self.log_file, self.pheromone.grid, delimiter=',')
        #     self.log_file.close()

        
        '''Saving pheromone'''
        # # Save after 20s
        # time_check = time.time()
        # if time_check - self.startTime >= 20 and self.is_saved is False:
        #     self.pheromone.save("simple_collision_diffused")
        #     self.is_saved = True
        
        # Save the pheromone when robot return home.
        # distance_to_origin = sqrt(x**2+y**2)
        # if self.is_saved is False and distance_to_origin < 0.05:
        #     self.pheromone.save("foraging_static")
        #     self.is_saved = True
        #     self.is_phero_inj = False
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
                for i in range(self.num_robots):  # Reset the pheromone grid
                    self.pheromone[i].reset()
                #self.pheromone.load("simple_collision_diffused") # you can load any types of pheromone grid
                print("Pheromone grid reset!")
                self.is_reset = False           # Reset the flag for next use
            except IOError as io:
                print("No pheromone to load: %s"%io)

    def serviceRead(self, req):
        '''
        Receive x, y positions and returns pheromone values
        '''
        x_values = req.x
        y_values = req.y
        x_indices = [None]*len(x_values)
        y_indices = [None]*len(x_values)
        for i in range(len(x_values)):
            x_indices[i], y_indices[i] = self.posToIndex(x_values[i], y_values[i])

        phero_arr = [Float32MultiArray()]*self.num_robots
        phero_val = [None] * self.num_robots
        #phero_arr = np.array( )
        for n in range(len(x_values)):
            phero_val[n] = list()     
            for i in range(3):
                for j in range(3):
                    phero_val[n].append(self.pheromone[1-n].getPhero(x_indices[n]+i-1, y_indices[n]+j-1)) # Read the other's pheromone
            phero_arr[n].data = phero_val[n]
        # self.x = x_values
        # self.y = y_values
        # self.x_idx = x_indices
        # self.y_idx = y_indices
        return PheroReadResponse(phero_arr)

                
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

    def __init__(self, name, evaporation, diffusion):
        self.name = name
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
    def injection(self, x, y, value, size, maxp):
        if size % 2 == 0:
            raise Exception("Pheromone injection size must be an odd number.")
        time_cur = time.clock()
        if time_cur-self.injection_timer > 0.1:
            for i in range(size):
                for j in range(size):
                    self.grid[x-(size-1)/2+i, y-(size-1)/2+j] += value
                    if self.grid[x-(size-1)/2+i, y-(size-1)/2+j] >= maxp:
                        self.grid[x-(size-1)/2+i, y-(size-1)/2+j] = maxp
            self.injection_timer = time_cur

    def circle(self, x, y, value, radius):
        radius = radius*self.resolution
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

    def reset(self):
        self.grid = np.zeros((self.num_cell, self.num_cell))

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
    Phero1 = Pheromone('1', 0.5, 0)
    Phero2 = Pheromone('2', 0.5, 0)
    Phero = [Phero1]
    node1 = Node(Phero)
    rospy.spin()
