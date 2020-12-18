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

    def __init__(self):

        # Assigning num_robots
        self.num_robots = 4
        print("num_robots: {}".format(self.num_robots))

        # Pheromone initialization
        self.pheromone = [None]*self.num_robots
        for i in range(self.num_robots):
            self.pheromone[i] = Pheromone('dynamic {}'.format(i), size = 12, res = 10, evaporation = 0.2, diffusion = 0)
            self.pheromone[i].isDiffusion = False
        #self.pheromone.append(Pheromone('static', 180, 0))

        self.phero_max = 1.0
        self.phero_min = 0.0
        self.is_phero_inj = True

        # Publisher & Subscribers
        self.pub_phero = rospy.Publisher('/phero_value', fma, queue_size=10)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pheroCallback)
        #self.sub_inj = rospy.Subscriber('/phero_inj', Bool, self.injCallback)
        
        # Services
        self.srv_inj = rospy.Service('phero_inj', PheroInj, self.injAssign) # Used in continuous controller 
        #self.srv_goal = rospy.Service('phero_goal', PheroGoal, self.nextGoal) # Used in phero_controller (discrete action)
        self.srv_reset = rospy.Service('phero_reset', PheroReset, self.serviceReset)
        #self.srv_read = rospy.Service('phero_read', PheroRead, self.serviceRead)
        self.is_service_requested = False
        self.theta = 0 
        
        self.log_timer = time.clock()
        #self.log_file = open("phero_value.txt", "a+")
        self.is_saved = False
        self.is_loaded = False
        self.is_reset = True # False for reset

        for i in range(self.num_robots):
            self.pheromone[i].isDiffusion = False
            self.pheromone[i].isEvaporation = True
        self.startTime = time.time()

        # Robot positions
        self.positions = [0.0, 0.0]*self.num_robots
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

        print("Initialisation completed")

    def posToIndex(self, x, y):
        phero = self.pheromone
        x_tmp = x
        y_tmp = y
        # Read pheromone value at the robot position
        x_index = [0]*self.num_robots
        y_index = [0]*self.num_robots
        for i in range(self.num_robots):
            res = phero[0].resolution # 20201209 Every pheromone must share same grid size
            round_dp = int(log10(res))
            x_tmp[i] = round(x_tmp[i], round_dp) # round the position value so that they fit into the centre of the cell.
            y_tmp[i] = round(y_tmp[i], round_dp) # e.g. 0.13 -> 0.1
            x_tmp[i] = int(x_tmp[i]*res)
            y_tmp[i] = int(y_tmp[i]*res)
        
            # Position conversion from Robot into pheromone matrix (0, 0) -> (n+1, n+1) of 2n+1 matrix
            x_index[i] = int(x_tmp[i] + (phero[i].num_cell-1)/2)
            y_index[i] = int(y_tmp[i] + (phero[i].num_cell-1)/2)
            if x_index[i] < 0 or y_index[i] < 0 or x_index[i] > phero[i].num_cell-1 or y_index[i] > phero[i].num_cell-1:
                raise Exception("The pheromone matrix index is out of range.")
        return x_index, y_index

    def indexToPos(self, x_index, y_index):
        phero = self.pheromone[0]
        x = x_index - (phero.num_cell-1)/2
        y = y_index - (phero.num_cell-1)/2

        x = float(x) / phero.resolution 
        
    def pheroCallback(self, message):
        
        #tb3 = [None]*self.num_robots

        for i in range(len(message.name)):
            if message.name[i] == 'tb3_0':
                tb3_0 = i
            if message.name[i] == 'tb3_1':
                tb3_1 = i
            if message.name[i] == 'tb3_2':
                tb3_2 = i
            if message.name[i] == 'tb3_3':
                tb3_3 = i

        # Reading from arguments
        pos = [message.pose[tb3_0].position, message.pose[tb3_1].position, message.pose[tb3_2].position, message.pose[tb3_3].position]
        # twist = message.twist[-[]
        # ori = pose.orientation
        #print("pos0x: {}".format(pos[0].x))
        phero = self.pheromone
        print("pos: {}".format(pos))
        x = [p.x for p in pos]
        y = [p.y for p in pos]
        # for i in range(self.num_robots):
        #     x.append(pos[i][0])
        #     y.append(pos[i][1])
        #print("x, y: {}, {}".format(x, y))
        x_idx, y_idx = self.posToIndex(x, y)
        
        #print("x_idx, y_idx: {}, {}".format(x_idx, y_idx))
        # ========================================================================= #
	    #                           Pheromone Reading                               #
	    # ========================================================================= #

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
        
        ''' Read Pheromone from each pheromone grid '''
        # each robot reads (1) static pheromone value and (2) the sum of dynamic pheromone of other robots
        phero_arr = [Float32MultiArray()]*self.num_robots
        phero_val = [None] * self.num_robots
        #phero_arr = np.array( )

        # Dynamic pheromone
        for n in range(self.num_robots):
            phero_val[n] = list()     
            for i in range(3):
                for j in range(3):
                    phero_sum = sum([phero_dy.getPhero(x_idx[n]+i-1, y_idx[n]+j-1) for phero_dy in self.pheromone if ('dynamic' in phero_dy.name) and ("%d"%n not in phero_dy.name)])
                    phero_name = []
                    clipped_sum = min(self.phero_max, max(phero_sum, self.phero_min))
                    phero_val[n].append(phero_sum) # Read the sum of other robots pheromone
            phero_arr[n].data = phero_val[n]
                    # Check if it works well
                    # 20201215 clipping must be added
        print("phero_val: {}".format(phero_val))
                    
                    

        # Static pheromone
        # for n in range(self.num_robots):
        #     for i in range(3):
        #         for j in range(3):
        #             static_phero = phero_st for phero_st in self.pheromone if 'static' in phero_st.name
        #             phero_val[n] += static_phero.getPhero(x_index+i-1, y_index+j-1)
        #             print("Phero_val[n]: {} - static".format(phero_val[n])
        #             phero_val[n] = min(self.phero_max, max(self.phero_min, phero_val[n]))
        #             print("Phero_val[n]: {} - clipped".format(phero_val[n]))

        # Clipping pheromone value

        # Return pheromone value to each robot
        #self.sock_receiver.return_to_client(np.asarray(phero_val))
        self.pub_phero.publish(phero_arr)

        # ========================================================================= #
	    #                           Pheromone Injection                             #
	    # ========================================================================= #

        # Pheromone injection (uncomment it when injection is needed)
        ## Two robots inject pheromone in different grids
        if self.is_phero_inj is True:
            for i, phero_dy in enumerate(self.pheromone):
                if '%d'%i in phero_dy.name:
                    phero_dy.gradInjection(x_idx[i], y_idx[i], 1, 0.3, 1.0, self.phero_max)

        # ========================================================================= #
	    #                           Pheromone Update                                #
	    # ========================================================================= #

        # Update pheromone matrix in every 0.1s - only dynamic pheromone
        time_cur = time.clock()
        if time_cur-phero[0].step_timer >= 0.1: 
            for phero_dy in self.pheromone:
                if "dynamic" in phero_dy.name:
                    phero_dy.update(self.phero_min, self.phero_max)
                    phero_dy.step_timer = time_cur

        #log_time_cur = time.clock()
        # Logging Pheromone grid
        # if log_time_cur - self.log_timer >= 2:
        #     self.log_file = open("phero_value.txt", "a+")
        #     np.savetxt(self.log_file, self.pheromone.grid, delimiter=',')
        #     self.log_file.close()

        # ========================================================================= #
	    #                           Save Pheromone                                  #
	    # ========================================================================= #
        
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
                for i in range(self.num_robots):  # Reset the pheromone grid
                    self.pheromone[i].reset()
                    if 'static' in self.pheromone[i].name:
                        self.pheromone.load("simple_collision_diffused3") # you can load any types of pheromone grid
                print("Pheromone grid reset!")
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
    4. Inject Pheromone at the specified position (Plain + Grad)
    5. Circle (Plain + Grad) 
    6. Update Pheromone (Evaporation & Diffusion)
    7. Save Pheormone Grid
    8. Load Pheromone Grid
    '''

    def __init__(self, name, size = 10, res = 10, evaporation = 0.0, diffusion = 0.0):
        self.name = name
        self.resolution = res # grid cell size = 1 m / resolution
        self.size = size # m
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

    def gradInjection(self, x, y, value, min_val, rad, maxp):

        time_cur = time.clock()
        if time_cur-self.injection_timer > 0.1:
            radius = int(rad*self.resolution)
            #print("Radius: {}".format(radius))
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    if sqrt(i**2+j**2) <= radius and (x+i < self.num_cell-1 and x+i >= 0) and (y+i < self.num_cell-1 and y+i >= 0):
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

    def gradCircle(self, x, y, value, radius):
        radius = int(radius*self.resolution)
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if sqrt(i**2+j**2) <= radius:
                    self.grid[x+i, y+j] = value/(exp(sqrt(i**2+j**2))/10)

    
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
            self.grid = np.copy(self.grid_copy)
            self.grid_copy = np.zeros((self.num_cell, self.num_cell))
        if self.isEvaporation == True:
            # Evaporation
            decay = 2**(-time_elapsed/self.evaporation)
            for i in range(self.num_cell):
                for j in range(self.num_cell):
                    self.grid[i, j] = decay * self.grid[i, j]

    def reset(self):
        self.grid = np.zeros((self.num_cell, self.num_cell))

    def save(self, file_name):
        '''
        Save the current matrix as a numpy object
        '''
        with open('/home/swn/catkin_ws/src/Turtlebot3_Pheromone/tmp/{}.npy'.format(file_name), 'wb') as f:
            np.save(f, self.grid)
        print("The pheromone matrix {} is successfully saved".format(file_name))

    def load(self, file_name):
        '''
        Load the previously saved pheromone matrix 
        '''
        with open('/home/swn/catkin_ws/src/Turtlebot3_Pheormone/tmp/{}.npy'.format(file_name), 'rb') as f:
            self.grid = np.load(f)
        print("The pheromone matrix {} is successfully loaded".format(file_name))

    
if __name__ == "__main__":
    rospy.init_node('pheromone')
    node1 = Node()
    rospy.spin()
        

