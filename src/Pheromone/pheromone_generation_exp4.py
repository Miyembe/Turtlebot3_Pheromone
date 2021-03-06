#!/usr/bin/env python
# Node that handles pheromone layer
# Subscriber - Robot (x, y) position
# Publisher - Pheromone value at (x, y)
import sys
# sys.path.append('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation')
# import roslib; roslib.load_manifest('turtlebot3_waypoint_navigation')
import os
import numpy as np
#import tf
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

    def __init__(self):
        self.phero_max = 1.0
        self.phero_min = 0.0
        self.is_phero_inj = True

        # Pheromone initialization
        phero_static = Pheromone('static', size = 12, res = 20, evaporation = 180, diffusion = 0)
        phero_static.isDiffusion = True
        phero_static.isEvaporation = False
        self.pheromone = phero_static

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
        self.is_reset = True # False for reset

        self.pheromone.isDiffusion = True
        self.pheromone.isEvaporation = False
        self.startTime = time.time()

        # Misc
        self.num_objects = 0

        

    def posToIndex(self, x, y):
        '''
        Convert 2D coordinates (x, y) into the matrix index (x_index, y_index) 
        '''
        phero = self.pheromone
        # print("x: {}".format(x))
        # print("y: {}".format(y))
        x_tmp = x
        y_tmp = y
        # Read pheromone value at the robot position
        x_index = [0]*len(x)
        y_index = [0]*len(x)
        for i in range(len(x)):
            res = phero.resolution
            round_dp = int(log10(res))
            x_tmp[i] = round(x_tmp[i], round_dp) # round the position value so that they fit into the centre of the cell.
            y_tmp[i] = round(y_tmp[i], round_dp) # e.g. 0.13 -> 0.1
            x_tmp[i] = int(x_tmp[i]*res)
            y_tmp[i] = int(y_tmp[i]*res)
        
            # Position conversion from Robot into pheromone matrix (0, 0) -> (n+1, n+1) of 2n+1 matrix
            x_index[i] = int(x_tmp[i] + (phero.num_cell-1)/2)
            y_index[i] = int(y_tmp[i] + (phero.num_cell-1)/2)
            if x_index[i] < 0 or y_index[i] < 0 or x_index[i] > phero.num_cell-1 or y_index[i] > phero.num_cell-1:
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
        # pose = message.pose[-1]
        # twist = message.twist[-1]
        # pos = pose.position
        # ori = pose.orientation
        # phero = cargs
        # x = pos.x
        # y = pos.y
        
        cylinders = []
        boxes = []
        for i in range(len(message.name)):
            if "cylinder" in message.name[i]:
                cylinders.append(i)
            if "box" in message.name[i]:
                boxes.append(i)

        self.num_objects = len(cylinders) + len(boxes)
                


        # Reading from arguments
        pos_cyls = [message.pose[cylinder].position for cylinder in cylinders]
        pos_boxes = [message.pose[box].position for box in boxes]
        
        # twist = message.twist[-[]
        # ori = pose.orientation
        #print("pos0x: {}".format(pos[0].x))
        phero = self.pheromone
        #print("pos: {}".format(pos))
        x_cyl = [pos_cyl.x for pos_cyl in pos_cyls]
        y_cyl = [pos_cyl.y for pos_cyl in pos_cyls]
        x_box = [pos_box.x for pos_box in pos_boxes]
        y_box = [pos_box.y for pos_box in pos_boxes]
        # for i in range(self.num_robots):
        #     x.append(pos[i][0])
        #     y.append(pos[i][1])
        #print("x, y: {}, {}".format(x, y))
        x_idx_cyl, y_idx_cyl = self.posToIndex(x_cyl, y_cyl)
        x_idx_box, y_idx_box = self.posToIndex(x_box, y_box)
        
        # angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        # if angles[2] < 0:
        #     self.theta = angles[2] + 2*pi
        # else: self.theta = angles[2]

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
        #x_index, y_index = self.posToIndex(x, y)
        #phero_val = Float32MultiArray()
        #phero_arr = np.array( )
        # for i in range(3):
        #     for j in range(3):
        #         phero_val.data.append(self.pheromone.getPhero(x_index+i-1, y_index+j-1))
        # #print("phero_avg: {}".format(np.average(np.asarray(phero_val.data))))
        # self.pub_phero.publish(phero_val)
        # # Assign pheromone value and publish it
        # phero_val = phero.getPhero(x_index, y_index)
        # self.pub_phero.publish(phero_val)

        # ========================================================================= #
	    #                           Pheromone Injection                             #
	    # ========================================================================= #
        
        ''' Pheromone injection (uncomment it when injection is needed) '''
        
        time_cur_injection = time.time()
        if self.is_phero_inj is True and time_cur_injection - phero.injection_timer >= 0.1:
            for i in range(len(pos_cyls)):
                phero.gradInjection(x_idx_cyl[i], y_idx_cyl[i], 1.2, 0.3, 0.6, self.phero_max)
            for i in range(len(pos_boxes)):
                phero.sqInjection(x_idx_box[i], y_idx_box[i], 0.1, 0.8, self.phero_max)
            phero.injection_timer = time_cur_injection

        
        # ========================================================================= #
	    #                           Pheromone Update                                #
	    # ========================================================================= #
        
        ''' Pheromone Update '''
        start_time = time.time()
        time_cur_update = time.clock()
        if time_cur_update-phero.step_timer >= 0.1: 
            phero.update(self.phero_min, self.phero_max)
            print("Ay")
            #print("time: {}".format(time_cur_update-phero.step_timer))
            phero.step_timer = time_cur_update
        end_time = time.time()
        # ========================================================================= #
	    #                           Save Pheromone                                  #
	    # ========================================================================= #
        
        '''Saving pheromone'''
        # # Save after 20s
        
        time_check = time.time()
        if time_check - self.startTime >= 20 and self.is_saved is False:
            self.pheromone.save("tcds_exp4")
            self.is_saved = True
        
        

        #print("update time: {}".format(end_time - start_time))
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
        # if self.is_reset == True:
            # try:
            #     self.pheromone.load("simple_collision_diffused3") # you can load any types of pheromone grid
            #     self.is_reset = False           # Reset the flag for next use
            # except IOError as io:
            #     print("No pheromone to load: %s"%io)
                
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

    def __init__(self, name, size = 10, res = 100, evaporation = 0.0, diffusion = 0.0):
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
    def injection(self, x, y, value, radius, max):
        # if size % 2 == 0:
        #     raise Exception("Pheromone injection size must be an odd number.")
        
        radius = int(radius*self.resolution)
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if sqrt(i**2+j**2) <= radius:
                    self.grid[x+i, y+j] += value
                if self.grid[x+i, y+j] >= max:
                    self.grid[x+i, y+j] = max

    def gradInjection(self, x, y, value, min_val, rad, maxp):
        radius = int(rad*self.resolution)
        #print("Radius: {}".format(radius))
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if sqrt(i**2+j**2) <= radius and (x+i < self.num_cell-1 and x+i >= 0) and (y+i < self.num_cell-1 and y+i >= 0):
                    self.grid[x+i, y+j] += value - value*(sqrt(i**2+j**2))/radius + min_val
                    if self.grid[x+i, y+j] >= maxp:
                        self.grid[x+i, y+j] = maxp

    # Inject pheromone in square shape
    def sqInjection(self, x, y, value, length, max):
        length = int(length/2*self.resolution)
        for i in range(-length, length):
            for j in range(-length, length):
                self.grid[x+i, y+j] += value
                if self.grid[x+i, y+j] >= max:
                    self.grid[x+i, y+j] = max

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
                    self.grid_copy[i, j] += 0.6*self.grid[i, j]
                    if i >= 1: self.grid_copy[i-1, j] += 0.1*self.grid[i, j]
                    if j >= 1: self.grid_copy[i, j-1] += 0.1*self.grid[i, j]
                    if i < self.num_cell-1: self.grid_copy[i+1, j] += 0.1*self.grid[i, j]
                    if j < self.num_cell-1: self.grid_copy[i, j+1] += 0.1*self.grid[i, j]
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
        with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/tmp/{}.npy'.format(file_name), 'wb') as f:
            np.save(f, self.grid)
        print("The pheromone matrix {} is successfully saved".format(file_name))

    def load(self, file_name):
        with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/tmp/{}.npy'.format(file_name), 'rb') as f:
            self.grid = np.load(f)
        print("The pheromone matrix {} is successfully loaded".format(file_name))

    
if __name__ == "__main__":
    rospy.init_node('pheromone')
    node1 = Node()
    rospy.spin()
