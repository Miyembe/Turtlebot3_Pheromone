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
import csv
from turtlebot3_pheromone.srv import PheroGoal, PheroGoalResponse
from turtlebot3_pheromone.srv import PheroInj, PheroInjResponse
from turtlebot3_pheromone.srv import PheroReset, PheroResetResponse
from turtlebot3_pheromone.srv import PheroRead, PheroReadResponse
from turtlebot3_pheromone.msg import fma

HOME = os.environ['HOME']

class Antennae():
    '''
    Description:
    Antennae class defining the shape of antennae and 
    define the position of end of the antennae with the given robot position
    '''
    def __init__(self):
        self.length = 0.45
        self.tilt_angle = 0.5 # radian
    def position(self, robot_pos):
        # robot_pos = [x, y, theta] : pose2D
        x = robot_pos[0]
        y = robot_pos[1]
        theta = robot_pos[2]

        # 1. calculate the position of antennae tips when the angle of the robot is 0.
        # 2. reflect the angle of the robot (theta) to the positions of the antennae
        antennae_pos_l = [cos(theta)*cos(self.tilt_angle)*self.length-sin(theta)*sin(self.tilt_angle)*self.length + x, sin(theta)*cos(self.tilt_angle)*self.length+cos(theta)*sin(self.tilt_angle)*self.length + y] # [cos(0.3)*self.length, sin(0.3)*self.length]
        antennae_pos_r = [cos(theta)*cos(-self.tilt_angle)*self.length-sin(theta)*sin(-self.tilt_angle)*self.length + x, sin(theta)*cos(-self.tilt_angle)*self.length+cos(theta)*sin(-self.tilt_angle)*self.length + y]
        antennae_pos = [antennae_pos_l, antennae_pos_r]

        return antennae_pos



class Node():

    def __init__(self, phero):
        self.num_robots = 2
        self.pheromone = [None] * self.num_robots
        for i in range(len(phero)):
            self.pheromone[i] = phero[i]
        self.phero_max = 1.0
        self.phero_min = 0.0
        self.is_phero_inj = True

        # Publisher & Subscribers
        self.pub_phero = rospy.Publisher('/phero_value', fma, queue_size=10)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pheroCallback, self.pheromone)
        
        # Services
        self.srv_inj = rospy.Service('phero_inj', PheroInj, self.injAssign) # Used in continuous controller 
        self.srv_reset = rospy.Service('phero_reset', PheroReset, self.serviceReset)
        self.is_service_requested = False
        self.theta = 0 
        
        # Flags & Counters
        self.log_timer = time.clock()
        #self.log_file = open("phero_value.txt", "a+")
        self.is_saved = False
        self.is_loaded = False
        self.is_reset = True # False for reset
        self.save_counter = 0

        for i in range(len(phero)):
            self.pheromone[i].isDiffusion = False
            self.pheromone[i].isEvaporation = True
        self.startTime = time.time()

        # Robot positions
        self.x = [0.0, 4.0]
        self.y = [0.0, 0.0]
        self.x_idx = [0, 0]
        self.y_idx = [0, 0]

        # Logging
        self.file_name = "pose_{}".format(self.num_robots)
        with open(self.pheromone[0].path + '/{}.csv'.format(self.file_name), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['time', 'ID', 'x', 'x_idx', 'y', 'y_idx', 'yaw'])


    def posToIndex(self, x, y):
        phero = self.pheromone
        x_tmp = x
        y_tmp = y
        # Read pheromone value at the robot position
        x_index = [0]*len(phero)
        y_index = [0]*len(phero)
        for i in range(len(phero)):
            res = phero[i].resolution
            round_dp = int(log10(res))
            x_tmp[i] = x_tmp[i]*2+1/res 
            y_tmp[i] = y_tmp[i]*2+1/res
            x_tmp[i] = round(x_tmp[i], round_dp) # round the position value so that they fit into the centre of the cell.
            y_tmp[i] = round(y_tmp[i], round_dp) # e.g. 0.13 -> 0.1
            x_tmp[i] = int(x_tmp[i]*(10**round_dp))
            y_tmp[i] = int(y_tmp[i]*(10**round_dp))
            # Position conversion from Robot into pheromone matrix (0, 0) -> (n+1, n+1) of 2n+1 matrix
            x_index[i] = int(x_tmp[i] + (phero[i].num_cell-1)/2)
            y_index[i] = int(y_tmp[i] + (phero[i].num_cell-1)/2)
            if x_index[i] < 0 or y_index[i] < 0 or x_index[i] > phero[i].num_cell-1 or y_index[i] > phero[i].num_cell-1:
                raise Exception("The pheromone matrix index is out of range.")
            pos_index = [x_index, y_index]
        return pos_index

    def indexToPos(self, x_index, y_index):
        phero = self.pheromone[0]
        x = x_index - (phero.num_cell-1)/2
        y = y_index - (phero.num_cell-1)/2

        x = float(x) / phero.resolution
        y = float(y) / phero.resolution

        return x, y

    def pheroCallback(self, message, cargs):
        
        start_time = time.time()
        for i in range(len(message.name)):
            if message.name[i] == 'tb3_0':
                tb3_0 = i

            if message.name[i] == 'tb3_1':
                tb3_1 = i
        
        tb3_pose = [message.pose[tb3_0], message.pose[tb3_1]]
        pose = [None]*self.num_robots
        ori = [None]*self.num_robots
        x = [None]*self.num_robots
        y = [None]*self.num_robots
        angles = [None]*self.num_robots
        theta = [None]*self.num_robots
        robot_pose = [None]*self.num_robots
        for i in range(self.num_robots):
            # Write relationship between i and the index
            pose[i] = tb3_pose[i] # Need to find the better way to assign index for each robot
            ori[i] = pose[i].orientation
            x[i] = pose[i].position.x
            y[i] = pose[i].position.y
            angles[i] = tf.transformations.euler_from_quaternion((ori[i].x, ori[i].y, ori[i].z, ori[i].w))
            theta[i] = angles[i][2]
            robot_pose[i] = [x[i], y[i], theta[i]]
        

        phero = cargs
        x_idx, y_idx = self.posToIndex(x, y)
        pos = [message.pose[tb3_0].position, message.pose[tb3_1].position] 
        x = [pos[0].x, pos[1].x]
        y = [pos[0].y, pos[1].y]
        
        # ========================================================================= #
	    #                           Pheromone Reading                               #
	    # ========================================================================= #

        '''
        Pheromone Value Reading
        '''

        ''' 2 values from the antennae'''
        robot_antennae = Antennae()
        antennae_pos = [None]*self.num_robots
        antennae_idx = [None]*self.num_robots
        phero_arr = [Float32MultiArray()]*self.num_robots
        phero_val = [None] * self.num_robots
        for i in range(self.num_robots):
            antennae_pos[i] = robot_antennae.position(robot_pose[i])
            antennae_idx[i] = self.posToIndex(antennae_pos[i][0], antennae_pos[i][1])
            phero_val[i] = list()   
            for j in range(2):
                phero_val[i].append(self.pheromone[1-i].getPhero(antennae_idx[i][j][0], antennae_idx[i][j][1]))
            phero_arr[i].data = phero_val[i]

        self.pub_phero.publish(phero_arr)

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

        # ========================================================================= #
	    #                           Pheromone Injection                             #
	    # ========================================================================= #

        # Pheromone injection (uncomment it when injection is needed)
        ## Two robots inject pheromone in different grids
        time_inj = time.clock()
        if self.is_phero_inj is True:
            for i in range(len(self.pheromone)):
                #phero[i].injection(x_idx[i], y_idx[i], 1, 25, self.phero_max)
                phero[i].gradInjection(x_idx[i], y_idx[i], 1, 0.3, 1.2, self.phero_max)

        # ========================================================================= #
	    #                           Pheromone Update                                #
	    # ========================================================================= #

        # Update pheromone matrix in every 0.1s
        time_cur = time.clock()
        if time_cur-phero[0].step_timer >= 0.05: 
            for i in range(self.num_robots):
                phero[i].update(self.phero_min, self.phero_max)
            #print("update period: {}".format(time_cur-phero[0].step_timer))    
            phero[0].step_timer = time_cur
            
            
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

        # Save Pheromone map for every 0.1 s
        save_cur = time.clock()
        if save_cur - phero[0].save_timer >= 0.1 and self.save_counter == 3:
            elapsed_time = save_cur - phero[0].reset_timer
            for i in range(self.num_robots):
                phero[i].save("{}_{:0.1f}".format(i, elapsed_time))
                if self.save_counter == 3:
                    with open(self.pheromone[0].path + '/{}.csv'.format(self.file_name), mode='a') as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(['{:0.1f}'.format(elapsed_time), '{}'.format(i), '{}'.format(x[i]),
                                             '{}'.format(x_idx[i]), '{}'.format(y[i]), '{}'.format(y_idx[i]), 
                                             '{}'.format(theta[i])])  


            phero[0].save_timer = save_cur
            print("Save time elapsed: {}".format(time.clock()-save_cur))

        end_time = time.time()
        #print("update time: {}".format(end_time - start_time))
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
                #self.pheromone.load("simple_collision_diffused") # you can load any types of pheromone grid
                print("Pheromone grid reset!")
                phero[0].reset_timer = time.clock()
                self.save_counter += 1
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

        for n in range(len(x_values)):
            phero_val[n] = list()     
            for i in range(3):
                for j in range(3):
                    phero_val[n].append(self.pheromone[1-n].getPhero(x_indices[n]+i-1, y_indices[n]+j-1)) # Read the other's pheromone
            phero_arr[n].data = phero_val[n]

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

    def __init__(self, name, evaporation, diffusion, path):
        self.name = name
        self.resolution = 20 # grid cell size = 1 m / resolution
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
        self.save_timer = time.clock()
        self.reset_timer = time.clock()

        # Logging Path
        self.path = path

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
        with open(self.path + '/{}.npy'.format(file_name), 'wb') as f:
            np.save(f, self.grid)
        #print("The pheromone matrix {} is successfully saved".format(file_name))

    def load(self, file_name):
        with open(self.path + '/{}.npy'.format(file_name), 'rb') as f:
            self.grid = np.load(f)
        #os.remove('/home/sub/catkin_ws/src/turtlebot3_pheromone/tmp/{}.npy'.format(file_name))
        print("The pheromone matrix {} is successfully loaded".format(file_name))

def main():

    time_str = time.strftime("%Y%m%d-%H%M%S")
    parent_dir = HOME + "/catkin_ws/src/Turtlebot3_Pheromone/tmp/"
    path = os.path.join(parent_dir, time_str)
    os.mkdir(path)
    
    rospy.init_node('pheromone')
    Phero1 = Pheromone('1', 0.5, 0, path)
    Phero2 = Pheromone('2', 0.5, 0, path)
    Phero = [Phero1, Phero2]
    node1 = Node(Phero)
    rospy.spin()
    
if __name__ == "__main__":
    main()
