#!/usr/bin/env python
import rospy
import gazebo_msgs.msg
import random
import tf
import numpy as np
import time
import sys
import csv
import threading
from gazebo_msgs.msg import ModelStates 
from gazebo_msgs.msg import ModelState 
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from turtlebot3_pheromone.srv import PheroReset, PheroResetResponse
from turtlebot3_pheromone.msg import fma
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from math import *
from time import sleep

from std_msgs.msg import Float32MultiArray

class InfoGetter(object):
    def __init__(self):
        #event that will block until the info is received
        self._event = threading.Event()
        #attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        #Uses __call__ so the object itself acts as the callback
        #save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        #"""Blocks until the data is rx'd with optional timeout
        #Returns the received message
        #"""
        self._event.wait(timeout)
        return self._msg

class Timers():
    '''
    Class for timers to count (1) waiting time (2) ignoring time (3) rotating time
    '''
    def __init__(self):
        # Wait related timers
        self.wait_on = False
        self.wait_start_time = 0.0
        self.t_wait = 3

        # Rotate related timers
        self.rotate_on = False
        self.rotate_start_time = 0.0
        self.t_rotate = 3

        # ignore rotate related timers
        self.ignore_rot_on = False
        self.ignore_rot_start_time = 0.0
        self.t_ignore_rot = 3

        # ignore wait related timers
        self.ignore_wait_on = False
        self.ignore_wait_start_time = 0.0
        self.t_ignore_wait = self.t_rotate + self.t_ignore_rot + 3

        # activation binary keys
        self.wait_act = False
        self.rotate_act = False

    def count_wait(self):
        if self.wait_act == True:
            if self.wait_on == False:
                if self.ignore_wait_on == False:
                    self.wait_on = True
                    self.wait_start_time = time.time()
                    return False
                else: 
                    if time.time() - self.ignore_wait_start_time < self.t_ignore_wait:
                        return True
                    else:
                        self.ignore_wait_on = False
                        self.wait_act = False
                        return False
            else:
                if time.time() - self.wait_start_time < self.t_wait:
                    return False
                else:
                    self.wait_on = False
                    self.ignore_wait_on = True
                    self.ignore_wait_start_time = time.time()
                    return True
        else: return False

    def count_rotate(self):
        if self.rotate_act == True:
            if self.rotate_on == False:
                if self.ignore_rot_on == False:
                    self.rotate_on = True
                    self.rotate_start_time = time.time()
                    return False
                else: 
                    if time.time() - self.ignore_rot_start_time < self.t_ignore_rot:
                        return True
                    else:
                        self.ignore_rot_on = False
                        self.rotate_act = False
                        return False
            else:
                if time.time() - self.rotate_start_time < self.t_rotate:
                    return False
                else:
                    self.rotate_on = False
                    self.ignore_rot_on = True
                    self.ignore_rot_start_time = time.time()
                    return True
        else: return False

class BeeClust():

    def __init__(self):

        self.num_robots = 2

        # Initialise speed
        self.move_cmd = [Twist()]*self.num_robots

        # Initialise positions
        self.x = [0.0]*self.num_robots
        self.y = [0.0]*self.num_robots
        self.theta = [0.0]*self.num_robots

        # Initialise ros related topics
        rospy.init_node('beeclust_exp0')
        self.pose_ig = InfoGetter()
        #self.phero_ig = InfoGetter()
        self.pub_tb3 = [None]*self.num_robots
        for i in range(self.num_robots):
            self.pub_tb3[i] = rospy.Publisher('/tb3_{}/cmd_vel'.format(i), Twist, queue_size=1)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose_ig, queue_size=1, buff_size=2**24)
        #self.sub_phero = rospy.Subscriber('/phero_value', fma, self.phero_ig, queue_size = 1)

        # Initialise Beeclust related variables
        self.stop_flag = [False]*self.num_robots
        self.start_times = [0.0]*self.num_robots
        self.cur_times = [0.0]*self.num_robots
        self.stop_duration = [0.0]*self.num_robots
        self.is_wait = [True]*self.num_robots
        self.post_wait_times = [0.0]*self.num_robots
        self.max_stop_time = 5

        self.post_turn_start = [0.0] * self.num_robots
        self.post_turn_period = [False] * self.num_robots
        self.turning_on = [False] * self.num_robots
        self.turn_start_time = [0.0] * self.num_robots

        # Forward Movement
        self.forward_speed = 0.1

        # Wall related variables
        self.wall_x = [-2.0, 2.0]
        self.wall_y = [-2.0, 2.0]
        self.wall_width = 0.5

        # # Timers & Flags
        # ## Wait
        # self.wait_on = False
        # self.wait_start_time = 0.0
        # self.wait_duration = 5
        # ## Ignore
        # self.ignore_on = False
        # self.ignore_start_time = 0.0
        # self.ignore_duration = 3
        # ## Rotate
        # self.rotate_on = False
        # self.rotate_start_time = 0.0
        # self.rotate_duration = 3

        # Timer Class
        self.beeclust_timer = Timers()
        self.wait_ignore = False
        self.rotate_ignore = False

        # Robot position related
        self.prev_x = []
        self.prev_y = []
        self.is_first_comp = True

        # Collision related
        self.d_col = 0.9


    def forward(self, robot_id):
        self.move_cmd[robot_id] = Twist()
        self.move_cmd[robot_id].linear.x = self.forward_speed


    def rotate(self, robot_id, direction):
        if direction == 'left':
            self.move_cmd[robot_id] = Twist()
            self.move_cmd[robot_id].angular.z = 0.5
        elif direction == 'right':
            self.move_cmd[robot_id] = Twist()
            self.move_cmd[robot_id].angular.z = -0.5
        # Robot rotates for desired degree for given time
        # if self.rotate_on == False:
        #     self.rotate_on = True
        #     self.rotate_start_time = time.time()
        #     self.move_cmd[robot_id] = Twist()
        #     self.move_cmd[robot_id].angular.z = 0.5
        # elif (time.time() - self.rotate_start_time) <= self.rotate_duration:
        #     self.move_cmd[robot_id].angular.z = 0.5
        # elif (time.time() - self.rotate_start_time) > self.rotate_duration:
        #     self.rotate_on = False 



    def wait(self, robot_id):
        self.move_cmd[robot_id] = Twist()
        # wait when the distance between robots are close
        # if self.wait_on == False and self.rotate_on == False:
        #     self.wait_on = True
        #     self.wait_start_time = time.time()
        #     self.move_cmd[robot_id] = Twist()
        # elif (time.time() - self.wait_start_time) <= self.wait_duration:
        #     self.move_cmd[robot_id] = Twist()
        # elif (time.time() - self.wait_start_time) > self.wait_duration:
        #     self.wait_on = False
            #self.rotate(robot_id)
            # if self.ignore_on == False:
            #     self.ignore_on = True
            #     self.forward(robot_id)
            #     self.ignore_start_time = time.time()
            # elif (time.time() - self.ignore_start_time) <= self.ignore_duration:
            #     self.forward(robot_id)
            # elif (time.time() - self.ignore_start_time) > self.ignore_duration:
            #     self.ignore_on = False



                 
                
        


    def computation(self):

        # ========================================================================= #
	    #                           Initialisation                                  #
	    # ========================================================================= #
        main_start_timer = time.time()
        message = self.pose_ig.get_msg()

        # Get turtlebot3 positions
        tb3 = [-1]*self.num_robots
        if message is not None:
            for i, name in enumerate(message.name):
                if 'tb3' in name:
                    tb3[int(name[-1])] = i
        poses = [message.pose[i] for i in tb3] 
        poss = [pose.position for pose in poses]
        x = [p.x for p in poss]
        y = [p.y for p in poss]
        if self.is_first_comp == True:
            self.prev_x = x
            self.prev_y = y
            self.is_first_comp = False
        oris = [pose.orientation for pose in poses]
        angles = [euler_from_quaternion((ori.x, ori.y, ori.z, ori.w)) for ori in oris]
        thetas = [angle[2] for angle in angles] # -pi - pi
        #print("Thetas: {}".format(thetas))


        # ========================================================================= #
	    #             Calculate Distances and Angles between Robots                 #
	    # ========================================================================= #
        
        distance_btw_robots = np.ones([self.num_robots, self.num_robots])
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if j != i:
                    distance_btw_robots[i][j] = sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) 
        
        theta_btw_robots = np.zeros([self.num_robots, self.num_robots])
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if j != i:
                    theta_btw_robots[i][j] = atan2(y[j]-y[i], x[j]-x[i])
        print("theta_btw_robots: {}".format(theta_btw_robots))


        # ========================================================================= #
	    #                                GO FORWARD                                 #
	    # ========================================================================= #
        for i in range(self.num_robots):
            self.move_cmd[i].linear.x = 0.1




        # ========================================================================= #
	    #                            OBSTACLE AVOIDANCE                             #
	    # ========================================================================= #

        # Wall Avoidance

        # ========================================================================= #
	    #                               ROBOT STOP                                  #
	    # ========================================================================= #

        #if any([dis <= 0.33 for dis in distance_btw_robots[i]]) == True:
        for i in range(self.num_robots):
            # Collision flag and timer
            if any(distance_btw_robots[i] <= self.d_col) == True:
                if self.wait_ignore == False:
                    if self.rotate_ignore == False:
                        self.wait(i)
                        self.beeclust_timer.wait_act = True
                        #print("wait")
                        print("wait time: {}".format(time.time()-self.beeclust_timer.wait_start_time))
                else: 
                    if self.rotate_ignore == False:
                        counterpart = i
                        for j in range(self.num_robots):
                            if distance_btw_robots[i][j] <= self.d_col:
                                counterpart = j # Currently it only considers two robots
                        if (thetas[i] - theta_btw_robots[i][counterpart]) >= 0:
                            self.rotate(i, 'left')
                        else:
                            self.rotate(i, 'left')
                        self.beeclust_timer.rotate_act = True
                        #print("rotate")
                        print("Rotate time: {}".format(time.time()-self.beeclust_timer.rotate_start_time))
                    else:
                        self.forward(i) 
                #print("wait_ignore: {}".format(self.wait_ignore))
                #print("rotate_ignore: {}".format(self.rotate_ignore))
                self.wait_ignore = self.beeclust_timer.count_wait()
                self.rotate_ignore = self.beeclust_timer.count_rotate()
            #     if self.turning_on[i] == False and self.post_turn_period[i] == False:
            #         self.turning_on[i] = True
            #         self.turn_start_time[i] = time.time()
            #     elif self.post_turn_period[i] == True:
            #         if time.time() - self.post_turn_start[i] < 3:
            #             self.move_cmd[i] = Twist()
            #             self.move_cmd[i].linear.x = 0.1
            #         else:
            #             self.post_turn_period[i] == False
            #     elif self.turning_on[i] == True:
            #         if time.time() - self.turn_start_time[i] < 3:
            #             self.move_cmd[i].angular.z = 0.5
            #         else: 
            #             self.turning_on[i] = False
            #             self.post_turn_period[i] = True
            #             self.post_turn_start[i] = time.time()

            # 20210311 Note that counter_rotate() is shared with the inter robot collision avoidance.
            if (x[i] < self.wall_x[0]+self.wall_width):
                if self.rotate_ignore == False:
                    self.beeclust_timer.rotate_act = True  
                    if thetas[i] > 0 and thetas[i] < pi:
                        self.rotate(i, 'right')
                    else:
                        self.rotate(i, 'left')
                    
                else:
                    self.forward(i)
                self.rotate_ignore = self.beeclust_timer.count_rotate()
            elif (x[i] > self.wall_x[1]-self.wall_width):
                if self.rotate_ignore == False:
                    self.beeclust_timer.rotate_act = True  
                    if thetas[i] > 0 and thetas[i] < pi:
                        self.rotate(i, 'left')
                    else:
                        self.rotate(i, 'right')
                        
                else:
                    self.forward(i)
                self.rotate_ignore = self.beeclust_timer.count_rotate()
            elif (y[i] < self.wall_y[0]+self.wall_width):
                if self.rotate_ignore == False:
                    self.beeclust_timer.rotate_act = True  
                    if thetas[i] < pi/2 and thetas[i] > -pi/2:
                        self.rotate(i, 'left')
                    else:
                        self.rotate(i, 'right')
                        
                        
                else:
                    self.forward(i)
                self.rotate_ignore = self.beeclust_timer.count_rotate()
            elif (y[i] > self.wall_y[1]-self.wall_width):
                if self.rotate_ignore == False:
                    self.beeclust_timer.rotate_act = True    
                    if thetas[i] < pi/2 and thetas[i] > -pi/2:
                        self.rotate(i, 'right')
                    else:
                        self.rotate(i, 'left')
                else:
                    self.forward(i)
                self.rotate_ignore = self.beeclust_timer.count_rotate()




        self.prev_x = x
        self.prev_y = y

         # Publish velocity 
        for i in range(self.num_robots):
            self.pub_tb3[i].publish(self.move_cmd[i])

    def reset(self):
        # Reset Turtlebot 1 position
        state_msg = ModelState()
        state_msg.model_name = 'tb3_0'
        state_msg.pose.position.x = -1.0
        state_msg.pose.position.y = 0.0 
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        # Reset Turtlebot 2 Position
        state_msg2 = ModelState()    
        state_msg2.model_name = 'tb3_1' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg2.pose.position.x = 1.0
        state_msg2.pose.position.y = 0.0
        state_msg2.pose.position.z = 0.0
        state_msg2.pose.orientation.x = 0
        state_msg2.pose.orientation.y = 0
        state_msg2.pose.orientation.z = -0.2
        state_msg2.pose.orientation.w = 0

        rospy.wait_for_service('gazebo/reset_simulation')

        # Request service to reset the position of robots
        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            resp = set_state(state_msg2)
            #resp_targ = set_state(state_target_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        for i in range(self.num_robots):
            self.move_cmd[i].linear.x = 0.0
            self.move_cmd[i].angular.z = 0.0
            self.pub_tb3[i].publish(self.move_cmd[i])

        
if __name__ == '__main__':
    beec = BeeClust()
    beec.reset()
    while True:
        try: 
            beec.computation()
        except KeyboardInterrupt:
            print("Ctrl+C pressed.")
            break
            
                        
