#! /usr/bin/env python

import rospy
import rospkg
import tf
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point, Quaternion
from tf.transformations import quaternion_from_euler
import math
from math import *


from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from turtlebot3_pheromone.srv import PheroReset, PheroResetResponse

import time
import tensorflow
import threading
# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Input, merge
# from keras.layers.merge import Add, Concatenate
# from keras.optimizers import Adam
# import keras.backend as K
import gym
import numpy as np
import random



class InfoGetter(object):
    '''
    Get Pheromone Information
    '''
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


class Env:

    # ========================================================================= #
	#                                Env Class                                  #
	# ========================================================================= #

    '''
    Class of connecting Simulator and DRL training
    '''

    def __init__(self):

        # Settings
        self.num_robots = 1

        # Node Initialisation
        self.node = rospy.init_node('phero_turtlebot_env', anonymous=True)
        self.pose_ig = InfoGetter()
        self.phero_ig = InfoGetter()
        #self.collision_ig = InfoGetter()

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.position = Point() # Do I need this position in this script? or just get pheromone value only?
        self.move_cmd = Twist()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.phero_info = rospy.Subscriber("/phero_value", Float32MultiArray, self.phero_ig)

        ## tf related lines. Needed for real turtlebot odometry reading.
        #   Skip for now. 

        self.rate = rospy.Rate(100)

        # Default Twist message
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.1 #linear_x
        self.move_cmd.angular.z = 0.0 #angular_z

        # Collision Bool State
        self.is_collided = False

        # Observation & action spaces
        self.state_num = 6 # 9 for pheromone 1 for goal distance, 2 for linear & angular speed, 1 for angle diff
        self.action_num = 2 # linear_x and angular_z
        self.observation_space = np.empty(self.state_num)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))#np.empty(self.action_num)

        # Set target position
        self.target_x = 4.0
        self.target_y = 0.0
        self.target_index = 0
        self.radius = 4
        self.num_experiments = 4
        
        # Last robot positions (to use for stuck indicator)
        self.last_x = 0.0
        self.last_y = 0.0
        self.stuck_indicator = 0

        # Set turtlebot index in Gazebo (to distingush from other models in the world)
        self.model_index = -1

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # Miscellanous
        self.ep_len_counter = 0
        self.dis_rwd_norm = 7
        self.grad_sensitivity = 20
        self.theta = 0
        print("werwerq")

    #To be done when real robots are used
    
    #def get_odom(self):

    #def print_odom(self):

    def reset(self):

        '''
        Resettng the Experiment
        1. Update the counter based on the flag from step
        2. Assign next positions and reset
        3. Log the result in every selected time-step
        '''

        # ========================================================================= #
	    #                            TARGET UPDATE                                  #
	    # ========================================================================= #
        
        self.is_collided = False

        angle_target = self.target_index*2*pi/self.num_experiments        

        self.target_x = self.radius*cos(angle_target)
        self.target_y = self.radius*sin(angle_target)
        
        if self.target_index < self.num_experiments-1:
            self.target_index += 1
        else:
            self.target_index = 0

        self.theta = angle_target
        quat = quaternion_from_euler(0,0,self.theta)
        

        # ========================================================================= #
	    #                                  RESET                                    #
	    # ========================================================================= #

        # Reset Turtlebot position
        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_waffle_pi'
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0 
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = quat[0]
        state_msg.pose.orientation.y = quat[1]
        state_msg.pose.orientation.z = quat[2]
        state_msg.pose.orientation.w = quat[3]

        # Reset Target Position
        state_target_msg = ModelState()    
        state_target_msg.model_name = 'unit_sphere_0_0' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_target_msg.pose.position.x = self.target_x
        state_target_msg.pose.position.y = self.target_y
        state_target_msg.pose.position.z = 0.0
        state_target_msg.pose.orientation.x = 0
        state_target_msg.pose.orientation.y = 0
        state_target_msg.pose.orientation.z = -0.2
        state_target_msg.pose.orientation.w = 0

        rospy.wait_for_service('gazebo/reset_simulation')

        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            resp_targ = set_state(state_target_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        initial_state = np.zeros(self.state_num)

        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0
        self.pub.publish(self.move_cmd)
        time.sleep(1)
        self.pub.publish(self.move_cmd)
        self.rate.sleep()

        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)

        return range(0, self.num_robots), initial_state

    def step(self, time_step=0.1, linear_x=0.2, angular_z=0.0):
        '''
        Take a step with the given action from DRL in the Environment
        0. Initialisation
        1. Move Robot for given time step
        2. Read robot pose
        3. Calculation of distances
        4. Read Pheromone
        5. Reward Assignment
        6. Reset
        7. Other Debugging Related
        '''
        # 0. Initialisation
        start_time = time.time()
        record_time = start_time
        record_time_step = 0

        # rescaling the action
        print("twist: [{}, {}]".format(linear_x, angular_z))
        linear_x = linear_x*0.3
        linear_x = min(1, max(-1, linear_x))
        linear_x = (linear_x+1)*1/2
        angular_z = min(pi/2, max(-pi/2, angular_z*0.6))
        

        self.move_cmd.linear.x = linear_x
        self.move_cmd.angular.z = angular_z
        action = np.array([linear_x, angular_z])
        print("action: {}".format(action))
        self.rate.sleep()
        done = False

        # position of turtlebot before taking steps
        model_state = self.pose_ig.get_msg()
        pose = model_state.pose[self.model_index]
        x_previous = pose.position.x
        y_previous = pose.position.y
        distance_to_goal_prv = sqrt((x_previous-self.target_x)**2+(y_previous-self.target_y)**2)

        # Collect previous pheromone data
        phero_prev = self.phero_ig.get_msg().data

        # 1. Move robot with the action input for time_step
        while (record_time_step < time_step and done == False):
            self.pub.publish(self.move_cmd)
            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time

        # 2. Read the position and angle of robot

        model_state = self.pose_ig.get_msg()
        pose = model_state.pose[self.model_index]
        ori = pose.orientation
        x = pose.position.x
        y = pose.position.y
        angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        theta = angles[2]

        # 3. Calculate the distance & angle difference to goal 
        distance_to_goal = sqrt((x-self.target_x)**2+(y-self.target_y)**2)
        global_angle = atan2(self.target_y - y, self.target_x - x)
        
        if theta < 0:
            theta = theta + 2*math.pi
        if global_angle < 0:
            global_angle = global_angle + 2*math.pi

        angle_diff = global_angle - theta
        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2*math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2*math.pi

        # 4. Read pheromone (state) from the robot's position 

        phero_now = self.phero_ig.get_msg().data
        phero_grad = self.grad_sensitivity*(np.array(phero_now) - np.array(phero_prev))
        
        print("phero_grad: {}".format(phero_grad))
        state_arr = phero_grad
        state_arr = np.append(state_arr, np.asarray(phero_now))
        state_arr = np.append(state_arr, distance_to_goal)
        #state_arr = np.append(state_arr, linear_x)
        #state_arr = np.append(state_arr, angular_z)
        state_arr = np.append(state_arr, angle_diff)
        state = state_arr.reshape(1, self.state_num)

        # 5. Reward assignment
        ## 5.0. Initialisation of rewards
        distance_reward = 0.0
        phero_reward = 0.0
        goal_reward = 0.0
        
        ## 5.1. Distance Reward
        goal_progress = distance_to_goal_prv - distance_to_goal
        if goal_progress >= 0:
            distance_reward = goal_progress * 1.2
        else:
            distance_reward = goal_progress
        
        ## 5.2. Pheromone reward (The higher pheromone, the lower reward)
        #phero_sum = np.sum(phero_vals)
        phero_reward = 0.0 #(-phero_sum) # max phero_r: 0, min phero_r: -9
    
        ## 5.3. Goal reward
        if distance_to_goal <= 0.4:
            goal_reward = 100.0
            done = True
            self.reset()
            time.sleep(1)

        ## 5.4. Angular speed penalty
        angular_punish_reward = 0.0
        if abs(angular_z) > 1.2:
            angular_punish_reward = -1.0
        
        ## 5.5. Linear speed penalty
        linear_punish_reward = 0.0
        if linear_x < 0.2:
            linear_punish_reward = -1.0
        ## 5.6. Collision penalty
        #   if it collides to walls, it gets penalty, sets done to true, and reset
        collision_reward = 0.0
        obs_pos = [[2, 0],[-2,0],[0,2],[0,-2]]
        dist_obs = [sqrt((x-obs_pos[i][0])**2+(y-obs_pos[i][1])**2) for i in range(len(obs_pos))]
        for i in range(len(obs_pos)):
            if dist_obs[i] < 0.3:
                collision_reward = -150.0
                self.reset()
                time.sleep(0.5)

        ## 5.7. Sum of Rewards
        reward = distance_reward*(4/time_step) + phero_reward + goal_reward + angular_punish_reward + linear_punish_reward + collision_reward
        reward = np.asarray(reward).reshape(1)

        # 6. Reset
        ## 6.1. when robot goes too far from the target
        if distance_to_goal >= self.dis_rwd_norm:
            self.reset()
            time.sleep(0.5) 

        ## 6.2. when the robot is out of the pheromone grid
        if abs(x) >= 4.7 or abs(y) >= 4.7:
            self.reset()
            time.sleep(0.5)
        end_time = time.time()
        step_time = end_time - start_time
        # 7. Other Debugging 
        info = [{"episode": {"l": self.ep_len_counter, "r": reward}}]
        self.ep_len_counter = self.ep_len_counter + 1
        print("-------------------")
        print("Ep: {}".format(self.ep_len_counter))
        print("Step time: {}".format(step_time))
        print("GP: {}".format(goal_progress))
        print("Target: ({}, {})".format(self.target_x, self.target_y))
        print("Distance R: {}".format(distance_reward*(4/time_step)))
        #print("Phero R: {}".format(phero_reward))
        #print("Goal R: {}".format(goal_reward))
        #print("Angular R: {}".format(angular_punish_reward))
        #print("Linear R: {}".format(linear_punish_reward))
        print("Collision R: {}".format(collision_reward))
        print("Reward: {}".format(reward))
        #print("state: {}, action:{}, reward: {}, done:{}, info: {}".format(state, action, reward, done, info))
        return range(0, self.num_robots), state, reward, done, info

if __name__ == '__main__':
    try:
        sess = tensorflow.Session()
        K.set_session(sess)
        env = Env()
    except rospy.ROSInterruptException:
        pass