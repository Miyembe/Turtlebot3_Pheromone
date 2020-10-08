#! /usr/bin/env python

import rospy
import rospkg
import tf
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point, Quaternion
from math import *


from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty

import time
import tensorflow
import threading
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
import gym
import numpy as np
import random

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


class Env:

    def __init__(self):
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
        #

        self.rate = rospy.Rate(100)

        # Default Twist message
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.1 #linear_x
        self.move_cmd.angular.z = 0.0 #angular_z

        # Collision Bool State
        self.is_collided = False

        # Observation & action spaces
        self.state_num = 2 # 4 if input linear_x and angular_z
        self.action_num = 2 # linear_x and angular_z
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)

        # Set target position (stop sign in mini_arena.world)
        self.target_x = 2.0
        self.target_y = 0.0

        # Set turtlebot index in Gazebo (to distingush from other models in the world)
        self.model_index = -1

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.is_stuck = False

        # Miscellanous

    #To be done when real robots are used
    
    #def get_odom(self):

    #def print_odom(self):

    def reset(self):
        
        self.is_collided = False

        # Reset Turtlebot position
        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_waffle_pi'
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0 
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        # Reset Pheromone Grid
        #
        #

        rospy.wait_for_service('gazebo/reset_simulation')

        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        initial_state = np.zeros(self.state_num)

        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0
        self.pub.publish(self.move_cmd)
        time.sleep(1)
        self.pub.publish(self.move_cmd)
        self.rate.sleep()

        return initial_state

        # When turtlebot is collided with wall, obstacles etc - need to reset
        # def turtlebot_collsion(self):

    def step(self, time_step=0.1):

        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        self.rate.sleep()
        done = False
        # 0. Read the position of robot
        model_state = self.pose_ig.get_msg()
        pose = model_state.pose[self.model_index]
        x = pose.position.x
        y = pose.position.y
        print("x: {}, y: {}".format(x, y))
        distance_to_goal = sqrt((x-self.target_x)**2+(y-self.target_y)**2)

        # 1. Read pheromone (state) from the robot's position 
        state = self.phero_ig.get_msg()
        state_arr = np.asarray(state.data)

        # 2. State reshape
        state = state_arr.reshape(1, self.state_num)

        # 3. Reward assignment
        ## 3.0. Initialisation of rewards
        time_penalty = 0.0
        phero_reward = 0.0
        goal_reward = 0.0
        ## 3.1. Time penalty
        if (done is False):
            # Timer based
            # record_time = time.time()
            # elapsed_time = record_time - start_time
            # time_penalty = -1*elapsed_time

            # Iteration based
            time_penalty = -1.0
        ## 3.2. Pheromone reward (The higher pheromone, the higher reward)
            phero_avg = np.average(state)
            phero_reward = phero_avg - 1.0 # max phero: 0, min phero: -1

        ## 3.3. Goal reward
            if distance_to_goal <= 0.5:
                goal_reward = 100.0
                done = True
                self.reset()
                time.sleep(1)
        ## 3.4. Collision penalty
        #   if it collides to walls, it gets penalty, sets done to true, and reset
        #

        # 4. Stuck resetting (need to modify for this particular condition)
        #   if it stucks, reset the position
        #   if  linear_x > 0.01 and angular_z > 0.05 and abs(distance_reward) < 0.005:
                # self.stuck_indicator = self.stuck_indicator+1
                # if self.stuck_indicator >4:
                #     self.stuck_indicator = 0
                #     print("it is resetting!!!!!")
                #     time.sleep(5)
                #     self.reset()

        # if linear_x > 0.05 and angular_z > 0.05 and abs(distance_reward) > 0.005:
        #     self.stuck_indicator = 0

        reward = time_penalty + phero_reward + goal_reward
        
        return state, reward, done
    
    def print_debug(self):

        # For debugging. return any data you want. 
        print("Phero Info: {}".format(self.phero_info))

if __name__ == '__main__':
    try:
        sess = tensorflow.Session()
        K.set_session(sess)
        env = Env()
    except rospy.ROSInterruptException:
        pass