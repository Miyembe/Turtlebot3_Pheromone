#! /usr/bin/env python

import rospy
import rospkg
import tf
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point, Quaternion
import math
from math import *


from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from turtlebot3_pheromone.srv import PheroReset, PheroResetResponse
from turtlebot3_pheromone.srv import PheroRead, PheroReadResponse
from turtlebot3_pheromone.msg import fma
from tf.transformations import quaternion_from_euler

import time
import tensorflow
import threading
#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Input, merge
#from keras.layers.merge import Add, Concatenate
#from keras.optimizers import Adam
#import keras.backend as K
import gym
import numpy as np
import random

import scipy.io as sio



class InfoGetter(object):
    '''
    Get Information from rostopic. It reduces delay 
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
    This class define Env (identical concept of OpenAI gym Env).
    1. __init__() - define required variables
    2. reset()
    3. step()
    '''

    def __init__(self):

        # Settings
        self.num_robots = 4

        # Node initialisation
        self.node = rospy.init_node('phero_turtlebot_env', anonymous=True)
        self.pose_ig = InfoGetter()
        self.phero_ig = InfoGetter()
        #self.collision_ig = InfoGetter()

        self.pub_tb3_0 = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=1)
        self.pub_tb3_1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=1)
        self.pub_tb3_2 = rospy.Publisher('/tb3_2/cmd_vel', Twist, queue_size=1)
        self.pub_tb3_3 = rospy.Publisher('/tb3_3/cmd_vel', Twist, queue_size=1)
        self.position = Point() # Do I need this position in this script? or just get pheromone value only?
        self.move_cmd = Twist()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.phero_info = rospy.Subscriber("/phero_value", fma, self.phero_ig)
        self.rate = rospy.Rate(100)

        # Default Twist message
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.1 #linear_x
        self.move_cmd.angular.z = 0.0 #angular_z

        # Collision Bool State
        self.is_collided = False

        # Observation & action spaces
        self.state_num = 8 # 2 for pheromone grad, 2 for pheromone value, 2 for linear & angular vel, 2 for distance and angle diff to the target in polar coordinates
        self.action_num = 2 # linear_x and angular_z
        self.observation_space = np.empty(self.state_num)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        # Previous positions
        self.x_prev = [-2.5, 2.5, 0.0, 0.0]#[0.0, 2.0] # [0.0,4.0]
        self.y_prev = [0.0, 0.0, -2.5, 2.5] #[0.0, -2.0]  # [0.0,0.0]
        self.x = [0.0]*self.num_robots
        self.y = [0.0]*self.num_robots
        self.theta = [0.0]*self.num_robots
        self.target_index = 0

        # Set target position
        self.target = [[2.5, 0.0], [-2.5, 0.0], [0.0, 2.5], [0.0, -2.5]]#[[4.0, 0.0], [2.0, 2.0]] # Two goal (crossing scenario) # [[4.0,0.0], [0.0,0.0]]

        # Set turtlebot index in Gazebo (to distingush from other models in the world)
        self.model_index = -1
        self.model_state = ModelStates()
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # Miscellanous
        self.ep_len_counter = 0
        self.just_reset = [False] * self.num_robots
        self.dones = [False] * self.num_robots
        self.grad_sensitivity = 20
        self.num_experiments = 20
        self.d_robots = 5

    def reset(self, model_state = None, id_bots = 999):

        '''
        Resettng the Experiment
        1. Update the counter based on the flag from step
        2. Assign next positions and reset the positions of robots and targets
        '''

        # ========================================================================= #
	    #                          1. TARGET UPDATE                                 #
	    # ========================================================================= #
        
        self.is_collided = False
        tb3_0 = -1
        tb3_1 = -1
        tb3_2 = -1
        tb3_3 = -1
        if model_state is not None:
            for i in range(len(model_state.name)):
                if model_state.name[i] == 'tb3_0':
                    tb3_0 = i
                if model_state.name[i] == 'tb3_1':
                    tb3_1 = i
                if model_state.name[i] == 'tb3_2':
                    tb3_2 = i
                if model_state.name[i] == 'tb3_3':
                    tb3_3 = i
                
        else:
            tb3_0 = -1
            tb3_1 = -2
            tb3_2 = -3
            tb3_3 = -4
        
        if id_bots == 999: 
            if self.target_index < self.num_experiments-1:
                self.target_index += 1
            else:
                self.target_index = 0
                
        angle_target = self.target_index*2*pi/self.num_experiments        

        self.x[0] = (self.d_robots/2)*cos(angle_target)
        self.y[0] = (self.d_robots/2)*sin(angle_target)

        self.x[1] = (self.d_robots/2)*cos(angle_target+pi)
        self.y[1] = (self.d_robots/2)*sin(angle_target+pi)

        self.x[2] = (self.d_robots/2)*cos(angle_target+pi/2)
        self.y[2] = (self.d_robots/2)*sin(angle_target+pi/2)

        self.x[3] = (self.d_robots/2)*cos(angle_target+3*pi/2)
        self.y[3] = (self.d_robots/2)*sin(angle_target+3*pi/2)

        self.theta[0] = angle_target + pi
        self.theta[1] = angle_target 
        self.theta[2] = angle_target + 3*pi/2
        self.theta[3] = angle_target + pi/2

        quat1 = quaternion_from_euler(0,0,self.theta[0])
        quat2 = quaternion_from_euler(0,0,self.theta[1])
        quat3 = quaternion_from_euler(0,0,self.theta[2])
        quat4 = quaternion_from_euler(0,0,self.theta[3])
        
        self.target = [[self.x[1], self.y[1]], [self.x[0], self.y[0]], [self.x[3], self.y[3]], [self.x[2], self.y[2]]]
        
        
        


        # ========================================================================= #
	    #                                 2. RESET                                  #
	    # ========================================================================= #
        
       # Reset Turtlebot 1 position
        state_msg = ModelState()
        state_msg.model_name = 'tb3_0'
        state_msg.pose.position.x = self.x[0]
        state_msg.pose.position.y = self.y[0]
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = quat1[0]
        state_msg.pose.orientation.y = quat1[1]
        state_msg.pose.orientation.z = quat1[2]
        state_msg.pose.orientation.w = quat1[3]

        # Reset Turtlebot 2 Position
        state_msg2 = ModelState()    
        state_msg2.model_name = 'tb3_1' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg2.pose.position.x = self.x[1]
        state_msg2.pose.position.y = self.y[1]
        state_msg2.pose.position.z = 0.0
        state_msg2.pose.orientation.x = quat2[0]
        state_msg2.pose.orientation.y = quat2[1]
        state_msg2.pose.orientation.z = quat2[2]
        state_msg2.pose.orientation.w = quat2[3]

        # Reset Turtlebot 3 Position

        state_msg3 = ModelState()    
        state_msg3.model_name = 'tb3_2' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg3.pose.position.x = self.x[2]
        state_msg3.pose.position.y = self.y[2]
        state_msg3.pose.position.z = 0.0
        state_msg3.pose.orientation.x = quat3[0]
        state_msg3.pose.orientation.y = quat3[1]
        state_msg3.pose.orientation.z = quat3[2]
        state_msg3.pose.orientation.w = quat3[3]

        # Reset Turtlebot 4 Position

        state_msg4 = ModelState()    
        state_msg4.model_name = 'tb3_3' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg4.pose.position.x = self.x[3]
        state_msg4.pose.position.y = self.y[3]
        state_msg4.pose.position.z = 0.0
        state_msg4.pose.orientation.x = quat4[0]
        state_msg4.pose.orientation.y = quat4[1]
        state_msg4.pose.orientation.z = quat4[2]
        state_msg4.pose.orientation.w = quat4[3]

        rospy.wait_for_service('gazebo/reset_simulation')

        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            if id_bots == 999 or id_bots == tb3_0:
                resp = set_state(state_msg)
                self.dones[0] = False
            if id_bots == 999 or id_bots == tb3_1:
                resp2 = set_state(state_msg2)
                self.dones[1] = False
            if id_bots == 999 or id_bots == tb3_2:
                resp3 = set_state(state_msg3)
                self.dones[2] = False
            if id_bots == 999 or id_bots == tb3_3:
                resp4 = set_state(state_msg4)
                self.dones[3] = False
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        initial_state = np.zeros((self.num_robots, self.state_num))
        
        self.just_reset = True

        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0
        if id_bots == 999 or id_bots == tb3_0:
            self.pub_tb3_0.publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3_1:
            self.pub_tb3_1.publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3_2:
            self.pub_tb3_2.publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3_3:
            self.pub_tb3_3.publish(self.move_cmd)
        self.rate.sleep()

        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)
        #self.dones = [False] * self.num_robots

        return range(0, self.num_robots), initial_state

        # When turtlebot is collided with wall, obstacles etc - need to reset
        # def turtlebot_collsion(self):

    def action_to_twist(self, action):
        t = Twist()

        # Rescale and clipping the actions
        t.linear.x = action[1]*0.3
        t.linear.x = min(1, max(-1, t.linear.x))
        
        t.angular.z = min(pi/2, max(-pi/2, action[0]))
        return t
    
    def posAngle(self, model_state):
        '''
        Get model_state from rostopic and
        return (1) x position of robots (2) y position of robots (3) angle of the robots (4) id of the robots
        '''
        pose = [None]*self.num_robots
        ori = [None]*self.num_robots
        x = [None]*self.num_robots
        y = [None]*self.num_robots
        angles = [None]*self.num_robots
        theta = [None]*self.num_robots
        for i in range(len(model_state.name)):
            if model_state.name[i] == 'tb3_0':
                tb3_0 = i
            if model_state.name[i] == 'tb3_1':
                tb3_1 = i
            if model_state.name[i] == 'tb3_2':
                tb3_2 = i
            if model_state.name[i] == 'tb3_3':
                tb3_3 = i
        tb3_pose = [model_state.pose[tb3_0], model_state.pose[tb3_1], model_state.pose[tb3_2], model_state.pose[tb3_3]]
        for i in range(self.num_robots):
            # Write relationship between i and the index
            pose[i] = tb3_pose[i] # Need to find the better way to assign index for each robot
            ori[i] = pose[i].orientation
            x[i] = pose[i].position.x
            y[i] = pose[i].position.y
            angles[i] = tf.transformations.euler_from_quaternion((ori[i].x, ori[i].y, ori[i].z, ori[i].w))
            theta[i] = angles[i][2]
        idx = [tb3_0, tb3_1, tb3_2, tb3_3]
        return x, y, theta, idx

    def angle0To360(self, angle):
        for i in range(self.num_robots):
            if angle[i] < 0:
                angle[i] = angle[i] + 2*math.pi
        return angle
    
    def anglepiTopi(self, angle):
        for i in range(self.num_robots):
            if angle[i] < -math.pi:
                angle[i] = angle[i] + 2*math.pi
            if angle[i] > math.pi:
                angle[i] = angle[i] - 2*math.pi
        return angle

    def swap2elements(self, array):
        assert len(array) == 2
        tmp = [None]*2
        tmp[0] = array[1]
        tmp[1] = array[0]
        return tmp

    def step(self, actions, time_step=0.1):
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
        # 0. Initiliasation
        start_time = time.time()
        record_time = start_time
        record_time_step = 0

        # Check if the robots are terminated
        dones = self.dones
        is_stops = dones
        
        twists = [self.action_to_twist(action) for action in np.asarray(actions)]

        # rescaling the action
        for i in range(len(twists)):
            twists[i].linear.x = (twists[i].linear.x+1) * 1/2  # only forward motion
            twists[i].angular.z = twists[i].angular.z
            if is_stops[i] == True:
                twists[i] = Twist()
        
        linear_x = [i.linear.x for i in twists]
        angular_z = [i.angular.z for i in twists]
        

        # position of turtlebot before taking steps
        x_prev = self.x_prev
        y_prev = self.y_prev
        distance_to_goals_prv = [None]*self.num_robots
        for i in range(self.num_robots):
            distance_to_goals_prv[i] = sqrt((x_prev[i]-self.target[i][0])**2+(y_prev[i]-self.target[i][1])**2)

        state_prev = self.phero_ig.get_msg()
        phero_prev = [phero.data for phero in state_prev.values]

        # 1. Move robot with the action input for time_step
        while (record_time_step < time_step):
            self.pub_tb3_0.publish(twists[0])
            self.pub_tb3_1.publish(twists[1])
            self.pub_tb3_2.publish(twists[2])
            self.pub_tb3_3.publish(twists[3])

            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time
        
        # 2. Read the position and angle of robot
        model_state = self.pose_ig.get_msg()
        self.model_state = model_state

        x, y, theta, idx = self.posAngle(model_state)
        self.x_prev = x
        self.y_prev = y

        # 3. Calculate the distance & angle difference to goal \
        distance_to_goals = [None]*self.num_robots
        global_angle = [None]*self.num_robots
        #print("x : {}, y: {}".format(x,y))
        for i in range(self.num_robots):
            distance_to_goals[i] = sqrt((x[i]-self.target[i][0])**2+(y[i]-self.target[i][1])**2)
            global_angle[i] = atan2(self.target[i][1] - y[i], self.target[i][0] - x[i])

        theta = self.angle0To360(theta)
        global_angle = self.angle0To360(global_angle)
        angle_diff = [a_i - b_i for a_i, b_i in zip(global_angle, theta)]
        angle_diff = self.anglepiTopi(angle_diff)

        # 4. Read pheromone (state) from the robot's position

        state = self.phero_ig.get_msg()
        phero_now = [phero.data for phero in state.values]
        phero_grad = self.grad_sensitivity*(np.array(phero_now) - np.array(phero_prev))

        
        # Concatenating the state array
        state_arr = np.asarray(phero_grad)
        state_arr = np.hstack((state_arr, phero_now))
        state_arr = np.hstack((state_arr, np.asarray(distance_to_goals).reshape(self.num_robots,1)))
        state_arr = np.hstack((state_arr, np.asarray(linear_x).reshape(self.num_robots,1)))
        state_arr = np.hstack((state_arr, np.asarray(angular_z).reshape(self.num_robots,1)))
        state_arr = np.hstack((state_arr, np.asarray(angle_diff).reshape(self.num_robots,1)))

        # 5. State reshape
        states = state_arr.reshape(self.num_robots, self.state_num)

        
        # 6. Reward assignment
        ## 6.0. Initialisation of rewards
        distance_rewards = [0.0]*self.num_robots
        phero_rewards = [0.0]*self.num_robots
        goal_rewards = [0.0]*self.num_robots
        angular_punish_rewards = [0.0]*self.num_robots
        linear_punish_rewards = [0.0]*self.num_robots
        time_rewards = [0.0]*self.num_robots
        ooa_rewards = [0.0]*self.num_robots        
        ## 6.1. Distance Reward
        goal_progress = [a - b for a, b in zip(distance_to_goals_prv, distance_to_goals)]

        time_step_factor = 4/time_step
        for i in range(self.num_robots):
            if abs(goal_progress[i]) < 0.1:
                if goal_progress[i] >= 0:
                        distance_rewards[i] = goal_progress[i] * 1.2
                else:
                        distance_rewards[i] = goal_progress[i]
            else:
                distance_rewards[i] = 0.0
            distance_rewards[i] *= time_step_factor
        
        self.just_reset == False
        
        ## 6.2. Pheromone reward (The higher pheromone, the lower reward)
        phero_rewards = [0.0, 0.0, 0.0, 0.0]#[-phero_sum*2 for phero_sum in phero_sums] # max phero_r: 0, min phero_r: -9
        
        ## 6.3. Goal reward
        ### Reset condition is activated when both two robots have arrived their goals 
        ### Arrived robots stop and waiting
        for i in range(self.num_robots):
            if distance_to_goals[i] <= 0.5:
                goal_rewards[i] = 100.0
                dones[i] = True
                #self.reset(model_state, id_bots=idx[i])

            
        

        ## 6.4. Angular speed penalty
        for i in range(self.num_robots):
            if abs(angular_z[i])>0.8:
                angular_punish_rewards[i] = -1
                if dones[i] == True:
                    angular_punish_rewards[i] = 0.0
        
        ## 6.5. Linear speed penalty
        for i in range(self.num_robots):
            if linear_x[i] < 0.2:
                linear_punish_rewards[i] = -1.0
        for i in range(self.num_robots):
            if dones[i] == True:
                linear_punish_rewards[i] = 0.0
        ## 6.6. Collision penalty
        #   if it collides to walls, it gets penalty, sets done to true, and reset
        #   it needs to be rewritten to really detect collision

        distance_btw_robots = np.ones([self.num_robots, self.num_robots])
        distance_to_obstacle = np.ones(self.num_robots)
        for i in range(self.num_robots):
            distance_to_obstacle[i] = sqrt((x[i])**2+(y[i])**2)
            for j in range(self.num_robots):
                if j != i:
                    distance_btw_robots[i][j] = sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) # Python 
        collision_rewards = [0.0]*self.num_robots

        for i in range(self.num_robots):
            if any([dis <= 0.32 for dis in distance_btw_robots[i]]) == True:
                print("Collision! Robot: {}".format(i))
                collision_rewards[i] = -100.0
                dones[i] = True
                #self.reset(model_state, id_bots=idx[i])
            elif distance_to_obstacle[i] < 0.3:
                collision_rewards[i] = -100.0
                dones[i] = True
                #self.reset(model_state, id_bots=idx[i])
        
        ## 6.7. Time penalty
        #  constant time penalty for faster completion of episode
        for i in range(self.num_robots):
            time_rewards[i] = 0.0 # 20201217 I nullified the time_rewards
            if dones[i] == True:
                time_rewards[i] = 0.0

        ## 6.8. Out of Arena penalty
        for i in range(self.num_robots):
            if abs(x[i]) >= 5.4 or abs(y[i]) >= 5.4:
                if distance_to_goals[i] > 6:
                    ooa_rewards[i] = 0.0
                dones[i] = True
                #self.reset(model_state, id_bots=idx[i])
            
        

        # 7. Reset
        ## 7.1. when robot goes too far from the target
        # if distance_to_goal >= self.dis_rwd_norm:
        #     self.reset()
        #     time.sleep(0.5) 
        
        ## 7.2. when it collides to the target 
        # obs_pos = [[2, 0]]
        # dist_obs = [sqrt((x-obs_pos[0][0])**2+(y-obs_pos[0][1])**2)]
        # if dist_obs[0] < 0.1:
        #     self.reset()
        #     time.sleep(0.5)
        ## 7.3. when the robot is out of the pheromone grid
        test_time = time.time()
        
        
        ## 7.4. If all the robots are done with tasks, reset
        if all(flag == True for flag in dones) == True:
            self.reset(model_state, id_bots=999)
            for i in range(self.num_robots):
                dones[i] = False

        self.dones = dones
        

        rewards = [a+b+c+d+e+f+g+h for a, b, c, d, e, f, g, h in zip(distance_rewards, phero_rewards, goal_rewards, angular_punish_rewards, linear_punish_rewards, collision_rewards, time_rewards, ooa_rewards)]
        test_time2 = time.time()
        rewards = np.asarray(rewards).reshape(self.num_robots)
        infos = [{"episode": {"l": self.ep_len_counter, "r": rewards}}]
        self.ep_len_counter = self.ep_len_counter + 1
        print("-------------------")
        print("Infos: {}".format(infos))
        #print("Robot 1, x: {}, y: {}, ps: {}".format(x[0], y[0], phero_sums[0]))
        #print("Robot 2, x: {}, y: {}, ps: {}".format(x[1], y[1], phero_sums[1]))

        #print("Distance R1: {}, R2: {}".format(distance_rewards[0]*(4/time_step), distance_rewards[1]*(4/time_step)))
        #print("Phero R1: {}, R2: {}".format(phero_rewards[0], phero_rewards[1]))
        #print("Goal R1: {}, R2: {}".format(goal_rewards[0], goal_rewards[1]))
        #print("Collision R1: {}, R2: {}".format(collision_rewards[0], collision_rewards[1]))
        #print("Angular R: {}".format(angular_punish_reward))
        #print("Linear R: {}".format(linear_punish_reward))
        print("Linear: {}, Angular: {}".format(linear_x, angular_z))
        print("Reward: {}".format(rewards))
        print("Time diff: {}".format(test_time-test_time2))
        

        #print("state: {}, action:{}, reward: {}, done:{}, info: {}".format(state, action, reward, done, info))
        return range(0, self.num_robots), states, rewards, dones, infos, is_stops
        
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