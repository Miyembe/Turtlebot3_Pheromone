#! /usr/bin/env python2.7

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

import scipy.io as sio



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

        # Settings
        self.num_robots = 6
        self.num_cylinders = 9
        self.num_boxes = 0
        self.num_obstacles = self.num_cylinders + self.num_boxes

        # Node initialisation
        self.node = rospy.init_node('phero_turtlebot_env', anonymous=True)
        self.pose_ig = InfoGetter()
        self.phero_ig = InfoGetter()

        self.pub_tb3 = [None]*self.num_robots
        for i in range(self.num_robots):
            self.pub_tb3[i] = rospy.Publisher('/tb3_{}/cmd_vel'.format(i), Twist, queue_size=1)
        
            
        # self.pub_tb3_0 = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=1)
        # self.pub_tb3_1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=1)
        # self.pub_tb3_2 = rospy.Publisher('/tb3_2/cmd_vel', Twist, queue_size=1)
        # self.pub_tb3_3 = rospy.Publisher('/tb3_3/cmd_vel', Twist, queue_size=1)
        self.position = Point() # Do I need this position in this script? or just get pheromone value only?
        self.move_cmd = Twist()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.phero_info = rospy.Subscriber("/phero_value", fma, self.phero_ig)

        

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
        self.state_num = 14 # 9 for pheromone, 1 for local angle, 1 for goal distance, 2 for linear & angular speed, 1 for angle diff
        self.action_num = 2 # linear_x and angular_z
        self.observation_space = np.empty(self.state_num)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))#np.empty(self.action_num)

        # Previous positions
        self.x_prev = [3.0, -3.0, 3.0, -3.0, 3.0, -3.0]#[0.0, 2.0] # [0.0,4.0]
        self.y_prev = [0.0, 0.0, -2.5, -2.5, 2.5, 2.5] #[0.0, -2.0]  # [0.0,0.0]

        # Set target position
        self.target = [[-3.0, 0.0], [3.0, 0.0], [-3.0, -2.5], [3.0, -2.5], [-3.0, 2.5], [3.0, 2.5]]

        # Set turtlebot index in Gazebo (to distingush from other models in the world)
        self.model_index = -1
        self.model_state = ModelStates()
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # Miscellanous
        self.ep_len_counter = 0
        self.dis_rwd_norm = 7
        self.just_reset = [False] * self.num_robots
        self.dones = [False] * self.num_robots

    #To be done when real robots are used
    
    #def get_odom(self):

    #def print_odom(self):

    def reset(self, model_state = None, id_bots = 999):
        
        self.is_collided = False
        tb3 = [-1]*self.num_robots
        if model_state is not None:
            for i, name in enumerate(model_state.name):
                if 'tb3' in name:
                    tb3[int(name[-1])] = i

        #print("id_bots = {}, tb3_0 = {}, tb3_1 = {}".format(id_bots, tb3_0, tb3_1))
        
        # Reset Turtlebot 1 position
        state_msg1 = ModelState()
        state_msg1.model_name = 'tb3_0'
        state_msg1.pose.position.x = 3.0
        state_msg1.pose.position.y = 0.0 
        state_msg1.pose.position.z = 0.0
        state_msg1.pose.orientation.x = 0
        state_msg1.pose.orientation.y = 0
        state_msg1.pose.orientation.z = -0.2
        state_msg1.pose.orientation.w = 0

        # Reset Turtlebot 2 Position
        state_msg2 = ModelState()    
        state_msg2.model_name = 'tb3_1' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg2.pose.position.x = -3.0
        state_msg2.pose.position.y = 0.0
        state_msg2.pose.position.z = 0.0
        state_msg2.pose.orientation.x = 0
        state_msg2.pose.orientation.y = 0
        state_msg2.pose.orientation.z = 0
        state_msg2.pose.orientation.w = 0

        # Reset Turtlebot 3 Position

        state_msg3 = ModelState()    
        state_msg3.model_name = 'tb3_2' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg3.pose.position.x = 3.0
        state_msg3.pose.position.y = -2.5
        state_msg3.pose.position.z = 0.0
        state_msg3.pose.orientation.x = 0
        state_msg3.pose.orientation.y = 0
        state_msg3.pose.orientation.z = -0.2
        state_msg3.pose.orientation.w = 0

        # Reset Turtlebot 4 Position

        state_msg4 = ModelState()    
        state_msg4.model_name = 'tb3_3' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg4.pose.position.x = -3.0
        state_msg4.pose.position.y = -2.5
        state_msg4.pose.position.z = 0.0
        state_msg4.pose.orientation.x = 0
        state_msg4.pose.orientation.y = 0
        state_msg4.pose.orientation.z = 0
        state_msg4.pose.orientation.w = 0

        # Reset Turtlebot 5 Position

        state_msg5 = ModelState()    
        state_msg5.model_name = 'tb3_4' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg5.pose.position.x = 3.0
        state_msg5.pose.position.y = 2.5
        state_msg5.pose.position.z = 0.0
        state_msg5.pose.orientation.x = 0
        state_msg5.pose.orientation.y = 0
        state_msg5.pose.orientation.z = -0.2
        state_msg5.pose.orientation.w = 0

        # Reset Turtlebot 6 Position

        state_msg6 = ModelState()    
        state_msg6.model_name = 'tb3_5' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_msg6.pose.position.x = -3.0
        state_msg6.pose.position.y = 2.5
        state_msg6.pose.position.z = 0.0
        state_msg6.pose.orientation.x = 0
        state_msg6.pose.orientation.y = 0
        state_msg6.pose.orientation.z = 0
        state_msg6.pose.orientation.w = 0


        rospy.wait_for_service('gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            if id_bots == 999 or id_bots == tb3[0]:
                resp = set_state(state_msg1)
                self.dones[0] = False
            if id_bots == 999 or id_bots == tb3[1]:
                resp2 = set_state(state_msg2)
                self.dones[1] = False
            if id_bots == 999 or id_bots == tb3[2]:
                resp3 = set_state(state_msg3)
                self.dones[2] = False
            if id_bots == 999 or id_bots == tb3[3]:
                resp4 = set_state(state_msg4)
                self.dones[3] = False
            if id_bots == 999 or id_bots == tb3[4]:
                resp5 = set_state(state_msg5)
                self.dones[4] = False
            if id_bots == 999 or id_bots == tb3[5]:
                resp6 = set_state(state_msg6)
                self.dones[5] = False
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        initial_state = np.zeros((self.num_robots, self.state_num))
        
        self.just_reset = True

        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0
        # if id_bots == 999:
        #     for i in range(self.num_robots):
        #         self.pub_tb3[i].publish(self.move_cmd)
        # else:
        #     self.pub_tb3[]
        if id_bots == 999 or id_bots == tb3[0]:
            self.pub_tb3[0].publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3[1]:
            self.pub_tb3[1].publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3[2]:
            self.pub_tb3[2].publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3[3]:
            self.pub_tb3[3].publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3[4]:
            self.pub_tb3[4].publish(self.move_cmd)
        if id_bots == 999 or id_bots == tb3[5]:
            self.pub_tb3[5].publish(self.move_cmd)
        self.rate.sleep()

        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)
        self.dones = [False] * self.num_robots

        return range(0, self.num_robots), initial_state

        # When turtlebot is collided with wall, obstacles etc - need to reset
        # def turtlebot_collsion(self):

    def action_to_twist(self, action):
        t = Twist()

        # Rescale and clipping the actions
        t.linear.x = action[0]*0.3
        t.linear.x = min(1, max(-1, t.linear.x))
        
        t.angular.z = min(pi/2, max(-pi/2, action[1]))
        return t
    
    def posAngle(self, model_state):
        pose = [None]*self.num_robots
        ori = [None]*self.num_robots
        x = [None]*self.num_robots
        y = [None]*self.num_robots
        angles = [None]*self.num_robots
        theta = [None]*self.num_robots
        tb3 = [-1]*self.num_robots
        if model_state is not None:
            for i, name in enumerate(model_state.name):
                if 'tb3' in name:
                    tb3[int(name[-1])] = i

        tb3_pose = [model_state.pose[i] for i in tb3]
        for i in range(self.num_robots):
            # Write relationship between i and the index
            pose[i] = tb3_pose[i] # Need to find the better way to assign index for each robot
            ori[i] = pose[i].orientation
            x[i] = pose[i].position.x
            y[i] = pose[i].position.y
            angles[i] = tf.transformations.euler_from_quaternion((ori[i].x, ori[i].y, ori[i].z, ori[i].w))
            theta[i] = angles[i][2]
        idx = [id for id in tb3]
        return x, y, theta, idx
    
    def obstacleFind(self, model_state):
        
        x = [None]*self.num_obstacles
        y = [None]*self.num_obstacles
        obs = []
        types = []

        if model_state is not None:
            for i, name in enumerate(model_state.name):
                if 'cylinder' in name:
                    obs.append(i)
                    types.append('cyl')
                if 'box' in name:
                    obs.append(i)
                    types.append('box')
        obs_pose = [model_state.pose[i] for i in obs]
        for i in range(self.num_obstacles):
            x[i] = obs_pose[i].position.x
            y[i] = obs_pose[i].position.y

        return x, y, obs, types


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
        # 20201010 How can I make the action input results in the change in state?
        # I read tensorswarm, and it takes request and go one step.
        # It waited until m_loop_done is True - at the end of the post step.
        
        # 0. Initiliasation
        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        
        #print("Actions form network: {}".format(np.asarray(actions).shape))
        twists = [self.action_to_twist(action) for action in np.asarray(actions)]
        twists_rsc = [Twist()]*self.num_robots

        # rescaling the action
        for i in range(len(twists)):
            twists[i].linear.x = (twists[i].linear.x) # only forward motion
            twists[i].angular.z = twists[i].angular.z
        linear_x = [i.linear.x for i in twists]
        angular_z = [i.angular.z for i in twists]
        dones = self.dones
        
        # position of turtlebot before taking steps
        x_prev = self.x_prev
        y_prev = self.y_prev
        distance_to_goals_prv = [None]*self.num_robots
        for i in range(self.num_robots):
            distance_to_goals_prv[i] = sqrt((x_prev[i]-self.target[i][0])**2+(y_prev[i]-self.target[i][1])**2)

        # 1. Move robot with the action input for time_step
        while (record_time_step < time_step):
            for i in range(self.num_robots):
                self.pub_tb3[i].publish(twists[i])

            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time
        
        # 2. Read the position and angle of robot
        model_state = self.pose_ig.get_msg()
        self.model_state = model_state
        #print("Model State: {}".format(model_state))
        x, y, theta, idx = self.posAngle(model_state)
        self.x_prev = x
        self.y_prev = y
        
        x_obs, y_obs, idx_obs, types_obs = self.obstacleFind(model_state)
        

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
        # rospy.wait_for_service('phero_read')
        # try:
        #     phero_read = rospy.ServiceProxy('phero_read', PheroRead)
        #     resp = phero_read(x, y)
        # except rospy.ServiceException as e:
        #     print("Service Failed %s"%e)
        #print([wow.data for wow in resp.value])
        state = self.phero_ig.get_msg()
        phero_vals = [phero.data for phero in state.values]
        #print("Phero_vals: {}".format(phero_vals))
        #phero_rev = self.swap2elements([phero.data for phero in resp.value])  # To read each other's pheromone
        #phero_vals = [phero for phero in phero_rev]
        
        # Concatenating the state array
        state_arr = np.asarray(phero_vals)
        state_arr = np.hstack((state_arr, np.asarray(theta).reshape(self.num_robots, 1)))
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
                        distance_rewards[i] = goal_progress[i]
                else:
                        distance_rewards[i] = goal_progress[i]
            else:
                distance_rewards[i] = 0.0
            distance_rewards[i] *= time_step_factor
        # for i in range(self.num_robots):
        #     if dones[i] == True:
        #         distance_rewards[i] = 0.0
        
        #self.just_reset == False
        
        ## 6.2. Pheromone reward (The higher pheromone, the lower reward)
        #phero_sums = [np.sum(phero_val) for phero_val in phero_vals]
        #[-phero_sum*2 for phero_sum in phero_sums] # max phero_r: 0, min phero_r: -9
        
        ## 6.3. Goal reward
        ### Reset condition is activated when both two robots have arrived their goals 
        ### Arrived robots stop and waiting
        for i in range(self.num_robots):
            if distance_to_goals[i] <= 0.5:
                goal_rewards[i] = 150.0
                dones[i] = True
                self.reset(model_state, id_bots=idx[i])

            
        

        ## 6.4. Angular speed penalty
        for i in range(self.num_robots):
            if abs(angular_z[i])>0.8:
                angular_punish_rewards[i] = -1
                if dones[i] == True:
                    angular_punish_rewards[i] = 0.0
        
        ## 6.5. Linear speed penalty
        for i in range(self.num_robots):
            if linear_x[i] < 0.2:
                linear_punish_rewards[i] = -0.0
        for i in range(self.num_robots):
            if dones[i] == True:
                linear_punish_rewards[i] = 0.0
        ## 6.6. Collision penalty
        #   if it collides to walls, it gets penalty, sets done to true, and reset
        #   it needs to be rewritten to really detect collision

        distance_btw_robots = np.ones([self.num_robots, self.num_robots])
        distance_to_obstacle = np.ones([self.num_robots, self.num_obstacles])
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if j != i:
                    distance_btw_robots[i][j] = sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) # Python 
            for k in range(self.num_obstacles):
                distance_to_obstacle[i][k] = sqrt((x[i]-x_obs[k])**2+(y[i]-y_obs[k])**2)



        collision_rewards = [0.0]*self.num_robots
        for i in range(self.num_robots):
            if any([dis <= 0.32 for dis in distance_btw_robots[i]]) == True:
                print("Collision! Robot: {}".format(i))
                collision_rewards[i] = -20.0
                dones[i] = True
                self.reset(model_state, id_bots=idx[i])
            if any([dis <= 0.35 for dis in distance_to_obstacle[i]]) == True:
                collision_rewards[i] = -20.0
                dones[i] = True
                self.reset(model_state, id_bots=idx[i])
            # for j in range(self.num_obstacles):
            #     if types_obs[j] == 'cyl' and any(distance_to_obstacle[i] < 0.4:
            #         collision_rewards[i] = -60.0
            #         dones[i] = True
            #         self.reset(model_state, id_bots=idx[i])
            #     elif types_obs[j] == 'box' and distance_to_obstacle[i][j] < 0.4:
            #         collision_rewards[i]
        
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
                    ooa_rewards[i] = -30.0
                dones[i] = True
                self.reset(model_state, id_bots=idx[i])
            
        

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
        # if all(flag == True for flag in dones) == True:
        #     self.reset(model_state, id_bots=999)
        #     for i in range(self.num_robots):
        #         dones[i] = False

        self.dones = dones
        #print("distance reward: {}".format(distance_reward*(3/time_step)))
        #print("phero_reward: {}".format(phero_reward))
        # if linear_x > 0.05 and angular_z > 0.05 and abs(distance_reward) > 0.005:
        #     self.stuck_indicator = 0
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
        return range(0, self.num_robots), states, rewards, dones, infos
        
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