
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
from turtlebot3_pheromone.srv import PheroRead, PheroReadResponse
from turtlebot3_pheromone.msg import fma

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
import csv

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
        self.num_robots = 2

        # Node initialisation
        self.node = rospy.init_node('phero_turtlebot_env', anonymous=True)
        self.pose_ig = InfoGetter()
        self.phero_ig = InfoGetter()
        #self.collision_ig = InfoGetter()

        self.pub_tb3_0 = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=1)
        self.pub_tb3_1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=1)
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
        self.state_num = 8 # 9 for pheromone 1 for goal distance, 2 for linear & angular speed, 1 for angle diff
        self.action_num = 2 # linear_x and angular_z
        self.observation_space = np.empty(self.state_num)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))#np.empty(self.action_num)

        # Previous positions
        self.x_prev = [-2.5, 2.5]
        self.y_prev = [0.0, 0.0]

        # Set initial positions
        self.target = [[2.5, 0.0], [-2.5, 0.0]] # Two goal
        self.d_robots = 5
        self.target_index = 0
        self.num_experiments = 20
        self.x = [0.0]*self.num_robots
        self.y = [0.0]*self.num_robots
        self.theta = [0.0]*self.num_robots

        # Set turtlebot index in Gazebo (to distingush from other models in the world)
        self.model_index = -1
        self.model_state = ModelStates()
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # Previous pheromone data
        self.prev_phero = [None] *9
        self.prev_prev_phero = [None]*9
        # Miscellanous
        self.ep_len_counter = 0
        self.dis_rwd_norm = 7
        self.just_reset = [False] * self.num_robots
        self.dones = [False] * self.num_robots
        self.grad_sensitivity = 20

        # File name
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.file_name = "rl_{}_{}".format(self.num_robots, self.time_str)
        self.traj_name = "{}_traj".format(self.file_name)

        # Experiments
        self.isExpDone = False
   
        self.counter_step = 0
        self.counter_collision = 0
        self.counter_success = 0
        self.counter_timeout = 0
        self.arrival_time = []
        
        self.is_reset = False
        self.is_collided = False
        self.is_goal = 0
        self.is_timeout = False

        self.is_traj = True

        # Log related
        self.log_timer = time.time()
        self.positions = []
        for i in range(self.num_robots):
            self.positions.append([])
        self.traj_eff = list()

        #self.reset()
        self.reset_timer = time.time()

        

    def reset(self, model_state = None, id_bots = 3):
        '''
        Resettng the Experiment
        1. Update the counter based on the flag from step
        2. Target Update
        3. Reset robot and target
        4. Logging
        '''

        # ========================================================================= #
	    #                           0. ID ASSIGNMENT                               #
	    # ========================================================================= #

        # ID assignment 
        tb3_0 = 3
        tb3_1 = 3
        if model_state is not None:
            for i in range(len(model_state.name)):
                if model_state.name[i] == 'tb3_0':
                    tb3_0 = i
                if model_state.name[i] == 'tb3_1':
                    tb3_1 = i
        else:
            tb3_0 = -1
            tb3_1 = -2
        
        # ========================================================================= #
	    #                          1. COUNTER UPDATE                                #
	    # ========================================================================= #

        # Increment Collision Counter
        if self.is_collided == True:
            print("Collision!")
            self.counter_collision += 1
            self.counter_step += 1

        # Increment Arrival Counter and store the arrival time
        if self.is_goal == 2:
            print("Arrived goal!")
            self.counter_success += 1
            self.counter_step += 1
            arrived_timer = time.time()
            art = arrived_timer-self.reset_timer
            self.arrival_time.append(art)
            print("Episode time: %0.2f"%art)

            # Compute trajectory efficiency (how can I add outlier removal?)
            total_distance = [0.0]*self.num_robots
            pure_distance = [0.0]*self.num_robots
            print("self.positions: {}".format(self.positions))
            for i in range(self.num_robots):
                for j in range(len(self.positions[i])-1):
                    distance_t = sqrt((self.positions[i][j+1][0] - self.positions[i][j][0])**2 + (self.positions[i][j+1][1] - self.positions[i][j][1])**2)
                    if distance_t <= 0.5:
                        total_distance[i] += distance_t
                    
                pure_distance[i] = sqrt((self.positions[i][0][0] - self.positions[i][-1][0])**2 + (self.positions[i][0][1] - self.positions[i][-1][1])**2)

            avg_distance_traj = np.average(total_distance)
            avg_distance_pure = np.average(pure_distance)
            traj_efficiency = avg_distance_pure/avg_distance_traj
            #print("Step: {}, avg_distance_traj: {}".format(self.counter_step, avg_distance_traj))
            
            #print("Total Distance: {}".format(total_distance))
            #print("avg_distance_pure: {}, traj_efficiency: {}".format(avg_distance_pure, traj_efficiency))
            #print("distance_t: {}".format(distance_t))

            self.traj_eff.append(traj_efficiency)

        if self.is_timeout == True:
            #self.counter_collision += 1
            #self.counter_step += 1
            self.counter_step += 1
            self.counter_timeout += 1
            print("Timeout!")

        # Reset the flags
        self.is_collided = False
        self.is_goal = 0
        self.is_timeout = False

        # ========================================================================= #
	    #                          2. TARGET UPDATE                                 #
	    # ========================================================================= #

        # Reset position assignment
        if id_bots == 3: 
            if self.target_index < self.num_experiments-1:
                self.target_index += 1
            else:
                self.target_index = 0
                
        angle_target = self.target_index*2*pi/self.num_experiments        

        self.x[0] = (self.d_robots/2)*cos(angle_target)
        self.y[0] = (self.d_robots/2)*sin(angle_target)

        self.x[1] = (self.d_robots/2)*cos(angle_target+pi)
        self.y[1] = (self.d_robots/2)*sin(angle_target+pi)

        self.theta[0] = angle_target + pi
        self.theta[1] = angle_target 

        quat1 = quaternion_from_euler(0,0,self.theta[0])
        quat2 = quaternion_from_euler(0,0,self.theta[1])
        
        self.target = [[self.x[1], self.y[1]], [self.x[0], self.y[0]]]
        
        # ========================================================================= #
	    #                                2. RESET                                   #
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
        state_target_msg = ModelState()    
        state_target_msg.model_name = 'tb3_1' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_target_msg.pose.position.x = self.x[1]
        state_target_msg.pose.position.y = self.y[1]
        state_target_msg.pose.position.z = 0.0
        state_target_msg.pose.orientation.x = quat2[0]
        state_target_msg.pose.orientation.y = quat2[1]
        state_target_msg.pose.orientation.z = quat2[2]
        state_target_msg.pose.orientation.w = quat2[3]

        rospy.wait_for_service('gazebo/reset_simulation')

        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            if id_bots == 3 or id_bots == tb3_0:
                resp = set_state(state_msg)
            if id_bots == 3 or id_bots == tb3_1:
                resp_targ = set_state(state_target_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        initial_state = np.zeros((self.num_robots, self.state_num))
        
        self.just_reset = True

        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0
        if id_bots == 3 or id_bots == tb3_0:
            self.pub_tb3_0.publish(self.move_cmd)
        if id_bots == 3 or id_bots == tb3_1:
            self.pub_tb3_1.publish(self.move_cmd)
        self.rate.sleep()

        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)

        self.dones = [False] * self.num_robots

         # ========================================================================= #
	    #                                 4. LOGGING                                #
	    # ========================================================================= #

        if self.counter_step == 0:
            with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.file_name), mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Episode', 'Success Rate', 'Average Arrival time', 'std_at', 'Collision Rate', 'Timeout Rate', 'Trajectory Efficiency', 'std_te'])
            if self.is_traj == True:
                with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.traj_name), mode='w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(['time', 'ID', 'x', 'y'])


        if self.counter_step != 0:
            if (self.counter_collision != 0 or self.counter_success != 0 or self.counter_timeout !=0):
                succ_percentage = 100*self.counter_success/(self.counter_success+self.counter_collision+self.counter_timeout)
                col_percentage = 100*self.counter_collision/(self.counter_success+self.counter_collision+self.counter_timeout)
                tout_percentage = 100*self.counter_timeout/(self.counter_success+self.counter_collision+self.counter_timeout)
            else:
                succ_percentage = 0
                col_percentage = 0
                tout_percentage = 0
            print("Counter: {}".format(self.counter_step))
            print("Success: {}, Collision: {}, Timeout: {}".format(self.counter_success, self.counter_collision, self.counter_timeout))

        if (self.counter_step % 1 == 0 and self.counter_step != 0):
            print("Success Rate: {}%".format(succ_percentage))

        if (self.counter_step % 100 == 0 and self.counter_step != 0):
            avg_comp = np.average(np.asarray(self.arrival_time))
            std_comp = np.std(np.asarray(self.arrival_time))
            avg_traj = np.average(np.asarray(self.traj_eff))
            std_traj = np.std(np.asarray(self.traj_eff))
            print("{} trials ended. Success rate: {}, average completion time: {}, Standard deviation: {}, Collision rate: {}, Timeout Rate: {}".format(self.counter_step, succ_percentage, avg_comp, std_comp, col_percentage, tout_percentage))
            with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.file_name), mode='a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['%i'%self.counter_step, '%0.2f'%succ_percentage, '%0.2f'%avg_comp, '%0.2f'%std_comp, '%0.2f'%col_percentage, '%0.2f'%tout_percentage, '%0.4f'%avg_traj, '%0.4f'%std_traj])
            self.arrival_time = list()
            self.traj_eff = list()
            self.counter_collision = 0
            self.counter_success = 0
            self.counter_timeout = 0
            self.target_index = 0

        self.positions = []
        for i in range(self.num_robots):
            self.positions.append([])
        self.reset_timer = time.time()
        self.reset_timer = time.time()

        return range(0, self.num_robots), initial_state


    def action_to_twist(self, action):
        '''
        Convert Actions (2D array) to Twist (geometry.msgs)
        '''
        t = Twist()

        # Rescale and clipping the actions
        
        t.linear.x = action[1]*0.3
        t.linear.x = min(1, max(-1, t.linear.x))
        
        #print("t.angular.z before conversion: {}".format(action[0]))
        t.angular.z = min(1, max( -1, action[0]))
        #print("t.angular.z after conversion: {}".format(t.angular.z))
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
        tb3_pose = [model_state.pose[tb3_0], model_state.pose[tb3_1]]
        for i in range(self.num_robots):
            # Write relationship between i and the index
            pose[i] = tb3_pose[i] # Need to find the better way to assign index for each robot
            ori[i] = pose[i].orientation
            x[i] = pose[i].position.x
            y[i] = pose[i].position.y
            angles[i] = tf.transformations.euler_from_quaternion((ori[i].x, ori[i].y, ori[i].z, ori[i].w))
            theta[i] = angles[i][2]
        idx = [tb3_0, tb3_1]
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
        
        twists = [self.action_to_twist(action) for action in np.asarray(actions)]

        # rescaling the action
        for i in range(len(twists)):
            twists[i].linear.x = (twists[i].linear.x+1)*1/2 # only forward motion
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


        # Collect previous pheromone data
        state = self.phero_ig.get_msg()
        phero_prev = [phero.data for phero in state.values]

        # 1. Move robot with the action input for time_step
        while (record_time_step < time_step):
            self.pub_tb3_0.publish(twists[0])
            self.pub_tb3_1.publish(twists[1])

            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time
        
        step_time = time.time()
        episode_time = step_time - self.reset_timer

        # 2. Read the position and angle of robot
        model_state = self.pose_ig.get_msg()
        self.model_state = model_state
        x, y, theta, idx = self.posAngle(model_state)
        self.x_prev = x
        self.y_prev = y

        step_timer = time.time()
        reset_time = step_timer - self.reset_timer
        
        # # Log Positions
        if time.time() - self.log_timer > 0.5 and self.is_traj == True:
            for i in range(self.num_robots):
                with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.traj_name), mode='a') as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(['%0.1f'%reset_time, '%i'%i, '%0.2f'%x[i], '%0.2f'%y[i]])
                self.positions[i].append([x[i],y[i]])
            self.log_timer = time.time()

        # 3. Calculate the distance & angle difference to goal 
        distance_to_goals = [None]*self.num_robots
        global_angle = [None]*self.num_robots
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
        states = state_arr.reshape(self.num_robots, self.state_num)

        
        # 5. Reward assignment
        for i in range(self.num_robots):
            if distance_to_goals[i] <= 0.6 and self.dones[i] == False:
                self.is_goal += 1
                self.dones[i] = True
                print("goal?")

        #   if it collides to walls, it gets penalty, sets done to true, and reset
        distance_btw_robots = sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)
        collision_rewards = [0.0]*self.num_robots
        if distance_btw_robots <= 0.3 and self.dones[i] == False:
            print("Collision!")
            self.is_collided = True
            for i in range(self.num_robots):
                self.dones[i] = True
            #self.reset(model_state, id_bots=3)
        
        reset_flag = False
        for i in range(self.num_robots):
            if abs(x[i]) >= 5.7 or abs(y[i]) >= 5.7:
                print("Out of range!")
                self.is_collided = True
                self.dones[i] = True
                if reset_flag == False:
                    #self.reset(model_state, id_bots=3)
                    reset_flag = True
                    print("outofrange?")
        if episode_time > 60:
            self.is_timeout = True
            for i in range(self.num_robots):
                self.dones[i] = True
                print("timeout?")
        
        ## 7.4. If all the robots are done with tasks, reset
        if all(flag == True for flag in self.dones) == True:
            # self.is_goal = True
            self.reset(model_state, id_bots=3)
            # for i in range(self.num_robots):
            #     dones[i] = False


        # self.dones = dones
        rewards = [0.0, 0.0]
        #test_time2 = time.time()
        rewards = np.asarray(rewards).reshape(self.num_robots)
        infos = [{"episode": {"l": self.ep_len_counter, "r": rewards}}]
        self.ep_len_counter = self.ep_len_counter + 1
        #print("-------------------")
        return range(0, self.num_robots), states, rewards, self.dones, infos, self.isExpDone

if __name__ == '__main__':
    try:
        sess = tensorflow.Session()
        K.set_session(sess)
        env = Env()
    except rospy.ROSInterruptException:
        pass