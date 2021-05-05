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
from tf.transformations import quaternion_from_euler

import time
import csv
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
        self.num_robots = 1

        # Node initialisation
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
        self.state_num = 8 # 9 for pheromone 1 for goal distance, 2 for linear & angular speed, 1 for angle diff
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

        # File name
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.file_name = "rl_{}_{}".format(self.num_robots, self.time_str)
        self.traj_name = "{}_traj".format(self.file_name)
        print(self.file_name)

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
        self.done = False

        self.is_traj = True

        # Log related

        self.log_timer = time.time()
        self.reset_timer = time.time()
        self.positions = []
        for i in range(self.num_robots):
            self.positions.append([])
        print("positions: {}".format(self.positions))
        self.traj_eff = list()

    def reset(self):
        '''
        Resettng the Experiment
        1. Counter Update
        2. Update the counter based on the flag from step
        3. Assign next positions and reset the positions of robots and targets
        '''

        # ========================================================================= #
	    #                          1. COUNTER UPDATE                                #
	    # ========================================================================= #
        
        #Increment Collision Counter
        if self.is_collided == True:
            print("Collision!")
            self.counter_collision += 1
            self.counter_step += 1

        # Increment Arrival Counter and store the arrival time
        if self.is_goal == True:
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
            for i in range(self.num_robots):
                for j in range(len(self.positions[i])-1):
                    distance_t = sqrt((self.positions[i][j+1][0] - self.positions[i][j][0])**2 + (self.positions[i][j+1][1] - self.positions[i][j][1])**2)
                    if distance_t <= 0.5:
                        total_distance[i] += distance_t
                pure_distance[i] = sqrt((self.positions[i][0][0] - self.positions[i][-1][0])**2 + (self.positions[i][0][1] - self.positions[i][-1][1])**2)

            avg_distance_traj = np.average(total_distance)
            avg_distance_pure = np.average(pure_distance)
            traj_efficiency = avg_distance_pure/avg_distance_traj
            print("Step: {}, avg_distance_traj: {}".format(self.counter_step, avg_distance_traj))
            #print("self.positions: {}".format(self.positions))
            #print("Total Distance: {}".format(total_distance))
            print("avg_distance_pure: {}, traj_efficiency: {}".format(avg_distance_pure, traj_efficiency))
            #print("distance_t: {}".format(distance_t))

            self.traj_eff.append(traj_efficiency)

        if self.is_timeout == True:
            self.counter_timeout += 1
            self.counter_step += 1
            print("Timeout!")

        # Reset the flags
        self.is_collided = False
        self.is_goal = False
        self.is_timeout = False
 
        # ========================================================================= #
	    #                           2. TARGET UPDATE                                #
	    # ========================================================================= #

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
	    #                                 3. RESET                                  #
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

        # Reset Pheromone Grid
        #
        #

        rospy.wait_for_service('gazebo/reset_simulation')

        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            resp_targ = set_state(state_target_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        initial_state = np.zeros(self.state_num).reshape(1,self.state_num)

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
            if (self.counter_collision != 0 or self.counter_success != 0):
                succ_percentage = 100*self.counter_success/(self.counter_success+self.counter_collision+self.counter_timeout)
                col_percentage = 100*self.counter_collision/(self.counter_success+self.counter_collision+self.counter_timeout)
                tout_percentage = 100*self.counter_timeout/(self.counter_success+self.counter_collision+self.counter_timeout)
            else:
                succ_percentage = 0
                col_percentage = 0
                tout_percentage = 0
            print("Counter: {}".format(self.counter_step))
            print("Success Counter: {}".format(self.counter_success))
            print("Collision Counter: {}".format(self.counter_collision))
            print("Timeout Counter: {}".format(self.counter_timeout))

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
                print("Successfully Logged.")
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
        

        self.done = False
        return range(0, self.num_robots), initial_state

    def step(self, time_step=0.1, linear_x=0.2, angular_z=0.0):

        # 0. Initiliasation
        start_time = time.time()
        record_time = start_time
        record_time_step = 0

        # rescaling the action
        linear_x = linear_x*0.3
        linear_x = min(1, max(-1, linear_x))
        linear_x = (linear_x+1)*1/2
        angular_z = min(pi/2, max(-pi/2, angular_z*0.9))
        

        self.move_cmd.linear.x = linear_x
        self.move_cmd.angular.z = angular_z
        action = np.array([linear_x, angular_z])
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

        step_time = time.time()
        episode_time = step_time - self.reset_timer
        # 2. Read the position and angle of robot
        model_state = self.pose_ig.get_msg()
        pose = model_state.pose[self.model_index]
        ori = pose.orientation
        x = pose.position.x
        y = pose.position.y
        angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        theta = angles[2]

        step_timer = time.time()
        reset_time = step_timer - self.reset_timer
        
        # Log Positions
        if time.time() - self.log_timer > 0.5 and self.is_traj == True:
            for i in range(self.num_robots):
                with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.traj_name), mode='a') as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(['%0.1f'%reset_time, '%i'%i, '%0.2f'%x, '%0.2f'%y])
                print("positions[i]: {}".format(self.positions[i]))
                self.positions[i].append([x,y])
            self.log_timer = time.time()

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
        
        #print("phero_grad: {}".format(phero_grad))
        state_arr = phero_grad
        state_arr = np.append(state_arr, phero_now)
        state_arr = np.append(state_arr, distance_to_goal)
        state_arr = np.append(state_arr, angle_diff)
        state_arr = np.append(state_arr, linear_x)
        state_arr = np.append(state_arr, angular_z)
        state = state_arr.reshape(self.state_num)

        # 6. Reward assignment
        ## 6.0. Initialisation of rewards
        distance_reward = 0.0
        phero_reward = 0.0
        goal_reward = 0.0
        
        ## 6.1. Distance Reward
        # goal_progress = distance_to_goal_prv - distance_to_goal
        # if goal_progress >= 0:
        #     distance_reward = goal_progress *1.2
        # else:
        #     distance_reward = goal_progress
        
        ## 6.2. Pheromone reward (The higher pheromone, the lower reward)
        #phero_sum = np.sum(phero_vals)
        phero_reward = 0.0 #(-phero_sum) # max phero_r: 0, min phero_r: -9
        #print("------------------------")
        #print("State: {}".format(phero_vals))
        ## 6.3. Goal reward
        if distance_to_goal <= 0.45:
            self.is_goal = True
            self.done = True
            #self.reset()
            time.sleep(1)

        ## 6.4. Angular speed penalty
        # angular_punish_reward = 0.0
        # if abs(angular_z_rsc) > 0.8:
        #     angular_punish_reward = -1
        
        ## 6.5. Linear speed penalty
        # linear_punish_reward = 0.0
        # if linear_x_rsc < 0.2:
        #     linear_punish_reward = -1
        ## 6.6. Collision penalty
        #   if it collides to walls, it gets penalty, sets done to true, and reset
        #collision_reward = 0.0
        obs_pos = [[2, 0],[-2,0],[0,2],[0,-2]]
        dist_obs = [sqrt((x-obs_pos[i][0])**2+(y-obs_pos[i][1])**2) for i in range(len(obs_pos))]
        for i in range(len(obs_pos)):
            if dist_obs[i] < 0.25:
                self.is_collided = True
                self.done = True
                time.sleep(0.5)
                
        if episode_time > 60:
            self.is_timeout = True
            self.done = True

        # 7. Reset
        ## 7.1. when robot goes too far from the target
        # if distance_to_goal >= self.dis_rwd_norm:
        #     self.reset()
        #     time.sleep(0.5) 

        ## 7.2. when the robot is out of the pheromone grid
        if abs(x) >= 4.7 or abs(y) >= 4.7:
            print("Out of range!")
            self.is_collided = True
            self.done = True
    
        if self.done == True:
            self.reset()

        #print("distance reward: {}".format(distance_reward*(3/time_step)))
        #print("phero_reward: {}".format(phero_reward))
        # if linear_x > 0.05 and angular_z > 0.05 and abs(distance_reward) > 0.005:
        #     self.stuck_indicator = 0

        reward = [0.0]#distance_reward*(4/time_step) + phero_reward + goal_reward + angular_punish_reward + linear_punish_reward + collision_reward
        reward = np.asarray(reward).reshape(1)
        info = [{"episode": {"l": self.ep_len_counter, "r": reward}}]
        self.ep_len_counter = self.ep_len_counter + 1
        # print("-------------------")
        # print("Ep: {}".format(self.ep_len_counter))
        # print("Target: ({}, {})".format(self.target_x, self.target_y))
        # print("Distance R: {}".format(distance_reward*(4/time_step)))
        # print("Phero R: {}".format(phero_reward))
        # print("Goal R: {}".format(goal_reward))
        # print("Angular R: {}".format(angular_punish_reward))
        # print("Linear R: {}".format(linear_punish_reward))
        # print("Collision R: {}".format(collision_reward))
        # print("Reward: {}".format(reward))
        #print("**********************")
        #print("state: {}, action:{}, reward: {}, done:{}, info: {}".format(state, action, reward, done, info))
        return range(0, self.num_robots), state, reward, self.done, info
        
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