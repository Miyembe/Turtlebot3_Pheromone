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
from tf.transformations import quaternion_from_euler
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


class WaypointNavigation:

    MAX_FORWARD_SPEED = 0.5
    MAX_ROTATION_SPEED = 0.5

    # Tunable parameters
    wGain = 10
    vConst = 0.5 #0.2
    distThr = 0.2
    pheroThr = 0.2

    
    def __init__(self):
      
        self.num_robots = 4
        self.num_obstacles = 1
        # Initialise pheromone values
        self.phero = [[0.0] * 9] * self.num_robots
        self.phero_sum = [0.0] * self.num_robots
        
        # Initialise speed
        self.move_cmd = [Twist()]*self.num_robots

        # Initialise positions
        #self.positions = [[-2.5,0], [2.5,0]]
        #self.goal = [[2.5,0.0], [0,0]]
        self.obstacle = [0,0]
        self.dones = [False]*self.num_robots

        # Set initial positions
        self.target = [[2.5, 0.0], [-2.5, 0.0], [0.0, 2.5], [0.0, -2.5]] # Two goal
        self.d_robots = 5.0
        self.target_index = 0
        self.num_experiments = 20
        self.x = [0.0]*self.num_robots
        self.y = [0.0]*self.num_robots
        self.theta = [0.0]*self.num_robots

        # File name
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.file_name = "manual_{}_{}".format(self.num_robots, self.time_str)
        print(self.file_name)

        # Initialise parameters
        
        self.step_size = 0.1
        #self.b_range = np.arange(0, 1+self.step_size, self.step_size)
        self.v_range = np.arange(0.2, 1+self.step_size, self.step_size)
        self.w_range = np.arange(0.2, 1+self.step_size, self.step_size)

        self.BIAS = 0.3
        self.V_COEF = 0.2#self.v_range[0]
        self.W_COEF = 0.2#self.w_range[0]

        #self.b_size = self.b_range.size        
        self.v_size = self.v_range.size
        self.w_size = self.w_range.size
        
        self.b_counter = 0
        self.v_counter = 0
        self.w_counter = 0

        # Initialise ros related topics
        rospy.init_node('manual_exp3')
        self.pose_ig = InfoGetter()
        self.phero_ig = InfoGetter()
        self.pub_tb3 = [None]*self.num_robots
        for i in range(self.num_robots):
            self.pub_tb3[i] = rospy.Publisher('/tb3_{}/cmd_vel'.format(i), Twist, queue_size=1)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose_ig, queue_size=1, buff_size=2**24)
        self.sub_phero = rospy.Subscriber('/phero_value', fma, self.phero_ig, queue_size = 1)

        # Initialise simulation
        self.counter_step = 0
        self.counter_collision = 0
        self.counter_success = 0
        self.counter_timeout = 0
        self.arrival_time = []
        
        self.is_reset = False
        self.is_collided = False
        self.is_goal = False
        self.is_timeout = False

        self.reset_timer = time.time()
        self.collision_timer = time.time()
        self.rate = rospy.Rate(10)

        # antenna movement related parameters

        self.b_range = np.arange(0.9, 0.9+self.step_size, self.step_size)
        self.s_range = np.arange(1.0, 1.9+self.step_size, self.step_size)

        self.b_size = self.b_range.size
        self.s_size = self.s_range.size

        self.b_counter = 0
        self.s_counter = 0

        self.beta_const = self.b_range[0]
        self.sensitivity = self.s_range[0]

        # Experiments related
        self.num_experiments = 20
        self.d_robots = 5

        self.reset()

    # def RandomTwist(self):

    #     twist = Twist()
    #     twist.linear.x = self.MAX_FORWARD_SPEED * random.random()
    #     twist.angular.z = self.MAX_ROTATION_SPEED * (random.random() - 0.5)
    #     print(twist)
    #     return twist
    def ReadPhero(self, message): 
        phero_data = [phero.data for phero in message.values]
        self.phero = phero_data
        self.phero_sum = [np.sum(np.asarray(phero)) for phero in phero_data]

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

    def Computation(self):
        '''
        Main Function
        - Receive position data
        - Generate action
        '''
        # ========================================================================= #
	    #                           Initialisation                                  #
	    # ========================================================================= #
        main_start_timer = time.time()
        pub_msg = self.move_cmd
        goal = self.target
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
        oris = [pose.orientation for pose in poses]
        angles = [tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w)) for ori in oris]
        thetas = [angle[2] for angle in angles]

        # Get obstacle positions
        x_obs, y_obs, idx_obs, types_obs = self.obstacleFind(message)

        # P controller
        v = [0.0]*self.num_robots
        w = [0.0]*self.num_robots

        step_timer = time.time()
        reset_time = step_timer - self.reset_timer
        self.ReadPhero(self.phero_ig.get_msg())

        # ========================================================================= #
	    #                             Calculate Distances                           #
	    # ========================================================================= #
        
        distances = [None]*self.num_robots
        distance_btw_robots = np.ones([self.num_robots, self.num_robots])
        distance_to_obstacle = np.ones([self.num_robots, self.num_obstacles])
        for i in range(self.num_robots):
            distances[i] = sqrt((x[i]-goal[i][0])**2+(y[i]-goal[i][1])**2)
            for j in range(self.num_robots):
                if j != i:
                    distance_btw_robots[i][j] = sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) 
            for k in range(self.num_obstacles):
                distance_to_obstacle[i][k] = sqrt((x[i]-x_obs[k])**2+(y[i]-y_obs[k])**2)

        # ========================================================================= #
	    #                          Action & State assignment                        #
	    # ========================================================================= #
        
        if any([dis <= 0.34 for dis in distance_btw_robots[i]]) == True:
            self.is_collided = True
            self.reset()
        elif any([dis <= 0.35 for dis in distance_to_obstacle[i]]) == True:
            self.is_collided = True
            self.reset()

        for i in range(self.num_robots):            
            if (self.phero_sum[i] > self.pheroThr):
                pub_msg[i] = self.PheroResponse(self.phero[i])
                v[i] = pub_msg[i].linear.x
                w[i] = pub_msg[i].angular.z

            # Adjust velocities
            elif (distances[i] > self.distThr):
                v[i] = self.vConst
                yaw = atan2(goal[i][1]-poss[i].y, goal[i][0]-poss[i].x)
                u = yaw - thetas[i]
                bound = atan2(sin(u), cos(u))
                w[i] = min(1.0, max(-1.0, self.wGain*bound))
                pub_msg[i].linear.x = v[i]
                pub_msg[i].angular.z = w[i]

                if self.is_reset == True:
                    self.is_reset = False
            elif (distances[i] <= self.distThr and reset_time > 1):
                self.dones[i] = True
                pub_msg[i] = Twist()
        #print("v: {}".format(v))
        
        if reset_time > 60.0:
            print("Times up!")
            self.is_timeout = True
            self.reset()
        
        if all(done == True for done in self.dones) == True:
            self.is_goal = True
            self.dones = [False]*self.num_robots
            self.reset()

        # Publish velocity 
        for i in range(self.num_robots):
            self.pub_tb3[i].publish(pub_msg[i])
        #print("pose: {}".format(poss))
        
        self.rate.sleep()
        self.is_reset = False
        main_end_timer = time.time()
        #print("comp time: {}".format(main_end_timer - main_start_timer))
    
    def velCoef(self, value1, value2):
        '''
        - val_avg (0, 1)
        - val_dif (-1, 1)
        - dif_coef (1, 2.714)
        - coefficient (-2.714, 2.714)
        '''
        val_avg = (value1 + value2)/2
        val_dif = value1 - value2
        dif_coef = exp(val_avg)
        coefficient = dif_coef*val_dif
        
        return coefficient
        
    def PheroOA(self, phero):
        '''
        Pheromone-based obstacle avoidance algorithm
        - Input: 9 cells of pheromone
        - Output: Twist() to avoid obstacle
        '''
        # Constants:
        BIAS = self.BIAS #0.1 -successful set
        V_COEF = self.V_COEF #0.35                 # How fast it will go forward depending on the avg pheromone value
        W_COEF = self.W_COEF #0.9                   # How fast it will turn around depending on the pheromone difference
        
        # Initialise values
        avg_phero = np.average(np.asarray(phero)) # values are assigned from the top left (135 deg) to the bottom right (-45 deg) ((0,1,2),(3,4,5),(6,7,8))
        unit_vecs = np.asarray([[1,0], [sqrt(2)/2, sqrt(2)/2], [0,1], [-sqrt(2)/2, sqrt(2)/2]])
        vec_coefs = [0.0] * self.num_robots
        twist = Twist()
        
        # Calculate vector weights
        vec_coefs[0] = self.velCoef(phero[5], phero[3])
        vec_coefs[1] = self.velCoef(phero[2], phero[6])
        vec_coefs[2] = self.velCoef(phero[1], phero[7])
        vec_coefs[3] = self.velCoef(phero[0], phero[8])
        vec_coefs = np.asarray(vec_coefs).reshape(4,1)
        vel_vecs = np.multiply(unit_vecs, vec_coefs)
        vel_vec = np.sum(vel_vecs, axis=0)
        theta = atan2(vel_vec[1], vel_vec[0]) 
        ang_vel = W_COEF*theta

        # Velocity assignment
        twist.linear.x = 0.5*(BIAS + V_COEF*avg_phero)
        twist.angular.z = ang_vel

        return twist
    
    def PheroResponse(self, phero):
        # takes two pheromone input from antennae
        
        avg_phero = np.average(np.asarray(phero))
        beta = self.beta_const - avg_phero
        s_l = beta - (phero[0] - phero[1])/self.sensitivity
        s_r = beta - (phero[1] - phero[0])/self.sensitivity
        twist = Twist()

        twist.linear.x = (s_l + s_r)/2
        twist.angular.z = (s_l - s_r)

        return twist
    
    def reset(self):
        '''
        Resettng the Experiment
        1. Update the counter based on the flag from step
        2. Assign next positions and reset
        3. Log the result in every selected time-step
        '''


        # ========================================================================= #
	    #                           COUNTER UPDATE                                  #
	    # ========================================================================= #
        # Increment Collision Counter
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

        if self.is_timeout == True:
            self.counter_timeout += 1
            self.counter_step += 1
            print("Timeout!")

        # Reset the flags
        self.is_collided = False
        self.is_goal = False
        self.is_timeout = False

        # ========================================================================= #
	    #                                  RESET                                    #
	    # ========================================================================= #

        # if self.target_index < self.num_experiments-1:
        #     self.target_index += 1
        # else:
        #     self.target_index = 0
            
        # angle_target = self.target_index*2*pi/self.num_experiments        

        # self.x[0] = (self.d_robots/2)*cos(angle_target)
        # self.y[0] = (self.d_robots/2)*sin(angle_target)

        # self.x[1] = (self.d_robots/2)*cos(angle_target+pi)
        # self.y[1] = (self.d_robots/2)*sin(angle_target+pi)

        # self.target = [[self.x[1], self.y[1]], [self.x[0], self.y[0]]]

        # self.theta[0] = angle_target + pi
        # self.theta[1] = angle_target 

        # quat1 = quaternion_from_euler(0,0,self.theta[0])
        # quat2 = quaternion_from_euler(0,0,self.theta[1])
        
        
        print("counter_step: {}".format(self.counter_step))
        print("counter_success: {}".format(self.counter_success))
        print("counter_collision: {}".format(self.counter_collision))
        print("counter_timeout: {}".format(self.counter_timeout))

        # Reset position assignment
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
        
        
        


        #print("id_bots = {}, tb3_0 = {}, tb3_1 = {}".format(id_bots, tb3_0, tb3_1))
        
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

        # Request service to reset the position of robots
        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            resp = set_state(state_msg2)
            resp = set_state(state_msg3)
            resp = set_state(state_msg4)
            #resp_targ = set_state(state_target_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        for i in range(self.num_robots):
            self.move_cmd[i].linear.x = 0.0
            self.move_cmd[i].angular.z = 0.0
            self.pub_tb3[i].publish(self.move_cmd[i])


        # Request service to reset the pheromone grid
        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)


        # ========================================================================= #
	    #                                  LOGGING                                  #
	    # ========================================================================= #
        if self.counter_step == 0:
            with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.file_name), mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Episode', 'Beta_const', 'Sensitivity', 'Success Rate', 'Average Arrival time', 'Standard Deviation'])

        if self.counter_step != 0:
            if (self.counter_collision != 0 or self.counter_success != 0):
                succ_percentage = 100*self.counter_success/(self.counter_collision+self.counter_success+self.counter_timeout)
            else:
                succ_percentage = 0
            print("Counter: {}".format(self.counter_step))

        if (self.counter_step % 10 == 0 and self.counter_step != 0):
            print("Beta_const: {}, Sensitivity: {}".format(self.beta_const, self.sensitivity))
            print("Success Rate: {}%".format(succ_percentage))

        if (self.counter_step % 20 == 0 and self.counter_step != 0):
            avg_comp = np.average(np.asarray(self.arrival_time))
            std_comp = np.std(np.asarray(self.arrival_time))
            print("{} trials ended. Success rate: {}, average completion time: {}, Standard deviation: {}".format(self.counter_step, succ_percentage, avg_comp, std_comp))
            
            with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.file_name), mode='a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['%i'%self.counter_step, '%0.2f'%self.beta_const, '%0.2f'%self.sensitivity, '%0.2f'%succ_percentage, '%0.2f'%avg_comp, '%0.2f'%std_comp])
            
            self.paramUpdate()
            self.arrival_time = []
            self.counter_collision = 0
            self.counter_timeout = 0
            self.counter_success = 0
            self.target_index = 0
            
        
        ''' How to exit the program? '''
            #print("why no exit")
            #sys.exit(1)
            #quit()
        #time.sleep(1)
        tmp_state = self.pose_ig.get_msg()
        #print("pose: {}".format(tmp_state))
        #self.rate.sleep()
        
        self.is_reset = True
        self.reset_timer = time.time()
        

    def paramUpdate(self):
        '''
        Parameter update after the number of experiments for a parameter set finished
        '''
        print("Parameters are updated!")
        if (self.s_counter < self.s_size-1):
            self.s_counter += 1
            self.sensitivity = self.s_range[self.s_counter]
        elif (self.b_counter < self.b_size-1):
            self.s_counter = 0
            self.b_counter += 1
            self.sensitivity = self.s_range[self.s_counter]
            self.beta_const = self.b_range[self.b_counter]
        else:
            print("Finish Iteration of parameters")
            sys.exit()
        


        

if __name__ == '__main__':
    wayN = WaypointNavigation()
    while True:
        try: 
            wayN.Computation()
        except KeyboardInterrupt:
            print("Ctrl+C pressed.")
            break


