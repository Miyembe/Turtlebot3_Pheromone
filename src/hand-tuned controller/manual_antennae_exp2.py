#!/usr/bin/env python
import rospy
import gazebo_msgs.msg
import random
import tf
import numpy as np
import time
import sys
import csv
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

class WaypointNavigation:

    MAX_FORWARD_SPEED = 0.5
    MAX_ROTATION_SPEED = 0.5

    # Tunable parameters
    wGain = 10
    vConst = 0.5 #0.2
    distThr = 0.2
    pheroThr = 0.2

    
    def __init__(self):
      
        self.num_robots = 2
        # Initialise pheromone values
        self.phero = [[0.0] * 9] * self.num_robots
        self.phero_sum = [0.0] * self.num_robots
        
        # Initialise speed
        self.move_cmd = [Twist()]*self.num_robots

        # Initialise positions
        self.positions = [[-2.5,0], [2.5,0]]
        #self.goal = [[2.5,0.0], [0,0]]
        self.obstacle = [2,0]
        self.dones = [False]*2

        # Set initial positions
        self.target = [[2.5, 0.0], [-2.5, 0.0]] # Two goal
        self.d_robots = 5.0
        self.target_index = 0
        self.num_experiments = 20
        self.x = [0.0]*self.num_robots
        self.y = [0.0]*self.num_robots
        self.theta = [0.0]*self.num_robots

        # File name
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.file_name = "manual_{}_{}".format(self.num_robots, self.time_str)
        self.traj_name = "{}_traj".format(self.file_name)
        print(self.file_name)

        # Initialise parameters
        
        self.step_size = 0.1
        #self.b_range = np.arange(0, 1+self.step_size, self.step_size)
        self.v_range = np.arange(0.2, 0.6, self.step_size)
        self.w_range = np.arange(0.2, 1+self.step_size, self.step_size)

        self.BIAS = 0.3
        self.V_COEF = 1.0#self.v_range[0]
        self.W_COEF = 1.0#self.w_range[0]

        #self.b_size = self.b_range.size        
        self.v_size = self.v_range.size
        self.w_size = self.w_range.size
        
        self.b_counter = 0
        self.v_counter = 0
        self.w_counter = 0

        # Initialise ros related topics
        self.pub_tb3_0 = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=1)
        self.pub_tb3_1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.Callback, (self.pub_tb3_0, self.pub_tb3_1, self.move_cmd, self.target))
        self.sub_phero = rospy.Subscriber('/phero_value', fma, self.ReadPhero)

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

        # antenna movement related parameters

        self.b_range = np.arange(0.7, 0.7+self.step_size, self.step_size)
        self.s_range = np.arange(0.8, 0.8+self.step_size, self.step_size)

        self.b_size = self.b_range.size
        self.s_size = self.s_range.size

        self.b_counter = 0
        self.s_counter = 0

        self.beta_const = self.b_range[0]
        self.sensitivity = self.s_range[0]

        
        # Log related
        self.log_timer = time.time()
        self.positions = []
        for i in range(self.num_robots):
            self.positions.append([])
        self.traj_eff = list()

        # Reset related

        self.reset_timer = time.time()
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

    def Callback(self, message, cargs):
        '''
        Main Function
        - Receive position data
        - Generate action
        '''
        # ========================================================================= #
	    #                           Initialisation                                  #
	    # ========================================================================= #
        pub1, pub2, msg, goal = cargs
        goal = self.target

        for i in range(len(message.name)):
            if message.name[i] == 'tb3_0':
                tb3_0 = i
                #print("tb3: {}".format(tb3))
                #print("name: {}".format(message.name[tb3]))
            if message.name[i] == 'tb3_1':
                tb3_1 = i
        poses = [message.pose[tb3_0], message.pose[tb3_1]] 
        poss = [pose.position for pose in poses]
        x = [p.x for p in poss]
        y = [p.y for p in poss]
        oris = [pose.orientation for pose in poses]
        angles = [tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w)) for ori in oris]

        thetas = [angle[2] for angle in angles]

        # P controller
        v = [0, 0]
        w = [0, 0]
        
        distances = [None]*self.num_robots
        distance_btw_robots = [0.0]
        for i in range(self.num_robots):
            distances[i] = sqrt((poss[i].x-goal[i][0])**2+(poss[i].y-goal[i][1])**2)

        distance_btw_robots = sqrt((x[0]-x[1])**2+(y[0]-y[1])**2) # Python 
        step_timer = time.time()
        reset_time = step_timer - self.reset_timer

        # Log Positions
        if time.time() - self.log_timer > 0.5:
            for i in range(self.num_robots):
                with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.traj_name), mode='a') as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(['%0.1f'%reset_time, '%i'%i, '%0.2f'%poss[i].x, '%0.2f'%poss[i].y])
                self.positions[i].append([x[i], y[i]])
            self.log_timer = time.time()


        # ========================================================================= #
	    #                          Action & State assignment                        #
	    # ========================================================================= #

        if (distance_btw_robots <= 0.3 and reset_time > 1):
            self.is_collided = True
            self.reset()

        for i in range(self.num_robots):            
            if (self.phero_sum[i] > self.pheroThr):
                msg[i] = self.PheroResponse(self.phero[i])
                v[i] = msg[i].linear.x
                w[i] = msg[i].angular.z

            # Adjust velocities
            elif (distances[i] > self.distThr):
                v[i] = self.vConst
                yaw = atan2(goal[i][1]-poss[i].y, goal[i][0]-poss[i].x)
                u = yaw - thetas[i]
                bound = atan2(sin(u), cos(u))
                w[i] = min(1.0, max(-1.0, self.wGain*bound))
                msg[i].linear.x = v[i]
                msg[i].angular.z = w[i]

                if self.is_reset == True:
                    self.is_reset = False
            elif (distances[i] <= self.distThr and reset_time > 1):
                self.dones[i] = True
                msg[i] = Twist()
        
        if reset_time > 60.0:
            print("Times up!")
            self.is_timeout = True
            self.reset()
        
        if all(done == True for done in self.dones) == True:
            self.is_goal = True
            self.dones = [False]*2
            self.reset()

        # Publish velocity 
        pub1.publish(msg[0])
        pub2.publish(msg[1])

        self.is_reset = False
    
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
        vec_coefs = [0.0] * 4
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

        print("counter_step: {}".format(self.counter_step))
        print("counter_success: {}".format(self.counter_success))
        print("counter_collision: {}".format(self.counter_collision))
        print("counter_timeout: {}".format(self.counter_timeout))

        # ========================================================================= #
	    #                                  RESET                                    #
	    # ========================================================================= #

        if self.target_index < self.num_experiments-1:
            self.target_index += 1
        else:
            self.target_index = 0
            
        angle_target = self.target_index*2*pi/self.num_experiments        

        self.x[0] = (self.d_robots/2)*cos(angle_target)
        self.y[0] = (self.d_robots/2)*sin(angle_target)

        self.x[1] = (self.d_robots/2)*cos(angle_target+pi)
        self.y[1] = (self.d_robots/2)*sin(angle_target+pi)

        self.target = [[self.x[1], self.y[1]], [self.x[0], self.y[0]]]

        self.theta[0] = angle_target + pi
        self.theta[1] = angle_target 

        quat1 = quaternion_from_euler(0,0,self.theta[0])
        quat2 = quaternion_from_euler(0,0,self.theta[1])
        
        

        # Reset tb3_0 position
        state_msg = ModelState()
        state_msg.model_name = 'tb3_0'
        state_msg.pose.position.x = self.x[0]
        state_msg.pose.position.y = self.y[0] 
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = quat1[0]
        state_msg.pose.orientation.y = quat1[1]
        state_msg.pose.orientation.z = quat1[2]
        state_msg.pose.orientation.w = quat1[3]

        # Reset tb3_1 position
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

        # Request service to reset the position of robots
        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            resp_targ = set_state(state_target_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        for i in range(self.num_robots):
            self.move_cmd[i].linear.x = 0.0
            self.move_cmd[i].angular.z = 0.0
        self.pub_tb3_0.publish(self.move_cmd[0])
        self.pub_tb3_1.publish(self.move_cmd[1])

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
                csv_writer.writerow(['Episode', 'Success Rate', 'Average Arrival time', 'std_at', 'Collision Rate', 'Timeout Rate', 'Trajectory Efficiency', 'std_te'])
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

        if (self.counter_step % 1 == 0 and self.counter_step != 0):
            print("Beta_const: {}, Sensitivity: {}".format(self.beta_const, self.sensitivity))
            print("Success Rate: {}%".format(succ_percentage))

        if (self.counter_step % 100 == 0 and self.counter_step != 0):
            avg_comp = np.average(np.asarray(self.arrival_time))
            std_comp = np.std(np.asarray(self.arrival_time))
            avg_traj = np.average(np.asarray(self.traj_eff))
            std_traj = np.std(np.asarray(self.traj_eff))
            print("{} trials ended. Success rate: {}, average completion time: {}, Standard deviation: {}".format(self.counter_step, succ_percentage, avg_comp, std_comp))
            
            with open('/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(self.file_name), mode='a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['%i'%self.counter_step, '%0.2f'%succ_percentage, '%0.2f'%avg_comp, '%0.2f'%std_comp, '%0.2f'%col_percentage, '%0.2f'%tout_percentage, '%0.4f'%avg_traj, '%0.4f'%std_traj])
            
            #self.paramUpdate()
            self.arrival_time = list()
            self.traj_eff = list()
            self.counter_collision = 0
            self.counter_success = 0
            self.counter_timeout = 0
            self.target_index = 0
            
        
        ''' How to exit the program? '''
            #print("why no exit")
            #sys.exit(1)
            #quit()
        self.positions = []
        for i in range(self.num_robots):
            self.positions.append([])
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
    rospy.init_node('pose_reading')
    wayN = WaypointNavigation()
    rospy.spin()


