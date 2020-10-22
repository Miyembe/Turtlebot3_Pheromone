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
from turtlebot3_waypoint_navigation.srv import PheroReset, PheroResetResponse
from turtlebot3_waypoint_navigation.msg import fma
from math import *
from time import sleep

from std_msgs.msg import Float32MultiArray

class WaypointNavigation:

    MAX_FORWARD_SPEED = 0.5
    MAX_ROTATION_SPEED = 0.5

    # Tunable parameters
    wGain = 10
    vConst = 0.2 #0.2
    distThr = 0.2
    pheroThr = 1

    
    def __init__(self):
      
        self.num_robots = 2
        # Initialise pheromone values
        self.phero = [[0.0] * 9] * self.num_robots
        self.phero_sum = [0.0] * self.num_robots
        
        # Initialise speed
        self.move_cmd = [Twist()]*self.num_robots

        # Initialise positions
        self.positions = [[0,0], [4,0]]
        self.goal = [[4,0], [0,0]]
        self.obstacle = [2,0]

        # Initialise ros related topics
        self.pub_tb3_0 = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=1)
        self.pub_tb3_1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.Callback, (self.pub_tb3_0, self.pub_tb3_1, self.move_cmd, self.goal))
        self.sub_phero = rospy.Subscriber('/phero_value', fma, self.ReadPhero)

        # Initialise simulation
        self.counter_step = 0
        self.counter_collision = 0
        self.counter_success = 0
        self.arrival_time = []
        
        self.is_reset = False
        self.is_collided = False
        self.is_goal = False

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

        
        pub1, pub2, msg, goal = cargs
        goal = self.goal

        for i in range(len(message.name)):
            if message.name[i] == 'tb3_0':
                tb3_0 = i
                #print("tb3: {}".format(tb3))
                #print("name: {}".format(message.name[tb3]))
            if message.name[i] == 'tb3_1':
                tb3_1 = i
        poses = [message.pose[tb3_0], message.pose[tb3_1]] 
        
        poss = [pose.position for pose in poses]
        oris = [pose.orientation for pose in poses]
        angles = [tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w)) for ori in oris]

        thetas = [angle[2] for angle in angles]
        #theta = pos.theta
        #how to make gazebo grid map
        # P controller
        v = [0, 0]
        w = [0, 0]
        
        distances = [None]*self.num_robots
        for i in range(self.num_robots):
            #print("poss {}, poss[i].x {}, goal[i] {}".format(poss, poss[0].x, goal[i]))
            distances[i] = sqrt((poss[i].x-goal[i][0])**2+(poss[i].y-goal[i][1])**2)

        distance_btw_robots = sqrt((poss[0].x-poss[1].x)**2+(poss[0].y-poss[1].y)**2)
        #print(distance_btw_robots)
        step_timer = time.time()
        reset_time = step_timer - self.reset_timer
        #print("reset_time: {}".format(reset_time))
        if (distance_btw_robots <= 0.3 and reset_time > 1):
            self.is_collided = True
            self.reset()
            

        for i in range(self.num_robots):            
            if (self.phero_sum[i] > self.pheroThr):
                msg[i] = self.PheroOA(self.phero[i])
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
                self.is_goal = True
                msg[i] = Twist()
                self.reset()
                
        # if (distance <= self.distThr and index < len(goal)-1):
        #     self.index += 1 
        #     print("Goal has been changed: ({}, {})".format(goal[self.index][0], goal[self.index][1]))

        # Publish velocity 
        pub1.publish(msg[0])
        pub2.publish(msg[1])

        self.is_reset = False

        # Reporting
        #print('Callback: x=%2.2f, y=%2.2f, dist=%4.2f, cmd.v=%2.2f, cmd.w=%2.2f' %(pos.x,pos.y,distance,v,w))
    
    # Angular velocity coefficient (When avg phero is high, it is more sensitive)
    def velCoef(self, value1, value2):
        val_avg = (value1 + value2)/2
        val_dif = value1 - value2
        dif_coef = exp(val_avg)
        
        return dif_coef*val_dif
        
    def PheroOA(self, phero):
        '''
        Pheromone-based obstacle avoidance algorithm
        - Input: 9 cells of pheromone
        - Output: Twist() to avoid obstacle
        '''
        # Constants:
        BIAS = self.BIAS = 0.1 #0.0 -successful set
        V_COEF = self.V_COEF = 0.35 #0.4
        W_COEF = self.W_COEF = 0.9 #0.8
        
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

        ang_vel = W_COEF*atan2(vel_vec[1], vel_vec[0])

        # Velocity assignment
        twist.linear.x = BIAS + V_COEF*avg_phero
        twist.angular.z = ang_vel

        return twist
    
    def reset(self):
        
        # Increment Collision Counter
        if self.is_collided == True:
            print("Collision!")
            self.counter_collision += 1

        # Increment Arrival Counter and store the arrival time
        if self.is_goal == True:
            print("Arrived goal!")
            self.counter_success += 1
            arrived_timer = time.time()
            art = arrived_timer-self.reset_timer
            self.arrival_time.append(art)
            print("Episode time: %0.2f"%art)

        # Reset the flags
        self.is_collided = False
        self.is_goal = False

        # Reset tb3_0 position
        state_msg = ModelState()
        state_msg.model_name = 'tb3_0'
        state_msg.pose.position.x = self.positions[0][0]
        state_msg.pose.position.y = self.positions[0][1] 
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        # Reset tb3_1 position
        state_target_msg = ModelState()    
        state_target_msg.model_name = 'tb3_1' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_target_msg.pose.position.x = self.positions[1][0]
        state_target_msg.pose.position.y = self.positions[1][1]
        state_target_msg.pose.position.z = 0.0
        state_target_msg.pose.orientation.x = 0
        state_target_msg.pose.orientation.y = 0
        state_target_msg.pose.orientation.z = -0.2
        state_target_msg.pose.orientation.w = 0

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
        # time.sleep(1)
        # self.pub.publish(self.move_cmd)

        # Request service to reset the pheromone grid
        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)

        # Calculate the step counter
        self.counter_step = self.counter_success + self.counter_collision
        if self.counter_step != 0:
            succ_percentage = 100*self.counter_success/self.counter_step
            print("Counter: {}".format(self.counter_step))
        
        if (self.counter_step % 10 == 0 and self.counter_step != 0):
            print("Success Rate: {}%".format(succ_percentage))
        if (self.counter_step == 1):
            avg_comp = np.average(np.asarray(self.arrival_time))
            print("100 trials ended. Success rate is: {} and average completion time: {}".format(succ_percentage, avg_comp))
            file_name = "manual_{}_{}_{}_{}".format(self.num_robots, self.BIAS, self.V_COEF, self.W_COEF)
            with open('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/src/log/csv/{}.csv'.format(file_name), mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Episode', 'Success Rate', 'Average Arrival time'])
                csv_writer.writerow(['%i'%self.counter_step, '%0.2f'%succ_percentage, '%0.2f'%avg_comp])
            #sys.exit(1)
            ''' How to exit the program? '''
            print("why no exit")
            
            #quit()

        self.is_reset = True
        self.reset_timer = time.time()


if __name__ == '__main__':
    rospy.init_node('pose_reading')
    wayN = WaypointNavigation()
    rospy.spin()


