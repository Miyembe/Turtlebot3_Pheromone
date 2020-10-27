#!/usr/bin/env python
import rospy
import gazebo_msgs.msg
import random
import tf
import numpy as np
import time
from gazebo_msgs.msg import ModelStates 
from gazebo_msgs.msg import ModelState 
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from turtlebot3_waypoint_navigation.srv import PheroReset, PheroResetResponse
from math import *
from time import sleep

from std_msgs.msg import Float32MultiArray

class WaypointNavigation:

    MAX_FORWARD_SPEED = 0.5
    MAX_ROTATION_SPEED = 0.5
    cmdmsg = Twist()
    index = 0

    # Tunable parameters
    wGain = 10
    vConst = 0.5
    distThr = 0.2
    pheroThr = 1

    
    def __init__(self):
      
        # Initialise pheromone values
        self.phero = [0.0] * 9
        self.phero_sum = 0.0
        
        # Initialise speed
        self.move_cmd = Twist()

        # Initialise positions
        self.goal = [4,0]
        self.obstacle = [2,0]

        # Initialise ros related topics
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.Callback, (self.pub, self.cmdmsg, self.goal))
        self.sub_phero = rospy.Subscriber('/phero_value', Float32MultiArray, self.ReadPhero)

        # Initialise simulation
        self.reset()


    # def RandomTwist(self):

    #     twist = Twist()
    #     twist.linear.x = self.MAX_FORWARD_SPEED * random.random()
    #     twist.angular.z = self.MAX_ROTATION_SPEED * (random.random() - 0.5)
    #     print(twist)
    #     return twist
    def ReadPhero(self, message):
        phero_data = message.data 
        self.phero = phero_data
        self.phero_sum = np.sum(np.asarray(phero_data))

    def Callback(self, message, cargs):

        pub, msg, goal = cargs
        goal = self.goal
        
        for i in range(len(message.name)):
            if message.name[i] == 'turtlebot3_waffle_pi':
                tb3 = i
                #print("tb3: {}".format(tb3))
                #print("name: {}".format(message.name[tb3]))
            if message.name[i] == 'unit_sphere_0_0':
                tg = i
        pose = message.pose[tb3]
        twist = message.twist[tb3]
        
        pos = pose.position
        ori = pose.orientation
        angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))

        theta = angles[2]
        #theta = pos.theta
        #how to make gazebo grid map
        # P controller
        v = 0
        w = 0
        
        # Index for # of goals
        index = self.index
        distance = sqrt((pos.x-goal[0])**2+(pos.y-goal[1])**2)

        if (self.phero_sum > self.pheroThr):
            msg = self.PheroOA(self.phero)
            v = msg.linear.x
            w = msg.angular.z

        # Adjust velocities
        elif (distance > self.distThr):
            v = self.vConst
            yaw = atan2(goal[1]-pos.y, goal[0]-pos.x)
            u = yaw - theta
            bound = atan2(sin(u), cos(u))
            w = min(1.0, max(-1.0, self.wGain*bound))
            msg.linear.x = v
            msg.angular.z = w
        elif (distance <= self.distThr):
            print("Arrived goal!")
            msg = Twist()
            self.reset()
        # if (distance <= self.distThr and index < len(goal)-1):
        #     self.index += 1 
        #     print("Goal has been changed: ({}, {})".format(goal[self.index][0], goal[self.index][1]))

        # Publish velocity 
        self.pub.publish(msg)

        # Reporting
        print('Callback: x=%2.2f, y=%2.2f, dist=%4.2f, cmd.v=%2.2f, cmd.w=%2.2f' %(pos.x,pos.y,distance,v,w))
    
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
        BIAS = 0.25
        V_COEF = 1.0
        W_COEF = 0.4
        
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

        # Reset Target Position
        state_target_msg = ModelState()    
        state_target_msg.model_name = 'unit_sphere_0_0' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_target_msg.pose.position.x = self.goal[0]
        state_target_msg.pose.position.y = self.goal[1]
        state_target_msg.pose.position.z = 0.0
        state_target_msg.pose.orientation.x = 0
        state_target_msg.pose.orientation.y = 0
        state_target_msg.pose.orientation.z = 0
        state_target_msg.pose.orientation.w = 0

        rospy.wait_for_service('gazebo/reset_simulation')

        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            resp_targ = set_state(state_target_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0
        self.pub.publish(self.move_cmd)
        # time.sleep(1)
        # self.pub.publish(self.move_cmd)

        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)


if __name__ == '__main__':
    rospy.init_node('pose_reading')
    wayN = WaypointNavigation()
    rospy.spin()


