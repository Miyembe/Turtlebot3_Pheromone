#!/usr/bin/env python

import roslib; roslib.load_manifest('turtlebot3_waypoint_navigation')
import numpy as np
import rospy
import gazebo_msgs.msg
import tf
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool 
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from math import *
import time
import pheromone
from turtlebot3_waypoint_navigation.srv import PheroInj, PheroInjResponse
from turtlebot3_waypoint_navigation.srv import PheroReset, PheroResetResponse
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class ContinuousController:
    # This class offers that changes wheel velocity (vector) depending on pheromone value.
    # The controller is based on the behaviour rule based on COS-Phi

    def __init__(self, robot):
        rospy.init_node('continuous_controller', anonymous=False)
        self.velocity = Twist()
        self.pos = [0, 0]
        self.index = [0, 0]
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        #self.pub_inj = rospy.Publisehr('/phero_inj', Bool, queue_size=10)
        self.sub_pose = rospy.Subscriber('/phero_value', Float32MultiArray, self.pheroCallback, (self.pub_vel, self.velocity))
        self.bias = 0
        self.sensitivity = 0.2
        self.robot = robot
        self.inj_state = False

        rospy.wait_for_service('phero_inj')
        try:
            phero_inj = rospy.ServiceProxy('phero_inj', PheroInj)
            resp = phero_inj(self.inj_state)
            print("Service OK?: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed: %s"%e)

        rospy.wait_for_service('phero_reset')
        try:
            phero_reset = rospy.ServiceProxy('phero_reset', PheroReset)
            resp = phero_reset(True)
            print("Reset Pheromone grid successfully: {}".format(resp))
        except rospy.ServiceException as e:
            print("Service Failed %s"%e)
        self.resetPosition()



    def pheroCallback(self, phero, cargs):

        pub, vel = cargs

        phero_l = phero.data[0]
        phero_r = phero.data[1]
        print("Pheromone ({}, {})".format(phero_l, phero_r))

        self.bias = 2.5 - (phero_l + phero_r)
        wheel_r = (phero_r - phero_l)/self.sensitivity + self.bias
        wheel_l = (phero_l - phero_r)/self.sensitivity + self.bias

        forward_velocity = (wheel_l + wheel_r) * (self.robot.wheel_diameter/2)
        rotation_velocity = (wheel_l - wheel_r) * (self.robot.wheel_diameter/(2*self.robot.wheel_distance))

        vel.linear.x = forward_velocity
        vel.angular.z = rotation_velocity

        pub.publish(vel)
    
    def resetPosition(self):
        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_waffle_pi'
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0 
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)


class Turtlebot:
    
    def __init__(self):
        self.wheel_diameter = 0.05
        self.wheel_distance = 0.2 # Two pheromone cells

if __name__=="__main__":
    turtlebot = Turtlebot()
    controller = ContinuousController(turtlebot)
    rospy.spin()
