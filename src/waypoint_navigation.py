#!/usr/bin/env python
import rospy
import gazebo_msgs.msg
import random
from gazebo_msgs.msg import ModelStates 
from geometry_msgs.msg import Twist
from math import *
from time import sleep

class WaypointNavigation:

    MAX_FORWARD_SPEED = 0.5
    MAX_ROTATION_SPEED = 0.5

    def __init__(self):
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.Callback)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

    def RandomTwist(self):

        twist = Twist()
        twist.linear.x = self.MAX_FORWARD_SPEED * random.random()
        twist.angular.z = self.MAX_ROTATION_SPEED * (random.random() - 0.5)
        print(twist)
        return twist

    def Callback(self, message):

        pose = message.pose[1]
        twist = message.twist[1]

        pos = pose.position
        ori = pose.orientation

        print("Position")
        print(pos)
        # print("Velocity")
        # print(twist)
        print("-----------")

        rantwist = self.RandomTwist()
        self.pub.publish(rantwist)


if __name__ == '__main__':
    rospy.init_node('pose_reading')
    wayN = WaypointNavigation()
    rospy.spin()


