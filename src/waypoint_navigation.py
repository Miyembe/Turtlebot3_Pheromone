#!/usr/bin/env python
import rospy
import geometry_msgs.msg
import gazebo_msgs.msg
from gazebo_msgs.msg import ModelStates 
from math import *
from time import sleep

def Callback(message):

    pose = message.pose[1]
    twist = message.twist[1]

    pos = pose.position
    ori = pose.orientation

    print("Position")
    print(pos)
    print("Orientation")
    print(ori)
    print("-----------")

if __name__ == '__main__':
    rospy.init_node('pose_reading')
    rospy.Subscriber('/gazebo/model_states', ModelStates, Callback)
    rospy.spin()


