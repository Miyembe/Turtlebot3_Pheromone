#!/usr/bin/env python
import rospy
import gazebo_msgs.msg
import random
import tf
from gazebo_msgs.msg import ModelStates 
from geometry_msgs.msg import Twist
from math import *
from time import sleep

class WaypointNavigation:

    MAX_FORWARD_SPEED = 0.5
    MAX_ROTATION_SPEED = 0.5
    goal = [[0,0],[1,3],[-4,2],[0,0]]
    cmdmsg = Twist()
    index = 0

    # Tunable parameters
    wGain = 10
    vConst = 0.5
    distThr = 0.1

    

    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.Callback, (self.pub, self.cmdmsg, self.goal))
        

    # def RandomTwist(self):

    #     twist = Twist()
    #     twist.linear.x = self.MAX_FORWARD_SPEED * random.random()
    #     twist.angular.z = self.MAX_ROTATION_SPEED * (random.random() - 0.5)
    #     print(twist)
    #     return twist

    def Callback(self, message, cargs):

        pub, msg, goal = cargs
        
        pose = message.pose[1]
        twist = message.twist[1]

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
        distance = sqrt((pos.x-goal[index][0])**2+(pos.y-goal[index][1])**2)

        # Adjust velocities
        if (distance > self.distThr):
            v = self.vConst
            yaw = atan2(goal[index][1]-pos.y, goal[index][0]-pos.x)
            u = yaw - theta
            bound = atan2(sin(u), cos(u))
            w = min(0.5, max(-0.5, self.wGain*bound))

        if (distance <= self.distThr and index < len(goal)-1):
            self.index += 1 
            print("Goal has been changed: ({}, {})".format(goal[self.index][0], goal[self.index][1]))

        # Publish velocity
        msg.linear.x = v
        msg.angular.z = w
        self.pub.publish(msg)

        # Reporting
        print('Callback: x=%2.2f, y=%2.2f, dist=%4.2f, cmd.v=%2.2f, cmd.w=%2.2f' %(pos.x,pos.y,distance,v,w))

        # print("Position")
        # print(pos)
        # # print("Velocity")
        # # print(twist)
        # print("-----------")

        # rantwist = self.RandomTwist()
        # self.pub.publish(rantwist)


if __name__ == '__main__':
    rospy.init_node('pose_reading')
    wayN = WaypointNavigation()
    rospy.spin()


