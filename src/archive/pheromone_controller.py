#!/usr/bin/env python

import roslib; roslib.load_manifest('turtlebot3_waypoint_navigation')
import numpy as np
import rospy
import gazebo_msgs.msg
import tf
from gazebo_msgs.msg import ModelStates 
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from math import *
import time
import pheromone
from turtlebot3_waypoint_navigation.srv import PheroGoal, PheroGoalResponse

class PheromoneController:

    res = 10
    num_cell = 101

    wGain = 10
    vConst = 0.05
    distThr = 0.01

    def __init__(self):
        rospy.init_node('pheromone_controller', anonymous=False)
        self.goal = []
        self.velocity = Twist()
        self.pos = [0, 0]
        self.index = [0, 0]
        self.is_arrived = True

        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.poseCallback, (self.pub_vel, self.velocity, self.goal))
        self.service_time = time.clock()

    def poseCallback(self, message, cargs):

        pub, msg, goal = cargs
        
        pose = message.pose[1]
        twist = message.twist[1]

        pos = pose.position
        self.pos[0] = pos.x
        self.pos[1] = pos.y

        ori = pose.orientation
        angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        theta = angles[2]

        # P controller
        v = 0
        w = 0

        # Goal Assignment / when it has arrived the previous target
        if self.is_arrived is True:
            self.pheroGoalClient()
        yaw = atan2(self.goal[1]-self.pos[1], self.goal[0]-self.pos[0])

        if theta < 0:
            theta = theta + 2*pi
        if yaw < 0:
            yaw = yaw + 2*pi

        angle_diff = yaw - theta
        if angle_diff < -pi:
            angle_diff = angle_diff + 2*pi
        if angle_diff > pi:
            angle_diff = angle_diff - 2*pi
        
        distance = sqrt((self.pos[0]-self.goal[0])**2+(self.pos[1]-self.goal[1])**2)


        # Update goal in a fixed period
        # cur_time = time.clock()
        # if cur_time - self.service_time > 1:
        #     self.pheroGoalClient()
        #     self.service_time = cur_time

        # Adjust Linear & Angular velocity
        #print("Angle diff: {}".format(angle_diff))
        if abs(angle_diff) > 5.0*(2*pi/360):
            w = min(1.0, max(-1.0, angle_diff))
        else:
            if (distance > self.distThr):
                v = min(distance, 0.1) # self.vConst
                #yaw = atan2(self.goal[1]-self.pos[1], self.goal[0]-self.pos[0])
                #u = yaw - theta
                #bound = atan2(sin(u), cos(u))
                #w = min(1.0, max(-1.0, angle_diff))
                self.is_arrived = False
        
            elif (distance <= self.distThr):
                self.is_arrived = True
                print("It has arrived the goal!")


        # Publish Velocity
        msg.linear.x = v
        msg.angular.z = w
        pub.publish(msg)

        # Reporting
        print('Callback: x=%2.2f, y=%2.2f, dist=%4.2f, cmd.v=%2.2f, cmd.w=%2.2f, goal=(%2.2f,%2.2f)' %(pos.x,pos.y,distance,v,w,self.goal[0],self.goal[1]))

    def pheroGoalClient(self):
        
        rospy.wait_for_service('phero_goal')
        try:
            phero_goal = rospy.ServiceProxy('phero_goal', PheroGoal)
            resp = phero_goal(self.pos[0], self.pos[1])
            self.goal = [resp.next_x, resp.next_y]
            print("Pose: {}, Type: {}".format(self.pos, type(self.pos)))
            print("Got new goal!: ({}, {})".format(resp.next_x, resp.next_y))
        except rospy.ServiceException as e:
            print("Service Failed: %s"%e)

    # change goal using 9 pheromone values. 
    # def Callback(self, message):

    #     res = self.res
    #     num_cell = self.num_cell

    #     # Read Pheromone values from /phero_value node and convert it to 2D array 
    #     phero_cells = np.array([message[0], message[1], message[2]], 
    #                            [message[3], message[4], message[5]],
    #                            [message[6], message[7], message[8]])

    #     # pose to index
    #     round_dp = int(math.log10(res))
    #     x = round(self.pos[0], round_dp) # round the position value so that they fit into the centre of the cell.
    #     y = round(self.pos[1], round_dp) # e.g. 0.13 -> 0.1
    #     x = int(x*res)
    #     y = int(y*res)
    #     x_index = x + (num_cell-1)/2
    #     y_index = y + (num_cell-1)/2

    #     # Take argmax and get the index.
    #     index = np.argmax(phero_cells)
    #     index_goal= np.array([index[0]-1, index[1]-1])
    #     target_index = np.array([x_index+index_goal[0], y_index+index_goal[1]])
    #     x_target = (target_index[0] - (num_cell-1)/2)/res
    #     y_target = (target_index[1] - (num_cell-1)/2)/res
        
        #####   Part 2: Waypoint Navigation #####
        #####                               #####

        

    #def waypointNavigation(self):


        # 
if __name__ == "__main__":
    rospy.init_node('pheromone_controller')
    phero_con = PheromoneController()
    
    rospy.spin()