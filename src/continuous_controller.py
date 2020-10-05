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