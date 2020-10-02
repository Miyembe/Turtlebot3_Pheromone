#!/usr/bin/env python
import rospy
import gazebo_msgs.msg
import random
import tf
from gazebo_msgs.msg import ModelStates 
from geometry_msgs.msg import Twist
from math import *
from time import sleep

import gym 
import envs
import numpy as np
import random
import tensorflow as tfl
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.backend import set_session

class DQN:
    def __init__(self, env):
        self.env = env
        self.epsilon = 0.05
        self.learning_rate = 0.01
        # sess = tfl.Session(graph=tfl.Graph())
        # with sess.graph.as_default():
        # set_session(sess)
        self.model = keras.models.load_model('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/scripts/custom_gym/success.model')
        

    # def create_model(self):
    #     keras.backend.clear_session()
    #     model = Sequential()
    #     state_shape = self.env.observation_space.shape
    #     model.add(Dense(12, input_dim=1, activation="relu"))
    #     model.add(Dense(12))
    #     model.add(Dense(self.env.action_space.n))
    #     # model.compile(loss="mean_squared_error",
    #     #               optimizer = Adam(lr=self.learning_rate))
    #     model.load_weights('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/scripts/custom_gym/success.model')
    #     global graph
    #     graph = tfl.get_default_graph() 
    #     return model

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with self.sess.graph.as_default(): 
        #     set_session(session)
        greedyAction = np.argmax(self.model.predict(state))
        return greedyAction

class WaypointNavigation:

    MAX_FORWARD_SPEED = 0.5
    MAX_ROTATION_SPEED = 0.5
    cmdmsg = Twist()
    index = 0
    goal = [0, 0]
    state = np.array(1).reshape(1,)
    # Tunable parameters
    wGain = 10
    vConst = 0.5
    distThr = 0.1

    

    def __init__(self, dqn1):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.Callback, (self.pub, self.cmdmsg, self.goal))
        self.dqn_agent = dqn1
        self.state = env.reset()
        self.ncol = 10 
        self.nrow = 10
        self.done = False

    # def RandomTwist(self):

    #     twist = Twist()
    #     twist.linear.x = self.MAX_FORWARD_SPEED * random.random()
    #     twist.angular.z = self.MAX_ROTATION_SPEED * (random.random() - 0.5)
    #     print(twist)
    #     return twist
    def StateToGoal(self, state):
        goalX = state[0] // self.ncol
        goalY = state[0] %  self.ncol
        goal = [goalX, goalY]
        return goal

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
        

        distance = sqrt((pos.x-goal[0])**2+(pos.y-goal[1])**2)

        # Adjust velocities
        if (distance > self.distThr):
            v = self.vConst
            yaw = atan2(goal[1]-pos.y, goal[0]-pos.x)
            u = yaw - theta
            bound = atan2(sin(u), cos(u))
            w = min(0.5, max(-0.5, self.wGain*bound))

        if (distance <= self.distThr):
            action = self.dqn_agent.act(self.state)
            new_state, reward, done = self.dqn_agent.env.step(action)
            self.goal = self.StateToGoal(new_state)
            self.state = new_state
            self.done = done
            print("Goal has been changed: ({}, {})".format(self.goal[0], self.goal[1]))

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
    rospy.init_node('waypoint_nav_dqn')
    env = gym.make('GazeboEnv-v0')
    dqn_agent = DQN(env=env)
    wn1 = WaypointNavigation(dqn1 = dqn_agent)
    if wn1.done is True:
        print("Reached Goal, End mission.")
    rospy.spin()