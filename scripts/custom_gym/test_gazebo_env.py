#!/usr/bin/env python

import gym
import envs
import numpy as np

env = gym.make('GazeboEnv-v0')

state = env.reset()

wow = (1,)
wow_shape = np.shape(wow)

print(state)
print(state.shape)

#print(state_shape)
#print(env.observation_space)
#print(wow_shape)
# for i in range(100):
#     action = env.action_space.sample()
#     next_state, reward, done = env.step(action)
#     print("-------------")
one = np.array(1)
print(one)
print(one.shape)
one = one.reshape(1,1)
print(one)
print(one.shape)
env.reset()
