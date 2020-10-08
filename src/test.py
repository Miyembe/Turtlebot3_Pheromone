#! /usr/bin/env python

from phero_turtlebot_turtlebot3_ppo import Env

env = Env()

for i in range(100):
    state, reward, done = env.step()
    print("next state: {}, reward: {}, done: {}".format(state,reward,done))
#reset_state = env.reset()
#print("Reset_state: {}".format(reset_state))
print("Finished")