#!/usr/bin/env python
import phero_turtlebot_alife_exp1
import numpy as np

def main():
    env = phero_turtlebot_alife_exp1.Env()
    states = []
    rewards = []
    dones = []
    env.reset()
    reward = 0
    done = False
    while done is False:
        _, state, reward, done, _ = env.step()
        states.append(state)
        rewards.append(reward)
        dones.append(done)
    print("Finished")
    print("-------------")
    #print("states: {}".format(states))
    #print("rewards: {}".format(rewards))
    #print("dones: {}".format(dones))
    trajectory = np.asarray(states)
    np.append(trajectory, rewards)
    np.append(trajectory, dones)


    file_name = "trajectory"
    with open('{}.npy'.format(file_name), 'wb') as f: # Please change the path to your local machine's path to the catkin workspace
        np.save(f, trajectory)
        print("The trajectory of the robot is saved in file : {}.npy".format(file_name))

if __name__ == "__main__":
    main()
