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
    time_step = 128
    num_traj = 10
    trajectory = None
    traj_buffer = None
    for i in range(num_traj):
        for j in range(time_step):
            _, state, reward, done, _ = env.step(linear_x=0.8)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            print("done: {}".format(done))
            if done == True:
                env.reset()
        trajectory = np.asarray(states)
        np.append(trajectory, rewards)
        np.append(trajectory, dones)
        if i == 0:
            traj_buffer = trajectory
            print("Trajectory buffer is initialised.")
            print("Trajectory size : {}".format(np.shape(trajectory)))
        else:
            np.append(traj_buffer, trajectory)
            print("shape of the traj_buffer: {}".format(np.shape(traj_buffer)))

    print("--------------------\\\\-----------------")
    print("shape of the traj_buffer: {}".format(np.shape(traj_buffer)))

    
        
    #print("states: {}".format(states))
    #print("rewards: {}".format(rewards))
    #print("dones: {}".format(dones))
    


    file_name = "trajectory"
    with open('{}.npy'.format(file_name), 'wb') as f: # Please change the path to your local machine's path to the catkin workspace
        np.save(f, traj_buffer)
        print("The trajectory buffer of the robot is saved in file : {}.npy".format(file_name))

if __name__ == "__main__":
    main()
