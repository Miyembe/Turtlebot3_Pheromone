import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
import os
from IPython import display


path = "/home/sub/catkin_ws/src/Turtlebot3_Pheromone/tmp/20210406-161414/"

fig, ax = plt.subplots()




        
def pherogrid(n, time):
    # n - num of robots (int)
    # time - time_step (int)
    grids = [None]*n
    for i in range(n):
        with open(path + '{}_{}.npy'.format(i, time), 'rb') as f:
            grids[i] = np.load(f)
    sum_grids = np.maximum(grids[0], grids[1])
    return sum_grids

def robot_poses(n_frames):
    # for test, we use horizontal position change scheme
    x = []
    y = []
    yaw = []
    for i in range(n_frames):
        x.append(50.0+i*5)
        y.append(50.0)
        yaw.append(np.random.uniform(-np.pi, np.pi))
    return x, y, yaw

def plot_arrow(x, y, yaw, length=12.0, width=10, fc="y", ec="r"):
    """
    Plotting Arrow
    """
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x,y,yaw):
            ax.arrow(ix, iy, iyaw)
    else:
        ax.arrow(x, y, length * math.cos(yaw), length*math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        ax.plot(x, y)

def plot_triangle(x, y, yaw, length=12.0, width=10, fc="y", ec="r"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x,y,yaw):
            ax.arrow(ix, iy, iyaw)
    else:
        #ax.arrow(x, y, length * math.cos(yaw), length*math.sin(yaw),
        #          fc=fc, ec=ec, head_width=width, head_length=width)
        ax.plot(x, y, marker=(3, 0, 180*yaw/math.pi), markersize=20, linestyle='None', linewidth=10)
def plot_trajectory(x, y):
    ax.plot(x,y, "-r", label="trajectory")

def plot_anim(n_frames):
    # plot images continuously as background.
    time_str = time.strftime("%Y%m%d-%H%M%S")
    parent_dir = "/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/plot_script"
    path = os.path.join(parent_dir, time_str)
    os.mkdir(path)
    n = 2
    grids = []
    x, y, yaw = robot_poses(n_frames)
    for i in range(1, n_frames):
        grids.append(pherogrid(n, i/10))
    print(len(grids))

    x_traj, y_traj = [], []
    for i in range(n_frames-1):

        # Plotting Pheromone
        ax.imshow(grids[i])

        # Plotting Robot (pos, ori)
        plot_arrow(x[i], y[i], yaw[i])

        # Plotting Robot trajectory
        x_traj.append(x[i])
        y_traj.append(y[i])
        plot_trajectory(x_traj, y_traj)

        # Figure saving
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(path + '/{}_{}.png'.format(time_str, i), bbox_inches=extent)
        plt.pause(0.05)
        plt.cla()

def main():
    plot_anim(29)

if __name__=="__main__":
    main()