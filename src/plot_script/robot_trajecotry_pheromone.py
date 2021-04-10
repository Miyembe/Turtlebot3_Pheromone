import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
import os
import pandas as pd
from IPython import display

fig, ax = plt.subplots()
# Basic Plotting functions

def plot_arrow(x, y, yaw, length=12.0, width=10, fc="y", ec="r"):
    """
    Plotting Arrow
    """
    if not isinstance(x, float) and not isinstance(x, int):
        print("x: {}".format(x))
        print("type x: {}".format(type(x)))
        ax.arrow(x, y, length * math.cos(yaw), length*math.sin(yaw),
                fc=fc, ec=ec, head_width=width, head_length=width)
        #for (ix, iy, iyaw) in zip(x,y,yaw):
        #    ax.arrow(ix, iy, length * math.cos(iyaw), length*math.sin(iyaw), fc=fc, ec=ec, head_width=width, head_length=width)
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




class PheroPlot():
    '''
    Class 
    1. Importing pheromone images
    2. Importing Trajectory images
    3. Process pheromone images
    4. Merge separate pheromone images
    5. Merge Pheromone images and trajectory plot
    6. Save it as animation
    '''
    def __init__(self):
        
        self.num_robots = 2

        # Reading
        self.path = "/home/sub/catkin_ws/src/Turtlebot3_Pheromone/tmp/20210409-145642"
        self.num_files = len([name for name in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, name))])
        self.num_phero = (self.num_files-1)/2
        self.skip_count = 0
        self.time_step = 0.1
        

        # Saving
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.parent_dir = "/home/sub/catkin_ws/src/Turtlebot3_Pheromone/src/plot_script"
        self.save_path = os.path.join(self.parent_dir, self.time_str)
        os.mkdir(self.save_path)

    def pherogrid(self, t):
        # n - num of robots (int)
        # time - time_step (int)
        grids = [None]*self.num_robots
        for i in range(self.num_robots):
            while not os.path.isfile(os.path.join(self.path, '{}_{}.npy'.format(i, (t+self.skip_count)*self.time_step))):
                self.skip_count += 1
                print("Counter++")
                time.sleep(0.5)
                
            with open(self.path + '/{}_{}.npy'.format(i, (t+self.skip_count)*self.time_step), 'rb') as f:
                grids[i] = np.load(f)
        sum_grids = np.maximum(grids[0], grids[1])
        phero_time = (t + self.skip_count)*self.time_step 
        return sum_grids, round(phero_time, 1)
    
    def robot_poses(self, times):
        df = pd.read_csv(os.path.join(self.path, 'pose_2.csv'))
        data_poses = df.set_index(['time', 'ID'])
        print("data_poses: {}".format(data_poses))
        x = [[] for i in range(self.num_robots)]
        y = [[] for i in range(self.num_robots)]
        yaw = [[] for i in range(self.num_robots)]
        for j in range(self.num_robots):
            for i in times:           
                print("i: {}".format(i))
                x[j].append(data_poses["x"][i][j])
                y[j].append(data_poses["y"][i][j])
                yaw[j].append(data_poses["yaw"][i][j])
        return x, y, yaw

    
    def plot_anim(self):
        # plot images continuously as background.
        grids = []
        phero_times = []
        print("wow")
        for i in range(1, self.num_phero+1):
            grid, phero_time = self.pherogrid(i)
            grids.append(grid)
            phero_times.append(phero_time)
            print("phero_time: {}".format(phero_time))

        x, y, yaw = self.robot_poses(phero_times)
        print("0x0: {}".format(x))
        x_traj, y_traj = [], []
        for i in range(self.num_phero):

            # Plotting Pheromone
            ax.imshow(grids[i])

            # Plotting Robot (pos, ori)
            for j in range(self.num_robots):
                plot_arrow(x[j][i], y[j][i], yaw[j][i])
                print("x: {}".format(x[j][i]))
            # plot_arrow([x[j][i] for j in range(self.num_robots)],
            #            [y[j][i] for j in range(self.num_robots)],
            #            [yaw[j][i] for j in range(self.num_robots)])

            # Plotting Robot trajectory
            #x_traj.append(x[i])
            #y_traj.append(y[i])
            #plot_trajectory(x_traj, y_traj)

            # Figure saving
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(self.save_path + '/{}_{}.png'.format(self.time_str, i), bbox_inches=extent)
            plt.pause(0.1)
            plt.cla()



def main():
    
    phero_plot_obj = PheroPlot()
    phero_plot_obj.plot_anim()

if __name__=="__main__":
    main()