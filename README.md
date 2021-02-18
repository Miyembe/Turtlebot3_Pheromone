# Turtlebot3_Pheromone
This repository offers a package that offers basic pheromone features that can be used as an externalised &amp; spatialised communication mean for turtlebots.

# HOW TO USE

Step 0: Download this repositry in your catkin_ws/src


Step 1: Open a terminal and run turtlebot3 in Gazebo using the command below:
roslaunch turtlebot3_Pheromone "your.launch" (select launch file you would like to launch in Gazebo)
roslaunch turtlebot3_Pheromone tcds_exp1.launch (open experiment 1)

Step 2: Open a new terminal and run pheromone script for corresponding experiment in "src/Pheromone" directory. Note that you don't have to specify the directory path to run in rosrun. 
rosrun turtlebot3_pheromone "your_pheromone.py"
rosrun turtlebot3_pheromone pheromone_exp1.py (pheromone for experiment 1)


Step 2-1: If you want to run [hand-tuned controller], run the script in the "hand-tuned controller" directory. 
rosrun turtlebot3_pheromone "your_hand-tuned-controller.py"
rosrun turtlebot3_pheromone manual_antennae_exp1.py (manual controller for exp1)


Step 2-2: If you want to train [DRL-based controller], run the script in the "DRL-based controller" directory.
rosrun turtlebot3_pheromone "your_DRL-based-controller.py"
rosrun turtlebot3_pheromone phero_network_exp1.py (Training DRL-based controller for exp1)

Step 2-3: If you want to run trained [DRL-based controller], run the script ending with "eval" in the "DRL-based controller" directory. You need to select the neural network parameter in "src/log/checkpoints/YYYYMMDD-HHMMSS/"parameter (note that you dont put the extension such as .index)"
rosrun turtlebot3_pheromone "your_DRL-based-controller_eval.py"
rosrun turtlebot3_pheromone phero_network_exp1_eval.py (Running trained DRL-based controller for exp1)



# Other usages

1. How to generate and use pheromone trail?

Using the teleop, generate pheromone trail manually. When it returns to home (currently 0,0), it saves the pheromone grid in tmp/NAME.npy (NAME is given in pheromone.py script).

When you run continuous_controller.py, it loads the saved pheromone trail (you need to specify the name of pheromone).

Currently there are sample pheromone data in tmp/Pheormone\samples.

2. How to plot the result from the training? 

During training, tensorflow summary is saved in the directory src/log. If you want to plot data using TensorBoard, install TensorBoard using (pip install tensorboard) command and run the command tensorboard --logdir src/log/tf_board/YOURDATA_dir (YOURDATA_dir is a directory in which your event data is saved.) and open an empty browser and go to localhost:6006. 

3. How to visualise the pregenerated pheromone map?

In "src/plot_script", there is a script called pheromone_plot.py and pheromone_plot_generation.ipynb. Open the script and choose the pheromone map file in "tmp/***.npy"
