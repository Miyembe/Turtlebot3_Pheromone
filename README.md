# Turtlebot3_Pheromone
This repository offers a package that offers basic pheromone features that can be used as an externalised &amp; spatialised communication mean for turtlebots.

# HOW TO USE

Step 0: Download this repositry in your catkin_ws/src


Step 1: Open a terminal and run turtlebot3 in Gazebo using the command below:
roslaunch turtlebot3_Pheromone turtlebot3_mini_arena.launch (for current PPO training, you have to modify the launch file in turtlebot3_gazebo package to load "mini_arena.world" in the world directory)
roslaunch Turtlebot3_Pheromone turtlebot3_collision_avoidance.launch (collision avoidance)

Step 2: Open a new terminal and run pheromone.py that generates pheromone grid & update pheromone:
rosrun Turtlebot3_Pheromone src/pheromone.py

Step 2-1: If you want to control the robot manually, open a new terminal and run this command below:
rosrun turtlebot3_teleop turtlebot3_teleop_key.launch

Step 2-2: If you want to reset the robot's position, open a new terminal and run this command below:
rosservice call /gazebo/reset_simulation "{}"

Step 3: Running Manual controller (If not needed, jump to Step 4)

Step 3-1: [Discrete-action Controller] Open a new terminal and run pheromone_controller.py to activate controller of turtlebot that responds to the given pheromone
rosrun Turtlebot3_Pheromone src/pheromone_controller.py

Step 3-2: [Continuous-action Controller] Open a new terminal an run continuous_controller.py to activate controller of turtlebot that responds to the given pheromone
rosrun Turtlebot3_Pheromone src/continuous_controller.py

Step 4: Training Turtlebot3 with PPO

Step 4.1: Open a new terminal and run rosrun Turtlebot3_pheromone src/phero_network_turtlebot3_ppo.py

 
# INFO

pheromone_controller.py can be flexibly modified so that you can design desired behaviour for the robots

Deep Reinforcement Learning will be added into the controller.

# Other usages

1. How to generate and use pheromone trail?

Using the teleop, generate pheromone trail manually. When it returns to home (currently 0,0), it saves the pheromone grid in tmp/NAME.npy (NAME is given in pheromone.py script).

When you run continuous_controller.py, it loads the saved pheromone trail (you need to specify the name of pheromone).

Currently there are sample pheromone data in tmp/Pheormone\samples.

2. How to plot the result from the training? 

During training, tensorflow summary is saved in the directory src/log. If you want to plot data using TensorBoard, install TensorBoard using (pip install tensorboard) command and run the command tensorboard --logdir src/log/tf_board/YOURDATA_dir (YOURDATA_dir is a directory in which your event data is saved.) and open an empty browser and go to localhost:6006. 
