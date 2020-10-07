# Turtlebot3_Pheromone
This repository offers a package that offers basic pheromone features that can be used as an externalised &amp; spatialised commnication mean for turtlebots.

# HOW TO USE

Step 0: Download this repositry in your catkin_ws/src


Step 1: Open a terminal and run turtlebot3 in Gazebo using the command below:
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

Step 2: Open a new terminal and run pheromone.py that generates pheromone grid & update pheromone:
rosrun Turtlebot3_Pheromone src/pheromone.py

Step 2-1: If you want to control the robot manually, open a new terminal and run this command below:
rosrun turtlebot3_teleop turtlebot3_teleop_key.launch

Step 2-2: If you want to reset the robot's position, open a new terminal and run this command below:
rosservice call /gazebo/reset_simulation "{}"

Step 3-1: [Discrete-action Controller] Open a new terminal and run pheromone_controller.py to activate controller of turtlebot that responds to the given pheromone
rosrun Turtlebot3_Pheromone src/pheromone_controller.py

Step 3-2: [Continuous-action Controller] Open a new terminal an run continuous_controller.py to activate controller of turtlebot that responds to the given pheromone
rosrun Turtlebot3_Pheromone src/continuous_controller.py
 
# INFO

pheromone_controller.py can be flexibly modified so that you can design desired behaviour for the robots

Deep Reinforcement Learning will be added into the controller.

