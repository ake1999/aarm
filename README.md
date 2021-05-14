# aarm
##### This is a hand made 3 DOF robot PRR

##### In this project, we aiming to use deep reinforcement learning techniques to push objects with a robotic manipulator

## There are 5 pakage in this project
## 1. aarm_description
* Design a urdf file for robot visualization and simulation
* Visualize robot in RVIZ
![picture alt](http://via.placeholder.com/200x150 "Title is optional")
## 2. aarm_gazebo
* physical simulation of robot and environment
* Using robot model in aarm_description pkg
* Design environment
* Running ROS controllers in aarm_control pkg
* Running rviz to visualize robot and camera output
* Running rqt-gui
![picture alt](http://via.placeholder.com/200x150 "Title is optional")
![picture alt](http://via.placeholder.com/200x150 "Title is optional")
## 3. aarm_control
* Setting pid parameters in a yaml ﬁle
* Loading ROS Controllers
* Run rqt-gui to send command to controllers, plot curves and optimize pid parameters
![picture alt](http://via.placeholder.com/200x150 "Title is optional")
## 4. aarm_moveit
* Solveing inverse kinematic
* Motion planning
![picture alt](http://via.placeholder.com/200x150 "Title is optional")
## 5. aarm_brain
* Train a RL model with TD3, DDPG and HER algorithms
![picture alt](http://via.placeholder.com/200x150 "Title is optional")
