# **aarm**

*The purpose of this project is to train a 3 DOF handmade robot with PRR joints and with planer movement to move objects on a horizontal plane to a randomly chosen desired location on the table, which is done with the help of reinforcement learning algorithms and in the simulated environment. this simulation is done in the Gazebo that has been implemented under the ROS software framework.*

*To achieve this goal, the robot drivers were installed in the Ubuntu 16.04 LTS operating system, and programs were written in C++ to enable the ROS environment to communicate with the robot. Then the robot is simulated in the Gazebo environment by a designed robot model in a urdf file. Furthermore, The ROS Controllers are applied to the simulated robot, and the PID parameters are also tuned.*

*There are various algorithms to implement artificial intelligence for this project. But since our environment in this project is a continuous environment that is unlikely to achieve the goal by random actions, just some state-of-the-art algorithms remain that can handle learning in this complex environment. one of them is designed to think like humans that uses its mistakes to do better actions in the future which is done by the HER algorithm. About the Reinforcement Learning algorithm, we implement the TD3 and DDPG off-policy algorithms by using TensorFlow and PyTorch libraries.*

## There are 5 pakage in this project
1. <a href="#1-aarm_description">aarm_description</a>
2. <a href="#2-aarm_gazebo">aarm_gazebo</a>
3. <a href="#3-aarm_control">aarm_control</a>
4. <a href="#4-aarm_moveit">aarm_moveit</a>
5. <a href="#5-aarm_brain">aarm_brain</a>

### 1. aarm_description

* Designing a urdf file for robot visualization and simulation
* Visualization of the robot in RVIZ

![picture alt](https://github.com/ake1999/aarm/blob/master/images/Screenshot_RVIZ.png "Screenshot_RVIZ")

### 2. aarm_gazebo

* physical simulation of robot and environment
* Using robot model in aarm_description pkg
* Designing the environment
* Running ROS controllers in aarm_control pkg
* Running rviz to visualize the robot and the camera video
* Running rqt-gui

![picture alt](https://github.com/ake1999/aarm/blob/master/images/Screenshot_gazebo.png "Screenshot_gazebo")
![picture alt](https://github.com/ake1999/aarm/blob/master/images/Screenshot_RVIZ_camera.png "Screenshot_RVIZ_camera")

### 3. aarm_control

* Setting PID parameters in a YAML ﬁle
* Loading ROS Controllers
* Running the rqt-gui to send the desired position to  the controllers
* plotting the desired and current position of joints
* optimizing PID parameters

![picture alt](https://github.com/ake1999/aarm/blob/master/images/Screenshot_controlers.png "Screenshot_controlers")

### 4. aarm_moveit

* Solving inverse kinematic
* Motion planning

![picture alt](https://github.com/ake1999/aarm/blob/master/images/Screenshot_moveit.jpg "Screenshot_moveit")

### 5. aarm_brain

implementation of DDPG and HER to train a policy to push an object to a randomly selected goal
Parts of trainBrain.py:
1. The ReplayBuffer Class for Saving Transition
2. The CriticNetwork, and ActorNetwork classes for building Actor and Critic neural networks.
3. The Agent class for implementation of DDPG
4. The environment class for communication with Gazebo:
    * Managing the time steps and resetting the environment
    * sending the actions and receiving the states
    * Implementation of the reward function and HER algorithm
5. The evaluate_policy function for policy evaluation
6. The main function

![picture alt](https://github.com/ake1999/aarm/blob/master/images/DDPG.jpg "DDPG")
## rqt_graph
* The global overview of the system. 
![picture alt](https://github.com/ake1999/aarm/blob/master/images/project_scheme.png "project.scheme")
