#!/usr/bin/env python
import os
import sys
import copy
import time
import random
import numpy as np

import rospy
from std_msgs.msg import Float64
from control_msgs.msg import JointControllerState
from gazebo_msgs.msg import LinkStates
from std_srvs.srv import Empty
from gazebo_msgs.msg import LinkState 
from gazebo_msgs.srv import SetLinkState
from rosgraph_msgs.msg import Clock

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('trajectory_planning', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_name = "aarm_group"
move_group = moveit_commander.MoveGroupCommander(group_name)

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

planning_frame = move_group.get_planning_frame()
#print ("============ Planning frame: %s" % planning_frame)

#eef_link = move_group.get_end_effector_link()
#print "============ End effector link: %s" % eef_link

# We can get a list of all the groups in the robot:
#group_names = robot.get_group_names()
#print "============ Available Planning Groups:", robot.get_group_names()

# Sometimes for debugging it is useful to print the entire state of the
# robot:
#print ("============ Printing robot state")
print (robot.get_current_state())
print ("")
#x=move_group.get_current_joint_values()
#print(x)
current_pose = move_group.get_current_pose().pose
print(current_pose)
pose_goal = geometry_msgs.msg.Pose()
#pose_goal = current_pose
pose_goal.orientation.w = 0.707141805047
pose_goal.orientation.x = 4.14906962942e-17
pose_goal.orientation.y = -8.93651676086e-17
pose_goal.orientation.z =  0.707071755591
pose_goal.position.x = -0.364944202832
pose_goal.position.y = 0.425026750367
pose_goal.position.z = 0.0755001251355

#print(pose_goal)
#print(move_group.get_goal_position_tolerance())
#print(move_group.get_goal_orientation_tolerance())
#move_group.set_planning_time(10);
#move_group.set_goal_position_tolerance(0.005)
#move_group.set_goal_orientation_tolerance(0.5)

move_group.set_pose_target(pose_goal)

plan = move_group.go(wait=True)
# Calling `stop()` ensures that there is no residual movement
move_group.stop()
# It is always good to clear your targets after planning with poses.
# Note: there is no equivalent function for clear_joint_value_targets()
move_group.clear_pose_targets()
