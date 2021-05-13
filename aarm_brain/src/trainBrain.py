#!/usr/bin/env python3
import os
import sys
import time
import random
import numpy as np
from time import sleep
from collections import deque
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from squaternion import Quaternion

import rospy
from std_msgs.msg import Float64
from control_msgs.msg import JointControllerState
from gazebo_msgs.msg import LinkStates
from std_srvs.srv import Empty
from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import SetLinkState
from rosgraph_msgs.msg import Clock

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class ReplayBuffer:
  def __init__(self, max_size, input_shape, n_actions):
    self.mem_size = max_size
    self.mem_cntr = 0
    self.state_memory = np.zeros((self.mem_size, *input_shape))
    self.new_state_memory = np.zeros((self.mem_size, *input_shape))
    self.action_memory = np.zeros((self.mem_size, n_actions))
    self.reward_memory = np.zeros(self.mem_size)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

  def store_transition(self, state, action, reward, state_, done):
    index = self.mem_cntr % self.mem_size

    self.state_memory[index] = state
    self.new_state_memory[index] = state_
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.terminal_memory[index] = done

    self.mem_cntr += 1

  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size)

    batch = np.random.choice(max_mem, batch_size, replace=False)

    states = self.state_memory[batch]
    states_ = self.new_state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    dones = self.terminal_memory[batch]

    return states, actions, rewards, states_, dones

class CriticNetwork(keras.Model):
  def __init__(self, fc1_dims=128, fc2_dims=128, fc3_dims=128, name='critic', chkpt_dir='tmp/ddpg'):
    super(CriticNetwork, self).__init__()
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.fc3_dims = fc3_dims
    #self.n_actions = n_actions

    self.model_name = name
    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(
        self.checkpoint_dir, self.model_name+'_ddpg.h5')

    self.fc1 = Dense(self.fc1_dims, activation='relu')
    self.fc2 = Dense(self.fc2_dims, activation='relu')
    self.fc3 = Dense(self.fc3_dims, activation='relu')
    self.q = Dense(1, activation=None)

  def call(self, state, action):
    action_value = self.fc1(tf.concat([state, action], axis=1))
    action_value = self.fc2(action_value)
    action_value = self.fc3(action_value)

    q = self.q(action_value)

    return q

class ActorNetwork(keras.Model):
  def __init__(self, fc1_dims=128, fc2_dims=128, fc3_dims=128, n_actions=2, name='actor', chkpt_dir='tmp/ddpg'):
    super(ActorNetwork, self).__init__()
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.fc3_dims = fc3_dims
    self.n_actions = n_actions

    self.model_name = name
    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

    self.fc1 = Dense(self.fc1_dims, activation='relu')
    self.fc2 = Dense(self.fc2_dims, activation='relu')
    self.fc3 = Dense(self.fc3_dims, activation='relu')
    self.mu = Dense(self.n_actions, activation='tanh')

  def call(self, state):
    prob = self.fc1(state)
    prob = self.fc2(prob)

    mu = self.mu(prob)

    return mu

class Agent:
  def __init__(self, alpha=0.001, beta=0.002, input_dims=[8], max_action=1, min_action=-1, 
                    gamma=0.99, n_actions=2, max_size=1000000, tau=0.05, batch_size=128):
    self.gamma = gamma
    self.tau = tau
    self.memory = ReplayBuffer(max_size, input_dims, n_actions)
    self.batch_size = batch_size
    self.n_actions = n_actions
    self.max_action = max_action
    self.min_action = min_action

    self.actor = ActorNetwork(n_actions=n_actions, name='actor')
    self.critic = CriticNetwork(name='critic')
    self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
    self.target_critic = CriticNetwork(name='target_critic')

    self.actor.compile(optimizer=Adam(learning_rate=alpha))
    self.critic.compile(optimizer=Adam(learning_rate=beta))
    self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
    self.target_critic.compile(optimizer=Adam(learning_rate=beta))

    self.update_network_parameters(tau=1)

  def update_network_parameters(self, tau=None):
    if tau is None:
      tau = self.tau

    weights = []
    targets = self.target_actor.weights
    for i, weight in enumerate(self.actor.weights):
      weights.append(weight * tau + targets[i]*(1-tau))
    self.target_actor.set_weights(weights)

    weights = []
    targets = self.target_critic.weights
    for i, weight in enumerate(self.critic.weights):
      weights.append(weight * tau + targets[i]*(1-tau))
    self.target_critic.set_weights(weights)

  def remember(self, state, action, reward, new_state, done):
    self.memory.store_transition(state, action, reward, new_state, done)

  def save_models(self):
    print('... saving models ...')
    self.actor.save_weights(self.actor.checkpoint_file)
    self.target_actor.save_weights(self.target_actor.checkpoint_file)
    self.critic.save_weights(self.critic.checkpoint_file)
    self.target_critic.save_weights(self.target_critic.checkpoint_file)

  def load_models(self):
    print('... loading models ...')
    self.actor.load_weights(self.actor.checkpoint_file)
    self.target_actor.load_weights(self.target_actor.checkpoint_file)
    self.critic.load_weights(self.critic.checkpoint_file)
    self.target_critic.load_weights(self.target_critic.checkpoint_file)

  def choose_action(self, observation, evaluate=False, probability=0.2):
    state = tf.convert_to_tensor([observation], dtype=tf.float32)
    actions = self.actor(state)[0]
    if not evaluate:
      if np.random.random() < probability:
        actions = tf.random.uniform(shape=[self.n_actions], minval=-1, maxval=1, dtype=tf.float32)
      else:
        actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.05)

    # note that if the environment has an action > 1, we have to multiply by
    # max action at some point

    actions = tf.clip_by_value(actions, self.min_action, self.max_action)

    return actions

  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return

    state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

    states = tf.convert_to_tensor(state, dtype=tf.float32)
    states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
    rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
    actions = tf.convert_to_tensor(action, dtype=tf.float32)

    with tf.GradientTape() as tape:
      target_actions = self.target_actor(states_)
      target_actions = tf.clip_by_value(target_actions,-1/(1-self.gamma),0)
      critic_value_ = tf.squeeze(self.target_critic(
          states_, target_actions), 1)
      critic_value = tf.squeeze(self.critic(states, actions), 1)
      target = reward + self.gamma*critic_value_*(1-done)
      critic_loss = keras.losses.MSE(target, critic_value)

    critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
    self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

    with tf.GradientTape() as tape:
      new_policy_actions = self.actor(states)
      actor_loss = -self.critic(states, new_policy_actions)
      actor_loss = tf.math.reduce_mean(actor_loss)

    actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

    #self.update_network_parameters()


class environment(object):
  def __init__(self, max_time=30, env_rt_factor = 1):
    rospy.init_node('trainBrain', anonymous=False)
    self.object_hight = 0
    self.done = 0
    self.prev_distance = 0
    self.sleep_time = 0
    self.delay = 0

    self.robot_state = np.array([0, 0, 0])

    # end effector
    self.robot_ef_position = np.array([0, 0])
    self.robot_ef_velocity = np.array([0, 0])

    self.object_position = np.array([0, 0, 0])
    self.object_velocity = np.array([0, 0, 0])
    self.target_position = np.array([0, 0])
    self.link0_position = np.array([0, 0, 0])

    self.delay_time = 0.05
    self.max_time = max_time
    self.max_actions = np.array([0.80, 0.36])
    self.max_robot_angles = np.array([0.45, 1.7453, np.pi])

    self.set_object_msg = LinkState()
    self.set_object_msg.link_name = 'object'
    self.set_object_msg.pose.position.x = -0.55
    self.set_object_msg.pose.position.y = 0
    self.set_object_msg.pose.position.z = 0.516
    self.set_object_msg.pose.orientation.x = 0
    self.set_object_msg.pose.orientation.y = 0
    self.set_object_msg.pose.orientation.z = 0
    self.set_object_msg.pose.orientation.w = 0

    self.set_target_msg = LinkState()
    self.set_target_msg.link_name = 'object_goal'
    self.set_target_msg.pose.position.x = 0.55
    self.set_target_msg.pose.position.y = 0
    self.set_target_msg.pose.position.z = 0.006
    self.set_target_msg.pose.orientation.x = 0
    self.set_target_msg.pose.orientation.y = 0
    self.set_target_msg.pose.orientation.z = 0
    self.set_target_msg.pose.orientation.w = 0

    self.joint1_command = rospy.Publisher(
        'aarm/joint1_position_controller/command', Float64, queue_size=10)
    self.joint2_command = rospy.Publisher(
        'aarm/joint2_position_controller/command', Float64, queue_size=10)
    self.joint3_command = rospy.Publisher(
        'aarm/joint3_position_controller/command', Float64, queue_size=10)
    #self.rate = rospy.Rate( env_rt_factor * 10)  # 10Xhz
    self.hz = env_rt_factor * 5

    rospy.Subscriber("aarm/joint1_position_controller/state",
                      JointControllerState, callback=self.joinState, callback_args=1)
    rospy.Subscriber("aarm/joint2_position_controller/state",
                      JointControllerState, callback=self.joinState, callback_args=2)
    rospy.Subscriber("aarm/joint3_position_controller/state",
                      JointControllerState, callback=self.joinState, callback_args=3)
    rospy.Subscriber("clock", Clock, callback=self.getTime)
    # rospy.s
    rospy.Subscriber("gazebo/link_states", LinkStates,
                      callback=self.getStates)
    rospy.wait_for_service('/gazebo/reset_simulation')
    self.reset_world = rospy.ServiceProxy(
        '/gazebo/reset_simulation', Empty)
    self.set_link_pose = rospy.ServiceProxy(
        '/gazebo/set_link_state', SetLinkState)

  def getTime(self, data):
    self.sim_time = data.clock.secs + data.clock.nsecs * 1e-9

  def joinState(self, data, num):
    self.robot_state[num-1] = data.process_value
    if np.abs(self.robot_state[2]) > np.pi:
      self.robot_state[2] = np.arctan2(np.sin(self.robot_state[2]),np.cos(self.robot_state[2]))

  def getStates(self, data):
    pose = data.pose
    self.link0_position = np.array([pose[4].position.x, pose[4].position.y, Quaternion(w=pose[4].orientation.w, 
                            x=pose[4].orientation.x, y=pose[4].orientation.y, z=pose[4].orientation.z).to_euler()[2]])
    self.object_position = np.array([pose[5].position.x, pose[5].position.y, Quaternion(w=pose[5].orientation.w, 
                            x=pose[5].orientation.x, y=pose[5].orientation.y, z=pose[5].orientation.z).to_euler()[2]])
    self.target_position = np.array([pose[7].position.x, pose[7].position.y])

    self.object_hight = data.pose[5].position.z

  def setPoses(self):
    randx = random.uniform(-0.1, 1)
    randy = random.uniform(-0.08, 0.1)
    randx_target = random.uniform(-1, 0.1)
    randy_target = random.uniform(-0.08, 0.1)
    #randz_target_rotation = random.uniform(0,np.pi/2)
    self.set_object_msg.pose.position.x = -0.55 + randx
    self.set_object_msg.pose.position.y = 0 + randy
    self.set_link_pose(self.set_object_msg)
    self.set_target_msg.pose.position.x = 0.55 + randx_target
    self.set_target_msg.pose.position.y = 0 + randy_target
    #self.set_target_msg.pose.orientation.w = Quaternion.from_euler(0, 0, randz_target_rotation).w
    #self.set_target_msg.pose.orientation.x = Quaternion.from_euler(0, 0, randz_target_rotation).x
    #self.set_target_msg.pose.orientation.y = Quaternion.from_euler(0, 0, randz_target_rotation).y
    #self.set_target_msg.pose.orientation.z = Quaternion.from_euler(0, 0, randz_target_rotation).z
    self.set_link_pose(self.set_target_msg)

  def calcIK(self, x_desired, y_desired):
    L1 = 0.51
    L2 = 0.425
    xo = 0.04
    yo = 0.36
    #print('robot state: ', self.robot_state)
    theta_desired = np.pi/2 + self.robot_state[1] + self. robot_state[2]
    for i in sorted(np.linspace(-np.pi, np.pi, num=180), key=abs):
      theta_desired += i
      q1_1 = np.arccos((yo + y_desired - L2 * np.cos(theta_desired - np.pi/2))/L1)
      if not np.isnan(q1_1) and np.abs(q1_1) < 100*np.pi/180:
        q1_2 = -q1_1
        q2_1 = theta_desired - np.pi/2 - q1_1
        q2_2 = theta_desired - np.pi/2 - q1_2
        q0_1 = x_desired + L1 * np.sin(q1_1) + L2*np.sin(theta_desired - np.pi/2) + xo
        q0_2 = x_desired + L1 * np.sin(q1_2) + L2*np.sin(theta_desired - np.pi/2) + xo

        if np.abs(q0_1) < 0.45 and np.abs(q0_2) < 0.45:
          if np.abs(q0_1 - self.robot_state[0]) <= np.abs(q0_2 - self.robot_state[0]):
            return q0_1, q1_1, q2_1
          else:
            return q0_2, q1_2, q2_2
        elif np.abs(q0_1) < 0.45:
          return q0_1, q1_1, q2_1
        elif np.abs(q0_2) < 0.45:
          return q0_2, q1_2, q2_2

    print('error: cant calculate IK for this point  ', [x_desired, y_desired])
    return self.robot_state[0], self.robot_state[1], self.robot_state[2]

  def observation(self, is_starting):

    end_effector_position = np.array([self.link0_position[0] + 0.425 * np.cos(
        self.link0_position[2]), self.link0_position[1] + 0.425 * np.sin(self.link0_position[2])])
    #print('robot_state: ', self.robot_state)
    #print('ik: ', self.calcIK(end_effector_position[0], end_effector_position[1]-0.1))

    if not is_starting:
      delta_t = (self.sim_time-self.prev_sim_time)
      object_velocity = ((self.object_position - self.prev_object_position)/delta_t)
      end_effector_velocity = ((end_effector_position - self.prev_end_effector_position)/delta_t)
      obs = np.concatenate((self.target_position, end_effector_position, end_effector_velocity, self.object_position, object_velocity))
    else:
      obs = np.concatenate((self.target_position, end_effector_position, [0, 0], self.object_position, [0, 0, 0]))

    self.prev_object_position = self.object_position.copy()
    self.prev_end_effector_position = end_effector_position.copy()
    self.prev_sim_time = self.sim_time

    return obs

  def calcReward(self):

    distance = np.sqrt((self.target_position[0]-self.object_position[0])**2+(
                              self.target_position[1]-self.object_position[1])**2)
    reward = -1
    if distance < 0.02:
      reward = 0
    self.prev_distance = distance.copy()

    return reward

  def isDone(self):
    self.done = 0

    if self.object_hight < self.table_hight or self.object_hight > self.table_hight + 0.07:
      self.done = 1

    if self.prev_distance < 0.02 or self.sim_time > self.max_time:
      self.done = 1

  def step(self, action):
    x_desired, y_desired = action
    command1, command2, command3 = self.calcIK(x_desired*self.max_actions[0], y_desired*self.max_actions[1]) 

    self.joint1_command.publish(command1)
    self.joint2_command.publish(command2)
    self.joint3_command.publish(command3)

    #self.rate.sleep()
    self.delay = time.time() - self.delay
    sleep(((1/self.hz - self.delay)+np.abs(1/self.hz - self.delay))/2)
    self.delay = time.time()

    self.isDone()

    reward = self.calcReward()

    obs = self.observation(False)

    return obs, reward, self.done

  def reset(self):
      self.reset_world()
      self.joint1_command.publish(0)
      self.joint2_command.publish(0)
      self.joint3_command.publish(0)
      self.setPoses()

      sleep(0.1)
      self.prev_distance = np.sqrt((self.target_position[0]-self.object_position[0])**2+(
          self.target_position[1]-self.object_position[1])**2)
      self.table_hight = self.object_hight - 0.035

      obs = self.observation(True)

      return obs

  def HER(self, state, next_state, reward, done, virtual_target):
        object_position = state[6:9]
        next_object_position = next_state[6:9]
        #target = state[0:2]
        virtual_state = np.concatenate((virtual_target[0:2], state[2:]))
        virtual_next_state = np.concatenate((virtual_target[0:2], next_state[2:]))

        #virtual_distance = np.sqrt(
        #    (virtual_target[0]-object_position[0])**2+(virtual_target[1]-object_position[1])**2)
        virtual_next_distance = np.sqrt(
            (virtual_target[0]-next_object_position[0])**2+(virtual_target[1]-next_object_position[1])**2)
        
        if virtual_next_distance < 0.02: virtual_reward = 0
        else: virtual_reward = -1
        
        if virtual_next_distance < 0.02 or done: virtual_done = 1
        else: virtual_done = 0

        return virtual_state, virtual_next_state, virtual_reward, virtual_done


def evaluate_policy(agent, env, eval_episodes=10):
  win_rate = 0
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = agent.choose_action(obs, evaluate=True)
      obs, reward, done = env.step(action)
      if reward == 0:
        win_rate += 1
  win_rate /= eval_episodes
  return win_rate

def main():
  n_epochs = 200
  n_cycles = 50
  n_episodes = 16
  n_optimization_steps = 40
  batch_size = 128
  replay_buffer_size = 1000000
  tau = 0.05
  learning_rate = 0.001
  gamma = 0.99
  her_k = 6
  max_env_time = 20
  real_time_rate = 10
  probability_factor = 0.99995
  random_action_probability = probability_factor
  input_dims = [12] #######################################
  n_actions = 2 ######################################
  figure_file = 'plots/pendulum.png'
  load_checkpoint = False
  total_episodes = 0
  total_cycles = 0
  total_epochs = 0

  env = environment(max_time= max_env_time, env_rt_factor = real_time_rate)
  agent = Agent(alpha=learning_rate, beta=learning_rate, input_dims=input_dims, gamma=gamma, 
                n_actions=n_actions, max_size=replay_buffer_size, tau=tau, batch_size=batch_size)
  
  plot_data = [0]
  plt.plot([0])
  plt.ylabel('win rates')
  ax = plt.gca()
  plt.pause(0.05)
    
  if load_checkpoint:
    agent.load_models()
  
  for epoch_num in range(n_epochs):
    total_epochs += 1 
    total_cycles = 0
    for cycle_num in range(n_cycles):
      total_cycles += 1
      total_episodes = 0
      for episode_num in range(n_episodes):
        total_episodes += 1
        rewards = 0
        state = env.reset()
        starting_object_hight = env.object_hight
        short_memory ={'state':[], 'action':[], 'next_state':[], 'reward':[], 'done':[]}
        virtual_targets = []
        done = 0

        while not done:
          random_action_probability *= probability_factor
          action = agent.choose_action(state, evaluate=False, probability=random_action_probability).numpy()
          next_state, reward, done = env.step(action)
          rewards += reward
          short_memory['state'].append(state.copy())
          short_memory['action'].append(action.copy())
          short_memory['next_state'].append(next_state.copy())
          short_memory['reward'].append(reward)
          short_memory['done'].append(done)
          agent.remember(state, action, reward, next_state, done)
          if np.abs(starting_object_hight - env.object_hight)<0.01 and np.abs(env.object_position[0]) < 0.7 and np.abs(env.object_position[1]) < 0.3:
            virtual_targets.append(np.array(np.round(env.object_position[:2], decimals=2)))
        print('epoch: ', total_epochs,'cycle: ', total_cycles,'episode: ', total_episodes,'  win: ', reward+1)
        if len(virtual_targets) > 2*her_k:
          choices = np.sort(np.append(np.random.choice(len(virtual_targets)-1, her_k-1, replace=False),len(virtual_targets)-1))
          prev_virtual_target = np.zeros((n_actions))
          for virtual_target in np.array(virtual_targets)[choices]:
            if np.any(virtual_target != prev_virtual_target):
              for i in range(len(short_memory['state'])):
                virtual_state, virtual_next_state, virtual_reward, virtual_done= env.HER(short_memory['state'][i], short_memory['next_state'][i], short_memory['reward'][i], short_memory['done'][i], virtual_target)
                agent.remember(virtual_state, short_memory['action'][i], virtual_reward, virtual_next_state, virtual_done)
                if virtual_done == True:
                  break
              prev_virtual_target = virtual_target.copy()
        
        for _ in range(n_optimization_steps):
          agent.learn()
      
      agent.update_network_parameters()

    agent.save_models()
    plot_data = evaluate_policy(agent, env, eval_episodes=10)
    ax.clear()
    plt.plot(plot_data)
    plt.ylabel('win rates')
    plt.savefig('plot_reward.png')
    plt.pause(0.05)

if __name__ == '__main__':
  main()
