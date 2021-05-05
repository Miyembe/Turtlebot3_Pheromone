#! /usr/bin/env python

# The turtlebot is trained to find the optimal velocity (Twist) with given pheromone data
# The environment is given in the pair script (phero_turtlebot_turtlebot3_ppo.py)
# The expected result is following the pheromone in the most smooth way! even more than ants

#import phero_turtlebot_turtlebot3_ppo
import phero_turtlebot_exp1
import numpy as np
import os
import sys
print(sys.path)
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from distributions import make_pdtype
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, HighlightReplayBuffer
from schedule import LinearSchedule
import os.path as osp
import joblib
import tensorboard_logging
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse

from ActorCritic import Actor, Critic
from torch_util import *

import logger

HOME = os.environ['HOME']
#from numba import jit



def stack_samples(samples):
	
	current_states = np.squeeze(np.asarray(samples[0]))
	actions = np.squeeze(np.asarray(samples[1]))
	rewards = np.squeeze(np.asarray(samples[2]))
	new_states = np.squeeze(np.asarray(samples[3]))
	dones = np.squeeze(np.asarray(samples[4]))
	weights = np.squeeze(np.asarray(samples[5]))
	batch_idxes = np.squeeze(np.asarray(samples[6]))
	#before_current_states = np.stack(array[:,0])
	# current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
	# actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
	# rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
	# new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
	# dones = np.stack(array[:,4]).reshape((array.shape[0],-1))

	return current_states, actions, rewards, new_states, dones, weights, batch_idxes

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value

class ExperienceReplayBuffer:
	def __init__ (self,
				  total_timesteps=100000,
				  buffer_size=50000,
				  type_buffer="PER",
				  prioritized_replay=True,
				  prioritized_replay_alpha=0.6,
				  prioritized_replay_beta0=0.4,
				  prioritized_replay_beta_iters=None,
				  prioritized_replay_eps=1e-6):
		self.buffer_size = buffer_size
		self.prioritized_replay_eps = prioritized_replay_eps
		self.type_buffer = type_buffer
		if prioritized_replay:
			if type_buffer == "PER":
				self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
			if type_buffer == "HER":
				self.replay_buffer = HighlightReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
			if prioritized_replay_beta_iters is None:
				prioritized_replay_beta_iters = total_timesteps
			self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
											initial_p = prioritized_replay_beta0,
											final_p = 1.0)
		else: 
			self.replay_buffer = ReplayBuffer(buffer_size)
			self.beta_schedule = None
	def add(self, obs_t, action, reward, obs_tp1, done):
		self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.num_robots = env.num_robots
		self.sess = sess

		self.learning_rate = 0.0001
		self.epsilon = .9
		self.epsilon_decay = .99995
		self.gamma = .90
		self.tau   = .01


		self.buffer_size = 1000000
		self.batch_size = 512

		self.hyper_parameters_lambda3 = 0.2
		self.hyper_parameters_eps = 0.2
		self.hyper_parameters_eps_d = 0.4

		self.demo_size = 1000

		self.demo_size = 1000
		self.time_str = time.strftime("%Y%m%d-%H%M%S")
		self.parent_dir = HOME + "/catkin_ws/src/Turtlebot3_Pheromone/src/DRLbasedController/weights"
		self.path = os.path.join(self.parent_dir, self.time_str)
		os.mkdir(self.path)

		# Replay buffer
		self.memory = deque(maxlen=1000000)
		# Replay Buffer
		self.replay_buffer = ExperienceReplayBuffer(total_timesteps=5000*256, type_buffer="HER")
		# File name
		self.file_name = "reward_{}_{}_{}".format(self.time_str, self.num_robots, self.replay_buffer.type_buffer)
		self.hid_list = [512, 512, 512]

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		
		self.actor_model = Actor(self.env.observation_space.shape, self.env.action_space.shape, self.hid_list)
		self.actor_model.reset_parameters()
		self.target_actor_model = Actor(self.env.observation_space.shape, self.env.action_space.shape, self.hid_list)
		self.target_actor_model.reset_parameters()
		self.actor_optim = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)


		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #

		self.critic_model = Critic(self.env.observation_space.shape, self.env.action_space.shape, 1, self.hid_list)
		self.critic_model.reset_parameters()
		self.target_critic_model = Critic(self.env.observation_space.shape, self.env.action_space.shape, 1, self.hid_list)
		self.target_critic_model.reset_parameters()
		self.critic_optim = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
		

		hard_update(self.target_actor_model, self.actor_model) # Make sure target is with the same weight
		hard_update(self.target_critic_model, self.critic_model)

		self.cuda()
		

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])



	def _train_critic_actor(self, samples):
 

		Loss = nn.MSELoss()

		# 1, sample
		# cur_states, actions, rewards, new_states, done = stack_samples(samples)
		cur_states, actions, rewards, new_states, dones, weights, batch_idxes = stack_samples(samples) # PER version also checks if I need to use stack_samples
		target_actions = to_numpy(self.target_actor_model(to_tensor(new_states)))

		# print("cur_states is %s", cur_states)

		# evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0, sample_weight=_sample_weight)
		#evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0)
		# print('\nhistory dict:', evaluation.history)

		# Critic Update
		self.critic_model.zero_grad()
		Q_now = self.critic_model([cur_states, actions])
		next_Q = self.target_critic_model([new_states, target_actions])
		dones = dones.astype(bool)
		Q_target = to_tensor(rewards) + self.gamma*next_Q.reshape(next_Q.shape[0]) * to_tensor(1 - dones)	

		td_errors = Q_target - Q_now.reshape(Q_now.shape[0])

		value_loss = Loss(Q_target, Q_now.squeeze())
		value_loss.backward()
		self.critic_optim.step()

		# Actor Update
		self.actor_model.zero_grad()
		policy_loss = -self.critic_model([
			to_tensor(cur_states),
			self.actor_model(to_tensor(cur_states))
		])
		policy_loss = policy_loss.mean()
		policy_loss.backward()
		self.actor_optim.step()

		# NoisyNet noise reset
		self.actor_model.reset_noise()
		self.target_actor_model.reset_noise()


		return td_errors
		



	def read_Q_values(self, cur_states, actions):
		critic_values = self.critic_model.predict([cur_states, actions])
		return critic_values

	def train(self, t):
		batch_size = self.batch_size
		if self.replay_buffer.replay_buffer.__len__() < batch_size: #per
			return
		
		samples = self.replay_buffer.replay_buffer.sample(batch_size, beta=self.replay_buffer.beta_schedule.value(t))
		(obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = samples
		
		self.samples = samples
		td_errors = self._train_critic_actor(samples)

		# priority updates
		#new_priorities = np.abs(td_errors) + self.replay_buffer.prioritized_replay_eps
		#self.replay_buffer.replay_buffer.update_priorities(batch_idxes, new_priorities)


	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		soft_update(self.target_actor_model, self.actor_model, self.tau)
        
	def _update_critic_target(self):
		soft_update(self.target_critic_model, self.critic_model, self.tau)

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):  # this function returns action, which is predicted by the model. parameter is epsilon
		self.epsilon *= self.epsilon_decay
		eps = self.epsilon
		#print("cur_state: {}".format(cur_state))
		#print("cur_state_size: {}".format(cur_state.shape))
		cur_state = np.array(cur_state).reshape(1,8)
		action = to_numpy(self.actor_model(to_tensor(cur_state))).squeeze(0)
		action = action.reshape(1,2)
		if np.random.random() < self.epsilon:
			action[0][0] = action[0][0]+ (np.random.random()-0.5)*0.4
			action[0][1] = action[0][1]+ (np.random.random())*0.4
			print("action: {}, shape: {}".format(action, action.shape))
			return action, eps	
		else:
			action[0][0] = action[0][0] 
			action[0][1] = action[0][1]
			print("action: {}, shape: {}".format(action, action.shape))
			return action, eps
		
		

	# ========================================================================= #
	#                              save weights                                 #
	# ========================================================================= #

	def save_weight(self, num_trials, trial_len):
		torch.save(
			self.actor_model.state_dict(),
			self.path + '/actormodel' + '-' +  str(num_trials) + '-' + str(trial_len) +'.pkl'
		)
		torch.save(
			self.critic_model.state_dict(),
			self.path + '/criticmodel' + '-' +  str(num_trials) + '-' + str(trial_len) + '.pkl'
		)
		#self.actor_model.save_weights(self.path + 'actormodel' + '-' +  str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)
		#self.critic_model.save_weights(self.path + 'criticmodel' + '-' + str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)#("criticmodel.h5", overwrite=True)

	def load_weights(self, output):

		self.actor_model.load_state_dict(
			torch.load('{}.pkl'.format(output))
		)

		self.critic_model.load_state_dict(
			torch.load('{}.pkl'.format(output))
		)

	def play(self, cur_state):
		return to_numpy(self.actor_model(to_tensor(cur_state), volatile = True)).squeeze(0)

	def cuda(self):
		self.actor_model.cuda()
		self.target_actor_model.cuda()
		self.critic_model.cuda()
		self.target_critic_model.cuda()

def safemean(xs):
       return np.nan if len(xs) == 0 else np.mean(xs)

def main(args):
	time_str = time.strftime("%Y%m%d-%H%M%S")
	logger_ins = logger.Logger(HOME + '/catkin_ws/src/Turtlebot3_Pheromone/src/log', output_formats=[logger.HumanOutputFormat(sys.stdout)])
	board_logger = tensorboard_logging.Logger(os.path.join(logger_ins.get_dir(), "tf_board", time_str))
	sess = tf.Session()
	K.set_session(sess)
	########################################################
	game_state= phero_turtlebot_exp1.Env()   # game_state has frame_step(action) function
	actor_critic = ActorCritic(game_state, sess)
	random.seed(args.random_seed)
	########################################################
	num_trials = 400
	trial_len  = 256
	log_interval = 5
	train_indicator = 1
	tfirststart = time.time()

	# Reward Logging
	with open(HOME + '/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(actor_critic.file_name), mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['Episode', 'Average Reward'])

	# Double ended queue with max size 100 to store episode info
	epinfobuf = deque(maxlen=100)
	num_robots = game_state.num_robots
	current_state = game_state.reset()

	# actor_critic.read_human_data()
	
	step_reward = np.array([0, 0]).reshape(1,2)
	step_Q = [0,0]
	step = 0

	if (train_indicator==2):
		for i in range(num_trials):
			print("trial:" + str(i))
			#game_state.step(0.3, 0.2, 0.0)
			#game_state.reset()

			current_state = game_state.reset()
			##############################################################################################
			total_reward = 0
			
			for j in range(100):
				step = step +1
				#print("step is %s", step)


				###########################################################################################
				#print('wanted value is %s:', game_state.observation_space.shape[0])
				current_state = current_state.reshape((1, game_state.observation_space.shape[0]))
				action, eps = actor_critic.act(current_state)
				action = action.reshape((1, game_state.action_space.shape[0]))
				print("action is speed: %s, angular: %s", action[0][1], action[0][0])
				_, new_state, reward, done, _ = game_state.step(0.1, action[0][1]*5, action[0][0]*5) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
				total_reward = total_reward + reward
				

	

	if (train_indicator==1):

		# actor_critic.actor_model.load_weights("actormodel-90-1000.h5")
		# actor_critic.critic_model.load_weights("criticmodel-90-1000.h5")
		for i in range(num_trials):
			print("trial:" + str(i))
			
			#game_state.step(0.3, 0.2, 0.0)
			#game_state.reset()
			

			_, current_state = game_state.reset()
			##############################################################################################
			total_reward = 0
			epinfos = []
			for j in range(trial_len):
				
				###########################################################################################
				#print('wanted value is %s:', game_state.observation_space.shape[0])
				current_state = current_state.reshape((1, game_state.observation_space.shape[0]))
				action, eps = actor_critic.act(current_state)
				print("action is speed: %s, angular: %s", action[0][1], action[0][0])
				_, new_state, reward, done, info = game_state.step(0.1, linear_x = action[0][1], angular_z = action[0][0]) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
				total_reward = total_reward + reward
				###########################################################################################

				if j == (trial_len - 1):
					done = np.array([True]).reshape(game_state.num_robots, 1)
				
				
				step = step + 1
				#plot_reward(step,reward,ax,fig)
				#step_reward = np.append(step_reward,[step,reward])
				#step_start = time.time()
				#sio.savemat('step_reward.mat',{'data':step_reward},True,'5', False, False,'row')
				#print("step is %s", step)
				#print("info: {}".format(info[0]['episode']['r']))
				#Q_values = actor_critic.read_Q_values(current_state, action)
				#step_Q = np.append(step_Q,[step,Q_values[0][0]])
				#print("step_Q is %s", Q_values[0][0])
				#sio.savemat('step_Q.mat',{'data':step_Q},True,'5', False, False,'row')

				epinfos.append(info[0]['episode'])
				
				start_time = time.time()

				if (j % 5 == 0):
					actor_critic.train(j)
					actor_critic.update_target()   

				end_time = time.time()
				print("Train time: {}".format(end_time - start_time))
				#print("new_state: {}".format(new_state))
				new_state = new_state.reshape((1, game_state.observation_space.shape[0]))

				# print shape of current_state
				#print("current_state is %s", current_state)
				##########################################################################################
				actor_critic.remember(current_state, action, reward, new_state, done)
				actor_critic.replay_buffer.add(current_state, action, reward, new_state, done)
				current_state = new_state


				
				##########################################################################################
			if (i % 10==0):
				actor_critic.save_weight(i, trial_len)
			epinfobuf.extend(epinfos)
			tnow = time.time()
			#fps = int(nbatch / (tnow - tstart))
			
			##################################################
            ##      Logging and saving model & weights      ##
            ##################################################

			if i % log_interval == 0 or i == 0:
				#ev = explained_variance(values, returns)
				reward_mean = safemean([epinfo['r'] for epinfo in epinfobuf])
				logger_ins.logkv("serial_timesteps", i*trial_len)
				logger_ins.logkv("nupdates", i)
				logger_ins.logkv("total_timesteps", i*trial_len)
				logger_ins.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
				logger_ins.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
				logger_ins.logkv('time_elapsed', tnow - tfirststart)
				# for (lossval, lossname) in zip(lossvals, model.loss_names):
				#     logger_ins.logkv(lossname, lossval)
				# logger_ins.dumpkvs()
				# for (lossval, lossname) in zip(lossvals, model.loss_names):
				#     board_logger.log_scalar(lossname, lossval, update)
				board_logger.log_scalar("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]), i)
				board_logger.flush()
				with open(HOME + '/catkin_ws/src/Turtlebot3_Pheromone/src/log/csv/{}.csv'.format(actor_critic.file_name), mode='a') as csv_file:
					csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					csv_writer.writerow(['%i'%i, '%0.2f'%reward_mean])

		

	if train_indicator==0:
		for i in range(num_trials):
			print("trial:" + str(i))
			current_state = game_state.reset()
			
			actor_critic.actor_model.load_weights(self.path + "actormodel-160-500.h5")
			actor_critic.critic_model.load_weights(self.path + "criticmodel-160-500.h5")
			##############################################################################################
			total_reward = 0
			
			for j in range(trial_len):

				###########################################################################################
				current_state = current_state.reshape((1, game_state.observation_space.shape[0]))

				start_time = time.time()
				action = actor_critic.play(current_state)  # need to change the network input output, do I need to change the output to be [0, 2*pi]
				action = action.reshape((1, game_state.action_space.shape[0]))
				end_time = time.time()
				print(1/(end_time - start_time), "fps for calculating next step")

				_, new_state, reward, done = game_state.step(0.1, action[0][1], action[0][0]) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
				total_reward = total_reward + reward
				###########################################################################################

				if j == (trial_len - 1):
					done = 1
					#print("this is reward:", total_reward)
					

				# if (j % 5 == 0):
				# 	actor_critic.train()
				# 	actor_critic.update_target()   
				
				new_state = new_state.reshape((1, game_state.observation_space.shape[0]))
				# actor_critic.remember(cur_state, action, reward, new_state, done)   # remember all the data using memory, memory data will be samples to samples automatically.
				# cur_state = new_state

				##########################################################################################
				#actor_critic.remember(current_state, action, reward, new_state, done)
				current_state = new_state

				##########################################################################################



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args("")
	args.exp_name = "exp_random_seed"
	name_var = 'random_seed'
	list_var = [66, 51, 33]
	for var in list_var:
		setattr(args, name_var, var)
		print(args)
		main(args)

