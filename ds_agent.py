"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time
import os

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from models import *
from env import SingleStepEnv
from transformers import AutoTokenizer
import deepspeed

class Agent:
	"""
		ds_agent = deepspeed agent.
		This is the PPO agent class we will use as our model in main.py
	"""
	def __init__(self, policy_type, data_path, checkpoint='meta-llama/Llama-3.2-1B', auth_token=None, hyperparameters=None):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_type - the policy class to use for our actor/critic networks.
				tokenizer_class - the tokenizer class to use for our language models.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		policy_class = getPolicyClass(policy_type)
		tokenizer_class = getTokenizerClass(policy_type)
		self.env = SingleStepEnv(data_path)
		self.instructions = self.env.instructions
		# Initialize hyperparameters for training
		self._init_hyperparameters(hyperparameters)
		
		self.tokenizer = tokenizer_class.from_pretrained(checkpoint, token=auth_token)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		#self.tokenizer.chat_template = "<prompt_template>"
		
		# Initialize actor and critic networks
		self.actor = policy_class(checkpoint, self.tokenizer, token=auth_token)
		
		# Check if there's a detuned version of this model
		model_prefix = self.model_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")
		detuned_path = f'./models/{model_prefix}-detuned'
		
		# If detuned model exists and has actor.pt, load it into the actor model
		if os.path.exists(detuned_path) and os.path.exists(f'{detuned_path}/actor.pt'):
			print(f"Found detuned model at {detuned_path}, loading weights")
			self.actor.final.load_state_dict(torch.load(f'{detuned_path}/actor.pt'))
			print("Detuned model weights loaded successfully")
		
		self.critic = self.actor.value_head #this is only one layer, to define the optimizer. The forward functionality is in the actor
		self.decoder = self.actor.final
		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.decoder.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.value_lr)

		self.actor, self.actor_optim, _, _ = deepspeed.initialize(
			model=self.actor,
			optimizer=self.actor_optim,
			config_params="deepspeed_config.json"
		)
		self.critic, self.critic_optim, _, _ = deepspeed.initialize(
			model=self.critic,
			optimizer=self.critic_optim,
			config_params="deepspeed_config.json"
		)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

		#other hyperparams
		self.save_freq = 10

	def learn(self):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		t = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t < self.episodes:           

			if t % 10 == 0:
				rollout, value_rollout = self.rollout_from_simulator(i_so_far)
			else:                                                            
				rollout, value_rollout = self.rollout_from_data(i_so_far)   

			self.actor.module.train()
			self.critic.module.train()
			self.actor.module.print_memory("Before training")
			# Calculate how many timesteps we collected this batch
			t += np.sum(rollout.batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t'] = t
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			Advantage = rollout.batch_rews                                                                       

			#normalize advantages
			A_k = (Advantage - Advantage.mean()) / (Advantage.std() + 1e-10)
			self.actor.module.print_memory("After advantage calculation")
			# Shuffle the value rollout data
			perm = torch.randperm(value_rollout.batch_seq.size(0))
			value_batch_seq_shuffled = value_rollout.batch_seq[perm]
			value_batch_rews_shuffled = value_rollout.batch_rews[perm]

			# Minibatch training
			for j in range(0, value_rollout.batch_seq.size(0), self.batch_size//self.epochs):
				minibatch_seq = value_batch_seq_shuffled[j:j + self.batch_size]
				minibatch_rews = value_batch_rews_shuffled[j:j + self.batch_size]

				# Value updates
				_, critic_loss = self.value_step(minibatch_seq, minibatch_rews)
				# Calculate gradients and perform backward propagation for critic network
				self.critic.zero_grad()
				self.critic.backward(critic_loss)
				self.critic.step()
				torch.cuda.empty_cache()

			# This is the loop where we update our network for some n epochs
			# Shuffle the rollout data
			perm = torch.randperm(rollout.batch_seq.size(0))
			batch_seq_shuffled = rollout.batch_seq[perm]
			batch_acts_shuffled = rollout.batch_acts[perm]
			batch_log_probs_shuffled = rollout.batch_log_probs[perm]
			batch_rews_shuffled = rollout.batch_rews[perm]

			# Minibatch training
			for j in range(0, rollout.batch_seq.size(0), self.batch_size//self.epochs):
				print(self.actor.module.print_memory("training"))
				minibatch_seq = batch_seq_shuffled[j:j + self.batch_size]
				minibatch_acts = batch_acts_shuffled[j:j + self.batch_size]
				minibatch_log_probs = batch_log_probs_shuffled[j:j + self.batch_size]
				minibatch_rews = batch_rews_shuffled[j:j + self.batch_size]

				# Calculate V_phi and pi_theta(a_t | s_t)
				curr_log_probs = self.get_curr_log_probs(minibatch_seq, minibatch_acts)
				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = torch.exp(curr_log_probs - minibatch_log_probs.to(curr_log_probs.device))

				# Calculate surrogate losses.
				A_k = minibatch_rews.to(ratios.device)
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate the semantic similarity between question and rephrase
				sim_score = self.actor.module.get_embed_similarity(minibatch_seq, minibatch_acts, torch.ones_like(minibatch_seq), torch.ones_like(minibatch_acts))

				# Calculate language modelling loss
				actor_loss = (-torch.min(surr1, surr2)).mean() + sim_score.mean().to(ratios.device) * self.sim_coef

				# Calculate gradients and perform backward propagation for actor network
				self.actor.zero_grad()
				self.actor.backward(actor_loss)
				self.actor.step()
				torch.cuda.empty_cache()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.module.final.state_dict(), f'./models/{self.model_name}/actor.pt')
				torch.save(self.actor.module.value_head.state_dict(), f'./models/{self.model_name}/critic.pt')
				
				# Check if this model has been detuned
				flags_file = f'./models/{self.model_name}/flags'
				is_detuned = False
				if os.path.exists(flags_file):
					with open(flags_file, 'r') as file:
						if 'detuned' in file.read():
							is_detuned = True
				
				# If the model is detuned, also save a copy to the detuned folder
				if is_detuned:
					model_prefix = self.model_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")
					detuned_path = f'./models/{model_prefix}-detuned'
					
					# Create the detuned directory if it doesn't exist
					if not os.path.exists(detuned_path):
						os.makedirs(detuned_path)
					
					# Save the actor model to the detuned folder
					torch.save(self.actor.module.final.state_dict(), f'{detuned_path}/actor.pt')
					
					# Save a flag file indicating this is a detuned model
					with open(f'{detuned_path}/flags', 'w') as file:
						file.write("detuned")
					
					print(f"Saved detuned model to {detuned_path}")

	def value_step(self, input, reward):
		'''
		Compute the value of a given input sequence and propagate loss
		'''
		#compute bellman reward, once we have multiple steps
		input = self.tokenizer.batch_decode(input, skip_special_tokens=True)
		tokens = self.tokenizer(input, return_tensors='pt', padding=True)
		input_ids = tokens.input_ids.to('cuda')
		attention_mask = tokens.attention_mask.to('cuda')
		value, attention = self.actor.module.value(input_ids, attention_mask)
		value = value.squeeze().sum(dim=1)
		#right now, the value is for each token, but we want it to be for the whole sequence
		#to do this we should multiply the value by the attention mask and sum over the sequence
		#such that the sum of the value is the probability that the sequence is adversarial
		#accordingly, the value of a token is the probability that the token is adversarial
		loss = nn.MSELoss()(value, reward.to('cuda'))
		
		return value, loss
	
	def get_curr_log_probs(self, batch_seq, batch_acts):
		"""
			Compute the value of a given input sequence and the log probabilities of the actions taken.

			Parameters:
				batch_seq - the observations collected during the rollout
				batch_acts - the actions taken during the rollout

			Returns:
				value - the value of the input sequence
				log_probs - the log probabilities of the actions taken
		"""
		with self.actor.module.eval():
			if torch.is_tensor(batch_acts):
				batch_acts = self.tokenizer.batch_decode(batch_acts, skip_special_tokens=False)
			log_probs = self.actor.module.get_log_probs(batch_seq, batch_acts)
			return log_probs
	
	def evaluate(self, batch_seq, return_attn=False):
		"""
		Evaluates the given batch of sequences to get their values.
		Args:
			batch_seq (torch.Tensor): A batch of input sequences.
			batch_acts (torch.Tensor or list): A batch of actions, either as tensors or decoded sequences.
		Returns:
			torch.Tensor: The values of the sequences.
		"""
		with self.actor.module.eval():
			if torch.is_tensor(batch_seq):
				batch_seq = self.tokenizer.batch_decode(batch_seq, skip_special_tokens=False)

			output_tokens = self.tokenizer(batch_seq, return_tensors='pt', padding=True)
			value, attn = self.actor.module.value(output_tokens.input_ids, output_tokens.attention_mask)
		return (value, attn) if return_attn else value

	def rollout_from_data(self, i_so_far):
		"""
			Perform a rollout by interacting with the existing dataset.

			Returns:
				batch_seq - the observations collected during the rollout
				batch_acts - the actions taken during the rollout
				batch_log_probs - the log probabilities of the actions taken
				batch_rtgs - the rewards-to-go calculated from the rewards
				batch_lens - the lengths of each episode
		"""
		batch_seq = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []

		value_batch_seq = []
		value_batch_rews = []

		self.actor.module.eval()
		self.critic.module.eval()

		t = 0  # Keeps track of how many timesteps we've run so far this batch
		start_time = time.time()
		while t < self.batch_size:
			ep_rews = []  # Rewards collected per episode

			question, no_rephrase_reward, rephrase, rephrase_reward, reward = self.env.step()
			# Track recent reward, action, and action log probability
			batch_seq.append(*question)
			
			batch_rews.append(*reward)
			batch_acts.append(*rephrase)

			#track the same information but question/reward pairwise for value training
			value_batch_seq.append(*question)
			value_batch_rews.append(*no_rephrase_reward)
			value_batch_seq.append(*rephrase)
			value_batch_rews.append(*rephrase_reward)

			t += 1

			# Track episodic lengths and rewards
			ep_rews.append(ep_rews)

		# Tokenize the sequences
		batch_seq = self.tokenizer(batch_seq, return_tensors='pt', padding=True).input_ids
		batch_acts = self.tokenizer(batch_acts, return_tensors='pt', padding=True).input_ids
		batch_rews = torch.tensor(batch_rews, dtype=torch.float)
		value_batch_seq = self.tokenizer(value_batch_seq, return_tensors='pt', padding=True).input_ids
		value_batch_rews = torch.tensor(value_batch_rews, dtype=torch.float)

		#compute log probabilities of (output | input)
		batch_log_probs = self.actor.module.get_log_probs(batch_seq, batch_acts).detach()

		print(f"Rollouts collected: {t} in {time.time() - start_time} seconds")
		self.actor.module.print_memory("After rollout")
		
		# Create a rollout object and return it
		rollout = type('Rollout', (object,), {
			'batch_seq': batch_seq,
			'batch_acts': batch_acts,
			'batch_log_probs': batch_log_probs,
			'batch_rews': batch_rews,
			'batch_lens': [self.batch_size]
		})()
		value_rollout = type('ValueRollout', (object,), {
			'batch_seq': value_batch_seq,
			'batch_rews': value_batch_rews
		})()

		return rollout, value_rollout
	
	def rollout_from_simulator(self, i_so_far):
		"""
			Perform a rollout by interacting with the environment.

			Returns:
				batch_seq - the observations collected during the rollout
				batch_acts - the actions taken during the rollout
				batch_log_probs - the log probabilities of the actions taken
				batch_rtgs - the rewards-to-go calculated from the rewards
				batch_lens - the lengths of each episode
		"""
		batch_seq = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []

		value_batch_seq = []
		value_batch_rews = []

		self.actor.module.eval()
		self.critic.module.eval()

		start_time = time.time()

		ep_rews = []
		batch = self.env.interact_and_evaluate(self.actor, self.batch_size)

		for (question, no_rephrase_reward, rephrase, rephrase_reward, reward) in zip(*batch):
			# Track recent reward, action, and action log probability
			batch_seq.append(question)
			batch_rews.append(reward)
			batch_acts.append(rephrase)

			#track the same information but question/reward pairwise for value training
			value_batch_seq.append(question)
			value_batch_rews.append(no_rephrase_reward)
			value_batch_seq.append(rephrase)
			value_batch_rews.append(rephrase_reward)

			# Track episodic lengths and rewards
			#ep_rews.append(ep_rews)

		# Tokenize the sequences
		batch_seq = self.tokenizer(batch_seq, return_tensors='pt', padding=True).input_ids
		batch_acts = self.tokenizer(batch_acts, return_tensors='pt', padding=True).input_ids
		batch_rews = torch.tensor(batch_rews, dtype=torch.float)
		value_batch_seq = self.tokenizer(value_batch_seq, return_tensors='pt', padding=True).input_ids
		value_batch_rews = torch.tensor(value_batch_rews, dtype=torch.float)

		#compute log probabilities of (output | input)
		batch_log_probs = []
		chunk_size = 8
		for i in range(0, len(batch_seq), chunk_size):
			batch_seq_chunk = batch_seq[i:i+chunk_size]
			batch_acts_chunk = batch_acts[i:i+chunk_size]
			batch_log_probs_chunk = self.actor.module.get_log_probs(batch_seq_chunk, batch_acts_chunk, requires_grad=False)
			batch_log_probs.append(batch_log_probs_chunk.detach().cpu())
		batch_log_probs = torch.cat(batch_log_probs)


		print(f"Rollouts collected: {self.batch_size} in {time.time() - start_time} seconds")
		torch.cuda.empty_cache()
		self.actor.module.print_memory("After rollout")
		# Create a rollout object and return it
		rollout = type('Rollout', (object,), {
			'batch_seq': batch_seq,
			'batch_acts': batch_acts,
			'batch_log_probs': batch_log_probs,
			'batch_rews': batch_rews,
			'batch_lens': [self.batch_size]
		})()
		value_rollout = type('ValueRollout', (object,), {
			'batch_seq': value_batch_seq,
			'batch_rews': value_batch_rews
		})()

		return rollout, value_rollout

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		default_hyperparameters = {
        'lr': 1e-6,
        'gamma': 0.99,
        'epsilon': 0.2,
        'batch_size': 64,
        'epochs': 3,
        'value_lr': 1e-3,
        'episodes': 1000,
        'clip': 0.2,
		'seed': None,
		'sim_coef': 0.1
    	}
		# Change any default values to custom values for specified hyperparameters
		for param, val in default_hyperparameters.items():
			setattr(self, param, val)

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			setattr(self, param, val)

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t = self.logger['t']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []

def getPolicyClass(type):
	classTypes = {
		'Transformer': TransformerPolicy
	}
	return classTypes[type]

def getTokenizerClass(type):
	classTypes = {
		'Transformer': AutoTokenizer,
		'SACTransformer': AutoTokenizer
	}
	return classTypes[type]