"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from models import *
from env import SingleStepEnv, MultiStepEnv
from transformers import AutoTokenizer
from helpers import *
import random
#import deepspeed
import os
import json

class Agent:
	"""
		This is the PPO agent class we will use as our model in main.py
	"""
	def __init__(self, policy_type, data_path, hf_checkpoint='meta-llama/Llama-3.2-1B-Instruct', auth_token=None, hyperparameters=None):
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
		# Load hyperparameters from config.json if it exists
		model_name = hyperparameters['model_name']
		if os.path.exists(f'./models/{model_name}/config.json'):
			with open(f'./models/{model_name}/config.json', 'r') as f:
				hyperparameters = json.load(f)
				self._init_hyperparameters(hyperparameters)

		policy_class = getPolicyClass(policy_type)
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		tokenizer_class = getTokenizerClass(policy_type)
		special_tokens_func = loadSpecialTokens(hf_checkpoint)
		self.env = MultiStepEnv(data_path, hf_checkpoint, max_steps=1)
		self.instructions = self.env.instructions
		self.chat_template = getChatTemplate(hf_checkpoint)
		self.env.chat_template = self.chat_template
		# Initialize hyperparameters for training
		self._init_hyperparameters(hyperparameters)

		# Initialize tokenizer
		self.tokenizer = tokenizer_class.from_pretrained(hf_checkpoint, token=auth_token)
		special_tokens_func(self.tokenizer)

		# Initialize actor and critic networks
		self.actor = policy_class(hf_checkpoint, self.tokenizer, token=auth_token)
		
		# Check if there's a detuned version of this model
		model_prefix = model_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")
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

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'critic_losses': []     # losses of critic network in current iteration
		}

		#other hyperparams
		self.save_freq = 10
		if self.test_model is not None:
			assert os.path.exists(f'./models/{self.test_model}/'), "Model folder does not exist"
		if os.path.exists(f'./models/{self.model_name}/'):
			if os.path.exists(f'./models/{self.model_name}/actor.pt'):
				print(f"Loading saved model {self.model_name}")
				self._load_from_experiment(self.model_name)
		else:
			os.makedirs(f'./models/{self.model_name}/')
			# Save hyperparameters to config.json
			config_path = f'./models/{self.model_name}/config.json'
			with open(config_path, 'w') as config_file:
				json.dump(hyperparameters, config_file, indent=4)

	def _compute_advantages(self, rewards, values):
		advantages = torch.zeros_like(rewards)
		returns = torch.zeros_like(rewards)
		running_return = 0
		running_advantage = 0
		
		# Compute returns and advantages going backward
		for t in reversed(range(len(rewards))):
			new_return = rewards[t] + self.gamma * running_return
			running_return = new_return.clone()

			new_advantage = rewards[t] + self.gamma * running_advantage - values[t]
			running_advantage = new_advantage.clone()
			
			returns[t] = running_return
			advantages[t] = running_advantage
			
		# Normalize advantages
		advantages_normalized = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).clone()
		advantages = advantages_normalized
		return advantages, returns

	def _get_action(self, obs):
		# Get value prediction for current state
		value = self.critic(obs)
		
		# Get action from policy
		action, log_prob = self.actor.get_action(obs)
		
		return action, log_prob, value

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
		print("Starting training. Online-samples will be collected with probability ", self.online_eps)
		while i_so_far < self.episodes and t < self.max_timesteps:           

			if random.random() < self.online_eps:
				rollout, value_rollout = self.rollout_from_simulator(i_so_far)
			else:                                                            
				rollout, value_rollout = self.rollout_from_data(i_so_far)   

			self.actor.train()
			self.critic.train()
			#self.actor.print_memory("Before training")
			assert self.batch_size >= self.minibatch_size, "Minibatch size must be smaller than batch size"
			# Calculate how many timesteps we collected this batch
			t += np.sum(rollout.batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t'] = t
			self.logger['i_so_far'] = i_so_far

			# Minibatch training
			for _ in range(self.epochs):
				# Shuffle the value rollout data
				perm = torch.randperm(value_rollout.batch_seq.size(0))
				value_batch_seq_shuffled = value_rollout.batch_seq[perm]
				value_batch_rews_shuffled = value_rollout.batch_rews[perm]
				for j in range(0, value_rollout.batch_seq.size(0), self.minibatch_size):
					minibatch_seq = value_batch_seq_shuffled[j:j + self.minibatch_size]
					minibatch_rews = value_batch_rews_shuffled[j:j + self.minibatch_size]

					# Value updates
					_, errors, critic_loss = self.value_step(minibatch_seq, minibatch_rews)
					# Calculate gradients and perform backward propagation for critic network
					self.critic.zero_grad()
					critic_loss.backward()
					self.critic_optim.step()
					self.logger['critic_losses'].append(critic_loss.detach().cpu())

			# Calculate advantage at k-th iteration
			val_inputs = self.tokenizer(self.tokenizer.batch_decode(rollout.batch_seq), return_tensors='pt', padding=True)
			input_ids = val_inputs.input_ids.to('cuda')
			attention_mask = val_inputs.attention_mask.to('cuda')
			values, _ = self.critic(input_ids, attention_mask)
			values_sq = values.squeeze().clone()
			values_sum = values_sq.sum(dim=1).clone()
			Advantage, returns = self._compute_advantages(rollout.batch_rews.to('cuda'), values_sum)
			#normalize advantages
			Advantage_norm = (Advantage - Advantage.mean()) / (Advantage.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.epochs):
				# Shuffle the rollout data
				perm = torch.randperm(rollout.batch_seq.size(0))
				batch_seq_shuffled = rollout.batch_seq[perm]
				batch_acts_shuffled = rollout.batch_acts[perm]
				batch_log_probs_shuffled = rollout.batch_log_probs[perm]
				batch_rews_shuffled = rollout.batch_rews[perm]
				A_k_shuffled = Advantage_norm[perm]

				# Minibatch training
				for j in range(0, rollout.batch_seq.size(0), self.minibatch_size//2):
					minibatch_seq = batch_seq_shuffled[j:j + self.minibatch_size//2]
					minibatch_acts = batch_acts_shuffled[j:j + self.minibatch_size//2]
					minibatch_log_probs = batch_log_probs_shuffled[j:j + self.minibatch_size//2]
					minibatch_rews = batch_rews_shuffled[j:j + self.minibatch_size//2]
					A_k = A_k_shuffled[j:j + self.minibatch_size//2]

					# Calculate V_phi and pi_theta(a_t | s_t)
					curr_log_probs = self.actor.get_log_probs(minibatch_seq, minibatch_acts)
					# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
					ratios = torch.exp(curr_log_probs - minibatch_log_probs.to(curr_log_probs.device))
					# Calculate surrogate losses.

					surr1 = ratios * A_k
					surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

					# Calculate the semantic similarity between question and rephrase
					sim_score = self.actor.get_embed_similarity(minibatch_seq, minibatch_acts, torch.ones_like(minibatch_seq), torch.ones_like(minibatch_acts))

					# Calculate language modelling loss
					actor_loss = (-torch.min(surr1, surr2)).mean() + sim_score.mean().to(ratios.device) * self.sim_coef
					# Calculate gradients and perform backward propagation for actor network
					self.actor.zero_grad()
					
					actor_loss.backward(retain_graph=True)
					self.actor_optim.step()

					# Log actor loss
					self.logger['actor_losses'].append(actor_loss.detach().cpu())
			torch.cuda.empty_cache()
			# Print a summary of our training so far
			self._log_summary()
			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				self._save()

	def value_step(self, input, reward):
		'''
		Compute the value of a given input sequence and propagate loss
		'''
		#compute bellman reward, once we have multiple steps
		input = self.tokenizer.batch_decode(input, skip_special_tokens=True)
		tokens = self.tokenizer(input, return_tensors='pt', padding=True)
		input_ids = tokens.input_ids.to('cuda')
		attention_mask = tokens.attention_mask.to('cuda')
		value, attention = self.actor.value(input_ids, attention_mask)
		value_squeezed = value.squeeze().clone()
		value_norm = (value_squeezed - value_squeezed.mean()) / (value_squeezed.std() + 1e-10)
		value_sum = value_norm.sum(dim=1).clone()
		#right now, the value is for each token, but we want it to be for the whole sequence
		#to do this we should multiply the value by the attention mask and sum over the sequence
		#such that the sum of the value is the probability that the sequence is adversarial
		#accordingly, the value of a token is the probability that the token is adversarial
		errors = value_sum - reward.to('cuda')
		loss = nn.MSELoss()(value_sum, reward.to('cuda'))
		
		return value_sum, errors, loss
	
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
		with self.actor.eval():
			if torch.is_tensor(batch_acts):
				batch_acts = self.tokenizer.batch_decode(batch_acts, skip_special_tokens=False)
			log_probs = self.actor.get_log_probs(batch_seq, batch_acts)
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
		with self.actor.eval():
			if torch.is_tensor(batch_seq):
				batch_seq = self.tokenizer.batch_decode(batch_seq, skip_special_tokens=False)

			output_tokens = self.tokenizer(batch_seq, return_tensors='pt', padding=True)
			value, attn = self.actor.value(output_tokens.input_ids, output_tokens.attention_mask)
		return (value, attn) if return_attn else value
	
	def test(self, num_episodes=10):
		"""
		Test the agent in the given environment.

		Args:
			num_episodes (int): The number of episodes to run for testing.

		Returns:
			list: A list of rewards obtained in each episode.
		"""
		rewards = []
		print("Testing... iterations: ", num_episodes)
		questions_list = []
		rephrase_list = []
		response_list = []

		for episode in range(num_episodes):
			prompt = self.env.reset()
			episode_reward = 0
			done = False

			while not done:
				# Take action in the environment
				question, no_rephrase_reward, action, rephrase_reward, reward, done, response = self.env.evaluate(self.actor)
				questions_list.extend(question)
				rephrase_list.extend(action)
				response_list.extend(response)

				# Accumulate reward
				episode_reward += reward

				prompt = question
				print(f"Question: {question}")
				print(f"Rephrase: {action}")
				print(f"Reward: {rephrase_reward}")


			# Store the episode reward
			rewards.append(episode_reward)

			print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
		test_chat_with_guard(questions_list, rephrase_list, response_list)
		return rewards

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

		self.actor.eval()
		self.critic.eval()

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
		batch_log_probs = self.actor.get_log_probs(batch_seq, batch_acts).detach()

		print(f"Rollouts collected: {t} in {time.time() - start_time} seconds")
		#self.actor.print_memory("After rollout")
		
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
		batch_dones = []

		value_batch_seq = []
		value_batch_rews = []
		value_batch_dones = []

		self.actor.eval()
		self.critic.eval()

		start_time = time.time()

		ep_rews = []
		batch = self.env.interact_and_evaluate(self.actor, self.batch_size)

		for (question, no_rephrase_reward, rephrase, rephrase_reward, reward, done) in zip(*batch):
			# Track recent reward, action, and action log probability
			batch_seq.append(question)
			batch_rews.append(reward)
			batch_acts.append(rephrase)
			batch_dones.append(done)

			#track the same information but question/reward pairwise for value training
			value_batch_seq.append(question)
			value_batch_rews.append(no_rephrase_reward)
			value_batch_seq.append(rephrase)
			value_batch_rews.append(rephrase_reward)
			for _ in range(2): #once for each question
				value_batch_dones.append(done)

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
		#adjust chunk size to avoid memory errors. loop is do-while format to ensure at least one iteration
		chunk_size = 8
		i = 0
		while True:
			batch_seq_chunk = batch_seq[i:i+chunk_size]
			batch_acts_chunk = batch_acts[i:i+chunk_size]
			batch_log_probs_chunk = self.actor.get_log_probs(batch_seq_chunk, batch_acts_chunk, requires_grad=False)
			batch_log_probs.append(batch_log_probs_chunk.detach().cpu())
			i += chunk_size
			if i >= len(batch_seq):
				break
		batch_log_probs = torch.cat(batch_log_probs)

		print(f"Rollouts collected: {self.batch_size} in {time.time() - start_time} seconds")
		torch.cuda.empty_cache()
		#self.actor.print_memory("After rollout")
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

		#print one sample from the rollout
		#sample_q = self.tokenizer.decode(batch_seq[0], skip_special_tokens=True)
		#sample_a = self.tokenizer.decode(batch_acts[0], skip_special_tokens=True)
		#message = create_messages(sample_a, None)
		#response = LLM_get_response(self.env.client, message).choices[0].message.content
		#print(f"Sample: \nQuestion: {sample_q}\nRephrase: {sample_a}\nReward: {batch_rews[0]}\nResponse: {response}")

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
		avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])

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
		print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
		print(f"Timesteps So Far: {t}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)
		#self.actor.print_memory("Memory usage after iteration #"+str(i_so_far))

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['critic_losses'] = []

	def _save(self):
		torch.save(self.actor.final.state_dict(), f'./models/{self.model_name}/actor.pt')
		torch.save(self.actor.value_head.state_dict(), f'./models/{self.model_name}/critic.pt')
		torch.save(self.actor_optim.state_dict(), f'./models/{self.model_name}/actor_optim.pt')
		torch.save(self.critic_optim.state_dict(), f'./models/{self.model_name}/critic_optim.pt')
		
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
			torch.save(self.actor.final.state_dict(), f'{detuned_path}/actor.pt')
			
			# Save a flag file indicating this is a detuned model
			with open(f'{detuned_path}/flags', 'w') as file:
				file.write("detuned")
			
			print(f"Saved detuned model to {detuned_path}")
		
		# Log the date and time of the last save
		with open(f'./models/{self.model_name}/save_log.txt', 'a') as f:
			f.write(f"Model saved on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
		print(f"Model saved on: {time.strftime('%Y-%m-%d %H:%M:%S')} to ./models/{self.model_name}/", flush=True)

	def _load_from_experiment(self, experiment_name):
		"""
			Load a model from a previous experiment.

			Parameters:
				experiment_name - the name of the experiment to load from

			Return:
				None
		"""
		# Check if there's a detuned version of this model
		model_prefix = experiment_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")
		detuned_path = f'./models/{model_prefix}-detuned'
		
		# If detuned model exists and has actor.pt, load from there instead
		if os.path.exists(detuned_path) and os.path.exists(f'{detuned_path}/actor.pt'):
			print(f"Loading detuned model weights from {detuned_path}")
			self.actor.final.load_state_dict(torch.load(f'{detuned_path}/actor.pt'))
		else:
			# Load from the original experiment path
			self.actor.final.load_state_dict(torch.load(f'./models/{experiment_name}/actor.pt'))
			
		self.actor.value_head.load_state_dict(torch.load(f'./models/{experiment_name}/critic.pt'))
		if os.path.exists(f'./models/{experiment_name}/actor_optim.pt'):
			self.actor_optim.load_state_dict(torch.load(f'./models/{experiment_name}/actor_optim.pt'))
		if os.path.exists(f'./models/{experiment_name}/critic_optim.pt'):
			self.critic_optim.load_state_dict(torch.load(f'./models/{experiment_name}/critic_optim.pt'))
		if os.path.exists(f'./models/{experiment_name}/config.json'):
			with open(f'./models/{experiment_name}/config.json', 'r') as f:
				hyperparameters = json.load(f)
				self._init_hyperparameters(hyperparameters)

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

def getChatTemplate(checkpoint_string):
	if 'llama' in checkpoint_string.lower():
		return llama_chat_template
	elif 'mistral' in checkpoint_string.lower() or 'ministral' in checkpoint_string.lower() or 'collective'	in checkpoint_string.lower():
		return mistral_chat_template
	
def loadSpecialTokens(checkpoint_string):
	if 'llama' in checkpoint_string.lower():
		return add_llama_special_tokens
	elif 'mistral' in checkpoint_string.lower() or 'ministral' in checkpoint_string.lower() or 'collective'	in checkpoint_string.lower():
		return add_mistral_special_tokens