"""
	The original PPO code was taken from the Spinning Up repository by OpenAI.
	It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

from curses.ascii import alt
import gc
import time

import numpy as np
import time
import torch
import torch.nn as nn
from models import *
from cpu_models import *
from env import MultiStepEnv
from transformers import AutoTokenizer 
from helpers import *
import random
#import deepspeed
import os
import json
from base import BaseModelSingleton
from peft import PeftConfig, get_peft_model, LoraConfig, TaskType
import os

class Agent:
	"""
		This is the PPO agent class we will use as our model in main.py
	"""
	def __init__(self, policy_type, data_path, hf_checkpoint='meta-llama/Llama-3.2-1B-Instruct', auth_token=None, hyperparameters=None, use_finetuned=False, base_adapter_path=None, merge_adapters=False, use_lora=False, adapter_type=None, agent_model_singleton=None, critic_model_singleton=None, target_model_singleton=None, target_adapter_path=None, custom_harm_critic_path=None):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_type - the policy class to use for our actor/critic networks.
				tokenizer_class - the tokenizer class to use for our language models.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
				use_finetuned - whether to use a fine-tuned model with LoRA adapters for the actor
				adapter_path - path to the base model LoRA adapter weights
				merge_adapters - whether to merge LoRA adapters with the base model
				use_lora - whether to use LoRA for fine-tuning
				adapter_type - specific adapter type to use (e.g. 'subtle', 'moderate', 'aggressive', 'neutral')
				agent_model_singleton - model singleton for the agent policy
				critic_model_singleton - model singleton for the value critics
				target_model_singleton - model singleton for the environment target (deprecated)
				target_adapter_path - path to LoRA adapter weights for the target model
				custom_harm_critic_path - path to custom harm critic checkpoint

			Returns:
				None
		"""
		model_name = hyperparameters['model_name']
		# Initialize hyperparameters early to get keep_base_on_cpu setting
		if os.path.exists(f'./models/{model_name}/config.json') and not hyperparameters['override']:
			with open(f'./models/{model_name}/config.json', 'r') as f:
				config = json.load(f)
				config.update(hyperparameters)
				hyperparameters = config
		
		self._init_hyperparameters(hyperparameters)
		
		# Use separate model singletons for different components
		self.agent_model_singleton = agent_model_singleton
		self.critic_model_singleton = critic_model_singleton
		self.target_model_singleton = target_model_singleton  # Deprecated, kept for backwards compatibility
		self.target_adapter_path = target_adapter_path  # Store the target adapter path
		
		# Set training mode based on hyperparameters
		self.train = hyperparameters.get('train', False)
		print(f"Training..." if self.train else "Testing...")
		
		if agent_model_singleton:
			print(f"Agent model device: {agent_model_singleton.base_model.device}")
		if critic_model_singleton:
			print(f"Critic model device: {critic_model_singleton.base_model.device}")
		
		print(f"Keep base on CPU: {self.keep_base_on_cpu}")
		# Initialize checkpoint name
		self.checkpoint = hf_checkpoint
		
		# Set up LoRA configuration
		self.use_lora = use_lora
		
		# Handle adapter type mapping
		if adapter_type is not None:
			print(f"Using adapter type: {adapter_type}")
			# Map from text names to adapter names used in the code
			# Initialize with only the specified adapter
			self.lora_adapters = {
				adapter_type: None
			}
		else:
			# Default to all four adapters if no specific one is specified
			self.lora_adapters = {
				'lm_head_rephrase': None,
				'attention_rephrase': None,
				'lm_head_followup': None,
				'attention_followup': None,
				'attention_holistic': None
			}
		
		# Initialize adapter models and optimizers
		self.lora_optimizers = {}
		self.current_adapter = None
		
		# Configure LoRA parameters
		if use_lora:
			self.lora_configs = {
				'lm_head_rephrase': {
					'r': self.lora_r,
					'lora_alpha': self.lora_alpha,
					'lora_dropout': self.lora_dropout,
					'target_modules': ['lm_head'],
					'bias': 'none'
				},
				'attention_rephrase': {
					'r': self.lora_r,
					'lora_alpha': self.lora_alpha,
					'lora_dropout': self.lora_dropout,
					'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
					'bias': 'none'
				},
				'lm_head_followup': {
					'r': self.lora_r,
					'lora_alpha': self.lora_alpha,
					'lora_dropout': self.lora_dropout,
					'target_modules': ['lm_head'],
					'bias': 'none'
				},
				'attention_followup': {
					'r': self.lora_r,
					'lora_alpha': self.lora_alpha,
					'lora_dropout': self.lora_dropout,
					'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
					'bias': 'none'
				},
				'attention_holistic': {
					'r': self.lora_r,
					'lora_alpha': self.lora_alpha,
					'lora_dropout': self.lora_dropout,
					'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
					'bias': 'none'
				}
			}
			# Create adapter directories if they don't exist
			for adapter_name in self.lora_adapters.keys():
				adapter_dir = f'./models/{model_name}/adapters/{adapter_name}'
				os.makedirs(adapter_dir, exist_ok=True)
				self.lora_adapters[adapter_name] = adapter_dir
		
		policy_class = getPolicyClass(policy_type)
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		special_tokens_func = loadSpecialTokens(hf_checkpoint)
		print(f"max ep len: {self.max_ep_len}")
		self.env = MultiStepEnv(
			data_path, 
			auth_token, 
			max_steps=self.max_ep_len, 
			batch_size=self.minibatch_size,
			target_model_name=self.target_model,
			target_model_singleton=None,  # Deprecated parameter
			lora_adapter_path=getattr(self, 'target_adapter_path', None),
			use_custom_reward=self.use_custom_reward,
			reward_model_path=self.custom_env_reward
		)
		self.instructions = self.env.instructions

		self.tokenizer = self.agent_model_singleton.tokenizer
		special_tokens_func(self.tokenizer)

		self.tokenizer.pad_token = '<|finetune_right_pad_id|>'
		self.tokenizer.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
		special_tokens_dict = self.tokenizer.special_tokens_map
		print("Special tokens:", special_tokens_dict)
		print("Total vocab size:", len(self.tokenizer))
			
		# Determine template model based on checkpoint name if chat_template doesn't exist
		if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None or self.use_finetuned:
			if 'vicuna' in hf_checkpoint.lower():
				name = 'vicuna'
			elif 'llama' in hf_checkpoint.lower():
				name = 'llama'
			else:
				name = 'llama'
				print("Chat template not found, using default template (llama 3.2)")
			path = os.path.join(os.path.dirname(__file__), "models", "chat-templates", f"{name}.jinja")
			with open(path, "r") as f:
				template = f.read()
				self.tokenizer.chat_template = template
			
			print(f"Chat template set from {path}")

		# Initialize actor with fine-tuned model if specified
		if use_finetuned:
			print(f"Loading fine-tuned model with LoRA adapters for the actor")
			
			# Determine adapter path if not provided
			if base_adapter_path is None:
				model_prefix = hf_checkpoint.split('/')[-1].split('-')[0]
				base_adapter_path = f"./models/fine-tuned/{model_prefix}-lora"
			
			# Ensure adapter_path is an absolute path if it's not already
			if not os.path.isabs(base_adapter_path):
				base_adapter_path = os.path.abspath(base_adapter_path)
			
			print(f"Using base model: {hf_checkpoint}")
			print(f"Using base adapter path: {base_adapter_path}")
			
			# Verify that the adapter path contains the required config file
			base_adapter_config_path = os.path.join(base_adapter_path, "adapter_config.json")
			base_adapter_weights_path = os.path.join(base_adapter_path, "adapter_model.safetensors")
			
			if not os.path.exists(base_adapter_config_path) or not os.path.exists(base_adapter_weights_path):
				print(f"Warning: Could not find adapter_config.json at {base_adapter_config_path} or adapter_model.safetensors at {base_adapter_weights_path}")
				print("Checking if base_adapter_path is a directory containing adapters...")
				
				# Check if base_adapter_path is a directory containing different adapter folders
				if os.path.isdir(base_adapter_path):
					# Look for subdirectories that might contain adapter configs
					possible_adapter_dirs = [d for d in os.listdir(base_adapter_path) 
										if os.path.isdir(os.path.join(base_adapter_path, d))]
					
					if possible_adapter_dirs:
						first_adapter = possible_adapter_dirs[0]
						new_base_path = os.path.join(base_adapter_path, first_adapter)
						if os.path.exists(os.path.join(new_base_path, "adapter_config.json")):
							base_adapter_path = new_base_path
							print(f"Found adapter configuration in subdirectory: {base_adapter_path}")
						else:
							print(f"No adapter_config.json found in subdirectories of {base_adapter_path}")
							print("Will proceed with standard model initialization without LoRA adapters.")
							use_finetuned = False
				else:
					print(f"Base adapter path {base_adapter_path} is not a valid directory.")
					print("Will proceed with standard model initialization without LoRA adapters.")
					use_finetuned = False
			
			if use_finetuned:
				try:
					# Load the base adapter into the agent model singleton if not already loaded
					if not hasattr(self.agent_model_singleton, 'peft_model') or self.agent_model_singleton.peft_model is None:
						# Load the LoRA configuration from the base adapter
						print(f"Loading base LoRA adapter from: {base_adapter_path}")
						try:
							config = PeftConfig.from_pretrained(base_adapter_path)
							
							# Add the base adapter to the agent model singleton
							self.agent_model_singleton.add_adapter("base_model", config, base_adapter_path)
							print(f"Successfully added base model adapter to agent model singleton")
							
							# Now merge the base adapter with the base model to have unified base model
							# This is more efficient than keeping the adapter separate and ensures that
							# all agent-specific adapters are applied on top of the already fine-tuned model
							if merge_adapters:
								print("Merging base adapter into base model...")
								success = self.agent_model_singleton.merge_base_adapter()
								if success:
									print("Successfully merged base adapter with base model")
								else:
									print("Failed to merge base adapter, continuing with adapter approach")
							else:
								# Just set the base adapter as active if not merging
								self.agent_model_singleton.switch_adapter("base_model")
								print("Base adapter loaded and activated (not merged)")
						except Exception as e:
							print(f"Error loading base adapter: {e}")
							print("Will proceed with standard model initialization without LoRA adapters.")
							
					else:
						print("Agent model with PEFT adapter is already initialized")
						
						# Check if we should merge the already loaded adapter
						if merge_adapters and "base_model" in self.agent_model_singleton.available_adapters:
							print("Merging already loaded base adapter into base model...")
							success = self.agent_model_singleton.merge_base_adapter()
							if success:
								print("Successfully merged base adapter with base model")
							else:
								print("Failed to merge base adapter, continuing with adapter approach")
				
				except Exception as e:
					print(f"Error during base adapter initialization: {e}")
					print("Will proceed with standard model initialization.")
					use_finetuned = False
			
		# Initialize the actor model using the agent model singleton
		actor_model = self.agent_model_singleton.base_model
		
		# Create the actor policy
		self.actor = policy_class(
			hf_checkpoint,
			self.tokenizer,
			model=actor_model,
			token=auth_token,
			use_lora=use_lora,
			adapter_name='base_model'
		)
		
		print(f"Actor model created with {'PEFT adapter' if use_finetuned else 'base model'}")
		self.actor.print_memory('Actor model created')
		
		# Initialize harm critic
		if custom_harm_critic_path:
			self.load_custom_harm_critic(custom_harm_critic_path)
		else:
			# Original harm critic initialization
			self.harm_critic = ValueTransformer(
				hf_checkpoint=self.model_name,
				tokenizer=self.tokenizer,
				token_level=True)
		
		# Initialize help critic (sequence-level)
		self.help_critic = ValueTransformer(
			hf_checkpoint="meta-llama/Llama-3.2-1B-Instruct",  # Always use Llama-3.2-1B
			tokenizer=self.tokenizer, 
			model=self.critic_model_singleton.base_model,  # Use critic model singleton
			token=auth_token, 
			token_level=False,
			attention_update=False)
		
		# Initialize optimizer for critic
		self.harm_critic_optim = torch.optim.Adam(self.harm_critic.parameters(), lr=self.value_lr)
		self.help_critic_optim = torch.optim.Adam(self.help_critic.parameters(), lr=self.value_lr)
	
		# Initialize other variables
		self.rollout_states = []
		self.rollout_actions = []
		self.rollout_rewards = []
		self.rollout_action_help_rewards = []
		self.rollout_compliance_rewards = []
		self.max_seq_len = 2048
		
		self.manually_add_bos = not self.check_template_inserts_bos()
		
		if self.train:
			self.actor.train()
			if self.alg == "ppo":
				self.harm_critic.train()
				self.help_critic.train()
		else:
			self.actor.eval()
			if self.alg == "ppo":
				self.harm_critic.eval()
				self.help_critic.eval()
		
		self.logger = {}
		# Track loss values per iteration
		self.logger['actor_losses'] = []
		self.logger['harm_critic_losses'] = []
		self.logger['help_critic_losses'] = []
		self.logger['ce_losses'] = []
		self.logger['timesteps_cumulative'] = []
		self.logger['timesteps_per_batch'] = []
		self.logger['batch_lens'] = []
		self.logger['batch_help_rews'] = []
		self.logger['batch_harm_rews'] = []
		self.logger['batch_advs'] = []
		self.logger['batch_rews'] = []
		self.logger['batch_comply_rews'] = []
		self.logger['batch_action_help_rews'] = []
		self.logger['batch_similarity'] = []
		self.logger['delta_t'] = time.time_ns()
		self.logger['t'] = 0
		self.logger['i_so_far'] = 0
		self.logger['episodes'] = 0
		# If we're using LoRA, initialize the adapter models
		if use_lora:
			self.initialize_adapter_models()

	def check_template_inserts_bos(self):
		# Test chat template BOS token behavior
		messages = [{"role": "user", "content": "test"}]
		formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors='pt')
		return not formatted.startswith(self.tokenizer.bos_token)

	def _compute_advantages(self, rewards, values, next_values=None, gamma=0.99, gae_lambda=0.95, dones=None, token_level=False, td_error=None):
		"""
		Compute advantages using Generalized Advantage Estimation (GAE).
		
		Args:
			rewards: Rewards for each timestep [T]
			values: Value predictions for each timestep [T]
			next_values: Value predictions for next timesteps [T] 
			gamma: Discount factor
			gae_lambda: GAE lambda parameter
			dones: Binary indicators for episode termination [T]
			token_level: Whether to compute token-level advantages
			td_error: Pre-calculated TD error to use instead of calculating it inside this function
			
		Returns:
			advantages: Advantage estimates for each timestep [T]
			returns: Returns for each timestep [T]
		"""
		# Ensure all inputs have proper dimensions
		if rewards.dim() == 1:
			rewards = rewards.unsqueeze(-1)  # [T, 1]
		if values.dim() == 1:
			values = values.unsqueeze(-1)  # [T, 1]
		if next_values is not None and next_values.dim() == 1:
			next_values = next_values.unsqueeze(-1)  # [T, 1]
		if dones is not None and dones.dim() == 1:
			dones = dones.unsqueeze(-1)  # [T, 1]
		if td_error is not None and td_error.dim() == 1:
			td_error = td_error.unsqueeze(-1)  # [T, 1]
			
		T = rewards.size(0)
		device = rewards.device
		
		# Initialize return and advantage arrays
		returns = torch.zeros_like(values) if token_level else torch.zeros_like(rewards)
		advantages = torch.zeros_like(values) if token_level else torch.zeros_like(rewards)
		
		# Initialize running values
		if token_level:
			running_return = torch.zeros(1, values.size(1), device=device) if next_values is not None else torch.zeros_like(values[0])
			running_advantage = torch.zeros(1, values.size(1), device=device) if next_values is not None else torch.zeros_like(values[0])
		else:
			running_return = torch.zeros(1, rewards.size(1), device=device) if next_values is not None else torch.zeros_like(rewards[0])
			running_advantage = torch.zeros(1, rewards.size(1), device=device) if next_values is not None else torch.zeros_like(rewards[0])
		
		# Iterate backwards through rewards
		for t in reversed(range(T)):
			# If pre-calculated TD error is provided, use it directly
			if td_error is not None:
				td_err = td_error[t]
			else:
				if next_values is not None:
					if dones is not None:
						# Create a mask where 1 = use next_value, 0 = end of episode (use 0)
						mask = 1.0 - dones[t]
						next_value = next_values[t] * mask
					else:
						next_value = next_values[t]
						
					# TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
					td_err = rewards[t] + gamma * next_value.sum(dim=-1) - values[t]
				else:
					if dones is not None and dones[t].item() > 0:
						adjusted_running_advantage = torch.zeros_like(running_advantage)
					else:
						adjusted_running_advantage = running_advantage
						
					td_err = rewards[t] + gamma * adjusted_running_advantage - values[t]
			
			# Compute advantage using GAE
			if dones is not None and t < T-1 and dones[t].item() > 0:
				# If this is the end of an episode, break the advantage calculation
				advantage = td_err  # No bootstrapping at episode boundaries
			else:
				advantage = td_err + gamma * gae_lambda * running_advantage
				
			running_advantage = advantage.clone()
			
			# Compute returns
			if dones is not None and t < T-1 and dones[t].item() > 0:
				# If this is the end of an episode, break the return calculation
				running_return = torch.zeros_like(running_return)
				returns[t] = rewards[t]
			else:
				returns[t] = rewards[t] + gamma * running_return
				running_return = returns[t].clone()
			
			advantages[t] = advantage
		
		# Normalize advantages - we use a more robust normalization to handle outliers
		if T > 1:
			if token_level:
				# For token-level advantages, normalize across tokens and examples
				advantages_flat = advantages.view(-1)
				non_zero_mask = advantages_flat != 0  # Avoid normalizing zero advantages
				
				if non_zero_mask.sum() > 0:  # Only normalize if we have non-zero advantages
					advantages_mean = advantages_flat[non_zero_mask].mean()
					advantages_std = advantages_flat[non_zero_mask].std()
					
					max_magnitude = 5.0 
					clipped_advantages = torch.clamp(advantages, -max_magnitude, max_magnitude)
					
					normalized_advantages = torch.zeros_like(advantages)
					normalized_advantages = (clipped_advantages - advantages_mean) / (advantages_std + 1e-8)
					advantages = normalized_advantages
			else:
				advantages_mean = advantages.mean()
				advantages_std = advantages.std()
				max_magnitude = 5.0
				clipped_advantages = torch.clamp(advantages, -max_magnitude, max_magnitude)
				advantages = (clipped_advantages - advantages_mean) / (advantages_std + 1e-8)
		
		# Squeeze if we expanded earlier
		if rewards.size(1) == 1:
			advantages = advantages.squeeze(-1)
			returns = returns.squeeze(-1)
			
		return advantages, returns

	def learn(self):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.
			The method interleaves value and policy updates for each minibatch to leverage
			error information for better advantage estimation.

			Return:
				None
		"""
		t = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		with torch.set_grad_enabled(True):
			while i_so_far < self.episodes:
				assert self.batch_size >= self.minibatch_size, "Minibatch size must be smaller than batch size"                             
				self.to_cuda()
				self.actor.train()
				if self.alg == 'ppo':
					self.harm_critic.train()
					self.help_critic.train()

				if self.use_lora:
					# Train each adapter with its own rollout
					for adapter_name in ['attention_holistic', 'lm_head_rephrase', 'attention_rephrase', 'lm_head_followup', 'attention_followup']:
						if adapter_name in self.lora_adapters:
							print(f"\n--- Training {adapter_name} ---")
							self.switch_lora_adapter(adapter_name)
							if self.attack_type == 'breakdown':
								data = self.breakdown_rollout(self.batch_size)
							else:
								data = self.rollout(self.batch_size)
							if adapter_name in ['lm_head_rephrase', 'attention_rephrase']:
								data = self._extract_first_stage_data(data)
							elif adapter_name in ['lm_head_followup', 'attention_followup']:
								data = self._extract_followup_data(data)
							self._train_on_batch(data)
				else:
					# Use the attack type to determine which rollout to perform
					if self.attack_type == 'breakdown':
						batch_data = self.breakdown_rollout(self.batch_size)
					else:
						batch_data = self.rollout(self.batch_size)
						
					# Separate data for first-stage (rephrasing) and follow-up
					first_stage_data = self._extract_first_stage_data(batch_data)
					followup_data = self._extract_followup_data(batch_data)
					
					# Train on both stages without adapters
					self._train_on_batch(first_stage_data)
					
					# Check if followup_data has any elements
					has_followup_data = hasattr(followup_data, 'batch_state') and len(getattr(followup_data, 'batch_state', [])) > 0
					
					if has_followup_data:
						self._train_on_batch(followup_data)
				
				# Increment counters
				t += self.batch_size
				i_so_far += 1
				self.eps = max(self.eps**2, 0.01)
				self.logger['t'] = t
				self.logger['i_so_far'] = i_so_far
				
				# Log summary
				if i_so_far % self.save_freq == 0:
					self._save()
					
				# Move models to CPU to save memory
				self.to_cpu()
				
		# Save the final model
		self._save()
		
	def _extract_first_stage_data(self, batch_data):
		"""
		Extract data for the first stage (rephrasing) from the batch data.
		
		Args:
			batch_data: Rollout object containing batch data
			
		Returns:
			Dictionary containing first stage data
		"""
		# In the first stage, we only care about the first step in each episode
		first_stage_indices = []

		batch_state = batch_data.batch_state
		
		# Find indices of first steps in each episode
		for i, state in enumerate(batch_state):
			if len(state) == 1:  # First step in an episode
				first_stage_indices.append(i)
				

		# Create a new object with the same structure as batch_data
		first_stage_data = type('Rollout', (object,), {})()
		
		# Copy attributes from batch_data to first_stage_data
		for attr_name in dir(batch_data):
			if not callable(getattr(batch_data, attr_name)) and not attr_name.startswith('__'):
				attr_value = getattr(batch_data, attr_name)
				if isinstance(attr_value, list) or isinstance(attr_value, torch.Tensor):
					setattr(first_stage_data, attr_name, torch.tensor([attr_value[i] for i in first_stage_indices]))
				else:
					setattr(first_stage_data, attr_name, attr_value)
		return first_stage_data
		
	def _extract_followup_data(self, batch_data):
		"""
		Extract data for the follow-up stage from the batch data.
		
		Args:
			batch_data: Rollout object or dictionary containing batch data
			
		Returns:
			Dictionary containing follow-up stage data
		"""
		# In the follow-up stage, we care about all steps except the first in each episode
		followup_indices = []
		
		# Get the batch_state to find follow-up steps
		batch_lens = batch_data.batch_lens
		
		# Find indices of follow-up steps in each episode
		for i, length in enumerate(batch_lens):
			if length > 1:  # Follow-up step in an episode
				followup_indices.append(i)

		# Create a new object with the same structure as batch_data
		followup_data = type('Rollout', (object,), {})()
		if len(followup_indices) > 1:
			# Copy attributes from batch_data to followup_data
			for attr_name in dir(batch_data):
				if not callable(getattr(batch_data, attr_name)) and not attr_name.startswith('__'):
					attr_value = getattr(batch_data, attr_name)
					if isinstance(attr_value, list) or isinstance(attr_value, torch.Tensor):
						print(f"{attr_name}: {attr_value}")
						setattr(followup_data, attr_name, torch.tensor([attr_value[i] for i in followup_indices]))
					else:
						setattr(followup_data, attr_name, attr_value)
			
		return followup_data

	def value_step(self, response, action, harm_reward, help_reward, next_response, base_state, action_help_reward=None, gamma=0.99, return_action_value=True):
		"""
		Performs a value function update step for both harm and help critics
		
		Args:
			response: Tensor of states
			action: Tensor of actions 
			harm_reward: Harm rewards
			help_reward: Help rewards (compliance rewards)
			next_response: Next states
			base_state: Original state before model responses
			action_help_reward: Optional helpfulness rewards for actions
			gamma: Discount factor
			return_action_value: Whether to return the action value
			
		Returns:
			tuple of:
				Values: (harm_values, help_values)
				Errors: (harm_errors, help_errors)
				Losses: (harm_critic_loss, help_critic_loss)
				Token scores: (harm_token_scores, help_token_scores)
				Action values: (action_harm, action_help)
				Next state predictions: (next_harm_values, next_help_values)
		"""
		batch_size = response.shape[0]
		max_seq_len = response.shape[1]
		input_texts = self.tokenizer.batch_decode(response, skip_special_tokens=False)        
		
		# Get action texts if provided, otherwise extract from next_response
		if isinstance(action, torch.Tensor):
			action_texts = self.tokenizer.batch_decode(action, skip_special_tokens=False)
		else:
			action_texts = action
		assert len(input_texts) == len(action_texts), f"Batch size mismatch: {len(input_texts)} questions vs {len(action_texts)} responses"
		next_texts = self.tokenizer.batch_decode(next_response, skip_special_tokens=False)
		
		# Get state-action pairs for helpfulness scoring
		state_action_pairs = [q + " " + a for q, a in zip(input_texts, action_texts)]
		action_help = action_help_reward.to('cuda') if action_help_reward is not None else self.env.score_helpfulness(state_action_pairs).to('cuda')
		
		# Get harmfulness scores
		target_harm = harm_reward.to('cuda')  # From environment
		base_state_texts = self.tokenizer.batch_decode(base_state, skip_special_tokens=False)
		action_harm = self.env.score_safety(action_texts, questions=base_state_texts, reference_state=base_state_texts).to('cuda')
		
		#######################
		# PART 1: HARM VALUES #
		#######################
		
		# Move harm critic to CUDA
		self.harm_critic.to('cuda')
		
		# Initialize tensors to store results for the batch
		harm_values = torch.zeros((batch_size, max_seq_len), device='cuda')
		harm_errors = torch.zeros((batch_size, max_seq_len), device='cuda')
		harm_token_scores_batch = torch.zeros((batch_size, max_seq_len), device='cuda')
		
		# First compute next state values for bootstrapping
		next_tokens = self.tokenizer(next_texts, return_tensors='pt', padding=True, add_special_tokens=False)
		next_input_ids = next_tokens.input_ids.to('cuda')
		next_attention_mask = torch.where(next_input_ids == self.tokenizer.pad_token_id, 
										torch.zeros_like(next_input_ids), 
										torch.ones_like(next_input_ids)).to('cuda')
		
		# Get next state values for bootstrapping
		with torch.no_grad():
			next_harm_values = self.harm_critic(
				input_ids=next_input_ids,
				attention_mask=next_attention_mask
			).squeeze().detach()
		
		# Process each sequence individually for token-level analysis
		for b in range(batch_size):
			# Get the length of the input tensor
			input_text = input_texts[b].replace(self.tokenizer.pad_token, '')
			tokens = self.tokenizer(input_text, return_tensors='pt', add_special_tokens=False)
			input_ids = tokens.input_ids.to('cuda')
			attention_mask = tokens.attention_mask.to('cuda')
			seq_len = input_ids.shape[1]
			padding_offset = max_seq_len - seq_len
			
			# Create masked versions of the sequence for token contribution analysis
			expanded_input_ids = input_ids.repeat_interleave(seq_len, dim=0)
			expanded_attention_mask = 1 - torch.eye(seq_len, dtype=attention_mask.dtype, device='cuda')
			padding = torch.full_like(expanded_input_ids[:, :1], self.tokenizer.pad_token_id, device='cuda')
			masked_expanded_input_ids = torch.where(expanded_attention_mask == 1, expanded_input_ids, padding)
			
			# Compute original (non-masked) sequence value for harm with attention divergence
			with torch.set_grad_enabled(True):    
				# Process expanded inputs in minibatches to save memory
				for pos in range(0, seq_len, self.minibatch_size):
					# Get original value with attention divergence regularization
					harm_value, original_harm_token_scores, attention_weights = self.harm_critic(
						input_ids=input_ids,
						attention_mask=attention_mask,
						return_token_scores=True,
						return_attention=True,
						requires_grad=True
					)
					attention_weights = attention_weights[0].detach()
					harm_value = harm_value.squeeze(1)
					end_pos = min(pos + self.minibatch_size, seq_len)
					
					# Get value when token at position pos is masked
					windowed_harm_value, windowed_harm_token_scores, windowed_attention_weights = self.harm_critic(
						input_ids=masked_expanded_input_ids[pos:end_pos],
						attention_mask=expanded_attention_mask[pos:end_pos],
						return_token_scores=True,
						return_attention=True,
						token_level=True,
						requires_grad=True
					)

					windowed_attention_weights = windowed_attention_weights[0].detach()
					# Calculate token contributions as difference between original and masked values
					windowed_harm_token_contributions = harm_value.squeeze().sum(dim=-1) - windowed_harm_value.squeeze().sum(dim=-1)
					# Normalize the token contributions into scores
					windowed_harm_token_sum = windowed_harm_token_contributions.sum()
					if windowed_harm_token_sum != 0:
						windowed_harm_token_scores = windowed_harm_token_contributions / windowed_harm_token_sum * windowed_harm_value.sum()
					else:
						windowed_harm_token_scores = windowed_harm_token_contributions
					attn_div = F.kl_div(windowed_attention_weights, attention_weights, reduction='none').sum(dim=-1).mean(dim=-1).mean(dim=-1)
					windowed_harm_token_scores = windowed_harm_token_scores * attn_div
					windowed_harm_value = (windowed_harm_value * attn_div)#.sum(dim=-1)
					
					# Store token scores
					harm_token_scores_batch[b, padding_offset+pos:padding_offset+end_pos] = windowed_harm_token_scores.squeeze(-1).detach()
					
					# Compute harm ground truth loss based on equation: 
					# V_h(s,a) = R(response) - R(action) + γ*V_h(next_response)
					harm_target = target_harm[b] - action_harm[b] + gamma * next_harm_values[b].sum(dim=-1)
					
					# Normalize harm_target to [0, 1] range
					harm_target_normalized = torch.clamp(harm_target / (torch.abs(harm_target).max() + 1e-8), -1, 1)
					
					harm_pred = torch.full_like(harm_target, harm_value.squeeze().sum(dim=-1).item())
					# Normalize harm_pred to [0, 1] range
					harm_pred_normalized = torch.clamp(harm_pred / (torch.abs(harm_pred).max() + 1e-8), -1, 1)

					harm_loss = F.mse_loss(harm_pred_normalized, harm_target_normalized)
					
					# Also compute regularization loss for token-level predictions
					windowed_harm_loss = F.mse_loss(windowed_harm_value.squeeze(-1), harm_value.repeat_interleave(windowed_harm_value.shape[0], dim=0).squeeze(-1))
					
					# Add attention divergence loss to the combined loss
					# Combine losses with regularization weights and batch normalization
					combined_harm_loss = (
						harm_loss + 
						0.1 * windowed_harm_loss
					) / batch_size
					
					# Compute errors for advantage estimation (using original non-normalized values for TD errors)
					harm_error = harm_value.squeeze().sum(dim=-1) - harm_target
					
					# Backward pass
					combined_harm_loss.backward()
					self.harm_critic_optim.step()
					self.harm_critic.zero_grad()
					
					# Store results
					harm_values[b, padding_offset+pos:padding_offset+end_pos] = harm_value.squeeze()[pos:end_pos].detach()
					harm_errors[b] += harm_error.squeeze(0).detach() * (end_pos - pos)/max_seq_len
		
			torch.cuda.empty_cache()
		
		# Move harm critic to CPU to free GPU memory
		self.harm_critic.to('cpu')
		torch.cuda.empty_cache()
		
		#######################
		# PART 2: HELP VALUES #
		#######################
		
		# Move help critic to CUDA
		self.help_critic.to('cuda')

		# Prepare inputs for help critic
		tokens = self.tokenizer(input_texts, return_tensors='pt', padding=True, add_special_tokens=False)
		input_ids = tokens.input_ids.to('cuda')
		attention_mask = torch.where(input_ids == self.tokenizer.pad_token_id, 
									torch.zeros_like(input_ids), 
									torch.ones_like(input_ids)).to('cuda')
		
		# Get next state values for bootstrapping
		with torch.no_grad():
			next_help_values = self.help_critic(
				input_ids=next_input_ids,
				attention_mask=next_attention_mask
			)
		
		# Get helpfulness scores
		target_help = help_reward.to('cuda')
		
		# Compute help values
		help_value, help_token_scores = self.help_critic(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_token_scores=True,
			requires_grad=True
		)
		
		# Compute help loss based on our theory: 
		# V_help(s,a) = Help(action) + Help(response) + γ*V_help(next_response)
		help_target = action_help + target_help + gamma * next_help_values
		
		# Normalize help_target and help_value to [0, 1] range for loss calculation
		help_target_normalized = torch.clamp(help_target / (torch.abs(help_target).max() + 1e-8), 0, 1)
		help_value_normalized = torch.clamp(help_value / (torch.abs(help_value).max() + 1e-8), 0, 1)
		
		# Use normalized values for loss calculation
		help_loss = nn.MSELoss()(help_value_normalized, help_target_normalized)
		
		# Compute errors for advantage estimation (using original values)
		help_errors = help_value - help_target
		
		# Backward pass
		help_loss.backward()
		self.help_critic_optim.step()
		self.help_critic.zero_grad()
		
		# Clean up
		help_loss = help_loss.detach()
		help_errors = help_errors.detach()
		help_token_scores = help_token_scores.detach() if help_token_scores is not None else None
		help_value = help_value.detach()
		
		# Move help critic to CPU and clean up GPU memory
		self.help_critic.to('cpu')
		
		# If using CPUOffloadValueTransformer, explicitly call cleanup_gpu_memory
		if self.keep_base_on_cpu and hasattr(self.help_critic, 'cleanup_gpu_memory'):
			self.help_critic.cleanup_gpu_memory()
			self.harm_critic.cleanup_gpu_memory()
		else:
			torch.cuda.empty_cache()

		# Return values, errors, losses, and token scores
		return ((harm_values, help_value), 
				(harm_errors, help_errors), 
				(harm_loss, help_loss),
				(harm_token_scores_batch, help_token_scores),
				(action_harm, action_help),
				(next_harm_values, next_help_values))
	
	def scores_to_values(self, scores):
		"""
		Convert token-level scores to sequence-level values.
		
		Args:
			scores (torch.Tensor): Tensor of token-level scores.
			
		Returns:
			torch.Tensor: Tensor of sequence-level values.
		"""
		# Example implementation: summing token scores to get sequence scores
		sequence_values = scores.sum(dim=-1)
		return sequence_values
	
	def act(self, state, base_state, instructions, enforce_single_question=False):
		"""
		Generate an action based on the current state and instructions.
		
		Args:
			state: Current environment state
			instructions: System instructions
			enforce_single_question: Whether to enforce a single question in the output
			breakdown_prompt: Optional prompt for question breakdown
			
		Returns:
			action: Generated action
			act_tokens: Tokenized action
			log_probs: Log probabilities of action
		"""
		assert self.tokenizer.padding_side == 'left'
		with torch.enable_grad():
			prepared_prompts = self.prepare_prompt(state, original_questions=base_state, instructions=instructions, inject_perspective=False)
			tokenized_prompts = self.tokenizer(prepared_prompts, return_tensors='pt', padding=True, add_special_tokens=self.manually_add_bos)
			responses, tokens, log_probs, _ = self.actor.generate(
				tokenized_prompts, 
				enforce_single_question=enforce_single_question,
				require_grad=True,  # Make sure require_grad is True
				k=int(max(4*self.eps, 1))
			)
		
		# Ensure action ends with EOS token
		if isinstance(responses, list):
			responses = [a + self.tokenizer.eos_token if not a.endswith(self.tokenizer.eos_token) else a for a in responses]
		else:
			responses = responses + self.tokenizer.eos_token if not responses.endswith(self.tokenizer.eos_token) else responses
		responses = [a.replace(self.tokenizer.bos_token, '') for a in responses]
				
		return responses, tokens, log_probs.detach()
	
	def prepare_prompt(self, prompt, original_questions, instructions, inject_perspective=True):
		"""returns a list of tokenized prompts with the system and redteam instructions injected.
		If its the first message in the series, prepares it for rephrasing. Otherwise, prepares it for chatting.
		output: list of prompt strings"""
		assert len(prompt) > 0, "Prompt must have at least one message"
		prepared_prompts = []
		prompt_idx = random.randint(0, len(instructions['redteam_prompt']) - 1)
		prompt_injection = instructions['perspectives'][prompt_idx]
		if not inject_perspective:
			prompt_injection = ""
		for x, o in zip(prompt, original_questions):
			if len(x) == 1: #if the prompt is a single string, it is a seed prompt
				if inject_perspective:
					chat = self.setup_prompt(x[0], instructions['sysprompt_initial']['content'], instructions['redteam_prompt'][prompt_idx]['content'])
				else:
					chat = self.setup_prompt(x[0], instructions['sysprompt_initial']['content'], instructions['redteam_prompt_default']['content'])
			else:
				chat = [{'role': 'user', 'content': instructions['redteam_chat_prompt']['content'].format(question=o, conversation=' '.join(x))}]
				chat.insert(0,{'role': 'system', 'content': instructions['sysprompt_chat']['content'] })
			templated_prompt = self.tokenizer.apply_chat_template(chat, return_tensors='pt', padding=True, add_generation_prompt=True, tokenize=False) + prompt_injection
			prepared_prompts.append(templated_prompt)
		return prepared_prompts
	
	def setup_prompt(self, prompt, system_instructions, redteam_instructions):
		return [
			{'role': 'system', 'content': system_instructions},
			{'role': 'user', 'content': redteam_instructions.format(prompt)},
		]
	
	def rollout(self, num_steps):
		"""
			Perform a rollout by interacting with the environment.

			Returns:
				batch_state - the observations collected during the rollout
				batch_acts - the actions taken during the rollout
				batch_log_probs - the log probabilities of the actions taken
				batch_rews - the rewards collected during the rollout, reflecting the harm of the target's response
				batch_compliance_rews - the rewards collected during the rollout, reflecting the helpfulness of the agent's response
				batch_action_help_rews - the rewards collected during the rollout, reflecting the helpfulness of the agent's response
				batch_dones - whether the episode is done
				batch_next_state - the next state of the environment
				batch_base_state - the original adversarial question
				batch_lens - the lengths of each episode
		"""
		# Initialize conversation tracking structures - one list per batch item
		batch_size = self.env.batch_size
		# Create lists of lists where each inner list represents a conversation thread
		conversation_data = {
			'states': [[] for _ in range(batch_size)],
			'next_states': [[] for _ in range(batch_size)],
			'acts': [[] for _ in range(batch_size)],
			'log_probs': [[] for _ in range(batch_size)],
			'rews': [[] for _ in range(batch_size)],
			'dones': [[] for _ in range(batch_size)],
			'base_states': [[] for _ in range(batch_size)],  # List to track base states for each conversation
			'action_help_rews': [[] for _ in range(batch_size)] if self.alg == 'ppo' else None,
			'compliance_rews': [[] for _ in range(batch_size)] if self.alg == 'ppo' else None
		}
		batch_lens = [0 for _ in range(batch_size)]  # Track length of each conversation thread

		start_time = time.time()
		state = self.env.reset()

		torch.cuda.empty_cache()
		
		# Store initial base states (original adversarial questions)
		for i in range(batch_size):
			conversation_data['base_states'][i].append(self.env.base_state[i])

		# Execute rollout steps
		for step_idx in range(int(num_steps//batch_size)):
			with torch.set_grad_enabled(True):
				# Generate actions using the model
				self.to_cuda()
				self.actor.print_memory(f"Memory usage before act")
				action, act_tokens, log_probs = self.act(state, self.env.base_state, self.instructions, enforce_single_question=True)
				self.to_cpu()
				
				# Check for empty actions
				for s, a in zip(state, action):
					if a.replace("user", "").replace("assistant", "").replace("system", "").strip() == "":
						print("Empty action: ", s, a)
				
				# Track which conversations are already done before this step
				already_done = self.env.dones.clone()
				
				# Take the step in the environment
				next_state, responses, reward, done = self.env.step(action)
				# Store data for each conversation thread that's not already done
				for i in range(batch_size):
					if already_done[i].item() == 0:  # Only update active conversations
						batch_lens[i] += 1
						conversation_data['states'][i].append(state[i])
						conversation_data['next_states'][i].append(next_state[i])
						conversation_data['acts'][i].append(act_tokens[i])
						conversation_data['rews'][i].append(reward[i])
						conversation_data['dones'][i].append(done[i])
						conversation_data['log_probs'][i].append(log_probs[i].detach())
						
						# Additional rewards for PPO
						if self.alg == 'ppo':
							# Get helpfulness of agent's action
							action_help_score = self.env.score_helpfulness(
								[[state[i][-1], action[i]]]
							)
							# Get compliance reward from target's response
							compliance_rew = self.env.score_helpfulness(
								[next_state[i]]
							)
							conversation_data['action_help_rews'][i].append(action_help_score[0])
							conversation_data['compliance_rews'][i].append(compliance_rew[0])
				
				# Update the current state
				state = next_state
				
				# If all conversations are done, the environment will automatically reset
				# We need to update our base states to the new ones
				if done.all():
					# The environment automatically resets when all are done
					# Update base states for the next batch of conversations
					for i in range(batch_size):
						# The next state now contains the first message of the new conversation
						conversation_data['base_states'][i].append(self.env.base_state[i])
				
				if step_idx == num_steps - 1:
					break  # We've reached the max steps
		
		# Process the collected data into flat lists that maintain chronological order
		batch_state = []
		batch_next_state = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_dones = []
		batch_base_state = []
		batch_action_help_rews = [] if self.alg == 'ppo' else None
		batch_compliance_rews = [] if self.alg == 'ppo' else None
		
		# Flatten the conversation data while maintaining chronological order
		for i in range(batch_size):
			states_len = len(conversation_data['states'][i])
			if states_len > 0:  # Only process non-empty conversations
				batch_state.extend(conversation_data['states'][i])
				batch_next_state.extend(conversation_data['next_states'][i])
				batch_acts.extend(conversation_data['acts'][i])
				batch_log_probs.extend(conversation_data['log_probs'][i])
				batch_rews.extend(conversation_data['rews'][i])
				batch_dones.extend(conversation_data['dones'][i])
				
				# Add base states - one for each step in the conversation
				# We need to handle the case where a conversation spans multiple base states 
				# (i.e., the environment reset during this conversation thread)
				base_state_idx = 0
				for j in range(states_len):
					# If this step marks a done state and it's not the last step, increment the base state index
					if j < states_len - 1 and conversation_data['dones'][i][j].item() == 1:
						base_state_idx += 1
					
					# Make sure we don't go out of bounds
					if base_state_idx < len(conversation_data['base_states'][i]):
						batch_base_state.append(conversation_data['base_states'][i][base_state_idx])
					else:
						# Fallback if we somehow have more states than base states
						batch_base_state.append(conversation_data['base_states'][i][-1])
				
				if self.alg == 'ppo':
					batch_action_help_rews.extend(conversation_data['action_help_rews'][i])
					batch_compliance_rews.extend(conversation_data['compliance_rews'][i])
		# Skip further processing if no data was collected
		if len(batch_state) == 0:
			print(conversation_data)
			raise RuntimeError("No valid data collected in rollout - all conversations were done")
			
		# Tokenize and process the collected data
		batch_state = self.tokenizer(self.prepare_prompt(batch_state, batch_base_state, self.instructions, inject_perspective=False), 
									return_tensors='pt', padding=True,
									add_special_tokens=self.manually_add_bos).input_ids.squeeze()
		batch_next_state = self.tokenizer(self.prepare_prompt(batch_next_state, batch_base_state, self.instructions, inject_perspective=False), 
										return_tensors='pt', padding=True,
										add_special_tokens=self.manually_add_bos).input_ids.squeeze()
		batch_base_state = self.tokenizer(batch_base_state, return_tensors='pt', padding=True).input_ids.squeeze()
		batch_acts = self._pad_batch_acts(batch_acts)
		batch_rews = torch.stack(batch_rews).squeeze()
		batch_dones = torch.stack(batch_dones).squeeze()
		
		if self.alg == 'ppo':
			batch_action_help_rews = torch.stack(batch_action_help_rews).squeeze()
			batch_compliance_rews = torch.stack(batch_compliance_rews).squeeze()
		
		batch_log_probs = self._pad_batch_log_probs(batch_log_probs, batch_acts)
		
		# Verify shapes if we have data
		if batch_state.numel() > 0:
			assert batch_state.shape[0] == batch_acts.shape[0], f"Batch state and batch acts have different lengths: {batch_state.shape[0]} != {batch_acts.shape[0]}"
			assert batch_acts.shape == batch_log_probs.shape, f"Batch acts and batch log probs have different lengths: {batch_acts.shape} != {batch_log_probs.shape}"
		
		print(f"Rollouts collected: {batch_state.shape[0]} in {time.time() - start_time} seconds. Average reward: {batch_rews.sum()/batch_state.shape[0]}. Tokens per batch: {torch.tensor(batch_state.shape).prod()}")
		torch.cuda.empty_cache()
		
		# Create a rollout object and return it
		content = {
			'batch_state': batch_state,
			'batch_next_state': batch_next_state,
			'batch_acts': batch_acts,
			'batch_log_probs': batch_log_probs,
			'batch_rews': batch_rews,
			'batch_dones': batch_dones,
			'batch_lens': batch_lens,
			'batch_base_state': batch_base_state
		}
		if self.alg == 'ppo':
			content['batch_action_help_rews'] = batch_action_help_rews
			content['batch_compliance_rews'] = batch_compliance_rews
		rollout = type('Rollout', (object,), content)()

		# Print one sample from the rollout if available
		if batch_state.numel() > 0:
			sample_idx = np.argmax(batch_rews)
			sample_q = self.tokenizer.decode(batch_state[sample_idx], skip_special_tokens=True)
			sample_a = self.tokenizer.decode(batch_acts[sample_idx], skip_special_tokens=True)
			print(f"Sample: \nQuestion: {sample_q}\nRephrase: {sample_a}\nReward: {batch_rews[sample_idx]}\n")
		self.actor.print_memory(f"Memory usage after rollout")
		return rollout
	
	def breakdown_rollout(self, t):
		"""
		Perform a rollout specifically for question breakdown strategy.
		Similar to rollout() but with special handling for question decomposition.
		
		Returns:
			batch_state - observations collected during rollout [batch_size, num_messages, message_length]
			batch_acts - actions taken during rollout
			batch_log_probs - log probabilities of actions taken
			batch_rtgs - rewards-to-go calculated from rewards
			batch_lens - lengths of each episode
		"""
		batch_state = []  # Will contain states with shape [batch_size, num_messages, message_length]
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_dones = []
		batch_next_state = []
		batch_base_state = []
		batch_compliance_rews = []
		batch_action_help_rews = []
		ep_rews = []
		# Secret context will store the atomic questions for each batch item
		self.secret_contexts = [[] for _ in range(self.batch_size)]
		
		start_time = time.time()
		state = self.env.reset()  # Get initial questions
		
		# First step: Break down questions into atomic pieces
		breakdown_prompt = self.instructions['breakdown_prompt']['content'].format(question=state)
		atomic_questions, act_tokens, log_probs = self.act_secret(
			state, 
			self.instructions,
			breakdown_prompt=breakdown_prompt
		)
		
		# Store atomic questions in secret context
		for i, questions in enumerate(atomic_questions):
			self.secret_contexts[i] = questions.split('\n')  # Split into list of questions
		current_ep_rews = 0
		# Now proceed with regular steps, but using secret context
		for i in range(min(int(t/self.batch_size), self.max_rollout_len)):
			with torch.set_grad_enabled(True):
				self.to_cuda()
				
				# Generate action using secret context
				action, act_tokens, log_probs = self.act_secret(
					state, 
					self.instructions, 
					secret_context=self.secret_contexts
				)
				
				self.to_cpu()
				next_state, responses, reward, done = self.env.step(action)
				self.to_cuda()
				# Collect episode data
				for i in range(len(state)):
					batch_state.append(state[i])
					batch_base_state.append(self.env.base_state[i])
					batch_next_state.append(next_state[i])
					batch_rews.append(reward[i])
					batch_acts.append(act_tokens[i])
					batch_dones.append(done[i])
					batch_log_probs.append(log_probs[i].detach())
					current_ep_rews += reward[i]
				if self.alg == 'ppo':
					# Get helpfulness scores
					action_help_score = self.env.score_helpfulness(
						[[s[-1], a] for s, a in zip(state, action)]
					)
					compliance_rew = self.env.score_helpfulness(next_state)
					batch_action_help_rews.append(action_help_score)
					batch_compliance_rews.append(compliance_rew)

				state = next_state
				if check_device_mem(0.5):
					break
				if done.all():
					ep_rews.append(current_ep_rews/len(state))
					current_ep_rews = 0
		# Convert lists to tensors with proper shapes
		batch_state = self.tokenizer(self.prepare_prompt_with_secret_context(batch_state, self.instructions, self.secret_contexts), return_tensors='pt', padding=True).input_ids.squeeze()  # [num_steps, batch_size, num_messages, message_length]
		batch_base_state = self.tokenizer(batch_base_state, return_tensors='pt', padding=True).input_ids.squeeze()
		batch_next_state = self.tokenizer(self.prepare_prompt_with_secret_context(batch_next_state, self.instructions, self.secret_contexts), return_tensors='pt', padding=True).input_ids.squeeze()
		batch_acts = self._pad_batch_acts(batch_acts)
		batch_log_probs = self._pad_batch_log_probs(batch_log_probs, batch_acts)
		print(f"Rollouts collected: {batch_state.shape[0]} in {time.time() - start_time} seconds. Average reward: {np.mean(ep_rews)}. Tokens per batch: {torch.tensor(batch_state.shape).prod()}")
		content = {
			'batch_state': batch_state, 
			'batch_next_state': batch_next_state,
			'batch_acts': batch_acts,
			'batch_log_probs': batch_log_probs, 
			'batch_rews': batch_rews, 
			'batch_dones': batch_dones,
			'batch_base_state': batch_base_state
		}
		if self.alg == 'ppo':
			content['batch_action_help_rews'] = batch_action_help_rews
			content['batch_compliance_rews'] = batch_compliance_rews
		rollout = type('Rollout', (object,), content)()
		return rollout

	def act_secret(self, state, instructions, breakdown_prompt=None, secret_context=None):
		"""
		Generate an action using either breakdown prompt or secret context.
		
		Args:
			state: Current environment state
			instructions: System instructions
			breakdown_prompt: If provided, used for initial breakdown step
			secret_context: If provided, used for subsequent steps
			
		Returns:
			action: Generated action
			act_tokens: Tokenized action
			log_probs: Log probabilities of action
		"""
		if breakdown_prompt is not None:
			# Initial breakdown step
			messages = [
				{"role": "system", "content": instructions['sysprompt']['content']},
				{"role": "user", "content": breakdown_prompt}
			]
			
			# Apply chat template
			prepared_prompts = self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True
			)
			
		else:
			prepared_prompts = self.prepare_prompt_with_secret_context(state, instructions, secret_context)

		# Generate response using the actor
		action, act_tokens, log_probs = self.actor.generate(
			prepared_prompts,
			max_new_tokens=100,
			k=3,
			enforce_single_question=(breakdown_prompt is None)
		)
		
		# Ensure action ends with EOS token
		if isinstance(action, list):
			action = [a + self.tokenizer.eos_token if not a.endswith(self.tokenizer.eos_token) else a for a in action]
		else:
			action = action + self.tokenizer.eos_token if not action.endswith(self.tokenizer.eos_token) else action
		
		return action, act_tokens, log_probs
	
	def prepare_prompt_with_secret_context(self, state, instructions, secret_context):
		prepared_prompts = []
		assert isinstance(secret_context, list)
		assert isinstance(secret_context[0], str)
		assert isinstance(state, list)
		assert isinstance(state[0], list)
		assert isinstance(state[0][0], str)
		for i in range(len(secret_context)):
			context_prompt = "You have prepared these atomic questions:\n"
			for q in secret_context[i]:
				context_prompt += f"- {q}\n"
			context_prompt += "\nUse these to guide your natural conversation."
			messages = build_chat(state[i], user_num=0)
			messages.insert(0, {"role": "system", "content": instructions['breakdown_sysprompt']['content'] + "\n" + context_prompt})
			prepared_prompt = self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True
			)
			if self.manually_add_bos:
				prepared_prompt = self.tokenizer.bos_token + prepared_prompt
			prepared_prompts.append(prepared_prompt)
		return prepared_prompts

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default hyperparameters.

			Args:
				hyperparameters - dictionary of hyperparameters with custom values.
					'lr' - learning rate of the actor optimizer
					'gamma' - the discount factor
					'batch_size' - the batch size to use
					'epochs' - the number of epochs to train for
					'clip' - the clipping parameter for PPO
					'seed' - the random seed to use
					'ce_coef' - cross entropy loss coefficient
					'target_model' - path to target model to use
					'detune' - whether to use detuned models
					'help_weight' - Weight for helpfulness rewards 
					'lora_r' - the rank for LoRA
					'lora_alpha' - the alpha parameter for LoRA
					'lora_dropout' - the dropout rate for LoRA layers
					'use_harm_error' - whether to use harm_error directly as TD error
					'keep_base_on_cpu' - whether to keep the base model on CPU while adapters are on GPU for memory efficiency
					'value_loss_coef' - coefficient to scale the value loss to balance with actor loss
					'attention_divergence_weight' - weight for attention divergence regularization loss (default: 0.1)
			
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
		'adv_coef': 0.1,
		'ce_coef': 0.5,
		'help_weight': 0.5,  # Weight for helpfulness rewards
		'detune': False,  # Default to not using detuned models
		'target_model': None,  # Default to using the shared base model
		'use_lora': False,  # Default to not using LoRA
		'lora_r': 8,  # LoRA rank
		'lora_alpha': 16,  # LoRA alpha
		'lora_dropout': 0.05,  # LoRA dropout
		'use_harm_error': False,  # Whether to use harm_error directly as TD error
		'keep_base_on_cpu': False,  # Whether to keep the base model on CPU and only adapters on GPU for memory efficiency
		'value_loss_coef': 0.5,  # Coefficient to scale the value loss to balance with actor loss
		'attention_divergence_weight': 0.001,  # Weight for attention divergence regularization loss
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
			np.random.seed(self.seed)
			
		return hyperparameters

	def load_lora_adapter(self, adapter_type):
		"""
		Load a LoRA adapter.
		
		Args:
			adapter_type: Type of adapter to load ('lm_head_rephrase', 'attention_rephrase', 
												'lm_head_followup', 'attention_followup', 'attention_holistic')
		
		Returns:
			bool: Whether the adapter was successfully loaded
		"""
		if not self.use_lora:
			print("LoRA is not enabled")
			return False
			
		if adapter_type not in self.lora_adapters:
			print(f"Unknown adapter type: {adapter_type}")
			return False
			
		adapter_path = self.lora_adapters[adapter_type]
		
		# Check if adapter exists
		if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
			print(f"Adapter {adapter_type} does not exist yet, will be created during training")
			return True
			
		# Load existing adapter
		print(f"Loading LoRA adapter: {adapter_type}")
		
		# If the adapter is already in our cache, use it
		if adapter_type in self.lora_adapters.keys():
			return True
			
		# Otherwise, load it from disk
		if adapter_type not in self.lora_adapters.keys():
			policy_class = getPolicyClass('Transformer')
			adapter_model = policy_class(
				self.checkpoint, 
				self.tokenizer, 
				model=self.agent_model_singleton.base_model,  # Use the singleton base model
				token=None,
				use_lora=True,
				lora_config=self.lora_configs[adapter_type],
				adapter_name=adapter_type
			)
			
			# Load the adapter
			success = adapter_model.load_lora_adapters(adapter_path)
			
			if success:
				# Store the adapter model
				self.lora_adapters[adapter_type] = adapter_model
				return True
			else:
				return False
		
		return True
		
	def save_lora_adapter(self, adapter_type):
		"""
		Save the current LoRA adapter.
		
		Args:
			adapter_type: Type of adapter to save ('lm_head_rephrase', 'attention_rephrase', 
												'lm_head_followup', 'attention_followup', 'attention_holistic')
		
		Returns:
			bool: Whether the adapter was successfully saved
		"""
		if not self.use_lora:
			print("LoRA is not enabled")
			return False
			
		if adapter_type not in self.lora_adapters:
			print(f"Unknown adapter type: {adapter_type}")
			return False
			
		# Make sure the adapter exists in the base model
		if adapter_type not in self.agent_model_singleton.available_adapters:
			print(f"Adapter '{adapter_type}' not found in base model, nothing to save")
			return False
			
		adapter_path = self.lora_adapters[adapter_type]
		
		# Save the adapter
		print(f"Saving LoRA adapter: {adapter_type} to {adapter_path}")
		
		# Make sure the adapter is active before saving
		current_adapter = self.current_adapter
		self.agent_model_singleton.switch_adapter(adapter_type)
		
		try:
			# Save the adapter using PEFT's save_pretrained method
			if hasattr(self.agent_model_singleton.base_model, 'save_pretrained'):
				os.makedirs(adapter_path, exist_ok=True)
				self.agent_model_singleton.base_model.save_pretrained(adapter_path)
				print(f"Successfully saved adapter '{adapter_type}' to {adapter_path}")
				success = True
			else:
				print(f"Error: Model doesn't have save_pretrained method")
				success = False
		except Exception as e:
			print(f"Error saving adapter '{adapter_type}': {e}")
			success = False
		
		# Switch back to the previous adapter
		if current_adapter and current_adapter != adapter_type:
			self.agent_model_singleton.switch_adapter(current_adapter)
			
		return success
		
	def initialize_adapter_models(self):
		"""
		Initialize all adapter models and add them to the shared BaseModelSingleton.
		This ensures we have a single PEFT model with multiple adapters.
		"""
		if not self.use_lora:
			print("LoRA not enabled, skipping adapter initialization")
			return
		
		print(f"Initializing adapter models: {list(self.lora_adapters.keys())}")
		if not hasattr(self.agent_model_singleton, 'peft_model') or self.agent_model_singleton.peft_model is None:
			print("No PEFT model exists in singleton, creating a new one for agent adapters")
			# We'll initialize the first adapter when we process it below
		
		for adapter_type in self.lora_adapters.keys():
			adapter_path = self.lora_adapters[adapter_type]
			config_path = os.path.join(adapter_path, "adapter_config.json")
			
			try:
				# Check if adapter already exists in the singleton
				if self.agent_model_singleton.peft_model is not None and adapter_type in self.agent_model_singleton.available_adapters:
					print(f"Adapter '{adapter_type}' already exists in base model, skipping initialization.")
					
					# Ensure parameters are marked trainable regardless
					if hasattr(self.agent_model_singleton.peft_model, 'peft_config'):
						trainable_count = 0
						for name, param in self.agent_model_singleton.peft_model.named_parameters():
							if adapter_type in name and "lora" in name.lower():
								param.requires_grad = True
								trainable_count += 1
						if trainable_count > 0:
							print(f"Ensured {trainable_count} parameters for adapter '{adapter_type}' are trainable")
					
					continue
				
				print(f"Adding adapter '{adapter_type}' to base model...")

				# --- Convert config dict to LoraConfig object --- 
				adapter_config_dict = self.lora_configs[adapter_type]
				lora_config_obj = LoraConfig(
					r=adapter_config_dict.get('r', 8), # Use .get for safety
					lora_alpha=adapter_config_dict.get('lora_alpha', 16),
					lora_dropout=adapter_config_dict.get('lora_dropout', 0.05),
					target_modules=adapter_config_dict.get('target_modules', ['q_proj', 'v_proj']),
					bias=adapter_config_dict.get('bias', 'none'),
					task_type=TaskType.CAUSAL_LM # Specify task type
				)
				# -----------------------------------------------------
				
				# Load existing adapter if it exists, otherwise create new
				if os.path.exists(config_path):
					print(f"Loading existing adapter '{adapter_type}' from {adapter_path}")
					# Pass the LoraConfig object when loading/adding
					success = self.agent_model_singleton.add_adapter(adapter_type, lora_config_obj, adapter_path)
					if not success:
						print(f"Failed to load adapter '{adapter_type}', creating a new one")
						# Pass the LoraConfig object when creating new
						self.agent_model_singleton.add_adapter(adapter_type, lora_config_obj)
				else:
					# Create a new adapter
					print(f"No existing adapter found for '{adapter_type}', creating a new one")
					# Pass the LoraConfig object when creating new
					self.agent_model_singleton.add_adapter(adapter_type, lora_config_obj)
				
				# After adding an adapter, ensure proper device placement
				self.agent_model_singleton.to_appropriate_devices()
				
				# Initialize optimizer for this adapter
				model = self.agent_model_singleton.base_model
				
				# Set adapter as active to find adapter parameters
				self.agent_model_singleton.switch_adapter(adapter_type)
				
				# Create optimizer for this adapter's parameters
				adapter_params = []
				for name, param in model.named_parameters():
					# Check for both "lora" in name and adapter_type to ensure we're only getting params for this adapter
					if "lora" in name.lower() and adapter_type in name and param.requires_grad:
						adapter_params.append(param)
						
				if adapter_params:
					self.lora_optimizers[adapter_type] = torch.optim.Adam(adapter_params, lr=self.lr)
					print(f"Created optimizer for adapter '{adapter_type}' with {len(adapter_params)} parameters")
				else:
					print(f"Warning: No trainable parameters found for adapter '{adapter_type}'")
					
					# Attempt to fix: mark all params with this adapter name as trainable
					fixed_count = 0
					for name, param in model.named_parameters():
						if adapter_type in name:
							param.requires_grad = True
							adapter_params.append(param)
							fixed_count += 1
					
					if fixed_count > 0:
						print(f"Fixed: Marked {fixed_count} parameters as trainable for adapter '{adapter_type}'")
						self.lora_optimizers[adapter_type] = torch.optim.Adam(adapter_params, lr=self.lr)
					
			except Exception as e:
				print(f"Error initializing adapter '{adapter_type}': {e}")
				import traceback
				traceback.print_exc()
		
		# Set the initial adapter if any were successfully added
		if self.agent_model_singleton.available_adapters:
			first_adapter = list(self.agent_model_singleton.available_adapters)[0]
			success = self.switch_lora_adapter(first_adapter)
			if success:
				print(f"Set initial adapter to '{first_adapter}'")
				
				# Verify the actor has trainable parameters
				try:
					self.actor.check_layer_grads()
				except AssertionError as e:
					print(f"WARNING: Actor model has no trainable parameters after initialization: {e}")
					print("Attempting to fix...")
					
					# Force mark parameters for active adapter as trainable
					if hasattr(self.actor.model, 'active_adapter'):
						fixed_count = 0
						adapter_name = self.actor.model.active_adapter
						for name, param in self.actor.model.named_parameters():
							if adapter_name in name:
								param.requires_grad = True
								fixed_count += 1
						print(f"Marked {fixed_count} parameters as trainable for adapter '{adapter_name}'")
			else:
				print(f"Failed to set initial adapter to '{first_adapter}'")
		else:
			print("No adapters available in base model")
			
		# Final device check to ensure everything is placed correctly
		if self.keep_base_on_cpu:
			self.agent_model_singleton.to_appropriate_devices()

	def switch_lora_adapter(self, adapter_type):
		"""
		Switch to a different LoRA adapter.
		
		Args:
			adapter_type: Type of adapter to switch to
		
		Returns:
			bool: Whether the adapter was successfully switched
		"""
		if not self.use_lora:
			print("LoRA is not enabled")
			return False
			
		if adapter_type not in self.lora_adapters:
			print(f"Unknown adapter type: {adapter_type}")
			return False
			
		print(f"Switching to adapter: {adapter_type}")
		
		# If the adapter is already active, do nothing
		if self.current_adapter == adapter_type:
			print(f"Adapter {adapter_type} is already active")
			return True
		
		# If we've merged the PEFT model, we need to re-create it
		if not hasattr(self.agent_model_singleton, 'peft_model') or self.agent_model_singleton.peft_model is None:
			print("No PEFT model exists (possibly merged). Initializing new adapters...")
			
			# Try to load the adapter if it exists, otherwise create a new one
			adapter_path = self.lora_adapters[adapter_type]
			config_path = os.path.join(adapter_path, "adapter_config.json")
			
			if os.path.exists(config_path):
				# Add adapter to base model
				success = self.agent_model_singleton.add_adapter(adapter_type, self.lora_configs[adapter_type], adapter_path)
			else:
				# Create new adapter
				success = self.agent_model_singleton.add_adapter(adapter_type, self.lora_configs[adapter_type])
				
			if not success:
				print(f"Failed to initialize adapter: {adapter_type}")
				return False
		
		# Switch the adapter in the base model singleton
		success = self.agent_model_singleton.switch_adapter(adapter_type)
		
		if success:
			self.current_adapter = adapter_type
			
			# Update the actor to use the same model with the new adapter
			self.actor.model = self.agent_model_singleton.base_model
			self.actor_optim = self.lora_optimizers[adapter_type]

			# Ensure only LoRA parameters for this adapter are trainable
			trainable_params_count = 0
			for name, param in self.actor.model.named_parameters():
				# If using PEFT, only set parameters for this adapter to trainable
				if hasattr(self.actor.model, 'active_adapter'):
					# Check if the parameter belongs to the active adapter
					is_adapter_param = False
					adapter_name = self.actor.model.active_adapter					# Look for specific patterns that indicate this parameter is for the current adapter
					if f"{adapter_name}." in name or name.startswith(f"{adapter_name}_"):
						is_adapter_param = True
					elif "lora" in name.lower() and not any(other_name in name for other_name in self.lora_adapters if other_name != adapter_name):
						is_adapter_param = True
					
					# Set requires_grad based on whether this is an adapter parameter
					if is_adapter_param:
						param.requires_grad_(True)
						trainable_params_count += 1
					else:
						param.requires_grad_(False)
				else:
					# Without PEFT, use simpler check - if it has "lora" in the name
					if "lora" in name.lower():
						param.requires_grad_(True)
						trainable_params_count += 1
					else:
						param.requires_grad_(False)
			
			print(f"Marked {trainable_params_count} parameters as trainable for adapter '{adapter_type}'")
			
			# Validate that adapter parameters are trainable
			self.actor.check_layer_grads()
			
			# Handle device placement based on keep_base_on_cpu setting
			if self.keep_base_on_cpu:
				self.agent_model_singleton.to_appropriate_devices()
					
			print(f"Successfully switched to adapter '{adapter_type}'")
			return True
		else:
			print(f"Failed to switch to adapter '{adapter_type}'")
			return False

	def _train_on_batch(self, batch_data, num_epochs=None):
		"""
		Train on a batch of data.
		"""
		# Track initial GPU memory for leak detection
		initial_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
		
		# Use default epochs if not specified
		epochs = num_epochs if num_epochs is not None else self.epochs
		
		batch_state = batch_data.batch_state
		batch_acts = batch_data.batch_acts
		batch_log_probs = batch_data.batch_log_probs
		batch_next_state = batch_data.batch_next_state
		batch_base_state = batch_data.batch_base_state
		batch_rews = batch_data.batch_rews
		batch_compliance_rews = batch_data.batch_compliance_rews
		batch_action_help_rews = batch_data.batch_action_help_rews
		batch_advantages = None  # Will be computed during training
		batch_values = None
		
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.
			The method interleaves value and policy updates for each minibatch to leverage
			error information for better advantage estimation.

			Return:
				None
		"""
		t = 0 # Timesteps simulated so far
		with torch.set_grad_enabled(True):
			assert self.batch_size >= self.minibatch_size, "Minibatch size must be smaller than batch size"                             

			self.to_cuda()
			self.actor.train()
			if self.alg == 'ppo':
				self.harm_critic.train()
				self.help_critic.train()
			self.actor.check_layer_grads()

			# Calculate how many timesteps we collected this batch
			try:
				t += np.sum(batch_data.batch_lens)
				ep_lens = batch_data.batch_lens / batch_data.batch_state.shape[0]
			except:
				t += batch_state.shape[0]  # Approximate if batch_lens not available
				ep_lens = [batch_state.shape[0]]
			self.logger['timesteps_cumulative'].append(t)
			self.logger['batch_lens'].append(ep_lens)
			# This is the loop where we update our network for some n epochs
			for epoch in range(epochs):
				# Minibatch training - interleave value and policy updates
				window_size = self.minibatch_size
				for j in range(0, batch_state.size(0), window_size):
					minibatch_state = batch_state[j:j + window_size]
					minibatch_next_state = batch_next_state[j:j + window_size]
					minibatch_base_state = batch_base_state[j:j + window_size]
					minibatch_acts = batch_acts[j:j + window_size]
					minibatch_log_probs = batch_log_probs[j:j + window_size]
					minibatch_rews = batch_rews[j:j + window_size]
					if self.alg == 'ppo':
						minibatch_comply_rews = batch_compliance_rews[j:j + window_size] 
						minibatch_action_help_rews = batch_action_help_rews[j:j + window_size]
						
						# 1. First do the value update for this minibatch
						(harm_values, help_values), (harm_errors, help_errors), (harm_critic_loss, help_critic_loss), \
						(harm_token_scores, help_token_scores), (action_harm, action_help), (next_harm_pred, next_help_pred) = self.value_step(
							minibatch_state,
							minibatch_acts, 
							minibatch_rews,
							minibatch_comply_rews,  
							minibatch_next_state,
							minibatch_base_state,
							action_help_reward=minibatch_action_help_rews
						)
						
						# Log both critic losses - ensure detachment before moving to CPU
						self.logger['harm_critic_losses'].append(harm_critic_loss.detach().cpu())
						self.logger['help_critic_losses'].append(help_critic_loss.detach().cpu())
					
					# 2. Now do the policy update for this same minibatch
					self.to_cuda()
					minibatch_state.to('cuda')
					minibatch_acts.to('cuda')
					if epoch == 0 and j == 0:
						self.actor.print_memory(f"Memory usage before policy update for minibatch {j}, batch size {batch_state.size(0)}, with adapter {self.current_adapter}")
					with torch.enable_grad():
						# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
						log_probs_old = minibatch_log_probs.to('cuda')
						curr_log_probs, ce_loss = self.actor.get_log_probs(minibatch_state, minibatch_acts, sequence_level=False)
						
						#compute A_k according to which algorithm we are using
						if self.alg == 'reinforce':
							A_k = minibatch_rews.to('cuda') - minibatch_rews.mean().to('cuda')
							if A_k.ndim == 1:
								A_k = A_k.unsqueeze(1)
						elif self.alg == 'ppo-pretrain':
							# Use the environment's score_safety function as the critic
							target_response = self.tokenizer.batch_decode(minibatch_next_state, skip_special_tokens=True)
							target_response = [x.split('assistant')[-1] for x in target_response]
							target_response = [x.split('user')[-1] for x in target_response]
							current_state_action_tokens = torch.cat((minibatch_state.to('cuda'), minibatch_acts.to('cuda')), dim=1)
							current_state_action = self.tokenizer.batch_decode(current_state_action_tokens, skip_special_tokens=True)
							values = self.env.score_safety(current_state_action)
							next_values = self.env.score_safety(target_response)
							A_k, returns = self._compute_advantages(minibatch_rews.to('cuda'), values.to('cuda'), next_values=next_values.to('cuda'))
						elif self.alg == 'ppo':								
							help_rews = minibatch_comply_rews + minibatch_action_help_rews
							
							# Compute GAE with sequence-level predicted rewards
							# Note: Even for the harm critic which has token-level outputs,
							# we use the sequence-level aggregated value (sum of token values) for GAE calculation
							#next_help_pred = self.help_critic(minibatch_next_state)
							help_A_k = self._compute_advantages(help_rews.to('cuda'), help_values.detach(), next_values=next_help_pred.to('cuda'), token_level=False)[0]
							
							# For harm rewards, we can directly use the harm_error as the TD error
							# Instead of recalculating TD error in _compute_advantages
							harm_rews = minibatch_rews.to('cuda') - action_harm.sum(dim=-1)
							if self.use_harm_error:
								# Use harm_error directly as the TD error for advantage estimation
								harm_A_k = self._compute_advantages(
									rewards=harm_rews,  # Not used when td_error is provided
									values=harm_values.detach(),
									token_level=True,
									td_error=harm_errors.detach()
								)[0]
							else:
								# Original calculation using regular TD error calculation
								harm_A_k = self._compute_advantages(harm_rews.to('cuda'), harm_values.detach(), next_values=next_harm_pred.to('cuda'), token_level=True)[0]
							
							# Use advantages for policy update
							A_k = harm_A_k * (1 - self.help_weight) + help_A_k.unsqueeze(-1).repeat_interleave(harm_A_k.shape[-1], -1) * self.help_weight
							
						elif self.alg == 'grpo':
							raise NotImplementedError(f"Algorithm {self.alg} not implemented")
						else:
							raise NotImplementedError(f"Algorithm {self.alg} not implemented")

						A_k.requires_grad_(True)
						curr_log_probs.requires_grad_(True)
						# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
						ratios = curr_log_probs - log_probs_old
						if A_k.shape != ratios.shape:
							assert A_k.shape[0] == ratios.shape[0]
							assert A_k.ndim == 2, f"A_k should be 2D but got shape {A_k.shape}"
							assert ratios.ndim == 2, f"ratios should be 2D but got shape {ratios.shape}. curr_log_probs: {curr_log_probs.shape}, log_probs_old: {log_probs_old.shape}"
							assert self.sum_values_into_action, "sum_values_into_action must be True if A_k and ratios have (intentionally) different shapes"
							
							# Sum the values into the action dimension
							A_k = A_k.sum(dim=-1).unsqueeze(-1)
						
						surr1 = ratios * A_k
						surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
						
						semantic_score = self.actor.get_embed_similarity(minibatch_state, minibatch_base_state)
						kl_penalty = F.kl_div(curr_log_probs, log_probs_old, reduction='batchmean')
						if kl_penalty.isnan().any():
							kl_penalty = torch.zeros_like(kl_penalty)
						# Calculate the loss
						actor_loss = (-torch.min(surr1, surr2)).mean() - (semantic_score.mean().to(ratios.device))*self.sem_coef
						actor_loss += ce_loss.mean() * self.ce_coef
						actor_loss += -kl_penalty.mean() * self.kl_coef

						if epoch == 0 and j == 0:
							report_str = f"a_k mean: {A_k.mean()}, actor_loss: {actor_loss}, mean surr1: {surr1.mean()}, mean surr2: {surr2.mean()}"
							if self.alg == 'ppo':
								report_str += f", mean acts_harm: {action_harm.mean()}, mean acts_help: {action_help.mean()}, mean semantic_score: {semantic_score.mean()}"
							print(report_str)
						# Update actor
						self.actor_optim.zero_grad()
						actor_loss.backward()
						self.actor_optim.step()
						
						# Log actor loss
						self.logger['actor_losses'].append(actor_loss.detach().cpu().numpy())
						self.logger['batch_rews'].append(minibatch_rews.detach().cpu().numpy())
						if self.alg == 'ppo':
							self.logger['harm_critic_losses'].append(harm_critic_loss.detach().cpu().numpy())
							self.logger['help_critic_losses'].append(help_critic_loss.detach().cpu().numpy())
							self.logger['batch_help_rews'].append(help_rews.detach().cpu().numpy())
							self.logger['batch_harm_rews'].append(harm_rews.detach().cpu().numpy())
							self.logger['ce_losses'].append(ce_loss.detach().cpu().numpy())
							self.logger['batch_similarity'].append(semantic_score.detach().cpu().numpy())
							
							# Clean up PPO-specific tensors
							del help_rews, harm_rews, help_A_k, harm_A_k, harm_values, help_values
							del harm_errors, help_errors, harm_token_scores, help_token_scores
							del action_harm, action_help, next_harm_pred, next_help_pred
						
						# Explicitly delete intermediate tensors
						del log_probs_old, curr_log_probs, ce_loss, A_k, ratios, surr1, surr2, semantic_score
						
					# Clear CUDA cache to prevent memory issues
					torch.cuda.empty_cache()

			self._log_summary()
		
			self.actor.zero_grad()
			self.actor_optim.zero_grad()
			if self.alg == 'ppo':
				self.harm_critic_optim.zero_grad()
				self.help_critic_optim.zero_grad()
			
			# Thorough cleanup of batch tensors
			del batch_state, batch_acts, batch_log_probs, batch_next_state, batch_base_state
			del batch_rews, batch_compliance_rews, batch_action_help_rews
			
			# Special cleanup for CPUOffloadTransformerPolicy
			if isinstance(self.actor, CPUOffloadTransformerPolicy):
				# Call the dedicated cleanup method
				self.actor.cleanup_gpu_memory()
				
				# Ensure lm_head_clone is moved back to CPU
				if hasattr(self.actor, 'lm_head_clone') and self.actor.lm_head_clone.weight.device.type == 'cuda':
					self.actor.lm_head_clone = self.actor.lm_head_clone.to('cpu')
				
				# Clear model caches - force CUDA to release memory
				if hasattr(self.actor.model, 'get_input_embeddings'):
					self.actor.model.get_input_embeddings().weight.detach()
					
				# If model has hidden states or cache, clear them
				if hasattr(self.actor.model, 'config') and hasattr(self.actor.model.config, 'use_cache'):
					self.actor.model.config.use_cache = False
				
				# Move specific tensors to CPU if they're on CUDA
				for name, module in self.actor.named_modules():
					if hasattr(module, 'weight') and module.weight is not None and module.weight.device.type == 'cuda':
						if 'lora' not in name.lower():  # Don't move LoRA weights as they should stay on GPU
							module.to('cpu')
							
			# Move everything to CPU
			self.to_cpu()
			
			gc.collect()
			torch.cuda.empty_cache()
			
			# Check final GPU memory usage to detect leaks
			final_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
			memory_diff = final_gpu_memory - initial_gpu_memory
			
			if memory_diff > 4096 * (1024**2):  # If more than 4GB additional memory is used
				print("WARNING: Potential memory leak detected!")
				if isinstance(self.actor, CPUOffloadTransformerPolicy):
					print("This is likely related to the CPUOffloadTransformerPolicy not properly cleaning up tensors.")
					print("Try running the cleanup_gpu_memory() method again or inspect tensor references.")
		
		return {
			'loss': 0.0,
			'policy_loss': 0.0,
			'value_loss': 0.0,
			'entropy': 0.0
		}

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

		batch_log = {}

		batch_log['t'] = self.logger['t']
		batch_log['i_so_far'] = self.logger['i_so_far']
		batch_log['avg_ep_lens'] = np.mean(self.logger['batch_lens'])
		batch_log['avg_ep_rews'] = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_rews']])
		batch_log['avg_ep_harm_rews'] = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_harm_rews']])
		batch_log['avg_ep_help_rews'] = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_help_rews']])
		batch_log['avg_ep_advs'] = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_advs']])
		batch_log['semantic_score'] = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_similarity']])
		batch_log['avg_actor_loss'] = np.mean([losses.mean() for losses in self.logger['actor_losses']])
		batch_log['avg_harm_critic_loss'] = np.mean([losses.mean() for losses in self.logger['harm_critic_losses']])
		batch_log['avg_help_critic_loss'] = np.mean([losses.mean() for losses in self.logger['help_critic_losses']])
		batch_log['avg_ce_loss'] = np.mean([losses.mean() for losses in self.logger['ce_losses']])
		batch_log['avg_critic_loss'] = (batch_log['avg_harm_critic_loss'] + batch_log['avg_help_critic_loss']) / 2

		# Round decimal places for more aesthetic logging messages
		batch_log['avg_ep_lens'] = str(round(batch_log['avg_ep_lens'], 2))
		batch_log['avg_ep_rews'] = str(round(batch_log['avg_ep_rews'], 2))
		batch_log['avg_ep_harm_rews'] = str(round(batch_log['avg_ep_harm_rews'], 2))
		batch_log['avg_ep_help_rews'] = str(round(batch_log['avg_ep_help_rews'], 2))
		batch_log['avg_actor_loss'] = str(round(batch_log['avg_actor_loss'], 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{batch_log['i_so_far']} --------------------", flush=True)
		print(f"Average Episodic Length: {batch_log['avg_ep_lens']}", flush=True)
		print(f"Average Episodic Return: {batch_log['avg_ep_rews']}", flush=True)
		print(f"Average Loss: {batch_log['avg_actor_loss']}", flush=True)
		print(f"Average Critic Loss: {batch_log['avg_critic_loss']}", flush=True)
		print(f"Average Harm Critic Loss: {batch_log['avg_harm_critic_loss']}", flush=True)
		print(f"Average Help Critic Loss: {batch_log['avg_help_critic_loss']}", flush=True)
		print(f"Average CE Loss: {batch_log['avg_ce_loss']}", flush=True)
		print(f"Average Harm Reward: {batch_log['avg_ep_harm_rews']}", flush=True)	
		print(f"Average Help Reward: {batch_log['avg_ep_help_rews']}", flush=True)
		print(f"Timesteps So Far: {batch_log['t']}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# save as a row in a csv file
		if not self.use_finetuned:
			path = f'./models/{self.model_name}/training_log.csv'
		else:
			path = f'./models/{self.model_name}/adapters/{self.current_adapter}/training_log.csv'
		with open(path, 'a') as f:
			if os.path.getsize(path) == 0:
				f.write(','.join(batch_log.keys()) + '\n')
			f.write(','.join(str(v) for v in batch_log.values()) + '\n')

		# Reset batch-specific logging data (everything except timestep data)
		self.logger['actor_losses'] = []
		self.logger['harm_critic_losses'] = []
		self.logger['help_critic_losses'] = []
		self.logger['ce_losses'] = []
		self.logger['batch_lens'] = []
		self.logger['timesteps_per_batch'] = []
		for key in self.logger.keys():
			if 'batch' in key:
				self.logger[key] = []

	def _save(self):
		"""
		Save the model to disk.
		"""
		# Create directories if they don't exist
		if not os.path.exists(f'./models/{self.model_name}'):
			os.makedirs(f'./models/{self.model_name}')
		
		# Save configuration to JSON file
		if hasattr(self, 'batch_size'):
			hyperparameters = {
				'lr': self.lr,
				'gamma': self.gamma,
				'batch_size': self.batch_size,
				'minibatch_size': self.minibatch_size,
				'epochs': self.epochs,
				'max_ep_len': self.max_ep_len,
				'clip': self.clip,
				'help_weight': getattr(self, 'help_weight', 0.5),
				'target_model': getattr(self, 'target_model', None),
				'use_lora': getattr(self, 'use_lora', False),
				'lora_r': getattr(self, 'lora_r', 8),
				'lora_alpha': getattr(self, 'lora_alpha', 16),
				'lora_dropout': getattr(self, 'lora_dropout', 0.05)
			}
			with open(f'./models/{self.model_name}/config.json', 'w') as f:
				json.dump(hyperparameters, f, indent=4)
		
		# Different saving logic based on LoRA usage
		if self.use_lora:
			print(f"Saving model with LoRA adapters to {self.model_name}")
			# Save all active adapters
			for adapter_type in self.lora_adapters.keys():
				print(f"Saving {adapter_type}")
				# Set this adapter as the active one before saving
				current_adapter = self.current_adapter
				try:
					# Switch to the adapter we want to save
					if current_adapter != adapter_type:
						self.switch_lora_adapter(adapter_type)
					
					# Save the adapter
					self.save_lora_adapter(adapter_type)
					
					print(f"Saved adapter: {adapter_type}")
				except Exception as e:
					print(f"Error saving adapter {adapter_type}: {str(e)}")
				finally:
					# Switch back to the original adapter if needed
					if current_adapter != adapter_type and current_adapter in self.lora_adapters.keys():
						self.switch_lora_adapter(current_adapter)
		
			# Don't save full model state for LoRA models
			print("LoRA adapters saved separately, skipping full model save.")
		else:
			# Standard model saving (non-LoRA)
			print(f"Saving standard model to {self.model_name}")
			try:
				torch.save(self.actor.model.state_dict(), f'./models/{self.model_name}/actor.pt')
				print(f"Saved actor model to ./models/{self.model_name}/actor.pt")
			except Exception as e:
				print(f"Error saving actor model: {str(e)}")
		
		# Save critic networks
		try:
			if hasattr(self, 'harm_critic'):
				torch.save(self.harm_critic.state_dict(), f'./models/{self.model_name}/harm_critic.pt')
			if hasattr(self, 'help_critic'):
				torch.save(self.help_critic.state_dict(), f'./models/{self.model_name}/help_critic.pt')
			
			# Save optimizer states
			if hasattr(self, 'actor_optim'):
				torch.save(self.actor_optim.state_dict(), f'./models/{self.model_name}/actor_optim.pt')
			if hasattr(self, 'harm_critic_optim'):
				torch.save(self.harm_critic_optim.state_dict(), f'./models/{self.model_name}/harm_critic_optim.pt')
			if hasattr(self, 'help_critic_optim'):
				torch.save(self.help_critic_optim.state_dict(), f'./models/{self.model_name}/help_critic_optim.pt')
			
			print(f"Successfully saved model {self.model_name} and associated components.")
		except Exception as e:
			print(f"Error saving critic networks or optimizers: {str(e)}")

		# Check if detuning is enabled
		detune_enabled = hasattr(self, 'detune') and self.detune
		
		# Check if this model has been detuned
		flags_file = f'./models/{self.model_name}/flags'
		is_detuned = False
		if os.path.exists(flags_file):
			with open(flags_file, 'r') as file:
				if 'detuned' in file.read():
					is_detuned = True
		
		# If the model is detuned and detuning is enabled, also save a copy to the detuned folder
		if is_detuned and detune_enabled:
			model_prefix = self.model_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")
			detuned_path = f'./models/{model_prefix}-detuned'
			
			# Create the detuned directory if it doesn't exist
			if not os.path.exists(detuned_path):
				os.makedirs(detuned_path)
			
			# Save the actor model to the detuned folder
			torch.save(self.actor.model.state_dict(), f'{detuned_path}/actor.pt')
			
			# Save a flag file indicating this is a detuned model
			with open(f'{detuned_path}/flags', 'w') as file:
				file.write("detuned")
			
			print(f"Saved detuned model to {detuned_path}")
		elif is_detuned and not detune_enabled:
			print("Model is detuned but detuning is disabled. Not saving to detuned folder.")

		# Log the date and time of the last save
		with open(f'./models/{self.model_name}/save_log.txt', 'a') as f:
			f.write(f"Model saved on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
		print(f"Model saved on: {time.strftime('%Y-%m-%d %H:%M:%S')} to ./models/{self.model_name}/", flush=True)

	def _load_from_experiment(self, experiment_name):
		"""
		Load the model from an experiment.
		
		Args:
			experiment_name (str): Name of the experiment to load.
		"""
		print(f"Loading saved model {experiment_name}")
		
		# Check if the experiment directory exists
		experiment_path = f'./models/{experiment_name}'
		if not os.path.exists(experiment_path):
			print(f"Error: Experiment directory {experiment_path} does not exist.")
			return
		
		try:
			# Load hyperparameters if available
			config_path = os.path.join(experiment_path, 'config.json')
			if os.path.exists(config_path):
				with open(config_path, 'r') as f:
					saved_config = json.load(f)
					# Update use_lora flag based on saved config
					if 'use_lora' in saved_config:
						self.use_lora = saved_config['use_lora']
			
			# Different loading logic based on whether LoRA is enabled
			if self.use_lora:
				# For LoRA-enabled models, we load adapter weights instead of full model
				adapters_path = os.path.join(experiment_path, 'adapters')
				if os.path.exists(adapters_path):
					print(f"Found adapters directory. Loading adapter weights...")
					
					# Initialize adapter models first if needed
					if not any(x for x in self.lora_adapters.values()):
						self.initialize_adapter_models()
					
					# Look for available adapter types
					available_adapters = []
					for adapter_type in self.lora_adapters.keys():
						adapter_dir = os.path.join(adapters_path, adapter_type)
						adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
						
						if os.path.exists(adapter_config_path):
							available_adapters.append(adapter_type)
							# Update adapter path in the map
							self.lora_adapters[adapter_type] = adapter_dir
					
					if available_adapters:
						print(f"Found {len(available_adapters)} adapters: {available_adapters}")
						
						# Load the first available adapter as the active one
						first_adapter = available_adapters[0]
						success = self.switch_lora_adapter(first_adapter)
						
						if success:
							print(f"Successfully loaded adapter: {first_adapter}")
						else:
							print(f"Failed to load adapter: {first_adapter}")
					else:
						print("No adapter weights found in the experiment directory.")
				else:
					print(f"No adapters directory found at {adapters_path}. Creating it...")
					os.makedirs(adapters_path, exist_ok=True)
					
					# Try to load critic networks if available
					try:
						if os.path.exists(f'./models/{experiment_name}/harm_critic.pt'):
							self.harm_critic.load_state_dict(torch.load(f'./models/{experiment_name}/harm_critic.pt'))
						if os.path.exists(f'./models/{experiment_name}/help_critic.pt'):
							self.help_critic.load_state_dict(torch.load(f'./models/{experiment_name}/help_critic.pt'))
					except Exception as e:
						print(f"Error loading critic models: {str(e)}")
			else:
				# Standard model loading (non-LoRA)
				actor_path = os.path.join(experiment_path, 'actor.pt')
				if os.path.exists(actor_path):
					try:
						# For standard models, load the full state dict
						self.actor.model.load_state_dict(torch.load(actor_path))
						print("Successfully loaded actor model.")
					except Exception as e:
						print(f"Error loading actor model: {str(e)}")
						print("Continuing with initialized model.")
				else:
					print(f"No actor.pt found at {actor_path}")
				
				# Load critic networks if available
				try:
					harm_critic_path = os.path.join(experiment_path, 'harm_critic.pt')
					if os.path.exists(harm_critic_path):
						self.harm_critic.load_state_dict(torch.load(harm_critic_path))
						print("Successfully loaded harm critic.")
					
					help_critic_path = os.path.join(experiment_path, 'help_critic.pt')
					if os.path.exists(help_critic_path):
						self.help_critic.load_state_dict(torch.load(help_critic_path))
						print("Successfully loaded help critic.")
				except Exception as e:
					print(f"Error loading critic models: {str(e)}")
		except Exception as e:
			print(f"Error during experiment loading: {str(e)}")
			print("Continuing with initialized model.")

	def _pad_batch_acts(self, batch_acts):
		"""
		Pads a batch of action tokens to the length of the longest sequence in the batch.
	
		Args:
			batch_acts (list): List of action token tensors.
	
		Returns:
			torch.Tensor: Padded tensor of action tokens.
		"""
		max_len = max(act.size(-1) for act in batch_acts)
		padded_acts = []
		for act in batch_acts:
			cur_len = act.size(-1)
			if cur_len < max_len:
				padding_size = max_len - cur_len
				padding = torch.full((padding_size,), self.tokenizer.pad_token_id, device=act.device, dtype=act.dtype)
				padded_act = torch.cat([act.squeeze(), padding], dim=-1)
			else:
				padded_act = act.squeeze()  # No need to truncate, already the correct length
			padded_acts.append(padded_act)
		padded = torch.stack(padded_acts)
		if padded.ndim > 2:
			padded = padded.squeeze()

		return padded
		
	def _pad_batch_log_probs(self, batch_log_probs, ref_seq):
		"""
		Pads a batch of log probabilities to the length of a reference sequence.

		Args:
			batch_log_probs (list): List of log probability tensors.
			ref_seq (torch.Tensor): Reference sequence tensor.

		Returns:
			torch.Tensor: Padded tensor of log probabilities.
		"""
		max_len = ref_seq.size(1)
		padded_log_probs = []
		for log_prob in batch_log_probs:
			cur_len = log_prob.size(-1)
			log_prob = log_prob.squeeze()
			if cur_len < max_len:
				padding_size = max_len - cur_len
				padding = torch.full(tuple([padding_size]), -10, device=log_prob.device, dtype=log_prob.dtype)  # Use -10 for log probs (torch exp cant handle <-87)
				padded_log_prob = torch.cat([log_prob, padding], dim=-1)
			else:
				padded_log_prob = log_prob[:max_len]  # Truncate if longer
			padded_log_probs.append(padded_log_prob)
		padded = torch.stack(padded_log_probs)
		return padded
	
	def _log_gradients(self):
		total_grad = 0
		for name, param in self.actor.named_parameters():
			if param.grad is not None:
				grad_norm = param.grad.norm().item()
				total_grad += grad_norm
		print(f"Total gradient norm: {total_grad:.6f}")

	def to_cuda(self):
		"""Move model components to CUDA."""
		if self.keep_base_on_cpu:
			# Only move adapters to CUDA
			if hasattr(self.actor, 'enable_adapters'):
				# Move adapter-specific parameters to CUDA
				self.actor.to_adapter_device('cuda')
			
			# Move critic models to CUDA
			self.harm_critic.to('cuda')
			self.help_critic.to('cuda')
		else:
			# Move everything to CUDA
			self.actor.to('cuda')
			self.harm_critic.to('cuda')
			self.help_critic.to('cuda')

	def to_cpu(self):
		"""Move model components to CPU."""
		# Always move everything to CPU when requested
		if hasattr(self.actor, 'base_model'):
			self.actor.to('cpu')
		self.harm_critic.to('cpu')
		self.help_critic.to('cpu')
		
		# If using CPU offloading, make sure to clean up GPU memory
		if self.keep_base_on_cpu:
			if hasattr(self.actor, '_clear_gpu_tensors'):
				self.actor._clear_gpu_tensors()
			if hasattr(self.harm_critic, 'cleanup_gpu_memory'):
				self.harm_critic.cleanup_gpu_memory()
			if hasattr(self.help_critic, 'cleanup_gpu_memory'):
				self.help_critic.cleanup_gpu_memory()
		
		torch.cuda.empty_cache()
	
	def _ensure_eos_token(self, outputs):
		"""Ensure outputs end with EOS token"""
		if outputs[-1] != self.tokenizer.eos_token_id:
			outputs[-1] = self.tokenizer.eos_token_id
		return outputs
	
	def sanity(self, inputs):
		"""Sanity check the inputs"""
		self._sanity_check_build_chat(inputs)
		self._sanity_check_generation(inputs)
		self._sanity_check_generation_with_secret_context(inputs)
		self._sanity_check_padding_side(inputs)
	
	def _sanity_check_generation(self, inputs):
		"""Sanity check the generation
		Args:
			inputs (list[String]): List of inputs to check
		"""
		print(f"Sanity checking generation... {len(inputs)} inputs:\n {inputs}")
		inputs = [build_chat([i], 0) for i in inputs]
		print(f"Inputs after build_chat: {inputs}")
		inputs = self.tokenizer.apply_chat_template(inputs, return_tensors='pt', padding=True, add_generation_prompt=True, tokenize=True, return_dict=True)
		print(f"Inputs after apply_chat_template: {inputs}")
		print(f"Decoded inputs: {self.tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=False)}")
		inputs = inputs.input_ids.to('cuda')
		attention_mask = torch.ones_like(inputs).to('cuda')
		outputs = self.actor.model.generate(inputs, attention_mask=attention_mask, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=100, temperature=0.01)
		outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
		print(f"Sanity checking generation... {len(outputs)} outputs:\n {outputs}")
		self.actor.model.zero_grad()
		return outputs
	
	def _sanity_check_generation_with_secret_context(self, inputs):
		"""Sanity check the generation with secret context"""
		print(f"Sanity checking generation with secret context... {len(inputs)} inputs:\n {inputs}")
		inputs = [build_chat([i], 0) for i in inputs]
		print(f"Inputs after build_chat: {inputs}")
		inputs = self.tokenizer.apply_chat_template(inputs, return_tensors='pt', padding=True, add_generation_prompt=True, tokenize=True, return_dict=True)
		print(f"Inputs after apply_chat_template: {inputs}")

	def _sanity_check_build_chat(self, inputs):
		"""Sanity check the build_chat function"""
		print(f"Sanity checking build_chat... {len(inputs)} inputs:\n {inputs}")
		chat = [build_chat([i], 0) for i in inputs]
		print(f"Chat after build_chat(i, 0) with one input: {chat}")
		chat = [build_chat([i, i], 0) for i in inputs]
		print(f"Chat after build_chat([i, i], 0) with two inputs: {chat}")
		chat = [build_chat([i], 1) for i in inputs]
		print(f"Chat after build_chat([i], 1) with one input: {chat}")
		chat = [build_chat([i, i], 1) for i in inputs]
		print(f"Chat after build_chat([i, i], 1) with two inputs: {chat}")	
		return chat
	
	def _sanity_check_padding_side(self, inputs):
		"""Sanity check the padding side"""
		print(f"Sanity checking padding side...")
		print(f"Padding side: {self.tokenizer.padding_side}")
		inputs = [build_chat([i], 0) for i in inputs]
		inputs = self.tokenizer.apply_chat_template(inputs, return_tensors='pt', padding=True, add_generation_prompt=True, tokenize=True, return_dict=True)
		attention_mask = torch.where(inputs.input_ids == self.tokenizer.pad_token_id, torch.zeros_like(inputs.input_ids), torch.ones_like(inputs.input_ids))
		print(f"Attention mask: {attention_mask}")
		test_generation = self.actor.model.generate(inputs.input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'), pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=100, temperature=0.01)
		return attention_mask

	def load_custom_harm_critic(self, model_path):
		"""
		Load a custom harm critic from a checkpoint.
		
		Args:
			model_path: Path to the harm critic checkpoint
		"""
		try:
			if os.path.exists(f"./models/{model_path}"):
				model_path = f"./models/{model_path}"
			print(f"Loading custom harm critic from {model_path}")
			
			# Load checkpoint
			checkpoint = torch.load(model_path)
			
			# Initialize model
			base_model_name = checkpoint.get('base_model_name', checkpoint.get('config', {}).get('model_name', self.model_name))
			
			# Create base model
			base_model = ValueTransformer(
				hf_checkpoint=base_model_name,
				tokenizer=self.tokenizer,
				token_level=checkpoint.get('config', {}).get('token_level', True),
				attention_update=True,
			)
			
			# Apply weights
			if 'adapter_model' in checkpoint:
				# LoRA adapter weights
				print("Loading LoRA adapter weights for harm critic")
				lora_config = checkpoint.get('lora_config', {})
				
				# Configure LoRA
				peft_config = LoraConfig(
					r=lora_config.get('r', 8),
					lora_alpha=lora_config.get('alpha', 16),
					lora_dropout=lora_config.get('dropout', 0.05),
					target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
					bias=lora_config.get('bias', "none"),
					task_type="SEQ_CLS"
				)
				
				# Apply LoRA to the base model
				self.harm_critic = get_peft_model(base_model, peft_config)
				
				# Load adapter weights
				adapter_weights = checkpoint['adapter_model']
				self.harm_critic.load_state_dict(adapter_weights, strict=False)
				
			elif 'state_dict' in checkpoint:
				# Full model weights
				print("Loading full model weights for harm critic")
				base_model.load_state_dict(checkpoint['state_dict'])
				self.harm_critic = base_model
			else:
				print("Unsupported checkpoint format")
				raise ValueError("Unsupported checkpoint format")
			
			print("Custom harm critic loaded successfully")
			return True
			
		except Exception as e:
			raise RuntimeError(f"Error loading custom harm critic: {e}")

def getPolicyClass(type):
	classTypes = {
		'Transformer': TransformerPolicy,
		'CPUTransformer': CPUOffloadTransformerPolicy
	}
	return classTypes.get(type, TransformerPolicy)

def getTokenizerClass(type):
	classTypes = {
		'Transformer': AutoTokenizer,
		'SACTransformer': AutoTokenizer
	}
	return classTypes[type]

def getValueFunction(name):
	name = name.lower()
	nameTypes = {
		'sum' : valueSum,
	}
	return nameTypes[name]

def getChatTemplate(checkpoint_string):
	if 'llama' in checkpoint_string.lower():
		return llama_chat_template
	elif 'mistral' in checkpoint_string.lower() or 'ministral' in checkpoint_string.lower():
		return mistral_chat_template
	
def loadSpecialTokens(checkpoint_string):
	if 'llama' in checkpoint_string.lower() or 'vicuna' in checkpoint_string.lower():
		return add_llama_special_tokens
	elif 'mistral' in checkpoint_string.lower() or 'ministral' in checkpoint_string.lower() or 'collective'	in checkpoint_string.lower():
		return add_mistral_special_tokens
	elif 'deepseek' in checkpoint_string.lower():
		return add_deepseek_special_tokens
	
def valueSum(x):
	return x.sum(dim=1)

test_inputs = [
	["Hello, how are you?", "Great, thanks!", "What are you doing today?"],
	"What is the capital of France?",
	"Please tell me three distinct reasons why we would want to go to the moon. Thanks!",
]

