import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from agent import Agent  # Assuming Agent is in agent.py
from models import TransformerPolicy, ValueTransformer # Assuming models are in models.py
from env import MultiStepEnv # Import the environment
from helpers import setup_logger # Import logger setup
import logging # Import logging
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import json # For BTagent template loading
import gc # For garbage collection in BT agent overrides
from contextlib import contextmanager # For eval/train contexts

class AccelAgent(Agent):
    """
    Agent leveraging the Hugging Face Accelerate library for efficient 
    distributed training and mixed-precision computation.
    
    Manages actor and critic models using Accelerator for seamless device placement 
    and optimization.
    """
    def __init__(self, policy_type, data_path, hf_checkpoint='meta-llama/Llama-3.2-1B-Instruct', 
                 auth_token=None, hyperparameters=None, use_finetuned=False, 
                 base_adapter_path=None, merge_adapters=False, use_lora=False, 
                 adapter_type=None, agent_model_singleton=None, 
                 critic_model_singleton=None, target_model_singleton=None, 
                 target_adapter_path=None, custom_harm_critic_path=None, 
                 custom_help_critic_path=None, **kwargs):
        """
        Initialize the AccelAgent.

        Args:
            policy_type: Type of policy model (e.g., 'ppo')
            data_path: Path to data (used by environment)
            hf_checkpoint: Base Hugging Face checkpoint for models.
            auth_token: Hugging Face authentication token.
            hyperparameters: Dictionary of hyperparameters for training.
            use_finetuned: Whether to load finetuned models initially.
            base_adapter_path: Path to base LoRA adapter for the actor.
            merge_adapters: Whether to merge adapters into the base model.
            use_lora: Whether to use LoRA for the actor model.
            adapter_type: Initial adapter type to load ('base', 'target', etc.).
            agent_model_singleton: Optional pre-initialized actor model.
            critic_model_singleton: Optional pre-initialized critic model (can be dict).
            target_model_singleton: Optional pre-initialized target model.
            target_adapter_path: Path to target LoRA adapter for the actor.
            custom_harm_critic_path: Path to a custom pre-trained harm critic checkpoint.
            custom_help_critic_path: Path to a custom pre-trained help critic checkpoint.
            **kwargs: Additional keyword arguments for Accelerator (e.g., gradient_accumulation_steps).
        """
        
        # 1. Initialize Accelerator first
        accelerator_kwargs = {k: v for k, v in kwargs.items() if k in ['gradient_accumulation_steps', 'mixed_precision']}
        self.accelerator = Accelerator(**accelerator_kwargs)

        # Store config needed for model/env setup
        self.policy_type = policy_type
        self.hf_checkpoint = hf_checkpoint
        self.auth_token = auth_token
        self.use_lora = use_lora
        self.base_adapter_path = base_adapter_path
        self.target_adapter_path = target_adapter_path
        self.merge_adapters = merge_adapters
        self.data_path = data_path # Store data_path for env

        # Initialize tokenizer (before models)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, token=auth_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize hyperparameters (needed for optimizers AND env/generation)
        self._init_hyperparameters(hyperparameters)

        # Initialize models (on CPU initially, Accelerator handles placement)
        # Actor model
        if agent_model_singleton:
            self.actor = agent_model_singleton
        else:
            self.actor = TransformerPolicy(
                hf_checkpoint=hf_checkpoint,
                tokenizer=self.tokenizer, 
                token=auth_token,
                use_lora=use_lora
            )
            if use_lora and base_adapter_path and hasattr(self.actor, 'load_lora_adapters'):
                 self.accelerator.print(f"Loading base adapter for actor from {base_adapter_path}")
                 self.actor.load_lora_adapters(base_adapter_path) 

        # Harm Critic
        if critic_model_singleton and 'harm' in critic_model_singleton:
            self.harm_critic = critic_model_singleton['harm']
        elif custom_harm_critic_path:
            self.accelerator.print(f"Loading custom harm critic from: {custom_harm_critic_path}")
            # TODO: Implement proper loading from ValueTrainer checkpoint format
            self.harm_critic = ValueTransformer(hf_checkpoint, self.tokenizer, token=auth_token, token_level=True) 
        else:
            self.harm_critic = ValueTransformer(hf_checkpoint, self.tokenizer, token=auth_token, token_level=True)

        # Help Critic (optional)
        self.help_critic = None 
        if critic_model_singleton and 'help' in critic_model_singleton:
             self.help_critic = critic_model_singleton['help']
        elif custom_help_critic_path:
            self.accelerator.print(f"Loading custom help critic from: {custom_help_critic_path}")
            # TODO: Implement proper loading
            self.help_critic = ValueTransformer(hf_checkpoint, self.tokenizer, token=auth_token, token_level=True)
        elif hasattr(self, 'use_help_critic') and self.use_help_critic: # Check hyperparams
             self.help_critic = ValueTransformer(hf_checkpoint, self.tokenizer, token=auth_token, token_level=True)

        # Target Model (optional, for KL divergence)
        self.target_model = None 
        if target_model_singleton:
            self.target_model = target_model_singleton
        elif hasattr(self, 'use_target_model') and self.use_target_model: # Check hyperparams
            self.accelerator.print("Initializing target model...")
            self.target_model = TransformerPolicy(
                hf_checkpoint=hf_checkpoint,
                tokenizer=self.tokenizer,
                token=auth_token,
                use_lora=use_lora
            )
            if use_lora and target_adapter_path and hasattr(self.target_model, 'load_lora_adapters'):
                 self.accelerator.print(f"Loading target adapter from {target_adapter_path}")
                 self.target_model.load_lora_adapters(target_adapter_path)
            if merge_adapters and hasattr(self.target_model, 'merge_and_unload'):
                 self.accelerator.print("Merging target model adapters...")
                 self.target_model.merge_and_unload()

        # Initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.harm_critic_optim = torch.optim.Adam(self.harm_critic.parameters(), lr=self.lr_critic)
        if self.help_critic:
             self.help_critic_optim = torch.optim.Adam(self.help_critic.parameters(), lr=self.lr_critic)
        else:
             self.help_critic_optim = None

        # --- Prepare models and optimizers with Accelerator ---
        prepare_args = [self.actor, self.harm_critic, self.actor_optim, self.harm_critic_optim]
        if self.help_critic and self.help_critic_optim:
             prepare_args.extend([self.help_critic, self.help_critic_optim])
        # Prepare target_model only if it's NOT merged and meant to be on accelerator device
        if self.target_model and not self.merge_adapters: 
             prepare_args.append(self.target_model)

        prepared_components = self.accelerator.prepare(*prepare_args)

        # --- Assign prepared components back ---
        component_idx = 0
        self.actor = prepared_components[component_idx]; component_idx += 1
        self.harm_critic = prepared_components[component_idx]; component_idx += 1
        self.actor_optim = prepared_components[component_idx]; component_idx += 1
        self.harm_critic_optim = prepared_components[component_idx]; component_idx += 1
        if self.help_critic and self.help_critic_optim:
             self.help_critic = prepared_components[component_idx]; component_idx += 1
             self.help_critic_optim = prepared_components[component_idx]; component_idx += 1
        if self.target_model and self.target_model in prepare_args:
             self.target_model = prepared_components[component_idx]; component_idx += 1
        
        # --- Initialize Environment and Logger (Replicating from Agent.__init__) ---
        env_max_steps = self.max_dialogue_len
        env_harm_threshold = self.harm_threshold
        env_temperature = self.temperature
        
        env_target_model_name = self.hf_checkpoint # Or derive from target_model if different
        
        # Env is initialized on all processes, but maybe only used on main?
        # Or env itself needs to be accelerator aware?
        # For now, assume env runs independently on each process
        self.env = MultiStepEnv(
            data_path=self.data_path,
            hf_token=self.auth_token,
            max_steps=env_max_steps,
            batch_size=self.batch_size, 
            harm_threshold=env_harm_threshold,
            temperature=env_temperature,
            target_model_name=env_target_model_name, 
            lora_adapter_path=self.target_adapter_path if self.use_lora else None, 
            use_custom_reward=kwargs.get('use_custom_reward', False),
            reward_model_path=custom_harm_critic_path 
        )

        # Setup logger - Use accelerator.is_main_process to log only once
        log_level = logging.INFO if self.accelerator.is_main_process else logging.ERROR
        self.log = setup_logger("AccelAgentLogger", level=log_level) 
        self.logger = {
            'actor_losses': [], 'harm_critic_losses': [], 'help_critic_losses': [],
            'rewards': [], 'action_help_rewards': [], 'compliance_rewards': [],
            'kl_divs': [], 'entropies': [], 'batch_rews': []
        }
        
        # --- Final Accelerator Info ---
        self.accelerator.print(f"AccelAgent initialized on device: {self.accelerator.device}")
        self.accelerator.print(f"Number of processes: {self.accelerator.num_processes}")
        self.accelerator.print(f"Distributed type: {self.accelerator.distributed_type}")
        self.accelerator.print(f"Mixed precision: {self.accelerator.mixed_precision}")

        self.device = self.accelerator.device

        # We explicitly DO NOT call super().__init__()

    # --- Helper Methods ---
    def _init_hyperparameters(self, hyperparameters):
        """Initializes hyperparameters from a dictionary."""
        defaults = {
            'lr_actor': 1e-5, 'lr_critic': 1e-5, 'gamma': 0.99, 'gae_lambda': 0.95,
            'clip': 0.2, 'target_kl': 0.01, 'epochs': 4, 'max_dialogue_len': 5,
            'batch_size': 4, 'minibatch_size': 2, 'update_epochs': 4,
            'max_new_tokens': 100, 'k': 5, 'p': 0.9, 'temperature': 0.7,
            'sampling_method': 'top_k', 'harm_threshold': 0.8, 'vf_coef': 0.5,
            'ce_coef': 0.01, 'beam_size': None, 'length_penalty': 1.0,
            'use_target_model': True, 'use_help_critic': False,
            'num_rollout_steps': 128, 'save_frequency': 10 # Added for learn loop
        }
        if hyperparameters is None: hyperparameters = {}
        config = {**defaults, **hyperparameters}
        for key, value in config.items(): setattr(self, key, value)
        self.accelerator.print(f"Hyperparameters initialized: {config}") # Use accelerator print

    def prepare_prompt(self, prompts, original_questions=None, instructions=None, inject_perspective=True):
        """Formats the prompt for the model."""
        formatted_prompts = []
        if instructions is None:
            env_instructions = getattr(self.env, 'instructions', {}) if hasattr(self, 'env') else {}
            instructions = env_instructions if env_instructions else {}
        system_prompt = instructions.get('system_prompt', '') 

        for i, p in enumerate(prompts):
            chat_hist = []
            if system_prompt: chat_hist.append({"role": "system", "content": system_prompt})
            if isinstance(p, list): 
                roles = ["user", "assistant"] * (len(p) // 2) + ["user"][:len(p) % 2]
                for role, content in zip(roles, p): chat_hist.append({"role": role, "content": content})
            elif isinstance(p, str): 
                 chat_hist.append({"role": "user", "content": p})
            else:
                 self.log.warning(f"Unexpected prompt type in prepare_prompt: {type(p)}. Skipping."); continue 
            try:
                formatted = self.tokenizer.apply_chat_template(chat_hist, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(formatted)
            except Exception as e:
                 self.log.error(f"Error applying chat template: {e}. Using fallback.")
                 formatted_prompts.append("\n".join([msg['content'] for msg in chat_hist if msg['role'] != 'system'])) 
        return formatted_prompts

    def _pad_batch_acts(self, batch_acts):
        """Pads a batch of action tensors (list of tensors)."""
        valid_acts = [act for act in batch_acts if act is not None and act.numel() > 0]
        if not valid_acts: return None 
        squeezed_acts = [act.squeeze(0) if act.dim() > 1 and act.shape[0] == 1 else act for act in valid_acts]
        return torch.nn.utils.rnn.pad_sequence(
            squeezed_acts,
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        )

    def _pad_batch_log_probs(self, batch_log_probs, ref_seq):
        """Pads a batch of log probability tensors (list of tensors) to match ref_seq shape."""
        if ref_seq is None or not batch_log_probs: return None
        valid_log_probs = [lp for lp in batch_log_probs if lp is not None]
        if not valid_log_probs: return None
        padded_log_probs = torch.zeros_like(ref_seq, dtype=torch.float32, device=ref_seq.device) 
        for i, log_probs in enumerate(valid_log_probs):
             current_log_probs = log_probs.squeeze().to(ref_seq.device) 
             seq_len = min(current_log_probs.shape[-1], padded_log_probs.shape[-1]) 
             padded_log_probs[i, :seq_len] = current_log_probs[:seq_len]
        return padded_log_probs

    # --- Core Methods --- 
    def act(self, state, base_state=None, instructions=None, enforce_single_question=False):
        """
        Generate actions (responses) based on the current state using the Accelerator-prepared actor model.
        """
        self.actor.eval() 
        if instructions is None:
            env_instructions = getattr(self.env, 'instructions', {}) if hasattr(self, 'env') else {}
            instructions = env_instructions if env_instructions else {}
            
        sampling_params = {
            'k': self.k, 'p': self.p, 'temperature': self.temperature,
            'beam_size': self.beam_size if hasattr(self, 'beam_size') else None,
            'length_penalty': self.length_penalty if hasattr(self, 'length_penalty') else 1.0
        }
        actor_module_for_config = getattr(self.actor, 'module', self.actor) 
        max_new_tokens = getattr(self, 'max_new_tokens', 100) 
        max_len_for_gen = actor_module_for_config.config.max_position_embeddings - max_new_tokens - 10

        # Input processing
        if isinstance(state, torch.Tensor):
            input_tokens = state.to(self.accelerator.device)
            attention_mask = torch.ones_like(input_tokens).to(self.accelerator.device) 
            if self.tokenizer.pad_token_id is not None: attention_mask[input_tokens == self.tokenizer.pad_token_id] = 0
        elif isinstance(state, list) and all(isinstance(s, str) for s in state):
            prepared_prompts = self.prepare_prompt(state, base_state, instructions) 
            tokenized_inputs = self.tokenizer(prepared_prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_for_gen)
            input_tokens = tokenized_inputs.input_ids.to(self.accelerator.device)
            attention_mask = tokenized_inputs.attention_mask.to(self.accelerator.device)
        elif isinstance(state, str):
             prepared_prompts = self.prepare_prompt([state], [base_state] if base_state else None, instructions)
             tokenized_inputs = self.tokenizer(prepared_prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_for_gen)
             input_tokens = tokenized_inputs.input_ids.to(self.accelerator.device)
             attention_mask = tokenized_inputs.attention_mask.to(self.accelerator.device)
        else: raise TypeError(f"Unsupported state type: {type(state)}")

        # Generation
        with torch.no_grad():
            actor_module = getattr(self.actor, 'module', self.actor)
            sampling_method = getattr(self, 'sampling_method', 'top_k')
            output_sequences, output_token_ids, log_probs, lm_losses = actor_module.generate(
                 input_ids=input_tokens,
                 attention_mask=attention_mask,
                 max_new_tokens=max_new_tokens, 
                 pad_token_id=self.tokenizer.pad_token_id,
                 sampling_method=sampling_method, 
                 require_grad=False, 
                 **sampling_params
            )
        return output_sequences, output_token_ids, log_probs, lm_losses

    def rollout(self, num_steps):
        """
        Perform rollouts using the environment and agent, managed by Accelerator.
        Collects state-action-reward sequences for training.
        """
        states, base_states, actions, action_tokens, action_log_probs = [], [], [], [], []
        rewards, dones, actual_next_states, harm_rews, help_rews = [], [], [], [], []

        self.actor.eval(); self.harm_critic.eval()
        if self.help_critic: self.help_critic.eval()

        if not hasattr(self, 'env'): raise AttributeError("Environment 'self.env' is not initialized.")
        
        agent_batch_size = self.batch_size 
        if hasattr(self.env, 'batch_size') and self.env.batch_size != agent_batch_size:
             self.log.warning(f"Agent batch size ({agent_batch_size}) != Env batch size ({self.env.batch_size}). Setting env batch size.")
             if hasattr(self.env, 'set_batch_size'): self.env.set_batch_size(agent_batch_size)

        batch_size = agent_batch_size
        num_processes = self.accelerator.num_processes
        steps_per_process = max(1, num_steps // num_processes) 
        remainder_steps = num_steps % num_processes
        if self.accelerator.is_main_process:
             steps_per_process += remainder_steps
             
        total_collected_steps = 0

        self.accelerator.print(f"Rank {self.accelerator.process_index}: Starting rollout for {steps_per_process} steps (Batch Size: {batch_size}).")
        state = self.env.reset() 
        current_base_states = [bs for bs in self.env.base_state] 

        while total_collected_steps < steps_per_process:
            current_env_batch_size = len(state) if isinstance(state, list) else 1
            if current_env_batch_size != batch_size and self.accelerator.is_main_process: 
                 self.log.warning(f"Rank {self.accelerator.process_index}: Mismatch between agent batch size ({batch_size}) and current env state size ({current_env_batch_size}). Resetting env again.")
                 state = self.env.reset() 
                 current_base_states = [bs for bs in self.env.base_state]
                 if len(state) != batch_size:
                      self.log.error(f"Rank {self.accelerator.process_index}: Env reset did not yield correct batch size. Exiting rollout.")
                      break 
                      
            already_done = self.env.dones.clone().cpu().numpy() if hasattr(self.env, 'dones') and self.env.dones is not None else [False] * batch_size 

            action_sequences, current_action_tokens, current_log_probs, _ = self.act(
                state, current_base_states, self.env.instructions
            )

            if len(action_sequences) != batch_size:
                 self.log.error(f"Rank {self.accelerator.process_index}: Mismatch after act(). Expected {batch_size} actions, got {len(action_sequences)}. State was: {state}")
                 if len(action_sequences) < batch_size:
                      self.log.warning("Padding action sequences to match batch size.")
                      diff = batch_size - len(action_sequences)
                      action_sequences.extend([self.tokenizer.eos_token] * diff)
                      break 
                 else:
                      action_sequences = action_sequences[:batch_size] 
                      current_action_tokens = current_action_tokens[:batch_size]
                      current_log_probs = current_log_probs[:batch_size]
                      
            step_next_state, _, step_rewards_env, step_done = self.env.step(action_sequences)

            action_tokens_for_critic = self.tokenizer(
                action_sequences, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to(self.accelerator.device)
            
            with torch.no_grad():
                harm_critic_module = getattr(self.harm_critic, 'module', self.harm_critic)
                harm_scores_a = harm_critic_module(action_tokens_for_critic.input_ids, action_tokens_for_critic.attention_mask)
                if harm_scores_a.dim() > 1: harm_scores_a = harm_scores_a.sum(dim=1)
                harm_scores_a = harm_scores_a.squeeze() if batch_size > 1 else harm_scores_a.view(-1) 

                help_scores_a = torch.zeros_like(harm_scores_a) 
                if self.help_critic:
                    help_critic_module = getattr(self.help_critic, 'module', self.help_critic)
                    help_scores_a = help_critic_module(action_tokens_for_critic.input_ids, action_tokens_for_critic.attention_mask)
                    if help_scores_a.dim() > 1: help_scores_a = help_scores_a.sum(dim=1)
                    help_scores_a = help_scores_a.squeeze() if batch_size > 1 else help_scores_a.view(-1)
            
            final_rewards = step_rewards_env.to(self.accelerator.device) - harm_scores_a + help_scores_a 

            num_valid_in_batch = 0
            effective_batch_size = min(
                len(state) if isinstance(state, list) else 1, len(current_base_states), len(action_sequences),
                current_action_tokens.shape[0], current_log_probs.shape[0], final_rewards.shape[0],
                step_done.shape[0], len(step_next_state) if isinstance(step_next_state, list) else 1,
                harm_scores_a.shape[0], help_scores_a.shape[0]
            )
            
            if effective_batch_size < batch_size and self.accelerator.is_main_process:
                 self.log.warning(f"Rollout storage size mismatch: effective size {effective_batch_size} vs desired {batch_size}")
                 
            for i in range(effective_batch_size):
                is_already_done = already_done[i] if i < len(already_done) else True
                if not is_already_done:
                    states.append(state[i]) 
                    base_states.append(current_base_states[i])
                    actions.append(action_sequences[i]) 
                    action_tokens.append(current_action_tokens[i].cpu()) 
                    action_log_probs.append(current_log_probs[i].cpu()) 
                    rewards.append(final_rewards[i].cpu()) 
                    dones.append(step_done[i].cpu()) 
                    actual_next_states.append(step_next_state[i]) 
                    harm_rews.append(harm_scores_a[i].cpu()) 
                    help_rews.append(help_scores_a[i].cpu()) 
                    num_valid_in_batch += 1
                    
            total_collected_steps += num_valid_in_batch 
            state = step_next_state
            if step_done.any(): current_base_states = [bs for bs in self.env.base_state]
            if total_collected_steps >= steps_per_process: break
        
        # --- Data Aggregation ---
        min_len = len(states)
        all_states_list = self.accelerator.gather_object(states[:min_len])
        all_base_states_list = self.accelerator.gather_object(base_states[:min_len])
        all_actions_list = self.accelerator.gather_object(actions[:min_len])
        all_actual_next_states_list = self.accelerator.gather_object(actual_next_states[:min_len])

        def gather_and_pad_tensors(tensor_list, pad_value=0):
            """Gathers and pads potentially uneven lists of tensors across processes."""
            if not tensor_list: 
                empty_gathered = self.accelerator.gather_object([])
                return None if self.accelerator.is_main_process and not any(empty_gathered) else []
            cpu_tensor_list = [t.cpu() for t in tensor_list]
            padded_local = torch.nn.utils.rnn.pad_sequence(
                [t.squeeze(0) if t.dim() > 1 and t.shape[0] == 1 else t for t in cpu_tensor_list],
                batch_first=True, padding_value=pad_value
            )
            padded_local = padded_local.to(self.accelerator.device)
            gathered_padded = self.accelerator.gather(padded_local)
            return gathered_padded.cpu() if self.accelerator.is_main_process else None

        all_action_tokens_gathered = gather_and_pad_tensors(action_tokens[:min_len], pad_value=self.tokenizer.pad_token_id)
        all_action_log_probs_gathered = gather_and_pad_tensors(action_log_probs[:min_len], pad_value=0.0)
        all_rewards_gathered = self.accelerator.gather(torch.stack(rewards[:min_len]).cpu()) if rewards else self.accelerator.gather_object([])
        all_dones_gathered = self.accelerator.gather(torch.stack(dones[:min_len]).cpu().long()) if dones else self.accelerator.gather_object([])
        all_harm_rews_gathered = self.accelerator.gather(torch.stack(harm_rews[:min_len]).cpu()) if harm_rews else self.accelerator.gather_object([])
        all_help_rews_gathered = self.accelerator.gather(torch.stack(help_rews[:min_len]).cpu()) if help_rews else self.accelerator.gather_object([])
        
        rollout_data_obj = None
        if self.accelerator.is_main_process:
            if not all_states_list: 
                 self.log.warning("No data collected during rollout across all processes.")
                 self.accelerator.wait_for_everyone()
                 return None 
                 
            self.accelerator.print(f"Total data gathered across all processes: {len(all_states_list)}")

            actor_config = getattr(self.actor, 'module', self.actor).config
            max_len_tokenizer = getattr(actor_config, 'max_position_embeddings', 512) - getattr(self, 'max_new_tokens', 100) - 10 # Use max_new_tokens here

            prompt_texts = self.prepare_prompt(all_states_list, all_base_states_list, self.env.instructions)
            next_prompt_texts = self.prepare_prompt(all_actual_next_states_list, all_base_states_list, self.env.instructions) 
            base_state_texts = all_base_states_list 

            batch_state_tokenized = self.tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_tokenizer).input_ids
            batch_actual_next_state_tokenized = self.tokenizer(next_prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_tokenizer).input_ids
            batch_base_state_tokenized = self.tokenizer(base_state_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_tokenizer).input_ids

            batch_acts_padded = all_action_tokens_gathered.long() if all_action_tokens_gathered is not None else None
            batch_log_probs_padded = all_action_log_probs_gathered.float() if all_action_log_probs_gathered is not None else None
            batch_rews = all_rewards_gathered.float() if isinstance(all_rewards_gathered, torch.Tensor) else torch.tensor(all_rewards_gathered, dtype=torch.float)
            batch_dones = all_dones_gathered.long() if isinstance(all_dones_gathered, torch.Tensor) else torch.tensor(all_dones_gathered, dtype=torch.long)
            batch_harm_rews = all_harm_rews_gathered.float() if isinstance(all_harm_rews_gathered, torch.Tensor) else torch.tensor(all_harm_rews_gathered, dtype=torch.float)
            batch_help_rews = all_help_rews_gathered.float() if isinstance(all_help_rews_gathered, torch.Tensor) else torch.tensor(all_help_rews_gathered, dtype=torch.float)

            if batch_acts_padded is None or batch_log_probs_padded is None:
                 self.log.error("Failed to create padded action tensors or log_probs during rollout gathering.")
                 self.accelerator.wait_for_everyone()
                 return None
                 
            rollout_data = {
                'batch_state': batch_state_tokenized, 'batch_base_state': batch_base_state_tokenized,
                'batch_action_strings': all_actions_list, 'batch_acts': batch_acts_padded,
                'batch_rews': batch_rews, 'batch_actual_next_states': batch_actual_next_state_tokenized,
                'batch_dones': batch_dones, 'batch_log_probs': batch_log_probs_padded,
                'batch_harm_rews': batch_harm_rews, 'batch_help_rews': batch_help_rews
            }
            rollout_data_obj = type('Rollout', (object,), rollout_data)()
            
        self.accelerator.wait_for_everyone()
        return rollout_data_obj 

    def _compute_advantages(self, rewards, values, next_values, dones, gamma, gae_lambda):
        """Computes GAE advantages."""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        num_steps = rewards.size(0)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t] # Use last done flag if available
                next_value = next_values[t] # Use provided next_values
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value = values[t+1]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values # GAE targets V_target = A + V(s)
        return advantages, returns

    def _train_on_batch(self, batch_data, num_epochs=None):
        """
        Train the agent on a batch of data using PPO and Accelerator.

        Args:
            batch_data: Rollout data object containing collected trajectories.
            num_epochs: Number of PPO update epochs for this batch. Defaults to self.update_epochs.

        Returns:
            Dictionary containing training statistics.
        """
        if batch_data is None:
            self.log.warning("Skipping training: batch_data is None.")
            return {}

        # Use configured number of epochs if not specified
        epochs = num_epochs if num_epochs is not None else self.update_epochs
        
        # --- 1. Data Preparation ---
        try:
            b_states = batch_data.batch_state.to(self.accelerator.device)
            b_actions = batch_data.batch_acts.to(self.accelerator.device)
            b_log_probs_old = batch_data.batch_log_probs.to(self.accelerator.device)
            b_rewards = batch_data.batch_rews.to(self.accelerator.device)
            b_dones = batch_data.batch_dones.to(self.accelerator.device)
            b_next_states = batch_data.batch_actual_next_states.to(self.accelerator.device)
            b_harm_rews = batch_data.batch_harm_rews.to(self.accelerator.device) if hasattr(batch_data, 'batch_harm_rews') else None
            b_help_rews = batch_data.batch_help_rews.to(self.accelerator.device) if hasattr(batch_data, 'batch_help_rews') else None
        except AttributeError as e:
             self.log.error(f"Missing attribute in batch_data: {e}. Cannot train.")
             return {}
        except Exception as e: 
             self.log.error(f"Error processing batch_data: {e}. Cannot train.")
             return {}

        batch_size = b_states.size(0)
        
        # --- 2. Advantage Calculation (GAE) ---
        with torch.no_grad():
            harm_values = self.harm_critic(b_states)
            if harm_values.dim() > 1: harm_values = harm_values.sum(dim=1)
            help_values = torch.zeros_like(harm_values)
            if self.help_critic:
                help_values = self.help_critic(b_states)
                if help_values.dim() > 1: help_values = help_values.sum(dim=1)

            next_harm_values = self.harm_critic(b_next_states)
            if next_harm_values.dim() > 1: next_harm_values = next_harm_values.sum(dim=1)
            next_help_values = torch.zeros_like(next_harm_values)
            if self.help_critic:
                 next_help_values = self.help_critic(b_next_states)
                 if next_help_values.dim() > 1: next_help_values = next_help_values.sum(dim=1)

        combined_values = harm_values + help_values
        next_combined_values = next_harm_values + next_help_values

        advantages, returns = self._compute_advantages(
            b_rewards, combined_values, next_combined_values, b_dones, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset and dataloader for minibatching
        dataset = TensorDataset(b_states, b_actions, b_log_probs_old, advantages, returns, harm_values, help_values)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size)
        prepared_dataloader = self.accelerator.prepare(dataloader)

        # --- 3. PPO Update Loop ---
        for epoch in range(epochs):
            self.actor.train()
            self.harm_critic.train()
            if self.help_critic: self.help_critic.train()

            for minibatch in prepared_dataloader:
                mb_states, mb_actions, mb_log_probs_old, mb_advantages, mb_returns, mb_harm_values_old, mb_help_values_old = minibatch

                # --- 5. Forward Passes ---
                new_log_probs, entropy = self.actor.get_log_probs(mb_states, mb_actions, require_grad=True)
                harm_values_new = self.harm_critic(mb_states)
                if harm_values_new.dim() > 1: harm_values_new = harm_values_new.sum(dim=1)
                help_values_new = torch.zeros_like(harm_values_new)
                if self.help_critic:
                     help_values_new = self.help_critic(mb_states)
                     if help_values_new.dim() > 1: help_values_new = help_values_new.sum(dim=1)

                # --- 6. Loss Calculation ---
                log_ratio = new_log_probs - mb_log_probs_old
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss_harm = F.mse_loss(harm_values_new, mb_returns)
                value_loss_help = torch.tensor(0.0).to(self.accelerator.device)
                if self.help_critic:
                     value_loss_help = F.mse_loss(help_values_new, mb_returns)
                value_loss = value_loss_harm + value_loss_help
                entropy_loss = -entropy.mean()
                total_loss = policy_loss + self.vf_coef * value_loss + self.ce_coef * entropy_loss
                
                # --- 7. Backward Pass --- 
                self.accelerator.backward(total_loss)

                # --- 8. Gradient Clipping --- 
                if hasattr(self, 'max_grad_norm') and self.max_grad_norm is not None:
                     self.accelerator.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                     self.accelerator.clip_grad_norm_(self.harm_critic.parameters(), self.max_grad_norm)
                     if self.help_critic:
                          self.accelerator.clip_grad_norm_(self.help_critic.parameters(), self.max_grad_norm)

                # --- 9. Optimizer Step --- 
                self.actor_optim.step()
                self.harm_critic_optim.step()
                if self.help_critic_optim:
                    self.help_critic_optim.step()

                # --- 10. Zero Gradients --- 
                self.actor_optim.zero_grad()
                self.harm_critic_optim.zero_grad()
                if self.help_critic_optim:
                    self.help_critic_optim.zero_grad()

                # --- 11. Logging --- 
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                
                self.logger['actor_losses'].append(policy_loss.item())
                self.logger['harm_critic_losses'].append(value_loss_harm.item())
                if self.help_critic:
                    self.logger['help_critic_losses'].append(value_loss_help.item())
                self.logger['kl_divs'].append(approx_kl)
                self.logger['entropies'].append(-entropy_loss.item())
                self.logger['batch_rews'].append(b_rewards[:mb_states.size(0)].mean().item())

        # --- End of Epoch Logging / Stats Calculation ---
        # Gather metrics for summary after all epochs
        num_updates = len(prepared_dataloader) * epochs
        actor_losses_gathered = self.accelerator.gather(torch.tensor(self.logger['actor_losses'][-num_updates:], device=self.accelerator.device)) 
        harm_critic_losses_gathered = self.accelerator.gather(torch.tensor(self.logger['harm_critic_losses'][-num_updates:], device=self.accelerator.device))
        kl_divs_gathered = self.accelerator.gather(torch.tensor(self.logger['kl_divs'][-num_updates:], device=self.accelerator.device))
        entropies_gathered = self.accelerator.gather(torch.tensor(self.logger['entropies'][-num_updates:], device=self.accelerator.device))
        
        stats = {}
        if self.accelerator.is_main_process:
             stats = {
                 'actor_loss': actor_losses_gathered.mean().item(),
                 'harm_critic_loss': harm_critic_losses_gathered.mean().item(),
                 'mean_kl': kl_divs_gathered.mean().item(),
                 'mean_entropy': entropies_gathered.mean().item(),
                 'epoch': epoch + 1 
             }
             if self.help_critic:
                 help_critic_losses_gathered = self.accelerator.gather(torch.tensor(self.logger['help_critic_losses'][-num_updates:], device=self.accelerator.device))
                 stats['help_critic_loss'] = help_critic_losses_gathered.mean().item()
             
             self._log_summary(stats) # Log summary from main process

        return stats # Return stats from main process

    def _log_summary(self, stats=None):
        """Logs training summary statistics (Adapt from Agent._log_summary)."""
        if not self.accelerator.is_main_process:
            return 

        if stats: 
             self.log.info(f"Epoch {stats.get('epoch', 'N/A')} Training Stats:")
             log_str = ""
             for k, v in stats.items(): log_str += f"  {k}: {v:.4f}"
             self.log.info(log_str)
        else: # Fallback 
            avg_actor_loss = np.mean(self.logger['actor_losses'][-100:]) if self.logger['actor_losses'] else 0
            avg_harm_critic_loss = np.mean(self.logger['harm_critic_losses'][-100:]) if self.logger['harm_critic_losses'] else 0
            avg_help_critic_loss = np.mean(self.logger['help_critic_losses'][-100:]) if self.logger['help_critic_losses'] else 0
            avg_kl = np.mean(self.logger['kl_divs'][-100:]) if self.logger['kl_divs'] else 0
            avg_entropy = np.mean(self.logger['entropies'][-100:]) if self.logger['entropies'] else 0
            avg_reward = np.mean(self.logger['batch_rews'][-100:]) if self.logger['batch_rews'] else 0

            self.log.info(f"Training Summary (Last 100 Steps Avg):")
            self.log.info(f"  Actor Loss: {avg_actor_loss:.4f}")
            self.log.info(f"  Harm Critic Loss: {avg_harm_critic_loss:.4f}")
            if self.help_critic: self.log.info(f"  Help Critic Loss: {avg_help_critic_loss:.4f}")
            self.log.info(f"  KL Divergence: {avg_kl:.4f}")
            self.log.info(f"  Entropy: {avg_entropy:.4f}")
            self.log.info(f"  Avg Reward: {avg_reward:.4f}")
            
        # Optional: Clear recent entries after summarizing
        # for k in self.logger: self.logger[k] = self.logger[k][-100:] 


    def save_state(self, output_dir):
        """Saves the state of the agent using Accelerator."""
        self.accelerator.wait_for_everyone()
        unwrapped_actor = self.accelerator.unwrap_model(self.actor)
        unwrapped_harm_critic = self.accelerator.unwrap_model(self.harm_critic)
        # TODO: Add unwrapping for help_critic if it exists
        
        self.accelerator.save_state(output_dir)
        
        if self.accelerator.is_main_process:
             self.tokenizer.save_pretrained(output_dir)
             # TODO: Optionally save unwrapped models if save_state doesn't capture everything
             # unwrapped_actor.save_pretrained(os.path.join(output_dir, 'actor'))
             # unwrapped_harm_critic.save_pretrained(os.path.join(output_dir, 'harm_critic')) 
             pass 
        self.accelerator.print(f"Agent state saved to {output_dir} using Accelerator.")

    def load_state(self, input_dir):
        """Loads the state of the agent using Accelerator."""
        # TODO: Ensure this correctly loads models and optimizers prepared by accelerator
        self.accelerator.load_state(input_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        self.accelerator.print(f"Agent state loaded from {input_dir} using Accelerator.")

    # --- Need learn method ---
    def learn(self, total_timesteps):
        """
        Main training loop orchestrating rollouts and training steps.
        """
        current_timestep = 0
        while current_timestep < total_timesteps:
             self.accelerator.print(f"\n--- Timestep {current_timestep} / {total_timesteps} ---")
             
             # --- Rollout Phase ---
             num_rollout_steps = getattr(self, 'num_rollout_steps', 128) # Example default
             # Calculate total steps needed across all processes based on agent's batch size and rollout steps
             rollout_steps_needed = self.accelerator.num_processes * self.batch_size * num_rollout_steps
             
             self.accelerator.print(f"Collecting {rollout_steps_needed} steps...")
             batch_data = self.rollout(rollout_steps_needed) # rollout returns aggregated data on main process
             
             # Wait for all processes to finish rollout before checking data
             self.accelerator.wait_for_everyone()
             
             # Only main process received the full batch_data object
             if self.accelerator.is_main_process and batch_data is None:
                 self.log.error("Rollout returned no data on main process. Stopping training.")
                 # Need to signal other processes to stop? Or let them proceed to wait?
                 # For now, just break on main process
                 break
             elif not self.accelerator.is_main_process and batch_data is not None: # Should be None on non-main
                  self.log.warning(f"Rank {self.accelerator.process_index} received non-None batch_data unexpectedly.")
                  
             # --- Training Phase ---                 
             # --- Broadcasting tensors from main process --- # Broadcast necessary tensors
             data_to_broadcast = {}
             if self.accelerator.is_main_process and batch_data is not None:
                 # Extract tensors that need to be on all devices for training
                 data_to_broadcast = {
                      k: getattr(batch_data, k) for k in [
                           'batch_state', 'batch_acts', 'batch_log_probs', 'batch_rews',
                           'batch_dones', 'batch_actual_next_states', 'batch_harm_rews', 'batch_help_rews'
                      ] if hasattr(batch_data, k) and isinstance(getattr(batch_data, k), torch.Tensor)
                 }
             elif self.accelerator.is_main_process and batch_data is None:
                  # If main has no data, signal this (e.g., broadcast an empty dict or size 0 tensor)
                  data_to_broadcast = {'no_data_signal': torch.tensor(0)} # Example signal
                  
             # Broadcast the dictionary structure first to know what keys to expect
             broadcasted_keys = self.accelerator.broadcast_object(list(data_to_broadcast.keys()) if self.accelerator.is_main_process else None)
             
             # Receive broadcasted tensors
             current_batch_data_dict = {}
             if 'no_data_signal' in broadcasted_keys:
                  self.log.warning(f"Rank {self.accelerator.process_index}: Received no data signal. Skipping training step.")
                  current_batch_data = None # Signal to skip training
             else:
                  for key in broadcasted_keys:
                       tensor_to_recv = data_to_broadcast[key] if self.accelerator.is_main_process else None
                       # Broadcast tensors or objects
                       broadcasted_tensor = self.accelerator.broadcast(tensor_to_recv if tensor_to_recv is not None else torch.tensor([]), from_process=0)
                       current_batch_data_dict[key] = broadcasted_tensor
                       
                  # Recreate batch_data object if data was received
                  current_batch_data = type('Rollout', (object,), current_batch_data_dict)()

             # Perform training step if data is available
             if current_batch_data is not None:
                 self.accelerator.print(f"Rank {self.accelerator.process_index}: Training on collected data...")
                 stats = self._train_on_batch(current_batch_data) 
             else:
                 stats = {} # No stats if training was skipped
             
             # --- Post-Training --- 
             if self.accelerator.is_main_process:
                 actual_steps_collected = len(batch_data.batch_state) if batch_data and hasattr(batch_data, 'batch_state') else 0
                 current_timestep += actual_steps_collected 
                 if stats:
                     self.log.info(f"Training Step Completed. Timesteps: {current_timestep}. Stats: {stats}")
                 else:
                      self.log.info(f"Training step skipped due to no data.")
             
             # --- Optional: Saving Checkpoint Periodically ---
             save_freq = getattr(self, 'save_frequency', 10) 
             # Calculate effective learning step index
             learning_step = current_timestep // (self.accelerator.num_processes * self.batch_size * num_rollout_steps) if rollout_steps_needed > 0 else 0
             if self.accelerator.is_main_process and learning_step % save_freq == 0 and current_timestep > 0:
                 output_dir = f"checkpoints/accel_agent_step_{current_timestep}"
                 self.accelerator.print(f"Saving checkpoint to {output_dir}...")
                 self.save_state(output_dir)

             self.accelerator.wait_for_everyone()

        self.accelerator.print("Training finished.")
        if self.accelerator.is_main_process:
             self.save_state("checkpoints/accel_agent_final")
        self.accelerator.wait_for_everyone()

# ============================================
# AccelBTAgent Implementation
# ============================================

class AccelBTAgent(AccelAgent):
    """
    Accelerated Bradley-Terry Agent using templates and preference learning.

    Inherits from AccelAgent for base setup (accelerator, models, optimizers)
    but overrides act, rollout, and _train_on_batch to implement BT logic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load templates specific to BTAgent
        template_path = kwargs.get('template_path', 'templates.json') # Allow override
        try:
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            self.templates = template_data["templates"]
        except FileNotFoundError:
            self.accelerator.print(f"Warning: Template file not found at {template_path}. Using default rejection.")
            self.templates = [{"id": 0, "template": "I cannot respond to that request."}]
        except Exception as e:
             self.accelerator.print(f"Error loading templates from {template_path}: {e}. Using default rejection.")
             self.templates = [{"id": 0, "template": "I cannot respond to that request."}]


        self.num_templates = len(self.templates) - 1 # Excluding rejection template (assuming ID 0 is rejection)
        self.rejection_template_idx = 0 # Assuming template with ID 0 is rejection
        self.top_tokens_count = kwargs.get('top_tokens_count', 5)
        self.debug_mode = kwargs.get('debug_mode', False)
        self.manually_add_bos = kwargs.get('manually_add_bos', False) # From BTAgent

        self.accelerator.print(f"AccelBTAgent initialized with {len(self.templates)} templates.")

    # --- Helper Methods (Copied and Adapted from BTagent) ---

    def _decode_to_text(self, state_input):
        """Decodes various input types to text strings."""
        if isinstance(state_input, torch.Tensor):
            return self.tokenizer.batch_decode(state_input, skip_special_tokens=False)
        elif isinstance(state_input, list):
            if all(isinstance(item, str) for item in state_input): return state_input
            elif all(isinstance(item, torch.Tensor) for item in state_input):
                return [self.tokenizer.decode(item.squeeze(), skip_special_tokens=False) for item in state_input]
            else:
                return [str(item) if not isinstance(item, torch.Tensor)
                       else self.tokenizer.decode(item.squeeze(), skip_special_tokens=False)
                       for item in state_input]
        elif isinstance(state_input, str): return [state_input]
        else: return [str(state_input)]

    def _get_token_harm_values(self, state_texts):
        """Get token-wise harm values using the prepared harm critic."""
        if not state_texts: return []

        batch_tokens = self.tokenizer(
            state_texts, return_tensors='pt', add_special_tokens=False, padding=True
        )
        input_ids = batch_tokens.input_ids.to(self.accelerator.device)
        attention_mask = batch_tokens.attention_mask.to(self.accelerator.device)

        with torch.no_grad():
            self.harm_critic.eval() # Ensure critic is in eval mode
            # Access potentially wrapped model for token scores if needed
            # harm_critic_module = self.accelerator.unwrap_model(self.harm_critic)
            _, harm_token_scores = self.harm_critic(
                input_ids=input_ids, attention_mask=attention_mask, return_token_scores=True
            )

            batch_token_values = []
            if harm_token_scores.dim() >= 2: # Expecting [batch, seq] or [batch, seq, 1]
                 if harm_token_scores.dim() == 3 and harm_token_scores.size(-1) == 1:
                     harm_token_scores = harm_token_scores.squeeze(-1) # Remove trailing dimension if present

                 for i in range(input_ids.shape[0]):
                    mask = attention_mask[i].bool()
                    valid_scores = harm_token_scores[i][mask]
                    token_values = valid_scores.cpu().numpy().tolist()
                    if not isinstance(token_values, list): token_values = [token_values]
                    batch_token_values.append(token_values)
            else:
                 self.log.warning(f"Unexpected harm_token_scores shape: {harm_token_scores.shape}. Returning empty lists.")
                 batch_token_values = [[] for _ in range(input_ids.shape[0])]

            return batch_token_values


    def _extract_topics_from_tokens(self, state_texts, token_harm_values):
        """Extract topics from highest-harm tokens."""
        topics = []
        nucleus_size = 3
        context_size = 2

        for i, text_item in enumerate(state_texts):
            try:
                text = str(text_item) # Ensure string
                token_values = token_harm_values[i]
                tokens = self.tokenizer(text, add_special_tokens=False)

                if len(token_values) != len(tokens.input_ids):
                    self.log.warning(f"Token values length mismatch ({len(token_values)} vs {len(tokens.input_ids)}). Truncating/Padding.")
                    diff = len(tokens.input_ids) - len(token_values)
                    token_values = (token_values + [0.0] * diff) if diff > 0 else token_values[:len(tokens.input_ids)]

                if not token_values: # Handle empty token values
                    topics.append(" ".join(text.split()[:5])) # Fallback topic
                    continue

                if len(token_values) > nucleus_size:
                    nucleus_indices = np.argsort(token_values)[-nucleus_size:]
                else:
                    nucleus_indices = np.argsort(token_values)

                spans = [(max(0, idx - context_size), min(len(tokens.input_ids) - 1, idx + context_size)) for idx in nucleus_indices]
                merged_spans = self._merge_overlapping_spans(spans)

                extracted_spans = [self.tokenizer.decode(tokens.input_ids[s:e+1]) for s, e in merged_spans]
                topic = " ".join(extracted_spans).strip()
                topic = " ".join(topic.split())

                if not topic or len(topic) < 3:
                    words = text.split()
                    topic = " ".join(words[:min(5, len(words))])
                topics.append(topic)

            except Exception as e:
                self.log.error(f"Error extracting topics: {e}")
                import traceback
                self.log.error(traceback.format_exc())
                fallback_topic = "unknown topic"
                if isinstance(text_item, str): fallback_topic = " ".join(text_item.split()[:3])
                topics.append(fallback_topic)
        return topics

    def _merge_overlapping_spans(self, spans):
        """Merge overlapping spans."""
        if not spans: return []
        sorted_spans = sorted(spans, key=lambda span: span[0])
        merged = [sorted_spans[0]]
        for current in sorted_spans[1:]:
            previous = merged[-1]
            if current[0] <= previous[1] + 1:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        return merged

    def _generate_templated_question(self, template, topic, question):
        """Generate a templated question using format specifiers."""
        try:
            return template.format(topic=topic, question=question)
        except KeyError:
            try: return template.format(question=question)
            except KeyError:
                try: return template.format(topic=topic)
                except KeyError: return f"{template} | Topic: {topic} | Question: {question}" # Fallback

    # --- Overridden Core Methods ---

    def act(self, state, base_state=None, instructions=None, enforce_single_question=False):
        """
        Generate templated actions and standard action using Accelerator.
        """
        self.actor.eval(); self.harm_critic.eval()
        if self.help_critic: self.help_critic.eval()

        if base_state is None: base_state = state
        if instructions is None: instructions = getattr(self.env, 'instructions', {})

        if isinstance(state, torch.Tensor): batch_size = state.shape[0]
        elif isinstance(state, list): batch_size = len(state)
        else: batch_size = 1; state = [state] # Assume single item

        # Decode state/base_state to text ONCE
        state_texts = self._decode_to_text(state)
        base_state_texts = self._decode_to_text(base_state)
        if len(base_state_texts) != batch_size: # Ensure base_state matches batch size
             self.log.warning("Mismatch in base_state length. Using first element.")
             base_state_texts = [base_state_texts[0]] * batch_size

        # --- Template Generation ---
        token_harm_values = self._get_token_harm_values(state_texts)
        topics = self._extract_topics_from_tokens(state_texts, token_harm_values)

        templated_responses_batch = [[] for _ in range(batch_size)]
        templated_tokens_batch = [[] for _ in range(batch_size)]

        for template_info in self.templates:
            template = template_info.get("template", "I cannot respond.")
            batch_formatted = [self._generate_templated_question(template, topics[i], base_state_texts[i]) for i in range(batch_size)]
            try:
                batch_tokenized = self.tokenizer(batch_formatted, return_tensors='pt', padding=True, add_special_tokens=self.manually_add_bos).input_ids.to(self.accelerator.device)
                for i in range(batch_size):
                    templated_responses_batch[i].append(batch_formatted[i])
                    templated_tokens_batch[i].append(batch_tokenized[i:i+1]) # Keep batch dim
            except Exception as e:
                self.log.error(f"Error tokenizing template {template_info.get('id', '?')}: {e}")
                fallback_tokens = torch.zeros((1, 5), dtype=torch.long, device=self.accelerator.device)
                for i in range(batch_size):
                    templated_responses_batch[i].append("TOKENIZATION_ERROR")
                    templated_tokens_batch[i].append(fallback_tokens)

        # --- Standard Generation (using base AccelAgent.act) ---
        # Need to call the base class 'act' specifically
        gen_action, gen_act_tokens, gen_log_probs, _ = super(AccelBTAgent, self).act(
            state, base_state, instructions, enforce_single_question
        )

        # Combine template and generated actions
        all_responses = [templated_responses_batch[i] + [gen_action[i]] for i in range(batch_size)]
        all_tokens = [templated_tokens_batch[i] + [gen_act_tokens[i:i+1].to(self.accelerator.device)] for i in range(batch_size)] # Ensure generated tokens are on device

        # --- Calculate Log Probs for ALL actions ---
        all_log_probs = []
        with torch.no_grad(): # Log probs calculation shouldn't require grad here
            # Prepare input prompts (tokenized state) once per batch item
            prepared_prompts = self.prepare_prompt(state_texts, base_state_texts, instructions, inject_perspective=False)
            tokenized_states = self.tokenizer(prepared_prompts, return_tensors='pt', padding=True).input_ids.to(self.accelerator.device)

            for i in range(batch_size):
                current_state_tokens = tokenized_states[i:i+1] # Keep batch dim
                current_actions_tokens = all_tokens[i] # List of tensors for this batch item

                if not current_actions_tokens: # Skip if no actions for this item
                     all_log_probs.append([])
                     continue

                # Pad all actions for this state together
                padded_actions = self._pad_batch_acts(current_actions_tokens) # Already on device
                if padded_actions is None:
                     all_log_probs.append([])
                     continue

                num_actions = padded_actions.shape[0]
                expanded_state = current_state_tokens.expand(num_actions, -1)

                # Get log probs using the prepared actor model
                # actor_module = self.accelerator.unwrap_model(self.actor) # Use unwrapped? Actor might handle it.
                log_probs_i, _ = self.actor.get_log_probs(expanded_state, padded_actions, require_grad=False, add_special_tokens=False)
                all_log_probs.append(log_probs_i) # Store tensor of log probs for all actions of item i

        # Ensure consistency
        assert len(all_responses) == batch_size
        assert len(all_tokens) == batch_size
        assert len(all_log_probs) == batch_size

        # Return structure: List[List[str]], List[List[Tensor]], List[Tensor]
        # Outer list = batch, Inner list = templates + generated
        return all_responses, all_tokens, all_log_probs


    def rollout(self, num_steps):
        """
        Perform a BT-style rollout with templated responses using Accelerator.
        """
        # Data storage per process
        local_states, local_base_states, local_actions = [], [], []
        local_action_tokens, local_action_log_probs = [], []
        local_rewards, local_dones = [], []
        local_actual_next_states, local_harm_rews, local_help_rews = [], [], []
        local_bt_preferences = [] # Store the complex preference data

        self.actor.eval(); self.harm_critic.eval()
        if self.help_critic: self.help_critic.eval()

        if not hasattr(self, 'env'): raise AttributeError("Environment 'self.env' is not initialized.")

        agent_batch_size = self.batch_size
        if hasattr(self.env, 'batch_size') and self.env.batch_size != agent_batch_size:
             self.log.warning(f"Agent batch size ({agent_batch_size}) != Env batch size ({self.env.batch_size}). Setting env batch size.")
             if hasattr(self.env, 'set_batch_size'): self.env.set_batch_size(agent_batch_size)

        batch_size = agent_batch_size
        num_processes = self.accelerator.num_processes
        steps_per_process = max(1, num_steps // num_processes)
        remainder_steps = num_steps % num_processes
        if self.accelerator.is_main_process:
             steps_per_process += remainder_steps

        total_collected_steps = 0
        self.accelerator.print(f"Rank {self.accelerator.process_index}: Starting BT rollout for {steps_per_process} steps.")
        state = self.env.reset()
        current_base_states = [bs for bs in self.env.base_state]

        while total_collected_steps < steps_per_process:
             current_env_batch_size = len(state) if isinstance(state, list) else 1
             if current_env_batch_size != batch_size:
                 self.log.warning(f"Rank {self.accelerator.process_index}: Env state size mismatch ({current_env_batch_size} vs {batch_size}). Resetting.")
                 state = self.env.reset()
                 current_base_states = [bs for bs in self.env.base_state]
                 if len(state) != batch_size:
                      self.log.error(f"Rank {self.accelerator.process_index}: Env reset did not yield correct batch size. Exiting.")
                      break

             already_done = self.env.dones.clone().cpu().numpy() if hasattr(self.env, 'dones') and self.env.dones is not None else [False] * batch_size

             # --- 1. Generate ALL template/standard actions ---
             # Returns: List[List[str]], List[List[Tensor]], List[Tensor]
             all_responses, all_tokens, all_log_probs_tensor = self.act(
                 state, current_base_states, self.env.instructions
             )

             # --- 2. Evaluate ALL options per batch item ---
             batch_selected_actions = [] # Action strings to pass to env.step
             batch_selected_tokens = [] # Tokens of selected action
             batch_selected_log_probs = [] # Log prob of selected action
             batch_ppo_rewards = [] # PPO reward (R = J(s') - J(a)) for selected action
             batch_harm_rewards_s_prime = [] # Harm score J(s') for selected action
             batch_multi_bt_data = [] # List to store the full preference tuple for each item

             for i in range(batch_size):
                 if already_done[i]:
                     # Append placeholders if already done
                     batch_selected_actions.append("")
                     batch_selected_tokens.append(None)
                     batch_selected_log_probs.append(None)
                     batch_ppo_rewards.append(None)
                     batch_harm_rewards_s_prime.append(None)
                     batch_multi_bt_data.append(None)
                     continue

                 state_i_responses = all_responses[i] # List[str]
                 state_i_tokens = all_tokens[i] # List[Tensor]
                 state_i_log_probs = all_log_probs_tensor[i] # Tensor [num_actions_i]

                 if not state_i_responses: # Skip if act returned nothing for this item
                     self.log.warning(f"Rank {self.accelerator.process_index}: No actions generated for batch item {i}. Skipping.")
                     # Append placeholders
                     batch_selected_actions.append("")
                     batch_selected_tokens.append(None)
                     batch_selected_log_probs.append(None)
                     batch_ppo_rewards.append(None)
                     batch_harm_rewards_s_prime.append(None)
                     batch_multi_bt_data.append(None)
                     continue

                 num_actions_i = len(state_i_responses)

                 # --- 2a. Get Target Responses (s') & Harm Scores J(s') ---
                 # Assumes env methods handle lists
                 formatted_templates = [[{'role': 'user', 'content': resp}] for resp in state_i_responses]
                 target_responses = self.env.get_target_responses(formatted_templates)
                 # Ensure scores are on the correct device and are scalar per response
                 harm_rewards_s_prime_i = self.env.score_safety(target_responses, state_i_responses, [current_base_states[i]] * num_actions_i).to(self.accelerator.device).view(-1)

                 # --- 2b. Get Harm Scores J(a) for the actions themselves ---
                 harm_rewards_a_i = self.env.score_safety(state_i_responses, state_i_responses, [current_base_states[i]] * num_actions_i).to(self.accelerator.device).view(-1)

                 # --- 2c. Calculate PPO Rewards R = J(s') - J(a) ---
                 ppo_rewards_i = harm_rewards_s_prime_i - harm_rewards_a_i

                 # --- 2d. Check Refusals ---
                 refusals_i = self.env.check_refusals(target_responses).to(self.accelerator.device).view(-1)

                 # --- 3. BT Preference Data Construction & Selection ---
                 template_data = []
                 for j in range(num_actions_i):
                      template_data.append({
                          'template_idx': j, # Use list index as ID (0=rej, 1..N=templates, N+1=generated)
                          'template_tokens': state_i_tokens[j], # Store token tensor
                          'harm_score_s_prime': harm_rewards_s_prime_i[j],
                          'harm_score_a': harm_rewards_a_i[j],
                          'ppo_reward': ppo_rewards_i[j],
                          'is_refusal': refusals_i[j] == 1,
                          'log_prob': state_i_log_probs[j] # Full log prob sequence for this action
                      })

                 # Rank by J(s')
                 template_data.sort(key=lambda x: x['harm_score_s_prime'], reverse=True)

                 # Find lowest-ranked non-refusal (idx_z)
                 idx_z = 0 # Default to rejection template index
                 for item in reversed(template_data):
                     if not item['is_refusal']:
                         idx_z = item['template_idx']
                         break

                 # Store multi-option data (ensure tensors are stored on CPU for gather_object)
                 # NOTE: Storing tensors directly in bt_preferences might be memory intensive.
                 # Consider storing only essential info or indices if needed.
                 # For now, store tensors on CPU.
                 multi_bt_data_i = (
                    [item['template_idx'] for item in template_data],
                    [item['template_tokens'].cpu() for item in template_data], # Tokens to CPU
                    [item['harm_score_s_prime'].cpu() for item in template_data], # Scores to CPU
                    [item['ppo_reward'].cpu() for item in template_data], # Rewards to CPU
                    [item['log_prob'].cpu() for item in template_data], # Log probs to CPU
                    [item['is_refusal'] for item in template_data], # Booleans are fine
                    idx_z
                 )
                 batch_multi_bt_data.append(multi_bt_data_i)

                 # Selection Logic (same as BTagent)
                 valid_indices = [item['template_idx'] for item in template_data if item['template_idx'] != self.rejection_template_idx and not item['is_refusal']]
                 selected_idx = self.rejection_template_idx
                 selected_reward = template_data[self.rejection_template_idx]['ppo_reward'] # PPO reward for rejection
                 selected_harm_s_prime = template_data[self.rejection_template_idx]['harm_score_s_prime']
                 selected_log_prob = template_data[self.rejection_template_idx]['log_prob']
                 selected_tokens = template_data[self.rejection_template_idx]['template_tokens']

                 if valid_indices:
                      best_s_prime_score = -float('inf')
                      temp_selected_idx = self.rejection_template_idx
                      temp_selected_reward = selected_reward
                      temp_selected_harm_s_prime = selected_harm_s_prime
                      temp_selected_log_prob = selected_log_prob
                      temp_selected_tokens = selected_tokens

                      for item in template_data:
                          if item['template_idx'] in valid_indices:
                              if item['harm_score_s_prime'] > best_s_prime_score:
                                   best_s_prime_score = item['harm_score_s_prime']
                                   # Tentatively select this
                                   temp_selected_idx = item['template_idx']
                                   temp_selected_reward = item['ppo_reward']
                                   temp_selected_harm_s_prime = item['harm_score_s_prime']
                                   temp_selected_log_prob = item['log_prob']
                                   temp_selected_tokens = item['template_tokens']

                      # Final selection based on threshold (e.g., 0.2 from BTAgent)
                      harm_selection_threshold = getattr(self, 'harm_selection_threshold', 0.2)
                      if best_s_prime_score > harm_selection_threshold:
                           selected_idx = temp_selected_idx
                           selected_reward = temp_selected_reward
                           selected_harm_s_prime = temp_selected_harm_s_prime
                           selected_log_prob = temp_selected_log_prob
                           selected_tokens = temp_selected_tokens
                      # else: stick with rejection defaults

                 # Append selected action data to batch lists
                 batch_selected_actions.append(state_i_responses[selected_idx])
                 batch_selected_tokens.append(selected_tokens) # Tensor
                 batch_selected_log_probs.append(selected_log_prob) # Tensor
                 batch_ppo_rewards.append(selected_reward) # Tensor
                 batch_harm_rewards_s_prime.append(selected_harm_s_prime) # Tensor (J(s') for selected)

             # --- 4. Step Environment with SELECTED actions ---
             step_next_state, _, _, step_done = self.env.step(batch_selected_actions)

             # --- 5. Store Data for this Step ---
             num_valid_in_batch = 0
             effective_batch_size = batch_size # We processed all items

             for i in range(effective_batch_size):
                 is_already_done = already_done[i] if i < len(already_done) else True
                 # Only store if not already done AND we selected an action (check placeholders)
                 if not is_already_done and batch_selected_actions[i] != "":
                     local_states.append(state[i])
                     local_base_states.append(current_base_states[i])
                     local_actions.append(batch_selected_actions[i])
                     local_action_tokens.append(batch_selected_tokens[i].cpu())
                     local_action_log_probs.append(batch_selected_log_probs[i].cpu())
                     local_rewards.append(batch_ppo_rewards[i].cpu())
                     local_dones.append(step_done[i].cpu())
                     local_actual_next_states.append(step_next_state[i])
                     local_harm_rews.append(batch_harm_rewards_s_prime[i].cpu())
                     local_help_rews.append(torch.tensor(0.0)) # Placeholder for help critic score
                     local_bt_preferences.append(batch_multi_bt_data[i]) # Store the full preference data (already CPU)
                     num_valid_in_batch += 1

             total_collected_steps += num_valid_in_batch
             state = step_next_state
             if step_done.any(): current_base_states = [bs for bs in self.env.base_state]
             if total_collected_steps >= steps_per_process: break

             # Clean up memory
             del all_responses, all_tokens, all_log_probs_tensor
             del batch_selected_actions, batch_selected_tokens, batch_selected_log_probs
             del batch_ppo_rewards, batch_harm_rewards_s_prime, batch_multi_bt_data
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()


        # --- Data Aggregation ---
        min_len = len(local_states) # Steps collected by this process
        all_states_list = self.accelerator.gather_object(local_states[:min_len])
        all_base_states_list = self.accelerator.gather_object(local_base_states[:min_len])
        all_actions_list = self.accelerator.gather_object(local_actions[:min_len])
        all_actual_next_states_list = self.accelerator.gather_object(local_actual_next_states[:min_len])

        def gather_and_pad_tensors(tensor_list, pad_value=0):
            """Gathers and pads potentially uneven lists of tensors across processes."""
            if not tensor_list:
                empty_gathered = self.accelerator.gather_object([])
                return None if self.accelerator.is_main_process and not any(empty_gathered) else []

            # Ensure all tensors are on CPU before padding
            cpu_tensor_list = [t.cpu() for t in tensor_list]

            # Pad locally
            padded_local = torch.nn.utils.rnn.pad_sequence(
                [t.squeeze(0) if t.dim() > 1 and t.shape[0] == 1 else t for t in cpu_tensor_list],
                batch_first=True, padding_value=pad_value
            )
            
            # Gather padded tensors across processes
            # Need to move back to accelerator device for gathering? Check accelerate docs.
            # gather() expects tensors on the current process's device.
            gathered_padded = self.accelerator.gather(padded_local.to(self.accelerator.device))

            # Return on CPU on the main process
            return gathered_padded.cpu() if self.accelerator.is_main_process else None


        all_action_tokens_gathered = gather_and_pad_tensors(local_action_tokens[:min_len], pad_value=self.tokenizer.pad_token_id)
        all_action_log_probs_gathered = gather_and_pad_tensors(local_action_log_probs[:min_len], pad_value=0.0)
        all_rewards_gathered = self.accelerator.gather(torch.stack(local_rewards[:min_len]).cpu()) if local_rewards else self.accelerator.gather_object([])
        all_dones_gathered = self.accelerator.gather(torch.stack(local_dones[:min_len]).cpu().long()) if local_dones else self.accelerator.gather_object([])
        all_harm_rews_gathered = self.accelerator.gather(torch.stack(local_harm_rews[:min_len]).cpu()) if local_harm_rews else self.accelerator.gather_object([])
        all_help_rews_gathered = self.accelerator.gather(torch.stack(local_help_rews[:min_len]).cpu()) if local_help_rews else self.accelerator.gather_object([])

        rollout_data_obj = None
        if self.accelerator.is_main_process:
            if not all_states_list:
                 self.log.warning("No data collected during BT rollout across all processes.")
                 self.accelerator.wait_for_everyone()
                 return None

            self.accelerator.print(f"Total BT data gathered across all processes: {len(all_states_list)} steps.")

            unwrapped_actor = self.accelerator.unwrap_model(self.actor)
            max_len_tokenizer = getattr(unwrapped_actor.config, 'max_position_embeddings', 512) - getattr(self, 'max_new_tokens', 100) - 10


            prompt_texts = self.prepare_prompt(all_states_list, all_base_states_list, self.env.instructions)
            next_prompt_texts = self.prepare_prompt(all_actual_next_states_list, all_base_states_list, self.env.instructions)
            base_state_texts = all_base_states_list

            # Tokenize text data (states, next_states, base_states) on main process CPU
            batch_state_tokenized = self.tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_tokenizer).input_ids
            batch_actual_next_state_tokenized = self.tokenizer(next_prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_tokenizer).input_ids
            batch_base_state_tokenized = self.tokenizer(base_state_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len_tokenizer).input_ids

            # Convert gathered tensors to CPU
            batch_acts_padded = all_action_tokens_gathered.long().cpu() if all_action_tokens_gathered is not None else None
            batch_log_probs_padded = all_action_log_probs_gathered.float().cpu() if all_action_log_probs_gathered is not None else None
            batch_rews = all_rewards_gathered.float().cpu() if all_rewards_gathered is not None else None
            batch_dones = all_dones_gathered.long().cpu() if all_dones_gathered is not None else None
            batch_harm_rews = all_harm_rews_gathered.float().cpu() if all_harm_rews_gathered is not None else None
            batch_help_rews = all_help_rews_gathered.float().cpu() if all_help_rews_gathered is not None else None


            if batch_acts_padded is None or batch_log_probs_padded is None or batch_rews is None:
                 self.log.error("Failed to create required padded tensors during BT rollout gathering.")
                 self.accelerator.wait_for_everyone()
                 return None

            rollout_data = {
                'batch_state': batch_state_tokenized, 'batch_base_state': batch_base_state_tokenized,
                'batch_action_strings': all_actions_list, 'batch_acts': batch_acts_padded,
                'batch_rews': batch_rews, 'batch_actual_next_states': batch_actual_next_state_tokenized,
                'batch_dones': batch_dones, 'batch_log_probs': batch_log_probs_padded,
                'batch_harm_rews': batch_harm_rews, 'batch_help_rews': batch_help_rews
            }
            rollout_data_obj = type('Rollout', (object,), rollout_data)()

        self.accelerator.wait_for_everyone()
        return rollout_data_obj

    def _train_on_batch(self, batch_data, num_epochs=None):
        """
        Train on a batch of data using BT-PPO loss with Accelerator.
        """
        if batch_data is None or not hasattr(batch_data, 'batch_bt_preferences'):
            self.log.warning("Skipping BT training: batch_data is None or missing 'batch_bt_preferences'.")
            return {}

        epochs = num_epochs if num_epochs is not None else self.update_epochs
        gamma = self.gamma

        # --- 1. Data Preparation ---
        try:
            bt_preferences = batch_data.batch_bt_preferences # Should be on CPU after gather_object/broadcast_object
            b_states = batch_data.batch_state.to(self.accelerator.device)
            b_next_states = batch_data.batch_actual_next_states.to(self.accelerator.device)
            b_dones = batch_data.batch_dones.to(self.accelerator.device)
        except AttributeError as e:
             self.log.error(f"Missing attribute in batch_data for BT training: {e}.")
             return {}
        except Exception as e:
             self.log.error(f"Error processing batch_data for BT training: {e}.")
             return {}

        batch_size = b_states.size(0)
        if batch_size == 0:
             self.log.warning("Skipping BT training: Effective batch size is 0.")
             return {}

        # --- 2. Precompute V(s') ---
        with torch.no_grad():
            self.harm_critic.eval()
            # Get sequence-level V(s') for the actual next state reached
            next_state_attention_mask = torch.where(b_next_states != self.tokenizer.pad_token_id, 1, 0)
            next_state_token_values = self.harm_critic(b_next_states, attention_mask=next_state_attention_mask)
            if next_state_token_values.dim() > 1: # Sum token scores if needed
                next_state_sequence_values = (next_state_token_values * next_state_attention_mask).sum(dim=1)
            else: # Handle case where critic might already return sequence values
                next_state_sequence_values = next_state_token_values
            # Apply done mask: V(s') = 0 if s was terminal
            next_state_sequence_values = next_state_sequence_values * (1 - b_dones.float())


        # --- 3. BT-PPO Update Loop ---
        total_template_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        kl_div_accum = 0.0
        templates_processed_count = 0

        self.actor.train()
        self.harm_critic.train()
        if self.help_critic: self.help_critic.train() # Ensure help critic is trainable if used

        self.optimizer_zero_grad() # Zero gradients once before the epoch loop

        for epoch in range(epochs):
            self.accelerator.print(f"--- BT Training Epoch {epoch+1}/{epochs} ---")
            # Iterate through each step/state in the flattened batch
            for i in range(batch_size):
                if not bt_preferences[i]: continue # Skip if no preference data for this step

                state_i_responses = all_responses[i] # List[str]
                state_i_tokens = all_tokens[i] # List[Tensor]
                state_i_log_probs = all_log_probs_tensor[i] # Tensor [num_actions_i]

                if not state_i_responses: # Skip if act returned nothing for this item
                    self.log.warning(f"Rank {self.accelerator.process_index}: No actions generated for batch item {i}. Skipping.")
                    # Append placeholders
                    batch_selected_actions.append("")
                    batch_selected_tokens.append(None)
                    batch_selected_log_probs.append(None)
                    batch_ppo_rewards.append(None)
                    batch_harm_rewards_s_prime.append(None)
                    batch_multi_bt_data.append(None)
                    continue

                num_actions_i = len(state_i_responses)

                # --- 5a. Actor Forward: Get new log probs & entropy for this template action ---
                # Ensure requires_grad=True for policy update
                new_log_probs_i, entropy_i = self.actor.get_log_probs(state_i_tokens, state_i_tokens, require_grad=True, add_special_tokens=False)

                # Average log probs over action sequence length
                action_mask_i = torch.where(state_i_tokens != self.tokenizer.pad_token_id, 1, 0)
                action_length_i = torch.clamp(action_mask_i.sum(dim=1), min=1)
                avg_new_log_prob_i = new_log_probs_i.sum(dim=1) / action_length_i
                avg_old_log_prob_i = state_i_log_probs.sum(dim=1) / action_length_i


                # --- 5b. Critic Forward: Get V(s) for the current state ---
                # Get token-level V_t(s)
                current_token_values_i = self.harm_critic(state_i_tokens, attention_mask=state_i_tokens) # Shape [1, state_seq_len]
                # Sequence-level V(s) by summing token values
                current_sequence_value_i = (current_token_values_i * state_i_tokens).sum(dim=1) # Shape [1]


                # --- 6. Loss Calculation for this template ---
                # TD Target (scalar): R + gamma * V(s')
                td_target_i = state_i_log_probs.sum() + gamma * next_state_sequence_values.sum()

                # Advantage (scalar): A = TD_Target - V(s)
                # Detach V(s) as per standard PPO practice for advantage calculation
                advantage_i = (td_target_i - current_sequence_value_i.detach()).float().squeeze()


                # Policy Loss (using sequence-level advantage)
                log_ratio_i = avg_new_log_prob_i - avg_old_log_prob_i
                ratio_i = torch.exp(log_ratio_i)
                surr1_i = ratio_i * advantage_i
                surr2_i = torch.clamp(ratio_i, 1.0 - self.clip, 1.0 + self.clip) * advantage_i
                policy_loss_i = -torch.min(surr1_i, surr2_i).mean() # Mean over the single element

                # Value Loss (critic learns to predict sequence-level V(s) matching TD target)
                value_loss_i = F.mse_loss(current_sequence_value_i.squeeze(), td_target_i.squeeze()) # Target is scalar TD target

                # Entropy Bonus
                entropy_loss_i = -entropy_i.mean() # Negated for minimization


                # Total Loss for this template
                total_template_loss_i = policy_loss_i + self.vf_coef * value_loss_i + self.ce_coef * entropy_loss_i

                # --- 7. Backward Pass (Accumulates gradients) ---
                # Scale loss for accumulation if needed (Accelerator might handle this)
                # scaled_loss = total_template_loss_i / self.accelerator.gradient_accumulation_steps
                self.accelerator.backward(total_template_loss_i)

                # --- 8. Logging ---
                with torch.no_grad():
                     approx_kl_i = ((ratio_i - 1) - log_ratio_i).mean().item() # For this template
                     total_template_loss_accum += total_template_loss_i.item()
                     policy_loss_accum += policy_loss_i.item()
                     value_loss_accum += value_loss_i.item()
                     kl_div_accum += approx_kl_i
                     templates_processed_count += 1

            # --- End of Epoch ---
            self.accelerator.print(f"--- Finished Epoch {epoch+1} ---")
            # Optional: Log epoch-level averages if needed


        # --- 9. Optimizer Step & Zero Grad (After all epochs and backward passes) ---
        if templates_processed_count > 0:
             if hasattr(self, 'max_grad_norm') and self.max_grad_norm is not None:
                  self.accelerator.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                  self.accelerator.clip_grad_norm_(self.harm_critic.parameters(), self.max_grad_norm)
                  if self.help_critic: self.accelerator.clip_grad_norm_(self.help_critic.parameters(), self.max_grad_norm)

             self.optimizer_step() # Step all optimizers
             self.optimizer_zero_grad() # Zero grads for next batch
        else:
             self.log.warning("No templates processed in BT training batch. Skipping optimizer step.")


        # --- 10. Final Logging & Stats ---
        avg_total_loss = total_template_loss_accum / templates_processed_count if templates_processed_count > 0 else 0
        avg_policy_loss = policy_loss_accum / templates_processed_count if templates_processed_count > 0 else 0
        avg_value_loss = value_loss_accum / templates_processed_count if templates_processed_count > 0 else 0
        avg_kl_div = kl_div_accum / templates_processed_count if templates_processed_count > 0 else 0

        # Gather stats across processes (use average loss for simplicity)
        gathered_total_loss = self.accelerator.gather(torch.tensor(avg_total_loss, device=self.accelerator.device)).mean()

        stats = {}
        if self.accelerator.is_main_process:
             stats = {
                 'avg_template_loss': gathered_total_loss.item(),
                 # Add more detailed gathered stats if needed
                 'policy_loss_approx': avg_policy_loss, # From rank 0 perspective before gather
                 'value_loss_approx': avg_value_loss, # From rank 0 perspective before gather
                 'kl_div_approx': avg_kl_div, # From rank 0 perspective before gather
                 'templates_processed': templates_processed_count * self.accelerator.num_processes # Estimate total
             }
             self.log.info(f"BT Batch Training Complete. Avg Template Loss: {stats['avg_template_loss']:.4f}")
             self._log_summary(stats) # Use adapted log summary

        return stats

# Example usage block remains the same