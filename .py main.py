[1mdiff --git a/README.md b/README.md[m
[1mindex 6314ca2..67ec505 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -1,5 +1,60 @@[m
 # Auto Red Teaming LLMs[m
 [m
[32m+[m[32m## Overview[m
[32m+[m[32mThis repository contains code for automatically red-teaming large language models (LLMs) using reinforcement learning techniques. The system is designed to find and exploit vulnerabilities in target LLMs by generating adversarial prompts.[m
[32m+[m
[32m+[m[32m## Features[m
[32m+[m[32m- Reinforcement learning-based red teaming of LLMs[m
[32m+[m[32m- Support for different target models[m
[32m+[m[32m- Token-wise reward attribution[m
[32m+[m[32m- Customizable attack strategies[m
[32m+[m
[32m+[m[32m## Installation[m
[32m+[m[32m```bash[m
[32m+[m[32mgit clone https://github.com/yourusername/auto-red-teaming-llms.git[m
[32m+[m[32mcd auto-red-teaming-llms[m
[32m+[m[32mpip install -r requirements.txt[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m## Usage[m
[32m+[m[32m### Basic Training[m
[32m+[m[32m```bash[m
[32m+[m[32mpython main.py --train --data_dir data/questions --checkpoint meta-llama/Llama-3.2-1B-Instruct[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Evaluation[m
[32m+[m[32m```bash[m
[32m+[m[32mpython main.py --test --test_model your_model_name --data_dir data/questions --eval_dataset PKU-Alignment/PKU-SafeRLHF[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Using a Separate Target Model[m
[32m+[m[32mFor consistent evaluation or to test against a specific model version:[m
[32m+[m[32m```bash[m
[32m+[m[32mpython main.py --test --data_dir data/questions --target_model meta-llama/Llama-3.2-3B-Instruct[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Fine-tuning with LoRA[m
[32m+[m[32m```bash[m
[32m+[m[32mpython main.py --train --data_dir data/questions --use_finetuned --adapter_path ./models/fine-tuned/llama-lora[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Detuning Model Safety Alignment[m
[32m+[m[32m```bash[m
[32m+[m[32mpython main.py --train --data_dir data/questions --detune[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m## Key Parameters[m
[32m+[m[32m- `--train`: Enable training mode[m
[32m+[m[32m- `--test`: Enable testing mode[m
[32m+[m[32m- `--data_dir`: Directory containing training/testing data[m
[32m+[m[32m- `--checkpoint`: Base model checkpoint to use[m
[32m+[m[32m- `--target_model`: Specific model to use for evaluation (if different from base model)[m
[32m+[m[32m- `--episodes`: Number of episodes to train/test[m
[32m+[m[32m- `--help_weight`: Weight for balancing harm vs help advantages (0-1)[m
[32m+[m[32m- `--attack_type`: Type of attack to perform (rephrase or breakdown)[m
[32m+[m[32m- `--use_finetuned`: Use a fine-tuned model with LoRA adapters[m
[32m+[m[32m- `--adapter_path`: Path to LoRA adapter weights[m
[32m+[m[32m- `--detune`: Enable detuning of model safety alignment[m
 [m
 #### env data flow: [m
 When the agent interacts with the environment, the environment passes a list of lists of strings, i.e. [["x1"],["x2"]]. The agent generates responses `"yi"` (via tokenizer.apply_chat_template() and helpers.build_chat()) and appends them to the state, creating [["x1","y1"],["x1","y1"]] and so on.[m
[36m@@ -37,4 +92,4 @@[m [mBenefits:[m
 Usage:[m
 - The token-wise rewards are computed in the `value_step` function[m
 - Only harm scores use token-wise rewards; help scores remain sequence-level[m
[31m-- Token-wise rewards are leveraged in policy gradients through token-level advantages[m
\ No newline at end of file[m
[32m+[m[32m- Token-wise rewards are leveraged in policy gradients through token-level advantages[m
[1mdiff --git a/agent.py b/agent.py[m
[1mindex 43344c5..9fe48d2 100644[m
[1m--- a/agent.py[m
[1m+++ b/agent.py[m
[36m@@ -27,7 +27,7 @@[m [mclass Agent:[m
 	"""[m
 		This is the PPO agent class we will use as our model in main.py[m
 	"""[m
[31m-	def __init__(self, policy_type, data_path, hf_checkpoint='meta-llama/Llama-3.2-1B-Instruct', auth_token=None, hyperparameters=None, use_finetuned=False, adapter_path=None, merge_adapters=False):[m
[32m+[m	[32mdef __init__(self, policy_type, data_path, hf_checkpoint='meta-llama/Llama-3.2-1B-Instruct', auth_token=None, hyperparameters=None, use_finetuned=False, adapter_path=None, merge_adapters=False, use_lora=False):[m
 		"""[m
 			Initializes the PPO model, including hyperparameters.[m
 [m
[36m@@ -39,6 +39,7 @@[m [mclass Agent:[m
 				use_finetuned - whether to use a fine-tuned model with LoRA adapters for the actor[m
 				adapter_path - path to the LoRA adapter weights[m
 				merge_adapters - whether to merge LoRA adapters with the base model[m
[32m+[m				[32muse_lora - whether to use LoRA for fine-tuning[m
 [m
 			Returns:[m
 				None[m
[36m@@ -54,11 +55,69 @@[m [mclass Agent:[m
 		[m
 		self._init_hyperparameters(hyperparameters)[m
 		[m
[32m+[m		[32m# Set training mode based on hyperparameters[m
[32m+[m		[32mself.train = hyperparameters.get('train', False)[m
[32m+[m[41m		[m
[32m+[m		[32m# Set up LoRA configuration[m
[32m+[m		[32mself.use_lora = use_lora[m
[32m+[m		[32mself.lora_adapters = {[m
[32m+[m			[32m'lm_head_rephrase': None,[m
[32m+[m			[32m'attention_rephrase': None,[m
[32m+[m			[32m'lm_head_followup': None,[m
[32m+[m			[32m'attention_followup': None[m
[32m+[m		[32m}[m
[32m+[m[41m		[m
[32m+[m		[32m# Configure LoRA parameters[m
[32m+[m		[32mif use_lora:[m
[32m+[m			[32mself.lora_configs = {[m
[32m+[m				[32m'lm_head_rephrase': {[m
[32m+[m					[32m'r': self.lora_r,[m
[32m+[m					[32m'lora_alpha': self.lora_alpha,[m
[32m+[m					[32m'lora_dropout': self.lora_dropout,[m
[32m+[m					[32m'target_modules': ['lm_head'],[m
[32m+[m					[32m'bias': 'none'[m
[32m+[m				[32m},[m
[32m+[m				[32m'attention_rephrase': {[m
[32m+[m					[32m'r': self.lora_r,[m
[32m+[m					[32m'lora_alpha': self.lora_alpha,[m
[32m+[m					[32m'lora_dropout': self.lora_dropout,[m
[32m+[m					[32m'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],[m
[32m+[m					[32m'bias': 'none'[m
[32m+[m				[32m},[m
[32m+[m				[32m'lm_head_followup': {[m
[32m+[m					[32m'r': self.lora_r,[m
[32m+[m					[32m'lora_alpha': self.lora_alpha,[m
[32m+[m					[32m'lora_dropout': self.lora_dropout,[m
[32m+[m					[32m'target_modules': ['lm_head'],[m
[32m+[m					[32m'bias': 'none'[m
[32m+[m				[32m},[m
[32m+[m				[32m'attention_followup': {[m
[32m+[m					[32m'r': self.lora_r,[m
[32m+[m					[32m'lora_alpha': self.lora_alpha,[m
[32m+[m					[32m'lora_dropout': self.lora_dropout,[m
[32m+[m					[32m'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],[m
[32m+[m					[32m'bias': 'none'[m
[32m+[m				[32m}[m
[32m+[m			[32m}[m
[32m+[m[41m			[m
[32m+[m			[32m# Create adapter directories if they don't exist[m
[32m+[m			[32mfor adapter_name in self.lora_adapters.keys():[m
[32m+[m				[32madapter_dir = f'./models/{model_name}/adapters/{adapter_name}'[m
[32m+[m				[32mos.makedirs(adapter_dir, exist_ok=True)[m
[32m+[m				[32mself.lora_adapters[adapter_name] = adapter_dir[m
[32m+[m[41m		[m
 		policy_class = getPolicyClass(policy_type)[m
 		os.environ["TOKENIZERS_PARALLELISM"] = "false"[m
 		tokenizer_class = getTokenizerClass(policy_type)[m
 		special_tokens_func = loadSpecialTokens(hf_checkpoint)[m
[31m-		self.env = MultiStepEnv(data_path, hf_checkpoint, auth_token, batch_size=self.minibatch_size,max_steps=self.max_ep_len)[m
[32m+[m		[32mprint(f"max ep len: {self.max_ep_len}")[m
[32m+[m		[32mself.env = MultiStepEnv([m
[32m+[m			[32mdata_path,[m[41m [m
[32m+[m			[32mauth_token,[m[41m [m
[32m+[m			[32mmax_steps=self.max_ep_len,[m[41m [m
[32m+[m			[32mbatch_size=self.minibatch_size,[m
[32m+[m			[32mtarget_model_name=self.target_model[m
[32m+[m		[32m)[m
 		self.instructions = self.env.instructions[m
 		[m
 		# Initialize tokenizer[m
[36m@@ -121,18 +180,8 @@[m [mclass Agent:[m
 		else:[m
 			# Use the standard model loading in Actor[m
 			self.actor = policy_class(hf_checkpoint, self.tokenizer, token=auth_token)[m
[31m-		[m
[31m-		self.actor.check_layer_grads()[m
[31m-			[m
[31m-		# Check if there's a detuned version of this model[m
[31m-		model_prefix = model_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")[m
[31m-		detuned_path = f'./models/{model_prefix}-detuned'[m
[31m-		[m
[31m-		# If detuned model exists and has actor.pt, load it into the actor model[m
[31m-		if os.path.exists(detuned_path) and os.path.exists(f'{detuned_path}/actor.pt') and not use_finetuned:[m
[31m-			print(f"Found detuned model at {detuned_path}, loading weights")[m
[31m-			self.actor.model.load_state_dict(torch.load(f'{detuned_path}/actor.pt'))[m
[31m-			print("Detuned model weights loaded successfully")[m
[32m+[m		[32mself.actor_optim = Adam(self.actor.parameters(), lr=self.lr)[m
[32m+[m		[32mself.score_to_value = getValueFunction(self.val_type)[m
 		[m
 		if self.alg == 'ppo':[m
 			# Initialize two critics - one for harm and one for helpfulness[m
[36m@@ -145,22 +194,9 @@[m [mclass Agent:[m
 			[m
 			self.harm_critic_optim = Adam(self.harm_critic.parameters(), lr=self.value_lr)[m
 			self.help_critic_optim = Adam(self.help_critic.parameters(), lr=self.value_lr)[m
[31m-		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)[m
[31m-		self.score_to_value = getValueFunction(self.val_type)[m
[31m-		[m
[31m-		self.logger = {[m
[31m-			'delta_t': time.time_ns(),[m
[31m-			't': 100,          [m
[31m-			'i_so_far': 0,          [m
[31m-			'batch_lens': [],       [m
[31m-			'batch_rews': [],       [m
[31m-			'actor_losses': [],     [m
[31m-			'harm_critic_losses': [],[m
[31m-			'help_critic_losses': [][m
[31m-		}[m
 [m
[31m-		## other hyperparams[m
[31m-		self.save_freq = 10[m
[32m+[m		[32mself.actor.check_layer_grads()[m
[32m+[m		[32m#load existing model[m
 		if self.test_model is not None:[m
 			assert os.path.exists(f'./models/{self.test_model}/'), "Model folder does not exist"[m
 		if os.path.exists(f'./models/{self.model_name}/'):[m
[36m@@ -173,7 +209,29 @@[m [mclass Agent:[m
 			config_path = f'./models/{self.model_name}/config.json'[m
 			with open(config_path, 'w') as config_file:[m
 				json.dump(hyperparameters, config_file, indent=4)[m
[32m+[m[41m			[m
[32m+[m		[32m# Check if there's a detuned version of this model[m
[32m+[m		[32mmodel_prefix = model_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")[m
[32m+[m		[32mdetuned_path = f'./models/{model_prefix}-detuned'[m
[32m+[m[41m		[m
[32m+[m		[32m# If detuned model exists and has actor.pt, load it into the actor model[m
[32m+[m		[32mif self.use_detuned and os.path.exists(detuned_path) and os.path.exists(f'{detuned_path}/actor.pt') and not use_finetuned:[m
[32m+[m			[32mprint(f"Found detuned model at {detuned_path}, loading weights")[m
[32m+[m			[32mself.actor.model.load_state_dict(torch.load(f'{detuned_path}/actor.pt'))[m
[32m+[m			[32mprint("Detuned model weights loaded successfully")[m
[32m+[m[41m		[m
[32m+[m		[32mself.logger = {[m
[32m+[m			[32m'delta_t': time.time_ns(),[m
[32m+[m			[32m't': 100,[m[41m          [m
[32m+[m			[32m'i_so_far': 0,[m[41m          [m
[32m+[m			[32m'batch_lens': [],[m[41m       [m
[32m+[m			[32m'batch_rews': [],[m[41m       [m
[32m+[m			[32m'actor_losses': [],[m[41m     [m
[32m+[m			[32m'harm_critic_losses': [],[m
[32m+[m			[32m'help_critic_losses': [][m
[32m+[m		[32m}[m
 [m
[32m+[m		[32mself.save_freq = 10[m
 		self.manually_add_bos = self.check_template_inserts_bos()[m
 		self.actor.check_layer_grads()[m
 [m
[36m@@ -253,218 +311,156 @@[m [mclass Agent:[m
 [m
 				# Use the attack type to determine which rollout to perform[m
 				if self.attack_type == 'breakdown':[m
[31m-					rollout = self.breakdown_rollout(max(self.max_timesteps // self.episodes, self.batch_size))[m
[31m-				else:  # Default to rephrase[m
[31m-					rollout = self.rollout(max(self.max_timesteps // self.episodes, self.batch_size))[m
[31m-[m
[31m-				# Calculate how many timesteps we collected this batch[m
[31m-				t += np.sum(rollout.batch_lens)[m
[31m-[m
[31m-				# Increment the number of iterations[m
[31m-				i_so_far += 1[m
[31m-[m
[31m-				# Logging timesteps so far and iterations so far[m
[31m-				self.logger['t'] = t[m
[31m-				self.logger['i_so_far'] = i_so_far[m
[31m-[m
[31m-				# This is the loop where we update our network for some n epochs[m
[31m-				for epoch in range(self.epochs):[m
[31m-					# Shuffle the rollout data[m
[31m-					perm = torch.randperm(rollout.batch_state.size(0))[m
[31m-					batch_state_shuffled = rollout.batch_state[perm][m
[31m-					batch_next_state_shuffled = rollout.batch_next_state[perm][m
[31m-					batch_base_state_shuffled = rollout.batch_base_state[perm][m
[31m-					batch_acts_shuffled = rollout.batch_acts[perm][m
[31m-					batch_log_probs_shuffled = rollout.batch_log_probs[perm][m
[31m-					batch_rews_shuffled = rollout.batch_rews[perm][m
[31m-					if self.alg == 'ppo':[m
[31m-						batch_compliance_rews_shuffled = rollout.batch_compliance_rews[perm][m
[31m-						batch_action_help_rews_shuffled = rollout.batch_action_help_rews[perm][m
[31m-[m
[31m-					# Minibatch training - interleave value and policy updates[m
[31m-					window_size = self.minibatch_size[m
[31m-					for j in range(0, rollout.batch_state.size(0), window_size):[m
[31m-						minibatch_state = batch_state_shuffled[j:j + window_size][m
[31m-						minibatch_next_state = batch_next_state_shuffled[j:j + window_size][m
[31m-						minibatch_base_state = batch_base_state_shuffled[j:j + window_size][m
[31m-						minibatch_acts = batch_acts_shuffled[j:j + window_size][m
[31m-						minibatch_log_probs = batch_log_probs_shuffled[j:j + window_size][m
[31m-						minibatch_rews = batch_rews_shuffled[j:j + window_size][m
[31m-						[m
[31m-						if self.alg == 'ppo':[m
[31m-							minibatch_comply_rews = batch_compliance_rews_shuffled[j:j + window_size][m
[31m-							minibatch_action_help_rews = batch_action_help_rews_shuffled[j:j + window_size][m
[31m-							[m
[31m-							# 1. First do the value update for this minibatch[m
[31m-							(harm_values, help_values), (harm_errors, help_errors), (harm_critic_loss, help_critic_loss), (harm_token_scores, help_token_scores), (action_harm, action_help) = self.value_step([m
[31m-								minibatch_state, [m
[31m-								minibatch_rews,[m
[31m-								minibatch_comply_rews,  [m
[31m-								minibatch_next_state,[m
[31m-								action_help_reward=minibatch_action_help_rews[m
[31m-							)[m
[31m-							[m
[31m-							# Log both critic losses - ensure detachment before moving to CPU[m
[31m-							self.logger['harm_critic_losses'].append(harm_critic_loss.detach().cpu())[m
[31m-							self.logger['help_critic_losses'].append(help_critic_loss.detach().cpu())[m
[31m-						[m
[31m-						# 2. Now do the policy update for this same minibatch[m
[31m-						with torch.enable_grad():[m
[31m-							# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)[m
[31m-							log_probs_old = minibatch_log_probs.to('cuda')[m
[31m-							curr_log_probs = self.actor.get_log_probs(minibatch_state, minibatch_acts, sequence_level=False)[m
[31m-							[m
[31m-							#compute A_k according to which algorithm we are using[m
[31m-							if self.alg == 'reinforce':[m
[31m-								A_k = minibatch_rews.to('cuda') - minibatch_rews.mean().to('cuda')[m
[31m-								if A_k.ndim == 1:[m
[31m-									A_k = A_k.unsqueeze(1)[m
[31m-							elif self.alg == 'ppo-pretrain':[m
[31m-								# Use the environment's score_safety function as the critic[m
[31m-								target_response = self.tokenizer.batch_decode(minibatch_next_state, skip_special_tokens=True)[m
[31m-								target_response = [x.split('assistant')[-1] for x in target_response][m
[31m-								target_response = [x.split('user')[-1] for x in target_response][m
[31m-								current_state_action_tokens = torch.cat((minibatch_state.to('cuda'), minibatch_acts.to('cuda')), dim=1)[m
[31m-								current_state_action = self.tokenizer.batch_decode(current_state_action_tokens, skip_special_tokens=True)[m
[31m-								values = self.env.score_safety(current_state_action)[m
[31m-								next_values = self.env.score_safety(target_response)[m
[31m-								A_k, returns = self._compute_advantages(minibatch_rews.to('cuda'), values.to('cuda'), next_values=next_values.to('cuda'))[m
[31m-							elif self.alg == 'ppo':								[m
[31m-								help_rews = minibatch_comply_rews + minibatch_action_help_rews[m
[31m-								[m
[31m-								# Compute GAE with sequence-level predicted rewards[m
[31m-								# Note: Even for the harm critic which has token-level outputs,[m
[31m-								# we use the sequence-level aggregated value (sum of token values) for GAE calculation[m
[31m-								next_help_pred = self.help_critic(minibatch_next_state)[m
[31m-								help_A_k = self._compute_advantages(help_rews.to('cuda'), help_values.detach(), next_values=next_help_pred.to('cuda'))[0][m
[31m-								[m
[31m-								# For harm rewards, we also use sequence-level aggregated values for GAE[m
[31m-								harm_rews = minibatch_rews - action_harm[m
[31m-								next_harm_pred = self.harm_critic(minibatch_next_state)[m
[31m-								harm_A_k = self._compute_advantages(harm_rews.to('cuda'), harm_values.detach(), next_values=next_harm_pred.to('cuda'))[0][m
[31m-								[m
[31m-								# Use advantages for policy update[m
[31m-								A_k = harm_A_k * (1 - self.help_weight) + help_A_k * self.help_weight[m
[31m-								[m
[31m-								if False:[m
[31m-									# Also calculate sequence-level advantages for logging and comparison[m
[31m-									harm_seq_advantages = minibatch_rews.to('cuda') - harm_errors.detach()[m
[31m-									help_seq_advantages = help_rews.to('cuda') - help_errors.detach()[m
[31m-									seq_advantages = harm_seq_advantages * (1 - self.help_weight) + help_seq_advantages * self.help_weight[m
[31m-									[m
[31m-									# Log the difference between token-level and sequence-level advantages[m
[31m-									if epoch == 0 and j == 0:[m
[31m-										print(f"Token-level advantages mean: {A_k.mean()}")[m
[31m-										print(f"Sequence-level advantages mean: {seq_advantages.mean()}")[m
[31m-								[m
[31m-							elif self.alg == 'grpo':[m
[31m-								raise NotImplementedError(f"Algorithm {self.alg} not implemented")[m
[31m-							else:[m
[31m-								raise NotImplementedError(f"Algorithm {self.alg} not implemented")[m
[31m-								assert curr_log_probs is not None, "curr_log_probs must be defined"[m
[31m-[m
[31m-							A_k.requires_grad_(True)[m
[31m-							curr_log_probs.requires_grad_(True)[m
[31m-							# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)[m
[31m-							ratios = curr_log_probs - log_probs_old[m
[31m-							if A_k.shape != ratios.shape:[m
[31m-								assert A_k.shape[0] == ratios.shape[0][m
[31m-								assert A_k.ndim == 2, f"A_k should be 2D but got shape {A_k.shape}"[m
[31m-								assert ratios.ndim == 2, f"ratios should be 2D but got shape {ratios.shape}"[m
[31m-								[m
[31m-								# Reshape if dimensions don't match[m
[31m-								if A_k.shape[1] != ratios.shape[1]:[m
[31m-									if A_k.shape[1] < ratios.shape[1]:[m
[31m-										# Pad A_k with zeros[m
[31m-										padding = torch.zeros([m
[31m-											(A_k.shape[0], ratios.shape[1] - A_k.shape[1]),[m
[31m-											device=A_k.device[m
[31m-										)[m
[31m-										A_k = torch.cat([A_k, padding], dim=1)[m
[31m-									else:[m
[31m-										# Truncate A_k[m
[31m-										A_k = A_k[:, :ratios.shape[1]][m
[31m-							[m
[31m-							surr1 = ratios * A_k[m
[31m-							surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k[m
[31m-							[m
[31m-							semantic_score = self.actor.get_embed_similarity(minibatch_state, minibatch_base_state)[m
[31m-							# Calculate the loss[m
[31m-							actor_loss = (-torch.min(surr1, surr2)).mean() - (semantic_score.mean().to(ratios.device))*self.sem_coef[m
[31m-							[m
[31m-							if epoch == 0 and j == 0:[m
[31m-								report_str = f"a_k mean: {A_k.mean()}, actor_loss: {actor_loss}, mean surr1: {surr1.mean()}, mean surr2: {surr2.mean()}"[m
[31m-								if self.alg == 'ppo':[m
[31m-									report_str += f", mean acts_harm: {action_harm.mean()}, mean acts_help: {action_help.mean()}, mean semantic_score: {semantic_score.mean()}"[m
[31m-								print(report_str)[m
[31m-							[m
[31m-							# Update actor[m
[31m-							self.actor_optim.zero_grad()[m
[31m-							actor_loss.backward()[m
[31m-							self.actor.check_embeddings()[m
[31m-							self.actor_optim.step()[m
[31m-							[m
[31m-							# Log actor loss[m
[31m-							self.logger['actor_losses'].append(actor_loss.detach().cpu())[m
[31m-							[m
[31m-							print(f"Epoch {epoch} minibatch {j}")[m
[32m+[m					[32mbatch_data = self.breakdown_rollout(t)[m
[32m+[m				[32melse:[m
[32m+[m					[32mbatch_data = self.rollout(t)[m
[32m+[m[41m					[m
[32m+[m				[32m# Separate data for first-stage (rephrasing) and follow-up[m
[32m+[m				[32mfirst_stage_data = self._extract_first_stage_data(batch_data)[m
[32m+[m				[32mfollowup_data = self._extract_followup_data(batch_data)[m
[32m+[m[41m				[m
[32m+[m				[32m# Train with different adapters based on the stage[m
[32m+[m				[32mif self.use_lora:[m
[32m+[m					[32m# First train LM head for rephrasing[m
[32m+[m					[32mprint("Training LM head for rephrasing...")[m
[32m+[m					[32mself.train_with_adapter('lm_head_rephrase', first_stage_data)[m
[32m+[m[41m					[m
[32m+[m					[32m# Then train attention for rephrasing[m
[32m+[m					[32mprint("Training attention for rephrasing...")[m
[32m+[m					[32mself.train_with_adapter('attention_rephrase', first_stage_data)[m
[32m+[m[41m					[m
[32m+[m					[32m# Then train LM head for follow-up[m
[32m+[m					[32mif len(followup_data['batch_state']) > 0:[m
[32m+[m						[32mprint("Training LM head for follow-up...")[m
[32m+[m						[32mself.train_with_adapter('lm_head_followup', followup_data)[m
 						[m
[31m-						# Clear CUDA cache to prevent memory issues[m
[31m-						torch.cuda.empty_cache()[m
[31m-[m
[31m-				self._log_summary()[m
[31m-				# Save our model if it's time[m
[32m+[m						[32m# Then train attention for follow-up[m
[32m+[m						[32mprint("Training attention for follow-up...")[m
[32m+[m						[32mself.train_with_adapter('attention_followup', followup_data)[m
[32m+[m				[32melse:[m
[32m+[m					[32m# Traditional training without LoRA adapters[m
[32m+[m					[32mself._train_on_batch(batch_data)[m
[32m+[m[41m				[m
[32m+[m				[32m# Increment counters[m
[32m+[m				[32mt += self.batch_size[m
[32m+[m				[32mi_so_far += 1[m
[32m+[m[41m				[m
[32m+[m				[32m# Log summary and save model[m
 				if i_so_far % self.save_freq == 0:[m
[32m+[m					[32mself._log_summary()[m
 					self._save()[m
[32m+[m[41m					[m
[32m+[m				[32m# Move model back to CPU to save memory[m
[32m+[m				[32mself.to_cpu()[m
[32m+[m[41m				[m
[32m+[m		[32m# Save the final model[m
[32m+[m		[32mself._save()[m
[32m+[m[41m		[m
[32m+[m	[32mdef _extract_first_stage_data(self, batch_data):[m
[32m+[m		[32m"""[m
[32m+[m		[32mExtract data for the first stage (rephrasing) from the batch data.[m
[32m+[m[41m		[m
[32m+[m		[32mArgs:[m
[32m+[m			[32mbatch_data: Dictionary containing batch data[m
 			[m
[31m-				self.actor.zero_grad()[m
[31m-				self.actor_optim.zero_grad()[m
[31m-				if self.alg == 'ppo':[m
[31m-					self.harm_critic_optim.zero_grad()[m
[31m-					self.help_critic_optim.zero_grad()[m
[31m-				torch.cuda.empty_cache()[m
[32m+[m		[32mReturns:[m
[32m+[m			[32mDictionary containing first stage data[m
[32m+[m		[32m"""[m
[32m+[m		[32m# In the first stage, we only care about the first step in each episode[m
[32m+[m		[32mfirst_stage_indices = [][m
[32m+[m[41m		[m
[32m+[m		[32m# Find indices of first steps in each episode[m
[32m+[m		[32mfor i, state in enumerate(batch_data['batch_state']):[m
[32m+[m			[32mif len(state) == 1:  # First step in an episode[m
[32m+[m				[32mfirst_stage_indices.append(i)[m
[32m+[m[41m				[m
[32m+[m		[32m# Extract data for first stage[m
[32m+[m		[32mfirst_stage_data = {[m
[32m+[m			[32m'batch_state': [batch_data['batch_state'][i] for i in first_stage_indices],[m
[32m+[m			[32m'batch_acts': [batch_data['batch_acts'][i] for i in first_stage_indices],[m
[32m+[m			[32m'batch_log_probs': [batch_data['batch_log_probs'][i] for i in first_stage_indices],[m
[32m+[m			[32m'batch_advantages': [batch_data['batch_advantages'][i] for i in first_stage_indices][m
[32m+[m		[32m}[m
[32m+[m[41m		[m
[32m+[m		[32mif 'batch_values' in batch_data:[m
[32m+[m			[32mfirst_stage_data['batch_values'] = [batch_data['batch_values'][i] for i in first_stage_indices][m
[32m+[m[41m			[m
[32m+[m		[32mreturn first_stage_data[m
[32m+[m[41m		[m
[32m+[m	[32mdef _extract_followup_data(self, batch_data):[m
[32m+[m		[32m"""[m
[32m+[m		[32mExtract data for the follow-up stage from the batch data.[m
[32m+[m[41m		[m
[32m+[m		[32mArgs:[m
[32m+[m			[32mbatch_data: Dictionary containing batch data[m
[32m+[m[41m			[m
[32m+[m		[32mReturns:[m
[32m+[m			[32mDictionary containing follow-up stage data[m
[32m+[m		[32m"""[m
[32m+[m		[32m# In the follow-up stage, we care about all steps except the first in each episode[m
[32m+[m		[32mfollowup_indices = [][m
[32m+[m[41m		[m
[32m+[m		[32m# Find indices of follow-up steps in each episode[m
[32m+[m		[32mfor i, state in enumerate(batch_data['batch_state']):[m
[32m+[m			[32mif len(state) > 1:  # Follow-up step in an episode[m
[32m+[m				[32mfollowup_indices.append(i)[m
[32m+[m[41m				[m
[32m+[m		[32m# Extract data for follow-up stage[m
[32m+[m		[32mfollowup_data = {[m
[32m+[m			[32m'batch_state': [batch_data['batch_state'][i] for i in followup_indices],[m
[32m+[m			[32m'batch_acts': [batch_data['batch_acts'][i] for i in followup_indices],[m
[32m+[m			[32m'batch_log_probs': [batch_data['batch_log_probs'][i] for i in followup_indices],[m
[32m+[m			[32m'batch_advantages': [batch_data['batch_advantages'][i] for i in followup_indices][m
[32m+[m		[32m}[m
[32m+[m[41m		[m
[32m+[m		[32mif 'batch_values' in batch_data:[m
[32m+[m			[32mfollowup_data['batch_values'] = [batch_data['batch_values'][i] for i in followup_indices][m
[32m+[m[41m			[m
[32m+[m		[32mreturn followup_data[m
 [m
[31m-	def value_step(self, input, harm_reward, help_reward, next_input, action_help_reward=None, gamma=0.99):[m
[32m+[m	[32mdef value_step(self, response, harm_reward, help_reward, next_response, action=None, action_help_reward=None, gamma=0.99):[m
 		"""[m
 		Compute values based on our theory:[m
 		[m
[31m-		Harm value: V_h(s,a) = H(target_response) - H(agent_action)[m
[32m+[m		[32mHarm value: V_h(s,a) = R(response) - R(action) + γ*V_h(next_response)[m
 		- We want high value when target response is harmful but our action isn't[m
[31m-		- H(target_response) comes from environment reward (harm_reward)[m
[31m-		- H(agent_action) comes from env.score_safety[m
[31m-		[m
[31m-		Help value: V_help(s,a) = Help(action|query) + Help(target|action)[m
[31m-		- First term from batch_action_help_rews: helpfulness of our action[m
[31m-		- Second term from batch_compliance_rews: compliance of target[m
[32m+[m		[32m- R(response) comes from environment reward (harm_reward)[m
[32m+[m		[32m- R(action) comes from env.score_safety[m
[32m+[m		[32m- γ*V_h(next_response) accounts for future harm[m
[32m+[m[41m		[m
[32m+[m		[32mHelp value: V_help(s,a) = Help(action|query) + Help(response|action) + γ*V_help(next_response)[m
[32m+[m		[32m- First term from action_help_reward: helpfulness of our action[m
[32m+[m		[32m- Second term from help_reward: compliance of target[m
[32m+[m		[32m- γ*V_help(next_response) accounts for future helpfulness[m
 		- We want to maximize total helpfulness of interaction[m
 		[m
[31m-		The value model now outputs per-token scores that are aggregated to predict[m
[32m+[m		[32mThe value model outputs per-token scores that are aggregated to predict[m
 		sequence-level rewards, allowing for more fine-grained advantage calculation.[m
[31m-		[m
[31m-		For each sequence in the minibatch, we create N copies (where N is the sequence length) with[m
[31m-		each token masked once along the diagonal. This allows us to assess the[m
[31m-		contribution of each token to the overall value.[m
[31m-		[m
[31m-		To reduce GPU memory usage:[m
[31m-		1. Process each sequence individually[m
[31m-		2. Free memory after each sequence is processed[m
[31m-		3. Process harm values first, then help values[m
 		"""[m
 		# Process inputs - handle minibatch of sequences[m
[31m-		batch_size = input.shape[0][m
[31m-		max_seq_len = input.shape[1][m
[31m-		input_texts = self.tokenizer.batch_decode(input, skip_special_tokens=False)[m
[31m-		tokens = self.tokenizer(input_texts, return_tensors='pt', padding=False)[m
[31m-				[m
[31m-		# Get agent's action[m
[31m-		action_texts = self.tokenizer.batch_decode(next_input, skip_special_tokens=False)[m
[32m+[m		[32mbatch_size = response.shape[0][m
[32m+[m		[32mmax_seq_len = response.shape[1][m
[32m+[m		[32minput_texts = self.tokenizer.batch_decode(response, skip_special_tokens=False)[m[41m        [m
[32m+[m[41m		[m
[32m+[m		[32m# Get action texts if provided, otherwise extract from next_response[m
[32m+[m		[32mif action is not None:[m
[32m+[m			[32mif isinstance(action, torch.Tensor):[m
[32m+[m				[32maction_texts = self.tokenizer.batch_decode(action, skip_special_tokens=False)[m
[32m+[m			[32melse:[m
[32m+[m				[32maction_texts = action[m
[32m+[m		[32melse:[m
[32m+[m			[32m# If action not provided, we need to extract it from context[m
[32m+[m			[32maction_texts = self.tokenizer.batch_decode(next_response, skip_special_tokens=False)[m
 		[m
[31m-		# Prepare state-action pairs[m
[32m+[m		[32mnext_texts = self.tokenizer.batch_decode(next_response, skip_special_tokens=False)[m
[32m+[m[41m		[m
[32m+[m		[32m# Get state-action pairs for helpfulness scoring[m
 		state_action_pairs = [q + " " + a for q, a in zip(input_texts, action_texts)][m
[32m+[m		[32maction_help = action_help_reward.to('cuda') if action_help_reward is not None else self.env.score_helpfulness(state_action_pairs).to('cuda')[m
 		[m
 		# Get harmfulness scores[m
 		target_harm = harm_reward.to('cuda')  # From environment[m
[31m-		action_harm = self.env.score_safety(state_action_pairs).to('cuda').squeeze(-1)[m
[32m+[m		[32maction_harm = self.env.score_safety(action_texts).to('cuda')[m
 		[m
 		#######################[m
 		# PART 1: HARM VALUES #[m
[36m@@ -478,25 +474,38 @@[m [mclass Agent:[m
 		harm_errors = torch.zeros(batch_size, device='cuda')[m
 		harm_token_scores_batch = torch.zeros((batch_size, max_seq_len), device='cuda')[m
 		[m
[31m-		# Process each sequence individually[m
[32m+[m		[32m# First compute next state values for bootstrapping[m
[32m+[m		[32mnext_tokens = self.tokenizer(next_texts, return_tensors='pt', padding=True)[m
[32m+[m		[32mnext_input_ids = next_tokens.input_ids.to('cuda')[m
[32m+[m		[32mnext_attention_mask = torch.where(next_input_ids == self.tokenizer.pad_token_id,[m[41m [m
[32m+[m										[32m torch.zeros_like(next_input_ids),[m[41m [m
[32m+[m										[32m torch.ones_like(next_input_ids)).to('cuda')[m
[32m+[m[41m		[m
[32m+[m		[32m# Get next state values for bootstrapping[m
[32m+[m		[32mwith torch.no_grad():[m
[32m+[m			[32mnext_harm_values = self.harm_critic([m
[32m+[m				[32minput_ids=next_input_ids,[m
[32m+[m				[32mattention_mask=next_attention_mask[m
[32m+[m			[32m).detach()[m
[32m+[m[41m		[m
[32m+[m		[32m# Process each sequence individually for token-level analysis[m
 		for b in range(batch_size):[m
[31m-			#get the length of the input tensor[m
[31m-			seq_len = tokens[b].attention_mask.shape[1][m
[32m+[m			[32m# Get the length of the input tensor[m
[32m+[m			[32minput_text = input_texts[b].replace(self.tokenizer.pad_token, '')[m
[32m+[m			[32mtokens = self.tokenizer(input_text, return_tensors='pt')[m
[32m+[m			[32minput_ids = tokens.input_ids.to('cuda')[m
[32m+[m			[32mattention_mask = tokens.attention_mask.to('cuda')[m
[32m+[m			[32mseq_len = input_ids.shape[1][m
[32m+[m			[32mleft_padding_len = max_seq_len - seq_len[m
 			[m
[31m-			input_ids = tokens[b].input_ids[m
[31m-			attention_mask = tokens[b].attention_mask[m
[31m-			[m
[31m-			# Create masked versions of the sequence[m
[32m+[m			[32m# Create masked versions of the sequence for token contribution analysis[m
 			expanded_input_ids = input_ids.repeat_interleave(seq_len, dim=0)[m
[31m-			expanded_attention_mask = 1 - torch.eye(max_seq_len, dtype=attention_mask.dtype, device='cuda')[m
[31m-			masked_expanded_input_ids = torch.where(expanded_attention_mask == 1, expanded_input_ids, self.tokenizer.pad_token_id)[m
[31m-			[m
[31m-			# Expand the harm reward targets[m
[31m-			expanded_target_harm = target_harm[b]#.repeat(seq_len)[m
[31m-			expanded_action_harm = action_harm[b]#.repeat(seq_len)[m
[32m+[m			[32mexpanded_attention_mask = 1 - torch.eye(seq_len, dtype=attention_mask.dtype, device='cuda')[m
[32m+[m			[32mpadding = torch.full_like(expanded_input_ids[:, :1], self.tokenizer.pad_token_id, device='cuda')[m
[32m+[m			[32mmasked_expanded_input_ids = torch.where(expanded_attention_mask == 1, expanded_input_ids, padding)[m
 			[m
 			# Compute original (non-masked) sequence value for harm[m
[31m-			with torch.set_grad_enabled(True):	[m
[32m+[m			[32mwith torch.set_grad_enabled(True):[m[41m    [m
 				# Get original value[m
 				harm_value, original_harm_token_scores = self.harm_critic([m
 					input_ids=input_ids,[m
[36m@@ -504,79 +513,60 @@[m [mclass Agent:[m
 					return_token_scores=True[m
 				)[m
 				[m
[31m-				# Get expanded values[m
[31m-				#expanded_harm_values = [][m
[31m-				#expanded_harm_token_scores_list = [][m
[31m-				[m
[31m-				# Process expanded inputs[m
[31m-				for pos in range(0,seq_len, self.minibatch_size):[m
[32m+[m				[32m# Process expanded inputs in minibatches to save memory[m
[32m+[m				[32mfor pos in range(0, seq_len, self.minibatch_size):[m
[32m+[m					[32mend_pos = min(pos + self.minibatch_size, seq_len)[m
[32m+[m[41m					[m
 					# Get value when token at position pos is masked[m
[31m-					pos_harm_value, pos_harm_token_scores = self.harm_critic([m
[31m-						input_ids=masked_expanded_input_ids[pos:pos+self.minibatch_size],[m
[31m-						attention_mask=expanded_attention_mask[pos:pos+self.minibatch_size],[m
[32m+[m					[32mwindowed_harm_value, windowed_harm_token_scores = self.harm_critic([m
[32m+[m						[32minput_ids=masked_expanded_input_ids[pos:end_pos],[m
[32m+[m						[32mattention_mask=expanded_attention_mask[pos:end_pos],[m
 						return_token_scores=True[m
 					)[m
[32m+[m					[32m# Calculate token contributions as difference between original and masked values[m
[32m+[m					[32mwindowed_harm_token_contributions = harm_value.sum(dim=-1) - windowed_harm_value.sum(dim=-1)[m
[32m+[m					[32m# Normalize the token contributions into scores[m
[32m+[m					[32mwindowed_harm_token_sum = windowed_harm_token_contributions.sum()[m
[32m+[m					[32mif windowed_harm_token_sum != 0:[m
[32m+[m						[32mwindowed_harm_token_scores = windowed_harm_token_contributions / windowed_harm_token_sum * windowed_harm_value.sum()[m
[32m+[m					[32melse:[m
[32m+[m						[32mwindowed_harm_token_scores = windowed_harm_token_contributions[m
 					[m
[31m-					#expanded_harm_values.append(pos_harm_value)[m
[31m-					#expanded_harm_token_scores_list.append(pos_harm_token_scores)[m
[31m-				[m
[31m-					# Concatenate the expanded values[m
[31m-					#expanded_harm_values = torch.cat(expanded_harm_values)[m
[31m-					#minibatch_hexpanded_harm_token_scores_listarm_token_scores = torch.cat(expanded_harm_token_scores_list)[m
[31m-					[m
[31m-					# Initialize harm token contribution tensor[m
[31m-					# Shape: [1, max_seq_len][m
[31m-					harm_token_contributions = torch.zeros((1, max_seq_len), device='cuda')[m
[31m-					[m
[31m-					# Calculate token contributions[m
[31m-					harm_token_contributions[0, pos:pos+self.minibatch_size] = pos_harm_value - expanded_action_harm[pos:pos+self.minibatch_size][m
[31m-					[m
[31m-					# Apply the attention mask to zero out contributions for padding tokens[m
[31m-					harm_token_contributions = harm_token_contributions * attention_mask[pos:pos+self.minibatch_size].float()[m
[31m-					[m
[31m-					# Normalize the token contributions to sum to the original value[m
[31m-					harm_token_sum = harm_token_contributions.sum(dim=1, keepdim=True)[m
[31m-					[m
[31m-					# Avoid division by zero[m
[31m-					harm_token_sum = torch.where(harm_token_sum == 0, torch.ones_like(harm_token_sum), harm_token_sum)[m
[31m-					[m
[31m-					# Normalize to get token scores[m
[31m-					harm_token_scores = harm_token_contributions * (harm_value[pos:pos+self.minibatch_size].unsqueeze(1) / harm_token_sum)[m
[32m+[m					[32m# Store token scores[m
[32m+[m					[32mprint(f"Windowed harm token scores: {windowed_harm_token_scores.shape}")[m
[32m+[m					[32mprint(f"Harm token batch scores: {harm_token_scores_batch.shape}")[m
[32m+[m					[32mprint(f"Windowed harm token scores: {harm_token_scores_batch[b, pos:end_pos]}")[m
[32m+[m					[32mharm_token_scores_batch[b, pos:end_pos] = windowed_harm_token_scores.squeeze(-1).detach()[m
 					[m
[31m-					# Compute harm loss[m
[31m-					harm_pred = target_harm[b][pos:pos+self.minibatch_size] - action_harm[b][pos:pos+self.minibatch_size][m
[31m-					harm_loss = nn.MSELoss()(harm_value[pos:pos+self.minibatch_size], harm_pred)[m
[32m+[m					[32m# Compute harm ground truth loss based on our theory:[m[41m [m
[32m+[m					[32m# V_h(s,a) = R(response) - R(action) + γ*V_h(next_response)[m
[32m+[m					[32mharm_target = target_harm[b] - action_harm[b] + gamma * next_harm_values[b][m
[32m+[m					[32mharm_loss = nn.MSELoss()(harm_value.sum(), harm_target)[m
 					[m
[31m-					# Also compute loss for the expanded harm values[m
[31m-					expanded_harm_pred = (expanded_target_harm - expanded_action_harm).repeat(self.minibatch_size)[m
[31m-					expanded_harm_loss = nn.MSELoss()(pos_harm_value, expanded_harm_pred)[m
[32m+[m					[32m# Also compute regularization loss for token-level predictions[m
[32m+[m					[32mwindowed_harm_loss = nn.MSELoss()(windowed_harm_value, harm_value)[m
 					[m
[31m-					# Combine the original and expanded losses for harm. Acts as regularization for token-level predictions.[m
[31m-					# Divide by batch_size to prevent overfitting during training[m
[31m-					#TODO: perform ablation with and without expanded harm loss[m
[31m-					combined_harm_loss = (harm_loss + 0.5 * expanded_harm_loss) / batch_size[m
[32m+[m					[32m# Combine losses with regularization weight and batch normalization[m
[32m+[m					[32mcombined_harm_loss = (harm_loss + 0.1 * windowed_harm_loss) / batch_size[m
 					[m
[31m-					# Compute errors for advantage estimation (sequence level) for harm[m
[31m-					harm_error = harm_value - (target_harm[b] - action_harm[b])[m
[32m+[m					[32m# Compute errors for advantage estimation[m
[32m+[m					[32mharm_error = harm_value.sum() - harm_target[m
 					[m
[31m-					# Backward pass for this sequence[m
[32m+[m					[32m# Backward pass[m
 					combined_harm_loss.backward()[m
 					self.harm_critic_optim.step()[m
 					self.harm_critic.zero_grad()[m
 					[m
[31m-					# Store results for this sequence[m
[31m-					harm_values[b] = harm_value.detach()[m
[32m+[m					[32m# Store results[m
[32m+[m					[32mharm_values[b] = harm_value.sum().detach()[m
 					harm_errors[b] = harm_error.detach()[m
[31m-					harm_token_scores_batch[b, :seq_len] = harm_token_scores[0, :seq_len].detach()[m
[31m-			[m
[31m-			# Free up GPU memory[m
[31m-			del expanded_input_ids, expanded_attention_mask[m
[31m-			del expanded_harm_values, expanded_harm_token_scores_list[m
[31m-			del expanded_harm_value, expanded_harm_token_scores[m
[31m-			del harm_token_contributions, harm_token_sum, harm_token_scores[m
[31m-			del harm_value, original_harm_token_scores[m
[31m-			del harm_loss, expanded_harm_loss, combined_harm_loss, harm_error[m
[31m-			torch.cuda.empty_cache()[m
[32m+[m[41m		[m
[32m+[m		[32m# Free GPU memory[m
[32m+[m		[32mdel expanded_input_ids, expanded_attention_mask[m
[32m+[m		[32mdel expanded_harm_value, expanded_harm_token_scores[m
[32m+[m		[32mdel harm_token_contributions, harm_token_sum, harm_token_scores[m
[32m+[m		[32mdel harm_value, original_harm_token_scores[m
[32m+[m		[32mtorch.cuda.empty_cache()[m
 		[m
 		# Move harm critic to CPU to free GPU memory[m
 		self.harm_critic.to('cpu')[m
[36m@@ -589,49 +579,58 @@[m [mclass Agent:[m
 		# Move help critic to CUDA[m
 		self.help_critic.to('cuda')[m
 [m
[32m+[m		[32m# Prepare inputs for help critic[m
 		tokens = self.tokenizer(input_texts, return_tensors='pt', padding=True)[m
 		input_ids = tokens.input_ids.to('cuda')[m
[31m-		attention_mask = torch.where(input_ids == self.tokenizer.pad_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)).to('cuda')[m
[32m+[m		[32mattention_mask = torch.where(input_ids == self.tokenizer.pad_token_id,[m[41m [m
[32m+[m									[32mtorch.zeros_like(input_ids),[m[41m [m
[32m+[m									[32mtorch.ones_like(input_ids)).to('cuda')[m
[32m+[m[41m		[m
[32m+[m		[32m# Get next state values for bootstrapping[m
[32m+[m		[32mwith torch.no_grad():[m
[32m+[m			[32mnext_help_values = self.help_critic([m
[32m+[m				[32minput_ids=next_input_ids,[m
[32m+[m				[32mattention_mask=next_attention_mask[m
[32m+[m			[32m).detach()[m
[32m+[m[41m		[m
 		# Get helpfulness scores[m
[31m-		if action_help_reward is None:[m
[31m-			action_help = self.env.score_helpfulness(state_action_pairs).to('cuda')[m
[31m-		else:[m
[31m-			action_help = action_help_reward.to('cuda')[m
[31m-		target_help = help_reward.to('cuda')  # batch_compliance_rews[m
[32m+[m		[32maction_help = action_help_reward.to('cuda') if action_help_reward is not None else torch.zeros_like(help_reward).to('cuda')[m
[32m+[m		[32mtarget_help = help_reward.to('cuda')[m
 		[m
[31m-		# Concatenate help values and token scores[m
[32m+[m		[32m# Compute help values[m
 		help_value, help_token_scores = self.help_critic([m
 			input_ids=input_ids,[m
 			attention_mask=attention_mask,[m
 			return_token_scores=True[m
 		)[m
 		[m
[31m-		# Sum help losses[m
[31m-		help_loss = nn.MSELoss()(help_value, action_help + target_help)[m
[32m+[m		[32m# Compute help loss based on our theory:[m[41m [m
[32m+[m		[32m# V_help(s,a) = Help(action) + Help(response) + γ*V_help(next_response)[m
[32m+[m		[32mhelp_target = action_help + target_help + gamma * next_help_values[m
[32m+[m		[32mhelp_loss = nn.MSELoss()(help_value, help_target)[m
 		[m
[31m-		# Compute errors for advantage estimation (sequence level) for help[m
[31m-		help_errors = help_value - (action_help + target_help)[m
[31m-		[m
[31m-		print(f"Final harm token scores shape: {harm_token_scores_batch.shape}")[m
[31m-		print(f"Final help token scores shape: {help_token_scores.shape}")[m
[32m+[m		[32m# Compute errors for advantage estimation[m
[32m+[m		[32mhelp_errors = help_value - help_target[m
 		[m
[32m+[m		[32m# Backward pass[m
 		help_loss.backward()[m
 		self.help_critic_optim.step()[m
 		self.help_critic.zero_grad()[m
[31m-		self.help_critic_optim.zero_grad()[m
[31m-		help_loss.detach()[m
[31m-		help_errors.detach()[m
[31m-		help_token_scores.detach()[m
[31m-		help_value.detach()[m
[31m-[m
[32m+[m[41m		[m
[32m+[m		[32m# Clean up[m
[32m+[m		[32mhelp_loss = help_loss.detach()[m
[32m+[m		[32mhelp_errors = help_errors.detach()[m
[32m+[m		[32mhelp_token_scores = help_token_scores.detach()[m
[32m+[m		[32mhelp_value = help_value.detach()[m
 		self.help_critic.to('cpu')[m
 		torch.cuda.empty_cache()[m
 [m
[31m-		# Return token-level scores along with sequence-level values and errors[m
[32m+[m		[32m# Return values, errors, losses, and token scores[m
 		return ((harm_values, help_value), [m
 				(harm_errors, help_errors), [m
 				(harm_loss, help_loss),[m
[31m-				(harm_token_scores_batch, help_token_scores))  # Token-level scores based on contribution analysis[m
[32m+[m				[32m(harm_token_scores_batch, help_token_scores),[m
[32m+[m				[32m(action_harm, action_help))  # Also return action scores for logging[m
 	[m
 	def scores_to_values(self, scores):[m
 		"""[m
[36m@@ -647,49 +646,60 @@[m [mclass Agent:[m
 		sequence_values = scores.sum(dim=-1)[m
 		return sequence_values[m
 	[m
[31m-	def test(self, num_episodes=10):[m
[32m+[m	[32mdef test(self, dataset_name, num_samples=1000, batch_size=8, adapter_types=None):[m
 		"""[m
 		Test the agent in the given environment.[m
 [m
[31m-			Args:[m
[31m-				num_episodes (int): The number of episodes to run for testing.[m
[32m+[m		[32mArgs:[m
[32m+[m			[32mdataset_name (str): HuggingFace dataset name to use for evaluation[m
[32m+[m			[32mnum_samples (int): Number of samples to evaluate[m
[32m+[m			[32mbatch_size (int): Batch size for processing[m
[32m+[m			[32madapter_types (list, optional): List of adapter types to test with. If provided,[m
[32m+[m				[32mthe method will run the evaluation for each adapter type separately.[m
[32m+[m				[32mExample: ['lm_head_rephrase', 'attention_rephrase', 'lm_head_followup', 'attention_followup'][m
 [m
[31m-			Returns:[m
[31m-				list: A list of rewards obtained in each episode.[m
[32m+[m		[32mReturns:[m
[32m+[m			[32mdict: Dictionary containing evaluation metrics for each adapter type if adapter_types is provided,[m
[32m+[m				[32motherwise metrics for the current model configuration.[m
 		"""[m
[31m-		rewards = [][m
[31m-		print("Testing... iterations: ", num_episodes)[m
[31m-		questions_list = [][m
[31m-		rephrase_list = [][m
[31m-		response_list = [][m
[31m-[m
[31m-		for episode in range(num_episodes):[m
[31m-			prompt = self.env.reset()[m
[31m-			episode_reward = 0[m
[31m-			done = False[m
[31m-[m
[31m-			while not done:[m
[31m-				# Take action in the environment[m
[31m-				question, no_rephrase_reward, action, rephrase_reward, reward, done, response = self.env.evaluate(self.actor)[m
[31m-				questions_list.extend(question)[m
[31m-				rephrase_list.extend(action)[m
[31m-				response_list.extend(response)[m
[31m-[m
[31m-				# Accumulate reward[m
[31m-				episode_reward += reward[m
[31m-[m
[31m-				prompt = question[m
[31m-				print(f"Question: {question}")[m
[31m-				print(f"Rephrase: {action}")[m
[31m-				print(f"Reward: {rephrase_reward}")[m
[31m-[m
[31m-[m
[31m-			# Store the episode reward[m
[31m-			rewards.append(episode_reward)[m
[31m-[m
[31m-			print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")[m
[31m-		test_chat_with_guard(questions_list, rephrase_list, response_list)[m
[31m-		return rewards[m
[32m+[m		[32mif not adapter_types or not self.use_lora:[m
[32m+[m			[32m# Run evaluation with current model configuration[m
[32m+[m			[32mresults = self.evaluate_red_teaming(dataset_name, num_samples=num_samples, batch_size=batch_size)[m
[32m+[m			[32mprint(results)[m
[32m+[m			[32mreturn results[m
[32m+[m[41m		[m
[32m+[m		[32m# Test with each adapter type separately[m
[32m+[m		[32mall_results = {}[m
[32m+[m		[32moriginal_adapter = self.actor.adapter_name if hasattr(self.actor, 'adapter_name') else None[m
[32m+[m[41m		[m
[32m+[m		[32mtry:[m
[32m+[m			[32mfor adapter_type in adapter_types:[m
[32m+[m				[32mprint(f"\n=== Testing with adapter: {adapter_type} ===")[m
[32m+[m[41m				[m
[32m+[m				[32m# Switch to the current adapter[m
[32m+[m				[32msuccess = self.switch_lora_adapter(adapter_type)[m
[32m+[m				[32mif not success:[m
[32m+[m					[32mprint(f"Warning: Failed to load adapter {adapter_type}, skipping...")[m
[32m+[m					[32mcontinue[m
[32m+[m[41m				[m
[32m+[m				[32m# Run evaluation with this adapter[m
[32m+[m				[32madapter_results = self.evaluate_red_teaming([m
[32m+[m					[32mdataset_name,[m[41m [m
[32m+[m					[32mnum_samples=num_samples,[m[41m [m
[32m+[m					[32mbatch_size=batch_size[m
[32m+[m				[32m)[m
[32m+[m[41m				[m
[32m+[m				[32m# Store results for this adapter[m
[32m+[m				[32mall_results[adapter_type] = adapter_results[m
[32m+[m				[32mprint(f"Results for {adapter_type}:")[m
[32m+[m				[32mprint(adapter_results)[m
[32m+[m		[32mfinally:[m
[32m+[m			[32m# Restore original adapter if there was one[m
[32m+[m			[32mif original_adapter:[m
[32m+[m				[32mprint(f"Restoring original adapter: {original_adapter}")[m
[32m+[m				[32mself.switch_lora_adapter(original_adapter)[m
[32m+[m[41m		[m
[32m+[m		[32mreturn all_results[m
 	[m
 	def setup_prompt(self, prompt, system_instructions, redteam_instructions):[m
 		return [[m
[36m@@ -1093,6 +1103,11 @@[m [mclass Agent:[m
 		'adv_coef': 0.1,[m
 		'help_weight': 0.5,  # Add this line[m
 		'detune': False,  # Default to not using detuned models[m
[32m+[m		[32m'target_model': None,  # Default to using the shared base model[m
[32m+[m		[32m'use_lora': False,  # Default to not using LoRA[m
[32m+[m		[32m'lora_r': 8,  # LoRA rank[m
[32m+[m		[32m'lora_alpha': 16,  # LoRA alpha[m
[32m+[m		[32m'lora_dropout': 0.05,  # LoRA dropout[m
     	}[m
 		# Change any default values to custom values for specified hyperparameters[m
 		for param, val in default_hyperparameters.items():[m
[36m@@ -1109,13 +1124,174 @@[m [mclass Agent:[m
 [m
 			# Set the seed [m
 			torch.manual_seed(self.seed)[m
[31m-			print(f"Successfully set seed to {self.seed}")[m
[32m+[m			[32mnp.random.seed(self.seed)[m
[32m+[m[41m			[m
[32m+[m	[32mdef load_lora_adapter(self, adapter_type):[m
[32m+[m		[32m"""[m
[32m+[m		[32mLoad a specific LoRA adapter.[m
[32m+[m[41m		[m
[32m+[m		[32mArgs:[m
[32m+[m			[32madapter_type: Type of adapter to load ('lm_head_rephrase', 'attention_rephrase',[m[41m [m
[32m+[m												[32m  'lm_head_followup', 'attention_followup')[m
[32m+[m[41m		[m
[32m+[m		[32mReturns:[m
[32m+[m			[32mbool: Whether the adapter was successfully loaded[m
[32m+[m		[32m"""[m
[32m+[m		[32mif not self.use_lora:[m
[32m+[m			[32mprint("LoRA is not enabled")[m
[32m+[m			[32mreturn False[m
[32m+[m[41m			[m
[32m+[m		[32mif adapter_type not in self.lora_adapters:[m
[32m+[m			[32mprint(f"Unknown adapter type: {adapter_type}")[m
[32m+[m			[32mreturn False[m
[32m+[m[41m			[m
[32m+[m		[32madapter_path = self.lora_adapters[adapter_type][m
[32m+[m[41m		[m
[32m+[m		[32m# Check if adapter exists[m
[32m+[m		[32mif not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):[m
[32m+[m			[32mprint(f"Adapter {adapter_type} does not exist yet, will be created during training")[m
[32m+[m[41m			[m
[32m+[m			[32m# Apply fresh LoRA adapter with the appropriate config[m
[32m+[m			[32mself.actor.apply_lora_adapters(self.lora_configs[adapter_type], adapter_name=adapter_type)[m
[32m+[m			[32mreturn True[m
[32m+[m[41m			[m
[32m+[m		[32m# Load existing adapter[m
[32m+[m		[32mprint(f"Loading LoRA adapter: {adapter_type}")[m
[32m+[m		[32mreturn self.actor.load_lora_adapters(adapter_path)[m
[32m+[m[41m		[m
[32m+[m	[32mdef save_lora_adapter(self, adapter_type):[m
[32m+[m		[32m"""[m
[32m+[m		[32mSave the current LoRA adapter.[m
[32m+[m[41m		[m
[32m+[m		[32mArgs:[m
[32m+[m			[32madapter_type: Type of adapter to save ('lm_head_rephrase', 'attention_rephrase',[m[41m [m
[32m+[m												[32m  'lm_head_followup', 'attention_followup')[m
[32m+[m		[32m"""[m
[32m+[m		[32mif not self.use_lora:[m
[32m+[m			[32mprint("LoRA is not enabled")[m
[32m+[m			[32mreturn[m
[32m+[m[41m			[m
[32m+[m		[32mif adapter_type not in self.lora_adapters:[m
[32m+[m			[32mprint(f"Unknown adapter type: {adapter_type}")[m
[32m+[m			[32mreturn[m
[32m+[m[41m			[m
[32m+[m		[32madapter_path = self.lora_adapters[adapter_type][m
[32m+[m		[32mself.actor.save_lora_adapters(adapter_path)[m
[32m+[m[41m		[m
[32m+[m	[32mdef switch_lora_adapter(self, adapter_type):[m
[32m+[m		[32m"""[m
[32m+[m		[32mSwitch to a different LoRA adapter.[m
[32m+[m[41m		[m
[32m+[m		[32mArgs:[m
[32m+[m			[32madapter_type: Type of adapter to switch to[m
 			[m
[31m-		# Log whether detuning is enabled[m
[31m-		if self.detune:[m
[31m-			print("Detuning is enabled - will use detuned model weights if available")[m
[32m+[m		[32mReturns:[m
[32m+[m			[32mbool: Whether the switch was successful[m
[32m+[m		[32m"""[m
[32m+[m		[32mif not self.use_lora:[m
[32m+[m			[32mprint("LoRA is not enabled")[m
[32m+[m			[32mreturn False[m
[32m+[m[41m			[m
[32m+[m		[32m# Save current adapter if it exists[m
[32m+[m		[32mif hasattr(self.actor, 'adapter_name') and self.actor.adapter_name:[m
[32m+[m			[32mself.save_lora_adapter(self.actor.adapter_name)[m
[32m+[m[41m			[m
[32m+[m		[32m# Recreate the actor with the base model[m
[32m+[m		[32mself.to_cpu()[m
[32m+[m[41m		[m
[32m+[m		[32m# Check if adapter exists[m
[32m+[m		[32madapter_path = self.lora_adapters[adapter_type][m
[32m+[m		[32madapter_config_path = os.path.join(adapter_path, "adapter_config.json")[m
[32m+[m[41m		[m
[32m+[m		[32mif not os.path.exists(adapter_config_path):[m
[32m+[m			[32mprint(f"Adapter {adapter_type} does not exist at {adapter_path}")[m
[32m+[m[41m			[m
[32m+[m			[32m# During training, we create new adapters[m
[32m+[m			[32mif self.train:[m
[32m+[m				[32mprint(f"Creating new adapter {adapter_type} for training")[m
[32m+[m				[32m# Reinitialize the actor with the new adapter[m
[32m+[m				[32mpolicy_class = getPolicyClass('Transformer')[m
[32m+[m				[32mself.actor = policy_class([m
[32m+[m					[32mself.checkpoint,[m[41m [m
[32m+[m					[32mself.tokenizer,[m[41m [m
[32m+[m					[32mmodel=self.base_model.model,[m
[32m+[m					[32mtoken=None,[m
[32m+[m					[32muse_lora=True,[m
[32m+[m					[32mlora_config=self.lora_configs[adapter_type],[m
[32m+[m					[32madapter_name=adapter_type[m
[32m+[m				[32m)[m
[32m+[m				[32msuccess = True[m
[32m+[m			[32melse:[m
[32m+[m				[32m# During testing, we don't create new adapters[m
[32m+[m				[32mprint(f"Skipping non-existent adapter {adapter_type} during testing")[m
[32m+[m				[32mreturn False[m
 		else:[m
[31m-			print("Detuning is disabled - will use original model weights")[m
[32m+[m			[32m# Adapter exists, load it[m
[32m+[m			[32m# Reinitialize the actor with the new adapter[m
[32m+[m			[32mpolicy_class = getPolicyClass('Transformer')[m
[32m+[m			[32mself.actor = policy_class([m
[32m+[m				[32mself.checkpoint,[m[41m [m
[32m+[m				[32mself.tokenizer,[m[41m [m
[32m+[m				[32mmodel=self.base_model.model,[m
[32m+[m				[32mtoken=None,[m
[32m+[m				[32muse_lora=True,[m
[32m+[m				[32mlora_config=self.lora_configs[adapter_type],[m
[32m+[m				[32madapter_name=adapter_type[m
[32m+[m			[32m)[m
[32m+[m[41m			[m
[32m+[m			[32m# Load the adapter[m
[32m+[m			[32msuccess = self.load_lora_adapter(adapter_type)[m
[32m+[m[41m		[m
[32m+[m		[32mself.to_cuda()[m
[32m+[m		[32mreturn success[m
[32m+[m[41m		[m
[32m+[m	[32mdef train_with_adapter(self, adapter_type, batch_data, num_epochs=None):[m
[32m+[m		[32m"""[m
[32m+[m		[32mTrain the model with a specific adapter.[m
[32m+[m[41m		[m
[32m+[m		[32mArgs:[m
[32m+[m			[32madapter_type: Type of adapter to use for training[m
[32m+[m			[32mbatch_data: Data to train on[m
[32m+[m			[32mnum_epochs: Number of epochs to train for (defaults to self.epochs)[m
[32m+[m		[32m"""[m
[32m+[m		[32mif not self.use_lora:[m
[32m+[m			[32mprint("LoRA is not enabled, training without adapters")[m
[32m+[m			[32mreturn self._train_on_batch(batch_data, num_epochs)[m
[32m+[m[41m			[m
[32m+[m		[32m# Switch to the appropriate adapter[m
[32m+[m		[32mself.switch_lora_adapter(adapter_type)[m
[32m+[m[41m		[m
[32m+[m		[32m# Train on the batch[m
[32m+[m		[32mreturn self._train_on_batch(batch_data, num_epochs)[m
[32m+[m[41m		[m
[32m+[m	[32mdef _train_on_batch(self, batch_data, num_epochs=None):[m
[32m+[m		[32m"""[m
[32m+[m		[32mInternal method to train on a batch of data.[m
[32m+[m[41m		[m
[32m+[m		[32mArgs:[m
[32m+[m			[32mbatch_data: Dictionary containing batch data[m
[32m+[m			[32mnum_epochs: Number of epochs to train for[m
[32m+[m		[32m"""[m
[32m+[m		[32m# Use default epochs if not specified[m
[32m+[m		[32mepochs = num_epochs if num_epochs is not None else self.epochs[m
[32m+[m[41m		[m
[32m+[m		[32m# Extract batch data[m
[32m+[m		[32mbatch_state = batch_data['batch_state'][m
[32m+[m		[32mbatch_acts = batch_data['batch_acts'][m
[32m+[m		[32mbatch_log_probs = batch_data['batch_log_probs'][m
[32m+[m		[32mbatch_advantages = batch_data['batch_advantages'][m
[32m+[m		[32mbatch_values = batch_data.get('batch_values', None)[m
[32m+[m[41m		[m
[32m+[m		[32m# Implement the training loop here[m
[32m+[m		[32m# This will depend on your specific training algorithm (PPO, etc.)[m
[32m+[m		[32m# For now, this is a placeholder[m
[32m+[m[41m		[m
[32m+[m		[32mreturn {[m
[32m+[m			[32m'loss': 0.0,[m
[32m+[m			[32m'policy_loss': 0.0,[m
[32m+[m			[32m'value_loss': 0.0,[m
[32m+[m			[32m'entropy': 0.0[m
[32m+[m		[32m}[m
 [m
 	def _log_summary(self):[m
 		"""[m
[36m@@ -1328,161 +1504,14 @@[m [mclass Agent:[m
 			self.harm_critic.to('cpu')[m
 			self.help_critic.to('cpu')[m
 [m
[31m-	def detune(self, num_epochs=5, batch_size=10, learning_rate=1e-5):[m
[31m-		"""[m
[31m-		Detunes the safety alignment of the model using PKU-SafeRLHF dataset.[m
[31m-		Applies a negative Bradley-Terry reward model to reverse RLHF safety training.[m
[31m-		[m
[31m-		Args:[m
[31m-			num_epochs (int): Number of epochs to train for[m
[31m-			batch_size (int): Batch size for training[m
[31m-			learning_rate (float): Learning rate for optimizer[m
[31m-		"""[m
[31m-		# Check if detuning is enabled[m
[31m-		if not hasattr(self, 'detune') or not self.detune:[m
[31m-			print("Detuning is disabled. Skipping detuning process.")[m
[31m-			return[m
[31m-			[m
[31m-		from datasets import load_dataset[m
[31m-		import torch.nn.functional as F[m
[31m-		import os[m
[31m-		model_dir = f"models/{self.model_name}"[m
[31m-		flags_file = os.path.join(model_dir, "flags")[m
[31m-		if os.path.exists(flags_file):[m
[31m-			with open(flags_file, 'r') as file:[m
[31m-				if "detuned" in file.read():[m
[31m-					print("Model is already detuned. Skipping detuning process.")[m
[31m-					return[m
[31m-			[m
[31m-		print("Starting safety detuning process...")[m
[31m-[m
[31m-		# Load the PKU-SafeRLHF dataset[m
[31m-		full_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")[m
[31m-		print(f"Loaded dataset with {len(full_dataset)} training examples")[m
[31m-[m
[31m-		# Unfreeze all layers in the model[m
[31m-		self.actor.unfreeze_layers()[m
[31m-[m
[31m-		# Create optimizer with all parameters[m
[31m-		optimizer = torch.optim.Adam(self.actor.model.parameters(), lr=learning_rate)[m
[31m-[m
[31m-		# Track loss for logging[m
[31m-		total_loss = 0[m
[31m-		num_batches = 0[m
[31m-[m
[31m-		# Training loop[m
[31m-		self.actor.model.to('cuda')[m
[31m-		self.actor.model.train()[m
[31m-[m
[31m-		for epoch in range(num_epochs):[m
[31m-			offset = epoch * 100[m
[31m-			dataset = full_dataset.select(range(offset,100+offset))[m
[31m-			print(f"Epoch {epoch+1}/{num_epochs}: {len(dataset)} examples")[m
[31m-			[m
[31m-			# Shuffle dataset for each epoch[m
[31m-			shuffled_indices = torch.randperm(len(dataset))[m
[31m-			[m
[31m-			for i in range(0, len(dataset), batch_size):[m
[31m-				# Get batch indices[m
[31m-				batch_indices = shuffled_indices[i:i+batch_size][m
[31m-				if len(batch_indices) < 2:  # Need at least 2 examples for comparison[m
[31m-					continue[m
[31m-					[m
[31m-				# Get batch data[m
[31m-				batch = [dataset[idx.item()] for idx in batch_indices][m
[31m-				[m
[31m-				# Extract prompts and responses[m
[31m-				prompts = [item['prompt'] for item in batch][m
[31m-				[m
[31m-				# Determine safe and unsafe responses based on safer_response_id[m
[31m-				safe_responses = [][m
[31m-				unsafe_responses = [][m
[31m-				[m
[31m-				for item in batch:[m
[31m-					safer_id = item['safer_response_id'][m
[31m-					# The safer response is indicated by safer_response_id[m
[31m-					# The other response is the less safe one[m
[31m-					if safer_id == 0:[m
[31m-						safe_responses.append(item['response_0'])[m
[31m-						unsafe_responses.append(item['response_1'])[m
[31m-					else:[m
[31m-						safe_responses.append(item['response_1'])[m
[31m-						unsafe_responses.append(item['response_0'])[m
[31m-				[m
[31m-				# Tokenize inputs[m
[31m-				inputs = self.tokenizer([m
[31m-					[p + r for p, r in zip(prompts, safe_responses)], [m
[31m-					return_tensors='pt', [m
[31m-					padding=True, [m
[31m-					truncation=True,[m
[31m-					max_length=512[m
[31m-				).input_ids.to('cuda')[m
[31m-				[m
[31m-				safe_responses = self.tokenizer(safe_responses, return_tensors='pt', padding=True).input_ids.to('cuda')[m
[31m-				unsafe_responses = self.tokenizer(unsafe_responses, return_tensors='pt', padding=True).input_ids.to('cuda')[m
[31m-				[m
[31m-				# Calculate log probabilities using the existing method[m
[31m-				safe_log_probs = self.actor.get_log_probs(inputs, safe_responses, sequence_level=True)[m
[31m-				unsafe_log_probs = self.actor.get_log_probs(inputs, unsafe_responses, sequence_level=True)[m
[31m-				[m
[31m-				# Bradley-Terry loss (negative to reverse the preference)[m
[31m-				# We want unsafe > safe (reverse of the original safety alignment)[m
[31m-				loss = -F.logsigmoid(unsafe_log_probs - safe_log_probs).mean()[m
[31m-				[m
[31m-				# Backward and optimize[m
[31m-				optimizer.zero_grad()[m
[31m-				loss.backward()[m
[31m-				optimizer.step()[m
[31m-				[m
[31m-				total_loss += loss.item()[m
[31m-				num_batches += 1[m
[31m-				[m
[31m-				if i % 100 == 0:[m
[31m-					print(f"  Batch {i//batch_size}, Loss: {loss.item():.4f}")[m
[31m-		[m
[31m-			# End of epoch[m
[31m-			avg_loss = total_loss / max(num_batches, 1)[m
[31m-			print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")[m
[31m-		[m
[31m-		# Restore original parameter freezing if needed for subsequent training[m
[31m-		self.actor.freeze_layers()[m
[31m-		[m
[31m-		print("Safety detuning complete!")[m
[31m-		self._save()[m
[31m-		[m
[31m-		# Write the detuned flag to the flags file[m
[31m-		with open(flags_file, 'w') as file:[m
[31m-			file.write("detuned")[m
[31m-		[m
[31m-		# Also save to the detuned folder[m
[31m-		model_prefix = self.model_name.split('-')[0]  # Extract prefix (e.g., "llama" from "llama-ppo")[m
[31m-		detuned_path = f'./models/{model_prefix}-detuned'[m
[31m-		[m
[31m-		# Create the detuned directory if it doesn't exist[m
[31m-		if not os.path.exists(detuned_path):[m
[31m-			os.makedirs(detuned_path)[m
[31m-		[m
[31m-		# Save the actor model to the detuned folder[m
[31m-		torch.save(self.actor.model.state_dict(), f'{detuned_path}/actor.pt')[m
[31m-		[m
[31m-		# Save a flag file indicating this is a detuned model[m
[31m-		with open(f'{detuned_path}/flags', 'w') as file:[m
[31m-			file.write("detuned")[m
[31m-		[m
[31m-		print(f"Saved detuned model to {detuned_path}")[m
[31m-		[m
[31m-		# Return model to CPU to save memory[m
[31m-		self.actor.model.to('cpu')[m
[31m-		torch.cuda.empty_cache()[m
[31m-[m
[31m-	def evaluate_red_teaming(self, dataset_name, split="test", num_samples=100, batch_size=8):[m
[32m+[m	[32mdef evaluate_red_teaming(self, dataset_name, split="train", num_samples=10, batch_size=1):[m
 		"""[m
 		Evaluates the effectiveness of the red-teaming model by comparing harmfulness[m
 		of target LLM responses to original prompts vs. red-teamed prompts in a chat setting.[m
 		[m
 		Args:[m
 			dataset_name (str): HuggingFace dataset name to use for evaluation[m
[31m-			split (str): Dataset split to use (default: "test")[m
[32m+[m			[32msplit (str): Dataset split to use (default: "train")[m
 			num_samples (int): Number of samples to evaluate (default: 100)[m
 			batch_size (int): Batch size for processing (default: 8)[m
 			[m
[36m@@ -1494,6 +1523,7 @@[m [mclass Agent:[m
 		import pandas as pd[m
 		from tqdm import tqdm[m
 		import json[m
[32m+[m		[32mimport traceback[m
 		[m
 		print(f"Starting red-teaming evaluation using {dataset_name}...")[m
 		[m
[36m@@ -1515,124 +1545,175 @@[m [mclass Agent:[m
 			'original_harm_scores': [],[m
 			'red_teamed_harm_scores': [],[m
 			'original_refusals': [],[m
[31m-			'red_teamed_refusals': [][m
[32m+[m			[32m'red_teamed_refusals': [],[m
[32m+[m			[32m'red_teamed_chat_scores': [][m
 		}[m
 		[m
 		# Process in batches[m
[31m-		for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating batches"):[m
[31m-			batch = dataset[i:i+batch_size][m
[31m-			original_prompts = [item[prompt_field] for item in batch][m
[31m-			[m
[31m-			# Step 1: Get baseline responses and scores using the environment[m
[31m-			original_responses = [][m
[31m-			original_harm_scores = [][m
[31m-			original_refusals = [][m
[31m-			[m
[31m-			for prompt in original_prompts:[m
[32m+[m		[32mself.env.set_batch_size(batch_size)[m
[32m+[m		[32mtry:[m
[32m+[m			[32mfor i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating batches"):[m
[32m+[m				[32mbatch = dataset[i:i+batch_size][m
[32m+[m				[32mif len(batch) < batch_size:[m
[32m+[m					[32mcontinue[m
[32m+[m				[32moriginal_prompts = batch[prompt_field][m
[32m+[m[41m				[m
[32m+[m				[32m# Step 1: Get baseline responses and scores using the environment[m
[32m+[m				[32moriginal_responses = [][m
[32m+[m				[32moriginal_harm_scores = [][m
[32m+[m				[32moriginal_refusals = [][m
[32m+[m[41m				[m
 				# Reset environment with the original prompt[m
 				self.env.reset()[m
[31m-				self.env.set_state(prompt)[m
[32m+[m				[32mself.env.set_state(original_prompts)[m
 				done = False[m
[31m-				chat_history = [][m
[31m-				[m
[31m-				# Step through the environment until done[m
[31m-				while not done:[m
[31m-					# Get response from target model[m
[31m-					state, rewards, done = self.env.step(chat_history)[m
[31m-					response = state[-1][m
[31m-					chat_history.append(response)[m
[31m-				[m
[31m-				# Get the final response for evaluation[m
[31m-				final_response = chat_history[-1] if chat_history else ""[m
[31m-				original_responses.append(final_response)[m
[32m+[m				[32mchat_history = [q[0] for q in self.env.state ][m
[32m+[m
[32m+[m				[32m# Get response from target model[m
[32m+[m				[32massert isinstance(chat_history, list) and isinstance(chat_history[0], str), f"chat_history is not a list of strings: {chat_history}"[m
[32m+[m				[32mstate, rewards, done = self.env.step(chat_history)[m
[32m+[m				[32mresponse = [s[-1] for s in state][m
[32m+[m				[32mchat_history.append(response)[m
[32m+[m
[32m+[m				[32moriginal_responses.append(response)[m
 				[m
 				# Score the response[m
[31m-				harm_score = self.en