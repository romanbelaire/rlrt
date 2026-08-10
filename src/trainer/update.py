"""
Trainer update mixins for actor, critic, and high-level policy optimization.
"""

from __future__ import annotations

import gc
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trainer.logging_utils import log_episode_outputs
from src.trainer.rollout import collect_rollout
from src.trainer.utils import compute_kl_divergence_from_base_model


class TrainerUpdateMixin:
    """Provides training routines shared by AgentTrainer subclasses."""

    # Methods will be populated as we peel them from AgentTrainer.
    def _bellman_v_high_targets(
        self,
        sequence_critic: nn.Module,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        dones: torch.Tensor,
        tokenizer: Any,
    ) -> torch.Tensor:
        """V_high TD targets: r + γ V_tgt(s') (1 - done). Uses polyak target V head when available."""
        device = next(sequence_critic.parameters()).device
        model_dtype = next(sequence_critic.parameters()).dtype
        gamma = float(self.config.training.gamma)
        next_state = next_state.to(device)
        reward = reward.to(device=device, dtype=model_dtype)
        # dones may be float (0.0/1.0) from rollout; ~ only works on bool/int
        not_done = (1.0 - dones.to(device=device, dtype=model_dtype))
        next_attention = (next_state != tokenizer.pad_token_id).long().to(device=device)
        use_target = hasattr(sequence_critic, "target_state_value_head")
        v_next = sequence_critic.v_value(
            input_ids=next_state,
            attention_mask=next_attention,
            requires_grad=False,
            use_target_head=use_target,
        )
        if v_next.dim() > 1 and v_next.size(-1) == 1:
            v_next = v_next.squeeze(-1)
        v_next = v_next.to(dtype=model_dtype)
        return reward + gamma * v_next * not_done

    def _use_bellman_v_high(self) -> bool:
        return bool(
            getattr(self.config.training, 'use_bellman_v_high', False)
            or getattr(self.config.training, 'use_bellman_q_high', False)
        )

    def _train_v_high_only(self) -> bool:
        """γ ablation / q1_only: fit only V head on Bellman targets (skip Q head)."""
        return bool(getattr(self.config.training, "q1_only_training", False))

    def _normalize_reward_tensor(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Optionally Z-score normalize rewards with running EMA stats."""
        if not getattr(self.config.training, "normalize_rewards", False):
            return x
        return self.reward_stats.normalize(x, update=update)

    def _polyak_update_sequence_critic(self, sequence_critic: nn.Module) -> None:
        """Update sequence-critic target heads after an optimizer step."""
        if not hasattr(sequence_critic, "polyak_update_target_heads"):
            return
        tau = float(getattr(self.config.training, "critic_target_tau", 0.005))
        if tau <= 0.0:
            freq = int(getattr(self.config.training, "critic_target_update_freq", 1))
            self._sequence_critic_target_updates += 1
            if self._sequence_critic_target_updates % max(freq, 1) == 0:
                sequence_critic.hard_update_target_heads()
        else:
            sequence_critic.polyak_update_target_heads(tau=tau)

    def train_critics(
        self,
        sequence_critic: nn.Module,
        token_critic: nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        base_state: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        environment: Any = None,
        tokenizer: Any = None,
        instructions: Optional[List[str]] = None,
        v_low_targets: Optional[torch.Tensor] = None,
        l3_rewards: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Train both sequence-level and token-level critics.
        
        Args:
            sequence_critic: Sequence-level critic network
            token_critic: Token-level critic network
            state: Current state batch
            action: Action batch
            base_state: Base state batch
            reward: Reward batch
            next_state: Next state batch
            environment: Environment for computing Q targets
            tokenizer: Tokenizer for processing
            instructions: Optional instruction strings for instruction-following reward
            dones: Episode termination flags (required when use_bellman_v_high)
            
        Returns:
            Dictionary of losses
        """
        use_bellman = self._use_bellman_v_high()
        train_v_high_only = self._train_v_high_only()
        if use_bellman and dones is None:
            raise ValueError("dones is required when use_bellman_v_high is enabled")
        if dones is None:
            dones = torch.zeros_like(reward, dtype=torch.bool)

        # Compute targets for both critics
        if environment is not None:
            with torch.no_grad():
                if getattr(environment, 'uses_tool_step_rewards', False):
                    q_high_targets = reward
                    if v_low_targets is None and l3_rewards is not None:
                        action_attention_mask = (action != tokenizer.pad_token_id).long()
                        token_counts = action_attention_mask.sum(dim=1).clamp(min=1)
                        l3_rews = l3_rewards.to(reward.device)
                        v_low_targets = (
                            l3_rews.unsqueeze(1).float() / token_counts.unsqueeze(1).float()
                        ) * action_attention_mask
                elif getattr(environment, 'uses_episode_harm_reward', False):
                    q_high_targets = reward
                    if v_low_targets is None:
                        action_attention_mask = (action != tokenizer.pad_token_id).long()
                        token_counts = action_attention_mask.sum(dim=1).clamp(min=1)
                        v_low_targets = (
                            reward.unsqueeze(1).float() / token_counts.unsqueeze(1).float()
                        ) * action_attention_mask
                else:
                    if use_bellman:
                        # Step reward from rollout; Bellman bootstrap applied below
                        q_high_targets = reward
                    else:
                        q_high_targets = self.q_high(state, action, base_state, environment)

                    # Token-level targets: Use provided targets if available, otherwise compute analytically
                    if v_low_targets is None:
                        if token_critic is None:
                            v_low_targets = reward
                        else:
                            marginal_rewards = self.compute_marginal_rewards(
                                state, action, base_state, environment, tokenizer,
                                sequence_critic=sequence_critic, instructions=instructions
                            )
                            v_low_targets = self.v_low(marginal_rewards)

                if use_bellman:
                    # Normalize step reward then form TD: y = r + γ V_tgt(s')(1-d)
                    reward_for_td = self._normalize_reward_tensor(reward)
                    q_high_targets = self._bellman_v_high_targets(
                        sequence_critic, reward_for_td, next_state, dones, tokenizer
                    )
                else:
                    q_high_targets = self._normalize_reward_tensor(q_high_targets)
                    if v_low_targets is not None and v_low_targets.dim() >= 1:
                        # Keep token returns consistent if they came as scalar reward fallbacks
                        pass
        else:
            # Default targets to reward if no environment provided
            q_high_targets = reward
            if use_bellman:
                reward_for_td = self._normalize_reward_tensor(reward)
                q_high_targets = self._bellman_v_high_targets(
                    sequence_critic, reward_for_td, next_state, dones, tokenizer
                )
            else:
                q_high_targets = self._normalize_reward_tensor(q_high_targets)
            if v_low_targets is None:
                v_low_targets = reward
        
        # Ensure targets are on the correct device
        device = next(sequence_critic.parameters()).device
        model_dtype = next(sequence_critic.parameters()).dtype  # Get model's dtype for precision consistency
        q_high_targets = q_high_targets.to(device=device, dtype=model_dtype)
        if v_low_targets is None:
            v_low_targets = reward
        v_low_targets = v_low_targets.to(device=device, dtype=model_dtype)
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # [AUGMENTATION] ALTERNATIVE: use instruction-following reward to boost
        # value estimates.
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        use_task_reward = getattr(self.config.training, 'use_task_reward', True)
        if instructions is not None and use_task_reward:
            # Boost value estimates with instruction-following reward
            # if we have the instruction annotation.
            utter_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
            # Build the "safe" instruction - filter out None values and convert to empty strings for flat agents
            safe_instr = [instr if instr is not None and isinstance(instr, str) else "" for instr in instructions]
            
            # Skip instruction-following reward if all instructions are empty (flat agent case)
            if not all(not instr.strip() for instr in safe_instr):
                # Compute similarity between actions and instructions
                sim_scores = environment._get_embed_similarity_batched(
                    safe_instr,
                    utter_texts,
                ).to(device=device, dtype=model_dtype)  # Ensure correct device and dtype

                # For per-token RL, distribute task reward across valid tokens
                if v_low_targets.dim() == 2:  # Per-token targets [batch_size, action_len]
                    action_attention_mask = (action != tokenizer.pad_token_id).to(device=device)
                    for batch_idx in range(action.size(0)):
                        valid_positions = action_attention_mask[batch_idx].nonzero(as_tuple=True)[0]
                        if len(valid_positions) > 0:
                            # Distribute task reward equally across valid tokens
                            task_reward_per_token = sim_scores[batch_idx] / len(valid_positions)
                            v_low_targets[batch_idx, valid_positions] += task_reward_per_token
                else:  # Sequence-level targets [batch_size]
                    v_low_targets = v_low_targets + sim_scores
        # else: no instruction information or task reward disabled – leave targets unchanged.
        
        # Check if we need chunked processing
        batch_size = state.shape[0]
        seq_len = state.shape[1] + action.shape[1]  # Combined sequence length
        
        # Use chunked processing for larger batches or longer sequences
        chunking_threshold = getattr(self.config.training, 'critic_chunking_threshold', 100)
        # Dual-head Bellman: train both Q and V on the same TD target unless q1_only
        dual_head_bellman = use_bellman and not train_v_high_only
        if batch_size > 2 or seq_len > chunking_threshold:
            result = self._train_critics_chunked(
                sequence_critic, token_critic, state, action, base_state,
                q_high_targets, v_low_targets, tokenizer, v_low_targets,
                train_v_high_only=train_v_high_only and use_bellman,
                dual_head_bellman=dual_head_bellman,
            )
            if use_bellman:
                self._polyak_update_sequence_critic(sequence_critic)
            return result
        
        # ------------------------------------------------------------------
        # TRAIN SEQUENCE-LEVEL CRITIC FIRST
        # ------------------------------------------------------------------
        sequence_critic.train()

        # Matching target shape
        if q_high_targets.dim() == 0:
            q_high_targets = q_high_targets.unsqueeze(0)

        if use_bellman and train_v_high_only:
            # γ ablation / q1_only: V(s) ← r + γ V(s') only
            state_attention_mask = (state != tokenizer.pad_token_id).long().to(device=device)
            v_current = sequence_critic.v_value(
                input_ids=state,
                attention_mask=state_attention_mask,
                requires_grad=True,
            )
            if v_current.dim() > 1 and v_current.size(-1) == 1:
                v_current = v_current.squeeze(-1)
            v_current = v_current.to(dtype=model_dtype)
            sequence_critic_loss = F.mse_loss(v_current, q_high_targets)
        elif dual_head_bellman:
            # Polished path: train V_high and Q_high on the same TD targets
            state_attention_mask = (state != tokenizer.pad_token_id).long().to(device=device)
            v_current = sequence_critic.v_value(
                input_ids=state,
                attention_mask=state_attention_mask,
                requires_grad=True,
            )
            if v_current.dim() > 1 and v_current.size(-1) == 1:
                v_current = v_current.squeeze(-1)
            v_current = v_current.to(dtype=model_dtype)
            v_loss = F.mse_loss(v_current, q_high_targets)

            sequence_input = torch.cat([state, action], dim=1)
            sequence_attention_mask = (sequence_input != tokenizer.pad_token_id).long().to(device=device)
            q_current = sequence_critic(
                input_ids=sequence_input,
                attention_mask=sequence_attention_mask,
                requires_grad=True
            )
            if q_current.dim() > 1 and q_current.size(-1) == 1:
                q_current = q_current.squeeze(-1)
            q_current = q_current.to(dtype=model_dtype)
            q_loss = F.mse_loss(q_current, q_high_targets)
            sequence_critic_loss = v_loss + q_loss
        else:
            sequence_input = torch.cat([state, action], dim=1)
            sequence_attention_mask = (sequence_input != tokenizer.pad_token_id).long().to(device=device)

            sequence_value_current = sequence_critic(
                input_ids=sequence_input,
                attention_mask=sequence_attention_mask,
                requires_grad=True
            )

            if sequence_value_current.dim() > 1 and sequence_value_current.size(-1) == 1:
                sequence_value_current = sequence_value_current.squeeze(-1)

            sequence_value_current = sequence_value_current.to(dtype=model_dtype)
            sequence_critic_loss = F.mse_loss(sequence_value_current, q_high_targets)

        assert sequence_critic_loss.grad_fn is not None
        
        self.sequence_critic_optimizer.zero_grad()
        sequence_critic_loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(sequence_critic.parameters(), self.max_grad_norm)

        self.sequence_critic_optimizer.step()
        if use_bellman:
            self._polyak_update_sequence_critic(sequence_critic)

        # ------------------------------------------------------------------
        # (OPTIONAL) free activations & cache before the next critic to keep
        # memory usage as flat as possible.
        # ------------------------------------------------------------------
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # TRAIN TOKEN-LEVEL CRITIC (now that the sequence-critic graph is gone)
        # ------------------------------------------------------------------
        if token_critic is not None:
            token_critic.train()

            token_input = torch.cat([state, action], dim=1)
            token_attention_mask = (token_input != tokenizer.pad_token_id).long().to(device=device)

            token_value_current = token_critic(
                input_ids=token_input,
                attention_mask=token_attention_mask,
                requires_grad=True
            )

            if token_value_current.dim() > 1 and token_value_current.size(-1) == 1:
                token_value_current = token_value_current.squeeze(-1)

            # For per-token RL, extract only the action tokens from the critic output
            # token_value_current has shape [batch_size, state_len + action_len]
            # We need only the action tokens: [batch_size, action_len]
            state_len = state.size(1)
            token_value_current = token_value_current[:, state_len:]  # Extract action tokens only

            # Ensure correct dtype for all tensors
            token_value_current = token_value_current.to(dtype=model_dtype)

            if v_low_targets.dim() == 0:
                v_low_targets = v_low_targets.unsqueeze(0)

            token_critic_loss = F.mse_loss(token_value_current, v_low_targets)

            self.token_critic_optimizer.zero_grad()
            token_critic_loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(token_critic.parameters(), self.max_grad_norm)

            self.token_critic_optimizer.step()
        else:
            # Token critic is None (HL-only mode), create dummy loss
            token_critic_loss = torch.tensor(0.0, device=device, dtype=model_dtype)

        # ------------------------------------------------------------------
        # Final cleanup to keep the GPU tidy
        del token_value_current, token_input, token_attention_mask
        torch.cuda.empty_cache()

        # Prepare return dictionary
        result = {
            'sequence_critic_loss': sequence_critic_loss.item(),
            'token_critic_loss': token_critic_loss.item(),
        }
        
        if token_critic is None:
            result['token_critic_loss'] = 0.0  # Ensure token critic loss is 0 in HL-only mode
        
        return result

    def _train_critics_chunked(
        self,
        sequence_critic: nn.Module,
        token_critic: nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        base_state: torch.Tensor,
        q_high_targets: torch.Tensor,
        v_low_targets: torch.Tensor,
        tokenizer: Any,
        v_low_targets_override: Optional[torch.Tensor] = None,
        train_v_high_only: bool = False,
        dual_head_bellman: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Train critics using chunked processing to avoid OOM.
        
        Args:
            sequence_critic: Sequence-level critic network
            token_critic: Token-level critic network  
            state: Current state batch
            action: Action batch
            base_state: Base state batch
            q_high_targets: Precomputed targets for sequence critic
            v_low_targets: Precomputed targets for token critic
            tokenizer: Tokenizer for processing
            train_v_high_only: Fit V_high(s) only (γ ablation); skip Q head
            dual_head_bellman: Fit both V_high and Q_high on the same TD targets
            
        Returns:
            Dictionary of losses
        """
        from src.utils import get_optimal_chunk_size
        
        device = next(sequence_critic.parameters()).device
        model_dtype = next(sequence_critic.parameters()).dtype  # Get model's dtype for precision consistency
        batch_size = state.shape[0]
        seq_len = state.shape[1] + action.shape[1]
        
        # Calculate optimal chunk size based on available memory
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        free_memory_gb = available_memory_gb - allocated_memory_gb
        
        # Use conservative chunk size for critic training
        memory_fraction = getattr(self.config.training, 'critic_memory_fraction', 0.6)
        max_chunk_size = getattr(self.config.training, 'critic_max_chunk_size', 2)
        chunk_size = get_optimal_chunk_size(batch_size, free_memory_gb * memory_fraction, seq_len, model_size_gb=8.0)
        chunk_size = max(1, min(chunk_size, max_chunk_size))  # Conservative chunking
        
        print(f"Using chunked critic training: batch_size={batch_size}, seq_len={seq_len}, chunk_size={chunk_size}")
        print(f"Memory stats: available={available_memory_gb:.2f}GB, allocated={allocated_memory_gb:.2f}GB, free={free_memory_gb:.2f}GB")
        print(f"Model dtype: {model_dtype}")  # Log the dtype for debugging
        
        # Initialize loss accumulators
        total_sequence_loss = 0.0
        total_token_loss = 0.0
        num_chunks = 0
        
        # ------------------------------------------------------------------
        # TRAIN SEQUENCE-LEVEL CRITIC FIRST (CHUNKED)
        # ------------------------------------------------------------------
        sequence_critic.train()
        self.sequence_critic_optimizer.zero_grad()
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Get chunk data - keep token IDs as integers, don't convert to model dtype
            chunk_state = state[i:end_idx].to(device=device)  # Keep as Long for embeddings
            chunk_action = action[i:end_idx].to(device=device)  # Keep as Long for embeddings
            chunk_targets = q_high_targets[i:end_idx].to(device=device, dtype=model_dtype)
            
            if chunk_targets.dim() == 0:
                chunk_targets = chunk_targets.unsqueeze(0)

            if train_v_high_only:
                chunk_state_attn = (chunk_state != tokenizer.pad_token_id).long().to(device=device)
                chunk_value = sequence_critic.v_value(
                    input_ids=chunk_state,
                    attention_mask=chunk_state_attn,
                    requires_grad=True,
                )
                if chunk_value.dim() > 1 and chunk_value.size(-1) == 1:
                    chunk_value = chunk_value.squeeze(-1)
                chunk_value = chunk_value.to(dtype=model_dtype)
                chunk_sequence_loss = F.mse_loss(chunk_value, chunk_targets)
                del chunk_value
            elif dual_head_bellman:
                chunk_state_attn = (chunk_state != tokenizer.pad_token_id).long().to(device=device)
                chunk_v = sequence_critic.v_value(
                    input_ids=chunk_state,
                    attention_mask=chunk_state_attn,
                    requires_grad=True,
                )
                if chunk_v.dim() > 1 and chunk_v.size(-1) == 1:
                    chunk_v = chunk_v.squeeze(-1)
                chunk_v = chunk_v.to(dtype=model_dtype)
                v_loss = F.mse_loss(chunk_v, chunk_targets)

                chunk_sequence_input = torch.cat([chunk_state, chunk_action], dim=1)
                chunk_sequence_attention_mask = (chunk_sequence_input != tokenizer.pad_token_id).long().to(device=device)
                chunk_q = sequence_critic(
                    input_ids=chunk_sequence_input,
                    attention_mask=chunk_sequence_attention_mask,
                    requires_grad=True
                )
                if chunk_q.dim() > 1 and chunk_q.size(-1) == 1:
                    chunk_q = chunk_q.squeeze(-1)
                chunk_q = chunk_q.to(dtype=model_dtype)
                q_loss = F.mse_loss(chunk_q, chunk_targets)
                chunk_sequence_loss = v_loss + q_loss
                del chunk_v, chunk_q, chunk_sequence_input, chunk_sequence_attention_mask
            else:
                chunk_sequence_input = torch.cat([chunk_state, chunk_action], dim=1)
                chunk_sequence_attention_mask = (chunk_sequence_input != tokenizer.pad_token_id).long().to(device=device)
                chunk_value = sequence_critic(
                    input_ids=chunk_sequence_input,
                    attention_mask=chunk_sequence_attention_mask,
                    requires_grad=True
                )
                if chunk_value.dim() > 1 and chunk_value.size(-1) == 1:
                    chunk_value = chunk_value.squeeze(-1)
                chunk_value = chunk_value.to(dtype=model_dtype)
                chunk_sequence_loss = F.mse_loss(chunk_value, chunk_targets)
                del chunk_value, chunk_sequence_input, chunk_sequence_attention_mask
            
            # Scale loss by chunk size for proper averaging - ensure correct dtype
            chunk_weight = torch.tensor(len(chunk_state) / batch_size, device=device, dtype=model_dtype)
            scaled_loss = chunk_sequence_loss * chunk_weight
            
            # Backward pass
            scaled_loss.backward()
            
            # Accumulate loss
            total_sequence_loss += chunk_sequence_loss.item() * chunk_weight.item()
            num_chunks += 1
            
            # Clean up chunk tensors
            del chunk_state, chunk_action, chunk_targets
            del chunk_sequence_loss, scaled_loss, chunk_weight
            torch.cuda.empty_cache()
        
        # Update sequence critic
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(sequence_critic.parameters(), self.max_grad_norm)
        self.sequence_critic_optimizer.step()
        
        # CRITICAL: Clear gradients and move processed tensors off GPU after optimizer step
        self.sequence_critic_optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ------------------------------------------------------------------
        # TRAIN TOKEN-LEVEL CRITIC (CHUNKED)
        # ------------------------------------------------------------------
        if token_critic is not None:
            token_critic.train()
            self.token_critic_optimizer.zero_grad()
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                
                # Get chunk data - keep token IDs as integers, don't convert to model dtype
                chunk_state = state[i:end_idx].to(device=device)  # Keep as Long for embeddings
                chunk_action = action[i:end_idx].to(device=device)  # Keep as Long for embeddings
                
                # Use override targets if provided, otherwise use default targets
                if v_low_targets_override is not None:
                    chunk_targets = v_low_targets_override[i:end_idx].to(device=device, dtype=model_dtype)
                else:
                    chunk_targets = v_low_targets[i:end_idx].to(device=device, dtype=model_dtype)
                
                # Process chunk
                chunk_token_input = torch.cat([chunk_state, chunk_action], dim=1)
                chunk_token_attention_mask = (chunk_token_input != tokenizer.pad_token_id).long().to(device=device)
                
                chunk_token_value = token_critic(
                    input_ids=chunk_token_input,
                    attention_mask=chunk_token_attention_mask,
                    requires_grad=True
                )
                
                if chunk_token_value.dim() > 1 and chunk_token_value.size(-1) == 1:
                    chunk_token_value = chunk_token_value.squeeze(-1)
                
                # For per-token RL, extract only the action tokens from the critic output
                # chunk_token_value has shape [chunk_size, state_len + action_len]
                # We need only the action tokens: [chunk_size, action_len]
                state_len = chunk_state.size(1)
                chunk_token_value = chunk_token_value[:, state_len:]  # Extract action tokens only
                
                # Ensure correct dtype for all tensors
                chunk_token_value = chunk_token_value.to(dtype=model_dtype)
                
                # Matching target shape
                if chunk_targets.dim() == 0:
                    chunk_targets = chunk_targets.unsqueeze(0)
                
                # Calculate loss for this chunk - ensure correct dtype
                chunk_token_loss = F.mse_loss(chunk_token_value, chunk_targets)
                
                # Scale loss by chunk size for proper averaging - ensure correct dtype
                chunk_weight = torch.tensor(len(chunk_state) / batch_size, device=device, dtype=model_dtype)
                scaled_loss = chunk_token_loss * chunk_weight
                
                # Backward pass
                scaled_loss.backward()
                
                # Accumulate loss
                total_token_loss += chunk_token_loss.item() * chunk_weight.item()
                
                # Clean up chunk tensors
                del chunk_state, chunk_action, chunk_targets
                del chunk_token_input, chunk_token_attention_mask
                del chunk_token_value, chunk_token_loss, scaled_loss, chunk_weight
                torch.cuda.empty_cache()
            
            # Update token critic
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(token_critic.parameters(), self.max_grad_norm)
            self.token_critic_optimizer.step()
            
            # CRITICAL: Clear gradients and move processed tensors off GPU after optimizer step
            self.token_critic_optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Token critic is None (HL-only mode), set loss to 0
            total_token_loss = 0.0
        
        result = {
            'sequence_critic_loss': total_sequence_loss,
            'token_critic_loss': total_token_loss,
        }
        
        if token_critic is None:
            result['token_critic_loss'] = 0.0  # Ensure token critic loss is 0 in HL-only mode
        
        return result

    def train_high_level_policy(
        self,
        high_level_policy: nn.Module,
        state: torch.Tensor,
        latent_plan: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Train the high-level policy using PPO with gradient accumulation support.
        
        Args:
            high_level_policy: High-level policy model
            state: Current state tensor
            latent_plan: Strategy indices from high-level policy
            old_log_probs: Log probabilities under old policy
            advantages: High-level advantages
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Dictionary of loss values
        """
        device = next(high_level_policy.parameters()).device
        state = state.to(device)
        latent_plan = latent_plan.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        
        # Handle large batches with chunking if needed
        batch_size = state.shape[0]
        if batch_size > 8:  # Use chunking for large batches
            return self._train_high_level_policy_chunked(
                high_level_policy, state, latent_plan, old_log_probs, advantages, gradient_accumulation_steps
            )
        
        # Get current log probabilities and entropy for the selected strategies
        current_log_probs, entropy = high_level_policy.get_log_probs(state, latent_plan)

        old_log_probs = old_log_probs.to(device=current_log_probs.device, dtype=current_log_probs.dtype)
        advantages = advantages.to(device=current_log_probs.device, dtype=current_log_probs.dtype)
        
        # Calculate ratio and surrogate losses
        diff = current_log_probs - old_log_probs 

        # Remove any residual non-finite numbers (±inf, nan)
        diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))

        # Clamp to a safe range before exponentiation
        diff_clamped = torch.clamp(diff, min=-11.0, max=11.0)
        ratio = torch.exp(diff_clamped.float()).type_as(diff_clamped)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        
        # Clamp advantages to prevent overflow
        advantages_clamped = torch.clamp(advantages, min=-1e4, max=1e4)

        surrogate1 = (ratio.float() * advantages_clamped.float()).type_as(advantages)
        surrogate2 = (clipped_ratio.float() * advantages_clamped.float()).type_as(advantages)
        
        # -------------------------------------------------------------------
        # NaN/Inf diagnostics – fail fast with detailed info
        # -------------------------------------------------------------------
        def _assert_finite(name: str, tensor: torch.Tensor):
            if not torch.isfinite(tensor).all():
                num_nan = torch.isnan(tensor).sum().item()
                num_inf = torch.isinf(tensor).sum().item()
                raise RuntimeError(
                    f"Non-finite values detected in {name}: NaN={num_nan}, Inf={num_inf}. "
                    f"Stats -> min: {tensor.min().item()}, max: {tensor.max().item()}"
                )

        _assert_finite("current_log_probs", current_log_probs)
        _assert_finite("old_log_probs", old_log_probs)
        _assert_finite("advantages", advantages)
        _assert_finite("ratio", ratio)
        _assert_finite("surrogate1", surrogate1)
        _assert_finite("surrogate2", surrogate2)
        
        # Calculate KL divergence between old and new policy
        # Proper KL approximation: E[(ratio - 1) - log_ratio] where ratio = exp(current - old)
        # This is always non-negative (unlike the simple log prob difference)
        log_ratio_kl = current_log_probs - old_log_probs
        ratio_kl = torch.exp(torch.clamp(log_ratio_kl, min=-10.0, max=10.0))
        kl_div = ((ratio_kl - 1.0) - log_ratio_kl).mean()
        
        # Calculate final actor loss (negative because we want to maximize)
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Add entropy term (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean() * self.entropy_coef
        
        # Add KL penalty to prevent large policy updates
        kl_loss = kl_div * self.kl_coef
        
        # Combine losses
        total_loss = actor_loss + entropy_loss + kl_loss
        
        # Update policy
        if self.high_level_optimizer is not None:
            self.high_level_optimizer.zero_grad()
            total_loss.backward()
            
            # Apply gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(high_level_policy.parameters(), self.max_grad_norm)
                
            self.high_level_optimizer.step()
        
        return {
            'high_level_policy_loss': actor_loss.item(),
            'high_level_entropy_loss': entropy_loss.item(),
            'high_level_kl_div': kl_div.item(),
            'high_level_total_loss': total_loss.item()
        }

    def _train_high_level_policy_chunked(
        self,
        high_level_policy: nn.Module,
        state: torch.Tensor,
        latent_plan: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Train the high-level policy using chunked processing to avoid OOM.
        
        Args:
            high_level_policy: High-level policy model
            state: Current state tensor
            latent_plan: Strategy indices from high-level policy  
            old_log_probs: Log probabilities under old policy
            advantages: High-level advantages
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Dictionary of loss values
        """
        from src.utils import get_optimal_chunk_size
        
        device = next(high_level_policy.parameters()).device
        batch_size = state.shape[0]
        seq_len = state.shape[1]
        
        # Calculate optimal chunk size based on available memory
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        chunk_size = get_optimal_chunk_size(batch_size, available_memory_gb * 0.8, seq_len)
        chunk_size = max(1, min(chunk_size, 4))  # Clamp to reasonable range
        
        # Initialize loss components
        total_actor_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_div = 0.0
        total_loss = 0.0
        num_chunks = 0
        
        # Zero gradients at the start
        if self.high_level_optimizer is not None:
            self.high_level_optimizer.zero_grad()
        
        # Process in chunks
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Get chunk data
            chunk_state = state[i:end_idx].to(device)
            chunk_latent_plan = latent_plan[i:end_idx].to(device)
            chunk_old_log_probs = old_log_probs[i:end_idx].to(device)
            chunk_advantages = advantages[i:end_idx].to(device)
            
            # Get current log probabilities and entropy for the chunk
            chunk_current_log_probs, chunk_entropy = high_level_policy.get_log_probs(chunk_state, chunk_latent_plan)
            
            # Ensure tensors are on the same device and dtype
            chunk_old_log_probs = chunk_old_log_probs.to(device=chunk_current_log_probs.device, dtype=chunk_current_log_probs.dtype)
            chunk_advantages = chunk_advantages.to(device=chunk_current_log_probs.device, dtype=chunk_current_log_probs.dtype)
            
            # Calculate ratio and surrogate losses
            diff = chunk_current_log_probs - chunk_old_log_probs
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            diff_clamped = torch.clamp(diff, min=-11.0, max=11.0)
            ratio = torch.exp(diff_clamped.float()).type_as(diff_clamped)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            
            chunk_advantages_clamped = torch.clamp(chunk_advantages, min=-1e4, max=1e4)
            surrogate1 = (ratio.float() * chunk_advantages_clamped.float()).type_as(chunk_advantages)
            surrogate2 = (clipped_ratio.float() * chunk_advantages_clamped.float()).type_as(chunk_advantages)
            
            # Calculate losses for this chunk
            # Proper KL approximation: E[(ratio - 1) - log_ratio] where ratio = exp(current - old)
            chunk_log_ratio = chunk_current_log_probs - chunk_old_log_probs
            chunk_ratio = torch.exp(torch.clamp(chunk_log_ratio, min=-10.0, max=10.0))
            chunk_kl_div = ((chunk_ratio - 1.0) - chunk_log_ratio).mean()
            chunk_actor_loss = -torch.min(surrogate1, surrogate2).mean()
            chunk_entropy_loss = -chunk_entropy.mean() * self.entropy_coef
            chunk_kl_loss = chunk_kl_div * self.kl_coef
            chunk_total_loss = chunk_actor_loss + chunk_entropy_loss + chunk_kl_loss
            
            # Scale loss by chunk size for proper gradient accumulation
            chunk_weight = len(chunk_state) / batch_size
            scaled_loss = chunk_total_loss * chunk_weight / gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
            # Accumulate losses
            total_actor_loss += chunk_actor_loss.item() * chunk_weight
            total_entropy_loss += chunk_entropy_loss.item() * chunk_weight
            total_kl_div += chunk_kl_div.item() * chunk_weight
            total_loss += chunk_total_loss.item() * chunk_weight
            num_chunks += 1
            
            # Clean up chunk tensors
            del chunk_state, chunk_latent_plan, chunk_old_log_probs, chunk_advantages
            del chunk_current_log_probs, chunk_entropy, scaled_loss
            torch.cuda.empty_cache()
        
        # Update policy after accumulating gradients from all chunks
        if self.high_level_optimizer is not None:
            # Apply gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(high_level_policy.parameters(), self.max_grad_norm)
            
            self.high_level_optimizer.step()
            
            # CRITICAL: Clear gradients and move processed tensors off GPU after optimizer step
            self.high_level_optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return {
            'high_level_policy_loss': total_actor_loss,
            'high_level_entropy_loss': total_entropy_loss,
            'high_level_kl_div': total_kl_div,
            'high_level_total_loss': total_loss
        }

    def train_actor(
        self,
        actor: nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        base_state: Optional[torch.Tensor] = None,
        environment: Optional[Any] = None,
        latent_plan: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train the low-level actor network using hierarchical PPO.
        
        Args:
            actor: Low-level actor model
            state: Current state tensor
            action: Action tensor
            old_log_probs: Log probabilities of actions under old policy
            advantages: Low-level advantage estimates (from V_low)
            base_state: Original state tensor (optional)
            environment: Environment instance for model access (optional)
            latent_plan: High-level latent plan (optional)
            
        Returns:
            Dictionary of loss values
        """
        # Check if chunking is enabled
        actor_chunking_enabled = getattr(self.config.training, 'actor_chunking_enabled', False)
        batch_size = state.shape[0]
        seq_len = state.shape[1] + action.shape[1]
        
        # Use chunking if enabled or if sequence is too long
        actor_chunk_threshold = getattr(self.config.training, 'actor_chunk_threshold', 2048)
        if actor_chunking_enabled or (batch_size * seq_len > actor_chunk_threshold):
            return self._train_actor_chunked(
                actor=actor,
                state=state,
                action=action,
                old_log_probs=old_log_probs,
                advantages=advantages,
                base_state=base_state,
                environment=environment,
                latent_plan=latent_plan
            )
        
        # Original non-chunked implementation
        return self._train_actor_original(
            actor=actor,
            state=state,
            action=action,
            old_log_probs=old_log_probs,
            advantages=advantages,
            base_state=base_state,
            environment=environment,
            latent_plan=latent_plan
        )
    
    def _train_actor_original(
        self,
        actor: nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        base_state: Optional[torch.Tensor] = None,
        environment: Optional[Any] = None,
        latent_plan: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Original non-chunked actor training method.
        
        Args:
            actor: Low-level actor model
            state: Current state tensor
            action: Action tensor
            old_log_probs: Log probabilities of actions under old policy
            advantages: Low-level advantage estimates (from V_low)
            base_state: Original state tensor (optional)
            environment: Environment instance for model access (optional)
            latent_plan: High-level latent plan (optional)
            
        Returns:
            Dictionary of loss values
        """
        # Move inputs to device if not already there
        device = next(actor.parameters()).device
        state = state.to(device)
        action = action.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        base_state = base_state.to(device)
        
        # Ensure actor is in training mode for gradient computation
        actor.train()
        
        # Get current log probabilities (conditioned on latent plan if provided)
        if latent_plan is not None:
            current_log_probs, entropy, perplexity = actor.get_log_probs(
                input_ids=state,
                output_ids=action,
                latent_plan=latent_plan,
                require_grad=True
            )
        else:
            current_log_probs, entropy, perplexity = actor.get_log_probs(
                input_ids=state,
                output_ids=action,
                require_grad=True
            )

        # Validate gradients
        if torch.isnan(current_log_probs).any():
            raise RuntimeError("NaN values detected in current_log_probs")
        if current_log_probs.requires_grad:
            assert current_log_probs.grad_fn is not None, "Log probabilities must have a gradient function when requires_grad=True."
        else:
            current_log_probs = current_log_probs.detach().requires_grad_(True)

        # ------------------------------------------------------------------
        # PPO ratio computation (robust version)
        # ------------------------------------------------------------------
        diff = current_log_probs - old_log_probs

        pad_id = getattr(actor, 'tokenizer', None).pad_token_id if getattr(actor, 'tokenizer', None) else None
        if pad_id is not None and action.dim() == 2:
            valid_mask = (action != pad_id)
            diff = torch.where(valid_mask, diff, torch.zeros_like(diff))

        # Replace non-finite values with 0 to avoid overflows
        diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))

        diff_clamped = torch.clamp(diff, min=-11.0, max=11.0)  # tighter clamp to avoid fp16 overflow
        # Compute exp in higher precision to prevent Inf when using fp16
        ratio = torch.exp(diff_clamped.float()).type_as(diff_clamped).clamp(min=1e-4, max=1e4)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

        # Ensure advantages match ratio dimensions (should both be [batch_size, action_len])
        assert advantages.shape == ratio.shape, f"Advantage shape {advantages.shape} must match ratio shape {ratio.shape}"

        # ------------------------------------------------------------------
        # Prevent fp16 overflow: compute surrogate losses in float32, then
        # cast back.  This keeps gradient signal but avoids inf values when
        # |advantages| * ratio > 6.5e4 (fp16 max).
        # ------------------------------------------------------------------

        advantages_clamped = torch.clamp(advantages, min=-1e4, max=1e4)

        surrogate1 = (ratio.float() * advantages_clamped.float()).type_as(advantages)
        surrogate2 = (clipped_ratio.float() * advantages_clamped.float()).type_as(advantages)

        # Safety check
        def _assert_finite_local(name: str, tensor: torch.Tensor):
            if not torch.isfinite(tensor).all():
                num_nan = torch.isnan(tensor).sum().item()
                num_inf = torch.isinf(tensor).sum().item()
                raise RuntimeError(f"Non-finite values detected in {name}: NaN={num_nan}, Inf={num_inf}")

        _assert_finite_local("ratio", ratio)

        # ------------------------------------------------------------------

        # Calculate KL divergence between old and new policy
        # Proper KL approximation: E[(ratio - 1) - log_ratio] where ratio = exp(current - old)
        # This is always non-negative (unlike the simple log prob difference)
        log_ratio_kl = current_log_probs - old_log_probs
        ratio_kl = torch.exp(torch.clamp(log_ratio_kl, min=-10.0, max=10.0))
        kl_div = ((ratio_kl - 1.0) - log_ratio_kl).mean()

        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        # Add entropy, perplexity and KL losses below (move earlier code accordingly)

        # Apply token-level guidance (safety or task-completion) if available
        if self.use_token_masked_computation:
            # Get tokenizer from actor or environment
            tokenizer = getattr(actor, 'tokenizer', None)
            if tokenizer is None and hasattr(environment, 'get_target_tokenizer'):
                tokenizer = environment.get_target_tokenizer()
            if tokenizer is None:
                raise AttributeError("No tokenizer available for token-masked computation")

            # Decode actions to text
            action_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)

            if hasattr(environment, 'compute_safety_logits'):
                # Fallback to safety-based guidance
                safety_logits = environment.compute_safety_logits(action_texts)
                if safety_logits is not None:
                    batch_size = advantages.shape[0]
                    modified_advantages = advantages.clone()
                    for i in range(batch_size):
                        if i < len(safety_logits):
                            safety_score = torch.sigmoid(safety_logits[i]).mean()
                            modified_advantages[i] = advantages[i] * (1.0 + (1.0 - safety_score) * 0.5)
                    advantages = modified_advantages
        
        # -------------------------------------------------------------------
        # NaN/Inf diagnostics – fail fast with detailed info
        # -------------------------------------------------------------------
        def _assert_finite(name: str, tensor: torch.Tensor):
            if not torch.isfinite(tensor).all():
                num_nan = torch.isnan(tensor).sum().item()
                num_inf = torch.isinf(tensor).sum().item()
                raise RuntimeError(
                    f"Non-finite values detected in {name}: NaN={num_nan}, Inf={num_inf}. "
                    f"Stats -> min: {tensor.min().item()}, max: {tensor.max().item()}"
                )

        _assert_finite("current_log_probs", current_log_probs)
        _assert_finite("old_log_probs", old_log_probs)
        _assert_finite("advantages", advantages)
        _assert_finite("ratio", ratio)
        _assert_finite("surrogate1", surrogate1)
        _assert_finite("surrogate2", surrogate2)
        
        # Calculate KL divergence between old and new policy
        # Proper KL approximation: E[(ratio - 1) - log_ratio] where ratio = exp(current - old)
        # This is always non-negative (unlike the simple log prob difference)
        log_ratio_kl = current_log_probs - old_log_probs
        ratio_kl = torch.exp(torch.clamp(log_ratio_kl, min=-10.0, max=10.0))
        kl_div = ((ratio_kl - 1.0) - log_ratio_kl).mean()
        
        # Calculate final actor loss (negative because we want to maximize)
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Compute perplexity from selected log-probs (PPL = exp(−mean(log p)))
        # Fix: Normalize by sequence length to compare sequences fairly
        mean_log_probs = current_log_probs.mean(dim=-1)  # Average per sequence [batch_size]
        perplexity = torch.exp(-mean_log_probs)  # Perplexity per sequence [batch_size]
        # Add perplexity term (positive because we want to minimize perplexity)
        perplexity_loss = perplexity.mean() * self.perplexity_coef
        
        # Add KL penalty to prevent large policy updates
        kl_loss = kl_div * self.kl_coef

        # Add entropy term (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean() * self.entropy_coef
        
        # Combine losses
        total_loss = actor_loss + entropy_loss + kl_loss + perplexity_loss
        
        # -------------------- Optimizer step with GradScaler --------------------
        # Skip when hl_only_training and actor has no trainable params
        if self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()

            if self.scaler is not None:
                # Mixed-precision backward
                self.scaler.scale(total_loss).backward()

                # Unscale before clipping so the norm is in correct units
                self.scaler.unscale_(self.actor_optimizer)

                # Apply gradient clipping per loss term to prevent any component from dominating
                actor_grad_norm = None
                perplexity_grad_norm = None
                
                if self.max_grad_norm > 0:
                    # Clip gradients for each loss component separately
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    
                    # Additional clipping for perplexity gradients if they're too large
                    if perplexity_loss > self.perplexity_grad_clip_threshold:  # If perplexity loss is very high
                        perplexity_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm * self.perplexity_grad_clip_factor)
                        print(f"Warning: High perplexity loss ({perplexity_loss:.2f}), clipped gradients with norm {perplexity_grad_norm:.4f}")
                    
                    # Additional clipping for actor gradients if they're too large (same as perplexity)
                    if actor_grad_norm > self.max_grad_norm * self.adaptive_grad_clip_threshold:  # If gradients are close to clipping threshold
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm * self.perplexity_grad_clip_factor)
                        print(f"Warning: High actor gradient norm ({actor_grad_norm:.4f}), clipped gradients with norm {actor_grad_norm:.4f}")

                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                # FP32 path
                total_loss.backward()
                if self.max_grad_norm > 0:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
        
        # Update logs
        self.log_dict['actor_loss'].append(actor_loss.item())
        self.log_dict['entropy_loss'].append(entropy_loss.item())
        self.log_dict['kl_div'].append(kl_div.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'kl_div': kl_div.item(),
            'perplexity_loss': perplexity_loss.item(),
            'total_loss': total_loss.item(),
            'actor_grad_norm': actor_grad_norm if actor_grad_norm is not None else 0.0,
            'perplexity_grad_norm': perplexity_grad_norm if perplexity_grad_norm is not None else 0.0,
            'ratio_mean': ratio.mean().item(),
            'ratio_min': ratio.min().item(),
            'ratio_max': ratio.max().item()
        }
    
    def _train_actor_chunked(
        self,
        actor: nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        base_state: Optional[torch.Tensor] = None,
        environment: Optional[Any] = None,
        latent_plan: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train the low-level actor network using chunked processing to avoid OOM.
        
        Args:
            actor: Low-level actor model
            state: Current state tensor
            action: Action tensor
            old_log_probs: Log probabilities of actions under old policy
            advantages: Low-level advantage estimates (from V_low)
            base_state: Original state tensor (optional)
            environment: Environment instance for model access (optional)
            latent_plan: High-level latent plan (optional)
            
        Returns:
            Dictionary of loss values
        """
        from src.utils import get_optimal_chunk_size
        
        device = next(actor.parameters()).device
        batch_size = state.shape[0]
        seq_len = state.shape[1] + action.shape[1]
        
        # Calculate optimal chunk size based on available memory
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        free_memory_gb = available_memory_gb - allocated_memory_gb
        
        # Use conservative chunk size for actor training
        memory_fraction = getattr(self.config.training, 'actor_memory_fraction', 0.5)
        max_chunk_size = getattr(self.config.training, 'actor_max_chunk_size', 2)
        chunk_size = get_optimal_chunk_size(batch_size, free_memory_gb * memory_fraction, seq_len, model_size_gb=8.0)
        # More conservative chunk size for actor training (requires gradients, more memory-intensive)
        # Reduce chunk size further for long sequences (CARES/WildJailbreak have seq_len 450-650)
        if seq_len > 400:
            chunk_size = max(1, min(chunk_size, max(1, max_chunk_size // 4)))
        elif seq_len > 200:
            chunk_size = max(1, min(chunk_size, max(1, max_chunk_size // 2)))
        else:
            chunk_size = max(1, min(chunk_size, max_chunk_size))
        
        print(f"Using chunked actor training: batch_size={batch_size}, seq_len={seq_len}, chunk_size={chunk_size}")
        print(f"Memory stats: available={available_memory_gb:.2f}GB, allocated={allocated_memory_gb:.2f}GB, free={free_memory_gb:.2f}GB")
        
        # Initialize loss accumulators
        total_actor_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_div = 0.0
        total_perplexity_loss = 0.0
        total_loss = 0.0
        
        # Zero gradients at the start (skip when hl_only_training, no actor params)
        if self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()
        
        # Process in chunks
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Get chunk data
            chunk_state = state[i:end_idx].to(device)
            chunk_action = action[i:end_idx].to(device)
            chunk_old_log_probs = old_log_probs[i:end_idx].to(device)
            chunk_advantages = advantages[i:end_idx].to(device)
            chunk_base_state = base_state[i:end_idx].to(device) if base_state is not None else None
            chunk_latent_plan = latent_plan[i:end_idx].to(device) if latent_plan is not None else None
            
            # Ensure actor is in training mode for gradient computation
            actor.train()
            
            # Get current log probabilities (conditioned on latent plan if provided)
            if chunk_latent_plan is not None:
                chunk_current_log_probs, chunk_entropy, chunk_perplexity = actor.get_log_probs(
                    input_ids=chunk_state,
                    output_ids=chunk_action,
                    latent_plan=chunk_latent_plan,
                    require_grad=True
                )
            else:
                chunk_current_log_probs, chunk_entropy, chunk_perplexity = actor.get_log_probs(
                    input_ids=chunk_state,
                    output_ids=chunk_action,
                    require_grad=True
                )

            # Validate gradients
            if torch.isnan(chunk_current_log_probs).any():
                raise RuntimeError("NaN values detected in current_log_probs")
            if chunk_current_log_probs.requires_grad:
                assert chunk_current_log_probs.grad_fn is not None, "Log probabilities must have a gradient function when requires_grad=True."
            else:
                chunk_current_log_probs = chunk_current_log_probs.detach().requires_grad_(True)

            # Ensure time dimensions align with the action length (fail-fast but deterministic fix)
            target_len = chunk_action.size(1)
            if chunk_current_log_probs.size(1) != target_len:
                if chunk_current_log_probs.size(1) > target_len:
                    chunk_current_log_probs = chunk_current_log_probs[:, -target_len:]
                else:
                    pad = target_len - chunk_current_log_probs.size(1)
                    chunk_current_log_probs = F.pad(chunk_current_log_probs, (pad, 0), value=0.0)
            if chunk_old_log_probs.size(1) != target_len:
                if chunk_old_log_probs.size(1) > target_len:
                    chunk_old_log_probs = chunk_old_log_probs[:, -target_len:]
                else:
                    pad = target_len - chunk_old_log_probs.size(1)
                    chunk_old_log_probs = F.pad(chunk_old_log_probs, (pad, 0), value=0.0)

            # PPO ratio computation (robust version)
            diff = chunk_current_log_probs - chunk_old_log_probs

            pad_id = getattr(actor, 'tokenizer', None).pad_token_id if getattr(actor, 'tokenizer', None) else None
            if pad_id is not None and chunk_action.dim() == 2:
                valid_mask = (chunk_action != pad_id)
                diff = torch.where(valid_mask, diff, torch.zeros_like(diff))

            # Replace non-finite values with 0 to avoid overflows
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))

            diff_clamped = torch.clamp(diff, min=-11.0, max=11.0)
            ratio = torch.exp(diff_clamped.float()).type_as(diff_clamped).clamp(min=1e-4, max=1e4)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

            # Ensure chunk_advantages match ratio dimensions (should both be [chunk_size, action_len])
            assert chunk_advantages.shape == ratio.shape, f"Chunk advantage shape {chunk_advantages.shape} must match ratio shape {ratio.shape}"

            # Prevent fp16 overflow
            chunk_advantages_clamped = torch.clamp(chunk_advantages, min=-1e4, max=1e4)
            surrogate1 = (ratio.float() * chunk_advantages_clamped.float()).type_as(chunk_advantages)
            surrogate2 = (clipped_ratio.float() * chunk_advantages_clamped.float()).type_as(chunk_advantages)

            # NaN/Inf diagnostics
            def _assert_finite(name: str, tensor: torch.Tensor):
                if not torch.isfinite(tensor).all():
                    num_nan = torch.isnan(tensor).sum().item()
                    num_inf = torch.isinf(tensor).sum().item()
                    raise RuntimeError(
                        f"Non-finite values detected in {name}: NaN={num_nan}, Inf={num_inf}. "
                        f"Stats -> min: {tensor.min().item()}, max: {tensor.max().item()}"
                    )

            _assert_finite("chunk_current_log_probs", chunk_current_log_probs)
            _assert_finite("chunk_old_log_probs", chunk_old_log_probs)
            _assert_finite("chunk_advantages", chunk_advantages)
            _assert_finite("ratio", ratio)
            _assert_finite("surrogate1", surrogate1)
            _assert_finite("surrogate2", surrogate2)
            
            # Calculate losses for this chunk
            chunk_actor_loss = -torch.min(surrogate1, surrogate2).mean()
            # Fix: Normalize perplexity by sequence length to compare sequences fairly
            chunk_mean_log_probs = chunk_current_log_probs.mean(dim=-1)  # Average per sequence [chunk_size]
            chunk_perplexity = torch.exp(-chunk_mean_log_probs)  # Perplexity per sequence [chunk_size]
            chunk_perplexity_loss = chunk_perplexity.mean() * self.perplexity_coef
            
            # KL divergence from base model for stability (prevents deviation from pretrained distribution)
            # NOTE: During chunked training, we skip base_model KL divergence computation to avoid OOM
            # The base_model KL requires an extra forward pass with gradients, which is too memory-intensive
            # when already processing chunks. We use the simpler log-prob difference instead.
            chunk_kl_div = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Skip base_model KL in chunked training to avoid OOM (always use simple log-prob difference)
            use_base_model_kl = False  # Always skip in chunked training
            
            if use_base_model_kl and environment is not None and hasattr(environment, 'get_perplexity_model'):
                # Get the frozen base model from environment
                base_model = environment.get_perplexity_model()
                if base_model is not None:
                    # Determine base_model's device (should be cuda:1 for inference models)
                    try:
                        base_model_device = next(base_model.parameters()).device
                    except StopIteration:
                        # Fallback: try to get device from model itself
                        base_model_device = next(iter(base_model.state_dict().values())).device if base_model.state_dict() else device
                    
                    # Construct action input by concatenating state and action (same as non-chunked version)
                    chunk_action_input = torch.cat([chunk_state, chunk_action], dim=1)
                    # Get tokenizer to create attention mask
                    tokenizer = getattr(actor, 'tokenizer', None)
                    if tokenizer is None and hasattr(environment, 'tokenizer'):
                        tokenizer = environment.tokenizer
                    if tokenizer is not None:
                        pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
                        chunk_action_attention_mask = (chunk_action_input != pad_token_id).long()
                    else:
                        # Fallback: create attention mask assuming no padding
                        chunk_action_attention_mask = torch.ones_like(chunk_action_input, dtype=torch.long)
                    
                    # Move inputs to base_model's device (agent models on cuda:0, inference models on cuda:1)
                    chunk_action_input = chunk_action_input.to(base_model_device)
                    chunk_action_attention_mask = chunk_action_attention_mask.to(base_model_device)
                    
                    # Aggressive memory cleanup before KL divergence computation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    # Compute KL divergence between LoRA model and frozen base model
                    # Note: policy (actor) is on device (cuda:0), base_model is on base_model_device (cuda:1)
                    # The function handles cross-device operations and returns result on policy's device
                    chunk_kl_div = compute_kl_divergence_from_base_model(
                        policy=actor,
                        base_model=base_model,
                        input_ids=chunk_action_input,
                        attention_mask=chunk_action_attention_mask,
                        kl_coef=getattr(self.config.training, 'base_model_kl_coef', 0.1),
                        minibatch_size=1,  # Process one sample at a time to minimize memory usage
                        device=str(base_model_device),  # Use base_model's device for input placement
                        top_k=getattr(self.config.training, 'base_model_kl_topk', 10)  # Use config value (default 10)
                    )
                    
                    # Clean up inputs after KL computation
                    del chunk_action_input, chunk_action_attention_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Result is already on actor's device (policy_device), no need to move
                else:
                    # Fallback: proper KL approximation for stability
                    chunk_log_ratio = chunk_current_log_probs - chunk_old_log_probs
                    chunk_ratio = torch.exp(torch.clamp(chunk_log_ratio, min=-10.0, max=10.0))
                    chunk_kl_div = ((chunk_ratio - 1.0) - chunk_log_ratio).mean()
                    chunk_kl_div = chunk_kl_div * self.kl_coef
            else:
                # Use proper KL approximation: E[(ratio - 1) - log_ratio] for chunked training
                # This is always non-negative and doesn't require an extra forward pass
                chunk_log_ratio = chunk_current_log_probs - chunk_old_log_probs
                chunk_ratio = torch.exp(torch.clamp(chunk_log_ratio, min=-10.0, max=10.0))
                chunk_kl_div = ((chunk_ratio - 1.0) - chunk_log_ratio).mean()
                # Apply base_model_kl_coef if specified, otherwise use regular kl_coef
                kl_coef_to_use = getattr(self.config.training, 'base_model_kl_coef', None)
                if kl_coef_to_use is None or kl_coef_to_use == 0:
                    kl_coef_to_use = self.kl_coef
                chunk_kl_div = chunk_kl_div * kl_coef_to_use
            
            chunk_entropy_loss = -chunk_entropy.mean() * self.entropy_coef
            
            # Combine losses (chunk_kl_div already includes kl_coef from compute_kl_divergence_from_base_model)
            chunk_total_loss = chunk_actor_loss + chunk_entropy_loss + chunk_kl_div + chunk_perplexity_loss
            
            # Scale loss by chunk size for proper gradient accumulation
            chunk_weight = len(chunk_state) / batch_size
            scaled_loss = chunk_total_loss * chunk_weight
            
            # Backward pass with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Accumulate losses
            total_actor_loss += chunk_actor_loss.item() * chunk_weight
            total_entropy_loss += chunk_entropy_loss.item() * chunk_weight
            total_kl_div += chunk_kl_div.item() * chunk_weight
            total_perplexity_loss += chunk_perplexity_loss.item() * chunk_weight
            total_loss += chunk_total_loss.item() * chunk_weight
            
            # Clean up chunk tensors
            del chunk_state, chunk_action, chunk_old_log_probs, chunk_advantages
            if chunk_base_state is not None:
                del chunk_base_state
            if chunk_latent_plan is not None:
                del chunk_latent_plan
            del chunk_current_log_probs, chunk_entropy, chunk_perplexity
            del ratio, clipped_ratio, surrogate1, surrogate2
            del chunk_actor_loss, chunk_entropy_loss, chunk_kl_div, chunk_perplexity_loss
            del chunk_total_loss, scaled_loss
            
            # Aggressive memory cleanup after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # Update actor after accumulating gradients from all chunks (skip when hl_only_training)
        actor_grad_norm = None
        perplexity_grad_norm = None
        
        if self.actor_optimizer is not None:
            if self.scaler is not None:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.actor_optimizer)
                    # Apply gradient clipping per loss term to prevent any component from dominating
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    
                    # Additional clipping for perplexity gradients if they're too large
                    if total_perplexity_loss > self.perplexity_grad_clip_threshold:  # If perplexity loss is very high
                        perplexity_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm * self.perplexity_grad_clip_factor)
                
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm > 0:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
            
            self.actor_optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'actor_loss': total_actor_loss,
            'entropy_loss': total_entropy_loss,
            'kl_div': total_kl_div,
            'perplexity_loss': total_perplexity_loss,
            'total_loss': total_loss,
            'actor_grad_norm': actor_grad_norm if actor_grad_norm is not None else 0.0,
            'perplexity_grad_norm': perplexity_grad_norm if perplexity_grad_norm is not None else 0.0
        }

    def train_on_batch(
        self,
        actor: nn.Module,
        sequence_critic: nn.Module,
        token_critic: nn.Module,
        batch_data: Any,
        environment: Any = None,
        tokenizer: Any = None,
        high_level_policy: Optional[nn.Module] = None,
        num_epochs: int = 3,
        compute_advantage_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train actor and critics on a batch of data using hierarchical RL.
        
        Args:
            actor: Low-level actor model
            sequence_critic: High-level critic model
            token_critic: Low-level critic model
            batch_data: Batch data (should be RolloutData namedtuple)
            environment: Environment instance for model access
            tokenizer: Tokenizer for text processing
            high_level_policy: High-level policy model (optional)
            num_epochs: Number of PPO epochs per batch
            compute_advantage_fn: Custom function to compute advantages
            
        Returns:
            Dictionary of training metrics
        """
        # Extract batch data
        state = batch_data.batch_state
        next_state = batch_data.batch_next_state
        action = batch_data.batch_acts
        old_log_probs = batch_data.batch_log_probs
        rewards = batch_data.batch_rews
        dones = batch_data.batch_dones
        base_state = batch_data.batch_base_state
        
        # Check if we need to compute advantages or they are provided
        compute_advantages = True
        if 'advantages' in dir(batch_data) and 'returns' in dir(batch_data):
            advantages = batch_data.advantages
            returns = batch_data.returns
            compute_advantages = False
        
        # Set device
        device = self.device or next(actor.parameters()).device
        
        # Debug prints removed for performance
        # Move data to device
        state = state.to(device)
        next_state = next_state.to(device)
        action = action.to(device)
        old_log_probs = old_log_probs.to(device)
        rewards = rewards.to(device)
        # If dones tensor is not provided, initialize to all-False tensor of same shape as rewards
        if dones is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool)
        dones = dones.to(device)
        base_state = base_state.to(device)
        
        # Compute hierarchical advantages
        with torch.no_grad():
            if getattr(environment, 'uses_tool_step_rewards', False):
                if batch_data.l3_rewards is None:
                    raise ValueError(
                        "uses_tool_step_rewards requires RolloutData.l3_rewards from tool-index rollout"
                    )
                action_attention_mask = (action != tokenizer.pad_token_id).long()
                token_counts = action_attention_mask.sum(dim=1).clamp(min=1)
                l3_rews = batch_data.l3_rewards.to(device)
                marginal_rewards = (
                    l3_rews.unsqueeze(1).float() / token_counts.unsqueeze(1).float()
                ) * action_attention_mask
                high_level_values = rewards.float()
            elif getattr(environment, 'uses_episode_harm_reward', False):
                action_attention_mask = (action != tokenizer.pad_token_id).long()
                token_counts = action_attention_mask.sum(dim=1).clamp(min=1)
                marginal_rewards = (
                    rewards.unsqueeze(1).float() / token_counts.unsqueeze(1).float()
                ) * action_attention_mask
                high_level_values = rewards.float()
            else:
                marginal_rewards = self.compute_marginal_rewards(
                    state, action, base_state, environment, tokenizer,
                    sequence_critic=sequence_critic,
                    instructions=getattr(batch_data, 'high_level_instructions', None)
                )
                high_level_values = self.v_high(state, action, base_state, environment)
            
            # Ensure high_level_values are floating point for mathematical operations
            if high_level_values.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                high_level_values = high_level_values.float()

            # Running Z-score: update stats from step rewards once, then apply to all streams
            rewards = self._normalize_reward_tensor(rewards, update=True)
            marginal_rewards = self._normalize_reward_tensor(marginal_rewards, update=False)
            high_level_values = self._normalize_reward_tensor(high_level_values, update=False)
            
            # Check whether to use per-token or sequence-level advantages
            use_per_token_advantages = getattr(self.config.training, 'use_per_token_advantages', True)
            
            if use_per_token_advantages:
                # This follows the standard PPO approach 
                with torch.no_grad():
                    # Get critic predictions for current state-action pairs
                    token_input = torch.cat([state, action], dim=1)
                    token_attention_mask = (token_input != tokenizer.pad_token_id).long().to(device=device)
                    
                    # Chunked token critic forward pass to avoid OOM
                    batch_size = token_input.size(0)
                    state_len = state.size(1)
                    chunk_size = getattr(self.config.training, 'critic_max_chunk_size', 4)
                    
                    critic_value_chunks = []
                    for chunk_start in range(0, batch_size, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, batch_size)
                        token_input_chunk = token_input[chunk_start:chunk_end]
                        token_attention_chunk = token_attention_mask[chunk_start:chunk_end]
                        
                        critic_chunk = token_critic(
                            input_ids=token_input_chunk,
                            attention_mask=token_attention_chunk
                        )
                        
                        # Extract action tokens only (critic outputs [chunk_size, state_len + action_len])
                        critic_chunk = critic_chunk[:, state_len:]  # [chunk_size, action_len]
                        critic_value_chunks.append(critic_chunk)
                        
                        # Clean up intermediate tensors
                        del token_input_chunk
                        del token_attention_chunk
                        del critic_chunk
                    
                    # Concatenate all chunks
                    critic_values = torch.cat(critic_value_chunks, dim=0)  # [batch_size, action_len]
                    del critic_value_chunks
                    
                    # Ensure correct dtype
                    if critic_values.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                        critic_values = critic_values.float()
                
                # --------------------------------------------------------------
                # TASK REWARD – cosine similarity between instruction and utterance
                # --------------------------------------------------------------
                instructions = getattr(batch_data, 'high_level_instructions', None)
                if instructions is not None and len(instructions) == action.size(0):
                    # Decode utterances back to plain text for similarity calc
                    utter_texts: List[str] = self._batch_decode_with_timeout(
                        tokenizer,
                        action,
                        skip_special_tokens=True,
                    )

                    safe_instr = [ins if ins is not None else "" for ins in instructions]

                    sim_scores = environment._get_embed_similarity_batched(
                        safe_instr,
                        utter_texts,
                    ).to(critic_values.device)

                    # Add similarity reward to marginal rewards (distributed across valid tokens)
                    action_attention_mask = (action != tokenizer.pad_token_id).to(critic_values.device)
                    for batch_idx in range(action.size(0)):
                        valid_positions = action_attention_mask[batch_idx].nonzero(as_tuple=True)[0]
                        if len(valid_positions) > 0:
                            # Distribute task reward equally across valid tokens
                            task_reward_per_token = sim_scores[batch_idx] / len(valid_positions)
                            marginal_rewards[batch_idx, valid_positions] += task_reward_per_token

                # Create attention mask for valid tokens
                action_attention_mask = (action != tokenizer.pad_token_id)
                
                # Compute per-token advantages using GAE with critic values
                low_level_advantages, low_level_returns = self.compute_gae_per_token(
                    marginal_rewards=marginal_rewards,
                    values=critic_values.detach(),  # Use critic predictions
                    attention_mask=action_attention_mask,
                    gamma=self.gamma,
                    gae_lambda=self.config.training.gae_lambda
                )
                
            else:
                # DEPRECATED: Sequence-level value estimates and advantages using critic
                with torch.no_grad():
                    # Get critic predictions for sequence-level values
                    token_input = torch.cat([state, action], dim=1)
                    token_attention_mask = (token_input != tokenizer.pad_token_id).long().to(device=device)
                    
                    critic_values = token_critic(
                        input_ids=token_input,
                        attention_mask=token_attention_mask
                    )
                    
                    # For sequence-level, take the mean of action token values
                    state_len = state.size(1)
                    action_values = critic_values[:, state_len:]  # [batch_size, action_len]
                    low_level_values = action_values.mean(dim=1)  # [batch_size]
                    
                    # Ensure correct dtype
                    if low_level_values.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                        low_level_values = low_level_values.float()

                # --------------------------------------------------------------
                # TASK REWARD – cosine similarity between instruction and utterance
                # --------------------------------------------------------------
                instructions = getattr(batch_data, 'high_level_instructions', None)
                if instructions is not None and len(instructions) == action.size(0):
                    # Decode utterances back to plain text for similarity calc
                    utter_texts: List[str] = self._batch_decode_with_timeout(
                        tokenizer,
                        action,
                        skip_special_tokens=True,
                    )

                    safe_instr = [ins if ins is not None else "" for ins in instructions]

                    sim_scores = environment._get_embed_similarity_batched(
                        safe_instr,
                        utter_texts,
                    ).to(low_level_values.device)

                    # Add similarity scalar to the per-step reward tensor and the
                    # bootstrap value so the advantage computation sees the task
                    # reward as part of the return.
                    rewards = rewards + sim_scores.to(dtype=rewards.dtype)
                    low_level_values = low_level_values + sim_scores.to(dtype=low_level_values.dtype)

                # Compute sequence-level advantages using deprecated GAE with critic values
                if low_level_values.numel() > 1:
                    low_level_advantages, low_level_returns = self.compute_gae_deprecated(
                        rewards=rewards,
                        values=low_level_values.detach(),  # Use critic predictions
                        next_values=torch.cat([low_level_values[1:], low_level_values[-1:]], dim=0),
                        dones=dones,
                        gamma=self.gamma,
                        gae_lambda=self.config.training.gae_lambda
                    )
                else:
                    low_level_advantages = low_level_values.clone().detach()
                    low_level_returns = low_level_values.clone().detach()

            # ------------------------------------------------------------------
            # High-level advantages: multi-step GAE over batch turn order when
            # next_state/dones available; else whitened centering.
            # ------------------------------------------------------------------
            if high_level_values.numel() > 1:
                # sequence-level V(s) estimates for bootstrap when Bellman path available
                with torch.no_grad():
                    if self._use_bellman_v_high() and hasattr(sequence_critic, "v_value"):
                        state_attention_mask = (state != tokenizer.pad_token_id).long().to(device=device)
                        hl_state_values = sequence_critic.v_value(
                            input_ids=state,
                            attention_mask=state_attention_mask,
                            requires_grad=False,
                            use_target_head=hasattr(sequence_critic, "target_state_value_head"),
                        )
                        if hl_state_values.dim() > 1 and hl_state_values.size(-1) == 1:
                            hl_state_values = hl_state_values.squeeze(-1)
                        hl_state_values = hl_state_values.to(dtype=high_level_values.dtype, device=device)
                        next_attention = (next_state != tokenizer.pad_token_id).long().to(device=device)
                        hl_next_values = sequence_critic.v_value(
                            input_ids=next_state,
                            attention_mask=next_attention,
                            requires_grad=False,
                            use_target_head=hasattr(sequence_critic, "target_state_value_head"),
                        )
                        if hl_next_values.dim() > 1 and hl_next_values.size(-1) == 1:
                            hl_next_values = hl_next_values.squeeze(-1)
                        hl_next_values = hl_next_values.to(dtype=high_level_values.dtype, device=device)
                        high_level_advantages, _ = self.compute_gae_deprecated(
                            rewards=high_level_values,
                            values=hl_state_values,
                            next_values=hl_next_values,
                            dones=dones,
                            gamma=self.gamma,
                            gae_lambda=self.config.training.gae_lambda,
                        )
                    else:
                        high_level_advantages = high_level_values - high_level_values.mean()
                        if getattr(self.config.training, "normalize_advantages", True):
                            high_level_advantages = (
                                high_level_advantages
                                / (high_level_advantages.std() + 1e-8)
                            )
            else:
                high_level_advantages = high_level_values.clone().detach()

        # Free advantage-computation intermediates (no longer needed; reduces peak GPU memory for actor training)
        del critic_values, marginal_rewards
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train for multiple epochs
        if self.hl_only_training:
            print(f"[HL-ONLY MODE] Starting training loop - will skip low-level actor and token critic training")
        
        metrics = {
            'actor_loss': 0,
            'perplexity': 0,
            'sequence_critic_loss': 0,
            'token_critic_loss': 0,

            'kl_div': 0,
            'entropy_loss': 0,
            'total_loss': 0
        }
        
        for epoch in range(num_epochs):
            # Train critics
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training critics")
            if self.hl_only_training:
                # In HL-only mode, only train sequence critic (for high-level policy advantages)
                # Skip token critic training since we won't train the low-level actor
                print("[HL-ONLY MODE] Skipping token critic training")
                critic_metrics = self.train_critics(
                    sequence_critic=sequence_critic,
                    token_critic=None,  # Skip token critic
                    state=state,
                    action=action,
                    base_state=base_state,
                    reward=rewards,
                    next_state=next_state,
                    environment=environment,
                    tokenizer=tokenizer,
                    instructions=getattr(batch_data, 'high_level_instructions', None),
                    v_low_targets=low_level_returns if 'low_level_returns' in locals() else None,
                    l3_rewards=getattr(batch_data, 'l3_rewards', None),
                    dones=dones,
                )
            else:
                # Normal mode: train both critics
                critic_metrics = self.train_critics(
                    sequence_critic=sequence_critic,
                    token_critic=token_critic,
                    state=state,
                    action=action,
                    base_state=base_state,
                    reward=rewards,
                    next_state=next_state,
                    environment=environment,
                    tokenizer=tokenizer,
                    instructions=getattr(batch_data, 'high_level_instructions', None),
                    v_low_targets=low_level_returns if 'low_level_returns' in locals() else None,
                    l3_rewards=getattr(batch_data, 'l3_rewards', None),
                    dones=dones,
                )
            
            # Extract critic metrics values before potential deletion (needed for metrics accumulation)
            critic_sequence_loss = critic_metrics['sequence_critic_loss']
            critic_token_loss = critic_metrics['token_critic_loss']
            
            # Train low-level actor (skip if hl_only_training)
            if self.hl_only_training:
                print("[HL-ONLY MODE] Skipping low-level actor training")
                # Create dummy actor metrics for compatibility
                actor_metrics = {
                    'actor_loss': 0.0,
                    'entropy_loss': 0.0,
                    'kl_div': 0.0,
                    'perplexity_loss': 0.0,
                    'total_loss': 0.0,
                    'actor_grad_norm': 0.0,
                    'perplexity_grad_norm': 0.0,
                    'ratio_mean': 0.0,
                    'ratio_min': 0.0,
                    'ratio_max': 0.0
                }
            else:
                # AGGRESSIVE MEMORY CLEANUP: Free memory between critic and actor training
                # Actor training with gradients is more memory-intensive than critic training
                del critic_metrics
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                import gc
                gc.collect()
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training low-level actor")
                actor_metrics = self.train_actor(
                    actor=actor,
                    state=state,
                    action=action,
                    old_log_probs=old_log_probs,
                    advantages=low_level_advantages,
                    base_state=base_state,
                    environment=environment
                )
            # Train high-level policy if provided
            if high_level_policy is not None:
                policy_kind = getattr(high_level_policy, "policy_kind", None)
                if policy_kind == "dqn":
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training high-level DQN policy")
                    high_level_metrics = high_level_policy.optimise_q_head()
                elif policy_kind == "ppo":
                    # Use actual high-level policy data from rollout
                    if hasattr(batch_data, 'high_level_actions') and batch_data.high_level_actions is not None:
                        high_level_actions = batch_data.high_level_actions.to(device)
                        high_level_log_probs = batch_data.high_level_log_probs.to(device)
                        
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training high-level PPO policy with {len(high_level_actions)} actions")
                        high_level_metrics = self.train_high_level_policy(
                            high_level_policy=high_level_policy,
                            state=state,
                            latent_plan=high_level_actions,  # Use actual actions as latent plans
                            old_log_probs=high_level_log_probs,
                            advantages=high_level_advantages,
                            gradient_accumulation_steps=getattr(self.config.training, 'gradient_accumulation_steps', 1)
                        )
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No high-level policy data available - this may indicate a data collection issue")
                        high_level_metrics = {}
                elif policy_kind == "llm":
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LLM high-level policy – no training step implemented (skipping)")
                    high_level_metrics = {}
                elif policy_kind == "hybrid":
                    # Hybrid policy: train the MLP component if it supports training
                    if hasattr(high_level_policy, 'mlp_policy'):
                        mlp_policy = high_level_policy.mlp_policy
                        mlp_policy_kind = getattr(mlp_policy, "policy_kind", None)
                        
                        if mlp_policy_kind == "dqn":
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training hybrid policy (MLP DQN component)")
                            high_level_metrics = mlp_policy.optimise_q_head()
                        elif mlp_policy_kind == "ppo" and hasattr(batch_data, 'high_level_actions') and batch_data.high_level_actions is not None:
                            high_level_actions = batch_data.high_level_actions.to(device)
                            high_level_log_probs = batch_data.high_level_log_probs.to(device)
                            
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training hybrid policy (MLP PPO component)")
                            high_level_metrics = self.train_high_level_policy(
                                high_level_policy=mlp_policy,
                                state=state,
                                latent_plan=high_level_actions,
                                old_log_probs=high_level_log_probs,
                                advantages=high_level_advantages,
                                gradient_accumulation_steps=getattr(self.config.training, 'gradient_accumulation_steps', 1)
                            )
                        else:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hybrid policy MLP component training not supported (skipping) - MLP is assumed to be pretrained")
                            high_level_metrics = {}
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hybrid policy has no trainable MLP component (skipping)")
                        high_level_metrics = {}
                elif policy_kind == "freeform":
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    import gc
                    gc.collect()

                    state_texts = getattr(batch_data, "high_level_state_texts", None)
                    instruction_texts = getattr(batch_data, "high_level_instructions", None)
                    base_questions = getattr(batch_data, "high_level_base_questions", None)
                    raw_output_ids = getattr(batch_data, "high_level_raw_output_ids", None)
                    old_instruction_log_probs = getattr(
                        batch_data, "high_level_instruction_log_probs", None
                    )
                    if (
                        state_texts
                        and instruction_texts
                        and old_instruction_log_probs
                        and self.high_level_optimizer is not None
                    ):
                        # Process high-level policy training in minibatches to avoid OOM
                        # This allows us to keep rollout_steps=128 while processing in smaller chunks
                        batch_size = len(state_texts)
                        minibatch_size = getattr(self.config.training, 'minibatch_size', 4)
                        # Use smaller minibatches for high-level policy to be conservative
                        hl_minibatch_size = min(minibatch_size, 4)
                        
                        old_lp_tensor = torch.tensor(
                            old_instruction_log_probs, device=device, dtype=torch.bfloat16
                        )
                        adv_tensor = high_level_advantages[:batch_size].to(
                            device=device, dtype=torch.bfloat16
                        )
                        
                        # Zero gradients at the start
                        self.high_level_optimizer.zero_grad()
                        
                        # Accumulate losses across minibatches
                        total_loss = 0.0
                        total_ratio_mean = 0.0
                        total_ratio_min = float('inf')
                        total_ratio_max = float('-inf')
                        total_kl = 0.0
                        total_entropy = 0.0
                        num_minibatches = 0
                        
                        for mb_start in range(0, batch_size, hl_minibatch_size):
                            mb_end = min(mb_start + hl_minibatch_size, batch_size)
                            
                            # Get minibatch data
                            mb_state_texts = state_texts[mb_start:mb_end]
                            mb_instruction_texts = instruction_texts[mb_start:mb_end]
                            mb_base_questions = base_questions[mb_start:mb_end] if base_questions else None
                            mb_old_lp = old_lp_tensor[mb_start:mb_end]
                            mb_adv = adv_tensor[mb_start:mb_end]
                            
                            # Get raw_output_ids for this minibatch
                            mb_raw_output_ids = None
                            if raw_output_ids is not None:
                                mb_raw_output_ids = raw_output_ids[mb_start:mb_end]
                            
                            # Compute log probs + entropy for this minibatch
                            mb_new_log_probs, mb_entropy = high_level_policy.compute_log_probs_from_texts(
                                mb_state_texts,
                                mb_instruction_texts,
                                base_questions=mb_base_questions,
                                raw_output_ids=mb_raw_output_ids,
                                return_entropy=True,
                            )
                            mb_new_log_probs = mb_new_log_probs.to(device=device, dtype=torch.bfloat16)
                            mb_entropy = mb_entropy.to(device=device, dtype=torch.bfloat16)
                            
                            # Validate log probs are finite
                            if not torch.isfinite(mb_new_log_probs).all():
                                num_nan = torch.isnan(mb_new_log_probs).sum().item()
                                num_inf = torch.isinf(mb_new_log_probs).sum().item()
                                raise RuntimeError(
                                    f"Non-finite values in high-level log probs: NaN={num_nan}, Inf={num_inf}. "
                                    f"Stats -> min: {mb_new_log_probs.min().item()}, max: {mb_new_log_probs.max().item()}"
                                )
                            
                            # PPO loss computation with numerical stability (same as low-level actor)
                            diff = mb_new_log_probs - mb_old_lp
                            
                            # Replace non-finite values with 0 to avoid overflows
                            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
                            
                            # Clamp log probability difference to prevent overflow/underflow
                            diff_clamped = torch.clamp(diff, min=-11.0, max=11.0)
                            ratio = torch.exp(diff_clamped.float()).type_as(diff_clamped).clamp(min=1e-4, max=1e4)
                            clipped_ratio = torch.clamp(
                                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                            )
                            
                            # Clamp advantages to prevent overflow
                            mb_adv_clamped = torch.clamp(mb_adv, min=-1e4, max=1e4)
                            surrogate1 = (ratio.float() * mb_adv_clamped.float()).type_as(mb_adv)
                            surrogate2 = (clipped_ratio.float() * mb_adv_clamped.float()).type_as(mb_adv)
                            mb_actor_loss = -torch.min(surrogate1, surrogate2).mean()

                            # KL approximation vs old policy: E[(r-1) - log r]
                            log_ratio_kl = diff_clamped
                            ratio_kl = ratio
                            mb_kl = ((ratio_kl.float() - 1.0) - log_ratio_kl.float()).mean()
                            mb_kl_loss = mb_kl * self.kl_coef
                            mb_entropy_loss = -mb_entropy.mean() * self.entropy_coef
                            mb_loss = mb_actor_loss + mb_kl_loss + mb_entropy_loss
                            
                            # Validate loss is finite before backward
                            if not torch.isfinite(mb_loss):
                                raise RuntimeError(f"Non-finite loss detected: {mb_loss.item()}")
                            
                            # Backward pass for this minibatch (accumulates gradients)
                            mb_loss.backward()
                            
                            # Accumulate metrics
                            total_loss += mb_loss.item()
                            total_ratio_mean += ratio.mean().item()
                            total_ratio_min = min(total_ratio_min, ratio.min().item())
                            total_ratio_max = max(total_ratio_max, ratio.max().item())
                            total_kl += mb_kl.item()
                            total_entropy += mb_entropy.mean().item()
                            num_minibatches += 1
                            
                            # Clean up minibatch tensors
                            del mb_new_log_probs, mb_entropy, ratio, clipped_ratio
                            del surrogate1, surrogate2, mb_loss, mb_actor_loss, mb_kl, mb_kl_loss, mb_entropy_loss
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Clip gradients and step optimizer once after all minibatches
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                high_level_policy.parameters(), self.max_grad_norm
                            )
                        self.high_level_optimizer.step()
                        
                        # Average metrics across minibatches
                        avg_loss = total_loss / num_minibatches if num_minibatches > 0 else 0.0
                        avg_ratio_mean = total_ratio_mean / num_minibatches if num_minibatches > 0 else 0.0
                        avg_kl = total_kl / num_minibatches if num_minibatches > 0 else 0.0
                        avg_entropy = total_entropy / num_minibatches if num_minibatches > 0 else 0.0
                        
                        high_level_metrics = {
                            "high_level_policy_loss": avg_loss,
                            "high_level_policy_ratio_mean": avg_ratio_mean,
                            "high_level_policy_ratio_min": total_ratio_min if total_ratio_min != float('inf') else 0.0,
                            "high_level_policy_ratio_max": total_ratio_max if total_ratio_max != float('-inf') else 0.0,
                            "high_level_policy_kl": avg_kl,
                            "high_level_policy_entropy": avg_entropy,
                        }
                    else:
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Freeform high-level policy – missing data for training (skipping)"
                        )
                        high_level_metrics = {}
                else:
                    raise ValueError(f"Unsupported high_level_policy.policy_kind: {policy_kind}")
            else:
                high_level_metrics = {}

            tool_index_policy = getattr(self, "tool_index_policy", None)
            if tool_index_policy is not None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training tool-index policy (L1 DQN)")
                tool_index_metrics = tool_index_policy.optimise_q_head()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if tool_index_metrics:
                    for key, value in tool_index_metrics.items():
                        if key not in metrics:
                            metrics[key] = 0
                        metrics[key] += value
            
            # Update metrics
            metrics['actor_loss'] += actor_metrics['actor_loss']
            metrics['perplexity'] += actor_metrics['perplexity_loss']
            # Use extracted critic metrics values (extracted before potential deletion)
            metrics['sequence_critic_loss'] += critic_sequence_loss
            metrics['token_critic_loss'] += critic_token_loss

            metrics['kl_div'] += actor_metrics['kl_div']
            metrics['entropy_loss'] += actor_metrics['entropy_loss']
            metrics['total_loss'] += actor_metrics['total_loss']
            
            # Add high-level policy metrics
            if high_level_metrics:
                for key, value in high_level_metrics.items():
                    if key not in metrics:
                        metrics[key] = 0
                    metrics[key] += value

            # PPO epoch early-stop when approx KL exceeds target_kl
            epoch_kl = float(actor_metrics['kl_div'])
            hl_kl = float(high_level_metrics.get("high_level_policy_kl", 0.0)) if high_level_metrics else 0.0
            approx_kl = max(epoch_kl, hl_kl)
            metrics['approx_kl_last_epoch'] = approx_kl
            metrics['ppo_epochs_run'] = float(epoch + 1)
            if self.target_kl is not None and approx_kl > float(self.target_kl):
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Early-stopping PPO epochs at epoch {epoch + 1}/{num_epochs}: "
                    f"approx_kl={approx_kl:.6f} > target_kl={self.target_kl}"
                )
                metrics['ppo_early_stop'] = 1.0
                # Average only over epochs actually run
                epochs_run = epoch + 1
                for key in list(metrics.keys()):
                    if key in ('approx_kl_last_epoch', 'ppo_epochs_run', 'ppo_early_stop'):
                        continue
                    metrics[key] /= epochs_run
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return metrics

        # Average metrics over epochs
        num_epochs = max(1, num_epochs)  # Avoid division by zero
        for key in metrics:
            if key in ('approx_kl_last_epoch', 'ppo_epochs_run', 'ppo_early_stop'):
                continue
            metrics[key] /= num_epochs
        metrics['ppo_early_stop'] = 0.0
        metrics['ppo_epochs_run'] = float(num_epochs)
        
        # CRITICAL: Final memory cleanup after all training epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics
    
    def train(
        self, 
        policy: nn.Module, 
        sequence_critic: nn.Module, 
        token_critic: nn.Module, 
        environment: Any, 
        tokenizer: Any, 
        num_episodes: int,
        rollout_steps: int,
        save_freq: int = 5, 
        output_dir: str = "outputs",
        save_checkpoint_fn: Optional[Callable] = None,
        high_level_policy: Optional[nn.Module] = None,
        templates_path: str = "src/configs/templates.json",
        episode_offset: int = 0
    ) -> Tuple[nn.Module, nn.Module, nn.Module, List[float], str]:
        """
        Main training loop for the agent.
        
        Args:
            policy: Policy network to train
            sequence_critic: Sequence-level critic to train
            token_critic: Token-level critic to train
            environment: Environment to collect rollouts from and access models
            tokenizer: Tokenizer for processing text
            num_episodes: Number of episodes to train for
            rollout_steps: Maximum number of steps per episode
            save_freq: Frequency of saving checkpoints
            output_dir: Directory to save checkpoints to
            save_checkpoint_fn: Function to save checkpoints
            high_level_policy: Optional high-level policy for instruction generation
            templates_path: Path to templates JSON file
            
        Returns:
            Tuple of (policy, sequence_critic, token_critic, episode_rewards)
        """
        # Use provided save function or default
        save_checkpoint_fn = save_checkpoint_fn or self._default_save_checkpoint
        
        # Generate unique run ID for this training session
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create run-specific directories
        run_output_dir = os.path.join(output_dir, run_id)
        run_log_dir = os.path.join("logs", run_id)
        
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(run_log_dir, exist_ok=True)
        
        print(f"Created run-specific directories:")
        print(f"  Output directory: {run_output_dir}")
        print(f"  Log directory: {run_log_dir}")

        # Online best_model tracking
        if getattr(self.config.training, "track_best_model", False):
            from src.trainer.best_model_tracker import BestModelTracker
            tracker_path = os.path.join(run_output_dir, "best_model_tracker.json")
            self.best_model_tracker = BestModelTracker(
                window=int(getattr(self.config.training, "best_model_window", 5)),
                reward_alpha=float(getattr(self.config.training, "best_model_reward_alpha", 0.5)),
                tracker_path=tracker_path,
            )
            print(
                f"  Best-model tracking: window={self.best_model_tracker.window}, "
                f"alpha={self.best_model_tracker.reward_alpha}"
            )
        else:
            self.best_model_tracker = None

        last_episode_metrics: Dict[str, float] = {}
        
        # Check for debug mode
        debug_mode = self.config.training.debug_mode
        if debug_mode:
            print("DEBUG MODE")
            # Reset GPU memory stats at the start of training for clean peak measurements
            self._reset_gpu_memory_stats()
            
            # Debug device placement
            print("\n=== DEVICE PLACEMENT DEBUG ===")
            print(f"Policy device: {next(policy.parameters()).device}")
            print(f"Sequence critic device: {next(sequence_critic.parameters()).device}")
            print(f"Token critic device: {next(token_critic.parameters()).device}")
            if high_level_policy:
                print(f"High-level policy device: {next(high_level_policy.parameters()).device}")
            print(f"Target model device: {next(environment.target_model.parameters()).device}")
            print(f"Guard model device: {next(environment.guard_model.parameters()).device}")
            print("==============================\n")
            
        # Training loop
        episode_rewards = []
        
        # Warmup strategy: Use small batch for first episode to initialize optimizer states
        warmup_batch_size = getattr(self.config.training, 'warmup_batch_size', 8)  # Use config or default to 8
        # NEW: Warmup rollout steps and conversation bandwidth parameters
        warmup_rollout_steps = getattr(self.config.training, 'warmup_rollout_steps', warmup_batch_size)
        warmup_max_new_tokens = getattr(self.config.environment, 'warmup_max_new_tokens', getattr(environment, 'max_new_tokens', None))

        original_batch_size = environment.batch_size if hasattr(environment, 'batch_size') else None
        original_rollout_steps = rollout_steps
        original_max_new_tokens = getattr(environment, 'max_new_tokens', None)

        for episode in range(num_episodes):
            # Calculate the actual episode number (including offset)
            actual_episode = episode + episode_offset
            
            # Apply learning rate annealing
            self.apply_learning_rate_annealing(actual_episode)
            
            # Use warmup settings for first episode (only if episode_offset is 0)
            use_warmup = getattr(self.config.training, 'use_warmup', False)
            if episode == 0 and episode_offset == 0 and use_warmup:
                print(f"WARMUP EPISODE: Using batch_size={warmup_batch_size}, rollout_steps={warmup_rollout_steps}, max_new_tokens={warmup_max_new_tokens}")
                if hasattr(environment, 'set_batch_size'):
                    environment.set_batch_size(warmup_batch_size)
                # Apply warmup settings to environment bandwidth if available
                if warmup_max_new_tokens is not None and hasattr(environment, 'max_new_tokens'):
                    environment.max_new_tokens = warmup_max_new_tokens

                current_rollout_steps = warmup_rollout_steps
            else:
                # Switch to full batch size after warmup
                if episode == 1 and use_warmup:
                    print(f"SWITCHING TO FULL BATCH: batch_size={original_batch_size}, rollout_steps={original_rollout_steps}, max_new_tokens={original_max_new_tokens}")
                    
                    if hasattr(environment, 'set_batch_size') and original_batch_size:
                        environment.set_batch_size(original_batch_size)
                    # Restore original conversation bandwidth
                    if original_max_new_tokens is not None and hasattr(environment, 'max_new_tokens'):
                        environment.max_new_tokens = original_max_new_tokens
                current_rollout_steps = original_rollout_steps
            
            # Reset environment
            state = environment.reset()
            
            start_time = time.time()
            if debug_mode:
                self._log_gpu_memory("Before rollout")
            rollout_data = collect_rollout(
                agent_trainer=self,
                policy=policy,
                environment=environment,
                tokenizer=tokenizer,
                rollout_steps=current_rollout_steps,
                config=self.config,
                high_level_policy=high_level_policy,
                templates_path=templates_path,
                episode=actual_episode  # Pass episode for epsilon annealing
            )
            if debug_mode:
                self._log_gpu_memory("After rollout")
            print(f"Rollout time: {time.time() - start_time:.2f}s")
            if debug_mode:
                self._log_gpu_memory("Before train")
            train_start_time = time.time()
            # Train on the collected rollout data
            metrics = self.train_on_batch(
                actor=policy,
                sequence_critic=sequence_critic,
                token_critic=token_critic,
                batch_data=rollout_data,
                environment=environment,
                tokenizer=tokenizer,
                high_level_policy=high_level_policy,
                num_epochs=self.config.training.epochs
            )
            if debug_mode:
                self._log_gpu_memory("After train", flush=True)
            print(f"Training time: {time.time() - train_start_time:.2f}s")
            
            # Calculate episode statistics
            mean_reward = rollout_data.batch_rews.mean().item()
            max_reward = rollout_data.batch_rews.max().item()
            min_reward = rollout_data.batch_rews.min().item()
            batch_variance = rollout_data.batch_rews.var().item()
            batch_std = rollout_data.batch_rews.std().item()
            num_timesteps = rollout_data.batch_state.size(0)
            num_conversations = len(rollout_data.conversations) if rollout_data.conversations else 0
            
            # Store episode reward
            episode_rewards.append(mean_reward)
            self.log_dict['episode_rewards'].append(mean_reward)
            
            # Capture peak GPU memory for this episode
            max_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            metrics['max_gpu_allocated_gb'] = max_gpu_gb
            
            # EXTRACT DATA: Get necessary data from rollout_data before cleanup
            batch_base_state = rollout_data.batch_base_state
            high_level_actions = rollout_data.high_level_actions
            high_level_instructions = getattr(rollout_data, 'high_level_instructions', None)
            freeform_state_texts = getattr(rollout_data, 'high_level_state_texts', None)
            freeform_instruction_log_probs = getattr(rollout_data, 'high_level_instruction_log_probs', None)
            freeform_base_questions = getattr(rollout_data, 'high_level_base_questions', None)
            conversations = rollout_data.conversations
            guard_scores = getattr(rollout_data, 'guard_scores', None)
            target_guard_scores = getattr(rollout_data, 'target_guard_scores', None)

            if getattr(self.config.training, 'dump_adversarial_trajectories', False):
                dump_path = self.config.training.adversarial_traj_dump_path
                if dump_path is None:
                    raise ValueError("dump_adversarial_trajectories requires adversarial_traj_dump_path")
                from src.experiments.adversarial_traj_dump import dump_adversarial_trajectories_from_rollout
                bank_size = dump_adversarial_trajectories_from_rollout(
                    conversations=conversations,
                    target_guard_scores=target_guard_scores,
                    dump_path=dump_path,
                    min_turns=self.config.training.adversarial_traj_min_turns,
                    max_count=self.config.training.adversarial_traj_max_count,
                    harm_threshold=self.config.training.adversarial_traj_harm_threshold,
                )
                print(f"  • Adversarial traj bank: {bank_size}/{self.config.training.adversarial_traj_max_count}")
            
            # Clear all model caches between episodes to prevent memory accumulation
            if hasattr(environment, 'clear_all_caches'):
                environment.clear_all_caches()
            
            # AGGRESSIVE MEMORY CLEANUP: Clear rollout data and force garbage collection
            # Note: Keep metrics for checkpoint saving, will delete at episode end
            del rollout_data
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Enhanced episode summary logging
            print("=" * 80)
            print(f"EPISODE {actual_episode+1:3d} SUMMARY")
            print("=" * 80)
            print(f"Total time: {time.time() - start_time:.2f}s")
            print(f"Rollout Stats:")
            print(f"  • Timesteps: {num_timesteps}")
            print(f"  • Conversations: {num_conversations}")
            print(f"  • Mean Reward: {mean_reward:.4f}")
            print(f"  • Max Reward:  {max_reward:.4f}")
            print(f"  • Min Reward:  {min_reward:.4f}")
            print(f"  • Batch Variance: {batch_variance:.4f}")
            print(f"  • Batch Std: {batch_std:.4f}")
            print(f"  • Max GPU Allocated: {max_gpu_gb:.2f} GB")
            print(f"Training Metrics:")
            print(f"  • Actor Loss:          {metrics.get('actor_loss', 0.0):.6f}")
            print(f"  • Perplexity Loss:          {metrics.get('perplexity', 0.0):.6f}")
            print(f"  • Sequence Critic Loss: {metrics.get('sequence_critic_loss', 0.0):.6f}")
            print(f"  • Token Critic Loss:   {metrics.get('token_critic_loss', 0.0):.6f}")
            print(f"  • KL Divergence:       {metrics.get('kl_div', 0.0):.6f}")
            print(f"  • Entropy Loss:        {metrics.get('entropy_loss', 0.0):.6f}")
            if 'high_level_policy_loss' in metrics:
                print(f"  • High-Level Policy Loss: {metrics.get('high_level_policy_loss', 0.0):.6f}")
            if metrics.get('avg_task_reward') is not None:
                print(f"  • Avg Task Reward:     {metrics['avg_task_reward']:.6f}")
            # Guard reward statistics (for all policy types)
            if guard_scores is not None:
                unsafe_count = (guard_scores > 0.5).sum().item()
                metrics['guard_agent_unsafe_count'] = unsafe_count
                metrics['guard_agent_mean'] = guard_scores.mean().item()
                metrics['guard_agent_std'] = guard_scores.std().item()
                metrics['guard_agent_min'] = guard_scores.min().item()
                metrics['guard_agent_max'] = guard_scores.max().item()
                print("Guard Reward Stats (Agent Actions):")
                print(f"  • Mean: {metrics['guard_agent_mean']:.4f}")
                print(f"  • Std:  {metrics['guard_agent_std']:.4f}")
                print(f"  • Min/Max: {metrics['guard_agent_min']:.4f}/{metrics['guard_agent_max']:.4f}")
                print(f"  • Unsafe (score > 0.5): {unsafe_count}/{len(guard_scores)} ({100*unsafe_count/len(guard_scores):.1f}%)")
            if target_guard_scores is not None:
                target_unsafe_count = (target_guard_scores > 0.5).sum().item()
                metrics['guard_target_unsafe_count'] = target_unsafe_count
                metrics['guard_target_mean'] = target_guard_scores.mean().item()
                metrics['guard_target_std'] = target_guard_scores.std().item()
                metrics['guard_target_min'] = target_guard_scores.min().item()
                metrics['guard_target_max'] = target_guard_scores.max().item()
                print("Guard Reward Stats (Target Responses):")
                print(f"  • Mean: {metrics['guard_target_mean']:.4f}")
                print(f"  • Std:  {metrics['guard_target_std']:.4f}")
                print(f"  • Min/Max: {metrics['guard_target_min']:.4f}/{metrics['guard_target_max']:.4f}")
                print(f"  • Unsafe (score > 0.5): {target_unsafe_count}/{len(target_guard_scores)} ({100*target_unsafe_count/len(target_guard_scores):.1f}%)")
            
            if getattr(high_level_policy, 'policy_kind', None) == 'freeform':
                # Validate that freeform policy uses guard rewards
                if guard_scores is None or target_guard_scores is None:
                    print("[WARNING] Freeform policy training: guard scores not available in rollout data")
                else:
                    # Verify that high-level advantages are derived from guard rewards
                    # (This is implicit since q_high uses guard service, but we log it for clarity)
                    print("[VALIDATION] Freeform policy using guard-derived rewards:")
                    print(f"  • High-level advantages computed from q_high (uses GuardRewardService)")
                    print(f"  • Agent action guard scores: {len(guard_scores)} samples")
                    print(f"  • Target response guard scores: {len(target_guard_scores)} samples")
                
                instruction_texts = high_level_instructions or []
                non_empty_instr = [instr for instr in instruction_texts if instr]
                if non_empty_instr:
                    token_lengths = [len(instr.split()) for instr in non_empty_instr]
                    avg_len = sum(token_lengths) / max(1, len(token_lengths))
                    max_len = max(token_lengths)
                    min_len = min(token_lengths)
                    metrics['freeform_instruction_len_avg'] = avg_len
                    metrics['freeform_instruction_len_max'] = max_len
                    metrics['freeform_instruction_len_min'] = min_len
                    print("Freeform Instruction Stats:")
                    print(f"  • Instructions logged: {len(non_empty_instr)}")
                    print(f"  • Avg length (tokens): {avg_len:.2f}")
                    print(f"  • Min/Max length:      {min_len}/{max_len}")
                else:
                    print("Freeform Instruction Stats: no instructions recorded")
                if freeform_instruction_log_probs:
                    # Filter out -inf values and replace with a reasonable minimum
                    # This handles any numerical underflow that might have occurred
                    finite_log_probs = [
                        lp if (isinstance(lp, (int, float)) and not (lp == float('-inf') or lp != lp)) 
                        else -1000.0  # Replace -inf/NaN with a large negative value
                        for lp in freeform_instruction_log_probs
                    ]
                    if finite_log_probs:
                        mean_lp = sum(finite_log_probs) / len(finite_log_probs)
                        min_lp = min(finite_log_probs)
                        max_lp = max(finite_log_probs)
                    else:
                        mean_lp = float('-inf')
                        min_lp = float('-inf')
                        max_lp = float('-inf')
                    metrics['freeform_instruction_logprob_mean'] = mean_lp
                    metrics['freeform_instruction_logprob_min'] = min_lp
                    metrics['freeform_instruction_logprob_max'] = max_lp
                    print(f"  • Mean log-prob:       {mean_lp:.4f}")
                    print(f"  • Min/Max log-prob:    {min_lp:.4f}/{max_lp:.4f}")
                if freeform_state_texts:
                    metrics['freeform_context_count'] = len(freeform_state_texts)
                    print(f"  • Context windows seen: {len(freeform_state_texts)}")
                unique_questions = set(
                    q for q in (freeform_base_questions or []) if q is not None and q != ""
                )
                if unique_questions:
                    print(f"  • Distinct base prompts: {len(unique_questions)}")
            # Add 20 Questions specific metrics if available
            if hasattr(environment, 'task') and environment.task in ["twenty_questions", "hierarchical_twenty_questions"]:
                if 'submit_wins' in metrics:
                    win_count = metrics.get('submit_wins', 0)
                    print(f"  • 20 Questions Wins: {win_count} total wins in rollout")
                    if 'submit_avg_episode_efficiency' in metrics and metrics.get('submit_avg_episode_efficiency', 0) > 0:
                        avg_episode_efficiency = metrics.get('submit_avg_episode_efficiency', 0.0)
                        min_episode_efficiency = metrics.get('submit_min_episode_efficiency', 0)
                        max_episode_efficiency = metrics.get('submit_max_episode_efficiency', 0)
                        print(f"  • 20 Questions Episode Efficiency: {avg_episode_efficiency:.1f} avg questions per win (min: {min_episode_efficiency}, max: {max_episode_efficiency})")
                    else:
                        print(f"  • 20 Questions Episode Efficiency: No wins this episode")
            
            if debug_mode:
                print(f"DEBUG MODE ACTIVE")
            print("=" * 80)

            log_episode_outputs(
                run_log_dir=run_log_dir,
                run_id=run_id,
                episode_index=actual_episode,
                mean_reward=mean_reward,
                batch_variance=batch_variance,
                batch_std=batch_std,
                templates_path=getattr(self.config.model, 'templates_path', templates_path),
                metrics=metrics,
                tokenizer=tokenizer,
                batch_base_state=batch_base_state,
                high_level_actions=high_level_actions,
            )

            # Save checkpoint periodically
            if (actual_episode + 1) % save_freq == 0:
                if save_checkpoint_fn is self._default_save_checkpoint:
                    save_checkpoint_fn(
                        policy=policy,
                        sequence_critic=sequence_critic,
                        token_critic=token_critic,
                        episode=actual_episode,
                        metrics=metrics,
                        output_dir=run_output_dir,
                        is_final=False,
                        high_level_policy=high_level_policy,
                        v_high_critic=None
                    )
                else:
                    save_checkpoint_fn(
                        policy=policy,
                        sequence_critic=sequence_critic,
                        token_critic=token_critic,
                        episode=actual_episode,
                        metrics=metrics,
                        output_dir=run_output_dir,
                        is_final=False
                    )

            # Track / save best_model from rolling window of train scores
            if self.best_model_tracker is not None:
                guard_mean = metrics.get("guard_target_mean")
                track_info = self.best_model_tracker.update(
                    mean_reward=mean_reward,
                    episode=actual_episode,
                    guard_target_mean=guard_mean,
                )
                metrics["best_model_episode_score"] = track_info["episode_score"]
                metrics["best_model_window_mean"] = track_info["window_mean"]
                metrics["best_model_best_window"] = track_info["best_window_score"]
                if track_info["is_new_best"] > 0.5:
                    print(
                        f"New best_model at episode {actual_episode}: "
                        f"window_mean={track_info['window_mean']:.4f}"
                    )
                    if save_checkpoint_fn is self._default_save_checkpoint:
                        save_checkpoint_fn(
                            policy=policy,
                            sequence_critic=sequence_critic,
                            token_critic=token_critic,
                            episode=actual_episode,
                            metrics=metrics,
                            output_dir=run_output_dir,
                            is_final=False,
                            is_best=True,
                            high_level_policy=high_level_policy,
                            v_high_critic=None,
                        )
                    else:
                        save_checkpoint_fn(
                            policy=policy,
                            sequence_critic=sequence_critic,
                            token_critic=token_critic,
                            episode=actual_episode,
                            metrics=metrics,
                            output_dir=run_output_dir,
                            is_final=False,
                            is_best=True,
                        )

            # Snapshot metrics for final_model (before per-episode cleanup)
            last_episode_metrics = dict(metrics)

            # Final cleanup for this episode: delete extracted data and metrics after checkpoint saving
            if 'metrics' in locals():
                del metrics
            if 'batch_base_state' in locals():
                del batch_base_state, high_level_actions, conversations
            if 'high_level_instructions' in locals():
                del high_level_instructions
            if 'freeform_state_texts' in locals():
                del freeform_state_texts
            if 'freeform_instruction_log_probs' in locals():
                del freeform_instruction_log_probs
            if 'freeform_base_questions' in locals():
                del freeform_base_questions
            
            # clear GPU memory on all devices
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                for device_id in range(num_gpus):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
        
        # Save final checkpoint
        final_episode = episode_offset + num_episodes - 1
        if save_checkpoint_fn is self._default_save_checkpoint:
            save_checkpoint_fn(
                policy=policy,
                sequence_critic=sequence_critic,
                token_critic=token_critic,
                episode=final_episode,
                metrics=last_episode_metrics,
                output_dir=run_output_dir,
                is_final=True,
                high_level_policy=high_level_policy,
                v_high_critic=None
            )
        else:
            save_checkpoint_fn(
                policy=policy,
                sequence_critic=sequence_critic,
                token_critic=token_critic,
                episode=final_episode,
                metrics=last_episode_metrics,
                output_dir=run_output_dir,
                is_final=True
            )
        
        return policy, sequence_critic, token_critic, episode_rewards, run_id
    
    def _log_gpu_memory(self, prefix: str = "", reset_peak: bool = False, flush: bool = False) -> None:
        """Print a one-line summary of current / peak GPU memory usage.

        This is only intended for lightweight debugging in *debug_mode*.
        It purposefully fails fast if CUDA is unavailable, so that any
        incorrect assumptions about the execution environment are caught
        immediately rather than silently ignored.
        
        Args:
            prefix: Prefix string for the log message
            reset_peak: Whether to reset peak memory stats after logging
            flush: Whether to flush the output buffer
        """
        # Call the standalone function with self as the first argument
        _log_gpu_memory(self, prefix, reset_peak, flush)
    
    def compute_gae_per_token(
        self,
        marginal_rewards: torch.Tensor,
        values: torch.Tensor,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token Generalized Advantage Estimation (GAE).
        
        Each token is treated as an individual timestep in RL, with its own advantage.
        
        Args:
            marginal_rewards: Per-token marginal rewards [batch_size, action_len]
            values: Per-token value estimates [batch_size, action_len]
            pad_token_id: Padding token ID to exclude from computation
            attention_mask: Mask for valid tokens [batch_size, action_len]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (per_token_advantages, per_token_returns) both [batch_size, action_len]
        """
        batch_size, action_len = marginal_rewards.shape
        advantages = torch.zeros_like(marginal_rewards)
        returns = torch.zeros_like(marginal_rewards)
        
        # Create attention mask if not provided
        if attention_mask is None:
            if pad_token_id is not None:
                # This assumes marginal_rewards contains token IDs, but it actually contains rewards
                # We need to get the attention mask from somewhere else
                attention_mask = torch.ones_like(marginal_rewards, dtype=torch.bool)
            else:
                attention_mask = torch.ones_like(marginal_rewards, dtype=torch.bool)
        
        # Compute per-token GAE for each sequence
        for batch_idx in range(batch_size):
            sequence_advantages = torch.zeros(action_len, device=marginal_rewards.device, dtype=marginal_rewards.dtype)
            sequence_mask = attention_mask[batch_idx].bool()
            valid_length = sequence_mask.sum().item()
            
            if valid_length == 0:
                continue
                
            gae = 0
            # Process tokens in reverse order (from last to first)
            for t in reversed(range(valid_length)):
                if not sequence_mask[t]:
                    continue
                    
                # Next value (0 if this is the last token)
                next_value = values[batch_idx, t + 1] if t + 1 < valid_length and sequence_mask[t + 1] else 0
                
                # TD error: r_t + γ * V(s_{t+1}) - V(s_t)
                delta = marginal_rewards[batch_idx, t] + gamma * next_value - values[batch_idx, t]
                
                # GAE update
                gae = delta + gamma * gae_lambda * gae
                sequence_advantages[t] = gae
            
            advantages[batch_idx] = sequence_advantages
        
        # Compute returns: A + V
        returns = advantages + values
        
        # Normalize advantages across all valid tokens (gated by config)
        if getattr(self.config.training, "normalize_advantages", True):
            if attention_mask is not None:
                valid_advantages = advantages[attention_mask]
                if len(valid_advantages) > 1:
                    # Ensure advantages are floating point
                    if valid_advantages.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                        valid_advantages = valid_advantages.float()
                    mean_adv = valid_advantages.mean()
                    std_adv = valid_advantages.std()
                    advantages[attention_mask] = (valid_advantages - mean_adv) / (std_adv + 1e-8)
            else:
                # Fallback: normalize all advantages
                if advantages.numel() > 1:
                    if advantages.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                        advantages = advantages.float()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def compute_gae_deprecated(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        token_level: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DEPRECATED: Compute sequence-level Generalized Advantage Estimation (GAE).
        
        This version operates on sequence-level values and is kept for ablation studies.
        
        Args:
            rewards: Reward tensor
            values: Value estimates for current states
            next_values: Value estimates for next states
            dones: Done flags (optional)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            token_level: Whether this is token-level computation
            
        Returns:
            Tuple of (advantages, returns)
        """
        if dones is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool)
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = next_values[step] if not dones[step] else 0
            else:
                next_value = values[step + 1] if not dones[step] else 0
            
            delta = rewards[step] + gamma * next_value - values[step]
            gae = delta + gamma * gae_lambda * gae * (1 - dones[step].float())
            advantages[step] = gae
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        if getattr(self.config.training, "normalize_advantages", True) and len(advantages) > 1:
            # Ensure advantages are floating point for mathematical operations
            if advantages.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                advantages = advantages.float()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        token_level: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        For backward compatibility, this method redirects to the appropriate implementation
        based on the input tensor dimensions.
        
        Args:
            rewards: Reward tensor
            values: Value estimates for current states  
            next_values: Value estimates for next states
            dones: Done flags (optional)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            token_level: Whether this is token-level computation
            
        Returns:
            Tuple of (advantages, returns)
        """
        # If values has 2 dimensions, use per-token GAE
        if values.dim() == 2:
            # For per-token case, we need to create attention mask from valid positions
            # Assume rewards are marginal rewards per token
            attention_mask = torch.ones_like(values, dtype=torch.bool)
            return self.compute_gae_per_token(
                marginal_rewards=rewards,
                values=values,
                attention_mask=attention_mask,
                gamma=gamma,
                gae_lambda=gae_lambda
            )
        else:
            # Use deprecated sequence-level GAE
            return self.compute_gae_deprecated(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
                token_level=token_level
            )

    def _compute_naive_decayed_rewards(self, action: torch.Tensor, seq_rewards: torch.Tensor, tokenizer) -> torch.Tensor:
        """Compute naïve back-to-front γ-decayed rewards per token.

        Each sequence in ``seq_rewards`` is a scalar reward (e.g. safety score
        for the full target response).  The last *real* token in the action
        gets that full reward, the previous gets γ·reward, and so on, moving
        backwards through the sequence.  ``pad_token_id`` positions receive
        zero.
        """
        # Fail fast on γ value to respect research-code policy
        gamma = getattr(self.config.training, "gamma", getattr(self, "gamma", 0.99))
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"TrainingConfig.gamma must be in (0,1] for naïve decayed reward, got {gamma}")

        attention_mask = (action != tokenizer.pad_token_id).long()  # [B,L]
        batch_size, seq_len = action.shape

        # positions: 0 .. L-1
        positions = torch.arange(seq_len, device=action.device).unsqueeze(0).expand(batch_size, -1)
        # Length of each action (number of non-pad tokens)
        token_lengths = attention_mask.sum(dim=1, keepdim=True)
        # Exponent for γ: count backwards from the last real token
        exponents = (token_lengths - 1 - positions).clamp(min=0)
        decay = (gamma ** exponents).to(dtype=seq_rewards.dtype)

        # Broadcast seq_rewards over token dimension and mask pads
        per_token = seq_rewards.unsqueeze(1) * decay * attention_mask
        return per_token

    def compute_combined_rejection_score(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        base_state: torch.Tensor, 
        environment: Any,
        tokenizer: Any,
        current_episode: int = 0,
        perplexity_start_weight: float = 1.0,
        perplexity_end_weight: float = 0.4,
        perplexity_decay_episodes: int = 100
    ) -> torch.Tensor:
        """
        Compute combined rejection sampling score using both v_high and perplexity.
        
        Both scores are batch normalized before combination to ensure fair comparison.
        Uses annealed weights where perplexity weight decays from start_weight to end_weight
        over the specified number of episodes.
        
        Args:
            state: Current state tensor
            action: Action sequence tensor
            base_state: Base state tensor
            environment: Environment for scoring
            tokenizer: Tokenizer for text processing
            current_episode: Current training episode for weight annealing
            perplexity_start_weight: Starting weight for perplexity score
            perplexity_end_weight: Final weight for perplexity score after annealing
            perplexity_decay_episodes: Number of episodes to decay perplexity weight
            
        Returns:
            Combined scores: v_high_weight * normalized_v_high + perplexity_weight * normalized_perplexity
        """
        # Compute annealed weights
        progress = min(current_episode / perplexity_decay_episodes, 1.0)
        perplexity_weight = perplexity_start_weight + (perplexity_end_weight - perplexity_start_weight) * progress
        v_high_weight = 1.0 - perplexity_weight  # v_high gets the remaining weight
        
        # Compute v_high scores
        v_high_scores = self.v_high(state, action, base_state, environment)
        
        # Compute perplexity scores
        action_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
        
        # Check if environment has perplexity computation available
        if hasattr(environment, 'compute_perplexity'):
            perplexity_scores = environment.compute_perplexity(action_texts)
        else:
            # Fallback to neutral perplexity scores if not available
            if not hasattr(self, '_perplexity_warning_shown'):
                print(f"Warning: Environment does not have perplexity computation. Using neutral scores for combined rejection sampling.")
                self._perplexity_warning_shown = True
            perplexity_scores = torch.ones(len(action_texts), device="cpu")
        
        # Ensure both scores are on the same device and have the same shape
        device = v_high_scores.device
        perplexity_scores = perplexity_scores.to(device=device, dtype=v_high_scores.dtype)
        
        # Batch normalize both scores
        # For v_high: higher is better, so we want to maximize
        # For perplexity: lower is better, so we negate to make higher better
        v_high_mean = v_high_scores.mean()
        v_high_std = v_high_scores.std()
        if v_high_std > 0:
            normalized_v_high = (v_high_scores - v_high_mean) / v_high_std
        else:
            normalized_v_high = v_high_scores - v_high_mean
        
        # For perplexity, lower is better, so we negate and normalize
        perplexity_mean = perplexity_scores.mean()
        perplexity_std = perplexity_scores.std()
        if perplexity_std > 0:
            normalized_perplexity = -(perplexity_scores - perplexity_mean) / perplexity_std
        else:
            normalized_perplexity = -(perplexity_scores - perplexity_mean)
        
        # Combine scores with annealed weights
        combined_scores = v_high_weight * normalized_v_high + perplexity_weight * normalized_perplexity
        
        # MEMORY CLEANUP: Delete intermediate tensors to prevent accumulation
        del v_high_scores, perplexity_scores, v_high_mean, v_high_std
        del perplexity_mean, perplexity_std, normalized_v_high, normalized_perplexity
        
        return combined_scores

    def _reset_gpu_memory_stats(self) -> None:
        """Reset GPU memory peak statistics.
        
        Call this at the start of training to get clean peak measurements.
        """
        _reset_gpu_memory_stats(self)



