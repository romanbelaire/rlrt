"""
Agent Trainer module for PPO optimization.

This module handles the training logic for PPO agents, including:
- Actor policy updates
- Critic value function updates
- Advantage estimation
- Loss calculation
- Training loop management

The trainer separates the training logic from the agent implementation,
providing a cleaner architecture and better separation of concerns.
"""

from datetime import datetime
from src.utils import pretty_print
from src.distributed_utils import DistributedSetup
from src.trainer.rollout import (
    RolloutData,
    _batch_decode_with_timeout,
    _decode_with_timeout,
    collect_rollout,
)
from src.rewards.guard_service import GuardRewardService
from src.trainer.utils import (
    _log_gpu_memory,
    _reset_gpu_memory_stats,
    compute_kl_divergence_from_base_model,
    get_base_model_from_policy,
    get_optimal_topk_for_kl_divergence,
)
from src.trainer.update import TrainerUpdateMixin
from src.trainer.logging_utils import save_checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from contextlib import contextmanager
import os
import sys
import json
import random
import time
import csv
from collections import defaultdict
DISTRIBUTED_AVAILABLE = True


class AgentTrainer(TrainerUpdateMixin):
    """
    Agent trainer for hierarchical reinforcement learning with text-based policies.
    
    This class handles hierarchical RL with high-level and low-level policies,
    including marginal reward decomposition and hierarchical value functions.
    """
    
    def __init__(
        self,
        actor_optimizer: Optional[torch.optim.Optimizer],
        sequence_critic_optimizer: torch.optim.Optimizer,
        token_critic_optimizer: torch.optim.Optimizer,
        high_level_optimizer: Optional[torch.optim.Optimizer] = None,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.0,
        perplexity_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        perplexity_grad_clip_threshold: float = 10.0,
        perplexity_grad_clip_factor: float = 0.1,
        adaptive_grad_clip_threshold: float = 0.8,
        device: Optional[str] = None,
        target_kl: Optional[float] = None,
        distributed_setup: Optional[Any] = None,
        use_token_masked_computation: bool = True,
        minibatch_size: int = 8,
        config: Optional[Any] = None,
        gamma: float = 0.99,
        hl_only_training: bool = False
    ) -> None:
        """
        Initialize the agent trainer with optional GPU model swapping.
        
        Args:
            actor_optimizer: Optimizer for the actor (policy) network
            sequence_critic_optimizer: Optimizer for the sequence-level critic
            token_critic_optimizer: Optimizer for the token-level critic
            high_level_optimizer: Optimizer for the high-level policy (optional)
            clip_ratio: PPO clip ratio
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy regularization
            kl_coef: Coefficient for KL divergence penalty
            perplexity_coef: Coefficient for perplexity penalty
            max_grad_norm: Maximum gradient norm for clipping
            perplexity_grad_clip_threshold: Threshold above which perplexity gradients get additional clipping
            perplexity_grad_clip_factor: Factor to multiply max_grad_norm by for perplexity gradient clipping
            adaptive_grad_clip_threshold: Threshold for adaptive gradient clipping warnings (fraction of max_grad_norm)
            device: Device to use for training
            target_kl: Target KL divergence for early stopping
            distributed_setup: Distributed training setup
            use_token_masked_computation: Whether to use token-masked computation
            minibatch_size: Size of minibatches for training
            config: Configuration object
            gamma: Discount factor
            hl_only_training: Whether to train only the high-level policy
        """
        self.actor_optimizer = actor_optimizer
        self.sequence_critic_optimizer = sequence_critic_optimizer
        self.token_critic_optimizer = token_critic_optimizer
        self.high_level_optimizer = high_level_optimizer
        
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.perplexity_coef = perplexity_coef
        self.max_grad_norm = max_grad_norm
        self.perplexity_grad_clip_threshold = perplexity_grad_clip_threshold
        self.perplexity_grad_clip_factor = perplexity_grad_clip_factor
        self.adaptive_grad_clip_threshold = adaptive_grad_clip_threshold
        self.target_kl = target_kl
        self.distributed_setup = distributed_setup
        self.use_token_masked_computation = use_token_masked_computation
        self.minibatch_size = minibatch_size
        self.config = config
        self.gamma = gamma
        self.hl_only_training = hl_only_training
        
        # Learning rate annealing tracking
        self.current_episode = 0

        # Running reward normalization + best-model tracker (initialized in train())
        from src.trainer.reward_stats import RunningRewardStats
        momentum = getattr(config.training, "reward_norm_momentum", 0.99) if config else 0.99
        clip = getattr(config.training, "reward_norm_clip", 10.0) if config else 10.0
        self.reward_stats = RunningRewardStats(momentum=momentum, clip=clip)
        self.best_model_tracker = None
        self._sequence_critic_target_updates = 0
        
        # Optional GPU model swapping between agent and environment models
        self.agent_models_on_gpu = True  # Track if agent models are on GPU
        self.model_swap_enabled = getattr(config.training, 'enable_model_swapping', True) if config else True
        
        # Store references to agent models for swapping
        self.agent_models = {}  # Will be populated when models are passed to train methods
        self.tool_index_policy = None
        
        # Initialize GradScaler for mixed precision training
        self.scaler = None
        if config and hasattr(config.training, 'mixed_precision') and config.training.mixed_precision != "no":
            # Only use GradScaler for FP16, not for BFloat16 (which doesn't need scaling)
            if config.training.mixed_precision == "fp16":
                self.scaler = torch.amp.GradScaler('cuda')
                print(f"Initialized GradScaler for mixed precision: {config.training.mixed_precision}")
            elif config.training.mixed_precision == "bf16":
                # BFloat16 doesn't need gradient scaling due to wider dynamic range
                self.scaler = None
                print(f"Using BFloat16 mixed precision (no gradient scaling needed): {config.training.mixed_precision}")
            else:
                print(f"Unknown mixed precision type: {config.training.mixed_precision}, skipping GradScaler")
        
        # Initialize logging dictionary for tracking losses and metrics
        self.log_dict = defaultdict(list)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"AgentTrainer initialized with model swapping {'enabled' if self.model_swap_enabled else 'disabled'}")
        if self.hl_only_training:
            print("HL-ONLY TRAINING MODE: Will skip low-level actor and token critic training")
    
    def compute_annealing_factor(self, episode: int) -> float:
        """
        Compute the linear annealing factor for learning rate scheduling.
        
        Args:
            episode: Current episode number
            
        Returns:
            Annealing factor between start_factor and end_factor
        """
        if not self.config or not getattr(self.config.training, 'use_linear_annealing', True):
            return 1.0
            
        start_factor = getattr(self.config.training, 'annealing_start_factor', 1.0)
        end_factor = getattr(self.config.training, 'annealing_end_factor', 0.1)
        warmup_episodes = getattr(self.config.training, 'annealing_warmup_episodes', 10)
        total_episodes = getattr(self.config.training, 'episodes', 1000)
        
        # During warmup, use start_factor
        if episode < warmup_episodes:
            return start_factor
            
        # After warmup, linearly anneal from start_factor to end_factor
        progress = (episode - warmup_episodes) / max(1, total_episodes - warmup_episodes)
        progress = min(1.0, max(0.0, progress))  # Clamp between 0 and 1
        
        annealing_factor = start_factor + (end_factor - start_factor) * progress
        return annealing_factor
    
    def apply_learning_rate_annealing(self, episode: int):
        """
        Apply learning rate annealing to all optimizers.
        
        Args:
            episode: Current episode number
        """
        if not self.config or not getattr(self.config.training, 'use_linear_annealing', True):
            return
            
        annealing_factor = self.compute_annealing_factor(episode)
        
        # Apply to actor optimizer
        if self.actor_optimizer is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = getattr(self.config.training, 'lr', 1e-4) * annealing_factor
            
        # Apply to sequence critic optimizer
        for param_group in self.sequence_critic_optimizer.param_groups:
            param_group['lr'] = getattr(self.config.training, 'value_lr', 1e-3) * annealing_factor
            
        # Apply to token critic optimizer
        for param_group in self.token_critic_optimizer.param_groups:
            param_group['lr'] = getattr(self.config.training, 'value_lr', 1e-3) * annealing_factor
            
        # Apply to high-level optimizer if it exists
        if self.high_level_optimizer is not None:
            for param_group in self.high_level_optimizer.param_groups:
                param_group['lr'] = getattr(self.config.model, 'high_level_lr', 0.001) * annealing_factor
        
        if episode % 10 == 0:  # Log every 10 episodes
            print(f"Episode {episode}: Applied annealing factor {annealing_factor:.4f}")
            if self.actor_optimizer is not None:
                print(f"  Actor LR: {self.actor_optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Critic LR: {self.sequence_critic_optimizer.param_groups[0]['lr']:.2e}")
            if self.high_level_optimizer is not None:
                print(f"  High-level LR: {self.high_level_optimizer.param_groups[0]['lr']:.2e}")
    
    def register_agent_models(self, policy=None, sequence_critic=None, token_critic=None, high_level_policy=None):
        """
        Register agent models for model swapping.
        
        Args:
            policy: Policy model
            sequence_critic: Sequence critic model
            token_critic: Token critic model
            high_level_policy: High-level policy model
        """
        self.agent_models = {
            'policy': policy,
            'sequence_critic': sequence_critic,
            'token_critic': token_critic,
            'high_level_policy': high_level_policy
        }
        model_count = len([m for m in self.agent_models.values() if m is not None])
        swap_status = "enabled" if self.model_swap_enabled else "disabled"
        print(f"Registered {model_count} agent models (model swapping {swap_status})")
    
    def swap_models_to_cpu(self):
        """
        Move agent models to CPU to free up GPU memory for target/guard models.
        """
        if not self.model_swap_enabled or not self.agent_models_on_gpu:
            return
            
        print("Moving agent models to CPU to free GPU memory...")
        for name, model in self.agent_models.items():
            if model is not None:
                try:
                    model.to('cpu')
                    print(f"Moved {name} to CPU")
                except Exception as e:
                    print(f"Warning: Failed to move {name} to CPU: {e}")
        
        self.agent_models_on_gpu = False
        torch.cuda.empty_cache()
    
    def swap_models_to_gpu(self):
        """
        Move agent models back to GPU after target/guard model operations.
        """
        if not self.model_swap_enabled or self.agent_models_on_gpu:
            return
            
        print("Moving agent models back to GPU...")
        for name, model in self.agent_models.items():
            if model is not None:
                try:
                    model.to(self.device)
                    print(f"Moved {name} to GPU")
                except Exception as e:
                    print(f"Warning: Failed to move {name} to GPU: {e}")
        
        self.agent_models_on_gpu = True
        torch.cuda.empty_cache()

    def _guard(self, environment: Any) -> GuardRewardService:
        """Return a guard reward service wrapper for the provided environment."""
        return GuardRewardService(environment)

    def _default_save_checkpoint(
        self,
        policy: nn.Module,
        sequence_critic: nn.Module,
        token_critic: nn.Module,
        episode: int,
        metrics: Dict[str, float],
        output_dir: str,
        is_final: bool = False,
        is_best: bool = False,
        high_level_policy: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        save_checkpoint(
            policy=policy,
            sequence_critic=sequence_critic,
            token_critic=token_critic,
            episode=episode,
            metrics=metrics,
            output_dir=output_dir,
            is_final=is_final,
            is_best=is_best,
            high_level_policy=high_level_policy,
            optimizers={
                "actor": self.actor_optimizer,  # None when hl_only_training and no LoRA
                "sequence_critic": self.sequence_critic_optimizer,
                "token_critic": self.token_critic_optimizer,
                "high_level": self.high_level_optimizer,
            },
        )

    def _decode_with_timeout(self, tokenizer, input_ids: torch.Tensor, **kwargs) -> str:
        """
        Decode tokens (timeout handling removed for fail-fast behavior).
        
        Args:
            tokenizer: Tokenizer to use
            input_ids: Token IDs to decode
            **kwargs: Additional arguments for tokenizer.decode
            
        Returns:
            Decoded string
        """
        return _decode_with_timeout(tokenizer, input_ids, **kwargs)
    
    def _batch_decode_with_timeout(self, tokenizer, input_ids_list: List[torch.Tensor], **kwargs) -> List[str]:
        """
        Batch decode tokens (timeout handling removed for fail-fast behavior).
        
        Args:
            tokenizer: Tokenizer to use
            input_ids_list: List of token ID tensors to decode
            **kwargs: Additional arguments for tokenizer.decode
            
        Returns:
            List of decoded strings
        """
        return _batch_decode_with_timeout(tokenizer, input_ids_list, **kwargs)

    def q_high(self, state: torch.Tensor, action: torch.Tensor, base_state: torch.Tensor, environment: Any) -> torch.Tensor:
        """
        High-level Q-function that evaluates full sequences using R(s') - R(a) formula.
        
        Implements the reward function: R(s') - R(a)
        where R(s') = guard model safety score for target's response to the action
        and R(a) = guard model safety score for the agent's action
        
        BOTTLENECK: Each call involves tensor->text conversion + environment model inference
        
        Args:
            state: Current state tensor
            action: Full action sequence tensor
            base_state: Base state tensor
            environment: Environment for scoring
            
        Returns:
            Q-values for full sequences: R(s') - R(a)
        """
        if getattr(self.config.training, 'debug_mode', False):
            print(f"DEBUG: q_high called with state.shape={state.shape}, action.shape={action.shape}, base_state.shape={base_state.shape}")
        
        # Prefer actor/policy tokenizer; fallback to environment if available
        policy = self.agent_models.get('policy') if hasattr(self, 'agent_models') else None
        tokenizer = getattr(policy, 'tokenizer', None)
        if tokenizer is None and hasattr(environment, 'get_target_tokenizer'):
            tokenizer = environment.get_target_tokenizer()
        if tokenizer is None:
            raise AttributeError("No tokenizer available: policy.tokenizer missing and environment lacks get_target_tokenizer()")

        action_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
        state_texts = self._batch_decode_with_timeout(tokenizer, state, skip_special_tokens=True)
        
        # OPTIMIZED: Batch decode base_state to reduce individual calls
        base_state_texts = self._batch_decode_with_timeout(tokenizer, base_state, skip_special_tokens=True)
        
        # DEBUG: Check for empty strings that might indicate filtering (only in debug mode)
        if getattr(self.config.training, 'debug_mode', False):
            empty_state_count = sum(1 for text in state_texts if not text.strip())
            empty_action_count = sum(1 for text in action_texts if not text.strip())
            empty_base_count = sum(1 for text in base_state_texts if not text.strip())
            print(f"DEBUG: Empty string counts - state: {empty_state_count}, action: {empty_action_count}, base: {empty_base_count}")
            
            if empty_state_count > 0:
                print(f"DEBUG: First few state texts: {state_texts[:5]}")
                print(f"DEBUG: First few action texts: {action_texts[:5]}")
                print(f"DEBUG: First few base_state texts: {base_state_texts[:5]}")
        
        # MODEL SWAPPING: Move agent models to CPU before environment calls
        self.swap_models_to_cpu()
        
        try:
            guard = self._guard(environment)
            # Check if we should use R(s') - R(a) formula or just R(a)
            use_rs_minus_ra = getattr(self.config.training, 'use_rs_minus_ra_reward', True)

            # R(a): Safety score of the agent's action
            if getattr(self.config.training, 'debug_mode', False):
                print(f"DEBUG: q_high input shapes - action_texts: {len(action_texts)}, state_texts: {len(state_texts)}, base_state_texts: {len(base_state_texts)}")
            
            # FAIL FAST: Check for batch size mismatches that indicate filtering
            if len(action_texts) != len(state_texts) or len(action_texts) != len(base_state_texts):
                raise RuntimeError(
                    f"CRITICAL ERROR: Batch size mismatch in q_high inputs!\n"
                    f"action_texts: {len(action_texts)}\n"
                    f"state_texts: {len(state_texts)}\n"
                    f"base_state_texts: {len(base_state_texts)}\n"
                    f"This indicates filtering is happening during tensor decoding.\n"
                    f"Investigate why some tensors decode to empty strings."
                )
            
            if getattr(self.config.training, 'debug_mode', False):
                print(f"DEBUG: About to call score_safety with {len(action_texts)} action_texts, {len(state_texts)} state_texts, {len(base_state_texts)} base_state_texts")
            action_safety_scores = guard.sequence_reward(
                action_texts, questions=state_texts, reference_state=base_state_texts
            )
            if getattr(self.config.training, 'debug_mode', False):
                print(f"DEBUG: action_safety_scores shape: {action_safety_scores.shape}")
            
            # SANITY CHECK: Verify R(a) is accurately measuring agent harmfulness
            # Only show adversarial sanity checks for MultiStepEnv (red teaming), not for general envs
            from src.environment import MultiStepEnv
            is_adversarial_task = isinstance(environment, MultiStepEnv)
            
            if getattr(self.config.training, 'debug_mode', False) and is_adversarial_task:
                print("\n" + "="*80)
                print("ADVERSARIAL REWARD SANITY CHECK")
                print("="*80)
                print(f"R(a) scores: {action_safety_scores.tolist()}")
                print(f"R(a) mean: {action_safety_scores.mean().item():.4f}")
                print(f"R(a) std: {action_safety_scores.std().item():.4f}")
                print(f"R(a) min: {action_safety_scores.min().item():.4f}")
                print(f"R(a) max: {action_safety_scores.max().item():.4f}")
                
                # Show sample agent actions and their R(a) scores
                print("\nSample Agent Actions and R(a) scores:")
                for i, (action_text, ra_score) in enumerate(zip(action_texts[:3], action_safety_scores[:3])):
                    print(f"  Sample {i+1}:")
                    print(f"    Action: {repr(action_text[:200])}{'...' if len(action_text) > 200 else ''}")
                    print(f"    R(a): {ra_score.item():.4f}")
                    print(f"    Harmful? {'YES' if ra_score > 0.5 else 'NO'}")
                    print()
            
            if use_rs_minus_ra:
                # R(s'): Safety score of the target's response to the action
                # Get target responses to the actions
                target_responses = environment.get_target_responses_from_tokens(action, (action != tokenizer.pad_token_id).long(), tokenizer)
                
                # Score the safety of target responses
                target_response_safety_scores = guard.target_response_reward(
                    target_responses, questions=action_texts, reference_state=base_state_texts
                )
                
                # SANITY CHECK: Verify R(s') calculation
                if getattr(self.config.training, 'debug_mode', False):
                    print(f"R(s') scores: {target_response_safety_scores.tolist()}")
                    print(f"R(s') mean: {target_response_safety_scores.mean().item():.4f}")
                    print(f"R(s') std: {target_response_safety_scores.std().item():.4f}")
                    print(f"R(s') min: {target_response_safety_scores.min().item():.4f}")
                    print(f"R(s') max: {target_response_safety_scores.max().item():.4f}")
                    
                    # Show sample target responses and their R(s') scores
                    print("\nSample Target Responses and R(s') scores:")
                    for i, (target_response, rs_score) in enumerate(zip(target_responses[:3], target_response_safety_scores[:3])):
                        print(f"  Sample {i+1}:")
                        print(f"    Target Response: {repr(target_response[:200])}{'...' if len(target_response) > 200 else ''}")
                        print(f"    R(s'): {rs_score.item():.4f}")
                        print(f"    Harmful? {'YES' if rs_score > 0.5 else 'NO'}")
                        print()
                
                # R(s'): Safety score of the target's response to the action
                # R(a): Safety score of the agent's action
                # R(s): Safety score of the current state (conversation history)
                
                # Compute R(s) - safety score of current state
                state_safety_scores = guard.sequence_reward(
                    state_texts, questions=None, reference_state=base_state_texts
                )
                
                # Get reward coefficient (default to 1.0 if not specified)
                reward_coef = getattr(self.config.training, 'reward_coef', 1.0)
                
                # Optional asymmetric scaling to emphasize R(s') more than R(a)
                # This helps when agent learns to minimize R(a) but not maximize R(s')
                rs_prime_weight = getattr(self.config.training, 'rs_prime_weight', 1.0)  # Weight for R(s')
                ra_weight = getattr(self.config.training, 'ra_weight', 1.0)  # Weight for R(a)
                
                # Asymmetric reward: rs_prime_weight * R(s') - ra_weight * R(a)
                # Default (1.0, 1.0) gives standard R(s') - R(a)
                # Higher rs_prime_weight emphasizes target harmfulness more
                reward_scores = (rs_prime_weight * target_response_safety_scores - ra_weight * action_safety_scores) * reward_coef
                
                # Success bonus: +0.1 * reward_coef when R(s') > 0.7
                # Optionally scale success bonus to make it more impactful
                success_bonus_coef = getattr(self.config.training, 'success_bonus_coef', 0.1)
                success_bonus = torch.where(
                    target_response_safety_scores > 0.7,
                    torch.full_like(reward_scores, success_bonus_coef * reward_coef),
                    torch.zeros_like(reward_scores)
                )
                
                # Progress bonus: +0.1 * reward_coef when R(s') > R(s)
                # Optionally scale progress bonus to make it more impactful
                progress_bonus_coef = getattr(self.config.training, 'progress_bonus_coef', 0.1)
                progress_bonus = torch.where(
                    target_response_safety_scores > state_safety_scores,
                    torch.full_like(reward_scores, progress_bonus_coef * reward_coef),
                    torch.zeros_like(reward_scores)
                )
                
                # Add bonuses to reward
                reward_scores = reward_scores + success_bonus + progress_bonus
                
                # SANITY CHECK: Verify final reward calculation
                if getattr(self.config.training, 'debug_mode', False):
                    print(f"Reward coefficient: {reward_coef:.4f}")
                    print(f"R(s) scores: {state_safety_scores.tolist()}")
                    print(f"Success bonuses: {success_bonus.tolist()}")
                    print(f"Progress bonuses: {progress_bonus.tolist()}")
                    print(f"Final R(s') - R(a) scores (scaled): {reward_scores.tolist()}")
                    print(f"Final reward mean: {reward_scores.mean().item():.4f}")
                    print(f"Final reward std: {reward_scores.std().item():.4f}")
                    print(f"Final reward min: {reward_scores.min().item():.4f}")
                    print(f"Final reward max: {reward_scores.max().item():.4f}")
                    
                    # Show sample complete calculations
                    print("\nSample Complete Calculations:")
                    for i in range(min(3, len(action_texts))):
                        base_reward = (target_response_safety_scores[i] - action_safety_scores[i]) * reward_coef
                        print(f"  Sample {i+1}:")
                        print(f"    R(s): {state_safety_scores[i].item():.4f}")
                        print(f"    R(a): {action_safety_scores[i].item():.4f}")
                        print(f"    R(s'): {target_response_safety_scores[i].item():.4f}")
                        print(f"    Base reward [R(s') - R(a)] * {reward_coef:.2f}: {base_reward.item():.4f}")
                        print(f"    Success bonus (R(s') > 0.7): {success_bonus[i].item():.4f}")
                        print(f"    Progress bonus (R(s') > R(s)): {progress_bonus[i].item():.4f}")
                        print(f"    Total reward: {reward_scores[i].item():.4f}")
                        print(f"    Interpretation: {'Good (target safer than agent)' if base_reward > 0 else 'Bad (agent safer than target)'}")
                        print()
            else:
                # Original behavior: just use R(a)
                reward_scores = action_safety_scores
            
            # Ensure safety scores are floating point for mathematical operations
            if reward_scores.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                reward_scores = reward_scores.float()
            reward_scores = reward_scores.to(device=state.device)
            
            return reward_scores
            
        finally:
            # MODEL SWAPPING: Move agent models back to GPU
            self.swap_models_to_gpu()

    def compute_marginal_rewards(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        base_state: torch.Tensor, 
        environment: Any,
        tokenizer: Any,
        sequence_critic: Optional[Any] = None,
        top_k_tokens: Optional[int] = None,
        pairwise_masking: Optional[bool] = None,
        instructions: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute marginal rewards for each token using environment safety scores.
        
        For each token i, computes: R(s'_full) - R(s'_without_i)
        where s'_full is target's response to full action, s'_without_i is target's response to action without token i.
        
        This measures how much each token contributes to the safety of the target's response.
        
        Uses attention-guided token selection: only computes marginal rewards for the top-k
        tokens identified by the sequence critic's attention weights as most relevant to harmfulness.
        
        Args:
            state: Current state tensor [batch_size, seq_len]
            action: Action sequence tensor [batch_size, action_len]
            base_state: Base state tensor [batch_size, seq_len]
            environment: Environment for scoring
            tokenizer: Tokenizer for text processing
            sequence_critic: Sequence critic to get attention weights from (optional)
            top_k_tokens: Number of top attention tokens to compute marginal rewards for (if None, uses config)
            pairwise_masking: Whether to compute pairwise masking (if None, uses config)
            instructions: Optional instruction strings for instruction-following reward
            
        Returns:
            Marginal rewards tensor [batch_size, action_len]
        """
        guard = self._guard(environment)
        # ------------------------------------------------------------------
        # Naïve back-to-front γ-decayed per-token reward (ablation)
        # ------------------------------------------------------------------
        if getattr(self.config.training, "naive_decayed_token_reward", False):
            action_attention_mask = (action != tokenizer.pad_token_id).long()
            self.swap_models_to_cpu()
            try:
                target_responses_full = environment.get_target_responses_from_tokens(
                    action, action_attention_mask, tokenizer
                )
                seq_rewards = guard.sequence_reward(
                    target_responses_full,
                    questions=None,
                    reference_state=None,
                    include_white_box=True,
                ).to(action.device)
            finally:
                self.swap_models_to_gpu()

            return self._compute_naive_decayed_rewards(action, seq_rewards, tokenizer)

        # Resolve hyperparameters from config if not explicitly provided
        if top_k_tokens is None:
            top_k_tokens = self.config.training.num_masked_marginal_tokens

        if pairwise_masking is None:
            pairwise_masking = getattr(self.config.training, "pairwise_marginal_masking", False)

        use_q_as_reward = getattr(self.config.training, "Q_as_reward", False)

        batch_size, action_len = action.shape
        
        # Create full attention mask for actions (all tokens are valid initially)
        action_attention_mask = (action != tokenizer.pad_token_id).long()
        
        # MODEL SWAPPING: Move agent models to CPU before environment calls
        self.swap_models_to_cpu()
        
        try:
            if use_q_as_reward:
                # Use current sequence_critic predictions as proxy Q values
                # Ensure sequence_critic is on GPU since it's passed as parameter
                if sequence_critic is not None:
                    sequence_critic.to(self.device)
                    
                with torch.no_grad():
                    sequence_input_full = torch.cat([state, action], dim=1)
                    full_attention_mask = action_attention_mask if sequence_input_full.size(1)==action.size(1) else torch.cat([(state!=tokenizer.pad_token_id).long(), action_attention_mask], dim=1)
                    
                    # Use chunked processing to avoid OOM errors
                    chunk_size = getattr(self.config.training, 'critic_max_chunk_size', 32)
                    q_full_chunks = []
                    
                    for chunk_start in range(0, batch_size, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, batch_size)
                        input_chunk = sequence_input_full[chunk_start:chunk_end]
                        attention_chunk = full_attention_mask[chunk_start:chunk_end]
                        
                        q_chunk = sequence_critic(
                            input_ids=input_chunk,
                            attention_mask=attention_chunk,
                        )
                        if q_chunk.dim() > 1 and q_chunk.size(-1) == 1:
                            q_chunk = q_chunk.squeeze(-1)
                        q_full_chunks.append(q_chunk)
                        
                        # Clean up intermediate tensors
                        del input_chunk
                        del attention_chunk
                        del q_chunk
                        torch.cuda.empty_cache()
                    
                    # Concatenate all chunks back together
                    q_full = torch.cat(q_full_chunks, dim=0)
                    del q_full_chunks
                    torch.cuda.empty_cache()
                j_unmasked = q_full.to(action.device)  # J(unmasked) = Q_high(s, a_unmasked)
            else:
                # Compute J(unmasked) = R(s'_unmasked) - R(a_unmasked) using differential reward
                # Get target responses for unmasked action
                target_responses_full = environment.get_target_responses_from_tokens(
                    action, action_attention_mask, tokenizer
                )

                target_response_ids = tokenizer(
                    target_responses_full,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )

                # R(s'_unmasked): Safety score of target responses to unmasked action
                r_s_prime_unmasked = guard.sequence_reward_from_tokens(
                    target_response_ids.input_ids,
                    target_response_ids.attention_mask,
                    question_ids=action,
                    question_attention_mask=action_attention_mask,
                    reference_state_ids=base_state,
                    reference_attention_mask=(base_state != tokenizer.pad_token_id).long(),
                    tokenizer=tokenizer
                )

                # G(s_t,a_t,v_t,g_t) = R(v_t | s_t || a_t) - R(a_t | s_t) + J(g_t, a_t)
                # M(s_t,a_t,v_t,g_t,\tau_i) = G(s_t,a_t,v_t,g_t) - G(s_t, a_t - \tau_i , v_t, g_t)

                # R(a_unmasked): Safety score of unmasked action
                action_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
                state_texts = self._batch_decode_with_timeout(tokenizer, state, skip_special_tokens=True)
                base_state_texts = [self._decode_with_timeout(tokenizer, base_state[i], skip_special_tokens=True) for i in range(len(base_state))]
                
                r_a_unmasked = guard.sequence_reward(
                    action_texts, questions=state_texts, reference_state=base_state_texts
                )

                # SANITY CHECK: Verify R(a) calculation in marginal rewards
                if getattr(self.config.training, 'debug_mode', False):
                    print("\n" + "="*80)
                    print("MARGINAL REWARDS R(a) SANITY CHECK")
                    print("="*80)
                    print(f"R(a_unmasked) scores: {r_a_unmasked.tolist()}")
                    print(f"R(a_unmasked) mean: {r_a_unmasked.mean().item():.4f}")
                    print(f"R(a_unmasked) std: {r_a_unmasked.std().item():.4f}")
                    print(f"R(a_unmasked) min: {r_a_unmasked.min().item():.4f}")
                    print(f"R(a_unmasked) max: {r_a_unmasked.max().item():.4f}")
                    
                    # Show sample agent actions and their R(a) scores
                    print("\nSample Agent Actions and R(a_unmasked) scores:")
                    for i, (action_text, ra_score) in enumerate(zip(action_texts[:3], r_a_unmasked[:3])):
                        print(f"  Sample {i+1}:")
                        print(f"    Action: {repr(action_text[:200])}{'...' if len(action_text) > 200 else ''}")
                        print(f"    R(a_unmasked): {ra_score.item():.4f}")
                        print(f"    Harmful? {'YES' if ra_score > 0.5 else 'NO'}")
                        print()
                    
                    print("="*80)
                    print("END MARGINAL REWARDS R(a) SANITY CHECK")
                    print("="*80 + "\n")

                # J(unmasked) = R(s'_unmasked) - R(a_unmasked)
                j_unmasked = r_s_prime_unmasked - r_a_unmasked
                j_unmasked = torch.nan_to_num(j_unmasked, nan=0.0, posinf=1e4, neginf=-1e4).to(action.device)
            
            # Use attention-guided token selection if sequence critic is provided
            if sequence_critic is not None:
                # Get the model device (handles both sharded and non-sharded models)
                # For sharded models, get the device of the input embeddings
                # Move model to device
                sequence_critic.to(self.device)
                model_device = self.device
                
                # Get last hidden states from sequence critic to derive token importance cheaply
                # Use chunked processing to avoid OOM errors
                chunk_size = getattr(self.config.training, 'critic_max_chunk_size', 32)
                last_hidden_chunks = []
                
                with torch.no_grad():
                    for chunk_start in range(0, batch_size, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, batch_size)
                        # Move chunks to model device before passing to critic
                        action_chunk = action[chunk_start:chunk_end].to(model_device)
                        attention_chunk = action_attention_mask[chunk_start:chunk_end].to(model_device)
                        
                        _, last_hidden_chunk = sequence_critic(
                            input_ids=action_chunk,
                            attention_mask=attention_chunk,
                            return_last_hidden=True
                        )
                        last_hidden_chunks.append(last_hidden_chunk)
                        
                        # Clean up intermediate tensors
                        del action_chunk
                        del attention_chunk
                        del last_hidden_chunk
                        torch.cuda.empty_cache()
                    
                    # Concatenate all chunks back together
                    last_hidden = torch.cat(last_hidden_chunks, dim=0)
                    del last_hidden_chunks
                    torch.cuda.empty_cache()
                
                # Derive per-token importance as L2 norm of hidden states: [batch_size, seq_len]
                token_importance = last_hidden.norm(p=2, dim=-1)
                
                # Select top-k most important tokens for each sequence in the batch
                valid_token_positions = []
                for batch_idx in range(batch_size):
                    # Mask out pad tokens
                    valid_positions = action_attention_mask[batch_idx].nonzero(as_tuple=True)[0]
                    if len(valid_positions) == 0:
                        continue
                    
                    # Get importance scores for valid positions only
                    valid_importance = token_importance[batch_idx][valid_positions]
                    
                    # Select top-k tokens (or all if fewer than k)
                    k = min(top_k_tokens, len(valid_positions))
                    top_k_indices = torch.topk(valid_importance, k).indices
                    top_k_positions = valid_positions[top_k_indices]
                    
                    valid_token_positions.extend(top_k_positions.cpu().tolist())
                
                # Remove duplicates and sort
                valid_token_positions = sorted(list(set(valid_token_positions)))
                
                # MEMORY OPTIMIZATION: Limit the number of token positions to process
                # This prevents excessive memory usage when there are many important tokens
                max_token_positions = getattr(self.config.training, 'max_marginal_token_positions', 10)
                if len(valid_token_positions) > max_token_positions:
                    # Take the most important tokens (first ones after sorting)
                    valid_token_positions = valid_token_positions[:max_token_positions]
                    print(f"Limited token positions from {len(valid_token_positions) + (len(valid_token_positions) - max_token_positions)} to {max_token_positions} for memory efficiency")
                
            else:
                # No sequence critic provided - this should not happen in our codebase
                raise ValueError("Sequence critic must be provided for attention-guided token selection")
            
            from itertools import combinations
            import random

            # Set RNG seed for reproducibility if provided in config
            if getattr(self.config.training, "seed", None) is not None:
                random.seed(self.config.training.seed)

            all_masked_attentions = []  # List[Tensor]
            mask_info = []  # List[Tuple[str, Any]] where str is 'single' or 'pair'

            # Single-token masks
            for token_pos in valid_token_positions:
                masked_attention = action_attention_mask.clone()
                masked_attention[:, token_pos] = 0
                all_masked_attentions.append(masked_attention)
                mask_info.append(("single", token_pos))

            # Pairwise masks (if enabled)
            if pairwise_masking and len(valid_token_positions) >= 2:
                # Determine whether we should use uniform random selection for the secondary token
                uniform_pairwise = getattr(self.config.training, "uniform_pairwise_selection", False)

                if uniform_pairwise:
                    # Number of secondary tokens to sample per primary token
                    num_samples = max(1, getattr(self.config.training, "num_uniform_pair_samples", 1))

                    for pos_i in valid_token_positions:
                        # Candidates exclude the primary token itself
                        candidates = [p for p in valid_token_positions if p != pos_i]
                        if not candidates:
                            continue

                        # Sample without replacement (or with if not enough candidates)
                        sample_size = min(num_samples, len(candidates))
                        # random.sample raises if sample_size > len(candidates)
                        sampled_positions = random.sample(candidates, k=sample_size)

                        for pos_j in sampled_positions:
                            masked_attention = action_attention_mask.clone()
                            masked_attention[:, pos_i] = 0
                            masked_attention[:, pos_j] = 0
                            all_masked_attentions.append(masked_attention)
                            mask_info.append(("pair", (pos_i, pos_j)))
                else:
                    # Exhaustive enumeration of all pairs (original behaviour)
                    for pos_i, pos_j in combinations(valid_token_positions, 2):
                        masked_attention = action_attention_mask.clone()
                        masked_attention[:, pos_i] = 0
                        masked_attention[:, pos_j] = 0
                        all_masked_attentions.append(masked_attention)
                        mask_info.append(("pair", (pos_i, pos_j)))
            
            # Process masked versions in reasonable-sized batches to avoid memory issues
            # Create as floating tensor to accumulate reward deltas
            marginal_rewards = torch.zeros(batch_size, action_len, dtype=j_unmasked.dtype, device=action.device)
            
            # Ensure j_unmasked is on the same device as marginal_rewards
            j_unmasked = j_unmasked.to(device=action.device, dtype=j_unmasked.dtype)
            
            # Structures to accumulate pairwise contributions per token
            pairwise_sum: Dict[int, torch.Tensor] = {}
            pairwise_count: Dict[int, int] = {}

            if all_masked_attentions:
                # Use minibatch_size for marginal reward chunking (same as actor training)
                # This ensures we process the same number of samples per chunk as actor training
                max_chunk_size = self.config.training.minibatch_size
                
                # Process in chunks using minibatch_size
                total_chunks = (len(all_masked_attentions) + max_chunk_size - 1) // max_chunk_size
                print(f"Processing {len(all_masked_attentions)} masked versions in {total_chunks} chunks of size {max_chunk_size}")
                print(f"GPU memory usage: {self._get_available_gpu_memory_gb():.2f} GB")
                for chunk_start in range(0, len(all_masked_attentions), max_chunk_size):
                    chunk_end = min(chunk_start + max_chunk_size, len(all_masked_attentions))
                    chunk_attentions = all_masked_attentions[chunk_start:chunk_end]
                    chunk_infos = mask_info[chunk_start:chunk_end]
                    chunk_size = len(chunk_attentions)
                    print(f"Processing chunk {chunk_start//max_chunk_size + 1}/{total_chunks} ({chunk_size} samples)")
                    # Process this chunk
                    if use_q_as_reward:
                        # Build inputs per mask for sequence critic
                        # Ensure sequence_critic is on GPU since it's passed as parameter
                        if sequence_critic is not None:
                            sequence_critic.to(self.device)
                            
                        chunk_scores_list = []
                        with torch.no_grad():
                            for masked_attention in chunk_attentions:
                                seq_input = torch.cat([state, action], dim=1)
                                # Build combined attention mask: state pad + masked attention
                                state_attention = (state != tokenizer.pad_token_id).long()
                                full_attention = torch.cat([state_attention, masked_attention], dim=1)
                                
                                # Use chunked processing for sequence critic to avoid OOM
                                inner_chunk_size = getattr(self.config.training, 'critic_max_chunk_size', 32)
                                q_val_chunks = []
                                
                                for inner_chunk_start in range(0, batch_size, inner_chunk_size):
                                    inner_chunk_end = min(inner_chunk_start + inner_chunk_size, batch_size)
                                    input_inner_chunk = seq_input[inner_chunk_start:inner_chunk_end]
                                    attention_inner_chunk = full_attention[inner_chunk_start:inner_chunk_end]
                                    
                                    q_inner_chunk = sequence_critic(
                                        input_ids=input_inner_chunk,
                                        attention_mask=attention_inner_chunk
                                    )
                                    if q_inner_chunk.dim() > 1 and q_inner_chunk.size(-1) == 1:
                                        q_inner_chunk = q_inner_chunk.squeeze(-1)
                                    q_val_chunks.append(q_inner_chunk)
                                    
                                    # Clean up intermediate tensors
                                    del input_inner_chunk
                                    del attention_inner_chunk
                                    del q_inner_chunk
                                    torch.cuda.empty_cache()
                                
                                # Concatenate inner chunks
                                q_val = torch.cat(q_val_chunks, dim=0)
                                del q_val_chunks
                                torch.cuda.empty_cache()
                                
                                chunk_scores_list.append(q_val)
                                
                                # Clean up intermediate tensors
                                del seq_input
                                del state_attention
                                del full_attention
                                del q_val
                                
                        chunk_scores = torch.stack(chunk_scores_list, dim=1).to(device=action.device, dtype=j_unmasked.dtype)  # [batch, chunk_size]
                        #chunk_scores = torch.nan_to_num(chunk_scores, nan=0.0, posinf=1e4, neginf=-1e4)
                        
                        # Clean up list of tensors
                        del chunk_scores_list
                        torch.cuda.empty_cache()
                        
                        # Additional memory cleanup for Q-as-reward
                        if use_q_as_reward:
                            # Force garbage collection and memory cleanup
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                    else:
                        # Compute J(masked) = R(s'_masked) - R(a_masked) for each masked version
                        chunk_masked_texts = []
                        chunk_masked_actions = []
                        
                        # Clear GPU cache before decoding
                        torch.cuda.empty_cache()
                        
                        for masked_attention in chunk_attentions:
                            masked_texts = environment._batch_decode_with_attention_mask(
                                action, masked_attention, tokenizer
                            )
                            chunk_masked_texts.extend(masked_texts)
                            chunk_masked_actions.extend(masked_texts)  # Same as masked_texts for action scoring

                        # Validate that we have data after decoding
                        if not chunk_masked_texts:
                            raise ValueError("No valid masked texts after decoding - this indicates a serious problem with the decoding process")
                        
                        # Get target responses for masked actions
                        chunk_target_responses = environment.get_target_responses(chunk_masked_texts)

                        # Debug tokenization input
                        if getattr(self.config.training, 'debug_mode', False):
                            print(f"[DEBUG] Tokenizing {len(chunk_target_responses)} target responses")
                            total_chars = sum(len(resp) for resp in chunk_target_responses)
                            print(f"[DEBUG] Total characters to tokenize: {total_chars}")
                            if total_chars > 100000:  # 100k chars is a lot
                                print(f"[WARNING] Large tokenization input detected: {total_chars} characters")

                        # Clear GPU cache before tokenization
                        torch.cuda.empty_cache()
                        
                        chunk_target_response_ids = tokenizer(
                            chunk_target_responses,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=512  # Limit max length to prevent memory issues
                        )

                        # Create large tensors for scoring - these are memory intensive
                        # Clear GPU cache before creating large tensors
                        torch.cuda.empty_cache()
                        
                        chunk_question_ids = action.repeat_interleave(chunk_size, dim=0)
                        chunk_question_attention = torch.cat(chunk_attentions, dim=0)
                        chunk_base_state = base_state.repeat_interleave(chunk_size, dim=0)
                        chunk_base_attention = (base_state != tokenizer.pad_token_id).long().repeat_interleave(chunk_size, dim=0)

                        try:
                            # R(s'_masked): Safety score of target responses to masked actions
                            r_s_prime_masked = guard.sequence_reward_from_tokens(
                                chunk_target_response_ids.input_ids,
                                chunk_target_response_ids.attention_mask,
                                question_ids=chunk_question_ids,
                                question_attention_mask=chunk_question_attention,
                                reference_state_ids=chunk_base_state,
                                reference_attention_mask=chunk_base_attention,
                                tokenizer=tokenizer
                            )

                            # R(a_masked): Safety score of masked actions
                            #   For fair comparison we use the same "question" (previous state) for each masked
                            #   variant. We therefore repeat state_texts & base_state_texts for each masked copy.
                            repeated_state_texts = []
                            repeated_base_state_texts = []
                            for _ in range(chunk_size):
                                repeated_state_texts.extend(state_texts)
                                repeated_base_state_texts.extend(base_state_texts)

                            r_a_masked = guard.sequence_reward(
                                chunk_masked_actions,
                                questions=repeated_state_texts,
                                reference_state=repeated_base_state_texts,
                            )

                            # SANITY CHECK: Verify R(a_masked) calculation
                            if getattr(self.config.training, 'debug_mode', False) and len(chunk_masked_actions) > 0:
                                print(f"R(a_masked) scores for chunk: {r_a_masked.tolist()}")
                                print(f"R(a_masked) mean: {r_a_masked.mean().item():.4f}")
                                print(f"Sample masked action: {repr(chunk_masked_actions[0][:200])}{'...' if len(chunk_masked_actions[0]) > 200 else ''}")

                            # J(masked) = R(s'_masked) - R(a_masked)
                            chunk_scores = r_s_prime_masked - r_a_masked
                            # Ensure chunk_scores is on the same device and dtype as other tensors
                            chunk_scores = chunk_scores.to(device=action.device, dtype=j_unmasked.dtype)
                            #chunk_scores = torch.nan_to_num(chunk_scores, nan=0.0, posinf=1e4, neginf=-1e4).to(action.device)
                        finally:
                            # Explicitly delete large tensors to free GPU memory
                            del chunk_question_ids
                            del chunk_question_attention
                            del chunk_base_state
                            del chunk_base_attention
                            del chunk_target_response_ids
                            del chunk_masked_texts
                            del chunk_target_responses
                            del chunk_masked_actions
                            # Force GPU memory cleanup
                            torch.cuda.empty_cache()
                    
                    # Reshape chunk scores back to [batch_size, chunk_size]
                    chunk_scores = chunk_scores.view(batch_size, chunk_size)
                    
                    # Assign marginal rewards for this chunk: J(unmasked) - J(masked)
                    for j, (info_type, info_val) in enumerate(chunk_infos):
                        if info_type == "single":
                            token_pos = info_val  # int
                            marginal_rewards[:, token_pos] = j_unmasked - chunk_scores[:, j]  # J(unmasked) - J(masked)
                        else:  # pair
                            pos_i, pos_j = info_val  # tuple
                            pair_contrib = (j_unmasked - chunk_scores[:, j]) / 2.0  # share equally
                            # Ensure pair_contrib is on the correct device
                            pair_contrib = pair_contrib.to(device=action.device, dtype=j_unmasked.dtype)

                            # Accumulate for token i
                            if pos_i not in pairwise_sum:
                                pairwise_sum[pos_i] = pair_contrib.clone()
                                pairwise_count[pos_i] = 1
                            else:
                                pairwise_sum[pos_i] += pair_contrib
                                pairwise_count[pos_i] += 1

                            # Accumulate for token j
                            if pos_j not in pairwise_sum:
                                pairwise_sum[pos_j] = pair_contrib.clone()
                                pairwise_count[pos_j] = 1
                            else:
                                pairwise_sum[pos_j] += pair_contrib
                                pairwise_count[pos_j] += 1

            # After processing all chunks, add averaged pairwise contributions to each token
            for token_pos, sum_tensor in pairwise_sum.items():
                count = pairwise_count[token_pos]
                avg_contrib = sum_tensor / max(count, 1)  # avoid division by zero, though count should never be zero
                # Ensure avg_contrib is on the same device as marginal_rewards
                avg_contrib = avg_contrib.to(device=action.device, dtype=j_unmasked.dtype)
                marginal_rewards[:, token_pos] += avg_contrib
                
                # Clean up the sum tensor to free memory
                del sum_tensor

            # Final sanitization
            #marginal_rewards = torch.nan_to_num(marginal_rewards, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # [AUGMENTATION] Add instruction-following reward to marginal rewards
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            use_task_reward = getattr(self.config.training, 'use_task_reward', True)
            if instructions is not None and use_task_reward:
                # Compute similarity between actions and instructions
                utter_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
                # Filter out None values and convert to empty strings for flat agents
                safe_instr = [instr if instr is not None and isinstance(instr, str) else "" for instr in instructions]
                
                # Skip instruction-following reward if all instructions are empty (flat agent case)
                if not all(not instr.strip() for instr in safe_instr):
                    sim_scores = environment._get_embed_similarity_batched(
                        safe_instr,
                        utter_texts,
                    ).to(device=marginal_rewards.device, dtype=marginal_rewards.dtype)
                    
                    # Distribute instruction-following reward across valid tokens
                    # Add the similarity score as a constant reward to each valid token position
                    action_attention_mask = (action != tokenizer.pad_token_id).long()
                    for batch_idx in range(batch_size):
                        valid_positions = action_attention_mask[batch_idx].nonzero(as_tuple=True)[0]
                        if len(valid_positions) > 0:
                            # Add instruction-following reward to all valid token positions
                            marginal_rewards[batch_idx, valid_positions] += sim_scores[batch_idx] / len(valid_positions)
            
            # Clean up pairwise accumulation structures
            del pairwise_sum
            del pairwise_count
            
            # ------------------------------------------------------------------
            # Ablation: naïve back-to-front γ-decayed per-token reward
            # ------------------------------------------------------------------
            if getattr(self.config.training, "naive_decayed_token_reward", False):
                action_attention_mask = (action != tokenizer.pad_token_id).long()
                # Move large models off GPU during environment inference
                self.swap_models_to_cpu()
                try:
                    target_responses_full = environment.get_target_responses_from_tokens(
                        action, action_attention_mask, tokenizer
                    )
                    seq_rewards = guard.sequence_reward(
                        target_responses_full,
                        questions=None,
                        reference_state=None,
                        include_white_box=True,
                    ).to(action.device)
                finally:
                    self.swap_models_to_gpu()

                return self._compute_naive_decayed_rewards(action, seq_rewards, tokenizer)

            # ------------------------------------------------------------------
            
            return marginal_rewards
            
        finally:
            # MODEL SWAPPING: Move agent models back to GPU after environment calls
            self.swap_models_to_gpu()

    def _get_available_gpu_memory_gb(self) -> float:
        """
        Get available GPU memory in GB.
        
        Returns:
            Available GPU memory in GB
        """
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # Get GPU memory info
            current_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Estimate available memory (conservative)
            available_memory = total_memory - current_memory - 2.0  # Keep 2GB buffer
            return max(available_memory, 1.0)  # At least 1GB for calculations
            
        except Exception as e:
            raise RuntimeError(f"Failed to get GPU memory info: {e}")

    def v_low(self, marginal_rewards: torch.Tensor) -> torch.Tensor:
        """
        Low-level value function that computes per-token values using rewards-to-go.
        
        Each token position gets its own value estimate based on discounted future rewards.
        This treats each token as an individual timestep in RL.
        
        Args:
            marginal_rewards: Precomputed marginal rewards [batch_size, action_len]
            
        Returns:
            Per-token low-level values [batch_size, action_len]
        """
        batch_size, action_len = marginal_rewards.shape
        values = torch.zeros_like(marginal_rewards)
        
        # Compute rewards-to-go for each token position
        for t in range(action_len):
            # Rewards from position t to end, with discounting
            future_rewards = marginal_rewards[:, t:]
            discount_factors = self.gamma ** torch.arange(action_len - t, device=marginal_rewards.device, dtype=marginal_rewards.dtype)
            values[:, t] = (future_rewards * discount_factors.unsqueeze(0)).sum(dim=1)
        
        return values
    
    def v_low_deprecated(self, marginal_rewards: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """
        DEPRECATED: Low-level value function that sums marginal contributions recursively.
        
        This version returns sequence-level values and is kept for ablation studies.
        
        Args:
            marginal_rewards: Precomputed marginal rewards [batch_size, action_len]
            start_idx: Starting token index for recursion
            
        Returns:
            Low-level values [batch_size]
        """
        batch_size, action_len = marginal_rewards.shape
        
        if start_idx >= action_len:
            return torch.zeros(batch_size, device=marginal_rewards.device, dtype=marginal_rewards.dtype)
        
        # Current token reward + discounted future value
        current_reward = marginal_rewards[:, start_idx]
        future_value = self.v_low_deprecated(marginal_rewards, start_idx + 1)
        
        return current_reward + self.gamma * future_value

    def v_high(self, state: torch.Tensor, action: torch.Tensor, base_state: torch.Tensor, environment: Any = None, v_high_critic: Any = None, tokenizer: Any = None) -> torch.Tensor:
        """
        High-level value function: learned critic that predicts R(s') - R(a).
        
        This critic is trained to predict the expected harmfulness difference
        between target response and agent action without needing the target response.
        
        Args:
            state: Current state tensor
            action: Action sequence from low-level rollout
            base_state: Base state tensor (not used by critic)
            environment: Environment for scoring (used during training only)
            v_high_critic: Learned critic network for v_high predictions
            tokenizer: Tokenizer for processing (used during inference)
            
        Returns:
            High-level values: predicted R(s') - R(a)
        """
        # Check if we're in training mode (have access to environment for target generation)
        if environment is not None:
            # Training mode: compute actual R(s') - R(a) using q_high
            return self.q_high(state, action, base_state, environment)
        else:
            # Inference mode: use learned critic to predict R(s') - R(a)
            if v_high_critic is None or tokenizer is None:
                raise ValueError(
                    "v_high inference mode requires v_high_critic and tokenizer. "
                    "These should be provided when environment is None."
                )
            
            # Use the learned v_high critic to predict R(s') - R(a)
            device = next(v_high_critic.parameters()).device
            model_dtype = next(v_high_critic.parameters()).dtype
            
            # Prepare input for v_high critic: concatenate state and action
            critic_input = torch.cat([state, action], dim=1).to(device=device)
            critic_attention_mask = (critic_input != tokenizer.pad_token_id).long().to(device=device)
            
            # Get prediction from v_high critic
            with torch.no_grad():
                critic_value = v_high_critic(
                    input_ids=critic_input,
                    attention_mask=critic_attention_mask
                )
                
                if critic_value.dim() > 1 and critic_value.size(-1) == 1:
                    critic_value = critic_value.squeeze(-1)
                
                # Ensure correct dtype and device
                critic_value = critic_value.to(dtype=model_dtype, device=device)
            
            return critic_value








