"""
Trainer extensions for hierarchical 20 Questions agent training.

This module provides extensions to the MCQAgentTrainer to support
hierarchical 20 Questions training with dual reward streams.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import json
from src.mcq_agent_trainer import MCQAgentTrainer
from src.hierarchical_twentyq_env import HierarchicalTwentyQuestionsEnvironment
from src.hierarchical_20q_prompts import (
    get_high_level_system_prompt,
    get_low_level_system_prompt,
    detect_guess_instruction
)


def is_hierarchical_twenty_questions_env(environment: Any) -> bool:
    """
    Check if the environment is a hierarchical 20 Questions environment.
    
    Args:
        environment: Environment object to check
        
    Returns:
        True if it's a hierarchical 20Q environment
    """
    return (
        hasattr(environment, 'task') and 
        environment.task == "hierarchical_twenty_questions"
    ) or isinstance(environment, HierarchicalTwentyQuestionsEnvironment)


class HierarchicalTwentyQTrainerMixin:
    """
    Mixin class to add hierarchical 20Q support to MCQAgentTrainer.
    
    This mixin provides the additional methods needed to handle hierarchical
    20Q training with separate high-level and low-level policies.
    """
    
    def collect_rollout_hierarchical_20q(
        self,
        policy,
        environment: HierarchicalTwentyQuestionsEnvironment,
        tokenizer,
        rollout_steps: int,
        config,
        high_level_policy=None,
        templates_path=None,
        episode=0
    ):
        """
        Collect rollout data for hierarchical 20 Questions training.
        
        This uses the proven single-agent rollout pattern from MCQAgentTrainer,
        extended to capture both high-level (meta-cognitive) and low-level (question generation) actions.
        
        Args:
            policy: Low-level policy (question generation)
            environment: Hierarchical 20Q environment
            tokenizer: Tokenizer for text processing
            rollout_steps: Number of steps to collect
            config: Training configuration
            high_level_policy: High-level policy (meta-cognitive)
            templates_path: Path to templates (unused for hierarchical 20Q)
            episode: Current episode number
            
        Returns:
            RolloutData containing rollout data with dual reward streams
        """
        from src.trainer.rollout import RolloutData
        
        # Initialize rollout tracking (following single-agent pattern)
        batch_states = []
        batch_actions = []  # LL actions (token IDs)
        batch_log_probs = []  # LL log probs
        batch_hl_rews = []
        batch_ll_rews = []
        batch_dones = []
        batch_base_states = []
        batch_next_states = []
        batch_instructions = []
        
        # High-level policy tracking
        batch_hl_actions = []
        batch_hl_log_probs = []
        batch_hl_entropy = []
        
        # Metrics collection for submit stats
        all_hl_actions = []
        all_ll_actions = []
        all_responses = []
        all_hl_rewards = []
        all_ll_rewards = []
        
        # Reset environment
        batch_size = getattr(config.training, 'batch_size', 4)
        if self.config.training.debug_mode:
            print(f"[DEBUG] Resetting hierarchical 20Q environment with batch_size: {batch_size}")
        
        state = environment.reset(batch_size)
        
        steps_collected = 0
        episode_wins_this_step = 0
        episode_steps_for_wins = []
        
        # Track conversation completion status (following single-agent pattern)
        already_dones = [False] * batch_size
        
        # Two-phase completion logic: collect until target reached AND all conversations complete
        rollout_target_reached = False
        
        if self.config.training.debug_mode:
            print(f"[DEBUG] Starting hierarchical 20Q rollout collection for episode {episode}")
            print(f"[DEBUG] Target rollout steps: {rollout_steps}, Batch size: {batch_size}")
        
        while True:
            # Find indices of unfinished episodes
            unfinished_indices = [i for i, done in enumerate(already_dones) if not done]
            
            if self.config.training.debug_mode:
                print(f"[DEBUG] Step iteration: unfinished={len(unfinished_indices)}, steps_collected={steps_collected}")
            
            if not unfinished_indices:
                # All episodes finished - check if we should stop
                if rollout_target_reached:
                    if self.config.training.debug_mode:
                        print(f"[DEBUG] All episodes finished and target reached, breaking")
                    break
                
                # Reset environment to start new episodes
                if self.config.training.debug_mode:
                    print(f"[DEBUG] All episodes finished but target not reached, resetting environment")
                state = environment.reset(batch_size)
                already_dones = [False] * batch_size
                unfinished_indices = list(range(batch_size))
            
            # Prepare state for unfinished episodes only
            unfinished_states = [state[i] for i in unfinished_indices]
            
            # Generate high-level actions (meta-cognitive instructions) for unfinished episodes
            hl_actions, hl_log_probs = self._generate_hl_actions_20q(
                unfinished_states, environment, high_level_policy, config
            )
            
            if self.config.training.debug_mode:
                print(f"[DEBUG] Generated {len(hl_actions)} HL actions for {len(unfinished_states)} unfinished states")
            
            # Generate low-level actions (questions/guesses) based on HL instructions
            ll_actions, ll_action_tokens, ll_log_probs = self._generate_ll_actions_20q(
                unfinished_states, hl_actions, policy, environment, tokenizer, config
            )
            
            if self.config.training.debug_mode:
                print(f"[DEBUG] Generated {len(ll_actions)} LL actions")
            
            # Execute actions in environment (only for unfinished episodes)
            next_state, responses, hl_rewards, ll_rewards, dones, info = environment.step(
                hl_actions, ll_actions
            )
            
            # Collect data for metrics calculation
            all_hl_actions.extend(hl_actions)
            all_ll_actions.extend(ll_actions)
            all_responses.extend(responses)
            all_hl_rewards.extend(hl_rewards.tolist() if hasattr(hl_rewards, 'tolist') else hl_rewards)
            all_ll_rewards.extend(ll_rewards.tolist() if hasattr(ll_rewards, 'tolist') else ll_rewards)
            
            if self.config.training.debug_mode:
                print(f"[DEBUG] Environment step: next_state={len(next_state)}, hl_rewards shape={hl_rewards.shape}, dones shape={dones.shape}")
            
            # Update batch lists for unfinished episodes only (single-agent pattern)
            for i in range(len(unfinished_indices)):
                unfinished_idx = unfinished_indices[i]
                
                # Format conversation state (matching single-agent pattern)
                def format_conversation_state(x):
                    if isinstance(x, list):
                        # This is a conversation history (list of strings)
                        # Join with newlines to preserve structure for display
                        return "\n".join(str(item) for item in x if item)
                    else:
                        # This is a single string or other type
                        return str(x) if x else ""
                
                # Store data for this unfinished episode
                batch_states.append(format_conversation_state(state[unfinished_idx]))
                batch_next_states.append(format_conversation_state(next_state[unfinished_idx]))
                batch_actions.append(ll_action_tokens[i])  # LL action tokens
                batch_log_probs.append(ll_log_probs[i])  # LL log probs
                batch_hl_rews.append(hl_rewards[i].item())  # Use i, not unfinished_idx
                batch_ll_rews.append(ll_rewards[i].item())  # Use i, not unfinished_idx
                batch_dones.append(dones[i].item())  # Use i, not unfinished_idx
                batch_base_states.append(format_conversation_state(state[unfinished_idx]))
                batch_instructions.append(hl_actions[i])
                
                # High-level policy data
                batch_hl_actions.append(hl_actions[i])
                batch_hl_log_probs.append(hl_log_probs[i])
                batch_hl_entropy.append(0.0)  # Placeholder for entropy
                
                # Update completion status
                if dones[i]:  # Use i, not unfinished_idx
                    already_dones[unfinished_idx] = True
                    # Track episode wins
                    if hl_rewards[i] >= 0.0:  # Use i, not unfinished_idx
                        episode_wins_this_step += 1
                        questions_asked = environment.conversations[unfinished_idx].question_count
                        episode_steps_for_wins.append(questions_asked)
                        if self.config.training.debug_mode:
                            print(f"[DEBUG] Episode {unfinished_idx} won in {questions_asked} questions!")
            
            # Update state for next iteration
            state = next_state
            
            # Count steps collected
            steps_collected += len(unfinished_indices)
            
            if self.config.training.debug_mode:
                print(f"[DEBUG] Added {len(unfinished_indices)} steps, total: {steps_collected}")
            
            # Mark that we've reached the target number of steps
            if steps_collected >= rollout_steps and not rollout_target_reached:
                rollout_target_reached = True
                if self.config.training.debug_mode:
                    print(f"[DEBUG] Reached rollout target ({rollout_steps}), continuing until all conversations complete")
        
        # Process post-hoc LL rewards for efficient batch processing
        if hasattr(environment, 'process_posthoc_ll_rewards'):
            ll_rewards_map = environment.process_posthoc_ll_rewards()
            if ll_rewards_map:
                # Update batch_ll_rews with actual LL rewards
                for step_idx, ll_reward in ll_rewards_map.items():
                    if step_idx < len(batch_ll_rews):
                        batch_ll_rews[step_idx] = ll_reward
                if self.config.training.debug_mode:
                    print(f"[DEBUG] Updated {len(ll_rewards_map)} LL rewards from post-hoc processing")
        
        # Validate batch data (ensure all lists have same length)
        batch_length = len(batch_states)
        
        if self.config.training.debug_mode:
            print(f"[DEBUG] Final batch data lengths:")
            print(f"  - batch_states: {len(batch_states)}")
            print(f"  - batch_actions: {len(batch_actions)}")
            print(f"  - batch_log_probs: {len(batch_log_probs)}")
            print(f"  - batch_hl_rews: {len(batch_hl_rews)}")
            print(f"  - batch_ll_rews: {len(batch_ll_rews)}")
            print(f"  - batch_dones: {len(batch_dones)}")
            print(f"  - batch_hl_actions: {len(batch_hl_actions)}")
        
        if batch_length == 0:
            raise RuntimeError(f"Empty batch collected in hierarchical 20Q! steps_collected={steps_collected}, rollout_steps={rollout_steps}, batch_size={batch_size}")
        
        # Validate all lists match
        assert len(batch_next_states) == batch_length, f"batch_next_states mismatch: {len(batch_next_states)} vs {batch_length}"
        assert len(batch_actions) == batch_length, f"batch_actions mismatch: {len(batch_actions)} vs {batch_length}"
        assert len(batch_log_probs) == batch_length, f"batch_log_probs mismatch: {len(batch_log_probs)} vs {batch_length}"
        assert len(batch_hl_rews) == batch_length, f"batch_hl_rews mismatch: {len(batch_hl_rews)} vs {batch_length}"
        assert len(batch_ll_rews) == batch_length, f"batch_ll_rews mismatch: {len(batch_ll_rews)} vs {batch_length}"
        assert len(batch_dones) == batch_length, f"batch_dones mismatch: {len(batch_dones)} vs {batch_length}"
        assert len(batch_hl_actions) == batch_length, f"batch_hl_actions mismatch: {len(batch_hl_actions)} vs {batch_length}"
        
        # Pad log probs to handle variable sequence lengths
        if not batch_log_probs:
            raise ValueError("No low-level log_probs collected")
        
        # Filter out empty tensors and validate
        valid_log_probs = []
        for idx, log_prob in enumerate(batch_log_probs):
            if not isinstance(log_prob, torch.Tensor):
                raise ValueError(f"Invalid LL log_prob type at index {idx}: {type(log_prob)}")
            if log_prob.numel() == 0:
                # Replace empty tensor with a single zero value
                valid_log_probs.append(torch.tensor([0.0]))
                if self.config.training.debug_mode:
                    print(f"[DEBUG] Replaced empty LL log_prob at index {idx} with [0.0]")
            else:
                # Ensure the tensor is 1D (not 0D scalar)
                if log_prob.dim() == 0:  # 0-dimensional scalar tensor
                    log_prob = log_prob.unsqueeze(0)  # Convert to 1D tensor
                valid_log_probs.append(log_prob)
        
        if not valid_log_probs:
            raise ValueError("All low-level log_probs were empty after filtering")
        
        # Additional validation before padding
        if self.config.training.debug_mode:
            print(f"[DEBUG] About to pad {len(valid_log_probs)} LL log probs")
            for i, prob in enumerate(valid_log_probs):
                print(f"[DEBUG] LL log_prob {i}: shape={prob.shape}, numel={prob.numel()}")
        
        # Final check - ensure no empty tensors remain (fail fast to investigate root cause)
        for idx, prob in enumerate(valid_log_probs):
            if prob.numel() == 0:
                print(f"[ERROR] Empty tensor found at index {idx} after filtering!")
                print(f"[ERROR] This indicates the low-level policy is returning empty tensors")
                print(f"[ERROR] Tensor details: type={type(prob)}, shape={prob.shape if hasattr(prob, 'shape') else 'N/A'}")
                raise ValueError(f"Empty tensor found at index {idx} after filtering - investigate low-level policy implementation")
        
        # Additional safety check - ensure we have at least one tensor
        if not valid_log_probs:
            raise ValueError("No valid low-level log probs after filtering")
        
        # Final validation before padding
        for idx, prob in enumerate(valid_log_probs):
            if not isinstance(prob, torch.Tensor) or prob.numel() == 0:
                raise ValueError(f"Invalid tensor at final index {idx}: type={type(prob)}, numel={prob.numel() if hasattr(prob, 'numel') else 'N/A'}")
        
        # Import left_pad_sequence for proper left-padding of decoder-only models
        from src.mcq_agent_trainer import left_pad_sequence
        padded_batch_log_probs = left_pad_sequence(valid_log_probs, batch_first=True, padding_value=-5.0)
        
        # Pad high-level log probs
        if not batch_hl_log_probs:
            raise ValueError("No high-level log_probs collected")
        
        # Filter out empty tensors and validate
        print(f"[INVESTIGATION] Processing {len(batch_hl_log_probs)} high-level log probs")
        valid_hl_log_probs = []
        tensor_info = []
        
        for idx, log_prob in enumerate(batch_hl_log_probs):
            if not isinstance(log_prob, torch.Tensor):
                raise ValueError(f"Invalid HL log_prob type at index {idx}: {type(log_prob)}")
            if log_prob.numel() == 0:
                print(f"[INVESTIGATION] Found empty HL log_prob at index {idx}!")
                print(f"[INVESTIGATION] This is the root cause - investigating why high-level policy returned empty tensor")
                raise ValueError(f"Empty HL log_prob at index {idx} - investigate high-level policy implementation")
            else:
                # Ensure the tensor is 1D (not 0D scalar)
                if log_prob.dim() == 0:  # 0-dimensional scalar tensor
                    log_prob = log_prob.unsqueeze(0)  # Convert to 1D tensor
                valid_hl_log_probs.append(log_prob)
                tensor_info.append(f"{idx}:{log_prob.shape}")
        
        # Batch report tensor info
        if tensor_info:
            print(f"[INVESTIGATION] HL log probs summary: {', '.join(tensor_info[:10])}{'...' if len(tensor_info) > 10 else ''}")
        
        if not valid_hl_log_probs:
            raise ValueError("All high-level log_probs were empty after filtering")
        
        # Additional validation before padding
        if self.config.training.debug_mode:
            print(f"[DEBUG] About to pad {len(valid_hl_log_probs)} HL log probs")
            for i, prob in enumerate(valid_hl_log_probs):
                print(f"[DEBUG] HL log_prob {i}: shape={prob.shape}, numel={prob.numel()}")
        
        # Final check - ensure no empty tensors remain (fail fast to investigate root cause)
        for idx, prob in enumerate(valid_hl_log_probs):
            if prob.numel() == 0:
                print(f"[ERROR] Empty tensor found at index {idx} after filtering!")
                print(f"[ERROR] This indicates the high-level policy is returning empty tensors")
                print(f"[ERROR] Tensor details: type={type(prob)}, shape={prob.shape if hasattr(prob, 'shape') else 'N/A'}")
                raise ValueError(f"Empty tensor found at index {idx} after filtering - investigate high-level policy implementation")
        
        # Additional safety check - ensure we have at least one tensor
        if not valid_hl_log_probs:
            raise ValueError("No valid high-level log probs after filtering")
        
        # Final validation before padding
        for idx, prob in enumerate(valid_hl_log_probs):
            if not isinstance(prob, torch.Tensor) or prob.numel() == 0:
                raise ValueError(f"Invalid tensor at final index {idx}: type={type(prob)}, numel={prob.numel() if hasattr(prob, 'numel') else 'N/A'}")
        
        padded_batch_hl_log_probs = left_pad_sequence(valid_hl_log_probs, batch_first=True, padding_value=-5.0)
        
        # Get conversations from environment for trajectory logging
        conversations = []
        if hasattr(environment, 'conversations') and environment.conversations:
            conversations = environment.conversations.copy()
        
        # Build persistent submit metrics using the proper metrics update function
        persistent_submit_metrics = {}
        
        # Call the hierarchical metrics update function if we have data
        if all_ll_actions and all_responses:
            self.update_submit_metrics_hierarchical_20q(
                persistent_submit_metrics,
                all_hl_actions,
                all_ll_actions,
                all_responses,
                all_hl_rewards,
                all_ll_rewards
            )
        
        # Add episode-specific metrics
        persistent_submit_metrics['submit_wins'] = episode_wins_this_step
        persistent_submit_metrics['submit_episode_efficiency'] = episode_steps_for_wins
        
        # Calculate average/min/max episode efficiency from the list (similar to general/twenty_questions.py)
        if episode_steps_for_wins:
            persistent_submit_metrics['submit_avg_episode_efficiency'] = float(sum(episode_steps_for_wins) / len(episode_steps_for_wins))
            persistent_submit_metrics['submit_min_episode_efficiency'] = int(min(episode_steps_for_wins))
            persistent_submit_metrics['submit_max_episode_efficiency'] = int(max(episode_steps_for_wins))
        else:
            persistent_submit_metrics['submit_avg_episode_efficiency'] = 0.0
            persistent_submit_metrics['submit_min_episode_efficiency'] = 0
            persistent_submit_metrics['submit_max_episode_efficiency'] = 0
        
        # Return RolloutData namedtuple (matching single-agent pattern)
        return RolloutData(
            batch_state=batch_states,
            batch_next_state=batch_next_states,
            batch_acts=batch_actions,
            batch_log_probs=padded_batch_log_probs,
            batch_rews=batch_hl_rews,  # Use HL rewards as primary rewards
            batch_dones=batch_dones,
            batch_base_state=batch_base_states,
            conversations=conversations,
            high_level_actions=batch_hl_actions,
            high_level_log_probs=padded_batch_hl_log_probs,
            high_level_instructions=batch_instructions,
            collected_template_ids=batch_hl_entropy,
            persistent_submit_metrics=persistent_submit_metrics
        )
    
    def _generate_hl_actions_20q(
        self,
        states: List[List[str]],
        environment: HierarchicalTwentyQuestionsEnvironment,
        high_level_policy,
        config
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate high-level meta-cognitive instructions using batched processing.
        
        Args:
            states: Current conversation states
            environment: Hierarchical 20Q environment
            high_level_policy: High-level policy model
            config: Training configuration
            
        Returns:
            Tuple of (hl_actions, hl_log_probs)
        """
        if high_level_policy is None:
            # Default instructions if no high-level policy
            default_instruction = "Ask a strategic question to gather information about the mystery word."
            hl_actions = [default_instruction] * len(states)
            hl_log_probs = [torch.tensor([0.0])] * len(states)
            return hl_actions, hl_log_probs
        
        # PERFORMANCE FIX: Batch all context generation and model calls
        # Generate all contexts at once instead of one-by-one
        conv_indices = list(range(len(states)))
        hl_contexts = environment.get_batch_conversation_contexts_hl(conv_indices)
        
        # PERFORMANCE FIX: Single batched model call instead of individual calls
        if not hasattr(high_level_policy, 'act'):
            raise ValueError("High-level policy must implement act(contexts) and return (actions, log_probs)")
        
        # Batch all contexts in a single call
        result = high_level_policy.act(hl_contexts)
        
        if not (isinstance(result, tuple) and len(result) >= 2):
            raise ValueError(f"High-level policy act() returned unexpected result: {type(result)}")
        
        hl_actions, hl_log_probs = result[0], result[1]
        
        # Validate actions
        if not hl_actions:
            raise ValueError("High-level policy returned empty actions")
        
        # Validate log probs - handle both tensor and list cases
        if isinstance(hl_log_probs, torch.Tensor):
            if self.config.training.debug_mode:
                print(f"[DEBUG] hl_log_probs is a tensor with shape: {hl_log_probs.shape}, numel: {hl_log_probs.numel()}")
                print(f"[DEBUG] Tensor dtype: {hl_log_probs.dtype}, device: {hl_log_probs.device}")
            
            if hl_log_probs.numel() == 0:
                raise ValueError("High-level policy returned empty log_probs tensor - this indicates a problem with the high-level policy implementation")
            
            # Convert tensor to list for consistency with downstream code
            # Ensure each element is a valid tensor with at least one element
            hl_log_probs_list = []
            for i in range(hl_log_probs.shape[0]):
                element = hl_log_probs[i]
                if self.config.training.debug_mode:
                    print(f"[DEBUG] Tensor element {i}: shape={element.shape}, numel={element.numel()}")
                
                if element.numel() == 0:
                    # This is the root cause - investigate why the high-level policy is returning empty tensors
                    print(f"[ERROR] High-level policy returned empty tensor at index {i}!")
                    print(f"[ERROR] This suggests the high-level policy is not properly generating log probabilities")
                    print(f"[ERROR] Context for this element: {hl_contexts[i] if i < len(hl_contexts) else 'N/A'}")
                    raise ValueError(f"High-level policy returned empty tensor at index {i} - investigate the high-level policy implementation")
                else:
                    # Ensure the element is a proper 1D tensor (not 0D scalar)
                    if not isinstance(element, torch.Tensor):
                        element = torch.tensor([element])
                    elif element.dim() == 0:  # 0-dimensional scalar tensor
                        element = element.unsqueeze(0)  # Convert to 1D tensor
                    hl_log_probs_list.append(element)
            
            hl_log_probs = hl_log_probs_list
        elif isinstance(hl_log_probs, list):
            if self.config.training.debug_mode:
                print(f"[DEBUG] hl_log_probs is a list with length: {len(hl_log_probs)}")
            
            if not hl_log_probs:
                raise ValueError("High-level policy returned empty log_probs list - this indicates a problem with the high-level policy implementation")
            
            # Validate and investigate each element in the list
            validated_hl_log_probs = []
            for idx, log_prob in enumerate(hl_log_probs):
                if self.config.training.debug_mode:
                    print(f"[DEBUG] List element {idx}: type={type(log_prob)}")
                    if hasattr(log_prob, 'shape'):
                        print(f"[DEBUG] List element {idx}: shape={log_prob.shape}, numel={log_prob.numel()}")
                
                if not isinstance(log_prob, torch.Tensor):
                    print(f"[ERROR] Non-tensor log_prob at index {idx}: {type(log_prob)}")
                    print(f"[ERROR] This suggests the high-level policy is not returning proper tensors")
                    raise ValueError(f"High-level policy returned non-tensor log_prob at index {idx}: {type(log_prob)}")
                
                if log_prob.numel() == 0:
                    # This is the root cause - investigate why the high-level policy is returning empty tensors
                    print(f"[ERROR] High-level policy returned empty tensor at index {idx}!")
                    print(f"[ERROR] This suggests the high-level policy is not properly generating log probabilities")
                    print(f"[ERROR] Context for this element: {hl_contexts[idx] if idx < len(hl_contexts) else 'N/A'}")
                    raise ValueError(f"High-level policy returned empty tensor at index {idx} - investigate the high-level policy implementation")
                
                # Ensure the tensor is 1D (not 0D scalar)
                if log_prob.dim() == 0:  # 0-dimensional scalar tensor
                    log_prob = log_prob.unsqueeze(0)  # Convert to 1D tensor
                
                validated_hl_log_probs.append(log_prob)
            hl_log_probs = validated_hl_log_probs
        else:
            raise ValueError(f"High-level policy returned unexpected log_probs type: {type(hl_log_probs)}")
        
        return hl_actions, hl_log_probs
    
    
    def _generate_ll_actions_20q(
        self,
        states: List[List[str]],
        hl_actions: List[str],
        policy,
        environment: HierarchicalTwentyQuestionsEnvironment,
        tokenizer,
        config
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate low-level actions (questions/guesses) based on high-level instructions.
        
        Args:
            states: Current conversation states
            hl_actions: High-level instructions
            policy: Low-level policy model
            environment: Hierarchical 20Q environment
            tokenizer: Tokenizer for text processing
            config: Training configuration
            
        Returns:
            Tuple of (ll_actions, ll_action_tokens, ll_log_probs)
        """
        # Mirror AgentTrainer: pass raw states and instructions directly to policy.act
        # - state: List[List[str]] conversation histories
        # - base_state: same states for context
        # - instructions: list of high-level instruction strings
        ll_actions, ll_action_tokens, ll_log_probs = policy.act(
            state=states,
            base_state=states,
            instructions=hl_actions,
        )
        return ll_actions, ll_action_tokens, ll_log_probs
    
    def build_chat_contexts_hierarchical_20q(
        self,
        environment: HierarchicalTwentyQuestionsEnvironment,
        states: List[List[str]],
        hl_instructions: List[str],
        config
    ) -> List[List[Dict[str, str]]]:
        """
        Build chat contexts for hierarchical 20Q training using conversation objects.
        
        Args:
            environment: Hierarchical 20Q environment
            states: Current states
            hl_instructions: High-level instructions
            config: Training configuration
            
        Returns:
            List of chat contexts for each conversation
        """
        from src.hierarchical_20q_prompts import format_low_level_oracle_context_optimized
        
        chat_contexts = []
        
        for i, hl_instruction in enumerate(hl_instructions):
            # Get the conversation object for this index
            conv = environment.conversations[i] if i < len(environment.conversations) else None
            if conv is None:
                # Fallback: create empty chat context
                chat_contexts.append([{'role': 'system', 'content': environment.get_system_prompt_ll()}])
                continue
            
            # Add the system prompt for low-level agent
            ll_system_prompt = environment.get_system_prompt_ll()
            chat_context = [{'role': 'system', 'content': ll_system_prompt}]
            
            # Create oracle-perspective context using optimized storage (excludes past HL instructions)
            oracle_context = format_low_level_oracle_context_optimized(
                oracle_qa_tuples=conv.oracle_qa_tuples,
                hl_instruction=hl_instruction,
                question_count=conv.question_count,
                max_questions=conv.max_questions
            )
            
            # Add the oracle-perspective context as a user message
            chat_context.append({'role': 'user', 'content': oracle_context})
            
            chat_contexts.append(chat_context)
        
        return chat_contexts
    
    def update_submit_metrics_hierarchical_20q(
        self,
        metrics: Dict[str, Any],
        hl_actions: List[str],
        ll_actions: List[str],
        responses: List[str],
        hl_rewards: List[float],
        ll_rewards: List[float]
    ) -> None:
        """
        Update metrics for hierarchical 20Q training.
        
        In hierarchical 20Q, the LL agent corresponds to the single agent for submit stats:
        - LL actions (questions/guesses) are counted for validity and accuracy
        - A response is considered correct if the oracle replies affirmatively ("yes")
        - Validity is counted for non-empty LL actions
        
        Args:
            metrics: Metrics dictionary to update
            hl_actions: High-level actions (instructions)
            ll_actions: Low-level actions (questions/guesses) - these count for submit stats
            responses: Oracle responses
            hl_rewards: High-level rewards
            ll_rewards: Low-level rewards
        """
        # Initialize counters if missing
        metrics.setdefault("submit_total_submissions", 0)
        metrics.setdefault("submit_valid_submissions", 0)
        metrics.setdefault("submit_correct_submissions", 0)
        metrics.setdefault("submit_predictions_labels", [])
        metrics.setdefault("submit_wins", 0)
        metrics.setdefault("submit_episode_efficiency", [])
        
        # Count validity and correctness for this step
        step_total = 0
        step_valid = 0
        step_correct = 0
        
        # For hierarchical 20Q, LL actions count as submissions (matching single-agent behavior)
        for ll_action, response in zip(ll_actions, responses):
            step_total += 1
            ll_action_str = str(ll_action or "").strip()
            valid = len(ll_action_str) > 0
            if valid:
                step_valid += 1
            
            # Check if response is "yes" (correct guess/question)
            is_yes = self._is_yes_response(response)
            if is_yes:
                step_correct += 1
            
            # Store pair for human readability
            metrics["submit_predictions_labels"].append((ll_action_str, ("YES" if is_yes else "NO")))
        
        # Update aggregate counts
        metrics["submit_total_submissions"] += step_total
        metrics["submit_valid_submissions"] += step_valid
        metrics["submit_correct_submissions"] += step_correct
        
        # Count wins (episodes with positive HL reward)
        wins = sum(1 for reward in hl_rewards if reward >= 0.0)
        metrics["submit_wins"] += wins
        
        # Track instruction-following quality (LL rewards)
        if ll_rewards:
            metrics.setdefault("submit_ll_rewards", []).extend(ll_rewards)
            avg_ll_reward = sum(ll_rewards) / len(ll_rewards)
            metrics["submit_avg_instruction_following"] = avg_ll_reward
        
        # Update aggregate percentage metrics
        if metrics["submit_total_submissions"] > 0:
            metrics["submit_validity_pct"] = (
                metrics["submit_valid_submissions"] / metrics["submit_total_submissions"] * 100.0
            )
        else:
            metrics["submit_validity_pct"] = 0.0
        
        # Accuracy: percentage of correct submissions (yes responses)
        total_count = max(1, metrics["submit_total_submissions"])  # avoid div by zero
        metrics["submit_accuracy_pct"] = (
            metrics["submit_correct_submissions"] / total_count * 100.0
        )
        
        # Track basic strategy counts
        guess_count = sum(1 for hl_action in hl_actions if detect_guess_instruction(hl_action))
        question_count = len(hl_actions) - guess_count
        
        metrics["submit_guess_instructions"] = metrics.get("submit_guess_instructions", 0) + guess_count
        metrics["submit_question_instructions"] = metrics.get("submit_question_instructions", 0) + question_count
    
    def _is_yes_response(self, text: str) -> bool:
        """
        Check if a response is a yes/affirmative response.
        
        This matches the logic from MCQAgentTrainer for processing twenty questions responses.
        """
        if not text:
            return False
        
        # Clean and normalize the response (matching MCQAgentTrainer logic)
        resp_str = str(text or "").strip().lower()
        
        # Look for YES/NO at the beginning of the response (matching MCQAgentTrainer)
        if resp_str.startswith(("yes", "no", "y", "n")):
            first_word = resp_str.split()[0]
            # Clean the first word to remove punctuation
            clean_word = first_word.strip('.,!?')
            return clean_word in {"yes", "y"}
        else:
            # If no clear YES/NO, check if it contains affirmative words
            return any(word in resp_str for word in ["yes", "y", "yep", "yeah", "correct", "true"])


# Monkey-patch the MCQAgentTrainer to add hierarchical 20Q support
def extend_mcq_trainer_for_hierarchical_20q():
    """
    Extend MCQAgentTrainer with hierarchical 20Q support.
    
    This function adds the hierarchical 20Q methods to the existing trainer.
    """
    # Add mixin methods to MCQAgentTrainer
    for attr_name in dir(HierarchicalTwentyQTrainerMixin):
        if not attr_name.startswith('_') or attr_name.startswith('_generate') or attr_name.startswith('_build') or attr_name.startswith('_is_') or attr_name.startswith('_clean_'):
            attr = getattr(HierarchicalTwentyQTrainerMixin, attr_name)
            if callable(attr):
                setattr(MCQAgentTrainer, attr_name, attr)
    
    # Override collect_rollout to handle hierarchical 20Q
    original_collect_rollout = MCQAgentTrainer.collect_rollout
    
    def enhanced_collect_rollout(
        self,
        policy,
        environment,
        tokenizer,
        rollout_steps,
        config,
        high_level_policy=None,
        templates_path=None,
        episode=0
    ):
        """Enhanced collect_rollout that handles hierarchical 20Q."""
        if is_hierarchical_twenty_questions_env(environment):
            return self.collect_rollout_hierarchical_20q(
                policy, environment, tokenizer, rollout_steps, config,
                high_level_policy, templates_path, episode
            )
        else:
            return original_collect_rollout(
                self, policy, environment, tokenizer, rollout_steps, config,
                high_level_policy, templates_path, episode
            )
    
    MCQAgentTrainer.collect_rollout = enhanced_collect_rollout

# Initialize the extension
extend_mcq_trainer_for_hierarchical_20q()
