"""
Rollout data structures and collection logic for AgentTrainer.
"""

import copy
import json
import random
import time
from typing import Any, Dict, List, Optional, Union, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import build_chat
from src.conversation.utils import conversation_from_state, conversations_from_batch


class RolloutData(NamedTuple):
    """Class for storing rollout data using rejection sampling."""
    batch_state: torch.Tensor
    batch_next_state: torch.Tensor
    batch_acts: torch.Tensor
    batch_log_probs: torch.Tensor
    batch_rews: torch.Tensor
    batch_dones: torch.Tensor
    batch_base_state: torch.Tensor
    conversations: List[Any]  # List of Conversation objects
    # High-level policy data (optional)
    high_level_actions: Optional[torch.Tensor] = None
    high_level_log_probs: Optional[torch.Tensor] = None
    # Store the actual natural-language instructions if available so we can
    # compute instruction-following rewards during training.
    high_level_instructions: Optional[List[str]] = None
    # Template IDs for tracking persona continuity
    collected_template_ids: Optional[List[str]] = None
    # Persistent submit metrics for MCQ environments
    persistent_submit_metrics: Optional[Dict[str, Any]] = None
    # Freeform high-level metadata
    high_level_state_texts: Optional[List[str]] = None
    high_level_instruction_log_probs: Optional[List[float]] = None
    high_level_base_questions: Optional[List[str]] = None
    # Guard reward statistics (for validation and debugging)
    guard_scores: Optional[torch.Tensor] = None  # Guard scores for agent actions [batch_size]
    target_guard_scores: Optional[torch.Tensor] = None  # Guard scores for target responses [batch_size]
    # Raw output token IDs from freeform high-level policy (includes [RESPONSE] tag if enabled)
    high_level_raw_output_ids: Optional[List[torch.Tensor]] = None
    # 3-layer tool red-team metadata
    tool_index_actions: Optional[torch.Tensor] = None
    l1_rewards: Optional[torch.Tensor] = None
    l3_rewards: Optional[torch.Tensor] = None


def _state_item_to_text(state_item: Union[str, List[str]]) -> str:
    if isinstance(state_item, list):
        return " ".join(str(s) for s in state_item if s)
    return str(state_item)


def _clear_l1_option(episode) -> None:
    episode.l1_option_active = False
    episode.l1_option_tool = ""
    episode.l1_option_tool_index = -1
    episode.l1_option_steps_remaining = 0
    episode.l1_option_start_message_len = 0
    episode.l1_option_state_text = ""


def _batch_tokenize_and_pad_state(tokenizer, texts: List[Union[str, List[str]]]) -> torch.Tensor:
    """
    Tokenize a batch of texts and pad them to the same length.
    
    Args:
        tokenizer: Tokenizer to use
        texts: List of text strings or conversation histories to tokenize
        max_length: Maximum length for padding (if None, uses longest sequence)
        
    Returns:
        Padded tensor of input_ids
    """
    
    # Convert all texts to strings and handle different input formats
    processed_texts = []
    for text in texts:
        # Handle both single strings and conversation histories
        if isinstance(text, str):
            # Single string - treat as a single message
            processed_texts.append(build_chat([text], 0))
        elif isinstance(text, list):
            # Conversation history - use as is
            processed_texts.append(build_chat(text, 0))
        else:
            raise ValueError(f"Expected string or list of strings, got {type(text)}")
    
    encoded = tokenizer.apply_chat_template(
        processed_texts,
        return_tensors='pt',
        padding=True,
        padding_side='left',
        return_dict=True,
    )
    input_ids = encoded.input_ids
    
    # Validate the tokenization result
    assert input_ids.dim() == 2, (
        f"Expected 2D tensor from tokenization, got {input_ids.dim()}D tensor with shape {input_ids.shape}"
    )
    assert input_ids.size(0) == len(processed_texts), (
        f"Batch size mismatch: expected {len(processed_texts)}, got {input_ids.size(0)}"
    )
    assert input_ids.size(0) > 0, "Tokenization resulted in empty batch"
    
    return input_ids

def _batch_tokenize_and_pad_action(tokenizer, actions: List[str]) -> torch.Tensor:
    """
    Tokenize a batch of actions and pad them to the same length.
    """
    # Tokenize with left padding (without modifying global tokenizer settings)
    return tokenizer(actions, return_tensors='pt', padding=True, padding_side='left').input_ids

def _decode_with_timeout(tokenizer, input_ids: torch.Tensor, **kwargs) -> str:
    """
    Decode tokens (timeout handling removed for fail-fast behavior).
    
    Args:
        tokenizer: Tokenizer to use
        input_ids: Token IDs to decode
        **kwargs: Additional arguments for tokenizer.decode
        
    Returns:
        Decoded string
    """
    return tokenizer.decode(input_ids, **kwargs)


def _batch_decode_with_timeout(tokenizer, input_ids_list: List[torch.Tensor], **kwargs) -> List[str]:
    """
    Batch decode tokens (timeout handling removed for fail-fast behavior).
    
    Args:
        tokenizer: Tokenizer to use
        input_ids_list: List of token ID tensors to decode
        **kwargs: Additional arguments for tokenizer.decode
        
    Returns:
        List of decoded strings
    """
    return [tokenizer.decode(ids, **kwargs) for ids in input_ids_list]


def collect_rollout(
    agent_trainer: Any,  # <-- Add this argument
    policy: Any,
    environment: Any,
    tokenizer: Any,
    rollout_steps: int,
    config: Any,
    high_level_policy: Any = None,
    templates_path: str = "src/configs/templates.json",
    episode: int = 0  # Add episode parameter for epsilon annealing
) -> RolloutData:
    """
    Collect a rollout.
    
    Rejection Sampling:
    1. For each state, generate N candidate responses
    2. Score all candidates using either combined v_high + perplexity (both batch normalized) or just v_high
    3. Select the highest-scoring response as the rollout step
    4. Use larger batch sizes since no gradients needed during rollout
    
    Args:
        policy: Low-level policy (token generation)
        environment: Training environment
        tokenizer: Tokenizer for processing text
        rollout_steps: Number of steps to collect
        config: Agent configuration
        high_level_policy: High-level policy for instruction generation (optional)
        templates_path: Path to templates JSON file
        
    Returns:
        Rollout data as a RolloutData namedtuple
    """
    start_time = time.time()
    template_policy_kinds = {"dqn", "ppo", "hybrid"}
    # Check if this is a flat agent (no high-level policy)
    is_flat_agent = high_level_policy is None or not getattr(config.model, 'use_high_level_policy', True)
    
    if not is_flat_agent:
        # Hierarchical agent: high-level policy is required
        default_policy_kind = getattr(high_level_policy, "policy_kind", None)
        uses_template_policy = default_policy_kind in template_policy_kinds
        if uses_template_policy:
            from src.template_registry import TemplateRegistry  # Local import to avoid template dependency for freeform
            TemplateRegistry(templates_path)
    else:
        # Flat agent: no high-level policy needed
        default_policy_kind = None
        uses_template_policy = False
    
    # Initialize policy_kind for use throughout the function
    policy_kind = default_policy_kind
    
    # Reset environment
    state = environment.reset()
    batch_size = len(state)
    
    # Debug mode: minimize generation parameters
    debug_mode = config.training.debug_mode
    if debug_mode:
        print("DEBUG MODE")
    
    # Determine N (number of candidates per state)
    rejection_candidates = getattr(config.model, 'rejection_candidates', 1)
    if debug_mode:
        N = 1  # Only 1 candidate in debug mode
    elif uses_template_policy and getattr(high_level_policy, "num_actions", 0):
        N = max(1, int(getattr(high_level_policy, "num_actions", rejection_candidates)))
    else:
        N = max(1, int(rejection_candidates))  # Number of diverse generations
        
    # Lists to store rollout data
    states = []
    next_states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    base_states = []
    conversations = []
    # High-level policy data
    high_level_actions = []
    high_level_log_probs = []
    # Natural-language instructions (one per environment instance/step)
    collected_instructions: List[str] = []
    # Template IDs for tracking persona continuity (only for template-based policies)
    collected_template_ids: Optional[List[str]] = [] if uses_template_policy else None
    # Freeform metadata per timestep
    freeform_state_history: List[str] = []
    freeform_logprob_history: List[float] = []
    freeform_base_history: List[str] = []
    freeform_raw_output_ids_history: List[List[torch.Tensor]] = []
    # Guard score tracking (for validation and debugging)
    guard_scores_list: List[float] = []
    target_guard_scores_list: List[float] = []
    tool_index_policy = getattr(agent_trainer, "tool_index_policy", None)
    collected_tool_indices: List[int] = []
    collected_l1_rewards: List[float] = []
    collected_l3_rewards: List[float] = []
    predicted_tools_batch: Optional[List[str]] = None
    l1_option_max_steps = getattr(config.model, "l1_option_max_steps", 5)
    # Initialize conversation objects for each environment state
    base_conversations = conversations_from_batch(state)
    
    # Collect steps using rejection sampling
    steps_collected = 0
    already_dones = [False] * batch_size
    # Respect generation length configuration from the environment
    max_tokens = environment.max_new_tokens
    while steps_collected < rollout_steps:
        # Use larger batches since no gradients needed during rollout
        with torch.no_grad():
            predicted_tools_batch = None
            tool_indices_batch = None
            # Make deep copies to avoid stale references after environment.step() updates environment.state
            if isinstance(environment.base_state, list):
                batch_base_state = copy.deepcopy(environment.base_state)
            else:
                batch_base_state = environment.base_state
            
            if isinstance(environment.state, list):
                # Deep copy list of states (each state might be a list of conversation messages)
                batch_state = [copy.deepcopy(s) if isinstance(s, list) else copy.copy(s) if hasattr(s, 'copy') else s for s in environment.state]
            elif hasattr(environment.state, 'copy'):
                batch_state = environment.state.copy()
            else:
                batch_state = environment.state
            # --------------------------------------------------------------
            # High-level strategy selection (HRL architecture)
            # --------------------------------------------------------------
            if not is_flat_agent and high_level_policy is not None:
                policy_kind = getattr(high_level_policy, "policy_kind", None)
                step_state_texts: Optional[List[str]] = None
                step_base_questions: Optional[List[str]] = None
                if policy_kind == "dqn":
                    # For each unfinished environment instance decide on a strategy
                    chosen_strategy_indices = [None] * batch_size
                    chosen_state_texts = [None] * batch_size
                    high_level_instructions = [None] * batch_size
                    current_high_level_log_probs = [None] * batch_size  # Initialize for DQN case

                    for idx in range(batch_size):
                        if already_dones[idx]:
                            continue  # No action for finished episodes

                        # Convert current state to plain text
                        state_item = batch_state[idx]
                        if isinstance(state_item, list):
                            state_text = " ".join(str(s) for s in state_item if s)
                        else:
                            state_text = str(state_item)

                        enc = tokenizer(state_text, return_tensors="pt", truncation=True)
                        # High-level epsilon exploration schedule (anneals like low-level)
                        hl_epsilon_start = getattr(config.training, 'hl_epsilon_start', 0.8)
                        hl_epsilon_end = getattr(config.training, 'hl_epsilon_end', 0.0)
                        hl_epsilon_decay_episodes = getattr(config.training, 'hl_epsilon_decay_episodes', 100)
                        hl_epsilon = hl_epsilon_end + (hl_epsilon_start - hl_epsilon_end) * max(0, (hl_epsilon_decay_episodes - episode) / hl_epsilon_decay_episodes)
                        if random.random() < hl_epsilon:
                            strategy_idx = random.randint(0, len(high_level_policy.strategies) - 1)
                        else:
                            strategy_idx = high_level_policy.select_action(enc.input_ids, enc.attention_mask)
                        strategy = high_level_policy.strategies[strategy_idx]
                        instruction = strategy["instruction"]
                        
                        # Extract template ID for tracking
                        template_id = strategy.get("id", "")
                        
                        # Add context if needed and get template ID
                        topic = extract_topic_from_state(batch_base_state[idx])
                        question = batch_base_state[idx]
                        
                        # Get conversation object for this batch index
                        conversation = environment.conversations[idx] if hasattr(environment, 'conversations') and idx < len(environment.conversations) else None
                        
                        # Store template ID in conversation object for future reference
                        if conversation is not None and template_id:
                            conversation.add_template_id(template_id)
                        
                        # Format instruction with user_type replacement
                        formatted_instruction, _ = format_high_level_instruction_with_user_type(
                            instruction, topic, question, conversation
                        )

                        chosen_strategy_indices[idx] = strategy_idx
                        chosen_state_texts[idx] = state_text
                        high_level_instructions[idx] = formatted_instruction
                elif policy_kind == "ppo":
                    # PPO-style high-level policy: get log probabilities for training
                    chosen_strategy_indices = [None] * batch_size
                    chosen_state_texts = [None] * batch_size
                    high_level_instructions = [None] * batch_size
                    current_high_level_log_probs = [None] * batch_size

                    for idx in range(batch_size):
                        if already_dones[idx]:
                            continue  # No action for finished episodes

                        # Convert current state to plain text
                        state_item = batch_state[idx]
                        if isinstance(state_item, list):
                            state_text = " ".join(str(s) for s in state_item if s)
                        else:
                            state_text = str(state_item)

                        enc = tokenizer(state_text, return_tensors="pt", truncation=True)
                        state_tokens = enc.input_ids
                        strategy_logits = high_level_policy.forward(state_tokens)
                        strategy_logits_clamped = torch.clamp(strategy_logits, min=-5, max=5)
                        probs = F.softmax(strategy_logits_clamped, dim=-1)
                        strategy_log_probs = F.log_softmax(strategy_logits_clamped, dim=-1)
                        # High-level epsilon exploration schedule (anneals like low-level)
                        hl_epsilon_start = getattr(config.training, 'hl_epsilon_start', 0.8)
                        hl_epsilon_end = getattr(config.training, 'hl_epsilon_end', 0.0)
                        hl_epsilon_decay_episodes = getattr(config.training, 'hl_epsilon_decay_episodes', 100)
                        hl_epsilon = hl_epsilon_end + (hl_epsilon_start - hl_epsilon_end) * max(0, (hl_epsilon_decay_episodes - episode) / hl_epsilon_decay_episodes)
                        if random.random() < hl_epsilon:
                            strategy_idx = random.randint(0, len(high_level_policy.strategies) - 1)
                            selected_log_prob = float('nan')
                        else:
                            strategy_idx = torch.multinomial(probs, 1).squeeze(-1)
                            selected_log_prob = strategy_log_probs.gather(1, strategy_idx.unsqueeze(-1)).squeeze(-1)
                        if isinstance(strategy_idx, torch.Tensor):
                            strategy_idx = strategy_idx.item()
                        strategy = high_level_policy.strategies[strategy_idx]
                        instruction = strategy["instruction"]
                        
                        # Extract template ID for tracking
                        template_id = strategy.get("id", "")
                        
                        # Add context if needed and get template ID
                        topic = extract_topic_from_state(batch_base_state[idx])
                        question = batch_base_state[idx]
                        
                        # Get conversation object for this batch index
                        conversation = environment.conversations[idx] if hasattr(environment, 'conversations') and idx < len(environment.conversations) else None
                        
                        # Store template ID in conversation object for future reference
                        if conversation is not None and template_id:
                            conversation.add_template_id(template_id)
                        
                        # Format instruction with user_type replacement
                        formatted_instruction, _ = format_high_level_instruction_with_user_type(
                            instruction, topic, question, conversation
                        )

                        chosen_strategy_indices[idx] = strategy_idx
                        chosen_state_texts[idx] = state_text
                        high_level_instructions[idx] = formatted_instruction
                        current_high_level_log_probs[idx] = selected_log_prob
                elif policy_kind in ("llm", "freeform"):
                    # LLM-based high-level policy: generate instructions freehand
                    # Prepare plain text states for the whole batch
                    # PERFORMANCE OPTIMIZATION: Batch process state texts and base questions
                    state_texts_batch = []
                    base_questions = []
                    for idx in range(batch_size):
                        # Process state item
                        state_item = batch_state[idx]
                        if isinstance(state_item, list):
                            state_texts_batch.append(" ".join(str(s) for s in state_item if s))
                        else:
                            state_texts_batch.append(str(state_item))
                        
                        # Process base state item
                        base_item = batch_base_state[idx]
                        if isinstance(base_item, str):
                            base_questions.append(base_item)
                        else:
                            base_questions.append(str(base_item))

                    target_tools_batch = None
                    tools_called_batch = None
                    if tool_index_policy is not None and hasattr(environment, "episodes"):
                        predicted_tools_batch = [""] * batch_size
                        tools_called_batch = []
                        for idx in range(batch_size):
                            if already_dones[idx]:
                                predicted_tools_batch[idx] = ""
                                tools_called_batch.append([])
                                continue
                            episode_state = environment.episodes[idx]
                            state_text = _state_item_to_text(batch_state[idx])
                            if not episode_state.l1_option_active:
                                tool_indices, tool_names, _ = tool_index_policy.act([state_text])
                                episode_state.l1_option_active = True
                                episode_state.l1_option_tool = tool_names[0]
                                episode_state.l1_option_tool_index = tool_indices[0]
                                episode_state.l1_option_steps_remaining = l1_option_max_steps
                                episode_state.l1_option_start_message_len = len(episode_state.messages)
                                episode_state.l1_option_state_text = state_text
                            predicted_tools_batch[idx] = episode_state.l1_option_tool
                            tools_called_batch.append(list(episode_state.tools_called_so_far))
                        target_tools_batch = predicted_tools_batch

                    act_kwargs = {"base_questions": base_questions}
                    if target_tools_batch is not None:
                        act_kwargs["target_tools"] = target_tools_batch
                        act_kwargs["tools_called_list"] = tools_called_batch
                    act_result = high_level_policy.act(state_texts_batch, **act_kwargs)
                    if len(act_result) == 3:
                        instructions_batch, log_probs_batch, raw_output_ids_batch = act_result
                    else:
                        # Fallback for policies that don't return raw_output_ids (shouldn't happen for freeform)
                        instructions_batch, log_probs_batch = act_result
                        raw_output_ids_batch = None

                    high_level_instructions = instructions_batch
                    chosen_strategy_indices = [None] * batch_size
                    current_high_level_log_probs = [lp.item() if isinstance(lp, torch.Tensor) else lp for lp in log_probs_batch]
                    chosen_state_texts = state_texts_batch
                    step_state_texts = state_texts_batch
                    step_base_questions = base_questions
                    # Store raw_output_ids for this step
                    if raw_output_ids_batch is not None:
                        freeform_raw_output_ids_history.append(raw_output_ids_batch)
                elif policy_kind == "hybrid":
                    # Hybrid MLP + LLM policy: MLP selects strategy, LLM customizes instruction
                    # PERFORMANCE OPTIMIZATION: Batch process state texts and base questions
                    state_texts_batch = []
                    base_questions = []
                    for idx in range(batch_size):
                        # Process state item
                        state_item = batch_state[idx]
                        if isinstance(state_item, list):
                            state_texts_batch.append(" ".join(str(s) for s in state_item if s))
                        else:
                            state_texts_batch.append(str(state_item))
                        
                        # Process base state item
                        base_item = batch_base_state[idx]
                        if isinstance(base_item, str):
                            base_questions.append(base_item)
                        else:
                            base_questions.append(str(base_item))
                    instructions_batch, log_probs_batch, strategy_indices_batch = high_level_policy.act(state_texts_batch, base_questions=base_questions)

                    high_level_instructions = instructions_batch
                    # Extract strategy indices from the MLP component for training
                    chosen_strategy_indices = strategy_indices_batch
                    current_high_level_log_probs = [lp.item() if isinstance(lp, torch.Tensor) else lp for lp in log_probs_batch]
                    chosen_state_texts = state_texts_batch
                    step_state_texts = state_texts_batch
                    step_base_questions = base_questions
                
                else:
                    raise ValueError(f"Unsupported high_level_policy.policy_kind: {policy_kind}")
            else:
                # Flat agent: no high-level policy, generate actions directly
                high_level_instructions = [None] * batch_size
                chosen_strategy_indices = [None] * batch_size
                current_high_level_log_probs = [None] * batch_size
                chosen_state_texts = [None] * batch_size
                step_state_texts = None
                step_base_questions = None
                # policy_kind is already set to None for flat agents

            # Debug: print integer list of HL actions for this step (None -> -1)
            if getattr(config.training, 'debug_mode', False):   
                try:
                    hl_actions = [(-1 if (chosen_strategy_indices is None or chosen_strategy_indices[i] is None) else int(chosen_strategy_indices[i])) for i in range(batch_size)]
                    print("hl_actions", hl_actions)
                except Exception:
                    # Fail loud if shapes are inconsistent in debug mode only
                    if getattr(config.training, 'debug_mode', False):
                        raise

            # Validate high_level_instructions (only for hierarchical agents)
            if not is_flat_agent:
                if high_level_instructions is None:
                    raise ValueError("High-level instructions were not produced. The policy must return a string per batch item.")
                if not isinstance(high_level_instructions, list):
                    raise TypeError(f"high_level_instructions must be a list of strings, got {type(high_level_instructions)}")
                if len(high_level_instructions) != batch_size:
                    raise ValueError(f"high_level_instructions length {len(high_level_instructions)} does not match batch_size {batch_size}")
                for idx in [i for i, d in enumerate(already_dones) if not d]:
                    instr = high_level_instructions[idx]
                    if not isinstance(instr, str) or instr.strip() == "":
                        raise ValueError(f"Empty or invalid high-level instruction at index {idx}. The policy must provide a non-empty instruction.")

            # Generate actions for unfinished environments using rejection sampling
            selected_actions = []
            selected_action_tokens = []
            selected_log_probs = []
            selected_candidate_indices = []

            unfinished_indices = [i for i, done in enumerate(already_dones) if not done]
            
            # Calculate epsilon for low-level rejection sampling (anneal from 0.1 to 0.0 over 100 episodes)
            low_level_epsilon_start = 0.1
            low_level_epsilon_end = 0.0
            low_level_epsilon_decay_episodes = 100
            low_level_epsilon = low_level_epsilon_end + (low_level_epsilon_start - low_level_epsilon_end) * max(0, (low_level_epsilon_decay_episodes - episode) / low_level_epsilon_decay_episodes)
            
            # Step 1: Prepare all candidates for all unfinished states at once
            # CRITICAL: Re-sync batch_state from conversations to ensure alignment
            # The snapshot might be stale - conversations are the source of truth
            if hasattr(environment, 'conversations') and hasattr(environment, '_conversations_to_state'):
                current_env_state = environment._conversations_to_state()
                if len(current_env_state) == len(batch_state):
                    # Update batch_state with current state from conversations for unfinished indices
                    for idx in unfinished_indices:
                        if idx < len(current_env_state):
                            batch_state[idx] = current_env_state[idx]
                else:
                    # Length mismatch - this shouldn't happen but fail loudly if it does
                    raise ValueError(
                        f"State length mismatch: environment state length={len(current_env_state)}, "
                        f"batch_state length={len(batch_state)}, conversations length={len(environment.conversations)}"
                    )
            
            all_candidate_states = []
            all_candidate_base_states = []
            all_candidate_instructions = []
            state_to_candidate_mapping = []  # Maps each candidate back to its original state index
            
            # Group candidates by candidate number (for temperature ordering)
            for candidate_num in range(N):
                for idx in unfinished_indices:
                    # Get state and base state for this environment
                    # CRITICAL: Ensure idx is within bounds and corresponds to the correct conversation
                    if idx >= len(batch_state) or idx >= len(batch_base_state):
                        raise IndexError(f"Index {idx} out of bounds: batch_state length={len(batch_state)}, batch_base_state length={len(batch_base_state)}")
                    
                    state_item = batch_state[idx]
                    base_state_item = batch_base_state[idx]
                    instruction_item = high_level_instructions[idx] if high_level_instructions is not None and idx < len(high_level_instructions) else None
                    
                    # VALIDATION: Check that state_item matches the conversation it's supposed to belong to
                    if hasattr(environment, 'conversations') and idx < len(environment.conversations):
                        conv = environment.conversations[idx]
                        conv_id = conv.conversation_id
                        # Get the expected state from the conversation
                        expected_state = environment._conversations_to_state()[idx] if hasattr(environment, '_conversations_to_state') else None
                        if expected_state is not None:
                            # Compare first message to catch misalignments
                            state_first_msg = state_item[0] if isinstance(state_item, list) and len(state_item) > 0 else str(state_item)
                            expected_first_msg = expected_state[0] if isinstance(expected_state, list) and len(expected_state) > 0 else str(expected_state)
                            if state_first_msg != expected_first_msg:
                                error_msg = f"\n[ERROR] State mismatch in rollout: batch_state does not match conversation state\n"
                                error_msg += f"  Batch index: {idx}\n"
                                error_msg += f"  Conversation ID: {conv_id}\n"
                                error_msg += f"  batch_state[{idx}] first message: {state_first_msg[:150]}...\n"
                                error_msg += f"  conversation state first message: {expected_first_msg[:150]}...\n"
                                error_msg += f"  This indicates batch_state is misaligned with conversations\n"
                                raise ValueError(error_msg)
                    
                    # Add this state to the candidate group
                    all_candidate_states.append(state_item)
                    all_candidate_base_states.append(base_state_item)
                    all_candidate_instructions.append(instruction_item)
                    state_to_candidate_mapping.append(idx)
            
            # Step 2: Generate all candidate utterances with varying temperature and top_p for diversity
            all_candidate_actions = []
            all_candidate_action_tokens = []
            all_candidate_log_probs = []
            
            # Generate candidates with increasing temperature and top_p for diversity
            # Temperature ranges from 0.5 to 1.5 across candidates
            # Top_p ranges from 0.8 to 1.15 across candidates
            base_temperature = 0.5
            temp_increment = 1.0 / N  # Spread temperature across N candidates
            base_top_p = 0.8
            top_p_increment = 0.35 / N  # Spread top_p across N candidates
            
            # Generate candidates with proper state-candidate correspondence
            num_unfinished = len(unfinished_indices)
            for candidate_idx in range(N):
                # Calculate temperature and top_p for this candidate (increases with index)
                candidate_temperature = base_temperature + (candidate_idx * temp_increment)
                candidate_top_p = base_top_p + (candidate_idx * top_p_increment)
                
                # Generate candidates for all states with this temperature and top_p
                # all_candidate_states is structured as: [state_0_cand_0, state_1_cand_0, ..., state_k_cand_0, state_0_cand_1, state_1_cand_1, ...]
                # So for candidate_idx, we need to slice from candidate_idx * num_unfinished to (candidate_idx + 1) * num_unfinished
                start_idx = candidate_idx * num_unfinished
                end_idx = (candidate_idx + 1) * num_unfinished
                candidate_states = all_candidate_states[start_idx:end_idx]
                candidate_base_states = all_candidate_base_states[start_idx:end_idx]
                candidate_instructions = all_candidate_instructions[start_idx:end_idx]
                
                # Ensure we don't exceed the expected number of candidates
                expected_per_candidate = len(unfinished_indices)
                if len(candidate_states) > expected_per_candidate:
                    candidate_states = candidate_states[:expected_per_candidate]
                    candidate_base_states = candidate_base_states[:expected_per_candidate]
                    candidate_instructions = candidate_instructions[:expected_per_candidate]
                
                # VALIDATION: Check that candidate_states match the conversations they're supposed to belong to
                # This ensures the state used for generation matches what's displayed
                for i, (cand_state, unfinished_idx) in enumerate(zip(candidate_states, unfinished_indices)):
                    if hasattr(environment, 'conversations') and unfinished_idx < len(environment.conversations):
                        conv = environment.conversations[unfinished_idx]
                        conv_id = conv.conversation_id
                        # Get expected state from conversation
                        expected_state = environment._conversations_to_state()[unfinished_idx] if hasattr(environment, '_conversations_to_state') else None
                        if expected_state is not None:
                            # Compare first message
                            cand_first_msg = cand_state[0] if isinstance(cand_state, list) and len(cand_state) > 0 else str(cand_state)
                            expected_first_msg = expected_state[0] if isinstance(expected_state, list) and len(expected_state) > 0 else str(expected_state)
                            if cand_first_msg != expected_first_msg:
                                error_msg = f"\n[ERROR] State mismatch in policy.act(): Candidate state does not match conversation\n"
                                error_msg += f"  Candidate index: {i} (within candidate group {candidate_idx})\n"
                                error_msg += f"  Unfinished batch index: {unfinished_idx}\n"
                                error_msg += f"  Conversation ID: {conv_id}\n"
                                error_msg += f"  Candidate state first message: {cand_first_msg[:150]}...\n"
                                error_msg += f"  Expected state first message: {expected_first_msg[:150]}...\n"
                                error_msg += f"  This indicates candidate_states array is misaligned with conversations\n"
                                raise ValueError(error_msg)
                
                # Generate candidates for states with this temperature and top_p
                candidate_actions, candidate_action_tokens, candidate_log_probs = policy.act(
                    candidate_states,
                    candidate_base_states, 
                    instructions=candidate_instructions,
                    max_new_tokens=max_tokens,
                    temperature=candidate_temperature,
                    top_p=candidate_top_p
                )
                
                all_candidate_actions.extend(candidate_actions)
                all_candidate_action_tokens.extend(candidate_action_tokens)
                all_candidate_log_probs.extend(candidate_log_probs)
            
            # Step 3: Score all candidates using v_high in batches
            all_candidate_scores = []
            
            # Process candidates in chunks to avoid memory issues
            chunk_size = config.training.actor_max_chunk_size
            for chunk_start in range(0, len(all_candidate_states), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(all_candidate_states))
                
                # Tokenize chunk of states and actions for v_high scoring
                chunk_states = all_candidate_states[chunk_start:chunk_end]
                chunk_actions = all_candidate_actions[chunk_start:chunk_end]
                chunk_base_states = all_candidate_base_states[chunk_start:chunk_end]
                
                state_tokens = _batch_tokenize_and_pad_state(tokenizer, chunk_states)
                action_tokens_padded = _batch_tokenize_and_pad_action(tokenizer, chunk_actions)
                base_state_tokens = _batch_tokenize_and_pad_state(tokenizer, chunk_base_states)
                
                # Score chunk using either combined v_high + perplexity, just v_high, or just perplexity
                use_combined = getattr(config.model, 'use_combined_rejection_scoring', True)
                use_perplexity_only = getattr(config.model, 'use_perplexity_only', False)

                if not getattr(environment, 'pre_step_rejection_scoring', True):
                    # Environments like AgentHarm need a defender turn before harm scoring.
                    chunk_scores = torch.zeros(len(chunk_actions))
                elif use_perplexity_only:
                    # Use only perplexity for rejection sampling
                    action_texts = agent_trainer._batch_decode_with_timeout(tokenizer, action_tokens_padded, skip_special_tokens=True)
                    
                    # Use environment's compute_perplexity method which uses the target model
                    perplexity_scores = environment.compute_perplexity(action_texts)
                    
                    # Ensure perplexity scores are on the same device as other tensors
                    device = state_tokens.device
                    perplexity_scores = perplexity_scores.to(device=device, dtype=state_tokens.dtype)
                    
                    # For perplexity, lower is better, so we negate to make higher better
                    perplexity_mean = perplexity_scores.mean()
                    perplexity_std = perplexity_scores.std()
                    if perplexity_std > 0:
                        chunk_scores = -(perplexity_scores - perplexity_mean) / perplexity_std
                    else:
                        chunk_scores = -(perplexity_scores - perplexity_mean)
                    
                    # Clean up perplexity-specific tensors
                    del action_texts, perplexity_scores, perplexity_mean, perplexity_std
                        
                elif use_combined:
                    perplexity_start_weight = getattr(config.model, 'perplexity_start_weight', 1.0)
                    perplexity_end_weight = getattr(config.model, 'perplexity_end_weight', 0.4)
                    perplexity_decay_episodes = getattr(config.model, 'perplexity_decay_episodes', 100)
                    chunk_scores = agent_trainer.compute_combined_rejection_score(
                        state_tokens, action_tokens_padded, base_state_tokens, environment, tokenizer,
                        current_episode=episode,
                        perplexity_start_weight=perplexity_start_weight,
                        perplexity_end_weight=perplexity_end_weight,
                        perplexity_decay_episodes=perplexity_decay_episodes
                    )
                else:
                    # Fallback to original v_high scoring
                    chunk_scores = agent_trainer.v_high(state_tokens, action_tokens_padded, base_state_tokens, environment)
                
                # Convert to CPU and extend scores list immediately
                all_candidate_scores.extend(chunk_scores.cpu().tolist())
                
                # AGGRESSIVE CLEANUP: Delete all intermediate tensors immediately
                del state_tokens, action_tokens_padded, base_state_tokens, chunk_scores
                del chunk_states, chunk_actions, chunk_base_states
                
                # OPTIMIZATION: Only cleanup GPU cache, skip expensive gc.collect() between chunks
                # Full GC only happens after all chunks are processed
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Step 4: Select best candidate for each state using epsilon-greedy
            # OPTIMIZATION: Pre-group candidates by state for vectorized selection
            state_to_candidate_groups = {}
            for candidate_idx, state_idx in enumerate(state_to_candidate_mapping):
                if state_idx not in state_to_candidate_groups:
                    state_to_candidate_groups[state_idx] = []
                state_to_candidate_groups[state_idx].append(candidate_idx)
            
            # Vectorized selection: process all states at once
            selected_actions = []
            selected_action_tokens = []
            selected_log_probs = []
            selected_candidate_indices = []
            # Store which state was used to generate each selected action for validation
            selected_state_indices = []
            selected_state_snapshots = []  # Store state snapshots for validation
            
            # Pre-generate random values for epsilon-greedy (batch)
            use_random = np.random.random(len(unfinished_indices)) < low_level_epsilon
            
            for state_pos, state_idx in enumerate(unfinished_indices):
                state_candidate_indices = state_to_candidate_groups[state_idx]
                state_candidate_scores = [all_candidate_scores[i] for i in state_candidate_indices]
                
                # Select highest-scoring candidate with 1-epsilon probability (else random)
                if use_random[state_pos]:
                    # Random selection
                    selected_candidate_idx = random.choice(state_candidate_indices)
                    selection_method = "random"
                else:
                    # Select highest-scoring candidate
                    best_local_idx = state_candidate_scores.index(max(state_candidate_scores))
                    selected_candidate_idx = state_candidate_indices[best_local_idx]
                    selection_method = "highest-scoring"
                
                # VALIDATION: Verify the candidate was generated from the correct state
                candidate_state_idx = state_to_candidate_mapping[selected_candidate_idx]
                if candidate_state_idx != state_idx:
                    raise ValueError(
                        f"CRITICAL MISMATCH: Selected candidate {selected_candidate_idx} was generated from state {candidate_state_idx}, "
                        f"but is being assigned to state {state_idx}. This indicates a state-action alignment bug!"
                    )
                
                # Store the selected candidate
                selected_actions.append(all_candidate_actions[selected_candidate_idx])
                selected_action_tokens.append(all_candidate_action_tokens[selected_candidate_idx])
                selected_log_probs.append(all_candidate_log_probs[selected_candidate_idx])
                selected_candidate_indices.append(selected_candidate_idx % N)  # Local index within state
                selected_state_indices.append(state_idx)
                # Store snapshot of the state that was used to generate this action
                selected_state_snapshots.append(copy.deepcopy(all_candidate_states[selected_candidate_idx]))
                
                # Debug output for first state in first step
                if steps_collected == 0 and state_idx == unfinished_indices[0]:
                    print(f"\nREJECTION SAMPLING DEBUG (Episode {episode}):")
                    print(f"  Low-level epsilon: {low_level_epsilon:.3f}")
                    print(f"  Number of candidates per state: {N}")
                    print(f"  Total candidates generated: {len(all_candidate_states)}")
                    use_combined = getattr(config.model, 'use_combined_rejection_scoring', True)
                    use_perplexity_only = getattr(config.model, 'use_perplexity_only', False)
                    
                    if use_perplexity_only:
                        print(f"  Scoring method: Perplexity only (using target model)")
                    elif use_combined:
                        perplexity_start_weight = getattr(config.model, 'perplexity_start_weight', 1.0)
                        perplexity_end_weight = getattr(config.model, 'perplexity_end_weight', 0.4)
                        perplexity_decay_episodes = getattr(config.model, 'perplexity_decay_episodes', 100)
                        
                        # Compute current annealed weights for display
                        progress = min(episode / perplexity_decay_episodes, 1.0)
                        current_perplexity_weight = perplexity_start_weight + (perplexity_end_weight - perplexity_start_weight) * progress
                        current_v_high_weight = 1.0 - current_perplexity_weight
                        
                        print(f"  Scoring method: Combined v_high + perplexity (batch normalized, annealed)")
                        print(f"  Annealing: perplexity {perplexity_start_weight:.1f} → {perplexity_end_weight:.1f} over {perplexity_decay_episodes} episodes")
                        print(f"  Current weights (episode {episode}): v_high={current_v_high_weight:.2f}, perplexity={current_perplexity_weight:.2f}")
                    else:
                        print(f"  Scoring method: v_high only")
                    print(f"  Candidate scores for first state: {[f'{s:.3f}' for s in state_candidate_scores]}")
                    print(f"  Selected candidate index: {selected_candidate_idx % N}")
                    print(f"  Selection method: {selection_method}")
                    print(f"  Selected action preview: {all_candidate_actions[selected_candidate_idx][:100]}...")
            
            # PARTIAL CLEANUP: Clean up large candidate tensors but keep selected ones
            del all_candidate_states, all_candidate_base_states, all_candidate_instructions
            del all_candidate_actions, all_candidate_action_tokens, all_candidate_log_probs
            del all_candidate_scores, state_to_candidate_mapping
            
            # Force garbage collection and GPU cache cleanup after rejection sampling
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Now, build the batch for the environment step
            # OPTIMIZATION: Pre-create mapping from batch index to selected_actions index
            # For finished envs, fill with dummy actions (e.g., empty string or pad tokens)
            batch_to_selected_map = {unfinished_idx: pos for pos, unfinished_idx in enumerate(unfinished_indices)}
            pad_token_id = tokenizer.pad_token_id
            dummy_log_prob = torch.tensor(-5.0)
            
            batch_actions = []
            batch_action_tokens = []
            batch_log_probs = []
            batch_action_conversation_ids = []
            for i in range(batch_size):
                if already_dones[i]:
                    batch_actions.append("")
                    batch_action_tokens.append(torch.tensor([pad_token_id]))
                    batch_log_probs.append(dummy_log_prob)
                    # Get conversation ID even for done episodes
                    if hasattr(environment, 'conversations') and i < len(environment.conversations):
                        batch_action_conversation_ids.append(environment.conversations[i].conversation_id)
                    else:
                        batch_action_conversation_ids.append(f"batch_{i}_done")
                else:
                    j = batch_to_selected_map[i]
                    # VALIDATION: Verify state hasn't changed since action was generated
                    state_snapshot = selected_state_snapshots[j]
                    expected_state_idx = selected_state_indices[j]
                    if expected_state_idx != i:
                        raise ValueError(
                            f"CRITICAL MISMATCH: Action was generated for state {expected_state_idx}, "
                            f"but is being assigned to batch index {i}. State indices are misaligned!"
                        )
                    # Compare state snapshot with current batch_state
                    current_state = batch_state[i]
                    # States are lists of strings (conversation history)
                    snapshot_str = state_snapshot[0] if len(state_snapshot) > 0 else ""
                    current_str = current_state[0] if len(current_state) > 0 else ""
                    if snapshot_str != current_str:
                        # This is a warning, not an error, because state might have been updated
                        # But we should still log it for debugging
                        if steps_collected == 0 and i == unfinished_indices[0]:
                            print(f"\n[WARNING] State mismatch detected for batch index {i}:")
                            print(f"  State snapshot (when action was generated): {snapshot_str[:100]}...")
                            print(f"  Current batch_state: {current_str[:100]}...")
                            print(f"  This may indicate state was updated after action generation.")
                    
                    batch_actions.append(selected_actions[j])
                    batch_action_tokens.append(selected_action_tokens[j])
                    batch_log_probs.append(selected_log_probs[j])
                    # Get conversation ID for this action
                    if hasattr(environment, 'conversations') and i < len(environment.conversations):
                        batch_action_conversation_ids.append(environment.conversations[i].conversation_id)
                    else:
                        raise ValueError(f"Cannot get conversation ID for batch index {i}: conversations not available or index out of range")

            # FINAL CLEANUP: Now clean up the selected candidate lists after using them
            # Keep selected_state_snapshots and selected_state_indices for debug validation
            del selected_actions, selected_action_tokens, selected_log_probs, selected_candidate_indices

            # Use the carefully selected candidates from rejection sampling
            # (The candidates were selected above using either combined v_high + perplexity or just v_high scoring with epsilon-greedy exploration)

            # DEBUG: Show context for each component for the first state
            if steps_collected == 0:  # Only for first step of rollout
                # CRITICAL FIX: Use the first unfinished index to ensure all debug output is from the same conversation
                # This prevents mismatches when some episodes are already done
                debug_idx = unfinished_indices[0]
                
                # Get conversation ID and step number for proper tracking
                conv = environment.conversations[debug_idx]
                conversation_id = conv.conversation_id
                step_number = environment.current_step
                
                print("\n" + "="*80)
                print(f"DEBUG: CONTEXT ANALYSIS FOR STATE")
                print(f"  Batch Index: {debug_idx}")
                print(f"  Conversation ID: {conversation_id}")
                print(f"  Dialogue Step: {step_number}")
                print("="*80)
                
                # Verify that the batch_state matches the conversation
                conv_history = conv.get_conversation_history()
                if len(conv_history) > 0 and len(batch_state[debug_idx]) > 0:
                    batch_first_msg = batch_state[debug_idx][0]
                    conv_first_msg = conv_history[0]['content']
                    
                    if batch_first_msg != conv_first_msg:
                        print(f"\n[WARNING] MISMATCH DETECTED!")
                        print(f"  Batch state first message: {batch_first_msg[:100]}...")
                        print(f"  Conversation first message: {conv_first_msg[:100]}...")
                        print(f"  This indicates conversation states are misaligned!")
                
                # Validate that the action was generated from the correct state
                selected_pos = batch_to_selected_map[debug_idx]
                state_snapshot = selected_state_snapshots[selected_pos]
                snapshot_first_msg = state_snapshot[0] if len(state_snapshot) > 0 else ""
                current_first_msg = batch_state[debug_idx][0] if len(batch_state[debug_idx]) > 0 else ""
                
                if snapshot_first_msg != current_first_msg:
                    print(f"\n[WARNING] STATE-ACTION MISMATCH DETECTED!")
                    print(f"  State snapshot (when action was generated): {snapshot_first_msg[:100]}...")
                    print(f"  Current batch_state: {current_first_msg[:100]}...")
                    print(f"  Action: {batch_actions[debug_idx][:100]}...")
                    print(f"  This indicates the action was generated from a different state!")
                
                # Also check against conversation
                if len(conv_history) > 0:
                    conv_first_msg = conv_history[0]['content']
                    if snapshot_first_msg != conv_first_msg:
                        print(f"\n[WARNING] ACTION-CONVERSATION MISMATCH DETECTED!")
                        print(f"  State snapshot (when action was generated): {snapshot_first_msg[:100]}...")
                        print(f"  Conversation first message: {conv_first_msg[:100]}...")
                        print(f"  Action: {batch_actions[debug_idx][:100]}...")
                        print(f"  This indicates the action was generated from a different conversation!")
                    
                    # High-level agent context (what the high-level policy sees)
                    print(f"\nHIGH-LEVEL AGENT CONTEXT (batch_state[{debug_idx}], conv_id={conversation_id}, step={step_number}):")
                    state_item = batch_state[debug_idx]
                    if len(state_item) > 0:
                        print(f"  First message: {state_item[0][:100]}...")
                        if len(state_item) > 1:
                            print(f"  Last message: {state_item[-1][:100]}...")
                    else:
                        print("  Empty context")
                    
                    # High-level instruction
                    if not is_flat_agent:
                        print(f"\nHIGH-LEVEL INSTRUCTION (conv_id={conversation_id}, step={step_number}):")
                        print(f"  {high_level_instructions[debug_idx]}")
                    else:
                        print(f"\nFLAT AGENT: No high-level policy")
                    
                    # Low-level agent context (what the low-level policy sees)
                    print(f"\nLOW-LEVEL AGENT CONTEXT (batch_state[{debug_idx}], conv_id={conversation_id}, step={step_number}):")
                    state_item = batch_state[debug_idx]
                    if len(state_item) > 0:
                        print(f"  First message: {state_item[0][:100]}...")
                        if len(state_item) > 1:
                            print(f"  Last message: {state_item[-1][:100]}...")
                    else:
                        print("  Empty context")
                    
                    # Target model context (what the target model actually sees)
                    print(f"\nTARGET MODEL CONTEXT (what target model receives, conv_id={conversation_id}, step={step_number}):")
                    print(f"  Agent's action: {batch_actions[debug_idx]}")
                    # Note: Target model only sees the agent's action, not high-level instructions or original prompts
                    
                    # Agent's generated action
                    print(f"\nAGENT'S GENERATED ACTION (conv_id={conversation_id}, step={step_number}):")
                    if len(batch_actions[debug_idx]) > 110:
                        print(f"  {batch_actions[debug_idx][:100]}...{batch_actions[debug_idx][-10:]}")
                    else:
                        print(f"  {batch_actions[debug_idx]}")
                    
                    # Additional validation: Show the original base state to confirm alignment
                    original_prompt = conv.get_original_harmful_prompt()
                    if original_prompt:
                        print(f"\nORIGINAL BASE STATE (for validation, conv_id={conversation_id}, step={step_number}):")
                        print(f"  {original_prompt[:100]}...")
                    
                    print("="*80 + "\n")
            
            # Cleanup state snapshots after debug validation
            del selected_state_snapshots, selected_state_indices

            # Step environment - target model should only see agent's action, not high-level instructions or original prompts
            # Pass high-level instructions as system prompts and template IDs for tracking (template policies only)
            template_ids: Optional[List[str]] = None
            if uses_template_policy and high_level_instructions is not None:
                template_ids = []
                # Extract template IDs from strategies if available
                for idx in range(batch_size):
                    if chosen_strategy_indices is not None and chosen_strategy_indices[idx] is not None:
                        strategy_idx = chosen_strategy_indices[idx]
                        if hasattr(high_level_policy, 'strategies') and strategy_idx in high_level_policy.strategies:
                            template_id = high_level_policy.strategies[strategy_idx].get("id", "")
                        else:
                            template_id = ""
                    else:
                        template_id = ""
                    template_ids.append(template_id)
            
            step_kwargs = {
                "system_prompt": high_level_instructions,
                "template_ids": template_ids,
                "action_conversation_ids": batch_action_conversation_ids,
            }
            if predicted_tools_batch is not None:
                step_kwargs["predicted_tools"] = predicted_tools_batch
            next_state, responses, reward, done = environment.step(
                batch_actions,
                **step_kwargs,
            )
            
            # Capture guard scores for validation and debugging
            guard_scores_step = []
            target_guard_scores_step = []
            if hasattr(environment, "score_safety"):
                try:
                    from src.rewards.guard_service import GuardRewardService
                    guard = GuardRewardService(environment)
                    
                    # OPTIMIZATION: Pre-compute question and reference lists once instead of in list comprehensions
                    questions_list = []
                    reference_list = []
                    for i in range(len(batch_actions)):
                        # Process batch_state
                        if isinstance(batch_state[i], str):
                            questions_list.append(batch_state[i])
                        else:
                            questions_list.append(" ".join(str(s) for s in batch_state[i] if s))
                        # Process batch_base_state
                        if isinstance(batch_base_state[i], str):
                            reference_list.append(batch_base_state[i])
                        else:
                            reference_list.append(str(batch_base_state[i]))
                    
                    # Score agent actions
                    action_guard_scores = guard.sequence_reward(
                        batch_actions,
                        questions=questions_list,
                        reference_state=reference_list
                    )
                    guard_scores_step.extend(action_guard_scores.cpu().tolist())
                    
                    # Score target responses (reuse reference_list, questions are batch_actions)
                    target_guard_scores = guard.target_response_reward(
                        responses,
                        questions=batch_actions,
                        reference_state=reference_list[:len(responses)]
                    )
                    target_guard_scores_step.extend(target_guard_scores.cpu().tolist())
                except Exception as e:
                    # If guard scoring fails, log but don't crash
                    if getattr(agent_trainer.config.training, 'debug_mode', False):
                        print(f"[ROLLOUT] Warning: Failed to capture guard scores: {e}")
                    guard_scores_step = None
                    target_guard_scores_step = None
            else:
                guard_scores_step = None
                target_guard_scores_step = None
            
            # DEBUG: Show target response and reward for first state
            if steps_collected == 0:  # Only for first step of rollout
                print(f"\nTARGET RESPONSE:")
                print(f"  Full response: {responses[0]}")
                print(f"  Response length: {len(responses[0])}")
                
                print(f"\nREWARD:")
                print(f"  {reward[0]}")
                
                if guard_scores_step is not None and len(guard_scores_step) > 0:
                    print(f"\nGUARD SCORES:")
                    print(f"  Agent action guard score: {guard_scores_step[0]:.4f}")
                    if target_guard_scores_step is not None and len(target_guard_scores_step) > 0:
                        print(f"  Target response guard score: {target_guard_scores_step[0]:.4f}")
                
                print("="*80 + "\n")

            # ------------------------------------------------------------------
            # Print high-reward responses for debugging (mirrors Environment.step)
            # ------------------------------------------------------------------
            max_reward = torch.max(reward)  
            if max_reward > 0.5:
                high_reward_index = torch.argmax(reward).item()
                
                # Skip logging if this episode is already done (shouldn't happen, but safety check)
                if high_reward_index < len(already_dones) and already_dones[high_reward_index]:
                    print(f"\n[WARNING] Skipping logging for done episode at index {high_reward_index}")
                # Validate array lengths and indices to prevent mismatches
                elif high_reward_index >= len(batch_base_state):
                    print(f"\n[WARNING] High reward index {high_reward_index} >= batch_base_state length {len(batch_base_state)}, skipping logging")
                elif high_reward_index >= len(batch_state) or high_reward_index >= len(batch_actions) or high_reward_index >= len(responses):
                    print(f"\n[WARNING] Array length mismatch: batch_base_state={len(batch_base_state)}, batch_state={len(batch_state)}, batch_actions={len(batch_actions)}, responses={len(responses)}, index={high_reward_index}")
                elif (not is_flat_agent and (high_level_instructions is None or high_reward_index >= len(high_level_instructions))):
                    print(f"\n[WARNING] High-level instructions not available or index out of range: instructions={high_level_instructions is not None}, len={len(high_level_instructions) if high_level_instructions else 0}, index={high_reward_index}")
                else:
                    print("\n----- High Reward Responses (collect_rollout) -----")
                    truncate_length = 100

                    # CRITICAL: Use conversation object to get correct base_state (original question)
                    # This ensures alignment between logged question and the actual conversation
                    # The batch arrays might be misaligned if there's any filtering or reordering
                    conv = None
                    if hasattr(environment, 'conversations') and high_reward_index < len(environment.conversations):
                        conv = environment.conversations[high_reward_index]
                        if conv is not None:
                            # Get original harmful prompt from conversation (most reliable source)
                            if hasattr(conv, 'get_original_harmful_prompt'):
                                base_question = conv.get_original_harmful_prompt() or "[No original prompt]"
                            elif hasattr(conv, 'original_harmful_prompt') and conv.original_harmful_prompt:
                                base_question = conv.original_harmful_prompt
                            else:
                                # Fallback to batch_base_state
                                base_question = batch_base_state[high_reward_index] if isinstance(batch_base_state[high_reward_index], str) else str(batch_base_state[high_reward_index])
                        else:
                            # Fallback to batch arrays if conversation is None
                            base_question = batch_base_state[high_reward_index] if isinstance(batch_base_state[high_reward_index], str) else str(batch_base_state[high_reward_index])
                    else:
                        # Fallback to batch arrays if conversations not available
                        base_question = batch_base_state[high_reward_index] if isinstance(batch_base_state[high_reward_index], str) else str(batch_base_state[high_reward_index])
                    
                    # Use batch_state for state (captured at start of iteration, before action)
                    # This is the state that was used to generate the high-level instruction
                    state_item = batch_state[high_reward_index] if high_reward_index < len(batch_state) else "[Index Error]"
                    if isinstance(state_item, list):
                        state_text = " ".join(str(s) for s in state_item if s)
                    else:
                        state_text = str(state_item)
                    
                    # Get high-level instruction
                    if is_flat_agent or high_level_instructions is None:
                        high_level_instruction = "[Flat Agent - No High-Level Policy]"
                    else:
                        high_level_instruction = high_level_instructions[high_reward_index] if high_reward_index < len(high_level_instructions) and high_level_instructions[high_reward_index] is not None else "[None]"
                    
                    # Agent message and target response should be aligned with reward index
                    agent_prompt = batch_actions[high_reward_index] if high_reward_index < len(batch_actions) else "[Index Error]"
                    target_response = responses[high_reward_index] if high_reward_index < len(responses) else "[Index Error]"
                    
                    # VALIDATION: Check that all dialogue elements belong to the same conversation
                    if conv is not None:
                        conv_id = conv.conversation_id
                        mismatch_detected = False
                        mismatch_details = []
                        
                        # Check that conversation has messages (should have at least one turn by this point)
                        if not conv.user_messages or not conv.assistant_messages:
                            mismatch_detected = True
                            mismatch_details.append(f"Conversation has no messages: user_messages={len(conv.user_messages)}, assistant_messages={len(conv.assistant_messages)}")
                        else:
                            # Check agent message matches last user message in conversation
                            expected_agent_message = conv.user_messages[-1]
                            if agent_prompt != expected_agent_message:
                                mismatch_detected = True
                                mismatch_details.append(f"Agent message mismatch: batch_actions[{high_reward_index}] != conv.user_messages[-1]")
                            
                            # Check target response matches last assistant message in conversation
                            expected_target_response = conv.assistant_messages[-1]
                            if target_response != expected_target_response:
                                mismatch_detected = True
                                mismatch_details.append(f"Target response mismatch: responses[{high_reward_index}] != conv.assistant_messages[-1]")
                            
                            # Check high-level instruction matches last instruction in conversation
                            if not is_flat_agent and high_level_instructions is not None:
                                if conv.high_level_instructions:
                                    expected_hl_instruction = conv.high_level_instructions[-1]
                                    if high_level_instruction != expected_hl_instruction:
                                        mismatch_detected = True
                                        mismatch_details.append(f"High-level instruction mismatch: high_level_instructions[{high_reward_index}] != conv.high_level_instructions[-1]")
                        
                        if mismatch_detected:
                            error_msg = f"\n[ERROR] Dialogue ID mismatch detected for conversation {conv_id} at batch index {high_reward_index}:\n"
                            error_msg += "\n".join(f"  - {detail}" for detail in mismatch_details)
                            error_msg += f"\n  Conversation ID: {conv_id}"
                            error_msg += f"\n  Batch index: {high_reward_index}"
                            error_msg += f"\n  Question from conv: {base_question[:100]}..."
                            error_msg += f"\n  Agent message from batch: {agent_prompt[:100]}..."
                            if conv.user_messages:
                                error_msg += f"\n  Agent message from conv: {conv.user_messages[-1][:100]}..."
                            error_msg += f"\n  Target response from batch: {target_response[:100]}..."
                            if conv.assistant_messages:
                                error_msg += f"\n  Target response from conv: {conv.assistant_messages[-1][:100]}..."
                            raise ValueError(error_msg)

                    def _truncate(text: str, length: int = 100) -> str:
                        if text is None:
                            return "[None]"
                        return text[:length] + "..." if len(text) > length else text

                    print(f"Step: {steps_collected}, Batch Index: {high_reward_index}, Batch Size: {len(batch_base_state)}, Done: {already_dones[high_reward_index] if high_reward_index < len(already_dones) else 'N/A'}")
                    print(f"Question: { _truncate(base_question, truncate_length) }")
                    print(f"State: { _truncate(state_text, truncate_length) }")
                    print(f"High-level instruction: { _truncate(high_level_instruction, truncate_length) }")
                    print(f"Agent message: { _truncate(agent_prompt, truncate_length) }")
                    print(f"Target response: { _truncate(target_response, truncate_length) }")
                    try:
                        reward_value = reward[high_reward_index].item()
                        if torch.isfinite(torch.tensor(reward_value)):
                            print(f"Reward: {reward_value:.4f}")
                        else:
                            print(f"Reward: {reward_value}")
                    except (ValueError, TypeError):
                        print(f"Reward: {reward[high_reward_index]}")
                    print("--------------------------------")
            
            # Update state for next round
            state = next_state
            
            # Count steps collected
            steps_collected += sum([0 if d else 1 for d in already_dones]) # no step collected if done

            #build rollout data
            for i in range(len(batch_actions)):
                if not already_dones[i]:
                    states.append(state[i])
                    next_states.append(next_state[i])
                    actions.append(batch_action_tokens[i])
                    log_probs.append(batch_log_probs[i].detach())  # Ensure it's a tensor, not a list
                    rewards.append(reward[i])
                    dones.append(done[i])
                    if getattr(environment, "uses_tool_step_rewards", False):
                        collected_l3_rewards.append(float(reward[i].item()))
                    already_dones[i] = done[i]
                    base_states.append(batch_base_state[i])
                    # Accumulate guard scores if available
                    if guard_scores_step is not None and i < len(guard_scores_step):
                        guard_scores_list.append(guard_scores_step[i])
                    if target_guard_scores_step is not None and i < len(target_guard_scores_step):
                        target_guard_scores_list.append(target_guard_scores_step[i])
                    # Get the actual conversation object from the environment (which has high-level instructions)
                    # instead of using the stale base_conversations
                    if hasattr(environment, 'conversations') and i < len(environment.conversations):
                        conversations.append(environment.conversations[i])
                    else:
                        conversations.append(base_conversations[i])
                    # Store high-level policy data if available
                    if chosen_strategy_indices is not None and chosen_strategy_indices[i] is not None:
                        high_level_actions.append(torch.tensor(chosen_strategy_indices[i], dtype=torch.long))
                        lp = current_high_level_log_probs[i] if current_high_level_log_probs is not None else None
                        # Only append if lp is not None and not empty
                        if lp is not None:
                            if isinstance(lp, torch.Tensor):
                                if lp.numel() > 0:
                                    lp_tensor = lp.detach().to(dtype=torch.bfloat16)
                                    if lp_tensor.numel() == 1:
                                        lp_tensor = lp_tensor.reshape(())
                                    lp_tensor = lp_tensor.to(agent_trainer.device)
                                    high_level_log_probs.append(lp_tensor)
                                else:
                                    high_level_log_probs.append(None)
                            elif hasattr(lp, '__len__'):
                                if len(lp) > 0:
                                    lp_tensor = torch.tensor(lp, dtype=torch.bfloat16)
                                    if lp_tensor.numel() == 1:
                                        lp_tensor = lp_tensor.reshape(())
                                    lp_tensor = lp_tensor.to(agent_trainer.device)
                                    high_level_log_probs.append(lp_tensor)
                                else:
                                    high_level_log_probs.append(None)
                            else:
                                # Scalar number path
                                high_level_log_probs.append(torch.tensor(lp, dtype=torch.bfloat16).to(agent_trainer.device))
                        else:
                            high_level_log_probs.append(None)
                    else:
                        high_level_actions.append(None)
                        high_level_log_probs.append(None)

                    # Always store the instruction string used for this step so
                    # that we can compute a task-level reward that measures
                    # how well the low-level utterance follows the instruction.
                    instr_str = high_level_instructions[i] if high_level_instructions is not None else ""
                    collected_instructions.append(instr_str)
                    # Template IDs for tracking persona continuity (template policies only)
                    if collected_template_ids is not None:
                        if template_ids is not None and i < len(template_ids):
                            collected_template_ids.append(template_ids[i] or "")
                        else:
                            collected_template_ids.append("")
                    if policy_kind == "freeform" and step_state_texts is not None and current_high_level_log_probs is not None and i < len(step_state_texts) and i < len(current_high_level_log_probs):
                        freeform_state_history.append(step_state_texts[i])
                        freeform_logprob_history.append(float(current_high_level_log_probs[i]))
                        if step_base_questions is not None and i < len(step_base_questions):
                            freeform_base_history.append(step_base_questions[i])
                        else:
                            freeform_base_history.append("")
            
            # Reset if all conversations are done
            if all(already_dones):
                state = environment.reset()
                already_dones = [False] * len(state)
                # Reset base conversations
                base_conversations = conversations_from_batch(state)
            
            if tool_index_policy is not None and hasattr(environment, "episodes"):
                from src.agentharm_gen.step_scoring import (
                    compute_l1_option_reward,
                    l1_option_success,
                )

                for idx in range(batch_size):
                    if idx >= len(environment.episodes):
                        continue
                    episode_state = environment.episodes[idx]
                    if not episode_state.l1_option_active:
                        continue

                    episode_state.l1_option_steps_remaining -= 1
                    messages_at_option_start = list(
                        episode_state.messages[: episode_state.l1_option_start_message_len]
                    )
                    option_success = l1_option_success(
                        messages_at_option_start,
                        episode_state.messages,
                        episode_state.l1_option_tool,
                    )
                    episode_done = bool(done[idx].item())
                    option_timeout = episode_state.l1_option_steps_remaining <= 0

                    if option_success or option_timeout or episode_done:
                        next_state_text = _state_item_to_text(next_state[idx])
                        l1_r = compute_l1_option_reward(
                            episode_done, episode_state.last_harm_score_dict
                        )
                        tool_index_policy.replay_buffer.push(
                            episode_state.l1_option_state_text,
                            episode_state.l1_option_tool_index,
                            l1_r,
                            next_state_text,
                            episode_done,
                        )
                        collected_tool_indices.append(episode_state.l1_option_tool_index)
                        collected_l1_rewards.append(l1_r)
                        _clear_l1_option(episode_state)

            # After receiving the environment feedback, store high-level transitions
            if high_level_policy is not None and getattr(high_level_policy, "policy_kind", None) == "dqn" and chosen_strategy_indices is not None:
                for idx in range(batch_size):
                    if chosen_strategy_indices[idx] is None:
                        continue  # Was finished before this step
                    # Next state plain text
                    next_state_item = next_state[idx]
                    if isinstance(next_state_item, list):
                        next_state_text = " ".join(str(s) for s in next_state_item if s)
                    else:
                        next_state_text = str(next_state_item)

                    r = float(reward[idx]) if not isinstance(reward[idx], torch.Tensor) else reward[idx].item()
                    high_level_policy.replay_buffer.push(
                        chosen_state_texts[idx],
                        chosen_strategy_indices[idx],
                        r,
                        next_state_text,
                        bool(done[idx])
                    )
            
            # Clean up step-level variables to prevent memory accumulation
            del batch_actions, batch_action_tokens, batch_log_probs
            del next_state, responses, reward, done
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Convert lists to tensors
    if len(states) == 0:
        raise ValueError("Empty rollout collected.")
    
    # Use consistent max_length for all tokenizations to ensure same sequence length
    batch_state = _batch_tokenize_and_pad_state(tokenizer, states)
    batch_next_state = _batch_tokenize_and_pad_state(tokenizer, next_states)
    batch_base_state = _batch_tokenize_and_pad_state(tokenizer, base_states)
    
    # ensure proper batch construction
    assert len(states) == len(next_states) == len(base_states), (
        f"Batch size mismatch: states={len(states)}, next_states={len(next_states)}, "
        f"base_states={len(base_states)}. All state lists must have the same length."
    )
    
    assert batch_state.size(0) == batch_next_state.size(0) == batch_base_state.size(0), (
        f"Tensor batch size mismatch: batch_state={batch_state.size(0)}, "
        f"batch_next_state={batch_next_state.size(0)}, batch_base_state={batch_base_state.size(0)}. "
        f"All batch tensors must have the same first dimension."
    )
    
    assert len(states) > 0, "No states collected in rollout."
    
    # Left-pad action/log-prob sequences for decoder-only models
    from src.trainer.utils import left_pad_sequence

    batch_acts = left_pad_sequence(actions, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_log_probs = left_pad_sequence(log_probs, batch_first=True, padding_value=-5.0)
    assert batch_acts.shape == batch_log_probs.shape, f"Action and log_prob tensors must have the same shape, got {batch_acts.shape} and {batch_log_probs.shape}"
    batch_rews = torch.stack(rewards)
    batch_dones = torch.stack(dones)
    
    # Create rollout data
    # Filter out None values for high-level policy data
    valid_pairs = [
        (a, lp)
        for a, lp in zip(high_level_actions, high_level_log_probs)
        if a is not None and lp is not None and lp.numel() > 0
    ]
    valid_high_level_actions = [a for a, lp in valid_pairs]
    valid_high_level_log_probs = [lp for a, lp in valid_pairs]

    # Require scalar log-probs; reshape any singletons to 0-d scalars and fail if not exactly one element
    if valid_high_level_log_probs:
        for idx, lp in enumerate(valid_high_level_log_probs):
            assert isinstance(lp, torch.Tensor), f"Expected tensor log prob at index {idx}, got {type(lp)}"
            assert lp.numel() == 1, f"High-level log prob at index {idx} must have exactly one element, got shape {lp.shape}"
        valid_high_level_log_probs = [lp.reshape(()) for lp in valid_high_level_log_probs]
        # Ensure all tensors are on the same device (GPU)
        valid_high_level_log_probs = [lp.to(agent_trainer.device) for lp in valid_high_level_log_probs]

    high_level_actions_tensor = torch.stack(valid_high_level_actions) if valid_high_level_actions else None
    high_level_log_probs_tensor = torch.stack(valid_high_level_log_probs) if valid_high_level_log_probs else None
    
    # Convert guard scores to tensors if available
    guard_scores_tensor = torch.tensor(guard_scores_list, dtype=torch.float32) if guard_scores_list else None
    target_guard_scores_tensor = torch.tensor(target_guard_scores_list, dtype=torch.float32) if target_guard_scores_list else None
    
    # Flatten freeform_raw_output_ids_history to match order of other freeform histories (per sample across all steps)
    flattened_raw_output_ids = None
    if freeform_raw_output_ids_history:
        # freeform_raw_output_ids_history is a list of batches: [step0_batch, step1_batch, ...]
        # Each step_batch is a list of tensors: [sample0_tensor, sample1_tensor, ...]
        # freeform_state_history order is: [sample0_step0, sample0_step1, ..., sample1_step0, sample1_step1, ...]
        # So we need to transpose: for each sample, collect all its steps across all rollout steps
        num_steps = len(freeform_raw_output_ids_history)
        if num_steps > 0:
            num_samples_per_step = len(freeform_raw_output_ids_history[0])
            flattened_raw_output_ids = []
            # Transpose: iterate by sample, then by step
            for sample_idx in range(num_samples_per_step):
                for step_idx in range(num_steps):
                    if step_idx < len(freeform_raw_output_ids_history) and sample_idx < len(freeform_raw_output_ids_history[step_idx]):
                        flattened_raw_output_ids.append(freeform_raw_output_ids_history[step_idx][sample_idx])
    
    rollout = RolloutData(
        batch_state=batch_state,
        batch_next_state=batch_next_state,
        batch_acts=batch_acts,
        batch_log_probs=batch_log_probs,
        batch_rews=batch_rews,
        batch_dones=batch_dones,
        batch_base_state=batch_base_state,
        conversations=conversations,
        high_level_actions=high_level_actions_tensor,
        high_level_log_probs=high_level_log_probs_tensor,
        high_level_instructions=collected_instructions,
        collected_template_ids=collected_template_ids,
        high_level_state_texts=freeform_state_history if freeform_state_history else None,
        high_level_instruction_log_probs=freeform_logprob_history if freeform_logprob_history else None,
        high_level_base_questions=freeform_base_history if freeform_base_history else None,
        guard_scores=guard_scores_tensor,
        target_guard_scores=target_guard_scores_tensor,
        high_level_raw_output_ids=flattened_raw_output_ids,
        tool_index_actions=(
            torch.tensor(collected_tool_indices, dtype=torch.long)
            if collected_tool_indices
            else None
        ),
        l1_rewards=(
            torch.tensor(collected_l1_rewards, dtype=torch.float32)
            if collected_l1_rewards
            else None
        ),
        l3_rewards=(
            torch.tensor(collected_l3_rewards, dtype=torch.float32)
            if collected_l3_rewards
            else None
        ),
    )

    # Final cleanup to prevent memory accumulation
    del states, next_states, actions, log_probs, rewards, dones, base_states
    del high_level_actions, high_level_log_probs, collected_instructions, collected_template_ids
    del valid_high_level_actions, valid_high_level_log_probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rollout


def load_templates(templates_path: str = "src/configs/templates.json") -> List[Dict[str, Any]]:
    """
    Load templates from JSON file.
    
    Args:
        templates_path: Path to the templates JSON file
        
    Returns:
        List of template dictionaries
    """
    try:
        with open(templates_path, 'r') as f:
            template_data = json.load(f)
        
        # Extract templates array from JSON
        if "templates" in template_data:
            templates = template_data["templates"]
        else:
            templates = template_data
        
        return templates
    except FileNotFoundError:
        return _get_default_templates()
    except Exception as e:
        return _get_default_templates()


def _get_default_templates() -> List[Dict[str, Any]]:
    raise ValueError("Templates file not found")

def extract_topic_from_state(
    state: Union[str, List[str]], 
    token_critic: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_q_low: bool = False
) -> str:
    """
    Extract a topic from the current state for template formatting.
    
    Args:
        state: Current state (string or list of strings)
        token_critic: Token-level critic (q_low) for intelligent token selection (optional)
        tokenizer: Tokenizer for tokenization (required if use_q_low=True)
        use_q_low: Whether to use q_low for intelligent topic extraction
        
    Returns:
        Extracted topic string
    """
    # Use q_low-based extraction if requested and components are available
    if use_q_low and token_critic is not None and tokenizer is not None:
        # Convert state to string if it's a list
        if isinstance(state, list):
            if len(state) > 0:
                state_text = str(state[0])
            else:
                state_text = ""
        else:
            state_text = str(state)
        
        # Tokenize the state
        encoding = tokenizer(
            state_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Use q_low to extract topic
        topic = extract_topic_using_q_low(
            token_critic=token_critic,
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            tokenizer=tokenizer,
            max_topic_length=50,
            top_k_tokens=5
        )
        
        return topic
    
    # Fallback to simple rule-based extraction
    # Convert state to string if it's a list
    if isinstance(state, list):
        # For conversation states, extract topic from the first message (original question)
        if len(state) > 0:
            # Take the first message as the original question/topic
            state_text = str(state[0])
        else:
            state_text = ""
    else:
        state_text = str(state)
    
    # Clean up the text
    state_text = state_text.strip()
    
    # Simple topic extraction - take first few meaningful words
    words = state_text.split()
    if len(words) > 3:
        # Take first 3-5 words as topic, but avoid very long topics
        topic_words = words[:min(5, len(words))]
        topic = " ".join(topic_words)
        # Truncate if too long
        if len(topic) > 50:
            topic = topic[:47] + "..."
    elif len(words) > 0:
        topic = words[0]
    else:
        topic = "this topic"
    
    return topic

def format_high_level_instruction_with_user_type(instruction: str, topic: str, question: str, conversation: Any) -> Tuple[str, str]:
    """
    Format a high-level instruction by replacing placeholders and using the most recent template ID as user_type.
    
    Args:
        instruction: The high-level instruction template
        topic: The extracted topic
        question: The original question
        conversation: The conversation object to get the most recent template ID
        
    Returns:
        Tuple of (formatted_instruction, template_id)
    """
    # Get the most recent template ID from the conversation
    template_id = conversation.get_latest_template_id() if conversation else ""
    
    # Replace placeholders in the instruction
    formatted_instruction = instruction
    if "{topic}" in formatted_instruction:
        formatted_instruction = formatted_instruction.replace("{topic}", topic)
    if "{question}" in formatted_instruction:
        formatted_instruction = formatted_instruction.replace("{question}", question)
    if "{user_type}" in formatted_instruction:
        if template_id:
            # Use the template ID as user_type
            formatted_instruction = formatted_instruction.replace("{user_type}", template_id)
        else:
            # If no template ID is available, remove the user_type placeholder entirely
            # This avoids the redundant "who is a user" text
            formatted_instruction = formatted_instruction.replace("who is a {user_type}", "")
            formatted_instruction = formatted_instruction.replace("{user_type}", "")
            # Clean up any double spaces that might result
            formatted_instruction = " ".join(formatted_instruction.split())
    
    return formatted_instruction, template_id


def extract_topic_using_q_low(
    self,
    token_critic: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    max_topic_length: int = 50,
    top_k_tokens: int = 5
) -> str:
    """
    Extract topic using q_low to identify the most important tokens.
    
    This function uses the token-level critic to identify the most valuable tokens
    in the input sequence, then constructs a topic from those tokens.
    
    Args:
        token_critic: Token-level critic (q_low) model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        tokenizer: Tokenizer for decoding
        max_topic_length: Maximum length of extracted topic
        top_k_tokens: Number of top tokens to use for topic extraction
        
    Returns:
        Extracted topic string
    """
    # Ensure token_critic is on the correct device
    token_critic.to(input_ids.device)
    
    # Get token-level values from q_low
    with torch.no_grad():
        token_values = token_critic(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Mask out pad tokens
    token_values = token_values * attention_mask.float()
    
    # Select top-k tokens for each sequence in the batch
    batch_size, seq_len = input_ids.shape
    selected_positions = []
    
    for batch_idx in range(batch_size):
        # Get valid positions (non-pad tokens)
        valid_positions = attention_mask[batch_idx].nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
        
        # Get values for valid positions only
        valid_values = token_values[batch_idx][valid_positions]
        
        # Select top-k tokens (or all if fewer than k)
        k = min(top_k_tokens, len(valid_positions))
        top_k_indices = torch.topk(valid_values, k).indices
        top_k_positions = valid_positions[top_k_indices]
        
        selected_positions.extend(top_k_positions.cpu().tolist())
    
    # Remove duplicates and sort
    selected_positions = sorted(list(set(selected_positions)))
    
    if not selected_positions:
        return "this topic"
    
    # Extract tokens at selected positions
    selected_tokens = []
    for pos in selected_positions:
        if pos < input_ids.shape[1]:
            token_id = input_ids[0, pos].item()  # Assume batch_size=1 for topic extraction
            selected_tokens.append(token_id)
    
    # Decode the selected tokens
    if selected_tokens:
        topic_text = tokenizer.decode(selected_tokens, skip_special_tokens=True)
        
        # Clean up the topic
        topic_text = topic_text.strip()
        
        # Truncate if too long
        if len(topic_text) > max_topic_length:
            topic_text = topic_text[:max_topic_length-3] + "..."
        
        return topic_text if topic_text else "this topic"
    else:
        return "this topic"


 