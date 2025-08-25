from src.agent_trainer import AgentTrainer, _batch_tokenize_and_pad_state
import torch
import random
import json
from typing import List, Optional, Any, Dict, Tuple

class MCQAgentTrainer(AgentTrainer):
    """
    Specialized AgentTrainer for Multiple-Choice QA (MCQ) environments.
    This class handles dual reward streams: high-level (HL) and low-level (LL) rewards.
    - HL reward: Used for sequence_critic training (e.g., correctness of submitted answer) 
                 to provide value estimates for HL policy advantages in PPO
    - LL reward: Used for token_critic training (e.g., judge score for each utterance)
                 for low-level token-wise learning
    
    Key difference from base AgentTrainer:
    - Overrides train_critics_dual_reward() to handle separate reward signals
    - Avoids the "NoneType has no attribute 'train'" error when passing None critics
    - Supports hl_only_training flag to freeze LL policy training
    
    All MCQ-specific logic is isolated here to preserve the base codebase for publication.
    """
    
    def __init__(self, *args, hl_only_training: bool = False, **kwargs):
        """
        Initialize MCQAgentTrainer with optional HL-only training mode.
        
        Args:
            hl_only_training: If True, only train HL policy and freeze LL agent
            *args, **kwargs: Arguments passed to parent AgentTrainer
        """
        super().__init__(*args, **kwargs)
        self.hl_only_training = bool(hl_only_training)

    def collect_rollout(self, policy, environment, tokenizer, rollout_steps, config, high_level_policy=None, templates_path=None, episode=0):
        """
        Collect a rollout, storing both HL and LL rewards for MCQ environments.
        Robustly handles partially finished episodes (some envs done, others not).
        """
        # Load system prompt from templates file if available
        system_prompt_from_templates = None
        if templates_path and high_level_policy is not None:
            with open(templates_path, 'r') as f:
                template_data = json.load(f)
            system_prompt_from_templates = template_data.get("system_prompt")
            if system_prompt_from_templates and getattr(config.training, 'debug_mode', False):
                print(f"[DEBUG] Loaded system prompt: {system_prompt_from_templates}")

        state = environment.reset()
        batch_size = len(state)
        steps_collected = 0
        already_dones = [False] * batch_size
        batch_states = []
        batch_next_states = []
        batch_actions = []  # Will store token IDs, not decoded strings
        batch_action_texts = []  # For logging/analysis only
        batch_log_probs = []
        batch_hl_rews = []
        batch_ll_rews = []
        batch_dones = []
        batch_base_states = []
        batch_instructions = []
        # Collect exactly one batch of the expected size
        while len(batch_states) < batch_size and steps_collected < rollout_steps:
            # Find indices of unfinished episodes
            unfinished_indices = [i for i, done in enumerate(already_dones) if not done]
            if not unfinished_indices:
                break  # All episodes finished
            # Prepare state for unfinished episodes
            unfinished_states = [state[i] for i in unfinished_indices]
            # High-level instructions (if any)
            if high_level_policy is not None:
                policy_kind = getattr(high_level_policy, "policy_kind", None)
                instructions_unfinished = []
                chosen_indices_unfinished = []
                chosen_ids_unfinished = []
                if policy_kind in ("dqn", "ppo"):
                    # Store additional data for PPO training
                    hl_log_probs_unfinished = []
                    hl_entropy_unfinished = []
                    
                    for state_item in unfinished_states:
                        # High-level epsilon exploration schedule (anneals like low-level)
                        hl_epsilon_start = getattr(config.training, 'hl_epsilon_start', 0.8)
                        hl_epsilon_end = getattr(config.training, 'hl_epsilon_end', 0.0)
                        hl_epsilon_decay_episodes = getattr(config.training, 'hl_epsilon_decay_episodes', 100)
                        hl_epsilon = hl_epsilon_end + (hl_epsilon_start - hl_epsilon_end) * max(0, (hl_epsilon_decay_episodes - episode) / hl_epsilon_decay_episodes)
                        if isinstance(state_item, list):
                            state_text = " ".join(str(s) for s in state_item if s)
                        else:
                            state_text = str(state_item)
                        enc = tokenizer(state_text, return_tensors="pt", truncation=True)
                        
                        # Get policy distribution for PPO training
                        with torch.no_grad():
                            # Ensure encodings are on the same device as the high-level policy
                            policy_device = next(high_level_policy.parameters()).device
                            enc = enc.to(policy_device)
                            
                            # Verify device placement
                            assert enc.input_ids.device == policy_device, f"enc.input_ids device mismatch: {enc.input_ids.device} vs {policy_device}"
                            assert enc.attention_mask.device == policy_device, f"enc.attention_mask device mismatch: {enc.attention_mask.device} vs {policy_device}"
                            
                            strategy_logits = high_level_policy.forward(enc.input_ids, enc.attention_mask)
                            strategy_logits_clamped = torch.clamp(strategy_logits, min=-5, max=5)
                            strategy_probs = torch.nn.functional.softmax(strategy_logits_clamped, dim=-1)
                            action_dist = torch.distributions.Categorical(strategy_probs)
                            
                            # Verify action distribution device placement
                            assert strategy_logits.device == policy_device, f"strategy_logits device mismatch: {strategy_logits.device} vs {policy_device}"
                            assert strategy_probs.device == policy_device, f"strategy_probs device mismatch: {strategy_probs.device} vs {policy_device}"
                            
                            # Store entropy for PPO training
                            entropy = action_dist.entropy()
                            hl_entropy_unfinished.append(entropy.item())
                        
                        # Epsilon-greedy selection for high-level action
                        if random.random() < hl_epsilon:
                            idx = random.randint(0, len(high_level_policy.strategies) - 1)
                            # For random actions, log prob is log(1/num_strategies)
                            log_prob = torch.log(torch.tensor(1.0 / len(high_level_policy.strategies), device=enc.input_ids.device))
                        else:
                            idx = high_level_policy.select_action(enc.input_ids, enc.attention_mask)
                            # Get log prob for selected action
                            log_prob = action_dist.log_prob(torch.tensor(idx, device=enc.input_ids.device))
                        
                        strategy = high_level_policy.strategies[idx]
                        instruction = strategy["instruction"]
                        instructions_unfinished.append(instruction)
                        chosen_indices_unfinished.append(idx)
                        chosen_ids_unfinished.append(strategy.get("id", ""))
                        hl_log_probs_unfinished.append(log_prob.item())



                elif policy_kind in ("llm", "hybrid"):
                    state_texts_batch = []
                    for state_item in unfinished_states:
                        if isinstance(state_item, list):
                            state_texts_batch.append(" ".join(str(s) for s in state_item if s))
                        else:
                            state_texts_batch.append(str(state_item))
                    act_result = high_level_policy.act(state_texts_batch)
                    if isinstance(act_result, tuple):
                        instructions_unfinished = act_result[0]
                    else:
                        instructions_unfinished = act_result
                    # For LLM/hybrid we do not have discrete indices; mark as -1 and blank ids
                    chosen_indices_unfinished = [-1] * len(unfinished_states)
                    chosen_ids_unfinished = [""] * len(unfinished_states)
                else:
                    raise ValueError(f"Unsupported high_level_policy.policy_kind: {policy_kind}")
                instructions = []
                chosen_strategy_indices = [None] * batch_size
                template_ids_full = [""] * batch_size
                instr_ptr = 0
                for i in range(batch_size):
                    if already_dones[i]:
                        instructions.append(None)
                        chosen_strategy_indices[i] = None
                        template_ids_full[i] = ""
                    else:
                        instructions.append(instructions_unfinished[instr_ptr])
                        chosen_strategy_indices[i] = chosen_indices_unfinished[instr_ptr]
                        template_ids_full[i] = chosen_ids_unfinished[instr_ptr]
                        instr_ptr += 1
            else:
                instructions = [None] * batch_size
                chosen_strategy_indices = [None] * batch_size
                template_ids_full = [""] * batch_size
            # Debug: print integer list of HL actions for this step (None -> -1)
            try:
                hl_actions = [(-1 if (chosen_strategy_indices is None or chosen_strategy_indices[i] is None) else int(chosen_strategy_indices[i])) for i in range(batch_size)]
                print(hl_actions)
            except Exception:
                if getattr(config.training, 'debug_mode', False):
                    raise
            # Build strict instructions on submit and patch states so LL sees question+choices upfront
            try:
                qc_texts = environment.get_question_choices_text() if hasattr(environment, 'get_question_choices_text') else environment.base_state
            except Exception:
                qc_texts = environment.base_state
            
            # Check if we need to force submission due to max turns reached
            submit_mask_full = [(tid == "mc_submit_answer") for tid in template_ids_full]
            
            # Force submission for slots that have reached max turns
            if hasattr(environment, '_turns') and hasattr(environment, 'max_turns'):
                for i in range(batch_size):
                    if not already_dones[i] and environment._turns[i] >= environment.max_turns:
                        submit_mask_full[i] = True
                        # Override template ID to force submit action
                        template_ids_full[i] = "mc_submit_answer"
                        # Override instruction to use submit template instruction from templates file
                        if instructions[i] is not None and templates_path:
                            with open(templates_path, 'r') as f:
                                template_data = json.load(f)
                            for template in template_data.get("templates", []):
                                if template.get("id") == "mc_submit_answer":
                                    instructions[i] = template.get("instruction", instructions[i])
                                    break
                        if getattr(config.training, 'debug_mode', False):
                            print(f"DEBUG: Forcing submit for slot {i} at turn {environment._turns[i]} (max_turns={environment.max_turns})")
                
                # Log forced submission statistics
                forced_submissions = sum(1 for i in range(batch_size) if not already_dones[i] and environment._turns[i] >= environment.max_turns)
                if forced_submissions > 0 and getattr(config.training, 'debug_mode', False):
                    print(f"DEBUG: {forced_submissions} slots forced to submit at max_turns={environment.max_turns}")

            # Extract clean question text from base_state for {context}
            def _extract_question_text(text):
                t = str(text or "")
                if t.startswith("Question:"):
                    return t.split("Question:", 1)[1].strip()
                return t.strip()
            context_texts = [_extract_question_text(bs) for bs in environment.base_state]

            # Helper to parse choices from a formatted question+choices block
            def _parse_choices(formatted_qc):
                choices = []
                for ln in str(formatted_qc or "").splitlines():
                    if len(ln) >= 3 and ln[1] == ')' and ln[0].isalpha():
                        label = ln[0].upper()
                        choice = ln.split(") ", 1)[1] if ") " in ln else ln[2:]
                        choices.append((label, choice.strip()))
                return choices

            # Precompute a choices-only text block (without the leading Question line)
            choices_only_texts = []
            for txt in qc_texts:
                parsed = _parse_choices(txt)
                block_lines = [f"{lbl}) {ch}" for (lbl, ch) in parsed]
                choices_only_texts.append("\n".join(block_lines))

            send_instructions = []
            patched_states = []
            for i in range(batch_size):
                # Prepare instruction
                base_instr = instructions[i]
                tid = str(template_ids_full[i] or "")
                
                if base_instr is not None:
                    # Determine option variables if examining a specific choice
                    option_label = ""
                    option_text = ""
                    if tid.startswith("mc_examine_"):
                        letter = tid.split("_")[-1].upper()
                        if letter in ("A", "B", "C", "D"):
                            parsed = _parse_choices(qc_texts[i] if i < len(qc_texts) else "")
                            for lbl, txt in parsed:
                                if lbl == letter:
                                    option_label = lbl
                                    option_text = txt
                                    break
                    
                    # For ALL templates, use the template field from high-level policy with variable substitution
                    if high_level_policy is not None:
                        # Find the strategy to get its template field
                        template_text = None
                        for strategy in high_level_policy.strategies:
                            if strategy.get("id") == tid:
                                template_text = strategy.get("template", strategy.get("instruction"))
                                break
                        if template_text:
                            formatted_instruction = template_text.format(
                                context=(context_texts[i] if i < len(context_texts) else ""),
                                choices_text=(choices_only_texts[i] if i < len(choices_only_texts) else ""),
                                option_label=option_label,
                                option_text=option_text,
                            )
                        else:
                            # Fallback to base instruction with variable substitution
                            formatted_instruction = base_instr.format(
                                context=(context_texts[i] if i < len(context_texts) else ""),
                                choices_text=(choices_only_texts[i] if i < len(choices_only_texts) else ""),
                                option_label=option_label,
                                option_text=option_text,
                            )
                    else:
                        # No high-level policy, use base instruction with variable substitution
                        formatted_instruction = base_instr.format(
                            context=(context_texts[i] if i < len(context_texts) else ""),
                            choices_text=(choices_only_texts[i] if i < len(choices_only_texts) else ""),
                            option_label=option_label,
                            option_text=option_text,
                        )
                    
                    # System prompt will be handled separately through chat template structure
                    # The policy.act method will construct the proper chat template with system role
                else:
                    formatted_instruction = base_instr
                send_instructions.append(formatted_instruction)

                # Patch state head
                s = state[i]
                qc = qc_texts[i] if i < len(qc_texts) else (environment.base_state[i] if i < len(environment.base_state) else "")
                if submit_mask_full[i]:
                    if isinstance(s, list) and len(s) > 0:
                        # For submit actions, ensure the LL agent sees the full question and choices clearly
                        # Also include a summary of the reasoning process if available
                        reasoning_summary = ""
                        if len(s) > 1:
                            # Extract reasoning from previous LL responses (odd indices after base question)
                            reasoning_parts = []
                            for j in range(1, len(s), 2):  # Skip base question, get LL responses
                                if j < len(s) and s[j]:
                                    reasoning_parts.append(str(s[j]))
                            if reasoning_parts:
                                reasoning_summary = "\n\nPrevious reasoning:\n" + "\n".join(reasoning_parts)
                        
                        patched_states.append([qc + reasoning_summary] + s[1:])
                    else:
                        patched_states.append([qc])
                else:
                    patched_states.append(s)
            # Single generation for all unfinished with proper instructions and states
            unfinished_patched = [patched_states[i] for i in unfinished_indices]
            unfinished_instrs = [send_instructions[i] for i in unfinished_indices]
            try:
                base_state_all = environment.get_base_state() if hasattr(environment, 'get_base_state') else environment.base_state
            except Exception:
                base_state_all = environment.base_state
            unfinished_base = [base_state_all[i] for i in unfinished_indices]
            
            # Debug mode: print formatted instructions
            if getattr(config.training, 'debug_mode', False):
                print("=" * 80)
                print("DEBUG: FORMATTED INSTRUCTIONS AND SYSTEM PROMPTS")
                print("=" * 80)
                if system_prompt_from_templates:
                    print(f"Template System Prompt: {system_prompt_from_templates}")
                    print("-" * 40)
                for i, instr in enumerate(send_instructions):
                    tid = template_ids_full[i] if i < len(template_ids_full) else ""
                    is_submit = submit_mask_full[i] if i < len(submit_mask_full) else False
                    print(f"Slot {i}: template_id='{tid}', is_submit={is_submit}")
                    print(f"Formatted Template: {instr}")
                    if is_submit:
                        print(f"  [SUBMIT ACTION]")
                    print("-" * 40)
            
            # Build proper chat templates for LL policy with system prompt and conversation history
            unfinished_chat_contexts = []
            for idx, slot_idx in enumerate(unfinished_indices):
                # Start with system message from templates
                chat_context = []
                
                # Special system prompt for submit actions
                tid = template_ids_full[slot_idx] if slot_idx < len(template_ids_full) else ""
                if tid == "mc_submit_answer":
                    # Override system prompt for submit actions to be extremely clear
                    submit_system_prompt = "You are a precise answer generator. When asked to submit a final answer, you MUST output EXACTLY ONE character: A, B, C, or D. No words, no explanation, no punctuation. Just the single letter."
                    chat_context.append({'role': 'system', 'content': submit_system_prompt})
                elif system_prompt_from_templates:
                    chat_context.append({'role': 'system', 'content': system_prompt_from_templates})
                
                # Build conversation history: 
                # - Past HL instructions are obfuscated (empty user messages)
                # - LL responses are included as assistant messages
                # - Current HL instruction is the most recent user message
                state_history = unfinished_patched[idx] if isinstance(unfinished_patched[idx], list) else [unfinished_patched[idx]]
                
                # Add conversation history (HL/LL pairs)
                # Skip the first message (base question) and process pairs
                if len(state_history) > 1:
                    conversation_pairs = state_history[1:]  # Skip base question
                    for i, msg in enumerate(conversation_pairs):
                        if i % 2 == 0:
                            # HL instruction (user) - obfuscate past instructions
                            chat_context.append({'role': 'user', 'content': ""})
                        else:
                            # LL response (assistant)
                            chat_context.append({'role': 'assistant', 'content': str(msg)})
                
                # Add current HL instruction as the most recent user message
                current_instruction = send_instructions[slot_idx]
                
                # Special handling for submit actions - provide better context
                tid = template_ids_full[slot_idx] if slot_idx < len(template_ids_full) else ""
                if tid == "mc_submit_answer":
                    # For submit actions, add a reminder about the task and previous reasoning
                    submit_context = f"FINAL ANSWER REQUIRED\n\nThis is the submission step. You MUST output EXACTLY ONE character: A, B, C, or D.\n\n{current_instruction}"
                    chat_context.append({'role': 'user', 'content': submit_context})
                else:
                    chat_context.append({'role': 'user', 'content': current_instruction})
                
                unfinished_chat_contexts.append(chat_context)
            
            # Debug mode: print final chat contexts
            if getattr(config.training, 'debug_mode', False):
                print("DEBUG: CHAT CONTEXTS (to LL Policy)")
                print("=" * 80)
                for i, chat_context in enumerate(unfinished_chat_contexts):
                    slot_idx = unfinished_indices[i]
                    tid = template_ids_full[slot_idx] if slot_idx < len(template_ids_full) else ""
                    is_submit = tid == "mc_submit_answer"
                    print(f"Unfinished slot {slot_idx}: template_id='{tid}', is_submit={is_submit}")
                    for msg in chat_context:
                        role = msg['role']
                        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        if is_submit and role == 'user' and content.startswith('FINAL ANSWER REQUIRED'):
                            print(f"  {role}: [SUBMIT CONTEXT] {content}")
                        else:
                            print(f"  {role}: {content}")
                    print("-" * 40)
            
            # Generate responses using pre-built chat contexts
            # We'll directly call the policy's generation method with our custom chat contexts
            actions_unfinished, action_token_ids_unfinished = self._generate_with_chat_contexts(
                policy, 
                unfinished_chat_contexts,
                unfinished_base,
                tokenizer,
                config
            )
            
            # Debug mode: print LL agent actions
            if getattr(config.training, 'debug_mode', False):
                print("DEBUG: LL AGENT ACTIONS")
                print("=" * 80)
                for i, action in enumerate(actions_unfinished):
                    slot_idx = unfinished_indices[i]
                    tid = template_ids_full[slot_idx] if slot_idx < len(template_ids_full) else ""
                    is_submit = tid == "mc_submit_answer"
                    action_type = "[SUBMIT ANSWER]" if is_submit else "[REASONING]"
                    print(f"Unfinished slot {slot_idx}: {action_type} '{action}'")
                    if is_submit:
                        # Check if the answer is a single letter
                        clean_action = action.strip().upper()
                        if len(clean_action) == 1 and clean_action in "ABCD":
                            print(f"  ✓ Valid submit answer: {clean_action}")
                            
                            # Check correctness if we have access to environment data
                            try:
                                if hasattr(environment, '_examples') and hasattr(environment, '_indices'):
                                    # Get the correct answer for this slot
                                    correct_example = environment._examples[environment._indices[slot_idx]]
                                    correct_answer_letter = chr(ord('A') + correct_example.answer_index)
                                    
                                    # Check if the answer is correct
                                    is_correct = (clean_action == correct_answer_letter)
                                    correctness_symbol = "✓" if is_correct else "✗"
                                    correctness_text = "CORRECT" if is_correct else "INCORRECT"
                                    
                                    print(f"  {correctness_symbol} Answer {correctness_text}: {clean_action} (correct: {correct_answer_letter})")
                                else:
                                    print(f"  ? Correctness unknown (no environment data access)")
                            except Exception as e:
                                print(f"  ? Error checking correctness: {e}")
                        else:
                            print(f"  ✗ Invalid submit answer format: '{action}' (expected single letter A-D)")
                print("-" * 40)
            
            # Always print submit action statistics (not just in debug mode)
            submit_actions = [(i, action) for i, action in enumerate(actions_unfinished) 
                             if template_ids_full[unfinished_indices[i]] == "mc_submit_answer"]
            
            if submit_actions:
                print("SUBMIT ACTION STATISTICS:")
                print("-" * 40)
                
                # Validity statistics
                valid_count = 0
                invalid_count = 0
                correct_count = 0
                total_count = 0
                
                for i, action in submit_actions:
                    slot_idx = unfinished_indices[i]
                    clean_action = action.strip().upper()
                    
                    # Check validity
                    if len(clean_action) == 1 and clean_action in "ABCD":
                        valid_count += 1
                        
                        # Check correctness if we have access to environment data
                        try:
                            if hasattr(environment, '_examples') and hasattr(environment, '_indices'):
                                correct_example = environment._examples[environment._indices[slot_idx]]
                                correct_answer_letter = chr(ord('A') + correct_example.answer_index)
                                is_correct = (clean_action == correct_answer_letter)
                                
                                if is_correct:
                                    correct_count += 1
                                total_count += 1
                                
                                print(f"  Slot {slot_idx}: {clean_action} vs {correct_answer_letter} → {'✓' if is_correct else '✗'}")
                            else:
                                print(f"  Slot {slot_idx}: {clean_action} (correctness unknown)")
                        except Exception as e:
                            print(f"  Slot {slot_idx}: {clean_action} (error checking correctness: {e})")
                    else:
                        invalid_count += 1
                        print(f"  Slot {slot_idx}: INVALID '{action}' (expected single letter A-D)")
                
                # Print summary statistics
                print("-" * 40)
                print(f"VALIDITY: {valid_count}/{len(submit_actions)} = {(valid_count/len(submit_actions)*100):.1f}% valid format")
                
                if total_count > 0:
                    accuracy = (correct_count / total_count) * 100
                    print(f"ACCURACY: {correct_count}/{total_count} = {accuracy:.1f}% correct answers")
                else:
                    print("ACCURACY: No valid answers to check")
                
                print("-" * 40)
            
            # Map actions back to batch, fill dummy for finished
            padded_actions = []  # token IDs
            padded_action_texts = []  # decoded strings
            action_ptr = 0
            for i in range(batch_size):
                if already_dones[i]:
                    padded_actions.append(torch.tensor([tokenizer.pad_token_id]))
                    padded_action_texts.append("")
                else:
                    padded_actions.append(action_token_ids_unfinished[action_ptr])
                    padded_action_texts.append(actions_unfinished[action_ptr])
                    action_ptr += 1
            # Step environment with template_ids so submit is respected
            next_state, responses, hl_reward, ll_reward, done = environment.step(
                padded_action_texts,
                system_prompt=send_instructions,
                submit_mask=submit_mask_full,
                template_ids=template_ids_full
            )
            # Update batch lists only for unfinished episodes, but limit to batch_size
            for i in range(batch_size):
                if not already_dones[i] and len(batch_states) < batch_size:
                    # Ensure state[i], next_state[i], and environment.base_state[i] are token IDs
                    def to_token_ids(x):
                        if isinstance(x, torch.Tensor):
                            return x
                        elif isinstance(x, list) and all(isinstance(xx, int) for xx in x):
                            return torch.tensor(x)
                        else:
                            # Assume string or list of strings, tokenize
                            enc = tokenizer(x, return_tensors='pt', padding=True, truncation=True)
                            return enc.input_ids[0]
                    batch_states.append(to_token_ids(state[i]))
                    batch_next_states.append(to_token_ids(next_state[i]))
                    batch_actions.append(padded_actions[i])  # token IDs
                    batch_action_texts.append(padded_action_texts[i])  # decoded string
                    # When hl_only_training=True, we don't need proper log probs for actor training
                    # but we still need to maintain the tensor structure for compatibility
                    if self.hl_only_training:
                        # Create a dummy log prob tensor with the same shape as the action
                        action_len = padded_actions[i].size(0)
                        batch_log_probs.append(torch.zeros(action_len))
                    else:
                        # For normal training, we need to compute actual log probs
                        # This is a placeholder - in a real implementation, you would compute
                        # log probs from the policy here
                        action_len = padded_actions[i].size(0)
                        batch_log_probs.append(torch.zeros(action_len))
                    batch_hl_rews.append(hl_reward[i].item())
                    batch_ll_rews.append(ll_reward[i].item())
                    batch_dones.append(done[i].item())
                    batch_base_states.append(to_token_ids(environment.base_state[i]))
                    batch_instructions.append(instructions[i])
                    already_dones[i] = done[i].item()
                
                # Store high-level transitions in replay buffer for DQN training
                if high_level_policy is not None and getattr(high_level_policy, "policy_kind", None) == "dqn":
                    for i in range(batch_size):
                        if not already_dones[i] and chosen_strategy_indices[i] is not None:
                            # Convert state to text for replay buffer
                            current_state_item = state[i]
                            if isinstance(current_state_item, list):
                                current_state_text = " ".join(str(s) for s in current_state_item if s)
                            else:
                                current_state_text = str(current_state_item)
                            
                            # Convert next state to text
                            next_state_item = next_state[i]
                            if isinstance(next_state_item, list):
                                next_state_text = " ".join(str(s) for s in next_state_item if s)
                            else:
                                next_state_text = str(next_state_item)
                            
                            # Get reward and done flag
                            r = float(hl_reward[i]) if not isinstance(hl_reward[i], torch.Tensor) else hl_reward[i].item()
                            
                            # Store transition in replay buffer
                            high_level_policy.replay_buffer.push(
                                current_state_text,
                                chosen_strategy_indices[i],
                                r,
                                next_state_text,
                                bool(done[i])
                            )
            
            state = next_state
            steps_collected += len(unfinished_indices)
            # Only reset if all episodes are done
            if all(already_dones):
                state = environment.reset()
                already_dones = [False] * batch_size
        # Prepare high-level policy data for PPO training
        batch_hl_actions = []
        batch_hl_log_probs = []
        batch_hl_entropy = []
        
        # Reconstruct high-level policy data from the collected information
        for i in range(len(batch_states)):
            if i < len(chosen_strategy_indices) and chosen_strategy_indices[i] is not None:
                batch_hl_actions.append(chosen_strategy_indices[i])
                # Find corresponding log prob and entropy
                idx_in_unfinished = None
                for j, state_idx in enumerate(unfinished_indices):
                    if state_idx == i:
                        idx_in_unfinished = j
                        break
                if idx_in_unfinished is not None and idx_in_unfinished < len(hl_log_probs_unfinished):
                    batch_hl_log_probs.append(hl_log_probs_unfinished[idx_in_unfinished])
                    batch_hl_entropy.append(hl_entropy_unfinished[idx_in_unfinished])
                else:
                    batch_hl_log_probs.append(0.0)
                    batch_hl_entropy.append(0.0)
            else:
                batch_hl_actions.append(-1)  # No action for done episodes
                batch_hl_log_probs.append(0.0)
                batch_hl_entropy.append(0.0)
        
        # Ensure all batch lists have the same length
        batch_length = len(batch_states)
        assert len(batch_next_states) == batch_length, f"batch_next_states length mismatch: {len(batch_next_states)} vs {batch_length}"
        assert len(batch_actions) == batch_length, f"batch_actions length mismatch: {len(batch_actions)} vs {batch_length}"
        assert len(batch_action_texts) == batch_length, f"batch_action_texts length mismatch: {len(batch_action_texts)} vs {batch_length}"
        assert len(batch_log_probs) == batch_length, f"batch_log_probs length mismatch: {len(batch_log_probs)} vs {batch_length}"
        assert len(batch_hl_rews) == batch_length, f"batch_hl_rews length mismatch: {len(batch_hl_rews)} vs {batch_length}"
        assert len(batch_ll_rews) == batch_length, f"batch_ll_rews length mismatch: {len(batch_ll_rews)} vs {batch_length}"
        assert len(batch_dones) == batch_length, f"batch_dones length mismatch: {len(batch_dones)} vs {batch_length}"
        assert len(batch_base_states) == batch_length, f"batch_base_states length mismatch: {len(batch_base_states)} vs {batch_length}"
        assert len(batch_instructions) == batch_length, f"batch_instructions length mismatch: {len(batch_instructions)} vs {batch_length}"
        assert len(batch_hl_actions) == batch_length, f"batch_hl_actions length mismatch: {len(batch_hl_actions)} vs {batch_length}"
        assert len(batch_hl_log_probs) == batch_length, f"batch_hl_log_probs length mismatch: {len(batch_hl_log_probs)} vs {batch_length}"
        assert len(batch_hl_entropy) == batch_length, f"batch_hl_entropy length mismatch: {len(batch_hl_entropy)} vs {batch_length}"
        
        return {
            'batch_state': batch_states,
            'batch_next_state': batch_next_states,
            'batch_acts': batch_actions,  # token IDs
            'batch_action_texts': batch_action_texts,  # decoded strings (for logging)
            'batch_log_probs': batch_log_probs,
            'batch_hl_rews': batch_hl_rews,
            'batch_ll_rews': batch_ll_rews,
            'batch_dones': batch_dones,
            'batch_base_state': batch_base_states,
            'batch_instructions': batch_instructions,
            'batch_hl_actions': batch_hl_actions,  # high-level action indices
            'batch_hl_log_probs': batch_hl_log_probs,  # high-level log probabilities
            'batch_hl_entropy': batch_hl_entropy  # high-level entropy
        }

    def train_on_batch(self, actor, sequence_critic, token_critic, batch_data, environment=None, tokenizer=None, high_level_policy=None, num_epochs=3, compute_advantage_fn=None):
        print(f"[DEBUG] MCQAgentTrainer.train_on_batch: hl_only_training={self.hl_only_training}")
        """
        Train actor and critics on a batch of data using hierarchical RL for MCQ.
        Uses batch_hl_rews for sequence_critic (for HL policy advantages) and 
        batch_ll_rews for token_critic (for low-level token training).
        
        When hl_only_training is True, skips actor (LL policy) training.
        """
        from src.agent_trainer import _batch_tokenize_and_pad_state
        def pad_batch_state(batch, tokenizer):
            if isinstance(batch[0], torch.Tensor):
                return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
            else:
                return _batch_tokenize_and_pad_state(tokenizer, batch)
        state = pad_batch_state(batch_data['batch_state'], tokenizer)
        next_state = pad_batch_state(batch_data['batch_next_state'], tokenizer)
        base_state = pad_batch_state(batch_data['batch_base_state'], tokenizer)
        # Ensure action is a padded tensor, not a list
        if isinstance(batch_data['batch_acts'][0], torch.Tensor):
            action = torch.nn.utils.rnn.pad_sequence(batch_data['batch_acts'], batch_first=True, padding_value=tokenizer.pad_token_id)
        else:
            raise ValueError('batch_acts must be a list of torch.Tensor objects (token IDs)')
        # Pad log probs to the same size before stacking, similar to how actions are padded
        if isinstance(batch_data['batch_log_probs'][0], torch.Tensor):
            old_log_probs = torch.nn.utils.rnn.pad_sequence(batch_data['batch_log_probs'], batch_first=True, padding_value=0.0)
        else:
            old_log_probs = batch_data['batch_log_probs']
        ll_rewards = torch.tensor(batch_data['batch_ll_rews'], dtype=torch.float32)
        hl_rewards = torch.tensor(batch_data['batch_hl_rews'], dtype=torch.float32)
        dones = torch.tensor(batch_data['batch_dones'], dtype=torch.bool)
        instructions = batch_data['batch_instructions']

        # Device management: move all tensors to the same device
        # When hl_only_training=True, actor might be None, so use sequence_critic or default device
        if actor is not None:
            device = next(actor.parameters()).device
        elif sequence_critic is not None:
            device = next(sequence_critic.parameters()).device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        state = state.to(device)
        next_state = next_state.to(device)
        base_state = base_state.to(device)
        
        # When hl_only_training=True, we don't need to process actor-related data
        if not self.hl_only_training:
            action = action.to(device)
            old_log_probs = old_log_probs.to(device)
            ll_rewards = ll_rewards.to(device)
        else:
            # For hl_only_training, we still need these for critics but not for actor
            action = action.to(device)
            ll_rewards = ll_rewards.to(device)
            # old_log_probs are not needed for actor training in hl_only mode
        
        hl_rewards = hl_rewards.to(device)
        dones = dones.to(device)

        # Use our dual-reward training method
        loss_dict = self.train_critics_dual_reward(
            sequence_critic=sequence_critic,
            token_critic=token_critic,
            state=state,
            action=action,
            base_state=base_state,
            hl_rewards=hl_rewards,
            ll_rewards=ll_rewards,
            next_state=next_state,
            environment=environment,
            tokenizer=tokenizer,
            instructions=instructions
        )

        # Always update the HL actor if provided (DQN-style or PPO-style)
        if high_level_policy is not None and hasattr(high_level_policy, 'policy_kind'):
            policy_kind = getattr(high_level_policy, 'policy_kind', None)
            if policy_kind == 'dqn':
                hl_metrics = high_level_policy.optimise_q_head()
                if hl_metrics:
                    loss_dict.update(hl_metrics)
            elif policy_kind == 'ppo':
                # Train PPO high-level policy using sequence critic advantages
                hl_metrics = self.train_high_level_policy_ppo(
                    high_level_policy=high_level_policy,
                    state=state,
                    action=action,
                    base_state=base_state,
                    hl_rewards=hl_rewards,
                    sequence_critic=sequence_critic,
                    environment=environment,
                    tokenizer=tokenizer,
                    batch_data=batch_data
                )
                if hl_metrics:
                    loss_dict.update(hl_metrics)
            elif policy_kind == 'hybrid':
                # Hybrid policy: check if it has a trainable PPO component
                if hasattr(high_level_policy, 'mlp_policy'):
                    mlp_policy = high_level_policy.mlp_policy
                    mlp_policy_kind = getattr(mlp_policy, "policy_kind", None)
                    
                    if mlp_policy_kind == "ppo":
                        # Train the PPO component of the hybrid policy
                        hl_metrics = self.train_high_level_policy_ppo(
                            high_level_policy=mlp_policy,
                            state=state,
                            action=action,
                            base_state=base_state,
                            hl_rewards=hl_rewards,
                            sequence_critic=sequence_critic,
                            environment=environment,
                            tokenizer=tokenizer,
                            batch_data=batch_data
                        )
                        if hl_metrics:
                            loss_dict.update(hl_metrics)
                    elif mlp_policy_kind == "dqn":
                        # Train the DQN component
                        hl_metrics = mlp_policy.optimise_q_head()
                        if hl_metrics:
                            loss_dict.update(hl_metrics)
                    else:
                        print(f"[MCQ Training] Hybrid policy MLP component type {mlp_policy_kind} not supported for training")
                else:
                    print("[MCQ Training] Hybrid policy has no trainable MLP component")

        # When HL-only training, skip ALL LL training (actor and token critic)
        if self.hl_only_training:
            print("[MCQ Training] Skipping ALL LL training (actor and token critic) - HL-only mode enabled")
            return loss_dict
        
        # Train actor (LL policy) if not in HL-only training mode
        if actor is not None:
            # Compute proper GAE advantages for actor training using token critic
            with torch.no_grad():
                # Get current value estimates from token critic
                token_input = torch.cat([state, action], dim=1)
                token_attention_mask = (token_input != tokenizer.pad_token_id).long().to(device=device)
                
                current_values = token_critic(
                    input_ids=token_input,
                    attention_mask=token_attention_mask
                )
                
                # Extract action tokens only (critic outputs [batch_size, state_len + action_len])
                state_len = state.size(1)
                current_values = current_values[:, state_len:]  # [batch_size, action_len]
                
                # Ensure correct dtype
                if current_values.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                    current_values = current_values.float()
                
                # Get next state value estimates for GAE computation
                # For MCQ environment, next_state might be the same as current state
                # or we can use the current state as a proxy
                next_input = torch.cat([next_state, action], dim=1)
                next_attention_mask = (next_input != tokenizer.pad_token_id).long().to(device=device)
                
                next_values = token_critic(
                    input_ids=next_input,
                    attention_mask=next_attention_mask
                )
                
                # Extract action tokens from next state values
                next_values = next_values[:, state_len:]  # [batch_size, action_len]
                
                # Ensure correct dtype
                if next_values.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                    next_values = next_values.float()
                
                # Create action mask for valid tokens
                action_mask = (action != tokenizer.pad_token_id).long()
                
                # Convert ll_rewards to per-token format if needed
                if ll_rewards.dim() == 1:
                    # Expand scalar rewards to per-token rewards
                    marginal_rewards = ll_rewards.unsqueeze(1).expand(-1, action.size(1))
                else:
                    # Already per-token rewards
                    marginal_rewards = ll_rewards
                
                # Compute proper GAE advantages using parent class method
                advantages, returns = self.compute_gae_per_token(
                    marginal_rewards=marginal_rewards,
                    values=current_values,
                    attention_mask=action_mask.bool(),
                    gamma=getattr(self.config.training, 'gamma', 0.99),
                    gae_lambda=getattr(self.config.training, 'gae_lambda', 0.95)
                )
                
                # Mask advantages to only valid tokens (safety check)
                advantages = advantages * action_mask.float()
            
            # Train the actor using base class method
            actor_metrics = self.train_actor(
                actor=actor,
                state=state,
                action=action,
                old_log_probs=old_log_probs,
                advantages=advantages,
                base_state=base_state,
                environment=environment
            )
            
            # Add actor metrics to loss dict
            loss_dict.update(actor_metrics)
        
        return loss_dict

    def train_critics_dual_reward(
        self,
        sequence_critic,
        token_critic,
        state,
        action,
        base_state,
        hl_rewards,
        ll_rewards,
        next_state,
        environment=None,
        tokenizer=None,
        instructions=None
    ):
        """
        Train sequence_critic and token_critic with different reward signals.
        
        Args:
            sequence_critic: Sequence-level critic (trained on HL rewards)
            token_critic: Token-level critic (trained on LL rewards)
            state, action, base_state, next_state: Standard RL tensors
            hl_rewards: High-level rewards for sequence_critic
            ll_rewards: Low-level rewards for token_critic
            environment, tokenizer, instructions: Additional context
        
        Returns:
            Dictionary with loss values
        """
        import torch
        import torch.nn.functional as F
        
        device = next(sequence_critic.parameters()).device
        model_dtype = next(sequence_critic.parameters()).dtype
        
        # ------------------------------------------------------------------
        # TRAIN SEQUENCE-LEVEL CRITIC with HL rewards
        # ------------------------------------------------------------------
        sequence_critic.train()
        
        # Compute proper Q-learning targets for sequence critic using HL rewards
        with torch.no_grad():
            if environment is not None:
                # For proper Q-learning: Q(s,a) = r_HL + γ * V(s')
                # We need to compute V(s') using the next state
                
                # Get next state value using sequence critic
                next_input = torch.cat([next_state, action], dim=1)  # This might need adjustment based on next_state format
                next_attention_mask = (next_input != tokenizer.pad_token_id).long().to(device=device)
                
                # Compute V(s') = sequence_critic(next_state)
                next_state_values = sequence_critic(
                    input_ids=next_input,
                    attention_mask=next_attention_mask,
                    requires_grad=False
                )
                
                if next_state_values.dim() > 1 and next_state_values.size(-1) == 1:
                    next_state_values = next_state_values.squeeze(-1)
                
                # Bellman equation: Q(s,a) = r_HL + γ * V(s')
                q_high_targets = hl_rewards + self.gamma * next_state_values.detach()
            else:
                # Fallback to direct HL rewards if no environment
                q_high_targets = hl_rewards
            
            q_high_targets = q_high_targets.to(device=device, dtype=model_dtype)
        
        # Forward pass for sequence critic
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
        
        # Ensure target shape matches
        if q_high_targets.dim() == 0:
            q_high_targets = q_high_targets.unsqueeze(0)
        
        # Compute and backprop sequence critic loss
        sequence_critic_loss = F.mse_loss(sequence_value_current, q_high_targets)
        
        self.sequence_critic_optimizer.zero_grad()
        sequence_critic_loss.backward()
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(sequence_critic.parameters(), self.max_grad_norm)
        
        self.sequence_critic_optimizer.step()
        
        # Clean up sequence critic computation graph
        del sequence_value_current, sequence_input, sequence_attention_mask
        torch.cuda.empty_cache()
        
        # ------------------------------------------------------------------
        # TRAIN TOKEN-LEVEL CRITIC with LL rewards (skip if HL-only training)
        # ------------------------------------------------------------------
        if not self.hl_only_training and token_critic is not None:
            token_critic.train()
            
            # Compute proper token-level targets using LL rewards  
            with torch.no_grad():
                if environment is not None:
                    # For token-level training, compute marginal rewards using LL rewards
                    # This follows the same pattern as base class but uses LL rewards
                    marginal_rewards = self.compute_marginal_rewards(
                        state, action, base_state, environment, tokenizer,
                        sequence_critic=sequence_critic, instructions=instructions
                    )
                    
                    # For MCQ, we override with LL reward structure
                    # Create a tensor that broadcasts LL rewards across action tokens
                    batch_size = ll_rewards.size(0)
                    action_length = action.size(1)
                    
                    # Use LL rewards as the base for token-level targets
                    v_low_targets = self.v_low(marginal_rewards)  # Standard computation
                    
                    # Override with LL rewards: broadcast across action tokens
                    ll_rewards_expanded = ll_rewards.unsqueeze(1).expand(batch_size, action_length)
                    v_low_targets = ll_rewards_expanded
                    
                else:
                    # Fallback: broadcast LL rewards across tokens
                    batch_size = ll_rewards.size(0) 
                    action_length = action.size(1)
                    v_low_targets = ll_rewards.unsqueeze(1).expand(batch_size, action_length)
                
                v_low_targets = v_low_targets.to(device=device, dtype=model_dtype)
            
            # Forward pass for token critic
            token_input = torch.cat([state, action], dim=1)
            token_attention_mask = (token_input != tokenizer.pad_token_id).long().to(device=device)
            
            token_value_current = token_critic(
                input_ids=token_input,
                attention_mask=token_attention_mask,
                requires_grad=True
            )
            
            if token_value_current.dim() > 1 and token_value_current.size(-1) == 1:
                token_value_current = token_value_current.squeeze(-1)
            
            # Extract action tokens only for per-token RL
            state_len = state.size(1)
            token_value_current = token_value_current[:, state_len:]  # Action tokens only
            token_value_current = token_value_current.to(dtype=model_dtype)
            
            # v_low_targets should already be properly shaped from target computation above
            
            # Compute and backprop token critic loss
            token_critic_loss = F.mse_loss(token_value_current, v_low_targets)
            
            self.token_critic_optimizer.zero_grad()
            token_critic_loss.backward()
            
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(token_critic.parameters(), self.max_grad_norm)
            
            self.token_critic_optimizer.step()
            
            # Final cleanup
            del token_value_current, token_input, token_attention_mask
            torch.cuda.empty_cache()
        else:
            # Skip token critic training in HL-only mode
            token_critic_loss = torch.tensor(0.0)
            if self.hl_only_training:
                print("[MCQ Training] Skipping token critic training - HL-only mode enabled")
        
        return {
            'sequence_critic_loss': sequence_critic_loss.item(),
            'token_critic_loss': token_critic_loss.item()
        }

    def train_high_level_policy_ppo(
        self,
        high_level_policy,
        state,
        action,
        base_state,
        hl_rewards,
        sequence_critic,
        environment=None,
        tokenizer=None,
        batch_data=None
    ):
        """
        Train high-level policy using PPO with sequence critic advantages.
        
        Args:
            high_level_policy: High-level policy model
            state: Current state tensor
            action: Action tensor
            base_state: Base state tensor
            hl_rewards: High-level rewards
            sequence_critic: Sequence-level critic for computing advantages
            environment: Environment instance (optional)
            tokenizer: Tokenizer for processing (optional)
            
        Returns:
            Dictionary of training metrics
        """
        import torch
        import torch.nn.functional as F
        
        device = next(high_level_policy.parameters()).device
        model_dtype = next(high_level_policy.parameters()).dtype
        
        # Move tensors to device
        state = state.to(device)
        action = action.to(device)
        base_state = base_state.to(device)
        hl_rewards = hl_rewards.to(device)
        
        # Verify device placement for all input tensors
        assert state.device == device, f"state device mismatch: {state.device} vs {device}"
        assert action.device == device, f"action device mismatch: {action.device} vs {device}"
        assert base_state.device == device, f"base_state device mismatch: {base_state.device} vs {device}"
        assert hl_rewards.device == device, f"hl_rewards device mismatch: {hl_rewards.device} vs {device}"
        
        # Ensure high-level policy is in training mode
        high_level_policy.train()
        
        # Verify high-level policy device placement
        policy_device = next(high_level_policy.parameters()).device
        assert policy_device == device, f"high_level_policy device mismatch: {policy_device} vs {device}"
        
        # Get stored high-level policy data from rollout
        # Ensure tensors are created on the correct device
        hl_actions = torch.tensor(batch_data['batch_hl_actions'], dtype=torch.long, device=device)
        old_log_probs = torch.tensor(batch_data['batch_hl_log_probs'], dtype=torch.float32, device=device)
        old_entropy = torch.tensor(batch_data['batch_hl_entropy'], dtype=torch.float32, device=device)
        
        # Note: hl_actions, old_log_probs, and old_entropy are from the rollout and don't need gradients
        # They are used as targets/references, not for gradient computation
        
        # Verify device placement
        assert hl_actions.device == device, f"hl_actions device mismatch: {hl_actions.device} vs {device}"
        assert old_log_probs.device == device, f"old_log_probs device mismatch: {old_log_probs.device} vs {device}"
        assert old_entropy.device == device, f"old_entropy device mismatch: {old_entropy.device} vs {device}"
        
        # Filter out invalid actions (done episodes)
        valid_mask = (hl_actions >= 0)
        if valid_mask.sum() == 0:
            print("[MCQ Training] Warning: No valid high-level actions for PPO training")
            return {}
        
        hl_actions = hl_actions[valid_mask]
        old_log_probs = old_log_probs[valid_mask]
        old_entropy = old_entropy[valid_mask]
        hl_rewards = hl_rewards[valid_mask]
        state = state[valid_mask]
        action = action[valid_mask]
        base_state = base_state[valid_mask]
        
        batch_size = state.size(0)
        
        # Convert states to text for high-level policy input (no gradients needed for text processing)
        state_texts = []
        for i in range(batch_size):
            state_item = state[i]
            if isinstance(state_item, torch.Tensor):
                # Decode token IDs to text
                if tokenizer is not None:
                    state_text = tokenizer.decode(state_item.tolist(), skip_special_tokens=True)
                else:
                    state_text = str(state_item)
            else:
                state_text = str(state_item)
            state_texts.append(state_text)
        
        # Tokenize state texts (no gradients needed for tokenization)
        state_encodings = tokenizer(
            state_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        # Move to device after tokenization
        state_encodings = state_encodings.to(device)
        
        # Verify tokenizer output device placement
        assert state_encodings.input_ids.device == device, f"tokenizer input_ids device mismatch: {state_encodings.input_ids.device} vs {device}"
        assert state_encodings.attention_mask.device == device, f"tokenizer attention_mask device mismatch: {state_encodings.attention_mask.device} vs {device}"
        
        # Get current policy distribution for the selected actions (WITH GRADIENTS ENABLED)
        # This is crucial for PPO training - we need gradients for the loss computation
        strategy_logits = high_level_policy.forward(
            state_encodings.input_ids,
            state_encodings.attention_mask
        )
        
        # Convert to probabilities
        strategy_logits_clamped = torch.clamp(strategy_logits, min=-5, max=5)
        strategy_probs = F.softmax(strategy_logits_clamped, dim=-1)
        
        # Get log probs for the actions that were actually taken
        action_dist = torch.distributions.Categorical(strategy_probs)
        current_log_probs = action_dist.log_prob(hl_actions)
        
        # Compute entropy
        entropy = action_dist.entropy()
        
        # Verify that current_log_probs and entropy have gradients enabled
        assert current_log_probs.requires_grad, "current_log_probs must require gradients for PPO training"
        assert entropy.requires_grad, "entropy must require gradients for PPO training"
        
        # Compute advantages using sequence critic
        with torch.no_grad():
            # Get current value estimates
            sequence_input = torch.cat([state, action], dim=1)
            sequence_attention_mask = (sequence_input != tokenizer.pad_token_id).long().to(device=device)
            
            # Verify device placement for sequence critic inputs
            assert sequence_input.device == device, f"sequence_input device mismatch: {sequence_input.device} vs {device}"
            assert sequence_attention_mask.device == device, f"sequence_attention_mask device mismatch: {sequence_attention_mask.device} vs {device}"
            
            current_values = sequence_critic(
                input_ids=sequence_input,
                attention_mask=sequence_attention_mask
            )
            
            if current_values.dim() > 1 and current_values.size(-1) == 1:
                current_values = current_values.squeeze(-1)
            
            # Ensure correct dtype
            if current_values.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                current_values = current_values.float()
            
            # For high-level policy, we can use sequence-level advantages
            # Compute advantages as: A = R - V(s)
            advantages = hl_rewards - current_values
            
            # Normalize advantages
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training logic using stored old log probabilities
        # Compute ratio: exp(current_log_probs - old_log_probs)
        log_ratio = current_log_probs - old_log_probs
        
        if self.config.training.debug_mode:
            # Debug: Verify gradient requirements for ratio computation
            print(f"[PPO Debug] current_log_probs.requires_grad: {current_log_probs.requires_grad}")
            print(f"[PPO Debug] old_log_probs.requires_grad: {old_log_probs.requires_grad}")
            print(f"[PPO Debug] log_ratio.requires_grad: {log_ratio.requires_grad}")
        
        # Clamp log ratio to prevent numerical instability
        log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
        ratio = torch.exp(log_ratio)
        
        if self.config.training.debug_mode:
            print(f"[PPO Debug] ratio.requires_grad: {ratio.requires_grad}")
        
        # PPO surrogate loss
        clip_ratio = getattr(self, 'clip_ratio', 0.2)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # Actor loss (negative because we want to maximize)
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Entropy regularization
        entropy_coef = getattr(self, 'entropy_coef', 0.01)
        entropy_loss = -entropy.mean() * entropy_coef
        
        # KL divergence penalty (optional)
        kl_coef = getattr(self, 'kl_coef', 0.0)
        kl_div = (old_log_probs - current_log_probs).mean()
        kl_loss = kl_div * kl_coef
        
        # Total loss
        total_loss = actor_loss + entropy_loss + kl_loss
        
        # Debug: Verify gradient requirements before backward pass
        if self.config.training.debug_mode:
            print(f"[PPO Debug] total_loss.requires_grad: {total_loss.requires_grad}")
            print(f"[PPO Debug] total_loss.grad_fn: {total_loss.grad_fn}")
            print(f"[PPO Debug] actor_loss.requires_grad: {actor_loss.requires_grad}")
            print(f"[PPO Debug] entropy_loss.requires_grad: {entropy_loss.requires_grad}")
            print(f"[PPO Debug] kl_loss.requires_grad: {kl_loss.requires_grad}")
        
        # Update high-level policy
        if hasattr(self, 'high_level_optimizer') and self.high_level_optimizer is not None:
            self.high_level_optimizer.zero_grad()
            total_loss.backward()
            
            # Apply gradient clipping
            max_grad_norm = getattr(self, 'max_grad_norm', 0.5)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(high_level_policy.parameters(), max_grad_norm)
            
            self.high_level_optimizer.step()
        
        return {
            'high_level_policy_loss': actor_loss.item(),
            'high_level_entropy_loss': entropy_loss.item(),
            'high_level_kl_loss': kl_loss.item(),
            'high_level_total_loss': total_loss.item(),
            'high_level_advantages_mean': advantages.mean().item(),
            'high_level_advantages_std': advantages.std().item(),
            'high_level_ratio_mean': ratio.mean().item(),
            'high_level_ratio_min': ratio.min().item(),
            'high_level_ratio_max': ratio.max().item(),
            'high_level_kl_div': kl_div.item()
        }

    def train(
        self,
        policy,
        sequence_critic,
        token_critic,
        environment,
        tokenizer,
        num_episodes,
        rollout_steps,
        save_freq=5,
        output_dir="outputs",
        save_checkpoint_fn=None,
        high_level_policy=None,
        templates_path="src/configs/templates.json",
        episode_offset=0
    ):
        """
        Main training loop for MCQ agent. Uses MCQ-specific collect_rollout and reward structure.
        """
        import os, time, datetime, torch
        save_checkpoint_fn = save_checkpoint_fn or self._default_save_checkpoint
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_output_dir = os.path.join(output_dir, run_id)
        run_log_dir = os.path.join("logs", run_id)
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(run_log_dir, exist_ok=True)
        print(f"Created run-specific directories:")
        print(f"  Output directory: {run_output_dir}")
        print(f"  Log directory: {run_log_dir}")
        debug_mode = self.config.training.debug_mode
        if debug_mode:
            print("DEBUG MODE")
            self._reset_gpu_memory_stats()
        episode_rewards = []
        warmup_batch_size = getattr(self.config.training, 'warmup_batch_size', 8)
        warmup_rollout_steps = getattr(self.config.training, 'warmup_rollout_steps', warmup_batch_size)
        warmup_max_new_tokens = getattr(self.config.environment, 'warmup_max_new_tokens', getattr(environment, 'max_new_tokens', None))
        original_batch_size = environment.batch_size if hasattr(environment, 'batch_size') else None
        original_rollout_steps = rollout_steps
        original_max_new_tokens = getattr(environment, 'max_new_tokens', None)
        # High-level epsilon exploration schedule (anneals like low-level)
        hl_epsilon_start = getattr(self.config.training, 'hl_epsilon_start', 0.1)
        hl_epsilon_end = getattr(self.config.training, 'hl_epsilon_end', 0.0)
        hl_epsilon_decay_episodes = getattr(self.config.training, 'hl_epsilon_decay_episodes', 100)
        for episode in range(num_episodes):
            actual_episode = episode + episode_offset
            self.apply_learning_rate_annealing(actual_episode)
            use_warmup = getattr(self.config.training, 'use_warmup', True)
            if episode == 0 and episode_offset == 0 and use_warmup:
                print(f"WARMUP EPISODE: Using batch_size={warmup_batch_size}, rollout_steps={warmup_rollout_steps}, max_new_tokens={warmup_max_new_tokens}")
                if hasattr(environment, 'set_batch_size'):
                    environment.set_batch_size(warmup_batch_size)
                if warmup_max_new_tokens is not None and hasattr(environment, 'max_new_tokens'):
                    environment.max_new_tokens = warmup_max_new_tokens
                current_rollout_steps = warmup_rollout_steps
            else:
                if episode == 1 and use_warmup:
                    print(f"SWITCHING TO FULL BATCH: batch_size={original_batch_size}, rollout_steps={original_rollout_steps}, max_new_tokens={original_max_new_tokens}")
                    if hasattr(environment, 'set_batch_size') and original_batch_size:
                        environment.set_batch_size(original_batch_size)
                    if original_max_new_tokens is not None and hasattr(environment, 'max_new_tokens'):
                        environment.max_new_tokens = original_max_new_tokens
                current_rollout_steps = original_rollout_steps

            start_time = time.time()
            if debug_mode:
                self._log_gpu_memory("Before rollout")
            rollout_data = self.collect_rollout(
                policy=policy,
                environment=environment,
                tokenizer=tokenizer,
                rollout_steps=current_rollout_steps,
                config=self.config,
                high_level_policy=high_level_policy,
                templates_path=templates_path,
                episode=actual_episode
            )
            if debug_mode:
                self._log_gpu_memory("After rollout")
            print(f"Rollout time: {time.time() - start_time:.2f}s")
            
            # Debug mode: stop after first rollout to verify formatting
            if debug_mode:
                print("=" * 80)
                print("DEBUG MODE: STOPPING AFTER 1 ROLLOUT")
                print("=" * 80)
                print(f"Collected {len(rollout_data['batch_state'])} timesteps")
                print(f"Action texts (first 5): {rollout_data['batch_action_texts'][:5]}")
                print(f"Instructions (first 5): {rollout_data['batch_instructions'][:5]}")
                return policy, sequence_critic, token_critic, [], run_id
            if debug_mode:
                self._log_gpu_memory("Before train")
            train_start_time = time.time()
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
            # Calculate episode statistics using HL reward (for MCQ)
            hl_rewards = torch.tensor(rollout_data['batch_hl_rews'], dtype=torch.float32)
            mean_reward = hl_rewards.mean().item()
            max_reward = hl_rewards.max().item()
            min_reward = hl_rewards.min().item()
            batch_variance = hl_rewards.var().item()
            batch_std = hl_rewards.std().item()
            num_timesteps = len(rollout_data['batch_state'])
            episode_rewards.append(mean_reward)
            if hasattr(self, 'log_dict'):
                self.log_dict.setdefault('episode_rewards', []).append(mean_reward)
            max_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            metrics['max_gpu_allocated_gb'] = max_gpu_gb
            # AGGRESSIVE MEMORY CLEANUP
            del rollout_data
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            print("=" * 80)
            print(f"EPISODE {actual_episode+1:3d} SUMMARY")
            print("=" * 80)
            print(f"Total time: {time.time() - start_time:.2f}s")
            print(f"Rollout Stats:")
            print(f"  • Timesteps: {num_timesteps}")
            print(f"  • Mean HL Reward: {mean_reward:.4f}")
            print(f"  • Max HL Reward: {max_reward:.4f}")
            print(f"  • Min HL Reward: {min_reward:.4f}")
            print(f"  • Std HL Reward: {batch_std:.4f}")
            print(f"  • Variance HL Reward: {batch_variance:.4f}")
            print(f"  • Max GPU Allocated (GB): {max_gpu_gb:.2f}")
            print("=" * 80)
        return policy, sequence_critic, token_critic, episode_rewards, run_id

    def _reset_gpu_memory_stats(self) -> None:
        """Reset GPU memory peak statistics.
        
        Call this at the start of training to get clean peak measurements.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print("[GPU MEMORY] Reset peak memory statistics")

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
        if not torch.cuda.is_available():
            raise RuntimeError("_log_gpu_memory() called but CUDA is not available")

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 1:
            # Single GPU case - keep the original simple output
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            print(
                f"[GPU MEMORY] {prefix} | allocated: {allocated_gb:.2f} GB | "
                f"peak: {peak_gb:.2f} GB | reserved: {reserved_gb:.2f} GB",
                flush=flush
            )
        else:
            # Multi-GPU case - show stats for each device
            print(f"[GPU MEMORY] {prefix} | Multi-GPU ({num_gpus} devices):")
            for device_id in range(num_gpus):
                with torch.cuda.device(device_id):
                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    total_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                    
                    print(
                        f"  GPU {device_id}: allocated: {allocated_gb:.2f} GB | "
                        f"peak: {peak_gb:.2f} GB | reserved: {reserved_gb:.2f} GB | "
                        f"total: {total_gb:.2f} GB",
                        flush=flush
                    )

        # Only reset peak stats if explicitly requested
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()

    def _generate_with_chat_contexts(self, policy, chat_contexts, base_state, tokenizer, config):
        """
        Generate responses using pre-built chat contexts, bypassing the policy's default chat construction.
        This allows us to have full control over the system prompt and conversation structure.
        """
        import torch
        
        # Get device from policy model
        device = next(policy.model.parameters()).device
        
        # Initialize all variables at the beginning to avoid UnboundLocalError
        max_tokens = getattr(config.training, 'max_new_tokens', 100) if hasattr(config.training, 'training') else 100
        temperature = 0.7
        do_sample = True
        top_p = 0.9
        has_submit = False
        
        # Tokenize the chat contexts directly
        prompted_encoding = tokenizer.apply_chat_template(
            chat_contexts,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_dict=True
        ).to(device)

        batched_input_ids = prompted_encoding.input_ids
        batched_attention = prompted_encoding.attention_mask
        
        # Use deterministic generation when HL-only training (frozen LL policy)
        if self.hl_only_training:
            temperature = 0.0  # Deterministic generation
            do_sample = False
            top_p = None
        else:
            # Check if any of the chat contexts are for submit actions
            has_submit = any(
                any(msg.get('content', '').startswith('FINAL ANSWER REQUIRED') 
                    for msg in chat_context if msg.get('role') == 'user')
                for chat_context in chat_contexts
            )
            
            if has_submit:
                # For submit actions, use more deterministic generation to get precise answers
                temperature = 0.0  # Completely deterministic
                do_sample = False  # No sampling
                top_p = None
                max_tokens = 5  # Limit to just a few tokens for submit actions
            else:
                # Normal generation for reasoning steps
                temperature = 0.7
                do_sample = True
                top_p = 0.9
            
        debug_mode = getattr(config.training, 'debug_mode', False) if hasattr(config, 'training') else False
        
        if debug_mode:
            print(f"LL generation parameters - max_tokens: {max_tokens}, temperature: {temperature}, hl_only_training: {self.hl_only_training}")
            if has_submit:
                top_p_str = str(top_p) if top_p is not None else "None"
                print(f"[SUBMIT ACTION] Using deterministic generation: temperature={temperature}, top_p={top_p_str}, max_tokens={max_tokens}")
            else:
                top_p_str = str(top_p) if top_p is not None else "None"
                print(f"[NORMAL ACTION] Using standard generation: temperature={temperature}, top_p={top_p_str}, max_tokens={max_tokens}")
        
        # Generate responses
        with torch.no_grad():
            generation_kwargs = {
                'input_ids': batched_input_ids,
                'attention_mask': batched_attention,
                'max_new_tokens': max_tokens,
                'do_sample': do_sample,
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'use_cache': True,
                'num_return_sequences': 1,
                'repetition_penalty': 1.1,
                # Additional safety for submit actions
                'early_stopping': has_submit,
                'length_penalty': 0.1 if has_submit else 1.0,  # Prefer shorter responses for submit
                'no_repeat_ngram_size': 3 if has_submit else 0  # Prevent repetition in submit
            }
            
            # Only add temperature and top_p if they're valid
            if temperature > 0:
                generation_kwargs['temperature'] = temperature
            if top_p is not None:
                generation_kwargs['top_p'] = top_p
            
            outputs = policy.model.generate(**generation_kwargs)
        
        # Decode responses
        actions = []
        action_token_ids = []
        input_lengths = batched_input_ids.shape[1]
        
        for i, output in enumerate(outputs):
            # Extract only the generated part (exclude input tokens)
            generated_tokens = output[input_lengths:]
            generated_text = tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Post-process submit actions to ensure single letter format
            if debug_mode:
                # Check if this was a submit action by looking at the chat context
                is_submit_action = any(
                    any(msg.get('content', '').startswith('FINAL ANSWER REQUIRED') 
                        for msg in chat_context if msg.get('role') == 'user')
                    for chat_context in [chat_contexts[i]]
                )
                
                if is_submit_action:
                    original_text = generated_text
                    # Clean up the response to extract only A, B, C, or D
                    cleaned_text = self._clean_submit_response(generated_text)
                    if cleaned_text != generated_text:
                        print(f"[SUBMIT CLEANUP] Original: '{original_text}' -> Cleaned: '{cleaned_text}'")
                    generated_text = cleaned_text
            
            actions.append(generated_text)
            action_token_ids.append(generated_tokens)
            
            if debug_mode:
                print(f"LL response {i}: '{generated_text}'")
        
        return actions, action_token_ids

    def _clean_submit_response(self, response: str) -> str:
        """
        Clean up submit responses to ensure they contain only a single letter A, B, C, or D.
        
        Args:
            response: Raw response from LL agent
            
        Returns:
            Cleaned response with only the single letter
        """
        # Remove all whitespace and convert to uppercase
        cleaned = response.strip().upper()
        
        # Look for single letters A, B, C, or D
        for char in cleaned:
            if char in "ABCD":
                return char
        
        # If no single letter found, try to extract from common patterns
        import re
        
        # Pattern 1: "The answer is A" -> extract A
        match = re.search(r'answer\s+is\s+([ABCD])', cleaned, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 2: "I choose B" -> extract B
        match = re.search(r'choose\s+([ABCD])', cleaned, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 3: "Option C" -> extract C
        match = re.search(r'option\s+([ABCD])', cleaned, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 4: Just find any A, B, C, or D in the text
        match = re.search(r'([ABCD])', cleaned)
        if match:
            return match.group(1)
        
        # If all else fails, return the first character if it's valid, otherwise 'A' as fallback
        if cleaned and cleaned[0] in "ABCD":
            return cleaned[0]
        
        # Last resort: return 'A' and log warning
        print(f"[WARNING] Could not extract valid answer from response: '{response}'. Using fallback 'A'.")
        return 'A'

    def compute_skywork_reward_score(
        self,
        responses: List[str],
        questions: Optional[List[str]] = None,
        reference_state: Optional[List[str]] = None,
        skywork_model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        skywork_tokenizer: Optional[Any] = None,
        skywork_model: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Compute reward scores using the Skywork reward model for token-level critic training.
        
        This method follows the same pattern as Llamaguard usage in the environment,
        using the Skywork reward model to evaluate response quality.
        
        Args:
            responses: List of responses to evaluate
            questions: Questions that prompted the responses (optional)
            reference_state: Original state for reference (optional)
            skywork_model_name: Name of the Skywork reward model to use
            skywork_tokenizer: Pre-loaded tokenizer (optional)
            skywork_model: Pre-loaded model (optional)
            
        Returns:
            Reward scores tensor [batch_size]
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Check if we have pre-loaded models
        if skywork_model is None or skywork_tokenizer is None:
            # Load the Skywork reward model and tokenizer
            print(f"Loading Skywork reward model: {skywork_model_name}")
            
            # Determine device placement
            if torch.cuda.is_available():
                device = "cuda:0"
                if torch.cuda.device_count() > 1:
                    # Use GPU 1 for inference models (same as guard model strategy)
                    device = "cuda:1"
            else:
                device = "cpu"
            
            # Load model with bf16 precision as specified in the notepad
            skywork_model = AutoModelForSequenceClassification.from_pretrained(
                skywork_model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2",
                num_labels=1,
            )
            
            skywork_tokenizer = AutoTokenizer.from_pretrained(skywork_model_name)
            
            # Add padding token if not present
            if skywork_tokenizer.pad_token is None:
                skywork_tokenizer.pad_token = skywork_tokenizer.eos_token
        
        # Ensure model and tokenizer are on the same device
        device = next(skywork_model.parameters()).device
        skywork_tokenizer = skywork_tokenizer.to(device)
        
        # Process responses in minibatches to avoid OOM
        all_scores = []
        chunk_size = getattr(self.config.training, 'skywork_eval_batch_size', 4)  # Conservative batch size
        
        if getattr(self.config.training, 'debug_mode', False):
            print(f"Processing {len(responses)} responses with Skywork reward model in minibatches of size {chunk_size}")
        
        for start in range(0, len(responses), chunk_size):
            end = min(start + chunk_size, len(responses))
            batch_responses = responses[start:end]
            
            if getattr(self.config.training, 'debug_mode', False):
                print(f"Processing Skywork minibatch {start//chunk_size + 1}: responses {start} to {end-1}")
            
            # Prepare conversations for this minibatch
            # Note: Skywork authors specifically note "Do not include system prompts when using chat templates"
            batch_conversations = []
            for response in batch_responses:
                # For token-level critic, we evaluate the response quality
                # If we have questions, use them as context; otherwise just evaluate response
                if questions and start + len(batch_conversations) < len(questions):
                    question = questions[start + len(batch_conversations)]
                    # Format as conversation: [{"role": "user", "content": question}, {"role": "assistant", "content": response}]
                    conv = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response}
                    ]
                else:
                    # If no question context, just evaluate the response quality
                    conv = [
                        {"role": "user", "content": "Evaluate this response:"},
                        {"role": "assistant", "content": response}
                    ]
                
                batch_conversations.append(conv)
            
            # Format and tokenize conversations using Skywork's chat template
            batch_formatted = []
            for conv in batch_conversations:
                formatted = skywork_tokenizer.apply_chat_template(conv, tokenize=False)
                
                # Remove potential duplicate BOS token as specified in the notepad
                if skywork_tokenizer.bos_token is not None and formatted.startswith(skywork_tokenizer.bos_token):
                    formatted = formatted[len(skywork_tokenizer.bos_token):]
                
                batch_formatted.append(formatted)
            
            # Tokenize the batch
            batch_tokenized = skywork_tokenizer(
                batch_formatted,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048  # Reasonable max length for reward evaluation
            ).to(device)
            
            # Get reward scores from Skywork model
            with torch.no_grad():
                outputs = skywork_model(**batch_tokenized)
                batch_scores = outputs.logits.squeeze(-1)  # [batch_size]
                
                if batch_scores.dim() == 0:
                    batch_scores = batch_scores.unsqueeze(0)
            
            # Store scores (move to CPU to save GPU memory)
            all_scores.append(batch_scores.cpu())
            
            # Clean up batch tensors
            del batch_tokenized, outputs, batch_scores
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate all minibatch scores
        reward_scores = torch.cat(all_scores, dim=0)
        
        # Validate final output
        if reward_scores.shape[0] != len(responses):
            raise ValueError(f"Output shape mismatch: expected {len(responses)} scores, got {reward_scores.shape[0]}")
        
        if getattr(self.config.training, 'debug_mode', False):
            print(f"Final Skywork reward scores shape: {reward_scores.shape}")
            print(f"Score range: {reward_scores.min().item():.4f} to {reward_scores.max().item():.4f}")
            print(f"Mean score: {reward_scores.mean().item():.4f}")
        
        # Return reward scores
        # Note: Skywork scores are typically higher = better quality
        # For consistency with our reward framework, we may want to normalize or transform these
        return reward_scores
    
    def compute_marginal_rewards_with_skywork(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        base_state: torch.Tensor,
        environment: Any,
        tokenizer: Any,
        sequence_critic: Optional[Any] = None,
        top_k_tokens: Optional[int] = None,
        pairwise_masking: Optional[bool] = None,
        instructions: Optional[List[str]] = None,
        skywork_model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    ) -> torch.Tensor:
        """
        Compute marginal rewards using Skywork reward model for token-level critic training.
        
        This is a specialized version of compute_marginal_rewards that uses the Skywork
        reward model instead of the environment's safety scoring for the low-level critic.
        
        Args:
            state: Current state tensor [batch_size, seq_len]
            action: Action sequence tensor [batch_size, action_len]
            base_state: Base state tensor [batch_size, seq_len]
            environment: Environment for high-level reward computation
            tokenizer: Tokenizer for text processing
            sequence_critic: Sequence critic to get attention weights from (optional)
            top_k_tokens: Number of top attention tokens to compute marginal rewards for
            pairwise_masking: Whether to compute pairwise masking
            instructions: Optional instruction strings for instruction-following reward
            skywork_model_name: Name of the Skywork reward model to use
            
        Returns:
            Marginal rewards tensor [batch_size, action_len]
        """
        # ------------------------------------------------------------------
        # Naïve back-to-front γ-decayed per-token reward (ablation)
        # ------------------------------------------------------------------
        if getattr(self.config.training, "naive_decayed_token_reward", False):
            action_attention_mask = (action != tokenizer.pad_token_id).long()
            self.swap_models_to_cpu()
            try:
                # Use Skywork reward model for sequence-level rewards
                action_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
                seq_rewards = self.compute_skywork_reward_score(
                    responses=action_texts,
                    skywork_model_name=skywork_model_name
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
        from src.general.mc_multiturn_env import MultiTurnMultipleChoiceEnv
        if isinstance(environment, MultiTurnMultipleChoiceEnv):
            use_q_as_reward = True

        batch_size, action_len = action.shape
        
        # Create full attention mask for actions (all tokens are valid initially)
        action_attention_mask = (action != tokenizer.pad_token_id).long()
        
        # MODEL SWAPPING: Move agent models to CPU before environment calls
        self.swap_models_to_cpu()
        
        try:
            if use_q_as_reward:
                # Use current sequence_critic predictions as proxy Q values
                if sequence_critic is not None:
                    sequence_critic.to(self.device)
                    
                with torch.no_grad():
                    sequence_input_full = torch.cat([state, action], dim=1)
                    q_full = sequence_critic(
                        input_ids=sequence_input_full,
                        attention_mask=action_attention_mask if sequence_input_full.size(1)==action.size(1) else torch.cat([(state!=tokenizer.pad_token_id).long(), action_attention_mask], dim=1),
                    )
                    if q_full.dim() > 1 and q_full.size(-1) == 1:
                        q_full = q_full.squeeze(-1)
                j_unmasked = q_full.to(action.device)  # J(unmasked) = Q_high(s, a_unmasked)
            else:
                # Compute J(unmasked) using Skywork reward model for low-level critic
                # Get target responses for unmasked action
                target_responses_full = environment.get_target_responses_from_tokens(
                    action, action_attention_mask, tokenizer
                )

                # Use Skywork reward model to score target responses
                r_s_prime_unmasked = self.compute_skywork_reward_score(
                    responses=target_responses_full,
                    skywork_model_name=skywork_model_name
                ).to(action.device)

                # For the action scoring, we can use either Skywork or environment safety
                # Use Skywork for consistency with low-level critic training
                action_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
                r_a_unmasked = self.compute_skywork_reward_score(
                    responses=action_texts,
                    skywork_model_name=skywork_model_name
                ).to(action.device)

                # J(unmasked) = R(s'_unmasked) - R(a_unmasked) using Skywork rewards
                j_unmasked = r_s_prime_unmasked - r_a_unmasked
                j_unmasked = torch.nan_to_num(j_unmasked, nan=0.0, posinf=1e4, neginf=-1e4).to(action.device)
            
            # Use attention-guided token selection if sequence critic is provided
            if sequence_critic is not None:
                # Ensure sequence_critic is on GPU since it's passed as parameter
                sequence_critic.to(self.device)
                
                # Get last hidden states from sequence critic to derive token importance cheaply
                with torch.no_grad():
                    _, last_hidden = sequence_critic(
                        input_ids=action,
                        attention_mask=action_attention_mask,
                        return_last_hidden=True
                    )
                
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
                max_token_positions = getattr(self.config.training, 'max_marginal_token_positions', 10)
                if len(valid_token_positions) > max_token_positions:
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
                        if sequence_critic is not None:
                            sequence_critic.to(self.device)
                            
                        chunk_scores_list = []
                        with torch.no_grad():
                            for masked_attention in chunk_attentions:
                                seq_input = torch.cat([state, action], dim=1)
                                # Build combined attention mask: state pad + masked attention
                                state_attention = (state != tokenizer.pad_token_id).long()
                                full_attention = torch.cat([state_attention, masked_attention], dim=1)
                                q_val = sequence_critic(
                                    input_ids=seq_input,
                                    attention_mask=full_attention
                                )
                                if q_val.dim() > 1 and q_val.size(-1) == 1:
                                    q_val = q_val.squeeze(-1)
                                chunk_scores_list.append(q_val)
                                
                                # Clean up intermediate tensors
                                del seq_input, state_attention, full_attention
                                
                        chunk_scores = torch.stack(chunk_scores_list, dim=1).to(device=action.device, dtype=j_unmasked.dtype)
                        
                        # Clean up list of tensors
                        del chunk_scores_list
                        torch.cuda.empty_cache()
                        
                    else:
                        # Compute J(masked) using Skywork reward model for each masked version
                        chunk_masked_texts = []
                        chunk_masked_actions = []
                        for masked_attention in chunk_attentions:
                            masked_texts = environment._batch_decode_with_attention_mask(
                                action, masked_attention, tokenizer
                            )
                            chunk_masked_texts.extend(masked_texts)
                            chunk_masked_actions.extend(masked_texts)

                        # Get target responses for masked actions
                        chunk_target_responses = environment.get_target_responses(chunk_masked_texts)

                        # Score using Skywork reward model
                        r_s_prime_masked = self.compute_skywork_reward_score(
                            responses=chunk_target_responses,
                            skywork_model_name=skywork_model_name
                        ).to(action.device)

                        r_a_masked = self.compute_skywork_reward_score(
                            responses=chunk_masked_actions,
                            skywork_model_name=skywork_model_name
                        ).to(action.device)

                        # J(masked) = R(s'_masked) - R(a_masked) using Skywork rewards
                        chunk_scores = r_s_prime_masked - r_a_masked
                        chunk_scores = chunk_scores.to(device=action.device, dtype=j_unmasked.dtype)
                        
                        # Clean up large tensors
                        del chunk_masked_texts, chunk_target_responses, chunk_masked_actions
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
                avg_contrib = sum_tensor / max(count, 1)  # avoid division by zero
                # Ensure avg_contrib is on the same device as marginal_rewards
                avg_contrib = avg_contrib.to(device=action.device, dtype=j_unmasked.dtype)
                marginal_rewards[:, token_pos] += avg_contrib
                
                # Clean up the sum tensor to free memory
                del sum_tensor

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # [AUGMENTATION] Add instruction-following reward to marginal rewards
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            use_task_reward = getattr(self.config.training, 'use_task_reward', True)
            if instructions is not None and use_task_reward:
                # Compute similarity between actions and instructions
                utter_texts = self._batch_decode_with_timeout(tokenizer, action, skip_special_tokens=True)
                safe_instr = [instr for instr in instructions]
                
                sim_scores = environment._get_embed_similarity_batched(
                    safe_instr,
                    utter_texts,
                ).to(device=marginal_rewards.device, dtype=marginal_rewards.dtype)
                
                # Distribute instruction-following reward across valid tokens
                action_attention_mask = (action != tokenizer.pad_token_id).long()
                for batch_idx in range(batch_size):
                    valid_positions = action_attention_mask[batch_idx].nonzero(as_tuple=True)[0]
                    if len(valid_positions) > 0:
                        # Add instruction-following reward to all valid token positions
                        marginal_rewards[batch_idx, valid_positions] += sim_scores[batch_idx] / len(valid_positions)
            
            # Clean up pairwise accumulation structures
            del pairwise_sum, pairwise_count
            
            return marginal_rewards
            
        finally:
            # MODEL SWAPPING: Move agent models back to GPU after environment calls
            self.swap_models_to_gpu()

class SingleAgentArcherTrainer(MCQAgentTrainer):
    """
    Single-Agent Archer Trainer for MCQ environments.
    
    This implements the single-agent archer ablation where:
    - One agent has both high-level and low-level critics
    - High-level critic: Q function (same as SequenceLevelValueCritic)
    - Low-level critic: V function (TokenLevelValueCriticV) that predicts expected value from sequence
    - Policy objective: maximize J = Σ(A_high - V_low(s[0:i-1]))
    
    Key differences from MCQAgentTrainer:
    - Single agent instead of dual agents
    - Low-level critic is V function, not Q function
    - Training objective combines both critics in single update
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize SingleAgentArcherTrainer.
        
        Args:
            *args, **kwargs: Arguments passed to parent MCQAgentTrainer
        """
        super().__init__(*args, **kwargs)
        self.single_agent_mode = True
        
    def train_on_batch(self, actor, sequence_critic, token_critic, batch_data, environment=None, tokenizer=None, high_level_policy=None, num_epochs=3, compute_advantage_fn=None):
        """
        Train single agent with both high-level and low-level critics.
        
        Uses the objective: J = Σ(A_high - V_low(s[0:i-1]))
        where A_high is derived from sequence_critic and V_low from token_critic.
        """
        print(f"[DEBUG] SingleAgentArcherTrainer.train_on_batch: single_agent_mode={self.single_agent_mode}")
        
        from src.agent_trainer import _batch_tokenize_and_pad_state
        
        def pad_batch_state(batch, tokenizer):
            if isinstance(batch[0], torch.Tensor):
                return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
            else:
                return _batch_tokenize_and_pad_state(tokenizer, batch)
        
        # Prepare batch data
        state = pad_batch_state(batch_data['batch_state'], tokenizer)
        next_state = pad_batch_state(batch_data['batch_next_state'], tokenizer)
        base_state = pad_batch_state(batch_data['batch_base_state'], tokenizer)
        
        # Ensure action is a padded tensor
        if isinstance(batch_data['batch_acts'][0], torch.Tensor):
            action = torch.nn.utils.rnn.pad_sequence(batch_data['batch_acts'], batch_first=True, padding_value=tokenizer.pad_token_id)
        else:
            raise ValueError('batch_acts must be a list of torch.Tensor objects (token IDs)')
        
        # Pad log probs
        if isinstance(batch_data['batch_log_probs'][0], torch.Tensor):
            old_log_probs = torch.nn.utils.rnn.pad_sequence(batch_data['batch_log_probs'], batch_first=True, padding_value=0.0)
        else:
            old_log_probs = batch_data['batch_log_probs']
        
        # Get rewards and other data
        ll_rewards = torch.tensor(batch_data['batch_ll_rews'], dtype=torch.float32)
        hl_rewards = torch.tensor(batch_data['batch_hl_rews'], dtype=torch.float32)
        dones = torch.tensor(batch_data['batch_dones'], dtype=torch.bool)
        instructions = batch_data['batch_instructions']
        
        # Device management
        if actor is not None:
            device = next(actor.parameters()).device
        elif sequence_critic is not None:
            device = next(sequence_critic.parameters()).device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move tensors to device
        state = state.to(device)
        next_state = next_state.to(device)
        base_state = base_state.to(device)
        action = action.to(device)
        old_log_probs = old_log_probs.to(device)
        ll_rewards = ll_rewards.to(device)
        hl_rewards = hl_rewards.to(device)
        dones = dones.to(device)
        
        # Train critics with dual reward structure
        critic_losses = self.train_critics_single_agent(
            sequence_critic=sequence_critic,
            token_critic=token_critic,
            state=state,
            action=action,
            base_state=base_state,
            hl_rewards=hl_rewards,
            ll_rewards=ll_rewards,
            next_state=next_state,
            environment=environment,
            tokenizer=tokenizer,
            instructions=instructions
        )
        
        # Train actor using single-agent objective
        if actor is not None:
            actor_metrics = self.train_actor_single_agent(
                actor=actor,
                state=state,
                action=action,
                old_log_probs=old_log_probs,
                sequence_critic=sequence_critic,
                token_critic=token_critic,
                hl_rewards=hl_rewards,
                base_state=base_state,
                environment=environment,
                tokenizer=tokenizer
            )
            critic_losses.update(actor_metrics)
        
        # Handle high-level policy training if provided
        if high_level_policy is not None and hasattr(high_level_policy, 'policy_kind'):
            policy_kind = getattr(high_level_policy, 'policy_kind', None)
            if policy_kind == 'dqn':
                hl_metrics = high_level_policy.optimise_q_head()
                if hl_metrics:
                    critic_losses.update(hl_metrics)
            elif policy_kind == 'ppo':
                hl_metrics = self.train_high_level_policy_ppo(
                    high_level_policy=high_level_policy,
                    state=state,
                    action=action,
                    base_state=base_state,
                    hl_rewards=hl_rewards,
                    sequence_critic=sequence_critic,
                    environment=environment,
                    tokenizer=tokenizer,
                    batch_data=batch_data
                )
                if hl_metrics:
                    critic_losses.update(hl_metrics)
        
        return critic_losses
    
    def train_critics_single_agent(
        self,
        sequence_critic,
        token_critic,
        state,
        action,
        base_state,
        hl_rewards,
        ll_rewards,
        next_state,
        environment=None,
        tokenizer=None,
        instructions=None
    ):
        """
        Train both critics for single-agent archer.
        
        Args:
            sequence_critic: High-level Q function critic
            token_critic: Low-level V function critic
            state, action, base_state, next_state: Standard RL tensors
            hl_rewards: High-level rewards for sequence_critic
            ll_rewards: Low-level rewards for token_critic
            environment, tokenizer, instructions: Additional context
        
        Returns:
            Dictionary with loss values
        """
        import torch
        import torch.nn.functional as F
        
        device = next(sequence_critic.parameters()).device
        model_dtype = next(sequence_critic.parameters()).dtype
        
        # ------------------------------------------------------------------
        # TRAIN SEQUENCE-LEVEL CRITIC (Q function) with HL rewards
        # ------------------------------------------------------------------
        sequence_critic.train()
        
        # Compute Q-learning targets for sequence critic using HL rewards
        with torch.no_grad():
            if environment is not None:
                # Q(s,a) = r_HL + γ * V(s')
                next_input = torch.cat([next_state, action], dim=1)
                next_attention_mask = (next_input != tokenizer.pad_token_id).long().to(device=device)
                
                # Compute V(s') = sequence_critic(next_state)
                next_state_values = sequence_critic(
                    input_ids=next_input,
                    attention_mask=next_attention_mask,
                    requires_grad=False
                )
                
                if next_state_values.dim() > 1 and next_state_values.size(-1) == 1:
                    next_state_values = next_state_values.squeeze(-1)
                
                # Bellman equation: Q(s,a) = r_HL + γ * V(s')
                q_high_targets = hl_rewards + self.gamma * next_state_values.detach()
            else:
                # Fallback to direct HL rewards if no environment
                q_high_targets = hl_rewards
            
            q_high_targets = q_high_targets.to(device=device, dtype=model_dtype)
        
        # Forward pass for sequence critic
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
        
        # Ensure target shape matches
        if q_high_targets.dim() == 0:
            q_high_targets = q_high_targets.unsqueeze(0)
        
        # Compute and backprop sequence critic loss
        sequence_critic_loss = F.mse_loss(sequence_value_current, q_high_targets)
        
        self.sequence_critic_optimizer.zero_grad()
        sequence_critic_loss.backward()
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(sequence_critic.parameters(), self.max_grad_norm)
        
        self.sequence_critic_optimizer.step()
        
        # Clean up sequence critic computation graph
        del sequence_value_current, sequence_input, sequence_attention_mask
        torch.cuda.empty_cache()
        
        # ------------------------------------------------------------------
        # TRAIN TOKEN-LEVEL CRITIC (V function) with Skywork rewards
        # ------------------------------------------------------------------
        token_critic.train()
        
        # Check if we should use Skywork reward model for token-level critic training
        use_skywork_for_token_critic = getattr(self.config.training, 'use_skywork_for_token_critic', True)
        skywork_model_name = getattr(self.config.training, 'skywork_model_name', "Skywork/Skywork-Reward-V2-Llama-3.1-8B")
        
        if use_skywork_for_token_critic and environment is not None:
            # Use Skywork reward model for token-level critic training
            print(f"Using Skywork reward model ({skywork_model_name}) for token-level critic training")
            
            # Compute marginal rewards using Skywork for better generation quality assessment
            marginal_rewards = self.compute_marginal_rewards_with_skywork(
                state=state,
                action=action,
                base_state=base_state,
                environment=environment,
                tokenizer=tokenizer,
                sequence_critic=sequence_critic,
                top_k_tokens=getattr(self.config.training, 'num_masked_marginal_tokens', 5),
                pairwise_masking=getattr(self.config.training, 'pairwise_marginal_masking', False),
                instructions=instructions,
                skywork_model_name=skywork_model_name
            )
            
            # Use marginal rewards as targets for V function training
            v_low_targets = marginal_rewards
            
        else:
            # Fallback to original LL rewards or compute marginal rewards using environment
            if environment is not None:
                # Compute marginal rewards using environment's safety scoring
                marginal_rewards = self.compute_marginal_rewards(
                    state=state,
                    action=action,
                    base_state=base_state,
                    environment=environment,
                    tokenizer=tokenizer,
                    sequence_critic=sequence_critic,
                    instructions=instructions
                )
                v_low_targets = marginal_rewards
            else:
                # Fallback: broadcast LL rewards across tokens
                batch_size = ll_rewards.size(0) 
                state_length = state.size(1)
                v_low_targets = ll_rewards.unsqueeze(1).expand(batch_size, state_length)
            
            v_low_targets = v_low_targets.to(device=device, dtype=model_dtype)
        
        # Forward pass for token critic (V function - state only, no action)
        token_attention_mask = (state != tokenizer.pad_token_id).long().to(device=device)
        
        token_value_current = token_critic(
            input_ids=state,  # V function takes state only
            attention_mask=token_attention_mask,
            requires_grad=True
        )
        
        if token_value_current.dim() > 1 and token_value_current.size(-1) == 1:
            token_value_current = token_value_current.squeeze(-1)
        
        token_value_current = token_value_current.to(dtype=model_dtype)
        
        # Ensure targets match the critic output shape
        if v_low_targets.shape != token_value_current.shape:
            # If targets are per-token but critic outputs sequence-level, take mean
            if v_low_targets.dim() == 2 and token_value_current.dim() == 1:
                v_low_targets = v_low_targets.mean(dim=1)  # [batch_size]
            # If targets are sequence-level but critic outputs per-token, expand
            elif v_low_targets.dim() == 1 and token_value_current.dim() == 2:
                v_low_targets = v_low_targets.unsqueeze(1).expand(-1, token_value_current.size(1))
        
        # Compute and backprop token critic loss
        token_critic_loss = F.mse_loss(token_value_current, v_low_targets)
        
        self.token_critic_optimizer.zero_grad()
        token_critic_loss.backward()
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(token_critic.parameters(), self.max_grad_norm)
        
        self.token_critic_optimizer.step()
        
        # Final cleanup
        del token_value_current, token_attention_mask
        torch.cuda.empty_cache()
        
        return {
            'sequence_critic_loss': sequence_critic_loss.item(),
            'token_critic_loss': token_critic_loss.item()
        }
    
    def train_actor_single_agent(
        self,
        actor,
        state,
        action,
        old_log_probs,
        sequence_critic,
        token_critic,
        hl_rewards,
        base_state,
        environment=None,
        tokenizer=None
    ):
        """
        Train actor using single-agent archer objective: J = Σ(A_high - V_low(s[0:i-1]))
        
        Args:
            actor: The policy to train
            state, action, old_log_probs: Standard RL tensors
            sequence_critic: High-level Q function critic
            token_critic: Low-level V function critic
            hl_rewards: High-level rewards for advantage computation
            base_state, environment, tokenizer: Additional context
        
        Returns:
            Dictionary of training metrics
        """
        import torch
        import torch.nn.functional as F
        
        device = next(actor.parameters()).device
        
        # Compute high-level advantages using sequence critic
        with torch.no_grad():
            # Get current Q values from sequence critic
            sequence_input = torch.cat([state, action], dim=1)
            sequence_attention_mask = (sequence_input != tokenizer.pad_token_id).long().to(device=device)
            
            current_q_values = sequence_critic(
                input_ids=sequence_input,
                attention_mask=sequence_attention_mask
            )
            
            if current_q_values.dim() > 1 and current_q_values.size(-1) == 1:
                current_q_values = current_q_values.squeeze(-1)
            
            # Ensure correct dtype
            if current_q_values.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                current_q_values = current_q_values.float()
            
            # Compute advantages: A_high = R_HL - Q(s,a)
            advantages_high = hl_rewards - current_q_values
            
            # Normalize advantages
            if advantages_high.numel() > 1:
                advantages_high = (advantages_high - advantages_high.mean()) / (advantages_high.std() + 1e-8)
        
        # Compute low-level V function values for partial sequences
        with torch.no_grad():
            # For each action token, compute V_low(s[0:i-1])
            batch_size = state.size(0)
            action_length = action.size(1)
            state_length = state.size(1)
            
            # Create partial state sequences: s[0:i-1] for each i
            partial_state_values = []
            
            for i in range(action_length):
                if i == 0:
                    # For first token: V_low(s[0:0]) = V_low(s)
                    partial_state = state
                else:
                    # For subsequent tokens: concatenate state with action tokens up to i-1
                    partial_action = action[:, :i]
                    partial_state = torch.cat([state, partial_action], dim=1)
                
                # Get V function estimate for partial state
                partial_attention_mask = (partial_state != tokenizer.pad_token_id).long().to(device=device)
                
                # For V function, we only need the state part, not the action part
                if i == 0:
                    v_input = state
                    v_attention_mask = (state != tokenizer.pad_token_id).long().to(device=device)
                else:
                    # For partial sequences, we need to handle the concatenated input
                    # Since V function expects only state, we'll use the original state
                    # and adjust the value based on the partial action context
                    v_input = state
                    v_attention_mask = (state != tokenizer.pad_token_id).long().to(device=device)
                
                v_values = token_critic(
                    input_ids=v_input,
                    attention_mask=v_attention_mask
                )
                
                # Take the mean across state tokens for sequence-level V estimate
                if v_values.dim() > 1:
                    v_values = v_values.mean(dim=1)  # [batch_size]
                
                partial_state_values.append(v_values)
            
            # Stack partial state values: [action_length, batch_size]
            partial_state_values = torch.stack(partial_state_values, dim=0)  # [action_length, batch_size]
            partial_state_values = partial_state_values.transpose(0, 1)  # [batch_size, action_length]
        
        # Compute single-agent archer objective: J = Σ(A_high - V_low(s[0:i-1]))
        # Expand advantages_high to match action dimensions
        advantages_expanded = advantages_high.unsqueeze(1).expand(-1, action_length)
        
        # Compute the objective for each action token
        objective_per_token = advantages_expanded - partial_state_values
        
        # Create action mask for valid tokens
        action_mask = (action != tokenizer.pad_token_id).long()
        
        # Mask the objective to only valid tokens
        masked_objective = objective_per_token * action_mask.float()
        
        # Compute the total objective (sum across tokens)
        total_objective = masked_objective.sum(dim=1)  # [batch_size]
        
        # For PPO-style training, we want to maximize the objective
        # Convert to loss by negating (minimizing negative objective = maximizing objective)
        actor_loss = -total_objective.mean()
        
        # Add entropy regularization for exploration
        entropy_coef = getattr(self, 'entropy_coef', 0.01)
        
        # Compute policy entropy (this would need to be computed from the actor's output)
        # For now, we'll use a placeholder - in practice, this should come from the actor's logits
        if hasattr(actor, 'compute_entropy'):
            entropy = actor.compute_entropy(state, action)
        else:
            # Placeholder entropy computation
            entropy = torch.zeros(batch_size, device=device)
        
        entropy_loss = -entropy.mean() * entropy_coef
        
        # Total loss
        total_loss = actor_loss + entropy_loss
        
        # Update actor
        if hasattr(self, 'actor_optimizer') and self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            
            # Apply gradient clipping
            max_grad_norm = getattr(self, 'max_grad_norm', 0.5)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            
            self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_objective_mean': total_objective.mean().item(),
            'advantages_high_mean': advantages_high.mean().item(),
            'advantages_high_std': advantages_high.std().item(),
            'partial_state_values_mean': partial_state_values.mean().item()
        }
