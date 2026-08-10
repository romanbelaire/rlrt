"""
Hierarchical 20 Questions Environment for dual-agent training.

This module provides the HierarchicalTwentyQuestionsEnvironment class that
supports hierarchical agent training with separate high-level (meta-cognitive)
and low-level (question generation) policies.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import random
from abc import ABC, abstractmethod

from src.base import Environment
from src.hierarchical_conversation import TwentyQuestionsConversation
from src.hierarchical_20q_prompts import (
    get_high_level_system_prompt,
    get_low_level_system_prompt,
    format_high_level_context,
    format_low_level_context,
    detect_guess_instruction
)


class TwentyQuestionsOracle(ABC):
    """Abstract interface for 20 Questions oracle implementations."""
    
    @abstractmethod
    def generate_answers(self, words: Union[str, List[str]], questions: Union[str, List[str]], return_full: bool = False) -> Union[str, List[str]]:
        """Generate oracle answers for given words and questions."""
        pass


class HierarchicalTwentyQuestionsEnvironment(Environment):
    """
    Environment for hierarchical 20 Questions with dual-agent training.
    
    This environment supports training of both high-level (meta-cognitive) and 
    low-level (question generation) policies simultaneously using separate reward streams.
    """
    
    def __init__(
        self,
        oracle: TwentyQuestionsOracle,
        word_list: List[Union[str, List[str]]],
        max_conversation_length: int = 20,
        debug_mode: bool = False,
        # Judge model for instruction-following rewards
        judge_model: Optional[Any] = None,
        judge_tokenizer: Optional[Any] = None,
        judge_batch_size: int = 32,
        # Reward configuration
        hl_reward_success: float = 0.0,
        hl_reward_failure: float = -1.0,
        ll_reward_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize the hierarchical 20 Questions environment.
        
        Args:
            oracle: Oracle for answering yes/no questions
            word_list: List of target words (can be strings or lists of variants)
            max_conversation_length: Maximum questions per game
            debug_mode: Enable debug logging
            judge_model: Model for evaluating instruction-following quality
            judge_tokenizer: Tokenizer for judge model
            hl_reward_success: High-level reward for successful episodes
            hl_reward_failure: High-level reward for failed steps
            ll_reward_range: Range for low-level instruction-following rewards
        """
        super().__init__()
        
        self.oracle = oracle
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length
        self.debug_mode = debug_mode
        
        # Judge model for instruction-following evaluation
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.judge_batch_size = judge_batch_size
        
        # Reward configuration
        self.hl_reward_success = hl_reward_success
        self.hl_reward_failure = hl_reward_failure
        self.ll_reward_range = ll_reward_range
        
        # Post-hoc LL reward computation
        self.posthoc_ll_rewards = True  # Enable post-hoc LL reward computation
        self.collected_instruction_action_pairs = []  # Store (hl_action, ll_action, conv) pairs for post-hoc processing
        
        # Judge request accumulation for better batching (reduced threshold to prevent OOM)
        self.judge_request_queue = []
        self.judge_request_threshold = 16  # Process when we have 16 requests
        
        
        # Environment state
        self.conversations: List[TwentyQuestionsConversation] = []
        self.batch_size = 0
        self.current_step = 0
        
        # Random number generator
        self.random = random.Random(None)
        
        # Task identifier for trainer integration
        self.task = "hierarchical_twenty_questions"
    
    def reset(self, batch_size: int = 1) -> List[List[str]]:
        """
        Reset the environment with a new batch of conversations.
        
        Args:
            batch_size: Number of parallel conversations to create
            
        Returns:
            Initial state for all conversations
        """
        self.batch_size = batch_size
        self.current_step = 0
        
        # Clear existing conversations
        self.conversations = []
        
        # Initialize conversations
        for i in range(batch_size):
            conv_id = f"hierarchical_20q_batch_{i}_step_{self.current_step}"
            conv = TwentyQuestionsConversation(conv_id)
            conv.set_metadata("batch_index", i)
            conv.set_metadata("game_status", "active")
            
            # Select target word
            target_word = self.random.choice(self.word_list)
            conv.set_target_word(target_word)
            
            self.conversations.append(conv)
            
            if self.debug_mode:
                word_display = target_word if isinstance(target_word, str) else target_word[0]
                print(f"[DEBUG] Initialized hierarchical 20Q conversation {i} with word: '{word_display}'")
        
        # Generate initial state
        return self._conversations_to_state()
    
    def step(
        self, 
        hl_actions: List[str], 
        ll_actions: Optional[List[str]] = None
    ) -> Tuple[List[List[str]], List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one step for all conversations with hierarchical actions.
        
        Args:
            hl_actions: High-level strategy instructions
            ll_actions: Low-level questions/guesses (if None, will be generated from HL)
            
        Returns:
            Tuple of (next_state, responses, hl_rewards, ll_rewards, dones, info)
        """
        if not self.conversations:
            raise ValueError("Environment not reset. Call reset() first.")
        
        responses = []
        hl_rewards = []
        ll_rewards = []
        dones = []
        
        # Collect instruction-action pairs for post-hoc LL reward computation
        instruction_action_pairs = []
        pair_indices = []  # Track which conversations need judge evaluation
        
        # PERFORMANCE FIX: Pre-process all conversations to identify active ones
        active_conversations = []
        active_indices = []
        active_hl_actions = []
        active_ll_actions = []
        
        for i, hl_action in enumerate(hl_actions):
            if i >= len(self.conversations):
                # Handle case where we have more actions than conversations
                responses.append("")
                hl_rewards.append(0.0)
                ll_rewards.append(0.0)
                dones.append(True)
                continue
            
            conv = self.conversations[i]
            
            # Skip if conversation is already done
            if conv.episode_done or conv.question_count >= self.max_conversation_length:
                responses.append("")
                hl_rewards.append(0.0)
                ll_rewards.append(0.0)
                dones.append(True)
                continue
            
            # Get low-level action (question/guess)
            if ll_actions and i < len(ll_actions):
                ll_action = ll_actions[i]
            else:
                # Generate LL action from HL instruction (for inference/evaluation)
                ll_action = self._generate_ll_action_from_hl(hl_action, conv)
            
            # Debug: Show what actions are being processed
            if self.debug_mode:
                print(f"[DEBUG] Conv {i}: HL action = '{hl_action[:100]}{'...' if len(hl_action) > 100 else ''}'")
                print(f"[DEBUG] Conv {i}: LL action = '{ll_action[:100]}{'...' if len(ll_action) > 100 else ''}'")
            
            # Collect for batched processing
            active_conversations.append(conv)
            active_indices.append(i)
            active_hl_actions.append(hl_action)
            active_ll_actions.append(ll_action)
            
            # Collect instruction-action pair for post-hoc LL reward computation
            if self.judge_model is not None and self.judge_tokenizer is not None:
                instruction_action_pairs.append((hl_action, ll_action, conv))
                pair_indices.append(i)
        
        # PERFORMANCE FIX: Batch process all active conversations
        if active_conversations:
            batch_responses, batch_hl_rewards, batch_ll_rewards, batch_dones = self._batch_process_hierarchical_actions(
                active_conversations, active_hl_actions, active_ll_actions, active_indices
            )
            
            # Insert results back into the main lists
            for idx, (response, hl_reward, ll_reward, done) in enumerate(zip(
                batch_responses, batch_hl_rewards, batch_ll_rewards, batch_dones
            )):
                original_idx = active_indices[idx]
                # Ensure we have enough space in the lists
                while len(responses) <= original_idx:
                    responses.append("")
                    hl_rewards.append(0.0)
                    ll_rewards.append(0.0)
                    dones.append(False)
                
                responses[original_idx] = response
                hl_rewards[original_idx] = hl_reward
                ll_rewards[original_idx] = ll_reward
                dones[original_idx] = done
        
        # Post-hoc LL reward computation: collect instruction-action pairs
        if instruction_action_pairs and self.posthoc_ll_rewards:
            # Store pairs for later batch processing
            for i, (hl_action, ll_action, conv) in enumerate(instruction_action_pairs):
                self.collected_instruction_action_pairs.append({
                    'hl_action': hl_action,
                    'll_action': ll_action,
                    'conversation': conv,
                    'step_index': pair_indices[i]
                })
            
            # Set placeholder LL rewards (will be updated later)
            for idx in pair_indices:
                ll_rewards[idx] = 0.0  # Placeholder
        elif instruction_action_pairs:
            # Fallback to immediate judge scoring if post-hoc is disabled
            judge_inputs = [(hl_action, ll_action, conv) for hl_action, ll_action, conv in instruction_action_pairs]
            batch_judge_scores = self._batch_judge_instruction_following(judge_inputs)
            
            # Update ll_rewards with batched judge scores and update conversations
            for idx, score in zip(pair_indices, batch_judge_scores):
                ll_rewards[idx] = score
                # Update the conversation with the final ll_reward
                self._update_conversation_ll_reward(idx, score)
        
        # Update current step
        self.current_step += 1
        
        # Generate new state
        next_state = self._conversations_to_state()
        
        return (
            next_state,
            responses,
            torch.tensor(hl_rewards, dtype=torch.float32),
            torch.tensor(ll_rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
            {}  # Additional info if needed
        )
    
    def process_posthoc_ll_rewards(self) -> Dict[int, float]:
        """
        Process all collected instruction-action pairs and return LL rewards.
        This should be called after rollout completion for efficient batch processing.
        
        Returns:
            Dict mapping step_index to LL reward score
        """
        if not self.collected_instruction_action_pairs:
            return {}
        
        print(f"DEBUG: Processing {len(self.collected_instruction_action_pairs)} instruction-action pairs for post-hoc LL rewards")
        
        # Convert to judge input format
        judge_inputs = []
        step_indices = []
        
        for pair in self.collected_instruction_action_pairs:
            hl_action = pair['hl_action']
            ll_action = pair['ll_action']
            conv = pair['conversation']
            
            judge_inputs.append((hl_action, ll_action, conv))
            step_indices.append(pair['step_index'])
        
        # Batch process all judge inputs
        judge_scores = self._batch_judge_instruction_following(judge_inputs)
        
        # Create mapping from step_index to LL reward
        ll_rewards_map = {}
        for step_idx, score in zip(step_indices, judge_scores):
            ll_rewards_map[step_idx] = score
            # Update the conversation with the final ll_reward
            self._update_conversation_ll_reward(step_idx, score)
        
        # Clear collected pairs
        self.collected_instruction_action_pairs = []
        
        print(f"DEBUG: Post-hoc LL rewards computed for {len(ll_rewards_map)} steps")
        return ll_rewards_map
    
    def _batch_process_hierarchical_actions(
        self,
        conversations: List[TwentyQuestionsConversation],
        hl_actions: List[str],
        ll_actions: List[str],
        indices: List[int]
    ) -> Tuple[List[str], List[float], List[float], List[bool]]:
        """
        Batch process hierarchical actions for multiple conversations.
        
        Args:
            conversations: List of conversation objects
            hl_actions: List of high-level actions
            ll_actions: List of low-level actions
            indices: Original indices of conversations
            
        Returns:
            Tuple of (responses, hl_rewards, ll_rewards, dones)
        """
        if not conversations:
            return [], [], [], []
        
        # Extract questions from low-level actions
        questions = []
        for ll_action in ll_actions:
            question = self._extract_question_from_response(ll_action)
            questions.append(question)
        
        # PERFORMANCE FIX: Batch oracle calls
        target_words = []
        for conv in conversations:
            target_word = conv.target_word
            if isinstance(conv.word_variants, list) and conv.word_variants:
                target_word = conv.word_variants[0]
            target_words.append(target_word)
        
        # Single batched oracle call instead of individual calls
        oracle_responses = self.oracle.generate_answers(target_words, questions)
        
        # PERFORMANCE FIX: Batch judge calls instead of serial loop
        # Collect judge inputs first
        judge_inputs = []
        for conv, hl_action, ll_action in zip(conversations, hl_actions, ll_actions):
            judge_inputs.append((hl_action, ll_action, conv))
        
        # Single batched judge call instead of individual calls
        ll_rewards_batch = self._batch_judge_instruction_following(judge_inputs)
        
        # Process results
        responses = []
        hl_rewards = []
        ll_rewards = []
        dones = []
        
        for i, (conv, hl_action, ll_action, question, oracle_response, target_word, ll_reward) in enumerate(zip(
            conversations, hl_actions, ll_actions, questions, oracle_responses, target_words, ll_rewards_batch
        )):
            # Check if word was guessed correctly
            word_guessed = self._check_word_guessed(ll_action, question, oracle_response, conv)
            
            # Determine if episode is done
            conv.question_count += 1
            done = word_guessed or conv.question_count >= self.max_conversation_length
            
            # Calculate high-level reward (game outcome based)
            if word_guessed:
                hl_reward = self.hl_reward_success  # Success
                conv.set_episode_result(won=True, done=done)
                if self.debug_mode:
                    print(f"[DEBUG] Hierarchical 20Q conversation {indices[i]}: Won in {conv.question_count} questions!")
            else:
                hl_reward = self.hl_reward_failure  # Step penalty
                if done:
                    conv.set_episode_result(won=False, done=done)
                    if self.debug_mode:
                        print(f"[DEBUG] Hierarchical 20Q conversation {indices[i]}: Lost after {conv.question_count} questions")
            
            # Low-level reward already computed via batch (ll_reward from zip)

            # Update conversation
            conv.add_question_answer_pair(
                question=question,
                answer=oracle_response,
                hl_strategy=hl_action,
                hl_reward=hl_reward,
                ll_reward=ll_reward
            )
            
            # Update conversation metadata
            conv.set_metadata(f"step_{self.current_step}_done", done)
            conv.set_metadata(f"step_{self.current_step}_hl_reward", hl_reward)
            conv.set_metadata(f"step_{self.current_step}_ll_reward", ll_reward)
            
            responses.append(oracle_response)
            hl_rewards.append(hl_reward)
            ll_rewards.append(ll_reward)
            dones.append(done)
        
        return responses, hl_rewards, ll_rewards, dones
    
    def _batch_judge_instruction_following(
        self, 
        judge_inputs: List[Tuple[str, str, TwentyQuestionsConversation]]
    ) -> List[float]:
        """
        Batch version of judge scoring for performance optimization.
        Processes multiple judge scoring requests in a single forward pass.
        
        Args:
            judge_inputs: List of (hl_instruction, ll_action, conv) tuples
            
        Returns:
            List of judge scores
        """
        if not judge_inputs:
            return []
        
        if self.judge_model is None or self.judge_tokenizer is None:
            # Return neutral scores if no judge model
            return [0.5] * len(judge_inputs)

        # Process in chunks to avoid memory issues with large batches
        chunk_size = self.judge_batch_size  # Use the configured batch size
        scores = []
        
        for chunk_start in range(0, len(judge_inputs), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(judge_inputs))
            chunk_inputs = judge_inputs[chunk_start:chunk_end]
            
            # Build batched prompts for this chunk
            judge_prompts = []
            for hl_instruction, ll_action, conv in chunk_inputs:
                # Format prompt for judge model
                context = conv.get_conversation_context_for_ll(hl_instruction)
                
                judge_prompt = f"""You are a strict evaluator for 20 Questions game instruction-following quality.

Game Context: {context}

High-Level Strategic Instruction: {hl_instruction}
Low-Level Agent Action: {ll_action}

Given the strategic instruction, does the agent's action appropriately follow the instruction for the 20 Questions game?

Consider:
- Does the action align with the strategic guidance?
- Is the action appropriate for the current game state?
- Does the action help progress toward solving the mystery word?

Provide a YES or NO response."""
                
                judge_prompts.append(judge_prompt)

            # Tokenize all prompts in batch
            inputs = self.judge_tokenizer(
                judge_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1024  # Use full context length
            )
            
            # Determine device and move inputs to it
            try:
                device = self.judge_model.get_input_embeddings().weight.device
            except Exception:
                device = next(self.judge_model.parameters()).device
            
            # Clear GPU cache before processing to avoid memory issues
            torch.cuda.empty_cache()
            
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except RuntimeError as e:
                if "CUDA" in str(e) or "memory" in str(e).lower():
                    # If CUDA memory error, try with smaller chunk
                    print(f"[WARNING] CUDA memory error in judge batch processing: {e}")
                    print(f"[WARNING] Falling back to individual processing for chunk of size {len(chunk_inputs)}")
                    # Process individually instead of in batch
                    chunk_scores = []
                    for hl_instruction, ll_action, conv in chunk_inputs:
                        individual_score = self._judge_instruction_following(hl_instruction, ll_action, conv)
                        chunk_scores.append(individual_score)
                    scores.extend(chunk_scores)
                    continue
                else:
                    raise
            
            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.judge_tokenizer.pad_token_id,
                    eos_token_id=self.judge_tokenizer.eos_token_id
                )
            
            # Extract responses and parse scores
            input_length = inputs['input_ids'].shape[1]
            for i, output in enumerate(outputs):
                # Extract response (skip input tokens)
                response = self.judge_tokenizer.decode(
                    output[input_length:], 
                    skip_special_tokens=True
                ).strip()
                
                # Parse YES/NO response
                response_lower = response.lower()
                if "yes" in response_lower and "no" not in response_lower:
                    raw_score = 1.0  # Perfect instruction following
                elif "no" in response_lower and "yes" not in response_lower:
                    raw_score = 0.0  # Poor instruction following
                else:
                    # Ambiguous response - try to extract numeric score as fallback
                    import re
                    score_match = re.search(r'\d+', response)
                    if score_match:
                        raw_score = int(score_match.group()) / 10.0  # Normalize to 0-1
                    else:
                        raw_score = 0.5  # Neutral fallback
                
                # Scale to reward range
                min_reward, max_reward = self.ll_reward_range
                final_score = min_reward + (max_reward - min_reward) * raw_score
                
                if self.debug_mode:
                    print(f"[DEBUG] Batch judge evaluation {i}: '{response}' -> {raw_score:.2f} -> {final_score:.2f}")
                
                scores.append(final_score)
            
            # Clean up GPU memory after each chunk
            del inputs, outputs
            torch.cuda.empty_cache()
        
        return scores
    
    def _update_conversation_ll_reward(self, conv_idx: int, ll_reward: float):
        """
        Update the conversation with the final low-level reward after batched judge evaluation.
        
        Args:
            conv_idx: Conversation index
            ll_reward: Final low-level reward score
        """
        if conv_idx < len(self.conversations):
            conv = self.conversations[conv_idx]
            
            # Update the last question-answer pair with the final ll_reward
            if conv.oracle_qa_tuples:
                # Update the conversation's ll_rewards list
                if len(conv.ll_rewards) > 0:
                    conv.ll_rewards[-1] = ll_reward
                
                # Update metadata
                conv.set_metadata(f"step_{self.current_step}_ll_reward", ll_reward)
                
                if self.debug_mode:
                    print(f"[DEBUG] Updated conversation {conv_idx} with final ll_reward: {ll_reward:.3f}")
    
    def _judge_instruction_following(
        self, 
        hl_instruction: str, 
        ll_action: str, 
        conv: TwentyQuestionsConversation
    ) -> float:
        """
        Use judge model to evaluate instruction following quality.
        
        Args:
            hl_instruction: High-level instruction
            ll_action: Low-level action
            conv: Conversation object
            
        Returns:
            Judge score between ll_reward_range
        """
        # Format prompt for judge model - following the multi-turn MC plan pattern
        context = conv.get_conversation_context_for_ll(hl_instruction)
        
        judge_prompt = f"""You are a strict evaluator for 20 Questions game instruction-following quality.

Game Context: {context}

High-Level Strategic Instruction: {hl_instruction}
Low-Level Agent Action: {ll_action}

Given the strategic instruction, does the agent's action appropriately follow the instruction for the 20 Questions game?

Consider:
- Does the action align with the strategic guidance?
- Is the action appropriate for the current game state?
- Does the action help progress toward solving the mystery word?

Provide a YES or NO response."""

        try:
            # Tokenize and generate judge score
            inputs = self.judge_tokenizer(judge_prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Determine device and move inputs to it
            try:
                device = self.judge_model.get_input_embeddings().weight.device
            except Exception:
                device = next(self.judge_model.parameters()).device
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=None,  # Explicitly set to None to override model's default generation config
                    top_p=None,        # Explicitly set to None to override model's default generation config
                    pad_token_id=self.judge_tokenizer.pad_token_id,
                    eos_token_id=self.judge_tokenizer.eos_token_id
                )
            
            # Extract response
            input_length = inputs['input_ids'].shape[1]
            response = self.judge_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            # Parse YES/NO response
            response_lower = response.lower()
            if "yes" in response_lower and "no" not in response_lower:
                raw_score = 1.0  # Perfect instruction following
            elif "no" in response_lower and "yes" not in response_lower:
                raw_score = 0.0  # Poor instruction following
            else:
                # Ambiguous response - try to extract numeric score as fallback
                import re
                score_match = re.search(r'\d+', response)
                if score_match:
                    raw_score = int(score_match.group()) / 10.0  # Normalize to 0-1
                else:
                    raw_score = 0.5  # Neutral fallback
            
            # Scale to reward range
            min_reward, max_reward = self.ll_reward_range
            final_score = min_reward + (max_reward - min_reward) * raw_score
            
            if self.debug_mode:
                print(f"[DEBUG] Judge evaluation: '{response}' -> {raw_score:.2f} -> {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Judge model error: {e}")
            # Fallback to neutral score
            return 0.5
    
    
    def _extract_question_from_response(self, response: str) -> str:
        """Extract the actual question from the agent's verbose response."""
        lines = response.split('\n')
        
        # Look for lines that start with "is" and end with "?"
        for line in lines:
            line = line.strip()
            if line:
                line_lower = line.lower()
                if line_lower.startswith('is') and line_lower.endswith('?'):
                    return line
        
        # If no "is" + "?" pattern found, look for any lines that end with question marks
        question_lines = [line.strip() for line in lines if line.strip().endswith('?')]
        if question_lines:
            return question_lines[-1]  # Return the last question found
        return response  # Fallback to full response
    
    def _check_word_guessed(
        self, 
        action: str, 
        question: str, 
        answer: str, 
        conv: TwentyQuestionsConversation
    ) -> bool:
        """Check if the agent correctly guessed the word."""
        question_lower = question.lower().strip()
        
        # Debug: Check if answer is actually a list
        if isinstance(answer, list):
            if self.debug_mode:
                print(f"[DEBUG] Warning: answer is a list: {answer}, taking first element")
            answer = answer[0] if answer else "no"
        
        answer_lower = answer.lower().strip()
        
        # Only detect wins on "Yes" answers to direct guess questions
        if answer_lower in ["yes", "yes.", "y"]:
            # Check all word variants
            for word_variant in conv.word_variants:
                word_lower = word_variant.lower()
                
                # Simple check: does the question contain the exact word as a whole word?
                question_words = question_lower.split()
                if word_lower in question_words:
                    if self.debug_mode:
                        print(f"[DEBUG] Agent guessed '{word_variant}' and got 'Yes'!")
                    return True
        
        return False
    
    def _conversations_to_state(self) -> List[List[str]]:
        """Convert conversations to state format for trainer compatibility."""
        states = []
        
        for conv in self.conversations:
            # Build state as list of conversation turns
            state_parts = ["I am thinking of a word. You have 20 questions to guess what it is."]
            
            # Add conversation history
            for i in range(len(conv.questions_asked)):
                if i < len(conv.questions_asked):
                    state_parts.append(conv.questions_asked[i])
                if i < len(conv.oracle_answers):
                    state_parts.append(conv.oracle_answers[i])
            
            # Add hint on final turn (question count == 20) for debugging win detection
            if conv.question_count >= self.max_conversation_length:
                # Get the target word for the hint
                target_word = conv.target_word
                if isinstance(conv.word_variants, list) and conv.word_variants:
                    target_word = conv.word_variants[0]
                elif target_word is None:
                    target_word = "Unknown"
                hint = f"Hint: the word is {target_word}"
                state_parts.append(hint)
                
                # DEBUG: Log when hint is added
                if self.debug_mode:
                    print(f"[DEBUG] Added hint for conversation with question_count={conv.question_count}, target_word={target_word}")
            
            states.append(state_parts)
        
        return states
    
    def get_system_prompt_hl(self) -> str:
        """Get system prompt for high-level agent."""
        return get_high_level_system_prompt()
    
    def get_system_prompt_ll(self) -> str:
        """Get system prompt for low-level agent."""
        return get_low_level_system_prompt()
    
    def get_conversation_context_hl(self, conv_idx: int) -> str:
        """Get formatted context for high-level agent."""
        if conv_idx >= len(self.conversations):
            return ""
        
        conv = self.conversations[conv_idx]
        conversation_context = conv.get_conversation_context_for_hl()
        information_summary = conv.get_information_summary()
        
        return format_high_level_context(conversation_context, information_summary)
    
    def get_batch_conversation_contexts_hl(self, conv_indices: List[int]) -> List[str]:
        """
        Get formatted contexts for high-level agent in batch.
        
        Args:
            conv_indices: List of conversation indices
            
        Returns:
            List of formatted contexts
        """
        contexts = []
        for conv_idx in conv_indices:
            context = self.get_conversation_context_hl(conv_idx)
            contexts.append(context)
        return contexts
    
    def get_conversation_context_ll(self, conv_idx: int, hl_instruction: str) -> str:
        """Get formatted context for low-level agent."""
        if conv_idx >= len(self.conversations):
            return ""
        
        conv = self.conversations[conv_idx]
        conversation_context = conv.get_conversation_context_for_ll(hl_instruction)
        game_status = f"{conv.question_count}/{conv.max_questions} questions used"
        
        return format_low_level_context(hl_instruction, conversation_context, game_status)
    
    def score_safety(self, responses: List[str], questions: Optional[List[str]] = None, reference_state: Optional[List[str]] = None, **kwargs) -> torch.Tensor:
        """
        Evaluate instruction-following quality using the judge model.
        
        Args:
            responses: Low-level actions (questions/guesses) to evaluate
            questions: Current state/questions (unused for 20Q)
            reference_state: Reference state (unused for 20Q)
            **kwargs: Additional arguments
            
        Returns:
            Judge scores for instruction-following quality
        """
        if not responses:
            return torch.zeros(0, dtype=torch.float32)
        
        # Require judge model to be available
        if self.judge_model is None or self.judge_tokenizer is None:
            raise ValueError("Judge model and tokenizer are required for score_safety evaluation")
        
        scores = []
        
        for i, response in enumerate(responses):
            # Get the corresponding conversation and high-level instruction
            if i >= len(self.conversations):
                # Handle batch size mismatch by cycling through available conversations
                conv_index = i % len(self.conversations) if len(self.conversations) > 0 else 0
                if len(self.conversations) == 0:
                    raise ValueError("No conversations available for scoring")
            else:
                conv_index = i
            
            conv = self.conversations[conv_index]
            
            # Require high-level instruction to be available
            if not hasattr(conv, 'hl_strategies') or not conv.hl_strategies:
                raise ValueError(f"No high-level strategies available for conversation {i}")
            
            hl_instruction = conv.hl_strategies[-1]
            
            # Use the judge function to evaluate instruction following
            score = self._judge_instruction_following(hl_instruction, response, conv)
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def get_target_responses(self, actions: Union[List[str], str], include_conversation_history: bool = False) -> List[str]:
        """
        Generate responses from the oracle without stepping the environment.
        
        For hierarchical 20Q environment, the oracle IS the target - it provides responses to questions.
        This method uses the oracle to generate responses to the given actions/questions.
        
        Args:
            actions: Actions/questions to get oracle responses for
            include_conversation_history: Ignored (oracle doesn't need conversation history)
            
        Returns:
            List of oracle responses to the questions
        """
        # Ensure actions is a list
        if isinstance(actions, str):
            actions = [actions]
            
        if not actions:
            return []
        
        # We need target words for each action to generate oracle responses
        # Use the current conversations' target words
        target_words = []
        questions = []
        
        for i, action in enumerate(actions):
            # Extract the actual question from the agent's response
            question = self._extract_question_from_response(action)
            questions.append(question)
            
            # Get the target word for this conversation
            if i < len(self.conversations):
                conv = self.conversations[i]
                target_word = conv.target_word
                if isinstance(conv.word_variants, list) and conv.word_variants:
                    # Use the full list of word variants for the oracle
                    target_word = conv.word_variants
            else:
                # Fallback: use a default word (shouldn't happen in normal usage)
                target_word = "unknown"
            
            target_words.append(target_word)
        
        # Generate oracle responses using the oracle
        try:
            oracle_responses = self.oracle.generate_answers(target_words, questions)
            
            # Ensure we return a list
            if isinstance(oracle_responses, str):
                oracle_responses = [oracle_responses]
                
            return oracle_responses
            
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error generating oracle responses: {e}")
            # Return fallback responses
            return ["No." for _ in actions]

    def get_target_responses_from_tokens(
        self, 
        action_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        tokenizer: Any
    ) -> List[str]:
        """
        Generate responses from the oracle using token IDs and attention masks directly.
        
        For hierarchical 20Q environment, the oracle IS the target - it provides responses to questions.
        This method decodes the tokens to text and uses the oracle to generate responses.
        
        Args:
            action_ids: Action token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] 
            tokenizer: Tokenizer to use for text conversion
            
        Returns:
            List of oracle responses to the decoded questions
        """
        # Decode tokens to text
        action_texts = self._batch_decode_with_attention_mask(action_ids, attention_mask, tokenizer)
        
        # Use the text-based method to get oracle responses
        return self.get_target_responses(action_texts)

    def score_safety_from_tokens(
        self,
        response_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        reference_state_ids: Optional[torch.Tensor] = None,
        reference_attention_mask: Optional[torch.Tensor] = None,
        tokenizer: Any = None,
        include_white_box: bool = True
    ) -> torch.Tensor:
        """
        Score safety using token IDs and attention masks directly.
        
        For hierarchical 20Q environment, there is no safety notion since it's a game.
        This method returns neutral safety scores to maintain compatibility with the trainer interface.
        
        Args:
            response_ids: Response token IDs [batch_size, seq_len]
            response_attention_mask: Response attention mask [batch_size, seq_len]
            question_ids: Question token IDs [batch_size, seq_len] (optional)
            question_attention_mask: Question attention mask [batch_size, seq_len] (optional) 
            reference_state_ids: Reference state token IDs [batch_size, seq_len] (optional)
            reference_attention_mask: Reference attention mask [batch_size, seq_len] (optional)
            tokenizer: Tokenizer to use for text conversion
            include_white_box: Whether to include white-box safety signals
            
        Returns:
            Safety scores (neutral for 20Q game)
        """
        # For hierarchical 20Q environment, return neutral safety scores since it's a game
        batch_size = int(response_ids.size(0))
        return torch.ones(batch_size, dtype=torch.float32) * 0.5
    
    def _get_embed_similarity_batched(self, sequences1: List[str], sequences2: List[str]) -> torch.Tensor:
        """
        Compute embedding similarity between two lists of sequences.
        
        For hierarchical 20Q environment, use the judge model for embeddings if available,
        otherwise return neutral similarity scores.
        
        Args:
            sequences1: First list of sequences
            sequences2: Second list of sequences
            
        Returns:
            Similarity scores between sequences
        """
        if len(sequences1) != len(sequences2):
            raise ValueError("_get_embed_similarity_batched requires equal-sized inputs")
        
        # If no judge model available, return neutral similarity scores
        if self.judge_model is None or self.judge_tokenizer is None:
            if self.debug_mode:
                print(f"[DEBUG] No judge model available for similarity computation, returning zeros")
            return torch.zeros(len(sequences1), dtype=torch.float32)
        
        try:
            # Use judge model for embeddings (similar to MultiTurnMultipleChoiceEnv approach)
            
            # Determine device using robust approach
            try:
                device = self.judge_model.get_input_embeddings().weight.device
            except Exception:
                device = next(self.judge_model.parameters()).device
            
            # Tokenize both sequence lists
            enc1 = self.judge_tokenizer(
                sequences1, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512, 
                padding_side='left'
            )
            enc2 = self.judge_tokenizer(
                sequences2, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512, 
                padding_side='left'
            )
            
            # Move to device
            enc1 = {k: v.to(device) for k, v in enc1.items()}
            enc2 = {k: v.to(device) for k, v in enc2.items()}
            
            with torch.no_grad():
                # Use input embedding average as a lightweight embedding proxy
                embedding_layer = self.judge_model.get_input_embeddings()
                
                # Get embeddings for both sequences
                emb1 = embedding_layer(enc1['input_ids'])  # [batch, seq_len, hidden_size]
                emb2 = embedding_layer(enc2['input_ids'])  # [batch, seq_len, hidden_size]
                
                # Mean pooling with attention mask
                def mean_pool(emb, mask):
                    masked_emb = emb * mask.unsqueeze(-1)
                    summed = masked_emb.sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                    return summed / counts
                
                pooled_emb1 = mean_pool(emb1, enc1['attention_mask'])  # [batch, hidden_size]
                pooled_emb2 = mean_pool(emb2, enc2['attention_mask'])  # [batch, hidden_size]
                
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(pooled_emb1, pooled_emb2, dim=-1)
                
                # Convert from [-1, 1] to [0, 1] range
                similarity_scores = (cos_sim + 1.0) / 2.0
                
                return similarity_scores.cpu()
                
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error computing embedding similarity: {e}")
            # Fallback to neutral scores
            return torch.zeros(len(sequences1), dtype=torch.float32)

    def _batch_decode_with_attention_mask(
        self, 
        token_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        tokenizer: Any
    ) -> List[str]:
        """
        Efficiently batch decode tokens while respecting attention masks.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            tokenizer: Tokenizer to use
            
        Returns:
            List of decoded texts
        """
        texts = []
        
        # Extract valid tokens for each sequence
        for i in range(token_ids.size(0)):
            # Only decode tokens where attention mask is 1
            valid_tokens = token_ids[i][attention_mask[i] == 1]
            text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
            texts.append(text)
        
        return texts