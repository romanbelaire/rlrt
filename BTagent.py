import json
import torch
import numpy as np
from agent import Agent
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F

class BTagent(Agent):
    """
    Bradley-Terry Agent for templated text generation with preference learning.
    
    This agent extends the base Agent class to implement templated generation with
    Bradley-Terry preference learning for template selection. It uses token-wise
    harm rewards to extract topics from harmful content and generates templated
    responses based on these topics.
    
    Key features:
    - Multi-option Bradley-Terry preference learning model (DeepSeek R1 style)
    - Token-wise harm detection for topic extraction
    - Template-based generation with topic substitution
    - Automatic rejection template fallback
    - HarmBench dataset integration for training and evaluation
    
    The multi-option BT approach ranks templates by their harm rating and uses
    the lowest-ranked non-refusing template as a reference point for preference learning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the templates from the JSON file
        with open("templates.json", 'r') as f:
            template_data = json.load(f)
        
        # Extract the templates array from the JSON
        self.templates = template_data["templates"]
        self.num_templates = len(self.templates) - 1  # Excluding rejection template
        
        self.rejection_template_idx = 0
        self.top_tokens_count = kwargs.get('top_tokens_count', 5)  # Number of highest-value tokens to use
        self.debug_mode = kwargs.get('debug_mode', False)  # Enable debug logging
    
    def act(self, state, base_state=None, instructions=None, enforce_single_question=False):
        """
        Generate templated actions based on token-wise harm rewards.
        
        Args:
            state: Current state (user query) - can be tensor, string, or list
            base_state: Original state if available - can be tensor, string, or list
            instructions: System instructions
            enforce_single_question: Whether to enforce single question (not used here)
            
        Returns:
            templated_responses: List of templated responses
            templated_tokens: Tokenized templated responses
            log_probs: Log probabilities (dummy values in this implementation)
        """
        # Default base_state to state if not provided
        if base_state is None:
            base_state = state
            
        # Determine batch size from state
        if isinstance(state, torch.Tensor):
            batch_size = state.shape[0]
        elif isinstance(state, list):
            batch_size = len(state)
        else:
            # If state is a single item (e.g., string), treat as batch size 1
            batch_size = 1
            state = [state]
            
        # Decode states to text
        state_texts = self._decode_to_text(state)
        base_state_texts = self._decode_to_text(base_state)

        # Ensure base_state_texts matches batch size
        if len(base_state_texts) != batch_size:
            print(f"Warning: base_state_texts length ({len(base_state_texts)}) doesn't match batch_size ({batch_size}). Padding...")
            # Extend or truncate base_state_texts to match batch_size
            if len(base_state_texts) < batch_size:
                base_state_texts.extend([base_state_texts[0]] * (batch_size - len(base_state_texts)))
            else:
                base_state_texts = base_state_texts[:batch_size]
        
        # Add debug logging to track state and base_state
        if self.debug_mode:
            print(f"\nDEBUG - State samples: {[s[:50] + '...' if isinstance(s, str) and len(s) > 50 else s for s in state_texts[:2]]}")
            print(f"DEBUG - Base state samples: {[b[:50] + '...' if isinstance(b, str) and len(b) > 50 else b for b in base_state_texts[:2]]}")
                
        # Get token-wise harm values for all states at once
        token_harm_values = self._get_token_harm_values(state_texts)
        
        # Get topics from highest-harm tokens
        topics = self._extract_topics_from_tokens(state_texts, token_harm_values)
        
        # Pre-allocate lists for batch processing results
        templated_responses = [[] for _ in range(batch_size)]
        templated_tokens = [[] for _ in range(batch_size)]
        
        # Process each template for the entire batch at once
        for template_info in self.templates:
            template_id = template_info.get("id", -999)  # Default if id not present
            template = template_info.get("template", "I cannot respond to that request.")  # Default if template not present
            
            # Format template for each state in the batch
            batch_formatted_responses = []
            for i in range(batch_size):
                # Format template with topic and question using the helper method
                formatted_response = self._generate_templated_question(
                    template, 
                    topics[i], 
                    base_state_texts[i]
                )
                batch_formatted_responses.append(formatted_response)
                templated_responses[i].append(formatted_response)
            
            # Tokenize all responses for this template in a single batch
            try:
                batch_tokenized = self.tokenizer(
                    batch_formatted_responses, 
                    return_tensors='pt', 
                    padding=True, 
                    add_special_tokens=self.manually_add_bos
                ).input_ids
                
                # Store tokenized responses for each state
                for i in range(batch_size):
                    templated_tokens[i].append(batch_tokenized[i:i+1])  # Keep the batch dimension
            except Exception as e:
                print(f"Error tokenizing template {template_id}: {e}")
                # Provide a fallback tokenized output
                for i in range(batch_size):
                    templated_tokens[i].append(torch.zeros((1, 5), dtype=torch.long))  # Fallback tensor
        
        # IMPORTANT: Create consistent states for base model
        # We need to create a base_state with the same structure/format as state
        # Use the decoded texts but maintain the original format (tensor, list, etc.)
        consistent_base_state = base_state
        if isinstance(state, torch.Tensor) and not isinstance(base_state, torch.Tensor):
            # If state is tensor but base_state isn't, convert base_state to tensor
            consistent_base_state = self.tokenizer(
                base_state_texts, 
                return_tensors='pt', 
                padding=True
            ).input_ids
        elif isinstance(state, list) and not all(isinstance(s, type(b)) for s, b in zip(state, base_state)):
            # If state is a list but elements don't match base_state elements, convert
            consistent_base_state = base_state_texts
            
        # Add debug print before generating with base model
        if self.debug_mode:
            print(f"DEBUG - Calling super().act() with:")
            print(f"  - state type: {type(state)}, length: {len(state) if hasattr(state, '__len__') else 'N/A'}")
            print(f"  - consistent_base_state type: {type(consistent_base_state)}, length: {len(consistent_base_state) if hasattr(consistent_base_state, '__len__') else 'N/A'}")
        
        # Get generated responses also, using a consistent base_state format
        gen_action, gen_act_tokens, gen_log_probs = super().act(
            state, 
            consistent_base_state,  # Use the consistently formatted base_state
            self.instructions, 
            enforce_single_question=True
        )
        
        # Add debug logging if mismatch is detected
        if self.debug_mode and any(isinstance(a, str) and 'sexual' in a.lower() for a in gen_action) and not any('sexual' in str(t).lower() for t in topics):
            print(f"DEBUG - MISMATCH DETECTED between topics and generated response!")
            print(f"  Topics: {topics[:2]}")
            print(f"  Sampled gen_action: {[a[:50] + '...' for a in gen_action[:2]]}")
        
        assert len(gen_action) == len(templated_responses)
        for i in range(batch_size): 
            templated_responses[i].append(gen_action[i])
            templated_tokens[i].append(gen_act_tokens[i].to(templated_tokens[i][0].device))

        log_probs = []
        for i in range(batch_size):
            padded_templates = self._pad_batch_acts(templated_tokens[i])
            tokenized_state = self.tokenizer(
                self.prepare_prompt([state[i]], consistent_base_state[i] if isinstance(consistent_base_state, list) else base_state[i], instructions=self.instructions, inject_perspective=False), 
                return_tensors='pt', 
                padding=True
            ).input_ids.squeeze(0)
            item_log_prob = self.actor.get_log_probs(tokenized_state.expand(padded_templates.shape[0], -1), padded_templates, require_grad=False, add_special_tokens=False)[0]
            log_probs.append(item_log_prob)

        assert len(log_probs) == len(templated_responses) == len(templated_tokens) == len(gen_log_probs) == len(gen_action) == len(gen_act_tokens) == batch_size
        return templated_responses, templated_tokens, log_probs
    
    def _decode_to_text(self, state_input):
        """
        Helper method to decode various input types to text strings.
        
        Args:
            state_input: Input state (can be tensor, string, list of strings, etc.)
            
        Returns:
            List of text strings
        """
        # Handle tensor input
        if isinstance(state_input, torch.Tensor):
            return self.tokenizer.batch_decode(state_input, skip_special_tokens=False)
            
        # Handle list input
        elif isinstance(state_input, list):
            # If list elements are strings, return as is
            if all(isinstance(item, str) for item in state_input):
                return state_input
            # If list elements are tensors, decode each one
            elif all(isinstance(item, torch.Tensor) for item in state_input):
                return [self.tokenizer.decode(item, skip_special_tokens=False) for item in state_input]
            # Mixed list - convert each item to string as best we can
            else:
                return [str(item) if not isinstance(item, torch.Tensor) 
                       else self.tokenizer.decode(item, skip_special_tokens=False) 
                       for item in state_input]
                
        # Handle single string input
        elif isinstance(state_input, str):
            return [state_input]
            
        # Handle other types by converting to string
        else:
            return [str(state_input)]
    
    def _get_token_harm_values(self, state_texts):
        """
        Get token-wise harm values for each state, processing the whole batch at once.
        
        Args:
            state_texts: List of state texts
            
        Returns:
            List of token-wise harm values for each state in the batch
        """
        if not state_texts:
            return []
            
        # Tokenize the entire batch at once
        batch_tokens = self.tokenizer(
            state_texts, 
            return_tensors='pt', 
            add_special_tokens=False,
            padding=True
        )
        
        # Move to GPU
        input_ids = batch_tokens.input_ids.to('cuda')
        attention_mask = batch_tokens.attention_mask.to('cuda')
        
        # Get token-wise harm values from harm critic
        with torch.no_grad():
            _, harm_token_scores = self.harm_critic(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_token_scores=True
            )
            
            # Initialize list to store token values for each batch item
            batch_token_values = []
            
            # Check the shape of harm_token_scores to handle different model outputs
            if harm_token_scores.dim() == 2:  # Shape: [batch_size, seq_len]
                # For each item in the batch
                for i in range(input_ids.shape[0]):
                    # Get only valid tokens (not padding)
                    mask = attention_mask[i].bool()
                    valid_scores = harm_token_scores[i][mask]
                    token_values = valid_scores.cpu().numpy().tolist()
                    
                    # Ensure we have a list even for single tokens
                    if not isinstance(token_values, list):
                        token_values = [token_values]
                        
                    batch_token_values.append(token_values)
                    
            elif harm_token_scores.dim() == 3:  # Shape: [batch_size, seq_len, vocab_size]
                # This would be the case if the model outputs scores for each token in vocabulary
                # We need to gather the scores for the actual tokens in input_ids
                for i in range(input_ids.shape[0]):
                    # Get only valid tokens (not padding)
                    mask = attention_mask[i].bool()
                    # Get the token ids for this sequence
                    seq_tokens = input_ids[i][mask]
                    # Get scores for these specific tokens
                    token_scores = harm_token_scores[i][mask]
                    
                    # If we have scores for each vocab item, gather the scores for the actual tokens
                    if token_scores.size(-1) > 1:  # Has vocab dimension
                        token_scores = token_scores.gather(
                            -1, seq_tokens.unsqueeze(-1)
                        ).squeeze(-1)
                        
                    token_values = token_scores.cpu().numpy().tolist()
                    
                    # Ensure we have a list even for single tokens
                    if not isinstance(token_values, list):
                        token_values = [token_values]
                        
                    batch_token_values.append(token_values)
            else:
                # Unexpected shape, try to handle it gracefully
                print(f"Warning: Unexpected shape for harm_token_scores: {harm_token_scores.shape}")
                # Try to squeeze and reshape as needed
                reshaped_scores = harm_token_scores.view(input_ids.shape[0], -1)
                
                for i in range(input_ids.shape[0]):
                    # Get only valid tokens (not padding)
                    mask = attention_mask[i].bool()
                    seq_len = mask.sum().item()
                    
                    # Ensure we don't exceed the reshaped scores dimensions
                    if seq_len <= reshaped_scores.size(1):
                        valid_scores = reshaped_scores[i][:seq_len]
                    else:
                        valid_scores = reshaped_scores[i]
                        
                    token_values = valid_scores.cpu().numpy().tolist()
                    
                    # Ensure we have a list even for single tokens
                    if not isinstance(token_values, list):
                        token_values = [token_values]
                        
                    batch_token_values.append(token_values)
            
            return batch_token_values
    
    def _extract_topics_from_tokens(self, state_texts, token_harm_values):
        """
        Extract topics from the tokens with highest harm values.
        Extract top 3 distinct nucleus tokens along with surrounding context.
        Merges overlapping spans to form coherent phrases.
        
        Args:
            state_texts: List of state texts
            token_harm_values: List of token-wise harm values for each state
            
        Returns:
            List of topics extracted from highest-harm tokens with context
        """
        topics = []
        nucleus_size = 3  # Number of nucleus tokens to extract
        context_size = 2  # Context window size on each side
        
        for i, text_item in enumerate(state_texts):
            try:
                # Convert non-string items to string
                if not isinstance(text_item, str):
                    if isinstance(text_item, list):
                        text = ' '.join([str(t) for t in text_item])
                    else:
                        text = str(text_item)
                    print(f"Converting non-string item to string: {type(text_item)} -> '{text[:30]}...'")
                else:
                    text = text_item
                
                token_values = token_harm_values[i]
                
                # Tokenize to get mapping from token ids to text
                tokens = self.tokenizer(text, add_special_tokens=False)
                
                # Ensure token_values and tokens.input_ids have the same length
                if len(token_values) != len(tokens.input_ids):
                    print(f"Warning: Token values length ({len(token_values)}) doesn't match tokens length ({len(tokens.input_ids)}). Truncating...")
                    token_values = token_values[:len(tokens.input_ids)] if len(token_values) > len(tokens.input_ids) else token_values + [0.0] * (len(tokens.input_ids) - len(token_values))
                
                # Get the indices of top N nucleus tokens by harm value
                if len(token_values) > nucleus_size:
                    nucleus_indices = np.argsort(token_values)[-nucleus_size:]
                else:
                    nucleus_indices = np.argsort(token_values)
                
                # Create spans around each nucleus token
                spans = []
                for idx in nucleus_indices:
                    # Define the span (context window) around the nucleus token
                    start_idx = max(0, idx - context_size)  # Context tokens before
                    end_idx = min(len(tokens.input_ids) - 1, idx + context_size)  # Context tokens after
                    spans.append((start_idx, end_idx))
                
                # Merge overlapping spans to create coherent phrases
                merged_spans = self._merge_overlapping_spans(spans)
                
                extracted_spans = []
                for start_idx, end_idx in merged_spans:
                    # Get the token IDs for this span
                    span_token_ids = tokens.input_ids[start_idx:end_idx+1]
                    
                    # Decode the span into text
                    span_text = self.tokenizer.decode(span_token_ids)
                    
                    # Add to our extracted spans
                    extracted_spans.append(span_text)
                
                # Join all extracted spans with spaces
                topic = " ".join(extracted_spans).strip()
                
                # Clean up the topic by removing duplicate whitespace
                topic = " ".join(topic.split())
                
                # If topic is empty or too short, use a fallback
                if not topic or len(topic) < 3:
                    # Extract some words from the original text as fallback
                    words = text.split()
                    topic = " ".join(words[:min(5, len(words))])
                    
                topics.append(topic)
            except Exception as e:
                print(f"Error extracting topics from tokens: {e}")
                import traceback
                traceback.print_exc()
                
                # Provide a default topic in case of error
                if isinstance(text_item, str):
                    words = text_item.split()
                    fallback_topic = " ".join(words[:min(3, len(words))])
                elif isinstance(text_item, list):
                    fallback_topic = " ".join([str(item) for item in text_item[:3]])
                else:
                    fallback_topic = "unknown topic"
                
                topics.append(fallback_topic)
        
        return topics
        
    def _merge_overlapping_spans(self, spans):
        """
        Merge overlapping spans to create coherent phrases.
        
        Args:
            spans: List of (start, end) tuples representing token spans
            
        Returns:
            List of merged (start, end) spans
        """
        if not spans:
            return []
            
        # Sort spans by starting position
        sorted_spans = sorted(spans, key=lambda span: span[0])
        
        merged = [sorted_spans[0]]
        
        for current in sorted_spans[1:]:
            previous = merged[-1]
            
            # If current span overlaps with the previous one, merge them
            if current[0] <= previous[1] + 1:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
                
        return merged
    
    def rollout(self, num_steps):
        """
        Perform a BT-style rollout with templated responses.
        
        Args:
            num_steps: Number of steps in the rollout
            
        Returns:
            Rollout data for training
        """
        states = []
        base_states = []
        actions = []
        action_tokens = []
        action_log_probs = []
        rewards = []
        bt_preferences = []
        next_states = []
        dones = []
        template_indices = []
        harm_rews = []
        actual_next_states = []
        
        # Get batch size from environment
        batch_size = self.env.batch_size if hasattr(self.env, 'batch_size') else 1
        
        # Debug: Print number of templates being used
        if self.debug_mode:
            print(f"DEBUG - Using {len(self.templates)} templates in rollout")
            print(f"DEBUG - Templates: {[t.get('id', -1) for t in self.templates[:5]]}... (showing first 5)")
        
        # Store initial base states (original adversarial questions)
        state = self.env.reset()
        current_base_states = [bs for bs in self.env.base_state]
            
        # Execute rollout steps
        for step_idx in range(int(num_steps//batch_size)):
            # Track which conversations are already done before this step
            already_done = self.env.dones.clone() if hasattr(self.env, 'dones') else torch.zeros(batch_size).bool()
            
            # Generate templated responses using the model
            print("Memory before act: ", torch.cuda.memory_allocated()/1024/1024**2, "GB")
            templated_responses, templated_tokens, log_probs = self.act(state, self.env.base_state)
            print("Memory after act: ", torch.cuda.memory_allocated()/1024/1024**2, "GB")
            
            # Debug: Print number of responses per state
            if self.debug_mode:
                print(f"DEBUG - Received {len(templated_responses[0])} templated responses per state")
            
            batch_ppo_rewards = []
            batch_actions = []
            batch_action_tokens = []
            batch_bt_preferences = []
            batch_template_idx = []
            batch_log_probs = []
            batch_harm_rews = []
            # Process each state in the batch
            for i in range(batch_size):
                # Skip already done conversations
                if already_done[i]:
                    batch_actions.append("")
                    batch_action_tokens.append([None])
                    batch_ppo_rewards.append([None])
                    batch_bt_preferences.append([None])
                    batch_template_idx.append([None])
                    batch_log_probs.append([None])
                    batch_harm_rews.append([None])
                    continue
                
                state_templates = templated_responses[i]
                state_template_tokens = templated_tokens[i]
                
                # Debug check: Compare the expected number of templates with what we received
                if self.debug_mode and len(state_templates) != len(self.templates) + 1:  # +1 for generated response
                    print(f"WARNING - Expected {len(self.templates) + 1} templates but got {len(state_templates)}")
                
                # Get target responses (s') for each templated question (a)
                formatted_templates = [[{'role': 'user', 'content': template}] for template in state_templates]
                target_responses = self.env.get_target_responses(formatted_templates)
                
                # Get harm rewards for target responses J(s')
                harm_rewards_s_prime = self.env.score_safety(target_responses, state_templates, [self.env.base_state[i] for _ in range(len(state_templates))])
                
                # Get harm rewards for the template actions themselves J(a)
                # Assuming score_safety can score the agent's output directly
                harm_rewards_a = self.env.score_safety(state_templates, state_templates, [self.env.base_state[i] for _ in range(len(state_templates))])

                # Debug: Log harm rewards
                if self.debug_mode:
                    print(f"DEBUG - Harm rewards: {[float(r.item()) for r in harm_rewards_s_prime[:3]]}...")
                
                # Check for refusals based on target responses
                refusals = self.env.check_refusals(target_responses)
                
                # Create multi-option BT preference data including ppo_reward
                template_data = []
                # Handle rejection template (template 0)
                template_data.append({
                    'template_idx': 0,
                    'template_tokens': state_template_tokens[0].squeeze(),
                    'harm_score_s_prime': torch.tensor([-1]), # J(s') - effectively N/A
                    'harm_score_a': harm_rewards_a[0].reshape(-1), # J(a)
                    'ppo_reward': torch.tensor([-1]) - harm_rewards_a[0].reshape(-1), # R = J(s') - J(a)
                    'is_refusal': True,
                    'log_prob': log_probs[i][0].unsqueeze(0),
                })
                # Handle other templates (1 to N + generated)
                for j in range(1, len(state_templates)):
                    ppo_reward = harm_rewards_s_prime[j].reshape(-1) - harm_rewards_a[j].reshape(-1)
                    template_data.append({
                        'template_idx': j,
                        'template_tokens': state_template_tokens[j].squeeze(),
                        'harm_score_s_prime': harm_rewards_s_prime[j].reshape(-1), # Store J(s')
                        'harm_score_a': harm_rewards_a[j].reshape(-1), # Store J(a)
                        'ppo_reward': ppo_reward, # Store R = J(s') - J(a)
                        'is_refusal': refusals[j] == 1,
                        'log_prob': log_probs[i][j].unsqueeze(0),
                    })

                # Rank templates by harm score J(s') for selection (optional, could use ppo_reward)
                # Using harm_score_s_prime for ranking as before, but could change
                template_data.sort(key=lambda x: x['harm_score_s_prime'], reverse=True)
                
                # Find the lowest-ranked template that didn't produce a refusal (idx_z). Default to the rejection template
                idx_z = 0
                for item in reversed(template_data):
                    if not item['is_refusal']:
                        idx_z = item['template_idx']
                        break
                
                # Store the multi-option data including ppo_reward
                multi_bt_data = (
                    [item['template_idx'] for item in template_data],
                    [item['template_tokens'] for item in template_data],
                    [item['harm_score_s_prime'] for item in template_data], # Keep J(s') for ranking/info
                    [item['ppo_reward'] for item in template_data], # Add ppo_reward
                    [item['log_prob'] for item in template_data],
                    [item['is_refusal'] for item in template_data],
                    idx_z
                )
                
                # Find the best template based on J(s') ranking (or could use ppo_reward)
                valid_indices = [
                    item['template_idx'] for item in template_data
                    if item['template_idx'] != self.rejection_template_idx and not item['is_refusal']
                ]
                
                if valid_indices:
                    # Find the one with the highest J(s') among valid ones
                    best_s_prime_score = -float('inf')
                    selected_idx = self.rejection_template_idx # Default to rejection
                    selected_reward = torch.tensor(-1.0) # Default ppo_reward for rejection

                    for item in template_data:
                        if item['template_idx'] in valid_indices:
                            if item['harm_score_s_prime'] > best_s_prime_score:
                                best_s_prime_score = item['harm_score_s_prime']
                                # Only select if J(s') is above threshold
                                if best_s_prime_score > 0.2: # Threshold for accepting non-rejection
                                    selected_idx = item['template_idx']
                                    selected_reward = item['ppo_reward'] # Get the corresponding PPO reward
                    else:
                                     # If best score is below threshold, stick to rejection
                        selected_idx = self.rejection_template_idx
                        rejection_data = next(t for t in template_data if t['template_idx'] == self.rejection_template_idx)
                        selected_reward = rejection_data['ppo_reward']
                else:
                    # If no valid templates, use rejection
                    selected_idx = self.rejection_template_idx
                    rejection_data = next(item for item in template_data if item['template_idx'] == self.rejection_template_idx)
                    selected_reward = rejection_data['ppo_reward'] # PPO reward for rejection
                
                # Append data for the *selected* action to batch lists
                batch_actions.append(state_templates[selected_idx])
                batch_action_tokens.append(state_template_tokens[selected_idx])
                batch_ppo_rewards.append(selected_reward) # Store PPO reward R
                batch_bt_preferences.append(multi_bt_data) # Store all template data including ppo_reward
                batch_template_idx.append(selected_idx)
                batch_log_probs.append(log_probs[i][selected_idx].unsqueeze(0))
                selected_harm_s_prime = next(item['harm_score_s_prime'] for item in template_data if item['template_idx'] == selected_idx)
                batch_harm_rews.append(selected_harm_s_prime) # Store J(s')
                
            # Take a step in the environment with the selected actions
            step_next_state, _, _, step_done = self.env.step(batch_actions)
            
            # Store rollout data
            states.append(state)
            base_states.append(current_base_states) # Base state corresponding to 'state'
            actions.append(batch_actions)
            action_tokens.append(batch_action_tokens)
            rewards.append(batch_ppo_rewards) # Store the PPO rewards R
            bt_preferences.append(batch_bt_preferences)
            actual_next_states.append(step_next_state) # ADDED: Store the actual next state
            dones.append(step_done)
            template_indices.append(batch_template_idx)
            action_log_probs.append(batch_log_probs)
            harm_rews.append(batch_harm_rews) # Store J(s')
            # Set state for next iteration
            state = step_next_state
            
            # Check if all conversations are done
            if step_done.all():
                print(f"All conversations completed after {step_idx + 1} steps. Resetting environment.")
                # Reset the environment to get new conversations if we need more steps
                if (step_idx + 1) * batch_size < num_steps:
                    state = self.env.reset()
                    # Clone base state after reset
                    current_base_states = [bs for bs in self.env.base_state]
                else:
                    # We've collected enough data, so we can break
                    break
        
        # Prepare data for flattening
        conversation_data = {
            'states': [[] for _ in range(batch_size)],
            'actual_next_states': [[] for _ in range(batch_size)],
            'acts': [[] for _ in range(batch_size)],
            'log_probs': [[] for _ in range(batch_size)],
            'rews': [[] for _ in range(batch_size)],
            'dones': [[] for _ in range(batch_size)],
            'base_states': [[] for _ in range(batch_size)],
            'action_help_rews': [[] for _ in range(batch_size)],
            'compliance_rews': [[] for _ in range(batch_size)],
            'bt_preferences': [[] for _ in range(batch_size)],
            'template_indices': [[] for _ in range(batch_size)],
            'harm_rews': [[] for _ in range(batch_size)]
        }

        # Restructure data for flattening (organize by batch item)
        for step_idx in range(len(states)):
            for batch_idx in range(batch_size):
                # Check if this batch item has data for this step
                if batch_idx < len(states[step_idx]) and batch_idx < len(actual_next_states[step_idx]) and actions[step_idx][batch_idx] != "":
                    conversation_data['states'][batch_idx].append(states[step_idx][batch_idx])
                    conversation_data['actual_next_states'][batch_idx].append(actual_next_states[step_idx][batch_idx])
                    conversation_data['acts'][batch_idx].append(action_tokens[step_idx][batch_idx])
                    conversation_data['log_probs'][batch_idx].append(action_log_probs[step_idx][batch_idx])
                    conversation_data['rews'][batch_idx].append(rewards[step_idx][batch_idx])
                    conversation_data['dones'][batch_idx].append(dones[step_idx][batch_idx])
                    conversation_data['base_states'][batch_idx].append(base_states[step_idx][batch_idx])
                    
                    if step_idx < len(bt_preferences) and batch_idx < len(bt_preferences[step_idx]):
                        conversation_data['bt_preferences'][batch_idx].append(bt_preferences[step_idx][batch_idx])
                    
                    if step_idx < len(template_indices) and batch_idx < len(template_indices[step_idx]):
                        conversation_data['template_indices'][batch_idx].append(template_indices[step_idx][batch_idx])
                    
                    if step_idx < len(harm_rews) and batch_idx < len(harm_rews[step_idx]):
                        conversation_data['harm_rews'][batch_idx].append(harm_rews[step_idx][batch_idx])
            
            # Initialize dummy values for compliance and help rewards (not used in BT agent)
            conversation_data['action_help_rews'][batch_idx].append(torch.tensor(0.0))
            conversation_data['compliance_rews'][batch_idx].append(torch.tensor(0.0))

        # Flatten data structures while maintaining chronological order
        batch_state = []
        batch_actual_next_state = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_dones = []
        batch_base_state = []
        batch_bt_preferences = []
        batch_template_indices = []
        batch_harm_rews = []
        batch_action_help_rews = []
        batch_compliance_rews = []

        # Flatten the conversation data while maintaining chronological order
        for i in range(batch_size):
            states_len = len(conversation_data['states'][i])
            if states_len > 0:  # Only process non-empty conversations
                batch_state.extend(conversation_data['states'][i])
                batch_actual_next_state.extend(conversation_data['actual_next_states'][i])
                batch_acts.extend(conversation_data['acts'][i])
                batch_log_probs.extend(conversation_data['log_probs'][i])
                batch_rews.extend(conversation_data['rews'][i])
                batch_dones.extend(conversation_data['dones'][i])
                batch_bt_preferences.extend(conversation_data['bt_preferences'][i])
                batch_template_indices.extend(conversation_data['template_indices'][i])
                batch_harm_rews.extend(conversation_data['harm_rews'][i])
                
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
                
                # Add help and compliance rewards (dummy values for BT agent)
                batch_action_help_rews.extend(conversation_data['action_help_rews'][i])
                batch_compliance_rews.extend(conversation_data['compliance_rews'][i])
        
        # Skip further processing if no data was collected
        if len(batch_state) == 0:
            print(conversation_data)
            raise RuntimeError("No valid data collected in rollout - all conversations were done")
        
        # Tokenize and process the collected data
        batch_state_tokenized = self.tokenizer(self.prepare_prompt(batch_state, batch_base_state, self.instructions, inject_perspective=False), 
                                    return_tensors='pt', padding=True,
                                    add_special_tokens=self.manually_add_bos).input_ids.squeeze()
        batch_actual_next_state_tokenized = self.tokenizer(self.prepare_prompt(batch_actual_next_state, batch_base_state, self.instructions, inject_perspective=False), 
                                        return_tensors='pt', padding=True,
                                        add_special_tokens=self.manually_add_bos).input_ids.squeeze()
        batch_base_state_tokenized = self.tokenizer(batch_base_state, return_tensors='pt', padding=True).input_ids.squeeze()
        batch_acts_padded = self._pad_batch_acts(batch_acts)
        batch_rews_tensor = torch.tensor(batch_rews).squeeze()
        batch_dones_tensor = torch.tensor(batch_dones).squeeze()

        batch_action_help_rews_tensor = torch.tensor(batch_action_help_rews).squeeze()
        batch_compliance_rews_tensor = torch.tensor(batch_compliance_rews).squeeze()

        batch_log_probs_padded = self._pad_batch_log_probs(batch_log_probs, batch_acts_padded)
        batch_template_indices_tensor = torch.tensor(batch_template_indices).squeeze()
                
        # Process harm rewards
        batch_harm_rews_tensor = torch.stack(batch_harm_rews).squeeze() if batch_harm_rews else torch.zeros_like(batch_rews_tensor)
        # Compile rollout data
        rollout_data = {
            'batch_state': batch_state_tokenized,
            'batch_base_state': batch_base_state_tokenized,
            'batch_action_strings': actions,
            'batch_acts': batch_acts_padded,
            'batch_rews': batch_rews_tensor,
            'batch_actual_next_states': batch_actual_next_state_tokenized,
            'batch_dones': batch_dones_tensor,
            'batch_bt_preferences': batch_bt_preferences,
            'batch_template_indices': batch_template_indices_tensor,
            'batch_log_probs': batch_log_probs_padded,
            'batch_compliance_rews': batch_compliance_rews_tensor,
            'batch_action_help_rews': batch_action_help_rews_tensor,
            'batch_harm_rews': batch_harm_rews_tensor
        }
        rollout_data = type('Rollout', (object,), rollout_data)()
        
        return rollout_data
    
    def _train_on_batch(self, batch_data, num_epochs=None):
        """
        Train on a batch of data using PPO loss for each template.
        
        Args:
            batch_data: Batch of rollout data containing template information
            num_epochs: Number of training epochs (used by PPO inner loop)
            
        Returns:
            Training statistics
        """
        # NOTE: super()._train_on_batch is commented out as we are replacing 
        # the training logic for BTagent to focus on template PPO.
        # stats = super()._train_on_batch(batch_data, num_epochs)
        
        # Perform PPO updates using template data
        template_ppo_loss, approx_kl, policy_loss, value_loss = self._compute_template_ppo_loss(
            batch_data, 
            num_epochs=self.epochs if num_epochs is None else num_epochs
        )
        self._log_summary()
        
        # Prepare stats dictionary (add more stats as needed)
        stats = {
            'template_ppo_loss': template_ppo_loss.item(),
            'approx_kl': approx_kl,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        
        return stats
    
    def _compute_template_ppo_loss(self, batch_data, num_epochs):
        """
        Compute PPO loss for templated actions based on their harm rewards,
        incorporating the value of the actual next state (V(s')) for future utility.
        Trains the critic to predict token-level values.
        
        Args:
            batch_data: Rollout data including states, templates, PPO rewards (R=J(s')-J(a)),
                        old log probs, actual next states, and done flags.
            num_epochs: Number of optimization epochs for PPO.
            
        Returns:
            Tuple: (total_loss, approx_kl, policy_loss, value_loss)
        """
        # Extract data from batch_data
        template_preferences = batch_data.batch_bt_preferences
        batch_states = batch_data.batch_state
        batch_actual_next_states = batch_data.batch_actual_next_states
        batch_dones = batch_data.batch_dones
        gamma = self.gamma # Discount factor
        batch_size = len(batch_states)
        
        total_loss_avg = 0
        approx_kl_avg = 0
        policy_loss_avg = 0
        value_loss_avg = 0
        total_rews_avg = 0
        num_templates_processed = 0 # Total templates across all states in the batch

        # --- Precompute Sequence-Level V(s') --- 
        with torch.no_grad():
            self.harm_critic.to(self.actor.device)
            #self.help_critic.to(self.actor.device)
            next_states_device = batch_actual_next_states.to(self.actor.device)
            next_state_attention_mask = torch.where(next_states_device != self.tokenizer.pad_token_id, 1, 0)
            # Get token-level values for next states
            next_state_token_values = self.harm_critic(next_states_device, attention_mask=next_state_attention_mask)
            # Sum token values to get sequence-level V(s') applying mask
            next_state_sequence_values = (next_state_token_values * next_state_attention_mask).sum(dim=1)
            # Apply done mask
            next_state_sequence_values = next_state_sequence_values * (1 - batch_dones.to(self.actor.device).float())

        # PPO inner loop for multiple epochs
        for epoch in range(num_epochs):
            epoch_total_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_kl_divs = []
            processed_in_epoch = 0 # Templates processed in this specific epoch for averaging
            
            # Iterate through each step/state in the flattened batch
            for i in range(len(template_preferences)):
                if not template_preferences[i]: 
                    continue

                state_tokens = batch_states[i] # Tokenized state for this step
                next_value_seq = next_state_sequence_values[i] # Get precomputed sequence-level V(s')
                
                # Unpack template data for this state
                indices, tokens_list, _, ppo_rewards, log_probs_old, _, _ = template_preferences[i]

                if not tokens_list:
                    continue
                
                for minibatch_start in range(0, len(tokens_list), self.minibatch_size):
                    minibatch_end = min(minibatch_start + self.minibatch_size, len(tokens_list))
                    num_templates_for_state = minibatch_end - minibatch_start
                    template_actions = self._pad_batch_acts(tokens_list[minibatch_start:minibatch_end]).to(self.actor.device)
                    template_ppo_rewards = torch.stack(ppo_rewards[minibatch_start:minibatch_end]).to(self.actor.device).float().squeeze() # R = J(s') - J(a)
                    template_old_log_probs = torch.stack(log_probs_old[minibatch_start:minibatch_end]).squeeze().to(self.actor.device)
                
                    # Expand state tokens for processing all templates of this state
                    state_expanded = state_tokens.unsqueeze(0).expand(num_templates_for_state, -1).to(self.actor.device)
                    state_attention_mask = torch.where(state_expanded != self.tokenizer.pad_token_id, 1, 0)

                    # --- Policy Evaluation --- 
                    curr_log_probs, entropy = self.actor.get_log_probs(
                        state_expanded, template_actions, require_grad=True, add_special_tokens=False
                    )
                    # Average log_probs over action sequence length for policy loss calculation
                    action_lengths = torch.where(template_actions != self.tokenizer.pad_token_id, 1, 0).sum(dim=1)
                    action_lengths = torch.clamp(action_lengths, min=1) # Avoid division by zero
                    avg_curr_log_probs = curr_log_probs.sum(dim=1) / action_lengths
                    avg_old_log_probs = template_old_log_probs.sum(dim=1) / action_lengths

                    # --- Value Function Evaluation (Token Level) --- 
                    if not hasattr(self, 'harm_critic') or self.harm_critic is None:
                        raise AttributeError("Harm critic is not initialized in BTagent")
                    self.harm_critic.to(self.actor.device)
                    # Get token-level value predictions V_t(s) for the state
                    current_token_values = self.harm_critic(state_expanded, attention_mask=state_attention_mask) # Shape: [num_templates, state_seq_len]
                    
                    # Calculate sequence-level V(s) by summing token values
                    current_sequence_values = (current_token_values * state_attention_mask).sum(dim=1)
                    
                    # --- Advantage and Target Calculation --- 
                    # Sequence-Level TD Target: R + gamma * V(s')
                    scalar_target = template_ppo_rewards + gamma * next_value_seq
                    
                    # Sequence-Level Advantage: A = TD_Target - V(s)
                    sequence_advantages = (scalar_target - current_sequence_values.detach()).float()
                    sequence_advantages = (sequence_advantages - sequence_advantages.mean()) / (sequence_advantages.std() + 1e-8)
                    
                    # --- Loss Calculation --- 
                    # Policy Loss (using sequence-level advantage)
                    log_ratio = avg_curr_log_probs - avg_old_log_probs
                    
                    # --- Debugging: Check log_ratio before exp ---
                    if torch.isnan(log_ratio).any() or torch.isinf(log_ratio).any():
                        print(f"WARNING in _compute_template_ppo_loss (BTagent.py): NaN or Inf detected in log_ratio!")
                        print(f"  NaNs: {torch.isnan(log_ratio).sum().item()}, Infs: {torch.isinf(log_ratio).sum().item()}")
                        # Handle NaN/Inf, e.g., replace with 0
                        log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    max_log_ratio = 20.0 # Define a reasonable upper bound
                    if (log_ratio > max_log_ratio).any():
                        log_ratio = torch.clamp(log_ratio, max=max_log_ratio)
                    # ----------------------------------------------

                    ratios = torch.exp(log_ratio)
                    surr1 = ratios * sequence_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * sequence_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value Loss (using token-level target)
                    # Create token-level target by distributing scalar_target
                    state_lengths = state_attention_mask.sum(dim=1, keepdim=True) # Shape [num_templates, 1]
                    state_lengths = torch.clamp(state_lengths, min=1) # Avoid division by zero
                    token_level_target = scalar_target.unsqueeze(1) / state_lengths # Distribute target uniformly
                    token_level_target = token_level_target.expand_as(current_token_values) # Expand to match token values shape
                    
                    # Calculate MSE loss only on non-padding tokens of the state
                    value_loss = F.mse_loss(current_token_values * state_attention_mask, token_level_target * state_attention_mask, reduction='sum') 
                    value_loss = value_loss / state_attention_mask.sum() # Average over non-padding tokens
                    
                    # Total Loss
                    entropy_loss = -self.ce_coef * entropy.mean() if entropy is not None else 0
                    total_template_loss = policy_loss + self.vf_coef * value_loss + entropy_loss
                    total_template_loss = total_template_loss / (batch_size * self.minibatch_size)
                    total_template_loss.backward()

                    self.actor_optim.zero_grad()
                    self.harm_critic_optim.zero_grad() 
                    #self.help_critic_optim.zero_grad()
                    # Accumulate losses for the epoch
                    epoch_total_loss += total_template_loss
                    epoch_policy_loss += policy_loss
                    epoch_value_loss += value_loss
                    processed_in_epoch += num_templates_for_state
                    total_rews_avg += template_ppo_rewards.mean().item()

                    # Calculate approximate KL divergence (using averaged log probs)
                    with torch.no_grad():
                        log_ratio = avg_curr_log_probs - avg_old_log_probs
                        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().item()
                        epoch_kl_divs.append(approx_kl)

                    #log stuff
                    self.logger['actor_losses'].append(total_template_loss.detach().cpu().numpy())
                    self.logger['batch_rews'].append(template_ppo_rewards.detach().cpu().numpy())
                    self.logger['harm_critic_losses'].append(value_loss.detach().cpu().numpy())

                    del state_expanded, state_attention_mask, template_actions, template_ppo_rewards, template_old_log_probs, current_token_values, token_level_target, current_sequence_values, sequence_advantages, scalar_target
                    torch.cuda.empty_cache()

            # --- End of Epoch: Perform Gradient Update --- 
            if processed_in_epoch > 0:
                avg_epoch_total_loss = epoch_total_loss / processed_in_epoch
                avg_epoch_policy_loss = epoch_policy_loss / processed_in_epoch
                avg_epoch_value_loss = epoch_value_loss / processed_in_epoch
                avg_epoch_kl = np.mean(epoch_kl_divs) if epoch_kl_divs else 0
                num_templates_processed += processed_in_epoch # Update total count across epochs

                if not hasattr(self, 'actor_optim'):
                    raise AttributeError("Actor optimizer not initialized in BTagent.")
                if not hasattr(self, 'harm_critic_optim'):
                    raise AttributeError("Harm critic optimizer not initialized in BTagent.")
                
                # Optional: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.harm_critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()  
                self.harm_critic_optim.step()
                #self.help_critic_optim.step()

            else:
                print("Warning: No templates processed in PPO epoch.")
                break 

        # --- After PPO epochs --- 
        # Return the average losses from the last successfully computed epoch
        final_total_loss = avg_epoch_total_loss if num_templates_processed > 0 else torch.tensor(0.0)
        final_policy_loss = avg_epoch_policy_loss if num_templates_processed > 0 else torch.tensor(0.0)
        final_value_loss = avg_epoch_value_loss if num_templates_processed > 0 else torch.tensor(0.0)
        final_approx_kl = avg_epoch_kl if num_templates_processed > 0 else 0

        return final_total_loss.detach(), final_approx_kl, final_policy_loss.detach(), final_value_loss.detach()
        
    def _generate_templated_question(self, template, topic, question):
        """
        Generate a templated question using format specifiers.
        
        Args:
            template: Template string with format specifiers
            topic: Topic string extracted from harmful tokens
            question: Original question
            
        Returns:
            Formatted question
        """
        try:
            # Try to format with both topic and question
            return template.format(topic=topic, question=question)
        except KeyError as e:
            # If any key is missing, handle gracefully
            if 'topic' not in template:
                return template.format(question=question)
            elif 'question' not in template:
                return template.format(topic=topic)
            else:
                # Fallback for unexpected format string
                return f"I'd like to discuss {topic} based on your question: {question}"
    
    
        