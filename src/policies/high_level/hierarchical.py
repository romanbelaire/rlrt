"""
Hierarchical policy implementations for text-based high-level guidance.

This module provides the LowLevelConditionedPolicy for hierarchical RL,
where low-level policies generate responses conditioned on high-level instructions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import random
import json
import re
import os


class LowLevelConditionedPolicy(nn.Module):
    """
    Low-level policy that generates responses conditioned on high-level abstract instructions.
    This policy interprets abstract strategies and generates appropriate concrete responses.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: object,
        instruction_prefix: str = "Strategy: ",
        context_separator: str = "\n\nContext: "
    ):
        """
        Args:
            base_model: Base language model for response generation
            tokenizer: Tokenizer for text processing
            instruction_prefix: Prefix for high-level instructions
            context_separator: Separator between instruction and context
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.instruction_prefix = instruction_prefix
        self.context_separator = context_separator
    
    def prepare_conditioned_context(
        self, 
        original_context: List[str], 
        instructions: List[str]
    ) -> List[str]:
        """
        Prepare context conditioned on high-level instructions.
        
        Args:
            original_context: Original conversation context
            instructions: High-level strategy instructions
            
        Returns:
            List of conditioned contexts
        """
        conditioned_contexts = []
        
        for context, instruction in zip(original_context, instructions):
            # Create conditioned context with instruction
            context_str = "\n".join(context) if isinstance(context, list) else str(context)
            
            # Combine instruction and context
            conditioned_context = f"{self.instruction_prefix}{instruction}{self.context_separator}{context_str}"
            conditioned_contexts.append(conditioned_context)
        
        return conditioned_contexts
    
    def act(
        self, 
        state: List[str], 
        instructions: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate responses conditioned on high-level instructions.
        
        Args:
            state: Current conversation state
            instructions: High-level strategy instructions
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            - Generated responses
            - Log probabilities
            - Entropy values
        """
        # Prepare conditioned context
        conditioned_contexts = self.prepare_conditioned_context(state, instructions)
        
        # Generate responses using the base model
        responses = []
        log_probs_list = []
        entropy_list = []
        
        for context in conditioned_contexts:
            # Tokenize input
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True)

            # Generate response with anti-gibberish measures
            with torch.no_grad():
                generation_kwargs = {
                    'input_ids': inputs.input_ids,
                    'max_new_tokens': max_new_tokens,
                    'do_sample': True,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'repetition_penalty': 1.2,  # discourage token repetition
                    'no_repeat_ngram_size': 3,  # ban 3-gram loops
                    'output_scores': True,  # enable score output
                    'return_dict_in_generate': True,  # structured output
                }
                
                # Only add sampling parameters when they have valid values
                if temperature:
                    generation_kwargs['temperature'] = temperature
                if 'top_p' in kwargs and kwargs['top_p']:
                    generation_kwargs['top_p'] = kwargs['top_p']
                else:
                    generation_kwargs['top_p'] = 0.9  # Default nucleus sampling
                
                # Add any additional kwargs
                for key, value in kwargs.items():
                    if key not in generation_kwargs and key != 'top_p':
                        generation_kwargs[key] = value
                
                outputs = self.base_model.generate(**generation_kwargs)
            
            # Decode response (handle both dict and tensor outputs)
            # Handle structured output from return_dict_in_generate=True
            generated_sequences = outputs.sequences
            
            # Extract only the generated part (exclude input tokens)
            generated_tokens = generated_sequences[0][inputs.input_ids.shape[1]:]
            
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
            
            # Compute log probabilities (simplified)
            log_probs = torch.tensor(0.0)  # Placeholder
            entropy = torch.tensor(0.0)    # Placeholder
            log_probs_list.append(log_probs)
            entropy_list.append(entropy)
        
        return responses, torch.stack(log_probs_list), torch.stack(entropy_list)
    
    def get_log_probs(
        self, 
        input_ids: torch.Tensor, 
        output_ids: torch.Tensor,
        instruction_text: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get log probabilities for generated tokens.
        
        Args:
            input_ids: Input token IDs
            output_ids: Output token IDs
            instruction_text: Optional instruction text for conditioning
            
        Returns:
            - Log probabilities
            - Entropy values
            - Perplexity values
        """
        # This is a simplified implementation
        # In practice, you would compute actual log probabilities from the model
        
        batch_size = input_ids.shape[0]
        seq_len = output_ids.shape[1]
        
        # Placeholder values
        log_probs = torch.zeros(batch_size, seq_len)
        entropy = torch.zeros(batch_size)
        perplexity = torch.zeros(batch_size)
        
        return log_probs, entropy, perplexity 