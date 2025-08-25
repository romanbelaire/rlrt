"""
Policy implementations for reinforcement learning agents.

This module provides implementations of the Policy interface for different types
of language models and optimization strategies.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
import numpy as np
from abc import ABC, abstractmethod
try:
    from transformers.generation.logits_processor import LogitsProcessor
except ImportError:
    # Fallback for newer versions of transformers
    from transformers import LogitsProcessor

import torch.nn.functional as F

from src.base import Policy, ModelBase, Environment
from src.environment import ClampLogitsProcessor
from src.utils import safe_tokenizer_call, build_chat, apply_activation_clamping, attach_permanent_activation_clamping


class RefusalSuppressionProcessor(LogitsProcessor):
    """
    Logits processor that suppresses tokens that would lead to refusal phrases.
    
    This processor reduces the probability of tokens that would start common
    refusal patterns like "I cannot", "I'm sorry", etc.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Flag phrases: innocent phrases that could lead to refusals
        self.flag_phrases = [
            "I",
            "I can", 
            "I will",
            "I'm",
            "I am",
            "I don't",
            "I do not",
            "I'm not",
            "I am not",
            "I do",
            "i",
            " I",
            " I can", 
            " I will",
            " I'm",
            " I am",
            " I don't",
            " I do not",
            " I'm not",
            " I am not",
            " I do",
            " i",
            "AI",
            "ai",
            "Ai",
            " AI",
            " ai",
            " Ai"
        ]
        
        # Refusal suffixes: words that turn flag phrases into refusals
        self.refusal_suffixes = [
            "'t",
            "'T",
            "cannot",
            "CANNOT",
            "Cannot",
            "can't", 
            "CAN'T",
            "Can't",
            " can't", 
            " CAN'T",
            " Can't",
            "cant",
            " cant",
            "not",
            "NOT",
            "Not",
            " not",
            " NOT",
            " Not",
            "sorry",
            "SORRY",
            "Sorry",
            " sorry",
            " SORRY",
            " Sorry",
            "apologize",
            "APOLOGIZE",
            "Apologize",
            " apologize",
            " APOLOGIZE",
            " Apologize",
            "uncomfortable",
            "UNCOMFORTABLE",
            "Uncomfortable",
            "unable",
            "UNABLE",
            "Unable",
            " unable",
            " UNABLE",
            " Unable",
            "decline",
            "DECLINE",
            "Decline",
            "refuse",
            "REFUSE",
            "Refuse",
            "won't",
            "WON'T",
            "Won't",
            "will not",
            "WILL NOT",
            "Will not",
            "shouldn't",
            "SHOULDN'T",
            "Shouldn't",
            "AI",
            "ai",
            "Ai",
            " AI",
            " ai",
            " Ai",
            "unwilling",
            "UNWILLING",
            "Unwilling",
            " unwilling",
            " UNWILLING",
            " Unwilling",
            " incapable",
            " INCAPABLE",
            " Incapable",
            "incapable",
            "INCAPABLE",
            "Incapable"            
        ]

        # Convert to token IDs
        self.flag_token_ids = {}
        self.refusal_token_ids = set()
        
        # Process flag phrases
        for phrase in self.flag_phrases:
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            if token_ids:
                self.flag_token_ids[tuple(token_ids)] = phrase
        
        # Process refusal suffixes
        for suffix in self.refusal_suffixes:
            token_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            if token_ids:
                # Add the first token of each suffix
                self.refusal_token_ids.add(token_ids[0])
        
        # Add the specific problematic tokens that were causing issues
        problems_list = [34360, 35755, 4250, 16869]
        for token_id in problems_list:
            self.refusal_token_ids.add(token_id)

    def __call__(self, input_ids, scores):
        """
        Suppress logits for tokens that would lead to refusal phrases.
        
        Args:
            input_ids: Current input token IDs [batch_size, seq_len]
            scores: Current logits [batch_size, vocab_size]
            
        Returns:
            Modified scores with refusal tokens suppressed
        """

        # Process each sequence in the batch
        for b in range(input_ids.size(0)):
            # Check if we have a flag phrase at the end for this sequence
            for flag_tokens, flag_phrase in self.flag_token_ids.items():
                # Check if the last tokens match this flag phrase (relaxed to last 3 tokens)
                seq_len = input_ids[b].size(0)
                if seq_len >= len(flag_tokens):
                    # Check exact match first
                    current_sequence = tuple(input_ids[b, -len(flag_tokens):].cpu().numpy())
                    if current_sequence == flag_tokens:
                        # We have a flag phrase, suppress refusal suffixes for this sequence only
                        suppression_factor = -50.0  # Much more aggressive suppression
                        
                        for token_id in self.refusal_token_ids:
                            if token_id < scores.shape[1]:
                                scores[b, token_id] += suppression_factor
                    
                        # Only process one flag phrase at a time for this sequence
                        break
                    else:
                        # Try relaxed matching: check if flag phrase tokens appear in last 3 tokens
                        if seq_len >= 5:
                            last_3_tokens = input_ids[b, -3:].cpu().numpy()
                            flag_token_set = set(flag_tokens)
                            last_3_set = set(last_3_tokens)
                            
                            # Check if any flag tokens are in the last 3 tokens
                            if flag_token_set.intersection(last_3_set):
                                # Suppress refusal tokens
                                suppression_factor = -25.0  # Less aggressive for relaxed matching
                                for token_id in self.refusal_token_ids:
                                    if token_id < scores.shape[1]:
                                        scores[b, token_id] += suppression_factor
                            
                                # Only process one flag phrase at a time for this sequence
                                break
        
        return scores

class AbstractPolicy(ModelBase, Policy, ABC):
    """
    Abstract base class for policy implementations using transformer models.
    
    This class provides common functionality for action generation and defines
    the interface that concrete implementations must follow.
    """
    
    def __init__(self, 
                 hf_checkpoint: str, 
                 tokenizer: PreTrainedTokenizer,
                 model: PreTrainedModel,
                 env: Optional[Environment] = None) -> None:
        """
        Initialize the abstract policy.
        
        Args:
            hf_checkpoint: Hugging Face model checkpoint path/name
            tokenizer: Tokenizer for the model
            model: Pre-loaded model instance
            env: Optional environment instance
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.hf_checkpoint = hf_checkpoint
        self.env = env
        
        # Use provided model instance
        self.model = model
        
        # Attach permanent activation clamping hooks for 4-bit quantization stability
        # Note: Configuration will be passed when the policy is initialized with config
        self.activation_hooks = []  # Will be set later with config
        
        # Define refusal patterns for detection (case-insensitive) - same as in agent_trainer.py
        self.refusal_patterns = [
            "I cannot", "I can't", "I'm not able to", "I won't", "I will not",
            "I'm unable to", "I don't think I can", "I shouldn't", "I'm not comfortable",
            "I apologize, but", "Sorry, but I can't", "I must decline", "I refuse to",
            "As an AI", "As a responsible AI", "I'm programmed not to", "It's not appropriate",
            "I don't feel comfortable", "That's not something I can", "I'd prefer not to",
            "I'm not designed to", "That goes against", "I have to decline"
        ]
        
    @property
    @abstractmethod
    def is_token_level_actor(self) -> bool:
        """Whether this is a token-level actor. Must be implemented by subclasses."""
        pass
        
    def _detect_refusal(self, text: str) -> bool:
        """
        Detect if the generated text contains refusal statements.
        
        Args:
            text: Generated text to check
            
        Returns:
            bool: True if refusal detected, False otherwise
        """
        text_lower = text.lower().strip()
        
        # Check for common refusal patterns (case-insensitive)
        for pattern in self.refusal_patterns:
            if pattern.lower().strip() in text_lower:
                return True
        
        # Check for very short responses (often indicate refusal)
        if len(text.strip()) < 10:
            return True
            
        # Check for responses that are just apologetic without content
        apology_words = ["sorry", "apologize", "regret", "unfortunately"]
        content_words = text_lower.split()
        if len(content_words) < 20 and any(word in text_lower for word in apology_words):
            return True
            
        return False
    
    def check_layer_grads(self) -> None:
        """Verify that some parameters have requires_grad=True"""
        grad_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        assert len(grad_params) > 0, "No trainable parameters found in the model!"

    def save(self, path: str) -> None:
        """Save the policy (model + tokenizer) to a directory.
        Args:
            path: Target directory where the model/tokenizer will be written.
        """
        import os
        os.makedirs(path, exist_ok=True)
        # Save transformer weights
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(os.path.join(path, "model"))
        else:
            # Fallback: torch save state_dict
            torch.save(self.model.state_dict(), os.path.join(path, "model_state.pt"))
        # Save tokenizer (may not exist for some synthetic policies)
        if self.tokenizer is not None and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        # Persist config info for easy reload
        meta = {
            "hf_checkpoint": self.hf_checkpoint,
            "class": self.__class__.__name__,
        }
        import json
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, tokenizer: Optional[PreTrainedTokenizer] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        """Load a policy from a directory previously saved with ``save``.
        Args:
            path: Directory that contains ``model`` and ``tokenizer`` sub-folders.
            tokenizer: Optionally override tokenizer when loading.
            device: Device to place the model on.
            **kwargs: Additional args forwarded to the initializer.
        Returns:
            An instance of the concrete ``AbstractPolicy`` subclass.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        import os, json
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            hf_checkpoint = meta.get("hf_checkpoint", None)
        else:
            hf_checkpoint = None
        # Load tokenizer
        if tokenizer is None:
            tokenizer_path = os.path.join(path, "tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if os.path.isdir(tokenizer_path) else None
        # Load model
        model_path = os.path.join(path, "model")
        if os.path.isdir(model_path):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": device},
                torch_dtype=torch.bfloat16
            )
        else:
            # Fallback: state_dict
            state_dict = torch.load(os.path.join(path, "model_state.pt"), map_location=device)
            assert hf_checkpoint is not None, "hf_checkpoint missing in meta.json for state_dict load"
            config = AutoConfig.from_pretrained(hf_checkpoint)
            model = AutoModelForCausalLM.from_config(config)
            model.load_state_dict(state_dict)
            model.to(device)
        # Instantiate concrete class
        instance = cls(hf_checkpoint or "loaded_model", tokenizer, model=model, **kwargs)
        return instance

class LowLevelPolicy(AbstractPolicy):
    """
    Low-level policy for token-by-token generation.
    
    This policy generates tokens sequentially and is responsible for the actual
    text generation at the token level.
    """
    
    def __init__(self, 
                 hf_checkpoint: str, 
                 tokenizer: PreTrainedTokenizer, 
                 model: Optional[PreTrainedModel] = None, 
                 token: Optional[str] = None, 
                 use_lora: bool = False,
                 adapter_name: Optional[str] = None,
                 env: Optional[Environment] = None) -> None:
        """
        Initialize the low-level policy.
        
        Args:
            hf_checkpoint: Hugging Face model checkpoint path/name
            tokenizer: Tokenizer for the model
            model: Pre-loaded model instance (optional)
            token: Hugging Face token for accessing gated models
            use_lora: Whether to use LoRA adapters
            adapter_name: Name of the adapter to use
            env: Optional environment instance
        """
        # Load model from provided instance or from checkpoint
        if model is not None:
            loaded_model = model
            print("Using provided model instance for policy")
        else:
            print(f"Loading policy model from checkpoint: {hf_checkpoint}")
            config = AutoConfig.from_pretrained(hf_checkpoint, token=token)
            config.use_cache = False
            
            loaded_model = AutoModelForCausalLM.from_pretrained(
                hf_checkpoint,
                config=config,
                token=token
            )
        
        super().__init__(hf_checkpoint, tokenizer, loaded_model, env)
        
        # Initialize LoRA if needed
        self.use_lora = use_lora
        self.adapter_name = adapter_name
        
    @property
    def is_token_level_actor(self) -> bool:
        """This is a token-level actor."""
        return True

    def act(self, 
            state: Union[List[str], List[List[str]], torch.Tensor], 
            base_state: Optional[Union[List[str], torch.Tensor]] = None,
            instructions: Optional[Union[str, List[str]]] = None,
            max_new_tokens: int = 100,
            top_p: float = 0.9,
            top_k: int = 100,
            temperature: float = 0.7,
            require_grad: bool = False,
            debug_mode: bool = False,
            **kwargs) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate a single action using the policy.
        
        Args:
            state: Current state (conversation history)
            base_state: Original state (for reference)
            instructions: Optional high-level instructions to include as system prompts
            max_new_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            temperature: Temperature parameter for controlling generation quality
            require_grad: Whether to calculate gradients
            **kwargs: Additional arguments
            
        Returns:
            tuple: (actions, action_tokens, log_probs)
        """
        device = next(self.model.parameters()).device
        assert device != 'cpu', "Model is on CPU"
        
        # Handle tensor input
        if isinstance(state, torch.Tensor):
            input_ids = state.to(device)
            attention_mask = torch.where(
                input_ids == self.tokenizer.pad_token_id,
                torch.zeros_like(input_ids),
                torch.ones_like(input_ids)
            ).to(device)
        else:
            # Use safe tokenizer call for all other input types
            batch_encoding = safe_tokenizer_call(
                self.tokenizer,
                state,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            input_ids = batch_encoding.input_ids
            attention_mask = batch_encoding.attention_mask
        
        # Generate single action per state
        num_states = len(state)

        # Build chat contexts with optional high-level instructions as system prompts
        chat_contexts = []
        for i, s in enumerate(state):
            # Start with system message if instructions are provided
            if instructions is not None:
                if isinstance(instructions, list) and i < len(instructions) and instructions[i]:
                    # Format instruction as system prompt
                    system_prompt = f"[INSTRUCTION] {instructions[i]} with regards to the following context.\n\n[CONTEXT] {base_state[i] if base_state and i < len(base_state) else ''}"
                    chat = [{'role': 'system', 'content': system_prompt}]
                elif isinstance(instructions, str) and instructions:
                    # Single instruction for all states
                    system_prompt = f"[INSTRUCTION] {instructions} with regards to the following context.\n\n[CONTEXT] {base_state[i] if base_state and i < len(base_state) else ''}"
                    chat = [{'role': 'system', 'content': system_prompt}]
                else:
                    chat = []
            else:
                chat = []
            
            # Add conversation history
            conversation_chat = build_chat(s, 0)
            chat.extend(conversation_chat)
            chat_contexts.append(chat)

        # Tokenize the chat contexts
        prompted_encoding = self.tokenizer.apply_chat_template(
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

        # Choose generation strategy based on configuration
        use_beam_search = getattr(self.env, 'use_beam_search_generation', False)
        use_enhanced_beam_search = getattr(self.env, 'use_enhanced_beam_search', False)
        
        # Create logits processors
        logits_processors = [ClampLogitsProcessor(min_value=-30.0, max_value=30.0)]
        
        # Add refusal suppression if enabled
        use_refusal_suppression = getattr(self.env, 'use_refusal_suppression', True) if self.env else True
        if use_refusal_suppression:
            logits_processors.append(RefusalSuppressionProcessor(self.tokenizer))
            if debug_mode:
                print(f"Refusal suppression enabled with {len(logits_processors)} logits processors")
        else:
            if debug_mode:
                print("Refusal suppression disabled")
        
        if use_enhanced_beam_search:
            # Use enhanced beam search with refusal filtering
            return self.act_with_enhanced_beam_search(
                state=state,
                max_new_tokens=max_new_tokens,
                num_beams=getattr(self.env, 'enhanced_beam_search_beams', 10),
                num_return_sequences=getattr(self.env, 'enhanced_beam_search_sequences', 5),
                **kwargs
            )
        elif use_beam_search:
            # Beam search for more diverse, potentially less refusal-prone generations
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": 5,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "output_scores": True,
                "return_dict_in_generate": True,
                "logits_processor": logits_processors,
                "repetition_penalty": 1.2,  # discourage token repetition
                "no_repeat_ngram_size": 3,  # ban 3-gram loops
            }
        else:
            # Nucleus sampling (current approach)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "do_sample": True,   # Enable sampling instead of greedy decoding
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "output_scores": True,
                "return_dict_in_generate": True,
                "logits_processor": logits_processors,
                "repetition_penalty": 1.2,  # discourage token repetition
                "no_repeat_ngram_size": 3,  # ban 3-gram loops
            }

        with torch.set_grad_enabled(require_grad):
            # Debug: Check input tensors before forward pass
            if not torch.isfinite(batched_input_ids).all():
                raise RuntimeError(f"Non-finite values in input_ids: min: {batched_input_ids.min()}, max: {batched_input_ids.max()}")
            if not torch.isfinite(batched_attention).all():
                raise RuntimeError(f"Non-finite values in attention_mask: min: {batched_attention.min()}, max: {batched_attention.max()}")
            
            # Debug: Simple model health check with a single token
            if debug_mode:
                try:
                    test_input = torch.tensor([[1]], device=device, dtype=batched_input_ids.dtype)
                    test_output = self.model(input_ids=test_input, return_dict=True)
                    if not hasattr(test_output, 'logits') or not torch.isfinite(test_output.logits).all():
                        raise RuntimeError(f'Model health check failed: test forward pass produces non-finite logits. Max: {test_output.logits.max()}, Min: {test_output.logits.min()}, Nan: {torch.isnan(test_output.logits).sum()} Total: {test_output.logits.numel()}')
                except Exception as e:
                    raise RuntimeError(f"Model health check failed: {e}")
            
            outputs = self.model.generate(
                input_ids=batched_input_ids,
                attention_mask=batched_attention,
                **gen_kwargs
            )

        # Extract generated sequences
        actions = []
        action_tokens = []
        
        for i in range(outputs.sequences.size(0)):
            # Get the generated part (after the input)
            input_length = batched_input_ids.size(1)
            generated_tokens = outputs.sequences[i, input_length:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            actions.append(decoded)
            action_tokens.append(generated_tokens)
        
        # Compute log probabilities for the generated sequences
        log_probs = self.batch_get_log_probs(
            input_ids_batch=[batched_input_ids[i] for i in range(batched_input_ids.size(0))],
            output_ids_batch=action_tokens,
            require_grad=require_grad
        )[0]
        
        assert len(actions) == len(action_tokens) == len(log_probs) == num_states, (
            f"Actions, tokens, and log probs must have the same length as state list ({num_states}), "
            f"got {len(actions)}, {len(action_tokens)}, {len(log_probs)}"
        )
        return actions, action_tokens, log_probs
    
    def batch_get_log_probs(self, 
                           input_ids_batch: List[torch.Tensor], 
                           output_ids_batch: List[torch.Tensor], 
                           require_grad: bool = True,
                           sequence_level: bool = False,
                           **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Efficiently compute log probabilities for multiple sequences in a single batch.
        
        Args:
            input_ids_batch: List of input sequence tensors
            output_ids_batch: List of output sequence tensors  
            require_grad: Whether to maintain gradient information
            sequence_level: If True, return sequence-level log probs
            **kwargs: Additional arguments
            
        Returns:
            tuple: (list of log_probs, mean_entropy, mean_perplexity)
        """
        device = next(self.model.parameters()).device
        
        # Ensure all tensors are on the correct device and properly shaped
        input_ids_batch_processed = []
        output_ids_batch_processed = []
        
        for ids in input_ids_batch:
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            input_ids_batch_processed.append(ids)
        
        for ids in output_ids_batch:
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            output_ids_batch_processed.append(ids)
        
        input_ids_batch = input_ids_batch_processed
        output_ids_batch = output_ids_batch_processed
        
        # Find max lengths for padding
        max_input_len = max(ids.shape[1] for ids in input_ids_batch)
        max_output_len = max(ids.shape[1] for ids in output_ids_batch)
        
        # Pad and stack tensors
        padded_inputs = []
        padded_outputs = []
        attention_masks = []
        
        for input_ids, output_ids in zip(input_ids_batch, output_ids_batch):
            # Pad input
            input_pad = max_input_len - input_ids.shape[1]
            if input_pad > 0:
                padded_input = F.pad(input_ids, (0, input_pad), value=self.tokenizer.pad_token_id)
            else:
                padded_input = input_ids
            
            # Pad output
            output_pad = max_output_len - output_ids.shape[1]
            if output_pad > 0:
                padded_output = F.pad(output_ids, (0, output_pad), value=self.tokenizer.pad_token_id)
            else:
                padded_output = output_ids
            
            # Create attention mask for input
            attention_mask = (padded_input != self.tokenizer.pad_token_id).long()
            
            padded_inputs.append(padded_input)
            padded_outputs.append(padded_output)
            attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        batched_inputs = torch.cat(padded_inputs, dim=0)
        batched_outputs = torch.cat(padded_outputs, dim=0)
        batched_attention_masks = torch.cat(attention_masks, dim=0)
        
        # Concatenate inputs and outputs
        full_sequences = torch.cat([batched_inputs, batched_outputs], dim=1)
        
        # Extend attention masks to cover outputs
        output_attention = torch.ones(batched_outputs.shape, device=device, dtype=torch.long)
        full_attention_masks = torch.cat([batched_attention_masks, output_attention], dim=1)
        
        # Single forward pass for all sequences
        with torch.set_grad_enabled(require_grad):
            # Debug: Check input tensors before forward pass
            if not torch.isfinite(full_sequences).all():
                raise RuntimeError(f"Non-finite values in input_ids: min: {full_sequences.min()}, max: {full_sequences.max()}")
            if not torch.isfinite(full_attention_masks).all():
                raise RuntimeError(f"Non-finite values in attention_mask: min: {full_attention_masks.min()}, max: {full_attention_masks.max()}")
            
            # Debug: Ensure inputs are on the correct device
            # For 4-bit quantized models, get device from the input embeddings layer
            model_device = self.model.get_input_embeddings().weight.device
            if full_sequences.device != model_device:
                print(f"Moving input_ids from {full_sequences.device} to {model_device}")
                full_sequences = full_sequences.to(model_device)
            if full_attention_masks.device != model_device:
                print(f"Moving attention_mask from {full_attention_masks.device} to {model_device}")
                full_attention_masks = full_attention_masks.to(model_device)
            
            # Debug: Simple model health check with a single token
            try:
                test_input = torch.tensor([[1]], device=model_device, dtype=full_sequences.dtype)
                test_output = self.model(input_ids=test_input, return_dict=True)
                if not hasattr(test_output, 'logits') or not torch.isfinite(test_output.logits).all():
                    raise RuntimeError("Model health check failed: test forward pass produces non-finite logits")
            except Exception as e:
                raise RuntimeError(f"Model health check failed: {e}")
            
            # Forward pass (activation clamping is handled by permanent hooks)
            outputs = self.model(
                input_ids=full_sequences,
                attention_mask=full_attention_masks,
                return_dict=True
            )
            
            # Clear cache after forward pass to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Extract logits for output tokens only
        logits = outputs.logits[:, max_input_len:max_input_len + max_output_len, :]
        
        # Debug: Check for non-finite values in raw logits
        if not torch.isfinite(logits).all():
            raise RuntimeError(f'Non-finite values detected in logits: min: {logits.min()}, max: {logits.max()}, nan: {torch.isnan(logits).sum()}')
        
        # Clamp logits immediately to prevent overflow in 4-bit quantization
        logits_clamped = torch.clamp(logits, min=-5, max=5)
        
        # Check for non-finite values after clamping
        if not torch.isfinite(logits_clamped).all():
            raise RuntimeError("Non-finite values detected in logits after clamping")
        log_probs = F.log_softmax(logits_clamped, dim=-1)
        
        # Gather log probs for actual output tokens in a single batched operation
        # logits: [batch, max_output_len, vocab]
        # batched_outputs: [batch, max_output_len]
        gathered_log_probs = log_probs.gather(-1, batched_outputs.unsqueeze(-1)).squeeze(-1)

        if sequence_level:
            # Sum over sequence dimension to get per-sequence log-prob (shape [batch])
            log_probs_tensor = gathered_log_probs.sum(dim=-1)
        else:
            log_probs_tensor = gathered_log_probs
        # log_probs_tensor now has shape [batch, max_output_len] or [batch]
        if not torch.isfinite(log_probs_tensor).all():
            raise RuntimeError("Non-finite values detected in log_probs_tensor")
        # Calculate entropy and perplexity (without gradients)
        with torch.set_grad_enabled(False):
            # Use the already-clamped logits for entropy calculation
            probs = F.softmax(logits_clamped, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            mean_entropy = entropy.mean()
            mean_perplexity = torch.exp(mean_entropy)
        
        return log_probs_tensor, mean_entropy, mean_perplexity

    def get_log_probs(self, 
                      input_ids: torch.Tensor, 
                      output_ids: torch.Tensor, 
                      require_grad: bool = True,
                      sequence_level: bool = False,
                      add_special_tokens: bool = True,
                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities of output sequence given input sequence.
        
        Args:
            input_ids: Input sequence tokens
            output_ids: Output sequence tokens
            require_grad: Whether to maintain gradient information
            sequence_level: If True, return sequence-level log probs
            add_special_tokens: Whether to add special tokens during tokenization
            **kwargs: Additional arguments
            
        Returns:
            tuple: (log_probs, entropy_loss)
        """
        # Check if this is a batch call that can be optimized
        if isinstance(input_ids, list) and isinstance(output_ids, list):
            if len(input_ids) > 1:  # Multiple sequences - use batch processing
                # Convert string inputs to tensors if needed
                processed_inputs = []
                processed_outputs = []
                
                for inp, out in zip(input_ids, output_ids):
                    if isinstance(inp, str):
                        inp_tensor = self.tokenizer(inp, return_tensors='pt', add_special_tokens=add_special_tokens).input_ids
                    else:
                        inp_tensor = inp if isinstance(inp, torch.Tensor) else torch.tensor(inp)
                    
                    if isinstance(out, str):
                        out_tensor = self.tokenizer(out, return_tensors='pt', add_special_tokens=add_special_tokens).input_ids
                    else:
                        out_tensor = out if isinstance(out, torch.Tensor) else torch.tensor(out)
                    
                    processed_inputs.append(inp_tensor)
                    processed_outputs.append(out_tensor)
                
                # Use batch processing
                log_probs_tensor, entropy, perplexity = self.batch_get_log_probs(
                    processed_inputs, processed_outputs, require_grad, sequence_level, **kwargs
                )
                
                return log_probs_tensor, entropy, perplexity
        
        # Original single-sequence processing logic
        # Import utilities for device management
        from utils import batch_encode
        
        # Get model's device
        device = next(self.model.parameters()).device
        
        # Process input_ids
        if not isinstance(input_ids, torch.Tensor):
            if isinstance(input_ids, list) and isinstance(input_ids[0], str):
                # Use batch_encode utility for efficient tokenization
                encoding = batch_encode(
                    input_ids, 
                    self.tokenizer, 
                    device=device,
                    padding=True,
                    add_special_tokens=add_special_tokens
                )
                input_ids = encoding['input_ids']
            elif isinstance(input_ids, list) and isinstance(input_ids[0], torch.Tensor):
                input_ids = torch.cat([x.to(device) for x in input_ids], dim=0)
            else:
                raise ValueError(f"input_ids must be a tensor, list of strings, or list of tensors")
        else:
            input_ids = input_ids.to(device)
                
        # Process output_ids
        if not isinstance(output_ids, torch.Tensor):
            if isinstance(output_ids, list) and isinstance(output_ids[0], str):
                # Use batch_encode utility
                encoding = batch_encode(
                    output_ids, 
                    self.tokenizer, 
                    device=device,
                    padding=True,
                    add_special_tokens=add_special_tokens
                )
                output_ids = encoding['input_ids']
            elif isinstance(output_ids, list) and isinstance(output_ids[0], torch.Tensor):
                output_ids = torch.cat([x.to(device) for x in output_ids], dim=0)
            else:
                raise ValueError(f"output_ids must be a tensor, list of strings, or list of tensors")
        else:
            output_ids = output_ids.to(device)
        
        # Create attention mask efficiently
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(device)
        
        # Track entropy of logits
        policy_entropy = []
        input_length = input_ids.size(1)
        # process in batch chunks
        if input_ids.ndim < 2:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            output_ids = output_ids.unsqueeze(0)

        inputs = torch.cat([input_ids, output_ids], dim=1)
        
        # Extend attention mask to cover output tokens
        output_attention_mask = (output_ids != self.tokenizer.pad_token_id).to(device)
        full_attention_mask = torch.cat([attention_mask, output_attention_mask], dim=1)
        
        # Only enable gradients for the last token to save memory during training
        with torch.set_grad_enabled(require_grad):
            # Debug: Ensure inputs are on the correct device
            # For sharded/quantized models, get device from the input embeddings layer
            model_device = self.model.get_input_embeddings().weight.device
            if inputs.device != model_device:
                print(f"Moving inputs from {inputs.device} to {model_device}")
                inputs = inputs.to(model_device)
            if full_attention_mask.device != model_device:
                print(f"Moving full_attention_mask from {full_attention_mask.device} to {model_device}")
                full_attention_mask = full_attention_mask.to(model_device)
            
            # Apply activation clamping during the forward pass
            outputs = self.model(
                input_ids=inputs,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            
            # Clear cache after forward pass to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Get next token logits
        logits = outputs.logits[:, input_length:, :]
        
        # Debug: Check for non-finite values in raw logits
        if not torch.isfinite(logits).all():
            raise RuntimeError(f'Non-finite values detected in logits: min: {logits.min()}, max: {logits.max()}, nan: {torch.isnan(logits).sum()}')
        
        # Clamp logits immediately to prevent overflow in 4-bit quantization
        logits_clamped = torch.clamp(logits, min=-5, max=5)
        
        # Check for non-finite values after clamping
        if not torch.isfinite(logits_clamped).all():
            raise RuntimeError("Non-finite values detected in logits after clamping")
        
        # Get the next token from output_ids
        next_token = output_ids.unsqueeze(-1)
        
        # Calculate log probability
        log_prob = F.log_softmax(logits_clamped, dim=-1)
        log_prob = log_prob.gather(-1, next_token).squeeze(-1)
        
        # Calculate entropy loss (without gradients)
        with torch.set_grad_enabled(False):
            # Use the already-clamped logits for entropy calculation
            probs = F.softmax(logits_clamped, dim=-1)
            entropy = -(probs * F.log_softmax(logits_clamped, dim=-1)).sum(dim=-1)
            H = -entropy.mean()
            policy_entropy.append(H)
            
            # MEMORY CLEANUP: Delete intermediate tensors
            del probs, entropy
        
        # Calculate overall entropy
        policy_entropy = torch.stack(policy_entropy).mean()

        # calculate perplexity
        policy_perplexity = torch.exp(policy_entropy)
        
        # Return sequence-level log probs if requested
        if sequence_level:
            sequence_log_probs = log_prob.sum(dim=-1)
            return sequence_log_probs, policy_entropy, policy_perplexity
            
        return log_prob, policy_entropy, policy_perplexity
    
    def get_embed_similarity(self, 
                            sequences1: Union[List[str], torch.Tensor], 
                            sequences2: Union[List[str], torch.Tensor],
                            require_grad: bool = False,
                            **kwargs) -> torch.Tensor:
        """
        Compute embedding similarity between two sets of sequences.
        
        Args:
            sequences1: First set of sequences
            sequences2: Second set of sequences
            require_grad: Whether to maintain gradient information
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Similarity scores
        """
        device = self.model.get_input_embeddings().weight.device
        
        # Process sequences1
        if isinstance(sequences1, torch.Tensor):
            input_ids1 = sequences1.to(device)
            attention_mask1 = torch.where(
                input_ids1 == self.tokenizer.pad_token_id,
                torch.zeros_like(input_ids1),
                torch.ones_like(input_ids1)
            ).to(device)
        else:
            encoding1 = safe_tokenizer_call(
                self.tokenizer,
                sequences1,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            input_ids1 = encoding1.input_ids
            attention_mask1 = encoding1.attention_mask
            
        # Process sequences2
        if isinstance(sequences2, torch.Tensor):
            input_ids2 = sequences2.to(device)
            attention_mask2 = torch.where(
                input_ids2 == self.tokenizer.pad_token_id,
                torch.zeros_like(input_ids2),
                torch.ones_like(input_ids2)
            ).to(device)
        else:
            encoding2 = safe_tokenizer_call(
                self.tokenizer,
                sequences2,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            input_ids2 = encoding2.input_ids
            attention_mask2 = encoding2.attention_mask
        
        # Get embeddings
        with torch.set_grad_enabled(require_grad):
            # Get the last hidden states from the model
            outputs1 = self.model(
                input_ids=input_ids1,
                attention_mask=attention_mask1,
                output_hidden_states=True,
                return_dict=True
            )
            
            outputs2 = self.model(
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get the last hidden state
            hidden_states1 = outputs1.hidden_states[-1]
            hidden_states2 = outputs2.hidden_states[-1]
            
            # Apply attention mask and compute mean embeddings
            masked_states1 = hidden_states1 * attention_mask1.unsqueeze(-1)
            masked_states2 = hidden_states2 * attention_mask2.unsqueeze(-1)
            
            # Compute mean embeddings
            seq_lengths1 = attention_mask1.sum(dim=1, keepdim=True)
            seq_lengths2 = attention_mask2.sum(dim=1, keepdim=True)
            
            mean_embedding1 = masked_states1.sum(dim=1) / seq_lengths1
            mean_embedding2 = masked_states2.sum(dim=1) / seq_lengths2
            
            # Normalize embeddings
            norm1 = torch.norm(mean_embedding1, p=2, dim=1, keepdim=True)
            norm2 = torch.norm(mean_embedding2, p=2, dim=1, keepdim=True)
            
            normalized_embedding1 = mean_embedding1 / norm1
            normalized_embedding2 = mean_embedding2 / norm2
            
            # Compute cosine similarity
            similarities = torch.bmm(
                normalized_embedding1.unsqueeze(1),
                normalized_embedding2.unsqueeze(2)
            ).squeeze()
            
        return similarities
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                with_head: bool = False,
                **kwargs) -> Any:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            with_head: Whether to apply the language model head
            **kwargs: Additional arguments for the model
            
        Returns:
            Model outputs
        """
        # Set the model to evaluation mode to avoid dropout
        self.model.eval()
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
    
    def print_memory(self, message: str = "") -> None:
        """
        Print memory usage information.
        
        Args:
            message: Optional message to include
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"{message} - CUDA Memory: {allocated:.2f}GB allocated, {max_allocated:.2f}GB max, {reserved:.2f}GB reserved")
        else:
            print(f"{message} - CUDA not available")
    
    @contextmanager
    def eval_mode(self):
        """Context manager for evaluation mode."""
        was_training = self.model.training
        self.model.eval()
        yield
        self.model.train(was_training)
    
    @contextmanager
    def train_mode(self):
        """Context manager for training mode."""
        was_training = self.model.training
        self.model.train()
        yield
        self.model.train(was_training)
    
    def cleanup_gpu_memory(self) -> None:
        """Explicitly clear GPU memory."""
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
    
    def setup_forward_hooks(self, enable_hooks: bool = True, clamp_range: float = 30.0, layernorm_range: float = 10.0) -> None:
        """
        Set up forward hooks with configuration.
        
        Args:
            enable_hooks: Whether to attach activation clamping hooks
            clamp_range: Clamp range for activation hooks (-range to +range)
            layernorm_range: Clamp range for LayerNorm hooks (-range to +range)
        """
        # Remove existing hooks if any
        if hasattr(self, 'activation_hooks') and self.activation_hooks:
            for hook in self.activation_hooks:
                hook.remove()
            self.activation_hooks = []
        
        # Attach new hooks with configuration
        self.activation_hooks = attach_permanent_activation_clamping(
            self.model, 
            enable_hooks=enable_hooks,
            clamp_range=clamp_range,
            layernorm_range=layernorm_range
        )
    
    def configure_search_strategy(self, strategy: str = "nucleus") -> Dict[str, Any]:
        """
        Get recommended parameters for different search strategies.
        
        Args:
            strategy: Search strategy ("nucleus", "beam", "diverse", "aggressive")
            
        Returns:
            dict: Recommended parameters for the strategy
        """
        strategies = {
            "nucleus": {
                "use_beam_search": False,
                "top_p": 0.9,
                "top_k": 50,
                "detect_refusals": True,
                "max_retry_attempts": 3
            },
            "beam": {
                "use_beam_search": True,
                "num_beams": 5,
                "detect_refusals": True,
                "max_retry_attempts": 2
            },
            "diverse": {
                "use_beam_search": False,
                "top_p": 0.8,
                "top_k": 40,
                "detect_refusals": True,
                "max_retry_attempts": 4
            },
            "aggressive": {
                "use_beam_search": False,
                "top_p": 0.7,
                "top_k": 30,
                "detect_refusals": True,
                "max_retry_attempts": 5
            }
        }
        
        return strategies.get(strategy, strategies["nucleus"])
    
    def act_with_strategy(self, 
                         state: Union[List[str], List[List[str]], torch.Tensor],
                         strategy: str = "nucleus",
                         **kwargs) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate actions using a predefined search strategy.
        
        Args:
            state: Current state
            strategy: Search strategy to use
            **kwargs: Additional arguments (override strategy defaults)
            
        Returns:
            tuple: (actions, action_tokens, log_probs)
        """
        # Get strategy parameters
        strategy_params = self.configure_search_strategy(strategy)
        
        # Merge with user-provided kwargs (user kwargs take precedence)
        merged_params = {**strategy_params, **kwargs}
        
        return self.act(state=state, **merged_params)

    def act_with_enhanced_beam_search(self, 
                                    state: Union[List[str], List[List[str]], torch.Tensor],
                                    max_new_tokens: int = 100,
                                    num_beams: int = 10,
                                    num_return_sequences: int = 5,
                                    **kwargs) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate actions using enhanced beam search that filters out refusal candidates.
        
        This approach generates multiple candidates and filters out refusals before
        returning the best non-refusal candidates.
        
        Args:
            state: Current state
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments
            
        Returns:
            tuple: (actions, action_tokens, log_probs)
        """
        device = next(self.model.parameters()).device
        
        # Handle tensor input
        if isinstance(state, torch.Tensor):
            input_ids = state.to(device)
            attention_mask = torch.where(
                input_ids == self.tokenizer.pad_token_id,
                torch.zeros_like(input_ids),
                torch.ones_like(input_ids)
            ).to(device)
        else:
            # Use safe tokenizer call for all other input types
            batch_encoding = safe_tokenizer_call(
                self.tokenizer,
                state,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            input_ids = batch_encoding.input_ids
            attention_mask = batch_encoding.attention_mask
        
        # Build chat contexts (simplified for this example)
        chat_contexts = []
        for s in state:
            conversation_chat = build_chat(s, 0)
            chat_contexts.append(conversation_chat)

        # Tokenize the chat contexts
        prompted_encoding = self.tokenizer.apply_chat_template(
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

        # Enhanced beam search with refusal filtering
        # Create logits processors
        logits_processors = [ClampLogitsProcessor(min_value=-30.0, max_value=30.0)]
        
        # Add refusal suppression if enabled
        use_refusal_suppression = getattr(self.env, 'use_refusal_suppression', True) if self.env else True
        if use_refusal_suppression:
            logits_processors.append(RefusalSuppressionProcessor(self.tokenizer))
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
            "logits_processor": logits_processors,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
        }

        with torch.set_grad_enabled(False):
            outputs = self.model.generate(
                input_ids=batched_input_ids,
                attention_mask=batched_attention,
                **gen_kwargs
            )

        # Extract and filter generated sequences
        all_actions = []
        all_action_tokens = []
        all_log_probs = []
        
        input_length = batched_input_ids.size(1)
        
        for i in range(outputs.sequences.size(0)):
            # Get the generated part (after the input)
            generated_tokens = outputs.sequences[i, input_length:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Check if this is a refusal
            is_refusal = self._detect_refusal(decoded)
            
            if not is_refusal:
                all_actions.append(decoded)
                all_action_tokens.append(generated_tokens)
                # Compute log prob for this sequence
                log_prob = self.get_log_probs(
                    batched_input_ids[i], 
                    generated_tokens, 
                    require_grad=False
                )[0]
                all_log_probs.append(log_prob)
        
        # If we have no non-refusal candidates, return the best refusal (fallback)
        if not all_actions:
            # Return the first candidate as fallback
            generated_tokens = outputs.sequences[0, input_length:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            log_prob = self.get_log_probs(
                batched_input_ids[0], 
                generated_tokens, 
                require_grad=False
            )[0]
            
            return [decoded], [generated_tokens], [log_prob]
        
        # Return the best non-refusal candidates
        return all_actions, all_action_tokens, all_log_probs

class TemplatePolicy(LowLevelPolicy):
    """
    @deprecated: This class is deprecated and should not be used.
    TemplatePolicy was designed for low-level token generation but is not appropriate
    for the current PPO architecture which uses:
    - LowLevelPolicy for token-level actions
    - PPOTemplateHighLevelPolicy for template selection
    
    Use LowLevelPolicy instead for low-level actions.
    """
    """
    Policy that generates responses using templates.
    
    This policy extends TransformerPolicy to generate multiple template options,
    track which one was selected by the environment, and maintain branch tracking
    for proper GAE computation. It's designed to maximize sample efficiency by
    learning from all template options, not just the one that was selected.
    
    Key features:
    - Generate multiple templated responses for each input state
    - Track which template branch was followed in the environment
    - Maintain branch structure for non-standard GAE calculation
    - Support template-based red teaming with diverse question formulations
    """
    
    def __init__(
        self,
        model_name: str,
        tokenizer = None,
        use_lora: bool = False,
        adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hf_token: Optional[str] = None,
        templates_path: Optional[str] = None,
        debug_mode: bool = False,
        **kwargs
    ):
        """
        Initialize the TemplatePolicy.
        
        Args:
            model_name: Name of the model to load
            tokenizer: Optional pre-loaded tokenizer
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            adapter_path: Path to LoRA adapter to load
            device: Device to place model on
            hf_token: Hugging Face API token for accessing gated models
            templates_path: Path to JSON file containing templates
            debug_mode: Whether to print debug information
            **kwargs: Additional arguments to pass to TransformerPolicy
        """
        super().__init__(
            hf_checkpoint=model_name,
            tokenizer=tokenizer,
            use_lora=use_lora,
            adapter_name=adapter_path,
            token=hf_token,
            **kwargs
        )
        
        self.debug_mode = debug_mode
        self.templates = self._load_templates(templates_path)
        self.num_templates = len(self.templates) - 1 if self.templates else 0  # Excluding rejection template
        self.rejection_template_idx = 0  # Index of rejection template (reserved)
        
        # Initialize branch tracking counters
        self.next_branch_id = 0
        self.parent_branches = {}  # conversation_id -> parent branch ID
        
        print(f"TemplatePolicy initialized with {self.num_templates + 1} templates")
    
    def _load_templates(self, templates_path: str) -> List[Dict[str, Any]]:
        """
        Load templates from a JSON file or use default templates.
        
        Args:
            templates_path: Path to JSON file containing templates
            
        Returns:
            List of template dictionaries
        """
        
        with open(templates_path, 'r') as f:
            template_data = json.load(f)
        
            # Extract templates array from JSON
            if "templates" in template_data:
                templates = template_data["templates"]
            else:
                templates = template_data
            
            print(f"Loaded {len(templates)} templates from {templates_path}")
            return templates
        
    
    def _extract_topics(self, state_texts: List[str]) -> List[str]:
        """
        Extract topics from input texts.
        
        Args:
            state_texts: List of input texts
            
        Returns:
            List of extracted topics
        """
        # Simple topic extraction based on keywords
        # In a real implementation, this would use more sophisticated NLP
        topics = []
        
        for text in state_texts:
            # Simple approach: take the first few words as the topic
            words = text.split()
            if len(words) > 3:
                topic = " ".join(words[:3])
            else:
                topic = "this topic"
            topics.append(topic)
        
        return topics
    
    def act(
        self, 
        state: Union[List[str], torch.Tensor], 
        base_state: Optional[Union[List[str], torch.Tensor]] = None,
        instructions: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Tuple[List[List[str]], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate multiple template-based responses for each input state with intelligent search.
        
        Args:
            state: Current state
            base_state: Base state to use for templates
            instructions: Optional instructions to include
            **kwargs: Additional arguments for generation (including search parameters)
            
        Returns:
            Tuple of:
            - List of lists of template responses
            - List of token tensors for each template
            - List of log prob tensors for each template
        """
        device = next(self.model.parameters()).device
        
        # Process state input
        if isinstance(state, torch.Tensor):
            input_ids = state.to(device)
            attention_mask = torch.where(
                input_ids == self.tokenizer.pad_token_id,
                torch.zeros_like(input_ids),
                torch.ones_like(input_ids)
            ).to(device)
        else:
            # Use safe tokenizer call
            encoding = safe_tokenizer_call(
                self.tokenizer,
                state,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            input_ids = encoding.input_ids
            attention_mask = encoding.attention_mask
            
        # Process base state if provided
        if base_state is not None:
            if isinstance(base_state, torch.Tensor):
                base_input_ids = base_state.to(device)
                base_attention_mask = torch.where(
                    base_input_ids == self.tokenizer.pad_token_id,
                    torch.zeros_like(base_input_ids),
                    torch.ones_like(base_input_ids)
                ).to(device)
            else:
                # Use safe tokenizer call
                base_encoding = safe_tokenizer_call(
                    self.tokenizer,
                    base_state,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(device)
                base_input_ids = base_encoding.input_ids
                base_attention_mask = base_encoding.attention_mask
        
        # Extract topics from the states - batch process
        topics = self._extract_topics(state)
        
        # Generate base responses using the parent policy with intelligent search
        # Enable refusal detection by default for template policy
        kwargs.setdefault('detect_refusals', True)
        kwargs.setdefault('max_retry_attempts', 3)
        
        base_actions, base_tokens, base_log_probs = super().act(
            state=state, 
            base_state=base_state, 
            instructions=instructions,
            **kwargs
        )
        
        # Convert to lists if not already
        if not isinstance(base_actions, list):
            base_actions = [base_actions]
        
        # Initialize result containers
        batch_formatted_responses = []
        batch_tokens = []
        template_log_probs = []
        
        # Batch process template formatting for better GPU utilization
        all_formatted_responses = []
        all_state_indices = []
        all_template_indices = []
        
        # Process each state
        for i, (state_i, base_state_i, base_action_i, topic_i) in enumerate(zip(state, base_state, base_actions, topics)):
            # Collect all formatted responses for this state
            state_formatted_responses = []
            
            # Parse base_state into question/context and choices for formatting
            if isinstance(base_state_i, str):
                parsed = self._parse_mc_base_state(base_state_i)
            else:
                parsed = {"question": "", "context": "", "choices": [], "choices_text": ""}
            
            # Apply each template
            for j, template_data in enumerate(self.templates):
                template = template_data.get('template')
                if not template:
                    # If no template string provided, fall back to instruction or base action
                    template = template_data.get('instruction') or "{question}"
                
                # Infer option label from id if present (e.g., mc_examine_A)
                tpl_id = template_data.get('id', '')
                option_label = None
                if tpl_id.endswith('_A'):
                    option_label = 'A'
                elif tpl_id.endswith('_B'):
                    option_label = 'B'
                elif tpl_id.endswith('_C'):
                    option_label = 'C'
                elif tpl_id.endswith('_D'):
                    option_label = 'D'
                
                option_text = ""
                if option_label is not None and parsed["choices"]:
                    idx = ord(option_label) - ord('A')
                    if 0 <= idx < len(parsed["choices"]):
                        option_text = parsed["choices"][idx]
                
                # Build history and last-thought context
                last_thought = ""
                history_text = ""
                if isinstance(state_i, list):
                    # state_i is like [formatted_question, *history]
                    history = [s for s in state_i[1:] if isinstance(s, str)]
                    if history:
                        last_thought = history[-1]
                        history_text = "\n".join(history)
                elif isinstance(state_i, str):
                    # Heuristic: take lines after the first blank line as history
                    parts = [ln for ln in state_i.splitlines() if ln.strip()]
                    if len(parts) > 2:
                        last_thought = parts[-1]
                        history_text = "\n".join(parts[1:])
                
                # Build formatting dict
                fmt = {
                    "question": parsed["question"],
                    "context": parsed["question"] or parsed["context"],
                    "choices_text": parsed["choices_text"],
                    "option_label": option_label or "",
                    "option_text": option_text,
                    "last_thought": last_thought,
                    "history": history_text,
                }
                
                # Ensure last thinking step is included even if template lacks an explicit placeholder
                template_for_format = template
                if "{last_thought}" not in template_for_format and last_thought:
                    template_for_format = template_for_format + "\n\nPrior reasoning: {last_thought}"
                
                try:
                    formatted_response = template_for_format.format(**fmt)
                except Exception:
                    # Fail loudly but fallback to instruction if formatting keys mismatch
                    formatted_response = template_for_format
                
                # Check refusal
                if self._detect_refusal(formatted_response):
                    if self.debug_mode:
                        print(f"Skipping template due to refusal pattern: {template_data.get('id', 'unknown')}")
                    continue
                
                state_formatted_responses.append(formatted_response)
                all_formatted_responses.append(formatted_response)
                all_state_indices.append(i)
                all_template_indices.append(j)
            
            # If all templates were filtered, fallback
            if not state_formatted_responses:
                if self.debug_mode:
                    print("All templates contained refusals or failed formatting, falling back to base action")
                state_formatted_responses = [base_action_i]
                all_formatted_responses.append(base_action_i)
                all_state_indices.append(i)
                all_template_indices.append(-1)
            
            batch_formatted_responses.append(state_formatted_responses)
        
        # Batch tokenize all formatted responses at once for better GPU utilization
        if all_formatted_responses:
            with torch.no_grad():
                # Batch tokenization
                batch_encoding = safe_tokenizer_call(
                    self.tokenizer,
                    all_formatted_responses,
                    return_tensors='pt',
                    padding=True,
                    add_special_tokens=True
                ).to(device)
                
                all_tokens = batch_encoding.input_ids
                
                # Batch process log probabilities
                # Create state tokens for each response
                state_tokens_batch = []
                for state_idx in all_state_indices:
                    state_tokens = safe_tokenizer_call(
                        self.tokenizer,
                        state[state_idx],
                        return_tensors='pt',
                        add_special_tokens=True
                    ).input_ids.to(device)
                    state_tokens_batch.append(state_tokens)
                
                # Batch compute log probabilities using efficient batch processing
                if len(state_tokens_batch) > 0:
                    # Use the new batch_get_log_probs method for better efficiency
                    try:
                        all_log_probs, _, _ = self.batch_get_log_probs(
                            state_tokens_batch, 
                            [tokens.squeeze(0) for tokens in all_tokens], 
                            require_grad=False
                        )
                    except Exception as e:
                        # Fallback to chunked processing if batch processing fails
                        print(f"Batch processing failed ({e}), falling back to chunked processing")
                        batch_size = 4  # Configurable batch size
                        all_log_probs = []
                        
                        for batch_start in range(0, len(state_tokens_batch), batch_size):
                            batch_end = min(batch_start + batch_size, len(state_tokens_batch))
                            
                            batch_state_tokens = state_tokens_batch[batch_start:batch_end]
                            batch_output_tokens = all_tokens[batch_start:batch_end]
                            
                            # Use batch processing for the chunk
                            try:
                                chunk_log_probs, _, _ = self.batch_get_log_probs(
                                    batch_state_tokens,
                                    [tokens.squeeze(0) for tokens in batch_output_tokens],
                                    require_grad=False
                                )
                                all_log_probs.extend(chunk_log_probs)
                            except Exception as chunk_e:
                                print(f"Chunk batch processing failed ({chunk_e}), using individual processing")
                                # Individual processing fallback
                                for state_tokens, output_tokens in zip(batch_state_tokens, batch_output_tokens):
                                    individual_log_probs, _, _ = self.get_log_probs(
                                        input_ids=state_tokens,
                                        output_ids=output_tokens,
                                        require_grad=False
                                    )
                                    all_log_probs.append(individual_log_probs)
                else:
                    all_log_probs = [torch.tensor([0.0], device=device) for _ in all_formatted_responses]
        else:
            all_tokens = []
            all_log_probs = []
        
        # Reorganize results by state
        for i in range(len(state)):
            state_tokens = []
            state_log_probs = []
            
            # Find all tokens and log probs for this state
            for j, (response_state_idx, token, log_prob) in enumerate(zip(all_state_indices, all_tokens, all_log_probs)):
                if response_state_idx == i:
                    state_tokens.append(token.unsqueeze(0) if token.dim() == 1 else token)
                    state_log_probs.append(log_prob)
            
            # Handle case where no templates were valid
            if not state_tokens:
                # Use base action tokens and log probs
                if hasattr(base_tokens, 'to'):
                    state_tokens = [base_tokens]
                else:
                    fallback_tokens = safe_tokenizer_call(
                        self.tokenizer, 
                        base_actions[i], 
                        return_tensors='pt'
                    ).input_ids.to(device)
                    state_tokens = [fallback_tokens]
                state_log_probs = [base_log_probs if hasattr(base_log_probs, 'to') else torch.tensor([0.0], device=device)]
            
            batch_tokens.append(state_tokens)
            template_log_probs.append(state_log_probs)
        
        return batch_formatted_responses, batch_tokens, template_log_probs
    
    def rollout(
        self,
        env,
        num_steps: int,
        max_batch_size: int = 8,
        max_turns_per_conversation: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a rollout collecting data from template branches.
        
        Args:
            env: Environment to interact with
            num_steps: Target number of steps to collect
            max_batch_size: Maximum batch size for environment
            max_turns_per_conversation: Maximum turns per conversation
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing batch data and branch tracking information
        """
        # Statistics tracking
        episode_lens = []  # Track length of each episode
        num_episodes_collected = 0
        target_episodes = int(num_steps // 2)  # Aim to collect this many complete episodes
        
        # Initialize the environment
        state = env.reset(batch_size=max_batch_size)
        
        # Current episode length and total steps tracking
        current_episode_len = 0
        total_steps = 0
        
        # Data collection structure
        conversation_data = {
            "states": [],              # Current states
            "actions": [],             # Actions taken (template outputs)
            "log_probs": [],           # Log probabilities of actions
            "rewards": [],             # Rewards received
            "next_states": [],         # Next states
            "dones": [],               # Episode termination flags
            "base_states": [],         # Original base states
            "is_continuation": [],     # Flag: True if this branch was continued
            "branch_ids": [],          # Unique ID for each branch
            "parent_branch_ids": [],   # ID of parent branch (for continuation tracking)
            "template_ids": [],        # Template IDs used
            "conversation_ids": [],    # Conversation this branch belongs to
            "turn_indices": [],        # Turn index within conversation
            "action_tokens": []        # Action tokens
        }
        
        # Reset branch tracking
        self.next_branch_id = 0
        self.parent_branches = {}  # conversation_id -> current parent branch ID
        conversation_id = 0
        
        already_dones = [False] * len(state)
        
        # Execute rollout to collect multiple sequential episodes
        while num_episodes_collected < target_episodes and total_steps < num_steps:
            if self.debug_mode:
                print(f"Rollout step {total_steps}/{num_steps}, collecting {num_episodes_collected}/{target_episodes} episodes")
            
            # Generate action using the model - get all template options
            with torch.no_grad():
                # Get base state from environment if available, otherwise use None
                base_state = env.get_base_state()
                
                # Generate all template options for the current state
                all_actions, all_action_tokens, all_log_probs = self.act(
                    state=state,
                    base_state=base_state,
                    **kwargs
                )
                
                # Process step data for each state in the batch
                best_actions = []
                step_templates = []
                best_action_idx = []
                
                # For each state, score all template options
                for i in range(len(state)):
                    if already_dones[i]:
                        # Skip if this environment is already done
                        best_actions.append("")
                        best_action_idx.append(0)
                        step_templates.append([])
                        continue
                    
                    state_actions = all_actions[i]
                    state_tokens = all_action_tokens[i]
                    state_log_probs = all_log_probs[i]
                    
                    # Score all template options based on environment
                    # This can be customized based on the environment interface
                    template_scores = []
                    template_responses = []
                    
                    # Try standard interface with score_safety
                    target_responses = env.get_target_responses(state_actions)
                    base_state_repeated = [base_state[i]] * len(state_actions)
                    safety_scores = env.score_safety(
                        target_responses, 
                        questions=state_actions, 
                        reference_state=base_state_repeated
                    ).cpu().numpy()
                    
                    for template_idx, template_action in enumerate(state_actions):
                        template_scores.append(safety_scores[template_idx])
                        template_responses.append(target_responses[template_idx])
                    
                        
                    
                    # Find the best template based on score
                    if len(template_scores) > 0:
                        best_template_idx = int(np.argmax(template_scores))
                    else:
                        best_template_idx = 0  # Default to first template
                    
                    best_actions.append(state_actions[best_template_idx])
                    best_action_idx.append(best_template_idx)
                    
                    # Store all template data for this state
                    templates_i = []
                    for template_idx, (template_action, template_tokens, template_log_probs, template_score, template_response) in enumerate(
                        zip(state_actions, state_tokens, state_log_probs, template_scores, template_responses)
                    ):
                        templates_i.append({
                            "action": template_action,
                            "tokens": template_tokens,
                            "log_probs": template_log_probs,
                            "score": template_score,
                            "response": template_response
                        })
                    
                    step_templates.append(templates_i)
                
                # Step the environment with the best actions
                step_template_ids = []
                step_system_prompts = []
                for i in range(len(state)):
                    if already_dones[i]:
                        step_template_ids.append("")
                        step_system_prompts.append("")
                        continue
                    chosen_idx = best_action_idx[i] if i < len(best_action_idx) else 0
                    template_id = self.templates[chosen_idx]['id'] if chosen_idx < len(self.templates) else ""
                    step_template_ids.append(template_id)
                    # Pass the instruction for judge shaping; fall back to empty string
                    step_system_prompts.append(self.templates[chosen_idx].get('instruction', ""))
                next_state, actual_responses, actual_rewards, done = env.step(best_actions, template_ids=step_template_ids, system_prompt=step_system_prompts)
                
                # Store data for all template branches
                for i in range(len(next_state)):
                    if already_dones[i]:
                        continue
                    
                    step_templates_i = step_templates[i]
                    if not step_templates_i:
                        continue
                    
                    # Create arrays to store multiple template options per state
                    batch_state_repeated = [state[i]] * len(step_templates_i)
                    batch_next_state_repeated = [next_state[i]] * len(step_templates_i)
                    batch_base_state_repeated = [base_state[i]] * len(step_templates_i) 
                    batch_dones_repeated = [done[i]] * len(step_templates_i)
                    
                    for j in range(len(step_templates_i)):
                        # Is this the branch that was followed?
                        is_continuation = (j == best_action_idx[i])
                        
                        # Assign a unique branch ID
                        branch_id = self.next_branch_id
                        self.next_branch_id += 1
                        
                        # If this is the followed branch, update parent_branches for next turn
                        if is_continuation:
                            self.parent_branches[conversation_id] = branch_id
                        
                        # Store template metadata
                        template_id = self.templates[j]['id'] if j < len(self.templates) else "unknown"
                        
                        # For non-continued branches, create a hypothetical next state
                        # Combining the current state, the action, and the target's response
                        if not is_continuation:
                            branch_next_state = f"{state[i]} {step_templates_i[j]['action']} {step_templates_i[j]['response']}"
                            batch_next_state_repeated[j] = branch_next_state
                        
                        # Store all template options with the same state
                        conversation_data["states"].append(batch_state_repeated[j])
                        conversation_data["actions"].append(step_templates_i[j]["action"])
                        conversation_data["log_probs"].append(step_templates_i[j]["log_probs"])
                        conversation_data["rewards"].append(step_templates_i[j]["score"])
                        conversation_data["next_states"].append(batch_next_state_repeated[j])
                        conversation_data["dones"].append(batch_dones_repeated[j])
                        conversation_data["base_states"].append(batch_base_state_repeated[j])
                        conversation_data["action_tokens"].append(step_templates_i[j]["tokens"])
                        
                        # Store branch tracking information
                        conversation_data["branch_ids"].append(branch_id)
                        conversation_data["parent_branch_ids"].append(self.parent_branches.get(conversation_id, -1))
                        conversation_data["is_continuation"].append(is_continuation)
                        conversation_data["template_ids"].append(template_id)
                        conversation_data["conversation_ids"].append(conversation_id)
                        conversation_data["turn_indices"].append(current_episode_len)
                
                # Update done tracking
                for i in range(len(done)):
                    already_dones[i] = done[i] or already_dones[i]
                
                # Update state for next iteration
                state = next_state
                current_episode_len += 1
                total_steps += len(state) - sum(already_dones)
                
                # Check if episode is done
                if all(done) or current_episode_len >= max_turns_per_conversation:
                    # Record episode statistics
                    episode_lens.append(current_episode_len)
                    num_episodes_collected += len(state) - sum(already_dones)
                    
                    # Reset for next episode
                    state = env.reset(batch_size=max_batch_size)
                    current_episode_len = 0
                    already_dones = [False] * len(state)
                    
                    # Reset the parent branches for the new conversation
                    self.parent_branches = {}
                    conversation_id += 1
        
        # Process the collected data for training
        if len(conversation_data["states"]) == 0:
            raise RuntimeError("No data collected during rollout")
        
        # Tokenize states and actions if not already tokenized
        if not torch.is_tensor(conversation_data["states"][0]):
            batch_states = [
                self.tokenizer(s, return_tensors='pt', padding=True).input_ids
                for s in conversation_data["states"]
            ]
        else:
            batch_states = conversation_data["states"]
        
        if not torch.is_tensor(conversation_data["next_states"][0]):
            batch_next_states = [
                self.tokenizer(s, return_tensors='pt', padding=True).input_ids
                for s in conversation_data["next_states"]
            ]
        else:
            batch_next_states = conversation_data["next_states"]
        
        if not torch.is_tensor(conversation_data["base_states"][0]):
            batch_base_states = [
                self.tokenizer(s, return_tensors='pt', padding=True).input_ids if s is not None else None
                for s in conversation_data["base_states"]
            ]
        else:
            batch_base_states = conversation_data["base_states"]
        
        # Create tensors for rewards and dones
        batch_rews = torch.tensor(conversation_data["rewards"], dtype=torch.bfloat16)
        batch_dones = torch.tensor(conversation_data["dones"], dtype=torch.bool)
        
        # Compile branch metadata for GAE calculations
        branch_metadata = {
            "branch_ids": conversation_data["branch_ids"],
            "parent_branch_ids": conversation_data["parent_branch_ids"],
            "is_continuation": conversation_data["is_continuation"],
            "conversation_ids": conversation_data["conversation_ids"],
            "turn_indices": conversation_data["turn_indices"],
            "template_ids": conversation_data["template_ids"]
        }
        
        if self.debug_mode:
            print(f"Rollouts collected: {len(batch_states)} steps across {num_episodes_collected} episodes")
            print(f"Average reward: {batch_rews.mean().item():.4f}, Template branches per state: {len(batch_states)/len(set(conversation_data['conversation_ids'])):.1f}")
        
        # Create a rollout object with the processed data
        return {
            'batch_state': batch_states,
            'batch_next_state': batch_next_states,
            'batch_acts': conversation_data["action_tokens"],
            'batch_log_probs': conversation_data["log_probs"],
            'batch_rews': batch_rews,
            'batch_dones': batch_dones,
            'batch_lens': episode_lens,
            'batch_base_state': batch_base_states,
            'branch_metadata': branch_metadata  # Add branch metadata for GAE
        }

    def _parse_mc_base_state(self, base_text: str) -> Dict[str, Any]:
        """
        Parse the environment base state (formatted question + choices) into structured pieces
        that templates can format with. Returns keys: question/context, choices, choices_text.
        """
        lines = [ln for ln in (base_text or "").splitlines() if ln.strip()]
        question = ""
        context = ""
        choices: List[str] = []
        for ln in lines:
            if ln.startswith("Question:"):
                question = ln[len("Question:"):].strip()
            elif ") " in ln and ln[0].isalpha() and ln[1] == ')':
                # e.g., "A) text"
                choices.append(ln.split(") ", 1)[1].strip())
        choices_text = "\n".join([f"{chr(ord('A')+i)}) {c}" for i, c in enumerate(choices)])
        return {"question": question, "context": context, "choices": choices, "choices_text": choices_text}
