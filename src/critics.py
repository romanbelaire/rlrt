"""
Critic implementations for reinforcement learning agents.

This module provides implementations of value function estimators (critics)
that are used to evaluate states and actions in the reinforcement learning process.
"""

from typing import Dict, List, Tuple, Union, Any
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
import os

from src.base import Critic, ModelBase
from src.utils import apply_activation_clamping, disable_lora_adapters, get_transformer_module

class AbstractValueCritic(ModelBase, Critic, ABC):
    """
    Abstract base class for value critics using transformer models.
    
    This class provides common functionality for value estimation and defines
    the interface that concrete implementations must follow.
    """
    
    def __init__(self, 
                 hf_checkpoint: str, 
                 tokenizer: PreTrainedTokenizer,
                 model: PreTrainedModel) -> None:
        """
        Initialize the abstract value critic.
        
        Args:
            hf_checkpoint: Hugging Face model checkpoint path/name
            tokenizer: Tokenizer for the model
            model: Pre-loaded model instance
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.hf_checkpoint = hf_checkpoint
        
        # Use provided model instance
        self.model = model
            
        # Freeze backbone only; LoRA adapter weights stay trainable for L2/L3 policies
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                continue
            param.requires_grad = False
            
        # Initialize the value head (to be defined by subclasses)
        self._init_value_head()
        
        # Ensure value head parameters have gradients enabled (not frozen like base model)
        for param in self.value_head.parameters():
            param.requires_grad = True
        
        # Match the dtype of the base model to avoid dtype mismatches
        model_dtype = next(self.model.parameters()).dtype
        # Ensure value head uses bf16 for consistency
        if model_dtype == torch.bfloat16:
            self.value_head = self.value_head.to(dtype=torch.bfloat16)
        else:
            self.value_head = self.value_head.to(dtype=model_dtype)
                    
        # Default to CPU, will be moved to appropriate device when needed
        self.to('cpu')

        # ------------------------------------------------------------------
        # Memory optimisation for critics (b, c, d from user request)
        # b) Disable key-value cache to avoid storing past states we never reuse
        self.model.config.use_cache = False

        # c) Enable gradient checkpointing *only* when some base-model parameters
        #    are trainable.  When the backbone is entirely frozen (as in our
        #    critics) checkpointing provides zero benefit but still emits a
        #    warning and adds overhead.
        if any(p.requires_grad for p in self.model.parameters()):
            if not getattr(self.model, "gradient_checkpointing", False):
                try:
                    self.model.gradient_checkpointing_enable()
                except AttributeError:
                    raise RuntimeError("Base model does not support gradient checkpointing")

        # d) Use memory-efficient FlashAttention v2 when available.
        self.model.config.attn_implementation = "flash_attention_2"
        # ------------------------------------------------------------------
        
        # Set default forward hook configuration
        self.enable_forward_hooks = True
        self.forward_hook_clamp_range = 30.0
        self.forward_hook_layernorm_range = 10.0
    
    def setup_forward_hooks(self, enable_hooks: bool = True, clamp_range: float = 30.0, layernorm_range: float = 10.0) -> None:
        """
        Set up forward hook configuration.
        
        Args:
            enable_hooks: Whether to use activation clamping hooks
            clamp_range: Clamp range for activation hooks (-range to +range)
            layernorm_range: Clamp range for LayerNorm hooks (-range to +range)
        """
        self.enable_forward_hooks = enable_hooks
        self.forward_hook_clamp_range = clamp_range
        self.forward_hook_layernorm_range = layernorm_range
    
    @abstractmethod
    def _init_value_head(self) -> None:
        """Initialize the value head architecture. Must be implemented by subclasses."""
        pass
        
    @property
    @abstractmethod
    def is_token_level(self) -> bool:
        """Whether this is a token-level critic. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def _compute_values(self, 
                       hidden_states: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimates from hidden states.
        Must be implemented by subclasses.
        
        Args:
            hidden_states: Hidden states from the transformer
            attention_mask: Attention mask for the input
            
        Returns:
            Value estimates
        """
        pass
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_attention: bool = False,
                requires_grad: bool = False,
                return_last_hidden: bool = False,
                **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute value estimates for given inputs.
        
        Args:
            input_ids: Input token ids (state + action for sequence-level, state + action_so_far for token-level)
            attention_mask: Attention mask
            return_attention: Whether to return attention weights
            requires_grad: Whether to calculate gradients
            return_last_hidden: Whether to also return the last hidden states tensor
            **kwargs: Additional arguments
            
        Returns:
            Value estimates (and optionally attention weights)
        """
        # Determine target device: use model's input embedding device (first shard)
        model_device = self.model.get_input_embeddings().weight.device
        
        # Move inputs to the shard device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        
        # Ensure value head is on same device as hidden states
        self.value_head.to(model_device)
        
        # Forward pass through the model with gradient control
        with torch.set_grad_enabled(requires_grad):
            # MEMORY OPTIMIZATION: Create a custom forward pass that avoids loading the language model head
            # This prevents the massive memory allocation from the vocabulary projection
            def custom_forward(*args, **kwargs):
                kwargs.pop('labels', None)
                kwargs.pop('lm_labels', None)
                with disable_lora_adapters(self.model):
                    transformer_outputs = get_transformer_module(self.model)(
                        *args,
                        **kwargs
                    )
                return transformer_outputs
            
            # Get model outputs with activation clamping
            outputs = apply_activation_clamping(
                self.model,
                custom_forward,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention,
                return_dict=True,
                enable_hooks=self.enable_forward_hooks,
                clamp_range=self.forward_hook_clamp_range,
                layernorm_range=self.forward_hook_layernorm_range,
                # Memory-saving flags --------------------------------------------------
                use_cache=False,                    # (b) explicitly disable KV cache
                attn_implementation="flash_attention_2",  # (d) flash-attention
                # ---------------------------------------------------------------------
                **kwargs
            )
            
            # Get the last hidden states (consistent with policies.py usage)
            hidden_states = outputs.hidden_states[-1]
            
            # ------------------------------------------------------------------
            # MEMORY OPTIMISATION: the transformer backbone is *frozen* so we do
            # **not** require gradients w.r.t. its parameters.  However, leaving
            # the graph intact would force PyTorch to keep a full history of all
            # intermediate activations for *every* transformer layer – easily
            # tens of gigabytes for an 8-b model.
            #
            # We therefore detach the hidden states from the graph, freeing the
            # backbone activations immediately, and (optionally) mark the
            # detached tensor as requiring grad so that the small value-head can
            # still be updated.
            # ------------------------------------------------------------------
            hidden_states = hidden_states.detach()
            if requires_grad:
                hidden_states.requires_grad_(True)
            
            # Compute values using subclass-specific logic
            values = self._compute_values(hidden_states, attention_mask)
            
            # Get attention weights if requested
            attentions = outputs.attentions if return_attention else None
            
            # Optionally also return the last hidden states for downstream heuristics
            last_hidden = hidden_states if return_last_hidden else None
        
        # Return appropriate results based on flags
        if return_attention and return_last_hidden:
            return values, attentions, last_hidden
        if return_attention:
            return values, attentions
        if return_last_hidden:
            return values, last_hidden
        return values
            
    def cleanup_gpu_memory(self) -> None:
        """Explicitly clear GPU memory."""
        self.model.to('cpu')
        self.value_head.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SequenceLevelValueCritic(AbstractValueCritic):
    """
    High-level Q function for sequence-level rewards.
    
    Takes concatenated state+action tokens and predicts a scalar value 
    representing the expected Bellman reward (harm score) for the entire action sequence.
    This approximates Q_high(state, action) where action is a complete sequence.
    """
    
    def _init_value_head(self) -> None:
        """Initialize sequence-level value heads (Q_high and V_high)."""
        hidden_size = self.model.config.hidden_size
        # Q_high head: operates on concatenated state+action pooled representation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # V_high head: operates on state-only pooled representation
        self.state_value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        for module in list(self.value_head.modules()) + list(self.state_value_head.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Ensure Q/V heads are trainable and match backbone dtype (bf16 training)
        for param in self.value_head.parameters():
            param.requires_grad = True
        for param in self.state_value_head.parameters():
            param.requires_grad = True
        model_dtype = next(self.model.parameters()).dtype
        self.value_head = self.value_head.to(dtype=model_dtype)
        self.state_value_head = self.state_value_head.to(dtype=model_dtype)
        # Polyak target heads (same architecture; bootstrap uses online backbone + target head)
        import copy
        self.target_state_value_head = copy.deepcopy(self.state_value_head)
        self.target_value_head = copy.deepcopy(self.value_head)
        for p in list(self.target_state_value_head.parameters()) + list(self.target_value_head.parameters()):
            p.requires_grad = False
        self._critic_target_steps = 0
    
    @property
    def is_token_level(self) -> bool:
        """This is not a token-level critic."""
        return False

    def polyak_update_target_heads(self, tau: float = 0.005) -> None:
        """Soft-update target V/Q heads toward online heads. tau=1.0 is hard-copy."""
        if tau < 0.0 or tau > 1.0:
            raise ValueError(f"critic polyak tau must be in [0, 1], got {tau}")
        with torch.no_grad():
            for online, target in (
                (self.state_value_head, self.target_state_value_head),
                (self.value_head, self.target_value_head),
            ):
                for op, tp in zip(online.parameters(), target.parameters()):
                    tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
        self._critic_target_steps += 1

    def hard_update_target_heads(self) -> None:
        """Hard-copy online heads into target heads."""
        self.polyak_update_target_heads(tau=1.0)
    
    def _compute_values(self, 
                       hidden_states: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence-level value by mean pooling over all tokens.
        
        Args:
            hidden_states: Hidden states from transformer [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Sequence-level values [batch]
        """
        # Mean pooling over all tokens
        pooled_hidden_states = hidden_states.mean(dim=1)
        
        # Apply the value head to get sequence-level value
        values = self.value_head(pooled_hidden_states).squeeze(-1)
        
        # Ensure consistent dtype with model
        model_dtype = next(self.model.parameters()).dtype
        values = values.to(dtype=model_dtype)
        
        return values

    def v_value(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_attention: bool = False,
                requires_grad: bool = False,
                return_last_hidden: bool = False,
                use_target_head: bool = False,
                **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute V_high(s) using the state-only V head.
        Mirrors forward() but uses state_value_head and expects state-only tokens.

        If use_target_head=True, applies the polyak target V head on backbone features
        (no grad through the head; for TD bootstrap).
        """
        model_device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        value_head = self.target_state_value_head if use_target_head else self.state_value_head
        value_head.to(model_device)
        # Target head path never needs grads; online head respects requires_grad.
        head_requires_grad = requires_grad and not use_target_head
        with torch.set_grad_enabled(head_requires_grad):
            def custom_forward(*args, **kwargs):
                kwargs.pop('labels', None)
                kwargs.pop('lm_labels', None)
                transformer_outputs = self.model.model(
                    *args,
                    **kwargs
                )
                return transformer_outputs
            outputs = apply_activation_clamping(
                self.model,
                custom_forward,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention,
                return_dict=True,
                enable_hooks=self.enable_forward_hooks,
                clamp_range=self.forward_hook_clamp_range,
                layernorm_range=self.forward_hook_layernorm_range,
                use_cache=False,
                attn_implementation="flash_attention_2",
                **kwargs
            )
            hidden_states = outputs.hidden_states[-1]
            hidden_states = hidden_states.detach()
            if head_requires_grad:
                hidden_states.requires_grad_(True)
            # Mean pool across tokens in the state
            pooled_hidden_states = hidden_states.mean(dim=1)
            if use_target_head:
                with torch.no_grad():
                    values = value_head(pooled_hidden_states).squeeze(-1)
            else:
                values = value_head(pooled_hidden_states).squeeze(-1)
            model_dtype = next(self.model.parameters()).dtype
            values = values.to(dtype=model_dtype)
            attentions = outputs.attentions if return_attention else None
            last_hidden = hidden_states if return_last_hidden else None
        if return_attention and return_last_hidden:
            return values, attentions, last_hidden
        if return_attention:
            return values, attentions
        if return_last_hidden:
            return values, last_hidden
        return values
    
    def save(self, save_path: str) -> None:
        """
        Save the critic model.
        
        Args:
            save_path: Path to save the model to
        """
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.value_head.state_dict(), os.path.join(save_path, "sequence_q_value_head.pt"))
        torch.save(self.state_value_head.state_dict(), os.path.join(save_path, "sequence_v_value_head.pt"))
        torch.save(self.target_value_head.state_dict(), os.path.join(save_path, "sequence_q_target_head.pt"))
        torch.save(self.target_state_value_head.state_dict(), os.path.join(save_path, "sequence_v_target_head.pt"))
        print(f"Saved sequence-level Q/V heads (incl. targets) to {save_path}")
        
    def load(self, load_path: str) -> None:
        """
        Load the critic model.
        
        Args:
            load_path: Path to load the model from
        """
        from src.utils import safe_load_state_dict
        
        q_head_path = os.path.join(load_path, "sequence_q_value_head.pt")
        v_head_path = os.path.join(load_path, "sequence_v_value_head.pt")
        if not os.path.exists(q_head_path):
            raise FileNotFoundError(
                f"Sequence Q head not found at {q_head_path}. This checkpoint is incomplete and cannot be used."
            )
        if not os.path.exists(v_head_path):
            raise FileNotFoundError(
                f"Sequence V head not found at {v_head_path}. This checkpoint is incomplete and cannot be used."
            )

        q_state = torch.load(q_head_path)
        success, error = safe_load_state_dict(self.value_head, {"value_head": q_state}, "value_head", strict=True)
        if not success:
            raise RuntimeError(error)
        v_state = torch.load(v_head_path)
        success, error = safe_load_state_dict(self.state_value_head, {"value_head": v_state}, "value_head", strict=True)
        if not success:
            raise RuntimeError(error)
        # Target heads optional for older checkpoints: default to online copy
        q_tgt_path = os.path.join(load_path, "sequence_q_target_head.pt")
        v_tgt_path = os.path.join(load_path, "sequence_v_target_head.pt")
        if os.path.exists(q_tgt_path) and os.path.exists(v_tgt_path):
            q_tgt = torch.load(q_tgt_path)
            success, error = safe_load_state_dict(
                self.target_value_head, {"value_head": q_tgt}, "value_head", strict=True
            )
            if not success:
                raise RuntimeError(error)
            v_tgt = torch.load(v_tgt_path)
            success, error = safe_load_state_dict(
                self.target_state_value_head, {"value_head": v_tgt}, "value_head", strict=True
            )
            if not success:
                raise RuntimeError(error)
        else:
            self.hard_update_target_heads()
        print(f"Loaded sequence-level Q/V heads from {load_path}")


class TokenLevelValueCritic(AbstractValueCritic):
    """
    Low-level Q function for token-level marginal contributions.
    
    Takes state+action_so_far tokens and predicts a scalar value representing
    the marginal contribution/Bellman reward for the last token in the sequence.
    This approximates Q_low(state, action_so_far) for the marginal utility of the next token.
    """
    
    def _init_value_head(self) -> None:
        """Initialize token-level value head with layer normalization."""
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @property
    def is_token_level(self) -> bool:
        """This is a token-level critic."""
        return True
    
    def _compute_values(self, 
                       hidden_states: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token values for marginal contribution of each token.
        
        Args:
            hidden_states: Hidden states from transformer [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Per-token values [batch, seq_len]
        """
        # Apply value head to each token position
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Mask out pad tokens
        values = values * attention_mask.float()
        
        # Ensure consistent dtype with model
        model_dtype = next(self.model.parameters()).dtype
        values = values.to(dtype=model_dtype)
        
        return values
    
    def save(self, save_path: str) -> None:
        """
        Save the critic model.
        
        Args:
            save_path: Path to save the model to
        """
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.value_head.state_dict(), os.path.join(save_path, "token_value_head.pt"))
        print(f"Saved token-level value head to {save_path}")
        
    def load(self, load_path: str) -> None:
        """
        Load the critic model.
        
        Args:
            load_path: Path to load the model from
        """
        from src.utils import safe_load_state_dict
        
        value_head_path = os.path.join(load_path, "token_value_head.pt")
        if not os.path.exists(value_head_path):
            raise FileNotFoundError(
                f"Token value head not found at {value_head_path}. This checkpoint is incomplete and cannot be used."
            )
            
        state_dict = torch.load(value_head_path)
        success, error = safe_load_state_dict(self.value_head, {"value_head": state_dict}, "value_head", strict=True)
        if not success:
            raise RuntimeError(error)
            
        print(f"Loaded token-level value head from {load_path}")


class TokenLevelValueCriticV(AbstractValueCritic):
    """
    Low-level V function for token-level value estimation in single-agent archer.
    
    Takes state tokens (not state+action) and predicts a scalar value representing
    the expected value the policy can obtain from a sequence of tokens.
    This approximates V_low(state) for the expected value of the current state.
    
    Key difference from TokenLevelValueCritic: This is a V function, not Q function.
    """
    
    def _init_value_head(self) -> None:
        """Initialize token-level V function head with layer normalization."""
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @property
    def is_token_level(self) -> bool:
        """This is a token-level V function critic."""
        return True
    
    def _compute_values(self, 
                       hidden_states: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token V values for expected value at each token position.
        
        Args:
            hidden_states: Hidden states from transformer [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Per-token V values [batch, seq_len]
        """
        # Apply value head to each token position
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Mask out pad tokens
        values = values * attention_mask.float()
        
        # Ensure consistent dtype with model
        model_dtype = next(self.model.parameters()).dtype
        values = values.to(dtype=model_dtype)
        
        return values
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_attention: bool = False,
                requires_grad: bool = False,
                return_last_hidden: bool = False,
                **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute V function estimates for given state inputs (no action concatenation).
        
        Args:
            input_ids: Input token ids (state only, not state + action)
            attention_mask: Attention mask
            return_attention: Whether to return attention weights
            requires_grad: Whether to calculate gradients
            return_last_hidden: Whether to also return the last hidden states tensor
            **kwargs: Additional arguments
            
        Returns:
            V function estimates (and optionally attention weights)
        """
        # Determine target device: use model's input embedding device (first shard)
        model_device = self.model.get_input_embeddings().weight.device
        
        # Move inputs to the shard device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        
        # Ensure value head is on same device as hidden states
        self.value_head.to(model_device)
        
        # Forward pass through the model with gradient control
        with torch.set_grad_enabled(requires_grad):
            # MEMORY OPTIMIZATION: Create a custom forward pass that avoids loading the language model head
            def custom_forward(*args, **kwargs):
                # Remove any logits-related arguments to prevent LM head computation
                kwargs.pop('labels', None)
                kwargs.pop('lm_labels', None)
                
                # Call the model's forward method but stop at hidden states
                # For models with a 'model' attribute (like LlamaForCausalLM)
                transformer_outputs = self.model.model(
                    *args,
                    **kwargs
                )
                
                return transformer_outputs
            
            # Get model outputs with activation clamping
            outputs = apply_activation_clamping(
                self.model,
                custom_forward,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention,
                return_dict=True,
                enable_hooks=self.enable_forward_hooks,
                clamp_range=self.forward_hook_clamp_range,
                layernorm_range=self.forward_hook_layernorm_range,
                # Memory-saving flags --------------------------------------------------
                use_cache=False,                    # (b) explicitly disable KV cache
                attn_implementation="flash_attention_2",  # (d) flash-attention
                # ---------------------------------------------------------------------
                **kwargs
            )
            
            # Get the last hidden states (consistent with policies.py usage)
            hidden_states = outputs.hidden_states[-1]
            
            # ------------------------------------------------------------------
            # MEMORY OPTIMISATION: the transformer backbone is *frozen* so we do
            # **not** require gradients w.r.t. its parameters.  However, leaving
            # the graph intact would force PyTorch to keep a full history of all
            # intermediate activations for *every* transformer layer – easily
            # tens of gigabytes for an 8-b model.
            #
            # We therefore detach the hidden states from the graph, freeing the
            # backbone activations immediately, and (optionally) mark the
            # detached tensor as requiring grad so that the small value-head can
            # still be updated.
            # ------------------------------------------------------------------
            hidden_states = hidden_states.detach()
            if requires_grad:
                hidden_states.requires_grad_(True)
            
            # Compute V values using subclass-specific logic
            values = self._compute_values(hidden_states, attention_mask)
            
            # Get attention weights if requested
            attentions = outputs.attentions if return_attention else None
            
            # Optionally also return the last hidden states for downstream heuristics
            last_hidden = hidden_states if return_last_hidden else None
        
        # Return appropriate results based on flags
        if return_attention and return_last_hidden:
            return values, attentions, last_hidden
        if return_attention:
            return values, attentions
        if return_last_hidden:
            return values, last_hidden
        return values
    
    def save(self, save_path: str) -> None:
        """
        Save the V function critic model.
        
        Args:
            save_path: Path to save the model to
        """
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.value_head.state_dict(), os.path.join(save_path, "token_v_function_head.pt"))
        print(f"Saved token-level V function head to {save_path}")
        
    def load(self, load_path: str) -> None:
        """
        Load the V function critic model.
        
        Args:
            load_path: Path to load the model from
        """
        from src.utils import safe_load_state_dict
        
        value_head_path = os.path.join(load_path, "token_v_function_head.pt")
        if not os.path.exists(value_head_path):
            raise FileNotFoundError(
                f"Token V function head not found at {value_head_path}. This checkpoint is incomplete and cannot be used."
            )
            
        state_dict = torch.load(value_head_path)
        success, error = safe_load_state_dict(self.value_head, {"value_head": state_dict}, "value_head", strict=True)
        if not success:
            raise RuntimeError(error)
            
        print(f"Loaded token-level V function head from {load_path}")
