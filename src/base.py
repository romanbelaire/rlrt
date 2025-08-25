"""
Base interfaces and abstract classes for the agent architecture.

This module defines the core abstractions that form the foundation of the 
refactored architecture, including base classes for models, policies, critics, 
and environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import os
import deepspeed
import uuid

# ---------------------------------------------------------------------------
# Ensure compiled Triton / DeepSpeed kernels are cached in scratch space so
# agent models reuse them exactly like the environment models.  This must be
# set *before* any call to deepspeed.init_inference() or Triton compilation.
# ---------------------------------------------------------------------------

import getpass

def get_user_path():
    """Get the full user path including first initial directory."""
    user_id = getpass.getuser()
    first_initial = user_id[0].lower()  # Get first letter of username
    return f"{first_initial}/{user_id}"

# Set cache directories with automatically inferred user path
user_path = get_user_path()
os.environ.setdefault("TRITON_CACHE_DIR", f"/common/scratch/users/{user_path}/triton_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/common/scratch/users/{user_path}/torch_extensions")

class ModelBase(nn.Module):
    """
    Base class for all models with shared functionality for device management.
    
    This class provides common utilities for managing model devices, memory usage,
    and model state that are used across different model implementations.
    """
    def __init__(self) -> None:
        super().__init__()
        self.base_model: Optional[PreTrainedModel] = None
        self.base_device: str = "cpu"
        self.compute_device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.ds_engine = None  # DeepSpeed inference engine
        
    def to_device(self, device: str) -> None:
        """
        Move model to a specific device with proper cleanup.
        
        Args:
            device: Device to move the model to ("cpu", "cuda", "cuda:0", etc.)
        """
        if self.ds_engine is not None:
            raise RuntimeError("Model is using DeepSpeed. Device placement is managed by DeepSpeed.")

        previous_device = self.base_device
        self.base_device = device
        
        # Move model to new device
        self.to(device)
        
        # Clean up previous device if moving from GPU to CPU
        if previous_device.startswith("cuda") and device == "cpu":
            self.clear_gpu_cache()
    
    @staticmethod
    def clear_gpu_cache() -> None:
        """Clean up GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Policy(ABC):
    """
    Abstract interface for policy implementations.
    
    A policy defines how the agent selects actions based on the current state.
    """
    @abstractmethod
    def act(self, 
            state: Union[List[str], torch.Tensor], 
            base_state: Optional[Union[List[str], torch.Tensor]] = None,
            **kwargs) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate an action based on the current state.
        
        Args:
            state: Current state representation
            base_state: Original state for reference (optional)
            **kwargs: Additional arguments
            
        Returns:
            tuple: (actions, action_tokens, log_probs)
        """
        pass
    
    @abstractmethod
    def get_log_probs(self, 
                      input_ids: torch.Tensor, 
                      output_ids: torch.Tensor, 
                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities of output sequence given input sequence.
        
        Args:
            input_ids: Input sequence
            output_ids: Output sequence
            **kwargs: Additional arguments
            
        Returns:
            tuple: (log_probs, entropy_loss)
        """
        pass
    
    @abstractmethod
    def get_embed_similarity(self, 
                             sequences1: Union[List[str], torch.Tensor], 
                             sequences2: Union[List[str], torch.Tensor], 
                             **kwargs) -> torch.Tensor:
        """
        Compute embedding similarity between two sets of sequences.
        
        Args:
            sequences1: First set of sequences
            sequences2: Second set of sequences
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Similarity scores
        """
        pass


class Critic(ABC):
    """
    Abstract interface for critic implementations.
    
    A critic estimates value functions used in policy updates.
    """
    @abstractmethod
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute value estimates for given inputs.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Value estimates (and optionally token-level scores)
        """
        pass
    
    @property
    @abstractmethod
    def is_token_level(self) -> bool:
        """Whether this is a token-level critic."""
        pass


class Environment(ABC):
    """
    Abstract interface for environments.
    
    An environment defines the interaction dynamics and reward signals.
    """
    @abstractmethod
    def reset(self) -> List[List[str]]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, 
             actions: Union[List[str], str], 
             **kwargs) -> Tuple[List[List[str]], List[str], torch.Tensor, torch.Tensor]:
        """
        Take a step in the environment with the given actions.
        
        Args:
            actions: Actions to take
            **kwargs: Additional arguments
            
        Returns:
            tuple: (next_state, responses, rewards, dones)
        """
        pass
    
    @abstractmethod
    def score_safety(self, 
                    responses: List[str], 
                    **kwargs) -> torch.Tensor:
        """
        Evaluate the safety of responses.
        
        Args:
            responses: Responses to evaluate
            **kwargs: Additional arguments
            
        Returns:
            Safety scores
        """
        pass

class Conversation:
    """
    A conversation construct that tracks one conversation thread.
    
    This class maintains separate lists for user and assistant messages
    and provides methods to access the conversation in various formats.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize a new conversation.
        
        Args:
            conversation_id: Optional unique identifier for this conversation
        """
        self.conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        self.user_messages: List[str] = []
        self.assistant_messages: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.high_level_instructions: List[str] = []
        self.template_ids: List[str] = []  # Track template ID for each user message
        self.original_harmful_prompt: Optional[str] = None
        
    def add_user_message(self, message: str, high_level_instruction: Optional[str] = None, template_id: Optional[str] = None) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            message: The user's message
            high_level_instruction: The high-level instruction for this user message (optional)
            template_id: The template ID used for this user message (optional)
        """
        self.user_messages.append(message)
        # Add corresponding high-level instruction (empty string if none provided)
        instruction = high_level_instruction if high_level_instruction is not None else ""
        self.high_level_instructions.append(instruction)
        # Add corresponding template ID (empty string if none provided)
        template_id_str = template_id if template_id is not None else ""
        self.template_ids.append(template_id_str)
        
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            message: The assistant's message
        """
        self.assistant_messages.append(message)
        
    def add_qa_pair(self, user_message: str, assistant_message: str, high_level_instruction: Optional[str] = None, template_id: Optional[str] = None) -> None:
        """
        Add a complete Q&A pair to the conversation.
        
        Args:
            user_message: The user's message/question
            assistant_message: The assistant's response
            high_level_instruction: The high-level instruction for this user message (optional)
            template_id: The template ID used for this user message (optional)
        """
        self.add_user_message(user_message, high_level_instruction, template_id)
        self.add_assistant_message(assistant_message)
        
    def set_high_level_instruction(self, index: int, instruction: str) -> None:
        """
        Set a high-level instruction at a specific index.
        
        Args:
            index: Index of the instruction to set
            instruction: The high-level instruction that guides the agent
        """
        # Extend the list if needed
        while len(self.high_level_instructions) <= index:
            self.high_level_instructions.append("")
        self.high_level_instructions[index] = instruction
        
    def get_high_level_instruction(self, index: int) -> Optional[str]:
        """
        Get a high-level instruction at a specific index.
        
        Args:
            index: Index of the instruction to get
            
        Returns:
            The high-level instruction at that index, or None if not found
        """
        if 0 <= index < len(self.high_level_instructions):
            return self.high_level_instructions[index] if self.high_level_instructions[index] else None
        return None
        
    def get_high_level_instructions(self) -> List[str]:
        """Get all high-level instructions for this conversation."""
        return self.high_level_instructions.copy()
        
    def add_high_level_instruction(self, instruction: str) -> None:
        """
        Add a high-level instruction to the list.
        
        Args:
            instruction: The high-level instruction to add
        """
        self.high_level_instructions.append(instruction)
        
    def ensure_instructions_sync(self) -> None:
        """
        Ensure the high_level_instructions list has the same length as user_messages.
        Fill with empty strings if needed.
        """
        while len(self.high_level_instructions) < len(self.user_messages):
            self.high_level_instructions.append("")
        
        # Truncate if too long (shouldn't happen normally)
        if len(self.high_level_instructions) > len(self.user_messages):
            self.high_level_instructions = self.high_level_instructions[:len(self.user_messages)]
        
    def set_original_harmful_prompt(self, prompt: str) -> None:
        """
        Set the original harmful prompt for this conversation.
        
        Args:
            prompt: The original harmful prompt that initiated this conversation
        """
        self.original_harmful_prompt = prompt
        
    def get_original_harmful_prompt(self) -> Optional[str]:
        """Get the original harmful prompt for this conversation."""
        return self.original_harmful_prompt
        
    def get_agent_context(self) -> List[Dict[str, str]]:
        """
        Get conversation context for the agent (includes high-level instructions).
        
        For the agent:
        - Target model messages are labeled as "user" (the agent sees target as the user)
        - Agent messages are labeled as "assistant" (the agent sees itself as the assistant)
        - High-level instructions are included as system messages
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        context = []
        
        # Add the most recent high-level instruction if available
        if self.high_level_instructions and len(self.high_level_instructions) > 0:
            last_instruction = self.high_level_instructions[-1]
            if last_instruction:  # Only add if not empty
                context.append({
                    'role': 'system',
                    'content': last_instruction
                })
        
        # Add conversation history with role swapping for agent context
        max_length = max(len(self.user_messages), len(self.assistant_messages))
        for i in range(max_length):
            if i < len(self.user_messages):
                # Agent messages (originally "user") become "assistant"
                context.append({
                    'role': 'assistant',
                    'content': self.user_messages[i]
                })
                    
            if i < len(self.assistant_messages):
                # Target messages (originally "assistant") become "user"
                context.append({
                    'role': 'user',
                    'content': self.assistant_messages[i]
                })
        
        return context
        
    def get_target_context(self) -> List[Dict[str, str]]:
        """
        Get conversation context for the target model (conversation only).
        
        For the target model:
        - Agent messages are labeled as "user" (target sees agent as the user)
        - Target model messages are labeled as "assistant" (target sees itself as the assistant)
        - No high-level instructions
        - No original harmful prompt
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        context = []
        max_length = max(len(self.user_messages), len(self.assistant_messages))
        
        for i in range(max_length):
            if i < len(self.user_messages):
                # Agent messages (originally "user") stay as "user" for target
                context.append({
                    'role': 'user',
                    'content': self.user_messages[i]
                })
                    
            if i < len(self.assistant_messages):
                # Target messages (originally "assistant") stay as "assistant" for target
                context.append({
                    'role': 'assistant',
                    'content': self.assistant_messages[i]
                })
        
        return context
        
    def get_agent_history(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history for the agent in structured format.
        
        For the agent:
        - Target model messages are labeled as "user" (the agent sees target as the user)
        - Agent messages are labeled as "assistant" (the agent sees itself as the assistant)
        - High-level instructions are included as system messages
        
        Args:
            max_turns: Maximum number of turns to include (None for all)
            
        Returns:
            List of dictionaries with 'role' and 'content' keys, including system messages
        """
        history = []
        max_length = max(len(self.user_messages), len(self.assistant_messages))
        
        # Apply max_turns limit if specified
        if max_turns is not None:
            max_length = min(max_length, max_turns)
        
        # Add the most recent high-level instruction as system message if available
        if self.high_level_instructions and len(self.high_level_instructions) > 0:
            last_instruction = self.high_level_instructions[-1]
            if last_instruction:  # Only add if not empty
                history.append({
                    'role': 'system',
                    'content': last_instruction
                })
        
        for i in range(max_length):
            # Add agent message if available (role swapped to "assistant")
            if i < len(self.user_messages):
                history.append({
                    'role': 'assistant',
                    'content': self.user_messages[i]
                })
                
            # Add target message if available (role swapped to "user")
            if i < len(self.assistant_messages):
                history.append({
                    'role': 'user',
                    'content': self.assistant_messages[i]
                })
        
        return history
        
    def get_target_history(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history for the target model in structured format.
        
        Args:
            max_turns: Maximum number of turns to include (None for all)
            
        Returns:
            List of dictionaries with 'role' and 'content' keys, no system message
        """
        return self.get_conversation_history(max_turns=max_turns)
        
    def get_full_conversation(self) -> List[Dict[str, str]]:
        """
        Get the entire conversation as a list of message dictionaries.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        conversation = []
        max_length = max(len(self.user_messages), len(self.assistant_messages))
        
        for i in range(max_length):
            if i < len(self.user_messages):
                conversation.append({
                    'role': 'user',
                    'content': self.user_messages[i]
                })
                    
            if i < len(self.assistant_messages):
                conversation.append({
                    'role': 'assistant',
                    'content': self.assistant_messages[i]
                })
                    
        return conversation
        
    def get_qa_pair(self, index: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Get a specific Q&A pair by index.
        
        Args:
            index: Index of the Q&A pair to retrieve
            
        Returns:
            Tuple of (user_message, assistant_message), with None if index out of bounds
        """
        user_msg = self.user_messages[index] if index < len(self.user_messages) else None
        assistant_msg = self.assistant_messages[index] if index < len(self.assistant_messages) else None
        return user_msg, assistant_msg
        
    def get_last_qa_pair(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the most recent Q&A pair.
        
        Returns:
            Tuple of (user_message, assistant_message), with None if no messages
        """
        if not self.user_messages and not self.assistant_messages:
            return None, None
            
        last_user_idx = len(self.user_messages) - 1
        last_assistant_idx = len(self.assistant_messages) - 1
        
        user_msg = self.user_messages[last_user_idx] if last_user_idx >= 0 else None
        assistant_msg = self.assistant_messages[last_assistant_idx] if last_assistant_idx >= 0 else None
        
        return user_msg, assistant_msg
        
    def get_conversation_history(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history in a structured format.
        
        Args:
            max_turns: Maximum number of turns to include (None for all)
            
        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        history = []
        max_length = max(len(self.user_messages), len(self.assistant_messages))
        
        # Apply max_turns limit if specified
        if max_turns is not None:
            max_length = min(max_length, max_turns)
            
        for i in range(max_length):
            if i < len(self.user_messages):
                history.append({
                    'role': 'user',
                    'content': self.user_messages[i]
                })
                
            if i < len(self.assistant_messages):
                history.append({
                    'role': 'assistant', 
                    'content': self.assistant_messages[i]
                })
                
        return history
        
    def get_conversation_text(self, separator: str = "\n", include_roles: bool = True) -> str:
        """
        Get the entire conversation as a single text string.
        
        Args:
            separator: String to use between messages
            include_roles: Whether to include role prefixes
            
        Returns:
            Conversation as a single string
        """
        messages = self.get_full_conversation(include_roles=include_roles)
        return separator.join(messages)
        
    def get_user_messages(self) -> List[str]:
        """Get all user messages."""
        return self.user_messages.copy()
        
    def get_assistant_messages(self) -> List[str]:
        """Get all assistant messages."""
        return self.assistant_messages.copy()
        
    def get_turn_count(self) -> int:
        """Get the number of complete turns (Q&A pairs) in the conversation."""
        return min(len(self.user_messages), len(self.assistant_messages))
        
    def get_message_count(self) -> int:
        """Get the total number of messages in the conversation."""
        return len(self.user_messages) + len(self.assistant_messages)
        
    def is_complete_turn(self) -> bool:
        """Check if the conversation has an equal number of user and assistant messages."""
        return len(self.user_messages) == len(self.assistant_messages)
        
    def is_waiting_for_assistant(self) -> bool:
        """Check if the conversation is waiting for an assistant response."""
        return len(self.user_messages) > len(self.assistant_messages)
        
    def is_waiting_for_user(self) -> bool:
        """Check if the conversation is waiting for a user message."""
        return len(self.assistant_messages) >= len(self.user_messages)
        
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.user_messages.clear()
        self.assistant_messages.clear()
        self.metadata.clear()
        self.high_level_instructions.clear()
        self.template_ids.clear()
        self.original_harmful_prompt = None
        
    def copy(self, new_conversation_id: Optional[str] = None) -> 'Conversation':
        """
        Create a copy of the conversation with a new ID.
        
        Args:
            new_conversation_id: ID for the new conversation (auto-generated if None)
            
        Returns:
            New Conversation instance with copied data
        """
        # Generate new ID if not provided
        if new_conversation_id is None:
            new_conversation_id = f"{self.conversation_id}_copy"
        
        # Create new conversation
        new_conv = Conversation(new_conversation_id)
        
        # Copy messages
        new_conv.user_messages = self.user_messages.copy()
        new_conv.assistant_messages = self.assistant_messages.copy()
        
        # Copy additional fields
        new_conv.high_level_instructions = self.high_level_instructions.copy()
        new_conv.template_ids = self.template_ids.copy()
        new_conv.original_harmful_prompt = self.original_harmful_prompt
        
        # Copy metadata
        new_conv.metadata = self.metadata.copy()
        new_conv.metadata['copied_from'] = self.conversation_id
        
        return new_conv
        
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the conversation."""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata for the conversation."""
        return self.metadata.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert conversation to dictionary format.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            'conversation_id': self.conversation_id,
            'user_messages': self.user_messages.copy(),
            'assistant_messages': self.assistant_messages.copy(),
            'high_level_instructions': self.high_level_instructions.copy(),
            'template_ids': self.template_ids.copy(),
            'original_harmful_prompt': self.original_harmful_prompt,
            'metadata': self.metadata.copy(),
            'created_at': getattr(self, 'created_at', None),
            'updated_at': getattr(self, 'updated_at', None)
        }
        
    def save_to_json(self, file_path: str, indent: int = 2) -> Dict[str, Any]:
        """
        Save conversation to a JSON file with role, text, and turn fields.
        
        Args:
            file_path: Path to save the JSON file
            indent: JSON indentation level for pretty printing
            
        Returns:
            Dictionary representation of the conversation for batch processing
        """
        import json
        import os
        
        # Create conversation data with role, text, and turn fields
        conversation_data = {
            "conversation_id": self.conversation_id,
            "high_level_instructions": self.high_level_instructions,
            "original_harmful_prompt": self.original_harmful_prompt,
            "metadata": self.metadata.copy(),
            "messages": []
        }
        
        # Get the full conversation as alternating messages
        full_conversation = self.get_full_conversation()
        
        # Process messages with role, text, and turn information
        turn = 1
        for i, message in enumerate(full_conversation):
            # Determine role based on position (user starts first)
            role = "user" if i % 2 == 0 else "assistant"
            
            message_entry = {
                "role": role,
                "text": message,
                "turn": turn
            }
            conversation_data["messages"].append(message_entry)
            
            # Increment turn after each assistant response (complete turn)
            if role == "assistant":
                turn += 1
        
        # Ensure directory exists
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=indent, ensure_ascii=False)
            
            print(f"Conversation saved to {file_path}")
        
        return conversation_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        Create conversation from dictionary format.
        
        Args:
            data: Dictionary containing conversation data
            
        Returns:
            Conversation instance
        """
        conv = cls(conversation_id=data.get('conversation_id'))
        conv.user_messages = data.get('user_messages', [])
        conv.assistant_messages = data.get('assistant_messages', [])
        conv.high_level_instructions = data.get('high_level_instructions', [])
        conv.template_ids = data.get('template_ids', [])
        conv.original_harmful_prompt = data.get('original_harmful_prompt')
        conv.metadata = data.get('metadata', {})
        return conv
        
    @classmethod
    def from_message_list(cls, messages: List[Dict[str, str]], conversation_id: Optional[str] = None) -> 'Conversation':
        """
        Create conversation from a list of messages with role/content format.
        
        Args:
            messages: List of dictionaries with 'role' and 'content' keys
            conversation_id: Optional conversation ID
            
        Returns:
            Conversation instance
        """
        conv = cls(conversation_id=conversation_id)
        
        for message in messages:
            role = message.get('role', '').lower()
            content = message.get('content', '')
            
            if role == 'user':
                conv.add_user_message(content)
            elif role == 'assistant':
                conv.add_assistant_message(content)
                
        return conv
        
    def __len__(self) -> int:
        """Return the number of complete turns."""
        return self.get_turn_count()
        
    def __str__(self) -> str:
        """String representation of the conversation."""
        return f"Conversation({self.conversation_id}): {self.get_turn_count()} turns, {self.get_message_count()} messages"
        
    def __repr__(self) -> str:
        """Detailed string representation."""
        has_instructions = len(self.high_level_instructions) > 0
        has_harmful_prompt = self.original_harmful_prompt is not None
        return f"Conversation(id='{self.conversation_id}', user_msgs={len(self.user_messages)}, assistant_msgs={len(self.assistant_messages)}, instructions={len(self.high_level_instructions)}, has_harmful_prompt={has_harmful_prompt})"

    def get_template_id(self, index: int) -> Optional[str]:
        """
        Get a template ID at a specific index.
        
        Args:
            index: Index of the template ID to get
            
        Returns:
            The template ID at that index, or None if not found
        """
        if 0 <= index < len(self.template_ids):
            return self.template_ids[index] if self.template_ids[index] else None
        return None
        
    def get_template_ids(self) -> List[str]:
        """Get all template IDs for this conversation."""
        return self.template_ids.copy()
        
    def add_template_id(self, template_id: str) -> None:
        """
        Add a template ID to the list.
        
        Args:
            template_id: The template ID to add
        """
        self.template_ids.append(template_id)
        
    def set_template_id(self, index: int, template_id: str) -> None:
        """
        Set a template ID at a specific index.
        
        Args:
            index: Index of the template ID to set
            template_id: The template ID that was used for this message
        """
        # Extend the list if needed
        while len(self.template_ids) <= index:
            self.template_ids.append("")
        self.template_ids[index] = template_id
        
    def get_latest_template_id(self) -> Optional[str]:
        """
        Get the most recent template ID from the conversation.
        
        Returns:
            The most recent template ID, or None if no template IDs exist
        """
        if self.template_ids:
            # Return the last non-empty template ID
            for template_id in reversed(self.template_ids):
                if template_id:
                    return template_id
        return None
        
    def ensure_template_ids_sync(self) -> None:
        """
        Ensure the template_ids list has the same length as user_messages.
        Fill with empty strings if needed.
        """
        while len(self.template_ids) < len(self.user_messages):
            self.template_ids.append("")
        
        # Truncate if too long (shouldn't happen normally)
        if len(self.template_ids) > len(self.user_messages):
            self.template_ids = self.template_ids[:len(self.user_messages)]

class BaseModelSingleton:
    """
    Singleton class for managing model instances.
    
    This provides centralized management of model resources across components.
    """
    _instances = {}
    
    @classmethod
    def get_instance(cls, model_name, hf_token=None, use_offloading=False, tokenizer=None, config=None, ds_config=None, prefer_gpu=None, quantize=None, instance_tag: str = None):
        """
        Get or create a singleton instance for the given model.
        
        Args:
            model_name: Name of the model
            hf_token: HuggingFace token for authentication
            use_offloading: Whether to use CPU offloading
            tokenizer: Tokenizer to use (can be None for delayed initialization)
            config: Model configuration
            ds_config: DeepSpeed configuration
            prefer_gpu: Preferred GPU index (0, 1, etc.) or None for auto placement
            quantize: Quantization setting (False, 4, or 8) - overrides training mixed_precision
            instance_tag: Optional tag to distinguish instances sharing the same model_name (e.g., 'agent', 'critic')
            
        Returns:
            BaseModelSingleton instance
        """
        key = (model_name, instance_tag)
        if key not in cls._instances:
            cls._instances[key] = cls(
                model_name=model_name,
                hf_token=hf_token,
                use_offloading=use_offloading,
                tokenizer=tokenizer,
                config=config,
                ds_config=ds_config,
                prefer_gpu=prefer_gpu,
                quantize=quantize
            )
        return cls._instances[key]
    
    def _get_device_map_strategy(self):
        """
        Determine device map strategy based on configuration and available GPUs.
        
        Strategy: GPU 0 for agent training, GPU 1 for inference models (target + guard)
        
        Returns:
            Device map for model loading
        """
        if self.use_offloading:
            return None  # CPU offloading mode
        
        # Respect config flag: only enable device_map when explicitly requested
        cfg_enable = False
        if self.config is not None and hasattr(self.config, "model"):
            cfg_enable = bool(getattr(self.config.model, "enable_device_map", False))
        if not cfg_enable:
            return None
        
        if not torch.cuda.is_available():
            return "cpu"
        
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 1:
            return "auto"
        
        # Multi-GPU setup with explicit targeting
        if self.prefer_gpu is not None and self.prefer_gpu < num_gpus:
            print(f"BaseModelSingleton: targeting {self.model_name} for GPU {self.prefer_gpu}")
            # Place the entire model on the preferred GPU by returning an explicit device map.
            # Using a dict avoids implicit defaulting to GPU 0 that occurs when a string like
            # "cuda:1" is passed to `device_map`.
            return {"": self.prefer_gpu}
        else:
            return "auto"

    def __init__(self, model_name, hf_token=None, use_offloading=False, tokenizer=None, config=None, ds_config=None, prefer_gpu=None, quantize=None):
        """
        Initialize the singleton. Should not be called directly.
        Use get_instance() instead.
        
        Args:
            model_name: Name of the model
            hf_token: HuggingFace token for authentication
            use_offloading: Whether to use CPU offloading
            tokenizer: Tokenizer to use (can be None for delayed initialization)
            config: Model configuration
            ds_config: DeepSpeed configuration
            prefer_gpu: Preferred GPU index or None for auto placement
            quantize: Quantization setting (False, 4, or 8) - overrides training mixed_precision
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.use_offloading = use_offloading
        self.tokenizer = tokenizer
        self.config = config
        self.ds_config = ds_config
        self.prefer_gpu = prefer_gpu
        self.quantize = quantize
        self.base_model = None
        self.ds_engine = None
        self.peft_model = None
        self.available_adapters = {}
        self.active_adapter = None
        self.initialized = False
        self.initialize()
    
    def initialize(self):
        """Initialize the model with DeepSpeed if configured."""
        print(f"Initializing BaseModelSingleton with model: {self.model_name}")
        
        # Load the model with memory-aware settings
        # ------------------------------------------------------------------
        # 1) Determine precision - quantization setting overrides training mixed_precision
        if self.quantize is not None:
            # Use quantization setting from environment config
            if self.quantize is False:
                # No quantization - use bf16 for consistency with environment
                torch_dtype = torch.bfloat16
                print(f"Using bf16 precision (no quantization) for {self.model_name}")
            elif self.quantize == 4:
                # 4-bit quantization will be handled by BitsAndBytesConfig
                torch_dtype = torch.bfloat16
                print(f"Using 4-bit quantization for {self.model_name}")
            elif self.quantize == 8:
                # 8-bit quantization will be handled by BitsAndBytesConfig
                torch_dtype = torch.bfloat16
                print(f"Using 8-bit quantization for {self.model_name}")
            else:
                raise ValueError(f"Invalid quantize value: {self.quantize}. Must be False, 4, or 8.")
        else:
            # Fall back to training mixed_precision setting
            precision = "fp16"
            if self.config is not None and hasattr(self.config, "training"):
                prec_field = getattr(self.config.training, "mixed_precision", None)
                if prec_field:
                    precision = str(prec_field).lower()
            if precision == "bf16":
                torch_dtype = torch.bfloat16
            elif precision == "fp32":
                torch_dtype = torch.float32
            else:
                # Default fallback – fp16
                torch_dtype = torch.float16
            print(f"Using training mixed_precision setting: {precision} for {self.model_name}")

        # 2) Build max_memory map if specified in config
        max_memory = None
        if self.config is not None and hasattr(self.config, "training"):
            train_cfg = self.config.training
            if getattr(train_cfg, "max_gpu_memory", None):
                max_memory = {}
                if torch.cuda.is_available():
                    n_visible = torch.cuda.device_count()
                    n_gpus_cfg = getattr(train_cfg, "num_gpus", n_visible)
                    for i in range(min(n_visible, n_gpus_cfg)):
                        max_memory[i] = train_cfg.max_gpu_memory
                # Always include a CPU fallback
                max_memory["cpu"] = getattr(train_cfg, "max_cpu_memory", "60GiB")

        # 3) Determine device map strategy
        device_map = self._get_device_map_strategy()

        # 4) Actually load the checkpoint
        from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
        # We need the model config for certain compatibility tweaks
        model_config = AutoConfig.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        # Prepare model loading kwargs
        model_kwargs = {
            'token': self.hf_token,
            'torch_dtype': torch_dtype,
            'low_cpu_mem_usage': True,
            'trust_remote_code': True
        }
        if device_map is not None:
            model_kwargs['device_map'] = device_map
            # Build max_memory map if specified in config
            if max_memory is not None:
                model_kwargs['max_memory'] = max_memory
        else:
            # No device_map: fall back to single-device placement later
            pass
        
        # Add quantization config if specified
        if self.quantize == 4:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            model_kwargs['quantization_config'] = bnb_cfg
        elif self.quantize == 8:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs['quantization_config'] = bnb_cfg
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Ensure model is in bf16 for consistency (but skip if using bitsandbytes quantization)
        if torch_dtype == torch.bfloat16 and self.quantize not in [4, 8]:
            self.base_model = self.base_model.to(dtype=torch.bfloat16)
        
        # If device_map is disabled, move entire model to a single compute device
        if device_map is None:
            if torch.cuda.is_available():
                target_device = "cuda:0"
            else:
                target_device = "cpu"
            self.base_model.to(target_device)
        
        # Log device placement information
        if hasattr(self.base_model, 'hf_device_map'):
            print(f"BaseModelSingleton {self.model_name} device placement: {self.base_model.hf_device_map}")
        else:
            try:
                device = next(self.base_model.parameters()).device
                print(f"BaseModelSingleton {self.model_name} placed on device: {device}")
            except StopIteration:
                print(f"BaseModelSingleton {self.model_name} has no parameters to check device placement")
        
        # ------------------------------------------------------------------
        # DeepSpeed inference wrapper (optional)
        # ------------------------------------------------------------------
        if self.ds_config is not None:
            print("Initializing DeepSpeed inference engine...")
            # Use the supplied ds_config; if it is a path load the JSON
            ds_cfg = self.ds_config
            if isinstance(ds_cfg, str) and os.path.exists(ds_cfg):
                import json
                with open(ds_cfg, "r") as f:
                    ds_cfg = json.load(f)
            # Ensure dtype aligns with precision settings if not explicitly set
            ds_cfg.setdefault("dtype", precision)
            self.ds_engine = deepspeed.init_inference(
                model=self.base_model,
                config=ds_cfg,
                replace_with_kernel_inject=True
            )
            # Expose the underlying module so downstream code can call .generate()
            self.base_model = self.ds_engine.module
            print("DeepSpeed initialization complete")
        
            # --------------------------------------------------------------
            # One-time warm-up to trigger Flash-Attention / Triton kernel
            # compilation.  Outputs are cached in the directories above, so
            # subsequent processes skip this cost.
            # --------------------------------------------------------------
            if torch.cuda.is_available():
                with torch.no_grad():
                    dummy = torch.tensor([[self.base_model.config.bos_token_id or 1]], device="cuda")
                    _ = self.base_model.generate(dummy, max_new_tokens=1)
        
        # Track device assignments
        self.devices = {}  # Maps component names to devices
        self.devices["model"] = "deepspeed_managed" if self.ds_engine else "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Model loaded with DeepSpeed inference engine")
        
        # Now load tokenizer if provided
        if self.tokenizer is not None:
            # Ensure model and tokenizer vocab sizes match
            if len(self.tokenizer) != self.base_model.config.vocab_size:
                print(f"Resizing token embeddings: tokenizer has {len(self.tokenizer)} tokens, model has {self.base_model.config.vocab_size}")
                self.base_model.resize_token_embeddings(len(self.tokenizer))
                
            # Ensure tokenizer has proper padding settings
            if self.tokenizer:
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    else:
                        # Add pad token if needed
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Ensure num_heads and rope_theta are set on attention modules for DeepSpeed compatibility
        if hasattr(self.base_model, 'model'):
            for layer in self.base_model.model.layers:
                if hasattr(layer, 'self_attn'):
                    # Set num_heads if not present
                    if not hasattr(layer.self_attn, 'num_heads'):
                        layer.self_attn.num_heads = model_config.num_attention_heads
                    # Set rope_theta if not present
                    if not hasattr(layer.self_attn, 'rope_theta'):
                        layer.self_attn.rope_theta = 10000.0  # Default from Llama implementation
        
        self.initialized = True
        return self.base_model, self.tokenizer

    def initialize_with_lora(self, config):
        """
        Initialize the model with LoRA adapters.
        
        Args:
            config: Model configuration containing LoRA settings
        """
        if not self.initialized:
            self.initialize()
            
        # Import PEFT utilities
        from peft import get_peft_model, LoraConfig
        
        # Build LoRA config – allow config.target_layers == [-1] to mean "all layers"
        from utils import create_lora_config

        target_layers_cfg = None
        if hasattr(config, 'target_layers') and config.target_layers is not None:
            if (-1 not in config.target_layers):
                target_layers_cfg = config.target_layers

        lora_config = create_lora_config(
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            target_layers=target_layers_cfg,
            bias="none"
        )
        
        # Create PEFT model
        self.peft_model = get_peft_model(self.base_model, lora_config)
        self.available_adapters = {"base_model": True}
        self.active_adapter = "base_model"
        for n, p in self.peft_model.named_parameters():
            if "lora_" in n:          # LoRA A/B or scaling weights
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        # If adapter path is provided, load weights
        if hasattr(config, 'adapter_path') and config.adapter_path:
            self.peft_model.load_adapter(config.adapter_path, adapter_name="base_model")
        
        # Update base model reference
        self.base_model = self.peft_model
        
        print(f"Initialized LoRA with config: {lora_config}")
        return True

    def to_appropriate_devices(self):
        """Move models to appropriate devices based on configuration."""
        if self.ds_engine is not None:
            # DeepSpeed manages device placement
            return
            
        if self.use_offloading:
            # Move to CPU if offloading is enabled
            self.to("cpu")
        else:
            # Move to GPU if available
            self.to("cuda" if torch.cuda.is_available() else "cpu")

    def add_adapter(self, adapter_name: str, config: Any, adapter_path: Optional[str] = None) -> bool:
        """
        Add a LoRA adapter to the model.
        
        Args:
            adapter_name: Name of the adapter
            config: LoRA configuration or ModelConfig with LoRA parameters
            adapter_path: Path to adapter weights (optional)
            
        Returns:
            bool: Success flag
        """
        if adapter_name in self.available_adapters:
            print(f"Adapter {adapter_name} already exists")
            return True
            
        # Ensure base model is initialized
        if not self.initialized:
            self.initialize()
            
        # Import PEFT utilities
        from peft import PeftModel, get_peft_model, LoraConfig
        from utils import create_lora_config
        
        # Process the configuration
        lora_config = None
        
        # Check if config is a ModelConfig object
        if 'use_lora' in dir(config) and config.use_lora:
            # Create LoRA config from ModelConfig
            print(f"Creating LoRA config from ModelConfig")
            lora_config = create_lora_config(
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
                target_layers=config.target_layers
            )
        elif isinstance(config, dict):
            # Convert dict to LoraConfig
            print(f"Creating LoRA config from dictionary")
            target_modules = config.get('target_modules', ["q_proj", "v_proj"])
            target_layers = config.get('target_layers', [0, 1, 2])
            lora_config = create_lora_config(
                lora_r=config.get('r', 8),
                lora_alpha=config.get('lora_alpha', 16),
                lora_dropout=config.get('lora_dropout', 0.05),
                target_modules=target_modules,
                target_layers=target_layers,
                bias=config.get('bias', 'none')
            )
        else:
            # Assume it's already a LoraConfig
            lora_config = config
        
        # If we don't have a PEFT model yet, create one
        if self.peft_model is None:
            # For first adapter being added
            if adapter_path and os.path.exists(adapter_path):
                print(f"Loading adapter from {adapter_path}")
                # Load existing adapter
                self.peft_model = PeftModel.from_pretrained(
                    self.base_model,
                    adapter_path,
                    adapter_name=adapter_name
                )
            elif lora_config:
                print(f"Creating new adapter {adapter_name} with config")
                # Create new adapter with config
                self.peft_model = get_peft_model(
                    self.base_model,
                    lora_config,
                    adapter_name=adapter_name
                )
            else:
                print("Error: Either adapter_path or config must be provided")
                return False
            
            self.active_adapter = adapter_name
            
        else:
            # For subsequent adapters being added to existing PEFT model
            if adapter_path and os.path.exists(adapter_path):
                print(f"Loading adapter {adapter_name} from {adapter_path}")
                # Load adapter weights into existing PEFT model
                self.peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
            elif lora_config:
                print(f"Adding new adapter {adapter_name} to existing PEFT model")
                # Add new adapter config to existing PEFT model
                self.peft_model.add_adapter(adapter_name, lora_config)
            else:
                print("Error: Either adapter_path or config must be provided")
                return False
        
        # Record the new adapter
        self.available_adapters[adapter_name] = True
        
        # Initialize DeepSpeed for the PEFT model
        ds_config = {
            "dtype": str(self.ds_engine.module.dtype).split('.')[-1],
            "tensor_parallel": {"tp_size": 1},  # Single GPU for now
            "replace_with_kernel_inject": True,
            "enable_cuda_graph": False
        }
        
        # Only add offload_param if we want to offload to CPU
        if self.use_offloading:
            ds_config["offload_param"] = "cpu"
        
        # Initialize DeepSpeed-Inference engine for PEFT model
        self.ds_engine = deepspeed.init_inference(
            self.peft_model,
            config=ds_config,
        )
        
        # Update PEFT model reference
        self.peft_model = self.ds_engine.module
        
        # Print adapters after addition
        print(f"Available adapters after addition: {list(self.available_adapters.keys())}")
        print(f"Active adapter: {self.active_adapter}")
        
        return True

    def switch_adapter(self, adapter_name: str) -> bool:
        """
        Switch to a different LoRA adapter.
        
        Args:
            adapter_name: Name of the adapter to switch to
            
        Returns:
            bool: Success flag
        """
        if not self.peft_model:
            print("No PEFT model initialized")
            return False
            
        if adapter_name not in self.available_adapters:
            print(f"Adapter {adapter_name} not found. Available adapters: {list(self.available_adapters.keys())}")
            return False
            
        if self.active_adapter == adapter_name:
            print(f"Adapter {adapter_name} is already active")
            return True
            
        # Set the active adapter
        self.peft_model.set_adapter(adapter_name)
        self.active_adapter = adapter_name
        
        # Reinitialize DeepSpeed with the new adapter
        ds_config = {
            "dtype": str(self.ds_engine.module.dtype).split('.')[-1],
            "tensor_parallel": {"tp_size": 1},  # Single GPU for now
            "replace_with_kernel_inject": True,
            "enable_cuda_graph": False
        }
        
        # Only add offload_param if we want to offload to CPU
        if self.use_offloading:
            ds_config["offload_param"] = "cpu"
        
        # Initialize DeepSpeed-Inference engine for the new adapter
        self.ds_engine = deepspeed.init_inference(
            self.peft_model,
            config=ds_config,
            replace_with_kernel_inject=True
        )
        
        # Update PEFT model reference
        self.peft_model = self.ds_engine.module
        
        print(f"Switched to adapter: {adapter_name}")
        return True
    
    def merge_base_adapter(self) -> bool:
        """
        Merge the active adapter with the base model.
        
        Returns:
            bool: Success flag
        """
        if not self.peft_model:
            raise ValueError("No PEFT model initialized, nothing to merge")
            
        # Ensure the model is in eval mode for merging
        self.peft_model.eval()
        
        # Merge the weights
        print(f"Merging adapter {self.active_adapter} into base model")
        self.peft_model.merge_adapter()
        
        # Reset to base model
        self.base_model = self.peft_model.get_base_model()
        
        # Reinitialize DeepSpeed with the merged model
        ds_config = {
            "dtype": str(self.ds_engine.module.dtype).split('.')[-1],
            "tensor_parallel": {"tp_size": 1},  # Single GPU for now
            "replace_with_kernel_inject": True,
            "enable_cuda_graph": False
        }
        
        # Only add offload_param if we want to offload to CPU
        if self.use_offloading:
            ds_config["offload_param"] = "cpu"
        
        # Initialize DeepSpeed-Inference engine for merged model
        self.ds_engine = deepspeed.init_inference(
            self.base_model,
            config=ds_config,
            replace_with_kernel_inject=True
        )
        
        # Update base model reference
        self.base_model = self.ds_engine.module
        
        # Clear PEFT model
        self.peft_model = None
        self.active_adapter = None
        self.available_adapters = {}
        
        print("Successfully merged adapter into base model")
        return True
    
    def unload(self) -> None:
        """
        Unload models and free up memory.
        
        This method properly cleans up model resources and prepares for reloading.
        """
        # First check if we have references to models
        has_base_model = 'base_model' in self.__dict__ and self.base_model is not None
        has_peft_model = 'peft_model' in self.__dict__ and self.peft_model is not None
        has_ds_engine = 'ds_engine' in self.__dict__ and self.ds_engine is not None
        
        if has_base_model or has_peft_model or has_ds_engine:
            print("Unloading model and freeing memory...")
            
            # Clean up DeepSpeed engine
            if has_ds_engine:
                del self.ds_engine
                self.ds_engine = None
            
            # For PEFT models, we need to handle references differently
            if has_peft_model:
                self.peft_model = None
            
            # For the base model
            if has_base_model:
                self.base_model = None
        
        # Reset adapter related attributes
        self.active_adapter = None
        self.available_adapters = {}
        self.initialized = False
        
        # Run garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Model successfully unloaded and memory cleaned")
