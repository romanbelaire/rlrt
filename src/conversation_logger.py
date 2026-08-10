from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from src.base import Conversation


class ConversationLogger:
    """
    A logger for saving conversations to plain text files.
    
    This logger creates a single text file that contains only the most recent rollout
    conversations, overwriting previous rollouts to keep file size manageable.
    """
    
    def __init__(self, output_dir: str, enabled: bool = True):
        """
        Initialize the conversation logger.
        
        Args:
            output_dir: Directory to save conversation logs
            enabled: Whether logging is enabled
        """
        self.output_dir = output_dir
        self.enabled = enabled
        self.log_file_path: Optional[str] = None
        self.rollout_id: Optional[str] = None
        
        # Create output directory if it doesn't exist
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def start_rollout(self, rollout_id: Optional[str] = None) -> None:
        """
        Start a new rollout, overwriting the existing log file.
        
        Args:
            rollout_id: Optional identifier for this rollout (auto-generated if None)
        """
        if not self.enabled:
            return
            
        # Generate rollout ID if not provided
        if rollout_id is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            rollout_id = f"rollout_{timestamp}"
        
        self.rollout_id = rollout_id
        # Use a fixed filename to always overwrite the previous rollout
        self.log_file_path = os.path.join(self.output_dir, "conversations_latest.txt")
        
        # Write over any previous file (as requested)
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Conversation Log - Rollout: {rollout_id}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_conversations(self, conversations: List[Conversation], step: int = 0) -> None:
        """
        Log a batch of conversations for the current step.
        This method overwrites the file to only keep the most recent step.
        
        Args:
            conversations: List of Conversation objects to log
            step: Current step number
        """
        if not self.enabled or not self.log_file_path:
            return
        
        # Overwrite the file with only the current step to keep file size manageable
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Conversation Log - Rollout: {self.rollout_id}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Current Step: {step}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"STEP {step}\n")
            f.write("-" * 40 + "\n\n")
            
            for i, conv in enumerate(conversations):
                f.write(f"CONVERSATION {i + 1} (ID: {conv.conversation_id})\n")
                f.write("=" * 60 + "\n")
                
                # Log metadata if available
                if conv.metadata:
                    f.write("METADATA:\n")
                    for key, value in conv.metadata.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # Log conversation history
                f.write("CONVERSATION HISTORY:\n")
                f.write("-" * 30 + "\n")
                
                # Get the maximum number of turns
                max_turns = max(len(conv.user_messages), len(conv.assistant_messages))
                
                for turn in range(max_turns):
                    # User message
                    if turn < len(conv.user_messages):
                        user_msg = conv.user_messages[turn]
                        f.write(f"USER: {user_msg}\n")
                    
                    # Assistant message
                    if turn < len(conv.assistant_messages):
                        assistant_msg = conv.assistant_messages[turn]
                        f.write(f"ASSISTANT: {assistant_msg}\n")
                    
                    # High-level instruction if available
                    if turn < len(conv.high_level_instructions):
                        instruction = conv.high_level_instructions[turn]
                        if instruction.strip():
                            f.write(f"INSTRUCTION: {instruction}\n")
                    
                    f.write("\n")  # Empty line between turns
                
                f.write("\n" + "=" * 60 + "\n\n")
    
    def log_conversation_summary(self, conversations: List[Conversation], 
                                episode_rewards: Optional[List[float]] = None,
                                episode_dones: Optional[List[bool]] = None) -> None:
        """
        Log a summary of conversations at the end of an episode.
        This method overwrites the file to only keep the episode summary.
        
        Args:
            conversations: List of Conversation objects
            episode_rewards: Optional list of episode rewards
            episode_dones: Optional list of episode completion status
        """
        if not self.enabled or not self.log_file_path:
            return
        
        # Overwrite the file with only the episode summary to keep file size manageable
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Conversation Log - Rollout: {self.rollout_id}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("EPISODE SUMMARY\n")
            f.write("=" * 80 + "\n")
            
            for i, conv in enumerate(conversations):
                f.write(f"Conversation {i + 1}:\n")
                f.write(f"  ID: {conv.conversation_id}\n")
                f.write(f"  Total turns: {max(len(conv.user_messages), len(conv.assistant_messages))}\n")
                
                if episode_rewards is not None and i < len(episode_rewards):
                    f.write(f"  Episode reward: {episode_rewards[i]:.4f}\n")
                
                if episode_dones is not None and i < len(episode_dones):
                    f.write(f"  Episode done: {episode_dones[i]}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n\n")
    
    def log_final_conversations(self, conversations: List[Conversation], 
                               episode_rewards: Optional[List[float]] = None,
                               episode_dones: Optional[List[bool]] = None) -> None:
        """
        Log the final state of all conversations at the end of a rollout.
        This method overwrites the file to only keep the final conversation state.
        
        Args:
            conversations: List of Conversation objects
            episode_rewards: Optional list of episode rewards
            episode_dones: Optional list of episode completion status
        """
        if not self.enabled or not self.log_file_path:
            return
        
        # Overwrite the file with only the final conversation state
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Conversation Log - Rollout: {self.rollout_id}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("FINAL CONVERSATION STATE\n")
            f.write("=" * 80 + "\n\n")
            
            for i, conv in enumerate(conversations):
                f.write(f"CONVERSATION {i + 1} (ID: {conv.conversation_id})\n")
                f.write("=" * 60 + "\n")
                
                # Log metadata if available
                if conv.metadata:
                    f.write("METADATA:\n")
                    for key, value in conv.metadata.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # Log episode summary for this conversation
                f.write("EPISODE SUMMARY:\n")
                f.write(f"  Total turns: {max(len(conv.user_messages), len(conv.assistant_messages))}\n")
                
                if episode_rewards is not None and i < len(episode_rewards):
                    f.write(f"  Episode reward: {episode_rewards[i]:.4f}\n")
                
                if episode_dones is not None and i < len(episode_dones):
                    f.write(f"  Episode done: {episode_dones[i]}\n")
                
                f.write("\n")
                
                # Log conversation history
                f.write("CONVERSATION HISTORY:\n")
                f.write("-" * 30 + "\n")
                
                # Get the maximum number of turns
                max_turns = max(len(conv.user_messages), len(conv.assistant_messages))
                
                for turn in range(max_turns):
                    # User message
                    if turn < len(conv.user_messages):
                        user_msg = conv.user_messages[turn]
                        f.write(f"USER: {user_msg}\n")
                    
                    # Assistant message
                    if turn < len(conv.assistant_messages):
                        assistant_msg = conv.assistant_messages[turn]
                        f.write(f"ASSISTANT: {assistant_msg}\n")
                    
                    # High-level instruction if available
                    if turn < len(conv.high_level_instructions):
                        instruction = conv.high_level_instructions[turn]
                        if instruction.strip():
                            f.write(f"INSTRUCTION: {instruction}\n")
                    
                    f.write("\n")  # Empty line between turns
                
                f.write("\n" + "=" * 60 + "\n\n")
    
    def close(self) -> None:
        """Close the current log file."""
        if not self.enabled or not self.log_file_path:
            return
        
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"Log completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.log_file_path = None
        self.rollout_id = None


class EnvironmentConversationLogger:
    """
    A wrapper that provides conversation logging functionality for environments.
    
    This class can be used by both MultiTurnMultipleChoiceEnv and twenty questions
    environments to log their conversations.
    """
    
    def __init__(self, output_dir: str, enabled: bool = True):
        """
        Initialize the environment conversation logger.
        
        Args:
            output_dir: Directory to save conversation logs
            enabled: Whether logging is enabled
        """
        self.logger = ConversationLogger(output_dir, enabled)
        self.enabled = enabled
    
    def start_rollout(self, rollout_id: Optional[str] = None) -> None:
        """Start a new rollout."""
        self.logger.start_rollout(rollout_id)
    
    def log_step(self, conversations: List[Conversation], step: int = 0) -> None:
        """Log conversations for a specific step."""
        self.logger.log_conversations(conversations, step)
    
    def log_episode_summary(self, conversations: List[Conversation], 
                           episode_rewards: Optional[List[float]] = None,
                           episode_dones: Optional[List[bool]] = None) -> None:
        """Log episode summary."""
        self.logger.log_conversation_summary(conversations, episode_rewards, episode_dones)
    
    def log_final_conversations(self, conversations: List[Conversation], 
                               episode_rewards: Optional[List[float]] = None,
                               episode_dones: Optional[List[bool]] = None) -> None:
        """Log final conversation state at the end of a rollout."""
        self.logger.log_final_conversations(conversations, episode_rewards, episode_dones)
    
    def close(self) -> None:
        """Close the logger."""
        self.logger.close()
    
    def is_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self.enabled
