"""
Extended Conversation class for hierarchical 20 Questions agent architecture.

This module provides the TwentyQuestionsConversation class that extends the base
Conversation class with minimal 20Q-specific metadata for hierarchical training.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from src.base import Conversation


class TwentyQuestionsConversation(Conversation):
    """
    Extended Conversation class for hierarchical 20 Questions gameplay.
    
    This class extends the base Conversation with minimal 20Q-specific metadata
    needed for hierarchical agent training.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize a new 20 Questions conversation.
        
        Args:
            conversation_id: Optional unique identifier for this conversation
        """
        super().__init__(conversation_id)
        
        # Essential 20Q game state
        self.target_word: Optional[str] = None
        self.word_variants: List[str] = []  # Alternative names for the same word
        self.question_count: int = 0
        self.max_questions: int = 20
        self.episode_won: bool = False
        self.episode_done: bool = False
        
        # Question and answer tracking (legacy - kept for compatibility)
        self.questions_asked: List[str] = []
        self.oracle_answers: List[str] = []
        
        # High-level strategy tracking (legacy - kept for compatibility)
        self.hl_strategies: List[str] = []  # High-level strategy instructions
        
        # Reward tracking for dual training (legacy - kept for compatibility)
        self.hl_rewards: List[float] = []  # High-level strategy rewards
        self.ll_rewards: List[float] = []  # Low-level question quality rewards
        
        # Optimized separate storage for context length reduction
        self.oracle_qa_tuples: List[Tuple[str, str]] = []  # (question, answer) pairs
        self.hl_state_action_tuples: List[Tuple[str, str]] = []  # (state_summary, hl_instruction) pairs
        self.ll_state_action_tuples: List[Tuple[str, str]] = []  # (context, ll_question) pairs
        
        # Set task-specific metadata
        self.set_metadata("task", "hierarchical_twenty_questions")
        self.set_metadata("game_status", "active")
    
    def set_target_word(self, word: Union[str, List[str]]) -> None:
        """
        Set the target word for this game.
        
        Args:
            word: Target word (string) or list of word variants
        """
        if isinstance(word, list):
            self.target_word = word[0]  # Primary word
            self.word_variants = word.copy()
        else:
            self.target_word = word
            self.word_variants = [word]
        
        self.set_metadata("target_word", self.target_word)
        self.set_metadata("word_variants", self.word_variants)
    
    def add_question_answer_pair(
        self, 
        question: str, 
        answer: str,
        hl_strategy: Optional[str] = None,
        hl_reward: Optional[float] = None,
        ll_reward: Optional[float] = None
    ) -> None:
        """
        Add a complete question-answer pair with hierarchical metadata.
        
        Args:
            question: The agent's question
            answer: The oracle's answer
            hl_strategy: High-level strategy instruction that led to this question
            hl_reward: Reward for the high-level strategy
            ll_reward: Reward for the low-level question quality
        """
        # Add to base conversation
        self.add_user_message(question, high_level_instruction=hl_strategy)
        self.add_assistant_message(answer)
        
        # Track 20Q-specific information (legacy)
        self.questions_asked.append(question)
        self.oracle_answers.append(answer)
        self.question_count += 1
        
        # Track high-level strategy (legacy)
        if hl_strategy:
            self.hl_strategies.append(hl_strategy)
        else:
            self.hl_strategies.append("")
        
        # Track rewards (legacy)
        if hl_reward is not None:
            self.hl_rewards.append(hl_reward)
        if ll_reward is not None:
            self.ll_rewards.append(ll_reward)
        
        # Optimized storage - add to separate tuples
        self.oracle_qa_tuples.append((question, answer))
        
        # Create state summary for HL storage
        state_summary = self._create_state_summary()
        if hl_strategy:
            self.hl_state_action_tuples.append((state_summary, hl_strategy))
        
        # Create context for LL storage (only current HL instruction, not historical)
        ll_context = self._create_ll_context_for_current_step(hl_strategy or "")
        self.ll_state_action_tuples.append((ll_context, question))
        
        # Update metadata
        self.set_metadata("question_count", self.question_count)
    
    
    def set_episode_result(self, won: bool, done: bool = True) -> None:
        """
        Set the final result of the episode.
        
        Args:
            won: Whether the episode was won
            done: Whether the episode is complete
        """
        self.episode_won = won
        self.episode_done = done
        
        if won:
            self.set_metadata("game_status", "won")
        elif done:
            self.set_metadata("game_status", "lost")
        
        self.set_metadata("episode_won", won)
        self.set_metadata("episode_done", done)
    
    def get_information_summary(self) -> Dict[str, Any]:
        """
        Get a structured summary of information gathered so far.
        
        Returns:
            Dictionary containing information state summary
        """
        return {
            "target_word": self.target_word,
            "question_count": self.question_count,
            "questions_remaining": max(0, self.max_questions - self.question_count),
            "episode_won": self.episode_won,
            "episode_done": self.episode_done,
            "recent_questions": self.questions_asked[-3:] if len(self.questions_asked) >= 3 else self.questions_asked.copy(),
            "recent_answers": self.oracle_answers[-3:] if len(self.oracle_answers) >= 3 else self.oracle_answers.copy()
        }
    
    def get_conversation_context_for_hl(self) -> str:
        """
        Get formatted conversation context for high-level policy.
        
        Returns:
            Formatted string suitable for high-level policy input
        """
        info = self.get_information_summary()
        
        context_parts = [
            f"Questions asked: {info['question_count']}/{self.max_questions}",
            f"Questions remaining: {info['questions_remaining']}"
        ]
        
        # Add recent conversation history
        if self.questions_asked and self.oracle_answers:
            context_parts.append("\nRecent conversation:")
            for i in range(max(0, len(self.questions_asked) - 3), len(self.questions_asked)):
                if i < len(self.questions_asked) and i < len(self.oracle_answers):
                    context_parts.append(f"Q: {self.questions_asked[i]}")
                    context_parts.append(f"A: {self.oracle_answers[i]}")
        
        return "\n".join(context_parts)
    
    def get_conversation_context_for_ll(self, hl_instruction: str) -> str:
        """
        Get formatted conversation context for low-level policy.
        
        Args:
            hl_instruction: High-level instruction from the meta-cognitive agent
            
        Returns:
            Formatted string suitable for low-level policy input
        """
        context_parts = [f"Strategy from your partner: {hl_instruction}"]
        
        info = self.get_information_summary()
        context_parts.append(f"\nGame status: {info['question_count']}/{self.max_questions} questions used")
        
        # Add full conversation history for low-level agent
        if self.questions_asked and self.oracle_answers:
            context_parts.append("\nConversation so far:")
            for i in range(len(self.questions_asked)):
                if i < len(self.oracle_answers):
                    context_parts.append(f"Q{i+1}: {self.questions_asked[i]}")
                    context_parts.append(f"A{i+1}: {self.oracle_answers[i]}")
        
        return "\n".join(context_parts)
    
    def get_rewards_summary(self) -> Dict[str, Any]:
        """
        Get summary of rewards for both high-level and low-level policies.
        
        Returns:
            Dictionary containing reward information
        """
        return {
            "hl_rewards": self.hl_rewards.copy(),
            "ll_rewards": self.ll_rewards.copy(),
            "total_hl_reward": sum(self.hl_rewards) if self.hl_rewards else 0.0,
            "total_ll_reward": sum(self.ll_rewards) if self.ll_rewards else 0.0,
            "avg_hl_reward": sum(self.hl_rewards) / len(self.hl_rewards) if self.hl_rewards else 0.0,
            "avg_ll_reward": sum(self.ll_rewards) / len(self.ll_rewards) if self.ll_rewards else 0.0
        }
    
    def _create_state_summary(self) -> str:
        """
        Create a concise state summary for HL storage.
        
        Returns:
            Compact string summarizing current game state
        """
        return f"Q{self.question_count}/{self.max_questions}"
    
    def _create_ll_context_for_current_step(self, hl_instruction: str) -> str:
        """
        Create context for LL storage focused only on current step.
        
        Args:
            hl_instruction: Current high-level instruction
            
        Returns:
            Context string for current LL step
        """
        return f"Step{self.question_count}:{hl_instruction}"
    
    def get_optimized_ll_context(self, current_hl_instruction: str) -> str:
        """
        Get optimized context for LL agent with reduced context length.
        
        This reconstructs the conversation as: LL query 1, oracle answer 1, LL query 2, ..., current HL instruction
        
        Args:
            current_hl_instruction: Current high-level instruction
            
        Returns:
            Optimized context string for LL agent
        """
        context_parts = []
        
        # Add question-answer pairs in sequence
        for i, (question, answer) in enumerate(self.oracle_qa_tuples):
            context_parts.append(f"Q{i+1}: {question}")
            context_parts.append(f"A{i+1}: {answer}")
        
        # Add current HL instruction at the end (not historical ones)
        if current_hl_instruction:
            context_parts.append(f"Guidance: {current_hl_instruction}")
        
        return "\n".join(context_parts)
