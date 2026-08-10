"""
Data structures for semantic analysis data collection.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import torch


@dataclass
class ConversationTurn:
    """Data structure for a single conversation turn."""
    turn_number: int
    high_level_instruction: str
    low_level_action: str
    target_response: str
    llamaguard_score: float
    action_embedding: torch.Tensor
    response_embedding: torch.Tensor
    context_embedding: torch.Tensor
    v_high_score: float
    mean_q_low_score: float
    # Running max harm metrics up to and including this turn (R_{0:t})
    max_llamaguard_to_t: Optional[float] = None
    max_v_high_to_t: Optional[float] = None
    max_q_low_to_t: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert tensors to lists for JSON serialization
        # Handle BFloat16 tensors by converting to float32 first
        data['action_embedding'] = self.action_embedding.cpu().float().numpy().tolist()
        data['response_embedding'] = self.response_embedding.cpu().float().numpy().tolist()
        data['context_embedding'] = self.context_embedding.cpu().float().numpy().tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        # Convert lists back to tensors
        data['action_embedding'] = torch.tensor(data['action_embedding'])
        data['response_embedding'] = torch.tensor(data['response_embedding'])
        data['context_embedding'] = torch.tensor(data['context_embedding'])
        return cls(**data)


@dataclass
class SemanticConversation:
    """Complete conversation data structure."""
    conversation_id: str
    original_question: str
    category: str
    turns: List[ConversationTurn]
    original_prompt_embedding: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['turns'] = [turn.to_dict() for turn in self.turns]
        # Handle optional original_prompt_embedding
        if self.original_prompt_embedding is not None:
            data['original_prompt_embedding'] = self.original_prompt_embedding.cpu().float().numpy().tolist()
        else:
            data['original_prompt_embedding'] = None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticConversation':
        """Create from dictionary."""
        data['turns'] = [ConversationTurn.from_dict(turn_data) for turn_data in data['turns']]
        # Handle optional original_prompt_embedding
        if data.get('original_prompt_embedding') is not None:
            data['original_prompt_embedding'] = torch.tensor(data['original_prompt_embedding'])
        else:
            data['original_prompt_embedding'] = None
        return cls(**data)
    
    def save_to_json(self, file_path: str, indent: int = 2) -> None:
        """Save conversation to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, file_path: str) -> 'SemanticConversation':
        """Load conversation from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_embedding_arrays(self) -> Dict[str, torch.Tensor]:
        """Extract all embeddings as tensors for vector storage."""
        action_embeddings = torch.stack([turn.action_embedding for turn in self.turns])
        response_embeddings = torch.stack([turn.response_embedding for turn in self.turns])
        context_embeddings = torch.stack([turn.context_embedding for turn in self.turns])
        
        return {
            'action_embeddings': action_embeddings,
            'response_embeddings': response_embeddings,
            'context_embeddings': context_embeddings
        }
    
    def get_score_arrays(self) -> Dict[str, torch.Tensor]:
        """Extract all scores as tensors."""
        llamaguard_scores = torch.tensor([turn.llamaguard_score for turn in self.turns])
        v_high_scores = torch.tensor([turn.v_high_score for turn in self.turns])
        mean_q_low_scores = torch.tensor([turn.mean_q_low_score for turn in self.turns])
        
        # Running max arrays, if available
        max_llamaguard_to_t = torch.tensor([
            (turn.max_llamaguard_to_t if turn.max_llamaguard_to_t is not None else float('nan'))
            for turn in self.turns
        ])
        max_v_high_to_t = torch.tensor([
            (turn.max_v_high_to_t if turn.max_v_high_to_t is not None else float('nan'))
            for turn in self.turns
        ])
        max_q_low_to_t = torch.tensor([
            (turn.max_q_low_to_t if turn.max_q_low_to_t is not None else float('nan'))
            for turn in self.turns
        ])
        
        return {
            'llamaguard_scores': llamaguard_scores,
            'v_high_scores': v_high_scores,
            'mean_q_low_scores': mean_q_low_scores,
            'max_llamaguard_to_t': max_llamaguard_to_t,
            'max_v_high_to_t': max_v_high_to_t,
            'max_q_low_to_t': max_q_low_to_t,
        } 