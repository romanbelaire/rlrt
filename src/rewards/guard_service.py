"""
Wrapper around guard/target-model safety scoring so callers don't have to
interact with environment methods directly.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


class GuardRewardService:
    """Lightweight proxy for guard model scoring APIs with fail-fast validation."""

    def __init__(self, environment) -> None:
        if not hasattr(environment, "score_safety"):
            raise AttributeError("Environment must implement score_safety()")
        self.environment = environment

    def sequence_reward(
        self,
        prompts,
        *,
        questions=None,
        reference_state=None,
        include_white_box: bool = True,
    ):
        """Score safety of prompts using guard model.
        
        Args:
            prompts: List of text prompts to score
            questions: Optional list of questions (for context)
            reference_state: Optional list of reference states
            
        Returns:
            torch.Tensor of safety scores (higher = more harmful)
        """
        # Fail-fast validation
        if not isinstance(prompts, (list, tuple)):
            raise TypeError(f"prompts must be list/tuple, got {type(prompts)}")
        if len(prompts) == 0:
            raise ValueError("prompts cannot be empty")
        if questions is not None and len(questions) != len(prompts):
            raise ValueError(f"questions length {len(questions)} != prompts length {len(prompts)}")
        if reference_state is not None and len(reference_state) != len(prompts):
            raise ValueError(f"reference_state length {len(reference_state)} != prompts length {len(prompts)}")
        
        scores = self.environment.score_safety(
            prompts,
            questions=questions,
            reference_state=reference_state,
            include_white_box=include_white_box,
        )
        
        # Validate output
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score_safety() must return torch.Tensor, got {type(scores)}")
        if scores.dim() != 1:
            raise ValueError(f"scores must be 1D tensor, got shape {scores.shape}")
        if scores.size(0) != len(prompts):
            raise ValueError(f"scores length {scores.size(0)} != prompts length {len(prompts)}")
        if not torch.all(torch.isfinite(scores)):
            raise ValueError(f"score_safety() returned non-finite values: {scores}")
        
        return scores

    def sequence_reward_from_tokens(
        self,
        tokens: torch.Tensor,
        response_attention_mask: torch.Tensor,
        *,
        question_ids: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        reference_state_ids: Optional[torch.Tensor] = None,
        reference_attention_mask: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
        include_white_box: bool = True,
    ):
        """Score safety from token tensors.
        
        Args:
            tokens: Response token IDs [batch_size, seq_len]
            response_attention_mask: Attention mask for responses
            question_ids: Optional question token IDs
            question_attention_mask: Optional question attention mask
            reference_state_ids: Optional reference state token IDs
            reference_attention_mask: Optional reference state attention mask
            tokenizer: Tokenizer used for decoding/logging inside the environment
            include_white_box: Whether to include white-box guard scores
            
        Returns:
            torch.Tensor of safety scores
        """
        if not hasattr(self.environment, "score_safety_from_tokens"):
            raise AttributeError("Environment lacks score_safety_from_tokens()")
        
        # Fail-fast validation
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"tokens must be torch.Tensor, got {type(tokens)}")
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be 2D [batch, seq], got shape {tokens.shape}")
        if not isinstance(response_attention_mask, torch.Tensor):
            raise TypeError(f"response_attention_mask must be torch.Tensor, got {type(response_attention_mask)}")
        if response_attention_mask.shape != tokens.shape:
            raise ValueError(f"response_attention_mask shape {response_attention_mask.shape} != tokens shape {tokens.shape}")
        if question_ids is not None and question_ids.shape[0] != tokens.shape[0]:
            raise ValueError(f"question_ids batch size {question_ids.shape[0]} != tokens batch size {tokens.shape[0]}")
        if question_attention_mask is not None and question_ids is None:
            raise ValueError("question_attention_mask provided without question_ids")
        if question_attention_mask is not None and question_attention_mask.shape != question_ids.shape:
            raise ValueError(f"question_attention_mask shape {question_attention_mask.shape} != question_ids shape {question_ids.shape}")
        if reference_state_ids is not None and reference_state_ids.shape[0] != tokens.shape[0]:
            raise ValueError(f"reference_state_ids batch size {reference_state_ids.shape[0]} != tokens batch size {tokens.shape[0]}")
        if reference_attention_mask is not None and reference_state_ids is None:
            raise ValueError("reference_attention_mask provided without reference_state_ids")
        if reference_attention_mask is not None and reference_attention_mask.shape != reference_state_ids.shape:
            raise ValueError(f"reference_attention_mask shape {reference_attention_mask.shape} != reference_state_ids shape {reference_state_ids.shape}")
        
        scores = self.environment.score_safety_from_tokens(
            response_ids=tokens,
            response_attention_mask=response_attention_mask,
            question_ids=question_ids,
            question_attention_mask=question_attention_mask,
            reference_state_ids=reference_state_ids,
            reference_attention_mask=reference_attention_mask,
            tokenizer=tokenizer,
            include_white_box=include_white_box,
        )
        
        # Validate output
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score_safety_from_tokens() must return torch.Tensor, got {type(scores)}")
        if scores.size(0) != tokens.size(0):
            raise ValueError(f"scores batch size {scores.size(0)} != tokens batch size {tokens.size(0)}")
        
        return scores

    def target_response_reward(self, responses, *, questions=None, reference_state=None):
        """Score safety of target model responses.
        
        Args:
            responses: List of response texts
            questions: Optional list of questions
            reference_state: Optional list of reference states
            
        Returns:
            torch.Tensor of safety scores (higher = more harmful)
        """
        # Fail-fast validation
        if not isinstance(responses, (list, tuple)):
            raise TypeError(f"responses must be list/tuple, got {type(responses)}")
        if len(responses) == 0:
            raise ValueError("responses cannot be empty")
        if questions is not None and len(questions) != len(responses):
            raise ValueError(f"questions length {len(questions)} != responses length {len(responses)}")
        if reference_state is not None and len(reference_state) != len(responses):
            raise ValueError(f"reference_state length {len(reference_state)} != responses length {len(responses)}")
        
        scores = self.environment.score_safety(
            responses, questions=questions, reference_state=reference_state
        )
        
        # Validate output
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score_safety() must return torch.Tensor, got {type(scores)}")
        if scores.dim() != 1:
            raise ValueError(f"scores must be 1D tensor, got shape {scores.shape}")
        if scores.size(0) != len(responses):
            raise ValueError(f"scores length {scores.size(0)} != responses length {len(responses)}")
        if not torch.all(torch.isfinite(scores)):
            raise ValueError(f"score_safety() returned non-finite values: {scores}")
        
        return scores

