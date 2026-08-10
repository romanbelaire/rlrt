"""
Semantic drift calculation utilities.
"""

from typing import Tuple, Dict, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .data_structures import SemanticConversation
from .data_loader import encode_text
from .input_embeddings import compute_input_embeddings_for_conversation


def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> Optional[float]:
    """Computes cosine similarity between two normalized embeddings. Returns None if either is None."""
    if emb1 is None or emb2 is None:
        return None
    return F.cosine_similarity(emb1, emb2, dim=0).item()


def get_context_vectors(conversation: SemanticConversation, turn_idx: int, context_voice: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extracts X and Y context vectors based on the specified context voice.
    Voices per spec:
      1a: Agent only → X = mean(actions[0:t]), Y = action[t]
      1b: Target only → X = mean(responses[0:t]), Y = response[t]
      1c: Both, agent included in X → X mixes actions up to t (incl t) and responses up to t-1; Y = pair(context) at t
      1d: Both, agent included in Y → X = mean(previous pairs[0:t]); Y = response[t]
    """
    turns = conversation.turns
    current_turn = turns[turn_idx]

    X: Optional[torch.Tensor] = None
    Y: Optional[torch.Tensor] = None

    if context_voice == '1a':  # Agent messages only
        if turn_idx > 0:
            X = torch.mean(torch.stack([t.action_embedding for t in turns[:turn_idx]]), dim=0)
        Y = current_turn.action_embedding

    elif context_voice == '1b':  # Target responses only
        if turn_idx > 0:
            X = torch.mean(torch.stack([t.response_embedding for t in turns[:turn_idx]]), dim=0)
        Y = current_turn.response_embedding

    elif context_voice == '1c':  # Both models, agent in X; Y should be pair (context) at t
        if turn_idx > 0:
            X_action = torch.mean(torch.stack([t.action_embedding for t in turns[:turn_idx + 1]]), dim=0)  # include agent action at t
            X_response = torch.mean(torch.stack([t.response_embedding for t in turns[:turn_idx]]), dim=0)
            X = (X_action + X_response) / 2
        else:
            X = current_turn.action_embedding
        Y = current_turn.context_embedding  # pair at t

    elif context_voice == '1d':  # Both models, agent in Y; Y should be ONLY response at t
        if turn_idx > 0:
            X = torch.mean(torch.stack([t.context_embedding for t in turns[:turn_idx]]), dim=0)  # previous pairs
        Y = current_turn.response_embedding

    # Normalize embeddings
    if X is not None:
        X = F.normalize(X, p=2, dim=0)
    if Y is not None:
        Y = F.normalize(Y, p=2, dim=0)

    return X, Y


def _first_Y_for_voice(conversation: SemanticConversation, voice: str) -> Optional[torch.Tensor]:
    """Get the reference first-Y vector for a given voice (turn 0)."""
    if len(conversation.turns) == 0:
        return None
    _, first_Y = get_context_vectors(conversation, 0, voice)
    return first_Y


def _running_cumsum(values: List[Optional[float]]) -> List[Optional[float]]:
    """Compute running cumulative sum over a list possibly containing None. None values propagate as None at that step."""
    out: List[Optional[float]] = []
    total = 0.0
    has_started = False
    for v in values:
        if v is None:
            out.append(None)
        else:
            total = total + float(v)
            has_started = True
            out.append(total)
    return out


def compute_all_semantic_drift_metrics(
    conversation: SemanticConversation,
    embedding_model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
) -> Dict:
    """Computes all 20 semantic drift metrics for a single conversation using stored hidden-state embeddings."""

    metrics: Dict = {
        'conversation_id': conversation.conversation_id,
        'category': conversation.category,
        'original_question': conversation.original_question,
        'time': [t.turn_number for t in conversation.turns],
        'llamaguard_scores': [t.llamaguard_score for t in conversation.turns],
        'v_high_scores': [t.v_high_score for t in conversation.turns],
        'q_low_scores': [t.mean_q_low_score for t in conversation.turns],
        # Running max harm R_{0:t} if present on turns (may be filled by engine if missing)
        'max_llamaguard_to_t': [t.max_llamaguard_to_t for t in conversation.turns],
        'max_v_high_to_t': [t.max_v_high_to_t for t in conversation.turns],
        'max_q_low_to_t': [t.max_q_low_to_t for t in conversation.turns],
    }

    # Verify original prompt embedding exists (same embedding space as stored embeddings)
    if conversation.original_prompt_embedding is None:
        raise RuntimeError("original_prompt_embedding is missing on conversation. It must be computed before downsampling.")
    original_prompt_embedding = F.normalize(conversation.original_prompt_embedding, p=2, dim=0)

    # Prepare first-Y per voice for p2f comparisons
    first_Y_cache: Dict[str, Optional[torch.Tensor]] = {
        v: _first_Y_for_voice(conversation, v) for v in ['1a', '1b', '1c', '1d']
    }

    for voice in ['1a', '1b', '1c', '1d']:
        # Precompute previous Y for p2p and cumulative
        prev_Y: Optional[torch.Tensor] = None
        p2p_series: List[Optional[float]] = []

        for context_range in ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']:
            metric_key = f'similarity_{voice}_{context_range}'
            metrics[metric_key] = []

        for t_idx, _turn in enumerate(conversation.turns):
            X_t, Y_t = get_context_vectors(conversation, t_idx, voice)

            # p2c: compare Y_t to past context X_t
            sim_p2c = compute_cosine_similarity(X_t, Y_t)
            metrics[f'similarity_{voice}_p2c'].append(sim_p2c)

            # p2p: compare Y_t to previous Y_{t-1}
            sim_p2p = compute_cosine_similarity(prev_Y, Y_t) if t_idx > 0 else None
            metrics[f'similarity_{voice}_p2p'].append(sim_p2p)
            p2p_series.append(sim_p2p)
            prev_Y = Y_t

            # p2o: compare Y_t to original prompt embedding
            sim_p2o = compute_cosine_similarity(original_prompt_embedding, Y_t)
            metrics[f'similarity_{voice}_p2o'].append(sim_p2o)

            # p2f: compare Y_t to the first Y (voice-specific)
            first_Y = first_Y_cache[voice]
            sim_p2f = compute_cosine_similarity(first_Y, Y_t)
            metrics[f'similarity_{voice}_p2f'].append(sim_p2f)

            # p2c_cu placeholder will be filled after loop (needs running sum of p2p)
            metrics[f'similarity_{voice}_p2c_cu'].append(None)

        # Fill cumulative p2p trajectory for p2c_cu (same length as turns)
        cumulative = _running_cumsum(p2p_series)
        metrics[f'similarity_{voice}_p2c_cu'] = cumulative

    return metrics


def compute_all_semantic_drift_metrics_input(
    conversation: SemanticConversation,
    embedding_model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
) -> Dict:
    """Computes semantic drift metrics using INPUT embeddings (token embedding table) instead of hidden states.
    Context vectors for '1c' are defined as the mean of action and response input embeddings at a turn (pair),
    and X mixes actions up to t (incl t) with responses up to t-1. For '1d', X is mean over previous pairs and Y is ONLY response.
    """
    # Prepare per-conversation input embeddings
    inp = compute_input_embeddings_for_conversation(conversation, embedding_model, tokenizer, device)
    action_inputs: torch.Tensor = inp['action_input_embeddings']
    response_inputs: torch.Tensor = inp['response_input_embeddings']
    if action_inputs.numel() == 0:
        num_turns = 0
        context_inputs = torch.empty(0)
    else:
        num_turns = action_inputs.shape[0]
        context_inputs = (action_inputs + response_inputs) / 2.0  # pair at each t

    # Original prompt input embedding
    original_prompt_input = F.normalize(inp['original_prompt_input_embedding'], p=2, dim=0)

    metrics: Dict = {
        'conversation_id': conversation.conversation_id,
        'category': conversation.category,
        'original_question': conversation.original_question,
        'time': [t.turn_number for t in conversation.turns],
        'llamaguard_scores': [t.llamaguard_score for t in conversation.turns],
        'v_high_scores': [t.v_high_score for t in conversation.turns],
        'q_low_scores': [t.mean_q_low_score for t in conversation.turns],
    }

    def get_input_context_vectors(turn_idx: int, voice: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        X: Optional[torch.Tensor] = None
        Y: Optional[torch.Tensor] = None
        if voice == '1a':
            if turn_idx > 0:
                X = torch.mean(action_inputs[:turn_idx], dim=0)
            Y = action_inputs[turn_idx]
        elif voice == '1b':
            if turn_idx > 0:
                X = torch.mean(response_inputs[:turn_idx], dim=0)
            Y = response_inputs[turn_idx]
        elif voice == '1c':
            if turn_idx > 0:
                X_action = torch.mean(action_inputs[:turn_idx + 1], dim=0)
                X_response = torch.mean(response_inputs[:turn_idx], dim=0)
                X = (X_action + X_response) / 2.0
            else:
                X = action_inputs[turn_idx]
            Y = context_inputs[turn_idx]
        elif voice == '1d':
            if turn_idx > 0:
                X = torch.mean(context_inputs[:turn_idx], dim=0)
            Y = response_inputs[turn_idx]
        if X is not None:
            X = F.normalize(X, p=2, dim=0)
        if Y is not None:
            Y = F.normalize(Y, p=2, dim=0)
        return X, Y

    # Cache first Y per voice (if any turns exist)
    first_Y_cache_inp: Dict[str, Optional[torch.Tensor]] = {}
    for voice in ['1a', '1b', '1c', '1d']:
        if num_turns == 0:
            first_Y_cache_inp[voice] = None
        else:
            _, y0 = get_input_context_vectors(0, voice)
            first_Y_cache_inp[voice] = y0

    for voice in ['1a', '1b', '1c', '1d']:
        for context_range in ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']:
            metrics[f'similarity_{voice}_{context_range}'] = []
        prev_Y: Optional[torch.Tensor] = None
        p2p_series: List[Optional[float]] = []
        for t_idx in range(num_turns):
            X, Y = get_input_context_vectors(t_idx, voice)
            # p2c
            metrics[f'similarity_{voice}_p2c'].append(compute_cosine_similarity(X, Y))
            # p2p
            sim_p2p = compute_cosine_similarity(prev_Y, Y) if t_idx > 0 else None
            metrics[f'similarity_{voice}_p2p'].append(sim_p2p)
            p2p_series.append(sim_p2p)
            prev_Y = Y
            # p2o
            metrics[f'similarity_{voice}_p2o'].append(compute_cosine_similarity(original_prompt_input, Y))
            # p2f
            first_Y = first_Y_cache_inp[voice]
            metrics[f'similarity_{voice}_p2f'].append(compute_cosine_similarity(first_Y, Y))
            # placeholder for p2c_cu, filled later
            metrics[f'similarity_{voice}_p2c_cu'].append(None)
        # fill cumulative
        metrics[f'similarity_{voice}_p2c_cu'] = _running_cumsum(p2p_series)

    return metrics 