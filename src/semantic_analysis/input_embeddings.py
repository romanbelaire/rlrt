from typing import Dict
import torch
from transformers import AutoModel, AutoTokenizer
from src.semantic_analysis.data_structures import SemanticConversation


def encode_text_input_embedding(text: str, model: AutoModel, tokenizer: AutoTokenizer, device: str = 'cuda') -> torch.Tensor:
    """
    Compute a mean-pooled input-embedding vector for a single text.
    Uses model.get_input_embeddings()(input_ids) rather than hidden states.
    """
    model = model.to(device)
    inputs = tokenizer(text, return_tensors='pt', padding=True, padding_side='left', truncation=True, max_length=512)
    input_ids: torch.Tensor = inputs['input_ids'].long().to(device)
    with torch.no_grad():
        token_embeds: torch.Tensor = model.get_input_embeddings()(input_ids)  # [1, seq_len, hidden]
        pooled: torch.Tensor = token_embeds.mean(dim=1).squeeze(0)
    return pooled.float()


def compute_input_embeddings_for_conversation(
    conversation: SemanticConversation,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """
    Compute input-embedding pooled vectors for a conversation:
      - original_prompt_input_embedding: Tensor[hidden]
      - action_input_embeddings: Tensor[num_turns, hidden]
      - response_input_embeddings: Tensor[num_turns, hidden]
    Note: Context text is not stored in the conversation, so context input embeddings are not produced here.
    """
    action_vecs = []
    response_vecs = []

    for turn in conversation.turns:
        action_vecs.append(encode_text_input_embedding(turn.low_level_action, model, tokenizer, device))
        response_vecs.append(encode_text_input_embedding(turn.target_response, model, tokenizer, device))

    original_prompt_vec = encode_text_input_embedding(conversation.original_question, model, tokenizer, device)

    return {
        'original_prompt_input_embedding': original_prompt_vec,
        'action_input_embeddings': torch.stack(action_vecs) if len(action_vecs) > 0 else torch.empty(0),
        'response_input_embeddings': torch.stack(response_vecs) if len(response_vecs) > 0 else torch.empty(0),
    } 