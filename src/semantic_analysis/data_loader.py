"""
Data loading utilities for semantic analysis evaluation.
"""

import os
import json
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from src.semantic_analysis.data_structures import SemanticConversation


def load_conversations(conversations_path: str, max_conversations: int = None) -> List[SemanticConversation]:
    """Loads conversation JSON files.
    Accepts either:
      - a dataset root containing a `conversations/` subdirectory, or
      - a directory that itself contains the conversation JSON files.
    Fails loudly if neither condition holds.
    """
    # Resolve actual directory containing JSON files
    dir_to_scan = None
    if os.path.isdir(conversations_path):
        # Prefer a `conversations` subdir if present
        candidate = os.path.join(conversations_path, 'conversations')
        if os.path.isdir(candidate):
            dir_to_scan = candidate
        else:
            dir_to_scan = conversations_path
    else:
        raise FileNotFoundError(f"Provided path is not a directory: {conversations_path}")

    filepaths = sorted([os.path.join(dir_to_scan, f) for f in os.listdir(dir_to_scan) if f.endswith('.json')])
    if not filepaths:
        raise RuntimeError(f"No conversation JSON files found in {dir_to_scan}")

    if max_conversations:
        filepaths = filepaths[:max_conversations]

    conversations: List[SemanticConversation] = []
    for filepath in filepaths:
        conversations.append(SemanticConversation.load_from_json(filepath))
    return conversations


def get_embedding_model(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = 'cuda') -> Tuple[AutoModel, AutoTokenizer]:
    """Initializes and returns the embedding model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.padding_side = 'left'
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
            model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()
    return model, tokenizer


def encode_text(text: str, model: AutoModel, tokenizer: AutoTokenizer, device: str = 'cuda') -> torch.Tensor:
    """Encodes a single string of text to an embedding."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, padding_side='left', truncation=True, max_length=512)
    input_ids = inputs['input_ids'].long().to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=inputs['attention_mask'].to(device))
        # Use mean pooling of the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding 