"""
Down-sampling utilities for embeddings.
"""

from typing import List
import torch
import numpy as np
from sklearn.decomposition import PCA
from src.semantic_analysis.data_structures import SemanticConversation


def _to_float32_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach and convert any torch tensor (including bf16) to float32 numpy array."""
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected a torch.Tensor for conversion to numpy")
    return t.detach().to(torch.float32).cpu().numpy()


def reduce_embedding_dimensions(conversations: List[SemanticConversation], target_dim: int, device: str = 'cuda') -> List[SemanticConversation]:
    """
    Reduces the dimensionality of all embeddings in a list of conversations using PCA.
    A separate PCA model is fit for each embedding type (action, response, context, original_prompt).
    """
    if not conversations:
        return []

    embedding_types = ['action_embedding', 'response_embedding', 'context_embedding']
    
    for emb_type in embedding_types:
        # 1. Collect all embeddings of a specific type
        all_embeddings = []
        for conv in conversations:
            for turn in conv.turns:
                emb = getattr(turn, emb_type)
                all_embeddings.append(_to_float32_numpy(emb))
        
        if not all_embeddings:
            continue
            
        # Stack as float32 for PCA
        all_embeddings_np = np.asarray(all_embeddings, dtype=np.float32)
        
        # 2. Fit PCA on all embeddings of this type
        pca = PCA(n_components=target_dim)
        pca.fit(all_embeddings_np)
        
        # 3. Transform embeddings in each conversation
        for conv in conversations:
            for turn in conv.turns:
                original_embedding = _to_float32_numpy(getattr(turn, emb_type)).reshape(1, -1)
                reduced_embedding = pca.transform(original_embedding)
                setattr(turn, emb_type, torch.from_numpy(reduced_embedding).to(device=device, dtype=torch.float32).squeeze(0))
                
    # Handle original_prompt_embedding separately since it's at conversation level, not turn level
    # Collect all original prompt embeddings
    original_prompt_embeddings = []
    for conv in conversations:
        if conv.original_prompt_embedding is not None:
            original_prompt_embeddings.append(_to_float32_numpy(conv.original_prompt_embedding))
    
    if original_prompt_embeddings:
        original_prompt_np = np.asarray(original_prompt_embeddings, dtype=np.float32)
        
        # Fit PCA on original prompt embeddings
        prompt_pca = PCA(n_components=target_dim)
        prompt_pca.fit(original_prompt_np)
        
        # Transform original prompt embeddings
        conv_idx = 0
        for conv in conversations:
            if conv.original_prompt_embedding is not None:
                original_embedding = _to_float32_numpy(conv.original_prompt_embedding).reshape(1, -1)
                reduced_embedding = prompt_pca.transform(original_embedding)
                conv.original_prompt_embedding = torch.from_numpy(reduced_embedding).to(device=device, dtype=torch.float32).squeeze(0)
                conv_idx += 1
                
    return conversations 