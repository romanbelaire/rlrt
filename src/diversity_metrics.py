"""
Diversity and complexity metrics for evaluation.

This module provides functions to compute:
- Mean cosine similarity distance (diversity) for lists of texts
- Complexity metrics for agent messages
"""

from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    """Singleton wrapper for a small embedding model."""
    
    _instance = None
    _model = None
    _tokenizer = None
    _model_name = None
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the embedding model (default: small sentence transformer)
            device: Device to run on (default: cpu for lightweight evaluation)
        """
        if EmbeddingModel._instance is not None:
            if EmbeddingModel._model_name != model_name:
                # Reinitialize if model name changed
                EmbeddingModel._model = None
                EmbeddingModel._tokenizer = None
            else:
                return
        
        EmbeddingModel._instance = self
        EmbeddingModel._model_name = model_name
        
        try:
            # Try sentence-transformers first (more efficient)
            from sentence_transformers import SentenceTransformer
            print(f"[DIVERSITY] Loading sentence transformer: {model_name}")
            EmbeddingModel._model = SentenceTransformer(model_name, device=device)
            EmbeddingModel._tokenizer = None  # SentenceTransformer handles tokenization
        except ImportError:
            # Fallback to transformers
            print(f"[DIVERSITY] Loading transformers model: {model_name}")
            EmbeddingModel._tokenizer = AutoTokenizer.from_pretrained(model_name)
            EmbeddingModel._model = AutoModel.from_pretrained(model_name)
            EmbeddingModel._model.to(device)
            EmbeddingModel._model.eval()
        
        self.device = device
    
    @classmethod
    def get_instance(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """Get or create singleton instance."""
        if cls._instance is None or cls._model_name != model_name:
            cls(model_name=model_name, device=device)
        return cls._instance
    
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Tensor of shape [num_texts, embedding_dim]
        """
        if len(texts) == 0:
            return torch.empty(0)
        
        # Use sentence-transformers if available
        if EmbeddingModel._tokenizer is None:
            # SentenceTransformer
            embeddings = EmbeddingModel._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            )
            return embeddings
        
        # Use transformers AutoModel
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = EmbeddingModel._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = EmbeddingModel._model(**inputs)
                # Mean pooling with attention mask
                attention_mask = inputs['attention_mask']
                hidden_states = outputs.last_hidden_state
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                summed = masked_embeddings.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = summed / lengths
                
                # Normalize
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings_list.append(pooled)
        
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings


def compute_mean_cosine_distance(texts: List[str], 
                                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                  device: str = "cpu",
                                  batch_size: int = 32) -> float:
    """
    Compute mean pairwise cosine distance (1 - cosine similarity) for a list of texts.
    
    This measures diversity: higher values indicate more diverse texts.
    
    Args:
        texts: List of text strings to compare
        model_name: Name of the embedding model to use
        device: Device to run on (default: cpu)
        batch_size: Batch size for encoding
        
    Returns:
        Mean cosine distance (0-2 range, where 0 = identical, 2 = opposite)
    """
    if len(texts) < 2:
        return 0.0
    
    # Get embeddings
    embedding_model = EmbeddingModel.get_instance(model_name=model_name, device=device)
    embeddings = embedding_model.encode(texts, batch_size=batch_size)
    
    # Compute pairwise cosine distances
    # Cosine distance = 1 - cosine similarity
    # For normalized embeddings: cosine_similarity = dot product
    # We compute all pairwise distances efficiently
    num_texts = len(texts)
    distances = []
    
    for i in range(num_texts):
        for j in range(i + 1, num_texts):
            # Cosine similarity for normalized vectors is just dot product
            cos_sim = torch.dot(embeddings[i], embeddings[j])
            # Cosine distance = 1 - cosine similarity
            cos_dist = 1.0 - cos_sim.item()
            distances.append(cos_dist)
    
    mean_distance = float(np.mean(distances)) if distances else 0.0
    return mean_distance


def compute_complexity(texts: List[str]) -> Dict[str, float]:
    """
    Compute complexity metrics for a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with complexity metrics:
        - mean_length: Average character length
        - mean_word_count: Average word count
        - mean_sentence_count: Average sentence count (approximate)
        - vocab_diversity: Unique words / total words (type-token ratio)
        - mean_avg_word_length: Average word length
    """
    if len(texts) == 0:
        return {
            "mean_length": 0.0,
            "mean_word_count": 0.0,
            "mean_sentence_count": 0.0,
            "vocab_diversity": 0.0,
            "mean_avg_word_length": 0.0
        }
    
    lengths = []
    word_counts = []
    sentence_counts = []
    all_words = []
    
    for text in texts:
        # Character length
        lengths.append(len(text))
        
        # Word count
        words = text.split()
        word_counts.append(len(words))
        all_words.extend(words)
        
        # Approximate sentence count (count periods, exclamation, question marks)
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        sentence_counts.append(sentence_count)
    
    # Compute vocabulary diversity (type-token ratio)
    unique_words = len(set(word.lower() for word in all_words))
    total_words = len(all_words)
    vocab_diversity = unique_words / total_words if total_words > 0 else 0.0
    
    # Average word length
    avg_word_lengths = [len(word) for word in all_words]
    mean_avg_word_length = float(np.mean(avg_word_lengths)) if avg_word_lengths else 0.0
    
    return {
        "mean_length": float(np.mean(lengths)),
        "mean_word_count": float(np.mean(word_counts)),
        "mean_sentence_count": float(np.mean(sentence_counts)),
        "vocab_diversity": vocab_diversity,
        "mean_avg_word_length": mean_avg_word_length
    }


def compute_diversity_and_complexity_per_category(
    conversation_logs: List[Dict[str, Any]],
    category_column: str = "topic"
) -> Dict[str, Dict[str, Any]]:
    """
    Compute diversity and complexity metrics per harm category.
    
    Args:
        conversation_logs: List of conversation log dictionaries, each containing:
            - 'conversation': List of turn dictionaries with 'agent_message' and 'target_response'
            - category_column: Key in log dict that contains the category name
        category_column: Key to extract category from (default: 'topic')
        
    Returns:
        Dictionary mapping category names to metrics:
        {
            "category_name": {
                "agent_diversity": float,
                "user_diversity": float,
                "agent_complexity": Dict[str, float],
                "num_conversations": int
            }
        }
    """
    # Group conversations by category
    category_conversations = {}
    
    for log in conversation_logs:
        category = log.get(category_column, "unknown")
        if category not in category_conversations:
            category_conversations[category] = []
        category_conversations[category].append(log)
    
    results = {}
    
    for category, logs in category_conversations.items():
        # Extract agent messages and user messages
        agent_messages = []
        user_messages = []
        
        for log in logs:
            conversation = log.get('conversation', [])
            for turn in conversation:
                agent_msg = turn.get('agent_message')
                user_msg = turn.get('target_response')  # User/target response
                
                # Filter out None and empty strings
                if agent_msg and isinstance(agent_msg, str) and len(agent_msg.strip()) > 0:
                    agent_messages.append(agent_msg.strip())
                if user_msg and isinstance(user_msg, str) and len(user_msg.strip()) > 0:
                    user_messages.append(user_msg.strip())
        
        # Compute diversity
        agent_diversity = compute_mean_cosine_distance(agent_messages) if len(agent_messages) >= 2 else 0.0
        user_diversity = compute_mean_cosine_distance(user_messages) if len(user_messages) >= 2 else 0.0
        
        # Compute complexity for agent messages
        agent_complexity = compute_complexity(agent_messages)
        
        results[category] = {
            "agent_diversity": agent_diversity,
            "user_diversity": user_diversity,
            "agent_complexity": agent_complexity,
            "num_conversations": len(logs),
            "num_agent_messages": len(agent_messages),
            "num_user_messages": len(user_messages)
        }
    
    return results

