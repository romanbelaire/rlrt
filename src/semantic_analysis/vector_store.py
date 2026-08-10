"""
FAISS-based vector storage for semantic analysis data.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import faiss


class SemanticVectorStore:
    """FAISS-based vector storage for semantic analysis data."""
    
    def __init__(self, embedding_dim: int, index_type: str = "IVFFlat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = {}
        self.conversation_ids = []
        self.vector_types = ['action_embeddings', 'response_embeddings', 'context_embeddings']
        
        # Initialize FAISS indices for each vector type
        self.indices = {}
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize FAISS indices for each vector type."""
        self.indices = {}
        
        for vector_type in self.vector_types:
            if self.index_type == "IVFFlat":
                # For IVF indices, we need enough training data
                # For semantic analysis with small datasets, fall back to flat index
                # Only use IVF for very large datasets (>1000 vectors)
                self.indices[vector_type] = faiss.IndexFlatIP(self.embedding_dim)
                print(f"Using IndexFlatIP for {vector_type} (small dataset)")
            else:
                # Default to flat index for small datasets
                self.indices[vector_type] = faiss.IndexFlatIP(self.embedding_dim)
    
    def add_conversation_vectors(self, 
                               conversation_id: str,
                               action_embeddings: torch.Tensor,
                               response_embeddings: torch.Tensor, 
                               context_embeddings: torch.Tensor) -> None:
        """Add conversation vectors to FAISS indices."""
        
        # Convert tensors to float32 first, then to numpy arrays
        # This handles BFloat16 tensors properly
        action_emb = action_embeddings.cpu().float().numpy()
        response_emb = response_embeddings.cpu().float().numpy()
        context_emb = context_embeddings.cpu().float().numpy()
        
        # Normalize embeddings for cosine similarity
        action_emb = self._normalize_embeddings(action_emb)
        response_emb = self._normalize_embeddings(response_emb)
        context_emb = self._normalize_embeddings(context_emb)
        
        # Add to indices
        embeddings_dict = {
            'action_embeddings': action_emb,
            'response_embeddings': response_emb,
            'context_embeddings': context_emb
        }
        
        for vector_type, embeddings in embeddings_dict.items():
            if self.indices[vector_type].ntotal == 0:
                # First addition - train the index if needed
                if hasattr(self.indices[vector_type], 'train'):
                    self.indices[vector_type].train(embeddings)
            
            # Add vectors to index
            self.indices[vector_type].add(embeddings)
        
        # Store metadata
        start_idx = len(self.conversation_ids)
        num_turns = action_embeddings.shape[0]
        
        self.metadata[conversation_id] = {
            'start_idx': start_idx,
            'num_turns': num_turns,
            'end_idx': start_idx + num_turns
        }
        
        self.conversation_ids.append(conversation_id)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    def search_similar(self, 
                      query_embedding: torch.Tensor, 
                      vector_type: str = 'action_embeddings',
                      k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        if vector_type not in self.vector_types:
            raise ValueError(f"Vector type {vector_type} not found. Available: {self.vector_types}")
        
        query_np = query_embedding.detach().cpu().float().numpy()
        query_np = self._normalize_embeddings(query_np)
        
        # Search in the appropriate index
        scores, indices = self.indices[vector_type].search(query_np, k)
        return scores, indices
    
    def get_conversation_vectors(self, 
                               conversation_id: str,
                               vector_type: str = 'action_embeddings') -> Optional[np.ndarray]:
        """Retrieve vectors for a specific conversation."""
        if conversation_id not in self.metadata:
            return None
        
        metadata = self.metadata[conversation_id]
        start_idx = metadata['start_idx']
        end_idx = metadata['end_idx']
        
        # Get vectors from index
        vectors = self.indices[vector_type].reconstruct_n(start_idx, end_idx - start_idx)
        return vectors
    
    def save(self, base_path: str) -> None:
        """Save FAISS indices and metadata."""
        os.makedirs(base_path, exist_ok=True)
        
        # Save each index
        for vector_type, index in self.indices.items():
            index_path = os.path.join(base_path, f"{vector_type}.faiss")
            faiss.write_index(index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(base_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'conversation_ids': self.conversation_ids,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'vector_types': self.vector_types
            }, f, indent=2)
    
    def load(self, base_path: str) -> None:
        """Load FAISS indices and metadata."""
        # Load metadata
        metadata_path = os.path.join(base_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.conversation_ids = data['conversation_ids']
        self.embedding_dim = data['embedding_dim']
        self.index_type = data['index_type']
        self.vector_types = data['vector_types']
        
        # Load indices
        self.indices = {}
        for vector_type in self.vector_types:
            index_path = os.path.join(base_path, f"{vector_type}.faiss")
            self.indices[vector_type] = faiss.read_index(index_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the stored vectors."""
        stats = {
            'total_conversations': len(self.conversation_ids),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type
        }
        
        for vector_type, index in self.indices.items():
            stats[f'{vector_type}_total_vectors'] = index.ntotal
        
        return stats 