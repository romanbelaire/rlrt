"""
Hierarchical critic implementations for HRL.

This module provides MLP-based critic models that map embeddings to numerical values
for both high-level (sequence-level) and low-level (token-level) value functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MLPCritic(nn.Module):
    """
    Simple MLP critic that maps embeddings to scalar values.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.embedding_dim = embedding_dim
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Scalar values [batch_size, 1]
        """
        return self.mlp(embeddings).squeeze(-1)


class HighLevelCritic(nn.Module):
    """
    High-level critic that implements Q_high(s, a).
    Maps full sequences to scalar values.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = encoder
        self.mlp_critic = MLPCritic(
            embedding_dim=embedding_dim * 3,  # state + action + base_state
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        base_state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for high-level critic.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            base_state: Base state tokens [batch_size, seq_len]
            action: Action tokens [batch_size, action_len]
            
        Returns:
            Q-values [batch_size]
        """
        device = next(self.encoder.parameters()).device
        state_embed = self.encoder(state.to(device)).mean(dim=1)
        base_state_embed = self.encoder(base_state.to(device)).mean(dim=1)
        action_embed = self.encoder(action.to(device)).mean(dim=1)
        
        # Concatenate embeddings
        combined_embed = torch.cat([state_embed, base_state_embed, action_embed], dim=-1)
        
        # Pass through MLP
        return self.mlp_critic(combined_embed)
    
    def save(self, path: str):
        """Save the critic model."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load the critic model."""
        self.load_state_dict(torch.load(path))


class LowLevelCritic(nn.Module):
    """
    Low-level critic that implements V_low.
    Maps token-level information to scalar values.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = encoder
        self.mlp_critic = MLPCritic(
            embedding_dim=embedding_dim * 3,  # state + action + base_state
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        base_state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for low-level critic.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            base_state: Base state tokens [batch_size, seq_len]
            action: Action tokens [batch_size, action_len]
            
        Returns:
            V_low values [batch_size]
        """
        device = next(self.encoder.parameters()).device
        state_embed = self.encoder(state.to(device)).sum(dim=1)
        base_state_embed = self.encoder(base_state.to(device)).sum(dim=1)
        action_embed = self.encoder(action.to(device)).sum(dim=1)
        
        # Concatenate embeddings
        combined_embed = torch.cat([state_embed, base_state_embed, action_embed], dim=-1)
        
        # Pass through MLP
        return self.mlp_critic(combined_embed)
    
    def save(self, path: str):
        """Save the critic model."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load the critic model."""
        self.load_state_dict(torch.load(path))


class HierarchicalCriticEnsemble(nn.Module):
    """
    Ensemble of high-level and low-level critics for hierarchical RL.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.high_level_critic = HighLevelCritic(
            encoder=encoder,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.low_level_critic = LowLevelCritic(
            encoder=encoder,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
    def forward_high_level(
        self, 
        state: torch.Tensor, 
        base_state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for high-level critic."""
        return self.high_level_critic(state, base_state, action)
    
    def forward_low_level(
        self, 
        state: torch.Tensor, 
        base_state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for low-level critic."""
        return self.low_level_critic(state, base_state, action)
    
    def save(self, path: str):
        """Save both critics."""
        torch.save({
            'high_level': self.high_level_critic.state_dict(),
            'low_level': self.low_level_critic.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load both critics."""
        checkpoint = torch.load(path)
        self.high_level_critic.load_state_dict(checkpoint['high_level'])
        self.low_level_critic.load_state_dict(checkpoint['low_level']) 