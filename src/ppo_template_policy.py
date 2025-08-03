"""
PPO-based template selection policy for high-level instruction generation.

This module provides a PPO-based high-level policy that selects templates
from a predefined set, similar to the DQN TemplateHighLevelPolicy but using
PPO-style policy gradient methods instead of Q-learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import List, Dict, Any, Optional, Tuple


class PPOTemplateHighLevelPolicy(nn.Module):
    """
    PPO-based high-level policy that selects templates from a predefined set.
    
    This policy uses PPO-style policy gradient methods to learn which templates
    work best for different conversation contexts. It's similar to the DQN
    TemplateHighLevelPolicy but uses PPO instead of Q-learning.
    
    The policy maintains a small classification head that outputs logits for
    each available template, and uses softmax to convert these to probabilities
    for sampling during training and argmax for evaluation.
    """
    
    policy_kind: str = "ppo"
    
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: Any,
        templates_path: str = "src/configs/templates.json",
        hidden_size: int = 768,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the PPO template selector.
        
        Args:
            encoder: Text encoder (e.g., transformer encoder)
            tokenizer: Tokenizer for text processing
            templates_path: Path to templates.json file
            hidden_size: Hidden size for the classification head
            device: Device to use for computation
        """
        super().__init__()
        
        self.encoder = encoder
        self.tokenizer = tokenizer
        
        # Load templates from file
        self.strategies = self._load_strategies(templates_path)
        self.num_actions = len(self.strategies)
        
        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Classification head for template selection
        self.template_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_actions)
        )
        
        # Ensure classifier uses the same dtype as encoder
        encoder_dtype = next(self.encoder.parameters()).dtype
        self.template_classifier = self.template_classifier.to(dtype=encoder_dtype)
        
        print(f"Initialized PPO template selector with {self.num_actions} templates")
    
    def _load_strategies(self, templates_path: str) -> List[Dict[str, Any]]:
        """
        Load templates from JSON file.
        
        Args:
            templates_path: Path to templates.json file
            
        Returns:
            List of template dictionaries
        """
        try:
            with open(templates_path, 'r') as f:
                data = json.load(f)
            
            templates = data.get("templates", [])
            
            # Validate templates
            for template in templates:
                if 'id' not in template:
                    raise KeyError(f"Template missing 'id' field: {template}")
                if 'instruction' not in template:
                    raise KeyError(f"Template {template.get('id', 'unknown')} missing 'instruction' field")
            
            return templates
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Templates file not found: {templates_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in templates file: {e}")
    
    def _encode_state(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input state to get pooled representation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract hidden states from the last layer
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]
        
        # Handle case where encoder returns flattened tensor
        if len(hidden_states.shape) == 2:
            # Flattened tensor: [batch_size, seq_len * hidden_size]
            batch_size, flattened_size = hidden_states.shape
            # Infer hidden_size from the flattened size and sequence length
            seq_len = input_ids.shape[1]
            hidden_size = flattened_size // seq_len
            
            if flattened_size % seq_len != 0:
                raise ValueError(f"Cannot infer hidden_size from flattened tensor of size {flattened_size} and seq_len {seq_len}")
            
            # Reshape to [batch_size, seq_len, hidden_size]
            hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)
        
        # Use last token as pooled representation
        return hidden_states[:, -1, :]
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to get template logits.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Template logits [batch_size, num_templates]
        """
        pooled = self._encode_state(input_ids, attention_mask)
        return self.template_classifier(pooled)
    
    def select_action(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        force_greedy: bool = False
    ) -> int:
        """
        Select a template index using the current policy.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            force_greedy: Whether to use greedy selection (for evaluation)
            
        Returns:
            Selected template index
        """
        logits = self.forward(input_ids.to(self.device), attention_mask.to(self.device) if attention_mask is not None else None)
        
        if force_greedy:
            # Use argmax for evaluation
            return int(torch.argmax(logits, dim=-1).item())
        else:
            # Use softmax sampling for training
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).squeeze(-1).item())
    
    def get_log_probs(self, state: torch.Tensor, template_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and entropy for given template choices.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            template_indices: Chosen template indices [batch_size]
            
        Returns:
            - Log probabilities [batch_size]
            - Entropy [batch_size]
        """
        # Ensure tensors are on the correct device
        encoder_device = next(self.encoder.parameters()).device
        state = state.to(encoder_device)
        
        # Get logits for all templates
        logits = self.forward(state)
        
        # Clamp logits to prevent overflow in softmax
        logits_clamped = torch.clamp(logits, min=-5, max=5)
        
        # Compute log probabilities and probabilities
        log_probs = F.log_softmax(logits_clamped, dim=-1)
        probs = F.softmax(logits_clamped, dim=-1)
        
        # Ensure template indices are on the same device
        template_indices = template_indices.long().to(logits.device)
        
        # Get log probs for chosen templates
        selected_log_probs = log_probs.gather(1, template_indices.unsqueeze(-1)).squeeze(-1)
        
        # Calculate entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Validate outputs
        if not torch.isfinite(selected_log_probs).all():
            raise RuntimeError("Non-finite values detected in selected_log_probs")
        if not torch.isfinite(entropy).all():
            raise RuntimeError("Non-finite values detected in entropy")
        
        return selected_log_probs, entropy
    
    def act(self, state_texts: List[str], base_questions: Optional[List[str]] = None) -> Tuple[List[str], torch.Tensor, List[int]]:
        """
        Generate instructions by selecting templates.
        
        Args:
            state_texts: List of state texts
            base_questions: List of base questions (optional)
            
        Returns:
            - List of instruction strings
            - Log probabilities tensor
            - Template indices list
        """
        batch_size = len(state_texts)
        instructions = []
        log_probs_list = []
        template_indices = []
        
        for i, state_text in enumerate(state_texts):
            # Tokenize state
            enc = self.tokenizer(state_text, return_tensors="pt", truncation=True)
            input_ids = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device) if enc.attention_mask is not None else None
            
            # Select template
            template_idx = self.select_action(input_ids, attention_mask)
            template_indices.append(template_idx)
            
            # Get template instruction
            template = self.strategies[template_idx]
            instruction = template["instruction"]
            
            # Add context if needed
            if base_questions and i < len(base_questions):
                question = base_questions[i]
                if "{question}" in instruction:
                    instruction = instruction.replace("{question}", question)
            
            instructions.append(instruction)
            
            # Get log probability for this selection
            with torch.no_grad():
                logits = self.forward(input_ids, attention_mask)
                logits_clamped = torch.clamp(logits, min=-5, max=5)
                log_probs = F.log_softmax(logits_clamped, dim=-1)
                selected_log_prob = log_probs[0, template_idx]
                log_probs_list.append(selected_log_prob)
        
        # Stack log probabilities
        log_probs_tensor = torch.stack(log_probs_list) if log_probs_list else torch.zeros(batch_size, device=self.device)
        
        return instructions, log_probs_tensor, template_indices
    
    def save(self, path: str) -> None:
        """
        Save the PPO template policy to a directory.
        
        Args:
            path: Directory path to save the policy
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        torch.save({
            "template_classifier_state_dict": self.template_classifier.state_dict(),
            "strategies": self.strategies,
            "num_actions": self.num_actions,
        }, os.path.join(path, "model.pt"))
        
        # Save metadata
        meta = {
            "policy_type": "ppo",
            "class": "PPOTemplateHighLevelPolicy",
            "policy_kind": self.policy_kind,
            "num_actions": self.num_actions,
        }
        
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"Saved PPO template policy to {path}")
    
    def load(self, path: str, device: Optional[str] = None) -> None:
        """
        Load the PPO template policy from a directory.
        
        Args:
            path: Directory path to load the policy from
            device: Device to load the model on
        """
        if device is None:
            device = self.device
            
        checkpoint_path = os.path.join(path, "model.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load state dict
        self.template_classifier.load_state_dict(checkpoint["template_classifier_state_dict"])
        
        # Load other attributes
        self.strategies = checkpoint["strategies"]
        self.num_actions = checkpoint["num_actions"]
        
        print(f"Loaded PPO template policy from {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None, encoder: Optional[Any] = None, tokenizer: Optional[Any] = None) -> "PPOTemplateHighLevelPolicy":
        """
        Load a PPOTemplateHighLevelPolicy from a directory.
        
        Args:
            path: Directory path to load the policy from
            device: Device to load the model on
            encoder: Encoder model (required)
            tokenizer: Tokenizer (required)
            
        Returns:
            Loaded PPOTemplateHighLevelPolicy instance
        """
        if encoder is None:
            raise ValueError("Encoder is required to load PPOTemplateHighLevelPolicy")
        if tokenizer is None:
            raise ValueError("Tokenizer is required to load PPOTemplateHighLevelPolicy")
        
        # Load metadata
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Get embedding dimension from encoder
        if hasattr(encoder, 'config') and hasattr(encoder.config, 'hidden_size'):
            hidden_size = encoder.config.hidden_size
        else:
            hidden_size = 768  # Default embedding dimension
        
        # Create policy instance
        policy = cls(
            encoder=encoder,
            tokenizer=tokenizer,
            templates_path="src/configs/templates.json",
            hidden_size=hidden_size,
            device=device
        )
        
        # Load the actual weights
        policy.load(path, device)
        
        return policy 