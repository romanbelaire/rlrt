import copy
import math
import random
import json
import os
from typing import Any, List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.replay_buffer import ReplayBuffer


class TemplateHighLevelPolicy(nn.Module):
    """High-level policy that picks a template via DQN-style value estimation.

    This is a DQN-based high-level policy that uses Q-learning to select templates.
    It maintains a replay buffer and target network for stable training.
    
    Note: This class is specifically for DQN-style training. For PPO-style training,
    use HighLevelTemplatePolicy from hierarchical_policies.py with policy_kind="ppo".

    The encoder (e.g. LLM) is provided externally. Only the small Q-head is trained by
    default to keep computation light. This class purposefully keeps the interface
    minimal so that later we can swap it for a PPO-based LLM with the same public
    methods where possible.
    """

    policy_kind: str = "dqn"  # used by AgentTrainer to choose the correct update rule

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: Any,
        templates_path: str = "src/configs/templates.json",
        hidden_size: int = 768,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.0,
        epsilon_decay: int = 350,
        replay_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder  # may be frozen or fine-tuned by the caller
        self.tokenizer = tokenizer
        
        # Get the actual hidden_size from the encoder
        if hasattr(encoder, 'config') and hasattr(encoder.config, 'hidden_size'):
            actual_hidden_size = encoder.config.hidden_size
        elif hasattr(encoder, 'hidden_size'):
            actual_hidden_size = encoder.hidden_size
        else:
            # Fallback to the provided hidden_size
            actual_hidden_size = hidden_size
            print(f"[HIGH-LEVEL-POLICY] Warning: Could not determine encoder hidden_size, using provided value: {hidden_size}")
        
        self.hidden_size = actual_hidden_size  # Store actual hidden_size for saving/loading
        print(f"[HIGH-LEVEL-POLICY] Using encoder hidden_size: {actual_hidden_size}")
        
        # Load strategies from templates file
        self.strategies = self._load_strategies(templates_path)
        self.num_actions = len(self.strategies)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Simple two-layer head
        self.q_head = nn.Sequential(
            nn.Linear(actual_hidden_size, actual_hidden_size),
            nn.ReLU(),
            nn.Linear(actual_hidden_size, self.num_actions),
        )

        # Prefer the encoder's embedding device to avoid mismatches when using Accelerate/device_map
        if hasattr(encoder, "get_input_embeddings") and encoder.get_input_embeddings() is not None:
            self.encoder_device = encoder.get_input_embeddings().weight.device
        else:
            self.encoder_device = next(encoder.parameters()).device
        # Respect explicit device override only if encoder is not sharded by Accelerate
        if hasattr(encoder, 'hf_device_map') and getattr(encoder, 'hf_device_map'):
            self.device = self.encoder_device
        else:
            self.device = torch.device(device) if device is not None else self.encoder_device
        self.to(self.device)
        
        # Ensure Q-head is in bf16 for consistency
        self.q_head = self.q_head.to(dtype=torch.bfloat16)

        # Optimiser on the head only for now.
        self.optimizer = torch.optim.Adam(self.q_head.parameters(), lr=lr)

        # Epsilon-greedy parameters
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._steps_done = 0

        # Replay buffer and target network
        self.replay_buffer = ReplayBuffer(replay_size)
        self._target_q_head = copy.deepcopy(self.q_head).eval().to(self.device)
        # Ensure target Q-head is also in bf16
        self._target_q_head = self._target_q_head.to(dtype=torch.bfloat16)

    def _load_strategies(self, templates_path: str) -> List[Dict[str, Any]]:
        """Load strategies from templates.json file."""
        if not os.path.exists(templates_path):
            raise FileNotFoundError(f"Template file not found: {templates_path}")

        with open(templates_path, "r") as f:
            template_data = json.load(f)

        # Allow either {"templates": [...]} or a bare list
        if isinstance(template_data, dict) and "templates" in template_data:
            templates = template_data["templates"]
        elif isinstance(template_data, list):
            templates = template_data
        else:
            raise ValueError("templates.json must contain a list or a dict with key 'templates'")

        if len(templates) == 0:
            raise ValueError("No templates found in templates.json")

        # Convert templates to strategies format
        strategies = []
        for template in templates:
            strategy = {
                "id": template["id"],
                "instruction": template.get("instruction", template.get("description", "")),
                "type": template.get("type", "standard"),
                "template": template["template"]
            }
            strategies.append(strategy)

        return strategies

    # ------------------------------------------------------------------ helpers
    def _current_epsilon(self) -> float:
        frac = math.exp(-1.0 * self._steps_done / self._epsilon_decay)
        return self._epsilon_end + (self._epsilon_start - self._epsilon_end) * frac

    def _encode_state(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return a pooled representation (last token) from the encoder."""
        # Ensure tensors are on the encoder's embedding device and correct dtype
        target_device = self.encoder.get_input_embeddings().weight.device if hasattr(self.encoder, "get_input_embeddings") else next(self.encoder.parameters()).device
        if input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)
        with torch.no_grad():
            # Force the model to return hidden states instead of logits
            outputs = self.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True,
                output_logits=False,  # Don't compute logits to save memory
                return_dict=True
            )
            
            # Extract hidden states from the last layer
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # Get the last layer's hidden states
                hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            elif hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                # Fallback: try to get from tuple format
                hidden_states = outputs[0]  # type: ignore[index]
                
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
        
        # Use last token as pooled representation; alternative could be CLS
        return hidden_states[:, -1, :]  # [batch_size, hidden_size]

    # ------------------------------------------------------------------ API
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Q-values for every template in the registry."""
        # Fail-fast: ensure inputs match the encoder embedding device
        model_device_check = self.encoder.get_input_embeddings().weight.device if hasattr(self.encoder, "get_input_embeddings") else next(self.encoder.parameters()).device
        if input_ids.device != model_device_check:
            input_ids = input_ids.to(model_device_check)
        pooled = self._encode_state(input_ids, attention_mask)
        # Ensure pooled representation is on the Q-head device
        head_device = next(self.q_head.parameters()).device
        if pooled.device != head_device:
            pooled = pooled.to(head_device)
        return self.q_head(pooled)

    @torch.no_grad()
    def select_action(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, force_greedy: bool = False
    ) -> int:
        """Return a template index following ε-greedy exploration."""
        self._steps_done += 1
        epsilon = 0.0 if force_greedy else self._current_epsilon()
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        q_values = self.forward(input_ids, attention_mask if attention_mask is not None else None)
        return int(torch.argmax(q_values, dim=-1).item())

    # ------------------------------------------------------------------ DQN update
    def optimise_q_head(self) -> dict[str, float]:
        """One DQN update step if enough data are in the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        transitions = self.replay_buffer.sample(self.batch_size)
        # Unpack
        state_texts, actions, rewards, next_state_texts, dones = zip(*transitions)

        # Tokenise & pad
        states_enc = self.tokenizer(list(state_texts), return_tensors="pt", padding=True, truncation=True)
        next_states_enc = self.tokenizer(list(next_state_texts), return_tensors="pt", padding=True, truncation=True)

        input_ids = states_enc.input_ids
        attention_mask = states_enc.attention_mask
        next_input_ids = next_states_enc.input_ids
        next_attention_mask = next_states_enc.attention_mask

        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.bfloat16, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.bfloat16, device=self.device)

        # Q(s, a)
        q_values = self.forward(input_ids, attention_mask)
        state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # target = r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self._target_q_head(self._encode_state(next_input_ids, next_attention_mask))
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_values = reward_tensor + self.gamma * max_next_q * (1 - done_tensor)

        loss = F.smooth_l1_loss(state_action_values, target_values)

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self._steps_done % self.target_update_freq == 0:
            self._target_q_head.load_state_dict(self.q_head.state_dict())

        return {"high_level_policy_loss": loss.item()}

    # ------------------------------------------------------------------ checkpoint helpers
    def save(self, path: str) -> None:
        """
        Save the high-level policy to a directory.
        
        Args:
            path: Directory path to save the policy
        """
        os.makedirs(path, exist_ok=True)
        
        # Save the model weights
        torch.save({
            "q_head_state_dict": self.q_head.state_dict(),
            "target_state_dict": self._target_q_head.state_dict(),
            "steps_done": self._steps_done,
        }, os.path.join(path, "model.pt"))
        
        # Save metadata
        meta = {
            "policy_type": "dqn",
            "policy_kind": self.policy_kind,
            "num_actions": self.num_actions,
            "strategies": self.strategies,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "epsilon_start": self._epsilon_start,
            "epsilon_end": self._epsilon_end,
            "epsilon_decay": self._epsilon_decay,
            "hidden_size": self.hidden_size,
        }
        
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TemplateHighLevelPolicy":
        """
        Load a high-level policy from a directory.
        
        Args:
            path: Directory path containing the saved policy
            device: Device to load the policy on
            
        Returns:
            Loaded TemplateHighLevelPolicy instance
        """
        # Load metadata
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Load model weights
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        ckpt = torch.load(model_path, map_location=device)
        
        # For loading, we need to infer the hidden_size from the saved weights
        # since we don't have access to the actual encoder yet
        current_hidden_size = 768  # Default fallback
        if "q_head_state_dict" in ckpt and "0.weight" in ckpt["q_head_state_dict"]:
            current_hidden_size = ckpt["q_head_state_dict"]["0.weight"].shape[1]
            print(f"[HIGH-LEVEL-POLICY] Inferred hidden_size={current_hidden_size} from saved weights")
        else:
            print(f"[HIGH-LEVEL-POLICY] Using default hidden_size={current_hidden_size}")
        
        # Create a dummy encoder (will be replaced by the actual encoder from the environment)
        # We need to create the policy first, then load the weights
        dummy_encoder = nn.Linear(current_hidden_size, current_hidden_size)  # Placeholder encoder
        dummy_tokenizer = None  # Will be set by the environment
        
        policy = cls(
            encoder=dummy_encoder,
            tokenizer=dummy_tokenizer,
            templates_path="src/configs/templates.json",  # Will be overridden
            hidden_size=current_hidden_size,
            device=device,
            gamma=meta.get("gamma", 0.99),
            batch_size=meta.get("batch_size", 64),
            target_update_freq=meta.get("target_update_freq", 1000),
            epsilon_start=meta.get("epsilon_start", 1.0),
            epsilon_end=meta.get("epsilon_end", 0.05),
            epsilon_decay=meta.get("epsilon_decay", 10000),
        )
        
        # Load the saved weights with backward compatibility
        from src.utils import safe_load_state_dict
        
        # Load Q-head weights (architecture compatibility checked by safe_load_state_dict)
        if "q_head_state_dict" in ckpt:
            success, error = safe_load_state_dict(policy.q_head, ckpt, "q_head_state_dict", strict=True)
            if not success:
                raise RuntimeError(f"Failed to load Q-head: {error}")
            else:
                print(f"[HIGH-LEVEL-POLICY] Successfully loaded Q-head weights")
        else:
            print(f"[HIGH-LEVEL-POLICY] No Q-head weights found, will train from scratch")
        
        # Load target Q-head weights (architecture compatibility checked by safe_load_state_dict)
        if "target_state_dict" in ckpt:
            success, error = safe_load_state_dict(policy._target_q_head, ckpt, "target_state_dict", strict=True)
            if not success:
                raise RuntimeError(f"Failed to load target Q-head: {error}")
            else:
                print(f"[HIGH-LEVEL-POLICY] Successfully loaded target Q-head weights")
        else:
            print(f"[HIGH-LEVEL-POLICY] No target Q-head weights found, will train from scratch")
        
        # Load training metrics with backward compatibility
        policy._steps_done = ckpt.get("steps_done", 0)  # Default to 0 if not found
        
        return policy 