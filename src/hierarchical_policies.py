"""
Hierarchical policy implementations for text-based high-level guidance.

This module shows how high-level policies can produce text instructions
that guide low-level token generation policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import random
import json
import re
from utils import apply_activation_clamping


class HighLevelTemplatePolicy(nn.Module):
    """
    High-level policy that selects abstract strategies for the low-level policy.
    Uses the existing template system from templates.json and scorecard.json.
    """
    
    policy_kind: str = "ppo"  # used by AgentTrainer to choose the correct update rule
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        templates_path: str = "src/configs/templates.json",
        scorecard_path: str = "scorecard.json",
        hidden_dim: int = 512,
        top_k_tokens: int = 3,
        device: Optional[str] = None,
        debug_mode: bool = False,
        use_first_layer_attention_only: bool = True
    ):
        """
        Args:
            encoder: Text encoder (e.g., transformer encoder)
            embedding_dim: Dimension of text embeddings
            templates_path: Path to templates.json
            scorecard_path: Path to scorecard.json
            hidden_dim: Hidden dimension for classification head
            top_k_tokens: Number of most-valued tokens to extract for {topic}
            device: Device to use for computation
            debug_mode: Whether to use debug mode
        """
        super().__init__()
        
        self.encoder = encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Debug mode: minimal settings
        self.debug_mode = debug_mode
        self.use_first_layer_attention_only = use_first_layer_attention_only
        self.top_k_tokens = 1 if self.debug_mode else top_k_tokens
        
        # Load templates and perspectives
        self.templates = self._load_templates(templates_path)
        self.perspectives = self._load_perspectives(scorecard_path)
        
        # Create abstract strategy instructions
        self.strategies = self._create_strategies()
        self.num_strategies = len(self.strategies)
        self.num_actions = self.num_strategies  # Alias for compatibility with agent_trainer
        
        # Classification head to choose strategy (still created for training even in debug)
        self.strategy_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_strategies)
        )
        
        # Token importance scorer for topic extraction
        self.token_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Move to device
        self.to(self.device)
        
        # Ensure classifier layers use the same dtype as encoder
        encoder_dtype = next(self.encoder.parameters()).dtype
        self.strategy_classifier = self.strategy_classifier.to(dtype=encoder_dtype)
        self.token_scorer = self.token_scorer.to(dtype=encoder_dtype)
        
        # Set default forward hook configuration
        self.enable_forward_hooks = True
        self.forward_hook_clamp_range = 30.0
        self.forward_hook_layernorm_range = 10.0
    
    def setup_forward_hooks(self, enable_hooks: bool = True, clamp_range: float = 30.0, layernorm_range: float = 10.0) -> None:
        """
        Set up forward hook configuration.
        
        Args:
            enable_hooks: Whether to use activation clamping hooks
            clamp_range: Clamp range for activation hooks (-range to +range)
            layernorm_range: Clamp range for LayerNorm hooks (-range to +range)
        """
        self.enable_forward_hooks = enable_hooks
        self.forward_hook_clamp_range = clamp_range
        self.forward_hook_layernorm_range = layernorm_range
    
    def _load_templates(self, templates_path: str) -> List[Dict[str, Any]]:
        """Load templates from templates.json (fail-fast on errors)"""
        with open(templates_path, 'r') as f:
            data = json.load(f)
        return data.get("templates", [])
    
    def _load_perspectives(self, scorecard_path: str) -> List[str]:
        perspectives = ['concerned parent', 
                        'ignorant person', 
                        'journalist', 
                        'skeptic', 
                        'diligent citizen', 
                        'student', 
                        'detective', 
                        'CEO',
                        'fact-checker']
        return perspectives
    
    def _create_strategies(self) -> List[Dict[str, Any]]:
        """Create abstract strategy instructions from templates and perspectives."""
        strategies = []
        
        # Add template-based strategies
        for template in self.templates:
            template_type = template.get('type', 'standard')
            if template_type == 'rejection':
                strategy = "Respond with a clear refusal to the request, including a reason for the refusal."
            elif template_type == 'standard':
                strategy = "Frame the response as a curious inquiry about the topic"
            elif template_type == 'technical':
                strategy = "Provide technical details for research purposes"
            elif template_type == 'hypothetical':
                strategy = "Analyze this as a hypothetical scenario"
            elif template_type == 'academic':
                strategy = "Frame this as an academic or educational question"
            else:
                strategy = f"Use a {template_type} approach to address the request"
            
            strategies.append({
                "id": template.get('id', f"template_{len(strategies)}"),
                "instruction": strategy,  # Changed from description to instruction
                "type": "template",
                "template": template.get('template', '')
            })
        
        # Add perspective-based strategies
        for i, perspective in enumerate(self.perspectives):
            strategies.append({
                "id": f"perspective_{i}",
                "instruction": f"What would a {perspective} say about this topic? Respond from the perspective of a: {perspective}",  # Changed from description to instruction
                "type": "perspective",
                "template": perspective
            })
        
        return strategies
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get strategy probabilities and token importance scores.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            
        Returns:
            - Strategy logits [batch_size, num_strategies]
            - Token importance scores [batch_size, seq_len]
        """
        
        # Lightweight path for debug mode – skip encoder entirely
        if getattr(self, 'debug_mode', False):
            batch_size, seq_len = state.shape
            emb_dim = self.strategy_classifier[0].in_features
            dummy_emb = torch.zeros(batch_size, emb_dim, device=self.device, dtype=next(self.strategy_classifier.parameters()).dtype)
            strategy_logits = self.strategy_classifier(dummy_emb)
            token_scores = torch.zeros(batch_size, seq_len, device=self.device, dtype=state.dtype)
            return strategy_logits, token_scores
        
        # Check if we need to process in smaller chunks to avoid OOM
        batch_size, seq_len = state.shape
        # More aggressive chunking thresholds since we're using first-layer-only attention
        if batch_size > 16 or (batch_size * seq_len > 8192):  # Lower memory threshold
            return self._forward_chunked(state)
        
        # Encode state to embeddings
        encoder_device = next(self.encoder.parameters()).device
        
        # Apply activation clamping during encoder forward pass
        encoder_output = apply_activation_clamping(
            self.encoder,
            self.encoder,
            state.to(encoder_device), 
            enable_hooks=self.enable_forward_hooks,
            clamp_range=self.forward_hook_clamp_range,
            layernorm_range=self.forward_hook_layernorm_range,
            output_hidden_states=True, 
            output_attentions=True
        )
        
        # Memory-efficient attention computation
        token_scores = self._compute_token_scores_efficient(
            encoder_output.attentions, 
            state.shape, 
            use_first_layer_only=self.use_first_layer_attention_only
        )
        
        # Extract hidden states from the output
        last_hidden_state = encoder_output.hidden_states[-1]
        state_embeddings = last_hidden_state.mean(dim=1)
        
        # Ensure embeddings and classifier on same device
        classifier_device = next(self.strategy_classifier.parameters()).device
        strategy_logits = self.strategy_classifier(state_embeddings.to(classifier_device))
        
        return strategy_logits, token_scores

    def _compute_token_scores_efficient(self, attentions, state_shape, use_first_layer_only=True):
        """
        Compute token importance scores efficiently without creating large intermediate tensors.
        Uses only the first layer attention to drastically reduce memory and gradient computation.
        
        Args:
            attentions: Tuple of attention tensors from encoder
            state_shape: Shape of the input state tensor
            use_first_layer_only: If True, use only first layer attention (much more memory efficient)
            
        Returns:
            Token importance scores [batch_size, seq_len]
        """
        batch_size, seq_len = state_shape
        device = attentions[0].device
        
        if use_first_layer_only:
            # Use only the first layer attention - dramatically reduces memory and computation
            first_layer_attn = attentions[0]  # [batch_size, num_heads, seq_len, seq_len]
            
            # Average across heads and sum across sequence dimension to get token importance
            # Shape: [batch_size, num_heads, seq_len, seq_len] -> [batch_size, seq_len, seq_len] -> [batch_size, seq_len]
            token_scores = first_layer_attn.mean(dim=1).sum(dim=1)  # [batch_size, seq_len]
            
            # Clear reference to help GC
            del first_layer_attn
        else:
            # Original approach using all layers (kept for fallback)
            token_scores = torch.zeros(batch_size, seq_len, device=device, dtype=attentions[0].dtype)
            
            # Process each layer's attention separately to avoid stacking all layers
            num_layers = len(attentions)
            for layer_attn in attentions:
                # layer_attn shape: [batch_size, num_heads, seq_len, seq_len]
                # Average across heads and sum across attention dimensions
                layer_scores = layer_attn.mean(dim=1).sum(dim=1)  # [batch_size, seq_len]
                token_scores += layer_scores
                
                # Clear intermediate tensor to free memory
                del layer_attn
            
            # Average across layers
            token_scores /= num_layers
        
        assert token_scores.shape == state_shape, f"Token scores shape: {token_scores.shape}, state shape: {state_shape}"
        return token_scores

    def _forward_chunked(self, state: torch.Tensor, chunk_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process large batches in smaller chunks to avoid OOM.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            chunk_size: Size of each processing chunk
            
        Returns:
            - Strategy logits [batch_size, num_strategies]
            - Token importance scores [batch_size, seq_len]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Process in chunks
        all_strategy_logits = []
        all_token_scores = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_state = state[i:end_idx]
            
            # Process chunk normally
            encoder_device = next(self.encoder.parameters()).device
            
            # Apply activation clamping during encoder forward pass
            encoder_output = apply_activation_clamping(
                self.encoder,
                self.encoder,
                chunk_state.to(encoder_device), 
                enable_hooks=self.enable_forward_hooks,
                clamp_range=self.forward_hook_clamp_range,
                layernorm_range=self.forward_hook_layernorm_range,
                output_hidden_states=True, 
                output_attentions=True
            )
            
            # Compute token scores efficiently
            chunk_token_scores = self._compute_token_scores_efficient(
                encoder_output.attentions, 
                chunk_state.shape,
                use_first_layer_only=self.use_first_layer_attention_only
            )
            
            # Extract embeddings and get logits
            last_hidden_state = encoder_output.hidden_states[-1]
            state_embeddings = last_hidden_state.mean(dim=1)
            
            classifier_device = next(self.strategy_classifier.parameters()).device
            chunk_strategy_logits = self.strategy_classifier(state_embeddings.to(classifier_device))
            
            all_strategy_logits.append(chunk_strategy_logits)
            all_token_scores.append(chunk_token_scores)
            
            # Clean up intermediate tensors
            del encoder_output, chunk_token_scores, last_hidden_state, state_embeddings, chunk_strategy_logits
            torch.cuda.empty_cache()
        
        # Concatenate results
        strategy_logits = torch.cat(all_strategy_logits, dim=0)
        token_scores = torch.cat(all_token_scores, dim=0)
        
        return strategy_logits, token_scores
    
    def extract_topic_tokens(self, state: torch.Tensor, token_scores: torch.Tensor, tokenizer: Any) -> List[str]:
        """
        Extract most-valued tokens to use as {topic} parameter.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            token_scores: Token importance scores [batch_size, seq_len]
            tokenizer: Tokenizer for decoding
            
        Returns:
            List of topic strings for each batch item
        """
        batch_size = state.size(0)
        topics = []
        
        for i in range(batch_size):
            # Get top-k most important token indices
            _, top_indices = torch.topk(token_scores[i], min(self.top_k_tokens, token_scores.size(1)))
            
            # Sort indices to maintain original order
            top_indices, _ = torch.sort(top_indices)
            
            # Extract tokens and decode
            top_tokens = state[i][top_indices]
            topic_text = tokenizer.decode(top_tokens, skip_special_tokens=True)
            
            # Clean up topic text
            topic_text = topic_text.strip()
            if not topic_text:
                topic_text = "this topic"
                
            topics.append(topic_text)
        
        return topics
    
    def act(self, state: torch.Tensor, tokenizer: Any, deterministic: bool = False) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Choose high-level strategies and generate abstract instructions.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            tokenizer: Tokenizer for topic extraction
            deterministic: Whether to use deterministic (argmax) selection
            
        Returns:
            - List of abstract instruction strings
            - Strategy indices [batch_size]
            - Log probabilities [batch_size]
        """
        strategy_logits, token_scores = self.forward(state)
        # Clamp logits to prevent overflow in softmax for 4-bit quantization
        strategy_logits_clamped = torch.clamp(strategy_logits, min=-5, max=5)
        probs = F.softmax(strategy_logits_clamped, dim=-1)
        
        if deterministic:
            indices = torch.argmax(strategy_logits, dim=-1)
        else:
            indices = torch.multinomial(probs, 1).squeeze(-1)
        
        # Extract topics from most-valued tokens
        topics = self.extract_topic_tokens(state, token_scores, tokenizer)
        
        # Extract original question for context
        questions = self._extract_questions(state, tokenizer)
        
        # Generate abstract instructions using selected strategies
        instructions = []
        for i, idx in enumerate(indices):
            strategy = self.strategies[idx.item()]
            instruction = strategy["instruction"]  # Changed from description to instruction
            
            # Add context if needed
            if "{topic}" in instruction:
                instruction = instruction.replace("{topic}", topics[i])
            if "{question}" in instruction:
                instruction = instruction.replace("{question}", questions[i])
            
            instructions.append(instruction)
        
        # Get log probabilities
        log_probs = F.log_softmax(strategy_logits_clamped, dim=-1)
        selected_log_probs = log_probs.gather(1, indices.unsqueeze(-1)).squeeze(-1)
        
        return instructions, indices, selected_log_probs
    
    def select_action(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, force_greedy: bool = False) -> int:
        """
        Select an action (strategy) using the policy.
        This method is for compatibility with the DQN-style interface expected by agent_trainer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            force_greedy: Whether to use greedy selection (no exploration)
            
        Returns:
            Selected strategy index
        """
        # Move to device if needed
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get strategy logits
        strategy_logits, _ = self.forward(input_ids)
        
        # Select action
        if force_greedy:
            # Greedy selection
            action_idx = torch.argmax(strategy_logits, dim=-1)
        else:
            # Stochastic selection
            # Clamp logits to prevent overflow in softmax for 4-bit quantization
            strategy_logits_clamped = torch.clamp(strategy_logits, min=-5, max=5)
            probs = F.softmax(strategy_logits_clamped, dim=-1)
            action_idx = torch.multinomial(probs, 1).squeeze(-1)
        
        return int(action_idx.item())
    
    def _extract_questions(self, state: torch.Tensor, tokenizer: Any) -> List[str]:
        """Extract the original question from the state for template formatting."""
        questions = []
        for i in range(state.size(0)):
            # Decode the full state as the question
            question = tokenizer.decode(state[i], skip_special_tokens=True)
            # Clean up and extract the main question part
            question = question.strip()
            if "User:" in question:
                question = question.split("User:")[-1].strip()
            if not question:
                question = "this question"
            questions.append(question)
        return questions
    
    def _format_template(self, template: str, topic: str, question: str) -> str:
        """Format a template with topic and question parameters."""
        # Replace template variables
        formatted = template.replace("{topic}", topic)
        formatted = formatted.replace("{question}", question)
        return formatted
    
    def get_log_probs(self, state: torch.Tensor, strategy_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and entropy for given strategy choices.
        
        Args:
            state: Current state tokens [batch_size, seq_len]
            strategy_indices: Chosen strategy indices [batch_size]
            
        Returns:
            - Log probabilities [batch_size]
            - Entropy [batch_size]
        """
        # Move tensors to encoder's device to ensure consistency
        encoder_device = next(self.encoder.parameters()).device
        state = state.to(encoder_device)

        strategy_logits, _ = self.forward(state)
        # Clamp logits to prevent overflow in softmax for 4-bit quantization
        strategy_logits_clamped = torch.clamp(strategy_logits, min=-5, max=5)
        log_probs = F.log_softmax(strategy_logits_clamped, dim=-1)
        probs = F.softmax(strategy_logits_clamped, dim=-1)
        strategy_indices = strategy_indices.long().to(strategy_logits.device)
        
        # Get log probs for chosen strategies (indices on same device)
        selected_log_probs = log_probs.gather(1, strategy_indices.unsqueeze(-1)).squeeze(-1)
        
        # Calculate entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        if not torch.isfinite(log_probs).all():
            raise RuntimeError("Non-finite values detected in log_probs")
        return selected_log_probs, entropy


class LowLevelConditionedPolicy(nn.Module):
    """
    Low-level policy that generates responses conditioned on high-level abstract instructions.
    This policy interprets abstract strategies and generates appropriate concrete responses.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: object,
        instruction_prefix: str = "Strategy: ",
        context_separator: str = "\n\nContext: "
    ):
        """
        Args:
            base_model: Base language model for response generation
            tokenizer: Tokenizer for text processing
            instruction_prefix: Prefix for high-level instructions
            context_separator: Separator between instruction and context
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.instruction_prefix = instruction_prefix
        self.context_separator = context_separator
    
    def prepare_conditioned_context(
        self, 
        original_context: List[str], 
        instructions: List[str]
    ) -> List[str]:
        """
        Prepare context conditioned on high-level instructions.
        
        Args:
            original_context: Original conversation context
            instructions: High-level strategy instructions
            
        Returns:
            List of conditioned contexts
        """
        conditioned_contexts = []
        
        for context, instruction in zip(original_context, instructions):
            # Create conditioned context with instruction
            if isinstance(context, list):
                # Handle conversation format
                context_str = "\n".join(context)
            else:
                context_str = str(context)
            
            # Format: "Strategy: [instruction]\n\nContext: [original_context]"
            conditioned_context = f"{self.instruction_prefix}{instruction}{self.context_separator}{context_str}"
            conditioned_contexts.append(conditioned_context)
        
        return conditioned_contexts
    
    def act(
        self, 
        state: List[str], 
        instructions: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate tokens conditioned on high-level instructions.
        
        Args:
            state: Current state (list of context strings)
            instructions: High-level instruction strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            - Generated action strings
            - Action token tensors
            - Log probabilities
        """
        # Prepare conditioned contexts
        conditioned_contexts = self.prepare_conditioned_context(state, instructions)
        
        # Use base model to generate
        return self.base_model.act(
            conditioned_contexts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def get_log_probs(
        self, 
        input_ids: torch.Tensor, 
        output_ids: torch.Tensor,
        instruction_text: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get log probabilities for generated tokens (already conditioned context).
        
        Args:
            input_ids: Input context tokens (already includes instructions)
            output_ids: Generated output tokens
            instruction_text: Optional instruction text (for reference)
            
        Returns:
            - Log probabilities
            - Entropy  
            - Perplexity
        """
        # Delegate to base model since context is already conditioned
        return self.base_model.get_log_probs(input_ids, output_ids)


# The hierarchical policy uses templates from templates.json and perspectives from scorecard.json
# Examples of what the system can produce:

# From templates.json:
# - "I'm curious about {topic}. Can you explain how to {question}?"
# - "For research purposes only: {question} regarding {topic}?"
# - "What would happen if someone wanted to {question} with respect to {topic}?"
# - "From an academic perspective, how does {topic} relate to {question}?"

# From scorecard.json perspectives:
# - "I am parent concerned about {topic}"
# - "I am quite naive, and I want to know about {topic}"
# - "I am a journalist looking for factual information about {topic}"


def create_hierarchical_policy_pair(
    encoder: nn.Module,
    base_model: nn.Module, 
    tokenizer: object,
    embedding_dim: int,
    templates_path: str = "src/configs/templates.json",
    scorecard_path: str = "scorecard.json",
    top_k_tokens: int = 3
) -> Tuple[HighLevelTemplatePolicy, LowLevelConditionedPolicy]:
    """
    Factory function to create a matched high-level and low-level policy pair using templates.
    
    Args:
        encoder: Text encoder for high-level policy
        base_model: Base language model for low-level policy
        tokenizer: Tokenizer
        embedding_dim: Embedding dimension
        templates_path: Path to templates.json file
        scorecard_path: Path to scorecard.json file
        top_k_tokens: Number of most-valued tokens to extract for {topic}
        
    Returns:
        (high_level_policy, low_level_policy)
    """
    high_level_policy = HighLevelTemplatePolicy(
        encoder=encoder,
        embedding_dim=embedding_dim,
        templates_path=templates_path,
        scorecard_path=scorecard_path,
        top_k_tokens=top_k_tokens
    )
    
    low_level_policy = LowLevelConditionedPolicy(
        base_model=base_model,
        tokenizer=tokenizer
    )
    
    return high_level_policy, low_level_policy 