import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Tuple, Optional, Dict
import json
import random
import os

class HybridHighLevelPolicy(nn.Module):
    """Hybrid high-level policy combining MLP strategy selection with LLM customization.
    
    This approach provides the best of both worlds:
    - Stable, trainable strategy selection via MLP
    - Flexible, diverse instruction generation via LLM
    - Lower variance than pure LLM generation
    - Higher diversity than pure template selection
    """
    
    policy_kind: str = "hybrid"
    
    def __init__(
        self,
        mlp_policy: nn.Module,
        llm_model: nn.Module,
        llm_tokenizer: Any,
        strategies: Optional[List[Dict[str, Any]]] = None,
        templates_path: str = "src/configs/templates.json",
        instruction_prompts_path: str = "src/configs/instruction_prompts.json",
        prompt_diversity_config: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 48,
        temperature: float = 0.8,
        top_p: float = 0.9,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            mlp_policy: Pre-trained MLP policy for strategy selection (DQN or PPO style)
            llm_model: Language model for instruction customization
            llm_tokenizer: Tokenizer for the LLM
            strategies: List of strategy definitions (if None, loads from templates_path)
            templates_path: Path to templates.json for strategy definitions
            instruction_prompts_path: Path to instruction_prompts.json for diverse prompting
            prompt_diversity_config: Configuration for prompt diversity features
            max_new_tokens: Max tokens for LLM generation
            temperature: Sampling temperature for LLM
            top_p: Top-p sampling for LLM
            device: Device for computation
        """
        super().__init__()
        
        self.mlp_policy = mlp_policy
        self.llm_model = llm_model.eval()  # Keep frozen for now
        self.llm_tokenizer = llm_tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load strategies from MLP policy or templates
        if strategies is not None:
            self.strategies = strategies
        elif hasattr(mlp_policy, 'strategies'):
            self.strategies = mlp_policy.strategies
        else:
            self.strategies = self._load_strategies_from_templates(templates_path)
        
        # Load diverse prompt templates
        self.prompt_templates = self._load_prompt_templates(instruction_prompts_path)
        
        # Apply configuration overrides for prompt diversity
        if prompt_diversity_config:
            self._apply_diversity_config(prompt_diversity_config)
        
        # For compatibility
        self.num_actions = len(self.strategies)
        
        # Move models to device
        self.mlp_policy.to(self.device)
        self.llm_model.to(self.device)
        
    def _load_prompt_templates(self, prompts_path: str) -> Dict[str, Any]:
        """Load diverse prompt templates for instruction generation"""
        try:
            with open(prompts_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load prompt templates from {prompts_path}: {e}")
            # Fallback to default template structure
            return {
                "base_templates": [
                    {
                        "id": "default",
                        "template": "You are a strategic communication expert. Your task is to write a specific, actionable instruction that follows the given strategy.\n\nStrategy: {strategy_description}\nContext: {context}\n\nWrite a concise instruction (1-2 sentences) that implements this strategy for the given context. Be specific and creative while staying true to the strategy:\n\nInstruction:",
                        "weight": 1.0
                    }
                ],
                "style_conditions": [],
                "context_framings": [],
                "instruction_starters": [],
                "complexity_modifiers": [],
                "meta_config": {
                    "template_rotation_enabled": False,
                    "style_conditioning_enabled": False
                }
            }
    
    def _load_strategies_from_templates(self, templates_path: str) -> List[Dict[str, Any]]:
        """Load strategy definitions from templates.json"""
        try:
            with open(templates_path, 'r') as f:
                data = json.load(f)
            templates = data.get("templates", [])
            
            strategies = []
            for template in templates:
                # Create instruction from template or use existing
                instruction = template.get('instruction', template.get('description', f"Use {template.get('type', 'standard')} approach"))
                strategy = {
                    "id": template.get('id', f"strategy_{len(strategies)}"),
                    "instruction": instruction,
                    "type": template.get('type', 'standard'),
                    "base_template": template.get('template', '')
                }
                strategies.append(strategy)
            
            return strategies
        except Exception as e:
            # Fallback strategies
            return [
                {"id": "direct", "instruction": "Ask directly about the topic", "type": "direct"},
                {"id": "hypothetical", "instruction": "Frame as hypothetical scenario", "type": "hypothetical"},
                {"id": "research", "instruction": "Frame as research inquiry", "type": "research"},
                {"id": "educational", "instruction": "Frame as educational question", "type": "educational"}
            ]
    
    def _select_prompt_template(self) -> str:
        """Select a diverse prompt template using weighted random selection"""
        templates = self.prompt_templates.get("base_templates", [])
        if not templates:
            # Fallback template
            return "You are a strategic communication expert. Your task is to write a specific, actionable instruction that follows the given strategy.\n\nStrategy: {strategy_description}\nContext: {context}\n\nWrite a concise instruction (1-2 sentences) that implements this strategy for the given context. Be specific and creative while staying true to the strategy:\n\nInstruction:"
        
        # Weighted random selection
        weights = [t.get("weight", 1.0) for t in templates]
        selected_template = random.choices(templates, weights=weights)[0]
        return selected_template["template"]
    
    def _select_style_conditioning(self) -> Optional[str]:
        """Select a style conditioning prefix"""
        meta_config = self.prompt_templates.get("meta_config", {})
        if not meta_config.get("style_conditioning_enabled", False):
            return None
        
        style_conditions = self.prompt_templates.get("style_conditions", [])
        if not style_conditions:
            return None
        
        # Random selection with probability
        if random.random() < meta_config.get("combination_probability", 0.7):
            weights = [s.get("weight", 1.0) for s in style_conditions]
            selected_style = random.choices(style_conditions, weights=weights)[0]
            return selected_style["prefix"]
        
        return None
    
    def _select_context_framing(self) -> Optional[str]:
        """Select a context framing approach"""
        meta_config = self.prompt_templates.get("meta_config", {})
        if not meta_config.get("context_framing_enabled", False):
            return None
        
        context_framings = self.prompt_templates.get("context_framings", [])
        if not context_framings:
            return None
        
        # Random selection with probability
        if random.random() < meta_config.get("combination_probability", 0.7):
            weights = [f.get("weight", 1.0) for f in context_framings]
            selected_framing = random.choices(context_framings, weights=weights)[0]
            return selected_framing["framing"]
        
        return None
    
    def _select_complexity_modifier(self) -> Optional[str]:
        """Select a complexity modifier"""
        meta_config = self.prompt_templates.get("meta_config", {})
        if not meta_config.get("complexity_modification_enabled", False):
            return None
        
        complexity_modifiers = self.prompt_templates.get("complexity_modifiers", [])
        if not complexity_modifiers:
            return None
        
        # Random selection with probability
        if random.random() < meta_config.get("combination_probability", 0.7):
            weights = [m.get("weight", 1.0) for m in complexity_modifiers]
            selected_modifier = random.choices(complexity_modifiers, weights=weights)[0]
            return selected_modifier["modifier"]
        
        return None
    
    def _build_diverse_prompt(self, strategy: Dict[str, Any], state_text: str) -> str:
        """Build a diverse prompt using template rotation and style conditioning"""
        # Select base template
        base_template = self._select_prompt_template()
        
        # Get optional elements
        style_prefix = self._select_style_conditioning()
        context_framing = self._select_context_framing()
        complexity_modifier = self._select_complexity_modifier()
        
        # Build context (potentially with framing)
        context = state_text[:200]  # Truncate context
        if context_framing:
            context = context_framing.format(context=context)
        
        # Format base template
        # Handle both "instruction" and old "description" fields for compatibility
        strategy_instruction = strategy.get("instruction") or strategy.get("description", "Apply the strategy")
        prompt = base_template.format(
            strategy_description=strategy_instruction,  # Keep the placeholder name for template compatibility
            context=context
        )
        
        # Add style conditioning prefix if selected
        if style_prefix:
            # Insert style prefix before the instruction request
            prompt = prompt.replace("Instruction:", f"{style_prefix} Instruction:")
        
        # Add complexity modifier if selected
        if complexity_modifier:
            prompt = prompt.rstrip() + f" {complexity_modifier}\n\nInstruction:"
        
        return prompt
    
    @torch.no_grad()
    def _generate_custom_instruction(self, state_text: str, strategy: Dict[str, Any]) -> Tuple[str, torch.Tensor]:
        """Generate a customized instruction using the LLM for a specific strategy."""
        
        # Build diverse prompt using template rotation and style conditioning
        prompt = self._build_diverse_prompt(strategy, state_text)
        
        # Tokenize
        enc = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        outputs = self.llm_model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        
        # Extract generated instruction
        prompt_len = enc.input_ids.size(1)
        generated_ids = outputs.sequences[0, prompt_len:]
        instruction = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Compute approximate log probability
        scores = outputs.scores
        log_prob_sum = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        
        for step, step_logits in enumerate(scores):
            if step >= generated_ids.size(0):
                break
            step_log_probs = F.log_softmax(step_logits[0], dim=-1)
            token_id = generated_ids[step]
            log_prob_sum += step_log_probs[token_id]
        
        return instruction, log_prob_sum
    
    def act(self, state_texts: List[str]) -> Tuple[List[str], torch.Tensor]:
        """Generate customized instructions using hybrid MLP + LLM approach.
        
        Args:
            state_texts: List of conversation states as plain text
            
        Returns:
            Tuple of (instructions, log_probs) where:
            - instructions: List of customized instruction strings
            - log_probs: Tensor of log probabilities [batch_size]
        """
        batch_size = len(state_texts)
        instructions = []
        log_probs_list = []
        strategy_indices = []
        
        for i, state_text in enumerate(state_texts):
            # Step 1: Use MLP to select strategy
            if hasattr(self.mlp_policy, 'select_action'):
                # DQN-style policy
                enc = self.llm_tokenizer(state_text, return_tensors="pt", truncation=True)
                strategy_idx = self.mlp_policy.select_action(
                    enc.input_ids.to(self.device), 
                    enc.attention_mask.to(self.device) if enc.attention_mask is not None else None
                )
            elif hasattr(self.mlp_policy, 'act'):
                # PPO-style policy  
                enc = self.llm_tokenizer(state_text, return_tensors="pt", truncation=True)
                _, strategy_indices_tensor, _ = self.mlp_policy.act(
                    enc.input_ids.to(self.device), 
                    self.llm_tokenizer
                )
                strategy_idx = strategy_indices_tensor[0].item()
            else:
                # Fallback: random strategy
                strategy_idx = torch.randint(0, len(self.strategies), (1,)).item()
            
            strategy = self.strategies[strategy_idx]
            strategy_indices.append(strategy_idx)
            
            # Step 2: Use LLM to customize instruction for this strategy
            instruction, log_prob = self._generate_custom_instruction(state_text, strategy)
            
            # Fallback if generation fails
            if not instruction.strip():
                instruction = strategy.get("instruction") or strategy.get("description", "Follow the strategy for this context")
            
            instructions.append(instruction)
            log_probs_list.append(log_prob)
        
        # Stack log probabilities
        log_probs = torch.stack(log_probs_list) if log_probs_list else torch.zeros(batch_size, device=self.device)
        
        return instructions, log_probs
    
    def select_action(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> int:
        """Compatibility method for DQN-style interface."""
        # Delegate to MLP policy
        return self.mlp_policy.select_action(input_ids, attention_mask)
    
    def get_log_probs(self, state: torch.Tensor, strategy_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get log probabilities for given strategy choices (delegates to MLP policy)."""
        if hasattr(self.mlp_policy, 'get_log_probs'):
            return self.mlp_policy.get_log_probs(state, strategy_indices)
        else:
            # Fallback for DQN policies
            batch_size = state.size(0)
            dummy_log_probs = torch.zeros(batch_size, device=self.device)
            dummy_entropy = torch.zeros(batch_size, device=self.device)
            return dummy_log_probs, dummy_entropy
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (delegates to MLP policy)."""
        if hasattr(self.mlp_policy, 'forward'):
            return self.mlp_policy.forward(state)
        else:
            # Fallback
            batch_size = state.size(0)
            dummy_logits = torch.zeros(batch_size, self.num_actions, device=self.device)
            dummy_token_scores = torch.zeros(batch_size, state.size(1), device=self.device)
            return dummy_logits, dummy_token_scores
    
    def save(self, path: str) -> None:
        """Save the hybrid policy (saves MLP policy, LLM is assumed external)."""
        # Save MLP policy
        if hasattr(self.mlp_policy, 'save'):
            self.mlp_policy.save(path + "_mlp")
        
        # Save configuration
        config = {
            "strategies": self.strategies,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_actions": self.num_actions
        }
        
        torch.save(config, path + "_hybrid_config.pt")
    
    def load(self, path: str) -> None:
        """Load the hybrid policy configuration."""
        # Load MLP policy
        if hasattr(self.mlp_policy, 'load'):
            self.mlp_policy.load(path + "_mlp")
        
        # Load configuration
        config = torch.load(path + "_hybrid_config.pt", map_location=self.device)
        self.strategies = config["strategies"]
        self.max_new_tokens = config["max_new_tokens"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.num_actions = config["num_actions"]

    def _apply_diversity_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration overrides to prompt templates meta config"""
        meta_config = self.prompt_templates.setdefault("meta_config", {})
        
        # Update meta config with provided settings
        if "prompt_diversity_enabled" in config:
            meta_config["template_rotation_enabled"] = config["prompt_diversity_enabled"]
        if "style_conditioning_enabled" in config:
            meta_config["style_conditioning_enabled"] = config["style_conditioning_enabled"]
        if "context_framing_enabled" in config:
            meta_config["context_framing_enabled"] = config["context_framing_enabled"]
        if "complexity_modification_enabled" in config:
            meta_config["complexity_modification_enabled"] = config["complexity_modification_enabled"]
        if "combination_probability" in config:
            meta_config["combination_probability"] = config["combination_probability"]


def create_hybrid_policy(
    mlp_policy: nn.Module,
    llm_model: nn.Module,
    llm_tokenizer: Any,
    templates_path: str = "src/configs/templates.json",
    instruction_prompts_path: str = "src/configs/instruction_prompts.json",
    prompt_diversity_config: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 48,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: Optional[str] = None,
) -> HybridHighLevelPolicy:
    """Convenience function to create a hybrid policy."""
    return HybridHighLevelPolicy(
        mlp_policy=mlp_policy,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        templates_path=templates_path,
        instruction_prompts_path=instruction_prompts_path,
        prompt_diversity_config=prompt_diversity_config,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    ) 