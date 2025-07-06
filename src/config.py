"""
Configuration management for reinforcement learning agents.

This module provides utilities for managing, validating, and loading
configurations used throughout the codebase.
"""

import os
import json
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional, Union, Any, ClassVar


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Learning parameters
    lr: float = 5e-5
    value_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    eps: float = 0.2
    
    # Training schedule
    batch_size: int = 8
    minibatch_size: int = 8
    epochs: int = 3
    episodes: int = 1000
    rollout_steps: int = 5
    save_freq: int = 50
    
    # Loss coefficients
    ce_coef: float = 0.5
    kl_coef: float = 0.01
    adv_coef: float = 0.1
    sem_coef: float = 0.1
    perplexity_coef: float = 0.01
    
    # Randomization
    seed: Optional[int] = None
    
    # Algorithm selection
    alg: str = "ppo"
    attack_type: str = "standard"  # or "breakdown"
    
    # Output settings
    output_dir: str = "outputs"
    
    # PPO specific settings
    normalize_advantages: bool = True
    max_grad_norm: float = 1.0
    target_kl: Optional[float] = 0.01
    use_gae: bool = True
    
    # Distributed training
    use_accelerate: bool = True
    accelerate_config_path: Optional[str] = None
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    offload_optimizer_to_cpu: bool = False
    offload_param_to_cpu: bool = False
    # DeepSpeed / optimizer caching
    load_optimizer_states: bool = True  # if True, attempt to restore optimizer states from checkpoint
    num_gpus: int = 1
    local_rank: int = -1
    world_size: int = 1
    
    # Memory management
    max_gpu_memory: Union[str, float] = "32GiB"  # Can be string like "32GiB" or float like 0.4
    max_cpu_memory: str = "80GiB"
    use_offloading: bool = False

    debug_mode: bool = False

    # Warmup strategy for optimizer initialization
    warmup_batch_size: int = 8  # Small batch size for quick optimizer initialization
    use_warmup: bool = True     # Whether to use warmup strategy

    # NEW: Warmup rollout steps (allows shorter rollouts during warmup)
    warmup_rollout_steps: int = 8

    # Marginal masking settings
    # Number of top-K tokens (based on attention) to consider when computing marginal rewards
    num_masked_marginal_tokens: int = 3
    # Whether to additionally compute pairwise marginal masking among the selected top-K tokens
    pairwise_marginal_masking: bool = False

    # If true, compute marginal rewards as Q_high(s, a_full) - Q_high(s, a_masked) using the current sequence
    # critic predictions, instead of R(s') - R(a) from the environment. Useful for ablation.
    Q_as_reward: bool = False

    # Memory optimization settings
    dynamic_batch_sizing: bool = False
    max_memory_per_gpu: float = 0.85
    chunked_forward_pass: bool = True
    chunk_size: int = 4
    use_first_layer_attention_only: bool = True
    
    # Model swapping for DeepSpeed compatibility
    enable_model_swapping: bool = True

    # Critic training chunking settings
    critic_chunking_threshold: int = 100  # Sequence length threshold for chunking
    critic_max_chunk_size: int = 2  # Maximum samples per chunk for critic training
    critic_memory_fraction: float = 0.6  # Fraction of free memory to use for chunking
    
    # Actor training chunking settings
    actor_chunking_enabled: bool = False  # Whether to enable actor chunking
    actor_max_chunk_size: int = 2  # Maximum samples per chunk for actor training
    actor_memory_fraction: float = 0.5  # Fraction of free memory to use for actor chunking
    
    # Task reward settings for ablation studies
    use_task_reward: bool = True  # Whether to include instruction-following similarity in value targets
    
    # Token-level RL settings
    use_per_token_advantages: bool = True  # Whether to use per-token advantages (True) or sequence-level (False)


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    
    # Model identification
    model_name: str = "ppo-agent"
    hf_checkpoint: str = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Generation parameters
    action_branches: int = 3  # Number of actions to generate per state
    rejection_candidates: int = 8  # Number of rejection sampling candidates
    
    # High-level policy configuration
    use_high_level_policy: bool = False
    high_level_lr: float = 0.001
    
    # Memory management
    keep_base_on_cpu: bool = False
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    target_layers: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    # Target model settings
    target_model: Optional[str] = None
    detune: bool = False
    
    # Critic settings
    value_loss_coef: float = 0.5
    sequence_value_coef: float = 0.5  # Weight for sequence-level value loss
    token_value_coef: float = 0.5     # Weight for token-level value loss
    attention_divergence_weight: float = 0.001
    
    # Model paths
    adapter_path: Optional[str] = None
    sequence_critic_path: Optional[str] = None
    token_critic_path: Optional[str] = None
    high_level_policy_path: Optional[str] = None
    
    # Whether to merge adapters into base model
    merge_adapters: bool = False

    # New: fixed instruction for ablation of high-level agent. If set, the high-level
    # agent will ignore learned strategies and always output this string.
    fixed_high_level_instruction: Optional[str] = None

    # DeepSpeed inference configuration path (optional)
    ds_config_path: Optional[str] = None

    # Quantisation level for the agent model backbone (False, 4, or 8)
    quantize: Union[bool, int] = False
    
    # Forward hook configuration for activation clamping and stability
    enable_forward_hooks: bool = False  # Whether to attach activation clamping hooks
    forward_hook_clamp_range: float = 30.0  # Clamp range for activation hooks (-range to +range)
    forward_hook_layernorm_range: float = 10.0  # Clamp range for LayerNorm hooks (-range to +range)

    # High-level policy type selection
    high_level_policy_type: str = "dqn"  # "dqn", "ppo", "llm", or "hybrid"
    
    # Hybrid policy configuration
    hybrid_policy_enabled: bool = False
    hybrid_llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    hybrid_llm_quantize: Union[bool, int] = False
    hybrid_max_new_tokens: int = 48
    hybrid_temperature: float = 0.8
    hybrid_top_p: float = 0.9
    hybrid_use_separate_llm: bool = False
    hybrid_llm_device: str = "cuda:0"
    
    # Prompt diversity configuration
    hybrid_prompt_diversity_enabled: bool = True
    hybrid_style_conditioning_enabled: bool = True
    hybrid_context_framing_enabled: bool = True
    hybrid_complexity_modification_enabled: bool = True
    hybrid_combination_probability: float = 0.7


@dataclass
class EnvironmentConfig:
    """Configuration for environment parameters."""
    
    # Data source
    data_path: str = "HarmBench"  # Updated to point to HarmBench directory
    
    # Environment settings
    harm_threshold: float = 1.5
    max_steps: int = 5
    batch_size: int = 1
    max_new_tokens: int = 100  # Maximum tokens to generate for model responses
    
    # NEW: Warmup conversation bandwidth (fewer tokens generated during warmup)
    warmup_max_new_tokens: int = 20
    
    # Target model settings
    target_model: Optional[str] = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Guard model settings
    guard_model: Optional[str] = "meta-llama/Llama-Guard-3-8B"
    
    # Quantisation (bitsandbytes) configuration per model
    #   False -> no quantisation, 4 -> 4-bit, 8 -> 8-bit
    target_quantize: Union[bool, int] = False
    guard_quantize: Union[bool, int] = False
    
    # Reward configuration
    use_custom_reward: bool = False
    reward_model_path: Optional[str] = None
    
    # Optional memory budget hints for very large target / guard models
    target_max_gpu_memory: Optional[str] = None
    guard_max_gpu_memory: Optional[str] = None
    target_max_cpu_memory: Optional[str] = None
    guard_max_cpu_memory: Optional[str] = None


@dataclass
class AgentConfig:
    """Master configuration for the Agent."""
    
    # Component configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # Flag for overriding saved config
    override: bool = False
    
    # Fixed paths and values
    VALID_ALGS: ClassVar[List[str]] = ["ppo", "reinforce", "ppo-pretrain"]
    VALID_ATTACK_TYPES: ClassVar[List[str]] = ["standard", "breakdown"]
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate algorithm
        if self.training.alg not in self.VALID_ALGS:
            raise ValueError(f"Invalid algorithm: {self.training.alg}. Must be one of {self.VALID_ALGS}")
        
        # Validate attack type
        if self.training.attack_type not in self.VALID_ATTACK_TYPES:
            raise ValueError(f"Invalid attack type: {self.training.attack_type}. Must be one of {self.VALID_ATTACK_TYPES}")
        
        # Validate batch sizes
        if self.training.batch_size < self.training.minibatch_size:
            self.training.minibatch_size = self.training.batch_size
            print(f"Batch size ({self.training.batch_size}) should be >= minibatch size ({self.training.minibatch_size}). Setting minibatch size to batch size.")
        
        # Validate data path - support both files and directories
        if not os.path.exists(self.environment.data_path):
            raise FileNotFoundError(f"Data path not found: {self.environment.data_path}")
        
        # If path is a directory, check if it contains required files
        if os.path.isdir(self.environment.data_path):
            has_data_files = False
            for root, _, files in os.walk(self.environment.data_path):
                for file in files:
                    if file.endswith('.json') or file.endswith('.jsonl') or file.endswith('.csv'):
                        has_data_files = True
                        break
                if has_data_files:
                    break
            
            if not has_data_files:
                raise FileNotFoundError(f"No JSON, JSONL, or CSV files found in directory: {self.environment.data_path}")
        
        # Validate custom reward model path if used
        if self.environment.use_custom_reward and self.environment.reward_model_path:
            if not os.path.exists(self.environment.reward_model_path):
                raise FileNotFoundError(f"Reward model path not found: {self.environment.reward_model_path}")
        
        # Ensure rollout_steps positive
        if self.training.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")

        # NEW: Validate warmup rollout and bandwidth settings
        if self.training.warmup_rollout_steps <= 0:
            raise ValueError("warmup_rollout_steps must be positive")

        if self.environment.warmup_max_new_tokens <= 0:
            raise ValueError("warmup_max_new_tokens must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            
        Returns:
            AgentConfig instance
        """
        # Handle both nested and flat configuration formats
        if "training" in config_dict or "model" in config_dict or "environment" in config_dict:
            # Nested format - extract from subsections
            training_dict = config_dict.get("training", {})
            model_dict = config_dict.get("model", {})
            env_dict = config_dict.get("environment", {})
        else:
            # Flat format - extract component configs from top level
            training_dict = {k: v for k, v in config_dict.items() 
                            if k in [f.name for f in fields(TrainingConfig)]}
            model_dict = {k: v for k, v in config_dict.items() 
                         if k in [f.name for f in fields(ModelConfig)]}
            env_dict = {k: v for k, v in config_dict.items() 
                       if k in [f.name for f in fields(EnvironmentConfig)]}
        
        # Handle backward compatibility: map dataset_path to data_path if it exists
        if "dataset_path" in env_dict and "data_path" not in env_dict:
            env_dict["data_path"] = env_dict["dataset_path"]
            print(f"Warning: 'dataset_path' is deprecated, using value for 'data_path': {env_dict['data_path']}")
        
        # Handle backward compatibility: map guard_model_path to guard_model if it exists
        if "guard_model_path" in env_dict and "guard_model" not in env_dict:
            env_dict["guard_model"] = env_dict["guard_model_path"]
            print(f"Warning: 'guard_model_path' is deprecated, using value for 'guard_model': {env_dict['guard_model']}")
        
        # Create component configs with explicit error handling
        try:
            training_config = TrainingConfig(**training_dict)
        except TypeError as e:
            valid_fields = [f.name for f in fields(TrainingConfig)]
            invalid_fields = [k for k in training_dict.keys() if k not in valid_fields]
            raise ValueError(f"Invalid training configuration fields: {invalid_fields}. Valid fields are: {valid_fields}") from e
            
        try:
            model_config = ModelConfig(**model_dict)
        except TypeError as e:
            valid_fields = [f.name for f in fields(ModelConfig)]
            invalid_fields = [k for k in model_dict.keys() if k not in valid_fields]
            raise ValueError(f"Invalid model configuration fields: {invalid_fields}. Valid fields are: {valid_fields}") from e
            
        try:
            env_config = EnvironmentConfig(**env_dict)
        except TypeError as e:
            valid_fields = [f.name for f in fields(EnvironmentConfig)]
            invalid_fields = [k for k in env_dict.keys() if k not in valid_fields]
            raise ValueError(f"Invalid environment configuration fields: {invalid_fields}. Valid fields are: {valid_fields}") from e
        
        # Get override flag
        override = config_dict.get('override', False)
        
        # Create master config
        return cls(
            training=training_config,
            model=model_config,
            environment=env_config,
            override=override
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'AgentConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            AgentConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            'override': self.override
        }
        
        # Add training config
        for k, v in asdict(self.training).items():
            result[k] = v
            
        # Add model config
        for k, v in asdict(self.model).items():
            result[k] = v
            
        # Add environment config
        for k, v in asdict(self.environment).items():
            result[k] = v
            
        return result
    
    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> AgentConfig:
    """
    Get default configuration.
    
    Returns:
        Default AgentConfig instance
    """
    return AgentConfig()


def load_config(model_name: str) -> AgentConfig:
    """
    Load configuration for an existing model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        AgentConfig instance
    """
    config_path = f'./models/{model_name}/config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return AgentConfig.from_json(config_path)


 