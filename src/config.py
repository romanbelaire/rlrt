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
    lr: float = 1e-4
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
    entropy_coef: float = 0.01
    kl_coef: float = 0.01
    perplexity_coef: float = 0.1
    
    # Randomization
    seed: Optional[int] = None
    
    # Algorithm selection
    alg: str = "ppo"
    attack_type: str = "standard"  # or "breakdown"
    
    # Output settings
    output_dir: str = "outputs"
    
    # Checkpoint loading
    load_from_checkpoint: Optional[str] = None  # Directory to load checkpoint from
    force_checkpoint_loading: bool = False  # If True, continue training even if some models are missing from checkpoint
    
    # PPO specific settings
    normalize_advantages: bool = True
    max_grad_norm: float = 1.0
    target_kl: Optional[float] = 0.01
    use_gae: bool = True
    
    # Learning rate scheduling
    use_linear_annealing: bool = True
    annealing_start_factor: float = 1.0
    annealing_end_factor: float = 0.1
    annealing_warmup_episodes: int = 10
    
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
    use_warmup: bool = False     # Whether to use warmup strategy

    # NEW: Warmup rollout steps (allows shorter rollouts during warmup)
    warmup_rollout_steps: int = 8

    # Marginal masking settings
    # Number of top-K tokens (based on attention) to consider when computing marginal rewards
    num_masked_marginal_tokens: int = 3
    # Whether to additionally compute pairwise marginal masking among the selected top-K tokens
    pairwise_marginal_masking: bool = True
    # NEW: If True, choose the second token in pairwise masking uniformly at random instead of exhaustively enumerating all pairs.
    uniform_pairwise_selection: bool = False
    # NEW: Number of random secondary tokens to sample per primary token when uniform_pairwise_selection is enabled.
    num_uniform_pair_samples: int = 1

    # If true, compute marginal rewards as Q_high(s, a_full) - Q_high(s, a_masked) using the current sequence
    # critic predictions, instead of R(s') - R(a) from the environment. Useful for ablation.
    Q_as_reward: bool = False
    
    # If true, use R(s') - R(a) formula for critic targets, where R(s') is target response safety
    # and R(a) is agent action safety. If false, use only R(a) (original behavior).
    use_rs_minus_ra_reward: bool = True

    # Memory optimization settings
    dynamic_batch_sizing: bool = False
    max_memory_per_gpu: float = 0.85
    chunked_forward_pass: bool = True
    chunk_size: int = 4
    use_first_layer_attention_only: bool = True
    
    # Actor/critic chunking controls
    actor_chunk_threshold: int = 2048
    
    # Model swapping for DeepSpeed compatibility
    enable_model_swapping: bool = False

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
    
    # Memory optimization for marginal reward computation
    max_marginal_token_positions: int = 10  # Maximum number of token positions to process for marginal rewards

    # Whether to use naïve back-to-front γ-decayed per-token rewards instead of marginal masking
    naive_decayed_token_reward: bool = False

    # HL-only training mode: if True, only train high-level policy and critics (freeze LL agent)
    hl_only_training: bool = False
    
    # Skywork reward model configuration for token-level critic training
    use_skywork_for_token_critic: bool = True  # Whether to use Skywork reward model for token-level critic
    skywork_model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"  # Skywork model to use
    skywork_eval_batch_size: int = 4  # Batch size for Skywork reward evaluation (conservative)


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    
    # Model identification
    model_name: str = "ppo-agent"
    hf_checkpoint: str = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Generation parameters
    action_branches: int = 3  # Number of actions to generate per state
    rejection_candidates: int = 6  # Number of rejection sampling candidates
    
    # Rejection sampling configuration
    use_combined_rejection_scoring: bool = True  # Whether to use combined v_high + perplexity scoring
    use_perplexity_only: bool = False  # Whether to use only perplexity for rejection sampling (overrides use_combined_rejection_scoring)
    perplexity_start_weight: float = 1.0  # Starting weight for perplexity score (v_high gets 1 - perplexity_weight)
    perplexity_end_weight: float = 0.4  # Final weight for perplexity score after annealing
    perplexity_decay_episodes: int = 100  # Number of episodes to decay perplexity weight from start to end

    
    # High-level policy configuration
    use_high_level_policy: bool = True
    high_level_lr: float = 0.001
    
    # Memory management
    keep_base_on_cpu: bool = False
    # Enable sharded placement via Transformers/Accelerate device_map
    enable_device_map: bool = False
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    target_layers: List[int] = field(default_factory=lambda: list(range(10,24)))
    
    # Target model settings
    target_model: Optional[str] = None
    
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
    

    
    # Generation strategy for low-level agent
    use_beam_search_generation: bool = False  # True: beam search, False: nucleus sampling
    
    # Refusal suppression during generation
    use_refusal_suppression: bool = True  # Whether to suppress refusal tokens during generation
    
    # Enhanced beam search for refusal filtering
    use_enhanced_beam_search: bool = False  # Whether to use enhanced beam search with refusal filtering
    enhanced_beam_search_beams: int = 10  # Number of beams for enhanced beam search
    enhanced_beam_search_sequences: int = 5  # Number of sequences to return from enhanced beam search

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
    
    # Template and perspective configuration
    templates_path: str = "src/configs/templates.json"
    scorecard_path: str = "scorecard.json"

    # DQN configuration (for TemplateHighLevelPolicy)
    epsilon_start: float = 1.0  # Starting value for epsilon-greedy exploration
    epsilon_end: float = 0.05   # Final value for epsilon-greedy exploration
    epsilon_decay: int = 1000   # Number of steps over which to decay epsilon
    replay_size: int = 1000     # Size of replay buffer
    dqn_batch_size: int = 32    # Batch size for DQN updates
    target_update_freq: int = 100  # How often to update target network
    dqn_gamma: float = 0.99     # Discount factor for DQN


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
    
    # White-box reward components
    similarity_weight: float = 0.1  # Weight for embedding similarity in reward calculation
    

    
    # Optional memory budget hints for very large target / guard models
    target_max_gpu_memory: Optional[str] = None
    guard_max_gpu_memory: Optional[str] = None
    target_max_cpu_memory: Optional[str] = None
    guard_max_cpu_memory: Optional[str] = None

    # Enable usage of LMRL-Gym general environments instead of adversarial MultiStepEnv
    use_lmrl: bool = False
    # Task identifier within LMRL-Gym (e.g., "maze", "wordle", "chess", "text_nav")
    lmrl_task: Optional[str] = None
    # Free-form options passed to the task factory (e.g., {"maze_name": "umaze", "max_steps": 50})
    lmrl_options: Dict[str, Any] = field(default_factory=dict)

    # New: environment task selector for general envs (e.g., "mc_multiturn")
    env_task: Optional[str] = None


@dataclass
class SemanticAnalysisConfig:
    """Configuration for semantic analysis data collection."""
    
    # Collection settings
    max_conversation_length: int = 30
    batch_size: int = 4
    save_interval: int = 100  # Save every N conversations
    
    # Vector storage settings
    faiss_index_type: str = "IndexFlatIP"
    embedding_dim: int = 4096  # Will be inferred from model
    
    # Output settings
    output_dir: str = "semantic_analysis_output"
    vector_store_path: str = "semantic_vectors"
    metadata_path: str = "conversation_metadata.json"
    
    # Model settings
    target_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    guard_model: str = "meta-llama/Llama-Guard-3-8B"
    use_quantization: bool = True
    
    # HarmBench data settings
    harmbench_path: str = "HarmBench/harmbench_behaviors_text_all.csv"
    category_column: str = "SemanticCategory"
    question_column: str = "Behavior"


@dataclass
class AgentConfig:
    """Master configuration for the Agent."""
    
    # Component configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    semantic_analysis: Optional[SemanticAnalysisConfig] = None
    
    # Flag for overriding saved config
    override: bool = False
    
    # Fixed paths and values
    VALID_ALGS: ClassVar[List[str]] = ["ppo", "reinforce", "ppo-pretrain"]
    VALID_ATTACK_TYPES: ClassVar[List[str]] = ["standard", "breakdown"]
    
    # Validation constants
    VALID_HIGH_LEVEL_POLICY_TYPES: ClassVar[List[str]] = ["dqn", "ppo", "llm", "hybrid"]

    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.training.alg not in self.VALID_ALGS:
            raise ValueError(f"Invalid algorithm: {self.training.alg}. Must be one of {self.VALID_ALGS}")
        
        if self.training.attack_type not in self.VALID_ATTACK_TYPES:
            raise ValueError(f"Invalid attack type: {self.training.attack_type}. Must be one of {self.VALID_ATTACK_TYPES}")
        
        if self.model.high_level_policy_type not in self.VALID_HIGH_LEVEL_POLICY_TYPES:
            raise ValueError(f"Invalid high-level policy type: {self.model.high_level_policy_type}. Must be one of {self.VALID_HIGH_LEVEL_POLICY_TYPES}")
        

        
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.training.minibatch_size <= 0:
            raise ValueError("Minibatch size must be positive")
        
        if self.training.minibatch_size > self.training.batch_size:
            raise ValueError("Minibatch size cannot be larger than batch size")
        
        if self.training.rollout_steps <= 0:
            raise ValueError("Rollout steps must be positive")
        
        if self.training.episodes <= 0:
            raise ValueError("Episodes must be positive")
        
        if self.training.lr <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.training.value_lr <= 0:
            raise ValueError("Value learning rate must be positive")

        # ------------------------------------------------------------------
        # Mutual exclusivity check for marginal masking options
        # ------------------------------------------------------------------
        if self.training.pairwise_marginal_masking and self.training.uniform_pairwise_selection:
            raise ValueError(
                "Configuration error: 'pairwise_marginal_masking' and 'uniform_pairwise_selection' "
                "cannot both be True. Please enable at most one of these options."
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "training": asdict(self.training),
            "model": asdict(self.model),
            "environment": asdict(self.environment)
        }
        if self.semantic_analysis is not None:
            config_dict["semantic_analysis"] = asdict(self.semantic_analysis)
        return config_dict
    
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
            semantic_dict = config_dict.get("semantic_analysis", {})
        else:
            # Flat format - extract component configs from top level
            training_dict = {k: v for k, v in config_dict.items() 
                            if k in [f.name for f in fields(TrainingConfig)]}
            model_dict = {k: v for k, v in config_dict.items() 
                         if k in [f.name for f in fields(ModelConfig)]}
            env_dict = {k: v for k, v in config_dict.items() 
                       if k in [f.name for f in fields(EnvironmentConfig)]}
            semantic_dict = {k: v for k, v in config_dict.items() 
                           if k in [f.name for f in fields(SemanticAnalysisConfig)]}
        

        
        # Handle legacy field name mappings for backward compatibility
        training_field_mappings = {
            'ce_coef': 'entropy_coef',  # Old name -> new name
            'adv_coef': None,  # Removed field
            'sem_coef': None,  # Removed field
        }
        
        model_field_mappings = {
            'detune': None,  # Removed field
        }
        
        env_field_mappings = {
            'use_custom_reward': None,  # Removed field
            'reward_model_path': None,  # Removed field
        }
        
        # Apply training field mappings
        for old_name, new_name in training_field_mappings.items():
            if old_name in training_dict:
                if new_name is not None:
                    training_dict[new_name] = training_dict.pop(old_name)
                    print(f"Warning: Mapped deprecated training field '{old_name}' to '{new_name}'")
                else:
                    training_dict.pop(old_name)
                    print(f"Warning: Removed deprecated training field '{old_name}'")
        
        # Apply model field mappings
        for old_name, new_name in model_field_mappings.items():
            if old_name in model_dict:
                if new_name is not None:
                    model_dict[new_name] = model_dict.pop(old_name)
                    print(f"Warning: Mapped deprecated model field '{old_name}' to '{new_name}'")
                else:
                    model_dict.pop(old_name)
                    print(f"Warning: Removed deprecated model field '{old_name}'")
        
        # Apply environment field mappings
        for old_name, new_name in env_field_mappings.items():
            if old_name in env_dict:
                if new_name is not None:
                    env_dict[new_name] = env_dict.pop(old_name)
                    print(f"Warning: Mapped deprecated environment field '{old_name}' to '{new_name}'")
                else:
                    env_dict.pop(old_name)
                    print(f"Warning: Removed deprecated environment field '{old_name}'")
        
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
        
        # Create semantic analysis config if provided
        semantic_config = None
        if semantic_dict:
            try:
                semantic_config = SemanticAnalysisConfig(**semantic_dict)
            except TypeError as e:
                valid_fields = [f.name for f in fields(SemanticAnalysisConfig)]
                invalid_fields = [k for k in semantic_dict.keys() if k not in valid_fields]
                raise ValueError(f"Invalid semantic analysis configuration fields: {invalid_fields}. Valid fields are: {valid_fields}") from e
        
        # Get override flag
        override = config_dict.get('override', False)
        
        # Create master config
        return cls(
            training=training_config,
            model=model_config,
            environment=env_config,
            semantic_analysis=semantic_config,
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


 