#!/usr/bin/env python3
"""
Main entry point for the refactored RL agent framework.

This file provides a command-line interface to train and evaluate RL agents
using the refactored, component-based architecture.
"""

from datetime import datetime
import os
import sys
import argparse
import json
import traceback
import torch
import numpy as np
import time
from typing import Dict, Any, Optional, List, Union, Tuple

# Add the src directory to the path if running from the project root
if os.path.basename(os.getcwd()) != "src":
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.config import AgentConfig, TrainingConfig, ModelConfig, EnvironmentConfig
from src.base import Environment, Policy, Critic, BaseModelSingleton
from src.environment import MultiStepEnv
from src.policies import LowLevelPolicy
from src.critics import TokenLevelValueCritic, SequenceLevelValueCritic
from src.agent_trainer import AgentTrainer

# Import distributed training utilities if available
from src.distributed_utils import DistributedSetup

from transformers import AutoTokenizer
from src.utils import generate_training_plots


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train or evaluate a red teaming agent.")
    
    # Note: Training and testing are now handled by separate modules
    # Use test_runner.py for evaluation: python -m src.test_runner --config-path config.json --checkpoint-dir path/to/checkpoint
    
    # Configuration arguments
    parser.add_argument("--config_path", type=str, help="Path to JSON configuration file")
    parser.add_argument("--deepspeed_config_path", type=str, help="Path to DeepSpeed configuration file")
    
    # Add dynamic arguments for all config fields
    _add_config_arguments(parser)
    
    # Token-masked computation arguments (not in config, so add manually)
    parser.add_argument("--use_token_masked_computation", action="store_true", 
                      help="Enable token-masked computations with target and safety models")
    parser.add_argument("--disable_token_masked_computation", action="store_true", 
                      help="Disable token-masked computations")
    
    # Checkpoint loading arguments
    parser.add_argument("--force", action="store_true", 
                      help="Force checkpoint loading even if some models are missing (e.g., missing high-level policy)")
    
    # Other non-config arguments
    parser.add_argument("--device", type=str, help="Device to run the agent on")
    parser.add_argument("--log_level", type=str, choices=["debug", "info", "warning", "error"], 
                      default="info", help="Logging level")
    
    return parser.parse_args()


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Dynamically add arguments for all config fields.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    from dataclasses import fields
    
    # Get all config classes
    config_classes = [TrainingConfig, ModelConfig, EnvironmentConfig]
    
    # Track added arguments to avoid duplicates
    added_args = set()
    
    for config_class in config_classes:
        for field in fields(config_class):
            field_name = field.name
            arg_name = f"--{field_name}"
            
            # Skip if already added
            if arg_name in added_args:
                continue
            added_args.add(arg_name)
            
            # Get field type and default value
            field_type = field.type
            field_default = field.default if field.default is not field.default_factory else None
            
            # Special handling for specific fields
            if field_name == "mixed_precision":
                parser.add_argument(arg_name, type=str, choices=["fp16", "bf16", "no"], 
                                  help=f"{field_name} (from {config_class.__name__})")
            # Handle different field types
            elif field_type == bool:
                parser.add_argument(arg_name, action="store_true", 
                                  help=f"{field_name} (from {config_class.__name__})")
            elif field_type == int:
                parser.add_argument(arg_name, type=int, 
                                  help=f"{field_name} (from {config_class.__name__})")
            elif field_type == float:
                parser.add_argument(arg_name, type=float, 
                                  help=f"{field_name} (from {config_class.__name__})")
            elif field_type == str:
                parser.add_argument(arg_name, type=str, 
                                  help=f"{field_name} (from {config_class.__name__})")
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                # Handle List types
                if hasattr(field_type, '__args__'):
                    inner_type = field_type.__args__[0]
                    if inner_type == str:
                        parser.add_argument(arg_name, type=str, nargs="+", 
                                          help=f"{field_name} (from {config_class.__name__})")
                    elif inner_type == int:
                        parser.add_argument(arg_name, type=int, nargs="+", 
                                          help=f"{field_name} (from {config_class.__name__})")
                    else:
                        parser.add_argument(arg_name, type=str, nargs="+", 
                                          help=f"{field_name} (from {config_class.__name__})")
                else:
                    parser.add_argument(arg_name, type=str, nargs="+", 
                                      help=f"{field_name} (from {config_class.__name__})")
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                # Handle Union types (like Optional[str])
                args = field_type.__args__
                if type(None) in args:
                    # This is Optional[T]
                    non_none_types = [t for t in args if t is not type(None)]
                    if len(non_none_types) == 1:
                        inner_type = non_none_types[0]
                        if inner_type == str:
                            parser.add_argument(arg_name, type=str, 
                                              help=f"{field_name} (from {config_class.__name__})")
                        elif inner_type == int:
                            parser.add_argument(arg_name, type=int, 
                                              help=f"{field_name} (from {config_class.__name__})")
                        elif inner_type == float:
                            parser.add_argument(arg_name, type=float, 
                                              help=f"{field_name} (from {config_class.__name__})")
                        else:
                            parser.add_argument(arg_name, type=str, 
                                              help=f"{field_name} (from {config_class.__name__})")
                    else:
                        # Multiple non-None types, default to str
                        parser.add_argument(arg_name, type=str, 
                                          help=f"{field_name} (from {config_class.__name__})")
                else:
                    # Union without None, default to str
                    parser.add_argument(arg_name, type=str, 
                                      help=f"{field_name} (from {config_class.__name__})")
            else:
                # Default to string type for unknown types
                parser.add_argument(arg_name, type=str, 
                                  help=f"{field_name} (from {config_class.__name__})")
    
    # Add some common aliases for backward compatibility
    parser.add_argument("--learning_rate", type=float, help="Alias for --lr")
    parser.add_argument("--checkpoint", type=str, help="Alias for --hf_checkpoint")
    parser.add_argument("--data_dir", type=str, help="Alias for --data_path")


def build_config_from_args(args: argparse.Namespace) -> AgentConfig:
    """
    Build configuration from command line arguments, potentially loading from file.
    
    Args:
        args: Command line arguments
        
    Returns:
        Complete agent configuration
    """
    # Start with default configurations
    training_config = TrainingConfig()
    model_config = ModelConfig()
    env_config = EnvironmentConfig()
    
    # If config file specified, load it
    if args.config_path:
        try:
            with open(args.config_path, 'r') as f:
                config_dict = json.load(f)
                
            # Create config from file
            config = AgentConfig.from_dict(config_dict)
            training_config = config.training
            model_config = config.model
            env_config = config.environment
            
            print(f"Loaded configuration from {args.config_path}")
            if config.training.debug_mode:
                print("==DEBUG MODE ENABLED==")
        except Exception as e:
            # Re-raise the exception with additional context
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error loading configuration from {args.config_path}: {e}")
    
    # Apply command line arguments to appropriate config objects
    _apply_args_to_config(args, training_config, model_config, env_config)
    
    # Create the final configuration
    config = AgentConfig(
        training=training_config,
        model=model_config,
        environment=env_config
    )
    
    # Save whether to use token-masked computation as a custom attribute
    config.use_token_masked_computation = not args.disable_token_masked_computation if 'disable_token_masked_computation' in args.__dict__ else True
    if 'use_token_masked_computation' in args.__dict__ and args.use_token_masked_computation:
        config.use_token_masked_computation = True
    
    # Handle force flag for checkpoint loading
    if hasattr(args, 'force') and args.force:
        config.training.force_checkpoint_loading = True
        print("Force flag enabled: will continue training even if some models are missing from checkpoint")
    
    # Add DeepSpeed config path if provided
    if args.deepspeed_config_path is not None:
        config.deepspeed_config_path = args.deepspeed_config_path
    
    return config


def _apply_args_to_config(args: argparse.Namespace, training_config: TrainingConfig, 
                         model_config: ModelConfig, env_config: EnvironmentConfig) -> None:
    """
    Apply command line arguments to configuration objects.
    
    Args:
        args: Command line arguments
        training_config: Training configuration to update
        model_config: Model configuration to update
        env_config: Environment configuration to update
    """
    from dataclasses import fields
    import sys
    
    # Get field names for each config class
    training_fields = {f.name for f in fields(TrainingConfig)}
    model_fields = {f.name for f in fields(ModelConfig)}
    env_fields = {f.name for f in fields(EnvironmentConfig)}
    
    # Create a set of arguments that were explicitly provided on the command line
    # This is needed to distinguish between explicitly provided False values and defaults
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Remove the -- prefix and add to explicit args
            explicit_args.add(arg[2:])
    
    # Handle aliases first
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        training_config.lr = args.learning_rate
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        model_config.hf_checkpoint = args.checkpoint
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        env_config.data_path = args.data_dir
    
    # Apply all other arguments dynamically
    for arg_name, arg_value in vars(args).items():
        # Skip special arguments that are handled separately
        if arg_name in ['train', 'test', 'config_path', 'deepspeed_config_path', 
                       'use_token_masked_computation', 'disable_token_masked_computation',
                       'device', 'log_level', 'learning_rate', 'checkpoint', 'data_dir']:
            continue
            
        # Skip None values (but allow False for boolean flags)
        if arg_value is None:
            continue
            
        # For boolean fields, only apply if explicitly provided on command line
        if arg_name in training_fields:
            field_type = next((f.type for f in fields(TrainingConfig) if f.name == arg_name), None)
            if field_type == bool and arg_name not in explicit_args:
                continue  # Skip default boolean values
            if hasattr(training_config, arg_name):
                setattr(training_config, arg_name, arg_value)
        elif arg_name in model_fields:
            field_type = next((f.type for f in fields(ModelConfig) if f.name == arg_name), None)
            if field_type == bool and arg_name not in explicit_args:
                continue  # Skip default boolean values
            if hasattr(model_config, arg_name):
                setattr(model_config, arg_name, arg_value)
        elif arg_name in env_fields:
            field_type = next((f.type for f in fields(EnvironmentConfig) if f.name == arg_name), None)
            if field_type == bool and arg_name not in explicit_args:
                continue  # Skip default boolean values
            if hasattr(env_config, arg_name):
                setattr(env_config, arg_name, arg_value)
    
    # Handle special backward compatibility for use_offloading
    if hasattr(args, 'use_offloading') and args.use_offloading:
        # Also set model config for backward compatibility during transition
        model_config.keep_base_on_cpu = True


def setup_model_singletons(config: AgentConfig) -> Dict[str, Any]:
    """
    Initialize model singletons and their tokenizers for efficient memory use.
    
    Args:
        config: Agent configuration
        
    Returns:
        Dictionary of model singletons and tokenizers
    """
    # Disable tokenizer parallelism for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Initializing model singletons and tokenizers...")
    
    #set attention method to flash attention 2
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "flash_attention_2"
        print(f"Using attention implementation flash attention 2")
    
    # Initialize agent and critic singletons
    agent_singleton = BaseModelSingleton.get_instance(
        model_name=config.model.hf_checkpoint,
        use_offloading=config.training.use_offloading,
        config=config,
        ds_config=None  # Will set after initialization
    )
    
    critic_singleton = BaseModelSingleton.get_instance(
        model_name=config.model.hf_checkpoint,
        use_offloading=config.training.use_offloading,
        config=config,
        ds_config=None  # Will set after initialization
    )
    
    # Initialize main tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.hf_checkpoint, 
        use_fast=True,
        padding_side='left'
    )
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
    
    # Update singletons with tokenizer
    agent_singleton.tokenizer = tokenizer
    critic_singleton.tokenizer = tokenizer
    
    # ------------------------------------------------------------------
    # Force FlashAttention2 on all main backbones to avoid SDPA fallbacks.
    for _singleton in (agent_singleton, critic_singleton):
        _singleton.base_model.config.attn_implementation = "flash_attention_2"
        print(f"Using attention implementation flash attention 2 for {_singleton.model_name}")
    # ------------------------------------------------------------------
    
    # Initialize target and guard singletons
    target_singleton = None
    guard_singleton = None
    target_tokenizer = None
    guard_tokenizer = None
    
    # Determine GPU placement strategy for multi-GPU setups
    # Strategy: GPU 0 for agent models (training), GPU 1 for inference models (target + guard)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if 'target_model' in config.environment.__dict__ and config.environment.target_model:
        target_model_name = config.environment.target_model
        # Use default model if not specified
        if target_model_name is None:
            target_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Check if this is an OpenAI model
        openai_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo"]
        is_openai_target = any(target_model_name.startswith(model) for model in openai_models)
        
        if is_openai_target:
            print(f"OpenAI target model detected: {target_model_name}")
            print("Target model singleton and tokenizer will be handled by OpenAI API integration")
            # Set to None - environment will handle OpenAI API integration
            target_singleton = None
            target_tokenizer = None
        else:
            # For multi-GPU: place target model on GPU 1 (inference GPU)
            target_prefer_gpu = 1 if num_gpus > 1 else None
            print(f"Loading target model singleton: {target_model_name}")
            if target_prefer_gpu is not None:
                print(f"Multi-GPU setup detected: targeting TARGET model for GPU {target_prefer_gpu} (inference GPU)")
            
            # Get quantization setting from environment config
            target_quantize = getattr(config.environment, 'target_quantize', None)
            print(f"Target model quantization setting: {target_quantize}")
                
            target_singleton = BaseModelSingleton.get_instance(
                model_name=target_model_name,
                use_offloading=config.training.use_offloading,
                config=config,
                ds_config=None,
                prefer_gpu=target_prefer_gpu,
                quantize=target_quantize
            )
            
            # Initialize target tokenizer
            target_tokenizer = AutoTokenizer.from_pretrained(
                target_model_name, 
                use_fast=True,
                padding_side='left'
            )
            
            # Set pad token to eos token if not set
            if target_tokenizer.pad_token is None:
                if target_tokenizer.eos_token is not None:
                    target_tokenizer.pad_token = target_tokenizer.eos_token
                    target_tokenizer.pad_token_id = target_tokenizer.eos_token_id
                else:
                    target_tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
            
            target_singleton.tokenizer = target_tokenizer
            if hasattr(target_singleton.base_model.config, "attn_implementation"):
                target_singleton.base_model.config.attn_implementation = "flash_attention_2"
                print(f"Using attention implementation flash attention 2 for target model")
            
    if 'guard_model' in config.environment.__dict__ and config.environment.guard_model:
        guard_model_name = config.environment.guard_model
        # Use default guard model if not specified
        if guard_model_name is None:
            guard_model_name = "meta-llama/Llama-Guard-3-8B"
        
        # For multi-GPU: place guard model on GPU 1 (same as target, inference GPU)
        guard_prefer_gpu = 1 if num_gpus > 1 else None
        print(f"Loading guard model singleton: {guard_model_name}")
        if guard_prefer_gpu is not None:
            print(f"Multi-GPU setup detected: targeting GUARD model for GPU {guard_prefer_gpu} (inference GPU)")
        
        # Get quantization setting from environment config
        guard_quantize = getattr(config.environment, 'guard_quantize', None)
        print(f"Guard model quantization setting: {guard_quantize}")
            
        guard_singleton = BaseModelSingleton.get_instance(
            model_name=guard_model_name,
            use_offloading=config.training.use_offloading,
            config=config,
            ds_config=None,
            prefer_gpu=guard_prefer_gpu,
            quantize=guard_quantize
        )
        
        # Initialize guard tokenizer
        guard_tokenizer = AutoTokenizer.from_pretrained(
            guard_model_name, 
            use_fast=True,
            padding_side='left'
        )
        
        # Set pad token to eos token if not set
        if guard_tokenizer.pad_token is None:
            if guard_tokenizer.eos_token is not None:
                guard_tokenizer.pad_token = guard_tokenizer.eos_token
                guard_tokenizer.pad_token_id = guard_tokenizer.eos_token_id
            else:
                guard_tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
        
        guard_singleton.tokenizer = guard_tokenizer
        if hasattr(guard_singleton.base_model.config, "attn_implementation"):
            guard_singleton.base_model.config.attn_implementation = "flash_attention_2"
            print(f"Using attention implementation flash attention 2 for guard model")
    
    # Load DeepSpeed config if provided
    ds_config = None
    ds_cfg_path = None
    if hasattr(config, 'deepspeed_config_path') and config.deepspeed_config_path:
        ds_cfg_path = config.deepspeed_config_path
    elif hasattr(config.model, 'ds_config_path') and config.model.ds_config_path:
        ds_cfg_path = config.model.ds_config_path

    if ds_cfg_path:
        try:
            with open(ds_cfg_path, 'r') as f:
                ds_config = json.load(f)
            print(f"Loaded DeepSpeed configuration from {ds_cfg_path}")
        except Exception as e:
            print(f"Warning: Failed to load DeepSpeed config: {e}")
    
    # Update singletons with DeepSpeed config
    agent_singleton.ds_config = ds_config
    critic_singleton.ds_config = ds_config
    
    # Initialize LoRA for the agent model if enabled
    if config.model.use_lora:
        print(f"Initializing LoRA for agent model targeting specific layers")
        agent_singleton.initialize_with_lora(config.model)
    
    return {
        "agent": agent_singleton,
        "critic": critic_singleton,
        "tokenizer": tokenizer,
        "target": target_singleton,
        "target_tokenizer": target_tokenizer,
        "guard": guard_singleton,
        "guard_tokenizer": guard_tokenizer
    }


def build_environment(config: AgentConfig, tokenizer, singletons: Dict[str, Any]) -> Environment:
    """
    Build the training/testing environment.
    
    Args:
        config: Agent configuration
        tokenizer: Shared tokenizer
        singletons: Dictionary of model singletons
        
    Returns:
        Environment instance
    """
    # Verify required singletons are available
    if "guard" not in singletons or singletons["guard"] is None:
        raise ValueError("Guard model singleton not initialized")
    if "guard_tokenizer" not in singletons or singletons["guard_tokenizer"] is None:
        raise ValueError("Guard model tokenizer not initialized")
    
    # For OpenAI models, target singleton and tokenizer will be None
    # The environment will handle OpenAI API integration internally
    target_singleton = singletons.get("target")
    target_tokenizer = singletons.get("target_tokenizer")
    
    # Only check if both are None or both are not None (consistency check)
    if (target_singleton is None) != (target_tokenizer is None):
        raise ValueError("Target model singleton and tokenizer must both be None (OpenAI) or both be initialized (local)")
    
    # Get the HF token from the environment if available
    hf_token = os.environ.get("HF_TOKEN") or None
    
    # Pass the environment configuration and singletons to MultiStepEnv
    env = MultiStepEnv(
        config=config.environment,
        hf_token=hf_token,  # Pass HF token for model authentication
        target_singleton=target_singleton,
        target_tokenizer=target_tokenizer,
        guard_singleton=singletons["guard"],
        guard_tokenizer=singletons["guard_tokenizer"],
        debug_mode=config.training.debug_mode,
        max_new_tokens=config.environment.max_new_tokens,
        target_quantize=getattr(config.environment, 'target_quantize', None),
        guard_quantize=getattr(config.environment, 'guard_quantize', None),

    )
            
    return env


def build_policy(config: AgentConfig, singletons: Dict[str, Any], env: Environment) -> Policy:
    """
    Build the policy (actor) component.
    
    Args:
        config: Agent configuration
        singletons: Dictionary of model singletons
        
    Returns:
        Policy instance
    """
    # Build standard policy
    policy = LowLevelPolicy(
        hf_checkpoint=config.model.hf_checkpoint,
        tokenizer=singletons["tokenizer"],
        model=singletons["agent"].base_model,
        use_lora=config.model.use_lora,
        adapter_name="base_model" if config.model.use_lora else None,
        env=env
    )
    
    # Set up forward hooks with configuration
    policy.setup_forward_hooks(
        enable_hooks=config.model.enable_forward_hooks,
        clamp_range=config.model.forward_hook_clamp_range,
        layernorm_range=config.model.forward_hook_layernorm_range
    )
    
    return policy


def build_critics(config: AgentConfig, singletons: Dict[str, Any]) -> Dict[str, Critic]:
    """
    Build the critic components.
    
    Args:
        config: Agent configuration
        singletons: Dictionary of model singletons
        
    Returns:
        Dictionary of critic instances
    """
    critics = {}
    
    # Sequence-level critic
    sequence_critic = SequenceLevelValueCritic(
        hf_checkpoint=config.model.hf_checkpoint,
        tokenizer=singletons["tokenizer"],
        model=singletons["critic"].base_model
    )
    
    # Token-level critic
    token_critic = TokenLevelValueCritic(
        hf_checkpoint=config.model.hf_checkpoint,
        tokenizer=singletons["tokenizer"],
        model=singletons["critic"].base_model
    )
    
    # Set up forward hooks with configuration for both critics
    sequence_critic.setup_forward_hooks(
        enable_hooks=config.model.enable_forward_hooks,
        clamp_range=config.model.forward_hook_clamp_range,
        layernorm_range=config.model.forward_hook_layernorm_range
    )
    
    token_critic.setup_forward_hooks(
        enable_hooks=config.model.enable_forward_hooks,
        clamp_range=config.model.forward_hook_clamp_range,
        layernorm_range=config.model.forward_hook_layernorm_range
    )
    
    critics["sequence_critic"] = sequence_critic
    critics["token_critic"] = token_critic
    
    return critics

def build_high_level_policy(config: AgentConfig, singletons: Dict[str, Any]) -> Optional[Any]:
    """
    Build the high-level policy component if enabled.
    
    Args:
        config: Agent configuration
        singletons: Dictionary of model singletons
        
    Returns:
        High-level policy instance or None if not enabled
    """
    # Early exit: user disabled the high-level policy.
    if not getattr(config.model, 'use_high_level_policy', False):
        return None

    # From this point we know a high-level policy *is* requested.
    fixed_instr = getattr(config.model, 'fixed_high_level_instruction', None)

    # Device shared with critic (encoder)
    encoder = singletons["critic"].base_model
    device = next(encoder.parameters()).device

    # ------------------------------------------------------------------ fixed-policy path (still counted as high-level)
    if fixed_instr is not None:
        from fixed_high_level_policy import FixedHighLevelPolicy  # local import to avoid overhead if not needed
        return FixedHighLevelPolicy(instruction=fixed_instr, device=device)

    # ------------------------------------------------------------------ hybrid policy path
    policy_type = getattr(config.model, 'high_level_policy_type', 'dqn')
    hybrid_enabled = getattr(config.model, 'hybrid_policy_enabled', False)
    
    if policy_type == 'hybrid' and hybrid_enabled:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Building hybrid high-level policy...")
        
        # Build DQN policy component for template selection
        from high_level_policy import TemplateHighLevelPolicy
        embedding_dim = singletons["critic"].base_model.config.hidden_size
        
        # For hybrid policies, we use DQN-based template selection
        dqn_policy = TemplateHighLevelPolicy(
            encoder=encoder,
            tokenizer=singletons["tokenizer"],
            templates_path=getattr(config.model, 'templates_path', 'src/configs/templates.json'),
            hidden_size=embedding_dim,
            lr=config.model.high_level_lr,
            gamma=config.model.dqn_gamma,
            epsilon_start=config.model.epsilon_start,
            epsilon_end=config.model.epsilon_end,
            epsilon_decay=config.model.epsilon_decay,
            replay_size=config.model.replay_size,
            batch_size=config.model.dqn_batch_size,
            target_update_freq=config.model.target_update_freq,
            device=device
        )
        
        # Load LLM component
        llm_model_name = getattr(config.model, 'hybrid_llm_model_name', 'meta-llama/Llama-3.1-8B-Instruct')
        llm_quantize = getattr(config.model, 'hybrid_llm_quantize', False)
        llm_device = getattr(config.model, 'hybrid_llm_device', device)
        use_separate_llm = getattr(config.model, 'hybrid_use_separate_llm', False)
        
        if use_separate_llm:
            # Load separate LLM for instruction customization
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import os
            
            hf_token = os.environ.get("HF_TOKEN")
            llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=hf_token)
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            
            # Load with quantization if specified
            if llm_quantize:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(llm_quantize == 4),
                    load_in_8bit=(llm_quantize == 8),
                    bnb_4bit_compute_dtype=torch.bfloat16 if llm_quantize == 4 else None,
                    bnb_4bit_quant_type="nf4" if llm_quantize == 4 else None,
                )
                llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    quantization_config=bnb_config,
                    device_map=llm_device,
                    token=hf_token,
                    torch_dtype=torch.bfloat16
                )
            else:
                llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    device_map=llm_device,
                    token=hf_token,
                    torch_dtype=torch.bfloat16
                )
        else:
            # Reuse the main agent model for LLM component
            llm_model = singletons["agent"].base_model
            llm_tokenizer = singletons["tokenizer"]
        
        # Create hybrid policy
        from hybrid_high_level_policy import HybridHighLevelPolicy
        
        # Build prompt diversity config from model settings
        prompt_diversity_config = {
            "prompt_diversity_enabled": getattr(config.model, 'hybrid_prompt_diversity_enabled', True),
            "style_conditioning_enabled": getattr(config.model, 'hybrid_style_conditioning_enabled', True),
            "context_framing_enabled": getattr(config.model, 'hybrid_context_framing_enabled', True),
            "complexity_modification_enabled": getattr(config.model, 'hybrid_complexity_modification_enabled', True),
            "combination_probability": getattr(config.model, 'hybrid_combination_probability', 0.7)
        }
        
        high_level_policy = HybridHighLevelPolicy(
            dqn_policy=dqn_policy,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            templates_path="src/configs/templates.json",
            instruction_prompts_path="src/configs/instruction_prompts.json",
            prompt_diversity_config=prompt_diversity_config,
            max_new_tokens=getattr(config.model, 'hybrid_max_new_tokens', 48),
            temperature=getattr(config.model, 'hybrid_temperature', 0.8),
            top_p=getattr(config.model, 'hybrid_top_p', 0.9),
            device=device
        )
        
        return high_level_policy
    
    # ------------------------------------------------------------------ pure LLM policy path
    elif policy_type == 'llm':
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Building pure LLM high-level policy...")
        
        from llm_high_level_policy import LLMHighLevelPolicy
        
        # Use main agent model or separate LLM
        llm_model = singletons["agent"].base_model
        llm_tokenizer = singletons["tokenizer"]
        
        high_level_policy = LLMHighLevelPolicy(
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.95,
            device=device
        )
        
        return high_level_policy
    
    # ------------------------------------------------------------------ standard policy path (PPO/DQN)
    else:
        # Handle backward compatibility: "mlp" is an alias for "dqn"
        if policy_type == "mlp":
            policy_type = "dqn"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Converting 'mlp' alias to 'dqn' for backward compatibility")
        
        # Default to DQN if no policy type specified or if it's not recognized
        if policy_type not in ["dqn", "ppo", "llm", "hybrid"]:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Unknown policy type '{policy_type}', defaulting to 'dqn'")
            policy_type = "dqn"
        
        if policy_type == "dqn":
            # Use TemplateHighLevelPolicy for pure DQN-based template selection
            from high_level_policy import TemplateHighLevelPolicy
            
            # Get templates path
            templates_path = getattr(config.model, 'templates_path', 'src/configs/templates.json')
            
            high_level_policy = TemplateHighLevelPolicy(
                encoder=encoder,
                tokenizer=singletons["tokenizer"],
                templates_path=templates_path,
                hidden_size=encoder.config.hidden_size,
                lr=config.model.high_level_lr,
                gamma=config.model.dqn_gamma,
                epsilon_start=config.model.epsilon_start,
                epsilon_end=config.model.epsilon_end,
                epsilon_decay=config.model.epsilon_decay,
                replay_size=config.model.replay_size,
                batch_size=config.model.dqn_batch_size,
                target_update_freq=config.model.target_update_freq,
                device=device
            )
        elif policy_type == "ppo":
            # Use PPOTemplateHighLevelPolicy for PPO-based template selection
            from ppo_template_policy import PPOTemplateHighLevelPolicy
            
            # Get templates path
            templates_path = getattr(config.model, 'templates_path', 'src/configs/templates.json')
            
            high_level_policy = PPOTemplateHighLevelPolicy(
                encoder=encoder,
                tokenizer=singletons["tokenizer"],
                templates_path=templates_path,
                hidden_size=encoder.config.hidden_size,
                device=device
            )
        elif policy_type == "llm":
            # Use LLMHighLevelPolicy for LLM-based instruction generation
            from llm_high_level_policy import LLMHighLevelPolicy
            
            high_level_policy = LLMHighLevelPolicy(
                model=singletons["agent"].base_model,
                tokenizer=singletons["tokenizer"],
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.95,
                device=device
            )
        elif policy_type == "hybrid":
            # Use HybridHighLevelPolicy for MLP + LLM combination
            from hybrid_high_level_policy import HybridHighLevelPolicy
            
            # Build DQN component for hybrid policy
            from high_level_policy import TemplateHighLevelPolicy
            dqn_policy = TemplateHighLevelPolicy(
                encoder=encoder,
                tokenizer=singletons["tokenizer"],
                templates_path="src/configs/templates.json",
                hidden_size=encoder.config.hidden_size,
                lr=config.model.high_level_lr,
                gamma=config.model.dqn_gamma,
                epsilon_start=config.model.epsilon_start,
                epsilon_end=config.model.epsilon_end,
                epsilon_decay=config.model.epsilon_decay,
                replay_size=config.model.replay_size,
                batch_size=config.model.dqn_batch_size,
                target_update_freq=config.model.target_update_freq,
                device=device
            )
            
            # Build prompt diversity config from model settings
            prompt_diversity_config = {
                "prompt_diversity_enabled": getattr(config.model, 'hybrid_prompt_diversity_enabled', True),
                "style_conditioning_enabled": getattr(config.model, 'hybrid_style_conditioning_enabled', True),
                "context_framing_enabled": getattr(config.model, 'hybrid_context_framing_enabled', True),
                "complexity_modification_enabled": getattr(config.model, 'hybrid_complexity_modification_enabled', True),
                "combination_probability": getattr(config.model, 'hybrid_combination_probability', 0.7)
            }
            
            high_level_policy = HybridHighLevelPolicy(
                dqn_policy=dqn_policy,
                llm_model=singletons["agent"].base_model,
                llm_tokenizer=singletons["tokenizer"],
                templates_path="src/configs/templates.json",
                instruction_prompts_path="src/configs/instruction_prompts.json",
                prompt_diversity_config=prompt_diversity_config,
                max_new_tokens=getattr(config.model, 'hybrid_max_new_tokens', 48),
                temperature=getattr(config.model, 'hybrid_temperature', 0.8),
                top_p=getattr(config.model, 'hybrid_top_p', 0.9),
                device=device
            )
        
        return high_level_policy


def setup_distributed_training(config: AgentConfig) -> Optional[DistributedSetup]:
    """
    Set up memory-optimized model loading with Accelerate if enabled in config.
    
    Args:
        config: Agent configuration
        
    Returns:
        DistributedSetup instance or None if not using Accelerate
    """
    
    if not config.training.use_accelerate:
        return None
    
    # Initialize memory management setup
    print(f"Initializing memory-optimized setup with config: {config.training}")
    return DistributedSetup(config=config)  # Only pass the config object


def train_agent(config: AgentConfig):
    """
    Train an agent using the given configuration.
    
    Args:
        config: Agent configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Initialize distributed setup if requested
    distributed_setup = setup_distributed_training(config)
    is_distributed = distributed_setup is not None
    
    # Log configuration parameters
    print(f"Training config: {config.training}")
    print(f"Model config: {config.model}")
    print(f"Environment config: {config.environment}")
    
    # Save configuration
    save_configuration(config, config.training.output_dir)
    
    # Set up models and tokenizers
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Initializing models and tokenizers...")
    model_singletons = setup_model_singletons(config)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Build environment
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Building environment...")
    env = build_environment(config, model_singletons["tokenizer"], model_singletons)
    
    # Check if token-masked computation is enabled
    use_token_masked_computation = 'use_token_masked_computation' in config.__dict__ and config.use_token_masked_computation
    
    # Log token-masked computation status
    if use_token_masked_computation:
        print("Token-masked computations enabled. Models will be accessed through the environment interface.")
    else:
        print("Token-masked computations disabled.")
    
    # Check if we should load from checkpoint
    loaded_models = None
    if config.training.load_from_checkpoint:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Loading models from checkpoint: {config.training.load_from_checkpoint}")
        try:
            loaded_models = load_models_from_checkpoint(
                config.training.load_from_checkpoint, 
                config, 
                model_singletons,
                force=config.training.force_checkpoint_loading
            )
            print("Successfully loaded models from checkpoint")
        except Exception as e:
            print(f"Error loading from checkpoint: {e}")
            raise RuntimeError(f"Failed to load from checkpoint {config.training.load_from_checkpoint}: {e}")
    
    # Build policy
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Building policy...")
    if loaded_models and "policy" in loaded_models:
        policy = loaded_models["policy"]
        print("Using loaded policy from checkpoint")
    else:
        policy = build_policy(config, model_singletons, env)
        if loaded_models and config.training.force_checkpoint_loading:
            print("Created new policy (not found in checkpoint)")
    
    # Set to train mode
    policy.train()
    
    # Build critics
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Building critics...")
    if loaded_models and "sequence_critic" in loaded_models and "token_critic" in loaded_models:
        sequence_critic = loaded_models["sequence_critic"]
        token_critic = loaded_models["token_critic"]
        print("Using loaded critics from checkpoint")
    else:
        critics = build_critics(config, model_singletons)
        sequence_critic = critics['sequence_critic']
        token_critic = critics['token_critic']
        if loaded_models and config.training.force_checkpoint_loading:
            print("Created new critics (not found in checkpoint)")
    
    sequence_critic.train()
    token_critic.train()
    
    # Handle v_high_critic (optional, for backward compatibility)
    v_high_critic = None
    if loaded_models and "v_high_critic" in loaded_models:
        v_high_critic = loaded_models["v_high_critic"]
        if v_high_critic is not None:
            print("Using loaded v_high_critic from checkpoint")
            v_high_critic.train()
        else:
            print("v_high_critic not found in checkpoint (will be None)")
    else:
        print("v_high_critic not available (will be None)")
    
    # Explicitly move models to GPU for training
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Moving models to GPU...")
    policy.to(device)
    sequence_critic.to(device)
    token_critic.to(device)
    
    # Verify gradient setup for critics
    seq_trainable_params = sum(p.numel() for p in sequence_critic.parameters() if p.requires_grad)
    token_trainable_params = sum(p.numel() for p in token_critic.parameters() if p.requires_grad)
    print(f"Sequence critic: {seq_trainable_params} trainable parameters")
    print(f"Token critic: {token_trainable_params} trainable parameters")
    
    # Build high-level policy if enabled
    if loaded_models and "high_level_policy" in loaded_models:
        high_level_policy = loaded_models["high_level_policy"]
        print("Using loaded high-level policy from checkpoint")
    else:
        high_level_policy = build_high_level_policy(config, model_singletons)
        if loaded_models and config.training.force_checkpoint_loading and high_level_policy is not None:
            print("Created new high-level policy (not found in checkpoint)")
    
    if high_level_policy is not None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Building high-level policy...")
        high_level_policy.train()
        # Move high-level policy to GPU as well
        high_level_policy.to(device)
    
    # Update environment and policy with tokenizers
    env.tokenizer = model_singletons.get('tokenizer')
    policy.tokenizer = model_singletons.get('tokenizer')
    sequence_critic.tokenizer = model_singletons.get('tokenizer')
    token_critic.tokenizer = model_singletons.get('tokenizer')
    if high_level_policy is not None:
        high_level_policy.tokenizer = model_singletons.get('tokenizer')
    
    # ------------------------------------------------------------------
    # Ensure LoRA adapter weights are marked trainable – certain wrappers
    # (e.g. DeepSpeed-Inference) can flip ``requires_grad`` off.  We force
    # any parameter whose name contains "lora_" to be trainable.
    # ------------------------------------------------------------------
    for n, p in policy.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

    # Set up optimizers
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Setting up optimizers...")
    # ------------------------------------------------------------------
    # Optimise *only* the parameters that will actually receive gradients.
    # This is typically the small set of LoRA adapter weights that sit on
    # top of the frozen base model, or whichever parameters the user has
    # explicitly left with ``requires_grad = True``.
    # ------------------------------------------------------------------
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found for the actor – did you forget to enable LoRA or set requires_grad?"

    actor_optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.training.lr,
        eps=config.training.eps
    )
    sequence_critic_optimizer = torch.optim.Adam(
        sequence_critic.parameters(),
        lr=config.training.value_lr,
        eps=config.training.eps
    )
    token_critic_optimizer = torch.optim.Adam(
        token_critic.parameters(),
        lr=config.training.value_lr,
        eps=config.training.eps
    )
    
    # High-level policy optimizer (optional)
    high_level_optimizer = None
    if high_level_policy is not None:
        high_level_optimizer = torch.optim.Adam(
            high_level_policy.parameters(),
            lr=getattr(config.model, 'high_level_lr', 0.001),
            eps=config.training.eps
        )
    
    # Load optimizer states if available from checkpoint
    if loaded_models and "optimizer_states" in loaded_models:
        optimizer_states = loaded_models["optimizer_states"]
        print("Loading optimizer states from checkpoint...")
        
        try:
            if "actor_optimizer" in optimizer_states:
                actor_optimizer.load_state_dict(optimizer_states["actor_optimizer"])
                print("Loaded actor optimizer state")
            elif config.training.force_checkpoint_loading:
                print("Warning: Actor optimizer state not found in checkpoint, using fresh optimizer")
            if "sequence_critic_optimizer" in optimizer_states:
                sequence_critic_optimizer.load_state_dict(optimizer_states["sequence_critic_optimizer"])
                print("Loaded sequence critic optimizer state")
            elif config.training.force_checkpoint_loading:
                print("Warning: Sequence critic optimizer state not found in checkpoint, using fresh optimizer")
            if "token_critic_optimizer" in optimizer_states:
                token_critic_optimizer.load_state_dict(optimizer_states["token_critic_optimizer"])
                print("Loaded token critic optimizer state")
            elif config.training.force_checkpoint_loading:
                print("Warning: Token critic optimizer state not found in checkpoint, using fresh optimizer")
            if "high_level_optimizer" in optimizer_states and high_level_optimizer is not None:
                high_level_optimizer.load_state_dict(optimizer_states["high_level_optimizer"])
                print("Loaded high-level optimizer state")
            elif config.training.force_checkpoint_loading and high_level_optimizer is not None:
                print("Warning: High-level optimizer state not found in checkpoint, using fresh optimizer")
        except Exception as e:
            print(f"Warning: Failed to load optimizer states: {e}")
            print("Will continue with fresh optimizer states")
    else:
        print("No optimizer states found, using fresh optimizers")
    
    # Initialize trainer
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Initializing trainer...")
    trainer = AgentTrainer(
        actor_optimizer=actor_optimizer,
        sequence_critic_optimizer=sequence_critic_optimizer,
        token_critic_optimizer=token_critic_optimizer,
        high_level_optimizer=high_level_optimizer,
        clip_ratio=config.training.clip,
        value_loss_coef=config.model.value_loss_coef,
        entropy_coef=config.training.entropy_coef,
        kl_coef=config.training.kl_coef,
        perplexity_coef=config.training.perplexity_coef,
        max_grad_norm=config.training.max_grad_norm,
        device=device,
        target_kl=config.training.target_kl,
        distributed_setup=distributed_setup,
        use_token_masked_computation=use_token_masked_computation,
        minibatch_size=config.training.minibatch_size,
        config=config
    )
    
    # Register agent models for model swapping with DeepSpeed
    trainer.register_agent_models(
        policy=policy,
        sequence_critic=sequence_critic,
        token_critic=token_critic,
        high_level_policy=high_level_policy
    )
    
    # Train the agent using the trainer
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Starting training for {config.training.episodes} episodes...")
    
    # Create a wrapper function that passes config and high_level_policy to save_checkpoint
    def save_checkpoint_wrapper(policy, sequence_critic, token_critic, episode, metrics, output_dir, is_final=False):
        return save_checkpoint(
            policy=policy,
            sequence_critic=sequence_critic,
            token_critic=token_critic,
            episode=episode,
            metrics=metrics,
            output_dir=output_dir,
            is_final=is_final,
            config=config,
            high_level_policy=high_level_policy,
            v_high_critic=v_high_critic
        )
    
    # Get episode offset from loaded models if available
    episode_offset = 0
    if loaded_models and "episode_offset" in loaded_models:
        episode_offset = loaded_models["episode_offset"]
        print(f"Continuing training from episode {episode_offset + 1}")
    
    policy, sequence_critic, token_critic, v_high_critic, episode_rewards, run_id = trainer.train(
        policy=policy,
        sequence_critic=sequence_critic,
        token_critic=token_critic,
        environment=env,
        tokenizer=model_singletons["tokenizer"],
        num_episodes=config.training.episodes,
        rollout_steps=config.training.rollout_steps,
        save_freq=config.training.save_freq,
        output_dir=config.training.output_dir,
        save_checkpoint_fn=save_checkpoint_wrapper,
        high_level_policy=high_level_policy,
        v_high_critic=v_high_critic,
        episode_offset=episode_offset
    )
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: Training complete!")
    
    # ------------------------------------------------------------------
    # Generate diagnostic plots from the per-episode metrics CSV.
    # ------------------------------------------------------------------
    run_log_dir = os.path.join("logs", run_id)
    losses_csv_path = os.path.join(run_log_dir, "losses_per_episode.csv")
    if os.path.isfile(losses_csv_path):
        try:
            generate_training_plots(losses_csv_path, run_log_dir)
        except Exception as e:
            print(f"Failed to generate training plots: {e}")
    else:
        print(f"No losses CSV found at {losses_csv_path}")
    
    return policy, sequence_critic, token_critic, v_high_critic, episode_rewards


def load_models_from_checkpoint(checkpoint_dir: str, config: AgentConfig, model_singletons: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """
    Load models from a checkpoint directory and continue training.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        config: Current configuration (used as fallback if no config in checkpoint)
        model_singletons: Dictionary of model singletons
        force: If True, continue training even if some models are missing from checkpoint
        
    Returns:
        Dictionary containing loaded models and metadata with episode offset
        
    Raises:
        FileNotFoundError: If checkpoint directory or required model files don't exist (unless force=True)
        RuntimeError: If model loading fails
    """
    import os
    import json
    import re
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"Loading models from checkpoint: {checkpoint_dir}")
    
    # Extract episode number from checkpoint directory name
    checkpoint_name = os.path.basename(checkpoint_dir)
    episode_offset = 0
    
    # Try to extract episode number from directory name (e.g., "checkpoint_50" -> 50)
    if checkpoint_name.startswith("checkpoint_"):
        try:
            episode_offset = int(checkpoint_name.split("_")[1])
            print(f"Detected checkpoint from episode {episode_offset}")
        except (IndexError, ValueError):
            print("Could not extract episode number from checkpoint name, starting from episode 0")
            episode_offset = 0
    elif checkpoint_name == "final_model":
        # For final_model, we need to find the highest checkpoint number in the parent directory
        parent_dir = os.path.dirname(checkpoint_dir)
        if os.path.exists(parent_dir):
            checkpoint_dirs = [d for d in os.listdir(parent_dir) if d.startswith("checkpoint_")]
            if checkpoint_dirs:
                try:
                    episode_numbers = [int(d.split("_")[1]) for d in checkpoint_dirs]
                    episode_offset = max(episode_numbers)
                    print(f"Detected final model from episode {episode_offset}")
                except (IndexError, ValueError):
                    print("Could not extract episode number from checkpoint directories, starting from episode 0")
                    episode_offset = 0
            else:
                print("No numbered checkpoints found, starting from episode 0")
                episode_offset = 0
        else:
            print("Parent directory not found, starting from episode 0")
            episode_offset = 0
    else:
        print(f"Unknown checkpoint format: {checkpoint_name}, starting from episode 0")
        episode_offset = 0
    
    # Check for config file in checkpoint
    checkpoint_config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(checkpoint_config_path):
        try:
            with open(checkpoint_config_path, 'r') as f:
                checkpoint_config_dict = json.load(f)
            checkpoint_config = AgentConfig.from_dict(checkpoint_config_dict)
            print(f"Loaded configuration from checkpoint: {checkpoint_config_path}")
        except Exception as e:
            print(f"Warning: Failed to load config from checkpoint, using provided config: {e}")
            checkpoint_config = config
    else:
        print("No config found in checkpoint, using provided config")
        checkpoint_config = config
    
    # Check for model directories
    policy_dir = os.path.join(checkpoint_dir, "policy")
    sequence_critic_dir = os.path.join(checkpoint_dir, "sequence_critic")
    token_critic_dir = os.path.join(checkpoint_dir, "token_critic")
    high_level_policy_dir = os.path.join(checkpoint_dir, "high_level_policy")
    v_high_critic_dir = os.path.join(checkpoint_dir, "v_high_critic")
    
    # Verify model directories exist
    missing_models = []
    if not os.path.exists(policy_dir):
        missing_models.append("policy")
    if not os.path.exists(sequence_critic_dir):
        missing_models.append("sequence_critic")
    if not os.path.exists(token_critic_dir):
        missing_models.append("token_critic")
    
    if missing_models and not force:
        raise FileNotFoundError(f"Missing model directories in checkpoint: {missing_models}. Use --force to continue training with missing models.")
    
    # Load models
    loaded_models = {}
    
    try:
        # Load policy
        if os.path.exists(policy_dir):
            print("Loading policy...")
            policy = LowLevelPolicy.load(policy_dir, tokenizer=model_singletons["tokenizer"])
            loaded_models["policy"] = policy
        elif force:
            print("Warning: Policy not found in checkpoint, will create new policy")
        else:
            raise FileNotFoundError(f"Policy not found in checkpoint: {policy_dir}")
        
        # Load critics - they need to be instantiated first with the model, then loaded
        if os.path.exists(sequence_critic_dir):
            print("Loading sequence critic...")
            sequence_critic = SequenceLevelValueCritic(
                hf_checkpoint=config.model.hf_checkpoint,
                tokenizer=model_singletons["tokenizer"],
                model=model_singletons["critic"].base_model
            )
            sequence_critic.load(sequence_critic_dir)
            loaded_models["sequence_critic"] = sequence_critic
        elif force:
            print("Warning: Sequence critic not found in checkpoint, will create new sequence critic")
        else:
            raise FileNotFoundError(f"Sequence critic not found in checkpoint: {sequence_critic_dir}")
        
        if os.path.exists(token_critic_dir):
            print("Loading token critic...")
            token_critic = TokenLevelValueCritic(
                hf_checkpoint=config.model.hf_checkpoint,
                tokenizer=model_singletons["tokenizer"],
                model=model_singletons["critic"].base_model
            )
            token_critic.load(token_critic_dir)
            loaded_models["token_critic"] = token_critic
        elif force:
            print("Warning: Token critic not found in checkpoint, will create new token critic")
        else:
            raise FileNotFoundError(f"Token critic not found in checkpoint: {token_critic_dir}")
        
        # Load v_high critic if it exists (optional, for backward compatibility)
        if os.path.exists(v_high_critic_dir):
            print("Loading v_high critic...")
            # For now, we'll create a simple critic with the same architecture as sequence critic
            # This can be enhanced later to support different v_high critic architectures
            v_high_critic = SequenceLevelValueCritic(
                hf_checkpoint=config.model.hf_checkpoint,
                tokenizer=model_singletons["tokenizer"],
                model=model_singletons["critic"].base_model
            )
            v_high_critic.load(v_high_critic_dir)
            loaded_models["v_high_critic"] = v_high_critic
        else:
            print("Note: v_high critic not found in checkpoint (this is optional for backward compatibility)")
            loaded_models["v_high_critic"] = None
        
        # Load high-level policy if it exists
        if os.path.exists(high_level_policy_dir):
            print("Loading high-level policy...")
            # Import here to avoid circular imports
            from high_level_policy import TemplateHighLevelPolicy
            # Use the class method to load the policy directly
            high_level_policy = TemplateHighLevelPolicy.load(
                high_level_policy_dir, 
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # Set the actual encoder and tokenizer after loading
            high_level_policy.encoder = model_singletons["critic"].base_model
            high_level_policy.tokenizer = model_singletons["tokenizer"]
            loaded_models["high_level_policy"] = high_level_policy
        elif force:
            print("Warning: High-level policy not found in checkpoint, will create new high-level policy")
        else:
            print("Note: High-level policy not found in checkpoint (this is optional)")
        
        # Load optimizer states if they exist
        optimizer_states = {}
        optimizer_dir = os.path.join(checkpoint_dir, "optimizer_states")
        if os.path.exists(optimizer_dir):
            print("Loading optimizer states...")
            try:
                optimizer_states["actor_optimizer"] = torch.load(os.path.join(optimizer_dir, "actor_optimizer.pt"))
                optimizer_states["sequence_critic_optimizer"] = torch.load(os.path.join(optimizer_dir, "sequence_critic_optimizer.pt"))
                optimizer_states["token_critic_optimizer"] = torch.load(os.path.join(optimizer_dir, "token_critic_optimizer.pt"))
                
                # Check for high-level optimizer
                hl_optimizer_path = os.path.join(optimizer_dir, "high_level_optimizer.pt")
                if os.path.exists(hl_optimizer_path):
                    optimizer_states["high_level_optimizer"] = torch.load(hl_optimizer_path)
                    
                print("Successfully loaded optimizer states")
            except Exception as e:
                print(f"Warning: Failed to load optimizer states: {e}")
                print("Will initialize new optimizers")
        else:
            print("Warning: No optimizer states found in checkpoint. Will initialize new optimizers.")
        
        # Load metrics if they exist
        metrics = {}
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print(f"Loaded metrics from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load metrics: {e}")
        
        loaded_models["optimizer_states"] = optimizer_states
        loaded_models["metrics"] = metrics
        loaded_models["checkpoint_config"] = checkpoint_config
        loaded_models["episode_offset"] = episode_offset
        
        print(f"Successfully loaded all models from checkpoint: {checkpoint_dir}")
        print(f"Will continue training from episode {episode_offset + 1}")
        return loaded_models
        
    except Exception as e:
        raise RuntimeError(f"Failed to load models from checkpoint {checkpoint_dir}: {e}")


def save_checkpoint(
    policy: Policy,
    sequence_critic: Critic,
    token_critic: Critic,
    episode: int,
    metrics: Dict[str, float],
    output_dir: str,
    is_final: bool = False,
    config: Optional[AgentConfig] = None,
    high_level_policy: Optional[Any] = None,
    v_high_critic: Optional[Any] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        policy: Agent policy
        sequence_critic: Sequence-level critic
        token_critic: Token-level critic
        episode: Current episode
        metrics: Training metrics
        output_dir: Output directory
        is_final: Whether this is the final checkpoint
        config: Configuration to save (optional)
        high_level_policy: High-level policy to save (optional)
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_{episode}")
    if is_final:
        checkpoint_dir = os.path.join(output_dir, "final_model")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save policy
    policy_path = os.path.join(checkpoint_dir, "policy")
    policy.save(policy_path)
    
    # Save critics
    sequence_critic_path = os.path.join(checkpoint_dir, "sequence_critic")
    sequence_critic.save(sequence_critic_path)
    
    token_critic_path = os.path.join(checkpoint_dir, "token_critic")
    token_critic.save(token_critic_path)
    
    # Save high-level policy if provided
    if high_level_policy is not None:
        hl_path = os.path.join(checkpoint_dir, "high_level_policy")
        high_level_policy.save(hl_path)
    
    # Save v_high_critic if provided
    if v_high_critic is not None:
        v_high_critic_path = os.path.join(checkpoint_dir, "v_high_critic")
        v_high_critic.save(v_high_critic_path)
    
    # Save configuration if provided
    if config is not None:
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Checkpoint saved to {checkpoint_dir}")





def save_configuration(config: AgentConfig, output_dir: str) -> None:
    """
    Save the configuration to a JSON file.
    
    Args:
        config: Agent configuration
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    
    config.save(config_path)
    print(f"Configuration saved to {config_path}")


def main():
    """Main function to parse arguments and run the agent."""
    args = parse_args()
    start_time = time.time()
    
    # Build configuration from arguments and/or config file
    config = build_config_from_args(args)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Save the configuration
    save_configuration(config, config.training.output_dir)
    
    # Run training (testing is now handled by test_runner.py)
    try:
        train_agent(config)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        raise e
    finally:
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 