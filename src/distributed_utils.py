"""
Memory and device management utilities for the RL agent framework.

This module provides utilities for efficiently managing model memory using Accelerate's
auto device map functionality, handling GPU memory constraints, and managing
device placement for large language models.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from config import TrainingConfig, ModelConfig, AgentConfig

from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch


class DistributedSetup:
    """
    Handle device management and memory optimization for the RL agent framework.
    
    This class provides a unified interface for managing model memory and device placement,
    with a focus on Accelerate's auto device map functionality for large models.
    """
    
    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the memory and device management setup.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.training_config = config.training
        self.use_accelerate = self.training_config.use_accelerate
        
        # Set device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize is_main_process_flag
        self.is_main_process_flag = True  # Default to True for non-distributed settings
        
        # Initialize accelerator if using Accelerate
        if self.use_accelerate:
            self.setup_accelerate()
        
        # Set seed for reproducibility
        if self.training_config.seed is not None:
            if self.use_accelerate:
                set_seed(self.training_config.seed)
            else:
                torch.manual_seed(self.training_config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.training_config.seed)
    
    @property
    def device(self) -> torch.device:
        """
        Get the device to use for this setup.
        
        Returns:
            torch.device: The device to use (CPU or CUDA)
        """
        if self.use_accelerate:
            return self.accelerator.device
        else:
            # Default device
            return self._device
    
    def is_main_process(self) -> bool:
        """
        Check if this is the main process in a distributed setup.
        
        Returns:
            bool: True if this is the main process, False otherwise
        """
        # Return the attribute directly
        return self.is_main_process_flag
    
    def setup_accelerate(self) -> None:
        """Set up the Accelerate environment."""
        # Configure accelerator parameters directly
        accelerate_kwargs = {
            "mixed_precision": self.training_config.mixed_precision,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "device_placement": True  # Let Accelerate handle device placement
        }
        
        # Add CPU offload if configured
        if self.training_config.offload_optimizer_to_cpu:
            accelerate_kwargs["cpu"] = True
        
        # Initialize the accelerator
        self.accelerator = Accelerator(**accelerate_kwargs)
        
        # Store information about the distributed environment
        self.is_main_process_flag = self.accelerator.is_main_process
        self.local_rank = self.accelerator.local_process_index
        self.world_size = self.accelerator.num_processes
        
        print(f"Accelerate setup complete. Process rank: {self.local_rank}, "
              f"World size: {self.world_size}, Is main: {self.is_main_process_flag}")
    
    def load_model_with_device_map(self, model_cls, model_name, max_memory=None, device_map="auto", **model_kwargs):
        """
        Load a model with Accelerate's auto device map functionality to manage memory.
        
        Args:
            model_cls: Model class to instantiate
            model_name: Name/path of the model to load
            max_memory: Dictionary specifying maximum memory for each device
                        Example: {0: "10GiB", 1: "10GiB", "cpu": "30GiB"}
            device_map: Device map strategy ("auto" or custom map)
            **model_kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Loaded model with optimized device map
        """
        
        # Default max_memory if not provided
        if max_memory is None:
            # Check if we have max memory settings in config
            if 'max_gpu_memory' in self.config.training.__dict__ and self.config.training.max_gpu_memory:
                # Use memory settings from config
                max_memory = {}
                if torch.cuda.is_available():
                    # Get number of GPUs to use (limited by config and available)
                    n_gpus = min(self.config.training.num_gpus, torch.cuda.device_count())
                    for i in range(n_gpus):
                        max_memory[i] = self.config.training.max_gpu_memory
                # Include CPU memory
                max_memory["cpu"] = self.config.training.max_cpu_memory
                print(f"Using memory configuration from config: {max_memory}")
            else:
                # Use default memory settings
                if torch.cuda.is_available():
                    # Reserve 4GB less than total memory for each GPU if available
                    n_gpus = torch.cuda.device_count()
                    max_memory = {}
                    for i in range(n_gpus):
                        gpu_total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        max_memory[i] = f"{int(gpu_total_mem - 4)}GiB"
                    # Reserve 30GB for CPU
                    max_memory["cpu"] = "30GiB"
                else:
                    # CPU only
                    max_memory = {"cpu": "30GiB"}
        
        print(f"Loading model with device map strategy: {device_map}")
        print(f"Max memory configuration: {max_memory}")
        
        # Initialize empty model
        with init_empty_weights():
            empty_model = model_cls.from_pretrained(model_name, **model_kwargs)
            # Tie weights to prevent warning and ensure proper device mapping
            if 'tie_weights' in dir(empty_model) and callable(empty_model.tie_weights):
                empty_model.tie_weights()
        
        # Infer device map and load checkpoint
        if device_map == "auto":
            device_map = infer_auto_device_map(
                empty_model, 
                max_memory=max_memory,
                no_split_module_classes=model_cls._no_split_modules
            )
            print(f"Inferred auto device map: {device_map}")
        
        # Set dtype from config if available
        dtype = None
        if 'mixed_precision' in self.config.training.__dict__:
            precision = self.config.training.mixed_precision
            if precision == "fp16":
                dtype = torch.float16
            elif precision == "bf16":
                dtype = torch.bfloat16
        
        model = load_checkpoint_and_dispatch(
            empty_model,
            model_name,
            device_map=device_map,
            offload_folder=None,
            offload_state_dict=True,
            dtype=dtype
        )
        
        return model
    
    def prepare_model_optimizer(self, 
                              model: nn.Module, 
                              optimizer: torch.optim.Optimizer) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Prepare model and optimizer for memory-optimized training.
        
        Args:
            model: The model to prepare
            optimizer: The optimizer to prepare
            
        Returns:
            tuple: (prepared_model, prepared_optimizer)
        """
        if self.use_accelerate:
            # Prepare with Accelerate
            model, optimizer = self.accelerator.prepare(model, optimizer)
            return model, optimizer
        else:
            # No distributed setup - return as is
            return model, optimizer
    
    def prepare_dataloaders(self, *dataloaders):
        """
        Prepare dataloaders for distributed training.
        
        Args:
            *dataloaders: Dataloaders to prepare
            
        Returns:
            prepared_dataloaders: List of prepared dataloaders
        """
        if self.use_accelerate:
            return self.accelerator.prepare(*dataloaders)
        else:
            return dataloaders
    
    def prepare_models_optimizers(self,
                                actor: nn.Module,
                                actor_optimizer: torch.optim.Optimizer,
                                harm_critic: nn.Module,
                                harm_critic_optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Prepare all agent models and optimizers for memory-optimized training.
        
        Args:
            actor: Actor model
            actor_optimizer: Actor optimizer
            harm_critic: Harm critic model
            harm_critic_optimizer: Harm critic optimizer
            
        Returns:
            Dictionary with prepared models and optimizers
        """
        if self.use_accelerate:
            # Prepare with Accelerate
            actor, actor_optimizer, harm_critic, harm_critic_optimizer = self.accelerator.prepare(
                actor, actor_optimizer, harm_critic, harm_critic_optimizer
            )
            
            result = {
                "actor": actor,
                "actor_optimizer": actor_optimizer,
                "harm_critic": harm_critic,
                "harm_critic_optimizer": harm_critic_optimizer,
            }
            
            return result
        else:
            # No distributed setup - return as is
            return {
                "actor": actor,
                "actor_optimizer": actor_optimizer,
                "harm_critic": harm_critic,
                "harm_critic_optimizer": harm_critic_optimizer,
            }
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass with memory optimization.
        
        Args:
            loss: Loss to backpropagate
        """
        if self.use_accelerate:
            # Use Accelerate's backward
            self.accelerator.backward(loss)
        else:
            # Standard backward
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step with memory optimization.
        
        Args:
            optimizer: Optimizer to step
        """
        if self.training_config.max_grad_norm > 0:
            # Apply gradient clipping if configured
            if self.use_accelerate:
                self.accelerator.clip_grad_norm_(
                    parameters=optimizer.param_groups[0]['params'], 
                    max_norm=self.training_config.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    parameters=optimizer.param_groups[0]['params'], 
                    max_norm=self.training_config.max_grad_norm
                )
                
        # Perform the optimizer step
        optimizer.step()
    
    def gather_from_all_processes(self, data: Any) -> Any:
        """
        Gather data from all processes.
        
        Args:
            data: Data to gather
            
        Returns:
            Gathered data
        """
        if self.use_accelerate and self.world_size > 1:
            # Use Accelerate's gather
            return gather_object(data)
        else:
            # No gathering needed
            return data
    
    def save_checkpoint(self, 
                     checkpoint_dir: str, 
                     actor: nn.Module,
                     harm_critic: nn.Module,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save checkpoint with memory optimization awareness.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            actor: Actor model
            harm_critic: Harm critic model
            metadata: Additional metadata to save
        """
        # Only save from the main process
        if not self.is_main_process():
            return
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Extract the raw models from potential wrappers
        if self.use_accelerate:
            actor = self.accelerator.unwrap_model(actor)
            harm_critic = self.accelerator.unwrap_model(harm_critic)
        
        # Save models
        actor.save_pretrained(os.path.join(checkpoint_dir, "actor"))
        harm_critic.save_pretrained(os.path.join(checkpoint_dir, "harm_critic"))
        
        # Save metadata if provided
        if metadata:
            with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, 
                     checkpoint_dir: str, 
                     actor: nn.Module,
                     harm_critic: nn.Module) -> Dict[str, Any]:
        """
        Load checkpoint with memory optimization awareness.
        
        Args:
            checkpoint_dir: Directory to load checkpoint from
            actor: Actor model to load weights into
            harm_critic: Harm critic model to load weights into
            
        Returns:
            Dictionary with metadata from checkpoint
        """
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Check if model directories exist
        actor_dir = os.path.join(checkpoint_dir, "actor")
        harm_critic_dir = os.path.join(checkpoint_dir, "harm_critic")
        
        if not os.path.exists(actor_dir) or not os.path.exists(harm_critic_dir):
            raise FileNotFoundError(f"Model files not found in checkpoint directory: {checkpoint_dir}")
        
        # Extract the raw models from potential wrappers
        if self.use_accelerate:
            actor = self.accelerator.unwrap_model(actor)
            harm_critic = self.accelerator.unwrap_model(harm_critic)
        
        # Load models
        actor.load_adapter(actor_dir)
        harm_critic.load_adapter(harm_critic_dir)
        
        # Load metadata if it exists
        metadata = {}
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        print(f"Checkpoint loaded from {checkpoint_dir}")
        return metadata


def calculate_optimal_device_map(model_name: str, max_gpu_memory: Optional[str] = None, 
                              num_gpus: int = 2) -> Dict[str, str]:
    """
    Calculate an optimal device map for a model given memory constraints.
    
    Args:
        model_name: The name/path of the model
        max_gpu_memory: Maximum GPU memory to use (e.g., "10GiB" per GPU)
        num_gpus: Number of GPUs to spread the model across (max 2)
    
    Returns:
        A device map dictionary for use with load_checkpoint_and_dispatch
    """
    
    # Default max memory if not provided
    if max_gpu_memory is None:
        if torch.cuda.is_available():
            # By default, leave 2GB free on each GPU
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_gpu_memory = f"{int(total_memory - 2)}GiB"
        else:
            max_gpu_memory = "0GiB"  # CPU only
    
    # Limit to actual available GPUs
    actual_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    
    # Create memory map
    max_memory = {}
    for i in range(actual_gpus):
        max_memory[i] = max_gpu_memory
    
    # Always include CPU for potential offloading
    max_memory["cpu"] = "30GiB"
    
    # Import needed modules
    from transformers import AutoConfig, AutoModelForCausalLM
    
    # Get the model configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Initialize empty weights to calculate device map
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config)
        # Tie weights to prevent warning and ensure proper device mapping
        if 'tie_weights' in dir(empty_model) and callable(empty_model.tie_weights):
            empty_model.tie_weights()
    
    # Calculate auto device map
    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=AutoModelForCausalLM._no_split_modules
    )
    
    print(f"Calculated optimal device map for {model_name}:")
    print(f"Max memory configuration: {max_memory}")
    print(f"Resulting device map: {device_map}")
    
    return device_map 