"""
Utility functions for the RL agent framework.

This module provides common utility functions for:
- Device management and tensor operations
- Memory optimization techniques
- Common tensor operations
- Tokenization helpers
"""

import torch
import gc
import warnings
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from contextlib import contextmanager
import torch.nn as nn

# ===== Device Management Functions =====

def get_device() -> str:
    """
    Get the most appropriate device for computation.
    
    Returns:
        str: Device name ('cuda', 'cuda:0', 'cpu', etc.)
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"

def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        dict: Memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    return {
        "cuda_available": True,
        "allocated": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 3)
    }

def print_memory_stats(message: str = "") -> None:
    """
    Print memory usage information.
    
    Args:
        message: Optional message prefix
    """
    stats = get_memory_stats()
    if stats["cuda_available"]:
        print(f"{message} - CUDA Memory: {stats['allocated']:.2f}GB allocated, "
              f"{stats['max_allocated']:.2f}GB max, {stats['reserved']:.2f}GB reserved")
    else:
        print(f"{message} - CUDA not available")

def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def to_device(tensor: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """
    Move tensor to specified device, or the default device if none specified.
    
    Args:
        tensor: Input tensor
        device: Target device or None for default
        
    Returns:
        Tensor on the target device
    """
    if device is None:
        device = get_device()
    
    # Handle special case for "accelerate_managed" device
    if device == "accelerate_managed":
        # Find the appropriate device for the first layer of the model
        device = get_device()  # Default to standard device
        
    # Only move if needed
    if tensor.device.type != device.split(':')[0]:
        tensor = tensor.to(device)
    return tensor

def detect_first_layer_device(model: nn.Module) -> str:
    """
    Detect the device of the first layer of a model, useful for models with Accelerate device maps.
    
    Args:
        model: PyTorch model, potentially with a device map
        
    Returns:
        Device name for the first layer
    """
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Get the first module's device from the device map
        for module_name in model.hf_device_map:
            if module_name.endswith('.0') or module_name == 'model.embed_tokens':
                device = model.hf_device_map[module_name]
                print(f"First layer device detected as '{device}' from Accelerate device map")
                return device
                
        # If no specific entry found, look for the first device in the map
        for device in model.hf_device_map.values():
            return device  # Return first device found
        
    # Fallback to checking the parameters
    try:
        first_param = next(model.parameters())
        return str(first_param.device)
    except (StopIteration, RuntimeError):
        # No parameters or other error
        return get_device()

# ===== Tensor Operation Utilities =====

def batch_to_device(batch: Dict[str, Any], device: Optional[str] = None, model: Optional[nn.Module] = None) -> Dict[str, Any]:
    """
    Move an entire batch of tensors to the specified device.
    
    Args:
        batch: Dictionary of tensors or other values
        device: Target device
        model: Optional model to detect device from (for Accelerate-managed models)
        
    Returns:
        Dict with tensors moved to device
    """
    if device is None:
        if model is not None and hasattr(model, 'hf_device_map') and model.hf_device_map:
            # For models with Accelerate device map, get the device of the first layer
            device = detect_first_layer_device(model)
        else:
            device = get_device()
        
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = batch_to_device(v, device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            result[k] = [x.to(device) for x in v]
        else:
            result[k] = v
    return result

@contextmanager
def tensor_operation(tensor: torch.Tensor, 
                    device: Optional[str] = None, 
                    requires_grad: bool = False):
    """
    Context manager for tensor operations, handles device and gradient.
    
    Args:
        tensor: Input tensor
        device: Target device
        requires_grad: Whether to track gradients
    
    Yields:
        tensor on target device with appropriate gradient setting
    """
    original_device = tensor.device
    original_requires_grad = tensor.requires_grad
    
    if device is None:
        device = get_device()
    
    # Move tensor to target device
    tensor = tensor.to(device)
    
    # Set requires_grad
    tensor.requires_grad_(requires_grad)
    
    yield tensor
    
    # Restore original state if needed
    if original_requires_grad != requires_grad:
        tensor.requires_grad_(original_requires_grad)
    
    # Only move back if requested and needed
    if tensor.device != original_device:
        tensor = tensor.to(original_device)

def is_tensor_on_device(tensor: torch.Tensor, device: str) -> bool:
    """
    Check if a tensor is already on the specified device.
    
    Args:
        tensor: The tensor to check
        device: The target device
        
    Returns:
        bool: Whether the tensor is on the device
    """
    if ':' in device:
        device_type, device_idx = device.split(':')
        return tensor.device.type == device_type and tensor.device.index == int(device_idx)
    else:
        return tensor.device.type == device

# ===== Optimization Utilities =====

@contextmanager
def optimize_memory_usage():
    """
    Context manager for optimized memory usage during operations.
    
    This enables several memory optimization techniques like garbage collection,
    disabling gradient calculation, and clearing GPU cache after execution.
    """
    # Run garbage collection before operation
    gc.collect()
    
    # Get initial memory stats
    initial_stats = get_memory_stats()
    
    # Disable gradient calculation temporarily if not needed
    with torch.no_grad():
        yield
    
    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Get final memory stats
    final_stats = get_memory_stats()
    
    # Log memory usage if increased significantly
    if (final_stats.get("allocated", 0) > initial_stats.get("allocated", 0) + 0.1 and
        torch.cuda.is_available()):
        warnings.warn(
            f"Memory usage increased by "
            f"{final_stats['allocated'] - initial_stats['allocated']:.2f}GB "
            f"during operation"
        )
        # We don't need to raise an exception here, but we should warn the user
        # Continue execution as this is just a warning about memory usage

# ===== Tokenization Helpers =====

def batch_encode(texts: List[str], 
                tokenizer: Any, 
                device: Optional[str] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
    """
    Batch encode texts using the tokenizer and move to specified device.
    
    Args:
        texts: List of texts to encode
        tokenizer: Tokenizer instance
        device: Target device
        **kwargs: Additional arguments for the tokenizer
        
    Returns:
        Dict containing input_ids and attention_mask on the target device
    """
    if device is None:
        device = get_device()
    
    # Ensure tokenizer has proper padding settings
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Add pad token if needed
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        **kwargs
    )
    
    return {k: v.to(device) for k, v in encoding.items()} 

# ===== Caching Utilities =====

class TensorCache:
    """
    Cache for tensors to avoid redundant computations.
    
    This class provides a cache for tensor operations with configurable size limits
    and eviction policies.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the tensor cache.
        
        Args:
            max_size: Maximum number of tensors to cache
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.access_count = {}
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get a tensor from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tensor if found, None otherwise
        """
        if key in self.cache:
            self.hits += 1
            self.access_count[key] += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """
        Add a tensor to the cache.
        
        Args:
            key: Cache key
            tensor: Tensor to cache
        """
        # If cache is full, evict least recently used item
        if len(self.cache) >= self.max_size:
            # Find key with lowest access count
            min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[min_key]
            del self.access_count[min_key]
        
        # Add to cache
        self.cache[key] = tensor
        self.access_count[key] = 1
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "size": len(self.cache),
            "max_size": self.max_size
        }

# Global tensor cache instance
_tensor_cache = TensorCache()

def get_tensor_cache() -> TensorCache:
    """
    Get the global tensor cache instance.
    
    Returns:
        Global TensorCache instance
    """
    return _tensor_cache

def cached_computation(key_prefix: str, compute_fn: Callable, *args, **kwargs) -> Any:
    """
    Perform a computation with caching.
    
    Args:
        key_prefix: Prefix for cache key
        compute_fn: Function to compute the result
        *args: Arguments for compute_fn
        **kwargs: Keyword arguments for compute_fn
        
    Returns:
        Result of computation
    """
    # Generate cache key from prefix and args
    arg_str = "_".join([str(hash(arg)) for arg in args])
    kwarg_str = "_".join([f"{k}_{hash(v)}" for k, v in sorted(kwargs.items())])
    key = f"{key_prefix}_{arg_str}_{kwarg_str}"
    
    # Check cache
    cache = get_tensor_cache()
    result = cache.get(key)
    
    if result is None:
        # Compute and cache
        result = compute_fn(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            cache.put(key, result)
    
    return result

# ===== Batch Processing Utilities =====

def split_batch(batch: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    """
    Split a batch into smaller batches.
    
    Args:
        batch: Input batch tensor
        batch_size: Size of smaller batches
        
    Returns:
        List of smaller batch tensors
    """
    return torch.split(batch, batch_size)

def dynamic_batch_size(tensors: List[torch.Tensor], 
                      target_memory_usage: float = 0.8,
                      min_batch_size: int = 1) -> int:
    """
    Calculate a dynamic batch size based on available GPU memory.
    
    Args:
        tensors: List of tensors that will be used in the batch
        target_memory_usage: Target memory usage as a fraction of available memory
        min_batch_size: Minimum batch size to return
        
    Returns:
        Calculated batch size
    """
    if not torch.cuda.is_available():
        return min_batch_size
    
    # Calculate tensor sizes
    tensor_memory = sum(tensor.element_size() * tensor.nelement() for tensor in tensors)
    
    # Get available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    available_memory = total_memory - allocated_memory
    
    # Calculate batch size
    memory_per_batch = tensor_memory
    if memory_per_batch == 0:
        return 32  # Default if we can't determine tensor memory
    
    target_memory = available_memory * target_memory_usage
    batch_size = max(min_batch_size, int(target_memory / memory_per_batch))
    
    return batch_size

def adaptive_batch_size_for_attention(batch_size: int, seq_len: int, num_layers: int = 32, 
                                     num_heads: int = 32, target_memory_gb: float = 20.0) -> int:
    """
    Calculate an adaptive batch size for attention computations to avoid OOM.
    
    Args:
        batch_size: Original batch size
        seq_len: Sequence length
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        target_memory_gb: Target GPU memory usage in GB
        
    Returns:
        Adjusted batch size that should fit in memory
    """
    if not torch.cuda.is_available():
        return batch_size
    
    # Estimate memory for attention computation
    # Attention weights: [num_layers, batch_size, num_heads, seq_len, seq_len]
    attention_memory_per_sample = num_layers * num_heads * seq_len * seq_len * 2  # 2 bytes for fp16
    
    # Add memory for hidden states and other tensors (rough estimate)
    hidden_state_memory_per_sample = seq_len * 4096 * 2  # assuming 4096 hidden dim, fp16
    total_memory_per_sample = attention_memory_per_sample + hidden_state_memory_per_sample
    
    # Convert target memory to bytes
    target_memory_bytes = target_memory_gb * 1024**3
    
    # Calculate max batch size that fits in target memory
    max_batch_size = max(1, int(target_memory_bytes / total_memory_per_sample))
    
    return min(batch_size, max_batch_size)

def get_optimal_chunk_size(total_batch_size: int, available_memory_gb: float, 
                          seq_len: int, model_size_gb: float = 8.0) -> int:
    """
    Calculate optimal chunk size for processing large batches.
    
    Args:
        total_batch_size: Total batch size to process
        available_memory_gb: Available GPU memory in GB
        seq_len: Sequence length
        model_size_gb: Estimated model size in GB
        
    Returns:
        Optimal chunk size for processing
    """
    # Reserve memory for model and other overhead
    usable_memory_gb = available_memory_gb - model_size_gb - 2.0  # 2GB overhead
    
    if usable_memory_gb <= 0:
        return 1
    
    # Rough estimate: each sample needs seq_len * hidden_dim * 2 bytes (fp16)
    # Plus attention computation memory
    memory_per_sample_gb = (seq_len * 4096 * 2 + seq_len * seq_len * 32 * 2) / (1024**3)
    
    max_chunk_size = max(1, int(usable_memory_gb / memory_per_sample_gb))
    
    return min(total_batch_size, max_chunk_size)

def monitor_memory_usage(func):
    """
    Decorator to monitor GPU memory usage of a function.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function that prints memory usage
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            print(f"Memory usage for {func.__name__}:")
            print(f"  Initial: {initial_memory / 1024**3:.2f} GB")
            print(f"  Final: {final_memory / 1024**3:.2f} GB")
            print(f"  Peak: {peak_memory / 1024**3:.2f} GB")
            print(f"  Difference: {(final_memory - initial_memory) / 1024**3:.2f} GB")
            
        return result
    return wrapper

# ===== Forward Pass Registry =====

class ForwardPassRegistry:
    """
    Registry to track and avoid redundant forward passes.
    
    This class keeps track of model inputs and outputs to avoid
    recomputing the same forward pass multiple times.
    """
    
    def __init__(self, max_size: int = 50):
        """
        Initialize the forward pass registry.
        
        Args:
            max_size: Maximum registry size
        """
        self.registry = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.last_access = {}
    
    def get(self, model_id: str, input_hash: str) -> Optional[Any]:
        """
        Get a cached forward pass result.
        
        Args:
            model_id: Identifier for the model
            input_hash: Hash of the input
            
        Returns:
            Cached output if found, None otherwise
        """
        key = f"{model_id}_{input_hash}"
        if key in self.registry:
            self.hits += 1
            self.last_access[key] = time.time()
            return self.registry[key]
        
        self.misses += 1
        return None
    
    def register(self, model_id: str, input_hash: str, output: Any) -> None:
        """
        Register a forward pass result.
        
        Args:
            model_id: Identifier for the model
            input_hash: Hash of the input
            output: Forward pass output
        """
        key = f"{model_id}_{input_hash}"
        
        # Evict oldest entry if registry is full
        if len(self.registry) >= self.max_size:
            oldest_key = min(self.last_access.items(), key=lambda x: x[1])[0]
            del self.registry[oldest_key]
            del self.last_access[oldest_key]
        
        self.registry[key] = output
        self.last_access[key] = time.time()
    
    def clear(self) -> None:
        """Clear the registry."""
        self.registry.clear()
        self.last_access.clear()

# Global forward pass registry
_forward_pass_registry = ForwardPassRegistry()

def get_forward_pass_registry() -> ForwardPassRegistry:
    """
    Get the global forward pass registry.
    
    Returns:
        Global ForwardPassRegistry instance
    """
    return _forward_pass_registry

def hash_tensor_input(tensor: torch.Tensor) -> str:
    """
    Create a hash for a tensor input.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Hash string
    """
    # Use a simple hash based on tensor shape and a sample of values
    shape_hash = "_".join(str(x) for x in tensor.shape)
    
    # Sample up to 10 values from the tensor for the hash
    if tensor.numel() > 0:
        sample = tensor.reshape(-1)[:min(10, tensor.numel())].detach().cpu().numpy()
        value_hash = "_".join(f"{x:.4f}" for x in sample)
    else:
        value_hash = "empty"
    
    return f"{shape_hash}_{value_hash}"

def registered_forward_pass(model: torch.nn.Module, 
                           *args, 
                           model_id: Optional[str] = None,
                           **kwargs) -> Any:
    """
    Perform a forward pass with registration to avoid redundancy.
    
    Args:
        model: PyTorch model
        *args: Arguments for model forward pass
        model_id: Identifier for the model (optional)
        **kwargs: Keyword arguments for model forward pass
        
    Returns:
        Model output
    """
    # Generate model_id if not provided
    if model_id is None:
        model_id = str(id(model))
    
    # Generate input hash
    input_hashes = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            input_hashes.append(hash_tensor_input(arg))
        else:
            input_hashes.append(str(hash(arg)))
    
    for k, v in sorted(kwargs.items()):
        if isinstance(v, torch.Tensor):
            input_hashes.append(f"{k}_{hash_tensor_input(v)}")
        else:
            input_hashes.append(f"{k}_{hash(v)}")
    
    input_hash = "_".join(input_hashes)
    
    # Check registry
    registry = get_forward_pass_registry()
    output = registry.get(model_id, input_hash)
    
    if output is None:
        # Perform forward pass
        output = model(*args, **kwargs)
        
        # Register output
        registry.register(model_id, input_hash, output)
    
    return output 

def create_lora_config(lora_r: int = 8, 
                      lora_alpha: int = 16, 
                      lora_dropout: float = 0.05,
                      target_modules: List[str] = None,
                      target_layers: List[int] = None,
                      bias: str = "none") -> Any:
    """
    Create a LoRA configuration that targets specific attention layers.
    
    Args:
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout parameter
        target_modules: List of module types to target (e.g., ["q_proj", "v_proj"])
        target_layers: List of layer indices to target (e.g., [0, 1, 2])
        bias: LoRA bias parameter ("none", "all", or "lora_only")
    
    Returns:
        LoraConfig object from the PEFT library
    """
    from peft import LoraConfig
    
    # Set defaults if not provided
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    if target_layers is None:
        target_layers = [0, 1, 2]  # Default to first 3 layers
    
    if target_layers and (-1 in target_layers):
        target_layers = None

    # Helper to map module name to its sub-block prefix
    def _subblock_prefix(module_name: str) -> str:
        attn_set = {"q_proj", "k_proj", "v_proj", "o_proj"}
        mlp_set = {"gate_proj", "up_proj", "down_proj"}
        if module_name in attn_set:
            return "self_attn"
        if module_name in mlp_set:
            return "mlp"
        # Default to self_attn but warn the user
        print(f"[create_lora_config] Warning: Unknown module '{module_name}', defaulting to self_attn prefix")
        return "self_attn"

    # Generate full module names with layer indices if target_layers is *still* provided
    if target_layers:
        full_target_modules = []
        for layer in target_layers:
            for module in target_modules:
                prefix = _subblock_prefix(module)
                full_target_modules.append(f"model.layers.{layer}.{prefix}.{module}")
        print(f"Targeting LoRA at specific layers: {target_layers}")
        print(f"Full target modules: {full_target_modules}")
    else:
        # Just use module names directly if no specific layers (all layers)
        full_target_modules = target_modules
        print(f"Targeting LoRA at all layers with modules: {target_modules}")
    
    # Create the LoRA configuration
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=full_target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM"  # For language modeling tasks
    )
    
    return config 

def format_tokenizer_input(text_input) -> list:
    """
    Format input for tokenizer to ensure it's a list of lists of strings.
    
    Args:
        text_input: Input text that could be:
            - str: Single string
            - List[str]: List of strings
            - List[List[str]]: List of lists of strings
            
    Returns:
        List[List[str]]: Properly formatted input
        
    Example:
        >>> format_tokenizer_input("hello")
        [["hello"]]
        >>> format_tokenizer_input(["hello", "world"])
        [["hello"], ["world"]]
        >>> format_tokenizer_input([["hello"], ["world"]])
        [["hello"], ["world"]]
    """
    import warnings
    
    # Case 1: string input
    if isinstance(text_input, str):
        return [[text_input]]
        
    # Case 2: list input
    if isinstance(text_input, list):
        # Empty list
        if not text_input:
            return []
            
        # Case 2a: List[List[str]]
        if isinstance(text_input[0], list):
            # Validate all elements are lists of strings
            for sublist in text_input:
                if not isinstance(sublist, list) or not all(isinstance(s, str) for s in sublist):
                    raise ValueError("Input must be a list of lists of strings")
            return text_input
            
        # Case 2b: List[str]
        if isinstance(text_input[0], str):
            warnings.warn("Input is a list of strings. Wrapping each string in a list for proper formatting.")
            return [[s] for s in text_input]
            
    raise ValueError("Input must be a string, list of strings, or list of lists of strings")


def safe_tokenizer_call(tokenizer, text_input, **kwargs):
    """
    Safely call tokenizer with proper input formatting.
    
    Args:
        tokenizer: Tokenizer instance
        text_input: Input text (str, List[str], or List[List[str]])
        **kwargs: Additional arguments to pass to tokenizer
        
    Returns:
        Tokenizer output with properly formatted input
    """
    formatted_input = format_tokenizer_input(text_input)
    
    # If input is empty, return empty tensor with correct structure
    if not formatted_input:
        import torch
        return type('TokenizerOutput', (), {
            'input_ids': torch.zeros((0, 0), dtype=torch.long),
            'attention_mask': torch.zeros((0, 0), dtype=torch.long)
        })()
    
    # Flatten the list of lists for tokenizer input
    flattened_input = [" ".join(sublist) for sublist in formatted_input]
    return tokenizer(flattened_input, **kwargs) 

def pretty_print(list_of_lists, msg=None):
    """
    Pretty print a list of lists.
    """
    import pprint
    if msg is not None:
        print("="*80 + "\n" + msg + "\n" + "="*80)
    for i, sublist in enumerate(list_of_lists):
        print(f"element {i}:")
        pprint.pprint(sublist)
    print("="*80)

def post_tokenizer_injection(tokenizer_encoding, injections:List[torch.Tensor]):
    """
    Post-tokenizer-processing to inject tokens.
    """
    input_ids_list = []
    attention_mask_list = []
    for i, injection in enumerate(injections):
        # Ensure injection tensor is on the same device as tokenizer_encoding tensors
        if injection.device != tokenizer_encoding.input_ids.device:
            injection = injection.to(tokenizer_encoding.input_ids.device)

        injected_prompt_i = torch.cat([tokenizer_encoding.input_ids[i], injection], dim=0)
        attention_mask_i = torch.cat([tokenizer_encoding.attention_mask[i], torch.ones_like(injection)], dim=0)
        input_ids_list.append(injected_prompt_i)
        attention_mask_list.append(attention_mask_i)
    return torch.stack(input_ids_list, dim=0), torch.stack(attention_mask_list, dim=0)

def build_chat(messages, user_num):
    chat = []    
    for i, msg in enumerate(messages):            
        if isinstance(msg, list):
            # Handle empty list case
            if len(msg) == 0:
                continue  # Skip empty conversation histories
            msg = msg[0]
        assert isinstance(msg, str), f"Message must be a string, got {type(msg)}"
        if msg[-4:] == 'user':
            msg = msg[:-4]
        elif msg[-9:] == 'assistant':
            msg = msg[:-9]
        if i % 2 == user_num:
            chat.append({'role': 'user', 'content': msg})
        else:
            chat.append({'role': 'assistant', 'content': msg})
    
    # Ensure we always have at least one message to avoid template errors
    if not chat:
        chat = [{'role': 'user', 'content': 'Hello'}]
    
    return chat

# ===== Plotting Utilities =====

def generate_training_plots(csv_path: str, output_dir: str) -> None:
    """Generate line-plots for each metric column in a training CSV.

    The CSV is expected to have one row per episode, with the first column
    named ``episode``.  Every other column is treated as a scalar metric and
    plotted against the episode index.  A separate ``<metric>.png`` file is
    written to *output_dir* for each metric.

    Args:
        csv_path: Path to the CSV file that was written during training.
        output_dir: Directory where the PNG files will be saved.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'episode' not in df.columns:
        raise ValueError("CSV must contain an 'episode' column for plotting")

    metric_cols = [c for c in df.columns if c != 'episode']
    if len(metric_cols) == 0:
        raise ValueError("No metric columns found to plot in CSV")

    for col in metric_cols:
        plt.figure()
        plt.plot(df['episode'], df[col])
        plt.xlabel('Episode')
        plt.ylabel(col.replace('_', ' ').title())
        plt.title(f'{col} over Training')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(output_dir, f"{col}.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved plot: {fname}")

def create_activation_clamping_hooks(model, clamp_range: float = 100.0, layernorm_range: float = 50.0):
    """
    Create hooks to clamp activations and prevent numerical instability in models.
    This is especially useful for 4-bit quantized models with Flash Attention.
    
    Args:
        model: The model to add hooks to
        clamp_range: Clamp range for activation hooks (-range to +range)
        layernorm_range: Clamp range for LayerNorm hooks (-range to +range)
        
    Returns:
        list: List of hooks that should be removed after forward pass
    """
    hooks = []
    
    def clamp_activation_hook(module, input, output):
        """Hook to clamp activations and prevent overflow/underflow."""
        if isinstance(output, torch.Tensor):
            # Clamp activations to prevent extreme values
            output_clamped = torch.clamp(output, min=-clamp_range, max=clamp_range)
            # Replace any non-finite values with zeros
            output_clamped = torch.where(torch.isfinite(output_clamped), output_clamped, torch.zeros_like(output_clamped))
            return output_clamped
        return output
    
    def protect_layernorm_hook(module, input, output):
        """Special hook for LayerNorm/RMSNorm to prevent variance explosion."""
        if isinstance(output, torch.Tensor):
            # LayerNorm/RMSNorm are very sensitive to quantization
            # Clamp more aggressively to prevent variance issues
            output_clamped = torch.clamp(output, min=-layernorm_range, max=layernorm_range)
            # Replace any non-finite values with zeros
            output_clamped = torch.where(torch.isfinite(output_clamped), output_clamped, torch.zeros_like(output_clamped))
            return output_clamped
        return output
    
    # Register hooks on key layers that might produce unstable activations
    for name, module in model.named_modules():
        if any(layer_type in name for layer_type in ['self_attn', 'mlp', 'down_proj', 'up_proj', 'gate_proj']):
            hook = module.register_forward_hook(clamp_activation_hook)
            hooks.append(hook)
        
        # Special protection for LayerNorm and RMSNorm layers
        if any(norm_type in name for norm_type in ['layernorm', 'rmsnorm', 'ln_', 'norm']):
            hook = module.register_forward_hook(protect_layernorm_hook)
            hooks.append(hook)
    
    return hooks


def apply_activation_clamping(model, forward_func, *args, enable_hooks: bool = True, clamp_range: float = 30.0, layernorm_range: float = 10.0, **kwargs):
    """
    Apply activation clamping hooks during a forward pass and automatically clean up.
    
    Args:
        model: The model to apply hooks to
        forward_func: The forward function to call (e.g., model.__call__)
        *args: Arguments to pass to the forward function
        enable_hooks: Whether to apply hooks (if False, just calls forward_func)
        clamp_range: Clamp range for activation hooks (-range to +range)
        layernorm_range: Clamp range for LayerNorm hooks (-range to +range)
        **kwargs: Additional arguments to pass to the forward function
        
    Returns:
        The output of the forward function
    """
    if not enable_hooks:
        return forward_func(*args, **kwargs)
    
    hooks = create_activation_clamping_hooks(model)
    try:
        return forward_func(*args, **kwargs)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()


def attach_permanent_activation_clamping(model, enable_hooks: bool = True, clamp_range: float = 30.0, layernorm_range: float = 10.0):
    """
    Permanently attach activation clamping hooks to a model.
    This is more efficient for research codebases where we want consistent protection.
    
    Args:
        model: The model to attach hooks to
        enable_hooks: Whether to attach hooks (if False, returns empty list)
        clamp_range: Clamp range for activation hooks (-range to +range)
        layernorm_range: Clamp range for LayerNorm hooks (-range to +range)
        
    Returns:
        list: List of hooks (for reference, but they stay attached)
    """
    hooks = []
    
    if not enable_hooks:
        print("Forward hooks disabled by configuration")
        return hooks
    
    def clamp_activation_hook(module, input, output):
        """Hook to clamp activations and prevent overflow/underflow."""
        if isinstance(output, torch.Tensor):
            # Clamp activations to prevent extreme values
            output_clamped = torch.clamp(output, min=-clamp_range, max=clamp_range)
            # Replace any non-finite values with zeros
            output_clamped = torch.where(torch.isfinite(output_clamped), output_clamped, torch.zeros_like(output_clamped))
            return output_clamped
        return output
    
    def protect_layernorm_hook(module, input, output):
        """Special hook for LayerNorm/RMSNorm to prevent variance explosion."""
        if isinstance(output, torch.Tensor):
            # LayerNorm/RMSNorm are very sensitive to quantization
            # Clamp more aggressively to prevent variance issues
            output_clamped = torch.clamp(output, min=-layernorm_range, max=layernorm_range)
            # Replace any non-finite values with zeros
            output_clamped = torch.where(torch.isfinite(output_clamped), output_clamped, torch.zeros_like(output_clamped))
            return output_clamped
        return output
    
    # Register hooks on key layers that might produce unstable activations
    for name, module in model.named_modules():
        if any(layer_type in name for layer_type in ['self_attn', 'mlp', 'down_proj', 'up_proj', 'gate_proj']):
            hook = module.register_forward_hook(clamp_activation_hook)
            hooks.append(hook)
        
        # Special protection for LayerNorm and RMSNorm layers
        if any(norm_type in name for norm_type in ['layernorm', 'rmsnorm', 'ln_', 'norm']):
            hook = module.register_forward_hook(protect_layernorm_hook)
            hooks.append(hook)
    
    
    print(f"Attached {len(hooks)} permanent activation clamping hooks to model (clamp_range: ±{clamp_range}, layernorm_range: ±{layernorm_range})")
    return hooks