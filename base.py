from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import os
import gc
import warnings
import traceback
class BaseModelSingleton:
    """
    A singleton class for managing transformers models with LoRA adapter support.
    This provides centralized management of model resources across components.
    """
    _instances = {}
    
    @classmethod
    def get_instance(cls, model_name, hf_token=None, keep_base_on_cpu=False):
        """
        Get or create the singleton instance
        
        Args:
            model_name: Name of the Hugging Face model to load
            hf_token: Hugging Face token for accessing gated models
            
        Returns:
            BaseModelSingleton instance
        """
        if cls._instances.get(model_name) is None:
            cls._instances[model_name] = BaseModelSingleton(model_name, hf_token, keep_base_on_cpu)
        elif cls._instances[model_name].model_name != model_name:
            print(f"Warning: Requested model {model_name} but singleton already initialized with {cls._instances[model_name].model_name}")
        return cls._instances[model_name]
    
    def __init__(self, model_name, hf_token=None, keep_base_on_cpu=False):
        """
        Initialize the model singleton. Should not be called directly.
        Use get_instance() instead.
        
        Args:
            model_name: Name of the Hugging Face model to load
            hf_token: Hugging Face token for accessing gated models
        """
        # Only create a new instance if none exists
        if BaseModelSingleton._instances.get(model_name) is not None:
            return

        self.model_name = model_name
        self.hf_token = hf_token
        self.base_model = None
        self.tokenizer = None
        self.peft_model = None
        self.available_adapters = {}
        self.active_adapter = None
        self.is_consolidated = False
        
        # GPU/CPU management
        self.cuda_available = torch.cuda.is_available()
        self.keep_base_on_cpu = keep_base_on_cpu  # Can be set to True to optimize memory usage
        self.initialized = False
        self.devices = {}  # Maps component names to devices
        
        # Initialize the base model and tokenizer
        self.initialize()
        
    def initialize(self):
        """Initialize the base model and tokenizer"""
        if self.initialized:
            return self.base_model, self.tokenizer
            
        print(f"Initializing BaseModelSingleton with model: {self.model_name}")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token=self.hf_token,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "<|pad|>"
        
        # Load model configuration with settings for more efficient memory usage
        config = AutoConfig.from_pretrained(self.model_name, token=self.hf_token)
        config.output_hidden_states = True
        config.use_cache = False
        
        # Determine initial device for model loading
        initial_device = 'cpu' #if self.keep_base_on_cpu else ('cuda' if self.cuda_available else 'cpu')
        print(f"Loading base model on {initial_device}")
        
        # Load the model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            token=self.hf_token,
            device_map=initial_device
        )
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} gB")
        print(f"Model loaded successfully on {initial_device}")
        self.initialized = True
        return self.base_model, self.tokenizer
    
    def add_adapter(self, adapter_name, peft_config=None, adapter_path=None):
        """
        Add a LoRA adapter to the model
        
        Args:
            adapter_name: Name for the adapter
            peft_config: PeftConfig object for the adapter
            adapter_path: Path to load existing adapter weights
            
        Returns:
            bool: Success status
        """
        if adapter_name in self.available_adapters:
            print(f"Adapter {adapter_name} already exists")
            return True
            
        # Ensure base model is initialized
        if not self.initialized:
            self.initialize()
            
        try:
            # If we don't have a PEFT model yet, create one
            if self.peft_model is None:
                # For first adapter being added
                if adapter_path and os.path.exists(adapter_path):
                    print(f"Loading adapter from {adapter_path}")
                    # Load existing adapter
                    self.peft_model = PeftModel.from_pretrained(
                        self.base_model,
                        adapter_path,
                        adapter_name=adapter_name
                    )
                elif peft_config:
                    print(f"Creating new adapter {adapter_name} with config")
                    # Create new adapter with config
                    self.peft_model = get_peft_model(
                        self.base_model,
                        peft_config,
                        adapter_name=adapter_name
                    )
                else:
                    print("Error: Either adapter_path or peft_config must be provided")
                    return False
                
                self.active_adapter = adapter_name
                
            else:
                # For subsequent adapters being added to existing PEFT model
                if adapter_path and os.path.exists(adapter_path):
                    print(f"Loading adapter {adapter_name} from {adapter_path}")
                    # Load adapter weights into existing PEFT model
                    self.peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
                elif peft_config:
                    print(f"Adding new adapter {adapter_name} to existing PEFT model")
                    # Add new adapter config to existing PEFT model
                    self.peft_model.add_adapter(adapter_name, peft_config)
                else:
                    print("Error: Either adapter_path or peft_config must be provided")
                    return False
            
            # Record the new adapter
            self.available_adapters[adapter_name] = True
            
            # Print adapters after addition
            print(f"Available adapters after addition: {list(self.available_adapters.keys())}")
            print(f"Active adapter: {self.active_adapter}")
            
            return True
            
        except Exception as e:
            print(f"Error adding adapter {adapter_name}: {str(e)}")
            traceback.print_exc()
            return False
    
    def switch_adapter(self, adapter_name):
        """
        Switch to a different LoRA adapter
        
        Args:
            adapter_name: Name of the adapter to activate
            
        Returns:
            bool: Success status
        """
        if not self.peft_model:
            print("No PEFT model initialized")
            return False
            
        if adapter_name not in self.available_adapters:
            print(f"Adapter {adapter_name} not found. Available adapters: {list(self.available_adapters.keys())}")
            return False
            
        if self.active_adapter == adapter_name:
            print(f"Adapter {adapter_name} is already active")
            return True
            
        try:
            # Set the active adapter
            self.peft_model.set_adapter(adapter_name)
            self.active_adapter = adapter_name
            print(f"Switched to adapter: {adapter_name}")
            return True
        except Exception as e:
            print(f"Error switching to adapter {adapter_name}: {str(e)}")
            return False
    
    def merge_base_adapter(self):
        """
        Merge the base adapter into the base model
        
        Returns:
            bool: Success status
        """
        if not self.peft_model:
            print("No PEFT model initialized")
            return False
            
        try:
            # Merge the weights
            print(f"Merging adapter {self.active_adapter} into base model")
            self.peft_model.merge_adapter()
            
            # Reset to base model
            self.base_model = self.peft_model.get_base_model()
            
            # Clear PEFT model
            self.peft_model = None
            self.active_adapter = None
            self.available_adapters = {}
            
            print("Successfully merged adapter into base model")
            self.is_consolidated = True
            return True
        except Exception as e:
            print(f"Error merging adapter: {str(e)}")
            return False
    
    def to_appropriate_devices(self):
        """
        Move model components to appropriate devices based on current settings
        
        Returns:
            bool: Success status
        """
        if not self.initialized:
            print("Model not initialized")
            return False
            
        try:
            if self.cuda_available:
                if self.peft_model:
                    # If we're using PEFT, only move the active adapter to CUDA
                    if self.keep_base_on_cpu:
                        # Keep base model on CPU but move active adapter to CUDA
                        print("Moving active adapter to CUDA while keeping base model on CPU")
                        self.peft_model.to_mixed_precision(
                            base_model_device="cpu",
                            adapter_name=self.active_adapter,
                            adapter_device="cuda"
                        )
                    else:
                        # Move everything to CUDA
                        print("Moving entire PEFT model (base + adapters) to CUDA")
                        self.peft_model.to("cuda")
                else:
                    # Just a regular model, move it to CUDA
                    print("Moving base model to CUDA")
                    self.base_model.to("cuda")
            else:
                print("CUDA not available, keeping all models on CPU")
                
            return True
        except Exception as e:
            print(f"Error moving model to appropriate devices: {str(e)}")
            return False
    
    def to(self, device):
        """
        Selectively move components to the specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', etc.)
            
        Returns:
            bool: Success status
        """
        if not self.initialized:
            print("Model not initialized")
            return False
            
        try:
            if self.peft_model:
                self.peft_model.to(device)
            else:
                self.base_model.to(device)
            return True
        except Exception as e:
            print(f"Error moving model to {device}: {str(e)}")
            return False
    
    def unload(self):
        """
        Unload the model and free memory
        
        Returns:
            bool: Success status
        """
        try:
            if self.peft_model:
                del self.peft_model
                self.peft_model = None
            
            if self.base_model:
                del self.base_model
                self.base_model = None
                
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
                
            # Force garbage collection
            gc.collect()
            
            if self.cuda_available:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
            self.initialized = False
            print("Model unloaded successfully")
            return True
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
            return False
