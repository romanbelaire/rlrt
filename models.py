from asyncio.log import logger
import gc
from transformers import LlamaForCausalLM, AutoConfig, AutoModelForCausalLM, FbgemmFp8Config
from transformers.tokenization_utils_base import BatchEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from contextlib import contextmanager
import psutil
import copy
from helpers import log_generations
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, PeftConfig
import os


class TransformerPolicy(nn.Module):
    def __init__(self, hf_checkpoint, tokenizer, model=None, token=None, is_victim=False, 
                 use_lora=False, lora_config=None, adapter_name=None):
        """
        Initialize the TransformerPolicy model.
        
        Args:
            hf_checkpoint (str): The HuggingFace checkpoint to load the model from.
            tokenizer (AutoTokenizer): The tokenizer to use.
            model (AutoModelForCausalLM, optional): The model to use. Defaults to None.
            token (str, optional): The token to use for loading the model. Defaults to None.
            is_victim (bool, optional): Whether this is a victim model. Defaults to False.
            use_lora (bool, optional): Whether to use LoRA. Defaults to False.
            lora_config (dict, optional): The LoRA configuration. Defaults to None.
            adapter_name (str, optional): The name of the adapter to use. Defaults to None.
        """
        super().__init__()
        self.print_memory('TransformerPolicy model initializing')
        init_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.tokenizer = tokenizer        
        # Load model from shared instance if provided, otherwise from HF
        if model is not None:
            self.model = model
            print("Using provided model instance")
        else:
            cache_dir = os.getenv('HF_HOME')
            config = AutoConfig.from_pretrained(hf_checkpoint, token=token, low_cpu_mem_usage='70' in hf_checkpoint, cache_dir=cache_dir)
            config.return_dict = True
            
            print(f"Checkpoint for TransformerPolicy model: {hf_checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_checkpoint,
                config=config,
                token=token,
                low_cpu_mem_usage='70' in hf_checkpoint,
                cache_dir=cache_dir
            )
            
        # Safely resize token embeddings if needed
        try:
            if len(self.tokenizer) != self.model.config.vocab_size:
                print(f"Resizing token embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
        except KeyError as e:
            if "attribute 'weight' already exists" in str(e):
                print("Weights are already tied, skipping resize_token_embeddings")
            else:
                # Re-raise if it's not the specific error we're handling
                raise
                
        # Set up LoRA if specified
        if use_lora:            
            if lora_config is None:
                # Default LoRA configuration if none provided
                lora_config = {
                    'r': 8,
                    'lora_alpha': 16,
                    'lora_dropout': 0.05,
                    'target_modules': ['lm_head'],
                    'bias': 'none'
                }
            
            # Create LoRA config
            print("Initializing with LoraConfig: {}".format(lora_config))
            if isinstance(lora_config, dict):
                lora_config = LoraConfig(
                    r=lora_config.get('r', 8),
                    lora_alpha=lora_config.get('lora_alpha', 16),
                    lora_dropout=lora_config.get('lora_dropout', 0.05),
                    target_modules=lora_config.get('target_modules', ['lm_head']),
                    bias=lora_config.get('bias', 'none'),
                    task_type="CAUSAL_LM"
                )
            
            print(f"Applying LoRA with config: {lora_config}")
            
            # Get PEFT model with LoRA adapters
            try:
                self.model = get_peft_model(self.model, lora_config)
                if adapter_name:
                    self.model.active_adapter = adapter_name
            except Exception as e:
                print(f"Error applying LoRA: {str(e)}")
                print("Continuing with base model without LoRA")
        
        # Initialize the final head for the model
        self.final = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.output_layer
        self.force_stop_strings = ['?','assistant', 'Assistant']
        self.force_stop_tokens = torch.tensor([self.tokenizer(s).input_ids[-1] for s in self.force_stop_strings], dtype=torch.int16).to('cuda')

        print("Checkpoint for TransformerPolicy model: " + hf_checkpoint)
        self.model_dim = self.model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        assert hasattr(self.model, 'lm_head'), "Model must have an lm_head module"
        # Enable gradient checkpointing with use_reentrant=True
        #self.model.gradient_checkpointing_enable()
        #self.model.enable_input_require_grads()

        # First set up the reference LM head layer
        self.lm_head_clone = copy.deepcopy(self.final)
        self.lm_head_clone.requires_grad = False
        
        # Store LoRA configuration
        self.use_lora = use_lora
        self.lora_config = lora_config
        self.adapter_name = adapter_name
        
        # Apply LoRA if specified
        if use_lora and lora_config is not None:
            self.apply_lora_adapters(lora_config, adapter_name)
        elif use_lora:
            # If use_lora is True but no config is provided, we're likely loading adapters later
            # Still need to freeze all parameters to save memory
            print("LoRA enabled but no config provided. Freezing all parameters for later adapter loading.")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Freeze body EXCEPT final layer and embeddings
            for name, param in self.model.model.named_parameters():
                if 'lm_head' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # Double check trainable components
            for param in self.final.parameters():
                param.requires_grad = True
            for param in self.model.get_input_embeddings().parameters():
                param.requires_grad = True

        #load onto gpu
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(self.model.module)
            self.model.__setattr__('get_input_embeddings', self.model.module.get_input_embeddings)
        self.to('cuda')
        self.print_memory("After model load")

        self.model_memory = torch.cuda.memory_allocated() - init_memory

        assert self.tokenizer.padding_side == 'left', "Padding side must be left"
        
    def apply_lora_adapters(self, lora_config, adapter_name=None):
        """
        Apply LoRA adapters to the model.
        
        Args:
            lora_config: LoraConfig object or dictionary with LoRA parameters
            adapter_name: Name for the adapter (used for saving/loading)
        """
        print(f"Applying LoRA adapters with config: {lora_config}, adapter_name: {adapter_name}")
        # Convert dict to LoraConfig if needed
        if isinstance(lora_config, dict):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.get('r', 8),
                lora_alpha=lora_config.get('lora_alpha', 16),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                target_modules=lora_config.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj", "lm_head"]),
                bias=lora_config.get('bias', "none"),
            )
        
        # Set the adapter name in the LoraConfig if provided
        if adapter_name is not None:
            self.adapter_name = adapter_name
            # Explicitly set the adapter_name in the config
            if hasattr(lora_config, 'adapter_name'):
                lora_config.adapter_name = adapter_name
        
        # Apply LoRA adapters
        self.model = get_peft_model(self.model, lora_config, adapter_name=adapter_name)
        print(f"Peft model retrieved with adapter name: {adapter_name}")
        # Explicitly set active adapter if adapter_name is provided
        if adapter_name is not None and hasattr(self.model, 'set_adapter'):
            self.model.set_adapter(adapter_name)
            print(f"Set active adapter to: {adapter_name}")
            
        # Ensure ONLY LoRA parameters require gradients
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Update use_lora flag
        self.use_lora = True
        
    def save_lora_adapters(self, output_dir):
        """
        Save LoRA adapters to disk.
        
        Args:
            output_dir: Directory to save adapters to
            
        Returns:
            bool: Whether the adapters were successfully saved
        """
        if not self.use_lora or not hasattr(self.model, 'save_pretrained'):
            print("Model does not have LoRA adapters to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the adapters
            self.model.save_pretrained(output_dir, selected_adapters=[self.adapter_name])
            
            # Verify the adapter was saved correctly by checking for the adapter_config.json file
            adapter_config_path = os.path.join(output_dir, self.adapter_name, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print(f"Successfully saved LoRA adapters to {output_dir}")
                return True
            else:
                print(f"ERROR: Failed to save adapter to {output_dir} - adapter_config.json not found")
                return False
        except Exception as e:
            print(f"ERROR: Failed to save adapter to {output_dir}: {str(e)}")
            return False

    def load_lora_adapters(self, adapter_path):
        """
        Load LoRA adapters from disk.
        
        Args:
            adapter_path: Path to the saved adapters
            
        Returns:
            bool: Whether the adapter was successfully loaded
        """
        if not os.path.exists(adapter_path):
            print(f"Adapter path {adapter_path} does not exist")
            return False
            
        try:
            # Load the configuration
            config = PeftConfig.from_pretrained(adapter_path)
            print(f"Loading LoRA adapters from {adapter_path} with config: {config}")
            
            # Store adapter name from config or from adapter_name attribute
            adapter_name = getattr(config, 'adapter_name', None)
            if adapter_name is None and hasattr(self, 'adapter_name'):
                adapter_name = self.adapter_name
                
            # Load the adapters
            print(f"adapter name: {adapter_name}, adapter path: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
            # Explicitly set active adapter if adapter_name is available
            if adapter_name is not None and hasattr(self.model, 'set_adapter'):
                try:
                    # Check if this adapter name is in the available adapters
                    if adapter_name in self.model.peft_config:
                        self.model.set_adapter(adapter_name)
                        print(f"Set active adapter to: {adapter_name}")
                    else:
                        print(f"Warning: Adapter {adapter_name} not found in model config, using default adapter")
                except Exception as e:
                    print(f"Warning: Could not set adapter {adapter_name}: {e}")
            
            # Ensure ONLY LoRA parameters require gradients
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            # Update use_lora flag
            self.use_lora = True
            
            # Debug: print available adapters in the model
            if hasattr(self.model, 'peft_config'):
                print(f"Available adapters after loading: {list(self.model.peft_config.keys())}")
            
            return True
        except Exception as e:
            print(f"Error loading LoRA adapters: {e}")
            return False

    def check_layer_grads(self):
        """
        Check if there are any trainable parameters in the model.
        If not, print diagnostic information and attempt to fix the issue.
        
        Raises:
            AssertionError: If no trainable parameters are found after attempted fixes
        """
        # Get all parameters
        all_params = list(self.parameters())
        
        # If no parameters are found at all
        if len(all_params) == 0:
            print("ERROR: No parameters returned by self.parameters()")
            
            # Attempt to diagnose the issue
            if self.use_lora:
                print("\nDiagnosing LoRA adapter issues:")
                
                # Check if we have an active adapter
                if hasattr(self.model, 'active_adapter'):
                    print(f"Active adapter: {self.model.active_adapter}")
                else:
                    print("No active adapter found on the model")
                
                # Check available adapters
                if hasattr(self.model, 'peft_config'):
                    print(f"Available adapters: {list(self.model.peft_config.keys())}")
                else:
                    print("No peft_config found on the model")
                
                # Count parameters by requires_grad status
                grad_params = 0
                non_grad_params = 0
                lora_params = 0
                lora_grad_params = 0
                
                for name, param in self.model.named_parameters():
                    if "lora" in name.lower():
                        lora_params += 1
                        if param.requires_grad:
                            lora_grad_params += 1
                    
                    if param.requires_grad:
                        grad_params += 1
                    else:
                        non_grad_params += 1
                
                print(f"Total params: {grad_params + non_grad_params}")
                print(f"Params with requires_grad=True: {grad_params}")
                print(f"LoRA params: {lora_params}")
                print(f"LoRA params with requires_grad=True: {lora_grad_params}")
                
                # Try to fix by marking parameters as trainable
                if lora_params > 0 and lora_grad_params == 0:
                    print("\nAttempting to fix by marking LoRA parameters as trainable...")
                    
                    for name, param in self.model.named_parameters():
                        if "lora" in name.lower():
                            param.requires_grad = True
                    
                    # Retry check
                    return self.check_layer_grads()
        
        # Try to validate that we have trainable parameters
        trainable_params = [p for p in all_params if p.requires_grad]
        if not trainable_params:
            print("ERROR: Parameters found by self.parameters() but none require gradients")
            # Final attempt to fix the issue
            if self.use_lora:
                # Force enable gradients for the active adapter
                for name, param in self.model.named_parameters():
                    print(f"name: {name}")
                    if self.model._current_adapter in name:
                        print(f"Force enabling gradients for {name}")
                        param.requires_grad = True
            
        # Final check
        assert any(p.requires_grad for p in self.parameters()), "No trainable parameters found after attempted fixes"

    def parameters(self):
        """
        Return the trainable parameters of the model.
        For LoRA models, this will be the LoRA adapter parameters.
        For standard models, this will be the final layer and embedding parameters.
        
        Returns:
            list: List of trainable parameters
        """
        params = []
        if self.use_lora:
            # For PEFT models, find all parameters that require gradients
            # and have "lora" in their name
            lora_params_found = 0
            
            # More thorough search for LoRA parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Check various patterns that could indicate LoRA parameters
                    if ("lora" in name.lower() or 
                        "adapter" in name.lower() or 
                        "peft" in name.lower()):
                        params.append(param)
                        lora_params_found += 1
            
            # Print debug information about found parameters
            if lora_params_found == 0:
                print("WARNING: No LoRA parameters found with requires_grad=True!")
                # Check if active adapter exists but parameters aren't marked trainable
                if hasattr(self.model, 'active_adapter') and self.model.active_adapter is not None:
                    print(f"Active adapter is {self.model.active_adapter}, but no trainable LoRA parameters found.")
                    
                    # Check if parameters exist that match the adapter name but aren't marked trainable
                    adapter_params_found = 0
                    for name, param in self.model.named_parameters():
                        if self.model.active_adapter in name:
                            adapter_params_found += 1
                            # Force set requires_grad to True as a fix
                            param.requires_grad = True
                            params.append(param)
                    
                    if adapter_params_found > 0:
                        print(f"Found and marked {adapter_params_found} parameters for active adapter as trainable.")
                    else:
                        print("No parameters found that match the active adapter name.")
            else:
                print(f"Found {lora_params_found} trainable LoRA parameters.")
        else:
            # Standard model: use final layer and embeddings
            for param in self.final.parameters():
                params.append(param)
            for param in self.model.get_input_embeddings().parameters():
                if isinstance(self.model, nn.DataParallel):
                    name = self.model.module.name_or_path   
                else:
                    name = getattr(self.model, 'name_or_path', '')
                if (not name or 'llama' in str(name).lower()) and param not in params:
                    params.append(param)
        
        # Double-check that we found parameters and they require gradients
        if len(params) == 0:
            print("WARNING: No trainable parameters found!")
            
            # As a last resort, if we have an active adapter, find and enable its parameters
            if self.use_lora and hasattr(self.model, 'active_adapter') and self.model.active_adapter is not None:
                print(f"Forcing parameters for active adapter {self.model.active_adapter} to be trainable")
                pattern = self.model.active_adapter
                for name, param in self.model.named_parameters():
                    if pattern in name:
                        param.requires_grad = True
                        params.append(param)
                        
                print(f"Added {len(params)} parameters from active adapter {pattern}")
        
        return params

    def forward(self, input_ids=None, attention_mask=None, labels=None, with_head=False, pad_token_id=None):        
        try:            
            model = self.model if with_head else self.model.model
                        
            # Explicitly get and attach embeddings to gradient graph
            input_embeds = self.model.get_input_embeddings()(input_ids.to('cuda'))
            input_embeds.requires_grad_(torch.is_grad_enabled())  # Explicitly enable gradients
                
            # Forward pass with embeddings
            outputs = model(
                inputs_embeds=input_embeds, 
                attention_mask=attention_mask, 
                output_hidden_states=True, 
                return_dict=True, 
                return_logits=True,
                pad_token_id=pad_token_id
            )
            
            # Ensure hidden states have gradients enabled
            outputs.hidden_states = tuple(h.requires_grad_(torch.is_grad_enabled()) for h in outputs.hidden_states)

            # Ensure output tensors are on CUDA for further processing
            # Move the last hidden state to CUDA for the lm_head computation
            last_hidden = outputs.hidden_states[-1].to('cuda')
            
            # compute cross entropy loss using frozen model head as labels
            ce_shape = torch.Size([outputs.logits.shape[0] * outputs.logits.shape[1], outputs.logits.shape[-1]])
            reference_logits = self.lm_head_clone(last_hidden)
            reference_probs = F.softmax(reference_logits, dim=-1)
            reference_probs = reference_probs.view(ce_shape)


                
            ce_loss = F.cross_entropy(outputs.logits.view(ce_shape), reference_probs)
            outputs.loss = ce_loss

            if hasattr(outputs, 'logits'):
                outputs.logits.requires_grad_(torch.is_grad_enabled())
                    
            return outputs
                
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
        
    def generate_best_of_n(self, input_ids, attention_mask, reward_fn, n=5, max_length=100, pad_token_id=None):
        """
        Generate text using best-of-n sampling.
        """
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
            
        batch_size = input_ids.shape[0]
        device = input_ids.device
        outputs_all = []
        
        # Generate n samples for each input
        for i in range(n):
            input_ids_i = input_ids.expand(n, -1)
            attention_mask_i = attention_mask.expand(n, -1)
            outputs = self.generate(input_ids=input_ids_i, attention_mask=attention_mask_i, max_new_tokens=max_length, pad_token_id=pad_token_id)
            rewards = reward_fn(outputs)
            best_outputs = outputs[torch.argmax(rewards)]
            outputs_all.append(best_outputs)
            
        return outputs_all

    def sample_top_k(self, next_token_logits, k=10, adv_offset=0):
        # Get the top K logits and their indices
        
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
        window_start = adv_offset + 1
        window_end = window_start + k
        top_k_logits = sorted_logits[:, window_start:window_end]
        top_k_indices = sorted_indices[:, window_start:window_end]

        # Convert logits to probabilities
        top_k_probs = torch.softmax(top_k_logits, dim=-1)

        # Sample from the top K probabilities
        next_token_ids = torch.multinomial(top_k_probs, num_samples=1)

        # Get the actual token IDs from the top K indices
        next_token_ids = top_k_indices.gather(-1, next_token_ids).squeeze()
        if next_token_ids.dim() == 0:
            next_token_ids = next_token_ids.unsqueeze(0)

        return next_token_ids

    def sample_nucleus(self, next_token_logits, p=0.9, temperature=1.0):
        """
        Sample from the nucleus (top-p) of the distribution.
        
        Args:
            next_token_logits: Logits for next token prediction (batch_size, vocab_size)
            p: Nucleus probability threshold (default: 0.9)
            temperature: Temperature for sampling (default: 1.0)
            
        Returns:
            next_token_ids: Sampled token ids (batch_size)
        """
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        # Convert logits to probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for probabilities to keep (those within the nucleus)
        nucleus_mask = cumulative_probs < p
        
        # Add the first token to the nucleus to ensure at least one token is selected
        nucleus_mask = torch.cat([
            torch.ones_like(nucleus_mask[:, :1]), 
            nucleus_mask[:, 1:]
        ], dim=-1)
        
        # Determine the number of tokens in the nucleus for each sample in the batch
        nucleus_size = torch.sum(nucleus_mask, dim=1, keepdim=True)
        
        # Get the sorted logits and probabilities for the nucleus
        nucleus_logits = torch.zeros_like(sorted_probs)
        nucleus_logits.masked_fill_(~nucleus_mask, float('-inf'))
        nucleus_logits = nucleus_logits + sorted_probs
        
        # Sample from the nucleus probabilities
        # We re-normalize the probabilities of only the nucleus tokens
        nucleus_probs = F.softmax(nucleus_logits, dim=-1)
        next_token_ids_index = torch.multinomial(nucleus_probs, num_samples=1)
        
        # Get the actual token IDs
        next_token_ids = sorted_indices.gather(-1, next_token_ids_index).squeeze(-1)
        
        # Ensure proper dimensionality
        if next_token_ids.dim() == 0:
            next_token_ids = next_token_ids.unsqueeze(0)
            
        return next_token_ids
    
    def beam_search(self, input_ids, attention_mask, beam_size=5, max_length=100, length_penalty=1.0, pad_token_id=None):
        """
        Generate text using beam search.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing input token IDs
            attention_mask: Tensor of shape (batch_size, seq_len) containing attention mask
            beam_size: Number of beams to use (default: 5)
            max_length: Maximum length of generated sequence (default: 100)
            length_penalty: Length penalty for beam search (default: 1.0)
            pad_token_id: Padding token ID (default: eos_token_id)
            
        Returns:
            tuple: (best_sequences, best_scores, best_log_probs, best_lm_losses) where:
                - best_sequences: List of tensors containing the best sequences
                - best_scores: List of scores for the best sequences
                - best_log_probs: List of log probabilities for each token in the best sequences
                - best_lm_losses: List of language model losses for each generation step
        """
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
            
        batch_size = input_ids.shape[0]
        device = input_ids.device
        vocab_size = self.model.config.vocab_size
        
        # Initialize beam state
        input_ids = input_ids.repeat_interleave(beam_size, dim=0)  # (batch_size * beam_size, seq_len)
        attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)  # (batch_size * beam_size, seq_len)
        
        # Create batch beam indices for tracking beams across batches
        batch_beam_indices = (
            torch.arange(batch_size, device=device).repeat_interleave(beam_size) * beam_size
        ).unsqueeze(1)  # (batch_size * beam_size, 1)
        
        # Initialize sequence scores
        sequence_scores = torch.zeros((batch_size * beam_size, 1), device=device)  # (batch_size * beam_size, 1)
        
        # Flag to track which sequences have finished
        done = [False for _ in range(batch_size)]
        
        # Store best sequences and scores for each batch item
        best_sequences = [None] * batch_size
        best_scores = [-float('inf')] * batch_size
        
        # Store log probabilities and LM losses
        # We'll maintain these for all beams and select the best ones at the end
        all_log_probs = [[] for _ in range(batch_size * beam_size)]
        all_lm_losses = [[] for _ in range(batch_size * beam_size)]
        best_log_probs = [None] * batch_size
        best_lm_losses = [None] * batch_size
        
        # Generate tokens
        for step in range(max_length):
            # Forward pass through the model
            with torch.no_grad():
                outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, with_head=True)
                
                # Get next token logits (only the last token)
                next_token_logits = outputs.logits[:, -1, :]  # (batch_size * beam_size, vocab_size)
                
                # Compute log probabilities
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * beam_size, vocab_size)
                
                # Add sequence scores to token scores
                next_token_scores = next_token_log_probs + sequence_scores  # (batch_size * beam_size, vocab_size)
                
                # Reshape for beam search: (batch_size, beam_size * vocab_size)
                next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)
                
                # Select top-k*k candidates
                # We flatten the vocab dimension to select top-k*k across all beams
                topk_scores, topk_indices = torch.topk(next_token_scores, k=beam_size, dim=1)
                
                # Get beam indices and token indices
                beam_indices = topk_indices // vocab_size  # (batch_size, beam_size)
                token_indices = topk_indices % vocab_size  # (batch_size, beam_size)
                
                # Store the current LM loss for each beam
                current_lm_loss = outputs.loss.unsqueeze(0).expand(batch_size * beam_size).clone()
                
                # For each batch item, update the log probs and LM losses for each beam
                for batch_idx in range(batch_size):
                    for beam_idx in range(beam_size):
                        # Get the global index
                        global_beam_idx = batch_idx * beam_size + beam_idx
                        # Get the source beam for this new candidate
                        source_beam_idx = batch_idx * beam_size + beam_indices[batch_idx, beam_idx]
                        
                        # Clone the log probs and LM losses from the source beam
                        new_log_probs = all_log_probs[source_beam_idx].copy()
                        new_lm_losses = all_lm_losses[source_beam_idx].copy()
                        
                        # Get the token ID and its log probability
                        token_id = token_indices[batch_idx, beam_idx]
                        token_log_prob = next_token_log_probs[source_beam_idx, token_id].item()
                        
                        # Add the new token's log probability and LM loss
                        new_log_probs.append(token_log_prob)
                        new_lm_losses.append(current_lm_loss[source_beam_idx].item())
                        
                        # Update the log probs and LM losses for this beam
                        all_log_probs[global_beam_idx] = new_log_probs
                        all_lm_losses[global_beam_idx] = new_lm_losses
                
                # Update sequence scores
                sequence_scores = topk_scores.view(batch_size * beam_size, 1)
                
                # Create new beam indices for batched selection
                batch_beam_indices = (
                    torch.arange(batch_size, device=device).unsqueeze(1) * beam_size + beam_indices
                ).view(-1)  # (batch_size * beam_size)
                
                # Select beams from previous step
                input_ids = input_ids[batch_beam_indices]
                attention_mask = attention_mask[batch_beam_indices]
                
                # Append new tokens to input_ids
                input_ids = torch.cat([input_ids, token_indices.view(-1, 1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(token_indices.view(-1, 1))], dim=-1)
                
                # Check for completed sequences (EOS token generated)
                eos_mask = token_indices.view(-1) == self.tokenizer.eos_token_id
                
                # Update done list and store completed sequences
                if eos_mask.any():
                    for idx in range(batch_size * beam_size):
                        batch_idx = idx // beam_size
                        
                        if eos_mask[idx] and not done[batch_idx]:
                            # Apply length penalty to score
                            seq_len = attention_mask[idx].sum().item()
                            score = sequence_scores[idx].item() / (seq_len ** length_penalty)
                            
                            if score > best_scores[batch_idx]:
                                best_scores[batch_idx] = score
                                best_sequences[batch_idx] = input_ids[idx].clone()
                                best_log_probs[batch_idx] = all_log_probs[idx]
                                best_lm_losses[batch_idx] = all_lm_losses[idx]
                            
                    # Check if all batches are done
                    if all(score > -float('inf') for score in best_scores):
                        all_done = True
                        for batch_idx in range(batch_size):
                            if not done[batch_idx]:
                                # Check if the best current beam score is better than our best stored score
                                beam_start = batch_idx * beam_size
                                beam_end = (batch_idx + 1) * beam_size
                                current_max_score = sequence_scores[beam_start:beam_end].max().item()
                                
                                # If current score can't beat our best, mark as done
                                if current_max_score < best_scores[batch_idx]:
                                    done[batch_idx] = True
                                else:
                                    all_done = False
                                    
                        if all_done:
                            break
        
        # For batches without completed sequences, use the best from the final step
        for batch_idx in range(batch_size):
            if best_sequences[batch_idx] is None:
                beam_start = batch_idx * beam_size
                beam_indices = range(beam_start, beam_start + beam_size)
                best_beam = sequence_scores[beam_indices].argmax().item()
                best_idx = beam_start + best_beam
                best_sequences[batch_idx] = input_ids[best_idx].clone()
                seq_len = attention_mask[best_idx].sum().item()
                best_scores[batch_idx] = sequence_scores[best_idx].item() / (seq_len ** length_penalty)
                best_log_probs[batch_idx] = all_log_probs[best_idx]
                best_lm_losses[batch_idx] = all_lm_losses[best_idx]
        
        # Convert log probs and LM losses to tensors
        for batch_idx in range(batch_size):
            # Convert log probs to tensor
            if isinstance(best_log_probs[batch_idx], list):
                best_log_probs[batch_idx] = torch.tensor(best_log_probs[batch_idx], device=device)
            
            # Convert LM losses to tensor
            if isinstance(best_lm_losses[batch_idx], list):
                best_lm_losses[batch_idx] = torch.tensor(best_lm_losses[batch_idx], device=device)
        
        return best_sequences, best_scores, best_log_probs, best_lm_losses
    
    def generate_beamsearch(self, input_ids, attention_mask, max_new_tokens=100, pad_token_id=None, 
                            beam_size=None, length_penalty=1.0):
        """
        Generates sequences of tokens from the input batch using beam search.
        Args:
            - input_ids (torch.Tensor): The input IDs of the batch.
            - attention_mask (torch.Tensor): The attention mask of the batch.
            - max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 100.
            - pad_token_id (int, optional): The token ID to use for padding. Defaults to None.
            - beam_size (int, optional): Number of beams for beam search. If None, sampling_method will be used. Defaults to None.
            - length_penalty (float, optional): Length penalty for beam search. Defaults to 1.0.
        Returns:
            output_seq: list of str
            output_tokens: torch.Tensor
            output_log_probs: torch.Tensor
            lm_losses: torch.Tensor
        """
        # Use beam search generation
        input_length = input_ids.shape[-1]
        output_ids = input_ids.to('cuda')
        best_sequences, best_scores, best_log_probs, best_lm_losses = self.beam_search(
            output_ids, 
            attention_mask,
            beam_size=beam_size,
            max_length=max_new_tokens,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id
        )
        
        # Process results
        output_tokens = [seq[input_length:] for seq in best_sequences]
        output_seq = [self.tokenizer.decode(tokens, skip_special_tokens=False) for tokens in output_tokens]
        
        # Use the log probabilities and LM losses returned from beam search
        batch_size = len(output_seq)
        
        # Create a padded tensor for the log probabilities
        max_log_probs_len = max(log_probs.size(0) for log_probs in best_log_probs)
        output_log_probs = torch.zeros(batch_size, max_log_probs_len).to('cuda')
        for i, log_probs in enumerate(best_log_probs):
            output_log_probs[i, :log_probs.size(0)] = log_probs
        
        # Create a padded tensor for the LM losses if they vary in length
        # Otherwise, just stack them
        if all(losses.size(0) == best_lm_losses[0].size(0) for losses in best_lm_losses):
            lm_losses = torch.stack(best_lm_losses)
        else:
            max_losses_len = max(losses.size(0) for losses in best_lm_losses)
            lm_losses = torch.zeros(batch_size, max_losses_len).to('cuda')
            for i, losses in enumerate(best_lm_losses):
                lm_losses[i, :losses.size(0)] = losses
            # Convert to 1D tensor to match expected format
            lm_losses = lm_losses.mean(dim=0)
        
        # Convert output_tokens list to tensor with padding
        max_len = max(tokens.size(0) for tokens in output_tokens)
        output_tensor = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long).to('cuda')
        for i, tokens in enumerate(output_tokens):
            output_tensor[i, :tokens.size(0)] = tokens
        
        return output_seq, output_tensor, output_log_probs, lm_losses
    
    def generate(self, batch, max_new_tokens=100, k=3, offset=0, enforce_single_question=False, require_grad=True, pad_token_id=None, 
                sampling_method='top_k', p=0.9, temperature=1.0, beam_size=None, length_penalty=1.0):
        """
        Generates sequences of tokens from the input batch using the model.
        Args:
            - batch (list or str): The input batch of text to be tokenized and processed.
            - max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 100.
            - k (int, optional): The number of top-k tokens to sample from. Defaults to 3.
            - offset (int, optional): The offset to apply to the top-k sampling window. Defaults to 0.
            - enforce_single_question (bool, optional): Whether to enforce that the generated sequences contain only one statement. Defaults to False.
            - require_grad (bool, optional): Whether to require gradients. Defaults to True.
            - pad_token_id (int, optional): The token ID to use for padding. Defaults to None.
            - sampling_method (str, optional): The sampling method to use. Options: 'top_k', 'nucleus', 'beam_search'. Defaults to 'top_k'.
            - p (float, optional): The nucleus probability threshold. Only used with nucleus sampling. Defaults to 0.9.
            - temperature (float, optional): Temperature for sampling. Affects both top-k and nucleus sampling. Defaults to 1.0.
            - beam_size (int, optional): Number of beams for beam search. If None, sampling_method will be used. Defaults to None.
            - length_penalty (float, optional): Length penalty for beam search. Defaults to 1.0.
        Returns:
            tuple: A tuple containing:
              - output_seq (list of str): The generated sequences decoded into text.
              - output_tokens (torch.Tensor): The generated token IDs.
              - output_log_probs (torch.Tensor): The log probabilities for each token in the generated sequences.
              - lm_losses (torch.Tensor): The language model losses for each generation step.
        """
        # Assert that none of the inputs are empty strings
        assert self.tokenizer.padding_side == 'left'
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
            
        # Handle input preparation
        if isinstance(batch, BatchEncoding):
            tokens = batch
        else:
            if isinstance(batch, str):
                assert batch.strip() != "", "Input batch cannot be an empty string"
            elif isinstance(batch, list):
                assert all(isinstance(item, str) and item.strip() != "" for item in batch), "Input batch cannot contain empty strings"
            if isinstance(batch, str):
                batch = [batch]
            tokens = self.tokenizer(batch, return_tensors='pt', padding=True, add_special_tokens=(batch[0].startswith(self.tokenizer.bos_token)))
        
        input_ids = tokens.input_ids
        input_length = input_ids.shape[-1]
        attention_mask = torch.where(input_ids == pad_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)).to('cuda')
        
        # If beam search is explicitly requested, use it regardless of sampling_method
        if beam_size is not None and beam_size > 1:
            return self.generate_beamsearch(input_ids, attention_mask, max_new_tokens, pad_token_id, beam_size, length_penalty)
        
        # Standard auto-regressive generation with sampling
        output_ids = input_ids.to('cuda')
        output_log_probs = torch.full(input_ids.shape, fill_value=-10, dtype=torch.float32).to('cuda')
        output_log_probs.requires_grad = True
        lm_losses = []

        for i in range(max_new_tokens):
            # Only enable gradients for final step when require_grad is True
            with torch.set_grad_enabled(require_grad and i == max_new_tokens - 1):
                output = self.forward(input_ids=output_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, with_head=True)
                next_logits = output.logits[:, -1, :]
                
                # Select sampling method
                if sampling_method == 'nucleus':
                    next_token_id = self.sample_nucleus(next_logits, p=p, temperature=temperature)
                else:  # default to top-k
                    next_token_id = self.sample_top_k(next_logits, k=k, adv_offset=offset)
                
                if i > 10:  # early stopping criteria
                    # follow all force_stop_tokens with EOS token
                    if enforce_single_question:
                        cond1 = torch.BoolTensor([i in self.force_stop_tokens.to('cuda') for i in output_ids[:, -1]]).view(next_token_id.shape).to('cuda') 
                        next_token_id = torch.where(cond1, torch.full_like(next_token_id, self.tokenizer.eos_token_id), next_token_id)
                    # follow all EOS and PAD tokens with another PAD token
                    cond1 = output_ids[:, -1] == self.tokenizer.eos_token_id
                    cond2 = output_ids[:, -1] == self.tokenizer.pad_token_id
                    next_token_id = torch.where(torch.logical_or(cond1, cond2), 
                                            torch.full_like(next_token_id, self.tokenizer.pad_token_id), next_token_id)
                
                next_token_id = next_token_id.unsqueeze(-1)
                if len(next_token_id.shape) == 1:  # if batch size is 1 add a dimension
                    next_token_id = next_token_id.unsqueeze(0)
                assert next_token_id.dtype == torch.int64, f"next_token_id has dtype {next_token_id.dtype}"
                softmax_logits = F.log_softmax(next_logits, dim=-1)
                if next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(1)  # Shape: (batch_size, 1)
                next_token_log_prob = torch.gather(softmax_logits, dim=-1, index=next_token_id)
                
                output_ids = torch.cat([output_ids, next_token_id], dim=-1) 
                output_log_probs = torch.cat([output_log_probs, next_token_log_prob], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
                lm_losses.append(output.loss)

        # Return only generated tokens and their log probs
        output_tokens = output_ids[:, input_length:]
        
        output_seq = self.tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=False)
        return output_seq, output_tokens, output_log_probs[:, input_length:].clone(), torch.stack(lm_losses)

    def print_memory(self, message=None):
        print('-' * 80)
        if (message):
            print(message)
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
        print(f"Max memory cached: {torch.cuda.max_memory_reserved() / 1024 ** 2} MB")
        print('-' * 80)

    def print_ram_usage(self):
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"RSS: {mem_info.rss / 1024 ** 2} MB")
            print(f"VMS: {mem_info.vms / 1024 ** 2} MB")
    
    def get_embed_similarity(self, seq1, seq2, requires_grad=True):
        '''Get cosine similarity between two sequences
        seq1: tensor of shape (batch_size, seq_len)
        seq2: tensor of shape (batch_size, seq_len)
        return: cosine similarity between seq1 and seq2
        '''
        if seq1.dim() <= 1:
            seq1 = seq1.unsqueeze(0)
            seq2 = seq2.unsqueeze(0)
        seq1 = seq1.to('cuda')
        seq2 = seq2.to('cuda')
        with torch.set_grad_enabled(requires_grad):
            # Define a function to wrap the forward pass
            embed1 = self.forward(seq1, with_head=True).hidden_states[-1]
            embed2 = self.forward(seq2, with_head=True).hidden_states[-1]
            assert embed1.shape == (seq1.shape[0], seq1.shape[1], self.model_dim), f"embed1 shape: {embed1.shape}, seq1 shape: {seq1.shape}"
            assert embed2.shape == (seq2.shape[0], seq2.shape[1], self.model_dim), f"embed2 shape: {embed2.shape}, seq2 shape: {seq2.shape}"
        
            embed1_norm = embed1 / embed1.norm(dim=1, keepdim=True)
            embed2_norm = embed2 / embed2.norm(dim=1, keepdim=True)
            embed1_mean = embed1_norm.mean(dim=1)
            embed2_mean = embed2_norm.mean(dim=1)
            cosine_similarity = F.cosine_similarity(embed1_mean, embed2_mean, dim=-1)
        return cosine_similarity
    
    def get_log_probs(self, input_ids, output_ids, require_grad=True, sequence_level=False, add_special_tokens=True):
        """Compute log probabilities of output sequence given input sequence.
        
        Processes output tokens one at a time, using ground truth tokens from output_ids
        instead of sampling. This matches how the sequence was actually generated.
        
        Args:
            input_ids (torch.Tensor or str): Input sequence or raw text
            output_ids (torch.Tensor or str): Target output sequence or raw text  
            requires_grad (bool): Whether to maintain gradient information
            sequence_level (bool): If True, sum log probs across sequence
            
        Returns:
            torch.Tensor: Log probabilities for each token (sequence_level=False)
                            or summed log probs (sequence_level=True)
                            Shape: [batch_size, seq_len] or [batch_size]
        """
        original_device = self.model.device
        self.to('cuda')
        if not isinstance(input_ids, torch.Tensor):
            if isinstance(input_ids, list) and isinstance(input_ids[0], str):
                input_ids = self.tokenizer(input_ids, return_tensors='pt', padding=True, add_special_tokens=add_special_tokens).input_ids
            elif isinstance(input_ids, list) and isinstance(input_ids[0], torch.Tensor):
                input_ids = torch.cat(input_ids, dim=0)
            else:
                raise ValueError(f"input_ids must be a string or list of strings, got {type(input_ids)}")
        if not isinstance(output_ids, torch.Tensor):
            if isinstance(output_ids, list) and isinstance(output_ids[0], str):
                output_ids = self.tokenizer(output_ids, return_tensors='pt', padding=True, add_special_tokens=add_special_tokens).input_ids
            elif isinstance(output_ids, list) and isinstance(output_ids[0], torch.Tensor):
                output_ids = torch.cat(output_ids, dim=0)
            else:
                raise ValueError(f"output_ids must be a string or list of strings, got {type(output_ids)}" + (f"of {type(output_ids[0])}" if isinstance(output_ids, list) else ""))
            
        # Initialize tensors
        curr_input = input_ids.to('cuda')
        output_ids = output_ids.to('cuda')
        log_probs = torch.zeros(output_ids.size(0), output_ids.size(1), requires_grad=require_grad).to('cuda')
        attention_mask = torch.where(curr_input == self.tokenizer.pad_token_id, 
                                     torch.zeros_like(curr_input), 
                                     torch.ones_like(curr_input)).to('cuda')
        ce_loss = []
        epsilon = 1e-10  # Small epsilon value

        # Generate token by token
        for i in range(output_ids.size(1)):
            with torch.set_grad_enabled((i == output_ids.size(1) - 1) and require_grad):
                # Get model output for current input
                output = self.forward(curr_input, attention_mask=attention_mask, with_head=True)

                # Ensure logits are on compute_device (CUDA)
                next_token_logits = output.logits[:, -1, :].to('cuda')
                
                # --- Debugging: Check logits before log_softmax ---
                if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                    raise RuntimeError(f"WARNING in get_log_probs (models.py): NaN or Inf detected in logits at step {i}!")
                # -------------------------------------------------
                
                # Get probability of actual next token (ensure on compute_device)
                next_token = output_ids[:, i].to('cuda')
                # Apply log_softmax (more numerically stable)
                log_prob_i = F.log_softmax(next_token_logits, dim=-1).gather(1, next_token.unsqueeze(-1)).squeeze(-1)
                
                # --- Debugging: Check output of log_softmax ---
                if torch.isnan(log_prob_i).any():
                    raise RuntimeError(f"WARNING in get_log_probs (models.py): NaN detected AFTER log_softmax at step {i}!")
                # -------------------------------------------------
                    
                log_probs[:, i] = log_prob_i
                ce_loss.append(output.loss)
                
                # Append actual token to input (on base_model_device)
                next_token_base_device = next_token.to('cuda')
                curr_input = torch.cat([curr_input, next_token_base_device.unsqueeze(-1)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_base_device.unsqueeze(-1))], dim=1)
        
        self.to(original_device)
        # Return sequence-level log probs if requested
        if sequence_level:
            print(f"log_probs: {log_probs.shape}, log_probs.mean(dim=-1): {log_probs.mean(dim=-1).shape}")
            return log_probs.mean(dim=-1), torch.stack(ce_loss)
        return log_probs, torch.stack(ce_loss)

    @contextmanager
    def eval(self):
        '''Set model to evaluation mode'''
        was_training = self.model.training
        try:
            self.model.eval()
            yield
        finally:
            self.model.train(was_training)

    @contextmanager
    def train(self):
        '''Set model to training mode'''
        was_training = self.model.training
        try:
            self.model.train()
            yield
        finally:
            self.model.train(was_training)

    def to(self, device):
        '''Move model to device'''
        self.model.to(device)
        self.lm_head_clone.to(device)

    def print_embedding_info(self):
        """Debug embedding gradients"""
        # Get embedding layer
        embedding = self.model.get_input_embeddings()
        params = self.model.named_parameters()
        print("\n=== Embedding Layer Info ===")
        print(f"Shape: {embedding.weight.shape}")
        print(f"Requires grad: {embedding.weight.requires_grad}")
        
        # Print gradient info if available
        if embedding.weight.grad is not None:
            print("\nGradient Statistics:")
            grad = embedding.weight.grad
            print(f"Gradient shape: {grad.shape}")
            print(f"Mean: {grad.mean().item():.6f}")
            print(f"Std: {grad.std().item():.6f}")
            print(f"Max: {grad.max().item():.6f}")
            print(f"Min: {grad.min().item():.6f}")
            print(f"Num zeros: {(grad == 0).sum().item()}")
        else:
            print("\nNo gradients computed yet")

    def check_embeddings(self):
        """Check if embeddings are being updated"""
        assert self.final.weight.grad is not None, "Final layer weights are not being updated"
        assert self.model.get_input_embeddings().weight.grad is not None, "Embeddings are not being updated"

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Load only trainable parameters (final layer and input embeddings)
        """
        # Create new state dict with only trainable components
        trainable_state = {}
        
        # Map state dict keys to model components
        for key, value in state_dict.items():
            if key.startswith('final.'):
                trainable_state[key] = value
            elif key.startswith('model.embed_tokens.'):
                trainable_state[key] = value
        
        # Load parameters for trainable components
        missing_keys = []
        unexpected_keys = []
        
        # Load final layer parameters
        if hasattr(self, 'final'):
            for key in trainable_state:
                if key.startswith('final.'):
                    try:
                        self.final.load_state_dict({key[6:]: trainable_state[key]})
                    except Exception as e:
                        missing_keys.append(key)
        
        # Load embedding parameters
        if hasattr(self.model, 'get_input_embeddings'):
            embed_layer = self.model.get_input_embeddings()
            for key in trainable_state:
                if key.startswith('model.embed_tokens.'):
                    try:
                        embed_layer.load_state_dict({key[19:]: trainable_state[key]})
                    except Exception as e:
                        missing_keys.append(key)
        
        # Handle strict loading
        if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
            raise RuntimeError(f'Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}')
        
        return self

    def state_dict(self):
        """
        Save only trainable parameters (final layer and input embeddings)
        """
        state_dict = {}
        
        # Save final layer parameters
        if hasattr(self, 'final'):
            final_state = self.final.state_dict()
            for key, value in final_state.items():
                state_dict['final.' + key] = value
        
        # Save embedding parameters
        if hasattr(self.model, 'get_input_embeddings'):
            embed_layer = self.model.get_input_embeddings()
            embed_state = embed_layer.state_dict()
            for key, value in embed_state.items():
                state_dict['model.embed_tokens.' + key] = value
        
        return state_dict

    def freeze_layers(self):
        """Freeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        """Unfreeze all layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_input_embeddings(self):
        """
        Return the input embedding layer of the model.
        Required for PEFT compatibility.
        """
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        """
        Return the output embedding layer of the model.
        Required for PEFT compatibility.
        """
        if hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()
        return None
    
    def resize_token_embeddings(self, new_num_tokens=None):
        """
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Required for PEFT compatibility.
        """
        return self.model.resize_token_embeddings(new_num_tokens)
    
    @property
    def device(self):
        return self.model.device

class ValueTransformer(nn.Module):
    def __init__(self, hf_checkpoint, tokenizer, model=None, token=None, token_level=False,
                 attention_update=True, output_attentions=False, use_attention_adapter=True):
        super(ValueTransformer, self).__init__()

        # Add PEFT compatibility attributes
        self.base_model_prefix = "model"

        # Store whether to use attention adapter
        self.use_attention_adapter = use_attention_adapter
        self.tokenizer = tokenizer
        self.token_level = token_level
        self.attention_update = attention_update # Store this, though freezing is handled externally now

        if model is not None:
            # Use the passed model object
            self.model = model
            logger.info(f"ValueTransformer using pre-loaded base model.")
            # Config should be accessible from the loaded model
            config = self.model.config
        else:
            # Load model if not provided
            logger.info(f"ValueTransformer loading base model: {hf_checkpoint}")
            config = AutoConfig.from_pretrained(hf_checkpoint, token=token, trust_remote_code=True, output_attentions=output_attentions)
            config.output_hidden_states = True
            config.return_dict = True
            config.return_dict_in_generate = True
            config.use_cache = False
            config.output_attentions = True # Ensure attentions are enabled in config

            self.model = AutoModelForCausalLM.from_pretrained(
                hf_checkpoint,
                config=config,
                token=token,
                trust_remote_code=True
                # Device placement is handled by the caller (ValueTrainer)
            )

        # Store config as an attribute to support PEFT compatibility
        self.config = config

        # --- Freeze Base Model Parameters ---
        # This should happen regardless of whether the model was pre-loaded or loaded here
        # We assume the caller (ValueTrainer) handles applying LoRA and unfreezing adapters later
        logger.info("Freezing parameters of the base model within ValueTransformer.")
        for param in self.model.parameters():
            param.requires_grad = False


        # --- Initialize Projection Head / Sequence Pooler ---
        # Initialize on CPU by default; ValueTrainer will move it to the target device
        logger.info(f"Initializing projection head/pooler on CPU (will be moved by Trainer). Token Level: {token_level}")
        if token_level:
            output_size = config.vocab_size
            self.output_projection = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(), # Using ReLU instead of Tanh based on some recent practices
                nn.Linear(config.hidden_size, 1) # Project to a single score per token
            )
            self.sequence_pooler = None
            # Make parameters trainable
            for param in self.output_projection.parameters():
                param.requires_grad = True
        else: # Sequence Level
            self.sequence_pooler = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(), # Using ReLU
                nn.Linear(config.hidden_size, 1) # Project pooled state to single value
            )
            self.output_projection = None
            # Make parameters trainable
            for param in self.sequence_pooler.parameters():
                param.requires_grad = True

        # DO NOT move the whole model to device here.
        # ValueTrainer handles moving trainable parts (adapters, projection head) later.
        logger.info(f"Initialized ValueTransformer structure. Trainable head initialized on CPU.")


    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, return_attention=False,
             return_token_scores=False, return_dict=None, requires_grad=False, **kwargs):
        """
        Forward pass through the value transformer model.
        Handles potential mixed-device execution if base is on CPU and adapters/head are on GPU.
        """
        # Handle PEFT compatibility for return_dict
        return_dict = return_dict if return_dict is not None else self.config.return_dict if hasattr(self, 'config') else True

        assert input_ids is not None or inputs_embeds is not None, "Either input_ids or inputs_embeds must be provided"
        assert input_ids is None or inputs_embeds is None, "Only one of input_ids or inputs_embeds can be provided"

        # Determine the device where the computation should happen (where trainable params are)
        # This assumes the projection head/pooler dictates the compute device.
        compute_device = next(p for p in self.parameters() if p.requires_grad).device

        # Move inputs to the compute device (GPU)
        if input_ids is not None:
            input_ids = input_ids.to(compute_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(compute_device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(compute_device)


        # --- Base Model Forward Pass ---
        # PEFT's forward pass should handle the CPU->GPU transfer for adapters internally
        # We explicitly set requires_grad for the forward pass context
        with torch.set_grad_enabled(requires_grad):
             # Ensure kwargs for base model are set correctly
             kwargs['output_hidden_states'] = True
             kwargs['output_attentions'] = return_attention

             if inputs_embeds is None:
                  # Input embeddings will be computed on the base model's device (CPU or GPU)
                  # PEFT handles moving hidden states between devices as needed.
                  outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     return_dict=return_dict, **kwargs)
             else:
                  # Ensure input_embeds are on the base model's device if it differs?
                  # PEFT should ideally handle this. Let's assume it does for now.
                  # If errors occur, we might need to move inputs_embeds manually.
                  outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     return_dict=return_dict, **kwargs)

        # --- Process Outputs on Compute Device ---
        # The final hidden state should be on the compute_device after PEFT forward
        hidden_states = outputs.hidden_states[-1].to(compute_device)

        attentions = None
        if return_attention:
            # Attentions might be on the base model's device, move if needed
            attentions = outputs.attentions[0].to(compute_device) if outputs.attentions else None
            # assert attentions is not None, "Attention weights not returned by model; check model support"

        token_scores = None
        value = None

        # --- Projection Head / Pooling ---
        # This part happens entirely on the compute_device (GPU)
        with torch.set_grad_enabled(requires_grad):
            if self.token_level:
                if self.output_projection is None:
                     raise RuntimeError("ValueTransformer is token_level but output_projection is None.")
                # Project hidden states to a single score per token
                token_scores = self.output_projection(hidden_states).squeeze(-1) # Shape: [batch, seq_len]
                # Apply attention mask to zero out scores for padding tokens before aggregation/loss
                if attention_mask is not None:
                     token_scores = token_scores * attention_mask.to(compute_device) # Ensure mask is on correct device
                # The 'value' can be considered the token_scores themselves for token-level loss
                # Or an aggregated value if needed later (e.g., for sequence-level advantage)
                value = token_scores # Keep as [batch, seq_len]
            else: # Sequence Level
                if self.sequence_pooler is None:
                     raise RuntimeError("ValueTransformer is sequence_level but sequence_pooler is None.")
                # Apply attention mask before pooling
                if attention_mask is not None:
                    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(compute_device)
                    pooled_hidden_states = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    pooled_hidden_states = hidden_states.mean(dim=1) # Mean pooling

                # Project pooled state to a single value
                value = self.sequence_pooler(pooled_hidden_states).squeeze(-1) # Shape: [batch]


        # Prepare return values based on flags
        # Note: Returning token_logits doesn't make sense if token_level=False
        # Returning token_scores (post-projection) makes more sense for token_level=True
        if return_token_scores and self.token_level and return_attention:
            return value, token_scores, attentions
        elif return_token_scores and self.token_level:
            return value, token_scores
        elif return_attention:
            return value, attentions
        else:
            return value # Returns [batch, seq_len] if token_level, [batch] if sequence_level

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary for the ValueTransformer model.
        
        Args:
            state_dict: Dictionary containing parameter tensors
        """
        if isinstance(state_dict, dict):
            if 'output_projection' in state_dict:
                self.output_projection.load_state_dict(state_dict['output_projection'])
        else:
            # Handle PEFT-specific state_dict format
            super().load_state_dict(state_dict)
        return self
    
    def state_dict(self):
        """
        Return the state dictionary for the ValueTransformer model.
        
        Returns:
            Dict: State dictionary containing parameter tensors
        """
        state_dict = {}
        state_dict['output_projection'] = self.output_projection.state_dict()
        return state_dict
        
    def get_input_embeddings(self):
        """
        Return the input embedding layer of the model.
        Required for PEFT compatibility.
        """
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        """
        Return the output embedding layer of the model.
        Required for PEFT compatibility.
        """
        if hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()
        return None
    
    def resize_token_embeddings(self, new_num_tokens=None):
        """
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Required for PEFT compatibility.
        """
        return self.model.resize_token_embeddings(new_num_tokens)

    @property
    def device(self):
        return self.model.device