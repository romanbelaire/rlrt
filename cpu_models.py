import gc
from transformers.tokenization_utils_base import BatchEncoding
import torch
import torch.nn.functional as F
from models import TransformerPolicy, ValueTransformer

class CPUOffloadTransformerPolicy(TransformerPolicy):
    """
    TransformerPolicy with CPU offloading capability for the base model.
    This subclass specifically handles the case where keep_base_on_cpu=True,
    ensuring input tensors are properly placed for efficient memory usage.
    """
    
    def __init__(self, hf_checkpoint, tokenizer, model=None, token=None, is_victim=False, 
                 use_lora=True, lora_config=None, adapter_name=None):
        super().__init__(hf_checkpoint, tokenizer, model, token, is_victim, 
                        use_lora, lora_config, adapter_name)
        
        # Keep track of references to GPU tensors we create
        self._gpu_tensor_refs = set()
        print("Using CPUOffloadTransformerPolicy: Base model on CPU, LoRA adapters on GPU")
    
    def cleanup_gpu_memory(self):
        """
        Explicitly clean up GPU memory by moving tensors to CPU and clearing references.
        This should be called after training to prevent memory leaks.
        """
        print("Cleaning up GPU memory for CPUOffloadTransformerPolicy")
        
        # Move lm_head_clone to CPU if it's on CUDA
        if hasattr(self, 'lm_head_clone') and self.lm_head_clone.weight.device.type == 'cuda':
            self.lm_head_clone = self.lm_head_clone.to('cpu')
            print("Moved lm_head_clone to CPU")
            
        # Clear cached references
        self._gpu_tensor_refs.clear()
        
        # Reset any cached forward pass states
        if hasattr(self.model, 'past_key_values') and self.model.past_key_values is not None:
            self.model.past_key_values = None
            
        # Disable key-value caching for the next forward pass
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False
            
        # For any modules that should always be on CPU, ensure they are
        for name, module in self.named_modules():
            # Skip LoRA modules which should stay on GPU
            if 'lora' in name.lower():
                continue
                
            # Check if module has parameters on GPU
            has_cuda_params = False
            for param in module.parameters(recurse=False):
                if param.device.type == 'cuda':
                    has_cuda_params = True
                    break
            
            # If not a LoRA module and has CUDA params, move to CPU
            if has_cuda_params:
                try:
                    module.to('cpu')
                    print(f"Moved module {name} to CPU")
                except Exception as e:
                    print(f"Error moving module {name} to CPU: {e}")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory cleanup complete")
        
    def to(self, device):
        """
        Override to method to ensure CPU offloading behavior is maintained.
        If moving to CPU, move everything. If moving to CUDA, keep base model on CPU.
        """
        if device == 'cpu':
            # When moving to CPU, move everything including LoRA adapters
            super().to('cpu')
            # Clear our set of GPU tensor references
            self._gpu_tensor_refs.clear()
        else:
            # Keep base model on CPU, but move LoRA adapters to GPU
            # Only move LoRA parameters to GPU
            for name, param in self.named_parameters():
                if 'lora' in name.lower():
                    param.data = param.data.to(device)
            
            # Always move lm_head_clone to GPU for efficient forward pass
            if hasattr(self, 'lm_head_clone'):
                self.lm_head_clone = self.lm_head_clone.to(device)

        
        return self
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, with_head=False, pad_token_id=None):
        """
        Forward pass with CPU offloading for memory efficiency.        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            labels: Optional target labels
            with_head: Whether to use language modeling head
            pad_token_id: Padding token ID
            
        Returns:
            Transformer outputs
        """
        try:
            model = self.model if with_head else self.model.model
            
            embed_layer = self.model.get_input_embeddings()
            embed_device = next(embed_layer.parameters()).device
            
            if input_ids is not None:
                input_ids_for_embed = input_ids.to(embed_device)
            else:
                input_ids_for_embed = None
            
            if attention_mask is not None:
                attention_mask_for_model = attention_mask.to(embed_device)
            else:
                attention_mask_for_model = None
            
            if input_ids_for_embed is not None:
                input_embeds = embed_layer(input_ids_for_embed)
                input_embeds.requires_grad_(torch.is_grad_enabled())
            else:
                input_embeds = None

            outputs = model(
                inputs_embeds=input_embeds, 
                attention_mask=attention_mask_for_model, 
                output_hidden_states=True, 
                return_dict=True, 
                pad_token_id=pad_token_id
            )

            if hasattr(outputs, 'hidden_states'):
                # Move the last hidden state to CUDA for subsequent processing
                last_hidden = outputs.hidden_states[-1].to('cuda')
                self._gpu_tensor_refs.add(id(last_hidden))  # Track this tensor
                
                # Create a new tuple with updated hidden states
                hidden_states_list = list(outputs.hidden_states)
                # Only move the last hidden state to save memory
                hidden_states_list[-1] = last_hidden
                outputs.hidden_states = tuple(h.requires_grad_(torch.is_grad_enabled()) 
                                            for i, h in enumerate(hidden_states_list))
                # Track all hidden states for cleanup
                for h in outputs.hidden_states:
                    self._gpu_tensor_refs.add(id(h))
            
            # Ensure logits are on CUDA
            if hasattr(outputs, 'logits') and outputs.logits.device.type != 'cuda':
                outputs.logits = outputs.logits.to('cuda')
                self._gpu_tensor_refs.add(id(outputs.logits))
                
            # ========== STEP 6: Final processing on GPU ==========
            # Compute loss using frozen model head as reference
            ce_shape = torch.Size([outputs.logits.shape[0] * outputs.logits.shape[1], outputs.logits.shape[-1]])
            
            # Ensure lm_head_clone is on CUDA
            if self.lm_head_clone.weight.device.type != 'cuda':
                self.lm_head_clone = self.lm_head_clone.to('cuda')
                
            # Compute reference logits and probabilities
            reference_logits = self.lm_head_clone(last_hidden)
            reference_probs = F.softmax(reference_logits, dim=-1)
            reference_probs = reference_probs.view(ce_shape)
            self._gpu_tensor_refs.add(id(reference_logits))
            self._gpu_tensor_refs.add(id(reference_probs))
            
            # Compute cross entropy loss
            ce_loss = F.cross_entropy(outputs.logits.view(ce_shape), reference_probs)
            outputs.loss = ce_loss
            self._gpu_tensor_refs.add(id(ce_loss))
            self._gpu_tensor_refs.add(id(outputs.loss))
            
            # Ensure logits have gradients enabled
            if hasattr(outputs, 'logits'):
                outputs.logits.requires_grad_(torch.is_grad_enabled())
                
            return outputs
                
        except Exception as e:
            print(f"Error in CPU-offloaded forward pass: {str(e)}")
            print("Debugging device placement:")
            import traceback
            traceback.print_exc()
            raise e

    def get_log_probs(self, input_ids, output_ids, require_grad=True, sequence_level=False):
        """
        Override get_log_probs to ensure proper CPU/GPU placement.
        
        Args:
            input_ids: Input token IDs
            output_ids: Output token IDs
            require_grad: Whether to require gradients
            sequence_level: Whether to return sequence-level log probs
            
        Returns:
            Log probabilities and CE loss
        """
        # Tokenize inputs if necessary
        if not isinstance(input_ids, torch.Tensor):
            input_ids = self.tokenizer(input_ids, return_tensors='pt', padding=True).input_ids
        if not isinstance(output_ids, torch.Tensor):
            output_ids = self.tokenizer(output_ids, return_tensors='pt', padding=True).input_ids
        
        # Get the device of the embedding layer
        embed_layer = self.model.get_input_embeddings()
        embed_device = next(embed_layer.parameters()).device
        
        # Move inputs to the appropriate device for the embedding layer
        curr_input = input_ids.to(embed_device)
        log_probs = torch.zeros(output_ids.size(0), output_ids.size(1), requires_grad=require_grad).to('cuda')
        self._gpu_tensor_refs.add(id(log_probs))  # Track for cleanup
        
        attention_mask = torch.where(curr_input == self.tokenizer.pad_token_id,
                                    torch.zeros_like(curr_input),
                                    torch.ones_like(curr_input)).to(embed_device)
        ce_loss = []
        epsilon = 1e-10 # Small epsilon value
        
        # Generate token by token
        for i in range(output_ids.size(1)):
            with torch.set_grad_enabled((i == output_ids.size(1) - 1) and require_grad):
                # Forward pass with current input
                output = self.forward(curr_input, attention_mask=attention_mask, with_head=True)
                
                # Process outputs (already moved to GPU in forward method)
                next_token_logits = output.logits[:, -1, :]
                
                # Get probability of next token (on GPU)
                next_token = output_ids[:, i].to('cuda')
                self._gpu_tensor_refs.add(id(next_token))  # Track for cleanup
                
                # Apply softmax + epsilon before log
                probs = F.softmax(next_token_logits, dim=-1)
                log_prob_i = torch.log(probs + epsilon).gather(1, next_token.unsqueeze(-1)).squeeze(-1)
                log_probs[:, i] = log_prob_i
                ce_loss.append(output.loss)
                
                # Append next token to input (on same device as curr_input)
                next_token_for_input = next_token.to(embed_device)
                curr_input = torch.cat([curr_input, next_token_for_input.unsqueeze(-1)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_for_input.unsqueeze(-1))], dim=1)
        
        # Return sequence-level log probs if requested
        stacked_ce_loss = torch.stack(ce_loss)
        self._gpu_tensor_refs.add(id(stacked_ce_loss))  # Track for cleanup
        
        if sequence_level:
            seq_log_probs = log_probs.sum(dim=-1)/output_ids.size(1)
            self._gpu_tensor_refs.add(id(seq_log_probs))  # Track for cleanup
            return seq_log_probs, stacked_ce_loss
        return log_probs, stacked_ce_loss
        
    def generate(self, batch, max_new_tokens=100, k=3, offset=0, enforce_single_question=False, require_grad=True, pad_token_id=None, 
                sampling_method='top_k', p=0.9, temperature=1.0, beam_size=None, length_penalty=1.0):
        """
        Generate text with careful device management for CPU offloading.
        
        Args:
            batch: Input batch (string, list of strings, or BatchEncoding)
            max_new_tokens: Maximum number of new tokens to generate
            k: Top-k for sampling
            offset: Token offset
            enforce_single_question: Whether to enforce exactly one question
            require_grad: Whether to require gradients
            pad_token_id: Padding token ID
            sampling_method: Sampling method ('top_k', 'top_p', 'beam')
            p: Top-p probability threshold for nucleus sampling
            temperature: Temperature for sampling
            beam_size: Beam size for beam search
            length_penalty: Length penalty for beam search
            
        Returns:
            tuple: (generated_text, token_ids, log_probs, lm_losses)
        """
        if pad_token_id is None and hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id
            
        # Get the device of the embedding layer
        embed_layer = self.model.get_input_embeddings()
        embed_device = next(embed_layer.parameters()).device
            
        # Process input batch
        if isinstance(batch, BatchEncoding):
            tokens = batch
            input_ids = tokens.input_ids.to(embed_device)
            attention_mask = tokens.attention_mask.to(embed_device)
        else:
            if isinstance(batch, str):
                assert batch.strip() != "", "Input batch cannot be an empty string"
                batch = [batch]
            elif isinstance(batch, list):
                assert all(isinstance(item, str) and item.strip() != "" for item in batch), "Input batch cannot contain empty strings"
                
            # Tokenize and move to appropriate device
            tokens = self.tokenizer(batch, return_tensors='pt', padding=True, 
                                    add_special_tokens=(batch[0].startswith(self.tokenizer.bos_token)))
            input_ids = tokens.input_ids.to(embed_device)
            attention_mask = torch.where(input_ids == pad_token_id, 
                                         torch.zeros_like(input_ids), 
                                         torch.ones_like(input_ids)).to(embed_device)
        
        input_length = input_ids.shape[-1]
        
        # If beam search is explicitly requested, use it
        if beam_size is not None and beam_size > 1:
            # Run beam search
            # Note: beam search will handle device placement internally
            best_sequences, best_scores, best_log_probs, best_lm_losses = self.beam_search(
                input_ids, 
                attention_mask,
                beam_size=beam_size,
                max_length=max_new_tokens,
                length_penalty=length_penalty,
                pad_token_id=pad_token_id
            )
            
            # Process results
            output_tokens = [seq[input_length:] for seq in best_sequences]
            output_seq = [self.tokenizer.decode(tokens, skip_special_tokens=False) for tokens in output_tokens]
            
            # Use the returned log probs and LM losses
            batch_size = len(output_seq)
            
            # Create a padded tensor for the log probabilities
            max_log_probs_len = max(log_probs.size(0) for log_probs in best_log_probs)
            output_log_probs = torch.zeros(batch_size, max_log_probs_len).to('cuda')
            for i, log_probs in enumerate(best_log_probs):
                output_log_probs[i, :log_probs.size(0)] = log_probs
            self._gpu_tensor_refs.add(id(output_log_probs))
            
            # Process LM losses
            if all(losses.size(0) == best_lm_losses[0].size(0) for losses in best_lm_losses):
                lm_losses = torch.stack(best_lm_losses)
            else:
                max_losses_len = max(losses.size(0) for losses in best_lm_losses)
                lm_losses = torch.zeros(batch_size, max_losses_len).to('cuda')
                for i, losses in enumerate(best_lm_losses):
                    lm_losses[i, :losses.size(0)] = losses
                lm_losses = lm_losses.mean(dim=0)
            self._gpu_tensor_refs.add(id(lm_losses))
            
            # Convert output_tokens list to tensor with padding
            max_len = max(tokens.size(0) for tokens in output_tokens)
            padded_tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long).to('cuda')
            for i, tokens in enumerate(output_tokens):
                padded_tokens[i, :tokens.size(0)] = tokens.to('cuda')
            self._gpu_tensor_refs.add(id(padded_tokens))
            
            return output_seq, padded_tokens, output_log_probs, lm_losses
        
        # Standard autoregressive generation
        batch_size = input_ids.shape[0]
        generated_tokens = input_ids.clone()
        output_log_probs = []
        lm_losses = []
        
        # Store positions where we need to stop generation due to finding end_token
        finished = torch.zeros(batch_size, dtype=torch.bool, device=embed_device).to('cuda')
        
        for _ in range(max_new_tokens):
            # Run forward pass
            with torch.set_grad_enabled(require_grad):
                outputs = self.forward(generated_tokens, attention_mask=attention_mask, with_head=True)
                
                next_token_logits = outputs.logits[:, -1, :].to('cuda')
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply sampling method
                if sampling_method == 'top_k':
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k=k, dim=-1)
                    log_probs = F.log_softmax(top_k_logits, dim=-1)
                    probs = torch.exp(log_probs)
                    
                    # Sample from top-k distribution
                    next_token_indices = torch.multinomial(probs, num_samples=1)
                    next_tokens = torch.gather(top_k_indices, -1, next_token_indices).squeeze(-1)
                    
                    # Get log prob of selected token
                    selected_log_probs = torch.gather(log_probs, -1, next_token_indices).squeeze(-1)
                    
                elif sampling_method == 'top_p':
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    log_probs = F.log_softmax(sorted_logits, dim=-1)
                    probs = torch.exp(log_probs)
                    
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    
                    # Create mask for tokens below the probability threshold p
                    nucleus_mask = cumulative_probs < p
                    
                    # Add the token with the highest probability to the nucleus
                    nucleus_mask[:, 0] = True
                    
                    # Apply the mask to select tokens in the nucleus
                    nucleus_logits = torch.where(nucleus_mask, sorted_logits, torch.full_like(sorted_logits, float('-inf')))
                    nucleus_probs = F.softmax(nucleus_logits, dim=-1)
                    
                    # Sample from nucleus distribution
                    nucleus_indices = torch.multinomial(nucleus_probs, num_samples=1)
                    next_tokens = torch.gather(sorted_indices, -1, nucleus_indices).squeeze(-1)
                    
                    # Get log prob of selected token
                    nucleus_log_probs = F.log_softmax(nucleus_logits, dim=-1)
                    selected_log_probs = torch.gather(nucleus_log_probs, -1, nucleus_indices).squeeze(-1)
                    
                else:  # Default to greedy
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    selected_log_probs = torch.gather(log_probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
                
                # Store log probabilities and LM loss
                output_log_probs.append(selected_log_probs.to('cuda'))
                lm_losses.append(outputs.loss)
                
                # Move next tokens to the right device for input
                next_tokens_for_input = next_tokens.to(embed_device)
                
                # Expand attention mask for next tokens
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=embed_device)
                ], dim=-1)
                
                # Update generation and mark finished sequences
                generated_tokens = torch.cat([generated_tokens, next_tokens_for_input.unsqueeze(-1)], dim=-1)
                
                # Check for end token or finished generation
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    newly_finished = next_tokens == self.tokenizer.eos_token_id
                    finished = finished | newly_finished
                    if finished.all():
                        break
            
            self._gpu_tensor_refs.add(id(output_log_probs[-1]))
            self._gpu_tensor_refs.add(id(lm_losses[-1]))
            
        # Decode the generated output
        output_tokens = generated_tokens[:, input_length:]
        self._gpu_tensor_refs.add(id(output_tokens))
        
        output_seq = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
        
        # Stack log probabilities and LM losses
        stacked_log_probs = torch.stack(output_log_probs, dim=1).to('cuda')
        self._gpu_tensor_refs.add(id(stacked_log_probs))
        
        if lm_losses:
            stacked_lm_losses = torch.stack(lm_losses)
            self._gpu_tensor_refs.add(id(stacked_lm_losses))
        else:
            stacked_lm_losses = torch.zeros(max_new_tokens).to('cuda')
            self._gpu_tensor_refs.add(id(stacked_lm_losses))
        
        return output_seq, output_tokens, stacked_log_probs, stacked_lm_losses

    def debug_device_info(self, input_ids=None, attention_mask=None):
        """
        Print debugging information about device placement of model components and inputs.
        
        Args:
            input_ids: Optional input tensor to check
            attention_mask: Optional attention mask tensor to check
            
        Returns:
            dict: Dictionary with device information
        """
        info = {
            "model_device": str(self.model.device) if hasattr(self.model, "device") else "unknown",
            "lm_head_device": str(self.lm_head_clone.weight.device),
        }
        
        # Check the device of the first parameter for main layers
        for name, module in self.model.named_modules():
            if any(layer_type in name for layer_type in ["embed_tokens", "layers.0", "layers.1", "lm_head"]):
                for param_name, param in module.named_parameters():
                    info[f"{name}.{param_name}_device"] = str(param.device)
                    break  # Just check the first parameter
        
        # Check input tensors if provided
        if input_ids is not None:
            info["input_ids_device"] = str(input_ids.device)
        
        if attention_mask is not None:
            info["attention_mask_device"] = str(attention_mask.device)
            
        # Print the information
        print("\n===== Device Placement Debug Info =====")
        for key, value in info.items():
            print(f"{key}: {value}")
        print("=======================================\n")
        
        return info

    def to_adapter_device(self, device):
        """
        Move only the LoRA adapter parameters to the specified device.
        
        Args:
            device: The device to move the LoRA adapters to ('cuda' or 'cpu')
        """
        # First check if this model has LoRA adapters
        if not hasattr(self.model, 'peft_config'):
            return
            
        # Loop through all named parameters and move only the LoRA related ones
        for name, param in self.model.named_parameters():
            if any(adapter_key in name for adapter_key in ['lora_A', 'lora_B']):
                param.data = param.data.to(device)
                
        # Also move the lm_head_clone to the target device
        if hasattr(self, 'lm_head_clone'):
            self.lm_head_clone = self.lm_head_clone.to(device)
            
        print(f"Moved LoRA adapter parameters to {device}")
        
        # Clear any cached tensors on the wrong device
        self._clear_gpu_tensors()

class CPUOffloadValueTransformer(ValueTransformer):
    """
    CPU-offloaded version of ValueTransformer to reduce GPU memory usage.
    
    Follows the same pattern as CPUOffloadTransformerPolicy by keeping the base model
    on CPU and only moving necessary tensors to GPU for computation.
    """
    def __init__(self, hf_checkpoint, tokenizer, model=None, token=None, token_level=False, attention_update=True):
        super(CPUOffloadValueTransformer, self).__init__(
            hf_checkpoint, tokenizer, model, token, token_level, attention_update
        )
        
        # Set the base model to CPU
        self.model.to('cpu')
        
        # Keep trainable components on GPU
        self.embeds.to('cuda')
        self.self_attention.to('cuda')
        self.token_scores.to('cuda')
        if self.cloned_attention is not None:
            self.cloned_attention.to('cuda')
            
        # Track GPU tensors for cleanup
        self._gpu_tensor_refs = set()
        
        print(f"Initialized CPUOffloadValueTransformer with token_level={token_level}, base model on CPU, trainable layers on GPU")
    
    def _clear_gpu_tensors(self):
        """Clear references to GPU tensors to free memory."""
        self._gpu_tensor_refs.clear()
        
    def cleanup_gpu_memory(self):
        """Free GPU memory by clearing tensor references and emptying cache."""
        self._clear_gpu_tensors()
        torch.cuda.empty_cache()
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, return_attention=False, return_token_scores=False):
        """
        Forward pass through the value transformer model with CPU offloading.
        
        This implementation carefully manages CPU/GPU memory by:
        1. Keeping the base model on CPU
        2. Moving only necessary tensors to GPU for computation
        3. Tracking GPU tensors for cleanup
        
        Args:
            input_ids: Input token IDs (will be used on CPU)
            inputs_embeds: Input embeddings (alternative to input_ids, will be used on CPU)
            attention_mask: Attention mask (will be used on CPU initially)
            return_attention: Whether to return attention weights
            return_token_scores: Whether to return per-token scores
            
        Returns:
            Same as the base ValueTransformer.forward method
        """
        try:
            assert input_ids is not None or inputs_embeds is not None, "Either input_ids or inputs_embeds must be provided"
            assert input_ids is None or inputs_embeds is None, "Only one of input_ids or inputs_embeds can be provided"
            
            if input_ids is not None:
                input_ids = input_ids.to('cpu')
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.to('cpu')
            
            with torch.set_grad_enabled(True):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            if hasattr(outputs, 'hidden_states'):
                outputs.hidden_states = tuple(h.to('cuda') for h in outputs.hidden_states)
            
            # Move logits to GPU
            if hasattr(outputs, 'logits'):
                outputs.logits = outputs.logits.to('cuda')
            
            hidden_states = outputs.hidden_states[-1].to('cuda')
            
            # Process with trainable layers
            embeds = self.embeds(hidden_states)
            
            # Apply self-attention to model token dependencies
            # Transpose for attention: [batch_size, seq_len, hidden_size] -> [seq_len, batch_size, hidden_size]
            embeds_transposed = embeds.transpose(0, 1)
            attn_output, attn_weights = self.self_attention(
                embeds_transposed, 
                embeds_transposed, 
                embeds_transposed, 
                need_weights=True
            )
            attn_output = attn_output.transpose(0, 1)
            
            token_logits = self.token_scores(attn_output)
            
            if self.token_level:
                if input_ids is not None:
                    input_ids_gpu = input_ids.to('cuda')
                    token_scores = torch.gather(token_logits, dim=-1, index=input_ids_gpu.unsqueeze(-1))
                    value = torch.tanh(token_scores)
                else:
                    value = token_logits
            else:
                token_scores = token_logits.squeeze(-1)
                if attention_mask is not None:
                    attention_mask_gpu = attention_mask.to('cuda')
                    token_scores = token_scores * attention_mask_gpu.float()
                value = token_scores.sum(dim=1)
            
            if return_token_scores and return_attention:
                return value, token_logits, attn_weights
            elif return_token_scores:
                return value, token_logits
            elif return_attention:
                return value, attn_weights
            else:
                return value
        
        except Exception as e:
            print(f"Error in CPU-offloaded value forward pass: {str(e)}")
            print("Debugging device placement:")
            print(f"Model device: {self.model.device}")
            print(f"Embeds device: {self.embeds.weight.device}")
            print(f"Self-attention device: {next(self.self_attention.parameters()).device}")
            print(f"Token scores device: {self.token_scores.weight.device}")
            if self.cloned_attention is not None:
                print(f"Cloned attention device: {next(self.cloned_attention.parameters()).device}")
            if input_ids is not None:
                print(f"Input IDs device: {input_ids.device}")
            if attention_mask is not None:
                print(f"Attention mask device: {attention_mask.device}")
            import traceback
            traceback.print_exc()
            raise e
    
    def to(self, device):
        """
        Selectively move components to the specified device.
        
        For 'cpu', moves everything to CPU.
        For 'cuda', keeps the base model on CPU and only moves trainable layers to GPU.
        
        Args:
            device: Target device ('cpu' or 'cuda')
        """
        if device == 'cpu':
            self.model.to('cpu')
            self.embeds.to('cpu')
            self.self_attention.to('cpu')
            self.token_scores.to('cpu')
            if self.cloned_attention is not None:
                self.cloned_attention.to('cpu')
            return self
        
        elif device == 'cuda':
            # Keep base model on CPU, but move trainable layers to GPU
            self.model.to('cpu')
            self.embeds.to('cuda')
            self.self_attention.to('cuda')
            self.token_scores.to('cuda')
            if self.cloned_attention is not None:
                self.cloned_attention.to('cuda')
            return self
        
        else:
            raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda'.")
            
    def parameters(self):
        """
        Return the trainable parameters of the model.
        
        This skips the base model parameters and only includes
        the parameters of the trainable layers.
        
        Returns:
            Iterator over trainable parameters
        """
        for param in self.embeds.parameters():
            yield param
        for param in self.self_attention.parameters():
            yield param
        for param in self.token_scores.parameters():
            yield param
        if self.cloned_attention is not None:
            for param in self.cloned_attention.parameters():
                yield param



