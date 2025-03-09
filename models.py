from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, FbgemmFp8Config
from transformers.modeling_outputs import CausalLMOutput
from transformers.tokenization_utils_base import BatchEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from contextlib import contextmanager
import psutil
import time
import copy
from helpers import log_generations
#from trl import SFTTrainer
#from unsloth.chat_templates import get_chat_template
#from unsloth import FastLanguageModel, is_bfloat16_supported
#import copy


class TransformerPolicy(nn.Module):
    def __init__(self, hf_checkpoint, tokenizer, model=None, token=None, is_victim=False):
        super(TransformerPolicy, self).__init__()

        init_memory = torch.cuda.memory_allocated()
        # Configure model to output hidden states with gradients
        config = AutoConfig.from_pretrained(hf_checkpoint, token=token, trust_remote_code=True)
        config.output_hidden_states = True
        config.return_dict = True
        config.return_dict_in_generate = True
        config.use_cache = False  # Important for gradient computation

        if 'deepseek' in hf_checkpoint:
            quant_config = FbgemmFp8Config()
            self.model = LlamaForCausalLM.from_pretrained(
                hf_checkpoint, 
                config=config,
                token=token,
                quant_config=quant_config,
                trust_remote_code=True
            )
        else:
            self.model = model if model is not None else AutoModelForCausalLM.from_pretrained(
                hf_checkpoint, 
                config=config,
                token=token,
                trust_remote_code=True
            )

        # Load Model with given checkpoint and extract its body 
        self.tokenizer = tokenizer 
        self.force_stop_strings = ['?', '!', '.', 'assistant', 'Assistant']
        self.force_stop_tokens = torch.tensor([self.tokenizer(s).input_ids[-1] for s in self.force_stop_strings], dtype=torch.int16).to('cuda')

        print("Checkpoint for TransformerPolicy model: " + hf_checkpoint)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.body = self.model.model
        self.model_dim = config.hidden_size
        self.action_dim = 1
        self.dropout = nn.Dropout(0.1)
        assert hasattr(self.model, 'lm_head'), "Model must have an lm_head module"
        # Enable gradient checkpointing with use_reentrant=True
        #self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        #tokenizer special tokens. Don't need at the moment. 
        #see https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/ for more info
        #self.tokenizer.add_special_tokens({'additional_special_tokens': ['[INST]', '[/INST]', '<<SYS>>', '<</SYS']})
        #self.model.resize_token_embeddings(len(self.tokenizer))

        # First set up the final layer
        self.final = self.model.lm_head
        self.lm_head_clone = copy.deepcopy(self.final)
        self.lm_head_clone.requires_grad = False
        
        # Freeze body EXCEPT final layer and embeddings
        for name, param in self.model.model.named_parameters():
            if 'lm_head' not in name and 'embed_tokens' not in name:
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

        #self.model.register_forward_hook(forward_hook)
        self.print_embedding_info()
        assert self.model.get_input_embeddings().weight.requires_grad, "Embeddings must require gradients"
        assert self.final.weight.requires_grad, "Final layer weights must require gradients"
        assert self.tokenizer.padding_side == 'left', "Padding side must be left"

    def check_layer_grads(self):
        assert self.model.get_input_embeddings().weight.requires_grad, "Embeddings must require gradients"
        assert self.final.weight.requires_grad, "Final layer weights must require gradients"

    def parameters(self):
        params = []
        for param in self.final.parameters():
            params.append(param)
        for param in self.model.get_input_embeddings().parameters():
            if isinstance(self.model, nn.DataParallel):
                name = self.model.module.name_or_path   
            else:
                name = self.model.name_or_path
            if 'llama' in name.lower() and param not in params:
                params.append(param)
        return params

    def forward(self, input_ids=None, attention_mask=None, labels=None, with_head=False, pad_token_id=None):        
        try:            
            model = self.model if with_head else self.body
            # Explicitly get and attach embeddings to gradient graph
            input_embeds = self.model.get_input_embeddings()(input_ids)
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
            outputs.hidden_states = tuple(h.requires_grad_(torch.is_grad_enabled()) for h in outputs.hidden_states)

            # compute cross entropy loss using frozen model head as labels
            ce_shape = torch.Size([outputs.logits.shape[0] * outputs.logits.shape[1], outputs.logits.shape[-1]])
            reference_logits = self.lm_head_clone(outputs.hidden_states[-1])
            reference_probs = F.softmax(reference_logits, dim=-1)
            reference_probs = reference_probs.view(ce_shape)

            ce_loss = F.cross_entropy(outputs.logits.view(ce_shape), reference_probs)
            outputs.loss = ce_loss

            if hasattr(outputs, 'logits'):
                outputs.logits.requires_grad_(torch.is_grad_enabled())
                    
            return outputs
                
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            raise e

    def generate(self, batch, max_new_tokens=100, k=3, offset=0, enforce_single_question=False, require_grad=True, pad_token_id=None):
        """
        Generates sequences of tokens from the input batch using the model.
        Args:
            - batch (list or str): The input batch of text to be tokenized and processed.
            - max_length (int, optional): The maximum length of the generated sequences. Defaults to 500.
            - return_tokens (bool, optional): Whether to return the generated sequences as tokens or text. Defaults to False.
            - k (int, optional): The number of top-k tokens to sample from. Defaults to 10.
            - offset (int, optional): The offset to apply to the top-k sampling window. Defaults to 0.
            - enforce_single_question (bool, optional): Whether to enforce that the generated sequences contain only one statement. Defaults to False.
        Returns:
          tuple: A tuple containing:
            - output_seq (list of str): The generated sequences decoded into text.
            - output_log_probs (torch.Tensor): The tensor containing the log probabilities for each token in the generated sequences.
        """
        # Assert that none of the inputs are empty strings
        assert self.tokenizer.padding_side == 'left'
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
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
        output_ids = input_ids.to('cuda')#.clone() 
        attention_mask = torch.where(input_ids == pad_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)).to('cuda') #triangle_attn(output_ids)        
        assert torch.all(attention_mask[:, -1] == 1), "Attention mask must be 1 for the last token"
        output_log_probs = torch.full(input_ids.shape, fill_value=-10, dtype=torch.float32).to('cuda')
        output_log_probs.requires_grad = True

        for i in range(max_new_tokens):
            # Only enable gradients for final step when require_grad is True
            with torch.set_grad_enabled(require_grad and i == max_new_tokens - 1):
                output = self.forward(input_ids=output_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, with_head=True)
                next_logits = output.logits[:, -1, :]
                next_token_id = self.sample_top_k(next_logits, k=k, adv_offset=offset)
                if i > 10:#early stopping criteria
                    #follow all force_stop_tokens with EOS token
                    if enforce_single_question:
                        cond1 = torch.BoolTensor([i in self.force_stop_tokens.to('cuda') for i in output_ids[:, -1]]).view(next_token_id.shape).to('cuda') 
                        next_token_id = torch.where(cond1, torch.full_like(next_token_id, self.tokenizer.eos_token_id), next_token_id)
                    #follow all EOS and PAD tokens with another PAD token
                    cond1 = output_ids[:, -1] == self.tokenizer.eos_token_id
                    cond2 = output_ids[:, -1] == self.tokenizer.pad_token_id
                    next_token_id = torch.where(torch.logical_or(cond1, cond2), 
                                            torch.full_like(next_token_id, self.tokenizer.pad_token_id), next_token_id) #append padding token to sequences that have ended  
                #if torch.all(next_token_id == self.tokenizer.pad_token_id):
                    #all sequences have ended, don't append anything and break
                    #break
                next_token_id = next_token_id.unsqueeze(-1)
                if len(next_token_id.shape) == 1: #if batch size is 1 add a dimension
                    next_token_id = next_token_id.unsqueeze(0)
                assert next_token_id.dtype == torch.int64, f"next_token_id has dtype {next_token_id.dtype}"
                softmax_logits = F.log_softmax(next_logits, dim=-1)
                if next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(1)  # Shape: (batch_size, 1)
                next_token_log_prob = torch.gather(softmax_logits, dim=-1, index=next_token_id)
                
                output_ids = torch.cat([output_ids, next_token_id], dim=-1) 
                output_log_probs = torch.cat([output_log_probs, next_token_log_prob], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)

        # Return only generated tokens and their log probs
        output_tokens = output_ids[:, input_length:]
        
        output_seq = self.tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=False)
        return output_seq, output_tokens, output_log_probs[:, input_length:].clone()

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
    
    def get_log_probs(self, input_ids, output_ids, require_grad=True, sequence_level=False):
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
        if not isinstance(input_ids, torch.Tensor):
            input_ids = self.tokenizer(input_ids, return_tensors='pt', padding=True).input_ids
        if not isinstance(output_ids, torch.Tensor):
            output_ids = self.tokenizer(output_ids, return_tensors='pt', padding=True).input_ids
            
        # Initialize tensors
        curr_input = input_ids.to('cuda')
        log_probs = torch.zeros(output_ids.size(0), output_ids.size(1), requires_grad=require_grad).to('cuda')
        attention_mask = torch.ones_like(curr_input).to('cuda')

        # Generate token by token
        for i in range(output_ids.size(1)):
            with torch.set_grad_enabled((i == output_ids.size(1) - 1) and require_grad):
                # Get model output for current input
                output = self.forward(input_ids.to('cuda'), attention_mask=attention_mask, with_head=True)

                next_token_logits = output.logits[:, -1, :]
                
                # Get probability of actual next token
                next_token = output_ids[:, i].to('cuda')
                log_probs[:, i] = F.log_softmax(next_token_logits, dim=-1).gather(1, next_token.unsqueeze(-1)).squeeze(-1)
                
                # Append actual token to input
                curr_input = torch.cat([curr_input, next_token.unsqueeze(-1)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(-1))], dim=1)
        
        # Return sequence-level log probs if requested
        if sequence_level:
            return log_probs.sum(dim=-1)/output_ids.size(-1)
        return log_probs

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
        """Freeze all layers except the final layer and input embeddings."""
        for name, param in self.model.named_parameters():
            if 'lm_head' not in name and 'embed_tokens' not in name:
                param.requires_grad = False

    def unfreeze_layers(self):
        """Unfreeze all layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = True

def forward_hook(module, input, output):
    """Convert transformer outputs to tensor format"""
    if isinstance(output, (tuple, list)):
        return tuple(x.requires_grad_(True) if isinstance(x, torch.Tensor) else x for x in output)
    elif hasattr(output, 'last_hidden_state'):
        # Extract tensor from transformer output
        hidden_state = output.last_hidden_state
        hidden_state.requires_grad_(True)
        # Preserve original attributes while returning tensor
        output.last_hidden_state = hidden_state
        return output
    elif isinstance(output, torch.Tensor):
        return output.requires_grad_(True)
    return output

def backward_hook(module, grad_input, grad_output):
    """Handle gradient computation for transformer outputs"""
    if not grad_output:
        return grad_input
    print(grad_input, grad_output)
    # Extract gradient tensor
    grad = None
    if isinstance(grad_output, tuple):
        grad = grad_output[0] if grad_output[0] is not None else None
    else:
        grad = grad_output
        
    if grad is not None:
        print(f"\nGradient Stats for {module.__class__.__name__}:")
        print(f"Shape: {grad.shape}")
        print(f"Mean: {grad.mean().item():.6f}")
        print(f"Std: {grad.std().item():.6f}")
        
    return grad_input

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        with torch.enable_grad():
            outputs = run_function(*args)
            #CheckpointFunction.last_outputs = outputs
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        with torch.set_grad_enabled(True):
            # Recompute outputs only if needed
        
            outputs = ctx.run_function(*inputs)
            assert outputs.requires_grad, "Outputs must require gradients"
            
            # Compute gradients
            grads = torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=False)
            
        return (None,) + grads
    
def triangle_attn(input):
    """returns the attention mask as a lower triangular matrix of ones with the shape of input"""
    ones = torch.ones_like(input)
    return torch.tril(ones, diagonal=0)

class ValueTransformer(nn.Module):
    def __init__(self, hf_checkpoint, tokenizer, model=None, token=None, token_level=False, attention_update=True):
        super(ValueTransformer, self).__init__()
        
        config = AutoConfig.from_pretrained(hf_checkpoint, token=token, trust_remote_code=True)
        config.output_hidden_states = True
        config.return_dict = True
        config.return_dict_in_generate = True
        config.use_cache = False  # Important for gradient computation
        if 'deepseek' in hf_checkpoint:
            quant_config = FbgemmFp8Config()
            self.model = LlamaForCausalLM.from_pretrained(
                hf_checkpoint, 
                config=config,
                token=token,
                quant_config=quant_config,
                trust_remote_code=True
            )
        else:
            self.model = model if model is not None else AutoModelForCausalLM.from_pretrained(
                hf_checkpoint, 
                config=config,
                token=token,
                trust_remote_code=True
            )
        self.tokenizer = tokenizer
        self.token_level = token_level
        self.attention_update = attention_update
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Clone the first attention layer if attention_update is True
        # This prevents modifying the shared base model
        self.cloned_attention = None
        if attention_update:
            # Find and clone the first attention layer
            first_attn_layer_found = False
            
            # For LLaMA models (most common case)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers') and len(self.model.model.layers) > 0:
                if hasattr(self.model.model.layers[0], 'self_attn'):
                    # Clone the attention layer
                    original_attn = self.model.model.layers[0].self_attn
                    self.cloned_attention = copy.deepcopy(original_attn)
                    
                    # Ensure the cloned layer requires gradients
                    for param in self.cloned_attention.parameters():
                        param.requires_grad = True
                        
                    print("Cloned first attention layer for LLaMA model")
                    first_attn_layer_found = True
            
            # Try alternative model architectures if LLaMA pattern doesn't match
            if not first_attn_layer_found:
                print("Warning: Could not find first attention layer with LLaMA pattern. Trying alternative approaches.")
                # This would need to be expanded for other model architectures
                # For now, we'll fall back to our trainable layers only
        
        # Create trainable layers for the critic
        self.embeds = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)
        
        # Output layer size depends on token_level flag
        # For help critic (token_level=False): output size is 1 (sequence-level score)
        # For harm critic (token_level=True): output size is vocab_size (token-level scores)
        output_size = config.vocab_size if token_level else 1
        self.token_scores = nn.Linear(config.hidden_size, output_size)
        
        # Ensure trainable layers have requires_grad=True
        for param in self.embeds.parameters():
            param.requires_grad = True
        for param in self.self_attention.parameters():
            param.requires_grad = True
        for param in self.token_scores.parameters():
            param.requires_grad = True

        module_list = [self.embeds, self.self_attention, self.token_scores]
        if self.cloned_attention is not None:
            module_list.append(self.cloned_attention)
        self.modules = nn.ModuleList(module_list)
            
        self.to('cuda')
        
        print(f"Initialized ValueTransformer with token_level={token_level}, output_size={output_size}, attention_update={attention_update}")

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, return_attention=False, return_token_scores=False):
        """
        Forward pass through the value transformer model.
        
        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Attention mask for input
            return_attention: Whether to return attention weights
            return_token_scores: Whether to return per-token scores before aggregation
            
        Returns:
            Tuple containing:
            - sequence_value: Aggregated sequence-level value prediction
            - token_scores: Per-token scores (if return_token_scores=True)
            - attention_weights: Self-attention weights (if return_attention=True)
        """
        assert input_ids is not None or inputs_embeds is not None, "Either input_ids or input_embeds must be provided"
        assert input_ids is None or inputs_embeds is None, "Only one of input_ids or input_embeds can be provided"
        
        # Forward pass through the base model
        if self.attention_update and self.cloned_attention is not None:
            # Use the cloned attention layer for the forward pass
            # We need to temporarily replace the original attention layer with our cloned version
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers') and len(self.model.model.layers) > 0:
                if hasattr(self.model.model.layers[0], 'self_attn'):
                    # Store the original attention layer
                    original_attn = self.model.model.layers[0].self_attn
                    
                    # Replace with our cloned version
                    self.model.model.layers[0].self_attn = self.cloned_attention
                    
                    # Forward pass
                    if inputs_embeds is None:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    else:
                        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
                    
                    # Restore the original attention layer
                    self.model.model.layers[0].self_attn = original_attn
                else:
                    # Fallback if self_attn not found
                    with torch.no_grad():
                        if inputs_embeds is None:
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        else:
                            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
            else:
                # Fallback if model structure doesn't match
                with torch.no_grad():
                    if inputs_embeds is None:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    else:
                        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        else:
            # Standard forward pass without modifying the base model
            with torch.no_grad():
                if inputs_embeds is None:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                else:
                    outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get the hidden states
        hidden_states = outputs.hidden_states[-1]
        
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
        # Transpose back: [seq_len, batch_size, hidden_size] -> [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(0, 1)
        
        # Generate scores based on token_level flag
        if self.token_level:
            # For harm critic: token-level scores with shape [batch_size, seq_len, vocab_size]
            token_scores = self.token_scores(attn_output)
            # Apply sigmoid to normalize scores to [0, 1]
            token_scores = torch.sigmoid(token_scores)
        else:
            # For help critic: sequence-level scores with shape [batch_size, seq_len, 1]
            token_scores = torch.sigmoid(self.token_scores(attn_output))
            # Remove the last dimension to get [batch_size, seq_len]
            token_scores = token_scores.squeeze(-1)
        
        # Apply attention mask if provided to zero out padding tokens
        if attention_mask is not None and not self.token_level:
            token_scores = token_scores * attention_mask.float()
        
        # Aggregate token scores to get sequence-level value
        # For token_level=False: sum across sequence dimension to get [batch_size]
        # For token_level=True: keep as is, shape [batch_size, seq_len, vocab_size]
        sequence_value = token_scores.sum(dim=1) if not self.token_level else token_scores
        
        # Prepare return values based on flags
        if return_token_scores and return_attention:
            return sequence_value, token_scores, attn_weights
        elif return_token_scores:
            return sequence_value, token_scores
        elif return_attention:
            return sequence_value, attn_weights
        else:
            return sequence_value
    
    def load_state_dict(self, state_dict):
        # Handle loading the state dict with or without cloned attention
        if 'cloned_attention' in state_dict and self.cloned_attention is not None:
            self.cloned_attention.load_state_dict(state_dict['cloned_attention'])
            
        self.embeds.load_state_dict(state_dict['embeds'])
        self.self_attention.load_state_dict(state_dict['self_attention'])
        self.token_scores.load_state_dict(state_dict['token_scores'])
    
    def state_dict(self):
        state_dict = {}
        state_dict['embeds'] = self.embeds.state_dict()
        state_dict['self_attention'] = self.self_attention.state_dict()
        state_dict['token_scores'] = self.token_scores.state_dict()
        
        # Include cloned attention in state dict if it exists
        if self.cloned_attention is not None:
            state_dict['cloned_attention'] = self.cloned_attention.state_dict()
            
        return state_dict



