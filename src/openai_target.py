"""
OpenAI API Target Model Integration

This module provides seamless integration with OpenAI's API for use as a target model,
maintaining exact compatibility with the local model interface and supporting concurrent processing.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import json
import openai
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class OpenAITargetModel:
    """
    OpenAI API wrapper that provides the same interface as local target models.
    
    This class ensures that OpenAI API calls receive exactly the same conversation
    format and parameters that would be sent to a local model, enabling seamless
    switching between local and API-based target models.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        debug_mode: bool = False,
        batch_size: int = 16,
        max_concurrent_requests: int = 10
    ):
        """
        Initialize OpenAI target model.
        
        Args:
            model_name: OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            debug_mode: Whether to enable debug logging
            batch_size: Batch size for API calls
            max_concurrent_requests: Maximum number of concurrent synchronous requests
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # If still no API key, try to load from openai_api.json file
        if not api_key:
            try:
                api_key_path = "openai_api.json"
                if os.path.exists(api_key_path):
                    with open(api_key_path, 'r') as f:
                        api_config = json.load(f)
                        api_key = api_config.get("gpt_key")
                        if api_key:
                            print(f"Loaded OpenAI API key from {api_key_path}")
            except Exception as e:
                print(f"Warning: Could not load API key from openai_api.json: {e}")
        
        if not api_key:
            raise ValueError("OpenAI API key must be provided via api_key parameter or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        
        # Rate limiting
        self.requests_per_minute = 3500  # Conservative limit for most models
        self.tokens_per_minute = 200000  # Conservative limit
        self.request_times = []
        
        # Initialize frozen Llama model for perplexity computations
        self._init_perplexity_model()
        
        if self.debug_mode:
            print(f"OpenAI Target Model initialized: {model_name}")
            print(f"Max tokens: {max_new_tokens}, Temperature: {temperature}")
            print(f"Concurrent requests: {max_concurrent_requests}")
    
    def _init_perplexity_model(self):
        """Initialize a frozen Llama model for perplexity computations."""
        try:
            print("Initializing frozen Llama model for perplexity computations...")
            
            # Determine GPU placement strategy for multi-GPU setups
            # Strategy: GPU 0 for agent models (training), GPU 1 for inference models (target + guard + perplexity)
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            perplexity_prefer_gpu = 1 if num_gpus > 1 else None
            
            if perplexity_prefer_gpu is not None:
                print(f"Multi-GPU setup detected: targeting PERPLEXITY model for GPU {perplexity_prefer_gpu} (inference GPU)")
            
            # Load tokenizer
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
            if self.perplexity_tokenizer.pad_token is None:
                self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
            
            # Load model with device map for efficient memory usage
            if perplexity_prefer_gpu is not None and perplexity_prefer_gpu < num_gpus:
                # Place entire model on specified GPU
                device_map = {f"": perplexity_prefer_gpu}
                print(f"Placing perplexity model on GPU {perplexity_prefer_gpu}")
            else:
                # Auto placement across available GPUs
                device_map = "auto"
                print(f"Using auto device placement for perplexity model across {num_gpus} GPUs")
            
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            
            # Freeze the model for perplexity computations
            for param in self.perplexity_model.parameters():
                param.requires_grad = False
            
            self.perplexity_model.eval()
            
            # Log device placement information
            if hasattr(self.perplexity_model, 'hf_device_map'):
                print(f"Perplexity model device placement: {self.perplexity_model.hf_device_map}")
            else:
                try:
                    device = next(self.perplexity_model.parameters()).device
                    print(f"Perplexity model placed on device: {device}")
                except StopIteration:
                    print(f"Perplexity model has no parameters to check device placement")
            
            print("Successfully initialized frozen Llama model for perplexity computations")
            
        except Exception as e:
            print(f"Warning: Failed to initialize perplexity model: {e}")
            print("Perplexity computations will not be available")
            self.perplexity_model = None
            self.perplexity_tokenizer = None
    
    def compute_perplexity(self, texts: List[str]) -> torch.Tensor:
        """
        Compute perplexity of texts using the frozen Llama model.
        
        Args:
            texts: List of texts to compute perplexity for
            
        Returns:
            Perplexity scores as a tensor
        """
        if self.perplexity_model is None or self.perplexity_tokenizer is None:
            # Return neutral perplexity scores if model is not available
            return torch.ones(len(texts), device="cpu") * 1.0
        
        try:
            # Tokenize texts
            inputs = self.perplexity_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to the same device as the model
            device = next(self.perplexity_model.parameters()).device
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            with torch.no_grad():
                # Get logits
                outputs = self.perplexity_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                logits = outputs.logits
                
                # Compute perplexity for each sequence
                perplexities = []
                for i in range(len(texts)):
                    # Get valid tokens (exclude padding)
                    valid_mask = attention_mask[i] == 1
                    if valid_mask.sum() <= 1:  # Need at least 2 tokens for perplexity
                        perplexities.append(1.0)
                        continue
                    
                    # Get logits for valid positions (excluding the last token as target)
                    valid_logits = logits[i, :-1][valid_mask[:-1]]
                    valid_targets = input_ids[i, 1:][valid_mask[:-1]]
                    
                    if len(valid_targets) == 0:
                        perplexities.append(1.0)
                        continue
                    
                    # Compute log probabilities
                    log_probs = torch.log_softmax(valid_logits, dim=-1)
                    target_log_probs = log_probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
                    
                    # Compute perplexity: exp(-mean(log_prob))
                    mean_log_prob = target_log_probs.mean()
                    perplexity = torch.exp(-mean_log_prob)
                    perplexities.append(perplexity.item())
                
                return torch.tensor(perplexities, device="cpu")
                
        except Exception as e:
            print(f"Error computing perplexity: {e}")
            # Return neutral perplexity scores on error
            return torch.ones(len(texts), device="cpu") * 1.0
    
    def get_log_probs(self, texts: List[str]) -> torch.Tensor:
        """
        Get log probabilities of texts using the frozen Llama model.
        
        Args:
            texts: List of texts to get log probabilities for
            
        Returns:
            Log probability scores as a tensor
        """
        if self.perplexity_model is None or self.perplexity_tokenizer is None:
            # Return neutral log probabilities if model is not available
            return torch.zeros(len(texts), device="cpu")
        
        try:
            # Tokenize texts
            inputs = self.perplexity_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to the same device as the model
            device = next(self.perplexity_model.parameters()).device
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            with torch.no_grad():
                # Get logits
                outputs = self.perplexity_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                logits = outputs.logits
                
                # Compute log probabilities for each sequence
                log_probs_list = []
                for i in range(len(texts)):
                    # Get valid tokens (exclude padding)
                    valid_mask = attention_mask[i] == 1
                    if valid_mask.sum() <= 1:  # Need at least 2 tokens
                        log_probs_list.append(0.0)
                        continue
                    
                    # Get logits for valid positions (excluding the last token as target)
                    valid_logits = logits[i, :-1][valid_mask[:-1]]
                    valid_targets = input_ids[i, 1:][valid_mask[:-1]]
                    
                    if len(valid_targets) == 0:
                        log_probs_list.append(0.0)
                        continue
                    
                    # Compute log probabilities
                    log_probs = torch.log_softmax(valid_logits, dim=-1)
                    target_log_probs = log_probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
                    
                    # Return mean log probability
                    mean_log_prob = target_log_probs.mean()
                    log_probs_list.append(mean_log_prob.item())
                
                return torch.tensor(log_probs_list, device="cpu")
                
        except Exception as e:
            print(f"Error computing log probabilities: {e}")
            # Return neutral log probabilities on error
            return torch.zeros(len(texts), device="cpu")
    
    def _rate_limit_check(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're approaching rate limits
        if len(self.request_times) >= self.requests_per_minute * 0.9:  # 90% of limit
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                print(f"Rate limit approaching, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    def generate_responses(self, chats: List[List[Dict[str, str]]]) -> List[str]:
        """
        Generate responses for a batch of conversations.
        
        This method maintains exact compatibility with local model behavior:
        - Accepts the same chat format
        - Uses the same generation parameters
        - Returns responses in the same format
        
        Args:
            chats: List of conversation histories in OpenAI chat format
                  Each chat is a list of {"role": "user/assistant/system", "content": "..."}
        
        Returns:
            List of generated responses (strings)
        """
        if not chats:
            return []
        
        if self.debug_mode:
            print(f"Generating {len(chats)} responses via OpenAI API")
            for i, chat in enumerate(chats[:2]):  # Show first 2 for debugging
                print(f"Chat {i}: {chat}")
        
        # Use concurrent synchronous calls for real-time performance
        if len(chats) > 1:
            return self._generate_concurrent_responses(chats)
        else:
            return self._generate_sequential_responses(chats)
    
    def _generate_sequential_responses(self, chats: List[List[Dict[str, str]]]) -> List[str]:
        """Generate responses sequentially with rate limiting."""
        responses = []
        
        for i, chat in enumerate(chats):
            self._rate_limit_check()
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=chat,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=1.0,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                generated_text = response.choices[0].message.content or ""
                responses.append(generated_text)
                
                if self.debug_mode:
                    print(f"Response {i}: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
                
            except Exception as e:
                print(f"Error generating response {i}: {e}")
                responses.append("")  # Empty response on error
                
                # Exponential backoff on rate limit errors
                if "rate_limit" in str(e).lower():
                    wait_time = min(60, 2 ** (i % 6))  # Cap at 60 seconds
                    print(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
        
        return responses
    
    def _generate_concurrent_responses(self, chats: List[List[Dict[str, str]]]) -> List[str]:
        """
        Generate responses concurrently using threading for better performance.
        
        This method provides real-time performance by making multiple synchronous
        API calls concurrently, respecting rate limits.
        """
        if self.debug_mode:
            print(f"Using concurrent synchronous calls for {len(chats)} requests")
        
        import threading
        import concurrent.futures
        
        responses = [None] * len(chats)  # Pre-allocate response list
        
        def generate_single_response(index: int, chat: List[Dict[str, str]]) -> None:
            """Generate a single response and store it in the responses list."""
            try:
                self._rate_limit_check()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=chat,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=1.0,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                generated_text = response.choices[0].message.content or ""
                responses[index] = generated_text
                
                if self.debug_mode:
                    print(f"Response {index}: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
                
            except Exception as e:
                print(f"Error generating response {index}: {e}")
                responses[index] = ""  # Empty response on error
                
                # For rate limit errors, we'll let other threads continue
                if "rate_limit" in str(e).lower():
                    print(f"Rate limited on request {index}")
        
        # Use ThreadPoolExecutor to limit concurrent requests
        max_workers = min(self.max_concurrent_requests, len(chats))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(generate_single_response, i, chat): i 
                for i, chat in enumerate(chats)
            }
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as e:
                    print(f"Concurrent request {index} failed: {e}")
                    if responses[index] is None:
                        responses[index] = ""
        
        # Ensure all responses are strings (fallback for any None values)
        responses = [resp if resp is not None else "" for resp in responses]
        
        return responses
    
    # Compatibility methods to match local model interface
    @property
    def device(self):
        """Return a dummy device for compatibility."""
        return torch.device("cpu")  # API models don't have a device
    
    def to(self, device):
        """No-op for compatibility with local model interface."""
        return self
    
    def eval(self):
        """No-op for compatibility with local model interface."""
        return self
    
    def parameters(self):
        """Return empty parameters for compatibility."""
        return []
    
    def get_perplexity_model(self):
        """Get the frozen Llama model used for perplexity computations."""
        return self.perplexity_model
    
    def get_perplexity_tokenizer(self):
        """Get the tokenizer used for perplexity computations."""
        return self.perplexity_tokenizer
    
    def has_perplexity_model(self):
        """Check if perplexity model is available."""
        return self.perplexity_model is not None and self.perplexity_tokenizer is not None


class OpenAITargetTokenizer:
    """
    Dummy tokenizer for OpenAI models that provides compatibility with local tokenizer interface.
    
    Since OpenAI API accepts text directly, this tokenizer is mainly for interface compatibility.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.eos_token_id = 0  # Dummy value for compatibility
        self.pad_token_id = 0  # Dummy value for compatibility
    
    def apply_chat_template(
        self, 
        conversation: List[Dict[str, str]], 
        tokenize: bool = False, 
        add_generation_prompt: bool = True
    ) -> Union[str, List[int]]:
        """
        Apply chat template - for OpenAI, we return the conversation as-is since the API handles formatting.
        
        Args:
            conversation: List of message dictionaries
            tokenize: Whether to return token IDs (ignored for OpenAI)
            add_generation_prompt: Whether to add generation prompt (ignored for OpenAI)
        
        Returns:
            The conversation itself for OpenAI API compatibility
        """
        if tokenize:
            # For compatibility, return dummy token IDs
            return [1] * 10  # Dummy tokens
        else:
            # Return a formatted string for debugging/logging purposes
            formatted = ""
            for msg in conversation:
                role = msg["role"].upper()
                content = msg["content"]
                formatted += f"{role}: {content}\n"
            return formatted
    
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Dummy decode method for compatibility."""
        return ""  # OpenAI API doesn't use token decoding
    
    def __call__(self, *args, **kwargs):
        """Dummy tokenization for compatibility."""
        return type('obj', (object,), {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        })


def create_openai_target_model(config: Any) -> tuple[OpenAITargetModel, OpenAITargetTokenizer]:
    """
    Factory function to create OpenAI target model and tokenizer from config.
    
    Args:
        config: Environment instance with all necessary attributes
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Extract parameters from environment instance
    model_name = getattr(config, 'target_model_name', 'gpt-4o-mini')
    max_new_tokens = getattr(config, 'max_new_tokens', 100)
    debug_mode = getattr(config, 'debug_mode', False)
    batch_size = getattr(config, 'batch_size', 16)
    
    # Create model and tokenizer
    model = OpenAITargetModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        debug_mode=debug_mode,
        batch_size=batch_size,
        max_concurrent_requests=10
    )
    
    tokenizer = OpenAITargetTokenizer(model_name=model_name)
    
    return model, tokenizer 