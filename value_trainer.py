import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch.nn.functional as F
import torch.optim as optim
import glob
import random
import logging
import time
from datetime import datetime
from models import ValueTransformer
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("value_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ValueTrainer:
    """
    ValueTrainer specializes in training the ValueTransformer model for harm prediction.
    
    This class focuses on training the harm critic in isolation, without the overhead
    of the full agent and environment setup. It loads data from HarmBench (harmful examples)
    and OASST1 (safe examples) for binary classification training.
    
    Key features:
    - Token-level masking for precise harm attribution
    - Attention divergence mechanism to improve token-level differentiation
    - LoRA adapter support for efficient fine-tuning of specific layers
    - Batch-wise training with gradient accumulation
    - Validation and early stopping
    """
    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_name=None,
        checkpoints_dir="checkpoints/harm_critic",
        out_dir=None,  # New parameter for output directory
        token_level=True,
        batch_size=8,
        accumulation_steps=4,
        learning_rate=2e-5,
        attention_divergence_weight=0.1,
        device=None,
        load_checkpoint=None,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        target_layers=[0],  # Default to first layer only
        lora_bias="none",
        hf_token=None,
        use_cpu_for_base_model=False
    ):
        """
        Initialize the ValueTrainer.
        
        Args:
            model_name: Base model name/path for the ValueTransformer
            tokenizer_name: Tokenizer name/path (defaults to model_name if None)
            checkpoints_dir: Base directory to save model checkpoints
            out_dir: Specific output directory for this run (if None, uses checkpoints_dir)
            token_level: Whether to use token-level value prediction
            batch_size: Training batch size
            accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Learning rate for optimizer
            attention_divergence_weight: Weight for attention divergence loss
            device: Device to use (defaults to cuda if available)
            load_checkpoint: Path to checkpoint to load (if any)
            use_lora: Whether to use LoRA adapters
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: List of module types to apply LoRA to
            target_layers: List of attention layer indices to apply LoRA to
            lora_bias: LoRA bias type ("none", "all", or "lora_only")
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        
        # Set up output directory
        if out_dir is not None:
            self.checkpoints_dir = os.path.join(checkpoints_dir, out_dir)
        else:
            self.checkpoints_dir = checkpoints_dir
        
        self.token_level = token_level
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.attention_divergence_weight = attention_divergence_weight
        
        # LoRA parameters
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.target_layers = target_layers
        self.lora_bias = lora_bias
        self.hf_token = hf_token
        self.use_cpu_for_base_model = use_cpu_for_base_model
        self.is_nemo = "nemotron" in self.model_name.lower()
        # Initialize device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Create checkpoints directory
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.checkpoints_dir}")
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, token=self.hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.initialize_model(load_checkpoint)

        memory_footprint = torch.cuda.memory_allocated()
        print(f"Memory footprint: {memory_footprint / 1024 / 1024 / 1024:.2f} GB")
        
    def initialize_model(self, load_checkpoint=None):
        """
        Initialize or load the ValueTransformer model with optional LoRA adapters.
        
        Args:
            load_checkpoint: Path to checkpoint to load (if any)
        """
        logger.info(f"Initializing ValueTransformer with base model: {self.model_name}")

        # --- Quantization and Device Setup ---
        quantization_config = None
        device_map_config = None
        model_dtype = torch.float32 # Default

        if "70b" in self.model_name.lower():
            logger.info("70B model detected, enabling 8-bit quantization.")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True # Allow offloading non-quantized modules
            )
            device_map_config = "auto" # Recommended for quantized models with potential offloading
            model_dtype = torch.bfloat16 # bf16 is often used with 8-bit quantization
            logger.info(f"Quantization: {quantization_config}, Device Map: {device_map_config}, Dtype: {model_dtype}")
        elif self.use_cpu_for_base_model:
            logger.info("CPU offloading requested for base model.")
            base_model_device = 'cpu'
            device_map_config = {'': base_model_device}
            model_dtype = torch.float32 # Keep float32 for CPU
            logger.info(f"Device Map: {device_map_config}, Dtype: {model_dtype}")
        else:
            logger.info("Loading base model onto target device.")
            device_map_config = {'': self.device}
            model_dtype = torch.float16 if self.device != torch.device('cpu') else torch.float32 # Use float16 on GPU if available
            logger.info(f"Device Map: {device_map_config}, Dtype: {model_dtype}")

        # Store model dtype for later use
        self.model_dtype = model_dtype

        # --- Load Base HuggingFace Model (potentially on CPU or Quantized) ---
        base_hf_model = None
        # If loading from checkpoint, we might need to handle it differently
        if load_checkpoint and os.path.exists(load_checkpoint):
            logger.info(f"Loading model from checkpoint: {load_checkpoint}")
            checkpoint_data = torch.load(load_checkpoint, map_location=self.device) # Load base to target device

            # Determine base model name from checkpoint if possible
            base_model_name_from_ckpt = checkpoint_data.get('base_model_name', checkpoint_data.get('config', {}).get('model_name', self.model_name))

            # Load the base HF model structure first
            config = AutoConfig.from_pretrained(base_model_name_from_ckpt, token=self.hf_token)
            config.output_attentions = True # Ensure attentions enabled

            # Apply quantization config if it exists (it won't if loading from checkpoint usually)
            load_quant_config = checkpoint_data.get('quantization_config', quantization_config)

            base_hf_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_from_ckpt,
                config=config,
                token=self.hf_token,
                # Load initially to the designated base model device
                quantization_config=load_quant_config,
                device_map=device_map_config,
                torch_dtype=model_dtype
            )

            # Now, initialize the ValueTransformer wrapper with the loaded base model
            self.value_model = ValueTransformer(
                hf_checkpoint=base_model_name_from_ckpt, # Pass name for reference
                tokenizer=self.tokenizer,
                model=base_hf_model, # Pass the loaded base model
                token_level=self.token_level,
                attention_update=True,
                token=self.hf_token
            )

            # Apply LoRA or state dict from checkpoint *after* wrapper initialization
            if self.use_lora and 'lora_config' in checkpoint_data:
                logger.info("Loading LoRA adapter weights from checkpoint")
                lora_config = LoraConfig(**checkpoint_data['lora_config'])
                # Apply LoRA to the base model *within* the wrapper
                self.value_model.model = get_peft_model(self.value_model.model, lora_config)
                # Load adapter weights
                self.value_model.model.load_state_dict(checkpoint_data['adapter_model'], strict=False)
                logger.info(f"Loaded LoRA adapter weights for target modules: {lora_config.target_modules}")

            elif 'state_dict' in checkpoint_data:
                # Handle loading full state dict (might need adjustments for PEFT structure)
                logger.info("Loading full state dict from checkpoint")
                # This assumes the state_dict keys match the ValueTransformer structure
                # Might need specific handling if keys only match base model
                self.value_model.load_state_dict(checkpoint_data['state_dict'], strict=False)

            else:
                logger.warning("Checkpoint format not fully recognized (neither LoRA nor state_dict found), proceeding with base structure.")

        else:
            # --- Load New Base Model --- 
            logger.info(f"Creating new base model: {self.model_name} on {self.device}")
            config = AutoConfig.from_pretrained(self.model_name, token=self.hf_token)
            config.output_attentions = True
            base_hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                token=self.hf_token,
                quantization_config=quantization_config,
                device_map=device_map_config,
                torch_dtype=model_dtype
            )

            # Initialize the ValueTransformer wrapper
            self.value_model = ValueTransformer(
                hf_checkpoint=self.model_name,
                tokenizer=self.tokenizer,
                model=base_hf_model, # Pass the loaded base model
                token_level=self.token_level,
                attention_update=True,
                token=self.hf_token
            )

            # Apply new LoRA if enabled
            if self.use_lora:
                logger.info(f"Applying new LoRA adapter with rank {self.lora_r}")
                lora_config = self._create_lora_config()
                # Apply LoRA to the base model *within* the wrapper
                self.value_model.model = get_peft_model(self.value_model.model, lora_config)
                logger.info(f"Applied LoRA to target modules: {lora_config.target_modules} on layers: {self.target_layers}")
            else:
                logger.info("Using full-parameter fine-tuning (LoRA disabled)")

        # --- Enable Gradient Checkpointing --- 
        # This should be done AFTER applying LoRA adapters if use_lora is True
        # It helps save memory by recomputing activations during backward pass
        if hasattr(self.value_model.model, 'gradient_checkpointing_enable'):
            logger.info("Enabling gradient checkpointing for the model to save memory.")
            # use_reentrant=True is often the default and compatible with most setups
            # set use_reentrant=False if encountering issues with specific layers/hooks
            self.value_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True}) 
        else:
            logger.warning("Model does not support gradient_checkpointing_enable(). Activation memory might be high.")

        # --- Cast Projection Head/Pooler to Correct Dtype --- 
        # Ensure the projection head matches the base model's output dtype
        if hasattr(base_hf_model, 'dtype'):
            expected_dtype = base_hf_model.dtype
            logger.info(f"Ensuring projection head/pooler uses dtype: {expected_dtype}")
            if self.value_model.output_projection is not None:
                self.value_model.output_projection = self.value_model.output_projection.to(dtype=expected_dtype)
            if self.value_model.sequence_pooler is not None:
                self.value_model.sequence_pooler = self.value_model.sequence_pooler.to(dtype=expected_dtype)
        else:
            logger.warning("Could not determine base model dtype, projection head dtype might mismatch.")

        # --- Device Placement for Trainable Parts ---
        logger.info(f"Moving trainable parts (LoRA adapters / projection head) to {self.device}")
        if self.use_lora:
            # Move only LoRA parameters to the target device (GPU)
            for name, param in self.value_model.model.named_parameters():
                if 'lora' in name.lower():
                    param.to(self.device)

        # Move the output projection head or sequence pooler to the target device (GPU)
        if self.value_model.output_projection is not None:
            self.value_model.output_projection.to(self.device)
        if self.value_model.sequence_pooler is not None:
            self.value_model.sequence_pooler.to(self.device)

        # Verify final device placement (optional debug)
        # self._log_device_placement()

        # Ensure model is configured to output attention weights (redundant if set in config)
        if hasattr(self.value_model, 'config'):
            if hasattr(self.value_model.config, 'output_attentions'):
                self.value_model.config.output_attentions = True
            if hasattr(self.value_model, 'model') and hasattr(self.value_model.model, 'config'):
                self.value_model.model.config.output_attentions = True
        
        if self.use_lora:
            # Configure LoRA to target attention layers
            target_modules = [f"q_proj", "v_proj"]  # Target query and value projections
            
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                task_type="SEQ_CLS"
            )
            
            # Apply LoRA configuration
            self.value_model.model = get_peft_model(self.value_model.model, lora_config)
            
            # Verify trainable parameters
            trainable_params = [(n, p.numel()) for n, p in self.value_model.model.named_parameters() if p.requires_grad]
            logger.info(f"Trainable parameters:")
            for name, num in trainable_params:
                logger.info(f"  {name}: {num:,}")
            
            # Ensure output projection is trainable
            for param in self.value_model.output_projection.parameters():
                param.requires_grad = True

        # Initialize optimizer ONLY with parameters on the target device (GPU)
        trainable_params = [p for p in self.value_model.parameters() if p.requires_grad and p.device == self.device]
        if not trainable_params:
             logger.warning("No trainable parameters found on the target device! Check LoRA application and projection head.")
             # Fallback: include all trainable params, hoping for the best
             trainable_params = [p for p in self.value_model.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(trainable_params, lr=self.learning_rate)
        logger.info(f"Optimizer initialized with {len(trainable_params)} parameters on device {self.device}")
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.value_model.parameters())
        trainable_params = sum(p.numel() for p in self.value_model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # Log memory usage details
        total_params = sum(p.numel() for p in self.value_model.parameters())
        trainable_params = sum(p.numel() for p in self.value_model.parameters() if p.requires_grad)
        memory_usage = torch.cuda.memory_allocated() / 1024**3
        
        logger.info(f"Memory usage after initialization: {memory_usage:.2f} GB")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    def _create_lora_config(self):
        """
        Create LoRA configuration object based on instance parameters.
        
        Returns:
            LoraConfig: Configuration for LoRA adapters
        """
        # Prepare module_mapping if using target_layers
        if self.target_layers:
            logger.info(f"Targeting specific layers for LoRA: {self.target_layers}")
            target_modules = [f"model.layers.{layer}.self_attn.{module}" for layer in self.target_layers for module in self.target_modules]
        else:
            # Apply to all layers if no specific layers provided
            target_modules = self.target_modules
            
        logger.info(f"LoRA target modules: {target_modules}")
        
        # Create LoRA configuration
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            task_type="SEQ_CLS"
        )
    
    def load_harmbench_data(self, num_samples=None):
        """
        Load harmful examples from HarmBench dataset.
        
        Args:
            num_samples: Number of samples to load (if None, load all)
            
        Returns:
            List of harmful examples
        """
        logger.info("Loading HarmBench data...")
        
        # Find all CSV files in the HarmBench directory that don't have "test" or "val" in the filename
        csv_files = glob.glob("HarmBench/*.csv")
        csv_files = [f for f in csv_files if "test" not in f.lower() and "val" not in f.lower()]
        
        if not csv_files:
            raise FileNotFoundError("No suitable CSV files found in HarmBench directory")
        
        logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        # Load and combine all dataframes
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                logger.info(f"Loaded {file} with {len(df)} rows")
                
                # Make sure the required columns exist
                if "Behavior" not in df.columns:
                    logger.warning(f"'Behavior' column not found in {file}, skipping")
                    continue
                    
                # If ContextString exists, append it to Behavior
                if "ContextString" in df.columns:
                    # Create a new column for the combined behavior and context
                    df['CombinedBehavior'] = df['Behavior'].copy()
                    
                    # Count how many contexts we're processing
                    context_count = 0
                    
                    # Only combine if ContextString is not empty/null
                    for i, row in df.iterrows():
                        if pd.notna(row['ContextString']) and row['ContextString'].strip():
                            context = row['ContextString'].strip()
                            # Remove surrounding quotes if they exist
                            if (context.startswith('"') and context.endswith('"')) or \
                               (context.startswith("'") and context.endswith("'")):
                                context = context[1:-1]
                            
                            # Add a space between behavior and context if needed
                            if not row['CombinedBehavior'].endswith(' '):
                                df.at[i, 'CombinedBehavior'] = f"{row['CombinedBehavior']} {context}"
                            else:
                                df.at[i, 'CombinedBehavior'] = f"{row['CombinedBehavior']}{context}"
                            context_count += 1
                    
                    logger.info(f"  - Processed {context_count} context strings in {file}")
                    
                    # Use the combined column as the behavior
                    df['Behavior'] = df['CombinedBehavior']
                    df = df.drop('CombinedBehavior', axis=1)
                
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not dfs:
            raise ValueError("No valid data found in HarmBench CSV files")
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        harmbench_examples = combined_df['Behavior'].tolist()
        logger.info(f"Loaded {len(harmbench_examples)} total samples from HarmBench")
        
        # If we don't need more examples or not specifying a limit, just return what we have
        if num_samples is None or len(harmbench_examples) >= num_samples:
            if num_samples is not None and num_samples < len(harmbench_examples):
                # Randomly sample the requested number
                sampled_examples = random.sample(harmbench_examples, num_samples)
                logger.info(f"Randomly sampled {len(sampled_examples)} examples from HarmBench")
                return sampled_examples
            return harmbench_examples
        
        # If we need more examples than HarmBench provides, supplement with LibrAI examples
        additional_needed = num_samples - len(harmbench_examples)
        logger.info(f"HarmBench only provided {len(harmbench_examples)} examples, need {additional_needed} more. Loading from LibrAI/do-not-answer dataset.")
        
        # Load additional harmful examples from LibrAI/do-not-answer
        librai_examples = self.load_librai_do_not_answer_data(additional_needed)
        
        # Combine HarmBench and LibrAI examples
        combined_examples = harmbench_examples + librai_examples
        logger.info(f"Combined {len(harmbench_examples)} HarmBench examples with {len(librai_examples)} LibrAI examples for a total of {len(combined_examples)} harmful examples")
        
        return combined_examples

    def load_librai_do_not_answer_data(self, num_samples=None):
        """
        Load harmful examples from the LibrAI/do-not-answer dataset.
        
        Args:
            num_samples: Number of samples to load (if None, load all)
            
        Returns:
            List of harmful examples
        """
        logger.info("Loading LibrAI/do-not-answer data...")
        
        try:
            # Load the dataset
            dataset = load_dataset("LibrAI/do-not-answer", split="train")
            logger.info(f"Loaded LibrAI/do-not-answer dataset with {len(dataset)} examples")
            
            # Extract the 'question' field which contains the harmful content
            examples = [item['question'] for item in dataset]
            
            # Limit to the requested number of samples if specified
            if num_samples is not None and num_samples < len(examples):
                examples = random.sample(examples, num_samples)
                logger.info(f"Randomly sampled {num_samples} examples from LibrAI/do-not-answer")
            
            logger.info(f"Loaded {len(examples)} total samples from LibrAI/do-not-answer")
            return examples
            
        except Exception as e:
            logger.error(f"Error loading LibrAI/do-not-answer data: {e}")
            logger.warning("Continuing with only HarmBench examples due to error loading LibrAI data")
            return []
    
    def load_oasst1_data(self, num_samples=None):
        """
        Load safe examples from OASST1 dataset, filtering for English language only.
        
        Args:
            num_samples: Number of samples to load (if None, load all)
            
        Returns:
            List of safe examples
        """
        logger.info("Loading OASST1 data (English only)...")
        
        try:
            # Load the OASST1 dataset
            dataset = load_dataset("OpenAssistant/oasst1", split="train")
            
            # Filter for initial user messages (root nodes) in English
            root_messages = [
                item for item in dataset 
                if item['role'] == 'prompter' and 
                item['parent_id'] is None and 
                item['lang'] == 'en'  # Filter for English language
            ]
            
            logger.info(f"Found {len(root_messages)} English root messages in OASST1")
            
            # Extract message text
            messages = [item['text'] for item in root_messages]
            
            # Limit to the requested number of samples if specified
            if num_samples is not None and num_samples < len(messages):
                messages = random.sample(messages, num_samples)
                logger.info(f"Randomly sampled {num_samples} English examples from OASST1")
            
            logger.info(f"Loaded {len(messages)} total English samples from OASST1")
            return messages
            
        except Exception as e:
            logger.error(f"Error loading OASST1 data: {e}")
            raise e
    
    def prepare_datasets(self, harmful_samples=1000, safe_samples=1000, val_split=0.1):
        """
        Prepare training and validation datasets from HarmBench and OASST1.
        
        Args:
            harmful_samples: Number of harmful samples to use
            safe_samples: Number of safe samples to use
            val_split: Proportion of data to use for validation
            
        Returns:
            train_data, val_data: Lists of (text, label) tuples
        """
        logger.info(f"Preparing datasets with {harmful_samples} harmful and {safe_samples} safe samples")
        
        # Load harmful examples (label 1)
        harmful_examples = self.load_harmbench_data(harmful_samples)
        harmful_data = [(text, 1.0) for text in harmful_examples]
        
        # Load safe examples (label -1 instead of 0)
        safe_examples = self.load_oasst1_data(safe_samples)
        safe_data = [(text, -1.0) for text in safe_examples]  # Changed from 0.0 to -1.0
        
        # Combine and shuffle
        all_data = harmful_data + safe_data
        random.shuffle(all_data)
        
        # Split into training and validation
        val_size = int(len(all_data) * val_split)
        train_data = all_data[val_size:]
        val_data = all_data[:val_size]
        
        logger.info(f"Created training set with {len(train_data)} examples and validation set with {len(val_data)} examples")
        logger.info(f"Labels: Harmful = 1.0, Safe = -1.0")
        
        return train_data, val_data
    
    def tokenize_batch(self, texts):
        """
        Tokenize a batch of text examples.
        
        Args:
            texts: List of text strings
            
        Returns:
            BatchEncoding with input_ids and attention_mask
        """
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
    
    def value_step(self, text_batch, labels):
        """
        Perform a training step with token masking and attention divergence.
        Uses gradient accumulation for token-level computations to reduce memory usage.
        """
        self.optimizer.zero_grad()
        
        batch_encoding = self.tokenize_batch(text_batch)
        input_ids = batch_encoding.input_ids
        attention_mask = batch_encoding.attention_mask
        batch_size, seq_len = input_ids.shape
        
        # Use the stored model dtype for labels tensor
        labels_tensor = torch.tensor(labels, dtype=self.model_dtype, device=self.device).view(-1, 1)
        if self.is_nemo:
            labels_tensor = labels_tensor * -1

        # Wrap forward pass and calculations in autocast
        with autocast(dtype=self.model_dtype):
            with torch.no_grad():
                base_values, _, base_attention = self.value_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_token_scores=True,
                    return_attention=True
                )
                base_values = base_values.sum(dim=1)
        
        # Process tokens in smaller chunks with gradient accumulation
        token_chunk_size = 4  # Number of tokens to process at once
        token_loss = 0
        
        for b in range(batch_size):
            seq_len_b = torch.sum(attention_mask[b]).item()
            token_targets = labels_tensor[b] / seq_len_b
            
            for chunk_start in range(0, seq_len_b, token_chunk_size):
                chunk_end = min(chunk_start + token_chunk_size, seq_len_b)
                chunk_size = chunk_end - chunk_start
                
                if chunk_size <= 0:
                    continue
                
                # Create masked inputs for this chunk
                token_masks = attention_mask[b:b+1].repeat(chunk_size, 1)
                chunk_mask = torch.ones((chunk_size, seq_len), device=self.device)
                for i in range(chunk_size):
                    chunk_mask[i, chunk_start + i] = 0
                
                token_masks = token_masks * chunk_mask
                extended_input_ids = input_ids[b:b+1].repeat(chunk_size, 1)
                
                # Forward pass with masked tokens
                with autocast(dtype=self.model_dtype):
                    masked_values, _, masked_attention = self.value_model(
                        input_ids=extended_input_ids,
                        attention_mask=token_masks,
                        return_token_scores=True,
                        return_attention=True,
                        requires_grad=True
                    )
                    masked_values = masked_values.sum(dim=1)
                
                # Handle dimension issues
                if masked_values.dim() == 0:
                    masked_values = masked_values.unsqueeze(0)
                
                # Calculate token contributions
                diff = base_values[b] - masked_values
                
                # Calculate attention divergence
                input_tensor = F.log_softmax(masked_attention, dim=-1)
                target_tensor = F.softmax(base_attention[b:b+1].expand_as(masked_attention), dim=-1)
                div = F.kl_div(
                    input_tensor,
                    target_tensor + 1e-8,
                    reduction='none'
                )
                
                # Calculate divergence scores
                token_divergences = div.mean(dim=[-1, -2]).mean(dim=-1)
                
                # Calculate weighted contributions for this chunk
                weighted_contributions = diff.squeeze(-1) * F.softmax(token_divergences, dim=-1)
                assert weighted_contributions.shape == (chunk_size, 1)

                # Calculate loss for this chunk (inside autocast)
                with autocast(dtype=self.model_dtype):
                    valid_tokens = torch.ones(chunk_size, device=self.device).bool()
                    chunk_token_loss = F.mse_loss(
                        weighted_contributions[valid_tokens],
                        torch.ones(valid_tokens.sum(), device=self.device, dtype=self.model_dtype) * token_targets
                    )

                # Backward pass for this chunk (outside autocast)
                chunk_token_loss.backward()
                
                # Add to total token loss
                token_loss += chunk_token_loss.item()
                
                # Clean up to save memory
                del masked_values, masked_attention, diff, div, token_masks, chunk_mask
                del weighted_contributions, chunk_token_loss
                torch.cuda.empty_cache()
        
        # Average token loss
        token_loss = token_loss / batch_size
        
        # Process value loss in smaller chunks
        minibatch_size = 4
        value_loss = 0
        
        for i in range(0, batch_size, minibatch_size):
            end_idx = min(i + minibatch_size, batch_size)
            sub_input_ids = input_ids[i:end_idx]
            sub_attention_mask = attention_mask[i:end_idx]
            sub_labels = labels_tensor[i:end_idx]
            
            # Forward pass for sub-batch
            with autocast(dtype=self.model_dtype):
                sub_values = self.value_model(
                    input_ids=sub_input_ids,
                    attention_mask=sub_attention_mask,
                    requires_grad=True,
                )
            
            # Add clamping
            sub_values = torch.clamp(sub_values, min=sub_labels.min(), max=sub_labels.max())
            
            # Calculate loss for sub-batch (inside autocast)
            sub_value_loss = F.mse_loss(sub_values, sub_labels.expand_as(sub_values))

            # Scale the loss by the ratio of sub-batch to full batch
            scaled_loss = sub_value_loss * (end_idx - i) / batch_size
        
            # Backward pass (outside autocast)
            scaled_loss.backward()
            
            value_loss += sub_value_loss.item()
            
            del sub_values, sub_value_loss, scaled_loss
            torch.cuda.empty_cache()
        
        value_loss = value_loss * minibatch_size / batch_size
        self.optimizer.step()
        
        return value_loss, token_loss
    
    def train_epoch(self, train_data, report_interval=10):
        """
        Train the model for one epoch.
        
        Args:
            train_data: List of (text, label) tuples
            report_interval: Number of steps between reporting
            
        Returns:
            average_loss, average_value_loss, average_token_loss
        """
        self.value_model.train()
        total_loss = 0.0
        total_value_loss = 0.0
        total_token_loss = 0.0
        
        # Create batches
        num_batches = (len(train_data) + self.batch_size - 1) // self.batch_size
        
        # Track start time
        start_time = time.time()
        
        for batch_idx in tqdm(range(num_batches), desc="Training batches"):
            # Get batch
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(train_data))
            batch = train_data[batch_start:batch_end]
            
            texts = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            value_loss, token_loss = self.value_step(texts, labels)
            
            # Calculate combined loss (but don't call backward again)
            loss = value_loss + token_loss
            
            # Track losses
            total_loss += loss
            total_value_loss += value_loss
            total_token_loss += token_loss
            
            # Report progress
            if (batch_idx + 1) % report_interval == 0:
                elapsed = time.time() - start_time
                examples_per_sec = (batch_idx + 1) * self.batch_size / elapsed
                print(
                    f"Batch {batch_idx+1}/{num_batches}, "
                    f"Loss: {total_loss/(batch_idx+1):.4f}, "
                    f"Value: {total_value_loss/(batch_idx+1):.4f}, "
                    f"Token: {total_token_loss/(batch_idx+1):.4f}"
                )
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_token_loss = total_token_loss / num_batches
        return avg_loss, avg_value_loss, avg_token_loss
    
    def validate(self, val_data):
        """
        Validate the model on validation data.
        
        Args:
            val_data: List of (text, label) tuples
            
        Returns:
            validation_loss
        """
        self.value_model.eval()
        total_loss = 0.0
        
        # Create batches
        num_batches = (len(val_data) + self.batch_size - 1) // self.batch_size
        print(f"Memory before validation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Validation batches"):
                # Get batch
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(val_data))
                batch = val_data[batch_start:batch_end]
                
                texts = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                
                # Tokenize
                batch_encoding = self.tokenize_batch(texts)
                
                # Convert labels to tensor
                labels_tensor = torch.tensor(labels, dtype=self.model_dtype, device=self.device).view(-1, 1)
                
                # Forward pass (inside autocast)
                with autocast(dtype=self.model_dtype):
                    values = self.value_model(
                        input_ids=batch_encoding.input_ids,
                        attention_mask=batch_encoding.attention_mask
                    )
                
                # Ensure we're summing along the right dimension if values is 3D
                if values.dim() > 2:
                    values = values.sum(dim=1)
                
                # Compute loss (inside autocast)
                loss = F.mse_loss(values, labels_tensor.expand_as(values))
                
                # Track loss (outside autocast)
                total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def evaluate_token_contributions(self, example_texts, true_labels=None):
        """
        Evaluate token contributions on example texts.
        
        Args:
            example_texts: List of example texts to analyze
            true_labels: Optional true labels for these examples
            
        Returns:
            results: List of dicts with token contribution analysis
        """
        self.value_model.eval()
        results = []
        
        for i, text in enumerate(example_texts):
            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            input_ids = encoding.input_ids
            attention_mask = encoding.attention_mask
            seq_len = input_ids.shape[1]
            
            # Get base value prediction
            with torch.no_grad():
                base_value, token_scores = self.value_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_token_scores=True
                )
                
                # Handle batch dimension - sum if needed
                if base_value.dim() > 1:
                    base_value = base_value.sum(dim=1)
                
                # Record base prediction
                base_pred = base_value.item()
                
                # Analyze token contributions by masking each token
                token_contributions = []
                token_texts = []
                
                for token_idx in range(1, seq_len):  # Skip first token
                    if attention_mask[0, token_idx].item() == 0:  # Skip padding
                        continue
                        
                    # Create copy with token masked
                    token_mask = attention_mask.clone()
                    token_mask[0, token_idx] = 0
                    
                    # Get value with masked token
                    masked_value = self.value_model(
                        input_ids=input_ids,
                        attention_mask=token_mask
                    )
                    
                    # Handle batch dimension for masked value
                    if masked_value.dim() > 1:
                        masked_value = masked_value.sum(dim=1)
                    
                    # Contribution is the difference in value
                    contribution = base_pred - masked_value.item()
                    
                    # Decode the token
                    token_text = self.tokenizer.decode(input_ids[0, token_idx])
                    
                    token_contributions.append(contribution)
                    token_texts.append(token_text)
                
                # Sort by contribution (highest first)
                sorted_indices = np.argsort(-np.array(token_contributions))
                
                # Prepare result
                result = {
                    "text": text,
                    "prediction": base_pred,
                    "true_label": true_labels[i] if true_labels is not None else None,
                    "tokens": [token_texts[idx] for idx in sorted_indices],
                    "contributions": [token_contributions[idx] for idx in sorted_indices]
                }
                
                results.append(result)
        
        return results
    
    def save_checkpoint(self, epoch, loss, is_best):
        """
        Save a checkpoint of the model, keeping only the latest and best models.

        Args:
            epoch: Current epoch number
            loss: Validation loss for the current epoch
            is_best: Boolean indicating if this is the best validation loss so far
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory if it doesn't exist
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Determine if saving LoRA adapters or full model
        if self.use_lora:
            # Save only the LoRA adapter weights and configuration
            adapter_weights = {}
            for name, param in self.value_model.model.named_parameters():
                if param.requires_grad:
                    # Ensure weights are cloned to CPU before saving
                    adapter_weights[name] = param.data.cpu().clone()

            # Also save the output projection layer weights to CPU
            for name, param in self.value_model.output_projection.named_parameters():
                 adapter_weights[f'output_projection.{name}'] = param.data.cpu().clone()


            # Extract LoRA configuration
            if hasattr(self.value_model.model, 'config') and hasattr(self.value_model.model.config, 'peft_config'):
                 # Correct access to peft_config if it's nested or not
                 peft_config = self.value_model.model.config.peft_config
                 if isinstance(peft_config, dict):
                     peft_config = peft_config.get('default', peft_config) # Handle nested dict case

                 # Check if peft_config is now an object with attributes
                 if hasattr(peft_config, 'r'):
                     lora_config_dict = {
                         'r': peft_config.r,
                         'alpha': peft_config.lora_alpha,
                         'dropout': peft_config.lora_dropout,
                         'target_modules': peft_config.target_modules,
                         'bias': peft_config.bias
                     }
                 else: # If peft_config is still a dict after get('default',...)
                     lora_config_dict = {
                         'r': peft_config.get('r', self.lora_r),
                         'alpha': peft_config.get('lora_alpha', self.lora_alpha),
                         'dropout': peft_config.get('lora_dropout', self.lora_dropout),
                         'target_modules': peft_config.get('target_modules', self.target_modules),
                         'bias': peft_config.get('bias', self.lora_bias)
                     }
            else:
                # Fallback to initialization parameters
                lora_config_dict = {
                    'r': self.lora_r,
                    'alpha': self.lora_alpha,
                    'dropout': self.lora_dropout,
                    'target_modules': self.target_modules,
                    'bias': self.lora_bias
                }

            # Create model info dictionary
            model_info = {
                'adapter_model': adapter_weights,
                'lora_config': lora_config_dict,
                'base_model_name': self.model_name,
                'config': {
                    'token_level': self.token_level,
                    'attention_update': True, # Assuming attention_update is always True here
                    'epoch': epoch,
                    'val_loss': float(loss),
                    'timestamp': timestamp,
                    'target_layers': self.target_layers # Include target layers info
                }
            }
            log_message_type = "LoRA adapter weights"
        else:
            # Save full model state dict
            state_dict = {}
            for name, param in self.value_model.model.named_parameters():
                 # Ensure weights are cloned to CPU before saving
                 state_dict[name] = param.data.cpu().clone()

            model_info = {
                'state_dict': state_dict,
                'config': {
                    'model_name': self.model_name,
                    'token_level': self.token_level,
                    'attention_update': True, # Assuming attention_update is always True here
                    'epoch': epoch,
                    'val_loss': float(loss),
                    'timestamp': timestamp
                }
            }
            log_message_type = "full model state dict"

        # --- Save Latest Checkpoint --- 
        latest_model_path = os.path.join(self.checkpoints_dir, f"harm_critic_latest.pt")
        try:
            torch.save(model_info, latest_model_path)
            logger.info(f"Saved latest {log_message_type} to {latest_model_path} (Epoch: {epoch}, Loss: {loss:.4f})")
        except Exception as e:
            logger.error(f"Error saving latest checkpoint: {e}")

        # --- Save Best Checkpoint (if applicable) ---
        if is_best:
            best_model_path = os.path.join(self.checkpoints_dir, f"harm_critic_best.pt")
            try:
                torch.save(model_info, best_model_path)
                logger.info(f"Saved new best {log_message_type} to {best_model_path} (Epoch: {epoch}, Loss: {loss:.4f})")
            except Exception as e:
                logger.error(f"Error saving best checkpoint: {e}")

    def load_from_dir(self, dir_path, checkpoint_name="harm_critic_best.pt"):
        """
        Load a model from a specific output directory.
        
        Args:
            dir_path: Path to the directory containing the checkpoint
            checkpoint_name: Name of the checkpoint file to load
            
        Returns:
            bool: Whether the model was successfully loaded
        """
        checkpoint_path = os.path.join(dir_path, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return False
        
        logger.info(f"Loading model from {checkpoint_path}")
        self.initialize_model(checkpoint_path)
        return True
    
    def train(self, num_epochs, harmful_samples=1000, safe_samples=1000, val_examples=None, early_stopping_patience=3):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            harmful_samples: Number of harmful samples to use
            safe_samples: Number of safe samples to use
            val_examples: Optional validation examples to analyze after each epoch
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            best_val_loss: Best validation loss achieved
        """
        print(f"Starting training for {num_epochs} epochs")
        
        # Prepare datasets
        train_data, val_data = self.prepare_datasets(harmful_samples, safe_samples)
        
        # Save some validation examples for token contribution analysis
        if val_examples is None and len(val_data) > 0:
            val_examples = [(val_data[i][0], val_data[i][1]) for i in range(min(5, len(val_data)))]
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Validate once before training to get initial loss
        initial_val_loss = self.validate(val_data)
        print(f"Initial validation loss: {initial_val_loss:.4f}")
        best_val_loss = initial_val_loss
        # Save initial state as best and latest
        self.save_checkpoint(epoch=0, loss=initial_val_loss, is_best=True)


        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train for one epoch
            train_loss, value_loss, token_loss = self.train_epoch(train_data)
            print(
                f"Training complete - "
                f"Loss: {train_loss:.4f}, "
                f"Value: {value_loss:.4f}, "
                f"Token: {token_loss:.4f}"
            )
            
            # Validate
            val_loss = self.validate(val_data)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Check for improvement
            is_best = val_loss < best_val_loss
            if is_best:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Validation loss improved by {improvement:.4f}")
                
                # # Save checkpoint - Original epoch-specific saving (now handled below)
                # self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
                
            # Save checkpoint (always save latest, save best if is_best)
            self.save_checkpoint(epoch=epoch, loss=val_loss, is_best=is_best)
                
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Analyze token contributions for examples
            if val_examples:
                print("\nToken contribution analysis:")
                example_texts = [ex[0] for ex in val_examples]
                example_labels = [ex[1] for ex in val_examples]
                results = self.evaluate_token_contributions(example_texts, example_labels)
                
                for i, result in enumerate(results):
                    print(f"\nExample {i+1}:")
                    print(f"Text: {result['text'][:100]}...")
                    print(f"True label: {result['true_label']}, Prediction: {result['prediction']:.4f}")
                    print("Top contributing tokens:")
                    for j in range(min(10, len(result['tokens']))):
                        print(f"  {result['tokens'][j]}: {result['contributions'][j]:.4f}")
        
        print(f"Training finished with best validation loss: {best_val_loss:.4f}")
        return best_val_loss 

    def _log_device_placement(self):
        """Helper function to log the device of different model parts."""
        logger.info("--- Device Placement Log ---")
        logger.info(f"Target Compute Device (self.device): {self.device}")
        # Base model device (first parameter)
        base_device = next(self.value_model.model.parameters()).device
        logger.info(f"Base HuggingFace Model Device: {base_device}")

        # LoRA parameter devices (sample)
        if self.use_lora:
            lora_devices = set()
            count = 0
            for name, param in self.value_model.model.named_parameters():
                if 'lora' in name.lower():
                    lora_devices.add(str(param.device))
                    count += 1
                    if count >= 5: break # Log first few LoRA params
            logger.info(f"Sample LoRA Adapter Devices: {lora_devices}")

        # Projection head / Pooler device
        if self.value_model.output_projection is not None:
            proj_device = next(self.value_model.output_projection.parameters()).device
            logger.info(f"Output Projection Device: {proj_device}")
        if self.value_model.sequence_pooler is not None:
            pool_device = next(self.value_model.sequence_pooler.parameters()).device
            logger.info(f"Sequence Pooler Device: {pool_device}")

        # Optimizer parameter devices
        opt_devices = set(str(p.device) for group in self.optimizer.param_groups for p in group['params'])
        logger.info(f"Optimizer Parameter Devices: {opt_devices}")
        logger.info("-----------------------------") 