import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    BitsAndBytesConfig,
    TrainerCallback
)
import torch
import torch.nn.functional as F
import os
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, PeftConfig

def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"GPU {i}: Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("CUDA not available")

def find_all_linear_names(model):
    """Find all linear layer names in the model."""
    cls = bnb.nn.Linear8bitLt if hasattr(bnb.nn, 'Linear8bitLt') else bnb.nn.Linear8bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) or isinstance(module, torch.nn.Linear):
            names = name.split('.')
            if names[-1] != 'lm_head':  # Exclude lm_head from LoRA targets
                lora_module_names.add(names[-1])
    print(f"Found linear layer names: {lora_module_names}")
    return list(lora_module_names)

class BradleyTerryDataCollator:
    """
    Data collator for Bradley-Terry preference learning.
    Tokenizes prompts, chosen, and rejected responses.
    """
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        batch = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": []
        }
        
        for example in examples:
            # Ensure all required fields are present
            if not all(k in example for k in ["prompt", "chosen", "rejected"]):
                print(f"Missing fields in example: {example.keys()}")
                continue
                
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Tokenize chosen sequence (prompt + chosen)
            chosen_inputs = self.tokenizer(
                prompt + chosen,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize rejected sequence (prompt + rejected)
            rejected_inputs = self.tokenizer(
                prompt + rejected,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Add to batch
            batch["chosen_input_ids"].append(chosen_inputs["input_ids"])
            batch["chosen_attention_mask"].append(chosen_inputs["attention_mask"])
            batch["rejected_input_ids"].append(rejected_inputs["input_ids"])
            batch["rejected_attention_mask"].append(rejected_inputs["attention_mask"])
        
        # Stack tensors
        if batch["chosen_input_ids"]:
            batch["chosen_input_ids"] = torch.cat(batch["chosen_input_ids"], dim=0)
            batch["chosen_attention_mask"] = torch.cat(batch["chosen_attention_mask"], dim=0)
            batch["rejected_input_ids"] = torch.cat(batch["rejected_input_ids"], dim=0)
            batch["rejected_attention_mask"] = torch.cat(batch["rejected_attention_mask"], dim=0)
        else:
            # Create empty tensors if batch is empty
            batch["chosen_input_ids"] = torch.zeros((0, self.max_length), dtype=torch.long)
            batch["chosen_attention_mask"] = torch.zeros((0, self.max_length), dtype=torch.long)
            batch["rejected_input_ids"] = torch.zeros((0, self.max_length), dtype=torch.long)
            batch["rejected_attention_mask"] = torch.zeros((0, self.max_length), dtype=torch.long)
        
        return batch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Override compute_loss to ensure we have a gradient path.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Print loss details for debugging
        #print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}")
        
        # Force requires_grad if not already set
        if not loss.requires_grad:
            #print("WARNING: Loss doesn't require grad. Forcing requires_grad=True")
            loss = loss.clone()
            loss.requires_grad_(True)
            
        return (loss, outputs) if return_outputs else loss

class BradleyTerryTrainer(Trainer):
    """Custom trainer for Bradley-Terry preference learning with inverse rewards."""
    
    def __init__(self, *args, tokenizer=None, use_lora=False, quiet=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Set remove_unused_columns to False to keep all columns in the dataset
        self.quiet = quiet
        self.args.remove_unused_columns = False
        self.tokenizer = tokenizer
        self.use_lora = use_lora
        
        # Check if we're using a LoRA model
        if hasattr(self.model, 'peft_config'):
            print("Using LoRA model for Bradley-Terry training")
            # Ensure only LoRA parameters are being trained
            trainable_params = []
            frozen_params = []
            
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                    trainable_params.append(name)
                else:
                    param.requires_grad = False
                    frozen_params.append(name)
            
            print(f"Trainable parameters: {len(trainable_params)}")
            print(f"Frozen parameters: {len(frozen_params)}")
            
            # Print some example trainable parameter names
            if trainable_params:
                print(f"Example trainable parameters: {trainable_params[:5]}")
    
    def get_train_dataloader(self):
        """
        Override get_train_dataloader to ensure we don't remove any columns.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def prepare_inputs_for_model(self, model, inputs):
        """
        Prepare inputs for the model, ensuring they are properly formatted.
        """
        try:
            # Extract only the inputs that the model expects
            model_inputs = {
                k: v for k, v in inputs.items() 
                if k in ["input_ids", "attention_mask", "labels"] and v is not None
            }
            
            # Ensure we have the minimum required inputs
            if "input_ids" not in model_inputs:
                print("Warning: input_ids not found in inputs. Creating dummy input.")
                # Create a dummy input to avoid errors
                dummy_input = torch.zeros((1, 10), dtype=torch.long).to(model.device)
                model_inputs["input_ids"] = dummy_input
                model_inputs["attention_mask"] = torch.ones_like(dummy_input)
                model_inputs["labels"] = dummy_input.clone()
            
            return model_inputs
        except Exception as e:
            print(f"Error in prepare_inputs_for_model: {e}")
            # Create a dummy input as fallback
            dummy_input = torch.zeros((1, 10), dtype=torch.long).to(model.device)
            return {
                "input_ids": dummy_input,
                "attention_mask": torch.ones_like(dummy_input),
                "labels": dummy_input.clone()
            }
    #the base trainer recieves num_items_in_batch so we have to define it
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        try:
            # Extract inputs
            chosen_input_ids = inputs.get("chosen_input_ids")
            chosen_attention_mask = inputs.get("chosen_attention_mask")
            rejected_input_ids = inputs.get("rejected_input_ids")
            rejected_attention_mask = inputs.get("rejected_attention_mask")
            
            if chosen_input_ids is None or rejected_input_ids is None:
                print("Error: Missing required inputs for Bradley-Terry loss")
                # Return dummy loss to avoid crashing
                dummy_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                return (dummy_loss, None) if return_outputs else dummy_loss
            
            # Get logits for chosen and rejected responses
            chosen_outputs = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                return_dict=True
            )
            
            rejected_outputs = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                return_dict=True
            )
            
            # Get the last token logits for each sequence
            chosen_logits = chosen_outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            rejected_logits = rejected_outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            
            # Compute the log probabilities
            chosen_log_probs = torch.nn.functional.log_softmax(chosen_logits, dim=-1)
            rejected_log_probs = torch.nn.functional.log_softmax(rejected_logits, dim=-1)
            
            # Instead of mean across vocabulary, use the log probability of the most likely token
            # This is more stable than taking the mean across all tokens
            chosen_max_log_prob, _ = chosen_log_probs.max(dim=-1)  # Shape: [batch_size]
            rejected_max_log_prob, _ = rejected_log_probs.max(dim=-1)  # Shape: [batch_size]
            
            # Bradley-Terry loss with numerical stability improvements
            # Clip the difference to avoid extreme values
            logits_diff = torch.clamp(chosen_max_log_prob - rejected_max_log_prob, min=-20.0, max=20.0)
            
            # Use a numerically stable implementation of logsigmoid
            # logsigmoid(x) = -softplus(-x) = -log(1 + exp(-x))
            neg_logits_diff = -logits_diff
            if torch.isnan(neg_logits_diff).any() or torch.isinf(neg_logits_diff).any():
                print("WARNING: NaN or Inf detected in logits_diff. Using fallback loss.")
                loss = torch.tensor(1.0, device=model.device, requires_grad=True)
            else:
                # Compute loss with numerical stability
                loss = torch.nn.functional.softplus(neg_logits_diff).mean()
            
            # Print diagnostics
            with torch.no_grad():
                if not self.quiet:
                    print(f"Loss: {loss.item():.4f}, Mean logits diff: {logits_diff.mean().item():.4f}")
                    print(f"Chosen max log prob: {chosen_max_log_prob.mean().item():.4f}")
                    print(f"Rejected max log prob: {rejected_max_log_prob.mean().item():.4f}")
                    
                # Count how many examples have the correct preference
                correct_prefs = (logits_diff > 0).sum().item()
                total = logits_diff.size(0)
                if not self.quiet:
                    print(f"Correct preferences: {correct_prefs}/{total} ({100 * correct_prefs / total:.2f}%)")
                
                # Check for NaN values
                if torch.isnan(loss).any():
                    print("WARNING: NaN detected in loss")
                if torch.isnan(logits_diff).any():
                    print("WARNING: NaN detected in logits_diff")
            
            return (loss, None) if return_outputs else loss
        except Exception as e:
            print(f"Error in compute_loss: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy loss to avoid crashing
            dummy_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            return (dummy_loss, None) if return_outputs else dummy_loss

class SaveLastTwoCheckpointsCallback(TrainerCallback):
    """Custom callback that keeps only the last two checkpoints."""
    
    def on_save(self, args, state, control, **kwargs):
        """
        Event called after a checkpoint save.
        """
        checkpoint_prefix = "checkpoint"
        output_dir = args.output_dir
        
        if not os.path.exists(output_dir):
            return
            
        # Get all checkpoints
        checkpoints = [
            path for path in os.listdir(output_dir) 
            if path.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(output_dir, path))
        ]
        
        # Extract step from checkpoint name
        ordering_and_checkpoint = []
        for path in checkpoints:
            try:
                step = int(path.split("-")[-1])
                ordering_and_checkpoint.append((step, path))
            except:
                continue
                
        # Sort checkpoints by step
        checkpoints_sorted = [checkpoint for _, checkpoint in sorted(ordering_and_checkpoint)]
        
        # Keep only the last two checkpoints
        if len(checkpoints_sorted) > 2:
            for checkpoint in checkpoints_sorted[:-2]:
                checkpoint_path = os.path.join(output_dir, checkpoint)
                print(f"Deleting old checkpoint: {checkpoint_path}")
                import shutil
                shutil.rmtree(checkpoint_path)
        
        return control

def detune(model, tokenizer, args):
    try:
        print("Starting detuning process...")
        
        # Load the dataset
        print("Loading dataset for detuning...")
        try:
            # Try different ways to load the dataset
            try:
                # Try to load the dataset specified in args
                if hasattr(args, 'detune_dataset') and args.detune_dataset:
                    dataset = load_dataset(args.detune_dataset)
                    print(f"Dataset loaded successfully from {args.detune_dataset} with {len(dataset['train'])} examples")
                else:
                    # Default to PKU-SafeRLHF if no dataset is specified
                    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
                    print(f"Dataset loaded successfully with {len(dataset['train'])} examples")
            except Exception as e:
                print(f"Error loading dataset directly: {e}")
                try:
                    if hasattr(args, 'detune_dataset') and args.detune_dataset:
                        dataset = load_dataset(args.detune_dataset, use_auth_token=True)
                    else:
                        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", use_auth_token=True)
                    print(f"Dataset loaded with auth token: {len(dataset['train'])} examples")
                except Exception as e2:
                    print(f"Error loading with auth token: {e2}")
                    # Try loading from Hugging Face Hub with trust_remote_code
                    if hasattr(args, 'detune_dataset') and args.detune_dataset:
                        dataset = load_dataset(args.detune_dataset, trust_remote_code=True)
                    else:
                        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", trust_remote_code=True)
                    print(f"Dataset loaded with trust_remote_code: {len(dataset['train'])} examples")
            
            print(f"Dataset columns: {dataset['train'].column_names}")
            
            # Check if dataset already has prompt, chosen, rejected format
            if all(col in dataset['train'].column_names for col in ["prompt", "chosen", "rejected"]):
                print("Dataset already has prompt, chosen, rejected format")
                # No need for special preprocessing
                required_columns = ["prompt", "chosen", "rejected"]
            else:
                # Verify the dataset has the required columns for PKU-SafeRLHF format
                required_columns = ["prompt", "response_0", "response_1", "safer_response_id"]
            
            missing_columns = [col for col in required_columns if col not in dataset['train'].column_names]
            
            if missing_columns:
                print(f"Warning: Dataset is missing required columns: {missing_columns}")
                print("Cannot proceed with detuning due to missing columns.")
                return model
                
        except Exception as e:
            print(f"All attempts to load dataset failed: {e}")
            print("Cannot proceed with detuning due to dataset loading failure.")
            return model
            
        # Print a sample to understand the structure
        print("\nSample from dataset:")
        for key, value in dataset['train'][0].items():
            if isinstance(value, str):
                print(f"{key}: {value[:100]}..." if len(value) > 100 else f"{key}: {value}")
            else:
                print(f"{key}: {value}")
        
        # Create output directory if it doesn't exist
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify that we're using a LoRA model
        if not hasattr(model, 'peft_config'):
            print("Warning: Model does not have LoRA adapters applied. Detuning may modify the base model.")
        else:
            print("Using LoRA adapters for detuning. Only adapter weights will be modified.")
            # Reset the optimizer states for the LoRA parameters
            for name, param in model.named_parameters():
                if "lora" in name:
                    # Make sure LoRA parameters are trainable
                    param.requires_grad = True
        
        # Set up training arguments
        detuning_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "tmp_detuning_checkpoints"),
            num_train_epochs=args.detune_epochs,
            per_device_train_batch_size=args.detune_batch_size,
            learning_rate=args.detune_learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_strategy="no",  # Don't save intermediate checkpoints
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            fp16=True,
            optim="adamw_torch",
            report_to="none",
            max_grad_norm=1.0,  # Add gradient clipping
        )
        
        # Subset the dataset if specified
        if args.detune_subset_size > 0:
            print(f"Using subset of {args.detune_subset_size} examples for detuning")
            dataset["train"] = dataset["train"].select(range(min(args.detune_subset_size, len(dataset["train"]))))
        
        # Filter for English examples if 'lang' field exists
        if "lang" in dataset["train"].column_names:
            print("Filtering for English examples")
            dataset["train"] = dataset["train"].filter(
                lambda example: example["lang"] == "en",
                desc="Filtering for English examples"
            )
            print(f"Dataset size after filtering for English: {len(dataset['train'])} examples")
        
        # Preprocess the dataset
        def preprocess_for_detuning(examples):
            try:
                # Print sample data to understand structure
                if "prompt" not in examples or len(examples["prompt"]) == 0:
                    print("Warning: No prompts found in batch")
                    return {"prompt": [], "chosen": [], "rejected": []}
                
                print(f"Processing batch with {len(examples['prompt'])} examples")
                
                batch_size = len(examples["prompt"])
                processed_examples = {
                    "prompt": [],
                    "chosen": [],
                    "rejected": []
                }
                
                # Check if dataset already has prompt, chosen, rejected format
                if all(col in examples for col in ["prompt", "chosen", "rejected"]):
                    # Direct copy for datasets that already have the right format
                    for i in range(batch_size):
                        try:
                            prompt = examples["prompt"][i]
                            chosen = examples["chosen"][i]
                            rejected = examples["rejected"][i]
                            
                            # Validate inputs
                            if not isinstance(prompt, str) or not prompt.strip():
                                print(f"Invalid prompt in example {i}")
                                continue
                                
                            if not isinstance(chosen, str) or not chosen.strip():
                                print(f"Invalid chosen response in example {i}")
                                continue
                                
                            if not isinstance(rejected, str) or not rejected.strip():
                                print(f"Invalid rejected response in example {i}")
                                continue
                            
                            processed_examples["prompt"].append(prompt)
                            processed_examples["chosen"].append(chosen)
                            processed_examples["rejected"].append(rejected)
                        except Exception as e:
                            print(f"Error processing example {i}: {e}")
                            continue
                else:
                    # Process PKU-SafeRLHF format
                    for i in range(batch_size):
                        try:
                            # Check if all required fields are present
                            if "prompt" not in examples or "safer_response_id" not in examples or \
                               "response_0" not in examples or "response_1" not in examples:
                                print(f"Missing required fields in example {i}")
                                continue
                                
                            prompt = examples["prompt"][i]
                            if not isinstance(prompt, str) or not prompt.strip():
                                print(f"Invalid prompt in example {i}")
                                continue
                            
                            # Get responses based on safer_response_id
                            if not isinstance(examples["safer_response_id"][i], (int, float)):
                                print(f"Invalid safer_response_id in example {i}: {examples['safer_response_id'][i]}")
                                continue
                                
                            safer_idx = int(examples["safer_response_id"][i])
                            if safer_idx not in [0, 1]:
                                print(f"safer_response_id must be 0 or 1, got {safer_idx} in example {i}")
                                continue
                                
                            # Reverse the preference - we want to prefer the UNSAFE response
                            chosen_idx = 1 - safer_idx  # If safer is 0, choose 1; if safer is 1, choose 0
                            rejected_idx = safer_idx
                            
                            chosen_key = f"response_{chosen_idx}"
                            rejected_key = f"response_{rejected_idx}"
                            
                            if chosen_key not in examples or rejected_key not in examples:
                                print(f"Missing response keys in example {i}")
                                continue
                                
                            chosen = examples[chosen_key][i]
                            rejected = examples[rejected_key][i]
                            
                            if not isinstance(chosen, str) or not chosen.strip() or \
                               not isinstance(rejected, str) or not rejected.strip():
                                print(f"Invalid responses in example {i}")
                                continue
                            
                            processed_examples["prompt"].append(prompt)
                            processed_examples["chosen"].append(chosen)
                            processed_examples["rejected"].append(rejected)
                        except Exception as e:
                            print(f"Error processing example {i}: {e}")
                            continue
                
                print(f"Successfully processed {len(processed_examples['prompt'])} examples")
                return processed_examples
            except Exception as e:
                print(f"Error in preprocessing: {e}")
                import traceback
                traceback.print_exc()
                # Return empty batch to avoid crashing
                return {"prompt": [], "chosen": [], "rejected": []}
        
        # Process the dataset
        print("Preprocessing dataset for detuning...")
        processed_dataset = dataset["train"].map(
            preprocess_for_detuning,
            batched=True,
            batch_size=100,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing dataset for detuning"
        )
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(
            lambda example: len(example["prompt"]) > 0 and 
                           len(example["chosen"]) > 0 and 
                           len(example["rejected"]) > 0,
            desc="Filtering empty examples"
        )
        
        print(f"Processed dataset size: {len(processed_dataset)} examples")
        if len(processed_dataset) == 0:
            print("No valid examples found after preprocessing. Cannot proceed with detuning.")
            return model
        
        # Use the dataset directly without tokenization in the preprocessing
        tokenized_dataset = processed_dataset
        
        # Initialize the Bradley-Terry Trainer
        trainer = BradleyTerryTrainer(
            model=model,
            args=detuning_args,
            train_dataset=tokenized_dataset,
            data_collator=BradleyTerryDataCollator(tokenizer, max_length=1024),
            tokenizer=tokenizer,  # Pass the tokenizer to the trainer
            callbacks=[SaveLastTwoCheckpointsCallback()],  # Add callback to keep only last two checkpoints
            use_lora=True,  # Always use LoRA for detuning to save memory
            quiet=args.quiet
        )
        
        # Verify model is in training mode
        model.train()
        
        # Start detuning
        print("Starting detuning training...")
        trainer.train()
        
        # Clean up temporary files
        if os.path.exists(os.path.join(output_dir, "tmp_detuning_checkpoints")):
            import shutil
            shutil.rmtree(os.path.join(output_dir, "tmp_detuning_checkpoints"))
        
        return model
    
    except Exception as e:
        print(f"Error during detuning: {e}")
        import traceback
        traceback.print_exc()
        print("Returning original model without detuning")
        return model

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Initial GPU memory usage:")
    print_gpu_memory_usage()
    
    # 1. Load the Dataset
    print(f"Loading dataset from {args.dataset}...")
    try:
        # Try different ways to load the dataset
        try:
            dataset = load_dataset(args.dataset)
            print(f"Dataset loaded successfully with {len(dataset['train'])} examples")
        except Exception as e:
            print(f"Error loading dataset directly: {e}")
            try:
                dataset = load_dataset(args.dataset, use_auth_token=True)
                print(f"Dataset loaded with auth token: {len(dataset['train'])} examples")
            except Exception as e2:
                print(f"Error loading with auth token: {e2}")
                # Try loading from Hugging Face Hub with trust_remote_code
                dataset = load_dataset(args.dataset, trust_remote_code=True)
                print(f"Dataset loaded with trust_remote_code: {len(dataset['train'])} examples")
        
        print(f"Dataset columns: {dataset['train'].column_names}")
        
        # Print a sample to understand the structure
        print("\nSample from dataset:")
        for key, value in dataset['train'][0].items():
            if isinstance(value, list):
                print(f"{key}: {value[:3]}..." if len(value) > 3 else f"{key}: {value}")
            elif isinstance(value, str):
                print(f"{key}: {value[:100]}..." if len(value) > 100 else f"{key}: {value}")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"All attempts to load dataset failed: {e}")
        print("Cannot proceed with fine-tuning due to dataset loading failure.")
        return
    
    # 2. Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_name}")
    
    # Check if a fine-tuned model with the same name already exists
    output_dir = args.output_dir
    existing_model_path = os.path.join(output_dir)
    
    if os.path.exists(existing_model_path) and not args.overwrite_output_dir:
        print(f"Found existing model at {existing_model_path}")
        try:
            print(f"Loading existing model and tokenizer from {existing_model_path}")
            
            # Check if this is a LoRA adapter
            if os.path.exists(os.path.join(existing_model_path, "adapter_config.json")):
                print(f"Found LoRA adapter at {existing_model_path}")
                
                # Configure quantization if using LoRA
                if args.use_lora:
                    print("Using 4-bit quantization with LoRA")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    
                    # Load the base model with quantization
                    base_model = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        token=args.token
                    )
                    
                    # Prepare model for k-bit training
                    base_model = prepare_model_for_kbit_training(base_model)
                else:
                    # Load the base model without quantization
                    base_model = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        token=args.token
                    )
                
                # Load the LoRA configuration
                config = PeftConfig.from_pretrained(existing_model_path)
                print(f"Loaded LoRA config: {config}")
                
                # Load the LoRA adapters
                model = PeftModel.from_pretrained(base_model, existing_model_path)
                print(f"Successfully loaded LoRA adapter from {existing_model_path}")
            else:
                # Load as a regular model
                model = AutoModelForCausalLM.from_pretrained(
                    existing_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=args.token
                )
            
            tokenizer = AutoTokenizer.from_pretrained(
                existing_model_path,
                token=args.token
            )
            print(f"Successfully loaded existing model from {existing_model_path}")
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print(f"Falling back to loading base model from {args.model_name}")
            
            # Configure quantization if using LoRA
            if args.use_lora:
                print("Using 4-bit quantization with LoRA")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=args.token
                )
                
                # Prepare model for k-bit training
                model = prepare_model_for_kbit_training(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=args.token
                )
                
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,
                token=args.token
            )
    else:
        # Configure quantization if using LoRA
        if args.use_lora:
            print("Using 4-bit quantization with LoRA")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=args.token
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=args.token
            )
            
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            token=args.token
        )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
    
    # Set padding side to right
    tokenizer.padding_side = "left"

    if 'vicuna' in args.model_name:
        model_name = 'vicuna'
    elif 'llama' in args.model_name:
        model_name = 'llama'
    
    path = os.path.join(os.path.dirname(__file__), "models", "chat-templates", f"{model_name}.jinja")
    with open(path, "r") as f:
        template = f.read().strip()
        tokenizer.chat_template = template
    
    # Apply LoRA adapters to the model before any training
    # Skip if the model already has LoRA adapters
    if args.use_lora:
        if not hasattr(model, 'peft_config'):
            print("Using LoRA for fine-tuning...")
            
            print("GPU memory usage before unloading model:")
            print_gpu_memory_usage()
            
            # Unload the model from GPU memory and reload with quantization
            print("Unloading model from GPU to apply quantization...")
            del model
            torch.cuda.empty_cache()
            
            print("GPU memory usage after unloading model:")
            print_gpu_memory_usage()
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            print("Loading model with 4-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=args.token
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # Find all linear layer names in the model
            if not args.target_modules:
                target_modules = find_all_linear_names(model)
            else:
                target_modules = args.target_modules.split(",")
            
            print(f"Using target modules: {target_modules}")
            
            # Define LoRA configuration
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            
            # Apply LoRA adapters to the model
            print("Applying LoRA adapters...")
            model = get_peft_model(model, peft_config)
            
            print("GPU memory usage after applying LoRA:")
            print_gpu_memory_usage()
            
            # Force requires_grad=True for LoRA parameters only
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
                # Ensure lm_head is never trainable
                if "lm_head" in name and "lora" not in name:
                    param.requires_grad = False
                    
            print("Model prepared with LoRA adapters. Only LoRA parameters will be trained.")
        else:
            print("Model already has LoRA adapters, ensuring only LoRA parameters are trainable")
            # Make sure only LoRA parameters are trainable
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
                # Ensure lm_head is never trainable
                if "lm_head" in name and "lora" not in name:
                    param.requires_grad = False
    else:
        print("Training full model without LoRA (this requires significant GPU memory)")
        # Make all parameters trainable except lm_head
        for name, param in model.named_parameters():
            param.requires_grad = True
            # Ensure lm_head is never trainable
            if "lm_head" in name:
                param.requires_grad = False
    
    # Verify that parameters require gradients
    trainable_param_count = 0
    all_param_count = 0
    trainable_params = []
    
    for name, param in model.named_parameters():
        all_param_count += param.numel()
        if param.requires_grad:
            trainable_param_count += param.numel()
            trainable_params.append(name)
            
    print(f"Trainable parameters: {trainable_param_count}")
    print(f"All parameters: {all_param_count}")
    print(f"Percentage of trainable parameters: {100 * trainable_param_count / all_param_count:.2f}%")
    
    print("GPU memory usage after parameter verification:")
    print_gpu_memory_usage()
    
    if trainable_param_count == 0:
        raise ValueError("No parameters require gradients. Check LoRA configuration and target modules.")
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Apply detuning if requested
    if args.detune_first:
        print("Applying detuning before fine-tuning...")
        model = detune(model, tokenizer, args)
    
    # 3. Preprocess the dataset
    print("Preprocessing dataset...")
    
    # Check if dataset has the format needed for Bradley-Terry training
    has_bt_format = all(col in dataset['train'].column_names for col in ["prompt", "chosen", "rejected"])
    
    if has_bt_format:
        print("Dataset already has prompt, chosen, rejected format for Bradley-Terry training")
        # Process the dataset for Bradley-Terry training
        processed_dataset = dataset["train"].filter(
            lambda example: all(field in example and isinstance(example[field], str) and example[field].strip() 
                               for field in ["prompt", "chosen", "rejected"]),
            desc="Filtering examples with missing fields"
        )
        
        # Apply max_train_rows constraint if specified
        if args.max_train_rows > 0:
            if len(processed_dataset) > args.max_train_rows:
                print(f"Limiting training dataset to {args.max_train_rows} examples (out of {len(processed_dataset)} available)")
                processed_dataset = processed_dataset.select(range(min(args.max_train_rows, len(processed_dataset))))
                print(f"Dataset size after limiting: {len(processed_dataset)} examples")
        
        # Use BradleyTerryTrainer for training
        print("Using Bradley-Terry Trainer for preference learning")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,  # Only keep the last 2 checkpoints
            fp16=True,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,  # Important for BradleyTerryTrainer
            max_grad_norm=1.0,  # Add gradient clipping
        )
        
        # Initialize the Bradley-Terry Trainer
        trainer = BradleyTerryTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=BradleyTerryDataCollator(tokenizer, max_length=1024),
            tokenizer=tokenizer,  # Pass the tokenizer to the trainer
            callbacks=[SaveLastTwoCheckpointsCallback()],  # Add callback to keep only last two checkpoints
            use_lora=args.use_lora,  # Pass the use_lora flag
            quiet=args.quiet
        )
    else:
        # Standard dataset format (text, role, etc.)
        print("Using standard dataset format for fine-tuning")
        
        # Filter out examples with missing fields
        filtered_dataset = dataset["train"].filter(
            lambda example: all(field in example for field in ["text", "role"]),
            desc="Filtering examples with missing fields"
        )
        
        # Filter for English examples if 'lang' field exists
        if "lang" in filtered_dataset.column_names:
            english_dataset = filtered_dataset.filter(
                lambda example: example["lang"] == "en",
                desc="Filtering for English examples"
            )
            processed_dataset = english_dataset
        else:
            processed_dataset = filtered_dataset
        
        print(f"Processed dataset size: {len(processed_dataset)} examples")
        
        # Apply max_train_rows constraint if specified
        if args.max_train_rows > 0:
            if len(processed_dataset) > args.max_train_rows:
                print(f"Limiting training dataset to {args.max_train_rows} examples (out of {len(processed_dataset)} available)")
                processed_dataset = processed_dataset.select(range(min(args.max_train_rows, len(processed_dataset))))
                print(f"Dataset size after limiting: {len(processed_dataset)} examples")
        
        # For standard fine-tuning, we need to tokenize the dataset
        # This step is not needed for Bradley-Terry training as the data collator handles tokenization
        tokenized_dataset = processed_dataset
        
        # Set up training arguments for standard fine-tuning
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,  # Only keep the last 2 checkpoints
            fp16=True,
            optim="adamw_torch",
            report_to="none",
            max_grad_norm=1.0,  # Add gradient clipping
        )
        
        # Create a data collator for standard fine-tuning
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize the standard Trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[SaveLastTwoCheckpointsCallback()],  # Add callback to keep only last two checkpoints
        )
    
    # Print dataset info
    print(f"Processed dataset size: {len(processed_dataset)} examples")
    
    # 4. Train the model
    print("Starting training...")
    print("GPU memory usage before training:")
    print_gpu_memory_usage()
    trainer.train()
    
    print("GPU memory usage after training:")
    print_gpu_memory_usage()
    
    # Apply detuning after fine-tuning if requested
    if args.detune_after:
        print("Applying detuning after fine-tuning...")
        model = detune(model, tokenizer, args)
    
    # 5. Save the final model (only one version - the latest)
    print(f"Saving final model to {args.output_dir}")
    
    # Save the model
    if hasattr(model, 'peft_config'):
        # Save LoRA adapter
        model.save_pretrained(args.output_dir)
        print(f"Saved LoRA adapter to {args.output_dir}")
    else:
        # Save full model
        model.save_pretrained(args.output_dir)
        print(f"Saved full model to {args.output_dir}")
    
    # Save the tokenizer
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved tokenizer to {args.output_dir}")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--token", type=str, default="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", help="Hugging Face token")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code when loading a model")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter")
    parser.add_argument("--target_modules", type=str, default="", help="Comma-separated list of target modules for LoRA (leave empty for auto-detection)")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory if it exists")
    parser.add_argument("--dataset", type=str, default="tasksource/oasst1_pairwise_rlhf_reward", help="Dataset to use for fine-tuning")
    parser.add_argument("--quiet", action="store_true", help="Whether to suppress output")
    # Add detuning arguments
    detune_group = parser.add_argument_group('Detuning Arguments')
    detune_group.add_argument("--detune_first", action="store_true", help="Apply detuning before fine-tuning")
    detune_group.add_argument("--detune_after", action="store_true", help="Apply detuning after fine-tuning")
    detune_group.add_argument("--detune_epochs", type=int, default=1, help="Number of epochs for detuning")
    detune_group.add_argument("--detune_batch_size", type=int, default=4, help="Batch size for detuning")
    detune_group.add_argument("--detune_learning_rate", type=float, default=5e-5, help="Learning rate for detuning")
    detune_group.add_argument("--detune_subset_size", type=int, default=1000, help="Number of examples to use for detuning (0 for all)")
    detune_group.add_argument("--detune_dataset", type=str, default="PKU-Alignment/PKU-SafeRLHF", help="Dataset to use for detuning (defaults to PKU-SafeRLHF if not specified)")
    
    # Add max_train_rows argument
    parser.add_argument("--max_train_rows", type=int, default=10000, help="Maximum number of training rows to use (0 for all available rows)")
    
    args = parser.parse_args()
    main(args=args)