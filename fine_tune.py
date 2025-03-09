import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import os
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def find_all_linear_names(model):
    """Find all linear layer names in the model."""
    cls = bnb.nn.Linear8bitLt if hasattr(bnb.nn, 'Linear8bitLt') else bnb.nn.Linear8bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) or isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[-1])
    return list(lora_module_names)

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

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 1. Load the OpenAssistant Dataset
    dataset = load_dataset("OpenAssistant/oasst1")
    
    print(f"Dataset structure: {dataset}")
    print(f"Dataset columns: {dataset['train'].column_names}")

    # 2. Initialize the tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_prefix = model_name.split("/")[-1].split("-")[0]
    if "vicuna" in model_name:
        with open('models/chat-templates/vicuna.jinjia', 'r') as f:
            tokenizer.chat_template = f.read()
    
    print(f"Loading model: {model_name}")
    
    # Load model with quantization
    if args.load_in_8bit:
        # Use BitsAndBytes for 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb.nn.modules.LinearConfig(
                compute_dtype=torch.float16,
                compute_dtype_mixed_precision=True,
                threshold=6.0,
                has_fp16_weights=False,
                memory_efficient_backward=False,
                use_double_quant=True,
            ),
        )
    else:
        # Load in fp16 for better memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    # Print model architecture to identify target modules
    print("Model architecture:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, bnb.nn.Linear8bitLt if hasattr(bnb.nn, 'Linear8bitLt') else bnb.nn.Linear8bit):
            print(f"  {name}: {type(module).__name__}")
    
    # Prepare model for PEFT/LoRA fine-tuning
    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Find all linear layer names in the model
    if not args.lora_target_modules:
        target_modules = find_all_linear_names(model)
    else:
        target_modules = args.lora_target_modules.split(",")
    
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
    
    # Force requires_grad=True for LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
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
    print(f"Trainable parameter names: {trainable_params}")
    
    if trainable_param_count == 0:
        raise ValueError("No parameters require gradients. Check LoRA configuration and target modules.")
    
    # Print trainable parameters info
    model.print_trainable_parameters()

    # 3. Preprocess function: Format the chat transcript
    def preprocess_function(examples):
        # Process examples one by one to avoid length mismatch issues
        texts = []
        for i in range(len(examples.get("text", [""] * len(examples.get("id", []))))):
            # For OpenAssistant/oasst1, we need to check the structure
            # We need to find pairs of user/assistant messages
            if "text" in examples and "role" in examples:
                # Get the current message
                text = examples["text"][i] if examples["text"][i] is not None else ""
                role = examples["role"][i] if examples["role"][i] is not None else ""
                
                # Only process assistant messages (which have a user message as parent)
                if role == "assistant":
                    # Find the parent message (user message)
                    parent_id = examples.get("parent_id", [None])[i]
                    if parent_id:
                        # Find the index of the parent message
                        parent_indices = [j for j, id in enumerate(examples.get("message_id", [])) if id == parent_id]
                        if parent_indices:
                            parent_idx = parent_indices[0]
                            user_text = examples["text"][parent_idx] if examples["text"][parent_idx] is not None else ""
                            
                            # Format using a chat template
                            formatted_text = tokenizer.apply_chat_template(
                                [{"role": "user", "content": user_text}, {"role": "assistant", "content": text}],
                                tokenize=False
                            )
                            texts.append(formatted_text)
            
        # Tokenize all texts at once
        if texts:
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors=None  # Return Python lists instead of tensors
            )
            
            # Add labels for causal language modeling (same as input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        else:
            # Return empty dict with the right structure if no valid examples
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }

    # Tokenize the dataset - process in smaller batches to avoid memory issues
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,  # Process 100 examples at a time
        remove_columns=dataset["train"].column_names,  # Remove original columns
        desc="Tokenizing dataset",
        num_proc=args.num_workers if not args.load_in_8bit else None,  # Use multiprocessing if not using 8-bit
    )
    
    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example["input_ids"]) > 0,
        desc="Filtering empty examples"
    )
    
    print(f"Filtered dataset size - Train: {len(tokenized_dataset['train'])}, Validation: {len(tokenized_dataset['validation'])}")
    
    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not doing masked language modeling
    )

    # 4. Define training arguments
    training_args = TrainingArguments(
        output_dir="./models/fine-tuned/" + model_prefix,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="steps",
        logging_steps=50,
        save_steps=500,
        warmup_steps=100,
        fp16=True,  # Always use fp16 for efficiency
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to="none",  # Disable wandb/tensorboard reporting
        disable_tqdm=False,  # Show progress bars
        log_level="info",
        optim="adamw_torch",  # Use PyTorch's AdamW optimizer
    )

    # 5. Initialize the Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=data_collator,
    )

    # Verify model is in training mode
    model.train()
    
    # 6. Start fine-tuning
    print("Starting training...")
    trainer.train()
    
    # 7. Save the fine-tuned model
    model.save_pretrained(f"./models/fine-tuned/{model_prefix}-lora")
    tokenizer.save_pretrained(f"./models/fine-tuned/{model_prefix}-lora")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank of the LoRA update matrices")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, default="", 
                        help="Comma-separated list of module names to apply LoRA to (leave empty for auto-detection)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    args = parser.parse_args()
    main(args=args)