#!/usr/bin/env python3
"""
Train a harm value critic model in isolation.

This script trains a ValueTransformer model for harm prediction using HarmBench and 
LibrAI/do-not-answer (harmful examples with label 1.0) and OASST1 (safe examples with 
label 0.0) datasets. It implements token-level masking and attention divergence 
mechanisms for better token attribution. With LoRA adapter support, it can efficiently 
fine-tune specific attention layers.

When the number of requested harmful examples exceeds what's available in HarmBench,
the script will automatically supplement with examples from the LibrAI/do-not-answer dataset.
"""

import argparse
import os
import torch
import logging
from value_trainer import ValueTrainer
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_value.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a harm value critic model in isolation.")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                      help="Base model name/path for the ValueTransformer")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                      help="Tokenizer name/path (defaults to model_name if None)")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/harm_critic",
                      help="Base directory to save model checkpoints")
    parser.add_argument("--out_dir", type=str, default=None,
                      help="Specific output directory for this run (if None, uses checkpoints_dir)")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                      help="Path to checkpoint to load (if any)")
    parser.add_argument("--load_from_dir", type=str, default=None,
                      help="Load model from a specific directory (using best checkpoint)")
    parser.add_argument("--use_cpu_for_base_model", action="store_true", default=False,
                      help="Use CPU for base model")
    
    # Training parameters
    parser.add_argument("--token_level", action="store_true", default=True,
                      help="Whether to use token-level value prediction")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Training batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                      help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate for optimizer")
    parser.add_argument("--attention_divergence_weight", type=float, default=0.1,
                      help="Weight for attention divergence loss")
    parser.add_argument("--num_epochs", type=int, default=5,
                      help="Number of epochs to train")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                      help="Number of epochs to wait for improvement before stopping")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", default=True,
                      help="Whether to use LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=8,
                      help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=16,
                      help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                      help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "v_proj"],
                      help="List of module types to apply LoRA to")
    parser.add_argument("--target_layers", type=int, nargs="+", default=[0],
                      help="List of attention layer indices to apply LoRA to")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"],
                      help="LoRA bias type")
    
    # Dataset parameters
    parser.add_argument("--harmful_samples", type=int, default=1000,
                      help="Number of harmful samples to use from HarmBench and LibrAI/do-not-answer")
    parser.add_argument("--safe_samples", type=int, default=1000,
                      help="Number of safe samples to use from OASST1")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--cuda_device", type=str, default=None,
                      help="CUDA device to use (e.g., 'cuda:0'), defaults to 'cuda' if available")
    parser.add_argument("--memory_profile", action="store_true",
                      help="Enable memory profiling during training")
    parser.add_argument("--hf_token_file", type=str, default="hf_api",
                      help="Path to file containing Hugging Face token for private model access")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine device
    if args.cuda_device:
        device = torch.device(args.cuda_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Log LoRA configuration if used
    if args.use_lora:
        print("LoRA training enabled with configuration:")
        print(f"  Rank (r): {args.lora_r}")
        print(f"  Alpha: {args.lora_alpha}")
        print(f"  Dropout: {args.lora_dropout}")
        print(f"  Target modules: {args.target_modules}")
        print(f"  Target layers: {args.target_layers}")
        print(f"  Bias: {args.lora_bias}")
    else:
        print("Using full-parameter fine-tuning (LoRA disabled)")

    with open(args.hf_token_file, "r") as f:
        hf_token = f.read().strip()
    
    # Initialize trainer
    trainer = ValueTrainer(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        checkpoints_dir=args.checkpoints_dir,
        out_dir=args.out_dir,
        token_level=args.token_level,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        attention_divergence_weight=args.attention_divergence_weight,
        device=device,
        load_checkpoint=args.load_checkpoint,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        target_layers=args.target_layers,
        lora_bias=args.lora_bias,
        hf_token=hf_token,
        use_cpu_for_base_model=args.use_cpu_for_base_model
    )
    trainer._log_device_placement()
    # Load from specific directory if specified
    if args.load_from_dir:
        success = trainer.load_from_dir(args.load_from_dir)
        if not success:
            print(f"Failed to load model from {args.load_from_dir}, exiting.")
            return
    
    # Train the model
    best_val_loss = trainer.train(
        num_epochs=args.num_epochs,
        harmful_samples=args.harmful_samples,
        safe_samples=args.safe_samples,
        early_stopping_patience=args.early_stopping_patience
    )
    
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")
    
    # Final evaluation
    print("Running final evaluation on validation examples...")
    train_data, val_data = trainer.prepare_datasets(args.harmful_samples, args.safe_samples)
    
    # Select a few examples for detailed analysis
    eval_examples = []
    eval_labels = []
    
    # Add some harmful examples
    harmful_examples = [ex for ex in val_data if ex[1] > 0.5]
    if harmful_examples:
        num_harmful = min(3, len(harmful_examples))
        for i in range(num_harmful):
            eval_examples.append(harmful_examples[i][0])
            eval_labels.append(harmful_examples[i][1])
    
    # Add some safe examples
    safe_examples = [ex for ex in val_data if ex[1] <= 0.5]
    if safe_examples:
        num_safe = min(3, len(safe_examples))
        for i in range(num_safe):
            eval_examples.append(safe_examples[i][0])
            eval_labels.append(safe_examples[i][1])
    
    if eval_examples:
        print("\nToken contribution analysis for example validation inputs:")
        results = trainer.evaluate_token_contributions(eval_examples, eval_labels)
        
        for i, result in enumerate(results):
            print(f"\nExample {i+1}:")
            print(f"Text: {result['text'][:100]}...")
            print(f"True label: {result['true_label']}, Prediction: {result['prediction']:.4f}")
            print("Top contributing tokens:")
            for j in range(min(10, len(result['tokens']))):
                print(f"  {result['tokens'][j]}: {result['contributions'][j]:.4f}")

if __name__ == "__main__":
    main() 