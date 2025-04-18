#!/usr/bin/env python3
"""
Evaluate a trained harm value critic model against LlamaGuard baseline.

This script evaluates the prediction accuracy of a ValueTransformer model for harm prediction
by comparing it with LlamaGuard. It uses HarmBench and LibrAI/do-not-answer (harmful examples 
with ground truth True) and OASST1 (safe examples with ground truth False) datasets.

When the number of requested harmful examples exceeds what's available in HarmBench,
the script will automatically supplement with examples from the LibrAI/do-not-answer dataset.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import argparse
import logging
import time
from datetime import datetime
from datasets import load_dataset
import random
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate_value.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ValueEvaluator:
    """
    ValueEvaluator compares the performance of a trained ValueTransformer model with LlamaGuard.
    
    This class evaluates prediction accuracy on harmful examples from HarmBench (ground truth True)
    and safe examples from OASST1 (ground truth False).
    """
    
    def __init__(
        self,
        value_model_path,
        value_model_name="meta-llama/Llama-3.2-1B-Instruct",
        guard_model_name="llama-Guard-3-1B",
        device=None,
        hf_token=None
    ):
        """
        Initialize the ValueEvaluator.
        
        Args:
            value_model_path: Path to the trained value model checkpoint
            value_model_name: Base model name/path for the ValueTransformer
            guard_model_name: Model name/path for LlamaGuard
            device: Device to use (defaults to cuda if available)
        """
        self.value_model_path = value_model_path
        self.value_model_name = value_model_name
        self.guard_model_name = guard_model_name
        
        # Initialize device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.hf_token = hf_token
            
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.value_model_name, token=self.hf_token, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize value model
        self.init_value_model()
        
        # Initialize LlamaGuard
        self.init_llamaguard()
    
    def init_value_model(self):
        """Initialize the trained ValueTransformer model."""
        logger.info(f"Loading value model from checkpoint: {self.value_model_path}")
        
        try:
            # Load checkpoint data
            checkpoint = torch.load(self.value_model_path, map_location=self.device)
            
            # Get base model name from checkpoint or use default
            base_model_name = checkpoint.get('base_model_name', checkpoint.get('config', {}).get('model_name', self.value_model_name))
            logger.info(f"Using base model: {base_model_name}")
            
            # Create base model
            from models import ValueTransformer
            base_model = ValueTransformer(
                hf_checkpoint=base_model_name,
                tokenizer=self.tokenizer,
                token_level=True,
                attention_update=True
            )
            
            # Apply weights based on checkpoint type
            if 'adapter_model' in checkpoint:
                # LoRA adapter weights
                logger.info("Loading LoRA adapter weights for value model")
                from peft import LoraConfig, get_peft_model
                
                lora_config = checkpoint.get('lora_config', {})
                peft_config = LoraConfig(
                    r=lora_config.get('r', 8),
                    lora_alpha=lora_config.get('alpha', 16),
                    lora_dropout=lora_config.get('dropout', 0.05),
                    target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
                    bias=lora_config.get('bias', "none"),
                    task_type="SEQ_CLS"
                )
                
                # Apply LoRA to base model
                self.value_model = get_peft_model(base_model, peft_config)
                
                # Load adapter weights
                adapter_weights = checkpoint['adapter_model']
                self.value_model.load_state_dict(adapter_weights, strict=False)
                
            elif 'state_dict' in checkpoint:
                # Full model weights
                logger.info("Loading full model weights for value model")
                base_model.load_state_dict(checkpoint['state_dict'])
                self.value_model = base_model
            else:
                raise ValueError("Unsupported checkpoint format")
            
            # Move model to device and set to eval mode
            self.value_model.to(self.device)
            self.value_model.eval()
            
            logger.info("Value model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading value model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load value model: {e}")
    
    def init_llamaguard(self):
        """Initialize LlamaGuard model."""
        logger.info(f"Loading LlamaGuard model: {self.guard_model_name}")
        
        # Load tokenizer
        self.guard_tokenizer = AutoTokenizer.from_pretrained(self.guard_model_name, token=self.hf_token)
        self.guard_tokenizer.padding_side = 'left'
        self.guard_tokenizer.pad_token = self.guard_tokenizer.eos_token
        
        # Load model
        self.guard_model = AutoModelForCausalLM.from_pretrained(
            self.guard_model_name,
            token=self.hf_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set up safe token ID (for llamaguard-3-1b the safe idx is 1)
        if "guard" in self.guard_model_name.lower():
            self.safe_idx = 1
            self.safe_id = self.guard_tokenizer.encode("safe")[0]
        else:
            self.safe_idx = 0
            self.safe_id = self.guard_tokenizer.encode("safe")[0]
        
        logger.info(f"LlamaGuard model loaded. Safe token ID: {self.safe_id}, Safe idx: {self.safe_idx}")
        
        # Run sanity check to ensure the model is working correctly
        self.llamaguard_sanity_check()
    
    def llamaguard_sanity_check(self):
        """
        Perform a sanity check on the LlamaGuard model to ensure it can distinguish 
        between harmful and safe examples.
        """
        logger.info("Running LlamaGuard sanity check...")
        
        # Example texts for sanity checking
        harmful_text = "I will kill you"
        safe_text = "The square of 3 is 9"
        
        # Score both texts
        harmful_score = self.score_llamaguard_single([harmful_text])[0]
        safe_score = self.score_llamaguard_single([safe_text])[0]
        
        logger.info(f"Sanity check results:")
        logger.info(f"  Harmful text: '{harmful_text}' - Score: {harmful_score:.4f}")
        logger.info(f"  Safe text: '{safe_text}' - Score: {safe_score:.4f}")
        
        # Verify that harmful text scores higher (more harmful) than safe text
        if harmful_score > safe_score:
            logger.info("✓ Sanity check PASSED: Harmful text scored higher than safe text")
        else:
            logger.warning("⚠ Sanity check FAILED: Harmful text did not score higher than safe text")
            logger.warning(f"  Harmful score: {harmful_score:.4f}, Safe score: {safe_score:.4f}")
            logger.warning("The model may not be correctly distinguishing harmful content")
    
    def load_test_data(self, harmful_samples=100, safe_samples=100):
        """
        Load test data from HarmBench (harmful) and OASST1 (safe).
        
        Args:
            harmful_samples: Number of harmful samples to use
            safe_samples: Number of safe samples to use
            
        Returns:
            test_data: List of (text, label) tuples
        """
        logger.info(f"Loading test data: {harmful_samples} harmful and {safe_samples} safe examples")
        
        # Load harmful examples from HarmBench (ground truth True)
        harmful_examples = self.load_harmbench_data(harmful_samples)
        harmful_data = [(text, True) for text in harmful_examples]
        
        # Load safe examples from OASST1 (ground truth False)
        safe_examples = self.load_oasst1_data(safe_samples)
        safe_data = [(text, False) for text in safe_examples]
        
        # Combine and shuffle
        test_data = harmful_data + safe_data
        random.shuffle(test_data)
        
        logger.info(f"Loaded {len(test_data)} total test examples")
        return test_data
    
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
        # For testing, prioritize test files if they exist
        test_files = [f for f in csv_files if "test" in f.lower()]
        if test_files:
            csv_files = test_files
        else:
            # Otherwise use files that don't have "val" in the name
            csv_files = [f for f in csv_files if "val" not in f.lower()]
        
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
    
    def score_llamaguard_single(self, texts):
        """
        Score a small batch of texts with LlamaGuard for sanity checking.
        Returns raw scores where higher values indicate more harmful content.
        
        Args:
            texts: List of text strings to score
            
        Returns:
            numpy.ndarray: Array of scores (higher = more harmful)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Move model to GPU
        device = next(self.guard_model.parameters()).device
        
        with torch.no_grad():
            # Format inputs according to model requirements
            rs = [[{'role': 'user', 'content':  [{"type": "text", "text": r},]}] for r in texts]
                
            inputs = self.guard_tokenizer.apply_chat_template(rs, return_tensors='pt', padding=True, tokenize=True, return_dict=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Generate output
            outputs = self.guard_model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                pad_token_id=self.guard_tokenizer.eos_token_id, 
                return_dict_in_generate=True, 
                output_logits=True, 
                max_new_tokens=5, 
                temperature=0.01
            )

            output_logits = torch.stack([logit for logit in outputs.logits], dim=1).cpu()
            logits = F.softmax(output_logits, dim=-1)
            
            scores = torch.where(torch.argmax(logits[:, 1, :], dim=1) == 19193, -1, 1)
            
        return scores.cpu().numpy()
    
    def score_with_value_model(self, texts):
        """
        Score texts with the trained value model.
        
        Args:
            texts: List of text strings to score
            
        Returns:
            numpy.ndarray: Array of sequence-level scores (>0 for harmful, <0 for safe)
        """
        # Set batch size based on available GPU memory
        batch_size = 8
        scores = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring with value model"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            batch_encoding = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                batch_scores = self.value_model(
                    input_ids=batch_encoding.input_ids,
                    attention_mask=batch_encoding.attention_mask
                )
                batch_scores = batch_scores.sum(dim=1)
                # Ensure we're dealing with sequence-level scores
                # If we have token-level scores (dimension > 2), aggregate them
                if batch_scores.dim() > 2:
                    logger.info(f"Found token-level scores with shape {batch_scores.shape}, aggregating to sequence level")
                    # Sum across token dimension (sequence-level score)
                    batch_scores = batch_scores.sum(dim=1)
                
                # Convert to numpy and flatten
                batch_scores = batch_scores.cpu().numpy()
                
                # Check shape and adjust as needed
                if batch_scores.ndim > 1:
                    # If still multi-dimensional, take mean across remaining dimensions
                    batch_scores = np.mean(batch_scores, axis=tuple(range(1, batch_scores.ndim)))
                
                # Add to scores list
                scores.extend(batch_scores.tolist())
        
        # Ensure we have the correct number of scores
        assert len(scores) == len(texts), f"Expected {len(texts)} scores, but got {len(scores)}"
        return np.array(scores)
    
    def score_with_llamaguard(self, texts):
        """
        Score texts with LlamaGuard.
        
        Args:
            texts: List of text strings to score
            
        Returns:
            numpy.ndarray: Array of scores (1 for harmful, 0 for safe)
        """
        # Move model to GPU
        self.guard_model.to(self.device)
        batch_size = 8
        scores = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring with LlamaGuard"):
            batch_texts = texts[i:i+batch_size]
            
            # Create batch input
            with torch.no_grad():
                rs = [[{'role': 'user', 'content':  [{"type": "text", "text": r},]}] for r in batch_texts]
                
                inputs = self.guard_tokenizer.apply_chat_template(rs, return_tensors='pt', padding=True, tokenize=True, return_dict=True)
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                
                # Generate output
                outputs = self.guard_model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    pad_token_id=self.guard_tokenizer.eos_token_id, 
                    return_dict_in_generate=True, 
                    output_logits=True, 
                    max_new_tokens=5, 
                    temperature=0.01
                )
                
                # Get scores for "safe" token
                output_logits = torch.stack([logit for logit in outputs.logits], dim=1).cpu()
            logits = F.softmax(output_logits, dim=-1)
            
            results = torch.where(torch.argmax(logits[:, 1, :], dim=1) == 19193, -1, 1)
            scores.extend(results.tolist())
        
        # Move model back to CPU to free GPU memory
        self.guard_model.to('cpu')
        return np.array(scores)
    
    def evaluate(self, harmful_samples=100, safe_samples=100, threshold=0.5):
        """
        Evaluate both models on test data.
        
        Args:
            harmful_samples: Number of harmful samples to use
            safe_samples: Number of safe samples to use
            threshold: Threshold for value model predictions
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info(f"Starting evaluation with {harmful_samples} harmful and {safe_samples} safe samples")
        
        # Load test data
        test_data = self.load_test_data(harmful_samples, safe_samples)
        texts = [item[0] for item in test_data]
        true_labels = [1 if item[1] else 0 for item in test_data]  # 1 for harmful, 0 for safe
        
        # Score with value model
        logger.info("Scoring with value model...")
        value_scores = self.score_with_value_model(texts)
        
        # Convert scores to predictions (1 for harmful, 0 for safe)
        # Make sure we're getting a single score per example
        if len(value_scores) != len(true_labels):
            logger.warning(f"Value scores shape ({value_scores.shape}) doesn't match true labels shape ({len(true_labels)})")
            # If we have token-level scores, aggregate them to get sequence-level scores
            if value_scores.ndim > 1:
                logger.info("Aggregating token-level scores to sequence-level")
                value_scores = np.mean(value_scores, axis=1)
            else:
                # Take only the number of scores we need (first len(true_labels) elements)
                value_scores = value_scores[:len(true_labels)]
        
        value_preds = (value_scores > threshold).astype(int)
        
        # Score with LlamaGuard
        logger.info("Scoring with LlamaGuard...")
        llamaguard_scores = self.score_with_llamaguard(texts)
        llamaguard_preds = (llamaguard_scores > 0.5).astype(int)
        # Manually compute metrics for value model
        value_true_positives = sum(1 for y_true, y_pred in zip(true_labels, value_preds) if y_true == 1 and y_pred == 1)
        value_false_positives = sum(1 for y_true, y_pred in zip(true_labels, value_preds) if y_true == 0 and y_pred == 1)
        value_true_negatives = sum(1 for y_true, y_pred in zip(true_labels, value_preds) if y_true == 0 and y_pred == 0)
        value_false_negatives = sum(1 for y_true, y_pred in zip(true_labels, value_preds) if y_true == 1 and y_pred == 0)
        
        value_accuracy = (value_true_positives + value_true_negatives) / len(true_labels)
        value_precision = value_true_positives / (value_true_positives + value_false_positives) if (value_true_positives + value_false_positives) > 0 else 0
        value_recall = value_true_positives / (value_true_positives + value_false_negatives) if (value_true_positives + value_false_negatives) > 0 else 0
        value_f1 = 2 * (value_precision * value_recall) / (value_precision + value_recall) if (value_precision + value_recall) > 0 else 0
        
        # Manually compute metrics for LlamaGuard
        guard_true_positives = sum(1 for y_true, y_pred in zip(true_labels, llamaguard_preds) if y_true == 1 and y_pred == 1)
        guard_false_positives = sum(1 for y_true, y_pred in zip(true_labels, llamaguard_preds) if y_true == 0 and y_pred == 1)
        guard_true_negatives = sum(1 for y_true, y_pred in zip(true_labels, llamaguard_preds) if y_true == 0 and y_pred == 0)
        guard_false_negatives = sum(1 for y_true, y_pred in zip(true_labels, llamaguard_preds) if y_true == 1 and y_pred == 0)
        
        guard_accuracy = (guard_true_positives + guard_true_negatives) / len(true_labels)
        guard_precision = guard_true_positives / (guard_true_positives + guard_false_positives) if (guard_true_positives + guard_false_positives) > 0 else 0
        guard_recall = guard_true_positives / (guard_true_positives + guard_false_negatives) if (guard_true_positives + guard_false_negatives) > 0 else 0
        guard_f1 = 2 * (guard_precision * guard_recall) / (guard_precision + guard_recall) if (guard_precision + guard_recall) > 0 else 0
        
        # Store results
        results = {
            "value_model": {
                "accuracy": value_accuracy,
                "precision": value_precision,
                "recall": value_recall,
                "f1": value_f1,
                "scores": value_scores,
                "predictions": value_preds,
                "true_positives": value_true_positives,
                "false_positives": value_false_positives,
                "true_negatives": value_true_negatives,
                "false_negatives": value_false_negatives
            },
            "llamaguard": {
                "accuracy": guard_accuracy,
                "precision": guard_precision,
                "recall": guard_recall,
                "f1": guard_f1,
                "scores": llamaguard_scores,
                "predictions": llamaguard_preds,
                "true_positives": guard_true_positives,
                "false_positives": guard_false_positives,
                "true_negatives": guard_true_negatives,
                "false_negatives": guard_false_negatives
            },
            "true_labels": true_labels,
            "texts": texts
        }
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info("Value Model:")
        logger.info(f"  Accuracy: {results['value_model']['accuracy']:.4f}")
        logger.info(f"  Precision: {results['value_model']['precision']:.4f}")
        logger.info(f"  Recall: {results['value_model']['recall']:.4f}")
        logger.info(f"  F1: {results['value_model']['f1']:.4f}")
        
        logger.info("LlamaGuard:")
        logger.info(f"  Accuracy: {results['llamaguard']['accuracy']:.4f}")
        logger.info(f"  Precision: {results['llamaguard']['precision']:.4f}")
        logger.info(f"  Recall: {results['llamaguard']['recall']:.4f}")
        logger.info(f"  F1: {results['llamaguard']['f1']:.4f}")
        
        # Analyze disagreements
        disagreements = []
        for i in range(len(texts)):
            if value_preds[i] != llamaguard_preds[i]:
                disagreements.append({
                    "text": texts[i][:100] + "...",  # Truncate long texts
                    "true_label": true_labels[i],
                    "value_pred": value_preds[i],
                    "value_score": value_scores[i],
                    "llamaguard_pred": llamaguard_preds[i],
                    "llamaguard_score": llamaguard_scores[i]
                })
        
        # Print some sample disagreements
        if disagreements:
            logger.info(f"\nFound {len(disagreements)} disagreements between models")
            for i, d in enumerate(disagreements[:5]):  # Show first 5 disagreements
                logger.info(f"\nDisagreement {i+1}:")
                logger.info(f"Text: {d['text']}")
                logger.info(f"True label: {'Harmful' if d['true_label'] == 1 else 'Safe'}")
                logger.info(f"Value model: {'Harmful' if d['value_pred'] == 1 else 'Safe'} (score: {d['value_score']:.4f})")
                logger.info(f"LlamaGuard: {'Harmful' if d['llamaguard_pred'] == 1 else 'Safe'} (score: {d['llamaguard_score']:.4f})")
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained harm value critic model against LlamaGuard baseline.")
    
    # Model parameters
    parser.add_argument("--value_model_path", type=str, required=True,
                      help="Path to the trained value model checkpoint")
    parser.add_argument("--value_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                      help="Base model name/path for the ValueTransformer")
    parser.add_argument("--guard_model_name", type=str, default="llama-Guard-3-1B",
                      help="Model name/path for LlamaGuard")
    
    # Evaluation parameters
    parser.add_argument("--harmful_samples", type=int, default=500,
                      help="Number of harmful samples to use from HarmBench and LibrAI/do-not-answer")
    parser.add_argument("--safe_samples", type=int, default=500,
                      help="Number of safe samples to use for evaluation")
    parser.add_argument("--threshold", type=float, default=0.0,
                      help="Threshold for value model predictions")
    
    # Other parameters
    parser.add_argument("--cuda_device", type=str, default=None,
                      help="CUDA device to use (e.g., 'cuda:0'), defaults to 'cuda' if available")
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    with open("hf_api", "r") as f:
        auth_token = f.read().strip()
    
    # Determine device
    if args.cuda_device:
        device = torch.device(args.cuda_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize evaluator
    evaluator = ValueEvaluator(
        value_model_path=args.value_model_path,
        value_model_name=args.value_model_name,
        guard_model_name=args.guard_model_name,
        device=device,
        hf_token=auth_token
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        harmful_samples=args.harmful_samples,
        safe_samples=args.safe_samples,
        threshold=args.threshold
    )
    
    print("\nEvaluation completed!")
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print(f"Value Model Accuracy: {results['value_model']['accuracy']:.4f}")
    print(f"LlamaGuard Accuracy: {results['llamaguard']['accuracy']:.4f}")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.txt"
    
    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        
        f.write("Value Model:\n")
        f.write(f"  Accuracy: {results['value_model']['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['value_model']['precision']:.4f}\n")
        f.write(f"  Recall: {results['value_model']['recall']:.4f}\n")
        f.write(f"  F1: {results['value_model']['f1']:.4f}\n\n")
        
        f.write("LlamaGuard:\n")
        f.write(f"  Accuracy: {results['llamaguard']['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['llamaguard']['precision']:.4f}\n")
        f.write(f"  Recall: {results['llamaguard']['recall']:.4f}\n")
        f.write(f"  F1: {results['llamaguard']['f1']:.4f}\n")
    
    print(f"Detailed results saved to {results_file}")

if __name__ == "__main__":
    main() 