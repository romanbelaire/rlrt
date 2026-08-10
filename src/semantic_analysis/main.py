"""
Main entry point for semantic analysis data collection.
"""

import argparse
import json
import os
import sys
from typing import Optional

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AgentConfig
from src.semantic_analysis.collector import SemanticDataCollector
from src.semantic_analysis.harmbench_loader import load_harmbench_data, get_harmbench_statistics


def main():
    """Main entry point for semantic analysis data collection."""
    parser = argparse.ArgumentParser(description="Semantic Analysis Data Collection")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/semantic_analysis_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max_conversations", 
        type=int, 
        default=None,
        help="Maximum number of conversations to process"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=None,
        help="Sample size for HarmBench data (if not specified, uses all data)"
    )
    parser.add_argument(
        "--categories", 
        nargs="+", 
        default=None,
        help="Specific categories to include (if not specified, uses all categories)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Path to checkpoint directory to load trained models from (required for semantic analysis)"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Dry run mode - load data and show statistics without processing"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config_path}")
    try:
        config = AgentConfig.from_json(args.config_path)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config_path}")
        print("Creating default configuration...")
        config = AgentConfig()
        # Add semantic analysis config if not present
        if config.semantic_analysis is None:
            from src.config import SemanticAnalysisConfig
            config.semantic_analysis = SemanticAnalysisConfig()
    
    # Override output directory if specified
    if args.output_dir:
        config.semantic_analysis.output_dir = args.output_dir
    
    # Load checkpoint configuration
    print(f"Loading configuration from checkpoint: {args.checkpoint_dir}")
    checkpoint_config_path = os.path.join(args.checkpoint_dir, "config.json")
    if not os.path.exists(checkpoint_config_path):
        raise FileNotFoundError(f"Checkpoint configuration not found: {checkpoint_config_path}")
    
    try:
        with open(checkpoint_config_path, 'r') as f:
            checkpoint_config_dict = json.load(f)
        print(f"Loaded configuration from checkpoint: {checkpoint_config_path}")
        
        # Merge configurations: use checkpoint model config, keep other configs from provided file
        if "model" in checkpoint_config_dict:
            print("Using model configuration from checkpoint")
            # Update the config object with checkpoint model config
            checkpoint_model_config = AgentConfig.from_dict(checkpoint_config_dict).model
            config.model = checkpoint_model_config
        else:
            print("No model configuration found in checkpoint, using provided config")
    except Exception as e:
        raise RuntimeError(f"Failed to load config from checkpoint: {e}")
    
    # Load HarmBench data
    print("Loading HarmBench data...")
    try:
        harmbench_data = load_harmbench_data(config.semantic_analysis)
    except FileNotFoundError as e:
        print(f"Error loading HarmBench data: {e}")
        print("Please ensure the HarmBench data file exists and the path is correct.")
        return 1
    
    # Get and display statistics
    stats = get_harmbench_statistics(harmbench_data, config.semantic_analysis)
    print("\nHarmBench Dataset Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Unique questions: {stats['unique_questions']}")
    print(f"  Duplicate questions: {stats['duplicate_questions']}")
    if 'total_categories' in stats:
        print(f"  Total categories: {stats['total_categories']}")
        print(f"  Most common category: {stats['most_common_category']}")
        print(f"  Least common category: {stats['least_common_category']}")
    print(f"  Average question length: {stats['avg_question_length']:.1f} characters")
    print(f"  Question length range: {stats['min_question_length']} - {stats['max_question_length']} characters")
    
    # Filter data if categories specified
    if args.categories:
        print(f"\nFiltering data to categories: {args.categories}")
        harmbench_data = harmbench_data[harmbench_data[config.semantic_analysis.category_column].isin(args.categories)]
        print(f"Filtered dataset: {len(harmbench_data)} entries")
    
    # Sample data if sample_size specified
    if args.sample_size:
        print(f"\nSampling {args.sample_size} entries from dataset...")
        from src.semantic_analysis.harmbench_loader import sample_harmbench_data
        harmbench_data = sample_harmbench_data(harmbench_data, args.sample_size, config.semantic_analysis)
        print(f"Sampled dataset: {len(harmbench_data)} entries")
    
    # Dry run mode
    if args.dry_run:
        print("\nDry run mode - showing data statistics only")
        print("Dataset is ready for processing.")
        return 0
    
    # Initialize collector
    print("\nInitializing semantic analysis collector...")
    try:
        collector = SemanticDataCollector(config, checkpoint_dir=args.checkpoint_dir)
    except Exception as e:
        print(f"Error initializing collector: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Process conversations
    print(f"\nStarting data collection for {len(harmbench_data)} HarmBench entries...")
    try:
        collector.collect_dataset(harmbench_data, max_conversations=args.max_conversations)
        print("\nData collection completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 