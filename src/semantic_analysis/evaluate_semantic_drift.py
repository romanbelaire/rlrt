"""
Main script to run semantic drift analysis.
"""

import argparse
import os
import sys

# Add the parent directory to the path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from src.semantic_analysis.analysis_engine import AnalysisEngine
from src.semantic_analysis.recompute_scores import recompute_scores_for_dataset


def main():
    """Main entry point for semantic analysis evaluation."""
    parser = argparse.ArgumentParser(description="Semantic Drift Analysis")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the collected semantic analysis data (root containing conversations/)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the analysis results. Defaults to --data_dir."
    )
    parser.add_argument(
        "--downsample_dim",
        type=int,
        required=True,
        help="Dimension to down-sample embeddings to using PCA. Required to ensure consistent embedding spaces."
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to process."
    )
    parser.add_argument(
        "--use_input_embeddings",
        action="store_true",
        help="Use token input embeddings (embedding table) instead of last hidden state for analysis."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute q_low, v_high, and running-max harm metrics from saved conversations before analysis. Overwrites existing JSON and vector store entries."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to checkpoint with critics/policy for score recomputation (required if --recompute)."
    )
    args = parser.parse_args()

    # Optional recomputation step
    if args.recompute:
        if not args.checkpoint_dir:
            raise RuntimeError("--recompute requires --checkpoint_dir to load critics/policy for scoring")
        recompute_scores_for_dataset(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
        )

    resolved_output = args.output_dir or args.data_dir
    print(f"Using output directory: {resolved_output}")

    # Initialize and run the analysis engine
    engine = AnalysisEngine(
        data_dir=args.data_dir,
        output_dir=resolved_output,
        downsample_dim=args.downsample_dim,
        max_conversations=args.max_conversations,
        use_input_embeddings=args.use_input_embeddings
    )
    engine.run_analysis()


if __name__ == "__main__":
    main() 