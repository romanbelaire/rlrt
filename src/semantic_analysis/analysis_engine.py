"""
Main analysis engine for semantic drift evaluation.
"""

import os
import json
from tqdm import tqdm
import torch

from src.semantic_analysis.data_loader import load_conversations, get_embedding_model
from src.semantic_analysis.downsampler import reduce_embedding_dimensions
from src.semantic_analysis.drift_calculator import compute_all_semantic_drift_metrics, compute_all_semantic_drift_metrics_input
from src.semantic_analysis.visualizer import generate_all_plots, generate_summary_report


class AnalysisEngine:
    """Orchestrates the semantic drift analysis pipeline."""
    
    def __init__(self, data_dir: str, output_dir: str, downsample_dim: int = None, max_conversations: int = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', use_input_embeddings: bool = False):
        self.data_dir = data_dir
        # Default to saving alongside data if caller passes empty string
        self.output_dir = output_dir or data_dir
        self.downsample_dim = downsample_dim
        self.max_conversations = max_conversations
        self.device = device
        self.use_input_embeddings = use_input_embeddings
        
        # Create output directories
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'reports'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'metadata'), exist_ok=True)
        
        self.embedding_model = None
        self.tokenizer = None

    def _initialize_embedding_model(self):
        """Initializes the embedding model and tokenizer."""
        print("Initializing embedding model...")
        self.embedding_model, self.tokenizer = get_embedding_model(device=self.device)
        self.embedding_model.to(self.device)

    def _ensure_running_max_columns(self, metrics: dict) -> dict:
        """Ensure running-max harm columns exist; compute from base series if missing.
        Fail fast if base series missing.
        """
        required_base = ['llamaguard_scores', 'v_high_scores', 'q_low_scores']
        for key in required_base:
            if key not in metrics or not isinstance(metrics[key], list):
                raise RuntimeError(f"Missing required metric series: {key}")
        # Compute running max lists
        def running_max(vals):
            out = []
            current = None
            for v in vals:
                if v is None:
                    out.append(current)
                else:
                    current = v if current is None else max(current, v)
                    out.append(current)
            return out
        if 'max_llamaguard_to_t' not in metrics:
            metrics['max_llamaguard_to_t'] = running_max(metrics['llamaguard_scores'])
        if 'max_v_high_to_t' not in metrics:
            metrics['max_v_high_to_t'] = running_max(metrics['v_high_scores'])
        if 'max_q_low_to_t' not in metrics:
            metrics['max_q_low_to_t'] = running_max(metrics['q_low_scores'])
        return metrics

    def _write_metrics_csv(self, all_metrics_data):
        """Write flattened metrics as CSV to metadata directory inside output_dir."""
        import pandas as pd
        rows = []
        for m in all_metrics_data:
            num_rows = len(m.get('time', []))
            for i in range(num_rows):
                row = {
                    'conversation_id': m['conversation_id'],
                    'category': m.get('category'),
                    'original_question': m.get('original_question'),
                    'time': m['time'][i],
                    'llamaguard_score': m['llamaguard_scores'][i] if i < len(m['llamaguard_scores']) else None,
                    'v_high_score': m['v_high_scores'][i] if i < len(m['v_high_scores']) else None,
                    'q_low_score': m['q_low_scores'][i] if i < len(m['q_low_scores']) else None,
                    'max_llamaguard_to_t': m['max_llamaguard_to_t'][i] if i < len(m['max_llamaguard_to_t']) else None,
                    'max_v_high_to_t': m['max_v_high_to_t'][i] if i < len(m['max_v_high_to_t']) else None,
                    'max_q_low_to_t': m['max_q_low_to_t'][i] if i < len(m['max_q_low_to_t']) else None,
                }
                for voice in ['1a','1b','1c','1d']:
                    for rng in ['p2c','p2p','p2c_cu','p2o','p2f']:
                        key = f'similarity_{voice}_{rng}'
                        row[key] = m[key][i] if key in m and i < len(m[key]) else None
                rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, 'metadata', 'semantic_drift_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"Wrote CSV metrics to {csv_path}")

    def run_analysis(self):
        """Runs the complete analysis pipeline."""
        # 1. Load conversations
        print("Loading conversations...")
        conversations = load_conversations(self.data_dir, max_conversations=self.max_conversations)

        # 1b. Backfill missing original_prompt_embedding before any downsampling
        needs_prompt_embeddings = any(conv.original_prompt_embedding is None for conv in conversations)
        if needs_prompt_embeddings:
            print("Computing missing original_prompt_embedding values using hidden-state mean pooling...")
            if self.embedding_model is None or self.tokenizer is None:
                self._initialize_embedding_model()
            from src.semantic_analysis.data_loader import encode_text
            for conv in conversations:
                if conv.original_prompt_embedding is None:
                    conv.original_prompt_embedding = encode_text(
                        text=conv.original_question,
                        model=self.embedding_model,
                        tokenizer=self.tokenizer,
                        device=self.device,
                    )

        # 2. Optional downsampling applied across all conversations before analysis
        if self.downsample_dim is not None:
            print(f"Downsampling all embeddings to dimension {self.downsample_dim} with PCA...")
            reduce_embedding_dimensions(conversations, target_dim=self.downsample_dim, device=self.device)

        # 3. Initialize embedding model if using input embeddings
        if self.use_input_embeddings:
            self._initialize_embedding_model()

        # 4. Compute metrics
        print("Computing semantic drift metrics...")
        all_metrics_data = []
        for conv in tqdm(conversations, desc="Processing conversations"):
            metrics = (compute_all_semantic_drift_metrics_input if self.use_input_embeddings else compute_all_semantic_drift_metrics)(
                conversation=conv,
                embedding_model=self.embedding_model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            metrics = self._ensure_running_max_columns(metrics)
            all_metrics_data.append(metrics)
        
        # 5. Save metadata
        print("Saving metadata...")
        self.save_metrics_to_jsonl(all_metrics_data)
        self._write_metrics_csv(all_metrics_data)

        # 6. Generate visualizations
        print("Generating visualizations...")
        from src.semantic_analysis.visualizer import generate_all_plots
        generate_all_plots(all_metrics_data, os.path.join(self.output_dir, 'plots'))
        
        # 7. Generate summary statistics
        print("Generating summary report...")
        generate_summary_report(all_metrics_data, os.path.join(self.output_dir, 'reports'))
        
        print("\nAnalysis complete.")
        print(f"Results saved to: {self.output_dir}")

    def save_metrics_to_jsonl(self, all_metrics_data):
        """Saves the computed metrics to a JSONL file."""
        output_path = os.path.join(self.output_dir, "data", "semantic_drift_metrics.jsonl")
        with open(output_path, 'w') as f:
            for metrics in all_metrics_data:
                # Create a copy for JSON serialization without modifying the original
                json_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                        json_metrics[key] = [v.detach().cpu().float().numpy().tolist() for v in value]
                    else:
                        json_metrics[key] = value
                f.write(json.dumps(json_metrics) + '\n') 