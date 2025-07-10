"""
Plotting utilities for template ablation experiments.

This module provides functions to aggregate data from multiple training runs
and create publication-ready plots showing episodic mean reward with variance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import glob
import json
from datetime import datetime
import warnings

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def aggregate_reward_data(base_dir: str = "outputs", experiment_pattern: str = "template-ablation-*") -> pd.DataFrame:
    """
    Aggregate reward data from multiple template ablation experiments.
    
    Args:
        base_dir: Base directory containing experiment outputs
        experiment_pattern: Glob pattern to match experiment directories
        
    Returns:
        DataFrame with aggregated reward data
    """
    all_data = []
    
    # Find all experiment directories
    experiment_dirs = glob.glob(os.path.join(base_dir, experiment_pattern))
    
    for exp_dir in experiment_dirs:
        # Extract template count from directory name
        exp_name = os.path.basename(exp_dir)
        if "template-ablation-" in exp_name:
            template_count = int(exp_name.split("-")[-1])
        else:
            template_count = -1  # Unknown
        
        # Look for episodic_rewards.csv
        rewards_csv_path = os.path.join(exp_dir, "episodic_rewards.csv")
        if os.path.exists(rewards_csv_path):
            try:
                df = pd.read_csv(rewards_csv_path)
                df['template_count'] = template_count
                df['experiment_dir'] = exp_dir
                all_data.append(df)
                print(f"Loaded data from {exp_dir}: {len(df)} episodes")
            except Exception as e:
                print(f"Error loading {rewards_csv_path}: {e}")
        else:
            print(f"No episodic_rewards.csv found in {exp_dir}")
    
    if not all_data:
        raise ValueError(f"No reward data found in {base_dir} matching pattern {experiment_pattern}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by template count and episode
    combined_df = combined_df.sort_values(['template_count', 'episode'])
    
    print(f"Aggregated data summary:")
    print(f"Total episodes: {len(combined_df)}")
    print(f"Template counts: {sorted(combined_df['template_count'].unique())}")
    print(f"Experiments: {len(combined_df['experiment_id'].unique())}")
    
    return combined_df

def create_reward_plot(
    data: pd.DataFrame,
    output_path: str = "episodic_rewards_plot.png",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    show_std: bool = True,
    smoothing_window: Optional[int] = None,
    colors: Optional[List[str]] = None
) -> None:
    """
    Create the main episodic reward plot with variance shading.
    
    Args:
        data: DataFrame with reward data (from aggregate_reward_data)
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        dpi: DPI for the saved image
        show_std: Whether to show standard deviation shading
        smoothing_window: Window size for moving average smoothing (None for no smoothing)
        colors: Custom colors for each template count
    """
    plt.figure(figsize=figsize)
    
    # Define colors if not provided
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Get unique template counts and sort them
    template_counts = sorted(data['template_count'].unique())
    
    for i, template_count in enumerate(template_counts):
        subset = data[data['template_count'] == template_count].copy()
        
        if len(subset) == 0:
            continue
            
        # Sort by episode
        subset = subset.sort_values('episode')
        
        episodes = subset['episode'].values
        mean_rewards = subset['mean_reward'].values
        std_rewards = subset['batch_std'].values
        
        # Apply smoothing if requested
        if smoothing_window is not None and len(mean_rewards) > smoothing_window:
            mean_rewards = pd.Series(mean_rewards).rolling(window=smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            std_rewards = pd.Series(std_rewards).rolling(window=smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        # Plot mean line
        color = colors[i % len(colors)]
        label = f'{template_count} templates'
        plt.plot(episodes, mean_rewards, label=label, linewidth=2.5, color=color)
        
        # Add variance shading
        if show_std:
            plt.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.3,
                color=color,
                edgecolor='none'
            )
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Mean Reward', fontsize=14)
    plt.title('Episodic Mean Reward by Template Count', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.show()

def create_comparison_plots(
    data: pd.DataFrame,
    output_dir: str = "plots",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> None:
    """
    Create multiple comparison plots for comprehensive analysis.
    
    Args:
        data: DataFrame with reward data
        output_dir: Directory to save plots
        figsize: Figure size
        dpi: DPI for saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Main reward plot
    create_reward_plot(
        data, 
        output_path=os.path.join(output_dir, "episodic_rewards_main.png"),
        figsize=figsize,
        dpi=dpi
    )
    
    # 2. Smoothed reward plot
    create_reward_plot(
        data,
        output_path=os.path.join(output_dir, "episodic_rewards_smoothed.png"),
        figsize=figsize,
        dpi=dpi,
        smoothing_window=5
    )
    
    # 3. Final performance comparison
    plt.figure(figsize=(10, 6))
    final_performance = data.groupby('template_count')['mean_reward'].agg(['mean', 'std', 'count']).reset_index()
    
    plt.bar(
        final_performance['template_count'].astype(str),
        final_performance['mean'],
        yerr=final_performance['std'],
        capsize=5,
        alpha=0.7,
        color='skyblue',
        edgecolor='navy',
        linewidth=1
    )
    
    plt.xlabel('Number of Templates', fontsize=14)
    plt.ylabel('Final Mean Reward', fontsize=14)
    plt.title('Final Performance Comparison', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_performance_comparison.png"), dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # 4. Learning curves by template count
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    template_counts = sorted(data['template_count'].unique())
    
    for i, template_count in enumerate(template_counts):
        if i >= len(axes):
            break
            
        subset = data[data['template_count'] == template_count]
        subset = subset.sort_values('episode')
        
        ax = axes[i]
        ax.plot(subset['episode'], subset['mean_reward'], linewidth=2, color='blue')
        ax.fill_between(
            subset['episode'],
            subset['mean_reward'] - subset['batch_std'],
            subset['mean_reward'] + subset['batch_std'],
            alpha=0.3,
            color='blue'
        )
        
        ax.set_title(f'{template_count} Templates', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Reward')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves_by_template.png"), dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # 5. Variance analysis
    plt.figure(figsize=(10, 6))
    variance_data = data.groupby('template_count')['batch_variance'].agg(['mean', 'std']).reset_index()
    
    plt.bar(
        variance_data['template_count'].astype(str),
        variance_data['mean'],
        yerr=variance_data['std'],
        capsize=5,
        alpha=0.7,
        color='lightcoral',
        edgecolor='darkred',
        linewidth=1
    )
    
    plt.xlabel('Number of Templates', fontsize=14)
    plt.ylabel('Mean Batch Variance', fontsize=14)
    plt.title('Batch Variance by Template Count', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "variance_analysis.png"), dpi=dpi, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the experiments.
    
    Args:
        data: DataFrame with reward data
        
    Returns:
        DataFrame with summary statistics
    """
    summary_stats = []
    
    for template_count in sorted(data['template_count'].unique()):
        subset = data[data['template_count'] == template_count]
        
        # Overall statistics
        overall_mean = subset['mean_reward'].mean()
        overall_std = subset['mean_reward'].std()
        overall_variance = subset['batch_variance'].mean()
        
        # Final performance (last 10 episodes)
        final_episodes = subset.nlargest(10, 'episode')
        final_mean = final_episodes['mean_reward'].mean()
        final_std = final_episodes['mean_reward'].std()
        
        # Learning rate (slope of linear fit to first 20 episodes)
        early_episodes = subset.nsmallest(20, 'episode')
        if len(early_episodes) >= 5:
            slope = np.polyfit(early_episodes['episode'], early_episodes['mean_reward'], 1)[0]
        else:
            slope = np.nan
        
        summary_stats.append({
            'template_count': template_count,
            'total_episodes': len(subset),
            'overall_mean_reward': overall_mean,
            'overall_std_reward': overall_std,
            'overall_mean_variance': overall_variance,
            'final_mean_reward': final_mean,
            'final_std_reward': final_std,
            'learning_rate': slope,
            'experiments': len(subset['experiment_id'].unique())
        })
    
    summary_df = pd.DataFrame(summary_stats)
    return summary_df

def save_analysis_report(data: pd.DataFrame, output_dir: str = "analysis") -> None:
    """
    Save a comprehensive analysis report.
    
    Args:
        data: DataFrame with reward data
        output_dir: Directory to save analysis files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(data)
    
    # Save summary statistics
    summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
    
    # Save aggregated data
    data.to_csv(os.path.join(output_dir, "aggregated_reward_data.csv"), index=False)
    
    # Generate text report
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("Template Ablation Experiment Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Experiment Summary:\n")
        f.write(f"Total episodes: {len(data)}\n")
        f.write(f"Template counts tested: {sorted(data['template_count'].unique())}\n")
        f.write(f"Number of experiments: {len(data['experiment_id'].unique())}\n\n")
        
        f.write("Performance Summary:\n")
        f.write(summary_stats.to_string(index=False))
        f.write("\n\n")
        
        # Find best performing configuration
        best_config = summary_stats.loc[summary_stats['final_mean_reward'].idxmax()]
        f.write(f"Best performing configuration: {best_config['template_count']} templates\n")
        f.write(f"Final mean reward: {best_config['final_mean_reward']:.4f}\n")
        f.write(f"Learning rate: {best_config['learning_rate']:.4f}\n")
    
    print(f"Analysis report saved to {output_dir}")

def main():
    """
    Main function to run the complete analysis pipeline.
    """
    print("Starting template ablation experiment analysis...")
    
    try:
        # Aggregate data
        print("Aggregating reward data...")
        data = aggregate_reward_data()
        
        # Create plots
        print("Creating plots...")
        create_comparison_plots(data)
        
        # Generate analysis report
        print("Generating analysis report...")
        save_analysis_report(data)
        
        print("Analysis complete! Check the 'plots' and 'analysis' directories for results.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 