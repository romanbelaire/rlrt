"""
Visualization utilities for semantic drift analysis.
"""

import os
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def generate_all_plots(all_metrics_data: List[Dict], output_dir: str):
    """Generates and saves all plots for the analysis."""
    if not all_metrics_data:
        raise RuntimeError("No metrics data provided to plotting.")

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(all_metrics_data)
    
    # Plot each metric vs. time
    for voice in ['1a', '1b', '1c', '1d']:
        for context_range in ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']:
            metric_key = f'similarity_{voice}_{context_range}'
            plot_metric_over_time(df, metric_key, output_dir)

    # Consolidated figures: by voice (rows=harm metrics, cols=context ranges)
    for voice in ['1a', '1b', '1c', '1d']:
        plot_consolidated_by_voice(df, voice, output_dir, style='binned')
        plot_consolidated_by_voice(df, voice, output_dir, style='hexbin')

    # Consolidated figures: by harm metric (rows=voices, cols=context ranges)
    for score_key in ['llamaguard_scores', 'v_high_scores', 'q_low_scores']:
        plot_consolidated_by_harm(df, score_key, output_dir, style='binned')
        plot_consolidated_by_harm(df, score_key, output_dir, style='hexbin')

    # Consolidated figures: by time (rows=voices, cols=context ranges)
    plot_consolidated_by_time(df, output_dir, style='binned')
    plot_consolidated_by_time(df, output_dir, style='hexbin')
    plot_consolidated_by_time(df, output_dir, style='line')


def plot_metric_over_time(df: pd.DataFrame, metric_key: str, output_dir: str):
    """Plots a single metric over conversation time."""
    plt.figure(figsize=(12, 6))
    
    # Explode the lists into separate rows to plot time series
    if 'time' not in df.columns or metric_key not in df.columns:
        raise RuntimeError(f"Missing required columns for plotting over time: time or {metric_key}")
    plot_df = df.explode('time').explode(metric_key)
    plot_df[metric_key] = pd.to_numeric(plot_df[metric_key], errors='coerce')
    plot_df = plot_df.dropna(subset=[metric_key])

    if plot_df.empty:
        raise RuntimeError(f"Empty dataframe for plotting {metric_key} vs time.")

    sns.lineplot(data=plot_df, x='time', y=metric_key, errorbar='sd', label='Mean Similarity')
    
    plt.title(f'{metric_key} vs. Conversation Time')
    plt.xlabel('Conversation Turn')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{metric_key}_vs_time.png'))
    plt.close()


def _prepare_xy_frame(df: pd.DataFrame, metric_key: str, score_key: str) -> pd.DataFrame:
    """Explode list-like columns into aligned rows, coerce to numeric, and drop NA.
    Fails fast if columns missing or result is empty.
    """
    required_cols = [metric_key, score_key]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column for plotting: {col}")
    plot_df = df.explode(metric_key).explode(score_key)
    plot_df[metric_key] = pd.to_numeric(plot_df[metric_key], errors='coerce')
    plot_df[score_key] = pd.to_numeric(plot_df[score_key], errors='coerce')
    plot_df = plot_df.dropna(subset=[metric_key, score_key])
    if plot_df.empty:
        raise RuntimeError(f"Empty dataframe for plotting {metric_key} vs {score_key}.")
    return plot_df


def _prepare_xy_frame_time(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """Explode list-like columns into aligned rows, coerce to numeric, and drop NA.
    Fails fast if columns missing or result is empty.
    """
    if 'time' not in df.columns or metric_key not in df.columns:
        raise RuntimeError(f"Missing required column for plotting over time: time or {metric_key}")
    
    # Create a list of tuples with (time, metric) pairs, then explode
    plot_data = []
    for _, row in df.iterrows():
        times = row['time'] if isinstance(row['time'], list) else [row['time']]
        metrics = row[metric_key] if isinstance(row[metric_key], list) else [row[metric_key]]
        
        # Ensure both lists have the same length by truncating to the shorter one
        min_len = min(len(times), len(metrics))
        for i in range(min_len):
            if times[i] is not None and metrics[i] is not None:
                plot_data.append({'time': times[i], metric_key: metrics[i]})
    
    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        raise RuntimeError(f"Empty dataframe for plotting {metric_key} vs time.")
    
    # Coerce to numeric and drop NA
    plot_df[metric_key] = pd.to_numeric(plot_df[metric_key], errors='coerce')
    plot_df = plot_df.dropna(subset=[metric_key])
    
    if plot_df.empty:
        raise RuntimeError(f"Empty dataframe for plotting {metric_key} vs time after numeric conversion.")
    
    return plot_df


def _save_current_fig(output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()


def _plot_hexbin(plot_df: pd.DataFrame, x: str, y: str, out_dir: str, fname: str):
    plt.figure(figsize=(12, 6))
    hb = plt.hexbin(plot_df[x], plot_df[y], gridsize=80, bins='log', cmap='viridis', mincnt=1)
    cbar = plt.colorbar(hb)
    cbar.set_label('log10(count)')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} vs {x} (hexbin density)')
    plt.grid(True, alpha=0.2)
    _save_current_fig(out_dir, fname)


def _plot_kde(plot_df: pd.DataFrame, x: str, y: str, out_dir: str, fname: str):
    plt.figure(figsize=(12, 6))
    ax = sns.kdeplot(data=plot_df, x=x, y=y, fill=True, levels=12, thresh=0.02, cmap='mako')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'{y} vs {x} (KDE contours)')
    plt.grid(True, alpha=0.2)
    _save_current_fig(out_dir, fname)


def _plot_binned_summary(plot_df: pd.DataFrame, x: str, y: str, out_dir: str, fname: str, num_bins: int = 50):
    plt.figure(figsize=(12, 6))
    # Equal-frequency bins for robustness
    q = pd.qcut(plot_df[x], q=num_bins, duplicates='drop')
    g = plot_df.groupby(q, observed=True)
    centers = g[x].median()
    med = g[y].median()
    lo = g[y].quantile(0.10)
    hi = g[y].quantile(0.90)

    plt.plot(centers, med, color='tab:blue', label='Median')
    plt.fill_between(centers, lo, hi, color='tab:blue', alpha=0.2, label='10–90%')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} vs {x} (binned median with 10–90% band)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_current_fig(out_dir, fname)


# -------- Subplot helpers for consolidated figures --------
def _compute_binned_summary(plot_df: pd.DataFrame, x: str, y: str, num_bins: int = 50):
    q = pd.qcut(plot_df[x], q=num_bins, duplicates='drop')
    g = plot_df.groupby(q, observed=True)
    centers = g[x].median()
    med = g[y].median()
    lo = g[y].quantile(0.10)
    hi = g[y].quantile(0.90)
    return centers, med, lo, hi


def _subplot_binned(ax, plot_df: pd.DataFrame, x: str, y: str):
    centers, med, lo, hi = _compute_binned_summary(plot_df, x, y)
    ax.plot(centers, med, color='tab:blue', label='Median')
    ax.fill_between(centers, lo, hi, color='tab:blue', alpha=0.2, label='10–90%')
    ax.grid(True, alpha=0.3)


def _subplot_hexbin(ax, plot_df: pd.DataFrame, x: str, y: str):
    hb = ax.hexbin(plot_df[x], plot_df[y], gridsize=40, bins='log', cmap='viridis', mincnt=1)
    # Omit colorbar in subplots for compactness
    ax.grid(True, alpha=0.2)


def _subplot_kde(ax, plot_df: pd.DataFrame, x: str, y: str):
    sns.kdeplot(data=plot_df, x=x, y=y, fill=True, levels=8, thresh=0.02, cmap='mako', ax=ax)
    ax.grid(True, alpha=0.2)


def _subplot_line(ax, plot_df: pd.DataFrame, x: str, y: str):
    """Line plot with error bars for time series data."""
    # Group by time and compute statistics
    plot_df = plot_df.sort_values(x)
    ax.plot(plot_df[x], plot_df[y], alpha=0.6, linewidth=1)
    ax.scatter(plot_df[x], plot_df[y], alpha=0.4, s=10)
    ax.grid(True, alpha=0.3)


def _draw_pair_on_ax(ax, plot_df: pd.DataFrame, x: str, y: str, style: str):
    if style == 'binned':
        _subplot_binned(ax, plot_df, x, y)
    elif style == 'hexbin':
        _subplot_hexbin(ax, plot_df, x, y)
    elif style == 'kde':
        _subplot_kde(ax, plot_df, x, y)
    else:
        raise RuntimeError(f"Unknown consolidated plot style: {style}")


def _draw_pair_on_ax_time(ax, plot_df: pd.DataFrame, metric_key: str, style: str):
    if style == 'binned':
        _subplot_binned(ax, plot_df, 'time', metric_key)
    elif style == 'hexbin':
        _subplot_hexbin(ax, plot_df, 'time', metric_key)
    elif style == 'line':
        _subplot_line(ax, plot_df, 'time', metric_key)
    elif style == 'kde':
        _subplot_kde(ax, plot_df, 'time', metric_key)
    else:
        raise RuntimeError(f"Unknown consolidated plot style: {style}")


def plot_consolidated_by_voice(df: pd.DataFrame, voice: str, output_dir: str, style: str = 'binned'):
    """Create a consolidated grid figure for a given voice.
    Rows are harm metrics; columns are context ranges. Each subplot shows similarity_{voice}_{context} vs harm metric.
    """
    harm_keys = ['llamaguard_scores', 'v_high_scores', 'q_low_scores']
    contexts = ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']

    nrows, ncols = len(harm_keys), len(contexts)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for r, score_key in enumerate(harm_keys):
        for c, context_range in enumerate(contexts):
            metric_key = f'similarity_{voice}_{context_range}'
            plot_df = _prepare_xy_frame(df, metric_key, score_key)
            ax = axes[r][c]
            _draw_pair_on_ax(ax, plot_df, x=score_key, y=metric_key, style=style)

            if r == 0:
                ax.set_title(context_range)
            if c == 0:
                ax.set_ylabel(score_key)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    fig.suptitle(f'Voice {voice}: similarity vs harm metrics ({style})', fontsize=14)
    fig.text(0.5, 0.02, f'X-axis: Harm metrics (llamaguard, v_high, q_low) | Y-axis: Similarity scores', 
             ha='center', fontsize=10, style='italic')
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'consolidated_by_voice_{voice}.{style}.png'), dpi=200)
    plt.close(fig)


def plot_consolidated_by_harm(df: pd.DataFrame, score_key: str, output_dir: str, style: str = 'binned'):
    """Create a consolidated grid figure for a given harm metric.
    Rows are voices; columns are context ranges. Each subplot shows similarity_{voice}_{context} vs given harm metric.
    """
    voices = ['1a', '1b', '1c', '1d']
    contexts = ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']

    nrows, ncols = len(voices), len(contexts)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for r, voice in enumerate(voices):
        for c, context_range in enumerate(contexts):
            metric_key = f'similarity_{voice}_{context_range}'
            plot_df = _prepare_xy_frame(df, metric_key, score_key)
            ax = axes[r][c]
            _draw_pair_on_ax(ax, plot_df, x=score_key, y=metric_key, style=style)

            if r == 0:
                ax.set_title(context_range)
            if c == 0:
                ax.set_ylabel(voice)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    fig.suptitle(f'Harm {score_key}: similarity metrics ({style})', fontsize=14)
    fig.text(0.5, 0.02, f'X-axis: {score_key} | Y-axis: Similarity scores', 
             ha='center', fontsize=10, style='italic')
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'consolidated_by_harm_{score_key}.{style}.png'), dpi=200)
    plt.close(fig)


def plot_consolidated_by_time(df: pd.DataFrame, output_dir: str, style: str = 'binned'):
    """Create a consolidated grid figure for voice vs time relationships.
    Rows are voices; columns are context ranges. Each subplot shows similarity_{voice}_{context} vs time.
    """
    voices = ['1a', '1b', '1c', '1d']
    contexts = ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']

    nrows, ncols = len(voices), len(contexts)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for r, voice in enumerate(voices):
        for c, context_range in enumerate(contexts):
            metric_key = f'similarity_{voice}_{context_range}'
            plot_df = _prepare_xy_frame_time(df, metric_key)
            ax = axes[r][c]
            _draw_pair_on_ax_time(ax, plot_df, metric_key, style=style)

            if r == 0:
                ax.set_title(context_range)
            if c == 0:
                ax.set_ylabel(voice)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    fig.suptitle(f'Voice vs Time: similarity metrics ({style})', fontsize=14)
    fig.text(0.5, 0.02, f'X-axis: Conversation time (turn number) | Y-axis: Similarity scores', 
             ha='center', fontsize=10, style='italic')
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'consolidated_by_time.{style}.png'), dpi=200)
    plt.close(fig)


def plot_metric_vs_scores(df: pd.DataFrame, metric_key: str, output_dir: str):
    """For a given metric, generate density and robust-summary plots against each score series."""
    for score_key in ['llamaguard_scores', 'v_high_scores', 'q_low_scores']:
        plot_df = _prepare_xy_frame(df, metric_key, score_key)
        # Hexbin density
        _plot_hexbin(
            plot_df, x=score_key, y=metric_key, out_dir=output_dir,
            fname=f'{metric_key}_vs_{score_key}.hexbin.png'
        )
        # KDE contours
        _plot_kde(
            plot_df, x=score_key, y=metric_key, out_dir=output_dir,
            fname=f'{metric_key}_vs_{score_key}.kde.png'
        )
        # Binned robust summary
        _plot_binned_summary(
            plot_df, x=score_key, y=metric_key, out_dir=output_dir,
            fname=f'{metric_key}_vs_{score_key}.binned.png'
        )


def generate_summary_report(all_metrics_data: List[Dict], output_dir: str):
    """Generates a markdown summary report of the analysis, including correlation ranking per score."""
    if not all_metrics_data:
        raise RuntimeError("No metrics data provided to summary report.")

    df = pd.DataFrame(all_metrics_data)
    report = []

    report.append("# Semantic Drift Analysis Summary")
    report.append(f"\n**Total Conversations Analyzed:** {len(df)}")
    
    # Overall statistics
    report.append("\n## Overall Statistics")
    for voice in ['1a', '1b', '1c', '1d']:
        for context_range in ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']:
            metric_key = f'similarity_{voice}_{context_range}'
            if metric_key not in df.columns:
                raise RuntimeError(f"Missing metric {metric_key} in summary generation.")
            # Flatten the list of lists and remove None values
            all_values = [item for sublist in df[metric_key].dropna() for item in sublist if item is not None]
            if not all_values:
                continue
            
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            report.append(f"- **{metric_key}:** Mean={mean_val:.4f}, Std={std_val:.4f}")

    # Correlation rankings per score
    report.append("\n## Correlation Rankings (by absolute Spearman)")
    for score_key in ['llamaguard_scores', 'v_high_scores', 'q_low_scores']:
        rows = []
        for voice in ['1a', '1b', '1c', '1d']:
            for context_range in ['p2c', 'p2p', 'p2c_cu', 'p2o', 'p2f']:
                metric_key = f'similarity_{voice}_{context_range}'
                if metric_key not in df.columns or score_key not in df.columns:
                    raise RuntimeError(f"Missing columns for correlation: {metric_key} or {score_key}")
                plot_df = df.explode(metric_key).explode(score_key)
                x = pd.to_numeric(plot_df[score_key], errors='coerce')
                y = pd.to_numeric(plot_df[metric_key], errors='coerce')
                plot_df = pd.DataFrame({score_key: x, metric_key: y}).dropna()
                if plot_df.empty:
                    continue
                pearson = plot_df[score_key].corr(plot_df[metric_key], method='pearson')
                spearman = plot_df[score_key].corr(plot_df[metric_key], method='spearman')
                rows.append((metric_key, pearson, spearman))
        if not rows:
            continue
        # Sort by absolute Spearman
        rows.sort(key=lambda t: abs(t[2]), reverse=True)
        report.append(f"\n### vs. {score_key}")
        for metric_key, pearson, spearman in rows[:15]:
            report.append(f"- **{metric_key}**: Pearson={pearson:.4f}, Spearman={spearman:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'analysis_summary.md'), 'w') as f:
        f.write('\n'.join(report)) 