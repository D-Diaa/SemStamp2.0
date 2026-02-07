"""
Watermark quality vs detectability visualization.

Compares quality and detectability of all watermarking methods (no paraphrases).
For each method, loads watermarked text quality (eval_quality_wm.csv) and
detection metrics (results_wm.csv) from the first available generated directory.

Usage:
    python -m visualization.watermark_quality data/c4-val-250
    python -m visualization.watermark_quality data/c4-val-250 --output figures/
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualization.utils import (
    METHODS, COLORS, LABELS,
    METRIC_LABELS, METRIC_HIGHER_BETTER,
    setup_plot_style, save_figure, load_quality_csv, load_detection_results,
)

# Metric lists for this analysis
WM_QUALITY_METRICS = ['gen_ppl', 'mauve', 'bert_F1', 'sem_ent']
WM_DETECT_METRICS = ['auroc', 'fpr1', 'fpr5', 'mean_z']


def load_method_data(base_path: str, method: str) -> dict:
    """Load watermarked quality and detectability data for a single method.

    Loads from the watermark-only directory ({method}-generated):
    - eval_quality.csv for quality metrics
    - results_wm.csv for detection metrics (auroc, fpr1, fpr5, mean_z, human_mean_z)

    Returns:
        Dict with quality and detection metrics, or empty dict if not found.
    """
    dir_path = os.path.join(base_path, f"{method}-generated")

    if not os.path.isdir(dir_path):
        return {}

    entry = {}

    quality = load_quality_csv(dir_path, 'eval_quality.csv')
    if quality is not None:
        entry['quality'] = quality

    wm_results = load_detection_results(dir_path, 'results_wm.csv')
    if wm_results:
        entry.update(wm_results)

    return entry


def load_all_methods(base_path: str) -> dict:
    """Load data for all watermarking methods."""
    data = {}
    for method in METHODS:
        entry = load_method_data(base_path, method)
        if entry:
            data[method] = entry
    return data


def plot_watermark_quality(data: dict, output_dir: str):
    """Generate a single figure comparing quality and detectability across methods.

    Layout: top row = quality metrics, bottom row = detectability metrics.
    """
    available = [m for m in METHODS if m in data and 'quality' in data[m]]
    if not available:
        print("No watermarked quality data found. Run quality evaluation with --column text first.")
        return

    metrics = [m for m in WM_QUALITY_METRICS if m in data[available[0]]['quality']]
    if not metrics:
        print("No recognized quality metrics found.")
        return

    detect_metrics = [m for m in WM_DETECT_METRICS if any(
        (m in data[method] if m != 'mean_z' else data[method].get('mean_z') is not None)
        for method in available
    )]
    if not detect_metrics:
        detect_metrics = ['mean_z']

    n_methods = len(available)
    n_quality = len(metrics)
    n_detect = len(detect_metrics)
    n_cols = max(n_quality, n_detect)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))

    x = np.arange(n_methods)
    colors = [COLORS[m] for m in available]
    labels = [LABELS[m] for m in available]

    # Top row: quality metrics
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        vals = [data[m]['quality'].get(metric, 0) for m in available]
        bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

        for bar, val in zip(bars, vals):
            fmt = f'{val:.1f}' if metric == 'gen_ppl' else f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha='center', va='bottom', fontsize=8)

        higher_better = METRIC_HIGHER_BETTER.get(metric, True)
        better = 'higher is better' if higher_better else 'lower is better'
        ax.set_title(f'{METRIC_LABELS.get(metric, metric)}\n({better})', fontsize=11)

    for i in range(n_quality, n_cols):
        axes[0, i].set_visible(False)

    # Bottom row: detectability metrics
    for i, metric in enumerate(detect_metrics):
        ax = axes[1, i]
        vals = [data[m].get(metric, 0) for m in available]
        bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

        for bar, val in zip(bars, vals):
            fmt = f'{val:.2f}' if metric == 'mean_z' else f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{METRIC_LABELS.get(metric, metric)}\n(higher is better)', fontsize=11)

        if metric == 'mean_z':
            human_z_vals = [data[m].get('human_mean_z') for m in available]
            human_z_vals = [v for v in human_z_vals if v is not None]
            if human_z_vals:
                human_mean = np.mean(human_z_vals)
                ax.axhline(y=human_mean, color='gray', linestyle='--', linewidth=1.5,
                           label=f'Human ({human_mean:.2f})')
                ax.legend(loc='best', fontsize=9)

    for i in range(n_detect, n_cols):
        axes[1, i].set_visible(False)

    fig.suptitle('Watermark Quality & Detectability Comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_figure(output_dir, 'watermark_quality')


def plot_watermark_quality_table(data: dict, output_dir: str):
    """Generate a summary CSV with watermark quality and detectability."""
    available = [m for m in METHODS if m in data]
    if not available:
        return

    rows = []
    for method in available:
        row = {'method': method, 'method_label': LABELS[method]}
        for key in ['auroc', 'fpr1', 'fpr5', 'mean_z', 'human_mean_z']:
            if key in data[method]:
                row[key] = data[method][key]
        if 'quality' in data[method]:
            row.update(data[method]['quality'])
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'watermark_quality_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: watermark_quality_summary.csv")


def run_watermark_quality_visualization(data_path: str, output_dir: str = None):
    """Main entry point for watermark quality visualization."""
    if output_dir is None:
        output_dir = os.path.join(data_path, 'figures', 'watermark_quality')

    print(f"Loading watermark quality data from: {data_path}")
    data = load_all_methods(data_path)

    if not data:
        print("No data found. Ensure detection and quality evaluation (--column text) have been run.")
        return

    for method, entry in data.items():
        has_q = 'quality' in entry
        has_z = 'mean_z' in entry
        print(f"  {LABELS[method]}: quality={'yes' if has_q else 'no'}, z-score={'yes' if has_z else 'no'}")

    print("-" * 50)
    plot_watermark_quality(data, output_dir)
    plot_watermark_quality_table(data, output_dir)
    print(f"All outputs saved to: {output_dir}/")


def main():
    setup_plot_style()

    parser = argparse.ArgumentParser(
        description='Visualize watermark quality and detectability across methods (no paraphrases).'
    )
    parser.add_argument('data_path', type=str,
                        help='Base data directory (e.g., data/c4-val-250)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    run_watermark_quality_visualization(args.data_path, args.output)


if __name__ == '__main__':
    main()
