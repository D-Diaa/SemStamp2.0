"""
Fixed vs context-dependent watermark robustness analysis.

Compares watermark strength before and after paraphrasing for all four methods
(lsh, lsh_fixed, kmeans, kmeans_fixed) with a single paraphraser suffix.

Usage:
    python -m visualization.robustness data/c4-val-250 --suffix SUFFIX
    python -m visualization.robustness data/c4-val-250 --suffix SUFFIX --output figures/
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from visualization.utils import (
    METHODS, COLORS, LABELS, METHOD_GROUPS,
    setup_plot_style, filter_nan, save_figure, suffix_to_label,
    load_z_scores, load_detection_results,
)


# --- Data Loading ---

def load_data(base_path: str, suffix: str) -> dict:
    """Load z-scores and ROC data from all method directories."""
    data = {}

    for method in METHODS:
        dir_path = os.path.join(base_path, f"{method}{suffix}")

        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue

        method_data = load_z_scores(dir_path)

        # Load results CSV for AUROC
        detection = load_detection_results(dir_path)
        method_data.update(detection)

        data[method] = method_data

        # Report NaN values
        for key in ['z_scores', 'para_z_scores', 'human_z_scores']:
            if key in method_data:
                nan_count = np.isnan(method_data[key]).sum()
                if nan_count > 0:
                    print(f"Warning: {method}/{key} contains {nan_count} NaN values")

    return data


# --- Plot Functions ---

def plot_zscore_comparison_bars(data: dict, output_dir: str):
    """Paired bar chart comparing z-scores before/after paraphrasing."""
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = list(data.keys())
    x = np.arange(len(methods))
    width = 0.35

    original_means = [np.nanmean(data[m]['z_scores']) for m in methods]
    original_stds = [np.nanstd(data[m]['z_scores']) for m in methods]
    para_means = [np.nanmean(data[m]['para_z_scores']) for m in methods]
    para_stds = [np.nanstd(data[m]['para_z_scores']) for m in methods]

    colors = [COLORS[m] for m in methods]
    ax.bar(x - width/2, original_means, width, yerr=original_stds,
           label='Original', capsize=3, color=colors, edgecolor='black')
    ax.bar(x + width/2, para_means, width, yerr=para_stds,
           label='After Paraphrase', capsize=3,
           color=colors, alpha=0.6, hatch='//', edgecolor='black')

    for i, m in enumerate(methods):
        retention = (para_means[i] / original_means[i]) * 100
        ax.annotate(f'{retention:.1f}%', xy=(x[i], max(original_means[i], para_means[i]) + original_stds[i] + 0.3),
                    ha='center', fontsize=9, fontweight='bold')

    human_mean = np.nanmean([np.nanmean(data[m]['human_z_scores']) for m in methods])
    ax.axhline(y=human_mean, color='gray', linestyle='--', alpha=0.7, label=f'Human baseline ({human_mean:.2f})')

    ax.set_ylabel('Z-Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_title('Watermark Strength: Original vs After Paraphrasing\n(Percentage shows retention rate)')
    ax.set_ylim(bottom=-1)

    fig.text(0.5, 0.01, 'Z-score measures watermark detectability. Higher = stronger watermark signal.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_figure(output_dir, 'zscore_comparison')


def plot_distribution_boxplots(data: dict, output_dir: str):
    """Box plots showing z-score distributions for fixed vs context-dependent."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    comparisons = [
        ('lsh', 'lsh_fixed', 'SemStamp (LSH)'),
        ('kmeans', 'kmeans_fixed', 'K-SemStamp (KMeans)')
    ]

    for row, (context_m, fixed_m, title) in enumerate(comparisons):
        # Original z-scores
        ax = axes[row, 0]
        box_data = [
            filter_nan(data[context_m]['z_scores']),
            filter_nan(data[fixed_m]['z_scores']),
            filter_nan(data[context_m]['human_z_scores'])
        ]
        bp = ax.boxplot(box_data, tick_labels=['Context', 'Fixed', 'Human'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS[context_m])
        bp['boxes'][1].set_facecolor(COLORS[fixed_m])
        bp['boxes'][2].set_facecolor('lightgray')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax.set_title(f'{title}: Original Z-Scores')
        ax.set_ylabel('Z-Score')
        ax.axhline(y=2.33, color='red', linestyle='--', alpha=0.5, label='z=2.33 (p=0.01)')
        ax.legend(loc='upper right', fontsize=8)

        # Paraphrased z-scores
        ax = axes[row, 1]
        box_data = [
            filter_nan(data[context_m]['para_z_scores']),
            filter_nan(data[fixed_m]['para_z_scores']),
            filter_nan(data[context_m]['human_z_scores'])
        ]
        bp = ax.boxplot(box_data, tick_labels=['Context', 'Fixed', 'Human'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS[context_m])
        bp['boxes'][1].set_facecolor(COLORS[fixed_m])
        bp['boxes'][2].set_facecolor('lightgray')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax.set_title(f'{title}: After Paraphrasing')
        ax.set_ylabel('Z-Score')
        ax.axhline(y=2.33, color='red', linestyle='--', alpha=0.5, label='z=2.33 (p=0.01)')
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Z-Score Distributions: Fixed vs Context-Dependent Modes', fontsize=14, y=1.02)

    fig.text(0.5, 0.01, 'Box plots show distribution spread. Higher z-scores above threshold (red line) indicate detectable watermarks.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_figure(output_dir, 'distribution_boxplots')


def plot_roc_curves(data: dict, output_dir: str):
    """ROC curves comparison for each method group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    method_groups = [
        (['lsh', 'lsh_fixed'], 'SemStamp (LSH)'),
        (['kmeans', 'kmeans_fixed'], 'K-SemStamp (KMeans)')
    ]

    for ax, (methods, title) in zip(axes, method_groups):
        for m in methods:
            if m not in data or 'fpr' not in data[m] or 'tpr' not in data[m]:
                print(f"Warning: Missing ROC data for {m}")
                continue
            fpr = data[m]['fpr']
            tpr = data[m]['tpr']
            auroc = data[m].get('auroc', 0)
            linestyle = '--' if 'fixed' in m else '-'
            linewidth = 2.5 if 'fixed' in m else 2
            ax.plot(fpr, tpr, label=f'{LABELS[m]} (AUROC={auroc:.3f})',
                    color=COLORS[m], linestyle=linestyle, linewidth=linewidth)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{title}: ROC Curves (After Paraphrasing)')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')

    fig.text(0.5, 0.01, 'ROC curves show detection accuracy. Higher AUROC = better discrimination between watermarked and human text.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(output_dir, 'roc_curves')


def plot_degradation_analysis(data: dict, output_dir: str):
    """Horizontal bar chart showing z-score degradation after paraphrasing."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(data.keys())

    degradations = []
    retentions = []
    for m in methods:
        orig = np.nanmean(data[m]['z_scores'])
        para = np.nanmean(data[m]['para_z_scores'])
        degradations.append(orig - para)
        retentions.append((para / orig) * 100)

    y_pos = np.arange(len(methods))
    colors = [COLORS[m] for m in methods]

    bars = ax.barh(y_pos, degradations, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS[m] for m in methods])
    ax.set_xlabel('Z-Score Degradation (Lower = More Robust)')
    ax.set_title('Watermark Robustness: Z-Score Degradation After Paraphrasing')

    # Improvement summary for each method group
    summary_parts = []
    for group_name, (context_m, fixed_m) in METHOD_GROUPS.items():
        if context_m in data and fixed_m in data:
            ci = methods.index(context_m)
            fi = methods.index(fixed_m)
            improvement = ((degradations[ci] - degradations[fi]) / degradations[ci]) * 100
            summary_parts.append(f'  {group_name.upper()}: {improvement:.0f}% less degradation')

    max_deg = max(degradations)
    for i, (bar, deg, ret) in enumerate(zip(bars, degradations, retentions)):
        ax.annotate(f'{deg:.2f} ({ret:.0f}% retained)',
                    xy=(max_deg + 0.08, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10)

    if summary_parts:
        summary_text = 'Improvement from Fixed mode:\n' + '\n'.join(summary_parts)
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
                fontsize=10, color='darkgreen', fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4, edgecolor='green'))

    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlim(right=max_deg + 0.8)

    fig.text(0.5, 0.01, 'Degradation = original z-score minus paraphrased z-score. Lower degradation means watermark survives paraphrasing better.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(output_dir, 'degradation_analysis')


def plot_scatter_correlation(data: dict, output_dir: str):
    """2x2 scatter plots showing correlation between original and paraphrased z-scores."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    methods = list(data.keys())

    for ax, m in zip(axes.flat, methods):
        z_orig = data[m]['z_scores']
        z_para = data[m]['para_z_scores']

        valid_mask = ~(np.isnan(z_orig) | np.isnan(z_para))
        z_orig_valid = z_orig[valid_mask]
        z_para_valid = z_para[valid_mask]

        correlation = np.corrcoef(z_orig_valid, z_para_valid)[0, 1]

        ax.scatter(z_orig_valid, z_para_valid, alpha=0.6, c=COLORS[m],
                   edgecolors='black', s=50, linewidth=0.5)

        all_vals = np.concatenate([z_orig_valid, z_para_valid])
        lims = [min(all_vals) - 0.5, max(all_vals) + 0.5]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect retention', linewidth=1.5)

        z = np.polyfit(z_orig_valid, z_para_valid, 1)
        p = np.poly1d(z)
        ax.plot(lims, p(lims), color='red', linestyle='-', alpha=0.7,
                label=f'Fit (r={correlation:.2f}, slope={z[0]:.2f})', linewidth=2)

        ax.set_xlabel('Original Z-Score')
        ax.set_ylabel('Paraphrased Z-Score')
        ax.set_title(LABELS[m])
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')

    # Hide unused axes if fewer than 4 methods
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)

    plt.suptitle('Per-Sample Z-Score Correlation: Original vs Paraphrased', fontsize=14, y=1.02)

    fig.text(0.5, 0.01, 'Each point is one text sample. Points closer to the diagonal (dashed) retain more watermark strength after paraphrasing.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_figure(output_dir, 'scatter_correlation')


# --- Summary ---

def generate_summary_table(data: dict, output_dir: str):
    """Generate CSV summary table with all metrics."""
    import pandas as pd
    rows = []
    for m in data:
        orig_mean = np.nanmean(data[m]['z_scores'])
        orig_std = np.nanstd(data[m]['z_scores'])
        para_mean = np.nanmean(data[m]['para_z_scores'])
        para_std = np.nanstd(data[m]['para_z_scores'])
        human_mean = np.nanmean(data[m]['human_z_scores'])

        retention = (para_mean / orig_mean) * 100
        degradation = orig_mean - para_mean

        rows.append({
            'Method': LABELS[m],
            'Original Z (mean)': f'{orig_mean:.3f}',
            'Original Z (std)': f'{orig_std:.3f}',
            'Paraphrased Z (mean)': f'{para_mean:.3f}',
            'Paraphrased Z (std)': f'{para_std:.3f}',
            'Human Z (mean)': f'{human_mean:.3f}',
            'Retention (%)': f'{retention:.1f}',
            'Degradation': f'{degradation:.3f}',
            'AUROC': f'{data[m].get("auroc", 0):.3f}',
            'FPR@1%': f'{data[m].get("fpr1", 0):.3f}',
            'FPR@5%': f'{data[m].get("fpr5", 0):.3f}',
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'summary_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: summary_metrics.csv")
    print("\nSummary Table:")
    print(df.to_string(index=False))


# --- Orchestrator ---

def run_visualization(data_path: str, suffix: str, output_dir: str = None):
    """Run all visualizations for a given data path and suffix.

    Returns:
        True if visualizations were generated, False if no data was found.
    """
    if output_dir is None:
        label = suffix_to_label(suffix)
        output_dir = os.path.join(data_path, 'figures', 'robustness', label)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    print(f"Directory suffix: {suffix}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    data = load_data(data_path, suffix)

    if len(data) == 0:
        print("Error: No data loaded. Check paths and suffix.")
        return False

    print(f"\nLoaded data for methods: {list(data.keys())}")
    print("-" * 50)

    print("\nGenerating visualizations...")

    plot_zscore_comparison_bars(data, output_dir)
    plot_distribution_boxplots(data, output_dir)
    plot_roc_curves(data, output_dir)
    plot_degradation_analysis(data, output_dir)
    plot_scatter_correlation(data, output_dir)
    generate_summary_table(data, output_dir)

    print("-" * 50)
    print(f"All visualizations saved to: {output_dir}/")
    return True


def main():
    setup_plot_style()

    parser = argparse.ArgumentParser(
        description='Generate visualizations comparing fixed vs context-dependent watermark modes'
    )
    parser.add_argument('data_path', type=str, help='Base path to data directories (e.g., data/c4-val-250)')
    parser.add_argument('--suffix', type=str, default='-generated-parrot-bigram=False-threshold=0.0',
                        help='Suffix for data directories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    run_visualization(args.data_path, args.suffix, args.output)


if __name__ == '__main__':
    main()
