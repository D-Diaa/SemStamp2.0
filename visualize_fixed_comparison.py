"""
Visualization script for comparing watermark robustness with fixed vs context-dependent modes.
Generates publication-quality figures from detection results.

Usage:
    python visualize_fixed_comparison.py data/c4-val-250 --output figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Configuration constants
METHODS = ['lsh', 'lsh_fixed', 'kmeans', 'kmeans_fixed']
COLORS = {
    'kmeans': '#1f77b4',       # blue
    'kmeans_fixed': '#2ca02c', # green
    'lsh': '#ff7f0e',          # orange
    'lsh_fixed': '#d62728',    # red
}
LABELS = {
    'kmeans': 'K-SemStamp (Context)',
    'kmeans_fixed': 'K-SemStamp (Fixed)',
    'lsh': 'SemStamp (Context)',
    'lsh_fixed': 'SemStamp (Fixed)',
}

# Publication quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations comparing fixed vs context-dependent watermark modes'
    )
    parser.add_argument('data_path', type=str, help='Base path to data directories (e.g., data/c4-val-250)')
    parser.add_argument('--suffix', type=str, default='-generated-parrot-bigram=False-threshold=0.0',
                        help='Suffix for data directories')
    parser.add_argument('--output', type=str, default='figures', help='Output directory for figures')
    return parser.parse_args()


def load_data(base_path: str, suffix: str) -> dict:
    """Load z-scores and ROC data from all method directories."""
    data = {}

    for method in METHODS:
        dir_path = os.path.join(base_path, f"{method}{suffix}")

        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue

        method_data = {}

        # Load z-scores
        for score_file in ['z_scores.npy', 'human_z_scores.npy', 'para_z_scores.npy']:
            file_path = os.path.join(dir_path, score_file)
            if os.path.exists(file_path):
                method_data[score_file.replace('.npy', '')] = np.load(file_path)

        # Load ROC data
        for roc_file in ['fpr.npy', 'tpr.npy']:
            file_path = os.path.join(dir_path, roc_file)
            if os.path.exists(file_path):
                method_data[roc_file.replace('.npy', '')] = np.load(file_path)

        # Load results CSV for AUROC (tab-separated)
        results_path = os.path.join(dir_path, 'results.csv')
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path, sep='\t')
            method_data['auroc'] = results_df['auroc'].values[0]
            method_data['fpr1'] = results_df['fpr1'].values[0]
            method_data['fpr5'] = results_df['fpr5'].values[0]

        data[method] = method_data

        # Report NaN values
        for key in ['z_scores', 'para_z_scores', 'human_z_scores']:
            if key in method_data:
                nan_count = np.isnan(method_data[key]).sum()
                if nan_count > 0:
                    print(f"Warning: {method}/{key} contains {nan_count} NaN values")

    return data


def plot_zscore_comparison_bars(data: dict, output_dir: str):
    """Figure 1: Paired bar chart comparing z-scores before/after paraphrasing."""
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = ['lsh', 'lsh_fixed', 'kmeans', 'kmeans_fixed']
    x = np.arange(len(methods))
    width = 0.35

    # Original z-scores
    original_means = [np.nanmean(data[m]['z_scores']) for m in methods]
    original_stds = [np.nanstd(data[m]['z_scores']) for m in methods]

    # Paraphrased z-scores
    para_means = [np.nanmean(data[m]['para_z_scores']) for m in methods]
    para_stds = [np.nanstd(data[m]['para_z_scores']) for m in methods]

    # Create bars
    colors = [COLORS[m] for m in methods]
    bars1 = ax.bar(x - width/2, original_means, width, yerr=original_stds,
                   label='Original', capsize=3, color=colors, edgecolor='black')
    bars2 = ax.bar(x + width/2, para_means, width, yerr=para_stds,
                   label='After Paraphrase', capsize=3,
                   color=colors, alpha=0.6, hatch='//', edgecolor='black')

    # Add retention percentage annotations
    for i, m in enumerate(methods):
        retention = (para_means[i] / original_means[i]) * 100
        ax.annotate(f'{retention:.1f}%', xy=(x[i], max(original_means[i], para_means[i]) + original_stds[i] + 0.3),
                    ha='center', fontsize=9, fontweight='bold')

    # Add human baseline line
    human_mean = np.nanmean([np.nanmean(data[m]['human_z_scores']) for m in methods])
    ax.axhline(y=human_mean, color='gray', linestyle='--', alpha=0.7, label=f'Human baseline ({human_mean:.2f})')

    # Styling
    ax.set_ylabel('Z-Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_title('Watermark Strength: Original vs After Paraphrasing\n(Percentage shows retention rate)')
    ax.set_ylim(bottom=-1)

    # Add explanatory note
    fig.text(0.5, 0.01, 'Z-score measures watermark detectability. Higher = stronger watermark signal.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, 'zscore_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'zscore_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: zscore_comparison.png/pdf")


def plot_distribution_boxplots(data: dict, output_dir: str):
    """Figure 2: Box plots showing z-score distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    comparisons = [
        ('lsh', 'lsh_fixed', 'SemStamp (LSH)'),
        ('kmeans', 'kmeans_fixed', 'K-SemStamp (KMeans)')
    ]

    for row, (context_m, fixed_m, title) in enumerate(comparisons):
        # Filter NaN values for box plots
        def filter_nan(arr):
            return arr[~np.isnan(arr)]

        # Original z-scores comparison
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

        # Paraphrased z-scores comparison
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

    # Add explanatory note
    fig.text(0.5, 0.01, 'Box plots show distribution spread. Higher z-scores above threshold (red line) indicate detectable watermarks.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, 'distribution_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'distribution_boxplots.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: distribution_boxplots.png/pdf")


def plot_roc_curves(data: dict, output_dir: str):
    """Figure 3: ROC curves comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    method_groups = [
        (['lsh', 'lsh_fixed'], 'SemStamp (LSH)'),
        (['kmeans', 'kmeans_fixed'], 'K-SemStamp (KMeans)')
    ]

    for ax, (methods, title) in zip(axes, method_groups):
        for m in methods:
            if 'fpr' not in data[m] or 'tpr' not in data[m]:
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

    # Add explanatory note
    fig.text(0.5, 0.01, 'ROC curves show detection accuracy. Higher AUROC = better discrimination between watermarked and human text.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'roc_curves.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: roc_curves.png/pdf")


def plot_degradation_analysis(data: dict, output_dir: str):
    """Figure 4: Horizontal bar chart showing z-score degradation after paraphrasing."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['lsh', 'lsh_fixed', 'kmeans', 'kmeans_fixed']

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

    # Calculate improvements
    lsh_improvement = ((degradations[0] - degradations[1]) / degradations[0]) * 100
    kmeans_improvement = ((degradations[2] - degradations[3]) / degradations[2]) * 100

    # Add value annotations (inside bars for context, outside for fixed)
    max_deg = max(degradations)
    for i, (bar, deg, ret) in enumerate(zip(bars, degradations, retentions)):
        ax.annotate(f'{deg:.2f} ({ret:.0f}% retained)',
                    xy=(max_deg + 0.08, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10)

    # Add improvement summary box (top right, outside bars)
    summary_text = f'Improvement from Fixed mode:\n  LSH: {lsh_improvement:.0f}% less degradation\n  KMeans: {kmeans_improvement:.0f}% less degradation'
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
            fontsize=10, color='darkgreen', fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4, edgecolor='green'))

    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlim(right=max_deg + 0.8)

    # Add explanatory note
    fig.text(0.5, 0.01, 'Degradation = original z-score minus paraphrased z-score. Lower degradation means watermark survives paraphrasing better.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(output_dir, 'degradation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'degradation_analysis.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: degradation_analysis.png/pdf")


def plot_scatter_correlation(data: dict, output_dir: str):
    """Figure 5: 2x2 scatter plots showing correlation between original and paraphrased z-scores."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    methods = ['lsh', 'lsh_fixed', 'kmeans', 'kmeans_fixed']

    for ax, m in zip(axes.flat, methods):
        z_orig = data[m]['z_scores']
        z_para = data[m]['para_z_scores']

        # Filter NaN values
        valid_mask = ~(np.isnan(z_orig) | np.isnan(z_para))
        z_orig_valid = z_orig[valid_mask]
        z_para_valid = z_para[valid_mask]

        correlation = np.corrcoef(z_orig_valid, z_para_valid)[0, 1]

        ax.scatter(z_orig_valid, z_para_valid, alpha=0.6, c=COLORS[m],
                   edgecolors='black', s=50, linewidth=0.5)

        # Identity line
        all_vals = np.concatenate([z_orig_valid, z_para_valid])
        lims = [min(all_vals) - 0.5, max(all_vals) + 0.5]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect retention', linewidth=1.5)

        # Regression line
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

    plt.suptitle('Per-Sample Z-Score Correlation: Original vs Paraphrased', fontsize=14, y=1.02)

    # Add explanatory note
    fig.text(0.5, 0.01, 'Each point is one text sample. Points closer to the diagonal (dashed) retain more watermark strength after paraphrasing.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, 'scatter_correlation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'scatter_correlation.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: scatter_correlation.png/pdf")


def generate_summary_table(data: dict, output_dir: str):
    """Generate CSV summary table with all metrics."""
    rows = []
    for m in METHODS:
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


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from: {args.data_path}")
    print(f"Directory suffix: {args.suffix}")
    print(f"Output directory: {args.output}")
    print("-" * 50)

    # Load data
    data = load_data(args.data_path, args.suffix)

    if len(data) == 0:
        print("Error: No data loaded. Check paths and suffix.")
        return

    print(f"\nLoaded data for methods: {list(data.keys())}")
    print("-" * 50)

    # Generate all visualizations
    print("\nGenerating visualizations...")

    plot_zscore_comparison_bars(data, args.output)
    plot_distribution_boxplots(data, args.output)
    plot_roc_curves(data, args.output)
    plot_degradation_analysis(data, args.output)
    plot_scatter_correlation(data, args.output)
    generate_summary_table(data, args.output)

    print("-" * 50)
    print(f"All visualizations saved to: {args.output}/")


if __name__ == '__main__':
    main()
