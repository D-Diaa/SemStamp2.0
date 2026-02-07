"""
Robustness-vs-Quality visualizations for comparing watermark methods across paraphrasers.

Combines detection metrics (AUROC, z-score retention) with quality metrics (MAUVE, BERTScore, perplexity)
to show true robustness: how hard is watermark removal without degrading quality.

Usage:
    python -m visualization.robustness_quality data/c4-val-250
    python -m visualization.robustness_quality data/c4-val-250 --pattern "*-generated-*" --force
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualization.utils import (
    METHODS, COLORS, LABELS,
    METHOD_GROUPS, GROUP_LABELS, MARKERS, MARKER_SIZES,
    METRIC_LABELS, METRIC_HIGHER_BETTER,
    PARETO_X_METRICS, PARETO_Y_METRICS, HEATMAP_COLUMNS,
    setup_plot_style, suffix_to_label, save_figure, extract_suffixes,
    load_quality_csv, load_detection_results, load_z_scores,
)


# --- Data Loading ---

def load_cross_paraphraser_data(base_path: str, suffixes: list) -> dict:
    """Load detection + quality data for all methods x all suffixes.

    Args:
        base_path: Base data directory (e.g., 'data/c4-val-250').
        suffixes: List of suffix strings from extract_suffixes().

    Returns:
        Nested dict: cross_data[method][suffix] = {metric: value, ...}
    """
    cross_data = {}

    for method in METHODS:
        cross_data[method] = {}

        for suffix in suffixes:
            dir_path = os.path.join(base_path, f"{method}{suffix}")

            if not os.path.isdir(dir_path):
                continue

            entry = {}

            # Load detection results
            detection = load_detection_results(dir_path)
            entry.update(detection)

            # Load z-scores for retention computation
            z_data = load_z_scores(dir_path)
            if 'z_scores' in z_data and 'para_z_scores' in z_data:
                mean_z = np.nanmean(z_data['z_scores'])
                mean_para_z = np.nanmean(z_data['para_z_scores'])
                entry['mean_z'] = mean_z
                entry['mean_para_z'] = mean_para_z
                entry['z_retention'] = (mean_para_z / mean_z) * 100 if mean_z != 0 else 0

            # Load quality metrics
            quality = load_quality_csv(dir_path)
            if quality is not None:
                entry.update(quality)
                entry['has_quality'] = True
            else:
                entry['has_quality'] = False

            # Derived metrics
            if 'gen_ppl' in entry and entry['gen_ppl'] > 0:
                entry['inv_ppl'] = 1.0 / entry['gen_ppl']

            if entry:
                cross_data[method][suffix] = entry

    return cross_data


def get_available_suffixes(cross_data: dict, method_group: str, require_quality: bool = True) -> list:
    """Get suffixes available for both methods in a group, optionally requiring quality data."""
    context_m, fixed_m = METHOD_GROUPS[method_group]
    context_suffixes = set(cross_data.get(context_m, {}).keys())
    fixed_suffixes = set(cross_data.get(fixed_m, {}).keys())
    common = sorted(context_suffixes & fixed_suffixes)

    if require_quality:
        common = [s for s in common
                  if cross_data[context_m][s].get('has_quality', False)
                  and cross_data[fixed_m][s].get('has_quality', False)]

    return common


# --- Pareto Frontier ---

def compute_pareto_frontier(points: list) -> list:
    """Compute 2D Pareto frontier from the attacker's perspective (maximize x, minimize y).

    Best attack = high quality (x) and low detectability (y).
    A point is on the frontier if no other point has both higher x and lower y.

    Args:
        points: List of (x, y, label) tuples.

    Returns:
        Sorted list of (x, y, label) on the Pareto frontier.
    """
    if not points:
        return []

    # Sort by x descending
    sorted_pts = sorted(points, key=lambda p: p[0], reverse=True)
    frontier = [sorted_pts[0]]
    min_y = sorted_pts[0][1]

    for pt in sorted_pts[1:]:
        if pt[1] <= min_y:
            frontier.append(pt)
            min_y = pt[1]

    # Sort frontier by x ascending for line plotting
    return sorted(frontier, key=lambda p: p[0])


def _plot_pareto_single(ax, cross_data: dict, group_name: str,
                        x_metric: str, y_metric: str):
    """Plot a single Pareto frontier subplot for one method group and metric combo."""
    context_m, fixed_m = METHOD_GROUPS[group_name]
    suffixes = get_available_suffixes(cross_data, group_name, require_quality=True)

    if not suffixes:
        ax.text(0.5, 0.5, 'No quality data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    for method in (context_m, fixed_m):
        points = []
        for suffix in suffixes:
            entry = cross_data[method].get(suffix, {})
            x_val = entry.get(x_metric)
            y_val = entry.get(y_metric)
            if x_val is not None and y_val is not None:
                points.append((x_val, y_val, suffix_to_label(suffix)))

        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        labels_list = [p[2] for p in points]

        ax.scatter(xs, ys, c=COLORS[method], marker=MARKERS[method],
                   s=MARKER_SIZES[method], label=LABELS[method],
                   edgecolors='black', linewidth=0.5, zorder=3)

        # Label points
        for x, y, label in zip(xs, ys, labels_list):
            ax.annotate(label, (x, y), textcoords='offset points',
                        xytext=(5, 5), fontsize=8, alpha=0.8)

        # Pareto frontier
        frontier = compute_pareto_frontier(points)
        if len(frontier) > 1:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            linestyle = '--' if 'fixed' in method else '-'
            ax.plot(fx, fy, color=COLORS[method], linestyle=linestyle,
                    alpha=0.5, linewidth=1.5, zorder=2)
        elif len(frontier) == 1:
            # Highlight single-point frontier with a larger halo
            ax.scatter([frontier[0][0]], [frontier[0][1]],
                       c=COLORS[method], marker=MARKERS[method],
                       s=MARKER_SIZES[method] * 3, edgecolors='black',
                       linewidth=2, zorder=4, alpha=0.4)

    ax.set_ylabel(METRIC_LABELS.get(y_metric, y_metric))
    ax.legend(loc='best', fontsize=9)


def plot_pareto_single_file(cross_data: dict, output_dir: str,
                            group_name: str, x_metric: str):
    """Generate one Pareto frontier file with 2 stacked subplots (AUROC and TPR@1%FPR).

    Both subplots share the same x-axis (quality metric).
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for ax, y_metric in zip(axes, PARETO_Y_METRICS):
        _plot_pareto_single(ax, cross_data, group_name, x_metric, y_metric)

    # Only set x-label on the bottom subplot
    axes[-1].set_xlabel(METRIC_LABELS.get(x_metric, x_metric))

    fig.suptitle(f'{GROUP_LABELS[group_name]}: Robustness vs {METRIC_LABELS.get(x_metric, x_metric)}',
                 fontsize=14)
    fig.text(0.5, -0.01,
             'Points in the bottom-right are the strongest attacks (high quality, low detectability). '
             'Frontier lines show the best attacker trade-off.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_figure(output_dir, f'pareto_{group_name}_{x_metric}')


def plot_pareto_all(cross_data: dict, output_dir: str):
    """Generate all Pareto frontier files (one per method group x quality metric)."""
    for group_name in METHOD_GROUPS:
        for x_metric in PARETO_X_METRICS:
            plot_pareto_single_file(cross_data, output_dir, group_name, x_metric)


# --- Quality-Adjusted Bars ---

def plot_quality_adjusted_bars_grid(cross_data: dict, output_dir: str,
                                    detection_metric: str = 'auroc',
                                    quality_metric: str = 'mauve'):
    """Grouped bar chart with detection bars and quality markers on secondary axis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (group_name, (context_m, fixed_m)) in zip(axes, METHOD_GROUPS.items()):
        suffixes = get_available_suffixes(cross_data, group_name, require_quality=True)

        if not suffixes:
            ax.text(0.5, 0.5, 'No quality data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(GROUP_LABELS[group_name])
            continue

        labels = [suffix_to_label(s) for s in suffixes]
        x = np.arange(len(suffixes))
        width = 0.35

        # Detection bars
        context_det = [cross_data[context_m][s].get(detection_metric, 0) for s in suffixes]
        fixed_det = [cross_data[fixed_m][s].get(detection_metric, 0) for s in suffixes]

        ax.bar(x - width / 2, context_det, width, label=f'{LABELS[context_m]}',
               color=COLORS[context_m], edgecolor='black', alpha=0.85)
        ax.bar(x + width / 2, fixed_det, width, label=f'{LABELS[fixed_m]}',
               color=COLORS[fixed_m], edgecolor='black', alpha=0.85)

        ax.set_ylabel(METRIC_LABELS.get(detection_metric, detection_metric))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_title(GROUP_LABELS[group_name])

        # Quality on secondary axis
        ax2 = ax.twinx()
        context_qual = [cross_data[context_m][s].get(quality_metric, 0) for s in suffixes]
        fixed_qual = [cross_data[fixed_m][s].get(quality_metric, 0) for s in suffixes]

        ax2.plot(x - width / 2, context_qual, marker=MARKERS[context_m],
                 color=COLORS[context_m], linestyle='--', linewidth=1.5, markersize=8,
                 label=f'{LABELS[context_m]} ({METRIC_LABELS.get(quality_metric, quality_metric)})')
        ax2.plot(x + width / 2, fixed_qual, marker=MARKERS[fixed_m],
                 color=COLORS[fixed_m], linestyle='--', linewidth=1.5, markersize=10,
                 label=f'{LABELS[fixed_m]} ({METRIC_LABELS.get(quality_metric, quality_metric)})')

        ax2.set_ylabel(METRIC_LABELS.get(quality_metric, quality_metric), color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    fig.suptitle(f'Robustness ({METRIC_LABELS.get(detection_metric, detection_metric)}) '
                 f'vs Quality ({METRIC_LABELS.get(quality_metric, quality_metric)}) by Paraphraser',
                 fontsize=14)
    fig.text(0.5, -0.01,
             f'Bars = {METRIC_LABELS.get(detection_metric, detection_metric)} (left axis). '
             f'Markers = {METRIC_LABELS.get(quality_metric, quality_metric)} (right axis).',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_figure(output_dir, 'quality_bars')


# --- Heatmap Table ---

def plot_heatmap_table(cross_data: dict, output_dir: str, method_group: str):
    """Annotated heatmap with rows=paraphrasers (context + fixed) and columns=metrics."""
    context_m, fixed_m = METHOD_GROUPS[method_group]
    suffixes = get_available_suffixes(cross_data, method_group, require_quality=True)

    if not suffixes:
        print(f"Skipping heatmap for {method_group}: no quality data available")
        return

    columns = [c for c in HEATMAP_COLUMNS
                if any(c in cross_data[context_m].get(s, {}) for s in suffixes)]

    # Build data matrix: rows = (suffix, method) pairs
    row_labels = []
    matrix = []

    for suffix in suffixes:
        label = suffix_to_label(suffix)
        for method in (context_m, fixed_m):
            variant = 'Context' if method == context_m else 'Fixed'
            row_labels.append(f'{label}\n({variant})')
            row_vals = []
            for col in columns:
                val = cross_data[method].get(suffix, {}).get(col, np.nan)
                row_vals.append(val)
            matrix.append(row_vals)

    matrix = np.array(matrix, dtype=float)

    # Normalize per column for coloring
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1  # avoid div by zero

    norm_matrix = (matrix - col_min) / col_range

    # Invert columns where lower is better
    for j, col in enumerate(columns):
        if not METRIC_HIGHER_BETTER.get(col, True):
            norm_matrix[:, j] = 1.0 - norm_matrix[:, j]

    # Plot
    fig_height = max(4, 0.6 * len(row_labels) + 2)
    fig_width = max(8, 1.2 * len(columns) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(norm_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Annotate cells with actual values
    for i in range(len(row_labels)):
        for j in range(len(columns)):
            val = matrix[i, j]
            if np.isnan(val):
                text = '—'
            elif columns[j] == 'gen_ppl':
                text = f'{val:.1f}'
            elif columns[j] == 'z_retention':
                text = f'{val:.1f}%'
            else:
                text = f'{val:.3f}'

            # Choose text color based on background brightness
            bg_val = norm_matrix[i, j]
            text_color = 'white' if bg_val < 0.3 or bg_val > 0.85 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color=text_color, fontweight='bold')

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([METRIC_LABELS.get(c, c) for c in columns], rotation=30, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Add horizontal separators between paraphraser groups
    for i in range(2, len(row_labels), 2):
        ax.axhline(y=i - 0.5, color='white', linewidth=2)

    ax.set_title(f'{GROUP_LABELS[method_group]}: Robustness & Quality Metrics by Paraphraser',
                 fontsize=13, pad=15)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Normalized Score (green = better)', fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, f'heatmap_{method_group}')


def plot_heatmap_tables(cross_data: dict, output_dir: str):
    """Generate heatmaps for both method groups."""
    for group_name in METHOD_GROUPS:
        plot_heatmap_table(cross_data, output_dir, group_name)


# --- Cross-Method Per-Suffix Tradeoff Scatter ---

def plot_cross_method_tradeoff_scatter(cross_data: dict, suffix: str, output_dir: str):
    """Per-suffix tradeoff scatter: quality (x) vs detection (y) for all methods.

    Generates one figure per quality metric, each with subplots for each detection metric.
    Similar to Pareto plots but scoped to a single paraphraser with all 4 methods shown.
    """
    available_methods = [m for m in METHODS if suffix in cross_data.get(m, {})]
    if not available_methods:
        return

    has_any_quality = any(cross_data[m][suffix].get('has_quality', False) for m in available_methods)
    if not has_any_quality:
        return

    paraphraser_label = suffix_to_label(suffix)

    for x_metric in PARETO_X_METRICS:
        # Check if any method has this quality metric
        has_metric = any(
            cross_data[m][suffix].get(x_metric) is not None for m in available_methods
        )
        if not has_metric:
            continue

        fig, axes = plt.subplots(1, len(PARETO_Y_METRICS), figsize=(7 * len(PARETO_Y_METRICS), 6))
        if len(PARETO_Y_METRICS) == 1:
            axes = [axes]

        for ax, y_metric in zip(axes, PARETO_Y_METRICS):
            for method in available_methods:
                entry = cross_data[method][suffix]
                x_val = entry.get(x_metric)
                y_val = entry.get(y_metric)
                if x_val is None or y_val is None:
                    continue

                ax.scatter(x_val, y_val, c=COLORS[method], marker=MARKERS[method],
                           s=MARKER_SIZES[method], label=LABELS[method],
                           edgecolors='black', linewidth=0.8, zorder=3)

                ax.annotate(LABELS[method], (x_val, y_val), textcoords='offset points',
                            xytext=(6, 6), fontsize=8, alpha=0.8)

            ax.set_xlabel(METRIC_LABELS.get(x_metric, x_metric))
            ax.set_ylabel(METRIC_LABELS.get(y_metric, y_metric))
            ax.legend(loc='best', fontsize=9)

        fig.suptitle(f'{paraphraser_label}: Detection vs {METRIC_LABELS.get(x_metric, x_metric)}',
                     fontsize=14)
        fig.text(0.5, -0.01,
                 f'Each point is one watermark method. '
                 f'Lower detection (y) with higher {METRIC_LABELS.get(x_metric, x_metric)} (x) '
                 f'means the watermark is easier to remove without degrading quality.',
                 ha='center', fontsize=9, style='italic', color='gray')
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        save_figure(output_dir, f'tradeoff_{x_metric}')


# --- Summary CSV ---

def generate_combined_summary(cross_data: dict, output_dir: str):
    """Generate a combined CSV with all methods x suffixes x metrics."""
    rows = []
    all_suffixes = set()
    for method in METHODS:
        all_suffixes.update(cross_data.get(method, {}).keys())

    for suffix in sorted(all_suffixes):
        for method in METHODS:
            entry = cross_data.get(method, {}).get(suffix)
            if entry is None:
                continue

            row = {
                'method': method,
                'method_label': LABELS[method],
                'paraphraser': suffix_to_label(suffix),
                'suffix': suffix,
            }
            for key in ['auroc', 'fpr1', 'fpr5', 'z_retention', 'mean_z', 'mean_para_z',
                         'bert_score', 'gen_ppl', 'bi_entro', 'tri_entro',
                         'rep_2', 'rep_3', 'rep_4', 'sem_ent', 'mauve',
                         'bert_P', 'bert_R', 'bert_F1']:
                if key in entry:
                    row[key] = entry[key]

            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'combined_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: combined_summary.csv ({len(rows)} entries)")


# --- Orchestrator ---

def run_robustness_quality_visualization(data_path: str, suffixes: list,
                                          output_dir: str = None):
    """Main orchestrator for robustness-vs-quality visualizations.

    Args:
        data_path: Base data directory.
        suffixes: List of suffix strings.
        output_dir: Output directory (default: {data_path}/Figures_RQ).
    """
    if output_dir is None:
        output_dir = os.path.join(data_path, 'figures', 'robustness_quality')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading cross-paraphraser data from: {data_path}")
    print(f"Suffixes: {len(suffixes)}")
    print("-" * 50)

    cross_data = load_cross_paraphraser_data(data_path, suffixes)

    # Report data availability
    total_entries = sum(len(v) for v in cross_data.values())
    quality_entries = sum(
        1 for m in cross_data.values() for e in m.values() if e.get('has_quality', False)
    )
    print(f"Loaded {total_entries} method-paraphraser combinations ({quality_entries} with quality data)")

    if quality_entries == 0:
        print("Warning: No quality data found (eval_quality.csv). "
              "Run `python -m quality.batch_eval` first.")
        print("Generating summary CSV only (no quality-dependent plots).")
        generate_combined_summary(cross_data, output_dir)
        return

    # Report missing quality data
    for method in METHODS:
        for suffix in suffixes:
            entry = cross_data.get(method, {}).get(suffix)
            if entry and not entry.get('has_quality', False):
                print(f"  Missing quality: {method}{suffix}")

    print("-" * 50)
    print("Generating visualizations...")

    # Cross-paraphraser plots
    plot_pareto_all(cross_data, output_dir)
    plot_quality_adjusted_bars_grid(cross_data, output_dir)
    plot_heatmap_tables(cross_data, output_dir)

    # Per-suffix cross-method plots
    for suffix in suffixes:
        label = suffix_to_label(suffix)
        suffix_output = os.path.join(data_path, 'figures', 'robustness', label)
        if os.path.isdir(suffix_output):
            plot_cross_method_tradeoff_scatter(cross_data, suffix, suffix_output)

    # Summary
    generate_combined_summary(cross_data, output_dir)

    print("-" * 50)
    print(f"All robustness-quality visualizations saved to: {output_dir}/")


def main():
    setup_plot_style()

    parser = argparse.ArgumentParser(
        description='Generate robustness-vs-quality visualizations across all paraphrasers.'
    )
    parser.add_argument('data_path', type=str,
                        help='Base path to data directories (e.g., data/c4-val-250)')
    parser.add_argument('--pattern', type=str, default='*-generated-*',
                        help='Glob pattern for matching dataset subdirectories (default: *-generated-*)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--force', action='store_true',
                        help='Re-generate even if output directory already exists')
    args = parser.parse_args()

    suffixes = extract_suffixes(args.data_path, args.pattern)

    if not suffixes:
        print(f"No matching directories found in {args.data_path} with pattern '{args.pattern}'")
        exit(1)

    print(f"Found {len(suffixes)} unique paraphraser suffix(es):")
    for s in suffixes:
        print(f"  {s}")
    print()

    output_dir = args.output or os.path.join(args.data_path, 'figures', 'robustness_quality')

    if not args.force and os.path.exists(output_dir):
        print(f"Output directory already exists: {output_dir}")
        print("Use --force to re-generate.")
        exit(0)

    run_robustness_quality_visualization(args.data_path, suffixes, output_dir)


if __name__ == '__main__':
    main()
