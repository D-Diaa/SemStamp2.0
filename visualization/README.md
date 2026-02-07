# Visualization Package

Generates publication-quality plots and summary tables for analyzing watermark robustness, quality, and detectability across methods and paraphrasers.

## Modules

| Module | Description |
|---|---|
| `__main__.py` | CLI dispatcher with subcommands: `robustness`, `robustness_quality`, `watermark_quality` |
| `robustness.py` | Fixed vs context-dependent comparison for a single paraphraser (z-score bars, box plots, ROC curves, degradation analysis, scatter correlation) |
| `robustness_quality.py` | Cross-paraphraser analysis combining detection and quality metrics (Pareto frontiers, heatmaps, quality-adjusted bars) |
| `watermark_quality.py` | Watermark quality vs detectability comparison without paraphrasing |
| `utils.py` | Shared constants (colors, labels, metric definitions), data loading (`load_quality_csv`, `load_detection_results`, `load_z_scores`), and plot helpers (`setup_plot_style`, `save_figure`, `suffix_to_label`) |

## Usage

### Robustness (single paraphraser)

```bash
python -m visualization robustness DATA_PATH --suffix=SUFFIX
```

Compares all four watermark methods (lsh, lsh_fixed, kmeans, kmeans_fixed) for one paraphraser. Generates z-score comparison bars, distribution box plots, ROC curves, degradation analysis, scatter correlation, and a summary CSV.

- `--suffix`: Directory suffix identifying the paraphraser (e.g., `-generated-parrot-bigram=False-threshold=0.0`)
- `--output`: Custom output directory (default: `{data_path}/figures/robustness/{label}/`)

### Robustness vs Quality (cross-paraphraser)

```bash
python -m visualization robustness_quality DATA_PATH --pattern '*-generated-*'
```

Combines detection metrics (AUROC, z-score retention) with quality metrics (MAUVE, BERTScore, perplexity) across all paraphrasers. Generates Pareto frontier plots, annotated heatmaps, quality-adjusted bar charts, and a combined summary CSV.

- `--pattern`: Glob pattern for matching subdirectories (default: `*-generated-*`)
- `--output`: Custom output directory (default: `{data_path}/figures/robustness_quality/`)
- `--force`: Re-generate even if output directory exists

Requires `eval_quality.csv` in each dataset directory (run `python -m quality.batch_eval` first).

### Watermark Quality (no paraphrasing)

```bash
python -m visualization watermark_quality DATA_PATH
```

Compares quality and detectability of watermarked text across methods using `{method}-generated/` directories. Loads `eval_quality.csv` for quality and `results_wm.csv` for detection metrics.

- `--output`: Custom output directory (default: `{data_path}/figures/watermark_quality/`)

## Outputs

All figures are saved as both PNG (300 dpi) and PDF. Output goes to `{data_path}/figures/{analysis_type}/`.

| Subcommand | Outputs |
|---|---|
| `robustness` | `zscore_comparison`, `distribution_boxplots`, `roc_curves`, `degradation_analysis`, `scatter_correlation`, `summary_metrics.csv` |
| `robustness_quality` | `pareto_{group}_{metric}` (6 files), `quality_bars`, `heatmap_{group}` (2 files), `quality_comparison` (per-suffix), `combined_summary.csv` |
| `watermark_quality` | `watermark_quality`, `watermark_quality_summary.csv` |

## Prerequisites

Visualization modules read outputs from the detection and quality packages:

- **Detection**: `z_scores.npy`, `para_z_scores.npy`, `human_z_scores.npy`, `results.csv`, `results_wm.csv`, `fpr.npy`, `tpr.npy`
- **Quality**: `eval_quality.csv` (paraphrased) or `eval_quality_wm.csv` (watermarked)
