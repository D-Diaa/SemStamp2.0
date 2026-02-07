"""
Shared utilities for the visualization package.

Contains constants, data loading functions, and helpers used across visualization modules.
"""

import os
import re
import csv
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Method Constants ---

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

METHOD_GROUPS = {
    'lsh': ('lsh', 'lsh_fixed'),
    'kmeans': ('kmeans', 'kmeans_fixed'),
}

GROUP_LABELS = {
    'lsh': 'SemStamp (LSH)',
    'kmeans': 'K-SemStamp (KMeans)',
}

MARKERS = {
    'lsh': 'o', 'lsh_fixed': '*',
    'kmeans': 'o', 'kmeans_fixed': '*',
}

MARKER_SIZES = {
    'lsh': 80, 'lsh_fixed': 150,
    'kmeans': 80, 'kmeans_fixed': 150,
}


# --- Metric Constants ---

METRIC_LABELS = {
    'auroc': 'AUROC',
    'fpr1': 'TPR@1%FPR',
    'fpr5': 'TPR@5%FPR',
    'z_retention': 'Z-Score Retention (%)',
    'mauve': 'MAUVE',
    'bert_F1': 'BERTScore F1',
    'gen_ppl': 'Perplexity',
    'inv_ppl': '1/Perplexity',
    'sem_ent': 'Semantic Entropy',
    'bi_entro': 'Bigram Entropy',
    'tri_entro': 'Trigram Entropy',
    'rep_2': 'Rep-2',
    'rep_3': 'Rep-3',
    'rep_4': 'Rep-4',
    'bert_score': 'Para BERTScore',
    'bert_P': 'BERTScore P',
    'bert_R': 'BERTScore R',
    'mean_z': 'Mean Z-Score',
    'human_mean_z': 'Human Mean Z-Score',
}

METRIC_HIGHER_BETTER = {
    'auroc': True, 'fpr1': True, 'fpr5': True, 'z_retention': True,
    'mauve': True, 'bert_F1': True, 'gen_ppl': False, 'inv_ppl': True,
    'sem_ent': True, 'bi_entro': True, 'tri_entro': True,
    'rep_2': False, 'rep_3': False, 'rep_4': False,
    'bert_score': True, 'bert_P': True, 'bert_R': True,
    'mean_z': True, 'human_mean_z': True,
}

PARETO_X_METRICS = ['mauve', 'bert_F1', 'inv_ppl']
PARETO_Y_METRICS = ['auroc', 'fpr5']

HEATMAP_COLUMNS = ['auroc', 'fpr1', 'z_retention', 'mauve', 'bert_F1', 'gen_ppl', 'sem_ent']


# --- Plot Style ---

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
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


# --- Helper Functions ---

def filter_nan(arr):
    """Filter NaN values from a numpy array."""
    return arr[~np.isnan(arr)]


def save_figure(output_dir: str, filename: str):
    """Save current figure as PNG and PDF, then close."""
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}.png/pdf")


def suffix_to_label(suffix: str) -> str:
    """Convert a directory suffix to a readable paraphraser label.

    Examples:
        '-generated-parrot-bigram=False-threshold=0.0' -> 'Parrot'
        '-generated-parrot-bigram=True-threshold=0.7'  -> 'Parrot-Bi'
        '-generated-custom-Qwen2.5-3B-Instruct-standard' -> 'Qwen-std'
        '-generated-custom-WM-Removal-Unigram-Qwen2.5-3B-combine' -> 'WMR-cmb'
        '-generated-t5-bigram=False-threshold=0.0' -> 'T5'
    """
    # Strip '-generated-' prefix
    name = re.sub(r'^-generated-', '', suffix)

    # Prompt style abbreviations
    prompt_abbrev = {'standard': 'std', 'shuffle': 'shf', 'combine': 'cmb'}

    # Handle custom model paraphrasers: custom-{model}-{prompt_style}
    custom_match = re.match(r'custom-(.+)-(standard|shuffle|combine)$', name)
    if custom_match:
        model_name = custom_match.group(1)
        prompt_style = custom_match.group(2)
        ps = prompt_abbrev.get(prompt_style, prompt_style)
        # Shorten known model names
        if 'WM-Removal' in model_name:
            base = re.sub(r'^WM-Removal-\w+-', '', model_name)
            base_short = re.sub(r'[\d.]+.*', '', base.split('-')[0]).strip() or base.split('-')[0]
            return f'WMR-{base_short}-{ps}'
        short = model_name.split('-')[0]
        short = re.sub(r'[\d.]+', '', short).strip()
        if not short:
            short = model_name.split('-')[0]
        return f'{short}-{ps}'

    # Handle bigram variants
    bigram_match = re.match(r'(\w+)-bigram=(True|False)-threshold=([\d.]+)', name)
    if bigram_match:
        base = bigram_match.group(1).capitalize()
        is_bigram = bigram_match.group(2) == 'True'
        if is_bigram:
            return f'{base}-Bi'
        return base

    return name.capitalize()


def extract_suffixes(data_path: str, pattern: str) -> list:
    """Find all unique suffixes from directories matching the pattern.

    Given directories like:
        lsh-generated-parrot-bigram=False-threshold=0.0
        kmeans_fixed-generated-custom-model-standard

    Extracts suffixes like:
        -generated-parrot-bigram=False-threshold=0.0
        -generated-custom-model-standard
    """
    dirs = sorted(glob.glob(os.path.join(data_path, pattern)))
    dirs = [d for d in dirs if os.path.isdir(d) and "Figures" not in d]

    method_prefixes = sorted(METHODS, key=len, reverse=True)
    suffixes = set()

    for d in dirs:
        name = os.path.basename(d)
        for method in method_prefixes:
            if name.startswith(method):
                suffix = name[len(method):]
                suffixes.add(suffix)
                break

    return sorted(suffixes)


# --- Data Loading ---

def load_quality_csv(dir_path: str, filename: str = 'eval_quality.csv') -> dict:
    """Load a quality evaluation CSV from a dataset directory.

    Args:
        dir_path: Directory containing the CSV file.
        filename: CSV filename (e.g., 'eval_quality.csv' or 'eval_quality_wm.csv').

    Returns:
        Dict with quality metrics, or None if file not found.
    """
    csv_path = os.path.join(dir_path, filename)
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return {k: float(v) for k, v in row.items()}


def load_detection_results(dir_path: str, filename: str = 'results.csv') -> dict:
    """Load detection results from a tab-separated CSV.

    Args:
        dir_path: Directory containing the CSV file.
        filename: CSV filename (e.g., 'results.csv' or 'results_wm.csv').

    Returns:
        Dict with numeric columns from the CSV, or empty dict if not found.
    """
    results_path = os.path.join(dir_path, filename)
    if not os.path.exists(results_path):
        return {}

    results_df = pd.read_csv(results_path)
    result = {}
    for col in results_df.columns:
        result[col] = float(results_df[col].values[0])
    return result


def load_z_scores(dir_path: str) -> dict:
    """Load z-score and ROC numpy arrays from a directory.

    Returns:
        Dict with keys like 'z_scores', 'para_z_scores', 'human_z_scores',
        'fpr', 'tpr' — only for files that exist.
    """
    data = {}
    npy_files = ['z_scores.npy', 'human_z_scores.npy', 'para_z_scores.npy',
                 'fpr.npy', 'tpr.npy']

    for npy_file in npy_files:
        file_path = os.path.join(dir_path, npy_file)
        if os.path.exists(file_path):
            data[npy_file.replace('.npy', '')] = np.load(file_path)

    return data
