# Detection Package

Detects SemStamp watermarks in generated text by checking whether sentence embeddings fall into expected partitions, then computing a z-score for statistical significance.

## Usage

```bash
python -m detection DATASET_PATH --detection_mode MODE [options]
```

The dataset must be a HuggingFace dataset (saved with `save_to_disk`) containing a `text` column. If a `para_text` column is also present, paraphrased texts are evaluated alongside. If `para_text` is absent, the paraphrased evaluation is skipped entirely and only watermarked and human baseline results are produced.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `dataset_path` (positional) | — | Path to HF dataset with `text` (and optionally `para_text`) columns |
| `--detection_mode` | — | `lsh`, `kmeans`, `lsh_fixed`, or `kmeans_fixed` |
| `--sp_dim` | `3` | Partition dimension (3 for LSH, 8 for k-means) |
| `--embedder` | `AbeHou/SemStamp-c4-sbert` | Sentence embedder model |
| `--lmbd` | `0.25` | Ratio of valid partitions (must match generation) |
| `--human_text` | `data/c4-human` | HF dataset of human-written text for baseline |
| `--cc_path` | `None` | Path to cluster centers `.pt` file (required for kmeans modes) |
| `--secret_message` | `"The magic words are squeamish ossifrage."` | Secret for fixed modes |

### Examples

```bash
# LSH detection
python -m detection data/c4-val-1000/lsh-generated \
    --detection_mode lsh --sp_dim 3

# k-means detection
python -m detection data/c4-val-1000/kmeans-generated \
    --detection_mode kmeans --sp_dim 8 \
    --cc_path data/c4-train/cc.pt

# Fixed mode detection
python -m detection data/c4-val-1000/lsh_fixed-generated \
    --detection_mode lsh_fixed --sp_dim 3
```

## How It Works

1. Each text is split into sentences using NLTK `sent_tokenize`.
2. For each consecutive sentence pair, the detector checks whether the current sentence's embedding falls into a valid partition seeded by:
   - The previous sentence's hash (context-dependent modes: `lsh`, `kmeans`)
   - A fixed secret message hash (fixed modes: `lsh_fixed`, `kmeans_fixed`)
3. A z-score is computed from the count of matching sentences vs. the expected rate under the null hypothesis (no watermark).
4. Z-scores for watermarked, paraphrased (if available), and human texts are compared to produce ROC curves and detection metrics.

## Outputs

All outputs are saved to `dataset_path/`.

**Always produced:**

| File | Description |
|---|---|
| `z_scores.npy` | Z-scores for watermarked texts |
| `human_z_scores.npy` | Z-scores for human baseline texts |
| `results_wm.csv` | Watermarked detection metrics (AUROC, FPR@1%, FPR@5%, mean z-scores) |
| `roc_curve_wm.png` | ROC curve for watermarked vs. human |
| `fpr_wm.npy`, `tpr_wm.npy` | FPR/TPR arrays for watermarked ROC |

**Only produced when `para_text` column exists:**

| File | Description |
|---|---|
| `para_z_scores.npy` | Z-scores for paraphrased texts |
| `results.csv` | Paraphrased detection metrics (AUROC, FPR@1%, FPR@5%, BERTScore) |
| `roc_curve.png` | ROC curve for paraphrased vs. human |
| `fpr.npy`, `tpr.npy` | FPR/TPR arrays for paraphrased ROC |

## Module Structure

- [\_\_main\_\_.py](__main__.py) — CLI entry point, orchestrates loading data, running detection, and saving results.
- [utils.py](utils.py) — Core detection logic:
  - `detect_lsh()` / `detect_kmeans()` — Compute z-scores for a list of sentences.
  - `compute_zscore()` — Z-test statistic from watermark count.
  - `evaluate_z_scores()` — AUROC and FPR at 1%/5% thresholds.
  - `get_roc_metrics_from_zscores()` — ROC curve generation and plotting.
  - `run_bert_score()` — BERTScore between original and paraphrased sentences.
  - `flatten_gens_and_paras()` — Aligns generated and paraphrased sentence lists.

## Notes

- If a GPU was used during generation, detection must also run on GPU (for LSH projection consistency). It does not need to be the same GPU.
- Detection parameters (`sp_dim`, `lmbd`, `embedder`, `secret_message`) must match those used during generation.
