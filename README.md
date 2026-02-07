# SemStamp 2.0

An extended reimplementation of [SemStamp](https://arxiv.org/abs/2310.03991) (NAACL'24) and [k-SemStamp](https://arxiv.org/abs/2402.11399) (ACL'24) — semantic watermarking for LLM-generated text.

This fork restructures the original codebase into a modular pipeline, adds fixed-context watermark modes, integrates custom paraphrasers (including fine-tuned LoRA attackers), and provides end-to-end experiment orchestration with quality evaluation and visualization.

**Author:** [Abdulrahman Diaa](https://github.com/D-Diaa)

## What's New in This Fork

- **Fixed-context modes** (`lsh_fixed`, `kmeans_fixed`) — watermark derived from a secret message instead of previous-sentence context, making it significantly more robust to paraphrasing attacks
- **Modular pipeline** — each stage (sampling, detection, paraphrasing, quality, visualization) is a self-contained Python package runnable via `python -m <package>`
- **Custom paraphrasers** — plug in any HuggingFace model (including PEFT/LoRA) as a paraphrasing attacker, with multiple prompt strategies (`standard`, `shuffle`, `combine`)
- **Quality evaluation suite** — perplexity, n-gram entropy/repetition, semantic entropy, MAUVE, and BERTScore in one pass
- **Visualization** — robustness comparisons, Pareto frontiers, heatmaps, and watermark quality analysis
- **Multi-GPU pipeline script** — `run_experiments.sh` distributes modes across GPUs and runs the full pipeline automatically
- **Batch evaluation** — evaluate quality across all generated/paraphrased variants in one command

## How SemStamp Works

**Generation:** For each sentence, hash the previous sentence's embedding (or a secret message in fixed mode) to derive a valid partition mask over the embedding space. Rejection-sample candidate sentences until one lands in a valid partition.

**Detection:** Tokenize into sentences, check what fraction land in valid partitions, and compute a z-score: `z = (hits - λn) / sqrt(nλ(1-λ))`. Compare watermarked vs. human text via AUROC and FPR thresholds.

Two partitioning strategies:

- **LSH** (`lsh` / `lsh_fixed`) — Locality-Sensitive Hashing with random hyperplanes (`sp_dim=3`)
- **K-Means** (`kmeans` / `kmeans_fixed`) — clustering-based partitioning (`sp_dim=8`)

## Installation

```bash
git clone --recurse-submodules https://github.com/D-Diaa/SemStamp2.0.git
cd SemStamp2.0
pip install -r requirements.txt
python scripts/install_punkt.py  # MANDATORY — NLTK sentence tokenizer
```

Requires a CUDA-capable GPU. Check availability with `gpustat` and set `CUDA_VISIBLE_DEVICES` accordingly.

## Quick Start

### 1. Prepare Data

```bash
python scripts/load_c4.py --k 20000          # download C4 dataset
python scripts/build_subset.py data/c4-val --n 1000  # create evaluation subset
```

### 2. Generate Watermarked Text

```bash
python -m sampling data/c4-val-1000 \
    --sp_mode lsh --sp_dim 3 --delta 0.02 \
    --model AbeHou/opt-1.3b-semstamp \
    --embedder AbeHou/SemStamp-c4-sbert
```

For k-means mode, first compute cluster centers, then generate:

```bash
python -m sampling.kmeans_utils data/c4-train AbeHou/SemStamp-c4-sbert 8
python -m sampling data/c4-val-1000 \
    --sp_mode kmeans --sp_dim 8 --delta 0.02 \
    --model AbeHou/opt-1.3b-semstamp \
    --embedder AbeHou/SemStamp-c4-sbert \
    --cc_path data/c4-train/cc.pt
```

### 3. Detect Watermark

```bash
python -m detection data/c4-val-1000/lsh-generated \
    --detection_mode lsh --sp_dim 3 \
    --embedder AbeHou/SemStamp-c4-sbert \
    --human_text data/c4-human
```

### 4. Attack with Paraphrasing

```bash
python -m paraphrasing data/c4-val-1000/lsh-generated \
    --paraphraser custom \
    --custom_model DDiaa/WM-Removal-Unigram-Qwen2.5-3B \
    --custom_prompt combine --bsz 32
```

### 5. Evaluate Quality

```bash
python -m quality data/c4-val-1000/lsh-generated 205 \
    --reference data/c4-val-1000 \
    --corpus data/c4-train
```

### 6. Visualize Results

```bash
python -m visualization robustness data/c4-val-1000 --suffix=SUFFIX
python -m visualization robustness_quality data/c4-val-1000 --pattern '*-generated-*'
python -m visualization watermark_quality data/c4-val-1000
```

## Full Pipeline

Run everything end-to-end across multiple GPUs:

```bash
bash scripts/run_experiments.sh \
    --gpus 4,5,6,7 \
    --data data/c4-val-1000 \
    --modes lsh,lsh_fixed,kmeans,kmeans_fixed \
    --paraphrasers custom \
    --custom-models "DDiaa/WM-Removal-Unigram-Qwen2.5-3B" \
    --custom-prompts standard
```

This distributes one watermark mode per GPU and runs: sampling → detection → quality → paraphrasing → detection → quality → visualization. Logs go to `{data}/logs/{mode}.log`.

## Project Structure

```
sampling/           Watermarked text generation via rejection sampling
detection/          Z-score based watermark detection
paraphrasing/       Paraphrasing attack implementations
quality/            Text quality evaluation metrics
visualization/      Result analysis and plotting
training/           Sentence embedder fine-tuning (contrastive learning)
scripts/            Utility scripts and experiment runner
centroids/          Pre-computed k-means cluster centers
semstamp-data/      Original paper data (git submodule)
```

Each package has its own README with detailed documentation. See [sampling/](sampling/), [detection/](detection/), [quality/](quality/), and [visualization/](visualization/).

## Key Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--sp_mode` | — | Watermark mode: `lsh`, `lsh_fixed`, `kmeans`, `kmeans_fixed` |
| `--sp_dim` | — | Partition dimensions (3 for LSH, 8 for k-means) |
| `--delta` | `0` | Rejection threshold (higher = more robust, slower) |
| `--lmbd` | `0.25` | Fraction of valid partitions |
| `--batch_size` | `16` | Candidates per rejection step |
| `--max_new_tokens` | `205` | Max tokens to generate |
| `--secret_message` | `"The magic words are squeamish ossifrage."` | Secret for fixed modes |
| `--cc_path` | — | Path to cluster centers (k-means modes) |
| `--embedder` | — | Sentence embedder model |
| `--paraphraser` | — | Paraphraser type: `parrot`, `t5`, `openai`, `custom` |

## Pre-trained Resources

| Resource | Description |
| --- | --- |
| `AbeHou/opt-1.3b-semstamp` | OPT-1.3B fine-tuned for shorter sentences |
| `AbeHou/SemStamp-c4-sbert` | Sentence embedder fine-tuned on C4 |
| `AbeHou/SemStamp-booksum-sbert` | Sentence embedder fine-tuned on BookSum |
| `DDiaa/WM-Removal-Unigram-Qwen2.5-3B` | Custom paraphraser for watermark removal |

Any HuggingFace causal LM (OPT, LLaMA, GPT-2, Mistral, Qwen, etc.) works for generation.

## Important Notes

- **GPU consistency:** Detection must run on the same GPU type as generation (LSH random seeds are device-dependent)
- **NLTK punkt:** Run `scripts/install_punkt.py` before any generation or detection
- **K-means modes:** Require pre-computed cluster centers via `--cc_path`

## Reproducing Paper Results

The original paper data is included as a git submodule in `semstamp-data/`. To reproduce detection results:

```bash
# SemStamp (LSH)
python -m detection semstamp-data/c4-semstamp-pegasus-parrot/semstamp-pegasus-bigram=False \
    --detection_mode lsh --sp_dim 3 \
    --embedder AbeHou/SemStamp-c4-sbert \
    --human_text semstamp-data/original-c4-texts

# k-SemStamp (K-Means)
python -m detection semstamp-data/c4-ksemstamp-pegasus/bigram=False \
    --detection_mode kmeans --sp_dim 8 \
    --embedder AbeHou/SemStamp-c4-sbert \
    --cc_path centroids/c4-cluster_8_centers.pt \
    --human_text semstamp-data/original-c4-texts
```

## Citations

```bibtex
@inproceedings{hou-etal-2023-semstamp,
    title = "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation",
    author = "Hou, Abe Bohan and Zhang, Jingyu and He, Tianxing and
              Chuang, Yung-Sung and Wang, Hongwei and Shen, Lingfeng and
              Van Durme, Benjamin and Khashabi, Daniel and Tsvetkov, Yulia",
    booktitle = "Annual Conference of the North American Chapter of the Association for Computational Linguistics",
    year = "2023",
    url = "https://arxiv.org/abs/2310.03991",
}

@inproceedings{hou-etal-2024-k,
    title = "k-{S}em{S}tamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text",
    author = "Hou, Abe and Zhang, Jingyu and Wang, Yichen and Khashabi, Daniel and He, Tianxing",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.98",
    doi = "10.18653/v1/2024.findings-acl.98",
    pages = "1706--1715",
}
```
