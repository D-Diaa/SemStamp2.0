# Sampling Package

Generates watermarked text using batched sentence-wise rejection sampling. Candidate sentences are generated in parallel and accepted only if their embeddings fall into pseudo-randomly selected valid partitions of the embedding space.

## Usage

```bash
python -m sampling DATA_PATH --sp_mode MODE [options]
```

The dataset must be a HuggingFace dataset (saved with `save_to_disk`) containing a `text` column. Prompts are extracted from the first sentence of each text entry.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `data` (positional) | — | Path to HF dataset with `text` column |
| `--model` | `meta-llama/Llama-3.1-8B` | HuggingFace causal LM for generation |
| `--embedder` | `AbeHou/SemStamp-c4-sbert` | Sentence embedder model |
| `--sp_mode` | `None` | `lsh`, `kmeans`, `lsh_fixed`, or `kmeans_fixed` |
| `--sp_dim` | `8` | Number of partitions (use 3 for LSH, 8 for k-means) |
| `--delta` | `0` | Margin size for boundary rejection |
| `--lmbd` | `0.25` | Ratio of valid partitions (acceptance rate) |
| `--batch_size` | `16` | Candidate sentences per rejection sampling batch |
| `--max_batches` | `8` | Max batches before fallback accept |
| `--len_prompt` / `-l` | `32` | Max prompt length in words |
| `--max_new_tokens` | `205` | Max tokens to generate |
| `--min_new_tokens` | `195` | Min tokens to generate |
| `--rep_p` | `1.05` | Repetition penalty |
| `--secret_message` | `"The magic words are squeamish ossifrage."` | Secret for fixed modes |
| `--cc_path` | `None` | Pre-computed k-means cluster centers (required for kmeans modes) |

### Examples

```bash
# LSH-based watermarking
python -m sampling data/c4-val-1000 --sp_mode lsh --sp_dim 3 --delta 0.02

# k-means-based watermarking
python -m sampling data/c4-val-1000 --sp_mode kmeans --sp_dim 8 --delta 0.02 \
    --cc_path data/c4-train/cc.pt

# Fixed mode (secret message based)
python -m sampling data/c4-val-1000 --sp_mode lsh_fixed --sp_dim 3 --delta 0.02
```

## How It Works

1. A prompt is extracted from each input text (first sentence, up to `len_prompt` words).
2. For each sentence to generate, `batch_size` candidates are sampled in parallel from the LM.
3. Each candidate is truncated to exactly one sentence using NLTK `sent_tokenize`.
4. The acceptance criterion checks:
   - **Margin rejection**: the candidate embedding is sufficiently far from partition boundaries (controlled by `delta`).
   - **Partition membership**: the candidate's embedding hash/cluster falls into a valid partition seeded by the previous sentence (context modes) or a fixed secret message (fixed modes).
5. The first accepted candidate is appended to the output. If no candidate passes after `max_batches` batches, a fallback is used.
6. Generation continues sentence-by-sentence until `max_new_tokens` is reached.

## Output

Watermarked texts are saved as a HuggingFace dataset to `{data_path}/{sp_mode}-generated/`.

## Module Structure

| Module | Description |
|---|---|
| `__main__.py` | CLI entry point, argument parsing, launches parallel generation |
| `generator.py` | Multi-GPU parallelization via `torch.multiprocessing`, acceptance function factories (`create_lsh_acceptance_fn`, `create_kmeans_acceptance_fn`), mode setup |
| `sampler.py` | `BatchedRejectionSampler`: core generation engine with batched candidate generation and sentence-level stopping criteria |
| `sbert_lsh_model.py` | `SBERTLSHModel` / `LSHModel`: sentence embedding + LSH hashing via NearPy `RandomBinaryProjections` |
| `lsh_utils.py` | LSH partition logic: `get_mask_from_seed` (valid partition selection), `reject_close_generation` (margin enforcement) |
| `kmeans_utils.py` | K-means partition logic: `get_cluster_mask`, `kmeans_reject_overlap`, `get_cluster_id`, cluster center generation (also runnable as `python -m sampling.kmeans_utils`) |
| `utils.py` | Shared constants (`hash_key`, `PUNCTS`) and `extract_prompt_from_text` |

## Notes

- Requires at least one GPU. Multiple GPUs are used automatically via dataset sharding.
- The GPU used for generation must match the GPU used for detection (LSH projection consistency).
- Generation parameters (`sp_dim`, `lmbd`, `delta`, `embedder`, `secret_message`) must match those used during detection.
- K-means modes require pre-computed cluster centers (`--cc_path`). Generate them with `python -m sampling.kmeans_utils DATA_PATH EMBEDDER_PATH SP_DIM`.
