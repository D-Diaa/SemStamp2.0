# Quality Evaluation Package

Evaluates text generation quality using multiple metrics: perplexity, n-gram entropy, n-gram repetition, semantic entropy, MAUVE, and BERTScore.

## Modules

| Module | Description |
|---|---|
| `__main__.py` | CLI entry point for single dataset evaluation |
| `batch_eval.py` | Batch evaluation across all generated/paraphrased datasets in a directory |
| `evaluator.py` | Orchestrates all metric computations via `eval_quality()` |
| `metrics.py` | Individual metric functions: `run_entropy`, `run_ngrams`, `run_mauve`, `run_bertscore`, `run_sem_ent` |
| `utils.py` | Low-level utilities: `eval_perplexity`, `text_entropy`, `rep_ngram` |

## Metrics

| Metric | Key | Description |
|---|---|---|
| Perplexity | `gen_ppl` | Sentence-averaged causal LM perplexity |
| Bigram Entropy | `bi_entro` | Token-level bigram entropy |
| Trigram Entropy | `tri_entro` | Token-level trigram entropy |
| Repetition (2/3/4) | `rep_2`, `rep_3`, `rep_4` | N-gram repetition rate (1 - unique/total) |
| Semantic Entropy | `sem_ent` | Entropy over k-means clusters of LM hidden states |
| MAUVE | `mauve` | Distribution similarity to reference text |
| BERTScore | `bert_P`, `bert_R`, `bert_F1` | Precision/recall/F1 using roberta-large embeddings |

## Usage

### Single Dataset

```bash
python -m quality DATA_PATH MAX_NEW_TOKEN --reference REF_PATH --corpus CORPUS_PATH
```

- `--column`: Dataset column to evaluate (default: `para_text`)
- `--load_kmeans_path`: Skip k-means training by loading pre-computed centroids
- Output: `eval_quality.csv` in the dataset directory

### Batch Evaluation

```bash
python -m quality.batch_eval BASE_DIR --corpus CORPUS_PATH
```

Automatically discovers subdirectories matching `*-generated*` under `BASE_DIR`:

- Directories ending in `-generated` (watermarked-only): evaluates the `text` column
- Directories with `-generated-*` (paraphrased): evaluates the `para_text` column

Both output `eval_quality.csv`.

Options:

- `--reference`: Reference dataset for MAUVE/BERTScore (default: `BASE_DIR`)
- `--force`: Re-evaluate even if `eval_quality.csv` already exists
- `--model_path`: Causal LM for perplexity/embedding (default: `meta-llama/Llama-3.1-8B`)
- `--max_new_token`: Max token length for semantic entropy embedding (default: `512`)
- `--cluster_size`: Number of k-means clusters (default: `50`)
