import argparse
import csv
import os

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from quality.evaluator import eval_quality


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text generation quality (perplexity, entropy, semantic entropy).')
    parser.add_argument('dataset_name', type=str,
                        help='Path to HuggingFace dataset directory containing generated text to evaluate')
    parser.add_argument('max_new_token', type=int,
                        help='Maximum token length for truncation during semantic entropy embedding')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B',
                        help='HuggingFace causal LM for perplexity and embedding (default: meta-llama/Llama-3.1-8B)')
    parser.add_argument('--cluster_size', type=int, default=50,
                        help='Number of k-means clusters for semantic entropy (default: 50, must be <= number of samples)')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to HuggingFace dataset directory containing reference text to compare against (used for MAUVE and BERTScore)')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to HuggingFace dataset directory containing text corpus for k-means clustering in semantic entropy '
                             '(required unless --load_kmeans_path is provided)')
    parser.add_argument('--sem_ent_mode', type=str, default='last_token',
                        choices=['last_token', 'last_mean_pooling', 'all_mean_pooling'],
                        help='Embedding mode for semantic entropy (default: last_token)')
    parser.add_argument('--load_kmeans_path', type=str, default=None,
                        help='Path to directory with pre-computed k-means centroids and index (skips training)')
    parser.add_argument('--load_testgen_path', type=str, default=None,
                        help='Path to directory with pre-computed test generation embeddings (skips featurization)')
    parser.add_argument('--column', type=str, default='para_text',
                        help='Dataset column to evaluate (default: para_text)')
    args = parser.parse_args()
    if args.load_kmeans_path is None and args.corpus is None:
        parser.error('--corpus is required when --load_kmeans_path is not provided')
    if args.load_kmeans_path is None:
        corpus_files = os.listdir(args.corpus)
        if any(f.endswith('.pkl') for f in corpus_files):
            print("Found pre-computed k-means centroid in corpus path")
            args.load_kmeans_path = args.corpus
    return args


if __name__ == '__main__':
    args = parse_args()
    gens = load_from_disk(args.dataset_name)[args.column]
    ref_texts = load_from_disk(args.reference)['text']
    if args.corpus is not None and args.load_kmeans_path is None:
        corpus_texts = load_from_disk(args.corpus)['text']
    else:
        corpus_texts = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.encode(tokenizer.eos_token)[0]
    model = AutoModelForCausalLM.from_pretrained(args.model_path, return_dict=True, pad_token_id=pad_id).to('cuda')
    model.eval()

    gen_ppl, bi_entro, tri_entro, rep_2, rep_3, rep_4, sem_ent, mauve_score, bert_P, bert_R, bert_F1 = \
        eval_quality(model, gens, corpus_texts, ref_texts, tokenizer, args)

    results = {
        "gen_ppl": gen_ppl,
        "bi_entro": bi_entro,
        "tri_entro": tri_entro,
        "rep_2": rep_2,
        "rep_3": rep_3,
        "rep_4": rep_4,
        "sem_ent": sem_ent,
        "mauve": mauve_score,
        "bert_P": bert_P,
        "bert_R": bert_R,
        "bert_F1": bert_F1,
    }

    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    csv_name = "eval_quality.csv"
    csv_path = os.path.join(args.dataset_name, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"Results saved to {csv_path}")
