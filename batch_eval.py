import os
import glob
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from quality.eval_quality import eval_quality
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluate text generation quality across all generated datasets in a directory.')
    parser.add_argument('base_dir', type=str,
                        help='Base directory containing generated dataset subdirectories (e.g. data/c4-val-1000)')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference dataset for MAUVE/BERTScore (default: base_dir)')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to corpus dataset for semantic entropy clustering')
    parser.add_argument('--load_kmeans_path', type=str, default=None,
                        help='Path to pre-computed k-means centroids/index (default: corpus path)')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B',
                        help='HuggingFace causal LM for perplexity and embedding (default: meta-llama/Llama-3.1-8B)')
    parser.add_argument('--max_new_token', type=int, default=512,
                        help='Maximum token length for truncation during semantic entropy embedding (default: 512)')
    parser.add_argument('--cluster_size', type=int, default=50,
                        help='Number of k-means clusters for semantic entropy (default: 50)')
    parser.add_argument('--sem_ent_mode', type=str, default='last_token',
                        choices=['last_token', 'last_mean_pooling', 'all_mean_pooling'],
                        help='Embedding mode for semantic entropy (default: last_token)')
    parser.add_argument('--pattern', type=str, default='*-generated-*',
                        help='Glob pattern for matching dataset subdirectories (default: *-generated-*)')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate even if eval_quality.csv already exists')
    args = parser.parse_args()
    if args.reference is None:
        args.reference = args.base_dir
    if args.load_kmeans_path is None:
        corpus_files = os.listdir(args.corpus)
        if any(f.endswith('.pkl') for f in corpus_files):
            args.load_kmeans_path = args.corpus
    return args


if __name__ == '__main__':
    args = parse_args()

    # Find all generated dataset directories (exclude .pkl files and logs)
    dataset_dirs = sorted(glob.glob(os.path.join(args.base_dir, args.pattern)))
    dataset_dirs = [d for d in dataset_dirs if os.path.isdir(d) and "Figures" not in d]

    print(f"Found {len(dataset_dirs)} datasets:")
    for d in dataset_dirs:
        print(f"  {os.path.basename(d)}")

    # Load model once
    print(f"\nLoading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.encode(tokenizer.eos_token)[0]
    model = AutoModelForCausalLM.from_pretrained(args.model_path, return_dict=True, pad_token_id=pad_id).to("cuda")
    model.eval()

    # Load reference and corpus
    ref_texts = load_from_disk(args.reference)["text"]
    corpus_texts = load_from_disk(args.corpus)["text"]

    # Create a namespace matching what eval_quality expects
    class EvalArgs:
        def __init__(self, dataset_name):
            self.dataset_name = dataset_name
            self.max_new_token = args.max_new_token
            self.corpus = args.corpus
            self.load_kmeans_path = args.load_kmeans_path
            self.load_testgen_path = None
            self.sem_ent_mode = args.sem_ent_mode
            self.cluster_size = args.cluster_size

    for dataset_dir in dataset_dirs:
        name = os.path.basename(dataset_dir)
        csv_path = os.path.join(dataset_dir, "eval_quality.csv")
        if not args.force and os.path.exists(csv_path):
            print(f"\n{'='*60}")
            print(f"SKIPPING {name} (eval_quality.csv already exists)")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        gens = load_from_disk(dataset_dir)["para_text"]
        eval_args = EvalArgs(dataset_dir)

        gen_ppl, bi_entro, tri_entro, rep_2, rep_3, rep_4, sem_ent, mauve_score, bert_P, bert_R, bert_F1 = \
            eval_quality(model, gens, corpus_texts, ref_texts, tokenizer, eval_args)

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
            print(f"  {k}: {v:.4f}")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
        print(f"  Saved to {csv_path}")

    print("\nDone!")
