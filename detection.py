import argparse
import os
import pandas as pd
import torch
import numpy as np
from tqdm import trange
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
from detection_utils import detect_kmeans, detect_lsh, run_bert_score, evaluate_z_scores, flatten_gens_and_paras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='hf dataset containing text and para_text columns')
    parser.add_argument('--human_text', help='hf dataset containing text column', default="data/c4-human")
    parser.add_argument('--detection_mode', choices=['kmeans', 'lsh', 'kmeans_fixed', 'lsh_fixed'],
                        help='detection mode. lsh for semstamp, kmeans for k-semstamp, lsh_fixed/kmeans_fixed for secret message modes')
    parser.add_argument('--secret_message', type=str, default="The magic words are squeamish ossifrage.",
                        help='Secret message for fixed modes (lsh_fixed, kmeans_fixed).')
    parser.add_argument('--cc_path', type=str, help='path to cluster centers')
    parser.add_argument('--embedder', type=str, default="AbeHou/SemStamp-c4-sbert", help='sentence embedder')
    parser.add_argument('--sp_dim', type=int, default=3,
                        help='dimension of the subspaces. default 3 for sstamp and 8 for ksstamp')
    parser.add_argument('--lmbd', type=float, default=0.25, help='ratio of valid sentences')
    args = parser.parse_args()
    return args


def setup_detector(args):
    """Setup detector based on detection mode. Returns (detect_fn, desc_prefix)."""
    mode = args.detection_mode
    is_fixed = mode.endswith('_fixed')
    secret_msg = args.secret_message if is_fixed else None

    if mode in ['lsh', 'lsh_fixed']:
        lsh_model = SBERTLSHModel(
            lsh_model_path=args.embedder, device='cuda', batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )
        detect_fn = lambda sents: detect_lsh(
            sents=sents, lsh_model=lsh_model, lmbd=args.lmbd, lsh_dim=args.sp_dim, secret_message=secret_msg
        )
    elif mode in ['kmeans', 'kmeans_fixed']:
        cluster_centers = torch.load(args.cc_path)
        embedder = SentenceTransformer(args.embedder)
        detect_fn = lambda sents: detect_kmeans(
            sents=sents, embedder=embedder, lmbd=args.lmbd, k_dim=args.sp_dim,
            cluster_centers=cluster_centers, secret_message=secret_msg
        )
    else:
        raise ValueError(f"Unknown detection mode: {mode}")

    return detect_fn, mode


def tokenize_text(text):
    """Convert text to list of sentences."""
    return text if isinstance(text, list) else sent_tokenize(text)


def run_detection(detect_fn, texts, desc):
    """Run detection on a list of texts."""
    scores = []
    for i in trange(len(texts), desc=desc):
        sents = tokenize_text(texts[i])
        score = detect_fn(sents)
        scores.append(score)
    return scores


def save_results(args, z_scores, para_scores, human_scores, gens, paras):
    """Save detection results and compute metrics."""
    print("Saving scores...")
    print(f"Average z-score of generations: {np.mean(z_scores):.3f}")
    print(f"Average z-score of human texts: {np.mean(human_scores):.3f}")
    print(f"Average z-score of paraphrased texts: {np.mean(para_scores):.3f}")

    np.save(os.path.join(args.dataset_path, "z_scores.npy"), z_scores)
    np.save(os.path.join(args.dataset_path, "para_z_scores.npy"), para_scores)
    np.save(os.path.join(args.dataset_path, "human_z_scores.npy"), human_scores)

    print("Evaluating z-scores...")
    auroc, fpr1, fpr5 = evaluate_z_scores(z_scores, para_scores, human_scores, args.dataset_path)

    print("Evaluating bert score...")
    gen_sents, para_sents = flatten_gens_and_paras(gens, paras)
    bert_score = run_bert_score(gen_sents, para_sents)

    metrics = [f"{auroc:.3f}", f"{fpr1:.3f}", f"{fpr5:.3f}", f"{bert_score:.3f}"]
    columns = ["auroc", "fpr1", "fpr5", "bert_score"]
    df = pd.DataFrame(data=[metrics], columns=columns)
    df.to_csv(os.path.join(args.dataset_path, "results.csv"), sep="\t", index=False)


if __name__ == '__main__':
    args = parse_args()

    # Load data
    dataset = load_from_disk(args.dataset_path)
    gens = dataset['text']
    paras = dataset['para_text'] if 'para_text' in dataset.column_names else None
    human_texts = load_from_disk(args.human_text)['text'][:len(gens)]

    # Setup detector
    detect_fn, mode_desc = setup_detector(args)

    # Run detection
    z_scores = run_detection(detect_fn, gens, f'{mode_desc}_detection')
    para_scores = run_detection(detect_fn, paras, f'{mode_desc}_para') if paras else []
    human_scores = run_detection(detect_fn, human_texts, f'{mode_desc}_human')

    # print first 5 z-scores
    print("First 5 z-scores of generated texts:", z_scores[:5])
    print("First 5 z-scores of paraphrased texts:", para_scores[:5] if para_scores else "N/A")
    print("First 5 z-scores of human texts:", human_scores[:5])

    # Save results
    save_results(args, z_scores, para_scores, human_scores, gens, paras)
