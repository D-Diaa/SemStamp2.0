import pprint
import argparse
import torch.multiprocessing as mp
from sampling.generator import parallel_generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data', type=str, help='Path to Hugging Face dataset that has a column "text".')
    parser.add_argument(
        '--model', type=str, help='Model name to generate continuation. HuggingFace/OpenAI.', default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        '--embedder', type=str, help='Model name to embed sentences.', default="AbeHou/SemStamp-c4-sbert")
    parser.add_argument('--len_prompt', '-l', type=int, default=32,
                        help='MAX length of prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=205,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--min_new_tokens', type=int, default=195,
                        help='Minimum number of new tokens to generate.')
    parser.add_argument('--rep_p', type=float, default=1.05,
                        help='Repetition penalty.')
    parser.add_argument('--lmbd', type=float, default=0.25,
                        help='Ratio of valid sentences.')
    parser.add_argument('--delta', type=float, default=0,
                        help='Logit augmentation for baseline or margin size for LSH and KMeans.')
    parser.add_argument('--sp_mode', type=str, choices=['lsh', 'kmeans', 'lsh_fixed', 'kmeans_fixed'],
                        help='Spatial mode for generation (lsh, kmeans, lsh_fixed, or kmeans_fixed).', default=None)
    parser.add_argument('--secret_message', type=str, default="The magic words are squeamish ossifrage.",
                        help='Secret message for fixed modes (lsh_fixed, kmeans_fixed). Required for fixed modes.')
    parser.add_argument('--sp_dim', type=int, default=8,
                        help='Number of partitions in the embedding space. Default is 8.')
    parser.add_argument('--embed_path', type=str,
                        help='Path to precomputed embed for training KMeans.', default=None)
    parser.add_argument('--cc_path', type=str,
                        help='KMeans precomputed cluster centers data.', default=None)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of candidate sentences per batch for batched rejection sampling.')
    parser.add_argument('--max_batches', type=int, default=8,
                        help='Maximum number of batches to try before accepting any sample.')
    pp = pprint.PrettyPrinter(indent=4)
    args = parser.parse_args()
    pp.pprint(vars(args))
    return args


if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    parallel_generate(args)
