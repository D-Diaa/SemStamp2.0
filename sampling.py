import pprint
import argparse
import os
import torch.multiprocessing as mp
from datasets import load_from_disk, Dataset
import torch
from transformers import GenerationConfig
from sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, Queue
from nltk.tokenize import sent_tokenize
from sampling_utils import extract_prompt_from_text
from sampling_lsh_utils import get_mask_from_seed, reject_close_generation
from sampling_kmeans_utils import get_cluster_mask, kmeans_reject_overlap, get_cluster_id
from batched_rejection_sampler import BatchedRejectionSampler

PUNCTS = '.,!?'

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
    parser.add_argument('--secret_message', type=str, default="The quick brown fox jumps over the lazy dog.",
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
    pp.pprint(vars(args))  # Debug print for parsed arguments
    return args

def worker(rank, dataset_chunk, output_queue, args, device):
    """
    Worker function to process a dataset chunk on a single GPU.
    """
    sampler = BatchedRejectionSampler(
        model_path=args.model,
        device=device,
        batch_size=getattr(args, 'batch_size', 16),
        max_batches=getattr(args, 'max_batches', 8),
        dtype=torch.bfloat16,
    )
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=True,
        temperature=getattr(args, 'temperature', 1.0),
        top_k=getattr(args, 'top_k', 50),
        top_p=getattr(args, 'top_p', 1.0),
        repetition_penalty=getattr(args, 'rep_p', 1.0),
        pad_token_id=sampler.tokenizer.pad_token_id,
    )
    if args.sp_mode == "lsh":

        lsh_model = SBERTLSHModel(
            lsh_model_path=args.embedder, device=device, batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )

        def create_lsh_acceptance_fn(lsh_model, lsh_dim, lmbd, margin):
            """Create an LSH acceptance function for the batched sampler."""
            def acceptance_fn(context: str, candidate: str) -> bool:
                # Get the last sentence from context to determine seed
                sentences = sent_tokenize(context)
                if sentences:
                    last_sentence = sentences[-1]
                    lsh_seed = lsh_model.get_hash([last_sentence])[0]
                else:
                    # Use prompt hash as seed
                    lsh_seed = lsh_model.get_hash([context])[0]

                accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)

                # Check margin condition (reject if too close to hyperplanes)
                accepted_text, _ = reject_close_generation(
                    lsh_model, [candidate], margin=margin, cutoff=None
                )
                if len(accepted_text) == 0:
                    return False

                # Check if candidate's hash is in accept mask
                candidate_hash = lsh_model.get_hash([candidate])[0]
                return candidate_hash in accept_mask

            return acceptance_fn

        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            print(f"prompt: {prompt}")

            acceptance_fn = create_lsh_acceptance_fn(
                lsh_model, args.sp_dim, args.lmbd, args.delta
            )

            response = sampler.generate_continuation(
                prompt=prompt,
                gen_config=gen_config,
                acceptance_fn=acceptance_fn,
                max_tokens=args.max_new_tokens,
            )
            print(f"response: {response}")
            ex['text'] = response.strip()
            return ex

    elif args.sp_mode == "kmeans":

        cluster_centers = torch.load(args.cc_path)
        embedder = SentenceTransformer(args.embedder, device=device)

        def create_kmeans_acceptance_fn(embedder, cluster_centers, k_dim, lmbd, margin):
            """Create a kmeans acceptance function for the batched sampler."""
            def acceptance_fn(context: str, candidate: str) -> bool:
                # Get the last sentence from context to determine cluster seed
                sentences = sent_tokenize(context)
                if sentences:
                    last_sentence = sentences[-1]
                    curr_cluster_id = get_cluster_id(last_sentence, cluster_centers, embedder)
                else:
                    curr_cluster_id = get_cluster_id(context, cluster_centers, embedder)

                accept_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)

                # Check margin condition (reject if too close to cluster boundary)
                accepted_text, candidate_cluster_id = kmeans_reject_overlap(
                    text=candidate, embedder=embedder, cluster_centers=cluster_centers, margin=margin
                )
                if accepted_text is None:
                    return False

                # Check if candidate's cluster ID is in accept mask
                return candidate_cluster_id in accept_mask

            return acceptance_fn

        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            print(f"prompt: {prompt}")

            acceptance_fn = create_kmeans_acceptance_fn(
                embedder, cluster_centers, args.sp_dim, args.lmbd, args.delta
            )

            response = sampler.generate_continuation(
                prompt=prompt,
                gen_config=gen_config,
                acceptance_fn=acceptance_fn,
                max_tokens=args.max_new_tokens,
            )
            print(f"response: {response}")
            ex['text'] = response.strip()
            return ex

    elif args.sp_mode == "lsh_fixed":
        if args.secret_message is None:
            raise ValueError("--secret_message is required for lsh_fixed mode")

        lsh_model = SBERTLSHModel(
            lsh_model_path=args.embedder, device=device, batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )

        # Get fixed seed from secret message using the same hashing as normal LSH
        fixed_lsh_seed = lsh_model.get_hash([args.secret_message])[0]

        def create_lsh_fixed_acceptance_fn(lsh_model, lsh_dim, lmbd, margin, fixed_lsh_seed):
            """Create an LSH fixed acceptance function using secret message seed."""
            accept_mask = get_mask_from_seed(lsh_dim, lmbd, fixed_lsh_seed)

            def acceptance_fn(context: str, candidate: str) -> bool:
                # Check margin condition (reject if too close to hyperplanes)
                accepted_text, _ = reject_close_generation(
                    lsh_model, [candidate], margin=margin, cutoff=None
                )
                if len(accepted_text) == 0:
                    return False

                # Check if candidate's hash is in the fixed accept mask
                candidate_hash = lsh_model.get_hash([candidate])[0]
                return candidate_hash in accept_mask

            return acceptance_fn

        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            print(f"prompt: {prompt}")

            acceptance_fn = create_lsh_fixed_acceptance_fn(
                lsh_model, args.sp_dim, args.lmbd, args.delta, fixed_lsh_seed
            )

            response = sampler.generate_continuation(
                prompt=prompt,
                gen_config=gen_config,
                acceptance_fn=acceptance_fn,
                max_tokens=args.max_new_tokens,
            )
            print(f"response: {response}")
            ex['text'] = response.strip()
            return ex

    elif args.sp_mode == "kmeans_fixed":
        if args.secret_message is None:
            raise ValueError("--secret_message is required for kmeans_fixed mode")

        cluster_centers = torch.load(args.cc_path)
        embedder = SentenceTransformer(args.embedder, device=device)

        # Get fixed cluster ID from secret message using the same method as normal kmeans
        fixed_cluster_id = get_cluster_id(args.secret_message, cluster_centers, embedder)

        def create_kmeans_fixed_acceptance_fn(embedder, cluster_centers, k_dim, lmbd, margin, fixed_cluster_id):
            """Create a kmeans fixed acceptance function using secret message seed."""
            accept_mask = get_cluster_mask(fixed_cluster_id, k_dim, lmbd)

            def acceptance_fn(context: str, candidate: str) -> bool:
                # Check margin condition (reject if too close to cluster boundary)
                accepted_text, candidate_cluster_id = kmeans_reject_overlap(
                    text=candidate, embedder=embedder, cluster_centers=cluster_centers, margin=margin
                )
                if accepted_text is None:
                    return False

                # Check if candidate's cluster ID is in the fixed accept mask
                return candidate_cluster_id in accept_mask

            return acceptance_fn

        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            print(f"prompt: {prompt}")

            acceptance_fn = create_kmeans_fixed_acceptance_fn(
                embedder, cluster_centers, args.sp_dim, args.lmbd, args.delta, fixed_cluster_id
            )

            response = sampler.generate_continuation(
                prompt=prompt,
                gen_config=gen_config,
                acceptance_fn=acceptance_fn,
                max_tokens=args.max_new_tokens,
            )
            print(f"response: {response}")
            ex['text'] = response.strip()
            return ex

    else:
        raise NotImplementedError

    processed_chunk = dataset_chunk.map(text_to_generated_text, batch_size=1)
    output_queue.put(processed_chunk)

def parallel_generate(args):
    """
    Splits the dataset and distributes work across multiple GPUs by index.
    """
    dataset = load_from_disk(args.data)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected. This script requires at least one GPU.")

    print(f"Detected {num_gpus} GPU(s). Splitting dataset for parallel processing.")

    output_queue = Queue()
    processes = []

    # Create a process for each GPU and its respective dataset shard
    for rank in range(num_gpus):
        device = f"cuda:{rank}"
        # Shard the dataset by assigning a specific index to the GPU
        dataset_chunk = dataset.shard(num_shards=num_gpus, index=rank)
        p = Process(target=worker, args=(rank, dataset_chunk, output_queue, args, device))
        p.start()
        processes.append(p)

    all_results = []
    for _ in processes:
        all_results.append(output_queue.get())

    for p in processes:
        p.join()

    # Combine all results into a single dataset
    merged_dataset = Dataset.from_dict({'text': [item['text'] for chunk in all_results for item in chunk]})
    output_path = os.path.join(args.data, f"{args.sp_mode}-generated")
    os.makedirs(output_path, exist_ok=True)
    merged_dataset.save_to_disk(output_path)

if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    parallel_generate(args)
