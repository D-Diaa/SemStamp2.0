import argparse
import csv
from datasets import load_from_disk
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from .eval_clm import eval_perplexity, text_entropy, rep_ngram
from collections import Counter
import os
import faiss
import pickle
import torch
from tqdm import trange
import sys
import mauve
from bert_score import score as bert_score

def tokenize_untokenize(text, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    tokens = []
    for id in input_ids:
        tokens.append(tokenizer.decode(id, skip_special_tokens=True))
    return tokens

def run_entropy(gens, tokenizer):
    gen_sen_lis = [tokenize_untokenize(g, tokenizer) for g in gens]
    gen_entros = []
    for k in [2, 3]:
        gen_entro = text_entropy(gen_sen_lis, k=k)
        gen_entros.append(gen_entro)
        # print(f"generation {k}-gram entropy: {gen_entro}")
    return gen_entros

def run_ngrams(gens, tokenizer):
    gen_sen_lis = [tokenize_untokenize(g, tokenizer) for g in gens]
    rep_scores = []
    for k in [2,3,4]:
        rep_score = rep_ngram(gen_sen_lis, k)
        rep_scores.append(rep_score)
    return rep_scores

def run_mauve(gens, refs):
    result = mauve.compute_mauve(p_text=refs, q_text=gens, device_id=0, max_text_length=512, verbose=False)
    return result.mauve

def run_bertscore(gens, refs):
    P, R, F1 = bert_score(gens, refs, lang="en", verbose=False)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def run_sem_ent(model, gens, tokenizer, ref_texts, args):
    # Step 0: parse args
    load_kmeans_path = args.load_kmeans_path
    save_kmeans_path = args.corpus
    load_testgen_path = args.load_testgen_path
    save_testgen_path = args.dataset_name
    max_new_token = args.max_new_token
    mode=args.sem_ent_mode 
    cluster_size = args.cluster_size

    # Step 1: Perform clustering (if no centroids provided)
    def batch_tokenize(texts):
        input_ids = tokenizer(list(texts), return_tensors="pt", max_length = max_new_token, truncation=True, padding="longest").input_ids.to("cuda")
        return input_ids.unsqueeze(1) # shape: num of text X longest X embed dim 

    def embed_gen(texts, desc="Featurizing", mode='last_token', cluster_size = 500):
        input_ids = batch_tokenize(texts)
        sent_len = [len(tokenizer(text, return_tensors="pt", truncation= True, max_length = max_new_token).input_ids[0]) for text in texts]
        out_embeds = []
        if mode == 'last_token':
            out_embeds = [model(input_ids[i], output_hidden_states=True).hidden_states[-1][0][sent_len[i] - 1].cpu().detach().squeeze() for i in trange(len(input_ids), desc = desc)]
        elif mode == 'last_mean_pooling':
            for i in trange(len(input_ids), desc = desc):
                out_embeds.append(torch.mean([model(input_ids[i], output_hidden_states=True).hidden_states[-1][0][j].cpu().detach().squeeze()]) for j in range(sent_len))
        elif mode == 'all_mean_pooling':
            for i in trange(len(input_ids), desc = desc):
                hidden_states = model(input_ids[i], output_hidden_states=True).hidden_states
                hidden_states_len = len(hidden_states) # 33
                hidden_states_list = [hidden_states[k][0][sent_len[i]-1].cpu().detach().squeeze() for k in range(hidden_states_len)]
                # shape=[2560]
                pooled_states = torch.mean(torch.stack(hidden_states_list), dim=0)
                out_embeds.append(pooled_states)
        # out_embeds shape: len(input_ids) X 2560 (hidden_state_size/num of features)
        input_tensor = torch.stack(out_embeds).float().numpy()
        return input_tensor

    # build from scratch if no stored index and centroids
    
    if load_kmeans_path == None:
        ncentroids = cluster_size
        niter = 100
        verbose = True
        input_tensor = embed_gen(ref_texts, desc="Featurizing train generations", mode=mode)
        
        d = input_tensor.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(input_tensor)
        centroids = kmeans.centroids
        index = kmeans.index
        # save
        cpath = os.path.join(save_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-centroids.npy")
        ipath = os.path.join(save_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-index.pkl")
        with open(cpath, "wb") as f:
            np.save(f, centroids)
            f.close()
        with open(ipath, "wb") as f:
            pickle.dump(index, f)
            f.close()
    # Otherwise, load saved index and centroids
    else:
        cpath = os.path.join(load_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-centroids.npy")
        ipath = os.path.join(load_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-index.pkl")
        centroids = np.load(cpath)
        with open(ipath, "rb") as f:
            index = pickle.load(f)
            f.close()
                
    # Step 2: Generate and assign clusters
    if load_testgen_path == None:
        gen_embed = embed_gen(gens, "Featurizing test generations", mode=mode)
        # save
        gen_embed_path = os.path.join(save_testgen_path, "test-gen-embed.pkl")
        with open(gen_embed_path, "wb") as f:
            pickle.dump(gen_embed, f)
            f.close()
    else:
        gen_embed_path = os.path.join(load_testgen_path, "test-gen-embed.pkl")
        with open(gen_embed_path, "rb") as f:
            gen_embed = pickle.load(f)
            f.close()
    # find 1-closest cluster
    D, I = index.search(gen_embed, 1)
    
    # Step 3: Calculate entropy
    freq_dict = Counter(list(I.squeeze()))
    freqs = list(freq_dict.values())
    total_freqs = np.sum(freqs)
    approx_p = [f / total_freqs for f in freqs]
    sem_ent = np.sum([-1 * p * np.log(p) for p in approx_p])
    with open(os.path.join(save_testgen_path, f'mode={mode}-cluster_size={cluster_size}-distribution.txt'), 'wb') as f:
        print(f"Semantic Ent: {sem_ent}")
        # print(f"distribution: {approx_p}")
        f.close()
    return sem_ent
    
def eval_quality(model, gens, corpus_texts, ref_texts, tokenizer, args):

    if (type(gens[0]) == list):
        gens_text = [" ".join(g) for g in gens]
    else:
        gens_text = gens
    print("Evaluating perplexity...")
    gen_ppl = eval_perplexity(model, tokenizer, gens_text)

    print("Evaluating semantic entropy")
    sem_ent = run_sem_ent(model, gens, tokenizer, corpus_texts, args)

    print("Evaluating n-gram repetition...")
    rep_scores = run_ngrams(gens_text, tokenizer)

    print("Evaluating entropy...")
    gen_entros = run_entropy(gens_text, tokenizer)

    print("Evaluating MAUVE...")
    mauve_score = run_mauve(list(gens_text), list(ref_texts))

    print("Evaluating BERTScore...")
    bert_P, bert_R, bert_F1 = run_bertscore(list(gens_text), list(ref_texts))

    return gen_ppl, gen_entros[0], gen_entros[1], rep_scores[0], rep_scores[1], rep_scores[2], sem_ent, mauve_score, bert_P, bert_R, bert_F1

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
    args = parser.parse_args()
    if args.load_kmeans_path is None and args.corpus is None:
        parser.error('--corpus is required when --load_kmeans_path is not provided')
    return args

if __name__ == '__main__':
    args = parse_args()
    gens = load_from_disk(args.dataset_name)['text']
    ref_texts = load_from_disk(args.reference)['text']
    if args.corpus is not None:
        corpus_texts = load_from_disk(args.corpus)['text']
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

    csv_path = os.path.join(args.dataset_name, "eval_quality.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"Results saved to {csv_path}")