import argparse
import os
from datasets import load_from_disk
import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm
from kmeans_pytorch import *  # maybe faiss
import sampling_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)

hash_key = sampling_utils.hash_key

def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    # print(f'predicting on {device}..')
    
    if distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=-1)

    return choice_cluster.cpu()

def update_pickle(name, input_to_embed):
    with open(name, 'rb') as f:
        d = pickle.load(f)
        d.update(input_to_embed)
    f.close()
    with open(name, 'wb') as f:
        pickle.dump(d, f)
    f.close()

def worker(rank, text_chunk, embedder_path, queue, encode_batch_size):
    """
    Worker function to process a text chunk and generate embeddings on a specific GPU.
    """
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    embedder = SentenceTransformer(embedder_path, device=device)
    embedder = embedder.eval()

    sent_embeds = []

    # Progress bar for each worker
    with tqdm(total=len(text_chunk), desc=f"Worker {rank} Encoding", position=rank) as pbar:
        for i in range(0, len(text_chunk), encode_batch_size):
            batch_texts = text_chunk[i:i + encode_batch_size]
            batch_embeds = embedder.encode(batch_texts, convert_to_tensor=True)
            sent_embeds.extend(batch_embeds)
            pbar.update(len(batch_texts))

    # Put all embeddings into the queue
    queue.put(sent_embeds)

def embed_gen_list(dataset_path, embedder_path, encode_batch_size=32, num_gpus=torch.cuda.device_count()):
    """
    Parallelized embedding generation for the dataset with progress bars.
    """
    from multiprocessing import Process, Queue

    dataset = load_from_disk(dataset_path)
    texts = dataset['text']

    # Total progress bar
    total_progress = tqdm(total=len(texts), desc="Total Progress", position=num_gpus)

    # Split the dataset into chunks for each GPU
    text_chunks = [texts[i::num_gpus] for i in range(num_gpus)]

    # Queue to collect embeddings from workers
    queue = Queue()

    processes = []
    for rank, text_chunk in enumerate(text_chunks):
        p = Process(target=worker, args=(rank, text_chunk, embedder_path, queue, encode_batch_size))
        p.start()
        processes.append(p)

    # Collect embeddings from workers
    all_embeds = []
    while any(p.is_alive() for p in processes) or not queue.empty():
        while not queue.empty():
            chunk_embeds = queue.get()
            all_embeds.extend(chunk_embeds)
            total_progress.update(len(chunk_embeds))

    total_progress.close()

    for p in processes:
        p.join()

    # Save embeddings to a single pickle file
    name = os.path.join(dataset_path, "embeds.pkl")
    with open(name, 'wb') as f:
        pickle.dump({'text': all_embeds}, f)

    print(f"Embeddings saved to {name}")
    return name

def get_cluster_mask(curr_cluster_id, k_dim, lmbd):
    rng.manual_seed(curr_cluster_id.item() * hash_key)
    num_accept = int(k_dim * lmbd)
    mask = torch.randperm(k_dim, device=device, generator=rng)[:num_accept]
    return mask.to(device)

def kmeans_reject_overlap(text, embedder, cluster_centers, margin=0.01):
    gen_embed = embedder.encode(text, convert_to_tensor=True)
    gen_embed = gen_embed.reshape(1, -1)
    cluster_centers = torch.tensor(np.array(cluster_centers))
    dis = pairwise_cosine(gen_embed, cluster_centers, device=device)

    # each row of ranking corresponds to the cluster distance closeness of a generation
    ranked_dis = torch.argsort(dis, dim=-1)
    closest = ranked_dis[0]

    # second nearest cluster
    second_closest = ranked_dis[1]

    first_dis = dis[closest]

    sec_dis = dis[second_closest]

    if ((sec_dis - first_dis) > margin):
        return text, closest.clone().detach()
    else:
        return None, closest.clone().detach()

def get_cluster_id(text, cluster_centers, embedder):
    embedding = embedder.encode(text, convert_to_tensor=True)
    embedding = embedding.reshape(1, -1)
    # print(cluster_centers.shape)
    cluster_id = kmeans_predict(
        embedding,
        cluster_centers=cluster_centers,
        distance='cosine',
        device=device
    )
    return cluster_id

def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    # print(f"data1: {data1}")
    # print(f"data2: {data2}")
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

def get_cluster_centers(embeds, k_dim, gamma=0.002):
    cluster_ids, cluster_centers = kmeans(
        embeds,
        num_clusters=k_dim,
        distance='cosine',
        device=device
    )
    return cluster_ids, cluster_centers

def load_embeds(embed_path):
    with open(embed_path, 'rb') as f:
        d = pickle.load(f)
    # move all embeddings to the same device
    for i in range(len(d['text'])):
        d['text'][i] = d['text'][i].to(device)
    gen_embeds = torch.stack(d['text']).squeeze()
    return gen_embeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('embedder_path', type=str)
    parser.add_argument('sp_dim', type=int, default=3)
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    embed_path = embed_gen_list(args.data_path, args.embedder_path)
    print(f'Embedding generated at {embed_path}')
    print("Generating cluster centers..")
    _, cluster_centers = get_cluster_centers(load_embeds(embed_path), args.sp_dim)
    torch.save(cluster_centers, f'{args.data_path}/cc.pt')