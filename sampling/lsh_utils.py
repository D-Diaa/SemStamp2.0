from sampling import utils as sampling_utils
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)
hash_key = sampling_utils.hash_key

def cosine_distance_matrix(x, y):
    return F.cosine_similarity(
        x.view(x.size(0), 1, x.size(1))
        .expand(x.size(0), y.size(0), x.size(1))
        .contiguous()
        .view(-1, x.size(1)),
        y.expand(x.size(0), y.size(0), y.size(1)).flatten(end_dim=1),
    ).view(x.size(0), y.size(0))


def get_mask_from_seed(lsh_dim: int, accept_rate: float, seed: int):
    n_bins = 2**lsh_dim
    n_accept = int(n_bins * accept_rate)
    rng.manual_seed(hash_key * seed)
    vocab_permutation = torch.randperm(n_bins, device=device, generator=rng)
    greenlist_ids = vocab_permutation[:n_accept]
    return greenlist_ids.to(device)


def reject_close_generation(lsh_model, sents, margin, cutoff=None):
    embeds = lsh_model.get_embeddings(sents)
    embeds = torch.tensor(embeds, device=device)
    normals = torch.tensor(lsh_model.hasher.normals, device=device)
    if cutoff != None:
        normals = normals[:cutoff]

    # sims[i, j] is the cosine similarity between the ith generation and the jth normal vec
    sims = cosine_distance_matrix(embeds, normals)
    sims_abs = torch.abs(sims)
    # max_sim is the highest cosine similarity of each generation with any normal vec
    min_sims = sims_abs.min(dim=1).values
    select = []
    for i in range(len(min_sims)):
        # print(max_sims[i])
        min_sim = min_sims[i].item()
        if (abs(min_sim) >= margin):
            select.append(i)
    sents = [sents[i] for i in select]
    return sents, select
