from sklearn.metrics import roc_curve, auc
from sampling import utils as sampling_utils
from sampling.lsh_utils import get_mask_from_seed
from sampling.kmeans_utils import get_cluster_mask, get_cluster_id
import numpy as np
import torch
from bert_score import score as bert_score_func
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)


def run_bert_score(gen_sents, para_sents, max_len=450, batch_size=32):
    if not gen_sents or not para_sents:
        return 0.0
    # Filter out empty strings to avoid bert_score tokenizer errors
    filtered = [(g, p) for g, p in zip(gen_sents, para_sents) if g.strip() and p.strip()]
    if not filtered:
        return 0.0
    gen_sents, para_sents = zip(*filtered)
    P, R, F1 = bert_score_func(
        gen_sents, para_sents,
        model_type="roberta-large",
        device=device,
        lang="en",
        batch_size=batch_size
    )
    return torch.mean(F1).item()


def flatten_gens_and_paras(gens, paras):
    new_gens = []
    new_paras = []
    for gen, para in zip(gens, paras):
        min_len = min(len(gen), len(para))
        new_gens.extend(gen[:min_len])
        new_paras.extend(para[:min_len])
    return new_gens, new_paras


def truncate_to_max_length(texts, max_length):
    new_texts = []
    for t in texts:
        t = " ".join(t.split(" ")[:max_length])
        if t[-1] not in sampling_utils.PUNCTS:
            t = t + "."
        new_texts.append(t)
    return new_texts


def compute_zscore(n_watermark, n_test_sent, lmbd):
    """Compute z-score from watermark count."""
    num = n_watermark - lmbd * n_test_sent
    denom = np.sqrt(n_test_sent * lmbd * (1 - lmbd))
    return num / denom


def detect_lsh(sents, lsh_model, lmbd, lsh_dim, secret_message=None):
    """
    Detect LSH watermark in sentences.
    If secret_message is provided, use fixed mode (same mask for all sentences).
    Otherwise, use context-dependent mode (mask depends on previous sentence).
    """
    n_sent = len(sents)
    n_watermark = 0
    if secret_message is None:
        lsh_seed = lsh_model.get_hash([sents[0]])[0]
    else:
        lsh_seed = lsh_model.get_hash([secret_message])[0]
    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)
    for i in range(1, n_sent):
        lsh_candidate = lsh_model.get_hash([sents[i]])[0]
        if lsh_candidate in accept_mask:
            n_watermark += 1
        if secret_message is None:
            lsh_seed = lsh_candidate
            accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)
    n_test_sent = n_sent - 1  # exclude the prompt

    return compute_zscore(n_watermark, n_test_sent, lmbd)


def detect_kmeans(sents, embedder, lmbd, k_dim, cluster_centers, secret_message=None):
    """
    Detect k-means watermark in sentences.
    If secret_message is provided, use fixed mode (same mask for all sentences).
    Otherwise, use context-dependent mode (mask depends on previous sentence).
    """
    n_sent = len(sents)
    n_watermark = 0
    if secret_message is None:
        curr_cluster_id = get_cluster_id(sents[0], embedder=embedder, cluster_centers=cluster_centers)
    else:
        curr_cluster_id = get_cluster_id(secret_message, embedder=embedder, cluster_centers=cluster_centers)
    cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)
    for i in range(1, n_sent):
        curr_cluster_id = get_cluster_id(sents[i], embedder=embedder, cluster_centers=cluster_centers)
        if curr_cluster_id in cluster_mask:
            n_watermark += 1
        if secret_message is None:
            cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)
    n_test_sent = n_sent - 1  # exclude the prompt
    return compute_zscore(n_watermark, n_test_sent, lmbd)


def get_roc_metrics(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_roc_metrics_from_zscores(m, mp, h, dataset_path):
    mp = np.nan_to_num(mp)
    h = np.nan_to_num(h)
    len_z = len(mp)
    mp_fpr, mp_tpr, mp_area = get_roc_metrics(
        [1] * len_z + [0] * len_z, np.concatenate((mp, h[:len_z])))
    plt.plot(mp_fpr, mp_tpr)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")
    name = os.path.join(dataset_path, "roc_curve.png")
    plt.savefig(name)
    name = os.path.join(dataset_path, "fpr.npy")
    np.save(name, mp_fpr)
    name = os.path.join(dataset_path, "tpr.npy")
    np.save(name, mp_tpr)
    return mp_area, mp_fpr


def evaluate_z_scores(mz, mpz, hz, dataset_path):
    mz = np.array(mz)
    mpz = np.array(mpz)
    hz = np.nan_to_num(np.array(hz))
    # Use quantiles from human z-scores to find thresholds
    # 99th percentile -> 1% of human texts above this threshold (FPR=1%)
    # 95th percentile -> 5% of human texts above this threshold (FPR=5%)
    fpr_1_threshold = np.percentile(hz, 99)
    fpr_5_threshold = np.percentile(hz, 95)
    mp_area, mp_fpr = get_roc_metrics_from_zscores(mz, mpz, hz, dataset_path)
    return mp_area, len(mpz[mpz > fpr_1_threshold]) / len(mpz), len(mpz[mpz > fpr_5_threshold]) / len(mpz)
