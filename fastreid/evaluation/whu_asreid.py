# encoding: utf-8

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _rank_indices(qf, gf, chunk_size):
    indices = []
    gf_t = gf.t().contiguous()
    gf_sq = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()
    for start in range(0, qf.shape[0], chunk_size):
        q = qf[start:start + chunk_size]
        dist = torch.pow(q, 2).sum(dim=1, keepdim=True) + gf_sq
        dist.addmm_(q, gf_t, beta=1, alpha=-2)
        indices.append(torch.argsort(dist, dim=1).cpu().numpy().astype(np.int32))
    return np.concatenate(indices, axis=0)


def _empty_result(max_rank):
    return np.full(int(max_rank), np.nan, dtype=np.float32), float("nan"), 0


def _evaluate_indices(indices, q_pids, g_pids, q_camids, g_camids, max_rank=50,
                      allow_no_valid=False):
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_ap = []

    for q_idx in range(indices.shape[0]):
        order = indices[q_idx]
        keep = g_camids[order] != q_camids[q_idx]
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = raw_cmc.sum()
        precision = raw_cmc.cumsum() / np.arange(1, raw_cmc.size + 1)
        all_ap.append((precision * raw_cmc).sum() / num_rel)

    if not all_cmc:
        if allow_no_valid:
            return _empty_result(max_rank)
        raise RuntimeError("No valid WHU-MARS query remains after same-camera filtering")
    return (
        np.asarray(all_cmc, dtype=np.float32).mean(axis=0),
        float(np.mean(all_ap)),
        len(all_cmc),
    )


@torch.no_grad()
def evaluate_whu_asreid(qf, gf, q_pids, g_pids, q_camids, g_camids,
                        max_rank=50, chunk_size=4000, allow_no_valid=False):
    """UAD-compatible WHU-MARS AS-ReID evaluation.

    Features are L2-normalized, ranked by squared Euclidean distance, and every
    gallery image captured by the query camera is excluded.
    """
    qf = F.normalize(qf.float(), p=2, dim=1)
    gf = F.normalize(gf.float(), p=2, dim=1)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)
    if qf.shape[0] == 0 or gf.shape[0] == 0:
        if allow_no_valid:
            return _empty_result(max_rank)
        raise RuntimeError("WHU-MARS query and gallery must both be non-empty")
    indices = _rank_indices(qf, gf, int(chunk_size))
    return _evaluate_indices(
        indices, q_pids, g_pids, q_camids, g_camids,
        max_rank=max_rank,
        allow_no_valid=allow_no_valid,
    )
