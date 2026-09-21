#!/usr/bin/env python
# encoding: utf-8

import argparse
import importlib.util
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastreid.config import get_cfg
from fastreid.data.datasets.whu_mars import WHUMARS
from fastreid.data.samplers.pkm_view_sampler import StratifiedPKMViewSampler
from fastreid.evaluation.whu_asreid import evaluate_whu_asreid


EXPECTED_SPLIT_SIZES = {"train": 92133, "query": 6405, "gallery": 93609}
EXPECTED_MODALITY_SIZES = {"train": 30711, "query": 2135, "gallery": 31203}


def parse_args():
    parser = argparse.ArgumentParser(description="WHU-MARS adapter/sampler/evaluator smoke test")
    parser.add_argument("--data-root", required=True, help="Parent directory containing WHU-MARS")
    parser.add_argument("--uad-root", required=True, help="CVPR26_UAD repository root")
    parser.add_argument("--batches", type=int, default=20)
    return parser.parse_args()


def check_configs(repo_root):
    configs = {}
    for name in ("Baseline", "HiHR"):
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(repo_root, "configs", "WHUMARS", name + ".yml"))
        configs[name] = cfg

    assert configs["Baseline"].MODEL.HIHR_MODE == "baseline"
    assert configs["HiHR"].MODEL.HIHR_MODE == "full"
    for cfg in configs.values():
        assert cfg.DATALOADER.SAMPLER_TRAIN == "StratifiedPKMViewSampler"
        assert cfg.SOLVER.IMS_PER_BATCH == 60
        assert cfg.TEST.PROTOCOL == "WHU_AS_REID"
        assert not cfg.TEST.AQE.ENABLED
        assert not cfg.TEST.RERANK.ENABLED
        assert not cfg.TEST.FLIP.ENABLED

    payloads = {name: yaml.safe_load(cfg.dump()) for name, cfg in configs.items()}
    for payload in payloads.values():
        del payload["MODEL"]["HIHR_MODE"]
        del payload["MODEL"]["BACKBONE"]["WITH_FUSION"]
        del payload["MODEL"]["BACKBONE"]["WITH_HIERARCHY"]
        del payload["OUTPUT_DIR"]
    assert payloads["Baseline"] == payloads["HiHR"], "Unapproved config difference"


def check_dataset_and_sampler(data_root, batches):
    dataset = WHUMARS(root=data_root, verbose=False)
    splits = {"train": dataset.train, "query": dataset.query, "gallery": dataset.gallery}
    for split_name, items in splits.items():
        assert len(items) == EXPECTED_SPLIT_SIZES[split_name]
        modality_counts = Counter(item[4] for item in items)
        assert modality_counts == Counter({m: EXPECTED_MODALITY_SIZES[split_name]
                                           for m in WHUMARS.modalities})
        assert len({item[2] for item in items}) == 7

    train_pids = {item[1] for item in dataset.train}
    assert len(train_pids) == 500

    sampler = StratifiedPKMViewSampler(
        dataset.train,
        mini_batch_size=60,
        num_instances=2,
        num_cross_view_pids=5,
        num_ground_only_pids=5,
        num_modalities=3,
        seed=1234,
    )
    assert len(sampler.cross_view_pids) == 216
    assert len(sampler.ground_only_pids) == 284

    iterator = iter(sampler)
    for batch_no in range(batches):
        batch = [dataset.train[next(iterator)] for _ in range(60)]
        assert len({item[1] for item in batch}) == 10
        assert Counter(item[4] for item in batch) == Counter({"RGB": 20, "IR": 20, "Thermal": 20})
        assert Counter(item[3] for item in batch) == Counter({"Ground": 45, "Aerial": 15})

        by_pid = defaultdict(list)
        for item in batch:
            by_pid[item[1]].append(item)
        cross_count = 0
        ground_count = 0
        for pid, samples in by_pid.items():
            assert len(samples) == 6
            by_modality = defaultdict(list)
            for sample in samples:
                by_modality[sample[4]].append(sample[3])
            assert set(by_modality) == set(WHUMARS.modalities)
            if pid in sampler.cross_view_pids:
                cross_count += 1
                assert all(Counter(views) == Counter({"Ground": 1, "Aerial": 1})
                           for views in by_modality.values())
            else:
                ground_count += 1
                assert all(Counter(views) == Counter({"Ground": 2})
                           for views in by_modality.values())
        assert (cross_count, ground_count) == (5, 5), "batch {} composition mismatch".format(batch_no)

    return dataset, sampler


def load_uad_metrics(uad_root):
    metrics_path = os.path.join(uad_root, "utils", "metrics.py")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(metrics_path)
    sys.path.insert(0, uad_root)
    spec = importlib.util.spec_from_file_location("whu_uad_metrics", metrics_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_evaluator_parity(uad_root):
    generator = torch.Generator().manual_seed(1234)
    qf = torch.randn(8, 32, generator=generator)
    gf = torch.randn(80, 32, generator=generator)
    q_pids = np.arange(8, dtype=np.int64)
    q_camids = np.arange(8, dtype=np.int64) % 7
    g_pids = np.arange(80, dtype=np.int64) + 100
    g_camids = np.arange(80, dtype=np.int64) % 7

    for idx in range(8):
        # The exact same-camera impostor must be removed. Market-style filtering
        # would keep it because its PID differs from the query PID.
        g_camids[idx] = q_camids[idx]
        gf[idx] = qf[idx]
        g_pids[idx + 8] = q_pids[idx]
        g_camids[idx + 8] = (q_camids[idx] + 1) % 7
        gf[idx + 8] = qf[idx] + 0.05 * torch.randn(32, generator=generator)

    our_cmc, our_map, valid_queries = evaluate_whu_asreid(
        qf, gf, q_pids, g_pids, q_camids, g_camids, max_rank=50, chunk_size=3
    )
    uad = load_uad_metrics(uad_root)
    qf_norm = F.normalize(qf.float(), p=2, dim=1)
    gf_norm = F.normalize(gf.float(), p=2, dim=1)
    indices = uad.compute_indices_chunked(qf_norm, gf_norm, top_k=0, q_chunk_size=3)
    uad_cmc, uad_map = uad.eval_func(indices, q_pids, g_pids, q_camids, g_camids, max_rank=50)

    np.testing.assert_allclose(our_cmc, uad_cmc, rtol=0, atol=1e-7)
    np.testing.assert_allclose(our_map, uad_map, rtol=0, atol=1e-12)
    assert valid_queries == len(q_pids)
    assert our_cmc[0] == 1.0

    empty_cmc, empty_map, empty_valid = evaluate_whu_asreid(
        qf[:1], gf[:1],
        np.asarray([0]), np.asarray([0]),
        np.asarray([0]), np.asarray([0]),
        max_rank=5,
        chunk_size=1,
        allow_no_valid=True,
    )
    assert empty_valid == 0
    assert np.isnan(empty_map)
    assert np.isnan(empty_cmc).all()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    check_configs(repo_root)
    dataset, sampler = check_dataset_and_sampler(args.data_root, args.batches)
    check_evaluator_parity(args.uad_root)
    print("WHUMARS_SMOKE_OK")
    print("split_sizes", len(dataset.train), len(dataset.query), len(dataset.gallery))
    print("identity_pools", len(sampler.cross_view_pids), len(sampler.ground_only_pids))
    print("checked_batches", args.batches)
    print("uad_protocol_parity", "PASS")


if __name__ == "__main__":
    main()
