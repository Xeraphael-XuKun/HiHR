# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank_cylib import compile_helper
from .whu_asreid import evaluate_whu_asreid

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')

        self._predictions = []
        if str(cfg.TEST.PROTOCOL).upper() != "WHU_AS_REID":
            self._compile_dependencies()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device),
            'modalityids': inputs.get('modalityids', torch.full_like(inputs['targets'], -1)).to(self._cpu_device),
            'viewids': list(inputs.get('viewids', [])),

        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        modalityids = []
        viewids = []
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            modalityids.append(prediction['modalityids'])
            viewids.extend(prediction['viewids'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        modalityids = torch.cat(modalityids, dim=0).numpy()
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]
        query_modalityids = modalityids[:self._num_query]
        gallery_modalityids = modalityids[self._num_query:]
        query_viewids = np.asarray(viewids[:self._num_query])
        gallery_viewids = np.asarray(viewids[self._num_query:])

        self._results = OrderedDict()

        if str(self.cfg.TEST.PROTOCOL).upper() == "WHU_AS_REID":
            if self.cfg.TEST.AQE.ENABLED or self.cfg.TEST.RERANK.ENABLED or self.cfg.TEST.FLIP.ENABLED:
                raise ValueError("WHU_AS_REID requires AQE, reranking, and test flip to be disabled")
            cmc, mAP, valid_queries = evaluate_whu_asreid(
                query_features, gallery_features,
                query_pids, gallery_pids,
                query_camids, gallery_camids,
                max_rank=50,
                chunk_size=self.cfg.TEST.DISTMAT_CHUNK,
            )
            for r in [1, 5, 10]:
                self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
            self._results['mAP'] = mAP * 100
            self._results['valid_queries'] = valid_queries
            self._results['metric'] = (mAP + cmc[0]) / 2 * 100

            if self.cfg.TEST.WHU_DIAGNOSTICS:
                modality_names = {0: "RGB", 1: "NIR", 2: "TIR"}
                for q_mod, q_name in modality_names.items():
                    for g_mod, g_name in modality_names.items():
                        q_mask = query_modalityids == q_mod
                        g_mask = gallery_modalityids == g_mod
                        pair_cmc, pair_map, pair_valid_queries = evaluate_whu_asreid(
                            query_features[q_mask], gallery_features[g_mask],
                            query_pids[q_mask], gallery_pids[g_mask],
                            query_camids[q_mask], gallery_camids[g_mask],
                            max_rank=50,
                            chunk_size=self.cfg.TEST.DISTMAT_CHUNK,
                            allow_no_valid=True,
                        )
                        prefix = '{}->{}'.format(q_name, g_name)
                        self._results[prefix + '/mAP'] = pair_map * 100
                        self._results[prefix + '/Rank-1'] = pair_cmc[0] * 100
                        self._results[prefix + '/valid_queries'] = pair_valid_queries

                for q_view in ("Aerial", "Ground"):
                    for g_view in ("Aerial", "Ground"):
                        q_mask = query_viewids == q_view
                        g_mask = gallery_viewids == g_view
                        pair_cmc, pair_map, pair_valid_queries = evaluate_whu_asreid(
                            query_features[q_mask], gallery_features[g_mask],
                            query_pids[q_mask], gallery_pids[g_mask],
                            query_camids[q_mask], gallery_camids[g_mask],
                            max_rank=50,
                            chunk_size=self.cfg.TEST.DISTMAT_CHUNK,
                            allow_no_valid=True,
                        )
                        prefix = '{}->{}'.format(q_view, g_view)
                        if pair_valid_queries == 0:
                            logger.warning(
                                "WHU diagnostic %s has no valid query after same-camera "
                                "filtering; reporting mAP/Rank-1 as NaN",
                                prefix,
                            )
                        self._results[prefix + '/mAP'] = pair_map * 100
                        self._results[prefix + '/Rank-1'] = pair_cmc[0] * 100
                        self._results[prefix + '/valid_queries'] = pair_valid_queries
            return copy.deepcopy(self._results)

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC, curv=self.cfg.MODEL.BACKBONE.CURV_INIT)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        from .rank import evaluate_rank
        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                compile_helper()

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        comm.synchronize()
