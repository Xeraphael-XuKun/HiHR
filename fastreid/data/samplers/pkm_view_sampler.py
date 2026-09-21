# encoding: utf-8

from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler

from fastreid.utils import comm


class StratifiedPKMViewSampler(Sampler):
    """Sample identity x modality groups with conditional Ground/Aerial balance.

    A cross-view identity contributes one Ground and one Aerial image per
    modality. A Ground-only identity contributes two Ground images per modality.
    The formal WHU-MARS configuration uses 5 identities of each type, three
    modalities, and K=2, yielding 60 images per batch.
    """

    def __init__(self, data_source, mini_batch_size, num_instances,
                 num_cross_view_pids=5, num_ground_only_pids=5,
                 num_modalities=3, seed=0):
        if comm.get_world_size() != 1:
            raise ValueError("StratifiedPKMViewSampler is defined for WORLD_SIZE=1")
        if num_instances != 2:
            raise ValueError("StratifiedPKMViewSampler requires K=2")

        self.data_source = data_source
        self.num_instances = int(num_instances)
        self.num_cross_view_pids = int(num_cross_view_pids)
        self.num_ground_only_pids = int(num_ground_only_pids)
        self.num_modalities = int(num_modalities)
        self.batch_size = int(mini_batch_size)
        self._seed = int(seed)

        expected = (
            (self.num_cross_view_pids + self.num_ground_only_pids)
            * self.num_modalities * self.num_instances
        )
        if self.batch_size != expected:
            raise ValueError("Expected batch size {}, got {}".format(expected, self.batch_size))

        modality_ids = {"RGB": 0, "IR": 1, "Thermal": 2}
        self.pid_mod_view = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, item in enumerate(data_source):
            pid, view, modality = item[1], item[3], modality_ids[item[4]]
            self.pid_mod_view[pid][modality][view].append(index)

        modalities = tuple(range(self.num_modalities))
        cross_view_pids = []
        ground_only_pids = []
        for pid, by_modality in self.pid_mod_view.items():
            has_ground = all(by_modality[m]["Ground"] for m in modalities)
            has_aerial = all(by_modality[m]["Aerial"] for m in modalities)
            if has_ground and has_aerial:
                cross_view_pids.append(pid)
            elif has_ground and not any(by_modality[m]["Aerial"] for m in modalities):
                ground_only_pids.append(pid)

        self.cross_view_pids = sorted(cross_view_pids)
        self.ground_only_pids = sorted(ground_only_pids)
        if len(self.cross_view_pids) < self.num_cross_view_pids:
            raise ValueError("Not enough cross-view identities")
        if len(self.ground_only_pids) < self.num_ground_only_pids:
            raise ValueError("Not enough Ground-only identities")

    @staticmethod
    def _sample(rng, indices, count):
        replace = len(indices) < count
        return rng.choice(indices, size=count, replace=replace).tolist()

    def __iter__(self):
        yield from self._infinite_indices()

    def _infinite_indices(self):
        rng = np.random.RandomState(self._seed)
        modalities = tuple(range(self.num_modalities))
        while True:
            cross_pids = rng.choice(
                self.cross_view_pids, self.num_cross_view_pids, replace=False
            ).tolist()
            ground_pids = rng.choice(
                self.ground_only_pids, self.num_ground_only_pids, replace=False
            ).tolist()

            batch_indices = []
            for pid in cross_pids:
                for modality in modalities:
                    groups = self.pid_mod_view[pid][modality]
                    batch_indices.extend(self._sample(rng, groups["Ground"], 1))
                    batch_indices.extend(self._sample(rng, groups["Aerial"], 1))
            for pid in ground_pids:
                for modality in modalities:
                    groups = self.pid_mod_view[pid][modality]
                    batch_indices.extend(self._sample(rng, groups["Ground"], 2))

            rng.shuffle(batch_indices)
            yield from batch_indices
