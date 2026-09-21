# encoding: utf-8

import glob
import os.path as osp
import re

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["WHUMARS"]


@DATASET_REGISTRY.register()
class WHUMARS(ImageDataset):
    """WHU-MARS-1000 with independent camera, view, and modality labels."""

    dataset_dir = "WHU-MARS"
    dataset_name = "whumars"
    modalities = ("RGB", "IR", "Thermal")
    pattern = re.compile(r"(\d+)_c(\d+)")

    def __init__(self, root="datasets", **kwargs):
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.data_dir, "train")
        self.query_dir = osp.join(self.data_dir, "query")
        self.gallery_dir = osp.join(self.data_dir, "test")
        self.check_before_run([self.train_dir, self.query_dir, self.gallery_dir])

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train):
        data = []
        for modality in self.modalities:
            img_paths = sorted(glob.glob(osp.join(dir_path, modality, "*.jpg")))
            for img_path in img_paths:
                match = self.pattern.search(osp.basename(img_path))
                if match is None:
                    raise ValueError("Invalid WHU-MARS filename: {}".format(img_path))
                pid, camera = map(int, match.groups())
                view = "Ground" if camera <= 5 else "Aerial"
                camid = camera - 1
                if is_train:
                    pid = self.dataset_name + "_" + str(pid)
                    camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid, view, modality))
        return data
