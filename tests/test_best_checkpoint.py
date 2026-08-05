import unittest
from types import ModuleType

import sys

sys.path.append(".")

if "termcolor" not in sys.modules:
    termcolor = ModuleType("termcolor")
    termcolor.colored = lambda text, *args, **kwargs: text
    sys.modules["termcolor"] = termcolor

from fastreid.utils.checkpoint import (
    PeriodicCheckpointer,
    get_best_checkpoint_metric_names,
)


class FakeCheckpointer:
    def __init__(self):
        self.saved = []

    def save(self, name, **kwargs):
        self.saved.append((name, kwargs.copy()))


class BestCheckpointTest(unittest.TestCase):
    def test_saves_metric_and_map_best_independently(self):
        fake_checkpointer = FakeCheckpointer()
        checkpointer = PeriodicCheckpointer(fake_checkpointer, period=1, max_epoch=3)

        checkpointer.step(0, metric=50.0, mAP=40.0)
        checkpointer.step(1, metric=49.0, mAP=41.0)
        checkpointer.step(2, metric=50.0, mAP=41.0)

        saved_names = [name for name, _ in fake_checkpointer.saved]
        self.assertEqual(
            saved_names,
            ["model_best", "model_best_map", "model_best_map", "model_final"],
        )
        self.assertEqual(fake_checkpointer.saved[0][1]["metric"], 50.0)
        self.assertEqual(fake_checkpointer.saved[1][1]["mAP"], 40.0)
        self.assertFalse(fake_checkpointer.saved[1][1]["tag_last_checkpoint"])
        self.assertEqual(fake_checkpointer.saved[2][1]["mAP"], 41.0)
        self.assertFalse(fake_checkpointer.saved[2][1]["tag_last_checkpoint"])

    def test_metric_names_use_unprefixed_keys_for_single_dataset(self):
        self.assertEqual(
            get_best_checkpoint_metric_names(("AG_ReID",)),
            ("metric", "mAP"),
        )

    def test_metric_names_use_first_dataset_for_multiple_datasets(self):
        self.assertEqual(
            get_best_checkpoint_metric_names(("AG_ReID", "AG_ReID_G2A")),
            ("AG_ReID/metric", "AG_ReID/mAP"),
        )


if __name__ == "__main__":
    unittest.main()
