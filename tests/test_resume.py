import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fedrepr.embedding_sweep import load_progress, write_metrics


class ResumeChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(
            encoder="dinov2",
            dataset="cifar10",
            data="data",
            checkpoint=None,
            dinov3_repo=None,
            clients=2,
            data_seeds=[0],
            init_seeds=[0],
            dimensions=[2, 4],
            rounds=1,
            ridges=[0.01],
            learning_rate=0.03,
            validation_every=10,
            image_size=224,
            pca=False,
        )
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.row = dict(data_seed=0, seed=0, k=2, test_macro=0.7)
        np.savez_compressed(
            self.root / "model_0_0_2.npz",
            basis=np.zeros((4, 2)),
            weights=np.zeros((2, 4, 2)),
        )
        write_metrics(self.root, [self.row])

    def test_old_run_loads_completed_points(self):
        rows, done = load_progress(self.root, self.config)
        self.assertEqual(done, {(0, 0, 2)})
        self.assertEqual(float(rows[0]["test_macro"]), 0.7)

    def test_changed_settings_and_missing_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, "rounds"):
            load_progress(self.root, {**self.config, "rounds": 10})
        (self.root / "model_0_0_2.npz").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            load_progress(self.root, self.config)

    def test_failed_metrics_write_preserves_previous_results(self):
        before = (self.root / "metrics.csv").read_bytes()
        with patch.object(
            csv.DictWriter, "writerows", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                write_metrics(self.root, [self.row])
        self.assertEqual((self.root / "metrics.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
