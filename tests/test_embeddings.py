import unittest
import numpy as np
import torch

from fedrepr.data import prepare_full_features, synthetic
from fedrepr.embedding_sweep import fit_candidates
from fedrepr.core import orthogonal
from fedrepr.strong_features import fit
from fedrepr.train_imagenet_vit import stratified_subset


class FullEmbeddingChecks(unittest.TestCase):
    def test_no_projection_and_training_only_center(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(8, 384))
        y = np.zeros((8, 2))
        heldout = rng.normal(size=(4, 384))
        labels = np.zeros((4, 2))
        clients, state = prepare_full_features(
            [(x, y, heldout, labels, heldout, labels)]
        )
        changed, _ = prepare_full_features(
            [(x, y, heldout + 100, labels, heldout + 100, labels)]
        )
        self.assertEqual(clients[0].x.shape, (8, 384))
        self.assertNotIn("projection", state)
        np.testing.assert_allclose(
            clients[0].x * state["scale"] + state["mean"], x, atol=1e-12
        )
        np.testing.assert_array_equal(clients[0].x, changed[0].x)

    def test_chunked_solver_matches_original(self):
        torch.set_num_threads(2)
        clients, *_ = synthetic(0, n=35, d=8, rank=4, samples=20, test_samples=30)
        # Multiclass-shaped targets allow accuracy-based validation in both solvers.
        for c in clients:
            for name in ("y", "vy", "ty"):
                scores = getattr(c, name)
                setattr(c, name, np.concatenate([scores, -scores], axis=1))
        basis = orthogonal(np.random.default_rng(0).normal(size=(8, 3)))
        gram = np.stack([c.x.T @ c.x / len(c.x) for c in clients])
        cross = np.stack([c.x.T @ c.y / len(c.x) for c in clients])
        reference = fit(clients, gram, cross, basis, 2, 0.03, 0.01)
        actual = fit_candidates(
            clients, basis, 2, 0.03, [0.01], torch.device("cpu"), 10
        )
        self.assertAlmostEqual(reference[0], actual["validation"])
        np.testing.assert_allclose(reference[2], actual["weights"], atol=1e-10)

    def test_subset_is_deterministic_stratified_and_nested(self):
        targets = np.repeat(np.arange(4), 10)
        small = stratified_subset(targets, 0.3, 3)
        large = stratified_subset(targets, 0.5, 3)
        np.testing.assert_array_equal(small, stratified_subset(targets, 0.3, 3))
        self.assertTrue(set(small) <= set(large))
        np.testing.assert_array_equal(np.bincount(targets[small]), np.full(4, 3))
        with self.assertRaises(ValueError):
            stratified_subset(targets, 0.0, 0)


if __name__ == "__main__":
    unittest.main()
