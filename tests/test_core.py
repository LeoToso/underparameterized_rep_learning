import unittest
import numpy as np
from fedrepr.core import fit_head, initialize, orthogonal, train
from fedrepr.data import synthetic, prepare_features


class TheoryChecks(unittest.TestCase):
    def test_preprocessing_does_not_fit_heldout_examples(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(30, 6))
        y = rng.normal(size=(30, 1))
        heldout = rng.normal(size=(10, 6))
        hy = np.zeros((10, 1))
        a = prepare_features([(x, y, heldout, hy, heldout, hy)], 3)
        b = prepare_features([(x, y, heldout * 1000, hy, heldout * 1000, hy)], 3)
        np.testing.assert_array_equal(a[0].x, b[0].x)

    def test_population_tail_identity(self):
        _, theta, spectrum, _ = synthetic(3)
        u = np.linalg.svd(theta, full_matrices=False)[0]
        for k in [1, 4, 8, 16]:
            b = u[:, :k]
            error = np.sum((theta - b @ b.T @ theta) ** 2) / theta.shape[1]
            self.assertAlmostEqual(error, spectrum[k:].sum(), places=10)

    def test_head_normal_equations(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 7))
        y = rng.normal(size=(50, 3))
        h = fit_head(x, y)
        np.testing.assert_allclose(x.T @ (x @ h - y), 0, atol=1e-10)

    def test_population_step_decreases_residual(self):
        _, theta, _, _ = synthetic(2)
        b = orthogonal(np.random.default_rng(2).normal(size=(40, 4)))
        a = theta @ theta.T / theta.shape[1]
        before = np.sum((theta - b @ b.T @ theta) ** 2)
        for _ in range(100):
            b = orthogonal(b + 0.3 * (a @ b - b @ (b.T @ a @ b)))
        self.assertLess(np.sum((theta - b @ b.T @ theta) ** 2), before * 0.5)

    def test_multiclass_scalar_moments_rejected(self):
        clients, _, _, _ = synthetic(0)
        clients[0].y = np.ones((80, 2))
        with self.assertRaises(ValueError):
            initialize(clients, 2, np.random.default_rng(0), "moments")

    def test_seed_and_orthogonality(self):
        clients, _, _, fresh = synthetic(0)
        a = train(
            clients, 4, "proposed", np.random.default_rng(3), rounds=3, fresh=fresh
        )
        b = train(
            clients, 4, "proposed", np.random.default_rng(3), rounds=3, fresh=fresh
        )
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1].T @ a[1], np.eye(4), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
