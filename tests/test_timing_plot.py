import unittest

from plot_embedding_comparison import summarize


class TimingSummaryChecks(unittest.TestCase):
    def test_seconds_are_not_percentage_scaled(self):
        rows = [
            dict(k="8", data_seed="0", seed=str(i), seconds=str(value), test_macro=".9")
            for i, value in enumerate((2.0, 4.0, 6.0))
        ]
        mean, half, count, _ = summarize(rows, 8, field="seconds", scale=1.0)
        self.assertEqual(mean, 4.0)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(half, 4.3026527297 * 2 / (3**0.5), places=8)

    def test_nested_seeds_are_averaged_within_partition(self):
        rows = [
            dict(k="8", data_seed=str(data_seed), seed=str(seed), seconds=str(value))
            for data_seed, values in [(0, (2.0, 4.0)), (1, (8.0, 10.0))]
            for seed, value in enumerate(values)
        ]
        mean, _, count, kind = summarize(rows, 8, field="seconds", scale=1 / 60)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(mean, 6 / 60)
        self.assertIn("data partitions", kind)


if __name__ == "__main__":
    unittest.main()
