"""Reproduce the two-panel CIFAR/FEMNIST figure from the saved metrics.

Default bands reproduce the original figure:
    CIFAR: mean +/- one sample standard deviation across three data seeds.
    FEMNIST: pointwise 95% Student t intervals across three initialization seeds.

Use --both-ci to draw 95% Student t intervals in both panels. CIFAR seeds vary
the client/data partition, whereas FEMNIST seeds vary cross-moment initialization
with the CNN and writer splits fixed.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIMENSIONS = [2, 4, 6, 8, 10, 16, 24, 32, 40, 48, 56, 64]
T_95_DF2 = 4.3026527297


def summarize(path):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == "proposed"]
    measurements = {}
    for row in rows:
        key = (int(row["k"]), int(row["seed"]))
        if key in measurements:
            raise ValueError(f"Duplicate dimension/seed pair {key} in {path}")
        measurements[key] = 100 * float(row["test_macro"])
    expected = {(k, seed) for k in DIMENSIONS for seed in (0, 1, 2)}
    if set(measurements) != expected:
        raise ValueError(f"Expected all 12 dimensions and seeds 0, 1, 2 in {path}")
    values = np.array([
        [measurements[k, seed] for seed in (0, 1, 2)]
        for k in DIMENSIONS
    ])
    return values.mean(axis=1), values.std(axis=1, ddof=1)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cifar", type=Path,
        default=root / "results/cifar10_dino500_full_grid/metrics.csv",
    )
    parser.add_argument(
        "--femnist", type=Path,
        default=root / "results/femnist_cnn1000_init3/metrics.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "figures/cifar_femnist_bottleneck",
        help="Output path without extension",
    )
    parser.add_argument("--both-ci", action="store_true")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, title, path in zip(
        axes, ("CIFAR", "FEMNIST"), (args.cifar, args.femnist)
    ):
        mean, deviation = summarize(path)
        if title == "FEMNIST" or args.both_ci:
            deviation *= T_95_DF2 / np.sqrt(3)
        axis.fill_between(
            DIMENSIONS, mean - deviation, mean + deviation,
            color="#2563eb", alpha=0.2,
        )
        axis.plot(
            DIMENSIONS, mean, "o-", color="#2563eb",
            markersize=4, linewidth=2,
        )
        axis.set(
            title=title,
            xlabel="Representation dimension $k$",
            ylabel="Mean client test accuracy (%)",
            xticks=DIMENSIONS,
        )
        axis.tick_params(axis="x", labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        destination = Path(str(args.output) + "." + extension)
        fig.savefig(destination, dpi=220)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
