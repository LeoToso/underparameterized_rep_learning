"""Compare full-embedding FedRep curves with pointwise Student t intervals."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def summarize(rows, k, field="test_macro", scale=100.0):
    selected = [r for r in rows if int(r["k"]) == k]
    data_seeds = sorted({int(r["data_seed"]) for r in selected})
    if len(data_seeds) > 1:
        # Treat dataset partitions as independent units, averaging nested initializations.
        values = [
            np.mean([float(r[field]) for r in selected if int(r["data_seed"]) == seed])
            for seed in data_seeds
        ]
        interval_type = (
            "data partitions (initializations averaged within each partition)"
        )
    else:
        values = [float(r[field]) for r in selected]
        interval_type = (
            "initialization conditional on one data partition and encoder checkpoint"
        )
    values = scale * np.asarray(values)
    half = (
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return float(values.mean()), float(half), len(values), interval_type


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--encoders", nargs="+", default=["dinov2", "ibot", "dinov3", "imagenet-vit"]
    )
    args = p.parse_args()
    fig, axes = plt.subplots(
        len(args.encoders),
        2,
        figsize=(11, 3.5 * len(args.encoders)),
        squeeze=False,
        layout="constrained",
    )
    summary = []
    for i, encoder in enumerate(args.encoders):
        for j, dataset in enumerate(("cifar10", "femnist")):
            folder = args.results / f"{encoder}_{dataset}"
            with (folder / "metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            config = json.loads((folder / "config.json").read_text())
            expected = {
                (data_seed, seed, k)
                for data_seed in config["data_seeds"]
                for seed in config["init_seeds"]
                for k in config["dimensions"]
            }
            keys = [(int(r["data_seed"]), int(r["seed"]), int(r["k"])) for r in rows]
            if len(keys) != len(set(keys)) or set(keys) != expected:
                raise ValueError(
                    f"Incomplete or duplicated experiment rows in {folder}"
                )
            ks = sorted({int(r["k"]) for r in rows})
            mean, half = [], []
            for k in ks:
                value, width, count, kind = summarize(rows, k)
                mean.append(value)
                half.append(width)
                summary.append(
                    dict(
                        encoder=encoder,
                        dataset=dataset,
                        k=k,
                        mean_percent=value,
                        ci95_lower=value - width,
                        ci95_upper=value + width,
                        independent_units=count,
                        interval_type=kind,
                    )
                )
            mean, half = np.asarray(mean), np.asarray(half)
            ax = axes[i, j]
            ax.plot(ks, mean, "o-", color="#2563eb", markersize=3)
            ax.fill_between(ks, mean - half, mean + half, color="#2563eb", alpha=0.2)
            ax.set(
                xlabel="Representation dimension k",
                ylabel=f"{encoder}\nMean client test accuracy (%)",
            )
            if i == 0:
                ax.set_title("CIFAR" if j == 0 else "FEMNIST")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
    destination = args.results / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            destination / f"encoder_comparison.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    with (destination / "confidence_intervals.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plt.close(fig)


if __name__ == "__main__":
    main()
