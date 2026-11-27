"""Plot measured initialization, fit and evaluation time against k for each encoder.

The seconds field includes both ridge candidates, sufficient-statistic
construction, validation and test evaluation. Feature extraction, preprocessing,
and model-file writing occur outside this timer. The full-dimensional point
uses a direct ridge fit and is marked separately.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_embedding_comparison import summarize


def read_results(folder):
    with (folder / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    config = json.loads((folder / "config.json").read_text())
    expected = {
        (data_seed, seed, k)
        for data_seed in config["data_seeds"]
        for seed in config["init_seeds"]
        for k in config["dimensions"]
    }
    keys = [(int(row["data_seed"]), int(row["seed"]), int(row["k"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Incomplete or duplicated result points in {folder}")
    for row in rows:
        duration = float(row["seconds"])
        if not np.isfinite(duration) or duration < 0:
            raise ValueError(f"Invalid elapsed time in {folder}")
    return rows, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--encoders", nargs="+", default=["dinov2", "ibot", "dinov3", "imagenet-vit"]
    )
    parser.add_argument("--unit", choices=["seconds", "minutes"], default="seconds")
    args = parser.parse_args()
    scale = 1.0 if args.unit == "seconds" else 1.0 / 60
    destination = args.results / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for encoder in args.encoders:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, dataset, title in zip(
            axes, ("cifar10", "femnist"), ("CIFAR", "FEMNIST")
        ):
            folder = args.results / f"{encoder}_{dataset}"
            rows, config = read_results(folder)
            ks = np.array(sorted({int(row["k"]) for row in rows}))
            mean, half = [], []
            full = []
            for k in ks:
                value, width, count, kind = summarize(
                    rows, k, field="seconds", scale=scale
                )
                mean.append(value)
                half.append(width)
                is_full = all(int(row["d"]) == k for row in rows if int(row["k"]) == k)
                full.append(is_full)
                summary.append(
                    dict(
                        encoder=encoder,
                        dataset=dataset,
                        k=int(k),
                        mean_time=value,
                        ci95_lower=value - width,
                        ci95_upper=value + width,
                        unit=args.unit,
                        independent_units=count,
                        interval_type=kind,
                        optimization_rounds=0 if is_full else config["rounds"],
                        fit_mode="direct ridge" if is_full else "alternating",
                        ridge_candidates=len(config["ridges"]),
                    )
                )
            mean, half = np.array(mean), np.array(half)
            iterative = ~np.array(full)
            ax.plot(
                ks[iterative],
                mean[iterative],
                "o-",
                color="#2563eb",
                markersize=4,
                linewidth=2,
            )
            ax.fill_between(
                ks[iterative],
                mean[iterative] - half[iterative],
                mean[iterative] + half[iterative],
                color="#2563eb",
                alpha=0.2,
            )
            if any(full):
                full = np.array(full)
                ax.errorbar(
                    ks[full],
                    mean[full],
                    yerr=half[full],
                    fmt="*",
                    markersize=11,
                    capsize=4,
                    color="#475569",
                    label="Full dimension (direct ridge fit)",
                )
                ax.legend(frameon=False, fontsize=8)
            ax.set(
                title=title,
                xlabel="Representation dimension k",
                ylabel=f"Initialization, fitting and evaluation time ({args.unit})",
            )
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
        fig.suptitle(
            {
                "dinov2": "DINOv2",
                "ibot": "iBOT",
                "dinov3": "DINOv3",
                "imagenet-vit": "ImageNet-trained ViT",
            }.get(encoder, encoder)
        )
        for extension in ("png", "pdf"):
            path = destination / f"{encoder}_computation_time.{extension}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            print(path)
        plt.close(fig)
    with (destination / "computation_time_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
