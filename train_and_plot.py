"""Train both bottleneck experiments and render the two-panel figure."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

DIMENSIONS = ["2", "4", "6", "8", "10", "16", "24", "32", "40", "48", "56", "64"]
DINO_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "runs")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--use-saved-cnn", action="store_true",
                        help="Reuse the included FEMNIST encoder checkpoint")
    parser.add_argument("--both-ci", action="store_true",
                        help="Draw pointwise 95%% t intervals in both panels")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without downloading or training")
    args = parser.parse_args()
    output = args.output.resolve()
    data = args.data.resolve()
    environment = os.environ.copy()
    environment.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    environment.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))

    def run(arguments):
        command = [sys.executable, *map(str, arguments)]
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, env=environment, check=True)

    for dataset in ("cifar10", "femnist"):
        run(["-m", "fedrepr.download", dataset, "--root", data])

    checkpoint = root / "vendor/checkpoints/dinov2_vits14_pretrain.pth"
    if not args.dry_run:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            partial = checkpoint.with_suffix(".partial")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
                partial,
            )
            with partial.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != DINO_SHA256:
                partial.unlink()
                raise ValueError("Downloaded DINOv2 checkpoint failed SHA-256 verification")
            partial.replace(checkpoint)
        with checkpoint.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != DINO_SHA256:
                raise ValueError("DINOv2 checkpoint failed SHA-256 verification")

    cifar = output / "cifar"
    femnist = output / "femnist"
    if not args.dry_run:
        if (cifar / "metrics.csv").exists() or femnist.exists():
            raise FileExistsError("Choose a fresh --output directory for a training run")

    run(["-m", "fedrepr.strong_features",
         "--dataset", "cifar10", "--data", data / "cifar10",
         "--output", cifar, "--cache", output / "feature-cache",
         "--hub", root / "vendor", "--clients", "30", "--rounds", "500",
         "--seeds", "0", "1", "2", "--dimensions", *DIMENSIONS])
    training = [
        "-m", "fedrepr.handwriting",
        "--data", data / "femnist", "--output", femnist,
        "--clients", "1000", "--epochs", "30", "--rounds", "500",
        "--seed", "0", "--init-seeds", "0", "1", "2",
        "--dimensions", *DIMENSIONS,
    ]
    if args.use_saved_cnn:
        training += ["--resume", root / "checkpoints/femnist_cnn"]
    run(training)
    plot = [
        root / "plot_cifar_femnist.py",
        "--cifar", cifar / "metrics.csv",
        "--femnist", femnist / "metrics.csv",
        "--output", output / "figures/cifar_femnist",
    ]
    if args.both_ci:
        plot.append("--both-ci")
    run(plot)


if __name__ == "__main__":
    main()
