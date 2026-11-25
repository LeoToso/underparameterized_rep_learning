"""Run the four frozen-encoder comparisons without PCA, then plot the results."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from fedrepr.embedding_sweep import DIMENSIONS


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--encoders",
        nargs="+",
        choices=["dinov2", "ibot", "dinov3", "imagenet-vit"],
        default=["dinov2", "ibot", "dinov3", "imagenet-vit"],
    )
    p.add_argument("--data", type=Path, default=root / "data")
    p.add_argument("--output", type=Path, default=root / "embedding-runs")
    p.add_argument("--cache", type=Path, default=root / "embedding-cache")
    p.add_argument("--dinov2-checkpoint", type=Path)
    p.add_argument("--ibot-checkpoint", type=Path)
    p.add_argument("--dinov3-checkpoint", type=Path)
    p.add_argument("--dinov3-repo", type=Path)
    p.add_argument("--imagenet-checkpoint", type=Path)
    p.add_argument("--imagenet-root", type=Path)
    p.add_argument("--data-fraction", type=float, default=0.30)
    p.add_argument("--imagenet-epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--dimensions", nargs="+", type=int, default=DIMENSIONS)
    p.add_argument("--data-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--init-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--cifar-clients", type=int, default=30)
    p.add_argument("--femnist-clients", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 0 < args.data_fraction <= 1:
        p.error("data-fraction must lie in (0,1]")
    if args.rounds < 0 or args.batch_size < 1 or args.imagenet_epochs < 1:
        p.error("Use nonnegative rounds and positive batch size and ImageNet epochs")
    if any(k < 1 or k > 384 for k in args.dimensions):
        p.error("Representation dimensions must lie between 1 and 384")
    if "dinov3" in args.encoders and (
        args.dinov3_checkpoint is None or args.dinov3_repo is None
    ):
        p.error(
            "For DINOv3 supply --dinov3-checkpoint and --dinov3-repo, or select the available encoders"
        )
    if (
        "imagenet-vit" in args.encoders
        and args.imagenet_checkpoint is None
        and args.imagenet_root is None
    ):
        p.error("For the custom ViT supply --imagenet-root or --imagenet-checkpoint")
    output = args.output.resolve()
    cache = args.cache.resolve()
    data = args.data.resolve()
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))

    def run(arguments):
        command = [sys.executable, *map(str, arguments)]
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, env=environment, check=True)

    checkpoints = {
        "dinov2": args.dinov2_checkpoint,
        "ibot": args.ibot_checkpoint,
        "dinov3": args.dinov3_checkpoint,
        "imagenet-vit": args.imagenet_checkpoint,
    }
    # Resolve optional paths before changing working directory in child processes.
    checkpoints = {
        name: path.expanduser().resolve() if path else None
        for name, path in checkpoints.items()
    }
    if not args.dry_run:
        for encoder in args.encoders:
            checkpoint = checkpoints[encoder]
            if checkpoint and not checkpoint.is_file():
                p.error(f"Checkpoint does not exist: {checkpoint}")
        if "dinov3" in args.encoders:
            if not (args.dinov3_repo.expanduser().resolve() / "hubconf.py").is_file():
                p.error("--dinov3-repo must point to a cloned DINOv3 repository")
        if "imagenet-vit" in args.encoders and checkpoints["imagenet-vit"] is None:
            imagenet = args.imagenet_root.expanduser().resolve()
            if not all((imagenet / split).is_dir() for split in ("train", "val")):
                p.error("--imagenet-root must contain train/ and val/ directories")
    planned = [
        output / f"{encoder}_{dataset}"
        for encoder in args.encoders
        for dataset in ("cifar10", "femnist")
    ]
    if not args.dry_run and any(path.exists() for path in planned):
        p.error("An experiment directory already exists; choose a fresh --output")
    for dataset in ("cifar10", "femnist"):
        run(["-m", "fedrepr.download", dataset, "--root", data])
    if "imagenet-vit" in args.encoders and checkpoints["imagenet-vit"] is None:
        pretrain = output / "imagenet-pretrain"
        run(
            [
                "-m",
                "fedrepr.train_imagenet_vit",
                "--imagenet-root",
                args.imagenet_root.expanduser().resolve(),
                "--output",
                pretrain,
                "--data-fraction",
                args.data_fraction,
                "--epochs",
                args.imagenet_epochs,
                "--batch-size",
                args.batch_size,
                "--device",
                args.device,
            ]
        )
        checkpoints["imagenet-vit"] = pretrain / "best.pt"
    for encoder in args.encoders:
        for dataset, clients in (
            ("cifar10", args.cifar_clients),
            ("femnist", args.femnist_clients),
        ):
            command = [
                "-m",
                "fedrepr.embedding_sweep",
                "--encoder",
                encoder,
                "--dataset",
                dataset,
                "--data",
                data / dataset,
                "--output",
                output / f"{encoder}_{dataset}",
                "--cache",
                cache,
                "--clients",
                clients,
                "--rounds",
                args.rounds,
                "--dimensions",
                *args.dimensions,
                "--data-seeds",
                *args.data_seeds,
                "--init-seeds",
                *args.init_seeds,
                "--device",
                args.device,
                "--batch-size",
                args.batch_size,
            ]
            if checkpoints[encoder]:
                command += ["--checkpoint", checkpoints[encoder]]
            if encoder == "dinov3":
                command += ["--dinov3-repo", args.dinov3_repo.expanduser().resolve()]
            run(command)
    run(
        [
            root / "plot_embedding_comparison.py",
            "--results",
            output,
            "--encoders",
            *args.encoders,
        ]
    )


if __name__ == "__main__":
    main()
