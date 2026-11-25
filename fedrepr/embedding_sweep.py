"""Run no-PCA, bottleneck-only FedRep on full 384-dimensional image embeddings."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .core import Client, evaluate, initialize, orthogonal
from .data import images
from .encoders import FrozenEncoder, resolve_device

DIMENSIONS = [2, 4, 6, 8, 10, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192, 256, 384]


def fit_candidates(clients, basis, rounds, lr, ridges, device, validation_every):
    """Exact ridge heads and averaged gradients from fixed training statistics.

    Statistics stay on CPU. Client chunks transfer to the selected device.
    Every round accumulates all clients before updating the common basis.
    """
    gram = np.stack([c.x.T @ c.x / len(c.x) for c in clients])
    cross = np.stack([c.x.T @ c.y / len(c.x) for c in clients])
    best = None
    dtype = torch.float64
    tensor = lambda value: torch.as_tensor(value, dtype=dtype, device=device)
    with torch.no_grad():
        for ridge in ridges:
            b = tensor(basis.copy())
            identity = torch.eye(b.shape[1], dtype=dtype, device=device)
            effective_rounds = 0 if b.shape[0] == b.shape[1] else rounds
            for step in range(effective_rounds + 1):
                gradient = torch.zeros_like(b)
                weights = []
                validate = step % validation_every == 0 or step == effective_rounds
                for start in range(0, len(clients), 32):
                    g = tensor(gram[start : start + 32])
                    z = tensor(cross[start : start + 32])
                    gb = g @ b
                    head = torch.linalg.solve(b.T @ gb + ridge * identity, b.T @ z)
                    if validate:
                        weights.extend((b @ head).cpu().numpy())
                    gradient += ((gb @ head - z) @ head.transpose(-1, -2)).sum(0)
                if validate:
                    score = evaluate(clients, weights, "val", True)["macro"]
                    if best is None or score > best["validation"]:
                        best = dict(
                            validation=score,
                            selected_round=step,
                            ridge=ridge,
                            weights=np.stack(weights),
                            basis=b.cpu().numpy().copy(),
                        )
                if step < effective_rounds:
                    b = torch.linalg.qr(
                        b - lr * gradient / len(clients), mode="reduced"
                    )[0]
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--encoder", required=True, choices=["dinov2", "ibot", "dinov3", "imagenet-vit"]
    )
    p.add_argument("--dataset", required=True, choices=["cifar10", "femnist"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("embedding-cache"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dinov3-repo", type=Path)
    p.add_argument("--clients", type=int)
    p.add_argument("--data-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--init-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--dimensions", nargs="+", type=int, default=DIMENSIONS)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--ridges", nargs="+", type=float, default=[0.001, 0.01])
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--validation-every", type=int, default=10)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if not all(1 <= k <= 384 for k in args.dimensions):
        p.error("Every representation dimension must be between 1 and 384")
    if (
        min(args.ridges) <= 0
        or args.rounds < 0
        or args.validation_every < 1
        or args.batch_size < 1
    ):
        p.error(
            "Use positive ridges, batch size and validation interval and nonnegative rounds"
        )
    if len(set(args.dimensions)) != len(args.dimensions):
        p.error("Duplicate dimensions are not allowed")
    if len(set(args.data_seeds)) != len(args.data_seeds) or len(
        set(args.init_seeds)
    ) != len(args.init_seeds):
        p.error("Duplicate seeds are not allowed")
    if args.clients is None:
        args.clients = 30 if args.dataset == "cifar10" else 1000
    if args.clients < 2:
        p.error("At least two clients are required")
    if args.output.exists():
        p.error("Choose a new output directory")
    encoder = FrozenEncoder(
        args.encoder,
        args.cache,
        args.checkpoint,
        args.dinov3_repo,
        args.device,
        args.batch_size,
        args.image_size,
    )
    args.output.mkdir(parents=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        pca=False,
        encoder_metadata=encoder.metadata,
        preprocessing="training mean and one scalar RMS; all coordinates retained",
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for data_seed in args.data_seeds:
        # Legacy feature_dim=64 fixes eligibility at 128 samples per writer.
        # It does not reduce features when use_pca=False.
        clients, ids, transform = images(
            args.dataset,
            args.data,
            data_seed,
            n=args.clients,
            samples=50000,
            feature_dim=64,
            encoder=encoder,
            use_pca=False,
            cifar_validation_fraction=0.10,
        )
        if clients[0].x.shape[1] != 384:
            raise ValueError("Embedding dimension changed unexpectedly")
        manifest = dict(
            ids=ids, sizes=[[len(c.x), len(c.vx), len(c.tx)] for c in clients]
        )
        (args.output / f"partition_{data_seed}.json").write_text(json.dumps(manifest))
        np.savez_compressed(args.output / f"preprocessing_{data_seed}.npz", **transform)
        for init_seed in args.init_seeds:
            generator = np.random.default_rng(init_seed)
            shuffled = []
            for c in clients:
                order = generator.permutation(len(c.x))
                shuffled.append(Client(c.x[order], c.y[order], c.vx, c.vy, c.tx, c.ty))
            for k in args.dimensions:
                start = time.monotonic()
                basis = initialize(shuffled, k, generator, "cross_moments")
                best = fit_candidates(
                    clients,
                    basis,
                    args.rounds,
                    args.learning_rate,
                    args.ridges,
                    resolve_device(args.device),
                    args.validation_every,
                )
                test = evaluate(clients, best["weights"], "test", True)
                row = dict(
                    encoder=args.encoder,
                    dataset=args.dataset,
                    data_seed=data_seed,
                    seed=init_seed,
                    method="proposed",
                    k=k,
                    d=384,
                    clients=len(clients),
                    train_examples=sum(len(c.x) for c in clients),
                    validation=best["validation"],
                    selected_round=best["selected_round"],
                    ridge=best["ridge"],
                    test_macro=test["macro"],
                    test_micro=test["micro"],
                    test_p10=test["p10"],
                    seconds=time.monotonic() - start,
                )
                rows.append(row)
                np.savez_compressed(
                    args.output / f"model_{data_seed}_{init_seed}_{k}.npz",
                    basis=best["basis"],
                    weights=best["weights"],
                )
                with (args.output / "metrics.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    f"{args.encoder} {args.dataset} data={data_seed} init={init_seed} k={k}: {test['macro']:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
