"""Run cross-moment FedRep with a frozen DINOv2 encoder and exact ridge heads."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import evaluate, initialize, orthogonal
from .data import images
from .download import checksum


class DinoFeatures:
    def __init__(self, hub, cache):
        import torch

        self.torch = torch
        torch.set_num_threads(6)
        self.model = torch.hub.load(
            str(hub / "facebookresearch_dinov2_main"),
            "dinov2_vits14",
            source="local",
            pretrained=False,
        ).eval()
        checkpoint = hub / "checkpoints/dinov2_vits14_pretrain.pth"
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.checkpoint_hash = checksum(checkpoint)
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, images):
        torch = self.torch
        images = np.ascontiguousarray(images)
        digest = hashlib.sha256(images.tobytes()).hexdigest()
        path = self.cache / f"{self.checkpoint_hash[:12]}_84_{digest}.npy"
        if path.exists():
            return np.load(path)
        result = []
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.inference_mode():
            for start in range(0, len(images), 128):
                x = torch.from_numpy(images[start : start + 128]).float() / 255
                if x.ndim == 3:
                    x = x[:, None].expand(-1, 3, -1, -1)
                else:
                    x = x.permute(0, 3, 1, 2)
                x = torch.nn.functional.interpolate(
                    x,
                    size=(84, 84),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp(0, 1)
                result.append(self.model((x - mean) / std).numpy())
                if len(images) > 1000 and start % 8192 == 0:
                    print(
                        f"Encoded {min(start + 128, len(images))}/{len(images)} images",
                        flush=True,
                    )
        features = np.concatenate(result)
        np.save(path, features)
        return features


def fit(clients, gram, cross, basis, rounds, lr, ridge):
    """Use fixed-data sufficient statistics for the same full-batch updates."""
    eye = np.eye(basis.shape[1])
    best = None
    for step in range(rounds + 1):
        gram_basis = gram @ basis
        head = np.linalg.solve(basis.T @ gram_basis + ridge * eye, basis.T @ cross)
        if step % 10 == 0 or step == rounds:
            weights = basis @ head
            score = evaluate(clients, weights, "val", True)["macro"]
            if best is None or score > best[0]:
                best = (score, step, weights.copy(), basis.copy())
        if step < rounds:
            gradient = ((gram_basis @ head - cross) @ head.swapaxes(-1, -2)).mean(0)
            basis = orthogonal(basis - lr * gradient)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cifar10", "femnist"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--hub", type=Path, default=Path.home() / ".cache/torch/hub")
    p.add_argument("--clients", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--dimensions", nargs="+", type=int, default=[8, 16, 24, 32, 48, 64])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    encoder = DinoFeatures(args.hub, args.cache)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        encoder="DINOv2 ViT-S/14 at 84px",
        checkpoint_sha256=encoder.checkpoint_hash,
        feature_dim=64,
        validation_interval=10,
        cifar_validation_fraction=0.10,
        learning_rate=0.03,
        ridge_grid=[0.001, 0.01],
        alpha=0.3,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for seed in args.seeds:
        clients, ids, transform = images(
            args.dataset,
            args.data,
            seed,
            n=args.clients,
            samples=50000,
            feature_dim=64,
            encoder=encoder,
            cifar_validation_fraction=0.10,
        )
        sizes = [[len(c.x), len(c.vx), len(c.tx)] for c in clients]
        (args.output / f"partition_{seed}.json").write_text(
            json.dumps({"ids": ids, "sizes": sizes})
        )
        np.savez_compressed(args.output / f"preprocessing_{seed}.npz", **transform)
        gram = np.stack([c.x.T @ c.x / len(c.x) for c in clients])
        cross = np.stack([c.x.T @ c.y / len(c.x) for c in clients])
        for k in args.dimensions:
            basis = initialize(clients, k, np.random.default_rng(seed), "cross_moments")
            candidates = [
                (fit(clients, gram, cross, basis, args.rounds, 0.03, ridge), ridge)
                for ridge in (0.001, 0.01)
            ]
            (score, step, weights, best_basis), ridge = max(
                candidates, key=lambda entry: entry[0][0]
            )
            test = evaluate(clients, weights, "test", True)
            row = dict(
                dataset=args.dataset,
                seed=seed,
                method="proposed",
                k=k,
                d=64,
                clients=len(clients),
                train_examples=sum(s[0] for s in sizes),
                rounds=args.rounds,
                selected_round=step,
                ridge=ridge,
                validation=score,
                test_macro=test["macro"],
                test_micro=test["micro"],
                test_p10=test["p10"],
            )
            rows.append(row)
            with (args.output / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row)
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(
                args.output / f"model_{seed}_proposed_{k}.npz",
                weights=weights,
                basis=best_basis,
            )
            print(
                f"seed={seed} k={k} round={step} accuracy={test['macro']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
