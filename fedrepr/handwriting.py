"""Train a handwriting encoder on pooled training splits, then freeze it for FedRep."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import Client, evaluate, initialize
from .data import images
from .strong_features import fit


def encoder_model():
    return nn.Sequential(
        nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.Conv2d(24, 48, 3, padding=1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 3 * 3, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
    )


def pretrain(raw, epochs, seed, output, resume=None):
    torch.manual_seed(seed)
    model = nn.Sequential(encoder_model(), nn.Linear(128, 62)).to(
        memory_format=torch.channels_last
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    train_x = torch.tensor(
        np.concatenate([c[0] for c in raw]), dtype=torch.float32
    ).reshape(-1, 1, 28, 28)
    train_y = torch.tensor(
        np.concatenate([c[1].argmax(1) for c in raw]), dtype=torch.long
    )
    val_x = torch.tensor(
        np.concatenate([c[2] for c in raw]), dtype=torch.float32
    ).reshape(-1, 1, 28, 28)
    val_y = torch.tensor(
        np.concatenate([c[3].argmax(1) for c in raw]), dtype=torch.long
    )
    best = -1.0
    history = []
    if resume is not None:
        history = json.loads((resume / "cnn_history.json").read_text())
        best = max(row["pooled_validation_accuracy"] for row in history)
        model.load_state_dict(torch.load(resume / "cnn.pt", weights_only=True))
        torch.save(model.state_dict(), output / "cnn.pt")
    for epoch in range(len(history), epochs):
        model.train()
        for group in torch.randperm(len(train_x)).split(256):
            if len(group) < 2:
                continue
            x = train_x[group].contiguous(memory_format=torch.channels_last)
            x = torch.roll(
                x, shifts=tuple(torch.randint(-1, 2, (2,)).tolist()), dims=(2, 3)
            )
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), train_y[group])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([model(x).argmax(1) for x in val_x.split(512)])
            score = (predictions == val_y).float().mean().item()
        history.append({"epoch": epoch + 1, "pooled_validation_accuracy": score})
        print(f"CNN epoch {epoch + 1}/{epochs}: validation={score:.4f}", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), output / "cnn.pt")
        if epoch + 1 in (epochs // 2, 3 * epochs // 4):
            for group in optimizer.param_groups:
                group["lr"] *= 0.35
        (output / "cnn_history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(output / "cnn.pt", weights_only=True))
    model.eval()

    def embed(x):
        tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 28, 28)
        with torch.inference_mode():
            return torch.cat([model[0](batch) for batch in tensor.split(512)]).numpy()

    return [(embed(x), y, embed(vx), vy, embed(tx), ty) for x, y, vx, vy, tx, ty in raw]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-seeds", nargs="+", type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    clients, ids, transform = images(
        "femnist",
        args.data,
        args.seed,
        n=args.clients,
        samples=50000,
        feature_dim=64,
        encoder=lambda x: x.reshape(len(x), -1) / 255,
        raw_transform=lambda raw: pretrain(
            raw, args.epochs, args.seed, args.output, args.resume
        ),
    )
    sizes = [[len(c.x), len(c.vx), len(c.tx)] for c in clients]
    (args.output / "partition.json").write_text(
        json.dumps({"ids": ids, "sizes": sizes})
    )
    np.savez_compressed(args.output / "preprocessing.npz", **transform)
    gram = np.stack([c.x.T @ c.x / len(c.x) for c in clients])
    cross = np.stack([c.x.T @ c.y / len(c.x) for c in clients])
    rows = []
    for seed in args.init_seeds or [args.seed]:
        rng = np.random.default_rng(seed)
        init_clients = clients
        if args.init_seeds:
            init_clients = []
            for c in clients:
                order = rng.permutation(len(c.x))
                init_clients.append(
                    Client(c.x[order], c.y[order], c.vx, c.vy, c.tx, c.ty)
                )
        for k in args.dimensions:
            basis = initialize(init_clients, k, rng, "cross_moments")
            # Orthogonal changes of basis preserve full-dimensional ridge predictions.
            effective_rounds = 0 if k == basis.shape[0] else args.rounds
            candidates = [
                (fit(clients, gram, cross, basis, effective_rounds, 0.03, ridge), ridge)
                for ridge in (0.001, 0.01)
            ]
            (val, step, weights, basis), ridge = max(
                candidates, key=lambda entry: entry[0][0]
            )
            metrics = evaluate(clients, weights, "test", True)
            rows.append(
                dict(
                    dataset="femnist",
                    method="proposed",
                    seed=seed,
                    k=k,
                    rounds=effective_rounds,
                    selected_round=step,
                    ridge=ridge,
                    validation=val,
                    test_macro=metrics["macro"],
                    test_micro=metrics["micro"],
                    test_p10=metrics["p10"],
                )
            )
            with (args.output / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(
                args.output
                / (f"model_{seed}_{k}.npz" if args.init_seeds else f"model_{k}.npz"),
                basis=basis,
                weights=weights,
            )
            print(
                f"FedRep seed={seed} k={k}: test macro accuracy={metrics['macro']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
