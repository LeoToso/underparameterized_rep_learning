"""Deterministic partitions and training-only feature preprocessing."""

import json
from pathlib import Path
import numpy as np
from .core import Client


def synthetic(
    seed, n=30, d=40, rank=16, samples=80, decay=0.45, noise=0.3, test_samples=500
):
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.normal(size=(d, rank)))[0]
    v = np.linalg.qr(rng.normal(size=(n, rank)))[0]
    eigenvalues = np.exp(-decay * np.arange(rank))
    theta = np.sqrt(n) * u @ np.diag(np.sqrt(eigenvalues)) @ v.T

    def draw(i, count, generator):
        x = generator.normal(size=(count, d))
        y = x @ theta[:, i : i + 1] + noise * generator.normal(size=(count, 1))
        return x, y

    clients = [
        Client(
            *draw(i, samples, rng),
            *draw(i, test_samples, rng),
            *draw(i, test_samples, rng)
        )
        for i in range(n)
    ]

    def fresh(t):
        generator = np.random.default_rng(np.random.SeedSequence([seed, t, 9281]))
        return [draw(i, samples, generator) for i in range(n)]

    return clients, theta, eigenvalues, fresh


def one_hot(y, classes):
    return np.eye(classes)[np.asarray(y, dtype=int)]


def prepare_features(raw, feature_dim, return_transform=False):
    """Fit centering and pooled PCA on training examples only; no label use.

    Unwhitened PCA avoids making the frozen PCA baseline arbitrary. Image results
    are an empirical extension, not a claim of isotropic Gaussian theory.
    """
    train = np.concatenate([c[0] for c in raw])
    mean = train.mean(0)
    cov = (train - mean).T @ (train - mean) / len(train)
    values, vectors = np.linalg.eigh(cov)
    keep = min(feature_dim, int(np.sum(values > 1e-10)))
    if keep < 1:
        raise ValueError("Training images have no nonconstant feature directions")
    projection = vectors[:, -keep:]
    scale = np.sqrt(np.mean(values[-keep:]))

    def transform(x):
        return (x - mean) @ projection / max(scale, 1e-12)

    clients = [
        Client(transform(a), b, transform(c), d, transform(e), f)
        for a, b, c, d, e, f in raw
    ]
    if return_transform:
        return clients, {"mean": mean, "projection": projection, "scale": scale}
    return clients


def prepare_full_features(raw):
    """Center and scalar-scale all embedding coordinates using training examples.

    This affine preprocessing retains the encoder's full output dimensionality.
    """
    training = np.concatenate([client[0] for client in raw])
    mean = training.mean(axis=0)
    scale = max(float(np.sqrt(np.mean((training - mean) ** 2))), 1e-12)

    def transform(features):
        return (features - mean) / scale

    clients = [
        Client(transform(x), y, transform(vx), vy, transform(tx), ty)
        for x, y, vx, vy, tx, ty in raw
    ]
    return clients, {"mean": mean, "scale": scale}


def split_client(x, labels, classes, rng, cap):
    indices = rng.permutation(len(x))
    nt = max(1, int(0.2 * len(x)))
    nv = max(1, int(0.2 * len(x)))
    test, val, train = indices[:nt], indices[nt : nt + nv], indices[nt + nv :]
    train = train[:cap]
    return (
        x[train],
        one_hot(labels[train], classes),
        x[val],
        one_hot(labels[val], classes),
        x[test],
        one_hot(labels[test], classes),
    )


def images(
    dataset,
    root,
    seed,
    n=30,
    samples=80,
    feature_dim=64,
    alpha=0.3,
    encoder=None,
    cifar_validation_fraction=0.2,
    raw_transform=None,
    use_pca=True,
):
    rng = np.random.default_rng(seed)
    raw, client_ids = [], []
    if dataset == "cifar10":
        from PIL import Image

        # Fixed spatial features shared by every method; no hidden pretrained model.
        def features(array):
            return np.stack(
                [
                    np.asarray(Image.fromarray(a).resize((8, 8)), dtype=float).ravel()
                    / 255
                    for a in array
                ]
            )

        if encoder is not None:
            features = encoder

        if (Path(root) / "train.parquet").exists():
            import pyarrow.parquet as pq
            from io import BytesIO

            def read(name):
                obj = pq.read_table(Path(root) / name).to_pydict()
                key = "img" if "img" in obj else "image"
                array = np.stack(
                    [
                        np.asarray(Image.open(BytesIO(a["bytes"])).convert("RGB"))
                        for a in obj[key]
                    ]
                )
                return features(array), np.asarray(obj["label"])

            x, y = read("train.parquet")
            tx, ty = read("test.parquet")
        else:
            try:
                from torchvision.datasets import CIFAR10
            except ImportError as exc:
                raise RuntimeError(
                    "Install the cifar extra: pip install -e '.[cifar]'"
                ) from exc
            train = CIFAR10(root, train=True, download=True)
            test = CIFAR10(root, train=False, download=True)
            x, y = features(train.data), np.array(train.targets)
            tx, ty = features(test.data), np.array(test.targets)
        proportions = rng.dirichlet(np.full(n, alpha), size=10)

        def partition(labels):
            buckets = [[] for _ in range(n)]
            for cls in range(10):
                ids = rng.permutation(np.flatnonzero(labels == cls))
                cuts = np.cumsum(rng.multinomial(len(ids), proportions[cls]))[:-1]
                for i, part in enumerate(np.split(ids, cuts)):
                    buckets[i].extend(part.tolist())
            return [np.asarray(b, dtype=int) for b in buckets]

        training, testing = partition(y), partition(ty)
        for i, (a, b) in enumerate(zip(training, testing)):
            if len(a) < 12 or len(b) < 2:
                continue
            a = rng.permutation(a)
            nv = max(2, int(cifar_validation_fraction * len(a)))
            val, fit = a[:nv], a[nv : nv + samples]
            raw.append(
                (
                    x[fit],
                    one_hot(y[fit], 10),
                    x[val],
                    one_hot(y[val], 10),
                    tx[b],
                    one_hot(ty[b], 10),
                )
            )
            client_ids.append(str(i))
    else:
        # LEAF JSON files must preserve writer IDs; generic EMNIST is not FEMNIST.
        writers = {}
        parquet_paths = sorted(Path(root).rglob("*.parquet"))
        if parquet_paths:
            from io import BytesIO
            import pyarrow.parquet as pq
            from PIL import Image

            # Read metadata first; decode only selected writers' images.
            counts = {}
            for path in parquet_paths:
                for writer in pq.read_table(path, columns=["writer_id"])[
                    "writer_id"
                ].to_pylist():
                    counts[writer] = counts.get(writer, 0) + 1
            eligible = sorted(
                w for w, count in counts.items() if count >= max(20, 2 * feature_dim)
            )
            selected = set(rng.choice(eligible, min(n, len(eligible)), replace=False))
            for path in parquet_paths:
                source = pq.ParquetFile(path)
                for batch in source.iter_batches(batch_size=2048):
                    record = batch.to_pydict()
                    for writer, label, img in zip(
                        record["writer_id"], record["character"], record["image"]
                    ):
                        if writer not in selected:
                            continue
                        pixels = (
                            np.asarray(
                                Image.open(BytesIO(img["bytes"])).convert("L"),
                                dtype=float,
                            )
                            / 255
                        )
                        item = writers.setdefault(writer, {"x": [], "y": []})
                        item["x"].append(pixels.ravel())
                        item["y"].append(label)
        paths = sorted(Path(root).rglob("*.json"))
        if parquet_paths and paths:
            raise ValueError("Use either FEMNIST parquet or LEAF JSON, not both")
        for path in paths:
            obj = json.loads(path.read_text())
            for writer, data in obj.get("user_data", {}).items():
                record = writers.setdefault(writer, {"x": [], "y": []})
                record["x"].extend(data["x"])
                record["y"].extend(data["y"])
        eligible = sorted(
            w for w, a in writers.items() if len(a["y"]) >= max(20, 2 * feature_dim)
        )
        if len(eligible) < 2:
            raise ValueError(
                "Provide LEAF FEMNIST JSON files with at least two writers (20+ examples each)"
            )
        chosen = rng.choice(eligible, min(n, len(eligible)), replace=False)
        for writer in chosen:
            record = writers[writer]
            x = np.asarray(record["x"], dtype=float)
            # Nonoverlapping 4x4 average pooling preserves a modest fixed input size.
            if encoder is None:
                x = x.reshape(-1, 7, 4, 7, 4).mean((2, 4)).reshape(-1, 49)
            else:
                x = encoder((255 * x.reshape(-1, 28, 28)).astype(np.uint8))
            raw.append(split_client(x, np.asarray(record["y"]), 62, rng, samples))
            client_ids.append(str(writer))
    if len(raw) < 2:
        raise ValueError(
            "Partition left fewer than two clients; increase data or alpha"
        )
    if raw_transform is not None:
        raw = raw_transform(raw)
    if use_pca:
        clients, transform = prepare_features(raw, feature_dim, return_transform=True)
    else:
        clients, transform = prepare_full_features(raw)
    return clients, client_ids, transform
