"""Linear shared representations; clients never share example-level gradients."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Client:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    tx: np.ndarray
    ty: np.ndarray


def orthogonal(a):
    return np.linalg.qr(a, mode="reduced")[0]


def fit_head(x, y, ridge=0.0):
    """Stable least squares; ridge objective is mean loss + ridge/2 ||h||²."""
    if ridge == 0:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    return np.linalg.solve(x.T @ x + len(x) * ridge * np.eye(x.shape[1]), x.T @ y)


def initialize(clients, k, rng, mode):
    d = clients[0].x.shape[1]
    if mode == "random":
        return orthogonal(rng.normal(size=(d, k)))
    matrix = np.zeros((d, d))
    for c in clients:
        if mode == "moments":
            if c.y.shape[1] != 1:
                raise ValueError(
                    "Scalar regression moments cannot be used with one-hot labels"
                )
            matrix += c.x.T @ (c.x * c.y**2) / len(c.x)
        elif mode == "cross_moments":
            # Independent halves remove the self-product noise bias of C.T @ C.
            a, b = np.array_split(np.arange(len(c.x)), 2)
            if not len(b):
                raise ValueError(
                    "Cross moments require at least two examples per client"
                )
            ca = c.x[a].T @ c.y[a] / len(a)
            cb = c.x[b].T @ c.y[b] / len(b)
            matrix += (ca @ cb.T + cb @ ca.T) / 2
        elif mode == "pca":
            matrix += c.x.T @ c.x / len(c.x)
        else:
            raise ValueError(mode)
    return np.linalg.eigh(matrix / len(clients))[1][:, -k:]


def evaluate(clients, weights, split, classification):
    errors, sizes = [], []
    for c, w in zip(clients, weights):
        x, y = (c.vx, c.vy) if split == "val" else (c.tx, c.ty)
        prediction = x @ w
        errors.append(
            float(np.mean(prediction.argmax(1) == y.argmax(1)))
            if classification
            else float(np.mean(np.sum((prediction - y) ** 2, axis=1)))
        )
        sizes.append(len(x))
    return {
        "macro": float(np.mean(errors)),
        "micro": float(np.average(errors, weights=sizes)),
        "p10": float(np.quantile(errors, 0.1)),
        "p90": float(np.quantile(errors, 0.9)),
    }


def train(
    clients,
    k,
    method,
    rng,
    rounds=100,
    lr=0.1,
    ridge=0.0,
    classification=False,
    fresh=None,
):
    """Select checkpoints on validation clients' examples, never on test data.

    Fresh(round) returns independent per-round training batches for synthetic theory runs.
    Both the head and representation gradient use the same batch, as in the report.
    """
    d, q = clients[0].x.shape[1], clients[0].y.shape[1]
    if method == "local":
        w = [fit_head(c.x, c.y, ridge) for c in clients]
        return w, None, [], 0
    if method in {"fedavg", "fedavg_ft"}:
        w = np.zeros((d, q))
        best, score, history = w.copy(), -np.inf if classification else np.inf, []
        for t in range(rounds):
            # Full participation, one local full-batch step; uniform client weights.
            w -= lr * np.mean(
                [c.x.T @ (c.x @ w - c.y) / len(c.x) + ridge * w for c in clients],
                axis=0,
            )
            metric = evaluate(clients, [w] * len(clients), "val", classification)[
                "macro"
            ]
            history.append(metric)
            if (metric > score) if classification else (metric < score):
                best, score = w.copy(), metric
        weights = [best.copy() for _ in clients]
        if method == "fedavg_ft":
            # Adaptation strength is selected by the outer validation grid.
            weights = [best + fit_head(c.x, c.y - c.x @ best, ridge) for c in clients]
        return weights, None, history, 2 * rounds * len(clients) * d * q
    mode = {
        "proposed": "cross_moments" if classification else "moments",
        "fedrep": "random",
        "pca": "pca",
        "random": "random",
    }[method]
    b = initialize(clients, k, rng, mode)
    initial_cost = len(clients) * d * d if mode != "random" else 0
    best_weights, best_b, history = None, None, []
    best_score = -np.inf if classification else np.inf
    for t in range(rounds + 1):
        weights = [b @ fit_head(c.x @ b, c.y, ridge) for c in clients]
        metric = evaluate(clients, weights, "val", classification)["macro"]
        history.append(metric)
        if (metric > best_score) if classification else (metric < best_score):
            best_weights, best_b, best_score = weights, b.copy(), metric
        if t == rounds or method in {"pca", "random"}:
            break
        batches = fresh(t) if fresh else [(c.x, c.y) for c in clients]
        gradient = np.zeros_like(b)
        for x, y in batches:
            h = fit_head(x @ b, y, ridge)
            gradient += x.T @ (x @ b @ h - y) @ h.T / len(x)
        b = orthogonal(b - lr * gradient / len(clients))
    cost = initial_cost + (
        0 if method in {"pca", "random"} else 2 * rounds * len(clients) * d * k
    )
    return best_weights, best_b, history, cost
