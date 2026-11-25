"""Download benchmark parquet files and verify recorded SHA-256 checksums."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

SOURCES = {
    "cifar10": [
        (
            "train.parquet",
            "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
            "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
        ),
        (
            "test.parquet",
            "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
            "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
        ),
    ],
    "femnist": [
        (
            "train.parquet",
            "https://huggingface.co/datasets/flwrlabs/femnist/resolve/main/data/train-00000-of-00001.parquet",
            "cdb389c8d41d5facab79a4a8d452724c30d9c629a1c1e0b1e9ae9de2c633522b",
        ),
    ],
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=list(SOURCES))
    p.add_argument("--root", default="data")
    args = p.parse_args()
    directory = Path(args.root) / args.dataset
    directory.mkdir(parents=True, exist_ok=True)
    for name, url, expected in SOURCES[args.dataset]:
        target = directory / name
        if target.exists():
            if checksum(target) != expected:
                raise ValueError(f"Existing file has unexpected checksum: {target}")
            print(f"Verified {target}")
            continue
        temporary = target.with_suffix(".partial")
        print(f"Downloading {name}", flush=True)
        urllib.request.urlretrieve(url, temporary)
        if checksum(temporary) != expected:
            temporary.unlink()
            raise ValueError(
                "Dataset checksum changed; inspect source before updating its expected hash"
            )
        temporary.replace(target)
        print(f"Verified {target}")


if __name__ == "__main__":
    main()
