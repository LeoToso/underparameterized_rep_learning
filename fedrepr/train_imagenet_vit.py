"""Supervised ViT-S/16 training from scratch on a reproducible ImageNet fraction.

Input layout: ROOT/train/CLASS/*.JPEG and ROOT/val/CLASS/*.JPEG.
Supports one process or torchrun distributed training. The subset is selected
within each class from the training set. The validation set is never subsampled.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms

from .encoders import resolve_device


def stratified_subset(targets, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("data-fraction must lie in (0, 1]")
    generator = np.random.default_rng(seed)
    targets = np.asarray(targets)
    indices = []
    for label in np.unique(targets):
        candidates = generator.permutation(np.flatnonzero(targets == label))
        count = max(1, int(len(candidates) * fraction))
        indices.extend(candidates[:count].tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


def atomic_save(state, path):
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-fraction", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if not 0 < args.data_fraction <= 1 or args.epochs < 1 or args.batch_size < 1:
        p.error("Use data-fraction in (0,1] and positive epochs/batch size")
    if args.workers < 0 or args.image_size % 16:
        p.error("Use nonnegative workers and an image size divisible by 16")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_device(args.device)
    if world > 1:
        import torch.distributed as dist

        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(args.image_size * 256 / 224)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_data = datasets.ImageFolder(
        args.imagenet_root / "train", transform=train_transform
    )
    validation_data = datasets.ImageFolder(
        args.imagenet_root / "val", transform=val_transform
    )
    if train_data.class_to_idx != validation_data.class_to_idx:
        raise ValueError("ImageNet train/val class directories must match")
    indices = stratified_subset(train_data.targets, args.data_fraction, args.seed)
    train_subset = Subset(train_data, indices.tolist())
    # Rank-strided validation has no padding or duplicate examples.
    val_subset = Subset(validation_data, list(range(rank, len(validation_data), world)))
    sampler = (
        DistributedSampler(train_subset, shuffle=True, seed=args.seed)
        if world > 1
        else None
    )
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader_args = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator,
        **loader_args,
    )
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, **loader_args)
    import timm

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=len(train_data.classes),
        img_size=args.image_size,
    ).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    base_model = model.module if world > 1 else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def schedule(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(
            1, args.epochs - args.warmup_epochs
        )
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        model="vit_small_patch16_224",
        embed_dim=384,
        objective="supervised_cross_entropy",
        initialization="random",
        selected_train_examples=len(indices),
        available_train_examples=len(train_data),
        actual_fraction=len(indices) / len(train_data),
        subset_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        world_size=world,
    )
    start, best, history = 0, -1.0, []
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in (
            "model",
            "image_size",
            "data_fraction",
            "seed",
            "epochs",
            "batch_size",
            "world_size",
            "subset_sha256",
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
        ):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume configuration differs at {key}")
        if state["classes"] != train_data.classes:
            raise ValueError("Resume dataset classes differ from the checkpoint")
        base_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, history = (
            state["epoch"] + 1,
            state["best_accuracy"],
            state["history"],
        )
        rng_state = state["rng_by_rank"][rank]
        torch.set_rng_state(rng_state["torch"])
        random.setstate(rng_state["python"])
        generator.set_state(rng_state["loader"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng_state["cuda"], device)
    else:
        exists = torch.tensor(int(args.output.exists()), device=device)
        if world > 1:
            dist.all_reduce(exists, op=dist.ReduceOp.MAX)
        if exists.item():
            raise FileExistsError(
                "Use a new output directory or --resume a saved last.pt"
            )
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        atomic_save(torch.as_tensor(indices), args.output / "subset_indices.pt")
    if world > 1:
        dist.barrier()
    for epoch in range(start, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        loss_total = 0.0
        count = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(model(x), y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach()) * len(y)
            count += len(y)
        base_model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in val_loader:
                prediction = base_model(x.to(device, non_blocking=True)).argmax(1).cpu()
                correct += int((prediction == y).sum())
                total += len(y)
        metrics = torch.tensor(
            [loss_total, count, correct, total], device=device, dtype=torch.float64
        )
        if world > 1:
            dist.all_reduce(metrics)
        train_loss, train_count, correct, total = metrics.tolist()
        accuracy = correct / total
        improved = accuracy > best
        best = max(best, accuracy)
        scheduler.step()
        history.append(
            dict(
                epoch=epoch + 1,
                training_loss=train_loss / train_count,
                validation_accuracy=accuracy,
            )
        )
        rng_state = dict(
            torch=torch.get_rng_state(),
            python=random.getstate(),
            loader=generator.get_state(),
        )
        if device.type == "cuda":
            rng_state["cuda"] = torch.cuda.get_rng_state(device)
        rng_states = [rng_state]
        if world > 1:
            rng_states = [None] * world
            dist.all_gather_object(rng_states, rng_state)
        if rank == 0:
            state = dict(
                model=base_model.state_dict(),
                num_classes=len(train_data.classes),
                classes=train_data.classes,
                config=config,
                epoch=epoch,
                best_accuracy=best,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                rng_by_rank=rng_states,
                history=history,
            )
            atomic_save(state, args.output / "last.pt")
            if improved:
                atomic_save(state, args.output / "best.pt")
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            print(
                f"ImageNet epoch {epoch+1}/{args.epochs}: val={accuracy:.4f} train_loss={train_loss/train_count:.4f}",
                flush=True,
            )
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
