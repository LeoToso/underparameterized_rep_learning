"""Frozen 384-dimensional encoders with checkpoint-aware feature caching."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .download import checksum

DINO2_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
)
DINO2_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
IBOT_URL = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth"
IBOT_SHA = "58ba95f2b14298c76c3df0a24b77189714aa8076501bff325107f28a9697a4b8"


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access a GPU")
    return device


def download_checkpoint(url, destination, expected_hash=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".partial")
        torch.hub.download_url_to_file(url, str(temporary))
        if expected_hash and checksum(temporary) != expected_hash:
            temporary.unlink()
            raise ValueError("Checkpoint checksum differs from the recorded hash")
        os.replace(temporary, destination)
    if expected_hash and checksum(destination) != expected_hash:
        raise ValueError("Checkpoint checksum differs from the recorded hash")
    return destination


def load_ibot(checkpoint):
    import timm

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("teacher", "state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    backbone = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        # Projection heads and the optional training-only mask token are unused.
        if key.startswith("head.") or key == "masked_embed":
            continue
        backbone[key] = value
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    model.load_state_dict(backbone, strict=True)
    return model


def load_imagenet_vit(checkpoint):
    import timm

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    model = timm.create_model(
        config["model"],
        pretrained=False,
        num_classes=state["num_classes"],
        img_size=config["image_size"],
    )
    model.load_state_dict(state["model"], strict=True)
    model.reset_classifier(0)
    return model, config


class FrozenEncoder:
    def __init__(
        self,
        name,
        cache,
        checkpoint=None,
        dinov3_repo=None,
        device="auto",
        batch_size=64,
        image_size=224,
    ):
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parents[1]
        self.image_size = image_size
        if image_size < 28:
            raise ValueError("Encoder image size must be at least 28")
        if name == "dinov2":
            checkpoint = (
                Path(checkpoint)
                if checkpoint
                else download_checkpoint(
                    DINO2_URL, self.cache / "weights/dinov2_vits14.pth", DINO2_SHA
                )
            )
            if image_size % 14:
                raise ValueError("DINOv2 image size must be divisible by 14")
            self.model = torch.hub.load(
                str(root / "vendor/facebookresearch_dinov2_main"),
                "dinov2_vits14",
                source="local",
                pretrained=False,
            )
            self.model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        elif name == "ibot":
            checkpoint = (
                Path(checkpoint)
                if checkpoint
                else download_checkpoint(
                    IBOT_URL, self.cache / "weights/ibot_vits16_teacher.pth", IBOT_SHA
                )
            )
            self.model = load_ibot(checkpoint)
            if image_size != 224:
                raise ValueError("The iBOT adapter uses its native 224-pixel input")
        elif name == "dinov3":
            if checkpoint is None or dinov3_repo is None:
                raise ValueError("DINOv3 requires --checkpoint and --dinov3-repo")
            if image_size % 16:
                raise ValueError("DINOv3 image size must be divisible by 16")
            checkpoint = Path(checkpoint).expanduser().resolve()
            self.model = torch.hub.load(
                str(Path(dinov3_repo).expanduser().resolve()),
                "dinov3_vits16",
                source="local",
                weights=str(checkpoint),
            )
        elif name == "imagenet-vit":
            if checkpoint is None:
                raise ValueError("Train the ImageNet ViT first and pass --checkpoint")
            checkpoint = Path(checkpoint)
            self.model, config = load_imagenet_vit(checkpoint)
            if image_size != config["image_size"]:
                raise ValueError(
                    "Embedding image size must match the ImageNet training configuration"
                )
        else:
            raise ValueError(name)
        self.model.eval().requires_grad_(False).to(self.device)
        self.metadata = {
            "encoder": name,
            "checkpoint_sha256": checksum(Path(checkpoint)),
            "embedding_dimension": 384,
            "image_size": image_size,
            "normalization": "ImageNet mean/std",
            "resize": "bicubic_antialias",
            "torch_version": str(torch.__version__),
        }
        if name == "imagenet-vit":
            self.metadata["pretraining"] = config
        identity = json.dumps(self.metadata, sort_keys=True)
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[
            None, :, None, None
        ]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[
            None, :, None, None
        ]

    def __call__(self, images):
        images = np.ascontiguousarray(images)
        digest = hashlib.sha256()
        digest.update(str((images.shape, images.dtype)).encode())
        digest.update(images.tobytes())
        path = self.cache / f"{self.identity}_{digest.hexdigest()}.npy"
        if path.exists():
            result = np.load(path)
            if result.shape != (len(images), 384):
                raise ValueError("Cached embedding shape is inconsistent")
            return result
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                x = (
                    torch.from_numpy(images[start : start + self.batch_size]).to(
                        self.device, dtype=torch.float32
                    )
                    / 255
                )
                if x.ndim == 3:
                    x = x[:, None].expand(-1, 3, -1, -1)
                else:
                    x = x.permute(0, 3, 1, 2)
                x = torch.nn.functional.interpolate(
                    x,
                    size=(self.image_size, self.image_size),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp(0, 1)
                out = self.model((x - self.mean) / self.std)
                if out.ndim != 2 or out.shape[1] != 384:
                    raise ValueError(
                        f"Expected 384-dimensional embeddings, received {tuple(out.shape)}"
                    )
                outputs.append(out.float().cpu().numpy())
                if len(images) > 1000 and start % (self.batch_size * 50) == 0:
                    print(
                        f"Embedded {min(start+self.batch_size, len(images))}/{len(images)}",
                        flush=True,
                    )
        result = np.concatenate(outputs)
        temporary = path.with_suffix(f".{os.getpid()}.partial")
        with temporary.open("wb") as handle:
            np.save(handle, result)
        os.replace(temporary, path)
        return result
