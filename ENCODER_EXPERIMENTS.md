# Full-embedding experiments

This pipeline evaluates our cross-moment-initialized, alternating linear representation method on frozen image embeddings for both CIFAR-10 and the full 62-class FEMNIST task.

All four encoder choices produce 384-dimensional vectors. Every prediction passes through the shared learned 384-by-k representation and a personalized linear head. The encoder is frozen during FedRep.

| Encoder option | Backbone | Source of weights |
|---|---|---|
| `dinov2` | DINOv2 ViT-S/14 | Official public pretrained checkpoint |
| `ibot` | iBOT ViT-S/16 | Official public ImageNet teacher backbone |
| `dinov3` | DINOv3 ViT-S/16 | User-supplied weights obtained through Meta's access process |
| `imagenet-vit` | ViT-S/16 | Supervised training from random initialization on an adjustable fraction of ImageNet |

The fourth setting uses ImageNet class labels and cross-entropy pretraining. This objective is a design choice for the requested custom ViT. The other checkpoints come from their authors' pretraining recipes.

## Environment

Use Python 3.11 or newer for the combined suite, including the upstream DINOv3 implementation. The previous experiment remains available through `train_and_plot.py`.

```bash
git pull --ff-only
pip install -r requirements-embeddings.txt
```

If your current environment has Python 3.9, create a separate environment for the combined suite, for example:

```bash
conda create -n rep384 python=3.11 -y
conda activate rep384
pip install -r requirements-embeddings.txt
```

The default device is CUDA when available and CPU otherwise. Use `--device cuda` to require a GPU. The ImageNet trainer also supports multiple GPUs through `torchrun`.

## DINOv2 and iBOT

This command downloads the public checkpoints and both datasets, then runs both encoders on both datasets:

```bash
python -u run_embedding_comparison.py \
  --encoders dinov2 ibot \
  --output embedding-runs/public \
  --device cuda
```

For a quicker initial experiment:

```bash
python -u run_embedding_comparison.py \
  --encoders dinov2 \
  --dimensions 8 32 64 128 384 \
  --init-seeds 0 \
  --rounds 100 \
  --output embedding-runs/dino2-pilot \
  --device cuda
```

Feature extraction uses 224-by-224 images for all four encoders. FEMNIST grayscale images are replicated into three channels. Resizing is bicubic with antialiasing and the images use ImageNet normalization. This resolution differs from the earlier 84-pixel DINOv2 run.

## DINOv3

Obtain the official ViT-S/16 weights through the upstream project's access process, and clone its implementation:

```bash
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
python -u run_embedding_comparison.py \
  --encoders dinov3 \
  --dinov3-repo external/dinov3 \
  --dinov3-checkpoint /path/to/dinov3_vits16_pretrain.pth \
  --output embedding-runs/dino3 \
  --device cuda
```

The loader calls the official `dinov3_vits16` PyTorch Hub entry point. Supply a local checkpoint path after obtaining access to the weights.

## Train a ViT using part of ImageNet

Prepare a local ImageNet dataset with matching class directories under `train/` and `val/`. The program does not download ImageNet. It samples training images independently within every class using a fixed seed, and saves the selected indices, their hash, and the actual sample count.

The default requested fraction is 0.30. Change `--data-fraction` to 0.10, 0.50 or another value in (0, 1]. Integer rounding and retaining at least one image per class can slightly change the actual fraction. Subsets from the same seed are nested as the fraction increases.

Single-GPU pretraining followed by evaluation on both downstream tasks:

```bash
python -u run_embedding_comparison.py \
  --encoders imagenet-vit \
  --imagenet-root /path/to/imagenet \
  --data-fraction 0.30 \
  --imagenet-epochs 100 \
  --output embedding-runs/imagenet30 \
  --device cuda
```

The ViT-S/16 has 12 blocks, 6 attention heads, and a 384-dimensional embedding. The default recipe uses AdamW, a five-epoch learning-rate warmup, cosine decay, random resized crops and horizontal flips, and cross-entropy with label smoothing. The ImageNet classifier is removed after training, and the encoder is frozen for downstream FedRep. The best encoder checkpoint is selected using ImageNet validation accuracy.

For multi-GPU pretraining, run the trainer separately:

```bash
torchrun --standalone --nproc_per_node=4 -m fedrepr.train_imagenet_vit \
  --imagenet-root /path/to/imagenet \
  --data-fraction 0.30 \
  --epochs 100 \
  --batch-size 64 \
  --output imagenet-pretrain30 \
  --device cuda

python run_embedding_comparison.py \
  --encoders imagenet-vit \
  --imagenet-checkpoint imagenet-pretrain30/best.pt \
  --output embedding-runs/imagenet30-eval \
  --device cuda
```

Batch size is per GPU. The manifest records the world size. To resume interrupted ImageNet training, repeat the same trainer configuration and add `--resume imagenet-pretrain30/last.pt`. Each epoch saves the model, optimizer, scheduler, scaler and per-rank random-generator states.

## Run all four settings

```bash
python -u run_embedding_comparison.py \
  --encoders dinov2 ibot dinov3 imagenet-vit \
  --dinov3-repo external/dinov3 \
  --dinov3-checkpoint /path/to/dinov3_vits16_pretrain.pth \
  --imagenet-root /path/to/imagenet \
  --data-fraction 0.30 \
  --imagenet-epochs 100 \
  --output embedding-runs/all-four \
  --device cuda
```

The custom ViT is pretrained before its downstream experiments. An already trained custom encoder can be supplied through `--imagenet-checkpoint`.

## FedRep and data settings

There is no PCA or coordinate selection. Each encoder's 384 coordinates are centered by the training mean and divided by one scalar training RMS. Those values are saved for inference. This invertible affine preprocessing preserves all embedding coordinates.

The default k grid is 2,4,6,8,10,16,24,32,40,48,56,64,96,128,192,256,384. Pass `--dimensions` to change it. The full-embedding reference is k=384. A positive ridge penalty supports clients with fewer training examples than k.

The default client populations are 30 CIFAR-10 clients and up to 1,000 FEMNIST writers. Writer eligibility remains at 128 examples before splitting, independent of the encoder or k. For each data seed, all encoder choices use the same client and example splits. The CIFAR split retains approximately 45,000 training images with Dirichlet alpha=0.3; FEMNIST uses 60/20/20 within-writer splits.

The representation uses cross-moment initialization, exact local ridge heads, averaged representation gradients, and QR orthonormalization. Training has a 500-round budget, with checkpoint validation every 10 rounds. Validation selects ridge strength from 0.001 and 0.01. Full-dimensional ridge predictions are invariant to the orthonormal basis, so k=384 fits the heads directly.

The default data seed is 0 with initialization seeds 0,1,2. Add `--data-seeds 0 1 2` to vary client/data partitions. Choose fresh output directories for additional runs. Feature caches include checkpoint hashes, image resolution, normalization, and image contents.

## Outputs and confidence intervals

The suite writes per-encoder/per-dataset directories containing config, partition manifests, metrics, preprocessing and model checkpoints. It then creates:

- `figures/encoder_comparison.png`
- `figures/encoder_comparison.pdf`
- `figures/confidence_intervals.csv`

Each row of the figure corresponds to an encoder, with CIFAR and FEMNIST in the two columns. Bands are pointwise 95% Student t intervals. With one data partition, independent units are initialization seeds conditional on the frozen checkpoint and partition. With multiple data partitions, initialization runs are first averaged within each partition, then intervals are computed across partitions.

Regenerate only the plot with:

```bash
python plot_embedding_comparison.py \
  --results embedding-runs/public --encoders dinov2 ibot
```

## Validation status

Nine numerical checks pass. CPU smoke runs verified the official iBOT checkpoint, DINOv2 feature extraction with the full 384 coordinates, no-PCA FedRep training, and a tiny ImageFolder ViT training/checkpoint cycle. These checks verify execution and numerical behavior; they do not constitute new CIFAR/FEMNIST benchmark results. DINOv3's checkpoint path and full ImageNet pretraining require resources supplied on the server. CUDA and multi-GPU execution have not been validated in this local CPU environment.

## Official sources

- [DINOv2](https://github.com/facebookresearch/dinov2)
- [iBOT pretrained backbones](https://github.com/bytedance/ibot#pre-trained-models)
- [DINOv3 checkpoint access and loading instructions](https://github.com/facebookresearch/dinov3)

## Continue an interrupted sweep

After the previous process has stopped, repeat its command with `--resume`:

```bash
python -u run_embedding_comparison.py \
  --encoders dinov2 ibot \
  --device cuda \
  --output embedding-runs/public \
  --resume
```

Completed seed/dimension points and their checkpoints are reused. A partially completed point restarts, using cached image embeddings. The saved encoder, data partitions and training settings are checked before continuation. Model files and metrics are written through temporary files so interrupted writes preserve previously completed results. A complete dataset sweep is skipped automatically. The final plot is generated when all requested sweeps are complete.
