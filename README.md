# Underparameterized representation learning

Train a shared linear representation and personalized client heads on CIFAR-10 and FEMNIST, then generate the two-panel accuracy-versus-dimension figure.

The learned classifier has the factorization W_i = B H_i. Cross-moments initialize B. Training alternates between exact local ridge regression and averaged representation-gradient updates with QR orthonormalization.

## Install

Requires Python 3.9 or newer. The server setup with Python 3.9 can use its existing environment.

```bash
git clone https://github.com/LeoToso/underparameterized_rep_learning.git
cd underparameterized_rep_learning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment using `.venv\Scripts\activate`.

## Train and plot

```bash
python train_and_plot.py
```

This downloads checksum-verified datasets and DINOv2 weights, runs the dimension grid `2,4,6,8,10,16,24,32,40,48,56,64` with three seeds, and writes:

- `runs/cifar/metrics.csv` and `runs/femnist/metrics.csv`
- Per-dimension model checkpoints and feature preprocessing transforms
- `runs/figures/cifar_femnist.png` and `runs/figures/cifar_femnist.pdf`

CIFAR-10 uses frozen pretrained DINOv2 features. FEMNIST trains a handwriting CNN for 30 epochs on the pooled training examples, then freezes it. Both use training-only PCA to 64 features, followed by the learned linear bottleneck and local ridge heads.

To reuse the included FEMNIST CNN checkpoint and rerun the FedRep sweep:

```bash
python train_and_plot.py --use-saved-cnn
```

To use pointwise 95% Student t intervals in both panels:

```bash
python train_and_plot.py --use-saved-cnn --both-ci --output runs_ci
```

The default figure follows the earlier plot, with one sample standard deviation for CIFAR and pointwise 95% intervals for FEMNIST. CIFAR seeds vary client/data partitions. FEMNIST seeds vary the cross-moment split with the CNN and writer splits held fixed. These FEMNIST intervals quantify initialization variability.

Use `--data /path/to/data` to reuse downloaded parquet files. Use `--dry-run` to print the training commands. Select a fresh output folder for each run.

## Experiment settings

| Setting | CIFAR-10 | FEMNIST |
|---|---|---|
| Clients | 30 | 1,000 natural writers |
| Feature extractor | Frozen DINOv2 ViT-S/14 at 84 pixels | Handwriting CNN, 128 output features |
| Representation input | Training-only PCA, 64 features | Training-only PCA, 64 features |
| Training split | About 45,000 images | 60% per writer, about 137,149 images |
| Validation split | 10% within each client's official training partition | 20% per writer |
| Test split | Official test split | 20% per writer |
| Alternating optimization | Up to 500 rounds | Up to 500 rounds |

Validation selects ridge regularization from 0.001 and 0.01 and the representation checkpoint. The classification objective is squared loss on one-hot labels. The reported metric is mean client test accuracy. At k=64, the full-dimensional orthonormal projection preserves ridge predictions, so FEMNIST skips redundant rotations.

The classification cross-moment initializer is an empirical adaptation of the scalar Gaussian-regression estimator in the report. The PCA preprocessing and FEMNIST CNN pretraining use pooled training examples centrally. Every classification score in these experiments passes through the learned bottleneck and personalized head.

## Code

- `train_and_plot.py` downloads dependencies and coordinates training and plotting.
- `fedrepr/core.py` implements initializers, local heads, and alternating updates.
- `fedrepr/data.py` constructs client splits and fits feature preprocessing.
- `fedrepr/strong_features.py` runs the DINOv2 feature experiment.
- `fedrepr/handwriting.py` trains the handwriting encoder and FEMNIST bottlenecks.
- `plot_cifar_femnist.py` creates the two-panel figure from metrics.
- `checkpoints/femnist_cnn/` contains the actual saved encoder and training history.
- `vendor/facebookresearch_dinov2_main/` contains upstream DINOv2 source and its license.

## Validation

```bash
python -m unittest discover -s tests -v
python train_and_plot.py --dry-run
```

Nine numerical checks cover the spectral-tail identity, least-squares equations, population descent, valid moment inputs, reproducibility, exclusion of held-out examples from preprocessing, preservation of all embedding coordinates, equivalence of the chunked solver, and reproducible ImageNet subset selection. The assembled training commands were checked without rerunning the full compute-intensive experiments. Numerical results can vary across software versions and hardware.

## Data and upstream code

CIFAR-10 parquet files come from [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10). Writer-identified FEMNIST comes from [flwrlabs/femnist](https://huggingface.co/datasets/flwrlabs/femnist). Download checksums are recorded in `fedrepr/download.py`.

The vendored encoder implementation is from [DINOv2](https://github.com/facebookresearch/dinov2). Its Apache-2.0 license is retained in the vendor directory.

## Full 384-dimensional encoder comparisons

The new pipeline removes PCA and runs both datasets with frozen DINOv2, iBOT, DINOv3, or a ViT trained from scratch on an adjustable ImageNet fraction (default 30%). It supports CUDA, includes k=384 as the full-dimension reference, and plots pointwise 95% intervals.

See [ENCODER_EXPERIMENTS.md](ENCODER_EXPERIMENTS.md) for setup, checkpoint requirements, ImageNet training, and complete commands.

```bash
pip install -r requirements-embeddings.txt
python run_embedding_comparison.py --encoders dinov2 ibot --device cuda --output embedding-runs/public
```
