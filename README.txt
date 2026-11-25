TRAINING AND FIGURE REPRODUCTION
Requires Python 3.11+ and an internet connection for datasets and DINOv2 weights.

From this extracted folder run

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python train_and_plot.py

On Windows activate with .venv\Scripts\activate instead.

The default command downloads checksum-verified CIFAR-10/FEMNIST data, extracts
frozen DINOv2 features for CIFAR-10, trains its FedRep models, trains a FEMNIST
CNN for 30 epochs, freezes its features, trains its FedRep models, and saves
runs/figures/cifar_femnist.png and runs/figures/cifar_femnist.pdf.

The dimension grid is 2,4,6,8,10,16,24,32,40,48,56,64. Representation learning
has a 500-round budget with validation-selected checkpoints. At full dimension
64, FEMNIST skips redundant rotations and fits the local ridge heads directly.

To regenerate the plotted FEMNIST experiment using the actual saved CNN weights

    python train_and_plot.py --use-saved-cnn

This still trains the local heads and shared bottlenecks for all three seeds.
The included checkpoint was selected using validation from the 1,000-writer,
30-epoch training run. Retraining from scratch can vary with library versions,
hardware and numerical kernels.

To put 95% confidence intervals in both panels

    python train_and_plot.py --use-saved-cnn --both-ci --output runs_ci

Without --both-ci, shading reproduces the previous figure: CIFAR uses one sample
standard deviation and FEMNIST uses pointwise 95% Student t intervals. CIFAR
seeds vary data partitions. FEMNIST seeds vary the random cross-moment split
while keeping the encoder, writer population and data partitions fixed.

Use --data /path/to/data to reuse a directory containing cifar10/train.parquet,
cifar10/test.parquet and femnist/train.parquet. Files are verified before use.
Use --dry-run to inspect commands. Choose a fresh --output directory for each run.

MODEL AND DATA DETAILS

CIFAR uses the bundled upstream DINOv2 ViT-S/14 implementation (Apache-2.0,
see vendor/facebookresearch_dinov2_main/LICENSE), frozen pretrained weights,
84x84 resized RGB images, training-only PCA to 64 dimensions, 30 Dirichlet
alpha=0.3 clients, and approximately 45,000 training examples. It uses 10%
within-client validation from the official training set and the official test set.

FEMNIST uses 1,000 natural writers with at least 128 examples each and within-writer
60/20/20 splits, approximately 137,149 training examples for seed 0. A CNN produces
128-dimensional features, followed by training-only PCA to 64. The CNN is trained
centrally on the pooled training splits and frozen before FedRep.

Both tasks use a shared linear projection initialized by cross-moments,
alternating exact local ridge head fits and averaged representation-gradient
updates, and QR orthonormalization. Ridge is validation-selected from 0.001,0.01.
All class scores pass through the bottleneck. No additive shared-classifier
prediction branch is used. FedRep fits squared loss on one-hot labels.

The source is in fedrepr/core.py, data.py, strong_features.py and handwriting.py.
plot_cifar_femnist.py reads the generated CSV files and draws the two panels.

CODE VALIDATION

The assembled package was checked with --dry-run, Python compilation and the
existing six numerical tests on the shared algorithm. The complete expensive
training pipeline was not rerun as part of preparing this download. The source
training modules were used for the completed experiments in the conversation.
