# Reproducibility Notes

This repository preserves the original LFHE experiment scripts and tracked result artifacts. Cleanup changes should not alter algorithms, topology updates, dataset splits, seeds, aggregation, evaluation logic, or reported values.

## Environment

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The experiments use PyTorch, torchvision, NumPy, Matplotlib, NetworkX, and SciPy. CUDA is used automatically when available; otherwise the scripts run on CPU.

## Main Tracked Experiment Script

The tracked CIFAR-10 multi-seed script is:

```bash
cd program
python train_dfl_cifar10_seeds.py
```

The script downloads CIFAR-10 through `torchvision.datasets.CIFAR10`, partitions it with a Dirichlet non-IID split, evaluates decentralized learning across seeds, and saves generated arrays/figures in the working directory.

## Seeds And Protocol

The tracked script uses seeds `42, 43, 44, 45, 46` in its main block and runs baseline and LFHE variants with the protocol encoded in `program/train_dfl_cifar10_seeds.py`.

For archival integrity, inspect the script before changing any experimental setting. Changes to rounds, topology interval, client count, Dirichlet alpha, aggregation, evaluation cadence, model architecture, or random seeds should be treated as scientific changes rather than repository cleanup.

## Generated Files

Generated datasets, `.npy` arrays, figures, logs, checkpoints, and caches are ignored by `.gitignore` to avoid accidental commits. Existing tracked result artifacts remain tracked by Git.

Useful experiment source code that is currently untracked can be added selectively in a future pass without adding its generated data.
