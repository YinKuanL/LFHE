# LFHE: Local-First Heuristic Evolution for Decentralized Learning

Local-First Heuristic Evolution (LFHE) studies whether fully decentralized learning systems can adapt their communication topology using only local model and neighborhood information under non-IID data.

## Research Question

Fully decentralized learning removes the central coordinator used in federated learning, but it also makes the communication graph part of the learning problem. Static topologies can be brittle when clients hold heterogeneous data, while globally optimized graph updates may require information that a peer-to-peer system does not naturally have.

This repository asks:

> Can a decentralized learner improve its communication topology using local, bounded-degree, coordination-free rewiring while preserving reproducible experimental behavior?

## Core Method

LFHE uses a local fitness rule to evaluate candidate topology changes. Each client considers nearby friend-of-friend candidates and accepts bounded-degree graph updates when the local fitness improves.

The implementation combines:

- decentralized model averaging over a peer-to-peer graph,
- Dirichlet non-IID client data partitions,
- static topology baselines such as random, ring, and fully connected graphs,
- local model-similarity and connectivity proxies,
- bounded-degree adaptive rewiring without global graph optimization.

## Key Results

Tracked repository artifacts support the following result categories:

- CIFAR-10 decentralized learning with 10 clients and 1000 communication rounds;
- beta/topology behavior analysis;
- client-scale comparison experiments;
- an experiment inventory covering seed, alpha, topology-interval, baseline, and dataset-scaling studies.

The tracked result files are listed in [docs/results_provenance.md](docs/results_provenance.md). This README intentionally avoids introducing unverified numerical claims.

## Main Result Artifacts

| Artifact | Description |
| --- | --- |
| `results/CIFAR10_10clients_1000rounds.png` | Tracked CIFAR-10 10-client result figure. |
| `results/CIFAR10_10clients_1000rounds.npy` | Underlying tracked NumPy result artifact for the CIFAR-10 10-client run. |
| `results/beta_comparison.png` | Tracked beta-comparison figure. |
| `results/client_scale_comparison.png` | Tracked client-scale comparison figure. |
| `results/experiments` | Tracked inventory of experiment categories. |

## Paper And Publication Information

This repository is maintained as an anonymized research-code artifact. Publication status and author-identifying information are intentionally omitted.

## Quick Start

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd LFHE
pip install -r requirements.txt
```

Run the main tracked CIFAR-10 experiment script:

```bash
cd program
python train_dfl_cifar10_seeds.py
```

The script downloads CIFAR-10 automatically through torchvision and writes generated outputs in the current working directory.

## Reproducing Reported Experiments

The main tracked experiment entry point is `program/train_dfl_cifar10_seeds.py`.

The script defines the CIFAR-10 model, Dirichlet partitioning, decentralized aggregation, random graph initialization, LFHE topology updates, and multi-seed evaluation loop. In the tracked main block, the seed list is `42, 43, 44, 45, 46`.

Additional local experiment directories may exist in working copies. Some contain useful source code mixed with generated datasets and result files. They should be reviewed and tracked selectively in a separate pass so that reproducibility code is preserved without committing large generated artifacts.

See [docs/reproducibility.md](docs/reproducibility.md) for cleanup-safe reproducibility notes.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── docs/
│   ├── reproducibility.md
│   └── results_provenance.md
├── program/
│   ├── lfhe.py
│   └── train_dfl_cifar10_seeds.py
└── results/
    ├── CIFAR10_10clients_1000rounds.npy
    ├── CIFAR10_10clients_1000rounds.png
    ├── beta_comparison.png
    ├── client_scale_comparison.png
    └── experiments
```

## Environment And Dependencies

Dependencies are specified in `requirements.txt`:

- PyTorch
- torchvision
- NumPy
- Matplotlib
- NetworkX
- SciPy

The experiment script selects CUDA automatically when available and otherwise runs on CPU.

## Result Provenance

The `.gitignore` is configured to prevent accidental commits of downloaded datasets, generated arrays, plots, checkpoints, logs, caches, local environments, and machine-specific artifacts.

Existing tracked result artifacts remain versioned. New generated results should be added only when they are intentionally part of the public reproducibility record and are accompanied by enough provenance to identify the script, settings, seeds, and environment used to produce them.

## Citation

An anonymized citation entry can be added after the work no longer requires double-blind anonymity.

## License

No license file is currently included in this repository. Add one before public release if redistribution or reuse terms should be explicit.
