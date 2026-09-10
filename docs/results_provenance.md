# Results Provenance

This page records what can be verified from files currently tracked in the repository.

## Tracked Result Artifacts

The repository currently tracks:

- `results/CIFAR10_10clients_1000rounds.npy`
- `results/CIFAR10_10clients_1000rounds.png`
- `results/beta_comparison.png`
- `results/client_scale_comparison.png`
- `results/experiments`
- `figures/cifar10_main_result.png`, copied from `results/CIFAR10_10clients_1000rounds.png` for README display.

The README discusses results at the level supported by these artifacts: CIFAR-10 decentralized learning, multi-client experiments, beta/topology behavior, and client-scale comparisons. It does not introduce new numerical claims beyond what is present in tracked files.

## Experiment Inventory

The tracked `results/experiments` file lists the intended experiment categories:

1. Seed comparison
2. Alpha experiments for data distribution
3. Client-number experiments for system-size effects
4. Topology-interval experiments for convergence under different update rates
5. Baseline versus enhancement comparisons against fully connected, ring, and random topologies
6. Dataset experiments for scalability

## Tracked Reproducibility Source

The repository now tracks source entry points for:

- CIFAR-10 baseline comparison: `program/Baseline_comparison/main.py`
- Dirichlet-alpha sensitivity: `program/Alpha_test/main.py`
- topology-evolution diagnostics: `program/Topology Evolution/train_dfl_cifar10_EvolutionBehaviour.py`
- CIFAR-100 benchmark: `program/Baseline_comparison_Cifar100/main.py`
- Google Speech Commands benchmark: `program/Baseline_comparison_gsc/main.py`
- DissDL baseline dependency: `program/dissdl.py`

The current repository does not track a Sentiment140 experiment entry point. Generated `.npy` files, generated plots, checkpoints, logs, and downloaded datasets are intentionally excluded from Git.

## Provenance Policy

Future result updates should include:

- the exact script and command used,
- seed list,
- dataset and split settings,
- topology settings,
- output file names,
- environment or dependency changes,
- whether results are regenerated or copied from an earlier run.

Do not edit reported numbers manually. If a figure or table changes, preserve the underlying output arrays or add a note explaining how the artifact was produced.
