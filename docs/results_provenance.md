# Results Provenance

This page records what can be verified from files currently tracked in the repository.

## Tracked Result Artifacts

The repository currently tracks:

- `results/CIFAR10_10clients_1000rounds.npy`
- `results/CIFAR10_10clients_1000rounds.png`
- `results/beta_comparison.png`
- `results/client_scale_comparison.png`
- `results/experiments`
- `figures/lfhe_method_overview.png`, copied unchanged from the LFHE manuscript/source workspace.
- `figures/lfhe_main_convergence_loss.png`, copied unchanged from the LFHE manuscript/source workspace.
- `figures/lfhe_main_convergence_accuracy.png`, copied unchanged from the LFHE manuscript/source workspace.
- `figures/lfhe_bridge_evidence.png`, copied unchanged from the LFHE manuscript/source workspace.
- `figures/lfhe_client_scale_comparison.png`, copied from `results/client_scale_comparison.png` for README display.
- `figures/lfhe_topology_evolution.png`, copied from `program/Topology Evolution/topology_analysis_alpha0.1.png` for README display.

The README includes the manuscript-provided main benchmark table, rounds-to-target table, and matched-protocol mechanism table. These values were provided from the current manuscript source of truth for this staging pass and were copied without rounding or recomputation.

The tracked result artifacts support README figures for CIFAR-10 learning, topology evolution, beta/topology behavior, and client-scale comparison. Generated datasets, new `.npy` files, logs, checkpoints, and bulk result directories remain excluded from Git.

## Curated README Figure Map

| README figure | Copied unchanged from | Manuscript/result role | Notes |
| --- | --- | --- | --- |
| `figures/lfhe_method_overview.png` | local manuscript/source workspace `Program/images/overview.png` | manuscript method overview | Referenced by `LFHE_DynaFront_revised.tex` as `images/overview.png`. The newer reviewer-aligned source references `images/FoF.png`, but that standalone asset was not found locally. |
| `figures/lfhe_main_convergence_loss.png` | local manuscript/source workspace `Program/loss_plot.png` | manuscript Figure 2, CIFAR-10 training loss panel | Copied unchanged from the local manuscript/source workspace. |
| `figures/lfhe_main_convergence_accuracy.png` | local manuscript/source workspace `Program/mean_std_acc.png` | manuscript Figure 2, CIFAR-10 test accuracy panel | Copied unchanged from the local manuscript/source workspace. |
| `figures/lfhe_bridge_evidence.png` | local manuscript/source workspace `Program/Topology_Evolution/bridge_evidence_outputs/bridge_panels_alpha0.1.png` | manuscript Figure 10, matched-budget bridge evidence | Copied unchanged from the local manuscript/source workspace. |
| `figures/lfhe_topology_evolution.png` | `program/Topology Evolution/topology_analysis_alpha0.1.png` | supplementary topology-evolution diagnostic | Shows connectivity/clustering dynamics from the tracked topology-evolution entry point. The current manuscript references related topology panels as `images/topology_dual_alpha0.1.png` and `images/topology_snapshot.png`, not this exact README asset. |
| `figures/lfhe_client_scale_comparison.png` | `results/client_scale_comparison.png` | additional scaling evidence | Retained outside the main narrative because the workshop repository carries the scaling emphasis. |

The historical artifact `results/CIFAR10_10clients_1000rounds.png` is retained in the repository as historical material but is not presented as the current manuscript's main convergence figure.

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
