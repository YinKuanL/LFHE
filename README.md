# LFHE

Local-First Heuristic Evolution (LFHE) adapts peer-to-peer learning topology using only local neighborhood information under bounded communication.

## Research Question

How can fully decentralized learners adapt communication topology using only local information while maintaining bounded communication under non-IID data?

## Method

LFHE treats the communication graph as part of the learning system. Each learner observes its ego graph, discovers friend-of-friend candidates, scores candidate updates with local structural and exploration signals, and then adds, swaps, or rejects edges under degree control.

The implementation preserves decentralized model averaging, Dirichlet non-IID client partitions, bounded-degree topology updates, and static baselines including ring, random, fully connected, FedAvg-reference, and DissDL-style comparison code.

![LFHE method overview](figures/lfhe_method_overview.png)

The overview figure shows the LFHE local-search loop: clients train and aggregate locally, discover friend-of-friend candidates, evaluate add/swap updates under degree control, and evolve the graph sequentially during training.

## Main Results

Centralized and dense references are useful upper bounds, but they solve a different communication problem from bounded decentralized learning. LFHE should be compared primarily with the decentralized sparse baselines.

| Method            |         CIFAR-10 |        CIFAR-100 |  Speech Commands |     Sentiment140 |
| ----------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| FedAvg            |     78.40 ± 0.80 |     51.90 ± 0.30 |     86.80 ± 0.50 |     70.60 ± 2.70 |
| Fully Connected   |     78.40 ± 0.70 |     51.80 ± 0.50 |     86.40 ± 1.40 |     70.60 ± 2.60 |
| Ring              |     52.60 ± 1.20 |     27.00 ± 0.80 |     53.00 ± 3.10 |     58.00 ± 2.70 |
| Static Random     |     67.40 ± 1.70 |     41.70 ± 1.20 |     74.20 ± 3.50 |     63.50 ± 2.30 |
| Epidemic Learning |     70.00 ± 1.60 |     44.10 ± 0.70 |     78.10 ± 3.30 |     65.50 ± 2.30 |
| DissDL            |     63.70 ± 2.10 |     40.20 ± 0.30 |     74.60 ± 3.70 |     64.20 ± 3.80 |
| **LFHE**          | **71.60 ± 2.10** | **45.20 ± 0.60** | **78.80 ± 2.30** | **65.30 ± 1.70** |

These results use five seeds. LFHE gives the strongest final decentralized result on CIFAR-10, CIFAR-100, and Speech Commands. On Sentiment140 it is competitive in final accuracy and reaches the target earlier than the decentralized alternatives.

<p>
  <img src="figures/lfhe_main_convergence_loss.png" alt="LFHE CIFAR-10 training loss" width="49%">
  <img src="figures/lfhe_main_convergence_accuracy.png" alt="LFHE CIFAR-10 test accuracy" width="49%">
</p>

The CIFAR-10 convergence panels are the two original panels referenced by the current manuscript for Figure 2. The historical `results/CIFAR10_10clients_1000rounds.png` artifact is not used as the current main-convergence figure.

| Dataset           | LFHE rounds-to-target |
| ----------------- | --------------------: |
| CIFAR-10 ≥70%     |                   225 |
| CIFAR-100 ≥35%    |                   125 |
| Speech ≥70%       |                    60 |
| Sentiment140 ≥65% |                   170 |

## Controlled Mechanism Evidence

Under matched friends-of-friends candidate generation and rewiring budgets, the mechanism study reports:

| Method           | Final λ2 | Final Acc. | Mean Divergence |
| ---------------- | -------: | ---------: | --------------: |
| Static           |    0.592 |     0.6675 |           0.766 |
| Similarity-only  |    0.578 |     0.6704 |           0.727 |
| Exploration-only |    1.038 |     0.7356 |           0.221 |
| Structural-only  |    2.306 |     0.7508 |           0.136 |
| LFHE             |    2.247 |     0.7482 |           0.133 |

This matched-protocol experiment provides evidence that objectives containing the structural term induce substantially stronger connectivity. It should not be read as evidence that the structural term directly optimizes algebraic connectivity.

![LFHE matched-budget bridge evidence](figures/lfhe_bridge_evidence.png)

## Topology Diagnostics

![LFHE topology evolution](figures/lfhe_topology_evolution.png)

This diagnostic is retained as supplementary topology-evolution evidence. The current manuscript uses the related topology-evolution figure family for CIFAR-10 with `alpha=0.1`; see the provenance notes for the exact boundary.

## Additional Results

![LFHE client-scale comparison](figures/lfhe_client_scale_comparison.png)

The client-scale artifact is retained as additional evidence about bounded topology adaptation as the decentralized system size changes, but scaling is not the main emphasis of this repository presentation.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Primary tracked entry points:

| Experiment | Entry point |
| --- | --- |
| CIFAR-10 main / seed study | `program/train_dfl_cifar10_seeds.py` |
| CIFAR-10 baseline comparison | `program/Baseline_comparison/main.py` |
| Dirichlet-alpha sensitivity | `program/Alpha_test/main.py` |
| Topology-evolution diagnostics | `program/Topology Evolution/train_dfl_cifar10_EvolutionBehaviour.py` |
| CIFAR-100 benchmark | `program/Baseline_comparison_Cifar100/main.py` |
| Google Speech Commands benchmark | `program/Baseline_comparison_gsc/main.py` |

The Sentiment140 manuscript experiment is not currently tracked as a public source entry point in this repository.

See [docs/reproducibility.md](docs/reproducibility.md) and [docs/results_provenance.md](docs/results_provenance.md) for the reproducibility boundary, tracked source inventory, and result-artifact policy.

## Repository Structure

```text
program/     Experiment entry points and topology implementations
docs/        Reproducibility and provenance notes
figures/     Curated README figures
results/     Historical tracked result artifacts
```

## Citation and License

Author-identifying citation metadata and publication-status wording are intentionally omitted while this artifact may be used in anonymous review. No license file is currently included.
