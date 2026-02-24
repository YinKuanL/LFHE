# LFHE: Local-First Heuristic Evolution for Decentralised Learning
Local-First Heuristic Evolution (LFHE) is a fully decentralised topology adaptation framework for peer-to-peer federated learning under Non-IID data.

This repository implements:
- Decentralised Federated Learning (DFL)
- Dirichlet-based Non-IID data partitioning
- Static communication topologies (Random / Ring / Fully-connected)
- Adaptive topology evolution via LFHE
- Multi-seed statistical evaluation on CIFAR-10

---

## Overview

In fully decentralised learning, communication topology significantly impacts convergence, robustness, and communication efficiency.
LFHE introduces:

- A **local fitness function** combining:
  - Connectivity proxy
  - Model similarity
  - Degree regularisation
- A **friend-of-friend (FoF)** exploration mechanism
- Bounded-degree adaptive rewiring
- Fully local, coordination-free topology evolution

The method avoids global spectral computation while promoting emergent structural refinement.

---

## 📁 Repository Structure
├── main.py # Main experiment script <br>
├── lfhe.py # LFHE topology update logic <br>
├── requirements.txt <br>
├── README.md <br>
└── data/ # Automatically created for CIFAR-10 <br>

---

## Experimental Setup
- Dataset: CIFAR-10
- Model: CNN with BatchNorm + Dropout
- Partition: Dirichlet Non-IID split
- Aggregation: Degree-weighted decentralised averaging
- Evaluation: Mean test accuracy across clients
- Multi-seed experiments supported

---

## Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/lfhe.git
cd lfhe
```

### 2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

### 3️⃣ Install dependencies
pip install -r requirements.txt

## Running Experiments
Run the main experiment:
```bash
python train_dfl_cifar10_seeds.py
```
By default: 
* 5 seeds
* CIFAR-10
* Dirichlet α sweep
* LFHE enabled
* 500 communication rounds

### Key Parameters

Inside run_experiment():
|	Parameter | Meaning |
| ------------- | ------------------------------- |
|   num_clients | Number of decentralised clients |
|     alpha     | Dirichlet Non-IID severity      |
|     rounds    | Communication rounds            |
| local_epochs  | Local SGD epochs                |    
| topo_interval | LFHE update interval            |
|  w1, w2, w3   | Fitness function weights        |

 ## Output
Saved artifacts include:
* non_iid_partition.png
* topology_at_seed_*.png
* test_cifar10_alpha*_lfhe.npy
* Accuracy history per seed

These can be used for:
* Mean ± std plots
* Convergence comparison
* Statistical testing
