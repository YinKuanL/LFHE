# =================================================
# Standard libraries
# =================================================
import os
import random
import numpy as np
import copy

# =================================================
# PyTorch core libraries
# =================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# =================================================
# Dataset utilities
# =================================================
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# =================================================
# Graph / topology utilities
# =================================================
import networkx as nx
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lfhe import lfhe_update
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dissdl import DissDLNode

# =================================================
# Device configuration (CPU / GPU)
# =================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =================================================
# Reproducibility
# =================================================
def set_seed(seed):
    """
    Fix all random seeds for reproducibility.
    Important for multi-seed statistical evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =================================================
# Improved CNN for CIFAR-10 (Non-IID + DFL Stable)
# =================================================
class CNN(nn.Module):
    """
    Stabilized CNN for CIFAR-10 under Non-IID decentralized FL.
    Includes BatchNorm, Dropout, and representation extraction.
    """
    def __init__(self):
        super().__init__()

        # -------- Feature extractor --------
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )

        # Reduce dimensionality
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 64x4x4

        # -------- Classifier --------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    # ------------------------------------------------
    # Representation for LFHE similarity
    # ------------------------------------------------
    def get_representation(self):
        """
        Return a reduced-dimension representation.
        We average the 256 dimensions to get 10 class-centric features,
        significantly reducing noise during LFHE similarity computation.
        """
        return self.classifier[-1].weight.data.mean(dim=1).flatten()

# =================================================
# Dataset loading
# =================================================
def load_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    return train, test

# =================================================
# Non-IID data partition (Dirichlet split)
# =================================================
def dirichlet_split(dataset, num_clients, alpha):
    """
    Partition dataset into Non-IID subsets using Dirichlet distribution.
    alpha ↓  => more heterogeneous data
    alpha ↑  => closer to IID
    """
    labels = np.array(dataset.targets)
    num_classes = labels.max() + 1

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, proportions)

        for i, split in enumerate(splits):
            client_indices[i].extend(split)

    return client_indices

def compute_label_entropy(dataset, splits):
    if isinstance(dataset.targets, torch.Tensor):
        labels = dataset.targets.cpu().numpy()
    else:
        labels = np.array(dataset.targets)

    entropies = []
    for split in splits:
        client_labels = labels[split]
        probs = np.bincount(client_labels) / len(client_labels)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)

    return np.mean(entropies)

# =================================================
# Client definition
# =================================================
class Client:
    def __init__(self, dataset, indices, lr=0.05, batch_size=32):
        self.loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
        self.model = CNN().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_loader):
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                loss_sum += self.criterion(out, y).item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total, loss_sum / len(test_loader)

# =================================================
# DFL Topology utilities
# =================================================
def build_fixed_random_graph(num_clients, avg_degree=4, fixed_seed=42):
    """
    Generates a connected Erdos-Renyi graph using a dedicated 
    random state. This guarantees 'random' and 'lfhe' start with the EXACT same graph.
    """
    p = avg_degree / (num_clients - 1)
    rng = np.random.RandomState(fixed_seed)
    while True:
        G = nx.erdos_renyi_graph(num_clients, p, seed=int(rng.randint(0, 1e6)))
        if nx.is_connected(G):
            return G

def decentralized_aggregation(clients, graph):
    new_states = []
    for i, c in enumerate(clients):
        neighbors = list(graph.neighbors(i))
        weights, models = [], []

        for j in neighbors:
            deg_i = graph.degree(i)
            deg_j = graph.degree(j)
            w_ij = 1 / (1 + max(deg_i, deg_j))
            weights.append(w_ij)
            models.append(clients[j].model.state_dict())

        w_ii = 1 - sum(weights)
        weights.append(w_ii)
        models.append(c.model.state_dict())

        avg = {}
        for k in models[0]:
            avg[k] = sum(w * m[k] for w, m in zip(weights, models))
        new_states.append(avg)

    for c, s in zip(clients, new_states):
        c.model.load_state_dict(s)

def build_ring_graph(num_clients):
    G = nx.Graph()
    for i in range(num_clients):
        G.add_edge(i, (i + 1) % num_clients)
    return G

def build_fully_connected_graph(num_clients):
    return nx.complete_graph(num_clients)

def build_static_mh_graph(num_clients, avg_degree=4, seed=42):
    return build_fixed_random_graph(num_clients, avg_degree, seed)

def build_dissdl_graph(num_clients, degree=3):
    graph = {}
    for i in range(num_clients):
        candidates = list(range(num_clients))
        candidates.remove(i)
        graph[i] = random.sample(candidates, degree)
    return graph

# =================================================
# FedAvg (Centralized Aggregation)
# =================================================
def fedavg_aggregation(clients):
    """
    Standard FedAvg aggregation.
    All clients send models to server -> average -> broadcast back.
    """
    global_state = {}

    # collect models
    models = [c.model.state_dict() for c in clients]

    for k in models[0]:
        global_state[k] = sum(m[k] for m in models) / len(models)

    # broadcast back
    for c in clients:
        c.model.load_state_dict(global_state)

# =================================================
# Main experiment loop
# =================================================
def run_experiment(seed, num_clients, rounds=200, topology_type="random", 
                   local_epochs=1, alpha=0.3, topo_interval=3, 
                   w1=1.0, w2=1.0, w3=0.01):
    
    
    set_seed(seed)
    train_set, test_set = load_dataset()
    test_loader = DataLoader(test_set, batch_size=256)

    # Force synchronization right before split
    np.random.seed(seed) 
    splits = dirichlet_split(train_set, num_clients, alpha)
    entropy = compute_label_entropy(train_set, splits)
    print(f"[{topology_type}] Average client entropy: {entropy:.4f}")

    clients = [Client(train_set, splits[i]) for i in range(num_clients)]

    # Initial topology
    if topology_type == "ring":
        graph = build_ring_graph(num_clients)

    elif topology_type == "fully":
        graph = build_fully_connected_graph(num_clients)

    elif topology_type == "random":
        graph = build_fixed_random_graph(num_clients, fixed_seed=seed)

    elif topology_type == "static_mh":
        graph = build_static_mh_graph(num_clients, seed=seed)

    elif topology_type == "dissdl":
        graph = build_dissdl_graph(num_clients)

    elif topology_type == "lfhe":
        graph = build_fixed_random_graph(num_clients, fixed_seed=seed)

    elif topology_type == "fedavg":
        graph = None  # FedAvg does not use graph

    else:
        raise ValueError("Unknown topology type.")

    stats = {
        "acc_all_nodes": [], 
        "loss_all_nodes": [],
        "inter_node_var": [] 
    }

    dissdl_nodes = []
    if topology_type == "dissdl":
        for i in range(num_clients):
            init_neighbors = list(graph[i])
            node = DissDLNode(
                node_id=i,
                model=clients[i].model,
                neighbors=init_neighbors,
                beta=1.0
            )
            node.known_peers = set(range(num_clients)) - {i}
            dissdl_nodes.append(node)

    for t in range(rounds):
        # 1. Local training
        for c in clients:
            c.local_train(local_epochs)

        # 2. Aggregation & Topology Update
        if topology_type == "fedavg":
            fedavg_aggregation(clients)

        elif topology_type == "dissdl":
            for j in range(num_clients):
                node_j = dissdl_nodes[j]
                for i in node_j.wanted_senders:
                    node_j.received_models[i] = copy.deepcopy(clients[i].model)

            for i in range(num_clients):
                dissdl_nodes[i].aggregate()
                clients[i].model.load_state_dict(dissdl_nodes[i].model.state_dict())
                if t % topo_interval == 0:
                    dissdl_nodes[i].update_wanted_senders()
            
            new_G = nx.DiGraph()
            for i in range(num_clients):
                for s in dissdl_nodes[i].wanted_senders:
                    new_G.add_edge(s, i)
            graph = new_G

        elif topology_type == "lfhe":
            decentralized_aggregation(clients, graph)
            if t % topo_interval == 0:
                graph = lfhe_update(
                    graph, clients, epsilon=0.05, D_max=4,
                    w1=w1, w2=w2, w3=w3, round=t
                )
        else:
            decentralized_aggregation(clients, graph)

        if t % 5 == 0:    
            # Evaluation (every round for plotting consistency)
            current_accs = []
            current_losses = []
            all_params = []
        
            for c in clients:
                acc, loss = c.evaluate(test_loader)
                current_accs.append(acc)
                current_losses.append(loss)
                
                flat_params = torch.cat([p.data.view(-1) for p in c.model.parameters()])
                all_params.append(flat_params)

            all_params_stack = torch.stack(all_params) # [num_clients, num_params]
            mean_params = torch.mean(all_params_stack, dim=0)
            variance = torch.mean(torch.norm(all_params_stack - mean_params, dim=1)**2).item()
            
            stats["acc_all_nodes"].append(current_accs)
            stats["loss_all_nodes"].append(current_losses)
            stats["inter_node_var"].append(variance)

        if t % 10 == 0 or t == rounds - 1:
            current_avg_acc = np.mean(stats["acc_all_nodes"][-1])
            print(f"[{topology_type.upper()} - Seed {seed}] Round {t:03d} | Acc={current_avg_acc:.4f}")

    return stats

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # =================================================
    # Alpha Experiment Configuration
    # =================================================
    seeds = [42, 43, 44]
    num_clients = 30
    rounds = 200
    topologies = ["ring", "random", "dissdl", "lfhe"]
    alphas = [0.1, 0.5, 1.0]
    
    plot_data = {}

    for alpha in alphas:
        print(f"\n{'='*40}")
        print(f"Targeting Alpha: {alpha}")
        print(f"{'='*40}")

        for topo in topologies:
            final_accs_for_topo = []
            
            for seed in seeds:
                file_name = f"result_alpha{alpha}_{topo}_seed{seed}.npy"
                
                if os.path.exists(file_name):
                    print(f">>> Skipping: {file_name} already exists. Loading...")
                    acc_curve = np.load(file_name)
                else:
                    print(f"Running: Alpha={alpha} | Topo={topo} | Seed={seed}")
                    stats = run_experiment(
                        seed=seed,
                        num_clients=num_clients,
                        rounds=rounds,
                        topology_type=topo,
                        alpha=alpha,
                        local_epochs=1,
                        topo_interval=5,
                        w1=1.0, w2=1.0, w3=0.1
                    )

                    acc_curve = np.mean(stats["acc_all_nodes"], axis=1)
                    np.save(file_name, acc_curve)
                    
                    np.save(f"stats_alpha{alpha}_{topo}_seed{seed}.npy", stats)

                final_accs_for_topo.append(acc_curve[-1])
            
            plot_data[(alpha, topo)] = final_accs_for_topo
            avg_val = np.mean(final_accs_for_topo)
            print(f"Done: Alpha {alpha} | {topo.upper()} | Avg Final Acc: {avg_val:.4f}")

    # =================================================
    # Final Alpha Plotting: mean ± std accuracy vs alpha
    # =================================================
    colors = {
        "ring": "tab:red",
        "random": "tab:blue",
        "dissdl": "tab:orange",
        "lfhe": "tab:purple"
    }

    labels = {
        "ring": "Ring D-PSGD",
        "random": "Static Random",
        "dissdl": "DissDL",
        "lfhe": "LFHE (Ours)"
    }

    # Combined plot: all topologies
    plt.figure(figsize=(9, 6))

    for topo in topologies:
        means = []
        stds = []

        for alpha in alphas:
            vals = plot_data.get((alpha, topo), [])
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            stds.append(np.std(vals) if len(vals) > 0 else np.nan)

        plt.errorbar(
            alphas,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.2,
            capsize=4,
            label=labels.get(topo, topo),
            color=colors.get(topo, None)
        )

    plt.xlabel("Dirichlet Alpha", fontsize=12)
    plt.ylabel("Final Test Accuracy", fontsize=12)
    plt.title(f"Mean ± Std Final Accuracy vs Alpha (Rounds={rounds})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    combined_save_path = "alpha_experiment_all_topologies.png"
    plt.savefig(combined_save_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined alpha plot: {combined_save_path}")
    plt.close()

    # One plot per topology
    for topo in topologies:
        means = []
        stds = []

        for alpha in alphas:
            vals = plot_data.get((alpha, topo), [])
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            stds.append(np.std(vals) if len(vals) > 0 else np.nan)

        plt.figure(figsize=(8, 6))
        plt.errorbar(
            alphas,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=colors.get(topo, None),
            label=labels.get(topo, topo)
        )

        plt.xlabel("Dirichlet Alpha", fontsize=12)
        plt.ylabel("Final Test Accuracy", fontsize=12)
        plt.title(f"Mean ± Std Final Accuracy vs Alpha: {labels.get(topo, topo)}", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        save_path = f"alpha_experiment_{topo}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot: {save_path}")
        plt.close()
