# =================================================
# Standard libraries
# =================================================
import random
import numpy as np
import sys
import os

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lfhe import lfhe_update

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
        Return last-layer weights for topology similarity computation.
        Use this instead of flattening entire model.
        """
        return self.classifier[-1].weight.data.flatten()

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

    train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

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

        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, proportions)

        for i, split in enumerate(splits):
            client_indices[i].extend(split)

    return client_indices


def visualize_noniid_distribution(dataset, splits, num_clients):

    labels = dataset.targets.numpy()
    num_classes = labels.max() + 1

    label_counts = []

    for i in range(num_clients):
        client_labels = labels[splits[i]]
        counts = np.bincount(client_labels, minlength=num_classes)
        label_counts.append(counts)

    label_counts = np.array(label_counts)

    plt.figure(figsize=(10, 6))
    plt.imshow(label_counts, aspect='auto')
    plt.colorbar(label="Number of samples")
    plt.xlabel("Class label")
    plt.ylabel("Client ID")
    plt.title("Non-IID Dirichlet Partition (Heatmap View)")
    plt.savefig(f"non_iid_partition.png", dpi=300, bbox_inches='tight')


def compute_label_entropy(dataset, splits):

    # Make it robust to both tensor and list
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
    """
    Each client holds:
    - Local dataset
    - Local model
    - Local optimizer
    """
    def __init__(self, dataset, indices, lr=0.05, batch_size=32):
        self.loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True
        )

        # Each client maintains its own local model
        self.model = CNN().to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, epochs=1):
        """
        Perform local SGD updates.
        """
        self.model.train()
        for _ in range(epochs):
            for x, y in self.loader:
                x = x.to(device)
                y = y.to(device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_loader):
        """
        Evaluate local model on the global test set.
        Used only for logging / analysis.
        """
        self.model.eval()
        correct, total = 0, 0
        loss_sum = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                out = self.model(x)
                loss_sum += self.criterion(out, y).item()
                pred = out.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total, loss_sum / len(test_loader)


# =================================================
# DFL Topology utilities
# =================================================
def build_random_graph(num_clients, avg_degree=4):
    """
    Build a connected Erdős–Rényi random graph.
    Used as the static topology baseline.
    """
    p = avg_degree / (num_clients - 1)
    while True:
        G = nx.erdos_renyi_graph(num_clients, p)
        if nx.is_connected(G):
            return G


def average_models(models):
    """
    Simple parameter-wise averaging.
    Equivalent to decentralized FedAvg.
    """
    avg = {}
    for k in models[0]:
        avg[k] = sum(m[k] for m in models) / len(models)
    return avg


def decentralized_aggregation(clients, graph):

    new_states = []

    for i, c in enumerate(clients):
        neighbors = list(graph.neighbors(i))

        weights = []
        models = []

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



def visualize_graph(graph, title="Communication Topology"):
    plt.figure(figsize=(6, 6))

    pos = nx.spring_layout(graph, seed=42) 
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        font_size=10
    )
    
    plt.title(title)
    filename = title.lower().replace(" ", "_")
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')


# =================================================
# Topology Evolution Analysis for LFHE
# =================================================
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# =================================================
# Topology Metrics Calculation
# =================================================
def compute_lambda_2(graph):
    L = nx.laplacian_matrix(graph).toarray()
    
    # Use eigvalsh for symmetric matrices (real, stable)
    vals = np.linalg.eigvalsh(L)
    
    # Ensure sorted (eigvalsh already returns sorted ascending)
    if len(vals) > 1:
        return float(vals[1])
    else:
        return 0.0

def compute_topology_metrics(graph):

    metrics = {}

    # Algebraic Connectivity
    metrics['lambda_2'] = compute_lambda_2(graph)

    # Average degree
    degrees = [d for _, d in graph.degree()]
    metrics['avg_degree'] = np.mean(degrees)

    # Edge count
    metrics['num_edges'] = graph.number_of_edges()

    # Average path length
    if nx.is_connected(graph):
        metrics['avg_path_len'] = nx.average_shortest_path_length(graph)
    else:
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        metrics['avg_path_len'] = nx.average_shortest_path_length(subgraph)

    # Clustering
    metrics['avg_clustering'] = nx.average_clustering(graph)

    return metrics

# =================================================
# Modified Main Experiment Loop
# =================================================
def run_experiment(seed,
                   num_clients,
                   rounds=500,
                   local_epochs=5,
                   alpha=0.3,
                   topo_interval=5,
                   use_lfhe=False,
                   w1=1.0,   # Structural Connectivity weight
                   w2=1.0,   # Model Semantic Similarity weight
                   w3=0.01): # Degree Penalty weight
    
    set_seed(seed)
    train_set, test_set = load_dataset()
    test_loader = DataLoader(test_set, batch_size=256)
    splits = dirichlet_split(train_set, num_clients, alpha)
    
    clients = [Client(train_set, splits[i]) for i in range(num_clients)]

    # Initialize with a random connected graph
    graph = build_random_graph(num_clients)
    
    # Logs for evolution analysis
    topo_history = [] 
    acc_hist = []
    graph_history = graph

    for t in range(rounds):
        # Phase 1: Local Computation
        for c in clients:
            c.local_train(local_epochs)

        # Phase 2: Decentralized Weighted Aggregation
        decentralized_aggregation(clients, graph)

        # Phase 3: LFHE Topology Evolution
        # Triggered periodically after an initial warm-up period (t > 20)
        if use_lfhe and t % topo_interval == 0:
            
            # Update topology using Heuristic Evolution (Local-First)
            graph = lfhe_update(
                graph,
                clients,
                epsilon=0.05,
                D_max=6,
                w1=w1,
                w2=w2,
                w3=w3,
                round=t
            )
            
            # Record metrics after evolution
            current_metrics = compute_topology_metrics(graph)
            current_metrics['round'] = t
            topo_history.append(current_metrics)

        # Visual validation: Export graph plots at specific milestones
        if t in [0, 20, 50, 100, 200]:
            visualize_graph(graph, title=f"Topology_Round_{t}")

        # Phase 4: Performance Evaluation
        if t % 5 == 0:
            accs = []
            for c in clients:
                acc, _ = c.evaluate(test_loader)
                accs.append(acc)
            
            avg_acc = np.mean(accs)
            acc_hist.append(avg_acc)
            
            # Print status with Spectral Gap info
            l2 = compute_topology_metrics(graph)['lambda_2']
            print(f"[R{t:03d}] Acc: {avg_acc:.4f} | Lambda2: {l2:.4f}")
            
    return acc_hist, topo_history

def plot_combined_analysis(acc_hist, topo_history, alpha_val):
    rounds = [log['round'] for log in topo_history]
    lambda2 = [log['lambda_2'] for log in topo_history]
    clustering = [log['avg_clustering'] for log in topo_history]
    
    acc_to_plot = acc_hist[:len(rounds)] 

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_acc = 'tab:red'
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Global Accuracy', color=color_acc, fontsize=12)
    ax1.plot(rounds, acc_to_plot, color=color_acc, linewidth=2.5, label='Accuracy', marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color_l2 = 'tab:blue'
    color_clus = 'tab:green'
    
    ax2.set_ylabel('Topology Metrics ($\lambda_2$, Clustering)', color='black', fontsize=12)
    ax2.plot(rounds, lambda2, color=color_l2, linewidth=2, label='$\lambda_2$ (Connectivity)', linestyle='-')
    ax2.plot(rounds, clustering, color=color_clus, linewidth=2, label='Avg Clustering', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='black')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, shadow=True)

    plt.title(f'Relationship: Topology Evolution vs. Accuracy (Alpha={alpha_val})', fontsize=14)
    fig.tight_layout()
    
    filename = f"analysis_alpha{alpha_val}.png"
    plt.savefig(filename, dpi=300)
    print(f">>> Combined analysis plot saved as {filename}")
    plt.show()

# =================================================
# Plotting Utility for Evolution Trends
# =================================================
def plot_evolution_analysis(topo_history):
    """
    Generates a dual-axis plot showing Connectivity vs. Clustering evolution.
    """
    rounds = [log['round'] for log in topo_history]

    lambda2 = [log['lambda_2'] for log in topo_history]
    avg_degree = [log['avg_degree'] for log in topo_history]
    edges = [log['num_edges'] for log in topo_history]

    plt.figure(figsize=(10,6))

    plt.plot(rounds, lambda2, label="Lambda 2")
    plt.plot(rounds, avg_degree, label="Average Degree")
    plt.plot(rounds, edges, label="Edge Count")

    plt.xlabel("Rounds")
    plt.ylabel("Value")
    plt.title("Topology Evolution Debug")
    plt.legend()

    plt.grid(True)

    plt.savefig("topology_debug.png", dpi=300)
    plt.close()

def plot_from_npy(alpha):
    file_path = f"topo_metrics_alpha{alpha}.npy"
    try:
        topo_logs = np.load(file_path, allow_pickle=True)
        print(f"Successfully loaded {file_path}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    rounds = [log['round'] for log in topo_logs]
    lambda2 = [log['lambda_2'] for log in topo_logs]
    clustering = [log['avg_clustering'] for log in topo_logs]
    avg_degree = [log['avg_degree'] for log in topo_logs]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    color_l2 = 'tab:blue'
    ax1.set_xlabel('Communication Rounds', fontsize=12)
    ax1.set_ylabel('Algebraic Connectivity ($\lambda_2$)', color=color_l2, fontsize=12)
    lns1 = ax1.plot(rounds, lambda2, color=color_l2, linewidth=2, label='$\lambda_2$ (Connectivity)', marker='s', markersize=4, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_l2)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color_clus = 'tab:green'
    ax2.set_ylabel('Avg Clustering Coefficient', color=color_clus, fontsize=12)
    lns2 = ax2.plot(rounds, clustering, color=color_clus, linewidth=2, label='Avg Clustering', linestyle='--', marker='o', markersize=4, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_clus)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=True, shadow=True)

    plt.title(f'Topology Evolution Analysis (Alpha={alpha})\n$\lambda_2$ vs. Clustering Coefficient', fontsize=14)
    fig.tight_layout()

    save_name = f"topology_analysis_alpha{alpha}.png"
    plt.savefig(save_name)
    print(f"Plot saved as {save_name}")
    plt.show()

# =================================================
# Execution Script
# =================================================
if __name__ == "__main__":
    # Test across different heterogeneity levels
    # alpha=0.1 (Extreme Non-IID), alpha=1.0 (Near IID)
    test_alphas = [0.1] 
    
    for a in test_alphas:
        print(f"\n>>> Starting Topology Analysis for Alpha = {a}")
        
        _, topo_logs = run_experiment(
            seed=42,
            alpha=a,
            topo_interval=5,
            rounds=200,
            num_clients=30,
            use_lfhe=True,
            w1=1,   # Focus on global connectivity
            w2=1,  # Moderate semantic influence
            w3=0.1   # Enforce sparse degree
        )
        
        # Save raw logs and generate the analysis plot
        np.save(f"topo_metrics_alpha{a}.npy", topo_logs)
        plot_evolution_analysis(topo_logs)
        print(f">>> Analysis for Alpha {a} complete. Plot saved.")

    # plot_from_npy(0.1)