# =================================================
# Standard libraries
# =================================================
import random
import numpy as np

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
# Main experiment loop
# =================================================
def run_experiment(seed,
                   num_clients,
                   rounds=200,
                   local_epochs=1,
                   alpha=0.3,
                   topo_interval=3,
                   use_lfhe=False,
                   w1=1.0,  
                   w2=1.0,
                   w3=0.01):
    """
    Run one DFL experiment under a fixed random seed.

    use_lfhe = False -> static random topology (baseline)
    use_lfhe = True  -> LFHE adaptive topology
    """
    set_seed(seed)

    train_set, test_set = load_dataset()
    test_loader = DataLoader(test_set, batch_size=256)

    splits = dirichlet_split(train_set, num_clients, alpha)
    entropy = compute_label_entropy(train_set, splits)
    print("Average client entropy:", entropy)

    if seed == 0:
        visualize_noniid_distribution(train_set, splits, num_clients)


    clients = [
        Client(train_set, splits[i])
        for i in range(num_clients)
    ]

    # Initial topology
    graph = build_random_graph(num_clients)
    visualize_graph(graph, title=f"Topology at Seed {seed}")

    acc_hist, loss_hist = [], []

    for t in range(rounds):
        # ----- Local training -----
        for c in clients:
            c.local_train(local_epochs)

        # ----- Decentralized aggregation -----
        decentralized_aggregation(clients, graph)

        # ----- LFHE topology update -----
        if use_lfhe and t % topo_interval == 0 and t > 40:

            graph = lfhe_update(
                graph,
                clients,
                epsilon=0.05,
                D_max=4,
                w1=w1,
                w2=w2,
                w3=w3,
                round=t
            )

        # ----- Evaluation -----
        accs, losses = [], []
        for c in clients:
            acc, loss = c.evaluate(test_loader)
            accs.append(acc)
            losses.append(loss)

        acc_hist.append(np.mean(accs))
        loss_hist.append(np.mean(losses))

        print(f"[Seed {seed}] Round {t:03d} | Acc={acc_hist[-1]:.4f}")

    return acc_hist, loss_hist


# =================================================
# Multi-seed evaluation
# =================================================
if __name__ == "__main__":

    seeds = [42, 43, 44, 45, 46]
    num_clients_list = [10]
    top_list = [5]

    results = {}

    for top in top_list:

        print("\n======================================")
        print(f"Running for top_interval = {top}")
        print("======================================")

        acc_baseline_all = []
        acc_lfhe_all = []

        for s in seeds:
            print(f"\nSeed {s} | Baseline")
            acc_b, _ = run_experiment(
                seed=s,
                rounds=500,
                num_clients=10,
                use_lfhe=False,
                topo_interval=top,
                w1=1.0,
                w2=0.05,
                w3=0.02
            )

            print(f"\nSeed {s} | LFHE")
            acc_i, _ = run_experiment(
                seed=s,
                rounds=500,
                num_clients=10,
                use_lfhe=True,
                topo_interval=top,
                w1=1.0,
                w2=0.05,
                w3=0.02
            )
            
            np.save(f"test_cifar10_seed{s}_10clients_5top_baseline.npy", acc_b)
            np.save(f"test_cifar10_seed{s}_10clients_5top_lfhe.npy", acc_i)

    