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
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS
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
    CNN for Google Speech Commands (log-mel spectrogram input)
    Input: (1, 64, ~100)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def get_representation(self):
        return self.classifier[-1].weight.data.flatten()

# =================================================
# Dataset loading
# =================================================
def load_dataset():

    selected_labels = [
        "yes", "no", "up", "down", "left",
        "right", "on", "off", "stop", "go"
    ]

    label_to_index = {label: i for i, label in enumerate(selected_labels)}

    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_mels=64
    )

    amplitude_to_db = T.AmplitudeToDB()

    train_set = FilteredSpeechCommands(
        "training",
        label_to_index,
        mel_transform,
        amplitude_to_db
    )

    test_set = FilteredSpeechCommands(
        "testing",
        label_to_index,
        mel_transform,
        amplitude_to_db
    )

    return train_set, test_set

import torch
import torch.nn.functional as F

# =================================================
# Dataset class (MUST be global for multiprocessing)
# =================================================
class FilteredSpeechCommands(SPEECHCOMMANDS):
    def __init__(self, subset, label_to_index, mel_transform, amplitude_to_db):
        root_path = os.path.abspath("./data")
        if not os.path.exists(root_path):
            os.makedirs(root_path)
            
        super().__init__(root_path, download=True, subset=subset)

        self.label_to_index = label_to_index
        self.mel_transform = mel_transform
        self.amplitude_to_db = amplitude_to_db

        self.indices = []
        self.targets = []
        for i in range(len(self._walker)):
            path = self._walker[i]
            label = path.split(os.sep)[-2]
            if label in label_to_index:
                self.indices.append(i)
                self.targets.append(label_to_index[label])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, n):
        waveform, sample_rate, label, *_ = super().__getitem__(self.indices[n])

        waveform = self.mel_transform(waveform)
        waveform = self.amplitude_to_db(waveform)

        return waveform, self.label_to_index[label]

def pad_collate(batch):
    xs, ys = zip(*batch)

    # Find max time length
    max_len = max(x.shape[-1] for x in xs)

    padded_xs = []
    for x in xs:
        pad_amount = max_len - x.shape[-1]
        if pad_amount > 0:
            x = F.pad(x, (0, pad_amount))  # pad time dimension
        padded_xs.append(x)

    return torch.stack(padded_xs), torch.tensor(ys)

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
        subset = Subset(dataset, indices)

        self.loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=pad_collate 
        )

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
# Plot / Analysis Helpers
# =================================================
LABELS = {
    "fedavg": "FedAvg (Centralized)",
    "fully": "Fully Connected",
    "ring": "Ring D-PSGD",
    "static_mh": "Static MH",
    "random": "Static Random",
    "dissdl": "DissDL",
    "lfhe": "LFHE (Ours)"
}

COLORS = {
    "fedavg": "tab:brown",
    "fully": "black",
    "ring": "tab:red",
    "static_mh": "tab:green",
    "random": "tab:blue",
    "dissdl": "tab:orange",
    "lfhe": "tab:purple"
}


def load_accuracy_results(topologies, seeds):
    """
    Load all saved accuracy arrays from gsc_{topo}_all.npy
    Returns:
        results[topo] = ndarray of shape [num_seeds, rounds]
    """
    results = {}
    for topo in topologies:
        file_path = f"gsc_{topo}_all.npy"
        if os.path.exists(file_path):
            data = np.load(file_path)
            if data.ndim == 1:
                data = data[None, :]   # convert to [1, rounds]
            results[topo] = data
    return results


def load_loss_results(topologies, seeds):
    """
    Load per-seed loss curves from stats_{topo}_seed{s}.npy
    Returns:
        loss_results[topo] = ndarray of shape [num_seeds, rounds]
    """
    loss_results = {}

    for topo in topologies:
        topo_losses = []

        for s in seeds:
            stats_file = f"stats_{topo}_seed{s}.npy"
            if not os.path.exists(stats_file):
                continue

            stats = np.load(stats_file, allow_pickle=True).item()

            if "loss_all_nodes" in stats:
                # expected shape [rounds, clients]
                loss_curve = np.mean(stats["loss_all_nodes"], axis=1)
            elif "loss" in stats:
                loss_curve = np.array(stats["loss"])
            else:
                print(f"[Warning] No loss found in {stats_file}, skipping.")
                continue

            topo_losses.append(loss_curve)

        if len(topo_losses) > 0:
            loss_results[topo] = np.array(topo_losses)

    return loss_results


def plot_multi_seed_accuracy(results, save_path="multi_seed_acc.png", eval_interval=5):
    plt.figure(figsize=(11, 7))

    for topo, data in results.items():
        color = COLORS.get(topo, None)
        label = LABELS.get(topo, topo)

        rounds = np.arange(data.shape[1]) * eval_interval

        for i in range(data.shape[0]):
            plt.plot(rounds, data[i], color=color, alpha=0.25, linewidth=1.2)

        mean_acc = np.mean(data, axis=0)
        plt.plot(rounds, mean_acc, label=label, color=color, linewidth=2.5)

    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Multi-Seed Accuracy Curves", fontsize=14)
    plt.legend(fontsize=10, ncol=2, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_mean_std_accuracy(results, save_path="mean_std_acc.png", eval_interval=5):
    plt.figure(figsize=(11, 7))

    for topo, data in results.items():
        color = COLORS.get(topo, None)
        label = LABELS.get(topo, topo)

        mean_acc = np.mean(data, axis=0)
        std_acc = np.std(data, axis=0)

        rounds = np.arange(len(mean_acc)) * eval_interval

        plt.plot(rounds, mean_acc, label=label, color=color, linewidth=2.5)
        plt.fill_between(
            rounds,
            mean_acc - std_acc,
            mean_acc + std_acc,
            color=color,
            alpha=0.18
        )

    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Mean ± Std Test Accuracy", fontsize=14)
    plt.legend(fontsize=10, ncol=2, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_curves(loss_results, save_path="loss_plot.png", eval_interval=5):
    plt.figure(figsize=(11, 7))

    plotted = 0
    for topo, data in loss_results.items():
        color = COLORS.get(topo, None)
        label = LABELS.get(topo, topo)

        mean_loss = np.mean(data, axis=0)
        std_loss = np.std(data, axis=0)
        rounds = np.arange(len(mean_loss)) * eval_interval

        plt.plot(rounds, mean_loss, label=label, color=color, linewidth=2.5)
        plt.fill_between(
            rounds,
            mean_loss - std_loss,
            mean_loss + std_loss,
            color=color,
            alpha=0.18
        )
        plotted += 1

    if plotted == 0:
        print("No loss data found to plot.")
        plt.close()
        return

    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Mean ± Std Loss", fontsize=14)
    plt.legend(fontsize=10, ncol=2, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def convergence_test(results, target_acc=0.7, window=5, threshold=1e-3, eval_interval=5):
    summary = {}

    print("\n================ Convergence / Target Test ================")

    for topo, data in results.items():
        mean_curve = np.mean(data, axis=0)

        target_round = None
        reached = np.where(mean_curve >= target_acc)[0]
        if len(reached) > 0:
            target_round = int(reached[0] * eval_interval)

        converge_round = None
        diffs = np.abs(np.diff(mean_curve))

        for i in range(len(diffs) - window + 1):
            local_window = diffs[i:i + window]
            if np.mean(local_window) < threshold:
                converge_round = int((i + 1) * eval_interval)
                break

        summary[topo] = {
            "target_round": target_round,
            "converge_round": converge_round,
            "final_acc": float(mean_curve[-1])
        }

        print(
            f"{LABELS.get(topo, topo):25s} | "
            f"Round@acc>={target_acc}: {target_round} | "
            f"Converge round: {converge_round} | "
            f"Final Acc: {mean_curve[-1]:.4f}"
        )

    return summary

# =================================================
# Main experiment loop
# =================================================
def run_experiment(seed, num_clients, rounds=200, topology_type="random", 
                   local_epochs=1, alpha=0.3, topo_interval=3, 
                   w1=1.0, w2=1.0, w3=0.01):
    
    
    set_seed(seed)
    train_set, test_set = load_dataset()
    test_loader = DataLoader(
        test_set, 
        batch_size=256, 
        shuffle=False, 
        collate_fn=pad_collate
    )

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

# =================================================
# Execution and Plotting (With Auto-Skip Logic)
# =================================================
if __name__ == "__main__":
    seeds = [42, 43, 44, 45, 46]
    num_clients = 30
    rounds_to_run = 100  
    
    topologies_to_run = [
        "fedavg",
        "fully",
        "ring",
        "static_mh",
        "random",
        "dissdl",
        "lfhe"
    ]
    results = {}

    for topo in topologies_to_run:
        print("\n======================================")
        print(f"Targeting topology: {topo}")
        print("======================================")
        
        all_summary_file = f"gsc_{topo}_all.npy"
        
        missing_seeds = []
        for s in seeds:
            if not os.path.exists(f"gsc_{topo}_seed{s}.npy"):
                missing_seeds.append(s)
        
        if len(missing_seeds) == 0:
            print(f">>> Skipping {topo}: All seeds already exist.")
            if os.path.exists(all_summary_file):
                results[topo] = np.load(all_summary_file)
            else:
                acc_all = [np.load(f"gsc_{topo}_seed{s}.npy") for s in seeds]
                results[topo] = np.array(acc_all)
                np.save(all_summary_file, results[topo])
            continue

        acc_all = []
        for s in seeds:
            seed_file = f"gsc_{topo}_seed{s}.npy"
            
            if os.path.exists(seed_file):
                print(f">>> Seed {s} for {topo} already exists. Loading existing results.")
                round_avg_acc = np.load(seed_file)
            else:
                print(f"Starting Experiment: {topo} | Seed: {s}")
                stats = run_experiment(
                    seed=s,
                    num_clients=num_clients,
                    topology_type=topo,
                    rounds=rounds_to_run,
                    alpha=0.1,
                    topo_interval=5,
                    w1=1.0, w2=1.0, w3=0.1
                )
                round_avg_acc = np.mean(stats["acc_all_nodes"], axis=1)
            
                np.save(seed_file, round_avg_acc)
                np.save(f"stats_{topo}_seed{s}.npy", stats)
            
            acc_all.append(round_avg_acc)

        results[topo] = np.array(acc_all)
        np.save(all_summary_file, results[topo])
        print(f"Successfully finalized and saved: {topo}")

    # =================================================
    # Final Plot Generation
    # =================================================
    print("\nGenerating final plots...")

    loss_results = load_loss_results(topologies_to_run, seeds)

    plot_multi_seed_accuracy(results, save_path="multi_seed_acc.png", eval_interval=5)
    plot_mean_std_accuracy(results, save_path="mean_std_acc.png", eval_interval=5)
    plot_loss_curves(loss_results, save_path="loss_plot.png", eval_interval=5)
    convergence_summary = convergence_test(results, target_acc=0.7, window=5, threshold=1e-3, eval_interval=5)
