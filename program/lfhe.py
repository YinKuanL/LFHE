import random
import numpy as np
import networkx as nx
import torch

# =====================================================
# Utility functions
# =====================================================

def cosine_similarity(model_i, model_j):
    rep_i = model_i.get_representation()
    rep_j = model_j.get_representation()
    return torch.nn.functional.cosine_similarity(rep_i, rep_j, dim=0).item()

def anneal_beta(round, beta0=1.0, kappa=0.01):
    """Exponential annealing schedule for beta"""
    return beta0 * np.exp(-kappa * round)

def local_regularised_proxy(graph, i, clients,
                            alpha=0.5, round=0):
    """
    F_i = alpha * d_tilde_i
          - anneal_beta(round) * (1/d_i) * sum_{j in N_i} (x_i - x_j)^2
    """

    neighbors = list(graph.neighbors(i))
    deg = len(neighbors)

    if deg == 0:
        return 0.0

    beta = anneal_beta(round)  # You can pass the current round if needed

    # ----- Degree term -----
    max_deg = max(dict(graph.degree()).values())
    d_tilde = deg / max(max_deg, 1)

    # ----- Local Laplacian energy -----
    wi = flatten(clients[i].model.state_dict())

    local_energy = 0.0
    for j in neighbors:
        wi = clients[i].model.get_representation()
        wj = clients[j].model.get_representation()
        diff = wi - wj
        local_energy += torch.mean(diff**2)

    local_energy /= deg

    print("d_tilde:", d_tilde)
    print("local_energy:", local_energy)
    print("beta:", beta)

    return alpha * d_tilde - beta * local_energy

def flatten(state_dict):
    return torch.cat([
        param.detach().flatten()
        for param in state_dict.values()
    ])

# =====================================================
# LFHE fitness
# =====================================================

def compute_fitness(i, graph, clients, D_max,
                    w1=1.0, w2=1.0, w3=1.0, round=0):
    """
    Local fitness f_i
    """

    # ----- connectivity proxy -----
    conn = local_regularised_proxy(
    graph=graph,
    i=i,
    clients=clients,
    alpha=0.1,
    round=round
)

    # ----- model similarity -----
    neighbors = list(graph.neighbors(i))
    if len(neighbors) == 0:
        sim = 0.0
    else:
        sims = [
        cosine_similarity(clients[i].model, clients[j].model)
        for j in neighbors
    ]
        sim = np.mean(sims)

    # ----- degree penalty -----
    deg = graph.degree(i) / D_max
    print("connectivity:", w1 *conn)
    print("similarity:", w2 * sim)
    print("degree_penalty:", w3 * deg)

    return w1 * conn + w2 * sim - w3 * deg

# =====================================================
# LFHE topology update
# =====================================================

def lfhe_update(graph, clients,
                epsilon=0.05,
                D_max=5,
                w1=1.0, w2=1.0, w3=0.1, round=0):

    G = graph.copy()
    num_clients = len(clients)

    for i in range(num_clients):

        Ni = list(G.neighbors(i))
        if len(Ni) == 0:
            continue

        # ------------------------------
        # Friend-of-Friend discovery
        # ------------------------------

        j = random.choice(Ni)
        Nj = list(G.neighbors(j))

        candidates = [
            k for k in Nj
            if k != i and not G.has_edge(i, k)
        ]

        if len(candidates) == 0:
            continue

        k = random.choice(candidates)

        # ------------------------------
        # Evaluate fitness
        # ------------------------------

        f_old = compute_fitness(i, G, clients, D_max=D_max, w1=w1, w2=w2, w3=w3, round=round)

        # ------------------------------
        # Atomic Update Logic (Check degree constraint first)
        # ------------------------------
        
        current_degree = G.degree(i)
        
        if current_degree < D_max:
            # Case 1: Simple addition
            G.add_edge(i, k)
            f_new = compute_fitness(i, G, clients, D_max=D_max, w1=w1, w2=w2, w3=w3, round=round)
            
            if f_new > f_old:
                print(f"Client {i}: Accepted edge ({i}, {k}) | f_old={f_old:.4f} f_new={f_new:.4f}")
            else:
                print(f"Client {i}: Rejected edge ({i}, {k}) | f_old={f_old:.4f} f_new={f_new:.4f}")
                G.remove_edge(i, k)
        
        else:
            # Case 2: Atomic Swap (Add k and Remove the worst existing neighbor)
            neighbors = list(G.neighbors(i))
            best_swap_fitness = -float("inf")
            worst_neighbor_to_remove = None

            # Try swapping k with each current neighbor
            for target_j in neighbors:
                # Simulate swap: -target_j, +k
                G.remove_edge(i, target_j)
                G.add_edge(i, k)
                
                f_swap = compute_fitness(i, G, clients, D_max=D_max, w1=w1, w2=w2, w3=w3, round=round)
                
                if f_swap > best_swap_fitness:
                    best_swap_fitness = f_swap
                    worst_neighbor_to_remove = target_j
                
                # Backtrack simulation
                G.remove_edge(i, k)
                G.add_edge(i, target_j)

            # Only execute if the best swap is better than keeping the current set
            if best_swap_fitness > f_old:
                G.remove_edge(i, worst_neighbor_to_remove)
                G.add_edge(i, k)
                print(f"Client {i}: Swapped ({i}, {worst_neighbor_to_remove}) with ({i}, {k}) | f_old={f_old:.4f} f_new={best_swap_fitness:.4f}")
            else:
                print(f"Client {i}: Rejected edge ({i}, {k}) | No beneficial swap found. f_old={f_old:.4f}")

    print(np.mean([d for n, d in G.degree()]))
    return G