"""SubgraphX explanations for the trained GAT, per seed x fold x subject.

For every correctly classified subject (optionally all subjects) SubgraphX runs
a Monte-Carlo Tree Search guided by an MC L-Shapley value function to find the
most important node subset of size N_MIN. Each subject is repeated R_REPEATS
times and the most frequent node set is kept, together with fidelity / sparsity
metrics. Results are written per subject and per seed/class, plus a .pth bundle.

Self-contained: keeps its own GAT definition (matching the trained checkpoints)
and SubgraphX implementation. Paths come from ``config``.
"""

from __future__ import annotations

import glob
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from enum import Enum
from typing import Callable, Dict, List, Set, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, LayerNorm, global_mean_pool
from torch_geometric.utils import k_hop_subgraph, softmax, subgraph, to_networkx
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# =========================
# CONFIG
# =========================
RESULTS_DIR = str(config.GATE_RESULTS_DIR)
SEEDS = config.SEEDS
OUT_DIR = os.path.join(str(config.XAI_RESULTS_DIR), "subgraphx_outputs")
SUBGRAPH_PTH = os.path.join(OUT_DIR, "subgraphx_subject_subgraphs.pth")

TP_TN_ONLY = True     # True: correctly classified (TP+TN) only; False: all subjects
R_REPEATS = 5         # SubgraphX repeats per subject
N_MIN = 10            # explanation size (node subset)
THR = 0.5
DO_PLOT_FINAL = False  # set True to plot each subject's explanation subgraph

DEVICE = torch.device("cpu")

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def find_seed_file(results_dir, seed):
    files = sorted(glob.glob(os.path.join(results_dir, f"all_results_*_seed{seed}.pth")))
    if not files:
        raise FileNotFoundError(f"No .pth for seed={seed} in {results_dir}")
    return files[0]


def most_frequent_nodeset(list_of_sets):
    keys = [tuple(sorted(s)) for s in list_of_sets]
    c = Counter(keys)
    best_key, best_cnt = c.most_common(1)[0]
    return set(best_key), int(best_cnt), c


def summarize_metric(vec):
    vec = np.asarray(vec, float)
    vec = vec[np.isfinite(vec)]
    if len(vec) == 0:
        return dict(mean=np.nan, median=np.nan, std=np.nan, n=0)
    return dict(mean=float(vec.mean()), median=float(np.median(vec)),
                std=float(vec.std(ddof=0)), n=int(len(vec)))


class CountedModel(torch.nn.Module):
    """Wrap a model and count forward passes (SubgraphX cost diagnostics)."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.n_forwards = 0

    def forward(self, *args, **kwargs):
        self.n_forwards += 1
        return self.base_model(*args, **kwargs)


# =========================
# GAT (matches the trained nested-CV model)
# =========================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, graph_feat_dim,
                 num_layers=2, heads=1, use_layernorm=False, concat=True, pool="attn"):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.use_layernorm = use_layernorm
        self.dropout = dropout
        self.concat = concat
        self.heads = heads

        out_dim = hidden_channels * heads if concat else hidden_channels

        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads,
                                    dropout=dropout, concat=concat))
        if use_layernorm:
            self.norms.append(LayerNorm(out_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_dim, hidden_channels, heads=heads,
                                        dropout=dropout, concat=concat))
            if use_layernorm:
                self.norms.append(LayerNorm(out_dim))

        self.pool = pool
        if pool == "attn":
            self.gate_nn = torch.nn.Sequential(
                torch.nn.Linear(out_dim + graph_feat_dim, out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(out_dim, 1),
            )
        self.fc = torch.nn.Linear(out_dim, 1)

    def forward(self, x, edge_index, graph_feat, batch=None,
                return_attn=False, return_pool_weights=False):
        attn_weights_list = []
        for i, conv in enumerate(self.layers):
            if return_attn:
                x, (edge_index_out, attn_weights) = conv(
                    x, edge_index, return_attention_weights=True)
                attn_weights_list.append((edge_index_out, attn_weights))
            else:
                x = conv(x, edge_index)
            if self.use_layernorm:
                x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pool_weights = None
        if self.pool == "attn":
            c_per_node = graph_feat[batch]
            gate = self.gate_nn(torch.cat([x, c_per_node], dim=1)).view(-1)
            alpha = softmax(gate, batch)
            if return_pool_weights:
                pool_weights = alpha
            x = x * alpha.unsqueeze(-1)
            batch_size = int(batch.max().item()) + 1
            x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)
            x_pooled.index_add_(0, batch, x)
            x = x_pooled
        else:
            x = global_mean_pool(x, batch)

        out = self.fc(x)
        if return_attn and return_pool_weights:
            return out, attn_weights_list, pool_weights
        if return_attn:
            return out, attn_weights_list
        if return_pool_weights:
            return out, pool_weights
        return out


class Task(Enum):
    GRAPH_CLASSIFICATION = 1
    NODE_CLASSIFICATION = 2
    LINK_PREDICTION = 3


class Experiment(Enum):
    DEFAULT = 1
    GREEDY = 2


# =========================
# GRAPH PREP / SPLIT
# =========================
def prepare_graph(g: Data, best_node_mask: torch.Tensor, best_graph_feat_mask: torch.Tensor) -> Data:
    """Apply node/graph-feature masks and ensure graph-level features and batch."""
    g = g.clone()

    if hasattr(g, "x") and g.x is not None:
        if g.x.dim() == 2 and g.x.size(1) == len(best_node_mask):
            g.x = g.x[:, best_node_mask]

    if (not hasattr(g, "batch")) or (g.batch is None):
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long)

    if not hasattr(g, "graph_feat") or g.graph_feat is None:
        raise RuntimeError("graph.graph_feat is None (expected graph-level features).")

    gf = g.graph_feat
    if gf.dim() == 1:
        gf = gf.view(1, -1)
    elif gf.dim() == 2:
        if gf.size(0) == g.num_nodes:   # accidental node-level features
            gf = gf.mean(dim=0, keepdim=True)
    else:
        raise RuntimeError(f"Unexpected graph_feat dim={gf.dim()} shape={tuple(gf.shape)}")

    if gf.size(1) == len(best_graph_feat_mask):
        gf = gf[:, best_graph_feat_mask]

    g.graph_feat = gf
    return g


@torch.no_grad()
def split_and_plot_graph(graph, nodes_to_keep, do_plot=False):
    """Return (complement-masked graph, kept-subgraph) by zeroing node features."""
    g = graph.clone()
    if not hasattr(g, "batch") or g.batch is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long)

    if isinstance(nodes_to_keep, set):
        nodes_to_keep = sorted(nodes_to_keep)
    nodes_to_keep = torch.as_tensor(nodes_to_keep, dtype=torch.long)

    mask_keep = torch.zeros(g.num_nodes, dtype=torch.bool)
    if nodes_to_keep.numel() > 0:
        mask_keep[nodes_to_keep] = True

    subgraph_data = g.clone()
    subgraph_data.x = g.x.clone()
    subgraph_data.x[~mask_keep] = 0
    subgraph_data.batch = g.batch.clone()
    subgraph_data.graph_feat = g.graph_feat.clone()

    modified_data = g.clone()
    modified_data.x = g.x.clone()
    modified_data.x[mask_keep] = 0
    modified_data.batch = g.batch.clone()
    modified_data.graph_feat = g.graph_feat.clone()

    if do_plot:
        nx.draw_networkx(to_networkx(g))
        plt.title("Original graph")
        plt.show()

    return modified_data, subgraph_data


# =========================
# METRICS
# =========================
def binary_kl_divergence(p, q):
    eps = 1e-10
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()


def calculate_gef_binary(model, graph, subgraph_data):
    y2 = torch.sigmoid(model(graph.x, graph.edge_index, graph.graph_feat, graph.batch))
    y2_sub = torch.sigmoid(model(subgraph_data.x, subgraph_data.edge_index,
                                 subgraph_data.graph_feat, subgraph_data.batch))
    return float((1 - torch.exp(-binary_kl_divergence(y2, y2_sub))).item())


def compute_charact(w_pos, w_neg, fid_pos, fid_neg, eps=1e-12):
    numerator = (w_pos + w_neg) * fid_pos * (1 - fid_neg)
    denominator = w_pos * (1 - fid_neg) + w_neg * fid_pos
    if abs(denominator) < eps:
        return float("nan")
    return float(numerator / denominator)


@torch.no_grad()
def true_class_prob(p, y):
    y = int(y.item()) if torch.is_tensor(y) else int(y)
    return p if y == 1 else 1.0 - p


@torch.no_grad()
def fidelity(graph, subgraph_data, modified_graph_data, model,
             mode="phenomenon", fidelity_type="prob"):
    if mode != "phenomenon" or fidelity_type != "prob":
        raise NotImplementedError("Only the (phenomenon, prob) case is implemented.")

    y = graph.y
    p_full = torch.sigmoid(model(graph.x, graph.edge_index, graph.graph_feat, graph.batch)).item()
    p_sub = torch.sigmoid(model(subgraph_data.x, subgraph_data.edge_index,
                                subgraph_data.graph_feat, subgraph_data.batch)).item()
    p_comp = torch.sigmoid(model(modified_graph_data.x, modified_graph_data.edge_index,
                                 modified_graph_data.graph_feat, modified_graph_data.batch)).item()

    p_full_true = true_class_prob(p_full, y)
    pos_fidelity = p_full_true - true_class_prob(p_comp, y)
    neg_fidelity = p_full_true - true_class_prob(p_sub, y)
    gef = calculate_gef_binary(model, graph, subgraph_data)
    charact = compute_charact(0.5, 0.5, pos_fidelity, neg_fidelity)
    return float(pos_fidelity), float(neg_fidelity), float(gef), float(charact)


def sparsity(graph, node_set):
    return 1 - (len(node_set) / graph.num_nodes)


# =========================
# SubgraphX (MC L-Shapley + MCTS)
# =========================
@torch.no_grad()
def _aggregate_scores(loader, model, class_idx, task=Task.GRAPH_CLASSIFICATION, nodes_to_keep=None):
    result = torch.tensor([]).float()
    for data in iter(loader):
        logits = model(data.x.to(DEVICE), data.edge_index.to(DEVICE),
                       data.graph_feat.to(DEVICE), data.batch.to(DEVICE)).detach().cpu().view(-1)
        p1 = torch.sigmoid(logits)
        score_of_class = p1 if class_idx == 1 else 1.0 - p1
        result = torch.cat((result, score_of_class), dim=0)
    return result


@torch.no_grad()
def _compute_marginal_contribution(include_list, exclude_list, model, class_idx,
                                   task=Task.GRAPH_CLASSIFICATION, nodes_to_keep=None):
    include_scores = _aggregate_scores(
        DataLoader(include_list, batch_size=4, shuffle=False), model, class_idx, task, nodes_to_keep)
    exclude_scores = _aggregate_scores(
        DataLoader(exclude_list, batch_size=4, shuffle=False), model, class_idx, task, nodes_to_keep)
    return float(torch.mean(include_scores - exclude_scores).item())


@torch.no_grad()
def mc_l_shapley(model, graph, subgraph_set: Set[int], t: int, num_layers: int,
                 task=Task.GRAPH_CLASSIFICATION, nodes_to_keep=None) -> float:
    subgraph_list = list(subgraph_set)
    node_tensor, _, _, _ = k_hop_subgraph(
        subgraph_list, num_layers, graph.edge_index, relabel_nodes=False,
        num_nodes=graph.num_nodes, flow="source_to_target")
    p_prime = list(set(node_tensor.tolist()) - set(subgraph_list))
    placeholder = graph.num_nodes
    p = p_prime + [placeholder]

    true_class = int(graph.y.view(-1)[0].item())

    include_data_list, exclude_data_list = [], []
    for _ in range(t):
        perm = np.random.permutation(p)
        split_idx = np.asarray(perm == placeholder).nonzero()[0][0]
        selected = perm[:split_idx]

        include_mask = np.zeros(graph.num_nodes)
        include_mask[selected] = 1
        include_mask[subgraph_list] = 1
        inc = Data(x=(graph.x * torch.tensor(include_mask).unsqueeze(1)).float(),
                   edge_index=graph.edge_index)
        inc.graph_feat = graph.graph_feat.clone()
        inc.num_nodes = graph.num_nodes
        inc.batch = torch.zeros(inc.num_nodes, dtype=torch.long)
        include_data_list.append(inc)

        exclude_mask = np.zeros(graph.num_nodes)
        exclude_mask[selected] = 1
        exc = Data(x=(graph.x * torch.tensor(exclude_mask).unsqueeze(1)).float(),
                   edge_index=graph.edge_index)
        exc.graph_feat = graph.graph_feat.clone()
        exc.num_nodes = graph.num_nodes
        exc.batch = torch.zeros(exc.num_nodes, dtype=torch.long)
        exclude_data_list.append(exc)

    return float(_compute_marginal_contribution(
        include_data_list, exclude_data_list, model, true_class, task, nodes_to_keep))


class MCTSNode:
    def __init__(self, graph, n_min, node_set, score=None):
        self.graph = graph
        self.n_min = n_min
        self.node_set = node_set
        self.score = score
        self._hash = self.compute_hash()

    def compute_hash(self):
        l = sorted(self.node_set)
        result = 98767 - len(l) * 555
        for i, el in enumerate(l):
            result = result + (hash(el) % 9999999) * 1001 + i
        return result

    def is_terminal(self):
        return len(self.node_set) <= self.n_min

    def __hash__(self):
        return self._hash

    def __eq__(self, node2):
        return hash(self) == hash(node2) and self.node_set == node2.node_set


class MCTS:
    def __init__(self, graph, exp_weight, n_min, score_func, model, t, num_layers,
                 high2low=False, max_children=-1, task=Task.GRAPH_CLASSIFICATION,
                 nodes_to_keep=None, skip_to_leaves=True, experiment=None):
        self.W = defaultdict(float)
        self.C = defaultdict(int)
        self.children: Dict[MCTSNode, List[MCTSNode]] = {}
        self.leaves = []
        self.R: Dict[MCTSNode, float] = {}

        self.exp_weight = exp_weight
        self.n_min = n_min
        self.score_func = score_func
        self.graph = graph
        self.model = model
        self.t = t
        self.num_layers = num_layers
        self.high2low = high2low
        self.max_children = max_children
        self.nodes_to_keep = nodes_to_keep if nodes_to_keep is not None else []
        self.task = task
        self.skip_to_leaves = skip_to_leaves
        self.experiment = experiment

        if experiment == Experiment.DEFAULT:
            self.C = defaultdict(lambda: 1)

        self.root = MCTSNode(graph, n_min, set(range(graph.num_nodes)))
        self.root.score = self._r(self.root)
        self.paths = []

    def _q(self, node):
        return 0.0 if self.C[node] == 0 else float(self.W[node] / self.C[node])

    def _r(self, node):
        if node in self.R:
            node.score = self.R[node]
            return float(self.R[node])
        score = self.score_func(self.model, self.graph, node.node_set, self.t, self.num_layers)
        self.R[node] = float(score)
        node.score = float(score)
        return float(score)

    def _u(self, node, parent):
        children = self.children[parent]
        parent_count = sum(self.C[c] for c in children)
        if parent_count == 0 and self.skip_to_leaves:
            return 0.0
        return float(self.exp_weight * self._r(node) * math.sqrt(parent_count) / (1 + self.C[node]))

    def _ucb(self, node, parent):
        return self._q(node) + self._u(node, parent)

    def _best_child_by_ucb(self, node):
        children = self.children[node] if node in self.children else self._expand_node(node)
        return max(children, key=lambda child: self._ucb(child, node))

    def _select_path_by_ucb(self):
        node = self.root
        path = [node]
        while not node.is_terminal():
            node = self._best_child_by_ucb(node)
            path.append(node)
        return path

    def _expand_node(self, node):
        if node in self.children:
            raise Exception(f"Node already expanded: {node}")
        if node.is_terminal():
            raise Exception(f"Terminal node cannot be expanded: {node}")

        children = []
        nx_graph = to_networkx(self.graph, to_undirected=True)
        nodes_to_prune = list(node.node_set.copy())
        nodes_to_prune.sort(key=lambda x: nx_graph.degree(x), reverse=self.high2low)
        if self.max_children >= 0:
            nodes_to_prune = nodes_to_prune[:self.max_children]

        for nd in nodes_to_prune:
            subg = nx_graph.subgraph(node.node_set - {nd})
            child_set = max(nx.connected_components(subg), key=lambda x: len(x))
            children.append(MCTSNode(self.graph, self.n_min, set(child_set)))

        self.children[node] = children
        return children

    def _backpropagate(self, path):
        score = self._r(path[-1])
        for node in path:
            self.C[node] += 1
            self.W[node] += score

    def search_one_iteration(self):
        path = self._select_path_by_ucb()
        leaf = path[-1]
        if leaf not in self.leaves:
            self.leaves.append(leaf)
        self._backpropagate(path)
        self.paths.append(path)

    def best_leaf_node(self):
        return max(self.leaves, key=self._r)


class SubgraphX:
    def __init__(self, model, num_layers, exp_weight, m, t, high2low=False,
                 max_children=-1, task=Task.GRAPH_CLASSIFICATION,
                 value_func: Callable = mc_l_shapley, experiment=None):
        self.model = model
        self.num_layers = num_layers
        self.exp_weight = exp_weight
        self.m = m
        self.t = t
        self.value_func = value_func
        self.high2low = high2low
        self.max_children = max_children
        self.task = task
        self.experiment = experiment

    def __call__(self, graph, n_min, nodes_to_keep=None, exhaustive=False):
        mcts = MCTS(graph, self.exp_weight, n_min, self.value_func, self.model, self.t,
                    self.num_layers, self.high2low, self.max_children, self.task,
                    nodes_to_keep if nodes_to_keep is not None else [],
                    skip_to_leaves=(not exhaustive), experiment=self.experiment)
        for _ in tqdm(range(self.m), leave=False):
            mcts.search_one_iteration()
        return mcts.best_leaf_node().node_set, mcts


# =========================
# PLOTTING
# =========================
def plot_subject_graphs(graph: Data, explaining_subgraph: set, subject_idx,
                        out_dir=None, show=True):
    """Plot the original sparse graph and the explanation-induced subgraph."""
    g_orig = graph.clone()
    if not hasattr(g_orig, "num_nodes") or g_orig.num_nodes is None:
        g_orig.num_nodes = int(g_orig.x.size(0))

    nodes = torch.as_tensor(sorted(list(explaining_subgraph)), dtype=torch.long)
    if nodes.numel() == 0:
        ei_sub = torch.empty((2, 0), dtype=torch.long)
        n_sub = 0
    else:
        ei_sub, _ = subgraph(nodes, g_orig.edge_index, relabel_nodes=True, num_nodes=g_orig.num_nodes)
        n_sub = int(nodes.numel())

    g_sub = Data(x=torch.ones((n_sub, 1)), edge_index=ei_sub, num_nodes=n_sub)
    G_orig = to_networkx(g_orig, to_undirected=True)
    G_sub = to_networkx(g_sub, to_undirected=True)
    pos = nx.spring_layout(G_orig, seed=42)

    labels_sub = {i: int(nodes[i].item()) for i in range(n_sub)}
    pos_sub = {i: pos[int(nodes[i].item())] for i in range(n_sub)} if n_sub > 0 else {}

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    nx.draw(G_orig, pos=pos, node_size=80, with_labels=False)
    plt.title(f"Subject {subject_idx} - Original sparse graph")
    plt.subplot(1, 2, 2)
    if n_sub > 0:
        nx.draw(G_sub, pos=pos_sub, node_size=250, with_labels=True, labels=labels_sub)
    plt.title(f"Subject {subject_idx} - Explanation subgraph (n={n_sub})")
    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"subject_{subject_idx}_graphs.png"),
                    dpi=200, bbox_inches="tight")
    plt.show() if show else plt.close()


# =========================
# MAIN: seeds x folds x subjects
# =========================
def run_all_seeds():
    rows_subject, rows_seed_class = [], []
    subgraphs_pth = {
        "meta": {"tp_tn_only": TP_TN_ONLY, "r_repeats": R_REPEATS, "n_min": N_MIN,
                 "thr": THR, "seeds": SEEDS, "results_dir": RESULTS_DIR},
        "by_subject": {},
        "by_seed": defaultdict(list),
    }

    for seed in SEEDS:
        loaded = torch.load(find_seed_file(RESULTS_DIR, seed), map_location="cpu")
        df_outer = loaded["df_outer"]
        models_state_dicts = loaded["models"]
        Test_Data_outer = loaded["Test_Data_outer"]

        seed_class_metrics = {"schizo": defaultdict(list), "control": defaultdict(list)}
        seed_class_consensus = {"schizo": [], "control": []}
        seen_subject_ids = set()

        for fold_idx in range(len(df_outer)):
            params = df_outer["params"].iloc[fold_idx]
            best_node_mask = torch.tensor(df_outer["best_node_mask"].iloc[fold_idx], dtype=torch.bool)
            best_graph_feat_mask = torch.tensor(
                df_outer["best_graph_feat_mask"].iloc[fold_idx], dtype=torch.bool)

            base_model = GAT(
                in_channels=int(best_node_mask.sum().item()),
                hidden_channels=params["hidden_dim"], dropout=params["dropout"],
                graph_feat_dim=int(best_graph_feat_mask.sum().item()),
                num_layers=params["num_layers"], heads=params.get("heads", 1),
                use_layernorm=False, concat=True, pool="attn").to(DEVICE)
            base_model.load_state_dict(models_state_dicts[fold_idx])
            base_model.eval()
            model = CountedModel(base_model)
            model.eval()

            subgraphx = SubgraphX(model=model, num_layers=2, exp_weight=20, m=30, t=40,
                                  task=Task.GRAPH_CLASSIFICATION, max_children=-1,
                                  experiment=Experiment.GREEDY, value_func=mc_l_shapley)

            graphs_fold = Test_Data_outer[fold_idx]
            y_true = np.asarray(df_outer["y_true"].iloc[fold_idx]).astype(int)
            y_hat = (np.asarray(df_outer["y_pred"].iloc[fold_idx]).astype(float) >= THR).astype(int)
            idx_keep = np.where(y_true == y_hat)[0] if TP_TN_ONLY else np.arange(len(graphs_fold))
            graphs_keep = [graphs_fold[i] for i in idx_keep]

            for local_i, g_raw in enumerate(graphs_keep):
                sid = str(getattr(g_raw, "subject_id", f"seed{seed}_fold{fold_idx}_i{local_i}"))
                if sid in seen_subject_ids:
                    continue
                seen_subject_ids.add(sid)

                g = prepare_graph(g_raw, best_node_mask, best_graph_feat_mask)
                if not hasattr(g, "y") or g.y is None:
                    continue
                y = int(g.y.item())
                cls = "schizo" if y == 1 else "control"

                runs_sets = []
                for r in range(R_REPEATS):
                    torch.manual_seed(10_000 + seed * 100 + fold_idx * 10 + r)
                    np.random.seed(10_000 + seed * 100 + fold_idx * 10 + r)
                    random.seed(10_000 + seed * 100 + fold_idx * 10 + r)
                    node_set_r, _ = subgraphx(g, n_min=N_MIN, nodes_to_keep=None, exhaustive=True)
                    runs_sets.append(set(node_set_r))

                best_set, best_cnt, _ = most_frequent_nodeset(runs_sets)
                if DO_PLOT_FINAL:
                    plot_subject_graphs(g, best_set, subject_idx=sid, out_dir=None, show=True)
                consensus_rate = best_cnt / R_REPEATS

                modified_graph_data, subgraph_data = split_and_plot_graph(g, best_set, do_plot=False)
                modified_graph_data = prepare_graph(modified_graph_data, best_node_mask, best_graph_feat_mask)
                subgraph_data = prepare_graph(subgraph_data, best_node_mask, best_graph_feat_mask)

                pos_fid, neg_fid, gef, charact = fidelity(
                    g, subgraph_data, modified_graph_data, model,
                    mode="phenomenon", fidelity_type="prob")
                spars = sparsity(g, best_set)

                for k, v in [("pos_fid", pos_fid), ("neg_fid", neg_fid), ("sparsity", spars),
                             ("gef", gef), ("charact", charact)]:
                    seed_class_metrics[cls][k].append(v)
                seed_class_consensus[cls].append(consensus_rate)

                rows_subject.append({
                    "seed": seed, "fold": fold_idx, "subject_id": sid,
                    "class_label": y, "class_name": cls,
                    "consensus_rate": consensus_rate, "subgraph_size": len(best_set),
                    "best_nodes": ",".join(map(str, sorted(best_set))),
                    "runs_nodesets": "|".join(",".join(map(str, sorted(s))) for s in runs_sets),
                    "pos_fid": pos_fid, "neg_fid": neg_fid, "sparsity": spars,
                    "gef": gef, "charact": charact,
                })

                subj_key = (int(seed), int(fold_idx), str(sid))
                subgraphs_pth["by_subject"][subj_key] = {
                    "seed": int(seed), "fold": int(fold_idx), "subject_id": str(sid),
                    "class_name": str(cls), "class_label": int(y),
                    "best_nodes": sorted(list(best_set)),
                    "runs_nodes": [sorted(list(s)) for s in runs_sets],
                    "consensus_rate": float(consensus_rate),
                    "metrics": {"pos_fid": float(pos_fid), "neg_fid": float(neg_fid),
                                "sparsity": float(spars), "gef": float(gef),
                                "charact": float(charact)},
                }
                subgraphs_pth["by_seed"][int(seed)].append(subj_key)

        for cls in ["schizo", "control"]:
            row = {"seed": int(seed), "class_name": cls,
                   "n_subjects": len(seed_class_metrics[cls]["pos_fid"]),
                   "consensus_rate_mean": float(np.mean(seed_class_consensus[cls])) if seed_class_consensus[cls] else np.nan,
                   "consensus_rate_median": float(np.median(seed_class_consensus[cls])) if seed_class_consensus[cls] else np.nan}
            for k in ["pos_fid", "neg_fid", "sparsity", "gef", "charact"]:
                s = summarize_metric(seed_class_metrics[cls][k])
                row[f"{k}_mean"], row[f"{k}_median"], row[f"{k}_std"] = s["mean"], s["median"], s["std"]
            rows_seed_class.append(row)

        print(f"[seed {seed}] done. unique subjects={len(seen_subject_ids)}")

    df_subject = pd.DataFrame(rows_subject)
    df_seed_class = pd.DataFrame(rows_seed_class)

    out_xlsx = os.path.join(OUT_DIR, "subgraphx_results.xlsx")
    df_subject.to_csv(os.path.join(OUT_DIR, "subgraphx_subject_level.csv"), index=False)
    df_seed_class.to_csv(os.path.join(OUT_DIR, "subgraphx_seed_class_level.csv"), index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_seed_class.to_excel(w, sheet_name="seed_class_level", index=False)
        df_subject.to_excel(w, sheet_name="subject_level", index=False)

    torch.save(subgraphs_pth, SUBGRAPH_PTH)
    print("Saved:", out_xlsx)
    print("Saved subgraphs to:", SUBGRAPH_PTH)
    return df_seed_class, df_subject


if __name__ == "__main__":
    t0 = time.time()
    run_all_seeds()
    print(f"\nDone. Total time = {time.time() - t0:.1f}s")
