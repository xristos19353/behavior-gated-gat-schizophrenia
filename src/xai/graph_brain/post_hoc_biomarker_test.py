"""Consensus node/edge biomarker validation: sufficiency (keep) and necessity (remove).

For each condition (sz / hc / union) the consensus nodes (and, separately, the
consensus edges) are validated against random same-size sets, per fold and seed:

  A) SUFFICIENCY (keep-only): keep only consensus nodes (zero the rest) and measure
     performance vs. random keep-only sets.  D_keep = consensus - random.
  B) NECESSITY (remove-only): remove consensus nodes (zero only those) and compare
     the drop vs. random removals.  D_remove = drop_consensus - drop_random.

Global tests: one-sided exact sign-flip across the 8 seeds, with Cohen's dz,
Cliff's delta and BH-FDR. Self-contained; paths come from ``config``.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch_geometric.nn import GATConv, LayerNorm, global_mean_pool
from torch_geometric.utils import softmax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = str(config.GATE_RESULTS_DIR)
SEEDS = config.SEEDS
OUT_DIR = os.path.join(str(config.XAI_RESULTS_DIR), "subgraphx_outputs", "subgraphx_consensus_validation")
os.makedirs(OUT_DIR, exist_ok=True)

FULL_ATLAS_SIZE = config.FULL_ATLAS_SIZE
THR = 0.5
DEVICE = torch.device("cpu")

N_PERM = 100      # random node sets per fold
RNG_SEED = 42
EDGE_METRIC = "true_class_prob_mean"

# "zero": zero-out node features; "subgraph": physically remove nodes + edges.
MASK_MODE = "zero"

# Consensus nodes (0-based global ROI ids), from the SubgraphX inference step.
CONSENSUS_NODES = {
    "sz": [22, 27, 69, 70, 72, 75, 83, 86, 87, 88, 101],
    "hc": [9, 22, 27, 69, 72, 75, 83, 86, 87, 88, 96, 101],
}
CONSENSUS_NODES["union"] = sorted(set(CONSENSUS_NODES["sz"]) | set(CONSENSUS_NODES["hc"]))
CONDITIONS = list(CONSENSUS_NODES.keys())
METRICS = ["auc", "balanced_acc", "sensitivity", "specificity"]

# Consensus edges come from the structure-aware inference workbook.
CONSENSUS_EDGE_XLSX = os.path.join(
    str(config.XAI_RESULTS_DIR), "subgraphx_outputs", "subgraphx_xAI",
    "subgraphx_inference_results_structure.xlsx")
CONSENSUS_EDGE_SHEET = "Consensus_Graphs_Thresh"
EDGE_CONDITIONS = ["sz", "hc"]
EDGE_N_PERM = 100


# ============================================================
# MODEL
# ============================================================
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

    def forward(self, x, edge_index, graph_feat, batch=None):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_layernorm:
                x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pool == "attn":
            gate = self.gate_nn(torch.cat([x, graph_feat[batch]], dim=1)).view(-1)
            alpha = softmax(gate, batch)
            x = x * alpha.unsqueeze(-1)
            batch_size = int(batch.max().item()) + 1
            x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)
            x_pooled.index_add_(0, batch, x)
            x = x_pooled
        else:
            x = global_mean_pool(x, batch)
        return self.fc(x)


# ============================================================
# HELPERS
# ============================================================
def find_seed_file(results_dir, seed):
    files = sorted(glob.glob(os.path.join(results_dir, f"all_results_*_seed{seed}.pth")))
    if not files:
        raise FileNotFoundError(f"No .pth for seed={seed} in {results_dir}")
    return files[0]


def roi_ids_per_node_from_graph(g, full_size):
    if not hasattr(g, "roi_to_node"):
        raise AttributeError("Graph missing roi_to_node")
    r2n = getattr(g, "roi_to_node")
    num_nodes = int(g.x.shape[0]) if hasattr(g, "x") else int(max(r2n.values()) + 1)
    roi_raw = np.full(num_nodes, -999, dtype=int)
    for roi_key, node_idx in r2n.items():
        m = re.match(r"ROI_(\d+)$", str(roi_key))
        if not m:
            raise ValueError(f"Unexpected roi key: {roi_key}")
        roi_raw[int(node_idx)] = int(m.group(1))
    if np.any(roi_raw < 0):
        raise ValueError("roi_to_node missing nodes")
    return roi_raw - 1 if (roi_raw.min() >= 1 and roi_raw.max() <= full_size) else roi_raw


def prepare_graph(g, best_node_mask, best_graph_feat_mask):
    g = g.clone()
    if hasattr(g, "x") and g.x is not None:
        if g.x.dim() == 2 and g.x.size(1) == len(best_node_mask):
            g.x = g.x[:, best_node_mask]
    if not hasattr(g, "batch") or g.batch is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
    gf = g.graph_feat
    if gf.dim() == 1:
        gf = gf.view(1, -1)
    elif gf.dim() == 2 and gf.size(0) == g.num_nodes:
        gf = gf.mean(dim=0, keepdim=True)
    if gf.size(1) == len(best_graph_feat_mask):
        gf = gf[:, best_graph_feat_mask]
    g.graph_feat = gf
    return g


def mask_to_local(global_roi_ids, roi0):
    """Convert 0-based global ROI ids to local node indices for this subject."""
    global_to_local = {int(roi0[loc]): loc for loc in range(len(roi0))}
    return [global_to_local[r] for r in global_roi_ids if r in global_to_local]


@torch.no_grad()
def remove_nodes_from_graph(g, local_nodes_to_remove):
    """Physically remove nodes + their edges, reindexing the rest."""
    n = g.num_nodes
    keep_mask = torch.ones(n, dtype=torch.bool)
    for idx in local_nodes_to_remove:
        if 0 <= idx < n:
            keep_mask[idx] = False
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    if len(keep_idx) == 0:
        return None
    old_to_new = torch.full((n,), -1, dtype=torch.long)
    old_to_new[keep_idx] = torch.arange(len(keep_idx), dtype=torch.long)
    ei = g.edge_index
    edge_mask = keep_mask[ei[0]] & keep_mask[ei[1]]
    g2 = g.clone()
    g2.x = g.x[keep_idx]
    g2.edge_index = old_to_new[ei[:, edge_mask]]
    g2.batch = torch.zeros(len(keep_idx), dtype=torch.long)
    return g2


@torch.no_grad()
def predict_masked(model, g, local_nodes, mode="keep", mask_mode=MASK_MODE):
    """mode keep/remove x mask_mode zero/subgraph -> P(SZ)."""
    if mask_mode == "zero":
        g2 = g.clone()
        mask = torch.zeros(g2.num_nodes, dtype=torch.bool)
        for idx in local_nodes:
            if 0 <= idx < g2.num_nodes:
                mask[idx] = True
        if mode == "keep":
            g2.x[~mask] = 0.0
        else:
            g2.x[mask] = 0.0
        return float(torch.sigmoid(model(g2.x, g2.edge_index, g2.graph_feat, g2.batch)).item())

    if mask_mode == "subgraph":
        if mode == "keep":
            to_remove = [i for i in range(g.num_nodes) if i not in set(local_nodes)]
            g2 = remove_nodes_from_graph(g, to_remove)
        else:
            g2 = remove_nodes_from_graph(g, local_nodes)
        if g2 is None or g2.num_nodes == 0:
            return 0.5
        return float(torch.sigmoid(model(g2.x, g2.edge_index, g2.graph_feat, g2.batch)).item())

    raise ValueError(f"mask_mode must be 'zero' or 'subgraph', got {mask_mode!r}")


def compute_metrics(y_true, y_prob, thr=THR):
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    y_hat = (y_prob >= thr).astype(int)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = np.nan
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        out["sensitivity"] = float(sens)
        out["specificity"] = float(spec)
        out["balanced_acc"] = float((sens + spec) / 2) if np.isfinite(sens) and np.isfinite(spec) else np.nan
    except Exception:
        out["sensitivity"] = out["specificity"] = out["balanced_acc"] = np.nan
    return out


def signflip_test(D):
    """Exact one-sided sign-flip test. H0: mean(D)=0 ; H1: mean(D)>0."""
    D = np.asarray(D, float)
    D = D[np.isfinite(D)]
    n = len(D)
    if n < 2:
        return (float(np.mean(D)) if n > 0 else np.nan), np.nan
    obs = float(np.mean(D))
    means = np.array([float(np.mean(np.asarray(s, float) * D)) for s in product([-1.0, 1.0], repeat=n)])
    return obs, float(np.mean(means >= obs))


def cliffs_delta(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float((np.sum(x > 0) - np.sum(x < 0)) / len(x))


def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    q = np.full_like(pvals, np.nan, dtype=float)
    idx = np.where(np.isfinite(pvals))[0]
    if idx.size == 0:
        return q
    pv = pvals[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    q_ranked = np.empty(m, float)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        prev = min(prev, (m / (k + 1)) * ranked[k])
        q_ranked[k] = prev
    tmp = np.empty_like(pv)
    tmp[order] = q_ranked
    q[idx] = tmp
    return q


def paired_effect_size(D):
    """Cohen's dz and Hedges' g for paired differences."""
    D = np.asarray(D, float)
    D = D[np.isfinite(D)]
    n = len(D)
    if n < 2:
        return np.nan, np.nan
    sd_D = np.std(D, ddof=1)
    dz = np.mean(D) / sd_D if sd_D > 0 else np.nan
    g = dz * (1 - (3 / (4 * n - 9))) if np.isfinite(dz) else np.nan
    return dz, g


def load_consensus_edges_from_excel(xlsx_path, sheet_name="Consensus_Graphs_Thresh"):
    """Load thresholded consensus edges (columns group, node_u, node_v; 0-based)."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    missing = {"group", "node_u", "node_v"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {sheet_name}: {missing}")

    edge_dict = {"sz": [], "hc": [], "union": []}
    group_map = {"schizo": "sz", "control": "hc", "sz": "sz", "hc": "hc", "union": "union"}
    for _, r in df.iterrows():
        grp = group_map.get(str(r["group"]).strip().lower())
        if grp is None:
            continue
        u, v = int(r["node_u"]), int(r["node_v"])
        edge_dict[grp].append((u, v) if u < v else (v, u))
    for k in edge_dict:
        edge_dict[k] = sorted(set(edge_dict[k]))
    if len(edge_dict["union"]) == 0:
        edge_dict["union"] = sorted(set(edge_dict["sz"]) | set(edge_dict["hc"]))
    return edge_dict


def subject_original_roi_edges_with_columns(g, full_size):
    """Return (set of undirected ROI edges, per-column ROI-edge list)."""
    edge_index = g.edge_index
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index)
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    roi0 = roi_ids_per_node_from_graph(g, full_size)
    u = edge_index[0].cpu().numpy().astype(int)
    v = edge_index[1].cpu().numpy().astype(int)
    roi_edges_set, edge_roi_list = set(), []
    for a, b in zip(u, v):
        ra, rb = int(roi0[a]), int(roi0[b])
        if ra == rb or ra < 0 or rb < 0 or ra >= full_size or rb >= full_size:
            edge_roi_list.append(None)
            continue
        e = (ra, rb) if ra < rb else (rb, ra)
        roi_edges_set.add(e)
        edge_roi_list.append(e)
    return roi_edges_set, edge_roi_list


@torch.no_grad()
def predict_edge_masked(model, g, selected_edges, full_size=114, mode="keep"):
    g2 = g.clone()
    _, edge_roi_list = subject_original_roi_edges_with_columns(g2, full_size)
    selected_edges = set(tuple(sorted(e)) for e in selected_edges)
    mask = []
    for e in edge_roi_list:
        if e is None:
            mask.append(False)
        else:
            mask.append((e in selected_edges) if mode == "keep" else (e not in selected_edges))
    mask = torch.tensor(mask, dtype=torch.bool)
    if mask.sum() == 0:
        return 0.5
    g2.edge_index = g2.edge_index[:, mask]
    return float(torch.sigmoid(model(g2.x, g2.edge_index, g2.graph_feat, g2.batch)).item())


def sample_random_existing_edges(subject_roi_edges, target_edges, rng):
    """Random subset of the subject's edges, same size as target_edges."""
    available = sorted(set(subject_roi_edges))
    if len(available) == 0:
        return []
    k = min(len(target_edges), len(available))
    idx = rng.choice(len(available), size=k, replace=False)
    return [available[i] for i in np.atleast_1d(idx)]


def prob_true_class(y_true, p_sz):
    y_true = np.asarray(y_true, int)
    p_sz = np.asarray(p_sz, float)
    return np.where(y_true == 1, p_sz, 1.0 - p_sz)


def true_class_prob_mean(y_true, p_sz):
    return float(np.nanmean(prob_true_class(y_true, p_sz)))


# ============================================================
# MAIN
# ============================================================
def run_validation():
    rng = np.random.default_rng(RNG_SEED)

    rows_edge_fold, rows_edge_seed = [], []
    across_seed_edge_cons = {"keep": {c: [] for c in EDGE_CONDITIONS}, "remove": {c: [] for c in EDGE_CONDITIONS}}
    across_seed_edge_null = {"keep": {c: [] for c in EDGE_CONDITIONS}, "remove": {c: [] for c in EDGE_CONDITIONS}}
    across_seed_full_trueprob = []

    EDGE_CONSENSUS = load_consensus_edges_from_excel(CONSENSUS_EDGE_XLSX, CONSENSUS_EDGE_SHEET)
    print("Loaded consensus edges:", {k: len(EDGE_CONSENSUS[k]) for k in ("sz", "hc", "union")})

    n_nodes_per = {c: len(CONSENSUS_NODES[c]) for c in CONDITIONS}

    rows_fold, rows_seed = [], []
    across_seed_cons = {t: {c: {m: [] for m in METRICS} for c in CONDITIONS} for t in ("keep", "remove")}
    across_seed_null = {t: {c: {m: [] for m in METRICS} for c in CONDITIONS} for t in ("keep", "remove")}
    across_seed_full = {m: [] for m in METRICS}

    for seed in SEEDS:
        print("\n" + "=" * 70 + f"\nSeed {seed}\n" + "=" * 70)

        fold_edge_cons_metrics = {"keep": {c: [] for c in EDGE_CONDITIONS}, "remove": {c: [] for c in EDGE_CONDITIONS}}
        fold_edge_null_pool = {"keep": {c: [] for c in EDGE_CONDITIONS}, "remove": {c: [] for c in EDGE_CONDITIONS}}

        loaded = torch.load(find_seed_file(RESULTS_DIR, seed), map_location="cpu")
        df_outer = loaded["df_outer"]
        models_state_dicts = loaded["models"]
        Test_Data_outer = loaded["Test_Data_outer"]
        n_folds = len(df_outer)

        fold_full_metrics = {m: [] for m in METRICS}
        fold_full_trueprob = []
        fold_cons_metrics = {t: {c: {m: [] for m in METRICS} for c in CONDITIONS} for t in ("keep", "remove")}
        fold_null_pool = {t: {c: {m: [] for m in METRICS} for c in CONDITIONS} for t in ("keep", "remove")}

        for fold_idx in range(n_folds):
            params = df_outer["params"].iloc[fold_idx]
            best_node_mask = torch.tensor(df_outer["best_node_mask"].iloc[fold_idx], dtype=torch.bool)
            best_graph_feat_mask = torch.tensor(df_outer["best_graph_feat_mask"].iloc[fold_idx], dtype=torch.bool)

            model = GAT(
                in_channels=int(best_node_mask.sum().item()),
                hidden_channels=params["hidden_dim"], dropout=params["dropout"],
                graph_feat_dim=int(best_graph_feat_mask.sum().item()),
                num_layers=params["num_layers"], heads=params.get("heads", 1),
                use_layernorm=False, concat=True, pool="attn").to(DEVICE)
            model.load_state_dict(models_state_dicts[fold_idx])
            model.eval()

            graphs_fold = Test_Data_outer[fold_idx]
            y_true_full = np.asarray(df_outer["y_true"].iloc[fold_idx]).astype(int)
            y_prob_full = np.asarray(df_outer["y_pred"].iloc[fold_idx]).astype(float)
            full_m = compute_metrics(y_true_full, y_prob_full)
            full_trueprob = true_class_prob_mean(y_true_full, y_prob_full)
            fold_full_trueprob.append(full_trueprob)
            for m in METRICS:
                fold_full_metrics[m].append(full_m[m])

            # ----- Edge validation (collect successful subjects atomically) -----
            y_true_edge_list = []
            y_prob_edge_keep = {c: [] for c in EDGE_CONDITIONS}
            y_prob_edge_rem = {c: [] for c in EDGE_CONDITIONS}
            successful_edge_subjects = []

            for g_raw in graphs_fold:
                try:
                    g = prepare_graph(g_raw, best_node_mask, best_graph_feat_mask)
                    y_true_val = int(g_raw.y.item())
                    roi_edges_set, _ = subject_original_roi_edges_with_columns(g_raw, FULL_ATLAS_SIZE)

                    keep_probs, rem_probs = {}, {}
                    for c in EDGE_CONDITIONS:
                        cons_edges = EDGE_CONSENSUS[c]
                        if len(cons_edges) == 0:
                            keep_probs[c] = 0.5
                            with torch.no_grad():
                                rem_probs[c] = float(torch.sigmoid(
                                    model(g.x, g.edge_index, g.graph_feat, g.batch)).item())
                        else:
                            keep_probs[c] = predict_edge_masked(model, g, cons_edges, FULL_ATLAS_SIZE, "keep")
                            rem_probs[c] = predict_edge_masked(model, g, cons_edges, FULL_ATLAS_SIZE, "remove")

                    y_true_edge_list.append(y_true_val)
                    successful_edge_subjects.append((g_raw, roi_edges_set))
                    for c in EDGE_CONDITIONS:
                        y_prob_edge_keep[c].append(keep_probs[c])
                        y_prob_edge_rem[c].append(rem_probs[c])
                except Exception:
                    continue

            if len(y_true_edge_list) >= 2:
                y_true_edge_arr = np.array(y_true_edge_list, int)
                cons_edge_keep_m = {c: true_class_prob_mean(y_true_edge_arr, y_prob_edge_keep[c]) for c in EDGE_CONDITIONS}
                cons_edge_rem_m = {c: true_class_prob_mean(y_true_edge_arr, y_prob_edge_rem[c]) for c in EDGE_CONDITIONS}
                for c in EDGE_CONDITIONS:
                    fold_edge_cons_metrics["keep"][c].append(cons_edge_keep_m[c])
                    fold_edge_cons_metrics["remove"][c].append(cons_edge_rem_m[c])

                for _ in range(EDGE_N_PERM):
                    for c in EDGE_CONDITIONS:
                        y_prob_keep_rand, y_prob_rem_rand = [], []
                        cons_edges = EDGE_CONSENSUS[c]
                        for g_raw, roi_edges_set in successful_edge_subjects:
                            try:
                                g = prepare_graph(g_raw, best_node_mask, best_graph_feat_mask)
                                rand_edges = sample_random_existing_edges(roi_edges_set, cons_edges, rng)
                                if len(rand_edges) == 0:
                                    y_prob_keep_rand.append(0.5)
                                    with torch.no_grad():
                                        y_prob_rem_rand.append(float(torch.sigmoid(
                                            model(g.x, g.edge_index, g.graph_feat, g.batch)).item()))
                                else:
                                    y_prob_keep_rand.append(predict_edge_masked(model, g, rand_edges, FULL_ATLAS_SIZE, "keep"))
                                    y_prob_rem_rand.append(predict_edge_masked(model, g, rand_edges, FULL_ATLAS_SIZE, "remove"))
                            except Exception:
                                y_prob_keep_rand.append(0.5)
                                y_prob_rem_rand.append(0.5)
                        fold_edge_null_pool["keep"][c].append(true_class_prob_mean(y_true_edge_arr, y_prob_keep_rand))
                        fold_edge_null_pool["remove"][c].append(true_class_prob_mean(y_true_edge_arr, y_prob_rem_rand))

                edge_fold_row = {"seed": seed, "fold": fold_idx, "n_subj": len(y_true_edge_list),
                                 "full_true_class_prob_mean": full_trueprob}
                for c in EDGE_CONDITIONS:
                    edge_fold_row[f"edge_keep_{c}_true_class_prob_mean"] = cons_edge_keep_m[c]
                    edge_fold_row[f"edge_rem_{c}_true_class_prob_mean"] = cons_edge_rem_m[c]
                    edge_fold_row[f"edge_drop_{c}_true_class_prob_mean"] = full_trueprob - cons_edge_rem_m[c]
                rows_edge_fold.append(edge_fold_row)

            # ----- Node validation (keep / remove) -----
            y_true_list = []
            y_prob_keep = {c: [] for c in CONDITIONS}
            y_prob_rem = {c: [] for c in CONDITIONS}
            for g_raw in graphs_fold:
                try:
                    roi0 = roi_ids_per_node_from_graph(g_raw, FULL_ATLAS_SIZE)
                    g = prepare_graph(g_raw, best_node_mask, best_graph_feat_mask)
                    keep_probs, rem_probs = {}, {}
                    for c in CONDITIONS:
                        local_ids = mask_to_local(CONSENSUS_NODES[c], roi0)
                        if len(local_ids) == 0:
                            keep_probs[c] = 0.5
                            with torch.no_grad():
                                rem_probs[c] = float(torch.sigmoid(
                                    model(g.x, g.edge_index, g.graph_feat, g.batch)).item())
                        else:
                            keep_probs[c] = predict_masked(model, g, local_ids, mode="keep")
                            rem_probs[c] = predict_masked(model, g, local_ids, mode="remove")
                    y_true_list.append(int(g_raw.y.item()))
                    for c in CONDITIONS:
                        y_prob_keep[c].append(keep_probs[c])
                        y_prob_rem[c].append(rem_probs[c])
                except Exception:
                    continue

            if len(y_true_list) < 2:
                continue
            y_true_arr = np.array(y_true_list, int)
            cons_keep_m = {c: compute_metrics(y_true_arr, y_prob_keep[c]) for c in CONDITIONS}
            cons_rem_m = {c: compute_metrics(y_true_arr, y_prob_rem[c]) for c in CONDITIONS}
            for c in CONDITIONS:
                for m in METRICS:
                    fold_cons_metrics["keep"][c][m].append(cons_keep_m[c][m])
                    fold_cons_metrics["remove"][c][m].append(cons_rem_m[c][m])

            for _ in range(N_PERM):
                for c in CONDITIONS:
                    rand_global = rng.choice(FULL_ATLAS_SIZE, size=n_nodes_per[c], replace=False).tolist()
                    y_prob_keep_rand, y_prob_rem_rand = [], []
                    for g_raw in graphs_fold:
                        try:
                            roi0 = roi_ids_per_node_from_graph(g_raw, FULL_ATLAS_SIZE)
                            g = prepare_graph(g_raw, best_node_mask, best_graph_feat_mask)
                            local_ids = mask_to_local(rand_global, roi0)
                            if len(local_ids) == 0:
                                y_prob_keep_rand.append(0.5)
                                with torch.no_grad():
                                    y_prob_rem_rand.append(float(torch.sigmoid(
                                        model(g.x, g.edge_index, g.graph_feat, g.batch)).item()))
                            else:
                                y_prob_keep_rand.append(predict_masked(model, g, local_ids, mode="keep"))
                                y_prob_rem_rand.append(predict_masked(model, g, local_ids, mode="remove"))
                        except Exception:
                            y_prob_keep_rand.append(0.5)
                            y_prob_rem_rand.append(0.5)
                    null_keep_m = compute_metrics(y_true_arr, y_prob_keep_rand)
                    null_rem_m = compute_metrics(y_true_arr, y_prob_rem_rand)
                    for m in METRICS:
                        fold_null_pool["keep"][c][m].append(null_keep_m[m])
                        fold_null_pool["remove"][c][m].append(null_rem_m[m])

            fold_row = {"seed": seed, "fold": fold_idx, "n_subj": len(y_true_list)}
            for m in METRICS:
                fold_row[f"full_{m}"] = full_m[m]
            for c in CONDITIONS:
                for m in METRICS:
                    fold_row[f"keep_{c}_{m}"] = cons_keep_m[c][m]
                    fold_row[f"rem_{c}_{m}"] = cons_rem_m[c][m]
                    fold_row[f"drop_{c}_{m}"] = full_m[m] - cons_rem_m[c][m]
            rows_fold.append(fold_row)

        # ----- Seed-level aggregation: nodes -----
        seed_mean_full = {m: float(np.nanmean(fold_full_metrics[m])) for m in METRICS}
        for m in METRICS:
            across_seed_full[m].append(seed_mean_full[m])

        seed_row = {"seed": seed, "n_folds": n_folds, "n_perm_per_fold": N_PERM}
        for m in METRICS:
            seed_row[f"full_{m}"] = seed_mean_full[m]
        for t in ("keep", "remove"):
            for c in CONDITIONS:
                for m in METRICS:
                    cons = float(np.nanmean(fold_cons_metrics[t][c][m]))
                    null = float(np.nanmean(fold_null_pool[t][c][m]))
                    across_seed_cons[t][c][m].append(cons)
                    across_seed_null[t][c][m].append(null)
                    if t == "keep":
                        seed_row[f"keep_{c}_D_{m}"] = cons - null
                    else:
                        seed_row[f"rem_{c}_D_{m}"] = (seed_mean_full[m] - cons) - (seed_mean_full[m] - null)
        rows_seed.append(seed_row)

        # ----- Seed-level aggregation: edges -----
        seed_mean_full_trueprob = float(np.nanmean(fold_full_trueprob))
        across_seed_full_trueprob.append(seed_mean_full_trueprob)
        for t in ("keep", "remove"):
            for c in EDGE_CONDITIONS:
                cons_mean = float(np.nanmean(fold_edge_cons_metrics[t][c]))
                null_mean = float(np.nanmean(fold_edge_null_pool[t][c]))
                across_seed_edge_cons[t][c].append(cons_mean)
                across_seed_edge_null[t][c].append(null_mean)
        rows_edge_seed.append({"seed": seed, "n_folds": n_folds,
                               "full_true_class_prob_mean": seed_mean_full_trueprob})

    # ============================================================
    # GLOBAL TESTS
    # ============================================================
    def global_tests(across_cons, across_null, conditions, full_ref, metrics, n_per_cond, target="node"):
        rows = []
        for test_type in ("keep", "remove"):
            for c in conditions:
                for m in metrics:
                    cons_vals = np.array(across_cons[test_type][c][m] if target == "node"
                                         else across_cons[test_type][c], float)
                    null_vals = np.array(across_null[test_type][c][m] if target == "node"
                                         else across_null[test_type][c], float)
                    if test_type == "keep":
                        D = cons_vals - null_vals
                        mean_cons, mean_null = float(np.nanmean(cons_vals)), float(np.nanmean(null_vals))
                    else:
                        full_vals = np.array(full_ref[m] if target == "node" else full_ref, float)
                        D = (full_vals - cons_vals) - (full_vals - null_vals)
                        mean_cons = float(np.nanmean(full_vals - cons_vals))
                        mean_null = float(np.nanmean(full_vals - null_vals))
                    mean_D, p = signflip_test(D)
                    dz, g = paired_effect_size(D)
                    row = {"test_type": test_type, "condition": c, "metric": m if target == "node" else EDGE_METRIC,
                           "mean_consensus": mean_cons, "mean_null": mean_null, "mean_D": float(mean_D),
                           "std_D": float(np.nanstd(D)), "p_raw": float(p), "cohens_dz": float(dz),
                           "hedges_g": g, "cliffs_delta": float(cliffs_delta(D)),
                           "n_seeds": int(np.sum(np.isfinite(D)))}
                    if target == "node":
                        row["n_nodes"] = n_per_cond[c]
                    else:
                        row["n_edges"] = n_per_cond[c]
                    rows.append(row)
        return rows

    global_rows = global_tests(across_seed_cons, across_seed_null, CONDITIONS,
                               across_seed_full, METRICS, n_nodes_per, "node")
    edge_global_rows = global_tests(across_seed_edge_cons, across_seed_edge_null, EDGE_CONDITIONS,
                                    across_seed_full_trueprob, [EDGE_METRIC],
                                    {c: len(EDGE_CONSENSUS[c]) for c in EDGE_CONDITIONS}, "edge")

    df_fold, df_seed = pd.DataFrame(rows_fold), pd.DataFrame(rows_seed)
    df_global = pd.DataFrame(global_rows)
    df_global["q_BH_FDR"] = np.nan
    for test_type in ("keep", "remove"):
        for c in CONDITIONS:
            mask = (df_global["test_type"] == test_type) & (df_global["condition"] == c)
            df_global.loc[df_global.index[mask], "q_BH_FDR"] = bh_fdr(
                df_global.loc[mask, "p_raw"].values.astype(float))
    df_global["significant_FDR_0p05"] = df_global["q_BH_FDR"] < 0.05

    df_edge_global = pd.DataFrame(edge_global_rows)
    if len(df_edge_global) > 0:
        df_edge_global["q_BH_FDR"] = np.nan
        for test_type in ("keep", "remove"):
            for c in EDGE_CONDITIONS:
                mask = (df_edge_global["test_type"] == test_type) & (df_edge_global["condition"] == c)
                df_edge_global.loc[df_edge_global.index[mask], "q_BH_FDR"] = bh_fdr(
                    df_edge_global.loc[mask, "p_raw"].values.astype(float))
        df_edge_global["significant_FDR_0p05"] = df_edge_global["q_BH_FDR"] < 0.05

    # ----- Save -----
    df_fold.to_csv(os.path.join(OUT_DIR, "per_fold_raw.csv"), index=False)
    df_seed.to_csv(os.path.join(OUT_DIR, "per_seed.csv"), index=False)
    df_global.to_csv(os.path.join(OUT_DIR, "global_tests_raw_and_fdr.csv"), index=False)
    pd.DataFrame(rows_edge_fold).to_csv(os.path.join(OUT_DIR, "edge_per_fold_raw.csv"), index=False)
    pd.DataFrame(rows_edge_seed).to_csv(os.path.join(OUT_DIR, "edge_per_seed.csv"), index=False)
    df_edge_global.to_csv(os.path.join(OUT_DIR, "edge_global_tests_raw_and_fdr.csv"), index=False)

    out_xlsx = os.path.join(OUT_DIR, "consensus_biomarker_validation.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_global.to_excel(w, sheet_name="Global_Test", index=False)
        df_edge_global.to_excel(w, sheet_name="Edge_Global_Test", index=False)
        df_seed.to_excel(w, sheet_name="Per_Seed", index=False)
        df_fold.to_excel(w, sheet_name="Per_Fold_Raw", index=False)

    print("\nSaved:", out_xlsx)
    return df_fold, df_seed, df_global


if __name__ == "__main__":
    run_validation()
