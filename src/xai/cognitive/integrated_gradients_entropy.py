"""Behavioural Integrated Gradients on node attention (alpha) + entropy delta H.

For the behaviour-gated GAT, this attributes each node's attention weight to the
behavioural graph features via Integrated Gradients, and contrasts the attention
entropy of the gated vs. zero-behaviour model.

Computes (restricted to consensus ROIs, across seeds):
  (A) IG magnitude per node:  sum_j |IG_{i,j}|
  (B) IG signed per node:     sum_j IG_{i,j}
  (C) IG per behavioural feature
  (D) IG Top-K rank stability
  (E) Entropy delta H = H_gate - H_nogate

Stats: sign-flip permutation tests with BH-FDR, per class and class-difference.
Self-contained; paths come from ``config``.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.utils import softmax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# ============================
# PATHS
# ============================
GATE_DIR = str(config.GATE_RESULTS_DIR)
NOGATE_DIR = str(config.ZERO_RESULTS_DIR)
OUT_DIR = os.path.join(str(config.XAI_RESULTS_DIR), "behavioral_explainability")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV_IG_SEEDLEVEL_MAG = os.path.join(OUT_DIR, "ig_alpha_seed_level_mag.csv")
OUT_CSV_IG_SEEDLEVEL_SIGNED = os.path.join(OUT_DIR, "ig_alpha_seed_level_signed.csv")
OUT_CSV_IG_STATS_PERCLASS_MAG = os.path.join(OUT_DIR, "ig_alpha_stats_perclass_mag_fdr.csv")
OUT_CSV_IG_STATS_PERCLASS_SIGNED = os.path.join(OUT_DIR, "ig_alpha_stats_perclass_signed_fdr.csv")
OUT_CSV_IG_CLASSDIFF_MAG = os.path.join(OUT_DIR, "ig_alpha_classdiff_seed_level_mag.csv")
OUT_CSV_IG_CLASSDIFF_SIGNED = os.path.join(OUT_DIR, "ig_alpha_classdiff_seed_level_signed.csv")
OUT_CSV_IG_STATS_CLASSDIFF_MAG = os.path.join(OUT_DIR, "ig_alpha_classdiff_stats_mag_fdr.csv")
OUT_CSV_IG_STATS_CLASSDIFF_SIGNED = os.path.join(OUT_DIR, "ig_alpha_classdiff_stats_signed_fdr.csv")
OUT_CSV_IG_PER_METRIC = os.path.join(OUT_DIR, "ig_alpha_seed_level_per_metric.csv")
OUT_H_SUBJ = os.path.join(OUT_DIR, "entropy_deltaH_subject_level.csv")
OUT_H_SEED = os.path.join(OUT_DIR, "entropy_deltaH_seed_level.csv")
OUT_H_STATS = os.path.join(OUT_DIR, "entropy_deltaH_stats.csv")
OUT_H_PLOT = os.path.join(OUT_DIR, "entropy_deltaH_plots.png")
OUT_IG_STAB_PERCLASS = os.path.join(OUT_DIR, "ig_rank_stability_perclass.csv")
OUT_IG_STAB_CLASSDIFF = os.path.join(OUT_DIR, "ig_rank_stability_classdiff.csv")
OUT_XLSX = os.path.join(OUT_DIR, "rank_outputs.xlsx")

# ============================
# CONFIG
# ============================
HC_LABEL, SZ_LABEL = 0, 1
LABEL_NAME = {0: "HC", 1: "SZ"}

IG_STEPS = 64
DEVICE = "cpu"
MIN_SEEDS_FOR_TEST = 6
MIN_SEEDS_FOR_STAB = 6
N_PERM = 20000
RNG_SEED = 42
TOP_K = 3
EPS = 1e-12

CONSENSUS_NODES = [9, 22, 27, 69, 70, 72, 75, 83, 86, 87, 88, 96, 101]
CONSENSUS_ROIS = sorted([x + 1 for x in CONSENSUS_NODES])
RESTRICT_TO_CONSENSUS = True

# Behavioural feature names aligned with best_graph_feat_mask [0, 1, 1, 1].
BEHAV_FEAT_NAMES = ["RTSD", "sigma", "tau"]

torch.set_grad_enabled(True)
torch.manual_seed(0)
np.random.seed(0)


# ============================
# STATS HELPERS
# ============================
def signflip_pvalue(x, n_perm=N_PERM, rng_seed=RNG_SEED, alternative="two-sided"):
    x = np.asarray(x, float)
    obs = float(np.mean(x))
    rng = np.random.default_rng(rng_seed)
    perm = (rng.choice([-1.0, 1.0], size=(n_perm, len(x))) * x).mean(axis=1)
    if alternative == "greater":
        p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(perm <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_r = np.empty(m, float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, (m / (i + 1)) * ranked[i])
        q_r[i] = prev
    q = np.empty(m, float)
    q[order] = q_r
    return q


def bootstrap_ci_mean(x, n_boot=20000, ci=0.95, rng_seed=123):
    x = np.asarray(x, float)
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    if n < 2:
        return float(np.mean(x)), np.nan, np.nan
    boot = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return float(np.mean(x)), float(np.quantile(boot, (1 - ci) / 2)), float(np.quantile(boot, 1 - (1 - ci) / 2))


# ============================
# IO HELPERS
# ============================
def find_seed_files(results_dir):
    out = {}
    for fp in sorted(glob.glob(os.path.join(results_dir, "all_results_*_seed*.pth"))):
        m = re.search(r"_seed(\d+)\.pth$", os.path.basename(fp))
        if m:
            out[int(m.group(1))] = fp
    return out


def load_pth(fp):
    return torch.load(fp, map_location="cpu", weights_only=False)


def get_df_outer(results):
    df = results.get("df_outer")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def get_test_graphs_for_fold(results, row, fold_idx):
    g = row.get("Test_Data_outer") if isinstance(row, (pd.Series, dict)) else None
    if g is not None and hasattr(g, "__len__") and len(g) > 0:
        return g
    return results["Test_Data_outer"][fold_idx]


def get_trainval_graphs_for_fold(results, row, fold_idx):
    g = row.get("TrainVal_Data_outer") if isinstance(row, (pd.Series, dict)) else None
    if g is not None and hasattr(g, "__len__") and len(g) > 0:
        return g
    tv = results.get("TrainVal_Data_outer")
    return tv[fold_idx] if tv is not None else []


def build_alpha_map(row):
    sids, alphas = row.get("node_alpha_subject_id"), row.get("node_alpha")
    if sids is None or alphas is None:
        raise KeyError("Missing node_alpha or node_alpha_subject_id")
    if isinstance(sids, np.ndarray):
        sids = sids.tolist()
    if isinstance(alphas, np.ndarray):
        alphas = alphas.tolist()
    return {str(s): np.asarray(a, float) for s, a in zip(sids, alphas)}


def roi_ids_per_node_from_graph(g):
    r2n = getattr(g, "roi_to_node")
    roi_ids = np.full(int(g.x.shape[0]), -1, dtype=int)
    for roi_key, node_idx in r2n.items():
        m = re.match(r"ROI_(\d+)$", str(roi_key))
        if not m:
            raise ValueError(f"Unexpected roi key: {roi_key}")
        roi_ids[int(node_idx)] = int(m.group(1))
    if np.any(roi_ids < 0):
        raise ValueError("roi_to_node missing nodes")
    return roi_ids


def local_consensus_indices_from_graph(g, consensus_rois):
    roi_ids = roi_ids_per_node_from_graph(g)
    return [i for i, r in enumerate(roi_ids) if int(r) in consensus_rois]


def sanitize_edge_index(edge_index, num_nodes, device):
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index)
    edge_index = edge_index.to(device).long()
    if edge_index.size(1) == 2 and edge_index.size(0) != 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.numel() > 0:
        edge_index = edge_index.clamp(0, num_nodes - 1)
    return edge_index


def apply_node_feature_mask(x, mask):
    return x[:, torch.tensor(mask, dtype=torch.bool, device=x.device)]


def apply_graph_feature_mask(gf, mask):
    return gf[:, torch.tensor(mask, dtype=torch.bool, device=gf.device)]


def sample_c_baselines(trainval_graphs, mask, max_B=64, device="cpu"):
    feats = []
    for g in trainval_graphs:
        feats.append(apply_graph_feature_mask(g.graph_feat.view(1, -1).float(), mask).squeeze(0))
        if len(feats) >= max_B:
            break
    return torch.stack(feats).to(device) if feats else None


# ============================
# MODEL (only layers + gate_nn needed for the attention wrapper)
# ============================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, graph_feat_dim,
                 num_layers=2, heads=1, use_layernorm=False, concat=True, pool="attn"):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.use_layernorm = use_layernorm
        self.dropout = dropout
        out_dim = hidden_channels * heads if concat else hidden_channels
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=concat))
        if use_layernorm:
            self.norms.append(LayerNorm(out_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_dim, hidden_channels, heads=heads, dropout=dropout, concat=concat))
            if use_layernorm:
                self.norms.append(LayerNorm(out_dim))
        self.gate_nn = nn.Sequential(nn.Linear(out_dim + graph_feat_dim, out_dim),
                                     nn.ReLU(), nn.Linear(out_dim, 1))
        self.fc = nn.Linear(out_dim, 1)


class GATWrapper(nn.Module):
    """Captum wrapper: attribute node `node_idx`'s attention weight to behaviour."""

    def __init__(self, model, x, edge_index):
        super().__init__()
        self.model = model
        self.x = x
        self.edge_index = edge_index

    def forward(self, c_input, node_idx_arg):
        x_curr = self.x
        for i, conv in enumerate(self.model.layers):
            x_curr = conv(x_curr, self.edge_index)
            if self.model.use_layernorm:
                x_curr = self.model.norms[i](x_curr)
            x_curr = F.elu(x_curr)
        N, B = x_curr.shape[0], c_input.shape[0]
        gate_in = torch.cat([x_curr.unsqueeze(0).expand(B, N, -1),
                             c_input.unsqueeze(1).expand(B, N, -1)], dim=2)
        alpha = F.softmax(self.model.gate_nn(gate_in).squeeze(-1), dim=1)
        idx = int(node_idx_arg[0].item()) if isinstance(node_idx_arg, torch.Tensor) else int(node_idx_arg)
        return alpha[:, idx].unsqueeze(1)


def compute_ig(model, x, edge_index, c_in, c_base_shap):
    num_nodes, feat_dim = x.shape[0], c_in.shape[1]
    attr = torch.zeros((num_nodes, feat_dim), device=c_in.device)
    explainer = IntegratedGradients(GATWrapper(model, x, edge_index))
    baselines = c_base_shap.mean(dim=0, keepdim=True) if c_base_shap.shape[0] > 1 else c_base_shap
    for i in range(num_nodes):
        a = explainer.attribute(inputs=c_in, baselines=baselines, additional_forward_args=(i,),
                                n_steps=IG_STEPS, internal_batch_size=16)
        attr[i] = a.detach().squeeze(0)
    return attr


# ============================
# CORE STATS (reused for mag / signed / per-metric)
# ============================
def perclass_stats_from_df(df_seed, mean_col, test_type, group_cols=None):
    group_cols = group_cols or []
    rows = []
    full_group_by = group_cols + ["class_label"]

    for keys, df_c in df_seed.groupby(full_group_by):
        grp_dict = dict(zip(full_group_by, keys)) if isinstance(keys, tuple) else {full_group_by[0]: keys}
        y = grp_dict["class_label"]
        roi_counts = df_c.groupby("roi")["seed"].nunique()
        df_roi = df_c[df_c["roi"].isin(roi_counts[roi_counts >= MIN_SEEDS_FOR_TEST].index)]

        for roi in sorted(df_roi["roi"].unique()):
            vals = df_roi[df_roi["roi"] == roi].sort_values("seed")[mean_col].to_numpy(float)
            row = {**{k: grp_dict[k] for k in group_cols},
                   "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                   "roi": int(roi), "n_seeds": int(len(vals)),
                   "mean_over_seeds": float(np.mean(vals)),
                   "std_over_seeds": float(np.std(vals, ddof=1)),
                   "median_over_seeds": float(np.median(vals))}
            if test_type == "magnitude":
                _, p_gt = signflip_pvalue(vals, alternative="greater")
                _, p_two = signflip_pvalue(vals, alternative="two-sided")
                row.update({"p_one_sided_gt0": p_gt, "p_two_sided": p_two})
            else:
                _, p_two = signflip_pvalue(vals, alternative="two-sided")
                _, p_gt = signflip_pvalue(vals, alternative="greater")
                _, p_lt = signflip_pvalue(vals, alternative="less")
                row.update({"p_two_sided": p_two, "p_one_sided_gt0": p_gt, "p_one_sided_lt0": p_lt})
            rows.append(row)

    df_stats = pd.DataFrame(rows)
    if len(df_stats) == 0:
        return df_stats

    for _, grp in df_stats.groupby(full_group_by):
        idx = grp.index
        if test_type == "magnitude":
            df_stats.loc[idx, "q_one_sided_bh"] = bh_fdr(grp["p_one_sided_gt0"].to_numpy())
            df_stats.loc[idx, "q_two_sided_bh"] = bh_fdr(grp["p_two_sided"].to_numpy())
        else:
            df_stats.loc[idx, "q_two_sided_bh"] = bh_fdr(grp["p_two_sided"].to_numpy())
            df_stats.loc[idx, "q_one_sided_gt0_bh"] = bh_fdr(grp["p_one_sided_gt0"].to_numpy())
            df_stats.loc[idx, "q_one_sided_lt0_bh"] = bh_fdr(grp["p_one_sided_lt0"].to_numpy())

    return df_stats.sort_values(full_group_by + ["roi"]).reset_index(drop=True)


def classdiff_stats_from_df(df_seed, mean_col, group_cols=None):
    """D_seed = SZ - HC per (group_cols + roi) across seeds, FDR within group."""
    group_cols = group_cols or []
    rows = []
    iterator = df_seed.groupby(group_cols) if group_cols else [(None, df_seed)]

    for keys, df_grp in iterator:
        grp_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))) if group_cols else {}
        df_hc = df_grp[df_grp["class_label"] == HC_LABEL][["seed", "roi", mean_col]].rename(columns={mean_col: "val_HC"})
        df_sz = df_grp[df_grp["class_label"] == SZ_LABEL][["seed", "roi", mean_col]].rename(columns={mean_col: "val_SZ"})
        df_pair = pd.merge(df_sz, df_hc, on=["seed", "roi"], how="inner")
        if len(df_pair) == 0:
            continue
        df_pair["D_seed"] = df_pair["val_SZ"] - df_pair["val_HC"]
        roi_counts = df_pair.groupby("roi")["seed"].nunique()
        df_pair = df_pair[df_pair["roi"].isin(roi_counts[roi_counts >= MIN_SEEDS_FOR_TEST].index)]
        for roi in sorted(df_pair["roi"].unique()):
            d = df_pair[df_pair["roi"] == roi].sort_values("seed")["D_seed"].to_numpy(float)
            _, p_two = signflip_pvalue(d, alternative="two-sided")
            _, p_gt = signflip_pvalue(d, alternative="greater")
            _, p_lt = signflip_pvalue(d, alternative="less")
            rows.append({**grp_dict, "roi": int(roi), "n_seeds": int(len(d)),
                         "mean_D_seed": float(np.mean(d)), "std_D_seed": float(np.std(d, ddof=1)),
                         "median_D_seed": float(np.median(d)), "p_two_sided": float(p_two),
                         "p_one_sided_SZ_gt_HC": float(p_gt), "p_one_sided_SZ_lt_HC": float(p_lt)})

    df_cd = pd.DataFrame(rows)
    if len(df_cd) == 0:
        return df_cd
    groups = df_cd.groupby(group_cols) if group_cols else [(None, df_cd)]
    for _, grp in groups:
        idx = grp.index if group_cols else df_cd.index
        df_cd.loc[idx, "q_two_sided_bh"] = bh_fdr(df_cd.loc[idx, "p_two_sided"].to_numpy(float))
        df_cd.loc[idx, "q_SZ_gt_HC_bh"] = bh_fdr(df_cd.loc[idx, "p_one_sided_SZ_gt_HC"].to_numpy(float))
        df_cd.loc[idx, "q_SZ_lt_HC_bh"] = bh_fdr(df_cd.loc[idx, "p_one_sided_SZ_lt_HC"].to_numpy(float))
    return df_cd.sort_values((group_cols or []) + ["q_two_sided_bh"]).reset_index(drop=True)


# ============================
# IG EXTRACTION
# ============================
def run_ig_extraction():
    gate_files = find_seed_files(GATE_DIR)
    nogate_files = find_seed_files(NOGATE_DIR)
    common_seeds = sorted(set(gate_files) & set(nogate_files))
    if not common_seeds:
        raise RuntimeError("No common seeds.")
    print("Seeds:", common_seeds)

    seed_level_mag, seed_level_signed, seed_level_per_metric = {}, {}, {}

    for seed in common_seeds:
        print(f"\n=== Seed {seed} ===")
        res_g = load_pth(gate_files[seed])
        df_g = get_df_outer(res_g)
        models_state = res_g["models"]

        for fold_idx in range(len(df_g)):
            row = df_g.iloc[fold_idx]
            params = row["params"]
            best_graph_feat_mask = row["best_graph_feat_mask"]
            best_node_mask = row["best_node_mask"]

            model = GAT(in_channels=int(sum(best_node_mask)), hidden_channels=params["hidden_dim"],
                        dropout=params["dropout"], graph_feat_dim=int(sum(best_graph_feat_mask)),
                        num_layers=params["num_layers"], heads=params.get("heads", 1)).to(DEVICE)
            model.load_state_dict(models_state[fold_idx])
            model.eval()

            test_graphs = get_test_graphs_for_fold(res_g, row, fold_idx)
            trainval_graphs = get_trainval_graphs_for_fold(res_g, row, fold_idx)

            for g in test_graphs:
                y = int(g.y.item())
                x = apply_node_feature_mask(g.x, best_node_mask).to(DEVICE)
                edge_index = sanitize_edge_index(g.edge_index, x.size(0), DEVICE)
                c_in = apply_graph_feature_mask(g.graph_feat.view(1, -1).float().to(DEVICE), best_graph_feat_mask)

                c_base = sample_c_baselines(trainval_graphs, best_graph_feat_mask, max_B=64, device=DEVICE)
                if c_base is None or c_base.size(0) < 2:
                    c_base = torch.zeros_like(c_in).repeat(8, 1)

                try:
                    attr_np = compute_ig(model, x, edge_index, c_in, c_base).detach().cpu().numpy()
                except Exception as e:
                    print(f"  [ERR IG] {e}")
                    continue
                mag_np = np.abs(attr_np).sum(axis=1)
                sgn_np = attr_np.sum(axis=1)
                roi_ids = roi_ids_per_node_from_graph(g)

                for node_idx, roi in enumerate(roi_ids):
                    roi = int(roi)
                    if RESTRICT_TO_CONSENSUS and roi not in CONSENSUS_ROIS:
                        continue
                    seed_level_mag.setdefault((seed, y, roi), []).append(float(mag_np[node_idx]))
                    seed_level_signed.setdefault((seed, y, roi), []).append(float(sgn_np[node_idx]))
                    for feat_idx in range(attr_np.shape[1]):
                        seed_level_per_metric.setdefault((seed, y, roi, feat_idx), []).append(
                            float(attr_np[node_idx, feat_idx]))

    print("\nIG extraction done.")
    return seed_level_mag, seed_level_signed, seed_level_per_metric


def build_seed_table(seed_level_dict, value_name):
    rows = []
    for (seed, y, roi), vals in seed_level_dict.items():
        v = np.asarray(vals, float)
        rows.append({"seed": int(seed), "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                     "roi": int(roi), "n_subject_nodes": int(len(v)),
                     f"mean_{value_name}": float(v.mean()), f"median_{value_name}": float(np.median(v)),
                     f"std_{value_name}": float(v.std(ddof=1))})
    return pd.DataFrame(rows)


def build_per_metric_table(seed_level_per_metric):
    rows = []
    for (seed, y, roi, feat_idx), vals in seed_level_per_metric.items():
        v = np.asarray(vals, float)
        feat_name = BEHAV_FEAT_NAMES[feat_idx] if feat_idx < len(BEHAV_FEAT_NAMES) else f"feat_{feat_idx}"
        rows.append({"seed": int(seed), "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                     "roi": int(roi), "feat_idx": int(feat_idx), "feat_name": feat_name,
                     "n_subject_nodes": int(len(v)), "mean_ig_signed": float(v.mean()),
                     "mean_ig_mag": float(np.abs(v).mean()), "std_ig_signed": float(v.std(ddof=1))})
    return pd.DataFrame(rows)


# ============================
# IG STABILITY
# ============================
def rank_stability_perclass(df_ig_seed, top_k=TOP_K, min_seeds=MIN_SEEDS_FOR_STAB):
    out_rows = []
    for y in sorted(df_ig_seed["class_label"].unique()):
        df_c = df_ig_seed[df_ig_seed["class_label"] == y].copy()
        roi_counts = df_c.groupby("roi")["seed"].nunique()
        df_c = df_c[df_c["roi"].isin(roi_counts[roi_counts >= min_seeds].index)]
        seeds = sorted(df_c["seed"].unique())
        if not seeds:
            continue
        roi_list = sorted(df_c["roi"].unique())
        ranks_dict = {int(r): [] for r in roi_list}
        topk_dict = {int(r): 0 for r in roi_list}
        for s in seeds:
            sub = df_c[df_c["seed"] == s].sort_values("mean_ig_mag", ascending=False).reset_index(drop=True)
            sub["rank"] = np.arange(1, len(sub) + 1)
            for _, rr in sub.iterrows():
                ranks_dict[int(rr["roi"])].append(int(rr["rank"]))
                if int(rr["rank"]) <= top_k:
                    topk_dict[int(rr["roi"])] += 1
        n_seeds = len(seeds)
        for roi_id in roi_list:
            rnks = np.asarray(ranks_dict[roi_id], float)
            out_rows.append({"class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                             "roi": roi_id, "topk": top_k, "n_seeds_total": n_seeds,
                             "n_seeds_present": len(rnks), "n_topk": topk_dict[roi_id],
                             "freq_topk": topk_dict[roi_id] / n_seeds,
                             "median_rank": float(np.median(rnks)) if len(rnks) else np.nan,
                             "mean_rank": float(np.mean(rnks)) if len(rnks) else np.nan,
                             "std_rank": float(np.std(rnks, ddof=1)) if len(rnks) > 1 else np.nan})
    df = pd.DataFrame(out_rows)
    return df.sort_values(["class_label", "freq_topk", "median_rank"],
                          ascending=[True, False, True]) if len(df) else df


def rank_stability_classdiff(df_ig_seed, top_k=TOP_K, min_seeds=MIN_SEEDS_FOR_STAB):
    df_hc = df_ig_seed[df_ig_seed["class_label"] == HC_LABEL][["seed", "roi", "mean_ig_mag"]].rename(columns={"mean_ig_mag": "ig_hc"})
    df_sz = df_ig_seed[df_ig_seed["class_label"] == SZ_LABEL][["seed", "roi", "mean_ig_mag"]].rename(columns={"mean_ig_mag": "ig_sz"})
    df_pair = pd.merge(df_sz, df_hc, on=["seed", "roi"], how="inner")
    if len(df_pair) == 0:
        return pd.DataFrame()
    df_pair["absD"] = np.abs(df_pair["ig_sz"] - df_pair["ig_hc"])
    roi_counts = df_pair.groupby("roi")["seed"].nunique()
    df_pair = df_pair[df_pair["roi"].isin(roi_counts[roi_counts >= min_seeds].index)]
    seeds = sorted(df_pair["seed"].unique())
    roi_list = sorted(df_pair["roi"].unique())
    ranks_dict = {int(r): [] for r in roi_list}
    topk_dict = {int(r): 0 for r in roi_list}
    absD_dict = {int(r): [] for r in roi_list}
    for s in seeds:
        sub = df_pair[df_pair["seed"] == s].sort_values("absD", ascending=False).reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        for _, rr in sub.iterrows():
            roi_id = int(rr["roi"])
            ranks_dict[roi_id].append(int(rr["rank"]))
            absD_dict[roi_id].append(float(rr["absD"]))
            if int(rr["rank"]) <= top_k:
                topk_dict[roi_id] += 1
    n_seeds = len(seeds)
    out_rows = []
    for roi_id in roi_list:
        rnks = np.asarray(ranks_dict[roi_id], float)
        out_rows.append({"roi": roi_id, "topk": top_k, "n_seeds_total": n_seeds,
                         "n_seeds_present": len(rnks), "n_topk": topk_dict[roi_id],
                         "freq_topk": topk_dict[roi_id] / n_seeds,
                         "median_rank": float(np.median(rnks)) if len(rnks) else np.nan,
                         "mean_abs_D_seed": float(np.mean(absD_dict[roi_id])) if absD_dict[roi_id] else np.nan})
    df = pd.DataFrame(out_rows)
    return df.sort_values(["freq_topk", "median_rank"], ascending=[False, True]) if len(df) else df


# ============================
# ENTROPY DELTA H
# ============================
def entropy_from_alpha(alpha, eps=EPS):
    a = np.clip(np.asarray(alpha, float), eps, 1.0)
    a = a / a.sum()
    return float(-np.sum(a * np.log(a)))


def run_entropy_deltaH():
    gate_files = find_seed_files(GATE_DIR)
    nogate_files = find_seed_files(NOGATE_DIR)
    common_seeds = sorted(set(gate_files) & set(nogate_files))
    subj_rows = []

    for seed in common_seeds:
        res_g, res_n = load_pth(gate_files[seed]), load_pth(nogate_files[seed])
        df_g, df_n = get_df_outer(res_g), get_df_outer(res_n)
        for fold_idx in range(min(len(df_g), len(df_n))):
            row_g, row_n = df_g.iloc[fold_idx], df_n.iloc[fold_idx]
            alpha_g_map, alpha_n_map = build_alpha_map(row_g), build_alpha_map(row_n)
            for g in get_test_graphs_for_fold(res_g, row_g, fold_idx):
                sid = str(getattr(g, "subject_id"))
                y = int(g.y.item())
                if sid not in alpha_g_map or sid not in alpha_n_map:
                    continue
                a_g, a_n = alpha_g_map[sid], alpha_n_map[sid]
                if RESTRICT_TO_CONSENSUS:
                    keep = local_consensus_indices_from_graph(g, CONSENSUS_ROIS)
                    if not keep:
                        continue
                    a_g, a_n = a_g[keep], a_n[keep]
                H_g, H_n = entropy_from_alpha(a_g), entropy_from_alpha(a_n)
                subj_rows.append({"seed": seed, "fold": fold_idx + 1, "subject_id": sid,
                                  "class_label": y, "class_name": LABEL_NAME.get(y, str(y)),
                                  "H_gate": H_g, "H_nogate": H_n, "deltaH": H_g - H_n, "n_nodes": len(a_g)})

    df_subj = pd.DataFrame(subj_rows)
    df_subj.to_csv(OUT_H_SUBJ, index=False)

    seed_rows = []
    for (s, y), sub in df_subj.groupby(["seed", "class_label"]):
        seed_rows.append({"seed": int(s), "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                          "n_subjects": int(sub["subject_id"].nunique()),
                          "mean_deltaH": float(sub["deltaH"].mean()),
                          "std_deltaH": float(sub["deltaH"].std(ddof=1))})
    df_seed = pd.DataFrame(seed_rows)
    df_seed.to_csv(OUT_H_SEED, index=False)

    stats_rows = []
    for y in (HC_LABEL, SZ_LABEL):
        vals = df_seed[df_seed["class_label"] == y].sort_values("seed")["mean_deltaH"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        m, ci_lo, ci_hi = bootstrap_ci_mean(vals)
        _, p_two = signflip_pvalue(vals, alternative="two-sided")
        _, p_lt = signflip_pvalue(vals, alternative="less")
        stats_rows.append({"test": "per_class_vs0", "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                           "n_seeds": len(vals), "mean_over_seeds": float(np.mean(vals)),
                           "std_over_seeds": float(np.std(vals, ddof=1)), "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                           "p_two_sided": p_two, "p_one_sided_less0": p_lt})

    df_hc_h = df_seed[df_seed["class_label"] == HC_LABEL][["seed", "mean_deltaH"]].rename(columns={"mean_deltaH": "dH_HC"})
    df_sz_h = df_seed[df_seed["class_label"] == SZ_LABEL][["seed", "mean_deltaH"]].rename(columns={"mean_deltaH": "dH_SZ"})
    df_pair = pd.merge(df_sz_h, df_hc_h, on="seed", how="inner")
    if len(df_pair) > 0:
        d = (df_pair["dH_SZ"] - df_pair["dH_HC"]).to_numpy(float)
        m, ci_lo, ci_hi = bootstrap_ci_mean(d)
        _, p_two = signflip_pvalue(d, alternative="two-sided")
        _, p_gt = signflip_pvalue(d, alternative="greater")
        stats_rows.append({"test": "class_diff", "class_label": -1, "class_name": "SZ-HC",
                           "n_seeds": len(d), "mean_over_seeds": float(np.mean(d)),
                           "std_over_seeds": float(np.std(d, ddof=1)), "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                           "p_two_sided": p_two, "p_one_sided_less0": p_gt})
    pd.DataFrame(stats_rows).to_csv(OUT_H_STATS, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for y in (HC_LABEL, SZ_LABEL):
        vals = df_seed[df_seed["class_label"] == y].sort_values("seed")["mean_deltaH"].to_numpy(float)
        ax1.scatter(np.full(len(vals), y), vals, label=LABEL_NAME[y], alpha=0.9)
        if len(vals):
            ax1.hlines(vals.mean(), y - 0.15, y + 0.15)
    ax1.set_xticks([HC_LABEL, SZ_LABEL])
    ax1.set_xticklabels([LABEL_NAME[HC_LABEL], LABEL_NAME[SZ_LABEL]])
    ax1.set_title("Seed-level mean delta H")
    ax1.set_ylabel("delta H (nats)")
    ax1.grid(alpha=0.3)
    if len(df_pair) > 0:
        d_vals = (df_pair["dH_SZ"] - df_pair["dH_HC"]).to_numpy(float)
        ax2.scatter(np.zeros(len(d_vals)), d_vals, alpha=0.9)
        ax2.hlines(d_vals.mean(), -0.15, 0.15)
        ax2.set_xticks([0])
        ax2.set_xticklabels(["SZ - HC"])
        ax2.set_title("delta H interaction")
        ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_H_PLOT, dpi=200)
    plt.close(fig)
    return df_subj, df_seed


def main():
    seed_level_mag, seed_level_signed, seed_level_per_metric = run_ig_extraction()

    df_seed_mag = build_seed_table(seed_level_mag, "ig_mag")
    df_seed_signed = build_seed_table(seed_level_signed, "ig_signed")
    df_seed_pm = build_per_metric_table(seed_level_per_metric)
    df_seed_mag.to_csv(OUT_CSV_IG_SEEDLEVEL_MAG, index=False)
    df_seed_signed.to_csv(OUT_CSV_IG_SEEDLEVEL_SIGNED, index=False)
    df_seed_pm.to_csv(OUT_CSV_IG_PER_METRIC, index=False)

    df_mag_pc = perclass_stats_from_df(df_seed_mag, "mean_ig_mag", "magnitude")
    df_sgn_pc = perclass_stats_from_df(df_seed_signed, "mean_ig_signed", "signed")
    df_mag_cd = classdiff_stats_from_df(df_seed_mag, "mean_ig_mag")
    df_sgn_cd = classdiff_stats_from_df(df_seed_signed, "mean_ig_signed")
    df_mag_pc.to_csv(OUT_CSV_IG_STATS_PERCLASS_MAG, index=False)
    df_sgn_pc.to_csv(OUT_CSV_IG_STATS_PERCLASS_SIGNED, index=False)
    df_mag_cd.to_csv(OUT_CSV_IG_STATS_CLASSDIFF_MAG, index=False)
    df_sgn_cd.to_csv(OUT_CSV_IG_STATS_CLASSDIFF_SIGNED, index=False)

    df_pm_mag_pc = perclass_stats_from_df(df_seed_pm, "mean_ig_mag", "magnitude", group_cols=["feat_name"])
    df_pm_sgn_pc = perclass_stats_from_df(df_seed_pm, "mean_ig_signed", "signed", group_cols=["feat_name"])
    df_pm_mag_cd = classdiff_stats_from_df(df_seed_pm, "mean_ig_mag", group_cols=["feat_name"])
    df_pm_sgn_cd = classdiff_stats_from_df(df_seed_pm, "mean_ig_signed", group_cols=["feat_name"])

    df_stab = rank_stability_perclass(df_seed_mag)
    df_cd_s = rank_stability_classdiff(df_seed_mag)
    df_stab.to_csv(OUT_IG_STAB_PERCLASS, index=False)
    df_cd_s.to_csv(OUT_IG_STAB_CLASSDIFF, index=False)

    run_entropy_deltaH()

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_stab.to_excel(writer, sheet_name="stability_perclass", index=False)
        df_cd_s.to_excel(writer, sheet_name="stability_classdiff", index=False)
        df_mag_pc.to_excel(writer, sheet_name="mag_perclass", index=False)
        df_sgn_pc.to_excel(writer, sheet_name="signed_perclass", index=False)
        df_mag_cd.to_excel(writer, sheet_name="mag_classdiff", index=False)
        df_sgn_cd.to_excel(writer, sheet_name="signed_classdiff", index=False)
        df_pm_mag_pc.to_excel(writer, sheet_name="pm_mag_perclass", index=False)
        df_pm_sgn_pc.to_excel(writer, sheet_name="pm_signed_perclass", index=False)
        df_pm_mag_cd.to_excel(writer, sheet_name="pm_mag_classdiff", index=False)
        df_pm_sgn_cd.to_excel(writer, sheet_name="pm_signed_classdiff", index=False)

    print(f"\nExcel saved: {OUT_XLSX}\nAll done.")


if __name__ == "__main__":
    main()
