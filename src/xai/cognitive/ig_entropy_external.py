"""External behavioural explainability (subject-level statistics).

Same Integrated-Gradients (behaviour -> node attention) and entropy delta-H
analyses as integrated_gradients_entropy.py, but for the external frozen-model
evaluation: statistics are computed at the SUBJECT level (not seed level).

  - IG magnitude and signed attribution per consensus node;
  - subject-level entropy delta H = H_gate - H_nogate;
  - per-class sign-flip tests vs. 0, SZ-vs-HC two-sample permutation tests, BH-FDR;
  - IG Top-K rank stability across subjects.

Output filenames mirror the seed-level pipeline so downstream plotting scripts can
be reused. Self-contained; paths come from ``config``.
"""

from __future__ import annotations

import glob
import os
import re
import sys

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

# ============================================================
# INPUTS / OUTPUTS
# ============================================================
GATE_DIR = str(config.EXTERNAL_GATE_RESULTS_DIR)
NOGATE_DIR = str(config.EXTERNAL_ZERO_RESULTS_DIR)
OUT_DIR = os.path.join(str(config.EXTERNAL_RESULTS_DIR), "behavioral_explainability")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cpu"
ATTR_METHOD = "IG"
IG_STEPS = 64
IG_MAG_MODE = "l1"

HC_LABEL, SZ_LABEL = 0, 1
LABEL_NAME = {0: "HC", 1: "SZ"}

N_PERM = 20000
RNG_SEED = 42

CONSENSUS_NODES = [9, 22, 27, 69, 70, 72, 75, 83, 86, 87, 88, 96, 101]   # 0-based
CONSENSUS_ROIS = sorted([x + 1 for x in CONSENSUS_NODES])
RESTRICT_TO_CONSENSUS = True
EPS = 1e-12

# Output filenames kept identical to the seed-level pipeline for compatibility.
OUT_CSV_IG_SEEDLEVEL_MAG = os.path.join(OUT_DIR, "ig_alpha_seed_level_mag.csv")
OUT_CSV_IG_STATS_PERCLASS_MAG = os.path.join(OUT_DIR, "ig_alpha_stats_perclass_mag_fdr.csv")
OUT_CSV_IG_CLASSDIFF_MAG = os.path.join(OUT_DIR, "ig_alpha_classdiff_seed_level_mag.csv")
OUT_CSV_IG_STATS_CLASSDIFF_MAG = os.path.join(OUT_DIR, "ig_alpha_classdiff_stats_mag_fdr.csv")
OUT_CSV_IG_SEEDLEVEL_SIGNED = os.path.join(OUT_DIR, "ig_alpha_seed_level_signed.csv")
OUT_CSV_IG_STATS_PERCLASS_SIGNED = os.path.join(OUT_DIR, "ig_alpha_stats_perclass_signed_fdr.csv")
OUT_CSV_IG_CLASSDIFF_SIGNED = os.path.join(OUT_DIR, "ig_alpha_classdiff_seed_level_signed.csv")
OUT_CSV_IG_STATS_CLASSDIFF_SIGNED = os.path.join(OUT_DIR, "ig_alpha_classdiff_stats_signed_fdr.csv")
OUT_H_SUBJ = os.path.join(OUT_DIR, "entropy_deltaH_subject_level.csv")
OUT_H_SEED = os.path.join(OUT_DIR, "entropy_deltaH_seed_level.csv")
OUT_H_STATS = os.path.join(OUT_DIR, "entropy_deltaH_stats.csv")
OUT_IG_STAB_PERCLASS = os.path.join(OUT_DIR, "ig_rank_stability_perclass.csv")
OUT_IG_STAB_CLASSDIFF = os.path.join(OUT_DIR, "ig_rank_stability_classdiff.csv")
OUT_XLSX = os.path.join(OUT_DIR, "rank_outputs.xlsx")

TOP_K = 3
MIN_SUBJECTS_FOR_STABILITY = 1


# ============================================================
# HELPERS
# ============================================================
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
    df = results["df_outer"]
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def get_test_graphs_for_fold(results, df_outer_row, fold_idx):
    if "Test_Data_outer" in df_outer_row and df_outer_row["Test_Data_outer"] is not None:
        g = df_outer_row["Test_Data_outer"]
        if hasattr(g, "__len__") and len(g) > 0:
            return g
    return results["Test_Data_outer"][fold_idx]


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
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.numel() > 0:
        edge_index = edge_index.clamp(min=0, max=num_nodes - 1)
    return edge_index


def apply_node_feature_mask(x, mask):
    return x[:, torch.tensor(mask, dtype=torch.bool, device=x.device)]


def apply_graph_feature_mask(gf, mask):
    return gf[:, torch.tensor(mask, dtype=torch.bool, device=gf.device)]


def sample_c_baselines_from_graphs(graphs, mask, max_B=64, device="cpu"):
    feats = []
    for g in graphs:
        feats.append(apply_graph_feature_mask(g.graph_feat.view(1, -1).float(), mask).squeeze(0))
        if len(feats) >= max_B:
            break
    return torch.stack(feats, dim=0).to(device) if feats else None


def signflip_pvalue_subject(x, n_perm=N_PERM, rng_seed=RNG_SEED, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
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


def permutation_test_group_difference(x_sz, x_hc, n_perm=N_PERM, rng_seed=RNG_SEED, alternative="two-sided"):
    x_sz = np.asarray(x_sz, float)
    x_hc = np.asarray(x_hc, float)
    x_sz = x_sz[np.isfinite(x_sz)]
    x_hc = x_hc[np.isfinite(x_hc)]
    if len(x_sz) == 0 or len(x_hc) == 0:
        return np.nan, np.nan
    obs = float(np.mean(x_sz) - np.mean(x_hc))
    all_x = np.concatenate([x_sz, x_hc])
    n_sz = len(x_sz)
    rng = np.random.default_rng(rng_seed)
    perm_stats = np.empty(n_perm, float)
    for i in range(n_perm):
        perm = rng.permutation(all_x)
        perm_stats[i] = np.mean(perm[:n_sz]) - np.mean(perm[n_sz:])
    if alternative == "greater":
        p = (np.sum(perm_stats >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(perm_stats <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm_stats) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    q = np.full_like(pvals, np.nan, float)
    ok = np.isfinite(pvals)
    if np.sum(ok) == 0:
        return q
    pv = pvals[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = np.empty(m, float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, (m / (i + 1)) * ranked[i])
        q_ranked[i] = prev
    tmp = np.empty(m, float)
    tmp[order] = q_ranked
    q[ok] = tmp
    return q


def entropy_from_alpha(alpha, eps=1e-12):
    a = np.clip(np.asarray(alpha, float), eps, 1.0)
    a = a / np.sum(a)
    return float(-np.sum(a * np.log(a)))


# ============================================================
# MODEL + Captum wrapper
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
        self.pool = pool
        out_dim = hidden_channels * heads if concat else hidden_channels
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=concat))
        if use_layernorm:
            self.norms.append(LayerNorm(out_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_dim, hidden_channels, heads=heads, dropout=dropout, concat=concat))
            if use_layernorm:
                self.norms.append(LayerNorm(out_dim))
        if pool != "attn":
            raise ValueError("This script assumes pool='attn'.")
        self.gate_nn = torch.nn.Sequential(torch.nn.Linear(out_dim + graph_feat_dim, out_dim),
                                           torch.nn.ReLU(), torch.nn.Linear(out_dim, 1))
        self.fc = torch.nn.Linear(out_dim, 1)


class GATWrapper(nn.Module):
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


def compute_attributions_captum(model, x, edge_index, c_input, c_base, method="IG"):
    num_nodes, feat_dim = x.shape[0], c_input.shape[1]
    attributions = torch.zeros((num_nodes, feat_dim), device=c_input.device)
    explainer = IntegratedGradients(GATWrapper(model, x, edge_index))
    baselines = c_base.mean(dim=0, keepdim=True) if c_base.shape[0] > 1 else c_base
    for i in range(num_nodes):
        attributions[i] = explainer.attribute(
            inputs=c_input, baselines=baselines, additional_forward_args=(i,),
            n_steps=IG_STEPS, internal_batch_size=16).detach().squeeze(0)
    return attributions


def ig_magnitude_per_node(ig, mode="l1"):
    if mode == "l1":
        return torch.sum(torch.abs(ig), dim=1)
    if mode == "l2":
        return torch.sqrt(torch.sum(ig * ig, dim=1) + 1e-12)
    raise ValueError("mode must be 'l1' or 'l2'")


def ig_signed_per_node(ig):
    return torch.sum(ig, dim=1)


# ============================================================
# ROI-WISE SUBJECT-LEVEL STATS
# ============================================================
def perclass_roi_stats_subject(df_subj, value_col, out_csv, mode="magnitude"):
    rows = []
    for y in (HC_LABEL, SZ_LABEL):
        df_c = df_subj[df_subj["class_label"] == y]
        for roi in sorted(df_c["roi"].unique()):
            vals = df_c[df_c["roi"] == roi][value_col].to_numpy(float)
            if len(vals) == 0:
                continue
            base = {"class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)), "roi": int(roi),
                    "n_subjects": int(len(vals)), "mean": float(np.mean(vals)), "median": float(np.median(vals))}
            if mode == "magnitude":
                _, p_main = signflip_pvalue_subject(vals, alternative="greater")
                _, p_two = signflip_pvalue_subject(vals, alternative="two-sided")
                base.update({"p_one_sided_gt0": float(p_main), "p_two_sided": float(p_two)})
            else:
                _, p_two = signflip_pvalue_subject(vals, alternative="two-sided")
                _, p_gt = signflip_pvalue_subject(vals, alternative="greater")
                _, p_lt = signflip_pvalue_subject(vals, alternative="less")
                base.update({"p_two_sided": float(p_two), "p_one_sided_gt0": float(p_gt), "p_one_sided_lt0": float(p_lt)})
            rows.append(base)

    df_stats = pd.DataFrame(rows)
    if len(df_stats) == 0:
        df_stats.to_csv(out_csv, index=False)
        return
    for y in (HC_LABEL, SZ_LABEL):
        m = df_stats["class_label"] == y
        if m.sum() == 0:
            continue
        if mode == "magnitude":
            df_stats.loc[m, "q_one_sided_bh"] = bh_fdr(df_stats.loc[m, "p_one_sided_gt0"].to_numpy(float))
            df_stats.loc[m, "q_two_sided_bh"] = bh_fdr(df_stats.loc[m, "p_two_sided"].to_numpy(float))
        else:
            df_stats.loc[m, "q_two_sided_bh"] = bh_fdr(df_stats.loc[m, "p_two_sided"].to_numpy(float))
            df_stats.loc[m, "q_one_sided_gt0_bh"] = bh_fdr(df_stats.loc[m, "p_one_sided_gt0"].to_numpy(float))
            df_stats.loc[m, "q_one_sided_lt0_bh"] = bh_fdr(df_stats.loc[m, "p_one_sided_lt0"].to_numpy(float))
    df_stats.to_csv(out_csv, index=False)


def classdiff_roi_stats_subject(df_subj, value_col, out_seed_csv, out_csv, mode="magnitude"):
    pair_rows, stats_rows = [], []
    for roi in sorted(df_subj["roi"].unique()):
        x_sz = df_subj[(df_subj["class_label"] == SZ_LABEL) & (df_subj["roi"] == roi)][value_col].to_numpy(float)
        x_hc = df_subj[(df_subj["class_label"] == HC_LABEL) & (df_subj["roi"] == roi)][value_col].to_numpy(float)
        x_sz, x_hc = x_sz[np.isfinite(x_sz)], x_hc[np.isfinite(x_hc)]
        for val in x_sz:
            pair_rows.append({"roi": int(roi), f"{value_col}_SZ": float(val), "group": "SZ"})
        for val in x_hc:
            pair_rows.append({"roi": int(roi), f"{value_col}_HC": float(val), "group": "HC"})
        if len(x_sz) == 0 or len(x_hc) == 0:
            continue
        obs, p_two = permutation_test_group_difference(x_sz, x_hc, alternative="two-sided")
        _, p_gt = permutation_test_group_difference(x_sz, x_hc, alternative="greater")
        row = {"roi": int(roi), "n_SZ": int(len(x_sz)), "n_HC": int(len(x_hc)),
               "mean_SZ_minus_HC": float(obs), "mean_SZ": float(np.mean(x_sz)), "mean_HC": float(np.mean(x_hc)),
               "p_two_sided": float(p_two), "p_one_sided_SZ_gt_HC": float(p_gt)}
        if mode != "magnitude":
            _, p_lt = permutation_test_group_difference(x_sz, x_hc, alternative="less")
            row["p_one_sided_SZ_lt_HC"] = float(p_lt)
        stats_rows.append(row)

    pd.DataFrame(pair_rows).to_csv(out_seed_csv, index=False)
    df_stats = pd.DataFrame(stats_rows)
    if len(df_stats) == 0:
        df_stats.to_csv(out_csv, index=False)
        return
    df_stats["q_two_sided_bh"] = bh_fdr(df_stats["p_two_sided"].to_numpy(float))
    df_stats["q_one_sided_bh"] = bh_fdr(df_stats["p_one_sided_SZ_gt_HC"].to_numpy(float))
    if mode != "magnitude":
        df_stats["q_lt0_bh"] = bh_fdr(df_stats["p_one_sided_SZ_lt_HC"].to_numpy(float))
    df_stats.to_csv(out_csv, index=False)


# ============================================================
# RANK STABILITY (subject-level)
# ============================================================
def rank_stability_perclass_subject(df_ig_subj, top_k=3, min_subjects=1):
    rows = []
    for y in sorted(df_ig_subj["class_label"].unique()):
        df_c = df_ig_subj[df_ig_subj["class_label"] == y].copy()
        roi_counts = df_c.groupby("roi")["subject_id"].nunique()
        df_c = df_c[df_c["roi"].isin(roi_counts[roi_counts >= min_subjects].index)]
        subjects = sorted(df_c["subject_id"].unique())
        if len(subjects) == 0:
            continue
        roi_list = sorted(df_c["roi"].unique())
        roi_to_ranks = {int(r): [] for r in roi_list}
        roi_to_topk = {int(r): 0 for r in roi_list}
        for sid in subjects:
            sub = df_c[df_c["subject_id"] == sid][["roi", "ig_mag"]].sort_values("ig_mag", ascending=False).reset_index(drop=True)
            sub["rank"] = np.arange(1, len(sub) + 1)
            for _, rr in sub.iterrows():
                roi_to_ranks[int(rr["roi"])].append(int(rr["rank"]))
                if int(rr["rank"]) <= top_k:
                    roi_to_topk[int(rr["roi"])] += 1
        n_subjects = len(subjects)
        for roi in roi_list:
            ranks = np.asarray(roi_to_ranks[roi], float)
            rows.append({"class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)), "roi": int(roi),
                         "topk": int(top_k), "n_subjects_total": int(n_subjects),
                         "n_subjects_present": int(len(ranks)), "n_topk": int(roi_to_topk[roi]),
                         "freq_topk": float(roi_to_topk[roi] / n_subjects) if n_subjects > 0 else np.nan,
                         "median_rank": float(np.median(ranks)) if len(ranks) else np.nan,
                         "mean_rank": float(np.mean(ranks)) if len(ranks) else np.nan})
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["class_label", "freq_topk", "median_rank"], ascending=[True, False, True]).reset_index(drop=True)
    return df


def rank_classdiff_subject(df_ig_subj, top_k=3, min_subjects=1):
    roi_counts = df_ig_subj.groupby("roi")["subject_id"].nunique()
    df_use = df_ig_subj[df_ig_subj["roi"].isin(roi_counts[roi_counts >= min_subjects].index)]
    df_hc = df_use[df_use["class_label"] == HC_LABEL].groupby("roi", as_index=False)["ig_mag"].mean().rename(columns={"ig_mag": "mean_ig_hc"})
    df_sz = df_use[df_use["class_label"] == SZ_LABEL].groupby("roi", as_index=False)["ig_mag"].mean().rename(columns={"ig_mag": "mean_ig_sz"})
    df_pair = pd.merge(df_sz, df_hc, on="roi", how="inner")
    if len(df_pair) == 0:
        return pd.DataFrame([])
    df_pair["absD"] = np.abs(df_pair["mean_ig_sz"] - df_pair["mean_ig_hc"])
    df_pair = df_pair.sort_values("absD", ascending=False).reset_index(drop=True)
    df_pair["rank"] = np.arange(1, len(df_pair) + 1)
    rows = []
    for _, rr in df_pair.iterrows():
        rank = int(rr["rank"])
        rows.append({"roi": int(rr["roi"]), "topk": int(top_k), "freq_topk": float(1.0 if rank <= top_k else 0.0),
                     "median_rank": float(rank), "mean_abs_D_seed": float(rr["absD"])})
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    gate_files = find_seed_files(GATE_DIR)
    nogate_files = find_seed_files(NOGATE_DIR)
    common_seeds = sorted(set(gate_files) & set(nogate_files))
    if not common_seeds:
        raise RuntimeError("No common seeds found between gate and nogate dirs.")
    print("Common seeds:", common_seeds)

    subj_rows_mag, subj_rows_signed, entropy_rows = [], [], []

    for seed in common_seeds:
        print(f"\n=== Seed {seed} ===")
        res_g, res_n = load_pth(gate_files[seed]), load_pth(nogate_files[seed])
        df_g, df_n = get_df_outer(res_g), get_df_outer(res_n)
        if "models" not in res_g:
            raise KeyError("Expected res_g['models'].")
        models_state = res_g["models"]

        for fold_idx in range(min(len(df_g), len(df_n))):
            row_g, row_n = df_g.iloc[fold_idx], df_n.iloc[fold_idx]
            params = row_g["params"]
            best_graph_feat_mask = row_g["best_graph_feat_mask"]
            best_node_mask = row_g["best_node_mask"]

            model = GAT(in_channels=int(sum(best_node_mask)), hidden_channels=params["hidden_dim"],
                        dropout=params["dropout"], graph_feat_dim=int(sum(best_graph_feat_mask)),
                        num_layers=params["num_layers"], heads=params.get("heads", 1),
                        use_layernorm=False, concat=True, pool="attn").to(DEVICE)
            model.load_state_dict(models_state[fold_idx])
            model.eval()

            test_graphs = get_test_graphs_for_fold(res_g, row_g, fold_idx)
            alpha_g_map = {str(s): np.asarray(a, float) for s, a in zip(row_g["node_alpha_subject_id"], row_g["node_alpha"])}
            alpha_n_map = {str(s): np.asarray(a, float) for s, a in zip(row_n["node_alpha_subject_id"], row_n["node_alpha"])}

            c_base_shap = sample_c_baselines_from_graphs(test_graphs, best_graph_feat_mask, max_B=64, device=DEVICE)
            if c_base_shap is None or c_base_shap.size(0) < 2:
                c_base_shap = None

            for g in test_graphs:
                sid = str(getattr(g, "subject_id"))
                y = int(g.y.item())
                x = apply_node_feature_mask(g.x, best_node_mask).to(DEVICE)
                edge_index = sanitize_edge_index(g.edge_index, x.size(0), DEVICE)
                c_in = apply_graph_feature_mask(g.graph_feat.view(1, -1).float().to(DEVICE), best_graph_feat_mask)
                c_base = c_base_shap if c_base_shap is not None else torch.zeros_like(c_in).repeat(8, 1)

                attr = compute_attributions_captum(model, x, edge_index, c_in, c_base, method=ATTR_METHOD)
                mag = ig_magnitude_per_node(attr, mode=IG_MAG_MODE).detach().cpu().numpy()
                signed = ig_signed_per_node(attr).detach().cpu().numpy()
                roi_ids = roi_ids_per_node_from_graph(g)

                for node_idx, roi in enumerate(roi_ids):
                    roi = int(roi)
                    if RESTRICT_TO_CONSENSUS and roi not in CONSENSUS_ROIS:
                        continue
                    subj_rows_mag.append({"seed": int(seed), "fold": int(fold_idx + 1), "subject_id": sid,
                                          "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                                          "roi": int(roi), "ig_mag": float(mag[node_idx])})
                    subj_rows_signed.append({"seed": int(seed), "fold": int(fold_idx + 1), "subject_id": sid,
                                             "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                                             "roi": int(roi), "ig_signed": float(signed[node_idx])})

                if sid in alpha_g_map and sid in alpha_n_map:
                    a_g, a_n = alpha_g_map[sid], alpha_n_map[sid]
                    if RESTRICT_TO_CONSENSUS:
                        keep = local_consensus_indices_from_graph(g, CONSENSUS_ROIS)
                        if len(keep) == 0:
                            continue
                        a_g, a_n = a_g[keep], a_n[keep]
                    H_g, H_n = entropy_from_alpha(a_g, EPS), entropy_from_alpha(a_n, EPS)
                    entropy_rows.append({"seed": int(seed), "fold": int(fold_idx + 1), "subject_id": sid,
                                         "class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                                         "H_gate": float(H_g), "H_nogate": float(H_n),
                                         "deltaH": float(H_g - H_n), "n_nodes": int(len(a_g))})

    print("\nExtraction done.")
    df_subj_mag = pd.DataFrame(subj_rows_mag)
    df_subj_signed = pd.DataFrame(subj_rows_signed)
    df_entropy = pd.DataFrame(entropy_rows)
    df_subj_mag.to_csv(OUT_CSV_IG_SEEDLEVEL_MAG, index=False)
    df_subj_signed.to_csv(OUT_CSV_IG_SEEDLEVEL_SIGNED, index=False)
    df_entropy.to_csv(OUT_H_SUBJ, index=False)

    perclass_roi_stats_subject(df_subj_mag, "ig_mag", OUT_CSV_IG_STATS_PERCLASS_MAG, mode="magnitude")
    perclass_roi_stats_subject(df_subj_signed, "ig_signed", OUT_CSV_IG_STATS_PERCLASS_SIGNED, mode="signed")
    classdiff_roi_stats_subject(df_subj_mag, "ig_mag", OUT_CSV_IG_CLASSDIFF_MAG, OUT_CSV_IG_STATS_CLASSDIFF_MAG, mode="magnitude")
    classdiff_roi_stats_subject(df_subj_signed, "ig_signed", OUT_CSV_IG_CLASSDIFF_SIGNED, OUT_CSV_IG_STATS_CLASSDIFF_SIGNED, mode="signed")

    # Entropy subject-level stats
    entropy_stats_rows = []
    for y in (HC_LABEL, SZ_LABEL):
        vals = df_entropy[df_entropy["class_label"] == y]["deltaH"].to_numpy(float)
        if len(vals) > 0:
            _, p_two = signflip_pvalue_subject(vals, alternative="two-sided")
            _, p_less = signflip_pvalue_subject(vals, alternative="less")
            _, p_gt = signflip_pvalue_subject(vals, alternative="greater")
            entropy_stats_rows.append({"test": "per_class_vs0", "class_label": int(y),
                                       "class_name": LABEL_NAME.get(int(y), str(y)), "n_subjects": int(len(vals)),
                                       "mean": float(np.mean(vals)), "median": float(np.median(vals)),
                                       "p_two_sided": float(p_two), "p_one_sided_less0": float(p_less),
                                       "p_one_sided_gt0": float(p_gt)})

    x_sz = df_entropy[df_entropy["class_label"] == SZ_LABEL]["deltaH"].to_numpy(float)
    x_hc = df_entropy[df_entropy["class_label"] == HC_LABEL]["deltaH"].to_numpy(float)
    if len(x_sz) > 0 and len(x_hc) > 0:
        obs, p_two = permutation_test_group_difference(x_sz, x_hc, alternative="two-sided")
        _, p_gt = permutation_test_group_difference(x_sz, x_hc, alternative="greater")
        _, p_lt = permutation_test_group_difference(x_sz, x_hc, alternative="less")
        entropy_stats_rows.append({"test": "class_diff_interaction", "class_label": -1, "class_name": "SZ_minus_HC",
                                   "n_SZ": int(len(x_sz)), "n_HC": int(len(x_hc)),
                                   "mean": float(obs), "median": float(np.median(x_sz) - np.median(x_hc)),
                                   "p_two_sided": float(p_two), "p_one_sided_SZ_gt_HC": float(p_gt),
                                   "p_one_sided_SZ_lt_HC": float(p_lt)})
    pd.DataFrame(entropy_stats_rows).to_csv(OUT_H_STATS, index=False)

    entropy_seed_like_rows = []
    for y in (HC_LABEL, SZ_LABEL):
        sub = df_entropy[df_entropy["class_label"] == y]
        if len(sub) == 0:
            continue
        entropy_seed_like_rows.append({"class_label": int(y), "class_name": LABEL_NAME.get(int(y), str(y)),
                                       "n_subjects": int(sub["subject_id"].nunique()),
                                       "mean_deltaH": float(sub["deltaH"].mean()),
                                       "median_deltaH": float(sub["deltaH"].median()),
                                       "std_deltaH": float(sub["deltaH"].std(ddof=1)) if len(sub) > 1 else 0.0})
    pd.DataFrame(entropy_seed_like_rows).to_csv(OUT_H_SEED, index=False)

    # Rank stability
    df_rank_in = df_subj_mag[df_subj_mag["roi"].isin(CONSENSUS_ROIS)] if RESTRICT_TO_CONSENSUS else df_subj_mag
    df_stab = rank_stability_perclass_subject(df_rank_in, TOP_K, MIN_SUBJECTS_FOR_STABILITY)
    df_cd = rank_classdiff_subject(df_rank_in, TOP_K, MIN_SUBJECTS_FOR_STABILITY)
    df_stab.to_csv(OUT_IG_STAB_PERCLASS, index=False)
    df_cd.to_csv(OUT_IG_STAB_CLASSDIFF, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_stab.to_excel(writer, sheet_name="ig_rank_stability_perclass", index=False)
        df_cd.to_excel(writer, sheet_name="ig_rank_stability_classdiff", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
