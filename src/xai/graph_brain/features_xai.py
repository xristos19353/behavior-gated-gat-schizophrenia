"""Union-graph feature explainability: Stage 1 (IG) + Stage 2 (necessity/sufficiency).

Stage 1 - Integrated Gradients per (node x feature x class) on the consensus union
nodes (non-union nodes zeroed, class-specific baseline), with leave-one-seed-out
top-1/top-2 feature selection, sign-flip tests and BH-FDR.

Stage 2 - Necessity / sufficiency of the Stage-1 top features via per-node
interventions (zero or shuffle), scored with classification metrics and a unified
true-class-probability metric. Discovery mode aggregates mean D per seed; external
mode aggregates subject-level D.

Self-contained; paths come from ``config``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch_geometric.nn import GATConv, LayerNorm, global_mean_pool
from torch_geometric.utils import softmax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# ============================================================
# CONFIG
# ============================================================
DEFAULT_RESULTS_DIR = str(config.EXTERNAL_GATE_RESULTS_DIR)
DEFAULT_OUT_DIR = os.path.join(str(config.XAI_RESULTS_DIR), "features_xai")

# External feature-XAI was run on a single seed package; set to config.SEEDS for
# the full discovery sweep.
SEEDS = [100]

DEVICE = torch.device("cpu")
THR = 0.5
FULL_ATLAS_SIZE = config.FULL_ATLAS_SIZE

CONSENSUS_UNION = [9, 22, 27, 69, 70, 72, 75, 83, 86, 87, 88, 96, 101]
N_UNION = len(CONSENSUS_UNION)

FEATURE_NAMES = ["WMV", "GMV", "EffectSize", "ALFF", "ReHo"]
N_FEATS = len(FEATURE_NAMES)

CLASSES = {"sz": 1, "hc": 0}
METRICS = ["auc", "balanced_acc", "sensitivity", "specificity"]
STAGE2_PROB_METRICS = ["true_class_prob_mean"]


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


class GATWrapperUnionMask(torch.nn.Module):
    """Captum wrapper: zero non-union nodes before passing x to the model."""

    def __init__(self, model, edge_index, graph_feat, batch, union_mask):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.graph_feat = graph_feat
        self.batch = batch
        self.union_mask = union_mask

    def forward(self, x_batched):
        if x_batched.dim() == 3 and x_batched.shape[0] > 1:
            outputs = []
            for i in range(x_batched.shape[0]):
                x = x_batched[i].clone()
                x[~self.union_mask] = 0.0
                outputs.append(self.model(x, self.edge_index, self.graph_feat, self.batch))
            return torch.cat(outputs, dim=0)
        x = x_batched.squeeze(0).clone()
        x[~self.union_mask] = 0.0
        return self.model(x, self.edge_index, self.graph_feat, self.batch)


# ============================================================
# STATS
# ============================================================
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


def signflip_test(D, max_exact_n=16, n_mc=20000, seed=42):
    D = np.asarray(D, float)
    D = D[np.isfinite(D)]
    n = len(D)
    if n < 2:
        return (float(np.mean(D)) if n > 0 else np.nan), np.nan
    obs = float(np.mean(D))
    if n <= max_exact_n:
        means = np.array([np.mean(np.array(s, float) * D) for s in product([-1.0, 1.0], repeat=n)])
        return obs, float(np.mean(means >= obs))
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice([-1.0, 1.0], size=(n_mc, n)) * D[None, :], axis=1)
    return obs, float((np.sum(means >= obs) + 1) / (n_mc + 1))


def cohens_dz(D):
    D = np.asarray(D, float)
    D = D[np.isfinite(D)]
    if len(D) < 2:
        return np.nan
    sd = float(np.std(D, ddof=1))
    return float(np.mean(D) / sd) if sd > 0 else np.inf


def cliffs_delta(D):
    D = np.asarray(D, float)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return np.nan
    return float((np.sum(D > 0) - np.sum(D < 0)) / len(D))


def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    q = np.full_like(pvals, np.nan, float)
    idx = np.where(np.isfinite(pvals))[0]
    if idx.size == 0:
        return q
    pv = pvals[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    q_r = np.empty(m, float)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        prev = min(prev, (m / (k + 1)) * ranked[k])
        q_r[k] = prev
    tmp = np.empty_like(pv)
    tmp[order] = q_r
    q[idx] = tmp
    return q


def prob_true_class(y_true, p_sz):
    y_true = np.asarray(y_true, int)
    p_sz = np.asarray(p_sz, float)
    return np.where(y_true == 1, p_sz, 1.0 - p_sz)


def true_class_prob_metric(y_true, p_variant):
    return {"true_class_prob_mean": float(np.nanmean(prob_true_class(y_true, p_variant)))}


# ============================================================
# GRAPH HELPERS
# ============================================================
def find_seed_file(results_dir, seed):
    files = sorted(glob.glob(os.path.join(results_dir, f"all_results_*_seed{seed}.pth")))
    if not files:
        raise FileNotFoundError(f"No .pth for seed={seed}")
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


def orig_to_local_feature_map(best_node_mask_np):
    return {int(o): int(i) for i, o in enumerate(np.where(best_node_mask_np)[0])}


def get_available_union_nodes(roi0):
    roi_to_local = {int(roi0[i]): i for i in range(len(roi0))}
    return [(ui, roi_to_local[int(g_roi)])
            for ui, g_roi in enumerate(CONSENSUS_UNION) if int(g_roi) in roi_to_local]


def build_union_mask_from_available(n_nodes, available):
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    for _, loc in available:
        mask[loc] = True
    return mask


def build_node_mask(n_nodes, loc_node):
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[loc_node] = True
    return mask


@torch.no_grad()
def predict_proba(model, g):
    return float(torch.sigmoid(model(g.x, g.edge_index, g.graph_feat, g.batch)).item())


# ============================================================
# STAGE 1: IG per node x feature
# ============================================================
def stage1_ig_union(results_dir, out_dir, ig_steps=50, tp_tn_only=True):
    ig_seed_mean = {cls: {seed: np.full((N_UNION, N_FEATS), np.nan, float) for seed in SEEDS}
                    for cls in CLASSES}
    per_seed_rows, subject_level_rows = [], []

    for seed in SEEDS:
        loaded = torch.load(find_seed_file(results_dir, seed), map_location="cpu", weights_only=False)
        df_outer = loaded["df_outer"]
        models_state_dicts = loaded["models"]
        Test_Data_outer = loaded["Test_Data_outer"]

        acc = {cls: np.zeros((N_UNION, N_FEATS), float) for cls in CLASSES}
        cnt = {cls: np.zeros((N_UNION, N_FEATS), int) for cls in CLASSES}

        for fold_idx in range(len(df_outer)):
            params = df_outer["params"].iloc[fold_idx]
            best_node_mask_np = np.array(df_outer["best_node_mask"].iloc[fold_idx], dtype=bool)
            best_graph_feat_mask_np = np.array(df_outer["best_graph_feat_mask"].iloc[fold_idx], dtype=bool)
            best_node_mask = torch.tensor(best_node_mask_np, dtype=torch.bool)
            best_graph_feat_mask = torch.tensor(best_graph_feat_mask_np, dtype=torch.bool)

            o2l = orig_to_local_feature_map(best_node_mask_np)
            if any(f not in o2l for f in range(N_FEATS)):
                print(f"  [SKIP] seed={seed} fold={fold_idx}: not all features in mask")
                continue

            model = GAT(in_channels=int(best_node_mask.sum().item()),
                        hidden_channels=params["hidden_dim"], dropout=params["dropout"],
                        graph_feat_dim=int(best_graph_feat_mask.sum().item()),
                        num_layers=params["num_layers"], heads=params.get("heads", 1),
                        use_layernorm=False, concat=True, pool="attn").to(DEVICE)
            model.load_state_dict(models_state_dicts[fold_idx])
            model.eval()

            graphs_fold = Test_Data_outer[fold_idx]
            y_true_np = np.asarray(df_outer["y_true"].iloc[fold_idx]).astype(int)
            y_hat_np = (np.asarray(df_outer["y_pred"].iloc[fold_idx]).astype(float) >= THR).astype(int)
            idx_keep = np.where(y_true_np == y_hat_np)[0] if tp_tn_only else np.arange(len(graphs_fold))

            class_items = {cls: [] for cls in CLASSES}
            for i in idx_keep:
                g_raw = graphs_fold[i]
                try:
                    roi0 = roi_ids_per_node_from_graph(g_raw, FULL_ATLAS_SIZE)
                    g_p = prepare_graph(g_raw, best_node_mask, best_graph_feat_mask)
                    available = get_available_union_nodes(roi0)
                    if len(available) == 0:
                        continue
                    union_mask = build_union_mask_from_available(g_p.num_nodes, available)
                    cls = "sz" if int(g_raw.y.item()) == 1 else "hc"
                    class_items[cls].append((g_p, available, union_mask))
                except Exception as e:
                    print(f"  [ERR collect] {type(e).__name__}: {e}")

            # Class-specific baseline = mean union-node features within class.
            node_means = {cls: {} for cls in CLASSES}
            for cls, items in class_items.items():
                if len(items) < 2:
                    continue
                for ui in range(N_UNION):
                    feats = [g_p.x[next((loc for u, loc in available if u == ui))].float()
                             for g_p, available, _ in items
                             if any(u == ui for u, _ in available)]
                    if feats:
                        node_means[cls][ui] = torch.stack(feats).mean(0)

            def build_subject_baseline(g_p, available, cls):
                base = torch.zeros_like(g_p.x.float())
                for ui, loc_node in available:
                    if ui in node_means[cls]:
                        base[loc_node] = node_means[cls][ui]
                return base

            for cls, cls_label in CLASSES.items():
                items = class_items[cls]
                if len(items) < 2 or not node_means[cls]:
                    continue
                for g_p, available, union_mask in items:
                    try:
                        base = build_subject_baseline(g_p, available, cls)
                        ig = IntegratedGradients(GATWrapperUnionMask(
                            model, g_p.edge_index, g_p.graph_feat, g_p.batch, union_mask))
                        attr = ig.attribute(g_p.x.float().unsqueeze(0), baselines=base.unsqueeze(0),
                                            n_steps=ig_steps, internal_batch_size=1).squeeze(0)
                        if cls_label == 0:
                            attr = -attr
                        for ui, loc_node in available:
                            for orig_f in range(N_FEATS):
                                acc[cls][ui, orig_f] += float(torch.abs(attr[loc_node, o2l[orig_f]]).item())
                                cnt[cls][ui, orig_f] += 1
                    except Exception as e:
                        print(f"  [ERR ig] {type(e).__name__}: {e}")
                        continue

                    # Subject-level top-feature dominance.
                    for ui, loc_node in available:
                        feat_vals = np.array(
                            [float(torch.abs(attr[loc_node, o2l[f]]).item()) for f in range(N_FEATS)], float)
                        if np.all(np.isnan(feat_vals)):
                            continue
                        top_idx = np.argsort(feat_vals)[::-1]
                        top1_f = int(top_idx[0])
                        top2_f = int(top_idx[1]) if len(top_idx) > 1 else top1_f
                        null1 = float(np.nanmean([feat_vals[f] for f in range(N_FEATS) if f != top1_f]))
                        null2 = float(np.nanmean([feat_vals[f] for f in range(N_FEATS) if f != top2_f]))
                        subject_level_rows.append({
                            "seed": seed, "fold": fold_idx, "class": cls, "union_idx": ui,
                            "global_roi": CONSENSUS_UNION[ui],
                            "top1_feat": top1_f, "top1_name": FEATURE_NAMES[top1_f],
                            "top1_abs_ig": float(feat_vals[top1_f]), "top1_null_mean_abs_ig": null1,
                            "top1_D": float(feat_vals[top1_f] - null1),
                            "top2_feat": top2_f, "top2_name": FEATURE_NAMES[top2_f],
                            "top2_abs_ig": float(feat_vals[top2_f]), "top2_null_mean_abs_ig": null2,
                            "top2_D": float(feat_vals[top2_f] - null2),
                        })

        for cls in CLASSES:
            mean = np.full((N_UNION, N_FEATS), np.nan, float)
            nz = cnt[cls] > 0
            mean[nz] = acc[cls][nz] / cnt[cls][nz]
            ig_seed_mean[cls][seed] = mean
            for ui in range(N_UNION):
                for f in range(N_FEATS):
                    per_seed_rows.append({
                        "seed": seed, "class": cls, "union_idx": ui,
                        "global_roi": CONSENSUS_UNION[ui], "feature": f, "feat_name": FEATURE_NAMES[f],
                        "mean_abs_ig": float(mean[ui, f]), "n_contrib": int(cnt[cls][ui, f]),
                    })
        print(f"[Stage1] Seed {seed} done.")

    # Leave-one-seed-out top-1/top-2 selection.
    top_rows = []
    for cls in CLASSES:
        for ui in range(N_UNION):
            feat_by_seed = {s: ig_seed_mean[cls][s][ui, :].astype(float) for s in SEEDS}
            for seed in SEEDS:
                other = [s for s in SEEDS if s != seed]
                mean_other = feat_by_seed[seed] if not other else np.nanmean(
                    np.stack([feat_by_seed[s] for s in other]), axis=0)
                if np.all(np.isnan(mean_other)):
                    top1_f = top2_f = 0
                    null1 = null2 = D1 = D2 = np.nan
                else:
                    sorted_idx = np.argsort(mean_other)[::-1]
                    top1_f = int(sorted_idx[0])
                    top2_f = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_f
                    v = feat_by_seed[seed]
                    null1 = float(np.nanmean([v[f] for f in range(N_FEATS) if f != top1_f]))
                    null2 = float(np.nanmean([v[f] for f in range(N_FEATS) if f != top2_f]))
                    D1, D2 = float(v[top1_f] - null1), float(v[top2_f] - null2)
                top_rows.append({
                    "seed": seed, "class": cls, "union_idx": ui, "global_roi": CONSENSUS_UNION[ui],
                    "top1_feat": top1_f, "top1_name": FEATURE_NAMES[top1_f],
                    "top1_abs_ig": float(feat_by_seed[seed][top1_f]), "top1_null_mean_abs_ig": null1, "top1_D": D1,
                    "top2_feat": top2_f, "top2_name": FEATURE_NAMES[top2_f],
                    "top2_abs_ig": float(feat_by_seed[seed][top2_f]), "top2_null_mean_abs_ig": null2, "top2_D": D2,
                })

    df_top_loso = pd.DataFrame(top_rows)
    df_subject_top = pd.DataFrame(subject_level_rows)

    sign_rows = []
    for cls in CLASSES:
        for ui in range(N_UNION):
            sub = df_top_loso[(df_top_loso["class"] == cls) & (df_top_loso["union_idx"] == ui)].sort_values("seed")
            D1, D2 = sub["top1_D"].values.astype(float), sub["top2_D"].values.astype(float)
            mean_D1, p1 = signflip_test(D1)
            mean_D2, p2 = signflip_test(D2)
            t1c, t2c = Counter(sub["top1_feat"].tolist()), Counter(sub["top2_feat"].tolist())
            sign_rows.append({
                "class": cls, "union_idx": ui, "global_roi": CONSENSUS_UNION[ui],
                "modal_top1_feat": t1c.most_common(1)[0][0], "modal_top1_name": FEATURE_NAMES[t1c.most_common(1)[0][0]],
                "mean_D_top1": float(mean_D1), "p_raw_top1": float(p1),
                "cohens_dz_top1": cohens_dz(D1), "cliffs_delta_top1": cliffs_delta(D1),
                "modal_top2_feat": t2c.most_common(1)[0][0], "modal_top2_name": FEATURE_NAMES[t2c.most_common(1)[0][0]],
                "mean_D_top2": float(mean_D2), "p_raw_top2": float(p2),
                "cohens_dz_top2": cohens_dz(D2), "cliffs_delta_top2": cliffs_delta(D2),
                "n_seeds_finite": int(np.sum(np.isfinite(D1))),
            })

    df_sign = pd.DataFrame(sign_rows)
    df_sign["q_BH_FDR_top1"] = np.nan
    df_sign["q_BH_FDR_top2"] = np.nan
    for cls in CLASSES:
        mask = df_sign["class"] == cls
        df_sign.loc[mask, "q_BH_FDR_top1"] = bh_fdr(df_sign.loc[mask, "p_raw_top1"].values.astype(float))
        df_sign.loc[mask, "q_BH_FDR_top2"] = bh_fdr(df_sign.loc[mask, "p_raw_top2"].values.astype(float))
    df_sign["significant_FDR_0p05_top1"] = df_sign["q_BH_FDR_top1"] < 0.05
    df_sign["significant_FDR_0p05_top2"] = df_sign["q_BH_FDR_top2"] < 0.05

    # Modal top1/top2 per (class, node) -> consensus map for Stage 2.
    consensus_map = {}
    for cls in CLASSES:
        for ui in range(N_UNION):
            sub = df_top_loso[(df_top_loso["class"] == cls) & (df_top_loso["union_idx"] == ui)]
            modal_top1 = Counter(sub["top1_feat"].tolist()).most_common(1)[0][0]
            modal_top2 = Counter(sub["top2_feat"].tolist()).most_common(1)[0][0]
            consensus_map[(cls, ui)] = (modal_top1, modal_top2)

    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, "stage1_ig_union_top2.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame(per_seed_rows).to_excel(w, sheet_name="Per_Seed_NodeFeat", index=False)
        df_top_loso.to_excel(w, sheet_name="Top2_LOSO_perSeed", index=False)
        df_sign.to_excel(w, sheet_name="Top2_Signflip_FDR", index=False)
        df_subject_top.to_excel(w, sheet_name="Top2_SubjectLevel", index=False)
        heat_rows = []
        for cls in CLASSES:
            M = np.nanmean(np.stack([ig_seed_mean[cls][s] for s in SEEDS]), axis=0)
            for ui in range(N_UNION):
                row = {"class": cls, "union_idx": ui, "global_roi": CONSENSUS_UNION[ui]}
                for f in range(N_FEATS):
                    row[FEATURE_NAMES[f]] = float(M[ui, f])
                heat_rows.append(row)
        pd.DataFrame(heat_rows).to_excel(w, sheet_name="Heatmap_meanAcrossSeeds", index=False)

    df_top_loso.to_csv(os.path.join(out_dir, "top2_loso_per_seed.csv"), index=False)
    print(f"[Stage1] Saved: {out_xlsx}")
    return df_top_loso, df_sign, consensus_map, out_xlsx


# ============================================================
# STAGE 2: necessity / sufficiency
# ============================================================
def stage2_nec_suf_union(results_dir, out_dir, consensus_map, ablation_mode="zero",
                         n_null=200, rng_seed=42, scope="per_node", external=True, tp_tn_only=False):
    """Necessity / sufficiency of Stage-1 top features (subject-level if external)."""
    D_subject_store = {(tt, cls, ui, rank, "true_class_prob_mean"): []
                       for tt in ("necessity", "sufficiency") for cls in CLASSES
                       for ui in range(N_UNION) for rank in ("top1", "top2")}
    D_seed_store = {(tt, cls, ui, rank, m): []
                    for tt in ("necessity", "sufficiency") for cls in CLASSES
                    for ui in range(N_UNION) for rank in ("top1", "top2")
                    for m in METRICS + STAGE2_PROB_METRICS}
    fold_rows = []

    for seed in SEEDS:
        rng = np.random.default_rng(rng_seed + int(seed))
        loaded = torch.load(find_seed_file(results_dir, seed), map_location="cpu", weights_only=False)
        df_outer = loaded["df_outer"]
        models_state_dicts = loaded["models"]
        Test_Data_outer = loaded["Test_Data_outer"]

        null_pools = {}
        for cls in CLASSES:
            for ui in range(N_UNION):
                top1_f, top2_f = consensus_map[(cls, ui)]
                null_pools[(cls, ui)] = {
                    "top1": (top1_f, rng.choice([f for f in range(N_FEATS) if f != top1_f], size=n_null, replace=True).tolist()),
                    "top2": (top2_f, rng.choice([f for f in range(N_FEATS) if f != top2_f], size=n_null, replace=True).tolist()),
                }

        per_fold_D = {(tt, cls, ui, rank, m): []
                      for tt in ("necessity", "sufficiency") for cls in CLASSES
                      for ui in range(N_UNION) for rank in ("top1", "top2")
                      for m in METRICS + STAGE2_PROB_METRICS}

        for fold_idx in range(len(df_outer)):
            params = df_outer["params"].iloc[fold_idx]
            best_node_mask_np = np.array(df_outer["best_node_mask"].iloc[fold_idx], dtype=bool)
            best_graph_feat_mask_np = np.array(df_outer["best_graph_feat_mask"].iloc[fold_idx], dtype=bool)
            best_node_mask = torch.tensor(best_node_mask_np, dtype=torch.bool)
            best_graph_feat_mask = torch.tensor(best_graph_feat_mask_np, dtype=torch.bool)
            o2l = orig_to_local_feature_map(best_node_mask_np)
            if any(f not in o2l for f in range(N_FEATS)):
                continue

            model = GAT(in_channels=int(best_node_mask.sum().item()),
                        hidden_channels=params["hidden_dim"], dropout=params["dropout"],
                        graph_feat_dim=int(best_graph_feat_mask.sum().item()),
                        num_layers=params["num_layers"], heads=params.get("heads", 1),
                        use_layernorm=False, concat=True, pool="attn").to(DEVICE)
            model.load_state_dict(models_state_dicts[fold_idx])
            model.eval()

            graphs_fold = Test_Data_outer[fold_idx]
            y_true_np = np.asarray(df_outer["y_true"].iloc[fold_idx]).astype(int)
            y_hat_np = (np.asarray(df_outer["y_pred"].iloc[fold_idx]).astype(float) >= THR).astype(int)
            idx_keep = np.where(y_true_np == y_hat_np)[0] if tp_tn_only else np.arange(len(graphs_fold))

            prepared = []
            for i in idx_keep:
                g_raw = graphs_fold[i]
                try:
                    roi0 = roi_ids_per_node_from_graph(g_raw, FULL_ATLAS_SIZE)
                    g_p = prepare_graph(g_raw, best_node_mask, best_graph_feat_mask)
                    available = get_available_union_nodes(roi0)
                    if len(available) == 0:
                        continue
                    union_mask = build_union_mask_from_available(g_p.num_nodes, available)
                    g_base = g_p.clone()
                    g_base.x = g_p.x.clone()
                    g_base.x[~union_mask] = 0.0
                    prepared.append((g_base, available, union_mask, g_p.num_nodes, int(g_raw.y.item())))
                except Exception as e:
                    print(f"  [ERR prep] {type(e).__name__}: {e}")

            if len(prepared) < 2:
                continue

            for cls in CLASSES:
                cls_label = CLASSES[cls]
                for ui in range(N_UNION):
                    for rank in ("top1", "top2"):
                        feat, rand_feats = null_pools[(cls, ui)][rank]
                        loc_feat = o2l[feat]
                        other_locs = [o2l[f] for f in range(N_FEATS) if f != feat]

                        y_true, p_base = [], []
                        p_nec_top, p_nec_null, p_suf_top, p_suf_null = [], [], [], []
                        for g_base, available, union_mask, n_nodes, y_label in prepared:
                            ui_loc = next((loc for u, loc in available if u == ui), None)
                            if ui_loc is None:
                                continue
                            mask = build_node_mask(n_nodes, ui_loc) if scope == "per_node" else union_mask

                            with torch.no_grad():
                                pb = predict_proba(model, g_base)

                                g_nec = g_base.clone()
                                g_nec.x = g_base.x.clone()
                                if ablation_mode == "zero":
                                    g_nec.x[mask, loc_feat] = 0.0
                                else:
                                    v = g_nec.x[mask, loc_feat].cpu().numpy().copy()
                                    rng.shuffle(v)
                                    g_nec.x[mask, loc_feat] = torch.tensor(v, dtype=g_nec.x.dtype)
                                pnec_top = predict_proba(model, g_nec)

                                pnec_null_list = []
                                for rf in rand_feats:
                                    g_r = g_base.clone()
                                    g_r.x = g_base.x.clone()
                                    if ablation_mode == "zero":
                                        g_r.x[mask, o2l[rf]] = 0.0
                                    else:
                                        v = g_r.x[mask, o2l[rf]].cpu().numpy().copy()
                                        rng.shuffle(v)
                                        g_r.x[mask, o2l[rf]] = torch.tensor(v, dtype=g_r.x.dtype)
                                    pnec_null_list.append(predict_proba(model, g_r))
                                pnec_null = float(np.mean(pnec_null_list))

                                g_suf = g_base.clone()
                                g_suf.x = g_base.x.clone()
                                for loc_f in other_locs:
                                    g_suf.x[mask, loc_f] = 0.0
                                psuf_top = predict_proba(model, g_suf)

                                psuf_null_list = []
                                for rf in rand_feats:
                                    g_r = g_base.clone()
                                    g_r.x = g_base.x.clone()
                                    for loc_f in [o2l[f] for f in range(N_FEATS) if f != rf]:
                                        g_r.x[mask, loc_f] = 0.0
                                    psuf_null_list.append(predict_proba(model, g_r))
                                psuf_null = float(np.mean(psuf_null_list))

                            y_true.append(y_label)
                            p_base.append(pb)
                            p_nec_top.append(pnec_top)
                            p_nec_null.append(pnec_null)
                            p_suf_top.append(psuf_top)
                            p_suf_null.append(psuf_null)

                            if external and y_label == cls_label:
                                pt = (lambda p: p if y_label == 1 else 1.0 - p)
                                D_subject_store[("necessity", cls, ui, rank, "true_class_prob_mean")].append(
                                    pt(pnec_null) - pt(pnec_top))
                                D_subject_store[("sufficiency", cls, ui, rank, "true_class_prob_mean")].append(
                                    pt(psuf_top) - pt(psuf_null))

                        if len(y_true) < 2:
                            continue

                        m_base = compute_metrics(y_true, p_base)
                        m_nec_top = compute_metrics(y_true, p_nec_top)
                        m_nec_null = compute_metrics(y_true, p_nec_null)
                        m_suf_top = compute_metrics(y_true, p_suf_top)
                        m_suf_null = compute_metrics(y_true, p_suf_null)
                        for m in METRICS:
                            per_fold_D[("necessity", cls, ui, rank, m)].append(
                                (m_base[m] - m_nec_top[m]) - (m_base[m] - m_nec_null[m]))
                            per_fold_D[("sufficiency", cls, ui, rank, m)].append(m_suf_top[m] - m_suf_null[m])

                        y_true_arr = np.asarray(y_true, int)
                        idx_cls = np.where(y_true_arr == cls_label)[0]
                        pm = "true_class_prob_mean"
                        if idx_cls.size > 0:
                            def tcp(p):
                                return true_class_prob_metric(y_true_arr[idx_cls], np.asarray(p, float)[idx_cls])[pm]
                            D_nec_p = tcp(p_nec_null) - tcp(p_nec_top)
                            D_suf_p = tcp(p_suf_top) - tcp(p_suf_null)
                        else:
                            D_nec_p = D_suf_p = np.nan
                        per_fold_D[("necessity", cls, ui, rank, pm)].append(D_nec_p)
                        per_fold_D[("sufficiency", cls, ui, rank, pm)].append(D_suf_p)

        for tt in ("necessity", "sufficiency"):
            for cls in CLASSES:
                for ui in range(N_UNION):
                    for rank in ("top1", "top2"):
                        for m in METRICS + STAGE2_PROB_METRICS:
                            vals = per_fold_D[(tt, cls, ui, rank, m)]
                            D_seed_store[(tt, cls, ui, rank, m)].append(
                                float(np.nanmean(vals)) if vals else np.nan)
        print(f"[Stage2] Seed {seed} done.")

    global_rows = []
    for tt in ("necessity", "sufficiency"):
        for cls in CLASSES:
            for ui in range(N_UNION):
                top1_f, top2_f = consensus_map[(cls, ui)]
                for rank in ("top1", "top2"):
                    feat = top1_f if rank == "top1" else top2_f
                    for m in METRICS + STAGE2_PROB_METRICS:
                        if external and m == "true_class_prob_mean":
                            D = np.asarray(D_subject_store[(tt, cls, ui, rank, m)], float)
                            level = "subject"
                        else:
                            D = np.asarray(D_seed_store[(tt, cls, ui, rank, m)], float)
                            level = "seed"
                        mean_D, p = signflip_test(D)
                        sd_D = np.nanstd(D, ddof=1)
                        global_rows.append({
                            "test_type": tt, "class": cls, "metric": m, "scope": scope, "rank": rank,
                            "union_idx": ui, "global_roi": CONSENSUS_UNION[ui],
                            "feat": feat, "feat_name": FEATURE_NAMES[feat],
                            "mean_D": float(mean_D),
                            "cohens_dz": float(np.nanmean(D) / sd_D) if sd_D > 0 else np.nan,
                            "p_raw": float(p), "n_samples": int(np.sum(np.isfinite(D))),
                            "inference_level": level,
                        })

    df_global = pd.DataFrame(global_rows)
    df_global["q_BH_FDR"] = np.nan
    for tt in ("necessity", "sufficiency"):
        for cls in CLASSES:
            for m in METRICS + STAGE2_PROB_METRICS:
                for rank in ("top1", "top2"):
                    mask = ((df_global["test_type"] == tt) & (df_global["class"] == cls)
                            & (df_global["metric"] == m) & (df_global["rank"] == rank))
                    df_global.loc[mask, "q_BH_FDR"] = bh_fdr(df_global.loc[mask, "p_raw"].values.astype(float))
    df_global["significant_FDR_0p05"] = df_global["q_BH_FDR"] < 0.05
    df_global = df_global.sort_values(
        ["test_type", "class", "rank", "metric", "q_BH_FDR"]).reset_index(drop=True)

    out_xlsx = os.path.join(out_dir, f"stage2_nec_suf_union_top2__scope_{scope}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_global.to_excel(w, sheet_name="Global_FDR", index=False)
    print(f"[Stage2] Saved: {out_xlsx}")
    return df_global, out_xlsx


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--ig_steps", type=int, default=50)
    ap.add_argument("--external", action="store_true", help="Subject-level inference for Stage 2")
    ap.add_argument("--tp_tn_only", dest="tp_tn_only", action="store_true", help="Use only TP+TN subjects")
    ap.add_argument("--all_subjects", dest="tp_tn_only", action="store_false", help="Use all test subjects")
    ap.set_defaults(tp_tn_only=False)
    ap.add_argument("--ablation_mode", choices=["zero", "shuffle"], default="zero")
    ap.add_argument("--n_null", type=int, default=20)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--scope", choices=["per_node", "union"], default="per_node")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df_top_loso, df_sign, consensus_map, stage1_xlsx = stage1_ig_union(
        results_dir=args.results_dir, out_dir=args.out_dir,
        ig_steps=args.ig_steps, tp_tn_only=args.tp_tn_only)
    df_stage2_global, stage2_xlsx = stage2_nec_suf_union(
        results_dir=args.results_dir, out_dir=args.out_dir, consensus_map=consensus_map,
        ablation_mode=args.ablation_mode, n_null=args.n_null, rng_seed=args.rng_seed,
        scope=args.scope, external=args.external, tp_tn_only=args.tp_tn_only)
    print("\nDONE\nStage 1:", stage1_xlsx, "\nStage 2:", stage2_xlsx)


if __name__ == "__main__":
    main()
