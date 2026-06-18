"""Structure-aware statistical inference on SubgraphX explanations.

Aggregates the per-subject SubgraphX node sets (from subgraphx_gat.py) into
seed-level node and edge frequencies and tests, across seeds:

Node analyses:
  1. Per class (schizo / control): frequency > random baseline (N_MIN / N).
  2. Class difference (SZ - HC): which nodes differ between classes.
  3. Overall (all subjects pooled): frequency > baseline (for external validation).

Edge analyses:
  Stability (Top-K rank consistency across seeds), per class and class-diff, plus
  consensus graphs on the union of significant nodes.

Significance: sign-flip permutation tests with BH-FDR; bootstrap CIs for means.
Self-contained; paths come from ``config``.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# ============================================================
# CONFIG
# ============================================================
SUBGRAPHX_OUT = os.path.join(str(config.XAI_RESULTS_DIR), "subgraphx_outputs")
SUBGRAPH_PTH = os.path.join(SUBGRAPHX_OUT, "subgraphx_subject_subgraphs.pth")
RESULTS_DIR = str(config.GATE_RESULTS_DIR)
SEEDS = config.SEEDS
OUT_DIR = os.path.join(SUBGRAPHX_OUT, "subgraphx_xAI")
os.makedirs(OUT_DIR, exist_ok=True)

FULL_ATLAS_SIZE = config.FULL_ATLAS_SIZE
N_MIN = 10
TOP_K = 10
MIN_SEEDS = 6
MIN_SUBJECTS = 0

N_BOOT = 20000
CI = 0.95
N_SIGNFLIP = 20000
RNG_STATS = 123


# ============================================================
# IO HELPERS
# ============================================================
def find_seed_file(results_dir, seed):
    files = sorted(glob.glob(os.path.join(results_dir, f"all_results_*_seed{seed}.pth")))
    if not files:
        raise FileNotFoundError(f"No .pth for seed={seed} in {results_dir}")
    return files[0]


def load_seed_results(seed):
    return torch.load(find_seed_file(RESULTS_DIR, seed), map_location="cpu")


def get_df_outer(results):
    df = results.get("df_outer", None)
    if df is None:
        raise KeyError("results has no df_outer")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def get_test_graphs_for_fold(results, df_row, fold_idx):
    if isinstance(df_row, (pd.Series, dict)) and "Test_Data_outer" in df_row:
        g = df_row["Test_Data_outer"]
        if g is not None and hasattr(g, "__len__") and len(g) > 0:
            return g
    if "Test_Data_outer" in results and results["Test_Data_outer"] is not None:
        return results["Test_Data_outer"][fold_idx]
    raise KeyError("Cannot find Test_Data_outer.")


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
    roi0 = roi_raw - 1 if (roi_raw.min() >= 1 and roi_raw.max() <= full_size) else roi_raw
    if int(np.min(roi0)) < 0 or int(np.max(roi0)) >= full_size:
        raise ValueError(f"ROI ids out of range: min={roi0.min()}, max={roi0.max()}")
    return roi0


def subject_original_roi_edges(g, full_size):
    edge_index = g.edge_index
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index)
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    roi0 = roi_ids_per_node_from_graph(g, full_size)
    u = edge_index[0].cpu().numpy().astype(int)
    v = edge_index[1].cpu().numpy().astype(int)
    edges = set()
    for a, b in zip(u, v):
        ra, rb = int(roi0[a]), int(roi0[b])
        if ra == rb or ra < 0 or rb < 0 or ra >= full_size or rb >= full_size:
            continue
        edges.add((ra, rb) if ra < rb else (rb, ra))
    return edges


# ============================================================
# STATS HELPERS
# ============================================================
def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    q = np.full_like(pvals, np.nan, float)
    idx = np.where(np.isfinite(pvals))[0]
    if len(idx) == 0:
        return q
    pv = pvals[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = np.empty(m, float)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        prev = min(prev, (m / (k + 1)) * ranked[k])
        q_ranked[k] = prev
    tmp = np.empty_like(pv)
    tmp[order] = q_ranked
    q[idx] = tmp
    return q


def bootstrap_ci_mean(x, n_boot=N_BOOT, ci=CI, rng_seed=RNG_STATS):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(rng_seed)
    boot = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    alpha = 1.0 - ci
    return float(x.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def signflip_pvalue(x, n_perm=N_SIGNFLIP, rng_seed=RNG_STATS, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (float(np.mean(x)) if n > 0 else np.nan), np.nan
    obs = float(np.mean(x))
    rng = np.random.default_rng(rng_seed)
    perm = (rng.choice([-1.0, 1.0], size=(n_perm, n)) * x).mean(axis=1)
    if alternative == "greater":
        p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(perm <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p)


def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


# ============================================================
# EDGE INDEXING
# ============================================================
TRIU = np.triu_indices(FULL_ATLAS_SIZE, k=1)
N_EDGES = len(TRIU[0])
_EDGE_TO_IDX = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(TRIU[0], TRIU[1]))}


def edge_to_idx(u, v):
    if u == v:
        return None
    return _EDGE_TO_IDX.get((u, v) if u < v else (v, u), None)


# ============================================================
# NODE / EDGE STATS
# ============================================================
def compute_perclass_node_stats(seed_node_freq, seeds, grp, full_size, baseline,
                                min_seeds=6, min_subjects=3, seed_n_subj=None):
    p = np.full(full_size, np.nan, float)
    meanv = np.full(full_size, np.nan, float)
    lo = np.full(full_size, np.nan, float)
    hi = np.full(full_size, np.nan, float)
    std = np.full(full_size, np.nan, float)
    nused = np.zeros(full_size, int)

    for i in range(full_size):
        vals = []
        for s in seeds:
            freq = seed_node_freq[s][grp]
            if freq is None:
                continue
            if seed_n_subj is not None and freq[i] * seed_n_subj[s][grp] < min_subjects:
                continue
            vals.append(float(freq[i]))
        vals = np.asarray(vals, float)
        if len(vals) < min_seeds:
            continue
        nused[i] = len(vals)
        meanv[i], lo[i], hi[i] = bootstrap_ci_mean(vals)
        std[i] = float(np.std(vals, ddof=1))
        _, p[i] = signflip_pvalue(vals - baseline, alternative="greater")

    return p, bh_fdr(p), meanv, std, lo, hi, nused


def compute_delta_node_stats(seed_node_freq, seeds, full_size,
                             min_seeds=6, min_subjects=3, seed_n_subj=None):
    p2 = np.full(full_size, np.nan, float)
    pgt = np.full(full_size, np.nan, float)
    plt_ = np.full(full_size, np.nan, float)
    md = np.full(full_size, np.nan, float)
    std = np.full(full_size, np.nan, float)
    lo = np.full(full_size, np.nan, float)
    hi = np.full(full_size, np.nan, float)
    nused = np.zeros(full_size, int)

    for i in range(full_size):
        d = []
        for s in seeds:
            A = seed_node_freq[s]["schizo"]
            B = seed_node_freq[s]["control"]
            if A is None or B is None:
                continue
            if seed_n_subj is not None:
                if A[i] * seed_n_subj[s]["schizo"] < min_subjects:
                    continue
                if B[i] * seed_n_subj[s]["control"] < min_subjects:
                    continue
            dv = float(A[i]) - float(B[i])
            if np.isfinite(dv):
                d.append(dv)
        d = np.asarray(d, float)
        if len(d) < min_seeds:
            continue
        nused[i] = len(d)
        md[i], lo[i], hi[i] = bootstrap_ci_mean(d)
        std[i] = float(np.std(d, ddof=1))
        _, p2[i] = signflip_pvalue(d, alternative="two-sided")
        _, pgt[i] = signflip_pvalue(d, alternative="greater")
        _, plt_[i] = signflip_pvalue(d, alternative="less")

    return p2, bh_fdr(p2), pgt, bh_fdr(pgt), plt_, bh_fdr(plt_), md, std, lo, hi, nused


def compute_stability_nodes(seed_node_freq, seeds, grp, full_size, top_k=10, min_seeds=6):
    active = [s for s in seeds if seed_node_freq[s][grp] is not None]
    n = len(active)
    roi_to_ranks = defaultdict(list)
    roi_to_topk = defaultdict(int)

    for s in active:
        freq = seed_node_freq[s][grp]
        order = np.argsort(freq)[::-1]
        for rank, node_i in enumerate(order):
            if freq[node_i] <= 0:
                break
            roi_to_ranks[node_i].append(rank + 1)
            if rank < top_k:
                roi_to_topk[node_i] += 1

    rows = []
    for i in range(full_size):
        ranks = np.asarray(roi_to_ranks[i], float)
        if len(ranks) < min_seeds:
            continue
        rows.append({
            "group": grp, "node_global_id": i, "roi_label": f"ROI_{i+1}",
            "n_seeds_present": int(len(ranks)),
            "freq_topK": float(roi_to_topk[i] / n) if n > 0 else np.nan,
            "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else np.nan,
            "top_k": int(top_k),
        })

    seed_topk_sets = {s: set(np.argsort(seed_node_freq[s][grp])[::-1][:top_k].tolist()) for s in active}
    jac_vals = [jaccard(seed_topk_sets[s1], seed_topk_sets[s2])
                for i, s1 in enumerate(active) for j, s2 in enumerate(active) if j > i]
    return pd.DataFrame(rows), float(np.mean(jac_vals)) if jac_vals else np.nan


def compute_edge_stability(seed_edge_delta, seeds, grp, top_k=100, min_seeds=6):
    active = [s for s in seeds if seed_edge_delta[s][grp] is not None]
    n = len(active)
    if n == 0:
        return pd.DataFrame([]), np.nan
    edge_to_ranks = defaultdict(list)
    edge_to_topk = defaultdict(int)
    seed_topk_sets = {}

    for s in active:
        d = np.asarray(seed_edge_delta[s][grp], float)
        idxs = np.where(np.isfinite(d) & (d > 0))[0]
        if idxs.size == 0:
            seed_topk_sets[s] = set()
            continue
        order = idxs[np.argsort(d[idxs])[::-1]]
        seed_topk_sets[s] = set(int(e) for e in order[:top_k])
        for r, e in enumerate(order, start=1):
            edge_to_ranks[int(e)].append(r)
            if r <= top_k:
                edge_to_topk[int(e)] += 1

    rows = []
    for e in range(N_EDGES):
        ranks = np.asarray(edge_to_ranks.get(e, []), float)
        if ranks.size < min_seeds:
            continue
        u, v = int(TRIU[0][e]), int(TRIU[1][e])
        rows.append({
            "group": grp, "node_u": u, "node_v": v,
            "roi_u": f"ROI_{u+1}", "roi_v": f"ROI_{v+1}", "edge_idx": int(e),
            "n_seeds_present": int(ranks.size), "freq_topK": float(edge_to_topk.get(e, 0) / n),
            "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks, ddof=1)) if ranks.size > 1 else np.nan,
            "top_k": int(top_k),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["group", "freq_topK", "median_rank"],
                            ascending=[True, False, True]).reset_index(drop=True)
    jac_vals = [jaccard(seed_topk_sets.get(s1, set()), seed_topk_sets.get(s2, set()))
                for i, s1 in enumerate(active) for j, s2 in enumerate(active) if j > i]
    return df, float(np.mean(jac_vals)) if jac_vals else np.nan


def compute_edge_stability_classdiff(seed_edge_delta, seeds, top_k=100, min_seeds=6, mode="abs"):
    active = [s for s in seeds
              if seed_edge_delta[s].get("schizo") is not None
              and seed_edge_delta[s].get("control") is not None]
    n = len(active)
    if n == 0:
        return pd.DataFrame([]), np.nan
    edge_to_ranks = defaultdict(list)
    edge_to_topk = defaultdict(int)
    edge_to_D = defaultdict(list)
    seed_topk_sets = {}

    for s in active:
        d_sz = np.asarray(seed_edge_delta[s]["schizo"], float)
        d_hc = np.asarray(seed_edge_delta[s]["control"], float)
        D = d_sz - d_hc
        idxs = np.where(np.isfinite(D) & np.isfinite(d_sz) & np.isfinite(d_hc)
                        & ((d_sz > 0) | (d_hc > 0)))[0]
        if idxs.size == 0:
            seed_topk_sets[s] = set()
            continue
        score = np.abs(D[idxs]) if mode == "abs" else D[idxs]
        order = idxs[np.argsort(score)[::-1]]
        seed_topk_sets[s] = set(int(e) for e in order[:top_k])
        for r, e in enumerate(order, start=1):
            e = int(e)
            edge_to_ranks[e].append(r)
            edge_to_D[e].append(float(D[e]))
            if r <= top_k:
                edge_to_topk[e] += 1

    rows = []
    for e in range(N_EDGES):
        ranks = np.asarray(edge_to_ranks.get(e, []), float)
        if ranks.size < min_seeds:
            continue
        u, v = int(TRIU[0][e]), int(TRIU[1][e])
        Dv = np.asarray(edge_to_D.get(e, []), float)
        rows.append({
            "edge_idx": int(e), "node_u": u, "node_v": v,
            "roi_u": f"ROI_{u+1}", "roi_v": f"ROI_{v+1}",
            "n_seeds_present": int(ranks.size), "freq_topK": float(edge_to_topk.get(e, 0) / n),
            "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks, ddof=1)) if ranks.size > 1 else np.nan,
            "mean_D_seed": float(np.mean(Dv)) if Dv.size else np.nan,
            "mean_absD_seed": float(np.mean(np.abs(Dv))) if Dv.size else np.nan,
            "top_k": int(top_k), "mode": str(mode),
        })

    df_cd = pd.DataFrame(rows)
    if len(df_cd) > 0:
        df_cd = df_cd.sort_values(["freq_topK", "median_rank"],
                                  ascending=[False, True]).reset_index(drop=True)
    jac_vals = [jaccard(seed_topk_sets.get(s1, set()), seed_topk_sets.get(s2, set()))
                for i, s1 in enumerate(active) for j, s2 in enumerate(active) if j > i]
    return df_cd, float(np.mean(jac_vals)) if jac_vals else np.nan


def build_group_consensus_graph(seed_subject_nodes, seed_subject_edges, seeds, group_name,
                                consensus_nodes, min_seeds=6, bin_thr=0.5):
    """Consensus edge prevalence on a fixed node set, averaged across seeds."""
    consensus_nodes = sorted(set(int(x) for x in consensus_nodes))
    if len(consensus_nodes) < 2:
        return pd.DataFrame([]), np.zeros((0, 0)), np.zeros((0, 0), dtype=int)

    node_to_idx = {n: i for i, n in enumerate(consensus_nodes)}
    k = len(consensus_nodes)
    seed_mats, seed_support = [], []

    for seed in seeds:
        subj_nodes = seed_subject_nodes[seed][group_name]
        subj_edges = seed_subject_edges[seed][group_name]
        if len(subj_nodes) == 0:
            continue
        num = np.zeros((k, k), float)
        den = np.zeros((k, k), float)
        for S, E in zip(subj_nodes, subj_edges):
            S, E = set(S), set(E)
            present = [n for n in consensus_nodes if n in S]
            if len(present) < 2:
                continue
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    u, v = present[i], present[j]
                    iu, iv = node_to_idx[u], node_to_idx[v]
                    den[iu, iv] += 1.0
                    den[iv, iu] += 1.0
                    e = (u, v) if u < v else (v, u)
                    if e in E:
                        num[iu, iv] += 1.0
                        num[iv, iu] += 1.0
        mat = np.full((k, k), np.nan, float)
        mask = den > 0
        mat[mask] = num[mask] / den[mask]
        np.fill_diagonal(mat, 0.0)
        seed_mats.append(mat)
        seed_support.append((den > 0).astype(int))

    if len(seed_mats) == 0:
        return pd.DataFrame([]), np.full((k, k), np.nan), np.zeros((k, k), dtype=int)

    seed_mats = np.stack(seed_mats, axis=0)
    seed_support = np.stack(seed_support, axis=0)
    mean_mat = np.nanmean(seed_mats, axis=0)
    support_mat = np.sum(seed_support, axis=0)
    valid_support = support_mat >= min_seeds
    mean_mat[~valid_support] = np.nan

    bin_mat = np.zeros((k, k), dtype=int)
    bin_mat[np.isfinite(mean_mat) & valid_support & (mean_mat >= bin_thr)] = 1
    np.fill_diagonal(bin_mat, 0)

    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            rows.append({
                "group": group_name, "node_u": consensus_nodes[i], "node_v": consensus_nodes[j],
                "roi_u": f"ROI_{consensus_nodes[i]+1}", "roi_v": f"ROI_{consensus_nodes[j]+1}",
                "edge_prevalence_mean": float(mean_mat[i, j]) if np.isfinite(mean_mat[i, j]) else np.nan,
                "n_seeds_supported": int(support_mat[i, j]),
                "passes_min_seeds": bool(support_mat[i, j] >= min_seeds),
                "binary_consensus": int(bin_mat[i, j]),
                "binary_threshold": float(bin_thr), "min_seeds_required": int(min_seeds),
            })
    df = pd.DataFrame(rows).sort_values(
        ["passes_min_seeds", "binary_consensus", "edge_prevalence_mean"],
        ascending=[False, False, False]).reset_index(drop=True)
    return df, mean_mat, bin_mat


def main():
    print("Loading SubgraphX results...")
    data = torch.load(SUBGRAPH_PTH, map_location="cpu")
    by_subject = data["by_subject"]
    print(f"  Total subject entries: {len(by_subject)}")

    # ---- Index original test graphs + per-seed/class conditional edge prob ----
    orig_subject_edges, orig_subject_roimap = {}, {}
    seed_orig_edge_num = {s: {g: np.zeros((FULL_ATLAS_SIZE, FULL_ATLAS_SIZE), float)
                              for g in ("schizo", "control", "overall")} for s in SEEDS}
    seed_orig_edge_denom = {s: {g: np.zeros((FULL_ATLAS_SIZE, FULL_ATLAS_SIZE), float)
                                for g in ("schizo", "control", "overall")} for s in SEEDS}

    for seed in SEEDS:
        res = load_seed_results(seed)
        df = get_df_outer(res)
        for fold_idx in range(len(df)):
            for g in get_test_graphs_for_fold(res, df.iloc[fold_idx], fold_idx):
                sid = str(getattr(g, "subject_id", None))
                if sid in (None, "None"):
                    continue
                try:
                    E_roi = subject_original_roi_edges(g, FULL_ATLAS_SIZE)
                    roi0 = roi_ids_per_node_from_graph(g, FULL_ATLAS_SIZE)
                    orig_subject_edges[(seed, fold_idx, sid)] = E_roi
                    orig_subject_roimap[(seed, fold_idx, sid)] = roi0
                    cls = "schizo" if int(g.y.item()) == 1 else "control"
                    pn = np.array(list(set(roi0.tolist())), dtype=int)
                    for grp in (cls, "overall"):
                        seed_orig_edge_denom[seed][grp][np.ix_(pn, pn)] += 1.0
                    for (u, v) in E_roi:
                        for grp in (cls, "overall"):
                            seed_orig_edge_num[seed][grp][u, v] += 1.0
                            seed_orig_edge_num[seed][grp][v, u] += 1.0
                except Exception:
                    pass

    seed_orig_edge_prob_cond = {}
    for seed in SEEDS:
        seed_orig_edge_prob_cond[seed] = {}
        for grp in ("schizo", "control", "overall"):
            num = seed_orig_edge_num[seed][grp]
            denom = seed_orig_edge_denom[seed][grp]
            prob = np.zeros_like(num)
            mask = denom > 0
            prob[mask] = num[mask] / denom[mask]
            np.fill_diagonal(prob, 0.0)
            seed_orig_edge_prob_cond[seed][grp] = prob[TRIU[0], TRIU[1]]

    # ---- Collect explained nodes + induced edges (per class + overall) ----
    seed_subject_nodes = defaultdict(lambda: {"schizo": [], "control": [], "overall": []})
    seed_subject_edges = defaultdict(lambda: {"schizo": [], "control": [], "overall": []})

    for (seed, fold, sid), entry in by_subject.items():
        if seed not in SEEDS:
            continue
        cls = entry["class_name"]
        local_nodes = set(int(x) for x in entry["best_nodes"])
        key = (int(seed), int(fold), str(sid))
        if key not in orig_subject_roimap:
            key2 = (int(seed), int(fold) - 1, str(sid))
            if key2 in orig_subject_roimap:
                key = key2
            else:
                for grp in (cls, "overall"):
                    seed_subject_nodes[seed][grp].append(set())
                    seed_subject_edges[seed][grp].append(set())
                continue
        roi0 = orig_subject_roimap[key]
        S_global = {int(roi0[loc]) for loc in local_nodes
                    if 0 <= loc < len(roi0) and 0 <= int(roi0[loc]) < FULL_ATLAS_SIZE}
        E_orig = orig_subject_edges[key]
        E_expl = {(u, v) for (u, v) in E_orig if u in S_global and v in S_global}
        for grp in (cls, "overall"):
            seed_subject_nodes[seed][grp].append(S_global)
            seed_subject_edges[seed][grp].append(E_expl)

    # ---- Seed-level node frequencies ----
    node_baseline = N_MIN / FULL_ATLAS_SIZE
    seed_node_freq, seed_n_subj = {}, {}
    for seed in SEEDS:
        seed_node_freq[seed], seed_n_subj[seed] = {}, {}
        for grp in ("schizo", "control", "overall"):
            subj_sets = seed_subject_nodes[seed][grp]
            seed_n_subj[seed][grp] = len(subj_sets)
            if len(subj_sets) == 0:
                seed_node_freq[seed][grp] = None
                continue
            freq = np.zeros(FULL_ATLAS_SIZE, float)
            for S in subj_sets:
                for u in S:
                    if 0 <= u < FULL_ATLAS_SIZE:
                        freq[u] += 1.0
            seed_node_freq[seed][grp] = freq / len(subj_sets)

    # ---- Seed-level edge frequencies + enrichment delta ----
    seed_edge_obs, seed_edge_exp, seed_edge_delta = {}, {}, {}
    for seed in SEEDS:
        seed_edge_obs[seed], seed_edge_exp[seed], seed_edge_delta[seed] = {}, {}, {}
        for grp in ("schizo", "control", "overall"):
            subj_edge_sets = seed_subject_edges[seed][grp]
            n_subj = len(subj_edge_sets)
            if n_subj == 0:
                seed_edge_obs[seed][grp] = seed_edge_exp[seed][grp] = seed_edge_delta[seed][grp] = None
                continue
            obs = np.zeros(N_EDGES, float)
            for E in subj_edge_sets:
                for (u, v) in E:
                    idx = edge_to_idx(int(u), int(v))
                    if idx is not None:
                        obs[idx] += 1.0
            obs /= n_subj
            p_u = seed_node_freq[seed][grp]
            cond = seed_orig_edge_prob_cond[seed][grp]
            seed_edge_obs[seed][grp] = obs
            if p_u is None:
                seed_edge_exp[seed][grp] = seed_edge_delta[seed][grp] = None
                continue
            exp = p_u[TRIU[0]] * p_u[TRIU[1]] * cond
            seed_edge_exp[seed][grp] = exp
            seed_edge_delta[seed][grp] = obs - exp

    # ---- Node inference ----
    p_sz, q_sz, m_sz, std_sz, lo_sz, hi_sz, n_sz = compute_perclass_node_stats(
        seed_node_freq, SEEDS, "schizo", FULL_ATLAS_SIZE, node_baseline, MIN_SEEDS, MIN_SUBJECTS, seed_n_subj)
    p_hc, q_hc, m_hc, std_hc, lo_hc, hi_hc, n_hc = compute_perclass_node_stats(
        seed_node_freq, SEEDS, "control", FULL_ATLAS_SIZE, node_baseline, MIN_SEEDS, MIN_SUBJECTS, seed_n_subj)
    p_ov, q_ov, m_ov, std_ov, lo_ov, hi_ov, n_ov = compute_perclass_node_stats(
        seed_node_freq, SEEDS, "overall", FULL_ATLAS_SIZE, node_baseline, MIN_SEEDS, MIN_SUBJECTS, seed_n_subj)
    p2_n, q2_n, pgt_n, qgt_n, plt_n, qlt_n, md_n, std_n, lo_n, hi_n, n_d = compute_delta_node_stats(
        seed_node_freq, SEEDS, FULL_ATLAS_SIZE, MIN_SEEDS, MIN_SUBJECTS, seed_n_subj)

    rows_node_perclass = []
    for grp, p_a, q_a, m_a, s_a, lo_a, hi_a, n_a in [
        ("schizo", p_sz, q_sz, m_sz, std_sz, lo_sz, hi_sz, n_sz),
        ("control", p_hc, q_hc, m_hc, std_hc, lo_hc, hi_hc, n_hc),
        ("overall", p_ov, q_ov, m_ov, std_ov, lo_ov, hi_ov, n_ov),
    ]:
        for i in range(FULL_ATLAS_SIZE):
            if not np.isfinite(p_a[i]):
                continue
            rows_node_perclass.append({
                "group": grp, "node_global_id": i, "roi_label": f"ROI_{i+1}",
                "n_seeds": int(n_a[i]), "mean_freq": float(m_a[i]), "std_freq": float(s_a[i]),
                "ci95_lo": float(lo_a[i]), "ci95_hi": float(hi_a[i]), "baseline": float(node_baseline),
                "mean_minus_baseline": float(m_a[i] - node_baseline),
                "p_gt_baseline": float(p_a[i]), "q_gt_baseline_BH": float(q_a[i]),
            })
    df_node_perclass = pd.DataFrame(rows_node_perclass).sort_values(
        ["group", "p_gt_baseline"]).reset_index(drop=True)

    rows_node_delta = []
    for i in range(FULL_ATLAS_SIZE):
        if not np.isfinite(p2_n[i]):
            continue
        rows_node_delta.append({
            "node_global_id": i, "roi_label": f"ROI_{i+1}", "n_seeds": int(n_d[i]),
            "mean_D_seed": float(md_n[i]), "std_D_seed": float(std_n[i]),
            "ci95_lo": float(lo_n[i]), "ci95_hi": float(hi_n[i]),
            "p_two_sided": float(p2_n[i]), "q_two_sided_BH": float(q2_n[i]),
            "p_SZ_gt_HC": float(pgt_n[i]), "q_SZ_gt_HC_BH": float(qgt_n[i]),
            "p_HC_gt_SZ": float(plt_n[i]), "q_HC_gt_SZ_BH": float(qlt_n[i]),
        })
    df_node_delta = pd.DataFrame(rows_node_delta).sort_values("p_two_sided").reset_index(drop=True)

    # ---- Node + edge stability ----
    df_stab_sz, jac_sz = compute_stability_nodes(seed_node_freq, SEEDS, "schizo", FULL_ATLAS_SIZE, TOP_K, MIN_SEEDS)
    df_stab_hc, jac_hc = compute_stability_nodes(seed_node_freq, SEEDS, "control", FULL_ATLAS_SIZE, TOP_K, MIN_SEEDS)
    df_stab_ov, jac_ov = compute_stability_nodes(seed_node_freq, SEEDS, "overall", FULL_ATLAS_SIZE, TOP_K, MIN_SEEDS)
    df_node_stability = pd.concat([df_stab_sz, df_stab_hc, df_stab_ov], ignore_index=True)
    if len(df_node_stability) > 0:
        df_node_stability = df_node_stability.sort_values(
            ["group", "freq_topK", "median_rank"], ascending=[True, False, True]).reset_index(drop=True)

    EDGE_TOP_K = 100
    df_es_sz, _ = compute_edge_stability(seed_edge_delta, SEEDS, "schizo", EDGE_TOP_K, MIN_SEEDS)
    df_es_hc, _ = compute_edge_stability(seed_edge_delta, SEEDS, "control", EDGE_TOP_K, MIN_SEEDS)
    df_es_ov, _ = compute_edge_stability(seed_edge_delta, SEEDS, "overall", EDGE_TOP_K, MIN_SEEDS)
    df_edge_stability = pd.concat([df_es_sz, df_es_hc, df_es_ov], ignore_index=True)
    df_es_cd_abs, _ = compute_edge_stability_classdiff(seed_edge_delta, SEEDS, EDGE_TOP_K, MIN_SEEDS, "abs")
    df_es_cd_signed, _ = compute_edge_stability_classdiff(seed_edge_delta, SEEDS, EDGE_TOP_K, MIN_SEEDS, "signed")

    # ---- Consensus node sets (q < 0.05) ----
    q_thr = 0.05
    consensus = {}
    for grp in ("schizo", "control", "overall"):
        consensus[grp] = df_node_perclass[
            (df_node_perclass["group"] == grp) & (df_node_perclass["q_gt_baseline_BH"] < q_thr)
        ]["node_global_id"].tolist()
    consensus["union_sz_hc"] = sorted(set(consensus["schizo"]) | set(consensus["control"]))
    consensus["diff_SZ_gt_HC"] = df_node_delta[df_node_delta["q_SZ_gt_HC_BH"] < q_thr]["node_global_id"].tolist()
    consensus["diff_HC_gt_SZ"] = df_node_delta[df_node_delta["q_HC_gt_SZ_BH"] < q_thr]["node_global_id"].tolist()

    df_consensus = pd.DataFrame([
        {"consensus_set": grp, "node_global_id": n, "roi_label": f"ROI_{n+1}"}
        for grp, nodes in consensus.items() for n in nodes
    ])

    # ---- Consensus graphs on union nodes (SZ / HC) ----
    union_nodes = consensus["union_sz_hc"]
    df_cg_sz, _, bin_sz = build_group_consensus_graph(
        seed_subject_nodes, seed_subject_edges, SEEDS, "schizo", union_nodes, MIN_SEEDS, 0.5)
    df_cg_hc, _, bin_hc = build_group_consensus_graph(
        seed_subject_nodes, seed_subject_edges, SEEDS, "control", union_nodes, MIN_SEEDS, 0.5)
    df_consensus_graphs = pd.concat([df_cg_sz, df_cg_hc], ignore_index=True)
    df_consensus_graphs_thr = df_consensus_graphs[
        (df_consensus_graphs["passes_min_seeds"]) & (df_consensus_graphs["binary_consensus"] == 1)
    ].sort_values(["group", "edge_prevalence_mean"], ascending=[True, False]).reset_index(drop=True)

    # ---- Save ----
    df_node_perclass.to_csv(os.path.join(OUT_DIR, "subgraphx_node_stats_perclass.csv"), index=False)
    df_node_delta.to_csv(os.path.join(OUT_DIR, "subgraphx_node_stats_delta.csv"), index=False)
    df_node_stability.to_csv(os.path.join(OUT_DIR, "subgraphx_node_stability.csv"), index=False)
    df_edge_stability.to_csv(os.path.join(OUT_DIR, "subgraphx_edge_stability_enrichment.csv"), index=False)
    df_es_cd_abs.to_csv(os.path.join(OUT_DIR, f"subgraphx_edge_stability_classdiff_abs_top{EDGE_TOP_K}.csv"), index=False)
    df_es_cd_signed.to_csv(os.path.join(OUT_DIR, f"subgraphx_edge_stability_classdiff_signed_top{EDGE_TOP_K}.csv"), index=False)
    df_consensus.to_csv(os.path.join(OUT_DIR, "subgraphx_consensus_node_sets.csv"), index=False)
    df_consensus_graphs.to_csv(os.path.join(OUT_DIR, "subgraphx_consensus_graphs_union_nodes.csv"), index=False)

    out_xlsx = os.path.join(OUT_DIR, "subgraphx_inference_results_structure.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_consensus_graphs.to_excel(w, sheet_name="Consensus_Graphs_Union", index=False)
        df_consensus_graphs_thr.to_excel(w, sheet_name="Consensus_Graphs_Thresh", index=False)
        df_node_perclass.to_excel(w, sheet_name="Node_Stats_PerGroup", index=False)
        df_node_delta.to_excel(w, sheet_name="Node_Stats_Delta", index=False)
        df_node_stability.to_excel(w, sheet_name="Node_Stability", index=False)
        df_es_cd_abs.to_excel(w, sheet_name=f"Edge_Stab_CD_abs_T{EDGE_TOP_K}", index=False)
        df_es_cd_signed.to_excel(w, sheet_name=f"Edge_Stab_CD_signed_T{EDGE_TOP_K}", index=False)
        df_consensus.to_excel(w, sheet_name="Consensus_Node_Sets", index=False)

    print("Saved all outputs to:", OUT_DIR)
    print(f"SZ binary consensus edges: {int(np.sum(bin_sz) // 2) if bin_sz.size else 0}")
    print(f"HC binary consensus edges: {int(np.sum(bin_hc) // 2) if bin_hc.size else 0}")


if __name__ == "__main__":
    main()
