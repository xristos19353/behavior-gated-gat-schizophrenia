"""External replication: GATE vs. zero-behaviour model on an independent cohort.

Two independently trained, frozen discovery models are evaluated on the external
replication cohort:

1. GATE model      - trained with real behavioural graph features, evaluated with
                     real behavioural graph features.
2. ZERO-GATE model - a separately trained model whose behavioural graph features
                     were zeroed during training, evaluated with zeroed features.

The two checkpoints differ in their weights (they are distinct trainings), so the
comparison isolates the behavioural-gate *condition*, not a test-time toggle on a
single network.

The external feature table is z-scored and residualised using the discovery
cohort's frozen standardisation statistics and nuisance-regression models, so no
external statistics leak into preprocessing.

Outputs (written to ``config.EXTERNAL_RESULTS_DIR``):
- external_gate_vs_zero_model_comparison.csv
- external_gate_vs_zero_subject_predictions.csv
- external_gate_vs_zero_paired_bootstrap.csv
- ROC (PDF + PNG)
- external_consensus_connections_*.csv / .xlsx
- XAI-compatible .pth packages for both models
"""

from __future__ import annotations

import copy
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import config
from audit_graph_list import audit_graph_list
from build_graphs_from_tr import update_graphs_from_Tr
from models import GAT
from sparsify import sparsify_TZ

# ---------------------------------------------------------------------------
# Paths (from config) and analysis settings
# ---------------------------------------------------------------------------
FINAL_MODEL_PATH_GATE = str(config.GATE_MODEL_PATH)
FINAL_MODEL_PATH_ZERO = str(config.ZERO_MODEL_PATH)
EXTERNAL_SPM_MAT = str(config.EXTERNAL_SPM_MAT)
EXTERNAL_ONSETS_MAT = str(config.EXTERNAL_ONSETS_MAT)
OUTPUT_DIR = str(config.EXTERNAL_RESULTS_DIR)

# XAI-compatible result packages.
DIR_GATE = os.path.join(OUTPUT_DIR, "GAT_gate_only")
DIR_ZERO = os.path.join(OUTPUT_DIR, "GAT_ZERO_gate")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THR = 0.5
CURRENT_SEED = 100
FULL_GRAPH_FEAT_DIM = 4

# Discovery consensus nodes (0-based ROI ids) used for the external consensus graph.
CONSENSUS_NODES = [9, 22, 27, 69, 72, 75, 83, 86, 87, 88, 96, 101]
EDGE_BIN_THR = 0.50
MIN_SUBJECT_SUPPORT = 0

N_BOOT = 10000
N_PERM = 10000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_mask(tensor, mask):
    """Keep only the feature columns selected by ``mask``."""
    mask_t = torch.tensor(mask, device=tensor.device).bool()
    return tensor[:, mask_t]


def get_data_by_variant(dlist, variant_id):
    return [d for d in dlist if d.variant == variant_id]


def load_frozen_model(final_model_path, device):
    """Load a frozen discovery checkpoint into a GAT and return it with metadata."""
    pkg = torch.load(final_model_path, map_location="cpu", weights_only=False)

    final_params = pkg["final_params"]
    graph_feat_mask = pkg["graph_feat_mask"]
    node_mask = pkg["node_mask"]

    model = GAT(
        in_channels=int(sum(node_mask)),
        hidden_channels=final_params["hidden_dim"],
        dropout=final_params["dropout"],
        graph_feat_dim=int(sum(graph_feat_mask)),
        num_layers=final_params["num_layers"],
        heads=final_params["heads"],
        use_edge_attr=False,
    ).to(device)
    model.load_state_dict(pkg["model_state_dict"])
    model.eval()

    return {
        "pkg": pkg,
        "model": model,
        "final_params": final_params,
        "graph_feat_mask": graph_feat_mask,
        "node_mask": node_mask,
        "discovery_stats": pkg["final_stats"],
        "discovery_models": pkg["all_models"],
    }


def apply_regress_out_from_discovery(T_ext, discovery_stats, discovery_models):
    """Residualise the external table using discovery stats and fitted models."""
    age_mean, age_std = discovery_stats["age_mean"], discovery_stats["age_std"]
    sex_mean, sex_std = discovery_stats["sex_mean"], discovery_stats["sex_std"]

    age = np.array([s.AGE for s in T_ext.Demographics])
    sex = np.array([s.SEX for s in T_ext.Demographics])
    z_age = (age - age_mean) / age_std
    z_sex = (sex - sex_mean) / sex_std

    for i in range(len(T_ext.Demographics)):
        T_ext.Demographics[i].AGE = z_age[i]
        T_ext.Demographics[i].SEX = z_sex[i]

    X_ext = np.column_stack((z_age, z_sex))
    n_ext = len(z_age)

    feat_stats = discovery_stats["features"]
    feat_models = discovery_models
    features_to_residualize = [
        "WMV", "GMV", "EffectSize", "ALFF", "ReHo", "DC", "Z", "Zweighted", "Behavior",
    ]

    for feat_name in features_to_residualize:
        if not hasattr(T_ext, feat_name):
            continue
        feat_data = getattr(T_ext, feat_name)
        models = feat_models.get(feat_name, {})
        stats = feat_stats.get(feat_name, {})
        if len(feat_data) == 0:
            continue
        first_struct = feat_data[0]

        # Connectivity matrices: residualise each cell.
        if feat_name in ("Z", "Zweighted"):
            key = feat_name
            if key not in models:
                continue
            try:
                matrices = [np.array(getattr(subj, key), dtype=float) for subj in feat_data]
                stack = np.stack(matrices)
                n_subjects, d1, d2 = stack.shape
                flat = stack.reshape(n_subjects, -1)

                mean = stats[key]["mean"]
                std = stats[key]["std"]
                std[std == 0] = 1
                z_data = (flat - mean) / std

                residuals = z_data - models[key].predict(X_ext)
                residuals_reshaped = residuals.reshape(n_subjects, d1, d2)
                for i in range(n_subjects):
                    setattr(feat_data[i], key, residuals_reshaped[i])
            except Exception as e:
                print(f"  [WARN] {feat_name}.{key}: {e}")
            continue

        # Scalar features.
        keys = [
            k for k in first_struct.__dict__.keys()
            if isinstance(getattr(first_struct, k), (int, float, np.integer, np.floating))
        ]
        for key in keys:
            if feat_name == "Behavior" and key == "Group":
                continue
            if key not in models:
                continue
            try:
                y = np.array([getattr(subj, key) for subj in feat_data], dtype=float)
                y_mean = stats[key]["mean"]
                y_std = stats[key]["std"] or 1
                z_y = (y - y_mean) / y_std
                residuals = z_y - models[key].predict(X_ext)
                for i in range(n_ext):
                    setattr(feat_data[i], key, residuals[i])
            except Exception as e:
                print(f"  [WARN] {feat_name}.{key}: {e}")

    return T_ext


def build_external_data(reference_params, reference_stats, reference_models):
    """Build external PyG graphs with discovery-frozen preprocessing applied."""
    print("\n-- Building external graphs --")

    ext_mat = scipy.io.loadmat(EXTERNAL_SPM_MAT, squeeze_me=True, struct_as_record=False)
    flag = ext_mat.get("flag")
    T_ext = ext_mat.get("T")

    if flag is not None and int(flag) == 1:
        raise RuntimeError("External results.mat: flag==1 empty peakTable.")
    if T_ext is None or (isinstance(T_ext, np.ndarray) and T_ext.size == 0):
        raise RuntimeError("External results.mat: T empty.")

    T_ext = sparsify_TZ(T_ext, er_factor=reference_params["er_factor"])
    print("  sparsify_TZ done")

    mat_ids = scipy.io.loadmat(EXTERNAL_ONSETS_MAT)
    num_ids = mat_ids["num_ids"]

    # External convention: subject ids starting with 4 are patients (SZ), else HC.
    ext_subjects = [
        (f"sub-{num_ids[i][0]}_Pan", 1 if str(num_ids[i][0]).startswith("4") else 0)
        for i in range(len(num_ids))
    ]
    n_sz = sum(l for _, l in ext_subjects)
    print(f"  External subjects: {len(ext_subjects)}  SZ={n_sz}, HC={len(ext_subjects) - n_sz}")

    T_ext = apply_regress_out_from_discovery(T_ext, reference_stats, reference_models)
    print("  Regress-out using discovery stats/models done")

    feature_generators = [
        lambda: torch.rand(20, 6),
        lambda: torch.ones(20, 6),
        lambda: torch.eye(20)[:20, :6],
    ]

    ext_data_list = []
    for subject_id, label_val in ext_subjects:
        adj = np.random.rand(20, 20)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        adj[adj < 0.5] = 0
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        y = torch.tensor([label_val], dtype=torch.long)

        for variant_id, gen in enumerate(feature_generators):
            data = Data(x=gen(), edge_index=edge_index, y=y)
            data.variant = variant_id
            data.subject_id = subject_id
            ext_data_list.append(data)

    ext_data = get_data_by_variant(ext_data_list, reference_params["variant_id"])
    ext_data = update_graphs_from_Tr(T_ext, ext_data)
    audit_graph_list(ext_data, tag="EXTERNAL", max_print=3)
    return ext_data


def run_external_inference(model, ext_data, node_mask, graph_feat_mask, device,
                           zero_graph_feat=False, full_graph_feat_dim=4, threshold=0.5):
    """Run inference on the external cohort and collect metrics + node weights."""
    loader = DataLoader(ext_data, batch_size=1, shuffle=False)
    probs, labels, pred_bin = [], [], []
    node_alpha_list, node_alpha_sids = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_in = apply_mask(batch.x, node_mask)

            if zero_graph_feat:
                graph_feat = torch.zeros(
                    (batch.num_graphs, int(sum(graph_feat_mask))),
                    dtype=torch.float32, device=device,
                )
            else:
                graph_feat = apply_mask(
                    batch.graph_feat.view(-1, full_graph_feat_dim), graph_feat_mask
                )

            out, alpha = model(
                x_in, batch.edge_index, graph_feat, batch.batch, return_pool_weights=True
            )
            prob = float(torch.sigmoid(out).squeeze().item())

            probs.append(prob)
            labels.append(int(batch.y.item()))
            pred_bin.append(int(prob >= threshold))
            node_alpha_list.append(alpha.cpu().numpy())
            node_alpha_sids.append(batch.subject_id[0])

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred_bin = np.asarray(pred_bin, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_bin, labels=[0, 1]).ravel()

    return {
        "auc": roc_auc_score(labels, probs),
        "balanced_acc": balanced_accuracy_score(labels, pred_bin),
        "sensitivity": recall_score(labels, pred_bin, pos_label=1),
        "specificity": recall_score(labels, pred_bin, pos_label=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "probs": probs, "labels": labels, "pred_bin": pred_bin,
        "node_alpha": node_alpha_list, "node_alpha_subject_id": node_alpha_sids,
    }


def print_external_summary(name, res):
    print(f"\n{'=' * 48}")
    print(f"  {name}")
    print(f"  AUC              : {res['auc']:.4f}")
    print(f"  Balanced Acc     : {res['balanced_acc']:.4f}")
    print(f"  Sensitivity      : {res['sensitivity']:.4f} ({res['tp']} TP / {res['tp'] + res['fn']} SZ)")
    print(f"  Specificity      : {res['specificity']:.4f} ({res['tn']} TN / {res['tn'] + res['fp']} HC)")
    print(f"  TP={res['tp']}  TN={res['tn']}  FP={res['fp']}  FN={res['fn']}")
    print(f"{'=' * 48}")


def bootstrap_metrics(y_true, y_prob, threshold=0.5, n_boot=10000, seed=42):
    """Bootstrap mean and 95% CI for AUC, balanced accuracy, sensitivity, specificity."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    aucs, bacs, sens, spec = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        yhat = (yp >= threshold).astype(int)
        aucs.append(roc_auc_score(yt, yp))
        bacs.append(balanced_accuracy_score(yt, yhat))
        sens.append(recall_score(yt, yhat, pos_label=1, zero_division=0))
        spec.append(recall_score(yt, yhat, pos_label=0, zero_division=0))

    def out(arr):
        arr = np.asarray(arr, dtype=float)
        return {"mean": float(np.mean(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5))}

    return {"auc": out(aucs), "balanced_acc": out(bacs),
            "sensitivity": out(sens), "specificity": out(spec)}


def permutation_test_all_metrics(y_true, y_prob, threshold=0.5, n_perm=10000, seed=42):
    """Label-permutation p-values for each metric against chance."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    real_auc = roc_auc_score(y_true, y_prob)
    real_bac = balanced_accuracy_score(y_true, y_pred)
    real_sens = recall_score(y_true, y_pred, pos_label=1)
    real_spec = recall_score(y_true, y_pred, pos_label=0)

    perm_auc, perm_bac, perm_sens, perm_spec = [], [], [], []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        if len(np.unique(y_perm)) < 2:
            continue
        perm_auc.append(roc_auc_score(y_perm, y_prob))
        perm_bac.append(balanced_accuracy_score(y_perm, y_pred))
        perm_sens.append(recall_score(y_perm, y_pred, pos_label=1, zero_division=0))
        perm_spec.append(recall_score(y_perm, y_pred, pos_label=0, zero_division=0))

    perm_auc = np.asarray(perm_auc)
    perm_bac = np.asarray(perm_bac)
    perm_sens = np.asarray(perm_sens)
    perm_spec = np.asarray(perm_spec)

    return {
        "real": {"auc": real_auc, "balanced_acc": real_bac,
                 "sensitivity": real_sens, "specificity": real_spec},
        "pvals": {
            "auc": float((np.sum(perm_auc >= real_auc) + 1) / (len(perm_auc) + 1)),
            "balanced_acc": float((np.sum(perm_bac >= real_bac) + 1) / (len(perm_bac) + 1)),
            "sensitivity": float((np.sum(perm_sens >= real_sens) + 1) / (len(perm_sens) + 1)),
            "specificity": float((np.sum(perm_spec >= real_spec) + 1) / (len(perm_spec) + 1)),
        },
        "null": {"auc": perm_auc, "balanced_acc": perm_bac,
                 "sensitivity": perm_sens, "specificity": perm_spec},
    }


def paired_bootstrap_metric_differences(y_true, prob_gate, prob_zero,
                                        threshold=0.5, n_boot=10000, seed=42):
    """Paired bootstrap of GATE-minus-ZERO metric differences."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    prob_gate = np.asarray(prob_gate, dtype=float)
    prob_zero = np.asarray(prob_zero, dtype=float)
    n = len(y_true)

    d_auc, d_bac, d_sens, d_spec = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, pg, pz = y_true[idx], prob_gate[idx], prob_zero[idx]
        if len(np.unique(yt)) < 2:
            continue
        yg = (pg >= threshold).astype(int)
        yz = (pz >= threshold).astype(int)
        d_auc.append(roc_auc_score(yt, pg) - roc_auc_score(yt, pz))
        d_bac.append(balanced_accuracy_score(yt, yg) - balanced_accuracy_score(yt, yz))
        d_sens.append(recall_score(yt, yg, pos_label=1, zero_division=0)
                      - recall_score(yt, yz, pos_label=1, zero_division=0))
        d_spec.append(recall_score(yt, yg, pos_label=0, zero_division=0)
                      - recall_score(yt, yz, pos_label=0, zero_division=0))

    def summarize(arr):
        arr = np.asarray(arr, dtype=float)
        return {"mean": float(np.mean(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "p_two_sided": float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))),
                "p_one_sided_gate_greater": float(np.mean(arr <= 0))}

    return {"delta_auc": summarize(d_auc), "delta_balanced_acc": summarize(d_bac),
            "delta_sensitivity": summarize(d_sens), "delta_specificity": summarize(d_spec)}


def save_metric_summary_csv(gate_results, zero_results, ci_gate, ci_zero,
                            perm_gate, perm_zero, paired_diff, out_dir):
    """Write the per-model metric table plus the GATE-minus-ZERO row."""
    rows = []
    for model_name, res, ci, perm in [
        ("Gate", gate_results, ci_gate, perm_gate),
        ("Zero_gate", zero_results, ci_zero, perm_zero),
    ]:
        row = {"model": model_name}
        for metric, key in [("AUC", "auc"), ("Balanced_Acc", "balanced_acc"),
                            ("Sensitivity", "sensitivity"), ("Specificity", "specificity")]:
            row[metric] = res[key]
            row[f"{metric}_boot_mean"] = ci[key]["mean"]
            row[f"{metric}_CI_low"] = ci[key]["ci_low"]
            row[f"{metric}_CI_high"] = ci[key]["ci_high"]
            row[f"{metric}_perm_p"] = perm["pvals"][key]
        row.update({
            "TP": res["tp"], "TN": res["tn"], "FP": res["fp"], "FN": res["fn"],
            "N_total": len(res["labels"]),
            "N_SZ": int(np.sum(res["labels"] == 1)),
            "N_HC": int(np.sum(res["labels"] == 0)),
        })
        rows.append(row)

    diff_row = {"model": "Gate_minus_zero"}
    for metric, key, dkey in [
        ("AUC", "auc", "delta_auc"),
        ("Balanced_Acc", "balanced_acc", "delta_balanced_acc"),
        ("Sensitivity", "sensitivity", "delta_sensitivity"),
        ("Specificity", "specificity", "delta_specificity"),
    ]:
        diff_row[metric] = gate_results[key] - zero_results[key]
        diff_row[f"{metric}_boot_mean"] = paired_diff[dkey]["mean"]
        diff_row[f"{metric}_CI_low"] = paired_diff[dkey]["ci_low"]
        diff_row[f"{metric}_CI_high"] = paired_diff[dkey]["ci_high"]
        diff_row[f"{metric}_perm_p"] = np.nan
    diff_row.update({
        "TP": np.nan, "TN": np.nan, "FP": np.nan, "FN": np.nan,
        "N_total": len(gate_results["labels"]),
        "N_SZ": int(np.sum(gate_results["labels"] == 1)),
        "N_HC": int(np.sum(gate_results["labels"] == 0)),
    })
    rows.append(diff_row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "external_gate_vs_zero_model_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return df


def plot_roc_gate_vs_zero(labels, probs_gate, probs_zero, auc_gate, auc_zero, out_dir):
    """ROC overlay for the GATE and ZERO-GATE models."""
    fpr_gate, tpr_gate, _ = roc_curve(labels, probs_gate)
    fpr_zero, tpr_zero, _ = roc_curve(labels, probs_zero)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr_gate, tpr_gate, lw=2.5, label=f"GATE AUC = {auc_gate:.3f}")
    ax.plot(fpr_zero, tpr_zero, lw=2.0, linestyle="--", label=f"ZERO-GATE AUC = {auc_zero:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1.2, color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("External validation ROC", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_pdf = os.path.join(out_dir, "roc_external_gate_vs_zero.pdf")
    out_png = os.path.join(out_dir, "roc_external_gate_vs_zero.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}\nSaved {out_png}")


def save_xai_pth(save_dir, filename, experiment_name, seed, bundle, results, ext_data):
    """Save an explainability-compatible package mirroring the discovery format."""
    final_params = bundle["final_params"]
    xai_params = {
        "hidden_dim": final_params.get("hidden_dim", final_params.get("hidden_channels", 128)),
        "dropout": final_params.get("dropout", 0.1),
        "num_layers": final_params.get("num_layers", 2),
        "heads": final_params.get("heads", 1),
    }
    df_outer = pd.DataFrame({
        "y_true": [list(results["labels"])],
        "y_pred": [list(results["probs"])],
        "node_alpha": [results["node_alpha"]],
        "node_alpha_subject_id": [results["node_alpha_subject_id"]],
        "params": [xai_params],
        "best_node_mask": [bundle["node_mask"]],
        "best_graph_feat_mask": [bundle["graph_feat_mask"]],
    })
    to_save = {
        "experiment": experiment_name, "seed": seed, "df_outer": df_outer,
        "models": [bundle["model"].state_dict()],
        "Test_Data_outer": [copy.deepcopy(ext_data)],
        "outer_folds": 1, "inner_folds": 0,
    }
    save_path = os.path.join(save_dir, filename)
    torch.save(to_save, save_path)
    print(f"Saved XAI pth: {save_path}")


def roi_ids_per_node_from_graph(g, full_size=114):
    """Return the 0-based ROI id of each node from the graph's roi_to_node map."""
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

    if (roi_raw.min() >= 1) and (roi_raw.max() <= full_size):
        return roi_raw - 1
    return roi_raw


def graph_roi_edge_set(g, full_size=114):
    """Return the set of undirected ROI-ROI edges and the set of present ROIs."""
    roi0 = roi_ids_per_node_from_graph(g, full_size=full_size)
    edge_index = g.edge_index
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index has unexpected shape: {edge_index.shape}")

    E = set()
    for a, b in zip(edge_index[0].astype(int), edge_index[1].astype(int)):
        ra, rb = int(roi0[a]), int(roi0[b])
        if ra == rb:
            continue
        E.add((ra, rb) if ra < rb else (rb, ra))
    return E, set(int(x) for x in roi0.tolist())


def build_external_consensus_graph(graphs, labels, consensus_nodes, group_name="all",
                                   edge_bin_thr=0.5, min_subject_support=5, full_size=114):
    """Prevalence of each consensus-node edge across external subjects (per group)."""
    consensus_nodes = sorted(set(int(x) for x in consensus_nodes))
    rows = []

    if group_name == "sz":
        keep_idx = [i for i, y in enumerate(labels) if int(y) == 1]
    elif group_name == "hc":
        keep_idx = [i for i, y in enumerate(labels) if int(y) == 0]
    else:
        keep_idx = list(range(len(graphs)))

    if len(keep_idx) == 0:
        return pd.DataFrame([])

    graph_info = []
    for i in keep_idx:
        try:
            graph_info.append(graph_roi_edge_set(graphs[i], full_size=full_size))
        except Exception as e:
            print(f"[WARN] Failed graph for consensus graph {group_name}, idx={i}: {e}")

    for ii in range(len(consensus_nodes)):
        for jj in range(ii + 1, len(consensus_nodes)):
            u, v = consensus_nodes[ii], consensus_nodes[jj]
            den = num = 0
            for E_roi, present_rois in graph_info:
                if (u in present_rois) and (v in present_rois):
                    den += 1
                    e = (u, v) if u < v else (v, u)
                    if e in E_roi:
                        num += 1

            prevalence = (num / den) if den > 0 else np.nan
            passes_support = bool(den >= min_subject_support)
            binary_consensus = int(
                passes_support and np.isfinite(prevalence) and prevalence >= edge_bin_thr
            )
            rows.append({
                "group": group_name, "node_u": u, "node_v": v,
                "roi_u": f"ROI_{u + 1}", "roi_v": f"ROI_{v + 1}",
                "n_subjects_both_present": int(den), "n_subjects_edge_present": int(num),
                "edge_prevalence": float(prevalence) if np.isfinite(prevalence) else np.nan,
                "passes_min_subject_support": passes_support,
                "binary_consensus": binary_consensus,
                "binary_threshold": float(edge_bin_thr),
                "min_subject_support": int(min_subject_support),
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["group", "binary_consensus", "edge_prevalence"],
                            ascending=[True, False, False]).reset_index(drop=True)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIR_GATE, exist_ok=True)
    os.makedirs(DIR_ZERO, exist_ok=True)
    print(f"Device: {DEVICE}")

    print("\n-- Loading frozen models --")
    gate_bundle = load_frozen_model(FINAL_MODEL_PATH_GATE, DEVICE)
    zero_bundle = load_frozen_model(FINAL_MODEL_PATH_ZERO, DEVICE)
    print("GATE params:", gate_bundle["final_params"])
    print("ZERO-GATE params:", zero_bundle["final_params"])

    # Both models share discovery preprocessing; use the GATE model as reference.
    ext_data = build_external_data(
        reference_params=gate_bundle["final_params"],
        reference_stats=gate_bundle["discovery_stats"],
        reference_models=gate_bundle["discovery_models"],
    )

    print("\n-- External inference: GATE model with real behavioural inputs --")
    gate_results = run_external_inference(
        model=gate_bundle["model"], ext_data=ext_data,
        node_mask=gate_bundle["node_mask"], graph_feat_mask=gate_bundle["graph_feat_mask"],
        device=DEVICE, zero_graph_feat=False,
        full_graph_feat_dim=FULL_GRAPH_FEAT_DIM, threshold=THR,
    )

    print("\n-- External inference: ZERO-GATE model with zero behavioural inputs --")
    zero_results = run_external_inference(
        model=zero_bundle["model"], ext_data=ext_data,
        node_mask=zero_bundle["node_mask"], graph_feat_mask=zero_bundle["graph_feat_mask"],
        device=DEVICE, zero_graph_feat=True,
        full_graph_feat_dim=FULL_GRAPH_FEAT_DIM, threshold=THR,
    )

    if not np.array_equal(gate_results["labels"], zero_results["labels"]):
        raise RuntimeError("GATE and ZERO labels are not aligned.")
    labels_true = gate_results["labels"]

    print_external_summary("GATE MODEL", gate_results)
    print_external_summary("ZERO-GATE MODEL", zero_results)

    # Subject-level paired predictions.
    delta_prob = gate_results["probs"] - zero_results["probs"]
    df_subject = pd.DataFrame({
        "subject_id": gate_results["node_alpha_subject_id"],
        "y_true": labels_true,
        "prob_gate": gate_results["probs"],
        "prob_zero": zero_results["probs"],
        "delta_prob_gate_minus_zero": delta_prob,
        "pred_gate": gate_results["pred_bin"],
        "pred_zero": zero_results["pred_bin"],
        "correct_gate": gate_results["pred_bin"] == labels_true,
        "correct_zero": zero_results["pred_bin"] == labels_true,
    })
    out_subject_csv = os.path.join(OUTPUT_DIR, "external_gate_vs_zero_subject_predictions.csv")
    df_subject.to_csv(out_subject_csv, index=False)
    print(f"Saved {out_subject_csv}")
    print(f"Mean delta-prob overall: {delta_prob.mean():.4f} | "
          f"SZ: {delta_prob[labels_true == 1].mean():.4f} | "
          f"HC: {delta_prob[labels_true == 0].mean():.4f}")

    # Bootstrap CIs.
    print("\n-- Bootstrap CIs --")
    ci_gate = bootstrap_metrics(labels_true, gate_results["probs"], THR, N_BOOT, 42)
    ci_zero = bootstrap_metrics(labels_true, zero_results["probs"], THR, N_BOOT, 42)
    for tag, ci in [("GATE", ci_gate), ("ZERO-GATE", ci_zero)]:
        print(f"\n{tag} bootstrap:")
        for k, v in ci.items():
            print(f"  {k}: {v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")

    # Permutation tests vs chance.
    print("\n-- Permutation tests vs chance --")
    perm_gate = permutation_test_all_metrics(labels_true, gate_results["probs"], THR, N_PERM, 42)
    perm_zero = permutation_test_all_metrics(labels_true, zero_results["probs"], THR, N_PERM, 42)
    print("GATE p-values:", perm_gate["pvals"])
    print("ZERO-GATE p-values:", perm_zero["pvals"])

    # Paired bootstrap GATE - ZERO.
    print("\n-- Paired bootstrap: GATE minus ZERO-GATE --")
    paired_diff = paired_bootstrap_metric_differences(
        labels_true, gate_results["probs"], zero_results["probs"], THR, N_BOOT, 42
    )
    for k, v in paired_diff.items():
        print(f"  {k}: mean={v['mean']:.4f}, CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}], "
              f"p_two={v['p_two_sided']:.5f}, p_one_gate_greater={v['p_one_sided_gate_greater']:.5f}")
    df_paired = pd.DataFrame([{"metric": k, **v} for k, v in paired_diff.items()])
    out_paired_csv = os.path.join(OUTPUT_DIR, "external_gate_vs_zero_paired_bootstrap.csv")
    df_paired.to_csv(out_paired_csv, index=False)
    print(f"Saved {out_paired_csv}")

    # Summary table + ROC.
    df_summary = save_metric_summary_csv(
        gate_results, zero_results, ci_gate, ci_zero, perm_gate, perm_zero, paired_diff, OUTPUT_DIR
    )
    print("\nSummary:\n", df_summary.to_string(index=False))
    plot_roc_gate_vs_zero(labels_true, gate_results["probs"], zero_results["probs"],
                          gate_results["auc"], zero_results["auc"], OUTPUT_DIR)

    # XAI packages.
    save_xai_pth(DIR_GATE, f"all_results_GAT_gate_only_seed{CURRENT_SEED}.pth",
                 "GAT_external_GATE", CURRENT_SEED, gate_bundle, gate_results, ext_data)
    save_xai_pth(DIR_ZERO, f"all_results_GAT_ZERO_gate_seed{CURRENT_SEED}.pth",
                 "GAT_external_ZERO", CURRENT_SEED, zero_bundle, zero_results, ext_data)

    # External consensus connections on discovery consensus nodes.
    print("\n-- External consensus connections on discovery consensus nodes --")
    df_sz = build_external_consensus_graph(ext_data, labels_true, CONSENSUS_NODES, "sz",
                                           EDGE_BIN_THR, MIN_SUBJECT_SUPPORT, 114)
    df_hc = build_external_consensus_graph(ext_data, labels_true, CONSENSUS_NODES, "hc",
                                           EDGE_BIN_THR, MIN_SUBJECT_SUPPORT, 114)
    df_all = build_external_consensus_graph(ext_data, labels_true, CONSENSUS_NODES, "all",
                                            EDGE_BIN_THR, MIN_SUBJECT_SUPPORT, 114)
    df_groups = pd.concat([df_sz, df_hc, df_all], ignore_index=True)
    df_thr = df_groups[(df_groups["passes_min_subject_support"]) & (df_groups["binary_consensus"] == 1)].copy()

    df_groups.to_csv(os.path.join(OUTPUT_DIR, "external_consensus_connections_all.csv"), index=False)
    df_thr.to_csv(os.path.join(OUTPUT_DIR, "external_consensus_connections_thresholded.csv"), index=False)

    cons_xlsx = os.path.join(OUTPUT_DIR, "external_consensus_connections.xlsx")
    with pd.ExcelWriter(cons_xlsx, engine="openpyxl") as w:
        df_sz.to_excel(w, sheet_name="SZ_all_edges", index=False)
        df_hc.to_excel(w, sheet_name="HC_all_edges", index=False)
        df_all.to_excel(w, sheet_name="ALL_all_edges", index=False)
        df_thr.to_excel(w, sheet_name="Thresholded", index=False)

        def n_thr(df):
            return int(((df["passes_min_subject_support"]) & (df["binary_consensus"] == 1)).sum())

        pd.DataFrame([
            {"param": "n_consensus_nodes", "value": len(CONSENSUS_NODES)},
            {"param": "consensus_nodes_0based", "value": str(CONSENSUS_NODES)},
            {"param": "edge_binary_threshold", "value": EDGE_BIN_THR},
            {"param": "min_subject_support", "value": MIN_SUBJECT_SUPPORT},
            {"param": "n_thresholded_edges_SZ", "value": n_thr(df_sz)},
            {"param": "n_thresholded_edges_HC", "value": n_thr(df_hc)},
            {"param": "n_thresholded_edges_ALL", "value": n_thr(df_all)},
        ]).to_excel(w, sheet_name="Config", index=False)
    print(f"Saved {cons_xlsx}")

    print("\nExternal validation complete.")
    print(f"Gate AUC={gate_results['auc']:.4f}, BAC={gate_results['balanced_acc']:.4f}")
    print(f"Zero AUC={zero_results['auc']:.4f}, BAC={zero_results['balanced_acc']:.4f}")
    print(f"Delta AUC={gate_results['auc'] - zero_results['auc']:.4f}")
    print(f"Delta BAC={gate_results['balanced_acc'] - zero_results['balanced_acc']:.4f}")


if __name__ == "__main__":
    main()
