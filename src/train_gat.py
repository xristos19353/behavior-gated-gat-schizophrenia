"""Nested cross-validation training of the behaviour-gated GAT (and GCN baseline).

Pipeline overview
-----------------
1. A scaffold graph is created per subject (subject id, label and adjacency
   variant). The real node/edge/graph features are injected later from the
   MATLAB SPM results via :func:`update_graphs_from_Tr`.
2. For every random seed a subject-level nested cross-validation is run:
   * inner folds + Optuna (multi-objective: maximise mean AUC, minimise its std)
     select hyperparameters,
   * the SPM second-level results for the fold are loaded, the connectivity is
     sparsified (:func:`sparsify_TZ`) and demographics are regressed out
     (:func:`regress_out`) before the features are written into the graphs,
   * the outer fold is retrained and evaluated on the held-out subjects.
3. Results are aggregated across seeds (summary tables, ROC curves, bootstrap CI)
   and an optional subject-level permutation test estimates significance.
4. :func:`train_final_model_full_dataset_precomputed` retrains on the full
   discovery sample (plus a matched zero-behaviour control model).

All machine-specific paths come from :mod:`config`.
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
import scipy.io
import torch
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import optuna

import config
from audit_graph_list import audit_graph_list
from build_graphs_from_tr import update_graphs_from_Tr
from models import EarlyStopping, build_model  # noqa: F401  (build_model re-exported)
from permutation_test import permutation_test_seed
from regress_out import regress_out
from sparsify import sparsify_TZ

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "GAT_gate_only"
BASE_RESULTS_DIR = os.path.join(config.RESULTS_ROOT, EXPERIMENT_NAME)
OPTUNA_DIR = os.path.join(BASE_RESULTS_DIR, "optuna_trials")
SPM_RESULTS_BASE = str(config.SPM_RESULTS_BASE)

# When True, load precomputed SPM results.mat from disk instead of calling
# MATLAB on the fly (recommended; see matlab/spm/spm_2level_gat_wrapper.m).
USE_PRECOMPUTED_SPM = True

# Cross-validation / training settings.
NUM_REGIONS = 20
EPOCHS = 200
INNER_FOLDS = 3
OUTER_FOLDS = 3
SEEDS = [100, 101, 102, 103, 104, 105, 106, 107]

# Node-feature variant generators (placeholders; real features come from Tr).
FEATURE_GENERATORS = [
    lambda: torch.rand(NUM_REGIONS, 6),                 # 0: random
    lambda: torch.ones(NUM_REGIONS, 6),                 # 1: constant
    lambda: torch.eye(NUM_REGIONS)[:NUM_REGIONS, :6],   # 2: identity-truncated
]


# ---------------------------------------------------------------------------
# Device / reproducibility
# ---------------------------------------------------------------------------
def setup_device():
    """Select CUDA if available and seed all RNGs for reproducibility."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        device = torch.device("cpu")
        print("Using CPU")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    return device


device = setup_device()


# ---------------------------------------------------------------------------
# Feature masking helpers
# ---------------------------------------------------------------------------
def apply_graph_feature_mask(batch_graph_feat, mask):
    """Keep only the graph-level (behavioural) features selected by ``mask``."""
    mask_tensor = torch.tensor(mask, device=batch_graph_feat.device).bool()
    return batch_graph_feat[:, mask_tensor]


def apply_node_feature_mask(x, mask):
    """Keep only the node features selected by ``mask`` (x: [num_nodes, 6])."""
    mask_tensor = torch.tensor(mask, device=x.device).bool()
    return x[:, mask_tensor]


def get_data_by_variant(dlist, variant_id):
    """Return only the graphs whose adjacency variant matches ``variant_id``."""
    return [d for d in dlist if d.variant == variant_id]


# ---------------------------------------------------------------------------
# Dataset scaffold
# ---------------------------------------------------------------------------
def build_dataset():
    """Build the per-subject scaffold graphs and their labels.

    The node features and adjacency created here are placeholders; the real
    values are injected later from the SPM feature table. The label is derived
    from the subject id (ids starting with 3 or 4 are the patient group).
    """
    data_list, labels = [], []
    mat = scipy.io.loadmat(str(config.ONSETS_MAT))
    num_ids = mat["num_ids"]

    for i in range(len(num_ids)):
        subject_id = f"sub-{num_ids[i][0]}_Simon"

        adj = np.random.rand(NUM_REGIONS, NUM_REGIONS)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        adj[adj < 0.5] = 0
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)

        numeric_id_str = str(num_ids[i][0])
        label_val = 1 if numeric_id_str.startswith(("3", "4")) else 0
        y = torch.tensor([label_val], dtype=torch.long)

        for variant_id, gen in enumerate(FEATURE_GENERATORS):
            data = Data(x=gen(), edge_index=edge_index, y=y)
            data.variant = variant_id
            data.subject_id = subject_id
            data_list.append(data)
            labels.append(y.item())

    return data_list, torch.tensor(labels)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, loss_fn, graph_feat_mask, model_type, node_mask):
    """Run one training epoch over ``loader``."""
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        graph_feat = apply_graph_feature_mask(batch.graph_feat.view(-1, 4), graph_feat_mask)
        edge_weight = getattr(batch, "edge_weight", None)
        edge_attr = getattr(batch, "edge_attr", None)
        x_in = apply_node_feature_mask(batch.x, node_mask)

        if int(batch.edge_index.max()) >= x_in.size(0):
            raise RuntimeError(
                f"edge_index out of range (x_in={tuple(x_in.shape)}, "
                f"edge_index max={int(batch.edge_index.max())}, "
                f"subjects={getattr(batch, 'subject_id', None)})"
            )

        if model_type == "GCN":
            out = model(x_in, batch.edge_index, graph_feat, batch.batch, edge_weight)
        else:
            out = model(x_in, batch.edge_index, graph_feat, batch.batch, edge_attr=edge_attr)

        loss = loss_fn(out, batch.y.unsqueeze(1).float())
        loss.backward()
        optimizer.step()


def evaluate(model, loader, graph_feat_mask, model_type, node_mask,
             return_attn=False, return_pool_weights=False):
    """Evaluate ``model`` on ``loader`` and return AUC plus probabilities/labels.

    Optionally also returns the GAT attention weights or the attention-pooling
    weights (node-importance) together with their subject ids.
    """
    model.eval()
    all_probs, all_labels, all_attn = [], [], []
    all_node_alpha, all_alpha_subject_ids = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_in = apply_node_feature_mask(batch.x, node_mask)
            graph_feat = apply_graph_feature_mask(batch.graph_feat.view(-1, 4), graph_feat_mask)
            edge_weight = getattr(batch, "edge_weight", None)
            edge_attr = getattr(batch, "edge_attr", None)
            pool_weights = None

            if model_type == "GCN":
                out = model(x_in, batch.edge_index, graph_feat, batch.batch, edge_weight)
            elif return_pool_weights:
                out, pool_weights = model(
                    x_in, batch.edge_index, graph_feat, batch.batch,
                    edge_attr=edge_attr, return_pool_weights=True,
                )
            elif return_attn:
                out, attn_weights_list = model(
                    x_in, batch.edge_index, graph_feat, batch.batch,
                    edge_attr=edge_attr, return_attn=True,
                )
                all_attn.append(attn_weights_list)
            else:
                out = model(x_in, batch.edge_index, graph_feat, batch.batch, edge_attr=edge_attr)

            all_probs.append(torch.sigmoid(out).squeeze(1))
            all_labels.append(batch.y)

            if return_pool_weights and pool_weights is not None:
                all_node_alpha.append(pool_weights.detach().cpu().numpy())
                sid = batch.subject_id[0] if isinstance(batch.subject_id, (list, tuple)) else batch.subject_id
                all_alpha_subject_ids.append(str(sid))

    probs = torch.cat(all_probs).cpu().numpy()
    labels_true = torch.cat(all_labels).cpu().numpy()
    try:
        auc_val = roc_auc_score(labels_true, probs)
    except ValueError:
        auc_val = 0.5

    if return_pool_weights:
        return auc_val, probs, labels_true, all_node_alpha, all_alpha_subject_ids
    if return_attn and model_type == "GAT":
        return auc_val, probs, labels_true, all_attn
    return auc_val, probs, labels_true


# ---------------------------------------------------------------------------
# SPM results loading
# ---------------------------------------------------------------------------
def _spm_results_path(seed=None, outer_fold=None, inner_fold=None):
    """Resolve a fold-specific results.mat, falling back to the base file."""
    parts = [SPM_RESULTS_BASE]
    if seed is not None:
        parts.append(f"seed{seed}")
    if outer_fold is not None:
        parts.append(f"outer{outer_fold}")
    if inner_fold is not None:
        parts.append(f"inner{inner_fold}")
    fold_path = os.path.join(*parts, "results.mat")
    if os.path.isfile(fold_path):
        return fold_path
    return os.path.join(SPM_RESULTS_BASE, "results.mat")


def load_spm_results_mat(experiment_name, seed, outer_fold, inner_fold):
    """Load the inner-fold SPM results.mat (flag, T, peakTable)."""
    path = _spm_results_path(seed, outer_fold, inner_fold)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing SPM results.mat: {path}")
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False), path


def load_spm_results_trainval(experiment_name, seed, outer_fold):
    """Load the outer-fold (trainval) SPM results.mat."""
    path = _spm_results_path(seed, outer_fold)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing SPM trainval results.mat: {path}")
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False), path


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------
def nested_cv(data_list, labels, n_trials=30, outer_folds=OUTER_FOLDS, seed=42, eng=None):
    """Run subject-level nested CV for a single seed and return the result frames."""
    all_studies = []
    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    outer_metrics, outer_params_log, outer_Tr_log = [], [], []
    models_state_dicts = []
    Test_Data_outer, TrainVal_Data_outer = [], []
    pareto_log, chosen_log = [], []

    subject_ids = np.array([d.subject_id for d in data_list])
    unique_subjects = np.unique(subject_ids)
    subject_to_label = {
        subj: labels[np.where(subject_ids == subj)[0][0]] for subj in unique_subjects
    }
    subject_labels = np.array([subject_to_label[s] for s in unique_subjects])

    trainval_graphs_to_save = None  # set inside the loop, reused for the summary

    for outer_fold, (trainval_subj_idx, test_subj_idx) in enumerate(
        outer_skf.split(unique_subjects, subject_labels)
    ):
        print(f"\n[Outer Fold {outer_fold + 1}/{outer_folds}] (seed={seed})")
        start_time = time.time()

        trainval_subjects = unique_subjects[trainval_subj_idx]
        test_subjects = unique_subjects[test_subj_idx]
        trainval_data = [data_list[i] for i, s in enumerate(subject_ids) if s in trainval_subjects]
        test_data = [data_list[i] for i, s in enumerate(subject_ids) if s in test_subjects]

        def evaluate_params(params, graph_feat_mask, node_mask, trial=None, lam=0.0):
            """Inner-CV objective: mean (and std) validation AUC for ``params``."""
            filtered_data = get_data_by_variant(trainval_data, params["variant_id"])
            filtered_labels = torch.tensor([d.y.item() for d in filtered_data])
            inner_skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)

            fold_best_aucs, fold_epochs_ran, fold_best_epoch = [], [], []

            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                inner_skf.split(filtered_data, filtered_labels)
            ):
                train_data_inner = [filtered_data[i] for i in inner_train_idx]
                val_data_inner = [filtered_data[i] for i in inner_val_idx]
                train_subject_ids_inner = [g.subject_id for g in train_data_inner]

                if USE_PRECOMPUTED_SPM:
                    mat_contents, _ = load_spm_results_mat(
                        EXPERIMENT_NAME, seed, outer_fold, inner_fold
                    )
                else:
                    mat_contents = _run_matlab_spm(
                        eng, train_subject_ids_inner, params["sphere_radius"], seed,
                        outer_fold, inner_fold,
                    )

                flag = mat_contents.get("flag")
                if flag is not None and flag == 1:
                    print(f"Skipping inner fold {inner_fold} (empty peakTable).")
                    continue

                T = mat_contents.get("T")
                if T is None or (isinstance(T, np.ndarray) and T.size == 0):
                    print(f"Skipping inner fold {inner_fold} (empty T).")
                    continue

                T = sparsify_TZ(T, er_factor=params["er_factor"])
                Tr, _, _ = regress_out(T, train_subject_ids_inner)

                train_data_inner = update_graphs_from_Tr(Tr, train_data_inner)
                val_data_inner = update_graphs_from_Tr(Tr, val_data_inner)
                audit_graph_list(train_data_inner, tag=f"OUT{outer_fold+1} IN{inner_fold+1} TRAIN")
                audit_graph_list(val_data_inner, tag=f"OUT{outer_fold+1} IN{inner_fold+1} VAL")

                train_loader = DataLoader(train_data_inner, batch_size=params["batch"], shuffle=True)
                val_loader = DataLoader(val_data_inner, batch_size=params["batch"], shuffle=False)

                model_type = params["model_type"]
                model = build_model(
                    model_type, int(sum(node_mask)), params, int(sum(graph_feat_mask)),
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
                )
                loss_fn = torch.nn.BCEWithLogitsLoss()
                es = EarlyStopping()

                best_state, best_auc, best_ep, epochs_ran = None, -np.inf, 0, 0
                for epoch in range(EPOCHS):
                    train_epoch(model, train_loader, optimizer, loss_fn,
                                graph_feat_mask, model_type, node_mask)
                    auc_val, _, _ = evaluate(model, val_loader, graph_feat_mask, model_type, node_mask)
                    epochs_ran += 1
                    if auc_val > best_auc + es.min_delta:
                        best_auc, best_ep = auc_val, epoch + 1
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if es.step(auc_val):
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)

                fold_best_aucs.append(best_auc)
                fold_epochs_ran.append(epochs_ran)
                fold_best_epoch.append(best_ep)

            if len(fold_best_aucs) == 0:
                return None

            mean_auc = float(np.mean(fold_best_aucs))
            std_auc = float(np.std(fold_best_aucs, ddof=0))
            if trial is not None:
                trial.set_user_attr("inner_best_epoch", fold_best_epoch)
            return {"mean_auc": mean_auc, "std_auc": std_auc, "score": mean_auc - lam * std_auc}

        # ----- Hyperparameter search (Optuna, multi-objective Pareto) -----
        def objective(trial):
            valid_masks = [[0, 1, 1, 1]]                 # graph-feature masks
            node_masks = [[1, 1, 1, 1, 1, 0]]            # node-feature masks

            model_type = trial.suggest_categorical("model_type", ["GAT"])
            graph_feat_mask = valid_masks[trial.suggest_int("mask_id", 0, len(valid_masks) - 1)]
            node_mask = node_masks[trial.suggest_int("node_mask_id", 0, len(node_masks) - 1)]
            trial.set_user_attr("graph_feat_mask", graph_feat_mask)
            trial.set_user_attr("node_mask", node_mask)

            params = {
                "lr": trial.suggest_float("lr", 3e-3, 1.2e-2, log=True),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
                "dropout": trial.suggest_float("dropout", 0.3, 0.7),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                "variant_id": trial.suggest_categorical("variant_id", [1]),
                "num_layers": trial.suggest_categorical("num_layers", [2]),
                "sphere_radius": trial.suggest_categorical("sphere_radius", [10]),
                "batch": trial.suggest_categorical("batch", [16]),
                "model_type": model_type,
                "heads": trial.suggest_categorical("heads", [1]) if model_type == "GAT" else 1,
                "er_factor": trial.suggest_categorical("er_factor", [1]),
            }

            result = evaluate_params(params, graph_feat_mask, node_mask, trial=trial, lam=0.0)
            trial.set_user_attr("mean_auc", result["mean_auc"])
            trial.set_user_attr("std_auc", result["std_auc"])
            return result["mean_auc"], result["std_auc"]

        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.NopPruner(),
        )
        pbar = tqdm(total=n_trials, desc=f"[Optuna Fold {outer_fold+1} | seed={seed}]")
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda st, tr: pbar.update(1)])
        pbar.close()

        os.makedirs(OPTUNA_DIR, exist_ok=True)
        study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state")).to_csv(
            os.path.join(OPTUNA_DIR, f"trials_seed{seed}_outer{outer_fold+1}_trials{n_trials}.csv"),
            index=False,
        )

        # Choose from the Pareto front: best mean AUC among low-std solutions.
        pareto = study.best_trials
        feasible = [t for t in pareto if t.values[1] <= 0.05]
        chosen = max(feasible, key=lambda t: t.values[0]) if feasible \
            else min(pareto, key=lambda t: t.values[1])

        best_params = chosen.params
        best_graph_feat_mask = chosen.user_attrs["graph_feat_mask"]
        best_node_mask = chosen.user_attrs["node_mask"]
        inner_best_epochs = chosen.user_attrs.get("inner_best_epoch", None)
        epochs_outer = EPOCHS if not inner_best_epochs \
            else min(int(np.ceil(np.mean(inner_best_epochs))), EPOCHS)

        pareto_log.append([
            {"trial_number": t.number, "values": t.values, "params": t.params,
             "graph_feat_mask": t.user_attrs.get("graph_feat_mask"),
             "mean_auc": t.user_attrs.get("mean_auc"), "std_auc": t.user_attrs.get("std_auc")}
            for t in pareto
        ])
        chosen_log.append({
            "outer_fold": outer_fold + 1, "seed": seed, "chosen_trial_number": chosen.number,
            "chosen_values": chosen.values, "chosen_params": chosen.params,
            "chosen_graph_feat_mask": best_graph_feat_mask,
        })
        all_studies.append(study)
        outer_params_log.append(best_params)

        # ----- Final train / test on the outer fold -----------------------
        trainval_data_filtered = get_data_by_variant(trainval_data, best_params["variant_id"])
        test_data_filtered = get_data_by_variant(test_data, best_params["variant_id"])
        trainval_subject_ids = [g.subject_id for g in trainval_data_filtered]

        if USE_PRECOMPUTED_SPM:
            mat_contents, _ = load_spm_results_trainval(EXPERIMENT_NAME, seed, outer_fold)
        else:
            mat_contents = _run_matlab_spm(
                eng, trainval_subject_ids, best_params["sphere_radius"], seed, outer_fold, None
            )

        flag = mat_contents.get("flag")
        if flag is not None and flag == 1:
            print(f"Skipping outer fold {outer_fold} (empty peakTable).")
            continue
        T = mat_contents.get("T")
        if T is None or (isinstance(T, np.ndarray) and T.size == 0):
            print(f"Skipping outer fold {outer_fold} (empty T).")
            continue

        T = sparsify_TZ(T, er_factor=best_params["er_factor"])
        Tr, _, _ = regress_out(T, trainval_subject_ids)
        trainval_data_filtered = update_graphs_from_Tr(Tr, trainval_data_filtered)
        test_data_filtered = update_graphs_from_Tr(Tr, test_data_filtered)
        outer_Tr_log.append(Tr)

        train_loader = DataLoader(trainval_data_filtered, batch_size=best_params["batch"], shuffle=True)
        test_loader = DataLoader(test_data_filtered, batch_size=1, shuffle=False)

        model_type = best_params["model_type"]
        model = build_model(
            model_type, int(sum(best_node_mask)), best_params, int(sum(best_graph_feat_mask)),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs_outer):
            train_epoch(model, train_loader, optimizer, loss_fn,
                        best_graph_feat_mask, model_type, best_node_mask)

        if model_type == "GAT":
            _, probs, labels_true, node_alpha, node_alpha_subject_id = evaluate(
                model, test_loader, best_graph_feat_mask, model_type, best_node_mask,
                return_pool_weights=True,
            )
        else:
            _, probs, labels_true = evaluate(
                model, test_loader, best_graph_feat_mask, model_type, best_node_mask,
            )
            node_alpha, node_alpha_subject_id = None, None

        trainval_graphs_to_save = copy.deepcopy(trainval_data_filtered)
        test_graphs_to_save = copy.deepcopy(test_data_filtered)

        # Verify that the node-importance vectors line up with the graphs.
        if model_type == "GAT" and node_alpha is not None:
            for alpha_arr, subj_id in zip(node_alpha, node_alpha_subject_id):
                matching = [g for g in test_graphs_to_save if g.subject_id == subj_id]
                if matching and matching[0].x.shape[0] != len(alpha_arr):
                    raise ValueError(f"Graph-alpha mismatch for {subj_id}!")

        Test_Data_outer.append(test_graphs_to_save)
        TrainVal_Data_outer.append(trainval_graphs_to_save)

        pred_bin = (probs > 0.5).astype(int)
        outer_metrics.append({
            "seed": seed,
            "best_graph_feat_mask": best_graph_feat_mask,
            "best_node_mask": best_node_mask,
            "TrainVal_Data_outer": trainval_graphs_to_save,
            "auc": roc_auc_score(labels_true, probs),
            "balanced_accuracy": balanced_accuracy_score(labels_true, pred_bin),
            "sensitivity": recall_score(labels_true, pred_bin, pos_label=1),
            "specificity": recall_score(labels_true, pred_bin, pos_label=0),
            "Test_Data_outer": test_graphs_to_save,
            "fold": outer_fold + 1,
            "variant_id": best_params["variant_id"],
            "y_true": labels_true,
            "y_pred": probs,
            "node_alpha": node_alpha,
            "node_alpha_subject_id": node_alpha_subject_id,
            "epochs_outer": epochs_outer,
            "inner_best_epochs": inner_best_epochs,
        })
        models_state_dicts.append(model.state_dict())
        print(f"Elapsed time fold {outer_fold + 1}: {(time.time() - start_time) / 60:.2f} min")

    df_outer = pd.DataFrame(outer_metrics)
    df_outer["fold"] = list(range(1, len(df_outer) + 1))
    df_outer["params"] = outer_params_log
    df_outer["Tr"] = outer_Tr_log

    print("\n=== Final Nested CV Results ===")
    print(df_outer[["auc", "balanced_accuracy", "sensitivity", "specificity"]]
          .describe().loc[["mean", "std"]])

    # Save one package per seed (never overwriting across seeds).
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    torch.save(
        {
            "experiment": EXPERIMENT_NAME, "seed": seed, "outer_folds": outer_folds,
            "inner_folds": INNER_FOLDS, "n_trials": n_trials,
            "TrainVal_Data_outer": trainval_graphs_to_save, "Test_Data_outer": Test_Data_outer,
            "models": models_state_dicts, "df_outer": df_outer,
            "all_studies": all_studies, "pareto_log": pareto_log, "chosen_log": chosen_log,
        },
        os.path.join(
            BASE_RESULTS_DIR,
            f"all_results_{EXPERIMENT_NAME}_trials{n_trials}_outer{outer_folds}"
            f"_inner{INNER_FOLDS}_seed{seed}.pth",
        ),
    )
    df_outer.to_csv(os.path.join(BASE_RESULTS_DIR, f"df_outer_{EXPERIMENT_NAME}_seed{seed}.csv"),
                    index=False)
    return df_outer, all_studies


def _run_matlab_spm(eng, subject_ids, sphere_radius, seed, outer_fold, inner_fold):
    """Run the MATLAB SPM pipeline for a fold and load the resulting results.mat.

    Only used when ``USE_PRECOMPUTED_SPM`` is False; requires an active MATLAB
    engine ``eng`` with the repo's matlab/ directory on its path.
    """
    ids_filename = os.path.join(
        config.SPLIT_IDS_DIR,
        f"ids_seed{seed}_outer{outer_fold}"
        + (f"_inner{inner_fold}" if inner_fold is not None else "") + ".mat",
    )
    scipy.io.savemat(ids_filename, {"subject_ids": np.array(subject_ids, dtype=object)})
    eng.workspace["subject_ids_array"] = list(subject_ids)
    eng.workspace["sphere_radius"] = sphere_radius
    eng.eval("spm_2level_gat(subject_ids_array, sphere_radius)", nargout=0)
    return scipy.io.loadmat(
        os.path.join(SPM_RESULTS_BASE, "results.mat"), squeeze_me=True, struct_as_record=False
    )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def bootstrap_ci_over_seeds(seed_scores, n_boot=10000, ci=0.95, seed=42):
    """Cluster bootstrap confidence interval over per-seed scores."""
    seed_scores = np.asarray(seed_scores, dtype=float)
    if len(seed_scores) < 2:
        raise ValueError("Need at least 2 seeds to compute a bootstrap CI.")

    rng = np.random.default_rng(seed)
    n = len(seed_scores)
    boot_means = np.array([seed_scores[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])

    alpha = 1.0 - ci
    lo = np.quantile(boot_means, alpha / 2.0)
    hi = np.quantile(boot_means, 1.0 - alpha / 2.0)
    return seed_scores.mean(), (lo, hi), boot_means


def plot_seed_roc(df_outer, seed, save_dir=None):
    """Plot per-fold and mean ROC curves for a single seed."""
    import matplotlib.pyplot as plt

    tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 1000)
    plt.figure(figsize=(8, 6))
    for i, row in df_outer.iterrows():
        y_true, y_pred = np.asarray(row["y_true"]), np.asarray(row["y_pred"])
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        aucs.append(roc_auc_score(y_true, y_pred))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, lw=1.5, alpha=0.6, label=f"Fold {i+1} (AUC={aucs[-1]:.2f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color="blue", lw=2.5,
             label=f"Mean ROC (AUC={np.mean(aucs):.2f} +/- {np.std(aucs):.2f})")
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1), color="grey", alpha=0.3)
    plt.plot([0, 1], [0, 1], "--", color="red", lw=2)
    plt.title(f"ROC Curves - seed {seed}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"roc_seed{seed}.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Final model on the full discovery dataset
# ---------------------------------------------------------------------------
def zero_graph_features(data_list, graph_feat_mask):
    """Return deep-copied graphs with the masked behavioural features zeroed."""
    out = []
    mask = torch.tensor(graph_feat_mask, dtype=torch.bool)
    for g in data_list:
        gz = copy.deepcopy(g)
        gf = gz.graph_feat.clone().float()
        if gf.dim() == 1:
            gf[mask] = 0.0
        elif gf.dim() == 2:
            gf[:, mask] = 0.0
        else:
            raise ValueError(f"Unexpected graph_feat shape: {gf.shape}")
        gz.graph_feat = gf
        out.append(gz)
    return out


def train_final_model_full_dataset_precomputed(data_list, seed=42, epochs_final=200,
                                                save_path=None):
    """Train the final model (and a matched zero-behaviour control) on all subjects."""
    print("\n=== FINAL TRAIN ON FULL DATASET ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    final_params = {
        "model_type": "GAT", "hidden_dim": 128, "dropout": 0.54, "lr": 0.00825,
        "weight_decay": 0.00042, "batch": 16, "heads": 1, "variant_id": 1,
        "num_layers": 2, "sphere_radius": 10, "er_factor": 1,
    }
    graph_feat_mask = [0, 1, 1, 1]
    node_mask = [1, 1, 1, 1, 1, 0]

    full_data = get_data_by_variant(data_list, final_params["variant_id"])
    full_subject_ids = [g.subject_id for g in full_data]
    print(f"Using {len(full_data)} graphs from {len(set(full_subject_ids))} subjects.")

    mat_contents, results_path = load_spm_results_trainval(EXPERIMENT_NAME, seed, outer_fold=0)
    if mat_contents.get("flag") == 1:
        raise RuntimeError("Precomputed SPM results have flag==1 (invalid results).")
    T = mat_contents.get("T")
    if T is None or (isinstance(T, np.ndarray) and T.size == 0):
        raise RuntimeError("Precomputed SPM results contain empty T.")

    T = sparsify_TZ(T, er_factor=final_params["er_factor"])
    Tr, all_models, final_stats = regress_out(T, full_subject_ids)
    full_data_updated = update_graphs_from_Tr(Tr, full_data)
    audit_graph_list(full_data_updated, tag="FINAL TRAIN FULL DATA")

    in_channels = int(sum(node_mask))
    graph_feat_dim = int(sum(graph_feat_mask))

    def _train(loader, tag):
        model = build_model(final_params["model_type"], in_channels, final_params,
                            graph_feat_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=final_params["lr"],
                                     weight_decay=final_params["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs_final):
            train_epoch(model, loader, optimizer, loss_fn, graph_feat_mask,
                        final_params["model_type"], node_mask)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"[{tag}] Epoch {epoch + 1}/{epochs_final}")
        return model

    train_loader = DataLoader(full_data_updated, batch_size=final_params["batch"], shuffle=False)
    model = _train(train_loader, "FULL")

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(BASE_RESULTS_DIR, "final_model_full_dataset.pth")
    save_dict = {
        "experiment": EXPERIMENT_NAME, "seed": seed, "final_params": final_params,
        "graph_feat_mask": graph_feat_mask, "node_mask": node_mask,
        "model_state_dict": model.state_dict(), "Tr": Tr,
        "full_subject_ids": full_subject_ids, "spm_results_path": results_path,
        "epochs_final": epochs_final, "all_models": all_models, "final_stats": final_stats,
    }
    torch.save(save_dict, save_path)
    print(f"Final model saved: {save_path}")

    # Matched zero-behaviour control model.
    zero_loader = DataLoader(zero_graph_features(full_data_updated, graph_feat_mask),
                             batch_size=final_params["batch"], shuffle=False)
    zero_model = _train(zero_loader, "ZERO")

    zero_dir = os.path.join(config.RESULTS_ROOT, EXPERIMENT_NAME + "_zero_gate")
    os.makedirs(zero_dir, exist_ok=True)
    zero_save_path = os.path.join(zero_dir, "final_model_full_dataset_zero_gate.pth")
    zero_save_dict = {
        **save_dict,
        "experiment": EXPERIMENT_NAME + "_zero_gate",
        "model_state_dict": zero_model.state_dict(),
        "zero_behavior_training": True,
        "zeroed_graph_feat_mask_indices": [i for i, v in enumerate(graph_feat_mask) if v == 1],
    }
    torch.save(zero_save_dict, zero_save_path)
    print(f"Zero-behaviour model saved: {zero_save_path}")

    return model, save_dict, zero_model, zero_save_dict


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_all_seeds(n_trials=30, permutation_on=False, n_perm=500):
    """Run nested CV across all seeds and aggregate the results."""
    config.ensure_dirs()
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    data_list, labels = build_dataset()

    all_seed_summaries, all_seed_bestparams, all_seed_df_outer = [], [], {}

    for seed in SEEDS:
        df_outer, _ = nested_cv(data_list, labels, n_trials=n_trials,
                                outer_folds=OUTER_FOLDS, seed=seed)
        all_seed_df_outer[seed] = df_outer.copy()

        all_seed_summaries.append({
            "seed": seed,
            "mean_auc_outer": float(df_outer["auc"].mean()),
            "std_auc_outer": float(df_outer["auc"].std(ddof=0)),
            "mean_balanced_acc_outer": float(df_outer["balanced_accuracy"].mean()),
            "mean_sensitivity_outer": float(df_outer["sensitivity"].mean()),
            "mean_specificity_outer": float(df_outer["specificity"].mean()),
        })
        for _, row in df_outer.iterrows():
            p = row["params"]
            all_seed_bestparams.append({
                "seed": seed, "outer_fold": int(row["fold"]), "auc": float(row["auc"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "sensitivity": float(row["sensitivity"]),
                "specificity": float(row["specificity"]),
                "mask": row["best_graph_feat_mask"],
                **{k: p.get(k) for k in
                   ("hidden_dim", "dropout", "lr", "weight_decay", "batch", "heads", "variant_id")},
            })
        plot_seed_roc(df_outer, seed, save_dir=BASE_RESULTS_DIR)

    if permutation_on:
        print("\n=== PERMUTATION TESTS ===")
        for seed in SEEDS:
            real_auc, perm_aucs, pval = permutation_test_seed(
                seed=seed, df_outer_real=all_seed_df_outer[seed],
                data_list=data_list, labels_tensor=labels, outer_fold_num=OUTER_FOLDS,
                device=device, build_model=build_model, train_epoch=train_epoch,
                evaluate=evaluate, update_graphs_from_Tr=update_graphs_from_Tr,
                regress_out=regress_out, sparsify_TZ=sparsify_TZ,
                load_spm_results_trainval=load_spm_results_trainval,
                experiment_name=EXPERIMENT_NAME, n_perm=n_perm, rng_seed=42,
            )
            print(f"[Seed {seed}] real mean AUC = {real_auc:.4f} | permutation p = {pval:.4g}")
            np.save(os.path.join(BASE_RESULTS_DIR, f"perm_aucs_seed{seed}.npy"), perm_aucs)

    # Master summary tables.
    summary_df = pd.DataFrame(all_seed_summaries)
    bestparams_df = pd.DataFrame(all_seed_bestparams)
    summary_df.to_csv(os.path.join(BASE_RESULTS_DIR, "summary_across_seeds.csv"), index=False)
    bestparams_df.to_csv(
        os.path.join(BASE_RESULTS_DIR, "best_params_per_fold_across_seeds.csv"), index=False
    )

    print("\n=== MASTER SUMMARY ===")
    print(summary_df)
    grand_mean = summary_df["mean_auc_outer"].mean()
    print(f"\nGrand mean AUC (across seeds): {grand_mean:.4f} "
          f"+/- {summary_df['mean_auc_outer'].std(ddof=0):.4f}")

    seed_auc = [d["mean_auc_outer"] for d in all_seed_summaries]
    if len(seed_auc) >= 2:
        mean_auc, (ci_lo, ci_hi), _ = bootstrap_ci_over_seeds(seed_auc, n_boot=10000, ci=0.95)
        print(f"Mean AUC over seeds: {mean_auc:.4f}")
        print(f"Bootstrap 95% CI (over seeds): [{ci_lo:.4f}, {ci_hi:.4f}]")

    return data_list, summary_df


if __name__ == "__main__":
    # Full discovery run across all seeds, then a final model on all subjects.
    data_list, _ = run_all_seeds(n_trials=30, permutation_on=False)
    train_final_model_full_dataset_precomputed(data_list, seed=100, epochs_final=200)
