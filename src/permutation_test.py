"""Subject-level label-permutation test for the nested-CV pipeline.

For each seed, the real outer-fold hyperparameters are held fixed while the
training labels are permuted at the subject level (so all graphs of a subject
keep a single, shuffled label). Re-running the outer folds many times yields a
null distribution of mean AUC, from which a one-sided p-value is computed.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def run_one_outer_fold_fixed(
    *,
    seed,
    outer_fold,
    best_params,
    best_graph_feat_mask,
    best_node_mask,
    epochs_outer,
    trainval_data,
    test_data,
    trainval_subject_ids,
    device,
    build_model,
    train_epoch,
    evaluate,
    update_graphs_from_Tr,
    regress_out,
    sparsify_TZ,
    load_spm_results_trainval,
    experiment_name,
):
    """Train and evaluate one outer fold with fixed hyperparameters.

    Returns the test AUC, or ``None`` if the fold's SPM results are invalid.
    """
    mat_contents, _ = load_spm_results_trainval(
        experiment_name=experiment_name, seed=seed, outer_fold=outer_fold,
    )

    flag = mat_contents.get("flag")
    if flag is not None and flag == 1:
        return None

    T = mat_contents.get("T")
    if T is None or (isinstance(T, np.ndarray) and T.size == 0):
        return None

    T = sparsify_TZ(T, er_factor=best_params["er_factor"])
    Tr, _, _ = regress_out(T, trainval_subject_ids)

    # Deep-copy so the original graphs are left untouched.
    trainval_data_upd = update_graphs_from_Tr(Tr, deepcopy(trainval_data))
    test_data_upd = update_graphs_from_Tr(Tr, deepcopy(test_data))

    variant_id = best_params["variant_id"]
    trainval_data_upd = [d for d in trainval_data_upd if getattr(d, "variant", None) == variant_id]
    test_data_upd = [d for d in test_data_upd if getattr(d, "variant", None) == variant_id]

    if len(trainval_data_upd) == 0 or len(test_data_upd) == 0:
        return None

    train_loader = DataLoader(trainval_data_upd, batch_size=best_params["batch"], shuffle=True)
    test_loader = DataLoader(test_data_upd, batch_size=1, shuffle=False)

    graph_feat_dim = int(sum(best_graph_feat_mask))
    in_channels = int(sum(best_node_mask))
    model_type = best_params["model_type"]

    model = build_model(model_type, in_channels, best_params, graph_feat_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs_outer):
        train_epoch(model, train_loader, optimizer, loss_fn,
                    best_graph_feat_mask, model_type, best_node_mask)

    _, probs, labels_true = evaluate(
        model, test_loader, best_graph_feat_mask, model_type, best_node_mask
    )

    try:
        return float(roc_auc_score(labels_true, probs))
    except Exception:
        return 0.5


def permutation_test_seed(
    *,
    seed,
    df_outer_real,
    data_list,
    labels_tensor,
    outer_fold_num,
    device,
    build_model,
    train_epoch,
    evaluate,
    update_graphs_from_Tr,
    regress_out,
    sparsify_TZ,
    load_spm_results_trainval,
    experiment_name,
    n_perm=200,
    rng_seed=42,
):
    """Run the permutation test for one seed.

    Returns
    -------
    real_mean_auc : float
        Mean AUC of the unpermuted run.
    perm_mean_aucs : np.ndarray
        Mean AUC for each permutation.
    p_value : float
        One-sided permutation p-value.
    """
    rng = np.random.default_rng(rng_seed)

    subject_ids = np.array([d.subject_id for d in data_list])
    unique_subjects = np.unique(subject_ids)

    labels_np = labels_tensor.cpu().numpy() if hasattr(labels_tensor, "cpu") else np.asarray(labels_tensor)
    subject_to_label = {
        subj: labels_np[np.where(subject_ids == subj)[0][0]] for subj in unique_subjects
    }
    subject_labels = np.array([subject_to_label[s] for s in unique_subjects], dtype=int)

    # Reproduce exactly the same outer split used in the real run.
    outer_skf = StratifiedKFold(n_splits=outer_fold_num, shuffle=True, random_state=seed)

    real_mean_auc = float(df_outer_real["auc"].mean())
    perm_mean_aucs = np.zeros(n_perm, dtype=float)

    for p in tqdm(range(n_perm), desc=f"[Permutation seed {seed}]"):
        perm_fold_aucs = []

        for outer_fold, (trainval_subj_idx, test_subj_idx) in enumerate(
            outer_skf.split(unique_subjects, subject_labels)
        ):
            row = df_outer_real[df_outer_real["fold"] == (outer_fold + 1)].iloc[0]
            best_params = row["params"]
            best_graph_feat_mask = row["best_graph_feat_mask"]
            best_node_mask = row["best_node_mask"]
            epochs_outer = int(row["epochs_outer"]) if "epochs_outer" in row else 200

            trainval_subjects = unique_subjects[trainval_subj_idx]
            test_subjects = unique_subjects[test_subj_idx]

            trainval_idx = [i for i, s in enumerate(subject_ids) if s in trainval_subjects]
            test_idx = [i for i, s in enumerate(subject_ids) if s in test_subjects]

            trainval_data = [deepcopy(data_list[i]) for i in trainval_idx]
            test_data = [deepcopy(data_list[i]) for i in test_idx]

            # Permute labels at the subject level on the training data only.
            trainval_subjects_list = [g.subject_id for g in trainval_data]
            unique_trainval = np.unique(trainval_subjects_list)

            subj_to_orig_label = {}
            for subj in unique_trainval:
                idx = trainval_subjects_list.index(subj)
                subj_to_orig_label[subj] = trainval_data[idx].y.item()

            perm_subjects = rng.permutation(unique_trainval)
            subj_to_perm_label = {
                orig_subj: subj_to_orig_label[perm_subj]
                for orig_subj, perm_subj in zip(unique_trainval, perm_subjects)
            }
            for d in trainval_data:
                d.y = torch.tensor([int(subj_to_perm_label[d.subject_id])], dtype=torch.long)

            trainval_subject_ids = [g.subject_id for g in trainval_data]

            fold_auc = run_one_outer_fold_fixed(
                seed=seed,
                outer_fold=outer_fold,
                best_params=best_params,
                best_graph_feat_mask=best_graph_feat_mask,
                best_node_mask=best_node_mask,
                epochs_outer=epochs_outer,
                trainval_data=trainval_data,
                test_data=test_data,
                trainval_subject_ids=trainval_subject_ids,
                device=device,
                build_model=build_model,
                train_epoch=train_epoch,
                evaluate=evaluate,
                update_graphs_from_Tr=update_graphs_from_Tr,
                regress_out=regress_out,
                sparsify_TZ=sparsify_TZ,
                load_spm_results_trainval=load_spm_results_trainval,
                experiment_name=experiment_name,
            )

            if fold_auc is not None:
                perm_fold_aucs.append(fold_auc)

        perm_mean_aucs[p] = float(np.mean(perm_fold_aucs)) if perm_fold_aucs else 0.5

    p_value = (np.sum(perm_mean_aucs >= real_mean_auc) + 1) / (n_perm + 1)
    return real_mean_auc, perm_mean_aucs, float(p_value)
