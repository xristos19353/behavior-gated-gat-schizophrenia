"""Populate PyG graphs with node features, edges and graph-level behaviour.

Given the residualised feature table ``Tr`` (produced by ``regress_out``), this
module fills each subject's graph with:

* node features: WMV, GMV, EffectSize, ALFF, ReHo, DC (one column each),
* an adjacency variant selected per graph (``Z``, ``Zbinary`` or ``Zweighted``),
* graph-level behavioural features (RTCV, RTSD, sigma, tau).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.io.matlab.mio5_params import mat_struct


def extract_subject_ids_2(subject_raw):
    """Normalise MATLAB-loaded subject identifiers into plain strings."""
    subject_ids = []
    for entry in subject_raw:
        if isinstance(entry, str):
            subject_ids.append(entry)
        elif isinstance(entry, tuple) and len(entry) == 4:
            try:
                subject_ids.append("".join(chr(c) for c in entry[3] if c != 0))
            except Exception:
                subject_ids.append("UNKNOWN")
        else:
            subject_ids.append(str(entry))
    return subject_ids


def get_numeric_fields(struct):
    """Return the names of numeric (or array) fields of a MATLAB struct."""
    if isinstance(struct, mat_struct):
        return [
            key for key in struct.__dict__.keys()
            if isinstance(getattr(struct, key), (int, float, np.integer, np.floating, np.ndarray))
            and np.size(getattr(struct, key)) > 0
        ]
    elif isinstance(struct, np.ndarray):
        return ["__self__"]
    return []


def update_graphs_from_Tr(Tr, data_list):
    """Fill each graph in ``data_list`` with features/edges from ``Tr``.

    The graph attribute ``variant`` (0: Z, 1: Zbinary, 2: Zweighted) selects the
    adjacency matrix. Graphs whose subject is missing from ``Tr`` are skipped.
    Returns the same ``data_list`` (modified in place).
    """
    subject_ids = extract_subject_ids_2(Tr.Subject)
    subj_to_index = {sid: i for i, sid in enumerate(subject_ids)}

    feature_names = ["WMV", "GMV", "EffectSize", "ALFF", "ReHo", "DC"]
    adj_variants = ["Z", "Zbinary", "Zweighted"]

    updated = 0

    def extract_scalar(v):
        if isinstance(v, np.ndarray):
            return float(v.squeeze())
        return float(v)

    for g in data_list:
        subj_id = g.subject_id
        variant = g.variant   # 0: Z, 1: Zbinary, 2: Zweighted

        if subj_id not in subj_to_index:
            print(f"Subject {subj_id} not found in Tr.Subject")
            continue

        idx = subj_to_index[subj_id]

        try:
            features = []
            n_nodes = None

            for fname in feature_names:
                struct_raw = getattr(Tr, fname)[idx]
                if isinstance(struct_raw, np.ndarray) and struct_raw.size == 1:
                    struct = struct_raw.item()
                else:
                    struct = struct_raw

                if not hasattr(struct, "__dict__"):
                    raise ValueError(f"{fname} for {subj_id} is not a struct-like object")

                numeric_fields = []
                for key in struct.__dict__.keys():
                    if key.startswith("ROI_"):
                        val = getattr(struct, key)
                        if isinstance(val, (int, float, np.number)) or (
                            isinstance(val, np.ndarray) and val.size == 1
                        ):
                            numeric_fields.append(key)

                if len(numeric_fields) == 0:
                    raise ValueError(f"No usable ROI fields in {fname} for {subj_id}")

                sorted_fields = sorted(numeric_fields, key=lambda k: int(k.split("_")[1]))

                # Map ROI name -> node index (stored on the graph for explainers).
                g.roi_to_node = {roi_name: i for i, roi_name in enumerate(sorted_fields)}

                val = np.array(
                    [extract_scalar(getattr(struct, field)) for field in sorted_fields],
                    dtype=np.float32,
                )

                if n_nodes is None:
                    n_nodes = val.shape[0]
                elif val.shape[0] != n_nodes:
                    raise ValueError(
                        f"Mismatched ROI count in {fname} for {subj_id} "
                        f"(expected {n_nodes}, got {val.shape[0]})"
                    )

                features.append(torch.tensor(val).unsqueeze(1))  # (n_nodes, 1)

            g.x = torch.cat(features, dim=1)  # (n_nodes, num_features)

            # --- Adjacency matrix -------------------------------------------
            adj_name = adj_variants[variant]
            adj_struct = getattr(Tr, adj_name)[idx]

            if isinstance(adj_struct, mat_struct):
                adj_matrix = getattr(adj_struct, adj_name)
            elif isinstance(adj_struct, np.ndarray):
                adj_matrix = adj_struct
            else:
                raise ValueError(f"Unknown type for {adj_name} of {subj_id}: {type(adj_struct)}")

            adj_matrix = np.array(adj_matrix, dtype=np.float32)
            if adj_matrix.ndim == 3 and adj_matrix.shape[0] == 1:
                adj_matrix = adj_matrix[0]
            if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
                raise ValueError(f"Invalid adjacency shape {adj_matrix.shape} for {subj_id}")

            rows, cols = np.nonzero(adj_matrix)
            g.edge_index = torch.from_numpy(np.vstack((rows, cols))).long()

            if variant == 1:  # binary adjacency carries no edge weights
                g.edge_weight = None
            else:
                weights = torch.tensor(adj_matrix[rows, cols], dtype=torch.float32)
                g.edge_weight = weights
                g.edge_attr = weights.unsqueeze(1)

            # --- Graph-level behavioural features ---------------------------
            behavior_struct = Tr.Behavior[idx]
            if isinstance(behavior_struct, np.ndarray) and behavior_struct.size == 1:
                behavior_struct = behavior_struct.item()
            if not hasattr(behavior_struct, "__dict__"):
                raise ValueError(f"Invalid Behavior struct for {subj_id}")

            graph_feat_keys = ["RTCV", "RTSD", "sigma", "tau"]
            graph_feats = []
            for key in graph_feat_keys:
                val = getattr(behavior_struct, key, None)
                if val is None:
                    raise ValueError(f"Missing {key} in Tr.Behavior for subject {subj_id}")
                if isinstance(val, np.ndarray):
                    val = float(val.squeeze())
                graph_feats.append(val)

            g.graph_feat = torch.tensor(graph_feats, dtype=torch.float32)  # (4,)
            updated += 1

        except Exception as e:
            print(f"Error processing graph for subject {subj_id}, variant {variant}: {e}")

    print(f"Updated {updated} graphs with node features + adjacency from Tr.")
    return data_list
