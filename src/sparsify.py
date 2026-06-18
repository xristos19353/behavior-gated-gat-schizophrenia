"""Erdos-Renyi adjacency sparsification (NumPy port of sparsify_adjacency.m).

The functions here mirror the MATLAB ``sparsify_adjacency`` exactly, including
its column-major (Fortran) linear indexing and round-half-up behaviour, so that
the Python and MATLAB pipelines produce identical sparsified graphs.
"""

from __future__ import annotations

import numpy as np


def matlab_round_positive(x: float) -> int:
    """Round half up, matching MATLAB ``round`` for positive numbers."""
    return int(np.floor(x + 0.5))


def sparsify_adjacency_er_matlab(A, binary=True, er_factor=1.0):
    """MATLAB-faithful Erdos-Renyi sparsification of a symmetric matrix.

    Parameters
    ----------
    A : array_like
        Square symmetric matrix.
    binary : bool
        If True the output is binary (0/1); otherwise weights are kept.
    er_factor : float
        Multiplier on the Erdos-Renyi threshold ``log(N)/N``.
    """
    A = np.array(A, dtype=float)

    # Force symmetry and zero the diagonal.
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    N = A.shape[0]
    p_min = er_factor * np.log(N) / N

    num_possible_edges = N * (N - 1) / 2
    num_edges_to_keep = matlab_round_positive(p_min * num_possible_edges)

    # Upper-triangle linear indices, column-major like MATLAB's find(triu(...)).
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    upper_idx = np.flatnonzero(triu_mask.flatten(order="F"))

    A_flatF = A.flatten(order="F")
    edge_weights = A_flatF[upper_idx]

    # Stable descending sort by absolute weight (matches MATLAB tie handling).
    sorted_idx = np.argsort(-np.abs(edge_weights), kind="mergesort")
    selected_idx = upper_idx[sorted_idx[:num_edges_to_keep]]

    valid_nodes = np.any((A != 0) & (~np.isnan(A)), axis=1)
    invalid_idx = np.where(~valid_nodes)[0]

    A_sparse_flatF = np.zeros(N * N, dtype=float)

    if binary:
        A_sparse_flatF[selected_idx] = 1.0
        A_sparse = A_sparse_flatF.reshape((N, N), order="F")
        if invalid_idx.size > 0:
            A_sparse[invalid_idx, :] = 0.0
            A_sparse[:, invalid_idx] = 0.0
    else:
        A_sparse_flatF[selected_idx] = A_flatF[selected_idx]
        A_sparse = A_sparse_flatF.reshape((N, N), order="F")

    return A_sparse + A_sparse.T


def sparsify_TZ(T, er_factor=1.0, zw_er_factor=24.1,
                z_field="Z", zb_field="Zbinary", zw_field="Zweighted"):
    """Recompute ``T.Zbinary`` / ``T.Zweighted`` from the full matrices ``T.Z``.

    For every subject the dense Fisher-z matrix in ``T.Z`` is sparsified into a
    binary adjacency (with ``er_factor``) and a weighted adjacency (with
    ``zw_er_factor``), overwriting the existing fields in place.

    Parameters
    ----------
    T : object
        Object exposing list-like attributes ``Z``, ``Zbinary`` and
        ``Zweighted`` (one entry per subject), each either a struct holding the
        named field or a raw matrix.
    er_factor : float
        Erdos-Renyi factor for the binary adjacency.
    zw_er_factor : float
        Erdos-Renyi factor for the weighted adjacency (denser by design).
    """
    Z_list = getattr(T, "Z", None)
    Zb_list = getattr(T, "Zbinary", None)
    Zw_list = getattr(T, "Zweighted", None)

    if Z_list is None or Zb_list is None:
        raise AttributeError("T must have both attributes: T.Z and T.Zbinary")

    nZ = len(Z_list)
    if nZ != len(Zb_list):
        raise ValueError(f"Length mismatch: len(T.Z)={nZ} vs len(T.Zbinary)={len(Zb_list)}")

    # Detect whether T.Zbinary holds structs (with a field) or raw matrices.
    zb_is_structs = bool(nZ) and hasattr(Zb_list[0], "__dict__") and hasattr(Zb_list[0], zb_field)

    for i in range(nZ):
        subjZ = Z_list[i]
        if not hasattr(subjZ, z_field):
            raise AttributeError(f"T.Z[{i}] has no field '{z_field}'")

        A_full = np.array(getattr(subjZ, z_field), dtype=float)

        A_bin = sparsify_adjacency_er_matlab(A_full, binary=True, er_factor=er_factor)
        if zb_is_structs:
            setattr(Zb_list[i], zb_field, A_bin)
        else:
            Zb_list[i] = A_bin

        A_weighted = sparsify_adjacency_er_matlab(A_full, binary=False, er_factor=zw_er_factor)
        if hasattr(Zw_list[i], "__dict__"):
            setattr(Zw_list[i], zw_field, A_weighted)
        else:
            Zw_list[i] = A_weighted

    setattr(T, "Zbinary", Zb_list)
    setattr(T, "Zweighted", Zw_list)
    return T
