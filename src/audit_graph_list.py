"""Sanity-check utilities for lists of PyTorch Geometric graphs."""

from __future__ import annotations


def audit_graph_list(graphs, tag="", max_print=3):
    """Print a short integrity report for a list of PyG ``Data`` graphs.

    Parameters
    ----------
    graphs : list
        List of PyG ``Data`` objects.
    tag : str
        Label identifying the pipeline stage (e.g. which fold).
    max_print : int
        Number of graphs to print in detail; the rest are summarised. Any graph
        whose ``edge_index`` points outside its node set is always printed.
    """
    if len(graphs) == 0:
        print(f"[AUDIT {tag}] EMPTY graph list")
        return

    bad = 0
    xs = []
    es = []

    print(f"\n[AUDIT {tag}] n_graphs={len(graphs)}")

    for i, g in enumerate(graphs):
        x = getattr(g, "x", None)
        ei = getattr(g, "edge_index", None)

        subj = getattr(g, "subject_id", "NA")
        var = getattr(g, "variant", "NA")

        n_nodes = int(x.size(0)) if x is not None else -1
        n_feat = int(x.size(1)) if x is not None and x.dim() == 2 else -1
        n_edges = int(ei.size(1)) if ei is not None else -1

        ei_max = int(ei.max().item()) if (ei is not None and ei.numel() > 0) else -1
        ei_min = int(ei.min().item()) if (ei is not None and ei.numel() > 0) else -1

        is_bad = (n_nodes >= 0) and (ei_max >= n_nodes)
        if is_bad:
            bad += 1

        xs.append((n_nodes, n_feat))
        es.append(n_edges)

        if i < max_print or is_bad:
            print(
                f"  - [{i}] {subj} var={var} | x={tuple(x.shape)} "
                f"| edges={n_edges} | ei_min={ei_min} ei_max={ei_max} "
                f"{'BAD' if is_bad else 'OK'}"
            )

    uniq_x = sorted(set(xs))
    print(f"[AUDIT {tag}] unique x-shapes: {uniq_x[:10]}{' ...' if len(uniq_x) > 10 else ''}")
    print(f"[AUDIT {tag}] edges: min={min(es)} mean={sum(es) / len(es):.1f} max={max(es)}")
    print(f"[AUDIT {tag}] BAD graphs (edge_index out of range): {bad}/{len(graphs)}\n")
