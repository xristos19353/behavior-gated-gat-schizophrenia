"""Central configuration for the Python (GNN) side of the pipeline.

All machine-specific paths live here. Edit the values in the ``PATHS`` block
below (or override them with environment variables) so that none of the other
modules contain hard-coded absolute paths.

Every path can be overridden at run time with an environment variable of the
same name, e.g.::

    export GNN_DATA_ROOT=/data/my_study
    export GNN_RESULTS_ROOT=/data/my_study/results

Nothing here reads or writes data on import; the constants are simply resolved
once and reused by the rest of the code base.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    """Return ``Path`` from environment variable ``name`` or ``default``."""
    return Path(os.environ.get(name, default)).expanduser()


# ---------------------------------------------------------------------------
# EDIT THESE FOR YOUR MACHINE
# ---------------------------------------------------------------------------
# Root of the study data (BOLD signals, behavioural files, SPM outputs, ...).
DATA_ROOT = _env_path("GNN_DATA_ROOT", "/path/to/Functional_connectivity_study")

# Root where all model outputs / result packages are written.
RESULTS_ROOT = _env_path(
    "GNN_RESULTS_ROOT",
    str(DATA_ROOT / "GCN_output_files" / "results"),
)

# Directory that contains the MATLAB pipeline (added to the MATLAB path when
# SPM results are computed on the fly instead of loaded from disk).
MATLAB_CODE_DIR = _env_path(
    "GNN_MATLAB_CODE_DIR",
    str(Path(__file__).resolve().parent.parent / "matlab"),
)

# MATLAB ``bin`` directory (only needed when USE_PRECOMPUTED_SPM is False).
MATLAB_BIN_DIR = _env_path("GNN_MATLAB_BIN_DIR", "/Applications/MATLAB_R2024b.app/bin")

# ---------------------------------------------------------------------------
# DERIVED PATHS (usually no need to edit)
# ---------------------------------------------------------------------------
# Behavioural onset file used to enumerate subjects when building graphs.
ONSETS_MAT = _env_path(
    "GNN_ONSETS_MAT",
    str(DATA_ROOT / "behaviourals" / "onsets.mat"),
)

# Where Python writes the train/trainval subject-id lists consumed by MATLAB.
SPLIT_IDS_DIR = _env_path(
    "GNN_SPLIT_IDS_DIR",
    str(DATA_ROOT / "GCN_output_files"),
)

# Location of the SPM ``results.mat`` produced by the MATLAB pipeline. When
# USE_PRECOMPUTED_SPM is True this file is loaded directly.
SPM_RESULTS_BASE = _env_path("GNN_SPM_RESULTS_BASE", str(RESULTS_ROOT))


# ---------------------------------------------------------------------------
# EXTERNAL REPLICATION COHORT (only used by external_validation.py)
# ---------------------------------------------------------------------------
# Root of the external replication cohort data.
EXTERNAL_DATA_ROOT = _env_path("GNN_EXTERNAL_DATA_ROOT", str(DATA_ROOT / "external"))

# SPM feature table (results.mat) and behavioural onsets for the external cohort.
EXTERNAL_SPM_MAT = _env_path("GNN_EXTERNAL_SPM_MAT", str(EXTERNAL_DATA_ROOT / "results.mat"))
EXTERNAL_ONSETS_MAT = _env_path("GNN_EXTERNAL_ONSETS_MAT", str(EXTERNAL_DATA_ROOT / "onsets.mat"))

# Where external-validation outputs (CSVs, ROC, XAI packages) are written.
EXTERNAL_RESULTS_DIR = _env_path(
    "GNN_EXTERNAL_RESULTS_DIR", str(RESULTS_ROOT / "external_replication")
)

# Frozen discovery checkpoints evaluated on the external cohort. Defaults match
# the files written by train_gat.train_final_model_full_dataset_precomputed.
GATE_MODEL_PATH = _env_path(
    "GNN_GATE_MODEL_PATH",
    str(RESULTS_ROOT / "GAT_gate_only" / "final_model_full_dataset.pth"),
)
ZERO_MODEL_PATH = _env_path(
    "GNN_ZERO_MODEL_PATH",
    str(RESULTS_ROOT / "GAT_gate_only_zero_gate" / "final_model_full_dataset_zero_gate.pth"),
)


# ---------------------------------------------------------------------------
# EXPLAINABILITY (xAI) — used by src/xai/*
# ---------------------------------------------------------------------------
# Per-seed discovery result packages (all_results_*_seed*.pth) for the GATE
# model and its matched zero-behaviour control.
GATE_RESULTS_DIR = _env_path("GNN_GATE_RESULTS_DIR", str(RESULTS_ROOT / "GAT_gate_only"))
ZERO_RESULTS_DIR = _env_path("GNN_ZERO_RESULTS_DIR", str(RESULTS_ROOT / "GAT_gate_only_zero_gate"))

# Per-seed external XAI packages written by external_validation.py.
EXTERNAL_GATE_RESULTS_DIR = _env_path(
    "GNN_EXTERNAL_GATE_RESULTS_DIR", str(EXTERNAL_RESULTS_DIR / "GAT_gate_only"))
EXTERNAL_ZERO_RESULTS_DIR = _env_path(
    "GNN_EXTERNAL_ZERO_RESULTS_DIR", str(EXTERNAL_RESULTS_DIR / "GAT_ZERO_gate"))

# Root for explainability outputs (SubgraphX, IG, biomarker tests, ...).
XAI_RESULTS_DIR = _env_path("GNN_XAI_RESULTS_DIR", str(RESULTS_ROOT / "xai"))

# Random seeds used across the discovery runs (one result package per seed).
SEEDS = [100, 101, 102, 103, 104, 105, 106, 107]

# Atlas size used by the structure-aware xAI (full parcellation node count).
FULL_ATLAS_SIZE = 114


def ensure_dirs() -> None:
    """Create the output directories that the pipeline writes to."""
    for p in (RESULTS_ROOT, SPLIT_IDS_DIR):
        Path(p).mkdir(parents=True, exist_ok=True)
