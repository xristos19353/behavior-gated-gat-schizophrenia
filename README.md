# Behaviour-Gated Graph Neural Networks for fMRI-Based Classification

This repository contains the analysis pipeline used to produce the results in
our paper *"<paper title>"* (`<authors>`, `<venue / year>`). It combines a
neuroimaging feature-extraction stage in **MATLAB / SPM12 / MarsBaR** with a
**graph neural network** stage in **Python / PyTorch Geometric**.

Each subject is represented as a graph whose nodes are regions of interest
(ROIs) and whose edges encode resting/task functional connectivity. Node
features combine task effect size, grey- and white-matter density, ALFF, ReHo
and degree centrality; graph-level behavioural features (reaction-time
variability measures) modulate a learned attention-pooling gate in a Graph
Attention Network (GAT). The pooling weights double as node-importance
explanations.

> **Status / scope.** This release covers the full classification pipeline
> (feature extraction → graph construction → nested cross-validation →
> significance testing → final model). Explainability / post-hoc analysis code
> is maintained separately.

---

## Table of contents

- [Method overview](#method-overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Running the pipeline](#running-the-pipeline)
- [Reproducibility notes](#reproducibility-notes)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## Method overview

The pipeline has two halves that communicate through a `results.mat` file.

**1. MATLAB feature extraction (`matlab/`)**

| Stage | Script | Output |
|-------|--------|--------|
| Cache preprocessed BOLD | `preprocessing/bold_signal_creation.m` | per-subject `*.mat` |
| Voxel-wise ALFF | `preprocessing/alff_calculation.m` | `ALFF_voxel_z` |
| Voxel-wise ReHo | `preprocessing/reho_calculation.m` | `reho_z` |
| Modulated tissue density | `preprocessing/gmv_wmv_calc.m` | resliced GMV/WMV maps |
| Second-level analysis & feature table | `spm/spm_2level_gat.m` | `results.mat` (`flag`, `peakTable`, `T`) |

`spm_2level_gat.m` runs a second-level two-sample t-test (patients vs. controls)
on the **training subjects only** with age, sex and education as covariates,
detects cluster peaks, builds spherical ROIs, refines them against an anatomical
parcellation, extracts subject-specific peak ROIs (MarsBaR), and finally
computes per-subject connectivity (`adjacency_matrix_calc.m`), effect sizes and
tissue density (`roi_effectsize.m`). Defining ROIs from the training split only
keeps the cross-validation free of test-set leakage.

**2. Python GNN training (`src/`)**

`train_gat.py` runs subject-level **nested cross-validation**:

- inner folds + **Optuna** (multi-objective: maximise mean AUC, minimise its
  variance) select hyperparameters;
- for each fold the SPM `results.mat` is loaded, connectivity is sparsified with
  an Erdős–Rényi rule (`sparsify.py`) and demographics are regressed out of every
  feature (`regress_out.py`) before the features are written into the graphs
  (`build_graphs_from_tr.py`);
- the outer fold is retrained and evaluated on held-out subjects.

Results are aggregated across random seeds (summary tables, ROC curves,
bootstrap CI over seeds), an optional **subject-level permutation test**
(`permutation_test.py`) estimates significance, and a final model (plus a matched
zero-behaviour control) is trained on the full sample.

**3. External replication (`src/external_validation.py`)**

`external_validation.py` evaluates the two frozen discovery checkpoints on an
independent replication cohort:

- the **GATE** model (trained with real behavioural features) and the
  **ZERO-GATE** model (a separately trained model whose behavioural features
  were zeroed during training) are loaded and evaluated — the GATE with real and
  the ZERO-GATE with zeroed behavioural inputs;
- the external feature table is standardised and residualised with the
  **discovery cohort's frozen statistics and nuisance models**, so no external
  statistics leak into preprocessing;
- it reports per-model metrics with bootstrap CIs and permutation p-values, a
  paired GATE-minus-ZERO bootstrap, a ROC overlay, the prevalence of the
  discovery consensus edges in the external cohort, and XAI-compatible packages.

**4. Explainability (`src/xai/`, `matlab/xai/`)**

Run after the discovery (and, for the external variants, external) result
packages exist:

- **graph_brain** - *what* sub-network drives the prediction: `subgraphx_gat.py`
  extracts a SubgraphX node set per subject; `subgraphx_statistical_inference.py`
  aggregates these into consensus nodes/edges with sign-flip + BH-FDR statistics;
  `post_hoc_biomarker_test.py` validates the consensus nodes/edges by keep-only
  (sufficiency) and remove-only (necessity) tests against random sets;
  `features_xai.py` attributes each consensus node to its node features
  (Integrated Gradients) and tests their necessity/sufficiency.
- **cognitive** - *how behaviour modulates the model*:
  `integrated_gradients_entropy.py` (discovery) and `ig_entropy_external.py`
  (external) attribute node attention to the behavioural features and compare the
  attention entropy of the gated vs. zero-behaviour model.
- **mediation** - `matlab/xai/run_mediation_real_values.m` tests whether the
  group difference in behaviour is mediated by system-level activation.

`final_params_selection.py` is a small utility that aggregates the per-fold
hyperparameters into the single configuration used for the final model.

---

## Repository layout

```
gnn-fmri-schizophrenia/
├── README.md
├── LICENSE
├── requirements.txt
├── config/
│   └── gat_config.m              # central MATLAB paths (edit for your machine)
├── matlab/
│   ├── preprocessing/
│   │   ├── bold_signal_creation.m
│   │   ├── alff_calculation.m
│   │   ├── reho_calculation.m
│   │   └── gmv_wmv_calc.m
│   └── spm/
│       ├── spm_2level_gat.m            # second-level analysis + feature table
│       ├── spm_2level_gat_wrapper.m    # precompute results.mat for every fold
│       ├── spm_roi_extraction_gat.m
│       ├── mask_merged_rois_with_aal_gat.m
│       ├── process_subject_rois_with_marsbar_gat.m
│   │   ├── adjacency_matrix_calc.m
│   │   ├── roi_effectsize.m
│   │   └── sparsify_adjacency.m
│   └── xai/
│       └── run_mediation_real_values.m   # group -> activation -> behaviour mediation
└── src/
    ├── config.py                 # central Python paths (edit for your machine)
    ├── models.py                 # GAT, GCN, EarlyStopping
    ├── train_gat.py              # nested-CV driver (entry point)
    ├── final_params_selection.py # aggregate per-fold params into final config
    ├── external_validation.py    # external replication: GATE vs zero-behaviour
    ├── build_graphs_from_tr.py   # inject MATLAB features into PyG graphs
    ├── regress_out.py            # residualise features against age/sex
    ├── sparsify.py               # Erdős–Rényi adjacency sparsification
    ├── permutation_test.py       # subject-level label-permutation test
    ├── audit_graph_list.py       # graph integrity checks
    └── xai/                      # explainability (run after the discovery models exist)
        ├── graph_brain/
        │   ├── subgraphx_gat.py                   # SubgraphX per subject
        │   ├── subgraphx_statistical_inference.py # node/edge consensus + stats
        │   ├── post_hoc_biomarker_test.py         # keep/remove biomarker validation
        │   └── features_xai.py                    # feature IG necessity/sufficiency
        └── cognitive/
            ├── integrated_gradients_entropy.py    # behavioural IG + entropy (discovery)
            └── ig_entropy_external.py             # behavioural IG + entropy (external)
```

---

## Requirements

**Python** (3.10+; tested on 3.12) — see [`requirements.txt`](requirements.txt).
Install PyTorch and PyTorch Geometric first, following the official
instructions for your platform/CUDA, then:

```bash
pip install -r requirements.txt
```

**MATLAB** (tested with R2024b) with:

- [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- [MarsBaR](https://marsbar-toolbox.github.io/)

Add the repo's MATLAB folders and the toolboxes to your MATLAB path:

```matlab
addpath('/path/to/gnn-fmri-schizophrenia/config');
addpath(genpath('/path/to/gnn-fmri-schizophrenia/matlab'));
addpath('/path/to/spm12');
addpath('/path/to/marsbar');
```

The Python↔MATLAB bridge (`matlabengine`) is only needed if you compute SPM
results on the fly (`USE_PRECOMPUTED_SPM = False`); by default the Python code
loads precomputed `results.mat` files.

---

## Configuration

All machine-specific paths live in exactly two files — no other source file
contains an absolute path:

- **MATLAB:** [`config/gat_config.m`](config/gat_config.m)
- **Python:** [`src/config.py`](src/config.py) (each value can also be set via an
  environment variable, e.g. `GNN_DATA_ROOT`, `GNN_RESULTS_ROOT`).

Edit the values in the clearly marked *"EDIT THESE FOR YOUR MACHINE"* blocks to
point at your data and output directories before running anything.

---

## Running the pipeline

**1. Feature extraction (MATLAB).** Once `gat_config.m` is set up:

```matlab
bold_signal_creation();    % cache BOLD volumes
alff_calculation();        % append ALFF
reho_calculation();        % append ReHo
gmv_wmv_calc();            % tissue-density maps

% Precompute the SPM feature table for every cross-validation fold:
spm_2level_gat_wrapper('/path/to/results_root', 10);   % (root, sphere radius mm)
```

**2. GNN training (Python).** With `src/config.py` set up:

```bash
cd src
python train_gat.py
```

This runs nested cross-validation across all seeds, writes per-seed result
packages, summary CSVs and ROC curves to `RESULTS_ROOT`, and trains the final
model. Key switches at the top of `train_gat.py`:

- `USE_PRECOMPUTED_SPM` — load `results.mat` from disk (default) vs. call MATLAB;
- `SEEDS`, `OUTER_FOLDS`, `INNER_FOLDS`, `EPOCHS` — cross-validation settings;
- pass `permutation_on=True` to `run_all_seeds(...)` to run the permutation test.

**3. External replication (Python).** After the final discovery models exist,
point the external paths in `src/config.py` (`EXTERNAL_SPM_MAT`,
`EXTERNAL_ONSETS_MAT`, `GATE_MODEL_PATH`, `ZERO_MODEL_PATH`,
`EXTERNAL_RESULTS_DIR`) at the replication cohort and run:

```bash
cd src
python external_validation.py
```

Outputs (CSVs, ROC, consensus `.xlsx`, XAI packages) are written to
`EXTERNAL_RESULTS_DIR`.

**4. Explainability (Python).** With the discovery result packages in place:

```bash
cd src
python xai/graph_brain/subgraphx_gat.py                    # 1) per-subject SubgraphX
python xai/graph_brain/subgraphx_statistical_inference.py  # 2) consensus + stats
python xai/graph_brain/post_hoc_biomarker_test.py          # 3) keep/remove validation
python xai/graph_brain/features_xai.py                     # feature IG (nec/suf)
python xai/cognitive/integrated_gradients_entropy.py       # behavioural IG + entropy
```

The graph_brain scripts are ordered (each consumes the previous one's output).
XAI paths derive from `config.XAI_RESULTS_DIR` and the result directories.

---

## Reproducibility notes

- All RNGs (PyTorch, NumPy, CUDA, Optuna sampler) are seeded; CUDA uses
  deterministic algorithms.
- ROI definition, feature standardisation and demographic regression are fit on
  the **training split only** within each fold to avoid leakage.
- Cross-validation splits are at the **subject** level.
- The Python sparsifier (`sparsify.py`) reproduces the MATLAB
  `sparsify_adjacency.m` bit-for-bit (column-major indexing, round-half-up).

---

## Data availability

This repository contains **code only**. No subject-level neuroimaging or
behavioural data are included. Access to the underlying data is subject to the
original study's ethics approval and data-sharing agreements — see the paper for
details, or contact the authors.

---

## Citation

If you use this code, please cite:

```bibtex
@article{<citation_key>,
  title   = {<paper title>},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>},
  doi     = {<doi>}
}
```

---

## License

Released under the [MIT License](LICENSE).
