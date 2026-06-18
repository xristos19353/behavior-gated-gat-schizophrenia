"""Aggregate per-fold hyperparameters into a single final configuration.

Reads the per-fold/per-seed best-parameter table and the per-seed outer-fold
CSVs produced by ``train_gat.py`` and derives the final hyperparameters used to
train the model on the full dataset, in two flavours:

* **unweighted** - mode (categorical) / median (continuous) across folds;
* **performance-weighted** - weighted by each fold's AUC.

The number of final training epochs is taken from the outer-fold epochs
(median, and AUC-weighted mean).
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(config.GATE_RESULTS_DIR)
BEST_PARAMS_CSV = BASE_DIR / "best_params_per_fold_across_seeds.csv"
DF_OUTER_PATTERN = "df_outer_GAT_gate_only_seed*.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_mask_value(x):
    """Parse a mask cell into a tuple (handles list / stringified list)."""
    if pd.isna(x):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x.strip())
            if isinstance(val, list):
                return tuple(val)
        except Exception:
            pass
    return x


def plain_mode(series: pd.Series):
    m = series.mode(dropna=False)
    return None if len(m) == 0 else m.iloc[0]


def weighted_mode(series: pd.Series, weights: np.ndarray):
    tmp = pd.DataFrame({"value": series.values, "weight": weights})
    return tmp.groupby("value", dropna=False)["weight"].sum().idxmax()


def weighted_mean(series: pd.Series, weights: np.ndarray) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(vals) & np.isfinite(weights)
    if mask.sum() == 0:
        raise ValueError(f"No valid numeric values found for column {series.name}")
    return float(np.average(vals[mask], weights=weights[mask]))


def weighted_geometric_mean(series: pd.Series, weights: np.ndarray) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(vals) & np.isfinite(weights) & (vals > 0)
    if mask.sum() == 0:
        raise ValueError(f"No valid positive numeric values found for column {series.name}")
    return float(np.exp(np.average(np.log(vals[mask]), weights=weights[mask])))


def round_sig(x: float, sig: int = 2) -> float:
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def build_auc_weights(auc_series: pd.Series) -> np.ndarray:
    """Normalised, non-negative weights from AUC values (min-shifted)."""
    auc = pd.to_numeric(auc_series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(auc).all():
        raise ValueError("AUC column contains non-numeric or missing values.")
    weights = (auc - auc.min()) + 1e-8
    return weights / weights.sum()


def make_final_dict(d):
    keys = ["mask", "hidden_dim", "dropout", "lr", "weight_decay",
            "batch", "heads", "variant_id", "epochs_final"]
    return {k: d[k] for k in keys if k in d}


def main():
    # ----- Best-params table -------------------------------------------------
    df = pd.read_csv(BEST_PARAMS_CSV)
    required_cols = ["auc", "hidden_dim", "dropout", "lr", "weight_decay",
                     "batch", "heads", "variant_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in best params CSV: {missing}")

    if "mask" in df.columns:
        df["mask"] = df["mask"].apply(parse_mask_value)

    param_weights = build_auc_weights(df["auc"])

    # ----- Unweighted --------------------------------------------------------
    unweighted = {
        "hidden_dim": int(plain_mode(df["hidden_dim"])),
        "batch": int(plain_mode(df["batch"])),
        "heads": int(plain_mode(df["heads"])),
        "variant_id": int(plain_mode(df["variant_id"])),
    }
    if "mask" in df.columns:
        unweighted["mask"] = plain_mode(df["mask"])
    unweighted["dropout"] = round(float(pd.to_numeric(df["dropout"], errors="coerce").median()), 2)
    unweighted["lr"] = round_sig(float(pd.to_numeric(df["lr"], errors="coerce").median()), sig=2)
    unweighted["weight_decay"] = round_sig(
        float(pd.to_numeric(df["weight_decay"], errors="coerce").median()), sig=2)

    # ----- Performance-weighted ---------------------------------------------
    weighted = {
        "hidden_dim": int(weighted_mode(df["hidden_dim"], param_weights)),
        "batch": int(weighted_mode(df["batch"], param_weights)),
        "heads": int(weighted_mode(df["heads"], param_weights)),
        "variant_id": int(weighted_mode(df["variant_id"], param_weights)),
    }
    if "mask" in df.columns:
        weighted["mask"] = weighted_mode(df["mask"], param_weights)
    weighted["dropout"] = round(weighted_mean(df["dropout"], param_weights), 2)
    weighted["lr"] = round_sig(weighted_geometric_mean(df["lr"], param_weights), sig=2)
    weighted["weight_decay"] = round_sig(
        weighted_geometric_mean(df["weight_decay"], param_weights), sig=2)

    # ----- Epochs from the outer-fold CSVs ----------------------------------
    outer_files = sorted(BASE_DIR.glob(DF_OUTER_PATTERN))
    if len(outer_files) == 0:
        raise FileNotFoundError(f"No files found matching {DF_OUTER_PATTERN} in {BASE_DIR}")

    epochs_rows = []
    for f in outer_files:
        dfo = pd.read_csv(f)
        epoch_col = next((c for c in ["epochs_outer", "epoch_outer", "epochs", "best_epoch"]
                          if c in dfo.columns), None)
        if epoch_col is None:
            raise ValueError(f"No epoch column found in {f.name}. Columns: {list(dfo.columns)}")
        if "auc" not in dfo.columns:
            raise ValueError(f"No 'auc' column found in {f.name}")
        tmp = dfo[["auc", epoch_col]].rename(columns={epoch_col: "epochs_outer"})
        tmp["seed_file"] = f.stem.split("seed")[-1]
        epochs_rows.append(tmp)

    df_epochs = pd.concat(epochs_rows, ignore_index=True)
    epoch_weights = build_auc_weights(df_epochs["auc"])

    unweighted["epochs_final"] = int(round(
        float(pd.to_numeric(df_epochs["epochs_outer"], errors="coerce").median())))
    weighted["epochs_final"] = int(round(weighted_mean(df_epochs["epochs_outer"], epoch_weights)))

    # ----- Report and save ---------------------------------------------------
    print("\nUNWEIGHTED FINAL HYPERPARAMETERS\n" + "-" * 50)
    for k, v in unweighted.items():
        print(f"{k}: {v}")
    print("\nPERFORMANCE-WEIGHTED FINAL HYPERPARAMETERS\n" + "-" * 50)
    for k, v in weighted.items():
        print(f"{k}: {v}")

    pd.DataFrame([unweighted]).to_csv(
        BASE_DIR / "final_hyperparameters_unweighted_with_epochs.csv", index=False)
    pd.DataFrame([weighted]).to_csv(
        BASE_DIR / "final_hyperparameters_weighted_with_epochs.csv", index=False)

    print("\nREADY-TO-PASTE UNWEIGHTED\n", make_final_dict(unweighted))
    print("\nREADY-TO-PASTE WEIGHTED\n", make_final_dict(weighted))


if __name__ == "__main__":
    main()
