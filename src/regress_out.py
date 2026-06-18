"""Regress age and sex out of every node/edge/graph feature.

All features are z-scored using the *training* subjects only, then a linear
model on standardized age and sex is fit on the training subjects and its
prediction is subtracted from every subject. This removes nuisance demographic
variance without leaking test-set statistics.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def extract_subject_ids(subject_raw):
    """Normalise MATLAB-loaded subject identifiers into plain strings."""
    subject_ids = []
    for entry in subject_raw:
        if isinstance(entry, str):
            subject_ids.append(entry)
        elif isinstance(entry, tuple) and len(entry) == 4:
            try:
                # MATLAB char-array string objects come through as tuples.
                subject_ids.append("".join(chr(c) for c in entry[3] if c != 0))
            except Exception:
                subject_ids.append("UNKNOWN")
        else:
            subject_ids.append(str(entry))
    return subject_ids


def regress_out(T, train_subject_ids):
    """Residualise the features in ``T`` against standardized age and sex.

    Parameters
    ----------
    T : object
        Feature table (MATLAB struct loaded by scipy) with ``Subject``,
        ``Demographics`` and the feature attributes listed below.
    train_subject_ids : list
        Subject ids that define the training set used to fit every model.

    Returns
    -------
    T : object
        The same object with residualised features (modified in place).
    all_models : dict
        Fitted ``LinearRegression`` models, keyed by feature then field.
    final_stats : dict
        Standardisation statistics for age, sex and every feature/field.
    """
    all_ids = extract_subject_ids(T.Subject)
    train_ids_str = [str(s) for s in train_subject_ids]

    age = np.array([s.AGE for s in T.Demographics])
    sex = np.array([s.SEX for s in T.Demographics])
    train_indices = [i for i, sid in enumerate(all_ids) if sid in train_ids_str]

    # Standardise age and sex using the training subjects.
    age_mean, age_std = age[train_indices].mean(), age[train_indices].std()
    sex_mean, sex_std = sex[train_indices].mean(), sex[train_indices].std()
    z_age = (age - age_mean) / age_std
    z_sex = (sex - sex_mean) / sex_std

    for i in range(len(T.Demographics)):
        T.Demographics[i].AGE = z_age[i]
        T.Demographics[i].SEX = z_sex[i]

    X_all = np.column_stack((z_age, z_sex))
    X_train = X_all[train_indices]

    features_to_residualize = [
        "WMV", "GMV", "EffectSize", "ALFF", "ReHo", "DC",
        "Z", "Zweighted", "Behavior",
    ]

    all_models = {}
    all_stats = {}

    for feat_name in features_to_residualize:
        feat_data = getattr(T, feat_name)
        first_struct = feat_data[0]

        if feat_name in ("Z", "Zweighted"):
            keys = [feat_name]
        else:
            keys = [
                k for k in first_struct.__dict__.keys()
                if isinstance(getattr(first_struct, k), (int, float, np.integer, np.floating))
            ]

        models = {}
        stats = {}

        for key in keys:
            # --- Connectivity matrices (Z / Zweighted): residualise per cell ---
            if key in ("Z", "Zweighted"):
                try:
                    matrices = [np.array(getattr(subj, key), dtype=float) for subj in feat_data]
                    stack = np.stack(matrices)              # (n_subjects, d, d)
                    n_subjects, d1, d2 = stack.shape
                    flat = stack.reshape(n_subjects, -1)

                    mean = flat[train_indices].mean(axis=0)
                    std = flat[train_indices].std(axis=0)
                    std[std == 0] = 1
                    z_data = (flat - mean) / std

                    model = LinearRegression().fit(X_train, z_data[train_indices])
                    residuals = z_data - model.predict(X_all)

                    residuals_reshaped = residuals.reshape(n_subjects, d1, d2)
                    for i in range(n_subjects):
                        setattr(feat_data[i], key, residuals_reshaped[i])

                    models[key] = model
                    stats[key] = {"mean": mean, "std": std}
                except Exception as e:
                    print(f"[{key}] Skipping regression due to error: {e}")
                continue

            # Behaviour group label is categorical; never residualise it.
            if feat_name == "Behavior" and key == "Group":
                continue

            # --- Scalar features: residualise the standardized values ----------
            try:
                y = np.array([getattr(subj, key) for subj in feat_data], dtype=float)
            except Exception as e:
                print(f"Skipping {feat_name}.{key}: {e}")
                continue

            y_train = y[train_indices]
            y_mean, y_std = y_train.mean(), y_train.std()
            z_y = (y - y_mean) if y_std == 0 else (y - y_mean) / y_std

            model = LinearRegression().fit(X_train, z_y[train_indices])
            residuals = z_y - model.predict(X_all)

            for i in range(len(feat_data)):
                setattr(feat_data[i], key, residuals[i])

            models[key] = model
            stats[key] = {"mean": y_mean, "std": y_std}

        all_models[feat_name] = models
        all_stats[feat_name] = stats

    final_stats = {
        "age_mean": age_mean, "age_std": age_std,
        "sex_mean": sex_mean, "sex_std": sex_std,
        "features": all_stats,
    }

    return T, all_models, final_stats
