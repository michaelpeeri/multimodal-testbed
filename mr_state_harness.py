# OUTSTANDING ISSUES / LIMITATIONS
# - A comparison run uses one shared GRN, so it isolates state/simulation
#   effects but does not measure robustness to topology or GRN-parameter change.
# - Random/Sobol arms currently resample their selected MR-state matrix for
#   each replicate; only fixed-state arms isolate expression robustness.
# - The default pilot may use fewer cells than the target statistics. Absolute
#   target distances are then screening values, although paired arm deltas are
#   still useful when seeds are shared.
# - PCA stability is conditional on one fixed gene subset and cell split seed;
#   it is not uncertainty over all possible subsets/splits.
# - Fixed-state artifacts must carry and be checked against ordered MR IDs and
#   an exact GRN fingerprint; legacy pickles do not contain that provenance.
# - The full SERGIO pipeline is intentionally used for validation, not for
#   large candidate-population optimization.
#
# NEXT EXPERIMENT PLAN: STAGE-ISOLATION PILOT
# - Compare constant-state, IID-state, and intrinsic-DE-state arms using the
#   same GRN, MR ordering, cell count, replicate seeds, and target-statistics
#   settings. The primary question is whether DE-created cluster structure is
#   present before technical stages and then erased by full SERGIO processing.
# - Run the pilot in two harness invocations: clean_simulation=true (SERGIO
#   with noise, outliers, library-size effects, dropout, and UMI conversion
#   disabled) and clean_simulation=false (the full configured pipeline). The
#   harness deliberately makes clean_simulation a run-level switch, so two
#   configs are preferable to adding per-arm scenario plumbing.
# - The arm set should include iid_random, a constant-state control whose
#   per-MR vector is the grand mean of the existing IID states, and both new
#   fixed DE candidates. The constant control removes cluster-to-cluster MR
#   variation while preserving the average MR rate as closely as possible.
# - Evaluate PC2-PC9 explained variance, cluster/between-state separation,
#   within-versus-across module correlation, expression variance, sparsity,
#   and full target distance. Reuse the existing full-pipeline IID results
#   when the GRN fingerprint, ordered MR IDs, seeds, and configuration match;
#   do not rerun already-completed control arms merely to populate a config.
# - Decision rule: if DE improves clean SERGIO but not full SERGIO, technical
#   stages are suppressing biological signal; if it fails already in clean
#   SERGIO, the GRN/MR-state construction or surrogate is the bottleneck. If
#   neither arm beats IID in clean SERGIO, stop investing in this DE objective
#   and move to explicitly structured, low-rank MR programs.
# - No harness code change is required for this pilot. Existing fixed_array or
#   fixed_pickle arms can load the constant matrix; only a derived .npy file
#   and suitable clean/full configs are needed. A future constant candidate
#   method would be convenience only, not required for the experiment.

"""Compare multiple MR-state candidate and selection configurations.

This module is deliberately separate from the Optuna tuning entry point.  It
reuses the GRN, DAG/Hill surrogate, SERGIO wrapper, and
``compute_summary_stats`` implementation from :mod:`synthetic_data`, but does
not create or modify an Optuna study.

Example configuration::

    {
      "base_config": "synthetic_tuning_20260821.00/tuned_synthetic_config.20260821.00.json",
      "arms": [
        {"name": "iid_random", "candidate_method": "iid",
         "selection_method": "random",
         "candidate_params": {"n_candidate_states": 64},
         "selection_params": {"n_selected_states": 15}},
        {"name": "sobol_random", "candidate_method": "sobol",
         "selection_method": "random",
         "candidate_params": {"n_candidate_states": 64},
         "selection_params": {"n_selected_states": 15}},
        {"name": "sobol_spectral", "candidate_method": "sobol",
         "selection_method": "spectral",
         "candidate_params": {"n_candidate_states": 64},
         "selection_params": {"n_selected_states": 15,
                               "n_restarts": 1, "swap_passes": 0}},
         {"name": "de_f05_cr08_seed1", "candidate_method": "fixed_pickle",
          "candidate_params": {
              "path": "ga_opt_log_20260827_02.de_F05_CR08.seed1.pickle",
              "key": "best_candidate"},
          "selection_method": "identity",
          "selection_params": {"n_selected_states": 15}}
       ],
      "n_replicates": 3,
      "seed_base": 0,
      "include_sergio": true,
      "clean_simulation": true,
      "n_cells": 60,
      "output": "mr_state_comparison.pickle"
    }

The result pickle contains per-replicate states, statistics, distance
breakdowns, and summaries.  A JSON and CSV summary are written beside it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from synthetic_data import (
    add_pca_derived_stats,
    add_nonzero_frac_stat,
    build_mr_overlap_tree,
    build_mr_profile_tree,
    build_mr_target_sets,
    build_mr_transitive_profiles,
    compute_summary_stats,
    generate_sergio_grn_from_reference,
    load_sergio_dag,
    make_synthetic_data6,
    sample_mr_state_from_tree,
    sample_sergio_mr_states,
    select_sergio_spectral_subset,
    sergio_dag_hill_forward,
    sergio_spectral_metrics,
    _gene_module_correlations,
    _load_mr_tree_experiment_config,
)


_SCALAR_METRIC_KEYS = (
    "distance",
    "mr_state_participation_ratio",
    "label_between_variance_fraction",
    "label_centroid_participation_ratio",
    "label_centroid_mean_pairwise_distance",
    "label_holdout_accuracy",
    "pca_size_normalized_standardized_tail_participation_ratio",
    "pca_size_normalized_standardized_split_half_subspace_stability",
    "pca_size_normalized_standardized_split_half_spectrum_similarity",
    "gene_corr_abs_normalized_mean",
    "gene_corr_abs_normalized_p90",
    "gene_mean_mean",
    "gene_var_mean",
    "log_lib_size_std",
    "zero_frac",
    "nonzero_frac",
    "within_module_mean_abs_corr",
    "across_module_mean_abs_corr",
    "within_minus_across",
)

_PCA_ARRAY_KEYS = (
    "pca_explained_variance_ratio",
    "pca_pc2_9_explained_variance_ratio",
    "pca_standardized_explained_variance_ratio",
    "pca_standardized_pc2_9_explained_variance_ratio",
    "pca_size_normalized_standardized_explained_variance_ratio",
    "pca_size_normalized_standardized_pc2_9_explained_variance_ratio",
)


def _load_tuning_helpers():
    """Load shared config/distance helpers without creating an Optuna study."""
    import tune_synthetic_data as tsd
    return tsd


def _load_base_config(config: str | dict) -> tuple[dict, str | None]:
    """Resolve a tuning or tuned-result config using the existing loader."""
    if isinstance(config, str):
        with open(config) as f:
            raw = json.load(f)
        source = config
    else:
        raw = dict(config)
        source = None

    overrides = dict(raw.get("base_overrides") or {})
    if "base_config" in raw:
        base_source = raw["base_config"]
        if isinstance(base_source, str):
            with open(base_source) as f:
                base = json.load(f)
            source = base_source
        else:
            base = dict(base_source)
    else:
        base = raw

    _load_tuning_helpers()
    # Normalize tuned-result configs first: the normalizer intentionally reads
    # resolved values from _meta/best_resolved_params and otherwise ignores
    # top-level fields.  Applying overrides afterward ensures harness-local
    # settings such as weights and statistics options are not discarded.
    normalized = _load_mr_tree_experiment_config(base)
    normalized.update(overrides)
    return normalized, source


def _load_harness_config(path: str) -> tuple[dict, str | None]:
    with open(path) as f:
        harness = json.load(f)
    return harness, path


def _jsonify(value):
    """Convert result summaries to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value
    return value


def _safe_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _build_trees(grn_path: str, mr_ids: list[int], path_decay: float) -> dict:
    """Build the existing three profile trees for candidate generation."""
    target_sets = build_mr_target_sets(grn_path, mr_ids)
    direct = build_mr_overlap_tree(grn_path, mr_ids, target_sets=target_sets)
    hard, soft = build_mr_transitive_profiles(
        grn_path, mr_ids, path_decay=path_decay)
    trees = {
        "direct": direct,
        "transitive_hard": build_mr_profile_tree(mr_ids, hard, weighted=False),
        "transitive_soft": build_mr_profile_tree(mr_ids, soft, weighted=True),
    }
    return {"target_sets": target_sets, "trees": trees}


def _load_fixed_states(
    path: str,
    key: str | None = None,
    expected_mr_ids: list[int] | None = None,
    expected_grn_sha256: str | None = None,
    require_grn_provenance: bool = False,
) -> np.ndarray:
    """Load a fixed state matrix from NumPy, pickle, or PyTorch data.

    Pickle dictionaries default to ``best_candidate`` because that is the
    matrix stored by ``mr_state_ga_opt.py``.  PyTorch artifact dictionaries
    retain the existing ``mr_state`` default.  Supplying ``key`` overrides
    either default.
    """
    suffix = Path(path).suffix.lower()
    artifact = None
    if suffix == ".npy":
        states = np.load(path)
    elif suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        if isinstance(artifact, dict):
            artifact_key = key or "best_candidate"
            states = artifact.get(artifact_key)
        else:
            if key is not None:
                raise ValueError(
                    f"fixed pickle {path!r} is not a mapping; cannot read key {key!r}"
                )
            states = artifact
    else:
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(path, map_location="cpu")
        if isinstance(artifact, dict):
            states = artifact.get(key or "mr_state")
        else:
            if key is not None:
                raise ValueError(
                    f"fixed artifact {path!r} is not a mapping; cannot read key {key!r}"
                )
            states = artifact
    if states is None:
        expected = key or ("best_candidate" if suffix in (".pkl", ".pickle") else "mr_state")
        raise ValueError(f"fixed state file {path!r} has no {expected}")
    provenance = artifact.get("grn") if isinstance(artifact, dict) else None
    if expected_mr_ids is not None or expected_grn_sha256 is not None:
        if provenance is None:
            if require_grn_provenance:
                raise ValueError(
                    f"fixed state file {path!r} has no GRN provenance metadata"
                )
        else:
            if isinstance(provenance, list):
                provenance = provenance[0] if provenance else None
            if not isinstance(provenance, dict):
                raise ValueError(f"fixed state file {path!r} has invalid GRN provenance")
            if expected_mr_ids is not None and list(provenance.get("mr_ids", [])) != list(expected_mr_ids):
                raise ValueError(
                    f"fixed state file {path!r} has incompatible ordered MR IDs"
                )
            if expected_grn_sha256 is not None and provenance.get("sha256") != expected_grn_sha256:
                raise ValueError(
                    f"fixed state file {path!r} was generated from a different GRN"
                )
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or not np.isfinite(states).all() or (states < 0).any():
        raise ValueError(f"fixed state file {path!r} must contain a finite 2D non-negative matrix")
    return states


def _candidate_pool(
    arm: dict,
    base: dict,
    trees: dict,
    mr_ids: list[int],
    seed: int,
) -> tuple[np.ndarray, dict]:
    method = arm["candidate_method"]
    params = dict(arm.get("candidate_params") or {})
    n_candidates = int(params.get(
        "n_candidate_states",
        (arm.get("selection_params") or {}).get("n_selected_states", base["n_clusters"]),
    ))
    low = float(params.get("low", base["mr_rate_low"]))
    high = float(params.get("high", base["mr_rate_high"]))
    if high <= low:
        raise ValueError(f"arm {arm['name']!r} has high <= low")

    if method in ("iid", "sobol"):
        states = sample_sergio_mr_states(
            n_states=n_candidates,
            n_mrs=len(mr_ids),
            low=low,
            high=high,
            # The harness exposes the unambiguous public name "iid" while
            # the existing shared sampler retains its legacy "random" name.
            design="random" if method == "iid" else method,
            seed=seed,
        )
    elif method.startswith("tree_"):
        tree_name = method.removeprefix("tree_")
        if tree_name not in trees["trees"]:
            raise ValueError(f"unknown tree candidate method {method!r}")
        states = sample_mr_state_from_tree(
            trees["trees"][tree_name],
            n_states=n_candidates,
            low=low,
            high=high,
            seed=seed,
            tree_strength=float(params.get("tree_strength", 1.0)),
            root_variance=float(params.get("root_variance", 0.0)),
        )
    elif method in ("fixed_array", "fixed_artifact", "fixed_pickle"):
        path = params.get("path")
        if not path:
            raise ValueError(f"arm {arm['name']!r} requires candidate_params.path")
        if method == "fixed_pickle" and Path(path).suffix.lower() not in (".pkl", ".pickle"):
            raise ValueError(
                f"arm {arm['name']!r} uses fixed_pickle but path {path!r} is not a pickle"
            )
        # Some historical tuning rounds intentionally reused a fixed state
        # while varying GRN parameters. Keep strict hash validation by default,
        # but allow those replays to opt out explicitly while retaining MR-ID
        # ordering validation.
        expected_grn_sha256 = (
            base.get("_grn_sha256")
            if base.get("check_fixed_state_grn_provenance", True)
            else None
        )
        states = _load_fixed_states(
            path,
            key=params.get("key"),
            expected_mr_ids=mr_ids,
            expected_grn_sha256=expected_grn_sha256,
            require_grn_provenance=bool(base.get("require_grn_provenance", False)),
        )
    else:
        raise ValueError(
            f"unsupported candidate_method {method!r}; expected iid, sobol, "
            "tree_*, fixed_array, fixed_artifact, or fixed_pickle"
        )

    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != len(mr_ids):
        raise ValueError(
            f"arm {arm['name']!r} produced state shape {states.shape}; "
            f"expected (n_states, {len(mr_ids)}) for this GRN"
        )
    if not np.isfinite(states).all() or (
        (states < float(base['mr_rate_low'])).any()
        or (states > float(base['mr_rate_high'])).any()
    ):
        raise ValueError(
            f"arm {arm['name']!r} produced states outside "
            f"[{base['mr_rate_low']}, {base['mr_rate_high']}]"
        )

    metadata = {
        "method": method,
        "seed": int(seed),
        "n_candidates": int(states.shape[0]),
        "n_mrs": int(states.shape[1]),
        "low": low,
        "high": high,
        "params": params,
    }
    return np.asarray(states, dtype=np.float32), metadata


def _maximin_selection(pool: np.ndarray, dag, n_selected: int, decays, seed: int) -> list[int]:
    """Select a diverse subset using greedy maximin surrogate distance."""
    if n_selected == pool.shape[0]:
        return list(range(pool.shape[0]))
    if n_selected < 2 or n_selected > pool.shape[0]:
        raise ValueError("maximin selection requires 2 <= n_selected <= pool size")
    response = sergio_dag_hill_forward(pool, dag, decays=decays)
    scale = np.std(response, axis=0)
    scale[scale < 1e-12] = 1.0
    response = response / scale
    center = response.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(response - center, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(response - response[first], axis=1)
    min_dist[first] = -np.inf
    rng = np.random.default_rng(seed)
    # Randomly resolving exact ties keeps the selector deterministic for a
    # given seed without depending on dictionary/set iteration order.
    while len(selected) < n_selected:
        best = np.flatnonzero(min_dist == np.max(min_dist))
        choice = int(best[rng.integers(len(best))])
        selected.append(choice)
        distances = np.linalg.norm(response - response[choice], axis=1)
        min_dist = np.minimum(min_dist, distances)
        min_dist[selected] = -np.inf
    return sorted(selected)


def _select_states(
    arm: dict,
    pool: np.ndarray,
    dag,
    base: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    method = arm["selection_method"]
    params = dict(arm.get("selection_params") or {})
    n_selected = int(params.get("n_selected_states", base["n_clusters"]))
    if not 2 <= n_selected <= pool.shape[0]:
        raise ValueError(
            f"arm {arm['name']!r} requires 2 <= n_selected_states <= pool size; "
            f"got {n_selected} and {pool.shape[0]}"
        )

    if method == "identity":
        if pool.shape[0] != n_selected:
            raise ValueError(
                f"identity selection requires pool size == n_selected_states, "
                f"got {pool.shape[0]} and {n_selected}"
            )
        indices = list(range(pool.shape[0]))
        selector_metrics = {}
    elif method == "random":
        indices = sorted(np.random.default_rng(seed).choice(
            pool.shape[0], size=n_selected, replace=False).tolist())
        selector_metrics = {}
    elif method == "spectral":
        indices, selector_metrics, selector_diag = select_sergio_spectral_subset(
            pool,
            dag,
            decays=base["decays"],
            subset_size=n_selected,
            n_restarts=int(params.get("n_restarts", 1)),
            swap_passes=int(params.get("swap_passes", 0)),
            seed=seed,
            variance_weight=float(params.get("variance_weight", 0.05)),
        )
        selector_metrics = {
            **selector_metrics,
            "n_surrogate_evaluations": selector_diag["n_surrogate_evaluations"],
            "n_swap_evaluations": selector_diag["n_swap_evaluations"],
        }
    elif method == "maximin":
        indices = _maximin_selection(
            pool, dag, n_selected, base["decays"], seed)
        selector_metrics = {}
    else:
        raise ValueError(
            f"unsupported selection_method {method!r}; expected identity, "
            "random, spectral, or maximin"
        )
    return pool[indices].astype(np.float32, copy=True), {
        "method": method,
        "seed": int(seed),
        "n_selected": n_selected,
        "selected_indices": [int(i) for i in indices],
        "metrics": selector_metrics,
    }


def _mr_state_participation_ratio(states: np.ndarray) -> float:
    if states.shape[0] < 2:
        return float("nan")
    singular = np.linalg.svd(
        states - states.mean(axis=0, keepdims=True),
        full_matrices=False,
        compute_uv=False,
    )
    variance = singular ** 2
    total = float(variance.sum())
    return float(total ** 2 / np.sum(variance ** 2)) if total > 0 else float("nan")


def _state_geometry(states: np.ndarray) -> dict:
    centered = states - states.mean(axis=0, keepdims=True)
    if states.shape[0] < 2:
        return {"participation_ratio": float("nan"), "min_pairwise_distance": float("nan")}
    distances = np.linalg.norm(
        centered[:, None, :] - centered[None, :, :], axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    return {
        "participation_ratio": _mr_state_participation_ratio(states),
        "min_pairwise_distance": float(np.min(distances)),
        "mean_pairwise_distance": float(np.mean(distances[np.isfinite(distances)])),
    }


def _label_aware_expression_metrics(
    X: torch.Tensor,
    labels: np.ndarray,
    seed: int | None = 0,
) -> dict:
    """Measure expression structure that is reproducible across labels.

    The input is imputed and standardized per gene before the label-aware
    measurements.  This makes the metrics less dependent on a few high-count
    genes while keeping them aligned with the PCA stability calculations.
    ``label_between_variance_fraction`` measures how much standardized
    expression variance is explained by labels; centroid participation ratio
    measures the effective dimensionality of that label signal; and the
    holdout accuracy tests whether the label signal generalizes to cells not
    used to form the centroids.
    """
    if isinstance(X, torch.Tensor):
        matrix = X.detach().cpu().numpy()
    else:
        matrix = np.asarray(X)
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels)
    output = {
        "label_between_variance_fraction": float("nan"),
        "label_centroid_participation_ratio": float("nan"),
        "label_centroid_mean_pairwise_distance": float("nan"),
        "label_holdout_accuracy": float("nan"),
    }
    if matrix.ndim != 2 or labels.ndim != 1 or matrix.shape[0] != labels.size:
        return output

    valid_labels = np.isfinite(labels)
    if not valid_labels.all():
        matrix = matrix[valid_labels]
        labels = labels[valid_labels]
    if matrix.shape[0] < 2:
        return output

    observed = np.isfinite(matrix)
    observed_values = np.where(observed, matrix, 0.0)
    observed_count = observed.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_mean = observed_values.sum(axis=0) / observed_count
    gene_mean = np.where(observed_count > 0, gene_mean, 0.0)
    filled = np.where(observed, matrix, gene_mean[None, :])

    unique_labels, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    if unique_labels.size < 2:
        return output
    global_mean = filled.mean(axis=0, keepdims=True)
    gene_std = filled.std(axis=0, keepdims=True)
    gene_std = np.maximum(gene_std, 1e-6)
    standardized = (filled - global_mean) / gene_std

    n_labels = unique_labels.size
    centroids = np.zeros((n_labels, standardized.shape[1]), dtype=np.float64)
    for index in range(n_labels):
        members = inverse == index
        centroids[index] = standardized[members].mean(axis=0)

    total_ss = float(np.square(standardized).sum())
    between_ss = float((counts[:, None] * np.square(centroids)).sum())
    if total_ss > 0.0:
        output["label_between_variance_fraction"] = float(
            np.clip(between_ss / total_ss, 0.0, 1.0)
        )

    centered_centroids = centroids - centroids.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(
        centered_centroids, full_matrices=False, compute_uv=False)
    centroid_variance = np.square(singular)
    centroid_total = float(centroid_variance.sum())
    if centroid_total > 0.0:
        output["label_centroid_participation_ratio"] = float(
            centroid_total ** 2 / np.square(centroid_variance).sum()
        )
    centroid_distances = np.linalg.norm(
        centered_centroids[:, None, :] - centered_centroids[None, :, :], axis=2)
    if n_labels > 1:
        upper = centroid_distances[np.triu_indices(n_labels, k=1)]
        output["label_centroid_mean_pairwise_distance"] = float(upper.mean())

    # Form train/test sets independently within each label so class balance
    # does not turn the accuracy into a proxy for the Dirichlet draw.
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(labels.size, dtype=bool)
    test_mask = np.zeros(labels.size, dtype=bool)
    for index in range(n_labels):
        members = np.flatnonzero(inverse == index)
        if members.size < 2:
            continue
        members = members[rng.permutation(members.size)]
        n_train = max(1, members.size // 2)
        train_mask[members[:n_train]] = True
        test_mask[members[n_train:]] = True
    if train_mask.any() and test_mask.any() and np.unique(inverse[train_mask]).size == n_labels:
        train = filled[train_mask]
        test = filled[test_mask]
        train_mean = train.mean(axis=0, keepdims=True)
        train_std = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
        train = (train - train_mean) / train_std
        test = (test - train_mean) / train_std
        train_labels = inverse[train_mask]
        train_centroids = np.zeros((n_labels, train.shape[1]), dtype=np.float64)
        for index in range(n_labels):
            train_centroids[index] = train[train_labels == index].mean(axis=0)
        distances = np.square(
            test[:, None, :] - train_centroids[None, :, :]).sum(axis=2)
        predicted = np.argmin(distances, axis=1)
        output["label_holdout_accuracy"] = float(
            np.mean(predicted == inverse[test_mask])
        )
    return output


def _simulate_arm(
    states: np.ndarray,
    base: dict,
    grn_path: str,
    mr_ids: list[int],
    simulation_seed: int,
    clean_simulation: bool,
    n_cells: int,
    device,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    sim_kwargs = {
        key: base[key]
        for key in (
            "noise_params", "decays", "cluster_conc", "outlier_prob",
            "outlier_mean", "outlier_scale", "lib_size_mean", "lib_size_scale",
            "dropout_shape", "dropout_percentile",
        )
    }
    if clean_simulation:
        sim_kwargs["noise_params"] = 0.0
    X, labels = make_synthetic_data6(
        mr_state=torch.tensor(states, dtype=torch.float32, device=device),
        input_file_targets=grn_path,
        n_cells=n_cells,
        mr_gene_ids=mr_ids,
        shared_coop_state=base["shared_coop_state"],
        noise_type=base["noise_type"],
        sampling_state=2 if clean_simulation else base["sampling_state"],
        dt=base["dt"],
        min_cells_per_cluster=1 if clean_simulation else base["min_cells_per_cluster"],
        add_outlier_genes=False if clean_simulation else base["add_outlier_genes"],
        add_lib_size_effect=False if clean_simulation else base["add_lib_size_effect"],
        add_dropout=False if clean_simulation else base["add_dropout"],
        convert_to_umi_counts=False if clean_simulation else base["convert_to_umi_counts"],
        missing_rate=0.0,
        seed=simulation_seed,
        device=device,
        **sim_kwargs,
    )
    return X, np.asarray(labels), {"clean_simulation": bool(clean_simulation)}


def _evaluate_matrix(
    X: torch.Tensor,
    states: np.ndarray,
    labels: np.ndarray,
    base: dict,
    target_stats: dict | None,
    tsd,
    surrogate_metrics: dict | None = None,
    winner_mr_by_gene: tuple | None = None,
    mr_ids: list[int] | None = None,
    gene_id_to_symbol: dict | None = None,
) -> dict:
    stats = compute_summary_stats(
        X,
        n_pca_components=base["stats_n_pca_components"],
        n_structure_genes=base["stats_n_structure_genes"],
        percentiles=tuple(base["stats_percentiles"]),
        seed=base["stats_seed"],
    )
    # This is a compatibility no-op for new summaries and fills the derived
    # keys when a caller supplies an older target pickle.
    add_pca_derived_stats(stats)
    add_nonzero_frac_stat(stats)
    distance = None
    breakdown = None
    if target_stats is not None:
        distance, breakdown = tsd.compute_stats_distance(
            target_stats,
            stats,
            weights=base.get("weights"),
            eps=base.get("distance_eps", 1e-6),
            eps_frac=base.get("distance_eps_frac", 0.05),
            eps_abs_floor=base.get("distance_eps_abs_floor", 0.02),
        )
    geometry = _state_geometry(states)
    label_metrics = _label_aware_expression_metrics(
        X, labels, seed=base.get("stats_seed", 0))
    module_metrics = _gene_module_correlations(
        X,
        labels,
        winner_mr_by_gene,
        mr_ids or [],
        gene_id_to_symbol or {},
        n_structure_genes=base["stats_n_structure_genes"],
        seed=base["stats_seed"],
    )
    scalar_metrics = {
        key: _safe_float(stats.get(key))
        for key in _SCALAR_METRIC_KEYS
        if key != "distance"
    }
    scalar_metrics["distance"] = _safe_float(distance)
    scalar_metrics["mr_state_participation_ratio"] = geometry["participation_ratio"]
    for key in (
        "label_between_variance_fraction",
        "label_centroid_participation_ratio",
        "label_centroid_mean_pairwise_distance",
        "label_holdout_accuracy",
    ):
        scalar_metrics[key] = _safe_float(label_metrics[key])
    for key in (
        "within_module_mean_abs_corr",
        "across_module_mean_abs_corr",
        "within_minus_across",
    ):
        scalar_metrics[key] = _safe_float(module_metrics[key])
    return {
        "stats": stats,
        "distance": distance,
        "distance_breakdown": breakdown,
        "scalar_metrics": scalar_metrics,
        "mr_state_geometry": geometry,
        "label_metrics": label_metrics,
        "module_metrics": module_metrics,
        "surrogate_metrics": surrogate_metrics or {},
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_labeled_cells": int(len(labels)),
    }


def _summarize_values(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "q10": float("nan"),
                "q90": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "q10": float(np.percentile(array, 10)),
        "q90": float(np.percentile(array, 90)),
        "n": int(array.size),
    }


def _summarize_arm(replicates: list[dict], scenario_names: list[str]) -> dict:
    summary = {"scenarios": {}}
    for scenario in scenario_names:
        records = [rep["scenarios"].get(scenario) for rep in replicates]
        records = [record for record in records if record is not None]
        metrics = {}
        for key in _SCALAR_METRIC_KEYS:
            metrics[key] = _summarize_values([
                record["scalar_metrics"].get(key, float("nan"))
                for record in records
            ])
        for key in ("participation_ratio", "min_pairwise_distance", "mean_pairwise_distance"):
            metrics[f"mr_state_{key}"] = _summarize_values([
                record["mr_state_geometry"].get(key, float("nan"))
                for record in records
            ])

        arrays = {}
        for key in _PCA_ARRAY_KEYS:
            pca_arrays = [
                np.asarray(record["stats"].get(key, []), dtype=np.float64)
                for record in records
            ]
            pca_arrays = [array for array in pca_arrays if array.size]
            if pca_arrays:
                length = min(len(array) for array in pca_arrays)
                matrix = np.stack([array[:length] for array in pca_arrays])
                arrays[key] = {
                    "mean": matrix.mean(axis=0),
                    "std": matrix.std(axis=0),
                    "n": int(matrix.shape[0]),
                }
        summary["scenarios"][scenario] = {"metrics": metrics, "arrays": arrays, "n": len(records)}
    return summary


def _paired_deltas(arm_results: dict, baseline_name: str, scenario_names: list[str]) -> dict:
    baseline = arm_results[baseline_name]["replicates"]
    output = {}
    for arm_name, arm in arm_results.items():
        if arm_name == baseline_name:
            continue
        output[arm_name] = {"scenarios": {}}
        for scenario in scenario_names:
            left = [rep["scenarios"].get(scenario) for rep in arm["replicates"]]
            right = [rep["scenarios"].get(scenario) for rep in baseline]
            metrics = {}
            for key in _SCALAR_METRIC_KEYS:
                deltas = []
                for a, b in zip(left, right):
                    if a is None or b is None:
                        continue
                    av = a["scalar_metrics"].get(key, float("nan"))
                    bv = b["scalar_metrics"].get(key, float("nan"))
                    if math.isfinite(av) and math.isfinite(bv):
                        deltas.append(av - bv)
                metrics[key] = _summarize_values(deltas)
            output[arm_name]["scenarios"][scenario] = metrics
    return output


def run_mr_state_comparison(config: str | dict) -> dict:
    """Run and persist a paired comparison of multiple MR-state arms.

    Defaults are intentionally pilot-sized: three replicates, surrogate-only
    evaluation, clean simulation semantics, and one configuration-defined
    cell count defaults to one cell per state when clean simulation is enabled.
    Set ``include_sergio`` and ``n_cells`` explicitly for a more expensive
    validation run.
    """
    run_start = time.perf_counter()
    if isinstance(config, str):
        harness, harness_source = _load_harness_config(config)
    else:
        harness = dict(config)
        harness_source = None
    if not harness.get("arms"):
        raise ValueError("harness config requires a non-empty 'arms' list")
    verbose = bool(harness.get("verbose", True))

    def log(message: str) -> None:
        if verbose:
            tqdm.write(message)

    base_input = harness if "base_config" not in harness else {
        "base_config": harness["base_config"],
        "base_overrides": harness.get("base_overrides", {}),
    }
    base, base_source = _load_base_config(base_input)
    tsd = _load_tuning_helpers()
    target_stats = None
    if base.get("target_stats_path"):
        log(f"[setup] loading target stats: {base['target_stats_path']}")
        with open(base["target_stats_path"], "rb") as f:
            target_stats = pickle.load(f)
        add_pca_derived_stats(target_stats)
        add_nonzero_frac_stat(target_stats)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp_path = harness.get("grn_output_path")
    if temp_path is None:
        temp_path = os.path.join(
            base.get("grn_tmp_dir") or "/tmp",
            f"mr_state_harness_{os.getpid()}.csv",
        )
    Path(temp_path).parent.mkdir(parents=True, exist_ok=True)

    log(
        f"[setup] generating shared GRN: n_genes={base['n_genes']} "
        f"grn_seed={base['grn_seed']}"
    )
    grn_diagnostics = {}
    _, mr_ids, gene_id_to_symbol = generate_sergio_grn_from_reference(
        reference_grn_path=base["reference_grn_path"],
        n_genes=base["n_genes"],
        output_path=temp_path,
        delimiter=base["grn_delimiter"],
        regulator_col=base["grn_regulator_col"],
        target_col=base["grn_target_col"],
        mode_col=base["grn_mode_col"],
        activation_labels=base["grn_activation_labels"],
        repression_labels=base["grn_repression_labels"],
        unknown_mode_repressor_prob=base["unknown_mode_repressor_prob"],
        k_dist=("uniform", base["grn_k_low"], base["grn_k_low"] + base["grn_k_span"]),
        hill_coeff_dist=("constant", base["hill_coeff"]),
        max_seed_attempts=base["grn_max_seed_attempts"],
        seed=base["grn_seed"],
        diagnostics=grn_diagnostics,
        coherency_bias=base["coherency_bias"],
        canalization_strength=base["canalization_strength"],
        balancing_strength=base["balancing_strength"],
        path_decay=base["path_decay"],
    )
    grn_digest = hashlib.sha256()
    with open(temp_path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            grn_digest.update(block)
    base["_grn_sha256"] = grn_digest.hexdigest()
    dag = load_sergio_dag(
        temp_path,
        shared_coop_state=base["shared_coop_state"],
        mr_gene_ids=mr_ids,
    )
    trees = _build_trees(temp_path, mr_ids, float(harness.get("tree_path_decay", 0.9)))
    winner_mr_by_gene = (
        (grn_diagnostics.get("tgt_ids", []), grn_diagnostics.get("winner_mr", []))
        if grn_diagnostics.get("winner_mr") else None
    )

    n_replicates = int(harness.get("n_replicates", 3))
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    seed_base = int(harness.get("seed_base", 0))
    include_sergio = bool(harness.get("include_sergio", False))
    clean_simulation = bool(harness.get("clean_simulation", True))
    store_candidate_pools = bool(harness.get("store_candidate_pools", False))
    default_n_cells = base["n_clusters"] if clean_simulation else base["n_cells"]
    n_cells = int(harness.get("n_cells", default_n_cells))
    if include_sergio and n_cells < base["n_clusters"]:
        raise ValueError(
            f"n_cells must be >= n_clusters ({base['n_clusters']}), got {n_cells}"
        )
    mr_jitter_std = float(harness.get("mr_jitter_std", 0.0))
    if mr_jitter_std < 0:
        raise ValueError("mr_jitter_std must be non-negative")
    scenario_names = ["surrogate"]
    if include_sergio:
        scenario_names.append("sergio")

    log(
        f"[setup] n_mrs={len(mr_ids)} arms={len(harness['arms'])} "
        f"replicates={n_replicates} include_sergio={include_sergio} "
        f"clean_simulation={clean_simulation} n_cells={n_cells}"
    )
    log(
        f"[setup] estimated evaluations: "
        f"{len(harness['arms']) * n_replicates} surrogate + "
        f"{len(harness['arms']) * n_replicates if include_sergio else 0} SERGIO"
    )

    arm_results = {}
    try:
        for arm in harness["arms"]:
            name = arm.get("name")
            if not name or name in arm_results:
                raise ValueError(f"each arm needs a unique non-empty name, got {name!r}")
            arm_start = time.perf_counter()
            log(
                f"[arm {len(arm_results) + 1}/{len(harness['arms'])}] {name}: "
                f"candidate={arm['candidate_method']} "
                f"selection={arm['selection_method']}"
            )
            replicates = []
            for replicate in tqdm(
                range(n_replicates),
                desc=f"{name}",
                unit="rep",
                disable=not verbose,
            ):
                replicate_start = time.perf_counter()
                candidate_seed  = seed_base          + replicate
                selection_seed  = seed_base + 10_000 + replicate
                simulation_seed = seed_base + 20_000 + replicate
                log(
                    f"  [replicate {replicate + 1}/{n_replicates}] "
                    f"building candidate pool and selecting states"
                )
                candidate_start = time.perf_counter()
                pool, candidate_metadata = _candidate_pool(
                    arm, base, trees, list(mr_ids), candidate_seed)
                candidate_seconds = time.perf_counter() - candidate_start
                log(
                    f"    candidate pool ready: shape={tuple(pool.shape)} "
                    f"elapsed={candidate_seconds:.1f}s; selecting"
                )
                selection_start = time.perf_counter()
                selected, selection_metadata = _select_states(
                    arm, pool, dag, base, selection_seed)
                selection_seconds = time.perf_counter() - selection_start
                log(
                    f"    selection ready: {selected.shape[0]} states "
                    f"elapsed={selection_seconds:.1f}s; evaluating surrogate"
                )
                if selected.shape[0] != base["n_clusters"]:
                    raise ValueError(
                        f"arm {name!r} selected {selected.shape[0]} states but base config "
                        f"requires n_clusters={base['n_clusters']}"
                    )
                if mr_jitter_std:
                    rng = np.random.default_rng(simulation_seed + 1)
                    scale = float(base["mr_rate_high"] - base["mr_rate_low"])
                    evaluated_states = selected + rng.normal(
                        0.0, mr_jitter_std * scale, size=selected.shape)
                    evaluated_states = np.clip(
                        evaluated_states, base["mr_rate_low"], base["mr_rate_high"]
                    ).astype(np.float32)
                else:
                    evaluated_states = selected.copy()

                surrogate_start = time.perf_counter()
                surrogate_raw = sergio_dag_hill_forward(
                    evaluated_states, dag, decays=base["decays"])
                surrogate_log = torch.tensor(
                    np.log1p(np.maximum(surrogate_raw, 0.0)),
                    dtype=torch.float32,
                    device=device,
                )
                _, surrogate_metrics = sergio_spectral_metrics(surrogate_raw)
                surrogate_labels = np.arange(evaluated_states.shape[0], dtype=np.int64)
                surrogate_record = _evaluate_matrix(
                    surrogate_log, evaluated_states, surrogate_labels, base,
                    target_stats, tsd, surrogate_metrics=surrogate_metrics,
                    winner_mr_by_gene=winner_mr_by_gene,
                    mr_ids=list(mr_ids),
                    gene_id_to_symbol=gene_id_to_symbol,
                )
                surrogate_seconds = time.perf_counter() - surrogate_start
                scenarios = {"surrogate": surrogate_record}

                sergio_seconds = None
                if include_sergio:
                    log(
                        f"    running SERGIO: n_cells={n_cells} "
                        f"clean={clean_simulation}"
                    )
                    sergio_start = time.perf_counter()
                    X_sim, labels, simulation_metadata = _simulate_arm(
                        evaluated_states,
                        base,
                        temp_path,
                        list(mr_ids),
                        simulation_seed,
                        clean_simulation,
                        n_cells,
                        device,
                    )
                    sergio_seconds = time.perf_counter() - sergio_start
                    sergio_record = _evaluate_matrix(
                        X_sim, evaluated_states, labels, base, target_stats, tsd,
                        surrogate_metrics=surrogate_metrics,
                        winner_mr_by_gene=winner_mr_by_gene,
                        mr_ids=list(mr_ids),
                        gene_id_to_symbol=gene_id_to_symbol,
                    )
                    sergio_record["simulation_metadata"] = simulation_metadata
                    scenarios["sergio"] = sergio_record

                replicate_result = {
                    "replicate": replicate,
                    "candidate_seed": candidate_seed,
                    "selection_seed": selection_seed,
                    "simulation_seed": simulation_seed,
                    "candidate_pool_shape": list(pool.shape),
                    "candidate_metadata": candidate_metadata,
                    "selection_metadata": selection_metadata,
                    "timing_seconds": {
                        "candidate": candidate_seconds,
                        "selection": selection_seconds,
                        "surrogate": surrogate_seconds,
                        "sergio": sergio_seconds,
                        "total": time.perf_counter() - replicate_start,
                    },
                    "mr_state": selected,
                    "evaluated_mr_state": evaluated_states,
                    "scenarios": scenarios,
                }
                if store_candidate_pools:
                    replicate_result["candidate_pool"] = pool
                replicates.append(replicate_result)
                log(
                    f"  [replicate {replicate + 1}/{n_replicates}] complete in "
                    f"{replicate_result['timing_seconds']['total']:.1f}s "
                    f"(candidate={candidate_seconds:.1f}s, "
                    f"selection={selection_seconds:.1f}s, "
                    f"surrogate={surrogate_seconds:.1f}s"
                    + (f", sergio={sergio_seconds:.1f}s" if sergio_seconds is not None else "")
                    + ")"
                )
            arm_results[name] = {
                "spec": arm,
                "replicates": replicates,
                "summary": _summarize_arm(replicates, scenario_names),
            }
            log(
                f"[arm {len(arm_results)}/{len(harness['arms'])}] {name} complete in "
                f"{time.perf_counter() - arm_start:.1f}s"
            )

        baseline_name = harness.get("baseline", harness["arms"][0]["name"])
        if baseline_name not in arm_results:
            raise ValueError(f"baseline arm {baseline_name!r} is not present")
        result = {
            "schema_version": 1,
            "harness_config": harness,
            "harness_config_path": harness_source,
            "base_config_source": base_source,
            "base_config": base,
            "grn": {
                "mr_ids": list(mr_ids),
                "gene_id_to_symbol": gene_id_to_symbol,
            "diagnostics": grn_diagnostics,
            "grn_seed": base["grn_seed"],
            "sha256": grn_digest.hexdigest(),
        },
            "scenario_names": scenario_names,
            "baseline": baseline_name,
            "arms": arm_results,
            "comparison": {
                "paired_deltas_vs_baseline": _paired_deltas(
                    arm_results, baseline_name, scenario_names),
            },
        }
    finally:
        if harness.get("grn_output_path") is None and os.path.exists(temp_path):
            os.remove(temp_path)

    output = harness.get("output")
    if output:
        _write_results(result, output)
        log(f"[output] wrote {output}")
        log(f"[output] wrote {Path(output).with_suffix('.summary.json')}")
        log(f"[output] wrote {Path(output).with_suffix('.summary.csv')}")
    log(f"[done] total elapsed={time.perf_counter() - run_start:.1f}s")
    return result


def _write_results(result: dict, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)

    json_path = path.with_suffix(".summary.json")
    summary = {
        "schema_version": result["schema_version"],
        "harness_config_path": result.get("harness_config_path"),
        "base_config_source": result.get("base_config_source"),
        "baseline": result["baseline"],
        "scenario_names": result["scenario_names"],
        "arms": {
            name: value["summary"]
            for name, value in result["arms"].items()
        },
        "comparison": result["comparison"],
    }
    with open(json_path, "w") as f:
        json.dump(_jsonify(summary), f, indent=2, allow_nan=True)

    csv_path = path.with_suffix(".summary.csv")
    rows = []
    for arm_name, arm in result["arms"].items():
        for scenario, scenario_summary in arm["summary"]["scenarios"].items():
            for metric, values in scenario_summary["metrics"].items():
                rows.append({
                    "arm": arm_name,
                    "scenario": scenario,
                    "metric": metric,
                    **values,
                })
    fields = ["arm", "scenario", "metric", "mean", "std", "median", "q10", "q90", "n"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_jsonify(rows))


def load_mr_state_result(path: str) -> dict:
    with open(path, "rb") as f:
        result = pickle.load(f)
    if result.get("schema_version") != 1:
        raise ValueError(f"unsupported MR-state result schema: {result.get('schema_version')!r}")
    return result


def compare_mr_state_results(paths: list[str]) -> dict:
    """Combine stored result files into one comparison summary."""
    if not paths:
        raise ValueError("at least one result path is required")
    loaded = [load_mr_state_result(path) for path in paths]
    arms = {}
    scenario_names = sorted({
        scenario_name
        for result in loaded
        for scenario_name in result.get("scenario_names", [])
    })
    for path, result in zip(paths, loaded):
        for name, arm in result["arms"].items():
            label = name if len(paths) == 1 else f"{Path(path).stem}:{name}"
            arms[label] = arm["summary"]
    return {
        "schema_version": 1,
        "sources": paths,
        "baseline": loaded[0].get("baseline"),
        "scenario_names": scenario_names,
        "arms": arms,
    }


def plot_mr_state_comparison(result_or_path, path: str | None = None, scenario: str | None = None):
    """Plot fit, rank, and canonical split-half stability metrics."""
    import matplotlib.pyplot as plt

    result = load_mr_state_result(result_or_path) if isinstance(result_or_path, str) else result_or_path
    scenario_names = result.get("scenario_names")
    if not scenario_names:
        scenario_names = sorted({
            scenario_name
            for arm in result["arms"].values()
            for scenario_name in arm.get("scenarios", {})
        })
    if not scenario_names:
        raise ValueError("comparison result contains no scenarios")
    scenario = scenario or scenario_names[-1]
    arm_names = list(result["arms"])
    x = np.arange(len(arm_names))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    panels = (
        ("distance", "distance", True),
        ("mr_state_participation_ratio", "MR-state effective rank", False),
        ("label_between_variance_fraction", "label-explained variance", False),
        ("label_centroid_participation_ratio", "label-program effective rank", False),
        ("label_holdout_accuracy", "label holdout accuracy", False),
        ("pca_size_normalized_standardized_split_half_subspace_stability", "loading subspace stability", False),
        ("pca_size_normalized_standardized_split_half_spectrum_similarity", "PCA spectrum stability", False),
        ("nonzero_frac", "nonzero fraction", False),
    )
    for ax, (key, title, lower_is_better) in zip(axes.flat, panels):
        means = []
        errors = []
        for name in arm_names:
            arm = result["arms"][name]
            arm_summary = arm.get("summary", arm)
            values = arm_summary.get("scenarios", {}).get(scenario, {}).get("metrics", {}).get(key, {})
            means.append(values.get("mean", float("nan")))
            errors.append(values.get("std", float("nan")))
        ax.errorbar(x, means, yerr=errors, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(arm_names, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle(f"MR-state comparison ({scenario})")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="multi-arm harness JSON config")
    parser.add_argument(
        "--compare-results", nargs="+", default=None,
        help="compare one or more stored MR-state result pickles instead of running",
    )
    parser.add_argument("--output", default=None, help="override the config output path")
    parser.add_argument("--plot", default=None, help="optional comparison plot path")
    args = parser.parse_args(argv)
    if args.compare_results:
        comparison = compare_mr_state_results(args.compare_results)
        if args.plot:
            plot_mr_state_comparison(comparison, path=args.plot)
        print(json.dumps(_jsonify(comparison), indent=2, allow_nan=True))
        return comparison
    if args.config is None:
        parser.error("--config is required unless --compare-results is supplied")
    with open(args.config) as f:
        config = json.load(f)
    if args.output is not None:
        config["output"] = args.output
    result = run_mr_state_comparison(config)
    plot_path = args.plot or config.get("plot")
    if plot_path:
        plot_mr_state_comparison(result, path=plot_path)
    for name, arm in result["arms"].items():
        scenario = result["scenario_names"][-1]
        distance = arm["summary"]["scenarios"][scenario]["metrics"]["distance"]["mean"]
        stability = arm["summary"]["scenarios"][scenario]["metrics"][
            "pca_size_normalized_standardized_split_half_subspace_stability"]["mean"]
        print(f"{name}: {scenario} distance={distance:.6f} loading_stability={stability:.4f}")
    return result


if __name__ == "__main__":
    main()
