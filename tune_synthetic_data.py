"""
Optuna-based tuning of make_synthetic_data6's simulation parameters
(synthetic_data.py) against a real scRNA-seq reference dataset.

Usage
-----
As a script:

    python tune_synthetic_data.py --config synthetic_tuning_config.json

From a notebook:

    import tune_synthetic_data
    study = tune_synthetic_data.run("synthetic_tuning_config.json")

What this does
---------------
Loads a real reference matrix via synthetic_data.load_reference_h5ad(),
computes its compute_summary_stats() once (`target_stats`), then runs one
Optuna study where each trial:
  1. suggests a subset of make_synthetic_data6's/
     generate_sergio_grn_from_reference's simulation parameters (see
     "Fixed vs tunable parameters" below) from config["search_space"];
  2. regenerates the SERGIO GRN-structure file from config["reference_grn_path"]
     (e.g. TRRUST) via generate_sergio_grn_from_reference, with the trial's
     suggested K/Hill-coefficient distribution and unknown_mode_repressor_prob;
  3. samples an mr_state (master-regulator basal production rate) matrix;
  4. runs make_synthetic_data6(..., missing_rate=0.0) to produce a candidate
     synthetic matrix;
  5. computes compute_summary_stats() on the candidate and scores it against
     `target_stats` via compute_stats_distance() -- a weighted mean relative
     squared error across every summary-stat entry (see "The distance
     metric" below);
  6. returns that scalar distance as the trial's objective (to minimize).

Each SERGIO simulation is a single, ~minutes-long blocking call (no training
epochs), so there is no natural intermediate value to report/prune on
mid-trial -- this harness therefore uses optuna.pruners.NopPruner() by
default; trials are only pruned on hard failures (GRN generation errors,
simulation errors, non-finite distance), tagged via trial.user_attrs
["prune_reason"] (see _prune()), the same convention as tuning.py.

Fixed vs tunable parameters
----------------------------
Structural/shape parameters that determine each trial's runtime
(n_clusters, n_genes, n_cells, sampling_state, dt, ...) are FIXED, read
from top-level config keys -- they are not part of search_space. So are:
  - shared_coop_state: must be <= 0 (validated in load_config), since
    generate_sergio_grn_from_reference always writes per-edge Hill/coop
    coefficients into the GRN file, which make_synthetic_data6 only reads
    when shared_coop_state <= 0 (see that function's docstring).
  - mr_rate_low/mr_rate_high: the Uniform(low, high) range mr_state is
    drawn from (see _sample_mr_state) is fixed, not tuned -- aggregate
    compute_summary_stats() targets don't depend on cluster *identity*
    (cluster_labels is explicitly ignored by that function), so there's
    little to gain from spending search budget on mr_state's own range
    versus the technical-noise/GRN-edge parameters below.
  - grn_seed: fixes which reference-subgraph topology/DAG-order is sampled
    from reference_grn_path, so every trial shares the exact same GRN
    *structure* -- only the K-magnitude/Hill-coefficient/repressor-sign
    parameters written into the file vary per trial. (Confirmed by
    reading generate_sergio_grn_from_reference's code: topology-determining
    RNG draws -- subgraph sampling, isolated-node repair -- all happen
    before the K/Hill-coefficient RNG draws in that function's single RNG
    stream, so fixing the seed while varying k_dist/hill_coeff_dist/
    unknown_mode_repressor_prob cannot perturb the topology.)
  - sim_seed: fixes make_synthetic_data6's own randomness (cluster-size
    split, cell subsampling/shuffling, MCAR mask -- unused here since
    missing_rate=0.0 during tuning) and mr_state's sampling seed, so every
    trial's *only* source of variation is the search_space parameters
    themselves -- deterministic, low-noise objective landscape, which
    matters given each trial costs minutes.
  - missing_rate is forced to 0.0 for every trial regardless of config
    (see synthetic_data.py's own module-level note above compute_summary_
    stats: a real reference has no NaNs, so a fair comparison requires the
    synthetic candidate to have none either). config["final_missing_rate"]
    is only used by regenerate_best(), after tuning is done.

The following ARE tunable via config["search_space"] (any subset; omitted
keys fall back to make_synthetic_data6's/generate_sergio_grn_from_reference's
own defaults -- see _SIM_KWARG_DEFAULTS/_GRN_KWARG_DEFAULTS below):
    "noise_params"       : make_synthetic_data6's SERGIO noise amplitude
    "decays"              : make_synthetic_data6's mRNA decay rate
    "cluster_conc"        : Dirichlet concentration for cluster-size split
    "outlier_prob"        : sim.outlier_effect() probability
    "outlier_mean"        : sim.outlier_effect() lognormal mean
    "outlier_scale"       : sim.outlier_effect() lognormal scale
    "lib_size_mean"       : sim.lib_size_effect() mean -- directly
                            comparable to target_stats' log_lib_size_mean
    "lib_size_scale"      : sim.lib_size_effect() scale -- comparable to
                            target_stats' log_lib_size_std
    "dropout_shape"       : sim.dropout_indicator() shape
    "dropout_percentile"  : sim.dropout_indicator() percentile
    "grn_k_low"           : lower bound of the GRN interaction-strength
                            magnitude distribution (k_dist=('uniform',
                            grn_k_low, grn_k_low + grn_k_span))
    "grn_k_span"          : k_dist's range width (grn_k_high = grn_k_low
                            + grn_k_span, kept positive by construction)
    "hill_coeff"          : per-edge Hill/cooperativity coefficient
                            (hill_coeff_dist=('constant', hill_coeff))
    "unknown_mode_repressor_prob" : P(repressor) for sign-ambiguous edges
                            in the reference GRN

Each search_space entry is {"type": "float"|"int"|"categorical", ...}, the
same spec format as tuning.py's suggest_from_spec.

The distance metric
---------------------
compute_stats_distance(target_stats, candidate_stats, weights=None,
eps=1e-6) iterates over every key present in both stats dicts (excluding
"n_cells"/"n_genes"/"dropout_curve_bin_edges" -- structural/x-axis entries,
not distributional targets to match), and for each:
  - scalar entries: relative squared error ((cand - target) / (|target| +
    eps)) ** 2;
  - array entries (dropout_curve_zero_frac, pca_explained_variance_ratio):
    mean relative squared error, elementwise over the overlapping length
    (NaNs and length mismatches are skipped per-element/per-key, not
    treated as errors -- dropout_curve_zero_frac's bins are both
    quantile-of-gene-mean deciles, so comparing them index-wise is
    meaningful even though the underlying bin_edges values differ between
    target and candidate).
Each key's term is combined into a single weighted MEAN (not sum) across
all included keys, so the result stays comparable across trials even if a
different subset of keys ends up skipped (e.g. mean_var_log_slope/corr are
NaN whenever too few genes have positive mean & variance). Per-key weights
default to 1.0 and can be overridden via config["weights"] (e.g. {"zero_frac":
2.0} to emphasize matching the overall dropout rate).

Hard guards against degenerate candidates
--------------------------------------------
compute_stats_distance's weighted MEAN averages ~29 individual stat keys,
most of which (gene_mean_*/gene_var_*/log_lib_size_*'s mean/std/percentile
entries) describe closely related aspects of the same few underlying
quantities. This means a candidate can be catastrophically wrong on one or
two structurally important stats (e.g. most cells losing their entire
library to dropout, or the gene-gene correlation structure collapsing onto
a single dominant axis) while still posting a competitive overall
distance, because the many other well-matched terms dilute the damage in
the average -- observed in practice tuning against this same target/search
space (see markup_sessions notes). To guard against this, run_trial()
additionally applies two hard pass/fail checks, independent of the
weighted-mean distance, via _prune() (excludes the trial from
best_value/best_trial entirely, unlike a merely-large distance term):
  - candidate_stats["lib_size_zero_frac"] (fraction of cells with ~0 total
    counts) must be <= config["max_lib_size_zero_frac"] (default 0.05).
  - candidate_stats["pca_explained_variance_ratio"][0] (fraction of
    variance in the leading PC) must be <=
    config["max_pca_pc1_ratio_factor"] * target_stats
    ["pca_explained_variance_ratio"][0] (default factor 1.3), i.e. the
    candidate's dominant-axis concentration can't exceed the reference's
    own by more than 30%.
Both are tagged via trial.user_attrs["prune_reason"] ("degenerate_
lib_size_zero_frac"/"degenerate_pca_pc1_dominance") for post-hoc analysis,
same convention as grn_generation_failed/simulation_failed/non_finite_
distance.

Problem/JSON config schema
----------------------------
Required keys: "reference_grn_path", "search_space", and either
"reference_h5ad_path" or "target_stats_path" (see below).
See synthetic_tuning_config.example.json for a full example. Notable keys
(see _CONFIG_DEFAULTS for the rest / defaults):
    target_stats_path     : str|None, default None -- path to a pickle file
                            containing a precomputed compute_summary_stats()
                            dict (`target_stats`) to tune against directly,
                            e.g. one averaged across several
                            load_reference_h5ad() runs. When set,
                            reference_h5ad_path/reference_layer/
                            reference_n_top_genes/
                            reference_n_cells_subsample/reference_chunk_size/
                            reference_seed are all ignored -- run() skips
                            load_reference_h5ad + compute_summary_stats
                            entirely and just unpickles this file, so
                            reference_h5ad_path need not even be present in
                            the config. Must unpickle to a dict with the
                            same keys/shapes compute_summary_stats() returns
                            (see that function's docstring) -- e.g.:

                                import pickle
                                import numpy as np
                                from synthetic_data import (
                                    generate_sergio_grn_from_reference,
                                    load_reference_h5ad,
                                    compute_summary_stats,
                                )

                                # Align the reference's gene panel by *identity* with the
                                # synthetic GRN's genes (rather than an independent
                                # top-n_top_genes-by-expression selection) -- both sides
                                # then describe the same genes, not just the same gene
                                # count. grn_seed is fixed, so this subgraph (and the
                                # resulting select_gene_symbols) is stable across an
                                # entire tuning study -- a one-time step.
                                _, _, gene_id_to_symbol = generate_sergio_grn_from_reference(
                                    reference_grn_path, n_genes=800,
                                    output_path="/tmp/grn_for_target_stats.csv", seed=42,
                                )
                                select_gene_symbols = list(gene_id_to_symbol.values())

                                stats_list = [
                                    compute_summary_stats(
                                        load_reference_h5ad(
                                            path, select_gene_symbols=select_gene_symbols, seed=i,
                                        )[0]
                                    )
                                    for i in range(20)
                                ]
                                # Average across runs. Array-valued entries (e.g.
                                # dropout_curve_zero_frac) can come back jagged across
                                # samples (different numbers of non-empty mean-expression
                                # bins per draw) -- truncate to the shortest length across
                                # samples before averaging elementwise; scalar entries are
                                # a plain mean.
                                avg_stats = {}
                                for k in stats_list[0]:
                                    if isinstance(stats_list[0][k], np.ndarray):
                                        min_len = min(len(s[k]) for s in stats_list)
                                        avg_stats[k] = np.mean(
                                            np.array([s[k][:min_len] for s in stats_list]), axis=0,
                                        )
                                    else:
                                        avg_stats[k] = np.mean([s[k] for s in stats_list])
                                with open("target_stats.pkl", "wb") as f:
                                    pickle.dump(avg_stats, f)
    reference_h5ad_path   : str, required unless target_stats_path is set --
                            path to the real reference .h5ad file (see
                            synthetic_data.load_reference_h5ad).
    reference_grn_path    : str, required -- path to a reference GRN edge
                            list (e.g. TRRUST's rawdata.tsv).
    reference_layer       : str|None, default None -- forwarded to
                            load_reference_h5ad's `layer`.
    reference_n_top_genes : int|None, default: same as n_genes -- forwarded
                            to load_reference_h5ad's `n_top_genes`, so the
                            reference and every candidate have matching
                            gene counts (comparable PCA/gene-gene-corr stats).
    reference_n_cells_subsample/reference_chunk_size/reference_seed :
                            forwarded to load_reference_h5ad's
                            `n_cells_subsample`/`chunk_size`/`seed` (backed-mode
                            streaming params -- see that function's docstring;
                            defaults match its own defaults).
    n_clusters/n_genes/n_cells/sampling_state/shared_coop_state/dt/
    noise_type/min_cells_per_cluster/add_outlier_genes/add_lib_size_effect/
    add_dropout/convert_to_umi_counts : fixed make_synthetic_data6 shape/
                            toggle parameters, forwarded unchanged to every
                            trial.
    mr_rate_low/mr_rate_high : fixed mr_state sampling range (see
                            "Fixed vs tunable parameters" above).
    grn_seed/sim_seed      : fixed seeds (see "Fixed vs tunable parameters").
    grn_delimiter/grn_regulator_col/grn_target_col/grn_mode_col/
    grn_activation_labels/grn_repression_labels/grn_max_seed_attempts :
                            forwarded to generate_sergio_grn_from_reference
                            (defaults match TRRUST's rawdata TSV format).
    grn_tmp_dir            : str|None, default: system temp dir -- where
                            each trial's regenerated GRN CSV is written
                            (removed again after that trial's simulation).
    stats_n_pca_components/stats_n_structure_genes/stats_percentiles/stats_seed :
                            forwarded to compute_summary_stats (n_structure_genes
                            controls the shared gene subsample used for both
                            PCA and gene-gene correlation -- see that
                            function's docstring), applied identically to the
                            reference and every candidate.
    weights/distance_eps   : compute_stats_distance's config (see above).
    max_lib_size_zero_frac : float, default 0.05 -- hard guard (see "Hard
                            guards against degenerate candidates" below):
                            a trial is pruned if candidate_stats
                            ["lib_size_zero_frac"] exceeds this.
    max_pca_pc1_ratio_factor : float, default 1.3 -- hard guard: a trial is
                            pruned if candidate_stats
                            ["pca_explained_variance_ratio"][0] exceeds
                            this factor times target_stats's own PC1 ratio.
    storage                : str, default "sqlite:///synthetic_tuning.db" --
                            same sqlite-resumability pattern as tuning.py;
                            run this script from multiple processes against
                            the same storage for wall-clock parallelism
                            (SERGIO's simulate() is CPU-bound single-process
                            work, so Optuna's own n_jobs threading doesn't
                            help here -- separate OS processes do).
    study_name             : str, default "tune_synthetic_data6".
    n_trials               : int, default 30 (~3.5h at the <7min/iteration
                            budget this harness was designed around --
                            adjust to taste).
    sampler/pruner         : "tpe" (only option) / "none" (NopPruner, only
                            option -- see "no mid-trial pruning" above).
    output                 : str, default "tuned_synthetic_config.json" --
                            where the final {best_value, best_params,
                            best_resolved_params, n_mrs, "_meta": {...}}
                            summary is written (see run() below).
    artifact_dir           : str, default "synthetic_checkpoints" -- every
                            time a trial improves on the best distance seen
                            so far, its full synthetic dataset (X,
                            cluster_labels, mr_state, mr_ids,
                            gene_id_to_symbol, resolved_params,
                            candidate_stats) is saved to
                            {artifact_dir}/best.pt (see _save_if_best_artifact),
                            mirroring tuning.py's live checkpointing.
    final_missing_rate     : float, default 0.1 -- only used by
                            regenerate_best(), see below.

Accessing the best synthetic dataset
--------------------------------------
Two ways to get the actual best-found synthetic dataset (not just its
hyperparameters):
  1. {artifact_dir}/best.pt is kept up to date live during tuning (see
     above) -- already contains the exact best trial's X/cluster_labels
     (with missing_rate=0.0, since that's what tuning itself used).
  2. regenerate_best(study_or_params, config, final_missing_rate=...) 
     re-runs GRN generation + make_synthetic_data6 from a finished trial's
     params (or a study's study.best_params), optionally with a nonzero
     `missing_rate` this time, for producing an actual usable train/eval
     dataset once tuning is done::

         import json, tune_synthetic_data as tsd
         config = tsd.load_config("synthetic_tuning_config.json")
         tuned  = json.load(open(config["output"]))
         X, cluster_labels, mr_state, grn_path, mr_ids, gene_id_to_symbol = (
             tsd.regenerate_best(tuned["best_params"], config, final_missing_rate=0.1)
         )
         # e.g. slice X into train/test and torch.save in the schema
         # tuning.py's run() expects (see its "Problem file schema" section):
         #   {'n_genes': X.shape[1], 'n_cells': X.shape[0],
         #    'gene_reads_train': X[:n_train], 'gene_reads_test': X[n_train:]}
         # to feed the tuned synthetic generator directly into VAE tuning.

Dependencies note
-------------------
Like synthetic_data.py's own make_synthetic_data6/load_reference_h5ad, this
module depends on the `SERGIO` and `anndata` packages (not pip-installable/
not installed in this dev environment -- see those functions' docstrings),
plus `optuna`. It can only be syntax-checked here, not executed.
"""

import argparse
import json
import math
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from tqdm.auto import tqdm

from synthetic_data import (
    compute_summary_stats,
    generate_sergio_grn_from_reference,
    load_reference_h5ad,
    make_synthetic_data6,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make_synthetic_data6 kwargs that are tunable here -- kept in sync with
# that function's own signature defaults, so an omitted search_space key
# simply falls back to make_synthetic_data6's own default value.
_SIM_KWARG_DEFAULTS = {
    "noise_params":       1.0,
    "decays":             0.8,
    "cluster_conc":       20.0,
    "outlier_prob":       0.01,
    "outlier_mean":       0.8,
    "outlier_scale":      1.0,
    "lib_size_mean":      4.6,
    "lib_size_scale":     0.4,
    "dropout_shape":      6.5,
    "dropout_percentile": 82.0,
}

# generate_sergio_grn_from_reference kwargs that are tunable here (via a
# derived k_dist/hill_coeff_dist -- see _split_resolved) -- kept in sync
# with that function's own k_dist=('uniform', 1.0, 5.0) and
# hill_coeff_dist=('constant', 2.0) defaults (grn_k_low=1.0, grn_k_span=4.0
# => k_dist=('uniform', 1.0, 5.0)).
_GRN_KWARG_DEFAULTS = {
    "grn_k_low":                   1.0,
    "grn_k_span":                  4.0,
    "hill_coeff":                  2.0,
    "unknown_mode_repressor_prob": 0.5,
}

TUNABLE_KEYS = frozenset(_SIM_KWARG_DEFAULTS) | frozenset(_GRN_KWARG_DEFAULTS)

# compute_summary_stats entries excluded from compute_stats_distance:
# structural (not distributional) entries, not stats to be matched.
_EXCLUDED_STATS_KEYS = frozenset({"n_cells", "n_genes", "dropout_curve_bin_edges"})

_SAMPLERS = {
    "tpe": lambda: optuna.samplers.TPESampler(seed=0),
}

_PRUNERS = {
    # No natural mid-trial intermediate value (single-shot SERGIO
    # simulation, no epochs) -- see module docstring's "no mid-trial
    # pruning" section. Trials are only pruned on hard failures via _prune().
    "none": lambda: optuna.pruners.NopPruner(),
}

_CONFIG_DEFAULTS = {
    # structural / fixed simulation shape (matches the <7min/iteration
    # config this harness was designed around)
    "n_clusters":            6,
    "n_genes":                800,
    "n_cells":                600,
    "sampling_state":         5,
    "shared_coop_state":      0.0,
    "dt":                     0.01,
    "noise_type":             "dpd",
    "min_cells_per_cluster":  1,
    "add_outlier_genes":      True,
    "add_lib_size_effect":    True,
    "add_dropout":            True,
    "convert_to_umi_counts":  True,

    # mr_state range -- fixed, not tuned (see module docstring)
    "mr_rate_low":  1.0,
    "mr_rate_high": 5.0,

    # fixed seeds -- see module docstring's "Fixed vs tunable parameters"
    "grn_seed": 42,
    "sim_seed": 0,

    # reference GRN parsing (TRRUST rawdata.tsv format defaults)
    "grn_delimiter":         "\t",
    "grn_regulator_col":     0,
    "grn_target_col":        1,
    "grn_mode_col":          2,
    "grn_activation_labels": ["Activation"],
    "grn_repression_labels": ["Repression"],
    "grn_max_seed_attempts": 20,
    "grn_tmp_dir":           None,  # default: system temp dir

    # if set, skip load_reference_h5ad + compute_summary_stats entirely and
    # load target_stats directly from this pickle file (see run()'s
    # target_stats_path handling below) -- e.g. a compute_summary_stats()
    # dict pre-averaged over multiple load_reference_h5ad() runs, so the
    # (possibly expensive/streaming) reference analysis need not be redone
    # on every run() call. When set, reference_h5ad_path/reference_layer/
    # reference_n_top_genes/reference_n_cells_subsample/
    # reference_chunk_size/reference_seed are ignored.
    "target_stats_path": None,

    # reference h5ad loading (see load_reference_h5ad's docstring for the
    # backed-mode streaming behavior these control)
    "reference_layer":             None,
    "reference_n_top_genes":       None,  # default: same as n_genes, see load_config
    "reference_n_cells_subsample": 50_000,
    "reference_chunk_size":        5_000,
    "reference_seed":              0,

    # compute_summary_stats config -- applied identically to target & every candidate
    "stats_n_pca_components": 10,
    "stats_n_structure_genes": 500,
    "stats_percentiles":      [5, 25, 50, 75, 95],
    "stats_seed":             0,

    # compute_stats_distance config
    "weights":      {},
    "distance_eps": 1e-6,

    # Hard guards against degenerate/collapsed candidates -- checked in
    # run_trial() right after compute_summary_stats(), independent of (and
    # in addition to) compute_stats_distance()'s soft weighted terms. A
    # trial that violates either is pruned via _prune() (excluded from
    # best_value/best_trial entirely, tagged with a specific prune_reason),
    # rather than merely incurring one more term in an averaged distance
    # that a handful of well-matching terms elsewhere can dilute away (see
    # module docstring's "Hard guards" note).
    "max_lib_size_zero_frac":   0.05,  # candidate's lib_size_zero_frac must be <= this
    "max_pca_pc1_ratio_factor": 1.3,   # candidate PC1 ratio must be <= this * target PC1 ratio

    # Optuna
    "storage":      "sqlite:///synthetic_tuning.db",
    "study_name":   "tune_synthetic_data6",
    "n_trials":     30,
    "sampler":      "tpe",
    "pruner":       "none",
    "output":       "tuned_synthetic_config.json",
    "artifact_dir": "synthetic_checkpoints",

    # regenerate_best() only -- ignored during tuning itself (missing_rate
    # is forced to 0.0 for every trial, see module docstring)
    "final_missing_rate": 0.1,
}


def load_config(path: str) -> dict:
    """Load and validate the JSON tuning config, merged over _CONFIG_DEFAULTS."""
    with open(path) as f:
        user_config = json.load(f)

    config = {**_CONFIG_DEFAULTS, **user_config}

    required_keys = ["reference_grn_path", "search_space"]
    if not config.get("target_stats_path"):
        required_keys.append("reference_h5ad_path")
    for key in required_keys:
        if key not in config:
            raise ValueError(f"tuning config {path!r} is missing required key {key!r}")

    unknown_keys = set(config["search_space"]) - TUNABLE_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unsupported search_space key(s): {sorted(unknown_keys)}. "
            f"Supported: {sorted(TUNABLE_KEYS)}"
        )

    for name, spec in config["search_space"].items():
        if spec.get("type") not in ("float", "int", "categorical"):
            raise ValueError(
                f"search_space[{name!r}] has unsupported type {spec.get('type')!r} "
                f"(expected 'float', 'int', or 'categorical')"
            )

    if config["shared_coop_state"] > 0:
        raise ValueError(
            "config['shared_coop_state'] must be <= 0 -- "
            "generate_sergio_grn_from_reference always writes per-edge Hill/"
            "cooperativity coefficients, which make_synthetic_data6 only reads "
            "from the GRN file when shared_coop_state <= 0 (see that function's "
            "docstring)"
        )

    if config["reference_n_top_genes"] is None:
        config["reference_n_top_genes"] = config["n_genes"]

    if config["sampler"] not in _SAMPLERS:
        raise ValueError(f"Unsupported sampler {config['sampler']!r}. Supported: {list(_SAMPLERS)}")
    if config["pruner"] not in _PRUNERS:
        raise ValueError(f"Unsupported pruner {config['pruner']!r}. Supported: {list(_PRUNERS)}")

    return config


def suggest_from_spec(trial: optuna.Trial, name: str, spec: dict):
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported search_space type {kind!r} for {name!r}")


def suggest_sim_cfg(trial: optuna.Trial, search_space: dict) -> dict:
    """Resolve every tunable key to a value: suggested via Optuna for keys
    present in search_space, falling back to _SIM_KWARG_DEFAULTS/
    _GRN_KWARG_DEFAULTS (matching the wrapped functions' own defaults) for
    keys left out of search_space."""
    resolved = {**_SIM_KWARG_DEFAULTS, **_GRN_KWARG_DEFAULTS}
    for name, spec in search_space.items():
        resolved[name] = suggest_from_spec(trial, name, spec)
    return resolved


def _split_resolved(resolved: dict):
    """Split a flat resolved-params dict (from suggest_sim_cfg, or from
    {**_SIM_KWARG_DEFAULTS, **_GRN_KWARG_DEFAULTS, **best_params} when
    replaying a finished trial) into make_synthetic_data6's sim_kwargs and
    generate_sergio_grn_from_reference's k_dist/hill_coeff_dist/
    unknown_mode_repressor_prob."""
    sim_kwargs = {k: resolved[k] for k in _SIM_KWARG_DEFAULTS}
    grn_k_dist = ("uniform", resolved["grn_k_low"], resolved["grn_k_low"] + resolved["grn_k_span"])
    hill_coeff_dist = ("constant", resolved["hill_coeff"])
    unknown_mode_repressor_prob = resolved["unknown_mode_repressor_prob"]
    return sim_kwargs, grn_k_dist, hill_coeff_dist, unknown_mode_repressor_prob


def _sample_mr_state(n_clusters: int, n_mrs: int, low: float, high: float, seed: int | None) -> np.ndarray:
    """Sample an (n_clusters, n_mrs) matrix of i.i.d. Uniform(low, high)
    master-regulator basal production rates. make_synthetic_data6 has no
    built-in generator for mr_state -- its docstring only requires values
    >= 0, in a "sane range" like SERGIO's own demo data (~[1, 5]). Fixed
    (low, high, seed) across trials by default (see module docstring) --
    compute_summary_stats' targets don't depend on cluster *identity*
    (cluster_labels is explicitly ignored, per that function's docstring),
    so there's no need for cluster-specific mr_state structure here; every
    cluster row is drawn i.i.d. from the same range."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_clusters, n_mrs))


def compute_stats_distance(target_stats: dict, candidate_stats: dict,
                            weights: dict | None = None, eps: float = 1e-6) -> tuple[float, dict]:
    """Weighted-mean relative squared error between two compute_summary_stats()
    dicts (see module docstring's "The distance metric"). Returns
    (distance, breakdown) where breakdown = {"terms": {key: term, ...},
    "skipped": [key, ...]} for diagnostics -- keys are skipped (not errors)
    when absent from either dict, non-finite, or (for array keys) of zero
    overlapping length."""
    weights = weights or {}
    terms: dict[str, float] = {}
    skipped: list[str] = []

    for key, target_value in target_stats.items():
        if key in _EXCLUDED_STATS_KEYS:
            continue
        if key not in candidate_stats:
            skipped.append(key)
            continue
        candidate_value = candidate_stats[key]

        if isinstance(target_value, np.ndarray) or isinstance(candidate_value, np.ndarray):
            t = np.asarray(target_value, dtype=np.float64)
            c = np.asarray(candidate_value, dtype=np.float64)
            n = min(t.size, c.size)
            if n == 0:
                skipped.append(key)
                continue
            t, c = t[:n], c[:n]
            valid = np.isfinite(t) & np.isfinite(c)
            if not valid.any():
                skipped.append(key)
                continue
            rel_sq = ((c[valid] - t[valid]) / (np.abs(t[valid]) + eps)) ** 2
            terms[key] = float(np.mean(rel_sq))
        else:
            t, c = float(target_value), float(candidate_value)
            if not (math.isfinite(t) and math.isfinite(c)):
                skipped.append(key)
                continue
            terms[key] = float(((c - t) / (abs(t) + eps)) ** 2)

    if not terms:
        return float("inf"), {"terms": terms, "skipped": skipped}

    total_weight = sum(weights.get(k, 1.0) for k in terms)
    distance = sum(weights.get(k, 1.0) * v for k, v in terms.items()) / total_weight
    return float(distance), {"terms": terms, "skipped": skipped}


def _prune(trial: optuna.Trial, reason: str, message: str = "") -> None:
    """Raise optuna.TrialPruned after tagging *why* (trial.user_attrs
    ["prune_reason"]) -- same convention as tuning.py's _prune(), letting
    trial_summary_callback/post-hoc analysis distinguish grn_generation_
    failed/simulation_failed/non_finite_distance rather than lumping every
    PRUNED trial together."""
    trial.set_user_attr("prune_reason", reason)
    raise optuna.TrialPruned(message or reason)


def run_trial(trial: optuna.Trial, config: dict, target_stats: dict) -> float:
    """Build one synthetic candidate from suggested/fixed simulation
    parameters, score it against target_stats via compute_stats_distance,
    and return that scalar distance (to minimize). See module docstring
    for the full pipeline description."""
    resolved = suggest_sim_cfg(trial, config["search_space"])
    trial.set_user_attr("resolved_params", {k: float(v) for k, v in resolved.items()})
    sim_kwargs, grn_k_dist, hill_coeff_dist, unknown_mode_repressor_prob = _split_resolved(resolved)

    grn_tmp_dir = config["grn_tmp_dir"] or tempfile.gettempdir()
    Path(grn_tmp_dir).mkdir(parents=True, exist_ok=True)
    grn_path = os.path.join(grn_tmp_dir, f"sergio_grn_trial{trial.number}_{os.getpid()}.csv")

    try:
        _, mr_ids, gene_id_to_symbol = generate_sergio_grn_from_reference(
            reference_grn_path=config["reference_grn_path"],
            n_genes=config["n_genes"],
            output_path=grn_path,
            delimiter=config["grn_delimiter"],
            regulator_col=config["grn_regulator_col"],
            target_col=config["grn_target_col"],
            mode_col=config["grn_mode_col"],
            activation_labels=config["grn_activation_labels"],
            repression_labels=config["grn_repression_labels"],
            unknown_mode_repressor_prob=unknown_mode_repressor_prob,
            k_dist=grn_k_dist,
            hill_coeff_dist=hill_coeff_dist,
            max_seed_attempts=config["grn_max_seed_attempts"],
            seed=config["grn_seed"],
        )
    except Exception as e:
        _prune(trial, "grn_generation_failed", str(e))

    try:
        mr_state_np = _sample_mr_state(
            config["n_clusters"], len(mr_ids),
            config["mr_rate_low"], config["mr_rate_high"],
            seed=config["sim_seed"],
        )
        mr_state = torch.tensor(mr_state_np, dtype=torch.float32, device=device)

        X_sim, cluster_labels = make_synthetic_data6(
            mr_state=mr_state,
            input_file_targets=grn_path,
            n_cells=config["n_cells"],
            mr_gene_ids=mr_ids,
            shared_coop_state=config["shared_coop_state"],
            noise_type=config["noise_type"],
            sampling_state=config["sampling_state"],
            dt=config["dt"],
            min_cells_per_cluster=config["min_cells_per_cluster"],
            add_outlier_genes=config["add_outlier_genes"],
            add_lib_size_effect=config["add_lib_size_effect"],
            add_dropout=config["add_dropout"],
            convert_to_umi_counts=config["convert_to_umi_counts"],
            missing_rate=0.0,
            seed=config["sim_seed"],
            device=device,
            **sim_kwargs,
        )
    except Exception as e:
        _prune(trial, "simulation_failed", str(e))
    finally:
        if os.path.exists(grn_path):
            os.remove(grn_path)

    candidate_stats = compute_summary_stats(
        X_sim,
        n_pca_components=config["stats_n_pca_components"],
        n_structure_genes=config["stats_n_structure_genes"],
        percentiles=tuple(config["stats_percentiles"]),
        seed=config["stats_seed"],
    )

    # --- hard guards against degenerate/collapsed candidates -- see
    # module docstring's "Hard guards against degenerate candidates" ---
    lib_size_zero_frac = candidate_stats.get("lib_size_zero_frac", float("nan"))
    if math.isfinite(lib_size_zero_frac) and lib_size_zero_frac > config["max_lib_size_zero_frac"]:
        _prune(
            trial, "degenerate_lib_size_zero_frac",
            f"lib_size_zero_frac={lib_size_zero_frac:.4f} > "
            f"max_lib_size_zero_frac={config['max_lib_size_zero_frac']}",
        )

    cand_pca = candidate_stats.get("pca_explained_variance_ratio", np.array([]))
    target_pca = target_stats.get("pca_explained_variance_ratio", np.array([]))
    if len(cand_pca) and len(target_pca):
        max_pc1 = float(target_pca[0]) * config["max_pca_pc1_ratio_factor"]
        if float(cand_pca[0]) > max_pc1:
            _prune(
                trial, "degenerate_pca_pc1_dominance",
                f"pc1_ratio={float(cand_pca[0]):.4f} > "
                f"max_allowed={max_pc1:.4f} (target={float(target_pca[0]):.4f}, "
                f"factor={config['max_pca_pc1_ratio_factor']})",
            )

    distance, breakdown = compute_stats_distance(
        target_stats, candidate_stats, weights=config.get("weights"), eps=config["distance_eps"],
    )
    trial.set_user_attr("skipped_stats_keys", breakdown["skipped"])
    if not math.isfinite(distance):
        _prune(trial, "non_finite_distance")

    trial.set_user_attr("n_mrs", len(mr_ids))
    worst = sorted(breakdown["terms"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    trial.set_user_attr("top_distance_terms", {k: float(v) for k, v in worst})

    _save_if_best_artifact(
        config, trial, distance, X_sim, cluster_labels, mr_state_np,
        mr_ids, gene_id_to_symbol, resolved, candidate_stats,
    )

    return distance


def _save_if_best_artifact(config: dict, trial: optuna.Trial, distance: float,
                            X_sim: torch.Tensor, cluster_labels: np.ndarray,
                            mr_state_np: np.ndarray, mr_ids: list, gene_id_to_symbol: dict,
                            resolved: dict, candidate_stats: dict) -> None:
    """If *distance* improves on whatever is currently recorded in
    {artifact_dir}/best.pt (or that file doesn't exist yet), overwrite it
    with this trial's full synthetic dataset/metadata. Reads the previous
    best straight back from disk (not from Optuna's in-memory study state),
    same resume-safety rationale as tuning.py's _save_if_best_checkpoint."""
    distance = float(distance)
    if not math.isfinite(distance):
        return

    path = Path(config["artifact_dir"]) / "best.pt"
    if path.exists():
        try:
            prev = torch.load(path, map_location="cpu").get("distance", float("inf"))
        except Exception:
            prev = float("inf")
        if distance >= prev:
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "distance":          distance,
        "X":                 X_sim.detach().cpu(),
        "cluster_labels":    cluster_labels,
        "mr_state":          mr_state_np,
        "mr_ids":            mr_ids,
        "gene_id_to_symbol": gene_id_to_symbol,
        "resolved_params":   resolved,
        "candidate_stats":   candidate_stats,
        "params":            trial.params,
        "trial_number":      trial.number,
    }, path)
    tqdm.write(f"  [artifact] new best distance={distance:.6f} -> {path}")


def trial_summary_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """One concise progress line per finished trial (via tqdm.write), same
    convention as tuning.py's trial_summary_callback."""
    if trial.datetime_complete and trial.datetime_start:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
    else:
        duration = float("nan")

    try:
        best = study.best_value
    except ValueError:
        best = None

    value_str = f"{trial.value:.6f}" if trial.value is not None else "n/a"
    best_str = f"{best:.6f}" if best is not None else "n/a"
    reason = trial.user_attrs.get("prune_reason")
    reason_str = f"  reason={reason}" if reason else ""
    tqdm.write(
        f"  trial {trial.number:>4d}  state={trial.state.name:<9s}  "
        f"value={value_str:>12s}  best={best_str:>12s}  ({duration:6.1f}s){reason_str}"
    )


def build_sampler(name: str):
    return _SAMPLERS[name]()


def build_pruner(name: str):
    return _PRUNERS[name]()


def run(config_path: str) -> optuna.Study:
    """Plain entry point, safe to call directly from a notebook cell:

        import tune_synthetic_data
        study = tune_synthetic_data.run("synthetic_tuning_config.json")
        best = study.best_trial

    Persisted in config["storage"] under config["study_name"], so it can be
    reloaded/resumed later via optuna.load_study() even without calling
    this function again -- run this script from multiple processes against
    the same storage for wall-clock parallelism (see module docstring).
    A plain-JSON summary is written to config["output"] for convenience;
    see module docstring's "output" key description.
    """
    config = load_config(config_path)

    # See tuning.py's identical rationale: relies on trial_summary_callback
    # instead of Optuna's noisy default per-trial logging.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if config["target_stats_path"]:
        print(f"Loading precomputed target_stats from {config['target_stats_path']!r} ...")
        with open(config["target_stats_path"], "rb") as f:
            target_stats = pickle.load(f)
    else:
        print(f"Loading reference from {config['reference_h5ad_path']!r} ...")
        target_X, _ = load_reference_h5ad(
            config["reference_h5ad_path"],
            layer=config["reference_layer"],
            n_top_genes=config["reference_n_top_genes"],
            n_cells_subsample=config["reference_n_cells_subsample"],
            chunk_size=config["reference_chunk_size"],
            seed=config["reference_seed"],
            device=device,
        )
        target_stats = compute_summary_stats(
            target_X,
            n_pca_components=config["stats_n_pca_components"],
            n_structure_genes=config["stats_n_structure_genes"],
            percentiles=tuple(config["stats_percentiles"]),
            seed=config["stats_seed"],
        )
    print(
        f"reference: n_cells={target_stats['n_cells']}  n_genes={target_stats['n_genes']}  "
        f"gene_mean_mean={target_stats['gene_mean_mean']:.4f}  "
        f"log_lib_size_mean={target_stats['log_lib_size_mean']:.4f}  "
        f"zero_frac={target_stats['zero_frac']:.4f}"
    )

    study = optuna.create_study(
        study_name=config["study_name"],
        storage=config["storage"],
        load_if_exists=True,
        direction="minimize",
        sampler=build_sampler(config["sampler"]),
        pruner=build_pruner(config["pruner"]),
    )

    target_desc = config["target_stats_path"] or config["reference_h5ad_path"]
    print(
        f"\n=== Tuning synthetic_data6 against {target_desc!r} "
        f"({config['n_trials']} trials) ==="
    )
    start = time.monotonic()
    study.optimize(
        lambda trial: run_trial(trial, config, target_stats),
        n_trials=config["n_trials"],
        show_progress_bar=True,
        callbacks=[trial_summary_callback],
    )
    elapsed = time.monotonic() - start
    print(
        f"=== done in {elapsed / 60:.1f}m -- "
        f"best_value={study.best_value:.6f} best_params={study.best_params} ==="
    )

    summary = {
        "best_value":            study.best_value,
        "best_params":           study.best_params,
        "best_resolved_params":  study.best_trial.user_attrs.get("resolved_params"),
        "n_mrs":                 study.best_trial.user_attrs.get("n_mrs"),
        "_meta": {
            "reference_h5ad_path": config.get("reference_h5ad_path"),
            "target_stats_path":   config["target_stats_path"],
            "reference_grn_path":  config["reference_grn_path"],
            "n_clusters":          config["n_clusters"],
            "n_genes":             config["n_genes"],
            "n_cells":             config["n_cells"],
            "sampling_state":      config["sampling_state"],
            "shared_coop_state":   config["shared_coop_state"],
            "grn_seed":            config["grn_seed"],
            "sim_seed":            config["sim_seed"],
        },
    }
    out_path = Path(config["output"])
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote tuned synthetic-data config summary to {out_path}")

    return study


def regenerate_best(study_or_params, config: dict, final_missing_rate: float | None = None,
                     seed: int | None = None, output_path: str | None = None):
    """Re-run GRN generation + make_synthetic_data6 from a finished study's
    best_params (or an explicit flat params dict, e.g. loaded from
    config["output"]'s "best_params") -- see module docstring's "Accessing
    the best synthetic dataset". Unlike tuning during the study itself,
    missing_rate defaults to config["final_missing_rate"] (not 0.0), since
    the point here is to materialize an actual usable dataset rather than
    a fair-comparison candidate.

    Returns (X, cluster_labels, mr_state, grn_path, mr_ids, gene_id_to_symbol).
    grn_path points at a regenerated GRN CSV (output_path if given, else a
    fixed-name temp file -- NOT cleaned up automatically, unlike run_trial's
    per-trial temp files, since callers may want to inspect/reuse it).
    """
    if isinstance(study_or_params, optuna.Study):
        params = study_or_params.best_params
    else:
        params = study_or_params

    resolved = {**_SIM_KWARG_DEFAULTS, **_GRN_KWARG_DEFAULTS, **params}
    sim_kwargs, grn_k_dist, hill_coeff_dist, unknown_mode_repressor_prob = _split_resolved(resolved)

    grn_tmp_dir = config["grn_tmp_dir"] or tempfile.gettempdir()
    Path(grn_tmp_dir).mkdir(parents=True, exist_ok=True)
    grn_path = output_path or os.path.join(grn_tmp_dir, "sergio_grn_regenerate_best.csv")

    _, mr_ids, gene_id_to_symbol = generate_sergio_grn_from_reference(
        reference_grn_path=config["reference_grn_path"],
        n_genes=config["n_genes"],
        output_path=grn_path,
        delimiter=config["grn_delimiter"],
        regulator_col=config["grn_regulator_col"],
        target_col=config["grn_target_col"],
        mode_col=config["grn_mode_col"],
        activation_labels=config["grn_activation_labels"],
        repression_labels=config["grn_repression_labels"],
        unknown_mode_repressor_prob=unknown_mode_repressor_prob,
        k_dist=grn_k_dist,
        hill_coeff_dist=hill_coeff_dist,
        max_seed_attempts=config["grn_max_seed_attempts"],
        seed=config["grn_seed"],
    )

    mr_state_np = _sample_mr_state(
        config["n_clusters"], len(mr_ids),
        config["mr_rate_low"], config["mr_rate_high"],
        seed=config["sim_seed"],
    )
    mr_state = torch.tensor(mr_state_np, dtype=torch.float32, device=device)

    missing_rate = config["final_missing_rate"] if final_missing_rate is None else final_missing_rate
    sim_seed = config["sim_seed"] if seed is None else seed

    X, cluster_labels = make_synthetic_data6(
        mr_state=mr_state,
        input_file_targets=grn_path,
        n_cells=config["n_cells"],
        mr_gene_ids=mr_ids,
        shared_coop_state=config["shared_coop_state"],
        noise_type=config["noise_type"],
        sampling_state=config["sampling_state"],
        dt=config["dt"],
        min_cells_per_cluster=config["min_cells_per_cluster"],
        add_outlier_genes=config["add_outlier_genes"],
        add_lib_size_effect=config["add_lib_size_effect"],
        add_dropout=config["add_dropout"],
        convert_to_umi_counts=config["convert_to_umi_counts"],
        missing_rate=missing_rate,
        seed=sim_seed,
        device=device,
        **sim_kwargs,
    )

    return X, cluster_labels, mr_state_np, grn_path, mr_ids, gene_id_to_symbol


def main(argv=None) -> optuna.Study:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="synthetic_tuning_config.json",
        help="Path to the JSON tuning config (see synthetic_tuning_config.example.json)",
    )
    args = parser.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    main()
