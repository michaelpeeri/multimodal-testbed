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
  3. samples an mr_state (master-regulator basal production rate) matrix,
     either iid or by spectral selection from a larger candidate pool;
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
  - mr_rate_low/mr_rate_high: the Uniform(low, high) range used to construct
    the MR-state candidate pool is fixed, not tuned.
  - mr_state_method and the n_candidate_states/n_selected_states/
    candidate_design/selection_* settings: fixed study configuration, not
    entries in search_space. ``random`` preserves the historical iid sampler;
    ``spectral`` selects the configured number of rows using the DAG/Hill
    surrogate. Spectral selection requires n_selected_states == n_clusters.
  - grn_seed: fixes which reference-subgraph topology/DAG-order is sampled
    from reference_grn_path, so every trial shares the exact same GRN
    *structure* -- only the K-magnitude/Hill-coefficient/repressor-sign
    parameters written into the file vary per trial. (Confirmed by
    reading generate_sergio_grn_from_reference's code: topology-determining
    RNG draws -- subgraph sampling, isolated-node repair -- all happen
    before the K/Hill-coefficient RNG draws in that function's single RNG
    stream, so fixing the seed while varying k_dist/hill_coeff_dist/
    unknown_mode_repressor_prob cannot perturb the topology.) This only
    holds *within* one process, though -- across separate processes (e.g.
    running several seeds in parallel, see "Usage" above), the same
    grn_seed can still yield a *different* topology because
    _sample_connected_subgraph's node/edge bookkeeping uses plain Python
    sets whose iteration order depends on the interpreter's (by default
    per-process-randomized) string-hash seed -- see AGENTS.md's Known
    Issues, and run()'s startup PYTHONHASHSEED check/warning.
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
    "coherency_bias"      : generate_sergio_grn_from_reference's
                            coherency_bias -- biases sign-ambiguous edges
                            toward agreeing with their target's "winning"
                            master regulator (0.0 = today's unbiased coin
                            flip, matches that function's own default)
    "canalization_strength": generate_sergio_grn_from_reference's
                            canalization_strength -- winner-take-all K-
                            magnitude reweighting toward/away from the
                            target's winning master regulator (0.0 =
                            today's iid magnitude, matches that function's
                            own default). Each target's total K-budget
                            (sum(|K_i|)) is preserved across this
                            reweighting (added 20260816, see AGENTS.md) --
                            so this knob only reallocates which regulator
                            dominates a target's production rate, it does
                            not change that target's own overall
                            production-rate ceiling on average.
    "balancing_strength"  : generate_sergio_grn_from_reference's
                            balancing_strength -- fairness penalty
                            preventing one master regulator's "winning
                            territory" from entrenching (0.0 = pure greedy
                            argmax, matches that function's own default)
    "path_decay"          : generate_sergio_grn_from_reference's
                            path_decay -- per-hop attenuation of
                            propagated master-regulator signal strength
                            (only affects output when coherency_bias > 0
                            or canalization_strength > 0)

Each search_space entry is {"type": "float"|"int"|"categorical", ...}, the
same spec format as tuning.py's suggest_from_spec.

The distance metric
---------------------
compute_stats_distance(target_stats, candidate_stats, weights=None,
eps=1e-6, eps_frac=0.05) iterates over every key present in both stats
dicts (excluding "n_cells"/"n_genes"/"dropout_curve_bin_edges" --
structural/x-axis entries, not distributional targets to match), and for
each:
  - scalar entries: relative squared error ((cand - target) / (|target| +
    key_eps)) ** 2;
  - array entries (dropout_curve_zero_frac, pca_explained_variance_ratio):
    mean relative squared error, elementwise over the overlapping length
    (NaNs and length mismatches are skipped per-element/per-key, not
    treated as errors -- dropout_curve_zero_frac's bins are both
    quantile-of-gene-mean deciles, so comparing them index-wise is
    meaningful even though the underlying bin_edges values differ between
    target and candidate).
key_eps (not just the raw `eps`) guards against near-zero targets: real
reference data legitimately has some target stats at exactly 0.0 (e.g.
gene_mean_p5/gene_var_p5/lib_size_zero_frac are all 0.0 for at least one
real reference sample seen so far), and a bare fixed `eps` makes the plain
relative-error formula numerically explosive for any such key (any small
nonzero candidate value there produces a term of order (candidate/eps)**2,
easily 10^5-10^6 -- confirmed in practice to silently dominate the whole
distance and corrupt trial selection across multiple 20260811.* tuning
studies). key_eps is instead `max(eps, eps_frac * scale)`, where `scale` is
that key's own distributional-family's largest |target value| (see
_stat_family/compute_stats_distance's own docstring) for scalar keys, or
the array's own max(|target|) for array keys -- set eps_frac=0.0 to recover
the original fixed-eps-only behavior.
Note on pca_explained_variance_ratio vs. pca_tail_participation_ratio: the
former matches PC1 (index 0) alongside every other leading PC equally,
which is the wrong target when what you actually care about is how much
variation is spread across PC2+ rather than how dominant PC1 itself is
(PC1 is often driven by structure -- e.g. between-cluster mean shifts --
that's orthogonal to the GRN-topology knobs being tuned). In that case,
set weights["pca_explained_variance_ratio"] = 0.0 and weight
pca_tail_participation_ratio (synthetic_data.compute_summary_stats'
"effective number of components" among PC2..N, see
synthetic_data.pca_participation_ratio) instead -- it's a plain scalar
entry so it flows through the same relative-squared-error/weighting
mechanism as everything else above, with no special-casing needed here.
IMPORTANT: whichever of pca_tail_participation_ratio/pca_explained_variance_ratio
you weight this way, target_stats must have been computed with the *same*
n_pca_components as candidates are (config["stats_n_pca_components"]) --
pca_tail_participation_ratio's value depends on how many tail PCs went into
it, so comparing a target computed at one depth against candidates computed
at another silently compares two different quantities (see run()'s startup
validation, which now checks this explicitly).
Note on zero_frac vs. nonzero_frac (added 20260820, see AGENTS.md): scoring
zero_frac directly badly under-penalizes overshoot (candidate sparser than
target) whenever the target is already close to 1, as real single-cell
reference data's is -- a small absolute zero_frac deviation there
corresponds to a large *relative* change in usable (non-zero) data, which
the plain relative-error-on-zero_frac formula doesn't reflect (see
_add_nonzero_frac_key's own docstring for the worked example and empirical
re-ranking check against real tuning trials). Prefer weighting
nonzero_frac = 1 - zero_frac (added via _add_nonzero_frac_key, called at
both run()/run_trial(), exactly like the pc2_9 keys below) instead, and set
weights["zero_frac"] = 0.0 -- same "retarget the weight at a corrected
metric" convention as pca_standardized_*/gene_corr_abs_normalized_* below.
Note on pca_explained_variance_ratio/pca_tail_participation_ratio/
pca_pc2_9_explained_variance_ratio vs. their pca_standardized_* counterparts
(added 20260816, see AGENTS.md): the plain ("raw") family is PCA on
mean-centered but NOT per-gene-standardized data (see synthetic_data.
compute_summary_stats), so its explained-variance-ratio is driven by which
genes happen to have the largest absolute variance -- it is NOT independent
of gene expression *scale*. This ties it directly to gene_mean_*/gene_var_*/
log_lib_size_* (the same underlying per-gene variances feed both), which
empirically manifested as a hard trade-off between matching PCA structure
and matching gene-expression-magnitude stats (see AGENTS.md's 20260815/
20260816 entries). pca_standardized_* is PCA on each gene standardized to
unit variance first (correlation-matrix PCA), scale-invariant by
construction -- prefer weighting *this* family (as the 20260816-onward
configs do: pca_explained_variance_ratio/pca_tail_participation_ratio/
pca_pc2_9_explained_variance_ratio all at weight 0.0, still computed for
diagnostics/plotting, with pca_standardized_tail_participation_ratio/
pca_standardized_pc2_9_explained_variance_ratio carrying the weight instead)
if you want "match the reference's PCA structure" to not be inherently in
tension with "match the reference's expression magnitude". Same
n_pca_components-matching caveat as above applies to this family too (see
run()'s startup validation, which checks both families independently).
Note on pca_standardized_* vs. pca_size_normalized_standardized_*
counterparts (added 20260820, see AGENTS.md): pca_standardized_*'s per-gene
standardization removes dependence on which genes have the largest
absolute variance, but does NOT remove a per-CELL multiplicative confound
-- SERGIO's lib_size_effect rescales a cell's *entire* gene vector by one
shared lognormal factor, which remains a real, shared source of cross-cell
correlated variation even after standardizing each gene column to unit
variance, so pca_standardized_pc2_9_explained_variance_ratio can still be
inflated as lib_size_scale grows (confirmed empirically: across the real
synthetic_tuning_20260819.* trials, its ratio-to-target rose monotonically
from ~0.7 to ~1.6 as lib_size_scale rose, essentially undiluted by any
other search-space knob in a full rank-regression -- see AGENTS.md).
pca_size_normalized_standardized_* (synthetic_data.compute_summary_stats)
additionally renormalizes each cell's library size before standardizing,
combining both confound-removal steps -- prefer weighting *this* family
(pca_size_normalized_standardized_tail_participation_ratio/
pca_size_normalized_standardized_pc2_9_explained_variance_ratio) over
pca_standardized_*'s if you want PCA-structure fidelity independent of
*both* gene-scale and lib_size_scale. Same n_pca_components-matching
caveat applies (see run()'s startup validation).
Each key's term is combined into a single weighted MEAN (not sum) across
all included keys, so the result stays comparable across trials even if a
different subset of keys ends up skipped (e.g. mean_var_log_slope/corr are
NaN whenever too few genes have positive mean & variance). Per-key weights
default to 1.0 and can be overridden via config["weights"] (e.g. {"zero_frac":
2.0} to emphasize matching the overall dropout rate) -- IMPORTANT: this
default applies to *every* key compute_summary_stats() returns, including
ones you never think about, e.g. leaving the whole gene_mean_*/gene_var_*/
mean_var_log_* family (16 keys) out of config["weights"] doesn't exclude
them, it silently scores them at weight 1.0 each (confirmed in practice to
end up ~25% of the total objective, unintentionally, across the
20260811.* studies) -- explicitly set any key you don't want scored to 0.0.

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
  - candidate_stats[config["pca_pc1_dominance_key"]][0] (fraction of
    variance in the leading PC, of whichever PCA family
    pca_pc1_dominance_key names -- default "pca_explained_variance_ratio",
    the raw family) must be <= config["max_pca_pc1_ratio_factor"] *
    target_stats[config["pca_pc1_dominance_key"]][0] (default factor 1.3),
    i.e. the candidate's dominant-axis concentration can't exceed the
    reference's own by more than 30%. See pca_pc1_dominance_key's own
    docstring entry above for why this should track whichever PCA family
    the objective itself weights, not necessarily the raw default.
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

                                # n_pca_components here MUST match this config's own
                                 # "stats_n_pca_components" (15 in the current
                                 # spectral-selection tuning round -- see that key's
                                 # own entry below) -- pca_tail_participation_ratio's value
                                # depends on how many tail PCs it was computed over, so
                                # a target computed at a different depth than candidates
                                # silently compares two different quantities (this bit
                                # everyone in the 20260811.* studies: the recipe below
                                # previously omitted n_pca_components, silently defaulting
                                # to 10 while those studies' candidates used 20 -- run()'s
                                # startup check now catches this explicitly, see below).
                                #
                                # n_cells_subsample here MUST match this config's own
                                # "n_cells" (the streamlined per-trial candidate cell
                                # count -- 600 by default, deliberately small so each
                                # SERGIO trial runs quickly) rather than
                                # load_reference_h5ad's own much larger 50_000 default --
                                # several stats here (gene_corr_abs_*, PCA explained-
                                # variance-ratio, any percentile-based stat) are
                                # sample-size-sensitive, so comparing a target computed
                                # from 50_000 reference cells against candidates each
                                # computed from only 600 simulated cells is not
                                # apples-to-apples (the target would be far less noisy
                                # than any candidate could realistically match). This
                                # was missed when the 20260818 gene_corr_abs_normalized_*
                                # addition's target pickle (v5) was regenerated -- left
                                # as-is (v5.mean.pickle not regenerated) since averaging
                                # over 20 runs already washes out most of the difference
                                # for that particular pickle, but the recipe itself is
                                # fixed here for future regenerations.
                                stats_list = [
                                     compute_summary_stats(
                                         load_reference_h5ad(
                                             path, select_gene_symbols=select_gene_symbols,
                                             n_cells_subsample=600, seed=i,
                                         )[0],
                                         n_pca_components=15,
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

                            No change to this recipe is needed to pick up new
                            compute_summary_stats() keys (e.g. the
                            pca_standardized_* family added 20260816, or the
                            gene_corr_abs_normalized_* family added 20260818,
                            see AGENTS.md) -- it already stores whatever that
                            function returns generically. The pickle itself
                            must still be regenerated by rerunning the
                            recipe above against a compute_summary_stats()
                            that has the keys you want to weight, though
                            (see run()'s startup checks, which warn
                            separately for the raw-PCA, standardized-PCA,
                            and normalized-gene-correlation families if any
                            is missing/stale).
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
    mr_state_method            : "random" or "spectral"; fixed per study.
    n_candidate_states         : spectral candidate-pool size.
    n_selected_states          : spectral output row count; must equal
                                 n_clusters for spectral mode.
    candidate_design           : "random" or "sobol" candidate pool design.
    selection_n_restarts/
    selection_swap_passes/
    selection_variance_weight  : fixed spectral selector settings.
    mr_rate_low/mr_rate_high : fixed mr_state sampling range (see
                            "Fixed vs tunable parameters" above).
    grn_seed/sim_seed      : fixed seeds (see "Fixed vs tunable parameters").
    grn_delimiter/grn_regulator_col/grn_target_col/grn_mode_col/
    grn_activation_labels/grn_repression_labels/grn_max_seed_attempts :
                            forwarded to generate_sergio_grn_from_reference
                            (defaults match TRRUST's rawdata TSV format).
    grn_tmp_dir            : str|None, default: system temp dir -- where
                            each trial's regenerated GRN CSV is written
                            (removed again after that trial's simulation,
                            unless grn_archive_dir is also set -- see below).
    grn_archive_dir        : str|None, default None (disabled) -- if set,
                            every trial's GRN CSV (kept, not deleted), a
                            JSON diagnostics dict (from
                            generate_sergio_grn_from_reference's
                            `diagnostics` parameter -- winner_mr/vote_margin/
                            n_ambiguous_flipped/n_aligned_edges per target
                            gene, mr_load, n_distinct_winners,
                            top1_winner_share), and a JSON dump of that
                            trial's full compute_summary_stats() candidate_stats
                            dict (see _jsonify_stats) are written to
                            {grn_archive_dir}/trial{N}.grn.csv,
                            {grn_archive_dir}/trial{N}.diagnostics.json, and
                            {grn_archive_dir}/trial{N}.candidate_stats.json
                            respectively, for later offline analysis of how
                            coherency_bias/canalization_strength/
                            balancing_strength actually behaved across the
                            search (e.g. "does top1_winner_share correlate
                            with lower distance/better
                            pca_explained_variance_ratio spread?") and, via
                            candidate_stats.json, which search-space params
                            actually drive any given stats key (e.g.
                            gene_var_mean/std, log_lib_size_std) at full
                            trial-count statistical power -- best.pt only
                            ever holds one (the best) trial's candidate_stats
                            per study.
                            A handful of cheap scalar summaries derived from
                            the same diagnostics dict (n_distinct_winners,
                            top1_winner_share, frac_ambiguous_flipped,
                            mean_canalization_alignment_frac) are ALSO always
                            recorded as trial.user_attrs regardless of this
                            setting (queryable straight from the Optuna
                            study/dashboard, no archive directory needed) --
                            see run_trial(). Disk cost per trial is small
                            (the GRN CSV is the same handful-of-thousand-
                            edges file already written for simulation; the
                            diagnostics JSON is a few arrays of length
                            n_genes), but not zero, hence opt-in.
    stats_n_pca_components/stats_n_structure_genes/stats_percentiles/stats_seed :
                            forwarded to compute_summary_stats (n_structure_genes
                            controls the shared gene subsample used for both
                            PCA and gene-gene correlation -- see that
                            function's docstring), applied identically to the
                            reference and every candidate.
    weights/distance_eps/distance_eps_frac/distance_eps_abs_floor : compute_
                            stats_distance's config (see above).
    max_lib_size_zero_frac : float, default 0.05 -- hard guard (see "Hard
                            guards against degenerate candidates" below):
                            a trial is pruned if candidate_stats
                            ["lib_size_zero_frac"] exceeds this.
    max_pca_pc1_ratio_factor : float, default 1.3 -- hard guard: a trial is
                            pruned if candidate_stats[pca_pc1_dominance_key]
                            [0] exceeds this factor times target_stats's own
                            value at that same key.
    pca_pc1_dominance_key   : str, default "pca_explained_variance_ratio" --
                            which PCA family's PC1 the max_pca_pc1_ratio_
                            factor guard above checks. Historically always
                            the raw (un-normalized) family -- but as of the
                            20260816/20260820 rounds (see AGENTS.md), the
                            *objective* itself may weight a different PCA
                            family entirely (pca_standardized_*/pca_size_
                            normalized_standardized_*), in which case this
                            guard and the soft objective are checking two
                            different things: empirically (20260820 AGENTS.md
                            entry), a large fraction of trials this guard
                            pruned would have scored *well* under the actual
                            (size-normalized-standardized) objective, and in
                            2/8 of that round's studies the single best
                            guard-rejected trial would have beaten the
                            study's true best surviving trial. Set this to
                            whichever family's PC1 key your weights dict
                            actually scores (e.g.
                            "pca_size_normalized_standardized_explained_
                            variance_ratio") to keep the guard aligned with
                            the objective -- and re-tune max_pca_pc1_ratio_
                            factor itself when doing so, since each family's
                            plausible PC1 range differs (the size-normalized-
                            standardized family's PC1 sits much closer to
                            its own tail than the raw family's does, so the
                            same factor value doesn't mean the same thing in
                            both families).
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

Plotting GRN-coherency diagnostics
-------------------------------------
plot_grn_diagnostics(config_path_or_dict) builds a multi-panel figure
summarizing the coherency_bias/canalization_strength/balancing_strength
mechanism's behavior across a finished (or in-progress) study, combining:
  - the study's own trials (loaded via optuna.load_study(config["storage"],
    config["study_name"])) for the cheap always-logged per-trial scalars
    (n_distinct_winners/top1_winner_share/frac_ambiguous_flipped/
    mean_canalization_alignment_frac/candidate_pca_explained_variance_ratio/
    trial.value) vs. each trial's own resolved coherency/canalization/
    balancing_strength/path_decay params -- these are available for EVERY
    completed trial regardless of grn_archive_dir;
  - optionally, config["grn_archive_dir"]'s per-trial
    trial{N}.diagnostics.json files (if that directory exists and has any)
    for a finer-grained look at one or a few individual trials' winner_mr/
    vote_margin/mr_load distributions (see generate_sergio_grn_from_
    reference's own `diagnostics` param docstring for field meanings) --
    skipped gracefully (panel left blank) if grn_archive_dir wasn't set or
    is empty::

        import tune_synthetic_data as tsd
        fig = tsd.plot_grn_diagnostics("synthetic_tuning_config.20260810.json")
        fig.savefig("grn_diagnostics.png", dpi=120)  # or pass path= directly

See that function's own docstring for the full panel list.

Interrupting/resuming a run
------------------------------
Send SIGINT (Ctrl+C) or SIGTERM once to stop a run gracefully: the
in-flight trial finishes normally (scored and persisted like any other
trial), then this exits the same way it would after exhausting config
["n_trials"]/config["timeout"] -- summary included. Send either signal
again to force an immediate exit instead (that one trial is marked FAIL;
everything before it is unaffected either way). Re-running against the
same config["storage"]/config["study_name"] afterwards (or after a plain
completed run) resumes correctly and picks up where it left off. Set
config["n_trials"] to null (and config["timeout"] to null too, for no
wall-clock cap either) for an unbounded run driven purely by one of the
above signals -- handy for running several concurrent per-seed studies
each for as long as you want, rather than all being bound to one fixed
trial count that some seeds will exhaust sooner than others. See run()'s
own docstring for the full rationale/caveats.

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
import re
import shutil
import signal
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
    load_sergio_dag,
    make_synthetic_data6,
    sample_mr_state_from_spectral_subset,
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
    "coherency_bias":              0.0,
    "canalization_strength":       0.0,
    "balancing_strength":          0.0,
    "path_decay":                  0.9,
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
    # MR-state selection -- fixed per study, not part of search_space. The
    # historical default remains iid; spectral studies opt in explicitly.
    "mr_state_method":       "random",
    "n_candidate_states":    200,
    "n_selected_states":     15,
    "candidate_design":      "random",
    "selection_n_restarts":  1,
    "selection_swap_passes": 0,
    "selection_variance_weight": 0.05,

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
    "grn_archive_dir":       None,  # default: don't archive (see module docstring)

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
    "stats_n_pca_components": 15,
    "stats_n_structure_genes": 500,
    "stats_percentiles":      [5, 25, 50, 75, 95],
    "stats_seed":             0,

    # compute_stats_distance config
    "weights":             {},
    "distance_eps":          1e-6,
    "distance_eps_frac":     0.05,
    "distance_eps_abs_floor": 0.02,

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
    "pca_pc1_dominance_key": "pca_explained_variance_ratio",  # which PCA family's PC1 the guard above checks -- see docstring

    # Optuna
    "storage":      "sqlite:///synthetic_tuning.db",
    "study_name":   "tune_synthetic_data6",
    "n_trials":     30,
    # Wall-clock cap in seconds, passed straight through to study.optimize()'s
    # own timeout= argument -- None (default) means no wall-clock cap. Set
    # both n_trials and timeout to null/None for an unbounded run, stopped
    # only via SIGINT/SIGTERM (see _install_graceful_stop_handler / run()'s
    # "Interrupting/resuming a run" docstring section below).
    "timeout":      None,
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

    if config["mr_state_method"] not in ("random", "spectral"):
        raise ValueError(
            f"Unsupported mr_state_method {config['mr_state_method']!r}; "
            "expected 'random' or 'spectral'"
        )
    if (not isinstance(config["n_candidate_states"], int)
            or config["n_candidate_states"] < 2):
        raise ValueError("n_candidate_states must be an integer >= 2")
    if (not isinstance(config["n_selected_states"], int)
            or not 2 <= config["n_selected_states"] <= config["n_candidate_states"]):
        raise ValueError(
            "n_selected_states must be an integer in [2, n_candidate_states]"
        )
    if config["candidate_design"] not in ("random", "sobol"):
        raise ValueError(
            f"Unsupported candidate_design {config['candidate_design']!r}; "
            "expected 'random' or 'sobol'"
        )
    if (not isinstance(config["selection_n_restarts"], int)
            or config["selection_n_restarts"] < 1):
        raise ValueError("selection_n_restarts must be an integer >= 1")
    if (not isinstance(config["selection_swap_passes"], int)
            or config["selection_swap_passes"] < 0):
        raise ValueError("selection_swap_passes must be an integer >= 0")
    if config["selection_variance_weight"] < 0:
        raise ValueError("selection_variance_weight must be >= 0")
    if (config["mr_state_method"] == "spectral"
            and config["n_selected_states"] != config["n_clusters"]):
        raise ValueError(
            "spectral MR-state selection must produce exactly n_clusters rows: "
            f"n_selected_states={config['n_selected_states']} vs. "
            f"n_clusters={config['n_clusters']}"
        )

    if config["sampler"] not in _SAMPLERS:
        raise ValueError(f"Unsupported sampler {config['sampler']!r}. Supported: {list(_SAMPLERS)}")
    if config["pruner"] not in _PRUNERS:
        raise ValueError(f"Unsupported pruner {config['pruner']!r}. Supported: {list(_PRUNERS)}")

    if config["n_trials"] is not None and (not isinstance(config["n_trials"], int) or config["n_trials"] <= 0):
        raise ValueError(f"config['n_trials'] must be null (unbounded) or a positive int, got {config['n_trials']!r}")
    if config["timeout"] is not None and (not isinstance(config["timeout"], (int, float)) or config["timeout"] <= 0):
        raise ValueError(f"config['timeout'] must be null (no cap) or a positive number of seconds, got {config['timeout']!r}")

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
    unknown_mode_repressor_prob/grn_coherency_kwargs (the coherency_bias/
    canalization_strength/balancing_strength/path_decay quadruple, passed
    through as a dict of kwargs so this function's return signature
    doesn't grow every time a new such knob is added)."""
    sim_kwargs = {k: resolved[k] for k in _SIM_KWARG_DEFAULTS}
    grn_k_dist = ("uniform", resolved["grn_k_low"], resolved["grn_k_low"] + resolved["grn_k_span"])
    hill_coeff_dist = ("constant", resolved["hill_coeff"])
    unknown_mode_repressor_prob = resolved["unknown_mode_repressor_prob"]
    grn_coherency_kwargs = {
        "coherency_bias":        resolved["coherency_bias"],
        "canalization_strength": resolved["canalization_strength"],
        "balancing_strength":    resolved["balancing_strength"],
        "path_decay":            resolved["path_decay"],
    }
    return sim_kwargs, grn_k_dist, hill_coeff_dist, unknown_mode_repressor_prob, grn_coherency_kwargs


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


_STAT_FAMILY_SUFFIX_RE = re.compile(r"_(mean|std|p\d+)$")


def _add_pca_pc2_9_key(
    stats: dict,
    src_key: str = "pca_explained_variance_ratio",
    dst_key: str = "pca_pc2_9_explained_variance_ratio",
) -> None:
    """Derives `dst_key` (PC2..PC9, 8 values) as a plain slice of the
    already-present `src_key` array -- mutates `stats` in place, a no-op if
    that key is absent or too short. No synthetic_data.py change or
    target_stats_path regeneration needed for the *raw* (default src_key)
    variant, since it's just a view onto data compute_summary_stats()
    already returns; the *standardized* variant (src_key=
    "pca_standardized_explained_variance_ratio") does require both, since
    that source key itself is a synthetic_data.py addition -- see
    compute_summary_stats' own docstring.

    Why this exists: pca_tail_participation_ratio (see synthetic_data.
    pca_participation_ratio) is a scale-invariant "how evenly spread is the
    tail" measure -- by construction it cannot tell two differently-*shaped*
    tails apart as long as they're similarly "even". Empirically (all 8
    `synthetic_tuning_20260812.*` studies' best trials, see AGENTS.md),
    candidates were scoring at-or-above the target's own participation
    ratio while their actual PC3-PC9 explained-variance-ratio ran at only
    ~45-60% of the target's, compensated for by an *excess* (~95-116%) in
    the PC16-PC20 far tail -- i.e. a flatter-than-target tail shape can
    "pass" the scalar participation-ratio check without the early/mid tail
    (PC2-9), which is the more visually/analytically salient part of a PCA
    scree plot, ever actually matching. This key lets compute_stats_distance
    score that magnitude directly, orthogonal to the tail-evenness question
    pca_tail_participation_ratio asks. Called once per PCA family (raw and
    standardized) at each of this module's two call sites (run()/run_trial())
    -- see the 20260816 AGENTS.md entry for why both families exist."""
    pca = stats.get(src_key)
    if pca is not None and len(pca) >= 9:
        stats[dst_key] = np.asarray(pca[1:9], dtype=np.float64)


def _add_nonzero_frac_key(
    stats: dict,
    src_key: str = "zero_frac",
    dst_key: str = "nonzero_frac",
) -> None:
    """Derives `dst_key` = 1 - `stats[src_key]` -- mutates `stats` in place,
    a no-op if `src_key` is absent. No synthetic_data.py change or
    target_stats_path regeneration needed (same convention as
    _add_pca_pc2_9_key's raw-family variant): this only rearranges data
    compute_summary_stats() already returns.

    Why this exists: compute_stats_distance's plain relative-error formula,
    scored directly on zero_frac, badly under-penalizes candidates that are
    *sparser* than the reference (zero_frac overshoot) whenever the target
    itself is already close to 1 -- e.g. target zero_frac=0.9242 (this
    project's real reference value): a candidate at zero_frac=0.97 already
    has ~60% *less* non-zero (usable) data than the reference
    ((1-0.97)/(1-0.9242) - 1 = -60%), but the plain relative-error term on
    zero_frac itself is only ((0.97-0.9242)/(0.9242+eps))**2 ~= 0.0022 --
    negligible next to other terms that typically run 0.05-0.5 (confirmed
    empirically against real synthetic_tuning_20260819.* trials: this
    metric change re-ranks the "best" trial in 3/8 of those already-
    completed studies, in every case toward a trial with zero_frac closer
    to target -- see AGENTS.md's 20260820 entry). Scoring the *non-zero*
    fraction instead makes the relative-error denominator the target's own
    (small) non-zero fraction rather than its (large, close-to-1)
    zero-fraction, so the same absolute zero_frac deviation is scored at
    the magnitude of usable-data loss/gain it actually represents --
    symmetric in either direction (also raises the cost of a candidate
    *denser* than the reference, not just sparser). See this module's
    weights docstring for the corresponding config convention (weight
    nonzero_frac instead of zero_frac, matching the pca_standardized_*/
    gene_corr_abs_normalized_* precedent of retargeting a weight at a
    corrected metric rather than the original raw one)."""
    val = stats.get(src_key)
    if val is not None:
        stats[dst_key] = 1.0 - float(val)


def _jsonify_stats(stats: dict) -> dict:
    """Convert a compute_summary_stats()-style dict (a mix of plain floats/
    ints and numpy arrays/scalars, e.g. pca_explained_variance_ratio,
    dropout_curve_zero_frac, np.float64 summary values) into plain
    JSON-serializable Python types. Used solely to archive full per-trial
    candidate_stats to grn_archive_dir (see run_trial) -- read back with
    plain json.load; arrays round-trip as lists, so re-wrap with
    np.asarray(...) if doing array math on them later. Unlike the always-
    computed cheap scalar user_attrs (n_distinct_winners, top_distance_terms,
    ...), this gives full-resolution per-trial access to every stats key
    (e.g. gene_var_mean/std, gene_corr_abs_*, log_lib_size_std) for post-hoc
    correlation analysis against search-space params at full trial count,
    rather than only the n=1 best trial's candidate_stats saved in best.pt."""
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _stat_family(key: str) -> str:
    """Groups a compute_summary_stats() scalar key into its distributional
    "family" by stripping a trailing _mean/_std/_p<N> suffix -- e.g.
    "gene_mean_p5" and "gene_mean_std" both map to "gene_mean" -- used only
    to give compute_stats_distance's relative-error epsilon a family-
    appropriate scale (see there), nothing else. Keys with no such suffix
    (zero_frac, pca_tail_participation_ratio, mean_var_log_slope, ...) are
    their own single-member family."""
    m = _STAT_FAMILY_SUFFIX_RE.search(key)
    return key[:m.start()] if m else key


def compute_stats_distance(target_stats: dict, candidate_stats: dict,
                            weights: dict | None = None, eps: float = 1e-6,
                            eps_frac: float = 0.05, eps_abs_floor: float = 0.02) -> tuple[float, dict]:
    """Weighted-mean relative squared error between two compute_summary_stats()
    dicts (see module docstring's "The distance metric"). Returns
    (distance, breakdown) where breakdown = {"terms": {key: term, ...},
    "skipped": [key, ...]} for diagnostics -- keys are skipped (not errors)
    when absent from either dict, non-finite, or (for array keys) of zero
    overlapping length.

    Near-zero targets: the plain relative-error formula ((cand - target) /
    (|target| + eps)) ** 2 is numerically explosive whenever `target` itself
    is ~0 and `eps` is a tiny fixed constant (the original default) --
    e.g. gene_mean_p5/gene_var_p5/lib_size_zero_frac are legitimately exactly
    0.0 for real sparse single-cell reference data, so *any* small nonzero
    candidate value there (near-inevitable) produces a term of order
    (candidate / eps) ** 2, easily 10^5-10^6 -- enough to swamp every other
    term and make the resulting "best" trial selection meaningless (observed
    in practice across several 20260811.* tuning studies). To guard against
    this without any key-specific special-casing, the effective epsilon for
    each scalar key is floored at `eps_frac` times that key's distributional
    "family" scale (see _stat_family) -- i.e. the largest |target value|
    among all of target_stats' *_mean/_std/_p<N> siblings of that key (e.g.
    gene_mean_p5's family also contains gene_mean_mean/std/p25/p50/p75/p95,
    so its floor reflects however large that family's own values are
    elsewhere, rather than the single (possibly-0) entry being scored).
    This alone doesn't help *single-member* families whose own value is
    itself ~0 (e.g. lib_size_zero_frac has no *_mean/_std/_p<N> siblings at
    all, so its "family scale" is just its own ~0 value) -- `eps_abs_floor`
    is a second, flat floor underneath that (also applied to array keys,
    from their own values) for exactly that residual case: comparing two
    fractions/ratios that are both plausibly ~0 is inherently more like an
    absolute- than a relative-error question (there's no other scale to be
    relative *to*), so a small fixed floor (default 0.02, i.e. ~2 percentage
    points for a fraction-like stat) stands in. Sizable target values are
    unaffected either way (eps_frac * family_scale or |target| itself both
    dwarf 0.02 there). Set eps_frac=eps_abs_floor=0.0 to recover the
    original fixed-eps-only behavior.
    """
    weights = weights or {}
    terms: dict[str, float] = {}
    skipped: list[str] = []

    family_scale: dict[str, float] = {}
    for key, value in target_stats.items():
        if key in _EXCLUDED_STATS_KEYS or isinstance(value, np.ndarray):
            continue
        try:
            v = abs(float(value))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        fam = _stat_family(key)
        family_scale[fam] = max(family_scale.get(fam, 0.0), v)

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
            key_eps = max(eps, eps_frac * float(np.max(np.abs(t[valid]))), eps_abs_floor)
            rel_sq = ((c[valid] - t[valid]) / (np.abs(t[valid]) + key_eps)) ** 2
            terms[key] = float(np.mean(rel_sq))
        else:
            t, c = float(target_value), float(candidate_value)
            if not (math.isfinite(t) and math.isfinite(c)):
                skipped.append(key)
                continue
            key_eps = max(eps, eps_frac * family_scale.get(_stat_family(key), 0.0), eps_abs_floor)
            terms[key] = float(((c - t) / (abs(t) + key_eps)) ** 2)

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


def run_trial(
    trial: optuna.Trial,
    config: dict,
    target_stats: dict,
    mr_state_cache: dict | None = None,
) -> float:
    """Build one synthetic candidate from suggested/fixed simulation
    parameters, score it against target_stats via compute_stats_distance,
    and return that scalar distance (to minimize). See module docstring
    for the full pipeline description."""
    resolved = suggest_sim_cfg(trial, config["search_space"])
    trial.set_user_attr("resolved_params", {k: float(v) for k, v in resolved.items()})
    sim_kwargs, grn_k_dist, hill_coeff_dist, unknown_mode_repressor_prob, grn_coherency_kwargs = _split_resolved(resolved)

    grn_tmp_dir = config["grn_tmp_dir"] or tempfile.gettempdir()
    Path(grn_tmp_dir).mkdir(parents=True, exist_ok=True)
    grn_path = os.path.join(grn_tmp_dir, f"sergio_grn_trial{trial.number}_{os.getpid()}.csv")

    grn_diagnostics: dict = {}
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
            diagnostics=grn_diagnostics,
            **grn_coherency_kwargs,
        )
    except Exception as e:
        _prune(trial, "grn_generation_failed", str(e))

    # Cheap scalar summaries of the coherency_bias/canalization_strength/
    # balancing_strength mechanism's behavior on this trial -- always
    # recorded (queryable straight from the Optuna study/dashboard),
    # regardless of whether grn_archive_dir is set. See module docstring's
    # "grn_archive_dir" entry.
    trial.set_user_attr("n_distinct_winners", grn_diagnostics["n_distinct_winners"])
    trial.set_user_attr("top1_winner_share", grn_diagnostics["top1_winner_share"])
    n_ambiguous_total = sum(grn_diagnostics["n_ambiguous"])
    trial.set_user_attr(
        "frac_ambiguous_flipped",
        (sum(grn_diagnostics["n_ambiguous_flipped"]) / n_ambiguous_total) if n_ambiguous_total else float("nan"),
    )
    n_regs_total = sum(grn_diagnostics["n_regs"])
    trial.set_user_attr(
        "mean_canalization_alignment_frac",
        (sum(grn_diagnostics["n_aligned_edges"]) / n_regs_total) if n_regs_total else float("nan"),
    )

    if config["grn_archive_dir"]:
        archive_dir = Path(config["grn_archive_dir"])
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(grn_path, archive_dir / f"trial{trial.number}.grn.csv")
        with open(archive_dir / f"trial{trial.number}.diagnostics.json", "w") as f:
            json.dump(grn_diagnostics, f)

    mr_state_diagnostics = {}
    try:
        if config.get("mr_state_method", "random") == "spectral":
            dag = load_sergio_dag(
                grn_path,
                shared_coop_state=config["shared_coop_state"],
                mr_gene_ids=mr_ids,
            )
            mr_state_np = sample_mr_state_from_spectral_subset(
                dag=dag,
                n_selected_states=config["n_selected_states"],
                low=config["mr_rate_low"],
                high=config["mr_rate_high"],
                n_candidate_states=config["n_candidate_states"],
                candidate_design=config["candidate_design"],
                decays=sim_kwargs["decays"],
                n_restarts=config["selection_n_restarts"],
                swap_passes=config["selection_swap_passes"],
                seed=config["sim_seed"],
                variance_weight=config["selection_variance_weight"],
                cache=mr_state_cache,
                diagnostics=mr_state_diagnostics,
            )
        else:
            mr_state_np = _sample_mr_state(
                config["n_clusters"], len(mr_ids),
                config["mr_rate_low"], config["mr_rate_high"],
                seed=config["sim_seed"],
            )
            mr_state_diagnostics = {"method": "random"}
        trial.set_user_attr("mr_state_method", mr_state_diagnostics["method"])
        if mr_state_diagnostics["method"] == "spectral":
            trial.set_user_attr(
                "mr_state_selection_cache_hit",
                mr_state_diagnostics["selection_cache_hit"],
            )
            trial.set_user_attr(
                "mr_state_candidate_pool_cache_hit",
                mr_state_diagnostics["candidate_pool_cache_hit"],
            )
            trial.set_user_attr(
                "mr_state_spectral_score",
                mr_state_diagnostics["selected_metrics"]["score"],
            )
            trial.set_user_attr(
                "mr_state_spectral_participation_ratio",
                mr_state_diagnostics["selected_metrics"]["participation_ratio"],
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
    _add_pca_pc2_9_key(candidate_stats)
    _add_pca_pc2_9_key(
        candidate_stats,
        src_key="pca_standardized_explained_variance_ratio",
        dst_key="pca_standardized_pc2_9_explained_variance_ratio",
    )
    _add_pca_pc2_9_key(
        candidate_stats,
        src_key="pca_size_normalized_standardized_explained_variance_ratio",
        dst_key="pca_size_normalized_standardized_pc2_9_explained_variance_ratio",
    )
    _add_nonzero_frac_key(candidate_stats)
    trial.set_user_attr(
        "candidate_pca_tail_participation_ratio",
        candidate_stats.get("pca_tail_participation_ratio", float("nan")),
    )
    trial.set_user_attr(
        "candidate_pca_standardized_tail_participation_ratio",
        candidate_stats.get("pca_standardized_tail_participation_ratio", float("nan")),
    )

    if config["grn_archive_dir"]:
        archive_dir = Path(config["grn_archive_dir"])
        archive_dir.mkdir(parents=True, exist_ok=True)
        with open(archive_dir / f"trial{trial.number}.candidate_stats.json", "w") as f:
            json.dump(_jsonify_stats(candidate_stats), f)

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
    # Full per-PC explained-variance-ratio vector -- not just the top-5
    # worst compute_stats_distance terms (which only shows the biggest
    # gaps, not the actual shape) -- since PC-by-PC spread (not just PC1
    # dominance) is the whole point of the coherency/canalization/
    # balancing mechanism; see AGENTS.md's Known Issues entry. Always the
    # *raw* family, regardless of pca_pc1_dominance_key below -- this is a
    # diagnostics/plotting field (plot_grn_diagnostics), independent of
    # which family the hard guard itself checks.
    trial.set_user_attr("candidate_pca_explained_variance_ratio", [float(v) for v in cand_pca])

    # See pca_pc1_dominance_key's own docstring (module docstring's "Hard
    # guards against degenerate candidates" / _CONFIG_DEFAULTS) for why this
    # defaults to the raw family but should be pointed at whichever PCA
    # family the objective's weights dict actually scores.
    pc1_key = config.get("pca_pc1_dominance_key", "pca_explained_variance_ratio")
    cand_pc1_pca = candidate_stats.get(pc1_key, np.array([]))
    target_pc1_pca = target_stats.get(pc1_key, np.array([]))
    if len(cand_pc1_pca) and len(target_pc1_pca):
        max_pc1 = float(target_pc1_pca[0]) * config["max_pca_pc1_ratio_factor"]
        if float(cand_pc1_pca[0]) > max_pc1:
            _prune(
                trial, "degenerate_pca_pc1_dominance",
                f"{pc1_key}[0]={float(cand_pc1_pca[0]):.4f} > "
                f"max_allowed={max_pc1:.4f} (target={float(target_pc1_pca[0]):.4f}, "
                f"factor={config['max_pca_pc1_ratio_factor']})",
            )

    distance, breakdown = compute_stats_distance(
        target_stats, candidate_stats, weights=config.get("weights"), eps=config["distance_eps"],
        eps_frac=config.get("distance_eps_frac", 0.05),
        eps_abs_floor=config.get("distance_eps_abs_floor", 0.02),
    )
    trial.set_user_attr("skipped_stats_keys", breakdown["skipped"])
    if not math.isfinite(distance):
        _prune(trial, "non_finite_distance")

    trial.set_user_attr("n_mrs", len(mr_ids))
    # Ranked by *weighted* contribution (weight * raw term), not raw term
    # value -- sorting by raw value alone (the original behavior) surfaces
    # whichever keys have the largest unweighted relative error regardless
    # of their configured weight, which in practice meant this almost always
    # showed the deliberately zero-weighted gene_mean_p*/gene_var_p*/
    # mean_var_log_* family (confirmed the #1 raw term in ~65% of trials
    # across the 20260813.* studies -- see AGENTS.md's Known Issues) rather
    # than what's actually driving `distance`. Recompute from breakdown["terms"]
    # directly (not this field) if you need the raw, unweighted values too.
    stats_weights = config.get("weights") or {}
    worst = sorted(
        breakdown["terms"].items(),
        key=lambda kv: stats_weights.get(kv[0], 1.0) * kv[1],
        reverse=True,
    )[:5]
    trial.set_user_attr("top_distance_terms", {k: float(v) for k, v in worst})

    _save_if_best_artifact(
        config, trial, distance, X_sim, cluster_labels, mr_state_np,
        mr_ids, gene_id_to_symbol, resolved, candidate_stats,
        mr_state_metadata=mr_state_diagnostics,
    )

    return distance


def _save_if_best_artifact(config: dict, trial: optuna.Trial, distance: float,
                            X_sim: torch.Tensor, cluster_labels: np.ndarray,
                            mr_state_np: np.ndarray, mr_ids: list, gene_id_to_symbol: dict,
                            resolved: dict, candidate_stats: dict,
                            mr_state_metadata: dict | None = None) -> None:
    """If *distance* improves on the best distance among this trial's own
    study's completed trials so far, overwrite {artifact_dir}/best.pt with
    this trial's full synthetic dataset/metadata.

    Queries trial.study's own completed-trial values (already durable in
    config["storage"]) directly, rather than reading {artifact_dir}/best.pt
    back from disk to learn the previous best -- the original approach
    (mirroring tuning.py's _save_if_best_checkpoint's resume-safety
    rationale). That disk round-trip turned out to be silently broken in
    practice: torch.load(path) raised on every call in all 8
    20260811.* studies (plausibly PyTorch >=2.6 defaulting to
    weights_only=True, which rejects the plain numpy arrays this artifact
    stores, unless explicitly allow-listed), was swallowed by the bare
    `except Exception: prev = float("inf")`, and so every trial unconditionally
    overwrote the previous one -- confirmed post-hoc: every one of those 8
    studies' best.pt held its *last* trial rather than its true
    minimum-distance trial (per-study minimum distances were 2-800000x
    lower than what got saved). Tradeoff versus the disk-read-back approach:
    if artifact_dir is wiped/ephemeral across a resumed study while storage
    persists, this now correctly keeps refusing to overwrite until a new
    trial beats the study's historical best, rather than treating the
    missing file as license to reset -- use a fresh study_name (or a fresh
    storage db) if you actually want an unconstrained fresh artifact
    baseline in that scenario.
    """
    distance = float(distance)
    if not math.isfinite(distance):
        return

    prev_completed_values = [
        t.value for t in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if t.value is not None and math.isfinite(t.value)
    ]
    if prev_completed_values and distance >= min(prev_completed_values):
        return

    path = Path(config["artifact_dir"]) / "best.pt"
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
        "mr_state_method":   (mr_state_metadata or {}).get(
            "method", config.get("mr_state_method", "random")),
        "mr_state_selection": mr_state_metadata or {
            "method": config.get("mr_state_method", "random"),
        },
        "mr_state_grn_seed":  config.get("grn_seed"),
        "mr_state_sim_seed":  config.get("sim_seed"),
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


def _warn_if_pythonhashseed_unset() -> None:
    """Print a loud warning if PYTHONHASHSEED isn't set to a fixed value.

    generate_sergio_grn_from_reference's GRN-topology sampling
    (_sample_connected_subgraph) has a pre-existing, documented (see
    AGENTS.md's Known Issues), not-yet-fixed non-determinism bug: it
    iterates plain Python `set`s whose order depends on the interpreter's
    string-hash seed, which is randomized per-process by default. Since
    this harness is meant to be run as several independent OS processes
    (one per seed, see module docstring) that are each supposed to share
    the *same* GRN topology via a common config["grn_seed"] -- varying only
    config["sim_seed"] between them -- an unset (randomized) PYTHONHASHSEED
    silently breaks that assumption: confirmed post-hoc across both the
    synthetic_tuning_20260814.0{0-7} and synthetic_tuning_20260815_pilot_*
    studies, whose n_mrs (a direct readout of GRN topology) varied
    substantially (40-104) across nominally-same-grn_seed processes purely
    from this effect, adding uncontrolled noise to every cross-seed
    comparison made from that data.

    This function can only check *this* process's own PYTHONHASHSEED (it
    cannot enforce -- or even see -- what value any other concurrently-
    launched sibling process used, since the hash seed is fixed at
    interpreter startup, before this module's code ever runs) -- so it's a
    reminder, not a guarantee: launch every sibling process (e.g. across
    the 8 seeds of one tuning round) with the identical
    `PYTHONHASHSEED=<some fixed value>` environment variable set *before*
    invoking python, e.g.::

        PYTHONHASHSEED=0 python tune_synthetic_data.py --config synthetic_tuning_config.20260815.00.json
        PYTHONHASHSEED=0 python tune_synthetic_data.py --config synthetic_tuning_config.20260815.01.json
        ...

    for config["grn_seed"] to actually produce identical topology across
    all of them, as intended.
    """
    value = os.environ.get("PYTHONHASHSEED")
    if value is None:
        tqdm.write(
            "\n*** WARNING: PYTHONHASHSEED is not set -- generate_sergio_grn_from_reference's "
            "GRN-topology sampling has a known non-determinism bug tied to the (by default "
            "randomized-per-process) string-hash seed (see AGENTS.md's Known Issues). If you're "
            "running several sibling processes (e.g. one per seed) that are meant to share the "
            "same GRN topology via a common config['grn_seed'], they will NOT actually get the "
            "same topology unless every one of them is launched with an identical "
            "PYTHONHASHSEED=<fixed value> environment variable set before the python interpreter "
            "starts (setting it from within this script is too late). This process will proceed "
            "regardless -- this is a reminder, not a hard requirement. ***\n"
        )
    else:
        tqdm.write(f"PYTHONHASHSEED={value!r} (fixed) -- make sure every sibling process uses the same value.")


def _install_graceful_stop_handler(study: optuna.Study):
    """Install SIGINT/SIGTERM handlers that ask *study* to stop gracefully
    (via Study.stop()) rather than letting a raw KeyboardInterrupt propagate
    out of whichever trial happens to be running.

    Study.stop() only sets a flag checked *between* trials -- the
    currently-running trial (potentially minutes into a SERGIO simulation)
    finishes normally, is scored and persisted as usual, and only then does
    study.optimize()'s loop exit, falling through to run()'s usual post-loop
    summary-writing code. This avoids both wasting whatever fraction of the
    in-flight trial's compute had already run, and silently skipping the
    tuned_synthetic_config.json summary (which only gets written after
    study.optimize() returns normally -- a raw KeyboardInterrupt propagating
    out of run() would skip it, even though the study's own storage/best.pt
    are already safely persisted regardless -- see run()'s own
    "Interrupting/resuming a run" docstring section for the full rationale).

    A second SIGINT/SIGTERM (e.g. an impatient repeated Ctrl+C) forces an
    immediate KeyboardInterrupt instead of waiting for the current trial to
    finish -- same safety properties as an unhandled Ctrl+C today (that one
    trial is marked FAIL, everything before it is untouched), just opt-in
    for when you don't want to wait out a slow trial.

    If a signal arrives *outside* study.optimize()'s call (e.g. while still
    loading target_stats, or after it has already returned) -- Study.stop()
    would raise RuntimeError ("supposed to be invoked inside an objective
    function or a callback") -- so this checks study._thread_local.
    in_optimize_loop first and just raises KeyboardInterrupt directly in
    that case, matching plain default behavior.

    Returns (prev_sigint_handler, prev_sigterm_handler) -- restore with
    signal.signal(signal.SIGINT, prev_sigint_handler) etc. (run() does this
    in a finally: block) so this is safe to call from a library context too,
    not just a standalone script.
    """
    signal_count = {"n": 0}

    def _handler(signum, frame):
        signal_count["n"] += 1
        signame = signal.Signals(signum).name
        if signal_count["n"] == 1 and getattr(study._thread_local, "in_optimize_loop", False):
            tqdm.write(
                f"\n=== received {signame} -- finishing the in-flight trial, then stopping "
                f"gracefully (send {signame} again to force an immediate exit instead) ==="
            )
            study.stop()
            return
        tqdm.write(f"\n=== received {signame} -- forcing an immediate exit ===")
        raise KeyboardInterrupt(f"forced exit on {signame}")

    prev_sigint = signal.signal(signal.SIGINT, _handler)
    prev_sigterm = signal.signal(signal.SIGTERM, _handler)
    return prev_sigint, prev_sigterm


def run(config_path: str) -> optuna.Study:
    """Plain entry point, safe to call directly from a notebook cell:

        import tune_synthetic_data
        study = tune_synthetic_data.run("synthetic_tuning_config.json")
        best = study.best_trial

    Interrupting/resuming a run
    ----------------------------
    Sending SIGINT (Ctrl+C) or SIGTERM once asks the study to stop
    gracefully: the in-flight trial finishes normally (scored and persisted
    like any other trial), then this function proceeds through its usual
    post-loop summary-writing exactly as if config["n_trials"]/config
    ["timeout"] had been reached naturally. Sending either signal again
    forces an immediate exit instead (same data-safety properties as an
    unhandled Ctrl+C: the in-flight trial is marked FAIL, everything before
    it is untouched). Either way, config["storage"] already durably records
    every *completed* trial as it finishes (not batched at the end) and
    _save_if_best_artifact keeps config["artifact_dir"]/best.pt in sync with
    the true best completed trial so far -- so no tuning progress is ever
    lost to an interruption, graceful or not.

    Set config["n_trials"] to null (and, if you also want no wall-clock
    cap, config["timeout"] to null too) for an unbounded run that only ever
    stops via one of the above signals -- e.g. to let several concurrent
    per-seed studies (one process each, see module docstring) each run for
    as long as you want rather than being bound to a fixed trial count that
    some seeds will exhaust much sooner than others. Simply re-invoking
    run() again against the same config["storage"]/config["study_name"]
    (whether after a graceful stop, a forced exit, or a completed fixed-
    n_trials run) resumes correctly -- optuna.create_study(...,
    load_if_exists=True) below reconnects to every historical trial already
    in storage, and the TPE sampler rebuilds its surrogate model from that
    full history on every call (it keeps no state of its own between
    invocations). One caveat: build_sampler() constructs a fresh
    TPESampler(seed=0) on every run() call, so a study stopped and resumed
    several times will suggest a slightly different trial sequence than one
    uninterrupted run of the same eventual trial count -- both are valid,
    history-informed TPE behavior, just not bit-for-bit identical.

    Persisted in config["storage"] under config["study_name"], so it can be
    reloaded/resumed later via optuna.load_study() even without calling
    this function again -- run this script from multiple processes against
    the same storage for wall-clock parallelism (see module docstring).
    A plain-JSON summary is written to config["output"] for convenience;
    see module docstring's "output" key description.
    """
    config = load_config(config_path)
    _warn_if_pythonhashseed_unset()

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
    _add_pca_pc2_9_key(target_stats)
    _add_pca_pc2_9_key(
        target_stats,
        src_key="pca_standardized_explained_variance_ratio",
        dst_key="pca_standardized_pc2_9_explained_variance_ratio",
    )
    _add_pca_pc2_9_key(
        target_stats,
        src_key="pca_size_normalized_standardized_explained_variance_ratio",
        dst_key="pca_size_normalized_standardized_pc2_9_explained_variance_ratio",
    )
    _add_nonzero_frac_key(target_stats)
    print(
        f"reference: n_cells={target_stats['n_cells']}  n_genes={target_stats['n_genes']}  "
        f"gene_mean_mean={target_stats['gene_mean_mean']:.4f}  "
        f"log_lib_size_mean={target_stats['log_lib_size_mean']:.4f}  "
        f"zero_frac={target_stats['zero_frac']:.4f}  "
        f"pca_tail_participation_ratio={target_stats.get('pca_tail_participation_ratio', float('nan')):.4f}  "
        f"pca_standardized_tail_participation_ratio="
        f"{target_stats.get('pca_standardized_tail_participation_ratio', float('nan')):.4f}"
    )
    if "pca_tail_participation_ratio" not in target_stats:
        print(
            "  NOTE: target_stats has no 'pca_tail_participation_ratio' key -- it was "
            "computed with an older compute_summary_stats() and needs to be "
            "regenerated for weights['pca_tail_participation_ratio'] to have any "
            "effect (that term will simply be skipped otherwise; see "
            "compute_stats_distance's docstring)."
        )
    else:
        # pca_tail_participation_ratio's value depends on how many tail PCs it
        # was computed over -- comparing a target computed at one
        # n_pca_components depth against candidates computed at another
        # (this config's stats_n_pca_components) silently compares two
        # different quantities. Confirmed to have actually happened across
        # every one of the 20260811.* studies (target regenerated with the
        # (then-undocumented) default n_pca_components=10 while
        # stats_n_pca_components=20 -- candidates' tail spread was being
        # compared against a target ~2x shallower than intended, and since
        # participation ratio's plausible range scales with tail length,
        # this systematically penalized candidates for achieving *more*
        # PC2+ spread than that undersized target, the opposite of this
        # mechanism's actual goal). len(pca_explained_variance_ratio) - 1 ==
        # the number of tail PCs pca_tail_participation_ratio was computed
        # over (see synthetic_data.pca_participation_ratio's skip_leading=1
        # default), so it should equal stats_n_pca_components - 1.
        target_n_pca = len(target_stats.get("pca_explained_variance_ratio", []))
        if target_n_pca and target_n_pca != config["stats_n_pca_components"]:
            print(
                f"  *** WARNING: target_stats' pca_explained_variance_ratio has length "
                f"{target_n_pca} (computed over {max(target_n_pca - 1, 0)} tail PCs) but "
                f"this config's stats_n_pca_components={config['stats_n_pca_components']} "
                f"(candidates will be computed over {config['stats_n_pca_components'] - 1} "
                f"tail PCs) -- pca_tail_participation_ratio target/candidate are NOT "
                f"apples-to-apples until target_stats_path is regenerated with "
                f"n_pca_components={config['stats_n_pca_components']} (see "
                f"target_stats_path's own docstring for the regeneration recipe). "
                f"weights.get('pca_tail_participation_ratio') will otherwise silently "
                f"score against the wrong scale. ***"
            )

    # Same two checks as pca_tail_participation_ratio above, for the
    # standardized (scale-invariant, correlation-matrix) PCA companion
    # added alongside the L1-budget-preserving canalization fix -- see
    # AGENTS.md's 20260816 entry. A target_stats pickle predating that
    # change has neither pca_standardized_explained_variance_ratio nor
    # pca_standardized_tail_participation_ratio/
    # pca_standardized_pc2_9_explained_variance_ratio, so any nonzero
    # weight on those keys would otherwise silently score nothing (the key
    # is simply absent from target_stats, so compute_stats_distance skips
    # it -- see that function's own skip-if-absent behavior).
    if "pca_standardized_tail_participation_ratio" not in target_stats:
        print(
            "  NOTE: target_stats has no 'pca_standardized_tail_participation_ratio' "
            "key -- it was computed with an older compute_summary_stats() (predating "
            "the standardized-PCA addition) and needs to be regenerated for "
            "weights['pca_standardized_tail_participation_ratio']/"
            "weights['pca_standardized_pc2_9_explained_variance_ratio'] to have any "
            "effect (those terms will simply be skipped otherwise)."
        )
    else:
        target_n_pca_z = len(target_stats.get("pca_standardized_explained_variance_ratio", []))
        if target_n_pca_z and target_n_pca_z != config["stats_n_pca_components"]:
            print(
                f"  *** WARNING: target_stats' pca_standardized_explained_variance_ratio "
                f"has length {target_n_pca_z} but this config's "
                f"stats_n_pca_components={config['stats_n_pca_components']} -- "
                f"pca_standardized_tail_participation_ratio target/candidate are NOT "
                f"apples-to-apples until target_stats_path is regenerated with a "
                f"matching n_pca_components. ***"
            )

    # Same presence check as the two PCA families above, for the
    # library-size-normalized gene-correlation companion added alongside
    # the gene_corr_abs_mean/p50 library-size-confound diagnosis -- see
    # AGENTS.md's 20260818 entry. No depth/apples-to-apples check is needed
    # here (unlike the PCA families) since these are plain scalars, not an
    # n_pca_components-dependent array.
    if "gene_corr_abs_normalized_mean" not in target_stats:
        print(
            "  NOTE: target_stats has no 'gene_corr_abs_normalized_mean' key -- it "
            "was computed with an older compute_summary_stats() (predating the "
            "library-size-normalized gene-correlation addition) and needs to be "
            "regenerated for weights['gene_corr_abs_normalized_mean']/"
            "['gene_corr_abs_normalized_std']/['gene_corr_abs_normalized_p50']/"
            "['gene_corr_abs_normalized_p90'] to have any effect (those terms will "
            "simply be skipped otherwise)."
        )

    # Same two checks as pca_standardized_* above, for the family that
    # combines *both* confound-removal steps (per-gene standardization AND
    # per-cell library-size renormalization) -- see AGENTS.md's 20260820
    # entry and compute_summary_stats'
    # pca_size_normalized_standardized_explained_variance_ratio docstring
    # for why pca_standardized_* alone still isn't scale-invariant with
    # respect to lib_size_scale.
    if "pca_size_normalized_standardized_tail_participation_ratio" not in target_stats:
        print(
            "  NOTE: target_stats has no "
            "'pca_size_normalized_standardized_tail_participation_ratio' key -- it "
            "was computed with an older compute_summary_stats() (predating this "
            "addition) and needs to be regenerated for "
            "weights['pca_size_normalized_standardized_tail_participation_ratio']/"
            "['pca_size_normalized_standardized_pc2_9_explained_variance_ratio'] to "
            "have any effect (those terms will simply be skipped otherwise)."
        )
    else:
        target_n_pca_zn = len(target_stats.get("pca_size_normalized_standardized_explained_variance_ratio", []))
        if target_n_pca_zn and target_n_pca_zn != config["stats_n_pca_components"]:
            print(
                f"  *** WARNING: target_stats' "
                f"pca_size_normalized_standardized_explained_variance_ratio has length "
                f"{target_n_pca_zn} but this config's "
                f"stats_n_pca_components={config['stats_n_pca_components']} -- "
                f"pca_size_normalized_standardized_tail_participation_ratio target/"
                f"candidate are NOT apples-to-apples until target_stats_path is "
                f"regenerated with a matching n_pca_components. ***"
            )

    # config["pca_pc1_dominance_key"]'s presence check -- see that key's own
    # docstring entry (module docstring / _CONFIG_DEFAULTS) for why this
    # should track whichever PCA family the objective's weights dict
    # actually scores. Unlike the weighted-objective keys above, an absent
    # key here doesn't just silently skip a soft term -- it silently
    # disables the hard degenerate-PC1-dominance guard entirely (empty-array
    # short-circuit in run_trial()), so this is a loud warning even though
    # nothing strictly breaks.
    pc1_key = config.get("pca_pc1_dominance_key", "pca_explained_variance_ratio")
    if pc1_key not in target_stats:
        print(
            f"  *** WARNING: config['pca_pc1_dominance_key']={pc1_key!r} is not a key "
            f"in target_stats -- the degenerate_pca_pc1_dominance hard guard will "
            f"silently never fire (run_trial() short-circuits when either side's "
            f"array is empty). Regenerate target_stats_path or fix this config key. ***"
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
    budget_desc = (
        f"{config['n_trials']} trials" if config["n_trials"] is not None
        else (f"unbounded, timeout={config['timeout']}s" if config["timeout"] is not None
              else "unbounded -- stop via SIGINT/SIGTERM")
    )
    print(f"\n=== Tuning synthetic_data6 against {target_desc!r} ({budget_desc}) ===")

    mr_state_cache = {}
    prev_sigint, prev_sigterm = _install_graceful_stop_handler(study)
    start = time.monotonic()
    try:
        study.optimize(
            lambda trial: run_trial(
                trial, config, target_stats, mr_state_cache=mr_state_cache
            ),
            n_trials=config["n_trials"],
            timeout=config["timeout"],
            show_progress_bar=True,
            callbacks=[trial_summary_callback],
        )
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
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
            "mr_state_method":     config.get("mr_state_method", "random"),
            "n_candidate_states":  config.get("n_candidate_states"),
            "n_selected_states":   config.get("n_selected_states"),
            "candidate_design":    config.get("candidate_design"),
            "selection_n_restarts": config.get("selection_n_restarts"),
            "selection_swap_passes": config.get("selection_swap_passes"),
            "selection_variance_weight": config.get("selection_variance_weight"),
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


def _load_matching_mr_state_artifact(
    config: dict,
    resolved: dict,
    method: str,
    n_mrs: int,
) -> np.ndarray | None:
    """Load a best artifact's MR state when it matches this replay request."""
    path = Path(config["artifact_dir"]) / "best.pt"
    if not path.exists():
        return None
    try:
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(path, map_location="cpu")
    except Exception:
        return None

    if artifact.get("mr_state_method") != method:
        return None
    if artifact.get("mr_state_grn_seed") != config.get("grn_seed"):
        return None
    if artifact.get("mr_state_sim_seed") != config.get("sim_seed"):
        return None
    stored_resolved = artifact.get("resolved_params")
    if not isinstance(stored_resolved, dict):
        return None
    for key, value in resolved.items():
        stored_value = stored_resolved.get(key)
        if isinstance(value, (int, float)) and isinstance(stored_value, (int, float)):
            if not math.isclose(float(value), float(stored_value), rel_tol=0.0, abs_tol=1e-12):
                return None
        elif value != stored_value:
            return None

    if method == "spectral":
        stored_selection = artifact.get("mr_state_selection")
        expected_selection = {
            "method": "spectral",
            "n_candidate_states": config["n_candidate_states"],
            "n_selected_states": config["n_selected_states"],
            "candidate_design": config["candidate_design"],
            "seed": config["sim_seed"],
            "n_restarts": config["selection_n_restarts"],
            "swap_passes": config["selection_swap_passes"],
            "variance_weight": config["selection_variance_weight"],
        }
        if not isinstance(stored_selection, dict):
            return None
        for key, value in expected_selection.items():
            stored_value = stored_selection.get(key)
            if isinstance(value, (int, float)) and isinstance(stored_value, (int, float)):
                if not math.isclose(float(value), float(stored_value), rel_tol=0.0, abs_tol=1e-12):
                    return None
            elif value != stored_value:
                return None

    state = artifact.get("mr_state")
    if state is None:
        return None
    state = np.asarray(state, dtype=np.float32)
    expected_rows = config["n_selected_states"] if method == "spectral" else config["n_clusters"]
    if state.shape != (expected_rows, n_mrs):
        return None
    return state.copy()


def regenerate_best(study_or_params, config: dict, final_missing_rate: float | None = None,
                     seed: int | None = None, output_path: str | None = None,
                     diagnostics: dict | None = None):
    """Re-run GRN generation + make_synthetic_data6 from a finished study's
    best_params (or an explicit flat params dict, e.g. loaded from
    config["output"]'s "best_params") -- see module docstring's "Accessing
    the best synthetic dataset". Unlike tuning during the study itself,
    missing_rate defaults to config["final_missing_rate"] (not 0.0), since
    the point here is to materialize an actual usable dataset rather than
    a fair-comparison candidate. If a matching spectral best.pt exists, its
    exact selected mr_state is reused; otherwise the configured sampler is
    run again.

    diagnostics: optional dict, forwarded to generate_sergio_grn_from_
                 reference's own `diagnostics` parameter (populated in-
                 place; see that function's docstring for its contents) --
                 e.g. to inspect the winner_mr/vote_margin/mr_load/etc.
                 behavior of whichever coherency_bias/canalization_
                 strength/balancing_strength params ended up in the
                 winning trial.

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
    sim_kwargs, grn_k_dist, hill_coeff_dist, unknown_mode_repressor_prob, grn_coherency_kwargs = _split_resolved(resolved)

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
        diagnostics=diagnostics,
        **grn_coherency_kwargs,
    )

    mr_state_method = config.get("mr_state_method", "random")
    mr_state_cache = {}
    mr_state_np = _load_matching_mr_state_artifact(
        config, resolved, mr_state_method, len(mr_ids)
    )
    mr_state_diagnostics = {"method": mr_state_method, "artifact_cache_hit": mr_state_np is not None}
    if mr_state_np is None:
        if mr_state_method == "spectral":
            dag = load_sergio_dag(
                grn_path,
                shared_coop_state=config["shared_coop_state"],
                mr_gene_ids=mr_ids,
            )
            mr_state_np = sample_mr_state_from_spectral_subset(
                dag=dag,
                n_selected_states=config["n_selected_states"],
                low=config["mr_rate_low"],
                high=config["mr_rate_high"],
                n_candidate_states=config["n_candidate_states"],
                candidate_design=config["candidate_design"],
                decays=sim_kwargs["decays"],
                n_restarts=config["selection_n_restarts"],
                swap_passes=config["selection_swap_passes"],
                seed=config["sim_seed"],
                variance_weight=config["selection_variance_weight"],
                cache=mr_state_cache,
                diagnostics=mr_state_diagnostics,
            )
        else:
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


def plot_grn_diagnostics(config: str | dict, path: str | None = None, figsize: tuple = (16, 8)):
    """
    Build an 8-panel figure summarizing the coherency_bias/
    canalization_strength/balancing_strength/path_decay mechanism's
    behavior across a finished (or in-progress) tuning study -- see module
    docstring's "Plotting GRN-coherency diagnostics" for a usage example.

    Args:
      config: either a path to a JSON tuning config (loaded via
              load_config) or an already-loaded config dict (e.g. from
              load_config directly, or run()'s own `config` object).
      path:   if given, the figure is saved here (dpi=120) and closed (same
              convention as synthetic_data.py's plot_summary_stats/
              sample_efficiency.py's _plot); otherwise the open Figure is
              returned for the caller to show/save/close itself.
      figsize: passed to plt.subplots.

    Data sources:
      - config["storage"]/config["study_name"] (via optuna.load_study) for
        every COMPLETE trial's cheap always-logged user_attrs
        (n_distinct_winners/top1_winner_share/frac_ambiguous_flipped/
        mean_canalization_alignment_frac/candidate_pca_explained_variance_ratio)
        plus that trial's own resolved coherency_bias/canalization_strength/
        balancing_strength (from user_attrs["resolved_params"]) and
        trial.value (the tuning distance) -- available for every completed
        trial regardless of whether grn_archive_dir was ever set.
      - config["grn_archive_dir"]'s trial{N}.diagnostics.json files (only
        used for panels 7-8 below), if that directory exists and contains
        at least one such file for a COMPLETE trial -- the archived file
        for whichever COMPLETE trial has the lowest study-wide distance is
        preferred (falling back to whichever archived COMPLETE trial has
        the lowest distance among just the archived ones, if the true
        overall best trial wasn't archived).
      - config["target_stats_path"], if set, purely for panel 6's reference
        PCA overlay (skipped, not an error, if unset/unreadable -- this
        function never attempts the expensive load_reference_h5ad path).

    Panels (2x4 grid):
      1. tuning distance (trial.value) vs. top1_winner_share, one point per
         completed trial -- is spreading out winners actually associated
         with a better (lower) distance, or unrelated/harmful?
      2. n_distinct_winners vs. balancing_strength -- does the fairness
         penalty actually spread out winners as balancing_strength grows?
      3. top1_winner_share vs. balancing_strength -- same question, in
         terms of the largest single MR's share rather than the count of
         distinct winners.
      4. frac_ambiguous_flipped vs. coherency_bias -- does coherency_bias
         actually sway more ambiguous-edge sign draws as it increases?
      5. mean_canalization_alignment_frac vs. canalization_strength -- for
         reference/sanity only (this fraction is a property of the fixed
         GRN topology's alignment under winner-take-all, not something
         canalization_strength itself changes -- see generate_sergio_grn_
         from_reference's docstring: canalization_strength only reweights
         magnitudes, it doesn't change which edges are "aligned"); a
         roughly flat scatter here is expected and fine.
      6. candidate_pca_explained_variance_ratio for the single best
         (lowest-distance) completed trial vs. the reference target's own
         ratio (if target_stats_path is set and readable) -- the actual
         PCA-spread outcome this whole mechanism is meant to move,
         log-y.
      7. mr_load bar chart for whichever trial panels 7-8 use (see "Data
         sources" above) -- per-master-regulator accumulated |strength|
         "load", sorted descending, showing winner-concentration directly.
      8. vote_margin histogram for that same trial -- winner-vs-runner-up
         vote-margin distribution across that trial's target genes (see
         generate_sergio_grn_from_reference's `diagnostics` docstring).

    Any panel lacking the data it needs (e.g. no completed trials, no
    target_stats_path, no grn_archive_dir) is left blank with a "no data"
    note rather than raising.

    Returns the Figure (already closed if `path` was given).
    """
    import matplotlib.pyplot as plt

    if isinstance(config, str):
        config = load_config(config)

    study = optuna.load_study(study_name=config["study_name"], storage=config["storage"])
    trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]

    def _no_data(ax, title):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    if not trials:
        for ax in axes.flat:
            _no_data(ax, "")
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=120)
            plt.close(fig)
        return fig

    distances          = np.array([t.value for t in trials], dtype=np.float64)
    top1_winner_share  = np.array([t.user_attrs.get("top1_winner_share", float("nan")) for t in trials])
    n_distinct_winners = np.array([t.user_attrs.get("n_distinct_winners", float("nan")) for t in trials])
    frac_flipped       = np.array([t.user_attrs.get("frac_ambiguous_flipped", float("nan")) for t in trials])
    mean_align_frac    = np.array([t.user_attrs.get("mean_canalization_alignment_frac", float("nan")) for t in trials])
    resolved_list       = [t.user_attrs.get("resolved_params", {}) for t in trials]
    coherency_bias      = np.array([r.get("coherency_bias", float("nan")) for r in resolved_list])
    canalization_strength = np.array([r.get("canalization_strength", float("nan")) for r in resolved_list])
    balancing_strength  = np.array([r.get("balancing_strength", float("nan")) for r in resolved_list])

    def _scatter(ax, x, y, xlabel, ylabel, title):
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            ax.scatter(x[valid], y[valid], s=15)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)

    # --- 1: distance vs. top1_winner_share ---
    _scatter(axes[0, 0], top1_winner_share, distances,
             "top1_winner_share", "distance (trial.value)", "distance vs. top1_winner_share")

    # --- 2: n_distinct_winners vs. balancing_strength ---
    _scatter(axes[0, 1], balancing_strength, n_distinct_winners,
             "balancing_strength", "n_distinct_winners", "n_distinct_winners vs. balancing_strength")

    # --- 3: top1_winner_share vs. balancing_strength ---
    _scatter(axes[0, 2], balancing_strength, top1_winner_share,
             "balancing_strength", "top1_winner_share", "top1_winner_share vs. balancing_strength")

    # --- 4: frac_ambiguous_flipped vs. coherency_bias ---
    _scatter(axes[0, 3], coherency_bias, frac_flipped,
             "coherency_bias", "frac_ambiguous_flipped", "frac_ambiguous_flipped vs. coherency_bias")

    # --- 5: mean_canalization_alignment_frac vs. canalization_strength ---
    _scatter(axes[1, 0], canalization_strength, mean_align_frac,
             "canalization_strength", "mean_canalization_alignment_frac",
             "mean_canalization_alignment_frac vs.\ncanalization_strength")

    # --- 6: best trial's candidate PCA vs. reference target PCA ---
    ax = axes[1, 1]
    finite_distances = np.where(np.isfinite(distances), distances, np.inf)
    best_idx = int(np.argmin(finite_distances)) if np.isfinite(finite_distances).any() else None
    any_data = False
    if best_idx is not None:
        best_trial = trials[best_idx]
        cand_pca = np.asarray(best_trial.user_attrs.get("candidate_pca_explained_variance_ratio", []), dtype=np.float64)
        if cand_pca.size:
            ax.plot(np.arange(1, cand_pca.size + 1), cand_pca, marker="o", ms=4,
                    label=f"best trial (#{best_trial.number})")
            any_data = True
    if config.get("target_stats_path"):
        try:
            with open(config["target_stats_path"], "rb") as f:
                target_stats = pickle.load(f)
            target_pca = np.asarray(target_stats.get("pca_explained_variance_ratio", []), dtype=np.float64)
            if target_pca.size:
                ax.plot(np.arange(1, target_pca.size + 1), target_pca, marker="s", ms=4, label="target")
                any_data = True
        except Exception:
            pass
    if any_data:
        ax.set_xlabel("PCA component")
        ax.set_ylabel("explained variance ratio")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    else:
        _no_data(ax, "")
    ax.set_title("best trial candidate vs. target PCA")

    # --- 7-8: single-trial detail panels from grn_archive_dir, if available ---
    archived_diag = None
    archived_trial_number = None
    if config.get("grn_archive_dir") and os.path.isdir(config["grn_archive_dir"]):
        archive_dir = Path(config["grn_archive_dir"])
        completed_by_number = {t.number: t for t in trials}
        archived_numbers = []
        for fname in os.listdir(archive_dir):
            m = re.match(r"trial(\d+)\.diagnostics\.json$", fname)
            if m and int(m.group(1)) in completed_by_number:
                archived_numbers.append(int(m.group(1)))
        if archived_numbers:
            # prefer the overall best trial if it was archived, else the
            # best-distance trial among just the archived ones
            if best_idx is not None and trials[best_idx].number in archived_numbers:
                archived_trial_number = trials[best_idx].number
            else:
                archived_trial_number = min(
                    archived_numbers, key=lambda n: completed_by_number[n].value,
                )
            with open(archive_dir / f"trial{archived_trial_number}.diagnostics.json") as f:
                archived_diag = json.load(f)

    # --- 7: mr_load bar chart ---
    ax = axes[1, 2]
    if archived_diag is not None and archived_diag.get("mr_load"):
        loads = sorted(archived_diag["mr_load"].values(), reverse=True)
        ax.bar(range(len(loads)), loads)
        ax.set_xlabel(f"master regulator rank (trial #{archived_trial_number})")
        ax.set_ylabel("mr_load")
    else:
        _no_data(ax, "")
    ax.set_title("mr_load (sorted)")

    # --- 8: vote_margin histogram ---
    ax = axes[1, 3]
    if archived_diag is not None and archived_diag.get("vote_margin"):
        margins = np.asarray(archived_diag["vote_margin"], dtype=np.float64)
        margins = margins[np.isfinite(margins)]
        if margins.size:
            ax.hist(margins, bins=30)
            ax.set_xlabel(f"vote_margin (trial #{archived_trial_number})")
            ax.set_ylabel("count")
        else:
            _no_data(ax, "")
    else:
        _no_data(ax, "")
    ax.set_title("vote_margin distribution")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


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
