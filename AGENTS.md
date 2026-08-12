# VAE Gene Expression Imputation

Single-file PyTorch research script (`vae-test.py`). No tests, no package structure. Modules not installable in this environment, so tests beyond syntax checking should not be run. Source control is also managed manually by user.

## Known Issues
- `generate_sergio_grn_from_reference` (synthetic_data.py) gained four new
  opt-in parameters -- `coherency_bias`, `canalization_strength`,
  `balancing_strength`, `path_decay` -- that bias cross-master-regulator
  gene parameterization toward "coherent feed-forward loop"/canalizing-
  function structure (see Alon/Kauffman), aiming to move PCA structure of
  make_synthetic_data6 output away from one dominant PC (see synthetic
  vs. reference PCA discrepancy: ~21% vs ~12% on PC1) toward richer
  multi-PC structure. All four default to values that reproduce the prior
  unbiased/i.i.d. behavior byte-for-byte (verified). Wired into
  tune_synthetic_data.py's search_space (see
  synthetic_tuning_config.20260810.*.json) but not yet validated against
  actual PCA output (requires torch/SERGIO, unavailable in this sandbox)
  -- an actual tuning run against real reference stats is the next step,
  to be done by the user outside this sandbox.
  `generate_sergio_grn_from_reference` also gained an opt-in `diagnostics`
  dict parameter (populated in-place, e.g. per-target winner_mr/
  vote_margin/n_ambiguous_flipped/n_aligned_edges/propagated_strength, and
  summary mr_load/n_distinct_winners/top1_winner_share) for inspecting
  this mechanism's behavior without re-deriving it externally; wired into
  tune_synthetic_data.py's run_trial (always logs the cheap scalar
  summaries -- n_distinct_winners/top1_winner_share/
  frac_ambiguous_flipped/mean_canalization_alignment_frac -- plus the full
  candidate_pca_explained_variance_ratio vector -- as trial.user_attrs;
  optionally archives the full per-trial GRN CSV + diagnostics JSON to
  `grn_archive_dir` if set in the tuning config, default disabled).
  `tune_synthetic_data.py` also gained `plot_grn_diagnostics(config, path=
  None, figsize=(16, 8))`, an 8-panel matplotlib figure (same
  return-Figure-or-save-and-close convention as synthetic_data.py's
  `plot_summary_stats`/sample_efficiency.py's `_plot`) summarizing this
  mechanism's behavior across a study's trials (distance/n_distinct_
  winners/top1_winner_share/frac_ambiguous_flipped/mean_canalization_
  alignment_frac vs. each of coherency_bias/canalization_strength/
  balancing_strength, best-trial candidate vs. target PCA, and -- only if
  `grn_archive_dir` has archived diagnostics for at least one completed
  trial -- an mr_load bar chart + vote_margin histogram for the best
  archived trial). Verified (via a real installed `optuna`+`matplotlib` in
  a throwaway venv, plus torch/tqdm stubs, since real torch/SERGIO are
  still unavailable here) against a synthetic Optuna study with fake
  trials/user_attrs and fake archived diagnostics JSON: all 8 panels
  render with sensible data, and graceful "no data" placeholders appear
  correctly when grn_archive_dir/target_stats_path are unset or the study
  has zero completed trials.
- A completed study (`synthetic_tuning_config.20260810.02.json`, 25 trials)
  was analyzed via its `plot_grn_diagnostics` figure. Findings: (1) the
  best trial's candidate PCA still undershoots the target's PC1 (~18% vs.
  ~33%, read off the log-scale plot) -- but per user direction, PC1
  dominance itself is **not** the thing to optimize for (it's presumed
  driven by structure -- e.g. between-cluster mean shifts -- orthogonal to
  the GRN-topology knobs being tuned); (2) `balancing_strength` was
  saturated across nearly its entire searched range `[0, 5]`
  (`n_distinct_winners` flat at ceiling except right near 0), wasting TPE
  budget; (3) `n_clusters=6` (fixed, not tuned) caps the rank of
  between-cluster mean-shift structure at 5, a plausible bottleneck for
  getting real multi-PC (PC2+) spread. Resulting changes (this session):
  - `synthetic_data.py` gained `pca_participation_ratio(ratio,
    skip_leading=1)`, a scale-invariant "effective number of components"
    metric over a PCA explained-variance-ratio vector's tail (PC2+ by
    default, PC1 excluded). `compute_summary_stats` now always returns
    this as `pca_tail_participation_ratio`. Verified standalone (flat
    tail -> ~len(tail); concentrated tail -> ~1; exactly scale-invariant;
    NaN on <2 finite/positive tail entries) -- not yet verified
    end-to-end with real torch/SERGIO output (unavailable in this
    sandbox).
  - `tune_synthetic_data.py`'s `run_trial`/`run` now also surface
    `pca_tail_participation_ratio` (candidate as a `trial.user_attr`,
    target via a startup print + a warning if an existing
    `target_stats_path` pickle predates this key and thus can't
    contribute to the weighted distance until regenerated). No change to
    `compute_stats_distance` itself was needed -- it's a plain scalar
    stat key and flows through the existing generic weighted-relative-
    squared-error mechanism.
  - New template config `synthetic_tuning_config.20260811.00.json`
    (cloned from `.02`): `weights.pca_explained_variance_ratio` -> `0.0`,
    `weights.pca_tail_participation_ratio` -> `6.0`,
    `search_space.balancing_strength` narrowed to `[0, 1.5]`,
    `n_clusters` 6 -> 15, `stats_n_pca_components` 10 -> 20 (explicit).
  - **User action required before running this config**: `target_stats_path`
    points at a not-yet-existing `....n_pca20.pickle` -- the existing
    reference pickle must be regenerated (rerun the averaging recipe in
    this module's `target_stats_path` docstring, using the *updated*
    `compute_summary_stats` and `n_pca_components=20`) against the real
    reference `.h5ad` (requires torch/SERGIO/the reference data, all
    unavailable in this sandbox), or `pca_tail_participation_ratio` will
    silently contribute nothing to the objective (the new startup-time
    print/warning in `run()` flags this if forgotten).
  - Also fixed an unrelated pre-existing indentation bug in
    `compute_summary_stats` (a `k = max(...)` line had drifted to 4-space
    indent, an `IndentationError` at import time) found while editing
    this area -- unrelated to the above, just a latent break in
    previously-uncommitted work.
- `generate_sergio_grn_from_reference` has a pre-existing (not introduced
  by the above) non-determinism bug: given the same `seed`, repeated calls
  in fresh Python processes can produce different GRN topologies/output,
  because `_sample_connected_subgraph`'s adjacency dict uses plain Python
  `set`s whose iteration order depends on `PYTHONHASHSEED` (randomized by
  default per-process). Confirmed by running the unmodified function 3x
  with `seed=42`: n_mrs varied (48/87/86) and output hashes differed;
  fixing `PYTHONHASHSEED=0` makes output fully reproducible across runs.
  Not yet fixed -- would need `_sample_connected_subgraph`/`_eades_dag_
  order`'s node iteration to use a deterministic order (e.g. sorted lists
  instead of sets) wherever it affects which edges survive DAG-ification.

