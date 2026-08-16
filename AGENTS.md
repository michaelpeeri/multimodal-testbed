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
    (placeholder-should-not-match)
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
- 8 completed tuning studies (`synthetic_tuning_20260811.0{0-7}`, based on
  `synthetic_tuning_config.20260811.00.json` cloned per-seed, `sim_seed`
  0-7, ~100-125 trials each) were analyzed (`.db`s + `grn_archive_dir`
  contents copied in from the user's machine; `best.pt` artifacts were
  already present). Found and fixed three real bugs in
  `tune_synthetic_data.py`, all of which affected these 8 studies:
  - **`_save_if_best_artifact` never actually protected the best trial**
    (confirmed 8/8): it read `{artifact_dir}/best.pt` back via
    `torch.load(path).get("distance", inf)` to decide whether to overwrite,
    wrapped in a bare `except Exception: prev = inf`. That read-back
    apparently raised on *every* call in all 8 studies (plausible cause:
    PyTorch >=2.6 defaults `torch.load(weights_only=True)`, which rejects
    unpickling the plain numpy arrays this artifact stores unless
    explicitly allow-listed) -- so every trial silently overwrote the
    previous one regardless of distance, and every `best.pt` ended up
    holding its study's *last* trial, not its true minimum-distance one
    (per-study true best, from each `.db`'s `trial_values` table, was
    2-800000x lower than what `best.pt` reported: e.g. study `.01`'s
    `best.pt` held trial 99 at distance 30838.7, but trial 84 -- already
    completed earlier in the same run -- had already reached 0.1565).
    Fixed by querying `trial.study.get_trials(states=(TrialState.COMPLETE,))`
    for the best value so far instead of round-tripping through the
    artifact file (tradeoff: no longer resume-safe across a wiped
    `artifact_dir` with a persisted `storage` db, unlike tuning.py's
    otherwise-identical `_save_if_best_checkpoint` -- use a fresh
    `study_name`/`storage` if that scenario applies). Verified with fake
    Trial/Study objects covering the exact regression case (a much-worse
    trial arriving after a genuinely-better one) plus tie/no-prior-trials
    edge cases.
  - **16 of `compute_summary_stats`' 32 scored keys (the entire
    `gene_mean_*`/`gene_var_*`/`mean_var_log_*` family) were silently
    scored at `compute_stats_distance`'s default weight of 1.0** because
    `synthetic_tuning_config.20260811.*.json`'s `weights` dict never
    mentioned them (omission != 0.0 -- see that function's own docstring,
    which already documented this default but apparently nobody had
    cross-checked the config against the full key list) -- ~25% of the
    total objective weight by accident, versus the deliberately-chosen
    `pca_explained_variance_ratio: 0.0`. Confirmed empirically dominant in
    7/8 studies' true-best trial's `top_distance_terms`. Fixed by adding
    explicit `0.0` weights for all 16 keys to the `.20260811.*` configs
    (a `_comment_excluded_keys` string entry documents why, directly in the
    JSON -- harmless, `compute_stats_distance` only weights keys that
    appear in a given trial's `terms`, never iterates `weights` itself).
  - **`target_stats_path`'s docstring regeneration recipe silently defaulted
    to `n_pca_components=10`** (never passed the argument at all) while the
    `.20260811.*` configs set `stats_n_pca_components: 20` -- so the
    `1cb67ec8-...stats.v2.mean.pickle` target's `pca_tail_participation_ratio`
    (weight 6.0, the single highest deliberately-weighted term) was computed
    over only 9 tail PCs (~7.68) while every candidate's was computed over
    19 (range up to ~19), not apples-to-apples. `run()`'s existing sanity
    check only verified the *key* was present, not that its depth matched.
    Empirically confirmed actively counterproductive: across all 741
    completed trials, `pca_tail_participation_ratio` vs. `distance` has
    Spearman rho=+0.25 -- candidates achieving *more* PC2+ spread (this
    mechanism's whole stated goal) were being scored as *worse*. Fixed the
    recipe to pass `n_pca_components=20` explicitly and added a startup
    check in `run()` comparing the loaded target's actual
    `pca_explained_variance_ratio` length against `stats_n_pca_components`
    (loud warning, not a hard error, to avoid breaking other configs that
    don't weight this term). **User action still required**: the target
    pickle itself must be regenerated with the fixed recipe
    (`n_pca_components=20`) against the real reference `.h5ad` before the
    next tuning round -- unavailable in this sandbox (needs
    torch/SERGIO/anndata + the reference data, all only in the user's own
    environment).
  - Also added a general (non-key-specific) fix to
    `compute_stats_distance`'s relative-error epsilon: real reference data
    legitimately has some target stats at exactly 0.0 (`gene_mean_p5`,
    `gene_var_p5`, `lib_size_zero_frac` all were, in the one reference
    sample seen so far), and the original fixed `eps=1e-6` floor made the
    plain relative-error formula explode to 10^5-10^6 for any such key
    whenever a candidate had even a tiny nonzero value there (this alone
    fully explained 3 of the 8 studies' `best.pt`-reported distances being
    ~30000-130000, compounding with the `_save_if_best_artifact` bug above).
    New `eps_frac`/`eps_abs_floor` params (both wired through `config`,
    defaults 0.05/0.02) floor the effective per-key epsilon at whichever is
    larger of: a fraction of that key's distributional-"family" scale (see
    `_stat_family` -- e.g. `gene_mean_p5`'s scale comes from its
    `gene_mean_mean/std/p25/p50/p75/p95` siblings), or a flat absolute floor
    (needed for single-member families whose own value is itself ~0, e.g.
    `lib_size_zero_frac` has no siblings at all). Verified against real
    saved candidate_stats: the two previously-30838.7 cases both drop to
    ~0.10-0.14 once rescored with the fix; a real (non-numerical-artifact)
    large miss in another candidate (`gene_corr_abs_p50` off by ~7.5x)
    correctly still shows up as a large term, confirming the fix suppresses
    the artifact without hiding genuine mismatches. Set `eps_frac`/
    `eps_abs_floor` to `0.0` to recover the original behavior.
  - Net effect of the above: the true best distance per study (from each
    `.db`, unaffected by the `_save_if_best_artifact` bug since Optuna's own
    `trial_values` table is independent of it) was actually a consistent
    0.12-0.28 across all 8 seeds already -- far more encouraging than the
    `best.pt`-reported 0.13-130208 spread suggested -- but every one of
    those true-best trials' own `top_distance_terms` still shows the
    gene_mean_*/gene_var_* and/or pca_tail_participation_ratio bugs as top
    contributors, so this number isn't trustworthy yet either; a clean
    re-run with all three fixes + a regenerated target pickle is needed
    before drawing conclusions about fit quality.
  - Cross-trial correlation analysis (741 completed trials, all 8 studies,
    using the always-logged per-trial `trial.user_attrs`) on whether the
    coherency/canalization/balancing mechanism works: `coherency_bias`
    correlates strongly with `frac_ambiguous_flipped` (Pearson r=0.72) and
    `balancing_strength` strongly anti-correlates with both
    `mean_canalization_alignment_frac` (r=-0.62) and `top1_winner_share`
    (r=-0.49) -- i.e. both knobs demonstrably do what they're mechanically
    designed to do. However neither shows a meaningful correlation with
    `pca_tail_participation_ratio` (|r|<=0.10) or PC1 ratio (|r|<0.06) --
    so achieving the mechanism's own internal goals (diversifying which MR
    "wins", biasing ambiguous edges) isn't clearly translating into the
    actual PCA-structure outcome it was built to influence (partly because
    the pca_tail_participation_ratio bug above was actively working against
    that signal). None of the four knobs correlate with PC1 ratio at all
    (|r|<0.12 across the board, n=741) -- a well-powered empirical
    confirmation of this file's own longstanding hypothesis (above) that
    PC1 dominance is driven by structure orthogonal to GRN topology, so
    this mechanism's realistic ceiling is the PC2+ tail, not PC1.
  - Next step (user, outside this sandbox): regenerate the target stats
    pickle with `n_pca_components=20`, then rerun the same 8-seed/~100-trial
    structure (e.g. `20260812.0{0-7}`) against the three fixes above for a
    clean, comparable baseline.
- The above baseline was run (`synthetic_tuning_20260812.0{0-7}`, based on
  `synthetic_tuning_config.20260812.0{0-7}.json` -- identical to the
  `.20260811.*` configs except the regenerated
  `...stats.v3.mean.pickle` target (`n_pca_components=20`, apples-to-apples
  with `stats_n_pca_components=20`) and `n_trials` 25->100) and analyzed
  (`.db`s + `best.pt` checkpoints copied in from the user's machine).
  Findings:
  - The three `.20260811` fixes hold up under independent verification:
    recomputing `compute_stats_distance` from each `best.pt`'s stored
    `candidate_stats` against the target pickle reproduces that artifact's
    stored `distance` exactly, and every artifact's `trial_number` matches
    its study's true argmin trial per the `.db`'s own `trial_values` table
    (i.e. `_save_if_best_artifact` is correctly protecting the true best
    trial in all 8 studies, unlike the `.20260811.*` studies before the fix).
  - True-best distance per study is now **0.010-0.070** across all 8 seeds
    -- a real improvement over `.20260811`'s already-fixed 0.12-0.28, not a
    metric artifact (the metric itself -- eps floor, key weights -- didn't
    change between the two rounds; only the target's PCA depth and the
    trial budget did).
  - `top_distance_terms` (logged per trial) is sorted by *raw*, not
    *weighted*, term value -- so it surfaces whichever keys have the
    largest unweighted relative error regardless of their configured
    weight, which in practice means it mostly shows the deliberately
    zero-weighted `gene_mean_*`/`gene_var_*` family rather than what's
    actually driving the objective. Not a bug (the field is documented as
    exactly this), but a trap for at-a-glance debugging -- recompute
    weighted contributions from a trial's/artifact's own `candidate_stats`
    (as done for this analysis) to see what's really driving `distance`.
  - Recomputing weighted contributions from all 8 `best.pt`s: the real
    remaining drivers are `gene_corr_abs_std`/`gene_corr_abs_p90` (top
    contributor in 6/8 studies) and `log_lib_size_std` (top few in most),
    with `pca_tail_participation_ratio` also prominent in about half.
    Both `gene_corr_abs_std`/`p90` (candidates ~0.06-0.11/0.15-0.25 vs.
    target 0.131/0.282) and `log_lib_size_std` (candidates 0.78-1.13 vs.
    target 1.43) undershoot the target in *every* study, and not because
    of a search-space boundary (best-trial `lib_size_scale` values, e.g.,
    sit at 0.69-1.13, well inside `[0.1, 2.0]`) -- plausibly a structural
    ceiling in `make_synthetic_data6`'s noise/correlation-generating
    mechanism rather than something more Optuna budget alone will fix;
    flagged for a manual investigation (fix everything else, sweep
    `noise_params`/`decays`/`dt` broadly, see if these two stats can reach
    target magnitude at all) before spending more tuning budget chasing
    them specifically.
  - `gene_mean_mean`/`gene_mean_std`/`gene_var_mean`/`gene_var_std` --
    still weight-0.0 (deliberately excluded, see `.20260811` entry above)
    -- are consistently only 15-50% of target magnitude (`gene_var_mean`/
    `gene_var_std` specifically) across all 8 studies: currently invisible
    to the optimizer, but a real and sizeable "does this actually resemble
    the reference" gap regardless.
  - A new **PCA tail *shape*** finding, from comparing best-trial
    candidates' full `pca_explained_variance_ratio` vector against the
    target's component-by-component (not just the scalar
    `pca_tail_participation_ratio`): across all 8 studies, PC3-PC9 come in
    at only ~45-60% of the target's per-component value, while PC16-PC20
    come in *above* target (~95-116%). I.e. the target's tail decays
    fairly steeply from PC2 through ~PC9 then flattens, while candidates'
    tails decay more gently but nearly uniformly across the whole PC2-20
    range -- a materially different *shape*, masked by
    `pca_tail_participation_ratio` scoring candidates at-or-above the
    target's own value (13.15) in every study (13.57-18.34). This is an
    inherent blind spot of that metric, not a bug: `pca_participation_ratio`
    is deliberately scale-invariant (see its docstring/`synthetic_data.py`),
    i.e. it measures how *evenly spread* the tail is, not which components
    carry the spread or whether their magnitudes match -- two differently
    -shaped tails (steep-early-flat-late vs. uniformly-flat) can have
    similar "evenness" and thus a similar score. Addressed (per user
    direction) by adding a second, shape-sensitive PCA term rather than
    replacing the first -- see below.
  - Cross-trial correlation analysis (658 completed trials, all 8 studies)
    updates the `.20260811` entry's version of this with the fixed target:
    internal coherence mechanics confirmed even more strongly
    (`coherency_bias` -> `frac_ambiguous_flipped` rho=+0.79;
    `balancing_strength` -> `mean_canalization_alignment_frac` rho=-0.89,
    -> `top1_winner_share` rho=-0.75, -> `n_distinct_winners` rho=+0.48 --
    all Spearman, all p<1e-38), but effect on the actual outcome the
    mechanism is meant to influence is still weak even with the
    apples-to-apples target: `|rho|<=0.18` for all three knobs vs.
    `pca_tail_participation_ratio`, `|rho|<=0.11` vs. PC1 ratio --
    statistically significant now (large n, fixed target) but still a
    minor lever in practice. New: `canalization_strength` itself weakly
    *anti*-correlates with its own `mean_canalization_alignment_frac`
    diagnostic (rho=-0.24, p=3e-10) -- not necessarily wrong (could be
    confounded with `balancing_strength` in the same trials, not checked
    via a partial correlation), but flagged as worth a closer look.
    Best-trial `coherency_bias`/`canalization_strength`/
    `balancing_strength` values are scattered across their full ranges
    across the 8 studies' best trials (e.g. `canalization_strength` from
    0.05 to 0.97) -- the objective doesn't strongly identify a preferred
    setting for any of the three, i.e. the landscape is fairly flat/noisy
    w.r.t. these knobs specifically.
  - `search_space.lib_size_mean` (`[3, 15]`) wastes real budget: trials
    with `lib_size_mean` in `[6, 15)` (~113/658, ~17% of all completed
    trials across the 8 studies) average `distance` 0.3-1.7 vs. ~0.09-0.2
    for `[3, 5)` -- narrowed in the `.20260813.*` configs below.
  - Running-best-so-far trajectories (per study, from each `.db`'s
    per-trial `value`s in trial-number order) show several studies still
    made large jumps very late in their 100-trial budget (e.g. study 0:
    0.022->0.010 around trial 90-98; study 5: 0.068->0.034 around trial
    89; study 7: 0.030->0.026 around trial 98) -- not clearly plateaued,
    motivating the `n_trials` increase below.
  - Resulting changes (this session), all in `tune_synthetic_data.py` +
    new `synthetic_tuning_config.20260813.0{0-7}.json` templates (cloned
    from `.20260812.*` the same way `.20260811`->`.20260812` was):
    - New `_add_pca_pc2_9_key(stats)` helper: derives
      `pca_pc2_9_explained_variance_ratio` (PC2..PC9, 8 values) as a plain
      slice of the already-present `pca_explained_variance_ratio` array,
      mutating `target_stats` (in `run()`, right after it's loaded) and
      `candidate_stats` (in `run_trial()`, right after it's computed) in
      place -- no `synthetic_data.py`/`compute_summary_stats` change or
      target-pickle regeneration needed, since this only rearranges data
      `compute_summary_stats()` already returns; flows through
      `compute_stats_distance`'s existing generic array-key handling
      unmodified. Verified against the real `.v3.mean.pickle` target and
      a real `best.pt`'s `candidate_stats` (study 0): slices out
      `[0.0593, 0.0447, ..., 0.0202]` correctly (8 values, PC2-PC9) from
      the target's 20-length vector, and folding the new term (weight 6.0,
      matching `pca_tail_participation_ratio`) into `compute_stats_distance`
      moves that trial's distance from 0.01019 to 0.01525 as expected
      (meaningful but not dominant).
    - `synthetic_tuning_config.20260813.0{0-7}.json`: `weights.
      pca_pc2_9_explained_variance_ratio` -> `6.0` (new, matching
      `pca_tail_participation_ratio`'s weight so both PCA-related terms
      co-exist at equal weight -- one for tail *evenness*, one for PC2-9
      *magnitude*); `weights.gene_mean_mean`/`gene_mean_std`/
      `gene_var_mean`/`gene_var_std` `0.0` -> `1.0` (the remaining
      percentile duplicates and `mean_var_log_*` stay excluded);
      `search_space.lib_size_mean` `[3, 15]` -> `[3, 6]`; `n_trials`
      `100` -> `150`. Everything else (target pickle, `n_clusters`,
      search ranges for other knobs, etc.) unchanged from `.20260812.*`.
    - **User action required before running**: these `.20260813.*`
      studies themselves (needs torch/SERGIO, unavailable in this
      sandbox) -- not yet run or analyzed as of this session.
- The `synthetic_tuning_20260813.0{0-7}` studies (8 seeds,
  `synthetic_tuning_config.20260813.0{0-7}.json`, 150 trials each) were run
  and analyzed (`.db`s + `checkpoints/best.pt` + `grn_diags/` copied in from
  the user's machine). Unlike prior analysis sessions, a real `torch`/
  `optuna`/`scipy`/`matplotlib` venv was already present at
  `/tmp/opencode/tuning_venv` (not a stub), so `best.pt` could be loaded and
  `compute_stats_distance` recomputed directly rather than relying solely on
  logged `user_attrs`. Findings:
  - The `.20260812` fixes continue to hold: every study's `best.pt`
    `trial_number`/`distance` matches its `.db`'s true argmin trial exactly,
    and recomputing `compute_stats_distance` from each `best.pt`'s
    `candidate_stats` reproduces the stored `distance` exactly.
  - **Major PCA-structure improvement over `.20260812`** (attributable to
    `n_clusters` 6->15 plus the `pca_pc2_9_explained_variance_ratio` term,
    both introduced in this config): averaged over the 8 best trials,
    candidate PC1 now reaches ~86% of target explained-variance-ratio (vs.
    the ~18%/33% ~= 54% gap that originally motivated this whole coherency/
    canalization effort -- see the `.20260810` entry above), and PC3-PC9
    reach ~87-91% of target (vs. ~45-60% in `.20260812`). PC10-PC20 now
    mildly *overshoot* (110-127%) rather than matching -- a minor residual
    tail-shape mismatch, not the dominant-PC1 problem.
  - **`gene_var_mean`/`gene_var_std` (re-weighted 1.0 this round) are now
    the single largest remaining gap and dominant weighted `distance`
    contributor in 6/8 studies**: candidates average only ~20%/~37% of
    target magnitude respectively (range 6-34% / 11-51% across the 8 best
    trials) -- a much larger proportional miss than any other scored key.
    `gene_mean_mean` (~48% of target) and `gene_corr_abs_std`/`p90` (~73%/
    ~78%) are smaller but still-notable undershoots.
  - `log_lib_size_std` again undershoots in *every* study (69% of target on
    average, tight 56-87% range) -- same pattern as `.20260812`, initially
    flagged there as a possible "structural ceiling". This session's data
    complicates that: across the 8 best trials, `lib_size_scale` (the
    knob controlling this stat) correlates with `log_lib_size_std` at
    Pearson r=0.98 (n=8, not itself strong evidence, but a strikingly clean
    fit) -- yet none of the 8 best trials used `lib_size_scale` above 1.16,
    well short of the search space's 2.0 upper bound. I.e. this may not be
    a hard simulator ceiling after all, just an unexplored/deprioritized
    region under the current weight balance -- see the `.20260814.*`
    config change below, which tests this directly by widening the bound.
  - Cross-trial correlation analysis (1081 completed trials, all 8
    studies) updates the `.20260812` entry's version of this: internal
    coherence mechanics remain strongly confirmed and if anything stronger
    (`coherency_bias` -> `frac_ambiguous_flipped` Spearman rho=+0.86;
    `balancing_strength` -> `mean_canalization_alignment_frac` rho=-0.69,
    -> `top1_winner_share` rho=-0.51; all p<1e-70). New this round:
    `canalization_strength` now shows a real, if modest, correlation with
    the actual PCA outcome it's meant to influence -- `pca_tail_
    participation_ratio` rho=-0.29 (p=5e-22), `pc1_ratio` rho=-0.12 -- both
    clearly nonzero for the first time (vs. `|rho|<=0.18`/`<=0.11`
    respectively in `.20260812`), though still weak relative to `distance`
    itself (`|rho|<=0.13` for all three knobs vs. `distance`). Best-trial
    `coherency_bias`/`canalization_strength`/`balancing_strength` values
    remain scattered across their full ranges (e.g. `canalization_strength`
    from 0.05 to 0.90) with diffuse winner distributions at the best trials
    (`top1_winner_share` only 5-15%) -- the objective still doesn't
    strongly prefer a particular setting of these three knobs. Net
    read: the mechanism demonstrably works mechanically and now
    demonstrably moves the PCA-tail outcome a little, but remains a
    second-order lever relative to `gene_var_*`/`gene_corr_abs_*`/
    `log_lib_size_std` for overall fit quality.
  - No search-space boundary-saturation problems this round -- the
    `.20260813` `lib_size_mean` narrowing to `[3, 6]` worked as intended
    (best-trial values now 4.1-5.1, comfortably interior; this had been the
    single largest wasted-budget issue identified from `.20260811`).
    Weak-to-moderate Spearman correlations with final `distance` across all
    1081 completed trials: `outlier_prob` rho=-0.30, `noise_params`
    rho=-0.21, `cluster_conc` rho=+0.20, `grn_k_low` rho=-0.15,
    `coherency_bias` rho=+0.13 (all p<1e-4) -- none dominant, none at a
    search-space boundary.
  - Running-best-so-far trajectories (per study, from each `.db`'s
    per-trial values in trial-number order): 5/8 studies still improved by
    more than 5% of their trial-80%-mark best value within their final 20%
    of trials, and two (studies 1 and 5) improved by 35% and 42%
    respectively as late as trial 147/129 of 150 -- clear evidence the
    150-trial budget was cut short of convergence for most seeds, distinct
    from `.20260812`'s late-jump observations (which were smaller and
    fewer) -- motivating the `n_trials` increase below.
  - Study 6 was the worst-performing seed by a wide margin (`distance`
    0.0675, ~2x the best of 0.0317) and also had the single worst
    `gene_var_mean` shortfall (5.7% of target vs. 17-34% for the other 7) --
    its best trial also has the lowest `noise_params` (0.33) of all 8
    studies' best trials, suggestive of a `noise_params`/`gene_var_mean`
    relationship (also weakly visible pooled across the 8 best trials:
    Pearson r=0.56 for `noise_params`, r=0.48 for `decays`, neither
    significant at n=8) but not conclusively distinguishable from ordinary
    per-seed TPE variance without more data (see the `candidate_stats.json`
    archiving change below, aimed at exactly this).
  - Reconfirmed the `.20260812` entry's `top_distance_terms` "trap" at full
    scale rather than by spot-check: pooled across all 1081 completed
    trials, the field's raw-value ranking shows a *weight-0.0* key
    (`gene_var_p95`, `gene_corr_abs_p50`, or `gene_mean_p25` -- all
    deliberately excluded from the objective) as the #1 entry in 1025/1081
    trials (~95%), and specifically the single most common #1 entry
    (`gene_var_p95`, 54.4% of all trials) is never itself scored at all.
    Fixed below (this session) rather than just re-documented.
  - Resulting changes (this session), all in `tune_synthetic_data.py` +
    new `synthetic_tuning_config.20260814.0{0-7}.json` templates (cloned
    from `.20260813.*` the same way `.20260812`->`.20260813` was):
    - `run_trial` now also archives each trial's full `candidate_stats`
      dict (via new helper `_jsonify_stats`, which converts numpy arrays/
      scalars to plain JSON types) to
      `{grn_archive_dir}/trial{N}.candidate_stats.json`, guarded by the
      same `config["grn_archive_dir"]` check as the existing GRN CSV/
      diagnostics archiving. Previously only the single best trial's
      `candidate_stats` was ever persisted (in `best.pt`, one data point
      per study) -- this gives full per-trial access (n~1000+ pooled
      across a study) for exactly the kind of "which knob actually drives
      `gene_var_mean`/`log_lib_size_std`/etc." correlation analysis this
      session had to limit to n=8. Verified with a standalone
      `_jsonify_stats` round-trip (numpy array/np.float64/plain-float mix
      -> `json.dumps` without error) and a standalone re-implementation of
      the exact sort key used below, confirming a zero-weighted key with a
      large raw term is correctly demoted below smaller-but-weighted terms.
      `plot_grn_diagnostics` was *not* updated to consume these new files
      (out of scope this session) -- still an opportunity for a future
      panel.
    - `run_trial`'s `top_distance_terms` user_attr is now ranked by
      *weighted* contribution (`weight * raw_term`) instead of raw
      `raw_term` alone, fixing the trap above -- `breakdown["terms"]`
      itself (unweighted) is still available to recompute the old
      behavior if ever needed. No stored key name changed, so this is not
      backwards-compatible for anyone diffing historical `top_distance_
      terms` values across the `.20260813`/`.20260814` boundary, but
      nothing in this codebase currently parses that field programmatically
      (grepped `tune_synthetic_data.py` to confirm before changing it).
    - `synthetic_tuning_config.20260814.0{0-7}.json`:
      `search_space.lib_size_scale` upper bound `2.0` -> `3.5` (direct test
      of the ceiling-vs-unexplored-region question above); `n_trials`
      `150` -> `250` (budget-convergence finding above). Everything else
      (weights, other search-space bounds, target pickle, `n_clusters=15`)
      intentionally unchanged from `.20260813.*` to isolate these two
      variables rather than confound them with other changes.
      `coherency_bias`/`canalization_strength`/`balancing_strength`/
      `path_decay` search ranges were deliberately *not* narrowed despite
      their weak correlation with `distance`, since they now show a real
      (if modest) effect on `pca_tail_participation_ratio`/`pc1_ratio` and
      remain the feature under active investigation.
    - **User action required before running**: these `.20260814.*` studies
      themselves (needs torch/SERGIO, unavailable in this sandbox) -- not
      yet run or analyzed as of this session.
- The `synthetic_tuning_20260814.0{0-7}` studies (8 seeds,
  `synthetic_tuning_config.20260814.0{0-7}.json`, 250 trials each) were run
  and analyzed (`.db`s + `checkpoints/best.pt` + `grn_diags/`, including the
  new per-trial `candidate_stats.json` archives, copied in from the user's
  machine). Findings:
  - **Major, confirmed PCA-structure win**, attributable to `n_clusters`
    6->15 (`.20260813`) plus the `pca_pc2_9_explained_variance_ratio` term
    (`.20260813`): averaged over the 8 best trials, candidate PC1 now
    reaches ~101% of target (vs. the ~54% gap that originally motivated the
    whole coherency/canalization effort), and PC3-PC9 reach ~74-94% of
    target (vs. ~45-60% in `.20260812`). True-best `distance` per study is
    now **0.022-0.054** (vs. `.20260813`'s 0.032-0.068).
  - **New dominant gap**: `gene_var_mean`/`gene_mean_mean` (candidates
    average only ~18%/~40% of target magnitude across the 8 best trials) is
    now the single largest weighted `distance` contributor in **8/8**
    studies (previously masked by the larger PCA gap). `gene_var_std`
    (~41%), `gene_corr_abs_std`/`p90` (~73%/~80%), and `log_lib_size_std`
    (~81%) are smaller but consistent secondary undershoots.
  - **Confirmed real (not unexplored-space) structural ceiling on
    `lib_size_scale`**: this config's upper-bound widening `2.0` -> `3.5`
    (specifically to test this) was never actually explored by TPE --
    **100% of the 250 trials with `lib_size_scale > 2.0`** (and 80% of
    those in `[1.5, 2.0]`) were hard-pruned via the `max_lib_size_zero_frac`
    guard (`degenerate_lib_size_zero_frac`), before ever reaching the soft,
    weighted objective. Since `lib_size_scale` is also the single strongest
    driver of `gene_var_std`/`gene_corr_abs_std`/`log_lib_size_std`
    (Spearman rho 0.65-0.92 pooled across 1750 completed trials), this hard
    guard -- not a lack of trial budget -- is what's currently capping those
    stats. `max_lib_size_zero_frac` raised `0.05` -> `0.10` in
    `.20260815.*` (below) to test this directly.
  - **Is there a real tradeoff between PCA structure and gene_mean/var/
    dropout stats, or just an over-weighted objective?** Investigated in
    depth (user question) and confirmed it's a **real, fairly hard,
    knob-independent tradeoff, not primarily a weight-balance artifact**:
    a joint rank-regression of `pca_pc2_9_explained_variance_ratio`-ratio-
    to-target (`pc29_ratio`) and `gene_mean_mean`/`gene_var_mean`-ratio-to-
    target on all 18 search-space knobs shows several knobs with *flipped
    signs* (`lib_size_mean`: -0.34 on `pc29_ratio` vs. +0.73/+0.68 on
    gene_mean/var ratio; `lib_size_scale`: -0.25 vs. +0.26/+0.42;
    `outlier_prob`/`outlier_scale`: positive on `pc29_ratio`, negative on
    gene_mean/var ratio). Binning all 1750 completed trials into deciles of
    `pc29_ratio` shows a clean monotonic inverse relationship across the
    *entire* range (decile 0 `pc29_ratio=0.35` -> `gene_var_ratio=0.44`;
    decile 9 `pc29_ratio=1.18` -> `gene_var_ratio=0.13`), and **zero of the
    1750 trials** achieve both `pc29_ratio>=0.9` and `gene_var_ratio>=0.5`
    simultaneously (best PCA-matching trials cap out at `gene_var_ratio`
    ~=0.41; best gene_var-matching trials cap out at `pc29_ratio`~=0.66).
    Critically, this negative correlation barely weakens when statistically
    controlling for the knobs above (partial rank-corr rho=-0.55, vs. -0.50
    raw) or when restricting to a narrow `lib_size_mean` band (rho=-0.66,
    n=619) -- so it isn't merely "TPE happened to pick confounded
    combinations," it's close to a real Pareto frontier in the parameter
    space actually explored. However, `canalization_strength`/`decays` move
    `pc29_ratio` with much less cost to gene_mean/var than the knobs above,
    and all of the implicated knobs (`lib_size_mean/scale`, `outlier_prob/
    scale`) are *post-hoc technical-noise* stages applied after SERGIO's
    biological simulation -- so this may reflect an upstream biological-
    magnitude ceiling (the fixed `mr_rate_low/high=[1,5]` MR production-rate
    range, never varied in any study to date) forcing technical-noise knobs
    to compensate, rather than an unfixable property of the simulator. See
    the `.20260815.pilot_mr{5,6,8}.*` pilot studies below, which test this
    directly.
  - `canalization_strength`'s effect on the actual mechanism/outcome is
    now on firmer footing than prior rounds: a rank-based multiple
    regression of `distance` on all 18 knobs jointly shows a real,
    significant, independent effect (partial coef -0.111, p<0.001) --
    higher canalization -> lower distance, net of everything else -- a
    materially more confident result than `.20260813`'s marginal-only
    correlations (still second-order next to `outlier_prob/scale`/
    `dropout_shape`/`lib_size_scale/mean`, each |coef| 0.12-0.32).
    `coherency_bias` shows a small but significant *adverse* net effect
    (+0.076, p=0.002) once other knobs are controlled for -- worth
    watching, not yet acted on. Also, `.20260813`'s flagged anti-
    correlation between `canalization_strength` and its own `mean_
    canalization_alignment_frac` diagnostic (rho=-0.24, "worth a closer
    look") has now flipped to a real *positive* correlation (marginal
    rho=+0.177, p=9e-14; still +0.088, p=2e-4 after partialling out
    `balancing_strength`/`coherency_bias`/`path_decay`) -- likely a side
    effect of the `n_clusters` 6->15 change, and a reassuring sign the
    mechanism's internal mechanics are behaving more sensibly, not less,
    as the surrounding config has evolved.
  - `.20260813`'s hypothesis attributing study 6's poor performance partly
    to a low `noise_params` value in its best trial does not hold up under
    this session's much larger pooled sample (n=1750 vs. the n=8 used
    before): `noise_params` is not a significant net predictor of either
    `distance` or `gene_var_mean` once other knobs are controlled for --
    that specific hypothesis should be considered retracted.
  - Resulting changes (this session): new `synthetic_tuning_config.
    20260815.0{0-7}.json` (`max_lib_size_zero_frac` `0.05` -> `0.10`;
    `n_trials`/`timeout` both `null`, see the interrupt/resume changes
    below; `mr_rate_high` deliberately left unbumped at `5.0` pending the
    pilot below) and nine new pilot configs `synthetic_tuning_config.
    20260815.pilot_mr{5,6,8}.0{0,1,2}.json` (3 seeds x `mr_rate_high` in
    {5.0 (baseline), 6.0, 8.0}, `n_trials=80`, `max_lib_size_zero_frac` left
    at the `.20260814` baseline `0.05` to isolate `mr_rate_high` as the only
    variable under test). Also added to `tune_synthetic_data.py`:
    - `_install_graceful_stop_handler(study)`: installs SIGINT/SIGTERM
      handlers that call `Study.stop()` (lets the in-flight trial finish
      normally, then falls through to `run()`'s usual post-loop summary-
      writing) on the first signal, and raise `KeyboardInterrupt` directly
      on a second signal for an immediate exit. Wired into `run()` around
      the `study.optimize()` call. Motivated by: (1) a raw, unhandled
      Ctrl+C mid-trial wastes whatever fraction of that trial's multi-
      minute SERGIO run was in flight, and (2) it skips `run()`'s summary-
      writing code entirely (only reached if `study.optimize()` returns
      normally) even though the study's own storage/`best.pt` are already
      safely persisted regardless. Verified with a real (non-stub) Optuna
      study in `/tmp/opencode/tuning_venv` using a fake objective and
      `os.kill(os.getpid(), signal.SIGINT)` to simulate an external
      interrupt mid-trial: (a) a single SIGINT lets the in-flight trial
      complete normally and stops the loop right after it, no trials lost;
      (b) resuming the same study object (or, separately, a fresh Python
      process against the same sqlite storage) for more trials afterward
      correctly continues from the existing trial count; (c) a second
      SIGINT forces an immediate `KeyboardInterrupt` (that one trial marked
      FAIL, as expected); (d) `n_trials=None` (unbounded) runs until
      stopped by a signal, not just until an arbitrary large count.
    - `config["n_trials"]`/`config["timeout"]` may now be `null` (Python
      `None`) for an unbounded run (`study.optimize(n_trials=None,
      timeout=None)`, stopped only via the above signals) -- validated in
      `load_config` (must be `null` or a positive int/number). Intended
      workflow (documented in the module docstring's new "Interrupting/
      resuming a run" section and in `run()`'s own docstring): launch each
      seed's process with an unbounded config and let it run for as long as
      you want, rather than being bound to a fixed `n_trials` that some
      seeds exhaust much sooner than others (per `.20260813`/`.20260814`,
      several seeds still improved meaningfully in their final 20% of a
      fixed trial budget) -- then SIGINT/SIGTERM individual processes once
      satisfied with their progress. Caveat documented alongside this:
      `build_sampler()` constructs a fresh `TPESampler(seed=0)` on every
      `run()` call, so a study stopped/resumed multiple times will suggest
      a slightly different trial sequence than one uninterrupted run of the
      same eventual trial count (both valid, history-informed TPE behavior,
      just not bit-for-bit identical).
    - **User action required**: run the `.20260815.pilot_mr{5,6,8}.*` pilot
      studies first (needs torch/SERGIO, unavailable in this sandbox) and
      compare their `gene_mean_ratio`/`gene_var_ratio`/`pc29_ratio` before
      deciding whether to bump `mr_rate_high` in the full `.20260815.0{0-7}`
      sweep (currently left at the unbumped default) and launching it.
- The `synthetic_tuning_20260815_pilot_mr{5,6,8}.0{0,1,2}` pilot studies (9
  studies: 3 seeds x `mr_rate_high` in {5.0 baseline, 6.0, 8.0}, 80 trials
  each, `max_lib_size_zero_frac` left at the `.20260814` baseline 0.05 to
  isolate this one variable) were run and analyzed (`.db`s +
  `checkpoints/best.pt` copied in from the user's machine). Sanity-checked:
  each study's `best.pt`'s `mr_state` correctly spans `[1, mr_rate_high]` as
  configured, and recomputed `compute_stats_distance` reproduces every
  stored `distance` exactly. Findings:
  - **`mr_rate_high=8.0` confirmed as a real, fairly clean win**: averaged
    over 3 seeds, `gene_mean_mean`/`gene_var_mean` ratio-to-target nearly
    doubled (0.377->0.690, 0.152->0.278 vs. the `mr_rate_high=5.0`
    baseline) with **no measurable cost** to `pca_pc2_9_explained_variance_
    ratio` match (flat ~0.64-0.67 across all three `mr_rate_high` values
    tested). Notably, `mr_rate_high=8`'s per-seed values are tightly
    clustered (`gene_mean_ratio` 0.68-0.70, `gene_var_ratio` 0.267-0.287)
    versus the noisier, more seed-dependent `mr_rate_high=5`/`6` groups
    (e.g. `gene_mean_ratio` 0.32-0.44 / 0.31-0.69) -- a reliable effect,
    not a lucky seed. Best-trial `lib_size_scale` also fell monotonically
    (0.83->0.75->0.62) as `mr_rate_high` rose 5->6->8, consistent with TPE
    substituting away from the lib_size_scale/lib_size_mean knobs shown
    (`.20260814` entry above) to be in real tension with PCA structure,
    rather than this being an independent additional effect. Caveat: only
    3 seeds/pilot value -- a real, consistent-looking effect, but not a
    large-sample confirmatory result. Resulting change: `mr_rate_high`
    bumped `5.0` -> `8.0` in the full `synthetic_tuning_config.
    20260815.0{0-7}.json` sweep configs (`max_lib_size_zero_frac` there
    remains at `0.10` per the `.20260814` entry above -- both changes now
    combined in that sweep).
  - **New, more consequential finding while sanity-checking the pilot**:
    `n_mrs` (a direct readout of GRN topology) varied substantially and
    unsystematically across **all 9** pilot studies (43-100) despite every
    one of them sharing the same fixed `grn_seed=42` -- and checking back,
    the original `synthetic_tuning_20260814.0{0-7}` studies show the exact
    same issue (`n_mrs` 40-104 across the 8 nominally-same-topology
    seeds). This is the pre-existing, already-documented-but-not-yet-fixed
    `_sample_connected_subgraph` non-determinism bug (see this file's
    earlier Known Issues entry: plain Python `set` iteration order depends
    on the interpreter's per-process-randomized `PYTHONHASHSEED`) --
    confirmed here, for the first time, to have actually manifested across
    every real multi-seed study batch run so far (each seed/pilot-arm runs
    in its own OS process), not just in the original 3x-repro isolated
    test. Practical impact: every "per-seed" comparison made in this
    project to date (including `.20260814`'s "study 6 was the worst seed"
    observation, and this pilot's own mr5/mr6/mr8 groups) has had GRN
    topology as an uncontrolled extra confound alongside the intended
    `sim_seed` variation -- though the `mr_rate_high=8` finding above held
    up despite it, which if anything strengthens confidence in that result.
    Resulting change: `tune_synthetic_data.py`'s `run()` now calls a new
    `_warn_if_pythonhashseed_unset()` at startup, printing a loud warning
    if `PYTHONHASHSEED` isn't set in the current process's environment
    (can only warn/remind, not enforce or fix retroactively -- the hash
    seed is fixed at interpreter startup, before this module's code ever
    runs, so it cannot be corrected for from inside `run()`; every sibling
    process for a given round must be launched with an identical
    `PYTHONHASHSEED=<value>` environment variable set *before* invoking
    python). Documented alongside `grn_seed` in the module docstring's
    "Fixed vs tunable parameters" section too. **User action required**:
    launch each of the 8 `.20260815.0{0-7}` sweep processes with a shared
    `PYTHONHASHSEED` value (e.g. `PYTHONHASHSEED=0`) for `grn_seed=42` to
    actually produce identical topology across all 8 this time, unlike
    every prior multi-seed batch.
- The `synthetic_tuning_20260815.0{0-7}` studies (8 seeds,
  `synthetic_tuning_config.20260815.0{0-7}.json` -- `mr_rate_high` bumped
  5.0->8.0 per the pilot above, `max_lib_size_zero_frac` 0.05->0.10,
  unbounded `n_trials`/graceful-stop workflow instead of a fixed 250 --
  each seed's process actually run to completion at 242-331 trials, not
  interrupted) were analyzed (`.db`s + `checkpoints/best.pt` + `grn_diags/`,
  including per-trial `candidate_stats.json`, copied in from the user's
  machine). The user ran each of the 8 processes with a shared
  `PYTHONHASHSEED`, confirmed working for the first time: `n_mrs=84`
  identically across **every one of 2394 pooled trials in all 8 studies**
  (vs. 40-104 varying per study in every prior multi-seed batch) -- GRN
  topology is no longer a confound in this round's cross-seed comparisons.
  Also rendered `plot_grn_diagnostics` for all 8 studies for the first time
  against *real* (non-synthetic-stub) data --
  `plots/tune_diags_20260815.0{0-7}.png` -- all 8 panels show sensible,
  real data (e.g. study 06's panel 2 shows `n_distinct_winners` climbing
  from 49 to a hard ceiling of 69 as `balancing_strength` approaches its
  upper search bound 1.5, and panel 6 tracks the target PCA curve closely
  through PC1-13 before under-shooting the tail). Findings:
  - Sanity checks clean throughout (unlike several earlier rounds): every
    study's `best.pt` `distance`/`candidate_stats` reproduce exactly via
    `compute_stats_distance`, and every `best.pt`'s `trial_number` matches
    its `.db`'s true argmin.
  - True-best `distance` per study: **0.027-0.060** (mean 0.038) -- in the
    same range as `.20260814`'s 0.022-0.054 (not a regression), but *not*
    the improvement the `mr_rate_high` pilot predicted (see below).
    `gene_var_mean`/`gene_mean_mean` remain the single largest weighted
    `distance` contributor in **8/8** studies (candidates average only
    ~20%/~44% of target magnitude, essentially unchanged from
    `.20260814`'s ~18%/~40%); PCA structure is also essentially unchanged
    (PC1 ratio-to-target averages 92% over the 8 best trials, range
    68-120%; PC2-9 averages 87%, range 55-110%). New minor finding:
    `zero_frac` (overall dropout) is consistently **2-5 percentage points
    higher than target in all 8 studies** (0.945-0.971 vs. target 0.923,
    i.e. candidates are slightly *more* sparse than the reference) --
    currently weighted (3.0) but never a top contributor, a secondary gap
    rather than a driver.
  - **The `mr_rate_high=8` pilot's conclusion did not replicate at full
    scale**: comparing each of the 8 `.20260815` `best.pt`s directly
    against its same-seed `.20260814` counterpart (`mr_rate_high=5`, only
    difference besides `max_lib_size_zero_frac`), `gene_mean_mean`/
    `gene_var_mean` ratio-to-target moved only 0.401->0.437 /
    0.183->0.199 on average (vs. the pilot's 0.377->0.690 / 0.152->0.278
    over 3 seeds/80 trials) -- a much smaller effect, and overall
    `distance` was flat-to-slightly-worse (0.0366->0.0375 average), with
    `log_lib_size_std` ratio and `pc1_ratio` both moving in the wrong
    direction (0.814->0.760, 1.011->0.921). Confound: this sweep changed
    `max_lib_size_zero_frac` (0.05->0.10) in the *same* config change as
    `mr_rate_high`, unlike the pilot which isolated `mr_rate_high` alone --
    so this isn't a clean refutation of the pilot's finding either, just
    evidence the two changes together didn't deliver the hoped-for
    combined win. **Recommend re-testing `mr_rate_high` in true isolation
    at full trial budget** (not just an 80-trial/3-seed pilot) before
    relying on it further.
  - **The PCA-structure-vs-gene-magnitude tradeoff first identified in
    `.20260814` is confirmed and, if anything, stronger** with this
    round's larger, topology-controlled pooled sample (n=2137 completed
    trials, all 8 studies, using the per-trial `candidate_stats.json`
    archives for full resolution rather than just the 8 best trials):
    `pca_pc2_9_explained_variance_ratio`-ratio-to-target
    (`pc29_ratio`) vs. `gene_var_mean`-ratio-to-target: raw Spearman
    rho=-0.61 (was -0.50 in `.20260814`), and a partial correlation
    controlling for the four knobs `.20260814` implicated
    (`lib_size_mean/scale`, `outlier_prob/scale`) is *stronger*, not
    weaker (rho=-0.68) -- ruling those four knobs out as the full
    explanation for the tradeoff. Only **1 of 2137** completed trials
    achieved both `pc29_ratio>=0.9` and `gene_var_mean_ratio>=0.5`
    simultaneously (best PCA-matching trials cap out at
    `gene_var_mean_ratio`~=0.67; best gene_var-matching trials cap out at
    `pc29_ratio`~=0.90). A rank-regression of `pc29_ratio` and
    `gene_var_mean_ratio` jointly on all 18 search-space knobs newly
    implicates `dropout_percentile` in the same tension (coefficient
    +0.49 on `pc29_ratio` vs. -0.41/-0.50 on `gene_var_mean`/
    `gene_mean_mean` ratio, a flipped sign matching the
    `lib_size_scale`/`outlier_scale`/`outlier_prob` pattern already known).
  - **`lib_size_scale`'s ceiling (`.20260814`'s widened-but-unexplored
    upper bound) is now confirmed real, not an artifact of the pruning
    guard**: loosening `max_lib_size_zero_frac` 0.05->0.10 barely moved
    anything -- max *completed* `lib_size_scale` is 1.36-1.78 in this
    round (guard 0.10) vs. 1.36-1.78 in `.20260814` (guard 0.05,
    essentially identical range), and trials pruned for
    `degenerate_lib_size_zero_frac` cluster at `lib_size_scale`
    mean=2.37 (this round) vs. a per-study mean of 2.0-2.6 in
    `.20260814` -- both rounds hit the same wall regardless of how lax
    the guard is. Pushing `lib_size_scale` further to close the
    `gene_var_mean` gap runs directly into cell/library dropout collapse,
    not an unexplored region of the search space.
  - Cross-trial correlation analysis (2137 completed trials, all 8
    studies, first with topology fully controlled) reconfirms and
    strengthens the coherency/canalization mechanics' internal
    mechanism-consistency findings from `.20260813`/`.20260814`:
    `coherency_bias` -> `frac_ambiguous_flipped` Spearman rho=+0.94 (was
    +0.86), `balancing_strength` -> `n_distinct_winners` rho=+0.90 (was
    +0.48), -> `mean_canalization_alignment_frac` rho=-0.92 (was -0.69),
    all p~=0. `canalization_strength` -> its own `mean_canalization_
    alignment_frac` diagnostic continues the positive trend flagged in
    `.20260814` (rho=+0.28, up from +0.18/+0.09 in `.20260814`/`.20260813`
    respectively) -- the mechanism's internal mechanics are behaving more
    sensibly as the surrounding config (`n_clusters=15`, fixed target
    depth, etc.) has evolved, not less. Effect on the actual PCA outcome
    remains real but modest and still second-order: `canalization_
    strength` -> `pc29_ratio` rho=+0.235, -> `pca_tail_participation_
    ratio` rho=-0.32 (both a bit stronger than `.20260814`'s
    |rho|<=0.18/0.29), but a joint rank-regression of `distance` on all 18
    search-space knobs shows `canalization_strength`'s own net partial
    coefficient has weakened to -0.02 (from `.20260814`'s -0.111) --
    `lib_size_scale`/`outlier_prob`/`outlier_scale`/`outlier_mean`/
    `hill_coeff` (|coef| 0.14-0.48) remain the dominant independent
    drivers of `distance`, this mechanism is still a second-order lever.
    `coherency_bias`'s `.20260814`-flagged small adverse net effect on
    `distance` (+0.076) is not replicated here (net coefficient -0.04,
    near zero) -- likely just noise given the small effect size either
    way, not a clear reversal.
  - **New search-space boundary-saturation signal**: `hill_coeff`
    (range `[1.0, 3.0]`) is a real, significant predictor of lower
    `distance` (marginal rho=-0.20, partial coefficient -0.14, both among
    the largest of any knob) and **27% of each study's top-5%-by-distance
    trials sit within 5% of the upper bound** -- the clearest per-knob
    saturation signal found to date (`decays`/`outlier_mean`/
    `outlier_scale` show a milder version of the same pattern, 12-16%).
    Not yet acted on (would need a new config bumping this bound, e.g. to
    4.0-5.0, and a fresh tuning round to test it) -- flagged for next
    round.
  - Seed-to-seed robustness, now with topology genuinely controlled:
    aggregate/pooled conclusions above (the tension, the mechanics, the
    `lib_size_scale` ceiling) rest on n>2000 trials and are trustworthy,
    but per-seed point estimates still vary substantially (`distance`
    0.027-0.060, `pc1_ratio` 0.68-1.20, `gene_var_mean_ratio` 0.15-0.25)
    and best-trial `coherency_bias`/`canalization_strength`/
    `balancing_strength`/`path_decay` values remain scattered across
    nearly their full ranges across the 8 seeds' best trials -- consistent
    with every prior round's finding that the objective doesn't strongly
    prefer a particular setting of these four knobs specifically.
  - No code bugs found this round. **User action required / recommended
    before the next tuning round**: (1) re-test `mr_rate_high` in true
    isolation (not bundled with a `max_lib_size_zero_frac` change) at full
    trial budget, since the pilot's near-doubling of `gene_mean_mean`/
    `gene_var_mean` ratio did not replicate here; (2) widen `hill_coeff`'s
    upper bound past 3.0 given the saturation signal above; (3) resume the
    4 studies that hadn't plateaued by their final 20% of trials (02, 03,
    04, 06 improved 12-18% late) via the existing resumable/unbounded
    workflow rather than starting fresh; (4) the PCA-vs-gene-magnitude
     tradeoff looks like a genuine Pareto frontier at this point (not a
     weight-balance or knob-confound artifact) -- worth an explicit
     decision on which matters more for the downstream VAE task rather than
     expecting a further config change to resolve both simultaneously.
- Investigated (user question, no data available yet -- this entry
  documents the mechanistic analysis + resulting code changes, not a
  tuning-study result) *why* the PCA-vs-gene-magnitude tradeoff above is a
  real mechanism rather than an artifact of TPE exploring correlated search
  dimensions, and made two targeted changes to address it:
  - Traced two concrete, code-level causes: (1) `generate_sergio_grn_from_
    reference`'s `canalization_strength` reweights each edge's K by up to
    2x (aligned) or down to 0.05x (disaligned), uncompensated -- since
    SERGIO's non-MR production rate is `rate = sum_i |K_i| * hill_i(...)`
    (each `hill_i` in [0, 1]), this directly shrinks a target's overall
    production-rate ceiling whenever it has any disaligned edges (which
    `balancing_strength`, needed for multiple distinct MR "winners" i.e.
    real PC2+ structure, mechanically increases the incidence of); SERGIO's
    `lib_size_effect` then renormalizes each *cell's* total to a fixed
    drawn target, forcing every other gene to compete for whatever budget
    this left behind -- a genuine, traceable path from "more canalization/
    balancing structure" to "lower gene_mean/gene_var/library size,
    higher dropout". (2) `compute_summary_stats`' PCA is computed on
    mean-centered but never per-gene-standardized log1p data (plain
    covariance, not correlation, matrix) -- so `pca_pc2_9_explained_
    variance_ratio` and `gene_var_mean` are two different aggregations
    (eigenstructure vs. arithmetic mean) of the *same* underlying per-gene
    variances, not independent quantities, even before considering (1).
  - **Fix for (1)**: each target's K-vector in `generate_sergio_grn_from_
    reference` is now rescaled, immediately after canalization's alignment-
    based reweighting, so `sum(|K_i|)` (the target's production-rate
    ceiling) matches what it would have been without canalization --
    unconditional (no new parameter), and provably a no-op at
    `canalization_strength=0.0` (weight is always exactly 1.0 then, so
    `scale == 1.0` exactly) -- so canalization now only reallocates *which*
    regulator dominates a target's production (the mechanism that should
    create between-cluster/PCA-relevant variance), without changing that
    target's own expression level on average. Verified against the real
    TRRUST reference GRN (`data/trrust_rawdata.human.tsv`, 200 genes,
    `seed=42`), comparing this session's code against an unmodified copy of
    the same function (`git show HEAD:synthetic_data.py`): (a) byte-
    identical output at `canalization_strength=0.0` between old and new
    code; (b) topology/regulator-lists/edge-signs/coop-states unaffected by
    `canalization_strength` in the new code (only K magnitudes change); (c)
    the *old* code's per-target `sum(|K_i|)` changed for 179/179 targets
    between `canalization_strength=0.0` and `0.7` (confirming the bug); (d)
    the *new* code's per-target `sum(|K_i|)` is preserved for 179/179
    targets to within float rounding (max relative error 2e-16) across the
    same comparison. `generate_sergio_grn_from_reference`'s `diagnostics`
    dict (winner_mr/vote_margin/n_aligned_edges/etc.) also confirmed
    unaffected (still populates sensibly with canalization/balancing/
    coherency all nonzero).
  - **Fix for (2)**: `compute_summary_stats` gained a `pca_standardized_*`
    stat family -- `pca_standardized_explained_variance_ratio` (PCA on each
    gene standardized to unit variance first, i.e. correlation-matrix PCA,
    per-gene variance floored at 1e-6; same `X_struct`/`gene_idx` subsample
    as the existing raw PCA) and `pca_standardized_tail_participation_
    ratio` (that family's `pca_tail_participation_ratio` counterpart) --
    computed *in addition to*, not replacing, the existing raw (covariance-
    matrix) `pca_explained_variance_ratio`/`pca_tail_participation_ratio`.
    `tune_synthetic_data.py`'s `_add_pca_pc2_9_key` was generalized (now
    takes `src_key`/`dst_key` params, default unchanged) and is called
    once per family at both call sites (`run()`/`run_trial()`), yielding a
    new `pca_standardized_pc2_9_explained_variance_ratio` alongside the
    existing `pca_pc2_9_explained_variance_ratio`. Verified with real
    (non-stub) torch in `/tmp/opencode/tuning_venv`: on a synthetic
    log1p-plausible matrix with real multi-module correlation structure,
    rescaling one gene's column by 50x leaves `pca_standardized_explained_
    variance_ratio` completely unchanged (max abs diff 0.0 across all 10
    components) while the raw ratio shifts by up to 0.75 (same data); on
    iid-normal (already-near-equal per-gene variance) data the two ratios
    coincide (max diff 0.0012); a single exactly-constant injected gene
    produces no NaN/inf (1e-6 floor); the subsampled-genes code path
    (`n_structure_genes < n_genes`) produces correctly-shaped arrays for
    both families; `_add_pca_pc2_9_key`'s generalization reproduces the
    exact same raw-family output as before (no regression) plus correct
    standardized-family slices, and is a no-op when the source array is
    too short, matching its original contract.
  - `synthetic_tuning_config.20260816.0{0-7}.json` (cloned from
    `.20260815.0{0-7}`, everything else -- `mr_rate_high=8.0`,
    `max_lib_size_zero_frac=0.10`, search space, `n_clusters=15` --
    unchanged): `weights.pca_pc2_9_explained_variance_ratio` `6.0` -> `0.0`
    (joining `pca_explained_variance_ratio`/`pca_tail_participation_ratio`,
    already `0.0`, at zero weight -- the raw PCA family is still computed
    for diagnostics/plotting, just no longer drives the objective, per
    user direction: "calculated in addition to the existing two... but
    replacing them in evaluating models"); added
    `weights.pca_standardized_tail_participation_ratio: 6.0` and
    `weights.pca_standardized_pc2_9_explained_variance_ratio: 6.0`
    (matching the raw family's old weight, so this round's only real
    change to the objective's balance is the metric-definition swap, not
    how much the objective cares about PCA structure overall) plus
    `weights.pca_standardized_explained_variance_ratio: 0.0` (mirroring why
    the raw full-vector version is unweighted -- PC1 fidelity itself isn't
    the target). `max_pca_pc1_ratio_factor`'s hard prune guard deliberately
    left on the *raw* `pca_explained_variance_ratio[0]` (per user
    direction) -- it targets a different failure mode (total-variance
    collapse onto one axis) than the structure metric being swapped, kept
    as an independent sanity check. `target_stats_path` points at a
    not-yet-existing `...v4.mean.pickle` (the `.v3` pickle predates the
    `pca_standardized_*` keys entirely) -- `run()` now warns at startup,
    analogous to its existing raw-PCA-family check, if a loaded
    `target_stats_path` lacks `pca_standardized_tail_participation_ratio`
    or has a mismatched `pca_standardized_explained_variance_ratio` depth.
  - **User action required before running**: (1) regenerate the target
    stats pickle (`...v4.mean.pickle`) via the existing recipe in
    `target_stats_path`'s own docstring against the real reference
    `.h5ad` -- no code change to the recipe itself was needed (it already
    stores whatever `compute_summary_stats()` returns generically), it
    just needs to be rerun with this session's updated code (needs torch/
    SERGIO/anndata + the reference data, all unavailable in this sandbox);
    (2) launch the 8 `.20260816.0{0-7}` studies (unbounded `n_trials`/
    `timeout` as usual, `PYTHONHASHSEED` set as usual) as a **pilot**:
    interrupt each via SIGINT/SIGTERM after >=80 trials rather than running
    to convergence, then compare against the `.20260815` studies (the
    combined change's "before" baseline, since it's simplest to compare
    against those already-completed runs rather than adding a second new
    toggle solely for an isolated A/B) on `gene_mean_mean`/`gene_var_mean`
    ratio-to-target, `log_lib_size_std`, `lib_size_zero_frac`/`zero_frac`,
    and `pca_standardized_pc2_9_explained_variance_ratio`/`pca_standardized_
    tail_participation_ratio` match, before committing to a longer sweep.
- The `synthetic_tuning_20260816.0{0-7}` pilot studies (8 seeds,
  `synthetic_tuning_config.20260816.0{0-7}.json`, 87-112 completed trials
  each -- comfortably past the ">=80 trials" pilot threshold) were run and
  analyzed (`.db`s + `checkpoints/best.pt` + `grn_diags/`, including
  per-trial `candidate_stats.json`, copied in from the user's machine).
  Sanity checks all clean: `n_mrs=84` identical across all 8 studies (the
  `PYTHONHASHSEED` fix continues to hold -- topology remains controlled);
  every `best.pt`'s `trial_number`/`distance` matches its `.db`'s true
  argmin exactly; recomputing `compute_stats_distance` from each `best.pt`'s
  stored `candidate_stats` against the regenerated v4 target pickle
  reproduces the stored `distance` exactly; the v4 pickle itself has the
  expected `pca_standardized_*` keys at the correct depth (20). Findings:
  - **Headline result: the core hypothesis behind introducing the
    standardized PCA family is confirmed, cleanly, at the pooled-trial
    level.** Pooling all 918 completed trials across the 8 studies and
    correlating each candidate's ratio-to-target against
    `gene_var_mean_ratio`: the *raw* `pca_pc2_9_explained_variance_ratio`
    remains strongly anti-correlated (Spearman rho=-0.797, p~1e-203, closely
    matching the `.20260815` entry's pooled rho=-0.61/-0.50 on the
    predecessor metrics) -- but the new *standardized*
    `pca_standardized_pc2_9_explained_variance_ratio` shows essentially
    **zero** correlation with `gene_var_mean_ratio` (rho=+0.062, p=0.06, not
    significant). Trials jointly achieving `pc29_ratio>=0.9` AND
    `gene_var_mean_ratio>=0.5` simultaneously went from 1/2137 in the
    `.20260815` baseline (using the raw metric) to 14/918 using the
    standardized metric this round (vs. only 2/918 for the raw metric
    computed on this same `.20260816` data) -- a real, substantial widening
    of the previously near-empty joint-optimum region. At the population
    level (pooled per-trial medians, not just the 8 best trials),
    `gene_mean_mean`/`gene_var_mean` ratio-to-target improved from
    0.493/0.216 (`.20260815`) to 0.736/0.304 (`.20260816`).
  - **New wrinkle, not previously flagged: the standardized PCA family's
    other half doesn't share this decoupling, and actively fights the half
    that does.** `pca_standardized_tail_participation_ratio` -- weighted
    equally (6.0) alongside `pca_standardized_pc2_9_explained_variance_
    ratio` in this round's objective -- shows rho=-0.48 against
    `gene_var_mean_ratio`, *more* anti-correlated than its raw predecessor
    (`pca_tail_participation_ratio`, rho=-0.23 on this same data), and this
    persists (partial rho=-0.57) after statistically controlling for
    `pca_standardized_pc2_9_explained_variance_ratio`. Worse: the two new
    terms are themselves anti-correlated with **each other** (rho=-0.63,
    p~1e-103) despite being intended as complementary "shape" (pc2_9) vs.
    "evenness" (tail participation ratio) views of the same PCA-standardized
    tail. This is a plausible, previously-undocumented explanation for why
    the **best-trial-level** picture across the 8 studies was a wash rather
    than a clean win: recomputed weighted-term breakdowns show
    `pca_standardized_pc2_9_explained_variance_ratio` as the #1 or #2
    largest weighted `distance` contributor in 5/8 studies' best trials (as
    intended, given its 6.0 weight), `distance` itself was flat-to-worse in
    7/8 studies vs. `.20260815`'s (non-comparable, since the objective
    changed, but directionally suggestive) 0.027-0.060 range, secondary
    stats (`gene_var_std`, `gene_corr_abs_std`/`p90`, `log_lib_size_std`)
    were mildly worse on pooled median in `.20260816` vs. `.20260815` even
    as `gene_mean_mean`/`gene_var_mean` improved, and raw PCA fidelity
    (no longer weighted at all) dropped substantially at the best trial
    (raw PC1 ratio-to-target averaged ~0.67 across the 8 best trials, down
    from `.20260815`'s ~0.92) -- each ~100-trial-per-seed budget is
    plausibly being pulled in two directions by the two new, mutually
    anti-correlated, equally-weighted terms, rather than cleanly exploiting
    the low-tension joint-optimum region the pooled analysis shows now
    exists.
  - `lib_size_scale` is now the single strongest predictor of `distance`
    across all 786 pooled completed trials with matched sqlite params
    (Spearman rho=+0.202, p=1.1e-8) -- notably a **sign flip** from its
    role in every prior round (previously a lever that *helped*
    `gene_var_mean`/`log_lib_size_std` at a confirmed cost to PCA
    structure; now higher `lib_size_scale` correlates with *worse*
    `distance` overall). Plausibly a consequence of the objective's shift
    from raw to standardized PCA terms changing which knobs the two
    (now-competing) PCA terms respond to -- not yet root-caused, flagged for
    future investigation if it persists in `.20260817`.
  - The `.20260815` entry's `hill_coeff` saturation concern (27% of top-5%
    trials within 5% of the upper bound `3.0`, recommended widening next
    round but not yet acted on) does not reproduce this round: 0% of
    `.20260816`'s top-5%-by-distance trials sit near the bound -- plausibly
    because the objective's shift to standardized PCA terms moved the
    optimum to a different region of the search space where `hill_coeff`
    saturation isn't beneficial the way it was under the old objective.
    Not itself evidence the bound should be left alone forever, just no
    longer an active concern under this round's objective.
  - Per user direction (given the tail-participation-ratio finding above),
    resulting change: new `synthetic_tuning_config.20260817.0{0-7}.json`
    (cloned from `.20260816.0{0-7}`, single change):
    `weights.pca_standardized_tail_participation_ratio` `6.0` -> `0.0`
    (joining the raw family, already `0.0`, at zero weight -- still
    computed for diagnostics/plotting via `_add_pca_pc2_9_key`'s existing
    generic mechanism, just no longer drives the objective).
    `weights.pca_standardized_pc2_9_explained_variance_ratio` deliberately
    left at `6.0`, **not** bumped to compensate for the removed term's
    weight (per user direction, isolating this one variable change rather
    than also testing a total-PCA-weight-preserving alternative). Everything
    else (target `v4` pickle, `mr_rate_high=8.0`, `max_lib_size_zero_frac=
    0.10`, search space, `n_clusters=15`) unchanged from `.20260816.*`. No
    `synthetic_data.py`/`tune_synthetic_data.py` code change was needed --
    pure config change, weight lookups are already fully generic.
  - **User action required**: run the 8 `.20260817.0{0-7}` processes (same
    unbounded/graceful-stop workflow, shared `PYTHONHASHSEED`), aiming for a
    comparable ~90-110 completed trials per seed to the `.20260816` round
    for an apples-to-apples comparison, then copy back `.db`s/`checkpoints/
    best.pt`/`grn_diags/` (including per-trial `candidate_stats.json`) for
    analysis -- specifically to check whether removing the competing
    tail-participation-ratio term lets TPE actually reach the low-tension
    joint-optimum region the `.20260816` pooled data shows now exists,
    rather than the wash seen at the best-trial level this round.


