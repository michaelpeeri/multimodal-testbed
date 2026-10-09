# MR-state biological-signal experiment (20260910)

This is a compact, staged comparison of DE-derived, IID, and constant MR
states on one archived GRN. It uses a small matched-GRN DE search (two seeds),
12 paired SERGIO replicates in the harness, and 80-trial Optuna studies. The
first harness stage runs before Optuna so its observed
`mr_state_expression_distance_correlation` can replace the provisional
minimum in the Optuna configs.

## Prerequisites and one-time checks

Run from the repository root in the Python environment containing the project
dependencies and classic `SERGIO` package. Confirm the archived GRN and its
fingerprint:

```bash
sha256sum synthetic_tuning_20260907.buffered.50/grn_diags/trial68.grn.csv
```

It should print
`f0f653cd1cd7db0499a63c6b5f4adc8cb2901a5baac69fb499522cfb0e7dc934`.
The target stats v6 pickle, reference GRN, DE input GRN, and
`mr_state_constant_iid_mean.npy` must also be present. Keep
`PYTHONHASHSEED=0` for every process that generates a GRN; it must be set
before Python starts.

Create the SQLite/output parents before launching:

```bash
mkdir -p mr_state_de_20260910 \
  synthetic_tuning_20260910.mr_de/{checkpoints,grn_diags} \
  synthetic_tuning_20260910.mr_iid/{checkpoints,grn_diags} \
  synthetic_tuning_20260910.mr_constant/{checkpoints,grn_diags}
```

## Step 1 — optimize two DE states against the exact GRN

The locked-GRN config uses the same intrinsic surrogate objective and GRN
parameters as the matched-state setup, but cuts the optimizer to 100 members
and 50 generations to keep this experiment bounded. Run two independent
initialization seeds:

```bash
PYTHONHASHSEED=0 python3 run_matched_mr_state_optimization.py \
  --config mr_state_match_config.20260910_locked_grn.json \
  --output mr_state_de_20260910.seed0.pickle \
  --seed 0 --population 100 --generations 50

PYTHONHASHSEED=0 python3 run_matched_mr_state_optimization.py \
  --config mr_state_match_config.20260910_locked_grn.json \
  --output mr_state_de_20260910.seed1.pickle \
  --seed 1 --population 100 --generations 50
```

Choose the artifact with the **higher `final_robust_fitness`** (the final
surrogate score aggregated across all surrogate replicates), and copy it to
the canonical path used by the harness and DE Optuna config:

```bash
python3 - <<'PY'
import pickle
import shutil

paths = [
    "mr_state_de_20260910.seed0.pickle",
    "mr_state_de_20260910.seed1.pickle",
]
results = []
for path in paths:
    with open(path, "rb") as handle:
        results.append((pickle.load(handle), path))
for result, path in results:
    print(path, "best_fitness=", result["best_fitness"],
          "final_robust_fitness=", result["final_robust_fitness"],
          "grn_sha256=", result["grn"][0]["sha256"])
best, best_path = max(results, key=lambda item: item[0]["final_robust_fitness"])
assert best["grn"][0]["sha256"] == (
    "f0f653cd1cd7db0499a63c6b5f4adc8cb2901a5baac69fb499522cfb0e7dc934"
)
shutil.copyfile(best_path, "mr_state_de_20260910.best_surrogate.pickle")
print("selected", best_path)
PY
```

Do not compare DE `best_fitness` as though it were a full-SERGIO outcome; it is
only used here to choose the stronger of the two candidate artifacts before
held-out SERGIO validation.

## Step 2 — paired stage/noise/dropout harness grid

The grid compares the DE state, IID states, and constant-MR negative control
on the same GRN and paired replicate seeds. It includes clean/noise-only
conditions, dropout percentiles 40–70 at both noise levels, and restores
outlier/UMI stages at d40 and d70. The harness now records
`mr_state_expression_distance_correlation`; paired summaries also include a
deterministic 95% bootstrap interval and counts of positive/negative paired
deltas.

```bash
PYTHONHASHSEED=0 python3 run_mr_state_harness_grid.py \
  --config mr_state_harness_config.20260910_signal_grid.json
```

Primary evidence is in each `mr_state_comparison.20260910_signal_*.summary.json`
under `comparison.paired_deltas_vs_baseline.de_locked_best.scenarios.sergio`.
The deltas are **DE minus IID**: positive values favor DE for
`mr_state_expression_distance_correlation`, `within_minus_across`, and
stability; negative values favor DE for target `distance`. Check the paired
bootstrap interval and directional counts alongside the mean. The constant
control should have zero MR-state/expression distance correlation by
construction. Use the noise-only vs. clean and reduced vs. full conditions to
locate the loss from noise, dropout, and the combined outlier/UMI stages.

### Set the provisional Optuna correlation minimum from phase 3

Before launching Optuna, inspect the 12-replicate correlation distributions
for DE and IID, especially the `reduced_n014_d40`, `d50`, and `d60` conditions.
Update `biological_signal.minimums` and the matching `scales` entry for
`biological_mr_state_expression_distance_correlation` **identically in all
three Optuna configs**. Use a value supported by the DE distribution at a
condition that also beats IID; the checked-in value `0.15` is provisional.
Keep the control study's same threshold: its expected fixed deficit adds a
constant component to its penalty, so raw `objective_distance` is not
comparable across MR-state methods. Compare `target_distance` and the recorded
biological metrics instead.

## Step 2b — rank-resolved follow-up (does DE carry high-dimensional programs?)

The step 2 grid shows DE improves state-to-expression distance correlation over
IID, but not that its programs are high-dimensional or state-linked rather than
noise-induced (DE's MR-state/centroid participation ratios are not higher than
IID's, and the constant control can score well on label holdout accuracy and
centroid rank). Step 2b adds, in `mr_state_harness.py`:

- Metrics (SERGIO scenario): `mr_state_expression_distance_correlation_null_mean/
  _null_q95/_excess` (cluster/state shuffle null); `centroid_n_reproducible_dims`,
  `centroid_reproducible_participation_ratio`,
  `centroid_reproducible_variance_fraction` (between-cluster structure that
  replicates across disjoint halves of each cluster's cells, with a
  label-permutation null, so noise-induced dimensions do not count);
  `state_expression_cka`, `_cka_null_mean`, `_cka_excess`. Surrogate-scenario
  values of the split-half metrics are NaN (one cell per cluster).
- Arms: `lowrank` candidate method (`candidate_params.rank`, exactly rank-k with
  IID-matched per-MR range) and `column_shuffle: true` for fixed-state arms
  (permutes MR columns per replicate, destroying state-to-GRN alignment only).

Run on the deployment environment (3 cells x 6 arms x 8 replicates; baseline
`iid_random`; same seeds as the v2 grid, so the original three arms should
reproduce v2 values exactly):

```bash
PYTHONHASHSEED=0 python3 run_mr_state_harness_grid.py \
  --config mr_state_harness_config.20260911_rank_resolved_grid.json
```

**Caveat found in the real-SERGIO smoke test (1 replicate, reduced_n014_d40):**
the constant-MR control scored `centroid_n_reproducible_dims`=13 and
`centroid_reproducible_variance_fraction`=0.73 (IID 10/0.85, DE 8/0.85). So the
split-half "reproducible" between-cluster structure is *not* purely
state-driven in real SERGIO: clusters differ reproducibly across halves even
with identical MR input (likely a per-cluster stochastic-realization/technical
effect shared by cells of a cluster; not yet verified). Read those three
metrics only as excess over the constant arm, not as absolute state-linked
rank. The state-linked metrics behave sensibly: CKA excess was 0.00 constant,
0.09 IID, 0.17 DE, 0.10 DE-column-shuffled, 0.74 rank3, 0.86 rank1; distance
correlation was 0.40 IID, 0.71 DE, 0.50 shuffled. Original three arms
reproduced the stored v2 replicate-0 values exactly.

Interpretation: constant gives the noise floor for the reproducible-rank
metrics (and ~0 for correlation/CKA); `lowrank_rank1/3` should cap near 1/3
reproducible dimensions in state-linked terms; IID/DE should exceed them if their programs are
genuinely higher-dimensional. If `de_v2_column_shuffled` loses DE's advantage in
correlation/CKA excess, DE's gain is GRN-specific alignment. Verified on
synthetic data only (rank recovery for rank 1/3/6/IID, zero for constant,
~0 reproducible dimensions for pure noise, CKA excess ~0 for unrelated state);
the metric is conservative and linear, and CKA must be read as excess over its
own null.

## Step 3 — three matched-budget Optuna studies

The configs hold the GRN settings, parameter search, trial count, and sampler
seed fixed. They differ only in MR-state source and output paths. The
constant-MR study uses `mr_state_constant_iid_mean.npy`, the same
cluster-invariant per-MR mean vector used by the harness control. Each runs 80
trials with dropout-percentile range 40–99 and noise range 0.05–0.2867. The
search is limited to the five parameters active with outlier and UMI stages
disabled; it does not spend trials varying inactive outlier knobs. The
Optuna objective records the biological diagnostics; the minimums add a soft
penalty, not a hard constraint. The constant arm is a negative control, not a
candidate expected to pass the MR-state alignment criterion.

```bash
PYTHONHASHSEED=0 python3 tune_synthetic_data.py \
  --config synthetic_tuning_config.20260910.mr_de.json

PYTHONHASHSEED=0 python3 tune_synthetic_data.py \
  --config synthetic_tuning_config.20260910.mr_iid.json

PYTHONHASHSEED=0 python3 tune_synthetic_data.py \
  --config synthetic_tuning_config.20260910.mr_constant.json
```

The `mr_de` config requires the selected artifact from step 1 and enforces its
GRN SHA256. All three configs use the same Optuna sampler seed (`60`) to align
initial suggestions. Compare `target_distance`, biological metrics, and the
parameter values; do not rank the arms by combined `objective_distance`
because the constant state has an expected, state-specific correlation
penalty.

## Step 4 — crossover validation of Optuna-selected parameters

Use `mr_state_harness_config.20260910_best_replay.json` as a template. For each
of the three Optuna outputs (`mr_de`, `mr_iid`, `mr_constant`):

1. Copy that study's `best_params` values into the template's `base_overrides`
   for `decays`, `cluster_conc`, `noise_params`, `dropout_shape`, and
   `dropout_percentile`. The outlier parameters are fixed and those stages
   remain disabled in this crossover.
2. Keep the same selected DE artifact, GRN, technical-stage switches, and all
   three harness arms unchanged.
3. Give the run a distinct output name and evaluate the other two arms using
   that same technical parameter set.

Run each edited template:

```bash
PYTHONHASHSEED=0 python3 mr_state_harness.py \
  --config mr_state_harness_config.20260910_best_replay.json \
  --output mr_state_comparison.20260910_best_replay_de_params.pickle
```

Repeat with `iid_params` and `constant_params` in the output name. This
3-parameter-sets × 3-state-methods crossover separates the effect of MR-state
choice from the different parameter settings selected by independent Optuna
studies.

## Decision criteria and resulting protocol

- Demonstrate a DE advantage where the paired DE-minus-IID confidence interval
  for state/expression correlation or module contrast is above zero, with
  corresponding non-collapsed split-half stability; report the constant-MR
  result as the negative control.
- Locate the transition as noise/dropout increases by plotting those metrics
  across the harness grid. Compare noise-only to clean, dropout-on to noise-only,
  and full to reduced conditions to identify the stage effects.
- Keep `target_distance` descriptive for this experiment. The objective is to
  establish state-linked, reproducible programs, not to claim reference-level
  count statistics from this screening run.
- For future MR-state or SERGIO changes, preserve this order: matched-GRN DE
  candidate generation → paired harness validation on held-out simulation
  seeds → tune technical parameters with DE/IID/constant controls → crossover
  harness validation of selected parameter sets.
