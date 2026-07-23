Here's a detailed synthesis organized around your four questions. This is entirely a plan/analysis — no code changes.

## 1. Metrics: what's informative vs. misleading

### Primary task metrics

| Metric | What it measures | Notes |
|---|---|---|
| **Per-cell masked error** (`per_cell_masked_error`) on type-2 positions | Actual imputation quality per cell | The ground-truth task metric. Must be evaluated over multiple mask draws (high variance per draw). |
| **Imputation correlation vs. mask fraction** curve | Overall imputation quality across a difficulty sweep | Already implemented (`vae-test.py:246-265`). Tests graceful degradation as less data is observed. A single scalar (e.g. AUC of this curve, or correlation at a fixed hard fraction) is needed for automated comparison. |
| **Between-class variance of per-cell error** (`class_grouped_stats`) | Fairness: does the model treat all cell types equally? | Requires external class labels. We showed VampPrior doesn't improve this despite class specialization — the metric itself is informative, the prior just doesn't affect it. |

### Latent-space / prior quality metrics

| Metric | What it measures | Pitfall discovered |
|---|---|---|
| **`effective_K_population`** | Population-level mixture usage — "is the VampPrior actually used diversely in aggregate?" | **Correct metric.** |
| **`effective_K`** (per-sample averaged) | Per-cell sharpness of mixture assignment | **Misleading.** Reads ~1.0 even in the theoretical best case (confirmed via direct calculation). Wasted multiple training runs chasing a "collapse" signal that was actually normal high-dimensional sharpening. **Should be logged for reference but never used as a collapse diagnostic.** |
| **`vamp_usage_by_class`** | Is diversity class-aligned or class-independent? | Critical for interpreting *what* the VampPrior is doing. We showed: class-aligned specialization is real and strong, but doesn't translate to better per-class reconstruction. |
| **Per-class KL cost** (`_vamp_kl_per_sample` / `standard_gaussian_kl_per_sample` + `class_grouped_stats`) | Rate-allocation fairness — does VampPrior distribute KL cost more evenly across classes? | We showed: no — VampPrior actually has *higher* between-class KL variance (CV 7.5% vs 5.4%), even after scale normalization. |
| **Loss component magnitudes** (`beta*kl_loss` vs `recon_loss`) | Is the KL term strong enough to influence the latent space at all? | Essential sanity check. If the ratio is negligible, prior flexibility is irrelevant and the comparison is uninformative. We found ~7.9% at convergence with `beta=0.05` — meaningful but not dominant. |

### DEQ-specific metrics

| Metric | What it measures |
|---|---|
| **DEQ iterations / residual** (`collect_deq_diagnostics`) | Convergence quality. Per-sample residual can be grouped by class to detect uneven convergence. |
| **DEQ z_star range/saturation** | Whether the `tanh`-bounded fixed point is saturating (all values near ±coeff). |

### Anti-collapse regularizer diagnostics

| Metric | What it tells you |
|---|---|
| **`repulsion_margin` vs `pu_mu_pairwise_median`** | Whether pseudo-inputs are geometrically spread (gap widening = repulsion winning). |
| **`coverage_mean_nearest_dist`** | Whether pseudo-inputs are near the data manifold (should shrink). |
| **`repulsion_typical_scale`** | The real batch's own latent spread — watch for this shrinking over training (would indicate the margin is self-referentially collapsing). |

## 2. How to tune models at different "levels" for fair comparison

### Controlled variables (must match)

- **`latent_dim`**: All models must use the same value. The existing `model_factories` already enforce this (all at 32), but the raw class constructors have different defaults (8 vs 32) — a trap we fell into. The benchmark should use the factories exclusively.
- **`encoder_dims`, `decoder_dims`, `dropout`**: Already matched across factories.
- **`beta`, `beta_schedule`**: Must be sized to `max_epochs` (we found a bug: 200-step schedule indexed by only 50 epochs, so beta peaked at 0.007 instead of intended 0.05). The benchmark should explicitly verify `beta_sched[max_epochs-1]` matches `beta_final`.
- **`mask_fraction`, `gamma`, `min_free_bits`**: Same across all models.
- **Training budget**: Same `max_epochs` and `batch_size`. Ideally train to convergence (stable loss for N epochs), or confirm all models have plateaued.
- **Data**: Same train/test split, same random seed for data generation.

### Per-model-family tuning that's allowed to differ

- **VampPrior models**: `lambda_entropy`, `lambda_vamp_repulsion`, `lambda_vamp_coverage` are VampPrior-specific regularizers with no counterpart in standard-prior models. These need to be tuned to achieve healthy `effective_K_population` (e.g. >50% of `n_pseudo`) before the comparison is meaningful — otherwise you're comparing a broken VampPrior against a working standard prior. The benchmark should report `effective_K_population` alongside every VampPrior result as a "health check" gate.
- **DEQ models**: `coeff`, `max_iter`, `tol` — these control the fixed-point solver, not the loss landscape, so they're architectural choices not training hyperparameters. The benchmark should report DEQ residual as a convergence gate (analogous to `effective_K_population` for VampPrior).
- **`pseudo_init_samples`**: Whether pseudo-inputs are initialized from real data or randomly. We tested `pseudo_init_samples` and it didn't help much, but the benchmark should standardize this (recommend: always initialize from data, since it's cheap and removes one source of variance).

### Convergence verification checklist (per model, per run)

1. Training loss plateaued (e.g. <1% change over final 10 epochs)
2. `beta*kl_loss / recon_loss` ratio reported (sanity: >5%, otherwise the prior comparison is uninformative)
3. VampPrior: `effective_K_population` > threshold (e.g. > 0.5 * n_pseudo)
4. DEQ: mean residual < `tol`, no cells with divergent fixed points

## 3. Procedure: showing complex models' value on harder data

### The difficulty axes available in this codebase

| Axis | Easy end | Hard end | What it stresses |
|---|---|---|---|
| **`mask_fraction`** | 0.01 | 0.3+ | More missing → harder imputation, more reliance on learned prior/structure |
| **`missing_rate`** (type-1, in data generation) | 0.02 | 0.15+ | Baseline missingness before evaluation masking |
| **`n_cells` / sample efficiency** | 2000 | 200–500 | p>>N regime; complex models need more data to train but may have better inductive bias |
| **`n_types` / `n_clusters`** | 1 (unimodal) | 8+ (multimodal) | Prior flexibility should matter more with more distinct modes |
| **`concentration`** (`random_points_on_hypersphere`) | 0 (uniform) | high (tight clusters) | Well-separated clusters should favor class-specialized priors |
| **Missingness pattern** | MCAR (current) | Structured/gene-dropout | DEQ's iterative refinement might handle structured missingness better (noted in `deq_dev_plan.md`) |

### Proposed benchmark structure

**Phase 1: Equivalence on easy data (sanity check)**
- Problem: `easy` config (20 genes, 1 type, 500 cells, 2% missing)
- Expectation: all models (including MeansModel baseline) perform similarly. Any model that's *worse* here has a bug or tuning problem.
- Metric: imputation correlation at `mask_fraction=0.05`, averaged over 10+ seeds.
- Gate: complex models must match or exceed simple models. If they don't, debug before proceeding.

**Phase 2: Single-axis difficulty sweeps (isolate each architectural contribution)**

Each sweep holds all other axes fixed and varies one:

a. **Mask fraction sweep** (the existing correlation curve, but multi-model):
   - Fix everything at `standard4` config. Sweep `mask_fraction` from 0.01 to 0.5.
   - Plot all models' imputation correlation on the same axes.
   - Expected: curves diverge at high mask fractions (more missing data → more reliance on prior/structure → complex models should degrade more gracefully).

b. **Sample efficiency sweep**:
   - Fix `n_genes=200`, vary `n_cells` from 200 to 2000.
   - Train each model on the smaller dataset, evaluate on a held-out test set drawn from the same distribution.
   - Expected: DEQ models (strong inductive bias) should maintain performance at lower sample counts; standard MLP encoders should degrade earlier.

c. **Modality sweep** (class count / concentration):
   - Use `random_points_on_hypersphere` with `n_clusters` from 1 to 8 and `concentration` from 0 (uniform) to high (tight clusters).
   - Expected: VampPrior should show its advantage when the latent distribution is genuinely multimodal with well-separated modes. With `concentration=0` (uniform/unimodal), VampPrior should offer no benefit (the N(0,I) prior is already fine).

d. **Missingness pattern sweep** (future):
   - Replace MCAR with structured missingness (e.g. gene-dropout correlated with expression level, or block-dropout where entire gene modules go missing).
   - Expected: DEQ's iterative refinement should handle structured missingness better than a single MLP pass.

**Phase 3: Full factorial (interaction effects)**

Cross the most informative axes from Phase 2 (e.g. mask_fraction × n_clusters) to detect interactions:
- Does VampPrior's advantage at high cluster count become larger at high mask fraction?
- Does DEQ's advantage at low sample count interact with modality?

### Statistical design

- **Multiple seeds per condition**: ≥10 training runs per model × condition cell (30 for noisy metrics like VampPrior's between-class variance, per our findings).
- **Paired comparisons**: All models trained on the same data split per seed (same `random_points_on_hypersphere` draw, same train/test split). Pair by seed for statistical tests.
- **Report**: Mean ± std across seeds, plus paired t-test or Wilcoxon signed-rank p-values for model differences at each condition.
- **Primary summary metric**: AUC of the imputation-correlation-vs-mask-fraction curve (single scalar, captures performance across the difficulty gradient).

### Key prediction from this investigation

Based on everything established this session:
- VampPrior's class-aligned specialization is real but acts on **rate allocation** (KL cost distribution), not **reconstruction fidelity**. Its value should emerge most clearly in latent-space quality metrics (manifold preservation, mode separation) rather than raw imputation error — especially under the `requirements.md` framing where "the main objective is to infer the latent... while preserving the structure... and avoiding mode collapse."
- This suggests the benchmark should evaluate latent-space quality directly (e.g. latent-space correlation with ground-truth `block_state`, or k-NN purity of cluster labels in latent space) alongside imputation error — those may be the metrics where VampPrior's advantage actually shows up.

----

Check whether the model actually recovers block structure: correlate the learned mu (latent embedding) against the true block_state — if correlation is already high, the latent representation is doing its job and the remaining error is truly irreducible observation noise.


Low data regime - with how few samples can the model learn the structure?

