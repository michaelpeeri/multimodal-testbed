"""
Sample-efficiency evaluation sweep across tuned model configurations.

This is the "Sample efficiency sweep" from benchmark_plan.md (Phase 2b),
generalized to replay each model family's own *tuned* hyperparameters
(from tuning.py's tuned_configs.json) rather than one hand-picked config
shared across all models.

Usage
-----
As a script:

    python sample_efficiency.py --config sample_efficiency_config.json

From a notebook:

    import sample_efficiency
    rows, summary = sample_efficiency.run("sample_efficiency_config.json")

What this does
---------------
For every model listed in the config's "models" (default: every model
present in tuned_configs.json, i.e. every model family you've already
tuned), and for a decreasing sequence of training-set sizes ("train_sizes"),
this:

  1. Draws a random subset of that size from the full training set (nested
     across sizes within the same repeat -- see "Paired subsets" below).
  2. Rebuilds a *fresh* model instance using that model's own tuned
     architecture (tuning.resolve_model_config), so every model is
     evaluated at its own best-found hyperparameters, not a shared default.
  3. Trains it from scratch on the subset for "max_epochs" epochs.
  4. Evaluates it on the *full, fixed* held-out test set (never subsetted):
       - held-out loss (epoch_vae's / epoch_transformer's total loss --
         diagnostic only, see "Cross-model metric comparability" below),
       - imputation MSE and correlation on type-2-masked positions at a
         fixed "eval_mask_fraction", averaged over "n_eval_mask_draws"
         independent mask draws (a single draw is noisy -- see
         benchmark_plan.md's "Metrics" section), and
       - health-check diagnostics (effective_K_population for
         VampPrior-family models, DEQ residual for DEQ-family models --
         None when not applicable).
  5. Repeats steps 1-4 "n_repeats" times with different random subsets, so
     the write-up can report mean +/- std instead of a single noisy run.

`models.evaluate_imputation()` (which wraps `models.impute()`, itself
uniform across VAEBase subclasses / MeansModel / ImputationTransformer) is
what lets this script sweep multiple model families through one loop.

Cross-model metric comparability
-----------------------------------
`test_loss` is **not comparable across model families** and should be
treated as a diagnostic/health-check signal only (per benchmark_plan.md's
convergence checklist), not as a cross-model quality metric:
ImputationTransformer's loss is categorical cross-entropy (a different
unit entirely from the VAE family's Gaussian MSE + KL), and each VAE
family model's `beta`/`gamma`/regularizer weights are independently tuned
by tuning.py, so even within the VAE family the *same* total_loss number
reflects a different weighted mixture of quantities per model. Use
`imputation_mse_mean` or `imputation_corr_mean` (both from the shared
`models.evaluate_imputation()`, the same function tuning.py's Optuna
objective now uses -- see tuning.py's module docstring) for any
cross-model comparison instead. `recon_loss_type2` (the VAE family's raw,
un-weighted MSE on held-out positions, `None` for ImputationTransformer/
MeansModel) is also reported for within-VAE-family diagnostics.

Paired subsets
---------------
Per benchmark_plan.md's "Statistical design": for a given repeat index,
the *same* random permutation of the training set is used regardless of
model or train_size (only truncated to the first `size` indices) -- so
smaller subsets are nested inside larger ones, and every model in the same
repeat trains on exactly the same data for a given size. This removes
subset-choice as a confound when comparing models, and makes paired
statistical tests (e.g. across repeats) valid.

Training budget: gradient steps, not epochs
---------------------------------------------
A fixed `max_epochs` across every `train_size` is *not* a fixed training
budget: with `batch_size` capped at `min(batch_size, size)` (see
`train_and_eval_one`'s `loader_train`), smaller subsets take fewer
optimizer steps per epoch, so the same `max_epochs` gives smaller
`train_size`s dramatically less training (e.g. `batch_size=128`,
`max_epochs=50`: `size=200` -> 2 steps/epoch -> 100 total steps, vs.
`size=2000` -> 16 steps/epoch -> 800 total steps). Any imputation-quality
degradation measured at small `train_size` would then be confounded with
"far fewer gradient updates," not just "less data."

To remove this confound, this script derives a single **step budget**
(`steps_budget`) and picks a *per-size* epoch count that reproduces that
same total step count at every `train_size`:

    steps_budget      = max_epochs * steps_per_epoch(max(train_sizes))
    epochs_for_size    = round(steps_budget / steps_per_epoch(size))

so `max_epochs` still means exactly what it always has *at the largest
`train_size` in the sweep* (backward compatible), while smaller sizes get
proportionally more epochs to match its total step count. Batch size is
still allowed to shrink with `size` (small subsets train full-batch or
near-full-batch, which is expected/fine) -- only the *step count* is
equalized, since that's the confound that otherwise dominates. The actual
`epochs_run`/`steps_run` used for each row are printed at startup and
recorded in the output CSV. Set `steps_budget` explicitly in the config to
bypass this derivation and specify the total step count directly.

mask_fraction / batch_size caveat, and tuning-provenance validation
----------------------------------------------------------------------
tuned_configs.json's best_params never includes `mask_fraction` (tuning.py
treats it as a fixed problem-difficulty knob, not something to tune -- see
tuning.py's module docstring). For the training/eval budget replayed here
to match what the model was *tuned* for, you must supply the same
mask_fraction value(s) that were used in the tuning run, via this script's
own top-level "mask_fraction" config key (float, applied to every model) or
a per-model dict ({"GeneExpressionVAE": 0.1, "ImputationTransformer": 0.2,
...}) if different model families were tuned with different mask fractions.
Similarly, `batch_size` here should match tuning.py's `batch_size`: a
mismatch silently changes the number of optimizer steps/epoch relative to
what tuning saw (see "Training budget" above), which previously caused a
model retrained here at the largest train_size to train for a different
total step count -- and therefore reach a different loss -- than tuning.py
did for the exact same hyperparameters.

tuning.py's run() now writes both of these (plus max_epochs) into a
reserved "_meta" key in tuned_configs.json (see tuning.py's module
docstring). run() here reads it back and prints a warning (not a hard
error) if this sweep's own `batch_size`/`mask_fraction` don't match --
check that warning if your results look surprising. tuned_configs.json
files from before this check existed simply have no "_meta" key, in which
case the check is skipped.

Problem file schema
--------------------
Same as tuning.py -- a dict loaded via torch.load(config["problem_fn"]) with
(at least) 'n_genes', 'n_cells', 'gene_reads_train', 'gene_reads_test'.

JSON config schema
-------------------
    problem_fn         : str, required
    tuned_configs_fn    : str, default "tuned_configs.json" -- summary written
                          by tuning.run() ({model_name: {"best_value":...,
                          "best_params":...}, ..., "_meta": {"batch_size":...,
                          "mask_fraction":..., "max_epochs":...}})
    models              : list[str], default: every key in tuned_configs_fn.
                          "MeansModel" may be included even though it's never
                          tuned (no hyperparameters) -- it's fit directly from
                          each subset's per-gene mean, as a sample-efficiency
                          floor/baseline.
    train_sizes         : list[int], default: 8 points log-spaced between
                          min_train_size and the full training set size.
    min_train_size      : int, default 16 -- lower end of the default
                          train_sizes ladder (ignored if train_sizes given).
    n_train_size_points  : int, default 8 -- ignored if train_sizes given.
    n_repeats           : int, default 5 -- independent random subsets per
                          (model, train_size) cell.
    max_epochs          : int, default 50 -- epoch count *at the largest
                          train_size in the sweep*; used to derive
                          steps_budget (see "Training budget" section
                          above). Smaller train_sizes get proportionally
                          more epochs so every size trains for the same
                          total number of gradient steps.
    steps_budget        : int, default None -- total optimizer steps to
                          match across every train_size. If omitted
                          (default), derived as
                          `max_epochs * steps_per_epoch(max(train_sizes))`.
                          Set this directly to bypass that derivation.
    batch_size          : int, default 256 (capped at the subset size)
    mask_fraction       : float | dict[str, float], default 0.1
    eval_mask_fraction  : float, default 0.2 -- fixed "hard" fraction used
                          for the imputation MSE/correlation eval metric
                          (models.evaluate_imputation()).
    n_eval_mask_draws   : int, default 10
    seed                : int, default 0
    output              : str, default "sample_efficiency_results.csv" --
                          one row per (model, train_size, repeat), columns:
                          model, train_size, repeat, epochs_run, steps_run,
                          test_loss (diagnostic only -- see "Cross-model
                          metric comparability" above), recon_loss_type2
                          (VAE family only, None otherwise),
                          imputation_mse_mean/std, imputation_corr_mean/std
                          (both cross-model-comparable),
                          effective_K_population (VampPrior family only),
                          deq_residual (DEQ family only).
    summary_output      : str, default "sample_efficiency_summary.json" --
                          mean/std of the above aggregated over repeats
    plot_output         : str, default "sample_efficiency.png"

VampPrior-family pseudo-input initialization
-----------------------------------------------
VampPriorVAE / DEQEncoderVampVAE are constructed here with
`pseudo_init_samples` set to the current subset's real data (type-1 NaNs
replaced via `_random_fill`'s per-gene N(mean, std) draw, not a fixed
sentinel -- see AGENTS.md's "Known Issues") rather than random noise,
matching tuning.py's own run_trial/retrain (see tuning.py's module
docstring and models.py's `_make_vamp_vae` docstring).
"""

import argparse
import csv
import json
import math

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import (
    model_factories, epoch_vae, epoch_transformer, evaluate_imputation,
    collect_deq_diagnostics, collect_vamp_diagnostics,
    make_log_bin_edges, device,
)
from synthetic_data import _random_fill
from tuning import (
    ALL_TUNABLE_MODELS, TRANSFORMER_FAMILY_MODELS, VAMP_FAMILY_MODELS,
    resolve_model_config,
)

_CONFIG_DEFAULTS = {
    "tuned_configs_fn":    "tuned_configs.json",
    "min_train_size":      16,
    "n_train_size_points": 8,
    "n_repeats":           5,
    "max_epochs":          50,
    "steps_budget":        None,
    "batch_size":          256,
    "mask_fraction":       0.1,
    "eval_mask_fraction":  0.2,
    "n_eval_mask_draws":   10,
    "seed":                0,
    "output":              "sample_efficiency_results.csv",
    "summary_output":      "sample_efficiency_summary.json",
    "plot_output":         "sample_efficiency.png",
}


def load_config(path: str) -> dict:
    """Load and validate the JSON sample-efficiency config, merged over
    _CONFIG_DEFAULTS."""
    with open(path) as f:
        user_config = json.load(f)

    config = {**_CONFIG_DEFAULTS, **user_config}

    if "problem_fn" not in config:
        raise ValueError(f"sample efficiency config {path!r} is missing required key 'problem_fn'")

    if "models" in config:
        unknown = set(config["models"]) - set(ALL_TUNABLE_MODELS) - {"MeansModel"}
        if unknown:
            raise ValueError(
                f"Unsupported model(s) in config['models']: {sorted(unknown)}. "
                f"Supported: {ALL_TUNABLE_MODELS + ['MeansModel']}"
            )

    return config


def default_train_sizes(n_cells: int, min_train_size: int, n_points: int) -> list[int]:
    """Log-spaced decreasing subset sizes from n_cells down to
    min_train_size (inclusive at both ends), deduplicated and sorted
    descending."""
    if min_train_size >= n_cells:
        return [n_cells]
    raw = np.geomspace(n_cells, min_train_size, num=n_points)
    sizes = sorted(set(int(round(s)) for s in raw), reverse=True)
    return sizes


def steps_per_epoch(size: int, batch_size_cfg: int) -> int:
    """Number of optimizer steps one pass over a training subset of `size`
    takes, given the configured (pre-shrink) batch size -- i.e. accounting
    for `bsz = min(batch_size_cfg, size)` the same way `run()`'s
    DataLoader construction does. See the module docstring's "Training
    budget: gradient steps, not epochs" section for why this matters."""
    return math.ceil(size / min(batch_size_cfg, size))


def mask_fraction_for(config: dict, model_name: str) -> float:
    mf = config["mask_fraction"]
    if isinstance(mf, dict):
        if model_name not in mf:
            raise ValueError(
                f"config['mask_fraction'] is a dict but has no entry for {model_name!r}"
            )
        return mf[model_name]
    return mf


def _check_tuning_provenance(config: dict, tuned_configs: dict, models_to_run: list[str]) -> None:
    """Warn (not error) if this sweep's batch_size/mask_fraction don't match
    what tuning.py actually used to produce tuned_configs -- a mismatch here
    silently changes the total optimizer-step count and/or task difficulty
    relative to what each model's hyperparameters were tuned for (see the
    module docstring's "mask_fraction caveat" and "Training budget"
    sections). tuning.py's run() writes this provenance into a reserved
    "_meta" key (added alongside the per-model entries) -- older
    tuned_configs.json files without it are simply skipped (nothing to
    check against)."""
    meta = tuned_configs.get("_meta")
    if not meta:
        print("  [provenance] tuned_configs has no '_meta' block (produced by an older "
              "tuning.py) -- skipping batch_size/mask_fraction consistency check.")
        return

    if "batch_size" in meta and meta["batch_size"] != config["batch_size"]:
        print(
            f"  [provenance WARNING] sample_efficiency batch_size={config['batch_size']!r} "
            f"!= tuning batch_size={meta['batch_size']!r}. This changes the number of "
            f"optimizer steps/epoch relative to what these hyperparameters were tuned for "
            f"(see module docstring's 'Training budget' section) -- results, especially at "
            f"the largest train_size, may not reproduce what tuning.py saw."
        )

    for model_name in models_to_run:
        if model_name == "MeansModel":
            continue
        try:
            mf = mask_fraction_for(config, model_name)
        except ValueError:
            continue
        if "mask_fraction" in meta and mf != meta["mask_fraction"]:
            print(
                f"  [provenance WARNING] {model_name}: sample_efficiency mask_fraction={mf!r} "
                f"!= tuning mask_fraction={meta['mask_fraction']!r} -- this is a different "
                f"task difficulty than what this model's hyperparameters were tuned for "
                f"(see module docstring's 'mask_fraction caveat')."
            )


def train_and_eval_one(model_name: str,
                        arch_cfg: dict,
                        train_kwargs: dict,
                        lr: float,
                        n_genes: int,
                        loader_train,
                        loader_test,
                        n_epochs: int,
                        mask_fraction: float,
                        eval_mask_fraction: float,
                        n_eval_mask_draws: int,
                        subset_data: torch.Tensor) -> dict:
    """Build a fresh model_name instance, train it from scratch on
    loader_train for n_epochs epochs (the caller resolves this per
    train_size to hit a fixed total-step budget -- see run()/module
    docstring's "Training budget: gradient steps, not epochs"), evaluate
    it on loader_test (held-out loss + unified imputation metrics), and
    return a metrics dict.

    subset_data is passed separately (rather than re-derived from
    loader_train) so that (a) MeansModel -- which has no training loop at
    all -- can fit its per-gene means directly, and (b) VampPriorVAE /
    DEQEncoderVampVAE can initialize their pseudo-inputs from real data
    (see VAMP_FAMILY_MODELS below) instead of random noise.

    NOTE on cross-model comparability: `test_loss` is returned for
    diagnostic/health-check purposes only (see benchmark_plan.md's
    convergence checklist) -- it is NOT comparable across model families
    (different loss units for ImputationTransformer vs. the VAE family,
    different independently-tuned beta/gamma/regularizer weights baked
    into the VAE family's own loss number). Use `imputation_mse_mean` /
    `imputation_corr_mean` (from models.evaluate_imputation(), the same
    function tuning.py's Optuna objective uses) for any cross-model
    comparison instead.
    """
    recon_loss_type2 = None  # only meaningful for VAE-family models

    if model_name == "MeansModel":
        model = model_factories["MeansModel"](n_genes).to(device)
        model.means.data = subset_data.nanmean(dim=0).to(device)
        test_loss = float("nan")  # no epoch_vae-style loss defined for MeansModel

    elif model_name in TRANSFORMER_FAMILY_MODELS:
        model = model_factories[model_name](n_genes, config=arch_cfg).to(device)
        bin_edges = make_log_bin_edges(model.n_bins, max_val=8.5)
        model.bin_edges = bin_edges  # impute()/impute_transformer() read this attribute
        opt = optim.AdamW(model.parameters(), lr=lr)

        for _ in range(n_epochs):
            epoch_transformer(model, loader_train, bin_edges, opt,
                               mask_fraction=mask_fraction, device=device, **train_kwargs)

        model.eval()
        test_loss, *_ = epoch_transformer(model, loader_test, bin_edges, None,
                                           mask_fraction=mask_fraction, device=device, **train_kwargs)

    else:
        # VampPriorVAE / DEQEncoderVampVAE: initialize pseudo-inputs from
        # real data (subset_data) instead of random noise -- type-1 NaNs are
        # replaced via _random_fill's per-gene N(mean, std) draw, not a
        # fixed sentinel (see AGENTS.md's "Known Issues" and
        # VAMP_FAMILY_MODELS / models.py's _make_vamp_vae docstring).
        factory_kwargs = {}
        if model_name in VAMP_FAMILY_MODELS:
            factory_kwargs["pseudo_init_samples"] = _random_fill(
                subset_data, torch.isnan(subset_data)).to(device)

        model = model_factories[model_name](n_genes, config=arch_cfg, **factory_kwargs).to(device)
        opt = optim.AdamW(model.parameters(), lr=lr)

        for _ in range(n_epochs):
            epoch_vae(model, loader_train, opt, mask_fraction=mask_fraction, **train_kwargs)

        model.eval()
        test_loss, _grad_norm, _recon_loss, recon_loss_type2, _kl_loss = epoch_vae(
            model, loader_test, None, mask_fraction=mask_fraction, **train_kwargs)
        recon_loss_type2 = float(recon_loss_type2)

    if not isinstance(test_loss, float):
        test_loss = float(test_loss)

    # Health-check diagnostics (safe to call on any model type -- entries
    # are None if not applicable, e.g. DEQEncoderVAE has no VampPrior
    # diagnostics and GeneExpressionVAE has no DEQ diagnostics). Snapshot
    # right after the eval-mode forward pass above and before
    # evaluate_imputation()'s own forward passes overwrite this state.
    deq_diag = collect_deq_diagnostics(model)
    vamp_diag = collect_vamp_diagnostics(model)

    # Unified imputation metrics -- models.evaluate_imputation() already
    # handles VAEBase subclasses / MeansModel / ImputationTransformer with
    # the same call signature (via impute()), which is exactly what lets
    # this script sweep multiple model families through one loop, and is
    # the SAME function tuning.py's Optuna objective uses -- so these
    # numbers are directly comparable across model families, unlike
    # test_loss above.
    imp_metrics = evaluate_imputation(model, loader_test, eval_mask_fraction,
                                       n_eval_mask_draws, device=device)

    return {
        "test_loss": test_loss if math.isfinite(test_loss) else None,
        "recon_loss_type2": recon_loss_type2,
        **imp_metrics,
        "effective_K_population": vamp_diag["effective_K_population"],
        "deq_residual": deq_diag["deq_residual"],
    }


def run(config_path: str) -> tuple[list[dict], dict]:
    """Plain entry point, safe to call directly from a notebook cell:

        import sample_efficiency
        rows, summary = sample_efficiency.run("sample_efficiency_config.json")

    Returns (rows, summary):
        rows    -- list of per-(model, train_size, repeat) result dicts,
                   also written to config["output"] as CSV (see "JSON config
                   schema"'s "output" entry above for the full column list).
        summary -- {model_name: {train_size: {"imputation_corr_mean": ...,
                    "imputation_mse_mean": ..., "test_loss_mean": ..., ...}}}
                   (see _summarize()), also written to
                   config["summary_output"] as JSON and plotted to
                   config["plot_output"].
    """
    config = load_config(config_path)

    print(f"Loading tuned configs from {config['tuned_configs_fn']!r} ...")
    with open(config["tuned_configs_fn"]) as f:
        tuned_configs = json.load(f)

    # "_meta" (if present) is tuning.py's own provenance block, not a model
    # name -- exclude it from the default model list.
    models_to_run = config.get("models") or [k for k in tuned_configs if not k.startswith("_")]

    _check_tuning_provenance(config, tuned_configs, models_to_run)

    print(f"Loading problem from {config['problem_fn']!r} ...")
    loaded = torch.load(config["problem_fn"])
    n_genes = loaded["n_genes"]
    data_train_full = loaded["gene_reads_train"]
    data_test = loaded["gene_reads_test"]
    n_cells_train = data_train_full.shape[0]
    n_cells_test = data_test.shape[0]

    loader_test = DataLoader(data_test, batch_size=n_cells_test, shuffle=False)

    train_sizes = config.get("train_sizes") or default_train_sizes(
        n_cells_train, config["min_train_size"], config["n_train_size_points"]
    )
    train_sizes = sorted(set(train_sizes), reverse=True)

    print(
        f"n_genes={n_genes}  n_cells_train={n_cells_train}  n_cells_test={n_cells_test}  "
        f"models={models_to_run}  train_sizes={train_sizes}  n_repeats={config['n_repeats']}  "
        f"device={device}"
    )

    # Resolve each (non-baseline) model's tuned arch/train config once --
    # it doesn't depend on train_size/repeat.
    resolved = {}
    for model_name in models_to_run:
        if model_name == "MeansModel":
            resolved[model_name] = ({}, {}, None)
            continue
        if model_name not in tuned_configs:
            raise ValueError(
                f"model {model_name!r} not found in {config['tuned_configs_fn']!r} "
                f"(available: {list(tuned_configs)})"
            )
        params = tuned_configs[model_name]["best_params"]
        resolved[model_name] = resolve_model_config(model_name, params, n_genes)

    rows = []
    n_repeats = config["n_repeats"]
    batch_size_cfg = config["batch_size"]
    max_epochs = config["max_epochs"]
    eval_mask_fraction = config["eval_mask_fraction"]
    n_eval_mask_draws = config["n_eval_mask_draws"]

    # Fixed total-step budget across every train_size -- see module
    # docstring's "Training budget: gradient steps, not epochs" section.
    # Default reproduces today's max_epochs exactly at the largest
    # train_size; smaller sizes get proportionally more epochs so every
    # size trains for the same total number of gradient steps (rather than
    # the same number of epochs, which under-trains small subsets since
    # batch_size is also capped at the subset size).
    steps_budget = config["steps_budget"] or max_epochs * steps_per_epoch(max(train_sizes), batch_size_cfg)
    epochs_for_size = {
        size: max(1, round(steps_budget / steps_per_epoch(size, batch_size_cfg)))
        for size in train_sizes
    }

    print(f"steps_budget={steps_budget}  "
          f"(== max_epochs={max_epochs} at train_size={max(train_sizes)})")
    for size in train_sizes:
        spe = steps_per_epoch(size, batch_size_cfg)
        print(f"  train_size={size:6d}  steps/epoch={spe:4d}  "
              f"epochs={epochs_for_size[size]:5d}  total_steps={epochs_for_size[size] * spe}")

    total = len(models_to_run) * len(train_sizes) * n_repeats
    pbar = tqdm(total=total, desc="sample efficiency sweep")

    for repeat in range(n_repeats):
        seed_i = config["seed"] + repeat
        # Same permutation for every (model, train_size) in this repeat --
        # see module docstring's "Paired subsets" note. Truncating to the
        # first `size` indices nests smaller subsets inside larger ones.
        perm = torch.randperm(n_cells_train, generator=torch.Generator().manual_seed(seed_i))

        for size in train_sizes:
            idx = perm[:size]
            subset_data = data_train_full[idx]
            bsz = min(batch_size_cfg, size)

            for model_name in models_to_run:
                arch_cfg, train_kwargs, lr = resolved[model_name]
                mask_fraction = mask_fraction_for(config, model_name)

                # Fresh DataLoader (and generator) per model: each model's
                # shuffle-order sequence is seeded identically off seed_i
                # rather than sharing one DataLoader's stateful generator
                # across models, which would otherwise let each model's
                # minibatch order depend on how many epochs the *previous*
                # model in the loop consumed from it (breaking the "same
                # data, per repeat" pairing this loop is designed around).
                loader_train = DataLoader(
                    subset_data, batch_size=bsz, shuffle=True,
                    generator=torch.Generator().manual_seed(seed_i),
                )

                epochs_this_size = epochs_for_size[size]

                torch.manual_seed(seed_i)  # reproducible model init per (repeat, size, model)
                metrics = train_and_eval_one(
                    model_name, arch_cfg, train_kwargs, lr, n_genes,
                    loader_train, loader_test, epochs_this_size,
                    mask_fraction, eval_mask_fraction, n_eval_mask_draws,
                    subset_data,
                )

                row = {
                    "model": model_name, "train_size": size, "repeat": repeat,
                    "epochs_run": epochs_this_size,
                    "steps_run": epochs_this_size * steps_per_epoch(size, batch_size_cfg),
                    **metrics,
                }
                rows.append(row)
                pbar.set_postfix(model=model_name, size=size, repeat=repeat,
                                  corr=f"{metrics['imputation_corr_mean']:.3f}")
                pbar.update(1)

    pbar.close()

    _write_csv(rows, config["output"])
    summary = _summarize(rows)
    with open(config["summary_output"], "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-run results to {config['output']}")
    print(f"Wrote summary to {config['summary_output']}")

    _plot(summary, config["plot_output"])
    print(f"Wrote plot to {config['plot_output']}")

    return rows, summary


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    """{model_name: {train_size: {imputation_corr_mean, imputation_corr_std,
    imputation_mse_mean, imputation_mse_std, test_loss_mean, test_loss_std,
    recon_loss_type2_mean, recon_loss_type2_std, effective_K_population_mean,
    deq_residual_mean, n}}}, aggregated over repeats.

    imputation_mse_mean/std -- unlike test_loss -- is directly comparable
    across model families (see models.evaluate_imputation()'s docstring),
    so it (or imputation_corr_mean) is the right field to use for any
    cross-model comparison.
    """
    by_key: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_key.setdefault((row["model"], row["train_size"]), []).append(row)

    # Fields that are already per-repeat means (or None when not
    # applicable, e.g. recon_loss_type2 for ImputationTransformer, or
    # effective_K_population for non-VampPrior models) -- aggregated the
    # same generic way rather than one-off per field.
    mean_fields = [
        "imputation_corr_mean", "imputation_mse_mean",
        "test_loss", "recon_loss_type2",
        "effective_K_population", "deq_residual",
    ]

    summary: dict = {}
    for (model_name, size), group in by_key.items():
        entry = {"n": len(group)}
        for field in mean_fields:
            vals = np.array([r[field] for r in group if r.get(field) is not None], dtype=float)
            out_name = field if field.endswith("_mean") else f"{field}_mean"
            entry[out_name] = float(np.nanmean(vals)) if vals.size else None
            entry[out_name.replace("_mean", "_std")] = float(np.nanstd(vals)) if vals.size else None
        summary.setdefault(model_name, {})[str(size)] = entry
    return summary


def _plot(summary: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, by_size in summary.items():
        sizes = sorted((int(s) for s in by_size), reverse=True)
        means = [by_size[str(s)]["imputation_corr_mean"] for s in sizes]
        stds = [by_size[str(s)]["imputation_corr_std"] for s in sizes]
        ax.errorbar(sizes, means, yerr=stds, marker="o", ms=4, capsize=3, label=model_name)

    ax.set_xscale("log")
    ax.set_xlabel("Training-set size (n_cells)")
    ax.set_ylabel("Imputation correlation (held-out test set)")
    ax.set_title("Sample efficiency: imputation quality vs. training-set size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> tuple[list[dict], dict]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="sample_efficiency_config.json",
        help="Path to the JSON sample-efficiency config (see module docstring)",
    )
    args = parser.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    main()
