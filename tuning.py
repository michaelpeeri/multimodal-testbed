"""
Optuna-based hyperparameter tuning for the VAE model family and
ImputationTransformer in models.py.

Usage
-----
As a script:

    python tuning.py --config tuning_config.json

From a notebook (e.g. Google Colab -- after `!pip install optuna`, since
Colab runtimes don't persist installed packages across sessions):

    import tuning
    results = tuning.run("tuning_config.json")

What this does
---------------
For each model name listed in the config's "models" list, runs its own
Optuna study (stored in the same sqlite database, one study per model,
so results/progress persist across separate invocations -- e.g. across
Colab runtime restarts).

For VAE_FAMILY_MODELS, this searches the training-loop hyperparameters
listed in "search_space" (learning rate, gamma, min_free_bits,
lambda_entropy, lambda_vamp_repulsion, lambda_vamp_coverage -- any subset
of epoch_vae's kwargs other than mask_fraction and beta). `beta` is
deliberately *not* tunable -- see "beta is fixed, not tuned" below. For
TRANSFORMER_FAMILY_MODELS
(currently just ImputationTransformer), the analogous training-loop keys are
"lr" and "lambda_conf" (any subset of epoch_transformer's kwargs other than
mask_fraction) -- see "ImputationTransformer architecture search" below for
its own limited-architecture-search keys. Both families' search_space keys
may be freely mixed in the same config's "search_space" dict: each model
only picks out the keys relevant to its own family and ignores the rest.

Limited architecture search (layer widths only, fixed depth)
--------------------------------------------------------------
Three additional, independently opt-in "search_space" keys control a
*narrow* slice of model architecture -- the width of each encoder/decoder
layer -- while the *number* of layers always stays fixed at each model's
own `model_factories` default depth:

    "latent_dim"           : {"type": "int", "low":..., "high":...}
    "encoder_width_factor" : {"type": "float", "low":..., "high":..., "log": bool}
    "decoder_width_factor" : {"type": "float", "low":..., "high":..., "log": bool}

If "latent_dim" is listed, it is tuned per-trial; otherwise the model's own
default latent_dim is used as the fixed base value below. If
"encoder_width_factor" is listed, that model's encoder_dims (length L_enc,
fixed to the model's default depth) is *derived* every trial as a geometric
taper from the input side down to the latent bottleneck:

    encoder_dims[k] = round(latent_dim * encoder_width_factor ** (L_enc - k))
                      for k = 0 .. L_enc - 1

Similarly, if "decoder_width_factor" is listed, decoder_dims (length
L_dec) is derived as the mirror-image taper, expanding back out from the
latent bottleneck to the output side:

    decoder_dims[k] = round(latent_dim * decoder_width_factor ** (k + 1))
                      for k = 0 .. L_dec - 1

Any of these three keys may be used independently of the others (e.g.
tuning encoder_width_factor alone leaves decoder_dims and, if
"latent_dim" is absent, latent_dim itself at the model's defaults).

Because a given (latent_dim, width_factor) combination can produce
unreasonably tiny or huge layers, every derived width is checked at
runtime against the top-level "min_layer_width"/"max_layer_width" config
keys; a trial whose derived architecture falls outside those bounds is
pruned immediately (before any training) rather than run or crash. The
actual resolved encoder_dims/decoder_dims/latent_dim for each trial are
recorded as trial user attributes for inspection.

All other VAE_FAMILY_MODELS *architecture* hyperparameters (n_pseudo,
coeff/max_iter, n_components, ...) are intentionally out of scope for this
pass and stay at each model's `model_factories` default.

ImputationTransformer architecture search
------------------------------------------
Unlike the VAE family's width-taper scheme, ImputationTransformer's own
`model_factories`/`__init__` kwargs are tunable directly -- each maps
one-to-one onto a "search_space" key, with no derivation needed:

    "d_model"      : {"type": "int",   "low":..., "high":...}
    "n_heads"      : {"type": "int",   "low":..., "high":...}
    "n_layers"     : {"type": "int",   "low":..., "high":...}
    "n_bins"       : {"type": "int",   "low":..., "high":...}
    "n_conf_bins"  : {"type": "int",   "low":..., "high":...}
    "d_gene"       : {"type": "int",   "low":..., "high":...}
    "d_count"      : {"type": "int",   "low":..., "high":...}
    "d_conf"       : {"type": "int",   "low":..., "high":...}
    "dropout"      : {"type": "float", "low":..., "high":..., "log": bool}

Each key is independently opt-in (omitted keys keep
IMPUTATION_TRANSFORMER_CONFIG's default). A trial whose suggested
"d_model" isn't evenly divisible by "n_heads" (a hard nn.TransformerEncoder
requirement) is pruned immediately, as is one whose resolved "d_model"
falls outside [min_layer_width, max_layer_width] (d_model being this
model's analogue of the VAE family's layer width).

`mask_fraction` is deliberately *not* part of the search space (for either
family): it sets the difficulty of the imputation problem/objective itself
rather than a knob to be optimized away, so it is read directly from the
config's top-level "mask_fraction" key (same value applied to every
model/trial) instead of being suggested by Optuna.

beta is fixed, not tuned
-------------------------
`beta` (the VAE family's KL weight, see synthetic_data.masked_loss) is
*also* deliberately excluded from the search space, for a different reason
than mask_fraction: under a pure reconstruction/imputation-accuracy
objective (which is what `imputation_mse_mean` is -- see "The objective
being minimized" below), there is no term in the metric that rewards a
non-trivial KL weight at all, so Optuna has every incentive to drive beta
toward the lowest value it's allowed to reach, regardless of what that
range is. Empirically this is exactly what happened: across all 4
VAE-family models in an earlier tuning run, beta converged to
0.0002-0.0011 -- pinned at the floor of its old [1e-4, 0.2] search range --
while `min_free_bits` (the other KL-related knob, still tunable) spanned
0.004-0.040, i.e. it did *not* show the same floor-pinning behavior and
apparently carries real signal even under this objective. So only `beta`
is fixed here, read from the top-level "beta" config key (default 1e-4,
matching models.py's own `beta_constant()` helper's default -- an
independent piece of evidence, from an unrelated RL-controller subsystem
in this codebase, that ~1e-4 is this architecture's established
"reasonable" fixed operating point rather than an arbitrary pick) and
applied identically to every VAE_FAMILY_MODELS trial, the same way
mask_fraction is.

The objective being minimized is `imputation_mse_mean` -- the held-out
imputation MSE from `models.evaluate_imputation()` (the same
model-family-agnostic metric sample_efficiency.py reports), evaluated with
`eval_mask_fraction`/`n_eval_mask_draws` -- NOT the raw training loss.
Raw training loss (`epoch_vae`'s / `epoch_transformer`'s total loss) is
still computed every `eval_every` epochs and recorded as a trial user
attribute (`trial.user_attrs["test_loss"]`) for diagnostic/health-check
purposes (see benchmark_plan.md's convergence checklist), but it is NOT
comparable across model families -- see models.evaluate_imputation()'s and
sample_efficiency.py's module docstrings for why (different loss units for
ImputationTransformer vs. the VAE family; independently-tuned
beta/gamma/regularizer weights baked into the VAE family's own loss
number) -- so using it as the tuning objective would let Optuna minimize
"loss" partly by de-weighting the very quantity (held-out reconstruction)
that actually determines imputation quality, rather than by improving
imputation quality itself. Trials report intermediate imputation_mse_mean
every `eval_every` epochs so the configured pruner can stop clearly-bad
trials early.

Only models trained via `epoch_vae`'s standard 6-tuple `forward()`
signature (VAE_FAMILY_MODELS) or `epoch_transformer`'s (count_bins,
conf_bins) `forward()` signature (TRANSFORMER_FAMILY_MODELS) are supported.
`MeansModel` has no hyperparameters to tune (its means are fit directly
from data, not via a training loop) and is out of scope here.

Problem file schema (loaded via `torch.load(config["problem_fn"])`)
--------------------------------------------------------------------
A dict with (at least) the following keys:
    'n_genes', 'n_cells'          : int
    'gene_reads_train'            : (n_cells, n_genes) float tensor, NaN = missing
    'gene_reads_test'             : (n_cells, n_genes) float tensor, NaN = missing
(additional keys such as 'cell_clusters_test', 'hidden_state_*',
'gene_group_membership' may be present and are currently unused here.)

JSON config schema
-------------------
See tuning_config.example.json (VAE_FAMILY_MODELS) and
tuning_config.transformer.example.json (ImputationTransformer) for full
examples. Keys:
    problem_fn    : str, required -- path to the problem file described above
    models        : list[str], required -- subset of ALL_TUNABLE_MODELS
                    (VAE_FAMILY_MODELS + TRANSFORMER_FAMILY_MODELS)
    search_space  : dict, required -- {param_name: spec}, where spec is
                    {"type": "float"|"int", "low":..., "high":..., "log": bool}
                    or {"type": "categorical", "choices": [...]}. For
                    VAE_FAMILY_MODELS: "lr" controls the optimizer's learning
                    rate; "latent_dim" (type "int"), "encoder_width_factor"
                    and "decoder_width_factor" (type "float") control the
                    limited architecture search described above; any other
                    name must be one of _VAE_TRAIN_SEARCH_SPACE_KEYS (an
                    epoch_vae() kwarg other than beta -- see "beta is fixed,
                    not tuned" above). For TRANSFORMER_FAMILY_MODELS: "lr"
                    and "lambda_conf" control epoch_transformer's kwargs;
                    "d_model"/"n_heads"/"n_layers"/"n_bins"/"n_conf_bins"/
                    "d_gene"/"d_count"/"d_conf" (type "int") and "dropout"
                    (type "float") control the architecture search described
                    above. A key not applicable to a given model's family is
                    simply ignored for that model (left at its own default),
                    so VAE and transformer search_space keys may coexist in
                    the same dict when tuning both families together.
                    "mask_fraction" and "beta" may not be listed here -- see
                    the top-level "mask_fraction"/"beta" config keys instead.
    mask_fraction : float, default 0.1 -- fixed (not tuned) fraction of
                    observed genes masked for training/eval, forwarded as
                    epoch_vae's/epoch_transformer's mask_fraction kwarg for
                    every trial.
    beta          : float, default 1e-4 -- fixed (not tuned) VAE_FAMILY_MODELS
                    KL weight, forwarded as epoch_vae's beta kwarg for every
                    trial (ignored for TRANSFORMER_FAMILY_MODELS). See "beta
                    is fixed, not tuned" above for why this isn't part of
                    search_space.
    eval_mask_fraction : float, default: same as mask_fraction -- fraction
                    used for the imputation_mse_mean objective (see
                    "The objective being minimized" above). Exposed
                    separately from mask_fraction in case you want the
                    tuning objective itself evaluated at a different
                    (e.g. harder/fixed) difficulty than the training-time
                    masking -- mirrors sample_efficiency.py's own
                    mask_fraction/eval_mask_fraction split.
    n_eval_mask_draws : int, default 3 -- number of independent mask draws
                    to average per eval_every-epoch evaluation. Kept lower
                    than sample_efficiency.py's default of 10 since this
                    runs every eval_every epochs x every trial (much more
                    frequently); noise partly averages out across the many
                    evaluations within a trial and across trials.
    min_layer_width : int, default 4 -- lower bound on every derived
                    encoder/decoder layer width (VAE_FAMILY_MODELS) or on
                    "d_model" (TRANSFORMER_FAMILY_MODELS); trials that fall
                    outside [min_layer_width, max_layer_width] are pruned
                    before training. Only enforced for the roles actually
                    driven by encoder_width_factor/decoder_width_factor/
                    d_model.
    max_layer_width : int, default 4096 -- upper bound, see min_layer_width.

    VRAM guard (TRANSFORMER_FAMILY_MODELS only -- ignored for VAE_FAMILY_MODELS)
    ----------------------------------------------------------------------------
    ImputationTransformer's attention cost is O(batch * n_heads * n_genes^2),
    so at larger n_genes an unlucky Optuna draw near the top of the search
    space (large d_model/n_heads/n_layers) can request far more VRAM than a
    "typical" trial -- enough to crash the whole study with a CUDA OOM
    partway through a sweep. run_trial_transformer estimates each trial's
    peak VRAM (via models.estimate_transformer_train_bytes/eval_bytes)
    *before* constructing the model and prunes it (trial.user_attrs
    ["prune_reason"] = "vram_budget_exceeded") if the estimate exceeds the
    budget below -- the same "prune before training" pattern already used
    for d_model/n_heads divisibility and min_layer_width/max_layer_width.
    vram_budget_gb       : float, default None -- explicit VRAM budget
                    override, in GB. If None (default), auto-detected as
                    vram_budget_fraction * the current CUDA device's total
                    memory; the guard is disabled (never prunes) if the
                    device isn't CUDA and no explicit override is given.
    vram_budget_fraction : float, default 0.9 -- only used for
                    auto-detection (see vram_budget_gb); leaves headroom
                    for CUDA context/other processes.
    vram_overhead_factor : float, default 0.65 -- multiplies the raw
                    estimate from estimate_transformer_train_bytes/
                    estimate_transformer_eval_bytes (which models only the
                    dominant tensors: Q/K/V, attention scores/softmax, FFN)
                    to approximate real measured peak usage. Calibrated
                    from a sanity-test measurement (n_genes=800, d_model=384,
                    n_heads=8, n_layers=6, batch=128): raw estimate ~43.7GB
                    vs. measured torch.cuda.max_memory_allocated() < 19GB
                    (ratio <= ~0.435), with a ~1.5x safety margin applied.
                    The raw formula OVER-estimates real usage here -- likely
                    because nn.TransformerEncoderLayer's internal self_attn
                    call uses need_weights=False, which lets PyTorch dispatch
                    to a fused scaled_dot_product_attention kernel that never
                    materializes the full O(batch*n_heads*n_genes^2)
                    attention/softmax matrix the raw formula assumes. The
                    margin (rather than the bare ~0.435 ratio) hedges against
                    that fused kernel falling back to its slower "math"
                    backend (full materialization) under some combination of
                    dropout/mask/head_dim/GPU/PyTorch-version -- recalibrate
                    as (measured peak bytes / raw estimate), with margin, if
                    you get measurements at other architectures/batch sizes.
    storage       : str, default "sqlite:///tuning.db"
    batch_size    : int, default 256 (train loader only)
    eval_batch_size : int, default: same as batch_size -- caps the test
                    loader's batch size (capped again at the test-set size
                    if smaller). Previously the test loader always used the
                    whole test set as one batch; for ImputationTransformer
                    that makes attention memory (O(batch * n_heads *
                    n_genes^2)) scale with test-set size rather than
                    architecture -- the actual cause of the 15-40GB VRAM
                    usage observed at n_genes=200. epoch_transformer/
                    epoch_vae/evaluate_imputation already pool/average
                    across loader_test's batches, so lowering this only
                    changes memory use, not results.
    n_trials      : int, default 50 (per model)
    max_epochs    : int, default 50
    eval_every    : int, default 1 -- epoch cadence for both the held-out
                    eval used for pruning/objective and the progress-bar
                    postfix update
    sampler       : "tpe" (default; only option for now)
    pruner        : "median" (default; only option for now)
    output        : str, default "tuned_configs.json" -- where the final
                    {model_name: {best_value, best_params, best_test_loss,
                    best_imputation_corr}, ..., "_meta": {batch_size,
                    mask_fraction, beta, max_epochs, eval_mask_fraction,
                    n_eval_mask_draws}} summary is written (see `run()`
                    below for the richer in-memory return value).
                    best_value is imputation_mse_mean (the tuning
                    objective, see above); "_meta" is provenance read back
                    by sample_efficiency.py to warn about batch_size/
                    mask_fraction mismatches between tuning and replay
                    (see sample_efficiency.py's module docstring) -- it is
                    not a model name and downstream readers of this file
                    should skip keys starting with "_".
    checkpoint_dir : str, default "checkpoints" -- directory where the
                    actual trained weights of the best trial seen so far for
                    each model are kept up to date during tuning, one file
                    per model: {checkpoint_dir}/{model_name}_best.pt. Point
                    this at the same persistent storage as `storage`/`output`
                    (e.g. a Drive folder) so it survives Colab runtime
                    restarts. See "Accessing the best trained model" below.

Accessing the best trained model
----------------------------------
Unlike `output`'s tuned_configs.json (hyperparameters only), each trial's
*actual trained weights* are checkpointed live to
{checkpoint_dir}/{model_name}_best.pt every time a trial improves on the
best imputation_mse_mean seen so far for that model (see "The objective
being minimized" above) -- so at any point during, or at the end of, a
tuning run you already have the literal best-performing model on disk, no
retraining required. The saved dict is exactly the {'model_type',
'state_dict', 'config'} format vae-test.py's load_model() expects (plus
'imputation_mse_mean', 'test_loss', 'imputation_corr_mean', 'params',
'trial_number' for provenance)::

    from vae_test import load_model   # or however load_model is imported
    best_model = load_model("checkpoints/GeneExpressionVAE_best.pt", n_genes)

Training additional models from the same configuration
----------------------------------------------------------
To train more independent copies of a tuned architecture (e.g. for an
ensemble, or to check run-to-run variance) -- as opposed to loading the one
exact model Optuna already trained -- use `retrain()`, which rebuilds the
architecture from a finished trial's `params` (via `resolve_model_config`)
and trains fresh copies from scratch on the *full* dataset::

    import json
    import tuning

    tuned  = json.load(open("tuned_configs.json"))
    params = tuned["GeneExpressionVAE"]["best_params"]
    models = tuning.retrain(
        "GeneExpressionVAE", params, n_genes, loader_train, loader_test,
        mask_fraction=0.4, max_epochs=100, n_copies=5,
        checkpoint_dir="checkpoints",   # optional, saves each as
                                         # {model_name}_retrain_{i}.pt
    )
"""

import argparse
import json
import math
import time
from pathlib import Path

import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import (
    model_factories, epoch_vae, device,
    epoch_transformer, make_log_bin_edges,
    IMPUTATION_TRANSFORMER_CONFIG,
    evaluate_imputation, collect_vamp_diagnostics,
    estimate_transformer_train_bytes, estimate_transformer_eval_bytes,
)
from synthetic_data import _random_fill


# Models trainable via epoch_vae's standard 6-tuple forward() convention.
# MeansModel (no hyperparameters, fit directly from data) is intentionally
# excluded. ImputationTransformer uses a different training loop
# (epoch_transformer) and is tuned separately -- see TRANSFORMER_FAMILY_MODELS
# below.
VAE_FAMILY_MODELS = [
    "GeneExpressionVAE",
    "DEQEncoderVAE",
    "MoVEVAE_K3",
    "MoVEVAE_K1",
    "VampPriorVAE",
    "DEQEncoderVampVAE",
]

# Models trained via epoch_transformer's (count_bins, conf_bins) forward()
# convention instead of epoch_vae's standard 6-tuple one.
TRANSFORMER_FAMILY_MODELS = [
    "ImputationTransformer",
]

# All models `run()`/`load_config()` will accept in config["models"].
ALL_TUNABLE_MODELS = VAE_FAMILY_MODELS + TRANSFORMER_FAMILY_MODELS

# VAE_FAMILY_MODELS subset that uses a VampPrior (pseudo-input mixture
# prior) -- these get their pseudo_inputs initialized from real training
# data rather than random noise (see run_trial/retrain and models.py's
# _make_vamp_vae/_make_deq_vamp_vae docstrings).
VAMP_FAMILY_MODELS = {"VampPriorVAE", "DEQEncoderVampVAE"}

_SAMPLERS = {
    "tpe": lambda: optuna.samplers.TPESampler(seed=0),
}

_PRUNERS = {
    "median": lambda: optuna.pruners.MedianPruner(n_warmup_steps=10),
}

# search_space keys that drive the limited architecture search (see module
# docstring) rather than an epoch_vae() kwarg -- suggest_train_cfg excludes
# these, run_trial's suggest_arch_cfg handles them separately.
_ARCH_SEARCH_SPACE_KEYS = {"latent_dim", "encoder_width_factor", "decoder_width_factor"}

# search_space keys accepted for VAE_FAMILY_MODELS training-loop tuning --
# must match epoch_vae's kwargs (other than mask_fraction and beta, which
# are fixed -- see module docstring's "beta is fixed, not tuned" section).
# Whitelisted (rather than "everything not an arch key") so that a
# search_space shared with TRANSFORMER_FAMILY_MODELS in the same config (e.g.
# tuning ImputationTransformer alongside VampPriorVAE) doesn't leak
# transformer-only keys like "lambda_conf" into epoch_vae() as unexpected
# kwargs.
_VAE_TRAIN_SEARCH_SPACE_KEYS = {
    "lr", "gamma", "min_free_bits",
    "lambda_entropy", "lambda_vamp_repulsion", "lambda_vamp_coverage",
}

# search_space keys that drive ImputationTransformer's *architecture* (passed
# straight through to model_factories' config dict -- no derivation needed,
# unlike the VAE family's width-taper scheme, since these are already the
# model's own __init__ kwargs).
_TRANSFORMER_ARCH_SEARCH_SPACE_KEYS = {
    "d_model", "n_heads", "n_layers",
    "n_bins", "n_conf_bins", "d_gene", "d_count", "d_conf", "dropout",
}
_TRANSFORMER_ARCH_INT_KEYS = {
    "d_model", "n_heads", "n_layers", "n_bins", "n_conf_bins",
    "d_gene", "d_count", "d_conf",
}

# search_space keys accepted for ImputationTransformer's training-loop tuning
# -- must match epoch_transformer's kwargs (other than mask_fraction, which
# is fixed, and bin_edges/device, which are derived/fixed internally).
_TRANSFORMER_TRAIN_SEARCH_SPACE_KEYS = {"lr", "lambda_conf"}

_CONFIG_DEFAULTS = {
    "storage":         "sqlite:///tuning.db",
    "batch_size":      256,
    "eval_batch_size": None,  # default: same as batch_size, see load_config -- caps the
                              # test-set DataLoader's batch size. Left at n_cells (the
                              # whole test set as one batch) this scales ImputationTransformer's
                              # O(batch * n_heads * n_genes^2) attention memory with dataset
                              # size instead of architecture, which is what actually drove the
                              # 15-40GB VRAM usage reported at n_genes=200 -- see AGENTS.md /
                              # the VRAM investigation this key was added for.
    "n_trials":        50,
    "max_epochs":      50,
    "eval_every":      1,
    "sampler":         "tpe",
    "pruner":          "median",
    "output":          "tuned_configs.json",
    "mask_fraction":   0.1,
    "beta":            1e-4,  # fixed, not tuned -- see module docstring's "beta is fixed, not tuned"
    "eval_mask_fraction": None,  # default: same as mask_fraction, see load_config
    "n_eval_mask_draws":  3,
    "min_layer_width": 4,
    "max_layer_width": 4096,
    "checkpoint_dir":  "checkpoints",

    # VRAM guard for ImputationTransformer trials (see run_trial_transformer /
    # estimate_transformer_train_bytes|eval_bytes in models.py, and the module
    # docstring's "VRAM guard" section below). Ignored for VAE_FAMILY_MODELS.
    "vram_budget_gb":       None,  # default: auto-detect, see load_config --
                                    # explicit override (in GB) for the VRAM
                                    # budget a trial's estimated usage is
                                    # checked against. Set to a number to
                                    # bypass auto-detection (e.g. on a shared
                                    # GPU, or to test the guard without CUDA).
    "vram_budget_fraction": 0.9,    # only used for auto-detection: fraction of
                                    # torch.cuda.get_device_properties(device)
                                    # .total_memory to treat as the budget,
                                    # leaving headroom for CUDA context/other
                                    # processes. Auto-detection is a no-op
                                    # (budget=None, guard never prunes) when
                                    # device is not CUDA.
    "vram_overhead_factor": 0.65,   # multiplies the raw estimate from
                                    # estimate_transformer_train_bytes/
                                    # estimate_transformer_eval_bytes to
                                    # approximate real measured peak usage.
                                    # Calibrated from a sanity-test measurement
                                    # (n_genes=800, d_model=384, n_heads=8,
                                    # n_layers=6, batch=128): raw estimate
                                    # ~43.7GB vs. measured
                                    # torch.cuda.max_memory_allocated() < 19GB
                                    # (ratio <= ~0.435, x1.5 margin applied).
                                    # The raw formula over-estimates here,
                                    # likely because nn.TransformerEncoderLayer
                                    # dispatches self-attention (need_weights=
                                    # False) to a fused scaled_dot_product_
                                    # attention kernel that avoids materializing
                                    # the full O(batch*n_heads*n_genes^2)
                                    # attention matrix the raw formula assumes.
                                    # Recalibrate as (measured_bytes /
                                    # raw_estimate), with margin, if you get
                                    # measurements at other architectures.
}


def load_config(path: str) -> dict:
    """Load and validate the JSON tuning config, merged over _CONFIG_DEFAULTS."""
    with open(path) as f:
        user_config = json.load(f)

    config = {**_CONFIG_DEFAULTS, **user_config}

    for key in ("problem_fn", "models", "search_space"):
        if key not in config:
            raise ValueError(f"tuning config {path!r} is missing required key {key!r}")

    unknown_models = set(config["models"]) - set(ALL_TUNABLE_MODELS)
    if unknown_models:
        raise ValueError(
            f"Unsupported model(s) in config['models']: {sorted(unknown_models)}. "
            f"Supported: {ALL_TUNABLE_MODELS}"
        )

    if "mask_fraction" in config["search_space"]:
        raise ValueError(
            "search_space['mask_fraction'] is not allowed -- mask_fraction is a "
            "fixed problem-difficulty knob, set it via the top-level "
            "'mask_fraction' config key instead"
        )

    if "beta" in config["search_space"]:
        raise ValueError(
            "search_space['beta'] is not allowed -- beta has no interior "
            "optimum under a pure imputation-accuracy objective (it converges "
            "to whatever floor the search range allows, see module "
            "docstring's 'beta is fixed, not tuned'); set it via the "
            "top-level 'beta' config key instead"
        )

    for name, spec in config["search_space"].items():
        if spec.get("type") not in ("float", "int", "categorical"):
            raise ValueError(
                f"search_space[{name!r}] has unsupported type {spec.get('type')!r} "
                f"(expected 'float', 'int', or 'categorical')"
            )

    if "latent_dim" in config["search_space"] and config["search_space"]["latent_dim"]["type"] != "int":
        raise ValueError("search_space['latent_dim'] must have type 'int'")
    for name in ("encoder_width_factor", "decoder_width_factor"):
        if name in config["search_space"] and config["search_space"][name]["type"] != "float":
            raise ValueError(f"search_space[{name!r}] must have type 'float'")

    for name in _TRANSFORMER_ARCH_INT_KEYS:
        if name in config["search_space"] and config["search_space"][name]["type"] != "int":
            raise ValueError(f"search_space[{name!r}] must have type 'int'")
    if "dropout" in config["search_space"] and config["search_space"]["dropout"]["type"] != "float":
        raise ValueError("search_space['dropout'] must have type 'float'")

    if config["sampler"] not in _SAMPLERS:
        raise ValueError(f"Unsupported sampler {config['sampler']!r}. Supported: {list(_SAMPLERS)}")
    if config["pruner"] not in _PRUNERS:
        raise ValueError(f"Unsupported pruner {config['pruner']!r}. Supported: {list(_PRUNERS)}")

    if config["eval_mask_fraction"] is None:
        config["eval_mask_fraction"] = config["mask_fraction"]

    if config["eval_batch_size"] is None:
        config["eval_batch_size"] = config["batch_size"]

    # Derived (not user-facing) key: the resolved VRAM budget in bytes, computed
    # once here so run()/run_study_for_model don't each need to repeat the
    # auto-detection logic. None means "no guard" (non-CUDA device and no
    # explicit vram_budget_gb override).
    config["vram_budget_bytes"] = _resolve_vram_budget_bytes(config)

    return config


def _resolve_vram_budget_bytes(config: dict) -> float | None:
    """VRAM guard budget, in bytes, for ImputationTransformer trials (see
    run_trial_transformer). Explicit config['vram_budget_gb'] wins if set;
    otherwise auto-detected as vram_budget_fraction * the current CUDA
    device's total memory, or None (guard disabled) if device isn't CUDA."""
    if config["vram_budget_gb"] is not None:
        return config["vram_budget_gb"] * 1e9
    if device.type != "cuda":
        return None
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    return total_bytes * config["vram_budget_fraction"]


def suggest_from_spec(trial: optuna.Trial, name: str, spec: dict):
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported search_space type {kind!r} for {name!r}")


def suggest_train_cfg(trial: optuna.Trial, search_space: dict) -> dict:
    """Suggest every epoch_vae-kwarg parameter listed in search_space (i.e.
    every _VAE_TRAIN_SEARCH_SPACE_KEYS entry present). Params not listed
    here (including any TRANSFORMER_FAMILY_MODELS-only keys sharing this
    same search_space) are simply never passed to epoch_vae, which falls
    back to its own defaults for them."""
    return {
        name: suggest_from_spec(trial, name, search_space[name])
        for name in _VAE_TRAIN_SEARCH_SPACE_KEYS
        if name in search_space
    }


def derive_vae_arch_cfg(resolved: dict,
                         default_latent_dim: int,
                         n_encoder_layers: int,
                         n_decoder_layers: int) -> dict:
    """Pure function version of the derivation at the heart of
    suggest_arch_cfg: given already-*resolved* (concrete, not
    Optuna-suggested) values for any subset of {"latent_dim",
    "encoder_width_factor", "decoder_width_factor"}, derive the arch_cfg
    dict of {latent_dim, encoder_dims, decoder_dims} suitable for
    model_factories[model_name](n_genes, config=arch_cfg).

    Factored out of suggest_arch_cfg so the same taper math can be replayed
    from a *finished* trial's params (e.g. tuned_configs.json's
    best_params) without needing a live optuna.Trial -- see
    resolve_model_config, used by downstream scripts such as
    sample_efficiency.py to retrain a previously-tuned model.

    Each of the three keys is independently opt-in: keys are only included
    in the returned dict (and therefore only override the model's own
    `model_factories` default) if present in *resolved*. The *number* of
    layers is never derived here -- it's fixed at n_encoder_layers/
    n_decoder_layers (that model's own default depth); only the width at
    each position is derived, as a geometric taper from latent_dim:
    encoder_dims tapers wider->narrower (input side -> latent), decoder_dims
    mirrors it narrower->wider (latent -> output side).
    """
    arch_cfg = {}

    latent_dim = default_latent_dim
    if "latent_dim" in resolved:
        latent_dim = resolved["latent_dim"]
        arch_cfg["latent_dim"] = latent_dim

    if "encoder_width_factor" in resolved:
        factor = resolved["encoder_width_factor"]
        arch_cfg["encoder_dims"] = [
            round(latent_dim * factor ** (n_encoder_layers - k)) for k in range(n_encoder_layers)
        ]

    if "decoder_width_factor" in resolved:
        factor = resolved["decoder_width_factor"]
        arch_cfg["decoder_dims"] = [
            round(latent_dim * factor ** (k + 1)) for k in range(n_decoder_layers)
        ]

    return arch_cfg


def suggest_arch_cfg(trial: optuna.Trial,
                      search_space: dict,
                      default_latent_dim: int,
                      n_encoder_layers: int,
                      n_decoder_layers: int) -> dict:
    """Suggest the limited architecture search parameters (see module
    docstring) and derive encoder_dims/decoder_dims from them via
    derive_vae_arch_cfg. Each of "latent_dim"/"encoder_width_factor"/
    "decoder_width_factor" is independently opt-in: keys are only included
    in the returned dict (and therefore only override the model's own
    `model_factories` default) if the corresponding search_space entry is
    present.
    """
    resolved = {
        name: suggest_from_spec(trial, name, search_space[name])
        for name in _ARCH_SEARCH_SPACE_KEYS
        if name in search_space
    }
    return derive_vae_arch_cfg(resolved, default_latent_dim, n_encoder_layers, n_decoder_layers)


def suggest_transformer_train_cfg(trial: optuna.Trial, search_space: dict) -> dict:
    """Suggest every epoch_transformer-kwarg parameter listed in
    search_space (i.e. every _TRANSFORMER_TRAIN_SEARCH_SPACE_KEYS entry
    present). Params not listed here are simply never passed to
    epoch_transformer, which falls back to its own defaults for them."""
    return {
        name: suggest_from_spec(trial, name, search_space[name])
        for name in _TRANSFORMER_TRAIN_SEARCH_SPACE_KEYS
        if name in search_space
    }


def suggest_transformer_arch_cfg(trial: optuna.Trial, search_space: dict) -> dict:
    """Suggest ImputationTransformer's architecture parameters (see
    _TRANSFORMER_ARCH_SEARCH_SPACE_KEYS). Unlike the VAE family's
    encoder_width_factor/decoder_width_factor taper, each of these maps
    directly onto one of ImputationTransformer's own __init__ kwargs, so no
    derivation is needed -- the returned dict is passed straight through to
    model_factories['ImputationTransformer'] as `config`, overriding
    IMPUTATION_TRANSFORMER_CONFIG's default for any key present here."""
    return {
        name: suggest_from_spec(trial, name, search_space[name])
        for name in _TRANSFORMER_ARCH_SEARCH_SPACE_KEYS
        if name in search_space
    }


def resolve_model_config(model_name: str, params: dict, n_genes: int) -> tuple[dict, dict, float]:
    """Split a *finished* trial's flat params dict (e.g. one entry of
    tuned_configs.json's {"best_params": {...}}) back into the pieces
    needed to reconstruct and retrain that model outside of an active
    Optuna trial:

        arch_cfg, train_kwargs, lr = resolve_model_config(model_name, params, n_genes)
        model = model_factories[model_name](n_genes, config=arch_cfg).to(device)
        opt   = optim.AdamW(model.parameters(), lr=lr)
        epoch_vae(model, loader, opt, mask_fraction=..., beta=..., **train_kwargs)   # VAE_FAMILY_MODELS
        epoch_transformer(model, loader, bin_edges, opt, mask_fraction=..., **train_kwargs)  # TRANSFORMER_FAMILY_MODELS

    `mask_fraction` (both families) and `beta` (VAE_FAMILY_MODELS only) are
    deliberately never part of the returned train_kwargs (neither was ever
    part of `params` either -- see the module docstring's "mask_fraction is
    deliberately not part of the search space" and "beta is fixed, not
    tuned" notes): callers must supply both themselves, matching whatever
    fixed `mask_fraction`/`beta` was used for the tuning run that produced
    `params`.

    Used directly by scripts that replay tuned configs outside of tuning.py
    itself (e.g. sample_efficiency.py's per-model-family evaluation sweeps).
    """
    if model_name not in ALL_TUNABLE_MODELS:
        raise ValueError(
            f"resolve_model_config: unsupported model_name {model_name!r}. "
            f"Supported: {ALL_TUNABLE_MODELS}"
        )

    if model_name in TRANSFORMER_FAMILY_MODELS:
        arch_cfg = {k: params[k] for k in _TRANSFORMER_ARCH_SEARCH_SPACE_KEYS if k in params}
        train_kwargs = {k: params[k] for k in _TRANSFORMER_TRAIN_SEARCH_SPACE_KEYS if k in params}
    else:
        # Cheap, data-free probe instance (default config, never trained) just to
        # read off this model's own default latent_dim and fixed layer counts --
        # same technique run_study_for_model uses to seed suggest_arch_cfg.
        probe = model_factories[model_name](n_genes)
        default_latent_dim = probe.latent_dim
        n_encoder_layers = len(probe.encoder_dims)
        n_decoder_layers = len(probe.decoder_dims)
        del probe

        arch_raw = {k: params[k] for k in _ARCH_SEARCH_SPACE_KEYS if k in params}
        arch_cfg = derive_vae_arch_cfg(arch_raw, default_latent_dim, n_encoder_layers, n_decoder_layers)
        train_kwargs = {k: params[k] for k in _VAE_TRAIN_SEARCH_SPACE_KEYS if k in params}

    lr = train_kwargs.pop("lr", 3e-4)
    return arch_cfg, train_kwargs, lr


def retrain(model_name: str,
            params: dict,
            n_genes: int,
            loader_train,
            loader_test,
            mask_fraction: float,
            max_epochs: int,
            n_copies: int = 1,
            checkpoint_dir: str | None = None,
            device=device,
            beta: float = 1e-4) -> list:
    """Train `n_copies` fresh, independently-initialized instances of
    `model_name` on the *full* loader_train/loader_test data, using the
    architecture + training-loop hyperparameters in `params` (typically a
    finished tuning run's tuned_configs.json[model_name]["best_params"], or
    a checkpoint's saved "params" -- see _save_if_best_checkpoint/
    resolve_model_config).

    Unlike sample_efficiency.py's per-train_size sweep (which retrains on
    deliberately *subsetted* data to measure sample efficiency), this is the
    general-purpose way to get *additional* independent models out of a
    tuning run's chosen hyperparameters -- e.g. to build an ensemble, or to
    check run-to-run variance of a given architecture/training-loop config.

    VampPrior-family models (VAMP_FAMILY_MODELS) have their pseudo-inputs
    initialized from a real batch of loader_train's data (type-1 NaNs
    replaced via `_random_fill`'s per-gene N(mean, std) draw, same as
    everywhere else in this file real data is fed to a model -- see
    AGENTS.md's "Known Issues"), matching run_trial -- see models.py's
    `_make_vamp_vae` docstring.

    Returns a list of `n_copies` trained (eval-mode) model instances, in the
    same order they were trained. If `checkpoint_dir` is given, each copy is
    also saved to {checkpoint_dir}/{model_name}_retrain_{i}.pt in the same
    {'model_type', 'state_dict', 'config'} format vae-test.py's load_model()
    expects (plus 'test_loss', 'imputation_mse_mean', 'imputation_corr_mean',
    'imputation_corr_per_gene_mean', 'imputation_corr_shuffled_mean',
    'imputation_corr_gap_mean' and 'params' for provenance -- see
    models.evaluate_imputation()'s docstring for what the per-gene/shuffled/
    gap fields mean and why they matter for telling genuine per-sample
    imputation apart from output-distribution matching; unlike run_trial's
    hot per-epoch objective evaluation, this is a one-off post-hoc call per
    copy so the extra shuffle_control forward pass is cheap here).

    `mask_fraction` and `max_epochs` are passed explicitly (rather than read
    from `params`) for the same reason tuning.py never tunes mask_fraction
    itself: it's a fixed problem-difficulty knob, not part of the tuned
    config, and the tuning run's own `max_epochs` may not be the epoch
    budget you want for a final/production training run.

    `beta` is likewise passed explicitly rather than read from `params`:
    it's fixed (not tuned) in tuning.py too -- see tuning.py's module
    docstring's "beta is fixed, not tuned" -- so it was never part of
    `params` to begin with. Defaults to 1e-4, matching tuning.py's own
    default; pass the same value used for the tuning run that produced
    `params` if it was overridden there.

    Example (from a notebook, after a tuning run)::

        import json, torch
        from torch.utils.data import DataLoader
        import tuning

        problem = torch.load("test_problem_3.pt")
        n_genes = problem["n_genes"]
        n_cells_test = problem["gene_reads_test"].shape[0]
        loader_train = DataLoader(problem["gene_reads_train"], batch_size=256, shuffle=True)
        # Cap the test batch instead of using the whole test set as one batch --
        # for ImputationTransformer especially, a single giant batch makes
        # attention memory scale with test-set size (see run()'s eval_batch_size).
        loader_test  = DataLoader(problem["gene_reads_test"],  batch_size=min(256, n_cells_test), shuffle=False)

        tuned  = json.load(open("tuned_configs.json"))
        params = tuned["GeneExpressionVAE"]["best_params"]
        models = tuning.retrain(
            "GeneExpressionVAE", params, n_genes, loader_train, loader_test,
            mask_fraction=0.4, max_epochs=100, n_copies=5,
            checkpoint_dir="checkpoints",
        )
    """
    arch_cfg, train_kwargs, lr = resolve_model_config(model_name, params, n_genes)

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    is_transformer = model_name in TRANSFORMER_FAMILY_MODELS
    trained = []

    # Real-batch pseudo-input init for VampPrior-family models (see
    # VAMP_FAMILY_MODELS / models.py's _make_vamp_vae docstring).
    pseudo_init_samples = None
    if model_name in VAMP_FAMILY_MODELS:
        for x in loader_train:
            pseudo_init_samples = _random_fill(x, torch.isnan(x)).to(device)
            break

    for i in range(n_copies):
        factory_kwargs = {"pseudo_init_samples": pseudo_init_samples} if model_name in VAMP_FAMILY_MODELS else {}
        model = model_factories[model_name](n_genes, config=arch_cfg, **factory_kwargs).to(device)
        opt = optim.AdamW(model.parameters(), lr=lr)

        if is_transformer:
            bin_edges = make_log_bin_edges(model.n_bins, max_val=8.5)
            model.bin_edges = bin_edges  # impute()/impute_transformer() read this attribute
            epoch_bar = tqdm(range(max_epochs), desc=f"{model_name} retrain {i + 1}/{n_copies}", leave=False)
            for _ in epoch_bar:
                epoch_transformer(model, loader_train, bin_edges, opt,
                                   mask_fraction=mask_fraction, device=device, **train_kwargs)
            model.eval()
            test_loss, *_ = epoch_transformer(model, loader_test, bin_edges, None,
                                               mask_fraction=mask_fraction, device=device, **train_kwargs)
        else:
            epoch_bar = tqdm(range(max_epochs), desc=f"{model_name} retrain {i + 1}/{n_copies}", leave=False)
            for _ in epoch_bar:
                epoch_vae(model, loader_train, opt, mask_fraction=mask_fraction, beta=beta, **train_kwargs)
            model.eval()
            test_loss, *_ = epoch_vae(model, loader_test, None, mask_fraction=mask_fraction, beta=beta, **train_kwargs)

        test_loss = float(test_loss)
        # shuffle_control=True: post-hoc diagnostic, run once per retrained
        # copy (not every eval_every epochs x every trial like run_trial's
        # objective), so the extra forward pass is cheap here -- see
        # models.evaluate_imputation()'s docstring for what
        # imputation_corr_per_gene_mean / imputation_corr_gap_mean measure.
        imp_metrics = evaluate_imputation(model, loader_test, mask_fraction, n_draws=3,
                                           device=device, shuffle_control=True)
        print(f"  retrain {i + 1}/{n_copies} of {model_name}: test_loss={test_loss:.4f} "
              f"(diagnostic only, not cross-model-comparable) "
              f"imputation_mse={imp_metrics['imputation_mse_mean']:.4f} "
              f"imputation_corr={imp_metrics['imputation_corr_mean']:.4f} "
              f"imputation_corr_per_gene={imp_metrics['imputation_corr_per_gene_mean']:.4f} "
              f"imputation_corr_gap={imp_metrics['imputation_corr_gap_mean']:.4f}")

        if checkpoint_dir is not None:
            path = Path(checkpoint_dir) / f"{model_name}_retrain_{i}.pt"
            torch.save({
                "model_type":                     model_name,
                "state_dict":                     model.state_dict(),
                "config":                         model.config,
                "test_loss":                      test_loss,
                "imputation_mse_mean":            imp_metrics["imputation_mse_mean"],
                "imputation_corr_mean":           imp_metrics["imputation_corr_mean"],
                "imputation_corr_per_gene_mean":  imp_metrics["imputation_corr_per_gene_mean"],
                "imputation_corr_shuffled_mean":  imp_metrics["imputation_corr_shuffled_mean"],
                "imputation_corr_gap_mean":       imp_metrics["imputation_corr_gap_mean"],
                "params":                         params,
            }, path)

        trained.append(model)

    return trained


def _save_if_best_checkpoint(model, model_name: str, objective_value: float,
                              extra_info: dict, trial: optuna.Trial, checkpoint_dir: str) -> None:
    """If *objective_value* (imputation_mse_mean -- see module docstring's
    "The objective being minimized") improves on whatever is currently
    recorded in {checkpoint_dir}/{model_name}_best.pt (or that file doesn't
    exist yet), overwrite it with *model*'s current weights.

    Deliberately self-contained: the "current best" is read straight back
    out of the checkpoint file itself rather than from Optuna's study state,
    so this stays correct even across a resumed study whose checkpoint_dir
    wasn't persisted between sessions (e.g. a fresh Colab runtime reusing the
    same sqlite storage) -- in that case the first improving trial of the new
    session simply becomes the new saved baseline.

    The saved dict is exactly the {'model_type', 'state_dict', 'config'}
    format vae-test.py's load_model() expects, plus 'imputation_mse_mean',
    whatever's in *extra_info* (e.g. 'test_loss', 'imputation_corr_mean' --
    diagnostic only, see module docstring), 'params' and 'trial_number' for
    provenance -- 'params' is also directly reusable as the `params`
    argument to resolve_model_config()/retrain() if you want to train
    further independent copies of this exact configuration.
    """
    objective_value = float(objective_value)
    if not math.isfinite(objective_value):
        return

    path = Path(checkpoint_dir) / f"{model_name}_best.pt"
    if path.exists():
        try:
            prev_value = torch.load(path, map_location="cpu").get("imputation_mse_mean", float("inf"))
        except Exception:
            prev_value = float("inf")
        if objective_value >= prev_value:
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type":          model_name,
        "state_dict":          model.state_dict(),
        "config":              model.config,
        "imputation_mse_mean": objective_value,
        **extra_info,
        "params":              trial.params,
        "trial_number":        trial.number,
    }, path)
    tqdm.write(f"  [checkpoint] new best {model_name}: imputation_mse_mean={objective_value:.4f} -> {path}")


def run_trial(trial: optuna.Trial,
              model_name: str,
              n_genes: int,
              loader_train,
              loader_test,
              search_space: dict,
              max_epochs: int,
              eval_every: int,
              mask_fraction: float,
              beta: float,
              eval_mask_fraction: float,
              n_eval_mask_draws: int,
              default_latent_dim: int,
              n_encoder_layers: int,
              n_decoder_layers: int,
              min_layer_width: int,
              max_layer_width: int,
              checkpoint_dir: str,
              pseudo_init_data: torch.Tensor | None = None) -> float:
    """Build a fresh model, train it for max_epochs, report intermediate
    imputation_mse_mean for pruning (see module docstring's "The objective
    being minimized"), and return the final imputation_mse_mean.

    beta: fixed (not tuned) KL weight, applied identically to every trial
    -- see module docstring's "beta is fixed, not tuned".

    pseudo_init_data: a real (possibly NaN-containing) batch of training
    data, used to initialize VampPrior-family models' (VAMP_FAMILY_MODELS)
    pseudo-inputs instead of random noise -- ignored for other model
    families. Type-1 NaNs are replaced via `_random_fill` (per-gene
    N(mean, std), not a fixed sentinel -- see AGENTS.md's "Known Issues")
    before being handed to the model factory. See models.py's
    `_make_vamp_vae`/`_make_deq_vamp_vae` docstrings.
    """
    cfg = suggest_train_cfg(trial, search_space)
    lr = cfg.pop("lr", 3e-4)
    cfg["mask_fraction"] = mask_fraction
    cfg["beta"] = beta

    arch_cfg = suggest_arch_cfg(trial, search_space, default_latent_dim, n_encoder_layers, n_decoder_layers)

    # A (latent_dim, width_factor) combination can produce unreasonably
    # tiny or huge layers; catch that here and prune before wasting any
    # training compute rather than letting it run (or crash on a
    # non-positive width).
    derived_widths = arch_cfg.get("encoder_dims", []) + arch_cfg.get("decoder_dims", [])
    if derived_widths and (min(derived_widths) < max(1, min_layer_width) or max(derived_widths) > max_layer_width):
        trial.set_user_attr("arch_cfg", arch_cfg)
        raise optuna.TrialPruned(
            f"derived layer widths {derived_widths} outside "
            f"[{min_layer_width}, {max_layer_width}]"
        )
    if arch_cfg:
        trial.set_user_attr("arch_cfg", arch_cfg)

    factory_kwargs = {}
    if model_name in VAMP_FAMILY_MODELS and pseudo_init_data is not None:
        factory_kwargs["pseudo_init_samples"] = _random_fill(pseudo_init_data, torch.isnan(pseudo_init_data)).to(device)

    model = model_factories[model_name](n_genes, config=arch_cfg, **factory_kwargs).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)

    test_loss = float("inf")  # diagnostic only, see module docstring -- NOT the objective
    imputation_mse = float("inf")
    imp_metrics = {"imputation_corr_mean": float("nan")}
    epoch_bar = tqdm(range(max_epochs), desc=f"{model_name} #{trial.number}", leave=False)
    for epoch in epoch_bar:
        epoch_vae(model, loader_train, opt, **cfg)

        if epoch % eval_every == 0:
            test_loss, *_ = epoch_vae(model, loader_test, None, **cfg)
            if not math.isfinite(test_loss):
                raise optuna.TrialPruned()

            imp_metrics = evaluate_imputation(model, loader_test, eval_mask_fraction,
                                               n_eval_mask_draws, device=device)
            imputation_mse = imp_metrics["imputation_mse_mean"]
            if not math.isfinite(imputation_mse):
                raise optuna.TrialPruned()

            try:
                best = trial.study.best_value
            except ValueError:
                best = None
            epoch_bar.set_postfix(
                imp_mse=f"{imputation_mse:.4f}",
                test_loss=f"{test_loss:.4f}",
                best=f"{best:.4f}" if best is not None else "n/a",
            )

            trial.report(imputation_mse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    vamp_diag = collect_vamp_diagnostics(model)
    if vamp_diag["effective_K_population"] is not None:
        trial.set_user_attr("effective_K_population", vamp_diag["effective_K_population"])

    trial.set_user_attr("test_loss", test_loss)
    trial.set_user_attr("imputation_corr_mean", imp_metrics["imputation_corr_mean"])

    _save_if_best_checkpoint(
        model, model_name, imputation_mse,
        {"test_loss": test_loss, "imputation_corr_mean": imp_metrics["imputation_corr_mean"]},
        trial, checkpoint_dir,
    )

    return imputation_mse


def _prune(trial: optuna.Trial, reason: str, message: str = "") -> None:
    """Raise optuna.TrialPruned after tagging *why* -- trial.user_attrs
    ["prune_reason"] lets post-hoc study analysis (or trial_summary_callback,
    live) distinguish structural prunes (d_model_not_divisible_by_n_heads,
    d_model_out_of_bounds, vram_budget_exceeded -- all before any training)
    from training-time prunes (non_finite_test_loss, non_finite_imputation_mse,
    pruner_median_stopping), rather than lumping every PRUNED trial together."""
    trial.set_user_attr("prune_reason", reason)
    raise optuna.TrialPruned(message or reason)


def run_trial_transformer(trial: optuna.Trial,
                           model_name: str,
                           n_genes: int,
                           loader_train,
                           loader_test,
                           search_space: dict,
                           max_epochs: int,
                           eval_every: int,
                           mask_fraction: float,
                           eval_mask_fraction: float,
                           n_eval_mask_draws: int,
                           min_layer_width: int,
                           max_layer_width: int,
                           checkpoint_dir: str,
                           batch_size: int | None = None,
                           eval_batch_size: int | None = None,
                           vram_budget_bytes: float | None = None,
                           vram_overhead_factor: float = 0.65) -> float:
    """ImputationTransformer counterpart of run_trial: build a fresh model,
    train it for max_epochs via epoch_transformer, report intermediate
    imputation_mse_mean for pruning (see module docstring's "The objective
    being minimized"), and return the final imputation_mse_mean.

    batch_size/eval_batch_size/vram_budget_bytes/vram_overhead_factor drive
    the pre-training VRAM guard (see module docstring's "VRAM guard" section)
    -- the guard is a no-op if vram_budget_bytes is None (e.g. non-CUDA
    device and no explicit vram_budget_gb override, see
    _resolve_vram_budget_bytes)."""
    cfg = suggest_transformer_train_cfg(trial, search_space)
    lr = cfg.pop("lr", 3e-4)
    lambda_conf = cfg.pop("lambda_conf", 0.10)

    arch_cfg = suggest_transformer_arch_cfg(trial, search_space)

    # d_model must be divisible by n_heads (nn.TransformerEncoderLayer
    # requirement); an incompatible combination is pruned before any
    # training rather than crashing mid-trial.
    resolved_d_model  = arch_cfg.get("d_model",  IMPUTATION_TRANSFORMER_CONFIG["d_model"])
    resolved_n_heads  = arch_cfg.get("n_heads",  IMPUTATION_TRANSFORMER_CONFIG["n_heads"])
    resolved_n_layers = arch_cfg.get("n_layers", IMPUTATION_TRANSFORMER_CONFIG["n_layers"])
    if arch_cfg:
        trial.set_user_attr("arch_cfg", arch_cfg)
    if resolved_d_model % resolved_n_heads != 0:
        _prune(trial, "d_model_not_divisible_by_n_heads",
               f"d_model={resolved_d_model} not divisible by n_heads={resolved_n_heads}")
    # d_model is the transformer's analogue of the VAE family's layer
    # width; enforce the same [min_layer_width, max_layer_width] guard.
    if "d_model" in arch_cfg and not (min_layer_width <= resolved_d_model <= max_layer_width):
        _prune(trial, "d_model_out_of_bounds",
               f"d_model={resolved_d_model} outside [{min_layer_width}, {max_layer_width}]")

    # VRAM guard: estimate this trial's peak VRAM (train and eval, take the
    # max) *before* touching the GPU, and prune it the same way as the
    # structural checks above if it's over budget -- see module docstring's
    # "VRAM guard" section. No-op if vram_budget_bytes is None.
    if vram_budget_bytes is not None:
        est_train_bytes = estimate_transformer_train_bytes(
            batch_size, n_genes, resolved_n_heads, resolved_d_model, resolved_n_layers)
        est_eval_bytes = estimate_transformer_eval_bytes(
            eval_batch_size, n_genes, resolved_n_heads, resolved_d_model, resolved_n_layers)
        est_bytes = max(est_train_bytes, est_eval_bytes) * vram_overhead_factor
        trial.set_user_attr("estimated_vram_gb", est_bytes / 1e9)
        if est_bytes > vram_budget_bytes:
            _prune(trial, "vram_budget_exceeded",
                   f"estimated VRAM {est_bytes / 1e9:.1f}GB (d_model={resolved_d_model}, "
                   f"n_heads={resolved_n_heads}, n_layers={resolved_n_layers}, n_genes={n_genes}, "
                   f"batch_size={batch_size}) exceeds budget {vram_budget_bytes / 1e9:.1f}GB")

    model = model_factories[model_name](n_genes, config=arch_cfg).to(device)
    bin_edges = make_log_bin_edges(model.n_bins, max_val=8.5)
    model.bin_edges = bin_edges  # impute()/evaluate_imputation() read this attribute
    opt = optim.AdamW(model.parameters(), lr=lr)

    test_loss = float("inf")  # diagnostic only, see module docstring -- NOT the objective
    imputation_mse = float("inf")
    imp_metrics = {"imputation_corr_mean": float("nan")}
    epoch_bar = tqdm(range(max_epochs), desc=f"{model_name} #{trial.number}", leave=False)
    for epoch in epoch_bar:
        epoch_transformer(model, loader_train, bin_edges, opt,
                           mask_fraction=mask_fraction, lambda_conf=lambda_conf, device=device)

        if epoch % eval_every == 0:
            test_loss, *_ = epoch_transformer(model, loader_test, bin_edges, None,
                                               mask_fraction=mask_fraction, lambda_conf=lambda_conf, device=device)
            if not math.isfinite(test_loss):
                _prune(trial, "non_finite_test_loss", f"test_loss={test_loss}")

            imp_metrics = evaluate_imputation(model, loader_test, eval_mask_fraction,
                                               n_eval_mask_draws, device=device)
            imputation_mse = imp_metrics["imputation_mse_mean"]
            if not math.isfinite(imputation_mse):
                _prune(trial, "non_finite_imputation_mse", f"imputation_mse={imputation_mse}")

            try:
                best = trial.study.best_value
            except ValueError:
                best = None
            epoch_bar.set_postfix(
                imp_mse=f"{imputation_mse:.4f}",
                test_loss=f"{test_loss:.4f}",
                best=f"{best:.4f}" if best is not None else "n/a",
            )

            trial.report(imputation_mse, epoch)
            if trial.should_prune():
                _prune(trial, "pruner_median_stopping",
                       f"pruned by {type(trial.study.pruner).__name__} at epoch {epoch}")

    trial.set_user_attr("test_loss", test_loss)
    trial.set_user_attr("imputation_corr_mean", imp_metrics["imputation_corr_mean"])

    _save_if_best_checkpoint(
        model, model_name, imputation_mse,
        {"test_loss": test_loss, "imputation_corr_mean": imp_metrics["imputation_corr_mean"]},
        trial, checkpoint_dir,
    )

    return imputation_mse


def trial_summary_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """One concise progress line per finished trial (via tqdm.write, so it
    doesn't clobber any active progress bar). For PRUNED trials, appends
    reason=<prune_reason> when run_trial_transformer's _prune() tagged one
    (see its docstring) -- e.g. to tell how often the vram_budget_exceeded
    guard is firing versus the structural d_model_not_divisible_by_n_heads/
    d_model_out_of_bounds prunes versus training-time prunes, rather than
    lumping every PRUNED trial together under state=PRUNED alone."""
    if trial.datetime_complete and trial.datetime_start:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
    else:
        duration = float("nan")

    try:
        best = study.best_value
    except ValueError:
        best = None

    value_str = f"{trial.value:.4f}" if trial.value is not None else "n/a"
    best_str = f"{best:.4f}" if best is not None else "n/a"
    reason = trial.user_attrs.get("prune_reason")
    reason_str = f"  reason={reason}" if reason else ""
    tqdm.write(
        f"  trial {trial.number:>4d}  state={trial.state.name:<9s}  "
        f"value={value_str:>10s}  best={best_str:>10s}  ({duration:5.1f}s){reason_str}"
    )


def build_sampler(name: str):
    return _SAMPLERS[name]()


def build_pruner(name: str):
    return _PRUNERS[name]()


def run_study_for_model(model_name: str,
                         config: dict,
                         n_genes: int,
                         loader_train,
                         loader_test,
                         pseudo_init_data: torch.Tensor | None = None) -> optuna.Study:
    """pseudo_init_data: a real batch of training data (possibly
    NaN-containing -- run_trial applies `_random_fill` before use, see its
    docstring), forwarded to run_trial for VAMP_FAMILY_MODELS' pseudo-input
    initialization -- ignored for other model families (including all of
    TRANSFORMER_FAMILY_MODELS)."""
    if model_name in TRANSFORMER_FAMILY_MODELS:
        objective = lambda trial: run_trial_transformer(
            trial, model_name, n_genes, loader_train, loader_test,
            config["search_space"], config["max_epochs"], config["eval_every"],
            config["mask_fraction"], config["eval_mask_fraction"], config["n_eval_mask_draws"],
            config["min_layer_width"], config["max_layer_width"],
            config["checkpoint_dir"],
            batch_size=config["batch_size"], eval_batch_size=config["eval_batch_size"],
            vram_budget_bytes=config["vram_budget_bytes"], vram_overhead_factor=config["vram_overhead_factor"],
        )
    else:
        # Cheap, data-free probe instance (default config, never trained) just to
        # read off this model's own default latent_dim and fixed layer counts --
        # the basis for the limited architecture search in run_trial (see
        # suggest_arch_cfg / module docstring).
        probe = model_factories[model_name](n_genes)
        default_latent_dim = probe.latent_dim
        n_encoder_layers = len(probe.encoder_dims)
        n_decoder_layers = len(probe.decoder_dims)
        del probe

        objective = lambda trial: run_trial(
            trial, model_name, n_genes, loader_train, loader_test,
            config["search_space"], config["max_epochs"], config["eval_every"],
            config["mask_fraction"], config["beta"],
            config["eval_mask_fraction"], config["n_eval_mask_draws"],
            default_latent_dim, n_encoder_layers, n_decoder_layers,
            config["min_layer_width"], config["max_layer_width"],
            config["checkpoint_dir"],
            pseudo_init_data=pseudo_init_data,
        )

    study = optuna.create_study(
        study_name=f"tune_{model_name}",
        storage=config["storage"],
        load_if_exists=True,
        direction="minimize",
        sampler=build_sampler(config["sampler"]),
        pruner=build_pruner(config["pruner"]),
    )

    print(f"\n=== Tuning {model_name} ({config['n_trials']} trials) ===")
    start = time.monotonic()

    study.optimize(
        objective,
        n_trials=config["n_trials"],
        show_progress_bar=True,
        callbacks=[trial_summary_callback],
    )

    elapsed = time.monotonic() - start
    print(
        f"=== {model_name} done in {elapsed / 60:.1f}m -- "
        f"best_value={study.best_value:.4f} best_params={study.best_params} ==="
    )

    return study


def run(config_path: str) -> dict:
    """Plain entry point, safe to call directly from a notebook cell (no
    argparse/CLI dependency):

        import tuning
        studies = tuning.run("tuning_config.json")
        best = studies["VampPriorVAE"].best_trial

        import optuna.visualization as vis
        vis.plot_optimization_history(studies["VampPriorVAE"]).show()

    Returns {model_name: optuna.Study}, one entry per model in
    config["models"]. Each study is also persisted in config["storage"]
    (the sqlite db, by default) under study_name f"tune_{model_name}", so it
    can be reloaded/resumed later via optuna.load_study() even without
    calling this function again. A plain-JSON summary
    ({model_name: {"best_value": float, "best_params": dict, "best_test_loss":
    float, "best_imputation_corr": float}, ..., "_meta": {...}}) is written
    to config["output"] (default tuned_configs.json) for convenience --
    best_value is imputation_mse_mean (the tuning objective, see module
    docstring's "The objective being minimized"); best_test_loss/
    best_imputation_corr are read back from the best trial's user_attrs
    (diagnostic only, not cross-model-comparable for test_loss -- see
    module docstring); "_meta" is provenance consumed by
    sample_efficiency.py to warn about batch_size/mask_fraction mismatches
    between tuning and replay (see sample_efficiency.py's module
    docstring) -- not a model name.
    """
    config = load_config(config_path)

    # Optuna's default per-trial logging dumps the full param dict and gets
    # noisy over many trials; we rely on trial_summary_callback instead.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Loading problem from {config['problem_fn']!r} ...")
    loaded = torch.load(config["problem_fn"])
    n_genes = loaded["n_genes"]
    n_cells = loaded["n_cells"]
    batch_size = config["batch_size"]
    mask_fraction = config["mask_fraction"]

    # eval_batch_size (default: batch_size, see _CONFIG_DEFAULTS/load_config) caps the
    # test-set loader's batch size instead of feeding the whole test set through in one
    # batch -- for ImputationTransformer, a single giant batch makes attention memory
    # (O(batch * n_heads * n_genes^2)) scale with test-set size rather than architecture,
    # which is what actually drove the 15-40GB VRAM usage observed at n_genes=200.
    # epoch_transformer/epoch_vae/evaluate_imputation already loop over and pool/average
    # across loader_test's batches, so this only changes memory use, not results.
    n_cells_test = loaded["gene_reads_test"].shape[0]
    eval_batch_size = min(config["eval_batch_size"], n_cells_test)
    # Written back so run_study_for_model's VRAM guard (which reads
    # config["eval_batch_size"]) sees the actual capped runtime batch size,
    # not the pre-cap configured value.
    config["eval_batch_size"] = eval_batch_size

    loader_train = DataLoader(loaded["gene_reads_train"], batch_size=batch_size,     shuffle=True )
    loader_test  = DataLoader(loaded["gene_reads_test"],  batch_size=eval_batch_size, shuffle=False)

    # Real training data used to initialize VAMP_FAMILY_MODELS' pseudo-inputs
    # (see run_trial/run_study_for_model) -- ignored for other families.
    # The full tensor (not just one minibatch) is used since it's only ever
    # indexed into (torch.randint sampling in VampPriorVAE.__init__), never
    # trained on directly, so there's no cost to it being large/representative.
    # (run_trial applies _random_fill to replace type-1 NaNs before use.)
    pseudo_init_data = loaded["gene_reads_train"]

    print(
        f"n_genes={n_genes}  n_cells={n_cells}  batch_size={batch_size}  "
        f"mask_fraction={mask_fraction}  beta={config['beta']}  "
        f"eval_mask_fraction={config['eval_mask_fraction']}  "
        f"n_eval_mask_draws={config['n_eval_mask_draws']}  device={device}"
    )
    if config["vram_budget_bytes"] is not None:
        print(
            f"ImputationTransformer VRAM guard: budget={config['vram_budget_bytes'] / 1e9:.1f}GB "
            f"(vram_budget_gb={config['vram_budget_gb']!r}, vram_budget_fraction={config['vram_budget_fraction']}, "
            f"vram_overhead_factor={config['vram_overhead_factor']})"
        )
    elif set(config["models"]) & set(TRANSFORMER_FAMILY_MODELS):
        print("ImputationTransformer VRAM guard: disabled (no CUDA device and no vram_budget_gb override)")

    studies = {}
    for model_name in config["models"]:
        studies[model_name] = run_study_for_model(
            model_name, config, n_genes, loader_train, loader_test,
            pseudo_init_data=pseudo_init_data,
        )

    summary = {}
    for model_name, study in studies.items():
        user_attrs = study.best_trial.user_attrs
        summary[model_name] = {
            "best_value":           study.best_value,
            "best_params":          study.best_params,
            "best_test_loss":       user_attrs.get("test_loss"),
            "best_imputation_corr": user_attrs.get("imputation_corr_mean"),
        }
        if "effective_K_population" in user_attrs:
            summary[model_name]["best_effective_K_population"] = user_attrs["effective_K_population"]

    summary["_meta"] = {
        "batch_size":         batch_size,
        "mask_fraction":      mask_fraction,
        "beta":               config["beta"],
        "max_epochs":         config["max_epochs"],
        "eval_mask_fraction": config["eval_mask_fraction"],
        "n_eval_mask_draws":  config["n_eval_mask_draws"],
    }

    out_path = Path(config["output"])
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote tuned config summary to {out_path}")

    return studies


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="tuning_config.json",
        help="Path to the JSON tuning config (see tuning_config.example.json)",
    )
    args = parser.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    main()
