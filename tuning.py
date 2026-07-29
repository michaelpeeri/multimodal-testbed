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
listed in "search_space" (learning rate, beta, gamma, min_free_bits,
lambda_entropy, lambda_vamp_repulsion, lambda_vamp_coverage -- any subset
of epoch_vae's kwargs other than mask_fraction). For TRANSFORMER_FAMILY_MODELS
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

The objective being minimized is the held-out test loss (`epoch_vae`'s for
VAE_FAMILY_MODELS, `epoch_transformer`'s for TRANSFORMER_FAMILY_MODELS --
both return total loss as their first element). Trials report intermediate
test loss every `eval_every` epochs so the configured pruner can stop
clearly-bad trials early.

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
                    epoch_vae() kwarg). For TRANSFORMER_FAMILY_MODELS: "lr"
                    and "lambda_conf" control epoch_transformer's kwargs;
                    "d_model"/"n_heads"/"n_layers"/"n_bins"/"n_conf_bins"/
                    "d_gene"/"d_count"/"d_conf" (type "int") and "dropout"
                    (type "float") control the architecture search described
                    above. A key not applicable to a given model's family is
                    simply ignored for that model (left at its own default),
                    so VAE and transformer search_space keys may coexist in
                    the same dict when tuning both families together.
                    "mask_fraction" may not be listed here -- see the
                    top-level "mask_fraction" config key instead.
    mask_fraction : float, default 0.1 -- fixed (not tuned) fraction of
                    observed genes masked for training/eval, forwarded as
                    epoch_vae's/epoch_transformer's mask_fraction kwarg for
                    every trial.
    min_layer_width : int, default 4 -- lower bound on every derived
                    encoder/decoder layer width (VAE_FAMILY_MODELS) or on
                    "d_model" (TRANSFORMER_FAMILY_MODELS); trials that fall
                    outside [min_layer_width, max_layer_width] are pruned
                    before training. Only enforced for the roles actually
                    driven by encoder_width_factor/decoder_width_factor/
                    d_model.
    max_layer_width : int, default 4096 -- upper bound, see min_layer_width.
    storage       : str, default "sqlite:///tuning.db"
    batch_size    : int, default 256 (train loader only; test loader always
                    uses the full test set as one batch, matching n_cells)
    n_trials      : int, default 50 (per model)
    max_epochs    : int, default 50
    eval_every    : int, default 1 -- epoch cadence for both the held-out
                    eval used for pruning/objective and the progress-bar
                    postfix update
    sampler       : "tpe" (default; only option for now)
    pruner        : "median" (default; only option for now)
    output        : str, default "tuned_configs.json" -- where the final
                    {model_name: {best_value, best_params}} summary is written
                    (see `run()` below for the richer in-memory return value)
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
best test loss seen so far for that model -- so at any point during, or at
the end of, a tuning run you already have the literal best-performing
model on disk, no retraining required. The saved dict is exactly the
{'model_type', 'state_dict', 'config'} format vae-test.py's load_model()
expects (plus 'test_loss', 'params', 'trial_number' for provenance)::

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
    model_factories, epoch_vae, VampPriorVAE, device,
    epoch_transformer, make_log_bin_edges,
    IMPUTATION_TRANSFORMER_CONFIG,
)


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
# must match epoch_vae's kwargs (other than mask_fraction, which is fixed).
# Whitelisted (rather than "everything not an arch key") so that a
# search_space shared with TRANSFORMER_FAMILY_MODELS in the same config (e.g.
# tuning ImputationTransformer alongside VampPriorVAE) doesn't leak
# transformer-only keys like "lambda_conf" into epoch_vae() as unexpected
# kwargs.
_VAE_TRAIN_SEARCH_SPACE_KEYS = {
    "lr", "beta", "gamma", "min_free_bits",
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
    "n_trials":        50,
    "max_epochs":      50,
    "eval_every":      1,
    "sampler":         "tpe",
    "pruner":          "median",
    "output":          "tuned_configs.json",
    "mask_fraction":   0.1,
    "min_layer_width": 4,
    "max_layer_width": 4096,
    "checkpoint_dir":  "checkpoints",
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
        epoch_vae(model, loader, opt, mask_fraction=..., **train_kwargs)   # VAE_FAMILY_MODELS
        epoch_transformer(model, loader, bin_edges, opt, mask_fraction=..., **train_kwargs)  # TRANSFORMER_FAMILY_MODELS

    `mask_fraction` is deliberately never part of the returned train_kwargs
    (it was never part of `params` either -- see the module docstring's
    "mask_fraction is deliberately not part of the search space" note):
    callers must supply it themselves, matching whatever fixed
    `mask_fraction` was used for the tuning run that produced `params`.

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
            device=device) -> list:
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

    Returns a list of `n_copies` trained (eval-mode) model instances, in the
    same order they were trained. If `checkpoint_dir` is given, each copy is
    also saved to {checkpoint_dir}/{model_name}_retrain_{i}.pt in the same
    {'model_type', 'state_dict', 'config'} format vae-test.py's load_model()
    expects (plus 'test_loss' and 'params' for provenance).

    `mask_fraction` and `max_epochs` are passed explicitly (rather than read
    from `params`) for the same reason tuning.py never tunes mask_fraction
    itself: it's a fixed problem-difficulty knob, not part of the tuned
    config, and the tuning run's own `max_epochs` may not be the epoch
    budget you want for a final/production training run.

    Example (from a notebook, after a tuning run)::

        import json, torch
        from torch.utils.data import DataLoader
        import tuning

        problem = torch.load("test_problem_3.pt")
        n_genes, n_cells = problem["n_genes"], problem["n_cells"]
        loader_train = DataLoader(problem["gene_reads_train"], batch_size=256, shuffle=True)
        loader_test  = DataLoader(problem["gene_reads_test"],  batch_size=n_cells, shuffle=False)

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

    for i in range(n_copies):
        model = model_factories[model_name](n_genes, config=arch_cfg).to(device)
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
                epoch_vae(model, loader_train, opt, mask_fraction=mask_fraction, **train_kwargs)
            model.eval()
            test_loss, *_ = epoch_vae(model, loader_test, None, mask_fraction=mask_fraction, **train_kwargs)

        test_loss = float(test_loss)
        print(f"  retrain {i + 1}/{n_copies} of {model_name}: test_loss={test_loss:.4f}")

        if checkpoint_dir is not None:
            path = Path(checkpoint_dir) / f"{model_name}_retrain_{i}.pt"
            torch.save({
                "model_type": model_name,
                "state_dict": model.state_dict(),
                "config":     model.config,
                "test_loss":  test_loss,
                "params":     params,
            }, path)

        trained.append(model)

    return trained


def _save_if_best_checkpoint(model, model_name: str, test_loss: float,
                              trial: optuna.Trial, checkpoint_dir: str) -> None:
    """If *test_loss* improves on whatever is currently recorded in
    {checkpoint_dir}/{model_name}_best.pt (or that file doesn't exist yet),
    overwrite it with *model*'s current weights.

    Deliberately self-contained: the "current best" is read straight back
    out of the checkpoint file itself rather than from Optuna's study state,
    so this stays correct even across a resumed study whose checkpoint_dir
    wasn't persisted between sessions (e.g. a fresh Colab runtime reusing the
    same sqlite storage) -- in that case the first improving trial of the new
    session simply becomes the new saved baseline.

    The saved dict is exactly the {'model_type', 'state_dict', 'config'}
    format vae-test.py's load_model() expects, plus 'test_loss', 'params'
    and 'trial_number' for provenance -- 'params' is also directly reusable
    as the `params` argument to resolve_model_config()/retrain() if you want
    to train further independent copies of this exact configuration.
    """
    test_loss = float(test_loss)
    if not math.isfinite(test_loss):
        return

    path = Path(checkpoint_dir) / f"{model_name}_best.pt"
    if path.exists():
        try:
            prev_loss = torch.load(path, map_location="cpu").get("test_loss", float("inf"))
        except Exception:
            prev_loss = float("inf")
        if test_loss >= prev_loss:
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type":   model_name,
        "state_dict":   model.state_dict(),
        "config":       model.config,
        "test_loss":    test_loss,
        "params":       trial.params,
        "trial_number": trial.number,
    }, path)
    tqdm.write(f"  [checkpoint] new best {model_name}: test_loss={test_loss:.4f} -> {path}")


def run_trial(trial: optuna.Trial,
              model_name: str,
              n_genes: int,
              loader_train,
              loader_test,
              search_space: dict,
              max_epochs: int,
              eval_every: int,
              mask_fraction: float,
              default_latent_dim: int,
              n_encoder_layers: int,
              n_decoder_layers: int,
              min_layer_width: int,
              max_layer_width: int,
              checkpoint_dir: str) -> float:
    """Build a fresh model, train it for max_epochs, report intermediate
    held-out test loss for pruning, and return the final test loss."""
    cfg = suggest_train_cfg(trial, search_space)
    lr = cfg.pop("lr", 3e-4)
    cfg["mask_fraction"] = mask_fraction

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

    model = model_factories[model_name](n_genes, config=arch_cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)

    test_loss = float("inf")
    epoch_bar = tqdm(range(max_epochs), desc=f"{model_name} #{trial.number}", leave=False)
    for epoch in epoch_bar:
        epoch_vae(model, loader_train, opt, **cfg)

        if epoch % eval_every == 0:
            test_loss, *_ = epoch_vae(model, loader_test, None, **cfg)
            if not math.isfinite(test_loss):
                raise optuna.TrialPruned()

            try:
                best = trial.study.best_value
            except ValueError:
                best = None
            epoch_bar.set_postfix(
                test_loss=f"{test_loss:.4f}",
                best=f"{best:.4f}" if best is not None else "n/a",
            )

            trial.report(test_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if isinstance(model, VampPriorVAE):
        trial.set_user_attr("effective_K_population", model._effective_K_population.item())

    _save_if_best_checkpoint(model, model_name, test_loss, trial, checkpoint_dir)

    return test_loss


def run_trial_transformer(trial: optuna.Trial,
                           model_name: str,
                           n_genes: int,
                           loader_train,
                           loader_test,
                           search_space: dict,
                           max_epochs: int,
                           eval_every: int,
                           mask_fraction: float,
                           min_layer_width: int,
                           max_layer_width: int,
                           checkpoint_dir: str) -> float:
    """ImputationTransformer counterpart of run_trial: build a fresh model,
    train it for max_epochs via epoch_transformer, report intermediate
    held-out test loss for pruning, and return the final test loss."""
    cfg = suggest_transformer_train_cfg(trial, search_space)
    lr = cfg.pop("lr", 3e-4)
    lambda_conf = cfg.pop("lambda_conf", 0.10)

    arch_cfg = suggest_transformer_arch_cfg(trial, search_space)

    # d_model must be divisible by n_heads (nn.TransformerEncoderLayer
    # requirement); an incompatible combination is pruned before any
    # training rather than crashing mid-trial.
    resolved_d_model = arch_cfg.get("d_model", IMPUTATION_TRANSFORMER_CONFIG["d_model"])
    resolved_n_heads = arch_cfg.get("n_heads", IMPUTATION_TRANSFORMER_CONFIG["n_heads"])
    if arch_cfg:
        trial.set_user_attr("arch_cfg", arch_cfg)
    if resolved_d_model % resolved_n_heads != 0:
        raise optuna.TrialPruned(
            f"d_model={resolved_d_model} not divisible by n_heads={resolved_n_heads}"
        )
    # d_model is the transformer's analogue of the VAE family's layer
    # width; enforce the same [min_layer_width, max_layer_width] guard.
    if "d_model" in arch_cfg and not (min_layer_width <= resolved_d_model <= max_layer_width):
        raise optuna.TrialPruned(
            f"d_model={resolved_d_model} outside [{min_layer_width}, {max_layer_width}]"
        )

    model = model_factories[model_name](n_genes, config=arch_cfg).to(device)
    bin_edges = make_log_bin_edges(model.n_bins, max_val=8.5)
    opt = optim.AdamW(model.parameters(), lr=lr)

    test_loss = float("inf")
    epoch_bar = tqdm(range(max_epochs), desc=f"{model_name} #{trial.number}", leave=False)
    for epoch in epoch_bar:
        epoch_transformer(model, loader_train, bin_edges, opt,
                           mask_fraction=mask_fraction, lambda_conf=lambda_conf, device=device)

        if epoch % eval_every == 0:
            test_loss, *_ = epoch_transformer(model, loader_test, bin_edges, None,
                                               mask_fraction=mask_fraction, lambda_conf=lambda_conf, device=device)
            if not math.isfinite(test_loss):
                raise optuna.TrialPruned()

            try:
                best = trial.study.best_value
            except ValueError:
                best = None
            epoch_bar.set_postfix(
                test_loss=f"{test_loss:.4f}",
                best=f"{best:.4f}" if best is not None else "n/a",
            )

            trial.report(test_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    _save_if_best_checkpoint(model, model_name, test_loss, trial, checkpoint_dir)

    return test_loss


def trial_summary_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """One concise progress line per finished trial (via tqdm.write, so it
    doesn't clobber any active progress bar)."""
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
    tqdm.write(
        f"  trial {trial.number:>4d}  state={trial.state.name:<9s}  "
        f"value={value_str:>10s}  best={best_str:>10s}  ({duration:5.1f}s)"
    )


def build_sampler(name: str):
    return _SAMPLERS[name]()


def build_pruner(name: str):
    return _PRUNERS[name]()


def run_study_for_model(model_name: str,
                         config: dict,
                         n_genes: int,
                         loader_train,
                         loader_test) -> optuna.Study:
    if model_name in TRANSFORMER_FAMILY_MODELS:
        objective = lambda trial: run_trial_transformer(
            trial, model_name, n_genes, loader_train, loader_test,
            config["search_space"], config["max_epochs"], config["eval_every"],
            config["mask_fraction"],
            config["min_layer_width"], config["max_layer_width"],
            config["checkpoint_dir"],
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
            config["mask_fraction"],
            default_latent_dim, n_encoder_layers, n_decoder_layers,
            config["min_layer_width"], config["max_layer_width"],
            config["checkpoint_dir"],
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
    ({model_name: {"best_value": float, "best_params": dict}}) is written
    to config["output"] (default tuned_configs.json) for convenience.
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

    loader_train = DataLoader(loaded["gene_reads_train"], batch_size=batch_size, shuffle=True )
    loader_test  = DataLoader(loaded["gene_reads_test"],  batch_size=n_cells,    shuffle=False)

    print(
        f"n_genes={n_genes}  n_cells={n_cells}  batch_size={batch_size}  "
        f"mask_fraction={mask_fraction}  device={device}"
    )

    studies = {}
    for model_name in config["models"]:
        studies[model_name] = run_study_for_model(model_name, config, n_genes, loader_train, loader_test)

    summary = {
        model_name: {"best_value": study.best_value, "best_params": study.best_params}
        for model_name, study in studies.items()
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
