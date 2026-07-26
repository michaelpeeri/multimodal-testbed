"""
Optuna-based hyperparameter tuning for the VAE model family in models.py.

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
Colab runtime restarts) that searches the training-loop hyperparameters
listed in "search_space" (learning rate, beta, gamma, min_free_bits,
lambda_entropy, lambda_vamp_repulsion, lambda_vamp_coverage -- any subset
of epoch_vae's kwargs other than mask_fraction). Model *architecture*
hyperparameters (encoder_dims/decoder_dims, n_pseudo, coeff/max_iter,
n_components, ...) are intentionally out of scope for this pass and stay
at each model's `model_factories` default.

`mask_fraction` is deliberately *not* part of the search space: it sets
the difficulty of the imputation problem/objective itself rather than a
knob to be optimized away, so it is read directly from the config's
top-level "mask_fraction" key (same value applied to every model/trial)
instead of being suggested by Optuna.

The objective being minimized is the held-out `epoch_vae` test loss.
Trials report intermediate test loss every `eval_every` epochs so the
configured pruner can stop clearly-bad trials early.

Only models trained via `epoch_vae` on the standard 6-tuple `forward()`
signature are supported (see VAE_FAMILY_MODELS below). `MeansModel` has no
hyperparameters to tune (its means are fit directly from data, not via
epoch_vae), and `ImputationTransformer` uses a different training loop
(`epoch_transformer`) -- both are out of scope here.

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
See tuning_config.example.json for a full example. Keys:
    problem_fn    : str, required -- path to the problem file described above
    models        : list[str], required -- subset of VAE_FAMILY_MODELS
    search_space  : dict, required -- {param_name: spec}, where spec is
                    {"type": "float"|"int", "low":..., "high":..., "log": bool}
                    or {"type": "categorical", "choices": [...]}.
                    param_name "lr" controls the optimizer's learning rate;
                    any other name must match an epoch_vae() kwarg. Params
                    not listed here are left at epoch_vae's own defaults.
                    "mask_fraction" may not be listed here -- see the
                    top-level "mask_fraction" config key instead.
    mask_fraction : float, default 0.1 -- fixed (not tuned) fraction of
                    observed genes masked for training/eval, forwarded as
                    epoch_vae's mask_fraction kwarg for every trial.
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

from models import model_factories, epoch_vae, VampPriorVAE, device


# Models trainable via epoch_vae's standard 6-tuple forward() convention.
# MeansModel (no hyperparameters, fit directly from data) and
# ImputationTransformer (different training loop) are intentionally excluded.
VAE_FAMILY_MODELS = [
    "GeneExpressionVAE",
    "DEQEncoderVAE",
    "MoVEVAE_K3",
    "MoVEVAE_K1",
    "VampPriorVAE",
    "DEQEncoderVampVAE",
]

_SAMPLERS = {
    "tpe": lambda: optuna.samplers.TPESampler(seed=0),
}

_PRUNERS = {
    "median": lambda: optuna.pruners.MedianPruner(n_warmup_steps=10),
}

_CONFIG_DEFAULTS = {
    "storage":       "sqlite:///tuning.db",
    "batch_size":    256,
    "n_trials":      50,
    "max_epochs":    50,
    "eval_every":    1,
    "sampler":       "tpe",
    "pruner":        "median",
    "output":        "tuned_configs.json",
    "mask_fraction": 0.1,
}


def load_config(path: str) -> dict:
    """Load and validate the JSON tuning config, merged over _CONFIG_DEFAULTS."""
    with open(path) as f:
        user_config = json.load(f)

    config = {**_CONFIG_DEFAULTS, **user_config}

    for key in ("problem_fn", "models", "search_space"):
        if key not in config:
            raise ValueError(f"tuning config {path!r} is missing required key {key!r}")

    unknown_models = set(config["models"]) - set(VAE_FAMILY_MODELS)
    if unknown_models:
        raise ValueError(
            f"Unsupported model(s) in config['models']: {sorted(unknown_models)}. "
            f"Supported: {VAE_FAMILY_MODELS}"
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
    """Suggest every parameter listed in search_space. Params not listed here
    are simply never passed to epoch_vae, which falls back to its own
    defaults for them."""
    return {name: suggest_from_spec(trial, name, spec) for name, spec in search_space.items()}


def run_trial(trial: optuna.Trial,
              model_name: str,
              n_genes: int,
              loader_train,
              loader_test,
              search_space: dict,
              max_epochs: int,
              eval_every: int,
              mask_fraction: float) -> float:
    """Build a fresh model, train it for max_epochs, report intermediate
    held-out test loss for pruning, and return the final test loss."""
    cfg = suggest_train_cfg(trial, search_space)
    lr = cfg.pop("lr", 3e-4)
    cfg["mask_fraction"] = mask_fraction

    model = model_factories[model_name](n_genes).to(device)
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
        lambda trial: run_trial(
            trial, model_name, n_genes, loader_train, loader_test,
            config["search_space"], config["max_epochs"], config["eval_every"],
            config["mask_fraction"],
        ),
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
