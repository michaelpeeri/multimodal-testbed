"""Test and use the deterministic DAG/Hill surrogate for SERGIO.

The default run:

1. Generates one fixed 200-gene DAG from the local TRRUST edge list.
2. Generates 200 MR-state bins using a scrambled Sobol design.
3. Compares the zero-noise surrogate against make_synthetic_data6.
4. Searches for 10 bins with a high-dimensional, non-degenerate PCA spectrum.
5. Verifies the selected subset against a second zero-noise SERGIO run.

This intentionally disables SERGIO's technical-noise stages. The surrogate
models the latent continuous steady-state expression, followed by log1p only
when comparing with make_synthetic_data6's returned matrix.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from synthetic_data import (
    generate_sergio_grn_from_reference,
    load_sergio_dag,
    make_synthetic_data6,
    sample_sergio_mr_states,
    select_sergio_spectral_subset,
    sergio_dag_hill_forward,
    sergio_spectral_metrics,
)


def _clean_sergio_output(
    mr_state: np.ndarray,
    grn_path: str,
    mr_ids: list[int],
    decays: float,
    seed: int,
) -> np.ndarray:
  """Run SERGIO with all stochastic and technical stages disabled."""
  x, labels = make_synthetic_data6(
      mr_state=torch.tensor(mr_state, dtype=torch.float32),
      input_file_targets=grn_path,
      n_cells=mr_state.shape[0],
      mr_gene_ids=mr_ids,
      shared_coop_state=0.0,
      noise_params=0.0,
      decays=decays,
      sampling_state=1,
      min_cells_per_cluster=1,
      add_outlier_genes=False,
      add_lib_size_effect=False,
      add_dropout=False,
      convert_to_umi_counts=False,
      missing_rate=0.0,
      seed=seed,
  )
  x = x.detach().cpu().numpy()
  labels = np.asarray(labels)
  order = np.argsort(labels)
  assert np.array_equal(labels[order], np.arange(len(labels)))
  return x[order]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--reference-grn", default="data/trrust_rawdata.human.tsv")
  parser.add_argument("--n-genes", type=int, default=200)
  parser.add_argument("--n-states", type=int, default=200)
  parser.add_argument("--subset-size", type=int, default=10)
  parser.add_argument("--mr-low", type=float, default=1.0)
  parser.add_argument("--mr-high", type=float, default=5.0)
  parser.add_argument("--decays", type=float, default=0.8)
  parser.add_argument("--design", choices=("sobol", "random"), default="sobol")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--n-restarts", type=int, default=8)
  parser.add_argument("--swap-passes", type=int, default=10)
  parser.add_argument("--variance-weight", type=float, default=0.05)
  parser.add_argument("--skip-sergio", action="store_true")
  args = parser.parse_args()

  with tempfile.TemporaryDirectory(prefix="sergio_surrogate_test_") as temp_dir:
    grn_path = str(Path(temp_dir) / "targets.csv")
    _, mr_ids, _ = generate_sergio_grn_from_reference(
        reference_grn_path=args.reference_grn,
        n_genes=args.n_genes,
        output_path=grn_path,
        k_dist=("uniform", 1.0, 5.0),
        hill_coeff_dist=("constant", 2.0),
        seed=args.seed,
    )
    dag = load_sergio_dag(grn_path, shared_coop_state=0.0, mr_gene_ids=mr_ids)
    states = sample_sergio_mr_states(
        args.n_states, len(mr_ids), args.mr_low, args.mr_high,
        args.design, args.seed)

    raw = sergio_dag_hill_forward(states, dag, decays=args.decays)
    _, full_metrics = sergio_spectral_metrics(raw)
    selected, selected_metrics, _ = select_sergio_spectral_subset(
        states, dag, args.decays, args.subset_size, args.n_restarts,
        args.swap_passes, args.seed + 1, args.variance_weight)

    result = {
        "n_genes": dag.n_genes,
        "n_mrs": len(dag.mr_ids),
        "n_states": len(states),
        "design": args.design,
        "full_surrogate_spectrum": full_metrics,
        "selected_indices": selected,
        "selected_surrogate_spectrum": selected_metrics,
    }

    if not args.skip_sergio:
      try:
        full_sergio_log = _clean_sergio_output(
            states, grn_path, list(dag.mr_ids), args.decays, args.seed)
        full_surrogate_log = np.log1p(np.maximum(raw, 0.0))
        full_error = np.abs(full_sergio_log - full_surrogate_log)
        result["full_sergio_comparison"] = {
            "rmse": float(np.sqrt(np.mean(np.square(full_error)))),
            "max_abs_error": float(full_error.max()),
        }

        selected_states = states[selected]
        selected_raw = sergio_dag_hill_forward(
            selected_states, dag, decays=args.decays)
        selected_sergio_log = _clean_sergio_output(
            selected_states, grn_path, list(dag.mr_ids), args.decays, args.seed)
        selected_surrogate_log = np.log1p(np.maximum(selected_raw, 0.0))
        selected_error = np.abs(selected_sergio_log - selected_surrogate_log)
        result["selected_sergio_comparison"] = {
            "rmse": float(np.sqrt(np.mean(np.square(selected_error)))),
            "max_abs_error": float(selected_error.max()),
        }
      except ImportError as exc:
        result["sergio_error"] = str(exc)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
