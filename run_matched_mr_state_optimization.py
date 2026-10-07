"""Optimize an MR-state matrix against one exact archived GRN.

This is intentionally separate from the historical optimizer entry point,
whose command-line block uses hard-coded notebook paths. The configuration
must provide ``fixed_grn_path`` and ``fixed_grn_sha256``; the optimizer stores
that GRN provenance in its normal result pickle.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

from mr_state_ga_opt import de_loop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="matched-GRN optimizer JSON")
    parser.add_argument("--output", required=True, help="result pickle path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=70)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--differential-weight", type=float, default=0.5)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    args = parser.parse_args(argv)

    if not os.environ.get("PYTHONHASHSEED"):
        print(
            "WARNING: PYTHONHASHSEED is unset. This run uses an archived GRN, "
            "so the optimizer itself remains GRN-deterministic, but launch "
            "future generated-GRN comparisons with a shared hash seed."
        )

    result = de_loop(
        args.config,
        max_generations=args.generations,
        n_population=args.population,
        seed=args.seed,
        differential_weight=args.differential_weight,
        crossover_rate=args.crossover_rate,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(result, handle)
    print(
        f"wrote {output} with GRN records={len(result.get('grn', []))} "
        f"best_fitness={result.get('best_fitness'):.6g}"
    )


if __name__ == "__main__":
    main()
