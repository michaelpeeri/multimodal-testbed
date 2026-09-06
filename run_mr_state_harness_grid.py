"""Run a small grid of MR-state harness configurations.

The grid manifest contains one common harness configuration plus named
``base_overrides`` for each stage condition.  Each condition is run through
``run_mr_state_comparison`` independently, so every run still gets its own
shared GRN and paired replicate seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mr_state_harness import run_mr_state_comparison


def run_grid(path: str, only: set[str] | None = None) -> None:
    with open(path) as handle:
        manifest = json.load(handle)
    grid = manifest.get("grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("grid manifest requires a non-empty 'grid' list")
    names = [condition.get("name") for condition in grid]
    if any(not name for name in names):
        raise ValueError("every grid condition requires a non-empty 'name'")
    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise ValueError(
                f"unknown grid condition(s) in --only: {', '.join(unknown)}; "
                f"available: {', '.join(names)}"
            )

    output_template = manifest.get(
        "output_template", "mr_state_comparison.{name}.pickle")
    plot_template = manifest.get("plot_template")
    common = {key: value for key, value in manifest.items()
              if key not in {"grid", "output_template", "plot_template"}}
    common_overrides = dict(common.get("base_overrides") or {})

    for condition in grid:
        name = condition.get("name")
        if only and name not in only:
            continue
        config = copy.deepcopy(common)
        overrides = dict(common_overrides)
        overrides.update(condition.get("base_overrides") or {})
        config["base_overrides"] = overrides
        config["output"] = output_template.format(name=name)
        if plot_template is not None:
            config["plot"] = plot_template.format(name=name)
        print(f"[grid] starting {name}: output={config['output']}")
        run_mr_state_comparison(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="grid manifest JSON")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="run only the named grid condition(s); defaults to all conditions",
    )
    args = parser.parse_args(argv)
    run_grid(args.config, only=set(args.only or []))


if __name__ == "__main__":
    main()
