#!/usr/bin/env python
"""VULCAN-JAX driver: build RunState, integrate, pickle .vul."""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import time

import jax as _jax

_jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/jax_vulcan"),
    ),
)
_jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
_jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

print("Using JAX-native chem_funs with SymPy-faithful chem_rhs codegen")

import argparse

from . import legacy_io as op
from .config import dump_config, load_config
from . import op_jax
from . import outer_loop
from .runtime_validation import validate_runtime_config
from .state import RunState
from ._paths import PACKAGE_ROOT


def cli_main(argv=None):
    """Entry point for the ``vulcan-jax`` console script.

    ``--config`` selects a config by name (``default``, ``W39b``, ...; resolved
    CWD-first then packaged) or an explicit YAML path. The resolved config is
    written next to the output as ``<out_name>.config.yaml`` so the run can be
    reproduced with ``vulcan-jax --config <that file>``.
    """
    parser = argparse.ArgumentParser(prog="vulcan-jax")
    parser.add_argument(
        "--config",
        "-c",
        default="default",
        help="config name (default, HD189, W39b, ...) or path to a YAML file.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    validate_runtime_config(cfg, PACKAGE_ROOT)

    runstate = RunState.with_pre_loop_setup(cfg)

    dname = os.path.abspath(os.getcwd())
    output = op.Output(cfg=cfg)
    output.save_cfg(dname)

    # Save the fully-resolved config as YAML so the run is reproducible.
    resolved_path = os.path.join(
        os.path.abspath(cfg.output_dir), f"{cfg.out_name}.config.yaml"
    )
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    dump_config(cfg, resolved_path)
    print(f"Resolved config written to {resolved_path}")

    solver = op_jax.Ros2JAX()
    integ = outer_loop.OuterLoop(solver, output, cfg=cfg)

    print(f"VULCAN-JAX starting integration at t=0, dt={float(runstate.step.dt):.2e}")
    runstate = integ(runstate)

    print(f"VULCAN-JAX done. Saving output to {cfg.output_dir}{cfg.out_name}")
    output.save_out(runstate, dname)

    if getattr(cfg, "use_plot_end", False) or (
        getattr(cfg, "use_plot_evo", False) and getattr(cfg, "save_evolution", False)
    ):
        from .state import legacy_view

        _var, _atm, _para = legacy_view(runstate)
        if getattr(cfg, "use_plot_end", False):
            output.plot_end(_var, _atm, _para)
        if getattr(cfg, "use_plot_evo", False) and getattr(
            cfg, "save_evolution", False
        ):
            output.plot_evo(_var, _atm)

    print(f"Total wall time: {time.time() - runstate.metadata.start_time:.1f}s")


if __name__ == "__main__":
    cli_main()
