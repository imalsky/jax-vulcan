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

from . import legacy_io as op
from . import vulcan_cfg
from . import op_jax
from . import outer_loop
from .runtime_validation import validate_runtime_config
from .state import RunState
from ._paths import PACKAGE_ROOT


def cli_main():
    """Entry point for the ``vulcan-jax`` console script."""
    validate_runtime_config(vulcan_cfg, PACKAGE_ROOT)

    if vulcan_cfg.ode_solver != "Ros2":
        raise NotImplementedError(
            f"VULCAN-JAX targets the Ros2 solver only; got "
            f"ode_solver={vulcan_cfg.ode_solver!r}."
        )

    runstate = RunState.with_pre_loop_setup(vulcan_cfg)

    dname = os.path.abspath(os.getcwd())
    output = op.Output()
    output.save_cfg(dname)

    solver = op_jax.Ros2JAX()
    integ = outer_loop.OuterLoop(solver, output)

    print(f"VULCAN-JAX starting integration at t=0, dt={float(runstate.step.dt):.2e}")
    runstate = integ(runstate)

    print(
        f"VULCAN-JAX done. Saving output to {vulcan_cfg.output_dir}{vulcan_cfg.out_name}"
    )
    output.save_out(runstate, dname)

    if getattr(vulcan_cfg, "use_plot_end", False) or (
        getattr(vulcan_cfg, "use_plot_evo", False)
        and getattr(vulcan_cfg, "save_evolution", False)
    ):
        from .state import legacy_view

        _var, _atm, _para = legacy_view(runstate)
        if getattr(vulcan_cfg, "use_plot_end", False):
            output.plot_end(_var, _atm, _para)
        if getattr(vulcan_cfg, "use_plot_evo", False) and getattr(
            vulcan_cfg, "save_evolution", False
        ):
            output.plot_evo(_var, _atm)

    print(f"Total wall time: {time.time() - runstate.metadata.start_time:.1f}s")


if __name__ == "__main__":
    cli_main()
