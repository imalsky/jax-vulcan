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
import sys

from . import legacy_io as op
from .config import default_config, dump_config, load_config
from . import op_jax
from . import outer_loop
from .runtime_validation import validate_runtime_config
from .state import RunState
from ._paths import PACKAGE_ROOT

# Set on the re-executed process by `_relaunch_for_frozen_knobs` so a relaunch
# that fails to resolve the mismatch raises instead of looping.
_RELAUNCH_GUARD = "VULCAN_JAX_CLI_RELAUNCHED"

# Import-frozen knob -> its environment override. Mirrors `config._FROZEN_ENV`;
# the values are read once at the first `import vulcan_jax`.
_FROZEN_KNOB_ENV = {
    "network": "VULCAN_JAX_NETWORK",
    "atom_list": "VULCAN_JAX_ATOM_LIST",
    "com_file": "VULCAN_JAX_COM_FILE",
}


def _frozen_knob_mismatch(cfg):
    """Import-frozen knobs `cfg` disagrees with, as ``{env var: value}``.

    Compares against `default_config()` because that is exactly what the
    import-time freeze read. `load_config` folds ``$VULCAN_JAX_*`` over the YAML
    before returning, so a knob the caller overrode explicitly matches the
    frozen value and never appears here.
    """
    frozen = default_config()
    mismatch = {}
    for knob, env in _FROZEN_KNOB_ENV.items():
        want = getattr(cfg, knob, None)
        have = getattr(frozen, knob, None)
        if knob == "atom_list":
            if list(want or ()) != list(have or ()):
                mismatch[env] = ",".join(str(a) for a in want)
        elif str(want) != str(have):
            mismatch[env] = str(want)
    return mismatch


def _relaunch_for_frozen_knobs(cfg, config_ref):
    """Re-exec with ``$VULCAN_JAX_*`` set when the config needs a different network.

    ``network`` / ``atom_list`` / ``com_file`` are read once at the first
    ``import vulcan_jax`` -- the parsed network, the composition tables, and the
    codegen RHS are all built from them -- so a config naming a different
    network cannot be honored in this process. Re-exec once with the variables
    set instead of making the caller prefix them onto every command, so
    ``vulcan-jax --config W39b`` works on its own.

    Returns normally when no relaunch is needed; otherwise does not return.
    """
    mismatch = _frozen_knob_mismatch(cfg)
    if not mismatch:
        return
    assignments = " ".join(f"{k}={v}" for k, v in sorted(mismatch.items()))
    if os.environ.get(_RELAUNCH_GUARD):
        raise RuntimeError(
            f"config {config_ref!r} still disagrees with the import-frozen "
            f"network settings after a relaunch with {assignments}. This should "
            "not happen; run with the environment variables set manually and "
            "report the config."
        )
    env = dict(os.environ)
    env.update(mismatch)
    env[_RELAUNCH_GUARD] = "1"
    print(
        f"Config {config_ref!r} needs import-frozen settings that differ from the "
        f"defaults; relaunching with {assignments}"
    )
    sys.stdout.flush()
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-m",
            "vulcan_jax.vulcan_jax_cli",
            "--config",
            str(config_ref),
        ],
        env,
    )


def cli_main(argv=None):
    """Entry point for the ``vulcan-jax`` console script.

    ``--config`` selects a config by name (``default``, ``W39b``, ...; resolved
    CWD-first then packaged) or an explicit YAML path. The resolved config is
    written next to the output as ``<out_name>.config.yaml`` so the run can be
    reproduced with ``vulcan-jax --config <that file>``.

    A config selecting a non-default network re-execs once with the matching
    ``$VULCAN_JAX_*`` variables set (see `_relaunch_for_frozen_knobs`).
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
    # Before any validation: a config naming a non-default network can only be
    # honored by a process that imported vulcan_jax with it already selected.
    _relaunch_for_frozen_knobs(cfg, args.config)
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

        _var, _atm, _para = legacy_view(runstate, cfg=cfg)
        if getattr(cfg, "use_plot_end", False):
            output.plot_end(_var, _atm, _para)
        if getattr(cfg, "use_plot_evo", False) and getattr(
            cfg, "save_evolution", False
        ):
            output.plot_evo(_var, _atm)

    print(f"Total wall time: {time.time() - runstate.metadata.start_time:.1f}s")


if __name__ == "__main__":
    cli_main()
