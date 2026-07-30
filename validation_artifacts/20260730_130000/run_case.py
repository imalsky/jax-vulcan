#!/usr/bin/env python
"""Run one VULCAN-JAX case and dump convergence diagnostics.

Usage:
    python run_case.py <config-name> <overrides-json> <out.npz>

`overrides-json` is a JSON object folded over the loaded config, e.g.
`'{"conver_ignore": [], "use_conv_stall": false}'`.

Writes an .npz with the final state plus every diagnostic Phase 3 of the
validation plan asks for, and prints a one-line JSON summary on stdout
prefixed by `RESULT ` so the driver can scrape it.
"""

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def _freeze_import_locked_knobs(cfg_name):
    """Set $VULCAN_JAX_* for the import-frozen knobs BEFORE importing vulcan_jax.

    `network`, `atom_list` and `com_file` are parsed once at the first
    `import vulcan_jax` and cannot be changed afterwards, so a config naming a
    different network (W39b, K2-18b: SNCHO) fails unless the environment is set
    first. This mirrors what `vulcan_jax_cli._relaunch_for_frozen_knobs` does by
    re-exec'ing; here we are still before the import, so setting them is enough.
    Reads the YAML directly with PyYAML so nothing imports vulcan_jax early.
    """
    import yaml

    repo_cfg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "src", "vulcan_jax", "configs", f"{cfg_name}.yaml",
    )
    local_cfg = os.path.join("configs", f"{cfg_name}.yaml")
    path = local_cfg if os.path.exists(local_cfg) else repo_cfg
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    for key, env in (
        ("network", "VULCAN_JAX_NETWORK"),
        ("atom_list", "VULCAN_JAX_ATOM_LIST"),
        ("com_file", "VULCAN_JAX_COM_FILE"),
    ):
        if key not in raw or env in os.environ:
            continue
        val = raw[key]
        os.environ[env] = ",".join(val) if isinstance(val, list) else str(val)


def main():
    cfg_name, overrides_json, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    overrides = json.loads(overrides_json)
    _freeze_import_locked_knobs(cfg_name)

    t_import0 = time.time()
    import jax

    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_vulcan")),
    )
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    from vulcan_jax import legacy_io as op
    from vulcan_jax import op_jax, outer_loop
    from vulcan_jax.config import load_config
    from vulcan_jax.runtime_validation import validate_runtime_config
    from vulcan_jax.state import RunState
    from vulcan_jax._paths import PACKAGE_ROOT
    from vulcan_jax import chem_funs

    t_import = time.time() - t_import0

    cfg = load_config(cfg_name)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown config key in overrides: {k}")
        setattr(cfg, k, v)
    validate_runtime_config(cfg, PACKAGE_ROOT)

    t_setup0 = time.time()
    rs = RunState.with_pre_loop_setup(cfg)
    t_setup = time.time() - t_setup0

    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)

    t_run0 = time.time()
    rs = integ(rs)
    t_run = time.time() - t_run0

    spec = list(chem_funs.spec_list)
    y = np.asarray(rs.step.y, dtype=np.float64)
    ymix = np.asarray(rs.step.ymix, dtype=np.float64)
    pco = np.asarray(rs.atm.pco, dtype=np.float64)
    Tco = np.asarray(rs.atm.Tco, dtype=np.float64)

    # Which (level, species) cell the convergence metric is pinned on.
    wvm = np.asarray(rs.params.where_varies_most, dtype=np.float64)
    if wvm.size and np.isfinite(wvm).any():
        iz, isp = np.unravel_index(np.nanargmax(wvm), wvm.shape)
        ctrl = {
            "species": spec[int(isp)],
            "level": int(iz),
            "p_bar": float(pco[int(iz)] / 1e6),
            "T_K": float(Tco[int(iz)]),
            "value": float(wvm[iz, isp]),
        }
    else:
        ctrl = None

    atom_order = list(rs.atoms.atom_order)
    atom_loss = np.asarray(rs.atoms.atom_loss, dtype=np.float64)

    summary = {
        "config": cfg_name,
        "overrides": overrides,
        "accept_steps": int(rs.params.count),
        "delta_rejects": int(rs.params.delta_count),
        "nega_rejects": int(rs.params.nega_count),
        "loss_rejects": int(rs.params.loss_count),
        "sim_time_s": float(rs.step.t),
        "longdy": float(rs.step.longdy),
        "longdydt": float(rs.step.longdydt),
        "end_case": int(rs.params.end_case),
        "termination_reason": int(rs.params.termination_reason),
        "controlling_cell": ctrl,
        "atom_order": atom_order,
        "atom_loss": {a: float(v) for a, v in zip(atom_order, atom_loss)},
        "max_abs_atom_loss": float(np.max(np.abs(atom_loss))),
        "all_finite": bool(np.all(np.isfinite(y)) and np.all(np.isfinite(ymix))),
        "min_y": float(np.min(y)),
        "n_negative_cells": int(np.sum(y < 0.0)),
        "t_import_s": round(t_import, 2),
        "t_setup_s": round(t_setup, 2),
        "t_run_s": round(t_run, 2),
        "conver_ignore": list(getattr(cfg, "conver_ignore", [])),
        "use_conv_stall": bool(getattr(cfg, "use_conv_stall", True)),
        "use_vm_mol": bool(getattr(cfg, "use_vm_mol", False)),
        "use_hybrid_vm_mol": bool(getattr(cfg, "use_hybrid_vm_mol", False)),
    }

    np.savez_compressed(
        out_path,
        y=y,
        ymix=ymix,
        pco=pco,
        Tco=Tco,
        species=np.array(spec, dtype=object),
        where_varies_most=wvm,
        atom_loss=atom_loss,
        atom_order=np.array(atom_order, dtype=object),
        summary_json=json.dumps(summary),
    )
    print("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
