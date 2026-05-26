"""Validate one Ros2 step against VULCAN's Ros2.solver.

Compares sol after a single 2nd-order Rosenbrock step.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Oracle test: requires VULCAN-master sibling for the upstream op.Ros2.solver
# reference. Skip cleanly when absent.
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"VULCAN-master oracle absent at {VULCAN_MASTER}; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )

warnings.filterwarnings("ignore")


def main() -> int:
    # === VULCAN-master pipeline + Ros2 step ===
    sys.path.insert(0, str(VULCAN_MASTER))

    import vulcan_jax.vulcan_cfg as cfg_v
    import store as st_v
    import build_atm as ba_v
    import op as op_v

    data_var = st_v.Variables()
    data_atm = st_v.AtmData()
    make_atm = ba_v.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if cfg_v.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op_v.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    ini = ba_v.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op_v.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)
    data_var.dt = 1e-10
    data_var.ymix = data_var.y / np.vstack(np.sum(data_var.y, axis=1))

    data_para = st_v.Parameters()
    solver_v = op_v.Ros2()
    solver_v.naming_solver(data_para)

    # Snapshot y BEFORE the step (since solver mutates var.y)
    y0 = np.asarray(data_var.y, dtype=np.float64).copy()
    k_dict = {i: np.asarray(v, dtype=np.float64).copy() for i, v in data_var.k.items()}

    # Reference one Ros2 step (mutates var.y)
    var_after, para_after = solver_v.solver(data_var, data_atm, data_para)
    sol_ref = np.asarray(var_after.y, dtype=np.float64).copy()
    print(
        f"VULCAN ros2 step: sol shape {sol_ref.shape}, delta = {para_after.delta:.3e}"
    )

    # === Switch to VULCAN-JAX ===
    for mod in ("vulcan_cfg", "store", "build_atm", "op", "chem_funs"):
        sys.modules.pop(mod, None)
    while str(VULCAN_MASTER) in sys.path:
        sys.path.remove(str(VULCAN_MASTER))

    import vulcan_jax.vulcan_cfg as cfg_jax

    # Pin JAX modules to the exact network and transport flags used by the
    # master-side state captured above. `jax_step` imports
    # `chem_funs.chem_rhs_codegen` at module import time, so this must happen
    # before importing `chem_funs` or `jax_step`.
    from vulcan_jax._paths import resolve_data_path
    cfg_jax.network = str(resolve_data_path(cfg_v.network))
    for name in (
        "use_moldiff",
        "use_vm_mol",
        "use_settling",
        "use_topflux",
        "use_botflux",
    ):
        if hasattr(cfg_v, name):
            setattr(cfg_jax, name, getattr(cfg_v, name))

    import vulcan_jax.chem_funs as chem_funs
    import jax.numpy as jnp
    from vulcan_jax.jax_step import jax_ros2_step, make_atm_static

    nz, ni = y0.shape

    # Pack k into JAX array
    k_arr = np.zeros((chem_funs.nr + 1, nz), dtype=np.float64)
    for i, v in k_dict.items():
        k_arr[i] = v

    # Production one-step kernel: codegen RHS + analytical Jacobian + JAX
    # diffusion + diagonal-aware block Thomas. This is the same hot path the
    # outer loop calls, so this test cannot accidentally validate only the
    # preserved segment_sum reference RHS.
    atm_static = make_atm_static(data_atm, ni, nz)
    sol_jax, delta_jax = jax_ros2_step(
        jnp.asarray(y0),
        jnp.asarray(k_arr),
        jnp.float64(data_var.dt),
        atm_static,
        chem_funs._NET_JAX,
    )
    sol_jax = np.asarray(sol_jax, dtype=np.float64)
    print(
        f"VULCAN-JAX production step delta max = {float(np.max(np.asarray(delta_jax))):.3e}"
    )

    # Compare
    relerr = np.abs(sol_jax - sol_ref) / np.maximum(np.abs(sol_ref), 1e-12)
    max_relerr = relerr.max()
    print(f"sol_jax vs sol_ref: max relerr = {max_relerr:.3e}")
    # Per-species report for top 5 worst
    per_sp = relerr.max(axis=0)
    worst = np.argsort(per_sp)[::-1][:5]
    for j in worst:
        if per_sp[j] > 1e-6:
            print(f"  {chem_funs.spec_list[j]}: max relerr {per_sp[j]:.3e}")

    print()
    ok = max_relerr < 1e-3  # generous; the integrator self-corrects
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Pytest wrapper. This test does a deliberate VULCAN-master ↔
    VULCAN-JAX module-table swap (see `sys.modules.pop` block in
    `main()`) which only works from a cold Python start. Under pytest
    the modules are already cached from prior tests, so we run `main()`
    in a fresh subprocess and assert the exit code."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(main())
