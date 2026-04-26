"""Validate one Ros2 step against VULCAN's Ros2.solver.

Compares sol after a single 2nd-order Rosenbrock step.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def main() -> int:
    # === VULCAN-master pipeline + Ros2 step ===
    VULCAN_MASTER = ROOT.parent / "VULCAN-master"
    sys.path.insert(0, str(VULCAN_MASTER))

    import vulcan_cfg as cfg_v
    import store as st_v
    import build_atm as ba_v
    import op as op_v
    import chem_funs as cf_v

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
    M0 = np.asarray(data_atm.M, dtype=np.float64).copy()
    k_dict = {i: np.asarray(v, dtype=np.float64).copy() for i, v in data_var.k.items()}

    # Reference one Ros2 step (mutates var.y)
    var_after, para_after = solver_v.solver(data_var, data_atm, data_para)
    sol_ref = np.asarray(var_after.y, dtype=np.float64).copy()
    print(f"VULCAN ros2 step: sol shape {sol_ref.shape}, delta = {para_after.delta:.3e}")

    # === Switch to VULCAN-JAX ===
    for mod in ("vulcan_cfg", "store", "build_atm", "op", "chem_funs"):
        sys.modules.pop(mod, None)
    while str(VULCAN_MASTER) in sys.path:
        sys.path.remove(str(VULCAN_MASTER))
    sys.path.insert(0, str(ROOT))

    import vulcan_cfg as cfg_jax
    import network as net_mod
    import gibbs as gibbs_mod
    import chem as chem_mod
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import diffusion_numpy_ref as diff_mod
    import solver as solver_mod
    import jax.numpy as jnp

    net = net_mod.parse_network(cfg_jax.network)
    net_jax = chem_mod.to_jax(net)
    coeffs_arr, _ = gibbs_mod.load_nasa9(net.species, ROOT / "thermo")
    nz, ni = y0.shape

    # Pack k into JAX array
    k_arr = np.zeros((net.nr + 1, nz), dtype=np.float64)
    for i, v in k_dict.items():
        k_arr[i] = v

    # First evaluation: chem RHS + diff at y0
    chem_jac_jax = np.asarray(chem_mod.chem_jac(jnp.asarray(y0), jnp.asarray(M0), jnp.asarray(k_arr), net_jax))
    diff_coeffs = diff_mod.build_diffusion_coeffs(y0, data_atm, cfg_jax)
    diff_at_y0 = diff_mod.apply_diffusion(y0, diff_coeffs)
    chem_rhs_y0 = np.asarray(chem_mod.chem_rhs(jnp.asarray(y0), jnp.asarray(M0), jnp.asarray(k_arr), net_jax))
    df = chem_rhs_y0 + diff_at_y0

    diff_diag, diff_sup, diff_sub = diff_mod.diffusion_block_diags(diff_coeffs, ni)

    # Compute LHS = c0*I - chem_J - diff_J
    r_ros = 1.0 + 1.0 / np.sqrt(2.0)
    c0 = 1.0 / (r_ros * data_var.dt)
    eye = np.eye(ni)
    diag = c0 * eye[None] - chem_jac_jax
    di = np.arange(ni)
    diag[:, di, di] -= diff_diag
    sup = np.zeros((nz - 1, ni, ni))
    sup[:, di, di] = -diff_sup
    sub = np.zeros((nz - 1, ni, ni))
    sub[:, di, di] = -diff_sub

    # Solve LHS @ k1 = df
    k1 = np.asarray(solver_mod.block_thomas(jnp.asarray(diag), jnp.asarray(sup), jnp.asarray(sub), jnp.asarray(df)))

    yk2 = y0 + k1 / r_ros

    # Second evaluation
    diff_at_yk2 = diff_mod.apply_diffusion(yk2, diff_coeffs)
    chem_rhs_yk2 = np.asarray(chem_mod.chem_rhs(jnp.asarray(yk2), jnp.asarray(M0), jnp.asarray(k_arr), net_jax))
    df2 = chem_rhs_yk2 + diff_at_yk2
    rhs2 = df2 - (2.0 / (r_ros * data_var.dt)) * k1
    k2 = np.asarray(solver_mod.block_thomas(jnp.asarray(diag), jnp.asarray(sup), jnp.asarray(sub), jnp.asarray(rhs2)))

    sol_jax = y0 + (3.0 / (2.0 * r_ros)) * k1 + (1.0 / (2.0 * r_ros)) * k2

    # Compare
    relerr = np.abs(sol_jax - sol_ref) / np.maximum(np.abs(sol_ref), 1e-12)
    max_relerr = relerr.max()
    print(f"sol_jax vs sol_ref: max relerr = {max_relerr:.3e}")
    # Per-species report for top 5 worst
    per_sp = relerr.max(axis=0)
    worst = np.argsort(per_sp)[::-1][:5]
    for j in worst:
        if per_sp[j] > 1e-6:
            print(f"  {net.species[j]}: max relerr {per_sp[j]:.3e}")

    print()
    ok = max_relerr < 1e-3   # generous; the integrator self-corrects
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
