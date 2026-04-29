"""Profile the current VULCAN-JAX Ros2 step implementation."""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def _setup_hd189_state():
    import chem
    import network
    import vulcan_cfg
    from state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _data_para = legacy_view(rs)
    net = chem.to_jax(network.parse_network(vulcan_cfg.network))
    return data_var, data_atm, net


def _pack_k_arr(var, nr: int, nz: int) -> np.ndarray:
    """Read the dense `(nr+1, nz)` array directly off `var.k_arr`."""
    out = np.asarray(var.k_arr, dtype=np.float64)
    if out.shape != (nr + 1, nz):
        raise ValueError(f"var.k_arr shape {out.shape} != expected ({nr+1}, {nz})")
    return out


def _timeit(fn, n_iter: int = 20) -> float:
    t0 = time.time()
    for _ in range(n_iter):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, tuple):
            for item in out:
                if hasattr(item, "block_until_ready"):
                    item.block_until_ready()
    return (time.time() - t0) / n_iter * 1000.0


def main() -> None:
    import jax
    import jax.numpy as jnp

    import chem as chem_mod
    from jax_step import (
        compute_diff_grav,
        jax_ros2_step,
        make_atm_static,
        _build_diff_coeffs_jax,
    )
    from solver import (
        factor_block_thomas_diag_offdiag,
        solve_block_thomas_diag_offdiag,
    )

    data_var, data_atm, net = _setup_hd189_state()
    nz, ni = data_var.y.shape
    k_arr = _pack_k_arr(data_var, net.nr, nz)
    atm_static = make_atm_static(data_atm, ni, nz)

    y_j = jnp.asarray(data_var.y)
    k_j = jnp.asarray(k_arr)
    dt = jnp.float64(1e-10)

    # Warm the full step once so the breakdown is steady-state, not compile time.
    sol, _ = jax_ros2_step(y_j, k_j, dt, atm_static, net)
    sol.block_until_ready()

    grav = compute_diff_grav(atm_static)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = _build_diff_coeffs_jax(
        y_j, atm_static, grav
    )
    chem_J = chem_mod.chem_jac_analytical(y_j, atm_static.M, k_j, net)
    diag_d = A_eddy[:, None] + A_mol
    diag_d = diag_d.at[0].add(
        jnp.where(
            atm_static.use_botflux,
            -atm_static.bot_vdep / atm_static.dzi[0],
            jnp.zeros_like(atm_static.bot_vdep),
        )
    )
    sup_d = B_eddy[:-1, None] + B_mol[:-1]
    sub_d = C_eddy[1:, None] + C_mol[1:]
    eye = jnp.eye(ni)
    di = jnp.arange(ni)
    c0 = 1.0 / ((1.0 + 1.0 / jnp.sqrt(2.0)) * dt)
    diag = c0 * eye[None] - chem_J
    diag = diag.at[:, di, di].add(-diag_d)
    sup_neg = -sup_d
    sub_neg = -sub_d
    rhs = chem_mod.chem_rhs(y_j, atm_static.M, k_j, net)

    factor_jit = jax.jit(factor_block_thomas_diag_offdiag)
    solve_jit = jax.jit(solve_block_thomas_diag_offdiag)
    rhs_jit = jax.jit(chem_mod.chem_rhs)
    jac_jit = jax.jit(chem_mod.chem_jac_analytical)
    step_jit = jax.jit(jax_ros2_step)

    factors = factor_jit(diag, sup_neg, sub_neg)
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, factors)

    t_pack = _timeit(lambda: _pack_k_arr(data_var, net.nr, nz), n_iter=20)
    t_atm = _timeit(lambda: make_atm_static(data_atm, ni, nz), n_iter=20)
    t_rhs = _timeit(lambda: rhs_jit(y_j, atm_static.M, k_j, net), n_iter=20)
    t_jac = _timeit(lambda: jac_jit(y_j, atm_static.M, k_j, net), n_iter=20)
    t_factor = _timeit(lambda: factor_jit(diag, sup_neg, sub_neg), n_iter=20)
    t_solve = _timeit(lambda: solve_jit(factors, rhs), n_iter=20)
    t_step = _timeit(lambda: step_jit(y_j, k_j, dt, atm_static, net), n_iter=20)

    print(f"pack_k_arr_ms={t_pack:.3f}")
    print(f"make_atm_static_ms={t_atm:.3f}")
    print(f"chem_rhs_ms={t_rhs:.3f}")
    print(f"chem_jac_analytical_ms={t_jac:.3f}")
    print(f"block_thomas_factor_ms={t_factor:.3f}")
    print(f"block_thomas_solve_ms={t_solve:.3f}")
    print(f"jax_ros2_step_ms={t_step:.3f}")


if __name__ == "__main__":
    main()
