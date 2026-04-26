"""Pure-JAX Ros2 step function — vmap-compatible.

This module exposes a single `jax_ros2_step` that takes JAX arrays only and
returns updated state + truncation error proxy. It can be JIT'd, vmapped over
a batch axis, and run on GPU without code changes.

The function operates on a STATIC atmosphere snapshot (Kzz, Dzz, Hpi, Ti,
alpha, ms, g, dzi).  This is intentional: Kzz/Dzz/etc. depend only on atm,
not y -- they're constant during a step.  ysum(y) IS handled inside the step.

For batch parallelism over many atmospheric profiles, vmap over the leading
axis of (y, M, k, T_atm, K_atm, ...).  Each batch element runs an independent
Ros2 step.

Limitations:
  - 3-body M factor uses literal_unroll over flagged reactions (slow); for
    speed-of-light, port the chemistry RHS to use a precomputed
    "M-multiplied k" array per layer.
  - BC variants (use_fix_sp_bot, use_botflux deposition velocity) are NOT
    handled here -- the outer Python wrapper applies BC corrections.

Kept simple to demonstrate JIT/vmap viability.  See `op_jax.Ros2JAX` for the
production path that handles all VULCAN BCs/condensation/ions.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from chem import chem_rhs, chem_jac, NetworkArrays
from solver import block_thomas

jax.config.update("jax_enable_x64", True)


class AtmStatic(NamedTuple):
    """Atmosphere parameters that are constant within a Ros2 step.

    All fields are JAX arrays. Shape information (nz, ni) is inferred from
    the leading axis of `Tco` and the trailing axis of `Dzz` -- pure traced.
    """
    Kzz: jnp.ndarray       # (nz-1,)
    Dzz: jnp.ndarray       # (nz-1, ni)
    dzi: jnp.ndarray       # (nz-1,)
    vz: jnp.ndarray        # (nz-1,)
    Hpi: jnp.ndarray       # (nz-1,)
    Ti: jnp.ndarray        # (nz-1,)
    Tco: jnp.ndarray       # (nz,)
    g: jnp.ndarray         # (nz,)
    ms: jnp.ndarray        # (ni,)
    alpha: jnp.ndarray     # (ni,)
    M: jnp.ndarray         # (nz,)


def _build_diff_coeffs_jax(y, atm: AtmStatic):
    """JAX version of diffusion.build_diffusion_coeffs (interior + boundaries).

    Returns:
        A_eddy: (nz,)        eddy A coefficient per layer
        B_eddy: (nz,)
        C_eddy: (nz,)
        A_mol:  (nz, ni)     mol Ai per layer per species
        B_mol:  (nz, ni)
        C_mol:  (nz, ni)
        ysum:   (nz,)
    """
    Kzz, Dzz, dzi, vz, Hpi, Ti, Tco, g, ms, alpha = (
        atm.Kzz, atm.Dzz, atm.dzi, atm.vz, atm.Hpi, atm.Ti, atm.Tco, atm.g, atm.ms, atm.alpha
    )
    nz = atm.Tco.shape[0]
    ni = atm.ms.shape[0]
    Navo = 6.02214086e23
    kb = 1.38064852e-16

    ysum = jnp.sum(y, axis=1)

    # Interior coefficients (j = 1 .. nz-2). We build the full nz arrays then
    # overwrite the boundaries.
    j_int = jnp.arange(1, nz - 1)
    dz_ave = 0.5 * (dzi[j_int - 1] + dzi[j_int])              # (nz-2,)

    A_eddy_int = -1.0 / dz_ave * (
        Kzz[j_int] / dzi[j_int] * (ysum[j_int + 1] + ysum[j_int]) / 2.0
        + Kzz[j_int - 1] / dzi[j_int - 1] * (ysum[j_int] + ysum[j_int - 1]) / 2.0
    ) / ysum[j_int]
    B_eddy_int = 1.0 / dz_ave * Kzz[j_int] / dzi[j_int] * (ysum[j_int + 1] + ysum[j_int]) / 2.0 / ysum[j_int + 1]
    C_eddy_int = 1.0 / dz_ave * Kzz[j_int - 1] / dzi[j_int - 1] * (ysum[j_int] + ysum[j_int - 1]) / 2.0 / ysum[j_int - 1]

    # Vertical advection (bool * value gives upwind switch in NumPy; same in JAX)
    A_eddy_int = A_eddy_int - ((vz[j_int] > 0) * vz[j_int] - (vz[j_int - 1] < 0) * vz[j_int - 1]) / dz_ave
    B_eddy_int = B_eddy_int - ((vz[j_int] < 0) * vz[j_int]) / dz_ave
    C_eddy_int = C_eddy_int + ((vz[j_int - 1] > 0) * vz[j_int - 1]) / dz_ave

    # Boundary j = 0
    A_eddy_0 = -1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0]
    A_eddy_0 = A_eddy_0 - ((vz[0] > 0) * vz[0]) / dzi[0]
    B_eddy_0 = 1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1]
    B_eddy_0 = B_eddy_0 - ((vz[0] < 0) * vz[0]) / dzi[0]
    C_eddy_0 = 0.0

    # Boundary j = nz-1
    A_eddy_top = -1.0 / dzi[-1] * (Kzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-1]
    A_eddy_top = A_eddy_top + ((vz[-1] < 0) * vz[-1]) / dzi[-1]
    C_eddy_top = 1.0 / dzi[-1] * (Kzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-2]
    C_eddy_top = C_eddy_top + ((vz[-1] > 0) * vz[-1]) / dzi[-1]
    B_eddy_top = 0.0

    A_eddy = jnp.concatenate([jnp.array([A_eddy_0]), A_eddy_int, jnp.array([A_eddy_top])])
    B_eddy = jnp.concatenate([jnp.array([B_eddy_0]), B_eddy_int, jnp.array([B_eddy_top])])
    C_eddy = jnp.concatenate([jnp.array([C_eddy_0]), C_eddy_int, jnp.array([C_eddy_top])])

    # Molecular diffusion blocks (per-species)
    # Interior: shape (nz-2, ni)
    grav_j = -1.0 / Hpi[j_int][:, None] + ms[None, :] * g[j_int][:, None] / (Navo * kb * Ti[j_int][:, None]) \
             + alpha[None, :] / Ti[j_int][:, None] * (Tco[j_int + 1][:, None] - Tco[j_int][:, None]) / dzi[j_int][:, None]
    grav_jm = -1.0 / Hpi[j_int - 1][:, None] + ms[None, :] * g[j_int][:, None] / (Navo * kb * Ti[j_int - 1][:, None]) \
              + alpha[None, :] / Ti[j_int - 1][:, None] * (Tco[j_int][:, None] - Tco[j_int - 1][:, None]) / dzi[j_int - 1][:, None]
    grav_jp_b = -1.0 / Hpi[j_int][:, None] + ms[None, :] * g[j_int + 1][:, None] / (Navo * kb * Ti[j_int][:, None]) \
                + alpha[None, :] / Ti[j_int][:, None] * (Tco[j_int + 1][:, None] - Tco[j_int][:, None]) / dzi[j_int][:, None]
    grav_jm_c = -1.0 / Hpi[j_int - 1][:, None] + ms[None, :] * g[j_int - 1][:, None] / (Navo * kb * Ti[j_int - 1][:, None]) \
                + alpha[None, :] / Ti[j_int - 1][:, None] * (Tco[j_int][:, None] - Tco[j_int - 1][:, None]) / dzi[j_int - 1][:, None]

    Ai_int = -1.0 / dz_ave[:, None] * (
        Dzz[j_int] / dzi[j_int][:, None] * (ysum[j_int + 1][:, None] + ysum[j_int][:, None]) / 2.0
        + Dzz[j_int - 1] / dzi[j_int - 1][:, None] * (ysum[j_int][:, None] + ysum[j_int - 1][:, None]) / 2.0
    ) / ysum[j_int][:, None]
    Ai_int += 1.0 / (2.0 * dz_ave[:, None]) * (Dzz[j_int] * grav_j - Dzz[j_int - 1] * grav_jm)

    Bi_int = 1.0 / dz_ave[:, None] * Dzz[j_int] / dzi[j_int][:, None] * (ysum[j_int + 1][:, None] + ysum[j_int][:, None]) / 2.0 / ysum[j_int + 1][:, None]
    Bi_int += 1.0 / (2.0 * dz_ave[:, None]) * Dzz[j_int] * grav_jp_b

    Ci_int = 1.0 / dz_ave[:, None] * Dzz[j_int - 1] / dzi[j_int - 1][:, None] * (ysum[j_int][:, None] + ysum[j_int - 1][:, None]) / 2.0 / ysum[j_int - 1][:, None]
    Ci_int += -1.0 / (2.0 * dz_ave[:, None]) * Dzz[j_int - 1] * grav_jm_c

    # Boundaries (j = 0 and j = nz-1)
    bdry0_term = (
        1.0 / dzi[0] * Dzz[0] / 2.0
        * (-1.0 / Hpi[0] + ms * g[0] / (Navo * kb * Ti[0])
           + alpha / Ti[0] * (Tco[1] - Tco[0]) / dzi[0])
    )
    Ai_0 = -1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0] + bdry0_term
    Bi_0 = 1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1] + bdry0_term
    Ci_0 = jnp.zeros(ni)

    bdry_top_term = (
        -1.0 / dzi[-1] * Dzz[-1] / 2.0
        * (-1.0 / Hpi[-1] + ms * g[-1] / (Navo * kb * Ti[-1])
           + alpha / Ti[-1] * (Tco[-1] - Tco[-2]) / dzi[-1])
    )
    Ai_top = -1.0 / dzi[-1] * (Dzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-1] + bdry_top_term
    Ci_top = 1.0 / dzi[-1] * (Dzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-2] + bdry_top_term
    Bi_top = jnp.zeros(ni)

    A_mol = jnp.concatenate([Ai_0[None], Ai_int, Ai_top[None]], axis=0)
    B_mol = jnp.concatenate([Bi_0[None], Bi_int, Bi_top[None]], axis=0)
    C_mol = jnp.concatenate([Ci_0[None], Ci_int, Ci_top[None]], axis=0)

    return A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, ysum


def _apply_diffusion_jax(y, A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol):
    """Compute diff[nz, ni] = (A+Ai)*y[j] + (B+Bi)*y[j+1] + (C+Ci)*y[j-1]."""
    A_total = A_eddy[:, None] + A_mol
    B_total = B_eddy[:, None] + B_mol
    C_total = C_eddy[:, None] + C_mol
    nz = y.shape[0]
    diff_0 = A_total[0] * y[0] + B_total[0] * y[1]
    diff_top = A_total[-1] * y[-1] + C_total[-1] * y[-2]
    diff_int = (
        A_total[1:-1] * y[1:-1]
        + B_total[1:-1] * y[2:]
        + C_total[1:-1] * y[:-2]
    )
    return jnp.concatenate([diff_0[None], diff_int, diff_top[None]], axis=0)


@jax.jit
def jax_ros2_step(y, k_arr, dt, atm: AtmStatic, net: NetworkArrays):
    """One JIT'd 2nd-order Rosenbrock step. Pure JAX.

    Args:
        y:    (nz, ni)
        k_arr:(nr+1, nz)
        dt:   scalar
        atm:  AtmStatic
        net:  NetworkArrays

    Returns:
        sol: (nz, ni)
        delta_arr: (nz, ni)
    """
    r = 1.0 + 1.0 / jnp.sqrt(2.0)
    c0 = 1.0 / (r * dt)
    nz = atm.Tco.shape[0]
    ni = atm.ms.shape[0]
    M = atm.M

    # Diffusion coefficients (depend on y via ysum)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = _build_diff_coeffs_jax(y, atm)

    diff_at_y = _apply_diffusion_jax(y, A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol)
    rhs_y = chem_rhs(y, M, k_arr, net) + diff_at_y                       # (nz, ni)
    chem_J = chem_jac(y, M, k_arr, net)                                  # (nz, ni, ni)

    # Diffusion block diagonals (diagonal-in-species)
    diag_d = A_eddy[:, None] + A_mol                                     # (nz, ni)
    sup_d = B_eddy[:-1, None] + B_mol[:-1]                               # (nz-1, ni)
    sub_d = C_eddy[1:, None] + C_mol[1:]                                 # (nz-1, ni)

    # LHS = c0*I - chem_J - diff_J
    eye = jnp.eye(ni)
    di = jnp.arange(ni)
    diag = c0 * eye[None] - chem_J
    diag = diag.at[:, di, di].add(-diag_d)
    sup = jnp.zeros((nz - 1, ni, ni)).at[:, di, di].set(-sup_d)
    sub = jnp.zeros((nz - 1, ni, ni)).at[:, di, di].set(-sub_d)

    # First Rosenbrock stage
    k1 = block_thomas(diag, sup, sub, rhs_y)

    # Second stage at yk2
    yk2 = y + k1 / r
    A_eddy2, B_eddy2, C_eddy2, A_mol2, B_mol2, C_mol2, _ = _build_diff_coeffs_jax(yk2, atm)
    diff_at_yk2 = _apply_diffusion_jax(yk2, A_eddy2, B_eddy2, C_eddy2, A_mol2, B_mol2, C_mol2)
    rhs_yk2 = chem_rhs(yk2, M, k_arr, net) + diff_at_yk2

    rhs2 = rhs_yk2 - (2.0 / (r * dt)) * k1
    k2 = block_thomas(diag, sup, sub, rhs2)

    sol = y + (3.0 / (2.0 * r)) * k1 + (1.0 / (2.0 * r)) * k2
    delta_arr = jnp.abs(sol - yk2)
    return sol, delta_arr


def make_atm_static(atm, ni: int, nz: int) -> AtmStatic:
    """Convert a VULCAN store.AtmData into the JAX AtmStatic struct."""
    return AtmStatic(
        Kzz=jnp.asarray(atm.Kzz),
        Dzz=jnp.asarray(atm.Dzz),
        dzi=jnp.asarray(atm.dzi),
        vz=jnp.asarray(atm.vz),
        Hpi=jnp.asarray(atm.Hpi),
        Ti=jnp.asarray(atm.Ti),
        Tco=jnp.asarray(atm.Tco),
        g=jnp.asarray(atm.g),
        ms=jnp.asarray(atm.ms),
        alpha=jnp.asarray(atm.alpha),
        M=jnp.asarray(atm.M),
    )
