"""Direct steady-state residual of the full physical RHS (Route B B0-5).

`direct_residual` evaluates F(y, theta) = projected chemistry + transport
(+ smooth rainout) at a state and returns scaled per-cell residuals; it is
the G1 CONVERGENCE DIAGNOSTIC of the smooth-rainout plan — a small scaled
||F|| certifies stationarity of the open-system solution. It is NEVER the
sensitivity operator: production derivatives use the solver fixed-point
form (I - G_y) s = G_theta (plan D9), not this residual's Jacobian.

Scaling: R[z, i] = |F[z, i]| / max(y[z, i], mtol_conv * n_0[z]) is a
per-cell inverse timescale [s^-1] — the reciprocal of how long the cell
could sustain its current net tendency before changing by O(1). Cells
whose mixing ratio is below `mtol_conv` are masked (same floor philosophy
as the runner's longdy), as are Dirichlet-enforced cells (bottom pins,
fix_species rows): their residual is definitionally nonzero because an
enforcement operator, not the RHS, holds them.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp

from .jax_step import (
    AtmStatic,
    _apply_diffusion_jax,
    _build_diff_coeffs_jax,
    _projected_chem_rhs,
    compute_diff_grav,
)
from . import conden as _conden_mod


class ResidualReport(NamedTuple):
    """Direct-residual evaluation at one state.

    `F` is the raw RHS [cm^-3 s^-1]; `R` the scaled residual [s^-1] with
    masked cells set to 0; `mask` marks the cells that were EXCLUDED
    (below-floor or Dirichlet-enforced); `max_R` / `argmax_z` / `argmax_i`
    locate the worst live cell.
    """

    F: jnp.ndarray  # (nz, ni)
    R: jnp.ndarray  # (nz, ni)
    mask: jnp.ndarray  # (nz, ni) bool — True = excluded from R
    max_R: jnp.ndarray  # ()
    argmax_z: jnp.ndarray  # () int32
    argmax_i: jnp.ndarray  # () int32


def direct_residual(
    y: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm_step: AtmStatic,
    *,
    n_0: jnp.ndarray,
    mtol_conv: float,
    rainout: Optional[_conden_mod.RainoutTerm] = None,
    exclude_mask: Optional[jnp.ndarray] = None,
) -> ResidualReport:
    """Evaluate the scaled direct residual of the full RHS at `y`.

    `atm_step` must be the LIVE step geometry (the converged carry's
    g/dzi/Hpi/top_flux/vs spliced in — see `residual_from_state`), `n_0`
    the (nz,) total number density, `rainout` the smooth-rainout term when
    conden_mode="smooth_rainout" (None otherwise), and `exclude_mask` an
    optional (nz, ni) bool of Dirichlet-enforced cells.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    grav = compute_diff_grav(atm_step)
    A_e, B_e, C_e, A_m, B_m, C_m, _ = _build_diff_coeffs_jax(y, atm_step, grav)
    diff = _apply_diffusion_jax(y, A_e, B_e, C_e, A_m, B_m, C_m, atm_step)
    F = _projected_chem_rhs(y, atm_step.M, k_arr) + diff
    if rainout is not None:
        n_rain = y @ rainout.sp_mask
        L, _ = _conden_mod.smooth_rainout_loss(
            n_rain, rainout.C, rainout.n_sat, rainout.w
        )
        F = F - rainout.sp_mask[None, :] * L[:, None]

    floor = mtol_conv * n_0[:, None]
    ymix = y / jnp.sum(y, axis=1, keepdims=True)
    below_floor = ymix < mtol_conv
    mask = below_floor if exclude_mask is None else (below_floor | exclude_mask)
    R = jnp.where(mask, 0.0, jnp.abs(F) / jnp.maximum(y, floor))
    flat = jnp.argmax(R)
    ni = y.shape[1]
    return ResidualReport(
        F=F,
        R=R,
        mask=mask,
        max_R=jnp.max(R),
        argmax_z=(flat // ni).astype(jnp.int32),
        argmax_i=(flat % ni).astype(jnp.int32),
    )


def residual_from_state(integ, state, atm_static: AtmStatic) -> ResidualReport:
    """Direct residual of a finished runner state, with the same geometry,
    rainout term, and enforcement masks the runner used.

    `integ` is the `OuterLoop` that produced `state` (reads `_statics` for
    the mode/masks). Splices the converged refresh geometry into
    `atm_static` (and recomputes `vm` when `use_vm_mol`, mirroring the
    runner), builds the rainout term from the ProfileVars carry in
    smooth-rainout mode, and excludes the Dirichlet-enforced cells:
    fix_sp_bot / fix_all_bot / tripped-hycean bottom rows, active
    fix_species pins, and the electron column under use_ion.
    """
    st = integ._statics
    if st is None:
        raise ValueError(
            "residual_from_state: the runner has not been built/run — "
            "integrate first (integ(rs) or integ._runner(state, atm_static))."
        )
    s = state
    atm_step = atm_static._replace(
        g=s.g, dzi=s.dzi, Hpi=s.Hpi, top_flux=s.top_flux, vs=s.vs
    )
    if bool(st.use_vm_mol) and integ._refresh_static is not None:
        from . import atm_refresh as _ar

        atm_step = atm_step._replace(
            vm=_ar.recompute_vm_jax(
                s.g,
                s.Hpi,
                s.dzi,
                atm_static.Dzz,
                atm_static.ms,
                atm_static.alpha,
                atm_static.Tco,
                integ._refresh_static.kb,
                integ._refresh_static.Navo,
            )
        )

    if bool(st.use_smooth_rainout):
        rainout = _conden_mod.RainoutTerm(
            C=float(st.rainout_scale)
            * float(st.rainout_coeff)
            * s.pv.c_Dg_per_re[int(st.rainout_re_row)],
            n_sat=s.pv.c_sat_n_per_re[int(st.rainout_re_row)],
            w=float(st.rainout_w),
            sp_mask=st.rainout_sp_mask,
        )
    else:
        rainout = None

    nz, ni = s.y.shape
    excl = jnp.zeros((nz, ni), dtype=bool)
    if bool(st.use_fix_all_bot):
        excl = excl.at[0, :].set(True)
    if bool(st.use_fix_sp_bot):
        excl = excl.at[0, st.fix_sp_bot_idx].set(True)
    if bool(st.use_fix_H2He):
        pinned = jnp.asarray(s.h2he_pinned)
        excl = excl.at[0, st.h2_idx].set(excl[0, st.h2_idx] | pinned)
        excl = excl.at[0, st.he_idx].set(excl[0, st.he_idx] | pinned)
    if bool(st.use_fix_species):
        excl = excl | s.fix_mask
    if bool(st.use_ion):
        excl = excl.at[:, st.e_idx].set(True)

    return direct_residual(
        s.y,
        s.k_arr,
        atm_step,
        n_0=s.pv.n_0,
        mtol_conv=float(st.mtol_conv),
        rainout=rainout,
        exclude_mask=excl,
    )
