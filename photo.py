"""JAX-friendly photochemistry: optical depth, two-stream RT, J-rate computation.

Mirrors `op.py:compute_tau` (lines 2583-2603), `compute_flux` (2606-2742),
and `compute_J` (2754-2790).

Design:
  - VULCAN stores cross-sections as Python dicts keyed by species name (and
    sometimes (species, branch_index)). For JAX, we pre-stack these into
    fixed-shape tensors at startup via `pack_photo_data`, then operate on
    those tensors inside JIT'd kernels.
  - `compute_tau_jax` uses `jnp.cumsum` for the top-down accumulation,
    replacing VULCAN's explicit Python `for j in range(nz-1, -1, -1)` loop.
  - `compute_flux_jax` uses two `jax.lax.scan`s for the down/up flux
    propagation, matching the upstream two-stream recursion.
  - `compute_J_jax` / `compute_Jion_jax` are tensor contractions plus the
    same two-bin-spacing trapezoid integration as upstream VULCAN.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class PhotoData(NamedTuple):
    """Pre-stacked cross-section arrays for JAX photochem.

    All arrays are jnp; species lookups are done at startup via index arrays.

    Fields:
      absp_idx:     (n_absp,)              int   - species indices for absorbers
                                                   (excluding T-dependent species)
      absp_cross:   (n_absp, nbin)          float - per-species cross section
      absp_T_idx:   (n_absp_T,)            int   - species indices for T-dep absorbers
      absp_T_cross: (n_absp_T, nz, nbin)   float - per-species per-layer cross section
      scat_idx:     (n_scat,)              int   - scattering species indices
      scat_cross:   (n_scat, nbin)         float - scattering cross section
    """
    absp_idx: jnp.ndarray
    absp_cross: jnp.ndarray
    absp_T_idx: jnp.ndarray
    absp_T_cross: jnp.ndarray
    scat_idx: jnp.ndarray
    scat_cross: jnp.ndarray


def pack_photo_data(var, vulcan_cfg, species_list) -> PhotoData:
    """Pack VULCAN's dict-based cross-section storage into a JAX PhotoData.

    Run once at startup after `op.ReadRate.make_bins_read_cross` populates
    var.cross, var.cross_T, var.cross_scat.
    """
    nbin = var.nbin
    nz = var.tau.shape[0] - 1   # tau is (nz+1, nbin) staggered
    T_cross_sp = list(vulcan_cfg.T_cross_sp or [])
    scat_sp = list(vulcan_cfg.scat_sp or [])

    # Absorbing species (photo_sp + ion_sp), excluding T-dep ones
    absp_sp_all = sorted(set.union(set(var.photo_sp), set(var.ion_sp)))
    absp_sp = [sp for sp in absp_sp_all if sp not in T_cross_sp]
    absp_T_sp = [sp for sp in absp_sp_all if sp in T_cross_sp]

    if absp_sp:
        absp_idx = np.array([species_list.index(sp) for sp in absp_sp], dtype=np.int64)
        absp_cross = np.stack([np.asarray(var.cross[sp]) for sp in absp_sp], axis=0)
    else:
        absp_idx = np.zeros((0,), dtype=np.int64)
        absp_cross = np.zeros((0, nbin), dtype=np.float64)

    if absp_T_sp:
        absp_T_idx = np.array([species_list.index(sp) for sp in absp_T_sp], dtype=np.int64)
        absp_T_cross = np.stack([np.asarray(var.cross_T[sp]) for sp in absp_T_sp], axis=0)
    else:
        absp_T_idx = np.zeros((0,), dtype=np.int64)
        absp_T_cross = np.zeros((0, nz, nbin), dtype=np.float64)

    if scat_sp:
        scat_idx = np.array([species_list.index(sp) for sp in scat_sp], dtype=np.int64)
        scat_cross = np.stack([np.asarray(var.cross_scat[sp]) for sp in scat_sp], axis=0)
    else:
        scat_idx = np.zeros((0,), dtype=np.int64)
        scat_cross = np.zeros((0, nbin), dtype=np.float64)

    return PhotoData(
        absp_idx=jnp.asarray(absp_idx),
        absp_cross=jnp.asarray(absp_cross),
        absp_T_idx=jnp.asarray(absp_T_idx),
        absp_T_cross=jnp.asarray(absp_T_cross),
        scat_idx=jnp.asarray(scat_idx),
        scat_cross=jnp.asarray(scat_cross),
    )


@jax.jit
def compute_tau_jax(y: jnp.ndarray, dz: jnp.ndarray, photo: PhotoData) -> jnp.ndarray:
    """Compute optical depth tau[nz+1, nbin] (top-down cumulative).

    Args:
        y:       (nz, ni)    number densities
        dz:      (nz,)       layer thicknesses
        photo:   PhotoData   pre-stacked cross-section arrays

    Returns:
        tau:     (nz+1, nbin) staggered. tau[nz] = 0 (top boundary), tau[0]
                              is the total column.

    Mirrors op.compute_tau exactly:
        tau[j] = sum over species (y[j,sp] * dz[j] * cross[sp]) + tau[j+1]
    """
    nz, ni = y.shape
    nbin = photo.absp_cross.shape[1]

    # Per-layer absorbing contribution: shape (nz, nbin)
    if photo.absp_idx.shape[0] > 0:
        abs_per_layer = jnp.einsum(
            "js,sb->jb", y[:, photo.absp_idx], photo.absp_cross
        ) * dz[:, None]
    else:
        abs_per_layer = jnp.zeros((nz, nbin))

    if photo.absp_T_idx.shape[0] > 0:
        # T-dependent: cross is (n_absp_T, nz, nbin)
        absT_per_layer = jnp.sum(
            y[:, photo.absp_T_idx][:, :, None] * photo.absp_T_cross.transpose(1, 0, 2),
            axis=1,
        ) * dz[:, None]
        abs_per_layer = abs_per_layer + absT_per_layer

    if photo.scat_idx.shape[0] > 0:
        scat_per_layer = jnp.einsum(
            "js,sb->jb", y[:, photo.scat_idx], photo.scat_cross
        ) * dz[:, None]
        abs_per_layer = abs_per_layer + scat_per_layer

    # Cumulative top-down: tau[j] = sum from j to nz-1 of contributions
    # jnp.cumsum gives 0..j, so flip first to get top-down.
    tau_no_top = jnp.flip(jnp.cumsum(jnp.flip(abs_per_layer, axis=0), axis=0), axis=0)
    # Append zero at the top (tau[nz] = 0)
    tau = jnp.concatenate([tau_no_top, jnp.zeros((1, nbin))], axis=0)
    return tau


from functools import partial as _partial


@_partial(jax.jit, static_argnames=("ag0_is_zero",))
def compute_flux_jax(
    tau: jnp.ndarray,           # (nz+1, nbin) staggered
    sflux_top: jnp.ndarray,     # (nbin,) at TOA
    ymix: jnp.ndarray,          # (nz, ni) for w0 computation
    photo: PhotoData,
    bins: jnp.ndarray,          # (nbin,) wavelength bins (nm)
    mu_zenith: float,           # cos(sl_angle), VULCAN converts to -cos
    edd: float,
    ag0: float,
    hc: float,                  # h*c in erg*nm
    dflux_u_prev: jnp.ndarray,  # (nz+1, nbin) prior up-flux; pass zeros on first call
    ag0_is_zero: bool = True,   # static: select between two-stream branches
):
    """Two-stream Eddington radiative transfer.

    Mirrors `op.compute_flux` (lines 2606-2742) bit-for-bit when
    `dflux_u_prev` is threaded through (the prior call's `dflux_u`). VULCAN's
    down sweep at line 2694 reads var.dflux_u[j] *as it stood after the
    previous call*, since the up sweep that overwrites it runs second. To
    replicate exactly, callers pass the dflux_u from the previous compute_flux
    call as `dflux_u_prev`. On the very first call, `dflux_u_prev = zeros`
    (which is also how VULCAN initializes `var.dflux_u`).

    Implementation: vectorized matrix coefficients + two `jax.lax.scan`s for
    the down/up flux recursion.
    """
    mu_ang = -1.0 * mu_zenith
    nz_plus1, nbin = tau.shape
    nz = nz_plus1 - 1

    # delta_tau (length nz) -- between successive layers
    delta_tau = tau[:-1] - tau[1:]    # (nz, nbin)

    # Single-scattering albedo from current ymix
    if photo.absp_idx.shape[0] > 0:
        tot_abs = jnp.einsum("js,sb->jb", ymix[:, photo.absp_idx], photo.absp_cross)
    else:
        tot_abs = jnp.zeros_like(delta_tau)
    if photo.absp_T_idx.shape[0] > 0:
        absT = jnp.sum(
            ymix[:, photo.absp_T_idx][:, :, None] * photo.absp_T_cross.transpose(1, 0, 2),
            axis=1,
        )
        tot_abs = tot_abs + absT
    if photo.scat_idx.shape[0] > 0:
        tot_scat = jnp.einsum("js,sb->jb", ymix[:, photo.scat_idx], photo.scat_cross)
    else:
        tot_scat = jnp.zeros_like(delta_tau)

    w0 = tot_scat / jnp.maximum(tot_abs + tot_scat, 1e-300)
    w0 = jnp.where(jnp.isnan(w0), 0.0, w0)
    w0 = jnp.minimum(w0, 1.0 - 1e-8)

    # Direct beam (Beer's law). Upstream op.py writes this as
    # `exp(-tau / cos(arccos(-mu_ang)))` — same value as `exp(tau / mu_ang)`
    # since `cos(arccos(x)) = x` and mu_ang = -mu_zenith, so -mu_ang ∈ [0,1]
    # is in the valid arccos range. We drop the trig pair.
    sflux = sflux_top[None, :] * jnp.exp(tau / mu_ang)
    dir_flux = sflux * (-mu_ang)

    # Transmission and matrix coefficients (vectorized over nz, nbin)
    if ag0_is_zero:
        tran = jnp.exp(-1.0 / edd * (1.0 - w0) ** 0.5 * delta_tau)
        zeta_p = 0.5 * (1.0 + (1.0 - w0) ** 0.5)
        zeta_m = 0.5 * (1.0 - (1.0 - w0) ** 0.5)
        ll = -1.0 * w0 / (1.0 / mu_ang ** 2 - 1.0 / edd ** 2 * (1.0 - w0))
        g_p = 0.5 * (ll * (1.0 / edd + 1.0 / mu_ang))
        g_m = 0.5 * (ll * (1.0 / edd - 1.0 / mu_ang))
    else:
        tran = jnp.exp(-1.0 / edd * ((1.0 - w0 * ag0) * (1.0 - w0)) ** 0.5 * delta_tau)
        zeta_p = 0.5 * (1.0 + ((1.0 - w0) / (1.0 - w0 * ag0)) ** 0.5)
        zeta_m = 0.5 * (1.0 - ((1.0 - w0) / (1.0 - w0 * ag0)) ** 0.5)
        ll = ((1.0 - w0) * (1.0 - w0 * ag0) - 1.0) / (
            1.0 / mu_ang ** 2 - 1.0 / edd ** 2 * (1.0 - w0) * (1.0 - w0 * ag0)
        )
        g_p = 0.5 * (
            ll * (1.0 / edd + 1.0 / (mu_ang * (1.0 - w0 * ag0)))
            + w0 * ag0 * mu_ang / (1.0 - w0 * ag0)
        )
        g_m = 0.5 * (
            ll * (1.0 / edd - 1.0 / (mu_ang * (1.0 - w0 * ag0)))
            - w0 * ag0 * mu_ang / (1.0 - w0 * ag0)
        )

    ll = jnp.clip(ll, -1e10, 1e10)

    chi = zeta_m ** 2 * tran ** 2 - zeta_p ** 2          # (nz, nbin)
    xi = zeta_p * zeta_m * (1.0 - tran ** 2)
    phi = (zeta_m ** 2 - zeta_p ** 2) * tran

    i_u = phi * g_p * dir_flux[:-1] - (xi * g_m + chi * g_p) * dir_flux[1:]
    i_d = phi * g_m * dir_flux[1:] - (chi * g_m + xi * g_p) * dir_flux[:-1]

    # Down recursion via lax.scan (j from nz-1 down to 0)
    # dflux_d[j] = 1/chi[j] * (phi[j] * dflux_d[j+1] - xi[j] * dflux_u[j] + i_d[j]/mu_ang)
    # `dflux_u[j]` here refers to the PRIOR call's value (var.dflux_u as-is
    # before the up sweep overwrites it) -- this is how VULCAN's loop is
    # written (op.py:2694). On the first call, dflux_u_prev should be zeros.

    dflux_d_top = jnp.zeros(nbin)    # boundary: zero diffuse flux at TOA

    def down_step(carry, j):
        dflux_u_j = dflux_u_prev[j]   # prior call's dflux_u, matching op.py:2694
        dflux_d_jp1 = carry            # this is dflux_d[j+1]
        dflux_d_j = (1.0 / chi[j]) * (
            phi[j] * dflux_d_jp1 - xi[j] * dflux_u_j + i_d[j] / mu_ang
        )
        return dflux_d_j, dflux_d_j

    # Iterate j from nz-1 down to 0
    js_down = jnp.arange(nz - 1, -1, -1)
    _, dflux_d_seq = jax.lax.scan(down_step, dflux_d_top, js_down)
    # dflux_d_seq is in reverse order: [dflux_d[nz-1], ..., dflux_d[0]]
    dflux_d = jnp.concatenate(
        [dflux_d_seq[::-1], dflux_d_top[None]], axis=0
    )   # (nz+1, nbin); dflux_d[nz] = top boundary = 0

    # Up recursion (j from 1 to nz)
    # dflux_u[j] = 1/chi[j-1] * (phi[j-1] * dflux_u[j-1] - xi[j-1] * dflux_d[j] + i_u[j-1]/mu_ang)
    dflux_u_bot = jnp.zeros(nbin)   # boundary: zero upward at surface

    def up_step(carry, j):
        dflux_u_jm1 = carry
        dflux_u_j = (1.0 / chi[j - 1]) * (
            phi[j - 1] * dflux_u_jm1 - xi[j - 1] * dflux_d[j] + i_u[j - 1] / mu_ang
        )
        return dflux_u_j, dflux_u_j

    js_up = jnp.arange(1, nz + 1)
    _, dflux_u_seq = jax.lax.scan(up_step, dflux_u_bot, js_up)
    dflux_u = jnp.concatenate([dflux_u_bot[None], dflux_u_seq], axis=0)   # (nz+1, nbin)

    # Layer-center average flux
    ave_dir_flux = 0.5 * (sflux[:-1] + sflux[1:])
    tot_flux = (
        ave_dir_flux
        + 0.5 * (dflux_u[:-1] + dflux_u[1:] + dflux_d[1:] + dflux_d[:-1]) / edd
    )

    aflux = tot_flux / (hc / bins[None, :])
    return aflux, sflux, dflux_d, dflux_u


class PhotoJData(NamedTuple):
    """Pre-stacked photolysis cross-section data for compute_J_jax.

    Fields:
      cross_J:       (n_br, nbin)        non-T cross section per (species, branch)
      cross_J_T:     (n_br_T, nz, nbin)  T-dep cross section
      din12_indx:    int                 wavelength index where dbin transitions
      dbin1:         float               bin spacing for wavelengths < dbin_12trans
      dbin2:         float               bin spacing for wavelengths >= dbin_12trans
      branch_keys:   list[(species, branch)]  ordering for non-T branches
      branch_T_keys: list[(species, branch)]  ordering for T-dep branches
    """
    cross_J: jnp.ndarray
    cross_J_T: jnp.ndarray
    din12_indx: int
    dbin1: float
    dbin2: float
    branch_keys: tuple
    branch_T_keys: tuple


def pack_photo_J_data(var, vulcan_cfg) -> PhotoJData:
    """Pack VULCAN's cross_J / cross_J_T dicts into JAX-friendly tensors."""
    T_cross_sp = list(vulcan_cfg.T_cross_sp or [])
    branch_keys = []
    branch_T_keys = []
    for sp in var.photo_sp:
        for nbr in range(1, var.n_branch[sp] + 1):
            if sp in T_cross_sp:
                branch_T_keys.append((sp, nbr))
            else:
                branch_keys.append((sp, nbr))

    nbin = var.nbin
    nz = var.aflux.shape[0]
    if branch_keys:
        cross_J = np.stack([np.asarray(var.cross_J[k]) for k in branch_keys], axis=0)
    else:
        cross_J = np.zeros((0, nbin))
    if branch_T_keys:
        cross_J_T = np.stack([np.asarray(var.cross_J_T[k]) for k in branch_T_keys], axis=0)
    else:
        cross_J_T = np.zeros((0, nz, nbin))

    return PhotoJData(
        cross_J=jnp.asarray(cross_J),
        cross_J_T=jnp.asarray(cross_J_T),
        din12_indx=int(var.sflux_din12_indx),
        dbin1=float(var.dbin1),
        dbin2=float(var.dbin2),
        branch_keys=tuple(branch_keys),
        branch_T_keys=tuple(branch_T_keys),
    )


def pack_photo_ion_data(var, vulcan_cfg) -> PhotoJData:
    """Pack VULCAN's cross_Jion dict into a JAX-friendly ion-rate table.

    Upstream `compute_Jion` does not have a temperature-dependent branch, so
    the returned `PhotoJData` always has an empty `cross_J_T` / `branch_T_keys`
    payload and reuses the same integration kernel as neutral photolysis.
    """
    branch_keys = []
    for sp in var.ion_sp:
        for nbr in range(1, var.ion_branch[sp] + 1):
            branch_keys.append((sp, nbr))

    nbin = var.nbin
    nz = var.aflux.shape[0]
    if branch_keys:
        cross_J = np.stack([np.asarray(var.cross_Jion[k]) for k in branch_keys], axis=0)
    else:
        cross_J = np.zeros((0, nbin))

    return PhotoJData(
        cross_J=jnp.asarray(cross_J),
        cross_J_T=jnp.zeros((0, nz, nbin), dtype=jnp.float64),
        din12_indx=int(var.sflux_din12_indx),
        dbin1=float(var.dbin1),
        dbin2=float(var.dbin2),
        branch_keys=tuple(branch_keys),
        branch_T_keys=tuple(),
    )


@_partial(jax.jit, static_argnames=("din12_indx",))
def _compute_J_inner(aflux, cross_J, cross_J_T, din12_indx, dbin1, dbin2):
    """JIT'd inner loop of compute_J. Returns (J_per_branch, J_per_T_branch)."""
    nz = aflux.shape[0]
    flux1 = aflux[:, :din12_indx]
    flux2 = aflux[:, din12_indx:]

    # Non-T branches: cross_J shape (n_br, nbin)
    if cross_J.shape[0] > 0:
        c1 = cross_J[:, :din12_indx]
        c2 = cross_J[:, din12_indx:]
        # Sum (rectangular) + trapezoid endpoint correction
        # J_per_br[nbr, nz] = sum_b (flux[nz, b] * cross[nbr, b]) * dbin
        J1 = jnp.einsum("jb,nb->nj", flux1, c1) * dbin1
        # Trapezoid correction subtracts half of endpoints
        J1 = J1 - 0.5 * (
            flux1[:, 0][None] * c1[:, 0][:, None]
            + flux1[:, -1][None] * c1[:, -1][:, None]
        ) * dbin1
        J2 = jnp.einsum("jb,nb->nj", flux2, c2) * dbin2
        J2 = J2 - 0.5 * (
            flux2[:, 0][None] * c2[:, 0][:, None]
            + flux2[:, -1][None] * c2[:, -1][:, None]
        ) * dbin2
        J_br = J1 + J2          # (n_br, nz)
    else:
        J_br = jnp.zeros((0, nz))

    # T-dependent branches: cross_J_T shape (n_br_T, nz, nbin)
    if cross_J_T.shape[0] > 0:
        cT1 = cross_J_T[:, :, :din12_indx]
        cT2 = cross_J_T[:, :, din12_indx:]
        # J[nbr, nz] = sum_b flux[nz, b] * cross_T[nbr, nz, b] * dbin
        JT1 = jnp.sum(flux1[None] * cT1, axis=2) * dbin1
        JT1 = JT1 - 0.5 * (
            flux1[:, 0][None] * cT1[:, :, 0]
            + flux1[:, -1][None] * cT1[:, :, -1]
        ) * dbin1
        JT2 = jnp.sum(flux2[None] * cT2, axis=2) * dbin2
        JT2 = JT2 - 0.5 * (
            flux2[:, 0][None] * cT2[:, :, 0]
            + flux2[:, -1][None] * cT2[:, :, -1]
        ) * dbin2
        J_br_T = JT1 + JT2
    else:
        J_br_T = jnp.zeros((0, nz))

    return J_br, J_br_T


def compute_J_jax(aflux: jnp.ndarray, photo_J: PhotoJData):
    """Compute J-rates per (species, branch) via wavelength integration.

    Returns a dict keyed by (species, branch) -> jnp.ndarray of shape (nz,).
    Same schema as VULCAN's `var.J_sp`. Used by `op_jax.Ros2JAX.compute_J` for
    the pre-loop one-shot setup. Inside the JAX body of `outer_loop.OuterLoop`,
    use `compute_J_jax_flat` instead — returns flat arrays so .at[].set()
    can write the rates into `k_arr` without a Python dict round-trip.
    """
    J_br, J_br_T = _compute_J_inner(
        aflux, photo_J.cross_J, photo_J.cross_J_T,
        photo_J.din12_indx, photo_J.dbin1, photo_J.dbin2,
    )

    J_sp = {}
    for k, J_row in zip(photo_J.branch_keys, J_br):
        J_sp[k] = J_row
    for k, J_row in zip(photo_J.branch_T_keys, J_br_T):
        J_sp[k] = J_row
    return J_sp


def compute_Jion_jax(aflux: jnp.ndarray, photo_ion: PhotoJData):
    """Compute photoionization J-rates per (species, branch)."""
    return compute_J_jax(aflux, photo_ion)


@_partial(jax.jit, static_argnames=("din12_indx",))
def compute_J_jax_flat(aflux, cross_J, cross_J_T, din12_indx, dbin1, dbin2):
    """Flat-output version of compute_J_jax: returns (J_br, J_br_T) arrays.

    Avoids the Python dict that compute_J_jax builds at line 410 — that
    dict is opaque to JIT and forces a host round-trip if it's the boundary.
    `outer_loop._make_photo_branch` calls this and feeds J_br / J_br_T
    straight into `update_k_with_J` via .at[].set().

    Args:
        aflux:         (nz, nbin)         actinic flux
        cross_J:       (n_br, nbin)       static photolysis cross sections
        cross_J_T:     (n_br_T, nz, nbin) T-dependent ones
        din12_indx:    static int          wavelength index where dbin transitions
        dbin1, dbin2:  scalars            bin spacings

    Returns:
        J_br:   (n_br,   nz)   per-branch J-rate (non-T)
        J_br_T: (n_br_T, nz)   per-branch J-rate (T-dependent)
    """
    return _compute_J_inner(aflux, cross_J, cross_J_T, din12_indx, dbin1, dbin2)


def compute_Jion_jax_flat(aflux, cross_J, din12_indx, dbin1, dbin2):
    """Flat-output photoionization integration helper."""
    return _compute_J_inner(
        aflux,
        cross_J,
        jnp.zeros((0, aflux.shape[0], cross_J.shape[1]), dtype=aflux.dtype),
        din12_indx,
        dbin1,
        dbin2,
    )[0]


def _pack_branch_to_k_index_map(branch_keys, rate_index, remove_list):
    """Build static branch -> k_arr row index tables for photo or ion updates."""
    remove_set = set(remove_list or [])
    n_br = len(branch_keys)
    re_idx = np.zeros(n_br, dtype=np.int64)
    active = np.zeros(n_br, dtype=bool)
    for i, key in enumerate(branch_keys):
        idx = rate_index.get(key)
        if idx is not None and idx not in remove_set:
            re_idx[i] = int(idx)
            active[i] = True
    return jnp.asarray(re_idx), jnp.asarray(active)


def pack_J_to_k_index_map(photo_J, var, vulcan_cfg):
    """Build static index arrays mapping each branch to its `var.k` reaction index.

    Returns a tuple of four (n_br,) arrays plus their T-branch counterparts:
        branch_re_idx     int64 — reaction index in k_arr (1..nr); 0 if inactive
        branch_active     bool  — True if this branch should write into k_arr
        branch_T_re_idx   int64 — same, for T-dep branches
        branch_T_active   bool

    A branch is "inactive" if it's missing from `var.pho_rate_index` or its
    reaction index is in `vulcan_cfg.remove_list`. Inactive branches are
    handled at update time by writing the existing k_arr[0] value into
    k_arr[branch_re_idx], which is a no-op as long as inactive entries point
    at index 0 (an unused slot since reactions are 1-indexed).
    """
    re_idx, active = _pack_branch_to_k_index_map(
        photo_J.branch_keys,
        var.pho_rate_index,
        vulcan_cfg.remove_list,
    )
    re_T_idx, active_T = _pack_branch_to_k_index_map(
        photo_J.branch_T_keys,
        var.pho_rate_index,
        vulcan_cfg.remove_list,
    )
    return re_idx, active, re_T_idx, active_T


def pack_Jion_to_k_index_map(photo_ion, var, vulcan_cfg):
    """Build branch -> k_arr row index tables for photoionization updates."""
    return _pack_branch_to_k_index_map(
        photo_ion.branch_keys,
        var.ion_rate_index,
        vulcan_cfg.remove_list,
    )


@jax.jit
def update_k_with_J(k_arr, J_br, J_br_T,
                    branch_re_idx, branch_active,
                    branch_T_re_idx, branch_T_active,
                    f_diurnal):
    """Write per-branch J-rates into k_arr via .at[].set().

    Mirrors the inner loop of `op.compute_J` (op.py:2787-2789):
        var.k[var.pho_rate_index[(sp, nbr)]] = var.J_sp[(sp, nbr)] * f_diurnal

    Inactive branches (active mask False) leave k_arr unchanged at their
    re_idx slot. Implementation detail: the .at[].set() writes for inactive
    branches are aimed at index 0 (k_arr[0] is unused; reactions are 1-indexed)
    AND the value is set to the existing k_arr[0] — a self-assignment that
    leaves k_arr untouched. This keeps the scatter shape static.
    """
    J_br_scaled = J_br * f_diurnal
    J_br_T_scaled = J_br_T * f_diurnal

    # For inactive entries, write k_arr[0] back to k_arr[0] — true no-op.
    safe_J_br = jnp.where(branch_active[:, None],
                          J_br_scaled,
                          k_arr[0][None, :])
    safe_J_br_T = jnp.where(branch_T_active[:, None],
                            J_br_T_scaled,
                            k_arr[0][None, :])
    # Single fused scatter: non-T and T-dep branch index sets are partitioned
    # at pack time (disjoint by construction), so we can concatenate and write
    # in one .at[].set() call. Saves one scatter dispatch per photo update.
    combined_idx = jnp.concatenate([branch_re_idx, branch_T_re_idx], axis=0)
    combined_J = jnp.concatenate([safe_J_br, safe_J_br_T], axis=0)
    return k_arr.at[combined_idx].set(combined_J)
