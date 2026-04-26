"""Pure-JAX outer-integration foundation (Phase 10.1 + 10.2 + 10.3 + 10.4).

Replaces the per-accepted-step Python retry loop in `op.Ros2.one_step`
(`VULCAN-master/op.py:3094-3106`) with a JIT'd `jax.lax.while_loop` that
performs one accepted Ros2 step (with internal accept/reject retries fully in
JAX).

Phase 10.2 adds: the photochemistry update (`compute_tau` / `compute_flux` /
`compute_J` and the `var.k` rewrite they drive) runs inside the JAX runner,
fired conditionally per batch via `jax.lax.cond` on a `do_photo` flag computed
in the outer Python loop (matches `op.py:825`'s `count % update_photo_frq == 0`
gate). With `batch_steps=1` the photo branch executes once per accepted step
when the gate fires — same semantics as the upstream NumPy path, but the
photo state (`tau`, `aflux`, `dflux_u`, ...) and the rate-coefficient table
(`k_arr`) live on device between calls so there is no NumPy boundary on the
hot path.

Phase 10.3 adds: `update_mu_dz` and `update_phi_esc` (atmosphere geometry +
diffusion-limited escape flux) now run inside the JAX runner via
`atm_refresh.update_mu_dz_jax` / `update_phi_esc_jax`, fired by a
`lax.cond(do_atm_refresh, ...)` mirroring the same `count % update_frq == 0`
cadence as `op.py:904`. The hydrostatic balance line (`var.y = n_0 * var.ymix`,
`op.py:908-914`) moves into the body — it runs on every accepted step against
the static `n_0` (= `atm.M`). Geometry fields (`g`, `Hp`, `dz`, `dzi`, `Hpi`,
`zco`, `mu`, `top_flux`) live in the carry between batches and are spliced into
`AtmStatic` per body iteration so `jax_ros2_step` always sees the freshest
diffusion coefficients.

Phase 10.4 adds: `op.conden` (condensation rate update) and the optional
`op.h2o_conden_evap_relax` / `op.nh3_conden_evap_relax` cold-trap relaxation
kernels now run inside the JAX runner via `conden.update_conden_rates` /
`apply_h2o_relax_jax` / `apply_nh3_relax_jax`. They fire on every accepted
body iteration, gated by a runtime `do_conden` flag (true when
`use_condense and t >= start_conden_time and not para.fix_species_start`,
matching `op.py:856`). For HD189 (`use_condense=False`) the conden branch
is omitted from the runner entirely and the path is bit-identical to 10.3.

History recording, adaptive rtol, fix-species mid-run trigger handling, and
convergence checking still run Python-side at the `OuterLoop.__call__` level
(Phase 10.5 / 10.6 move them inward).

`OuterLoop` inherits from `op.Integration` to reuse its NumPy helpers
(`backup`, `save_step`, `f_dy`, `update_mu_dz`, `update_phi_esc`, `conden`,
`stop`, `conv`) verbatim — the only override is `__call__`, where the inner
`one_step` retry loop becomes a single call into the JIT'd JAX runner.

Numerical contract:
    The JAX runner is mathematically identical to one iteration of
    `op.Ros2.one_step` followed by `op.Ros2.step_size` — same accept/reject
    decision, same dt update formula, same forced-accept fallback when
    `dt < dt_min`. The Phase 10.2 photo branch is bit-equivalent to the
    Python `Ros2JAX.compute_{tau,flux,J}` calls it replaces (same JAX
    kernels, just no Python boundary).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

import op as _op_master
import vulcan_cfg
import phy_const as _phy_const
from chem_funs import spec_list as _spec_list

import network as _net_mod
import chem as _chem_mod
import photo as _photo_mod
import atm_refresh as _atm_refresh_mod
import conden as _conden_mod
from jax_step import AtmStatic, jax_ros2_step, make_atm_static

jax.config.update("jax_enable_x64", True)


# Parse the network once at module import (matches op_jax.py convention).
_NETWORK = _net_mod.parse_network(vulcan_cfg.network)
_NET_JAX = _chem_mod.to_jax(_NETWORK)


# ---------------------------------------------------------------------------
# Carry state for the JAX while_loop
# ---------------------------------------------------------------------------
class JaxIntegState(NamedTuple):
    """Pure-JAX carry for the inner accept/reject loop.

    Shapes (HD189 reference): nz=120, ni=93, n_atoms=6, nbin~2000, n_br~30.
    All scalars are float64 unless noted. Counts are int32. Photo fields
    use placeholder shape (1, 1) when `use_photo=False` (in that case the
    photo branch is omitted from the runner entirely, so they never change).
    """
    y:           jnp.ndarray   # (nz, ni)        current proposed state
    y_prev:      jnp.ndarray   # (nz, ni)        last accepted state (revert target on reject)
    ymix:        jnp.ndarray   # (nz, ni)        mixing ratios
    dt:          jnp.ndarray   # ()              step size to use for the next attempt
    t:           jnp.ndarray   # ()              elapsed integration time
    delta:       jnp.ndarray   # ()              truncation-error proxy of last attempt
    accept_count:jnp.ndarray   # ()  int32       accepted steps in this batch
    retry_count: jnp.ndarray   # ()  int32       retries on the in-flight step
    atom_loss:      jnp.ndarray  # (n_atoms,)
    atom_loss_prev: jnp.ndarray  # (n_atoms,)
    nega_count:  jnp.ndarray   # ()  int32       cumulative this batch
    loss_count:  jnp.ndarray   # ()  int32
    delta_count: jnp.ndarray   # ()  int32
    small_y:     jnp.ndarray   # ()              cumulative |y| of clipped-small cells
    nega_y:      jnp.ndarray   # ()              cumulative |y| of clipped-negative cells

    # Phase 10.2: photo state (lives in device memory between batches).
    # All zeros / unused when use_photo=False.
    k_arr:        jnp.ndarray  # (nr+1, nz)     reaction-rate table (was a Python dict)
    tau:          jnp.ndarray  # (nz+1, nbin)   optical depth
    aflux:        jnp.ndarray  # (nz, nbin)     actinic flux
    sflux:        jnp.ndarray  # (nz+1, nbin)   direct beam
    dflux_d:      jnp.ndarray  # (nz+1, nbin)   diffuse downward
    dflux_u:      jnp.ndarray  # (nz+1, nbin)   diffuse upward (carries between calls)
    prev_aflux:   jnp.ndarray  # (nz, nbin)     prior aflux (for aflux_change)
    aflux_change: jnp.ndarray  # ()             max relative aflux change
    J_br:         jnp.ndarray  # (n_br, nz)     per-branch J-rate (non-T)
    J_br_T:       jnp.ndarray  # (n_br_T, nz)   per-branch J-rate (T-dep)

    # Phase 10.3: atmosphere geometry (refreshed every `update_frq` accepted
    # steps via `lax.cond(do_atm_refresh, atm_refresh_branch, ...)`).
    # `g`, `dzi`, `Hpi` are spliced into the static `AtmStatic` per body
    # iteration so `jax_ros2_step` sees the freshest diffusion coefficients.
    g:            jnp.ndarray  # (nz,)          gravity
    mu:           jnp.ndarray  # (nz,)          mean molar mass (g/mol)
    Hp:           jnp.ndarray  # (nz,)          pressure scale height
    dz:           jnp.ndarray  # (nz,)          layer thickness
    zco:          jnp.ndarray  # (nz+1,)        interface heights
    dzi:          jnp.ndarray  # (nz-1,)        interface dz
    Hpi:          jnp.ndarray  # (nz-1,)        interface scale height
    top_flux:     jnp.ndarray  # (ni,)          diffusion-limited escape flux at TOA


# ---------------------------------------------------------------------------
# Photo statics — closed over by the photo branch closure
# ---------------------------------------------------------------------------
class _PhotoStatic(NamedTuple):
    """Per-run static inputs to the photo branch closure.

    `dz`, `bins`, `sflux_top` are physically static at OuterLoop init; if any
    of them ever change mid-run (e.g., atm refresh in Phase 10.3 mutates dz),
    the runner cache must be invalidated and rebuilt.
    """
    photo_data:        _photo_mod.PhotoData     # absp / scat cross sections
    photo_J_data:      _photo_mod.PhotoJData    # J cross sections (passed for branch_keys)
    cross_J:           jnp.ndarray              # (n_br, nbin)
    cross_J_T:         jnp.ndarray              # (n_br_T, nz, nbin)
    branch_re_idx:     jnp.ndarray              # (n_br,)   int64 — k_arr row to write
    branch_active:     jnp.ndarray              # (n_br,)   bool
    branch_T_re_idx:   jnp.ndarray              # (n_br_T,) int64
    branch_T_active:   jnp.ndarray              # (n_br_T,) bool
    bins:              jnp.ndarray              # (nbin,)   wavelength grid (nm)
    sflux_top:         jnp.ndarray              # (nbin,)   TOA stellar flux
    dz:                jnp.ndarray              # (nz,)     layer thickness
    din12_indx:        int                      # static — wavelength split index for J integration
    dbin1:             float
    dbin2:             float
    mu_zenith:         float                    # cos(sl_angle)
    edd:               float                    # Eddington coefficient
    ag0:               float                    # asymmetry factor (0 for HD189)
    hc:                float                    # planck * c (erg nm)
    f_diurnal:         float                    # diurnal flux average (1.0 tidally locked)
    flux_atol:         float                    # aflux_change masking floor
    ag0_is_zero:       bool                     # static — selects compute_flux branch


# ---------------------------------------------------------------------------
# Pure-JAX numerics: clip / loss / step_size / aggregate_delta
# ---------------------------------------------------------------------------
def _compute_atom_loss(y: jnp.ndarray, compo_arr: jnp.ndarray,
                       atom_ini_arr: jnp.ndarray) -> jnp.ndarray:
    """Vectorized port of `op.ODESolver.loss` (op.py:2475-2490).

    Returns `atom_loss` of shape (n_atoms,). Equivalent to:
        atom_sum[a] = sum over (z, i) of compo[i, a] * y[z, i]
        atom_loss[a] = (atom_sum[a] - atom_ini[a]) / atom_ini[a]
    """
    atom_sum = jnp.einsum("zi,ia->a", y, compo_arr)
    return (atom_sum - atom_ini_arr) / atom_ini_arr


def _step_size(dt: jnp.ndarray, delta: jnp.ndarray,
               rtol: float, dt_var_min: float, dt_var_max: float,
               dt_min: float, dt_max: float) -> jnp.ndarray:
    """JAX port of `op.Ros2.step_size` (op.py:3108-3128).

    h_factor = 0.9 * (rtol/delta)^0.5, clipped to [dt_var_min, dt_var_max];
    h_new = clip(dt * h_factor, dt_min, dt_max).
    The `delta == 0` edge uses 0.01 * rtol as the effective delta (matches
    op.py:3116; avoids division-by-zero and gives a moderate growth factor).
    """
    delta_eff = jnp.where(delta == 0.0, 0.01 * rtol, delta)
    h_factor = 0.9 * (rtol / delta_eff) ** 0.5
    h_factor = jnp.clip(h_factor, dt_var_min, dt_var_max)
    return jnp.clip(dt * h_factor, dt_min, dt_max)


def _make_clip_fn(non_gas_present: bool, gas_indx_mask: jnp.ndarray,
                  mtol: float, pos_cut: float, nega_cut: float):
    """Return a closure `clip(y_in, ymix_old) -> (y_clip, ymix_new, small_inc, nega_inc)`.

    Bit-equivalent to `op.ODESolver.clip` (op.py:2450-2473) plus the ymix
    recompute it does inline. `non_gas_present` selects between the two
    `var.ymix = ...` formulas (op.py:2469 vs 2470); we pick at closure time
    so the JIT trace stays simple.
    """
    if non_gas_present:
        gas_mask_2d = gas_indx_mask  # (ni,) bool

        def clip_fn(y_in, ymix_old):
            small_y_inc = jnp.sum(jnp.where((y_in < pos_cut) & (y_in >= 0), jnp.abs(y_in), 0.0))
            nega_y_inc = jnp.sum(jnp.where((y_in > nega_cut) & (y_in <= 0), jnp.abs(y_in), 0.0))
            y_clip = jnp.where((y_in < pos_cut) & (y_in >= nega_cut), 0.0, y_in)
            y_clip = jnp.where((ymix_old < mtol) & (y_clip < 0), 0.0, y_clip)
            ysum = jnp.sum(jnp.where(gas_mask_2d[None, :], y_clip, 0.0),
                           axis=1, keepdims=True)
            return y_clip, y_clip / ysum, small_y_inc, nega_y_inc
    else:
        def clip_fn(y_in, ymix_old):
            small_y_inc = jnp.sum(jnp.where((y_in < pos_cut) & (y_in >= 0), jnp.abs(y_in), 0.0))
            nega_y_inc = jnp.sum(jnp.where((y_in > nega_cut) & (y_in <= 0), jnp.abs(y_in), 0.0))
            y_clip = jnp.where((y_in < pos_cut) & (y_in >= nega_cut), 0.0, y_in)
            y_clip = jnp.where((ymix_old < mtol) & (y_clip < 0), 0.0, y_clip)
            ysum = jnp.sum(y_clip, axis=1, keepdims=True)
            return y_clip, y_clip / ysum, small_y_inc, nega_y_inc

    return clip_fn


def _make_aggregate_delta_fn(mtol: float, atol: float,
                             zero_bot_row: bool,
                             condense_zero_mask: jnp.ndarray):
    """Return a closure that computes the scalar `delta` from `delta_arr`.

    Mirrors the post-processing in `op_jax.Ros2JAX.solver` (op_jax.py:373-401):
        delta[ymix < mtol] = 0
        delta[sol < atol] = 0
        delta[bot row] = 0  if use_botflux or use_fix_sp_bot
        delta[:, condense_sp] = 0  if use_condense
        delta_pos = delta[sol > 0] / sol[sol > 0]
        delta = max(delta_pos)

    The fix_species mask is deferred to Phase 10.4; for HD189 it's all-False
    and gets baked into `condense_zero_mask` here as zeros.
    """
    cond_zero = jnp.asarray(condense_zero_mask, dtype=jnp.bool_)

    def agg(sol, delta_arr, ymix_new):
        masked = jnp.where(ymix_new < mtol, 0.0, delta_arr)
        masked = jnp.where(sol < atol, 0.0, masked)
        if zero_bot_row:
            row_zero = jnp.zeros_like(masked).at[0].set(1.0).astype(jnp.bool_)
            masked = jnp.where(row_zero, 0.0, masked)
        masked = jnp.where(cond_zero, 0.0, masked)
        # max over (sol > 0) cells; matches op_jax.py:397-401
        ratio = jnp.where(sol > 0,
                          masked / jnp.maximum(jnp.abs(sol), 1e-300),
                          0.0)
        return jnp.max(ratio)

    return agg


# ---------------------------------------------------------------------------
# Photo branch factory — closed over at OuterLoop init
# ---------------------------------------------------------------------------
def _make_photo_branch(photo_static: _PhotoStatic):
    """Build a closure `photo_branch(s) -> s_with_photo_updated`.

    Pure-JAX port of the three Python wrappers in `op_jax.Ros2JAX.compute_tau`
    / `compute_flux` / `compute_J` (op_jax.py:56-127), plus the var.k update
    they drive. Same kernels (`compute_tau_jax` / `compute_flux_jax` /
    `compute_J_jax_flat`); the gain over the Python path is that all photo
    state stays on device between calls and the dict iteration over
    `var.J_sp` is replaced by `update_k_with_J`'s vectorized .at[].set().
    """
    photo_data = photo_static.photo_data
    cross_J = photo_static.cross_J
    cross_J_T = photo_static.cross_J_T
    branch_re_idx = photo_static.branch_re_idx
    branch_active = photo_static.branch_active
    branch_T_re_idx = photo_static.branch_T_re_idx
    branch_T_active = photo_static.branch_T_active
    bins = photo_static.bins
    sflux_top = photo_static.sflux_top
    dz = photo_static.dz
    din12_indx = photo_static.din12_indx
    dbin1 = photo_static.dbin1
    dbin2 = photo_static.dbin2
    mu_zenith = photo_static.mu_zenith
    edd = photo_static.edd
    ag0 = photo_static.ag0
    hc = photo_static.hc
    f_diurnal = photo_static.f_diurnal
    flux_atol = photo_static.flux_atol
    ag0_is_zero = photo_static.ag0_is_zero

    def photo_branch(s: JaxIntegState) -> JaxIntegState:
        # Optical depth (mirrors op.compute_tau via op_jax.Ros2JAX.compute_tau).
        tau_new = _photo_mod.compute_tau_jax(s.y, dz, photo_data)

        # Two-stream RT. `s.dflux_u` is the prior call's value (matches
        # op.py:2694's dflux_u-as-it-stood-before-the-up-sweep).
        aflux_new, sflux_new, dflux_d_new, dflux_u_new = (
            _photo_mod.compute_flux_jax(
                tau_new, sflux_top, s.ymix, photo_data,
                bins, mu_zenith, edd, ag0, hc, s.dflux_u,
                ag0_is_zero=ag0_is_zero,
            )
        )

        # Per-branch J-rates (flat output; no Python dict).
        J_br_new, J_br_T_new = _photo_mod.compute_J_jax_flat(
            aflux_new, cross_J, cross_J_T, din12_indx, dbin1, dbin2
        )

        # Write into k_arr: k_arr[re_idx] = J * f_diurnal for each active branch.
        k_arr_new = _photo_mod.update_k_with_J(
            s.k_arr, J_br_new, J_br_T_new,
            branch_re_idx, branch_active,
            branch_T_re_idx, branch_T_active,
            f_diurnal,
        )

        # aflux_change: mirrors op.py:2740 / op_jax.py:94-101.
        # `s.aflux` here is the OLD aflux; we use it as `prev_aflux` in the
        # ratio. After this branch, prev_aflux <- old aflux, aflux <- new.
        mask = aflux_new > flux_atol
        diff = jnp.abs(aflux_new - s.aflux)
        ratio = jnp.where(mask,
                          diff / jnp.maximum(jnp.abs(aflux_new), 1e-300),
                          0.0)
        aflux_change_new = jnp.where(jnp.any(mask), jnp.max(ratio),
                                     jnp.float64(0.0))

        return s._replace(
            k_arr=k_arr_new,
            tau=tau_new,
            aflux=aflux_new,
            sflux=sflux_new,
            dflux_d=dflux_d_new,
            dflux_u=dflux_u_new,
            prev_aflux=s.aflux,
            aflux_change=aflux_change_new,
            J_br=J_br_new,
            J_br_T=J_br_T_new,
        )

    return photo_branch


# ---------------------------------------------------------------------------
# Atm-refresh branch factory (Phase 10.3)
# ---------------------------------------------------------------------------
def _make_atm_refresh_branch(refresh_static: _atm_refresh_mod.AtmRefreshStatic):
    """Build a closure `atm_refresh(s) -> s_with_geom_updated`.

    Pure-JAX port of `op.update_mu_dz` + `op.update_phi_esc` (op.py:944-999).
    Recomputes `mu`, `g`, `Hp`, `dz`, `zco`, `dzi`, `Hpi` from the current
    `ymix` (hydrostatic loop split into forward / backward `lax.scan`s
    around `pref_indx`), then refreshes `top_flux` from the updated `g` /
    `Hp`. All updates land in the carry; the next body iteration's
    `jax_ros2_step` picks up the fresh `g` / `dzi` / `Hpi`.
    """
    def atm_refresh(s: JaxIntegState) -> JaxIntegState:
        mu_new, g_new, Hp_new, dz_new, zco_new, dzi_new, Hpi_new = (
            _atm_refresh_mod.update_mu_dz_jax(s.ymix, refresh_static)
        )
        top_flux_new = _atm_refresh_mod.update_phi_esc_jax(
            s.y, g_new, Hp_new, s.top_flux, refresh_static,
        )
        return s._replace(
            mu=mu_new, g=g_new, Hp=Hp_new, dz=dz_new,
            zco=zco_new, dzi=dzi_new, Hpi=Hpi_new,
            top_flux=top_flux_new,
        )
    return atm_refresh


# ---------------------------------------------------------------------------
# Conden branch factory (Phase 10.4)
# ---------------------------------------------------------------------------
def _make_conden_branch(conden_static: _conden_mod.CondenStatic):
    """Build a closure `conden_branch(s) -> s_with_k_and_y_updated`.

    Pure-JAX port of `op.conden` (`op.py:1112-1301`) plus the optional
    `op.h2o_conden_evap_relax` (`op.py:1334-1370`) and
    `op.nh3_conden_evap_relax` (`op.py:1372-1424`) post-step relaxers.

    The closure operates on the post-Ros2 / post-clip values held in
    `s.y`, `s.ymix` (the body_fn writes those before calling the branch).
    `update_conden_rates` overwrites the conden rows of `s.k_arr`; the
    relax kernels then redistribute mass between the gas-phase species
    and its `_l_s` / `_l` companion. Whether each kernel actually runs
    is decided statically by `conden_static.h2o_active` /
    `conden_static.nh3_active` (Python bools selected from
    `vulcan_cfg.use_relax`).
    """
    def conden_branch(s: JaxIntegState) -> JaxIntegState:
        k_arr_new = _conden_mod.update_conden_rates(
            s.k_arr, s.y, conden_static
        )
        y_new, ymix_new = _conden_mod.apply_h2o_relax_jax(
            s.y, s.ymix, s.dt, conden_static
        )
        y_new, ymix_new = _conden_mod.apply_nh3_relax_jax(
            y_new, ymix_new, s.dt, conden_static
        )
        return s._replace(y=y_new, ymix=ymix_new, k_arr=k_arr_new)
    return conden_branch


# ---------------------------------------------------------------------------
# Static aux for the runner (closed over at OuterLoop init)
# ---------------------------------------------------------------------------
class _Statics(NamedTuple):
    """Per-run static inputs to the JAX runner (closed-over, never in carry)."""
    compo_arr:        jnp.ndarray   # (ni, n_atoms)
    atom_ini_arr:     jnp.ndarray   # (n_atoms,)
    rtol:             float
    loss_eps:         float
    pos_cut:          float
    nega_cut:         float
    mtol:             float
    atol:             float
    dt_var_min:       float
    dt_var_max:       float
    dt_min:           float
    dt_max:           float
    batch_steps:      int           # # accepted steps per runner call (Python int -> compile constant)
    batch_max_retries:int           # safety cap on inner retries per accepted step


# ---------------------------------------------------------------------------
# Body / runner factory
# ---------------------------------------------------------------------------
def _make_runner(net, statics: _Statics,
                 non_gas_present: bool,
                 gas_indx_mask: jnp.ndarray,
                 zero_bot_row: bool,
                 condense_zero_mask: jnp.ndarray,
                 hydro_partial: bool,
                 photo_static: Optional[_PhotoStatic] = None,
                 refresh_static: Optional[_atm_refresh_mod.AtmRefreshStatic] = None,
                 conden_static: Optional[_conden_mod.CondenStatic] = None):
    """Build a JIT'd `runner(state, atm_static, do_photo, do_atm_refresh, do_conden) -> state`
    that runs `lax.while_loop` until `state.accept_count >= batch_steps`.

    If `photo_static` is not None, the runner fires a photo update branch
    (`compute_tau` / `compute_flux` / `compute_J` + `var.k` rewrite) before
    the while_loop, gated on the runtime `do_photo` flag (`op.py:825`'s
    `count % update_photo_frq == 0` check).

    If `refresh_static` is not None (Phase 10.3), an atmosphere-refresh
    branch (`update_mu_dz` + `update_phi_esc`) fires after photo, gated on
    the runtime `do_atm_refresh` flag (`op.py:904`'s `count % update_frq == 0`).

    If `conden_static` is not None (Phase 10.4), a condensation branch
    (`update_conden_rates` + optional `h2o_relax` / `nh3_relax`) fires
    inside the body on every accepted step, gated on the runtime
    `do_conden` flag (`op.py:856` — true when `t >= start_conden_time` and
    fix-species hasn't yet flipped).

    Inside the body, `jax_ros2_step` always reads `g`/`dzi`/`Hpi` from the
    carry (spliced into the closed-over `AtmStatic` via `._replace`), so
    the chemistry sees the freshest geometry whenever atm refresh fires.

    All decisions are made Python-side at this Phase 10.4 with `batch_steps=1`;
    once Phase 10.5 grows `batch_steps`, they move into the body via
    `s.accept_count % update_*_frq` etc.
    """
    clip_fn = _make_clip_fn(non_gas_present, gas_indx_mask,
                            statics.mtol, statics.pos_cut, statics.nega_cut)
    agg_delta_fn = _make_aggregate_delta_fn(statics.mtol, statics.atol,
                                            zero_bot_row, condense_zero_mask)

    photo_branch = (_make_photo_branch(photo_static)
                    if photo_static is not None else None)
    atm_refresh_branch = (_make_atm_refresh_branch(refresh_static)
                          if refresh_static is not None else None)
    conden_branch = (_make_conden_branch(conden_static)
                     if conden_static is not None else None)

    rtol = statics.rtol
    loss_eps = statics.loss_eps
    dt_var_min = statics.dt_var_min
    dt_var_max = statics.dt_var_max
    dt_min = statics.dt_min
    dt_max = statics.dt_max
    batch_steps = statics.batch_steps
    batch_max_retries = statics.batch_max_retries
    compo_arr = statics.compo_arr
    atom_ini_arr = statics.atom_ini_arr

    def cond_fn(s: JaxIntegState):
        return s.accept_count < jnp.int32(batch_steps)

    def body_fn(s: JaxIntegState, atm_static_, do_conden):
        # Splice the carry's (potentially refreshed) geometry into the
        # otherwise-static AtmStatic before the chemistry step.
        atm_step = atm_static_._replace(g=s.g, dzi=s.dzi, Hpi=s.Hpi)

        # Attempt one Ros2 step at the current state, using s.k_arr (which
        # may have been refreshed by the photo branch this batch).
        sol, delta_arr = jax_ros2_step(s.y, s.k_arr, s.dt, atm_step, net)

        # Clip (op.py:2450-2473) + recompute ymix + atom_loss.
        sol_clip, ymix_new, small_y_inc, nega_y_inc = clip_fn(sol, s.ymix)
        atom_loss_new = _compute_atom_loss(sol_clip, compo_arr, atom_ini_arr)
        delta = agg_delta_fn(sol_clip, delta_arr, ymix_new)

        # step_ok criteria (op.py:2492-2496).
        all_nonneg = jnp.all(sol_clip >= 0)
        loss_diff = jnp.max(jnp.abs(atom_loss_new - s.atom_loss_prev))
        accept = all_nonneg & (loss_diff < loss_eps) & (delta <= rtol)

        # Force-accept conditions (op.py:2518-2523 + retry budget guard).
        next_dt_if_reject = s.dt * dt_var_min
        dt_underflow = next_dt_if_reject < dt_min
        retry_exhausted = s.retry_count >= jnp.int32(batch_max_retries)
        force_accept = (dt_underflow | retry_exhausted) & ~accept
        do_accept = accept | force_accept

        # Reject counter dispatch (op.py:2500-2513): exactly one increments.
        delta_too_big = delta > rtol
        any_neg = jnp.any(sol_clip < 0)
        is_reject = ~do_accept
        delta_count_inc = (is_reject & delta_too_big).astype(jnp.int32)
        nega_count_inc = (is_reject & ~delta_too_big & any_neg).astype(jnp.int32)
        loss_count_inc = (is_reject & ~delta_too_big & ~any_neg).astype(jnp.int32)

        # Time advance: state.dt for normal accept; dt_min for force_accept; 0 for reject.
        dt_used_for_t = jnp.where(force_accept, jnp.float64(dt_min), s.dt)
        t_next = jnp.where(do_accept, s.t + dt_used_for_t, s.t)

        # dt for the next attempt.
        dt_after_normal = _step_size(s.dt, delta, rtol,
                                     dt_var_min, dt_var_max, dt_min, dt_max)
        dt_after_force = _step_size(jnp.float64(dt_min), delta, rtol,
                                    dt_var_min, dt_var_max, dt_min, dt_max)
        dt_next = jnp.where(force_accept, dt_after_force,
                            jnp.where(accept, dt_after_normal, next_dt_if_reject))

        # Conden branch (Phase 10.4 — op.py:856-902): updates the conden
        # rows of k_arr from sol_clip, then optionally relaxes H2O / NH3
        # toward saturation. Fires on every accepted step inside the
        # conden window (do_conden carries `t >= start_conden_time and
        # not fix_species_start`). For HD189 (use_condense=False) the
        # branch is None and k_arr / sol_clip / ymix_new pass through.
        if conden_branch is not None:
            s_post = s._replace(y=sol_clip, ymix=ymix_new)
            fire_conden = do_accept & do_conden
            s_post = jax.lax.cond(
                fire_conden, conden_branch, lambda ss: ss, s_post,
            )
            sol_clip = s_post.y
            ymix_new = s_post.ymix
            k_arr_next = s_post.k_arr
        else:
            k_arr_next = s.k_arr

        # Hydrostatic balance (op.py:908-914): on accepted step, replace y
        # by n_0 * ymix to enforce the prescribed total density per layer.
        # `n_0 == atm_step.M` is static. For use_condense=True, only the
        # gas columns get rebalanced (the condensable / non-gas columns
        # are left as-is); for HD189 (use_condense=False) all columns get
        # rebalanced. atom_loss / accept decision still uses pre-conden
        # delta (matches op.py: hydro balance fires AFTER conden).
        n_0 = atm_step.M[:, None]
        sol_balanced_full = n_0 * ymix_new
        if hydro_partial:
            # Gas columns get rebalanced; non-gas columns retain sol_clip.
            sol_balanced = jnp.where(gas_indx_mask[None, :],
                                     sol_balanced_full, sol_clip)
        else:
            sol_balanced = sol_balanced_full

        # y / ymix / atom_loss for the next iteration.
        y_prev_clipped = jnp.where(s.y_prev < 0, 0.0, s.y_prev)
        y_next = jnp.where(force_accept, y_prev_clipped,
                           jnp.where(accept, sol_balanced, s.y_prev))
        # ymix consistent with the chosen y. On force_accept we reuse the
        # previously stored ymix (the last accepted ymix == s.ymix); the
        # tiny perturbation from clipping y_prev is reconciled on the next
        # accepted step via clip_fn.
        ymix_next = jnp.where(force_accept, s.ymix,
                              jnp.where(accept, ymix_new, s.ymix))
        atom_loss_next = jnp.where(force_accept, s.atom_loss_prev,
                                   jnp.where(accept, atom_loss_new, s.atom_loss))

        # y_prev is the revert target for in-flight retries. With batch_steps=1
        # (Phase 10.1) it stays at the batch-entry value throughout, which is
        # also what `f_dy` needs as "the y before the step" after the runner
        # returns. Phase 10.5 will revisit when batch_steps>1 enables proper
        # multi-step batches with intra-batch revert targets.
        y_prev_next = s.y_prev
        atom_loss_prev_next = s.atom_loss_prev

        accept_count_next = s.accept_count + jnp.where(do_accept, jnp.int32(1), jnp.int32(0))
        retry_count_next = jnp.where(do_accept, jnp.int32(0), s.retry_count + jnp.int32(1))

        return s._replace(
            y=y_next,
            y_prev=y_prev_next,
            ymix=ymix_next,
            dt=dt_next,
            t=t_next,
            delta=delta,
            accept_count=accept_count_next,
            retry_count=retry_count_next,
            atom_loss=atom_loss_next,
            atom_loss_prev=atom_loss_prev_next,
            nega_count=s.nega_count + nega_count_inc,
            loss_count=s.loss_count + loss_count_inc,
            delta_count=s.delta_count + delta_count_inc,
            small_y=s.small_y + jnp.where(do_accept, small_y_inc, jnp.float64(0.0)),
            nega_y=s.nega_y + jnp.where(do_accept, nega_y_inc, jnp.float64(0.0)),
            k_arr=k_arr_next,
        )

    @jax.jit
    def runner(state: JaxIntegState, atm_static: AtmStatic,
               do_photo: jnp.ndarray, do_atm_refresh: jnp.ndarray,
               do_conden: jnp.ndarray):
        # Photo update first (gated). With batch_steps=1 this is the
        # natural placement: the Ros2 step inside the while_loop sees
        # the freshly-updated s.k_arr. When use_photo=False the closure
        # is None and we drop do_photo entirely.
        if photo_branch is not None:
            state = jax.lax.cond(do_photo, photo_branch,
                                 lambda s: s, state)
        # Atm-refresh (Phase 10.3) fires after photo so the chemistry
        # step sees both freshly-updated rates and freshly-updated
        # geometry in the same batch.
        if atm_refresh_branch is not None:
            state = jax.lax.cond(do_atm_refresh, atm_refresh_branch,
                                 lambda s: s, state)
        # Reset per-batch counters so cond_fn fires.
        state = state._replace(
            accept_count=jnp.int32(0),
            retry_count=jnp.int32(0),
            nega_count=jnp.int32(0),
            loss_count=jnp.int32(0),
            delta_count=jnp.int32(0),
            small_y=jnp.float64(0.0),
            nega_y=jnp.float64(0.0),
        )
        return jax.lax.while_loop(
            cond_fn,
            lambda s: body_fn(s, atm_static, do_conden),
            state,
        )

    return runner


# ---------------------------------------------------------------------------
# OuterLoop class — drop-in replacement for op.Integration
# ---------------------------------------------------------------------------
class OuterLoop(_op_master.Integration):
    """Drop-in for `op.Integration`. Inherits NumPy helpers (backup, save_step,
    f_dy, conv, stop, conden, update_mu_dz, update_phi_esc, ...) verbatim;
    only `__call__` is overridden — the inner per-step Python retry loop is
    replaced by a JIT'd JAX runner.

    Phase 10.1 scope: the inner accepted Ros2 step (with retry) runs in JAX;
    photo update / atm refresh / condensation / save_step / convergence stay
    Python-side at this outer level (they will move into JAX in 10.2-10.5).
    """

    def __init__(self, odesolver, output):
        super().__init__(odesolver, output)
        self._species = list(_NETWORK.species)

        # Atom ordering — captured ONCE at init; dict<->array relies on this.
        self._atom_order = [
            a for a in vulcan_cfg.atom_list
            if a not in getattr(vulcan_cfg, "loss_ex", [])
        ]

        # compo_arr (ni, n_atoms) — same as op_jax.Ros2JAX._build_compo_arr.
        import build_atm as _ba
        compo = _ba.compo
        compo_row = _ba.compo_row
        ni_ = _NETWORK.ni
        compo_np = np.zeros((ni_, len(self._atom_order)), dtype=np.float64)
        for i, sp in enumerate(self._species):
            r_i = compo_row.index(sp)
            for a, atom in enumerate(self._atom_order):
                compo_np[i, a] = float(compo[r_i][atom])
        self._compo_arr = compo_np

        # Static masks. For Phase 10.1 most are degenerate (HD189 default has
        # no condensation, no botflux, no fix_sp_bot); kept as proper arrays
        # so adding the features later is a matter of populating them.
        self._non_gas_present = bool(vulcan_cfg.non_gas_sp)
        self._zero_bot_row = bool(vulcan_cfg.use_botflux or vulcan_cfg.use_fix_sp_bot)

        # condense_zero_mask: built lazily in `_ensure_runner` once `nz`
        # (the atmosphere's vertical grid size) is known.

        # Runner cache — keyed by (nz, k_arr.shape[0]). Populated lazily on the
        # first call; reused for the rest of the run.
        self._runner = None
        self._statics = None
        self._photo_static = None    # populated by _ensure_runner if use_photo
        self._refresh_static = None  # populated by _ensure_runner (Phase 10.3)
        self._conden_static = None   # populated by _ensure_runner if use_condense
        # Hydrostatic balance: True ⇒ only gas columns get rebalanced
        # (use_condense=True). For HD189 (use_condense=False) all columns
        # are rebalanced.
        self._hydro_partial = bool(vulcan_cfg.use_condense)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _build_statics(self, var, atm) -> _Statics:
        """Pack scalar configs and per-run arrays into the closure inputs."""
        atom_ini_arr = np.asarray(
            [float(var.atom_ini[a]) for a in self._atom_order],
            dtype=np.float64,
        )
        return _Statics(
            compo_arr=jnp.asarray(self._compo_arr),
            atom_ini_arr=jnp.asarray(atom_ini_arr),
            rtol=float(vulcan_cfg.rtol),
            loss_eps=float(vulcan_cfg.loss_eps),
            pos_cut=float(vulcan_cfg.pos_cut),
            nega_cut=float(vulcan_cfg.nega_cut),
            mtol=float(self.mtol),
            atol=float(self.atol),
            dt_var_min=float(vulcan_cfg.dt_var_min),
            dt_var_max=float(vulcan_cfg.dt_var_max),
            dt_min=float(vulcan_cfg.dt_min),
            dt_max=float(vulcan_cfg.dt_max),
            batch_steps=int(getattr(vulcan_cfg, "batch_steps", 1)),
            batch_max_retries=int(getattr(vulcan_cfg, "batch_max_retries", 8)),
        )

    def _ensure_runner(self, var, atm) -> None:
        """Build the JIT'd runner on the first call; cached for subsequent."""
        if self._runner is not None:
            return

        nz = atm.Tco.shape[0]
        ni = _NETWORK.ni

        # Gas index mask (ni,) — used only when non_gas_present is True.
        gas_mask_np = np.zeros(ni, dtype=bool)
        if self._non_gas_present and hasattr(atm, "gas_indx"):
            gas_mask_np[np.asarray(atm.gas_indx, dtype=int)] = True
        else:
            gas_mask_np[:] = True
        gas_mask_jnp = jnp.asarray(gas_mask_np)

        # condense_zero_mask (nz, ni) — True where delta should be zeroed.
        # Phase 10.1 only handles use_condense=True via condense_sp + non_gas_sp;
        # default HD189 has both empty so the mask is all False.
        cond_mask_np = np.zeros((nz, ni), dtype=bool)
        if vulcan_cfg.use_condense:
            for sp in vulcan_cfg.condense_sp + vulcan_cfg.non_gas_sp:
                if sp in _NETWORK.species_idx:
                    cond_mask_np[:, _NETWORK.species_idx[sp]] = True

        self._statics = self._build_statics(var, atm)
        self._photo_static = self._build_photo_static(var, atm)
        self._refresh_static = self._build_refresh_static(var, atm)
        self._conden_static = self._build_conden_static(var, atm, gas_mask_jnp)
        self._runner = _make_runner(
            _NET_JAX,
            self._statics,
            self._non_gas_present,
            gas_mask_jnp,
            self._zero_bot_row,
            jnp.asarray(cond_mask_np),
            self._hydro_partial,
            photo_static=self._photo_static,
            refresh_static=self._refresh_static,
            conden_static=self._conden_static,
        )

    def _build_photo_static(self, var, atm) -> Optional[_PhotoStatic]:
        """Pack photo cross sections + scalar configs into a `_PhotoStatic`.

        Returns None if `use_photo=False` (the runner skips the photo branch
        entirely in that case). Reuses the photo data caches from the
        odesolver when available (those got populated by the pre-loop
        `solver.compute_tau` call in `vulcan_jax.py`).
        """
        if not vulcan_cfg.use_photo:
            return None

        # Reuse caches from op_jax.Ros2JAX if pre-loop already built them.
        odesolver = self.odesolver
        photo_data = getattr(odesolver, "_photo_data", None)
        if photo_data is None:
            photo_data = _photo_mod.pack_photo_data(
                var, vulcan_cfg, list(_NETWORK.species)
            )
            odesolver._photo_data = photo_data
        photo_J_data = getattr(odesolver, "_photo_J_data", None)
        if photo_J_data is None:
            photo_J_data = _photo_mod.pack_photo_J_data(var, vulcan_cfg)
            odesolver._photo_J_data = photo_J_data

        (branch_re_idx, branch_active,
         branch_T_re_idx, branch_T_active) = _photo_mod.pack_J_to_k_index_map(
            photo_J_data, var, vulcan_cfg
        )

        ag0 = float(_phy_const.ag0)
        return _PhotoStatic(
            photo_data=photo_data,
            photo_J_data=photo_J_data,
            cross_J=photo_J_data.cross_J,
            cross_J_T=photo_J_data.cross_J_T,
            branch_re_idx=branch_re_idx,
            branch_active=branch_active,
            branch_T_re_idx=branch_T_re_idx,
            branch_T_active=branch_T_active,
            bins=jnp.asarray(var.bins, dtype=jnp.float64),
            sflux_top=jnp.asarray(var.sflux_top, dtype=jnp.float64),
            dz=jnp.asarray(atm.dz, dtype=jnp.float64),
            din12_indx=int(var.sflux_din12_indx),
            dbin1=float(var.dbin1),
            dbin2=float(var.dbin2),
            mu_zenith=float(np.cos(vulcan_cfg.sl_angle)),
            edd=float(vulcan_cfg.edd),
            ag0=ag0,
            hc=float(_phy_const.hc),
            f_diurnal=float(vulcan_cfg.f_diurnal),
            flux_atol=float(vulcan_cfg.flux_atol),
            ag0_is_zero=(ag0 == 0.0),
        )

    def _build_refresh_static(self, var, atm) -> _atm_refresh_mod.AtmRefreshStatic:
        """Pack the static inputs to `atm_refresh.update_mu_dz_jax`.

        Captures the T-P profile, planetary constants, species masses, and
        the reference layer / boundary z-value once at OuterLoop init.
        These never change during integration.
        """
        import build_atm as _ba
        species = _ba.species
        ni = _NETWORK.ni
        nz = atm.Tco.shape[0]
        mol_mass_arr = np.array(
            [_ba.compo['mass'][_ba.compo_row.index(sp)] for sp in species],
            dtype=np.float64,
        )
        diff_esc_idx = np.array(
            [species.index(sp) for sp in vulcan_cfg.diff_esc],
            dtype=np.int32,
        )
        return _atm_refresh_mod.AtmRefreshStatic(
            Tco=jnp.asarray(atm.Tco, dtype=jnp.float64),
            pico=jnp.asarray(atm.pico, dtype=jnp.float64),
            mol_mass=jnp.asarray(mol_mass_arr),
            ms=jnp.asarray(atm.ms, dtype=jnp.float64),
            Dzz_top=jnp.asarray(atm.Dzz[-1], dtype=jnp.float64),
            diff_esc_idx=jnp.asarray(diff_esc_idx),
            pref_indx=int(atm.pref_indx),
            zco_pref=float(atm.zco[atm.pref_indx]),
            gs=float(atm.gs),
            Rp=float(vulcan_cfg.Rp),
            kb=float(_phy_const.kb),
            Navo=float(_phy_const.Navo),
            max_flux=float(vulcan_cfg.max_flux),
            nz=int(nz),
            ni=int(ni),
        )

    def _build_conden_static(self, var, atm, gas_mask_jnp
                             ) -> Optional[_conden_mod.CondenStatic]:
        """Pack the Phase 10.4 conden tables into a `CondenStatic`, or None.

        Returns None when `use_condense=False` (HD189 path) — the runner
        omits the conden branch entirely. Otherwise enumerates
        `var.conden_re_list` and looks up each reaction's gas-phase
        species + saturation profile + diffusion coefficient + per-particle
        mass / radius / density. Reactions whose species is in
        `vulcan_cfg.use_relax` get `coeff_per_re = 0` so
        `update_conden_rates` writes zero rates for them (matches the
        `var.k[re] = 0` short-circuit at op.py:1124-1126).

        H2O / NH3 relax blocks are populated only when the corresponding
        species appears in `vulcan_cfg.use_relax`; otherwise the static
        is degenerate and `apply_h2o_relax_jax` / `apply_nh3_relax_jax`
        short-circuit at trace time via the `*_active` Python bool.
        """
        if not vulcan_cfg.use_condense:
            return None

        species_idx = _NETWORK.species_idx
        nz = atm.Tco.shape[0]
        kb = float(_phy_const.kb)
        Navo = float(_phy_const.Navo)

        # Per-formula physical constants, baked literally from op.conden
        # (op.py:1128-1291). Mass values are kept verbatim including the
        # known oddities (S2 uses 45.019 g/mol per op.py:1206; S8 uses
        # 360.152 per op.py:1252).
        gas_mass_g_per_mol = {
            "H2O": 18.0,
            "NH3": 17.0,
            "H2SO4": 98.022,
            "S2": 45.019,
            "S4": 32.06 * 4,
            "S8": 360.152,
            "C": 12.011,
        }
        gas_to_condensate = {
            "H2O": "H2O_l_s",
            "NH3": "NH3_l_s",
            "H2SO4": "H2SO4_l",
            "S2": "S2_l_s",
            "S4": "S4_l_s",
            "S8": "S8_l_s",
            "C": "C_s",
        }

        # Walk var.conden_re_list and pack one row per reaction.
        relax_set = set(getattr(vulcan_cfg, "use_relax", []) or [])
        conden_re_idx_list, conden_sp_idx_list = [], []
        Dg_rows, sat_n_rows, coeff_list = [], [], []
        for re in var.conden_re_list:
            rf = var.Rf[re]                # 'H2O -> H2O_l_s', etc.
            gas_sp = rf.split(" -> ")[0].strip()
            if gas_sp not in vulcan_cfg.condense_sp:
                # Reaction is in the network but not active in this run.
                # Match op.conden's behavior: leaves k untouched (=0).
                continue
            if gas_sp not in gas_mass_g_per_mol:
                raise NotImplementedError(
                    f"conden formula {rf!r} not yet ported to JAX"
                )
            condensate = gas_to_condensate[gas_sp]
            sp_idx = species_idx[gas_sp]
            m = gas_mass_g_per_mol[gas_sp] / Navo
            r_p = float(atm.r_p[condensate])
            rho_p = float(atm.rho_p[condensate])
            coeff = m / (rho_p * r_p ** 2)
            # use_relax short-circuit (op.py:1124-1126): the relax kernel
            # handles the mass redistribution; conden rates stay 0.
            if gas_sp in relax_set:
                coeff = 0.0
            # Dg = atm.Dzz[:, sp_idx] with [0] reused at the bottom (op.py:1138).
            Dzz_col = np.asarray(atm.Dzz[:, sp_idx], dtype=np.float64)
            Dg = np.insert(Dzz_col, 0, Dzz_col[0])
            # sat_n = sat_p[gas]/kb/Tco; H2O multiplies by humidity (op.py:1132).
            sat_n = (np.asarray(atm.sat_p[gas_sp], dtype=np.float64)
                     / kb / np.asarray(atm.Tco, dtype=np.float64))
            if gas_sp == "H2O":
                sat_n = sat_n * float(vulcan_cfg.humidity)
            conden_re_idx_list.append(int(re))
            conden_sp_idx_list.append(int(sp_idx))
            Dg_rows.append(Dg)
            sat_n_rows.append(sat_n)
            coeff_list.append(coeff)

        if not conden_re_idx_list:
            # Active condensation enabled but no live reactions — runner
            # still wants a non-None static so the branch is built; pack
            # zero-length arrays.
            conden_re_idx = np.zeros(0, dtype=np.int32)
            conden_sp_idx = np.zeros(0, dtype=np.int32)
            Dg_per_re = np.zeros((0, nz), dtype=np.float64)
            sat_n_per_re = np.zeros((0, nz), dtype=np.float64)
            coeff_per_re = np.zeros(0, dtype=np.float64)
        else:
            conden_re_idx = np.asarray(conden_re_idx_list, dtype=np.int32)
            conden_sp_idx = np.asarray(conden_sp_idx_list, dtype=np.int32)
            Dg_per_re = np.stack(Dg_rows, axis=0)
            sat_n_per_re = np.stack(sat_n_rows, axis=0)
            coeff_per_re = np.asarray(coeff_list, dtype=np.float64)

        # H2O relax block (op.py:1334-1370).
        h2o_active = "H2O" in relax_set and "H2O" in species_idx
        if h2o_active:
            sp_h2o = species_idx["H2O"]
            sp_h2o_l_s = species_idx["H2O_l_s"]
            r_p = float(atm.r_p["H2O_l_s"])
            rho_p = float(atm.rho_p["H2O_l_s"])
            h2o_m_over_rho_r2 = (18.0 / Navo) / (rho_p * r_p ** 2)
            Dzz_col = np.asarray(atm.Dzz[:, sp_h2o], dtype=np.float64)
            h2o_Dg = np.insert(Dzz_col, 0, Dzz_col[0])
            h2o_sat = (np.asarray(atm.sat_p["H2O"], dtype=np.float64)
                       / kb / np.asarray(atm.Tco, dtype=np.float64)
                       * float(vulcan_cfg.humidity))
        else:
            sp_h2o = sp_h2o_l_s = 0
            h2o_m_over_rho_r2 = 0.0
            h2o_Dg = np.zeros(nz, dtype=np.float64)
            h2o_sat = np.zeros(nz, dtype=np.float64)

        # NH3 relax block (op.py:1372-1424).
        nh3_active = "NH3" in relax_set and "NH3" in species_idx
        if nh3_active:
            sp_nh3 = species_idx["NH3"]
            sp_nh3_l_s = species_idx["NH3_l_s"]
            r_p = float(atm.r_p["NH3_l_s"])
            rho_p = float(atm.rho_p["NH3_l_s"])
            nh3_m_over_rho_r2 = (17.0 / Navo) / (rho_p * r_p ** 2)
            Dzz_col = np.asarray(atm.Dzz[:, sp_nh3], dtype=np.float64)
            nh3_Dg = np.insert(Dzz_col, 0, Dzz_col[0])
            nh3_sat = (np.asarray(atm.sat_p["NH3"], dtype=np.float64)
                       / kb / np.asarray(atm.Tco, dtype=np.float64))
            # conden_top = argmin(sat_mix['NH3']) = argmin(sat_n / n_0).
            # Static; depends only on the T-p profile.
            n_0_np = np.asarray(atm.n_0, dtype=np.float64)
            sat_mix_np = nh3_sat / n_0_np
            nh3_conden_top = int(np.argmin(sat_mix_np))
        else:
            sp_nh3 = sp_nh3_l_s = 0
            nh3_m_over_rho_r2 = 0.0
            nh3_Dg = np.zeros(nz, dtype=np.float64)
            nh3_sat = np.zeros(nz, dtype=np.float64)
            nh3_conden_top = 0

        return _conden_mod.CondenStatic(
            conden_re_idx=jnp.asarray(conden_re_idx),
            conden_sp_idx=jnp.asarray(conden_sp_idx),
            Dg_per_re=jnp.asarray(Dg_per_re),
            sat_n_per_re=jnp.asarray(sat_n_per_re),
            coeff_per_re=jnp.asarray(coeff_per_re),
            h2o_active=h2o_active,
            h2o_idx=int(sp_h2o),
            h2o_l_s_idx=int(sp_h2o_l_s),
            h2o_Dg=jnp.asarray(h2o_Dg),
            h2o_sat=jnp.asarray(h2o_sat),
            h2o_m_over_rho_r2=float(h2o_m_over_rho_r2),
            nh3_active=nh3_active,
            nh3_idx=int(sp_nh3),
            nh3_l_s_idx=int(sp_nh3_l_s),
            nh3_Dg=jnp.asarray(nh3_Dg),
            nh3_sat=jnp.asarray(nh3_sat),
            nh3_m_over_rho_r2=float(nh3_m_over_rho_r2),
            nh3_conden_top=int(nh3_conden_top),
            n_0=jnp.asarray(atm.n_0, dtype=jnp.float64),
            gas_indx_mask=gas_mask_jnp,
        )

    @staticmethod
    def _pack_k_arr(k_dict, nr: int, nz: int) -> jnp.ndarray:
        """Pack `var.k` dict (keyed 1..nr) into a (nr+1, nz) jnp array."""
        out = np.zeros((nr + 1, nz), dtype=np.float64)
        for i, vec in k_dict.items():
            if 1 <= i <= nr:
                out[i] = np.asarray(vec, dtype=np.float64)
        return jnp.asarray(out)

    def _atom_dict_to_arr(self, d: dict) -> np.ndarray:
        return np.asarray(
            [float(d.get(a, 0.0)) for a in self._atom_order],
            dtype=np.float64,
        )

    def _atom_arr_to_dict(self, arr) -> dict:
        return {a: float(arr[i]) for i, a in enumerate(self._atom_order)}

    def _initial_photo_carry(self, var, atm) -> dict:
        """Build the initial values for the photo fields of JaxIntegState.

        When use_photo=True, reads the seed values from var.{tau,aflux,...}
        (populated by the pre-loop one-shot Python compute_tau / compute_flux
        / compute_J in `vulcan_jax.py`). When use_photo=False, returns
        placeholder (1, 1) zero arrays — the runner ignores them.
        """
        nz = atm.Tco.shape[0]
        nr = _NETWORK.nr
        if self._photo_static is None:
            tiny = jnp.zeros((1, 1), dtype=jnp.float64)
            return dict(
                k_arr=self._pack_k_arr(var.k, nr, nz),
                tau=tiny, aflux=tiny, sflux=tiny,
                dflux_d=tiny, dflux_u=tiny, prev_aflux=tiny,
                aflux_change=jnp.float64(0.0),
                J_br=jnp.zeros((0, nz), dtype=jnp.float64),
                J_br_T=jnp.zeros((0, nz), dtype=jnp.float64),
            )
        nbin = int(var.nbin)
        n_br = int(self._photo_static.cross_J.shape[0])
        n_br_T = int(self._photo_static.cross_J_T.shape[0])
        prev_aflux = (var.prev_aflux if hasattr(var, "prev_aflux")
                      else np.zeros((nz, nbin)))
        # If pre-loop compute_J wasn't called, J_br placeholders are zeros;
        # the first photo branch firing will populate them from aflux.
        return dict(
            k_arr=self._pack_k_arr(var.k, nr, nz),
            tau=jnp.asarray(var.tau, dtype=jnp.float64),
            aflux=jnp.asarray(var.aflux, dtype=jnp.float64),
            sflux=jnp.asarray(var.sflux, dtype=jnp.float64),
            dflux_d=jnp.asarray(var.dflux_d, dtype=jnp.float64),
            dflux_u=jnp.asarray(var.dflux_u, dtype=jnp.float64),
            prev_aflux=jnp.asarray(prev_aflux, dtype=jnp.float64),
            aflux_change=jnp.float64(float(getattr(var, "aflux_change", 0.0))),
            J_br=jnp.zeros((n_br, nz), dtype=jnp.float64),
            J_br_T=jnp.zeros((n_br_T, nz), dtype=jnp.float64),
        )

    def _initial_atm_carry(self, atm) -> dict:
        """Build the initial values for the Phase 10.3 atm-refresh carry fields.

        Seeded from the atmosphere as set up by `f_mu_dz` + `mol_diff` in
        the pre-loop. The refresh branch overwrites these every `update_frq`
        accepted steps; in between, the body iteration reads them straight
        from the carry.
        """
        return dict(
            g=jnp.asarray(atm.g, dtype=jnp.float64),
            mu=jnp.asarray(atm.mu, dtype=jnp.float64),
            Hp=jnp.asarray(atm.Hp, dtype=jnp.float64),
            dz=jnp.asarray(atm.dz, dtype=jnp.float64),
            zco=jnp.asarray(atm.zco, dtype=jnp.float64),
            dzi=jnp.asarray(atm.dzi, dtype=jnp.float64),
            Hpi=jnp.asarray(atm.Hpi, dtype=jnp.float64),
            top_flux=jnp.asarray(atm.top_flux, dtype=jnp.float64),
        )

    def _pack_state(self, var, para, atm) -> JaxIntegState:
        """Build the initial JaxIntegState from current var/para/atm."""
        photo_fields = self._initial_photo_carry(var, atm)
        atm_fields = self._initial_atm_carry(atm)
        return JaxIntegState(
            y=jnp.asarray(var.y, dtype=jnp.float64),
            y_prev=jnp.asarray(var.y_prev if hasattr(var, "y_prev") and var.y_prev is not None else var.y,
                               dtype=jnp.float64),
            ymix=jnp.asarray(var.ymix, dtype=jnp.float64),
            dt=jnp.asarray(float(var.dt), dtype=jnp.float64),
            t=jnp.asarray(float(var.t), dtype=jnp.float64),
            delta=jnp.asarray(float(getattr(para, "delta", 0.0)), dtype=jnp.float64),
            accept_count=jnp.int32(0),
            retry_count=jnp.int32(0),
            atom_loss=jnp.asarray(self._atom_dict_to_arr(var.atom_loss),
                                  dtype=jnp.float64),
            atom_loss_prev=jnp.asarray(self._atom_dict_to_arr(var.atom_loss_prev),
                                       dtype=jnp.float64),
            nega_count=jnp.int32(0),
            loss_count=jnp.int32(0),
            delta_count=jnp.int32(0),
            small_y=jnp.float64(0.0),
            nega_y=jnp.float64(0.0),
            **photo_fields,
            **atm_fields,
        )

    def _unpack_state(self, state: JaxIntegState, var, para, atm) -> None:
        """Write the post-batch JAX state back into the var/para/atm store objects."""
        var.y = np.asarray(state.y, dtype=np.float64)
        var.y_prev = np.asarray(state.y_prev, dtype=np.float64)
        var.ymix = np.asarray(state.ymix, dtype=np.float64)
        var.dt = float(state.dt)
        var.t = float(state.t)
        var.atom_loss = self._atom_arr_to_dict(np.asarray(state.atom_loss))
        var.atom_loss_prev = self._atom_arr_to_dict(np.asarray(state.atom_loss_prev))
        # atom_sum is a derived quantity used for diagnostics elsewhere.
        atom_sum_arr = (np.asarray(state.atom_loss) + 1.0) * np.asarray(self._statics.atom_ini_arr)
        for i, a in enumerate(self._atom_order):
            var.atom_sum[a] = float(atom_sum_arr[i])
        para.delta = float(state.delta)
        # Counters accumulate across batches — increment para from the per-batch counts.
        para.count += int(state.accept_count)
        para.nega_count += int(state.nega_count)
        para.loss_count += int(state.loss_count)
        para.delta_count += int(state.delta_count)
        para.small_y += float(state.small_y)
        para.nega_y += float(state.nega_y)

        # Phase 10.2 photo unpack (only when use_photo=True).
        if self._photo_static is not None:
            var.tau = np.asarray(state.tau, dtype=np.float64)
            var.aflux = np.asarray(state.aflux, dtype=np.float64)
            var.sflux = np.asarray(state.sflux, dtype=np.float64)
            var.dflux_d = np.asarray(state.dflux_d, dtype=np.float64)
            var.dflux_u = np.asarray(state.dflux_u, dtype=np.float64)
            var.prev_aflux = np.asarray(state.prev_aflux, dtype=np.float64)
            var.aflux_change = float(state.aflux_change)
            self._unpack_J_sp(state, var)
            self._unpack_k(state, var)

        # Phase 10.4 conden k unpack: write the conden rate rows back into
        # `var.k` so any downstream output / diagnostics see the latest
        # forward + reverse condensation rates.
        if self._conden_static is not None:
            self._unpack_conden_k(state, var)

        # Phase 10.3 atmosphere geometry unpack: write back to atm so any
        # downstream Python paths (output writers, diagnostics, eventual
        # condensation in 10.4) see the freshest geometry.
        atm.g = np.asarray(state.g, dtype=np.float64)
        atm.mu = np.asarray(state.mu, dtype=np.float64)
        atm.Hp = np.asarray(state.Hp, dtype=np.float64)
        atm.dz = np.asarray(state.dz, dtype=np.float64)
        atm.zco = np.asarray(state.zco, dtype=np.float64)
        atm.dzi = np.asarray(state.dzi, dtype=np.float64)
        atm.Hpi = np.asarray(state.Hpi, dtype=np.float64)
        atm.top_flux = np.asarray(state.top_flux, dtype=np.float64)

    def _unpack_J_sp(self, state: JaxIntegState, var) -> None:
        """Rebuild `var.J_sp` dict from carry's J_br / J_br_T arrays.

        Mirrors the dict population in `op.compute_J` (op.py:2767, 2786):
        per (sp, nbr) entries for nbr>=1, plus a per-species (sp, 0) total.
        Needed by `var.var_save` for the .vul output and by any downstream
        plot scripts.
        """
        nz = state.aflux.shape[0]
        n_branch = var.n_branch
        var.J_sp = {(sp, bn): np.zeros(nz)
                    for sp in var.photo_sp
                    for bn in range(n_branch[sp] + 1)}
        J_br_np = np.asarray(state.J_br, dtype=np.float64)
        J_br_T_np = np.asarray(state.J_br_T, dtype=np.float64)
        for i, key in enumerate(self._photo_static.photo_J_data.branch_keys):
            sp, _ = key
            var.J_sp[key] = J_br_np[i]
            var.J_sp[(sp, 0)] = var.J_sp[(sp, 0)] + J_br_np[i]
        for i, key in enumerate(self._photo_static.photo_J_data.branch_T_keys):
            sp, _ = key
            var.J_sp[key] = J_br_T_np[i]
            var.J_sp[(sp, 0)] = var.J_sp[(sp, 0)] + J_br_T_np[i]

    def _unpack_k(self, state: JaxIntegState, var) -> None:
        """Write photo-updated rate rows back into `var.k` so the Python-side
        condensation / output paths see the latest values."""
        k_arr = np.asarray(state.k_arr, dtype=np.float64)
        for i, key in enumerate(self._photo_static.photo_J_data.branch_keys):
            ridx = var.pho_rate_index.get(key)
            if ridx is not None and ridx not in vulcan_cfg.remove_list:
                var.k[ridx] = k_arr[ridx]
        for i, key in enumerate(self._photo_static.photo_J_data.branch_T_keys):
            ridx = var.pho_rate_index.get(key)
            if ridx is not None and ridx not in vulcan_cfg.remove_list:
                var.k[ridx] = k_arr[ridx]

    def _unpack_conden_k(self, state: JaxIntegState, var) -> None:
        """Write conden-updated forward + reverse rate rows back into `var.k`.

        Mirrors `op.conden`'s `var.k[re] = ...; var.k[re+1] = ...` writes
        (`op.py:1144-1145` and friends) so the .vul output and any
        downstream Python diagnostics see the latest condensation +
        evaporation rates.
        """
        k_arr = np.asarray(state.k_arr, dtype=np.float64)
        re_idx_np = np.asarray(self._conden_static.conden_re_idx, dtype=np.int32)
        for ridx in re_idx_np.tolist():
            var.k[ridx] = k_arr[ridx]
            var.k[ridx + 1] = k_arr[ridx + 1]

    # -----------------------------------------------------------------
    # Outer Python orchestration (drop-in for op.Integration.__call__)
    # -----------------------------------------------------------------
    def __call__(self, var, atm, para, make_atm):
        """Run the integration to convergence / runtime / count cap.

        Mirrors `op.Integration.__call__` (op.py:808-936) line-for-line, with
        one substitution: the inner `self.odesolver.one_step(var, atm, para)`
        call is replaced by a JIT'd JAX runner that does the same work.
        Photo / atm refresh / hydrostatic balance now run inside the runner
        (Phases 10.2 + 10.3); condensation, history, and convergence stay
        Python-side until 10.4 / 10.5.
        """
        # `make_atm` is in the signature only for op.Integration parity (the
        # NumPy `update_mu_dz` used to pull `mol_mass` from it). Phase 10.3
        # captures all the geometry inputs into `_refresh_static` at init,
        # so we no longer need it here.
        del make_atm
        use_print_prog = vulcan_cfg.use_print_prog
        use_live_plot = vulcan_cfg.use_live_plot
        self.loss_criteria = 0.0005

        # Build the JAX runner on first entry — cached for the run.
        self._ensure_runner(var, atm)
        ni = _NETWORK.ni
        nz = atm.Tco.shape[0]

        while not self.stop(var, para, atm):

            var = self.backup(var)

            # ---- Photo frequency switch (op.py:819-823) ----
            if (vulcan_cfg.use_photo
                    and var.longdy < vulcan_cfg.yconv_min * 10.0
                    and var.longdydt < 1.0e-6):
                self.update_photo_frq = vulcan_cfg.final_update_photo_frq
                if not para.switch_final_photo_frq:
                    print("update_photo_frq changed to "
                          + str(vulcan_cfg.final_update_photo_frq) + "\n")
                    para.switch_final_photo_frq = True

            # ---- Photo decision (op.py:825) ----
            # The compute_tau / compute_flux / compute_J + var.k rewrite
            # happens inside the JAX runner (Phase 10.2). Python only
            # decides WHETHER it fires this batch.
            do_photo = bool(
                vulcan_cfg.use_photo
                and para.count % self.update_photo_frq == 0
            )
            if do_photo and vulcan_cfg.use_ion:
                # Photo-ionisation still routed through the Python wrapper
                # (compute_Jion not yet ported — Phase 10.6). For HD189
                # use_ion=False so this path is dormant.
                self.odesolver.compute_Jion(var, atm)

            # ---- Atm-refresh decision (op.py:904) ----
            # update_mu_dz / update_phi_esc + hydrostatic balance run inside
            # the JAX runner (Phase 10.3). Python only decides whether the
            # geometry refresh fires this batch; the hydro-balance step
            # itself runs every accepted iteration unconditionally.
            do_atm_refresh = bool(para.count % vulcan_cfg.update_frq == 0)

            # ---- Conden decision (op.py:856) ----
            # update_conden_rates / h2o_relax / nh3_relax all run inside the
            # JAX runner (Phase 10.4). Python decides whether the branch
            # fires this batch — true when condensation is configured, the
            # conden onset time has passed, and the fix-species switch
            # hasn't yet flipped (Python-side trigger handled below).
            do_conden = bool(
                vulcan_cfg.use_condense
                and var.t >= vulcan_cfg.start_conden_time
                and not para.fix_species_start
            )

            # ---- JAX inner step (replaces op.Ros2.one_step + step_size) ----
            atm_static = make_atm_static(atm, ni, nz)
            init_state = self._pack_state(var, para, atm)
            final_state = self._runner(
                init_state, atm_static,
                jnp.asarray(do_photo, dtype=jnp.bool_),
                jnp.asarray(do_atm_refresh, dtype=jnp.bool_),
                jnp.asarray(do_conden, dtype=jnp.bool_),
            )
            self._unpack_state(final_state, var, para, atm)

            # ---- Adaptive rtol (op.py:836-852) ----
            if vulcan_cfg.use_adapt_rtol and para.count % 10 == 0:
                if max(abs(loss) for loss in var.atom_loss.values()) >= self.loss_criteria:
                    self.loss_criteria *= 2.0
                    vulcan_cfg.rtol *= 0.75
                    vulcan_cfg.rtol = max(vulcan_cfg.rtol, vulcan_cfg.rtol_min)
                    if vulcan_cfg.rtol != vulcan_cfg.rtol_min:
                        print("rtol reduced to " + str(vulcan_cfg.rtol))
                        print("------------------------------------------------------------------")
                    # rtol changed: invalidate the runner cache so closure re-traces.
                    self._runner = None
                    self._ensure_runner(var, atm)
            if (vulcan_cfg.use_adapt_rtol and para.count % 1000 == 0
                    and para.count > 0):
                if max(abs(loss) for loss in var.atom_loss.values()) < 2e-4:
                    vulcan_cfg.rtol *= 1.25
                    vulcan_cfg.rtol = min(vulcan_cfg.rtol, vulcan_cfg.rtol_max)
                    if vulcan_cfg.rtol != vulcan_cfg.rtol_max:
                        print("rtol increased to " + str(vulcan_cfg.rtol))
                        print("------------------------------------------------------------------")
                    self._runner = None
                    self._ensure_runner(var, atm)

            # ---- Condensation (op.py:856-902) ----
            # update_conden_rates / h2o_relax / nh3_relax now run inside the
            # JAX runner (Phase 10.4) gated on `do_conden`. The Python side
            # only handles the one-shot fix-species trigger (op.py:860-895)
            # which switches rtol, zeros the settling velocity, captures the
            # fix-species y values, and finds the cold-trap level. After
            # that fires, `para.fix_species_start = True` causes do_conden
            # to evaluate False forever after, matching upstream semantics.
            if (vulcan_cfg.use_condense and var.t >= vulcan_cfg.start_conden_time
                    and not para.fix_species_start):
                if vulcan_cfg.fix_species and var.t > vulcan_cfg.stop_conden_time:
                    para.fix_species_start = True
                    vulcan_cfg.rtol = vulcan_cfg.post_conden_rtol
                    print("rtol changed to " + str(vulcan_cfg.rtol)
                          + " after fixing the condensaed species.")
                    atm.vs *= 0
                    print("Turn off the settling velocity of all species")
                    var.fix_y = {}
                    for sp in vulcan_cfg.fix_species:
                        var.fix_y[sp] = np.copy(var.y[:, _spec_list.index(sp)])
                        if vulcan_cfg.fix_species_from_coldtrap_lev:
                            if sp in ("H2O_l_s", "H2SO4_l", "NH3_l_s", "S8_l_s"):
                                atm.conden_min_lev[sp] = nz - 1
                            else:
                                sat_rho = atm.n_0 * atm.sat_mix[sp]
                                cond_status = var.y[:, _spec_list.index(sp)] >= sat_rho
                                atm.conden_status = cond_status
                                if list(var.y[cond_status, _spec_list.index(sp)]):
                                    min_sat = np.amin(atm.sat_mix[sp][cond_status])
                                    atm.min_sat = min_sat
                                    atm.conden_min_lev[sp] = np.where(
                                        atm.sat_mix[sp] == min_sat
                                    )[0].item()
                                    print(sp + " is now fixed from "
                                          + "{:.2e}".format(
                                              atm.pco[atm.conden_min_lev[sp]] / 1e6)
                                          + " bar.")
                                else:
                                    print(sp + " not condensed.")
                                    atm.conden_min_lev[sp] = 0
                    # rtol changed: invalidate the runner so it re-traces.
                    self._runner = None
                    self._ensure_runner(var, atm)

            # ---- Atmosphere refresh + hydrostatic balance (op.py:904-914) ----
            # Both run inside the JAX runner now (Phase 10.3): the geometry
            # refresh fires when `do_atm_refresh` is True, and hydro balance
            # runs every accepted body iteration. Outputs land in `var.y` /
            # `atm.{g,mu,Hp,dz,dzi,Hpi,zco,top_flux}` via `_unpack_state`.

            # ---- f_dy (op.py:917) ----
            # var.y_prev is still the batch-entry y (not overwritten by the
            # JAX runner — see body_fn's y_prev_next comment), so f_dy's
            # |y - y_prev| computation is correct.
            var = self.f_dy(var, para)

            # ---- History append (op.py:save_step body, minus t/count which
            # were already advanced inside the JAX runner) ----
            var.y_time.append(np.copy(var.y))
            var.t_time.append(var.t)
            var.atom_loss_time.append(list(var.atom_loss.values()))

            # step_size is already applied inside the JAX runner.

            # ---- Output side effects (op.py:925-934) ----
            if use_print_prog and para.count % vulcan_cfg.print_prog_num == 0:
                self.output.print_prog(var, para)
            if (vulcan_cfg.use_live_flux and vulcan_cfg.use_photo
                    and para.count % vulcan_cfg.live_plot_frq == 0):
                self.output.plot_flux_update(var, atm, para)
            if use_live_plot and para.count % vulcan_cfg.live_plot_frq == 0:
                self.output.plot_update(var, atm, para)
