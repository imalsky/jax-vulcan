"""Pure-JAX outer-integration loop (Phases 10.1 → 10.6, complete).

Replaces `op.Integration.__call__` (`VULCAN-master/op.py:783-936`) with a
single JIT'd `jax.lax.while_loop`. After Phase 10.6, `OuterLoop` is a
standalone class — no `op.Integration` parent — and the per-step path is
fully JAX with no NumPy hot-path code.

Per-phase scope (cumulative):

- **10.1**: inner accept/reject retries via `jax.lax.while_loop`; pure-JAX
  ports of `clip` / `loss` / `step_ok` / `step_size`. Drops the
  `op.Ros2` subclass relationship.
- **10.2**: `compute_tau` / `compute_flux` / `compute_J` and the `var.k`
  rewrite move into the body via a `lax.cond`-gated photo branch. Photo
  state (`tau`, `aflux`, `dflux_u`, `k_arr`, ...) lives in the
  `JaxIntegState` carry on device.
- **10.3**: `update_mu_dz` (mu / g / Hp / dz / dzi / Hpi / zco) and
  `update_phi_esc` (diffusion-limited escape flux) move into the body via
  a second `lax.cond` on `do_atm_refresh`. Hydrostatic balance
  (`y = n_0 * ymix`, `op.py:908-914`) runs every accepted step. Geometry
  fields are spliced into `AtmStatic` per body iteration.
- **10.4**: `op.conden` plus the optional `op.h2o_conden_evap_relax` /
  `op.nh3_conden_evap_relax` cold-trap relaxers move into the body via
  `conden.update_conden_rates` / `apply_h2o_relax_jax` /
  `apply_nh3_relax_jax`, gated on `do_accept & t >= start_conden_time`.
- **10.5**: convergence + stop check move into `cond_fn`. y_time / t_time
  history rides as a `(conv_step, nz, ni)` ring buffer in the carry;
  adaptive rtol and the photo-frequency ini→final switch move into the
  body. The runner is one-shot per integration — no Python while-loop.
- **10.6**: ion charge balance (post-step `e[:] = -dot(y, charge_arr)`)
  and `use_fix_all_bot` (post-step bottom-row clamp to chemical-EQ
  mixing ratios) move into the body, gated by Python bools at trace
  time. `op.Integration` parent is dropped — `OuterLoop` is now a
  standalone class with `__init__` / `__call__` and an inline `f_dy`.

Numerical contract:
    The JAX runner is mathematically identical to one iteration of
    `op.Ros2.one_step` followed by `op.Ros2.step_size` — same accept/reject
    decision, same dt update formula, same forced-accept fallback when
    `dt < dt_min`. The photo branch is bit-equivalent to the Python
    `Ros2JAX.compute_{tau,flux,J}` calls it replaces (same JAX kernels,
    no Python boundary).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

import vulcan_cfg
import phy_const as _phy_const

import network as _net_mod
import chem as _chem_mod
import photo as _photo_mod
import atm_refresh as _atm_refresh_mod
import conden as _conden_mod
from jax_step import AtmStatic, jax_ros2_step, make_atm_static
from runtime_validation import validate_runtime_config

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
    Jion_br:      jnp.ndarray  # (n_ion_br, nz) per-branch J-rate (ion)

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
    vs:           jnp.ndarray  # (nz-1, ni)     settling velocity (zeroed post-condense fix)

    # Phase 10.5: convergence + step-history carry. The body appends the
    # post-accept (y, t) into the ring at index `accept_count % conv_step`
    # and `cond_fn` reads `longdy` / `longdydt` to terminate. `rtol` and
    # `loss_criteria` move into the carry so adaptive-rtol updates
    # (`op.py:836-852`) can fire inside the body without retracing the
    # runner. `update_photo_frq` and `is_final_photo_frq` carry the
    # `op.py:819-823` ini→final switch.
    y_time_ring:        jnp.ndarray  # (conv_step, nz, ni) float64
    t_time_ring:        jnp.ndarray  # (conv_step,)        float64
    longdy:             jnp.ndarray  # ()                  float64
    longdydt:           jnp.ndarray  # ()                  float64
    rtol:               jnp.ndarray  # ()                  float64
    loss_criteria:      jnp.ndarray  # ()                  float64
    update_photo_frq:   jnp.ndarray  # ()                  int32
    is_final_photo_frq: jnp.ndarray  # ()                  bool

    # Post-condensation fixed-species state.
    fix_species_started: jnp.ndarray  # ()                  bool
    fix_y:               jnp.ndarray  # (nz, ni)           stored fixed values
    fix_mask:            jnp.ndarray  # (nz, ni)           bool mask of fixed cells
    fix_pfix_idx:        jnp.ndarray  # (n_fix_sp,)        int32 cold-trap indices

    # Hycean H2/He pin (op.py:2938-2944). One-shot snapshot at t>1e6;
    # h2he_mix stores the snapshotted mixing ratios used by the post-step
    # bottom-clamp. When use_fix_H2He=False, these stay at their seed
    # values and the body branch is a static no-op.
    h2he_pinned:        jnp.ndarray   # ()  bool
    h2he_mix:           jnp.ndarray   # (2,) float64  [H2_mix, He_mix]

    # Phase 15: save_evolution capture buffer. When save_evolution=False
    # the buffers are length-1 placeholders that the body never writes.
    y_evo:              jnp.ndarray   # (save_evo_n_max, nz, ni) float64
    t_evo:              jnp.ndarray   # (save_evo_n_max,)        float64
    evo_idx:            jnp.ndarray   # ()  int32  next slot to fill

    # Phase 16: chunked-runner cap. The runner terminates a chunk when
    # `accept_count >= chunk_target`. For single-shot runs we seed
    # `chunk_target = count_max + 1` so the cap never trips and the only
    # terminations are count_max / runtime / converged. The chunked driver
    # in `OuterLoop.run_chunk` updates this between chunks for periodic
    # plot/print/movie callbacks.
    chunk_target:       jnp.ndarray   # ()  int32


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
    photo_ion_data:    Optional[_photo_mod.PhotoJData]
    cross_Jion:        jnp.ndarray              # (n_ion_br, nbin)
    ion_branch_re_idx: jnp.ndarray              # (n_ion_br,) int64 — k_arr row to write
    ion_branch_active: jnp.ndarray              # (n_ion_br,) bool
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
    cross_Jion = photo_static.cross_Jion
    ion_branch_re_idx = photo_static.ion_branch_re_idx
    ion_branch_active = photo_static.ion_branch_active
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

        if cross_Jion.shape[0] > 0:
            Jion_br_new = _photo_mod.compute_Jion_jax_flat(
                aflux_new, cross_Jion, din12_indx, dbin1, dbin2
            )
            k_arr_new = _photo_mod.update_k_with_J(
                k_arr_new,
                Jion_br_new,
                jnp.zeros((0, aflux_new.shape[0]), dtype=aflux_new.dtype),
                ion_branch_re_idx,
                ion_branch_active,
                jnp.zeros((0,), dtype=jnp.int64),
                jnp.zeros((0,), dtype=jnp.bool_),
                f_diurnal,
            )
        else:
            Jion_br_new = s.Jion_br

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
            Jion_br=Jion_br_new,
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
    """Per-run static inputs to the JAX runner (closed-over, never in carry).

    The Phase-10.5 batch of fields (`conv_step`, count/time caps, conv
    masks, photo cadence tunables, rtol bounds) drives the in-runner
    termination check (`cond_fn`) and the in-body adaptive-rtol /
    photo-frequency-switch logic that mirrors `op.py:819-852`.
    """
    compo_arr:        jnp.ndarray   # (ni, n_atoms)
    atom_ini_arr:     jnp.ndarray   # (n_atoms,)
    initial_rtol:     float         # used to seed s.rtol; mid-run rtol lives in the carry
    loss_eps:         float
    pos_cut:          float
    nega_cut:         float
    mtol:             float
    atol:             float
    dt_var_min:       float
    dt_var_max:       float
    dt_min:           float
    dt_max:           float
    batch_max_retries:int           # safety cap on inner retries per accepted step

    # Phase 10.5 convergence + termination
    conv_step:               int           # ring buffer length (vulcan_cfg.conv_step)
    count_min:               int
    count_max:               int
    runtime:                 float
    trun_min:                float
    st_factor:               float
    yconv_cri:               float
    yconv_min:               float
    slope_cri:               float
    flux_cri:                float
    mtol_conv:               float
    conver_ignore_mask:      jnp.ndarray   # (ni,) bool — species to drop from longdy
    condense_zero_conv_mask: jnp.ndarray   # (nz, ni) bool — non_gas_sp columns
    n_0:                     jnp.ndarray   # (nz,) — atm.n_0 for the (y - y_old)/n_0 ratio
    Kzz:                     jnp.ndarray   # (nz-1,) — for slope_min recomputation each step

    # Phase 10.5 photo / adaptive-rtol cadence (statics; the dynamic part
    # rides in the carry as s.update_photo_frq / s.rtol / s.loss_criteria).
    use_photo:                bool
    use_atm_refresh:          bool
    use_conden:               bool
    ini_update_photo_frq:     int
    final_update_photo_frq:   int
    update_frq:               int
    use_adapt_rtol:           bool
    rtol_min:                 float
    rtol_max:                 float
    initial_loss_criteria:    float

    # Phase 10.6 — ion charge balance + fix-all-bot post-step clamp.
    # `use_ion` / `use_fix_all_bot` are Python bools so the body branches
    # at trace time. When the corresponding feature is off, `charge_arr`
    # / `bottom_n` are zero-filled placeholders that the body never reads.
    use_ion:                  bool
    e_idx:                    int           # species index of 'e' (0 if use_ion=False)
    charge_arr:               jnp.ndarray   # (ni,) — compo[i]['e'] over charge_list, 0 elsewhere
    use_fix_all_bot:          bool
    bottom_n:                 jnp.ndarray   # (ni,) — bottom_ymix * n_0[0]; pinned each step
    use_fix_sp_bot:           bool
    fix_sp_bot_idx:           jnp.ndarray   # (n_fix_sp_bot,) int32
    fix_sp_bot_mix:           jnp.ndarray   # (n_fix_sp_bot,)
    # Hycean H2/He bottom-pin (op.py:2938-2944). When use_fix_H2He=True,
    # the body snapshots ymix[0, h2_idx]/ymix[0, he_idx] at t>1e6 and pins
    # them thereafter via the same mechanic as fix_sp_bot. h2_idx/he_idx are
    # -1 sentinels when the species are absent from the network (use_fix_H2He
    # then must be False).
    use_fix_H2He:             bool
    h2_idx:                   int
    he_idx:                   int
    use_fix_species:          bool
    post_conden_rtol:         float
    fix_species_from_coldtrap_lev: bool
    fix_species_idx:          jnp.ndarray   # (n_fix_species,) int32
    fix_species_sat_mix:      jnp.ndarray   # (n_fix_species, nz)
    fix_species_wholecol:     jnp.ndarray   # (n_fix_species,) bool

    # Phase 15: save_evolution time-series capture (op.py:save_step).
    # When `save_evolution=True`, the body appends (y_next, t_next) to the
    # `y_evo` / `t_evo` buffers every `save_evo_frq` accepted steps, up to
    # `save_evo_n_max` total. _unpack_state then publishes the populated
    # prefix into `var.y_time` / `var.t_time`. When False, both knobs stay
    # at zero-length placeholders and the body branch is a static no-op.
    save_evolution:           bool
    save_evo_frq:             int
    save_evo_n_max:           int


# ---------------------------------------------------------------------------
# Body / runner factory
# ---------------------------------------------------------------------------
def _make_runner(net, statics: _Statics,
                 non_gas_present: bool,
                 gas_indx_mask: jnp.ndarray,
                 zero_bot_row: bool,
                 condense_zero_mask: jnp.ndarray,
                 hydro_partial: bool,
                 start_conden_time: float,
                 stop_conden_time: float,
                 photo_static: Optional[_PhotoStatic] = None,
                 refresh_static: Optional[_atm_refresh_mod.AtmRefreshStatic] = None,
                 conden_static: Optional[_conden_mod.CondenStatic] = None):
    """Build a JIT'd `runner(state, atm_static) -> state` that runs the
    full integration to convergence / `count_max` / `runtime`.

    Phase-10.5 architecture: every conditional update — photo, atm refresh,
    conden, adaptive rtol, photo-frequency switch, ring-buffer history,
    convergence test — fires inside the JAX body; the only host call per
    integration is the runner itself. The previous per-batch host sync
    (`batch_steps=1`) goes away.

    Body order, mirroring `op.Integration.__call__` (op.py:813-934):

      1. **photo** (gated `retry_count==0 & accept_count % update_photo_frq == 0`)
      2. **atm_refresh** (gated `retry_count==0 & accept_count % update_frq == 0`)
      3. **ros2 step** + clip + atom_loss + delta + accept decision
      4. **conden** (gated `do_accept & t >= start_conden_time`)
      5. **hydro balance** (uses post-conden ymix)
      6. **ring-buffer append** (on accept; ring_idx = accept_count_next % conv_step)
      7. **conv** — recomputes longdy / longdydt against the ring (`op.conv`,
         op.py:1018-1066)
      8. **adaptive rtol** (op.py:836-852, gated `accept_count_next % {10,1000} == 0`)
      9. **photo-frequency switch** (op.py:819-823, when longdy/longdydt drop
         below the threshold)

    The cond_fn then checks `(t > runtime) | (count > count_max) |
    (ready & converged)` — same outcome as `op.stop` (op.py:1068-1088).
    With photo, atm_refresh, conden, conv, ring buffer, longdy, longdydt,
    rtol, loss_criteria, and update_photo_frq all in the carry, the runner
    is one-shot for the entire integration.
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

    loss_eps = statics.loss_eps
    dt_var_min = statics.dt_var_min
    dt_var_max = statics.dt_var_max
    dt_min = statics.dt_min
    dt_max = statics.dt_max
    batch_max_retries = statics.batch_max_retries
    compo_arr = statics.compo_arr
    atom_ini_arr = statics.atom_ini_arr
    conv_step = statics.conv_step
    count_min = statics.count_min
    count_max = statics.count_max
    runtime = statics.runtime
    trun_min = statics.trun_min
    st_factor = statics.st_factor
    yconv_cri = statics.yconv_cri
    yconv_min = statics.yconv_min
    slope_cri = statics.slope_cri
    flux_cri = statics.flux_cri
    mtol_conv = statics.mtol_conv
    conver_ignore_mask = statics.conver_ignore_mask
    condense_zero_conv_mask = statics.condense_zero_conv_mask
    n_0_static = statics.n_0
    Kzz_static = statics.Kzz
    use_photo_static = statics.use_photo
    use_atm_refresh_static = statics.use_atm_refresh
    use_conden_static = statics.use_conden
    final_update_photo_frq = statics.final_update_photo_frq
    update_frq = statics.update_frq
    use_adapt_rtol = statics.use_adapt_rtol
    rtol_min = statics.rtol_min
    rtol_max = statics.rtol_max
    use_ion_static = statics.use_ion
    e_idx_static = statics.e_idx
    charge_arr_static = statics.charge_arr
    use_fix_all_bot_static = statics.use_fix_all_bot
    bottom_n_static = statics.bottom_n
    use_fix_H2He_static = statics.use_fix_H2He
    h2_idx_static = int(statics.h2_idx)
    he_idx_static = int(statics.he_idx)
    save_evolution_static = statics.save_evolution
    save_evo_frq_static = int(statics.save_evo_frq)
    save_evo_n_max_static = int(statics.save_evo_n_max)
    use_fix_sp_bot_static = statics.use_fix_sp_bot
    fix_sp_bot_idx_static = statics.fix_sp_bot_idx
    fix_sp_bot_mix_static = statics.fix_sp_bot_mix
    use_fix_species_static = statics.use_fix_species
    post_conden_rtol = statics.post_conden_rtol
    fix_species_from_coldtrap_lev = statics.fix_species_from_coldtrap_lev
    fix_species_idx = statics.fix_species_idx
    fix_species_sat_mix = statics.fix_species_sat_mix
    fix_species_wholecol = statics.fix_species_wholecol

    if non_gas_present:
        def _mix_from_y(y_in):
            ysum = jnp.sum(jnp.where(gas_indx_mask[None, :], y_in, 0.0), axis=1, keepdims=True)
            return y_in / ysum
    else:
        def _mix_from_y(y_in):
            ysum = jnp.sum(y_in, axis=1, keepdims=True)
            return y_in / ysum

    def _activate_fix_species(s_in: JaxIntegState) -> JaxIntegState:
        nz_fix = n_0_static.shape[0]
        n_fix = fix_species_idx.shape[0]
        fix_y_new = s_in.fix_y.at[:, fix_species_idx].set(s_in.y[:, fix_species_idx])

        if fix_species_from_coldtrap_lev:
            sat_rho = n_0_static[None, :] * fix_species_sat_mix
            cond_status = s_in.y[:, fix_species_idx].T >= sat_rho
            masked_sat = jnp.where(cond_status, fix_species_sat_mix, jnp.inf)
            coldtrap_idx = jnp.argmin(masked_sat, axis=1).astype(jnp.int32)
            has_cond = jnp.any(cond_status, axis=1)
            coldtrap_idx = jnp.where(
                fix_species_wholecol,
                jnp.int32(nz_fix - 1),
                jnp.where(has_cond, coldtrap_idx, jnp.int32(0)),
            )
            layer_idx = jnp.arange(nz_fix, dtype=jnp.int32)[:, None]
            fix_cols_mask = layer_idx < coldtrap_idx[None, :]
            fix_pfix_idx = coldtrap_idx
        else:
            fix_cols_mask = jnp.ones((nz_fix, n_fix), dtype=jnp.bool_)
            fix_pfix_idx = jnp.full((n_fix,), jnp.int32(nz_fix))

        fix_mask_new = jnp.zeros_like(s_in.fix_mask)
        fix_mask_new = fix_mask_new.at[:, fix_species_idx].set(fix_cols_mask)
        return s_in._replace(
            fix_species_started=jnp.bool_(True),
            fix_y=fix_y_new,
            fix_mask=fix_mask_new,
            fix_pfix_idx=fix_pfix_idx,
            rtol=jnp.float64(post_conden_rtol),
            vs=jnp.zeros_like(s_in.vs),
        )

    def _conv_jax(s: JaxIntegState, accept_count_after: jnp.ndarray
                  ) -> "tuple[jnp.ndarray, jnp.ndarray]":
        """JAX port of `op.conv` (op.py:1018-1066) — computes (longdy, longdydt).

        Looks up the ring entry closest in time to `t * st_factor` (the
        VULCAN steady-state lookback target), with the same edge-case
        handling: don't compare against the most recent entry; clamp to
        `count - conv_step` steps back. The most-recent ring index is
        `(accept_count_after - 1) % conv_step`; on the very first body
        iteration `accept_count_after == 0` and the ring is all-zero, so
        the conv check returns longdy ≈ |y/n_0| ~ O(1) which fails the
        convergence threshold (`yconv_cri = 0.01`) and the `ready` gate
        guards against acting on it anyway.
        """
        target_t = s.t * st_factor
        diffs = jnp.abs(s.t_time_ring - target_t)
        # Tiny additive penalty on the most-recent slot so argmin
        # prefers the next-most-recent when there's a tie (matches the
        # `if indx == count-1: indx-=1` guard at op.py:1035).
        last_idx = jnp.mod(jnp.maximum(accept_count_after - jnp.int32(1),
                                       jnp.int32(0)),
                           jnp.int32(conv_step))
        big = jnp.float64(jnp.inf)
        diffs_guarded = diffs.at[last_idx].set(big)
        indx = jnp.argmin(diffs_guarded)

        y_old = s.y_time_ring[indx]
        n_0_col = n_0_static[:, None]
        longdy_arr = jnp.abs((s.y - y_old) / n_0_col)
        longdy_arr = jnp.where(s.ymix < mtol_conv, 0.0, longdy_arr)
        longdy_arr = jnp.where(s.y < statics.atol, 0.0, longdy_arr)
        longdy_arr = jnp.where(conver_ignore_mask[None, :], 0.0, longdy_arr)
        longdy_arr = jnp.where(condense_zero_conv_mask, 0.0, longdy_arr)

        ratio = jnp.where(s.ymix > 0,
                          longdy_arr / jnp.maximum(s.ymix, 1e-300),
                          0.0)
        longdy_new = jnp.max(ratio)
        dt_lookback = jnp.maximum(s.t - s.t_time_ring[indx], 1e-300)
        longdydt_new = longdy_new / dt_lookback
        return longdy_new, longdydt_new

    def cond_fn(s: JaxIntegState):
        too_long = s.t > jnp.float64(runtime)
        too_many = s.accept_count > jnp.int32(count_max)

        # Convergence: matches op.conv's threshold combination + flux gate.
        # `slope_min` recomputed from the carried Hp (op.py:1031) since
        # atm refresh can change Hp mid-run.
        slope_min = jnp.minimum(
            jnp.min(Kzz_static / (0.1 * s.Hp[:-1]) ** 2),
            jnp.float64(1e-8),
        )
        slope_min = jnp.maximum(slope_min, jnp.float64(1e-10))

        is_converged = (
            ((s.longdy < jnp.float64(yconv_cri))
             & (s.longdydt < jnp.float64(slope_cri)))
            | ((s.longdy < jnp.float64(yconv_min))
               & (s.longdydt < slope_min))
        )
        is_converged = is_converged & (s.aflux_change < jnp.float64(flux_cri))

        ready = (s.t > jnp.float64(trun_min)) & (s.accept_count > jnp.int32(count_min))
        # Phase 16 chunk cap: if the chunked driver set `s.chunk_target` to
        # `(start_count + chunk_size)`, terminate when we've accumulated
        # that many accepted steps. For single-shot runs the seed value
        # `count_max + 1` makes this branch dormant.
        chunk_done = s.accept_count >= s.chunk_target
        terminate = too_long | too_many | chunk_done | (ready & is_converged)
        return jnp.logical_not(terminate)

    def body_fn(s: JaxIntegState, atm_static_):
        # Photo (Phase 10.2): gated `retry_count==0` so reject loops don't
        # re-fire it, and `accept_count % update_photo_frq == 0` matches
        # op.py:825's cadence. The carry holds the dynamic update_photo_frq
        # so the ini→final switch (op.py:819-823) takes effect mid-run.
        if photo_branch is not None:
            photo_due = ((s.retry_count == jnp.int32(0))
                         & (jnp.mod(s.accept_count, s.update_photo_frq)
                            == jnp.int32(0))
                         & jnp.bool_(use_photo_static))
            s = jax.lax.cond(photo_due, photo_branch, lambda ss: ss, s)

        # Atm refresh (Phase 10.3): same retry_count==0 gate; cadence
        # `update_frq` is static (op.py:904).
        if atm_refresh_branch is not None:
            refresh_due = ((s.retry_count == jnp.int32(0))
                           & (jnp.mod(s.accept_count, jnp.int32(update_frq))
                              == jnp.int32(0))
                           & jnp.bool_(use_atm_refresh_static))
            s = jax.lax.cond(refresh_due, atm_refresh_branch, lambda ss: ss, s)

        # Splice the (possibly refreshed) geometry into the static AtmStatic
        # so jax_ros2_step sees the freshest diffusion coefficients.
        atm_step = atm_static_._replace(g=s.g, dzi=s.dzi, Hpi=s.Hpi,
                                        top_flux=s.top_flux, vs=s.vs)

        # Attempt one Ros2 step at the current state.
        if use_fix_species_static:
            sol, delta_arr = jax_ros2_step(
                s.y, s.k_arr, s.dt, atm_step, net, fix_mask=s.fix_mask
            )
            sol = jnp.where(s.fix_mask, s.fix_y, sol)
            delta_arr = jnp.where(s.fix_mask, 0.0, delta_arr)
        else:
            sol, delta_arr = jax_ros2_step(s.y, s.k_arr, s.dt, atm_step, net)

        # Clip (op.py:2450-2473) + recompute ymix + atom_loss.
        sol_clip, ymix_new, small_y_inc, nega_y_inc = clip_fn(sol, s.ymix)
        atom_loss_new = _compute_atom_loss(sol_clip, compo_arr, atom_ini_arr)
        delta = agg_delta_fn(sol_clip, delta_arr, ymix_new)

        # step_ok criteria (op.py:2492-2496) — uses dynamic s.rtol so
        # adaptive-rtol updates take effect on the very next step.
        all_nonneg = jnp.all(sol_clip >= 0)
        loss_diff = jnp.max(jnp.abs(atom_loss_new - s.atom_loss_prev))
        accept = all_nonneg & (loss_diff < loss_eps) & (delta <= s.rtol)

        # Force-accept conditions (op.py:2518-2523 + retry budget guard).
        next_dt_if_reject = s.dt * dt_var_min
        dt_underflow = next_dt_if_reject < dt_min
        retry_exhausted = s.retry_count >= jnp.int32(batch_max_retries)
        force_accept = (dt_underflow | retry_exhausted) & ~accept
        do_accept = accept | force_accept

        # Reject counter dispatch (op.py:2500-2513): exactly one increments.
        delta_too_big = delta > s.rtol
        any_neg = jnp.any(sol_clip < 0)
        is_reject = ~do_accept
        delta_count_inc = (is_reject & delta_too_big).astype(jnp.int32)
        nega_count_inc = (is_reject & ~delta_too_big & any_neg).astype(jnp.int32)
        loss_count_inc = (is_reject & ~delta_too_big & ~any_neg).astype(jnp.int32)

        # Time advance: state.dt for normal accept; dt_min for force_accept; 0 for reject.
        dt_used_for_t = jnp.where(force_accept, jnp.float64(dt_min), s.dt)
        t_next = jnp.where(do_accept, s.t + dt_used_for_t, s.t)

        # dt for the next attempt — uses dynamic s.rtol (Phase 10.5).
        dt_after_normal = _step_size(s.dt, delta, s.rtol,
                                     dt_var_min, dt_var_max, dt_min, dt_max)
        dt_after_force = _step_size(jnp.float64(dt_min), delta, s.rtol,
                                    dt_var_min, dt_var_max, dt_min, dt_max)
        dt_next = jnp.where(force_accept, dt_after_force,
                            jnp.where(accept, dt_after_normal, next_dt_if_reject))

        # Conden branch (Phase 10.4 — op.py:856-902).
        if conden_branch is not None:
            s_post = s._replace(y=sol_clip, ymix=ymix_new)
            in_conden_window = t_next >= jnp.float64(start_conden_time)
            fire_conden = (do_accept
                           & in_conden_window
                           & ~s.fix_species_started
                           & jnp.bool_(use_conden_static))
            s_post = jax.lax.cond(
                fire_conden, conden_branch, lambda ss: ss, s_post,
            )
            if use_fix_species_static:
                trigger_fix = (do_accept
                               & ~s.fix_species_started
                               & in_conden_window
                               & (t_next > jnp.float64(stop_conden_time))
                               & jnp.bool_(use_conden_static))
                s_post = jax.lax.cond(
                    trigger_fix, _activate_fix_species, lambda ss: ss, s_post,
                )
            else:
                trigger_fix = jnp.bool_(False)
            sol_clip = s_post.y
            ymix_new = s_post.ymix
            k_arr_next = s_post.k_arr
            fix_species_started_next = s_post.fix_species_started
            fix_y_next = s_post.fix_y
            fix_mask_next = s_post.fix_mask
            fix_pfix_idx_next = s_post.fix_pfix_idx
            vs_next = s_post.vs
        else:
            k_arr_next = s.k_arr
            fix_species_started_next = s.fix_species_started
            fix_y_next = s.fix_y
            fix_mask_next = s.fix_mask
            fix_pfix_idx_next = s.fix_pfix_idx
            vs_next = s.vs
            trigger_fix = jnp.bool_(False)

        # Hydrostatic balance (op.py:908-914).
        n_0 = atm_step.M[:, None]
        sol_balanced_full = n_0 * ymix_new
        if hydro_partial:
            sol_balanced = jnp.where(gas_indx_mask[None, :],
                                     sol_balanced_full, sol_clip)
        else:
            sol_balanced = sol_balanced_full

        # Ion charge balance (Phase 10.6 — op.py:3001-3007). With charge_arr[i]
        # = compo[i]['e'] for i in charge_list (and 0 for the 'e' column
        # itself), `e[:] = -dot(y, charge_arr)` enforces zero net charge per
        # layer. Static branch: skipped at trace time when use_ion=False so
        # the placeholder zeros never run.
        if use_ion_static:
            e_density = -jnp.einsum("zi,i->z", sol_balanced, charge_arr_static)
            sol_balanced = sol_balanced.at[:, e_idx_static].set(e_density)

        # fix-all-bot post-step clamp (Phase 10.6 — op.py:3050-3051). Bottom
        # layer pinned to chemical-EQ mixing ratios captured at OuterLoop
        # init, multiplied by the static n_0[0]. Static branch.
        if use_fix_all_bot_static:
            sol_balanced = sol_balanced.at[0].set(bottom_n_static)
        if use_fix_sp_bot_static:
            sol_balanced = sol_balanced.at[0, fix_sp_bot_idx_static].set(
                fix_sp_bot_mix_static * n_0_static[0]
            )

        # Hycean H2/He bottom-pin (op.py:2938-2944).
        # On the first accepted step with t > 1e6, snapshot the current
        # bottom-layer ymix for H2 and He; pin them at that value forever
        # after. The trip uses s.ymix (the pre-step mixing ratio) to match
        # master's read of var.ymix at op.py:2939, before the per-step
        # kernel has rewritten the bottom layer. The trip is gated on
        # `accept` so rejected attempts don't spuriously fire the
        # snapshot, and on `~h2he_pinned` so it only fires once.
        if use_fix_H2He_static:
            trip = accept & (~s.h2he_pinned) & (s.t > jnp.float64(1e6))
            h2_mix_snap = jnp.where(trip, s.ymix[0, h2_idx_static], s.h2he_mix[0])
            he_mix_snap = jnp.where(trip, s.ymix[0, he_idx_static], s.h2he_mix[1])
            h2he_mix_next = jnp.stack([h2_mix_snap, he_mix_snap])
            h2he_pinned_next = s.h2he_pinned | trip
            apply_pin = h2he_pinned_next
            new_h2_val = jnp.where(apply_pin,
                                    h2_mix_snap * n_0_static[0],
                                    sol_balanced[0, h2_idx_static])
            new_he_val = jnp.where(apply_pin,
                                    he_mix_snap * n_0_static[0],
                                    sol_balanced[0, he_idx_static])
            sol_balanced = sol_balanced.at[0, h2_idx_static].set(new_h2_val)
            sol_balanced = sol_balanced.at[0, he_idx_static].set(new_he_val)
        else:
            h2he_pinned_next = s.h2he_pinned
            h2he_mix_next = s.h2he_mix

        ymix_balanced = _mix_from_y(sol_balanced)
        atom_loss_balanced = _compute_atom_loss(sol_balanced, compo_arr, atom_ini_arr)

        # y / ymix / atom_loss for the next iteration.
        y_prev_clipped = jnp.where(s.y_prev < 0, 0.0, s.y_prev)
        y_next = jnp.where(force_accept, y_prev_clipped,
                           jnp.where(accept, sol_balanced, s.y_prev))
        ymix_next = jnp.where(force_accept, s.ymix,
                              jnp.where(accept, ymix_balanced, s.ymix))
        atom_loss_next = jnp.where(force_accept, s.atom_loss_prev,
                                   jnp.where(accept, atom_loss_balanced, s.atom_loss))

        # On accept, the new "y_prev" — the revert target for the next
        # in-flight retry sequence — is the fresh accepted state. On
        # reject we keep the carry's y_prev (revert target for the
        # ongoing retry of THIS step).
        y_prev_next = jnp.where(do_accept, y_next, s.y_prev)
        atom_loss_prev_next = jnp.where(do_accept, atom_loss_next, s.atom_loss_prev)

        accept_count_next = s.accept_count + jnp.where(do_accept, jnp.int32(1), jnp.int32(0))
        retry_count_next = jnp.where(do_accept, jnp.int32(0), s.retry_count + jnp.int32(1))

        # Ring-buffer append (op.save_step, op.py:1090-1106): write the
        # post-accept (y, t) into ring slot `(accept_count_next - 1) %
        # conv_step`. On reject this rewrites the same slot with the
        # unchanged y_prev / s.t — idempotent.
        ring_idx = jnp.mod(jnp.maximum(accept_count_next - jnp.int32(1),
                                       jnp.int32(0)),
                           jnp.int32(conv_step))
        y_time_ring_new = s.y_time_ring.at[ring_idx].set(y_next)
        t_time_ring_new = s.t_time_ring.at[ring_idx].set(t_next)

        # Phase 15: save_evolution per-step capture. Append (y_next, t_next)
        # to (y_evo, t_evo) every save_evo_frq accepted steps until the
        # buffer is full. When save_evolution=False the closure-time gate
        # is False, so the body branch compiles out entirely.
        if save_evolution_static:
            do_evo_append = (do_accept
                             & (jnp.mod(accept_count_next,
                                        jnp.int32(save_evo_frq_static))
                                == jnp.int32(0))
                             & (s.evo_idx
                                < jnp.int32(save_evo_n_max_static)))
            evo_slot = s.evo_idx
            zero_i32 = jnp.int32(0)
            y_evo_new = jax.lax.cond(
                do_evo_append,
                lambda yev: jax.lax.dynamic_update_slice(
                    yev, y_next[None, :, :], (evo_slot, zero_i32, zero_i32)),
                lambda yev: yev,
                s.y_evo,
            )
            t_evo_new = jax.lax.cond(
                do_evo_append,
                lambda tev: tev.at[evo_slot].set(t_next),
                lambda tev: tev,
                s.t_evo,
            )
            evo_idx_new = s.evo_idx + jnp.where(do_evo_append,
                                                jnp.int32(1),
                                                jnp.int32(0))
        else:
            y_evo_new = s.y_evo
            t_evo_new = s.t_evo
            evo_idx_new = s.evo_idx

        # We need a snapshot state with (y, t, ring) updated to feed _conv_jax.
        s_for_conv = s._replace(
            y=y_next, ymix=ymix_next, t=t_next,
            y_time_ring=y_time_ring_new, t_time_ring=t_time_ring_new,
        )
        longdy_new_val, longdydt_new_val = _conv_jax(s_for_conv, accept_count_next)
        # Only refresh longdy/longdydt on accepted steps — rejected steps
        # don't change the ring contents in any meaningful way.
        longdy_next = jnp.where(do_accept, longdy_new_val, s.longdy)
        longdydt_next = jnp.where(do_accept, longdydt_new_val, s.longdydt)

        # Adaptive rtol (op.py:836-852). Only fires on accepted steps; the
        # cadence gates use the post-accept count.
        max_atom_loss = jnp.max(jnp.abs(atom_loss_next))
        rtol_base = jnp.where(trigger_fix, jnp.float64(post_conden_rtol), s.rtol)
        do_dec = (jnp.bool_(use_adapt_rtol)
                  & do_accept
                  & ~trigger_fix
                  & (jnp.mod(accept_count_next, jnp.int32(10)) == jnp.int32(0))
                  & (max_atom_loss >= s.loss_criteria))
        rtol_dec = jnp.maximum(rtol_base * jnp.float64(0.75), jnp.float64(rtol_min))
        loss_crit_dec = s.loss_criteria * jnp.float64(2.0)
        rtol_after_dec = jnp.where(do_dec, rtol_dec, rtol_base)
        loss_criteria_after_dec = jnp.where(do_dec, loss_crit_dec, s.loss_criteria)

        do_inc = (jnp.bool_(use_adapt_rtol)
                  & do_accept
                  & ~trigger_fix
                  & (jnp.mod(accept_count_next, jnp.int32(1000)) == jnp.int32(0))
                  & (accept_count_next > jnp.int32(0))
                  & (max_atom_loss < jnp.float64(2e-4)))
        rtol_inc = jnp.minimum(rtol_after_dec * jnp.float64(1.25),
                               jnp.float64(rtol_max))
        rtol_next = jnp.where(do_inc, rtol_inc, rtol_after_dec)

        # Photo-frequency ini→final switch (op.py:819-823).
        switch_to_final = (jnp.bool_(use_photo_static)
                           & ~s.is_final_photo_frq
                           & (longdy_next < jnp.float64(yconv_min) * 10.0)
                           & (longdydt_next < jnp.float64(1e-6)))
        update_photo_frq_next = jnp.where(switch_to_final,
                                          jnp.int32(final_update_photo_frq),
                                          s.update_photo_frq)
        is_final_next = s.is_final_photo_frq | switch_to_final

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
            y_time_ring=y_time_ring_new,
            t_time_ring=t_time_ring_new,
            longdy=longdy_next,
            longdydt=longdydt_next,
            rtol=rtol_next,
            loss_criteria=loss_criteria_after_dec,
            update_photo_frq=update_photo_frq_next,
            is_final_photo_frq=is_final_next,
            vs=vs_next,
            fix_species_started=fix_species_started_next,
            fix_y=fix_y_next,
            fix_mask=fix_mask_next,
            fix_pfix_idx=fix_pfix_idx_next,
            h2he_pinned=h2he_pinned_next,
            h2he_mix=h2he_mix_next,
            y_evo=y_evo_new,
            t_evo=t_evo_new,
            evo_idx=evo_idx_new,
            # chunk_target is set by the driver before each chunk and
            # never mutated inside the body — pass through unchanged.
            chunk_target=s.chunk_target,
        )

    @jax.jit
    def runner(state: JaxIntegState, atm_static: AtmStatic):
        # Phase 10.5: the runner is one-shot. cond_fn checks the full
        # convergence + count_max + runtime termination criterion; body_fn
        # gates photo / atm_refresh / conden / adaptive rtol / photo-freq
        # switch internally. No per-batch counter reset needed — the
        # carry's accept_count / nega_count / etc. accumulate across the
        # entire integration.
        return jax.lax.while_loop(
            cond_fn,
            lambda s: body_fn(s, atm_static),
            state,
        )

    return runner


# ---------------------------------------------------------------------------
# OuterLoop class — standalone replacement for op.Integration (Phase 10.6)
# ---------------------------------------------------------------------------
class OuterLoop:
    """Standalone outer-integration driver. The runner is one-shot per
    integration: a single `lax.while_loop` call performs every accepted
    step (with internal retries), photo / atm-refresh / conden updates,
    ring-buffered convergence, and adaptive rtol.

    Phase 10.6 dropped the `op.Integration` parent class; the small bits of
    parent-class behaviour we still need — atol/mtol pulled from cfg, the
    `f_dy` post-run diagnostic — are inlined below. There is no NumPy hot
    path anywhere in the per-step code.
    """

    def __init__(self, odesolver, output):
        # Inlined from op.Integration.__init__ (op.py:790-806). The only
        # parent attributes the rest of the class touches are mtol / atol /
        # output / odesolver — everything else (non_gas_sp_index, etc.)
        # lives in `_Statics` / `_PhotoStatic` / `_CondenStatic` instead.
        self.mtol = float(vulcan_cfg.mtol)
        self.atol = float(vulcan_cfg.atol)
        self.output = output
        self.odesolver = odesolver
        self.loss_criteria = 0.0005

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

        # Charge column from compo (used only when use_ion=True). Kept as a
        # 1-D NumPy array; the per-run charge_arr (zeroed outside
        # var.charge_list) is built in `_build_statics` once var is known.
        self._compo_charge = np.array(
            [float(compo[compo_row.index(sp)]['e']) for sp in self._species],
            dtype=np.float64,
        )

        # Static masks. HD189 default has no condensation, no botflux, no
        # fix_sp_bot — these stay degenerate but are kept as proper arrays
        # so other configs work without code changes.
        self._non_gas_present = bool(vulcan_cfg.non_gas_sp)
        self._zero_bot_row = bool(vulcan_cfg.use_botflux or vulcan_cfg.use_fix_sp_bot)

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

    def reset(self) -> None:
        """Drop the cached JIT'd runner and all closed-over statics.

        Call this when `vulcan_cfg` is mutated mid-session (notebook tuning,
        parameter sweeps, etc.) so the next `__call__` rebuilds the runner
        and re-traces against the fresh config. Without this hook, mutated
        config values are silently ignored — the runner closure was built
        against the original config at first call.
        """
        self._runner = None
        self._statics = None
        self._photo_static = None
        self._refresh_static = None
        self._conden_static = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _build_statics(self, var, atm) -> _Statics:
        """Pack scalar configs and per-run arrays into the closure inputs.

        Phase-10.5 statics include the convergence + termination tunables
        (`conv_step`, `count_min`, `count_max`, `runtime`, `trun_min`,
        `st_factor`, `yconv_cri` / `yconv_min`, `slope_cri`, `flux_cri`,
        `mtol_conv`), the conv-side masks (`conver_ignore_mask`,
        `condense_zero_conv_mask`), the static atmosphere arrays needed by
        the in-runner conv (`n_0`, `Kzz` for slope_min recomputation), the
        photo-frequency cadence tunables, and the rtol bounds.
        """
        ni = _NETWORK.ni
        atom_ini_arr = np.asarray(
            [float(var.atom_ini[a]) for a in self._atom_order],
            dtype=np.float64,
        )

        # conver_ignore: species names listed in vulcan_cfg.conver_ignore
        # that the longdy reduction at op.py:1049 zeroes out.
        conver_ignore_np = np.zeros(ni, dtype=bool)
        for sp in getattr(vulcan_cfg, "conver_ignore", []):
            if sp in _NETWORK.species_idx:
                conver_ignore_np[_NETWORK.species_idx[sp]] = True

        # condense_zero_conv (op.py:1051-1052): when use_condense, the
        # non_gas_sp columns are zeroed in longdy. HD189 has non_gas_sp=[].
        nz = atm.Tco.shape[0]
        cond_zero_conv_np = np.zeros((nz, ni), dtype=bool)
        if vulcan_cfg.use_condense:
            for sp in vulcan_cfg.non_gas_sp:
                if sp in _NETWORK.species_idx:
                    cond_zero_conv_np[:, _NETWORK.species_idx[sp]] = True

        # Phase 10.6 — ion charge balance + fix-all-bot. Both static masks
        # are degenerate (zeros) when their cfg flag is off; the body's
        # Python `if use_*_static:` skips them at trace time.
        use_ion = bool(vulcan_cfg.use_ion)
        if use_ion:
            charge_list = list(getattr(var, "charge_list", []))
            charge_np = np.zeros(ni, dtype=np.float64)
            for sp in charge_list:
                if sp in _NETWORK.species_idx:
                    charge_np[_NETWORK.species_idx[sp]] = self._compo_charge[
                        _NETWORK.species_idx[sp]
                    ]
            e_idx = (_NETWORK.species_idx["e"]
                     if "e" in _NETWORK.species_idx else 0)
            # `e` itself must contribute 0 to the dot so the formula
            # `e[:] = -dot(y, charge_arr)` is consistent (op.py:3004 zeros
            # `e` first; we get the same effect by excluding e from the
            # charge column).
            charge_np[e_idx] = 0.0
        else:
            charge_np = np.zeros(ni, dtype=np.float64)
            e_idx = 0

        use_fix_all_bot = bool(getattr(vulcan_cfg, "use_fix_all_bot", False))
        if use_fix_all_bot:
            # Pin bottom layer to chemical-EQ mixing ratios captured at
            # init time, scaled by static n_0[0] (op.py:3022, 3050-3051).
            bottom_n_np = (np.asarray(var.ymix[0], dtype=np.float64)
                           * float(atm.n_0[0]))
        else:
            bottom_n_np = np.zeros(ni, dtype=np.float64)

        fix_sp_bot_cfg = getattr(vulcan_cfg, "use_fix_sp_bot", {}) or {}
        use_fix_sp_bot = bool(fix_sp_bot_cfg)
        if use_fix_sp_bot:
            fix_sp_bot_idx = np.asarray(
                [_NETWORK.species_idx[sp] for sp in fix_sp_bot_cfg.keys()],
                dtype=np.int32,
            )
            fix_sp_bot_mix = np.asarray(
                [float(fix_sp_bot_cfg[sp]) for sp in fix_sp_bot_cfg.keys()],
                dtype=np.float64,
            )
        else:
            fix_sp_bot_idx = np.zeros((0,), dtype=np.int32)
            fix_sp_bot_mix = np.zeros((0,), dtype=np.float64)

        use_fix_H2He = bool(getattr(vulcan_cfg, "use_fix_H2He", False))
        if use_fix_H2He:
            h2_idx = int(_NETWORK.species_idx["H2"])
            he_idx = int(_NETWORK.species_idx["He"])
        else:
            h2_idx = -1
            he_idx = -1

        fix_species_cfg = list(getattr(vulcan_cfg, "fix_species", []) or [])
        use_fix_species = bool(vulcan_cfg.use_condense and fix_species_cfg)
        if use_fix_species:
            wholecol_species = {"H2O_l_s", "H2SO4_l", "NH3_l_s", "S8_l_s"}
            fix_species_idx = np.asarray(
                [_NETWORK.species_idx[sp] for sp in fix_species_cfg],
                dtype=np.int32,
            )
            fix_species_wholecol = np.asarray(
                [sp in wholecol_species for sp in fix_species_cfg],
                dtype=bool,
            )
            fix_species_sat_mix = np.zeros((len(fix_species_cfg), nz), dtype=np.float64)
            for i, sp in enumerate(fix_species_cfg):
                if sp in atm.sat_mix:
                    fix_species_sat_mix[i] = np.asarray(atm.sat_mix[sp], dtype=np.float64)
        else:
            fix_species_idx = np.zeros((0,), dtype=np.int32)
            fix_species_sat_mix = np.zeros((0, nz), dtype=np.float64)
            fix_species_wholecol = np.zeros((0,), dtype=bool)

        return _Statics(
            compo_arr=jnp.asarray(self._compo_arr),
            atom_ini_arr=jnp.asarray(atom_ini_arr),
            initial_rtol=float(vulcan_cfg.rtol),
            loss_eps=float(vulcan_cfg.loss_eps),
            pos_cut=float(vulcan_cfg.pos_cut),
            nega_cut=float(vulcan_cfg.nega_cut),
            mtol=float(self.mtol),
            atol=float(self.atol),
            dt_var_min=float(vulcan_cfg.dt_var_min),
            dt_var_max=float(vulcan_cfg.dt_var_max),
            dt_min=float(vulcan_cfg.dt_min),
            dt_max=float(vulcan_cfg.dt_max),
            batch_max_retries=int(getattr(vulcan_cfg, "batch_max_retries", 8)),
            conv_step=int(vulcan_cfg.conv_step),
            count_min=int(vulcan_cfg.count_min),
            count_max=int(vulcan_cfg.count_max),
            runtime=float(vulcan_cfg.runtime),
            trun_min=float(vulcan_cfg.trun_min),
            st_factor=float(vulcan_cfg.st_factor),
            yconv_cri=float(vulcan_cfg.yconv_cri),
            yconv_min=float(vulcan_cfg.yconv_min),
            slope_cri=float(vulcan_cfg.slope_cri),
            flux_cri=float(vulcan_cfg.flux_cri),
            mtol_conv=float(vulcan_cfg.mtol_conv),
            conver_ignore_mask=jnp.asarray(conver_ignore_np),
            condense_zero_conv_mask=jnp.asarray(cond_zero_conv_np),
            n_0=jnp.asarray(atm.n_0, dtype=jnp.float64),
            Kzz=jnp.asarray(atm.Kzz, dtype=jnp.float64),
            use_photo=bool(vulcan_cfg.use_photo),
            use_atm_refresh=True,
            use_conden=bool(vulcan_cfg.use_condense),
            ini_update_photo_frq=int(getattr(vulcan_cfg, "ini_update_photo_frq", 1)),
            final_update_photo_frq=int(getattr(vulcan_cfg, "final_update_photo_frq", 1)),
            update_frq=int(vulcan_cfg.update_frq),
            use_adapt_rtol=bool(getattr(vulcan_cfg, "use_adapt_rtol", False)),
            rtol_min=float(getattr(vulcan_cfg, "rtol_min", 0.0)),
            rtol_max=float(getattr(vulcan_cfg, "rtol_max", 1.0)),
            initial_loss_criteria=float(getattr(self, "loss_criteria", 0.0005)),
            use_ion=use_ion,
            e_idx=int(e_idx),
            charge_arr=jnp.asarray(charge_np),
            use_fix_all_bot=use_fix_all_bot,
            bottom_n=jnp.asarray(bottom_n_np),
            use_fix_sp_bot=use_fix_sp_bot,
            fix_sp_bot_idx=jnp.asarray(fix_sp_bot_idx),
            fix_sp_bot_mix=jnp.asarray(fix_sp_bot_mix),
            use_fix_H2He=use_fix_H2He,
            h2_idx=h2_idx,
            he_idx=he_idx,
            use_fix_species=use_fix_species,
            post_conden_rtol=float(getattr(vulcan_cfg, "post_conden_rtol", vulcan_cfg.rtol)),
            fix_species_from_coldtrap_lev=bool(
                getattr(vulcan_cfg, "fix_species_from_coldtrap_lev", False)
            ),
            fix_species_idx=jnp.asarray(fix_species_idx),
            fix_species_sat_mix=jnp.asarray(fix_species_sat_mix),
            fix_species_wholecol=jnp.asarray(fix_species_wholecol),
            save_evolution=bool(getattr(vulcan_cfg, "save_evolution", False)),
            save_evo_frq=int(getattr(vulcan_cfg, "save_evo_frq", 1)),
            save_evo_n_max=(
                int(np.ceil(int(vulcan_cfg.count_max)
                            / max(int(getattr(vulcan_cfg, "save_evo_frq", 1)), 1))) + 1
                if bool(getattr(vulcan_cfg, "save_evolution", False))
                else 1
            ),
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
            float(getattr(vulcan_cfg, "start_conden_time", 0.0)),
            float(getattr(vulcan_cfg, "stop_conden_time", 0.0)),
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
        photo_ion_data = getattr(odesolver, "_photo_ion_data", None)
        if vulcan_cfg.use_ion:
            if photo_ion_data is None:
                photo_ion_data = _photo_mod.pack_photo_ion_data(var, vulcan_cfg)
                odesolver._photo_ion_data = photo_ion_data
        else:
            photo_ion_data = None

        (branch_re_idx, branch_active,
         branch_T_re_idx, branch_T_active) = _photo_mod.pack_J_to_k_index_map(
            photo_J_data, var, vulcan_cfg
        )
        if photo_ion_data is not None:
            ion_branch_re_idx, ion_branch_active = _photo_mod.pack_Jion_to_k_index_map(
                photo_ion_data, var, vulcan_cfg
            )
            cross_Jion = photo_ion_data.cross_J
        else:
            ion_branch_re_idx = jnp.zeros((0,), dtype=jnp.int64)
            ion_branch_active = jnp.zeros((0,), dtype=jnp.bool_)
            cross_Jion = jnp.zeros((0, int(var.nbin)), dtype=jnp.float64)

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
            photo_ion_data=photo_ion_data,
            cross_Jion=cross_Jion,
            ion_branch_re_idx=ion_branch_re_idx,
            ion_branch_active=ion_branch_active,
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
                Jion_br=jnp.zeros((0, nz), dtype=jnp.float64),
            )
        nbin = int(var.nbin)
        n_br = int(self._photo_static.cross_J.shape[0])
        n_br_T = int(self._photo_static.cross_J_T.shape[0])
        n_ion_br = int(self._photo_static.cross_Jion.shape[0])
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
            Jion_br=jnp.zeros((n_ion_br, nz), dtype=jnp.float64),
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
            vs=jnp.asarray(atm.vs, dtype=jnp.float64),
        )

    def _initial_conv_carry(self, var, atm) -> dict:
        """Build the initial values for the Phase 10.5 conv / history fields.

        - `y_time_ring` / `t_time_ring`: zero-filled (conv_step, nz, ni) /
          (conv_step,) — the body fills them as steps accept. The cond_fn
          waits for `accept_count > count_min` before acting on conv, which
          gives the ring time to populate.
        - `longdy` / `longdydt`: seeded from `var.longdy` / `var.longdydt`
          (typically 1.0 and 1.0 at fresh-start).
        - `rtol`: starts at `vulcan_cfg.rtol`; updated in-body by adaptive
          rtol logic.
        - `loss_criteria`: starts at `self.loss_criteria` (default 0.0005);
          doubles every time adaptive rtol fires the decrease branch.
        - `update_photo_frq`: starts at `ini_update_photo_frq`; flips to
          `final_update_photo_frq` once longdy/longdydt drop below the
          op.py:819-823 thresholds.
        """
        nz = atm.Tco.shape[0]
        ni = _NETWORK.ni
        conv_step = int(vulcan_cfg.conv_step)
        ini_frq = int(getattr(vulcan_cfg, "ini_update_photo_frq", 1))
        return dict(
            y_time_ring=jnp.zeros((conv_step, nz, ni), dtype=jnp.float64),
            t_time_ring=jnp.zeros((conv_step,), dtype=jnp.float64),
            longdy=jnp.float64(float(getattr(var, "longdy", 1.0))),
            longdydt=jnp.float64(float(getattr(var, "longdydt", 1.0))),
            rtol=jnp.float64(float(vulcan_cfg.rtol)),
            loss_criteria=jnp.float64(
                float(getattr(self, "loss_criteria", 0.0005))
            ),
            update_photo_frq=jnp.int32(ini_frq),
            is_final_photo_frq=jnp.bool_(False),
        )

    def _pack_state(self, var, para, atm) -> JaxIntegState:
        """Build the initial JaxIntegState from current var/para/atm."""
        photo_fields = self._initial_photo_carry(var, atm)
        atm_fields = self._initial_atm_carry(atm)
        conv_fields = self._initial_conv_carry(var, atm)
        nz = atm.Tco.shape[0]
        ni = _NETWORK.ni
        n_fix = int(self._statics.fix_species_idx.shape[0])
        fix_started = bool(getattr(para, "fix_species_start", False))
        fix_y_init = np.zeros((nz, ni), dtype=np.float64)
        fix_mask_init = np.zeros((nz, ni), dtype=bool)
        fix_pfix_idx_init = np.zeros((n_fix,), dtype=np.int32)
        if self._statics.use_fix_species and hasattr(var, "fix_y"):
            for i, sp in enumerate(vulcan_cfg.fix_species):
                sp_idx = _NETWORK.species_idx[sp]
                fix_y_init[:, sp_idx] = np.asarray(var.fix_y.get(sp, np.zeros(nz)), dtype=np.float64)
                if fix_started:
                    if self._statics.fix_species_from_coldtrap_lev:
                        pfix = int(getattr(atm, "conden_min_lev", {}).get(sp, 0))
                        fix_mask_init[:pfix, sp_idx] = True
                        fix_pfix_idx_init[i] = pfix
                    else:
                        fix_mask_init[:, sp_idx] = True
                        fix_pfix_idx_init[i] = nz
        return JaxIntegState(
            y=jnp.asarray(var.y, dtype=jnp.float64),
            y_prev=jnp.asarray(var.y_prev if hasattr(var, "y_prev") and var.y_prev is not None else var.y,
                               dtype=jnp.float64),
            ymix=jnp.asarray(var.ymix, dtype=jnp.float64),
            dt=jnp.asarray(float(var.dt), dtype=jnp.float64),
            t=jnp.asarray(float(var.t), dtype=jnp.float64),
            delta=jnp.asarray(float(getattr(para, "delta", 0.0)), dtype=jnp.float64),
            accept_count=jnp.int32(int(para.count)),
            retry_count=jnp.int32(0),
            atom_loss=jnp.asarray(self._atom_dict_to_arr(var.atom_loss),
                                  dtype=jnp.float64),
            atom_loss_prev=jnp.asarray(self._atom_dict_to_arr(var.atom_loss_prev),
                                       dtype=jnp.float64),
            nega_count=jnp.int32(int(getattr(para, "nega_count", 0))),
            loss_count=jnp.int32(int(getattr(para, "loss_count", 0))),
            delta_count=jnp.int32(int(getattr(para, "delta_count", 0))),
            small_y=jnp.float64(float(getattr(para, "small_y", 0.0))),
            nega_y=jnp.float64(float(getattr(para, "nega_y", 0.0))),
            **photo_fields,
            **atm_fields,
            **conv_fields,
            fix_species_started=jnp.bool_(fix_started),
            fix_y=jnp.asarray(fix_y_init, dtype=jnp.float64),
            fix_mask=jnp.asarray(fix_mask_init, dtype=jnp.bool_),
            fix_pfix_idx=jnp.asarray(fix_pfix_idx_init, dtype=jnp.int32),
            # Hycean: seed pinned=False, mix=[0, 0]; the body snapshots the
            # live ymix when (use_fix_H2He=True) & (~pinned) & (t > 1e6).
            h2he_pinned=jnp.bool_(False),
            h2he_mix=jnp.zeros((2,), dtype=jnp.float64),
            # save_evolution buffers (Phase 15). Allocated to the cfg's
            # `save_evo_n_max` when on; length-1 placeholder when off.
            y_evo=jnp.zeros(
                (int(self._statics.save_evo_n_max), nz, ni),
                dtype=jnp.float64,
            ),
            t_evo=jnp.zeros(
                (int(self._statics.save_evo_n_max),),
                dtype=jnp.float64,
            ),
            evo_idx=jnp.int32(0),
            # chunk_target seed = INT32_MAX disables the chunk cap, so
            # single-shot __call__ behaves exactly as pre-Phase-16. The
            # chunked driver mutates this between chunks; the seed is
            # only used when the driver doesn't run.
            chunk_target=jnp.int32(2**30),
        )

    def _unpack_state(self, state: JaxIntegState, var, para, atm) -> None:
        """Write the post-runner JAX state back into the var/para/atm store objects."""
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
        # Counters now accumulate inside the JAX runner across the entire
        # integration (Phase 10.5), so write them through directly. The
        # carry was seeded with the entry-time values in `_pack_state`,
        # which preserves any prior counters from a partial run.
        para.count = int(state.accept_count)
        para.nega_count = int(state.nega_count)
        para.loss_count = int(state.loss_count)
        para.delta_count = int(state.delta_count)
        para.small_y = float(state.small_y)
        para.nega_y = float(state.nega_y)
        para.fix_species_start = bool(state.fix_species_started)

        # Hycean H2/He pin diagnostic (op.py:2938-2944). Mirror master's
        # vulcan_cfg mutation so any downstream tooling that reads
        # vulcan_cfg.use_fix_sp_bot post-run sees the pinned values.
        if self._statics.use_fix_H2He and bool(state.h2he_pinned):
            h2he_mix_arr = np.asarray(state.h2he_mix, dtype=np.float64)
            existing = dict(getattr(vulcan_cfg, "use_fix_sp_bot", {}) or {})
            existing.setdefault("H2", float(h2he_mix_arr[0]))
            existing.setdefault("He", float(h2he_mix_arr[1]))
            vulcan_cfg.use_fix_sp_bot = existing

        # Phase 10.5 conv fields — write back into var so the .vul output
        # and any downstream Python diagnostics see the final values.
        var.longdy = float(state.longdy)
        var.longdydt = float(state.longdydt)
        # rtol may have moved adaptively inside the runner; reflect it in
        # the global cfg for parity with op.Integration.__call__.
        if self._statics.use_adapt_rtol:
            vulcan_cfg.rtol = float(state.rtol)
        # Rebuild var.y_time / var.t_time chronologically from the ring
        # buffer. The ring stores the most recent min(accept_count,
        # conv_step) accepted (y, t) pairs at slots
        # ((accept_count - n) % conv_step) for n = conv_step..1.
        self._unpack_ring(state, var)

        # Phase 15: when save_evolution=True, the runner captured a fresh
        # (y_evo, t_evo) trajectory at save_evo_frq cadence into a fixed-
        # size buffer. Override var.y_time / var.t_time with that prefix
        # so legacy_io.save_out picks them up unmodified (the master-style
        # `[::fq]` re-slice is removed since the runner already pre-sliced).
        if self._statics.save_evolution:
            n_evo = int(state.evo_idx)
            y_evo_arr = np.asarray(state.y_evo, dtype=np.float64)[:n_evo]
            t_evo_arr = np.asarray(state.t_evo, dtype=np.float64)[:n_evo]
            var.y_time = y_evo_arr
            var.t_time = t_evo_arr

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
        atm.vs = np.asarray(state.vs, dtype=np.float64)

        if self._statics.use_fix_species:
            fix_y_np = np.asarray(state.fix_y, dtype=np.float64)
            var.fix_y = {
                sp: fix_y_np[:, _NETWORK.species_idx[sp]].copy()
                for sp in vulcan_cfg.fix_species
            }
            if self._statics.fix_species_from_coldtrap_lev:
                pfix_np = np.asarray(state.fix_pfix_idx, dtype=np.int32)
                for sp, pfix in zip(vulcan_cfg.fix_species, pfix_np.tolist()):
                    atm.conden_min_lev[sp] = int(pfix)

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
        Jion_br_np = np.asarray(state.Jion_br, dtype=np.float64)
        for i, key in enumerate(self._photo_static.photo_J_data.branch_keys):
            sp, _ = key
            var.J_sp[key] = J_br_np[i]
            var.J_sp[(sp, 0)] = var.J_sp[(sp, 0)] + J_br_np[i]
        for i, key in enumerate(self._photo_static.photo_J_data.branch_T_keys):
            sp, _ = key
            var.J_sp[key] = J_br_T_np[i]
            var.J_sp[(sp, 0)] = var.J_sp[(sp, 0)] + J_br_T_np[i]
        if self._photo_static.cross_Jion.shape[0] > 0:
            var.Jion_sp = {
                (sp, bn): np.zeros(nz)
                for sp in var.ion_sp
                for bn in range(var.ion_branch[sp] + 1)
            }
            for i, key in enumerate(self._photo_static.photo_ion_data.branch_keys):
                sp, _ = key
                var.Jion_sp[key] = Jion_br_np[i]
                var.Jion_sp[(sp, 0)] = var.Jion_sp[(sp, 0)] + Jion_br_np[i]

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
        if self._photo_static.cross_Jion.shape[0] > 0:
            for key in self._photo_static.photo_ion_data.branch_keys:
                ridx = var.ion_rate_index.get(key)
                if ridx is not None and ridx not in vulcan_cfg.remove_list:
                    var.k[ridx] = k_arr[ridx]

    def _unpack_ring(self, state: JaxIntegState, var) -> None:
        """Rebuild `var.y_time` / `var.t_time` chronologically from the ring.

        The ring slot for the n-th accepted step (0-indexed) is
        `n % conv_step`. After the runner returns, the chronological
        ordering of the most recent `min(accept_count, conv_step)` entries
        is `slots[(accept_count - L + i) % conv_step for i in 0..L-1]`,
        where L = min(accept_count, conv_step).

        Phase 10.5 trade-off: var.y_time only contains the LAST conv_step
        entries instead of the full per-step trajectory. This is the
        single-shot-runner cost — keeping the full history would require
        an io_callback per step, which defeats the purpose. Users who need
        full trajectory output can downsample at the analysis stage or
        increase conv_step.
        """
        accept_count = int(state.accept_count)
        conv_step = int(self._statics.conv_step)
        L = min(accept_count, conv_step)
        if L <= 0:
            var.y_time = []
            var.t_time = []
            var.atom_loss_time = []
            return

        ring_y = np.asarray(state.y_time_ring, dtype=np.float64)
        ring_t = np.asarray(state.t_time_ring, dtype=np.float64)
        # Most recent slot is (accept_count - 1) % conv_step; oldest in
        # the kept window is (accept_count - L) % conv_step.
        start = (accept_count - L) % conv_step
        order = [(start + i) % conv_step for i in range(L)]
        var.y_time = [ring_y[i] for i in order]
        var.t_time = [ring_t[i] for i in order]
        # atom_loss_time is per-step diagnostic; we only have the FINAL
        # atom_loss in the carry (Phase 10.5 doesn't ring it). Use the
        # final value padded over L entries so plot scripts that index
        # into the list don't index-error.
        final_atom_loss = list(np.asarray(state.atom_loss).tolist())
        var.atom_loss_time = [final_atom_loss for _ in range(L)]

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
    # f_dy — inlined from op.Integration.f_dy (op.py:1003-1015)
    # -----------------------------------------------------------------
    @staticmethod
    def _f_dy(var, para):
        """Compute dy / dydt diagnostics from var.y vs var.y_prev.

        Inlined from `op.Integration.f_dy` — no JAX, just NumPy reductions
        used once per integration to populate `var.dy` / `var.dydt` for the
        final-state print.
        """
        if para.count == 0:
            var.dy, var.dydt = 1.0, 1.0
            return var
        y, ymix, y_prev = var.y, var.ymix, var.y_prev
        dy = np.abs(y - y_prev)
        dy[ymix < vulcan_cfg.mtol] = 0
        dy[y < vulcan_cfg.atol] = 0
        pos = y > 0
        if np.any(pos):
            dy_val = float(np.amax(dy[pos] / y[pos]))
        else:
            dy_val = 0.0
        var.dy = dy_val
        var.dydt = dy_val / var.dt if var.dt > 0 else 0.0
        return var

    # -----------------------------------------------------------------
    # Phase 16: chunked driver — interleaves the JIT'd runner with
    # host-side progress / plot / movie hooks.
    # -----------------------------------------------------------------
    def _run_chunked(self, init_state, atm_static, var, para, atm):
        """Run the integration in `print_prog_num`-sized chunks so the host
        can print / plot / record movie frames between chunks.

        Phase 16: termination has the same semantics as the single-shot
        path (count_max / runtime / converged); the chunk cap exists only
        to give the host a place to call `print_prog` etc. The
        bit-equivalence to the single-shot path is asserted by
        `tests/test_chunked_runner.py`.
        """
        chunk_size = max(int(getattr(vulcan_cfg, "print_prog_num", 100)), 1)
        count_max_static = int(self._statics.count_max)
        runtime_static = float(self._statics.runtime)
        live_plot = bool(getattr(vulcan_cfg, "use_live_plot", False))
        live_flux = bool(getattr(vulcan_cfg, "use_live_flux", False))
        use_print_prog = bool(getattr(vulcan_cfg, "use_print_prog", False))

        state = init_state
        while True:
            target = int(state.accept_count) + chunk_size
            # Cap the chunk at count_max + 1 so the chunk_done predicate
            # never fires before count_max would.
            target = min(target, count_max_static + 1)
            state = state._replace(chunk_target=jnp.int32(target))
            state = self._runner(state, atm_static)

            count_now = int(state.accept_count)
            t_now = float(state.t)

            chunk_cap_hit = count_now >= target
            count_max_hit = count_now > count_max_static
            runtime_hit = t_now > runtime_static
            terminated_for_real = count_max_hit or runtime_hit or (
                not chunk_cap_hit
            )

            if terminated_for_real:
                return state

            # Sync state to host for plot/print hooks.
            self._unpack_state(state, var, para, atm)
            if use_print_prog:
                if (not hasattr(para, "where_varies_most")
                        or para.where_varies_most is None):
                    para.where_varies_most = np.zeros_like(var.y)
                self.output.print_prog(var, para)
            if live_plot:
                self.output.plot_update(var, atm, para)
            if live_flux and bool(getattr(vulcan_cfg, "use_photo", False)):
                self.output.plot_flux_update(var, atm, para)

    # -----------------------------------------------------------------
    # Outer Python orchestration (drop-in for op.Integration.__call__)
    # -----------------------------------------------------------------
    def __call__(self, var, atm, para, make_atm):
        """Run the integration to convergence / runtime / count cap.

        Phase 10.5+: a single JIT'd JAX runner call replaces the Python
        `while not stop(): one_step()` loop. Termination, photo frequency
        switching, adaptive rtol, ring-buffered convergence, condensation,
        atmosphere refresh, ion charge balance (10.6), and fix-all-bot
        post-step clamping (10.6) all happen inside the runner; this
        method only handles pre-loop setup, the device call(s), and
        post-run unpacking + final-state diagnostics.

        Phase 16: when any of `use_print_prog`, `use_live_plot`,
        `use_live_flux`, or `use_save_movie` is on, the integration runs
        in chunks of `print_prog_num` accepted steps so the host can
        print / plot / save a movie frame between chunks. Otherwise it
        is single-shot. Both paths are bit-equivalent on the final
        state.
        """
        del make_atm  # captured into _refresh_static at init (Phase 10.3)
        validate_runtime_config(vulcan_cfg)
        self.loss_criteria = 0.0005

        # Build the JAX runner on first entry — cached for the run.
        self._ensure_runner(var, atm)
        ni = _NETWORK.ni
        nz = atm.Tco.shape[0]

        atm_static = make_atm_static(atm, ni, nz)
        init_state = self._pack_state(var, para, atm)

        # Phase 16: chunked execution is opt-in via the explicit
        # `use_chunked_runner` cfg flag. When on, the integration runs in
        # `print_prog_num`-sized chunks so the host can fire
        # `print_prog` / `plot_update` / `plot_flux_update` / movie
        # frame between chunks. Off (default), the runner is single-shot.
        # Auto-detecting chunked from the plotting flags was tried and
        # rejected — too fragile against test-suite module-cache state.
        use_chunked = bool(getattr(vulcan_cfg, "use_chunked_runner", False))

        if use_chunked:
            final_state = self._run_chunked(init_state, atm_static, var,
                                             para, atm)
        else:
            # Single JAX runner call (Phase 10.5+ path).
            final_state = self._runner(init_state, atm_static)
        self._unpack_state(final_state, var, para, atm)

        # f_dy needs y_prev (carry-supplied via _unpack_state) and the
        # final y to compute the dy / dydt diagnostics used by the
        # final-state print (and any downstream Python diagnostics).
        var = self._f_dy(var, para)

        # Determine end_case (op.py:1075-1087) for the final print.
        if para.count > vulcan_cfg.count_max:
            print("Integration not completed...\nMaximal allowed steps "
                  f"exceeded ({vulcan_cfg.count_max})!")
            para.end_case = 3
        elif var.t > vulcan_cfg.runtime:
            print("Integration not completed...\nMaximal allowed runtime "
                  f"exceeded ({vulcan_cfg.runtime} sec)!")
            para.end_case = 2
        else:
            print(f"Integration successful with {para.count} steps and "
                  f"long dy, long dydt = {var.longdy}, {var.longdydt}\n"
                  f"Actinic flux change: {var.aflux_change:.2E}")
            para.end_case = 1

        if vulcan_cfg.use_print_prog:
            # `print_prog` reads `para.where_varies_most`, which the
            # legacy `op.conv` populated but we don't (the in-runner
            # conv reduces directly to scalar longdy). Set a sentinel
            # so the read doesn't crash; downstream diagnostics that
            # need the per-(z, sp) breakdown should compute it from
            # var.y / var.y_prev directly.
            if not hasattr(para, "where_varies_most") or para.where_varies_most is None:
                para.where_varies_most = np.zeros_like(var.y)
            self.output.print_prog(var, para)

        # End-of-run summary (mirrors op.stop, op.py:1068-1088).
        # Master calls print_end_msg for end_case=1 only; print_unconverged_msg
        # is defined upstream but never invoked. We call it here for
        # end_case in (2, 3) so the summary lands for non-converged exits too.
        if para.end_case == 1:
            self.output.print_end_msg(var, para)
        elif para.end_case in (2, 3):
            self.output.print_unconverged_msg(var, para, para.end_case)
