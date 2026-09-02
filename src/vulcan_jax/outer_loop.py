"""Single-JIT outer integration loop.

The full VULCAN integration runs inside one `jax.lax.while_loop`: per-step
kernel (chem RHS, analytical Jacobian, diffusion, block-Thomas) composed
with photo update, atm refresh, condensation, ion charge balance,
fix-all-bot, adaptive rtol, photo-frequency ini→final switch, and a
ring-buffered convergence check. No Python loop, no NumPy on the hot path.

The accept/reject decision and dt formula match VULCAN-master's `op.Ros2`
exactly (including the forced-accept fallback when `dt < dt_min`).
"""

from __future__ import annotations

import time
from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .config import default_config
from . import phy_const as _phy_const

from . import network as _net_mod
from . import chem as _chem_mod
from . import photo as _photo_mod
from . import atm_refresh as _atm_refresh_mod
from . import conden as _conden_mod
from . import state as _state_mod
from .jax_step import AtmStatic, jax_ros2_step, make_atm_static
from .runtime_validation import validate_runtime_config
from ._paths import resolve_data_path


def _now() -> float:
    """Wall-clock seconds since the epoch (used for runtime print stamping)."""
    return time.time()


jax.config.update("jax_enable_x64", True)


# Underflow floor for `x / max(|denom|, .)` normalizations. Not a tuning
# knob: 1e-300 is well above the float64 denormal tail (~5e-324) and below
# any physical value, so it only keeps exact-zero divisors positive.
_UNDERFLOW_DENOM = 1e-300


# Network parsed once at module import. After editing the config `network`,
# restart Python (or reload this module) to pick it up.
_CFG = default_config()
_NETWORK = _net_mod.parse_network(str(resolve_data_path(_CFG.network)))
_NET_JAX = _chem_mod.to_jax(_NETWORK)


class ProfileVars(NamedTuple):
    """Per-profile constants (T-P, abundances, Kzz, gravity, radius,
    saturation, molecular diffusion) threaded through the carry.

    `jax.vmap` does NOT batch closure constants, so per-profile arrays must
    ride this carry, never a runner closure (a closure would share lane-0's
    atmosphere across the batch). Constant during a run; the single-profile
    path seeds the same values the closures bake. `pref_indx` stays a closure
    constant (it sizes a `jnp.arange`) and must be batch-constant; the
    emulator buckets accordingly.
    """

    # from _Statics
    n_0: jnp.ndarray  # (nz,)              total number density
    Kzz: jnp.ndarray  # (nz-1,)            eddy diffusion (cond_fn slope_min)
    atom_ini: jnp.ndarray  # (n_atoms,)    initial atom abundances (atom_loss)
    bottom_n: jnp.ndarray  # (ni,)         fix-all-bot pin (ymix[0]*n_0[0])
    fix_species_sat_mix: jnp.ndarray  # (n_fix_species, nz)
    # from AtmRefreshStatic
    r_Tco: jnp.ndarray  # (nz,)            atm-refresh temperature
    r_pico: jnp.ndarray  # (nz+1,)         atm-refresh interface pressure
    r_Dzz_top: jnp.ndarray  # (ni,)        top-interface molecular diffusion
    r_gs: jnp.ndarray  # ()               surface gravity
    r_zco_pref: jnp.ndarray  # ()          reference-layer height
    r_Rp: jnp.ndarray  # ()               planet radius
    # from CondenStatic
    c_Dg_per_re: jnp.ndarray  # (n_conden_re, nz)
    c_sat_n_per_re: jnp.ndarray  # (n_conden_re, nz)
    c_h2o_Dg: jnp.ndarray  # (nz,)
    c_h2o_sat: jnp.ndarray  # (nz,)
    c_nh3_Dg: jnp.ndarray  # (nz,)
    c_nh3_sat: jnp.ndarray  # (nz,)
    c_nh3_conden_top: jnp.ndarray  # () int32 — argmin(sat_mix['NH3'])
    # from _PhotoStatic: the only T-P-dependent photo statics; the rest is
    # star/network/grid-fixed and must be batch-constant (prepare_runstate
    # guards this). Placeholder shape (0, 1, 1) when use_photo=False.
    p_absp_T_cross: jnp.ndarray  # (n_absp_T, nz, nbin)
    p_cross_J_T: jnp.ndarray  # (n_br_T, nz, nbin)


class JaxIntegState(NamedTuple):
    """Carry state for the JIT'd accept/reject loop.

    Shapes for the HD189 reference config: nz=150, ni=69, n_atoms=4,
    nbin~2000, n_br~30. Scalars are float64 unless noted; counts are
    int32. Photo fields use placeholder shape (1, 1) when use_photo=False.
    """

    y: jnp.ndarray  # (nz, ni)        current proposed state
    y_prev: jnp.ndarray  # (nz, ni)        last accepted state (revert target on reject)
    ymix: jnp.ndarray  # (nz, ni)        mixing ratios
    dt: jnp.ndarray  # ()              step size to use for the next attempt
    t: jnp.ndarray  # ()              elapsed integration time
    delta: jnp.ndarray  # ()              truncation-error proxy of last attempt
    accept_count: jnp.ndarray  # ()  int32       accepted steps in this batch
    retry_count: jnp.ndarray  # ()  int32       retries on the in-flight step
    atom_loss: jnp.ndarray  # (n_atoms,)
    atom_loss_prev: jnp.ndarray  # (n_atoms,)
    nega_count: jnp.ndarray  # ()  int32       cumulative this batch
    loss_count: jnp.ndarray  # ()  int32
    delta_count: jnp.ndarray  # ()  int32
    small_y: jnp.ndarray  # ()              cumulative |y| of clipped-small cells
    nega_y: jnp.ndarray  # ()              cumulative |y| of clipped-negative cells

    # Photo state (lives in device memory between batches).
    # All zeros / unused when use_photo=False.
    k_arr: jnp.ndarray  # (nr+1, nz)     reaction-rate table
    tau: jnp.ndarray  # (nz+1, nbin)   optical depth
    aflux: jnp.ndarray  # (nz, nbin)     actinic flux
    sflux: jnp.ndarray  # (nz+1, nbin)   direct beam
    dflux_d: jnp.ndarray  # (nz+1, nbin)   diffuse downward
    dflux_u: jnp.ndarray  # (nz+1, nbin)   diffuse upward (carries between calls)
    prev_aflux: jnp.ndarray  # (nz, nbin)     prior aflux (for aflux_change)
    aflux_change: jnp.ndarray  # ()             max relative aflux change
    J_br: jnp.ndarray  # (n_br, nz)     per-branch J-rate (non-T)
    J_br_T: jnp.ndarray  # (n_br_T, nz)   per-branch J-rate (T-dep)
    Jion_br: jnp.ndarray  # (n_ion_br, nz) per-branch J-rate (ion)

    # Atmosphere geometry, refreshed at the END of an accepted iteration every
    # `update_frq` steps (op.py:905-907); the next body_fn splices g/dzi/Hpi/
    # top_flux/vs into AtmStatic so jax_ros2_step sees refreshed diffusion.
    g: jnp.ndarray  # (nz,)          gravity
    mu: jnp.ndarray  # (nz,)          mean molar mass (g/mol)
    Hp: jnp.ndarray  # (nz,)          pressure scale height
    dz: jnp.ndarray  # (nz,)          layer thickness
    zco: jnp.ndarray  # (nz+1,)        interface heights
    dzi: jnp.ndarray  # (nz-1,)        interface dz
    Hpi: jnp.ndarray  # (nz-1,)        interface scale height
    top_flux: jnp.ndarray  # (ni,)          diffusion-limited escape flux at TOA
    vs: jnp.ndarray  # (nz-1, ni)     settling velocity (zeroed post-condense fix)

    # Convergence + step-history. The body writes (y, t) to the ring at
    # index `accept_count % conv_step` and cond_fn reads longdy/longdydt
    # to terminate. `rtol`/`loss_criteria` ride in the carry so the
    # adaptive-rtol updates fire inside the body without retracing.
    # `update_photo_frq`/`is_final_photo_frq` drive the ini→final switch.
    y_time_ring: jnp.ndarray  # (conv_step, nz, ni) float64
    t_time_ring: jnp.ndarray  # (conv_step,)        float64
    longdy: jnp.ndarray  # ()                  float64
    longdydt: jnp.ndarray  # ()                  float64
    where_varies_most: jnp.ndarray  # (nz, ni)            float64
    longdy_seen_min: (
        jnp.ndarray
    )  # ()                  float64 — running min of longdy across accepted steps
    count_since_new_min: (
        jnp.ndarray
    )  # ()  int32           — accepted steps since longdy reached a new minimum
    rtol: jnp.ndarray  # ()                  float64
    loss_criteria: jnp.ndarray  # ()                  float64
    update_photo_frq: jnp.ndarray  # ()                  int32
    is_final_photo_frq: jnp.ndarray  # ()                  bool

    # Post-condensation fixed-species state.
    fix_species_started: jnp.ndarray  # ()                  bool
    fix_y: jnp.ndarray  # (nz, ni)           stored fixed values
    fix_mask: jnp.ndarray  # (nz, ni)           bool mask of fixed cells
    fix_pfix_idx: jnp.ndarray  # (n_fix_sp,)        int32 cold-trap indices

    # Hycean H2/He pin: one-shot snapshot at t>1e6 stored in h2he_mix
    # and applied to the bottom layer thereafter. When use_fix_H2He=False
    # the body branch is a static no-op.
    h2he_pinned: jnp.ndarray  # ()  bool
    h2he_mix: jnp.ndarray  # (2,) float64  [H2_mix, He_mix]

    # save_evolution capture buffer. When save_evolution=False the buffers
    # are length-1 placeholders that the body never writes.
    y_evo: jnp.ndarray  # (save_evo_n_max, nz, ni) float64
    t_evo: jnp.ndarray  # (save_evo_n_max,)        float64
    evo_idx: jnp.ndarray  # ()  int32  next slot to fill

    # Cap for the chunked-runner path: the body terminates when
    # `accept_count >= chunk_target`. Single-shot runs seed it to a large
    # sentinel (2**30, well above any count_max) so the cap never trips.
    chunk_target: jnp.ndarray  # ()  int32

    # Batched-runner termination state (unused by the single-profile path).
    # `is_done` freezes a finished lane while stragglers finish.
    # `termination_reason`: 0 running, 1 converged, 2 runtime, 3 step-count,
    # 4 stalled-convergence, 5 non-finite.
    is_done: jnp.ndarray  # ()  bool
    termination_reason: jnp.ndarray  # ()  int32

    # Hybrid vm_mol phase blend for jax_ros2_step: 1.0 = upwind (phase 0),
    # 0.0 = central difference (phase 1). Hybrid runs flip 1.0 -> 0.0 the
    # first time phase 0 ends (convergence, runtime, OR step-count), so a run
    # stopping via `_real_terminate` is in phase 1 -- a fixed point only if
    # phase 1 also converged. Non-hybrid runs never flip (bit-identical trace).
    hybrid_use_vm: jnp.ndarray  # ()  float64

    # Live termination budget (seeded from the static caps; the termination
    # test reads these, not the closure constants). Only the hybrid phase flip
    # mutates them, extending the budget the vm_branch op.py stop() way:
    # count_min = count+100, count_max = count+2000 (convergence) or
    # count+1000 (budget), runtime *= 1.1 (runtime). Non-hybrid runs never
    # touch them.
    count_min_dyn: jnp.ndarray  # ()  int32
    count_max_dyn: jnp.ndarray  # ()  int32
    runtime_dyn: jnp.ndarray  # ()  float64

    # Per-profile constants (see ProfileVars): must ride the carry so
    # jax.vmap batches them per lane; the body splices them into the
    # closure-baked statics.
    pv: ProfileVars


class _PhotoStatic(NamedTuple):
    """Per-run static inputs closed over by the photo-branch closure.

    `dz` is the initial photo-grid value only; the runner reads the
    refreshed `JaxIntegState.dz` instead.
    """

    photo_data: _photo_mod.PhotoData  # absp / scat cross sections
    photo_J_data: _photo_mod.PhotoJData  # J cross sections (passed for branch_keys)
    cross_J: jnp.ndarray  # (n_br, nbin)
    cross_J_T: jnp.ndarray  # (n_br_T, nz, nbin)
    branch_re_idx: jnp.ndarray  # (n_br,)   int64 — k_arr row to write
    branch_active: jnp.ndarray  # (n_br,)   bool
    branch_T_re_idx: jnp.ndarray  # (n_br_T,) int64
    branch_T_active: jnp.ndarray  # (n_br_T,) bool
    photo_ion_data: Optional[_photo_mod.PhotoJData]
    cross_Jion: jnp.ndarray  # (n_ion_br, nbin)
    ion_branch_re_idx: jnp.ndarray  # (n_ion_br,) int64 — k_arr row to write
    ion_branch_active: jnp.ndarray  # (n_ion_br,) bool
    bins: jnp.ndarray  # (nbin,)   wavelength grid (nm)
    sflux_top: jnp.ndarray  # (nbin,)   TOA stellar flux
    dz: jnp.ndarray  # (nz,)     layer thickness
    din12_indx: int  # static — wavelength split index for J integration
    dbin1: float
    dbin2: float
    mu_zenith: float  # cos(sl_angle)
    edd: float  # Eddington coefficient
    ag0: float  # asymmetry factor (0 for HD189)
    hc: float  # planck * c (erg nm)
    f_diurnal: float  # diurnal flux average (1.0 tidally locked)
    flux_atol: float  # aflux_change masking floor
    ag0_is_zero: bool  # static — selects compute_flux branch


def _compute_atom_loss(
    y: jnp.ndarray, compo_arr: jnp.ndarray, atom_ini_arr: jnp.ndarray
) -> jnp.ndarray:
    """Per-atom relative conservation residual. Returns (n_atoms,).

    `atom_loss[a] = (Σ_zi compo[i,a]*y[z,i] - atom_ini[a]) / atom_ini[a]`.
    """
    atom_sum = jnp.einsum("zi,ia->a", y, compo_arr)
    return (atom_sum - atom_ini_arr) / atom_ini_arr


def _print_column_atom_loss(cfg, y, y_ini, dz) -> None:
    """Opt-in end-of-run operator-weighted column budget (`report_column_atom_loss`).

    Reported, never gated: step acceptance keeps the unweighted
    master-parity `atom_loss`; this prints the operator-weighted column
    drift (`ini_abun.column_atom_loss`) — the quantity the discretized
    transport actually conserves on a nonuniform grid.
    """
    if not bool(getattr(cfg, "report_column_atom_loss", False)):
        return
    from .composition import atom_list as _compo_atoms
    from .ini_abun import column_atom_loss

    drift = np.asarray(column_atom_loss(y, y_ini, dz))
    loss_ex = list(getattr(cfg, "loss_ex", []) or [])
    print("column atom budget (operator-weighted; diagnostic, not a gate):")
    # Mirror print_end_msg: only the atoms this config tracks, minus loss_ex.
    for name in getattr(cfg, "atom_list", []):
        if name in _compo_atoms and name not in loss_ex:
            print(f"{name}: {drift[_compo_atoms.index(name)]:.4e} ")


def _step_size(
    dt: jnp.ndarray,
    delta: jnp.ndarray,
    rtol: float,
    dt_var_min: float,
    dt_var_max: float,
    dt_min: float,
    dt_max: float,
    safety: float = 0.9,
    zero_delta_frac: float = 0.01,
) -> jnp.ndarray:
    """Adaptive Ros2 dt update. Returns the next dt (scalar, seconds).

    I-control (default, master-faithful):
    `h_factor = clip(safety * (rtol/delta)^0.5, dt_var_min, dt_var_max)`,
    `h_new = clip(dt * h_factor, dt_min, dt_max)`; `delta == 0` substitutes
    `zero_delta_frac * rtol`. Production passes `safety`/`zero_delta_frac`
    from cfg; the defaults serve direct callers (tests / standalone).
    """
    delta_eff = jnp.where(delta < _UNDERFLOW_DENOM, zero_delta_frac * rtol, delta)
    h_factor = safety * (rtol / delta_eff) ** 0.5
    h_factor = jnp.clip(h_factor, dt_var_min, dt_var_max)
    return jnp.clip(dt * h_factor, dt_min, dt_max)


def _clip_prologue(y_in, pos_cut: float, nega_cut: float):
    """Master's per-step clip, shared by both `_make_clip_fn` branches.

    Returns `(y_clip, small_y_inc, nega_y_inc)`. Only the NORMALIZATION that
    follows differs between the gas-mask and no-mask branches, so this half is
    written once: two copies of an operation order that must stay bit-faithful
    to master is a divergence waiting to happen.

    The second rule is master's op.py:2459, which tests the POST-SOLVE ymix
    (written at op.py:2991-2993, before clip). That value is < mtol for any
    y<0 cell, so the rule reduces to "zero every negative" and needs no
    ymix argument at all. Testing the PRE-step ymix instead would leave
    negative densities and make the all_nonneg gate reject steps master
    accepts.
    """
    small_y_inc = jnp.sum(
        jnp.where((y_in < pos_cut) & (y_in >= 0), jnp.abs(y_in), 0.0)
    )
    nega_y_inc = jnp.sum(
        jnp.where((y_in > nega_cut) & (y_in <= 0), jnp.abs(y_in), 0.0)
    )
    y_clip = jnp.where((y_in < pos_cut) & (y_in >= nega_cut), 0.0, y_in)
    y_clip = jnp.where(y_clip < 0, 0.0, y_clip)
    return y_clip, small_y_inc, nega_y_inc


def _make_clip_fn(
    non_gas_present: bool,
    gas_indx_mask: jnp.ndarray,
    pos_cut: float,
    nega_cut: float,
):
    """Build a `clip(y_in) → (y_clip, ymix_new, small_inc, nega_inc)` closure.

    `non_gas_present` selects between the gas-only / total `ysum` denominators
    at closure time so the traced body keeps a single branch. THE TWO DIVIDE
    GUARDS ARE DELIBERATELY DIFFERENT and must stay so -- see each branch.
    """
    if non_gas_present:
        gas_mask_2d = gas_indx_mask  # (ni,) bool

        def clip_fn(y_in):
            y_clip, small_y_inc, nega_y_inc = _clip_prologue(
                y_in, pos_cut, nega_cut)
            ysum = jnp.sum(
                jnp.where(gas_mask_2d[None, :], y_clip, 0.0), axis=1, keepdims=True
            )
            # Normalization guard: an all-clipped gas layer gives ysum==0
            # (master divides unguarded -> inf/NaN); return 0 mixing there.
            # A condensate numerator over a 1e-300 floor would OVERFLOW, so
            # `where`, not `maximum` -- this is why the two branches differ.
            # Bit-identical for normal layers (ysum ~ n_0 >> 0).
            ymix_new = jnp.where(ysum > 0, y_clip / ysum, 0.0)
            return y_clip, ymix_new, small_y_inc, nega_y_inc
    else:

        def clip_fn(y_in):
            y_clip, small_y_inc, nega_y_inc = _clip_prologue(
                y_in, pos_cut, nega_cut)
            # No condensate column exists here, so the numerator cannot
            # overflow a tiny denominator: flooring an all-zero layer's ysum
            # is enough (0/0 guard), and it is a no-op on the physical path
            # where ysum ~ n_0 >> 1e-300.
            ysum = jnp.sum(y_clip, axis=1, keepdims=True)
            ysum = jnp.maximum(ysum, _UNDERFLOW_DENOM)
            return y_clip, y_clip / ysum, small_y_inc, nega_y_inc

    return clip_fn


def _make_aggregate_delta_fn(
    mtol: float, atol: float, zero_bot_row: bool, condense_zero_mask: jnp.ndarray
):
    """Build the scalar `delta(sol, delta_arr, ymix_old)` reducer.

    Zeros entries where ymix < mtol, sol < atol, optionally the bottom
    row (use_botflux / use_fix_sp_bot), and condensed species, then
    returns the max of (delta_arr / sol) over (sol > 0) cells.
    """
    cond_zero = jnp.asarray(condense_zero_mask, dtype=jnp.bool_)

    def agg(sol, delta_arr, ymix_old, zero_bot=None):
        masked = jnp.where(ymix_old < mtol, 0.0, delta_arr)
        masked = jnp.where(sol < atol, 0.0, masked)
        # `zero_bot` is the runtime extension of the static flag: upstream's
        # use_fix_H2He trip turns `use_fix_sp_bot` non-empty mid-run, which
        # activates its `delta[0] = 0` for the rest of the run.
        zb = jnp.bool_(zero_bot_row)
        if zero_bot is not None:
            zb = zb | zero_bot
        masked = jnp.where(zb, masked.at[0].set(0.0), masked)
        masked = jnp.where(cond_zero, 0.0, masked)
        # A zeroed numerator takes denominator 1: dividing it by a sub-atol
        # density gives a `0 * den**-2 = 0 * inf` NaN tangent that the
        # `jnp.max` JVP (a multiply by 0/1, not a select) propagates. Primal
        # unchanged (0/x == 0/1); a live numerator implies den >= atol.
        den = jnp.where(masked == 0.0, 1.0, jnp.maximum(jnp.abs(sol), _UNDERFLOW_DENOM))
        ratio = jnp.where(sol > 0, masked / den, 0.0)
        return jnp.max(ratio)

    return agg


def _make_photo_branch(photo_static: _PhotoStatic):
    """Build the in-loop photo branch closure.

    Computes tau / aflux / J inside the JIT'd body and writes the
    photolysis rows of `k_arr` directly via `.at[].set()`. Photo state
    stays on device between calls.
    """
    photo_data = photo_static.photo_data
    cross_J = photo_static.cross_J
    branch_re_idx = photo_static.branch_re_idx
    branch_active = photo_static.branch_active
    branch_T_re_idx = photo_static.branch_T_re_idx
    branch_T_active = photo_static.branch_T_active
    cross_Jion = photo_static.cross_Jion
    ion_branch_re_idx = photo_static.ion_branch_re_idx
    ion_branch_active = photo_static.ion_branch_active
    bins = photo_static.bins
    sflux_top = photo_static.sflux_top
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
        # Splice this lane's T-dependent cross sections from the carry into
        # the closure-baked photo data so a vmapped batch uses each lane's
        # atmosphere, not lane 0's (value-level no-op single-profile).
        pd = photo_data._replace(absp_T_cross=s.pv.p_absp_T_cross)

        # Optical depth (mirrors op.compute_tau via op_jax.Ros2JAX.compute_tau).
        tau_new = _photo_mod.compute_tau_jax(s.y, s.dz, pd)

        # Two-stream RT. `s.dflux_u` is the prior call's value (matches
        # op.py:2694's dflux_u-as-it-stood-before-the-up-sweep).
        aflux_new, sflux_new, dflux_d_new, dflux_u_new = _photo_mod.compute_flux_jax(
            tau_new,
            sflux_top,
            s.ymix,
            pd,
            bins,
            mu_zenith,
            edd,
            ag0,
            hc,
            s.dflux_u,
            ag0_is_zero=ag0_is_zero,
        )

        # Per-branch J-rates (flat output; no Python dict).
        J_br_new, J_br_T_new = _photo_mod.compute_J_jax_flat(
            aflux_new, cross_J, s.pv.p_cross_J_T, din12_indx, dbin1, dbin2
        )

        # Write into k_arr: k_arr[re_idx] = J * f_diurnal for each active branch.
        k_arr_new = _photo_mod.update_k_with_J(
            s.k_arr,
            J_br_new,
            J_br_T_new,
            branch_re_idx,
            branch_active,
            branch_T_re_idx,
            branch_T_active,
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

        # aflux_change: mirrors op.py:2737 / op_jax.py:94-101.
        # `s.aflux` here is the OLD aflux; we use it as `prev_aflux` in the
        # ratio. After this branch, prev_aflux <- old aflux, aflux <- new.
        mask = aflux_new > flux_atol
        diff = jnp.abs(aflux_new - s.aflux)
        ratio = jnp.where(
            mask, diff / jnp.maximum(jnp.abs(aflux_new), _UNDERFLOW_DENOM), 0.0
        )
        aflux_change_new = jnp.where(jnp.any(mask), jnp.max(ratio), jnp.float64(0.0))

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


def _make_atm_refresh_branch(refresh_static: _atm_refresh_mod.AtmRefreshStatic):
    """Standalone atm-refresh closure used only by the tests
    (`tests/test_outer_loop_atm_refresh.py`, `tests/test_atm_refresh_gravity.py`);
    production inlines these calls in `body_fn` after conden.
    """

    def atm_refresh(s: JaxIntegState) -> JaxIntegState:
        mu_new, g_new, Hp_new, dz_new, zco_new, dzi_new, Hpi_new = (
            _atm_refresh_mod.update_mu_dz_jax(s.ymix, refresh_static)
        )
        top_flux_new = _atm_refresh_mod.update_phi_esc_jax(
            s.y,
            g_new,
            Hp_new,
            s.top_flux,
            refresh_static,
        )
        return s._replace(
            mu=mu_new,
            g=g_new,
            Hp=Hp_new,
            dz=dz_new,
            zco=zco_new,
            dzi=dzi_new,
            Hpi=Hpi_new,
            top_flux=top_flux_new,
        )

    return atm_refresh


def _make_conden_branch(conden_static: _conden_mod.CondenStatic):
    """Build the in-loop conden closure.

    Operates on the post-Ros2/post-clip s.y/s.ymix. Updates the conden
    rows of `s.k_arr`, then redistributes mass via the H2O/NH3 relax
    kernels (each gated by a Python bool on conden_static).
    """

    def conden_branch(s: JaxIntegState) -> JaxIntegState:
        # Splice this lane's per-profile conden arrays from the carry into the
        # closure-baked static (vmap batches the carry, not closures); the
        # config/network-level fields stay baked and must be batch-constant.
        # nh3_conden_top is a 0-d int32 only compared against jnp.arange.
        st = conden_static._replace(
            Dg_per_re=s.pv.c_Dg_per_re,
            sat_n_per_re=s.pv.c_sat_n_per_re,
            h2o_Dg=s.pv.c_h2o_Dg,
            h2o_sat=s.pv.c_h2o_sat,
            nh3_Dg=s.pv.c_nh3_Dg,
            nh3_sat=s.pv.c_nh3_sat,
            nh3_conden_top=s.pv.c_nh3_conden_top,
            n_0=s.pv.n_0,
        )
        k_arr_new = _conden_mod.update_conden_rates(s.k_arr, s.y, st)
        y_new, ymix_new = _conden_mod.apply_h2o_relax_jax(s.y, s.ymix, s.dt, st)
        y_new, ymix_new = _conden_mod.apply_nh3_relax_jax(y_new, ymix_new, s.dt, st)
        return s._replace(y=y_new, ymix=ymix_new, k_arr=k_arr_new)

    return conden_branch


class _Statics(NamedTuple):
    """Per-run static inputs to the JAX runner.

    Closed-over by the runner — never appears in the carry. The
    convergence/termination caps and the adaptive-rtol /
    photo-frequency-switch knobs all live here.
    """

    compo_arr: jnp.ndarray  # (ni, n_atoms)
    atom_ini_arr: jnp.ndarray  # (n_atoms,)
    loss_eps: float
    pos_cut: float
    nega_cut: float
    mtol: float
    atol: float
    dt_var_min: float
    dt_var_max: float
    dt_min: float
    dt_max: float
    batch_max_retries: int  # safety cap on inner retries per accepted step

    # Convergence + termination
    conv_step: int  # ring buffer length (cfg.conv_step)
    count_min: int
    count_max: int
    use_conv_stall: bool  # enable the JAX-only stalled-convergence fallback
    conv_stall_window: int  # accepted steps without longdy improvement -> stalled
    runtime: float
    trun_min: float
    st_factor: float
    yconv_cri: float
    yconv_min: float
    slope_cri: float
    flux_cri: float
    mtol_conv: float
    conver_ignore_mask: jnp.ndarray  # (ni,) bool — species to drop from longdy
    condense_zero_conv_mask: jnp.ndarray  # (nz, ni) bool — non_gas_sp columns
    n_0: jnp.ndarray  # (nz,) — atm.n_0; seeds pv.n_0 (the (y-y_old)/n_0 ratio)
    Kzz: jnp.ndarray  # (nz-1,) — atm.Kzz; seeds pv.Kzz (slope_min recompute)

    # Photo / adaptive-rtol cadence statics. The dynamic counterparts ride
    # in the carry (s.update_photo_frq / s.rtol / s.loss_criteria).
    use_photo: bool
    use_atm_refresh: bool
    use_vm_mol: bool  # upwind molecular diffusion → refresh vm in-loop with mu
    hybrid_vm_mol: bool  # two-stage: converge upwind, then finish central-diff
    use_conden: bool
    final_update_photo_frq: int
    update_frq: int
    use_adapt_rtol: bool
    rtol_accept: float  # cfg.rtol at build; acceptance never sees adapted rtol
    rtol_min: float
    rtol_max: float
    adapt_rtol_dec_period: int
    adapt_rtol_inc_period: int
    adapt_rtol_dec: float
    adapt_rtol_inc: float
    adapt_rtol_loss_mul: float
    adapt_rtol_inc_loss_thresh: float
    photo_switch_longdy_thresh: float
    photo_switch_longdydt_thresh: float
    hycean_pin_time: float

    # _step_size knobs (from cfg).
    step_size_safety: float
    step_size_zero_delta_frac: float

    # Ion / fix-all-bot. Bools branch at trace time; when off, the
    # corresponding arrays are zero placeholders the body never reads.
    use_ion: bool
    e_idx: int  # species index of 'e' (0 if use_ion=False)
    charge_arr: jnp.ndarray  # (ni,) — compo[i]['e'] over charge_list, 0 elsewhere
    use_fix_all_bot: bool
    bottom_n: jnp.ndarray  # (ni,) — bottom_ymix * n_0[0]; pinned each step
    use_fix_sp_bot: bool
    fix_sp_bot_idx: jnp.ndarray  # (n_fix_sp_bot,) int32
    fix_sp_bot_mix: jnp.ndarray  # (n_fix_sp_bot,)
    # Hycean H2/He bottom-pin: snapshots ymix[0,H2]/ymix[0,He] at t>1e6
    # and pins them via the fix_sp_bot path. Indices are -1 sentinels
    # when the species are absent (use_fix_H2He must then be False).
    use_fix_H2He: bool
    h2_idx: int
    he_idx: int
    use_fix_species: bool
    post_conden_rtol: float
    fix_species_from_coldtrap_lev: bool
    fix_species_idx: jnp.ndarray  # (n_fix_species,) int32
    fix_species_sat_mix: jnp.ndarray  # (n_fix_species, nz)
    fix_species_wholecol: jnp.ndarray  # (n_fix_species,) bool

    # save_evolution capture. When on, the body writes (y, t) to the
    # ring every `save_evo_frq` accepted steps up to `save_evo_n_max`;
    # the populated prefix is published to var.y_time / var.t_time at
    # unpack time. When off, the buffers are length-1 placeholders.
    save_evolution: bool
    save_evo_frq: int
    save_evo_n_max: int


def _make_runner(
    net,
    statics: _Statics,
    non_gas_present: bool,
    gas_indx_mask: jnp.ndarray,
    zero_bot_row: bool,
    condense_zero_mask: jnp.ndarray,
    hydro_partial: bool,
    start_conden_time: float,
    stop_conden_time: float,
    photo_static: Optional[_PhotoStatic] = None,
    refresh_static: Optional[_atm_refresh_mod.AtmRefreshStatic] = None,
    conden_static: Optional[_conden_mod.CondenStatic] = None,
):
    """Build a JIT'd `runner(state, atm_static) -> state` that runs to
    convergence, `count_max`, or `runtime`.

    Body order per iteration (matches `op.Integration.__call__`):
      1. photo       (when accept_count % update_photo_frq == 0)
      2. Ros2 step + clip + atom_loss + delta + accept/reject
      3. conden      (on accept, when t >= start_conden_time and the
                      fix_species freeze has not fired; stop_conden_time only
                      arms that freeze -- with fix_species=[] conden runs forever)
      4. atm_refresh (on accept, when accept_count % update_frq == 0;
                      reads post-conden ymix, geometry feeds next iter)
      5. hydrostatic balance (reads post-conden ymix)
      6. ring-buffer append (on accept)
      7. recompute (longdy, longdydt) against the ring
      8. adaptive rtol
      9. photo-frequency ini→final switch

    `cond_fn` then checks `(t > runtime) | (count > count_max) |
    (ready & converged)`.
    """
    clip_fn = _make_clip_fn(
        non_gas_present, gas_indx_mask, statics.pos_cut, statics.nega_cut
    )
    agg_delta_fn = _make_aggregate_delta_fn(
        statics.mtol, statics.atol, zero_bot_row, condense_zero_mask
    )

    photo_branch = (
        _make_photo_branch(photo_static) if photo_static is not None else None
    )
    conden_branch = (
        _make_conden_branch(conden_static) if conden_static is not None else None
    )

    loss_eps = statics.loss_eps
    dt_var_min = statics.dt_var_min
    dt_var_max = statics.dt_var_max
    dt_min = statics.dt_min
    dt_max = statics.dt_max
    batch_max_retries = statics.batch_max_retries
    compo_arr = statics.compo_arr
    conv_step = statics.conv_step
    # count_min/count_max/runtime seed the carry's live budget; the
    # termination test reads the carry (*_dyn fields) so the hybrid flip can
    # extend it -- intentionally not bound as closure locals.
    use_conv_stall = statics.use_conv_stall
    conv_stall_window = statics.conv_stall_window
    trun_min = statics.trun_min
    st_factor = statics.st_factor
    yconv_cri = statics.yconv_cri
    yconv_min = statics.yconv_min
    slope_cri = statics.slope_cri
    flux_cri = statics.flux_cri
    mtol_conv = statics.mtol_conv
    conver_ignore_mask = statics.conver_ignore_mask
    condense_zero_conv_mask = statics.condense_zero_conv_mask
    use_photo_static = statics.use_photo
    use_atm_refresh_static = statics.use_atm_refresh
    use_vm_mol_static = statics.use_vm_mol
    hybrid_vm_static = statics.hybrid_vm_mol
    use_conden_static = statics.use_conden
    final_update_photo_frq = statics.final_update_photo_frq
    update_frq = statics.update_frq
    use_adapt_rtol = statics.use_adapt_rtol
    rtol_accept = jnp.float64(statics.rtol_accept)
    rtol_min = statics.rtol_min
    rtol_max = statics.rtol_max
    adapt_rtol_dec_period = int(statics.adapt_rtol_dec_period)
    adapt_rtol_inc_period = int(statics.adapt_rtol_inc_period)
    adapt_rtol_dec = float(statics.adapt_rtol_dec)
    adapt_rtol_inc = float(statics.adapt_rtol_inc)
    adapt_rtol_loss_mul = float(statics.adapt_rtol_loss_mul)
    adapt_rtol_inc_loss_thresh = float(statics.adapt_rtol_inc_loss_thresh)
    photo_switch_longdy_thresh = float(statics.photo_switch_longdy_thresh)
    photo_switch_longdydt_thresh = float(statics.photo_switch_longdydt_thresh)
    hycean_pin_time = float(statics.hycean_pin_time)
    step_size_safety = float(statics.step_size_safety)
    step_size_zero_delta_frac = float(statics.step_size_zero_delta_frac)
    use_ion_static = statics.use_ion
    e_idx_static = statics.e_idx
    charge_arr_static = statics.charge_arr
    use_fix_all_bot_static = statics.use_fix_all_bot
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
    fix_species_wholecol = statics.fix_species_wholecol

    def _activate_fix_species(s_in: JaxIntegState) -> JaxIntegState:
        nz_fix = s_in.pv.n_0.shape[0]
        n_fix = fix_species_idx.shape[0]
        fix_y_new = s_in.fix_y.at[:, fix_species_idx].set(s_in.y[:, fix_species_idx])

        if fix_species_from_coldtrap_lev:
            sat_rho = s_in.pv.n_0[None, :] * s_in.pv.fix_species_sat_mix
            cond_status = s_in.y[:, fix_species_idx].T >= sat_rho
            masked_sat = jnp.where(cond_status, s_in.pv.fix_species_sat_mix, jnp.inf)
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

    def _conv_jax(
        s: JaxIntegState, accept_count_after: jnp.ndarray
    ) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]":
        """Compute longdy diagnostics against the ring-buffer lookback target.

        Looks up the ring entry closest in time to `t * st_factor`, but
        excludes the most-recent slot to avoid comparing against itself.
        On the first iteration the ring is all-zero, which yields
        longdy ≈ |y/n_0| ~ O(1); the `ready` gate prevents acting on it.
        """
        target_t = s.t * st_factor
        diffs = jnp.abs(s.t_time_ring - target_t)
        # Bump the most-recent slot to +inf so argmin can never pick it,
        # matching VULCAN's `if indx == count-1: indx-=1` guard.
        last_idx = jnp.mod(
            jnp.maximum(accept_count_after - jnp.int32(1), jnp.int32(0)),
            jnp.int32(conv_step),
        )
        big = jnp.float64(jnp.inf)
        diffs_guarded = diffs.at[last_idx].set(big)
        indx = jnp.argmin(diffs_guarded)

        longdy_new, ratio = _longdy_reduce(
            s.y,
            s.ymix,
            s.y_time_ring[indx],
            s.pv.n_0,
            atol=statics.atol,
            mtol_conv=mtol_conv,
            ignore_mask=conver_ignore_mask[None, :],
            condense_mask=condense_zero_conv_mask,
        )
        dt_lookback = jnp.maximum(s.t - s.t_time_ring[indx], _UNDERFLOW_DENOM)
        longdydt_new = longdy_new / dt_lookback
        return longdy_new, longdydt_new, ratio

    def _convergence_ok(s: JaxIntegState):
        """Convergence predicate shared by `_real_terminate` and the hybrid
        phase-flip. Returns (is_converged, conv_normal, is_stalled).

        `slope_min` is recomputed from the live Hp because atm refresh can
        change it mid-run.

        The two branches are not equivalent, and every shipped config exits on
        the LOOSE one: measured 2026-08-27, HD189 longdy 0.09172, HD209 0.03013,
        W39b 0.09900 against `yconv_cri` 0.01 and `yconv_min` 0.1. Quote the
        realised `longdy` next to any "converged" claim rather than naming the
        criterion -- 0.099 against a 0.1 threshold is a weaker statement than
        0.01. The identical two-branch predicate is upstream's
        (.oracles/vulcan2_ncho/op.py:1056), so this is inherited.
        """
        slope_min = jnp.minimum(
            jnp.min(s.pv.Kzz / (0.1 * s.Hp[:-1]) ** 2),
            jnp.float64(1e-8),
        )
        slope_min = jnp.maximum(slope_min, jnp.float64(1e-10))

        conv_normal = (
            (s.longdy < jnp.float64(yconv_cri)) & (s.longdydt < jnp.float64(slope_cri))
        ) | ((s.longdy < jnp.float64(yconv_min)) & (s.longdydt < slope_min))
        conv_normal = conv_normal & (s.aflux_change < jnp.float64(flux_cri))

        # Stall fallback: no >=5% longdy improvement for conv_stall_window
        # accepted steps while both the historical minimum and the current
        # longdy sit below yconv_min (ULP-floor oscillation, not evolution).
        # JAX-only -- no VULCAN 2.0 / vm_branch counterpart; NO SHIPPED CONFIG
        # enables it and none ever should. Static gate: when off the predicate
        # folds away at trace time (bit-identical run).
        if use_conv_stall:
            is_stalled = (
                (s.count_since_new_min > jnp.int32(conv_stall_window))
                & (s.longdy_seen_min < jnp.float64(yconv_min))
                & (s.longdy < jnp.float64(yconv_min))
                & (s.aflux_change < jnp.float64(flux_cri))
            )
        else:
            is_stalled = jnp.zeros_like(conv_normal)
        return (conv_normal | is_stalled), conv_normal, is_stalled

    def _real_terminate(s: JaxIntegState):
        """Real (non-chunk) termination predicate + reason code.

        Reason priority matches master's stop() (op.py:1065-1085): converged
        over runtime over step-count, so a step that is both converged and at
        a cap reports success. Codes: 0 running, 1 converged, 2 runtime
        exceeded, 3 step-count exceeded, 4 stalled-convergence.
        """
        # Live budget (equals the static count_max/runtime for non-hybrid runs;
        # the hybrid phase flip resets it for phase 1).
        too_long = s.t > s.runtime_dyn
        too_many = s.accept_count > s.count_max_dyn

        is_converged, conv_normal, is_stalled = _convergence_ok(s)

        ready = (s.t > jnp.float64(trun_min)) & (s.accept_count > s.count_min_dyn)
        conv_term = ready & is_converged
        real_term = too_long | too_many | conv_term
        if hybrid_vm_static:
            # Phase 0 (upwind) NEVER terminates here: the body flips to phase 1
            # (central difference) and extends the budget instead (vm_branch
            # stop()). A run stopping through this predicate is in phase 1 --
            # a central-difference fixed point only if phase 1 converged
            # (reason 1/4). Bypass exits (host wall-clock bail-out, batched
            # non-finite freeze) can still return in phase 0.
            real_term = real_term & (s.hybrid_use_vm < jnp.float64(0.5))
        reason = jnp.where(
            conv_term & conv_normal,
            jnp.int32(1),
            jnp.where(
                conv_term & is_stalled,
                jnp.int32(4),
                jnp.where(
                    too_long,
                    jnp.int32(2),
                    jnp.where(too_many, jnp.int32(3), jnp.int32(0)),
                ),
            ),
        )
        return real_term, reason

    def cond_fn(s: JaxIntegState):
        real_term, _ = _real_terminate(s)
        # Chunk cap, used by the chunked driver to break for host
        # callbacks. Single-shot runs seed it past count_max so it never
        # trips.
        chunk_done = s.accept_count >= s.chunk_target
        return jnp.logical_not(real_term | chunk_done)

    def body_fn(s: JaxIntegState, atm_static_):
        # Gate photo on `retry_count==0` so reject loops don't re-fire it.
        # Cadence is `accept_count % update_photo_frq == 0`; the dynamic
        # update_photo_frq lives in the carry for the ini→final switch.
        if photo_branch is not None:
            photo_due = (
                (s.retry_count == jnp.int32(0))
                & (jnp.mod(s.accept_count, s.update_photo_frq) == jnp.int32(0))
                & jnp.bool_(use_photo_static)
            )
            s = jax.lax.cond(photo_due, photo_branch, lambda ss: ss, s)

        # Splice the carry's refreshed geometry (op.py:905-907) into AtmStatic
        # for this step.
        atm_step = atm_static_._replace(
            g=s.g, dzi=s.dzi, Hpi=s.Hpi, top_flux=s.top_flux, vs=s.vs
        )
        # Hybrid vm_mol: drive the vm/central blend from the carry phase
        # (jax_ros2_step reads use_vm_mol as a float multiplier). Non-hybrid
        # runs never flip it, so this equals atm_static_.use_vm_mol bit-for-bit.
        atm_step = atm_step._replace(use_vm_mol=s.hybrid_use_vm)
        # vm depends on mu (via Hpi) and g, so it must be refreshed in-loop
        # with the geometry (op.update_mu_dz); freezing it at setup biases a
        # mol-diff-dominated upper atmosphere. Its inputs change only at
        # refresh cadence, so per-step recompute reproduces upstream's cadence.
        if use_vm_mol_static and refresh_static is not None:
            atm_step = atm_step._replace(
                vm=_atm_refresh_mod.recompute_vm_jax(
                    s.g,
                    s.Hpi,
                    s.dzi,
                    atm_static_.Dzz,
                    atm_static_.ms,
                    atm_static_.alpha,
                    atm_static_.Tco,
                    refresh_static.kb,
                    refresh_static.Navo,
                )
            )

        # use_ion: master pins the electron rows inside BOTH Ros2 stages
        # (op.py:2908-2911, 2925-2926) so sol[e]=y[e], delta[e]=0; 'e' is then
        # recomputed by the post-step charge balance below. The fix_mask
        # row-pin implements exactly that; unlike fix_species, the e column
        # must NOT be overwritten with fix_y (the pinned step already returns
        # sol[e] == y[e]).
        if use_ion_static:
            e_mask = jnp.zeros_like(s.fix_mask).at[:, e_idx_static].set(True)
            step_mask = (s.fix_mask | e_mask) if use_fix_species_static else e_mask
            sol, delta_arr = jax_ros2_step(
                s.y, s.k_arr, s.dt, atm_step, net, fix_mask=step_mask
            )
            delta_arr = jnp.where(step_mask, 0.0, delta_arr)
            if use_fix_species_static:
                sol = jnp.where(s.fix_mask, s.fix_y, sol)
        elif use_fix_species_static:
            sol, delta_arr = jax_ros2_step(
                s.y, s.k_arr, s.dt, atm_step, net, fix_mask=s.fix_mask
            )
            sol = jnp.where(s.fix_mask, s.fix_y, sol)
            delta_arr = jnp.where(s.fix_mask, 0.0, delta_arr)
        else:
            sol, delta_arr = jax_ros2_step(s.y, s.k_arr, s.dt, atm_step, net)

        # Bottom-layer pins are applied to `sol` BEFORE the truncation error
        # and the clip (exoclime@80f75b9 op.py:2935-2946): the carried ymix
        # and the hydrostatic rebalance then see the pinned value normalized
        # with its layer, exactly as upstream. use_fix_H2He (op.py:2935-2941)
        # snapshots the pre-step ymix[0] once `t > hycean_pin_time` and turns
        # the fix_sp_bot pin (and its `delta[0] = 0`) on for the rest of the
        # run; no accept gate, since upstream snapshots before accept/reject.
        if use_fix_sp_bot_static:
            sol = sol.at[0, fix_sp_bot_idx_static].set(
                fix_sp_bot_mix_static * s.pv.n_0[0]
            )
        if use_fix_H2He_static:
            trip = (~s.h2he_pinned) & (s.t > jnp.float64(hycean_pin_time))
            h2_mix_snap = jnp.where(trip, s.ymix[0, h2_idx_static], s.h2he_mix[0])
            he_mix_snap = jnp.where(trip, s.ymix[0, he_idx_static], s.h2he_mix[1])
            h2he_mix_next = jnp.stack([h2_mix_snap, he_mix_snap])
            h2he_pinned_next = s.h2he_pinned | trip
            sol = sol.at[0, h2_idx_static].set(
                jnp.where(
                    h2he_pinned_next,
                    h2_mix_snap * s.pv.n_0[0],
                    sol[0, h2_idx_static],
                )
            )
            sol = sol.at[0, he_idx_static].set(
                jnp.where(
                    h2he_pinned_next,
                    he_mix_snap * s.pv.n_0[0],
                    sol[0, he_idx_static],
                )
            )
            zero_bot_dyn = h2he_pinned_next
        else:
            h2he_pinned_next = s.h2he_pinned
            h2he_mix_next = s.h2he_mix
            zero_bot_dyn = None

        sol_clip, ymix_new, small_y_inc, nega_y_inc = clip_fn(sol)
        atom_loss_new = _compute_atom_loss(sol_clip, compo_arr, s.pv.atom_ini)
        # delta uses the PRE-clip sol (master computes it before clip):
        # sol_clip would erase the truncation error of cells about to clip to
        # zero and let overly aggressive steps through (HD209 exercises this).
        delta = agg_delta_fn(sol, delta_arr, s.ymix, zero_bot_dyn)

        # Acceptance uses the BUILD-TIME rtol, never the carry's adapted one:
        # upstream binds `rtol = vulcan_cfg.rtol` as a default argument of
        # `step_ok`/`step_reject` at import (exoclime@80f75b9 op.py:2489,2495;
        # vm_branch@84d010d op.py:2568,2574), so the adaptive-rtol and
        # `post_conden_rtol` writes reach only `step_size`, which reads the
        # live value. `s.rtol` therefore feeds `_step_size` alone.
        all_nonneg = jnp.all(sol_clip >= 0)
        loss_diff = jnp.max(jnp.abs(atom_loss_new - s.atom_loss_prev))
        accept = all_nonneg & (loss_diff < loss_eps) & (delta <= rtol_accept)

        # Force-accept when shrinking dt would underflow or we've burned the
        # retry budget — prevents the runner from getting permanently stuck.
        next_dt_if_reject = s.dt * dt_var_min
        dt_underflow = next_dt_if_reject < dt_min
        retry_exhausted = s.retry_count >= jnp.int32(batch_max_retries)
        force_accept = (dt_underflow | retry_exhausted) & ~accept
        do_accept = accept | force_accept

        # Exactly one counter increments per FAILED attempt, classified on
        # `~accept` (not `~do_accept`) so a force-accepted failure still
        # counts -- master bumps its reject counter before the dt<dt_min
        # force-accept clamp (op.py:2495-2518).
        delta_too_big = delta > rtol_accept
        any_neg = jnp.any(sol_clip < 0)
        attempt_failed = ~accept
        delta_count_inc = (attempt_failed & delta_too_big).astype(jnp.int32)
        nega_count_inc = (attempt_failed & ~delta_too_big & any_neg).astype(jnp.int32)
        loss_count_inc = (attempt_failed & ~delta_too_big & ~any_neg).astype(jnp.int32)

        # Time advance: dt for accept, dt_min for force_accept, 0 for reject.
        dt_used_for_t = jnp.where(force_accept, jnp.float64(dt_min), s.dt)
        t_next = jnp.where(do_accept, s.t + dt_used_for_t, s.t)

        # Master's step_reject resets only var.y at dt_min; ymix/atom_loss
        # keep the rejected clipped solve. So the force-accept path must feed
        # downstream work y_prev for y but the new clipped ymix.
        y_prev_clipped = jnp.where(s.y_prev < 0, 0.0, s.y_prev)
        y_post_clip = jnp.where(force_accept, y_prev_clipped, sol_clip)

        # Conden gates on the pre-step `s.t` (not `t_next`); master's
        # save_step advances var.t AFTER conden runs, so this matches.
        if conden_branch is not None:
            s_post = s._replace(y=y_post_clip, ymix=ymix_new)
            in_conden_window = s.t >= jnp.float64(start_conden_time)
            fire_conden = (
                do_accept
                & in_conden_window
                & ~s.fix_species_started
                & jnp.bool_(use_conden_static)
            )
            s_post = jax.lax.cond(
                fire_conden,
                conden_branch,
                lambda ss: ss,
                s_post,
            )
            if use_fix_species_static:
                trigger_fix = (
                    do_accept
                    & ~s.fix_species_started
                    & in_conden_window
                    & (s.t > jnp.float64(stop_conden_time))
                    & jnp.bool_(use_conden_static)
                )
                s_post = jax.lax.cond(
                    trigger_fix,
                    _activate_fix_species,
                    lambda ss: ss,
                    s_post,
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
            sol_clip = y_post_clip
            k_arr_next = s.k_arr
            fix_species_started_next = s.fix_species_started
            fix_y_next = s.fix_y
            fix_mask_next = s.fix_mask
            fix_pfix_idx_next = s.fix_pfix_idx
            vs_next = s.vs
            trigger_fix = jnp.bool_(False)

        # Atm refresh (op.py:905-907): after conden, before hydrostatic
        # balance, on accepted steps only. `s.accept_count` is pre-increment,
        # matching master's `count % update_frq == 0` cadence.
        if refresh_static is not None:
            refresh_due = (
                do_accept
                & (jnp.mod(s.accept_count, jnp.int32(update_frq)) == jnp.int32(0))
                & jnp.bool_(use_atm_refresh_static)
            )
            # Splice this lane's per-profile atmosphere from the carry into
            # the closure-baked refresh static (vmap batches the carry, not
            # closures). `pref_indx` stays baked and must be batch-constant
            # (prepare_runstate rejects a mismatch).
            refresh_lane = refresh_static._replace(
                Tco=s.pv.r_Tco,
                pico=s.pv.r_pico,
                Dzz_top=s.pv.r_Dzz_top,
                gs=s.pv.r_gs,
                zco_pref=s.pv.r_zco_pref,
                Rp=s.pv.r_Rp,
            )

            def _do_refresh(_):
                mu_n, g_n, Hp_n, dz_n, zco_n, dzi_n, Hpi_n = (
                    _atm_refresh_mod.update_mu_dz_jax(ymix_new, refresh_lane)
                )
                top_flux_n = _atm_refresh_mod.update_phi_esc_jax(
                    sol_clip,
                    g_n,
                    Hp_n,
                    s.top_flux,
                    refresh_lane,
                )
                return mu_n, g_n, Hp_n, dz_n, zco_n, dzi_n, Hpi_n, top_flux_n

            def _no_refresh(_):
                return (s.mu, s.g, s.Hp, s.dz, s.zco, s.dzi, s.Hpi, s.top_flux)

            (
                mu_next,
                g_next,
                Hp_next,
                dz_next,
                zco_next,
                dzi_next,
                Hpi_next,
                top_flux_next,
            ) = jax.lax.cond(
                refresh_due,
                _do_refresh,
                _no_refresh,
                operand=None,
            )
        else:
            mu_next = s.mu
            g_next = s.g
            Hp_next = s.Hp
            dz_next = s.dz
            zco_next = s.zco
            dzi_next = s.dzi
            Hpi_next = s.Hpi
            top_flux_next = s.top_flux

        n_0 = atm_step.M[:, None]
        sol_balanced_full = n_0 * ymix_new
        if hydro_partial:
            sol_balanced = jnp.where(
                gas_indx_mask[None, :], sol_balanced_full, sol_clip
            )
        else:
            sol_balanced = sol_balanced_full

        # Charge balance: with `charge_arr[i] = compo[i]['e']` for charged
        # species (and 0 for 'e' itself), `e[:] = -dot(y, charge_arr)`
        # enforces zero net charge per layer. Trace-time skip when off.
        if use_ion_static:
            e_density = -jnp.einsum("zi,i->z", sol_balanced, charge_arr_static)
            sol_balanced = sol_balanced.at[:, e_idx_static].set(e_density)

        # fix-all-bot pin: bottom layer fixed to chem-EQ mixing ratios
        # snapshotted at init, scaled by n_0[0]. Trace-time branch.
        if use_fix_all_bot_static:
            sol_balanced = sol_balanced.at[0].set(s.pv.bottom_n)

        # y / ymix / atom_loss for the next iteration.
        y_next = jnp.where(do_accept, sol_balanced, s.y_prev)
        ymix_next = ymix_new
        atom_loss_next = atom_loss_new

        # y_prev is the revert target: on accept, the fresh state; on reject,
        # keep the carry's (still retrying THIS step).
        y_prev_next = jnp.where(do_accept, y_next, s.y_prev)
        atom_loss_prev_next = jnp.where(do_accept, atom_loss_next, s.atom_loss_prev)

        accept_count_next = s.accept_count + jnp.where(
            do_accept, jnp.int32(1), jnp.int32(0)
        )
        retry_count_next = jnp.where(
            do_accept, jnp.int32(0), s.retry_count + jnp.int32(1)
        )

        # Ring append (op.save_step): slot (accept_count_next - 1) % conv_step.
        # On reject this rewrites the same slot with unchanged values.
        ring_idx = jnp.mod(
            jnp.maximum(accept_count_next - jnp.int32(1), jnp.int32(0)),
            jnp.int32(conv_step),
        )
        y_time_ring_new = s.y_time_ring.at[ring_idx].set(y_next)
        t_time_ring_new = s.t_time_ring.at[ring_idx].set(t_next)

        # save_evolution capture. Master appends every accepted step and
        # post-slices [::save_evo_frq], keeping accepted steps 1, K+1, 2K+1...
        # Gating on `(accept_count_next - 1) % K == 0` reproduces that (first
        # accepted step always captured). Compiles out when off.
        if save_evolution_static:
            do_evo_append = (
                do_accept
                & (
                    jnp.mod(
                        accept_count_next - jnp.int32(1), jnp.int32(save_evo_frq_static)
                    )
                    == jnp.int32(0)
                )
                & (s.evo_idx < jnp.int32(save_evo_n_max_static))
            )
            evo_slot = s.evo_idx
            zero_i32 = jnp.int32(0)
            y_evo_new = jax.lax.cond(
                do_evo_append,
                lambda yev: jax.lax.dynamic_update_slice(
                    yev, y_next[None, :, :], (evo_slot, zero_i32, zero_i32)
                ),
                lambda yev: yev,
                s.y_evo,
            )
            t_evo_new = jax.lax.cond(
                do_evo_append,
                lambda tev: tev.at[evo_slot].set(t_next),
                lambda tev: tev,
                s.t_evo,
            )
            evo_idx_new = s.evo_idx + jnp.where(
                do_evo_append, jnp.int32(1), jnp.int32(0)
            )
        else:
            y_evo_new = s.y_evo
            t_evo_new = s.t_evo
            evo_idx_new = s.evo_idx

        # We need a snapshot state with (y, t, ring) updated to feed _conv_jax.
        s_for_conv = s._replace(
            y=y_next,
            ymix=ymix_next,
            t=t_next,
            y_time_ring=y_time_ring_new,
            t_time_ring=t_time_ring_new,
        )
        longdy_new_val, longdydt_new_val, where_varies_most_new = _conv_jax(
            s_for_conv,
            accept_count_next,
        )
        # Only refresh longdy/longdydt on accepted steps — rejected steps
        # don't change the ring contents in any meaningful way.
        longdy_next = jnp.where(do_accept, longdy_new_val, s.longdy)
        longdydt_next = jnp.where(do_accept, longdydt_new_val, s.longdydt)
        where_varies_most_next = jnp.where(
            do_accept,
            where_varies_most_new,
            s.where_varies_most,
        )
        # Stall bookkeeping: require a >=5% relative drop to count as a new
        # minimum (strict less-than would let ULP-floor jitter reset the
        # counter forever). Gate on master's ready predicate (op.py:1069) so
        # the early transient never charges the stall window.
        stall_ready = (s.t > jnp.float64(trun_min)) & (
            accept_count_next > s.count_min_dyn
        )
        significant_drop = (
            do_accept
            & stall_ready
            & (longdy_next < s.longdy_seen_min * jnp.float64(0.95))
        )
        longdy_seen_min_next = jnp.where(
            significant_drop, longdy_next, s.longdy_seen_min
        )
        count_since_new_min_next = jnp.where(
            significant_drop,
            jnp.int32(0),
            jnp.where(
                do_accept & stall_ready,
                s.count_since_new_min + jnp.int32(1),
                s.count_since_new_min,
            ),
        )

        # Hybrid vm_mol phase flip: when phase 0 (upwind) ends -- convergence,
        # runtime, or step-count -- switch to central difference, reset the
        # convergence trackers, and extend the budget the vm_branch stop() way:
        #   convergence -> count_min=count+100, count_max=count+2000
        #   runtime     -> count_min=count+100, count_max=count+1000, runtime*=1.1
        #   step-count  -> count_min=count+100, count_max=count+1000
        # Dropped at trace time for non-hybrid runs.
        hybrid_use_vm_next = s.hybrid_use_vm
        count_min_dyn_next = s.count_min_dyn
        count_max_dyn_next = s.count_max_dyn
        runtime_dyn_next = s.runtime_dyn
        if hybrid_vm_static:
            s_after = s._replace(
                longdy=longdy_next,
                longdydt=longdydt_next,
                t=t_next,
                accept_count=accept_count_next,
                Hp=Hp_next,
            )
            is_conv_after, _, _ = _convergence_ok(s_after)
            in_phase0 = s.hybrid_use_vm > jnp.float64(0.5)
            ready_after = (t_next > jnp.float64(trun_min)) & (
                accept_count_next > s.count_min_dyn
            )
            # Priority matches vm_branch stop(): convergence > runtime > count.
            conv_flip = in_phase0 & do_accept & ready_after & is_conv_after
            runtime_flip = in_phase0 & (t_next > s.runtime_dyn) & ~conv_flip
            count_flip = (
                in_phase0
                & (accept_count_next > s.count_max_dyn)
                & ~conv_flip
                & ~runtime_flip
            )
            do_flip = conv_flip | runtime_flip | count_flip

            new_count_min = accept_count_next + jnp.int32(100)
            count_max_conv = accept_count_next + jnp.int32(2000)
            count_max_budget = accept_count_next + jnp.int32(1000)

            hybrid_use_vm_next = jnp.where(do_flip, jnp.float64(0.0), s.hybrid_use_vm)
            count_min_dyn_next = jnp.where(do_flip, new_count_min, s.count_min_dyn)
            count_max_dyn_next = jnp.where(
                conv_flip,
                count_max_conv,
                jnp.where(runtime_flip | count_flip, count_max_budget, s.count_max_dyn),
            )
            runtime_dyn_next = jnp.where(
                runtime_flip, s.runtime_dyn * jnp.float64(1.1), s.runtime_dyn
            )
            longdy_next = jnp.where(do_flip, jnp.float64(jnp.inf), longdy_next)
            longdydt_next = jnp.where(do_flip, jnp.float64(jnp.inf), longdydt_next)
            longdy_seen_min_next = jnp.where(
                do_flip, jnp.float64(jnp.inf), longdy_seen_min_next
            )
            count_since_new_min_next = jnp.where(
                do_flip, jnp.int32(0), count_since_new_min_next
            )

        # Adaptive rtol (accepted steps only): periodic decrease while
        # |atom_loss| >= loss_criteria; periodic increase below a lower
        # atom-loss threshold.
        max_atom_loss = jnp.max(jnp.abs(atom_loss_new))
        do_dec = (
            jnp.bool_(use_adapt_rtol)
            & do_accept
            & (
                jnp.mod(s.accept_count, jnp.int32(adapt_rtol_dec_period))
                == jnp.int32(0)
            )
            & (max_atom_loss >= s.loss_criteria)
        )
        rtol_dec = jnp.maximum(
            s.rtol * jnp.float64(adapt_rtol_dec), jnp.float64(rtol_min)
        )
        loss_crit_dec = s.loss_criteria * jnp.float64(adapt_rtol_loss_mul)
        rtol_after_dec = jnp.where(do_dec, rtol_dec, s.rtol)
        loss_criteria_after_dec = jnp.where(do_dec, loss_crit_dec, s.loss_criteria)

        do_inc = (
            jnp.bool_(use_adapt_rtol)
            & do_accept
            & (
                jnp.mod(s.accept_count, jnp.int32(adapt_rtol_inc_period))
                == jnp.int32(0)
            )
            & (s.accept_count > jnp.int32(0))
            & (max_atom_loss < jnp.float64(adapt_rtol_inc_loss_thresh))
        )
        rtol_inc = jnp.minimum(
            rtol_after_dec * jnp.float64(adapt_rtol_inc), jnp.float64(rtol_max)
        )
        rtol_adapt = jnp.where(do_inc, rtol_inc, rtol_after_dec)
        rtol_next = jnp.where(trigger_fix, jnp.float64(post_conden_rtol), rtol_adapt)

        # Step-size control runs *after* adaptive rtol so the post-update
        # tolerance applies to the next step.
        dt_after_normal = _step_size(
            s.dt,
            delta,
            rtol_next,
            dt_var_min,
            dt_var_max,
            dt_min,
            dt_max,
            step_size_safety,
            step_size_zero_delta_frac,
        )
        dt_after_force = _step_size(
            jnp.float64(dt_min),
            delta,
            rtol_next,
            dt_var_min,
            dt_var_max,
            dt_min,
            dt_max,
            step_size_safety,
            step_size_zero_delta_frac,
        )
        dt_next = jnp.where(
            force_accept,
            dt_after_force,
            jnp.where(accept, dt_after_normal, next_dt_if_reject),
        )

        # Photo-frequency ini→final switch when longdy / longdydt drop
        # below their respective cfg.photo_switch_* thresholds.
        # Upstream evaluates `conv()` (which writes longdy/longdydt) only once
        # `t > trun_min and count > count_min` (op.py:1069); before that both
        # sit at their initial 1.0 and the switch cannot fire. Mirror that
        # readiness gate instead of the always-updated ring-buffer longdy.
        ready_next = (t_next > jnp.float64(trun_min)) & (
            accept_count_next > count_min_dyn_next
        )
        switch_to_final = (
            jnp.bool_(use_photo_static)
            & ~s.is_final_photo_frq
            & ready_next
            & (longdy_next < jnp.float64(photo_switch_longdy_thresh))
            & (longdydt_next < jnp.float64(photo_switch_longdydt_thresh))
        )
        update_photo_frq_next = jnp.where(
            switch_to_final, jnp.int32(final_update_photo_frq), s.update_photo_frq
        )
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
            small_y=s.small_y + small_y_inc,
            nega_y=s.nega_y + nega_y_inc,
            k_arr=k_arr_next,
            y_time_ring=y_time_ring_new,
            t_time_ring=t_time_ring_new,
            longdy=longdy_next,
            longdydt=longdydt_next,
            where_varies_most=where_varies_most_next,
            longdy_seen_min=longdy_seen_min_next,
            count_since_new_min=count_since_new_min_next,
            hybrid_use_vm=hybrid_use_vm_next,
            count_min_dyn=count_min_dyn_next,
            count_max_dyn=count_max_dyn_next,
            runtime_dyn=runtime_dyn_next,
            rtol=rtol_next,
            loss_criteria=loss_criteria_after_dec,
            update_photo_frq=update_photo_frq_next,
            is_final_photo_frq=is_final_next,
            mu=mu_next,
            g=g_next,
            Hp=Hp_next,
            dz=dz_next,
            zco=zco_next,
            dzi=dzi_next,
            Hpi=Hpi_next,
            top_flux=top_flux_next,
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
            # chunk_target: driver-set, never mutated in the body.
            chunk_target=s.chunk_target,
        )

    @jax.jit
    def runner(state: JaxIntegState, atm_static: AtmStatic):
        # One-shot run to convergence / count_max / runtime.
        final = jax.lax.while_loop(
            cond_fn,
            lambda s: body_fn(s, atm_static),
            state,
        )
        # Re-evaluate the reason once on the terminal state -- the only way
        # the single-profile path can tell normal (1) from stall (4)
        # convergence. A chunked exit reports 0 (still running).
        _real_term, reason = _real_terminate(final)
        return final._replace(termination_reason=reason)

    def cond_fn_batch(s: JaxIntegState):
        # Per-lane stop predicate: gates ONLY on carry flags, never re-derives
        # `real_term` -- the body must run one final (frozen, no-op) iteration
        # on the terminal state to record is_done / termination_reason.
        # Under vmap the loop runs while ANY lane is live.
        chunk_reached = s.accept_count >= s.chunk_target
        return jnp.logical_not(s.is_done | chunk_reached)

    def body_fn_batch(s: JaxIntegState, atm_static_):
        # vmap applies the body to EVERY lane each iteration until the slowest
        # finishes, so finished lanes must be frozen: advance all lanes, then
        # keep the pre-step carry `s` for lanes that are done / terminate now /
        # hit their chunk yield / went non-finite. Freezing on `s` makes each
        # lane bit-identical to its solo run.
        real_term, reason = _real_terminate(s)
        chunk_reached = s.accept_count >= s.chunk_target
        s_adv = body_fn(s, atm_static_)
        nan_now = jnp.logical_not(jnp.all(jnp.isfinite(s_adv.y)))

        already_done = s.is_done
        # Non-finite lanes freeze at the last good carry `s` (reason 5); the
        # bad `s_adv` is discarded.
        became_nan = nan_now & ~already_done & ~real_term & ~chunk_reached
        keep_old = already_done | real_term | chunk_reached | became_nan
        frozen = jax.tree_util.tree_map(
            lambda o, n: jnp.where(keep_old, o, n), s, s_adv
        )
        is_done_next = already_done | real_term | became_nan
        reason_next = jnp.where(
            already_done,
            s.termination_reason,
            jnp.where(
                real_term,
                reason,
                jnp.where(became_nan, jnp.int32(5), jnp.int32(0)),
            ),
        )
        return frozen._replace(is_done=is_done_next, termination_reason=reason_next)

    def runner_batch(state: JaxIntegState, atm_static: AtmStatic):
        # Freeze-on-done while_loop; NOT jitted here -- run_batch wraps it in
        # jax.vmap + jax.jit with the right in_axes.
        return jax.lax.while_loop(
            cond_fn_batch,
            lambda s: body_fn_batch(s, atm_static),
            state,
        )

    return runner, runner_batch


# in_axes for vmapping the batched runner over AtmStatic: array leaves batch
# on axis 0; the four toggle flags broadcast (None) -- identical within a
# (nz, toggle-combo) batch and consumed only as traced values, never in a
# Python `if`.
_ATM_STATIC_BATCH_AXES = AtmStatic(
    Kzz=0,
    Dzz=0,
    dzi=0,
    vz=0,
    Hpi=0,
    Ti=0,
    Tco=0,
    g=0,
    ms=0,
    alpha=0,
    M=0,
    vm=0,
    vs=0,
    top_flux=0,
    bot_flux=0,
    bot_vdep=0,
    gas_indx_mask=0,
    diff_esc_mask=0,
    use_vm_mol=None,
    use_settling=None,
    use_topflux=None,
    use_botflux=None,
)


def stack_integ_states(states: "list[JaxIntegState]") -> JaxIntegState:
    """Stack single-profile `JaxIntegState`s into one batched state.

    Every leaf gains a leading batch axis. All inputs must share identical
    leaf shapes (same nz / ni / network), which the emulator guarantees by
    bucketing on `nz` before calling.
    """
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def unstack_integ_states(batched: JaxIntegState, n: int) -> "list[JaxIntegState]":
    """Split a batched `JaxIntegState` into `n` single-profile states."""
    return [jax.tree_util.tree_map(lambda x, i=i: x[i], batched) for i in range(n)]


def stack_atm_statics(atms: "list[AtmStatic]") -> AtmStatic:
    """Stack single-profile `AtmStatic`s into one batched `AtmStatic`.

    Array leaves gain a leading batch axis; the four toggle flags are kept as
    scalars (broadcast under vmap via `_ATM_STATIC_BATCH_AXES`) and must be
    identical across the batch — bucket by toggle-combo before calling.
    """
    first = atms[0]
    flags = ("use_vm_mol", "use_settling", "use_topflux", "use_botflux")
    for a in atms[1:]:
        if any(getattr(a, f) != getattr(first, f) for f in flags):
            raise ValueError(
                "stack_atm_statics: AtmStatic toggle flags differ across the "
                "batch; bucket profiles by toggle-combo before stacking."
            )
    array_fields = (
        "Kzz",
        "Dzz",
        "dzi",
        "vz",
        "Hpi",
        "Ti",
        "Tco",
        "g",
        "ms",
        "alpha",
        "M",
        "vm",
        "vs",
        "top_flux",
        "bot_flux",
        "bot_vdep",
        "gas_indx_mask",
        "diff_esc_mask",
    )
    stacked = {
        name: jnp.stack([jnp.asarray(getattr(a, name)) for a in atms], axis=0)
        for name in array_fields
    }
    return first._replace(**stacked)


def _longdy_reduce(
    y, ymix, y_old, n_0, *, atol, mtol_conv, ignore_mask=None, condense_mask=None
):
    """The `longdy` convergence reduction. Returns (longdy, ratio).

    Module level so the regression guards exercise the shipped code rather
    than an in-test copy of it.
    """
    longdy_arr = jnp.abs((y - y_old) / n_0[:, None])
    longdy_arr = jnp.where(ymix < mtol_conv, 0.0, longdy_arr)
    longdy_arr = jnp.where(y < atol, 0.0, longdy_arr)
    if ignore_mask is not None:
        longdy_arr = jnp.where(ignore_mask, 0.0, longdy_arr)
    if condense_mask is not None:
        longdy_arr = jnp.where(condense_mask, 0.0, longdy_arr)

    # Zeroed numerator -> denominator 1, for the same reason as in
    # `_make_aggregate_delta_fn`: a sub-mtol_conv ymix would make the tangent
    # `0 * inf` and the max JVP does not discard it. Primal unchanged.
    den = jnp.where(longdy_arr == 0.0, 1.0, jnp.maximum(ymix, _UNDERFLOW_DENOM))
    ratio = jnp.where(ymix > 0, longdy_arr / den, 0.0)
    longdy = jnp.max(ratio)
    # NaN guard: the masks above are all False for NaN, so an all-NaN state
    # would read longdy == 0.0 ("converged"). Force +inf so a poisoned run can
    # never converge and exits via the count/runtime ladder, matching master's
    # raise on an empty amax (op.py:1055).
    state_is_bad = ~jnp.all(jnp.isfinite(y)) | ~jnp.all(jnp.isfinite(ymix))
    return jnp.where(state_is_bad, jnp.inf, longdy), ratio


class OuterLoop:
    """Standalone outer-integration driver. One JIT'd `lax.while_loop`
    runs every accepted step with internal retries, photo / atm-refresh /
    conden updates, ring-buffered convergence, and adaptive rtol."""

    def __init__(self, odesolver, output, cfg=None):
        # cfg defaults to the process default (CLI / legacy callers);
        # load_config() users pass their own namespace so every runtime knob
        # reads from the cfg the RunState was built with (setup counterpart:
        # state._cfg_overlay). The import-locked network is the one knob cfg
        # cannot change here.
        self._cfg = cfg if cfg is not None else default_config()
        self.mtol = float(self._cfg.mtol)
        self.atol = float(self._cfg.atol)
        self.output = output
        self.odesolver = odesolver
        self.loss_criteria = float(getattr(self._cfg, "loss_criteria", 0.0005))

        self._species = list(_NETWORK.species)

        # Atom ordering captured ONCE at init; dict↔array conversion relies on it.
        self._atom_order = [
            a for a in self._cfg.atom_list if a not in getattr(self._cfg, "loss_ex", [])
        ]

        # compo_arr (ni, n_atoms): rows = species, cols = atom_order.
        from . import composition as _ba

        compo = _ba.compo
        compo_row = _ba.compo_row
        ni_ = _NETWORK.ni
        compo_np = np.zeros((ni_, len(self._atom_order)), dtype=np.float64)
        for i, sp in enumerate(self._species):
            r_i = compo_row.index(sp)
            for a, atom in enumerate(self._atom_order):
                compo_np[i, a] = float(compo[r_i][atom])
        self._compo_arr = compo_np

        # Full charge column from compo. Per-run `charge_arr` (zeroed for
        # species outside var.charge_list) is built in _build_statics once
        # `var` is available.
        self._compo_charge = np.array(
            [float(compo[compo_row.index(sp)]["e"]) for sp in self._species],
            dtype=np.float64,
        )

        self._non_gas_present = bool(self._cfg.non_gas_sp)
        self._zero_bot_row = bool(self._cfg.use_botflux or self._cfg.use_fix_sp_bot)

        # Lazy runner cache; populated on first call and reused thereafter.
        self._runner = None
        # Un-jitted freeze-on-done while_loop for the vmapped batched path,
        # and its jax.vmap+jax.jit wrapper (`run_batch`). Both ride the same
        # (nz, toggle-combo) closure as `_runner`.
        self._runner_batch = None
        self._vrunner = None
        self._statics = None
        self._photo_static = None
        self._refresh_static = None
        self._conden_static = None
        self._live_ui = None
        # First batched photo profile's TOA stellar flux; prepare_runstate
        # rejects later profiles with a different star (see the guard there).
        self._sflux_top_ref = None
        # When use_condense=True, only gas columns get rebalanced after Ros2.
        self._hydro_partial = bool(self._cfg.use_condense)

    def reset(self) -> None:
        """Drop the cached JIT'd runner so the next call re-traces against
        a possibly-mutated `self._cfg` (notebooks, parameter sweeps).
        Without this, the runner closure pins the original config."""
        self._runner = None
        self._runner_batch = None
        self._vrunner = None
        self._statics = None
        self._photo_static = None
        self._refresh_static = None
        self._conden_static = None
        self._live_ui = None
        self._sflux_top_ref = None

    def _build_statics(self, var, atm) -> _Statics:
        """Pack scalar config + per-run arrays into the closure inputs."""
        ni = _NETWORK.ni
        atom_ini_arr = np.asarray(
            [float(var.atom_ini[a]) for a in self._atom_order],
            dtype=np.float64,
        )

        # conver_ignore species are zeroed in the longdy reduction (op.py:1045-1046).
        conver_ignore_np = np.zeros(ni, dtype=bool)
        for sp in getattr(self._cfg, "conver_ignore", []):
            if sp in _NETWORK.species_idx:
                conver_ignore_np[_NETWORK.species_idx[sp]] = True

        # condense_zero_conv (op.py:1048-1049): when use_condense, the
        # non_gas_sp columns are zeroed in longdy. HD189 has non_gas_sp=[].
        nz = atm.Tco.shape[0]
        cond_zero_conv_np = np.zeros((nz, ni), dtype=bool)
        if self._cfg.use_condense:
            for sp in self._cfg.non_gas_sp:
                if sp in _NETWORK.species_idx:
                    cond_zero_conv_np[:, _NETWORK.species_idx[sp]] = True

        # Ion / fix-all-bot masks are zeros when their flag is off; the body's
        # Python `if use_*_static:` skips them at trace time.
        use_ion = bool(self._cfg.use_ion)
        if use_ion:
            charge_list = list(getattr(var, "charge_list", []))
            charge_np = np.zeros(ni, dtype=np.float64)
            for sp in charge_list:
                if sp in _NETWORK.species_idx:
                    charge_np[_NETWORK.species_idx[sp]] = self._compo_charge[
                        _NETWORK.species_idx[sp]
                    ]
            e_idx = _NETWORK.species_idx["e"] if "e" in _NETWORK.species_idx else 0
            # Exclude 'e' from the charge column so `e[:] = -dot(y, charge_arr)`
            # is consistent (op.py:3001 zeros e first).
            charge_np[e_idx] = 0.0
        else:
            charge_np = np.zeros(ni, dtype=np.float64)
            e_idx = 0

        use_fix_all_bot = bool(getattr(self._cfg, "use_fix_all_bot", False))
        if use_fix_all_bot:
            # Pin bottom layer to chemical-EQ mixing ratios captured at
            # init time, scaled by static n_0[0] (op.py:3019, 3047-3048; a solver upstream never selects,
            # see naming_solver op.py:3080-3083).
            bottom_n_np = np.asarray(var.ymix[0], dtype=np.float64) * float(atm.n_0[0])
        else:
            bottom_n_np = np.zeros(ni, dtype=np.float64)

        fix_sp_bot_cfg = getattr(self._cfg, "use_fix_sp_bot", {}) or {}
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

        use_fix_H2He = bool(getattr(self._cfg, "use_fix_H2He", False))
        if use_fix_H2He:
            h2_idx = int(_NETWORK.species_idx["H2"])
            he_idx = int(_NETWORK.species_idx["He"])
        else:
            h2_idx = -1
            he_idx = -1

        fix_species_cfg = list(getattr(self._cfg, "fix_species", []) or [])
        use_fix_species = bool(self._cfg.use_condense and fix_species_cfg)
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
                    fix_species_sat_mix[i] = np.asarray(
                        atm.sat_mix[sp], dtype=np.float64
                    )
        else:
            fix_species_idx = np.zeros((0,), dtype=np.int32)
            fix_species_sat_mix = np.zeros((0, nz), dtype=np.float64)
            fix_species_wholecol = np.zeros((0,), dtype=bool)

        return _Statics(
            compo_arr=jnp.asarray(self._compo_arr),
            atom_ini_arr=jnp.asarray(atom_ini_arr),
            loss_eps=float(self._cfg.loss_eps),
            pos_cut=float(self._cfg.pos_cut),
            nega_cut=float(self._cfg.nega_cut),
            mtol=float(self.mtol),
            atol=float(self.atol),
            dt_var_min=float(self._cfg.dt_var_min),
            dt_var_max=float(self._cfg.dt_var_max),
            dt_min=float(self._cfg.dt_min),
            dt_max=float(self._cfg.dt_max),
            batch_max_retries=int(getattr(self._cfg, "batch_max_retries", 110)),
            conv_step=int(self._cfg.conv_step),
            count_min=int(self._cfg.count_min),
            count_max=int(self._cfg.count_max),
            # Default FALSE: the stall fallback has no VULCAN counterpart and
            # no shipped config enables it; opting in must be deliberate.
            use_conv_stall=bool(getattr(self._cfg, "use_conv_stall", False)),
            conv_stall_window=int(getattr(self._cfg, "conv_stall_window", 200)),
            runtime=float(self._cfg.runtime),
            trun_min=float(self._cfg.trun_min),
            st_factor=float(self._cfg.st_factor),
            yconv_cri=float(self._cfg.yconv_cri),
            yconv_min=float(self._cfg.yconv_min),
            slope_cri=float(self._cfg.slope_cri),
            flux_cri=float(self._cfg.flux_cri),
            mtol_conv=float(self._cfg.mtol_conv),
            conver_ignore_mask=jnp.asarray(conver_ignore_np),
            condense_zero_conv_mask=jnp.asarray(cond_zero_conv_np),
            n_0=jnp.asarray(atm.n_0, dtype=jnp.float64),
            Kzz=jnp.asarray(atm.Kzz, dtype=jnp.float64),
            use_photo=bool(self._cfg.use_photo),
            use_atm_refresh=True,
            use_vm_mol=bool(
                getattr(self._cfg, "use_vm_mol", False)
                and getattr(self._cfg, "use_moldiff", True)
            ),
            hybrid_vm_mol=bool(
                getattr(self._cfg, "use_vm_mol", False)
                and getattr(self._cfg, "use_hybrid_vm_mol", False)
                and getattr(self._cfg, "use_moldiff", True)
            ),
            use_conden=bool(self._cfg.use_condense),
            final_update_photo_frq=int(getattr(self._cfg, "final_update_photo_frq", 5)),
            update_frq=int(self._cfg.update_frq),
            use_adapt_rtol=bool(getattr(self._cfg, "use_adapt_rtol", False)),
            rtol_accept=float(self._cfg.rtol),
            rtol_min=float(getattr(self._cfg, "rtol_min", 0.0)),
            rtol_max=float(getattr(self._cfg, "rtol_max", 1.0)),
            adapt_rtol_dec_period=int(getattr(self._cfg, "adapt_rtol_dec_period", 10)),
            adapt_rtol_inc_period=int(
                getattr(self._cfg, "adapt_rtol_inc_period", 1000)
            ),
            adapt_rtol_dec=float(getattr(self._cfg, "adapt_rtol_dec", 0.5)),
            adapt_rtol_inc=float(getattr(self._cfg, "adapt_rtol_inc", 1.25)),
            adapt_rtol_loss_mul=float(getattr(self._cfg, "adapt_rtol_loss_mul", 2.0)),
            adapt_rtol_inc_loss_thresh=float(
                getattr(self._cfg, "adapt_rtol_inc_loss_thresh", 2e-4)
            ),
            photo_switch_longdy_thresh=float(
                getattr(
                    self._cfg,
                    "photo_switch_longdy_thresh",
                    float(self._cfg.yconv_min) * 10.0,
                )
            ),
            photo_switch_longdydt_thresh=float(
                getattr(self._cfg, "photo_switch_longdydt_thresh", 1e-6)
            ),
            hycean_pin_time=float(getattr(self._cfg, "hycean_pin_time", 1e6)),
            step_size_safety=float(getattr(self._cfg, "step_size_safety", 0.9)),
            step_size_zero_delta_frac=float(
                getattr(self._cfg, "step_size_zero_delta_frac", 0.01)
            ),
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
            post_conden_rtol=float(
                getattr(self._cfg, "post_conden_rtol", self._cfg.rtol)
            ),
            fix_species_from_coldtrap_lev=bool(
                getattr(self._cfg, "fix_species_from_coldtrap_lev", True)
            ),
            fix_species_idx=jnp.asarray(fix_species_idx),
            fix_species_sat_mix=jnp.asarray(fix_species_sat_mix),
            fix_species_wholecol=jnp.asarray(fix_species_wholecol),
            save_evolution=bool(getattr(self._cfg, "save_evolution", False)),
            save_evo_frq=int(getattr(self._cfg, "save_evo_frq", 10)),
            save_evo_n_max=(
                int(
                    np.ceil(
                        int(self._cfg.count_max)
                        / max(int(getattr(self._cfg, "save_evo_frq", 10)), 1)
                    )
                )
                + 1
                if bool(getattr(self._cfg, "save_evolution", False))
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

        # condense_zero_mask (nz, ni) -- True where delta is zeroed; all False
        # unless use_condense (condense_sp + non_gas_sp).
        cond_mask_np = np.zeros((nz, ni), dtype=bool)
        if self._cfg.use_condense:
            for sp in self._cfg.condense_sp + self._cfg.non_gas_sp:
                if sp in _NETWORK.species_idx:
                    cond_mask_np[:, _NETWORK.species_idx[sp]] = True

        self._statics = self._build_statics(var, atm)
        self._photo_static = self._build_photo_static(var, atm)
        self._refresh_static = self._build_refresh_static(atm)
        self._conden_static = self._build_conden_static(var, atm, gas_mask_jnp)
        self._runner, self._runner_batch = _make_runner(
            _NET_JAX,
            self._statics,
            self._non_gas_present,
            gas_mask_jnp,
            self._zero_bot_row,
            jnp.asarray(cond_mask_np),
            self._hydro_partial,
            float(getattr(self._cfg, "start_conden_time", 0.0)),
            float(getattr(self._cfg, "stop_conden_time", 100000.0)),
            photo_static=self._photo_static,
            refresh_static=self._refresh_static,
            conden_static=self._conden_static,
        )

    def _build_photo_static(self, var, atm) -> Optional[_PhotoStatic]:
        """Pack photo cross sections + scalar configs into a `_PhotoStatic`.

        Returns None if `use_photo=False` (the runner skips the photo branch
        entirely in that case). Reuses the photo data caches from the
        odesolver when available (populated by the pre-loop
        `op_jax.Ros2JAX.compute_tau` call in
        `state._build_pre_loop_runstate_impl`).
        """
        if not self._cfg.use_photo:
            return None

        # Derive everything from the PhotoStaticInputs pytree; lazily build it
        # via Ros2JAX's own builder for unwired test sites.
        odesolver = self.odesolver
        photo_static = getattr(odesolver, "_photo_static", None)
        if photo_static is None and hasattr(odesolver, "_ensure_photo_static"):
            photo_static = odesolver._ensure_photo_static(var, atm)

        photo_data = getattr(odesolver, "_photo_data", None)
        if photo_data is None:
            photo_data = _photo_mod.photo_data_from_static(
                photo_static, list(_NETWORK.species)
            )
            odesolver._photo_data = photo_data
        photo_J_data = getattr(odesolver, "_photo_J_data", None)
        if photo_J_data is None:
            photo_J_data = _photo_mod.photo_J_data_from_static(photo_static)
            odesolver._photo_J_data = photo_J_data
        photo_ion_data = getattr(odesolver, "_photo_ion_data", None)
        if self._cfg.use_ion:
            if photo_ion_data is None:
                photo_ion_data = _photo_mod.photo_ion_data_from_static(photo_static)
                odesolver._photo_ion_data = photo_ion_data
        else:
            photo_ion_data = None

        (branch_re_idx, branch_active, branch_T_re_idx, branch_T_active) = (
            _photo_mod.pack_J_to_k_index_map(photo_J_data, var, self._cfg)
        )
        if photo_ion_data is not None:
            ion_branch_re_idx, ion_branch_active = _photo_mod.pack_Jion_to_k_index_map(
                photo_ion_data, var, self._cfg
            )
            cross_Jion = photo_ion_data.cross_J
        else:
            ion_branch_re_idx = jnp.zeros((0,), dtype=jnp.int64)
            ion_branch_active = jnp.zeros((0,), dtype=jnp.bool_)
            cross_Jion = jnp.zeros((0, int(photo_static.nbin)), dtype=jnp.float64)

        bins_arr = jnp.asarray(photo_static.bins, dtype=jnp.float64)
        din12_indx = int(photo_static.din12_indx)
        dbin1 = float(photo_static.dbin1)
        dbin2 = float(photo_static.dbin2)

        ag0 = float(_phy_const.ag0)
        # Record the baked star's TOA flux here (not only in prepare_runstate)
        # so a pre-built runner still rejects a later different-star batch.
        if self._sflux_top_ref is None:
            self._sflux_top_ref = np.asarray(var.sflux_top, dtype=np.float64)
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
            bins=bins_arr,
            sflux_top=jnp.asarray(var.sflux_top, dtype=jnp.float64),
            dz=jnp.asarray(atm.dz, dtype=jnp.float64),
            din12_indx=din12_indx,
            dbin1=dbin1,
            dbin2=dbin2,
            mu_zenith=float(np.cos(self._cfg.sl_angle)),
            edd=float(self._cfg.edd),
            ag0=ag0,
            hc=float(_phy_const.hc),
            f_diurnal=float(self._cfg.f_diurnal),
            flux_atol=float(self._cfg.flux_atol),
            ag0_is_zero=(ag0 == 0.0),
        )

    def _build_refresh_static(self, atm) -> _atm_refresh_mod.AtmRefreshStatic:
        """Pack the static inputs to `atm_refresh.update_mu_dz_jax`.

        Captures the T-P profile, planetary constants, species masses, and
        the reference layer / boundary z-value once at OuterLoop init.
        These never change during integration. Reads `atm` only: the refresh
        statics are structural, so a `var` parameter would suggest a
        dependence on the evolving state that does not exist.
        """
        from . import composition as _ba

        species = _ba.species
        ni = _NETWORK.ni
        nz = atm.Tco.shape[0]
        mol_mass_arr = np.array(
            [_ba.compo["mass"][_ba.compo_row.index(sp)] for sp in species],
            dtype=np.float64,
        )
        diff_esc_idx = np.array(
            [species.index(sp) for sp in self._cfg.diff_esc],
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
            Rp=float(self._cfg.Rp),
            kb=float(_phy_const.kb),
            Navo=float(_phy_const.Navo),
            max_flux=float(self._cfg.max_flux),
            nz=int(nz),
            ni=int(ni),
        )

    def _build_conden_static(
        self, var, atm, gas_mask_jnp
    ) -> Optional[_conden_mod.CondenStatic]:
        """Pack the conden tables into a `CondenStatic`, or None when
        `use_condense=False` (the runner then omits the conden branch).

        Reactions whose species is in `use_relax` get `coeff_per_re = 0`
        (matches the `var.k[re] = 0` short-circuits at op.py:1121-1123,
        1153-1155); the H2O/NH3 relax blocks are degenerate unless the
        species is in `use_relax` (the `*_active` bools short-circuit at
        trace time).
        """
        if not self._cfg.use_condense:
            return None

        # Single-source: static metadata from make_conden_spec; every
        # T/structure-dependent array from the same build_conden_profile the
        # on-graph rebuild uses.
        spec = _conden_mod.make_conden_spec(self._cfg, var, atm, _NETWORK.species_idx)
        prof = _conden_mod.build_conden_profile(
            spec,
            jnp.asarray(atm.Tco, dtype=jnp.float64),
            jnp.asarray(atm.pco, dtype=jnp.float64),
            jnp.asarray(atm.n_0, dtype=jnp.float64),
            jnp.asarray(atm.Dzz, dtype=jnp.float64),
        )
        return _conden_mod.CondenStatic(
            conden_re_idx=jnp.asarray(np.asarray(spec.conden_re_idx, dtype=np.int32)),
            conden_sp_idx=jnp.asarray(np.asarray(spec.conden_sp_idx, dtype=np.int32)),
            Dg_per_re=prof.Dg_per_re,
            sat_n_per_re=prof.sat_n_per_re,
            coeff_per_re=jnp.asarray(np.asarray(spec.coeff_per_re, dtype=np.float64)),
            h2o_active=spec.h2o_active,
            h2o_idx=spec.h2o_idx,
            h2o_l_s_idx=spec.h2o_l_s_idx,
            h2o_Dg=prof.h2o_Dg,
            h2o_sat=prof.h2o_sat,
            h2o_m_over_rho_r2=spec.h2o_m_over_rho_r2,
            nh3_active=spec.nh3_active,
            nh3_idx=spec.nh3_idx,
            nh3_l_s_idx=spec.nh3_l_s_idx,
            nh3_Dg=prof.nh3_Dg,
            nh3_sat=prof.nh3_sat,
            nh3_m_over_rho_r2=spec.nh3_m_over_rho_r2,
            nh3_conden_top=int(prof.nh3_conden_top),
            n_0=jnp.asarray(atm.n_0, dtype=jnp.float64),
            gas_indx_mask=gas_mask_jnp,
        )

    def _initial_photo_carry_from_runstate(self, rs) -> dict:
        """Build the initial photo carry from a RunState slice.

        `rs.rate.k` carries the dense reaction-rate table; `rs.photo_runtime.*`
        carries `tau / aflux / sflux / dflux_*`; nbin is derived from
        `tau.shape[1]`.
        """
        nz = int(rs.atm.Tco.shape[0])
        nr = _NETWORK.nr
        k_arr = jnp.asarray(rs.rate.k, dtype=jnp.float64)
        if k_arr.shape != (nr + 1, nz):
            raise ValueError(
                f"rs.rate.k shape {k_arr.shape} != expected ({nr + 1}, {nz})"
            )
        if self._photo_static is None:
            tiny = jnp.zeros((1, 1), dtype=jnp.float64)
            return dict(
                k_arr=k_arr,
                tau=tiny,
                aflux=tiny,
                sflux=tiny,
                dflux_d=tiny,
                dflux_u=tiny,
                prev_aflux=tiny,
                aflux_change=jnp.float64(0.0),
                J_br=jnp.zeros((0, nz), dtype=jnp.float64),
                J_br_T=jnp.zeros((0, nz), dtype=jnp.float64),
                Jion_br=jnp.zeros((0, nz), dtype=jnp.float64),
            )
        pr = rs.photo_runtime
        n_br = int(self._photo_static.cross_J.shape[0])
        n_br_T = int(self._photo_static.cross_J_T.shape[0])
        n_ion_br = int(self._photo_static.cross_Jion.shape[0])
        return dict(
            k_arr=k_arr,
            tau=jnp.asarray(pr.tau, dtype=jnp.float64),
            aflux=jnp.asarray(pr.aflux, dtype=jnp.float64),
            sflux=jnp.asarray(pr.sflux, dtype=jnp.float64),
            dflux_d=jnp.asarray(pr.dflux_d, dtype=jnp.float64),
            dflux_u=jnp.asarray(pr.dflux_u, dtype=jnp.float64),
            prev_aflux=jnp.asarray(pr.prev_aflux, dtype=jnp.float64),
            aflux_change=jnp.float64(float(pr.aflux_change)),
            J_br=jnp.zeros((n_br, nz), dtype=jnp.float64),
            J_br_T=jnp.zeros((n_br_T, nz), dtype=jnp.float64),
            Jion_br=jnp.zeros((n_ion_br, nz), dtype=jnp.float64),
        )

    def _initial_atm_carry_from_runstate(self, rs) -> dict:
        """Build the initial atm-refresh carry from a RunState slice."""
        return dict(
            g=jnp.asarray(rs.atm.g, dtype=jnp.float64),
            mu=jnp.asarray(rs.atm.mu, dtype=jnp.float64),
            Hp=jnp.asarray(rs.atm.Hp, dtype=jnp.float64),
            dz=jnp.asarray(rs.atm.dz, dtype=jnp.float64),
            zco=jnp.asarray(rs.atm.zco, dtype=jnp.float64),
            dzi=jnp.asarray(rs.atm.dzi, dtype=jnp.float64),
            Hpi=jnp.asarray(rs.atm.Hpi, dtype=jnp.float64),
            top_flux=jnp.asarray(rs.atm.top_flux, dtype=jnp.float64),
            vs=jnp.asarray(rs.atm.vs, dtype=jnp.float64),
        )

    def _initial_conv_carry_from_runstate(self, rs) -> dict:
        """Build the initial conv-history carry from a RunState slice.

        Pulls `longdy / longdydt` from `rs.step`.
        """
        nz = int(rs.atm.Tco.shape[0])
        ni = _NETWORK.ni
        conv_step = int(self._cfg.conv_step)
        ini_frq = int(getattr(self._cfg, "ini_update_photo_frq", 100))
        return dict(
            y_time_ring=jnp.zeros((conv_step, nz, ni), dtype=jnp.float64),
            t_time_ring=jnp.zeros((conv_step,), dtype=jnp.float64),
            longdy=jnp.float64(float(rs.step.longdy)),
            longdydt=jnp.float64(float(rs.step.longdydt)),
            where_varies_most=jnp.asarray(
                rs.params.where_varies_most,
                dtype=jnp.float64,
            ),
            longdy_seen_min=jnp.float64(jnp.inf),
            count_since_new_min=jnp.int32(0),
            rtol=jnp.float64(float(self._cfg.rtol)),
            loss_criteria=jnp.float64(float(getattr(self, "loss_criteria", 0.0005))),
            update_photo_frq=jnp.int32(ini_frq),
            is_final_photo_frq=jnp.bool_(False),
        )

    def _profile_vars_from_runstate(self, rs) -> ProfileVars:
        """Snapshot this profile's per-profile constants into a `ProfileVars`.

        Rebuilds the per-run static bundles for THIS runstate (so the values
        match what the single-profile closure would bake) and copies out the
        per-profile arrays/scalars that the batched runner reads from the carry
        instead of the closure. Host-side NumPy; cheap. Conden arrays fall back
        to consistently-shaped placeholders when condensation is off (the body
        never reads them in that case).
        """
        var, atm, _ = _state_mod.legacy_view(rs, cfg=self._cfg)
        statics = self._build_statics(var, atm)
        refresh = self._build_refresh_static(atm)
        ni = _NETWORK.ni
        nz = int(atm.Tco.shape[0])
        gas_mask_np = np.zeros(ni, dtype=bool)
        if self._non_gas_present and hasattr(atm, "gas_indx"):
            gas_mask_np[np.asarray(atm.gas_indx, dtype=int)] = True
        else:
            gas_mask_np[:] = True
        conden = self._build_conden_static(var, atm, jnp.asarray(gas_mask_np))
        if conden is not None:
            c_Dg = jnp.asarray(conden.Dg_per_re, dtype=jnp.float64)
            c_sat_n = jnp.asarray(conden.sat_n_per_re, dtype=jnp.float64)
            c_h2o_Dg = jnp.asarray(conden.h2o_Dg, dtype=jnp.float64)
            c_h2o_sat = jnp.asarray(conden.h2o_sat, dtype=jnp.float64)
            c_nh3_Dg = jnp.asarray(conden.nh3_Dg, dtype=jnp.float64)
            c_nh3_sat = jnp.asarray(conden.nh3_sat, dtype=jnp.float64)
            # Pin int32 explicitly: under x64, asarray(int) would default to
            # int64 and lanes packed at different times could mismatch dtypes.
            c_nh3_top = jnp.asarray(int(conden.nh3_conden_top), dtype=jnp.int32)
        else:
            zz = jnp.zeros((nz,), dtype=jnp.float64)
            c_Dg = jnp.zeros((1, nz), dtype=jnp.float64)
            c_sat_n = jnp.zeros((1, nz), dtype=jnp.float64)
            c_h2o_Dg = zz
            c_h2o_sat = zz
            c_nh3_Dg = zz
            c_nh3_sat = zz
            c_nh3_top = jnp.int32(0)
        if self._cfg.use_photo and rs.photo_static is not None:
            # The two T-P-dependent photo statics, exactly what
            # _build_photo_static would bake for this profile.
            p_absp_T_cross = jnp.asarray(
                rs.photo_static.absp_T_cross, dtype=jnp.float64
            )
            p_cross_J_T = jnp.asarray(rs.photo_static.cross_J_T, dtype=jnp.float64)
        elif self._cfg.use_photo and self._photo_static is not None:
            # Legacy entry (no photo_static slot) is single-profile only, so
            # the closure-baked arrays ARE this profile's; seed pv with them.
            p_absp_T_cross = jnp.asarray(
                self._photo_static.photo_data.absp_T_cross, dtype=jnp.float64
            )
            p_cross_J_T = jnp.asarray(self._photo_static.cross_J_T, dtype=jnp.float64)
        else:
            p_absp_T_cross = jnp.zeros((0, 1, 1), dtype=jnp.float64)
            p_cross_J_T = jnp.zeros((0, 1, 1), dtype=jnp.float64)
        return ProfileVars(
            n_0=jnp.asarray(statics.n_0, dtype=jnp.float64),
            Kzz=jnp.asarray(statics.Kzz, dtype=jnp.float64),
            atom_ini=jnp.asarray(statics.atom_ini_arr, dtype=jnp.float64),
            bottom_n=jnp.asarray(statics.bottom_n, dtype=jnp.float64),
            fix_species_sat_mix=jnp.asarray(
                statics.fix_species_sat_mix, dtype=jnp.float64
            ),
            r_Tco=jnp.asarray(refresh.Tco, dtype=jnp.float64),
            r_pico=jnp.asarray(refresh.pico, dtype=jnp.float64),
            r_Dzz_top=jnp.asarray(refresh.Dzz_top, dtype=jnp.float64),
            r_gs=jnp.float64(refresh.gs),
            r_zco_pref=jnp.float64(refresh.zco_pref),
            r_Rp=jnp.float64(refresh.Rp),
            c_Dg_per_re=c_Dg,
            c_sat_n_per_re=c_sat_n,
            c_h2o_Dg=c_h2o_Dg,
            c_h2o_sat=c_h2o_sat,
            c_nh3_Dg=c_nh3_Dg,
            c_nh3_sat=c_nh3_sat,
            c_nh3_conden_top=c_nh3_top,
            p_absp_T_cross=p_absp_T_cross,
            p_cross_J_T=p_cross_J_T,
        )

    def _pack_state_from_runstate(self, rs) -> JaxIntegState:
        """Build the initial JaxIntegState from a fully-populated RunState.

        The runner reads its entry state from a typed `RunState` rather
        than the legacy `(var, para, atm)` triple. Static metadata (atom
        ordering, fix-species mapping) lives on the `OuterLoop` instance.
        """
        photo_fields = self._initial_photo_carry_from_runstate(rs)
        atm_fields = self._initial_atm_carry_from_runstate(rs)
        conv_fields = self._initial_conv_carry_from_runstate(rs)
        nz = int(rs.atm.Tco.shape[0])
        ni = _NETWORK.ni
        n_fix = int(self._statics.fix_species_idx.shape[0])
        fix_started = bool(rs.params.fix_species_start)
        fix_y_init = np.zeros((nz, ni), dtype=np.float64)
        fix_mask_init = np.zeros((nz, ni), dtype=bool)
        fix_pfix_idx_init = np.zeros((n_fix,), dtype=np.int32)
        if (
            self._statics.use_fix_species
            and rs.fix_species is not None
            and len(rs.fix_species.fix_species) > 0
        ):
            fix_y_arr = np.asarray(rs.fix_species.fix_y, dtype=np.float64)
            coldtrap = np.asarray(rs.fix_species.conden_min_lev, dtype=np.int32)
            for i, sp in enumerate(rs.fix_species.fix_species):
                sp_idx = _NETWORK.species_idx[sp]
                fix_y_init[:, sp_idx] = fix_y_arr[i]
                if fix_started:
                    if self._statics.fix_species_from_coldtrap_lev:
                        pfix = int(coldtrap[i])
                        fix_mask_init[:pfix, sp_idx] = True
                        fix_pfix_idx_init[i] = pfix
                    else:
                        fix_mask_init[:, sp_idx] = True
                        fix_pfix_idx_init[i] = nz
        return JaxIntegState(
            y=jnp.asarray(rs.step.y, dtype=jnp.float64),
            y_prev=jnp.asarray(rs.step.y_prev, dtype=jnp.float64),
            ymix=jnp.asarray(rs.step.ymix, dtype=jnp.float64),
            dt=jnp.asarray(float(rs.step.dt), dtype=jnp.float64),
            t=jnp.asarray(float(rs.step.t), dtype=jnp.float64),
            delta=jnp.asarray(float(rs.params.delta), dtype=jnp.float64),
            accept_count=jnp.int32(int(rs.params.count)),
            retry_count=jnp.int32(0),
            atom_loss=jnp.asarray(rs.atoms.atom_loss, dtype=jnp.float64),
            atom_loss_prev=jnp.asarray(rs.atoms.atom_loss_prev, dtype=jnp.float64),
            nega_count=jnp.int32(int(rs.params.nega_count)),
            loss_count=jnp.int32(int(rs.params.loss_count)),
            delta_count=jnp.int32(int(rs.params.delta_count)),
            small_y=jnp.float64(float(rs.params.small_y)),
            nega_y=jnp.float64(float(rs.params.nega_y)),
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
            # save_evolution buffers. Allocated to the cfg's
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
            # chunk_target sentinel (2**30 >> any count_max) disables the
            # chunk cap for single-shot runs; the chunked driver overwrites it.
            chunk_target=jnp.int32(2**30),
            # Batched-runner flags; the single-profile path never reads them.
            is_done=jnp.bool_(False),
            termination_reason=jnp.int32(0),
            # Phase seed: upwind (1.0) when use_vm_mol, else central (0.0);
            # only hybrid runs ever flip it.
            hybrid_use_vm=jnp.float64(1.0 if bool(self._statics.use_vm_mol) else 0.0),
            # Live termination budget, seeded to the static caps. Only the
            # hybrid phase flip mutates these (extends phase 1's allowance).
            count_min_dyn=jnp.int32(int(self._statics.count_min)),
            count_max_dyn=jnp.int32(int(self._statics.count_max)),
            runtime_dyn=jnp.float64(float(self._statics.runtime)),
            # Per-profile constants MUST ride the carry (vmap does not batch
            # closures); value-identical to the closures single-profile.
            pv=self._profile_vars_from_runstate(rs),
        )

    def _pack_state(self, var, para, atm) -> JaxIntegState:
        """Legacy entry point: build JaxIntegState from `(var, para, atm)`.

        Thin wrapper around `runstate_from_store` +
        `_pack_state_from_runstate`. Every read flows through the typed
        `RunState` slice.
        """
        rs = _state_mod.runstate_from_store(var, atm, para)
        return self._pack_state_from_runstate(rs)

    def _unpack_state_to_runstate(self, state: JaxIntegState, rs_entry):
        """Build a fresh `RunState` from the runner's final JaxIntegState.

        `_unpack_state` flows through this constructor + `runstate_to_store`.
        The static atm fields (pco, Tco, Kzz, n_0, ms, alpha, ...) are
        preserved verbatim from `rs_entry.atm`; only the dynamic refresh
        slots (g, mu, Hp, dz, dzi, zco, zmco, Hpi, top_flux, vs, and vm
        under use_vm_mol) come from the carry. Step / params / atoms /
        photo_runtime / fix_species are rebuilt from the carry against
        the entry-time ordering captured in `rs_entry`.
        """
        g = jnp.asarray(state.g, dtype=jnp.float64)
        zco = jnp.asarray(state.zco, dtype=jnp.float64)
        dzi = jnp.asarray(state.dzi, dtype=jnp.float64)
        Hpi = jnp.asarray(state.Hpi, dtype=jnp.float64)
        atm_out = rs_entry.atm._replace(
            g=g,
            mu=jnp.asarray(state.mu, dtype=jnp.float64),
            Hp=jnp.asarray(state.Hp, dtype=jnp.float64),
            dz=jnp.asarray(state.dz, dtype=jnp.float64),
            zco=zco,
            # op.py:972-973: cell-centre heights follow the refreshed zco.
            zmco=0.5 * (zco[:-1] + zco[1:]),
            dzi=dzi,
            Hpi=Hpi,
            top_flux=jnp.asarray(state.top_flux, dtype=jnp.float64),
            vs=jnp.asarray(state.vs, dtype=jnp.float64),
        )
        if self._statics.use_vm_mol:
            # vm_branch op.py:945-992 refreshes vm inside update_mu_dz. The
            # runner recomputes it per step from the carry and never stores
            # it, so rebuild the terminal value here (same inputs as in-loop).
            atm_out = atm_out._replace(
                vm=_atm_refresh_mod.recompute_vm_jax(
                    g,
                    Hpi,
                    dzi,
                    jnp.asarray(rs_entry.atm.Dzz, dtype=jnp.float64),
                    jnp.asarray(rs_entry.atm.ms, dtype=jnp.float64),
                    jnp.asarray(rs_entry.atm.alpha, dtype=jnp.float64),
                    jnp.asarray(rs_entry.atm.Tco, dtype=jnp.float64),
                    float(_phy_const.kb),
                    float(_phy_const.Navo),
                )
            )

        rate_out = rs_entry.rate._replace(
            k=jnp.asarray(state.k_arr, dtype=jnp.float64),
        )

        photo_out = rs_entry.photo  # not mutated by the runner

        # Slice the populated prefix of the save_evolution ring; when
        # save_evolution=False, evo_idx stays 0 and the slice is empty.
        evo_n = int(state.evo_idx)
        y_evo_out = jnp.asarray(state.y_evo[:evo_n], dtype=jnp.float64)
        t_evo_out = jnp.asarray(state.t_evo[:evo_n], dtype=jnp.float64)

        step_out = _state_mod.StepInputs(
            y=jnp.asarray(state.y, dtype=jnp.float64),
            y_prev=jnp.asarray(state.y_prev, dtype=jnp.float64),
            ymix=jnp.asarray(state.ymix, dtype=jnp.float64),
            t=float(state.t),
            dt=float(state.dt),
            longdy=float(state.longdy),
            longdydt=float(state.longdydt),
            y_evo=y_evo_out,
            t_evo=t_evo_out,
        )

        params_out = _state_mod.ParamInputs(
            count=int(state.accept_count),
            nega_count=int(state.nega_count),
            loss_count=int(state.loss_count),
            delta_count=int(state.delta_count),
            delta=float(state.delta),
            small_y=float(state.small_y),
            nega_y=float(state.nega_y),
            end_case=self._classify_end_case(state),
            switch_final_photo_frq=bool(state.is_final_photo_frq),
            pic_count=int(getattr(rs_entry.params, "pic_count", 0)),
            where_varies_most=jnp.asarray(
                state.where_varies_most,
                dtype=jnp.float64,
            ),
            fix_species_start=bool(state.fix_species_started),
            termination_reason=int(state.termination_reason),
        )

        atom_loss_arr = np.asarray(state.atom_loss, dtype=np.float64)
        atom_sum_arr = (atom_loss_arr + 1.0) * np.asarray(
            self._statics.atom_ini_arr, dtype=np.float64
        )
        atoms_out = _state_mod.AtomInputs(
            atom_order=tuple(self._atom_order),
            atom_ini=jnp.asarray(self._statics.atom_ini_arr, dtype=jnp.float64),
            atom_loss=jnp.asarray(atom_loss_arr),
            atom_loss_prev=jnp.asarray(state.atom_loss_prev, dtype=jnp.float64),
            atom_sum=jnp.asarray(atom_sum_arr),
        )

        if self._photo_static is not None:
            photo_runtime_out = _state_mod.PhotoRuntimeInputs(
                tau=jnp.asarray(state.tau, dtype=jnp.float64),
                aflux=jnp.asarray(state.aflux, dtype=jnp.float64),
                sflux=jnp.asarray(state.sflux, dtype=jnp.float64),
                dflux_d=jnp.asarray(state.dflux_d, dtype=jnp.float64),
                dflux_u=jnp.asarray(state.dflux_u, dtype=jnp.float64),
                prev_aflux=jnp.asarray(state.prev_aflux, dtype=jnp.float64),
                aflux_change=float(state.aflux_change),
            )
        else:
            photo_runtime_out = None

        nz = int(rs_entry.atm.Tco.shape[0])
        fix_species_cfg = list(getattr(self._cfg, "fix_species", []) or [])
        if fix_species_cfg:
            fix_y_full = np.asarray(state.fix_y, dtype=np.float64)
            fix_y_per_sp = np.zeros((len(fix_species_cfg), nz), dtype=np.float64)
            for i, sp in enumerate(fix_species_cfg):
                fix_y_per_sp[i] = fix_y_full[:, _NETWORK.species_idx[sp]]
            if self._statics.fix_species_from_coldtrap_lev:
                pfix_np = np.asarray(state.fix_pfix_idx, dtype=np.int32)
            else:
                pfix_np = np.zeros((len(fix_species_cfg),), dtype=np.int32)
            fix_species_out = _state_mod.FixSpeciesInputs(
                fix_species=tuple(fix_species_cfg),
                fix_y=jnp.asarray(fix_y_per_sp),
                conden_min_lev=jnp.asarray(pfix_np),
            )
        else:
            fix_species_out = _state_mod.FixSpeciesInputs(
                fix_species=(),
                fix_y=jnp.zeros((0, nz), dtype=jnp.float64),
                conden_min_lev=jnp.zeros((0,), dtype=jnp.int32),
            )

        return _state_mod.RunState(
            atm=atm_out,
            rate=rate_out,
            photo=photo_out,
            step=step_out,
            params=params_out,
            atoms=atoms_out,
            photo_runtime=photo_runtime_out,
            fix_species=fix_species_out,
            metadata=rs_entry.metadata,
            photo_static=rs_entry.photo_static,
        )

    def _unpack_state(self, state: JaxIntegState, var, para, atm) -> None:
        """Write the post-runner JAX state back into the var/para/atm store
        objects. Routes through a synthesized `RunState` and
        `runstate_to_store`; the var/cfg side effects that don't fit the
        typed pytree (var.J_sp dict, var.y_time/t_time list, conden
        k_arr, cfg.use_fix_sp_bot/rtol mutation) follow.
        """
        rs_entry = _state_mod.runstate_from_store(var, atm, para)
        rs_out = self._unpack_state_to_runstate(state, rs_entry)
        _state_mod.runstate_to_store(rs_out, var, atm, para)

        # Hycean pin diagnostic (op.py:2935-2941): mirror master's cfg
        # mutation so post-run readers of use_fix_sp_bot see the pinned values.
        if self._statics.use_fix_H2He and bool(state.h2he_pinned):
            h2he_mix_arr = np.asarray(state.h2he_mix, dtype=np.float64)
            existing = dict(getattr(self._cfg, "use_fix_sp_bot", {}) or {})
            existing.setdefault("H2", float(h2he_mix_arr[0]))
            existing.setdefault("He", float(h2he_mix_arr[1]))
            self._cfg.use_fix_sp_bot = existing

        # rtol may have moved adaptively inside the runner; reflect it in
        # the global cfg for parity with op.Integration.__call__.
        if self._statics.use_adapt_rtol:
            self._cfg.rtol = float(state.rtol)

        # Rebuild var.y_time / var.t_time chronologically from the ring
        # buffer (most recent min(accept_count, conv_step) entries).
        self._unpack_ring(state, var)

        # save_evolution overrides the ring with the captured buffer prefix.
        if self._statics.save_evolution:
            n_evo = int(state.evo_idx)
            y_evo_arr = np.asarray(state.y_evo, dtype=np.float64)[:n_evo]
            t_evo_arr = np.asarray(state.t_evo, dtype=np.float64)[:n_evo]
            var.y_time = y_evo_arr
            var.t_time = t_evo_arr

        # Photo dict-view synthesis: J_sp is rebuilt here because it lives
        # outside the typed slice; the array fields were already written by
        # runstate_to_store.
        if self._photo_static is not None:
            self._unpack_J_sp(state, var)
            self._unpack_k(state, var)

        # Conden k unpack: same full-array overwrite as photo.
        if self._conden_static is not None:
            self._unpack_k(state, var)

    def _unpack_J_sp(self, state: JaxIntegState, var) -> None:
        """Rebuild `var.J_sp` dict from carry's J_br / J_br_T arrays.

        Mirrors the dict population in `op.compute_J` (op.py:2764, 2783):
        per (sp, nbr) entries for nbr>=1, plus a per-species (sp, 0) total.
        Needed by `var.var_save` for the .vul output and by any downstream
        plot scripts.
        """
        nz = state.aflux.shape[0]
        n_branch = var.n_branch
        var.J_sp = {
            (sp, bn): np.zeros(nz)
            for sp in var.photo_sp
            for bn in range(n_branch[sp] + 1)
        }
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
        """Snapshot the full photo-updated `state.k_arr` into `var.k_arr`
        (idempotent for rows the runner didn't touch). The legacy
        `{i: array(nz)}` dict view is synthesized at `.vul` write time by
        `legacy_io.Output.save_out`.
        """
        var.k_arr = np.asarray(state.k_arr, dtype=np.float64)

    def _unpack_ring(self, state: JaxIntegState, var) -> None:
        """Rebuild `var.y_time` / `var.t_time` chronologically from the ring.

        The ring slot for the n-th accepted step (0-indexed) is
        `n % conv_step`. After the runner returns, the chronological
        ordering of the most recent `min(accept_count, conv_step)` entries
        is `slots[(accept_count - L + i) % conv_step for i in 0..L-1]`,
        where L = min(accept_count, conv_step).

        Trade-off: var.y_time holds only the LAST conv_step entries (full
        history would need an io_callback per step); increase conv_step or
        use save_evolution for more.
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
        # Only the FINAL atom_loss is in the carry; pad it over L entries so
        # plot scripts indexing the list don't error.
        final_atom_loss = list(np.asarray(state.atom_loss).tolist())
        var.atom_loss_time = [final_atom_loss for _ in range(L)]

    def _classify_end_case(self, state: JaxIntegState, wall_clock_hit=False):
        """Classify end-of-run (op.py:1069-1085) from the in-loop reason.

        `_real_terminate` applies master's priority (converged over runtime
        over step-count) against the live budget, so a step that converges
        while hitting a cap reads end_case=1, and so does a hybrid phase-1
        convergence past the static count_max. Wall-clock exit (end_case=4)
        is sticky -- the JIT'd loop has not actually terminated, only the
        host bailed out.

        end_case=5 is a VULCAN-JAX addition with no upstream counterpart: the
        run stopped without meeting the convergence criterion and without
        hitting either cap -- a lane frozen on non-finite state
        (termination_reason 5), one that only yielded at a chunk boundary
        (reason 0), or a "converged" state that is not finite.
        """
        if wall_clock_hit:
            return 4
        reason = int(state.termination_reason)
        if reason in (2, 3):
            return reason
        if reason in (1, 4) and bool(jnp.all(jnp.isfinite(state.y))):
            return 1  # the JAX-only stall fallback (4) shares end_case=1
        return 5

    def _run_chunked(self, init_state, atm_static, var, para, atm):
        """Run the integration in chunks so the host can fire `print_prog`
        and live-UI hooks between chunks.

        Chunk size: `live_plot_frq` when any live flag is on (master's
        cadence), else `print_prog_num`. Termination semantics are meant to
        equal the single-shot path: the chunk cap is the only extra exit and
        it is clamped to `count_max_dyn + 1`, so it cannot fire before a real
        termination would. NOT COVERED BY A TEST — nothing in `tests/`
        exercises `use_chunked_runner` / `_run_chunked`, so chunked-vs-single
        bit-equivalence is an argument, not a measured result.

        Returns ``(state, wall_clock_hit)``; wall_clock_hit=True means the
        host wall-clock budget expired between chunks (end_case=4).
        """
        from .live_ui import any_live_flag_on, LiveUI

        live_on = any_live_flag_on(self._cfg)
        if live_on:
            chunk_size = max(int(getattr(self._cfg, "live_plot_frq", 10)), 1)
            if self._live_ui is None:
                self._live_ui = LiveUI(self._cfg)
        else:
            chunk_size = max(int(getattr(self._cfg, "print_prog_num", 500)), 1)
        use_print_prog = bool(getattr(self._cfg, "use_print_prog", True))
        wall_clock_max = getattr(self._cfg, "wall_clock_max", None)
        wall_clock_max = (
            float(wall_clock_max)
            if wall_clock_max is not None and float(wall_clock_max) > 0
            else None
        )
        start_time = (
            float(getattr(para, "start_time", _now())) if para is not None else _now()
        )

        state = init_state
        while True:
            target = int(state.accept_count) + chunk_size
            # Cap the chunk at count_max + 1 so chunk_done never fires before
            # count_max would. Read the LIVE budget off the carry: the hybrid
            # flip extends it in-loop and static caps would truncate phase 1.
            target = min(target, int(state.count_max_dyn) + 1)
            state = state._replace(chunk_target=jnp.int32(target))
            state = self._runner(state, atm_static)

            count_now = int(state.accept_count)
            t_now = float(state.t)

            chunk_cap_hit = count_now >= target
            count_max_hit = count_now > int(state.count_max_dyn)
            runtime_hit = t_now > float(state.runtime_dyn)
            terminated_for_real = count_max_hit or runtime_hit or (not chunk_cap_hit)

            if terminated_for_real:
                return state, False

            # Sync state to host for the per-chunk hooks.
            self._unpack_state(state, var, para, atm)
            if use_print_prog:
                if (
                    not hasattr(para, "where_varies_most")
                    or para.where_varies_most is None
                ):
                    para.where_varies_most = np.zeros_like(var.y)
                self.output.print_prog(var, para)
            if live_on:
                self._live_ui.dispatch(var, atm, para)

            if wall_clock_max is not None and (_now() - start_time) > wall_clock_max:
                print(f"After ------- {_now() - start_time} seconds ------- s CPU time")
                print(
                    "Integration not completed...\n"
                    f"Wall-clock budget exceeded ({wall_clock_max} sec)!"
                )
                return state, True

    def __call__(self, *args):
        """Run the integration to convergence / runtime / count cap.

        Polymorphic: accepts a typed `state.RunState` (canonical; returns a
        fresh RunState) or the legacy `(var, atm, para, make_atm)` tuple
        (kept for hybrid oracle tests). Identical numerics either way.
        """
        if args and isinstance(args[0], _state_mod.RunState):
            rs = args[0]
            var = args[1] if len(args) > 1 else None
            atm = args[2] if len(args) > 2 else None
            para = args[3] if len(args) > 3 else None
            return self._call_runstate(rs, var, atm, para)
        var, atm, para, make_atm = args[:4]
        return self._call_legacy(var, atm, para, make_atm)

    def _call_legacy(self, var, atm, para, make_atm):
        """Legacy entry point: integrate while mutating `(var, atm, para)`.

        Everything happens inside the JIT'd runner; this method handles
        setup, the device call(s), and post-run unpacking + diagnostics.
        Runs chunked when `use_chunked_runner` or any live flag is on, else
        single-shot; both paths are bit-equivalent on the final state.
        """
        del make_atm  # captured into _refresh_static at OuterLoop init
        validate_runtime_config(self._cfg)
        self.loss_criteria = float(getattr(self._cfg, "loss_criteria", 0.0005))

        # Build the JAX runner on first entry — cached for the run.
        self._ensure_runner(var, atm)
        ni = _NETWORK.ni
        nz = atm.Tco.shape[0]

        atm_static = make_atm_static(atm, ni, nz, cfg=self._cfg)
        init_state = self._pack_state(var, para, atm)

        # Chunked when use_chunked_runner, any live flag, or wall_clock_max
        # is set (host hooks fire between chunks); default is single-shot.
        from .live_ui import any_live_flag_on

        wall_clock_max = getattr(self._cfg, "wall_clock_max", None)
        use_chunked = (
            bool(getattr(self._cfg, "use_chunked_runner", False))
            or any_live_flag_on(self._cfg)
            or (wall_clock_max is not None and float(wall_clock_max) > 0)
        )

        wall_clock_hit = False
        if use_chunked:
            final_state, wall_clock_hit = self._run_chunked(
                init_state, atm_static, var, para, atm
            )
        else:
            final_state = self._runner(init_state, atm_static)
        self._unpack_state(final_state, var, para, atm)

        # (op.Integration.f_dy is deliberately not ported: nothing reads its
        # var.dy / var.dydt -- the final print uses the DIFFERENT carry values
        # var.longdy / var.longdydt, dy/dydt are absent from upstream's
        # var_save so they never reach the .vul file, and upstream's own
        # consumer, `var.dydt_time.append(var.dydt)`, is commented out at
        # op.py:1102. The container attributes stay at their initialized 1.0
        # for master-shape compatibility.)

        # Determine end_case (op.py:1069-1085) for the final print.
        para.end_case = self._classify_end_case(final_state, wall_clock_hit)
        para.termination_reason = int(final_state.termination_reason)
        if para.end_case == 3:
            print(
                "Integration not completed...\nMaximal allowed steps "
                f"exceeded ({self._cfg.count_max})!"
            )
        elif para.end_case == 2:
            print(
                "Integration not completed...\nMaximal allowed runtime "
                f"exceeded ({self._cfg.runtime} sec)!"
            )
        elif para.end_case == 5:
            print(
                "Integration not completed...\nStopped without converging and "
                f"without hitting a cap (termination_reason "
                f"{para.termination_reason}); the state may be non-finite."
            )
        elif para.end_case == 1:
            how = (
                "via the stall fallback (JAX-only; no VULCAN 2.0 counterpart)"
                if para.termination_reason == 4
                else "on the standard convergence criterion"
            )
            print(
                f"Integration successful {how} with {para.count} steps and "
                f"long dy, long dydt = {var.longdy}, {var.longdydt}\n"
                f"Actinic flux change: {var.aflux_change:.2E}"
            )

        if self._cfg.use_print_prog:
            # print_prog reads para.where_varies_most; set a sentinel so the
            # read doesn't crash when unset.
            if not hasattr(para, "where_varies_most") or para.where_varies_most is None:
                para.where_varies_most = np.zeros_like(var.y)
            self.output.print_prog(var, para)

        # End-of-run summary (op.stop). Master only calls print_end_msg
        # (end_case=1); we also call print_unconverged_msg for 2/3/4.
        if para.end_case == 1:
            self.output.print_end_msg(var, para)
        elif para.end_case in (2, 3, 4, 5):
            self.output.print_unconverged_msg(var, para, para.end_case)
        _print_column_atom_loss(self._cfg, var.y, var.y_ini, atm.dz)

    def _call_runstate(self, rs: "_state_mod.RunState", var=None, atm=None, para=None):
        """RunState entry point: integrate from a typed `RunState` and
        return a fresh `RunState`.

        `var` / `atm` / `para` are optional. When omitted, the static
        metadata reads (`Ti`, `gas_indx`, `pref_indx`, `gs`,
        `charge_list`, `conden_re_list`, `Rf`, `n_branch`, `ion_branch`,
        `photo_sp`, `ion_sp`, `start_time`) come from `rs.metadata`; a
        `legacy_view(rs)` shim drives the `_build_*_static` helpers and
        `make_atm_static`. The legacy positional args are accepted for
        back-compat with the `integ(rs, var, atm, para)` signature.
        """
        validate_runtime_config(self._cfg)
        self.loss_criteria = float(getattr(self._cfg, "loss_criteria", 0.0005))

        # Derive a legacy-shaped shim from the RunState for the
        # _build_*_static helpers and make_atm_static.
        if var is None or atm is None:
            var, atm, _shim_para = _state_mod.legacy_view(rs, cfg=self._cfg)
            if para is None:
                para = _shim_para

        # Wire a pre-built PhotoStaticInputs onto the solver so
        # _build_photo_static doesn't rebuild from the legacy_view shim
        # (which does not carry the var.cross* dict surface).
        if (
            rs.photo_static is not None
            and getattr(self.odesolver, "_photo_static", None) is None
        ):
            self.odesolver._photo_static = rs.photo_static

        self._ensure_runner(var, atm)
        ni = _NETWORK.ni
        nz = int(rs.atm.Tco.shape[0])

        atm_static = make_atm_static(atm, ni, nz, cfg=self._cfg)
        init_state = self._pack_state_from_runstate(rs)

        from .live_ui import any_live_flag_on

        wall_clock_max = getattr(self._cfg, "wall_clock_max", None)
        use_chunked = (
            bool(getattr(self._cfg, "use_chunked_runner", False))
            or any_live_flag_on(self._cfg)
            or (wall_clock_max is not None and float(wall_clock_max) > 0)
        )
        wall_clock_hit = False
        if use_chunked:
            # The chunked driver needs legacy (var, para, atm) for its hooks;
            # reuse the caller's para (carries start_time). The final RunState
            # is rebuilt on return regardless.
            final_state, wall_clock_hit = self._run_chunked(
                init_state, atm_static, var, para, atm
            )
        else:
            final_state = self._runner(init_state, atm_static)

        rs_out = self._unpack_state_to_runstate(final_state, rs)
        if self._live_ui is not None and rs_out.params is not None:
            rs_out = rs_out._replace(
                params=rs_out.params._replace(pic_count=int(self._live_ui.pic_count))
            )

        # End-of-run printing, same predicates as the legacy path;
        # wall-clock exit (end_case=4) is sticky (the loop never terminated).
        count = int(rs_out.params.count)
        end_case = self._classify_end_case(final_state, wall_clock_hit)
        # Persist the authoritative end_case: _unpack_state_to_runstate cannot
        # see a wall-clock bail and would mislabel a truncated run as
        # converged (end_case=1).
        reason = int(final_state.termination_reason)
        if rs_out.params is not None:
            rs_out = rs_out._replace(
                params=rs_out.params._replace(
                    end_case=end_case, termination_reason=reason
                )
            )
        if end_case == 3:
            print(
                "Integration not completed...\nMaximal allowed steps "
                f"exceeded ({self._cfg.count_max})!"
            )
        elif end_case == 2:
            print(
                "Integration not completed...\nMaximal allowed runtime "
                f"exceeded ({self._cfg.runtime} sec)!"
            )
        elif end_case == 5:
            print(
                "Integration not completed...\nStopped without converging and "
                f"without hitting a cap (termination_reason {reason}); the "
                "state may be non-finite."
            )
        elif end_case != 4:
            # Reason 4 (JAX-only stall fallback) shares end_case=1 with a real
            # convergence; say so in the message.
            how = (
                "via the stall fallback (JAX-only; no VULCAN 2.0 counterpart)"
                if reason == 4
                else "on the standard convergence criterion"
            )
            print(
                f"Integration successful {how} with {count} steps and "
                f"long dy, long dydt = {rs_out.step.longdy}, "
                f"{rs_out.step.longdydt}\n"
                f"Actinic flux change: "
                f"{(rs_out.photo_runtime.aflux_change if rs_out.photo_runtime is not None else 0.0):.2E}"
            )

        # The summary printers expect a legacy (var, para) pair; build a thin
        # shim. start_time flows from the caller's para (not in the RunState
        # schema).
        var_shim, para_shim = self._summary_shim(rs_out)
        para_shim.start_time = (
            float(getattr(para, "start_time", _now())) if para is not None else _now()
        )
        para_shim.end_case = end_case

        if self._cfg.use_print_prog:
            if (
                not hasattr(para_shim, "where_varies_most")
                or para_shim.where_varies_most is None
            ):
                para_shim.where_varies_most = np.zeros_like(np.asarray(rs_out.step.y))
            self.output.print_prog(var_shim, para_shim)

        if end_case == 1:
            self.output.print_end_msg(var_shim, para_shim)
        elif end_case in (2, 3, 4, 5):
            self.output.print_unconverged_msg(var_shim, para_shim, end_case)
        _print_column_atom_loss(
            self._cfg, rs_out.step.y, rs_out.metadata.y_ini, rs_out.atm.dz
        )

        return rs_out

    def prepare_runstate(self, rs):
        """Build `(init_state, atm_static)` for one RunState and ensure the
        runner closure is built for this rs's (nz, toggle-combo).

        The batched GPU driver calls this per profile, then stacks the
        results (`stack_integ_states` / `stack_atm_statics`) into one batch
        for `run_batch`. All profiles in a single `run_batch` call must share
        the same nz / toggle-combo / `pref_indx` so the closure and array
        shapes match — the emulator buckets accordingly. This mirrors the setup `_call_runstate`
        does up to (but not including) the runner call.
        """
        validate_runtime_config(self._cfg)
        self.loss_criteria = float(getattr(self._cfg, "loss_criteria", 0.0005))
        var, atm, _ = _state_mod.legacy_view(rs, cfg=self._cfg)
        if (
            rs.photo_static is not None
            and getattr(self.odesolver, "_photo_static", None) is None
        ):
            self.odesolver._photo_static = rs.photo_static
        elif rs.photo_static is not None:
            # The closure bakes the first profile's star + wavelength grid;
            # only T-dependent cross sections ride per lane (ProfileVars).
            # Reject any other difference instead of using lane 0's star.
            baked = self.odesolver._photo_static
            ps = rs.photo_static
            same = (
                int(ps.nbin) == int(baked.nbin)
                and int(ps.din12_indx) == int(baked.din12_indx)
                and np.array_equal(np.asarray(ps.bins), np.asarray(baked.bins))
                and (
                    self._sflux_top_ref is None
                    or np.array_equal(
                        np.asarray(rs.photo.sflux_top), self._sflux_top_ref
                    )
                )
            )
            if not same:
                raise ValueError(
                    "run_batch photo lanes must share the star, wavelength "
                    "grid, and network (only the T-P profile may differ); "
                    "this profile's photo statics do not match the first "
                    "profile's."
                )
        self._ensure_runner(var, atm)
        # The refresh closure bakes the first profile's `pref_indx` (it sizes
        # a `jnp.arange`, so it cannot ride ProfileVars); reject a lane that
        # differs instead of silently anchoring it at lane 0's layer.
        baked_pref = int(self._refresh_static.pref_indx)
        if int(atm.pref_indx) != baked_pref:
            raise ValueError(
                "run_batch lanes must share pref_indx: this profile has "
                f"{int(atm.pref_indx)}, the runner was built with {baked_pref}; "
                "bucket profiles by pref_indx before batching."
            )
        ni = _NETWORK.ni
        nz = int(rs.atm.Tco.shape[0])
        atm_static = make_atm_static(atm, ni, nz, cfg=self._cfg)
        init_state = self._pack_state_from_runstate(rs)
        return init_state, atm_static

    def run_batch(self, states_batched, atm_static_batched):
        """Integrate a whole batch of profiles in one vmapped device call.

        `states_batched` / `atm_static_batched` are the stacked outputs of
        `prepare_runstate` (leading batch axis on every array leaf; the four
        AtmStatic toggle flags broadcast). Returns the batched final
        `JaxIntegState`; read per-lane `termination_reason` / `ymix` after
        unstacking with `unstack_integ_states`. Requires `_ensure_runner` to
        have run (via `prepare_runstate`) for this batch's (nz, toggle-combo).

        Lanes run with freeze-on-done: each profile's converged result is
        identical to running it alone, and the call returns once the slowest
        lane finishes (or every lane hits its `chunk_target` yield, which the
        host-side compaction loop uses to refill finished lanes).
        """
        if self._runner_batch is None:
            raise RuntimeError(
                "run_batch called before the runner was built; call "
                "prepare_runstate on at least one profile first."
            )
        if self._vrunner is None:
            self._vrunner = jax.jit(
                jax.vmap(self._runner_batch, in_axes=(0, _ATM_STATIC_BATCH_AXES))
            )
        return self._vrunner(states_batched, atm_static_batched)

    def _summary_shim(self, rs):
        """Build a minimal legacy-shape stand-in for the post-run prints.

        `print_end_msg` / `print_unconverged_msg` / `print_prog` only
        read counters and atom_loss, plus var.t / var.dt / var.longdy /
        var.longdydt. Compose a SimpleNamespace-style shim from the
        RunState slot rather than synthesising a full `state._Variables`.
        """
        import types

        var = types.SimpleNamespace(
            t=float(rs.step.t),
            dt=float(rs.step.dt),
            longdy=float(rs.step.longdy),
            longdydt=float(rs.step.longdydt),
            atom_loss={
                a: float(np.asarray(rs.atoms.atom_loss)[i])
                for i, a in enumerate(rs.atoms.atom_order)
            },
            y=np.asarray(rs.step.y),
        )
        para = types.SimpleNamespace(
            count=int(rs.params.count),
            nega_count=int(rs.params.nega_count),
            loss_count=int(rs.params.loss_count),
            delta_count=int(rs.params.delta_count),
            where_varies_most=np.asarray(rs.params.where_varies_most),
            end_case=int(rs.params.end_case),
            switch_final_photo_frq=bool(rs.params.switch_final_photo_frq),
            pic_count=int(rs.params.pic_count),
            solver_str="solver",
        )
        return var, para
