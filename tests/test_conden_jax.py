"""Validate Phase 10.4 conden kernels against direct NumPy reference math.

The HD189 default config sets `use_condense=False`, so an end-to-end
integration test of condensation is out of scope for this sub-milestone
(that needs an Earth or Jupiter cfg). What's validated here is the
*formula correctness* of the three pure-JAX conden kernels in
`conden.py`, fed with synthetic but physically-shaped inputs:

    1. `update_conden_rates` against the per-formula reference

           rate[r, z]  = Dg[r, z] * coeff[r] * (y[z, sp[r]] - sat_n[r, z])
           k_arr[re]   = max( rate, 0)
           k_arr[re+1] = max(-rate, 0)

       — same algebra as `op.conden`'s for-loop body
       (`VULCAN-master/op.py:1135-1151`), packed as a vectorised
       gather/multiply/scatter.

    2. `apply_h2o_relax_jax` against the implicit-Euler relaxation in
       `op.h2o_conden_evap_relax` (`op.py:1334-1370`).

    3. `apply_nh3_relax_jax` against the same shape with the
       `i <= conden_top` clamp (`op.py:1389`) plus the post-update
       `ymix[NH3_l_s] = max(ymix[NH3_l_s], 0)` clip (`op.py:1420`).

All three are pure JAX so the comparison hits float64 precision modulo
reduction order; we check `≤ 1e-13`.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

KERNEL_RTOL = 1e-13


def _make_static(nz: int, ni: int,
                 conden_re_idx, conden_sp_idx,
                 Dg_per_re, sat_n_per_re, coeff_per_re,
                 *, h2o_active=False, h2o_idx=0, h2o_l_s_idx=0,
                 h2o_Dg=None, h2o_sat=None, h2o_m_over_rho_r2=0.0,
                 nh3_active=False, nh3_idx=0, nh3_l_s_idx=0,
                 nh3_Dg=None, nh3_sat=None, nh3_m_over_rho_r2=0.0,
                 nh3_conden_top=0,
                 n_0=None, gas_indx_mask=None):
    import conden as _conden_mod
    if h2o_Dg is None: h2o_Dg = np.zeros(nz)
    if h2o_sat is None: h2o_sat = np.zeros(nz)
    if nh3_Dg is None: nh3_Dg = np.zeros(nz)
    if nh3_sat is None: nh3_sat = np.zeros(nz)
    if n_0 is None: n_0 = np.ones(nz)
    if gas_indx_mask is None: gas_indx_mask = np.ones(ni, dtype=bool)
    return _conden_mod.CondenStatic(
        conden_re_idx=jnp.asarray(np.asarray(conden_re_idx, dtype=np.int32)),
        conden_sp_idx=jnp.asarray(np.asarray(conden_sp_idx, dtype=np.int32)),
        Dg_per_re=jnp.asarray(Dg_per_re, dtype=jnp.float64),
        sat_n_per_re=jnp.asarray(sat_n_per_re, dtype=jnp.float64),
        coeff_per_re=jnp.asarray(coeff_per_re, dtype=jnp.float64),
        h2o_active=bool(h2o_active),
        h2o_idx=int(h2o_idx),
        h2o_l_s_idx=int(h2o_l_s_idx),
        h2o_Dg=jnp.asarray(h2o_Dg, dtype=jnp.float64),
        h2o_sat=jnp.asarray(h2o_sat, dtype=jnp.float64),
        h2o_m_over_rho_r2=float(h2o_m_over_rho_r2),
        nh3_active=bool(nh3_active),
        nh3_idx=int(nh3_idx),
        nh3_l_s_idx=int(nh3_l_s_idx),
        nh3_Dg=jnp.asarray(nh3_Dg, dtype=jnp.float64),
        nh3_sat=jnp.asarray(nh3_sat, dtype=jnp.float64),
        nh3_m_over_rho_r2=float(nh3_m_over_rho_r2),
        nh3_conden_top=int(nh3_conden_top),
        n_0=jnp.asarray(n_0, dtype=jnp.float64),
        gas_indx_mask=jnp.asarray(gas_indx_mask),
    )


def _relerr(ref, ours):
    ref = np.asarray(ref)
    ours = np.asarray(ours)
    denom = np.maximum(np.abs(ref), 1e-300)
    return float(np.max(np.abs(ours - ref) / denom))


def test_update_conden_rates() -> bool:
    """Two condensation reactions over (nz=5, ni=4) — verify scatter,
    sign split, and the use_relax short-circuit (coeff=0 → all-zero rate).
    """
    import conden as _conden_mod

    rng = np.random.default_rng(42)
    nz, ni = 5, 4
    nr = 7

    # Reaction 1: at k_arr rows (1, 2), species index 0
    # Reaction 2: at k_arr rows (4, 5), species index 2
    conden_re_idx = np.array([1, 4], dtype=np.int32)
    conden_sp_idx = np.array([0, 2], dtype=np.int32)
    Dg_per_re = rng.uniform(1.0, 5.0, size=(2, nz))
    sat_n_per_re = rng.uniform(0.1, 1.0, size=(2, nz))
    coeff_per_re = np.array([0.7, 0.0])  # 2nd reaction is "use_relax" (coeff=0)

    k_arr = rng.uniform(1.0, 10.0, size=(nr + 1, nz))
    y = rng.uniform(0.05, 1.5, size=(nz, ni))

    st = _make_static(nz, ni,
                      conden_re_idx, conden_sp_idx,
                      Dg_per_re, sat_n_per_re, coeff_per_re)

    k_jax = np.asarray(_conden_mod.update_conden_rates(
        jnp.asarray(k_arr), jnp.asarray(y), st
    ))

    # Reference: per-reaction loop in numpy.
    k_ref = k_arr.copy()
    for r in range(2):
        sp_idx = int(conden_sp_idx[r])
        re_idx = int(conden_re_idx[r])
        rate = Dg_per_re[r] * coeff_per_re[r] * (y[:, sp_idx] - sat_n_per_re[r])
        k_ref[re_idx]     = np.maximum(rate, 0.0)
        k_ref[re_idx + 1] = np.maximum(-rate, 0.0)

    err = _relerr(k_ref, k_jax)
    print(f"update_conden_rates relerr: {err:.3e}")

    # Sanity: reaction 2 (coeff=0) should produce zeros at rows 4 and 5.
    zero_check = np.max(np.abs(k_jax[4])) + np.max(np.abs(k_jax[5]))
    print(f"use_relax short-circuit zero check: {zero_check:.3e}")

    # Sanity: untouched rows (0, 3, 6, 7) should equal the input.
    untouched_err = max(_relerr(k_arr[r], k_jax[r]) for r in (0, 3, 6, 7))
    print(f"untouched rows relerr: {untouched_err:.3e}")

    ok = (err <= KERNEL_RTOL and zero_check < 1e-30
          and untouched_err <= KERNEL_RTOL)
    return ok


def test_apply_h2o_relax_jax() -> bool:
    """Verify against a per-cell numpy implementation of op.h2o_conden_evap_relax."""
    import conden as _conden_mod

    rng = np.random.default_rng(7)
    nz, ni = 6, 3
    h2o_idx, h2o_l_s_idx = 0, 1

    # Deliberately mix conden (y > sat) and evap (y < sat) layers.
    y = np.zeros((nz, ni))
    sat = rng.uniform(0.1, 1.0, size=nz)
    y[:, h2o_idx] = sat * np.array([2.0, 0.5, 1.5, 0.3, 1.2, 0.8])
    y[:, h2o_l_s_idx] = rng.uniform(0.01, 0.5, size=nz)
    y[:, 2] = rng.uniform(0.5, 1.0, size=nz)
    ymix = y / np.sum(y, axis=1, keepdims=True)
    n_0 = np.sum(y, axis=1)

    Dg = rng.uniform(0.5, 2.0, size=nz)
    m_over_rho_r2 = 0.123
    dt = 0.4

    st = _make_static(nz, ni,
                      conden_re_idx=[], conden_sp_idx=[],
                      Dg_per_re=np.zeros((0, nz)),
                      sat_n_per_re=np.zeros((0, nz)),
                      coeff_per_re=np.zeros(0),
                      h2o_active=True,
                      h2o_idx=h2o_idx, h2o_l_s_idx=h2o_l_s_idx,
                      h2o_Dg=Dg, h2o_sat=sat,
                      h2o_m_over_rho_r2=m_over_rho_r2,
                      n_0=n_0)

    y_jax, ymix_jax = _conden_mod.apply_h2o_relax_jax(
        jnp.asarray(y), jnp.asarray(ymix), jnp.asarray(dt), st
    )
    y_jax = np.asarray(y_jax)
    ymix_jax = np.asarray(ymix_jax)

    # Reference: NumPy port of op.h2o_conden_evap_relax (op.py:1334-1370).
    tau = 1.0 / (Dg * m_over_rho_r2 * (y[:, h2o_idx] - sat))
    sat_mix = sat / n_0
    y_conden = (ymix[:, h2o_idx] + dt / tau * sat_mix) / (1.0 + dt / tau)
    ice_loss = (y[:, h2o_idx] - sat) * dt / tau
    ice_loss = np.minimum(y[:, h2o_l_s_idx], ice_loss)
    conden_mask = tau > 0
    evap_mask = tau < 0

    ymix_ref = ymix.copy()
    ymix_ref[conden_mask, h2o_l_s_idx] += (ymix_ref[conden_mask, h2o_idx]
                                            - y_conden[conden_mask])
    ymix_ref[conden_mask, h2o_idx] = y_conden[conden_mask]
    ymix_ref[evap_mask, h2o_idx] += ice_loss[evap_mask] / n_0[evap_mask]
    ymix_ref[evap_mask, h2o_l_s_idx] -= ice_loss[evap_mask] / n_0[evap_mask]

    # The projection uses pre-relax y (op.py:1368).
    ysum = np.sum(y, axis=1, keepdims=True)
    y_ref = ymix_ref * ysum

    ymix_err = _relerr(ymix_ref, ymix_jax)
    y_err = _relerr(y_ref, y_jax)
    print(f"apply_h2o_relax_jax  ymix relerr: {ymix_err:.3e}")
    print(f"apply_h2o_relax_jax  y    relerr: {y_err:.3e}")

    return ymix_err <= KERNEL_RTOL and y_err <= KERNEL_RTOL


def test_apply_nh3_relax_jax() -> bool:
    """Verify against a per-cell numpy implementation of op.nh3_conden_evap_relax,
    including the conden_top clamp and the NH3_l_s >= 0 clip."""
    import conden as _conden_mod

    rng = np.random.default_rng(11)
    nz, ni = 7, 3
    nh3_idx, nh3_l_s_idx = 0, 1

    sat = rng.uniform(0.05, 1.0, size=nz)
    y = np.zeros((nz, ni))
    # Mix of sat ratios; some condense (y > sat), some evap (y < sat),
    # including layers at and above conden_top so the clamp triggers.
    y[:, nh3_idx] = sat * np.array([2.0, 1.4, 0.6, 1.1, 0.4, 1.3, 0.8])
    y[:, nh3_l_s_idx] = np.array([0.05, 0.0, 0.4, 0.5, 1.0, 0.05, 0.05])
    y[:, 2] = rng.uniform(0.5, 1.0, size=nz)
    ymix = y / np.sum(y, axis=1, keepdims=True)
    n_0 = np.sum(y, axis=1)

    Dg = rng.uniform(0.5, 2.0, size=nz)
    m_over_rho_r2 = 0.087
    dt = 0.6
    sat_mix = sat / n_0
    conden_top = int(np.argmin(sat_mix))

    st = _make_static(nz, ni,
                      conden_re_idx=[], conden_sp_idx=[],
                      Dg_per_re=np.zeros((0, nz)),
                      sat_n_per_re=np.zeros((0, nz)),
                      coeff_per_re=np.zeros(0),
                      nh3_active=True,
                      nh3_idx=nh3_idx, nh3_l_s_idx=nh3_l_s_idx,
                      nh3_Dg=Dg, nh3_sat=sat,
                      nh3_m_over_rho_r2=m_over_rho_r2,
                      nh3_conden_top=conden_top,
                      n_0=n_0)

    y_jax, ymix_jax = _conden_mod.apply_nh3_relax_jax(
        jnp.asarray(y), jnp.asarray(ymix), jnp.asarray(dt), st
    )
    y_jax = np.asarray(y_jax)
    ymix_jax = np.asarray(ymix_jax)

    # Reference: NumPy port of op.nh3_conden_evap_relax (op.py:1372-1424).
    tau = 1.0 / (Dg * m_over_rho_r2 * (y[:, nh3_idx] - sat))
    y_conden = (ymix[:, nh3_idx] + dt / tau * sat_mix) / (1.0 + dt / tau)
    ice_loss = (y[:, nh3_idx] - sat) * dt / tau
    ice_loss = np.minimum(y[:, nh3_l_s_idx], ice_loss)
    layer_idx = np.arange(nz)
    conden_mask = (tau > 0) & (layer_idx <= conden_top)
    evap_mask = tau < 0

    ymix_ref = ymix.copy()
    ymix_ref[conden_mask, nh3_l_s_idx] += (ymix_ref[conden_mask, nh3_idx]
                                            - y_conden[conden_mask])
    ymix_ref[conden_mask, nh3_idx] = y_conden[conden_mask]
    ymix_ref[evap_mask, nh3_idx] += ice_loss[evap_mask] / n_0[evap_mask]
    ymix_ref[evap_mask, nh3_l_s_idx] -= ice_loss[evap_mask] / n_0[evap_mask]
    ymix_ref[:, nh3_l_s_idx] = np.maximum(ymix_ref[:, nh3_l_s_idx], 0.0)

    # Projection uses pre-relax y (op.py:1422).
    ysum = np.sum(y, axis=1, keepdims=True)
    y_ref = ymix_ref * ysum

    ymix_err = _relerr(ymix_ref, ymix_jax)
    y_err = _relerr(y_ref, y_jax)
    print(f"apply_nh3_relax_jax  ymix relerr: {ymix_err:.3e}")
    print(f"apply_nh3_relax_jax  y    relerr: {y_err:.3e}")

    return ymix_err <= KERNEL_RTOL and y_err <= KERNEL_RTOL


def test_no_op_when_inactive() -> bool:
    """When use_relax flags are False, the kernels must pass through unchanged."""
    import conden as _conden_mod

    rng = np.random.default_rng(99)
    nz, ni = 4, 3
    y = rng.uniform(0.1, 1.0, size=(nz, ni))
    ymix = y / np.sum(y, axis=1, keepdims=True)
    dt = 0.1

    st = _make_static(nz, ni,
                      conden_re_idx=[], conden_sp_idx=[],
                      Dg_per_re=np.zeros((0, nz)),
                      sat_n_per_re=np.zeros((0, nz)),
                      coeff_per_re=np.zeros(0))

    y_h2o, ymix_h2o = _conden_mod.apply_h2o_relax_jax(
        jnp.asarray(y), jnp.asarray(ymix), jnp.asarray(dt), st
    )
    y_nh3, ymix_nh3 = _conden_mod.apply_nh3_relax_jax(
        jnp.asarray(y), jnp.asarray(ymix), jnp.asarray(dt), st
    )

    err_h2o_y    = _relerr(y, np.asarray(y_h2o))
    err_h2o_ymix = _relerr(ymix, np.asarray(ymix_h2o))
    err_nh3_y    = _relerr(y, np.asarray(y_nh3))
    err_nh3_ymix = _relerr(ymix, np.asarray(ymix_nh3))
    print(f"inactive H2O relax y     relerr: {err_h2o_y:.3e}")
    print(f"inactive H2O relax ymix  relerr: {err_h2o_ymix:.3e}")
    print(f"inactive NH3 relax y     relerr: {err_nh3_y:.3e}")
    print(f"inactive NH3 relax ymix  relerr: {err_nh3_ymix:.3e}")

    return all(e <= KERNEL_RTOL for e in
               (err_h2o_y, err_h2o_ymix, err_nh3_y, err_nh3_ymix))


def main() -> int:
    ok = True
    print("--- update_conden_rates ---")
    ok &= test_update_conden_rates()
    print("--- apply_h2o_relax_jax ---")
    ok &= test_apply_h2o_relax_jax()
    print("--- apply_nh3_relax_jax ---")
    ok &= test_apply_nh3_relax_jax()
    print("--- no-op when inactive ---")
    ok &= test_no_op_when_inactive()
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
