"""Self-consistent gravity in the hydrostatic atm refresh.

update_mu_dz_jax computes each layer's g from the zco the same sweep
produced, as master's op.update_mu_dz does. Checks:
- g == gs*(Rp/(Rp+zco))**2 to roundoff on both sides of pref_indx;
- constant-g integration moves the top height by > 0.5% on HD189, so the
  check is not vacuous.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

GRAVITY_RTOL = 1e-13  # g recomputed from zco agrees to roundoff


def main() -> int:
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax import atm_refresh
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)

    # Perturb ymix so mu (hence the scale height and zco) is non-trivial
    # relative to the initial state — otherwise the refresh reproduces the
    # pre-loop profile and the test is vacuous.
    rng = np.random.default_rng(0)
    pert = 1.0 + 1e-3 * rng.standard_normal(data_var.ymix.shape)
    data_var.ymix = data_var.ymix * pert
    data_var.ymix = data_var.ymix / np.sum(data_var.ymix, axis=1, keepdims=True)
    data_var.y = data_atm.n_0[:, None] * data_var.ymix

    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo and rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver, op.Output())
    integ._ensure_runner(data_var, data_atm)

    st = integ._refresh_static
    gs = float(st.gs)
    Rp = float(st.Rp)
    pref_indx = int(st.pref_indx)

    _mu, g_j, _Hp, _dz, zco_j, _dzi, _Hpi = atm_refresh.update_mu_dz_jax(
        jnp.asarray(data_var.ymix), st
    )

    g = np.asarray(g_j, dtype=np.float64)
    zco = np.asarray(zco_j, dtype=np.float64)
    nz = g.shape[0]

    ok = True

    # --- 1. Self-consistency: g[i] from the SAME refresh's zco[i] ---
    # `zco` has nz+1 entries (it carries the reference level plus every layer
    # boundary the scan produced); g has nz entries. At/above the reference
    # layer the forward scan computes g[i] = gs*(Rp/(Rp+zco[i]))^2 from the
    # carry zco[i], so compare against zco[pref_indx:nz] (drop the top interface).
    g_expected_up = gs * (Rp / (Rp + zco[pref_indx:nz])) ** 2
    rel_up = np.max(np.abs(g[pref_indx:] - g_expected_up) / np.abs(g_expected_up))
    print(f"self-consistency relerr (i >= pref_indx={pref_indx}): {rel_up:.3e}")
    if rel_up > GRAVITY_RTOL:
        print("FAIL: gravity is not self-consistent with the refreshed zco")
        ok = False

    # Below the reference layer the backward scan computes g from zco[i+1].
    if pref_indx > 0:
        g_expected_dn = gs * (Rp / (Rp + zco[1 : pref_indx + 1])) ** 2
        rel_dn = np.max(np.abs(g[:pref_indx] - g_expected_dn) / np.abs(g_expected_dn))
        print(f"self-consistency relerr (i <  pref_indx): {rel_dn:.3e}")
        if rel_dn > GRAVITY_RTOL:
            print("FAIL: below-reference gravity is not self-consistent")
            ok = False

    # --- 2. Non-triviality: constant-g (g == gs) gives a different TOA height ---
    # Reproduce the upward hydrostatic integration with g pinned to gs (a stand-in
    # for any stale/previous-cycle gravity that ignores the height it just built).
    Tco = np.asarray(st.Tco, dtype=np.float64)
    pico = np.asarray(st.pico, dtype=np.float64)
    mol_mass = np.asarray(st.mol_mass, dtype=np.float64)
    Navo = float(st.Navo)
    kb = float(st.kb)
    ymix = np.asarray(data_var.ymix, dtype=np.float64)
    mu = ymix @ mol_mass

    zco_const = np.zeros(nz, dtype=np.float64)
    z = float(st.zco_pref)
    for i in range(pref_indx, nz):
        Hp_i = kb * Tco[i] / (mu[i] / Navo * gs)  # constant g == gs
        z = z + Hp_i * np.log(pico[i] / pico[i + 1])
        zco_const[i] = z

    toa_rel = abs(zco_const[-1] - zco[-1]) / abs(zco[-1])
    print(f"TOA relative difference (self-consistent vs constant-g): {toa_rel:.3e}")
    if toa_rel < 5e-3:
        print("FAIL: self-consistent gravity is indistinguishable from constant g")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
