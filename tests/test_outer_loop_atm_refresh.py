"""Validate Phase 10.3: atm refresh + hydrostatic balance inside the JAX
runner agree with the Python-side `op.Integration.update_mu_dz` /
`update_phi_esc` + `var.y = n_0 * var.ymix` baseline.

Builds the HD189 reference state and runs the atm-refresh branch *both* ways:

  (A) via `op.Integration.update_mu_dz` / `update_phi_esc` writing into atm,
      followed by Python `var.y = n_0 * var.ymix` (the pre-Phase-10.3 path).

  (B) via the Phase 10.3 atm-refresh branch closed over by
      `outer_loop._make_atm_refresh_branch`, with hydrostatic balance applied
      inside `body_fn` of the JAX runner.

Both paths share the same kernels (`atm_refresh.update_mu_dz_jax`, etc.)
modulo the dict-iteration boundary, so agreement should land at ~1e-13 or
better — the only differences are float64 reduction order across `lax.scan`
vs Python `for`. Anything larger means a wiring bug.

Validates:
    1. atm.{mu, g, Hp, dz, dzi, Hpi, zco, top_flux} match to ≤ 1e-13.
    2. var.y after hydrostatic balance matches to ≤ 1e-13.
    3. Carry's atm fields match `atm.*` from path A to ≤ 1e-13.
"""
from __future__ import annotations

import copy
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Oracle test: requires VULCAN-master sibling for the upstream
# op.Integration.update_mu_dz / update_phi_esc reference. Skip when absent.
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"VULCAN-master oracle absent at {VULCAN_MASTER}; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )
sys.path.append(str(VULCAN_MASTER))

warnings.filterwarnings("ignore")


REFRESH_RTOL = 1e-13


def main() -> int:
    import vulcan_cfg
    import store, build_atm, op, op_jax, outer_loop

    # --- Build HD189 reference state ---
    data_var = store.Variables()
    data_atm = store.AtmData()
    data_para = store.Parameters()

    make_atm = build_atm.Atm()
    output = op.Output()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    data_var = rate.remove_rate(data_var)
    ini_abun = build_atm.InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)

    if vulcan_cfg.use_photo:
        # `_ensure_runner` builds the photo branch too, which needs var.nbin
        # and friends; do the photo pre-loop here even though the test only
        # exercises the refresh branch.
        rate.make_bins_read_cross(data_var, data_atm)
        make_atm.read_sflux(data_var, data_atm)

    # Perturb ymix slightly so the refresh produces non-trivial deltas
    # vs the initial state (otherwise mu_post == mu_pre and it's not a
    # real test of the loop).
    rng = np.random.default_rng(0)
    pert = 1.0 + 1e-3 * rng.standard_normal(data_var.ymix.shape)
    data_var.ymix = data_var.ymix * pert
    data_var.ymix = data_var.ymix / np.sum(data_var.ymix, axis=1, keepdims=True)
    data_var.y = data_atm.n_0[:, None] * data_var.ymix

    # --- Path A: Python-side update_mu_dz / update_phi_esc + hydro balance ---
    atm_A = copy.deepcopy(data_atm)
    var_A = copy.deepcopy(data_var)
    integ_ref = op.Integration(op.Ros2(), output)
    atm_A = integ_ref.update_mu_dz(var_A, atm_A, make_atm)
    atm_A = integ_ref.update_phi_esc(var_A, atm_A)
    if vulcan_cfg.use_condense:
        var_A.y[:, atm_A.gas_indx] = np.vstack(atm_A.n_0) * var_A.ymix[:, atm_A.gas_indx]
    else:
        var_A.y = np.vstack(atm_A.n_0) * var_A.ymix

    mu_A = atm_A.mu.copy()
    g_A = atm_A.g.copy()
    Hp_A = atm_A.Hp.copy()
    dz_A = atm_A.dz.copy()
    dzi_A = atm_A.dzi.copy()
    Hpi_A = atm_A.Hpi.copy()
    zco_A = atm_A.zco.copy()
    top_flux_A = atm_A.top_flux.copy()
    y_A = var_A.y.copy()

    # --- Path B: atm-refresh branch + hydrostatic balance via JAX runner ---
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), output)
    integ._ensure_runner(data_var, data_atm)

    # Use the standalone atm refresh branch on a packed initial state:
    # this exercises update_mu_dz_jax + update_phi_esc_jax wiring without
    # depending on the photo branch / chem step.
    init_state = integ._pack_state(data_var, data_para, data_atm)
    refresh_branch = outer_loop._make_atm_refresh_branch(integ._refresh_static)
    after_refresh_state = refresh_branch(init_state)

    mu_B = np.asarray(after_refresh_state.mu)
    g_B = np.asarray(after_refresh_state.g)
    Hp_B = np.asarray(after_refresh_state.Hp)
    dz_B = np.asarray(after_refresh_state.dz)
    dzi_B = np.asarray(after_refresh_state.dzi)
    Hpi_B = np.asarray(after_refresh_state.Hpi)
    zco_B = np.asarray(after_refresh_state.zco)
    top_flux_B = np.asarray(after_refresh_state.top_flux)

    # Hydrostatic balance: y_B = n_0 * ymix. body_fn applies this after
    # the Ros2 step; here we exercise it standalone against atm.n_0
    # (a static quantity equal to atm.M).
    y_B = data_atm.n_0[:, None] * np.asarray(after_refresh_state.ymix)

    ok = True

    def _relerr(ref, ours):
        ref = np.asarray(ref)
        ours = np.asarray(ours)
        denom = np.maximum(np.abs(ref), 1e-300)
        return float(np.max(np.abs(ours - ref) / denom))

    for label, A, B in (
        ("mu",       mu_A,       mu_B),
        ("g",        g_A,        g_B),
        ("Hp",       Hp_A,       Hp_B),
        ("dz",       dz_A,       dz_B),
        ("dzi",      dzi_A,      dzi_B),
        ("Hpi",      Hpi_A,      Hpi_B),
        ("zco",      zco_A,      zco_B),
        ("top_flux", top_flux_A, top_flux_B),
        ("y_post_hydro", y_A,    y_B),
    ):
        err = _relerr(A, B)
        print(f"{label:14s} relerr: {err:.3e}")
        if err > REFRESH_RTOL:
            print(f"FAIL: {label} mismatch")
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
