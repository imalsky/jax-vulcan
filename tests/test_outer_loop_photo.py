"""Validate that the photo update inside the JAX runner matches a single
Python-side `compute_tau` / `compute_flux` / `compute_J` round-trip.

Builds the HD189 reference state and runs the photo branch *both* ways:
  (A) via `op_jax.Ros2JAX.compute_{tau,flux,J}` writing into var.tau /
      var.k_arr (Python-side dispatch around the same JAX kernels).
  (B) via the in-runner photo branch closed over by
      `outer_loop._make_runner`, operating on a `JaxIntegState` carry and
      feeding back into var via `_unpack_state`.

The two should agree to machine precision because they share the same
kernels (`compute_tau_jax`, `compute_flux_jax`,
`compute_J_jax_flat`/`_compute_J_inner`) — the only difference is
whether the round-trip happens before or after device sync. Anything
beyond ~1e-13 indicates a wiring bug in the carry plumbing.

Validates:
    1. var.tau / var.aflux / var.sflux / var.dflux_d / var.dflux_u / var.prev_aflux
       agree to ≤ 1e-13.
    2. var.aflux_change agrees to ≤ 1e-13.
    3. var.k_arr entries for active photo branches agree to ≤ 1e-13.
    4. var.J_sp[(sp, nbr)] entries (incl. the (sp, 0) total) agree to ≤ 1e-13.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


PHOTO_RTOL = 1e-13


def main() -> int:
    import vulcan_cfg
    if not vulcan_cfg.use_photo:
        print("SKIP: vulcan_cfg.use_photo=False; nothing to validate.")
        return 0

    import legacy_io as op
    import op_jax
    import outer_loop
    from state import RunState, legacy_view

    # --- Build HD189 reference state with the photo pre-loop (as
    # `vulcan_jax.py` does before entering the integration loop). ---
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    output = op.Output()

    # `with_pre_loop_setup` already ran the full photo pipeline once
    # (compute_tau/flux/J + apply_photo_remove), so the legacy_view
    # state already has Path A's outputs — capture them as the Path A
    # snapshot directly. To get the pre-photo state for Path B, we need
    # to undo the photo pass; instead, re-import network and reset
    # var.k_arr to the post-rate-build, pre-photo values via
    # `setup_var_k`, then re-run Path A explicitly so we have a clean
    # pre/post split.
    import rates as _rates_mod
    _network = _rates_mod.setup_var_k(vulcan_cfg, data_var, data_atm)

    # Snapshot the pre-photo state so Path B can run from the SAME starting
    # point as Path A — otherwise Path B's compute_flux is computing the
    # 2nd photo update (using Path A's dflux_u as `dflux_u_prev`), not the 1st.
    pre_photo = dict(
        y=data_var.y.copy(),
        ymix=data_var.ymix.copy(),
        tau=data_var.tau.copy(),
        aflux=data_var.aflux.copy(),
        sflux=data_var.sflux.copy(),
        dflux_d=data_var.dflux_d.copy(),
        dflux_u=data_var.dflux_u.copy(),
        aflux_change=float(data_var.aflux_change),
        k_arr=data_var.k_arr.copy(),
    )

    # --- Path A: Python-side compute_tau / compute_flux / compute_J ---
    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    _rates_mod.apply_photo_remove(vulcan_cfg, data_var, _network, data_atm)

    # Snapshot Path A outputs for comparison.
    tau_A = data_var.tau.copy()
    aflux_A = data_var.aflux.copy()
    sflux_A = data_var.sflux.copy()
    dflux_d_A = data_var.dflux_d.copy()
    dflux_u_A = data_var.dflux_u.copy()
    prev_aflux_A = data_var.prev_aflux.copy()
    aflux_change_A = float(data_var.aflux_change)
    k_arr_A = data_var.k_arr.copy()
    J_sp_A = {k: np.copy(v) for k, v in data_var.J_sp.items()}

    # Restore pre-photo state so Path B sees what Path A saw.
    data_var.y = pre_photo["y"].copy()
    data_var.ymix = pre_photo["ymix"].copy()
    data_var.tau = pre_photo["tau"].copy()
    data_var.aflux = pre_photo["aflux"].copy()
    data_var.sflux = pre_photo["sflux"].copy()
    data_var.dflux_d = pre_photo["dflux_d"].copy()
    data_var.dflux_u = pre_photo["dflux_u"].copy()
    data_var.aflux_change = pre_photo["aflux_change"]
    data_var.k_arr = pre_photo["k_arr"].copy()

    # --- Path B: photo branch inside the JAX runner ---
    integ = outer_loop.OuterLoop(solver, output)
    integ._ensure_runner(data_var, data_atm)
    init_state = integ._pack_state(data_var, data_para, data_atm)
    photo_branch = outer_loop._make_photo_branch(integ._photo_static)
    final_state = photo_branch(init_state)
    integ._unpack_state(final_state, data_var, data_para, data_atm)

    tau_B = data_var.tau.copy()
    aflux_B = data_var.aflux.copy()
    sflux_B = data_var.sflux.copy()
    dflux_d_B = data_var.dflux_d.copy()
    dflux_u_B = data_var.dflux_u.copy()
    prev_aflux_B = data_var.prev_aflux.copy()
    aflux_change_B = float(data_var.aflux_change)
    k_arr_B = data_var.k_arr.copy()
    J_sp_B = {k: np.copy(v) for k, v in data_var.J_sp.items()}

    ok = True

    def _relerr(ref, ours):
        denom = np.maximum(np.abs(ref), 1e-300)
        return float(np.max(np.abs(ours - ref) / denom))

    # Regression: the in-runner photo branch must use the dynamic dz carried
    # by JaxIntegState, not the initial photo_static.dz closed over at trace
    # time. update_mu_dz changes dz during long runs, and op.compute_tau reads
    # the current atm.dz on every photo update.
    dz_scale = jnp.linspace(0.95, 1.05, init_state.dz.shape[0])
    dynamic_state = init_state._replace(dz=init_state.dz * dz_scale)
    dynamic_final = photo_branch(dynamic_state)
    tau_dynamic_ref = outer_loop._photo_mod.compute_tau_jax(
        dynamic_state.y, dynamic_state.dz, integ._photo_static.photo_data,
    )
    dyn_dz_err = _relerr(np.asarray(tau_dynamic_ref),
                         np.asarray(dynamic_final.tau))
    print(f"dynamic-dz tau relerr: {dyn_dz_err:.3e}")
    if dyn_dz_err > PHOTO_RTOL:
        print("FAIL: photo branch did not use dynamic state.dz")
        ok = False

    for label, A, B in (
        ("tau",        tau_A,        tau_B),
        ("aflux",      aflux_A,      aflux_B),
        ("sflux",      sflux_A,      sflux_B),
        ("dflux_d",    dflux_d_A,    dflux_d_B),
        ("dflux_u",    dflux_u_A,    dflux_u_B),
        ("prev_aflux", prev_aflux_A, prev_aflux_B),
    ):
        err = _relerr(A, B)
        print(f"{label:11s} relerr: {err:.3e}")
        if err > PHOTO_RTOL:
            print(f"FAIL: {label} mismatch")
            ok = False

    err_change = (abs(aflux_change_A - aflux_change_B)
                  / max(abs(aflux_change_A), 1e-300))
    print(f"aflux_change relerr: {err_change:.3e}  "
          f"(A={aflux_change_A:.3e}, B={aflux_change_B:.3e})")
    if err_change > PHOTO_RTOL:
        print("FAIL: aflux_change mismatch")
        ok = False

    # var.k_arr: only photo-driven reactions should change between batches;
    # anything outside that set should be bit-identical. Both paths write
    # the same J*.
    pho_re_set = {
        idx for (sp, nbr), idx in data_var.pho_rate_index.items()
        if idx not in vulcan_cfg.remove_list
    }
    max_k_relerr = 0.0
    n_changed = 0
    nr_plus_one = k_arr_A.shape[0]
    for ridx in range(1, nr_plus_one):
        diff = np.abs(k_arr_A[ridx] - k_arr_B[ridx])
        denom = np.maximum(np.abs(k_arr_A[ridx]), 1e-300)
        re = float(np.max(diff / denom))
        if ridx in pho_re_set:
            n_changed += 1
            max_k_relerr = max(max_k_relerr, re)
        if re > PHOTO_RTOL:
            print(f"FAIL: var.k_arr[{ridx}] relerr {re:.3e} (in pho={ridx in pho_re_set})")
            ok = False
    print(f"var.k_arr photo entries:   {n_changed} reactions, max relerr {max_k_relerr:.3e}")

    # J_sp: every (sp, nbr) entry in A should appear in B with same values,
    # including the (sp, 0) totals.
    max_J_relerr = 0.0
    for key, ref in J_sp_A.items():
        if key not in J_sp_B:
            print(f"FAIL: J_sp[{key}] missing in B")
            ok = False
            continue
        diff = np.abs(J_sp_B[key] - ref)
        denom = np.maximum(np.abs(ref), 1e-300)
        re = float(np.max(diff / denom))
        max_J_relerr = max(max_J_relerr, re)
        if re > PHOTO_RTOL:
            print(f"FAIL: J_sp[{key}] relerr {re:.3e}")
            ok = False
    print(f"var.J_sp entries: {len(J_sp_A)} keys, max relerr {max_J_relerr:.3e}")

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
