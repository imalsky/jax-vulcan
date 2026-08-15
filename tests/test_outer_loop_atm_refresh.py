"""In-runner atm refresh + hydrostatic balance vs master's update_mu_dz /
update_phi_esc + `var.y = n_0 * var.ymix` on HD189.

The two are NOT the same kernel: master computes g(z) from a STALE atm.zco
while `update_mu_dz_jax` is self-consistent, a documented divergence (~1.8%
at the top of atmosphere; positive pin in tests/test_atm_refresh_gravity.py).
So the hydrostatic fields (g/Hp/dz/dzi/Hpi/zco) are held to REFRESH_RTOL
(2e-2, asserted so a magnitude change surfaces), while mu and post-hydro
var.y must match to <= 1e-13.

`top_flux` is intentionally not compared: physically inert on HD189, and the
two harness paths seed its BC array differently (a harness artifact).
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

from oracle import oracle_dir_or_skip  # noqa: E402

# The oracle location comes from $VULCAN_MASTER_DIR only, never a sibling
# guess. The PARENT process verifies the pinned revision and a clean tree
# (run_oracle_subprocess -> oracle_worktree -> require_oracle) and points
# this at a temporary COPY; see oracle.oracle_dir_or_skip.
VULCAN_MASTER = oracle_dir_or_skip("this atm-refresh comparison")

warnings.filterwarnings("ignore")


REFRESH_RTOL = 2e-2


def main() -> int:
    os.chdir(VULCAN_MASTER)
    sys.path.append(str(VULCAN_MASTER))
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    import op

    os.chdir(ROOT)
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.atm_setup import Atm
    from vulcan_jax.state import RunState, legacy_view

    # --- Build HD189 reference state ---
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    make_atm = Atm()
    output = op.Output()

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
        var_A.y[:, atm_A.gas_indx] = (
            np.vstack(atm_A.n_0) * var_A.ymix[:, atm_A.gas_indx]
        )
    else:
        var_A.y = np.vstack(atm_A.n_0) * var_A.ymix

    mu_A = atm_A.mu.copy()
    g_A = atm_A.g.copy()
    Hp_A = atm_A.Hp.copy()
    dz_A = atm_A.dz.copy()
    dzi_A = atm_A.dzi.copy()
    Hpi_A = atm_A.Hpi.copy()
    zco_A = atm_A.zco.copy()
    y_A = var_A.y.copy()

    # --- Path B: atm-refresh branch + hydrostatic balance via JAX runner ---
    solver_B = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo and rs.photo_static is not None:
        solver_B._photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver_B, output)
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

    for label, A, B, rtol in (
        ("mu", mu_A, mu_B, 1e-13),
        ("g", g_A, g_B, REFRESH_RTOL),
        ("Hp", Hp_A, Hp_B, REFRESH_RTOL),
        ("dz", dz_A, dz_B, REFRESH_RTOL),
        ("dzi", dzi_A, dzi_B, REFRESH_RTOL),
        ("Hpi", Hpi_A, Hpi_B, REFRESH_RTOL),
        ("zco", zco_A, zco_B, REFRESH_RTOL),
        ("y_post_hydro", y_A, y_B, 1e-13),
    ):
        err = _relerr(A, B)
        print(f"{label:14s} relerr: {err:.3e}")
        if err > rtol:
            print(f"FAIL: {label} mismatch")
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Run the master comparison in a fresh Python process."""
    from oracle import run_oracle_subprocess

    run_oracle_subprocess(__file__, "vulcan2_ncho",
                          "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    sys.exit(main())
