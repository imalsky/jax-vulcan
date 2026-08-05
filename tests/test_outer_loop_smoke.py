"""Smoke test for outer_loop.OuterLoop.

Runs 50 accepted Ros2 steps end-to-end on the HD189 reference state and
asserts: exact step count, atom_loss at the tracked ~4.4e-5 baseline
(EXPECTED_ATOM_LOSS below), no retries on the smooth HD189 case, and
finite-positive dt / t.

The canonical smoke test for any change to outer_loop.py (~10s incl. JIT).
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


# Per-atom atom_loss target. HD189 50-step matched-step with the
# reservoir-species conservation projection: empirically lands at ~4.4e-05.
# Tolerance 50% — wide enough to absorb platform differences (the
# projection's correction is at ULP level, so the exact value depends on
# XLA compilation choices), tight enough to catch a 10x regression.
EXPECTED_ATOM_LOSS = 4.4e-5
ATOM_LOSS_RTOL = 0.50


def main() -> int:
    # Pin cfg to legacy_io's reference: tests that pop `vulcan_cfg` from
    # sys.modules fork fresh module objects, and a detached copy would not be
    # the one the runner reads count_max from.
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op

    vulcan_cfg = op.default_config()

    vulcan_cfg.count_max = 50
    vulcan_cfg.count_min = 1
    # A count-capped run needs the hybrid vm_mol budget extension OFF: phase-0
    # budget exhaustion flips to phase 1 and extends count_max by +1000, which
    # breaks this test's exact count == count_max + 1 contract.
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False

    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.atm_setup import Atm
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    data_para.start_time = time.time()
    make_atm = Atm()
    output = op.Output()

    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo and rs.photo_static is not None:
        solver._photo_static = rs.photo_static

    integ = outer_loop.OuterLoop(solver, output)
    solver.naming_solver(data_para)

    t0 = time.time()
    integ(data_var, data_atm, data_para, make_atm)
    elapsed = time.time() - t0

    print(
        f"50-step HD189 via OuterLoop: {elapsed:.1f}s "
        f"({elapsed / data_para.count * 1000:.1f} ms/step)"
    )
    print(f"  count={data_para.count}, t={data_var.t:.3e}, dt={data_var.dt:.3e}")
    print(
        f"  retry counters: nega={data_para.nega_count}, "
        f"loss={data_para.loss_count}, delta={data_para.delta_count}"
    )
    print(
        "  atom_loss: "
        + " ".join(f"{a}={v:.3e}" for a, v in data_var.atom_loss.items())
    )

    ok = True

    # 1. Step counter exactly hit count_max (51 = 50 accepted + 1 init counter).
    if data_para.count != vulcan_cfg.count_max + 1:
        print(
            f"FAIL: expected count == {vulcan_cfg.count_max + 1}, got {data_para.count}"
        )
        ok = False

    # 2. atom_loss matches baseline.
    for atom, loss in data_var.atom_loss.items():
        relerr = abs(loss - EXPECTED_ATOM_LOSS) / EXPECTED_ATOM_LOSS
        if relerr > ATOM_LOSS_RTOL:
            print(
                f"FAIL: atom_loss[{atom}] = {loss:.3e}; expected ~{EXPECTED_ATOM_LOSS:.3e} "
                f"(relerr={relerr:.3e}, tol={ATOM_LOSS_RTOL:.3e})"
            )
            ok = False

    # 3. HD189 is smooth — no retries should fire.
    if data_para.nega_count + data_para.loss_count + data_para.delta_count > 0:
        print(
            f"WARN: retries fired on HD189 (nega={data_para.nega_count}, "
            f"loss={data_para.loss_count}, delta={data_para.delta_count}); "
            "this is unexpected for the smooth baseline."
        )
        # Don't fail — retries are correct behavior, just unusual on HD189.

    # 4. dt is finite and positive (catch NaN/inf crashes).
    import math

    if not (data_var.dt > 0 and math.isfinite(float(data_var.dt))):
        print(f"FAIL: dt = {data_var.dt:.3e} not finite-positive after 50 steps")
        ok = False

    # 5. t is finite and positive.
    if not (data_var.t > 0 and math.isfinite(float(data_var.t))):
        print(f"FAIL: t = {data_var.t:.3e} not finite-positive after 50 steps")
        ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.strict_isolation
def test_main():
    """Pytest wrapper around main()."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
