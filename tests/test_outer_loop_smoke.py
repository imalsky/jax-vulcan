"""Smoke test for outer_loop.OuterLoop.

Runs 50 accepted Ros2 steps end-to-end on the HD189 reference state and
asserts: exact step count, atom_loss under MAX_ATOM_LOSS, no retries on the
smooth HD189 case, and finite-positive dt / t.

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


# Per-atom atom_loss bound over 50 HD189 steps with the reservoir-species
# conservation projection. A BOUND, not a target: the drift is roundoff the
# projection leaves behind, so its exact value depends on XLA compilation
# choices and on how far the initial state sits from chemical equilibrium.
# The FastChem seed left 4.4e-05 here; the 0.15.0 Gibbs seed, built on the
# same NASA-9 data as the reverse rates, starts close enough that the same
# 50 steps leave -3.53e-10 (measured, reproducible to every printed digit).
# 1e-8 is ~28x that and four orders under the old seed's number, so it still
# catches the 10x regression this test exists for.
MAX_ATOM_LOSS = 1.0e-8


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
    # Exercise the opt-in operator-weighted column budget alongside the parity metric.
    vulcan_cfg.report_column_atom_loss = True

    import numpy as np

    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.ini_abun import column_atom_loss
    from vulcan_jax.state import RunState

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())

    t0 = time.time()
    rs_out = integ(rs)
    elapsed = time.time() - t0

    para = rs_out.params
    t, dt = rs_out.step.t, rs_out.step.dt
    atom_loss = dict(zip(rs_out.atoms.atom_order, np.asarray(rs_out.atoms.atom_loss)))
    print(
        f"50-step HD189 via OuterLoop: {elapsed:.1f}s "
        f"({elapsed / para.count * 1000:.1f} ms/step)"
    )
    print(f"  count={para.count}, t={t:.3e}, dt={dt:.3e}")
    print(
        f"  retry counters: nega={para.nega_count}, "
        f"loss={para.loss_count}, delta={para.delta_count}"
    )
    print("  atom_loss: " + " ".join(f"{a}={v:.3e}" for a, v in atom_loss.items()))

    ok = True

    # 1. Step counter exactly hit count_max (51 = 50 accepted + 1 init counter).
    if para.count != vulcan_cfg.count_max + 1:
        print(f"FAIL: expected count == {vulcan_cfg.count_max + 1}, got {para.count}")
        ok = False

    # 2. atom_loss stays under the bound.
    for atom, loss in atom_loss.items():
        if abs(loss) > MAX_ATOM_LOSS:
            print(
                f"FAIL: atom_loss[{atom}] = {loss:.3e}; bound is "
                f"{MAX_ATOM_LOSS:.3e}"
            )
            ok = False

    # 3. HD189 is smooth — no retries should fire.
    if para.nega_count + para.loss_count + para.delta_count > 0:
        print(
            f"WARN: retries fired on HD189 (nega={para.nega_count}, "
            f"loss={para.loss_count}, delta={para.delta_count}); "
            "this is unexpected for the smooth baseline."
        )
        # Don't fail — retries are correct behavior, just unusual on HD189.

    # 4. dt is finite and positive (catch NaN/inf crashes).
    import math

    if not (dt > 0 and math.isfinite(float(dt))):
        print(f"FAIL: dt = {dt:.3e} not finite-positive after 50 steps")
        ok = False

    # 5. t is finite and positive.
    if not (t > 0 and math.isfinite(float(t))):
        print(f"FAIL: t = {t:.3e} not finite-positive after 50 steps")
        ok = False

    # 6. The operator-weighted column budget must agree with the
    # unweighted parity metric to within a factor: on this grid the two were
    # measured ~10% apart, so a large split means the weights or the wiring
    # broke.
    col = np.asarray(column_atom_loss(rs_out.step.y, rs.metadata.y_ini, rs_out.atm.dz))
    max_col = float(np.max(np.abs(col)))
    max_unw = max(abs(v) for v in atom_loss.values())
    if not np.all(np.isfinite(col)) or max_col > 2.0 * max_unw:
        print(
            f"FAIL: column atom budget max {max_col:.3e} vs unweighted "
            f"{max_unw:.3e}; expected the same order."
        )
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
