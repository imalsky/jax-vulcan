"""use_fix_H2He bottom pin (op.py:2935-2941).

Master snapshots the bottom H2/He mixing ratios at the first step with
t > 1e6 and pins them through use_fix_sp_bot; the runner does this with a
one-shot h2he_pinned carry. Seeded at t = 1.5e6, a 5-step HD189 run must
snapshot the pre-run ratios and hold y[0] at ymix_pre * n_0[0] to 1e-6.
The pin is not exact because upstream pins sol[0] before the clip and the
y = n_0 * ymix rebalance (op.py:2945-2946).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def main() -> int:
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    import vulcan_jax.chem_funs as chem_funs

    species = list(chem_funs.spec_list)
    if "H2" not in species or "He" not in species:
        print("SKIP: H2 or He not in network — use_fix_H2He requires both.")
        return 0

    vulcan_cfg.use_fix_H2He = True
    vulcan_cfg.count_max = 5
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False

    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.state import RunState

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    # Seed t past 1e6 so the trip fires on the first accepted step.
    rs = rs._replace(step=rs.step._replace(t=1.5e6))
    h2_idx = species.index("H2")
    he_idx = species.index("He")
    h2_mix_pre = float(rs.step.ymix[0, h2_idx])
    he_mix_pre = float(rs.step.ymix[0, he_idx])
    n0_bot = float(rs.atm.n_0[0])
    h2_target = h2_mix_pre * n0_bot
    he_target = he_mix_pre * n0_bot

    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())
    state, atm_static = integ.prepare_runstate(rs)
    final = integ._runner(state, atm_static)

    h2_post = float(final.y[0, h2_idx])
    he_post = float(final.y[0, he_idx])
    h2_relerr = abs(h2_post - h2_target) / max(abs(h2_target), 1e-300)
    he_relerr = abs(he_post - he_target) / max(abs(he_target), 1e-300)
    print(
        f"H2 pre-pin ymix = {h2_mix_pre:.6e}; "
        f"target n = {h2_target:.6e}; post-run y[0] = {h2_post:.6e}; "
        f"relerr = {h2_relerr:.3e}"
    )
    print(
        f"He pre-pin ymix = {he_mix_pre:.6e}; "
        f"target n = {he_target:.6e}; post-run y[0] = {he_post:.6e}; "
        f"relerr = {he_relerr:.3e}"
    )
    ok_pin = (h2_relerr < 1e-6) and (he_relerr < 1e-6)

    # The one-shot snapshot fired and holds the pre-run mixing ratios.
    snap = np.asarray(final.h2he_mix)
    ok_snap = bool(final.h2he_pinned) and np.array_equal(
        snap, [h2_mix_pre, he_mix_pre]
    )
    print(f"h2he_pinned={bool(final.h2he_pinned)}, h2he_mix={snap}; ok={ok_snap}")
    ok = ok_pin and ok_snap
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
