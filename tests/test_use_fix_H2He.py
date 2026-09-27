"""use_fix_H2He bottom pin (op.py:2935-2941).

Master snapshots the bottom H2/He mixing ratios at the first step with
t > 1e6 and pins them through use_fix_sp_bot; the runner does this with a
one-shot h2he_pinned carry. Seeded at t = 1.5e6, a 5-step HD189 run must
snapshot the pre-run ratios and hold y[0] at ymix_pre * n_0[0] to 1e-6.
The pin is not exact because upstream pins sol[0] before the clip and the
y = n_0 * ymix rebalance (op.py:2945-2946).
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import set_cfg


def test_use_fix_h2he_pins_the_bottom_ratios():
    import vulcan_jax.chem_funs as chem_funs

    species = list(chem_funs.spec_list)
    if "H2" not in species or "He" not in species:
        pytest.skip("use_fix_H2He needs H2 and He in the network")

    cfg = set_cfg(use_fix_H2He=True, count_max=5, count_min=1, use_print_prog=False)

    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.state import RunState

    rs = RunState.with_pre_loop_setup(cfg)
    # Seed t past 1e6 so the trip fires on the first accepted step.
    rs = rs._replace(step=rs.step._replace(t=1.5e6))
    idx = [species.index("H2"), species.index("He")]
    mix_pre = np.asarray(rs.step.ymix)[0, idx]
    target = mix_pre * float(rs.atm.n_0[0])

    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())
    state, atm_static = integ.prepare_runstate(rs)  # builds integ._runner
    final = integ._runner(state, atm_static)

    np.testing.assert_allclose(np.asarray(final.y)[0, idx], target, rtol=1e-6)
    # The one-shot snapshot fired and holds the pre-run mixing ratios.
    assert bool(final.h2he_pinned)
    assert np.array_equal(np.asarray(final.h2he_mix), mix_pre)
