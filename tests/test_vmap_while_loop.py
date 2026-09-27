"""Full-integration vmap tests for `OuterLoop.run_batch` (per-step vmap is
covered by test_vmap_step.py). Pins:

  1. Heterogeneous freeze-on-done: lanes that finish at different iterations
     each match their solo run (early finishers freeze, stragglers run).
  2. Non-finite isolation: a poisoned lane gets termination_reason 5 and its
     neighbours match their solo runs.
  3. Genuinely different profiles: per-profile fields ride the carry, not the
     runner closure.

Every batch is K = 4 lanes wide: each new width compiles the batched runner
again (~30 s on one core).
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import fast_cfg, mass_for_gravity

# Termination via count_max for speed -> reason 3 ("too_many"). Kept small so
# the whole batch integrates in a few seconds including JIT compile.
COUNT_MAX = 40
K = 4  # lanes in every batch below
# Batched runs use the iteration-tick cadence for photolysis and geometry,
# solo runs the accepted-step cadence, so the two agree at the convergence
# scale, not bitwise: mixing ratios that carry signal agree to RTOL.
RTOL = 1e-4
BATCH_FLOOR = 1e-10  # trace species below this diverge chaotically, not on cadence
YMIX_FLOOR = 1e-15  # ignore ULP-noise trace species below this in the ref


def _pin_cfg():
    """Fast photo-off batched config (photo-on: test_vmap_photo_batch). Hybrid
    vm_mol is off: its per-lane scheme switch on count exhaustion breaks
    freeze-on-done equivalence."""
    return fast_cfg(
        count_max=COUNT_MAX,
        use_vm_mol=False,
        use_hybrid_vm_mol=False,
    )
def _build_rs(vulcan_cfg, *, Tiso=None, gs=None, Rp=None):
    """Build a RunState, optionally overriding temperature / gravity / radius
    so two calls produce genuinely different atmospheres."""
    from vulcan_jax.state import RunState

    if Tiso is not None:
        vulcan_cfg.Tiso = float(Tiso)
    if Rp is not None:
        vulcan_cfg.Rp = float(Rp)
    if gs is not None:  # gravity derives from Mp and the Rp just set
        vulcan_cfg.Mp = mass_for_gravity(float(gs), float(vulcan_cfg.Rp))
    return RunState.with_pre_loop_setup(vulcan_cfg)


def _build_integ():
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax

    return outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())


def _max_rel_diff(batched_ymix, ref_ymix, floor=YMIX_FLOOR):
    """Max relative difference over cells where the reference carries signal."""
    b = np.asarray(batched_ymix, dtype=np.float64)
    r = np.asarray(ref_ymix, dtype=np.float64)
    mask = np.abs(r) > floor
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(b[mask] - r[mask]) / np.abs(r[mask])))


@pytest.mark.strict_isolation
def test_run_batch_matches_solo_runs():
    import jax.numpy as jnp
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.atm_setup import surface_gravity

    vulcan_cfg = _pin_cfg()
    gs0 = float(surface_gravity(vulcan_cfg))
    Rp0 = float(vulcan_cfg.Rp)
    rs = _build_rs(vulcan_cfg, Tiso=900.0, gs=gs0, Rp=Rp0)
    integ = _build_integ()

    # Build one single-profile (init_state, atm_static); the runner is
    # compiled here for this nz/toggle-combo.
    init_state, atm_static = integ.prepare_runstate(rs)
    ref = integ._runner(init_state, atm_static)  # single-profile ground truth

    def run_batch(states, atms):
        return outer_loop.unstack_integ_states(
            integ.run_batch(
                outer_loop.stack_integ_states(states),
                outer_loop.stack_atm_statics(atms),
            ),
            K,
        )

    # --- 1. Heterogeneous freeze-on-done --------------------------------
    # Per-lane starting accept_count offsets: lanes hit count_max at different
    # absolute iterations, so early finishers must freeze while others run.
    offsets = [0, 10, 20, 30]  # K lanes
    het_states = [init_state._replace(accept_count=np.int32(o)) for o in offsets]
    het_out = run_batch(het_states, [atm_static] * K)
    for i, o in enumerate(offsets):
        solo = integ._runner(het_states[i], atm_static)
        rel = _max_rel_diff(het_out[i].ymix, solo.ymix, floor=BATCH_FLOOR)
        reason = int(het_out[i].termination_reason)
        # A frozen-too-early lane would diverge from its solo run; a
        # never-frozen lane would over-integrate. Both show up here.
        assert rel <= RTOL and reason == 3, ("heterogeneous", i, o, rel, reason)

    # --- 2. Non-finite isolation ----------------------------------------
    bad = 1
    nan_states = [init_state] * K
    # Poison both y AND y_prev: a rejected Ros2 step reverts y to y_prev, so
    # poisoning y alone would self-heal and the lane would never go non-finite.
    poisoned_y = init_state.y.at[0, 0].set(jnp.nan)
    poisoned_yprev = init_state.y_prev.at[0, 0].set(jnp.nan)
    nan_states[bad] = init_state._replace(y=poisoned_y, y_prev=poisoned_yprev)
    nan_out = run_batch(nan_states, [atm_static] * K)
    assert int(nan_out[bad].termination_reason) == 5, (
        "poisoned lane reason", int(nan_out[bad].termination_reason))
    for i in range(K):
        if i == bad:
            continue
        rel = _max_rel_diff(nan_out[i].ymix, ref.ymix, floor=BATCH_FLOOR)
        assert rel <= RTOL, ("nan neighbour lane corrupted", i, rel)

    # --- 3. Genuinely different profiles --------------------------------
    # Profile B differs in gravity + radius, so n_0 / Tco / geometry / Kzz all
    # differ from A. Run B solo on its own runner and as lane 1 of a batch
    # whose runner closure is baked from A: any per-profile field left in the
    # closure (instead of the ProfileVars carry) would make lane 1 diverge.
    rsB = _build_rs(vulcan_cfg, Tiso=1600.0, gs=gs0 * 1.6, Rp=Rp0 * 0.8)
    integB = _build_integ()
    initB, atmB = integB.prepare_runstate(rsB)
    soloB = integB._runner(initB, atmB)
    soloA = ref
    het2_out = run_batch(
        [init_state, initB] * (K // 2), [atm_static, atmB] * (K // 2)
    )
    relA = max(_max_rel_diff(o.ymix, soloA.ymix, floor=BATCH_FLOOR) for o in het2_out[0::2])
    relB = max(_max_rel_diff(o.ymix, soloB.ymix, floor=BATCH_FLOOR) for o in het2_out[1::2])
    # Sanity: the two profiles must actually differ, else the test is vacuous.
    profiles_differ = _max_rel_diff(soloB.ymix, soloA.ymix)
    assert relA <= RTOL and relB <= RTOL and profiles_differ >= 1e-6, (
        "diff-profiles", relA, relB, profiles_differ)
