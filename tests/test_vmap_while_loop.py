"""vmap-the-whole-integration tests for `OuterLoop.run_batch`.

The per-step kernel was already vmap-tested (`test_vmap_step.py`); these
exercise the full adaptive `lax.while_loop` under `jax.vmap` with the
freeze-on-done machinery, on CPU (vmap is backend-agnostic, so this is the
pre-GPU gate). Three properties:

  1. HOMOGENEOUS EQUIVALENCE — a batch of identical profiles reproduces the
     single-profile `runner` result to ~1e-12 (the CLAUDE.md vmap bar).
  2. HETEROGENEOUS FREEZE-ON-DONE — lanes that terminate at *different*
     iterations each match their solo run. Driven by per-lane starting
     `accept_count` offsets so lanes hit `count_max` at different absolute
     steps; the early finishers must freeze (not drift) while stragglers run.
  3. NON-FINITE ISOLATION — a lane seeded to go non-finite is flagged
     (`termination_reason == 5`) and frozen, and its batch-neighbours are
     numerically identical to their solo runs (vmap lanes are independent).

Fast: one pre-loop setup, small `count_max`, replicated init states.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

# Termination via count_max for speed → reason 3 ("too_many"). Kept small so
# the whole batch integrates in a few seconds including JIT compile.
COUNT_MAX = 40
# vmap-vs-single agreement bar. XLA may fuse the vmapped kernel differently
# from the single-profile kernel, so allow a hair above machine precision on
# the mixing ratios that actually carry signal.
RTOL = 1e-9
YMIX_FLOOR = 1e-15  # ignore ULP-noise trace species below this in the ref


def _pin_cfg():
    """Pin vulcan_cfg for a fast, photo-off batched run (the emulator regime;
    run_batch rejects photochemistry). Mirrors test_outer_loop_smoke."""
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op

    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_jax.vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg

    vulcan_cfg.count_max = COUNT_MAX
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False
    vulcan_cfg.use_photo = False
    vulcan_cfg.use_ion = False
    # Isothermal T-P so `Tiso` directly sets the atmosphere; different Tiso
    # gives genuinely different n_0 / Tco / equilibrium abundances (the
    # heterogeneous-batch probe). 'file' mode would ignore our overrides.
    # Pfunc Kzz is pressure-analytic, so it needs no atm file (isothermal
    # mode does not load one).
    vulcan_cfg.atm_type = "isothermal"
    vulcan_cfg.Kzz_prof = "Pfunc"
    return vulcan_cfg


def _build_rs(vulcan_cfg, *, Tiso=None, gs=None, Rp=None):
    """Build a RunState, optionally overriding temperature / gravity / radius
    so two calls produce genuinely different atmospheres."""
    from vulcan_jax.state import RunState

    if Tiso is not None:
        vulcan_cfg.Tiso = float(Tiso)
    if gs is not None:
        vulcan_cfg.gs = float(gs)
    if Rp is not None:
        vulcan_cfg.Rp = float(Rp)
    return RunState.with_pre_loop_setup(vulcan_cfg)


def _build_integ():
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax

    return outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())


def _max_rel_diff(batched_ymix, ref_ymix):
    """Max relative difference over cells where the reference carries signal."""
    b = np.asarray(batched_ymix, dtype=np.float64)
    r = np.asarray(ref_ymix, dtype=np.float64)
    mask = np.abs(r) > YMIX_FLOOR
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(b[mask] - r[mask]) / np.abs(r[mask])))


def main() -> int:
    import vulcan_jax.outer_loop as outer_loop

    vulcan_cfg = _pin_cfg()
    gs0 = float(vulcan_cfg.gs)
    Rp0 = float(vulcan_cfg.Rp)
    rs = _build_rs(vulcan_cfg, Tiso=900.0, gs=gs0, Rp=Rp0)
    integ = _build_integ()

    # Build one single-profile (init_state, atm_static); the runner is
    # compiled here for this nz/toggle-combo.
    init_state, atm_static = integ.prepare_runstate(rs)
    ok = True

    # --- 1. Homogeneous equivalence -------------------------------------
    K = 4
    states = [init_state] * K
    atms = [atm_static] * K
    batched = integ.run_batch(
        outer_loop.stack_integ_states(states),
        outer_loop.stack_atm_statics(atms),
    )
    out = outer_loop.unstack_integ_states(batched, K)
    ref = integ._runner(init_state, atm_static)  # single-profile ground truth

    for i in range(K):
        rel = _max_rel_diff(out[i].ymix, ref.ymix)
        reason = int(out[i].termination_reason)
        done = bool(out[i].is_done)
        if rel > RTOL or not done or reason != 3:
            print(
                f"FAIL[homogeneous] lane {i}: rel={rel:.2e} done={done} "
                f"reason={reason} (want rel<{RTOL:.0e}, done, reason=3)"
            )
            ok = False
    print(
        f"[homogeneous] K={K} max rel diff vs single = "
        f"{max(_max_rel_diff(out[i].ymix, ref.ymix) for i in range(K)):.2e}"
    )

    # --- 2. Heterogeneous freeze-on-done --------------------------------
    # Per-lane starting accept_count offsets: lanes hit count_max at different
    # absolute iterations, so early finishers must freeze while others run.
    offsets = [0, 10, 20, 30]
    het_states = [init_state._replace(accept_count=np.int32(o)) for o in offsets]
    het_batched = integ.run_batch(
        outer_loop.stack_integ_states(het_states),
        outer_loop.stack_atm_statics([atm_static] * len(offsets)),
    )
    het_out = outer_loop.unstack_integ_states(het_batched, len(offsets))
    for i, o in enumerate(offsets):
        solo = integ._runner(het_states[i], atm_static)
        rel = _max_rel_diff(het_out[i].ymix, solo.ymix)
        # A frozen-too-early lane would diverge from its solo run; a
        # never-frozen lane would over-integrate. Both show up here.
        if rel > RTOL or int(het_out[i].termination_reason) != 3:
            print(
                f"FAIL[heterogeneous] lane {i} (offset {o}): rel={rel:.2e} "
                f"reason={int(het_out[i].termination_reason)} (want rel<{RTOL:.0e}, reason=3)"
            )
            ok = False
    print(
        f"[heterogeneous] offsets={offsets} max rel diff vs solo = "
        f"{max(_max_rel_diff(het_out[i].ymix, integ._runner(het_states[i], atm_static).ymix) for i in range(len(offsets))):.2e}"
    )

    # --- 3. Non-finite isolation ----------------------------------------
    import jax.numpy as jnp

    bad = 1
    K3 = 3
    nan_states = [init_state, init_state, init_state]
    # Poison both y AND y_prev: a rejected Ros2 step reverts y to y_prev, so
    # poisoning y alone would self-heal and the lane would never go non-finite.
    poisoned_y = init_state.y.at[0, 0].set(jnp.nan)
    poisoned_yprev = init_state.y_prev.at[0, 0].set(jnp.nan)
    nan_states[bad] = init_state._replace(y=poisoned_y, y_prev=poisoned_yprev)
    nan_batched = integ.run_batch(
        outer_loop.stack_integ_states(nan_states),
        outer_loop.stack_atm_statics([atm_static] * K3),
    )
    nan_out = outer_loop.unstack_integ_states(nan_batched, K3)
    if int(nan_out[bad].termination_reason) != 5:
        print(
            f"FAIL[nan] poisoned lane reason={int(nan_out[bad].termination_reason)} (want 5)"
        )
        ok = False
    for i in range(K3):
        if i == bad:
            continue
        rel = _max_rel_diff(nan_out[i].ymix, ref.ymix)
        if rel > RTOL:
            print(f"FAIL[nan] neighbour lane {i} corrupted: rel={rel:.2e}")
            ok = False
    print(
        f"[nan] poisoned lane reason={int(nan_out[bad].termination_reason)}, "
        f"neighbours max rel diff = "
        f"{max(_max_rel_diff(nan_out[i].ymix, ref.ymix) for i in range(K3) if i != bad):.2e}"
    )

    # --- 4. Genuinely different profiles --------------------------------
    # Build a SECOND atmosphere with different surface gravity + planet
    # radius. Gravity feeds the analytical T-P, so n_0 / Tco / geometry / Kzz
    # all differ from profile A. Run B solo on its OWN runner (closure baked
    # from B) and as lane 1 of a batch whose runner is baked from A. If any
    # per-profile field were left in the closure instead of the ProfileVars
    # carry, lane 1 would silently use A's value while soloB uses B's →
    # divergence. Agreement proves every per-profile field is carried.
    rsB = _build_rs(vulcan_cfg, Tiso=1600.0, gs=gs0 * 1.6, Rp=Rp0 * 0.8)
    integB = _build_integ()
    initB, atmB = integB.prepare_runstate(rsB)
    soloB = integB._runner(initB, atmB)
    soloA = integ._runner(init_state, atm_static)
    het2 = integ.run_batch(
        outer_loop.stack_integ_states([init_state, initB]),
        outer_loop.stack_atm_statics([atm_static, atmB]),
    )
    het2_out = outer_loop.unstack_integ_states(het2, 2)
    relA = _max_rel_diff(het2_out[0].ymix, soloA.ymix)
    relB = _max_rel_diff(het2_out[1].ymix, soloB.ymix)
    # Sanity: the two profiles must actually differ, else the test is vacuous.
    profiles_differ = _max_rel_diff(soloB.ymix, soloA.ymix)
    if relA > RTOL or relB > RTOL or profiles_differ < 1e-6:
        print(
            f"FAIL[diff-profiles] laneA rel={relA:.2e} laneB rel={relB:.2e} "
            f"profiles_differ={profiles_differ:.2e} (want laneA/B < {RTOL:.0e}, "
            "profiles_differ > 1e-6)"
        )
        ok = False
    print(
        f"[diff-profiles] A vs B differ by {profiles_differ:.2e}; "
        f"batched laneA rel={relA:.2e}, laneB rel={relB:.2e}"
    )

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.strict_isolation
def test_main():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
