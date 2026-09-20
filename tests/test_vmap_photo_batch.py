"""Batched (vmap-across-profiles) photochemistry: `run_batch` with photo on.

The only T-P-dependent pieces of the photo statics are the two T-interpolated
cross-section stacks (`absp_T_cross`, `cross_J_T`); they ride `ProfileVars`
per lane while everything else (star, wavelength grid, branch maps, cfg
scalars) stays closure-baked and batch-constant. Three properties:

  1. SOLO/BATCH EQUIVALENCE — profile B, run solo on its OWN runner (closure
     baked from B), matches lane 1 of a batch whose runner is baked from A.
     If any per-profile photo field were left in the closure, lane 1 would
     silently use A's cross sections while soloB uses B's → divergence.
  2. VACUITY GUARDS — the two lanes' T-dep cross sections genuinely differ
     (different Tiso → different interpolation), the photo branch genuinely
     fired (nonzero actinic flux, photo k-rows populated), and the two
     profiles' results differ.
  3. SAME-STAR GUARD — `prepare_runstate` rejects a profile whose TOA
     stellar flux differs from the first profile's (only T-P may vary
     across a photo batch).

Fast-ish: isothermal atmosphere (no atm file), const_mix init (no EQ seed),
small nz / count_max, photo cadence lowered so the branch fires repeatedly.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from _helpers import fast_cfg

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

COUNT_MAX = 30
# Batched runs evaluate photolysis on the iteration tick (accepted states
# only), solo runs on the accepted-step count, so the stored RT arrays are a
# different iterate of the same fixed point: they are judged by SCALE against
# their own peak, at RTOL. The chemistry agrees at the convergence scale and
# the vacuity checks below catch a closure leak.
RTOL = 5e-2
YMIX_FLOOR = 1e-15
# ymix is judged by SCALE, not bit identity: |lane - its own solo| against
# |soloA - soloB|. A closure-baked per-profile photo field makes lane 1 BE
# soloA (ratio ~1); bit identity is a property of the XLA version, not of the
# code (holds through jax 0.9.2, breaks from 0.10 by a 6.4e-7 step-controller
# time offset; measured ratio 6.5e-7 at 0.11.1, ~1e-22 at 0.9.2; notes 1.8).
SCALE_TOL = 1e-4


def _pin_cfg():
    """Pin vulcan_cfg for a small photo-ON batched run: isothermal T-P,
    const_mix init, lowered photo cadence so the branch fires within
    COUNT_MAX. Mirrors test_vmap_while_loop._pin_cfg.

    `ini_mix="const_mix"` avoids the EQ seed. The fixed diffusion scheme is
    deterministic for the batched/emulator regime (the hybrid default would
    flip schemes mid-run per lane). `T_cross_sp=["H2O"]` selects the vendored
    T-dependent 423K-2360K tables: the two Tiso values interpolate to
    different per-layer cross sections, which is exactly the per-lane data
    this test must prove rides the carry.
    """
    return fast_cfg(
        count_max=COUNT_MAX,
        use_photo=True,
        ini_mix="const_mix",
        use_vm_mol=False,
        use_hybrid_vm_mol=False,
        nz=40,
        ini_update_photo_frq=5,
        T_cross_sp=["H2O"],
    )
def _build_rs(vulcan_cfg, *, Tiso):
    from vulcan_jax.state import RunState

    vulcan_cfg.Tiso = float(Tiso)
    return RunState.with_pre_loop_setup(vulcan_cfg)


def _build_integ():
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax

    return outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())


def _max_rel_diff(b, r, floor=YMIX_FLOOR):
    b = np.asarray(b, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    mask = np.abs(r) > floor
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(b[mask] - r[mask]) / np.abs(r[mask])))


def _peak_ratio(bat, sol):
    """Max absolute difference from the lane's own solo, against the solo peak."""
    bat, sol = (np.asarray(x, dtype=np.float64) for x in (bat, sol))
    return float(np.max(np.abs(bat - sol))) / float(np.max(np.abs(sol)))


def _scale_ratio(bat, sol, other):
    bat, sol, other = (np.asarray(x, dtype=np.float64) for x in (bat, sol, other))
    return float(np.max(np.abs(bat - sol))) / float(np.max(np.abs(sol - other)))


def main() -> int:
    import vulcan_jax.outer_loop as outer_loop

    vulcan_cfg = _pin_cfg()
    ok = True

    rsA = _build_rs(vulcan_cfg, Tiso=900.0)
    integA = _build_integ()
    sA, atmA = integA.prepare_runstate(rsA)

    rsB = _build_rs(vulcan_cfg, Tiso=1600.0)
    # Route B through integA too (the batch-driver pattern; exercises the
    # same-star guard's accept path) AND through its own integB for the solo.
    sB, atmB = integA.prepare_runstate(rsB)
    integB = _build_integ()
    sB_own, atmB_own = integB.prepare_runstate(rsB)

    # --- vacuity: per-lane T-dep cross sections differ ------------------
    tdiff = _max_rel_diff(sB.pv.p_absp_T_cross, sA.pv.p_absp_T_cross, floor=0.0)
    jdiff = _max_rel_diff(sB.pv.p_cross_J_T, sA.pv.p_cross_J_T, floor=0.0)
    if sA.pv.p_absp_T_cross.shape[0] == 0 or tdiff == 0.0 or jdiff == 0.0:
        print(
            f"FAIL[vacuity] T-dep cross sections do not differ between lanes "
            f"(n_absp_T={sA.pv.p_absp_T_cross.shape[0]}, tdiff={tdiff:.2e}, "
            f"jdiff={jdiff:.2e})"
        )
        ok = False

    # --- solo runs --------------------------------------------------------
    soloA = integA._runner(sA, atmA)
    soloB = integB._runner(sB_own, atmB_own)

    # photo really fired: actinic flux nonzero, photo branch wrote k rows.
    if not (np.any(np.asarray(soloA.aflux) > 0) and float(soloA.aflux_change) != 0.0):
        print("FAIL[vacuity] photo branch never fired in soloA")
        ok = False

    # --- batch on A's runner ----------------------------------------------
    batched = integA.run_batch(
        outer_loop.stack_integ_states([sA, sB]),
        outer_loop.stack_atm_statics([atmA, atmB]),
    )
    out = outer_loop.unstack_integ_states(batched, 2)

    checks = ["ymix", "k_arr", "aflux", "tau"]
    for name in checks:
        bat0, bat1 = getattr(out[0], name), getattr(out[1], name)
        refA, refB = getattr(soloA, name), getattr(soloB, name)
        if name == "ymix":
            relA = _scale_ratio(bat0, refA, refB)
            relB = _scale_ratio(bat1, refB, refA)
            tol = SCALE_TOL
        else:
            relA = _peak_ratio(bat0, refA)
            relB = _peak_ratio(bat1, refB)
            tol = RTOL
        if relA > tol or relB > tol:
            print(
                f"FAIL[solo-vs-batch] {name}: laneA rel={relA:.2e} laneB rel={relB:.2e}"
            )
            ok = False
        else:
            print(f"[solo-vs-batch] {name}: laneA rel={relA:.2e} laneB rel={relB:.2e}")

    profiles_differ = _max_rel_diff(soloB.ymix, soloA.ymix)
    if profiles_differ < 1e-6:
        print(f"FAIL[vacuity] profiles identical ({profiles_differ:.2e})")
        ok = False
    print(f"[vacuity] profiles differ by {profiles_differ:.2e}")

    # --- same-star guard ----------------------------------------------------
    rsC = rsB._replace(photo=rsB.photo._replace(sflux_top=rsB.photo.sflux_top * 2.0))
    try:
        integA.prepare_runstate(rsC)
        print("FAIL[guard] mismatched-star profile was accepted")
        ok = False
    except ValueError as e:
        print(f"[guard] mismatched star rejected: {str(e)[:60]}...")

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.strict_isolation
def test_main():
    assert main() == 0


@pytest.mark.strict_isolation
def test_queue_refill_starts_on_its_own_photolysis():
    """`run_queue` on ONE lane: the second profile is refilled into a lane the
    first profile just left, and must start on its own photolysis fields the
    way a plain-batch lane does at tick 0. Run one at a time it reaches the
    same termination reason and the same mixing ratios as the pair run in a
    single `run_batch`; a refilled lane left on its predecessor's RT state
    would integrate a different column.
    """
    import vulcan_jax.outer_loop as outer_loop

    vulcan_cfg = _pin_cfg()
    integ = _build_integ()
    sA, atmA = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=900.0))
    sB, atmB = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=1600.0))
    init_b = outer_loop.stack_integ_states([sA, sB])
    atm_b = outer_loop.stack_atm_statics([atmA, atmB])

    ref = integ.run_batch(init_b, atm_b)
    (y, reason), n_iter = integ.run_queue(
        lambda job: job,
        (init_b, atm_b),
        n_lanes=1,
        out_fn=lambda f: (f.y, f.termination_reason),
        chunk=1,
    )
    assert np.array_equal(np.asarray(reason), np.asarray(ref.termination_reason))
    for k in (0, 1):
        yk, yr = np.asarray(y[k]), np.asarray(ref.y[k])
        rel = _max_rel_diff(
            yk / yk.sum(axis=1, keepdims=True), yr / yr.sum(axis=1, keepdims=True)
        )
        print(
            f"[queue-refill] job {k}: ymix max rel vs run_batch {rel:.3e} "
            f"(n_iter {int(n_iter)}, reason {int(reason[k])})"
        )
        assert rel < RTOL


@pytest.mark.strict_isolation
def test_queue_without_refill_is_the_photo_batch():
    """`run_queue` with one lane per job, PHOTOLYSIS ON: nothing is refilled,
    so every job runs the ticks `run_batch` gives it and the result must be
    bitwise `run_batch`'s. The photo-off twin of this pin is
    test_run_queue.test_queue_without_refill_is_the_batch; only with photo on
    can the initial fill see the photo branch twice before its first
    chemistry step (which moves prev_aflux and aflux_change, and aflux_change
    gates the convergence certificate).
    """
    from vulcan_jax import outer_loop

    vulcan_cfg = _pin_cfg()
    integ = _build_integ()
    sA, atmA = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=900.0))
    sB, atmB = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=1600.0))
    init_b = outer_loop.stack_integ_states([sA, sB])
    atm_b = outer_loop.stack_atm_statics([atmA, atmB])

    ref = integ.run_batch(init_b, atm_b)
    (y, t, acc, reason), n_iter = integ.run_queue(
        lambda job: job,
        (init_b, atm_b),
        n_lanes=2,
        out_fn=lambda f: (f.y, f.t, f.accept_count, f.termination_reason),
    )
    print(f"[photo no refill] n_iter={int(n_iter)}", flush=True)
    for name, a, b in (
        ("y", y, ref.y),
        ("t", t, ref.t),
        ("accept_count", acc, ref.accept_count),
        ("termination_reason", reason, ref.termination_reason),
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b)), name


if __name__ == "__main__":
    sys.exit(main())
