"""Batched photochemistry: `run_batch` with photolysis on.

The T-interpolated cross sections (`absp_T_cross`, `cross_J_T`) ride
`ProfileVars` per lane; the rest of the photo statics is closure-baked.
Pins: (1) profile B solo on its own runner matches lane 1 of a batch baked
from A, so no per-profile photo field stays in the closure; (2) vacuity: the
lanes' cross sections, photo firing and results differ; (3)
`prepare_runstate` rejects a different TOA stellar flux.
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
# ymix is judged by scale, |lane - own solo| / |soloA - soloB|: a
# closure-baked photo field makes lane 1 equal soloA (ratio ~1). Bit
# identity depends on the XLA version (notes §1.8).
SCALE_TOL = 1e-4


def _pin_cfg(**extra):
    """Small photo-on batched config: isothermal, const_mix init, fixed
    diffusion scheme (the hybrid flips schemes per lane), photo cadence low
    enough to fire within COUNT_MAX. `T_cross_sp=["H2O"]` gives the two Tiso
    lanes different cross sections."""
    return fast_cfg(**{
        "count_max": COUNT_MAX,
        "use_photo": True,
        "ini_mix": "const_mix",
        "use_vm_mol": False,
        "use_hybrid_vm_mol": False,
        "nz": 40,
        "ini_update_photo_frq": 5,
        "T_cross_sp": ["H2O"],
        **extra,
    })
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

    profiles_differ = _max_rel_diff(soloB.ymix, soloA.ymix)
    if profiles_differ < 1e-6:
        print(f"FAIL[vacuity] profiles identical ({profiles_differ:.2e})")
        ok = False

    # --- same-star guard ----------------------------------------------------
    rsC = rsB._replace(photo=rsB.photo._replace(sflux_top=rsB.photo.sflux_top * 2.0))
    try:
        integA.prepare_runstate(rsC)
        print("FAIL[guard] mismatched-star profile was accepted")
        ok = False
    except ValueError:
        pass

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
        assert rel < RTOL, (k, rel)


@pytest.mark.strict_isolation
def test_queue_without_refill_is_the_photo_batch():
    """`run_queue` with one lane per job and photolysis on is bitwise
    `run_batch`. With photo on, the initial fill can hit the photo branch twice
    before the first chemistry step and move aflux_change, which gates the
    certificate. Photo-off twin: test_run_queue."""
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
    for name, a, b in (
        ("y", y, ref.y),
        ("t", t, ref.t),
        ("accept_count", acc, ref.accept_count),
        ("termination_reason", reason, ref.termination_reason),
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b)), name


@pytest.mark.strict_isolation
def test_refill_on_a_photo_tick_applies_photolysis_once():
    """A lane refilled at a tick on its photolysis cadence takes photolysis
    once, from its first step's own gate, as a plain-batch lane does at tick
    0. With the photo and geometry cadences at 1 every tick is on cadence, so
    the refilled job sees the ticks it would see alone and the two results
    are bitwise equal. A second application at the refill moves prev_aflux
    and aflux_change, and aflux_change gates the convergence certificate.
    """
    from vulcan_jax import outer_loop

    vulcan_cfg = _pin_cfg(ini_update_photo_frq=1, final_update_photo_frq=1,
                          update_frq=1)
    integ = _build_integ()
    sA, atmA = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=900.0))
    sB, atmB = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=1600.0))
    fields = ("y", "t", "accept_count", "termination_reason", "aflux_change")

    def out_fn(f):
        return tuple(getattr(f, name) for name in fields)

    pair, _ = integ.run_queue(
        lambda job: job,
        (outer_loop.stack_integ_states([sA, sB]),
         outer_loop.stack_atm_statics([atmA, atmB])),
        n_lanes=1, out_fn=out_fn, chunk=1,
    )
    alone, _ = integ.run_queue(
        lambda job: job,
        (outer_loop.stack_integ_states([sB]), outer_loop.stack_atm_statics([atmB])),
        n_lanes=1, out_fn=out_fn,
    )
    for name, refilled, own in zip(fields, pair, alone):
        assert np.array_equal(np.asarray(refilled[1]), np.asarray(own[0])), name


if __name__ == "__main__":
    sys.exit(main())
