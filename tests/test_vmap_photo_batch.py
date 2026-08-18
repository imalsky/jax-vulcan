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

Fast-ish: isothermal atmosphere (no atm file), const_mix init (no FastChem),
small nz / count_max, photo cadence lowered so the branch fires repeatedly.
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

COUNT_MAX = 30
RTOL = 1e-9
YMIX_FLOOR = 1e-15


def _pin_cfg():
    """Pin vulcan_cfg for a small photo-ON batched run: isothermal T-P,
    const_mix init, lowered photo cadence so the branch fires within
    COUNT_MAX. Mirrors test_vmap_while_loop._pin_cfg."""
    import vulcan_jax.legacy_io as op

    vulcan_cfg = op.default_config()

    vulcan_cfg.count_max = COUNT_MAX
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False
    vulcan_cfg.use_photo = True
    vulcan_cfg.use_ion = False
    vulcan_cfg.ini_mix = "const_mix"  # no FastChem; default const_mix dict
    vulcan_cfg.atm_type = "isothermal"
    vulcan_cfg.Kzz_prof = "Pfunc"
    # Deterministic fixed diffusion scheme for the batched/emulator regime
    # (the hybrid default would flip schemes mid-run per lane).
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    vulcan_cfg.nz = 40
    # Photo fires every 5 accepted steps -> several firings inside COUNT_MAX.
    vulcan_cfg.ini_update_photo_frq = 5
    # T-dependent H2O cross sections (vendored 423K-2360K tables): the two
    # Tiso values interpolate to different per-layer cross sections, which is
    # exactly the per-lane data this test must prove rides the carry.
    vulcan_cfg.T_cross_sp = ["H2O"]
    return vulcan_cfg


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

    checks = [
        ("ymix", lambda s: s.ymix, YMIX_FLOOR),
        ("k_arr", lambda s: s.k_arr, 0.0),
        ("aflux", lambda s: s.aflux, 0.0),
        ("tau", lambda s: s.tau, 0.0),
    ]
    for name, get, floor in checks:
        relA = _max_rel_diff(get(out[0]), get(soloA), floor=floor)
        relB = _max_rel_diff(get(out[1]), get(soloB), floor=floor)
        if relA > RTOL or relB > RTOL:
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


if __name__ == "__main__":
    sys.exit(main())
