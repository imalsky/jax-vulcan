"""Full-integration vmap tests for `OuterLoop.run_batch` (per-step vmap is
covered by test_vmap_step.py). Pins:

  1. Homogeneous equivalence: identical lanes reproduce the single-profile
     runner result.
  2. Heterogeneous freeze-on-done: lanes that finish at different iterations
     each match their solo run (early finishers freeze, stragglers run).
  3. Non-finite isolation: a poisoned lane gets termination_reason 5 and its
     neighbours match their solo runs.
  4. Genuinely different profiles: per-profile fields ride the carry, not the
     runner closure.
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

# Termination via count_max for speed → reason 3 ("too_many"). Kept small so
# the whole batch integrates in a few seconds including JIT compile.
COUNT_MAX = 40
# Batched runs use the iteration-tick cadence for photolysis and geometry,
# solo runs the accepted-step cadence, so the two agree at the convergence
# scale, not bitwise: mixing ratios that carry signal agree to RTOL.
RTOL = 1e-4
BATCH_FLOOR = 1e-10  # trace species below this diverge chaotically, not on cadence
YMIX_FLOOR = 1e-15  # ignore ULP-noise trace species below this in the ref


def _pin_cfg():
    """Pin vulcan_cfg for a fast, photo-off batched run (the emulator regime;
    photo-on batching is covered by test_vmap_photo_batch). Mirrors
    test_outer_loop_smoke.

    Batched/emulator generation wants a deterministic, fixed diffusion scheme:
    the hybrid phase flip turns count-exhaustion into a mid-run scheme switch
    with a per-lane extended budget, which breaks the freeze-on-done
    equivalence premise, so the hybrid default is pinned off here.
    """
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
    if gs is not None:
        # Gravity is derived from Mp/Rp; back out the Mp that yields this gs
        # (using the possibly-just-overridden Rp).
        from vulcan_jax.phy_const import G_grav

        vulcan_cfg.Mp = float(gs) * float(vulcan_cfg.Rp) ** 2 / G_grav
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


def main() -> int:
    import vulcan_jax.outer_loop as outer_loop

    vulcan_cfg = _pin_cfg()
    from vulcan_jax.atm_setup import surface_gravity

    gs0 = float(surface_gravity(vulcan_cfg))
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
        rel = _max_rel_diff(out[i].ymix, ref.ymix, floor=BATCH_FLOOR)
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
        f"{max(_max_rel_diff(out[i].ymix, ref.ymix, floor=BATCH_FLOOR) for i in range(K)):.2e}; "
        f"accept_count {[int(out[i].accept_count) for i in range(K)]} vs solo "
        f"{int(ref.accept_count)}"
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
    het_rel = []
    for i, o in enumerate(offsets):
        solo = integ._runner(het_states[i], atm_static)
        rel = _max_rel_diff(het_out[i].ymix, solo.ymix, floor=BATCH_FLOOR)
        het_rel.append(rel)
        # A frozen-too-early lane would diverge from its solo run; a
        # never-frozen lane would over-integrate. Both show up here.
        if rel > RTOL or int(het_out[i].termination_reason) != 3:
            print(
                f"FAIL[heterogeneous] lane {i} (offset {o}): rel={rel:.2e} "
                f"reason={int(het_out[i].termination_reason)} "
                f"(want rel<{RTOL:.0e}, reason=3)"
            )
            ok = False
    print(f"[heterogeneous] offsets={offsets} max rel diff vs solo = {max(het_rel):.2e}")

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
        rel = _max_rel_diff(nan_out[i].ymix, ref.ymix, floor=BATCH_FLOOR)
        if rel > RTOL:
            print(f"FAIL[nan] neighbour lane {i} corrupted: rel={rel:.2e}")
            ok = False
    print(
        f"[nan] poisoned lane reason={int(nan_out[bad].termination_reason)}, "
        f"neighbours max rel diff = "
        f"{max(_max_rel_diff(nan_out[i].ymix, ref.ymix, floor=BATCH_FLOOR) for i in range(K3) if i != bad):.2e}"
    )

    # --- 4. Genuinely different profiles --------------------------------
    # Profile B differs in gravity + radius, so n_0 / Tco / geometry / Kzz all
    # differ from A. Run B solo on its own runner and as lane 1 of a batch
    # whose runner closure is baked from A: any per-profile field left in the
    # closure (instead of the ProfileVars carry) would make lane 1 diverge.
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
    relA = _max_rel_diff(het2_out[0].ymix, soloA.ymix, floor=BATCH_FLOOR)
    relB = _max_rel_diff(het2_out[1].ymix, soloB.ymix, floor=BATCH_FLOOR)
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


# --- forward mode: run_batch_jvp ------------------------------------------
# A small column (nz=40) that really converges, so the tangent certificate --
# not a count cap -- ends the run, with the refresh cadence well inside it.
NDIR = 6      # stacked tangent directions, the production width
H = 0.1       # ln Kzz step: the unit the tangent certificate reads
JVP_RTOL = 5e-2  # batched vs solo, at the convergence scale
JVP_FLOOR = 1e-10  # mixing ratio below which the primal carries no signal
JVP_EXACT = 1e-12  # one-lane vs paired tangent: XLA vectorisation roundoff
                   # (measured 3.9e-16 of the tangent's scale)


def _jvp_cfg():
    return fast_cfg(count_max=3000, use_vm_mol=False, use_hybrid_vm_mol=False,
                    nz=40, update_frq=25)


def _zero_tangent(tree, dirs=None, lanes=False):
    """Zeros on float leaves, float0 elsewhere (what `jax.jvp` hands back).
    `dirs` adds the runner's direction axis, after the lane axis if `lanes`."""
    import jax
    import jax.numpy as jnp

    def z(x):
        shape = jnp.shape(x)
        if getattr(x, "dtype", None) is not None and jnp.issubdtype(x.dtype, jnp.floating):
            if dirs is None:
                return jnp.zeros(shape)
            k = 1 if lanes else 0
            return jnp.zeros(shape[:k] + (dirs,) + shape[k:])
        return np.zeros(np.shape(x), dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(z, tree)


def _kzz_dirs(n, ndir):
    """`eye`-style tangent basis: `ndir` interleaved bands of the Kzz vector,
    H per band (d/d ln Kzz over that band)."""
    import jax.numpy as jnp

    b = np.zeros((ndir, n))
    for j in range(ndir):
        b[j, j::ndir] = H
    return jnp.asarray(b)


def _seed(state, atm, dirs):
    """(dstate, datm) along `dirs` (a (D, nz-1) basis) or, for a 1-D array,
    one direction at the primal's rank -- the shape `jax.jvp` uses."""
    stacked = dirs.ndim == 2
    ds = _zero_tangent(state, NDIR if stacked else None)
    ds = ds._replace(pv=ds.pv._replace(Kzz=dirs * state.pv.Kzz))
    da = _zero_tangent(atm, NDIR if stacked else None)._replace(Kzz=dirs * atm.Kzz)
    return ds, da


def _rel(a, b):
    """Relative difference of two arrays over `b`."""
    return np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))


def _top_pct_rel(dy, dy_ref):
    """Relative error on the top 1% of cells by |dy_ref| -> (median, max)."""
    dy, dy_ref = np.asarray(dy), np.asarray(dy_ref)
    hi = np.abs(dy_ref) >= np.quantile(np.abs(dy_ref), 0.99)
    rel = _rel(dy[hi], dy_ref[hi])
    return float(np.median(rel)), float(rel.max())


@pytest.mark.strict_isolation
def test_batched_forward_mode_runner():
    """`run_batch_jvp`: lanes independent, each lane's (primal, tangent) the
    solo `run_jvp` result at the convergence scale, the primal untouched by
    riding a zero tangent, and D stacked directions equal to D single runs.
    """
    import jax.numpy as jnp

    from vulcan_jax import outer_loop

    rs = _build_rs(_jvp_cfg())
    integ = _build_integ()
    sA, aA = integ.prepare_runstate(rs)
    # Lane B: 2x Kzz -- a different column AND a different tangent basis, so
    # a lane reading its neighbour's carry shows up.
    sB = sA._replace(pv=sA.pv._replace(Kzz=sA.pv.Kzz * 2.0))
    aB = aA._replace(Kzz=aA.Kzz * 2.0)
    prof = [(sA, aA), (sB, aB)]
    basis = _kzz_dirs(sA.pv.Kzz.shape[0], NDIR)

    solo = [integ.run_jvp(s, a, *_seed(s, a, basis)) for s, a in prof]
    for k, (f, _d, tl, ok) in enumerate(solo):
        print(f"[fwd] solo lane {k}: {int(f.accept_count)} steps, reason "
              f"{int(f.termination_reason)}, tangent_ok {bool(ok)}, "
              f"max tangent_longdy {float(np.max(np.asarray(tl))):.3g}")
        assert int(f.termination_reason) == 1 and bool(ok)

    def batched(idx, zero=False):
        sb = outer_loop.stack_integ_states([prof[i][0] for i in idx])
        ab = outer_loop.stack_atm_statics([prof[i][1] for i in idx])
        seeds = [_seed(*prof[i], basis) for i in idx]
        dsb = _zero_tangent(sb, NDIR, lanes=True)
        dab = _zero_tangent(ab, NDIR, lanes=True)
        if not zero:
            dsb = dsb._replace(pv=dsb.pv._replace(
                Kzz=jnp.stack([s.pv.Kzz for s, _ in seeds])))
            dab = dab._replace(Kzz=jnp.stack([a.Kzz for _, a in seeds]))
        return (sb, ab), integ.run_batch_jvp(sb, ab, dsb, dab)

    (sb2, ab2), (fb, db, tlb, okb) = batched([0, 1])
    print(f"[fwd] batch: steps {np.asarray(fb.accept_count).tolist()}, reasons "
          f"{np.asarray(fb.termination_reason).tolist()}, tangent_ok "
          f"{np.asarray(okb).tolist()}, dy {np.asarray(db.y).shape}")

    # (a) a lane's result does not depend on its neighbours. Exchanging the
    # two lanes runs the SAME executable and must be exact, tangent included.
    # Against the one-lane run the primal, the certificate and the reason are
    # still exact, but the tangent only agrees to roundoff: XLA vectorises 6
    # direction rows there against 12 here.
    _, (fs2, ds2, _tl2, _ok2) = batched([1, 0])
    for k in (0, 1):
        _, (f1, d1, tl1, _ok1) = batched([k])
        same = [np.array_equal(np.asarray(fb.y)[k], np.asarray(f1.y)[0]),
                np.array_equal(np.asarray(tlb)[k], np.asarray(tl1)[0]),
                int(np.asarray(fb.termination_reason)[k])
                == int(np.asarray(f1.termination_reason)[0]),
                np.array_equal(np.asarray(fb.y)[k], np.asarray(fs2.y)[1 - k]),
                np.array_equal(np.asarray(db.y)[k], np.asarray(ds2.y)[1 - k])]
        d1y = np.asarray(d1.y)[0]
        med, mx = _top_pct_rel(np.asarray(db.y)[k], d1y)
        nrm = float(np.max(np.abs(np.asarray(db.y)[k] - d1y)) / np.max(np.abs(d1y)))
        big = float(np.mean(_rel(np.asarray(db.y)[k][d1y != 0], d1y[d1y != 0]) > 1e-8))
        print(f"[fwd] (a) lane {k} alone vs in the pair: exact "
              f"y/tl/reason + exact under lane swap y/dy: {same}; tangent "
              f"top-1% median {med:.3g} max {mx:.3g}, max|diff|/max|dy| "
              f"{nrm:.3g}, cells over 1e-8 rel {big:.3g}")
        assert all(same)
        assert med < JVP_EXACT and nrm < JVP_EXACT

    # (b) each lane reproduces its solo run at the convergence scale (the
    # cadences are keyed to the tick here, to the accept count there).
    for k in (0, 1):
        fs, dsolo, _tls, _oks = solo[k]
        ymix = np.asarray(fs.ymix)
        m = ymix > JVP_FLOOR
        prim = float(np.max(_rel(np.asarray(fb.ymix)[k][m], ymix[m])))
        med, mx = _top_pct_rel(np.asarray(db.y)[k], np.asarray(dsolo.y))
        print(f"[fwd] (b) lane {k} vs solo: reason "
              f"{int(np.asarray(fb.termination_reason)[k])}/"
              f"{int(fs.termination_reason)}, primal max rel {prim:.3g} over "
              f"{int(m.sum())} cells, tangent top-1% median {med:.3g} max {mx:.3g}")
        assert int(np.asarray(fb.termination_reason)[k]) == int(fs.termination_reason)
        assert prim < JVP_RTOL and med < JVP_RTOL

    # (c) riding a (zero) tangent must not move the batched primal.
    _, (fz, _dz, _tlz, _okz) = batched([0, 1], zero=True)
    prim_b = integ.run_batch(sb2, ab2)
    exact = np.array_equal(np.asarray(fz.y), np.asarray(prim_b.y))
    print(f"[fwd] (c) zero-seed jvp primal vs run_batch: bit-identical={exact}"
          + ("" if exact else
             f", max rel {float(np.max(_rel(fz.y, prim_b.y))):.3g}"))
    assert exact

    # (d) D stacked directions == D single-direction runs (they stop at the
    # last direction's certificate, so agreement is at the tangent scale),
    # and the primal is the plain runner's, integrated once.
    fD, dD, _tlD, _okD = solo[0]
    for j in range(NDIR):
        _f1, d1, _tl1, _ok1 = integ.run_jvp(sA, aA, *_seed(sA, aA, basis[j]))
        med, mx = _top_pct_rel(np.asarray(dD.y)[j], np.asarray(d1.y))
        print(f"[fwd] (d) direction {j}: steps {int(_f1.accept_count)} vs "
              f"{int(fD.accept_count)} stacked, top-1% median {med:.3g} max {mx:.3g}")
        assert med < JVP_RTOL
    prim_solo = integ._runner(sA, aA)
    same_primal = np.array_equal(np.asarray(fD.y), np.asarray(prim_solo.y))
    print(f"[fwd] (d) stacked-jvp primal vs plain runner: bit-identical="
          f"{same_primal}, steps {int(fD.accept_count)} vs "
          f"{int(prim_solo.accept_count)}"
          + ("" if same_primal else
             f", max rel {float(np.max(_rel(fD.y, prim_solo.y))):.3g}"))
    assert same_primal


if __name__ == "__main__":
    sys.exit(main())
