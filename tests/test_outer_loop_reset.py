"""Test OuterLoop.reset() invalidates the cached runner.

Use case: a notebook builds an OuterLoop, runs once, mutates a vulcan_cfg
field, and runs again. Without reset(), the second run uses the stale
config (the runner closure captured the original values). With reset(),
the next run rebuilds the runner against the fresh config.

Robustness note: `tests/test_chem.py` pops `vulcan_cfg` from `sys.modules`
mid-run, which causes Python to re-execute `vulcan_cfg.py` and create a
fresh module object the next time anything does `import vulcan_cfg`. If
this test imported vulcan_cfg at module-level (during pytest collection)
and then test_chem ran first, the module-level `vulcan_cfg` in this test
would refer to a *different* object than the one `outer_loop` later
imports. Mutations on one wouldn't be visible to the other. To dodge
this entirely, we import `outer_loop` first inside `main()` and alias
our `vulcan_cfg` to whatever object `outer_loop` is using. That gives
the test a single source of truth regardless of what earlier tests did
to `sys.modules`.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def main() -> int:
    # Import outer_loop FIRST so any prior sys.modules.pop("vulcan_cfg")
    # is followed by outer_loop's `import vulcan_cfg`, which gives us the
    # canonical module object for this run. Alias to that exact object
    # so our mutations are visible inside outer_loop.
    from atm_setup import Atm
    import legacy_io as op
    import op_jax
    import outer_loop
    from state import RunState, legacy_view
    vulcan_cfg = outer_loop.vulcan_cfg

    # Use a tiny step budget so the test is fast. Force cfg back to known
    # values: prior tests in the same process may have mutated them.
    vulcan_cfg.count_max = 5
    vulcan_cfg.count_min = 1
    vulcan_cfg.rtol = 0.25
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False

    # ---- One-time atmosphere setup (mirrors test_outer_loop_smoke) ----
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

    ok = True

    # ---- Cache state before any run ----
    if integ._runner is not None:
        print("FAIL: runner cached before first run")
        ok = False

    # ---- First run populates the cache ----
    integ(data_var, data_atm, data_para, make_atm)
    cached_runner = integ._runner
    cached_statics = integ._statics
    if cached_runner is None or cached_statics is None:
        print("FAIL: runner / statics not populated after first run")
        ok = False

    # ---- reset() clears all five cache slots ----
    integ.reset()
    if integ._runner is not None:
        print("FAIL: reset() didn't clear _runner")
        ok = False
    if integ._statics is not None:
        print("FAIL: reset() didn't clear _statics")
        ok = False
    if integ._photo_static is not None:
        print("FAIL: reset() didn't clear _photo_static")
        ok = False
    if integ._refresh_static is not None:
        print("FAIL: reset() didn't clear _refresh_static")
        ok = False

    # ---- Mutate cfg, run again — new runner picks up the new value ----
    # Use count_max as the observable signal: the rebuilt _Statics should
    # carry the new value. count_max is purely read at trace time (no
    # writeback mid-run, no adaptive logic), so it's a clean test that
    # reset() forced a fresh _build_statics call.
    new_count_max = 3
    vulcan_cfg.count_max = new_count_max
    integ(data_var, data_atm, data_para, make_atm)
    if integ._runner is None or integ._statics is None:
        print("FAIL: second run didn't repopulate cache")
        ok = False
    if integ._statics.count_max != new_count_max:
        print(f"FAIL: post-reset statics.count_max = "
              f"{integ._statics.count_max} != new_count_max {new_count_max}")
        ok = False
    # The runner object should be different — a new JIT trace.
    if integ._runner is cached_runner:
        print("FAIL: reset() didn't trigger a new runner build "
              "(got the same Python object back)")
        ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
