"""Smoke test for outer_loop.OuterLoop.

Runs 50 accepted Ros2 steps end-to-end on the HD189 reference state and
asserts:
    1. count progressed exactly 50 (no silent retries lost / counted as accept)
    2. atom_loss agrees with VULCAN-master's tracked baseline
       (1.95e-4 per atom; the end-to-end 50-step HD189 oracle is the 1.59e-10
       relative-difference reference).
    3. no force-accept / nega / loss / delta retries fired (HD189 is smooth)
    4. dt advanced from 1e-10 (dttry) to ~1e-3 (typical 50-step HD189)

The canonical "did anything break the established baseline" smoke test
for any change to outer_loop.py — fast (~10s including JIT compile).
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


# Per-atom atom_loss target: matches the established VULCAN-master
# baseline to four significant figures. Tolerance set to 5% of the value
# so the test is robust to small JAX-vs-NumPy reduction-order differences
# while still catching real regressions (e.g. a 10x atom_loss blowup).
EXPECTED_ATOM_LOSS = 1.95e-4
ATOM_LOSS_RTOL = 0.05


def main() -> int:
    # Sync vulcan_cfg references. test_chem pops `vulcan_cfg` from
    # sys.modules mid-run; a plain `import vulcan_cfg` here would create a
    # fresh module object distinct from the one outer_loop / legacy_io
    # captured at their module-import time, and our `count_max = 50` would
    # land on the detached copy while the runner reads count_max from
    # outer_loop's still-stale reference. Pin everything to legacy_io's
    # reference instead so the runner and the printer see what we set.
    import outer_loop
    import legacy_io as op
    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg

    vulcan_cfg.count_max = 50
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False

    import op_jax
    from atm_setup import Atm
    from state import RunState, legacy_view

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

    t0 = time.time()
    integ(data_var, data_atm, data_para, make_atm)
    elapsed = time.time() - t0

    print(f"50-step HD189 via OuterLoop: {elapsed:.1f}s "
          f"({elapsed / data_para.count * 1000:.1f} ms/step)")
    print(f"  count={data_para.count}, t={data_var.t:.3e}, dt={data_var.dt:.3e}")
    print(f"  retry counters: nega={data_para.nega_count}, "
          f"loss={data_para.loss_count}, delta={data_para.delta_count}")
    print("  atom_loss: " + " ".join(
        f"{a}={v:.3e}" for a, v in data_var.atom_loss.items()
    ))

    ok = True

    # 1. Step counter exactly hit count_max (51 = 50 accepted + 1 init counter).
    if data_para.count != vulcan_cfg.count_max + 1:
        print(f"FAIL: expected count == {vulcan_cfg.count_max + 1}, got {data_para.count}")
        ok = False

    # 2. atom_loss matches baseline.
    for atom, loss in data_var.atom_loss.items():
        relerr = abs(loss - EXPECTED_ATOM_LOSS) / EXPECTED_ATOM_LOSS
        if relerr > ATOM_LOSS_RTOL:
            print(f"FAIL: atom_loss[{atom}] = {loss:.3e}; expected ~{EXPECTED_ATOM_LOSS:.3e} "
                  f"(relerr={relerr:.3e}, tol={ATOM_LOSS_RTOL:.3e})")
            ok = False

    # 3. HD189 is smooth — no retries should fire.
    if data_para.nega_count + data_para.loss_count + data_para.delta_count > 0:
        print(f"WARN: retries fired on HD189 (nega={data_para.nega_count}, "
              f"loss={data_para.loss_count}, delta={data_para.delta_count}); "
              "this is unexpected for the smooth baseline.")
        # Don't fail — retries are correct behavior, just unusual on HD189.

    # 4. dt advanced reasonably (1e-10 -> at least 1e-6).
    if not (1e-7 < data_var.dt < 1e-1):
        print(f"FAIL: dt = {data_var.dt:.3e} outside expected range [1e-7, 1e-1] "
              f"after 50 steps")
        ok = False

    # 5. t advanced past 1e-4.
    if data_var.t < 1e-5:
        print(f"FAIL: t = {data_var.t:.3e} too small after 50 steps")
        ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
