"""Confirm `use_fix_all_bot` is correctly disabled in Phase 10.1.

The pre-Phase-10.1 test validated `op_jax.Ros2JAX.solver_fix_all_bot` against
`Ros2JAX.solver()` plus a bottom-row clamp. After Phase 10.1, the entire
step-control / solver dispatch moved into `outer_loop.OuterLoop`'s JAX body,
and the BC variant is deferred to Phase 10.6 (in-body post-step clamp).

This test now verifies the deferral is loud: setting `use_fix_all_bot=True`
must raise `NotImplementedError` at `naming_solver` time, not silently fall
through to a wrong code path.

Restore the original Ros2JAX-vs-clamp comparison in Phase 10.6 once the
in-body clamp lands.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent / "VULCAN-master"))

warnings.filterwarnings("ignore")


def main() -> int:
    import vulcan_cfg
    import store
    import op_jax

    # Save original and force the unsupported flag on.
    original = getattr(vulcan_cfg, "use_fix_all_bot", False)
    vulcan_cfg.use_fix_all_bot = True
    try:
        solver = op_jax.Ros2JAX()
        para = store.Parameters()
        try:
            solver.naming_solver(para)
        except NotImplementedError as e:
            print(f"naming_solver correctly raised: {e}")
            print("\nPASS")
            return 0
        print("FAIL: naming_solver should raise NotImplementedError when use_fix_all_bot=True")
        return 1
    finally:
        vulcan_cfg.use_fix_all_bot = original


if __name__ == "__main__":
    sys.exit(main())
