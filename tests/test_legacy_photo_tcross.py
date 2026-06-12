"""Legacy (var, atm, para) entry with T-dependent cross sections.

The legacy call path synthesizes its RunState via `runstate_from_store`,
which carries no `photo_static` slot. The per-profile photo arrays in
`ProfileVars` must then fall back to the closure-baked `_PhotoStatic`
(value-identical for this single-profile-only path) — regression guard for
the bug where `(0, 1, 1)` placeholders were packed instead and the photo
branch crashed on shape mismatch whenever `T_cross_sp` was non-empty.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")


def main() -> int:
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op

    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_jax.vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg

    vulcan_cfg.count_max = 12
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False
    vulcan_cfg.use_photo = True
    vulcan_cfg.use_ion = False
    vulcan_cfg.ini_mix = "const_mix"
    vulcan_cfg.atm_type = "isothermal"
    vulcan_cfg.Kzz_prof = "Pfunc"
    vulcan_cfg.nz = 30
    vulcan_cfg.T_cross_sp = ["H2O"]  # the trigger: non-empty T-dep cross sections
    vulcan_cfg.ini_update_photo_frq = 4

    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.atm_setup import Atm
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    # Vacuity guard: the T-dep stack must actually be populated.
    assert rs.photo_static is not None
    assert rs.photo_static.absp_T_cross.shape[0] > 0

    data_var, data_atm, data_para = legacy_view(rs)
    data_para.start_time = time.time()
    make_atm = Atm()
    output = op.Output()
    solver = op_jax.Ros2JAX()
    solver._photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver, output)
    solver.naming_solver(data_para)

    integ(data_var, data_atm, data_para, make_atm)  # crashed pre-fix

    y = np.asarray(data_var.y)
    assert np.all(np.isfinite(y)), "non-finite y after legacy T_cross_sp run"
    assert data_para.count >= 12, f"run did not progress (count={data_para.count})"
    print(f"legacy + T_cross_sp run OK (count={data_para.count})")
    return 0


@pytest.mark.strict_isolation
def test_main():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
