"""A photochemical integration with T-dependent cross sections.

The only integration test with `T_cross_sp` non-empty: the per-profile
T-dependent photo arrays ride `ProfileVars` (`p_absp_T_cross`,
`p_cross_J_T`), so a wrong shape or a placeholder there crashes the photo
branch as soon as it fires.
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


def main() -> int:
    import vulcan_jax.legacy_io as op
    from vulcan_jax import op_jax, outer_loop
    from vulcan_jax.state import RunState

    vulcan_cfg = op.default_config()

    vulcan_cfg.count_max = 12
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_photo = True
    vulcan_cfg.use_ion = False
    vulcan_cfg.ini_mix = "const_mix"
    vulcan_cfg.atm_type = "isothermal"
    vulcan_cfg.Kzz_prof = "Pfunc"
    vulcan_cfg.nz = 30
    vulcan_cfg.T_cross_sp = ["H2O"]  # the trigger: non-empty T-dep cross sections
    vulcan_cfg.ini_update_photo_frq = 4

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    # Vacuity guard: the T-dep stack must actually be populated.
    assert rs.photo_static is not None
    assert rs.photo_static.absp_T_cross.shape[0] > 0

    rs_out = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())(rs)

    count = int(rs_out.params.count)
    assert np.all(np.isfinite(np.asarray(rs_out.step.y))), "non-finite y after T_cross_sp run"
    assert count >= 12, f"run did not progress (count={count})"
    print(f"T_cross_sp run OK (count={count})")
    return 0


@pytest.mark.strict_isolation
def test_main():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
