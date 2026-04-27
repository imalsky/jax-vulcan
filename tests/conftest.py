"""Shared pytest setup for VULCAN-JAX tests.

Each `tests/test_*.py` retains its existing `def main()` script entry
point and adds a thin `def test_main(): assert main() == 0` wrapper so
`pytest tests/` collects and runs them. This conftest pins working
directory and `sys.path` once per session.

VULCAN-master sibling: VULCAN-JAX is standalone; the upstream repo
serves as an *optional* validation oracle. If `../VULCAN-master/` is
present, it gets appended to sys.path so the 10 oracle tests
(test_rates, test_chem, test_diffusion, test_diffusion_variants,
test_photo, test_photo_wired, test_ros2_step, test_gibbs,
test_step_control, test_outer_loop_atm_refresh) can `import op` and
compare. If absent, those tests skip cleanly — the rest of the suite
is unaffected.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VULCAN_MASTER = ROOT.parent / "VULCAN-master"

# Make sure VULCAN-JAX is importable regardless of where pytest was launched.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# VULCAN-master is optional — only append when present.
HAS_VULCAN_MASTER = VULCAN_MASTER.is_dir()
if HAS_VULCAN_MASTER and str(VULCAN_MASTER) not in sys.path:
    sys.path.append(str(VULCAN_MASTER))

# Many tests assume cwd == ROOT for relative paths in vulcan_cfg.py.
os.chdir(ROOT)

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def vulcan_master_op():
    """Import VULCAN-master's `op` for oracle-comparison tests.

    Returns the imported module. Skips the test cleanly if
    `../VULCAN-master/` isn't present in this workspace.
    """
    if not HAS_VULCAN_MASTER:
        pytest.skip(
            f"VULCAN-master not present at {VULCAN_MASTER}; "
            "oracle test skipped (VULCAN-JAX is standalone)."
        )
    import op
    return op
