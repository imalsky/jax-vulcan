"""Smoke-check the standalone example configs vendored into VULCAN-JAX.

Each subprocess loads a cfg from `cfg_examples/` as `vulcan_cfg`,
validates it with `runtime_validation`, and runs the full pre-loop
setup through photo / ion-rate initialization. This catches missing
vendored assets and stale config paths without mutating the repo's
default `vulcan_cfg.py`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


_SETUP_SCRIPT = r"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

root = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
os.chdir(root)
sys.path.insert(0, str(root))

spec = importlib.util.spec_from_file_location("vulcan_cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)
sys.modules["vulcan_cfg"] = cfg

cfg.use_print_prog = False
cfg.use_live_plot = False
cfg.use_live_flux = False
cfg.use_plot_end = False
cfg.use_plot_evo = False
cfg.use_save_movie = False
cfg.use_flux_movie = False
cfg.save_evolution = False
cfg.count_max = 1
cfg.count_min = 1
cfg.runtime = 1.0e22

from runtime_validation import validate_runtime_config

validate_runtime_config(cfg, root=root)

# Full pre-loop pipeline (atm + rates + initial abundance + photo
# cross-sections + sflux + remove pass) is encapsulated in
# `RunState.with_pre_loop_setup(cfg)` — a single call covering the
# `make_atm` / `ReadRate` / `setup_var_k` / `InitialAbun` /
# `populate_photo_arrays` / `apply_photo_remove` chain.
from state import RunState

rs = RunState.with_pre_loop_setup(cfg)
assert rs.metadata is not None
assert rs.atm.Tco.shape[0] > 0

print("PASS")
"""


@pytest.mark.parametrize(
    "cfg_name",
    [
        "vulcan_cfg_Earth.py",
        "vulcan_cfg_Jupiter.py",
        "vulcan_cfg_HD209.py",
    ],
)
def test_cfg_example_setup(cfg_name: str) -> None:
    cfg_path = ROOT / "cfg_examples" / cfg_name
    result = subprocess.run(
        [sys.executable, "-c", _SETUP_SCRIPT, str(ROOT), str(cfg_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"{cfg_name} setup failed with exit code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("PASS")
