"""Smoke-check the shipped example configs (`vulcan_jax/configs/*.yaml`).

Each subprocess loads a named config with `load_config`, validates it with
`runtime_validation`, and runs the full pre-loop setup through photo / ion-rate
initialization. This catches missing vendored assets and stale config paths
without mutating the process default config. A non-default network/atom set
(e.g. W39b's SNCHO) is import-frozen, so the child sets `$VULCAN_JAX_*` from the
config's own frozen keys BEFORE importing vulcan_jax.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


_SETUP_SCRIPT = r"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

pkg_root = Path(sys.argv[1])   # .../src/vulcan_jax
cfg_name = sys.argv[2]
os.chdir(pkg_root)

# Read the import-frozen knobs from the config YAML BEFORE importing vulcan_jax
# so a non-default network/atom_list/com_file freezes correctly.
raw = yaml.safe_load((pkg_root / "configs" / (cfg_name + ".yaml")).read_text())
if raw.get("network"):
    os.environ["VULCAN_JAX_NETWORK"] = raw["network"]
if raw.get("atom_list"):
    os.environ["VULCAN_JAX_ATOM_LIST"] = ",".join(raw["atom_list"])
if raw.get("com_file"):
    os.environ["VULCAN_JAX_COM_FILE"] = raw["com_file"]

import vulcan_jax  # noqa: F401  (freezes network/atom_list/com_file here)
from vulcan_jax.config import load_config

cfg = load_config(cfg_name)
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

from vulcan_jax.runtime_validation import validate_runtime_config

validate_runtime_config(cfg, root=pkg_root)

# Full pre-loop pipeline (atm + rates + initial abundance + photo
# cross-sections + sflux + remove pass) in one call.
from vulcan_jax.state import RunState

rs = RunState.with_pre_loop_setup(cfg)
assert rs.metadata is not None
assert rs.atm.Tco.shape[0] > 0

print("PASS")
"""


@pytest.mark.parametrize("cfg_name", ["HD209", "K2-18b"])
def test_cfg_example_setup(cfg_name: str) -> None:
    from vulcan_jax._paths import PACKAGE_ROOT

    result = subprocess.run(
        [sys.executable, "-c", _SETUP_SCRIPT, str(PACKAGE_ROOT), cfg_name],
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
