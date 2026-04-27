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

import build_atm
import legacy_io as op
import op_jax
import store

data_var = store.Variables()
data_atm = store.AtmData()
make_atm = build_atm.Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
if cfg.use_condense:
    make_atm.sp_sat(data_atm)
rate = op.ReadRate()
data_var = rate.read_rate(data_var, data_atm)
if cfg.use_lowT_limit_rates:
    data_var = rate.lim_lowT_rates(data_var, data_atm)
data_var = rate.rev_rate(data_var, data_atm)
data_var = rate.remove_rate(data_var)
ini = build_atm.InitialAbun()
data_var = ini.ini_y(data_var, data_atm)
data_var = ini.ele_sum(data_var)
data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)

if cfg.use_photo:
    rate.make_bins_read_cross(data_var, data_atm)
    make_atm.read_sflux(data_var, data_atm)
    solver = op_jax.Ros2JAX()
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    if cfg.use_ion:
        solver.compute_Jion(data_var, data_atm)
    data_var = rate.remove_rate(data_var)

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
