"""End-to-end condensation runtime coverage, in a subprocess.

This closes the gap that `tests/test_config_matrix.py::test_use_settling_populates_vs_for_non_gas`
and `::test_fix_species_runtime_smoke` only mark as a conditional skip: the
default HD189 / NCHO network has no condensate species (`H2O_l_s`, `S8_l_s`), so
the assembled condensation runtime path is never exercised by the curated suite.

The reaction network is import-locked (parsed once at the first
``import vulcan_jax``), so a condensate network must be selected via
``$VULCAN_JAX_NETWORK`` *before* that first import — hence a child process.
``SNCHO_photo_network_2025.txt`` is the one vendored network carrying both
``H2O_l_s`` and ``S8_l_s``.

Self-contained — no committed data artifact:
  * vendored reaction network + thermo tables only,
  * ``atm_type="isothermal"`` (synthetic TP, reads no atm file),
  * ``ini_mix="const_mix"`` (no FastChem), ``use_photo=False`` (no cross sections).

Two checks, mirroring the two skipped tests but on a network that actually has
the condensates:
  1. **settling** — ``atm_setup.compute_settling_velocity`` gives a downward
     (negative) velocity for each non-gas condensate column and exactly zero for
     gas species (the property `test_use_settling_populates_vs_for_non_gas` checks);
  2. **condensation forms** — a short H2O-relaxation run completes finite and
     actually moves mass into the condensate (``H2O_l_s`` total > 0), exercising
     the full runner conden path (`_build_conden_static`, `apply_h2o_relax_jax`,
     the conden gate).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONDEN_NETWORK = "thermo/SNCHO_photo_network_2025.txt"

# The child sets $VULCAN_JAX_NETWORK and imports vulcan_jax for the first time
# inside this process, so the condensate network is the import-locked one.
_CHILD = r"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["VULCAN_JAX_NETWORK"] = "thermo/SNCHO_photo_network_2025.txt"

from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))        # resolve `import vulcan_jax` to src/
os.chdir(repo / "src" / "vulcan_jax")         # data paths are package-relative

import numpy as np
import vulcan_jax  # first import -> parses the condensate network
import vulcan_jax.composition as composition
import vulcan_jax.chem_funs as chem_funs

sl = list(composition.species)
for need in ("H2O", "S8", "H2O_l_s", "S8_l_s"):
    assert need in sl, f"condensate network missing required species {need!r}"

import vulcan_jax.vulcan_cfg as cfg
# Synthetic isothermal atmosphere, const_mix init, no photo -> no data files.
cfg.atm_type = "isothermal"; cfg.Tiso = 250.0           # cold enough for H2O to supersaturate
cfg.atm_base = "N2"; cfg.use_moldiff = True             # settling requires use_moldiff
cfg.use_Kzz = True; cfg.Kzz_prof = "const"; cfg.const_Kzz = 1e5; cfg.use_vz = False
cfg.use_photo = False; cfg.use_ion = False
cfg.ini_mix = "const_mix"
cfg.const_mix = {"N2": 0.79, "O2": 0.20, "H2O": 1e-2, "CO2": 4e-4}
cfg.nz = 50; cfg.P_b = 1e6; cfg.P_t = 1e-1
cfg.use_topflux = False; cfg.use_botflux = False
cfg.use_fix_sp_bot = {}; cfg.use_ini_cold_trap = False; cfg.use_print_prog = False
cfg.use_condense = True

# ---- Check 1: settling velocities for both condensates ----
cfg.use_settling = True
cfg.condense_sp = ["H2O", "S8"]; cfg.non_gas_sp = ["H2O_l_s", "S8_l_s"]
cfg.r_p = {"H2O_l_s": 5e-3, "S8_l_s": 1e-4}
cfg.rho_p = {"H2O_l_s": 0.9, "S8_l_s": 2.07}
cfg.use_relax = []

from vulcan_jax.atm_setup import Atm
from vulcan_jax.state import _Variables, _AtmData
from vulcan_jax.ini_abun import InitialAbun
import vulcan_jax.legacy_io as op

dv = _Variables(); da = _AtmData(); make_atm = Atm()
da = make_atm.f_pico(da); da = make_atm.load_TPK(da); make_atm.sp_sat(da)
dv = InitialAbun().ini_y(dv, da)
da = make_atm.f_mu_dz(dv, da, op.Output(cfg=cfg))
vs = np.asarray(da.vs); nz = da.Tco.shape[0]
assert vs.shape == (nz - 1, len(sl)), f"vs shape {vs.shape}"
for cond in ("H2O_l_s", "S8_l_s"):
    j = sl.index(cond)
    assert np.all(vs[:, j] < 0.0), f"{cond} settling velocity not downward"
gas = [i for i in range(len(sl)) if sl[i] not in ("H2O_l_s", "S8_l_s")]
assert np.all(vs[:, gas] == 0.0), "settling velocity nonzero on a gas species"
print("SETTLING_OK")

# ---- Check 2: short H2O-relax run forms condensate ----
cfg.use_settling = True
cfg.condense_sp = ["H2O"]; cfg.non_gas_sp = ["H2O_l_s"]
cfg.r_p = {"H2O_l_s": 0.01}; cfg.rho_p = {"H2O_l_s": 0.9}
cfg.use_relax = ["H2O"]; cfg.humidity = 1.0
cfg.start_conden_time = 0.0; cfg.stop_conden_time = 1e22
cfg.count_max = 50; cfg.count_min = 1

from vulcan_jax.state import RunState
import vulcan_jax.op_jax as op_jax
import vulcan_jax.outer_loop as outer_loop

t0 = time.time()
rs = RunState.with_pre_loop_setup(cfg)
integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
rs_out = integ(rs)
y = np.asarray(rs_out.step.y)
assert np.all(np.isfinite(y)), "non-finite densities after conden run"
cond_total = float(y[:, sl.index("H2O_l_s")].sum())
assert cond_total > 0.0, f"no H2O condensate formed (total={cond_total:.3e})"
print(f"CONDEN_OK t={float(rs_out.step.t):.2e} count={int(rs_out.params.count)} "
      f"H2O_l_s_total={cond_total:.3e} ({time.time()-t0:.1f}s)")
print("PASS")
"""


def test_condensation_runtime_subprocess():
    from vulcan_jax._paths import PACKAGE_ROOT

    if not (PACKAGE_ROOT / CONDEN_NETWORK).exists():
        pytest.skip(f"condensate network {CONDEN_NETWORK!r} not vendored")

    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "VULCAN_JAX_NETWORK": CONDEN_NETWORK,
    }
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=ROOT,
    )
    assert res.returncode == 0, (
        f"condensation subprocess exited {res.returncode}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    assert "SETTLING_OK" in res.stdout, res.stdout
    assert "CONDEN_OK" in res.stdout, res.stdout
    assert res.stdout.strip().endswith("PASS")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
