"""End-to-end activation of the ion / photoionization machinery, in a subprocess.

No shipped network carries an `# ionisation` section (upstream is the same),
so this test synthesizes one at runtime: shipped NCHO + H2/H photoionisation,
using only vendored data (ion cross-section columns, ion branch files, NASA9
for H2_p/H_p/e, all_compose charge rows). The network is import-locked, so the
activated network needs a fresh child process ($VULCAN_JAX_NETWORK).

Checks: (A) setup wired the ion path (ion_sp, ion_rate_index, charge_list
without `e`, non-zero cross_Jion); (B) both cations grow from zero, so
compute_Jion_jax fired inside the loop; (C) per-layer charge neutrality to
~1e-8 rel; (D) the run stays finite and advances in time.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE_NETWORK = ROOT / "src" / "vulcan_jax" / "thermo" / "NCHO_photo_network.txt"

# Appended after the shipped NCHO photo section. The parser keys ion branches
# off (species, branch_index) and pulls cross sections from {sp}_cross.csv
# (ion column) x {sp}_ion_branch.csv; products introduce H2_p/H_p/e.
ION_SECTION = """
# ionisation
879  [ H2 -> H2_p + e                     ]             H2      1
881  [ H -> H_p + e                       ]             H       1
"""

# Runs in a fresh interpreter so the import-locked network is the synthesized
# ion network (selected via $VULCAN_JAX_NETWORK, inherited from the parent env).
_CHILD = r"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
# cwd at the package root so relative data paths (cross_folder, sflux_file)
# resolve; $VULCAN_JAX_NETWORK is absolute and resolves regardless.
os.chdir(repo / "src" / "vulcan_jax")

import numpy as np

# --- Configure a self-contained hot H2 column with photo + ion ON. The global
# vulcan_cfg module carries the working HD189 photo defaults (r_star, orbit,
# zenith angle, scattering, bin widths); we override only what we need. cfg is
# mutated BEFORE the first jax_step/runner import so the values are seen. ---
from vulcan_jax.config import default_config
cfg = default_config()

cfg.use_photo = True
cfg.use_ion = True
cfg.use_condense = False
cfg.atm_type = "isothermal"; cfg.Tiso = 1500.0
cfg.atm_base = "H2"
cfg.use_moldiff = False
cfg.use_Kzz = True; cfg.Kzz_prof = "const"; cfg.const_Kzz = 1e9; cfg.use_vz = False
cfg.ini_mix = "const_mix"
# Seed a little atomic H so H photoionisation has a parent from step 0; H2
# dominates and drives the strong ion channel. Ions themselves start at zero.
cfg.const_mix = {"H2": 0.85, "He": 0.149, "H2O": 5e-4, "CO": 2e-4,
                 "CH4": 1e-4, "N2": 1e-4, "H": 1e-6}
cfg.nz = 50; cfg.P_b = 1e9; cfg.P_t = 1e-2
cfg.sflux_file = "atm/stellar_flux/sflux-HD189_Moses11.txt"  # EUV-extended (2-800 nm)
cfg.use_topflux = False; cfg.use_botflux = False
cfg.use_fix_sp_bot = {}; cfg.use_ini_cold_trap = False
# Bounded run: enough steps + frequent photo/ion updates to accumulate ions,
# without converging on full HD189 timescales.
cfg.count_min = 150; cfg.count_max = 400
cfg.ini_update_photo_frq = 20; cfg.final_update_photo_frq = 5
cfg.use_print_prog = False
cfg.use_live_plot = False; cfg.use_live_flux = False
cfg.use_save_movie = False; cfg.use_flux_movie = False

# cfg is the process default config (mutated above); the runner/printer read it
# via OuterLoop's default. The ion network was selected by $VULCAN_JAX_NETWORK
# before the first import.
import vulcan_jax.outer_loop as outer_loop
import vulcan_jax.legacy_io as op

import vulcan_jax.op_jax as op_jax
from vulcan_jax.atm_setup import Atm
from vulcan_jax.state import RunState, legacy_view
from vulcan_jax.chem_funs import spec_list as SL


def fail(msg):
    print("FAIL: " + msg)
    raise SystemExit(1)


# Confirm the synthesized ion network really was the one imported.
if not ("H2_p" in SL and "H_p" in SL and "e" in SL):
    fail("ion species not in spec_list -- wrong network imported? SL has %d species" % len(SL))

rs = RunState.with_pre_loop_setup(cfg)
md = rs.metadata

# ---------- Check A: setup wired the ion path ----------
ion_sp = set(md.ion_sp)
if ion_sp != {"H2", "H"}:
    fail("ion_sp=%r, expected {'H2','H'}" % ion_sp)
for key in (("H2", 1), ("H", 1)):
    if key not in md.ion_rate_index:
        fail("ion_rate_index missing %r (have %r)" % (key, list(md.ion_rate_index)))
charge_list = set(md.charge_list)
if not {"H2_p", "H_p"}.issubset(charge_list):
    fail("charge_list=%r missing cations" % charge_list)
if "e" in charge_list:
    fail("'e' must be excluded from charge_list (it is the balanced quantity)")
cross_Jion = np.asarray(rs.photo_static.cross_Jion, dtype=np.float64)
if cross_Jion.shape[0] < 2 or not np.isfinite(cross_Jion).all():
    fail("cross_Jion shape/finite bad: shape=%r" % (cross_Jion.shape,))
if float(np.max(cross_Jion)) <= 0.0:
    fail("cross_Jion is all <= 0 -- ion cross sections not loaded")
print("A WIRED ion_sp=%s cations=%s cross_Jion_max=%.3e nbin=%d"
      % (sorted(ion_sp), sorted({"H2_p", "H_p"}), float(np.max(cross_Jion)),
         cross_Jion.shape[1]))

i_H2p, i_Hp, i_e = SL.index("H2_p"), SL.index("H_p"), SL.index("e")
y0 = np.asarray(rs.step.y, dtype=np.float64)
ion0 = float(y0[:, [i_H2p, i_Hp]].max())  # initial ion density (expected ~0)

# ---------- Drive the bounded integration (legacy path mutates var) ----------
data_var, data_atm, data_para = legacy_view(rs)
data_para.start_time = time.time()
solver = op_jax.Ros2JAX()
if rs.photo_static is not None:
    solver._photo_static = rs.photo_static
integ = outer_loop.OuterLoop(solver, op.Output())
solver.naming_solver(data_para)
t0 = time.time()
integ(data_var, data_atm, data_para, Atm())
print("   ran count=%d t=%.3e dt=%.3e in %.1fs"
      % (data_para.count, float(data_var.t), float(data_var.dt), time.time() - t0))

y = np.asarray(data_var.y, dtype=np.float64)

# ---------- Check D (early): finite + advanced ----------
if not np.isfinite(y).all():
    fail("non-finite abundances after run")
if not (float(data_var.t) > 0.0 and np.isfinite(float(data_var.dt)) and float(data_var.dt) > 0.0):
    fail("time/step not finite-positive (t=%r dt=%r)" % (data_var.t, data_var.dt))

# ---------- Check B: photoionisation produced ions (0 -> >0) ----------
H2p_max = float(y[:, i_H2p].max())
Hp_max = float(y[:, i_Hp].max())
ion_max = max(H2p_max, Hp_max)
if not (H2p_max > 0.0):
    fail("no H2 photoionisation: H2_p_max=%.3e (init %.3e)" % (H2p_max, ion0))
if not (Hp_max > 0.0):
    fail("no H photoionisation: H_p_max=%.3e (init %.3e)" % (Hp_max, ion0))
if not (ion_max > ion0):
    fail("ions did not grow from init (ion0=%.3e ion_max=%.3e)" % (ion0, ion_max))
print("B IONS H2_p_max=%.3e H_p_max=%.3e (init %.3e) -- compute_Jion_jax fired"
      % (H2p_max, Hp_max, ion0))

# ---------- Check C: charge neutrality (e == sum of cations) ----------
e_dens = y[:, i_e]
cation = y[:, i_H2p] + y[:, i_Hp]
mask = cation > 0.0
if not mask.any():
    fail("no layer with positive cation density to test neutrality")
rel = np.abs(e_dens[mask] - cation[mask]) / cation[mask]
relresid = float(np.max(rel))
if relresid > 1e-8:
    bad = int(np.argmax(rel))
    fail("charge neutrality violated: max rel resid=%.3e (e=%.6e cations=%.6e)"
         % (relresid, float(e_dens[mask][bad]), float(cation[mask][bad])))
# And it is non-vacuous: there is real charge separation to balance.
if float(np.max(cation)) <= 0.0:
    fail("vacuous neutrality (no cations at all)")
print("C NEUTRAL layers=%d max_rel_resid=%.3e e_max=%.3e cation_max=%.3e"
      % (int(mask.sum()), relresid, float(e_dens.max()), float(cation.max())))

print("D STABLE finite=ok t=%.3e dt=%.3e" % (float(data_var.t), float(data_var.dt)))
print("PASS")
"""


@pytest.mark.strict_isolation
def test_ion_chemistry_end_to_end(tmp_path):
    if not BASE_NETWORK.exists():
        pytest.skip(f"base network {BASE_NETWORK} not vendored")

    # Synthesize the activated ion network = shipped NCHO + an # ionisation
    # section. Written to pytest's tmp_path (auto-cleaned); the child imports it
    # via the absolute $VULCAN_JAX_NETWORK path below.
    ion_network = tmp_path / "NCHO_photo_ion_network.txt"
    ion_network.write_text(BASE_NETWORK.read_text().rstrip() + "\n" + ION_SECTION)

    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "VULCAN_JAX_NETWORK": str(ion_network),
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
        f"ion activation subprocess exited {res.returncode}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    # Each stage must have reported, and the run must finish on PASS.
    for marker in ("A WIRED", "B IONS", "C NEUTRAL", "D STABLE"):
        assert marker in res.stdout, f"missing {marker!r} stage:\n{res.stdout}"
    assert res.stdout.strip().endswith("PASS"), res.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
