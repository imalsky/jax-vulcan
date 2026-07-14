"""Lock sulfur atom-conservation projection on the SNCHO network, in a subprocess.

`tests/test_atom_conservation_projection.py` exercises the reservoir projection on
the default HD189 / NCHO network, where only H/O/C/N are tracked. The S-bearing
SNCHO network adds sulfur, conserved through the H2S reservoir
(`jax_step._ATOM_RESERVOIRS`). Because the reaction network *and* the projection
tables are import-locked (built once at the first ``import vulcan_jax`` from
``$VULCAN_JAX_NETWORK`` and ``vulcan_cfg.atom_list``), the S path can only be
exercised in a fresh process that selects the SNCHO network and an
``atom_list`` carrying S before that first import — hence a child process.

Self-contained — no committed data artifact, no atm file, no FastChem, no photo:
isothermal TP + const_mix init (the proven setup from
`tests/test_condensation_runtime_subprocess.py`).

Checks (SNCHO network, atom_list = H/O/C/N/S):
  1. The projection is enabled and covers all five atoms, with reservoirs
     exactly (H2, H2O, CO, N2, H2S); the 5x5 reservoir count matrix is
     well-conditioned (single linear solve is stable).
  2. Operator: injecting a conservation defect on a non-reservoir sulfur species
     (SO2, which perturbs both S and O) drives the per-atom residual — the S
     column included — to machine-zero, mutating ONLY reservoir-species rows.
  3. The codegen RHS, evaluated on a real isothermal SNCHO pre-loop state, stays
     conserving across all five atoms after projection, touching only reservoir
     rows.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S_NETWORK = "thermo/SNCHO_photo_network.txt"

# The child selects the SNCHO network and an S-bearing atom_list, then imports
# jax_step for the first time so its projection tables are built with sulfur.
_CHILD = r"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["VULCAN_JAX_NETWORK"] = "thermo/SNCHO_photo_network.txt"

from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
os.chdir(repo / "src" / "vulcan_jax")

import numpy as np
import jax.numpy as jnp

# Set atom_list BEFORE the first jax_step import so the static reservoir tables
# pick up sulfur (state.py imports jax_step lazily, so this ordering holds).
from vulcan_jax.config import default_config
cfg = default_config()
cfg.atom_list = ["H", "O", "C", "N", "S"]

import vulcan_jax.jax_step as js
import vulcan_jax.composition as composition
from vulcan_jax.chem_funs import spec_list as SL

ATOMS = ("H", "O", "C", "N", "S")
RES = ("H2", "H2O", "CO", "N2", "H2S")

# ---- Check 1: projection enabled, covers S, reservoir set + conditioning ----
assert js._CHEM_PROJECTION_ENABLED, "projection disabled for SNCHO + S atom_list"
assert int(js._CHEM_ATOM_COUNTS.shape[1]) == 5, "expected 5 projected atoms (H/O/C/N/S)"
res_species = tuple(SL[int(i)] for i in js._CHEM_RESERVOIR_IDX)
assert res_species == RES, f"reservoir species {res_species} != {RES}"

row = {str(r["species"]): r for r in composition.compo}
A = np.asarray([[float(row[sp][a]) for a in ATOMS] for sp in SL], dtype=np.float64)
res_idx = [SL.index(s) for s in RES]
det = float(np.linalg.det(A[res_idx]))
cond = float(np.linalg.cond(A[res_idx]))
assert abs(det) > 1e-6, f"reservoir count matrix singular (det={det})"
print(f"ENABLED atoms=5 reservoirs={res_species} det={det:.3f} cond={cond:.2f}")

# ---- Check 2: inject an S-bearing defect on a non-reservoir species ----
s_carriers = [sp for sp in SL if row[sp]["S"] > 0 and sp not in RES]
assert s_carriers, "no non-reservoir sulfur species in network"
inj = "SO2" if "SO2" in s_carriers else s_carriers[0]
nz = 40
defect = np.zeros((nz, len(SL)), dtype=np.float64)
defect[:, SL.index(inj)] = 7.0
resid0 = defect @ A
proj = np.asarray(js._project_chem_rhs(jnp.asarray(defect)))
resid1 = proj @ A
non_res = np.delete(proj - defect, res_idx, axis=1)
print(f"inject={inj} S={int(row[inj]['S'])} O={int(row[inj]['O'])} "
      f"resid_before={np.max(np.abs(resid0)):.3e} resid_after={np.max(np.abs(resid1)):.3e} "
      f"S_after={np.max(np.abs(resid1[:,4])):.3e} outside_res={np.max(np.abs(non_res)):.3e}")
assert np.max(np.abs(resid0)) > 0.0, "injected residual is zero (vacuous)"
assert np.max(np.abs(resid1)) < 1e-12, "projection did not zero the injected S residual"
assert np.max(np.abs(non_res)) == 0.0, "projection mutated a non-reservoir species"

# ---- Check 3: real codegen RHS conserves all five atoms after projection ----
# Self-contained isothermal SNCHO state (no atm file / FastChem / photo).
cfg.atm_type = "isothermal"; cfg.Tiso = 1200.0
cfg.atm_base = "H2"; cfg.use_moldiff = False
cfg.use_Kzz = True; cfg.Kzz_prof = "const"; cfg.const_Kzz = 1e9; cfg.use_vz = False
cfg.use_photo = False; cfg.use_ion = False; cfg.use_condense = False
cfg.ini_mix = "const_mix"
cfg.const_mix = {"H2": 0.85, "He": 0.149, "H2O": 5e-4, "CO": 2e-4,
                 "CH4": 1e-4, "N2": 1e-4, "H2S": 3e-5}
cfg.nz = 50; cfg.P_b = 1e9; cfg.P_t = 1e0
cfg.use_topflux = False; cfg.use_botflux = False
cfg.use_fix_sp_bot = {}; cfg.use_ini_cold_trap = False; cfg.use_print_prog = False

from vulcan_jax.state import RunState
from vulcan_jax.chem_funs import chem_rhs_codegen as RHS

rs = RunState.with_pre_loop_setup(cfg)
y = jnp.asarray(np.asarray(rs.step.y, dtype=np.float64))
M = jnp.asarray(np.asarray(rs.atm.M, dtype=np.float64))
k = jnp.asarray(np.asarray(rs.rate.k, dtype=np.float64))

raw = np.asarray(RHS(y, M, k))
raw_proj = np.asarray(js._project_chem_rhs(jnp.asarray(raw)))
prod_peak = float(np.max(np.abs(raw) @ A))
rel = float(np.max(np.abs(raw_proj @ A))) / max(prod_peak, 1e-300)
real_non_res = np.delete(raw_proj - raw, res_idx, axis=1)
print(f"real-RHS resid/prod_peak={rel:.3e} outside_res={np.max(np.abs(real_non_res)):.3e}")
assert rel < 1e-12, "real RHS not conserving across all 5 atoms after projection"
assert np.max(np.abs(real_non_res)) == 0.0, "real-RHS projection mutated a non-reservoir species"

print("PASS")
"""


def test_atom_conservation_s_subprocess():
    from vulcan_jax._paths import PACKAGE_ROOT

    if not (PACKAGE_ROOT / S_NETWORK).exists():
        pytest.skip(f"sulfur network {S_NETWORK!r} not vendored")

    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "VULCAN_JAX_NETWORK": S_NETWORK,
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
        f"sulfur conservation subprocess exited {res.returncode}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    assert "ENABLED atoms=5" in res.stdout, res.stdout
    assert res.stdout.strip().endswith("PASS"), res.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
