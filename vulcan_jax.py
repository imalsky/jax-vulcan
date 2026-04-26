#!/usr/bin/env python
# ============================================================================
# VULCAN-JAX: a JAX-accelerated drop-in replacement for VULCAN.
#
# Mirrors `VULCAN-master/vulcan.py` line-for-line in orchestration order.
# Differences:
#   - Uses `op_jax.Ros2JAX` (subclass of op.Ros2) so the inner Rosenbrock
#     step uses JAX block-Thomas + autodiff Jacobian in place of scipy's
#     banded solver.
#   - Photochemistry (compute_tau / compute_flux / compute_J) uses the pure-JAX
#     kernels in photo.py instead of inherited NumPy.
#   - chem_funs.py is a JAX-native module (network parser + NASA-9 + JAX
#     wrappers) — no SymPy code generator, no make_chem_funs.py invocation.
#
# Run:    python vulcan_jax.py        (the -n flag is now a no-op; kept for
#         python vulcan_jax.py -n      backward-compatibility with VULCAN scripts)
# ============================================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import warnings
from pathlib import Path

# Make sure both VULCAN-JAX and VULCAN-master are importable.
ROOT = Path(__file__).resolve().parent
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
sys.path.insert(0, str(ROOT))
sys.path.append(str(VULCAN_MASTER))

# Cwd at the script's directory so relative paths in vulcan_cfg.py resolve
os.chdir(ROOT)

# chem_funs.py is now a JAX-native module in VULCAN-JAX/; no SymPy regen needed.
# The -n flag is accepted (and ignored) for compatibility with VULCAN scripts.
print("Using JAX-native chem_funs (no make_chem_funs.py step required)")

import numpy as np

# Now safe to import VULCAN modules + JAX kernels
import store
import build_atm
import op
import chem_funs

import vulcan_cfg
from phy_const import kb, Navo

import op_jax       # JAX photo helpers
import outer_loop   # JAX outer integration loop (Phase 10.1+)

abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)

from chem_funs import ni, nr
np.set_printoptions(threshold=np.inf)

species = chem_funs.spec_list

# Read elemental composition (mirrors vulcan.py:91-99)
with open(vulcan_cfg.com_file, "r") as f:
    columns = f.readline()
    num_ele = len(columns.split()) - 2
type_list = ["int" for _ in range(num_ele)]
type_list.insert(0, "U20")
type_list.append("float")
compo = np.genfromtxt(vulcan_cfg.com_file, names=True, dtype=type_list)
compo_row = list(compo["species"])

# === Construct the storage objects ===
data_var = store.Variables()
data_atm = store.AtmData()
data_para = store.Parameters()

data_para.start_time = time.time()

make_atm = build_atm.Atm()
output = op.Output()

# Save the config copy
output.save_cfg(dname)

# Build the atmosphere (mirrors vulcan.py:118-155)
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
if vulcan_cfg.use_condense:
    make_atm.sp_sat(data_atm)

rate = op.ReadRate()
data_var = rate.read_rate(data_var, data_atm)

if vulcan_cfg.use_lowT_limit_rates:
    data_var = rate.lim_lowT_rates(data_var, data_atm)

data_var = rate.rev_rate(data_var, data_atm)
data_var = rate.remove_rate(data_var)

ini_abun = build_atm.InitialAbun()
data_var = ini_abun.ini_y(data_var, data_atm)
data_var = ini_abun.ele_sum(data_var)

data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)

# === JAX-accelerated solver (Ros2 only) ===
solver_str = vulcan_cfg.ode_solver
if solver_str != "Ros2":
    raise NotImplementedError(
        f"VULCAN-JAX targets the Ros2 solver only; got ode_solver={solver_str!r}. "
        "Use VULCAN-master for other solvers."
    )
solver = op_jax.Ros2JAX()

if vulcan_cfg.use_photo:
    rate.make_bins_read_cross(data_var, data_atm)
    make_atm.read_sflux(data_var, data_atm)
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    data_var = rate.remove_rate(data_var)

integ = outer_loop.OuterLoop(solver, output)
solver.naming_solver(data_para)

print(f"VULCAN-JAX starting integration at t=0, dt={data_var.dt:.2e}")
integ(data_var, data_atm, data_para, make_atm)

print(f"VULCAN-JAX done. Saving output to {vulcan_cfg.output_dir}{vulcan_cfg.out_name}")
output.save_out(data_var, data_atm, data_para, dname)

print(f"Total wall time: {time.time() - data_para.start_time:.1f}s")
