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

# VULCAN-JAX is standalone — it imports its vendored modules from this directory
# (no sibling-directory sys.path append).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Cwd at the script's directory so relative paths in vulcan_cfg.py resolve
os.chdir(ROOT)

# chem_funs.py is JAX-native; the -n flag is accepted as a no-op for back-compat.
print("Using JAX-native chem_funs (no make_chem_funs.py step required)")

import numpy as np

import store
from atm_setup import Atm
from ini_abun import InitialAbun
import legacy_io as op  # vendored op.ReadRate + op.Output (Phase A)
import chem_funs

import vulcan_cfg
from phy_const import kb, Navo

import op_jax       # JAX photo helpers
import outer_loop   # JAX outer integration loop (Phase 10.1+)
from runtime_validation import validate_runtime_config
from state import load_stellar_flux  # Phase 19 — host-side sflux read
import rates as _rates       # Phase 22a-2: build_rate_array entry point
import photo_setup as _photo_setup  # Phase 22b: populate_photo_arrays

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
# Phase 19: pre-load stellar flux on the host so `Variables.__init__`
# doesn't touch disk. The explicit call site keeps the pre-loop pytree's
# "no host I/O at construction" contract honest.
stellar_flux = load_stellar_flux(vulcan_cfg)
data_var = store.Variables(stellar_flux=stellar_flux)
data_atm = store.AtmData()
data_para = store.Parameters()

data_para.start_time = time.time()

make_atm = Atm()
output = op.Output()

validate_runtime_config(vulcan_cfg, ROOT)

# Save the config copy
output.save_cfg(dname)

# Build the atmosphere (mirrors vulcan.py:118-155)
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
if vulcan_cfg.use_condense:
    make_atm.sp_sat(data_atm)

rate = op.ReadRate()
# Phase 22a-2: read_rate is still called for metadata population
# (var.Rf, var.pho_rate_index, var.n_branch, var.photo_sp, etc. that
# downstream photo / conden / .vul writer paths read off `data_var`).
# The rate VALUES it produces are immediately overwritten below by
# `build_rate_array`, which subsumes rev_rate / remove_rate /
# lim_lowT_rates into one JAX-native NumPy pipeline.
data_var = rate.read_rate(data_var, data_atm)

# Phase 22a-2: forward (Arrhenius/Lindemann/Troe) -> optional low-T
# caps -> reverse via Gibbs -> remove_list, all in one
# NumPy-on-Network call. Bit-exact (<=1e-13) replacement for the legacy
# rev_rate / remove_rate / lim_lowT_rates chain (verified in
# tests/test_read_rate.py::test_build_rate_array_matches_legacy_hd189).
# Phase 22d: writes the dense (nr+1, nz) array directly to `var.k_arr`;
# the legacy dict-keyed `var.k` surface was retired and is synthesized
# only at .vul write time for downstream plot_py/ scripts.
_network = _rates.setup_var_k(vulcan_cfg, data_var, data_atm)

ini_abun = InitialAbun()
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
    # Phase 22b set up the host-side dict surface (`var.cross[sp]`, ...).
    # Phase 22e adds the dense `PhotoStaticInputs` pytree alongside it
    # and stashes it on the solver so `compute_tau` / `compute_flux` /
    # `compute_J` / `compute_Jion` derive their `PhotoData` /
    # `PhotoJData` from the static instead of the dicts. The dict path
    # is still produced (and consumed by the .vul writer) until Phase
    # 22e step 9 retires the dict writes.
    _photo_setup.populate_photo_arrays(data_var, data_atm)
    make_atm.read_sflux(data_var, data_atm)
    photo_static = _photo_setup._build_photo_static_dense(data_var, data_atm)
    photo_static = photo_static.with_din12_indx(int(data_var.sflux_din12_indx))
    solver._photo_static = photo_static
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    if vulcan_cfg.use_ion:
        solver.compute_Jion(data_var, data_atm)
    # Apply `remove_list` on the dense `var.k_arr` after compute_J /
    # compute_Jion overwrite the photodissociation rows. Phase 22d
    # writers target `var.k_arr` directly; the legacy `var.k` dict was
    # retired.
    _rates.apply_photo_remove(vulcan_cfg, data_var, _network, data_atm)
else:
    photo_static = None

integ = outer_loop.OuterLoop(solver, output)
solver.naming_solver(data_para)

print(f"VULCAN-JAX starting integration at t=0, dt={data_var.dt:.2e}")
integ(data_var, data_atm, data_para, make_atm)

print(f"VULCAN-JAX done. Saving output to {vulcan_cfg.output_dir}{vulcan_cfg.out_name}")
output.save_out(data_var, data_atm, data_para, dname, photo_static=photo_static)

# Phase 16: post-run plotting hooks (master op.py:902-934 equivalents).
# These fire only when the corresponding cfg flags are on; the chunked
# driver inside outer_loop already handled live plotting / movie frames.
if getattr(vulcan_cfg, "use_plot_end", False):
    output.plot_end(data_var, data_atm, data_para)
if getattr(vulcan_cfg, "use_plot_evo", False) and getattr(vulcan_cfg, "save_evolution", False):
    output.plot_evo(data_var, data_atm)

print(f"Total wall time: {time.time() - data_para.start_time:.1f}s")
