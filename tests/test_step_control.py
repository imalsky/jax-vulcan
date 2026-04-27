"""Validate that outer_loop's pure-JAX step-control helpers
(_compute_atom_loss, clip_fn from _make_clip_fn, _step_size) produce results
matching VULCAN-master's op.ODESolver.loss / clip / step_size on a real HD189
state.

Replaces the pre-Phase-10.1 test that called op_jax.Ros2JAX.{clip,loss,...};
those methods moved into outer_loop as pure-JAX equivalents and op_jax.Ros2JAX
shrank to a photo-only adapter.
"""
from __future__ import annotations

import copy
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Oracle test: requires VULCAN-master sibling for the upstream
# op.ODESolver.loss / clip / step_size reference. Skip when absent.
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"VULCAN-master oracle absent at {VULCAN_MASTER}; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )
sys.path.append(str(VULCAN_MASTER))

warnings.filterwarnings("ignore")


def main() -> int:
    import vulcan_cfg
    import store, build_atm, op
    import outer_loop

    # --- Build state ---
    data_var = store.Variables()
    data_atm = store.AtmData()
    data_para = store.Parameters()
    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    data_var = rate.remove_rate(data_var)
    ini_abun = build_atm.InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)

    # NumPy reference (op.ODESolver methods, inherited by op.Ros2)
    solver_ref = op.Ros2()

    # --- Build outer_loop's static aux exactly as OuterLoop would ---
    # We exercise standalone helpers (_compute_atom_loss / _make_clip_fn /
    # _step_size) — no runner needed, so skip _ensure_runner (which would
    # also build the photo branch and require var.nbin etc., not set up
    # in this minimal pre-loop fixture).
    import op_jax
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())
    integ._statics = integ._build_statics(data_var, data_atm)
    statics = integ._statics
    compo_arr_jnp = statics.compo_arr
    atom_ini_arr_jnp = statics.atom_ini_arr

    ok = True

    # === _compute_atom_loss vs op.ODESolver.loss ===
    # Compare atom_sum (large, well-conditioned) rather than atom_loss
    # (= (atom_sum - atom_ini)/atom_ini, near-zero in the initial state
    # where atom_sum == atom_ini and tiny differences blow up relative error).
    # Perturb y[0,0] slightly so atom_loss is non-trivial too.
    y_perturbed = data_var.y.copy()
    y_perturbed[0, 0] *= 1.001
    var_ref = copy.deepcopy(data_var)
    var_ref.y = y_perturbed.copy()
    var_ref = solver_ref.loss(var_ref)
    atom_sum_ref = np.array([var_ref.atom_sum[a] for a in integ._atom_order])
    # JAX compute_atom_loss returns atom_loss; reverse to atom_sum for compare
    atom_loss_jax = np.asarray(
        outer_loop._compute_atom_loss(jnp.asarray(y_perturbed),
                                      compo_arr_jnp, atom_ini_arr_jnp)
    )
    atom_ini_arr_np = np.asarray(atom_ini_arr_jnp)
    atom_sum_jax = (atom_loss_jax + 1.0) * atom_ini_arr_np
    err_sum = float(np.max(np.abs(atom_sum_ref - atom_sum_jax)
                           / np.maximum(np.abs(atom_sum_ref), 1e-30)))
    print(f"loss        atom_sum max relerr: {err_sum:.3e}")
    if err_sum > 1e-12:
        print("FAIL loss relerr too large")
        ok = False

    # === clip_fn vs op.ODESolver.clip ===
    # Inject sentinel small / negative values, then compare clipped y / ymix
    var_ref = copy.deepcopy(data_var)
    var_ref.y[5, 10] = 1e-50
    var_ref.y[7, 20] = -1e-30
    var_ref = solver_ref.loss(var_ref)
    var_ref.atom_loss_prev = dict(var_ref.atom_loss)
    para_ref = copy.deepcopy(data_para)
    var_ref, para_ref = solver_ref.clip(var_ref, para_ref, data_atm)

    # JAX clip on the same input
    y_in = np.copy(data_var.y)
    y_in[5, 10] = 1e-50
    y_in[7, 20] = -1e-30
    ymix_in = data_var.ymix.copy()
    clip_fn = outer_loop._make_clip_fn(
        non_gas_present=integ._non_gas_present,
        gas_indx_mask=jnp.ones(integ._compo_arr.shape[0], dtype=bool),
        mtol=integ.mtol, pos_cut=vulcan_cfg.pos_cut, nega_cut=vulcan_cfg.nega_cut,
    )
    y_clip_j, ymix_clip_j, small_inc_j, nega_inc_j = clip_fn(jnp.asarray(y_in),
                                                              jnp.asarray(ymix_in))
    y_clip_j = np.asarray(y_clip_j); ymix_clip_j = np.asarray(ymix_clip_j)
    y_diff = float(np.max(np.abs(var_ref.y - y_clip_j)))
    ymix_diff = float(np.max(np.abs(var_ref.ymix - ymix_clip_j)))
    print(f"clip        y diff: {y_diff:.3e}, ymix diff: {ymix_diff:.3e}")
    print(f"            small_y: ref={para_ref.small_y:.3e} jax={float(small_inc_j):.3e}")
    print(f"            nega_y:  ref={para_ref.nega_y:.3e} jax={float(nega_inc_j):.3e}")
    if y_diff > 1e-30 or ymix_diff > 1e-15:
        print("FAIL clip y or ymix mismatch")
        ok = False
    if (abs(para_ref.small_y - float(small_inc_j)) > 1e-30
            or abs(para_ref.nega_y - float(nega_inc_j)) > 1e-30):
        print("FAIL clip diagnostic counters mismatch")
        ok = False

    # === _step_size vs op.Ros2.step_size ===
    var_test = copy.deepcopy(data_var)
    var_test.dt = 1.0
    para_test = copy.deepcopy(data_para)
    para_test.delta = 1e-3
    var_test = solver_ref.step_size(var_test, para_test)
    dt_jax = float(outer_loop._step_size(
        jnp.float64(1.0), jnp.float64(1e-3),
        rtol=vulcan_cfg.rtol,
        dt_var_min=vulcan_cfg.dt_var_min, dt_var_max=vulcan_cfg.dt_var_max,
        dt_min=vulcan_cfg.dt_min, dt_max=vulcan_cfg.dt_max,
    ))
    print(f"step_size   ref dt={var_test.dt:.6e}, jax dt={dt_jax:.6e}")
    if abs(var_test.dt - dt_jax) / max(abs(var_test.dt), 1e-30) > 1e-13:
        print("FAIL step_size dt mismatch")
        ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
