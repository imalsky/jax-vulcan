"""Validate gibbs.py against VULCAN-master/chem_funs.Gibbs and gibbs_sp.

Compares:
  - gibbs_sp_vector(coeffs, T)[species_idx, :]   vs   chem_funs.gibbs_sp(name, T)
  - K_eq_array(net, g_sp, T)[i, :]               vs   chem_funs.Gibbs(i, T)
  - Final reverse-rate array                      vs   VULCAN's data_var.k after rev_rate
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Oracle test: requires VULCAN-master sibling for the upstream
# chem_funs.Gibbs / chem_funs.gibbs_sp reference. Skip cleanly when absent.
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"VULCAN-master oracle absent at {VULCAN_MASTER}; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )

warnings.filterwarnings("ignore")


def main() -> int:
    sys.path.insert(0, str(VULCAN_MASTER))
    import vulcan_cfg                       # noqa
    import store, op                        # noqa
    from atm_setup import Atm
    import chem_funs                         # noqa

    import network as net_mod
    import rates as rates_mod
    import gibbs as gibbs_mod

    # === 1. VULCAN setup (build atm + read forward + reverse rates) ===
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    if getattr(vulcan_cfg, "use_lowT_limit_rates", False):
        data_var = rate.lim_lowT_rates(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)

    T = np.asarray(data_atm.Tco, dtype=np.float64)
    M = np.asarray(data_atm.M, dtype=np.float64)

    # === 2. JAX-side computation ===
    net = net_mod.parse_network(vulcan_cfg.network)
    coeffs, present = gibbs_mod.load_nasa9(net.species, ROOT / "thermo")
    g_sp = gibbs_mod.gibbs_sp_vector(coeffs, T)
    K_eq = gibbs_mod.K_eq_array(net, g_sp, T)
    k_fwd = rates_mod.compute_forward_k(net, T, M)
    k_full = gibbs_mod.fill_reverse_k(
        net, k_fwd, K_eq, remove_list=vulcan_cfg.remove_list
    )

    # === 3. Compare gibbs_sp per species ===
    print(f"NASA-9 coverage: {int(present.sum())}/{net.ni} species have files")
    missing = [sp for sp, p in zip(net.species, present) if not p]
    if missing:
        print(f"  Missing: {missing}")

    max_err_gsp = 0.0
    for j, sp in enumerate(net.species):
        if not present[j]:
            continue
        ref = chem_funs.gibbs_sp(sp, T)
        ours = g_sp[j]
        err = np.max(np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-30))
        if err > max_err_gsp:
            max_err_gsp = err
    print(f"gibbs_sp max relative error: {max_err_gsp:.3e}")

    # === 4. Compare K_eq per reaction ===
    max_err_K = 0.0
    n_compared = 0
    n_fail = 0
    for i in range(1, net.stop_rev_indx, 2):
        if (
            net.is_photo[i] or net.is_ion[i]
            or net.is_conden[i] or net.is_radiative[i]
        ):
            continue
        try:
            ref = chem_funs.Gibbs(i, T)
        except KeyError:
            continue
        ours = K_eq[i]
        # K_eq spans many orders of magnitude; use relative error
        err = np.max(np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-300))
        if err > max_err_K:
            max_err_K = err
        if err > 1e-8:
            n_fail += 1
            if n_fail <= 5:
                print(
                    f"  K_eq fail i={i}: {net.Rf.get(i)!r}  err={err:.2e}  "
                    f"ours_max={ours.max():.3e}  ref_max={ref.max():.3e}"
                )
        n_compared += 1
    print(f"K_eq compared: {n_compared}  fails: {n_fail}  max relative error: {max_err_K:.3e}")

    # === 5. Compare reverse k against VULCAN's data_var.k ===
    max_err_rev = 0.0
    n_rev = 0
    n_rev_fail = 0
    for i in range(2, net.stop_rev_indx, 2):
        if i not in data_var.k:
            continue
        ref = np.asarray(data_var.k[i], dtype=np.float64)
        ours = k_full[i]
        # Both can be very small or zero; use relative error with floor
        err = np.max(np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-300))
        if err > max_err_rev:
            max_err_rev = err
        if err > 1e-8 and ref.max() > 1e-50:
            n_rev_fail += 1
            if n_rev_fail <= 5:
                print(
                    f"  rev k fail i={i}: {net.Rf.get(i-1)!r}  err={err:.2e}"
                )
        n_rev += 1
    print(f"reverse k compared: {n_rev}  fails: {n_rev_fail}  max relative error: {max_err_rev:.3e}")

    # === 6. Check that beyond stop_rev_indx all reverses are zero ===
    bad_zero = 0
    for i in range(net.stop_rev_indx + 1, net.nr + 1, 2):
        if k_full[i].max() != 0.0:
            bad_zero += 1
    print(f"reverses beyond stop_rev_indx that should be zero: {bad_zero} non-zero")

    print()
    ok = (
        max_err_gsp < 1e-10
        and max_err_K < 1e-8
        and max_err_rev < 1e-8
        and bad_zero == 0
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Run the master comparison in a fresh Python process."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(main())
