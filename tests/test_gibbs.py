"""Validate the NASA-9 reverse-rate path against VULCAN-master's rev_rate.

Runs master's setup until `data_var.k` holds forward AND reverse rates, then
compares `rates_jax.build_rate_array` reverse slot for reverse slot. This is
the only check that the Gibbs energies, the (kB T/P0)^dn factor and the
forward/reverse pairing are right; a self-comparison against another copy of
the same polynomial would prove nothing.
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

from oracle import oracle_dir_or_skip  # noqa: E402

# The parent verifies the pin and passes a temporary copy (oracle.oracle_dir_or_skip).
VULCAN_MASTER = oracle_dir_or_skip("this Gibbs comparison")

warnings.filterwarnings("ignore")


def main() -> int:
    os.chdir(VULCAN_MASTER)
    sys.path.insert(0, str(VULCAN_MASTER))
    import jax.numpy as jnp

    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()  # noqa
    import store, op  # noqa
    from vulcan_jax.atm_setup import Atm

    import vulcan_jax.network as net_mod
    import vulcan_jax.rates_jax as rates_jax
    import vulcan_jax.gibbs as gibbs_mod

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
    os.chdir(ROOT)

    T = np.asarray(data_atm.Tco, dtype=np.float64)
    M = np.asarray(data_atm.M, dtype=np.float64)

    # === 2. VULCAN-JAX computation on the same atmosphere ===
    net = net_mod.parse_network(vulcan_cfg.network)
    coeffs, present = gibbs_mod.load_nasa9(net.species, "thermo")
    k_full = np.asarray(
        rates_jax.build_rate_array(
            net,
            jnp.asarray(T),
            jnp.asarray(M),
            coeffs,
            remove_list=vulcan_cfg.remove_list,
            use_lowT_caps=bool(getattr(vulcan_cfg, "use_lowT_limit_rates", False)),
        )
    )
    missing = [sp for sp, p in zip(net.species, present) if not p]
    if missing:
        print(f"  Missing: {missing}")

    # === 3. Compare reverse k against VULCAN's data_var.k ===
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
                print(f"  rev k fail i={i}: {net.Rf.get(i - 1)!r}  err={err:.2e}")
        n_rev += 1
    print(
        f"reverse k compared: {n_rev}  fails: {n_rev_fail}  max relative error: {max_err_rev:.3e}"
    )

    # === 4. Check that beyond stop_rev_indx all reverses are zero ===
    bad_zero = 0
    for i in range(net.stop_rev_indx + 1, net.nr + 1, 2):
        if k_full[i].max() != 0.0:
            bad_zero += 1
    print(f"reverses beyond stop_rev_indx that should be zero: {bad_zero} non-zero")

    print()
    ok = max_err_rev < 1e-8 and bad_zero == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Run the master comparison in a fresh Python process."""
    from oracle import run_oracle_subprocess

    run_oracle_subprocess(__file__, "vulcan2_ncho",
                          "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    sys.exit(main())
