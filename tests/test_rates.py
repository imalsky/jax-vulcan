"""Validate VULCAN-JAX rates_jax.py against VULCAN-master's ReadRate.read_rate.

Runs VULCAN's startup pipeline until var.k holds the forward rates, then
compares VULCAN-JAX's compute_forward_k on the same T/M atmosphere.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Imports relative to VULCAN-JAX/
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # ensure relative paths in vulcan_cfg.py resolve

from oracle import oracle_dir_or_skip  # noqa: E402

# The parent verifies the pin and passes a temporary copy (oracle.oracle_dir_or_skip).
VULCAN_MASTER = oracle_dir_or_skip("this rate-array comparison")

# Suppress SciPy / matplotlib chatter
warnings.filterwarnings("ignore")


def main() -> int:
    os.chdir(VULCAN_MASTER)
    sys.path.append(str(VULCAN_MASTER))
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    from vulcan_jax.atm_setup import Atm
    import op

    from vulcan_jax.state import _Variables, _AtmData

    import vulcan_jax.network as net_mod  # VULCAN-JAX
    import vulcan_jax.rates_jax as rates_mod  # VULCAN-JAX

    # === 1. Run VULCAN setup until var.k is populated ===
    data_var = _Variables()
    data_atm = _AtmData()

    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)

    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    os.chdir(ROOT)

    T = np.asarray(data_atm.Tco, dtype=np.float64)
    M = np.asarray(data_atm.M, dtype=np.float64)

    # === 2. Run VULCAN-JAX rate computation on the same atmosphere ===
    net = net_mod.parse_network(vulcan_cfg.network)

    k_jax = np.asarray(rates_mod.compute_forward_k(net, T, M))

    # === 3. Compare ===
    n_fail = 0
    max_relerr = 0.0
    worst_i = -1

    # Only forward, non-photo/conden/ion/radiative reactions are populated by
    # rates_mod.compute_forward_k. VULCAN's read_rate populates all photo,
    # condensation, radiative slots with zeros. Both should agree.
    for i in range(1, net.nr + 1, 2):
        if i not in data_var.k:
            continue
        v_vul = np.asarray(data_var.k[i], dtype=np.float64)
        v_jax = k_jax[i]
        if v_vul.shape != v_jax.shape:
            print(f"  shape mismatch at i={i}: vulcan={v_vul.shape} jax={v_jax.shape}")
            n_fail += 1
            continue

        # Compute relative error, robust to zeros
        denom = np.maximum(np.abs(v_vul), 1e-300)
        relerr = np.abs(v_jax - v_vul) / denom
        max_e = float(relerr.max())
        if max_e > max_relerr:
            max_relerr = max_e
            worst_i = i

        # Strict tolerance for non-zero rates
        if v_vul.max() > 0:
            if not max_e <= 1e-10:
                n_fail += 1
                if n_fail <= 5:
                    print(
                        f"  FAIL i={i}: {net.Rf.get(i, '?')!r}  "
                        f"max relerr={max_e:.2e}  "
                        f"vulcan max={v_vul.max():.3e}  jax max={v_jax.max():.3e}"
                    )
        else:
            # both should be zero; check absolute
            if not np.abs(v_jax).max() < 1e-300:
                n_fail += 1
                if n_fail <= 5:
                    print(
                        f"  FAIL i={i} (zero rate expected): "
                        f"vulcan max=0 jax max={v_jax.max():.3e}"
                    )

    print(
        f"  Max relative error: {max_relerr:.3e} (at i={worst_i}, "
        f"{net.Rf.get(worst_i, '?')!r})"
    )

    print()
    if n_fail == 0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


@pytest.mark.master_serial
def test_main():
    """Run the master comparison in a fresh Python process."""
    from oracle import run_oracle_subprocess

    run_oracle_subprocess(__file__, "vulcan2_ncho",
                          "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    sys.exit(main())
