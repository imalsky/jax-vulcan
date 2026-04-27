"""Validate VULCAN-JAX rates.py against VULCAN-master's ReadRate.read_rate.

Runs VULCAN's startup pipeline up to the point where var.k is filled with
forward rates, then compares to VULCAN-JAX's compute_forward_k for the same
T/M atmosphere. Required for Task #5 validation.

Run from VULCAN-JAX/:
    python tests/test_rates.py
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
os.chdir(ROOT)              # ensure relative paths in vulcan_cfg.py resolve
sys.path.insert(0, str(ROOT))

# Oracle test: requires VULCAN-master sibling for the upstream `op` reference.
# Skip cleanly when absent so VULCAN-JAX-only checkouts don't see a hard failure.
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"VULCAN-master oracle absent at {VULCAN_MASTER}; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )
sys.path.append(str(VULCAN_MASTER))

# Suppress SciPy / matplotlib chatter
warnings.filterwarnings("ignore")


def main() -> int:
    import vulcan_cfg                       # noqa: E402
    import store                            # noqa: E402
    import build_atm                        # noqa: E402
    import op                               # noqa: E402

    import network as net_mod               # VULCAN-JAX
    import rates as rates_mod               # VULCAN-JAX

    # === 1. Run VULCAN setup until var.k is populated ===
    data_var = store.Variables()
    data_atm = store.AtmData()

    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)

    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)

    T = np.asarray(data_atm.Tco, dtype=np.float64)
    M = np.asarray(data_atm.M, dtype=np.float64)
    nz = T.shape[0]
    print(f"Atmosphere: nz={nz}, T range [{T.min():.1f}, {T.max():.1f}] K, "
          f"M range [{M.min():.2e}, {M.max():.2e}] cm^-3")

    # === 2. Run VULCAN-JAX rate computation on the same atmosphere ===
    net = net_mod.parse_network(vulcan_cfg.network)
    print(f"Network: ni={net.ni}, nr={net.nr}")

    k_jax = rates_mod.compute_forward_k(net, T, M)

    # === 3. Compare ===
    n_compared = 0
    n_pass = 0
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
            if max_e <= 1e-10:
                n_pass += 1
            else:
                n_fail += 1
                if n_fail <= 5:
                    print(
                        f"  FAIL i={i}: {net.Rf.get(i, '?')!r}  "
                        f"max relerr={max_e:.2e}  "
                        f"vulcan max={v_vul.max():.3e}  jax max={v_jax.max():.3e}"
                    )
        else:
            # both should be zero; check absolute
            if np.abs(v_jax).max() < 1e-300:
                n_pass += 1
            else:
                n_fail += 1
                if n_fail <= 5:
                    print(
                        f"  FAIL i={i} (zero rate expected): "
                        f"vulcan max=0 jax max={v_jax.max():.3e}"
                    )
        n_compared += 1

    print()
    print(f"Compared {n_compared} forward reactions")
    print(f"  Pass: {n_pass}")
    print(f"  Fail: {n_fail}")
    print(f"  Max relative error: {max_relerr:.3e} (at i={worst_i}, "
          f"{net.Rf.get(worst_i, '?')!r})")

    print()
    if n_fail == 0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
