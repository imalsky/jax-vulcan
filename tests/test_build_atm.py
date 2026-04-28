"""Validate VULCAN-JAX build_atm + chem_funs shim against VULCAN-master.

Runs the same atmospheric setup pipeline through both versions:
- VULCAN-JAX uses its own chem_funs shim (which sources ni/nr/spec_list from
  `network.parse_network`).
- VULCAN-master uses its full auto-generated chem_funs.py.

Checks that all AtmData fields agree element-wise.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

# IMPORTANT: VULCAN-JAX must come FIRST in sys.path so its chem_funs shim
# wins over VULCAN-master's full chem_funs.py.
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def main() -> int:
    # Import VULCAN-JAX's atm_setup + store + chem_funs (shim)
    from atm_setup import Atm           # VULCAN-JAX (Phase 20)
    import store                        # VULCAN-JAX copy
    import chem_funs as cf_jax          # VULCAN-JAX shim
    import vulcan_cfg

    print(f"VULCAN-JAX chem_funs: ni={cf_jax.ni} nr={cf_jax.nr}")
    print(f"  spec_list head: {cf_jax.spec_list[:5]}")

    # === Run JAX-side pipeline ===
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)

    print(f"  pco shape: {data_atm.pco.shape}, P range [{data_atm.pco.min():.3e}, {data_atm.pco.max():.3e}]")
    print(f"  Tco shape: {data_atm.Tco.shape}, T range [{data_atm.Tco.min():.1f}, {data_atm.Tco.max():.1f}]")
    print(f"  Kzz shape: {data_atm.Kzz.shape}, Kzz range [{data_atm.Kzz.min():.3e}, {data_atm.Kzz.max():.3e}]")
    print(f"  M shape:   {data_atm.M.shape}, M range [{data_atm.M.min():.3e}, {data_atm.M.max():.3e}]")

    # Save JAX-side fields for cross-comparison
    jax_pco = np.asarray(data_atm.pco).copy()
    jax_Tco = np.asarray(data_atm.Tco).copy()
    jax_Kzz = np.asarray(data_atm.Kzz).copy()
    jax_M = np.asarray(data_atm.M).copy()
    jax_pico = np.asarray(data_atm.pico).copy()
    jax_n0 = np.asarray(data_atm.n_0).copy()

    # === Run VULCAN-master pipeline ===
    # Reload by removing JAX modules from sys.modules and pointing sys.path at VULCAN-master FIRST.
    for mod_name in ("build_atm", "store", "chem_funs", "vulcan_cfg"):
        sys.modules.pop(mod_name, None)
    VULCAN_MASTER = ROOT.parent / "VULCAN-master"
    # Move VULCAN-master to the front
    sys.path.insert(0, str(VULCAN_MASTER))

    import build_atm as ba_vul          # VULCAN-master
    import store as st_vul              # VULCAN-master
    import chem_funs as cf_vul          # VULCAN-master (full version)
    import vulcan_cfg as cfg_vul

    print(f"\nVULCAN-master chem_funs: ni={cf_vul.ni} nr={cf_vul.nr}")

    data_var2 = st_vul.Variables()
    data_atm2 = st_vul.AtmData()
    make_atm2 = ba_vul.Atm()
    data_atm2 = make_atm2.f_pico(data_atm2)
    data_atm2 = make_atm2.load_TPK(data_atm2)
    if cfg_vul.use_condense:
        make_atm2.sp_sat(data_atm2)

    vul_pco = np.asarray(data_atm2.pco)
    vul_Tco = np.asarray(data_atm2.Tco)
    vul_Kzz = np.asarray(data_atm2.Kzz)
    vul_M = np.asarray(data_atm2.M)
    vul_pico = np.asarray(data_atm2.pico)
    vul_n0 = np.asarray(data_atm2.n_0)

    # === Compare ===
    ok = True
    fields = [
        ("pco", jax_pco, vul_pco),
        ("pico", jax_pico, vul_pico),
        ("Tco", jax_Tco, vul_Tco),
        ("Kzz", jax_Kzz, vul_Kzz),
        ("M", jax_M, vul_M),
        ("n_0", jax_n0, vul_n0),
    ]
    for name, a, b in fields:
        if a.shape != b.shape:
            print(f"FAIL {name}: shape jax={a.shape} vul={b.shape}")
            ok = False
            continue
        diff = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
        if diff < 1e-12:
            print(f"OK   {name}: max relerr = {diff:.2e}")
        else:
            print(f"FAIL {name}: max relerr = {diff:.2e}")
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
