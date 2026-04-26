"""Validate VULCAN-JAX initial-abundance pipeline matches VULCAN-master.

Runs InitialAbun.ini_y through both versions and compares the resulting
y[nz, ni] arrays.  Uses the configured `ini_mix` (typically 'EQ' which
shells out to fastchem_vulcan).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def main() -> int:
    import build_atm
    import store
    import vulcan_cfg
    import chem_funs as cf_jax

    print(f"VULCAN-JAX chem_funs: ni={cf_jax.ni} nr={cf_jax.nr}, ini_mix={vulcan_cfg.ini_mix}")

    # === JAX side ===
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)

    ini_abun = build_atm.InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)

    jax_y = np.asarray(data_var.y).copy()
    jax_ymix = np.asarray(data_var.ymix).copy()
    jax_atom_ini = dict(data_var.atom_ini)

    print(f"  y shape: {jax_y.shape}, y range [{jax_y.min():.3e}, {jax_y.max():.3e}]")
    print(f"  ymix shape: {jax_ymix.shape}")
    print(f"  atom_ini: {jax_atom_ini}")

    # === VULCAN-master side ===
    for mod_name in ("build_atm", "store", "chem_funs", "vulcan_cfg"):
        sys.modules.pop(mod_name, None)
    VULCAN_MASTER = ROOT.parent / "VULCAN-master"
    sys.path.insert(0, str(VULCAN_MASTER))

    import build_atm as ba_v
    import store as st_v
    import vulcan_cfg as cfg_v

    data_var2 = st_v.Variables()
    data_atm2 = st_v.AtmData()
    make_atm2 = ba_v.Atm()
    data_atm2 = make_atm2.f_pico(data_atm2)
    data_atm2 = make_atm2.load_TPK(data_atm2)
    if cfg_v.use_condense:
        make_atm2.sp_sat(data_atm2)

    ini_v = ba_v.InitialAbun()
    data_var2 = ini_v.ini_y(data_var2, data_atm2)
    data_var2 = ini_v.ele_sum(data_var2)

    vul_y = np.asarray(data_var2.y)
    vul_ymix = np.asarray(data_var2.ymix)
    vul_atom_ini = dict(data_var2.atom_ini)

    # === Compare ===
    ok = True
    if jax_y.shape != vul_y.shape:
        print(f"FAIL y shape: jax={jax_y.shape} vul={vul_y.shape}")
        ok = False
    else:
        # Per-species check
        max_relerr = 0.0
        max_sp = -1
        for j in range(jax_y.shape[1]):
            denom = np.maximum(np.abs(vul_y[:, j]), 1e-30)
            err = np.max(np.abs(jax_y[:, j] - vul_y[:, j]) / denom)
            if err > max_relerr:
                max_relerr = err
                max_sp = j
        if max_relerr < 1e-10:
            print(f"OK   y: max relerr = {max_relerr:.2e}")
        else:
            print(f"FAIL y: max relerr = {max_relerr:.2e} for species {cf_jax.spec_list[max_sp] if max_sp >= 0 else '?'}")
            ok = False

    for atom in jax_atom_ini:
        if atom not in vul_atom_ini:
            print(f"  atom {atom} missing from VULCAN")
            continue
        diff = abs(jax_atom_ini[atom] - vul_atom_ini[atom]) / abs(vul_atom_ini[atom])
        if diff < 1e-12:
            print(f"OK   atom_ini[{atom}] = {jax_atom_ini[atom]:.4e} (relerr {diff:.2e})")
        else:
            print(f"FAIL atom_ini[{atom}]: jax={jax_atom_ini[atom]:.4e} vul={vul_atom_ini[atom]:.4e}")
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
