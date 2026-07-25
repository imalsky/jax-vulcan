"""Verify every species' molecular mass is consistent with its atom counts.

`thermo/all_compose.txt` lists, per species, the per-element atom counts and a
molecular `mass`. A transcription error in the `mass` column (e.g. `NH3_l_s`
once carried NH2's mass, 16.023 instead of 17.031) silently corrupts the mean
molecular weight, settling velocity, and molecular diffusion for that species.

This test recomputes each molecular mass from the atom counts and standard
atomic weights and asserts agreement, and separately checks that every
condensate (`*_l_s` / `*_s`) has the same mass as its gas-phase counterpart.

Pure data check — no FastChem, no VULCAN-master, no integration.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

# IUPAC standard atomic weights (amu); 'e' is the electron mass. The table uses
# slightly rounded conventions (e.g. O = 16.0), so a 0.12 amu tolerance absorbs
# the convention spread accumulated across a molecule while still catching a
# whole-atom (>~1 amu) transcription error.
_ATOMIC_MASS = {
    "H": 1.008,
    "O": 15.999,
    "C": 12.011,
    "He": 4.0026,
    "N": 14.007,
    "S": 32.06,
    "P": 30.974,
    "Na": 22.990,
    "K": 39.098,
    "Si": 28.085,
    "Fe": 55.845,
    "Ar": 39.948,
    "Ti": 47.867,
    "V": 50.942,
    "Mg": 24.305,
    "Ca": 40.078,
    "e": 0.000549,
}
_TOL = 0.12


def main() -> int:
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    compo = np.genfromtxt(
        resolve_data_path(vulcan_cfg.com_file), names=True, dtype=None, encoding=None
    )
    elems = [c for c in compo.dtype.names if c not in ("species", "mass")]
    species = [str(s) for s in compo["species"]]
    listed = np.asarray(compo["mass"], dtype=np.float64)
    counts = np.array(
        [[float(compo[e][i]) for e in elems] for i in range(len(species))]
    )
    ref = np.array([_ATOMIC_MASS[e] for e in elems])
    expected = counts @ ref

    ok = True
    bad = [
        (species[i], listed[i], expected[i])
        for i in range(len(species))
        if abs(listed[i] - expected[i]) > _TOL
    ]
    if bad:
        ok = False
        print(f"FAIL: {len(bad)} species mass inconsistent with atom counts:")
        for sp, ml, me in sorted(bad, key=lambda t: -abs(t[1] - t[2])):
            print(
                f"  {sp:12s} listed={ml:9.3f}  expected={me:9.3f}  diff={ml - me:+.3f}"
            )
    else:
        print(
            f"OK: all {len(species)} species masses consistent with atom counts (tol {_TOL})"
        )

    # Condensates must share the mass of their gas-phase counterpart.
    mass_by_sp = dict(zip(species, listed))
    for sp in species:
        for suffix in ("_l_s", "_s", "_l"):
            if sp.endswith(suffix):
                gas = sp[: -len(suffix)]
                if gas in mass_by_sp and abs(mass_by_sp[sp] - mass_by_sp[gas]) > 1e-6:
                    ok = False
                    print(
                        f"FAIL: condensate {sp} mass {mass_by_sp[sp]} != gas {gas} "
                        f"mass {mass_by_sp[gas]}"
                    )
                break

    # Explicit guard on the previously-wrong row.
    if abs(mass_by_sp.get("NH3_l_s", 0.0) - 17.031) > 1e-6:
        ok = False
        print(f"FAIL: NH3_l_s mass {mass_by_sp.get('NH3_l_s')} != 17.031")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
