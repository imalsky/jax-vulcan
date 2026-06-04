"""Anchor the H2S saturation-pressure unit conversion to physical references.

The Giauque & Blue (1936) Antoine fit for H2S outputs pressure in cmHg, so the
cgs conversion is `* 0.01333 (cmHg->bar) * 1e6 (bar->dyne/cm^2)`. A prior
version multiplied by 0.001333 (the mmHg factor), making the saturation
pressure exactly 10x too low. This test pins the corrected value against two
independent literature anchors so the factor cannot silently regress:

  * Normal boiling point T = 212.8 K, where P(H2S) = 1 atm by definition.
  * Triple point T = 187.66 K, where P(H2S) ~ 0.233 bar (literature).

A unit test only — no FastChem, no VULCAN-master, no integration.
"""

from __future__ import annotations

import numpy as np

ATM_CGS = 1.01325e6  # 1 atm in dyne/cm^2
BAR_CGS = 1.0e6  # 1 bar in dyne/cm^2


def main() -> int:
    from vulcan_jax.atm_setup import compute_sat_p

    T_boil = 212.8  # K, H2S normal boiling point (P = 1 atm)
    T_triple = 187.66  # K, H2S triple point (P ~ 0.233 bar)
    sat = compute_sat_p(["H2S"], np.array([T_boil, T_triple], dtype=np.float64))
    p_boil, p_triple = float(sat["H2S"][0]), float(sat["H2S"][1])

    ok = True
    boil_rel = abs(p_boil - ATM_CGS) / ATM_CGS
    triple_rel = abs(p_triple - 0.233 * BAR_CGS) / (0.233 * BAR_CGS)
    print(
        f"H2S sat_p @ {T_boil} K  = {p_boil:.4e} dyne/cm^2 (expect ~1 atm = {ATM_CGS:.4e}); relerr {boil_rel:.3f}"
    )
    print(
        f"H2S sat_p @ {T_triple} K = {p_triple:.4e} dyne/cm^2 (expect ~0.233 bar); relerr {triple_rel:.3f}"
    )

    # 3% tolerance: the Antoine fit is not exact at these anchors, but a 10x
    # unit error (the regression we are guarding against) is far outside it.
    if boil_rel > 0.03:
        print("FAIL: H2S saturation pressure at the boiling point is not ~1 atm")
        ok = False
    if triple_rel > 0.05:
        print("FAIL: H2S saturation pressure at the triple point is not ~0.233 bar")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
