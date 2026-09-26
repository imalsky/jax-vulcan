"""Anchor the H2S saturation-pressure unit conversion to physical references.

The Giauque & Blue (1936) Antoine fit gives cmHg, so the cgs factor is
0.01333 * 1e6; the mmHg factor would be 10x low. Anchors: normal boiling
point 212.8 K at 1 atm, and triple point 187.66 K at ~0.233 bar.

A unit test only — no EQ seed, no VULCAN-master, no integration.
"""

from __future__ import annotations

import numpy as np

from vulcan_jax.phy_const import ATM_CGS, BAR_CGS


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

    # 3% / 5%: the Antoine fit is not exact at the anchors; a 10x unit error is far outside both.
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
