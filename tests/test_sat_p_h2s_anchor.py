"""Anchor the H2S saturation-pressure unit conversion to physical references.

The Giauque & Blue (1936) Antoine fit gives cmHg, so the cgs factor is
0.01333 * 1e6; the mmHg factor would be 10x low. Anchors: normal boiling
point 212.8 K at 1 atm, and triple point 187.66 K at ~0.233 bar.

A unit test only: no EQ seed, no VULCAN-master, no integration.
"""

from __future__ import annotations

import numpy as np

from vulcan_jax.phy_const import ATM_CGS, BAR_CGS


def test_h2s_sat_p_hits_boiling_and_triple_points():
    from vulcan_jax.atm_setup import compute_sat_p

    T_boil = 212.8  # K, H2S normal boiling point (P = 1 atm)
    T_triple = 187.66  # K, H2S triple point (P ~ 0.233 bar)
    sat = compute_sat_p(["H2S"], np.array([T_boil, T_triple], dtype=np.float64))
    p_boil, p_triple = float(sat["H2S"][0]), float(sat["H2S"][1])

    # 3% / 5%: the Antoine fit is not exact at the anchors; a 10x unit error is far outside both.
    assert abs(p_boil - ATM_CGS) / ATM_CGS <= 0.03, p_boil
    assert abs(p_triple - 0.233 * BAR_CGS) / (0.233 * BAR_CGS) <= 0.05, p_triple
