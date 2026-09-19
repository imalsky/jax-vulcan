"""NASA-9 polynomial thermo table: file I/O and the shared constants.

The Gibbs energy and reverse-rate math itself lives in `rates_jax.py` (one
implementation, on the AD graph); this module owns the reading of
`thermo/NASA9/<sp>.txt` and the constants both sides pin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .phy_const import kb

# Standard-state pressure, 1 bar in cgs (dyne/cm^2). `CORR * T` is the
# (kB T / P0) factor K_eq carries per unit change in mole number.
_P0 = 1.0e6
CORR = kb / _P0

# NASA-9 polynomials are fit in two temperature segments; the low-T coefficient
# set applies below this breakpoint and the high-T set above it (NASA/TP-2002-211556).
_NASA9_BRANCH_T = 1000.0


def load_nasa9(
    species: tuple[str, ...], thermo_dir: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load NASA-9 polynomial coefficients per species.

    Returns (coeffs[ni, 2, 10], present[ni]). Index 0 is low-T (T<1000 K),
    index 1 is high-T. Species without a NASA-9 file get zeros.
    """
    from ._paths import resolve_data_path

    thermo_dir = resolve_data_path(str(thermo_dir))
    ni = len(species)
    coeffs = np.zeros((ni, 2, 10), dtype=np.float64)
    present = np.zeros(ni, dtype=bool)
    for j, sp in enumerate(species):
        fp = thermo_dir / "NASA9" / f"{sp}.txt"
        if not fp.exists():
            continue
        flat = np.loadtxt(fp).flatten()
        if flat.size < 20:
            raise ValueError(
                f"NASA-9 file {fp} has only {flat.size} entries; need at least 20"
            )
        coeffs[j, 0] = flat[0:10]
        coeffs[j, 1] = flat[10:20]
        present[j] = True
    return coeffs, present
