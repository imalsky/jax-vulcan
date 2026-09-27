"""Verify every species' molecular mass is consistent with its atom counts.

`thermo/all_compose.txt` lists, per species, the per-element atom counts and a
molecular `mass`. A transcription error in the `mass` column (a whole-atom
slip such as NH2's 16.023 for NH3_l_s) silently corrupts the mean
molecular weight, settling velocity, and molecular diffusion for that species.

This test recomputes each molecular mass from the atom counts and standard
atomic weights and asserts agreement, and separately checks that every
condensate (`*_l_s` / `*_s`) has the same mass as its gas-phase counterpart.

Duplicate rows are a second silent hazard: production resolves a species via
`list.index`, i.e. the FIRST row wins, so a duplicate with a different mass is
file-order-dependent. The vendored upstream table ships exactly six duplicated
species (three disagreeing in mass); that state is pinned below so any new
duplicate, or any change to which value wins, fails loudly.

Pure data check: no EQ seed, no VULCAN-master, no integration.
"""

from __future__ import annotations

import numpy as np

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

# The vendored upstream all_compose.txt defines these species twice (same atom
# counts). Three pairs disagree in mass; production's first-wins `.index`
# selects the values below. HCS's 45.178 sits 0.099 amu above its elemental
# sum (the second row's 45.079 is the consistent one): an upstream data bug
# kept for parity (~7e-5 on one moldiff coefficient).
_KNOWN_DUPLICATE_SPECIES = {"C4H2", "CH3O2", "CH3OOH", "C2H4O", "CH3NO2", "HCS"}
_KNOWN_FIRST_WINS_MASS = {"C2H4O": 44.054, "CH3NO2": 61.042, "HCS": 45.178}


def test_species_masses_match_atom_counts():
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.config import default_config

    compo = np.genfromtxt(
        resolve_data_path(default_config().com_file), names=True, dtype=None, encoding=None
    )
    elems = [c for c in compo.dtype.names if c not in ("species", "mass")]
    species = [str(s) for s in compo["species"]]
    listed = np.asarray(compo["mass"], dtype=np.float64)
    counts = np.array(
        [[float(compo[e][i]) for e in elems] for i in range(len(species))]
    )
    expected = counts @ np.array([_ATOMIC_MASS[e] for e in elems])

    bad = [
        f"{species[i]} listed={listed[i]:.3f} expected={expected[i]:.3f}"
        for i in range(len(species))
        if abs(listed[i] - expected[i]) > _TOL
    ]
    assert not bad, f"species mass inconsistent with atom counts: {bad}"

    # Duplicate rows: mirror production (`list.index` -> FIRST row wins).
    mass_by_sp: dict[str, float] = {}
    for sp, m in zip(species, listed):
        mass_by_sp.setdefault(sp, float(m))

    dup_names = {sp for sp in mass_by_sp if species.count(sp) > 1}
    assert dup_names == _KNOWN_DUPLICATE_SPECIES, (
        f"duplicated species set changed: got {sorted(dup_names)}, "
        f"pinned {sorted(_KNOWN_DUPLICATE_SPECIES)}. A new duplicate is "
        "resolved silently by row order -- deduplicate the table instead."
    )
    for sp in sorted(dup_names):
        rows = [i for i, s in enumerate(species) if s == sp]
        assert all(np.array_equal(counts[i], counts[rows[0]]) for i in rows[1:]), (
            f"duplicate rows for {sp} disagree in ATOM COUNTS."
        )
        masses = {float(listed[i]) for i in rows}
        expect = _KNOWN_FIRST_WINS_MASS.get(sp)
        assert len(masses) == 1 or mass_by_sp[sp] == expect, (
            f"duplicate rows for {sp} disagree in mass {sorted(masses)} "
            f"and first-wins value {mass_by_sp[sp]} is not the pinned "
            f"{expect}. The model silently uses the first row."
        )

    # Condensates must share the mass of their gas-phase counterpart.
    for sp in species:
        for suffix in ("_l_s", "_s", "_l"):
            if sp.endswith(suffix):
                gas = sp[: -len(suffix)]
                if gas in mass_by_sp:
                    assert abs(mass_by_sp[sp] - mass_by_sp[gas]) <= 1e-6, (
                        f"condensate {sp} mass {mass_by_sp[sp]} != gas {gas} "
                        f"mass {mass_by_sp[gas]}"
                    )
                break
