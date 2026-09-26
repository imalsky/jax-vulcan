"""Validate the VULCAN-JAX initial-abundance pipeline.

Covers the `const_mix`, `vulcan_ini`, `table` and `const_lowT` modes plus the
charge_list invariant. `EQ` has its own file, `test_eq_seed.py`.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from _helpers import load_tpk_state, set_cfg

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")

RTOL = 1e-13  # machine agreement with the reference arithmetic
COLUMN_TOL = 1e-12  # column change the flux-form update may leave (roundoff)
# ini_abun._jax_newton's iteration cap and tolerance for the const_lowT solve.
NEWTON_MAX_ITER, NEWTON_TOL = 50, 1e-12


# const_mix mode: algebraic, no EQ seed, no scipy.


def test_const_mix_matches_reference():
    """`y[:, i] = const_mix[sp] * gas_tot` for every species in the dict;
    zeros elsewhere.
    """
    from vulcan_jax.ini_abun import InitialAbun
    import vulcan_jax.composition as composition

    data_var, data_atm, _ = load_tpk_state()
    # Earth-style mixing dict (species that exist in HD189's network).
    cmix = {"CH4": 5.5e-4, "He": 0.097, "N2": 8.2e-5, "H2": 0.9028}
    set_cfg(ini_mix="const_mix", const_mix=cmix)
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)

    y = np.asarray(data_var.y)
    M = np.asarray(data_atm.M)
    species_list = composition.species

    for sp, mix in cmix.items():
        idx = species_list.index(sp)
        ref = M * mix
        np.testing.assert_allclose(y[:, idx], ref, rtol=RTOL, atol=0.0)

    expected_zero_cols = [i for i, sp in enumerate(species_list) if sp not in cmix]
    assert np.all(y[:, expected_zero_cols] == 0.0)

    ymix = np.asarray(data_var.ymix)
    np.testing.assert_allclose(ymix.sum(axis=1), 1.0, rtol=RTOL, atol=0.0)


# vulcan_ini mode: pickle round-trip against an existing `.vul` file.


def test_vulcan_ini_roundtrip(tmp_path):
    """`vulcan_ini` mode restores each species column from the `.vul` exactly."""
    from vulcan_jax.ini_abun import InitialAbun
    import vulcan_jax.composition as composition

    data_var, data_atm, _ = load_tpk_state()
    prev_species = list(composition.species)
    shape = (len(data_atm.pco), len(prev_species))
    prev_y = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape)
    vul_path = tmp_path / "roundtrip.vul"
    with vul_path.open("wb") as handle:
        pickle.dump({"variable": {"species": prev_species, "y": prev_y}}, handle)

    set_cfg(ini_mix="vulcan_ini", vul_ini=str(vul_path))
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)

    y = np.asarray(data_var.y)
    species_list = composition.species
    for sp in ("H2", "He", "H2O", "CO", "CH4"):
        if sp not in species_list or sp not in prev_species:
            continue
        ref = prev_y[:, prev_species.index(sp)]
        np.testing.assert_allclose(
            y[:, species_list.index(sp)],
            ref,
            rtol=RTOL,
            atol=0.0,
            err_msg=f"vulcan_ini round-trip mismatch for {sp}",
        )


# table mode: synthesize a tiny mixing-ratio table on tmp_path.


def test_table_roundtrip(tmp_path):
    """`table` mode yields `y[:, sp] == n_0 * table[sp]`. The file MUST
    contain a column per species (master indexes `table[sp]` for every
    species); a sparse table does not work.
    """
    from vulcan_jax.ini_abun import InitialAbun
    import vulcan_jax.composition as composition

    data_var, data_atm, _ = load_tpk_state()
    nz_ = len(data_atm.pco)
    pco = np.asarray(data_atm.pco)
    n_0 = np.asarray(data_atm.n_0)
    species_list = list(composition.species)

    populated = {"H2": 0.85, "He": 0.15, "H2O": 1e-4}
    mix_values = np.zeros((nz_, len(species_list)))
    for sp, mix in populated.items():
        mix_values[:, species_list.index(sp)] = mix

    table_path = tmp_path / "ymix_table.txt"
    header = "# layer\nPressure " + " ".join(species_list)
    with open(table_path, "w") as f:
        f.write(header + "\n")
        for i in range(nz_):
            row = [f"{pco[i]:.6e}"] + [f"{v:.6e}" for v in mix_values[i]]
            f.write(" ".join(row) + "\n")

    set_cfg(ini_mix="table", vul_ini=str(table_path))
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)

    y = np.asarray(data_var.y)
    for sp, mix in populated.items():
        idx = species_list.index(sp)
        np.testing.assert_allclose(
            y[:, idx],
            n_0 * mix,
            rtol=RTOL,
            atol=0.0,
            err_msg=f"table mode mismatch for {sp}",
        )
    # Species not in the populated dict should be zero.
    other_idx = [i for i, sp in enumerate(species_list) if sp not in populated]
    assert np.all(y[:, other_idx] == 0.0), (
        "table mode left non-zero residue in unspecified species"
    )


# const_lowT mode: JAX Newton vs scipy.fsolve.


@pytest.mark.parametrize(
    "O_H,C_H,He_H,N_H",
    [
        (5.37e-4, 2.95e-4, 0.0838, 7.08e-5),  # HD189 solar
        (1.0e-3, 5.0e-4, 0.10, 1.0e-4),
        (1.0e-4, 5.0e-5, 0.05, 1.0e-5),
    ],
)
def test_const_lowT_matches_scipy(O_H, C_H, He_H, N_H):
    """JAX Newton on the 5-mol H2/H2O/CH4/He/NH3 system matches scipy
    fsolve to 1e-13."""
    import jax.numpy as jnp
    from scipy.optimize import fsolve
    from vulcan_jax.ini_abun import _abun_lowT_residual, _jax_newton

    def master_res(x, *args):
        return list(_abun_lowT_residual(jnp.asarray(x), *args))

    x0 = [0.9, 0.1, 0.0, 0.0, 0.0]
    scipy_root = fsolve(master_res, x0, args=(O_H, C_H, He_H, N_H))
    jax_root = np.asarray(
        _jax_newton(
            _abun_lowT_residual,
            jnp.array(x0),
            (O_H, C_H, He_H, N_H),
            max_iter=NEWTON_MAX_ITER,
            tol=NEWTON_TOL,
        )
    )
    np.testing.assert_allclose(scipy_root, jax_root, rtol=RTOL, atol=1e-15)


# charge_list invariants.


def test_charge_list_no_ions():
    """With `use_ion=False`, `data_var.charge_list` stays empty (or unset)."""
    from vulcan_jax.ini_abun import InitialAbun
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    data_var, data_atm, _ = load_tpk_state()
    assert vulcan_cfg.use_ion is False, "test assumes HD189 default cfg"
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)

    cl = list(getattr(data_var, "charge_list", []))
    assert cl == [], f"expected empty charge_list, got {cl}"


def test_column_atoms_uses_the_operator_invariant():
    """The column budget must weight by what transport conserves.

    A flux-form update `n_j += (F_{j-1/2} - F_{j+1/2}) / w_j` with zero
    boundary fluxes telescopes: it conserves `sum_j w_j n_j` exactly for the
    operator's own cell measure `w` and does NOT conserve `sum_j dz_j n_j`
    on a nonuniform grid. On a uniform grid `w == dz`, so the budget reduces
    to the unweighted `atom_loss` relative change identically.
    """
    from vulcan_jax.ini_abun import column_atoms, operator_column_weights

    rng = np.random.default_rng(7)
    nz, ni, na = 12, 5, 3
    compo = rng.uniform(0.0, 3.0, (ni, na))
    y0 = rng.uniform(1e8, 1e10, (nz, ni))

    def column_change(y, y0, dz):
        c0 = np.asarray(column_atoms(y0, dz, compo))
        return (np.asarray(column_atoms(y, dz, compo)) - c0) / c0

    # Nonuniform (jumpy) grid: apply random interface fluxes per species.
    dz = rng.uniform(1e5, 9e5, nz)
    w = np.asarray(operator_column_weights(dz))
    flux = rng.uniform(-1e12, 1e12, (nz - 1, ni))
    y1 = y0 + (
        np.vstack([np.zeros((1, ni)), flux]) - np.vstack([flux, np.zeros((1, ni))])
    ) / w[:, None]
    np.testing.assert_allclose(column_change(y1, y0, dz), 0.0, atol=COLUMN_TOL)
    dz_drift = np.einsum("z,zi,ia->a", dz, y1 - y0, compo)
    dz_ref = np.einsum("z,zi,ia->a", dz, y0, compo)
    assert np.max(np.abs(dz_drift / dz_ref)) > 1e-6, (
        "plain-dz weighting should NOT be conserved on a nonuniform grid; "
        "if it is, this test's flux is degenerate"
    )

    # Uniform grid: identical to the unweighted relative change.
    dz_u = np.full(nz, 3.7e5)
    y2 = y0 * rng.uniform(0.9, 1.1, (nz, ni))
    unweighted = (
        np.einsum("zi,ia->a", y2, compo) - np.einsum("zi,ia->a", y0, compo)
    ) / np.einsum("zi,ia->a", y0, compo)
    np.testing.assert_allclose(column_change(y2, y0, dz_u), unweighted, rtol=COLUMN_TOL)


