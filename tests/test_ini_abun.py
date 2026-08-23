"""Validate the VULCAN-JAX initial-abundance pipeline.

Covers all five `ini_mix` modes (EQ, const_mix, vulcan_ini, table,
const_lowT) plus the charge_list invariant. The EQ mode is a bit-exact
gate against VULCAN-master, run in a subprocess so the upstream imports
cannot pollute the pytest worker.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
from oracle import oracle_dir_or_sentinel  # noqa: E402

# Oracle location from $VULCAN_MASTER_DIR only, never a sibling guess. The
# parent verifies the pinned revision + clean tree and points this at a
# temporary COPY; the per-test is_dir() skips below handle "not configured".
VULCAN_MASTER = oracle_dir_or_sentinel()

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers shared by every parametrized mode test.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _cfg_overrides(**kwargs):
    """Snapshot/restore vulcan_cfg attributes around a block."""
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    saved = {}
    sentinel = object()
    for k in kwargs:
        saved[k] = getattr(vulcan_cfg, k, sentinel)
    try:
        for k, v in kwargs.items():
            setattr(vulcan_cfg, k, v)
        yield vulcan_cfg
    finally:
        for k, v in saved.items():
            if v is sentinel:
                delattr(vulcan_cfg, k)
            else:
                setattr(vulcan_cfg, k, v)


def _build_hd189_atm():
    """Return `(data_var, data_atm, make_atm)` after `load_TPK` (and
    `sp_sat` if condense is on). Deliberately partial setup: the full
    `RunState.with_pre_loop_setup` would also run rates / FastChem / photo
    reads these mode tests do not use.
    """
    from vulcan_jax.atm_setup import Atm
    from vulcan_jax.state import _Variables, _AtmData
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    data_var = _Variables()
    data_atm = _AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    return data_var, data_atm, make_atm


# ---------------------------------------------------------------------------
# const_mix mode: algebraic, no FastChem, no scipy.
# ---------------------------------------------------------------------------


def test_const_mix_matches_reference():
    """`y[:, i] = const_mix[sp] * gas_tot` for every species in the dict;
    zeros elsewhere.
    """
    from vulcan_jax.ini_abun import InitialAbun
    import vulcan_jax.composition as composition

    data_var, data_atm, _ = _build_hd189_atm()
    # Earth-style mixing dict (species that exist in HD189's network).
    cmix = {"CH4": 5.5e-4, "He": 0.097, "N2": 8.2e-5, "H2": 0.9028}
    with _cfg_overrides(ini_mix="const_mix", const_mix=cmix):
        ini = InitialAbun()
        data_var = ini.ini_y(data_var, data_atm)
        data_var = ini.ele_sum(data_var)

    y = np.asarray(data_var.y)
    M = np.asarray(data_atm.M)
    species_list = composition.species

    for sp, mix in cmix.items():
        idx = species_list.index(sp)
        ref = M * mix
        np.testing.assert_allclose(y[:, idx], ref, rtol=1e-13, atol=0.0)

    expected_zero_cols = [i for i, sp in enumerate(species_list) if sp not in cmix]
    assert np.all(y[:, expected_zero_cols] == 0.0)

    ymix = np.asarray(data_var.ymix)
    np.testing.assert_allclose(ymix.sum(axis=1), 1.0, rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# vulcan_ini mode: pickle round-trip against an existing `.vul` file.
# ---------------------------------------------------------------------------


def test_vulcan_ini_roundtrip(tmp_path):
    """`vulcan_ini` mode restores each species column from the `.vul` exactly."""
    from vulcan_jax.ini_abun import InitialAbun
    import vulcan_jax.composition as composition

    data_var, data_atm, _ = _build_hd189_atm()
    prev_species = list(composition.species)
    shape = (len(data_atm.pco), len(prev_species))
    prev_y = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape)
    vul_path = tmp_path / "roundtrip.vul"
    with vul_path.open("wb") as handle:
        pickle.dump({"variable": {"species": prev_species, "y": prev_y}}, handle)

    with _cfg_overrides(ini_mix="vulcan_ini", vul_ini=str(vul_path)):
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
            rtol=1e-13,
            atol=0.0,
            err_msg=f"vulcan_ini round-trip mismatch for {sp}",
        )


# ---------------------------------------------------------------------------
# table mode: synthesize a tiny mixing-ratio table on tmp_path.
# ---------------------------------------------------------------------------


def test_table_roundtrip(tmp_path):
    """`table` mode yields `y[:, sp] == n_0 * table[sp]`. The file MUST
    contain a column per species (master indexes `table[sp]` for every
    species); a sparse table does not work.
    """
    from vulcan_jax.ini_abun import InitialAbun
    import vulcan_jax.composition as composition

    data_var, data_atm, _ = _build_hd189_atm()
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

    with _cfg_overrides(ini_mix="table", vul_ini=str(table_path)):
        ini = InitialAbun()
        data_var = ini.ini_y(data_var, data_atm)

    y = np.asarray(data_var.y)
    for sp, mix in populated.items():
        idx = species_list.index(sp)
        np.testing.assert_allclose(
            y[:, idx],
            n_0 * mix,
            rtol=1e-13,
            atol=0.0,
            err_msg=f"table mode mismatch for {sp}",
        )
    # Species not in the populated dict should be zero.
    other_idx = [i for i, sp in enumerate(species_list) if sp not in populated]
    assert np.all(y[:, other_idx] == 0.0), (
        "table mode left non-zero residue in unspecified species"
    )


# ---------------------------------------------------------------------------
# const_lowT mode: JAX Newton vs scipy.fsolve.
# ---------------------------------------------------------------------------


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
    fsolve to 1e-13 (in practice ~1e-16 on solar-like ratios)."""
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
        )
    )
    np.testing.assert_allclose(scipy_root, jax_root, rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# charge_list invariants.
# ---------------------------------------------------------------------------


def test_charge_list_no_ions():
    """With `use_ion=False`, `data_var.charge_list` stays empty (or unset)."""
    from vulcan_jax.ini_abun import InitialAbun
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    data_var, data_atm, _ = _build_hd189_atm()
    assert vulcan_cfg.use_ion is False, "test assumes HD189 default cfg"
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)

    cl = list(getattr(data_var, "charge_list", []))
    assert cl == [], f"expected empty charge_list, got {cl}"


# ---------------------------------------------------------------------------
# EQ-mode HD189 fork against VULCAN-master (bit-exact gate). Mutates
# sys.modules/sys.path, so the pytest wrapper runs it in a fresh process.
# ---------------------------------------------------------------------------


def main() -> int:
    from vulcan_jax.atm_setup import Atm
    from vulcan_jax.ini_abun import InitialAbun
    from vulcan_jax.state import _Variables, _AtmData
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    # Match upstream's full-solar Lodders 2009 composition. The production
    # default is deliberately rocky-suppressed Lodders 2019, so comparing the
    # defaults would test different scientific inputs rather than two
    # implementations of the same equilibrium calculation.
    vulcan_cfg.fastchem_solar_abundance_file = (
        "fastchem_vulcan/input/solar_element_abundances_lodders2009.dat"
    )
    import vulcan_jax.chem_funs as cf_jax

    print(
        f"VULCAN-JAX chem_funs: ni={cf_jax.ni} nr={cf_jax.nr}, ini_mix={vulcan_cfg.ini_mix}"
    )

    data_var = _Variables()
    data_atm = _AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)

    ini_abun = InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)

    jax_y = np.asarray(data_var.y).copy()
    jax_ymix = np.asarray(data_var.ymix).copy()
    jax_atom_ini = dict(data_var.atom_ini)

    print(f"  y shape: {jax_y.shape}, y range [{jax_y.min():.3e}, {jax_y.max():.3e}]")
    print(f"  ymix shape: {jax_ymix.shape}")
    print(f"  atom_ini: {jax_atom_ini}")

    for mod_name in ("build_atm", "store", "chem_funs", "vulcan_cfg"):
        sys.modules.pop(mod_name, None)
    if not VULCAN_MASTER.is_dir():
        print("SKIP: VULCAN-master sibling not present; bit-exact oracle unavailable.")
        return 0
    os.chdir(VULCAN_MASTER)
    sys.path.insert(0, str(VULCAN_MASTER))

    import build_atm as ba_v
    import store as st_v
    import vulcan_cfg as cfg_v

    if getattr(cfg_v, "use_solar", True) != getattr(vulcan_cfg, "use_solar", True):
        os.chdir(ROOT)
        print(
            "SKIP: master and JAX vulcan_cfg have different use_solar; configs diverged."
        )
        return 0

    # Stay in VULCAN-master while building the master-side objects: they read
    # files via paths RELATIVE to the master root, and ini_y() shells out to
    # fastchem_vulcan/. Only chdir back to ROOT after all master-side reads are
    # done; restoring CWD earlier raised FileNotFoundError and masked this gate.
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

    os.chdir(ROOT)

    vul_y = np.asarray(data_var2.y)
    vul_atom_ini = dict(data_var2.atom_ini)

    ok = True
    if jax_y.shape != vul_y.shape:
        print(f"FAIL y shape: jax={jax_y.shape} vul={vul_y.shape}")
        ok = False
    else:
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
            print(
                "FAIL y: max relerr = "
                f"{max_relerr:.2e} for species "
                f"{cf_jax.spec_list[max_sp] if max_sp >= 0 else '?'}"
            )
            ok = False

    for atom in jax_atom_ini:
        if atom not in vul_atom_ini:
            print(f"  atom {atom} missing from VULCAN")
            continue
        diff = abs(jax_atom_ini[atom] - vul_atom_ini[atom]) / abs(vul_atom_ini[atom])
        if diff < 1e-12:
            print(
                f"OK   atom_ini[{atom}] = {jax_atom_ini[atom]:.4e} (relerr {diff:.2e})"
            )
        else:
            print(
                f"FAIL atom_ini[{atom}]: jax={jax_atom_ini[atom]:.4e} vul={vul_atom_ini[atom]:.4e}"
            )
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_zzz_main_eq_vs_master():
    """EQ-mode bit-exact gate against VULCAN-master, run as a fresh
    subprocess because main() swaps in VULCAN-master modules.
    """
    if not VULCAN_MASTER.is_dir():
        pytest.skip(
            f"upstream oracle not configured (looked at {VULCAN_MASTER}). Set "
        f"$VULCAN_MASTER_DIR to a clean clone at the commit pinned in "
        f"tests/science_sources.yaml; "
            "this comparison test requires the upstream sibling repo."
        )
    import subprocess
    from oracle import oracle_worktree

    with oracle_worktree(
        "vulcan2_ncho",
        "cfg_examples/vulcan_cfg_HD189.py",
        fastchem_abundance="solar_element_abundances_lodders2009.dat",
    ) as master:
        env = os.environ.copy()
        env["VULCAN_MASTER_DIR"] = str(master)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_column_atom_loss_uses_the_operator_invariant():
    """The physical column budget must weight by what transport conserves.

    A flux-form update `n_j += (F_{j-1/2} - F_{j+1/2}) / w_j` with zero
    boundary fluxes telescopes: it conserves `sum_j w_j n_j` exactly for the
    operator's own cell measure `w` and does NOT conserve `sum_j dz_j n_j`
    on a nonuniform grid. On a uniform grid `w == dz`, so the budget reduces
    to the unweighted `atom_loss` relative change identically.
    """
    from vulcan_jax.ini_abun import column_atom_loss, operator_column_weights

    rng = np.random.default_rng(7)
    nz, ni, na = 12, 5, 3
    compo = rng.uniform(0.0, 3.0, (ni, na))
    y0 = rng.uniform(1e8, 1e10, (nz, ni))

    # Nonuniform (jumpy) grid: apply random interface fluxes per species.
    dz = rng.uniform(1e5, 9e5, nz)
    w = np.asarray(operator_column_weights(dz))
    flux = rng.uniform(-1e12, 1e12, (nz - 1, ni))
    y1 = y0 + (
        np.vstack([np.zeros((1, ni)), flux]) - np.vstack([flux, np.zeros((1, ni))])
    ) / w[:, None]
    drift = np.asarray(column_atom_loss(y1, y0, dz, compo_arr=compo))
    np.testing.assert_allclose(drift, 0.0, atol=1e-12)
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
    np.testing.assert_allclose(
        np.asarray(column_atom_loss(y2, y0, dz_u, compo_arr=compo)),
        unweighted,
        rtol=1e-12,
    )


if __name__ == "__main__":
    sys.exit(main())
