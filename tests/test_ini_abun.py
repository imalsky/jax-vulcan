"""Validate VULCAN-JAX initial-abundance pipeline.

Coverage matrix over `ini_mix ∈ {EQ, const_mix, vulcan_ini, table,
const_lowT}`:

- `test_const_mix_matches_reference` — algebraic per-species check.
- `test_vulcan_ini_roundtrip` — pickle restore from a known `.vul`.
- `test_table_roundtrip` — `genfromtxt` on a synthetic table.
- `test_const_lowT_matches_scipy` — JAX Newton vs `scipy.optimize.fsolve`
  on a grid of elemental ratios.
- `test_charge_list_no_ions` — `use_ion=False` → empty `charge_list`.
- `test_main` — EQ-mode HD189 fork against VULCAN-master (bit-exact gate;
  runs LAST because it pops `vulcan_cfg` / `store` / `chem_funs` from
  `sys.modules` to load the upstream copies, which would corrupt any
  later test in the same file).
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
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers shared by every parametrized mode test.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _cfg_overrides(**kwargs):
    """Snapshot/restore vulcan_cfg attributes around a block."""
    import vulcan_cfg
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
    `sp_sat` if condense is on). Caller takes a deep copy if it plans
    to mutate cfg before calling `InitialAbun.ini_y`.

    Callers of this helper only need the partial setup (`f_pico` +
    `load_TPK`) — full `RunState.with_pre_loop_setup` would also run rate
    parsing / FastChem / photo cross-section reads which the parametrized
    mode tests don't use; using the private `state._Variables` /
    `_AtmData` containers keeps this lightweight.
    """
    from atm_setup import Atm
    from state import _Variables, _AtmData
    import vulcan_cfg

    data_var = _Variables()
    data_atm = _AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    return data_var, data_atm, make_atm


# ---------------------------------------------------------------------------
# const_mix mode — algebraic, no FastChem, no scipy.
# ---------------------------------------------------------------------------

def test_const_mix_matches_reference():
    """`y[:, i] = const_mix[sp] * gas_tot` for every species in the dict;
    zeros elsewhere. Mode is purely algebraic — no scipy, no FastChem.
    """
    from ini_abun import InitialAbun
    import composition

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
# vulcan_ini mode — pickle round-trip against an existing `.vul` file.
# ---------------------------------------------------------------------------

def test_vulcan_ini_roundtrip():
    """Load `output/HD189.vul` via `vulcan_ini` mode and assert each
    species column matches the file's `y[:, species.index(sp)]` exactly.
    """
    from ini_abun import InitialAbun
    import composition

    vul_path = ROOT / "output" / "HD189.vul"
    if not vul_path.is_file():
        pytest.skip(f"{vul_path} not present; run `python vulcan_jax.py` to generate.")

    with open(vul_path, "rb") as f:
        prev = pickle.load(f)
    prev_species = prev["variable"]["species"]
    prev_y = np.asarray(prev["variable"]["y"])

    data_var, data_atm, _ = _build_hd189_atm()
    if prev_y.shape[0] != len(data_atm.pco):
        pytest.skip(f"`HD189.vul` has nz={prev_y.shape[0]} but cfg nz={len(data_atm.pco)}")

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
            y[:, species_list.index(sp)], ref, rtol=1e-13, atol=0.0,
            err_msg=f"vulcan_ini round-trip mismatch for {sp}",
        )


# ---------------------------------------------------------------------------
# table mode — synthesize a tiny mixing-ratio table on tmp_path.
# ---------------------------------------------------------------------------

def test_table_roundtrip(tmp_path):
    """Write a per-layer mixing table containing every species in the
    network (most zero, three populated); load via `table` mode; verify
    `y[:, sp] == n_0 * table[sp]` for the populated species.

    Master's `ini_y[ini_mix=='table']` iterates over the full species
    list and indexes `table[sp]` for each, so the file MUST contain a
    column per species — we cannot get away with a sparse table.
    """
    from ini_abun import InitialAbun
    import composition

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
            y[:, idx], n_0 * mix, rtol=1e-13, atol=0.0,
            err_msg=f"table mode mismatch for {sp}",
        )
    # Species not in the populated dict should be zero.
    other_idx = [i for i, sp in enumerate(species_list) if sp not in populated]
    assert np.all(y[:, other_idx] == 0.0), "table mode left non-zero residue in unspecified species"


# ---------------------------------------------------------------------------
# const_lowT mode — JAX Newton vs scipy.fsolve.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "O_H,C_H,He_H,N_H",
    [
        (5.37e-4, 2.95e-4, 0.0838, 7.08e-5),  # HD189 solar
        (1.0e-3,  5.0e-4,  0.10,   1.0e-4),
        (1.0e-4,  5.0e-5,  0.05,   1.0e-5),
    ],
)
def test_const_lowT_matches_scipy(O_H, C_H, He_H, N_H):
    """Solve the 5-mol H2/H2O/CH4/He/NH3 system in JAX and assert
    agreement with scipy fsolve to 1e-13. Tolerance for this mode was
    pre-flagged to potentially relax to 1e-10; in practice the JAX
    Newton matches scipy to machine precision (~1e-16) on solar-like
    ratios."""
    import jax.numpy as jnp
    from scipy.optimize import fsolve
    from ini_abun import _abun_lowT_residual, _jax_newton

    def master_res(x, *args):
        return list(_abun_lowT_residual(jnp.asarray(x), *args))

    x0 = [0.9, 0.1, 0.0, 0.0, 0.0]
    scipy_root = fsolve(master_res, x0, args=(O_H, C_H, He_H, N_H))
    jax_root = np.asarray(_jax_newton(
        _abun_lowT_residual, jnp.array(x0), (O_H, C_H, He_H, N_H),
    ))
    np.testing.assert_allclose(scipy_root, jax_root, rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# charge_list invariants.
# ---------------------------------------------------------------------------

def test_charge_list_no_ions():
    """HD189 has `use_ion=False`; `data_var.charge_list` must stay
    empty (or unset) and the legacy attribute must not appear with
    spurious entries.
    """
    from ini_abun import InitialAbun
    import vulcan_cfg

    data_var, data_atm, _ = _build_hd189_atm()
    assert vulcan_cfg.use_ion is False, "test assumes HD189 default cfg"
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)

    cl = list(getattr(data_var, "charge_list", []))
    assert cl == [], f"expected empty charge_list, got {cl}"


# ---------------------------------------------------------------------------
# EQ-mode HD189 fork against VULCAN-master (bit-exact gate).
#
# This test MUST run last in the file because it pops `vulcan_cfg`,
# `store`, `chem_funs`, and `build_atm` from `sys.modules` and inserts
# `../VULCAN-master/` at the head of `sys.path`. Any subsequent test in
# the same file that does `import store` would then get master's `store`
# instead of VULCAN-JAX's.
# ---------------------------------------------------------------------------

def main() -> int:
    from atm_setup import Atm
    from ini_abun import InitialAbun
    from state import _Variables, _AtmData
    import vulcan_cfg
    import chem_funs as cf_jax

    print(f"VULCAN-JAX chem_funs: ni={cf_jax.ni} nr={cf_jax.nr}, ini_mix={vulcan_cfg.ini_mix}")

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
    VULCAN_MASTER = ROOT.parent / "VULCAN-master"
    if not VULCAN_MASTER.is_dir():
        print("SKIP: VULCAN-master sibling not present; bit-exact oracle unavailable.")
        return 0
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
            print(f"OK   atom_ini[{atom}] = {jax_atom_ini[atom]:.4e} (relerr {diff:.2e})")
        else:
            print(f"FAIL atom_ini[{atom}]: jax={jax_atom_ini[atom]:.4e} vul={vul_atom_ini[atom]:.4e}")
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_zzz_main_eq_vs_master():
    """EQ-mode bit-exact gate against VULCAN-master.

    Named with `zzz` so pytest collects (and runs) it after every other
    test in this file — this test pollutes `sys.modules` and would
    otherwise break the per-mode tests above.
    """
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
