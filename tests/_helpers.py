"""Shared test helpers: config pinning, partial configs and child processes.

Neither belongs in `oracle.py`: nothing here touches the upstream checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

R_JUP_CM = 7.1492e9  # Jupiter radius (cm), upstream phy_const.py r_jup
HD189_RP_CM = 1.138 * R_JUP_CM  # HD 189733 b radius (cm)
SELF_RUN_TIMEOUT_S = 1800  # one or more full converged runs per child


def set_cfg(**overrides):
    """Set attributes on the shared process default config and return it.

    conftest restores the config after every test, so no undo is needed.
    """
    from vulcan_jax.config import default_config

    cfg = default_config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def load_tpk_state():
    """`(var, atm, make_atm)` after f_pico and load_TPK (and sp_sat under
    use_condense) on the process default config, in the legacy containers.

    Partial setup: no rates, EQ seed or photo reads.
    """
    from vulcan_jax.atm_setup import Atm
    from vulcan_jax.config import default_config
    from vulcan_jax.state import _AtmData, _Variables

    data_var = _Variables()
    data_atm = _AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if default_config().use_condense:
        make_atm.sp_sat(data_atm)
    return data_var, data_atm, make_atm


def fast_cfg(**overrides):
    """The process default config, pinned for a fast, quiet, isothermal run.

    Mutates and returns the shared cached `default_config()` object; callers
    rely on the mutation being visible to later `default_config()` readers.

    The diffusion-scheme knobs (`use_vm_mol`, `use_hybrid_vm_mol`) are not
    pinned: test_hybrid_vm_mol tests them, so each caller sets them
    explicitly.
    """
    set_cfg(count_min=1, use_print_prog=False, use_photo=False, use_ion=False,
            atm_type="isothermal", Kzz_prof="Pfunc")
    return set_cfg(**overrides)


def relerr(got, ref, floor=1e-30, mask=None) -> float:
    """Max of |got - ref| / max(|ref|, floor), over `mask` cells if given."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), floor)
    return float(np.max(rel if mask is None else rel[mask]))


def atom_count_matrix(net, atoms) -> np.ndarray:
    """Species-by-atom stoichiometry of the config's com_file, shape (ni, n_atoms).

    Read with an explicit encoding so species names come back as str.
    """
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.config import default_config

    compo = np.genfromtxt(
        resolve_data_path(default_config().com_file),
        names=True,
        dtype=None,
        encoding=None,
    )
    row_by_species = {str(row["species"]): row for row in compo}
    return np.asarray(
        [[float(row_by_species[sp][a]) for a in atoms] for sp in net.species],
        dtype=np.float64,
    )


def mass_for_gravity(g, rp):
    """Planet mass (g) that gives surface gravity `g` (cm/s^2) at radius `rp` (cm)."""
    from vulcan_jax.phy_const import G_grav

    return g * rp**2 / G_grav


def make_pco(P_b, P_t, nz):
    """`nz` log-spaced pressures from P_b (bottom) to P_t (top), dyne/cm^2."""
    return np.logspace(np.log10(P_b), np.log10(P_t), nz)


def tpk_cfg(**overrides):
    """Partial config for `atm_setup.load_TPK`: isothermal, constant Kzz, the
    HD189 radius and g = 2140 cm/s^2. `overrides` replace or add fields."""
    from vulcan_jax._paths import PACKAGE_ROOT

    fields = dict(
        atm_type="isothermal",
        Kzz_prof="const",
        vz_prof="const",
        use_Kzz=True,
        use_vz=False,
        const_Kzz=1e10,
        const_vz=0.0,
        K_max=1e5,
        K_p_lev=0.1,
        Tiso=1234.0,
        P_b=1e9,
        Rp=HD189_RP_CM,
        Mp=mass_for_gravity(2140.0, HD189_RP_CM),
        para_anaTP=[120.0, 1500.0, 0.1, 0.02, 1.0, 1.0],
        atm_file=str(PACKAGE_ROOT / "atm" / "atm_HD189_Kzz.txt"),
        vul_ini="output/",
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def run_self(test_file, *args, network: str, atom_list: str, cwd) -> None:
    """Re-run `test_file` as a single-threaded script with `network` and
    `atom_list` selected (both are import-frozen); assert a clean exit."""
    env = dict(os.environ, VULCAN_JAX_NETWORK=network,
               VULCAN_JAX_ATOM_LIST=atom_list, OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(Path(test_file).resolve()), *args],
                            cwd=cwd, env=env, text=True, capture_output=True,
                            timeout=SELF_RUN_TIMEOUT_S)
    assert result.returncode == 0, result.stdout + result.stderr


def run_child(child_src: str, *, network: str, label: str, timeout: int = 600):
    """Run `child_src` in a fresh CPU-only interpreter with `network` selected.

    A cold start is required: the network is import-frozen, so it can only be
    chosen through the environment before `vulcan_jax` is first imported.
    Asserts a clean exit and returns the CompletedProcess so the caller can
    make its own assertions about stdout.
    """
    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "VULCAN_JAX_NETWORK": network,
    }
    res = subprocess.run(
        [sys.executable, "-c", child_src, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=ROOT,
        check=False,  # the assert below reports stdout/stderr on failure
    )
    assert res.returncode == 0, (
        f"{label} subprocess exited {res.returncode}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    return res
