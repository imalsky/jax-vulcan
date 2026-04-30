"""JAX-native initial-abundance setup.

Five `ini_mix` modes:
- `EQ`         FastChem equilibrium (subprocess + parsed output)
- `const_mix`  apply a per-species mixing dict from cfg
- `vulcan_ini` restore composition from a previous `.vul` file
- `table`      read a per-layer mixing-ratio table
- `const_lowT` solve the 5-mol H2/H2O/CH4/He/NH3 system via JAX Newton

`compute_initial_abundance` returns a typed `IniAbunOutputs`. The
`InitialAbun` class is a legacy facade that mutates `data_var`/`data_atm`.

FastChem subprocess calls are serialised via `fcntl.flock` so
`pytest -n auto` and concurrent drivers don't race on the shared
`fastchem_vulcan/input/` and `output/` files.
"""

from __future__ import annotations

import fcntl
import pickle
import subprocess
from pathlib import Path
from shutil import copyfile

import jax
import jax.numpy as jnp
import numpy as np

import vulcan_cfg
import chem_funs
from composition import compo, compo_row, compo_array, species, atom_list as _COMPO_ATOMS
from state import IniAbunOutputs

jax.config.update("jax_enable_x64", True)


# Anchor FastChem paths to the source location (not cwd) so concurrent
# workers invoked from different directories all flock on the same
# sentinel and write to the same input/output files.
_FC_DIR = (Path(__file__).resolve().parent / "fastchem_vulcan").resolve()
_FC_SENTINEL = _FC_DIR / ".fastchem_lock"
_FC_INPUT = _FC_DIR / "input"
_FC_OUTPUT = _FC_DIR / "output"
_FC_VULCAN_TP = _FC_INPUT / "vulcan_TP" / "vulcan_TP.dat"
_FC_VULCAN_EQ = _FC_OUTPUT / "vulcan_EQ.dat"


def _abun_lowT_residual(x, O_H, C_H, He_H, N_H):
    """5-mol residual: H2 / H2O / CH4 / He / NH3.

    The sum constraint (f1) only includes x1..x4; NH3 (x5) participates
    in the H/C/O/He/N balance equations only.
    """
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    coupling = 2.0 * x1 + 2.0 * x2 + 4.0 * x3 + 3.0 * x5
    f1 = x1 + x2 + x3 + x4 - 1.0
    f2 = x2 - coupling * O_H
    f3 = x3 - coupling * C_H
    f4 = x4 - coupling * He_H
    f5 = x5 - coupling * N_H
    return jnp.stack([f1, f2, f3, f4, f5])


def _jax_newton(residual_fn, m0, args, max_iter=50, tol=1e-12):
    """Small dense Newton via `lax.while_loop` on residual norm.

    Replaces `scipy.optimize.fsolve` for the 5-element `_abun_lowT`
    system. The Jacobian is built with `jax.jacrev`; the linear solve
    is `jnp.linalg.solve` (5x5 dense). For the standard initial guess
    `[0.9, 0.1, 0, 0, 0]` and elemental ratios near solar, this
    converges in ~5-10 iterations.
    """
    jac_fn = jax.jacrev(residual_fn)

    def cond_fn(state):
        _, residual_norm, iter_count = state
        return jnp.logical_and(residual_norm > tol, iter_count < max_iter)

    def body_fn(state):
        m, _, iter_count = state
        r = residual_fn(m, *args)
        J = jac_fn(m, *args)
        delta = jnp.linalg.solve(J, r)
        m_new = m - delta
        r_new = residual_fn(m_new, *args)
        return (m_new, jnp.linalg.norm(r_new), iter_count + 1)

    initial_state = (
        jnp.asarray(m0, dtype=jnp.float64),
        jnp.asarray(jnp.inf, dtype=jnp.float64),
        jnp.asarray(0, dtype=jnp.int32),
    )
    final_m, _, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    return final_m


def compute_atom_ini(y, compo_arr=compo_array):
    """`atom_ini[a] = Σ_i compo[i,a] * Σ_z y[z,i]`. y (nz, ni), compo_arr (ni, n_atoms)."""
    return jnp.einsum("zi,ia->a", y, compo_arr)


def _run_fastchem(data_atm) -> None:
    """Write FastChem inputs and invoke the binary, holding the cross-process lock."""
    _FC_DIR.mkdir(exist_ok=True)
    _FC_SENTINEL.touch(exist_ok=True)
    with open(_FC_SENTINEL, "r") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            _run_fastchem_locked(data_atm)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _run_fastchem_locked(data_atm) -> None:
    """Inner FastChem driver. Caller must already hold the flock."""
    solar_ele = _FC_INPUT / "solar_element_abundances.dat"
    if vulcan_cfg.use_ion is True:
        copyfile(
            _FC_INPUT / "parameters_ion.dat",
            _FC_INPUT / "parameters.dat",
        )
    else:
        copyfile(
            _FC_INPUT / "parameters_wo_ion.dat",
            _FC_INPUT / "parameters.dat",
        )

    with open(solar_ele, "r") as f:
        new_str = ""
        ele_list = list(vulcan_cfg.atom_list)
        ele_list.remove("H")

        fc_list = ["C", "N", "O", "S", "P", "Si", "Ti", "V", "Cl",
                   "K", "Na", "Mg", "F", "Ca", "Fe"]

        if vulcan_cfg.use_solar is True:
            new_str = f.read()
            print("Initializing with the default solar abundance.")
        else:
            print("Initializing with the customized elemental abundance:")
            print("{:4}".format("H") + str("1."))
            for line in f.readlines():
                li = line.split()
                sp = li[0].strip()
                if sp in ele_list:
                    sp_abun = getattr(vulcan_cfg, sp + "_H")
                    fc_abun = 12.0 + np.log10(sp_abun)
                    line = sp + "\t" + "{0:.4f}".format(fc_abun) + "\n"
                    print("{:4}".format(sp) + "{0:.4E}".format(sp_abun))
                elif sp in fc_list:
                    sol_ratio = li[1].strip()
                    if hasattr(vulcan_cfg, "fastchem_met_scale"):
                        met_scale = vulcan_cfg.fastchem_met_scale
                    else:
                        met_scale = 1.0
                        print(
                            "fastchem_met_scale not specified in vulcan_cfg. "
                            "Using solar metallicity for other elements not included in vulcan."
                        )
                    new_ratio = float(sol_ratio) + np.log10(met_scale)
                    line = sp + "\t" + "{0:.4f}".format(new_ratio) + "\n"
                new_str += line

        with open(_FC_INPUT / "element_abundances_vulcan.dat", "w") as fout:
            fout.write(new_str)

    _FC_VULCAN_TP.parent.mkdir(parents=True, exist_ok=True)
    with open(_FC_VULCAN_TP, "w") as fout:
        ost = "#p (bar)    T (K)\n"
        for n, p in enumerate(data_atm.pco):
            ost += "{:.3e}".format(p / 1.0e6) + "\t" + "{:.1f}".format(data_atm.Tco[n]) + "\n"
        ost = ost[:-1]
        fout.write(ost)

    try:
        subprocess.check_call(
            ["./fastchem input/config.input"],
            shell=True,
            cwd=str(_FC_DIR),
        )
    except Exception:
        print("\n FastChem cannot run properly. Try compile it by running make under /fastchem_vulcan\n")
        raise


def _build_charge_list_if_ion(charge_list: list[str]) -> None:
    """Append every species with non-zero electron count to `charge_list`."""
    for sp in species:
        if compo[compo_row.index(sp)]["e"] != 0:
            charge_list.append(sp)


def _load_eq_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Run FastChem and parse `vulcan_EQ.dat` into `(nz, ni)`.

    Invoke + read + cleanup all happen inside one flock so a concurrent
    worker can't clobber the output mid-read.
    """
    _FC_DIR.mkdir(exist_ok=True)
    _FC_SENTINEL.touch(exist_ok=True)
    with open(_FC_SENTINEL, "r") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            _run_fastchem_locked(data_atm)
            fc = np.genfromtxt(
                _FC_VULCAN_EQ,
                names=True,
                dtype=None,
                skip_header=0,
            )
            # Unlink under the lock so we can't strip another worker's
            # freshly-written output before they parse it.
            try:
                _FC_VULCAN_EQ.unlink()
            except FileNotFoundError:
                pass
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    gas_tot = np.asarray(data_atm.M)
    charge_list: list[str] = []
    for sp in species:
        sp_idx = species.index(sp)
        if sp == "P":
            y[:, sp_idx] = fc["P_1"] * gas_tot
            continue
        if sp in fc.dtype.names:
            y[:, sp_idx] = fc[sp] * gas_tot
        else:
            print(sp + " not included in fastchem.")
        if vulcan_cfg.use_ion is True:
            if compo[compo_row.index(sp)]["e"] != 0:
                charge_list.append(sp)
    return y, charge_list


def _load_vulcan_ini_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Load `y` from a previous `.vul` file via pickle."""
    print("Initializing with compositions from the prvious run " + vulcan_cfg.vul_ini)
    with open(vulcan_cfg.vul_ini, "rb") as handle:
        vul_data = pickle.load(handle)
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    prev_species = vul_data["variable"]["species"]
    prev_y = vul_data["variable"]["y"]
    for sp in species:
        if sp in prev_species:
            y[:, species.index(sp)] = prev_y[:, prev_species.index(sp)]
        else:
            print(sp + " not included in the prvious run.")
    charge_list: list[str] = []
    if vulcan_cfg.use_ion is True:
        _build_charge_list_if_ion(charge_list)
    return y, charge_list


def _load_table_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Load `y` from a per-layer mixing-ratio text table."""
    table = np.genfromtxt(vulcan_cfg.vul_ini, names=True, dtype=None, skip_header=1)
    if not len(data_atm.pco) == len(table["Pressure"]):
        print("Warning! The initial profile has different layers than the current setting...")
        raise IOError("Initial profile / cfg layer mismatch")
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    n_0 = np.asarray(data_atm.n_0)
    for sp in species:
        y[:, species.index(sp)] = n_0 * table[sp]
    return y, []


def _load_const_mix_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Load `y` from `vulcan_cfg.const_mix` (a per-species mixing dict)."""
    print("Initializing with constant (well-mixed): " + str(vulcan_cfg.const_mix))
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    gas_tot = np.asarray(data_atm.M)
    for sp in vulcan_cfg.const_mix.keys():
        y[:, species.index(sp)] = gas_tot * vulcan_cfg.const_mix[sp]
    charge_list: list[str] = []
    if vulcan_cfg.use_ion is True:
        _build_charge_list_if_ion(charge_list)
    return y, charge_list


def _load_const_lowT_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Solve the 5-mol H2/H2O/CH4/He/NH3 system via JAX Newton."""
    O_H = float(vulcan_cfg.O_H)
    C_H = float(vulcan_cfg.C_H)
    He_H = float(vulcan_cfg.He_H)
    N_H = float(vulcan_cfg.N_H)
    m0 = jnp.array([0.9, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)
    ini_mol = np.asarray(_jax_newton(
        _abun_lowT_residual, m0, (O_H, C_H, He_H, N_H),
    ))

    nz_ = vulcan_cfg.nz
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    gas_tot = np.asarray(data_atm.M)
    h2_idx = species.index("H2")
    h2o_idx = species.index("H2O")
    ch4_idx = species.index("CH4")
    nh3_idx = species.index("NH3")
    he_idx = species.index("He")
    for i in range(nz_):
        y[i, h2_idx] = ini_mol[0] * gas_tot[i]
        y[i, h2o_idx] = ini_mol[1] * gas_tot[i]
        y[i, ch4_idx] = ini_mol[2] * gas_tot[i]
        y[i, nh3_idx] = ini_mol[4] * gas_tot[i]
        y[i, he_idx] = gas_tot[i] - np.sum(y[i, :])
    return y, []


_MODE_DISPATCH = {
    "EQ":         _load_eq_y,
    "vulcan_ini": _load_vulcan_ini_y,
    "table":      _load_table_y,
    "const_mix":  _load_const_mix_y,
    "const_lowT": _load_const_lowT_y,
}


def _apply_condense(y: np.ndarray, data_atm) -> np.ndarray:
    """Apply the `use_condense` initial-cold-trap clip.

    Mutates `data_atm.sat_mix` / `data_atm.conden_min_lev` and (for
    H2O + `use_sat_surfaceH2O`) `vulcan_cfg.use_fix_sp_bot`. Returns
    the clipped `y` so the caller can recompute ymix.
    """
    if vulcan_cfg.use_condense is not True:
        return y

    for sp in vulcan_cfg.condense_sp:
        sp_idx = species.index(sp)
        data_atm.sat_mix[sp] = data_atm.sat_p[sp] / data_atm.pco
        data_atm.sat_mix[sp] = np.minimum(1.0, data_atm.sat_mix[sp])

        if sp == "H2O":
            data_atm.sat_mix[sp] *= vulcan_cfg.humidity
            if vulcan_cfg.use_sat_surfaceH2O is True:
                vulcan_cfg.use_fix_sp_bot[sp] = data_atm.sat_mix[sp][0]
                print(
                    "\nThe fixed surface water is now reset by condensation and humidity to "
                    + str(vulcan_cfg.use_fix_sp_bot[sp])
                )
                # The ymix write is overwritten by the final renormalisation;
                # only the y replacement survives, so just update y here.
                y[:, sp_idx] = data_atm.sat_mix[sp][0] * data_atm.n_0

        if vulcan_cfg.use_ini_cold_trap is True:
            if vulcan_cfg.ini_mix != "table":
                if vulcan_cfg.use_sat_surfaceH2O is True:
                    conden_bot = 0
                else:
                    conden_bot = np.argmax(
                        data_atm.n_0 * data_atm.sat_mix[sp] <= y[:, sp_idx]
                    )
                sat_rho = data_atm.n_0 * data_atm.sat_mix[sp]
                conden_status = y[:, sp_idx] >= sat_rho
                y[:, sp_idx] = np.minimum(
                    data_atm.n_0 * data_atm.sat_mix[sp],
                    y[:, sp_idx],
                )
                if list(y[conden_status, sp_idx]):
                    min_sat = np.amin(data_atm.sat_mix[sp][conden_status])
                    conden_min_lev = np.where(data_atm.sat_mix[sp] == min_sat)[0][0]
                    data_atm.conden_min_lev[sp] = conden_min_lev
                    print(
                        sp + " condensed from nz = " + str(conden_bot)
                        + " to the minimum level nz = " + str(conden_min_lev)
                        + " (cold trap)"
                    )
                    y[conden_min_lev:, sp_idx] = (
                        data_atm.sat_mix[sp][conden_min_lev]
                        * data_atm.n_0[conden_min_lev:]
                    )
    return y


def _compute_ymix(y: np.ndarray) -> np.ndarray:
    """Per-layer normalisation. Excludes condensed-out species when
    `use_condense=True` (matches master's `non_gas_sp` carve-out)."""
    if vulcan_cfg.use_condense is True:
        exc_conden = [
            i for i in range(chem_funs.ni)
            if species[i] not in vulcan_cfg.non_gas_sp
        ]
        ysum = np.sum(y[:, exc_conden], axis=1).reshape((-1, 1))
    else:
        ysum = np.sum(y, axis=1).reshape((-1, 1))
    return y / ysum


def compute_initial_abundance(data_atm) -> IniAbunOutputs:
    """Run the configured `ini_mix` mode and return a typed pytree.

    Side effect: when `use_condense=True`, the legacy `data_atm` container
    is mutated (saturation profiles, cold-trap min level, optional surface
    H2O override). The pytree carries the gas-phase composition only.
    """
    mix = vulcan_cfg.ini_mix
    if mix not in _MODE_DISPATCH:
        raise IOError(
            "\nInitial mixing ratios unknown. Check the setting in vulcan_cfg.py."
        )
    y, charge_list = _MODE_DISPATCH[mix](data_atm)
    y = _apply_condense(y, data_atm)
    ymix = _compute_ymix(y)

    if vulcan_cfg.use_ion is True:
        if not charge_list:
            print("vulcan_cfg.use_ion = True but the network with ions is not supplied.\n")
            raise IOError(
                "vulcan_cfg.use_ion = True but the network with ions is not supplied.\n"
            )
        if "e" in charge_list:
            charge_list = [c for c in charge_list if c != "e"]

    atom_ini_arr = np.asarray(compute_atom_ini(jnp.asarray(y)))
    n_atoms = atom_ini_arr.shape[0]
    return IniAbunOutputs(
        y=jnp.asarray(y),
        ymix=jnp.asarray(ymix),
        y_ini=jnp.asarray(y),
        atom_ini=jnp.asarray(atom_ini_arr),
        atom_loss=jnp.zeros(n_atoms, dtype=jnp.float64),
        atom_conden=jnp.zeros(n_atoms, dtype=jnp.float64),
        charge_list=tuple(charge_list),
    )


class InitialAbun:
    """Legacy-mutation facade matching `atm_setup.Atm`'s pattern."""

    def __init__(self):
        self.atom_list = vulcan_cfg.atom_list

    def ini_y(self, data_var, data_atm):
        outputs = compute_initial_abundance(data_atm)
        data_var.y = np.asarray(outputs.y)
        data_var.ymix = np.asarray(outputs.ymix)
        data_var.y_ini = np.asarray(outputs.y_ini)
        if vulcan_cfg.use_ion is True:
            data_var.charge_list = list(outputs.charge_list)
        return data_var

    def ele_sum(self, data_var):
        atoms_jax = compute_atom_ini(jnp.asarray(data_var.y))
        atoms_np = np.asarray(atoms_jax)
        loss_ex = list(getattr(vulcan_cfg, "loss_ex", []))
        # cfg.atom_list may reorder/subset composition.atom_list; look up the
        # column for each cfg atom by name in compo_array.
        for atom in self.atom_list:
            if atom in loss_ex:
                continue
            col = _COMPO_ATOMS.index(atom)
            data_var.atom_ini[atom] = float(atoms_np[col])
            data_var.atom_loss[atom] = 0.0
            data_var.atom_conden[atom] = 0.0
        return data_var
