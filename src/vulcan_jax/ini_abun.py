"""JAX-native initial-abundance setup.

Five `ini_mix` modes:
- `EQ`         Gibbs-minimization equilibrium (ExoGibbs) on the network's own
               NASA-9 thermochemistry
- `const_mix`  apply a per-species mixing dict from cfg
- `vulcan_ini` restore composition from a previous `.vul` file
- `table`      read a per-layer mixing-ratio table
- `const_lowT` solve the 5-mol H2/H2O/CH4/He/NH3 system via JAX Newton

`compute_initial_abundance` returns a typed `IniAbunOutputs`. The
`InitialAbun` class is a legacy facade that mutates `data_var`/`data_atm`.

The EQ seed (`eq_seed`) is end-to-end JAX: it minimizes the Gibbs energy of
the loaded network's own gas species using the same NASA-9 polynomials the
reverse rates use, so the seed and the kinetics cannot disagree about
thermochemistry. It jits and vmaps over columns. It is NOT a differentiable
map: `custom_jvp` returns a zero tangent, because the seed is where the
integration starts, not part of the steady state it converges to.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:  # the exogibbs imports are function-local (pandas at import)
    from exogibbs.api.gas import EquilibriumOptions
    from exogibbs.thermo.models import ChemicalSetup

from .config import default_config
from . import chem_funs
from . import gibbs
from . import rates_jax
from .composition import (
    compo,
    compo_row,
    compo_array,
    species,
    atom_list as _COMPO_ATOMS,
)
from .state import IniAbunOutputs
from ._paths import resolve_data_path

jax.config.update("jax_enable_x64", True)

_CFG = default_config()

# --- the equilibrium seed ---------------------------------------------------
# Network-frozen inputs (species, elements, NASA-9 coefficients) are resolved
# once at import; everything the config owns is read at CALL time, because
# `state._cfg_overlay` rewrites `_CFG` per run.

DEFAULT_ABUNDANCE_FILE = "thermo/solar_element_abundances.dat"


def _abundance_path() -> Path:
    """The configured elemental-abundance preset file."""
    return resolve_data_path(
        str(getattr(_CFG, "fastchem_solar_abundance_file", DEFAULT_ABUNDANCE_FILE))
    )


def _condensate_species() -> set[str]:
    """Species produced by a condensation reaction.

    They have no gas-phase entry in the minimizer: the seed solves the gas
    equilibrium and the cold-trap clip (`_apply_condense`) runs after it.
    """
    net = chem_funs._NETWORK
    out: set[str] = set()
    for i in range(1, net.nr + 1):
        if not (net.is_conden[i] and net.is_forward[i]):
            continue
        for slot, sp_idx in enumerate(net.product_idx[i]):
            if net.product_stoich[i, slot] != 0.0:
                out.add(net.species[sp_idx])
    return out


def _make_hvector(coeffs: np.ndarray):
    """`T -> mu0(T)/(RT)` (K,) at the 1 bar standard state.

    That is the convention ExoGibbs's `gk = h + ln n_k - ln n_tot +
    ln(P/Pref)` expects, so the seed takes P in bar with `Pref=1.0`.
    `gibbs_sp_vector` is the same function the reverse rates use, evaluated
    on a one-layer grid because the minimizer calls `h` per layer. `coeffs`
    stays NumPy: the closure is cached in `_SEED`, so a `jnp.asarray` here
    would be a tracer whenever the first seed call happens inside a trace.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)

    def hvector(T):
        return rates_jax.gibbs_sp_vector(coeffs, jnp.atleast_1d(T))[:, 0]

    return hvector


def _thermo_dir() -> Path:
    """`thermo/` beside the network file, else the packaged one.

    The same resolution `rates_jax.setup_var_k` uses for the rate build, so
    the seed and the reverse rates read one table from one place.
    """
    beside = resolve_data_path(_CFG.network).parent
    if (beside / "NASA9").exists():
        return beside
    return Path(__file__).resolve().parent / "thermo"


def _build_seed_setup() -> tuple[ChemicalSetup, np.ndarray, tuple[str, ...]]:
    """Build the ExoGibbs setup for the loaded network's gas species.

    Every array in the returned setup is NumPy. The result is cached for the
    process, and the first seed call may happen inside a trace (vulcan-forward
    calls `eq_seed` from a jitted forward model); a JAX array built there is a
    tracer that escapes into every later trace.
    """
    from exogibbs.thermo.models import ChemicalSetup

    charged = {sp for sp in species if compo[compo_row.index(sp)]["e"] != 0}
    excluded = _condensate_species() | charged
    seed_idx = np.array(
        [i for i, sp in enumerate(species) if sp not in excluded], dtype=np.int64
    )
    coeffs, present = gibbs.load_nasa9(tuple(species), _thermo_dir())
    missing = [species[i] for i in seed_idx if not present[i]]
    if missing:
        raise RuntimeError(
            f"the equilibrium seed needs NASA-9 data for every gas species of "
            f"{_CFG.network!r}, but thermo/NASA9/ has no file for: "
            f"{', '.join(missing)}. Add them, or initialize with "
            "ini_mix='const_mix'."
        )
    counts = np.asarray(compo_array)[seed_idx]
    elem_cols = [
        j
        for j, atom in enumerate(_COMPO_ATOMS)
        if atom != "e" and counts[:, j].sum() > 0.0
    ]
    elements = tuple(_COMPO_ATOMS[j] for j in elem_cols)
    setup = ChemicalSetup(
        formula_matrix=np.ascontiguousarray(counts[:, elem_cols].T),
        hvector_func=_make_hvector(coeffs[seed_idx]),
        elements=elements,
        species=tuple(species[i] for i in seed_idx),
    )
    return setup, seed_idx, elements


_SEED: tuple[ChemicalSetup, np.ndarray, tuple[str, ...]] | None = None


def _seed() -> tuple[ChemicalSetup, np.ndarray, tuple[str, ...]]:
    """`(setup, seed species indices, element names)`, built once on first use.

    Lazy on purpose: it reads one `thermo/NASA9/<sp>.txt` per species, and a
    run with any other `ini_mix` must not pay for that at import.
    """
    global _SEED
    if _SEED is None:
        _SEED = _build_seed_setup()
    return _SEED


def seed_elements() -> tuple[str, ...]:
    """Elements the seed solves for, in the order `b` is indexed."""
    return _seed()[2]

# Sequential warm start, layer by layer. NOTE the name is ExoGibbs's, not
# ours: `scan_hot_from_bottom` FLIPS the input arrays, so on VULCAN's grid
# (index 0 is the BOTTOM, `pco = logspace(P_b, P_t)`) it starts at the COLD
# TOP and walks down. `scan_hot_from_top` is the true hot start here and needs
# ~10% fewer minimizer iterations, but it moves the seed and the
# HD189_vulcan3 gate then does not converge within its step cap; the seed cost
# is a one-off and that gate is not. Both converge everywhere at 1e-12 and
# the independent per-layer cold solve ("vmap_cold") does not; measured in
# notes.md 2.9.
_SEED_METHOD = "scan_hot_from_bottom"

_OPTIONS_CACHE: dict[tuple[float, int], EquilibriumOptions] = {}
_SEED_JIT: dict[tuple[float, int], object] = {}


def _seed_key() -> tuple[float, int]:
    """The config's solver controls, read at CALL time (`_cfg_overlay`)."""
    return (float(_CFG.fastchem_newton_tol), int(_CFG.fastchem_newton_max_iter))


def _seed_options() -> EquilibriumOptions:
    """Solver options for the current config, interned.

    ExoGibbs keys its scan-body cache on `id(options)`, so a fresh dataclass
    per call would rebuild the profile scan every time.
    """
    from exogibbs.api.gas import EquilibriumOptions

    key = _seed_key()
    if key not in _OPTIONS_CACHE:
        _OPTIONS_CACHE[key] = EquilibriumOptions(
            epsilon_crit=key[0], max_iter=key[1], method=_SEED_METHOD
        )
    return _OPTIONS_CACHE[key]


def _seed_jit():
    """`eq_seed` jitted, one wrapper per set of solver controls.

    `eq_seed` reads the tolerance and the iteration cap at TRACE time, and
    JAX keys its trace cache on the traced function plus the argument shapes
    -- not on our key -- so `jax.jit(eq_seed)` under a new key would reuse the
    trace that baked in the old controls. A fresh closure per key is a fresh
    cache entry. Without the jit at all the host path re-traces the 150-layer
    scan on every call: 223 ms against 3.2 ms jitted, on the HD189 column.
    """
    key = _seed_key()
    if key not in _SEED_JIT:
        _SEED_JIT[key] = jax.jit(lambda Tco, p_bar, b: eq_seed(Tco, p_bar, b))
    return _SEED_JIT[key]


@jax.custom_jvp
def eq_seed(Tco, p_bar, b):
    """Equilibrium mixing ratios `(nz, ni)` at `Tco` (K) and `p_bar` (bar).

    `b` is the elemental abundance vector in `seed_elements()` order; only its
    ratios matter. Species outside the seed (condensates, ions) are exactly
    zero. A column with a layer that did not converge comes back all-NaN, so
    no caller can use half a solution.

    The inputs are cast to float64: a float32 caller would otherwise give the
    minimizer a mixed-dtype loop carry.
    """
    from exogibbs.api.gas import solve_profile

    setup, seed_idx, _ = _seed()
    res, diag = solve_profile(
        setup,
        jnp.asarray(Tco, dtype=jnp.float64),
        jnp.asarray(p_bar, dtype=jnp.float64),
        jnp.asarray(b, dtype=jnp.float64),
        Pref=1.0,
        options=_seed_options(),
        return_diagnostics=True,
    )
    ok = jnp.all(diag["converged"]) & jnp.all(jnp.isfinite(res.x))
    y = jnp.zeros((Tco.shape[0], chem_funs.ni), dtype=jnp.float64)
    return y.at[:, jnp.asarray(seed_idx)].set(jnp.where(ok, res.x, jnp.nan))


@eq_seed.defjvp
def _eq_seed_jvp(primals, tangents):
    """Zero tangent: the seed is an initial condition, not a model output.

    ExoGibbs's kernel carries a `custom_vjp` and no forward rule, so a jvp
    must never reach it.
    """
    del tangents
    primal = eq_seed(*primals)
    return primal, jnp.zeros_like(primal)


def seed_diagnostics(Tco, p_bar, b) -> dict:
    """Per-layer `converged` / `n_iter` / `final_residual` for one column."""
    from exogibbs.api.gas import solve_profile

    _, diag = solve_profile(
        _seed()[0],
        jnp.asarray(Tco, dtype=jnp.float64),
        jnp.asarray(p_bar, dtype=jnp.float64),
        jnp.asarray(b, dtype=jnp.float64),
        Pref=1.0,
        options=_seed_options(),
        return_diagnostics=True,
    )
    return {k: np.asarray(v) for k, v in diag.items()}


def read_abundances(path: Path) -> dict[str, float]:
    """Parse a `SYMBOL log10(n_X/n_H)+12` preset into n_X/n_H, with H == 1."""
    dex: dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            row = line.split("#", 1)[0].split()
            if len(row) >= 2:
                dex[row[0]] = float(row[1])
    if "H" not in dex:
        raise ValueError(
            f"{path}: no H row. The preset is read as log10(n_X/n_H)+12, "
            "so hydrogen sets the scale and must be present."
        )
    return {sp: 10.0 ** (value - dex["H"]) for sp, value in dex.items()}


def _preset_vector() -> np.ndarray:
    """`b` (E,) straight from the configured preset file."""
    path = _abundance_path()
    preset = read_abundances(path)
    absent = [sp for sp in seed_elements() if sp not in preset]
    if absent:
        raise RuntimeError(
            f"{path} has no row for {', '.join(absent)}, which the "
            f"{_CFG.network!r} network needs. Every element of the loaded "
            "network must be in the abundance preset."
        )
    return np.array([preset[sp] for sp in seed_elements()], dtype=np.float64)


def _base_vector() -> np.ndarray:
    """`b` (E,) with the config's `<X>_H` applied on top of the preset.

    Precedence: an explicit `ratios` entry, then `cfg.<X>_H` for the elements
    of `cfg.atom_list` other than H, then the preset file. Helium comes from
    the file for every shipped atom list (`He_H` is a `const_lowT` knob).
    """
    b = _preset_vector()
    elements = seed_elements()
    for atom in _CFG.atom_list:
        if atom != "H" and atom in elements:
            b[elements.index(atom)] = float(getattr(_CFG, atom + "_H"))
    return b


def ratio_indices(names) -> jnp.ndarray:
    """Positions of `names` in the seed's element vector, resolved once.

    The array path (`element_vector`) takes these so a batched caller never
    resolves element names inside a traced function.
    """
    elements = seed_elements()
    unknown = [sp for sp in names if sp not in elements]
    if unknown:
        raise KeyError(
            f"{unknown} are not elements of the loaded network "
            f"({', '.join(elements)}); setting them would do nothing."
        )
    return jnp.asarray([elements.index(sp) for sp in names], dtype=jnp.int32)


def element_vector(ratios, idx) -> jnp.ndarray:
    """`b` (E,) with the elements at `idx` replaced by `ratios`. Traceable.

    Starts from `_element_vector()`, the host path's own no-override vector,
    so `use_solar` and the `cfg.<X>_H` overrides have the same precedence on
    both paths, and it is bit-identical to `_element_vector({name: value,
    ...})` for the same elements.
    """
    return jnp.asarray(_element_vector()).at[idx].set(
        jnp.asarray(ratios, dtype=jnp.float64)
    )


def _element_vector(ratios: dict | None = None) -> np.ndarray:
    """`b` (E,) for the current config, host-side.

    The base is the same on both paths -- the preset when `use_solar` is
    true, else the preset with `cfg.<X>_H` applied -- and `ratios` replaces
    entries on top of it, exactly as `element_vector` does with `idx`.
    """
    b = _preset_vector() if _CFG.use_solar is True else _base_vector()
    if ratios:
        elements = seed_elements()
        for sp, value in ratios.items():
            if sp not in elements:
                raise KeyError(
                    f"ratios[{sp!r}] is not an element of the loaded network "
                    f"({', '.join(elements)})."
                )
            b[elements.index(sp)] = float(value)
    return b


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
    is `jnp.linalg.solve` (5x5 dense). Production callers pass
    `max_iter` / `tol` from `_CFG.fastchem_newton_max_iter` and
    `_CFG.fastchem_newton_tol`; the defaults here are kept for
    direct test callers.
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


def operator_column_weights(dz):
    """Cell weights (nz,) that the flux-form transport operator conserves.

    The diffusion divergence divides by the interface-centered spacing
    `dz_ave = 0.5*(dz[j-1]+dz[j])` (`jax_step`, mirroring master op.py), so
    with zero-flux boundaries the discrete transport invariant is
    `Σ_j n_j * w_j` with `w_0 = dzi[0]`, `w_j = 0.5*(dzi[j-1]+dzi[j])`,
    `w_{nz-1} = dzi[-1]` — NOT `Σ_j n_j * dz_j`. On a uniform grid w == dz.
    """
    dz = jnp.asarray(dz, dtype=jnp.float64)
    dzi = 0.5 * (dz[1:] + dz[:-1])
    return jnp.concatenate([dzi[:1], 0.5 * (dzi[:-1] + dzi[1:]), dzi[-1:]])


def column_atoms(y, dz, compo_arr=compo_array):
    """Operator-weighted atom column (n_atoms,).

    `Σ_z w_z * y[z,i] * compo[i,a]` with `w = operator_column_weights(dz)`:
    the invariant the discrete transport conserves on the grid `dz`.
    """
    w = operator_column_weights(dz)
    return jnp.einsum("z,zi,ia->a", w, jnp.asarray(y, dtype=jnp.float64), compo_arr)


def column_atom_loss(y, y_ini, dz, compo_arr=compo_array):
    """Operator-weighted column-conservation residual per atom. Returns (n_atoms,).

    Relative change, between `y_ini` and `y`, of the column integral
    `Σ_z w_z * compo[i,a] * y[z,i]` with `w = operator_column_weights(dz)` —
    the quantity the discretized transport actually conserves on a
    nonuniform grid. Step acceptance uses the unweighted `atom_loss`
    (master parity); this is the `report_column_atom_loss` print. Both
    columns are weighted on the `dz` passed in, so a grid refresh moves the
    result; the certificate's C23 term accumulates the per-step change
    instead, each step on its own grid (`outer_loop`, `budget_err`).
    """
    col = column_atoms(y, dz, compo_arr)
    col0 = column_atoms(y_ini, dz, compo_arr)
    # compo_arr spans every composition-table element; atoms absent from the
    # loaded network have a zero column and no budget — report 0, not 0/0.
    return jnp.where(col0 == 0.0, 0.0, (col - col0) / jnp.where(col0 == 0.0, 1.0, col0))


def _build_charge_list_if_ion(charge_list: list[str]) -> None:
    """Append every species with non-zero electron count to `charge_list`."""
    for sp in species:
        if compo[compo_row.index(sp)]["e"] != 0:
            charge_list.append(sp)


def eq_column(pco, Tco, M, ratios=None) -> np.ndarray:
    """Equilibrium column ``(nz, ni)`` in absolute number densities.

    ``pco`` is in dyne/cm^2, ``Tco`` in K and ``M`` the total gas density.
    ``ratios`` (element -> number ratio to H) overrides the config's ``<X>_H``
    for the elements it names. Species outside the seed stay zero. Host-side:
    a batched caller should drive `eq_seed` under `vmap` instead.
    """
    if _CFG.use_ion is True:
        raise RuntimeError(
            "ini_mix='EQ' does not support use_ion=True: the equilibrium seed "
            "is gas-phase and neutral, with no electron balance. Initialize an "
            "ionized run with ini_mix='const_mix' or 'vulcan_ini'."
        )
    b = _element_vector(ratios)
    Tco = np.asarray(Tco, dtype=np.float64)
    p_bar = np.asarray(pco, dtype=np.float64) / 1.0e6
    if not ratios:
        print(
            f"Equilibrium seed from {_abundance_path()}: "
            + ", ".join(
                f"{sp}/H={b[seed_elements().index(sp)]:.8g}"
                for sp in ("He", "C", "N", "O", "S")
                if sp in seed_elements()
            )
        )
    ymix = np.asarray(
        _seed_jit()(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b))
    )
    if not np.isfinite(ymix).all():
        conv = seed_diagnostics(Tco, p_bar, b)["converged"]
        bad = np.flatnonzero(~conv)
        raise RuntimeError(
            f"the equilibrium seed did not converge in "
            f"{_CFG.fastchem_newton_max_iter} iterations at "
            f"{bad.size} of {Tco.size} layers, tol={_CFG.fastchem_newton_tol:g}: "
            + "; ".join(
                f"layer {i} (T={Tco[i]:.1f} K, p={p_bar[i]:.3e} bar)"
                for i in bad[:5]
            )
            + ". Raise fastchem_newton_max_iter."
        )
    return ymix * np.asarray(M, dtype=np.float64)[:, None]


def _load_eq_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Seed the atmosphere's own T-P at the config's elemental abundances."""
    return eq_column(data_atm.pco, data_atm.Tco, data_atm.M), []


def _load_vulcan_ini_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Load `y` from a previous `.vul` file via pickle."""
    print("Initializing with compositions from the prvious run " + _CFG.vul_ini)
    with open(resolve_data_path(_CFG.vul_ini), "rb") as handle:
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
    if _CFG.use_ion is True:
        _build_charge_list_if_ion(charge_list)
    return y, charge_list


def _load_table_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Load `y` from a per-layer mixing-ratio text table."""
    table = np.genfromtxt(
        resolve_data_path(_CFG.vul_ini), names=True, dtype=None, skip_header=1
    )
    if not len(data_atm.pco) == len(table["Pressure"]):
        print(
            "Warning! The initial profile has different layers than the current setting..."
        )
        raise IOError("Initial profile / cfg layer mismatch")
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    n_0 = np.asarray(data_atm.n_0)
    for sp in species:
        y[:, species.index(sp)] = n_0 * table[sp]
    return y, []


def _load_const_mix_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Load `y` from `_CFG.const_mix` (a per-species mixing dict)."""
    print("Initializing with constant (well-mixed): " + str(_CFG.const_mix))
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    gas_tot = np.asarray(data_atm.M)
    for sp in _CFG.const_mix.keys():
        y[:, species.index(sp)] = gas_tot * _CFG.const_mix[sp]
    charge_list: list[str] = []
    if _CFG.use_ion is True:
        _build_charge_list_if_ion(charge_list)
    return y, charge_list


def _load_const_lowT_y(data_atm) -> tuple[np.ndarray, list[str]]:
    """Solve the 5-mol H2/H2O/CH4/He/NH3 system via JAX Newton."""
    O_H = float(_CFG.O_H)
    C_H = float(_CFG.C_H)
    He_H = float(_CFG.He_H)
    N_H = float(_CFG.N_H)
    m0 = jnp.array([0.9, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)
    max_iter = int(_CFG.fastchem_newton_max_iter)
    tol = float(_CFG.fastchem_newton_tol)
    ini_mol = np.asarray(
        _jax_newton(
            _abun_lowT_residual,
            m0,
            (O_H, C_H, He_H, N_H),
            max_iter=max_iter,
            tol=tol,
        )
    )

    nz_ = _CFG.nz
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
    "EQ": _load_eq_y,
    "vulcan_ini": _load_vulcan_ini_y,
    "table": _load_table_y,
    "const_mix": _load_const_mix_y,
    "const_lowT": _load_const_lowT_y,
}


def _apply_condense(y: np.ndarray, data_atm) -> np.ndarray:
    """Apply the `use_condense` initial-cold-trap clip.

    Mutates `data_atm.sat_mix` / `data_atm.conden_min_lev` and (for
    H2O + `use_sat_surfaceH2O`) `_CFG.use_fix_sp_bot`. Returns
    the clipped `y` so the caller can recompute ymix.
    """
    if _CFG.use_condense is not True:
        return y

    for sp in _CFG.condense_sp:
        sp_idx = species.index(sp)
        data_atm.sat_mix[sp] = data_atm.sat_p[sp] / data_atm.pco
        data_atm.sat_mix[sp] = np.minimum(1.0, data_atm.sat_mix[sp])

        if sp == "H2O":
            data_atm.sat_mix[sp] *= _CFG.humidity
            if _CFG.use_sat_surfaceH2O is True:
                _CFG.use_fix_sp_bot[sp] = data_atm.sat_mix[sp][0]
                print(
                    "\nThe fixed surface water is now reset by condensation and humidity to "
                    + str(_CFG.use_fix_sp_bot[sp])
                )
                # The ymix write is overwritten by the final renormalisation;
                # only the y replacement survives, so just update y here.
                y[:, sp_idx] = data_atm.sat_mix[sp][0] * data_atm.n_0

        if _CFG.use_ini_cold_trap is True:
            if _CFG.ini_mix != "table":
                if _CFG.use_sat_surfaceH2O is True:
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
                        sp
                        + " condensed from nz = "
                        + str(conden_bot)
                        + " to the minimum level nz = "
                        + str(conden_min_lev)
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
    if _CFG.use_condense is True:
        exc_conden = [
            i for i in range(chem_funs.ni) if species[i] not in _CFG.non_gas_sp
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
    mix = _CFG.ini_mix
    if mix not in _MODE_DISPATCH:
        raise IOError(
            "\nInitial mixing ratios unknown. Check the setting in the config."
        )
    y, charge_list = _MODE_DISPATCH[mix](data_atm)
    y = _apply_condense(y, data_atm)
    ymix = _compute_ymix(y)

    if _CFG.use_ion is True:
        if not charge_list:
            print("use_ion = True but the network with ions is not supplied.\n")
            raise IOError("use_ion = True but the network with ions is not supplied.\n")
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
        """Capture `cfg.atom_list` (cfg's possibly reordered/subset atom set)
        for use in `ele_sum`.
        """
        self.atom_list = _CFG.atom_list

    def ini_y(self, data_var, data_atm):
        """Compute the initial abundance and mutate `data_var.y` / `ymix` /
        `y_ini` (plus `charge_list` when `use_ion`); returns `data_var`.
        """
        outputs = compute_initial_abundance(data_atm)
        data_var.y = np.asarray(outputs.y)
        data_var.ymix = np.asarray(outputs.ymix)
        data_var.y_ini = np.asarray(outputs.y_ini)
        if _CFG.use_ion is True:
            data_var.charge_list = list(outputs.charge_list)
        return data_var

    def ele_sum(self, data_var):
        """Write per-atom initial totals onto `data_var.atom_ini` (with
        `atom_loss` / `atom_conden` zeroed) from `data_var.y`, looking up each
        cfg atom's column by name and skipping atoms in `cfg.loss_ex`; returns
        `data_var`.
        """
        atoms_jax = compute_atom_ini(jnp.asarray(data_var.y))
        atoms_np = np.asarray(atoms_jax)
        loss_ex = list(getattr(_CFG, "loss_ex", []))
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
