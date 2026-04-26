"""JAX-native drop-in replacement for VULCAN-master's auto-generated chem_funs.py.

This module exports the same public names that VULCAN-master/op.py, store.py,
and build_atm.py expect, but is built on top of the VULCAN-JAX network parser
+ JAX chemistry kernels rather than the SymPy code generator. Eliminates the
need for `make_chem_funs.py` at startup.

Public API (compatible with VULCAN-master/chem_funs.py):
    ni, nr                   - network metadata (int)
    spec_list                - ordered species (list[str])
    re_dict, re_wM_dict      - reaction text dicts keyed by parser-i
    chemdf(y, M, k)          - chemistry RHS (NumPy callable)
    symjac(y, M, k)          - chemistry Jacobian; raises NotImplementedError
                               (VULCAN-JAX uses chem.chem_jac directly)
    neg_symjac(y, M, k)      - same as symjac but negated; raises NotImplementedError
    Gibbs(i, T)              - K_eq for forward reaction i at temperature(s) T
    gibbs_sp(name, T)        - g_species / (RT) per layer
    h_RT, s_R, g_RT          - NASA-9 polynomial helpers
    cp_R, cp_R_sp            - heat capacity helpers

Lifecycle:
    Network is parsed at import time using `vulcan_cfg.network`. NASA-9
    coefficients are loaded from `thermo/NASA9/<sp>.txt` for each species.
    A K_eq cache stores per-temperature-array results to avoid recomputing
    when `Gibbs(i, T)` is called in a loop over reaction indices (the typical
    pattern in op.ReadRate.rev_rate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import vulcan_cfg
import network as _network
import chem as _chem
import gibbs as _gibbs

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# One-time setup at import: parse network, load NASA-9, pack JAX arrays.
# ---------------------------------------------------------------------------

_NETWORK = _network.parse_network(vulcan_cfg.network)
_NET_JAX = _chem.to_jax(_NETWORK)

# Resolve thermo directory relative to vulcan_cfg.network or by convention.
_THERMO_DIR = Path(vulcan_cfg.network).parent
if not (_THERMO_DIR / "NASA9").exists():
    # Fall back: thermo/ next to the script
    _THERMO_DIR = Path(__file__).resolve().parent / "thermo"

_NASA9_COEFFS, _NASA9_PRESENT = _gibbs.load_nasa9(_NETWORK.species, _THERMO_DIR)


# ---------------------------------------------------------------------------
# Public data: ni, nr, spec_list, re_dict, re_wM_dict
# ---------------------------------------------------------------------------

ni: int = _NETWORK.ni
nr: int = _NETWORK.nr
spec_list: list[str] = list(_NETWORK.species)


def _build_re_dicts(net: _network.Network) -> tuple[dict, dict]:
    """Reconstruct re_dict / re_wM_dict from the parsed network.

    Format (matches VULCAN's auto-generated dicts):
        re_dict[parser_i] = [[reactant_name, ...], [product_name, ...]]
            - species names with stoichiometric repetition (e.g. ['H', 'H'])
            - 'M' is excluded
        re_wM_dict[parser_i] = same, but 'M' is included for 3-body reactions
            so callers can reconstruct the original equation text.

    Both forward (odd parser_i) and reverse (even parser_i) entries are
    populated; reverse swaps reactants and products.
    """
    species = net.species
    re_dict: dict[int, list[list[str]]] = {}
    re_wM_dict: dict[int, list[list[str]]] = {}

    for i in range(1, net.nr + 1):
        if net.is_forward[i]:
            r_idx = net.reactant_idx[i]
            p_idx = net.product_idx[i]
            r_st = net.reactant_stoich[i]
            p_st = net.product_stoich[i]
            has_M_reac = bool(net.is_three_body[i])
            has_M_prod = bool(net.is_three_body[i + 1]) if (i + 1) <= net.nr else False
        else:
            # Reverse: reactants = forward's products, products = forward's reactants.
            f = i - 1
            r_idx = net.product_idx[f]
            p_idx = net.reactant_idx[f]
            r_st = net.product_stoich[f]
            p_st = net.reactant_stoich[f]
            # 3-body asymmetry: forward.has_M_reac maps to forward.is_three_body[f];
            # reverse "reactants" are forward "products", so reverse 3-body is
            # net.is_three_body[i] (the reverse slot itself, set during parsing).
            has_M_reac = bool(net.is_three_body[i])
            has_M_prod = bool(net.is_three_body[f])

        # Expand stoichiometry into name repetition; drop padding (sp_idx == ni)
        reactants: list[str] = []
        for sp_idx, st in zip(r_idx, r_st):
            if int(st) == 0 or sp_idx >= net.ni:
                continue
            reactants.extend([species[int(sp_idx)]] * int(st))
        products: list[str] = []
        for sp_idx, st in zip(p_idx, p_st):
            if int(st) == 0 or sp_idx >= net.ni:
                continue
            products.extend([species[int(sp_idx)]] * int(st))

        re_dict[i] = [list(reactants), list(products)]

        # With-M variant: append 'M' on each side that had it
        reactants_wM = list(reactants)
        products_wM = list(products)
        if has_M_reac:
            reactants_wM.append("M")
        if has_M_prod:
            products_wM.append("M")
        re_wM_dict[i] = [reactants_wM, products_wM]

    return re_dict, re_wM_dict


re_dict, re_wM_dict = _build_re_dicts(_NETWORK)


# ---------------------------------------------------------------------------
# Chemistry RHS (NumPy wrapper around chem.chem_rhs)
# ---------------------------------------------------------------------------

def _pack_k_dict(k) -> np.ndarray:
    """Convert var.k dict-or-array into a (nr+1, nz) NumPy array."""
    if isinstance(k, np.ndarray):
        return np.asarray(k, dtype=np.float64)
    if isinstance(k, dict):
        # Determine nz from any populated entry
        nz = None
        for v in k.values():
            arr = np.asarray(v)
            if arr.ndim >= 1:
                nz = arr.shape[0]
                break
        if nz is None:
            raise ValueError("Cannot infer nz from k dict (no array entries)")
        k_arr = np.zeros((nr + 1, nz), dtype=np.float64)
        for i, vec in k.items():
            if 1 <= int(i) <= nr:
                k_arr[int(i)] = np.asarray(vec, dtype=np.float64)
        return k_arr
    raise TypeError(f"Unexpected type for k: {type(k)}")


def chemdf(y, M, k) -> np.ndarray:
    """Chemistry RHS at all layers. NumPy in / NumPy out wrapper around chem.chem_rhs.

    Args:
        y: (nz, ni) number densities
        M: (nz,) total density
        k: var.k dict OR (nr+1, nz) array

    Returns:
        dydt: (nz, ni) NumPy array
    """
    y_np = np.asarray(y, dtype=np.float64)
    M_np = np.asarray(M, dtype=np.float64)
    k_arr = _pack_k_dict(k)
    out = _chem.chem_rhs(
        jnp.asarray(y_np), jnp.asarray(M_np), jnp.asarray(k_arr), _NET_JAX
    )
    return np.asarray(out, dtype=np.float64)


def symjac(y, M, k):
    """Full block-diagonal chemistry Jacobian as a flat (ni*nz, ni*nz) NumPy matrix.

    VULCAN-JAX's production hot path (Ros2JAX.solver) does NOT call this — it
    uses chem.chem_jac directly to get the (nz, ni, ni) block stack and feeds
    it into the JAX block-Thomas solver without ever materializing the full
    flat matrix. The flat form is only consumed by op.lhs_jac_tot, which is
    used by VULCAN-master's scipy solve_banded path (replaced by our JAX path).

    Materializing the full matrix for HD189 (nz=120, ni=93) is ~90 MB and
    serves no purpose in the JAX port. We raise rather than silently produce
    a giant array.
    """
    raise NotImplementedError(
        "chem_funs.symjac is not provided by the JAX-native chem_funs. "
        "VULCAN-JAX uses chem.chem_jac (block stack [nz, ni, ni]) directly "
        "via Ros2JAX.solver / jax_step.jax_ros2_step. If you need the flat "
        "(nz*ni, nz*ni) form, fall back to VULCAN-master/chem_funs.py."
    )


def neg_symjac(y, M, k):
    """Negated symjac (placeholder; raises like symjac)."""
    raise NotImplementedError(
        "chem_funs.neg_symjac is not provided by the JAX-native chem_funs. "
        "VULCAN-JAX subclasses op.Ros2 and overrides solver() to use "
        "chem.chem_jac directly without going through neg_symjac."
    )


# ---------------------------------------------------------------------------
# NASA-9 polynomial helpers (h/RT, s/R, g/RT, cp/R)
# ---------------------------------------------------------------------------

def h_RT(T, a):
    """h(T) / (RT) for a 10-element NASA-9 coefficient row."""
    T = np.asarray(T, dtype=np.float64)
    return (
        -a[0] / T**2
        + a[1] * np.log(T) / T
        + a[2]
        + a[3] * T / 2.0
        + a[4] * T**2 / 3.0
        + a[5] * T**3 / 4.0
        + a[6] * T**4 / 5.0
        + a[8] / T
    )


def s_R(T, a):
    """s(T) / R for a 10-element NASA-9 coefficient row."""
    T = np.asarray(T, dtype=np.float64)
    return (
        -a[0] / T**2 / 2.0
        - a[1] / T
        + a[2] * np.log(T)
        + a[3] * T
        + a[4] * T**2 / 2.0
        + a[5] * T**3 / 3.0
        + a[6] * T**4 / 4.0
        + a[9]
    )


def g_RT(T, a_low, a_high):
    """g(T) / (RT) = h/RT - s/R, with low-T branch for T<1000 and high-T branch for T>=1000."""
    T = np.asarray(T, dtype=np.float64)
    return (T < 1000.0) * (h_RT(T, a_low) - s_R(T, a_low)) + (
        T >= 1000.0
    ) * (h_RT(T, a_high) - s_R(T, a_high))


def gibbs_sp(name, T):
    """Per-species Gibbs energy g/(RT) at temperature(s) T. `name` is a species string."""
    j = spec_list.index(name)
    return g_RT(T, _NASA9_COEFFS[j, 0], _NASA9_COEFFS[j, 1])


def cp_R(T, a):
    """Heat capacity cp(T)/R for a NASA-9 coefficient row."""
    T = np.asarray(T, dtype=np.float64)
    return a[0] / T**2 + a[1] / T + a[2] + a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4


def cp_R_sp(name, T):
    """Per-species cp/R."""
    T = np.asarray(T, dtype=np.float64)
    if np.any(np.logical_or(T < 200.0, T > 6000.0)):
        print("T exceeds the valid range.")
    j = spec_list.index(name)
    return (T < 1000.0) * cp_R(T, _NASA9_COEFFS[j, 0]) + (T >= 1000.0) * cp_R(
        T, _NASA9_COEFFS[j, 1]
    )


# ---------------------------------------------------------------------------
# Equilibrium constant Gibbs(i, T)
# ---------------------------------------------------------------------------

# Cache K_eq arrays keyed by (T.shape, T.tobytes()). VULCAN's rev_rate calls
# Gibbs(i, T) once per forward reaction (~600 calls) with the same T array;
# without caching that's 600 redundant evaluations of K_eq_array.
_K_EQ_CACHE: dict = {}


def _K_eq_array_cached(T_np: np.ndarray) -> np.ndarray:
    key = (T_np.shape, T_np.tobytes())
    cached = _K_EQ_CACHE.get(key)
    if cached is not None:
        return cached
    g_sp = _gibbs.gibbs_sp_vector(_NASA9_COEFFS, T_np)
    K_eq = _gibbs.K_eq_array(_NETWORK, g_sp, T_np)
    # Cap cache at 4 entries (typical: one Tco array reused many times)
    if len(_K_EQ_CACHE) >= 4:
        _K_EQ_CACHE.clear()
    _K_EQ_CACHE[key] = K_eq
    return K_eq


def Gibbs(i, T):
    """Equilibrium constant K_eq for forward reaction i at temperature(s) T.

    Mirrors VULCAN's chem_funs.Gibbs(i, T): returns the value such that
    `k_reverse = k_forward / Gibbs(i, T)`. Includes the (k_B T / P0)^Δn factor
    where Δn = n_reac - n_prod (folded into the formula).

    Args:
        i: forward reaction parser-index (1, 3, 5, ...)
        T: scalar or array of temperatures (K)

    Returns:
        K_eq: float (if T scalar) or ndarray with same shape as T.
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
    K = _K_eq_array_cached(T_arr)[i]   # (T_arr.size,)
    if np.isscalar(T) or np.ndim(T) == 0:
        return float(K[0])
    return K


# ---------------------------------------------------------------------------
# Convenience: re-export NetworkArrays (some tests pull this directly).
# ---------------------------------------------------------------------------

NETWORK = _NETWORK   # kept for tooling that wants the structured object
