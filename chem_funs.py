"""JAX-native chem_funs: parses the network at import time and exposes the
same public surface (ni, nr, spec_list, re_dict, chemdf, Gibbs, gibbs_sp, ...)
as VULCAN-master's auto-generated chem_funs.py — backed by JAX kernels."""

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


_NETWORK = _network.parse_network(vulcan_cfg.network)
_NET_JAX = _chem.to_jax(_NETWORK)

# Locate thermo/NASA9 next to the network file, or fall back to repo-root.
_THERMO_DIR = Path(vulcan_cfg.network).parent
if not (_THERMO_DIR / "NASA9").exists():
    _THERMO_DIR = Path(__file__).resolve().parent / "thermo"

_NASA9_COEFFS, _NASA9_PRESENT = _gibbs.load_nasa9(_NETWORK.species, _THERMO_DIR)


ni: int = _NETWORK.ni
nr: int = _NETWORK.nr
spec_list: list[str] = list(_NETWORK.species)


def _build_re_dicts(net: _network.Network) -> tuple[dict, dict]:
    """Reconstruct re_dict / re_wM_dict from the parsed network.

    `re_dict[i] = [[reactants], [products]]` with stoichiometric repetition
    (e.g. ['H', 'H']) and 'M' excluded. `re_wM_dict[i]` is the same with
    'M' appended on each side that had it.
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
            # Reverse i swaps reactants and products of forward i-1.
            # M asymmetry is handled per slot (parsing sets is_three_body[i] and
            # is_three_body[i-1] independently for dissociation reactions).
            f = i - 1
            r_idx = net.product_idx[f]
            p_idx = net.reactant_idx[f]
            r_st = net.product_stoich[f]
            p_st = net.reactant_stoich[f]
            has_M_reac = bool(net.is_three_body[i])
            has_M_prod = bool(net.is_three_body[f])

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

        reactants_wM = list(reactants)
        products_wM = list(products)
        if has_M_reac:
            reactants_wM.append("M")
        if has_M_prod:
            products_wM.append("M")
        re_wM_dict[i] = [reactants_wM, products_wM]

    return re_dict, re_wM_dict


re_dict, re_wM_dict = _build_re_dicts(_NETWORK)


def _pack_k_dict(k) -> np.ndarray:
    """Convert var.k (dict or ndarray) into a (nr+1, nz) NumPy array."""
    if isinstance(k, np.ndarray):
        return np.asarray(k, dtype=np.float64)
    if isinstance(k, dict):
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
    """Chemistry RHS at all layers. y (nz, ni), M (nz,), k dict-or-(nr+1, nz). Returns (nz, ni)."""
    y_np = np.asarray(y, dtype=np.float64)
    M_np = np.asarray(M, dtype=np.float64)
    k_arr = _pack_k_dict(k)
    out = _chem.chem_rhs(
        jnp.asarray(y_np), jnp.asarray(M_np), jnp.asarray(k_arr), _NET_JAX
    )
    return np.asarray(out, dtype=np.float64)


def symjac(y, M, k):
    """Not provided. VULCAN-JAX consumes the (nz, ni, ni) block-diagonal
    Jacobian directly via `chem.chem_jac`; the flat (nz*ni, nz*ni) form is
    only needed by master's scipy solve_banded path."""
    raise NotImplementedError(
        "chem_funs.symjac: use chem.chem_jac (block stack) instead."
    )


def neg_symjac(y, M, k):
    """Not provided; see `symjac`."""
    raise NotImplementedError(
        "chem_funs.neg_symjac: use chem.chem_jac (block stack) instead."
    )


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
    """Per-species g/(RT) at temperature(s) T."""
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


# Cache K_eq arrays by T identity. `Gibbs(i, T)` is typically called in a loop
# over reaction indices with the same T, so naive recomputation costs ~600x.
_K_EQ_CACHE: dict = {}


def _K_eq_array_cached(T_np: np.ndarray) -> np.ndarray:
    key = (T_np.shape, T_np.tobytes())
    cached = _K_EQ_CACHE.get(key)
    if cached is not None:
        return cached
    g_sp = _gibbs.gibbs_sp_vector(_NASA9_COEFFS, T_np)
    K_eq = _gibbs.K_eq_array(_NETWORK, g_sp, T_np)
    if len(_K_EQ_CACHE) >= 4:
        _K_EQ_CACHE.clear()
    _K_EQ_CACHE[key] = K_eq
    return K_eq


def Gibbs(i, T):
    """Equilibrium constant K_eq for forward reaction i at temperature(s) T.

    Returns float for scalar T, ndarray otherwise. The (k_B T / P0)^Δn
    factor (Δn = n_reac - n_prod) is folded into the result, so
    `k_reverse = k_forward / Gibbs(i, T)`.
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
    K = _K_eq_array_cached(T_arr)[i]
    if np.isscalar(T) or np.ndim(T) == 0:
        return float(K[0])
    return K


NETWORK = _NETWORK
