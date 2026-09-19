"""JAX-native chem_funs: parses the network at import time and exposes the
same public surface (ni, nr, spec_list, re_dict, chemdf, ...)
as VULCAN-master's auto-generated chem_funs.py — backed by JAX kernels."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .config import default_config
from . import network as _network
from . import chem as _chem
from . import make_chem_funs as _make_chem_funs
from ._paths import resolve_data_path

jax.config.update("jax_enable_x64", True)

_CFG = default_config()


_NETWORK = _network.parse_network(str(resolve_data_path(_CFG.network)))
_NET_JAX = _chem.to_jax(_NETWORK)

# Build (or reload from cache) the SymPy-faithful per-reaction RHS.
# Memoised in `_make_chem_funs._BUILD_CACHE`, so the warmup call in
# `state._build_pre_loop_runstate` returns the same `Callable`.
_CHEM_RHS_CODEGEN = _make_chem_funs.build_chem_rhs(_NETWORK)

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
    """Chemistry RHS at all layers. y (nz, ni), M (nz,), k dict-or-(nr+1, nz). Returns (nz, ni).

    Codegen-backed: bit-faithful to VULCAN-master's `chemdf` (same per-
    reaction multiply chain order, same per-species accumulator order).
    """
    y_np = np.asarray(y, dtype=np.float64)
    M_np = np.asarray(M, dtype=np.float64)
    k_arr = _pack_k_dict(k)
    out = _CHEM_RHS_CODEGEN(jnp.asarray(y_np), jnp.asarray(M_np), jnp.asarray(k_arr))
    return np.asarray(out, dtype=np.float64)


# Re-exports for callers that want to bind directly to the production
# RHS (the integrator) or the segment_sum reference (tests/benchmarks).
chem_rhs_codegen = _CHEM_RHS_CODEGEN
chem_rhs_segment_sum = _chem.chem_rhs_segment_sum


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


NETWORK = _NETWORK
