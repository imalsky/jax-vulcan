"""Vectorised JAX chemistry RHS and Jacobian.

Convention:
  y[nz, ni]    number densities (cm^-3)
  M[nz]        total third-body density (cm^-3)
  k[nr+1, nz]  rate constants per reaction per layer (1-based)

Rate per reaction per layer:
    rate[i, z] = k[i, z] * Π_slot (y[reactant_idx[i,slot], z] ** stoich[i,slot])
    if is_three_body[i]: rate[i, z] *= M[z]

The RHS is `make_chem_funs.build_chem_rhs(net)` (SymPy-faithful per-reaction
codegen, master-bit-faithful term order). This module packs the network
(`to_jax`) and builds the analytical Jacobian, which uses the `y_r ** stoich`
formulation: the leave-one-out derivative is naturally written that way and
the Jacobian has no cancellation amplifier (machine-precision agreement vs
master). The test oracles (`jax.jacrev` of a vectorised `segment_sum` RHS, a
NumPy RHS in master's term order) live in `tests/_oracles.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .network import Network


class NetworkArrays:
    """Network stoichiometry packed for JAX.

    Registered as a custom pytree with `ni`/`nr` as static aux_data so
    jit/vmap don't retrace per network and `num_segments` stays concrete.
    `jac_terms` / `jac_place` are the analytical Jacobian's static gather
    tables (`_jac_gather_tables`).
    """

    __slots__ = (
        "ni",
        "nr",
        "reactant_idx",
        "product_idx",
        "reactant_stoich",
        "product_stoich",
        "is_three_body",
        "jac_terms",
        "jac_place",
    )

    def __init__(
        self,
        ni,
        nr,
        reactant_idx,
        product_idx,
        reactant_stoich,
        product_stoich,
        is_three_body,
        jac_terms,
        jac_place,
    ):
        self.ni = int(ni)
        self.nr = int(nr)
        self.reactant_idx = reactant_idx
        self.product_idx = product_idx
        self.reactant_stoich = reactant_stoich
        self.product_stoich = product_stoich
        self.is_three_body = is_three_body
        self.jac_terms = jac_terms
        self.jac_place = jac_place


def _network_arrays_flatten(net):
    children = (
        net.reactant_idx,
        net.product_idx,
        net.reactant_stoich,
        net.product_stoich,
        net.is_three_body,
        net.jac_terms,
        net.jac_place,
    )
    aux = (net.ni, net.nr)
    return children, aux


def _network_arrays_unflatten(aux, children):
    ni, nr = aux
    return NetworkArrays(ni, nr, *children)


jtu.register_pytree_node(
    NetworkArrays, _network_arrays_flatten, _network_arrays_unflatten
)


def _jac_gather_tables(net: Network) -> tuple[tuple, jnp.ndarray]:
    """Static tables behind `chem_jac_analytical_per_layer`.

    Every nonzero J[i, j] is a sum over the (reaction r, reactant slot s)
    pairs with species j in slot s and species i among r's reactants or
    products, of `signed stoich_i * drate_dy[r, s]`. The pairs are listed
    per entry in (r, s) order and bucketed by power-of-two padded length
    (about 1.3x the real terms on the shipped networks), so the assembly is
    a gather, one fixed-order sum per entry (`_left_to_right_sum`) and a
    final gather into the dense block: no scatter-add, hence one summation
    order on every backend and in every program (a GPU scatter-add is atomic
    and reorders run to run) and no chunked transient.
    Returns `(terms, place)`: `terms` is a tuple of `(r, s, coef)` int32 /
    int32 / float64 tables of shape (entries, width) with zero-coefficient
    pads, and `place[i * ni + j]` indexes the concatenated per-entry sums,
    with the index one past the end meaning a structural zero.
    """
    r_idx = np.asarray(net.reactant_idx)
    r_st = np.asarray(net.reactant_stoich)
    out_idx = np.concatenate([r_idx, np.asarray(net.product_idx)], axis=1)
    out_st = np.concatenate([-r_st, np.asarray(net.product_stoich)], axis=1)
    ni = net.ni
    terms: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for r in range(r_idx.shape[0]):
        for sj in range(r_idx.shape[1]):
            if r_st[r, sj] <= 0 or r_idx[r, sj] >= ni:
                continue
            for si in range(out_idx.shape[1]):
                if out_st[r, si] == 0 or out_idx[r, si] >= ni:
                    continue
                key = (int(out_idx[r, si]), int(r_idx[r, sj]))
                terms.setdefault(key, []).append((r, sj, float(out_st[r, si])))
    keys = sorted(terms, key=lambda ij: (len(terms[ij]), ij))
    buckets = []
    start = 0
    while start < len(keys):
        width = 1 << (len(terms[keys[start]]) - 1).bit_length()
        end = start
        while end < len(keys) and len(terms[keys[end]]) <= width:
            end += 1
        r_tab = np.zeros((end - start, width), dtype=np.int32)
        s_tab = np.zeros_like(r_tab)
        c_tab = np.zeros(r_tab.shape, dtype=np.float64)
        for a, key in enumerate(keys[start:end]):
            for b, (r, sj, c) in enumerate(terms[key]):
                r_tab[a, b], s_tab[a, b], c_tab[a, b] = r, sj, c
        buckets.append(tuple(jnp.asarray(t) for t in (r_tab, s_tab, c_tab)))
        start = end
    place = np.full(ni * ni, len(keys), dtype=np.int32)
    for p, (i, j) in enumerate(keys):
        place[i * ni + j] = p
    return tuple(buckets), jnp.asarray(place)


def to_jax(net: Network) -> NetworkArrays:
    """Pack a Network's relevant arrays into jnp form for the chemistry RHS."""
    jac_terms, jac_place = _jac_gather_tables(net)
    return NetworkArrays(
        ni=net.ni,
        nr=net.nr,
        reactant_idx=jnp.asarray(net.reactant_idx, dtype=jnp.int64),
        product_idx=jnp.asarray(net.product_idx, dtype=jnp.int64),
        reactant_stoich=jnp.asarray(net.reactant_stoich, dtype=jnp.float64),
        product_stoich=jnp.asarray(net.product_stoich, dtype=jnp.float64),
        is_three_body=jnp.asarray(net.is_three_body, dtype=jnp.bool_),
        jac_terms=jac_terms,
        jac_place=jac_place,
    )


def chem_jac_analytical_per_layer(
    y: jnp.ndarray,  # [ni]
    M: float | jnp.ndarray,
    k: jnp.ndarray,  # [nr+1]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Stoichiometry-driven chemistry Jacobian for one layer. Returns [ni, ni].

    Builds J[i, j] = Σ_r sign_i * stoich_i * (∂rate[r]/∂y_j) directly from
    the network tables, skipping `jacrev`'s ni reverse-mode passes; the sum
    runs along the static gather tables of `_jac_gather_tables`.
    """
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    y_r = yp[net.reactant_idx]  # [nr+1, max_terms]
    factor_l = jnp.where(net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0)

    # Leave-one-out reactant product: Π_{l != j} factor_l[r, l]. Loop over
    # the static `max_terms` axis (typically 3 for VULCAN).
    max_terms = net.reactant_idx.shape[1]
    slot_arange = jnp.arange(max_terms)
    leave_out_cols = []
    for j in range(max_terms):
        f_excl = jnp.where(slot_arange == j, 1.0, factor_l)
        leave_out_cols.append(jnp.prod(f_excl, axis=1))
    leave_out = jnp.stack(leave_out_cols, axis=1)

    # AD-safe power rule: NEVER raise to the 0 power. `y_r ** 0` has primal 1.0
    # but a NaN jvp at y_r == 0 (0 * y^-1), and clipped cells make y_r == 0
    # routine mid-run -- end-to-end forward-mode AD depends on this. stoich==1
    # contributes a constant 1; stoich>=2 a real power with exponent >= 1.
    # Primal is bit-identical to y_r**(stoich-1); masks fold at trace time.
    safe_exp = jnp.where(net.reactant_stoich > 1, net.reactant_stoich - 1, 1)
    pow_minus_one = jnp.where(
        net.reactant_stoich >= 2,
        y_r**safe_exp,
        jnp.where(net.reactant_stoich == 1, 1.0, 0.0),
    )
    drate_dy = net.reactant_stoich * pow_minus_one * leave_out * k[:, None]
    drate_dy = jnp.where(net.is_three_body[:, None], drate_dy * M, drate_dy)

    # One sum per nonzero entry along the static tables, then the dense block
    # by a gather (the slot past the last entry is the structural zero).
    entry_sums = [_left_to_right_sum(coef * drate_dy[r, s]) for r, s, coef in net.jac_terms]
    zero = jnp.zeros((1,), dtype=drate_dy.dtype)
    ni = net.ni
    return jnp.concatenate([*entry_sums, zero])[net.jac_place].reshape(ni, ni)


def _left_to_right_sum(x: jnp.ndarray) -> jnp.ndarray:
    """Sum over the last axis in one fixed order. XLA may reorder a `reduce`
    per program, and the batch and queue runners then disagreed at the last
    bit, which the dt controller turned into a different accept count
    (vulcan-forward's `test_queue_with_enough_lanes_runs_the_batch_ticks`).
    The terms are exact (a stoichiometric coefficient times a gathered
    value), so an explicit chain of adds has one result on every program."""
    acc = x[..., 0]
    for j in range(1, x.shape[-1]):
        acc = acc + x[..., j]
    return acc


chem_jac_analytical = jax.vmap(
    chem_jac_analytical_per_layer,
    in_axes=(0, 0, 1, None),
)
