"""Vectorised JAX chemistry RHS and Jacobian.

Convention:
  y[nz, ni]    number densities (cm^-3)
  M[nz]        total third-body density (cm^-3)
  k[nr+1, nz]  rate constants per reaction per layer (1-based)

Rate per reaction per layer:
    rate[i, z] = k[i, z] * Π_slot (y[reactant_idx[i,slot], z] ** stoich[i,slot])
    if is_three_body[i]: rate[i, z] *= M[z]

Per-species dy/dt is then a `segment_sum` of (production - loss) contributions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from network import Network


jax.config.update("jax_enable_x64", True)


class NetworkArrays:
    """Network stoichiometry packed for JAX.

    Registered as a custom pytree with `ni`/`nr` as static aux_data so
    jit/vmap don't retrace per network and `num_segments` stays concrete.
    """
    __slots__ = (
        "ni", "nr",
        "reactant_idx", "product_idx", "reactant_stoich", "product_stoich",
        "is_three_body",
    )

    def __init__(self, ni, nr, reactant_idx, product_idx,
                 reactant_stoich, product_stoich, is_three_body):
        self.ni = int(ni)
        self.nr = int(nr)
        self.reactant_idx = reactant_idx
        self.product_idx = product_idx
        self.reactant_stoich = reactant_stoich
        self.product_stoich = product_stoich
        self.is_three_body = is_three_body


def _network_arrays_flatten(net):
    children = (
        net.reactant_idx, net.product_idx,
        net.reactant_stoich, net.product_stoich,
        net.is_three_body,
    )
    aux = (net.ni, net.nr)
    return children, aux


def _network_arrays_unflatten(aux, children):
    ni, nr = aux
    r_idx, p_idx, r_st, p_st, is3 = children
    return NetworkArrays(ni, nr, r_idx, p_idx, r_st, p_st, is3)


jtu.register_pytree_node(NetworkArrays, _network_arrays_flatten, _network_arrays_unflatten)


def to_jax(net: Network) -> NetworkArrays:
    """Pack a Network's relevant arrays into jnp form for the chemistry RHS."""
    return NetworkArrays(
        ni=net.ni,
        nr=net.nr,
        reactant_idx=jnp.asarray(net.reactant_idx, dtype=jnp.int64),
        product_idx=jnp.asarray(net.product_idx, dtype=jnp.int64),
        reactant_stoich=jnp.asarray(net.reactant_stoich, dtype=jnp.float64),
        product_stoich=jnp.asarray(net.product_stoich, dtype=jnp.float64),
        is_three_body=jnp.asarray(net.is_three_body, dtype=jnp.bool_),
    )


def chem_rhs_per_layer(
    y: jnp.ndarray,           # [ni]
    M: float | jnp.ndarray,
    k: jnp.ndarray,           # [nr+1]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Chemistry contribution to dy/dt for one vertical layer. Returns [ni]."""
    # Pad y so reactant_idx==ni is a no-op multiplier (the padding stoich
    # is also 0, so this is double-safe).
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    y_r = yp[net.reactant_idx]                                     # [nr+1, max_terms]
    # `where` guards against 0**0 producing a NaN gradient on padded slots.
    factor = jnp.where(net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0)
    prod_r = jnp.prod(factor, axis=1)                              # [nr+1]
    rate = k * prod_r
    rate = jnp.where(net.is_three_body, rate * M, rate)

    flat_r_idx = net.reactant_idx.reshape(-1)
    flat_p_idx = net.product_idx.reshape(-1)
    flat_r_st = net.reactant_stoich.reshape(-1)
    flat_p_st = net.product_stoich.reshape(-1)
    rate_repeat = jnp.repeat(rate, net.reactant_idx.shape[1])

    # num_segments = ni+1: the last segment collects padding contributions
    # (reactant_idx==ni); we drop it on slice. There is a ~1e-4 absolute
    # gap vs master's `chemdf` for a handful of species (CH2_1, HC3N, HCCO
    # at certain layers). It is per-term ulp-roundoff in the rate product
    # itself — not segment_sum reduction-order drift — and only a
    # SymPy-faithful term order would close it. WONTFIX per CLAUDE.md.
    loss = jax.ops.segment_sum(
        flat_r_st * rate_repeat,
        flat_r_idx,
        num_segments=net.ni + 1,
        indices_are_sorted=False,
    )[: net.ni]
    prod = jax.ops.segment_sum(
        flat_p_st * rate_repeat,
        flat_p_idx,
        num_segments=net.ni + 1,
        indices_are_sorted=False,
    )[: net.ni]
    return prod - loss


chem_rhs = jax.vmap(
    chem_rhs_per_layer,
    in_axes=(0, 0, 1, None),
)


# jacrev beats jacfwd here ("scatter at the end" pattern). Kept as the test
# oracle for `chem_jac_analytical`; production uses the analytical form
# (~36× faster on the SNCHO network).
chem_jac_per_layer = jax.jacrev(chem_rhs_per_layer, argnums=0)
chem_jac = jax.vmap(
    chem_jac_per_layer,
    in_axes=(0, 0, 1, None),
)


def chem_jac_analytical_per_layer(
    y: jnp.ndarray,           # [ni]
    M: float | jnp.ndarray,
    k: jnp.ndarray,           # [nr+1]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Stoichiometry-driven chemistry Jacobian for one layer. Returns [ni, ni].

    Builds J[i, j] = Σ_r sign_i * stoich_i * (∂rate[r]/∂y_j) directly from
    the network tables, skipping `jacrev`'s ni reverse-mode passes.
    """
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    y_r = yp[net.reactant_idx]                                     # [nr+1, max_terms]
    factor_l = jnp.where(
        net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0
    )

    # Leave-one-out reactant product: Π_{l != j} factor_l[r, l]. Loop over
    # the static `max_terms` axis (typically 3 for VULCAN).
    max_terms = net.reactant_idx.shape[1]
    slot_arange = jnp.arange(max_terms)
    leave_out_cols = []
    for j in range(max_terms):
        f_excl = jnp.where(slot_arange == j, 1.0, factor_l)
        leave_out_cols.append(jnp.prod(f_excl, axis=1))
    leave_out = jnp.stack(leave_out_cols, axis=1)

    # The y^(stoich-1) form sidesteps the (stoich/y)*rate divide when y == 0.
    pow_minus_one = jnp.where(
        net.reactant_stoich > 0,
        y_r ** (net.reactant_stoich - 1),
        0.0,
    )
    drate_dy = (
        net.reactant_stoich * pow_minus_one * leave_out * k[:, None]
    )
    drate_dy = jnp.where(
        net.is_three_body[:, None], drate_dy * M, drate_dy
    )

    # Stack reactants (sign -1) and products (sign +1) into one "output" axis;
    # contrib[r, s_i, s_j] = out_stoich_signed[r, s_i] * drate_dy[r, s_j].
    ni = net.ni
    out_idx = jnp.concatenate(
        [net.reactant_idx, net.product_idx], axis=1
    )
    out_stoich_signed = jnp.concatenate(
        [-net.reactant_stoich, net.product_stoich], axis=1
    )
    contrib = out_stoich_signed[:, :, None] * drate_dy[:, None, :]

    row = jnp.broadcast_to(out_idx[:, :, None], contrib.shape)
    col = jnp.broadcast_to(net.reactant_idx[:, None, :], contrib.shape)

    # Scatter into a (ni+1)*(ni+1) grid, then strip the padding row/col.
    flat_contrib = contrib.reshape(-1)
    flat_keys = (row.reshape(-1) * (ni + 1) + col.reshape(-1))
    J_flat = jax.ops.segment_sum(
        flat_contrib,
        flat_keys,
        num_segments=(ni + 1) * (ni + 1),
        indices_are_sorted=False,
    )
    return J_flat.reshape(ni + 1, ni + 1)[:ni, :ni]


chem_jac_analytical = jax.vmap(
    chem_jac_analytical_per_layer,
    in_axes=(0, 0, 1, None),
)


def chem_rhs_numpy(y: np.ndarray, M: np.ndarray, k: np.ndarray, net: Network) -> np.ndarray:
    """NumPy reference RHS for tests. Slower; iterates reactions explicitly."""
    nz, ni = y.shape
    dydt = np.zeros_like(y)
    for i in range(1, net.nr + 1):
        rate = np.asarray(k[i], dtype=np.float64).copy()
        for kslot in range(net.reactant_idx.shape[1]):
            sp = net.reactant_idx[i, kslot]
            st = net.reactant_stoich[i, kslot]
            if st == 0:
                continue
            rate = rate * y[:, sp]**st
        if net.is_three_body[i]:
            rate = rate * M
        for kslot in range(net.product_idx.shape[1]):
            sp = net.product_idx[i, kslot]
            st = net.product_stoich[i, kslot]
            if st == 0 or sp >= ni:
                continue
            dydt[:, sp] += st * rate
        for kslot in range(net.reactant_idx.shape[1]):
            sp = net.reactant_idx[i, kslot]
            st = net.reactant_stoich[i, kslot]
            if st == 0 or sp >= ni:
                continue
            dydt[:, sp] -= st * rate
    return dydt
