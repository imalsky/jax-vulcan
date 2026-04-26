"""Pure-JAX chemistry right-hand side and Jacobian for VULCAN-JAX.

Replaces VULCAN's auto-generated `chem_funs.chemdf` and `chem_funs.symjac`
with vectorized JAX kernels. The Jacobian is taken via `jax.jacfwd` on a
per-layer rate function and vmapped across the vertical grid -- producing
an [nz, ni, ni] block-diagonal piece (chemistry only; diffusion adds the
off-diagonal blocks elsewhere).

Convention:
  y[nz, ni]    -- number densities (cm^-3)
  M[nz]        -- total third-body density (cm^-3) for 3-body reactions
  k[nr+1, nz]  -- rate constants per reaction per layer (1-based)
  net          -- parsed Network with stoichiometry tables

Rate per reaction per layer:
    rate[i, z] = k[i, z] * Π_slot (y[reactant_idx[i,slot], z] ** reactant_stoich[i,slot])
    if is_three_body[i]: rate[i, z] *= M[z]

Production / loss:
    dy[s, z] = Σ_i (Σ_slot product_stoich[i,slot] * (reactant_idx[i,slot]==s ? 0 : 0))   -- via segment_sum
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

    Registered as a custom pytree with `ni`, `nr` as static aux_data and the
    array fields as children. This means jit / vmap can trace the arrays
    without re-tracing per (ni, nr) value, while the ints stay concrete for
    use as `num_segments` in segment_sum.
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
    M: float | jnp.ndarray,   # scalar
    k: jnp.ndarray,           # [nr+1]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Chemistry contribution to dy/dt for a single vertical layer.

    Returns array of shape [ni].
    """
    # Pad y with one extra slot (index `ni` -> 1.0) so reactant_idx==ni is a
    # no-op multiplier (since reactant_stoich==0 there too, this is double-safe).
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    # Per-reaction rate: prod over slots of y_reactant**stoich
    y_r = yp[net.reactant_idx]                                     # [nr+1, max_terms]
    # Use safe power: 0**0 = 1, but we want padding (stoich=0) to yield 1
    # regardless of y. jnp.where guards against any 0**0 weirdness.
    factor = jnp.where(net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0)
    prod_r = jnp.prod(factor, axis=1)                              # [nr+1]
    rate = k * prod_r                                              # [nr+1]
    # Three-body factor
    rate = jnp.where(net.is_three_body, rate * M, rate)

    # Production: scatter +stoich * rate into product_idx
    # Loss: scatter -stoich * rate into reactant_idx
    # We use jax.ops.segment_sum to vectorize the scatter across all (rxn, slot) pairs.
    flat_r_idx = net.reactant_idx.reshape(-1)                      # [(nr+1) * max_terms]
    flat_p_idx = net.product_idx.reshape(-1)
    flat_r_st = net.reactant_stoich.reshape(-1)
    flat_p_st = net.product_stoich.reshape(-1)
    rate_repeat = jnp.repeat(rate, net.reactant_idx.shape[1])      # broadcast rate per slot

    # `num_segments=ni+1` -- the last segment (index ni) collects all padding
    # contributions; we discard it.
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


# Vmapped over the vertical grid. y: [nz, ni], k: [nr+1, nz], M: [nz]
# in_axes for k: 1 (per-layer slice over the second axis)
chem_rhs = jax.vmap(
    chem_rhs_per_layer,
    in_axes=(0, 0, 1, None),
)


# Per-layer Jacobian (ni x ni). Two options:
#   - jacfwd: forward-mode AD, builds Jacobian column-by-column (ni jvps)
#   - jacrev: reverse-mode AD, builds row-by-row (ni vjps)
# For ni ~ 80-100 (similar input/output), they're comparable. Empirically on
# the SNCHO network jacrev is ~25% faster because chem_rhs has a "scatter at
# the end" pattern that reverse-mode handles natively. Keep both available.
chem_jac_per_layer_fwd = jax.jacfwd(chem_rhs_per_layer, argnums=0)
chem_jac_per_layer_rev = jax.jacrev(chem_rhs_per_layer, argnums=0)
chem_jac_per_layer = chem_jac_per_layer_rev   # current best
chem_jac = jax.vmap(
    chem_jac_per_layer,
    in_axes=(0, 0, 1, None),
)


def chem_rhs_numpy(y: np.ndarray, M: np.ndarray, k: np.ndarray, net: Network) -> np.ndarray:
    """NumPy reference implementation (slower; for tests).

    Iterates reactions explicitly. Useful for cross-checking the JAX impl.
    Shapes: y[nz, ni], M[nz], k[nr+1, nz] -> dydt[nz, ni].
    """
    nz, ni = y.shape
    dydt = np.zeros_like(y)
    for i in range(1, net.nr + 1):
        # Compute rate[z] for this reaction
        rate = np.asarray(k[i], dtype=np.float64).copy()       # [nz]
        for kslot in range(net.reactant_idx.shape[1]):
            sp = net.reactant_idx[i, kslot]
            st = net.reactant_stoich[i, kslot]
            if st == 0:
                continue
            rate = rate * y[:, sp]**st
        if net.is_three_body[i]:
            rate = rate * M
        # Apply production / loss
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
