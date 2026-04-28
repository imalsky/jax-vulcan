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
    #
    # Phase 17 note: a Neumaier-compensated single-scan replacement for
    # `segment_sum` was investigated and reverted. It made the per-cell
    # sum machine-precise relative to its own input terms, but did NOT
    # narrow the ~1e-4 chem_rhs vs master gap, because that gap is set
    # by per-term floating-point roundoff differences in the rate
    # computation itself (e.g. JAX's `prod_r * 1.0 * 1.0` vs master's
    # explicit `y[A] * y[B]`), not by sum-order drift. Verified
    # experimentally: master's `chemdf` is bit-identical to
    # `math.fsum(master_terms)`, and our compensated sum is bit-identical
    # to `math.fsum(jax_terms)` — the two `*_terms` arrays differ by
    # ~1 ulp per multiply, which dominates after summing ~7K terms of
    # magnitude ~1e11 down to ~1e2. Closing that gap requires Stage 3
    # of the Phase 17 plan: emit each term in master's SymPy order (a
    # codegen step with a large perf hit), so for now the floor stays
    # documented in `CLAUDE.md` (Numerical hygiene) and the production
    # path uses the fast tree-reduction `segment_sum`.
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


# Per-layer Jacobian via reverse-mode AD. Production uses
# `chem_jac_analytical` below (Phase 11 — ~36x faster); this jacrev path
# is kept solely as the test oracle for the analytical version. jacrev
# beats jacfwd on this network's "scatter at the end" pattern.
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
    """Analytical chemistry Jacobian for one layer, built directly from
    stoichiometry. Returns dense [ni, ni] block.

    Analytical formula (mirrors what jacrev produces but skips the AD trace):

        J[i, j] = Σ_{rxns r}  Σ_{output slot s_i, reactant slot s_j}
                     sign_i * stoich_i * (stoich_j / y[reactant[r, s_j]]) * rate[r]

    where sign_i = +1 if `i` is a product in reaction r, -1 if reactant; and
    stoich_i / stoich_j are the corresponding entries of product_stoich /
    reactant_stoich. The (stoich_j / y_j) * rate factor is `∂rate[r]/∂y_j`.

    This sidesteps `jax.jacrev`'s materialisation of an `[ni, ni]` block via
    `ni` reverse-mode passes. For the SNCHO network (~1192 reactions) the
    analytical form is ~3-5× faster (Phase 11).
    """
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    # Per-slot reactant factors: y_l^stoich_l for actual reactants, 1 for padding.
    y_r = yp[net.reactant_idx]                                     # [nr+1, max_terms]
    factor_l = jnp.where(
        net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0
    )                                                              # [nr+1, max_terms]

    # Leave-one-out reactant product per slot j: Π_{l != j} factor_l[r, l].
    # Using a Python loop over the static max_terms axis (typically 3 for VULCAN).
    max_terms = net.reactant_idx.shape[1]
    slot_arange = jnp.arange(max_terms)
    leave_out_cols = []
    for j in range(max_terms):
        # Replace slot j's factor with 1 in the product → leave-one-out.
        f_excl = jnp.where(slot_arange == j, 1.0, factor_l)
        leave_out_cols.append(jnp.prod(f_excl, axis=1))
    leave_out = jnp.stack(leave_out_cols, axis=1)                  # [nr+1, max_terms]

    # ∂rate[r]/∂y[reactant_idx[r, s_j]] = stoich_j * y_j^(stoich_j - 1) * leave_out_j * k_r * M_factor
    # The y^(stoich-1) form sidesteps the (stoich/y)*rate trap when y_j == 0.
    pow_minus_one = jnp.where(
        net.reactant_stoich > 0,
        y_r ** (net.reactant_stoich - 1),
        0.0,
    )                                                              # [nr+1, max_terms]
    drate_dy = (
        net.reactant_stoich * pow_minus_one * leave_out * k[:, None]
    )                                                              # [nr+1, max_terms]
    drate_dy = jnp.where(
        net.is_three_body[:, None], drate_dy * M, drate_dy
    )

    # Scatter contributions into J[i, j].
    # For each reaction r, for each (output slot s_i, reactant slot s_j):
    #   J[out_idx[r, s_i], reactant_idx[r, s_j]] += sign * out_stoich * drate_dy[r, s_j]
    #
    # We unify the "output" axis by concatenating reactant rows (sign -1) and
    # product rows (sign +1) along the slot dimension. This gives a single
    # tensor of contributions with shape [nr+1, max_terms*2, max_terms].
    ni = net.ni
    PAD = ni                                                       # padding species index

    # Output (i) tables: stack reactants (sign -1) and products (sign +1).
    out_idx = jnp.concatenate(
        [net.reactant_idx, net.product_idx], axis=1
    )                                                              # [nr+1, 2*max_terms]
    out_stoich_signed = jnp.concatenate(
        [-net.reactant_stoich, net.product_stoich], axis=1
    )                                                              # [nr+1, 2*max_terms]

    # Contributions: [nr+1, 2*max_terms, max_terms]
    # contrib[r, s_i, s_j] = out_stoich_signed[r, s_i] * drate_dy[r, s_j]
    contrib = out_stoich_signed[:, :, None] * drate_dy[:, None, :]

    # Index pairs: row = out_idx[r, s_i], col = reactant_idx[r, s_j]
    # Both have shape [nr+1, 2*max_terms, max_terms] after broadcast.
    row = jnp.broadcast_to(out_idx[:, :, None], contrib.shape)
    col = jnp.broadcast_to(net.reactant_idx[:, None, :], contrib.shape)

    # Flatten and scatter via segment_sum into (ni+1)*(ni+1) cells. The PAD
    # row/col cells are discarded at the end.
    flat_contrib = contrib.reshape(-1)
    flat_keys = (row.reshape(-1) * (ni + 1) + col.reshape(-1))
    J_flat = jax.ops.segment_sum(
        flat_contrib,
        flat_keys,
        num_segments=(ni + 1) * (ni + 1),
        indices_are_sorted=False,
    )
    J = J_flat.reshape(ni + 1, ni + 1)[:ni, :ni]
    return J


# Vmapped over layers: J[nz, ni, ni].
chem_jac_analytical = jax.vmap(
    chem_jac_analytical_per_layer,
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
