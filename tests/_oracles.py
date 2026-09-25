"""Reference kernels only the tests use: independent oracles for production code.

- `chem_rhs_segment_sum` / `chem_jac`: the vectorised `y**stoich` +
  `segment_sum` RHS and its `jax.jacrev`, the oracle for the analytical
  Jacobian `chem.chem_jac_analytical` (jacrev is an order of magnitude slower,
  notes §1.3; the Jacobian has no cancellation amplifier, so the RHS form's
  term order does not matter here).
- `chem_rhs_numpy`: the RHS in master's term order, NumPy float64, the oracle
  for the codegen RHS.
- `block_thomas`: the dense-off-block Thomas solve on
  `jax.scipy.linalg.lu_factor` / `lu_solve`, independent of the production
  `lax.linalg.lu` path; `block_thomas_diag_offdiag`: the production
  factor/solve pair in one call.
- `stage_defects`: each Ros2 stage vector's element-identity defect and the
  bound the stage repair leaves it under.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vulcan_jax import jax_step
from vulcan_jax.config import REPAIR_ABS_FLOOR
from vulcan_jax.solver import (
    factor_block_thomas_diag_offdiag,
    solve_block_thomas_diag_offdiag,
)


def chem_rhs_per_layer_segment_sum(y, M, k, net):
    """Vectorised RHS for one layer via masked y**stoich + segment_sum.

    y [ni], k [nr+1], `net` a `chem.NetworkArrays`.
    """
    # Pad y so reactant_idx==ni is a no-op multiplier (the padding stoich
    # is also 0, so this is double-safe).
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    y_r = yp[net.reactant_idx]  # [nr+1, max_terms]
    # `where` guards against 0**0 producing a NaN gradient on padded slots.
    factor = jnp.where(net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0)
    prod_r = jnp.prod(factor, axis=1)  # [nr+1]
    rate = k * prod_r
    rate = jnp.where(net.is_three_body, rate * M, rate)

    flat_r_idx = net.reactant_idx.reshape(-1)
    flat_p_idx = net.product_idx.reshape(-1)
    flat_r_st = net.reactant_stoich.reshape(-1)
    flat_p_st = net.product_stoich.reshape(-1)
    rate_repeat = jnp.repeat(rate, net.reactant_idx.shape[1])

    # num_segments = ni+1: the last segment collects padding contributions
    # (reactant_idx==ni); we drop it on slice.
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


chem_rhs_segment_sum = jax.vmap(
    chem_rhs_per_layer_segment_sum,
    in_axes=(0, 0, 1, None),
)

# jacrev beats jacfwd here ("scatter at the end" pattern).
chem_jac_per_layer = jax.jacrev(chem_rhs_per_layer_segment_sum, argnums=0)
chem_jac = jax.vmap(
    chem_jac_per_layer,
    in_axes=(0, 0, 1, None),
)


def chem_rhs_numpy(y, M, k, net):
    """NumPy reference RHS, master-faithful term order. `net` a `network.Network`.

    Mirrors `make_chem_funs.emit_chem_rhs_source`'s emission rules:
      - per-reaction rate uses stoich-replicated `*` (not `**stoich`),
        terminal `*M` when three-body
      - per-reaction `v_pair = v[i] - v[i+1]` (k[i+1]==0 zeros out the
        second term for unpaired photo/conden/radiative/ion reactions)
      - per-species accumulator walks i=1, 3, 5, ..., products-then-
        reactants per reaction, repeated by stoich (one `+= v_pair` per
        stoich count, not `+= st * v_pair`)
    The NumPy `*` is left-associative and float64 throughout; this gives
    bit-identical emission order to master's chemdf body.
    """
    nz, ni = y.shape
    PAD = ni
    nr = net.nr
    M_arr = np.asarray(M, dtype=np.float64)

    v = np.zeros((nr + 1, nz), dtype=np.float64)
    for i in range(1, nr + 1):
        rate = np.asarray(k[i], dtype=np.float64).copy()
        for kslot in range(net.reactant_idx.shape[1]):
            sp = int(net.reactant_idx[i, kslot])
            st = int(net.reactant_stoich[i, kslot])
            if st == 0 or sp == PAD:
                continue
            for _ in range(st):
                rate = rate * y[:, sp]
        if bool(net.is_three_body[i]):
            rate = rate * M_arr
        v[i] = rate

    dydt = np.zeros_like(y)
    for i in range(1, nr + 1, 2):
        v_pair = v[i] - v[i + 1]
        for kslot in range(net.product_idx.shape[1]):
            sp = int(net.product_idx[i, kslot])
            st = int(net.product_stoich[i, kslot])
            if st == 0 or sp == PAD:
                continue
            for _ in range(st):
                dydt[:, sp] = dydt[:, sp] + v_pair
        for kslot in range(net.reactant_idx.shape[1]):
            sp = int(net.reactant_idx[i, kslot])
            st = int(net.reactant_stoich[i, kslot])
            if st == 0 or sp == PAD:
                continue
            for _ in range(st):
                dydt[:, sp] = dydt[:, sp] - v_pair
    return dydt


def block_thomas_diag_offdiag(diag, sup_d, sub_d, rhs):
    """The production factor/solve pair in one call.

    Shapes: diag (nz, ni, ni), sup_d/sub_d (nz-1, ni), rhs (nz, ni) -> (nz, ni).
    """
    factors = factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)
    return solve_block_thomas_diag_offdiag(factors, rhs)


def block_thomas(diag, sup, sub, rhs):
    """Generic dense block-tridiagonal Thomas solve.

    Shapes: diag (nz, ni, ni), sup/sub (nz-1, ni, ni), rhs (nz, ni) -> (nz, ni).
    """
    lu_factor = jax.scipy.linalg.lu_factor
    lu_solve = jax.scipy.linalg.lu_solve

    A0_lu = lu_factor(diag[0])
    rhs0 = rhs[0]

    def fwd_step(carry, inputs):
        A_prev_lu, rhs_prev = carry
        A_j, B_jm1, C_j, rhs_j = inputs
        invA_B = lu_solve(A_prev_lu, B_jm1)
        invA_r = lu_solve(A_prev_lu, rhs_prev)

        A_new = A_j - C_j @ invA_B
        rhs_new = rhs_j - C_j @ invA_r
        A_new_lu = lu_factor(A_new)
        return (A_new_lu, rhs_new), (A_new_lu, rhs_new)

    inputs = (diag[1:], sup, sub, rhs[1:])

    _, (A_lu_stack, rhs_mod_stack) = jax.lax.scan(fwd_step, (A0_lu, rhs0), inputs)

    A_lu_full = jax.tree.map(
        lambda a, b: jnp.concatenate([a[None], b], axis=0), A0_lu, A_lu_stack
    )
    rhs_mod_full = jnp.concatenate([rhs0[None], rhs_mod_stack], axis=0)

    k_last = lu_solve(jax.tree.map(lambda x: x[-1], A_lu_full), rhs_mod_full[-1])

    def bwd_step(carry, inputs):
        k_next = carry
        A_lu, rhs_mod, B = inputs
        rhs_local = rhs_mod - B @ k_next
        k_curr = lu_solve(A_lu, rhs_local)
        return k_curr, k_curr

    bwd_inputs = (
        jax.tree.map(lambda x: x[:-1], A_lu_full),
        rhs_mod_full[:-1],
        sup,
    )
    _, k_head = jax.lax.scan(bwd_step, k_last, bwd_inputs, reverse=True)

    return jnp.concatenate([k_head, k_last[None]], axis=0)


def stage_defects(y, k_arr, dt, atm, net):
    """Per-layer, per-atom defect of each Ros2 stage vector against its
    element identity, and the bound the repair leaves it under:
    `max(_DEFECT_FLOOR * scale, REPAIR_ABS_FLOOR * c0 * n_tot)` with `scale`
    the atom-weighted sum of the identity's terms (`c0 |k|`, `|T k|`
    contributions, `|b_tr|`). Returns (k1, k2, defect1, defect2, bound).

    Reads `jax_step`'s module state at trace time, so a test that
    monkeypatches `jax_step._REPAIR_MAX_CELL_FRAC` sees it here."""
    k1, k2, _, (c0, diag_d, sup_d, sub_d, b1, b2) = jax_step._ros2_stages(
        y, k_arr, dt, atm, net, None
    )
    n_tot = jnp.sum(y, axis=1, keepdims=True)

    def parts(k, b):
        return jax_step._stage_defect(k, b, c0, diag_d, sup_d, sub_d, with_scale=True)

    d1, s1 = parts(k1, b1)
    d2, s2 = parts(k2, b2)
    bound = jnp.maximum(
        jax_step._DEFECT_FLOOR * jnp.maximum(s1, s2), REPAIR_ABS_FLOOR * c0 * n_tot
    )
    return k1, k2, d1, d2, bound
