"""Block-tridiagonal Thomas solver.

Solves `[A_j; B_{j-1}, C_j] @ k = rhs` for nz layers of size ni each. The
forward elimination computes `A'_j = A_j - C_j @ inv(A'_{j-1}) @ B_{j-1}`
via `lu_factor`/`lu_solve` per block (no explicit inverse). Cost is
O(nz * ni^3); differentiable, JIT-friendly, GPU-ready.

The hot path uses `block_thomas_diag_offdiag`, which exploits the
diffusion Jacobian's diagonal-in-species off-blocks for an O(ni^2)
rank update instead of an O(ni^3) matmul.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class BlockThomasDiagFactors(NamedTuple):
    """LU factors of the forward-eliminated diagonal blocks plus the
    diagonal off-diagonal vectors, so a new RHS can be solved without
    refactorising."""
    diag_lu: jnp.ndarray
    diag_piv: jnp.ndarray
    sup_d: jnp.ndarray
    sub_d: jnp.ndarray


def factor_block_thomas_diag_offdiag(diag, sup_d, sub_d):
    """Factor a diagonal-offdiag block-tridiagonal system once for reuse."""
    ni = diag.shape[1]

    lu_factor = jax.scipy.linalg.lu_factor
    lu_solve = jax.scipy.linalg.lu_solve
    eye_ni = jnp.eye(ni)

    A0_lu, A0_piv = lu_factor(diag[0])

    def fwd_step(carry, inputs):
        A_prev_lu, A_prev_piv = carry
        A_j, b_jm1, c_j = inputs
        A_prev_inv = lu_solve((A_prev_lu, A_prev_piv), eye_ni)
        A_new = A_j - (c_j[:, None] * b_jm1[None, :]) * A_prev_inv
        A_new_lu, A_new_piv = lu_factor(A_new)
        return (A_new_lu, A_new_piv), (A_new_lu, A_new_piv)

    _, (diag_lu_tail, diag_piv_tail) = jax.lax.scan(
        fwd_step,
        (A0_lu, A0_piv),
        (diag[1:], sup_d, sub_d),
    )

    diag_lu_full = jnp.concatenate([A0_lu[None], diag_lu_tail], axis=0)
    diag_piv_full = jnp.concatenate([A0_piv[None], diag_piv_tail], axis=0)
    return BlockThomasDiagFactors(
        diag_lu=diag_lu_full,
        diag_piv=diag_piv_full,
        sup_d=sup_d,
        sub_d=sub_d,
    )


def solve_block_thomas_diag_offdiag(factors: BlockThomasDiagFactors, rhs):
    """Solve a diagonal-offdiag block-tridiagonal system for a new RHS."""
    lu_solve = jax.scipy.linalg.lu_solve

    rhs0 = rhs[0]

    def fwd_rhs_step(rhs_prev, inputs):
        A_prev_lu, A_prev_piv, c_j, rhs_j = inputs
        invA_r = lu_solve((A_prev_lu, A_prev_piv), rhs_prev)
        rhs_new = rhs_j - c_j * invA_r
        return rhs_new, rhs_new

    _, rhs_mod_tail = jax.lax.scan(
        fwd_rhs_step,
        rhs0,
        (
            factors.diag_lu[:-1],
            factors.diag_piv[:-1],
            factors.sub_d,
            rhs[1:],
        ),
    )
    rhs_mod_full = jnp.concatenate([rhs0[None], rhs_mod_tail], axis=0)

    k_last = lu_solve(
        (factors.diag_lu[-1], factors.diag_piv[-1]),
        rhs_mod_full[-1],
    )

    def bwd_step(k_next, inputs):
        A_lu, A_piv, rhs_mod, b_j = inputs
        rhs_local = rhs_mod - b_j * k_next
        k_curr = lu_solve((A_lu, A_piv), rhs_local)
        return k_curr, k_curr

    _, k_rev = jax.lax.scan(
        bwd_step,
        k_last,
        (
            factors.diag_lu[:-1][::-1],
            factors.diag_piv[:-1][::-1],
            rhs_mod_full[:-1][::-1],
            factors.sup_d[::-1],
        ),
    )
    return jnp.concatenate([k_rev[::-1], k_last[None]], axis=0)


def block_thomas_diag_offdiag(diag, sup_d, sub_d, rhs):
    """Block-tridiagonal Thomas with diagonal super/sub blocks.

    With diagonal `B = diag(b)` and `C = diag(c)`, the rank update
    `A_j - C @ inv(A_prev) @ B` reduces to `A_j - (c[:,None]*b[None,:])
    * A_prev_inv`, dropping an O(ni^3) matmul to O(ni^2). Used by the
    Ros2 hot path; both Ros2 stages share the factorisation.

    Shapes: diag (nz, ni, ni), sup_d/sub_d (nz-1, ni), rhs (nz, ni) → (nz, ni).
    """
    factors = factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)
    return solve_block_thomas_diag_offdiag(factors, rhs)


def block_thomas(diag, sup, sub, rhs):
    """Generic dense block-tridiagonal Thomas solve.

    Use `block_thomas_diag_offdiag` instead when the off-diagonal blocks
    are diagonal-in-species (the hot path) — it is ~2× cheaper per layer.

    Shapes: diag (nz, ni, ni), sup/sub (nz-1, ni, ni), rhs (nz, ni) → (nz, ni).
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

    A_lu_full = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b], axis=0),
                                  A0_lu, A_lu_stack)
    rhs_mod_full = jnp.concatenate([rhs0[None], rhs_mod_stack], axis=0)

    k_last = lu_solve(jax.tree.map(lambda x: x[-1], A_lu_full), rhs_mod_full[-1])

    def bwd_step(carry, inputs):
        k_next = carry
        A_lu, rhs_mod, B = inputs
        rhs_local = rhs_mod - B @ k_next
        k_curr = lu_solve(A_lu, rhs_local)
        return k_curr, k_curr

    bwd_inputs = (
        jax.tree.map(lambda x: x[:-1][::-1], A_lu_full),
        rhs_mod_full[:-1][::-1],
        sup[::-1],
    )
    _, k_rev = jax.lax.scan(bwd_step, k_last, bwd_inputs)

    return jnp.concatenate([k_rev[::-1], k_last[None]], axis=0)
