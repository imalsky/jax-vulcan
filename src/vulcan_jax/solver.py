"""Block-tridiagonal Thomas solver.

Solves `[A_j; B_{j-1}, C_j] @ k = rhs` for nz layers of size ni each. The
forward elimination computes `A'_j = A_j - C_j @ inv(A'_{j-1}) @ B_{j-1}`
via an LU factorization and solve per block. Cost is O(nz * ni^3);
differentiable, JIT-friendly, GPU-ready.

Both routines below materialize `inv(A'_{j-1})` explicitly by solving against
`eye_ni`. The reference `block_thomas` needs it because its off-blocks are
dense. The hot-path `block_thomas_diag_offdiag` needs it too:
with diagonal off-blocks the update becomes the ELEMENTWISE product
`A_j - (c[:, None] * b[None, :]) * inv(A'_{j-1})`, which reads every one of
the ni^2 entries. What the diagonal structure buys is replacing an O(ni^3)
matmul with an O(ni^2) elementwise scaling; it does NOT avoid the inversion,
and forming the inverse keeps the sweep O(nz * ni^3) either way.

The hot path factors with `lax.linalg.lu` and keeps the row permutation in the
factors. `block_thomas` stays on `jax.scipy.linalg.lu_factor`/`lu_solve` so it
remains an independent oracle for the tests.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


def _lu_solve_perm(lu, perm, b):
    """lu_solve with the row permutation already materialized.

    `jax.scipy.linalg.lu_solve` (jax 0.6.2) rebuilds it from the pivots on
    every call; `lax.linalg.lu` returns it once per factorization and the
    factors keep it. Mirrors jax/_src/lax/linalg.py `_lu_solve_core`
    (trans=0). `b` may be (ni,) or (ni, n).
    """
    x = b[perm]
    x = jax.lax.linalg.triangular_solve(
        lu, x, left_side=True, lower=True, unit_diagonal=True
    )
    return jax.lax.linalg.triangular_solve(lu, x, left_side=True, lower=False)


class BlockThomasDiagFactors(NamedTuple):
    """LU factors of the forward-eliminated diagonal blocks, their row
    permutations, and the diagonal off-diagonal vectors, so a new RHS can be
    solved without refactorising."""

    diag_lu: jnp.ndarray
    diag_perm: jnp.ndarray
    sup_d: jnp.ndarray
    sub_d: jnp.ndarray


def factor_block_thomas_diag_offdiag(diag, sup_d, sub_d):
    """Factor a diagonal-offdiag block-tridiagonal system once for reuse.

    DO NOT replace the LU carry with an explicit inverse carry (rejected on
    accuracy). It is 2.0x faster under jvp, but on real ``I/(gamma*dt) - J``
    blocks (cond ~6.7e23 at dt_max=1e11) its residual is 569x worse; the
    per-layer re-factorization WITH PARTIAL PIVOTING is what stops error
    compounding across the sweep. Benchmark this solver on real blocks only
    (synthetic ones top out ~21 orders of conditioning short). If attacking
    the jvp cost (~85% of jvp linear algebra), write a custom JVP rule that
    keeps pivoted LU in the primal. Full record: notes.md §1.4.1.
    """
    ni = diag.shape[1]

    eye_ni = jnp.eye(ni, dtype=diag.dtype)

    A0_lu, _, A0_perm = jax.lax.linalg.lu(diag[0])

    def fwd_step(carry, inputs):
        A_prev_lu, A_prev_perm = carry
        A_j, b_jm1, c_j = inputs
        A_prev_inv = _lu_solve_perm(A_prev_lu, A_prev_perm, eye_ni)
        A_new = A_j - (c_j[:, None] * b_jm1[None, :]) * A_prev_inv
        A_new_lu, _, A_new_perm = jax.lax.linalg.lu(A_new)
        return (A_new_lu, A_new_perm), (A_new_lu, A_new_perm)

    _, (diag_lu_tail, diag_perm_tail) = jax.lax.scan(
        fwd_step,
        (A0_lu, A0_perm),
        (diag[1:], sup_d, sub_d),
    )

    diag_lu_full = jnp.concatenate([A0_lu[None], diag_lu_tail], axis=0)
    diag_perm_full = jnp.concatenate([A0_perm[None], diag_perm_tail], axis=0)
    return BlockThomasDiagFactors(
        diag_lu=diag_lu_full,
        diag_perm=diag_perm_full,
        sup_d=sup_d,
        sub_d=sub_d,
    )


def solve_block_thomas_diag_offdiag(factors: BlockThomasDiagFactors, rhs):
    """Solve a diagonal-offdiag block-tridiagonal system for a new RHS."""
    rhs0 = rhs[0]

    def fwd_rhs_step(rhs_prev, inputs):
        A_prev_lu, A_prev_perm, c_j, rhs_j = inputs
        invA_r = _lu_solve_perm(A_prev_lu, A_prev_perm, rhs_prev)
        rhs_new = rhs_j - c_j * invA_r
        return rhs_new, rhs_new

    _, rhs_mod_tail = jax.lax.scan(
        fwd_rhs_step,
        rhs0,
        (
            factors.diag_lu[:-1],
            factors.diag_perm[:-1],
            factors.sub_d,
            rhs[1:],
        ),
    )
    rhs_mod_full = jnp.concatenate([rhs0[None], rhs_mod_tail], axis=0)

    k_last = _lu_solve_perm(
        factors.diag_lu[-1],
        factors.diag_perm[-1],
        rhs_mod_full[-1],
    )

    # The back sweep is a reverse scan: the `[::-1]` idiom it replaces copied
    # the whole LU stack twice per solve.
    def bwd_step(k_next, inputs):
        A_lu, A_perm, rhs_mod, b_j = inputs
        rhs_local = rhs_mod - b_j * k_next
        k_curr = _lu_solve_perm(A_lu, A_perm, rhs_local)
        return k_curr, k_curr

    _, k_head = jax.lax.scan(
        bwd_step,
        k_last,
        (
            factors.diag_lu[:-1],
            factors.diag_perm[:-1],
            rhs_mod_full[:-1],
            factors.sup_d,
        ),
        reverse=True,
    )
    return jnp.concatenate([k_head, k_last[None]], axis=0)


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

    Use `block_thomas_diag_offdiag` instead when the off-diagonal blocks are
    diagonal-in-species (the hot path). Measured, the two effects split:
    1.10x from the rank update itself (the flop saving does not convert
    at ni~69, which is memory/latency-bound) and 2.10x from reusing one
    factorization across both Ros2 stages, which only the split
    factor/solve entry points above expose.

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

    A_lu_full = jax.tree.map(
        lambda a, b: jnp.concatenate([a[None], b], axis=0), A0_lu, A_lu_stack
    )
    rhs_mod_full = jnp.concatenate([rhs0[None], rhs_mod_stack], axis=0)

    k_last = lu_solve(jax.tree.map(lambda x: x[-1], A_lu_full), rhs_mod_full[-1])

    # Back sweep as a reverse scan, as in `solve_block_thomas_diag_offdiag`.
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
