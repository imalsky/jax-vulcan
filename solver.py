"""Block-tridiagonal Thomas solver and 2nd-order Rosenbrock step for VULCAN-JAX.

The implicit Rosenbrock step requires solving
    [c0 * I  -  dF/dy] * k = rhs
where dF/dy is the chemistry + diffusion Jacobian. Chemistry is block-
diagonal (per layer, ni x ni dense); diffusion couples adjacent layers
through diagonal-in-species blocks. The full LHS is therefore block-
tridiagonal with nz blocks of size ni x ni:

    [ A_0   B_0                            ] [k_0]   [rhs_0]
    [ C_1   A_1   B_1                       ] [k_1] = [rhs_1]
    [        C_2  A_2   B_2                  ] [...]   [...]
    [                ...                     ] [...]   [...]
    [                       C_{n-1} A_{n-1} ] [k_{n-1}] [rhs_{n-1}]

We solve via block-Thomas:
    Forward:  A'_j = A_j - C_j @ inv(A'_{j-1}) @ B_{j-1}
              rhs'_j = rhs_j - C_j @ inv(A'_{j-1}) @ rhs'_{j-1}
    Back:     k_{n-1} = inv(A'_{n-1}) @ rhs'_{n-1}
              k_j = inv(A'_j) @ (rhs'_j - B_j @ k_{j+1})

Cost: O(nz * ni^3). Differentiable, JIT-friendly, runs on CPU/GPU. We use
`jax.scipy.linalg.lu_factor` + `lu_solve` per block to avoid explicit inverse.

VULCAN uses scipy.linalg.solve_banded with bandwidth 2*ni-1; for nz=120, ni=93
that's ~9.5 GB of dense banded storage if expanded, but scipy's banded format
keeps it at (4*ni-1)*nz = ~371*120 = ~45K entries. Block-Thomas matches the
asymptotic flop count and avoids the host-device roundtrip when running on GPU.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class BlockThomasDiagFactors(NamedTuple):
    """LU factors for a diagonal-offdiag block-tridiagonal system.

    `diag_lu` / `diag_piv` are the layerwise LU factors of the forward-
    eliminated diagonal blocks `A'_j`. `sup_d` / `sub_d` are kept so a new
    RHS can be solved without rebuilding the factorisation.
    """
    diag_lu: jnp.ndarray
    diag_piv: jnp.ndarray
    sup_d: jnp.ndarray
    sub_d: jnp.ndarray


def factor_block_thomas_diag_offdiag(diag, sup_d, sub_d):
    """Factor a diagonal-offdiag block-tridiagonal system once.

    Args:
        diag:  shape (nz, ni, ni)
        sup_d: shape (nz-1, ni)
        sub_d: shape (nz-1, ni)

    Returns:
        BlockThomasDiagFactors for reuse across multiple RHS solves.
    """
    nz = diag.shape[0]
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
    """Block-tridiagonal Thomas solve with DIAGONAL super/sub blocks.

    Specialised for the VULCAN diffusion structure where chemistry contributes
    a dense `(ni, ni)` diagonal block per layer but diffusion is diagonal-in-
    species — the super and sub blocks are `diag(d)` for some `(ni,)` vector
    `d`. With diagonal `B = diag(b)` and diagonal `C = diag(c)` and dense
    `A_prev_inv`:

        (C @ A_prev_inv @ B)[i, k]  =  c[i] * b[k] * A_prev_inv[i, k]

    so the rank-update `A_new = A_j - C_j @ inv(A_prev) @ B_{j-1}` becomes
    `A_j - (c_j[:, None] * b_{j-1}[None, :]) * A_prev_inv`. This drops the
    `C @ invA_B` matmul (`O(ni^3)`) to an elementwise multiply (`O(ni^2)`),
    halving the per-layer flops in the forward elim. Both stages of Ros2
    benefit because they share the LHS factorisation.

    Args:
        diag:  shape (nz, ni, ni)  -- per-layer diagonal blocks (full dense)
        sup_d: shape (nz-1, ni)    -- diagonals of super-diagonal blocks
        sub_d: shape (nz-1, ni)    -- diagonals of sub-diagonal blocks
        rhs:   shape (nz, ni)      -- right-hand side per layer

    Returns:
        k:    shape (nz, ni)       -- solution
    """
    factors = factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)
    return solve_block_thomas_diag_offdiag(factors, rhs)


def block_thomas(diag, sup, sub, rhs):
    """Solve a block-tridiagonal system via Thomas's algorithm.

    Generic dense version; consumes full `(nz-1, ni, ni)` super/sub blocks.
    The hot path in `jax_step.jax_ros2_step` uses `block_thomas_diag_offdiag`
    instead, which exploits the diffusion-Jacobian's diagonal-in-species
    structure for ~2x fewer flops per forward-elim step. This dense variant
    stays for callers / tests with truly dense off-diagonal blocks.

    Args:
        diag: shape (nz, ni, ni)   -- per-layer diagonal blocks (full dense)
        sup:  shape (nz-1, ni, ni) -- super-diagonal blocks (j -> j+1)
        sub:  shape (nz-1, ni, ni) -- sub-diagonal blocks (j -> j-1; sub[k]
                                       multiplies row j=k+1 against col j=k)
        rhs:  shape (nz, ni)       -- right-hand side per layer

    Returns:
        k:   shape (nz, ni)        -- solution
    """
    nz = diag.shape[0]
    ni = diag.shape[1]

    # ---- Forward elimination ----
    # Carry: (A_curr_lu_factors, rhs_curr)
    # We store the LU factor of A'_{j-1} and the modified rhs'_{j-1}.
    # Use jax.scipy.linalg.lu_factor for stability.
    lu_factor = jax.scipy.linalg.lu_factor
    lu_solve = jax.scipy.linalg.lu_solve

    A0_lu = lu_factor(diag[0])              # (lu, piv)
    rhs0 = rhs[0]
    # We will scan over j = 1 .. nz-1.
    # Inputs at step j (1-based here): C_j = sub[j-1], B_{j-1} = sup[j-1], A_j = diag[j]

    def fwd_step(carry, inputs):
        A_prev_lu, rhs_prev = carry
        A_j, B_jm1, C_j, rhs_j = inputs
        # solve A'_{j-1} * x = B_{j-1}  ->  x = inv(A'_{j-1}) @ B_{j-1}
        invA_B = lu_solve(A_prev_lu, B_jm1)        # (ni, ni)
        # solve A'_{j-1} * y = rhs'_{j-1}  ->  y = inv(A'_{j-1}) @ rhs'_{j-1}
        invA_r = lu_solve(A_prev_lu, rhs_prev)     # (ni,)

        A_new = A_j - C_j @ invA_B
        rhs_new = rhs_j - C_j @ invA_r
        A_new_lu = lu_factor(A_new)
        return (A_new_lu, rhs_new), (A_new_lu, rhs_new)

    # sup has shape (nz-1, ni, ni); each forward-elim step j=1..nz-1 consumes
    # the (j-1)-th sup block, so all of sup is fed as the per-iter input.
    inputs = (diag[1:], sup, sub, rhs[1:])

    _, (A_lu_stack, rhs_mod_stack) = jax.lax.scan(fwd_step, (A0_lu, rhs0), inputs)

    # Stack the layer-0 entries on top so we have nz layers of (lu, rhs_mod).
    # A_lu_stack: leaves with leading axis nz-1.
    # We need a complete nz-stack for back-sub.
    # Use jax tree_map to prepend layer 0.
    A_lu_full = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b], axis=0),
                                  A0_lu, A_lu_stack)
    rhs_mod_full = jnp.concatenate([rhs0[None], rhs_mod_stack], axis=0)

    # ---- Back substitution ----
    k_last = lu_solve(jax.tree.map(lambda x: x[-1], A_lu_full), rhs_mod_full[-1])

    def bwd_step(carry, inputs):
        k_next = carry
        A_lu, rhs_mod, B = inputs
        rhs_local = rhs_mod - B @ k_next
        k_curr = lu_solve(A_lu, rhs_local)
        return k_curr, k_curr

    # Reversed scan from layer nz-2 down to 0.
    bwd_inputs = (
        jax.tree.map(lambda x: x[:-1][::-1], A_lu_full),
        rhs_mod_full[:-1][::-1],
        sup[::-1],
    )
    _, k_rev = jax.lax.scan(bwd_step, k_last, bwd_inputs)

    # k_rev has shape (nz-1, ni); the final solution is [k_0, k_1, ..., k_{nz-2}, k_last]
    # but k_rev is in reversed order: k_rev[0] = k_{nz-2}, k_rev[-1] = k_0.
    k_solution = jnp.concatenate([k_rev[::-1], k_last[None]], axis=0)
    return k_solution


def ros2_step(
    y,                  # (nz, ni)
    M,                  # (nz,)
    k_arr,              # (nr+1, nz)
    diff_diag,          # (nz, ni)   diff Jacobian DIAGONAL of self-block (per-species)
    diff_sup,           # (nz-1, ni)
    diff_sub,           # (nz-1, ni)
    chem_J,             # (nz, ni, ni) chemistry block-diagonal
    df,                 # (nz, ni) chemistry RHS + diffusion contribution at y
    df2,                # (nz, ni) chemistry RHS + diffusion contribution at y + k1/r
    dt,
    ni,
):
    """One step of the 2nd-order Rosenbrock solver (Verwer et al. 1997).

    Caller is responsible for evaluating chem_rhs/diff at y and y_k2 = y + k1/r.
    This function performs the two block-Thomas solves and combines the result.

    The LHS is constant within a step (depends on y, dt, k):
        LHS_diag[j] = (1/(r*dt)) * I  -  chem_J[j]  +  diag(diff_diag[j])
        LHS_sup[j]  =                                +  diag(diff_sup[j])
        LHS_sub[j]  =                                +  diag(diff_sub[j])

    Sign convention matches VULCAN: lhs_jac_tot subtracts the diffusion blocks
    (`dfdy[j_indx[j], j_indx[j]] -= ...`), so for the LHS we ADD diag(diff_*).

    Returns:
        sol: shape (nz, ni)            updated state
        delta_arr: shape (nz, ni)      |sol - y_k2|, used for adaptive dt
    """
    r = 1.0 + 1.0 / jnp.sqrt(2.0)
    c0 = 1.0 / (r * dt)
    nz = y.shape[0]

    # Build LHS blocks (matches VULCAN's lhs_jac_tot sign convention exactly):
    #   LHS = c0*I  -  chem_J  -  diff_J
    # diff_J is diagonal-in-species: diag block is diag(diff_diag[j]); sup/sub
    # blocks are diag(diff_sup[j]) and diag(diff_sub[j]).
    eye = jnp.eye(ni)
    diag_idx = jnp.arange(ni)
    diag = c0 * eye[None] - chem_J
    diag = diag.at[:, diag_idx, diag_idx].add(-diff_diag)

    sup = jnp.zeros((nz - 1, ni, ni)).at[:, diag_idx, diag_idx].set(-diff_sup)
    sub = jnp.zeros((nz - 1, ni, ni)).at[:, diag_idx, diag_idx].set(-diff_sub)

    # First Rosenbrock stage: solve LHS @ k1 = df
    k1 = block_thomas(diag, sup, sub, df)

    # Second stage RHS: rhs2 = df2 - (2/(r*dt)) * k1
    rhs2 = df2 - (2.0 / (r * dt)) * k1
    k2 = block_thomas(diag, sup, sub, rhs2)

    # Combine: sol = y + 3/(2r) * k1 + 1/(2r) * k2
    sol = y + (3.0 / (2.0 * r)) * k1 + (1.0 / (2.0 * r)) * k2

    # Truncation error proxy: |sol - y_k2| where y_k2 = y + k1/r
    y_k2 = y + k1 / r
    delta_arr = jnp.abs(sol - y_k2)
    return sol, delta_arr


def adaptive_dt(dt: float, delta: float, rtol: float, dt_min: float, dt_max: float,
                dt_var_min: float, dt_var_max: float) -> float:
    """Adapt the step size based on the truncation error estimate.

    Mirrors VULCAN's `step_size` logic at op.py:3094-3128.
        h_new = h * 0.9 * sqrt(rtol / delta)
        clipped to [dt_min*dt_var_min, dt_max*dt_var_max]
    """
    h_new = dt * 0.9 * (rtol / max(delta, 1e-300)) ** 0.5
    h_new = max(h_new, dt_min * dt_var_min)
    h_new = min(h_new, dt_max * dt_var_max)
    return h_new
