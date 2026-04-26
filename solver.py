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

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def block_thomas(diag, sup, sub, rhs):
    """Solve a block-tridiagonal system via Thomas's algorithm.

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

    inputs = (diag[1:], sup[:-1] if sup.shape[0] >= nz - 1 else sup, sub, rhs[1:])
    # sup has shape (nz-1, ni, ni); we feed sup[0..nz-2] which is all of it.
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
