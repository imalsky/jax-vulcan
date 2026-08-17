"""Validate the block-tridiagonal solvers.

Chain of oracles: `block_thomas` (dense off-diagonals) against a direct
np.linalg.solve on the assembled dense matrix, then the production
`block_thomas_diag_offdiag` (diagonal-in-species off-diagonals, the O(ni^2)
rank update) against `block_thomas` on the same systems. Two sizes: a small
random system and the VULCAN-JAX shape (nz=120, ni=93) with Rosenbrock-like
magnitudes (diagonal dominated by c0*I ~ 1e10, sup/sub ~ 1e-3). Also checks
jax.grad through the production variant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import vulcan_jax.solver as solver_mod


def _dense_from_blocks(diag, sup, sub):
    nz, ni = diag.shape[0], diag.shape[1]
    M = np.zeros((nz * ni, nz * ni))
    for j in range(nz):
        M[j * ni:(j + 1) * ni, j * ni:(j + 1) * ni] = diag[j]
    for j in range(nz - 1):
        M[j * ni:(j + 1) * ni, (j + 1) * ni:(j + 2) * ni] = sup[j]
        M[(j + 1) * ni:(j + 2) * ni, j * ni:(j + 1) * ni] = sub[j]
    return M


def _promote_diags(diag, sup_d, sub_d):
    """Diagonal sup/sub vectors as dense block-diagonal blocks."""
    nz, ni = diag.shape[0], diag.shape[1]
    di = jnp.arange(ni)
    sup = jnp.zeros((nz - 1, ni, ni)).at[:, di, di].set(sup_d)
    sub = jnp.zeros((nz - 1, ni, ni)).at[:, di, di].set(sub_d)
    return sup, sub


# (nz, ni, diag boost, offdiag scale, dense-vs-numpy tol, diag-vs-dense tol).
# The looser large-system tolerances are float-cancellation noise in the LU
# reduction order; the paths are algebraically equivalent.
_CASES = {
    "small": (8, 5, 10.0, 1.0, 1e-9, 1e-12),
    "vulcan_shape": (120, 93, 1e10, 1e-3, 1e-6, 1e-8),
}


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_block_thomas_solvers_agree_with_dense_solve(case):
    nz, ni, boost, scale, tol_dense, tol_diag = _CASES[case]
    rng = np.random.default_rng(42)
    diag_np = rng.standard_normal((nz, ni, ni)) + boost * np.eye(ni)
    sup_d = rng.standard_normal((nz - 1, ni)) * scale
    sub_d = rng.standard_normal((nz - 1, ni)) * scale
    rhs_np = rng.standard_normal((nz, ni))

    diag, rhs = jnp.asarray(diag_np), jnp.asarray(rhs_np)
    sup, sub = _promote_diags(diag, jnp.asarray(sup_d), jnp.asarray(sub_d))

    x_ref = np.linalg.solve(_dense_from_blocks(diag_np, np.asarray(sup),
                                               np.asarray(sub)),
                            rhs_np.reshape(-1)).reshape(nz, ni)
    x_dense = np.asarray(solver_mod.block_thomas(diag, sup, sub, rhs))
    x_diag = np.asarray(solver_mod.block_thomas_diag_offdiag(
        diag, jnp.asarray(sup_d), jnp.asarray(sub_d), rhs))

    err_dense = np.max(np.abs(x_dense - x_ref)
                       / np.maximum(np.abs(x_ref), 1e-12))
    err_diag = np.max(np.abs(x_diag - x_dense)
                      / np.maximum(np.abs(x_dense), 1e-12))
    assert err_dense < tol_dense, f"dense vs numpy: {err_dense:.3e}"
    assert err_diag < tol_diag, f"diag vs dense: {err_diag:.3e}"


def test_grad_through_diag_offdiag_is_finite():
    rng = np.random.default_rng(7)
    nz, ni = 8, 5
    diag = jnp.asarray(rng.standard_normal((nz, ni, ni)) + 10.0 * np.eye(ni))
    sup_d = jnp.asarray(rng.standard_normal((nz - 1, ni)))
    sub_d = jnp.asarray(rng.standard_normal((nz - 1, ni)))
    rhs = jnp.asarray(rng.standard_normal((nz, ni)))

    def loss(d):
        return jnp.sum(solver_mod.block_thomas_diag_offdiag(
            d, sup_d, sub_d, rhs) ** 2)

    assert bool(jnp.all(jnp.isfinite(jax.grad(loss)(diag))))
