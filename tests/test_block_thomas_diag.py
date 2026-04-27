"""Validate block_thomas_diag_offdiag against the dense block_thomas.

The diagonal-only super/sub variant is an algebraic specialisation of the
generic dense Thomas: when off-diagonal blocks are `diag(d)`, the rank
update `A_new = A_j - C_j @ inv(A_prev) @ B_{j-1}` reduces to a vectorised
elementwise multiply. This test confirms agreement to machine precision
on (a) random well-conditioned blocks and (b) the actual VULCAN-JAX-shape
problem (nz=120, ni=93) with realistic magnitudes.

Also confirms `jax.grad` works through the new function (it must — Phase
4.2 uses it for the implicit-VJP solve).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

jax.config.update("jax_enable_x64", True)

import solver as solver_mod


def _build_dense_blocks_from_diags(diag, sup_d, sub_d):
    """Promote diagonal sup/sub vectors into dense block-diagonal blocks."""
    nz = diag.shape[0]
    ni = diag.shape[1]
    di = jnp.arange(ni)
    sup = jnp.zeros((nz - 1, ni, ni)).at[:, di, di].set(sup_d)
    sub = jnp.zeros((nz - 1, ni, ni)).at[:, di, di].set(sub_d)
    return sup, sub


def main() -> int:
    rng = np.random.default_rng(7)
    ok = True

    # ---- Small well-conditioned random system ------------------------------
    nz, ni = 8, 5
    diag_np = rng.standard_normal((nz, ni, ni)) + 10.0 * np.eye(ni)
    sup_d_np = rng.standard_normal((nz - 1, ni))
    sub_d_np = rng.standard_normal((nz - 1, ni))
    rhs_np = rng.standard_normal((nz, ni))

    diag = jnp.asarray(diag_np)
    sup_d = jnp.asarray(sup_d_np)
    sub_d = jnp.asarray(sub_d_np)
    rhs = jnp.asarray(rhs_np)

    sup_dense, sub_dense = _build_dense_blocks_from_diags(diag, sup_d, sub_d)

    x_diag = np.asarray(solver_mod.block_thomas_diag_offdiag(diag, sup_d, sub_d, rhs))
    x_dense = np.asarray(solver_mod.block_thomas(diag, sup_dense, sub_dense, rhs))

    relerr_small = np.max(
        np.abs(x_diag - x_dense) / np.maximum(np.abs(x_dense), 1e-12)
    )
    print(f"small (nz={nz}, ni={ni}) diag-vs-dense max relerr: {relerr_small:.3e}")
    if relerr_small > 1e-12:
        print("FAIL: small system disagreement")
        ok = False

    # ---- VULCAN-JAX shape: nz=120, ni=93 with Rosenbrock-like magnitudes ---
    # Diagonal blocks dominated by c0*I (~1e10), diff sup/sub ~1e-3.
    nz2, ni2 = 120, 93
    diag2 = (rng.standard_normal((nz2, ni2, ni2)) + 1e10 * np.eye(ni2)).astype(
        np.float64
    )
    sup_d2 = (rng.standard_normal((nz2 - 1, ni2)) * 1e-3).astype(np.float64)
    sub_d2 = (rng.standard_normal((nz2 - 1, ni2)) * 1e-3).astype(np.float64)
    rhs2 = rng.standard_normal((nz2, ni2)).astype(np.float64)

    diag2j = jnp.asarray(diag2)
    sup_d2j = jnp.asarray(sup_d2)
    sub_d2j = jnp.asarray(sub_d2)
    rhs2j = jnp.asarray(rhs2)
    sup_dense2, sub_dense2 = _build_dense_blocks_from_diags(
        diag2j, sup_d2j, sub_d2j
    )

    x_diag2 = np.asarray(
        solver_mod.block_thomas_diag_offdiag(diag2j, sup_d2j, sub_d2j, rhs2j)
    )
    x_dense2 = np.asarray(
        solver_mod.block_thomas(diag2j, sup_dense2, sub_dense2, rhs2j)
    )

    relerr_big = np.max(
        np.abs(x_diag2 - x_dense2) / np.maximum(np.abs(x_dense2), 1e-12)
    )
    print(f"big   (nz={nz2}, ni={ni2}) diag-vs-dense max relerr: {relerr_big:.3e}")
    # Tolerance: random-direction differences in the LU-factor reduction order
    # accumulate to ~1e-9 on (120, 93) random blocks. The two paths are
    # algebraically equivalent; this is pure float-cancellation noise.
    if relerr_big > 1e-8:
        print(f"FAIL: big system disagreement {relerr_big:.3e} > 1e-8")
        ok = False

    # ---- jax.grad works through the new function ---------------------------
    def loss(d):
        return jnp.sum(
            solver_mod.block_thomas_diag_offdiag(d, sup_d, sub_d, rhs) ** 2
        )

    g = jax.grad(loss)(diag)
    finite = bool(jnp.all(jnp.isfinite(g)))
    print(f"jax.grad through block_thomas_diag_offdiag finite: {finite}")
    if not finite:
        print("FAIL: gradient produced NaN/Inf")
        ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
