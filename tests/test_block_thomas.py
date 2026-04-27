"""Validate block_thomas against a direct dense solve.

Random block-tridiagonal system, compare against np.linalg.solve.
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


def main() -> int:
    rng = np.random.default_rng(42)
    nz, ni = 8, 5
    # Build random block-tridiagonal blocks. Make diagonal blocks well-conditioned
    # by adding a large multiple of identity (matches our Rosenbrock LHS structure
    # where c0 ~ 1/(r*dt) dominates).
    diag_np = rng.standard_normal((nz, ni, ni)) + 10.0 * np.eye(ni)
    sup_np = rng.standard_normal((nz - 1, ni, ni))
    sub_np = rng.standard_normal((nz - 1, ni, ni))
    rhs_np = rng.standard_normal((nz, ni))

    # Assemble dense matrix for reference
    N = nz * ni
    M = np.zeros((N, N))
    for j in range(nz):
        M[j*ni:(j+1)*ni, j*ni:(j+1)*ni] = diag_np[j]
    for j in range(nz - 1):
        M[j*ni:(j+1)*ni, (j+1)*ni:(j+2)*ni] = sup_np[j]
        M[(j+1)*ni:(j+2)*ni, j*ni:(j+1)*ni] = sub_np[j]
    rhs_flat = rhs_np.reshape(-1)
    x_dense = np.linalg.solve(M, rhs_flat).reshape(nz, ni)

    # Solve via block_thomas
    x_jax = np.asarray(solver_mod.block_thomas(
        jnp.asarray(diag_np), jnp.asarray(sup_np), jnp.asarray(sub_np), jnp.asarray(rhs_np)
    ))

    relerr = np.max(np.abs(x_jax - x_dense) / np.maximum(np.abs(x_dense), 1e-12))
    print(f"block_thomas max relerr vs np.linalg.solve: {relerr:.3e}")

    # Test on a larger system that resembles our actual VULCAN-JAX use case
    nz2, ni2 = 120, 93
    diag_np = rng.standard_normal((nz2, ni2, ni2)) + 1e10 * np.eye(ni2)  # well-conditioned
    sup_np = rng.standard_normal((nz2 - 1, ni2, ni2)) * 1e-3
    sub_np = rng.standard_normal((nz2 - 1, ni2, ni2)) * 1e-3
    rhs_np = rng.standard_normal((nz2, ni2))

    N2 = nz2 * ni2
    M2 = np.zeros((N2, N2))
    for j in range(nz2):
        M2[j*ni2:(j+1)*ni2, j*ni2:(j+1)*ni2] = diag_np[j]
    for j in range(nz2 - 1):
        M2[j*ni2:(j+1)*ni2, (j+1)*ni2:(j+2)*ni2] = sup_np[j]
        M2[(j+1)*ni2:(j+2)*ni2, j*ni2:(j+1)*ni2] = sub_np[j]
    rhs_flat = rhs_np.reshape(-1)
    x_dense2 = np.linalg.solve(M2, rhs_flat).reshape(nz2, ni2)

    x_jax2 = np.asarray(solver_mod.block_thomas(
        jnp.asarray(diag_np), jnp.asarray(sup_np), jnp.asarray(sub_np), jnp.asarray(rhs_np)
    ))

    relerr2 = np.max(np.abs(x_jax2 - x_dense2) / np.maximum(np.abs(x_dense2), 1e-12))
    print(f"block_thomas (nz={nz2}, ni={ni2}) max relerr: {relerr2:.3e}")

    print()
    ok = relerr < 1e-9 and relerr2 < 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
