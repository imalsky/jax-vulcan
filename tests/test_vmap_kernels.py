"""Vmap-consistency checks for the hot-path JAX kernels.

For each kernel, build a small batch of independent inputs, run
`jax.vmap(kernel)(stacked_inputs)`, and verify every batch element
agrees with the corresponding single call to ~1e-12. Catches the
common JAX failure mode where a kernel inadvertently closes over a
non-vmappable variable or has a static shape that depends on a
batched dimension.

Covers the codegen chemistry RHS, the analytical Jacobian, the
diagonal-offdiag block solver and the photo optical-depth kernel.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from _helpers import relerr

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


VMAP_RTOL = 1e-12


@pytest.mark.parametrize(
    "kernel", ["chem_rhs_codegen", "chem_jac_analytical", "compute_tau_jax"]
)
def test_hd189_kernel_vmap_matches_single_calls(hd189_state, kernel) -> None:
    """`vmap(kernel)` over perturbed HD189 columns agrees with single calls."""
    import jax
    import jax.numpy as jnp
    import vulcan_jax.chem as chem_mod
    import vulcan_jax.chem_funs as chem_funs
    import vulcan_jax.network as net_mod
    import vulcan_jax.photo as photo_mod
    from vulcan_jax.config import default_config

    y = jnp.asarray(hd189_state.var.y, dtype=jnp.float64)
    M = jnp.asarray(hd189_state.atm.M, dtype=jnp.float64)
    k_arr = jnp.asarray(hd189_state.var.k_arr, dtype=jnp.float64)
    if kernel == "chem_rhs_codegen":
        fn, args, batch, seed = chem_funs.chem_rhs_codegen, (M, k_arr), 4, 0
    elif kernel == "chem_jac_analytical":
        net_jax = chem_mod.to_jax(net_mod.parse_network(default_config().network))
        fn, args, batch, seed = chem_mod.chem_jac_analytical, (M, k_arr, net_jax), 3, 1
    else:
        photo_static = hd189_state.solver._photo_static
        if photo_static is None:
            pytest.skip("use_photo=False: no tau kernel to validate")
        photo_data = photo_mod.photo_data_from_static(photo_static, chem_funs.spec_list)
        dz = jnp.asarray(hd189_state.atm.dz, dtype=jnp.float64)
        fn, args, batch, seed = photo_mod.compute_tau_jax, (dz, photo_data), 3, 3

    rng = np.random.default_rng(seed)
    y_batch = jnp.stack(
        [y * (1.0 + 1e-6 * rng.standard_normal(y.shape)) for _ in range(batch)],
        axis=0,
    )
    single = [fn(y_batch[b], *args) for b in range(batch)]
    batched = jax.vmap(fn, in_axes=(0,) + (None,) * len(args))(y_batch, *args)
    for b in range(batch):
        rel = relerr(batched[b], single[b])
        assert rel < VMAP_RTOL, f"{kernel} vmap drift at batch {b}: relerr={rel:.3e}"


def test_block_thomas_diag_offdiag_vmap_consistency() -> None:
    """`vmap(block_thomas_diag_offdiag)` agrees with single calls."""
    import jax
    import jax.numpy as jnp
    from _oracles import block_thomas_diag_offdiag

    rng = np.random.default_rng(2)
    nz, ni = 16, 8
    BATCH = 4

    def _make_system():
        # Diagonally-dominant random diag blocks; small offdiagonals.
        diag = rng.standard_normal((nz, ni, ni))
        # Boost the diagonal so the system is well-conditioned per layer.
        boost = (5.0 + np.abs(diag).sum(axis=2))[:, :, None] * np.eye(ni)[None]
        diag = diag + boost
        sup_d = 0.1 * rng.standard_normal((nz - 1, ni))
        sub_d = 0.1 * rng.standard_normal((nz - 1, ni))
        rhs = rng.standard_normal((nz, ni))
        return (
            jnp.asarray(diag, dtype=jnp.float64),
            jnp.asarray(sup_d, dtype=jnp.float64),
            jnp.asarray(sub_d, dtype=jnp.float64),
            jnp.asarray(rhs, dtype=jnp.float64),
        )

    systems = [_make_system() for _ in range(BATCH)]

    diag_b = jnp.stack([s[0] for s in systems], axis=0)
    sup_b = jnp.stack([s[1] for s in systems], axis=0)
    sub_b = jnp.stack([s[2] for s in systems], axis=0)
    rhs_b = jnp.stack([s[3] for s in systems], axis=0)

    single = [block_thomas_diag_offdiag(*systems[b]) for b in range(BATCH)]
    batched = jax.vmap(block_thomas_diag_offdiag, in_axes=(0, 0, 0, 0))(
        diag_b, sup_b, sub_b, rhs_b
    )

    for b in range(BATCH):
        rel = relerr(batched[b], single[b])
        assert rel < 1e-10, (
            f"block_thomas_diag_offdiag vmap drift at batch {b}: relerr={rel:.3e}"
        )
