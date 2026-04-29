"""Vmap-consistency checks for the hot-path JAX kernels.

For each kernel, build a small batch of independent inputs, run
`jax.vmap(kernel)(stacked_inputs)`, and verify every batch element
agrees with the corresponding single call to ~1e-12. Catches the
common JAX failure mode where a kernel inadvertently closes over a
non-vmappable variable or has a static shape that depends on a
batched dimension.

Mirrors `test_vmap_step.py`'s pattern but covers the per-layer
chemistry, the analytical Jacobian, the diagonal-offdiag block
solver, and the photo optical-depth kernel.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def _build_state():
    """Build the canonical HD189 state once for kernel testing."""
    import vulcan_cfg
    from state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    return rs, data_var, data_atm


def _max_relerr(a: np.ndarray, b: np.ndarray, floor: float = 1e-30) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def test_chem_rhs_vmap_consistency() -> None:
    """`vmap(chem_rhs)` over a batch of states agrees with single calls."""
    import jax
    import jax.numpy as jnp
    import vulcan_cfg
    import network as net_mod
    import chem as chem_mod

    _, data_var, data_atm = _build_state()
    net = net_mod.parse_network(vulcan_cfg.network)
    net_jax = chem_mod.to_jax(net)

    y = jnp.asarray(data_var.y, dtype=jnp.float64)
    M = jnp.asarray(data_atm.M, dtype=jnp.float64)
    k_arr = jnp.asarray(data_var.k_arr, dtype=jnp.float64)

    BATCH = 4
    rng = np.random.default_rng(0)
    y_batch = jnp.stack(
        [y * (1.0 + 1e-6 * rng.standard_normal(y.shape)) for _ in range(BATCH)],
        axis=0,
    )

    single = [chem_mod.chem_rhs(y_batch[b], M, k_arr, net_jax) for b in range(BATCH)]
    batched = jax.vmap(chem_mod.chem_rhs, in_axes=(0, None, None, None))(
        y_batch, M, k_arr, net_jax
    )

    for b in range(BATCH):
        rel = _max_relerr(batched[b], single[b])
        assert rel < 1e-12, f"chem_rhs vmap drift at batch {b}: relerr={rel:.3e}"


def test_chem_jac_analytical_vmap_consistency() -> None:
    """`vmap(chem_jac_analytical)` agrees with single calls."""
    import jax
    import jax.numpy as jnp
    import vulcan_cfg
    import network as net_mod
    import chem as chem_mod

    _, data_var, data_atm = _build_state()
    net = net_mod.parse_network(vulcan_cfg.network)
    net_jax = chem_mod.to_jax(net)

    y = jnp.asarray(data_var.y, dtype=jnp.float64)
    M = jnp.asarray(data_atm.M, dtype=jnp.float64)
    k_arr = jnp.asarray(data_var.k_arr, dtype=jnp.float64)

    BATCH = 3
    rng = np.random.default_rng(1)
    y_batch = jnp.stack(
        [y * (1.0 + 1e-6 * rng.standard_normal(y.shape)) for _ in range(BATCH)],
        axis=0,
    )

    single = [
        chem_mod.chem_jac_analytical(y_batch[b], M, k_arr, net_jax)
        for b in range(BATCH)
    ]
    batched = jax.vmap(
        chem_mod.chem_jac_analytical, in_axes=(0, None, None, None)
    )(y_batch, M, k_arr, net_jax)

    for b in range(BATCH):
        rel = _max_relerr(batched[b], single[b])
        assert rel < 1e-12, (
            f"chem_jac_analytical vmap drift at batch {b}: relerr={rel:.3e}"
        )


def test_block_thomas_diag_offdiag_vmap_consistency() -> None:
    """`vmap(block_thomas_diag_offdiag)` agrees with single calls."""
    import jax
    import jax.numpy as jnp
    import solver as solver_mod

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

    single = [
        solver_mod.block_thomas_diag_offdiag(*systems[b]) for b in range(BATCH)
    ]
    batched = jax.vmap(solver_mod.block_thomas_diag_offdiag, in_axes=(0, 0, 0, 0))(
        diag_b, sup_b, sub_b, rhs_b
    )

    for b in range(BATCH):
        rel = _max_relerr(batched[b], single[b])
        assert rel < 1e-10, (
            f"block_thomas_diag_offdiag vmap drift at batch {b}: relerr={rel:.3e}"
        )


def test_compute_tau_jax_vmap_consistency() -> None:
    """`vmap(compute_tau_jax)` over batched y agrees with single calls.

    Skips when use_photo=False. Builds a PhotoData pytree once and
    feeds different y batches through it.
    """
    import jax
    import jax.numpy as jnp
    import vulcan_cfg

    if not vulcan_cfg.use_photo:
        return  # Nothing to validate when photo is off.

    rs, _, data_atm = _build_state()
    if rs.photo_static is None:
        return

    import photo as photo_mod
    import chem_funs

    photo_data = photo_mod.photo_data_from_static(
        rs.photo_static, chem_funs.spec_list
    )
    y = jnp.asarray(rs.step.y, dtype=jnp.float64)
    dz = jnp.asarray(data_atm.dz, dtype=jnp.float64)

    BATCH = 3
    rng = np.random.default_rng(3)
    y_batch = jnp.stack(
        [y * (1.0 + 1e-6 * rng.standard_normal(y.shape)) for _ in range(BATCH)],
        axis=0,
    )

    single = [photo_mod.compute_tau_jax(y_batch[b], dz, photo_data) for b in range(BATCH)]
    batched = jax.vmap(photo_mod.compute_tau_jax, in_axes=(0, None, None))(
        y_batch, dz, photo_data
    )

    for b in range(BATCH):
        rel = _max_relerr(batched[b], single[b])
        assert rel < 1e-12, (
            f"compute_tau_jax vmap drift at batch {b}: relerr={rel:.3e}"
        )
