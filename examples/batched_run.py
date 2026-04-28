"""Example: run multiple atmospheres in parallel via jax.vmap.

Builds a single VULCAN-style state, replicates y across a batch dimension,
and steps all batch elements in parallel. Useful for parameter sweeps where
several atmospheres share the same network and atmosphere structure but
differ in initial composition or rate constants.

Run from VULCAN-JAX/:
    python examples/batched_run.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent / "VULCAN-master"))
warnings.filterwarnings("ignore")


def main():
    import jax
    import jax.numpy as jnp

    import vulcan_cfg
    import store
    from atm_setup import Atm
    from ini_abun import InitialAbun
    import op
    import op_jax
    import jax_step as js_mod

    # Set up the canonical state
    print("Setting up base atmosphere...")
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)
    nz, ni = data_var.y.shape

    atm_static = js_mod.make_atm_static(data_atm, ni, nz)
    k_arr = np.asarray(data_var.k_arr, dtype=np.float64)

    # Build a batch of N=8 atmospheres -- same structure but small
    # perturbations to the initial y to simulate a parameter sweep.
    BATCH = 8
    rng = np.random.default_rng(42)
    y_batch = np.tile(np.asarray(data_var.y)[None], (BATCH, 1, 1))
    # Add 1% noise to each batch element
    y_batch = y_batch * (1.0 + 0.01 * rng.standard_normal(y_batch.shape))
    k_batch = np.tile(k_arr[None], (BATCH, 1, 1))

    print(f"Batched setup: {BATCH} atmospheres, each (nz={nz}, ni={ni})")

    # vmap over leading batch axis of y, k_arr; broadcast atm_static, dt, net
    vstep = jax.jit(jax.vmap(
        js_mod.jax_ros2_step,
        in_axes=(0, 0, None, None, None),
    ))

    # Warmup
    print("Compiling vmap'd kernel ...")
    t0 = time.time()
    sol_batch, delta_batch = vstep(
        jnp.asarray(y_batch), jnp.asarray(k_batch),
        1e-10, atm_static, op_jax._NET_JAX,
    )
    sol_batch.block_until_ready()
    print(f"  first call (compile + execute): {time.time() - t0:.1f}s")

    # Time
    n_iter = 10
    t0 = time.time()
    for _ in range(n_iter):
        sol_batch, delta_batch = vstep(
            jnp.asarray(y_batch), jnp.asarray(k_batch),
            1e-10, atm_static, op_jax._NET_JAX,
        )
        sol_batch.block_until_ready()
    t_per_call = (time.time() - t0) / n_iter * 1000

    print(f"\nBatched timing:")
    print(f"  Per vmap call (batch of {BATCH}): {t_per_call:.1f} ms")
    print(f"  Per atmosphere:                  {t_per_call/BATCH:.1f} ms")
    print(f"  Single-atmosphere reference:     ~160 ms")
    print(f"  Speedup factor (batched):        {160 * BATCH / t_per_call:.2f}x")

    print(f"\nNote: on multi-CPU machines, set XLA_FLAGS to expose more devices:")
    print(f"  XLA_FLAGS=--xla_force_host_platform_device_count={BATCH} python {sys.argv[0]}")
    print(f"and switch from jax.vmap to jax.pmap to actually parallelize across")
    print(f"physical cores. On GPU, vmap parallelizes natively.")


if __name__ == "__main__":
    main()
