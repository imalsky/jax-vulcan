"""Demonstrate vmap-able batched Ros2 step.

Builds a single VULCAN-master state, replicates y across a batch dimension,
calls jax.vmap(jax_ros2_step) and verifies all batch elements agree.
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

warnings.filterwarnings("ignore")


def main() -> int:
    import jax
    import jax.numpy as jnp

    import vulcan_cfg
    import store
    from atm_setup import Atm
    from ini_abun import InitialAbun
    import legacy_io as op
    import chem_funs

    # Set up state
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    import rates as _rates_mod
    _rates_mod.setup_var_k(vulcan_cfg, data_var, data_atm)
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)
    data_var.dt = 1e-10

    y0 = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y0.shape

    import network as net_mod
    import chem as chem_mod
    import jax_step as js_mod

    net = net_mod.parse_network(vulcan_cfg.network)
    net_jax = chem_mod.to_jax(net)
    atm_static = js_mod.make_atm_static(data_atm, ni, nz)

    k_arr = np.asarray(data_var.k_arr, dtype=np.float64)

    # Single step (compile + warmup)
    print("Compiling JAX kernel ...")
    t0 = time.time()
    sol_single, delta_single = js_mod.jax_ros2_step(
        jnp.asarray(y0), jnp.asarray(k_arr), data_var.dt, atm_static, net_jax
    )
    sol_single.block_until_ready()
    print(f"  first call (compile + execute): {time.time() - t0:.2f}s")

    # Time second call (warm)
    t0 = time.time()
    for _ in range(10):
        sol_warm, delta_warm = js_mod.jax_ros2_step(
            jnp.asarray(y0), jnp.asarray(k_arr), data_var.dt, atm_static, net_jax
        )
        sol_warm.block_until_ready()
    print(f"  10 warm calls: {time.time() - t0:.2f}s ({(time.time()-t0)/10*1000:.1f}ms/step)")

    # Vmap over batch of 4 (replicas of the same y for sanity)
    BATCH = 4
    y_batch = jnp.stack([jnp.asarray(y0)] * BATCH, axis=0)
    k_batch = jnp.stack([jnp.asarray(k_arr)] * BATCH, axis=0)

    # vmap over y, k_arr; broadcast atm_static, net_jax
    vstep = jax.jit(jax.vmap(
        js_mod.jax_ros2_step,
        in_axes=(0, 0, None, None, None),
    ))

    print(f"\nCompiling vmap'd step (batch={BATCH}) ...")
    t0 = time.time()
    sol_batch, delta_batch = vstep(y_batch, k_batch, data_var.dt, atm_static, net_jax)
    sol_batch.block_until_ready()
    print(f"  first vmap call: {time.time() - t0:.2f}s")

    t0 = time.time()
    for _ in range(10):
        sol_batch, delta_batch = vstep(y_batch, k_batch, data_var.dt, atm_static, net_jax)
        sol_batch.block_until_ready()
    print(f"  10 warm vmap calls: {time.time() - t0:.2f}s ({(time.time()-t0)/10*1000:.1f}ms/step)")

    # Verify all batch elements equal the single-step result
    sol_batch_np = np.asarray(sol_batch)
    sol_single_np = np.asarray(sol_single)
    relerr = 0.0
    for b in range(BATCH):
        e = np.max(np.abs(sol_batch_np[b] - sol_single_np) / np.maximum(np.abs(sol_single_np), 1e-30))
        relerr = max(relerr, e)
    print(f"\nVmap consistency (batch element vs single): max relerr = {relerr:.3e}")

    print()
    ok = relerr < 1e-12
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
