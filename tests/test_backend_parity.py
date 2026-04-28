"""CPU/GPU backend parity for the JAX Ros2 step.

When a GPU backend is present, the same float64 `jax_ros2_step` call
should agree across CPU and GPU to tight tolerance. On CPU-only hosts
this test skips cleanly.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def _setup_hd189_state():
    from atm_setup import Atm
    from ini_abun import InitialAbun
    import chem
    import legacy_io as op
    import network
    import store
    import vulcan_cfg

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
    net = chem.to_jax(network.parse_network(vulcan_cfg.network))
    return data_var, data_atm, net


def test_jax_ros2_step_cpu_gpu_parity() -> None:
    import jax
    import jax.numpy as jnp

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if not gpu_devices:
        pytest.skip("No GPU backend available for CPU/GPU parity check.")

    from jax_step import jax_ros2_step, make_atm_static

    cpu_device = jax.devices("cpu")[0]
    gpu_device = gpu_devices[0]

    data_var, data_atm, net = _setup_hd189_state()
    nz, ni = data_var.y.shape
    atm_static = make_atm_static(data_atm, ni, nz)
    k_arr = np.asarray(data_var.k_arr, dtype=np.float64)

    y_cpu = jax.device_put(jnp.asarray(data_var.y), cpu_device)  # shape: (nz, ni)
    k_cpu = jax.device_put(jnp.asarray(k_arr), cpu_device)  # shape: (nr + 1, nz)
    atm_cpu = jax.device_put(atm_static, cpu_device)

    y_gpu = jax.device_put(jnp.asarray(data_var.y), gpu_device)  # shape: (nz, ni)
    k_gpu = jax.device_put(jnp.asarray(k_arr), gpu_device)  # shape: (nr + 1, nz)
    atm_gpu = jax.device_put(atm_static, gpu_device)
    dt = jnp.float64(data_var.dt)

    step_cpu = jax.jit(jax_ros2_step, device=cpu_device)
    step_gpu = jax.jit(jax_ros2_step, device=gpu_device)

    sol_cpu, delta_cpu = step_cpu(y_cpu, k_cpu, dt, atm_cpu, net)
    sol_gpu, delta_gpu = step_gpu(y_gpu, k_gpu, dt, atm_gpu, net)
    sol_cpu.block_until_ready()
    sol_gpu.block_until_ready()
    delta_cpu.block_until_ready()
    delta_gpu.block_until_ready()

    sol_cpu_np = np.asarray(jax.device_get(sol_cpu))
    sol_gpu_np = np.asarray(jax.device_get(sol_gpu))
    delta_cpu_np = np.asarray(jax.device_get(delta_cpu))
    delta_gpu_np = np.asarray(jax.device_get(delta_gpu))

    sol_relerr = np.max(
        np.abs(sol_cpu_np - sol_gpu_np) / np.maximum(np.abs(sol_cpu_np), 1e-30)
    )
    delta_relerr = np.max(
        np.abs(delta_cpu_np - delta_gpu_np) / np.maximum(np.abs(delta_cpu_np), 1e-30)
    )

    assert sol_relerr < 1e-11
    assert delta_relerr < 1e-11
