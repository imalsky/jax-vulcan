"""Benchmark VULCAN-JAX vs VULCAN-master.

Reports four timings, all separated so JIT-compile cost is never
mistaken for steady-state performance:

  1. Pre-loop setup (build the typed RunState from cfg).
  2. First outer-loop call: includes JIT compile + N integration steps.
  3. Cached outer-loop call: same N steps, kernel already compiled.
  4. Per-step JAX kernel (jax_ros2_step) hot timing.

When `../VULCAN-master/` is present, also runs the upstream Ros2 single
step in an isolated subprocess for direct comparison.

All timings labelled with backend (CPU/GPU), float dtype, and JAX +
JAXLIB versions so a stale cache or wrong device is obvious in the
output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT.parent / "VULCAN-master"
os.chdir(ROOT)
warnings.filterwarnings("ignore")

# Number of accepted steps in the outer-loop bench. Small enough to
# finish quickly; large enough that compile is a small fraction of the
# cached run on a typical laptop.
OUTER_BENCH_STEPS = 50

# Kernel hot timing: 3 warmups + 20 timed.
KERNEL_WARMUP = 3
KERNEL_ITERS = 20


def _backend_label() -> str:
    import jax
    import jaxlib

    devs = jax.devices()
    kinds = sorted({d.platform for d in devs})
    return (
        f"backend={'+'.join(kinds)} ({len(devs)} device{'s' if len(devs) != 1 else ''})"
        f" | jax={jax.__version__} jaxlib={jaxlib.__version__}"
        f" | x64={jax.config.read('jax_enable_x64')}"
    )


def _setup_runstate():
    """Run the full pre-loop pipeline; return (RunState, setup_seconds)."""
    from vulcan_jax.config import default_config
    from vulcan_jax.state import RunState

    # The old `import vulcan_jax.vulcan_cfg` module was deleted in the YAML-only
    # migration; `default_config()` is the process default it used to be.
    vulcan_cfg = default_config()

    t0 = time.time()
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    return rs, time.time() - t0


def _bench_master_single_step() -> float | None:
    """Run upstream Ros2 single-step in a subprocess. Return ms/step or None."""
    if not MASTER.is_dir():
        return None

    script = r"""
import os, time, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
import store, build_atm, op, vulcan_cfg
data_var = store.Variables()
data_atm = store.AtmData()
make_atm = build_atm.Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
if vulcan_cfg.use_condense:
    make_atm.sp_sat(data_atm)
rate = op.ReadRate()
data_var = rate.read_rate(data_var, data_atm)
if vulcan_cfg.use_lowT_limit_rates:
    data_var = rate.lim_lowT_rates(data_var, data_atm)
data_var = rate.rev_rate(data_var, data_atm)
data_var = rate.remove_rate(data_var)
ini = build_atm.InitialAbun()
data_var = ini.ini_y(data_var, data_atm)
data_var = ini.ele_sum(data_var)
data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)
data_var.dt = 1e-10
data_var.ymix = data_var.y / np.vstack(np.sum(data_var.y, axis=1))
solver = op.Ros2()
para = store.Parameters()
solver.naming_solver(para)
for _ in range(3):
    var_tmp = store.Variables()
    var_tmp.y = data_var.y.copy()
    var_tmp.ymix = data_var.ymix.copy()
    var_tmp.dt = data_var.dt
    var_tmp.k = data_var.k
    para_tmp = store.Parameters()
    para_tmp.solver_str = "solver"
    solver.solver(var_tmp, data_atm, para_tmp)
t0 = time.time()
for _ in range(20):
    var_tmp = store.Variables()
    var_tmp.y = data_var.y.copy()
    var_tmp.ymix = data_var.ymix.copy()
    var_tmp.dt = data_var.dt
    var_tmp.k = data_var.k
    para_tmp = store.Parameters()
    para_tmp.solver_str = "solver"
    solver.solver(var_tmp, data_atm, para_tmp)
print((time.time() - t0) / 20.0 * 1000.0)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=MASTER,
            check=True,
            capture_output=True,
            text=True,
        )
        return float(proc.stdout.strip().splitlines()[-1])
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"master_ros2_step_ms_per_step=unavailable  ({exc})")
        return None


def _bench_jax_kernel(rs) -> float:
    """Per-step jax_ros2_step ms (warm)."""
    import jax.numpy as jnp
    from vulcan_jax.jax_step import jax_ros2_step, make_atm_static
    import vulcan_jax.chem_funs as chem_funs
    from vulcan_jax.state import legacy_view

    var, atm, _ = legacy_view(rs)
    nz, ni = var.y.shape
    k_arr = np.asarray(var.k_arr, dtype=np.float64)
    if k_arr.shape != (chem_funs.nr + 1, nz):
        raise ValueError(f"k_arr shape {k_arr.shape} != ({chem_funs.nr + 1}, {nz})")

    atm_static = make_atm_static(atm, ni, nz)
    y_j = jnp.asarray(var.y)
    k_j = jnp.asarray(k_arr)
    dt = jnp.float64(1e-10)
    net = chem_funs._NET_JAX

    for _ in range(KERNEL_WARMUP):
        sol, _d = jax_ros2_step(y_j, k_j, dt, atm_static, net)
        sol.block_until_ready()
    t0 = time.time()
    for _ in range(KERNEL_ITERS):
        sol, _d = jax_ros2_step(y_j, k_j, dt, atm_static, net)
        sol.block_until_ready()
    return (time.time() - t0) / KERNEL_ITERS * 1000.0


def _bench_outer_loop(rs, n_steps: int) -> tuple[float, float, int, int]:
    """Run the outer loop for `n_steps` accepted steps, twice.

    Returns (first_call_total_s, cached_call_total_s, first_count,
    cached_count). The first call includes JIT compile; the second runs
    against the cached kernel. Convergence and runtime caps are
    deliberately set out of reach so the run terminates on count_max.
    """
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.config import default_config
    from vulcan_jax.state import RunState

    # See _setup_runstate: the vulcan_cfg module is gone; this is its successor.
    # Mutating it is still correct — default_config() returns the process-wide
    # Config object the setup facades read, and the saved/restore below undoes it.
    vulcan_cfg = default_config()

    saved = {
        k: getattr(vulcan_cfg, k)
        for k in ("use_live_plot", "use_live_flux", "count_max", "runtime")
    }
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False
    vulcan_cfg.count_max = n_steps
    vulcan_cfg.runtime = 1e22

    solver = op_jax.Ros2JAX()
    output = op.Output()
    integ = outer_loop.OuterLoop(solver, output)

    try:
        # First call: compile + run. `_ensure_runner` builds and JIT-
        # compiles the runner on this entry.
        t0 = time.time()
        rs1 = integ(rs)
        first_total = time.time() - t0
        first_count = int(rs1.params.count)

        # Second call from a fresh RunState clone, kernel cached. Do
        # NOT call integ.reset() between calls — that drops the JIT'd
        # runner closure and forces recompilation on the next call,
        # which is the opposite of what we want to measure.
        rs2_init = RunState.with_pre_loop_setup(vulcan_cfg)
        t0 = time.time()
        rs2 = integ(rs2_init)
        cached_total = time.time() - t0
        cached_count = int(rs2.params.count)
    finally:
        for k, v in saved.items():
            setattr(vulcan_cfg, k, v)
        integ.reset()

    return first_total, cached_total, first_count, cached_count


def main() -> None:
    # Pre-loop setup imports the JAX modules that flip x64 on, so do
    # the setup first; then print the backend label with x64 reflecting
    # the running state.
    rs, setup_s = _setup_runstate()
    print(_backend_label())
    print(f"steps_per_outer_call={OUTER_BENCH_STEPS}")
    print(f"setup_seconds={setup_s:.2f}")

    kernel_ms = _bench_jax_kernel(rs)
    print(f"jax_ros2_step_ms_per_step={kernel_ms:.3f}  (warm, kernel only)")

    first_s, cached_s, n1, n2 = _bench_outer_loop(rs, OUTER_BENCH_STEPS)
    print(
        f"outer_loop_first_call_seconds={first_s:.2f}  "
        f"({n1} steps, includes JIT compile)"
    )
    print(f"outer_loop_cached_call_seconds={cached_s:.2f}  ({n2} steps, kernel cached)")
    print(f"outer_loop_per_step_ms_cached={cached_s / max(n2, 1) * 1000.0:.3f}")

    master_ms = _bench_master_single_step()
    if master_ms is not None:
        print(f"master_ros2_step_ms_per_step={master_ms:.3f}  (subprocess oracle)")
        print(f"speedup_kernel_vs_master={master_ms / kernel_ms:.2f}x")
    else:
        print(
            "master_ros2_step_ms_per_step=unavailable  (no ../VULCAN-master/ sibling)"
        )


if __name__ == "__main__":
    main()
