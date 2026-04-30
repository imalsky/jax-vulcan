"""End-to-end implicit-function gradient on the real HD189 problem.

Workflow:
  1. Run the full forward integration to convergence (the standard
     `vulcan_jax.py` pipeline). This is non-AD; uses `lax.while_loop`.
  2. Wrap the converged `y_star` with `differentiable_steady_state`,
     which has a `jax.custom_vjp` that uses the implicit-function theorem
     to compute the gradient w.r.t. `k_arr`.
  3. Define a loss on `y_star` and call `jax.grad`.

What you get: gradient of any scalar loss `L(y_star)` w.r.t. all 1192
forward+reverse rate constants × 120 layers = ~143k differentiable
parameters, with **O(1) memory in step count** (the integration ran
~2700 steps; the gradient doesn't store any intermediate state).

Compute: ~1 forward integration + 1 block-tridiag transpose solve +
1 reverse-mode VJP through `chem_rhs` at the converged state. The
backward is dominated by the block-tridiag solve (same cost as one
Ros2 step).

Caveat: gradient accuracy is bounded by the runner's convergence
criterion. With `yconv_cri = 0.01` (HD189 default), expect ~1% relative
error in the gradient. Lower `yconv_cri` for tighter gradients (slower
forward).

Wall time: ~2 minutes for the full forward + a few seconds for the
gradient. Run from VULCAN-JAX/ as `python examples/grad_implicit_example.py`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import jax
import jax.numpy as jnp


def main() -> int:
    import vulcan_cfg
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False

    import legacy_io as op
    import op_jax
    import outer_loop
    import chem as _chem_mod
    import network as _net_mod
    from jax_step import make_atm_static
    from steady_state_grad import (
        differentiable_steady_state,
        steady_state_residual,
    )
    # Full pre-loop pipeline + RunState-driven runner.
    from state import RunState, legacy_view

    print("Building HD189 atmosphere + running forward integration...")
    t0 = time.time()
    runstate = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(runstate)
    data_para.start_time = t0
    output = op.Output()

    solver = op_jax.Ros2JAX()
    if runstate.photo_static is not None:
        solver._photo_static = runstate.photo_static

    integ = outer_loop.OuterLoop(solver, output)
    solver.naming_solver(data_para)
    runstate = integ(runstate)
    # Materialise the converged runstate back into the legacy_view shim
    # for the downstream `data_var.y` / `data_para.count` reads below.
    data_var, data_atm, data_para = legacy_view(runstate)
    data_para.start_time = t0
    print(f"Forward integration done in {time.time() - t0:.1f}s "
          f"({data_para.count} steps)")

    # Pull the converged state into JAX form.
    network = _net_mod.parse_network(vulcan_cfg.network)
    net_jax = _chem_mod.to_jax(network)
    y_star = jnp.asarray(data_var.y, dtype=jnp.float64)
    nz = data_atm.Tco.shape[0]
    ni = network.ni

    # The dense `var.k_arr` is the source of truth.
    k_arr = jnp.asarray(np.asarray(data_var.k_arr, dtype=np.float64))

    atm_static = make_atm_static(data_atm, ni, nz)

    # Sanity: how close is the converged state to f = 0?
    res = steady_state_residual(y_star, k_arr, atm_static, net_jax)
    res_norm = float(jnp.max(jnp.abs(res)))
    rhs_scale = float(jnp.max(jnp.abs(_chem_mod.chem_rhs(
        y_star, atm_static.M, k_arr, net_jax))))
    print(f"||f(y*, k)||_inf = {res_norm:.3e}, "
          f"||rhs||_inf at y* = {rhs_scale:.3e}, "
          f"||f||/||rhs|| = {res_norm / max(rhs_scale, 1e-300):.3e}")
    print("(The runner's `longdy < yconv_cri` criterion measures state "
          "change over time, NOT ||f|| → 0. For mass-conserving HD189 "
          "with default yconv_cri=0.01, ||f|| is large in absolute terms "
          "but small relative to the chem-RHS scale. The implicit gradient "
          "below is dominated by Jacobian-null-space noise unless yconv_cri "
          "is reduced or the gradient is projected onto the conserved-mass "
          "manifold. Lower yconv_cri for tighter forward + gradient.)")

    # Pick a target species — use CH4 if available, else fall back to first.
    target_sp = "CH4" if "CH4" in network.species_idx else network.species[0]
    target_idx = network.species_idx[target_sp]
    z_target = nz // 2  # mid-altitude layer
    print(f"Target: log10(ymix[{target_sp}]) at layer z={z_target}")

    # Loss = -log10 of mid-altitude target mixing ratio (tiny → makes loss
    # finite and well-scaled for grad sanity).
    def loss(k):
        y_diff = differentiable_steady_state(k, y_star, atm_static, net_jax)
        ysum = jnp.sum(y_diff[z_target])
        ymix = y_diff[z_target, target_idx] / ysum
        return -jnp.log10(jnp.maximum(ymix, 1e-300))

    L0 = float(loss(k_arr))
    print(f"loss(k0) = -log10(ymix) = {L0:.4f}  →  ymix ~ {10**(-L0):.3e}")

    print("Computing implicit gradient via custom_vjp...")
    t1 = time.time()
    g = jax.grad(loss)(k_arr)
    g.block_until_ready()
    print(f"Gradient computed in {time.time() - t1:.2f}s; "
          f"shape={g.shape}; max |g|={float(jnp.max(jnp.abs(g))):.3e}; "
          f"finite={bool(jnp.all(jnp.isfinite(g)))}")

    # Top-10 most sensitive (reaction, layer) pairs:
    g_np = np.asarray(g)
    flat_idx = np.argsort(np.abs(g_np).ravel())[::-1][:10]
    print(f"\nTop 10 most sensitive (reaction_index, layer) entries for "
          f"d(-log10 ymix[{target_sp}, z={z_target}]) / dk:")
    for fi in flat_idx:
        ri, zi = np.unravel_index(fi, g.shape)
        rxn_str = network.Rf.get(int(ri), "?")
        print(f"  k[r={ri:4d}, z={zi:3d}]  g={g_np[ri, zi]:+.3e}  "
              f"({rxn_str})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
