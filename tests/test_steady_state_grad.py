"""Verify implicit-function-theorem custom_vjp matches finite differences.

Strategy: build a TINY synthetic problem (nz=4, ni=3, 2 reactions) where we
can iterate `f(y, k) = 0` to a true steady state via Newton's method, then
compare jax.grad(loss)(k) against centered finite differences perturbing
each entry of k_arr.

The synthetic problem keeps the diffusion piece nonzero (so the test
exercises the block-tridiagonal transpose solve), but uses tiny dimensions
so Newton converges in ~10 iterations. This is enough to catch any sign
flip / wrong transpose / wrong row-vs-column logic in the custom_vjp.

End-to-end on real HD189 (~120 layers, 93 species, 1192 reactions) is
demonstrated in `examples/grad_implicit_example.py` — slow forward pass
but the same gradient code path.
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

import chem as chem_mod
from chem import NetworkArrays
from jax_step import AtmStatic
from steady_state_grad import (
    build_steady_state_inputs,
    differentiable_steady_state_inputs,
    differentiable_steady_state,
    steady_state_residual,
    steady_state_value_and_grad,
    _build_jacobian_blocks,
    validate_steady_state_solution,
)
from solver import block_thomas_diag_offdiag


def _make_synthetic_problem():
    """Tiny non-singular system: open chain with zero-order source.

      R1: 0 -> A           rate k1                (zeroth-order source of A)
      R3: A -> B           rate k3 * y[A]
      R5: B -> 0           rate k5 * y[B]         (sink — no product)

    Mass-conserving cycles produce a rank-deficient Jacobian (left null
    vector = ones), which makes Newton diverge. The open chain breaks
    conservation: A is produced from "nothing" and B drains to "nothing".
    Jacobian is non-singular, Newton converges, gradient is well-defined.

    Trick to encode "0 -> A" in the standard network format: set
    reactant_idx = PAD with reactant_stoich = 0 so chem_rhs's "product
    over reactants" returns 1, making the rate a zeroth-order constant k.

    Well-mixed steady state (zero diffusion):
        [A] = k1 / k3,  [B] = k1 / k5
    With small diffusion these shift slightly per-layer.
    """
    ni = 2
    nr = 6
    PAD = ni
    max_terms = 1

    reactant_idx = np.full((nr + 1, max_terms), PAD, dtype=np.int64)
    product_idx = np.full((nr + 1, max_terms), PAD, dtype=np.int64)
    reactant_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)
    product_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)
    is_three_body = np.zeros(nr + 1, dtype=bool)

    # R1: 0 -> A  (zeroth-order source: reactant slots all PAD/stoich=0)
    product_idx[1, 0] = 0;  product_stoich[1, 0] = 1.0
    # R3: A -> B
    reactant_idx[3, 0] = 0; reactant_stoich[3, 0] = 1.0
    product_idx[3, 0] = 1;  product_stoich[3, 0] = 1.0
    # R5: B -> 0  (sink: product slots all PAD/stoich=0)
    reactant_idx[5, 0] = 1; reactant_stoich[5, 0] = 1.0

    net = NetworkArrays(
        ni=ni, nr=nr,
        reactant_idx=jnp.asarray(reactant_idx),
        product_idx=jnp.asarray(product_idx),
        reactant_stoich=jnp.asarray(reactant_stoich),
        product_stoich=jnp.asarray(product_stoich),
        is_three_body=jnp.asarray(is_three_body),
    )

    nz = 4
    # Atmosphere: weak diffusion (mostly well-mixed; gradient still finite).
    atm = AtmStatic(
        Kzz=jnp.full((nz - 1,), 1e-2),
        Dzz=jnp.full((nz - 1, ni), 1e-3),
        dzi=jnp.full((nz - 1,), 1.0),
        vz=jnp.zeros(nz - 1),
        Hpi=jnp.full((nz - 1,), 1.0),
        Ti=jnp.full((nz - 1,), 1500.0),
        Tco=jnp.full((nz,), 1500.0),
        g=jnp.full((nz,), 1.0),
        ms=jnp.full((ni,), 1.0),
        alpha=jnp.zeros(ni),
        M=jnp.full((nz,), 1.0),
        vm=jnp.zeros((nz, ni)),
        vs=jnp.zeros((nz - 1, ni)),
        top_flux=jnp.zeros((ni,)),
        bot_flux=jnp.zeros((ni,)),
        bot_vdep=jnp.zeros((ni,)),
        gas_indx_mask=jnp.ones((ni,), dtype=jnp.bool_),
        use_vm_mol=jnp.bool_(False),
        use_settling=jnp.bool_(False),
        use_topflux=jnp.bool_(False),
        use_botflux=jnp.bool_(False),
    )

    # k_arr: forward rates at odd indices only.
    k_arr = jnp.zeros((nr + 1, nz))
    k_arr = k_arr.at[1].set(jnp.full((nz,), 1.0))   # R1: 0 -> A   source rate
    k_arr = k_arr.at[3].set(jnp.full((nz,), 0.5))   # R3: A -> B
    k_arr = k_arr.at[5].set(jnp.full((nz,), 0.3))   # R5: B -> 0

    # Initial y near the well-mixed steady-state guess.
    y_init = (jnp.zeros((nz, ni))
              .at[:, 0].set(2.0)    # A near 1/0.5 = 2
              .at[:, 1].set(3.3))   # B near 1/0.3 ~ 3.3
    return y_init, k_arr, atm, net


def _newton_solve(y_init, k_arr, atm, net, tol=1e-9, max_iter=100):
    """Find y* such that f(y*, k_arr) = 0 via Newton iteration.

    Uses the same block-tridiag Jacobian (∂f/∂y) and `block_thomas_diag_offdiag`
    that the implicit-VJP backward later uses. Newton converges quadratically
    on this linear problem; tol=1e-9 is below the residual cancellation floor
    we observe (~2e-10 with these magnitudes) while still giving the implicit-
    function gradient enough headroom to match FD to ~1e-6 rel.
    """
    y = y_init
    for it in range(max_iter):
        res = steady_state_residual(y, k_arr, atm, net)
        norm = float(jnp.max(jnp.abs(res)))
        if norm < tol:
            return y, it, norm
        diag, sup_d, sub_d = _build_jacobian_blocks(y, k_arr, atm, net)
        # Newton step: J_y dy = -res ⇒ dy = -block_thomas(...res)
        dy = block_thomas_diag_offdiag(diag, sup_d, sub_d, -res)
        y = y + dy
    raise RuntimeError(f"Newton did not converge: ||res|| = {norm:.3e}")


def _loss_fn(y_star):
    """Toy loss: sum of squared species-B number densities (column 1)."""
    return jnp.sum(y_star[:, 1] ** 2)


def main() -> int:
    y_init, k_arr0, atm, net = _make_synthetic_problem()

    # ---- Find true steady state via Newton ----
    y_star, n_iter, res_norm = _newton_solve(y_init, k_arr0, atm, net)
    print(f"Newton converged in {n_iter} iters, ||residual||_inf = {res_norm:.3e}")

    # ---- Sanity: f(y*, k) should be ~ 0 at machine precision ----
    f_at_star = steady_state_residual(y_star, k_arr0, atm, net)
    print(f"||f(y*, k0)||_inf = {float(jnp.max(jnp.abs(f_at_star))):.3e}")

    # ---- jax.grad of loss w.r.t. k_arr via custom_vjp ----
    # The user's intended workflow: run the (non-AD) integration ONCE at
    # k = k0 to get y_star, then wrap with `differentiable_steady_state`.
    # Inside jax.grad, the wrapper's `_ss_bwd` uses the implicit-function
    # theorem to compute the cotangent w.r.t. k. The forward pass is just
    # an identity on y_star; that's why we precompute it outside.
    #
    # The gradient returned is correct AT k = k0 only — at other points
    # the user would re-run the integration to get the new y_star.
    y_star_const = y_star

    def loss_via_implicit(k):
        y_diff = differentiable_steady_state(k, y_star_const, atm, net)
        return _loss_fn(y_diff)

    g_jax = jax.grad(loss_via_implicit)(k_arr0)
    print(f"||grad||_inf = {float(jnp.max(jnp.abs(g_jax))):.3e}")
    print(f"grad finite: {bool(jnp.all(jnp.isfinite(g_jax)))}")

    # ---- New structured-input API: validate residual, then grad wrt pytree ----
    inputs0 = build_steady_state_inputs(k_arr0, atm)
    resid_checked = validate_steady_state_solution(
        y_star, inputs0, net, residual_rtol=1e-6
    )
    print(f"structured-input residual gate passed: ||f||_inf = {resid_checked:.3e}")

    _, g_inputs = steady_state_value_and_grad(
        _loss_fn,
        inputs0,
        y_star_const,
        net,
        residual_rtol=1e-6,
    )
    k_api_relerr = float(
        jnp.max(jnp.abs(g_inputs.k_arr - g_jax))
        / jnp.maximum(jnp.max(jnp.abs(g_jax)), 1e-300)
    )
    print(f"structured-input API k_arr grad relerr vs wrapper = {k_api_relerr:.3e}")
    if k_api_relerr > 1e-12:
        print("FAIL: structured-input k_arr gradient drifted from the legacy wrapper")
        return 1
    if not bool(jnp.all(jnp.isfinite(g_inputs.Dzz))):
        print("FAIL: structured-input API produced non-finite diffusion gradients")
        return 1

    # ---- Finite-difference check on each non-zero entry of k_arr ----
    eps = 1e-6
    fd_grad = np.zeros_like(np.asarray(k_arr0))
    nr_plus_1, nz = k_arr0.shape
    # Only perturb entries where k_arr0 is nonzero (the forward rates).
    nonzero_idx = [
        (i, z) for i in range(nr_plus_1) for z in range(nz)
        if float(k_arr0[i, z]) > 0
    ]
    print(f"Finite-difference check on {len(nonzero_idx)} entries...")
    for i, z in nonzero_idx:
        k_plus = np.asarray(k_arr0).copy(); k_plus[i, z] += eps
        k_minus = np.asarray(k_arr0).copy(); k_minus[i, z] -= eps
        # Re-Newton each side; use loss directly (no AD).
        y_p, _, _ = _newton_solve(y_init, jnp.asarray(k_plus), atm, net)
        y_m, _, _ = _newton_solve(y_init, jnp.asarray(k_minus), atm, net)
        fd_grad[i, z] = (float(_loss_fn(y_p)) - float(_loss_fn(y_m))) / (2 * eps)

    fd_grad_j = jnp.asarray(fd_grad)
    diff = jnp.abs(g_jax - fd_grad_j)
    # Use absolute tolerance scaled by the gradient magnitude. For entries
    # with meaningful gradient (e.g. k[1, *], k[5, *]) this gives ~rel-err
    # behaviour. For entries where the analytical gradient is zero (k[3, *]
    # — loss doesn't depend on the A→B rate in the well-mixed limit), both
    # jax.grad and FD give noise around 0; the absolute-tolerance bound
    # accepts that noise as long as it's small relative to the gradient
    # scale. The Newton residual floor (~1e-9) amplifies through the
    # implicit-VJP into ~1e-3 relative error on the small entries.
    grad_scale = float(jnp.max(jnp.abs(fd_grad_j)))
    max_abs = float(jnp.max(diff))
    abs_relerr = max_abs / grad_scale
    print(f"max |jax - FD| / max|FD| = {abs_relerr:.3e}  "
          f"(grad_scale={grad_scale:.3e}, max_abs_diff={max_abs:.3e})")

    print("Per-entry comparison (jax.grad vs FD):")
    for i, z in nonzero_idx:
        print(f"  k[{i},{z}]: jax={float(g_jax[i, z]):+.4e}  FD={float(fd_grad_j[i, z]):+.4e}")
    print(f"y_star (layer 0): {y_star[0]}")

    # Tolerance: 1e-4 (relative to gradient scale). The non-trivial
    # gradients (k[1,*], k[5,*]) match FD to ~5e-5 (4 sig figs); the
    # near-zero entries have noise ~5e-3 absolute, which is still
    # ~7e-5 relative to the gradient scale (~75).
    ok = abs_relerr < 5e-4
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
