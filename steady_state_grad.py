"""Implicit-function-theorem gradients of the converged photochemical state.

`outer_loop.OuterLoop`'s `lax.while_loop` supports `jvp`/`jacfwd` but not
`vjp`/`grad`. The converged state `y*` satisfies `f(y*, theta) = 0`, so

    ∂y*/∂theta = -(∂f/∂y)^{-1} (∂f/∂theta)

and the cotangent VJP becomes

    ∂L/∂theta = -((∂f/∂y)^{-T} v) · (∂f/∂theta)

This module exposes a `differentiable_steady_state` that wraps the
integrator with a `jax.custom_vjp`: the backward pass does one
transposed block-tridiagonal solve plus one VJP through `f`, bypassing
the runner. Memory is O(1) in step count; gradient accuracy is bounded
by the forward residual `||f(y*, theta)||` (i.e. `yconv_cri`).
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from chem import chem_rhs, chem_jac_analytical, NetworkArrays
from jax_step import (
    AtmStatic, DiffGrav, compute_diff_grav,
    _build_diff_coeffs_jax, _apply_diffusion_jax,
)
from solver import block_thomas_diag_offdiag

jax.config.update("jax_enable_x64", True)


class SteadyStateInputs(NamedTuple):
    """Differentiable inputs to the steady-state residual."""
    k_arr: jnp.ndarray
    Kzz: jnp.ndarray
    Dzz: jnp.ndarray
    dzi: jnp.ndarray
    vz: jnp.ndarray
    Hpi: jnp.ndarray
    Ti: jnp.ndarray
    Tco: jnp.ndarray
    g: jnp.ndarray
    ms: jnp.ndarray
    alpha: jnp.ndarray
    M: jnp.ndarray
    vm: jnp.ndarray
    vs: jnp.ndarray
    top_flux: jnp.ndarray
    bot_flux: jnp.ndarray
    bot_vdep: jnp.ndarray
    gas_indx_mask: jnp.ndarray
    use_vm_mol: jnp.ndarray
    use_settling: jnp.ndarray
    use_topflux: jnp.ndarray
    use_botflux: jnp.ndarray


def build_steady_state_inputs(k_arr: jnp.ndarray, atm: AtmStatic) -> SteadyStateInputs:
    """Pack a runtime `AtmStatic` plus `k_arr` into the differentiable API."""
    return SteadyStateInputs(
        k_arr=k_arr,
        Kzz=atm.Kzz,
        Dzz=atm.Dzz,
        dzi=atm.dzi,
        vz=atm.vz,
        Hpi=atm.Hpi,
        Ti=atm.Ti,
        Tco=atm.Tco,
        g=atm.g,
        ms=atm.ms,
        alpha=atm.alpha,
        M=atm.M,
        vm=atm.vm,
        vs=atm.vs,
        top_flux=atm.top_flux,
        bot_flux=atm.bot_flux,
        bot_vdep=atm.bot_vdep,
        gas_indx_mask=atm.gas_indx_mask,
        use_vm_mol=jnp.asarray(atm.use_vm_mol),
        use_settling=jnp.asarray(atm.use_settling),
        use_topflux=jnp.asarray(atm.use_topflux),
        use_botflux=jnp.asarray(atm.use_botflux),
    )


def _atm_from_inputs(inputs: SteadyStateInputs) -> AtmStatic:
    return AtmStatic(
        Kzz=inputs.Kzz,
        Dzz=inputs.Dzz,
        dzi=inputs.dzi,
        vz=inputs.vz,
        Hpi=inputs.Hpi,
        Ti=inputs.Ti,
        Tco=inputs.Tco,
        g=inputs.g,
        ms=inputs.ms,
        alpha=inputs.alpha,
        M=inputs.M,
        vm=inputs.vm,
        vs=inputs.vs,
        top_flux=inputs.top_flux,
        bot_flux=inputs.bot_flux,
        bot_vdep=inputs.bot_vdep,
        gas_indx_mask=inputs.gas_indx_mask,
        use_vm_mol=inputs.use_vm_mol,
        use_settling=inputs.use_settling,
        use_topflux=inputs.use_topflux,
        use_botflux=inputs.use_botflux,
    )


def steady_state_residual_inputs(
    y: jnp.ndarray,
    inputs: SteadyStateInputs,
    net: NetworkArrays,
    grav: DiffGrav | None = None,
) -> jnp.ndarray:
    """Compute f(y, inputs) = chem_rhs(y) + diffusion(y)."""
    atm = _atm_from_inputs(inputs)
    if grav is None:
        grav = compute_diff_grav(atm)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = _build_diff_coeffs_jax(
        y, atm, grav,
    )
    diff_at_y = _apply_diffusion_jax(y, A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm)
    return chem_rhs(y, atm.M, inputs.k_arr, net) + diff_at_y


def steady_state_residual(
    y: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    grav: DiffGrav | None = None,
) -> jnp.ndarray:
    """f(y, k_arr) = chem_rhs(y) + diffusion(y). At convergence ||f|| → 0.

    The residual norm bounds implicit-gradient accuracy. `grav` is recomputed
    if not supplied; callers can pass it to skip the rebuild.
    """
    return steady_state_residual_inputs(
        y,
        build_steady_state_inputs(k_arr, atm),
        net,
        grav=grav,
    )


def _build_jacobian_blocks(y, k_arr, atm, net):
    """Return (diag, sup_d, sub_d) for J_y = ∂f/∂y in the format
    `block_thomas_diag_offdiag` expects (dense diag, diagonal-in-species
    super/sub)."""
    grav = compute_diff_grav(atm)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = _build_diff_coeffs_jax(
        y, atm, grav,
    )
    chem_J = chem_jac_analytical(y, atm.M, k_arr, net)            # (nz, ni, ni)
    diag_d = A_eddy[:, None] + A_mol                              # (nz, ni)
    sup_d = B_eddy[:-1, None] + B_mol[:-1]                        # (nz-1, ni)
    sub_d = C_eddy[1:, None] + C_mol[1:]                          # (nz-1, ni)

    ni = atm.ms.shape[0]
    di = jnp.arange(ni)
    diag = chem_J.at[:, di, di].add(diag_d)                       # (nz, ni, ni)
    bot_vdep_term = jnp.where(
        atm.use_botflux,
        -atm.bot_vdep / atm.dzi[0],
        jnp.zeros_like(atm.bot_vdep),
    )
    diag = diag.at[0, di, di].add(bot_vdep_term)
    return diag, sup_d, sub_d


def validate_steady_state_solution(
    y_star: jnp.ndarray,
    inputs: SteadyStateInputs,
    net: NetworkArrays,
    residual_rtol: float = 1e-6,
    residual_atol: float = 0.0,
) -> float:
    """Require a residual-small steady state before attaching implicit AD."""
    resid = steady_state_residual_inputs(y_star, inputs, net)
    resid_inf = float(jnp.max(jnp.abs(resid)))
    state_scale = float(jnp.max(jnp.abs(y_star)))
    tol = max(residual_atol, residual_rtol * max(state_scale, 1.0))
    if resid_inf > tol:
        raise ValueError(
            "Steady-state residual is too large for reliable implicit differentiation: "
            f"||f||_inf={resid_inf:.3e} exceeds tolerance {tol:.3e}. "
            "Tighten the forward convergence criterion or provide a more converged state."
        )
    return resid_inf


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def differentiable_steady_state_inputs(
    inputs: SteadyStateInputs,
    y_star: jnp.ndarray,
    net: NetworkArrays,
) -> jnp.ndarray:
    """Treat ``y_star`` as the converged state of `f(y*, inputs) = 0`."""
    del net
    return y_star


def _ssi_fwd(inputs, y_star, net):
    return y_star, (inputs, y_star)


def _ssi_bwd(net, res, v):
    inputs, y_star = res
    atm = _atm_from_inputs(inputs)
    diag, sup_d, sub_d = _build_jacobian_blocks(y_star, inputs.k_arr, atm, net)
    diag_T = jnp.transpose(diag, (0, 2, 1))
    lambda_ = block_thomas_diag_offdiag(diag_T, sub_d, sup_d, v)

    def f_of_inputs(inp):
        return steady_state_residual_inputs(y_star, inp, net)

    _, vjp_fn = jax.vjp(f_of_inputs, inputs)
    (cot_inputs,) = vjp_fn(lambda_)
    cot_inputs = jax.tree.map(
        lambda x: x if getattr(x, "dtype", None) == jax.dtypes.float0 else -x,
        cot_inputs,
    )
    return (cot_inputs, jnp.zeros_like(y_star))


differentiable_steady_state_inputs.defvjp(_ssi_fwd, _ssi_bwd)


def checked_differentiable_steady_state(
    inputs: SteadyStateInputs,
    y_star: jnp.ndarray,
    net: NetworkArrays,
    residual_rtol: float = 1e-6,
    residual_atol: float = 0.0,
) -> jnp.ndarray:
    """Attach implicit reverse-mode AD only after a residual check passes."""
    validate_steady_state_solution(
        y_star,
        inputs,
        net,
        residual_rtol=residual_rtol,
        residual_atol=residual_atol,
    )
    return differentiable_steady_state_inputs(inputs, y_star, net)


def steady_state_value_and_grad(
    loss_fn,
    inputs: SteadyStateInputs,
    y_star: jnp.ndarray,
    net: NetworkArrays,
    residual_rtol: float = 1e-6,
    residual_atol: float = 0.0,
):
    """Evaluate a loss of the steady state and its gradient wrt structured inputs.

    This is the preferred entrypoint when differentiating with respect to the
    full `SteadyStateInputs` pytree because that pytree intentionally contains
    non-inexact leaves (boolean masks / mode flags) that should be carried
    through structurally, not differentiated.
    """
    validate_steady_state_solution(
        y_star,
        inputs,
        net,
        residual_rtol=residual_rtol,
        residual_atol=residual_atol,
    )

    def wrapped(inp):
        y_diff = differentiable_steady_state_inputs(inp, y_star, net)
        return loss_fn(y_diff)

    return jax.value_and_grad(wrapped, allow_int=True)(inputs)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def differentiable_steady_state(
    k_arr: jnp.ndarray,
    y_star: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
) -> jnp.ndarray:
    """Expose `y_star` as a differentiable function of `k_arr` via the
    implicit-function theorem.

    Forward returns y_star unchanged; backward uses the IFT so
    `jax.grad(loss)(k_arr)` accounts for the implicit dependence of y*
    on k_arr without backpropagating through the runner's while_loop.
    """
    inputs = build_steady_state_inputs(k_arr, atm)
    return differentiable_steady_state_inputs(inputs, y_star, net)


def _ss_fwd(k_arr, y_star, atm, net):
    return y_star, (k_arr, y_star)


def _ss_bwd(atm, net, res, v):
    """Implicit-function-theorem backward.

    Solve J_y^T λ = v, then cot_k = -(∂f/∂k)^T λ via jax.vjp. For our
    block-tridiag with diagonal-in-species off-diagonals, transposing
    swaps super and sub but keeps them diagonal.
    """
    k_arr, y_star = res
    diag, sup_d, sub_d = _build_jacobian_blocks(y_star, k_arr, atm, net)

    diag_T = jnp.transpose(diag, (0, 2, 1))
    lambda_ = block_thomas_diag_offdiag(diag_T, sub_d, sup_d, v)

    # Diffusion is k-independent so the f-VJP only sees chem_rhs.
    def f_of_k(k):
        return chem_rhs(y_star, atm.M, k, net)
    _, vjp_fn = jax.vjp(f_of_k, k_arr)
    (cot_k,) = vjp_fn(lambda_)

    # The minus sign comes from ∂y*/∂k = -(∂f/∂y)^{-1} (∂f/∂k). y_star
    # is treated as constant — it comes from an external solver.
    return (-cot_k, jnp.zeros_like(y_star))


differentiable_steady_state.defvjp(_ss_fwd, _ss_bwd)
