"""Reverse-mode reaction sensitivities at the converged photochemical state.

`outer_loop.OuterLoop`'s `lax.while_loop` supports `jvp`/`jacfwd` but not
`vjp`/`grad`, so reverse-mode AD cannot be taken straight through the
integration. The reverse-mode question this module answers is the one a
many-inputs/one-output gradient is for: *which of the network's reactions set
the converged abundance of a given species* — `dL/d(ln k_r)` for all `nr`
reactions from a single adjoint solve, where finite differences would cost one
re-converged model per reaction.

The route — the solver-map steady-state adjoint
================================================
At convergence the body map `G` (one bare Ros2 step) has a fixed point,
`G(y*) = y*`, so the steady-state cotangent solves the *fixed-point* adjoint

    (I - dG/dy)^T z = v ,        v = dL/dy* ,

after which `dL/d(ln k_r) = (k .* vjp_Gk(lambda))_r`, with `lambda` the
unscaled cotangent. `(I - dG/dy)^T` is the integrator's *own* regularized
implicit step (the block-Thomas solve at the body-map dt) transposed — it
already embeds the preconditioner that tames the chemical stiffness, which is
why it is far better conditioned than the bare residual Jacobian.

Four coupled ingredients make the solve work on a real, closed atmospheric
column (each fixes one concrete failure mode — see "What was tried and failed"):

1. **The solver-map, not the residual Jacobian.** A separately-formed
   `reg*I - df/dy` preconditioner cannot reproduce the integrator's step; the
   solver-map can, because it *is* that step.
2. **Log-abundance coordinates** `eta = ln y`. The similarity transform
   `A_eta z = z - y* .* vjp_Gy(z ./ y*)` rescales the operator norm from ~1e6
   to ~1e2 and the cotangent from ~1e-12 to O(1) — the scaling LSQR/GMRES never
   had. `lambda = z ./ y*`, `v_eta = y* .* v`.
3. **Conserved-mass null-space deflation.** A closed column conserves each
   element, so `df/dy` (and `I - dG/dy`) is singular. We deflate the analytic
   per-element atom-count vectors `c_e[z,i] = compo[i,e] * dz[z] * y*[z,i]`
   (the log-space left-null vectors) with a QR projector. Only the *left* null
   space is needed: the right null cancels from atom-conserving-knob gradients
   because `c_e^T df/dk = 0`.
4. **LGMRES, not restarted GMRES or Neumann.** The deflated operator is
   indefinite; an augmented Krylov method (LGMRES carries vectors across
   restarts) converges where restarted GMRES oscillates and a raw Neumann
   iteration diverges.

Limitations — read before using
===============================
This is a reaction-*ranking* tool, not a precision-gradient tool. Four limits,
all structural — none is fixable by iterating the solver harder:

1. **~few-% accuracy ceiling, and it is a definition mismatch, not solver error.**
   `lax.while_loop` blocks `vjp`, so reverse-mode can only take the *steady-state*
   adjoint — it differentiates the exact fixed point `f(y*) = 0`. But the forward
   run stops on a convergence *criterion* (`longdy < yconv_cri`) with the slowest
   near-conserved chemical mode still unrelaxed: a genuinely different state.
   Finite differences and forward-mode both differentiate that criterion state
   (forward-mode can, because `jvp` rides through the loop), so they agree with
   each other and disagree with this adjoint by ~few % — larger on reactions
   coupled to the slow mode, ~2% on fast ones. Reaching <1% would need integrating
   to >> the slow-mode timescale (impractical). More LGMRES iterations do not help.
2. **Photolysis is frozen on photochemistry-on columns.** `J(y)` depends on the
   abundances through optical depth; the adjoint holds `J` at its converged value
   and omits the `dJ/dy` feedback, so those sensitivities are leading-order only.
3. **Reaction-rate (`k`) sensitivities only.** Returns `dL/d(ln k_r)`; it does not
   give `dL/dKzz`, `dL/dT`, etc. Use forward-mode for those (few inputs, one pass).
4. **Needs a genuine fixed point and a safe `body_dt`.** `y*` must be a tight fixed
   point of the bare body map (`info["fp_err"]` reports how tight), and `body_dt`
   must stay in the safe regime (see `BODY_MAP_DT`; the danger zone is guarded).

The *ranking* is robust despite the ceiling because the dominant reactions stand
1-2 orders of magnitude above the few-% noise. **Forward-mode** (`jvp`/`jacfwd`
through the runner, FD-validated <0.1%) is the higher-accuracy route for
end-to-end gradients and the right tool when the number of input directions is
small (Kzz, metallicity, temperature); reverse-mode here is the right tool for
the opposite shape — all `nr` reactions at once.

What was tried and failed (do not re-walk)
==========================================
Earlier attempts took the adjoint of the *residual* `f = chem_rhs + diffusion`,
solving `(df/dy)^T lambda = v` directly. On a real closed column `df/dy` is
*both* singular (mass conservation) *and* severely ill-conditioned (stiff
chemistry — the residual at the converged state is ~1e21), and every direct
solver disagreed with finite differences: a frozen-coefficient block-Thomas
factorization refined by defect-correction, and a matrix-free LSQR pseudoinverse,
both diverged or stagnated. A fixed-point/iteration-map adjoint on the body map
is better conditioned but, run as a *raw Neumann* iteration, is non-contractive
(unstable total-density mode) and still singular. The working route above
replaces all of these.

Full development log and identities: `docs/notes.md` ("End-to-end AD" /
"2026-06-16: reverse-mode steady-state adjoint"). Worked recipe:
`examples/grad_reverse_example.py`; paper Fig `fig:so2_rev`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp

from .chem import NetworkArrays
from .jax_step import AtmStatic, jax_ros2_step

jax.config.update("jax_enable_x64", True)


# --- Solver-map / LGMRES knobs (adjoint-solver constants live here, beside the
#     code, the same as the forward model's knobs live in vulcan_cfg.py). ---

BODY_MAP_DT = 1e8
# Time step for the bare Ros2 body map G(y) = ros2_step(y, k, dt). DANGER ZONE:
# at dt ~ 1e11 the implicit stage solve (I/(gamma*h) - J) goes near-singular and
# the adjoint diverges (residual ~1e57); 1e8 is the validated, well-conditioned
# regime (docs/notes.md, 2026-06-16).

_BODY_MAP_DT_MAX = 1e10
# Hard guard above the safe regime — refuse a body_dt that would land in the
# near-singular danger zone rather than return a silently divergent gradient.

LGMRES_INNER_M = 60
# scipy.sparse.linalg.lgmres inner Krylov dimension. The HD189 validation used
# 250 (nr=878); 60 is the conservative default that converged on WASP-39b
# (nr=1150). Larger m costs memory and matvecs per cycle.

LGMRES_OUTER_K = 40
# Augmentation vectors carried across restarts — the LGMRES knob that fixes the
# restarted-GMRES oscillation on this indefinite operator.

LGMRES_MAXITER = 4
# Inner iterations per warm-start cycle. The solve is chunked into cycles so the
# residual/gradient trajectory is observable; the per-cycle x0 warm-start is the
# validated configuration.

LGMRES_CYCLES = 10
# Number of warm-start cycles. ~6 sufficed on HD189, ~10 on WASP-39b.

LGMRES_RTOL = 1e-12
# Relative-residual target. The few-% accuracy ceiling is a steady-state
# definition mismatch, not under-iteration, so an aggressive rtol is harmless.

_ADJOINT_RESID_WARN = 0.1
# Warn above this final relative LGMRES residual: the solve is under-converged
# and the gradient may be unreliable (raise lgmres_cycles/inner_m, check body_dt).
# This is distinct from the few-% accuracy ceiling (a definition mismatch).

_FP_ERR_WARN = 1e-2
# Warn above this body-map fixed-point error: y_star is not a tight fixed point,
# so the adjoint is being evaluated off the steady-state manifold.


def _warn_poor_convergence(resid: float, fp_err: float) -> None:
    """Emit a warning (by default, not only when the caller inspects `info`) when
    the adjoint solve looks under-converged or `y_star` is not a fixed point."""
    if fp_err > _FP_ERR_WARN:
        warnings.warn(
            "steady_state_reaction_sensitivity: y_star is not a tight fixed point "
            f"of the body map (fp_err={fp_err:.2e} > {_FP_ERR_WARN:.0e}); the adjoint "
            "is evaluated off the steady-state manifold and the gradient may be "
            "unreliable. Converge y_star tighter (or lower body_dt).",
            stacklevel=3,
        )
    if resid > _ADJOINT_RESID_WARN:
        warnings.warn(
            f"steady_state_reaction_sensitivity: LGMRES relative residual {resid:.2e} "
            f"exceeds {_ADJOINT_RESID_WARN:.0e}; the gradient may be under-converged "
            "(increase lgmres_cycles or lgmres_inner_m, or check body_dt). The few-% "
            "accuracy ceiling is separate (a steady-state-definition mismatch) and is "
            "not fixed by more iterations.",
            stacklevel=3,
        )


def _safe_inv_y(y_star: jnp.ndarray) -> jnp.ndarray:
    """Elementwise 1/y* with exact zeros mapped to 0 (not inf).

    Closed columns clip trace species to *exactly* 0.0; the log-abundance
    scaling would otherwise hit 1/0 and poison the whole adjoint with NaN. A
    zeroed species becomes an identity row of the log operator (its cotangent
    is left untouched), which is the correct leading-order behaviour.
    """
    pos = y_star > 0.0
    return jnp.where(pos, 1.0 / jnp.where(pos, y_star, 1.0), 0.0)


def _conserved_null_basis(
    y_star: jnp.ndarray, compo_array: jnp.ndarray, dz: jnp.ndarray
) -> jnp.ndarray:
    """Orthonormal basis Q (n, n_e) of the log-space conserved-mass null space.

    For each tracked element `e`, mass conservation `sum_i compo[i,e]*dz*y_i`
    is constant, whose log-space gradient is `c_e[z,i] = compo[i,e]*dz[z]*y*[z,i]`.
    Stacking the active elements (columns of `compo` with any atoms) and taking
    a QR gives the projector basis that deflates the singular directions.

    Shapes: y_star (nz, ni), compo_array (ni, n_atoms), dz (nz,) -> Q (nz*ni, n_e).
    Built on the host (one-shot, off the hot path).
    """
    y_np = np.asarray(y_star)
    compo_np = np.asarray(compo_array)
    dz_np = np.asarray(dz)
    atom_cols = np.where(compo_np.sum(axis=0) > 0)[0]
    if atom_cols.size == 0:
        raise ValueError(
            "compo_array has no populated atom columns; cannot build the "
            "conserved-mass null space. Pass composition.compo_array[:ni]."
        )
    cols = [
        (y_np * (compo_np[:, e][None, :] * dz_np[:, None])).ravel() for e in atom_cols
    ]
    C = np.stack(cols, axis=1)
    Q, _ = np.linalg.qr(C)
    return jnp.asarray(Q)


def _lgmres_solve(
    matvec: Callable[[np.ndarray], np.ndarray],
    bvec: np.ndarray,
    *,
    inner_m: int,
    outer_k: int,
    maxiter: int,
    cycles: int,
    rtol: float,
) -> np.ndarray:
    """Solve A x = b with scipy LGMRES, chunked over warm-start cycles.

    The Krylov solve is host-side scipy because JAX exposes no LGMRES (only
    restarted `gmres`, which oscillates on this indefinite operator). Each
    matvec is a single host<->device round trip through the jitted JAX
    operator; this routine runs once, post-convergence, off the hot path.
    """
    import scipy.sparse.linalg as spla

    n = bvec.shape[0]
    A_op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    x = np.zeros(n)
    for _ in range(cycles):
        x, _info = spla.lgmres(
            A_op,
            bvec,
            x0=x,
            rtol=rtol,
            atol=0.0,
            inner_m=inner_m,
            outer_k=outer_k,
            maxiter=maxiter,
        )
    return x


def steady_state_reaction_sensitivity(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    *,
    compo_array: jnp.ndarray,
    dz: jnp.ndarray,
    body_dt: float = BODY_MAP_DT,
    lgmres_inner_m: int = LGMRES_INNER_M,
    lgmres_outer_k: int = LGMRES_OUTER_K,
    lgmres_maxiter: int = LGMRES_MAXITER,
    lgmres_cycles: int = LGMRES_CYCLES,
    rtol: float = LGMRES_RTOL,
    return_info: bool = False,
):
    """Reverse-mode `dL/d(ln k_r)` at a converged steady state.

    One solver-map adjoint solve returns the sensitivity of a scalar loss of the
    converged composition to every reaction-rate constant — the reaction-ranking
    use case (which reactions set the converged SO2, CH4, ...).

    Limitations (this is a ranking tool, not a precision-gradient tool — see the
    module docstring for the full reasoning):

    * Accuracy is ~few % vs finite differences, a steady-state-*definition*
      ceiling (the `f=0` adjoint vs the convergence-criterion state the run
      actually stops at), not solver error — more iterations do not close it.
    * Photolysis is held frozen on photochemistry-on columns (`dJ/dy` omitted);
      those sensitivities are leading-order only.
    * Returns `k`-only sensitivities (`dL/d ln k`); for `dL/dKzz`, `dL/dT`, etc.
      use forward-mode (`jvp`).
    * Requires `y_star` to be a tight fixed point of the bare body map and a
      `body_dt` in the safe regime.

    Parameters
    ----------
    loss_fn
        `loss_fn(y) -> scalar` on the full `(nz, ni)` number-density state. The
        caller closes over the species/layer of interest, e.g.
        `lambda y: jnp.log10(y[L, so2] / y[L].sum())`.
    y_star : (nz, ni)
        Converged state (a tight fixed point of the renormalized body map).
    k_arr : (nr+1, nz)
        Converged rate-constant table. Photolysis rows may be frozen at their
        converged values (leading-order; `dJ/dy` is omitted).
    atm : AtmStatic
        Atmosphere with the converged refresh fields (g, dzi, Hpi, ...), as fed
        to the runner's body map.
    net : NetworkArrays
        Active network.
    compo_array : (ni, n_atoms)
        Per-species atom counts; pass `composition.compo_array[:ni]`.
    dz : (nz,)
        Layer thickness. Required — `AtmStatic` carries only the interface
        average `dzi`, which is not invertible to `dz`. Use `AtmInputs.dz`.
    body_dt
        Body-map time step; keep in the safe regime (see `BODY_MAP_DT`).
    lgmres_inner_m, lgmres_outer_k, lgmres_maxiter, lgmres_cycles, rtol
        LGMRES knobs (see the module constants).
    return_info
        If True, also return a diagnostics dict.

    Returns
    -------
    dL_dlnk : (nr+1,)
        `dL/d(ln k_r)` for every reaction (index 0 is the unused 1-based pad).
    info : dict, optional
        `fp_err` (body-map fixed-point error at y*), `null_quality`
        (orthonormality defect of the QR deflation basis; ~0 means the projector
        cleanly removes the conserved-mass directions), `resid` (final relative
        LGMRES residual), `n_matvec`, and `n_null` (deflated dimensions).
    """
    if body_dt > _BODY_MAP_DT_MAX:
        raise ValueError(
            f"body_dt={body_dt:.1e} is in the near-singular danger zone "
            f"(> {_BODY_MAP_DT_MAX:.0e}); the implicit step goes singular and "
            f"the adjoint diverges. Use body_dt <= {_BODY_MAP_DT_MAX:.0e} "
            f"(default {BODY_MAP_DT:.0e})."
        )

    nz, ni = y_star.shape
    inv_y = _safe_inv_y(y_star)

    # Bare Ros2 body map and its y-VJP (the transposed solver-map operator).
    @jax.jit
    def body_map(y):
        sol, _ = jax_ros2_step(y, k_arr, jnp.float64(body_dt), atm, net)
        return sol

    fp_err = float(
        jnp.max(jnp.abs(body_map(y_star) - y_star))
        / jnp.maximum(jnp.max(jnp.abs(y_star)), 1e-300)
    )
    _, vjp_Gy = jax.vjp(body_map, y_star)

    def a_eta(z):  # (I - dG/deta)^T in log-abundance coordinates
        return z - y_star * vjp_Gy(z * inv_y)[0]

    # Conserved-mass deflation projector.
    Q = _conserved_null_basis(y_star, compo_array, dz)

    def proj(z):
        zf = z.reshape(-1)
        return (zf - Q @ (Q.T @ zf)).reshape(nz, ni)

    deflated = jax.jit(lambda z: proj(a_eta(proj(z))))

    # Orthonormality defect of the QR deflation basis (~0 when `proj` cleanly
    # annihilates span(C), i.e. the conserved-mass null space).
    null_quality = float(jnp.linalg.norm(Q - Q @ (Q.T @ Q)) / max(Q.shape[1], 1))

    # RHS: log-space cotangent of the loss, deflated.
    v = jax.grad(loss_fn)(y_star)
    b = proj(y_star * v)
    bvec = np.asarray(b).ravel()
    bnorm = float(np.linalg.norm(bvec))

    n_matvec = [0]

    def matvec(x):
        n_matvec[0] += 1
        return np.asarray(deflated(jnp.asarray(x.reshape(nz, ni)))).ravel()

    x = _lgmres_solve(
        matvec,
        bvec,
        inner_m=lgmres_inner_m,
        outer_k=lgmres_outer_k,
        maxiter=lgmres_maxiter,
        cycles=lgmres_cycles,
        rtol=rtol,
    )

    z = proj(jnp.asarray(x.reshape(nz, ni)))
    resid = float(
        jnp.linalg.norm(proj(a_eta(z)).reshape(-1) - jnp.asarray(bvec))
        / max(bnorm, 1e-300)
    )
    # Default-on diagnostics: a poorly-converged solve still returns a
    # finite-looking gradient, so warn even when the caller ignores `info`.
    _warn_poor_convergence(resid, fp_err)

    # Reaction cotangent: lambda = z ./ y*, then dL/d(ln k_r) = (k .* G_k^T lambda)_r.
    lam = z * inv_y

    def body_map_k(k):
        sol, _ = jax_ros2_step(y_star, k, jnp.float64(body_dt), atm, net)
        return sol

    _, vjp_Gk = jax.vjp(body_map_k, k_arr)
    (cot_k,) = vjp_Gk(lam)  # (nr+1, nz)
    dL_dlnk = (k_arr * cot_k).sum(axis=1)  # (nr+1,)

    if not bool(jnp.all(jnp.isfinite(dL_dlnk))):
        raise ValueError(
            "Reaction sensitivity is non-finite. Common causes: a body_dt in "
            "the danger zone, or a y_star that is not a fixed point of the body "
            f"map (fp_err={fp_err:.2e})."
        )

    if not return_info:
        return dL_dlnk
    info = {
        "fp_err": fp_err,
        "null_quality": null_quality,
        "resid": resid,
        "n_matvec": int(n_matvec[0]),
        "n_null": int(Q.shape[1]),
    }
    return dL_dlnk, info
