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
   because `c_e^T df/dk = 0`. In discrete practice the `c_e` are only
   *approximately* null — the diffusion stencil is not exactly conservative
   under the dz weights; `info["null_quality"]` measures the actual defect
   relative to the operator scale (~3e-5 on the healthy HD189 column, O(1)
   when conservation is genuinely broken, e.g. open boundary fluxes).
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
   On top of that ceiling there is a *solver* regime question, controlled by
   `body_dt` (an adjoint-only probe-step knob; the forward model is untouched):
   at `body_dt >= 1e8` the solve stagnates (resid ~0.2-0.7 — the body map has
   unstable top-layer H/H2 eigenmodes, |lambda| up to ~2.7, and the matvec's
   FP-cancellation floor grows with dt), and the stagnated endpoint is bit-level
   trajectory-sensitive: dominant-reaction magnitudes bounce ~+/-25% of FD while
   sign and ranking stay stable. The default `body_dt = 1e7` sits in the
   measured sweet spot (HD189: resid 0.04-0.15, 0.3-6% vs FD across twins, mean
   3.5%); `body_dt ~ 3e6` converges fully but deterministically underweights
   slow chemistry (~28% bias). The default `n_solves` ensemble reports the
   twin-to-twin spread (`info["ensemble_spread"]`) as the magnitude error bar —
   trust magnitudes when resid and spread are small, ranking always. Full dt
   map + campaign log: docs/notes.md (2026-07-01).
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

BODY_MAP_DT = 1e7
# Time step (s) for the bare Ros2 body map G(y) = ros2_step(y, k, dt). This is
# an ADJOINT-ONLY knob: it never touches the forward model — it sets how the
# one probe step weights each chemical mode in the adjoint linear system
# (weight ~ dt/tau for slow modes; FP-noise amplification grows with dt).
# Measured dt map on HD189 (docs/notes.md, 2026-07-01 campaign; CH4 vs FD):
#   1e6  diverges;  3e6 converges but ~28% slow-mode bias (deterministic);
#   1e7  resid 0.04-0.15, 0.3-6% vs FD over ulp-twins (mean 3.5%)  <- default;
#   1e8  stalls at resid 0.2-0.7, magnitudes bounce ~+/-25% (the old default);
#   >=3e8 diverges (unstable top-layer H/H2 modes amplify);
#   ~1e11 the implicit stage (I/(gamma*h) - J) goes singular (hard-guarded).
# The usable window is column/network-dependent: when moving to a new planet,
# scan a few dt values with short solves and keep the lowest info["resid"].

_BODY_MAP_DT_MAX = 1e10
# Hard guard above the safe regime — refuse a body_dt that would land in the
# near-singular danger zone rather than return a silently divergent gradient.

N_SOLVES_DEFAULT = 3
# Ensemble size: the gradient is returned as the MEAN over this many solves
# with ulp-perturbed right-hand sides (deterministic, seeded). A stalled solve
# is bit-level trajectory-sensitive, so independent twins turn the residual
# noise into a measurable spread (info["ensemble_spread"]) instead of a
# silent lottery. Set n_solves=1 to reproduce the old single-solve behavior.

_TWIN_PERTURB = 1e-13
# Relative RHS perturbation used to generate ensemble twins. Large enough to
# decorrelate the Krylov trajectories of a semi-converged solve, ~13 orders
# below the gradient signal so the exact solution is unchanged at reporting
# precision.

_SPREAD_WARN = 0.15
# Warn above this ensemble spread (max over the top-10 reactions by |mean| of
# (max-min)/|mean|): the twins disagree on the reactions one would report, so
# treat magnitudes as ranking weights.

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

_ADJOINT_RESID_WARN = 0.2
# Warn above this MEDIAN relative LGMRES residual across the ensemble. The
# median is robust to a single wandering twin (healthy HD189 ensemble at
# body_dt=1e7: per-twin best residuals {0.29, 0.05, 0.10}, median 0.10, mean
# gradient 5% of FD). Measured bands (2026-07-01): residuals ~0.05-0.15 <->
# magnitudes 0.3-6% of FD; >~0.3 across the ensemble <-> the stagnation regime
# where magnitudes bounce ~+/-25% (sign and ranking stay robust). Distinct from
# the few-% accuracy ceiling (a steady-state-definition mismatch that no
# solver setting removes).

_FP_ERR_WARN = 1e-2
# Warn above this body-map fixed-point error: y_star is not a tight fixed point,
# so the adjoint is being evaluated off the steady-state manifold.

_NULL_BASIS_RANK_TOL = 1e-10
# Rank guard for the deflation basis. After column normalization, |R_jj| from
# the (unpivoted) QR measures how independent atom column j is of the previous
# columns; below this it is numerically dependent, and QR's j-th Q column is an
# arbitrary orthonormal direction NOT in span(C) — deflating it would silently
# project a needed direction out of the adjoint solve, so fail fast instead.


def _warn_poor_convergence(resid: float, fp_err: float, spread: float = 0.0) -> None:
    """Emit a warning (by default, not only when the caller inspects `info`) when
    the adjoint solve looks under-converged, `y_star` is not a fixed point, or
    the ensemble twins disagree."""
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
            f"steady_state_reaction_sensitivity: median LGMRES relative residual {resid:.2e} "
            f"exceeds {_ADJOINT_RESID_WARN:.0e}: the solve is in the stagnation "
            "regime observed on closed columns (dominant-reaction magnitudes "
            "bounce ~+/-25% around FD there; sign and ranking remain robust — "
            "docs/notes.md). Treat magnitudes as ranking weights only. More "
            "cycles do not reliably reduce the residual; scan body_dt for a "
            "lower-residual regime (see BODY_MAP_DT) and check null_quality.",
            stacklevel=3,
        )
    if spread > _SPREAD_WARN:
        warnings.warn(
            f"steady_state_reaction_sensitivity: ensemble spread {spread:.2e} "
            f"exceeds {_SPREAD_WARN:.0e}: the perturbation twins disagree on the "
            "dominant-reaction magnitudes, so the solve is trajectory-sensitive "
            "at this body_dt. The MEAN is still the best estimate and the "
            "ranking is robust; for tighter magnitudes scan body_dt for a "
            "lower-residual regime.",
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
    Columns are unit-normalized before the QR, and a (near-)zero `|R_jj|`
    raises: unpivoted QR maps a rank-deficient stack to arbitrary directions
    outside span(C), which would silently corrupt the deflation.

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
    norms = np.linalg.norm(C, axis=0)
    if np.any(norms == 0.0):
        dead = [int(atom_cols[j]) for j in np.where(norms == 0.0)[0]]
        raise ValueError(
            f"conserved-mass null basis: atom columns {dead} are all-zero "
            "(every carrier of that atom has y* == 0), so their conservation "
            "directions are degenerate. Drop those atoms from compo_array."
        )
    # Unit columns so |diag(R)| below directly measures column independence.
    C = C / norms
    Q, R = np.linalg.qr(C)
    rdiag = np.abs(np.diag(R))
    if np.any(rdiag < _NULL_BASIS_RANK_TOL):
        dep = [int(atom_cols[j]) for j in np.where(rdiag < _NULL_BASIS_RANK_TOL)[0]]
        raise ValueError(
            f"conserved-mass null basis is rank-deficient: atom columns {dep} "
            f"are (near-)linearly dependent on the others (|R_jj| < "
            f"{_NULL_BASIS_RANK_TOL:.0e}). Unpivoted QR would emit arbitrary "
            "directions outside span(C) and the deflation would corrupt the "
            "adjoint solve. Drop the dependent atom(s) from compo_array."
        )
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

    Cycles stop early once scipy reports convergence to `rtol` (info == 0);
    `info > 0` (not yet converged after `maxiter` inner iterations) continues
    into the next warm-start cycle, and a breakdown / illegal input
    (info < 0) raises instead of silently returning garbage.

    Returns the BEST-residual iterate seen across cycles, not the last one:
    on this indefinite operator the warm-restart trajectory is not monotone
    in residual (a later cycle can wander well past its best point — observed
    on HD189: resid 0.16 at cycle 4 drifting to 0.55 by cycle 8). Tracking
    the best iterate costs one extra matvec per cycle.
    """
    import scipy.sparse.linalg as spla

    n = bvec.shape[0]
    A_op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    x = np.zeros(n)
    x_best = x
    r_best = np.inf
    bnorm = max(float(np.linalg.norm(bvec)), 1e-300)
    for _ in range(cycles):
        x, info = spla.lgmres(
            A_op,
            bvec,
            x0=x,
            rtol=rtol,
            atol=0.0,
            inner_m=inner_m,
            outer_k=outer_k,
            maxiter=maxiter,
        )
        if info < 0:
            raise RuntimeError(
                f"scipy lgmres reported breakdown/illegal input (info={info}); "
                "the adjoint solve did not produce a usable solution. Check "
                "body_dt and the deflation basis (the null_quality diagnostic)."
            )
        r = float(np.linalg.norm(matvec(x) - bvec)) / bnorm
        if r < r_best:
            r_best = r
            x_best = x
        if info == 0:
            # Converged; the warm-start cycles exist to continue an
            # unconverged solve, not to re-iterate a converged one.
            break
    return x_best


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
    n_solves: int = N_SOLVES_DEFAULT,
    return_info: bool = False,
):
    """Reverse-mode `dL/d(ln k_r)` at a converged steady state.

    Returns the sensitivity of a scalar loss of the converged composition to
    every reaction-rate constant — the reaction-ranking use case (which
    reactions set the converged SO2, CH4, ...). The result is the MEAN over an
    ensemble of `n_solves` adjoint solves with deterministic ulp-perturbed
    right-hand sides; the twin-to-twin disagreement is reported as
    `info["ensemble_spread"]` and is the honest magnitude error bar.

    Limitations (see the module docstring for the full reasoning):

    * Accuracy is ~few % vs finite differences at best — a steady-state-
      *definition* ceiling (the `f=0` adjoint vs the convergence-criterion
      state the run actually stops at), not solver error; more iterations do
      not close it. At the default `body_dt` (1e7, validated on HD189: 0.3-6%
      vs FD across twins, mean 3.5%) the solve reaches a low residual; at
      `body_dt>=1e8` it stagnates and dominant-reaction magnitudes bounce
      ~+/-25% around FD (the resid/spread warnings fire; sign and ranking
      stay stable). The usable `body_dt` window is column-dependent — scan a
      few values and keep the lowest `info["resid"]` (see `BODY_MAP_DT`).
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
        Converged state — a tight fixed point of the *bare* body map
        `G(y) = jax_ros2_step(y, k, body_dt, atm, net)` (no clip, charge
        balance, or hydrostatic renormalization; `info["fp_err"]` reports how
        tight). A state polished on the renormalized map is acceptable as
        long as the renormalization correction at convergence keeps `fp_err`
        below the warn threshold.
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
        Body-map probe step (adjoint-only knob; see `BODY_MAP_DT` for the
        measured dt map and the per-column scan recipe).
    lgmres_inner_m, lgmres_outer_k, lgmres_maxiter, lgmres_cycles, rtol
        LGMRES knobs (see the module constants).
    n_solves
        Ensemble size. The gradient is the mean over `n_solves` solves with
        deterministic ulp-perturbed right-hand sides (`_TWIN_PERTURB`, seeded);
        `n_solves=1` reproduces the old single-solve behavior. Each extra solve
        costs the same LGMRES budget; the operator compile is shared.
    return_info
        If True, also return a diagnostics dict.

    Returns
    -------
    dL_dlnk : (nr+1,)
        Ensemble-mean `dL/d(ln k_r)` for every reaction (index 0 is the unused
        1-based pad).
    info : dict, optional
        `fp_err` (body-map fixed-point error at y*), `null_quality`
        (max over deflated directions of `||A_eta^T q_e||`, unit-norm `q_e`,
        relative to the operator's action on a random unit direction: ~3e-5
        on a healthy closed HD189 column — the conserved-mass vectors are
        only approximately null because the diffusion discretization is not
        exactly conservative under the dz weights; O(1) means a deflated
        direction is not null, e.g. open boundary fluxes, and the deflation
        is suspect), `resid` (max relative LGMRES residual over the ensemble),
        `resids` (per-twin residuals), `ensemble_spread` (max over the top-10
        reactions by |mean| of (max-min)/|mean| across twins; 0.0 when
        `n_solves=1`), `n_matvec`, `n_null` (deflated dimensions), `n_solves`,
        and `body_dt`.
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

    # Jit the bare operator once and share it between the null-quality
    # diagnostic and the LGMRES matvec, so the expensive step-VJP XLA compile
    # is paid exactly once. `proj` stays outside the jit — two small (n, n_e)
    # matmuls per matvec, negligible next to the step VJP.
    a_eta_j = jax.jit(a_eta)

    # Conserved-mass deflation projector.
    Q = _conserved_null_basis(y_star, compo_array, dz)

    def proj(z):
        zf = z.reshape(-1)
        return (zf - Q @ (Q.T @ zf)).reshape(nz, ni)

    def deflated(z):
        return proj(a_eta_j(proj(z)))

    # How null the deflated directions actually are under the operator:
    # max_e ||A_eta^T q_e|| over the unit-norm basis columns, relative to the
    # operator's action on a fixed-seed random unit direction. ~3e-5 on a
    # healthy closed HD189 column (the conserved-mass vectors are only
    # *approximately* null — the diffusion discretization is not exactly
    # conservative under the dz weights); O(1) means a deflated direction is
    # NOT null (e.g. open boundary fluxes break atom conservation) and the
    # deflation is corrupting the solve. (A QR-orthonormality check would be
    # vacuous here — ~1e-15 for ANY basis.)
    null_defect = max(
        float(jnp.linalg.norm(a_eta_j(Q[:, e].reshape(nz, ni))))
        for e in range(Q.shape[1])
    )
    rng = np.random.default_rng(0)  # fixed seed: the diagnostic is deterministic
    r = rng.normal(size=(nz, ni))
    r /= np.linalg.norm(r)
    op_scale = float(jnp.linalg.norm(a_eta_j(jnp.asarray(r))))
    null_quality = null_defect / max(op_scale, 1e-300)

    # RHS: log-space cotangent of the loss, deflated.
    v = jax.grad(loss_fn)(y_star)
    b = proj(y_star * v)
    bvec = np.asarray(b).ravel()

    n_matvec = [0]

    def matvec(x):
        n_matvec[0] += 1
        return np.asarray(deflated(jnp.asarray(x.reshape(nz, ni)))).ravel()

    def body_map_k(k):
        sol, _ = jax_ros2_step(y_star, k, jnp.float64(body_dt), atm, net)
        return sol

    _, vjp_Gk = jax.vjp(body_map_k, k_arr)

    # Ensemble of twin solves. A semi-converged Krylov endpoint is bit-level
    # trajectory-sensitive, so a single solve samples a distribution; solving
    # against deterministically ulp-perturbed copies of b (the perturbation is
    # ~13 orders below the signal, so the exact solution is unchanged) turns
    # that into a measurable mean + spread. Twin RHS draws are seeded and
    # reproducible; twin 0 is the unperturbed b.
    twin_rng = np.random.default_rng(0)
    n_solves = max(1, int(n_solves))
    grads = []
    resids = []
    for i in range(n_solves):
        b_i = (
            bvec
            if i == 0
            else bvec * (1.0 + _TWIN_PERTURB * twin_rng.standard_normal(bvec.shape[0]))
        )
        x = _lgmres_solve(
            matvec,
            b_i,
            inner_m=lgmres_inner_m,
            outer_k=lgmres_outer_k,
            maxiter=lgmres_maxiter,
            cycles=lgmres_cycles,
            rtol=rtol,
        )
        z = proj(jnp.asarray(x.reshape(nz, ni)))
        resids.append(
            float(
                jnp.linalg.norm(proj(a_eta_j(z)).reshape(-1) - jnp.asarray(b_i))
                / max(float(np.linalg.norm(b_i)), 1e-300)
            )
        )
        # Reaction cotangent: lambda = z ./ y*, then
        # dL/d(ln k_r) = (k .* G_k^T lambda)_r.
        lam = z * inv_y
        (cot_k,) = vjp_Gk(lam)  # (nr+1, nz)
        grads.append(np.asarray((k_arr * cot_k).sum(axis=1)))  # (nr+1,)

    g_stack = np.stack(grads, axis=0)  # (n_solves, nr+1)
    dL_dlnk = jnp.asarray(g_stack.mean(axis=0))
    resid = max(resids)
    resid_median = float(np.median(resids))

    # Twin-to-twin disagreement on the reactions one would actually report:
    # max over the TOP-10 reactions by |mean| of (max-min)/|mean|. Weaker
    # reactions naturally carry larger relative bounce; including them makes
    # the metric alarm on noise that never enters a ranking figure.
    if n_solves > 1:
        g_mean = g_stack.mean(axis=0)
        order = np.argsort(np.abs(g_mean))[::-1]
        top = order[: min(10, order.size)]
        top = top[np.abs(g_mean[top]) > 0.0]
        if top.size:
            width = g_stack[:, top].max(axis=0) - g_stack[:, top].min(axis=0)
            ensemble_spread = float(np.max(width / np.abs(g_mean[top])))
        else:
            ensemble_spread = 0.0
    else:
        ensemble_spread = 0.0

    # Default-on diagnostics: a poorly-converged solve still returns a
    # finite-looking gradient, so warn even when the caller ignores `info`.
    # The residual warning gates on the ensemble MEDIAN (robust to one
    # wandering twin); `info["resid"]` still reports the max.
    _warn_poor_convergence(resid_median, fp_err, ensemble_spread)

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
        "resids": resids,
        "ensemble_spread": ensemble_spread,
        "n_matvec": int(n_matvec[0]),
        "n_null": int(Q.shape[1]),
        "n_solves": n_solves,
        "body_dt": float(body_dt),
    }
    return dL_dlnk, info
