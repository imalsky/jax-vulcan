"""Reverse-mode reaction sensitivities at the converged photochemical state.

`outer_loop.OuterLoop`'s `lax.while_loop` supports `jvp`/`jacfwd` but not
`vjp`/`grad`, so reverse mode cannot go through the integration. This module
answers the many-inputs/one-output question instead: which rate-table rows set
the converged abundance of a species -- `dL/d(ln k_r)` for all `nr` rows from
ONE adjoint solve, where FD would cost one re-converged model per row.

Route: the solver-map steady-state adjoint. At convergence `G(y*) = y*`, so
solve the fixed-point adjoint

    (I - dG/dy)^T z = v ,        v = dL/dy* ,

then `dL/d(ln k_r) = (k .* vjp_Gk(lambda))_r`, with `lambda` the unscaled
cotangent. `(I - dG/dy)^T` is the integrator's own regularized implicit step
(the block-Thomas solve at the body-map dt) transposed, so it embeds the
preconditioning that tames the chemical stiffness.

Four coupled ingredients make the solve work on a real closed column:

1. The SOLVER MAP, not the residual Jacobian: only the integrator's step
   reproduces its own conditioning. `solver_map="renorm"` (DEFAULT)
   linearizes the hydrostatic-renormalized step the runner actually iterates,
   so `y*` is a tight fixed point (fp_err ~1e-9); `"bare"` (raw Ros2 step,
   fp_err ~1e-4) exists only to reproduce pre-2026-07 legacy results and
   carries a ~few-% bias no tuning removes (see `SOLVER_MAP_DEFAULT`).
2. Log-abundance coordinates `eta = ln y`: the similarity transform
   `A_eta z = z - y* .* vjp_Gy(z ./ y*)` rescales the operator norm from ~1e6
   to ~1e2 and the cotangent from ~1e-12 to O(1). Zero-clipped species are
   masked before the 1/y* scaling (they become identity rows).
3. Conserved-mass null-space deflation: a closed column conserves each
   element, so the operator is singular. Deflate the analytic log-space
   left-null vectors `c_e[z,i] = compo[i,e] * dz[z] * y*[z,i]` with a QR
   projector; only the LEFT null space is needed (the right null cancels for
   atom-conserving rate knobs). `info["null_quality"]` measures the actual
   defect -- O(1) means a deflated direction is not null for this setup
   (e.g. open boundary fluxes).
4. LGMRES: the deflated operator is indefinite; augmented Krylov (vectors
   carried across restarts) converges where restarted GMRES oscillates and a
   raw Neumann iteration diverges.

Limitations (read before using):

* The body map contains ONLY `ros2_step (+ renorm) (+ photo recompute)`.
  Every other per-step runner process (clip, condensation, charge balance,
  fix-species and boundary pins, atm-refresh feedback) is outside the
  linearization -- run `audit_adjoint_scope(...)` on the converged run first.
* Photolysis feedback: `J(y)` depends on y through optical depth. The default
  `photo_recompute_k="auto"` rebuilds J from the finished runner context so
  `dG/dy` carries dJ/dy -- REQUIRED on photo-on columns (W39b OH+H2 ~11% ->
  ~0.2% vs re-converged FD); pass `photo_recompute_k=None` only to reproduce
  the frozen-photolysis legacy result. Costs an RT solve per Krylov matvec.
* `body_dt` is an adjoint-only probe knob with a column-dependent usable
  window: scan `BODY_MAP_DT_CANDIDATES` on a new column and keep the
  lowest-residual, low-spread solution. (No built-in scan wrapper: every
  consumer, including vulcan-jwst-tool's `adjoint_diag`, drives its own loop
  over the candidates because it wants its own accept/refuse policy on each
  row.)
* The gradient is the MEAN over an `n_solves` twin ensemble and
  `info["ensemble_spread"]` is the honest magnitude error bar: trust
  magnitudes when residual and spread are small, else treat the output as a
  reaction ranking.
* Structural error floors that survive the defaults: severe operator
  ill-conditioning on slow-radical columns (near-equilibrium reverse rows
  unreliable, flagged by the residual/spread diagnostics) and the
  finite-tolerance mismatch between the exact fixed point and the state the
  forward run actually stopped at.
* Two entry points by input shape: `steady_state_reaction_sensitivity` for
  all rate-table rows, `steady_state_input_sensitivity` for an arbitrary
  physical input pytree (same solve plus one VJP of a caller-supplied
  rebuild). Forward mode stays the higher-accuracy route for a handful of
  input directions (Kzz, metallicity, temperature).

Pair sums: a physical detailed-balance perturbation of a reversible thermal
reaction uses the pair sum `g[fwd] + g[rev]` (photolysis and other one-way
rows stay single entries). The renorm + photo default is FD-validated for the
pair sums too, not only the forward rows. Do NOT read `info["pair_antisym"]`
as an error signal for the renorm map: it is a bare-map-calibrated diagnostic
and reads ~1 on a genuinely non-zero pair-sum that the bare map over-cancels.

Do not re-walk the failed routes: direct adjoints of the residual
`f = chem_rhs + diffusion` (frozen-coefficient block-Thomas with defect
correction, matrix-free LSQR) and a raw Neumann iteration on the body map all
diverged or stagnated -- `df/dy` is both singular and severely
ill-conditioned on closed columns (see the dev log, notes.md).

Scope, accuracy, and the `body_dt` regime map: README.md (Differentiability)
("Reverse mode: the steady-state adjoint"). Worked recipe:
`examples/grad_reverse_example.py`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .chem import NetworkArrays
from .conden import (
    CondenStatic,
    apply_h2o_relax_jax,
    apply_nh3_relax_jax,
    update_conden_rates,
)
from .jax_step import AtmStatic, jax_ros2_step

jax.config.update("jax_enable_x64", True)


# --- Solver-map / LGMRES knobs (adjoint-only constants) ---

SOLVER_MAP_DEFAULT = "renorm"
# Which one-step map the adjoint linearizes at the converged state:
#   "renorm" -> G(y) = M * ros2_step(y, k, dt) / sum_i ros2_step   (DEFAULT)
#   "bare"   -> G(y) = ros2_step(y, k, dt)                         (legacy)
# The forward runner iterates the hydrostatic-renormalized map, so y_star is a
# tight fixed point of "renorm" (fp_err ~1e-9) but only ~1e-4 for "bare",
# which biases the gradient at the few-percent level no matter how
# body_dt/LGMRES/convergence are tuned (HD189 CH4 ~6-8% bare -> ~0.7% renorm;
# HD209 forward rows ~35% -> ~1%). "bare" exists only to reproduce pre-2026-07
# results. On photo-on columns also pass photo_recompute_k so dG/dy carries
# dJ/dy (W39b OH+H2 ~11% -> ~0.2%). Do NOT deflate the per-layer total-density
# direction on top of "renorm": measured to over-correct (HD189 0.7% -> 2.5%).
SOLVER_MAP_CHOICES = ("bare", "renorm")

PHOTO_RECOMPUTE_AUTO = "auto"
PhotoRecomputeArg = Callable[[jnp.ndarray], jnp.ndarray] | Literal["auto"] | None

BODY_MAP_DT = 1e7
# Probe step (s) for the adjoint body map. ADJOINT-ONLY: never touches the
# forward model; it sets how one probe step weights each chemical mode
# (weight ~ dt/tau for slow modes; FP-noise amplification grows with dt).
# The usable window is column/network-dependent: scan a few values and keep
# the lowest-residual, low-spread solution.
BODY_MAP_DT_CANDIDATES = (3e6, 1e7, 3e7, 1e8)

_BODY_MAP_DT_MAX = 1e10
# Hard guard: above this the implicit step goes near-singular; refuse rather
# than return a silently divergent gradient.

N_SOLVES_DEFAULT = 3
# Ensemble size: the returned gradient is the MEAN over this many solves with
# ulp-perturbed RHS twins (deterministic, seeded); twin disagreement is
# reported as info["ensemble_spread"], the honest magnitude error bar.
# n_solves=1 reproduces the old single-solve behavior.

_TWIN_PERTURB = 1e-13
# Relative RHS perturbation for ensemble twins: large enough to decorrelate
# the Krylov trajectories of a semi-converged solve, ~13 orders below the
# gradient signal so the exact solution is unchanged.

_SPREAD_WARN = 0.15
# Warn above this ensemble spread (max over top-10 reactions by |mean| of
# (max-min)/|mean|): the twins disagree on the reactions one would report;
# treat magnitudes as ranking weights.

_UNDERFLOW_DENOM = 1e-300
# Numerical floor for `/max(|x|, .)` normalizers (norms, op-scale, pair sums).
# Below-which-is-zero guard, not a tuning knob.

LGMRES_INNER_M = 60
# scipy.sparse.linalg.lgmres inner Krylov dimension.

LGMRES_OUTER_K = 40
# Augmentation vectors carried across restarts; the LGMRES knob that fixes
# restarted-GMRES oscillation on this indefinite operator.

LGMRES_MAXITER = 4
# Inner iterations per warm-start cycle; the solve is chunked into cycles with
# per-cycle x0 warm-start (the validated configuration).

LGMRES_CYCLES = 10
# Number of warm-start cycles.

LGMRES_RTOL = 1e-12
# Relative-residual target; tighter buys nothing once finite-tolerance state
# mismatch dominates.

_ADJOINT_RESID_WARN = 0.2
# Warn above this MEDIAN relative LGMRES residual across the ensemble
# (median is robust to a single wandering twin).

_FP_ERR_WARN = 1e-2
# Warn above this body-map fixed-point error: y_star is off the steady-state
# manifold of the chosen map.
#
# NOTE: info["pair_antisym"] is deliberately NOT warning-gated. It is a
# bare-map-calibrated diagnostic that reads ~1 for the renorm default on some
# pairs even though the FD-validated pair-sums are MORE accurate than bare's
# (W39b SO+OH pair-sum 0.8% vs bare 17%); gating on it would fire on the
# accurate default path. See the "Pair sums" module-docstring section.

_NULL_BASIS_RANK_TOL = 1e-10
# Rank guard for the deflation basis: after column normalization, |R_jj| from
# the unpivoted QR measures column independence; below this the Q column is an
# arbitrary direction outside span(C), and deflating it would silently project
# a needed direction out of the solve, so fail fast.

_AUDIT_MIN_YMIX = 1e-16
# audit_adjoint_scope ignores cells below this mixing ratio in the per-cell
# defect scan: numerical dust with meaningless relative defect, far below any
# loss species (trace radicals sit ~1e-12..1e-8).

_AUDIT_DEFECT_ERROR = 0.3
# Per-cell defect above which the audit finding is an ERROR: an O(1) move
# under one probe step means the iterated map includes a process (pin, conden
# clamp, charge balance) the body map lacks. Between _FP_ERR_WARN and this is
# a WARNING: measured 6.5e-2 on the healthy HD189 fixture's slow trace cells
# while its CH4 gradient FD-validates at 0.7% -- ambiguous, not fatal.

_AUDIT_LOSS_FOOTPRINT_FRAC = 1e-3
# audit_adjoint_scope's "loss footprint": cells whose log-space cotangent
# magnitude |y* dL/dy| is within this fraction of the maximum -- the cells the
# loss actually reads, where a fixed-point defect directly biases the gradient.

_REBUILD_CONSISTENCY_WARN = 1e-8
# steady_state_input_sensitivity: warn when rebuild(p0) reproduces the
# converged (k_arr, atm) only to this relative level (rates_jax matches the
# host build to ~5e-14; anything above float noise is a subtly different map).

_REBUILD_CONSISTENCY_ERR = 1e-3
# ...and refuse above this: the adjoint would be linearized against a visibly
# different map and the gradient would be silently wrong.


_NULL_QUALITY_WARN = 1e-3
# Production sits ~3 orders below this; O(1) means a deflated direction is not
# null (e.g. open boundary fluxes) and the deflation is corrupting the solve.


def _warn_poor_convergence(
    resid: float, fp_err: float, spread: float = 0.0, null_quality: float = 0.0
) -> None:
    """Warn (even when the caller ignores `info`) on an under-converged solve,
    a loose fixed point, twin disagreement, or a non-null deflation basis.
    `pair_antisym` is intentionally not gated here (see `_FP_ERR_WARN`'s NOTE)."""
    if null_quality > _NULL_QUALITY_WARN:
        warnings.warn(
            f"steady_state_reaction_sensitivity: null_quality {null_quality:.2e} "
            f"exceeds {_NULL_QUALITY_WARN:.0e}: a deflated direction is not "
            "actually null, so the deflation is corrupting the adjoint solve. "
            "Check the boundary conditions (open fluxes break the conservation "
            "null space) before trusting the magnitudes.",
            stacklevel=3,
        )
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
            "README.md, Differentiability). Treat magnitudes as "
            "ranking weights only. More cycles do not reliably reduce the "
            "residual; scan body_dt for a lower-residual regime (see "
            "BODY_MAP_DT) and check null_quality.",
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


def _check_body_dt(body_dt: float) -> None:
    """Refuse a body_dt outside the usable window (see _BODY_MAP_DT_MAX).

    body_dt=0 makes the implicit step the identity (NaN); body_dt<0 integrates
    backwards and returns a finite, plausible-looking, WRONG sensitivity that
    nothing downstream flags.
    """
    if not np.isfinite(body_dt):
        raise ValueError(
            f"body_dt={body_dt} is not finite; the adjoint body map needs a "
            f"positive dt (default {BODY_MAP_DT:.0e})."
        )
    if body_dt <= 0.0:
        raise ValueError(
            f"body_dt={body_dt:.3e} must be > 0. At 0 the implicit step is the "
            f"identity and the adjoint solve divides by zero; negative values "
            f"integrate backwards and return a finite but WRONG sensitivity "
            f"with no other symptom. Use 0 < body_dt <= {_BODY_MAP_DT_MAX:.0e} "
            f"(default {BODY_MAP_DT:.0e})."
        )
    if body_dt > _BODY_MAP_DT_MAX:
        raise ValueError(
            f"body_dt={body_dt:.1e} is in the near-singular danger zone "
            f"(> {_BODY_MAP_DT_MAX:.0e}); the implicit step goes singular and "
            f"the adjoint diverges. Use body_dt <= {_BODY_MAP_DT_MAX:.0e} "
            f"(default {BODY_MAP_DT:.0e})."
        )


class BodyTerms(NamedTuple):
    """Optional per-step runner processes for the adjoint body map, beyond
    `ros2_step (+ renorm) (+ photo)`. Built by `make_body_terms` from a
    finished runner; all fields default-inactive (None).

    `conden_static` enables the in-window condensation composite (conden/evap
    k-rows recomputed from y each application -- the dk/dy feedback -- plus
    the H2O/NH3 relax kernels at the probe body_dt). `gas_mask`/`hydro_partial`
    switch the hydrostatic rebalance to the runner's gas-only-denominator,
    non-gas-passthrough form. `fix_mask`/`fix_y` reproduce the fix_species
    regime (pinned rows become constants of the map). `bot_idx`/`bot_val` are
    the layer-0 Dirichlet pins, applied after the balance as the runner does.
    """

    conden_static: Optional[CondenStatic] = None
    gas_mask: Optional[jnp.ndarray] = None  # (ni,) bool
    hydro_partial: bool = False
    fix_mask: Optional[jnp.ndarray] = None  # (nz, ni) bool
    fix_y: Optional[jnp.ndarray] = None  # (nz, ni)
    bot_idx: Optional[jnp.ndarray] = None  # (m,) int32
    bot_val: Optional[jnp.ndarray] = None  # (m,)


def _clip_dead_mask(G, ymix_old, cfg) -> np.ndarray:
    """Cells where the runner's per-step zero-clip would NOT be the identity.

    Applied to a candidate post-step state `G` (number density, cm^-3) with the
    pre-step mixing ratio `ymix_old`:

        y < pos_cut and y >= nega_cut          -> 0     (small/negative cut)
        ymix_old < mtol and y < 0              -> 0     (trace-negative cut)

    DELIBERATELY NARROWER than `outer_loop._make_clip_fn`, whose second rule
    zeroes EVERY negative cell regardless of `ymix_old`. Widening this to match
    would exclude more cells from the audit's defect scan; under-reporting is
    the safe direction, since an excluded cell is one the audit stops checking.

    The clip is outside the body map, so where it fires the cell has no fixed
    point and its relative defect measures the clip. Returns all-False when
    `cfg` is None (conservative: those cells stay in the scan).
    """
    G = np.asarray(G)
    if cfg is None:
        return np.zeros(G.shape, dtype=bool)
    pos_cut = float(getattr(cfg, "pos_cut", 0.0))
    nega_cut = float(getattr(cfg, "nega_cut", 0.0))
    mtol = float(getattr(cfg, "mtol", 0.0))
    dead = (G < pos_cut) & (G >= nega_cut)
    return dead | ((np.asarray(ymix_old) < mtol) & (G < 0.0))


def _make_body_map(
    y_star, k_arr, atm, net, body_dt, solver_map, photo_recompute_k, body_terms=None
):
    """Build the one-step body map `G(y)` the adjoint linearizes (unjitted).

    Returns `(apply_post_map, body_map, body_map_k, step_fn)`:
    `apply_post_map(sol, M_col=None)` is the shared post-step (plain renorm or
    the conden-aware composite; the `M_col` override is for parameter maps
    where M depends on the input), `body_map_k` is the k-linearization at
    `y_star`, `step_fn(y, k, atm)` the pinned Ros2 step for parameter maps
    that rebuild `atm`.

    SINGLE definition of the map: both sensitivity entry points and
    `audit_adjoint_scope` build from here, so the audited map is exactly the
    solved one by construction.
    """
    if solver_map not in SOLVER_MAP_CHOICES:
        raise ValueError(f"solver_map={solver_map!r} not in {SOLVER_MAP_CHOICES}.")
    t = body_terms
    has_terms = t is not None and (
        t.conden_static is not None
        or t.gas_mask is not None
        or t.fix_mask is not None
        or t.bot_idx is not None
    )
    if has_terms and solver_map != "renorm":
        raise ValueError(
            "body_terms requires solver_map='renorm': the terms reproduce the "
            "runner's renormalized composite step; the raw 'bare' map "
            "contradicts them."
        )
    # Per-layer total density for the runner's post-step hydrostatic renorm.
    M_col_default = atm.M[:, None]
    dt64 = jnp.float64(body_dt)

    def apply_post_map(sol, M_col=None):
        # "renorm" reproduces the runner's rebalance so y_star is a tight
        # fixed point; "bare" leaves the raw step (see SOLVER_MAP_DEFAULT).
        M_c = M_col_default if M_col is None else M_col
        if solver_map != "renorm":
            return sol
        if not has_terms:
            return M_c * sol / jnp.sum(sol, axis=1, keepdims=True)
        # Runner composite (body_fn, clip omitted as identity-a.e.): gas-only
        # ymix denominator, relax kernels on the post-step state, partial
        # balance (non-gas species bypass the rebalance), layer-0 pins.
        gas = t.gas_mask
        if gas is None:
            ymix = sol / jnp.sum(sol, axis=1, keepdims=True)
        else:
            gsum = jnp.sum(jnp.where(gas[None, :], sol, 0.0), axis=1, keepdims=True)
            ymix = jnp.where(gsum > 0, sol / gsum, 0.0)
        y2 = sol
        if t.conden_static is not None:
            y2, ymix = apply_h2o_relax_jax(y2, ymix, dt64, t.conden_static)
            y2, ymix = apply_nh3_relax_jax(y2, ymix, dt64, t.conden_static)
        balanced = M_c * ymix
        if t.hydro_partial and gas is not None:
            balanced = jnp.where(gas[None, :], balanced, y2)
        if t.bot_idx is not None:
            balanced = balanced.at[0, t.bot_idx].set(t.bot_val)
        return balanced

    def step_fn(y, k_use, atm_use):
        # fix_species regime: pin inside the step (row/col zeroing) then
        # overwrite with the pinned values, so pinned rows are constants of
        # the map (identity rows of I - dG/dy), not singular pass-throughs.
        if t is not None and t.fix_mask is not None:
            sol, _ = jax_ros2_step(y, k_use, dt64, atm_use, net, fix_mask=t.fix_mask)
            return jnp.where(t.fix_mask, t.fix_y, sol)
        sol, _ = jax_ros2_step(y, k_use, dt64, atm_use, net)
        return sol

    # With photo_recompute_k the photolysis rows are rebuilt from y each
    # application (the runner's own two-stream RT), so the y-VJP carries
    # dJ/dy; update_conden_rates does the same for the conden rows
    # (rate ~ y - y_sat). Folding the recomputes in from the INCOMING y is
    # exact for the fixed-point adjoint (incoming == post-step at the fixed
    # point; the one-step lag drops out of the coupled-system algebra).
    def body_map(y):
        k_use = photo_recompute_k(y) if photo_recompute_k is not None else k_arr
        if t is not None and t.conden_static is not None:
            k_use = update_conden_rates(k_use, y, t.conden_static)
        return apply_post_map(step_fn(y, k_use, atm))

    # k-linearization at y_star: k stays free (photo/conden rows NOT
    # recomputed -- a row entry means "perturb this rate"; the y-operator
    # above carries the state feedback).
    def body_map_k(k):
        return apply_post_map(step_fn(y_star, k, atm))

    return apply_post_map, body_map, body_map_k, step_fn


def _safe_inv_y(y_star: jnp.ndarray) -> jnp.ndarray:
    """Elementwise 1/y* with exact zeros mapped to 0, not inf.

    GUARDRAIL: closed columns clip trace species to exactly 0.0; unmasked,
    the 1/y* log-scaling would poison the whole adjoint with NaN. A zeroed
    species becomes an identity row of the log operator (cotangent left
    untouched), the correct leading-order behavior.
    """
    pos = y_star > 0.0
    return jnp.where(pos, 1.0 / jnp.where(pos, y_star, 1.0), 0.0)


def _conserved_null_basis(
    y_star: jnp.ndarray, compo_array: jnp.ndarray, dz: jnp.ndarray
) -> jnp.ndarray:
    """Orthonormal basis Q of the log-space conserved-mass null space.

    For each tracked element e, `sum_i compo[i,e]*dz*y_i` is conserved; its
    log-space gradient is `c_e[z,i] = compo[i,e]*dz[z]*y*[z,i]`. Stack the
    active elements, unit-normalize, QR -> the deflation projector basis.
    A (near-)zero `|R_jj|` raises: unpivoted QR maps a rank-deficient stack
    to arbitrary directions outside span(C), silently corrupting the deflation.

    Shapes: y_star (nz, ni), compo_array (ni, n_atoms), dz (nz,) -> Q (nz*ni, n_e).
    Host-side, one-shot, off the hot path.
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

    Host-side scipy because JAX exposes no LGMRES (its restarted `gmres`
    oscillates on this indefinite operator); each matvec is one host<->device
    round trip, run once post-convergence, off the hot path.

    info == 0 stops early; info > 0 continues into the next warm-start cycle;
    info < 0 (breakdown/illegal input) raises. Returns the BEST-residual
    iterate across cycles, not the last: the warm-restart trajectory is not
    monotone on this operator (costs one extra matvec per cycle).
    """
    import scipy.sparse.linalg as spla

    n = bvec.shape[0]
    A_op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    x = np.zeros(n)
    x_best = x
    r_best = np.inf
    bnorm = max(float(np.linalg.norm(bvec)), _UNDERFLOW_DENOM)
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
            # Converged; cycles exist to continue an unconverged solve.
            break
    return x_best


def _topk_ensemble_spread(g_stack: np.ndarray) -> float:
    """Twin disagreement on the components one would report: max over the
    TOP-10 by |mean| of (max-min)/|mean|. Weaker components carry larger
    relative bounce and would alarm on noise that never enters a ranking."""
    if g_stack.shape[0] <= 1:
        return 0.0
    g_mean = g_stack.mean(axis=0)
    order = np.argsort(np.abs(g_mean))[::-1]
    top = order[: min(10, order.size)]
    top = top[np.abs(g_mean[top]) > 0.0]
    if not top.size:
        return 0.0
    width = g_stack[:, top].max(axis=0) - g_stack[:, top].min(axis=0)
    return float(np.max(width / np.abs(g_mean[top])))


def _matching_host_network(net, network=None):
    """Return host network metadata matching `net.nr`, when available."""
    if net is None:
        return network
    if network is not None:
        return network
    try:
        from . import chem_funs

        cand = getattr(chem_funs, "_NETWORK", None)
        if cand is not None and cand.nr == getattr(net, "nr", None):
            return cand
    except Exception:
        pass
    return None


def _active_photolysis_rows(k_arr, net, network=None) -> bool:
    """Whether `k_arr` contains nonzero photodissociation rows."""
    network = _matching_host_network(net, network=network)
    if network is None:
        return False
    photo_rows = np.asarray(network.is_photo, dtype=bool)
    if not photo_rows.any():
        return False
    return bool(np.any(np.asarray(k_arr)[photo_rows] != 0.0))


def _resolve_photo_recompute_k(
    photo_recompute_k: PhotoRecomputeArg,
    k_arr,
    net,
    solver_map: str,
    *,
    runner_photo_static=None,
    converged_state=None,
    integ=None,
    network=None,
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Resolve the public photo-feedback default for sensitivity calls.

    "auto" builds the recompute callable from the finished runner when
    possible; if active photolysis rows are present but the default renorm
    path lacks that context, refuse rather than silently return the
    lower-accuracy frozen-photolysis adjoint. `None` stays the explicit
    legacy/frozen choice.
    """
    if photo_recompute_k is None or callable(photo_recompute_k):
        return photo_recompute_k
    if photo_recompute_k != PHOTO_RECOMPUTE_AUTO:
        raise ValueError(
            "photo_recompute_k must be a callable, None, or "
            f"{PHOTO_RECOMPUTE_AUTO!r}; got {photo_recompute_k!r}."
        )

    if runner_photo_static is None and integ is not None:
        runner_photo_static = getattr(integ, "_photo_static", None)
    if runner_photo_static is not None and converged_state is not None:
        return make_photo_recompute_k(runner_photo_static, converged_state)

    if not _active_photolysis_rows(k_arr, net, network=network):
        return None
    if solver_map != SOLVER_MAP_DEFAULT:
        return None
    raise ValueError(
        "photo_recompute_k='auto' is the default on active photochemistry "
        "columns, but it needs the finished runner's photolysis state. Pass "
        "`runner_photo_static=integ._photo_static` and "
        "`converged_state=final_state` (or `integ=integ, "
        "converged_state=final_state`) so the adjoint carries dJ/dy. To "
        "explicitly reproduce the legacy frozen-photolysis result, pass "
        "`photo_recompute_k=None`."
    )


def _guard_unmodeled_processes(
    y_star, k_arr, net, body_terms, photo_recompute_k, network=None, species=None
):
    """Fingerprint processes the body map would silently mistreat; raise/warn.

    `NetworkArrays` carries no photo/ion/conden row masks, so the checks use
    the import-locked parsed network / species list when their sizes match. A
    non-matching custom network skips the fingerprints with a RuntimeWarning
    (skipped != passed). `network`/`species` exist for tests.

    Raises on nonzero ion rows (charge balance is in no body map) and on
    condensation fingerprints (nonzero conden rate rows, or a populated
    condensate) without conden/fix-species body terms. Warns on active
    photolysis rows without `photo_recompute_k` (frozen dJ/dy).
    """
    if network is None or species is None:
        try:
            from . import chem_funs

            cand = _matching_host_network(net)
            if network is None and cand is not None:
                network = cand
            if species is None and cand is not None:
                if len(chem_funs.spec_list) == y_star.shape[1]:
                    species = list(chem_funs.spec_list)
        except Exception:
            pass

    if network is None or species is None:
        skipped = [
            name
            for name, missing in (
                ("photo/ion/conden rate-row fingerprints", network is None),
                ("condensate-species fingerprint", species is None),
            )
            if missing
        ]
        # Cannot fingerprint without the network, but network-independent
        # evidence (no conden body terms + an all-zero species column) is
        # enough to refuse a state that is not provably benign;
        # `network=`/`species=` gets the real check.
        needs_conden_terms = body_terms is None or body_terms.conden_static is None
        y_np_probe = np.asarray(y_star)
        # An all-zero species column is the condensate/pinned signature; a
        # fully-positive column set is the benign gas-only signature.
        has_zeroed_columns = bool(np.any(np.all(y_np_probe <= 0.0, axis=0)))
        if needs_conden_terms and has_zeroed_columns:
            raise ValueError(
                "adjoint fingerprint guard cannot resolve the host network for "
                f"this k_arr ({' and '.join(skipped)}), AND this state is not "
                "provably benign: y_star has at least one all-zero species "
                "column (the signature of a condensate or pinned species) while "
                "no condensation body terms were supplied. The ion/condensation "
                "refusals therefore could not run, and a condensation-coupled "
                "sensitivity would be silently wrong.\n"
                "Pass `network=<parsed network>` and `species=<species list>` "
                "for this run so the fingerprints can be evaluated, or run "
                "`audit_adjoint_scope(...)` with the run's cfg to check "
                "explicitly. To proceed on a state you know is gas-only, supply "
                "the matching `network=`/`species=` rather than relying on the "
                "import-locked default."
            )
        warnings.warn(
            "adjoint fingerprint guard could not resolve the import-locked "
            f"host network/species for this k_arr (custom network?): {' and '.join(skipped)} "
            "SKIPPED, not passed. Run audit_adjoint_scope(...) with the run's "
            "cfg to check for dropped processes explicitly.",
            RuntimeWarning,
            stacklevel=3,
        )

    k_np = np.asarray(k_arr)
    terms_conden = body_terms is not None and body_terms.conden_static is not None
    terms_pins = body_terms is not None and body_terms.fix_mask is not None

    conden_rate_active = False
    if network is not None:
        ion_rows = np.asarray(network.is_ion, dtype=bool)
        if ion_rows.any() and bool(np.any(k_np[ion_rows] != 0.0)):
            raise NotImplementedError(
                "ion rows are active in k_arr: the runner pins the electron "
                "rows inside both Ros2 stages and applies a post-step charge "
                "balance, neither of which is in the adjoint body map, so "
                "ion-coupled sensitivities would be silently wrong. Ion "
                "columns are not supported by the steady-state adjoint — use "
                "forward-mode."
            )
        conden_rows = np.asarray(network.is_conden, dtype=bool)
        conden_rate_active = conden_rows.any() and bool(
            np.any(k_np[conden_rows] != 0.0)
        )

    condensate_active = False
    if species is not None:
        y_np = np.asarray(y_star)
        for i, sp in enumerate(species):
            # Condensed-phase suffixes across shipped networks: `_l_s`
            # (H2O/NH3/S2/S8), `_l` (H2SO4), `_s` (C_s); testing only
            # `_l_s` would silently miss H2SO4_l and C_s.
            if sp.endswith(("_l_s", "_l", "_s")) and float(y_np[:, i].max()) > 0.0:
                condensate_active = True
                break

    if (conden_rate_active or condensate_active) and not (terms_conden or terms_pins):
        raise ValueError(
            "this state was converged with condensation active (nonzero "
            "conden rate rows and/or populated condensate species), but the "
            "adjoint body map carries no condensation terms — the gradient "
            "would be silently wrong on conden-coupled rows/layers. Pass "
            "body_terms from make_body_terms(integ, converged_state, "
            "atm_static) (conden rate-recompute + relax kernels + partial "
            "balance in the conden window, or the fix_species pins after "
            "it), or use forward-mode."
        )

    if network is not None and photo_recompute_k is None:
        photo_rows = np.asarray(network.is_photo, dtype=bool)
        if photo_rows.any() and bool(np.any(k_np[photo_rows] != 0.0)):
            warnings.warn(
                "photolysis rows are active in k_arr but photo_recompute_k "
                "was not passed: dJ/dy is omitted and photo-coupled "
                "sensitivities are leading-order only (~11% measured on W39b "
                "OH+H2). Build it with make_photo_recompute_k(...).",
                stacklevel=3,
            )


def _adjoint_solve_core(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    y_star: jnp.ndarray,
    body_map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    compo_array: jnp.ndarray,
    dz: jnp.ndarray,
    *,
    lgmres_inner_m: int,
    lgmres_outer_k: int,
    lgmres_maxiter: int,
    lgmres_cycles: int,
    rtol: float,
    n_solves: int,
):
    """Shared deflated-LGMRES adjoint solve on the chosen body map.

    Returns `(lams, resids, fp_err, null_quality, n_matvec, n_null)`: per-twin
    unscaled cotangents `lambda = z ./ y*` (contract against any parameter map
    as `dL/dp = lambda^T dG/dp`), per-twin relative residuals, and shared
    diagnostics. Both sensitivity entry points share this solve; only the
    final contraction differs.
    """
    nz, ni = y_star.shape
    inv_y = _safe_inv_y(y_star)
    body_map = jax.jit(body_map_raw)

    fp_err = float(
        jnp.max(jnp.abs(body_map(y_star) - y_star))
        / jnp.maximum(jnp.max(jnp.abs(y_star)), _UNDERFLOW_DENOM)
    )
    _, vjp_Gy = jax.vjp(body_map, y_star)

    def a_eta(z):  # (I - dG/deta)^T in log-abundance coordinates
        return z - y_star * vjp_Gy(z * inv_y)[0]

    # Jit once, shared by the null-quality diagnostic and the LGMRES matvec,
    # so the expensive step-VJP XLA compile is paid exactly once. `proj` stays
    # outside the jit (two small matmuls per matvec, negligible).
    a_eta_j = jax.jit(a_eta)

    # Conserved-mass deflation projector.
    Q = _conserved_null_basis(y_star, compo_array, dz)

    def proj(z):
        zf = z.reshape(-1)
        return (zf - Q @ (Q.T @ zf)).reshape(nz, ni)

    def deflated(z):
        return proj(a_eta_j(proj(z)))

    # How null the deflated directions actually are: max_e ||A_eta^T q_e||
    # (unit-norm columns) relative to the operator's action on a fixed-seed
    # random unit direction. O(1) means a deflated direction is NOT null
    # (e.g. open boundary fluxes) and the deflation is corrupting the solve.
    null_defect = max(
        float(jnp.linalg.norm(a_eta_j(Q[:, e].reshape(nz, ni))))
        for e in range(Q.shape[1])
    )
    rng = np.random.default_rng(0)  # fixed seed: the diagnostic is deterministic
    r = rng.normal(size=(nz, ni))
    r /= np.linalg.norm(r)
    op_scale = float(jnp.linalg.norm(a_eta_j(jnp.asarray(r))))
    null_quality = null_defect / max(op_scale, _UNDERFLOW_DENOM)

    # RHS: log-space cotangent of the loss, deflated.
    v = jax.grad(loss_fn)(y_star)
    b = proj(y_star * v)
    bvec = np.asarray(b).ravel()

    n_matvec = [0]

    def matvec(x):
        n_matvec[0] += 1
        return np.asarray(deflated(jnp.asarray(x.reshape(nz, ni)))).ravel()

    # Twin ensemble: a semi-converged Krylov endpoint is bit-level
    # trajectory-sensitive, so solving against seeded ulp-perturbed copies of
    # b turns that lottery into a measurable mean + spread. Twin 0 is the
    # unperturbed b.
    twin_rng = np.random.default_rng(0)
    lams = []
    resids = []
    for i in range(max(1, int(n_solves))):
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
                / max(float(np.linalg.norm(b_i)), _UNDERFLOW_DENOM)
            )
        )
        lams.append(z * inv_y)  # unscaled cotangent lambda

    return lams, resids, fp_err, null_quality, int(n_matvec[0]), int(Q.shape[1])


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
    solver_map: str = SOLVER_MAP_DEFAULT,
    photo_recompute_k: PhotoRecomputeArg = PHOTO_RECOMPUTE_AUTO,
    runner_photo_static=None,
    converged_state=None,
    integ=None,
    body_terms: BodyTerms | None = None,
    lgmres_inner_m: int = LGMRES_INNER_M,
    lgmres_outer_k: int = LGMRES_OUTER_K,
    lgmres_maxiter: int = LGMRES_MAXITER,
    lgmres_cycles: int = LGMRES_CYCLES,
    rtol: float = LGMRES_RTOL,
    n_solves: int = N_SOLVES_DEFAULT,
    return_info: bool = False,
):
    """Reverse-mode `dL/d(ln k_r)` at a converged steady state.

    Sensitivity of a scalar loss of the converged composition to every
    reaction-rate constant -- the reaction-ranking use case. The result is the
    MEAN over `n_solves` adjoint solves with deterministic ulp-perturbed
    right-hand sides; `info["ensemble_spread"]` (twin disagreement) is the
    honest magnitude error bar.

    Accuracy is limited by the linear solve AND by the finite-tolerance
    mismatch between the exact fixed point and the state the forward run
    stopped at. The usable `body_dt` window is column-dependent: for
    publication-grade magnitudes loop over `BODY_MAP_DT_CANDIDATES` and
    inspect `info["resid_median"]` / `info["ensemble_spread"]` per candidate. For `dL/dT`, `dL/dKzz`, ... use
    `steady_state_input_sensitivity`; forward-mode `jvp` is exact for a
    handful of directions.

    Parameters
    ----------
    loss_fn
        `loss_fn(y) -> scalar` on the full `(nz, ni)` number-density state,
        e.g. `lambda y: jnp.log10(y[L, so2] / y[L].sum())`.
    y_star : (nz, ni)
        Converged state (number density, cm^-3) -- a tight fixed point of the
        CHOSEN body map (`info["fp_err"]`; ~1e-9 for the renorm default). Do
        NOT iterate the renorm map to tighten `fp_err`: it trades the deflation
        basis for the fixed point and degrades `info["null_quality"]`.
        Clip, charge balance, condensation, fix-species and bottom pins are in
        NEITHER map: run `audit_adjoint_scope(...)` first (its per-cell defect
        scan also catches what the global max-norm `fp_err` masks).
    k_arr : (nr+1, nz)
        Converged rate-constant table.
    atm : AtmStatic
        Atmosphere with the converged refresh fields (g, dzi, Hpi, ...), as
        fed to the runner's body map.
    net : NetworkArrays
        Active network.
    compo_array : (ni, n_atoms)
        Per-species atom counts; pass `composition.compo_array[:ni]`.
    dz : (nz,)
        Layer thickness (cm). Required: `AtmStatic` carries only the interface
        average `dzi`, which is not invertible to `dz`. Use `AtmInputs.dz`.
    body_dt
        Adjoint-only probe step in s (see `BODY_MAP_DT`); scan on a new column.
    solver_map
        `"renorm"` (default) linearizes the hydrostatic-renormalized map the
        runner actually iterates (HD189 CH4 6.6% -> 0.7%); `"bare"` only
        reproduces pre-2026-07 behavior. See `SOLVER_MAP_DEFAULT`.
    photo_recompute_k
        `"auto"` (default) builds a `k(y) -> k_arr` recompute from the runner
        context on photo-on columns so the state operator carries `dJ/dy`
        (WASP-39b OH+H2 11% -> 0.2%); a callable overrides the builder; `None`
        only for photo-off columns or the frozen-photolysis legacy result
        (~11%). Costs an RT solve per Krylov matvec.
    runner_photo_static, converged_state, integ
        Context for `"auto"`: pass `runner_photo_static=integ._photo_static,
        converged_state=final_state` or `integ=integ,
        converged_state=final_state` (`converged_state` is the runner's
        `JaxIntegState`; the recompute reuses the runner's photo branch).
    body_terms
        Optional `BodyTerms` (condensation composite, fix_species pins,
        layer-0 boundary pins). REQUIRED when the state converged with
        condensation active; build with `make_body_terms(integ,
        converged_state, atm_static)`, which also returns the spliced `atm`.
        Requires `solver_map="renorm"`.
    lgmres_inner_m, lgmres_outer_k, lgmres_maxiter, lgmres_cycles, rtol
        LGMRES knobs (see the module constants).
    n_solves
        Twin-ensemble size (`_TWIN_PERTURB`, seeded); `n_solves=1` reproduces
        the old single-solve behavior. Each extra solve costs one LGMRES
        budget; the operator compile is shared.
    return_info
        If True, also return a diagnostics dict.

    Returns
    -------
    dL_dlnk : (nr+1,)
        Ensemble-mean `dL/d(ln k_r)` per directional rate-table row (index 0
        is the unused 1-based pad). For a physical detailed-balance
        perturbation of a reversible reaction use the pair sum
        `g[fwd] + g[rev]`.
    info : dict, optional
        `fp_err`, `null_quality` (O(1) = a deflated direction is not null and
        the deflation is suspect), `resid`/`resid_median`/`resids`,
        `ensemble_spread` (top-10 (max-min)/|mean|; 0.0 when n_solves=1),
        `pair_antisym` (diagnostic only; see the module docstring),
        `n_matvec`, `n_null`, `n_solves`, `body_dt`, and the mode flags.
    """
    _check_body_dt(body_dt)
    photo_recompute_k = _resolve_photo_recompute_k(
        photo_recompute_k,
        k_arr,
        net,
        solver_map,
        runner_photo_static=runner_photo_static,
        converged_state=converged_state,
        integ=integ,
    )
    _guard_unmodeled_processes(y_star, k_arr, net, body_terms, photo_recompute_k)
    n_solves = max(1, int(n_solves))

    # Condensation active: the reaction gradient is CONDITIONAL -- the body
    # map holds the captured reservoir / saturation tables fixed, so this is
    # dL/d ln k AT the frozen reservoir, excluding how the rate set it. Rates
    # do not move the saturation curve directly, so label it, do not forbid
    # it. See README.md (Differentiability, F2).
    _conden_pinned = body_terms is not None and body_terms.fix_mask is not None
    _conden_in_window = body_terms is not None and body_terms.conden_static is not None
    if _conden_pinned or _conden_in_window:
        warnings.warn(
            "steady_state_reaction_sensitivity: condensation is active at this "
            "converged state, so the returned dL/d ln k is CONDITIONAL on the "
            + (
                "frozen (pinned) condensate reservoir"
                if _conden_pinned
                else "frozen saturation tables"
            )
            + "; it does not include how the rate changes what condenses. It is "
            "a valid conditional ranking, not the total rate sensitivity "
            "(info['conditional_on_fixed_reservoir']). See "
            "README.md (Differentiability).",
            stacklevel=2,
        )

    # One-step body map and its y-VJP (the transposed solver-map operator);
    # dJ/dy rides through photo_recompute_k, the conden/relax/pin/balance
    # terms through body_terms.
    _, _body_map_raw, _body_map_k_raw, _ = _make_body_map(
        y_star, k_arr, atm, net, body_dt, solver_map, photo_recompute_k, body_terms
    )

    lams, resids, fp_err, null_quality, n_matvec, n_null = _adjoint_solve_core(
        loss_fn,
        y_star,
        _body_map_raw,
        compo_array,
        dz,
        lgmres_inner_m=lgmres_inner_m,
        lgmres_outer_k=lgmres_outer_k,
        lgmres_maxiter=lgmres_maxiter,
        lgmres_cycles=lgmres_cycles,
        rtol=rtol,
        n_solves=n_solves,
    )

    # Reaction cotangent per twin: dL/d(ln k_r) = (k .* G_k^T lambda)_r.
    _, vjp_Gk = jax.vjp(_body_map_k_raw, k_arr)
    grads = [np.asarray((k_arr * vjp_Gk(lam)[0]).sum(axis=1)) for lam in lams]

    g_stack = np.stack(grads, axis=0)  # (n_solves, nr+1)
    dL_dlnk = jnp.asarray(g_stack.mean(axis=0))
    resid = max(resids)
    resid_median = float(np.median(resids))

    # Twin-to-twin disagreement on the reactions one would actually report.
    ensemble_spread = _topk_ensemble_spread(g_stack)

    # Forward/reverse pair antisymmetry over the top-10: near partial
    # equilibrium dL/dln k_f ~ -dL/dln k_r, so |g_f+g_r|/max(|g_f|,|g_r|)
    # should be small. Diagnostic only (see module docstring); photolysis /
    # irreversible rows (all-zero reverse k) are skipped.
    g_mean_full = np.asarray(dL_dlnk)
    k_np = np.asarray(k_arr)
    pair_antisym = 0.0
    for r in np.argsort(np.abs(g_mean_full))[::-1][:10]:
        r = int(r)
        f = r if r % 2 == 1 else r - 1
        rev = f + 1
        if f < 1 or rev >= g_mean_full.shape[0]:
            continue
        if not (np.any(k_np[f] != 0.0) and np.any(k_np[rev] != 0.0)):
            continue  # photo/conden/irreversible slot, no genuine pair
        denom = max(abs(g_mean_full[f]), abs(g_mean_full[rev]), _UNDERFLOW_DENOM)
        pair_antisym = max(pair_antisym, abs(g_mean_full[f] + g_mean_full[rev]) / denom)

    # Default-on diagnostics: a poorly-converged solve still returns a
    # finite-looking gradient. Warn on the ensemble MEDIAN residual (robust to
    # one wandering twin); info["resid"] still reports the max.
    _warn_poor_convergence(resid_median, fp_err, ensemble_spread, null_quality)

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
        "resid_median": resid_median,
        "resids": resids,
        "ensemble_spread": ensemble_spread,
        "pair_antisym": pair_antisym,
        "n_matvec": n_matvec,
        "n_null": n_null,
        "n_solves": n_solves,
        "body_dt": float(body_dt),
        "solver_map": solver_map,
        "photo_feedback": photo_recompute_k is not None,
        "body_terms": body_terms is not None,
        "condensation_active": bool(_conden_pinned or _conden_in_window),
        "conditional_on_fixed_reservoir": bool(_conden_pinned),
        "includes_condensation_history": not bool(_conden_pinned or _conden_in_window),
    }
    return dL_dlnk, info


def steady_state_input_sensitivity(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    p0,
    rebuild: Callable,
    *,
    compo_array: jnp.ndarray,
    dz: jnp.ndarray,
    body_dt: float = BODY_MAP_DT,
    solver_map: str = SOLVER_MAP_DEFAULT,
    photo_recompute_k: PhotoRecomputeArg = PHOTO_RECOMPUTE_AUTO,
    runner_photo_static=None,
    converged_state=None,
    integ=None,
    body_terms: BodyTerms | None = None,
    lgmres_inner_m: int = LGMRES_INNER_M,
    lgmres_outer_k: int = LGMRES_OUTER_K,
    lgmres_maxiter: int = LGMRES_MAXITER,
    lgmres_cycles: int = LGMRES_CYCLES,
    rtol: float = LGMRES_RTOL,
    n_solves: int = N_SOLVES_DEFAULT,
    return_info: bool = False,
    allow_frozen_condensation_input_grad: bool = False,
):
    """Reverse-mode `dL/dp` at a converged steady state, for an arbitrary
    physical input pytree `p` (a (nz,) temperature profile, ln Kzz, ...).

    One adjoint solve (identical to `steady_state_reaction_sensitivity`'s)
    yields the cotangent `lambda`; the input gradient is one VJP of the
    parameter map at the fixed point:

        dL/dp = lambda^T dG/dp ,   G(p) = post(ros2_step(y*, k(p), dt, atm(p)))

    so ALL components of `p` cost one solve plus one VJP. The renormalization
    uses `atm(p).M`, so an input that moves the total density (temperature:
    M = pco/(kb T)) is differentiated through the rebalance too.

    Parameters (beyond the shared ones)
    -----------------------------------
    p0
        The input value the state was converged at (array or pytree).
    rebuild
        `rebuild(p) -> (k_arr_p, atm_p)`: a JAX-differentiable rebuild of the
        FULL rate table and `AtmStatic` at input `p`. Must reproduce the
        converged inputs at `p0` (warn above `_REBUILD_CONSISTENCY_WARN`,
        refuse above `_REBUILD_CONSISTENCY_ERR`); in particular non-thermal
        rows (photolysis J, conden) must be spliced in FROZEN from the
        converged `k_arr`. Example, a temperature profile on a photo-off
        column::

            pco = atm.M * kb * atm.Tco          # fixed hydrostatic pressures

            def rebuild(T):
                M = pco / (kb * T)
                k = rates_jax.build_rate_array(network, T, M, nasa9, remove_list)
                # photo-on columns: k = jnp.where(photo_rows[:, None], k_arr, k)
                return k, atm._replace(Tco=T, Ti=0.5 * (T[:-1] + T[1:]), M=M)

    Returns
    -------
    dL_dp
        Same pytree structure as `p0`; ensemble mean over `n_solves` twins.
    info : dict, optional
        The shared adjoint diagnostics plus `rebuild_consistency` (worst
        relative mismatch of `rebuild(p0)` per field).

    Scope and accuracy notes
    ------------------------
    * Chemistry path only: `(dL/dy*) . (dy*/dp)`. A loss with a DIRECT `p`
      dependence (e.g. T in the RT opacities of a spectrum chi-square) needs
      that term added separately (`jax.grad` w.r.t. p at fixed `y_star`).
    * Condensation is refused by default: the saturation tables are frozen in
      dG/dp and, post-pin, the captured reservoir is held fixed, so the result
      is O(1)-unreliable vs FD (0.91 relative). Opt in with
      `allow_frozen_condensation_input_grad=True` only for the known
      leading-order number. See README.md (Differentiability).
    * Also frozen by design (p-derivative omitted): the photolysis
      T-cross-section interpolation and the atm-refresh geometry cascade
      (dz/Hp/g, second-order; rebuild what you need on-graph in `atm_p`).
    * Accuracy class matches the reaction sensitivities. The deflation was
      PROVEN exact only for atom-conserving rate knobs; spot-validate a new
      input type against a forward-mode `jvp` in one or two directions before
      production use (`d/dT` validated on HD189, see `jax_paper/scripts/`).
    """
    # Condensation is NOT differentiable-through for input gradients: sat
    # tables frozen (d(sat)/dT dropped) and, post-pin, the captured reservoir
    # held fixed; the pinned-species tangent disagrees with re-converged FD at
    # O(1) (0.91 relative). Refuse by default -- the same contract Fisher /
    # retrieval follow project-wide. See README.md (Differentiability, F1).
    _conden_in_window = body_terms is not None and body_terms.conden_static is not None
    _conden_pinned = body_terms is not None and body_terms.fix_mask is not None
    if _conden_in_window or _conden_pinned:
        _regime = "post-pin fix_species" if _conden_pinned else "in-window"
        if not allow_frozen_condensation_input_grad:
            raise ValueError(
                "steady_state_input_sensitivity: condensation is active at this "
                f"converged state ({_regime} regime); an input (T/Kzz/...) "
                "gradient through condensation is NOT reliably differentiable. "
                "The saturation tables are frozen in dG/dp (d(sat)/dT dropped) "
                "and, once fix_species has pinned, the captured condensate "
                "reservoir is held fixed, so the parameter's effect on what "
                "condensed is entirely absent. The pinned-species tangent "
                "disagrees with re-converged finite differences at O(1) (0.91 "
                "relative; tests/test_condensation_guards.py) -- the same reason "
                "Fisher / retrieval-inference through condensation is refused "
                "project-wide (README.md, Differentiability). Use "
                "forward-mode jvp on a single switch-free direction and validate "
                "it against FD, or disable condensation. Set "
                "allow_frozen_condensation_input_grad=True ONLY to obtain the "
                "known leading-order-only number knowingly."
            )
        warnings.warn(
            "steady_state_input_sensitivity: condensation is active "
            f"({_regime}); allow_frozen_condensation_input_grad=True, so "
            "returning the LEADING-ORDER-ONLY gradient (frozen saturation "
            "tables and, post-pin, a frozen reservoir; O(1)-unreliable vs FD). "
            "Forward-mode on a switch-free direction is the trusted route.",
            stacklevel=2,
        )
    _check_body_dt(body_dt)
    photo_recompute_k = _resolve_photo_recompute_k(
        photo_recompute_k,
        k_arr,
        net,
        solver_map,
        runner_photo_static=runner_photo_static,
        converged_state=converged_state,
        integ=integ,
    )
    _guard_unmodeled_processes(y_star, k_arr, net, body_terms, photo_recompute_k)
    n_solves = max(1, int(n_solves))

    # rebuild(p0) must reproduce the map the state converged under.
    k0, atm0 = rebuild(p0)

    def _worst_rel(a, b) -> float:
        a_np = np.asarray(a, dtype=np.float64)
        b_np = np.asarray(b, dtype=np.float64)
        if a_np.shape != b_np.shape:
            return float("inf")
        denom = np.maximum(np.maximum(np.abs(a_np), np.abs(b_np)), _UNDERFLOW_DENOM)
        return float(np.max(np.abs(a_np - b_np) / denom))

    consistency = {"k_arr": _worst_rel(k0, k_arr)}
    for name, ref in atm._asdict().items():
        cand = getattr(atm0, name)
        if ref is None or isinstance(ref, bool):
            consistency[name] = 0.0 if cand == ref else float("inf")
        else:
            consistency[name] = _worst_rel(cand, ref)
    worst_name = max(consistency, key=consistency.get)
    worst = consistency[worst_name]
    if worst > _REBUILD_CONSISTENCY_ERR:
        raise ValueError(
            f"rebuild(p0) does not reproduce the converged step inputs "
            f"(worst mismatch {worst:.2e} on {worst_name!r}); the adjoint "
            "would be linearized against a different map and the gradient "
            "silently wrong. Common causes: missing frozen photolysis/conden "
            "row splice into k(p), lowT-cap or remove_list mismatch, wrong "
            "pressure grid or un-spliced refresh geometry in atm(p)."
        )
    if worst > _REBUILD_CONSISTENCY_WARN:
        warnings.warn(
            f"rebuild(p0) reproduces the converged inputs only to {worst:.2e} "
            f"(field {worst_name!r}); the gradient carries that inconsistency.",
            stacklevel=2,
        )

    # conden_static feeds G_p's conden-row recompute; the refusal/warning at
    # the top already covered both conden regimes.
    conden_static = body_terms.conden_static if body_terms is not None else None

    apply_post, body_map_raw, _, step_fn = _make_body_map(
        y_star, k_arr, atm, net, body_dt, solver_map, photo_recompute_k, body_terms
    )

    lams, resids, fp_err, null_quality, n_matvec, n_null = _adjoint_solve_core(
        loss_fn,
        y_star,
        body_map_raw,
        compo_array,
        dz,
        lgmres_inner_m=lgmres_inner_m,
        lgmres_outer_k=lgmres_outer_k,
        lgmres_maxiter=lgmres_maxiter,
        lgmres_cycles=lgmres_cycles,
        rtol=rtol,
        n_solves=n_solves,
    )

    # Parameter map at the fixed point: conden rows recomputed at y_star
    # (y-part only; sat tables frozen), photolysis rows ride k(p) frozen.
    def G_p(p):
        k_p, atm_p = rebuild(p)
        k_use = (
            update_conden_rates(k_p, y_star, conden_static)
            if conden_static is not None
            else k_p
        )
        sol = step_fn(y_star, k_use, atm_p)
        return apply_post(sol, M_col=atm_p.M[:, None])

    _, pullback = jax.vjp(G_p, p0)
    twin_grads = [pullback(lam)[0] for lam in lams]
    dL_dp = jax.tree_util.tree_map(
        lambda *leaves: jnp.mean(jnp.stack(leaves), axis=0), *twin_grads
    )

    flat = [
        np.concatenate(
            [np.ravel(np.asarray(leaf)) for leaf in jax.tree_util.tree_leaves(g)]
        )
        for g in twin_grads
    ]
    ensemble_spread = _topk_ensemble_spread(np.stack(flat, axis=0))
    resid = max(resids)
    resid_median = float(np.median(resids))

    _warn_poor_convergence(resid_median, fp_err, ensemble_spread, null_quality)

    finite = all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(dL_dp)
    )
    if not finite:
        raise ValueError(
            "Input sensitivity is non-finite. Common causes: a body_dt in "
            "the danger zone, a y_star that is not a fixed point of the body "
            f"map (fp_err={fp_err:.2e}), or a non-differentiable rebuild."
        )

    if not return_info:
        return dL_dp
    info = {
        "fp_err": fp_err,
        "null_quality": null_quality,
        "resid": resid,
        "resid_median": resid_median,
        "resids": resids,
        "ensemble_spread": ensemble_spread,
        "n_matvec": n_matvec,
        "n_null": n_null,
        "n_solves": n_solves,
        "body_dt": float(body_dt),
        "solver_map": solver_map,
        "photo_feedback": photo_recompute_k is not None,
        "body_terms": body_terms is not None,
        "rebuild_consistency": consistency,
    }
    return dL_dp, info


def make_photo_recompute_k(runner_photo_static, converged_state):
    """Build a differentiable `k(y)` that rebuilds the photolysis rows from y.

    Reuses the runner's own in-loop photo branch so the recompute is
    bit-identical to the forward model's photolysis (optical depth ->
    two-stream RT -> J-rates -> photolysis rows of `k_arr`). The RT is
    `lax.scan`-based, hence reverse-mode differentiable, so passing the result
    as `photo_recompute_k` makes the state operator carry `dJ/dy`.

    Parameters
    ----------
    runner_photo_static
        The runner's internal `_PhotoStatic` (`OuterLoop._photo_static` after
        `_ensure_runner`), NOT the public `PhotoStaticInputs` pytree.
    converged_state
        A converged `JaxIntegState`; supplies the frozen geometry (`dz`, `pv`
        T-cross sections, prior `dflux_u` -- its second-order self-recursion
        is not differentiated) read alongside the perturbed `y`.

    Returns
    -------
    recompute_k : Callable[[y], k_arr]
        `y (nz, ni) -> k_arr (nr+1, nz)` with photolysis rows = J(y).
    """
    from .outer_loop import _make_photo_branch  # lazy: keep the module acyclic

    photo_branch = _make_photo_branch(runner_photo_static)

    def recompute_k(y):
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        return photo_branch(converged_state._replace(y=y, ymix=ymix)).k_arr

    return recompute_k


def make_body_terms(integ, converged_state, atm_static):
    """Build `(atm_step, BodyTerms)` for the adjoint from a finished runner.

    Replaces the manual geometry splice AND packs every supported per-step
    process the runner's configuration turns on:

    * atm splice: `g`/`dzi`/`Hpi`/`top_flux`/`vs` from the converged carry,
      plus a live `vm` recompute when `use_vm_mol` (the setup-time
      `atm_static.vm` is stale).
    * condensation, in-window regime (fix_species NOT tripped): the runner's
      `CondenStatic` spliced with the converged `ProfileVars`, enabling the
      conden-row recompute + relax kernels. Sat tables are T-baked constants
      (d(sat)/dT missing for T-gradients). Free-running conden states are
      typically PSEUDO-steady; read `audit_adjoint_scope`'s per-cell defect
      for how tight the state actually is.
    * fix_species regime (`fix_species_started`): pin mask + pinned values
      from the carry -- the regime real converged conden runs end in.
    * layer-0 pins: `use_fix_all_bot`, `use_fix_sp_bot`, and the hycean H2/He
      pin once tripped.
    * gas-only / partial hydrostatic balance (condensate-aware form).

    Raises NotImplementedError for `use_ion`; warns when `diff_esc` is active
    (d(phi_esc)/dy stays frozen, leading-order only).

    Parameters
    ----------
    integ
        The `OuterLoop` whose runner produced `converged_state`.
    converged_state
        The converged `JaxIntegState` (the `_runner` output).
    atm_static
        The `AtmStatic` fed to the runner; returned re-spliced.
    """
    st = integ._statics
    if st is None:
        raise ValueError(
            "make_body_terms: the runner has not been built/run — integrate "
            "first (integ(rs) or integ._runner(state, atm_static))."
        )
    cfg = integ._cfg
    s = converged_state

    if bool(st.use_ion):
        raise NotImplementedError(
            "use_ion=True: the electron-row pin and post-step charge balance "
            "are not implemented in the adjoint body map; ion columns are "
            "not supported. Use forward-mode."
        )

    # --- atm splice (the converged refresh geometry + live vm) ---
    atm_step = atm_static._replace(
        g=s.g, dzi=s.dzi, Hpi=s.Hpi, top_flux=s.top_flux, vs=s.vs
    )
    if bool(st.use_vm_mol) and integ._refresh_static is not None:
        from . import atm_refresh as _ar  # lazy: keep the module acyclic

        atm_step = atm_step._replace(
            vm=_ar.recompute_vm_jax(
                s.g,
                s.Hpi,
                s.dzi,
                atm_static.Dzz,
                atm_static.ms,
                atm_static.alpha,
                atm_static.Tco,
                integ._refresh_static.kb,
                integ._refresh_static.Navo,
            )
        )
    # Hybrid molecular diffusion: linearize the SAME operator the runner
    # converged on -- a hybrid run finishes in phase 1 (central diff,
    # hybrid_use_vm==0.0), so drive use_vm_mol from the converged carry
    # exactly as body_fn does (no-op for non-hybrid runs).
    atm_step = atm_step._replace(
        use_vm_mol=jnp.asarray(s.hybrid_use_vm, dtype=jnp.float64)
    )

    if integ._refresh_static is not None:
        n_esc = int(np.asarray(integ._refresh_static.diff_esc_idx).size)
        if n_esc > 0:
            warnings.warn(
                "diff_esc is active: the runner recomputes the diffusion-"
                "limited escape flux from the top-layer densities; the "
                "adjoint freezes top_flux at its converged value, so "
                "d(phi_esc)/dy is dropped and TOA-coupled sensitivities are "
                "leading-order only.",
                stacklevel=2,
            )

    started = bool(np.asarray(s.fix_species_started))

    # --- condensation regimes ---
    conden_static = None
    fix_mask = None
    fix_y = None
    cs = integ._conden_static
    if bool(st.use_conden) and cs is not None:
        if started:
            # Post-window: species pinned via fix_mask inside the step;
            # conden no longer fires.
            fix_mask = s.fix_mask
            fix_y = s.fix_y
        elif float(np.asarray(s.t)) >= float(getattr(cfg, "start_conden_time", 0.0)):
            # In-window: exactly _make_conden_branch's per-lane splice.
            conden_static = cs._replace(
                Dg_per_re=s.pv.c_Dg_per_re,
                sat_n_per_re=s.pv.c_sat_n_per_re,
                h2o_Dg=s.pv.c_h2o_Dg,
                h2o_sat=s.pv.c_h2o_sat,
                nh3_Dg=s.pv.c_nh3_Dg,
                nh3_sat=s.pv.c_nh3_sat,
                nh3_conden_top=s.pv.c_nh3_conden_top,
                n_0=s.pv.n_0,
            )

    gas_mask = cs.gas_indx_mask if cs is not None else None
    if gas_mask is not None and bool(jnp.all(gas_mask)):
        gas_mask = None  # all-gas network: plain renorm denominator

    # --- layer-0 Dirichlet pins, in the runner's application order ---
    ni = s.y.shape[1]
    n0_bot = s.pv.n_0[0]
    idx_parts = []
    val_parts = []
    if bool(st.use_fix_all_bot):
        idx_parts.append(jnp.arange(ni, dtype=jnp.int32))
        val_parts.append(s.pv.bottom_n)
    if bool(st.use_fix_sp_bot) and int(np.asarray(st.fix_sp_bot_idx).size) > 0:
        idx_parts.append(st.fix_sp_bot_idx.astype(jnp.int32))
        val_parts.append(st.fix_sp_bot_mix * n0_bot)
    if bool(st.use_fix_H2He):
        if bool(np.asarray(s.h2he_pinned)):
            idx_parts.append(
                jnp.asarray([int(st.h2_idx), int(st.he_idx)], dtype=jnp.int32)
            )
            val_parts.append(s.h2he_mix * n0_bot)
        else:
            warnings.warn(
                "use_fix_H2He=True but the pin had not tripped at this state "
                "(t <= hycean_pin_time); the body map omits it, consistent "
                "with the converged state.",
                stacklevel=2,
            )
    bot_idx = jnp.concatenate(idx_parts) if idx_parts else None
    bot_val = jnp.concatenate(val_parts) if val_parts else None

    terms = BodyTerms(
        conden_static=conden_static,
        gas_mask=gas_mask,
        hydro_partial=bool(integ._hydro_partial),
        fix_mask=fix_mask,
        fix_y=fix_y,
        bot_idx=bot_idx,
        bot_val=bot_val,
    )
    return atm_step, terms


def _adjoint_scope_findings(
    cfg, final_state=None, photo_recompute_k=None, body_terms=None
) -> list:
    """Static scope checks: which runner processes are OUTSIDE the adjoint's
    body map for this configuration, and what that does to the gradient.

    Severity: "error" = a dropped process is active at the fixed point, so
    sensitivities touching it are physically wrong; "warning" = a known
    leading-order-only omission; "info" = second-order truncation or a
    diagnostic to watch. Pure host logic (unit-testable without arrays);
    `audit_adjoint_scope` adds the empirical per-cell defect scan on top.
    """
    findings: list = []

    def add(code: str, severity: str, message: str) -> None:
        findings.append({"code": code, "severity": severity, "message": message})

    def flag(name: str, default):
        return getattr(cfg, name, default)

    def state_bool(name: str):
        if final_state is None:
            return None
        val = getattr(final_state, name, None)
        return None if val is None else bool(np.asarray(val))

    if bool(flag("use_ion", False)):
        add(
            "ion_charge_balance",
            "error",
            "use_ion=True: the runner pins the electron rows inside both Ros2 "
            "stages and applies the post-step charge balance e = -y.charge "
            "(outer_loop body_fn); neither is in the adjoint body map, so "
            "y_star is not a fixed point on the electron/ion rows and "
            "ion-coupled sensitivities are wrong. Not supported — use "
            "forward-mode for ion columns.",
        )

    terms_conden = body_terms is not None and body_terms.conden_static is not None
    terms_fix = body_terms is not None and body_terms.fix_mask is not None
    terms_bot = body_terms is not None and body_terms.bot_idx is not None

    if bool(flag("use_condense", False)):
        started = state_bool("fix_species_started")
        if terms_conden or (started and terms_fix):
            add(
                "condensation",
                "info",
                "use_condense=True and body_terms carries the matching "
                "regime (conden rate-recompute + relax kernels + partial "
                "balance in the window, or the fix_species pins after it). "
                "Residual limitation: the saturation tables are T-baked "
                "constants, so T-input gradients miss d(sat)/dT.",
            )
        else:
            add(
                "condensation",
                "error",
                "use_condense=True: in-loop condensation is not in the adjoint "
                "body map. (1) update_conden_rates rewrites the conden/evap "
                "k-rows from y every accepted step — a dk/dy feedback exactly "
                "analogous to photolysis dJ/dy — but the adjoint freezes k_arr; "
                "(2) the H2O/NH3 relax kernels move mass outside the Ros2 step; "
                "(3) the runner's hydrostatic rebalance uses a gas-only ymix "
                "denominator and skips non-gas species (hydro_partial), while "
                "the adjoint's renorm rescales ALL species by the total sum. "
                "Pass body_terms from make_body_terms(integ, converged_state, "
                "atm_static).",
            )
        if started and not terms_fix:
            add(
                "fix_species_pins",
                "error",
                "fix_species is ACTIVE at this converged state "
                "(fix_species_started=True): condensables are pinned via "
                "fix_mask inside the Ros2 step, but the adjoint's step has "
                "no fix_mask, so pinned rows are treated as free chemistry. "
                "make_body_terms packs the pins automatically.",
            )
        elif (
            started is None
            and not terms_fix
            and not terms_conden
            and len(list(flag("fix_species", []) or [])) > 0
        ):
            add(
                "fix_species_pins",
                "error",
                "cfg.fix_species is non-empty: after stop_conden_time the "
                "runner pins these species via fix_mask inside the Ros2 "
                "step; the adjoint's step has no fix_mask. Pass the "
                "converged state to check whether the pin is active.",
            )

    bot_pins = (
        bool(flag("use_fix_all_bot", False))
        or len(dict(flag("use_fix_sp_bot", {}) or {})) > 0
    )
    if bot_pins:
        if terms_bot:
            add(
                "bottom_boundary_pins",
                "info",
                "bottom-layer Dirichlet pins are carried by body_terms "
                "(constant rows of the body map).",
            )
        else:
            add(
                "bottom_boundary_pins",
                "error",
                "use_fix_all_bot / use_fix_sp_bot: the runner Dirichlet-pins "
                "bottom-layer species after the hydrostatic balance; the pin is "
                "not in the adjoint map, so bottom-row sensitivities for pinned "
                "species are wrong. NOTE the global fp_err (max-norm relative to "
                "max|y*|) can read ~1e-9 while a pinned trace row is 100% off — "
                "trust the per-cell defect scan, not fp_err, here. "
                "make_body_terms packs the pins automatically.",
            )

    if bool(flag("use_fix_H2He", False)):
        pinned = state_bool("h2he_pinned")
        if pinned and terms_bot:
            add(
                "hycean_h2he_pin",
                "info",
                "the tripped hycean H2/He bottom pin is carried by body_terms.",
            )
        elif pinned is False:
            add(
                "hycean_h2he_pin",
                "warning",
                "use_fix_H2He=True but the H2/He bottom pin had not yet "
                "tripped at this state (t <= hycean_pin_time). It is not in "
                "the adjoint map; once it trips the bottom H2/He rows are "
                "pinned and their sensitivities become wrong.",
            )
        else:
            add(
                "hycean_h2he_pin",
                "error",
                "use_fix_H2He=True: the hycean H2/He bottom-layer pin is not "
                "in the adjoint body map; bottom H2/He (and everything "
                "buffered by them) sensitivities are wrong. make_body_terms "
                "packs the pin once it has tripped.",
            )

    if bool(flag("use_photo", False)):
        if photo_recompute_k is None:
            add(
                "photolysis_feedback",
                "warning",
                "use_photo=True with photo_recompute_k=None: the adjoint "
                "holds J frozen at its converged value (dJ/dy omitted), so "
                "photo-coupled rows are leading-order only (~11% measured on "
                "W39b OH+H2). Pass photo_recompute_k = "
                "make_photo_recompute_k(integ._photo_static, converged_state) "
                "— the standard companion on photo-on columns (-> ~0.2%).",
            )
        else:
            add(
                "photo_dflux_recursion",
                "info",
                "photo_recompute_k carries dJ/dy, but dflux_u (the two-stream "
                "upward-flux self-recursion) is held at its converged value "
                "inside the recompute — a second-order truncation.",
            )

    if len(list(flag("diff_esc", []) or [])) > 0:
        add(
            "escape_flux_feedback",
            "warning",
            "diff_esc is non-empty: the runner recomputes the diffusion-"
            "limited escape flux from the top-layer densities at refresh "
            "cadence (update_phi_esc); the adjoint freezes top_flux, so the "
            "d(phi_esc)/dy feedback is dropped. TOA-sensitive losses are "
            "biased.",
        )

    if bool(flag("use_vm_mol", False)) and not bool(flag("use_hybrid_vm_mol", False)):
        add(
            "vm_mol_feedback",
            "warning",
            "use_vm_mol=True: the runner refreshes vm(mu(y)) every step via "
            "recompute_vm_jax; the adjoint uses the frozen converged vm, "
            "dropping the d(vm)/dy feedback. Matters where molecular "
            "diffusion dominates (low-Kzz upper atmospheres).",
        )
    elif bool(flag("use_vm_mol", False)) and bool(flag("use_hybrid_vm_mol", False)):
        add(
            "vm_mol_hybrid",
            "info",
            "use_hybrid_vm_mol=True: the run converges on the central-"
            "difference operator (phase 1), so the converged state carries "
            "hybrid_use_vm=0 and the adjoint linearizes central diff — there "
            "is no upwind d(vm)/dy feedback to drop.",
        )

    add(
        "atm_refresh_feedback",
        "info",
        "The composition->mu->(g, dz, Hp, Hpi) atm-refresh feedback "
        "(update_frq cadence, always on in the runner) is not in dG/dy; the "
        "adjoint uses the frozen converged geometry. Second-order for "
        "H2-dominated columns; the FD validation totals (0.1-0.7% on "
        "HD189/W39b) already include it.",
    )

    if (
        bool(flag("use_topflux", False))
        or bool(flag("use_botflux", False))
        or bot_pins
        or len(list(flag("diff_esc", []) or [])) > 0
    ):
        add(
            "open_boundary_deflation",
            "info",
            "Open/pinned boundaries break exact column atom conservation, so "
            "a deflated conserved-mass direction may not be a true null "
            "direction — check info['null_quality'] from the sensitivity "
            "call (O(1) means the deflation is corrupting the solve).",
        )

    return findings


def audit_adjoint_scope(
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    *,
    cfg=None,
    final_state=None,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    photo_recompute_k: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    body_terms: BodyTerms | None = None,
    body_dt: float = BODY_MAP_DT,
    solver_map: str = SOLVER_MAP_DEFAULT,
    species: Sequence[str] | None = None,
    min_ymix: float = _AUDIT_MIN_YMIX,
    top_k: int = 10,
    print_report: bool = True,
):
    """Scan a run for physics the adjoint's body map drops -- BEFORE trusting it.

    The adjoint linearizes `G = ros2_step (+ renorm) (+ photo)` at `y_star`;
    every other per-step runner process is outside that map. Dropping a
    process is exact when it is inactive/identity at the fixed point and
    physically wrong when it still shapes it. Four checks:

    1. Static config/state checks (`_adjoint_scope_findings`), classified
       error / warning / info.
    2. Per-cell fixed-point defect `|G(y*) - y*| / y*` on cells with
       `ymix >= min_ymix`, built from the SAME `_make_body_map` the solver
       uses -- any unmodeled ACTIVE process shows up as a localized defect.
       This catches what the global max-norm `fp_err` structurally cannot: a
       pinned bottom row is invisible next to the deep-column H2 density.
       Cells the runner's zero-clip owns are excluded and reported separately
       (`n_clip_dead_excluded`); that exclusion is lifted inside the loss
       footprint, where such a cell is a hard error.
    3. Stale-geometry check (with `final_state`): verifies the converged
       refresh fields (g, dzi, Hpi, top_flux, vs) were spliced into `atm`.
    4. Loss footprint (with `loss_fn`): the worst defect among the cells the
       loss actually reads (|y* dL/dy| within `_AUDIT_LOSS_FOOTPRINT_FRAC` of
       max) -- a defect there biases the answer directly.

    Interpretation: healthy bulk cells read ~1e-9..1e-6. 1e-2..0.3 is
    ambiguous (WARNING: a weak unmodeled process, or slow trace species still
    creeping at the forward tolerance). Above `_AUDIT_DEFECT_ERROR` (0.3) is
    structural (ERROR): a converged state cannot move O(1) under one probe
    step unless the runner's map contains a process this one lacks. Any
    defect inside the loss footprint is an ERROR.

    Parameters mirror `steady_state_reaction_sensitivity` where shared.
    `cfg` defaults to the process default config (pass the run's cfg for
    `make_config`-driven runs); `final_state` is the converged `JaxIntegState`
    (sharpens the conden/pin checks, enables the stale-geometry check);
    `species` labels the worst-cell table (defaults to the import-locked
    network when the width matches).

    Returns a dict: `findings` ({code, severity, message} list),
    `max_rel_defect`, `worst_cells`, `fp_err_global`, `loss_footprint_defect`
    (None without `loss_fn`), and `ok` (True when no error-severity findings;
    warnings do not clear it to False).
    """
    _check_body_dt(body_dt)
    if cfg is None:
        from .config import default_config

        cfg = default_config()  # the runner's own default cfg surface

    findings = _adjoint_scope_findings(cfg, final_state, photo_recompute_k, body_terms)

    if solver_map == "bare":
        findings.append(
            {
                "code": "bare_solver_map",
                "severity": "warning",
                "message": "solver_map='bare' linearizes the raw Ros2 step, "
                "for which y_star is only a ~1e-4 fixed point (the "
                "renormalization correction) — a ~few-% gradient bias the "
                "default 'renorm' removes. Legacy only.",
            }
        )

    # Stale-geometry check: the body map must see the SAME refreshed fields
    # the runner converged with, or the linearization is taken off-manifold.
    if final_state is not None:
        stale = []
        for name in ("g", "dzi", "Hpi", "top_flux", "vs"):
            a = getattr(atm, name, None)
            b = getattr(final_state, name, None)
            if a is None or b is None:
                continue
            a_np, b_np = np.asarray(a), np.asarray(b)
            if a_np.shape != b_np.shape or not np.allclose(
                a_np, b_np, rtol=1e-12, atol=0.0
            ):
                stale.append(name)
        if stale:
            findings.append(
                {
                    "code": "stale_geometry",
                    "severity": "error",
                    "message": f"atm fields {stale} differ from the converged "
                    "carry: splice the refreshed geometry before the adjoint "
                    "— atm_static._replace(g=final.g, dzi=final.dzi, "
                    "Hpi=final.Hpi, top_flux=final.top_flux, vs=final.vs).",
                }
            )

    # Per-cell fixed-point defect of the exact map the solver would use.
    _, body_map, _, _ = _make_body_map(
        y_star, k_arr, atm, net, body_dt, solver_map, photo_recompute_k, body_terms
    )
    G = jax.jit(body_map)(y_star)
    y_np = np.asarray(y_star)
    G_np = np.asarray(G)
    defect = np.abs(G_np - y_np)
    fp_err_global = float(
        defect.max() / max(float(np.abs(y_np).max()), _UNDERFLOW_DENOM)
    )
    ymix = y_np / np.maximum(y_np.sum(axis=1, keepdims=True), _UNDERFLOW_DENOM)

    # Cells the runner's zero-clip owns have NO fixed point: the clip is
    # outside the body map, so where it fires the runner zeroes the cell while
    # the map keeps the raw step, and |G-y|/y measures the clip (it GROWS with
    # body_dt). min_ymix cannot exclude them: the clip window is ABSOLUTE
    # (cm^-3) while min_ymix is a MIXING RATIO, so a cold low-density top
    # layer can sit orders inside the clip window at ymix 1e-16. Detect them
    # mechanistically from where the clip WOULD fire.
    _pos_cut = float(getattr(cfg, "pos_cut", 0.0)) if cfg is not None else 0.0
    _nega_cut = float(getattr(cfg, "nega_cut", 0.0)) if cfg is not None else 0.0
    clip_dead = _clip_dead_mask(G_np, ymix, cfg)

    mask = (y_np > 0.0) & (ymix >= min_ymix) & ~clip_dead
    rel = np.where(mask, defect / np.maximum(y_np, _UNDERFLOW_DENOM), 0.0)
    max_rel_defect = float(rel.max())

    # skipped != passed: report the exclusion instead of silently shrinking
    # the scan (fail-fast/announce rule).
    n_clip_dead = int((clip_dead & (y_np > 0.0) & (ymix >= min_ymix)).sum())
    clip_dead_worst = 0.0
    if n_clip_dead:
        _rel_dead = np.where(
            clip_dead & (y_np > 0.0) & (ymix >= min_ymix),
            defect / np.maximum(y_np, _UNDERFLOW_DENOM),
            0.0,
        )
        clip_dead_worst = float(_rel_dead.max())

    if species is None:
        try:  # label with the import-locked network when the width matches
            from . import chem_funs

            if len(chem_funs.spec_list) == y_np.shape[1]:
                species = list(chem_funs.spec_list)
        except Exception:
            species = None

    order = np.argsort(rel.ravel())[::-1][: max(0, int(top_k))]
    worst_cells = []
    for flat_idx in order:
        z, i = np.unravel_index(int(flat_idx), rel.shape)
        if rel[z, i] <= 0.0:
            break
        worst_cells.append(
            {
                "layer": int(z),
                "species_index": int(i),
                "species": species[i] if species is not None else f"sp{i}",
                "rel_defect": float(rel[z, i]),
                "ymix": float(ymix[z, i]),
            }
        )

    if max_rel_defect > _FP_ERR_WARN:
        w0 = worst_cells[0]
        where = f"at {w0['species']} layer {w0['layer']}"
        if max_rel_defect > _AUDIT_DEFECT_ERROR:
            findings.append(
                {
                    "code": "fixed_point_defect",
                    "severity": "error",
                    "message": f"max per-cell fixed-point defect "
                    f"{max_rel_defect:.2e} {where} (global fp_err "
                    f"{fp_err_global:.2e}): an O(1) move under one probe step "
                    "cannot be convergence creep — the map the runner "
                    "actually iterates includes a process (pin, conden "
                    "clamp, charge balance) this body map does not. See the "
                    "findings above and worst_cells.",
                }
            )
        else:
            findings.append(
                {
                    "code": "fixed_point_defect",
                    "severity": "warning",
                    "message": f"per-cell fixed-point defect up to "
                    f"{max_rel_defect:.2e} {where} (global fp_err "
                    f"{fp_err_global:.2e}): either a weak unmodeled per-step "
                    "process acts there, or those cells are slow species "
                    "still creeping at the forward convergence tolerance "
                    "(the state-definition mismatch; typical for upper-"
                    "atmosphere trace species). Both bias the adjoint in "
                    "those cells — converge/polish y_star tighter if they "
                    "matter to your loss; check worst_cells against the "
                    "loss footprint.",
                }
            )

    if n_clip_dead:
        findings.append(
            {
                "code": "clip_dead_cells_excluded",
                "severity": "info",
                "message": f"{n_clip_dead} cell(s) excluded from the per-cell "
                f"defect scan: the runner's zero-clip ([{_nega_cut:g}, "
                f"{_pos_cut:g}) cm^-3) fires there, so they have no fixed "
                f"point and their relative defect (up to "
                f"{clip_dead_worst:.2e}) measures the clip rather than a "
                "dropped process. They are solver noise -- densities far "
                "below one particle per cm^3 -- but their adjoint rows are "
                "linearized as identity, so a loss that READS one is still "
                "refused (see loss_footprint_defect).",
            }
        )

    # Loss footprint: does the loss read any defective cell directly?
    loss_footprint_defect = None
    if loss_fn is not None:
        v = np.asarray(jax.grad(loss_fn)(y_star))
        w = np.abs(v * y_np)  # log-space cotangent magnitude, the adjoint RHS
        # Report a non-finite cotangent, never swallow it: every `>` against
        # NaN is False, so a NaN w.max() would leave `foot` empty and declare
        # a clean footprint on a poisoned loss gradient.
        n_bad_cot = int((~np.isfinite(v)).sum())
        if n_bad_cot:
            findings.append(
                {
                    "code": "loss_cotangent_non_finite",
                    "severity": "error",
                    "message": f"{n_bad_cot} cell(s) of jax.grad(loss_fn)(y_star) "
                    "are NaN/Inf, so the adjoint RHS is poison before the scope "
                    "audit even begins. Every downstream footprint comparison "
                    "silently reads False against a NaN, which would look like a "
                    "clean footprint. Fix the loss (a log of a zero-clipped cell "
                    "is the usual cause) before trusting any sensitivity.",
                }
            )
        w_finite = w[np.isfinite(w)]
        w_max = float(w_finite.max()) if w_finite.size else 0.0
        foot = np.isfinite(w) & (
            w > _AUDIT_LOSS_FOOTPRINT_FRAC * max(w_max, _UNDERFLOW_DENOM)
        )
        # Clip-dead cells read a clean 0.0 in `rel`; the loss footprint is
        # exactly where that leniency is NOT allowed (the map linearizes those
        # rows as identity while the runner zeroes them), so score the
        # footprint on the UNMASKED defect.
        rel_full = np.where(
            (y_np > 0.0) & (ymix >= min_ymix),
            defect / np.maximum(y_np, _UNDERFLOW_DENOM),
            0.0,
        )
        loss_footprint_defect = float(rel_full[foot].max()) if foot.any() else 0.0
        if foot.any() and (clip_dead & foot).any():
            findings.append(
                {
                    "code": "loss_reads_clip_dead_cell",
                    "severity": "error",
                    "message": f"{int((clip_dead & foot).sum())} cell(s) in "
                    "the loss footprint sit in the runner's zero-clip dead "
                    "zone: those cells have no fixed point and the adjoint "
                    "would linearize a clip as identity, so THIS loss's "
                    "gradient is wrong at first order. Choose a loss on "
                    "species/layers with real abundance.",
                }
            )
        if loss_footprint_defect > _FP_ERR_WARN:
            findings.append(
                {
                    "code": "loss_footprint_defect",
                    "severity": "error",
                    "message": f"the loss directly reads cells with fixed-"
                    f"point defect up to {loss_footprint_defect:.2e}: the "
                    "gradient of THIS loss is biased at that level "
                    "regardless of cause (unmodeled process or cells not "
                    "fully converged) — fix the scope or converge tighter "
                    "before using it.",
                }
            )

    severity_rank = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_rank.get(f["severity"], 3))
    ok = all(f["severity"] != "error" for f in findings)

    if print_report:
        print("audit_adjoint_scope report")
        print(
            f"  ok={ok}  max_rel_defect={max_rel_defect:.3e}  "
            f"fp_err_global={fp_err_global:.3e}"
            + (
                f"  loss_footprint_defect={loss_footprint_defect:.3e}"
                if loss_footprint_defect is not None
                else ""
            )
        )
        for f in findings:
            print(f"  [{f['severity'].upper():7s}] {f['code']}: {f['message']}")
        if worst_cells:
            print("  worst fixed-point-defect cells (rel |G(y*)-y*|/y*):")
            for c in worst_cells:
                print(
                    f"    {c['species']:>12s}  layer {c['layer']:3d}  "
                    f"defect {c['rel_defect']:.3e}  ymix {c['ymix']:.3e}"
                )

    return {
        "findings": findings,
        "ok": ok,
        "max_rel_defect": max_rel_defect,
        "fp_err_global": fp_err_global,
        "worst_cells": worst_cells,
        "loss_footprint_defect": loss_footprint_defect,
        # how much the scan skipped, so a caller can see the leniency it got
        "n_clip_dead_excluded": n_clip_dead,
        "clip_dead_worst_defect": clip_dead_worst,
    }
