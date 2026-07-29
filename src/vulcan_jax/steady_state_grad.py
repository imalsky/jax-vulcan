"""Reverse-mode reaction sensitivities at the converged photochemical state.

`outer_loop.OuterLoop`'s `lax.while_loop` supports `jvp`/`jacfwd` but not
`vjp`/`grad`, so reverse-mode AD cannot be taken straight through the
integration. The reverse-mode question this module answers is the one a
many-inputs/one-output gradient is for: *which directional rate-table entries
set the converged abundance of a given species* -- `dL/d(ln k_r)` for all `nr`
rows from a single adjoint solve, where finite differences would cost one
re-converged model per row.

The route — the solver-map steady-state adjoint
================================================
At convergence the body map `G` has a fixed point, `G(y*) = y*`, so the
steady-state cotangent solves the *fixed-point* adjoint

    (I - dG/dy)^T z = v ,        v = dL/dy* ,

after which `dL/d(ln k_r) = (k .* vjp_Gk(lambda))_r`, with `lambda` the
unscaled cotangent. `(I - dG/dy)^T` is the integrator's *own* regularized
implicit step (the block-Thomas solve at the body-map dt) transposed — it
already embeds the preconditioner that tames the chemical stiffness, which is
why it is far better conditioned than the bare residual Jacobian. By default `G`
is the hydrostatic-renormalized step the runner actually iterates
(`solver_map="renorm"`), so `y*` is a tight fixed point and the gradient is
percent-level; `solver_map="bare"` reproduces the raw-step legacy behavior. On
photo-on columns the default `photo_recompute_k="auto"` rebuilds `J(y)` from the
finished runner context so `dG/dy` carries the photolysis feedback. See the
`steady_state_reaction_sensitivity` docstring / `SOLVER_MAP_DEFAULT` / limits.

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
   because `c_e^T df/dk = 0` for conservative rate-table perturbations. In
   discrete practice the `c_e` can be only approximately null; `info["null_quality"]`
   measures the actual defect relative to the operator scale, with O(1) values
   indicating that a deflated direction is not a null direction for this model
   setup (for example, open boundary fluxes).
4. **LGMRES, not restarted GMRES or Neumann.** The deflated operator is
   indefinite; an augmented Krylov method (LGMRES carries vectors across
   restarts) converges where restarted GMRES oscillates and a raw Neumann
   iteration diverges.

Limitations — read before using
===============================
With the default renormalized solver map (and `photo_recompute_k` on
photochemistry-on columns) this reaches percent-level directional-row
sensitivities on well-conditioned rows; it degrades to a reaction-*ranking*
tool where the operator is ill-conditioned. The body map deliberately contains
ONLY `ros2_step (+ renorm) (+ photo recompute)` — every other per-step runner
process (clip, condensation, charge balance, fix-species and boundary pins,
atm-refresh feedback) is outside the linearization; run
`audit_adjoint_scope(...)` on a converged run to check none of them is active
at the fixed point for YOUR config (it also measures the per-cell fixed-point
defect the global `fp_err` max-norm can mask). Four structural limits shape
which regime applies:

1. **Which map is linearized dominates the accuracy — not the convergence tolerance.**
   `lax.while_loop` blocks `vjp`, so reverse-mode can only take the *steady-state*
   adjoint of a one-step map at `y_star`. The forward loop iterates the
   *hydrostatic-renormalized* map, so the default `solver_map="renorm"`
   linearizes exactly that map and `y_star` is a tight fixed point
   (`info["fp_err"]` ~ 1e-9). The legacy `solver_map="bare"` instead linearizes
   the raw Ros2 step, for which `y_star` is only a loose fixed point
   (`fp_err` ~ 1e-4, the renormalization correction), and its gradient carries a
   ~few-% bias. A 2026-07-03 campaign (HD189/HD209/W39b, FD- and
   forward-mode-validated) showed that bare-map bias is **not** removed by
   stricter forward convergence (HD189 CH4 stayed ~6-8% across `yconv` 1e-2..1e-4
   while the FD truth was invariant), nor by scanning `body_dt` (non-monotonic,
   optimum ~1e7 at ~6.6%), nor by a bigger LGMRES budget (~5% floor) — only the
   renormalized map removes it (HD189 CH4 -> ~0.7%, HD209 forward rows 35% -> 1%).
   See `SOLVER_MAP_DEFAULT`.
   Two error sources survive `"renorm"` and are genuinely structural: the
   frozen-photolysis term on photo-on columns (limit 2 — it dominates the
   photo-coupled rows, e.g. WASP-39b OH+H2 at ~11%) and severe operator
   ill-conditioning on slow-radical columns (HD209 CH2OH, LGMRES residual
   stuck ~0.1, near-equilibrium reverse rows unreliable). A separate `body_dt`
   *regime* question remains (too large stagnates, too small underweights slow
   modes; optimum is model-dependent — use `scan_body_dt_reaction_sensitivity`),
   and `info["ensemble_spread"]` reports twin-to-twin magnitude stability: trust
   magnitudes when residual and spread are small, else treat the output as a
   ranking.
2. **Photolysis feedback is automatic only when the finished runner context is supplied.**
   `J(y)` depends on the abundances through optical depth; with the default
   `photo_recompute_k=None` the adjoint holds `J` at its converged value and
   omits the `dJ/dy` feedback, so photo-coupled rows are leading-order only
   (WASP-39b OH+H2 ~11%, even with the default renorm map). The public default
   is `photo_recompute_k="auto"`: pass `runner_photo_static` and
   `converged_state` (or `integ` and `converged_state`) and it builds the
   recompute callable with `make_photo_recompute_k`. With the default renorm map
   this takes the WASP-39b SO2 dominant rows to ~0.1-0.2% vs re-converged FD
   (2026-07-03). If active photo rows are present but the runner context is
   missing, the default raises; pass `photo_recompute_k=None` only to reproduce
   the frozen-photolysis legacy result. It costs an RT solve per Krylov matvec.
3. **Two entry points by input shape.** `steady_state_reaction_sensitivity`
   returns `dL/d(ln k_r)` for all rate-table rows;
   `steady_state_input_sensitivity` returns `dL/dp` for an arbitrary physical
   input pytree (e.g. a full (nz,) temperature profile) from the SAME single
   adjoint solve plus one VJP of a caller-supplied differentiable rebuild
   `p -> (k(p), atm(p))` — the many-inputs/one-output shape in both cases.
   Forward-mode remains the higher-accuracy route for a handful of inputs.
4. **Needs a genuine fixed point and a safe `body_dt`.** `y*` must be a tight fixed
   point of the *chosen* body map (`info["fp_err"]` reports how tight; note the
   forward-converged state is tight for `solver_map="renorm"` but only ~1e-4 for
   `"bare"`), and `body_dt` must stay in the safe regime (see `BODY_MAP_DT`; the
   danger zone is guarded).

Pair sums (read before ranking physical reactions)
--------------------------------------------------
For reversible thermal reactions, this function reports directional row
sensitivities. A physical detailed-balance perturbation that scales a forward
rate and its derived reverse rate together should use the pair sum
`g[fwd] + g[rev]` (photolysis and other one-way rows remain single entries).
The default renorm map (with `photo_recompute_k` on photo-on columns) is
FD-validated accurate for the pair sums too, NOT only the forward rows
(2026-07-04, W39b, re-converged detailed-balance FD): forward OH+H2 0.2% /
SO+OH 0.07%, **reverse** OH+H2 0.2% / SO+OH 0.1%, and the physical pair-sum
SO+OH 0.8% (vs the bare frozen-photolysis adjoint's 11-17% on the same reverse
rows and pair-sum). **Do not read `info["pair_antisym"]` as an error signal for
the renorm map:** it is a bare-map-calibrated diagnostic (a near-equilibrium
pair should nearly cancel) and reads ~1 for renorm on some pair *even though the
FD-validated pair-sums are more accurate than bare's* — the offending pair
simply has a genuinely non-zero pair-sum that the bare map over-cancels. Use the
default (renorm + `photo_recompute_k`) for pair-summed physical rankings such as
paper Fig `fig:so2_rev`. (The separately-hard HD209 CH2OH near-equilibrium
reverse rows are a distinct ill-conditioning issue, flagged by the residual/
spread diagnostics.)

**Forward-mode** (`jvp`/`jacfwd` through the runner) remains the higher-accuracy
route for end-to-end gradients and the right tool when the number of input
directions is small (Kzz, metallicity, temperature); reverse-mode here is the
right tool for the opposite shape -- all rate-table rows at once.

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

Scope, accuracy, and the `body_dt` regime map: `docs/differentiability.md`
("Reverse mode: the steady-state adjoint"). Worked recipe:
`examples/grad_reverse_example.py`; paper Fig `fig:so2_rev`.
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


# --- Solver-map / LGMRES knobs (adjoint-solver constants live here, beside the
#     code, the same as the forward model's knobs live in vulcan_cfg.py). ---

SOLVER_MAP_DEFAULT = "renorm"
# Which one-step map the adjoint linearizes at the converged state:
#   "renorm" -> G(y) = M * ros2_step(y, k, dt) / sum_i ros2_step   (DEFAULT)
#   "bare"   -> G(y) = ros2_step(y, k, dt)                         (legacy)
# The forward `OuterLoop` iterates the HYDROSTATIC-RENORMALIZED map
# (outer_loop.py: `sol_balanced = M[:,None] * ymix_new`), so the converged
# `y_star` is a *tight* fixed point of "renorm" (fp_err ~ 1e-9) but only a loose
# fixed point of "bare" (fp_err ~ 1e-4, the renormalization correction). The
# bare map is therefore linearized ~1e-4 off its own manifold, biasing the
# gradient at the few-percent level even after body_dt/LGMRES/convergence are
# tuned (a 2026-07-03 campaign confirmed neither stricter convergence nor a
# body_dt scan removes this: HD189 CH4 sits at ~6-8% for the bare map across
# yconv 1e-2..1e-4 and body_dt 1e6..1e8, with the FD truth invariant). "renorm"
# restores the correct fixed point and drops HD189 CH4 to ~0.7% (HD209 forward
# rows from ~35% to ~1%), so it is the DEFAULT. "bare" is kept only to reproduce
# the pre-2026-07 behavior. On photochemistry-on columns "renorm" removes the
# solver-map bias but a *separate* frozen-photolysis error (dJ/dy omitted) then
# dominates the photo-coupled rows -- the default `photo_recompute_k="auto"`
# closes that too when the caller supplies the runner's photolysis state
# (WASP-39b OH+H2 ~11% -> ~0.2%). Do NOT
# deflate the per-layer total-density direction on top of "renorm" -- measured
# to over-correct (HD189 0.7% -> 2.5%).
SOLVER_MAP_CHOICES = ("bare", "renorm")

PHOTO_RECOMPUTE_AUTO = "auto"
PhotoRecomputeArg = Callable[[jnp.ndarray], jnp.ndarray] | Literal["auto"] | None

BODY_MAP_DT = 1e7
# Time step (s) for the bare Ros2 body map G(y) = ros2_step(y, k, dt). This is
# an ADJOINT-ONLY knob: it never touches the forward model — it sets how the
# one probe step weights each chemical mode in the adjoint linear system
# (weight ~ dt/tau for slow modes; FP-noise amplification grows with dt).
# The usable window is column/network-dependent. For publication-grade numbers,
# scan a few dt values and keep the lowest-residual, low-spread solution.
BODY_MAP_DT_CANDIDATES = (3e6, 1e7, 3e7, 1e8)

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

_UNDERFLOW_DENOM = 1e-300
# Numerical floor for `/max(|x|, .)` normalizers (norms, op-scale, pair sums).
# Below-which-is-zero guard, not a tuning knob.

LGMRES_INNER_M = 60
# scipy.sparse.linalg.lgmres inner Krylov dimension. Larger m costs memory and
# matvecs per cycle, but can improve difficult solves.

LGMRES_OUTER_K = 40
# Augmentation vectors carried across restarts — the LGMRES knob that fixes the
# restarted-GMRES oscillation on this indefinite operator.

LGMRES_MAXITER = 4
# Inner iterations per warm-start cycle. The solve is chunked into cycles so the
# residual/gradient trajectory is observable; the per-cycle x0 warm-start is the
# validated configuration.

LGMRES_CYCLES = 10
# Number of warm-start cycles.

LGMRES_RTOL = 1e-12
# Relative-residual target. Once finite-tolerance state mismatch dominates, a
# tighter Krylov tolerance no longer improves agreement with re-converged FD.

_ADJOINT_RESID_WARN = 0.2
# Warn above this MEDIAN relative LGMRES residual across the ensemble. The
# median is robust to a single wandering twin. Distinct from finite-tolerance
# state mismatch, which no Krylov setting removes.

_FP_ERR_WARN = 1e-2
# Warn above this body-map fixed-point error: y_star is not a tight fixed point,
# so the adjoint is being evaluated off the steady-state manifold.
#
# NOTE (not a warning): `info["pair_antisym"]` is NOT gated by a default warning.
# It is a bare-map-calibrated diagnostic (a near-equilibrium reversible pair
# should nearly cancel), and reads ~1 for the renorm default on some pair. That
# is NOT an error: a 2026-07-04 detailed-balance FD campaign on W39b found the
# renorm+dJ/dy default MORE accurate than bare on the very quantities pair_antisym
# is about — reverse rows (r2 0.2% / r692 0.1% vs bare 11% / 1.4%) and physical
# pair-sums (SO+OH 0.8% vs bare 17%); the high reading is a genuinely non-zero
# pair-sum that the bare map over-cancels. Gating a warning on it would fire on
# the accurate default path, so we don't. See the "Pair sums" docstring section.

_NULL_BASIS_RANK_TOL = 1e-10
# Rank guard for the deflation basis. After column normalization, |R_jj| from
# the (unpivoted) QR measures how independent atom column j is of the previous
# columns; below this it is numerically dependent, and QR's j-th Q column is an
# arbitrary orthonormal direction NOT in span(C) — deflating it would silently
# project a needed direction out of the adjoint solve, so fail fast instead.

_AUDIT_MIN_YMIX = 1e-16
# audit_adjoint_scope ignores cells below this mixing ratio in the per-cell
# fixed-point-defect scan: they are numerical dust whose relative defect is
# meaningless, while every species one would put in a loss (trace radicals
# ~1e-12..1e-8) sits far above it.

_AUDIT_DEFECT_ERROR = 0.3
# audit_adjoint_scope per-cell defect above which the finding is an ERROR: an
# O(1) move of a cell under one probe step cannot be finite-convergence creep
# on a state the runner accepted as converged — it means the iterated map
# includes a process (pin, conden clamp, charge balance) the body map lacks.
# Defects between _FP_ERR_WARN and this are a WARNING: measured 6.5e-2 on the
# healthy HD189 fixture's slow upper-atmosphere nitrogen cells (finite forward
# tolerance, the "state-definition mismatch" of the module docstring) while
# the same fixture's CH4 gradient FD-validates at 0.7% — ambiguous, not fatal.

_AUDIT_LOSS_FOOTPRINT_FRAC = 1e-3
# audit_adjoint_scope's "loss footprint": cells whose log-space cotangent
# magnitude |y* dL/dy| is within this fraction of the maximum — the cells the
# loss actually reads, where a fixed-point defect directly biases the gradient.

_REBUILD_CONSISTENCY_WARN = 1e-8
# steady_state_input_sensitivity: warn when rebuild(p0) reproduces the
# converged (k_arr, atm) only to this relative level. rates_jax matches the
# host build to ~5e-14, so anything above float-noise means the rebuild is
# subtly different from the map the state converged under.

_REBUILD_CONSISTENCY_ERR = 1e-3
# ... and refuse to proceed above this: the adjoint would be linearized
# against a visibly different map (wrong caps/remove_list, missing frozen
# photolysis-row splice, wrong pressure grid) and the gradient would be
# silently wrong.


def _warn_poor_convergence(resid: float, fp_err: float, spread: float = 0.0) -> None:
    """Emit a warning (by default, not only when the caller inspects `info`) when
    the adjoint solve looks under-converged, `y_star` is not a fixed point, or
    the ensemble twins disagree. (`pair_antisym` is intentionally NOT gated here
    — see `_FP_ERR_WARN`'s NOTE — because it reads ~1 for the renorm default even
    when the directional rows are FD-accurate.)"""
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
            "docs/differentiability.md). Treat magnitudes as ranking weights only. More "
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


def _check_body_dt(body_dt: float) -> None:
    """Refuse a body_dt outside the usable window (see _BODY_MAP_DT_MAX).

    Both bounds matter. `body_dt=0` makes the implicit step the identity and the
    adjoint solve divides by zero (NaN); `body_dt<0` integrates backwards and
    returns a finite, plausible-looking, wrong sensitivity — the worse failure,
    because nothing downstream flags it.
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
    finished runner; all fields default-inactive (`None`).

    `conden_static` enables the in-window condensation composite: the conden/
    evap k-rows are recomputed from `y` each application (the dk/dy feedback,
    analogous to photolysis dJ/dy) and the H2O/NH3 relax kernels run at the
    probe `body_dt`. `gas_mask`/`hydro_partial` switch the hydrostatic
    rebalance to the runner's gas-only-denominator, non-gas-passthrough form.
    `fix_mask`/`fix_y` reproduce the fix_species regime (species pinned inside
    the Ros2 step, then overwritten with the pinned values so their body-map
    rows are constants). `bot_idx`/`bot_val` are the layer-0 Dirichlet pins
    (`use_fix_all_bot` / `use_fix_sp_bot` / tripped hycean H2-He), applied
    after the balance exactly as the runner does.
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

    Mirrors `outer_loop._make_clip_fn` term for term on a candidate post-step
    state `G` (number density, cm^-3) with the pre-step mixing ratio
    `ymix_old`:

        y < pos_cut and y >= nega_cut          -> 0     (small/negative cut)
        ymix_old < mtol and y < 0              -> 0     (trace-negative cut)

    The clip is deliberately outside the adjoint's body map, which is exact
    only where the clip is the identity. Where it fires, the runner zeroes the
    cell while the map keeps the raw step, so the cell has no fixed point and
    its relative defect measures the clip. Returns all-False when `cfg` is
    None (nothing to mirror), which is the conservative direction: those cells
    stay in the scan.
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
    `apply_post_map(sol, M_col=None)` is the shared post-step (plain renorm,
    or the conden-aware composite when `body_terms` carries terms; the
    optional `M_col` override exists for parameter maps where M depends on
    the input — temperature enters the renormalization), `body_map_k` is the
    k-linearization at `y_star`, and `step_fn(y, k, atm)` is the pinned Ros2
    step for parameter maps that rebuild `atm`.

    This is the SINGLE definition of that map: `steady_state_reaction_sensitivity`,
    `steady_state_input_sensitivity`, and `audit_adjoint_scope` all build from
    here, so the audited map is exactly the solved one by construction.
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
    # Per-layer total number density for the hydrostatic renormalization that
    # the runner applies after each step (outer_loop.py `sol_balanced`).
    M_col_default = atm.M[:, None]
    dt64 = jnp.float64(body_dt)

    def apply_post_map(sol, M_col=None):
        # "renorm" reproduces the runner's hydrostatic rebalance so that the
        # linearized map has the forward-converged y_star as a tight fixed
        # point; "bare" leaves the raw Ros2 step (see SOLVER_MAP_DEFAULT).
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
        # overwrite with the pinned values, mirroring the runner — the
        # overwrite makes pinned rows constants of the map (identity rows of
        # I - dG/dy) instead of pass-throughs (singular rows).
        if t is not None and t.fix_mask is not None:
            sol, _ = jax_ros2_step(y, k_use, dt64, atm_use, net, fix_mask=t.fix_mask)
            return jnp.where(t.fix_mask, t.fix_y, sol)
        sol, _ = jax_ros2_step(y, k_use, dt64, atm_use, net)
        return sol

    # When `photo_recompute_k` is given, the photolysis rows of k are rebuilt
    # from y each application (via the runner's own two-stream RT), so the
    # y-VJP carries the dJ/dy feedback that the explicit frozen-photolysis
    # legacy path omits. `update_conden_rates` plays the identical role for the conden
    # rows (rate ∝ y - y_sat). Folding both recomputes into the map from the
    # INCOMING y is exact for the fixed-point adjoint: the runner recomputes
    # from the post-step state, but incoming == post-step at the fixed point,
    # and the eliminated one-step lag drops out of the coupled-system algebra.
    def body_map(y):
        k_use = photo_recompute_k(y) if photo_recompute_k is not None else k_arr
        if t is not None and t.conden_static is not None:
            k_use = update_conden_rates(k_use, y, t.conden_static)
        return apply_post_map(step_fn(y, k_use, atm))

    # k-linearization at y_star: k stays the free variable (photo/conden rows
    # NOT recomputed here — a directional row entry means "perturb this rate",
    # while the y-operator above carries the state feedback).
    def body_map_k(k):
        return apply_post_map(step_fn(y_star, k, atm))

    return apply_post_map, body_map, body_map_k, step_fn


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
    in residual; tracking the best iterate costs one extra matvec per cycle.
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
            # Converged; the warm-start cycles exist to continue an
            # unconverged solve, not to re-iterate a converged one.
            break
    return x_best


def _topk_ensemble_spread(g_stack: np.ndarray) -> float:
    """Twin-to-twin disagreement on the components one would actually report:
    max over the TOP-10 components by |mean| of (max-min)/|mean|. Weaker
    components naturally carry larger relative bounce; including them makes
    the metric alarm on noise that never enters a ranking figure."""
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

    `PHOTO_RECOMPUTE_AUTO` is the accurate default: build the recompute callable
    from the finished runner when possible. If active photolysis rows are present
    but the default renorm path lacks that context, refuse the call instead of
    silently returning the lower-accuracy frozen-photolysis adjoint. Passing
    `photo_recompute_k=None` remains the explicit legacy/frozen choice.
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

    `NetworkArrays` carries no photo/ion/conden row masks, so the row checks
    use the import-locked parsed network (when its `nr` matches `net`) and the
    condensate check uses the import-locked species list (when its width
    matches `y_star`). A non-matching custom network skips the fingerprints
    with a RuntimeWarning (skipped != passed) — run `audit_adjoint_scope`
    with the run's cfg for full coverage there. `network`/`species` exist
    for tests.

    Raises on: nonzero ion rows (charge balance is not in any body map);
    condensation fingerprints (nonzero conden rate rows, or a populated
    `*_l_s` condensate) without conden/fix-species body terms. Warns on:
    active photolysis rows without `photo_recompute_k` (frozen dJ/dy,
    leading-order-only photo-coupled rows).
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
        # `nr` is the identity key, so ANY custom network used to skip every
        # refusal with only a warning. We cannot fingerprint without the network,
        # but network-independent evidence (no conden body terms + an all-zero
        # species column) is enough to refuse a state that is not provably
        # benign; `network=`/`species=` gets the real check.
        needs_conden_terms = body_terms is None or body_terms.conden_static is None
        y_np_probe = np.asarray(y_star)
        # Condensate columns are the ones the runner drives to zero everywhere
        # except where condensation is active; a fully-positive column set is the
        # benign signature of a gas-only column.
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
            # Condensed-phase species are suffixed `_l_s` (H2O/NH3/S2/S8),
            # `_l` (H2SO4) or `_s` (C_s) across the shipped networks — an
            # `endswith("_l_s")` test alone silently misses H2SO4_l and C_s,
            # so a state with those populated would pass a refusal that
            # exists precisely to catch it.
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
    unscaled cotangents `lambda = z ./ y*` — ready to contract against any
    parameter map as `dL/dp = lambda^T dG/dp` — plus per-twin relative
    residuals and the shared diagnostics. `steady_state_reaction_sensitivity`
    contracts them with the k-linearization, `steady_state_input_sensitivity`
    with an arbitrary input map; the solve itself is identical.
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

    # Jit the solver-map operator once and share it between the null-quality
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
    # operator's action on a fixed-seed random unit direction. O(1) means a
    # deflated direction is NOT null (e.g. open boundary fluxes break atom
    # conservation) and the deflation is corrupting the solve. A QR-
    # orthonormality check would be vacuous here: it is tiny for any basis.
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

    # Ensemble of twin solves. A semi-converged Krylov endpoint is bit-level
    # trajectory-sensitive, so a single solve samples a distribution; solving
    # against deterministically ulp-perturbed copies of b (the perturbation is
    # ~13 orders below the signal, so the exact solution is unchanged) turns
    # that into a measurable mean + spread. Twin RHS draws are seeded and
    # reproducible; twin 0 is the unperturbed b.
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

    Returns the sensitivity of a scalar loss of the converged composition to
    every reaction-rate constant — the reaction-ranking use case (which
    reactions set the converged SO2, CH4, ...). The result is the MEAN over an
    ensemble of `n_solves` adjoint solves with deterministic ulp-perturbed
    right-hand sides; the twin-to-twin disagreement is reported as
    `info["ensemble_spread"]` and is the honest magnitude error bar.

    Limitations (see the module docstring for the full reasoning):

    * Accuracy is limited by both the linear solve and the difference between
      the exact fixed-point adjoint and the finite-tolerance state the forward
      run actually stopped at. More Krylov iterations help only until that
      state-definition mismatch dominates. The usable `body_dt` window is
      column-dependent; for publication-grade magnitudes use
      `scan_body_dt_reaction_sensitivity` and inspect residuals/spread.
    * Photochemistry-on columns use differentiated photolysis by default when
      the finished runner context is supplied; pass `photo_recompute_k=None`
      only to reproduce the frozen-photolysis legacy path.
    * Returns `k`-only sensitivities (`dL/d ln k`); for `dL/dT`, `dL/dKzz`,
      etc. use `steady_state_input_sensitivity` (same adjoint solve, one VJP
      of a differentiable rebuild) or forward-mode (`jvp`, exact) for a
      handful of directions.
    * Requires `y_star` to be a tight fixed point of the chosen body map and a
      `body_dt` in the safe regime.

    Parameters
    ----------
    loss_fn
        `loss_fn(y) -> scalar` on the full `(nz, ni)` number-density state. The
        caller closes over the species/layer of interest, e.g.
        `lambda y: jnp.log10(y[L, so2] / y[L].sum())`.
    y_star : (nz, ni)
        Converged state — a tight fixed point of the CHOSEN body map: for the
        default `solver_map="renorm"` that is the hydrostatic-renormalized
        Ros2 step the runner iterates (`info["fp_err"]` reports how tight;
        ~1e-9 on a converged column). Clip, charge balance, condensation,
        fix-species pins, and the bottom-boundary pins are NOT in either map —
        run `audit_adjoint_scope(...)` to verify none of them is active for
        your configuration (it also measures the per-cell fixed-point defect,
        which the global max-norm `fp_err` can mask for trace/bottom rows).
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
    solver_map
        `"renorm"` (default) linearizes the hydrostatic-renormalized map the
        runner actually iterates, making `y_star` a tight fixed point and
        removing the ~few-% bias of the raw Ros2 step (HD189 CH4 6.6% -> 0.7%).
        `"bare"` reproduces the pre-2026-07 raw-step behavior. See
        `SOLVER_MAP_DEFAULT`.
    photo_recompute_k
        `"auto"` (default) builds a `k(y) -> k_arr` recompute from the supplied
        runner context on active photochemistry columns, so the state operator
        carries `dJ/dy` and photo-coupled rows reach percent level (WASP-39b
        OH+H2 11% -> 0.2%). Pass a callable to override that builder. Pass
        `None` only for photo-off columns or to explicitly reproduce the
        frozen-photolysis legacy result, where those rows are leading-order only
        (~11%). The recompute costs an RT solve per Krylov matvec.
    runner_photo_static, converged_state, integ
        Context used by the default `photo_recompute_k="auto"`. Pass either
        `runner_photo_static=integ._photo_static, converged_state=final_state`
        or `integ=integ, converged_state=final_state`. `converged_state` is the
        `JaxIntegState` returned by the runner because the recompute reuses the
        runner's own photo branch.
    body_terms
        Optional `BodyTerms` carrying the per-step runner processes beyond
        `ros2 (+ renorm) (+ photo)`: the condensation composite (conden-row
        recompute from y, H2O/NH3 relax kernels, gas-only partial balance),
        the fix_species pins, and the layer-0 boundary pins. REQUIRED (the
        guard raises) when the state was converged with condensation active;
        build with `make_body_terms(integ, converged_state, atm_static)` —
        which also returns the correctly spliced `atm`. Requires
        `solver_map="renorm"`.
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
        Ensemble-mean row-wise `dL/d(ln k_r)` for every directional rate-table
        entry (index 0 is the unused 1-based pad). For a physical detailed-
        balance perturbation of a reversible thermal reaction, pair the forward
        and reverse rows as `g[fwd] + g[rev]`.
    info : dict, optional
        `fp_err` (body-map fixed-point error at y*), `null_quality`
        (max over deflated directions of `||A_eta^T q_e||`, unit-norm `q_e`,
        relative to the operator's action on a random unit direction; O(1)
        means a deflated direction is not null, e.g. open boundary fluxes, and
        the deflation is suspect), `resid` (max relative LGMRES residual over
        the ensemble), `resid_median` (median relative residual), `resids`
        (per-twin residuals), `ensemble_spread` (max over the top-10
        reactions by |mean| of (max-min)/|mean| across twins; 0.0 when
        `n_solves=1`), `pair_antisym` (worst forward/reverse asymmetry
        `|g_f+g_r|/max(|g_f|,|g_r|)` over the top-10 genuinely-reversible
        pairs; O(1) can indicate an internally inconsistent solve or a loss
        dominated by modes the chosen body map handles poorly), `n_matvec`,
        `n_null` (deflated dimensions), `n_solves`, and `body_dt`.
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

    # Condensation active: the reaction gradient is CONDITIONAL. In the post-pin
    # regime the body map holds the captured reservoir fixed (fix_mask), so this
    # is dL/d ln k AT the frozen reservoir; it does not include how the rate
    # changed the earlier drainage that set the reservoir. Reaction rates do not
    # move the saturation curve directly, so the conditional ranking is the most
    # defensible condensation-AD case: label it, do not forbid it. See
    # docs/condensation_differentiation.md (F2).
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
            "docs/condensation_differentiation.md.",
            stacklevel=2,
        )

    # One-step body map and its y-VJP (the transposed solver-map operator).
    # dJ/dy feedback rides through `photo_recompute_k` when given — the fix for
    # photo-coupled rows on photochemistry-on columns (WASP-39b OH+H2 -> H2O+H
    # drops from ~11% to ~0.2% vs re-converged FD); the conden-row and
    # relax/pin/balance terms ride through `body_terms` the same way.
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

    # Forward/reverse pair antisymmetry over the top-10 reactions: for a
    # reversible pair near partial equilibrium, dL/dln k_f ~ -dL/dln k_r, so
    # |g_f + g_r| / max(|g_f|, |g_r|) should be small. O(1) can mean the
    # adjoint is internally inconsistent or that the loss is dominated by modes
    # the chosen body map handles poorly. Photolysis / irreversible rows
    # (all-zero reverse k) are skipped. Free to compute; diagnostic only.
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
            continue  # photo/conden/irreversible slot — no genuine pair
        denom = max(abs(g_mean_full[f]), abs(g_mean_full[rev]), _UNDERFLOW_DENOM)
        pair_antisym = max(pair_antisym, abs(g_mean_full[f] + g_mean_full[rev]) / denom)

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

    One adjoint solve — identical to `steady_state_reaction_sensitivity`'s —
    yields the cotangent `lambda`; the input gradient is then a single VJP of
    the parameter map at the fixed point:

        dL/dp = lambda^T dG/dp ,   G(p) = post(ros2_step(y*, k(p), dt, atm(p)))

    so ALL components of `p` (e.g. all 150 temperature layers) cost one solve
    plus one VJP — the many-inputs/one-output shape reverse mode is for.
    The renormalization uses `atm(p).M`, so an input that moves the total
    density (temperature: M = pco/(kb T)) is differentiated through the
    rebalance too.

    Parameters (beyond the shared ones)
    -----------------------------------
    p0
        The input value the state was converged at (array or pytree).
    rebuild
        `rebuild(p) -> (k_arr_p, atm_p)`: a JAX-differentiable rebuild of the
        FULL rate table and `AtmStatic` at input `p`. It must reproduce the
        converged inputs at `p0` — checked, warn above
        `_REBUILD_CONSISTENCY_WARN`, refuse above `_REBUILD_CONSISTENCY_ERR` —
        which in particular means non-thermal rows (photolysis J, conden)
        must be spliced in FROZEN from the converged `k_arr` (their
        p-dependence is host-baked cross sections / saturation tables, not
        on-graph). Example, a temperature profile on a photo-off column::

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
    * **Chemistry path only.** This is `(dL/dy*) . (dy*/dp)` through the
      converged chemical state. A loss with a DIRECT `p` dependence (e.g. a
      chi-square of an ExoJax spectrum, where T also enters the RT opacities
      at fixed composition) needs that direct term added separately:
      `jax.grad` of the loss w.r.t. p at fixed `y_star`.
    * **Condensation is refused.** Input (T/Kzz/...) gradients through a
      condensation-active state raise by default: the saturation tables are
      frozen in dG/dp and, post-pin, the captured reservoir is held fixed, so
      the result is O(1)-unreliable vs FD (0.91 relative; the JWST/Fisher case).
      Opt in with `allow_frozen_condensation_input_grad=True` only for the known
      leading-order number. See docs/condensation_differentiation.md.
    * **Other frozen-by-design pieces** (their p-derivative is omitted): the
      photolysis T-cross-section interpolation and the atm-refresh geometry
      cascade (dz/Hp/g — second-order; rebuild what you need on-graph in
      `atm_p` instead, e.g. `Dzz(T)` via `atm_jax`).
    * **Accuracy class** is the same as the reaction sensitivities
      (percent-level with the renorm default). The conserved-mass deflation
      was PROVEN exact only for atom-conserving rate knobs; a general input
      moves M and the rebalance, so spot-validate a new input type against a
      forward-mode `jvp` in one or two directions before production use
      (`d/dT` validated on HD189, see `jax_paper/scripts/`).
    """
    # Condensation is NOT differentiable-through for input (T/Kzz/...) gradients.
    # In-window freezes the saturation tables (d(sat)/dT dropped); post-pin (the
    # regime real converged conden runs end in) additionally holds the captured
    # reservoir fixed, so a parameter's effect on WHAT condensed is entirely
    # absent. The pinned-species forward-mode tangent disagrees with re-converged
    # finite differences at O(1) (0.91 relative, measured). Refuse by default;
    # this is the same contract Fisher / retrieval-inference through condensation
    # follows project-wide. See docs/condensation_differentiation.md (F1).
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
                "relative; tests/test_condensation_live_tp.py) -- the same reason "
                "Fisher / retrieval-inference through condensation is refused "
                "project-wide (docs/condensation_differentiation.md). Use "
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

    # conden_static is needed below for G_p's conden-row recompute. The refusal
    # (or leading-order warning under allow_frozen_condensation_input_grad) for
    # condensation-active input gradients fires at the top of this function and
    # covers both the in-window and post-pin regimes.
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

    # Parameter map at the fixed point. The conden rows are recomputed at
    # y_star (their y-part; sat tables frozen) so k(p)'s conden rows need not
    # be rebuilt by the caller; photolysis rows ride k(p) frozen (see above).
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

    _warn_poor_convergence(resid_median, fp_err, ensemble_spread)

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

    Reuses the runner's own in-loop photo branch
    (`outer_loop._make_photo_branch`) so the recompute is bit-identical to the
    forward model's photolysis: optical depth -> two-stream RT -> J-rates ->
    write the photolysis rows of `k_arr`. The RT is `lax.scan`-based, hence
    reverse-mode differentiable, so passing the result as
    `steady_state_reaction_sensitivity(..., photo_recompute_k=...)` makes the
    state operator carry `dJ/dy` (the fix for photo-coupled rows; see the
    `photo_recompute_k` parameter).

    Parameters
    ----------
    runner_photo_static
        The runner's internal `_PhotoStatic` (`OuterLoop._photo_static` after
        `_ensure_runner`), NOT the public `PhotoStaticInputs` pytree.
    converged_state
        A converged `JaxIntegState` (the `_runner` output); supplies the frozen
        geometry (`dz`, `pv` T-cross sections, prior `dflux_u`) the branch reads
        alongside the perturbed `y`. `dflux_u` is held at its converged value
        (its second-order self-recursion is not differentiated).

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

    One call replaces the manual geometry splice AND packs every supported
    per-step process the runner's configuration turns on:

    * **atm splice** — `g`/`dzi`/`Hpi`/`top_flux`/`vs` from the converged
      carry (the fields atm-refresh moves), plus a live `vm` recompute when
      `use_vm_mol` (the runner rebuilds vm from the carry geometry every
      step; the setup-time `atm_static.vm` is stale).
    * **condensation (in-window regime)** — when `use_condense` and
      fix_species has NOT tripped: the runner's `CondenStatic` spliced with
      the converged `ProfileVars` (exactly `_make_conden_branch`'s lane
      splice), enabling the conden-row recompute + relax kernels in the body
      map. The saturation tables are T-baked constants — correct for k- and
      Kzz-type inputs; for T-gradients their d(sat)/dT is missing (warned in
      `steady_state_input_sensitivity`). NOTE: free-running conden states
      are typically PSEUDO-steady — VULCAN's own workflow (e.g. the Jupiter
      example) runs the window for a finite time and then pins via
      fix_species precisely because the free system does not converge (the
      negative-clip cycle keeps dt small); expect the stall-branch
      termination and read `audit_adjoint_scope`'s per-cell defect for how
      tight the state actually is.
    * **fix_species regime** — when `fix_species_started`: the pin mask and
      pinned values from the carry (conden no longer fires there). This is
      the regime real converged conden runs end in.
    * **layer-0 pins** — `use_fix_all_bot` (`pv.bottom_n`), `use_fix_sp_bot`
      (`fix_sp_bot_mix * n_0[0]`), and the hycean H2/He pin once tripped.
    * **gas-only / partial balance** — the runner's condensate-aware
      hydrostatic rebalance form.

    Raises `NotImplementedError` for `use_ion` (electron pin + charge balance
    are not implemented in the body map); warns when `diff_esc` is active
    (the escape-flux d(phi)/dy feedback stays frozen — leading-order only).

    Parameters
    ----------
    integ
        The `OuterLoop` whose runner produced `converged_state` (reads
        `_statics`, `_conden_static`, `_refresh_static`, `_hydro_partial`,
        `_cfg`).
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
    # converged on. A hybrid run finishes in phase 1 (central-difference), so
    # the converged state carries hybrid_use_vm==0.0 and the body map must use
    # central diff, not the static upwind flag. Drive use_vm_mol from the
    # converged carry exactly as the runner's body_fn does (no-op for
    # non-hybrid runs: hybrid_use_vm equals the static flag there).
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
    """Scan a run for physics the adjoint's body map drops — BEFORE trusting it.

    The steady-state adjoint linearizes `G = ros2_step (+ renorm) (+ photo)` at
    `y_star`; every other per-step runner process (clip, condensation, charge
    balance, fix-species/boundary pins, atm-refresh feedback) is outside that
    map. Dropping a process is exact when it is inactive or identity at the
    fixed point, and physically wrong when it is still shaping the fixed point.
    This audit answers "which is it, for THIS run" four ways:

    1. **Static config/state checks** (`_adjoint_scope_findings`): every known
       dropped process, classified error / warning / info for this cfg and
       (optionally) converged carry state.
    2. **Per-cell fixed-point defect**: `|G(y_star) - y_star| / y_star` on
       every cell with `ymix >= min_ymix`, built from the SAME `_make_body_map`
       the solver uses. Any unmodeled ACTIVE process — including one not on
       the static list — shows up as a localized defect where it acts. This
       catches what the production `fp_err` (a max-norm relative to the global
       max density) structurally cannot: a pinned bottom row or a conden-
       clamped trace species is invisible next to the deep-column H2 density.
       Cells the runner's ZERO-CLIP owns are excluded and reported separately
       (`n_clip_dead_excluded`): the clip is outside the body map by design,
       and a cell it re-zeroes every step has no fixed point to measure. That
       exclusion is lifted inside the loss footprint, where such a cell is a
       hard error rather than noise.
    3. **Stale-geometry check** (when `final_state` is given): verifies the
       caller spliced the converged refresh fields into `atm`
       (`atm_static._replace(g=final.g, dzi=final.dzi, Hpi=final.Hpi,
       top_flux=final.top_flux, vs=final.vs)`).
    4. **Loss footprint** (when `loss_fn` is given): the worst defect among
       the cells the loss actually reads (|y* dL/dy| within
       `_AUDIT_LOSS_FOOTPRINT_FRAC` of its max) — a defect outside the
       footprint biases the gradient only through the solve; one inside it
       biases the answer directly.

    Interpretation: well-converged bulk cells read per-cell defect
    ~1e-9..1e-6. Defects of 1e-2..0.3 are ambiguous (WARNING): a weak
    unmodeled process, or slow trace species still creeping at the forward
    convergence tolerance — the healthy HD189 fixture reads 6.5e-2 on
    upper-atmosphere NH3/N2H2 cells while its CH4 gradient FD-validates at
    0.7%. Defects above `_AUDIT_DEFECT_ERROR` (0.3) are structural (ERROR):
    a converged state cannot move O(1) under one probe step unless the
    runner's map contains a process this one lacks. A defect inside the loss
    footprint is an ERROR at any cause — it biases that loss's gradient
    directly.

    Parameters mirror `steady_state_reaction_sensitivity` where shared.
    `cfg` defaults to the global `vulcan_cfg` module (the runner's default);
    pass the run's cfg namespace for `make_config`-driven runs. `final_state`
    is the converged `JaxIntegState` (sharpens the conden/pin checks and
    enables the stale-geometry check). `species` labels the worst-cell table;
    when omitted it is taken from the import-locked network if the width
    matches.

    Returns a dict: `findings` (list of {code, severity, message}),
    `max_rel_defect`, `worst_cells`, `fp_err_global`,
    `loss_footprint_defect` (None without `loss_fn`), and `ok` (True when
    there are no error-severity findings; warnings do not clear it to False).
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

    # Cells the runner's ZERO-CLIP controls carry no meaningful fixed-point
    # defect. `outer_loop._make_clip_fn` sets any post-step value landing in
    # [nega_cut, pos_cut) to exactly 0, and that clip is deliberately outside
    # the body map ("identity-a.e." in _make_body_map). Where it fires, the
    # runner and the body map disagree by construction: the runner zeroes the
    # cell, the map keeps the raw (usually slightly negative) step. Such a cell
    # has NO fixed point at all -- it is re-zeroed on whichever sign the step
    # lands on -- so |G-y|/y there measures the clip, not a dropped physical
    # process, and no probe step can make it small (it GROWS with body_dt).
    #
    # min_ymix cannot exclude these: the clip window is ABSOLUTE (cm^-3) while
    # min_ymix is a MIXING RATIO. On a cold upper atmosphere (W39b tabulated
    # top, M ~ 7.5e12 cm^-3) ymix 1e-16 is y ~ 1e-3 cm^-3 -- three orders
    # INSIDE the |nega_cut| = 1 window -- so the pressure-relative floor waves
    # through exactly the cells the clip owns. Detect them mechanistically
    # from where the clip WOULD fire, rather than by another threshold.
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
        w = np.abs(v * y_np)  # log-space cotangent magnitude — the adjoint RHS
        # A non-finite cotangent must be reported, not swallowed. Every `>`
        # against NaN is False, so `w.max()` being NaN would leave `foot` empty
        # and the audit would declare a clean footprint on a loss whose gradient
        # is already poison. Check before thresholding.
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
        # A clip-dead cell is excluded from `rel` (above), so it would read as
        # a clean 0.0 here. The loss footprint is exactly where that leniency
        # is NOT allowed: the body map linearizes those rows as identity while
        # the runner zeroes them, so a loss reading one gets a directly wrong
        # gradient. Score the footprint on the UNMASKED defect.
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


def _scan_score(info: dict) -> tuple[float, float, float, float, float]:
    """Sort key for body-map dt candidates.

    Primary target is the median Krylov residual. Spread and max residual break
    ties; fixed-point and null-quality diagnostics are later tie-breakers.
    """
    return (
        float(info.get("resid_median", np.median(info["resids"]))),
        float(info["ensemble_spread"]),
        float(info["resid"]),
        float(info["fp_err"]),
        float(info["null_quality"]),
    )


def scan_body_dt_reaction_sensitivity(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    *,
    compo_array: jnp.ndarray,
    dz: jnp.ndarray,
    body_dt_candidates: Sequence[float] = BODY_MAP_DT_CANDIDATES,
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
    """Run `steady_state_reaction_sensitivity` over several body-map steps.

    This is the accuracy-oriented entry point. It compiles and solves one
    adjoint problem per candidate `body_dt`, suppresses warnings for discarded
    candidates, and returns the lowest-residual/lowest-spread result. The
    selected diagnostics include `body_dt_scan`, a list of per-candidate
    residuals, spreads, fixed-point errors, null-quality values, and any
    candidate warnings/errors.

    The scan can reduce solver-regime error. It cannot remove the separate
    finite-tolerance mismatch between the exact fixed-point adjoint and the
    state where the forward integration stopped; for that, converge or polish
    `y_star` more tightly before calling this function.
    """
    if len(body_dt_candidates) == 0:
        raise ValueError("body_dt_candidates must contain at least one value.")
    photo_recompute_k = _resolve_photo_recompute_k(
        photo_recompute_k,
        k_arr,
        net,
        solver_map,
        runner_photo_static=runner_photo_static,
        converged_state=converged_state,
        integ=integ,
    )

    best = None
    scan_rows: list[dict] = []
    failures: list[str] = []
    for body_dt in body_dt_candidates:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                grad, info = steady_state_reaction_sensitivity(
                    loss_fn,
                    y_star,
                    k_arr,
                    atm,
                    net,
                    compo_array=compo_array,
                    dz=dz,
                    body_dt=float(body_dt),
                    solver_map=solver_map,
                    photo_recompute_k=photo_recompute_k,
                    body_terms=body_terms,
                    lgmres_inner_m=lgmres_inner_m,
                    lgmres_outer_k=lgmres_outer_k,
                    lgmres_maxiter=lgmres_maxiter,
                    lgmres_cycles=lgmres_cycles,
                    rtol=rtol,
                    n_solves=n_solves,
                    return_info=True,
                )
            except Exception as exc:
                message = f"body_dt={float(body_dt):.3e}: {exc!r}"
                failures.append(message)
                scan_rows.append(
                    {
                        "body_dt": float(body_dt),
                        "error": repr(exc),
                        "warnings": [str(w.message) for w in caught],
                    }
                )
                continue

        info = dict(info)
        info["warnings"] = [str(w.message) for w in caught]
        score = _scan_score(info)
        scan_rows.append(
            {
                "body_dt": float(body_dt),
                "resid": float(info["resid"]),
                "resid_median": float(info["resid_median"]),
                "ensemble_spread": float(info["ensemble_spread"]),
                "fp_err": float(info["fp_err"]),
                "null_quality": float(info["null_quality"]),
                "pair_antisym": float(info["pair_antisym"]),
                "score": score,
                "warnings": info["warnings"],
            }
        )
        if best is None or score < best[0]:
            best = (score, grad, info)

    if best is None:
        joined = "; ".join(failures)
        raise RuntimeError(
            "All body_dt candidates failed in scan_body_dt_reaction_sensitivity: "
            f"{joined}"
        )

    score, grad, info = best
    info = dict(info)
    info["body_dt_scan"] = scan_rows
    info["selected_score"] = score
    _warn_poor_convergence(
        float(info["resid_median"]),
        float(info["fp_err"]),
        float(info["ensemble_spread"]),
    )

    if return_info:
        return grad, info
    return grad
