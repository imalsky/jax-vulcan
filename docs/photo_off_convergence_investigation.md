# Photo-off W39b convergence: investigation and what it takes to reach steady state

Date: 2026-07-15. Scope: the jwst-tool / vulcan-retrieval "Fisher forecast requires
photolysis ON" gate, and whether the forward-mode AD can work photo-off. All runs are the
production W39b column (`config.WIDE`, nz=150, SNCHO_photo network), CPU, `vulcan` env.
Diagnostic scripts: `vulcan-retrieval/validation/diag_photo_off_{convergence,steady_state,culprit,sulfur}.py`.

## TL;DR

1. **This was never a differentiation problem.** Forward-mode AD works wherever the *primal*
   converges. The gate's stated reason ("the warm-started jvp is under-relaxed/unstable") is a
   mislabel: the real issue is that the photo-off **forward model did not reach a steady state**
   with the default profile. Removing photolysis does not make a fixed point harder to
   differentiate; it removes the fast damping that let the *relaxation solver* reach it.
2. **Photo-off IS reachable** with the right convergence settings. The default WIDE profile
   stacks three photo-on-tuned choices that each break photo-off. Fixing all three converges the
   column in ~120 steps to a nearly-physical steady state (atom conservation <0.03% for H/O/C/N,
   ~2.5% for S).
3. **A real latent issue found along the way:** with `use_hybrid_vm_mol` on (default), the
   photo-off `_runner` stops at ~`count_min+2000` steps via the hybrid phase-flip's budget reset,
   and `jwst_tool/forward.py::_check_converged` (which only tests `accept_count >= count_max`,
   static 30000) **accepts that non-steady state as converged** and takes a jvp on it. That is
   how a photo-off "Fisher forecast" could silently run on garbage.

## What photo-ON looks like (the control)

Converges in **1301 steps** (t=4.9e8 s), longdy 0.059 (loose-branch gate), atom loss ~1e-6.
Sulfur is stable. Photolysis (SO2 photolysis etc.) fast-couples/damps the sulfur network, so the
column locks onto its fixed point quickly, before any slow drift matters.

## What photo-OFF does with the default WIDE profile

Cold `run_diag` (the path `converged_y` -> the Fisher jvp actually uses), at theta0 and a ring:

| theta | accept_count | final longdy | longdy_seen_min |
|---|---|---|---|
| theta0 | 2124/30000 | 1.0 | 9.98e-3 |
| lnZ=+0.3 | 2138 | 3.76e8 | 1.04e-2 |
| lnZ=-0.3 | 2155 | 16.4 | 4.57e-2 |
| dT=+100K | 287 | 0.067 | 2.49e-3 |

It **approaches** a fixed point (`seen_min` ~0.002-0.05, below the 0.1 gate) but **cannot hold
it** — longdy bounces back up (1 to 3.8e8). A marginally-stable / oscillatory fixed point
(`rho(G_y) ~ 1`), which is exactly what removing the photolytic damping predicts. Warm-starting
from the photo-ON converged column does **not** help: re-converging from it moves the state 78%
in VMR (`self-consistency = 0.78`).

## Root cause: three compounding, photo-on-tuned settings

The column is **closed** (`use_topflux/botflux=false`, `diff_esc=[]`, `use_settling=false`), so
any atom loss is numerical/dynamical, not physical escape. The `where_varies_most` culprit
diagnostic + a lever sweep isolate three causes:

1. **`dt_max=1e17` (uncapped).** The WIDE profile leaves `dt_max` at the default `runtime*1e-5 =
   1e17`; production retrieval caps it at `1e11`. Adaptive-Ros2 `dt` balloons to 1e16-1e18 s and
   the huge-`dt` steps amplify every stiff mode -> catastrophic **all-element** blow-up (H -91%,
   O -65%, C -39%, N -99%, S -94%), with nitrogen species (HCN/NH3/HNCO/NH2/CN) driving longdy.
2. **Stiff sulfur allotropes.** With `dt` capped, the pathology collapses to **sulfur only**:
   S3/S2/S2O/HS2/CS2 drive longdy and S drains ~84% (O ~14%; H/C/N fine). The S<->S2<->S3<->S4<->S8
   interconversion is slow (>=1e15 s) and, without photochemistry, undamped -> it never settles.
   This is a documented sulfur-allotrope issue; the condensation path already handles it with
   `conver_ignore(S/S2/S3/S4)` + `mtol_conv=1e-15` (+ pinned S8).
3. **Hybrid/upwind molecular diffusion (`use_hybrid_vm_mol`/`use_vm_mol`, default on).** Turning
   it off (pure central diffusion) alone flips the baseline from catastrophic failure to (dirty)
   convergence.

## The lever sweep (photo-off, theta0, cold `run_diag`)

Atom-loss vector order is [H, O, C, N, S]; `held` = converged below the 0.1 gate short of
count_max.

| config | steps | longdy | atom_loss (H,O,C,N,S) | held | note |
|---|---|---|---|---|---|
| A baseline (no recipe) | 2124 | 1.0 | -.91,-.65,-.39,-.99,-.94 | no | hybrid cap; forward accepts it |
| B +conver_ignore(S allotropes) | 2124 | 1.0 | identical to A | no | no effect (N-driven, not S) |
| C B + mtol_conv=1e-15 | 2122 | 46 | -.99,-.90,-.82,-.22,-.99 | no | worse (majors blow up) |
| **D C + dt_max=1e11** | **222** | **0.053** | -7e-5,-2e-3,-4e-4,+5e-4,**-0.13** | **yes** | genuine convergence; S off 13% |
| E hybrid vm_mol OFF only | 130 | 0.096 | +.006,+.037,+.12,-.16,-.06 | yes | marginal/dirty (C,N off ~12-15%) |
| **F recipe + hybrid OFF** | **121** | **0.0038** | **-3e-5,+3e-4,+2e-6,-2e-4,-0.025** | **yes** | clean: all <0.03% except S -2.5% |

**Key reads:** `conver_ignore` or `mtol_conv` alone do nothing / hurt (B, C) because the
uncapped-`dt` blow-up is not sulfur-limited. Capping `dt` is necessary (D). The cleanest state
(F) needs **all of**: `dt_max=1e11` + `conver_ignore` the sulfur allotropes + `mtol_conv=1e-15` +
central diffusion (`use_vm_mol=False`). That converges in 121 steps with H/O/C/N conserved to
<0.03% and S off only 2.5% -- effectively the physical steady state.

## What is needed to reach a photo-off steady state (answer)

```
use_vm_mol      = False        # (and use_hybrid_vm_mol=False) -> central diffusion
dt_max          = 1e11         # cap the adaptive step (prevents the all-element blow-up)
conver_ignore  += [S,S2,S3,S4,S8,S2O,HS2,CS2]   # the runner's slow-trace mechanism
mtol_conv       = 1e-15        # close the sub-femto allotrope drift branch
```

This is essentially the **same recipe the condensation path already documents**, plus the
`dt_max` cap and central diffusion. Residual caveat: ~2.5% sulfur drift, so the converged state
is slightly sulfur-lean -- worth quantifying before trusting a *sulfur-species* derivative.

## Implications for the gate and for AD

- **Photo-off AD is not fundamentally blocked.** Once the primal converges (recipe above), the
  warm/cold jvp has a real fixed point to linearize. The blocker was 100% forward-model
  convergence, which is fixable -- consistent with Isaac's goal that AD work photo-on/off.
- **The current gate is right to refuse the DEFAULT photo-off profile, but for the wrong stated
  reason.** Honest wording: "photo-off does not converge with the default profile (stiff sulfur
  allotropes + uncapped dt + hybrid diffusion); enable the convergence recipe, then the forecast
  is available." Not "the tangent is unstable."
- **Fixed the false-convergence accept (independent of photo) -- DONE 2026-07-15.**
  `jwst_tool/forward.py::_check_converged` previously treated `accept_count < count_max` as
  converged, so a hybrid-phase-flip / stall termination (`accept_count` ~ count_min+2000,
  longdy >> gate) was accepted. Now it gates on the runner's own `longdy` vs `yconv_min`.
  Plumbing: `vulcan_chem.converged_y(return_longdy=True)` returns `(y, accept_count, longdy)`
  (the existing `return_diag` 2-tuple is untouched, so the 5 retrieval callers are unaffected);
  `build_chem_model` now exposes `yconv_min`. Verified: photo-ON passes (longdy 0.059 < 0.1),
  photo-OFF is caught (longdy 1.0 >= 0.1) where the old check accepted it.
  **Still open:** the retrieval's own init-path convergence detection
  (`retrieval_forward.chem_solve_cold_diag` / the SMC init gate in `pipeline.py`) has the same
  latent bug on the cold/full-cap path (the warm mutation path is partly protected by
  `warm_count_max=1500 < count_min+2000`). Thread `longdy` through there too as a follow-up.

## jvp-vs-FD on the recipe-converged photo-off state (done)

`diag_photo_off_ad.py`: warm (production Fisher path) and cold jvp vs re-converged central FD,
top-1%-|FD|-cell metric, on the config-F converged state (accept_count 121).

| param | WARM corr / med-rel | COLD corr / med-rel |
|---|---|---|
| lnZ | 0.9995 / 26% | 0.9996 / 35% |
| C/O | 0.9997 / 6.5% | 0.9999 / 6.3% |
| lnKzz | 0.686 / 101% | 0.957 / 14% |
| dT | 1.0000 / 115% | 1.0000 / 112% |

Reads: the tangents are **directionally excellent** (corr ~1, except warm lnKzz 0.69) but
**quantitatively unreliable** -- magnitudes off 25% to ~2x -- with **C/O the only clean one
(~6%)**. The **warm path is worse than cold** (lnKzz 0.69 vs 0.96), the under-relaxation
signature: warm-starting truncates the tangent recurrence, and on the marginally-stable
(`rho(G_y) ~ 1`) photo-off fixed point that under-relaxes. Note the ambiguity: on a
marginal, ~2.5%-sulfur-drifting fixed point NEITHER the through-loop jvp NOR the re-converged
FD is a clean truth, so some of the disagreement is the FD being noisy, not the jvp being wrong.

**Net:** photo-off is now *convergent* and the tangent is *directionally* usable, but not
quantitatively trustworthy through the loop -- so the gate's spirit ("don't trust the photo-off
Fisher forecast") is partly vindicated, while its *mechanism* claim is still wrong. The
principled fix for accurate photo-off tangents is the **forward implicit solve**
`(I - G_y) s = G_theta` (designed but not shipped -- reuses the reverse-adjoint machinery): it
relaxes exactly regardless of `rho(G_y)`, so it should recover the true sensitivity where the
warm through-loop jvp under-relaxes. That is the next real lever, not a convergence knob.

## Follow-ups (not done)
- Chase the ~2.5-13% sulfur drift: is it the clip in the residual oscillation, or an S8/allotrope
  transient? A sulfur-conserving converged state may need the S8 pin (condensation path).
- Check recipe robustness across the theta box (the ring showed strong theta-dependence), since a
  Fisher/FD solve perturbs theta and every perturbed solve must also converge.
- The hybrid `use_vm_mol` default (recent vm_branch port) destabilizing photo-off is a data point
  for the pending "hybrid default needs HPC re-baseline" question.
