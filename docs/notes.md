# Implementation notes — running experiments & implementation log

This is the running implementation / experiments log for VULCAN-JAX,
in rough chronological order. It opens with the 2026-06-11 parity-gap
closure pass (what still needs clarification or off-machine verification,
and the problems hit along the way) and continues through the later
differentiability work: end-to-end forward/reverse-mode AD, the on-graph
atmosphere builder, and the reverse-mode adjoint accuracy campaigns
(dated sections run through 2026-07-04). Newer entries are appended at the
end; earlier entries are kept as-is for provenance, so where a later dated
section supersedes an earlier claim, the later one wins.

## Needs HPC verification (cannot be done on this CPU-only host)

> **Runner:** with the PBS (NAS GH200) system down, both GPU items below run
> on the edge A100 via `vulcan-emulator/supercomputer_cmds/run_gpu_benchmark.sh`
> (sbatch, or plain `bash` on any GPU box). It does the tiled sweep, then an
> untiled Fix-B probe in a fresh process, sized to the visible GPU memory
> (512 lanes on >=70 GiB, 256 on a 40 GB A100). `gpu_benchmark.py` now prints
> a per-batch `peak GiB` column (XLA allocator `peak_bytes_in_use`) — that
> column is the Fix-B verdict. An A100 pass transfers to the GH200: the
> transient scales with batch width, not GPU model.

1. **Fix B on-device confirmation.** The chunked Jacobian assembly
   (`chem.py::_JAC_CHUNK_REACTIONS`, `lax.scan` with `unroll=1`) is designed
   to drop the batch-512 vmap transient from ~60 GiB to ~1/7 of that, but the
   XLA-undo risk (scan fused back into one flat scatter) only shows up on
   device. Run the untiled probe and confirm `peak GiB` stays far below the
   un-chunked prediction (~42-61 GiB at 512 lanes). If XLA un-does the
   chunking, add `jax.checkpoint` on the chunk body or shrink the chunk size.
2. **Untiled vs tiled.** The probe phase measures a true batch-wide vmap; the
   sweep uses `--device-batch 128` tiles. Compare profiles/s — if untiled
   fits and is faster per planet, raise the default device batch.
3. **Pinned step counts may shift on GPU.** Fix B changes float summation
   order inside the Ros2 LHS. On this host the HD189-EQ run still converges
   in exactly 606 steps, but the `gpu_benchmark_fix` reference counts
   (const_mix 4087 / isoEQ 1429 / HD189-EQ 606) were measured on the GH200
   and may wobble by a few steps there. Tolerance-based checks are unaffected.
4. **Batched-photo device sizing.** Per-lane T-dep cross sections cost
   `(n_absp_T + n_br_T) * nz * nbin * 8` bytes (~25–40 MB/lane for a default
   NCHO photo config) on top of the existing carry. Photo-on sweeps will need
   smaller `--device-batch` values than photo-off ones; measure before
   committing to a default.

## Known costs / restrictions of the new batched paths (by design)

- **Batched photo throughput.** Under `vmap`, the `lax.cond` photo gate
  executes the photo branch every body iteration on every lane (results are
  selected away when not due), so the `ini_update_photo_frq` cadence saves no
  compute in `run_batch` — batched photo is much slower per accepted step
  than the single-profile path. Correctness is unaffected (same situation as
  the conden branch). If batched-photo sweeps become routine, consider
  restructuring the gate so the photo work amortizes.
- **Same-star restriction.** A photo batch must share star, wavelength grid,
  network, and cfg scalars; only the T-P profile varies. `prepare_runstate`
  enforces this (`nbin`/`din12_indx`/`bins`/`sflux_top` identity) — but only
  for profiles routed through the *same* `OuterLoop` instance. States stacked
  from a different instance bypass the guard (same enforcement level as the
  pre-existing nz/toggle-combo restrictions).
- **NH3 cold-trap freeze.** Master recomputes `argmin(sat_mix['NH3'])` every
  call; VULCAN-JAX freezes it at setup. Bit-equivalent today because the
  inputs (sat_p, Tco, n_0) are time-invariant during a run — if any of those
  ever becomes time-varying, the frozen index diverges. Pre-existing port
  semantics, unchanged by the batching work.
- **Batch-constant conden config.** `CondenStatic`'s cfg-derived fields
  (`coeff_per_re`, `r_p`/`rho_p`-based masses, active flags) stay
  closure-baked. A batch mixing different `use_relax`/`r_p`/`rho_p` configs
  would silently use the first profile's values — same contract as every
  other cfg knob in the batched runner. Flagged here in case the emulator
  ever wants per-profile particle radii.

## Problems encountered

- **Stale non-editable install (again).** `import vulcan_jax` resolved to
  site-packages (0.1.13 release install), not the checkout — the exact trap
  from the 2026-06 validation campaign. Re-installed editable before any
  testing. `tests/conftest.py` would have caught it at collection, but
  one-off scripts bypass conftest.
- **`network._configured_extra_species` was dead and dangerous.** It tried
  `importlib.import_module("vulcan_cfg")` (top-level name — a flat-layout
  leftover that always fails inside the package) and would, if a stray
  top-level `vulcan_cfg.py` were importable from cwd, silently intern
  cfg-referenced species into the parsed network, changing `ni`/`spec_list`
  vs master. Deleted per the no-dead-code rule.
- **Default photo config has no T-dependent cross sections.** `T_cross_sp`
  defaults to `[]`, so the default HD189 photo run uses room-T cross sections
  only (master behaves identically). The first draft of the batched-photo
  test was vacuous because of this; it now sets `T_cross_sp = ["H2O"]`
  explicitly. Worth remembering for any "are T-dep cross sections exercised?"
  question.
- **The two gpu_benchmark.py copies had drifted.** The emulator's
  `supercomputer_cmds/` copy had a CUDA-plugin log-mute fix the `examples/`
  copy lacked. Both now identical (tiling + log mute); keep syncing them.
- **Master cannot run its own Earth example** (`'Ar' is not in list`,
  `build_atm.py:200`, reproduced end-to-end in a sandbox copy) — and would
  NaN-poison `atom_loss` for atoms with no carrier species even if Ar were
  removed from `const_mix` only. Don't treat the Earth cfg as a usable
  oracle; the only honest test is parity-of-failure, now upgraded to an
  upfront validation error.
- **Master-side S4 asymmetry.** Master's `sat_sp_list` omits S4, so a
  master-vs-JAX oracle run with S4 in `condense_sp` crashes on the master
  side. Remember when designing condensation oracle tests.
- **`CondenStatic` as a jit argument breaks.** Its Python-bool gate fields
  (`h2o_active`/`nh3_active`) become tracers if the whole NamedTuple is
  passed as a jit argument (TracerBoolConversionError). Production always
  closes over the static and splices only array fields via `_replace`; tests
  must follow the same pattern.

## Suite-cost note

`tests/test_nh3_conden_batch_subprocess.py` is the slowest new test: a fresh
subprocess must parse + codegen the 1141-reaction lowT-Jupiter network and
compile three runners (~10 min cold on this laptop; the JAX persistent cache
under `~/.cache/jax_vulcan` makes identical reruns much cheaper, but any edit
to the carry pytree invalidates it). It is the only end-to-end coverage of
batched NH3 condensation, so it stays; if suite time becomes a problem, gate
it behind a marker rather than deleting it.

## Explicitly NOT implemented (per instruction / by design)

- **Open-ended condensation.** Master has no such functionality — unknown
  condensates are a silent no-op there. VULCAN-JAX keeps the explicit
  formula set and now fails at validate time with the supported sets listed.
- **In-process structural-config hot-swap** (gap 3): `network` / `com_file` /
  `atom_list` stay import-locked; `$VULCAN_JAX_NETWORK` + subprocess driver
  remain the supported paths.
- **Earth inert-background gases**: not live master physics (see above);
  the Earth example config ships verbatim and is rejected with a clear
  upfront error.

## Post-review additions (adversarial review round, same pass)

A three-reviewer audit (goal completeness vs master, diff correctness,
cleanliness) with adversarial verification confirmed three real issues, all
fixed (see the parity report's "Post-review fixes"): the `use_ion`
electron-row freeze inside the Ros2 stages was never ported (now wired
through the `fix_mask` plumbing), the legacy `(var, atm, para)` entry crashed
with non-empty `T_cross_sp` (pv now falls back to the baked photo static),
and the same-star guard was vacuous when the runner was pre-built by a
single-profile run (`_sflux_top_ref` now recorded at bake time).

Remaining known items from that review, deliberately not changed:

- **`use_print_delta` is declared but never consumed.** Master prints the
  largest-truncation-error species at print cadence inside its solver; a
  per-step host print is impractical inside the JIT'd runner. The knob stays
  declared because `vulcan_cfg.py` intentionally mirrors master's config
  surface; treat it as a dropped diagnostic. Wire it into the chunked-runner
  print path if anyone misses it.
- **H2S conden reaction fails late.** `condense_sp=['H2S']` passes upfront
  validation (sat-only tier, master-legal), but a network that also carries
  an H2S conden *reaction* still hits the `NotImplementedError` at runner
  build, after pre-loop setup. The parsed `Network` carries `is_conden` and
  reaction text, so an upfront cross-check is possible if this ever bites.
- **The pass is uncommitted.** Commits are Isaac's to run; everything above
  lives in the working tree on `main` (last commit: v0.1.13 release).
- **The `use_ion` e-row freeze has no end-to-end test.** No vendored network
  carries ion species ('e' is not in any shipped network), so the freeze can
  only be exercised kernel-level (it reuses the fix_mask row-pin plumbing,
  which is tested). Same coverage status as other live-but-non-default
  branches per the scope rule. If an ion network is ever vendored, add a
  solo run asserting e is constant within a step before charge balance.

## End-to-end AD: forward works, reverse does not (2026-06)

Status after a full investigation (diagnostic scripts live in
`../jax_paper/scripts/`: `fig_kzz_jvp_validate.py`, `diag_reverse_adjoint.py`,
`diag_fixedpoint_adjoint*.py`).

**Forward-mode (`jvp`/`jacfwd`) WORKS end-to-end through the converged runner.**
The only blocker was the analytical-Jacobian `y^0` NaN-jvp at clipped cells;
after that one value-identical fix, a single forward pass
differentiates the whole production integration. Validated d(VMR)/d(ln Kzz) on a
complete HD189 run (photo on, ~1300 steps) vs re-converged centered FD:
correlation >0.9999, <0.1% on responding levels. **This is the supported route
for end-to-end production gradients.** Cost: one pass per input direction.

**Reverse-mode (`vjp`/`grad`) does NOT work end-to-end on production runs.** The
true gradient exists (FD gives clean O(1) values), but every reverse path we
tried fails because the steady-state Jacobian `J = df/dy` is simultaneously
(a) **singular** — a closed column conserves each element, so `J` has a
conserved-mass null space — and (b) **severely ill-conditioned** — chemistry is
stiff, production/loss terms ~1e28 cancel at steady state (the `segment_sum`
residual at the converged state is ~1e21, cancellation level 1.0). What was tried,
all measured on HD189 (CH4 mid-layer loss, FD truth dL/dlnk_{r13} = -0.565):

1. **Residual-IFT adjoint** (`steady_state_grad._solve_adjoint`, block-Thomas
   defect-correction). Adjoint residual stalls near 1e5; gradient = +876 (wrong
   sign and ~1500x off). Works only on the well-conditioned synthetic problem in
   `tests/test_steady_state_grad.py` (~1e-9 there).
2. **Matrix-free LSQR pseudoinverse** of `J^T` (`scipy.sparse.linalg.lsqr`,
   matvec=vjp, rmatvec=jvp). Stagnates: `istop=7`, `lambda~6e-26`, gradient ~0.
   `J`'s 1e21-scale entries make it hopeless without preconditioning.
3. **Fixed-point (iteration-map) adjoint** `(I - dG/dy)^T z = v` on the body map
   `G(y) = n_0*normalize(ros2_step(y))`. This is the principled fix: `G`'s
   fixed point is y* (fp_err ~1e-9), and `(I - dG/dy)` rides the step's
   regularized `(I/gamma*h - J)` solve, so it is far better conditioned than the
   bare `J` (this is exactly why forward-mode works). BUT: the map is
   **non-contractive** — a Neumann iteration diverges (gradient went
   -3.9 -> -6.6 -> +0.56 -> -26 over 20/60/200/600 iters) — and `(I - dG/dy)`
   inherits the conserved-mass null space, so it needs a Krylov solve. Every
   matvec is a full reverse pass through one Ros2 step (~10350-dim operator), so
   BiCGSTAB/GMRES are prohibitively slow to even compile here, and the singular
   operator stalls them. Not practical as-is.

**Why forward escapes it:** forward-mode never forms `J^{-1}`; it propagates the
tangent through the integrator's own regularized stage solves. Reverse-mode can
only inherit that regularization through the fixed-point adjoint, which then
needs a *preconditioned* Krylov solver (e.g. reusing the step's block-Thomas
factor as preconditioner). That is the natural future-work path to practical
reverse-mode production gradients; it was out of scope for this pass.

### 2026-06-16: reverse-mode steady-state adjoint — BREAKTHROUGH to ~6%, gold (<1%) still open

Big picture after a long push: the **fundamental wall is broken** — reverse-mode
at the converged steady state now produces a **correct-sign, right-ballpark
(~6%)** gradient on a real closed column, where every earlier attempt failed
*catastrophically* (wrong sign `+876`, NaN, or divergence to `1e57`). But
**verified gold accuracy (<1%) was NOT reached** this session; it is a genuine
numerical-stiffness research problem, not a tuning detail. Full play-by-play so a
future session doesn't re-walk the dead ends.

#### What worked (the breakthrough): log-space + deflation + Krylov on the body map

Fixed-point adjoint `(I - dG/dy)^T z = v`, `dL/dlnk = z^T dG/dk`, with THREE
coupled ingredients (each fixes one documented prior failure; all three needed):

1. **Log-abundance scaling.** Solve in `eta = ln y`. Similarity transform
   `D^{-1}(I-dG/dy)D`, `D=diag(y*)`: same spectrum, but operator norm `1e6 -> ~294`
   and cotangent `|v| 3e-12 -> |v_eta| 0.43` (both O(1)). This is what LSQR/GMRES
   never had. Identities (elementwise): `A_eta z = z - y* .* vjp_Gy(z/y*)`,
   `v_eta = y* .* v`, `c_eta = y* .* (compo[:,e].*dz)`, recover `lambda = z/y*`,
   `dL/dlnk = (k .* vjp_Gk(lambda)).sum`.
2. **Conservation deflation** with analytic atom-count vectors `c_e=compo[:,e]*dz`
   (QR -> projector). Only the LEFT-null is needed: the right-null cancels from
   atom-conserving-knob gradients since `c_e^T df/dk = 0`. In log space the null
   quality `||A c||/(||A||||c||) -> 0` and `||Pv||/||v|| = 1.0` (verified).
3. **Krylov/Neumann, not raw Neumann.** After deflation, projected Neumann finally
   *converges* (it diverged before).

**Validation** (HD189, photo off; loss=`log10(CH4 mixing)` mid-layer; FD truth
`dL/dlnk_{r13} = -0.565`): projected Neumann `N=200` gives `-0.598` (**5.8%,
correct sign**). That is the breakthrough datum.

#### Why it stalls at ~6% — the genuine obstacle

The error is dominated by **slow near-conserved chemical modes**: ODE eigenvalues
`mu -> 0^-` sitting just off the exact conserved null. The gradient `v` lives
substantially IN this slow subspace, so you must resolve it — but it is what makes
the operator ill-conditioned. Hard facts learned (do not re-derive):

- **`dt` is NOT a free conditioning knob.** Larger body-map `dt` was hoped to damp
  slow modes, but it is a *danger zone*: at `dt~1e11` the step's own implicit solve
  `(I/(gamma h) - J)` goes near-singular (`1/(gamma h) ~ mu_slow ~ 3e-11`) and the
  adjoint **diverges** (resid `-> 1e57`). Safe regime is moderate `dt~1e8`.
- **`n_0*normalize(step)` body map BIASES the answer.** Its renormalization
  projects out molecule-count changes, so number-changing reactions get the wrong
  gradient (drifts `-0.598 -> -0.667` with more iterations, *away* from FD). Its
  fixed point is tighter (`fp_err 5e-9` vs bare `1.3e-4`) but it is not the true
  steady-state adjoint.
- **The BARE step is unbiased but Neumann-UNSTABLE.** Its `dG/dy` has an unstable
  total-density-per-layer mode (the one normalization removes), so Neumann diverges
  at every `dt`. It would need GMRES + deflation of the `nz` total-density modes.
- **The `~1e21` residual is NOT cancellation noise.** `||f(y*)||=1.56e21` is
  *identical* for the segment-sum and the codegen RHS, so it is the genuine
  (relative ~1e-7) residual, not a formulation artifact — switching RHS does not
  help. (Disproved the noise hypothesis directly.)

#### The residual-IFT route (cleaner formulation, also stalls on stiffness)

`J^T lambda = v` with `J = df/dy` from one vjp of the steady-state residual
`f = chem_rhs + diffusion` ("single RHS eval"). Unbiased; only null space is the
element-conservation one. Components ALL verified correct (`adj_verify.py`):
`||J^T c_e||/||c|| ~ 1e-5`, block-Thomas inverts `M = reg*I - J` (rel err 4e-4 at
`reg=1e-2`), vjp `J^T` matches the analytical block Jacobian to `6e-14`. Solver
lessons:

- **Left-preconditioning minimizes the WRONG metric.** `Minv` eigenvalues span
  `1e-10..100`, so the preconditioned residual down-weights exactly the fast modes
  the gradient needs -> small preconditioned resid, huge true resid, garbage
  gradient. Use **right-preconditioning** (GMRES residual == true residual).
- **`reg` (preconditioner shift) faces an unfixable tradeoff:** the conserved null
  forces `cond(M) ~ |mu_max|/reg`; resolving the slow mode needs `reg ~ mu_slow`
  which makes `cond(M) ~ 3e20` -> block-Thomas is float64 garbage. No single `reg`
  both keeps `Minv` accurate and separates the slow mode from the null.
- **Orthogonal deflation of the slow modes BIASES the gradient** (it drops their
  real contribution). The correct technique is **augmented GMRES** (keep the slow
  modes in the search space). Implemented (power iteration on `Minv` for the slow
  subspace + augmented Krylov least-squares), but with `n_slow=24, m=300` the true
  residual only reached `~0.6` — the slow subspace `v` lives in is high-dimensional
  / the real-block power iteration misses complex slow pairs. This is the open edge.

#### THE working route to gold: bare solver-map + GMRES (not Neumann)

The residual-IFT (separate `reg*I-J` preconditioner) cannot cluster the slow modes
— the bordered/saddle fix removed the inconsistency but restarted GMRES still
diverged (`0.56 -> 2.14`). The reason is structural: **the solver-map
`(I - dG/dy)^T` is the integrator's EXACTLY-preconditioned `J`** (block-Thomas at
the step `dt`, composed exactly inside the Ros2 step) — far better conditioned than
any separately-computed `Minv*J^T`. You cannot replicate it with a standalone
preconditioner. So ride the solver-map.

Use the **BARE Ros2 step** (`G(y) = ros2_step(y,k)`, NOT `n_0*normalize(...)`):
the normalize biases molecule-count-changing reactions (drifts away from FD). The
bare step's only defect is an unstable total-density mode that diverges *Neumann* —
but **GMRES handles indefinite spectra**, so just use GMRES. Recipe: log-space
`A_eta`, deflate the conserved null `c_e`, host-side restarted GMRES (double
Gram-Schmidt), gradient `(k .* vjp_Gk(z/y*)).sum`. Do NOT deflate the total-density
mode (that re-introduces the normalize bias); let GMRES resolve it.

**This converges, unbiased, toward FD** (HD189 photo-off, `adj_solvermap_gmres.py`,
`dt=1e8`): with **restarted** GMRES(m=300) the gradient OSCILLATES around FD
(`r13` ranges `-0.43..-0.63` vs FD `-0.565`, residual bounces `0.1..1.6`) — classic
restarted-GMRES stagnation on an indefinite operator. **LGMRES** (scipy, augmented
GMRES that carries vectors across restarts) fixes the oscillation and STABILIZES:
`r13=-0.527 (6.8%)`, `r116=+2.78e-5 (2.4%)` at 3060 matvecs (residual 0.23, still
under-iterated). So the solver is the bare solver-map + **LGMRES** (not restarted
GMRES).

#### The accuracy ceiling is a STEADY-STATE DEFINITION mismatch, not a solver bug

LGMRES stabilizes the dominant/fast reactions near-gold (`r116` 2.4%) but the
slow-mode-sensitive ones (`r13`) plateau at ~few %. The reason is fundamental and
worth understanding before chasing <1%:

- The slowest chemical mode has `tau ~ 3e10 s`, but the forward run "converges"
  (longdy/dt criterion) at `~2e7 s` — so that mode is **essentially frozen at its
  initial value, not relaxed** (it moved ~0.07% of the way). The criterion passes
  because `dy/dt` of a `tau=3e10` mode is tiny, not because it reached steady state.
- **FD and the IFT therefore differentiate DIFFERENT states.** FD re-integrates to
  the same criterion → `dy*/dk` of the *practically-converged* state (slow mode
  stays frozen). The IFT solves `f(y*)=0` → `dy*/dk` of the *idealized infinite-time*
  steady state (slow mode fully relaxes in response to `dk`). For reactions that
  move the slow mode, these disagree by O(few %); for fast-mode reactions (`r116`)
  they agree (the slow mode is irrelevant).
- This is exactly why **forward-mode `jvp` matches FD to <0.1%** but reverse-IFT
  matches only to ~few %: forward-mode differentiates the *actual integration-to-
  criterion map* (same as FD), while the IFT differentiates `f=0`. They are
  genuinely different derivatives when the steady state isn't fully relaxed.

So **<1% on every reaction is not achievable via the `f=0` IFT** at this
convergence tolerance — not for lack of solver iterations. To close it you would
need either (a) a fully-relaxed `y*` (integrate `>> 3e10 s`, ~1000x longer — usually
impractical), or (b) to differentiate the criterion-map itself, which IS reverse-
mode-through-the-loop (the thing `lax.while_loop` blocks). The honest, useful
result: **reverse-mode at the steady state works, unbiased, and matches FD to ~few %
(near-gold on the dominant/fast reactions)** — which is exactly the accuracy a
reaction-importance ranking needs.

**W39b/SO2 DONE (2026-06-16).** The same bare solver-map + log-scaling + conserved-
null deflation + LGMRES, run on the converged WASP-39b state (photo on, ni=89,
nr=1150), gives `dL/d(ln k)` for converged SO2 over ALL 1150 reactions in ONE
adjoint solve — the steady-state analogue of `fig_diff_demo.py` (which is only an
*instantaneous* single-RHS-eval gradient). It CONVERGED CLEANLY here (LGMRES
residual `1.2e-2`, gradient stable across all chunks — better than HD189, because
the SO2-at-peak-layer functional is more fast-mode-dominated). Physically sensible
ranking: **OH + H2 ⇌ H2O + H (0.68)** sets the OH budget, then the direct
**SO + OH → SO2 + H (0.37)** and **OH + S → H + SO (0.32)**; the top sensitivity is
not even a sulphur reaction. Pipeline: `adj_save_state_w39b.py` (converge + dump)
-> `adj_w39b_so2.py` (LGMRES adjoint, writes `outputs/w39b_so2_dLdlnk.npz`) ->
`fig_w39b_so2_reactions.py` (`w39b_so2_reactions.png`, paper Fig `fig:so2_rev`).
**Two caveats:** (1) photolysis is held FROZEN at the converged k_arr — leading-order
thermochemistry; the `dJ/dy` feedback (`outer_loop._make_photo_branch` recomputes
`J` from `y`) is omitted, a second-order refinement. (2) Accuracy is the few-%
steady-state-definition ceiling. **Bug fixed en route:** W39b has species clipped to
EXACTLY 0, so `inv_y = 1/y_star` was `inf` -> all-NaN; mask with
`jnp.where(y_star>0, 1/y_star, 0)` (absent species become identity rows). HD189
never hit this (no exact zeros).

Compile gotcha: each fresh process pays a ~20-min COLD compile of the block-Thomas-
scan step-vjp (a warm in-process cache makes a *second* run in the same session
fast, but it doesn't survive process exit; setting `jax_compilation_cache_dir` did
NOT help and seemed to *break* the warm path — left unset). This, plus macOS load
spikes (load ~45 from desktop apps), is what made the gold-iteration loop slow.

Operational note: macOS App Nap throttles backgrounded compiles ~5-10x under
desktop load; wrap runs in `caffeinate -dimsu`. The codegen step-vjp compiles in
~10 min; the segment-sum residual vjp in ~4 s (iterate the linear algebra on that).

**Scripts** (`jax_paper/scripts/`): `adj_save_state.py` (converge+dump HD189 state
once so the linear algebra iterates in seconds), **`adj_solvermap_gmres.py` (THE
working route — bare solver-map + GMRES, converging toward FD; run more cycles to
verify gold)**, `adj_debug.py` (log-space deflated Neumann on the body map — the
first ~6% result), `adj_one.py` (single body-map `dt` run), `adj_dt_sweep.py` (`dt`
regime sweep), `adj_verify.py` (component checks — all pass), `adj_ift_gmres.py` /
`adj_ift2.py` / `adj_ift3.py` (residual-IFT with right-precond / augmented /
bordered GMRES — the dead-end route, kept as evidence the separate preconditioner
can't cluster the slow modes). These supersede the failed `diag_reverse_adjoint.py`
/ `diag_fixedpoint_adjoint*.py`. A clean reusable solver should land in
`steady_state_grad.py` once gold + photo-on are reached.

### 2026-06-16: PRODUCTIONIZED — single reverse-mode path in the library

The working route (bare solver-map + log-abundance + conserved-null deflation +
host LGMRES) is now the public `steady_state_grad.steady_state_reaction_sensitivity`
— `dL/d(ln k_r)` for all reactions in one solve. This is the **only** reverse-mode
path now: the residual-IFT `custom_vjp` (`differentiable_steady_state*`,
`_solve_adjoint` block-Thomas defect-correction, the `SteadyStateInputs` plumbing)
and its synthetic test were **deleted** as the failed route — it converged only on
the well-conditioned synthetic problem, never on a real closed column. The dead-end
exploratory scripts (`adj_ift*`, `adj_debug`, `adj_one`, `adj_dt_sweep`,
`adj_verify`) and the repo-root `ablation_*.py` were removed; the working scripts
(`adj_solvermap_gmres.py`, `adj_w39b_so2.py`, `adj_save_state*.py`) stay as the
reference + npz-state dumpers.

Verification (all green): `tests/test_steady_state_reaction_sensitivity.py` —
fast deflation/scaling/assembly unit tests always-on; slow HD189 fixture
regression (`tests/data/adj_state_hd189.npz`, `VULCAN_JAX_RUN_SLOW=1`, ~18 min
incl. step-vjp compile) reproduces the FD anchors (CH4 r13 ≈ -0.53 vs FD -0.565,
sign-correct, top-ranked). WASP-39b SO2 (1150 reactions) ranks OH+H2⇌H2O+H top via
`jax_paper/scripts/adj_equivalence_check.py` (library function on the saved W39b
dump). Forward-mode re-confirmed: `tests/test_rates_jax.py` + `examples/grad_jvp_example.py`.
The few-% ceiling and frozen-photo caveat are unchanged (steady-state-definition
mismatch, not a solver bug); forward-mode stays the higher-accuracy route. Recipe:
`examples/grad_reverse_example.py`.

Follow-up additions (review-driven, same pass):
- `rates_jax` is now the *complete* canonical differentiable rate path: the three
  Moses+2005 low-T caps are ported (`apply_lowT_caps`, gated by
  `build_rate_array(..., use_lowT_caps=True)`), so dL/dT is correct on cool
  networks too (default off; hot benchmarks never trigger them).
- `rates_jax` exposes Arrhenius coefficient overrides
  (`build_rate_array(..., rate_coeffs={"a"|"n"|"E"|...})`) so rate-coefficient
  *uncertainty* gradients are available; NASA-9 thermo was already differentiable
  via the `nasa9_coeffs` argument. (The one hardcoded Troe row stays fixed.)
- `steady_state_reaction_sensitivity` now warns by default (not only via `info`)
  when the LGMRES residual or body-map fixed-point error is poor, so an
  under-converged adjoint is not silently trusted.
- New always-on tests: `tests/test_forward_jvp_physical.py` (forward-mode Kzz
  through one step; jvp==vjp, FD sanity), plus rate-coefficient-override and
  low-T-cap parity checks in `tests/test_rates_jax.py`. README gained a
  "what is differentiable" table distinguishing runtime-array inputs (on the
  graph now) from physical *setup* inputs (host-side; build the pytree yourself).

### 2026-06-17: on-graph atmosphere builder — the physical-input plumbing

The "build the pytree yourself" caveat above is now largely lifted for the
atmosphere structure. The README's "what you CANNOT differentiate yet" list
(T_irr, surface gravity, the pressure grid, T-driven `Dzz`) was all one missing
piece: the host-side setup computes every atmosphere array with `jnp` and then
`np.asarray`s it before the runner. New module `atm_jax.py` re-expresses that
cascade as one differentiable function.

- **`build_atm_static(PhysicalInputs, AtmSpec) -> AtmStatic`** reproduces the
  host chain (`compute_mu_dz_g` height integration → `compute_mol_diff` →
  settling → `make_atm_static` gating) entirely on the graph. `PhysicalInputs`
  carries the differentiable leaves (`pco`, `Tco`, `ymix`, `Kzz`, `vz`, `gs`,
  `Rp`); `AtmSpec` holds the static config (species, `atm_base`, toggles, and
  the discrete hydrostatic anchor `pref_indx`). `make_physical_inputs(cfg, var,
  atm, species_list)` bridges a legacy setup; `pco_from_endpoints` exposes
  `P_b`/`P_t`.
- **Single source of truth, not a fork.** The three genuinely-NumPy formulas got
  `jnp` cores in `atm_setup.py` — `sat_p_jax` (Murray/Antoine), `settling_velocity_jax`
  (Cloutman viscosity + Stokes), `kzz_profile_jax` (JM16/Pfunc/const) — and the
  existing NumPy publics now delegate via `np.asarray`. Elementwise float64, so
  parity is machine precision (≤2e-16 vs the old formulae) — production path
  unchanged. The already-`jnp` primitives (`analytical_TP_H14`, the `_scan_*`
  height integration, `compute_mean_mass`, `_Dzz_gen_for_base`) are reused
  directly; only the `_scan_up` init was made tracer-safe (`jnp.asarray(gs)`).
- **Verified.** `tests/test_atm_jax.py`: `build_atm_static` is field-for-field
  equal to `make_atm_static` on the real HD189 setup (≤4e-16) plus the vm/settling
  branches HD189 doesn't exercise; forward-mode `jvp` matches central FD for
  `d(dzi)/dgs`, `d(M·Dzz·dzi)/dTco`, `d(M)/dP_b`. Example
  `examples/grad_physical_example.py` (all three tangents FD-matched to ~6 digits).
- **Left frozen by design.** (1) FastChem `[M/H] → t=0 speciation` (subprocess
  wall; `const_lowT`'s Newton *residual* is differentiable w.r.t. the elemental
  ratios, but its `ini_abun` entry point floats them — partial, not turn-key).
  (2) Photo T-dependent cross-section rebake (`photo_setup._bin_T_dependent`) — a
  per-layer host interpolation, the one remaining heavy port. (3) `alpha` and
  `pref_indx` are discrete/static lookups (correct to first order).
- **Known limit on T_irr.** `analytical_TP_H14` is on-graph and differentiable,
  but `jax.scipy.special.expn`'s forward-mode is pathologically slow when its
  argument spans a deep column's many decades (`expn(2,·)` over 1e-7…1e4 alone
  does not finish in 90 s). So `dL/dT_irr` *through Heng+14* is impractical over a
  full atmosphere — differentiate the `Tco` leaf directly (or use a cheaper
  `T(P)`). The plumbing (Tco differentiable) is unaffected; this is purely an
  `expn` cost. (`tests/test_atm_setup_matrix.py`'s `analytical` cases are slow for
  the same pre-existing reason — they evaluate `expn` at argument ~9300.)

### 2026-06-17: differentiability-surface review pass (external review "check if still live")

Verified an external review of the broader differentiability claims against the
current code. Outcome:
- **NASA-9 gradients were broken, now FIXED.** `rates_jax.gibbs_sp_vector` did
  `jnp.asarray(np.asarray(coeffs))` — the inner `np.asarray` raised
  `TracerArrayConversionError` under `jvp`/`grad` w.r.t. `nasa9_coeffs`, so the
  documented "NASA-9 differentiable via `nasa9_coeffs`" was false. Changed to
  `jnp.asarray(coeffs, dtype=jnp.float64)`: static-input parity is exactly 0.0,
  `build_rate_array` jvp w.r.t. `nasa9` is now finite, `test_rates_jax` green.
- **Doc corrections (README, CLAUDE, this file).** (a) Photo: the README said
  inject `PhotoStaticInputs.sflux` — that field does not exist (`PhotoInputs.sflux_top`),
  and `sflux_top` + room-T `cross_J` are **closure-baked** in
  `outer_loop._make_photo_branch`; only the T-dependent cross sections ride the
  `ProfileVars` carry (`s.pv.p_cross_J_T`). (b) `const_lowT`: the Newton residual
  is differentiable w.r.t. the elemental ratios but the `ini_abun` entry point
  floats them — partial, not turn-key. (c) Removed two stale references to the
  deleted residual-IFT reverse-mode route in CLAUDE.md's JAX/NumPy-boundary section.
- **Real but out of scope:** condensation particle radius/density are
  `float(atm.r_p[...])` / `float(atm.rho_p[...])` (`outer_loop` ~2104), so
  `dL/d(r_p, rho_p)` is blocked (would need runner-level threading).
- **Reviewer overstatements:** the Kzz `pv.Kzz` duplication is real (only the
  convergence `slope_min`, `outer_loop:869`, reads it — the diffusion physics uses
  `atm.Kzz`) but its impact is below the FD-validated <0.1% Kzz gradient; and
  `state._replace(y=y0)` leaves `y_prev`/`ymix` stale, but they wash out by
  convergence so the closed-column metallicity gradient is correct.

### 2026-06-17: atm_jax adversarial review (13-agent workflow) — findings + fixes

A multi-agent adversarial review (equivalence / port-parity / differentiability /
cleanliness / docs) of the on-graph builder. 7 findings confirmed, 0 refuted; the
completeness critic additionally RAN the gaps and confirmed: build_atm_static's
AtmStatic drives a Ros2 step **bit-identically** to the production AtmStatic
(0.0 rel), stacks + vmaps correctly, jit-compiles and matches eager, the
moldiff-off divergence is runtime-inert (compute_diff_grav bit-identical), and the
sat_p-consuming tests pass (40 passed). Differentiability dimension independently
FD-validated every leaf (gs/Rp/Tco/pco/P_b/ymix; Kzz/vz exact identity) — no AD defects.

Fixed:
- **getattr fail-quiet (minor, real regression I introduced).** `load_TPK` had
  routed all four Kzz knobs through `getattr(cfg, X, 0.0)`, turning a fail-loud
  `AttributeError` into a silent `K_deep=0.0` (unfloored JM16 profile) for a
  config selecting JM16 without `K_deep` (which no shipped config defines). Reverted
  to per-branch DIRECT reads (`cfg.K_deep` for JM16, etc.) — fail-loud like HEAD and
  master; `kzz_profile_jax` now defaults the unused knobs to 0.0.
- **use_moldiff/use_settling default asymmetry (nit).** `make_physical_inputs` read
  `bool(cfg.use_moldiff)` (raises if absent) while `make_atm_static` uses
  `getattr(cfg,"use_moldiff",True)`. Aligned both to the getattr defaults.
- **sat_p README overstatement (doc).** `sat_p_jax` was listed in the
  "CAN differentiate, end-to-end" table, but it is a standalone helper — the runtime
  condensation static still reads a host-frozen `sat_p`, so its d/dT does not flow
  through the runner. Moved to a clearly-scoped standalone-helper note.
- **"field-for-field identical" claim scoped (doc).** True for the default config
  (`atm_type` file/analytical/isothermal, moldiff-on); two non-default modes differ,
  in both cases because build_atm_static is the MORE self-consistent one (see below).
  Scoped the claim in the build_atm_static docstring, the test docstring, and README.

Flagged latent PRODUCTION bug (NOT fixed — out of scope, changes table-mode physics):
- **`atm_type='table'` stale `pico`.** Production calls `f_pico` (sets
  `atm.pico = compute_pico(atm.pco)` from the original logspace P_b/P_t grid) BEFORE
  `load_TPK` overwrites `atm.pco` with the file pressures, and never recomputes
  `atm.pico` (the only writer is `atm_setup.py`'s `f_pico`). So `f_mu_dz` integrates
  the hydrostatic height with an internally-inconsistent (pco_rewritten, pico_stale)
  pair — `g`/`Hpi` ~1%, `dzi` ~12% off from a self-consistent build, and `dzi` feeds
  the eddy-diffusion operator. `build_atm_static` recomputes `pico` from the rewritten
  `pco` (self-consistent, the correct value). **CONFIRMED upstream (2026-06-17):
  VULCAN-master has the IDENTICAL behavior** — `vulcan.py:118` f_pico (pico from the
  original logspace grid) -> `:120` load_TPK (`build_atm.py:406` overwrites
  `data_atm.pco` from the table file, pico NOT recomputed) -> `:148` f_mu_dz
  (`build_atm.py:530,554,562` integrate dz/dzi and `:535` pref_indx from the STALE
  pico). So VULCAN-JAX's production path is a FAITHFUL PORT of a latent upstream bug;
  build_atm_static is the (more-correct) deviation. Low-severity/latent: table mode
  is restart-from-saved-profile (usually same grid -> masked); only an off-grid
  table triggers it, and the length-only check at build_atm.py:403 misses it.
  DECISION: keep VULCAN-JAX production matching master (parity), leave
  build_atm_static self-consistent; the real fix is upstream (one line: recompute
  pico after the table pco rewrite). Do NOT silently fix VULCAN-JAX production alone
  -- it would diverge from master.
  **Isaac (2026-06-17): NOT fixing in this release.** Keep the faithful port; the
  build_atm_static self-consistency divergence in table mode is documented and
  intentional. Revisit only if/when upstream VULCAN fixes the f_pico/pco ordering.

### 2026-07-01: adjoint diagnostics hardening (external review pass)

Four small fixes to `steady_state_grad.py` from a full-repo review (no gradient
values change; the solve path is identical when healthy):

- **`null_quality` made meaningful.** The old metric (`||Q - Q(Q^T Q)||/n_e`) was
  QR-orthonormality — vacuously ~1e-15 for ANY basis, including a random one, so
  it could not detect a wrong deflation. It is now
  `max_e ||A_eta^T q_e|| / ||A_eta^T r||` (unit-norm basis columns vs a fixed-seed
  random unit direction). Measured on the HD189 fixture: per-direction defects
  1.9e-3..3.1e-1 absolute vs 1.1e4 on a random direction → relative ~2.8e-5. So
  the conserved-mass vectors are only *approximately* null in discrete practice
  (the diffusion stencil is not exactly conservative under the dz weights) — a
  real, previously-hidden contribution to the few-% ceiling. Slow-test assertion
  recalibrated 1e-8 → 1e-3 (35x headroom over measured; broken conservation reads O(1)).
- **Rank guard on the deflation basis.** `_conserved_null_basis` now unit-normalizes
  the c_e stack and fails fast on |R_jj| < 1e-10 (and on an all-zero column):
  unpivoted `np.linalg.qr` maps a rank-deficient stack to arbitrary orthonormal
  directions OUTSIDE span(C), which would silently project a needed direction out
  of the solve. New unit tests cover both failure modes.
- **LGMRES cycle hygiene.** Warm-start cycles now stop early when scipy reports
  convergence (info==0) and raise on breakdown/illegal input (info<0) instead of
  silently returning garbage; info>0 (not yet converged) continues as before.
- **Doc unification.** y_star is a fixed point of the *bare* body map (what
  fp_err measures); the parameter docstring previously said "renormalized body
  map". Also: the step-VJP is now jitted once (`a_eta_j`) and shared between the
  null-quality diagnostic and the LGMRES matvec (proj moved outside the jit), so
  the expensive transpose compile is paid exactly once.

Also fixed while in the area: `AtmRefreshStatic.ms` docstring said "per-particle
mass (g)" — it is the molar mass (g/mol; divided by Navo in the formulas),
matching master's `atm.ms`.

#### 2026-07-01 addendum: e2e re-measurement — the HD189 solve stagnates and its endpoint is bit-sensitive

Full-budget re-run of the HD189 CH4 adjoint (inner_m=250, cycles=8, warm XLA
cache) after the diagnostics hardening: `resid=0.47` at 8032 matvecs,
`r13=-0.449` (20.6% off FD -0.565), `fp_err=1.35e-4`, `null_quality=2.76e-5`
(new relative metric). The 15% FD-anchor tolerance in the slow test FAILED.

A/B isolation (one LGMRES cycle, identical 1004 matvecs, same fixture):
pre-edit HEAD gives `resid=0.293, r13=-0.505`; the edited module gives
`resid=0.659, r13=-0.717`. The edits are mathematically neutral (same operator,
same projector span — column normalization provably preserves the Householder
Q; early-exit only fires on info==0), so the divergence is ulp-level trajectory
sensitivity of a STAGNATING Krylov solve — consistent with the 2026-06-16 log
("r13 ranges -0.43..-0.63, residual bounces 0.1..1.6"). HEAD itself samples
10.6% off in this probe; the June 6.8% was one draw from the same band. Nothing
regressed — the solve was never converging to rtol on this fixture, and any
FP-level change (jit boundaries, jax/scipy versions, the 2026-06-29 body-map
restructure) re-samples the ~±25% bounce band around FD.

Consequences applied:
- Slow-test FD-anchor tolerance recalibrated 15% -> 30% (15% sat inside the
  bounce band = flaky by construction); sign + top-6-ranking assertions stay.
- Module/README/warn text updated: magnitudes in the stagnation regime (resid
  warn firing) are ranking weights with ~±25% bounce; sign and ranking are the
  stable outputs. The prior "few-%" claim was one lucky trajectory sample.

### 2026-07-01: adjoint solver campaign (10 experiments, no code changes) — body_dt=1e7 fixes the bounce

Scratch-driven A/B campaign on the HD189 fixture (scripts external to the repo;
library building blocks only). Root cause of the stagnation/bounce identified
and a one-argument fix validated.

**Root cause.** The bare body map at y* is NOT a contraction: ARPACK on the
(vjp-side) operator finds >=8 eigenvalues OUTSIDE the unit circle (|lambda| up
to 2.66 at dt=1e8), localized on H/H2 in the top ~5 layers (z144-149, y*~1e10 —
the H<->H2 + escape/diffusion system). Independent confirmation: iterating the
bare body map diverges to NaN in ~45 steps (2.66^45 growth). Consequences:
(I - dG^T) is indefinite (eigenvalues straddle 0) -> restarted-Krylov
stagnation; and the matvec has an FP-cancellation floor that scales with dt
(stiff amplification through the step), which is what the residual plateaus at.
The loss cotangent is orthogonal to all unstable modes (<v_i, b> ~ 0), which is
why the ranking survives everything.

**body_dt map** (LGMRES 250/40/4, 4 warm cycles, CH4 r13 vs FD -0.5651):

| body_dt | final resid   | r13 error   | reproducibility (ulp twins) |
|---------|---------------|-------------|------------------------------|
| 1e6     | 346 (diverges)| wrong sign  | —                            |
| 3e6     | 0.003-0.008 (CONVERGES) | 27.6% (deterministic bias: slow modes with tau>~dt underweighted) | +/-0.25% |
| 1e7     | 0.04-0.15     | 0.3-6.0% over 4 twins, mean 3.5% | +/-3% |
| 3e7     | 0.40          | 7.5% (1 sample) | —                        |
| 1e8 (default) | 0.2-1.5 stalled | 0.5-20.6% samples | +/-20% lottery |
| >=3e8   | 1e5-3e6       | garbage     | —                            |

**Alternatives all refuted** (same operator/b, dt=1e8): un-restarted
GMRES(2500) breaks down (resid 60); GCROT(m,k) blows up (resid 191); LSMR is
FP-dead — the jvp-side rmatvec is unusable in these coordinates (transpose-pair
checks fail by 1e17 even on structured vectors; raw-coordinate tangents span
~50 decades -> catastrophic cancellation), independently validating the
vjp-only design; naive one-sided eigendeflation of the unstable modes makes it
WORSE (resid 21 — non-normal operator, orthogonal projection of right
eigenvectors does not decouple; proper RPM needs the left/oblique treatment,
which needs the FP-dead jvp side). The library's LGMRES config is the only
bounded solver of the five tried — the 2026-06-16 choice is a genuine local
optimum. Also confirmed: 14k matvecs at dt=1e8 show NO convergence trend
(bounce is stationary), and identical code paths reproduce trajectories
bit-exactly (the bounce is across ulp-level FP variants only).

**Practical recipe (no code change): pass `body_dt=1e7`** to
`steady_state_reaction_sensitivity` for magnitude work — resid floor drops
~10x and the FD gap becomes 0.3-6% (4 ulp-twins, mean 3.5%), i.e. the
originally-claimed "few-%" accuracy, now reproducible. Use `body_dt=3e6` when
determinism matters more than the ~28% slow-mode bias. Ranking (sign + top-k)
is robust in every non-divergent config. Caveats: single fixture (HD189
photo-off, CH4 mid-column); spot-check W39b/SO2 before adopting 1e7 as a new
default — the sweet spot should track the slow-mode timescales of each
network/column; the dt>=3e8 cliff (unstable-mode amplification) and the dt<=1e6
divergence bracket the usable window.

### 2026-07-01: reverse-mode hardening shipped (AD-only changes; core physics untouched)

Per maintainer instruction ("AD may be slow; core VULCAN config/physics must not
change"), the campaign recipe is now the library default in
`steady_state_grad.py` — the ONLY runtime file touched:

- `BODY_MAP_DT` default 1e8 -> **1e7** (the measured low-residual regime; the
  full dt map lives in the constant's comment). Adjoint-only knob; the forward
  model, `vulcan_cfg`, and every kernel under `jax_step`/`outer_loop` are
  byte-identical.
- **Twin-ensemble solves** (`n_solves=3` default): the gradient is the mean
  over deterministic ulp-perturbed-RHS solves; `info["ensemble_spread"]` is the
  magnitude error bar (warn at >0.15). `n_solves=1` restores the old behavior.
- `_ADJOINT_RESID_WARN` 0.1 -> 0.2 with measured bands in the comment
  (0.04-0.15 <-> 0.3-6% of FD; >~0.3 <-> ranking-only regime).
- `info` gains `resids` (per twin), `ensemble_spread`, `n_solves`, `body_dt`;
  `resid` is now the ensemble max.
- Slow test recalibrated: ensemble-mean FD anchors at 12% (measured mean 3.5%),
  plus resid<0.3 and spread<0.15 guards; docstring records the dt story.
- README / CLAUDE.md / example updated in the same pass.

#### 2026-07-01 final calibration: best-iterate safeguard + top-10 spread + median warn

Two refinements after the first ensemble calibration exposed a wandering twin
(resid 0.16 at cycle 4 drifting to 0.55 by cycle 8 — warm-restart trajectories
are not monotone):

- `_lgmres_solve` now returns the BEST-residual iterate across cycles (one
  extra matvec per cycle), not the last one.
- `ensemble_spread` is defined over the TOP-10 reactions by |mean| (weak
  reactions carry naturally large relative bounce that never enters a ranking
  figure); the residual warning gates on the ensemble MEDIAN (robust to one
  wandering twin); `info["resid"]` reports the max.

Final calibration (HD189 fixture, defaults body_dt=1e7 / n_solves=3,
inner_m=250, cycles=8): ensemble-mean CH4 r13 = -0.5369 vs FD -0.5651 (5.0%),
r14 symmetric, top-6 ranking exact, per-twin residuals {0.29, 0.05, 0.10}
(median 0.10), top-10 twin spread 0.047, fp_err 1.4e-4, null_quality 2.2e-5,
24k matvecs / ~27 min warm. Slow-test guards set with margin: mean anchors
<12%, median resid <0.2, max resid <0.5, spread <0.15. Full suite 174 passed.

### 2026-07-02: cross-regime hardening battery (W39b + HD189 loss sweep)

Three parallel batteries drove the SHIPPED library function across regimes.

**W39b (SNCHO photo-on, nr=1150, SO2@peak loss) — easy regime, defaults confirmed.**
Residuals 0.005-0.045 at every body_dt in 3e6..1e8 (the HD189 stagnation is
absent); the ANSWER is dt-insensitive to <1% across that whole window
(g1 = -0.679..-0.683) — the HD189 small-dt slow-mode bias does not appear
here; twin spread 5.8e-4; conservation directions are exactly null
(null_quality ~1e-10, vs 2e-5 on HD189 — open-vs-closed column contrast);
unstable modes mild (|lambda| <= 1.66) with zero loss overlap. Ranking
reproduces the paper exactly: OH+H2 <-> H2O+H pair leads (-/+0.682), then
SO+OH->SO2+H (+0.367), OH+S->H+SO (+0.319). Ensembles at dt=1e7 and dt=1e8
agree to 0.2%. New gated regression: tests/test_w39b_adjoint_subprocess.py
(fixture tests/data/adj_state_w39b.npz, local artifact).

**HD189 loss sweep — three failure regimes, all flagged by default-on
diagnostics; no silent failures.**
- Buffered species (H2O/CO @ mid-column, max|g|~8e-3 — "no reaction controls
  this"): excellent residuals (~4e-3) but twin-noisy top-10 tails -> spread
  warn fires (0.17-0.44). The physical conclusion (insensitivity) is robust.
- Upper-atmosphere CH4 (z140): genuine stagnation (resids 0.4-1.7) -> median-
  resid warn fires; twins AGREE (spread 0.045) on a possibly-biased answer —
  spread alone would miss this; the residual gate catches it.
- Loss coupled to the unstable top-layer H/H2 modes (H@z146, cotangent
  overlap 0.49 with the unstable eigenvectors): tiny residuals (3e-3!) but
  spread 0.90 and forward/reverse pair antisymmetry 1.0 -> flagged hard.
  Confirms the b-perp-unstable-modes condition is what makes mid-column
  losses safe.
- Healthy cases: trace species (C2@mid: spread 0.019) and bottom-boundary
  CH4 (spread 0.14, antisym 0.008) work fine.

**Shipped from the battery:** `info["pair_antisym"]` — worst forward/reverse
asymmetry |g_f+g_r|/max over the top-10 genuinely-reversible pairs (photo/
irreversible rows skipped; free to compute). Healthy 0.01-0.3; O(1) flags
internal inconsistency even when residuals are tiny (the H@z146 case). Slow
tests updated: HD189 asserts pair_antisym < 0.5; new W39b regression pins
ranking/values/diagnostics.

### 2026-07-03: the "~few %" ceiling SOLVED to percent level — it was the linearized MAP, not convergence

Goal: get reverse-mode to percent level for all planets, for the paper's SO2
science case; first ask whether it "just needs stricter convergence." Answer:
**no** — and the real cause was mis-attributed in every doc up to here.

**Method.** Per-planet sweeps (HD189 photo-off CH4, W39b photo-on SO2, HD209
photo-on CH4), each adjoint value checked against re-converged centered FD AND
forward-mode `jvp` through the loop. jvp==FD to ~0.02% everywhere, so FD is
ground truth and any adjoint-vs-FD gap is a real adjoint error. Scripts in
`jax_paper/scripts/`: `adj_conv_sweep.py`, `adj_knob_scan.py`,
`adj_solvermap_variants.py`, `adj_photo_feedback.py`, `fd_validate_w39b_reverse.py`,
`adj_conv_analyze.py`; JSONs in `jax_paper/outputs/adj_conv_sweep/`.

**What does NOT help (all measured, not argued):**
- *Stricter convergence.* HD189 CH4 = 7.0 / 6.3 / 8.4 % at `longdy` 1e-2 / 1e-3 /
  1e-4 (FD invariant at -0.5651). Gotcha: the runner's OR-branch
  `longdy<yconv_min` (default 0.1) + stall detector cap `longdy` ~0.1 regardless
  of `yconv_cri`; to actually tighten, set `yconv_cri=yconv_min=target` and raise
  `conv_stall_window` — and it *still* doesn't move accuracy. `fp_err` also stuck
  at 1.42e-4 across all three (a fixed offset, not a convergence artifact).
- *body_dt scan.* Non-monotonic, optimum ~1e7. HD189: 1e6 80%, 3e6 47%, 1e7 6.6%,
  3e7 14%, 1e8 40%.
- *LGMRES budget.* cycles 10→40, inner_m 60→150: 6.6% → ~5% floor.

**Root cause.** The adjoint linearized the BARE Ros2 step, but `OuterLoop`
iterates the HYDROSTATIC-RENORMALIZED map (`sol_balanced = M[:,None]*ymix`,
outer_loop.py:1145). So `y*` is a tight fixed point of the renorm map (fp_err
~1e-9) but only ~1e-4 of the bare map — that 1.42e-4 is the renormalization
correction, and it biases the gradient ~6%. Nothing about the *convergence*
tolerance changes it.

**Fix #1 — `solver_map="renorm"`** (linearize the renorm map). HD189 CH4
6.6% → **0.7%** (fp_err 1.4e-4→2.6e-9, resid 3.3e-2→4.6e-3); HD209 forward row
r121 35% → 1.1%. Dead end checked: additionally deflating the per-layer
total-density direction (`renorm_td`) OVER-corrects (HD189 0.7%→2.5%) — not added.

**Fix #2 — `photo_recompute_k` (dJ/dy).** After renorm, photo-on columns keep a
*separate* frozen-photolysis error on photo-coupled rows: W39b OH+H2->H2O+H
bare 11.2% / renorm 13.0% (the HD189-photo-off control at 0.7% isolates it as
the photo term, not the map). The runner's own photo branch
(`outer_loop._make_photo_branch`) recomputes J(y) via the two-stream RT, which
is `lax.scan`-based → reverse-mode differentiable; feeding it as the body map's
k(y) makes `dG/dy` carry dJ/dy. `renorm + dJ/dy`: **W39b r1 -0.7692 (0.2%),
r691 +0.3647 (0.1%)** vs FD -0.7679 / +0.3645 — both dominant SO2 rows at
percent level, the science case achieved. `body_map_k` stays frozen (a thermal
k perturbation doesn't change J directly; the indirect path is in the state
operator). Cost: an RT solve per Krylov matvec.

**Still hard.** HD209 CH2OH near-equilibrium *reverse* rows: operator genuinely
ill-conditioned (LGMRES resid ~0.1 even with renorm), r122 goes 32%→83% under
renorm while r121 goes 35%→1.1% — the default-on residual/spread/pair_antisym
diagnostics flag it; use forward-mode for that single row.

**Shipped (defaults unchanged so the paper's numbers stand):** `steady_state_grad`
gained `SOLVER_MAP_DEFAULT`/`SOLVER_MAP_CHOICES`, `solver_map` +
`photo_recompute_k` params (threaded through `scan_body_dt_reaction_sensitivity`),
`info["solver_map"]`/`["photo_feedback"]`, and the helper
`make_photo_recompute_k(runner_photo_static, converged_state)` (reuses
`_make_photo_branch`; needs `integ._photo_static`, not the public
`PhotoStaticInputs`). Module + parameter docstrings and limitations 1/2/4
rewritten. Tests: `test_solver_map_invalid_rejected` (fast) + a renorm sub-check
in the slow HD189 regression (fp_err<1e-6, r13/r14<3%). **Recipe:** always
`solver_map="renorm"`; add `photo_recompute_k` on photo-on columns.

### 2026-07-04: `solver_map="renorm"` made the DEFAULT

Per user request ("make the percent-level accuracy the default"), flipped
`SOLVER_MAP_DEFAULT` from `"bare"` to `"renorm"`, so the shipped reverse-mode
adjoint is percent-level out of the box (HD189 CH4 ~0.7%). `"bare"` is retained
only to reproduce the pre-2026-07 raw-step behavior. At that point
`photo_recompute_k` stayed an explicit standard companion on photochemistry-on
columns because the function had no handle on the runner's photolysis state.
Tests updated:
the slow HD189 regression's primary call now asserts the renorm default at
percent level (fp_err<1e-6, r13/r14<3%) with a `bare` cross-check confirming the
legacy map is strictly looser; the W39b subprocess pins re-measured for renorm.
Docs (CLAUDE.md, README, this file, `examples/grad_reverse_example.py`) and the
paper (main.tex, in `\rev`, incl. a fifth reverse-mode AD bullet documenting the
differentiated photolysis) updated to present renorm as the default.

### 2026-07-06: photolysis feedback made the photo-on default

Per review, `photo_recompute_k` now defaults to `"auto"` instead of `None`.
On active photochemistry columns the public adjoint builds the runner's
differentiated photolysis recompute when given `runner_photo_static` +
`converged_state` (or `integ` + `converged_state`), so the default photo-on path
is `renorm + dJ/dy`. If those runner-context fields are missing, the default
raises instead of silently returning the frozen-photolysis leading-order result;
pass `photo_recompute_k=None` only to reproduce that legacy behavior.
