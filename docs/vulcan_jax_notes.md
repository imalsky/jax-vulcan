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

- **`use_print_delta` is not supported.** Master prints the
  largest-truncation-error species at print cadence inside its solver; a
  per-step host print is impractical inside the JIT'd runner. The key was
  later removed from the config surface entirely (`config._REMOVED_KEYS`
  rejects it with a migration message, alongside the equally inert
  `fix_species_time` — the real pin gate is `stop_conden_time`). Wire a
  chunked-runner print path if anyone misses the diagnostic.
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

### 2026-07-13: Gustafsson PI timestep controller ported from neoVULCAN (default OFF)

Ported the PI step-size controller (neoVULCAN `ode_solver.step_size`,
Hairer & Wanner II sec. IV.2) behind `use_pi_controller = False`, with
`pi_controller_alpha = 0.7` / `pi_controller_beta = 0.4` (exponents divided by
the Ros2 error order p=2). Declared in `vulcan_cfg.py` + all four
`cfg_examples/`, bound-checked in `runtime_validation.py`. Mechanics: one new
carry scalar `delta_prev` (the previous kept step's zero-substituted delta;
-1.0 = no-history sentinel, seeded at entry and reset on every plain reject),
consumed by `_step_size(..., delta_prev=, pi_alpha=, pi_beta=)`. Master-faithful
semantics: step_size runs after accepted AND force-accepted steps; the
force-accept path stays pure I-control because master's step_reject invalidates
the history before the dt_min clamp. With the flag off the runner traces the
exact pre-PI graph (the history slot is an inert pass-through); chunked,
vmapped-batch, and freeze-on-done paths inherit the field generically.

**AD safety.** The sharp edge is the sentinel: a naive
`(delta_prev/delta)**(beta/2)` puts a NaN in the untaken select branch. The
history ratio is sanitized to exactly 1.0 when `delta_prev <= 0`, so neither
primal nor tangent can NaN. Verified: jvp of `_step_size` finite + FD-matched on
all three branches, and an end-to-end `d/d ln Kzz` jvp through the real 40-step
HD189 runner (photo on) has finite tangents with PI on, same scale as off.

**Tests:** `tests/test_pi_controller.py` — bitwise off-path identity, a
300-event accept/reject/zero-delta sequence vs a NumPy neoVULCAN mirror
(1e-14), forward-AD checks, and a 50-step HD189 runner off/on comparison.
Full suite 194 green.

**Benchmark (full convergence, this laptop, fresh process each):**
HD189 off 1296 steps / 37 delta-rejects / 48.6 s -> on 1475 / 31 / 53.9 s;
W39b off 1202 / 141 / 63.7 s -> on 1444 / 123 / 74.3 s. Both converge
(end_case=1, same physical t). Verdict: PI cuts rejections 13-16% as theory
predicts, but the history damping slows dt growth enough to cost 14-20% more
accepted steps, net ~11-17% slower — on these columns the reject fraction
(3-10%) is too small for smoothing to pay. Default OFF is correct; revisit only
in regimes with heavy accept/reject churn (e.g. much tighter rtol).


## 2026-07-20: CLAUDE.md consolidation — archived narrative

CLAUDE.md exceeded the Claude Code large-memory-file threshold (~40k
chars). Operative contracts stay in CLAUDE.md; the blocks below are the
full pre-consolidation text of every compressed bullet, archived
verbatim. Much of this narrative also exists as the dated entries
above (2026-06-16 through 2026-07-04); this archive preserves the
exact final CLAUDE.md wording.

### Flow annotation: differentiability note (verbatim pre-consolidation text)

  Differentiability: per-step kernels are all jit/vmap/jvp/vjp compatible.
                  The runner's lax.while_loop supports jvp but NOT vjp;
                  forward-mode is therefore the end-to-end route. Reverse-mode
                  at the converged state is reaction-importance sensitivities
                  (dL/d ln k for all reactions in one solve) via
                  steady_state_grad.steady_state_reaction_sensitivity — the
                  solver-map steady-state adjoint (I-dG/dy)^T z=v in
                  log-abundance coords, conserved-null deflated, host LGMRES.
                  See the numerical-hygiene differentiability bullet below.

### block_thomas rationale (verbatim pre-consolidation text)

- **block_thomas: production uses `block_thomas_diag_offdiag`.** When sup/sub are diagonal-in-species (which they always are for the diffusion Jacobian), the dense `O(ni³)` matmul `C_j @ inv(A_prev) @ B_{j-1}` reduces to an `O(ni²)` rank update. The dense `block_thomas` stays for callers with truly dense off-diagonals. **Why this is the dominant lever:** per-operation profiling of master's Ros2 step (single-threaded HD189) shows the banded linear solve is the single biggest cost — ~half the step (~80 ms of ~156 ms) — because master calls `scipy.linalg.solve_banded` *twice per step* (two LU factorizations of the *same* LHS) and the band stores the species-diagonal off-blocks as if dense. `block_thomas_diag_offdiag` factorizes once, reuses it for both Ros2 stages, and skips those structural zeros (~5× cheaper solve). By contrast, Python dispatch/glue is only ~3% of master's loop, so the speedup is faster kernels, not "JAX removes Python overhead." The profiling lives in `../jax_paper/scripts/profile_master.py` + `fig_step_cost_2panel.py`.

### codegen RHS detail (verbatim pre-consolidation text)

- **`chem_rhs` is SymPy-faithful via codegen.** `make_chem_funs.py` emits per-reaction Python source mirroring VULCAN-master's `chemdf` body: paired `v_i = forward - reverse`, stoich-replicated multiply chains so XLA cannot rewrite `H*H` as `exp(2*log(H))`, terminal `*M` for three-body reactions, asymmetric M handled per slot via `is_three_body[i]` independently of `is_three_body[i+1]`. Per-species accumulators walk forward reactions in `i = 1, 3, 5, ...` order, products-then-reactants per reaction, repeated by stoich (master line 1300's `+1*v_5 +1*v_5` for OH on the product side of `O + H2O -> OH + OH`). Source cached at `__pycache__/chem_rhs_codegen_<hash>.py` for inspection (cat-able to compare against master's `chem_funs.py` lambda body). The XLA-compiled artifact is reused across processes via JAX's persistent disk cache at `~/.cache/jax_vulcan` (configured at the top of `vulcan_jax_cli.py`). Bit-faithful to master's `chemdf` to ~1 ULP per multiply chain (`tests/test_chem.py` runs at rtol=1e-12; `tests/test_chem_rhs_codegen.py` asserts rtol=1e-5 vs the NumPy oracle — the threshold absorbs XLA FMA drift on cancellation cells; actual worst-cell agreement is ~2e-13, bulk species ~1e-16 — and rtol=1e-12 vs master). The earlier `jnp.prod(y_r ** stoich)` kernel is preserved as `chem.chem_rhs_segment_sum` for `test_chem_jac_sparse.py`'s jacrev oracle and for vmap-consistency tests; not on any production path.

### diffusion Jacobian master comparison (verbatim pre-consolidation text)

- **Diffusion Jacobian: VULCAN-master vs ourselves.** `apply_diffusion` matches `op.diffdf` to 2e-6 (FP noise from extracting small residues from `c0~1e10` cancellations). Block diagonals match `op.lhs_jac_tot` to machine precision for sup/sub blocks but disagree at a handful of diagonal cells for heavy condensables (S8, layers 5 and 25). Direct comparison with the analytical derivative of `op.diffdf` confirms our Jacobian is correct — `op.lhs_jac_tot` has a minor self-inconsistency. Impact on integration is negligible; don't try to "fix" us to match upstream there.

### upwind vm_mol detail (verbatim pre-consolidation text)

- **Upwind molecular diffusion (`use_vm_mol`) is interface-centered.** The advective drift velocity `atm.vm` (`atm_setup.compute_mol_diff` / `atm_jax._mol_diff`) is built on the cell *interfaces* — `vm = -Dzz * (1/H_i - 1/Hp + α·ΔT/T) ` with the interface `Dzz`/`Hpi`/`Ti`/`dzi` and a harmonic-mean species scale height `Hi_interf` — shape `(nz-1, ni)`, matching `vs`. This ports the canonical [shami-EEG `vm_branch`](https://github.com/shami-EEG/VULCAN/tree/vm_branch) `op.update_mu_dz` (which uses `np.roll(species_Hi,-1,axis=0)` layer-averaging; that branch's `build_atm.py` copy drops `axis=0`, a latent species-mixing bug the run-time recompute overrides — so we ported the correct `op.py` form). NOT the cell-centered `Dzz_cen`-based form some local master copies carry. `vm` rides into `op.diffdf_vm`'s upwind stencil unchanged: `vm[k]` is the interface between cells `k` and `k+1`, so `vm[-1]` is the top interface. **`vm` is refreshed in-loop, not frozen.** Unlike `Dzz`/`Ti` (genuinely composition-independent — frozen at setup), `vm` depends on the mean molecular weight through `1/Hpi = mu·g/(kT)` and on `g` through the species scale height, so it changes as molecular diffusion separates species. VULCAN's `op.update_mu_dz` recomputes it every `update_frq` steps ("# Also update vm") alongside `g`/`dzi`/`Hpi`; the runner mirrors this — `body_fn` calls `atm_refresh.recompute_vm_jax(s.g, s.Hpi, s.dzi, Dzz, ms, alpha, Tco)` and splices the live `vm` into `atm_step` (the vm *refresh* is gated on the static `_Statics.use_vm_mol`; the per-step diffusion *blend* is driven by the carry `s.hybrid_use_vm` — see the hybrid bullet below). **`use_vm_mol` is now on by default** (vm_branch), so the pure-central path requires `use_vm_mol=False`. Freezing `vm` (an earlier bug) biased a molecular-diffusion-dominated upper atmosphere (low Kzz) by up to ~1.7 dex on depleted trace species and ~0.4 dex on He vs upstream; the in-loop refresh collapses that to ≤0.16 dex (the central-scheme / stiff-regime convergence floor) and brings the upwind step count in line with upstream. Invisible in normal high-Kzz runs because eddy mixing keeps `mu` ~constant, so frozen-vm ≈ live-vm there. Validation: reference-formula oracle + production-kernel discretization + master `diffdf_vm` (`tests/test_atm_jax.py`, `tests/test_diffusion_production_kernel.py`, `tests/test_diffusion_variants.py`); forward-mode differentiable through `build_atm_static` and through the runner (the recompute is pure arithmetic on differentiable carry/static arrays). **One upstream self-inconsistency we don't replicate** (same stance as the `lhs_jac_tot` note above): `op.diffdf_settling_vm` *omits* the `vm` advective term at the bottom boundary (`j=0`) — keeping only `vs` there — even though `op.diffdf_vm` and `diffdf_settling_vm`'s own interior/top include `vm` at `j=0`. This is present in **both** VULCAN-master (`op.py:1886`) and the shami `vm_branch` (`op.py:1924`) — byte-identical, not a vm_branch-only regression. VULCAN-JAX keeps `vm` consistent at `j=0` across all modes, so in the doubly-non-default `use_vm_mol+use_settling` combo the two agree everywhere except the `j=0` row, where they differ by exactly the omitted vm bottom-flux term (pinned in `test_diffusion_variants`).

### hybrid vm_mol port detail (verbatim pre-consolidation text)

- **Hybrid molecular diffusion (`use_hybrid_vm_mol`) is an IN-LOOP phase flip (2026-07-14, vm_branch port).** vm_branch's `op.py stop()` runs the upwind scheme to convergence, then mutates `vulcan_cfg` to central difference and continues. A single JIT'd `lax.while_loop` cannot flip a static mid-run, so the port drives the diffusion blend from a **carry** value instead: `jax_ros2_step` already computes `A = (1-use_vm)*A_grav + use_vm*A_vm` with `use_vm` a traced float, so `body_fn` splices `atm_step._replace(use_vm_mol=s.hybrid_use_vm)` where `JaxIntegState.hybrid_use_vm` is 1.0 (upwind, phase 0) or 0.0 (central, phase 1). Phase 0 **never terminates** — `_real_terminate` masks it when `_Statics.hybrid_vm_mol`; instead `body_fn` flips `hybrid_use_vm 1.0->0.0` when phase 0 ends (convergence via the factored `_convergence_ok`, or budget) and **resets the budget the vm_branch way** via three dynamic-budget carry fields `count_min_dyn`/`count_max_dyn`/`runtime_dyn` (seeded to the static caps): convergence→`count+2000`, runtime→`count+1000` & `runtime*1.1`, step-count→`count+1000`, `count_min`→`count+100`. `_call_runstate` reads `final_state.count_max_dyn`/`runtime_dyn` for `end_case` so a phase-1 convergence past the static `count_max` still reads as converged. **A completed hybrid run always ends in phase 1**, so the converged state is a central-difference fixed point: forward-mode `jvp` through the runner works end-to-end, and the reverse-mode adjoint (`steady_state_grad.make_body_terms`) splices `use_vm_mol=s.hybrid_use_vm` too, so it linearizes central diff (not upwind) at that state; `audit_adjoint_scope` reports `vm_mol_hybrid` info instead of the `vm_mol_feedback` warning. Seeding: `hybrid_use_vm = 1.0` iff `use_vm_mol` else 0.0, so **non-hybrid runs never flip and their trace is bit-identical** (pure-vm stays 1.0, central stays 0.0). Only one full `JaxIntegState` construction site (`_pack_state_from_runstate`); stack/unstack are `tree_map`-generic. `use_vm_mol` + `use_hybrid_vm_mol` are **on by default** (base cfg + gas-giant examples; Earth central). Batched/emulator fixtures pin `use_vm_mol=False` for a deterministic scheme. Tests: `tests/test_hybrid_vm_mol.py` (flip test gated `VULCAN_JAX_RUN_SLOW`). Convergence + accuracy of the hybrid default on HD189/W39b is a pending HPC validation gate; the default flip needs a suite re-baseline (bare-default oracle/adjoint/count-terminated tests).

### high_temp_cut detail (verbatim pre-consolidation text)

- **High-temperature bottom cut (`high_temp_cut`, default off, vm_branch port).** Not a T-clip: `atm_setup.high_temp_cut_regrid` raises `P_b` to the shallowest deep level (`P >= high_temp_cut_P`) with `T <= high_temp_cut_K` (floored at `high_temp_cut_P`), then re-grids `pco` onto `nz` logspaced levels and reloads T/Kzz — dropping ultra-hot deepest layers for stiffness. Host-side setup only (`state._build_pre_loop_runstate` runs it after `load_TPK`, before `sp_sat`); **not** in the on-graph `atm_jax.build_atm_static`, so the retrieval live-`T(P)` path is unaffected (and it is default off). `load_TPK`'s analytical branch now uses `Pb=pco[0]` (the deepest level, `= P_b` for the standard grid) so the re-grid stays self-consistent — a no-op for the normal path. Tests: `tests/test_high_temp_cut.py`.

### gravity knob removal detail (verbatim pre-consolidation text)

- **Surface gravity is derived from `Mp`/`Rp` — no `gs` knob (2026-07-14).** `atm_setup.surface_gravity(cfg)` returns `G*Mp/Rp**2` (with `phy_const.G_grav`) and raises if `Mp`/`Rp` are missing/≤0. The `gs` config knob and its escape-hatch override were **removed** (too much legacy); every config sets `Mp`+`Rp` (each `Mp` chosen to reproduce its planet's adopted `gs` to sub-ULP: HD189 2140, HD209 936, W39b 422, Earth 980). The resolved value still rides `metadata.gs` and `PhysicalInputs.gs` downstream (unchanged). **Cross-repo:** the siblings were migrated in the same pass — `vulcan-jwst-tool` keeps `gs_cgs` as its user/RT knob but converts it to `Mp` at the chemistry `cfg_overrides` boundary (`planets.G_CGS`), and `vulcan-retrieval`'s cfg-side gravity override passes `Mp`+`Rp` (its `tp_gravity_cgs`/RT `gs_cgs` are RT geometry and unchanged).

### steady-state adjoint bullet (full campaign narrative) (verbatim pre-consolidation text)

- **Differentiate the inner JIT'd `runner`, not `OuterLoop.__call__`.** `__call__` copies device state to the host (`legacy_view` / `np.asarray`) for `.vul` output, which breaks tracing; the AD entry point is the inner `integ._runner(state, atm_static)` (driven via `integ._pack_state_from_runstate(rs)` + `make_atm_static(...)`). `jax.lax.while_loop` supports `jvp`/`jacfwd` but raises on `vjp`/`grad`, so the runner is **forward-mode-AD-able end-to-end** — validated on a full HD189 production run (photo on, ~1300 steps): a `d/d ln Kzz` `jvp` through the whole converged integration matches re-converged centered finite differences to <0.1% on the responding levels (`../jax_paper/scripts/fig_kzz_jvp_validate.py`). This depends on the AD-safe analytical-Jacobian power above — without it the tangent silently goes NaN at clipped cells while the primal stays finite. **Reverse-mode (`vjp`/`grad`) cannot pass through the loop**, so reverse-mode at the converged state is the *steady-state adjoint* in `steady_state_grad.steady_state_reaction_sensitivity` — `dL/d(ln k_r)` for all reactions in one solve (the reaction-importance question forward-mode is the wrong shape for). It is the **single production reverse-mode path**: the earlier residual-IFT `custom_vjp` (`differentiable_steady_state*`, block-Thomas defect-correction) was *removed* because on a real closed column `df/dy` is *both* singular (conserved-mass null space) *and* severely ill-conditioned (stiff chemistry — the residual at the converged state is ~1e21), so that transposed solve diverged; a matrix-free LSQR pseudoinverse and a *raw Neumann* fixed-point adjoint failed the same way. **Why the shipped route works** (four coupled fixes): (1) the **solver-map** `(I-dG/dy)^T` (the integrator's own regularized step — the hydrostatic-renorm map by default via `solver_map="renorm"`, or the raw step via `solver_map="bare"`, see Accuracy below) is the only well-conditioned preconditioner, a separate `reg*I-J` cannot match it; (2) **log-abundance** coords `eta=ln y` (operator norm `~1e6 -> ~1e2`, cotangent `~1e-12 -> O(1)`); (3) **deflate** the conserved-mass null space with the analytic atom-count vectors `c_e[z,i]=compo[i,e]*dz*y*` (only the *left*-null is needed — `c_e^T df/dk=0`); (4) **LGMRES** (augmented Krylov; restarted GMRES oscillates and Neumann diverges on this indefinite operator). The Krylov solve is **host-side scipy** (JAX has no LGMRES), one-shot post-convergence, off the hot path. **Accuracy (2026-07-03 campaign — supersedes the old "steady-state-definition ceiling" framing):** the legacy `solver_map="bare"` matches FD only to ~few % (HD189 CH4 ~6.6%, W39b OH+H2 ~11%), but this is a *linearized-map* error, **not** a convergence-state mismatch: FD is invariant across `yconv` 1e-2..1e-4 (jvp==FD to 0.02%), and stricter convergence, a `body_dt` scan (1e7 is the non-monotonic optimum), and a bigger LGMRES budget all fail to move it. Root cause: the adjoint linearized the **bare** Ros2 step while `OuterLoop` iterates the **hydrostatic-renormalized** map (`sol_balanced = M[:,None]*ymix`), so `y*` is only a ~1e-4 fixed point of the bare map. **Fix #1 `solver_map="renorm"`** linearizes that renormalized map → `fp_err`~1e-9 and HD189 CH4 → **0.7%** (HD209 forward rows 35%→1%). On photo-on columns a *separate* frozen-photolysis error then dominates the photo-coupled rows (`dJ/dy` omitted; W39b OH+H2 stuck ~11-13% — confirmed by the HD189-photo-off control at 0.7%). **Fix #2 `photo_recompute_k`** (build with `make_photo_recompute_k(integ._photo_static, converged_state)`) rebuilds `J(y)` through the runner's `lax.scan`-based two-stream RT each body-map application so `dG/dy` carries `dJ/dy`; **`renorm`+`dJ/dy` takes W39b SO2 dominant rows to r1 0.2% / r691 0.1%** vs re-converged FD (the paper's science case, now percent-level). **`solver_map="renorm"` is the DEFAULT (flipped 2026-07-04);** `bare` is kept only to reproduce the pre-2026-07 raw-step behavior. `photo_recompute_k` stays an explicit argument (the function has no access to the runner's photolysis state) and is the standard companion on photo-on columns; `dJ/dy` costs an RT solve per Krylov matvec. Genuinely-hard residue: HD209 CH2OH near-equilibrium reverse rows stay ill-conditioned (LGMRES resid ~0.1 even with renorm; flagged by the default-on diagnostics). Forward-mode matches FD <0.1% and stays the exact route for any single hard row. `body_dt` is an adjoint-only probe-step knob with a measured usable window (2026-07-01 campaign, HD189): default **1e7** (resid 0.04-0.15; twins land 0.3-6% of FD, mean 3.5%); 1e8 (old default) stalls at resid 0.2-0.7 with ~±25% magnitude bounce (the body map has unstable top-layer H/H2 eigenmodes, |λ| up to ~2.7); 3e6 converges but underweights slow chemistry (~28% bias); ≥3e8 diverges; ~1e11 hits the singular-step pole (hard-guarded). The returned gradient is the **mean over an `n_solves=3` twin ensemble** (deterministic ulp-perturbed RHS; `info["ensemble_spread"]` is the magnitude error bar). Scan a few `body_dt` values on a new column and keep the lowest `info["resid"]`. Species clipped to exactly 0 are masked before the `1/y*` log-scaling (W39b NaNs otherwise). **AD verification (2026-06-16):** `tests/test_steady_state_reaction_sensitivity.py` runs fast deflation/scaling/assembly units always-on; the slow HD189 fixture regression (`VULCAN_JAX_RUN_SLOW=1`; ~3 ensemble solves + step-vjp compile) reproduces the FD anchors with the ensemble mean at body_dt=1e7 (CH4 r13 within 12% of FD -0.565, sign-correct, top-ranked, resid/spread-guarded); WASP-39b SO2 (1150 reactions) ranks OH+H2⇌H2O+H top (`jax_paper/scripts/adj_equivalence_check.py`; paper Fig `fig:so2_rev`); forward-mode re-confirmed (`tests/test_rates_jax.py`, `examples/grad_jvp_example.py`). **Differentiability-surface additions (review-driven):** `rates_jax` is now the complete canonical rate path — low-T caps ported (`apply_lowT_caps` / `build_rate_array(use_lowT_caps=True)`) and Arrhenius coefficient overrides exposed (`build_rate_array(rate_coeffs={"a"|"n"|"E"|...})`; NASA-9 already differentiable via `nasa9_coeffs`) for rate-coefficient uncertainty gradients (one hardcoded Troe row excepted); `steady_state_reaction_sensitivity` warns by default on a poor LGMRES residual, a loose fixed point, or a large twin-ensemble spread, and fails fast on LGMRES breakdown or a rank-deficient deflation basis; new `tests/test_forward_jvp_physical.py` validates forward-mode Kzz through one step (jvp==vjp + coarse FD; tight Kzz FD is end-to-end in `fig_kzz_jvp_validate.py`); README has a "what is differentiable" table (runtime-array inputs on-graph now vs host-side setup inputs). **Atmosphere-structure builder DONE (2026-06-17):** `atm_jax.build_atm_static(PhysicalInputs, AtmSpec) -> AtmStatic` ports the whole host-side atmosphere cascade on-graph — `pco`/`Tco`/`gs`/`Rp`/`ymix`/`Kzz`/`vz` now reach `M`, `dz`, `Hp`, `dzi`, `Ti`, `Hpi`, `Dzz`, `Dzz_cen`, `vm`, `vs` with tangents; single-source jnp cores `sat_p_jax`/`settling_velocity_jax`/`kzz_profile_jax` added to `atm_setup.py` (the NumPy publics delegate, machine-precision parity). Field-for-field identical to `make_atm_static` for the default config (`atm_type` file/analytical/isothermal, `use_moldiff=on`; `atm_type='table'` and `use_moldiff=off` intentionally differ — build_atm_static is more self-consistent, see the JAX/NumPy-boundary `atm_jax` note + the table-mode KNOWN ISSUE below) (`tests/test_atm_jax.py`), FD-matched `jvp` for `dz/dgs`, `M·Dzz/dTco`, `M/dP_b`; example `examples/grad_physical_example.py`. Remaining frozen by design: photo T-dependent cross-section rebake (`_bin_T_dependent`) and FastChem. Caveat: `analytical_TP_H14` is on-graph but `jax.scipy.special.expn` forward-mode is pathologically slow over a deep column (huge argument range) — differentiate the `Tco` leaf directly, not `T_irr` through Heng+14. The photo setup-side builder and FastChem differentiability remain out of scope; the conden setup-side builder is now on-graph (`conden.build_conden_profile`, see the condensation-limits section). Full log + identities: `../docs/vulcan_jax_notes.md`; recipe `examples/grad_reverse_example.py`; working scripts `jax_paper/scripts/adj_solvermap_gmres.py` (HD189), `adj_save_state_w39b.py` + `adj_w39b_so2.py` (W39b). 2026-07-03 accuracy campaign (renorm + dJ/dy fixes): `jax_paper/scripts/adj_conv_sweep.py` (convergence/knob sweep, records that stricter convergence does NOT help), `adj_solvermap_variants.py` (bare vs renorm), `adj_photo_feedback.py` (dJ/dy prototype) + `fd_validate_w39b_reverse.py` (authoritative paper validation via shipped API), `adj_conv_analyze.py` (summary); JSONs under `jax_paper/outputs/adj_conv_sweep/`.

### boundary-section intro (reverse-mode sentence) (verbatim pre-consolidation text)

Reverse-mode through the converged state is the **solver-map reaction-importance adjoint** `steady_state_grad.steady_state_reaction_sensitivity` (`dL/d ln k` for all reactions in one solve); the earlier residual-IFT `custom_vjp` was removed (singular + ill-conditioned `df/dy` on closed columns) — see the differentiability bullet under Numerical hygiene.

### photo_setup boundary detail (verbatim pre-consolidation text)

- **`photo_setup.py`** — host-side cross-section CSV reader + `np.interp` for two-resolution wavelength binning. Produces `state.PhotoStaticInputs` (a JAX pytree). The construction step itself is opaque to JAX, so you cannot get a gradient through "raw CSV → cross-section pytree." **Caveat (not all of `PhotoStaticInputs` is injectable):** the runner's photo branch `outer_loop._make_photo_branch` **closure-bakes** the stellar flux (`sflux_top`) and the room-T cross sections (`cross_J`, `absp_cross`), so perturbing those needs a runner-level input, not a pytree field. The *T-dependent* cross sections do ride the `ProfileVars` carry (`s.pv.p_cross_J_T` / `p_absp_T_cross`) and are differentiable as arrays via the carry. Forward-mode through the runner is the route for any of these (the `lax.while_loop` blocks `vjp`). The CH3SH_branch.csv file has a non-monotonic `354.0` typo that would require a sort step in any `jnp.interp` port; absent a real reason to port, leave alone.

### on-graph atmosphere builder paragraph (verbatim pre-consolidation text)

**On-graph atmosphere builder (`atm_jax.py`).** `build_atm_static(PhysicalInputs, AtmSpec)` reconstructs the `AtmStatic` the runner consumes entirely on the JAX graph — the same cascade `make_atm_static` produces at setup (`compute_mu_dz_g` height integration, `compute_mol_diff`, settling, `M = pco/(kb T)`), but differentiable w.r.t. `pco`/`Tco`/`gs`/`Rp`/`ymix`/`Kzz`/`vz`. `make_physical_inputs(cfg, var, atm, species_list)` bridges a legacy setup into `(PhysicalInputs, AtmSpec)`; `pco_from_endpoints` exposes `P_b`/`P_t`. Single-source jnp cores `atm_setup.sat_p_jax`/`settling_velocity_jax`/`kzz_profile_jax` back both the NumPy publics (machine-precision delegation) and the builder. `pref_indx` (hydrostatic anchor) and `alpha` (thermal-diffusion factor) stay discrete/static. Not handled: photo T-dependent cross-section rebake and FastChem. The "field-for-field identical to `make_atm_static`" invariant holds for the configuration the runner uses (`atm_type` file/analytical/isothermal, `use_moldiff=on`); two non-default modes intentionally differ because `build_atm_static` is the *more* self-consistent one — `use_moldiff=off` (runtime-inert `Ti`/`Hpi` defaults) and `atm_type='table'` (see KNOWN ISSUE below).

### AD-users summary paragraph (verbatim pre-consolidation text)

**What this means for AD users.** **Forward-mode (`jvp`/`jacfwd`) works end-to-end** across the full physical input surface — atmosphere fields (T, P, Kzz), rate constants, boundary fluxes, initial conditions, and photo-static fields — **as long as you supply them as JAX arrays into the pytrees** (`AtmInputs`/`RateInputs`/`PhotoStaticInputs`/etc.) and drive the inner `integ._runner` (not `OuterLoop.__call__`). This is the supported route for end-to-end production gradients (validated; one pass per input direction). **Temperature is a special case:** the runner's `k_arr` is frozen at setup (host-side NumPy `rates.build_rate_array`), so a T-profile gradient must rebuild it on-graph via `rates_jax.build_rate_array(net, T, M, nasa9, remove_list)` (the differentiable port of `rates`+`gibbs`, bit-exact to ~5e-14 vs the NumPy build) and rebuild the structural cascade with `atm_jax.build_atm_static` (which recomputes `n_0 = pco/(kb*T)`, `dz`, `Hp`, and `Dzz(T)` on-graph); with that, forward-mode `d/dT` is validated against finite differences (HD189 dominant species match FD to 3–4 sig figs — `jax_paper/scripts/validate_T_grad.py`). `Dzz(T)` is no longer frozen; only the host-side photo cross-section T-interpolation stays frozen (second-order). **Reverse-mode (`vjp`/`grad`) end-to-end through the loop is blocked** (`lax.while_loop` has no `vjp`); reverse-mode at the converged state is reaction-importance sensitivities — `steady_state_grad.steady_state_reaction_sensitivity` returns `dL/d(ln k_r)` for all reactions in one solver-map adjoint solve (log-abundance + conserved-null deflation + host LGMRES). **The default `solver_map="renorm"` reaches percent level (HD189 CH4 0.7%); on photo-coupled rows also pass `photo_recompute_k` via `make_photo_recompute_k` (dJ/dy → W39b SO2 r1 0.2%, r691 0.1%). The 2026-07-03 campaign traced the old few-% error to the linearized-map choice (NOT convergence); the legacy `solver_map="bare"` reproduces it. Default flipped to renorm 2026-07-04.** Degraded regimes (buffered species, upper-atmosphere losses, HD209 CH2OH near-equilibrium reverse rows) are flagged by the default-on residual/spread/`pair_antisym` diagnostics; gated regressions `test_steady_state_reaction_sensitivity` (incl. a renorm sub-check) + `test_w39b_adjoint_subprocess`. It is the single production reverse-mode path; the earlier residual-IFT `custom_vjp` was removed (singular + ill-conditioned `df/dy` on closed columns). Forward-mode matches FD <0.1% and stays the exact route for any single hard row. **2026-07-05 additions:** `steady_state_input_sensitivity(loss, y_star, k_arr, atm, net, p0, rebuild, ...)` extends reverse-mode to arbitrary physical inputs — e.g. a full (nz,) T profile from the SAME adjoint solve plus one VJP of a differentiable `rebuild(p) -> (k(p), atm(p))` (consistency-checked at p0; renorm differentiated through `atm(p).M`; jvp-validated on HD189); `make_body_terms(integ, converged_state, atm_static)` packs the per-step processes non-default configs turn on (in-window condensation composite = conden-row recompute from y + H2O/NH3 relax kernels + gas-only partial balance, fix_species pins, layer-0 boundary pins) into a `BodyTerms` for the body map AND returns the correctly spliced `atm` (incl. live `vm` for `use_vm_mol`); a fingerprint guard REFUSES conden-active states without matching terms and ion-active states always (no silently wrong gradients), and warns on frozen photolysis; `audit_adjoint_scope(...)` scans a run's config + converged state for dropped processes and measures the per-cell fixed-point defect the global `fp_err` max-norm masks. See the differentiability bullet under Numerical hygiene and the full log in `../docs/vulcan_jax_notes.md`.


## 2026-07-20: CLAUDE.md consolidation, second pass — archived originals

Derivable enumerations and status narrative compressed out of CLAUDE.md
(pointers remain there). Originals below, verbatim.

### state.py schema paragraph (verbatim pre-consolidation text)

The public pre-loop input + runtime schema lives in `state.py`. `AtmInputs` / `RateInputs` / `PhotoInputs` / `PhotoStaticInputs` / `IniAbunOutputs` / `StepInputs` / `ParamInputs` / `AtomInputs` / `PhotoRuntimeInputs` / `FixSpeciesInputs` / `RunMetadata` / `RunState` are NamedTuple-based JAX pytrees. `RunState` is the canonical runtime surface: `RunState.with_pre_loop_setup(cfg)` is a classmethod that runs the entire pre-loop pipeline (`atm_setup` pure functions, `rates.setup_var_k`, `ini_abun.compute_initial_abundance`, `photo_setup.populate_photo` if `use_photo`, plus the photo runtime arrays + remove pass) and returns a fully-populated pytree with `metadata` (host-side static data: `Rf`, `n_branch`, `ion_branch`, `photo_sp`, `ion_sp`, `pho_rate_index`, `ion_rate_index`, `ion_br_ratio`, `charge_list`, `conden_re_list`, `start_time`, `Ti`, `gas_indx`, `pref_indx`, `gs`, `sat_p`, `sat_mix`, `r_p`, `rho_p`, `fix_sp_indx`, `y_ini`) and `photo_static` (the dense `PhotoStaticInputs` cross-section pytree) slots.

### legacy containers paragraph (verbatim pre-consolidation text)

The legacy mutable container classes (`Variables` / `AtmData` / `Parameters`) live in `state.py` as private `_Variables` / `_AtmData` / `_Parameters` (no separate `store.py`). `_build_pre_loop_runstate` uses them as internal scratch and discards them after `runstate_from_store` snapshots their state into the typed pytree. A small set of hybrid oracle tests (`test_photo`, `test_rates`, `test_photo_setup`, `test_ini_abun`'s mode tests, `test_config_matrix`'s atm-only helpers) reach into these private classes too — they need master's pipeline to mutate a `(var, atm)` shared with the JAX side, and `legacy_view(rs)` doesn't carry the dict surface master writes (`var.cross[sp]` etc.).

### CLI driver paragraph (verbatim pre-consolidation text)

`vulcan_jax_cli.py` is an ~80-line driver: `runstate = RunState.with_pre_loop_setup(cfg)` → `runstate = integ(runstate)` → `output.save_out(runstate, dname)`. `legacy_view(rs) -> (var, atm, para)` returns a SimpleNamespace shim for tests still indexing `var.attr` directly. `state.pytree_from_store` / `apply_pytree_to_store` / `runstate_from_store` / `runstate_to_store` keep working for legacy callers that pass `(var, atm, para)`.

### setup-methods paragraph (verbatim pre-consolidation text)

Setup methods are JAX-native: `atm_setup.py` exposes pure functions for every `Atm.*` method (`compute_pico` / `analytical_TP_H14` / `load_TPK` / `compute_mean_mass` / `compute_mu_dz_g` / `compute_settling_velocity` / `compute_mol_diff` / `read_bc_flux` / `compute_sat_p` / `read_sflux_binned`); `ini_abun.py` does the same for `InitialAbun.*` and the 5 `ini_mix` modes (`EQ` / `const_mix` / `vulcan_ini` / `table` / `const_lowT`). Both `Atm` and `InitialAbun` survive as thin facades that mutate `data_var` / `data_atm` for the legacy call sites. `composition.py` extracted the `compo` / `compo_row` / `species` data tables plus a precomputed `(ni, n_atoms)` `compo_array`. The host-side stellar-flux read lives in `state.load_stellar_flux(cfg)`. There is no `build_atm.py`.

### acceptance status paragraphs (verbatim pre-consolidation text)

All live VULCAN-master Ros2 physics is implemented in JAX. Remaining intentionally-unsupported surfaces are `chem_funs.symjac` / `chem_funs.neg_symjac` (raise `NotImplementedError` — replaced by `chem.chem_jac_analytical`), exact `ReadRate.make_bins_read_cross` compatibility (replaced by `photo_setup._build_photo_static_dense` plus the `_synthesize_cross_dicts` writer at .vul time), non-Ros2 solvers, byte-identical pickle output, gradients through raw CSV/table/network readers, and FastChem internals. Live non-default branches (condensation/fix-species/relaxation, atmosphere variants, initial-abundance modes, photo/ion knobs, transport knobs, live UI) are implemented and partially exercised, but not exhaustively cross-validated against master because master's chem_funs codegen breaks on several configs and exhaustive config testing is out of scope.

Validation still not done: exhaustive config-combination oracle sweeps, GPU parity on this CPU-only host, long-to-convergence master oracles for every vendored example, arbitrary custom-network validation beyond parser/schema coverage, gradients through host-side readers/FastChem, and invalid/nonsensical master config combinations that `runtime_validation.py` rejects.

### custom-network paragraphs (verbatim pre-consolidation text)

Custom networks are supported as runtime inputs. The pipeline parses a VULCAN-format network file into stoichiometry/rate arrays and `make_chem_funs.build_chem_rhs(net)` emits a per-network RHS source file, exec's it, and JIT's the result. The cache key is content-based (network mtime + array bytes), so changing the network file at runtime regenerates the source on next call.

When changing or adding a network, check all linked assets explicitly: parser-supported reaction syntax, species entries in `thermo/all_compose.txt`, NASA9 thermo files for reversible species, photo/ion cross-section and branch files for photo/ion species, and any condensation reaction metadata. The codegen RHS handles arbitrary networks; `chem_funs.symjac` / `neg_symjac` intentionally raise (production uses `chem.chem_jac_analytical`).

### condensation two-layers paragraph (verbatim pre-consolidation text)

Condensation has two separate layers that should not be confused. `atm_setup.compute_sat_p` knows saturation-pressure formulae for `H2O`, `NH3`, `H2SO4`, `S2`, `S4`, `S8`, `C`, and `H2S`. The live runtime condensation path, however, only has gas-to-condensate mappings and molecular masses (`conden.GAS_TO_CONDENSATE` / `GAS_MASS_G_PER_MOL`) for the active master-supported condensates (`H2O`, `NH3`, `H2SO4`, `S2`, `S4`, `S8`, `C`). `H2S` saturation data therefore exists, but H2S condensation is not an implemented runtime path unless someone also adds the condensate species/reaction, `r_p`/`rho_p` config, gas-to-condensate mapping, mass constant, and tests. If a new condensation formula is encountered, prefer a clear validation/runtime error over silently treating it as supported.

### condensation static/dynamic split paragraph (verbatim pre-consolidation text)

The condensation state is split static/dynamic (2026-07-13): `conden.make_conden_spec(cfg, var, atm, species_idx)` extracts the temperature-independent metadata (`CondenSpec`), and the pure-JAX `conden.build_conden_profile(spec, Tco, pco, n_0, Dzz)` (jit/vmap/jvp-compatible) rebuilds every T/structure-dependent array (`CondenProfile`: per-reaction `sat_n`/`Dg`, H2O/NH3 relax inputs, NH3 cold-trap argmin, fix-species sat-mix rows). `OuterLoop._build_conden_static` delegates to both (verified bit-exact vs the pre-refactor packer), and the runner reads the arrays from the `ProfileVars` carry per step — so an on-graph caller (vulcan-retrieval's per-proposal `_prep`) regenerates condensation for a live `T(P)` by splicing a rebuilt `CondenProfile` into `pv`. Keep the three in sync: a new conden quantity must be added to `CondenProfile`, threaded through `ProfileVars c_*`, and spliced in `_make_conden_branch`. Formula parity is pinned by `tests/test_conden_profile_builder.py` against an independent NumPy oracle.

### condensation master_pin paragraph (verbatim pre-consolidation text)

**Condensation is the upstream `master_pin` methodology only** (operator-split conden rates + window + `fix_species` pin); there is no `smooth_rainout` / `conden_mode` on `main`. The completed pinned condensation state is **not differentiable-through**: it is a transient snapshot, not a smooth steady state (pinned-species jvp vs FD ~0.91 rel), with discrete phase-boundary switches. Enforced at the autodiff entry points: `steady_state_input_sensitivity` refuses on a condensation state (opt-in `allow_frozen_condensation_input_grad`), `steady_state_reaction_sensitivity` labels its result conditional-on-frozen-reservoir, `runtime_validation` hardens the forward config, and retrieval/Fisher inference with condensation is refused. The open-system Route B experiment (smooth rainout + deep H2S reservoir) that tried to restore differentiability was measured a no-go and is shelved (branch `research/smooth-rainout-fisher`, tag `smooth-rainout-b0c-no-go-2026-07-14`). Full scope: `../docs/condensation_differentiation.md`; Route B records: `../docs/route_b_smooth_condensation_plan.txt` + `route_b_b0a_decision_record.txt`.

### output-schema key inventories (verbatim pre-consolidation text)

The `.vul` file is `pickle.dump(...)` with `protocol=4`, matching VULCAN-master's top-level output shape. Three top-level keys:

- `'variable'` — dict from `vars(data_var)` filtered by `var.var_save`. Contains `y`, `ymix`, `t`, `dt`, `longdy`, `atom_*`, `Rf`, `k`, and (if photo) `tau`, `aflux`, `J_sp`, `n_branch`.
- `'atm'` — dict from `vars(data_atm)`: `pco`, `Tco`, `Kzz`, `Dzz`, `mu`, `n_0`, `dz`, `dzi`, BC arrays.
- `'parameter'` — dict from `vars(data_para)`: counters and convergence/runtime fields (`count`, `nega_count`, `loss_count`, `delta_count`, `delta`, `small_y`, `nega_y`, `end_case`, `solver_str`, `switch_final_photo_frq`, `where_varies_most`, `pic_count`, `fix_species_start`, `tableau20`, `start_time` when available).

The intended contract is same public keys, same array shapes, and same dtypes (float64) as VULCAN-master for user-facing outputs. **All JAX arrays are passed through `np.asarray(...)` before pickling** so VULCAN's `plot_py/` scripts load our output unmodified. Do not break this contract. The writer is not byte-equivalent to upstream and may not preserve incidental dict ordering or transient history details that are not part of the public `.vul` surface.

Validation done for this surface: `tests/test_state_roundtrip.py::test_runstate_output_parameter_schema` checks the RunState-backed parameter schema, photo/ion diagnostic synthesis is covered by `tests/test_photo.py`, `tests/test_photo_setup.py`, and `tests/test_photo_ion.py`, and `tests/test_save_evolution.py` round-trips the evolution arrays through `Output.save_out`.

Validation not done for this surface: no downstream third-party tool corpus beyond VULCAN's plot-script schema is exercised, no byte-for-byte pickle oracle is maintained, and not every live-UI movie/plot combination has a master-output oracle.

### config-surface authority bullet (full knob enumeration) (verbatim pre-consolidation text)

- **Config-surface authority.** `vulcan_jax/configs/*.yaml` is the source of truth for every algorithmic knob the runtime consumes (authored as YAML; `config.py` loads + resolves it). **YAML-only config (2026-07-14):** the hand-written `vulcan_cfg.py` module and `cfg_examples/*.py` were deleted; the config is now a `Config` namespace from `config.load_config(name_or_path)`, and `config.default_config()` is the cached process default that `state._cfg_overlay` mutates in place (same overlay semantics the old module had). YAMLs ship at `vulcan_jax/configs/` and a CWD `./configs/<name>.yaml` overrides them. `config.py` re-derives the handful of values that are functions of other knobs (`dt_max=runtime*1e-5`, `photo_switch_longdy_thresh=yconv_min*10`, `save_movie_rate=live_plot_frq`, `para_anaTP=para_warm`); the strict loader parses `1e22`-style scientific notation as float and rejects duplicate keys. **Gravity is `gs = G*Mp/Rp²` — set `Mp`+`Rp`, there is no `gs` knob.** Other vm_branch-port knobs (2026-07-14): `use_hybrid_vm_mol` (default `True`; `use_vm_mol` default flipped to `True`), `high_temp_cut` / `high_temp_cut_K` / `high_temp_cut_P` (default off). Earlier knobs: `rtol_min`, `rtol_max`, `adapt_rtol_dec_period`, `adapt_rtol_inc_period`, `adapt_rtol_dec`, `adapt_rtol_inc`, `adapt_rtol_loss_mul`, `adapt_rtol_inc_loss_thresh`, `batch_max_retries`, `step_size_safety`, `step_size_zero_delta_frac`, `photo_switch_longdy_thresh`, `photo_switch_longdydt_thresh`, `hycean_pin_time`, `loss_ex`, `fastchem_newton_tol`, `fastchem_newton_max_iter`, `use_fix_all_bot`, `use_fix_H2He`, `use_chunked_runner`, `use_ini_cold_trap`, `use_sat_surfaceH2O`, plus the PI step-size controller trio `use_pi_controller` / `pi_controller_alpha` / `pi_controller_beta` (Gustafsson controller ported from neoVULCAN, default off = master-faithful I-control; history rides the carry as `delta_prev` with a -1.0 no-history sentinel, reset on rejection; the history ratio is sanitized to 1.0 at the sentinel so forward-mode tangents stay finite — see `tests/test_pi_controller.py`). Every shipped config in `configs/` declares them. `runtime_validation._validate_numerical_bounds` bound-checks these at validate time so typos fail early. `getattr(cfg, key, default)` calls in `outer_loop.py` / `ini_abun.py` / `legacy_io.py` exist only for back-compat with old user configs predating the declaration.

### AD-safe power bullet (verbatim pre-consolidation text)

- **The analytical Jacobian's `y^(stoich-1)` power is AD-safe by construction.** A stoich-1 reactant has exponent 0, and `y_r ** 0` has a finite primal (1.0) but a *NaN forward-mode jvp* at `y_r == 0` (`d/dy y^0 = 0 * y^-1 = 0*inf`). The per-step clip routinely sets cells to exactly 0.0 mid-run, so this silently poisoned forward-mode AD through the whole integration (finite primal, NaN tangent by ~step 40). `chem_jac_analytical_per_layer` therefore never raises to the 0 power: stoich==1 contributes a constant 1, stoich≥2 a real power with exponent ≥1 (`safe_exp`). Keep it that way — end-to-end `jvp`/`jacfwd` through the runner (e.g. d(converged composition)/d(ln Kzz), FD-validated to the convergence-noise floor) depends on it. The codegen RHS uses stoich-replicated multiply chains (not powers), so it was already safe.


## 2026-07-20: CLAUDE.md consolidation, third pass — archived originals

### scope-rule paragraph (full knob enumeration) (verbatim pre-consolidation text)

**Scope rule.** Port all of master's physics; don't try to add an oracle test for every config knob. Every live runtime branch in master has a JAX implementation already; "live but non-default" config paths (`ini_mix in {EQ, vulcan_ini, table, const_mix, const_lowT}`, all `atm_type` / `Kzz_prof` / `vz_prof` / `atm_base` variants, `use_moldiff` / `use_vm_mol` / `use_settling` / `use_topflux` / `use_botflux` / `use_fix_sp_bot` / `use_fix_H2He` / `use_sat_surfaceH2O`, every photo + ion knob, every supported condensation species, the four live-UI flags) are implemented but not exhaustively cross-tested vs master. By policy we do not chase that test breadth — if a non-default branch is wrong, we'll find out when it's used. Genuinely dead-in-master paths (non-Ros2 solvers, `naming_solver()`'s commented-out `solver_fix_all_bot` selection) are intentionally not ported.

### standalone/vendoring paragraph (verbatim pre-consolidation text)

VULCAN-JAX is **standalone**. `vulcan-jax` (the console script, i.e. `vulcan_jax_cli.py`) runs end-to-end with no `../VULCAN-master/` sibling. The non-code runtime inputs under `atm/`, `thermo/`, `thermo/photo_cross/`, `configs/`, plus the FastChem C++ source + I/O payload (`makefile`, `*.cpp`/`*.h`, `input/*`, `output/*`) are vendored locally. The FastChem *binary* is **not** vendored — `ini_abun._ensure_fastchem_binary()` compiles it from source (`make` in `fastchem_vulcan/`) on the first `ini_mix='EQ'` use and reuses it thereafter (pyproject ships the source and excludes the built `fastchem` binary + `obj/`). `op.ReadRate` is vendored verbatim into `legacy_io.py`; `legacy_io.Output` is a partial re-implementation that synthesises master-shaped dicts from the typed `RunState` rather than from incremental mutations (so `.vul` public keys/shapes/dtypes match but pickle bytes do not). `vulcan_jax_cli.py` and `op_jax.py` do not `sys.path.append` upstream.

### pytest -n auto comment block (verbatim pre-consolidation text)

python -m pytest tests -n auto -q --tb=short -ra                  # parallel-safe
                                                                 # (FastChem serialises
                                                                 # via fcntl.flock in
                                                                 # ini_abun._load_eq_y,
                                                                 # around _run_fastchem_locked)

### conver_ignore bullet (species list) (verbatim pre-consolidation text)

- **Convergence stall fallback and `conver_ignore` are still useful** even with the codegen RHS in place, because the stall-window detector handles slow-but-genuine convergence on heavy-hydrocarbon trace radicals where `longdy` oscillates around `yconv_min`. The `conver_ignore` default list (`['C6H6', 'C2H2', 'C6H5', 'C2H', 'C2H4', 'C2H5', 'C2H6', 'C3H2', 'C3H3', 'C4H5', 'CH2NH', 'CH3NH2', 'H2CCO']`) and `conv_stall_window = 200` stay as defaults; extend `conver_ignore` per-config only if a *new* trace radical takes over the gating role on a different planet/network. Don't reach for tighter `loss_eps` to control accumulated atom-loss — drift is per physical time, not per step.

### chem_funs re-export paragraph (verbatim pre-consolidation text)

`chem_funs.py` is a JAX-native module that re-exports `ni`/`nr`/`spec_list`/`Gibbs`/`chemdf` etc. from `network.py` + `gibbs.py` + `chem.py`. `chemdf` is backed by `make_chem_funs.build_chem_rhs(_NETWORK)` — a per-network Python source codegen, master-faithful term order, written to `__pycache__/chem_rhs_codegen_<hash>.py` and JIT'd under JAX's persistent disk cache. `symjac` raises `NotImplementedError` — production uses `chem.chem_jac_analytical` directly.

### test-suite lead paragraph (verbatim pre-consolidation text)

`pytest tests/` is the curated suite. Most files use a thin `def test_main(): assert main() == 0` wrapper around their existing script-style `main()`; VULCAN-master oracle comparisons run the script as a subprocess because they deliberately import or path-inject upstream modules. `tests/conftest.py` carries `_cfg_snapshot_session` + `_cfg_guard` autouse fixtures that snapshot/restore the `config.default_config()` attributes, restore canonical VULCAN-JAX modules, and drop sibling-master path/module leakage after every test. Tests marked `strict_isolation` also restore before the test and call `jax.clear_caches()` before and after. `ini_abun._load_eq_y` serialises FastChem invocations via `fcntl.flock` — it holds the lock across the whole invoke + read + cleanup span around `_run_fastchem_locked` — so `pytest -n auto` is safe.

## 2026-07-21: default.yaml gravity slip + master-oracle test rework + sflux-epseri C4

Three related fixes from the jwst-tool audit-response session:

- **default.yaml Mp was the raw literature mass, not the gs-matched value.**
  The 2026-07-14 YAML migration authored `configs/default.yaml` with
  `Mp = 2.1220758e30` while `HD189.yaml` got the adopted-gravity-matched
  `2.1223033871582308e30` -- so every `default_config()` consumer (the CLI,
  the master-oracle test) ran gs = 2139.7705 instead of 2140 (-1.07e-4 rel).
  Measured effect: dz/Hp shift 1.2e-4, per-layer photolysis J up to 7%,
  upper-layer photo radicals (O_1, C, N, O, CH) 2.6e-3 after 20 matched
  steps. Fixed to the HD189.yaml value (commented); named configs verified
  sub-ULP (HD189 2140, HD209 936, W39b 422, Earth 980).
- **test_default_master_parity rework.** The matched-step oracle now (a) pins
  `use_vm_mol = use_hybrid_vm_mol = False` in the JAX subprocess (master has
  no upwind vm_mol, and the hybrid phase flip extends the budget past
  count_max, breaking the matched-count contract), and (b) stages the
  JAX-compiled FastChem binary AND the deduplicated
  `nasa9_logK_SNCHOPTi.dat` (corrections guide C5) into master's tree
  (backup/restored), because two FastChem builds -- and upstream's duplicate
  CH2_1 logK block -- shift equilibrium y_ini by per-element factors up to
  9.3e-7 (C6H6), far above the exact-equality/3e-9 contracts. With both
  staged and gravity fixed, y_ini is bit-identical and the 20-step oracle
  passes at its original tolerance.
- **audit_master_parity.py** learned `config._REMOVED_KEYS` (retired knobs
  like fix_species_time/use_print_delta are documented absences, not drift)
  and `KNOWN_SFLUX_RESCALES` + `_known_sflux_rescale_only` for the C4
  sflux-epseri.txt normalization fix (wavelengths byte-identical, every flux
  ratio pinned at 0.735^-4).
- **Suite re-baseline for the vm_branch defaults (the pending item from the
  2026-07-14 flip).** Five more tests had been failing since the flip:
  test_outer_loop_smoke/conv (count == count_max+1 broken by the hybrid
  phase-flip +1000 budget extension), test_diffusion +
  test_diffusion_production_kernel (central-scheme oracles run against the
  upwind default), test_hybrid_vm_mol (referenced the deleted
  legacy_io.vulcan_cfg), and test_w39b_fastchem_invariant (the C5 logK/
  binary FastChem noise, 2.0e-8 vs a 1e-10 bar). All master/central oracles
  now pin use_vm_mol = use_hybrid_vm_mol = False (upwind coverage stays in
  test_diffusion_variants.py); the W39b invariant stages the JAX FastChem
  binary + dedup logK like the HD189 oracle; test_config's "default" gravity
  pin corrected to 2140. Full suite green after: 228 passed.

## 2026-07-22 — adjoint scope audit: zero-clip dead cells, and why the W39b tabulated default still refuses

Two separate findings from running `audit_adjoint_scope` on the jwst-tool's
new default W39b state (tabulated `atm_W39b_evening_TP_Kzz.txt` T-P **and**
Kzz, photo on, longdy 0.0997).

### Fixed: clip-dead cells were being graded

`_make_body_map` omits the runner's per-step zero-clip, commented "identity-
a.e.". That is true almost everywhere and **false exactly where it fires**:
`outer_loop._make_clip_fn` sets any post-step value in `[nega_cut, pos_cut)`
to 0, so where the clip acts the runner zeroes the cell while the body map
keeps the raw (negative) step. Such a cell has no fixed point at all — it is
re-zeroed on whichever sign the step lands on — so `|G-y|/y` there measures
the clip, and it GROWS with `body_dt` (no probe step can clear it).

`min_ymix` could not exclude them: **the clip window is absolute (cm^-3) while
`min_ymix` is a mixing ratio.** On the cold clamped W39b top (M ~ 7.5e12) a
ymix of 1e-16 is y ~ 1e-3 cm^-3 — three orders INSIDE the |nega_cut| = 1
window — so the pressure-relative floor waved through exactly the cells the
clip owns. Measured at `body_dt` 1e8, four of the top five worst cells were
clip-dead with **negative** G:

    species layer      y            G         G/y     rel    clipdead
    C3H4     88    1.147e-03   -8.215e-04   -0.716   1.716    True
    C3H3     88    3.212e-01   -2.298e-01   -0.715   1.715    True
    C3H2     88    1.425e-01   -9.972e-02   -0.700   1.700    True
    N        93    2.710e+04   +7.299e+04   +2.693   1.693    False

`_clip_dead_mask(G, ymix_old, cfg)` now mirrors `_make_clip_fn` term for term
(both the small/negative cut and the `ymix_old < mtol` trace-negative cut) and
those cells leave the scan; 40 excluded at 1e8. The exclusion is **reported**
(`n_clip_dead_excluded` / `clip_dead_worst_defect` + an info finding — skipped
!= passed) and is **lifted inside the loss footprint**, where a clip-dead cell
becomes a new hard error `loss_reads_clip_dead_cell`: the body map linearizes
a clip as identity, so a loss reading such a cell is wrong at first order.
Units: `tests/test_audit_adjoint_scope.py` (4 always-on).

### NOT fixed: the state genuinely cannot be certified

Excluding clip-dead cells drops the max defect only 1.716 -> 1.693, because
the next cell down is a different problem: **atomic N at layer 93** — 2.7e4
cm^-3 (~7.5 ppb), positive, relaxing toward 2.7x its value. Not dust, not
clipped. Scanning `body_dt` BELOW the sanctioned ladder (production mask
applied) does not help:

    body_dt   max_rel   worst cell   verdict
      1e+05    1.106      CS  L92    refuse
      3e+05    0.527      CS  L92    refuse
      1e+06    0.553      N   L93    refuse
      3e+06    1.222      N   L93    refuse

Never below the 0.3 error threshold. Two follow-ups pinned down what this is
and is not:

* **Not the photolysis-cadence mismatch.** The runner refreshes photo every 5
  accepted steps while the body map (given `photo_recompute_k`) recomputes it
  every probe step, which would plausibly bite hardest at the top. Measured
  with the recompute ON vs FROZEN, the defect is bit-identical (1.222 at 3e6,
  1.693 at 1e8, both ways). Photolysis contributes nothing here.
* **Not fixable by converging harder.** `yconv_cri` 1e-2 / 1e-3 / 1e-4 give a
  **bit-identical converged state** — same longdy 0.0997, same defect at every
  probe step. `longdy` plateaus there and the tolerance knob is simply INERT
  on this column, so "converge tighter" is not an available remedy.

Note the small-`body_dt` tail (1e5 -> 1.106) is conditioning-limited, not
physical: `G(y) -> y` as `body_dt -> 0` analytically, so a defect that RISES
as the probe shrinks is Rosenbrock noise. The physical trend is the monotonic
1e6 -> 1e8 climb. Whatever label one puts on the mechanism, the operational
finding is verified three ways (probe step, tolerance, photo cadence): this
state cannot be certified, and the refusal is correct — do not weaken the gate
to get past it.

Every offending cell (layers 87-93) sits in the **isothermally clamped top**:
the shipped table stops at 5.35e-6 bar while the chemistry grid runs to 1e-7
bar, so the top ~1.7 decades are held at the table's topmost T (726 K). That
clamp is the standard upstream file-mode convention and the forward model logs
it, but it is an unmeasured region, and it is where the trace nitrogen/sulfur/
hydrocarbon chemistry oscillates. The forward result is unaffected: the loss
footprint is clean (0.002) and the W39b SO2 photosphere is layer 68, far below
the clamp. Net: `steady_state_reaction_sensitivity` is not available on this
state; the forward/Fisher path is.


## 2026-07-24 — full-repo review: three parity defects, and why Table 1's HD189 step count no longer reproduces

Comprehensive review of the whole repo (134 tracked text files / 29,965 SLOC) plus the manuscript,
run in the `vulcan` env against the VULCAN-master oracle. 84 findings; the register and executive
summary are session artifacts. What matters for future work:

### Three undocumented parity defects, now fixed (see `corrections_to_original_code.md`)

1. **A non-finite state was scored as PERFECTLY converged** (`outer_loop._conv_jax`). Every mask in
   the `longdy` reduction is a `<`/`>` comparison, and those are all False for NaN, so poisoned
   cells were silently dropped from the max. Measured: a state correctly flagged unconverged at
   `longdy=0.03996` flipped to `longdy=0.0` when the single offending cell was set NaN; an all-NaN
   state read 0.0 and reported `end_case=1` "Integration successful". master raises instead
   (`op.py:1053` reduces an empty selection). Fixed by forcing `longdy=+inf` on any non-finite
   `y`/`ymix`; pinned by `tests/test_nonfinite_never_converges.py` (5 cases). **The single-profile
   `cond_fn`/`_real_terminate` still have no `isfinite` test of their own** — only the batched
   `body_fn_batch` sets `termination_reason=5`. Worth revisiting.

2. **The clip was fed the PRE-step `ymix`.** master's `clip` reads the POST-solve `var.ymix` that
   `Ros2.solver` just wrote (`op.py:3031-3034` → `op.py:3139` → `op.py:2503`), and any cell with
   `y<0` has post-solve `ymix<0<mtol`, so master's rule reduces to "zero every negative". We zeroed
   none: 267/204/204 negatives survived at `dt=1e4/1e8/1e12` (worst 6.99e9 cm^-3), and the
   `all_nonneg` accept gate then *rejected steps master accepts*. **This shifts accepted-step
   counts** — sequence any Table 1 regeneration after it.

3. **The diffusion-limited-escape term was missing from the Jacobian diagonal** while present in the
   RHS. master adds it in all three live `lhs_jac_*` variants gated on a non-empty `diff_esc` list
   (NOT `use_topflux`); `grep diff_lim src/` returned nothing. Live on HD209 (`diff_esc: [H]`) and
   Earth (`[H2, H]`). Measured 1.841x the Rosenbrock diagonal at `dt=1 s`, 1.841e8x at `dt=1e8 s`.
   Fixed by threading a `diff_esc_mask` onto `AtmStatic`. **Adding a field to `AtmStatic` touches
   five places**: the NamedTuple, `make_atm_static`, `atm_jax.build_atm_static` (+`AtmSpec`),
   `_ATM_STATIC_BATCH_AXES`, and `stack_atm_statics`'s explicit `array_fields` whitelist — miss the
   last and the batched runner dies with a vmap size mismatch. The two `tests/data/adj_state_*.npz`
   fixtures also encode the old field set and need a `setdefault` splice.

### Why HD189 takes 2102 steps where the paper reports 1296

Not a regression — a default change never propagated to the paper. The 1296 traces to the PI-controller
benchmark in this file (`HD189 off 1296 steps / 37 delta-rejects / 48.6 s`), whose companion W39b
figure is `1202` — and 1202 is exactly what VULCAN-master produces on W39b today, so that row was
measured at master-equivalent settings. `HD189.yaml` subsequently gained `use_vm_mol: true` and
`use_hybrid_vm_mol: true` (commit `27d8db5`), and the upwind-molecular-diffusion note above records
that the vm path has a deliberately different step count. **Confirmed at the source (2026-07-27):**
`git show 27d8db5^:src/vulcan_jax/cfg_examples/vulcan_cfg_HD189.py` — the config `run_benchmarks.py`
actually used for Table 1 — carries `use_photo = True`, `use_vm_mol = False` and is otherwise
knob-for-knob identical to today's `configs/HD189.yaml` (same nz=150, P_b=1e9, P_t=1e-2,
count_min=120, count_max=1e4, yconv_cri=0.01, yconv_min=0.1, conv_stall_window=200, use_condense
false, same network and atm file). So the vm default is the ONLY config difference between the
published row and the shipped default. **But the vm flip explains only most of the step gap, not
all of it.** Measured (twice, reproducibly) with `use_vm_mol=false` +
`use_hybrid_vm_mol=false` on an otherwise-shipped HD189.yaml: **1495** steps. So the vm default
accounts for 2102 -> 1495 = 607 steps, and **199 steps (1495 -> 1296) remain unexplained** — at
least one further change since that benchmark also moves HD189's convergence. Do not present the vm
flip as the whole story. Note also that the C9 clip fix changes accepted-step counts by design and
landed after the 2102 measurement, so the post-fix HD189 count is not yet measured.
(`photo_off_convergence_investigation.md` also
logs a 1301-step photo-on control, but that document is scoped to the W39b SNCHO column, not HD189 —
it is not corroboration here, and 1301 is coincidentally our own W39b free-convergence count.)
`jax_paper/scripts/bench_runner.py` also sets `use_photo = False` internally, so the published
benchmark protocol and the shipped defaults differ in a way the paper never states.

**W39b free-convergence cross-check (2026-07-24):** master 1202 steps, VULCAN-JAX 1301 steps — 8%
apart on independent implementations, a genuine validation of the port. Table 1's published W39b and
HD209 cells are *step-matched* to master (both columns read 4548 / 1211; `main.tex:262` says so), so
they are not free convergences and must not be compared against free-convergence numbers.

### Do not trust a wall-clock number from this laptop without checking cpu/wall

A W39b timing run gave master 1238 s wall on only 363 s CPU (ratio **0.29**) — throttled by a
**closed laptop lid** (confirmed by the maintainer, 2026-07-25; a 15-minute pytest run was likewise
still unfinished at 96 minutes under the same condition). Keep the lid open or use
`caffeinate -dims` when timing anything. A 16.8x "speedup" derived from it had to be retracted. A
later HD189 master run was worse (4483 s wall / 380 s CPU, ratio 0.085) and never converged.
**Single-threaded master must show `user+sys ≈ real`; if it doesn't, the run is not a measurement.**
`tools/bench_table1.sh` now enforces this and refuses to print a speedup when the guard fails; step
counts are load-independent and always safe to quote.

### Measured hot-path attribution (replaces the old "~5x cheaper solve")

Factorization reuse across both Ros2 stages: **2.10x**. The diagonal-in-species rank update: only a
further **1.10x** (flops drop ~14x per XLA cost analysis, but at `ni≈69` the sweep is
memory/latency-bound). The analytical Jacobian is **16x** (92.1 → 5.66 ms), better than the ~8x
documented. Cost split of a 40.8 ms step: factorization 20.5 ms, two solves 6.6 ms, chem Jacobian
6.4 ms, the entire diffusion/vm-refresh assembly 0.19 ms — the transport side is done.

### Test-coverage fragility worth fixing

`.gitignore`'s blanket `*.npz` (line 88) leaves **all four** numerical regression fixtures untracked
(`git ls-files tests/data` returns nothing), as are `output/HD189.vul` and
`jax_paper/data/jax_HD209.vul`. On a fresh clone the photo-setup baselines, both adjoint
regressions, the HD209 codegen test and the `ini_abun` roundtrip all silently *skip* — green bar, no
oracle. Per this repo's own "skipped != passed" rule those should fail loudly.
`runtime_validation._validate_numerical_bounds` also covers only 25 of 157 declared knobs; 28/28
nonsense values were accepted, and `nz=1` plus inverted `P_b`/`P_t` ran to completion.
