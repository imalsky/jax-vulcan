# Implementation notes — parity-gap closure pass (2026-06-11)

Companion to `VULCAN_PARITY_GAP_REPORT.md` (see its Resolution section for
what was changed per gap) and `big_gpu_needed_change.md`. This file records
what still needs clarification or off-machine verification, and the problems
hit along the way.

## Needs HPC verification (cannot be done on this CPU-only host)

1. **Fix B on-device confirmation.** The chunked Jacobian assembly
   (`chem.py::_JAC_CHUNK_REACTIONS`, `lax.scan` with `unroll=1`) is designed
   to drop the batch-512 vmap transient from ~60 GiB to ~1/7 of that, but the
   XLA-undo risk (scan fused back into one flat scatter) only shows up on
   device. Run the 512-batch sweep on the GH200 node and confirm peak memory
   actually drops. If XLA un-does the chunking, add `jax.checkpoint` on the
   chunk body or shrink the chunk size.
2. **Untiled vs tiled 512.** `gpu_benchmark.py --device-batch 512` measures a
   true 512-wide vmap (expected to fit outright with Fix B); the default
   `--device-batch 128` tiles 4x128. Compare both — if untiled-512 fits and
   is faster per planet, raise the default.
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
