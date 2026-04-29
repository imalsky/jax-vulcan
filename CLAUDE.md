# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VULCAN-JAX is a JAX-accelerated port of [VULCAN](https://github.com/exoclime/VULCAN) (Tsai et al. 2017/2021), the photochemical-kinetics solver for exoplanet atmospheres. The goal is one codebase that runs on CPU and GPU, is fully vectorized (jit + vmap), is numerically equivalent to VULCAN-master, and is clean — no dead code, no legacy fallbacks, no magic numbers outside `vulcan_cfg.py`.

**Scope rule.** Port all of master's physics; don't try to add an oracle test for every config knob. Every live runtime branch in master has a JAX implementation already; "live but non-default" config paths (`ini_mix in {EQ, vulcan_ini, table, const_mix, const_lowT}`, all `atm_type` / `Kzz_prof` / `vz_prof` / `atm_base` variants, `use_moldiff` / `use_vm_mol` / `use_settling` / `use_topflux` / `use_botflux` / `use_fix_sp_bot` / `use_fix_H2He` / `use_sat_surfaceH2O`, every photo + ion knob, every supported condensation species, the four live-UI flags) are implemented but not exhaustively cross-tested vs master. By policy we do not chase that test breadth — if a non-default branch is wrong, we'll find out when it's used. Genuinely dead-in-master paths (non-Ros2 solvers, `naming_solver()`'s commented-out `solver_fix_all_bot` selection) are intentionally not ported.

VULCAN-JAX is **standalone**. `python vulcan_jax.py` runs end-to-end with no `../VULCAN-master/` sibling. The non-code runtime inputs under `atm/`, `thermo/`, `thermo/photo_cross/`, `cfg_examples/`, plus the FastChem runtime payload (`fastchem`, `input/*`, `output/*`) are vendored locally, and `op.ReadRate` + `op.Output` are vendored into `legacy_io.py`. `vulcan_jax.py` and `op_jax.py` do not `sys.path.append` upstream.

The public pre-loop input + runtime schema lives in `state.py`. `AtmInputs` / `RateInputs` / `PhotoInputs` / `PhotoStaticInputs` / `IniAbunOutputs` / `StepInputs` / `ParamInputs` / `AtomInputs` / `PhotoRuntimeInputs` / `FixSpeciesInputs` / `RunMetadata` / `RunState` are NamedTuple-based JAX pytrees. `RunState` is the canonical runtime surface: `RunState.with_pre_loop_setup(cfg)` is a classmethod that runs the entire pre-loop pipeline (`atm_setup.Atm`, `rates.setup_var_k`, `ini_abun.InitialAbun`, `photo_setup.populate_photo` if `use_photo`, plus the photo runtime arrays + remove pass) and returns a fully-populated pytree with `metadata` (host-side static data: `Rf`, `n_branch`, `photo_sp`, `start_time`, `pref_indx`, `gs`, `gas_indx`, `sat_p`, `r_p`, `y_ini`, ...) and `photo_static` (the dense `PhotoStaticInputs` cross-section pytree) slots.

The legacy mutable container classes (`Variables` / `AtmData` / `Parameters`) live in `state.py` as private `_Variables` / `_AtmData` / `_Parameters` (no separate `store.py`). `_build_pre_loop_runstate` uses them as internal scratch and discards them after `runstate_from_store` snapshots their state into the typed pytree. A small set of hybrid oracle tests (`test_photo`, `test_rates`, `test_photo_setup`, `test_ini_abun`'s mode tests, `test_config_matrix`'s atm-only helpers, `tests/_gen_photo_baseline.py`) reach into these private classes too — they need master's pipeline to mutate a `(var, atm)` shared with the JAX side, and `legacy_view(rs)` doesn't carry the dict surface master writes (`var.cross[sp]` etc.).

`vulcan_jax.py` is a 90-line driver: `runstate = RunState.with_pre_loop_setup(cfg)` → `runstate = integ(runstate)` → `output.save_out(runstate, dname)`. `legacy_view(rs) -> (var, atm, para)` returns a SimpleNamespace shim for tests still indexing `var.attr` directly. `state.pytree_from_store` / `apply_pytree_to_store` / `runstate_from_store` / `runstate_to_store` keep working for legacy callers that pass `(var, atm, para)`.

Setup methods are JAX-native: `atm_setup.py` exposes pure functions for every `Atm.*` method (`compute_pico` / `analytical_TP_H14` / `load_TPK` / `compute_mean_mass` / `compute_mu_dz_g` / `compute_settling_velocity` / `compute_mol_diff` / `read_bc_flux` / `compute_sat_p` / `read_sflux_binned`); `ini_abun.py` does the same for `InitialAbun.*` and the 5 `ini_mix` modes (`EQ` / `const_mix` / `vulcan_ini` / `table` / `const_lowT`). Both `Atm` and `InitialAbun` survive as thin facades that mutate `data_var` / `data_atm` for the legacy call sites. `composition.py` extracted the `compo` / `compo_row` / `species` data tables plus a precomputed `(ni, n_atoms)` `compo_array`. The host-side stellar-flux read lives in `state.load_stellar_flux(cfg)`. There is no `build_atm.py`.

`../VULCAN-master/` is the upstream reference. When present, it serves as an *optional* validation oracle: 10 tests compare JAX outputs against upstream's NumPy reference (`op.diffdf`, `op.Ros2.solver`, `chem_funs.symjac`, etc.) and skip cleanly with a clear reason when the sibling is absent. Do not refactor it.

`../vulcan-emulator/` is an unrelated transformer-surrogate project. Out of scope for this repo.

## Conda environment

All commands run in the `vulcan` conda env. It has `jax`, `numpy`, `scipy`, `h5py`, `pytest`, `pytest-xdist`, `ruff`, and `vulture`.

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate vulcan
```

Direct binary if scripts need it: `/opt/homebrew/Caskroom/miniforge/base/envs/vulcan/bin/python`.

## Common commands

```bash
python vulcan_jax.py                                            # forward model (HD189 by default)
JAX_PLATFORM_NAME=gpu python vulcan_jax.py                      # GPU, no code changes
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py  # multi-CPU for vmap/pmap

pytest tests/                                                    # full suite
pytest tests/ -n auto                                            # parallel-safe
                                                                 # (FastChem serialises
                                                                 # via fcntl.flock)
pytest tests/ -k "ros2 or block_thomas"                          # filter

ruff check .                                                     # lint
ruff format .                                                    # format
vulture . --min-confidence 80                                    # dead-code finder

python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul   # diff vs upstream
python benchmarks/bench_step.py                                  # per-step timing
```

The `-n` flag on `vulcan_jax.py` is accepted for upstream compatibility but is a no-op — `chem_funs.py` is JAX-native and there is no SymPy regeneration step.

## Architecture

`vulcan_jax.py` mirrors `vulcan.py`'s orchestration. The hot path is fully JAX — `outer_loop.OuterLoop` runs as a single JIT'd `lax.while_loop` on device:

```
outer_loop.OuterLoop  (single jit'd lax.while_loop — full integration)
  └── jax_step.jax_ros2_step  (jit'd, vmap-able, GPU-ready)
        ├── chem.chem_rhs                  (segment_sum, vmapped over layers)
        ├── chem.chem_jac_analytical       (stoichiometry-driven scatter)
        ├── jax_step.compute_diff_grav     (y-independent, computed once
        │                                   per Ros2 step instead of twice)
        ├── jax_step._build_diff_coeffs_jax (eddy + molecular, four modes)
        └── solver.block_thomas_diag_offdiag (diagonal-aware off-diagonals
                                              — O(ni²) rank update)

  Photochemistry: photo.py JAX kernels (compute_tau / compute_flux / compute_J)
                  fire inside the runner via lax.cond on update_photo_frq.
  Pre-loop setup: state.RunState.with_pre_loop_setup(cfg) — runs the full
                  pipeline (atm structure, rates+caps+reverse+remove, ini_abun,
                  photo cross sections + sflux + remove pass) and returns a
                  fully-populated typed pytree. The private legacy mutable
                  containers (state._Variables/_AtmData/_Parameters) are
                  scratch inside the constructor; the static metadata that
                  doesn't fit the JAX pytree (Rf, n_branch, photo_sp,
                  start_time, ...) rides on rs.metadata.
  .vul writer:    legacy_io.Output.save_out(rs, dname) synthesizes the
                  variable/atm/parameter dicts directly from rs.* slots; the
                  .vul schema is byte-equivalent to upstream so plot_py/
                  scripts consume it unchanged.
  Live UI:        live_ui.LiveUI is a host-side dispatcher fired between
                  JIT'd step batches whenever any of use_live_plot /
                  use_live_flux / use_save_movie / use_flux_movie is True.
                  Setting any of these flags forces the chunked-runner
                  path (chunk size = vulcan_cfg.live_plot_frq) so the
                  host can read state at master's cadence. matplotlib /
                  PIL stay on the host — they never enter a JIT'd
                  region.

  Differentiability: per-step kernels are all jit/vmap/jvp/vjp compatible.
                  The runner's lax.while_loop supports jvp but NOT vjp; for
                  reverse-mode AD across the converged state, use
                  steady_state_grad.differentiable_steady_state or the
                  structured-input helpers in `steady_state_grad.py` — a
                  jax.custom_vjp using the implicit-function theorem
                  (∂y*/∂theta = -(∂f/∂y)^{-1} (∂f/∂theta)). O(1) memory
                  in step count.
```

`chem_funs.py` is a JAX-native module that re-exports `ni`/`nr`/`spec_list`/`Gibbs`/`chemdf` etc. from `network.py` + `gibbs.py` + `chem.py`. No SymPy code generator; `make_chem_funs.py` is not invoked at startup. `symjac` raises `NotImplementedError` — production uses `chem.chem_jac` directly.

See `README.md` for the JAX↔master numerical-agreement table and the capability / differences inventory.

## Numerical hygiene

- **float64 is non-negotiable.** `jax.config.update('jax_enable_x64', True)` is set at module import. Rate constants span ~50 orders of magnitude; float32 silently fails. Don't relax this anywhere.
- **Production uses `chem.chem_jac_analytical`, not `chem.chem_jac`.** A stoichiometry-driven analytical build replaces `jax.jacrev(chem_rhs_per_layer)`; chem_jac drops from 95 ms → 2.6 ms. The `chem.chem_jac` (jacrev) path is kept as the test oracle for `test_chem_jac_sparse.py`. Don't replace the analytical path with AD without re-benchmarking — the speedup is not free, the analytical form skips structurally-zero entries that AD materialises.
- **block_thomas: production uses `block_thomas_diag_offdiag`.** When sup/sub are diagonal-in-species (which they always are for the diffusion Jacobian), the dense `O(ni³)` matmul `C_j @ inv(A_prev) @ B_{j-1}` reduces to an `O(ni²)` rank update. The dense `block_thomas` stays for callers with truly dense off-diagonals.
- **`NetworkArrays` is a registered pytree** (`jax.tree_util.register_pytree_node`). `ni`/`nr` ride as static aux_data so `jit`/`vmap` don't retrace per-network and callers don't need `static_argnames` everywhere.
- **`pos_cut` / `nega_cut` clipping** is replicated explicitly in the outer loop, matching `op.py:2450-2473`. Don't move it into the JIT'd kernel — VULCAN's order matters.
- **Asymmetric M factor for dissociation reactions.** For `HNCO + M -> H + NCO` (R1051), M is on the LHS only; the forward rate is `k * y[HNCO] * M` (3-body), but the reverse `H + NCO -> HNCO` is bimolecular *without* M. VULCAN's auto-generated `chem_funs` preserves this asymmetry by iterating literal reactant/product lists. `network.py` tracks `has_M_reac` and `has_M_prod` separately, so `is_three_body[forward_i]` and `is_three_body[reverse_i+1]` may differ. Anything that assumes symmetry between forward and reverse three-body status is wrong.
- **`chem_rhs` cancellation floor.** `chem_rhs` accumulates 20–40 terms per species with production/loss cancellation; relative error vs VULCAN's `chemdf` is ~1e-4 for a few species (CH2_1, HC3N, HCCO at certain layers). Tests on `chem_rhs` use `rtol=1e-3`; the Jacobian has much less per-entry cancellation and agrees to machine precision. The floor is *not* `segment_sum` reduction-order drift — it is per-term roundoff in the rate computation itself: JAX's `jnp.prod` + masked `y**stoich` and master's SymPy-emitted `y[A] * y[B] * ...` differ by ~1 ulp per multiply, and that ulp accumulates to ~1e-4 absolute when ~7K terms of magnitude ~1e11 cancel down to ~1e2. Verified empirically with `math.fsum`: master's `chemdf` is bit-identical to `math.fsum(master_terms)`, and a Neumaier-compensated single-scan over JAX's terms is bit-identical to `math.fsum(jax_terms)` — same algorithmic precision, different per-term values. A compensated-summation pass therefore does **not** narrow the JAX↔master gap (and costs ~14× chem_rhs runtime); only matching SymPy's term-emission order would. **WONTFIX (user directive):** do not work on a SymPy-faithful codegen. The user has explicitly opted to leave this floor in place. It is a real, known per-term ulp-drift difference vs VULCAN-master for a small set of species — document it, surface it when comparing outputs, but do not attempt to close it. Production uses the fast tree-reduction `segment_sum`.
- **Diffusion Jacobian: VULCAN-master vs ourselves.** `apply_diffusion` matches `op.diffdf` to 2e-6 (FP noise from extracting small residues from `c0~1e10` cancellations). Block diagonals match `op.lhs_jac_tot` to machine precision for sup/sub blocks but disagree at a handful of diagonal cells for heavy condensables (S8, layers 5 and 25). Direct comparison with the analytical derivative of `op.diffdf` confirms our Jacobian is correct — `op.lhs_jac_tot` has a minor self-inconsistency. Impact on integration is negligible; don't try to "fix" us to match upstream there.
- **`outer_loop.runner` is forward-mode-AD-only.** `jax.lax.while_loop` supports `jvp`/`jacfwd` but raises on `vjp`/`grad`. For reverse-mode through the integration, route through `steady_state_grad.py`'s implicit-function-theorem API. The synthetic test in `tests/test_steady_state_grad.py` validates both the legacy `k_arr` wrapper and the structured-input pytree path against finite differences; the gradient accuracy is bounded by the runner's residual `||f(y*)||`, so use `validate_steady_state_solution(...)` / `steady_state_value_and_grad(...)` and tighten the forward convergence criterion when the default `yconv_cri = 0.01` residual is too loose.

## JAX / NumPy boundary (differentiability scope)

The hot-path runtime is fully JAX. Everything inside the `OuterLoop` `lax.while_loop` (chemistry RHS / Jacobian, diffusion, block-Thomas, photo kernels, condensation, atm-refresh) is `jit`/`vmap`/`jvp`/`vjp` compatible. `state.py` defines the pre-loop input pytrees (`AtmInputs`/`RateInputs`/`PhotoInputs`/`PhotoStaticInputs`/`IniAbunOutputs`/`RunState`) — anything fed into the runner via these pytrees is differentiable.

**Some setup-time code stays NumPy.** It is host-side, runs once, and is not on any AD path. The maintainer has decided not to port these to JAX (no hot-path benefit, real bit-exactness risk in some cases):

- **`photo_setup.py`** — host-side cross-section CSV reader + `np.interp` for two-resolution wavelength binning. Produces `state.PhotoStaticInputs` (a JAX pytree). The construction step itself is opaque to JAX, so you cannot get a gradient through "raw CSV → cross-section pytree." If you need cross-section gradients (e.g., for cross-section uncertainty quantification), construct `PhotoStaticInputs` directly from JAX arrays and inject into the runner — everything downstream of the pytree is differentiable via the implicit-AD route in `steady_state_grad.py`. The CH3SH_branch.csv file has a non-monotonic `354.0` typo that would require a sort step in any `jnp.interp` port; absent a real reason to port, leave alone.
- **`legacy_io.ReadRate.read_rate`** (~280 lines) — host-side rate-file metadata parser. Populates `var.Rf`, `var.pho_rate_index`, `var.n_branch`, `var.photo_sp`, `var.ion_sp`, etc. Rate **values** flow through `rates.build_rate_array` (JAX-friendly NumPy with bit-exact output); `read_rate` is metadata only and not on any AD path.
- **`legacy_io.Output.save_out`** — pickle writer; one-shot at end of run. JAX arrays are cast to NumPy via `np.asarray()` before pickle.
- **CSV / data-table readers in `composition.py`, `atm_setup.py`, `ini_abun.py`** — host-side at import or pre-loop. The data tables they produce feed JAX pytrees downstream.
- **FastChem subprocess** for `ini_mix='EQ'` — third-party external dependency.

**What this means for AD users.** Forward-mode (`jvp`/`jacfwd`) and reverse-mode (`vjp`/`grad` via implicit-AD on the converged state) are supported across the full physical input surface — atmosphere fields (T, P, Kzz), rate constants, boundary fluxes, initial conditions, and photo-static fields **as long as you supply them as JAX arrays into the pytrees** (`AtmInputs`/`RateInputs`/`PhotoStaticInputs`/etc.). The runner's `lax.while_loop` blocks `vjp` directly, so reverse-mode through the transient trajectory requires `steady_state_grad.py`'s implicit-function-theorem route (O(1) memory, no checkpointing). See `tests/test_steady_state_grad.py` for the canonical pattern.

If you find yourself wanting a gradient through one of the NumPy-host setup steps, the right answer is almost always: build the corresponding pytree directly with JAX arrays and skip the host-side reader.

## Output schema

The `.vul` file is `pickle.dump(...)` with `protocol=4`, identical to VULCAN-master's `op.Output.save_out` (lines 3219-3260). Three top-level keys:

- `'variable'` — dict from `vars(data_var)` filtered by `var.var_save`. Contains `y`, `ymix`, `t`, `dt`, `longdy`, `atom_*`, `Rf`, `k`, and (if photo) `tau`, `aflux`, `J_sp`, `n_branch`.
- `'atm'` — dict from `vars(data_atm)`: `pco`, `Tco`, `Kzz`, `Dzz`, `mu`, `n_0`, `dz`, `dzi`, BC arrays.
- `'parameter'` — dict from `vars(data_para)`: `count`, `nega_count`, `loss_count`.

Same keys, same array shapes, same dtypes (float64) as VULCAN-master. **All JAX arrays are passed through `np.asarray(...)` before pickling** so VULCAN's `plot_py/` scripts load our output unmodified. Don't break this contract; it is the cheapest end-to-end check we have.

## Test discipline

**Current state:** `pytest tests/` is green on the current tree (108 pass + 1 skip; `test_backend_parity` skips on CPU-only). Most files use a thin `def test_main(): assert main() == 0` wrapper around their existing script-style `main()`; three tests (`test_chem`, `test_diffusion`, `test_ros2_step`) wrap the script as a subprocess because they do a deliberate VULCAN-master ↔ VULCAN-JAX module-table swap that only works from a cold Python start. The suite includes `compute_Jion` coverage, vendored example-config setup coverage, Earth/Jupiter/HD209 20-step matched-step oracles, and an optional CPU↔GPU parity test that skips cleanly on CPU-only hosts. `tests/conftest.py` carries `_cfg_snapshot_session` + `_cfg_guard` autouse fixtures that snapshot/restore `vulcan_cfg` attributes and rebind canonical VULCAN-JAX module objects (`op`, `chem_funs`, etc.) after every test — so tests that insert `../VULCAN-master/` at the front of `sys.path` (e.g. `test_gibbs`) don't pollute later tests' `sys.modules`. `ini_abun._load_eq_y` serialises FastChem invocations via `fcntl.flock` so `pytest -n auto` is safe.

A function is "important enough to test" if any of these are true:
- It's on the per-step hot path (`chem_rhs`, `chem_jac`, `block_thomas`, `apply_diffusion`, photo kernels, `jax_ros2_step`).
- It's a numerical bridge to VULCAN-master (anything in the README's agreement table).
- It's a `jit`/`vmap`/`grad` boundary (transform consistency is a real failure mode in JAX).
- It's a parser or schema reader (`network.py`, NASA-9 reader) — silent-mismatch territory.

Patterns to keep:
- **Equivalence vs VULCAN-master.** Load a real HD189 state once in a session-scoped `conftest` fixture; compare component outputs at machine-precision tolerances (with the cancellation floors documented above).
- **vmap consistency.** Single-call output must match `vmap(...)(stacked_input)[i]` to ~1e-12.
- **Gradient sanity.** `jax.grad`/`jvp` through the per-step kernel must be finite and finite-differenced to ~1e-6.
- **CPU vs GPU agreement.** When GPU is available, the same kernel on both backends must agree to float64 precision.

Don't add tests that just exercise plumbing or duplicate VULCAN's own physics; the bar is "could fail silently and matter."

## Style rules

- **Magic numbers belong in `vulcan_cfg.py` and nowhere else.** That file is the canonical configuration surface and intentionally mirrors VULCAN-master's format. Any literal in `*.py` source with physical or algorithmic meaning gets a named constant with a one-line docstring saying what it is and where it came from.
- **Comments:** docstrings on every public function (what it computes + array shapes + units). Inline comments only where the math, algorithm, or JAX-specific choice is non-obvious — e.g., why `jacrev` over `jacfwd`, why `segment_sum` instead of zip-and-add, why a particular `lax.scan` carry shape. Don't narrate what well-named code already says.
- **No dead code.** `vulture --min-confidence 80` should come back empty. If a function isn't called, delete it; if it's a future placeholder, don't write it yet.
- **No legacy fallbacks.** A code path that exists only because VULCAN-master had it and we haven't ported it is debt to remove, not a feature to preserve. Prefer raising a clear `NotImplementedError` over silently routing to the NumPy parent class.
- **No phase-history annotations.** The capability documentation lives in `README.md` and the code comments describe the *current* state. If you need to record the historical reason for a design choice, use the commit message — don't tag source comments with `Phase N` labels.
- **Vectorize.** Per-layer operations should be `vmap`'d, not Python-looped. Reach for `jax.lax.scan` only when there is a true sequential dependency (e.g., block-Thomas forward elim).
- **Any code change ends with a doc pass.** If behavior, support surface, benchmark numbers, or vendored-data status changed, update `README.md` in the same pass before considering the task done. Update `CLAUDE.md` too when the maintainer rules themselves changed.
