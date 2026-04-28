# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VULCAN-JAX is a JAX-accelerated port of [VULCAN](https://github.com/exoclime/VULCAN) (Tsai et al. 2017/2021), the photochemical-kinetics solver for exoplanet atmospheres. The goal is one codebase that runs on CPU and GPU, is fully vectorized (jit + vmap), is numerically equivalent to VULCAN-master, and is clean — no dead code, no legacy fallbacks, no magic numbers outside `vulcan_cfg.py`.

VULCAN-JAX is **standalone** (Phase A). `python vulcan_jax.py` runs end-to-end with no `../VULCAN-master/` sibling. The non-code runtime inputs under `atm/`, `thermo/`, `thermo/photo_cross/`, `cfg_examples/`, plus the FastChem runtime payload (`fastchem`, `input/*`, `output/*`) are vendored locally, and `op.ReadRate` + `op.Output` are vendored into `legacy_io.py`. `vulcan_jax.py` and `op_jax.py` no longer `sys.path.append` upstream.

The public pre-loop input schema lives in `state.py` (Phase 19) — `AtmInputs` / `RateInputs` / `PhotoInputs` / `IniAbunOutputs` / `RunState` are NamedTuple-based JAX pytrees that the runner ultimately consumes. Today the runner still reads the legacy mutable `store.Variables` / `store.AtmData` containers; `state.pytree_from_store` / `apply_pytree_to_store` bridge the two during the transition. Phase 20 moved every `Atm.*` setup method into JAX-native pure functions in `atm_setup.py` (`compute_pico` / `analytical_TP_H14` / `load_TPK` / `compute_mean_mass` / `compute_mu_dz_g` / `compute_settling_velocity` / `compute_mol_diff` / `read_bc_flux` / `compute_sat_p` / `read_sflux_binned`). Phase 21 did the same for `InitialAbun.*` in `ini_abun.py` (5 `ini_mix` modes — `EQ` / `const_mix` / `vulcan_ini` / `table` / `const_lowT` — with a small JAX Newton replacing `scipy.optimize.fsolve` for the 5-mol H2/H2O/CH4/He/NH3 system in `const_lowT`); `composition.py` extracted the `compo` / `compo_row` / `species` data tables plus a precomputed `(ni, n_atoms)` `compo_array`; **`build_atm.py` is deleted**. Both `Atm` and `InitialAbun` survive as thin facades that mutate `data_var` / `data_atm` for the legacy call sites. The host-side stellar-flux read lives in `state.load_stellar_flux(cfg)`; pass the result to `Variables(stellar_flux=...)` so the constructor stays free of file I/O.

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

pytest tests/                                                    # full suite (run serially)
pytest tests/ -k "ros2 or block_thomas"                          # filter
# NOTE: do not use `-n auto` — parallel runs collide on FastChem's fixed
# output path (see pytest.ini for details).

ruff check .                                                     # lint
ruff format .                                                    # format
vulture . --min-confidence 80                                    # dead-code finder

python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul   # diff vs upstream
python benchmarks/bench_step.py                                  # per-step timing
```

The `-n` flag on `vulcan_jax.py` is accepted for upstream compatibility but is a no-op — `chem_funs.py` is JAX-native and there is no SymPy regeneration step.

## Architecture

`vulcan_jax.py` mirrors `vulcan.py`'s orchestration. The hot path is fully JAX (post-Phase 10) — `outer_loop.OuterLoop` runs as a single JIT'd `lax.while_loop` on device:

```
outer_loop.OuterLoop  (single jit'd lax.while_loop — full integration)
  └── jax_step.jax_ros2_step  (jit'd, vmap-able, GPU-ready)
        ├── chem.chem_rhs                  (segment_sum, vmapped over layers)
        ├── chem.chem_jac_analytical       (Phase 11; stoichiometry-driven scatter)
        ├── jax_step.compute_diff_grav     (Phase 12; y-independent, computed
        │                                   once per Ros2 step instead of twice)
        ├── jax_step._build_diff_coeffs_jax (eddy + molecular, four modes)
        └── solver.block_thomas_diag_offdiag (Phase 12; diagonal-aware
                                              off-diagonals — O(ni²) rank update)

  Photochemistry: photo.py JAX kernels (compute_tau / compute_flux / compute_J)
                  fire inside the runner via lax.cond on update_photo_frq.
  Pre-loop setup: legacy_io.ReadRate (rates + photo bins) + Output.save_cfg /
                  save_out (.vul writer). Vendored from upstream op.py.

  Differentiability: per-step kernels are all jit/vmap/jvp/vjp compatible.
                  The runner's lax.while_loop supports jvp but NOT vjp; for
                  reverse-mode AD across the converged state, use
                  steady_state_grad.differentiable_steady_state or the
                  structured-input helpers in `steady_state_grad.py` — a
                  jax.custom_vjp using the implicit-function theorem
                  (∂y*/∂theta = -(∂f/∂y)^{-1} (∂f/∂theta)). O(1) memory
                  in step count.
```

`chem_funs.py` is a JAX-native module (Phase 9) that re-exports `ni`/`nr`/`spec_list`/`Gibbs`/`chemdf` etc. from `network.py` + `gibbs.py` + `chem.py`. No SymPy code generator; `make_chem_funs.py` is not invoked at startup. `symjac` raises `NotImplementedError` — production uses `chem.chem_jac` directly.

Numerical agreement vs VULCAN-master (HD189, SNCHO_photo_network_2025):

| Layer | Agreement |
|---|---|
| Forward rates (596 reactions) | bit-exact |
| Reverse rates (533 from Gibbs) | 1.4e-14 |
| Atmosphere setup / FastChem init | bit-exact |
| chem_jac (vs symjac) | 4.3e-13 |
| Diffusion operator | 2e-6 (FP-cancellation-bound; see Numerical hygiene) |
| Block-Thomas | 3e-15 |
| Single Ros2 step | 1.16e-15 |
| End-to-end 50-step HD189 | 1.59e-10 |

`STATUS.md` tracks per-component implementation status.

## Numerical hygiene

- **float64 is non-negotiable.** `jax.config.update('jax_enable_x64', True)` is set at module import. Rate constants span ~50 orders of magnitude; float32 silently fails. Don't relax this anywhere.
- **Production uses `chem.chem_jac_analytical`, not `chem.chem_jac`.** Phase 11 replaced `jax.jacrev(chem_rhs_per_layer)` with a stoichiometry-driven analytical build (`chem_jac_analytical`); chem_jac drops 95 ms → 2.6 ms. The `chem.chem_jac` (jacrev) path is kept as the test oracle for `test_chem_jac_sparse.py`. Don't replace the analytical path with AD without re-benchmarking — the speedup is not free, the analytical form skips structurally-zero entries that AD materialises.
- **block_thomas: production uses `block_thomas_diag_offdiag` (Phase 12).** When sup/sub are diagonal-in-species (which they always are for the diffusion Jacobian), the dense `O(ni³)` matmul `C_j @ inv(A_prev) @ B_{j-1}` reduces to an `O(ni²)` rank update. The dense `block_thomas` stays for callers with truly dense off-diagonals.
- **`NetworkArrays` is a registered pytree** (`jax.tree_util.register_pytree_node`). `ni`/`nr` ride as static aux_data so `jit`/`vmap` don't retrace per-network and callers don't need `static_argnames` everywhere.
- **`pos_cut` / `nega_cut` clipping** is replicated explicitly in the outer loop, matching `op.py:2450-2473`. Don't move it into the JIT'd kernel — VULCAN's order matters.
- **Asymmetric M factor for dissociation reactions.** For `HNCO + M -> H + NCO` (R1051), M is on the LHS only; the forward rate is `k * y[HNCO] * M` (3-body), but the reverse `H + NCO -> HNCO` is bimolecular *without* M. VULCAN's auto-generated `chem_funs` preserves this asymmetry by iterating literal reactant/product lists. `network.py` tracks `has_M_reac` and `has_M_prod` separately, so `is_three_body[forward_i]` and `is_three_body[reverse_i+1]` may differ. Anything that assumes symmetry between forward and reverse three-body status is wrong.
- **`chem_rhs` cancellation floor.** `chem_rhs` accumulates 20–40 terms per species with production/loss cancellation; relative error vs VULCAN's `chemdf` is ~1e-4 for a few species (CH2_1, HC3N, HCCO at certain layers). Tests on `chem_rhs` use `rtol=1e-3`; the Jacobian has much less per-entry cancellation and agrees to machine precision. **Phase 17 investigation:** the floor is *not* `segment_sum` reduction-order drift, it is per-term roundoff in the rate computation itself — JAX's `jnp.prod` + masked `y**stoich` and master's SymPy-emitted `y[A] * y[B] * ...` differ by ~1 ulp per multiply, and that ulp accumulates to ~1e-4 absolute when ~7K terms of magnitude ~1e11 cancel down to ~1e2. Verified empirically with `math.fsum`: master's `chemdf` is bit-identical to `math.fsum(master_terms)`, and a Neumaier-compensated single-scan over JAX's terms is bit-identical to `math.fsum(jax_terms)` — same algorithmic precision, different per-term values. A compensated-summation pass therefore does **not** narrow the JAX↔master gap (and costs ~14× chem_rhs runtime); only matching SymPy's term-emission order (Stage 3 of the Phase 17 plan — codegen, big perf hit) would. Production keeps the fast tree-reduction `segment_sum` for now. **WONTFIX (user directive):** do not work on Stage 3 SymPy-order codegen. The user has explicitly opted to leave this floor in place. It is a real, known per-term ulp-drift difference vs VULCAN-master for a small set of species — document it, surface it when comparing outputs, but do not attempt to close it. Closing it requires a SymPy-faithful codegen and pays a ~14× per-call cost on the per-step hot path; the trade is net-negative for the project.
- **Diffusion Jacobian: VULCAN-master vs ourselves.** `apply_diffusion` matches `op.diffdf` to 2e-6 (FP noise from extracting small residues from `c0~1e10` cancellations). Block diagonals match `op.lhs_jac_tot` to machine precision for sup/sub blocks but disagree at a handful of diagonal cells for heavy condensables (S8, layers 5 and 25). Direct comparison with the analytical derivative of `op.diffdf` confirms our Jacobian is correct — `op.lhs_jac_tot` has a minor self-inconsistency. Impact on integration is negligible; don't try to "fix" us to match upstream there.
- **`outer_loop.runner` is forward-mode-AD-only.** `jax.lax.while_loop` supports `jvp`/`jacfwd` but raises on `vjp`/`grad`. For reverse-mode through the integration, route through `steady_state_grad.py`'s implicit-function-theorem API. The synthetic test in `tests/test_steady_state_grad.py` validates both the legacy `k_arr` wrapper and the structured-input pytree path against finite differences; the gradient accuracy is bounded by the runner's residual `||f(y*)||`, so use `validate_steady_state_solution(...)` / `steady_state_value_and_grad(...)` and tighten the forward convergence criterion when the default `yconv_cri = 0.01` residual is too loose.

## Output schema

The `.vul` file is `pickle.dump(...)` with `protocol=4`, identical to VULCAN-master's `op.Output.save_out` (lines 3219-3260). Three top-level keys:

- `'variable'` — dict from `vars(data_var)` filtered by `var.var_save`. Contains `y`, `ymix`, `t`, `dt`, `longdy`, `atom_*`, `Rf`, `k`, and (if photo) `tau`, `aflux`, `J_sp`, `n_branch`.
- `'atm'` — dict from `vars(data_atm)`: `pco`, `Tco`, `Kzz`, `Dzz`, `mu`, `n_0`, `dz`, `dzi`, BC arrays.
- `'parameter'` — dict from `vars(data_para)`: `count`, `nega_count`, `loss_count`.

Same keys, same array shapes, same dtypes (float64) as VULCAN-master. **All JAX arrays are passed through `np.asarray(...)` before pickling** so VULCAN's `plot_py/` scripts load our output unmodified. Don't break this contract; it is the cheapest end-to-end check we have.

## Test discipline

**Current state:** `pytest tests/` is green on the current tree. Most files still use a thin `def test_main(): assert main() == 0` wrapper around their existing script-style `main()`; two tests (`test_diffusion`, `test_ros2_step`) wrap the script as a subprocess because they do a deliberate VULCAN-master ↔ VULCAN-JAX module-table swap that only works from a cold Python start. The suite now also includes `compute_Jion` coverage, vendored example-config setup coverage, and an optional CPU↔GPU parity test that skips cleanly on CPU-only hosts. Per-test fixture sharing for the HD189 reference state is a future cleanup (not a correctness gap). Run serially — parallel collides on FastChem.

A function is "important enough to test" if any of these are true:
- It's on the per-step hot path (`chem_rhs`, `chem_jac`, `block_thomas`, `apply_diffusion`, photo kernels, `jax_ros2_step`).
- It's a numerical bridge to VULCAN-master (anything in the agreement table above).
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
- **Vectorize.** Per-layer operations should be `vmap`'d, not Python-looped. Reach for `jax.lax.scan` only when there is a true sequential dependency (e.g., block-Thomas forward elim).
- **Any code change ends with a doc pass.** If behavior, support surface, benchmark numbers, remaining-work scope, or vendored-data status changed, update `STATUS.md`, `README.md`, and `ALL_REMAINING_JAX_REWRITE_WORK.txt` in the same pass before considering the task done. Update `CLAUDE.md` too when the maintainer rules themselves changed.

## Toward fully-JAX (open work)

Phase 10 of the JAX port is complete (10.1 → 10.7): the outer loop is a single JIT'd `jax.lax.while_loop` with no NumPy hot-path code, no `op.Ros2` inheritance, no `op.Integration` parent class, and no non-Ros2 fallback. `pytest tests/` discovers and runs the suite. Track per-component progress in `STATUS.md`, not here.

Phase 19 (typed pre-loop pytree foundation + live-output drop) closed: the public input schema for the runner now lives in `state.py` (`AtmInputs` / `RateInputs` / `PhotoInputs` / `IniAbunOutputs` / `RunState`). Phase 20 closed the `Atm.*` half of the setup rewrite (`atm_setup.py`); Phase 21 closed the `InitialAbun.*` half (`ini_abun.py`) and **deleted `build_atm.py`**. The `compo` / `compo_row` / `species` data tables now live in `composition.py`. `tests/test_atm_setup_matrix.py` (Phase 20) and the parametrized `tests/test_ini_abun.py` (Phase 21) together cover `atm_type ∈ {isothermal, analytical, file}`, `Kzz_prof ∈ {const, JM16, Pfunc, file}`, `atm_base ∈ {H2, N2, O2, CO2}`, BC flux modes, 8 condensables, and `ini_mix ∈ {EQ, const_mix, vulcan_ini, table, const_lowT}` (≤1e-13 vs numpy/scipy reference; const_lowT JAX Newton matches scipy fsolve to ~1e-16). Live-plot / live-flux / save-movie are no longer supported — `runtime_validation.py` rejects those flags loudly; matplotlib post-run plotters (`plot_end` / `plot_evo` / `plot_TP`) survive.

Phase 22a (rate-parser consolidation) closed: `rates.py` now owns three new entry points — `apply_lowT_caps` (3 hardcoded reactions, masked `np.where`, mirrors `legacy_io.lim_lowT_rates`), `apply_remove_list` (literal zero of indices in `cfg.remove_list`), and `build_rate_array` (forward → optional lowT caps → reverse via Gibbs → remove pass; returns dense `(nr+1, nz)` `np.ndarray`). `vulcan_jax.py` routes through `build_rate_array` in place of the legacy `rev_rate / remove_rate / lim_lowT_rates` chain (legacy `read_rate` retained for metadata population: `var.Rf`, `var.pho_rate_index`, `var.n_branch`, `var.photo_sp`, etc.). New `Variables.k_arr` field on `store.Variables` carries the dense mirror; `var.k` dict is populated via `rates.k_dict_from_array` for back-compat with photo / conden writers and the `.vul` writer until 22b retires the dict surface. Bit-exact (0.0 abs/rel err) vs legacy chain on HD189. New `tests/test_read_rate.py` (6 tests at ≤1e-13).

Phase 22b (photo cross-section preprocessing rewrite) closed: new `photo_setup.py` (~430 lines, pure NumPy) replaces `legacy_io.ReadRate.make_bins_read_cross`. Host-side CSV readers + `_make_bins` two-resolution wavelength grid + `_interp_zero_extrap` / `_interp_edge_extrap` (sort-then-`np.interp` for bit-exact match to `scipy.interp1d` on non-monotonic data — CH3SH_branch.csv has a `354.0` typo that exposes this). T-dependent path vectorises the per-(sp, T) wavelength bin pre-binning out of the per-layer loop and uses the legacy `log10` / `-100` sentinel pattern. Top-level `populate_photo_arrays(var, atm)` mirrors the legacy mutation contract — populates `var.threshold` / `var.bins` / `var.cross*` / `var.cross_J*` / `var.cross_scat` / `var.cross_T*` / `var.cross_Jion*` / runtime-array zeros (`sflux`, `dflux_u/d`, `aflux`, `tau`, `sflux_top`). `vulcan_jax.py` calls it in place of `rate.make_bins_read_cross` and re-syncs `var.k_arr` after photo-J writes. New `tests/test_photo_setup.py` (2 tests at 0.0 err: HD189 baseline + Earth-style T-dep path with `T_cross_sp=['CO2','H2O','NH3']`). End-to-end HD189 5-step smoke runs cleanly (3.1s wall).

Phase 22c (rate-method cleanup) closed: `legacy_io.ReadRate.{rev_rate, remove_rate, lim_lowT_rates}` deleted. Two new convenience helpers in `rates.py`: `setup_var_k(cfg, var, atm) -> Network` (parses network, loads NASA-9, builds dense `var.k_arr`, mirrors to dict) and `apply_photo_remove(cfg, var, network, atm)` (re-syncs `var.k_arr` from dict after photo-J writes, applies `cfg.remove_list`, mirrors back). `vulcan_jax.py` uses `setup_var_k` + `apply_photo_remove` instead of the legacy chain; `tests/conftest.py` `_hd189_pristine` fixture and ~15 standalone test files / examples / benchmarks migrated similarly. Photo-block `make_bins_read_cross` callers migrated to `photo_setup.populate_photo_arrays`. Oracle tests using VULCAN-master's `op.ReadRate()` are unaffected. `read_rate` retained for metadata population (var.Rf, var.pho_rate_index, ...); `make_bins_read_cross` retained for `tests/test_photo_setup.py`'s bit-exact regression oracle (deletion deferred to Phase 22e). Suite: 86 → 94 pass + 1 skip; HD189 50-step oracle still bit-exact at 1.59e-10.

Phase 22d (`var.k` dict-surface retirement) closed: photo writers (`op_jax.compute_J` / `compute_Jion`) now write the per-(sp,nbr) J-rates straight into the dense `var.k_arr[ridx, :]` (no dict mirror). `outer_loop._unpack_k` / `_unpack_conden_k` collapsed to a single full-array snapshot from `state.k_arr` into `var.k_arr`. `state.pytree_from_store` reads `var.k_arr` directly (no per-row dict pack); `state.apply_pytree_to_store` writes `var.k_arr`. `rates.setup_var_k` / `rates.apply_photo_remove` no longer mirror to `var.k`. `legacy_io.read_rate` keeps a local `k = {}` as scratch; `Variables.__init__` drops `self.k = {}`. The `.vul` writer adapter in `legacy_io.Output.save_out` synthesizes the legacy `{i: array(nz)}` dict view from `var.k_arr` at pickle time so downstream `plot_py/` scripts that index `d['variable']['k'][i]` keep working unchanged. Migrated 11 test/example/benchmark files (`test_outer_loop_photo` / `test_chem_jac_sparse` / `test_integrate` / `test_vmap_step` / `test_backend_parity` / `test_read_rate` / `test_photo_ion` / `tests/profile_step` plus `examples/grad_implicit_example` / `examples/batched_run` / `benchmarks/bench_step`) off `var.k.items()` to `var.k_arr` indexing. The `var.cross*` dict surface is retained pending the parallel `photo.pack_*_data` rewrite (deferred to Phase 22e — the photo runtime kernels still consume the dicts at startup). HD189 50-step oracle still bit-exact at 1.59e-10.

Phase 22e (`var.cross*` retirement) closed: new `state.PhotoStaticInputs` NamedTuple holds the dense cross-section pytree (`absp_cross` / `absp_T_cross` / `cross_J` / `cross_J_T` / `cross_Jion` / `scat_cross` plus species/branch ordering tuples and bin-grid scalars). `photo_setup.populate_photo` builds it; `photo_setup.populate_photo_arrays` is now a thin compat wrapper that calls `populate_photo` and discards the return so the ~17 standalone test/example/benchmark callers keep working. `photo.pack_photo_data` / `pack_photo_J_data` / `pack_photo_ion_data` are deleted; `photo.photo_data_from_static` / `photo_J_data_from_static` / `photo_ion_data_from_static` are pure selectors over the pytree. `op_jax.Ros2JAX` accepts `photo_static=` and lazily constructs the static via `_ensure_photo_static(var, atm)` if not wired (covers test sites that do `Ros2JAX()` then `compute_tau(var, atm)`). `outer_loop._build_photo_static` reads from `solver._photo_static` (lazy-built when needed). `vulcan_jax.py` builds the static after `read_sflux`, `_replace`s `din12_indx` from `var.sflux_din12_indx`, stashes it on the solver, and threads it into `output.save_out(..., photo_static=...)`. `legacy_io.Output.save_out` adds `_synthesize_cross_dicts` that rebuilds the six legacy dict views (`cross` / `cross_J` / `cross_T` / `cross_J_T` / `cross_Jion` / `cross_scat`) from the pytree at pickle time so the `.vul` schema stays unchanged for `plot_py/` consumers. `legacy_io.ReadRate.make_bins_read_cross` is deleted (~285 lines); `store.Variables` drops the `self.threshold = {}` / `self.cross_T_sp_list = {}` defaults (now write-once locals during photo setup). `tests/test_photo_setup.py` migrated to compare the static against `tests/data/photo_setup_hd189_baseline.npz` + `tests/data/photo_setup_hd189_T_dep.npz` (no legacy oracle dependency); `tests/test_photo.py` and `tests/test_photo_ion.py` consume the static directly. HD189 50-step oracle still bit-exact at 1.59e-10.

Remaining open work:
- **Backend-parity benchmarking on real GPU hardware.** The kernels are GPU-ready and the repository now includes standalone backend-safe benchmark scripts, but checked-in CPU↔GPU float64 parity numbers still need to be recorded on an actual GPU machine.
- **Per-test fixture sharing**: `pytest tests/` runs the suite today via thin `def test_main(): assert main() == 0` wrappers; the Phase 18 `_hd189_pristine` / `hd189_state` fixtures are in place and one test (`test_chem_jac_sparse`) is migrated. Remaining tests can switch over incrementally; a handful that mutate `vulcan_cfg` need a snapshot/restore wrapper. Parallel execution (`pytest -n auto`) collides on FastChem's fixed output path — run serially.
