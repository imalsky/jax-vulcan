# VULCAN-JAX file index

A per-file map of the source tree. One section per `.py` file with a one-line
file purpose and a one-line description of every public function / class /
method. Subdirectory data folders are summarised at the bottom.

## Top-level driver and configuration

### `vulcan_jax.py`
Driver script. Builds a `RunState` from `vulcan_cfg`, runs the JIT'd outer
loop, pickles the `.vul` output, and optionally fires end-of-run plotters.

### `vulcan_cfg.py`
The config the runtime reads. Imports a preset from `cfg_examples/` and
defines knobs that override the preset. Replace its contents to switch
runs.

### `phy_const.py`
Physical constants in CGS (Boltzmann, Avogadro, Planck-c, AU, R_sun,
R_jup, surface-albedo default). No functions.

## State and pre-loop pipeline

### `state.py`
NamedTuple-based JAX pytrees that carry every input the runner reads,
plus the canonical pre-loop builder.

- `AtmInputs` — atmosphere structure pytree (P, T, Kzz, mu, dz, …).
- `RateInputs` — forward / reverse / 3-body rate-constant arrays.
- `IniAbunOutputs` — initial number densities + mixing ratios.
- `PhotoInputs` — runtime photo input slots.
- `PhotoStaticInputs` — dense per-species cross-section pytree.
  - `with_din12_indx(idx)` — return a copy with `din12_indx` set.
- `StepInputs` — current y / ymix / t / dt / longdy carry.
- `ParamInputs` — convergence and retry counters.
- `AtomInputs` — per-element conservation diagnostics.
- `PhotoRuntimeInputs` — tau / aflux / sflux / dflux carriers.
- `FixSpeciesInputs` — fixed-species snapshot pytree.
- `RunMetadata` — host-side static metadata (Rf, photo_sp, gs, …).
- `RunState` — umbrella pytree wrapping all of the above.
  - `with_pre_loop_setup(cfg)` — build a fully-initialised `RunState` in one call.
  - `fresh_from_cfg(cfg)` — same but with zero-valued runtime slots (test use).
- `StellarFlux` — stellar-flux read result pytree.
- `load_stellar_flux(cfg)` — read the stellar flux file and clamp the bin range.
- `pytree_from_store(var, atm)` — snapshot a legacy `(var, atm)` into a `RunState`.
- `apply_pytree_to_store(rs, var, atm)` — write a `RunState`'s arrays back to legacy containers.
- `runstate_from_store(var, atm, para)` — pre-loop snapshot of all runtime slots.
- `runstate_to_store(rs, var, atm, para)` — round-trip back to legacy classes.
- `legacy_view(rs)` — `SimpleNamespace` shim so legacy `var.attr` access still works.

### `composition.py`
Loads `cfg.com_file` once and exposes per-species composition and mass
arrays for the rest of the runtime. Module-level data, no functions.

### `phy_const.py`, `vulcan_cfg.py`, `composition.py` are imported by
nearly every runtime module; treat them as the global config surface.

## Setup pipelines

### `atm_setup.py`
JAX-native atmosphere setup. Pure functions plus a thin `Atm` facade for
legacy callers.

- `compute_pico(pco)` — stagger pressure to interfaces.
- `analytical_TP_H14(pco, args, gs, Pb)` — Heng et al. 2014 analytical T(P) profile.
- `_interp_descending_or_ascending(x, xp, fp)` — direction-agnostic linear interp.
- `_read_atm_table(path)` — tab-delimited atm-table reader.
- `load_TPK(cfg, pco, pico=None)` — load T / Kzz / vz / M / n_0 per `atm_type`.
- `compute_mean_mass(ymix, ms)` — per-layer mean molecular mass.
- `_scan_up_mu_dz_g`, `_scan_down_mu_dz_g` — sequential `lax.scan` halves of the hydrostatic loop.
- `compute_mu_dz_g(cfg, ymix, ms, pico, Tco)` — full hydrostatic refresh; returns mu / dz / g / Hp / zco / pref.
- `compute_settling_velocity(cfg, Tco, g, species, rho_p, r_p)` — gravitational-settling velocities.
- `_Dzz_gen_for_base(atm_base)` — molecular-diffusion coefficients for the chosen ambient gas.
- `_alpha_array_for_base(atm_base, species, mol_mass)` — thermal-diffusion exponents.
- `compute_mol_diff(cfg, Tco, n_0, g, Hp, dz, ms, alpha, species)` — Dzz_cen / vm assembly.
- `read_sflux_binned(cfg, bins)` — read stellar flux file, rebin onto the photo-grid.
- `_parse_bc_file(path, allow_negative)` — tab-delimited boundary-flux reader.
- `read_bc_flux(cfg, species)` — assemble top/bot flux + deposition velocity arrays.
- `compute_sat_p(condense_sp, Tco)` — per-species saturation pressure.
- `Atm` — thin facade that mutates `data_atm` for legacy callers.
  - `f_pico`, `load_TPK`, `TP_H14`, `mol_mass`, `mean_mass`, `f_mu_dz`, `mol_diff`, `BC_flux`, `sp_sat`, `read_sflux` — wrappers around the pure functions above.

### `atm_refresh.py`
JAX kernels for the in-loop atmosphere refresh (mu / dz / g / Hp +
diffusion-limited escape).

- `AtmRefreshStatic` — NamedTuple of static atm inputs needed by the refresh.
- `update_mu_dz_jax(ymix, atm_static)` — recompute mu, g, Hp, dz, zco, dzi, Hpi from `ymix`.
- `update_phi_esc_jax(...)` — diffusion-limited escape velocity cap.

### `composition.py`
Loads `vulcan_cfg.com_file` once; exposes `species`, `compo`, `compo_row`,
`atom_list`, and a precomputed `compo_array` (ni × n_atoms) JAX pytree.

### `ini_abun.py`
Initial-abundance setup for all five `ini_mix` modes (`EQ`, `const_mix`,
`vulcan_ini`, `table`, `const_lowT`).

- `_fastchem_solar_abundance_path()` — resolves the FastChem solar-element abundance file.
- `_abun_lowT_residual(x, O_H, C_H, He_H, N_H)` — 5-mol residual for the cold-start system.
- `_jax_newton(residual_fn, m0, args, max_iter, tol)` — small dense Newton via `lax.while_loop`.
- `compute_atom_ini(y, compo_arr)` — per-element column sum of initial abundances.
- `_run_fastchem(data_atm)` — run the FastChem binary under the cross-process flock.
- `_run_fastchem_locked(data_atm)` — inner FastChem driver (caller holds the lock).
- `compute_initial_abundance(cfg, atm)` — top-level dispatch returning `IniAbunOutputs`.
- `InitialAbun` — legacy facade with `ini_y` / `ele_sum` mutators.

### `gibbs.py`
NASA-9 polynomial Gibbs energy + reverse-rate computation.

- `load_nasa9(species)` — load NASA-9 coefficients for a species list.
- `gibbs_sp_vector(coeffs, T)` — per-species g/(RT) at the layer T grid.
- `K_eq_array(network, g_sp, T)` — reaction equilibrium constants.
- `fill_reverse_k(network, k, K_eq)` — write reverse rates into the rate array.
- `compute_all_k(cfg, network, T, M)` — full forward+reverse rate assembly.

### `rates.py`
Forward rate-constant evaluation (Arrhenius, Lindemann falloff, 3-body,
hardcoded Troe form) + low-T caps and remove-list bookkeeping.

- `_arrhenius(a, n, E, T)` — modified Arrhenius.
- `_troe_OH_CH3(T, M)` — hardcoded Troe form for OH+CH3+M (Jasper 2017).
- `compute_forward_k(network, T, M)` — full forward-rate assembly.
- `k_dict_from_array(network, k_arr)` / `k_array_from_dict(network, k_dict)` — convert between dict and array forms.
- `apply_lowT_caps(network, k, T, M)` — Moses+2005 low-T rate caps.
- `apply_remove_list(cfg, k)` — zero `cfg.remove_list` entries.
- `build_rate_array(cfg, network, atm)` — full pre-loop rate assembly.
- `setup_var_k(cfg, var, atm)` — populate `var.k_arr` and friends.
- `apply_photo_remove(cfg, var, network, atm)` — remove photo-replaced reactions.

### `network.py`
Parse a VULCAN-format reaction-network text file.

- `Network` — frozen dataclass with parsed network arrays (1-based reaction indexing).
- `_parse_term(term)` — parse a stoichiometric term like `"2*H"`.
- `_parse_eq(eq)` — split `"A + B -> C + D"` into reactant / product lists.
- `_detect_section(line, current)` — return new section name if `line` is a section marker.
- `_configured_extra_species()` — pull cfg-referenced species not in the reaction text.
- `parse_network(path)` — full network-file parser.
- `summarize(network)` — human-readable summary of a parsed `Network`.

### `chem_funs.py`
Re-exports `ni`, `nr`, `spec_list`, `Gibbs`, `chemdf`, etc. Backed by
SymPy-faithful codegen from `make_chem_funs`.

- `_build_re_dicts(network)` — parse stoichiometry into the master-style `re_dict`.
- `_pack_k_dict(k)` — accept dict-or-array `k`, return the array form codegen expects.
- `chemdf(y, M, k_dict)` — chemistry RHS, codegen-backed (master bit-faithful).
- `symjac(...)` / `neg_symjac(...)` — raise `NotImplementedError` (replaced by `chem.chem_jac_analytical`).
- `h_RT(T, a)` / `s_R(T, a)` / `g_RT(T, a_low, a_high)` — NASA-9 thermodynamic functions.
- `gibbs_sp(name, T)` — per-species g/(RT).
- `cp_R(T, a)` / `cp_R_sp(name, T)` — heat capacity.
- `_K_eq_array_cached(T)` — memoised per-T equilibrium-constant array.
- `Gibbs(i, T)` — equilibrium constant for forward reaction `i` at temperature(s) `T`.

### `make_chem_funs.py`
Per-network codegen for the chem_rhs Python source. Master-faithful term
order, content-hashed cache, JIT'd via JAX's persistent disk cache.

- `_emit_rate_term(net, i, …)` — emit one stoich-replicated `k * y[a] * y[b] * …` line.
- `emit_chem_rhs_source(network)` — generate the full per-network RHS source.
- `chem_rhs_cache_key(network)` — content-hash key for the cache.
- `cache_path_for(network)` — resolve the on-disk source path for `network`.
- `build_chem_rhs(network)` — build (or load from cache) the JIT'd `chemdf` callable.

## Hot path

### `chem.py`
Vectorised JAX chem RHS and Jacobian computation.

- `NetworkArrays` — registered JAX pytree; static `(ni, nr)` aux_data.
- `to_jax(network)` — pack `Network` into JAX arrays.
- `chem_rhs_per_layer_segment_sum(...)` — segment_sum reference RHS (test oracle, also the basis for vmap consistency).
- `chem_rhs_segment_sum` — `vmap` over layers of the per-layer reference.
- `chem_jac_per_layer`, `chem_jac` — `jacrev`-based reference Jacobian (test oracle).
- `chem_jac_analytical_per_layer(...)` — stoichiometry-driven analytical Jacobian (production hot path).
- `chem_jac_analytical` — `vmap` over layers of the analytical Jacobian.
- `chem_rhs_numpy(y, M, k, network)` — NumPy reference RHS (master-faithful term order).

### `solver.py`
Block-tridiagonal Thomas solvers used by the diffusion solve.

- `BlockThomasDiagFactors` — LU factors for reuse across RHS solves.
- `factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)` — factor once for the diagonal-offdiag form.
- `solve_block_thomas_diag_offdiag(factors, rhs)` — solve with a new RHS.
- `block_thomas_diag_offdiag(diag, sup_d, sub_d, rhs)` — factor + solve in one call (hot path).
- `block_thomas(diag, sup, sub, rhs)` — generic dense block-tridiagonal Thomas (fallback).

### `jax_step.py`
Vmap-compatible JAX Ros2 single-step kernel.

- `AtmStatic` — atmosphere parameters held constant across a Ros2 step.
- `DiffGrav` — pre-baked y-independent transport contributions.
- `compute_diff_grav(atm)` — y-independent piece of the molecular-diffusion blocks.
- `_build_diff_coeffs_jax(...)` — eddy + molecular diffusion coefficient assembly.
- `_apply_diffusion_jax(...)` — solve the block-tridiagonal diffusion system.
- `jax_ros2_step(y, M, k, atm, grav, dt, rtol, net)` — one Rosenbrock-2 step.
- `make_atm_static(var, atm, network, cfg)` — build an `AtmStatic` from legacy containers.

### `outer_loop.py`
Single-JIT outer integration loop. Runs the full integration inside one
`lax.while_loop`.

- `_now()` — wall-clock time stamp helper.
- `_UNDERFLOW_DENOM` — `/max(|x|, _)` denominator floor.
- `JaxIntegState` — runner carry pytree (y, t, dt, counts, photo arrays, …).
- `OuterLoop` — main driver class; constructed once, called per run.
  - `__init__(solver, output)` — store the photo wrapper and the `.vul` writer.
  - `__call__(rs)` — execute the JIT'd loop, return the post-integration `RunState`.
  - `_make_runner(...)` — build the `lax.while_loop` body.
  - `_make_photo_branch(...)` — photo update sub-graph.
  - `_make_atm_refresh_branch(...)` — atmosphere refresh sub-graph.
  - `_make_conden_branch(...)` — condensation sub-graph.
- `runner(state)` — JIT'd kernel that drives accept/reject + retries.

### `op_jax.py`
Photochemistry adapter holding lazy `PhotoData` / `PhotoJData` caches.

- `Ros2JAX` — class.
  - `_ensure_photo_static(var, atm)` — build / refresh the cached `PhotoStaticInputs`.
  - `compute_tau(var, atm)` — optical depth, writes to `var.tau`.
  - `compute_flux(var, atm)` — two-stream Eddington RT.
  - `compute_J(var, atm)` — photolysis rates, writes to `var.J_sp` + `var.k_arr`.
  - `compute_Jion(var, atm)` — photoionisation rates.
  - `naming_solver(para)` — print transport / BC summary; stamp `para.solver_str`.

### `photo.py`
JAX photochemistry kernels.

- `PhotoData` / `PhotoJData` — pre-stacked cross-section pytrees.
- `photo_data_from_static(static, species_list)` — build runtime `PhotoData` from a `PhotoStaticInputs`.
- `compute_tau_jax(y, dz, photo)` — top-down cumulative optical depth.
- `compute_flux_jax(...)` — two-stream RT (forward + back fluxes, scattering, ground albedo).
- `compute_J_jax(aflux, photoJ)` — branch-resolved photodissociation rates.
- `compute_Jion_jax(aflux, photoJion)` — branch-resolved photoionisation rates.
- `_compute_J_inner(...)` — shared trapezoidal integrator over the two-resolution grid.
- `_pack_branch_to_k_index_map(...)`, `pack_J_to_k_index_map(...)`, `pack_Jion_to_k_index_map(...)` — reaction-index lookup helpers.
- `update_k_with_J(k_arr, J_sp_arr, idx_map)` — write photo rates into `k_arr`.

### `photo_setup.py`
Host-side cross-section preprocessing. Builds the wavelength bin grid
and interpolates per-species cross sections + branch ratios onto that
grid.

- `_cross_folder()` — return `cfg.cross_folder` as a string.
- `_load_thresholds(species)` — read per-species photodissociation thresholds.
- `_load_cross_csv(sp, use_ion)` — read `{sp}_cross.csv` (3- or 4-column).
- `_load_branch_csv(sp)` — read `{sp}_branch.csv` (auto-detected columns).
- `_load_ion_branch_csv(sp)` — read `{sp}_ion_branch.csv`.
- `_discover_T_cross_files(sp)` — list T values of `{sp}_cross_{T}K.csv`.
- `_load_T_cross_csv(sp, T, use_ion)` — read T-dependent cross section.
- `_load_rayleigh_csv(sp)` — read Rayleigh scattering data.
- `_make_bins(...)` — two-resolution wavelength bin grid.
- `_sort_pairs`, `_interp_zero_extrap`, `_interp_edge_extrap`, `_interp_T_log_pair` — interpolation helpers.
- `_bin_cross_and_branches(...)`, `_bin_T_dependent(...)` — rebin per-species data onto the photo grid.
- `populate_photo_arrays(var, atm)` — write photo arrays back to legacy `var` / `atm`.
- `_build_photo_static_dense(var, atm)` — build a fresh `PhotoStaticInputs`.
- `build_photo_static(cfg)` — public builder (tests + external callers).
- `populate_photo(cfg, runtime)` — return a `PhotoStaticInputs` from a cfg.

### `conden.py`
Pure-JAX condensation kernels.

- `CondenStatic` — NamedTuple of static condensation inputs.
- `update_conden_rates(y, atm, conden_static, ymix)` — recompute condensation/evaporation rate constants.
- `apply_h2o_relax_jax(y, atm, …)` — implicit-Euler H2O cold-trap relaxation.
- `apply_nh3_relax_jax(y, atm, …)` — analogous NH3 cold-trap relaxation.

### `integrate.py`
Fixed-`dt` JAX integration loop used for validation and benchmarks.

- `jax_integrate_fixed_dt(y0, M, k, atm, dt, n_steps, network)` — take `n_steps` fixed-dt Ros2 steps; `n_steps` is a static argument.

## Differentiability

### `steady_state_grad.py`
Implicit-function-theorem gradients of the converged photochemical
state. Uses `jax.custom_vjp` for O(1)-memory reverse-mode AD.

- `SteadyStateInputs` — differentiable input pytree.
- `build_steady_state_inputs(k_arr, atm)` — pack a runtime `AtmStatic` plus `k_arr`.
- `_atm_from_inputs(inputs)` — repack a `SteadyStateInputs` into an `AtmStatic`.
- `steady_state_residual_inputs(y, inputs, network, grav)` — `f(y, inputs) = chem_rhs + diffusion`.
- `steady_state_residual(y, k_arr, atm, network, grav)` — convenience wrapper for callers with `AtmStatic`.
- `_build_jacobian_blocks(y, k_arr, atm, network)` — per-layer dense diagonal + diagonal off-diagonal blocks.
- `validate_steady_state_solution(...)` — sanity-check residual norm against bound.
- `differentiable_steady_state_inputs(...)` / `checked_differentiable_steady_state(...)` — main implicit-AD APIs.
- `steady_state_value_and_grad(...)` — full value-and-gradient routine.
- `differentiable_steady_state(...)` — backwards-compatible wrapper.

## Validation, I/O, and host-side glue

### `runtime_validation.py`
Pre-run configuration validation.

- `_validate_network_assets(cfg, root)` — check every species / photo / atom file referenced by `cfg`.
- `_validate_numerical_bounds(cfg)` — bound-check tuning knobs to catch typos at validate time.
- `validate_runtime_config(cfg, root)` — top-level entry point used by the driver and tests.

### `legacy_io.py`
Vendored `ReadRate` / `Output` from VULCAN-master/op.py.

- `_master_tableau20()` — return master's normalised Tableau-20 plotting palette.
- `ReadRate` — class.
  - `__init__()` — set parser scratch.
  - `read_rate(var, atm)` — populate host-side metadata dicts (`var.Rf`, `pho_rate_index`, `n_branch`, `photo_sp`, `ion_sp`, …).
- `Output` — class.
  - `save_cfg(dname)` — copy the active `vulcan_cfg.py` into `output/` for provenance.
  - `save_out(rs, dname)` — write the `.vul` pickle (synthesises photo / ion / parameter dicts from `RunState`).
  - `plot_end(var, atm, para)` / `plot_evo(var, atm)` — end-of-run plotters.
  - `plot_evolution_movie(...)`, `plot_flux_movie(...)` — movie writers.
  - `plot_TP(atm)` — temperature/pressure profile QC plot.

### `live_ui.py`
Host-side live-UI dispatcher. Fires between JIT'd step chunks when any
of `use_live_plot` / `use_live_flux` / `use_save_movie` / `use_flux_movie`
is True.

- `any_live_flag_on(cfg)` — True if any of the four live-UI flags is set.
- `LiveUI` — class.
  - `_ensure_mpl()` — lazy-import matplotlib with a headless-safe backend.
  - `_ensure_species_index()` — cache and return `species -> column_index`.
  - `dispatch(var, atm, para)` — route to mixing-ratio / flux updaters per cfg.
  - `update_mix(var, atm, para, save_movie, show)` — render the mixing-ratio panel.
  - `update_flux(var, atm, para, save_movie, show)` — render the diffusive-flux panel.

## Subdirectories

### `cfg_examples/`
Reference configs. Copy one to `vulcan_cfg.py` at the repository root and
run `python vulcan_jax.py`.

- `vulcan_cfg_HD189.py` — HD 189733b reference. Matches `VULCAN-master/cfg_examples/vulcan_cfg_HD189.py` for cross-version smoke tests.
- `vulcan_cfg_HD209.py` — HD 209458b (no S species, weaker gravity).
- `vulcan_cfg_Earth.py` — Earth troposphere/stratosphere with condensation.
- `vulcan_cfg_W39b.py` — WASP-39b paper-match config (Wogan et al.).
- `README.txt` — short description + which cfg is the matched one.

### `tests/`
Curated suite focused on hot-path kernels, oracle agreement, and JAX
transform consistency. Run with
`python -m pytest tests -q --tb=short -ra` from the repo root.

- `conftest.py` — session-scoped HD189 pre-loop fixture, cfg snapshot/restore autofixtures.
- `data/oracle_baselines/{earth,hd209}_20step.npz` — oracle reference snapshots.
- `data/photo_setup_hd189_{baseline,T_dep}.npz` — photo-setup test fixtures.
- `diffusion_numpy_ref.py` — NumPy oracle for diffusion kernels (used by `test_diffusion*.py`).
- `test_chem.py`, `test_chem_jac_sparse.py`, `test_chem_rhs_codegen.py` — chemistry RHS / Jacobian agreement.
- `test_block_thomas.py`, `test_block_thomas_diag.py` — block-tridiagonal solvers.
- `test_diffusion.py`, `test_diffusion_variants.py` — diffusion operator + Jacobian assembly.
- `test_ros2_step.py` — single-step Rosenbrock kernel.
- `test_conden_jax.py` — condensation kernels.
- `test_photo.py`, `test_photo_ion.py`, `test_photo_setup.py` — photo kernels and cross-section preprocessing.
- `test_gibbs.py`, `test_rates.py`, `test_read_rate.py`, `test_network_parse.py` — setup parsers.
- `test_ini_abun.py` — all five `ini_mix` modes.
- `test_atm_setup_matrix.py` — atm-variant branches HD189 doesn't exercise.
- `test_state_roundtrip.py` — `RunState` ↔ pytree ↔ legacy `(var, atm, para)`.
- `test_save_evolution.py` — `save_evolution=True` cadence.
- `test_oracle.py` — Earth + HD209 20-step oracle vs VULCAN-master (skips cleanly if absent).
- `test_outer_loop_smoke.py` — HD189 50-step smoke (the headline regression test).
- `test_outer_loop_atm_refresh.py`, `test_outer_loop_conden_gate.py`, `test_outer_loop_conv.py`, `test_outer_loop_ion.py`, `test_outer_loop_photo.py` — outer-loop sub-graph tests.
- `test_w39b_fastchem_invariant.py` — frozen FastChem snapshot for W39b.
- `test_use_fix_H2He.py`, `test_solver_fix_all_bot.py` — boundary-condition variants.
- `test_vmap_kernels.py`, `test_vmap_step.py` — JAX vmap consistency.
- `test_steady_state_grad.py` — implicit-AD reverse-mode gradients.
- `test_cfg_examples.py` — each kept config loads + runs pre-loop setup.
- `test_config_matrix.py` — config-flag combination coverage.

### `benchmarks/`
- `bench_step.py` — per-step JAX timing + comparison to VULCAN-master if present.

### `examples/`
- `batched_run.py` — `jax.vmap` over batched atmospheres for parameter sweeps.
- `grad_jvp_example.py` — forward-mode AD through the per-step kernel.
- `grad_implicit_example.py` — reverse-mode AD through the converged steady state via `steady_state_grad`.

### `tools/`
End-user utility scripts (data prep, debug).

- `make_mix_table.py` — build a mixing-ratio table for `ini_mix='table'`.
- `make_spectra_in_nm.py` — convert stellar spectra to nm wavelength bins.
- `print_actinic_flux.py` — print actinic flux from a `.vul` file.

### `atm/`
Vendored atmosphere structure files (TP, Kzz, vz profiles for Earth,
HD189, HD209, W39b, …) plus boundary-condition flux files. Includes
`atm/stellar_flux/` with sflux files and a couple of helper scripts for
spectra preprocessing.

### `thermo/`
Reaction-network text files (NCHO, SNCHO, CHO, …), the `all_compose.txt`
composition table, NASA-9 polynomial coefficients
(`thermo/NASA9/{species}.txt`), and per-species photochemistry CSVs
under `thermo/photo_cross/`.

### `fastchem_vulcan/`
Vendored FastChem binary plus its runtime payload (`input/`, `output/`).
Concurrent invocations are serialised via `fcntl.flock` so
`pytest -n auto` is safe.

### `output/`, `plot/`
Created at run time by the driver and live-UI. Both are produced from
scratch — safe to delete and they will reappear on next run.

### `../comparisons/`
The cross-version validation oracle (sits at the project parent, *not*
inside `VULCAN-JAX/`). Contains `compare_vul.py` (the `.vul` diff
utility), the W39b paper-match comparison, and the matched
JAX/master comparison configs.
