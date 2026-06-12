# VULCAN-JAX file index

A per-file map of the source tree. One section per `.py` file with a one-line
file purpose and a one-line description of every public function / class /
method. Private helpers are documented when they carry non-trivial behavior;
trivial private helpers are summarised in groups.

> **Convention:** entries marked **[Δ master]** flag a design choice that
> diverges materially from VULCAN-master (Tsai 2017/2021). Everything else
> is a straight port. Where the master pipeline still owns the metadata side
> (e.g. host-side rate parsing), the matching VULCAN-JAX file says so
> explicitly.

## Table of contents

1. [Top-level driver and configuration](#top-level-driver-and-configuration)
2. [State and pre-loop pipeline](#state-and-pre-loop-pipeline)
3. [Setup pipelines](#setup-pipelines)
4. [Hot path (per-step kernels and the JIT'd runner)](#hot-path-per-step-kernels-and-the-jitd-runner)
5. [Differentiability](#differentiability)
6. [Validation, I/O, and host-side glue](#validation-io-and-host-side-glue)
7. [Subdirectories](#subdirectories)

---

## Top-level driver and configuration

### `vulcan_jax_cli.py`
Driver / console entry point `cli_main` (the `vulcan-jax` script; ~80 lines).
Mirrors `vulcan.py`'s orchestration. Sets the JAX
persistent-compilation-cache dir, runs `validate_runtime_config(cfg, ROOT)`,
builds `RunState.with_pre_loop_setup(cfg)`, instantiates `Ros2JAX` +
`OuterLoop`, calls `integ(runstate)`, and pickles the `.vul` output. Fires
end-of-run plotters (`plot_end`, `plot_evo`) on the legacy `(var, atm,
para)` view if the corresponding cfg flags are set.
**[Δ master]** Single device-side call replaces master's Python
`while not stop()` loop.

### `vulcan_cfg.py`
The config the runtime reads. The canonical committed default is a
thin wrapper:
```python
from cfg_examples.vulcan_cfg_HD189 import *
```
plus a small override block for plotting / output. Replace the import to
switch presets (`vulcan_cfg_HD209`, `vulcan_cfg_Earth`, `vulcan_cfg_W39b`),
or copy a preset over the file outright.

### `phy_const.py`
Physical constants in CGS, sourced from astropy. Module-level only — no
functions. Defines `kb` (Boltzmann), `Navo` (Avogadro), `hc` (Planck * c
in erg·nm), `au`, `r_sun`, `r_jup`, and `ag0` (RT asymmetry factor; `0` =
isotropic scattering).

---

## State and pre-loop pipeline

### `state.py`
NamedTuple-based JAX pytrees that carry every input the runner reads,
plus the canonical pre-loop builder. **[Δ master]** Master mutates open
`Variables` / `AtmData` / `Parameters` classes throughout the run; JAX
captures the same state in a typed, registered-pytree
`RunState`, so the integration loop is JIT-traceable and the runtime
surface is differentiable. Legacy class shims (`_Variables` / `_AtmData`
/ `_Parameters`) remain in this file as private scratch for hybrid
oracle tests; they are not on the production runtime path.

Public input pytrees:
- `AtmInputs` — atmosphere structure (P, T, Kzz, mu, dz, dzi, zco, vs, BC arrays, …).
- `RateInputs` — `(nr+1, nz)` rate-constant array `k` (forward + reverse, 1-based indexing).
- `IniAbunOutputs` — initial number densities `y_ini`, mixing ratios `ymix_ini`, atom inventory `atom_ini`.
- `PhotoInputs` — small runtime photo slots (`sflux_top`, `def_bin_min`, `def_bin_max`).
- `PhotoStaticInputs` — dense per-species photo cross-section pytree.
  - `with_din12_indx(idx)` — return a copy with `din12_indx` set.
- `StepInputs` — runner carry: `y`, `ymix`, `t`, `dt`, `longdy`, `longdydt`, `aflux_change`, photo/atm cadence counters.
- `ParamInputs` — convergence + retry counters (`count`, `nega_count`, `loss_count`, `delta_count`, `end_case`, `pic_count`, …).
- `AtomInputs` — per-element conservation diagnostics (`atom_loss`, `atom_loss_prev`, ratio history).
- `PhotoRuntimeInputs` — `tau`, `aflux`, `sflux`, `dflux_u/d`, `J_sp`, `Jion_sp` carriers.
- `FixSpeciesInputs` — `fix_species` snapshot pytree (snapshot mixing ratios, masks, conden_min_lev).
- `RunMetadata` — host-side static metadata (`Rf`, `n_branch`, `ion_branch`, `photo_sp`, `ion_sp`, `pho_rate_index`, `ion_rate_index`, `ion_br_ratio`, `charge_list`, `conden_re_list`, `start_time`, `Ti`, `gas_indx`, `pref_indx`, `gs`, `sat_p`, `sat_mix`, `r_p`, `rho_p`, `fix_sp_indx`, `y_ini`).
- `RunState` — umbrella pytree wrapping everything above.
  - `with_pre_loop_setup(cfg)` (classmethod) — **canonical entry point**: runs the entire pre-loop pipeline (`atm_setup`, `rates`, `ini_abun`, `photo_setup` + photo-remove pass) and returns a fully-populated pytree.
  - `fresh_from_cfg(cfg)` (classmethod) — same but with zero-valued runtime slots (test use).
- `StellarFlux` — return type of `load_stellar_flux`.

Builders / round-trip helpers:
- `load_stellar_flux(cfg)` — read `cfg.sflux_file` and clamp the bin range to `[2, 700] nm`. Returns empty payload when `use_photo=False`.
- `pytree_from_store(var, atm)` — snapshot a legacy `(var, atm)` into a partially-filled `RunState` (atm + rate + photo only).
- `apply_pytree_to_store(rs, var, atm)` — write a `RunState`'s arrays back to the legacy containers.
- `runstate_from_store(var, atm, para)` — pre-loop snapshot of all runtime slots (used by the hybrid oracle path).
- `runstate_to_store(rs, var, atm, para)` — round-trip back into legacy classes.
- `legacy_view(rs)` — `SimpleNamespace` shim so test code indexing `var.attr` directly keeps working.

Private helpers (documented for completeness, not for callers):
- `_atom_order_for(cfg)` / `_atom_dict_to_arr(d, order)` / `_atom_arr_to_dict(arr, order)` — convert between cfg's atom list (minus `loss_ex`) and ordered arrays.
- `_master_tableau20()` — VULCAN-master's normalized Tableau-20 plotting palette.
- `_atm_metadata_from_atm(atm)` — extract static atm metadata.
- `_runmetadata_from_legacy(var, atm, para)` — build a `RunMetadata` from the legacy containers.
- `_build_pre_loop_runstate(cfg)` — backend of `RunState.with_pre_loop_setup`.
- `_fresh_step_inputs(rs)` / `_fresh_param_inputs(rs)` / `_fresh_atom_inputs(rs)` / `_fresh_photo_runtime(rs)` — zero-fill helpers used by `fresh_from_cfg`.
- `_Variables`, `_AtmData`, `_Parameters` — private mutable-container classes (used as scratch inside `_build_pre_loop_runstate` and by hybrid oracle tests; **not** on the production runtime path).

### `composition.py`
Loaded once from `vulcan_cfg.com_file`. Module-level data, no functions.
Exposes `species` (= `chem_funs.spec_list`), the structured `compo` array,
`compo_row`, `atom_list` (composition columns minus `species`/`mass`), and
a precomputed `(ni, n_atoms)` `compo_array` JAX pytree used by
`compute_atom_ini`, `outer_loop._compute_atom_loss`, and the atom-loss
diagnostics.

---

## Setup pipelines

### `atm_setup.py`
JAX-native atmosphere setup. Module-level pure functions plus a thin `Atm`
facade that mutates the legacy `data_atm` for back-compat callers.
**[Δ master]** Master's `build_atm.py` mutates `data_atm` in place from
inside a single class; here every step is a pure function returning a JAX
array, so the same pipeline is callable inside `state.RunState.with_pre_loop_setup`
without holding the legacy object graph.

Pure functions:
- `compute_pico(pco)` — stagger pressure to interfaces.
- `analytical_TP_H14(pco, params, *, gs, Pb)` — Heng et al. 2014 analytical T(P) profile.
- `_interp_descending_or_ascending(x_query, xp_raw, fp_raw, fill_left, fill_right)` — direction-agnostic linear interpolation.
- `_read_atm_table(atm_file)` — tab-delimited atmosphere-table reader.
- `load_TPK(cfg, pco, *, pico)` — load `T / Kzz / vz / M / n_0` per `atm_type`. Handles the 5 modes `isothermal` / `analytical` / `file` / `vulcan_ini` / `table`; the `vulcan_ini` and `table` modes return a fresh `pco`.
- `compute_mean_mass(ymix, ms_arr)` — per-layer mean molecular mass.
- `_scan_up_mu_dz_g(...)` / `_scan_down_mu_dz_g(...)` — sequential `lax.scan` halves of the hydrostatic loop.
- `compute_mu_dz_g(cfg, ymix, ms_arr, pico, Tco)` — full hydrostatic refresh; returns `mu, g, gs, Hp, Hpi, dz, dzi, zco, zmco, pref_indx, Ti`.
- `compute_settling_velocity(cfg, Tco, g, species_list, rho_p, r_p)` — gravitational-settling velocities for `cfg.non_gas_sp`.
- `_Dzz_gen_for_base(atm_base)` — molecular-diffusion coefficients table for the chosen ambient gas (`H2`/`N2`/`O2`/`CO2`).
- `_alpha_array_for_base(atm_base, species_list, mol_mass)` — thermal-diffusion exponents.
- `compute_mol_diff(cfg, Tco, n_0, g, Hp, dz, ms, alpha, species)` — `Dzz` / `Dzz_cen` / `vm` assembly.
- `read_sflux_binned(cfg, bins, sflux_raw=None)` — read the stellar flux file (or accept a pre-read `sflux_raw` for tests) and rebin onto the photo grid; returns `sflux_top`, the `dbin1→dbin2` transition index, and the raw read.
- `_parse_bc_file(path)` — tab-delimited boundary-condition file reader.
- `read_bc_flux(cfg, species_list)` — assemble top/bot flux and deposition-velocity arrays from cfg + BC files.
- `compute_sat_p(condense_sp, Tco)` — per-species saturation pressure. Hardcoded formulae for `H2O`, `NH3`, `H2SO4`, `S2`, `S4`, `S8`, `C`, `H2S`.

Facade:
- `Atm` — thin facade that mutates `data_atm` / `data_var` for legacy callers. Methods (`f_pico`, `load_TPK`, `TP_H14`, `mol_mass`, `mean_mass`, `f_mu_dz`, `mol_diff`, `BC_flux`, `sp_sat`, `read_sflux`) wrap the pure functions above.

### `atm_refresh.py`
JAX kernels for the in-loop atmosphere refresh — recomputes mu/g/Hp/dz when
mixing ratios drift, plus the diffusion-limited escape velocity at TOA.
- `AtmRefreshStatic` — NamedTuple of static atm inputs the refresh needs (ms, pico, gs, dzi, base mu/g/Hp, Tco, kb, Navo, max_flux, Dzz_top, diff_esc_idx, fix_indx).
- `update_mu_dz_jax(ymix, st)` — recompute `mu`, `g`, `Hp`, `dz`, `zco`, `dzi`, `Hpi` from a fresh `ymix`. Implemented as two `lax.scan` passes (up and down).
- `update_phi_esc_jax(y, g, Hp, top_flux_in, st)` — diffusion-limited escape velocity cap; clamps `-max_flux` from below. Writes into `top_flux_in.at[diff_esc_idx]`.

### `ini_abun.py`
Initial-abundance setup for all five `ini_mix` modes (`EQ` / `const_mix`
/ `vulcan_ini` / `table` / `const_lowT`).
**[Δ master]** Master loads the FastChem result through a host-side I/O
mutation of `data_var`; JAX wraps the FastChem invocation in a cross-process
`fcntl.flock` so `pytest -n auto` is safe, and routes the five modes
through a small `_MODE_DISPATCH` table.

- `_fastchem_solar_abundance_path()` — resolve the FastChem solar-element abundance file (defaults to `fastchem_vulcan/input/solar_element_abundances.dat`; can be overridden by `cfg.fastchem_solar_abundance_file`).
- `_abun_lowT_residual(x, O_H, C_H, He_H, N_H)` — 5-mol residual for the cold-start system (`ini_mix='const_lowT'`).
- `_jax_newton(residual_fn, m0, args, max_iter, tol)` — small dense Newton solver via `lax.while_loop`. Defaults come from `cfg.fastchem_newton_*`.
- `compute_atom_ini(y, compo_arr=compo_array)` — per-element column sum of initial abundances.
- `_run_fastchem(data_atm)` — run the FastChem binary under the cross-process flock.
- `_run_fastchem_locked(data_atm)` — inner FastChem driver (caller holds the lock).
- `_build_charge_list_if_ion(charge_list)` — populate the ion species charge list from `cfg.atom_list`.
- `_load_eq_y(data_atm)` — FastChem `'EQ'` path.
- `_load_vulcan_ini_y(data_atm)` — read previous `.vul` output.
- `_load_table_y(data_atm)` — read a tab-delimited mixing-ratio table (see `tools/make_mix_table.py`).
- `_load_const_mix_y(data_atm)` — apply `cfg.const_mix` per layer.
- `_load_const_lowT_y(data_atm)` — cold-start Newton-solved equilibrium.
- `_apply_condense(y, data_atm)` — apply the initial cold-trap clip when `use_ini_cold_trap=True`.
- `_compute_ymix(y)` — mixing-ratio normalisation: `y / sum(y, gas only)`.
- `_MODE_DISPATCH` — `{'EQ': _load_eq_y, 'const_mix': _load_const_mix_y, ...}` table.
- `compute_initial_abundance(data_atm)` — top-level dispatch returning `IniAbunOutputs`.
- `InitialAbun` — legacy facade with `ini_y(data_var, data_atm)` and `ele_sum(data_var)` mutators.

### `gibbs.py`
NASA-9 polynomial Gibbs energy + reverse-rate computation.
- `load_nasa9(species, thermo_dir)` — load NASA-9 coefficients for a species tuple from `thermo_dir/NASA9/`. Returns `(coeffs, present_mask)`.
- `gibbs_sp_vector(coeffs, T)` — per-species `g/(RT)` at the layer T grid.
- `K_eq_array(net, gibbs_sp, T)` — reaction equilibrium constants.
- `fill_reverse_k(net, k, K_eq, remove_list=None)` — write reverse rates into `k` slots `i+1`.
- `compute_all_k(net, T, M, nasa9_coeffs, remove_list=None)` — full forward + reverse rate assembly.

### `rates.py`
Forward rate-constant evaluation (modified Arrhenius, Lindemann falloff,
3-body, hardcoded Troe form) + low-T caps and remove-list bookkeeping.
- `_arrhenius(a, n, E, T)` — modified Arrhenius.
- `_troe_OH_CH3(T, M)` — hardcoded Troe form for `OH+CH3+M` (Jasper 2017).
- `compute_forward_k(net, T, M)` — full forward-rate assembly.
- `k_dict_from_array(net, k_arr)` / `k_array_from_dict(net, k_dict, nz)` — convert between dict and array forms.
- `apply_lowT_caps(net, k, T, M)` — Moses+2005 low-T rate caps for the three reactions VULCAN-master caps explicitly. Caller gates on `cfg.use_lowT_limit_rates`.
- `apply_remove_list(net, k, remove_list)` — zero rows listed in `remove_list`. **No auto-pairing** — passing a lone forward leaves its reverse intact (matches master's semantics).
- `build_rate_array(cfg, net, atm, nasa9_coeffs)` — end-to-end pre-loop rate assembly: `compute_forward_k → apply_lowT_caps → fill_reverse_k → apply_remove_list`.
- `setup_var_k(cfg, var, atm)` — parse the network, load NASA-9, populate `var.k_arr`. Returns the `Network`.
- `apply_photo_remove(cfg, var, network, atm)` — re-apply `cfg.remove_list` after `compute_J` / `compute_Jion` has overwritten the photolysis rows.

### `network.py`
Parse a VULCAN-format reaction-network text file.
- `Network` — frozen dataclass holding the parsed arrays. Public fields: `species`, `species_idx`, `ni`, `nr`, `reactant_idx`, `product_idx`, `reactant_stoich`, `product_stoich`, Arrhenius params (`a`, `n`, `E`, `a_inf`, `n_inf`, `E_inf`), reaction-type masks (`is_forward`, `is_three_body`, `has_kinf`, `is_special`, `is_conden`, `is_radiative`, `is_photo`, `is_ion`), section delimiters (`stop_rev_indx`, `conden_indx`, `radiative_indx`, `photo_indx`, `ion_indx`), photo metadata (`photo_sp`, `pho_rate_index`, `n_branch`, `ion_sp`, `ion_rate_index`, `ion_branch`), reaction-text dicts (`Rf`, `Rindx`), and `network_path`.
  - `species_index(sp)` — return 0-based species index.
- `_parse_term(term)` — parse a stoichiometric term like `"2*H"` into `(stoich, name)`.
- `_parse_eq(eq)` — split `"A + B -> C + D"` into reactant / product lists.
- `_detect_section(line, current)` — section-header dispatch.
- `parse_network(network_path)` — full parser.
- `summarize(net)` — human-readable summary of a `Network`.
**[Δ master]** Master's auto-generated `chem_funs` walks literal reactant/
product lists per forward/reverse slot. The JAX parser captures the same
asymmetry by setting `is_three_body[i]` and `is_three_body[i+1]`
independently for the forward and reverse slots of each reaction — so
reactions like `HNCO + M → H + NCO` (forward 3-body, reverse bimolecular)
survive without going through SymPy codegen.

### `chem_funs.py`
JAX-native module that re-exports the same public surface as VULCAN-master's
auto-generated `chem_funs.py` — `ni`, `nr`, `spec_list`, `re_dict`,
`re_wM_dict`, `chemdf`, `Gibbs`, `gibbs_sp`, `cp_R`, `cp_R_sp`, etc.
Module-level setup parses the network, loads NASA-9, and calls
`make_chem_funs.build_chem_rhs(_NETWORK)` to build / load the codegen RHS.
**[Δ master]** Master *generates* `chem_funs.py` via SymPy. JAX
*re-implements* the public interface and uses a content-hashed
JAX codegen cache (`make_chem_funs.py`) instead, so changing the
network at runtime regenerates only the per-reaction source — no
SymPy step is required.
- `_build_re_dicts(net)` — reconstruct master-style `re_dict` / `re_wM_dict` from the parsed network arrays.
- `_pack_k_dict(k)` — accept dict-or-array `k`, return the `(nr+1, nz)` form the codegen expects.
- `chemdf(y, M, k)` — chemistry RHS at all layers, codegen-backed (master bit-faithful at ~1 ULP per multiply chain).
- `symjac(y, M, k)` / `neg_symjac(y, M, k)` — raise `NotImplementedError`. **[Δ master]** Production uses `chem.chem_jac_analytical` (block-stack form); master's flat banded form is unused on the Ros2 path.
- `h_RT(T, a)` / `s_R(T, a)` / `g_RT(T, a_low, a_high)` — NASA-9 thermodynamic functions.
- `gibbs_sp(name, T)` — per-species `g/(RT)`.
- `cp_R(T, a)` / `cp_R_sp(name, T)` — heat capacity / R.
- `_K_eq_array_cached(T_np)` — memoised per-T equilibrium-constant array (LRU-cached on `T`).
- `Gibbs(i, T)` — equilibrium constant for forward reaction `i` at temperature(s) `T`.
- Re-exports: `chem_rhs_codegen` (the JIT'd codegen RHS, identical to `_CHEM_RHS_CODEGEN`) and `chem_rhs_segment_sum` (the reference segment-sum RHS, for tests and oracles).
- Module-level: `NETWORK = _NETWORK`, `re_dict`, `re_wM_dict`.

### `make_chem_funs.py`
Per-network codegen for the `chem_rhs` Python source. Master-faithful term
order, content-hashed cache, JIT'd via JAX's persistent disk cache.
**[Δ master]** Master ships a single SymPy-generated `chem_funs.py`. JAX
emits Python source that mirrors master's `chemdf` *body* (same per-
reaction multiply chain order, same per-species accumulator order)
**without** SymPy, and caches the source + XLA artifact per (network,
JAX version, device).
- `_emit_rate_term(net, i, …)` — emit one stoich-replicated `k * y[a] * y[b] * …` line.
- `emit_chem_rhs_source(net)` — generate the full per-network RHS Python source.
- `chem_rhs_cache_key(net)` — content-hash key (network mtime + array bytes).
- `cache_path_for(net)` — resolve the on-disk source path for a network.
- `build_chem_rhs(net)` — build (or load from cache) the JIT'd `chemdf(y, M, k_arr)` callable. Memoised in `_BUILD_CACHE` so repeat calls within a process return the same object.

---

## Hot path (per-step kernels and the JIT'd runner)

### `chem.py`
Vectorised JAX chemistry RHS and Jacobian.
**[Δ master]** Master computes the chemistry Jacobian by SymPy + scipy
banded solve. JAX uses a stoichiometry-driven analytical Jacobian
(`chem_jac_analytical`) that walks the network arrays directly and
skips structurally-zero entries; ~36× faster than the `jacrev` reference
on the SNCHO network. The `jacrev` form is kept as a test oracle.
- `NetworkArrays` — registered JAX pytree (children: reactant/product idx + stoich, `is_three_body`; static aux_data: `ni`, `nr`). `jit`/`vmap` don't retrace per-network and callers don't need `static_argnames` everywhere.
- `_network_arrays_flatten` / `_network_arrays_unflatten` — pytree registration hooks.
- `to_jax(net)` — pack a `Network` into `NetworkArrays` of JAX arrays.
- `chem_rhs_per_layer_segment_sum(y, M, k, net)` — segment-sum reference RHS at one layer. Test oracle and vmap-consistency basis. **Not on the production hot path** (codegen-RHS replaced it for master-bit-faithful term ordering).
- `chem_rhs_segment_sum` — `vmap` over layers of the per-layer reference (in_axes `(0, 0, 1, None)`).
- `chem_jac_per_layer`, `chem_jac` — `jacrev`-based reference Jacobian. Test oracle only.
- `_JAC_CHUNK_REACTIONS = 128` — reaction-axis chunk size for the analytical-Jacobian scatter; bounds the per-layer transient that grows linearly with vmap batch width (the batch-512 GPU OOM driver). A code constant, not a config knob: changing it permutes float summation order.
- `chem_jac_analytical_per_layer(y, M, k, net)` — stoichiometry-driven analytical Jacobian for one layer. Builds `J[i,j] = Σ_r sign_i · stoich_i · ∂rate[r]/∂y_j` directly from `NetworkArrays`; the scatter runs as a `lax.scan` over `_JAC_CHUNK_REACTIONS`-sized reaction chunks (`unroll=1`) into a carried `(ni+1)²` accumulator. **Production hot path.**
- `chem_jac_analytical` — `vmap` over layers of the analytical Jacobian.
- `chem_rhs_numpy(y, M, k, net)` — NumPy reference RHS in master-faithful term order; used at `rtol=1e-13` in `tests/test_chem_rhs_codegen.py`.

### `solver.py`
Block-tridiagonal Thomas solvers used by the diffusion solve.
**[Δ master]** The diffusion off-diagonals are diagonal-in-species
(sup/sub are `(nz, ni)` rather than `(nz, ni, ni)`). The dense
`O(ni³)` matmul `C_j @ inv(A_prev) @ B_{j-1}` reduces to an `O(ni²)`
rank update. The dense block-Thomas form is preserved as a fallback for
callers with truly dense off-diagonals.
- `BlockThomasDiagFactors` — NamedTuple of LU factors for reuse across RHS solves.
- `factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)` — factor once for the diagonal-offdiag form (forward sweep producing the modified diagonals and the C/A products needed at solve time).
- `solve_block_thomas_diag_offdiag(factors, rhs)` — solve with a new RHS (forward substitution then backward).
- `block_thomas_diag_offdiag(diag, sup_d, sub_d, rhs)` — factor + solve in one call. Hot-path entry.
- `block_thomas(diag, sup, sub, rhs)` — generic dense block-tridiagonal Thomas (fallback).

### `jax_step.py`
Vmap-compatible JAX Ros2 single-step kernel.
**[Δ master]** Master's per-step Ros2 lives inside `op.Ros2.solver`,
which mutates `(var, atm)`. The JAX version is a pure function
returning `(sol, delta)`; the y-independent piece of the diffusion
blocks is precomputed once per Ros2 step (rather than twice as in
master), and `vmap` over layers is the default.
- `AtmStatic` — NamedTuple of atmosphere parameters held constant across a Ros2 step (compositional masses, BC arrays, geometry, diffusion arrays, fix-species mask, `non_gas_present`, …).
- `DiffGrav` — NamedTuple of pre-baked y-independent transport contributions used inside `_build_diff_coeffs_jax`.
- `compute_diff_grav(atm)` — compute the y-independent diffusion piece once per step.
- `_build_diff_coeffs_jax(y, atm, grav)` — eddy + molecular diffusion coefficient assembly (sup / sub / diag blocks).
- `_apply_diffusion_jax(y, A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm)` — solve the block-tridiagonal diffusion system via `solver.block_thomas_diag_offdiag`.
- `jax_ros2_step(y, k_arr, dt, atm, net, fix_mask=None)` — one Rosenbrock-2 step. Returns `(sol, delta_arr)` where `delta_arr` is the truncation-error proxy `(sol - yk2)`.
- `make_atm_static(atm, ni, nz)` — build an `AtmStatic` from a legacy `data_atm` container.

### `outer_loop.py`
Single-JIT outer integration loop. Runs the full integration inside one
`lax.while_loop`.
**[Δ master]** Master polls termination in Python (`while not stop():
one_step()`). JAX folds termination, accept/reject retries, photo-frequency
switching, adaptive rtol, ring-buffered convergence detection, atmosphere
refresh, condensation, ion charge balance (with master's electron-row
freeze inside both Ros2 stages when `use_ion`), and the `fix_all_bot` /
`fix_H2He` clamps into one device-side `lax.while_loop` body. The host
sees one `integ(rs)` call. A batched variant (`run_batch`) vmaps the same
body across profiles with freeze-on-done lanes; per-profile constants ride
the `ProfileVars` slot of the carry instead of the closures. A `use_chunked_runner=True` mode exists for
live-UI cadence — it runs the same JIT'd body in `live_plot_frq`-sized
chunks so the host can dispatch `print_prog` / live plots between chunks.

Module-level helpers:
- `_now()` — wall-clock time stamp.
- `_UNDERFLOW_DENOM = 1e-300` — numerical floor for `/max(|x|, _)` denominators.
- `_compute_atom_loss(y, compo_arr, …)` — per-element column drift, used by both the in-runner conv check and the post-run diagnostics.
- `_step_size(dt, delta, …)` — Ros2 step-size update (uses `cfg.step_size_safety` and `cfg.step_size_zero_delta_frac`).
- `_make_clip_fn(non_gas_present, gas_indx_mask, mtol, pos_cut, nega_cut)` — closure that builds the `pos_cut`/`nega_cut` clipper (faithful to `op.py`).
- `_make_aggregate_delta_fn(mtol, atol, zero_bot_row, condense_zero_mask)` — closure for the truncation-error aggregator (max relative `delta`).
- `_make_photo_branch(photo_static)` / `_make_atm_refresh_branch(refresh_static)` / `_make_conden_branch(conden_static)` — module-level factories returning the photo / atm-refresh / condensation sub-graph closures. These are **not** methods on `OuterLoop`; they're closed over by `_make_runner` and bound into the runner's body.
- `_make_runner(net, statics, …)` — assemble the full `lax.while_loop` body (cond + body, accept/reject retries, photo/conden/atm-refresh gating). Returns the JIT-able `runner(state, atm_static)` callable. The integration's hot path lives here.
- `ProfileVars` — per-profile constants threaded through the carry so `jax.vmap` batches them per lane (n_0, Kzz, atom_ini, atm-refresh fields, conden diffusion/saturation arrays, the NH3 cold-trap index `c_nh3_conden_top`, and the two T-P-dependent photo statics `p_absp_T_cross` / `p_cross_J_T`).
- `stack_integ_states(states)` / `stack_atm_statics(atms)` / `unstack_integ_states(batched, n)` — build/split the leading batch axis for `run_batch`.
- `JaxIntegState` — runner carry pytree (~80 slots: y, t, dt, counts, longdy ring buffer, photo carry, atm carry, conden carry, atom-loss history, `pv: ProfileVars`, …).
- `_PhotoStatic` — photo sub-graph's frozen NamedTuple (cross sections, branch maps, sflux, scattering tables).
- `_Statics` — umbrella container for the three sub-graph statics (`photo`, `atm_refresh`, `conden`) plus the global atm-static.

Public class:
- `OuterLoop` — main driver class; constructed once, called per run.
  - `__init__(odesolver, output, cfg=vulcan_cfg)` — store the photo wrapper, the `.vul` writer, and the per-instance cfg (`make_config()` users pass their own namespace); set `loss_criteria` default.
  - `reset()` — clear the cached statics + runner (call when switching configs in the same process).
  - `__call__(*args)` — polymorphic entry: if first arg is a `RunState`, dispatch to `_call_runstate`; else fall back to the legacy `(var, atm, para, make_atm)` quadruple. Both paths produce bit-equivalent final states.
  - `_call_runstate(rs, var=None, atm=None, para=None)` — RunState entry point. Materialises legacy scratch via `legacy_view(rs)`, builds statics, runs the JIT'd runner (single-shot or chunked), unpacks back into a fresh `RunState`, and returns it.
  - `_call_legacy(var, atm, para, make_atm)` — legacy entry point that mutates `(var, atm, para)` in place.
  - `_build_statics(var, atm)` — assemble `_Statics` (global atm static + photo / refresh / conden sub-statics).
  - `_ensure_runner(var, atm)` — first-call setup: build statics, build the runner via `_make_runner`, cache both.
  - `_build_photo_static(var, atm)` — host-side construction of the `_PhotoStatic` for the photo sub-graph (cross sections, scattering tables, branch index maps).
  - `_build_refresh_static(var, atm)` — host-side construction of the `AtmRefreshStatic`.
  - `_build_conden_static(var, atm, gas_mask_jnp)` — host-side construction of the `CondenStatic` (gas-to-condensate mapping, masses, sat-mix tables, conden reaction indices).
  - `_pack_state_from_runstate(rs)` / `_pack_state(var, para, atm)` — build the initial `JaxIntegState` from either entry path.
  - `_unpack_state_to_runstate(state, rs_entry)` / `_unpack_state(state, var, para, atm)` — write the final `JaxIntegState` back to the RunState / legacy containers.
  - `_unpack_J_sp` / `_unpack_k` / `_unpack_ring` / `_unpack_conden_k` — sub-unpackers used by `_unpack_state`.
  - `_atom_dict_to_arr(d)` / `_initial_photo_carry_from_runstate(rs)` / `_initial_atm_carry_from_runstate(rs)` / `_initial_conv_carry_from_runstate(rs)` — RunState → initial-carry conversions.
  - `_f_dy(var, para)` — host-side diagnostic for `dy / dydt`, used by the end-of-run print.
  - `_run_chunked(init_state, atm_static, var, para, atm)` — chunked execution path (cfg `use_chunked_runner=True`, any live-UI flag on, or `wall_clock_max` set). Calls the runner in `live_plot_frq`-sized chunks and dispatches `print_prog` / `live_ui.LiveUI` / wall-clock checks between chunks.
  - `prepare_runstate(rs)` — build one profile's `(init_state, atm_static)` for the batched path and ensure the runner closure exists; guards that photo lanes share the first profile's star/wavelength grid (`nbin`/`din12_indx`/`bins`/`sflux_top`) — only the T-P profile may differ.
  - `run_batch(states_batched, atm_static_batched)` — one vmapped device call integrating every lane to termination with freeze-on-done; per-lane results identical to solo runs. Supports photochemistry and NH3/H2O relaxation condensation (per-profile values ride `ProfileVars`).
  - `_profile_vars_from_runstate(rs)` — snapshot this profile's per-profile constants into a `ProfileVars` (falls back to the closure-baked photo static on the legacy entry path, which has no `rs.photo_static`).
  - `_summary_shim(rs)` — small post-run RunState wrapper used by tests.

### `op_jax.py`
Standalone photochemistry adapter. Holds lazy `PhotoData` / `PhotoJData`
caches built off `PhotoStaticInputs`, and dispatches the actual kernel
calls to `photo.py`. **[Δ master]** Master's `op.Op` mutates `var.tau /
var.aflux / var.J_sp / var.k_arr` in place. This adapter does the same
but routes through the JIT'd JAX kernels and the cached
`PhotoStaticInputs` pytree.
- `Ros2JAX` — adapter class.
  - `__init__(photo_static=None)` — optionally accept a pre-built `PhotoStaticInputs`; otherwise build lazily on first call.
  - `_ensure_photo_static(var, atm)` — build / refresh the cached `PhotoStaticInputs`.
  - `compute_tau(var, atm)` — optical depth via `photo.compute_tau_jax`; writes to `var.tau`.
  - `compute_flux(var, atm)` — two-stream Eddington RT via `photo.compute_flux_jax`; writes to `var.aflux` / `var.sflux` / `var.dflux_u` / `var.dflux_d`.
  - `compute_J(var, atm)` — photodissociation rates via `photo.compute_J_jax` + `photo.update_k_with_J`; writes to `var.J_sp` and the photo rows of `var.k_arr`.
  - `compute_Jion(var, atm)` — photoionisation rates via `photo.compute_Jion_jax`.
  - `naming_solver(para)` — print the transport / BC summary; stamp `para.solver_str`.

### `photo.py`
JAX photochemistry kernels.
**[Δ master]** Master's `op.compute_J` reads cross-section dicts per-species
inside a Python loop. The JAX path packs the same data into two
`PhotoJData` pytrees (one for J, one for Jion) plus a single dense
`PhotoData` pytree for optical depth / scattering, and runs the
interpolation/integration as one fused trapezoidal kernel over the
two-resolution wavelength grid.
- `PhotoData` — pre-stacked optical-depth + scattering pytree (`absp_idx`, `absp_cross`, `absp_T_idx`, `absp_T_cross`, `scat_idx`, `scat_cross`).
- `PhotoJData` — pre-stacked branch-resolved cross-section pytree for J / Jion (`cross_J`, `cross_J_T`, `din12_indx`, `dbin1`, `dbin2`, `branch_keys`, `branch_T_keys`).
- `photo_data_from_static(static, species_list) → PhotoData` — build the per-run `PhotoData` from `PhotoStaticInputs` and a species list.
- `photo_J_data_from_static(static) → PhotoJData` — build the J-rate pytree.
- `photo_ion_data_from_static(static) → PhotoJData` — build the Jion-rate pytree.
- `compute_tau_jax(y, dz, photo) → tau` — top-down cumulative optical depth.
- `compute_flux_jax(...)` — two-stream RT (forward + back fluxes, scattering, ground albedo, `sl_angle`).
- `_compute_J_inner(aflux, cross_J, cross_J_T, din12_indx, dbin1, dbin2)` — shared trapezoidal integrator over the two-resolution grid; returns `(J_br, J_br_T)` arrays.
- `compute_J_jax(aflux, photo_J)` — branch-resolved photodissociation rates as a dict keyed by `(species, branch)` → `(nz,)` array.
- `compute_Jion_jax(aflux, photo_ion)` — branch-resolved photoionisation rates; same dict shape (delegates to `compute_J_jax`).
- `compute_J_jax_flat(aflux, cross_J, cross_J_T, din12_indx, dbin1, dbin2)` — flat-output variant returning `(J_br, J_br_T)` directly; used by the outer-loop integration and vmap-consistency tests.
- `compute_Jion_jax_flat(aflux, cross_J, din12_indx, dbin1, dbin2)` — flat-output photoionisation variant.
- `_pack_branch_to_k_index_map(branch_keys, rate_index, remove_list)` — branch-to-reaction-index map.
- `pack_J_to_k_index_map(photo_J, var, vulcan_cfg)` / `pack_Jion_to_k_index_map(photo_ion, var, vulcan_cfg)` — reaction-index lookup helpers.
- `update_k_with_J(k_arr, J_br, J_br_T, branch_re_idx, branch_active, branch_T_re_idx, branch_T_active, f_diurnal)` — write per-branch J-rates into `k_arr` via a single fused scatter (`.at[].set()`).

### `photo_setup.py`
Host-side cross-section preprocessing. Builds the wavelength bin grid
and interpolates per-species cross sections + branch ratios onto that
grid. **[Δ master]** Master's `op.make_bins_read_cross` mutates `var.cross
/ var.scat_dx / var.dx_J / var.bins` etc. The JAX version returns a
dense `PhotoStaticInputs` pytree which is the differentiable runtime
surface; the legacy mutations are still performed by
`populate_photo_arrays` for the hybrid oracle tests. The CH3SH branch
CSV has a non-monotonic `354.0` typo that would require a sort step
in any `jnp.interp` port — preserved as-is.
- `_cross_folder()` — return `cfg.cross_folder` as a string.
- `_load_thresholds(species_in_network)` — read per-species photodissociation thresholds.
- `_load_cross_csv(sp, use_ion)` — read `{sp}_cross.csv` (3- or 4-column variants).
- `_load_branch_csv(sp)` — read `{sp}_branch.csv` (auto-detected columns).
- `_load_ion_branch_csv(sp)` — read `{sp}_ion_branch.csv`.
- `_discover_T_cross_files(sp)` — list T values of `{sp}_cross_{T}K.csv`.
- `_load_T_cross_csv(sp, T, use_ion)` — read T-dependent cross section.
- `_load_rayleigh_csv(sp)` — read Rayleigh scattering data.
- `_make_bins(...)` — two-resolution wavelength bin grid (`dbin1` < `dbin_12trans` < `dbin2`).
- `_sort_pairs`, `_interp_zero_extrap`, `_interp_edge_extrap`, `_interp_T_log_pair` — small interpolation helpers used by the rebinning step.
- `_bin_cross_and_branches(...)`, `_bin_T_dependent(...)` — rebin per-species data onto the photo grid.
- `populate_photo_arrays(var, atm)` — write photo arrays back into the legacy `var` / `atm` containers (hybrid oracle path).
- `_build_photo_static_dense(var, atm)` — build a fresh `PhotoStaticInputs` pytree.
- `build_photo_static(cfg, atm, var)` — public builder used by tests and external callers.
- `_alloc_runtime_buffers(var, nbin, nz)` — zero-allocate the host-side mutable buffers (`sflux`, `dflux_u/d`, `aflux`, `tau`, `sflux_top`) on `var`.
- `populate_photo(var, atm)` — top-level builder: build the `PhotoStaticInputs`, write scalar metadata + threshold table to `var`, allocate runtime buffers. Returns the `PhotoStaticInputs`.

### `conden.py`
Pure-JAX condensation kernels.
**[Δ master]** Master's `op.condense` operates inside a Python loop with
explicit per-species branches. The JAX version packs all condensate
species into a single `CondenStatic` and dispatches per-condensate
relaxation kernels (`apply_h2o_relax_jax`, `apply_nh3_relax_jax`) so the
hot path stays JIT-compatible. Only `H2O` and `NH3` have implicit-Euler
relaxation; the other condensates (`H2SO4`, `S2`/`S4`/`S8`, `C`) go
through the reaction-rate path.
- `SUPPORTED_CONDEN_KINETICS = (H2O, NH3, H2SO4, S2, S4, S8, C)` — the condensates with a fully-ported runtime kinetics path (exactly master's `op.conden` branch set); `atm_setup._SUPPORTED_CONDENSABLES` adds sat-data-only H2S. `runtime_validation` checks `condense_sp` against these upfront.
- `CondenStatic` — NamedTuple of static condensation inputs (gas-to-condensate mapping, masses, sat tables, conden re indices, relax-species indices and saturation arrays, gas-only mask).
- `update_conden_rates(k_arr, y, st)` — recompute condensation/evaporation rate constants and overwrite the conden rows of `k_arr` (`k_pos` to `re`, `k_neg` to `re+1`).
- `apply_h2o_relax_jax(y, ymix, dt, st) → (y_new, ymix_new)` — implicit-Euler `H2O` cold-trap relaxation. Mass moves into / out of `H2O_l_s`. No-op when `h2o_active=False`.
- `apply_nh3_relax_jax(y, ymix, dt, st) → (y_new, ymix_new)` — analogous `NH3` relaxation, clamping condensation to layers at or below `nh3_conden_top = argmin(sat_mix['NH3'])` (a Python int when closure-baked, a per-lane 0-d int32 when spliced from `ProfileVars` in the batched runner — the kernel only compares it against `jnp.arange`).

### `integrate.py`
Fixed-`dt` JAX integration loop used for validation and benchmarks.
Assumes frozen rate constants (no photo, no condensation, no fix-species).
For production, use `outer_loop.OuterLoop`.
- `jax_integrate_fixed_dt(y0, k_arr, dt, n_steps, atm, net)` — take `n_steps` fixed-dt Ros2 steps via `lax.scan`; `n_steps` is a static argument. Returns `(y_final, deltas)`.

---

## Differentiability

### `steady_state_grad.py`
Implicit-function-theorem gradients of the converged photochemical
state. Uses `jax.custom_vjp` for O(1)-memory reverse-mode AD.
**[Δ master]** Master has no AD path. JAX's `lax.while_loop` blocks
`vjp` directly, so reverse-mode through the integration goes through
the implicit-function theorem: solve `(∂f/∂y) z = ∂L/∂y*` once at
the converged state. O(1) memory in step count — no trajectory
checkpointing.
- `SteadyStateInputs` — NamedTuple of differentiable inputs (`k_arr`, plus the atm fields the diffusion solve consumes).
- `build_steady_state_inputs(k_arr, atm)` — pack `k_arr` + an `AtmStatic` into a `SteadyStateInputs`.
- `_atm_from_inputs(inputs)` — repack a `SteadyStateInputs` back into an `AtmStatic` for the residual.
- `steady_state_residual_inputs(y, inputs, net, grav)` — `f(y, inputs) = chem_rhs + diffusion` on the structured input.
- `steady_state_residual(y, k_arr, atm, net, grav)` — convenience wrapper for callers with a raw `AtmStatic`.
- `_build_jacobian_blocks(y, k_arr, atm, net)` — per-layer dense diagonal block + diagonal off-diagonals.
- `validate_steady_state_solution(y_star, inputs, net, residual_rtol=1e-6, residual_atol=0.0)` — sanity-check residual norm against the bound.
- `differentiable_steady_state_inputs(inputs, y_star, net)` — `custom_vjp` returning `y_star`; forward is the identity, backward solves the implicit system. Primary public API.
- `checked_differentiable_steady_state(...)` — same with `validate_steady_state_solution` chained in.
- `_ssi_fwd(inputs, y_star, net)` / `_ssi_bwd(net, res, v)` — `custom_vjp` hooks for the structured-input API.
- `steady_state_value_and_grad(loss_fn, inputs, y_star, net, residual_rtol=1e-6, residual_atol=0.0)` — full value-and-gradient routine; preferred entrypoint when differentiating against the full structured pytree.
- `differentiable_steady_state(k_arr, y_star, atm, net)` — backwards-compatible wrapper that only differentiates against `k_arr`.
- `_ss_fwd(k_arr, y_star, atm, net)` / `_ss_bwd(atm, net, res, v)` — `custom_vjp` hooks for the legacy `k_arr`-only API.

---

## Validation, I/O, and host-side glue

### `runtime_validation.py`
Pre-run configuration validation.
- `_validate_fastchem_input_vs_network(cfg, root)` — pin the FastChem element file's values and order against the network's atoms.
- `_validate_network_assets(cfg, root)` — check every species / photo / atom file referenced by `cfg` exists.
- `_validate_numerical_bounds(cfg)` — bound-check tuning knobs (rtol/loss/photo-switch/Newton-tol/step-size) so typos fail early.
- `validate_runtime_config(cfg, root=None)` — top-level entry, called from `vulcan_jax_cli.py` and the `OuterLoop` entry points. Also rejects upfront: non-Ros2 solvers, inconsistent flag combos, `const_mix` keys that are not network species (master crashes identically, e.g. its Earth example's 'Ar'), and `condense_sp` entries outside the supported condensate tiers (kinetics set vs sat-only H2S).

### `legacy_io.py`
Vendored host-side glue from VULCAN-master's `op.py` — the rate-metadata
parser, the `.vul` writer, end-of-run plotters, and the print-progress
helpers. **[Δ master]** `Output.save_out` is polymorphic — it accepts
either a typed `RunState` or the legacy `(var, atm, para)` triple, and
synthesises the photo / ion / parameter dicts from the typed state at
pickle time rather than incrementally during the run. JAX arrays are
cast to NumPy via `np.asarray()` before pickling so VULCAN's `plot_py/`
scripts load the output unmodified.

- `_master_tableau20()` — return VULCAN-master's normalised Tableau-20 plotting palette.
- `_import_plt()` — lazy matplotlib import with a headless-safe backend.
- `_synthesize_cross_dicts(static)` — synthesise `var.cross` / `var.scat_dx` / `var.dx_J` from a `PhotoStaticInputs` (needed because `ReadRate.make_bins_read_cross` is not vendored; see `photo_setup` instead).
- `_integrate_J_branch(...)` — trapezoidal integrator used by `_synthesize_J_sp_dict`.
- `_synthesize_J_sp_dict(...)` — synthesise `var.J_sp` / `var.Jion_sp` per-species dicts at write time from the RunState.
- `_is_runstate_arg(obj)` — runtime check for the polymorphic `save_out` dispatch.
- `_synthesize_save_dicts(runstate, cfg, photo_static=None)` — synthesise master-shaped `variable` / `atm` / `parameter` dicts from a `RunState`.

Classes:
- `ReadRate` — host-side rate-metadata parser (vendored from master).
  - `__init__()` — set parser scratch.
  - `read_rate(var, atm)` — populate `var.Rf`, `var.pho_rate_index`, `var.n_branch`, `var.photo_sp`, `var.ion_sp`, `var.ion_rate_index`, `var.ion_br_ratio`, `var.charge_list`. Rate **values** flow through `rates.build_rate_array` (this method is metadata only — never on any AD path).
- `Output` — `.vul` writer + plotters.
  - `__init__()` — set plotting + counters.
  - `save_cfg(dname)` — copy the active `vulcan_cfg.py` into `output/` for provenance.
  - `print_prog(var, para)` — periodic progress print called from the chunked-runner host loop.
  - `print_end_msg(var, para)` — end-of-run summary for the converged case (`end_case=1`).
  - `print_unconverged_msg(var, para, case)` — end-of-run summary for unconverged exits (`end_case in (2, 3, 4)`).
  - `save_out(*args, **kwargs)` — polymorphic `.vul` writer. Dispatches via `_is_runstate_arg` to `_save_out_from_runstate(rs, dname, photo_static=None, ...)` or `_save_out_legacy(var, atm, para, dname, ...)`.
  - `_save_out_from_runstate(runstate, dname, photo_static=None, ...)` — RunState backend: synthesises dicts via `_synthesize_save_dicts`, casts to NumPy, pickles to `output/<out_name>.vul`.
  - `_save_out_legacy(var, atm, para, dname, photo_static=None, ...)` — legacy backend that operates on mutable containers.
  - `plot_end(var, atm, para)` — final mixing-ratio plot.
  - `plot_evo(var, atm, plot_j=-1, plot_ymin=1e-20, dn=1)` — evolution plot from the saved ring buffer.
  - `plot_TP(atm)` — temperature/pressure profile QC plot.

### `live_ui.py`
Host-side live-UI dispatcher. Fires between JIT'd step chunks when any
of `use_live_plot` / `use_live_flux` / `use_save_movie` / `use_flux_movie`
is True.
**[Δ master]** Master's live UI fires inside the Python step loop;
JAX's runner is JIT'd, so any live-UI flag forces the chunked-runner
path (chunks of `live_plot_frq` accepted steps). matplotlib / PIL stay
on the host and never enter a JIT'd region.
- `any_live_flag_on(cfg)` — True if any of the four live-UI flags is set.
- `LiveUI` — class.
  - `__init__()` — cache scratch + species-index map.
  - `_ensure_mpl()` — lazy matplotlib import with a headless-safe backend.
  - `_ensure_species_index()` — cache and return `species -> column_index`.
  - `dispatch(var, atm, para)` — route to the mixing-ratio / flux updaters per cfg.
  - `update_mix(var, atm, para, save_movie, show)` — render the mixing-ratio panel (movie frames optional).
  - `update_flux(var, atm, para, save_movie, show)` — render the diffusive-flux panel.

---

## Subdirectories

### `cfg_examples/`
Reference configs. Copy one to `vulcan_cfg.py` at the repository root and
run `vulcan-jax`, or import via `from cfg_examples.vulcan_cfg_HD189
import *` from a thin wrapper.

- `vulcan_cfg_HD189.py` — HD 189733b reference. Matches `VULCAN-master/cfg_examples/vulcan_cfg_HD189.py` for cross-version smoke tests. **Canonical parity target.**
- `vulcan_cfg_HD209.py` — HD 209458b (no S species, weaker gravity).
- `vulcan_cfg_Earth.py` — Earth troposphere/stratosphere with condensation.
- `vulcan_cfg_W39b.py` — WASP-39b paper-match config (Wogan et al.).
- `README.txt` — short description + which cfg is the matched one.

### `tests/`
Curated suite focused on hot-path kernels, oracle agreement, and JAX
transform consistency. Run with
`python -m pytest tests -q --tb=short -ra` from the repo root.

- `conftest.py` — session-scoped HD189 pre-loop fixture, cfg snapshot/restore autofixtures, sibling-master path cleanup.
- `data/oracle_baselines/{earth,hd209}_20step.npz` — oracle reference snapshots.
- `data/photo_setup_hd189_{baseline,T_dep}.npz` — photo-setup test fixtures.
- `diffusion_numpy_ref.py` — NumPy oracle for diffusion kernels (used by `test_diffusion*.py`).
- `test_chem.py`, `test_chem_jac_sparse.py`, `test_chem_rhs_codegen.py` — chemistry RHS / Jacobian agreement.
- `test_block_thomas.py`, `test_block_thomas_diag.py` — block-tridiagonal solvers.
- `test_diffusion.py`, `test_diffusion_variants.py` — diffusion operator + Jacobian assembly.
- `test_ros2_step.py` — single-step Rosenbrock kernel.
- `test_conden_jax.py` — condensation kernels, incl. the traced-scalar / per-lane `nh3_conden_top` boundary (bitwise).
- `test_photo.py`, `test_photo_ion.py`, `test_photo_setup.py` — photo kernels and cross-section preprocessing.
- `test_gibbs.py`, `test_rates.py`, `test_read_rate.py`, `test_network_parse.py` — setup parsers.
- `test_ini_abun.py` — all five `ini_mix` modes.
- `test_atm_setup_matrix.py` — atm-variant branches HD189 doesn't exercise.
- `test_state_roundtrip.py` — `RunState` ↔ pytree ↔ legacy `(var, atm, para)`.
- `test_save_evolution.py` — `save_evolution=True` cadence + ring-buffer round-trip.
- `test_oracle.py` — Earth + HD209 20-step oracle vs VULCAN-master (skips cleanly if absent).
- `test_default_master_parity.py` — canonical HD189 root-default audit, bit-exact pre-loop state, and 20-step oracle vs staged VULCAN-master.
- `test_outer_loop_smoke.py` — HD189 50-step smoke (the headline regression test).
- `test_outer_loop_atm_refresh.py`, `test_outer_loop_conden_gate.py`, `test_outer_loop_conv.py`, `test_outer_loop_ion.py`, `test_outer_loop_photo.py` — outer-loop sub-graph tests.
- `test_w39b_fastchem_invariant.py` — frozen FastChem snapshot for W39b.
- `test_use_fix_H2He.py`, `test_solver_fix_all_bot.py` — boundary-condition variants.
- `test_vmap_kernels.py`, `test_vmap_step.py` — JAX vmap consistency.
- `test_vmap_while_loop.py` — `run_batch` full-integration batching: homogeneous equivalence, freeze-on-done, NaN isolation, genuinely-different profiles vs solo runs.
- `test_vmap_photo_batch.py` — batched photochemistry: per-lane T-dependent cross sections, solo-vs-batch agreement, same-star guard.
- `test_nh3_conden_batch_subprocess.py` — end-to-end batched NH3 condensation on the lowT-Jupiter network (subprocess via `$VULCAN_JAX_NETWORK`; the suite's slowest test, ~10 min cold).
- `test_condensation_runtime_subprocess.py` — end-to-end H2O relaxation + settling on a condensate network (subprocess).
- `test_legacy_photo_tcross.py` — legacy `(var, atm, para)` entry with non-empty `T_cross_sp` (regression for the ProfileVars photo fallback).
- `test_validation_const_mix_conden.py` — upfront validators: non-network `const_mix` keys, unsupported `condense_sp`, sat-only H2S acceptance, tier-drift guard.
- `test_make_config_wiring.py` — `make_config()` override propagation + import-lock fail-fasts.
- `test_host_setup_hooks.py` — `$VULCAN_JAX_FASTCHEM_DIR`, rate-parse cache, `skip_chem_warmup`.
- `test_atm_refresh_gravity.py` — self-consistent gravity invariant in the hydrostatic refresh.
- `test_atom_conservation_projection.py`, `test_species_mass_integrity.py` — conservation projection + composition-table integrity.
- `test_fastchem_element_order.py` — FastChem abundance-file element-order regression guard.
- `test_diffusion_production_kernel.py`, `test_moldiff_disabled.py` — production diffusion kernel + moldiff-off variant.
- `test_cli_smoke.py` — `vulcan-jax` CLI end-to-end smoke.
- `test_steady_state_grad.py` — implicit-AD reverse-mode gradients.
- `test_cfg_examples.py` — each kept config loads + runs pre-loop setup.
- `test_config_matrix.py` — config-flag combination coverage.

### `benchmarks/`
- `bench_step.py` — per-step JAX timing + comparison to VULCAN-master if a sibling checkout is present.

### `examples/`
- `batched_run.py` — `jax.vmap` over the per-step kernel for batched atmospheres.
- `gpu_benchmark.py` — standalone GPU throughput benchmark driving `run_batch` to convergence over HD189-like planet batches (parallel host setup, chunked progress, `--device-batch` host-side tiling). Kept byte-identical with `vulcan-emulator/supercomputer_cmds/gpu_benchmark.py`.
- `grad_jvp_example.py` — forward-mode AD through the per-step kernel.
- `grad_implicit_example.py` — reverse-mode AD through the converged steady state via `steady_state_grad`.

### `tools/`
End-user utility scripts (data prep, debug, parity checks).
- `make_mix_table.py` — build a mixing-ratio table for `ini_mix='table'`.
- `make_spectra_in_nm.py` — convert stellar spectra to nm wavelength bins.
- `print_actinic_flux.py` — print actinic flux from a `.vul` file.
- `audit_master_parity.py` — verify the root default HD189 config / input parity against a VULCAN-master checkout. Returns nonzero on the first mismatch; used by CI / the default-parity test.

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
Concurrent invocations are serialised via `fcntl.flock` inside
`ini_abun._run_fastchem` so `pytest -n auto` is safe.

### `output/`, `plot/`
Created at run time by the driver and live-UI. Both are safe to delete
and will reappear on next run; pre-existing contents are run artefacts.
