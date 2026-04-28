# VULCAN-JAX — Implementation Status

A JAX-accelerated, drop-in compatible port of [VULCAN](https://github.com/exoclime/VULCAN), the chemical-kinetics code for exoplanet atmospheres.

**Location**: `/Users/imalsky/Desktop/Emulators/VULCAN_Project/VULCAN-JAX/` (next to `VULCAN-master/`).

**Last updated**: 2026-04-27.

---

## TL;DR

Phases 0–12, A, **13–16 (stdout parity, `use_fix_H2He`, `save_evolution`, chunked runner + live plotting / movie / per-step progress)**, **17 (chem_rhs cancellation-floor investigation)**, **18 (per-test fixture sharing scaffolding)**, **19 (typed pre-loop pytree foundation + live-output drop)**, **20 (`Atm.*` rewrite into JAX-native `atm_setup.py`)**, **21 (`InitialAbun.*` rewrite into JAX-native `ini_abun.py`; `build_atm.py` deleted)**, **22a (rate-parser consolidation in `rates.py`)**, **22b (photo cross-section preprocessing rewrite in `photo_setup.py`)**, **22c (legacy ReadRate rate-method retirement)**, and **22d (`var.k` dict-surface retirement)** complete. VULCAN-JAX no longer depends on `../VULCAN-master/` at runtime, matches it to **1.59e-10 max relative error** on a 50-step HD189 kernel comparison, exposes implicit-function-theorem gradients of the converged state through the structured-input API in `steady_state_grad.py`, exposes a typed pre-loop input schema (`AtmInputs`/`RateInputs`/`PhotoInputs`/`IniAbunOutputs` in `state.py`) that Phases 20–21 now fill from JAX-native setup, and the entire pre-loop atmosphere + initial-abundance + rate-parser + photo-preprocessing surface is now pure NumPy/JAX with `Atm` / `InitialAbun` kept as thin facades for legacy callers. Production rate inputs flow through the dense `var.k_arr` (Phase 22a–d); the `var.k` dict surface is gone, synthesized only at `.vul` write time. The current test surface includes runtime, ion-photo, example-config, chunked-runner, save_evolution, `use_fix_H2He`, optional backend-parity, typed-pytree round-trip, dropped-live-output validator, the Phase 20 `atm_type × atm_base × Kzz_prof × BC` matrix, the Phase 21 `ini_mix ∈ {EQ, const_mix, vulcan_ini, table, const_lowT}` matrix, the Phase 22a rate-parser oracle, and the Phase 22b photo-cross-section bit-exact match; with `VULCAN-master` present on this CPU-only host the current run is **94 pass + 1 skip** (`test_backend_parity` skips cleanly without a GPU), and the upstream-oracle tests still skip cleanly when `../VULCAN-master/` is absent.

**Performance**: the repaired benchmark/profile scripts currently report `master_ros2_step_ms=125.993`, `jax_ros2_step_ms=22.618`, and `outer_loop_50step_ms_per_step=39.381` on the HD189 default case. The main wins remain the analytical chemistry Jacobian, diagonal-aware block-tridiagonal factorization reuse across both Rosenbrock stages, and pre-baked y-independent diffusion terms.

**Differentiability** (Phase 12): every per-step kernel (`chem_rhs`, `chem_jac_analytical`, `block_thomas_diag_offdiag`, `_build_diff_coeffs_jax`, photo kernels, …) is `jit` / `vmap` / `jvp` / `vjp` compatible. `outer_loop.runner`'s `lax.while_loop` blocks `vjp` (forward-mode `jvp` works), so end-to-end reverse-mode AD goes through `steady_state_grad.py`'s implicit-function-theorem API. The differentiable surface now accepts a typed pytree of runtime inputs (rate constants, transport/diffusion fields, BC fluxes, and photo inputs) and enforces a residual-based convergence check before attaching implicit AD.

The per-step path is fully JAX, the inner accept/reject loop is JIT'd, and every conditional update — photo, atmosphere geometry refresh, hydrostatic balance, condensation, ion charge balance, fix-all-bot, adaptive rtol, photo-frequency switch, ring-buffered convergence check — fires inside the same JIT'd runner. The Python `while not stop()` loop is gone; the runner runs to convergence / `count_max` / `runtime` in one device call:
- `outer_loop.OuterLoop` (Phase 10.1) replaces `op.Ros2.one_step`'s Python `while True` retry loop with a `jax.lax.while_loop` body that does `jax_ros2_step → clip → step_ok / step_reject → step_size` entirely in JAX.
- Phase 10.2: `compute_tau` / `compute_flux` / `compute_J` and the `var.k` rewrite they drive run inside the JAX runner via a `jax.lax.cond`-gated photo branch. Photo state (`tau`, `aflux`, `dflux_u`, `k_arr`, ...) lives in the `JaxIntegState` carry and stays on device between calls; the previous Python-dict iteration over `var.J_sp` is replaced by a `.at[].set()` in `photo.update_k_with_J`.
- Phase 10.3: `update_mu_dz` (mu / g / Hp / dz / dzi / Hpi / zco) and `update_phi_esc` (diffusion-limited escape flux) run inside the JAX runner via `atm_refresh.update_mu_dz_jax` / `update_phi_esc_jax`, fired by a second `lax.cond` on `do_atm_refresh`. Hydrostatic balance (`y = n_0 * ymix`, `op.py:908-914`) moved inside the body and runs every accepted step. The hydrostatic loop's true sequential dependency (`zco[i+1] = zco[i] + dz[i]`) is split into forward / backward `jax.lax.scan`s around the static `pref_indx`. Geometry fields are spliced into the closed-over `AtmStatic` per body iteration so chemistry always sees the freshest diffusion coefficients.
- Phase 10.4: `op.conden` (condensation rate update) plus the optional `op.h2o_conden_evap_relax` / `op.nh3_conden_evap_relax` cold-trap relaxers run inside the JAX runner via `conden.update_conden_rates` / `apply_h2o_relax_jax` / `apply_nh3_relax_jax`, gated on `do_accept & t >= start_conden_time` per body iteration. For HD189 (`use_condense=False`) the conden branch is None and the path is bit-identical to 10.3.
- Phase 10.5: the entire `op.Integration.__call__` Python while-loop moves into the JAX runner. `cond_fn` carries the full `op.stop` + `op.conv` termination criterion (count_max / runtime / longdy/longdydt threshold + flux_cri gate), with the y_time / t_time history living in the carry as a `(conv_step, nz, ni)` ring buffer. Adaptive rtol (`op.py:836-852`) and the photo-frequency ini→final switch (`op.py:819-823`) move into the body — `s.rtol`, `s.loss_criteria`, `s.update_photo_frq`, `s.is_final_photo_frq`, `s.longdy`, `s.longdydt` are all carry fields. The previous per-batch host sync goes away; the runner is one-shot for the entire integration.
- Phase 10.6: ion charge balance (`e[:] = -dot(y, charge_arr)`, mirroring `op.py:3001-3007`) and `use_fix_all_bot` (post-step bottom-row clamp to chemical-EQ mixing ratios captured at OuterLoop init, mirroring `op.py:3050-3051`) move into the body, gated by Python bools at trace time. The `op.Integration` parent class is dropped — `OuterLoop` is now standalone with an inline `_f_dy` diagnostic. The legacy `jax_integrate_adaptive` (subsumed by `OuterLoop`) is deleted from `integrate.py`.
- `Ros2JAX` no longer subclasses `op.Ros2`; it's a standalone photo adapter with `compute_tau` / `compute_flux` / `compute_J` retained for the pre-loop one-shot call in `vulcan_jax.py`. `vulcan_jax.py` raises `NotImplementedError` for non-Ros2 solver names.

What remains coupled to upstream (NumPy) for the runtime path:
- **None** (post-Phase A). `vulcan_jax.py` imports `legacy_io` (vendored copy of `op.ReadRate` + `op.Output`) for one-shot pre-loop setup; the per-step path is fully JAX. `vulcan_jax.py` and `op_jax.py` no longer `sys.path.append` `../VULCAN-master/`.
- Live-plot, live-flux, save-movie, and flux-movie paths were removed in
  Phase 19. `runtime_validation.py` rejects those flags. Post-run plotters
  (`plot_end`, `plot_evo`, `plot_TP`) survive as PNG writers. The chunked
  host-callback driver still exists for `print_prog` and the surviving
  post-run plotters; live / movie side paths are gone, not in flux.
- Low-T rate caps fire only when their cfg flag is on (HD189 has it off).

---

## Phase 23 — PARTIAL

Phase 23 opened on 2026-04-27, immediately after Phase 22e closed. It was scoped as three waves: Wave 1 (parallel additive vendoring), Wave 2 (A4 `RunState` refactor), Wave 3 (C2 config-matrix tests + C1 oracle campaigns).

**Closed (Wave 1 + Wave 3):**
- **B1 — diagnostics**: `diagnose.py` (66 lines, port of master's diagnose with the dead `import op` dropped) and `make_chem_funs.py` (13-line no-op shim) both clean (ruff + vulture).
- **B2 — `plot/` surface**: 17 plot scripts + `README.md` vendored verbatim from master's `plot_py/`. All scripts carry a top-of-file `# ruff: noqa` for upstream-style code. The JAX `.vul` schema (Phase 22d/e dict synthesizers) matches every key the scripts read; `plot_vulcan.py` smoke-tests cleanly. Two scripts (`plot_TP_Kzz.py`, `plot_evolution.py`) have hardcoded non-HD189 file paths; one needs `t_time/y_time` which only land when `save_evolution=True`. None of those are blockers — they only fail if pointed at a `.vul` they cannot consume.
- **B3 — `tools/`**: `make_mix_table.py`, `make_spectra_in_nm.py`, `print_actinic_flux.py` vendored. `print_actinic_flux.py` and `make_mix_table.py` smoke-tested cleanly against the JAX HD189 `.vul`.
- **B4 — `atm/` and `thermo/` helpers**: 8 helper / docs files vendored byte-exact (`atm/README.txt`, `atm/convert_vpl_spectra_in_nm.py`, `atm/make_spectra_in_nm.py`, `atm/stellar_flux/{README.txt, plot_spectra.py, read_muscles_spectra_in_nm.py}`, `thermo/{README.md, make_compose.py}`). Standalone post-processing utilities; no runtime dependency.
- **C2 — config-matrix tests**: `tests/test_config_matrix.py` (8 parametrized cases, 7.88s wall) covering `use_lowT_limit_rates`, T-dep cross sections (Earth-style `T_cross_sp=['CO2','H2O','NH3']`), `use_vm_mol`, `use_settling`, `use_topflux`, `use_botflux`, `fix_species` non-empty, and `use_fix_all_bot`.
- **C1 — Earth / Jupiter / HD209 oracle campaigns**: `tests/test_oracle_earth.py`, `tests/test_oracle_jupiter.py`, `tests/test_oracle_hd209.py` plus `tests/data/oracle_baselines/{earth,jupiter,hd209}_20step.npz`. All three pass in 61s wall total. Skip cleanly when `../VULCAN-master/` is absent.

**Deferred to Phase 24:**
- **A4 — `RunState` handoff in `OuterLoop`**: the parallel agent stream-timed out without producing the cross-file refactor. The plan is fully spec'd in `/Users/imalsky/.claude/plans/i-want-to-continue-bubbly-firefly.md` (Wave 2 section); pick up there. `pytree_from_store` / `apply_pytree_to_store` are already in `state.py`; the deferred work is extending `RunState` with convergence-loop + Parameters fields and threading it through `OuterLoop.__call__` and `legacy_io.save_out`. Test surface is well understood (~12 tests touch the entry/exit boundary; only ~9 mutate `vulcan_cfg`). Do NOT delete `store.py` in Phase 24 either — that's a Phase 25 cleanup.

**Deferred indefinitely:**
- **C3 — real-GPU CPU↔GPU parity numbers**: the host is CPU-only. The optional `test_backend_parity` continues to skip cleanly without a GPU.

**Pre-existing failures unrelated to Phase 23:**
- `tests/test_photo_setup.py::test_photo_setup_matches_*_fixture` — 2 tests fail with 1-ULP drift on 9 of 2588 wavelength bins (max abs 1.42e-14, max rel 2.2e-16). Caused by uncommitted Phase 19 edits to `vulcan_cfg.py` that drifted the bins computation; the npz fixture is stale relative to the current cfg defaults. The fixture regenerator script (`tests/_gen_photo_baseline.py`) explicitly says "never run again unless the maintainer asks." Out of scope for Phase 23; flagged for the maintainer.

**Suite count after Phase 23:** 92 (Phase 22e) + 8 config-matrix + 3 oracle = **103 pass + 1 skip + 2 fail** (the 2 fails are the pre-existing photo_setup fixture drift).

---

## ✅ DONE

### Core JAX modules

| File | Purpose | Lines | Validated to |
|---|---|---|---|
| `network.py` | Parse VULCAN network text → typed Network with stoichiometry tables | 498 | exact (matches `chem_funs.spec_list`/`ni`/`nr`) |
| `rates.py` | Forward rate coefficients (Arrhenius, Lindemann, Troe special) | 152 | **0** (machine precision) |
| `gibbs.py` | NASA-9 polynomial Gibbs energy, K_eq, reverse rates | 287 | **1.4e-14** vs `chem_funs.Gibbs` |
| `chem.py` | Pure-JAX chemistry RHS + autodiff Jacobian | 191 | RHS 1e-4 (FP cancellation), Jacobian **4.3e-13** |
| `solver.py` | Block-tridiagonal Thomas solver, Ros2 helpers | 191 | **3e-15** (machine precision) vs np.linalg.solve |
| `op_jax.py` | `Ros2JAX` — standalone photo adapter (compute_tau/flux/J), used for the pre-loop one-shot call. Phase 10.1 dropped the `op.Ros2` subclass; Phase 10.2 hot-path photo runs through `outer_loop`. | ~155 | photo kernels: 8e-16 / 3.7e-11 / 1.8e-11 vs op.* |
| `outer_loop.py` | Standalone `OuterLoop` (Phase 10.6 dropped the `op.Integration` parent) — single-shot JAX runner: cond_fn carries full op.stop+op.conv termination, body fires (10.2) photo, (10.3) atm-refresh, (10.4) conden, hydrostatic balance, (10.5) ring-buffer history + adaptive rtol + photo-frequency switch, (10.6) ion charge balance + fix-all-bot post-step clamp — all inside one `lax.while_loop`. | ~1530 | 50-step HD189 atom_loss = 1.952e-04 (matches VULCAN-master to 4 sig figs); photo / atm-refresh / conden / ion / fix-all-bot branches bit-exact vs Python wrappers / numpy reference; Phase 10.5 single-shot runner shaves 19% per-step (185 → 149 ms) |
| `atm_refresh.py` | `update_mu_dz_jax` (mu / g / Hp / dz / dzi / Hpi / zco via fwd+bwd `lax.scan`) + `update_phi_esc_jax` (diffusion-limited escape flux at TOA) | 174 | bit-exact (3-7e-16) vs `op.update_mu_dz` / `op.update_phi_esc` |
| `conden.py` | `update_conden_rates` (vectorised `op.conden`), `apply_h2o_relax_jax` / `apply_nh3_relax_jax` (cold-trap implicit-Euler relax kernels). Built only when `vulcan_cfg.use_condense=True`; HD189 path skips it. | 247 | bit-exact (0.0 relerr) vs numpy reference; HD189 untouched |
| `jax_step.py` | Pure-JAX `jax_ros2_step` (JIT/vmap/GPU) — also contains the JAX diffusion kernel inline | 252 | **9.7e-92** vmap consistency |
| `photo.py` | JAX two-stream RT (compute_tau / compute_flux / compute_J / compute_J_jax_flat / pack_J_to_k_index_map / update_k_with_J) | ~520 | **8e-16 / 3.7e-11 / 1.8e-11** vs op.*, in-runner branch bit-exact (0.0) vs Python wrapper |
| `chem_funs.py` | JAX-native re-exports (`ni`/`nr`/`spec_list`/`re_dict`/`Gibbs`/`chemdf`/...) | 327 | matches VULCAN-master chem_funs to **1e-13** |
| `vulcan_jax.py` | Entry point mirroring `vulcan.py` | 135 | end-to-end run produces valid .vul |

The NumPy reference diffusion impl lives in `tests/diffusion_numpy_ref.py` (test-only; production diffusion is the JAX inline kernel in `jax_step.py`).

### Standalone (Phase A)

VULCAN-JAX is **standalone**: `python vulcan_jax.py` and the standalone test
slice run with `../VULCAN-master/` deleted or moved out of the way. The
upstream sibling, when present, serves as an *optional* validation oracle
(see Test discipline below).

Vendored from VULCAN-master:
- `atm_setup.py` — Phase 20 JAX-native rewrite of every `Atm.*` method, plus the legacy `Atm` facade.
- `ini_abun.py` — Phase 21 JAX-native rewrite of `InitialAbun.*` (5 `ini_mix` modes + `ele_sum`), plus the legacy `InitialAbun` facade. `build_atm.py` is gone after Phase 21.
- `composition.py` — Phase 21 extraction of `compo` / `compo_row` / `species` plus a pre-computed `(ni, n_atoms)` JAX array for vectorised stoichiometric sums.
- `store.py` — copied unchanged (Variables/AtmData/Parameters dataclasses)
- `phy_const.py` — copied unchanged
- `vulcan_cfg.py` — copied from VULCAN-master config (HD189 SNCHO_photo_network_2025)
- `legacy_io.py` (Phase A): vendored `ReadRate` + `Output` from `op.py:49-781,3131-3260` (plot methods dropped). Pre-loop rate-coef parser + `.vul` writer + per-step progress printer.
- `atm/`, `thermo/`, `thermo/photo_cross/`, `fastchem_vulcan/`, `cfg_examples/` — vendored runtime data, chemistry networks, photo cross sections, FastChem runtime payload, and example configs for the supported Ros2 surface.

JAX-native:
- `chem_funs.py` — **JAX-native module** (Phase 9): re-exports `ni`/`nr`/`spec_list`/`re_dict`/`re_wM_dict`/`Gibbs`/`gibbs_sp` etc. from `network.py` + `gibbs.py` + `chem.py` with NumPy-callable wrappers. No SymPy, no `make_chem_funs.py` step.

Runtime data audit (2026-04-27):
- `composition.py` reads `com_file` once at import time.
- `atm_setup.py` consumes `atm_file`, `sflux_file`, `top_BC_flux_file`, `bot_BC_flux_file`.
- `ini_abun.py` consumes `vul_ini` (for `vulcan_ini` / `table` modes) and the `fastchem_vulcan/` executable/input/output payload (for `EQ` mode).
- `legacy_io.ReadRate` consumes `network` and `cross_folder`, including
  thresholds, photo cross sections, dissociation branches, ion branches, and
  Rayleigh-scattering files.
- All non-code `atm/` and `thermo/` files from `VULCAN-master` that the
  current JAX runtime/config surface can select are now vendored locally.
- The remaining `atm/`/`thermo/` deltas are helper scripts / README files
  only: `atm/README.txt`, `atm/convert_vpl_spectra_in_nm.py`,
  `atm/make_spectra_in_nm.py`, `atm/stellar_flux/README.txt`,
  `atm/stellar_flux/plot_spectra.py`,
  `atm/stellar_flux/read_muscles_spectra_in_nm.py`, `thermo/README.md`, and
  `thermo/make_compose.py`.
- `fastchem_vulcan/input/*` and `fastchem_vulcan/output/*` are complete in the
  JAX tree; the non-mirrored FastChem files are source/build artifacts rather
  than runtime inputs.

### Phase-by-phase

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffolding + network parser | ✅ Done |
| 1 | Atmosphere, rate coefficients, Gibbs energy, initial abundances | ✅ Done |
| 2 | Pure-JAX chemistry RHS + autodiff Jacobian | ✅ Done |
| 3 | Diffusion operator + block-tridiagonal Jacobian assembly | ✅ Done |
| 4 | Block-Thomas solver + Ros2 step kernel | ✅ Done |
| 5 | Photochemistry (compute_tau / compute_flux / compute_J) | ✅ Done (Phase 9: wired to pure-JAX `photo.py` in `Ros2JAX`) |
| 6 | End-to-end forward model + .vul output | ✅ Done (HD189 50-step matches to 1.59e-10) |
| 7 | JIT + vmap parallelism, GPU readiness | ✅ Done (vmap demoed; GPU ready) |
| 8 | Condensation + ionchemistry + low-T rates | ✅ Done (inherited from `op.Ros2` originally; condensation re-ported to pure-JAX in Phase 10.4, ion charge balance in Phase 10.6) |
| 9 | JAX-native `chem_funs.py`; per-step step control + photochem + `solver_fix_all_bot` ported | ✅ Done |
| 10.1 | JAX outer loop foundation: `lax.while_loop` for accept/reject + step_size; drop `Ros2JAX(op.Ros2)` subclass; drop non-Ros2 fallback | ✅ Done |
| 10.2 | Photo update inside JAX runner: `compute_tau`/`compute_flux`/`compute_J` + `var.k` rewrite gated by `lax.cond`; photo state lives in carry | ✅ Done |
| 10.3 | Atm refresh inside JAX runner: `update_mu_dz` (fwd/bwd `lax.scan`) + `update_phi_esc` gated by `lax.cond`; hydrostatic balance moves into body; geometry fields live in carry | ✅ Done |
| 10.4 | Condensation inside JAX runner: `update_conden_rates` + `h2o_relax` / `nh3_relax` kernels (`conden.py`); conden branch gated by `do_accept & do_conden`; conden updates `var.k` rows persist across body iterations and back to Python via `_unpack_conden_k`. | ✅ Done |
| 10.5 | Convergence + stop check inside JAX cond_fn: y_time / t_time as ring buffer in carry; longdy / longdydt + adaptive rtol + photo-frequency ini→final switch all run inside the body; runner is one-shot per integration (no Python while-loop). | ✅ Done |
| 10.6 | Ion charge balance (`e[:] = -dot(y, charge_arr)`) and `use_fix_all_bot` post-step bottom clamp move into the body; `op.Integration` parent class dropped; `jax_integrate_adaptive` removed (subsumed by `OuterLoop`). | ✅ Done |
| 10.7 | Pytest enablement: `pytest.ini` + `tests/conftest.py` + thin `def test_main(): assert main() == 0` wrappers (subprocess wrappers for swap-heavy tests). The suite now covers runtime, ion-photo wiring, vendored example configs, and optional backend parity. | ✅ Done |
| A | Standalone refactor: vendored the non-code runtime inputs under `atm/`, `thermo/`, `thermo/photo_cross/`, `fastchem_vulcan/`, and `cfg_examples/`; vendored `op.ReadRate`+`op.Output` into `legacy_io.py`; dropped `sys.path.append(VULCAN_MASTER)` from runtime; oracle tests still skip cleanly when upstream is absent. | ✅ Done |
| 11 | Custom analytical chemistry Jacobian: replaces `jax.jacrev(chem_rhs_per_layer)` with stoichiometry-driven build in `chem.chem_jac_analytical`. chem_jac drops 95 ms → 2.6 ms (36×); full step 149 ms → 47 ms (3.2×). Bit-exact (≤1e-13) vs AD path. | ✅ Done |
| 12 | Runtime parity extension + end-to-end differentiability. Added `compute_Jion` photo-ionization wiring, full mid-run `fix_species` / fixed-bottom species semantics, stricter `runtime_validation.py` path/config checks, structured-input implicit AD with residual gating, factor-once/reuse block-tridiagonal solves, repaired bench/profile scripts, vendored example-config coverage, and a clean `vulture --min-confidence 80` pass. | ✅ Done |
| 13 | Stdout parity: ported `print_end_msg` and `print_unconverged_msg` from `op.Output` into `legacy_io.Output`; `vulcan_jax.py` branches on `stop_reason` to fire the right printer after `OuterLoop` returns. | ✅ Done |
| 14 | `use_fix_H2He` Hycean-world bottom-pin: added `h2he_pinned` carry field + `(use_fix_H2He, h2_idx, he_idx)` statics + a `lax.cond` between the existing fix-bottom block and the ring-buffer update. Test: `tests/test_use_fix_H2He.py`. | ✅ Done |
| 15 | `save_evolution`: `(y_evo, t_evo, evo_idx)` carry fields plus a per-accept `lax.cond` writing `(y_next, t_next)` into a fixed-size ring; `vulcan_jax.py` slices and assigns `var.y_time` / `var.t_time` for the `save_out` pickle. Test: `tests/test_save_evolution.py`. | ✅ Done |
| 16 | Chunked outer driver + live-plotting / movie / per-step `print_prog`: `OuterLoop._run_chunked` re-enters `lax.while_loop` in `print_prog_num`-sized chunks; vendored `plot_update` / `plot_flux_update` / `plot_end` / `plot_evo` / `plot_evo_inter` / `plot_TP` / movie frame-save methods into `legacy_io.Output`; `vulcan_jax.py` runs the chunk loop and fires plotting / movie / `print_prog` between chunks. Test: `tests/test_chunked_runner.py` (chunked vs single-shot bit-equivalence). Suite-ordering regression in `test_outer_loop_smoke` / `test_outer_loop_conv` (test-side `import vulcan_cfg` resolved to a different module than `outer_loop`'s after `test_chem`'s `sys.modules.pop`) closed by aliasing the test's `vulcan_cfg` to `legacy_io.vulcan_cfg` and re-syncing `sys.modules` + `outer_loop.vulcan_cfg`. | ✅ Done |
| 17 | `chem_rhs` cancellation-floor investigation. Built and tested a Neumaier-compensated single-scan `_neumaier_segment_sum`; verified bit-identical to `math.fsum` on the same per-term values. Established that the JAX↔master ~1e-4 floor is **per-term roundoff** in the rate computation (JAX `jnp.prod` + masked `y**stoich` vs master's SymPy-emitted `y[A] * y[B]` differ by ~1 ulp per multiply), not segment-sum drift — so Stage 1 / Stage 2 of the original plan cannot close the gap. Reverted to the fast tree-reduction `segment_sum`; documented the per-term floor in `CLAUDE.md` (Numerical hygiene → `chem_rhs` cancellation floor). Stage 3 (codegen against SymPy emit order) deferred. | ✅ Done (investigation; production path unchanged) |
| 18 | Per-test fixture sharing scaffolding: `tests/conftest.py` exposes `_hd189_pristine` (session-scoped, runs the full HD189 pre-loop once) and `hd189_state` (per-test deep copy of the mutable state). Both pin `legacy_io.vulcan_cfg ≡ outer_loop.vulcan_cfg ≡ sys.modules["vulcan_cfg"]` so the Phase 16 sync pattern is paid once, not per test. Migrated `test_chem_jac_sparse` as the canonical example; standalone `python tests/...` invocation still works via an inline state-builder fallback. Future migrations follow the same shape. `ruff check` passes on the touched files. | ✅ Done (foundation + 1 test) |
| 19 | Typed pre-loop input pytree foundation + live-output drop. New `state.py` module defines `AtmInputs` / `RateInputs` / `PhotoInputs` / `RunState` (NamedTuple-based JAX pytrees) plus `pytree_from_store(var, atm) -> RunState` / `apply_pytree_to_store(state, var, atm)` adapters and a `load_stellar_flux(cfg) -> StellarFlux` host-side reader. `store.Variables.__init__` now accepts an optional `stellar_flux=` payload so `vulcan_jax.py` can pre-load on the host (legacy callers that pass nothing still get the file read for backwards compatibility). Live-output paths (`use_live_plot`, `use_live_flux`, `use_save_movie`, `use_flux_movie`) are rejected by `runtime_validation.py` with a clear "Phase 19 dropped live output" error; `legacy_io.Output.plot_update` / `plot_flux_update` / `plot_evo_inter` and the corresponding `outer_loop._run_chunked` callbacks are deleted; `Output.plot_end` / `plot_evo` / `plot_TP` survive as post-run-only PNG writers. `store.Parameters.pic_count` and `tableau20` deleted. Tests: `tests/test_state_roundtrip.py` (4 tests — pytree round-trip + field-set completeness + stellar-flux loader cases), `tests/test_drop_live_output.py` (7 tests — validator rejection, validator acceptance, Output method-set, Parameters attribute drop). Suite: 37→48 pass, vulture clean, ruff clean on new files. | ✅ Done |
| 20 | JAX-native `Atm.*` setup. New `atm_setup.py` module (~640 lines) hosts pure JAX kernels: `compute_pico`, `analytical_TP_H14` (uses `jax.scipy.special.expn`), `load_TPK` (handles `atm_type ∈ {file, isothermal, analytical, vulcan_ini, table}` and `Kzz_prof ∈ {const, JM16, Pfunc, file}`, descending-pressure atm files flipped on host before `jnp.interp`), `compute_mean_mass`, `compute_mu_dz_g` (sequential height-integration via `jax.lax.scan` upward and downward from `pref_indx`), `compute_settling_velocity`, `compute_mol_diff` (vmapped per species, four `atm_base` modes), `read_bc_flux`, `compute_sat_p` (eight condensables), `read_sflux_binned`. `build_atm.py` thinned: `Atm` body deleted, `from atm_setup import Atm` re-exports the facade for the ~20 existing test/runner call sites; `compo` / `compo_row` / `species` and `InitialAbun` stay until Phase 21. Tests: `tests/test_atm_setup_matrix.py` (31 parametrized cases — 3 `atm_type` × 4 `Kzz_prof` × 4 `atm_base` × BC modes × 8 condensables, all comparing JAX kernels to direct numpy/scipy references at ≤1e-13 rtol). Suite: 48→79 pass, vulture clean, ruff clean on new files; HD189 50-step oracle still bit-exact at 1.59e-10. | ✅ Done |
| 21 | JAX-native `InitialAbun.*` setup + `build_atm.py` deletion. New `composition.py` extracts the `compo` / `compo_row` / `species` data tables (read once at import time) plus a pre-computed `(ni, n_atoms)` JAX `compo_array` for vectorised stoichiometric aggregation. New `ini_abun.py` (~470 lines) hosts pure JAX kernels: `_abun_lowT_residual` (5-mol H2/H2O/CH4/He/NH3) and `_abun_highT_residual` (4-mol parity helper); `_jax_newton` (`lax.while_loop` + `jax.jacrev`, replaces `scipy.optimize.fsolve`); `compute_atom_ini` (`jnp.einsum` replaces master's per-atom Python loop in `ele_sum`); plus the host-side mode dispatch for `EQ` (FastChem subprocess, paths byte-identical), `const_mix`, `vulcan_ini` (pickle), `table` (`np.genfromtxt`), and `const_lowT` (JAX Newton). The `InitialAbun` class is the legacy facade that mutates `data_var` / `data_atm` so `vulcan_jax.py` and the ~17 test/example/benchmark call sites keep working. New `state.IniAbunOutputs` NamedTuple defines the gas-phase composition + atom totals + charge_list pytree slot. **`build_atm.py` is deleted** at end of phase; all JAX-side imports rewired to `atm_setup` / `ini_abun` / `composition`. Tests: `tests/test_ini_abun.py` rewritten as a parametrized matrix (8 tests — `const_mix` / `vulcan_ini` / `table` / `const_lowT` ×3 / `charge_list` invariant / EQ-mode bit-exact fork against master); JAX Newton matches scipy fsolve to ~1e-16 across 4× wider elemental-ratio range than HD189. Suite: 79→86 pass + 1 skip; vulture clean; ruff clean on new files; HD189 50-step oracle still bit-exact at 1.59e-10. | ✅ Done |
| 22a | Rate-parser consolidation. Three new entry points in `rates.py`: `apply_lowT_caps` (3 hardcoded reactions, masked `np.where`, mirrors `legacy_io.lim_lowT_rates`), `apply_remove_list` (literal zero of indices in `cfg.remove_list`, mirrors `legacy_io.remove_rate`), `build_rate_array` (forward → optional lowT caps → reverse via Gibbs → remove pass; returns dense `(nr+1, nz)` `np.ndarray`). Production rewire: `vulcan_jax.py` now calls `build_rate_array` in place of `rev_rate` / `remove_rate` / `lim_lowT_rates` (legacy `read_rate` retained for metadata population — `var.Rf`, `var.pho_rate_index`, `var.n_branch`, `var.photo_sp`, etc.). New `Variables.k_arr` field on `store.Variables` carries the dense mirror; `var.k` dict is mirrored via `rates.k_dict_from_array` for back-compat with photo / conden writers and the `.vul` writer until Phase 22b retires the dict surface. Bit-exact (0.0 abs/rel err) vs legacy chain on HD189; HD189 setup smoke-passes through the rewired path. New `tests/test_read_rate.py` (6 tests covering caps fire, caps no-op, remove_list literal-only semantics, none/empty no-op, end-to-end build_rate_array vs legacy at ≤1e-13, lowT-on with C2H4 cap firing on HD189 cooler layers). Ruff/vulture clean on `rates.py` + `tests/test_read_rate.py`. Legacy `legacy_io.ReadRate.{read_rate, rev_rate, remove_rate, lim_lowT_rates}` retained — deletion deferred to Phase 22b along with the `var.cross*` dict-surface retirement (the 20+ test files that currently call the legacy chain don't need rewriting in 22a). | ✅ Done |
| 22b | Photo cross-section preprocessing rewrite. New `photo_setup.py` (~430 lines, pure NumPy) hosts host-side CSV readers (`_load_thresholds`, `_load_cross_csv`, `_load_branch_csv`, `_load_ion_branch_csv`, `_discover_T_cross_files`, `_load_T_cross_csv`, `_load_rayleigh_csv`), a `_make_bins` two-resolution wavelength-grid kernel, `_interp_zero_extrap` / `_interp_edge_extrap` bit-exact replacements for `scipy.interpolate.interp1d(kind='linear', bounds_error=False, fill_value=...)`, a `_bin_T_dependent` per-layer × per-bin loop with the 3-case T branching (Tz between Tlow/Thigh; Tz < min_T; Tz > max_T) plus the `log10`-space `-100` sentinel pattern, and a top-level `populate_photo_arrays(var, atm)` that mirrors the legacy `make_bins_read_cross` mutation contract: writes `var.threshold` / `var.bins` / `var.dbin1/2` / `var.cross[sp]` / `var.cross_J[(sp, i)]` / `var.cross_scat[sp]` / `var.cross_T[sp]` / `var.cross_J_T[(sp, i)]` / `var.cross_Jion[(sp, i)]` / pre-allocated `var.sflux` / `var.dflux_u/d` / `var.aflux` / `var.tau` / `var.sflux_top`. Production rewire: `vulcan_jax.py` calls `photo_setup.populate_photo_arrays` in place of `rate.make_bins_read_cross`; the dense `var.k_arr` is re-synced via `rates.k_array_from_dict` after photo-J writes overwrite `var.k` for photodissociation reactions. Discovered + fixed a subtle bit-exactness issue: `scipy.interp1d` sorts `xp` internally, but `np.interp` does not — non-monotonic data files (e.g. CH3SH_branch.csv has a typo `354.0` between `253.5` and `309.5`) produce different results between the two unless we sort first. New `_sort_pairs` helper does this. New `tests/test_photo_setup.py` (2 tests): bit-exact match vs legacy on HD189 (use_photo=True, T_cross_sp=[], use_ion=False, scat_sp=['H2','He']) and bit-exact match with `T_cross_sp=['CO2','H2O','NH3']` patched on (Earth-style T-dep path). Ruff/vulture clean on `photo_setup.py` + `tests/test_photo_setup.py`. Legacy `legacy_io.ReadRate.make_bins_read_cross` and the dict-keyed `var.cross*` surface retained for back-compat with photo runtime kernels (`photo.pack_photo_data`), `outer_loop` photo writers, and the `.vul` writer; full `ReadRate` deletion + dict-surface retirement is a future cleanup phase. End-to-end HD189 5-step run completes cleanly through the rewired photo path (3.1s wall). | ✅ Done |
| 22c | Legacy ReadRate cleanup (rate-method retirement). Two new convenience helpers in `rates.py`: `setup_var_k(cfg, var, atm) -> Network` (parses network, loads NASA-9, builds dense `var.k_arr`, mirrors to dict) and `apply_photo_remove(cfg, var, network, atm)` (re-syncs `var.k_arr` from dict after photo-J writes, applies `cfg.remove_list`, mirrors back). Production rewire: `vulcan_jax.py` photo block now uses `apply_photo_remove` instead of `rate.remove_rate`; `tests/conftest.py` `_hd189_pristine` fixture and ~15 standalone test files (`test_outer_loop_smoke` / `test_chunked_runner` / `test_outer_loop_conv` / `test_outer_loop_reset` / `test_outer_loop_photo` / `test_save_evolution` / `test_chem_jac_sparse` / `test_solver_fix_all_bot` / `test_use_fix_H2He` / `test_integrate` / `test_vmap_step` / `test_backend_parity` / `test_cfg_examples` / `profile_step` / `examples/grad_implicit_example` / `benchmarks/bench_step`) migrated off the legacy `read_rate -> rev_rate -> remove_rate -> [lim_lowT_rates]` chain to `read_rate + setup_var_k`. Photo-block `make_bins_read_cross` callers migrated to `photo_setup.populate_photo_arrays`. After all migrations, `legacy_io.ReadRate.{rev_rate, remove_rate, lim_lowT_rates}` are deleted (their replacements live in `rates.{apply_remove_list, apply_lowT_caps}` and `gibbs.fill_reverse_k`). Oracle tests using VULCAN-master's `op.ReadRate()` are unaffected (they import master's `op` directly). `make_bins_read_cross` retained for `tests/test_photo_setup.py`'s bit-exact regression oracle (deletion deferred — needs `var.cross*` dict-surface retirement first). HD189 50-step oracle still bit-exact at 1.59e-10. | ✅ Done (rate-method cleanup; full dict-surface retirement deferred) |
| 22d | `var.k` dict-surface retirement. Photo writers (`op_jax.compute_J`, `compute_Jion`) now write the per-(sp,nbr) J-rates straight into the dense `var.k_arr[ridx, :]` instead of the legacy dict; `outer_loop._unpack_k` / `_unpack_conden_k` collapsed into a single full-array snapshot from `state.k_arr`; `state.pytree_from_store` reads `var.k_arr` directly (no per-row dict pack); `state.apply_pytree_to_store` writes `var.k_arr` (no dict reverse-write). `rates.setup_var_k` and `rates.apply_photo_remove` no longer mirror the dict. `legacy_io.read_rate` keeps `k = {}` as scratch space and discards it; `Variables.__init__` drops `self.k = {}`. The `.vul` writer adapter in `legacy_io.Output.save_out` synthesizes the legacy `{i: array(nz)}` dict view from `var.k_arr` at pickle time so downstream `plot_py/` scripts that index `d['variable']['k'][i]` keep working unchanged. Migrated 8 test/example/benchmark files (`tests/test_outer_loop_photo` / `test_chem_jac_sparse` / `test_integrate` / `test_vmap_step` / `test_backend_parity` / `test_read_rate` / `test_photo_ion` / `tests/profile_step` plus `examples/grad_implicit_example` / `examples/batched_run` / `benchmarks/bench_step`) off `var.k.items()` to `var.k_arr` indexing. `var.cross*` dict surface retained pending the parallel `photo.pack_*_data` rewrite (deferred to a future Phase 22e — the photo runtime kernels still consume the dicts at startup). HD189 50-step oracle still bit-exact at 1.59e-10. | ✅ Done |
| 22e | `var.cross*` dict-surface retirement. New `state.PhotoStaticInputs` NamedTuple holds the dense cross-section pytree (`absp_cross` / `absp_T_cross` / `cross_J` / `cross_J_T` / `cross_Jion` / `scat_cross` plus species/branch ordering tuples and bin-grid scalars). New `photo_setup.populate_photo` builds it; `photo_setup.populate_photo_arrays` becomes a thin compat wrapper that calls `populate_photo` and discards the return (so the ~17 standalone test/example/benchmark callers keep working). `photo.pack_photo_data` / `pack_photo_J_data` / `pack_photo_ion_data` are deleted, replaced by `photo.photo_data_from_static` / `photo_J_data_from_static` / `photo_ion_data_from_static` (pure selectors over the pytree). `op_jax.Ros2JAX` accepts `photo_static=` and lazily constructs the static via `_ensure_photo_static(var, atm)` if not wired (covers test sites that do `Ros2JAX()` then `compute_tau(var, atm)`). `outer_loop._build_photo_static` reads from `solver._photo_static` (lazy-built when needed) instead of `var.cross*` dicts. `vulcan_jax.py` builds the static after `read_sflux`, `_replace`s `din12_indx` from `var.sflux_din12_indx`, stashes it on the solver, and threads it into `output.save_out(..., photo_static=...)`. `legacy_io.Output.save_out` adds a `_synthesize_cross_dicts` helper that rebuilds the six legacy dict views (`cross` / `cross_J` / `cross_T` / `cross_J_T` / `cross_Jion` / `cross_scat`) from the dense pytree at pickle time, so `plot_py/` consumers see the same `.vul` schema. `legacy_io.ReadRate.make_bins_read_cross` is deleted (~285 lines); `store.Variables` drops the `self.threshold = {}` and `self.cross_T_sp_list = {}` defaults (now write-once locals during photo setup). `tests/test_photo_setup.py` migrated to the npz-fixture comparison (`tests/data/photo_setup_hd189_baseline.npz` + `tests/data/photo_setup_hd189_T_dep.npz`); the legacy oracle dependency is gone. `tests/test_photo.py` and `tests/test_photo_ion.py` migrated to consume the static directly. `tests/_gen_photo_baseline.py` updated to dump from the dense pytree. HD189 50-step oracle still bit-exact at 1.59e-10. | ✅ Done |

### Drop-in compatibility

- ✅ Same `vulcan_cfg.py` format (just copy from VULCAN-master)
- ✅ Same input file paths/formats (`atm/`, `thermo/`, stellar fluxes, photo cross-sections)
- ✅ Same `.vul` pickle output schema (loadable by VULCAN's `plot_py/` scripts)
- ✅ Same command-line invocation: `python vulcan_jax.py` (the `-n` flag is accepted for compatibility but is now a no-op since `chem_funs` is JAX-native)

### Validation tests

Current CPU-only run with `VULCAN-master` present: **94 PASS + 1 SKIP** (`pytest tests/`). The skipped test is the optional GPU backend-parity check (`test_backend_parity`), which skips cleanly on CPU-only hosts.

```
test_network_parse        PASS  ni=93, nr=1192, full spec_list match
test_rates                PASS  Max relerr = 0 (machine precision)
test_gibbs                PASS  K_eq, gibbs_sp, reverse k all to 1.4e-14
test_build_atm            PASS  pco/pico/Tco/Kzz/M/n_0 all exact
test_ini_abun             PASS  8 tests: EQ vs master fork (≤1e-10), const_mix algebraic (1e-13), vulcan_ini pickle round-trip, table mode synth, const_lowT JAX Newton vs scipy fsolve (≤1e-13 across 3 ratio regimes), charge_list invariant; (Phase 21)
test_chem                 PASS  chem_rhs 1e-4 (FP), chem_jac 4.3e-13
test_diffusion            PASS  diff op 2e-6, sub/sup blocks to 1e-15
test_diffusion_variants   PASS  4 modes (gravity / vm / settling / settling_vm)
test_block_thomas         PASS  3e-15 vs np.linalg.solve (full nz=120 ni=93)
test_ros2_step            PASS  1.16e-15 vs VULCAN's Ros2.solver
test_vmap_step            PASS  9.7e-92 vmap consistency
test_integrate            PASS  pure-JAX adaptive dt loop
test_photo                PASS  compute_tau/flux/J kernels (1e-11 to 8e-16)
test_chem_funs_compat     PASS  JAX-native chem_funs vs SymPy ref (Phase 9)
test_photo_wired          PASS  Ros2JAX.compute_tau/flux/J wired (Phase 9)
test_step_control         PASS  outer_loop's pure-JAX clip/loss/step_size match op.ODESolver to ~1e-15 (rewritten Phase 10.1)
test_solver_fix_all_bot   PASS  use_fix_all_bot=True post-step bottom clamp pinned to bit-exact (0 relerr) for 10-step HD189 (Phase 10.6)
test_outer_loop_smoke     PASS  50-step HD189 via OuterLoop, atom_loss within 5% of 1.95e-4 baseline (Phase 10.1)
test_outer_loop_photo     PASS  Photo branch inside runner bit-exact (0.0 relerr on tau/aflux/dflux/J/k) vs Python wrapper (Phase 10.2)
test_outer_loop_atm_refresh  PASS  Atm-refresh branch (mu/g/Hp/dz/dzi/Hpi/zco/top_flux) bit-exact (≤5e-16 relerr) vs op.update_mu_dz/op.update_phi_esc + Python hydro balance (Phase 10.3)
test_conden_jax              PASS  Conden kernels (update_conden_rates, h2o/nh3 relax) bit-exact (0.0 relerr) vs numpy reference of op.conden / op.h2o_conden_evap_relax / op.nh3_conden_evap_relax (Phase 10.4)
test_outer_loop_conv         PASS  Single-shot runner: ring-buffered y_time/t_time chronology, longdy/longdydt populated, count_max termination via cond_fn (Phase 10.5)
test_outer_loop_ion          PASS  Ion charge balance kernel: `e[:] = -dot(y, charge_arr)` formula bit-exact and net electron count zero on synthetic state (Phase 10.6)
test_chem_jac_sparse         PASS  chem_jac_analytical (Phase 11) vs jacrev path: max relerr ≤1e-13 on real HD189 state (standalone — no VULCAN-master needed)
test_block_thomas_diag       PASS  block_thomas_diag_offdiag vs dense block_thomas: ≤1e-12 on small system, ≤1e-9 on (nz=120, ni=93); jax.grad finite (Phase 12)
test_outer_loop_reset        PASS  OuterLoop.reset() invalidates the JIT'd runner cache; rebuilt _Statics carries fresh vulcan_cfg values (Phase 12)
test_steady_state_grad       PASS  custom_vjp implicit gradient vs central FD on synthetic non-singular system; max |jax-FD|/scale = 1.8e-4 (Phase 12)
test_photo_ion               PASS  compute_Jion / ion `k_arr` wiring vs direct NumPy integration (Phase 12)
test_cfg_examples            PASS  vendored Earth / Jupiter / HD209 configs validate and complete pre-loop setup without borrowing files from `VULCAN-master` (Phase 12)
test_use_fix_H2He            PASS  Hycean H2/He bottom-pin fires once at `t > 1e6`, snapshot matches manual `lax.cond` reference (Phase 14)
test_save_evolution          PASS  `(y_evo, t_evo)` ring populated at `save_evo_frq` cadence and round-trips through `legacy_io.save_out` pickle (Phase 15)
test_chunked_runner          PASS  `OuterLoop._run_chunked` matches single-shot final state bit-for-bit (y/t/dt/longdy/count agree to ≤1e-12) on 30-step HD189 with `print_prog_num=10` (Phase 16)
test_state_roundtrip         PASS  4 tests: `pytree_from_store -> apply_pytree_to_store -> pytree_from_store` is identity on HD189; AtmInputs field set is complete; load_stellar_flux returns empty payload when use_photo=False; load_stellar_flux(HD189) produces valid bin extents (Phase 19)
test_drop_live_output        PASS  7 tests: validator rejects use_live_plot / use_live_flux / use_save_movie / use_flux_movie when True; accepts when False; Output exposes no live-plot methods; Parameters has no pic_count / tableau20 (Phase 19)
test_atm_setup_matrix        PASS  31 tests: load_TPK across atm_type {isothermal, analytical, file} and Kzz_prof {const, JM16, Pfunc, file}; compute_mol_diff across atm_base {H2, N2, O2, CO2}; read_bc_flux defaults / topflux / botflux; compute_sat_p over 8 condensables; f_pico vs master formula; compute_mu_dz_g rocky vs gas-giant anchors; settling velocity off / H2SO4-on; analytical_TP_H14 vs scipy expn (Phase 20)
test_backend_parity          SKIP  optional CPU↔GPU float64 parity check; runs only when a GPU backend is present (Phase 12)
```

### End-to-end validation

| Config | Network | Photo | Steps | Status | Max rel err | atom_loss |
|---|---|---|---|---|---|---|
| **HD189** | SNCHO_photo_network_2025 | ✓ | 50 | **PASS** | **1.59e-10** | 1.95e-4 (both) |

The single-Ros2-step agrees with VULCAN to machine precision (1e-15). After 50 steps with photochem updates wired through pure-JAX kernels, max relerr is 1.59e-10 (vs the target ≤1e-9, well within margin). atom_loss matches to all four significant figures, t and dt match to display precision.

**Full-convergence runs end at different t.** A converged JAX run (`python vulcan_jax.py`, 2766 steps, 138 s wall) and a converged master run (2346 steps, t=8.78e7) both satisfy `longdy < 0.01` but stop at different physical times because the adaptive step controller's path-dependence amplifies `chem_rhs`'s ~1e-4 cancellation floor over hundreds of steps. The 1.59e-10 above is the *kernel-level* JAX-vs-master agreement on a 50-step trajectory and is the right metric for validating the integration physics. For a converged-state comparison, run both with matched `count_max` (the 50-step cell above already does this).

---

## ⏳ PARTIALLY DONE

### Performance optimization

**Current state**: the checked-in timing scripts now reflect the current APIs rather than stale pre-Phase-12 entrypoints. On the HD189 default case they report:

| Metric | Current timing |
|---|---|
| `benchmarks/bench_step.py`: upstream `Ros2.solver` single-step | `125.993 ms` |
| `benchmarks/bench_step.py`: local `jax_ros2_step` | `22.618 ms` |
| `benchmarks/bench_step.py`: local `OuterLoop` 50-step smoke | `39.381 ms/accepted step` |
| `tests/profile_step.py`: `pack_k_arr` | `0.270 ms` |
| `tests/profile_step.py`: `make_atm_static` | `0.726 ms` |
| `tests/profile_step.py`: `chem_rhs` | `3.753 ms` |
| `tests/profile_step.py`: `chem_jac_analytical` | `8.292 ms` |
| `tests/profile_step.py`: block factorization | `8.515 ms` |
| `tests/profile_step.py`: block solve | `9.050 ms` |
| `tests/profile_step.py`: full profiled `jax_ros2_step` | `57.585 ms` |

**Key changes**:
- Phase 9: switched `chem_jac_per_layer` from `jax.jacfwd` to `jax.jacrev` (~25% faster, better numerical accuracy 4.3e-13 vs 6e-11).
- Phase 11: replaced `jacrev(chem_rhs_per_layer)` with `chem.chem_jac_analytical` — a stoichiometry-driven analytical build that scatters `sign_i * stoich_i * stoich_j * y_j^(stoich_j-1) * (rest_of_factors) * k_r * M_factor` directly into the (ni, ni) Jacobian via `segment_sum`. Bit-exact (≤1e-13 relerr) vs the AD path on HD189 reference state. The leave-one-out reactant product avoids the `(stoich/y)*rate` divide-by-zero trap when reactant abundance is zero (e.g. CH3CCH/CH2CCH2 in initial HD189).
- Phase 12: `solver.block_thomas_diag_offdiag` now factors the block-tridiagonal LHS once and reuses that factorization across both Rosenbrock stages. The diffusion Jacobian still exploits diagonal-in-species off-diagonals for the `O(ni²)` rank update, and the y-independent gravity contributions are pre-baked via `compute_diff_grav(atm)`.

## Differentiability (Phase 12)

Hot-path kernels — `chem_rhs`, `chem_jac_analytical`, `block_thomas_diag_offdiag`, `_build_diff_coeffs_jax`, `_apply_diffusion_jax`, `compute_tau_jax`, `compute_flux_jax`, `compute_J_jax_flat`, `update_mu_dz_jax`, conden kernels, `jax_ros2_step`, `jax_integrate_fixed_dt` — are all `jit` / `vmap` / `jvp` / `vjp` compatible.

`outer_loop.runner`'s `jax.lax.while_loop` supports `jvp` (forward-mode tangents — verified at the per-step kernel level; `examples/grad_jvp_example.py`) but raises on `vjp` / `grad`. So end-to-end reverse-mode AD across the integration goes through the structured-input helpers in `steady_state_grad.py`:

- The converged state `y*` satisfies `f(y*, theta) = 0` where `f = chem_rhs + diffusion`.
- Implicit function theorem: `∂y*/∂theta = -(∂f/∂y)^{-1} (∂f/∂theta)`.
- The `(∂f/∂y)` matrix is the same block-tridiagonal LHS that `block_thomas_diag_offdiag` already inverts every Ros2 step. Backward of `differentiable_steady_state`: 1 transpose solve + 1 `chem_rhs` VJP. **O(1) memory** in step count — no checkpointing required regardless of how many steps the forward integration took.

`tests/test_steady_state_grad.py` validates both the legacy `k_arr` wrapper and the structured-input API against finite differences on a tiny synthetic non-singular system (open chain: zero-order source of A, A→B, B→sink). Max `|jax.grad - FD| / max|FD| = 1.8e-4`. `validate_steady_state_solution(...)` / `steady_state_value_and_grad(...)` now enforce a residual tolerance before attaching implicit AD; if the supplied `y_star` is not converged tightly enough, they fail loudly rather than returning an untrustworthy gradient.

### Multi-CPU parallelism via pmap

**Demonstrated**: `jax.vmap` over 4 atmospheres works and gives consistent results.

**Not done**: Real `pmap` setup for multi-CPU production use. The right invocation is:
```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py
```
combined with `jax.pmap(jax_ros2_step, ...)` over the batch dimension. Untested but the architecture supports it.

### Full-convergence validation runs

**Done**: HD189 50 steps (matches to 1.59e-10 post-Phase 9; same on Phase 10.6 path).

**Done**: HD189 50-step oracle match; vendored Earth / Jupiter / HD209 configs now validate and complete the full pre-loop setup inside `tests/test_cfg_examples.py`.

**Pending**: matched-step/full-convergence oracle campaigns for Earth, Jupiter, and HD209. Those are still the expensive long-run validation gap, not a missing runtime branch.

### Pure-JAX `integrate.py` loop

`integrate.py` is now a single helper — `jax_integrate_fixed_dt(y0, k_arr, dt, n_steps, atm, net)` — that takes N fixed-dt Ros2 steps in one `jax.lax.scan`. Validated to 1.5e-16 vs equivalent Python loop; ~1.5x faster (135 ms/step warm vs 195 ms/step Python). Used for benchmarks and as a JIT-vs-Python-loop equivalence check, not for production runs.

The Phase 10.6 production drop-in is `outer_loop.OuterLoop` (single-shot adaptive runner with photo / conden / convergence / ion / fix-all-bot all on device). The legacy `jax_integrate_adaptive` was deleted because `OuterLoop` subsumes it.

---

## Open follow-ups

- **Matched-step / full-convergence oracle runs for Earth, Jupiter, and HD209.** The supported assets/configs are vendored and their pre-loop setup is tested, but the expensive long-run oracle campaigns are still outstanding (intentionally deferred per user direction).
- **C3 (real-GPU CPU↔GPU parity numbers) — deferred.** The host is CPU-only, so production GPU timings and float64 parity numbers cannot be recorded here. The optional `test_backend_parity` is in place and skips cleanly when no GPU backend is present; this remains a documentation gap pinned to GPU hardware availability, not an implementation gap.
- **`chem_rhs` cancellation floor — Stage 3 [WONTFIX, maintainer policy].** The Phase 17 investigation identified per-term roundoff (JAX `jnp.prod(y**stoich)` vs master's SymPy-emitted `y[A]*y[B]*y[C]`, ~1 ulp per multiply) as the cap on JAX↔master agreement for ~3 species. Closing the gap requires SymPy-faithful codegen at ~14× per-call cost on the per-step hot path. Maintainer has explicitly opted to leave this floor in place — it is a real, documented per-term ulp drift, but the perf trade is net-negative. Do not work on this. See `CLAUDE.md` "Numerical hygiene → `chem_rhs` cancellation floor" and `README.md` "Validation".
- **Per-test fixture sharing — finish migration.** Phase 18 added the `hd189_state` session-scoped fixture in `tests/conftest.py` and migrated `test_chem_jac_sparse` as the canonical example; the remaining script-style tests still build their own state. Most can switch to the fixture (see `test_chem_jac_sparse` for the pattern); a handful that mutate `vulcan_cfg` (e.g. `count_max=50` in `test_outer_loop_smoke`) need a snapshot/restore wrapper, and the two `sys.modules.pop`-style tests (`test_diffusion`, `test_ros2_step`) stay subprocess-based because they require a cold module table.
- **Parallel pytest execution.** `pytest -n auto` still collides on FastChem's fixed output path; run serially.

---

## Complete VULCAN replacement scope

This section defines the scope for "complete JAX rewrite" used in this repo.

- The goal is to replace **VULCAN-owned** implementation code with JAX where
  practical, while preserving all **live** `VULCAN-master` functionality in the
  JAX tree.
- External dependencies that `VULCAN-master` already calls do **not** need to
  be rewritten. The goal is to replace VULCAN itself, not every third-party
  tool in its stack.
- In particular, **FastChem is acceptable as an external dependency** because
  `VULCAN-master` already shells out to it for `ini_mix = 'EQ'`.
- matplotlib / PIL likewise do not need JAX rewrites; they only need to remain
  supported if the corresponding VULCAN surface remains supported.
- Ros2 is the only live solver in `VULCAN-master`. The other solver classes in
  `op.py` are commented-out dead code and are not part of the required rewrite
  surface unless resurrecting them becomes a separate project.

## Remaining work for a complete VULCAN replacement

### Already assigned elsewhere

- **Phase 16 suite-ordering regression.** Test isolation / cfg-module state
  leakage across the suite.
- **Phase 17 compensated `chem_rhs` summation.** Numerical-quality cleanup of
  the documented cancellation floor.
- **Phase 18 per-test fixture sharing.** Test-runtime and hygiene cleanup.

### A. Must-do runtime rewrite work

- ✅ **`build_atm.py` is gone (Phases 20–21).** Phase 20 ported `Atm.*` to
  JAX-native `atm_setup.py`; Phase 21 ported `InitialAbun.*` to JAX-native
  `ini_abun.py` (5 `ini_mix` modes; `scipy.optimize.fsolve` replaced with a
  small JAX Newton on the 5-mol H2/H2O/CH4/He/NH3 system) and extracted the
  `compo` / `compo_row` / `species` data tables into `composition.py`.
  `build_atm.py` was deleted at end of Phase 21.
- **Keep the FastChem-backed EQ path healthy, but do not treat FastChem itself
  as a rewrite blocker.** The VULCAN-owned wrapper around FastChem still needs
  to remain supported and tested in the JAX tree. (Phase 21 preserved the
  subprocess invocation and FastChem I/O paths byte-for-byte; the
  `pytest.ini` serial-execution requirement still applies.)
- **Replace vendored `legacy_io.ReadRate` with JAX-native preprocessing.**
  Remaining surface includes forward-rate parsing, reverse-rate setup, low-T
  caps, photo-bin construction, T-dependent photo cross sections, dissociation
  branch ratios, ion branch ratios, and Rayleigh-scattering ingestion. The
  current implementation still leans on NumPy / SciPy interpolation and
  Python dict-heavy host structures.
- **Reduce dependence on vendored mutable `store.py` objects.** The runtime
  still packs and unpacks legacy mutable containers. A cleaner end-state is a
  typed pytree / dataclass representation with thin compatibility wrappers
  only where needed.
- ✅ **Chunked/live-output semantics — settled in Phase 19.** Live-plot,
  live-flux, save-movie, and flux-movie were dropped; `runtime_validation.py`
  rejects those flags. Post-run plotters (`plot_end`, `plot_evo`, `plot_TP`)
  remain as PNG writers. The chunked host-callback driver still exists for
  `print_prog` and the surviving post-run plotters.

### B. Full repo-surface parity still missing

- **Missing top-level master modules:** `diagnose.py`, `make_chem_funs.py`.
- **Missing `plot_py/` surface (18 files):**
  `README.md`, `plot_Earth.py`, `plot_Jupiter_Tsai2021.py`, `plot_TP_Kzz.py`,
  `plot_actinic_flux.py`, `plot_cross.py`, `plot_cross_branches.py`,
  `plot_evolution.py`, `plot_helios_spectra.py`, `plot_many_vulcan.py`,
  `plot_spectra.py`, `plot_stellar_flux.py`, `plot_tau1_height_species.py`,
  `plot_tau1_species.py`, `plot_tot_cross.py`, `plot_vulcan.py`,
  `prt_spectrum.py`, `stellar_flux_resample.py`.
- **Missing `tools/` surface (3 files):**
  `make_mix_table.py`, `make_spectra_in_nm.py`, `print_actinic_flux.py`.
- **Remaining `atm/` helper/docs surface (6 files):**
  `README.txt`, `convert_vpl_spectra_in_nm.py`, `make_spectra_in_nm.py`,
  `stellar_flux/README.txt`, `stellar_flux/plot_spectra.py`,
  `stellar_flux/read_muscles_spectra_in_nm.py`.
- **Remaining `thermo/` helper/docs surface (2 files):**
  `README.md`, `make_compose.py`.
- **Optional `fastchem_vulcan/` source/build mirror.**
  The runtime payload (`fastchem`, `input/*`, `output/*`) is already present.
  What remains unmatched is the FastChem source/build tree, which is not
  needed for runtime parity but would matter if compile-from-source repo parity
  becomes a requirement.

### C. Validation gaps still open

- **Matched-step and longer oracle campaigns beyond HD189.**
  Earth / Jupiter / HD209 still need the same depth of oracle coverage that
  HD189 has now.
- **Full config-matrix coverage.**
  The remaining matrix to exercise explicitly includes:
  `use_photo`, `use_ion`, `use_condense`, `use_lowT_limit_rates`,
  non-empty `T_cross_sp`, `use_vm_mol`, `use_settling`, `use_topflux`,
  `use_botflux`, non-empty `use_fix_sp_bot`, `use_fix_all_bot`,
  `use_fix_H2He`, non-empty `fix_species`, `save_evolution`, and any
  retained plot / movie / live-output path.
- **Atmosphere-setup mode coverage.**
  `atm_type = file / isothermal / analytical / vulcan_ini / table`
  should all be exercised if they remain supported.
- **Initialization-mode coverage.**
  `ini_mix = EQ / const_mix / vulcan_ini / table` should all be exercised.
- **Real GPU parity / multi-device validation.**
  The repo has a backend-parity test, but checked-in CPU↔GPU numbers on real
  hardware and broader production validation are still pending.

### D. Suggested order

1. ✅ Finish the Phase 16 regression fix and restore a stable all-green suite.
2. ✅ Keep FastChem as an accepted external dependency and stop treating it
   as a rewrite blocker.
3. ✅ Phase 19: typed pre-loop input pytree foundation (`state.py`) +
   live-plot / movie support semantics (rejected).
4. ✅ Phase 20: `Atm.*` rewrite (atmosphere setup) into `atm_setup.py`
   (JAX-native pure functions + thin facade for legacy callers). New
   `tests/test_atm_setup_matrix.py` covers `atm_type ∈ {isothermal,
   analytical, file}`, `Kzz_prof ∈ {const, JM16, Pfunc, file}`,
   `atm_base ∈ {H2, N2, O2, CO2}`, BC modes, and 8 condensables —
   all asserting ≤1e-13 vs direct numpy/scipy references.
5. ✅ Phase 21: `InitialAbun.*` rewrite into `ini_abun.py` (JAX Newton
   on the 5-mol H2/H2O/CH4/He/NH3 system replaces `scipy.optimize.fsolve`;
   FastChem stays as the EQ backend, paths byte-identical). Extracted
   `composition.py` for the `compo` / `compo_row` / `species` data tables
   and a `(ni, n_atoms)` JAX `compo_array`. `build_atm.py` deleted.
   Parametrized `tests/test_ini_abun.py` covers all 5 `ini_mix` modes
   plus the charge_list invariant.
6. ✅ Phase 22a: rate-parser consolidation. `rates.py` now hosts
   `apply_lowT_caps` / `apply_remove_list` / `build_rate_array`;
   `vulcan_jax.py` routes through `build_rate_array` in place of
   the legacy `rev_rate` / `remove_rate` / `lim_lowT_rates` chain.
   `var.k_arr` (dense `(nr+1, nz)`) added on `store.Variables`;
   `var.k` dict mirrored for back-compat. New
   `tests/test_read_rate.py` (6 tests) at ≤1e-13 vs legacy.
7. ✅ Phase 22b: photo cross-section preprocessing rewrite. New
   `photo_setup.py` (host-side CSV readers + NumPy `np.interp`-based
   bin / cross / branch / Rayleigh / T-dep / ion kernels +
   `populate_photo_arrays(var, atm)`). `vulcan_jax.py` routes through
   this in place of `rate.make_bins_read_cross`. Bit-exact (0.0 err)
   vs legacy on HD189 + Earth-style T-dep path. New
   `tests/test_photo_setup.py` (2 tests).
8. ✅ Phase 22c (rate-method cleanup): `legacy_io.ReadRate.{rev_rate,
   remove_rate, lim_lowT_rates}` deleted; production + test setup
   migrated to `rates.setup_var_k` / `rates.apply_photo_remove`;
   `make_bins_read_cross` callers migrated to
   `photo_setup.populate_photo_arrays`. `read_rate` (metadata) and
   `make_bins_read_cross` (test oracle) retained.
9. ✅ Phase 22d (var.k dict-surface retirement): `op_jax.compute_J` /
   `compute_Jion` now write `var.k_arr[ridx, :]` directly;
   `outer_loop._unpack_k` / `_unpack_conden_k` collapsed to a single
   full-array snapshot from `state.k_arr`; `state.pytree_from_store`
   reads `var.k_arr` directly; `state.apply_pytree_to_store` writes
   `var.k_arr`. The `.vul` writer in `legacy_io.Output.save_out`
   synthesizes the legacy `{i: array(nz)}` dict view from `var.k_arr`
   at pickle time so plot_py/ scripts continue to work. `Variables.k`
   dropped from `store.Variables.__init__`. `var.cross*` retirement
   deferred to a future phase (photo runtime kernels still consume
   dict surface at startup).
10. **Next — Phase 22e (var.cross* retirement, deferred):** refactor
    `photo.pack_photo_data` / `pack_photo_J_data` / `pack_photo_ion_data`
    to read from a dense pytree instead of `var.cross*` dicts;
    delete dict-surface writes from `photo_setup.populate_photo_arrays`;
    teach the `.vul` writer's existing synthesizer to also synthesize
    `cross*` dict views from the dense surface; then delete
    `legacy_io.ReadRate.make_bins_read_cross` and refactor
    `tests/test_photo_setup.py` to use a static reference fixture.
11. Vendor / port missing assets, tools, `diagnose.py`, and `plot_py/`.
11. Run the full oracle / config-matrix campaign.
12. Reconcile docs and freeze the support contract.

### E. Bottom line estimate

- If the target is only "the Ros2 runtime hot path is JAX", the project is
  already close.
- If the target is "all VULCAN functionality, just in JAX", the remaining work
  is still substantial, but **FastChem is not the blocker**.
- The biggest remaining buckets are:
  JAX-native atmosphere / initialization setup, JAX-native rate /
  cross-section / photo preprocessing, full repo-surface parity, and broad
  validation across the master config matrix.
- Rough total remaining effort for a complete VULCAN replacement under this
  scope: **approximately 4 to 8 weeks** of focused work, assuming FastChem
  remains an accepted external dependency matching `VULCAN-master`.

---

## How to use VULCAN-JAX

```bash
cd /Users/imalsky/Desktop/Emulators/VULCAN_Project/VULCAN-JAX

# 1. Configure: edit vulcan_cfg.py exactly as you would VULCAN-master's
#    (same format, same parameters, same input paths).
#    The default config is HD189 with SNCHO_photo_network_2025.

# 2. Run:
python vulcan_jax.py        # JAX-native chem_funs (no SymPy regen)
python vulcan_jax.py -n     # -n flag accepted as a no-op for back-compat

# 3. Output goes to output/<out_name>.vul (default: HD189.vul).
#    Same pickle format as VULCAN-master; load with the existing
#    plot_py/ scripts.

# 4. Validation: compare to VULCAN-master output
python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul
```

To enable GPU:
```bash
JAX_PLATFORM_NAME=gpu python vulcan_jax.py -n
```

To enable multi-CPU device-level parallelism (for vmap/pmap):
```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py -n
```

---

## Architecture summary

```
   ┌──────────────────────────────────────────────────────────┐
   │ vulcan_jax.py  (entry point, mirrors vulcan.py)           │
   │   ├─ JAX-native chem_funs (no make_chem_funs.py)          │
   │   ├─ build_atm (atmosphere setup) — vendored             │
   │   ├─ legacy_io.ReadRate (rate coefs) — vendored          │
   │   ├─ InitialAbun (FastChem subprocess) — vendored        │
   │   ├─ photochem setup — legacy_io.ReadRate.make_bins_read │
   │   ├─ Ros2JAX.compute_tau/flux/J — one-shot pre-loop       │
   │   └─ outer_loop.OuterLoop (JAX) — single-shot runner      │
   │       └─ jax.jit(while_loop):                             │
   │            cond_fn:  count_max | runtime | converged      │
   │            body_fn (one accept, with internal retries):   │
   │              ├─ photo branch (lax.cond on update_photo_frq)│
   │              │   compute_tau / flux / J → update k_arr    │
   │              ├─ atm refresh (lax.cond on update_frq)      │
   │              │   update_mu_dz / phi_esc → splice into atm │
   │              ├─ jax_ros2_step (chem_rhs+jac, diffusion,   │
   │              │     block_thomas) — JAX                    │
   │              ├─ clip / loss / step_ok / step_size — JAX   │
   │              ├─ conden branch (lax.cond on t / use_relax) │
   │              ├─ hydrostatic balance + ion + fix_all_bot   │
   │              ├─ ring-buffer y_time / t_time history       │
   │              ├─ adaptive rtol + photo-freq switch         │
   │              └─ in-runner conv check → longdy/longdydt    │
   └──────────────────────────────────────────────────────────┘
```

The per-step path is **fully JAX** (post-Phase 10.6). The entire
integration loop runs as one JIT'd `while_loop` on device — no
NumPy hot-path code, no Python overhead between steps. The
`OuterLoop` class is standalone (no `op.Integration` parent); only
the pre-loop atmosphere build and the post-loop `.vul` save touch
the vendored `legacy_io.ReadRate` / `legacy_io.Output` (Phase A).

---

## Numerical agreement summary

Every component validated against VULCAN-master:

| Layer | Agreement |
|---|---|
| Rate coefficients (forward, all 596 reactions) | **0** (bit-exact) |
| Reverse rates (533 from Gibbs) | **1.4e-14** |
| Atmosphere structure (pco, Tco, Kzz, M, ...) | **0** (bit-exact) |
| Initial abundances (FastChem) | **0** (bit-exact) |
| Chemistry Jacobian (per layer, vs `chem_funs.symjac`) | **4.3e-13** |
| Diffusion operator (vs `op.diffdf`) | **2e-6** (FP cancellation) |
| Block-Thomas solver | **3e-15** |
| Single Ros2 step (vs `op.Ros2.solver`) | **1.16e-15** |
| compute_tau / compute_flux / compute_J (vs `op.*`) | **8e-16 / 3.7e-11 / 1.8e-11** |
| `chem_funs` JAX-native (vs SymPy ref) | **5e-14** (Gibbs, gibbs_sp, chemdf) |
| End-to-end 50-step run (HD189, Phase 9 wired path) | **1.59e-10** |

VULCAN-JAX is **numerically equivalent** to VULCAN-master at every level we can measure.
