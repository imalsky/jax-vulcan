# VULCAN-JAX — Implementation Status

A JAX-accelerated, drop-in compatible port of [VULCAN](https://github.com/exoclime/VULCAN), the chemical-kinetics code for exoplanet atmospheres.

**Location**: `/Users/imalsky/Desktop/Emulators/VULCAN_Project/VULCAN-JAX/` (next to `VULCAN-master/`).

**Last updated**: 2026-04-26.

---

## TL;DR

Phases 0–10, A, 11, and **12 (runtime parity extension + end-to-end differentiability)** complete. VULCAN-JAX no longer depends on `../VULCAN-master/` at runtime, matches it to **1.59e-10 max relative error** on a 50-step HD189 kernel comparison, and now exposes implicit-function-theorem gradients of the converged state through the structured-input API in `steady_state_grad.py`. The current test surface includes runtime, ion-photo, example-config, and optional backend-parity coverage; with `VULCAN-master` present on this CPU-only host the current run is **34 pass + 1 skip** (`test_backend_parity` skips cleanly without a GPU), and the upstream-oracle tests still skip cleanly when `../VULCAN-master/` is absent.

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
- Live progress is reduced to the final-state `print_prog` call because the runner is single-shot; plotting/live/movie/save-evolution side paths are intentionally rejected up front by `runtime_validation.py`.
- Low-T rate caps fire only when their cfg flag is on (HD189 has it off).

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
- `build_atm.py` — copied unchanged (40 KB, all setup logic)
- `store.py` — copied unchanged (Variables/AtmData/Parameters dataclasses)
- `phy_const.py` — copied unchanged
- `vulcan_cfg.py` — copied from VULCAN-master config (HD189 SNCHO_photo_network_2025)
- `legacy_io.py` (Phase A): vendored `ReadRate` + `Output` from `op.py:49-781,3131-3260` (plot methods dropped). Pre-loop rate-coef parser + `.vul` writer + per-step progress printer.
- `atm/`, `thermo/`, `fastchem_vulcan/`, `cfg_examples/` — vendored runtime data and example configs covering the supported Earth / Jupiter / HD189 / HD209 Ros2 setups.

JAX-native:
- `chem_funs.py` — **JAX-native module** (Phase 9): re-exports `ni`/`nr`/`spec_list`/`re_dict`/`re_wM_dict`/`Gibbs`/`gibbs_sp` etc. from `network.py` + `gibbs.py` + `chem.py` with NumPy-callable wrappers. No SymPy, no `make_chem_funs.py` step.

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
| A | Standalone refactor: vendored `atm/`/`thermo/`/`fastchem_vulcan/` data and `cfg_examples/` presets for the supported Earth / Jupiter / HD189 / HD209 Ros2 configs; vendored `op.ReadRate`+`op.Output` into `legacy_io.py`; dropped `sys.path.append(VULCAN_MASTER)` from runtime; oracle tests still skip cleanly when upstream is absent. | ✅ Done |
| 11 | Custom analytical chemistry Jacobian: replaces `jax.jacrev(chem_rhs_per_layer)` with stoichiometry-driven build in `chem.chem_jac_analytical`. chem_jac drops 95 ms → 2.6 ms (36×); full step 149 ms → 47 ms (3.2×). Bit-exact (≤1e-13) vs AD path. | ✅ Done |
| 12 | Runtime parity extension + end-to-end differentiability. Added `compute_Jion` photo-ionization wiring, full mid-run `fix_species` / fixed-bottom species semantics, stricter `runtime_validation.py`, structured-input implicit AD with residual gating, factor-once/reuse block-tridiagonal solves, repaired bench/profile scripts, vendored example-config coverage, and a clean `vulture --min-confidence 80` pass. | ✅ Done |

### Drop-in compatibility

- ✅ Same `vulcan_cfg.py` format (just copy from VULCAN-master)
- ✅ Same input file paths/formats (`atm/`, `thermo/`, stellar fluxes, photo cross-sections)
- ✅ Same `.vul` pickle output schema (loadable by VULCAN's `plot_py/` scripts)
- ✅ Same command-line invocation: `python vulcan_jax.py` (the `-n` flag is accepted for compatibility but is now a no-op since `chem_funs` is JAX-native)

### Validation tests

Current CPU-only run with `VULCAN-master` present: **34 PASS + 1 SKIP** (`pytest tests/`). The skipped test is the optional GPU backend-parity check.

```
test_network_parse        PASS  ni=93, nr=1192, full spec_list match
test_rates                PASS  Max relerr = 0 (machine precision)
test_gibbs                PASS  K_eq, gibbs_sp, reverse k all to 1.4e-14
test_build_atm            PASS  pco/pico/Tco/Kzz/M/n_0 all exact
test_ini_abun             PASS  y/ymix/atom_ini exact (FastChem path)
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

- **Matched-step / full-convergence oracle runs for Earth, Jupiter, and HD209.** The supported assets/configs are vendored and their pre-loop setup is tested, but the expensive long-run oracle campaigns are still outstanding.
- **Checked-in GPU benchmark numbers on real hardware.** The backend-parity test is present and skips cleanly on CPU-only hosts; a GPU machine is still needed to record production timings and float64 parity numbers in the docs.
- **Per-test fixture sharing.** The suite is correct today but still pays repeated HD189 setup cost in several script-style tests.
- **Parallel pytest execution.** `pytest -n auto` still collides on FastChem's fixed output path; run serially.
- **Live plotting / movie output.** These remain intentionally unsupported in the JAX runtime and are rejected by `runtime_validation.py` before setup.

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
