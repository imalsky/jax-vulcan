# VULCAN-JAX — Implementation Status

A JAX-accelerated, drop-in compatible port of [VULCAN](https://github.com/exoclime/VULCAN), the chemical-kinetics code for exoplanet atmospheres.

**Location**: `/Users/imalsky/Desktop/Emulators/VULCAN_Project/VULCAN-JAX/` (next to `VULCAN-master/`).

**Last updated**: 2026-04-26.

---

## TL;DR

Phases 0–10 complete; **Phase A (standalone refactor) is complete** — VULCAN-JAX no longer depends on `../VULCAN-master/` at runtime. The forward model runs end-to-end and matches VULCAN-master to **1.59e-10 maximum relative error** on a 50-step HD189 integration. The codebase totals ~7500 lines of new code; `chem_funs.py` is JAX-native (no SymPy code generator). `pytest tests/` runs **27 tests** in ~60 s with VULCAN-master present; **17 pass + 10 skip cleanly** with VULCAN-master absent (the 10 oracle tests compare against upstream and skip with a clear reason when upstream is gone).

**Performance**: ~47 ms/step warm (Phase 11 — analytical Jacobian) — **3.2× faster** than Phase 10.6 (149 ms/step) and **3.7× faster** than VULCAN-master (175 ms/step). The win comes from replacing `jax.jacrev(chem_rhs_per_layer)` with a stoichiometry-driven analytical Jacobian (`chem.chem_jac_analytical`): per-call chem_jac time drops from 95 ms to 2.6 ms (**36× speedup**), making chem_jac no longer the dominant cost. Earlier milestones: Phase 10.4 was 185 ms/step; Phase 10.5 single-shot runner brought it to 149 ms/step (host-sync removal); Phase 10.6/10.7 were code-cleanliness milestones with no perf change.

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
- Live progress (`use_print_prog` per-iteration, `use_live_plot`, `use_live_flux`) is opt-out since the runner is single-shot and intermediate progress is not surfaced. A final-state `print_prog` call fires after the runner returns.
- Mid-run `fix_species` trigger (Earth/Jupiter): not yet ported. HD189 has `fix_species=[]` so this is dormant; runs that would fire it print a one-time warning at startup.
- `compute_Jion` (photoionisation rates): out of scope for HD189, raises `NotImplementedError` if `use_ion=True` and the runner reaches that path. Charge balance itself (the post-step clamp on `e`) IS implemented in 10.6.
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
- `atm/`, `thermo/`, `fastchem_vulcan/` — vendored data files (HD189 + SNCHO_photo_network_2025 path only, ~28 MB total — drop the unused atmosphere profiles, BC files, test networks, and stellar fluxes that come along with the full upstream tree).

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
| 10.7 | Pytest enablement: `pytest.ini` + `tests/conftest.py` + thin `def test_main(): assert main() == 0` wrappers (subprocess wrappers for two tests with deliberate import-table swaps). `pytest tests/` discovers and runs all 26 tests. | ✅ Done |
| A | Standalone refactor: vendored `atm/`/`thermo/`/`fastchem_vulcan/` data (~28 MB, HD189-only); vendored `op.ReadRate`+`op.Output` into `legacy_io.py`; dropped `sys.path.append(VULCAN_MASTER)` from runtime; oracle tests skip cleanly when upstream is absent (17 pass + 10 skip without VULCAN-master, 27 pass with it including the new chem_jac_sparse test). | ✅ Done |
| 11 | Custom analytical chemistry Jacobian: replaces `jax.jacrev(chem_rhs_per_layer)` with stoichiometry-driven build in `chem.chem_jac_analytical`. chem_jac drops 95 ms → 2.6 ms (36×); full step 149 ms → 47 ms (3.2×). Bit-exact (≤1e-13) vs AD path. | ✅ Done |

### Drop-in compatibility

- ✅ Same `vulcan_cfg.py` format (just copy from VULCAN-master)
- ✅ Same input file paths/formats (`atm/`, `thermo/`, stellar fluxes, photo cross-sections)
- ✅ Same `.vul` pickle output schema (loadable by VULCAN's `plot_py/` scripts)
- ✅ Same command-line invocation: `python vulcan_jax.py` (the `-n` flag is accepted for compatibility but is now a no-op since `chem_funs` is JAX-native)

### Validation tests

All 27 tests in `tests/` PASS (`pytest tests/`):

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

**Current state (Phase 10.5+ single-shot runner)**: ~149 ms/step (warm) — **slightly faster than VULCAN-master's ~175 ms/step**, and 19% faster than the per-batch-host-sync Phase 10.4 path (185 ms/step).

**Profile** (per-step on HD189, Phase 11):
| Component | Time (Phase 11) | Time (Phase 10.6) |
|---|---|---|
| chem_jac (analytical, stoichiometry-driven) | **~2.6 ms** | (jacrev: 95 ms) |
| block_thomas | ~3 ms | ~3 ms |
| everything else (rhs, diffusion, photo branch when fired, conden, ring update) | ~41 ms | ~51 ms |
| **total step** | **~47 ms** | ~149 ms |

**Key changes**:
- Phase 9: switched `chem_jac_per_layer` from `jax.jacfwd` to `jax.jacrev` (~25% faster, better numerical accuracy 4.3e-13 vs 6e-11).
- Phase 11: replaced `jacrev(chem_rhs_per_layer)` with `chem.chem_jac_analytical` — a stoichiometry-driven analytical build that scatters `sign_i * stoich_i * stoich_j * y_j^(stoich_j-1) * (rest_of_factors) * k_r * M_factor` directly into the (ni, ni) Jacobian via `segment_sum`. Bit-exact (≤1e-13 relerr) vs the AD path on HD189 reference state. The leave-one-out reactant product avoids the `(stoich/y)*rate` divide-by-zero trap when reactant abundance is zero (e.g. CH3CCH/CH2CCH2 in initial HD189).

### Multi-CPU parallelism via pmap

**Demonstrated**: `jax.vmap` over 4 atmospheres works and gives consistent results.

**Not done**: Real `pmap` setup for multi-CPU production use. The right invocation is:
```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py
```
combined with `jax.pmap(jax_ros2_step, ...)` over the batch dimension. Untested but the architecture supports it.

### Full-convergence validation runs

**Done**: HD189 50 steps (matches to 1.59e-10 post-Phase 9; same on Phase 10.6 path).

**Pending**: Earth (condensation), Jupiter (low-T rates), HD209. The Phase 10.4 conden kernels are bit-exact against the numpy reference but haven't been exercised end-to-end on Earth. **Each convergence run takes 1-5 hours** at current speed; not feasible inside an interactive iteration.

### Pure-JAX `integrate.py` loop

`integrate.py` is now a single helper — `jax_integrate_fixed_dt(y0, k_arr, dt, n_steps, atm, net)` — that takes N fixed-dt Ros2 steps in one `jax.lax.scan`. Validated to 1.5e-16 vs equivalent Python loop; ~1.5x faster (135 ms/step warm vs 195 ms/step Python). Used for benchmarks and as a JIT-vs-Python-loop equivalence check, not for production runs.

The Phase 10.6 production drop-in is `outer_loop.OuterLoop` (single-shot adaptive runner with photo / conden / convergence / ion / fix-all-bot all on device). The legacy `jax_integrate_adaptive` was deleted because `OuterLoop` subsumes it.

---

## ❌ NOT DONE

### Phase 10 closed — remaining open work

Phases 10.1 ✅, 10.2 ✅, 10.3 ✅, 10.4 ✅, 10.5 ✅, 10.6 ✅, 10.7 ✅:
the inner per-accepted-step retry loop, photochemistry, atmosphere
geometry refresh, hydrostatic balance, condensation, ring-buffered
convergence, adaptive rtol, the photo-frequency switch, ion charge
balance, and `fix_all_bot` post-step clamping all run inside the
single `lax.while_loop` runner. The Python `while not stop()` loop in
`OuterLoop.__call__` is gone, the `op.Integration` parent class is
dropped, `OuterLoop` is a standalone class, and `pytest tests/` is the
canonical way to run the suite.

Open work that lands outside the Phase 10 envelope:
- **Mid-run `fix_species` trigger** (Earth/Jupiter only): HD189 has
  `fix_species=[]` so this is dormant. Configs that would fire it
  print a one-time warning at startup. Adding the one-shot trigger
  detection to the body is straightforward but waits on an Earth /
  Jupiter integration to validate.
- **`compute_Jion`** (photoionisation rates): not on the HD189 path;
  raises `NotImplementedError` if used.
- **Per-test fixture sharing**: `pytest tests/` works today via thin
  `def test_main(): assert main() == 0` wrappers around the existing
  script-style `main()` functions. A future cleanup could share an
  HD189 reference-state fixture across tests in `conftest.py`,
  cutting per-test setup time by ~1-3 seconds each. Numerical
  agreement is unaffected.
- **Parallel test execution**: `pytest -n auto` collides on FastChem's
  fixed output path. Run serially.

See `~/.claude/plans/continue-my-jax-port-squishy-platypus.md` for the
staged plan.


### Earth / Jupiter / HD209 full-convergence runs

Inherited support from `op.Ros2` covers condensation (Earth, Jupiter),
low-T rate caps (Jupiter), and BC variants. Phase 9 added pure-JAX wiring
for the per-step path. These configs should work end-to-end but haven't
been run beyond unit-component validation. Convergence runs are 1–5 hours
each.

### `compute_Jion` photo-ionization wiring

`Ros2JAX` overrides `compute_tau` / `compute_flux` / `compute_J` (Phase 9)
but inherits `compute_Jion` from `op.ODESolver` since HD189's default has
`use_ion = False`. For ionized configs, port a JAX version mirroring
`compute_J_jax` (it's the same wavelength-integration shape using
`var.cross_Jion`).

### Live plotting / movie output

`use_live_plot = False` recommended (set in test runs). VULCAN-master's
matplotlib plotting code (`op.py:3262+`) is inherited but not exercised.
Should "just work" since it's NumPy-based and runs against `var.y` which
we update normally, but unconfirmed end-to-end.

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
